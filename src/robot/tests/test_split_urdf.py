"""Risk-focused tests for the reviewed URDF split implementation."""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from robot.split_urdf import SplitError, split_urdf, validate_urdf

_JOINTS = """
  <joint name="right_slide" type="prismatic">
    <parent link="right_2"/><child link="right_3"/>
    <origin xyz="0 0.25 0.2" rpy="0.3 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0.4" effort="1" velocity="1"/>
  </joint>
  <joint name="left_shoulder" type="revolute">
    <parent link="base"/><child link="left_1"/>
    <origin xyz="0.1 0.2 0.3" rpy="0.2 -0.1 0.4"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="right_shoulder" type="revolute">
    <parent link="base"/><child link="right_1"/>
    <origin xyz="0.1 -0.2 0.3" rpy="-0.2 0.1 -0.4"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="left_slide" type="prismatic">
    <parent link="left_2"/><child link="left_3"/>
    <origin xyz="0 0.25 0.2" rpy="0.3 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0.4" effort="1" velocity="1"/>
  </joint>
  <joint name="right_elbow" type="continuous">
    <parent link="right_1"/><child link="right_2"/>
    <origin xyz="0.4 0 0.1" rpy="0 0 0.3"/><axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>
  <joint name="left_elbow" type="continuous">
    <parent link="left_1"/><child link="left_2"/>
    <origin xyz="0.4 0 0.1" rpy="0 0 0.3"/><axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>
"""


def _write_source_urdf(
    directory: Path,
    *,
    visual_filename: str = "package://assembly/meshes/visual.stl",
    collision_filename: str = "meshes/collision.stl",
    include_visual: bool = True,
    include_collision: bool = True,
) -> tuple[Path, Path, Path]:
    """Create a small bimanual URDF and the mesh files it references."""

    mesh_directory = directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    visual_mesh = mesh_directory / "visual.stl"
    collision_mesh = mesh_directory / "collision.stl"
    visual_mesh.write_text("solid visual\nendsolid visual\n", encoding="utf-8")
    collision_mesh.write_text("solid collision\nendsolid collision\n", encoding="utf-8")

    visual_geometry = (
        f'<visual><geometry><mesh filename="{visual_filename}"/></geometry></visual>'
        if include_visual
        else ""
    )
    collision_geometry = (
        f'<collision><geometry><mesh filename="{collision_filename}"/>'
        "</geometry></collision>"
        if include_collision
        else ""
    )
    links = "".join(
        f'<link name="{name}">'
        + (visual_geometry + collision_geometry if name == "base" else "")
        + "</link>"
        for name in (
            "base",
            "left_1",
            "left_2",
            "left_3",
            "right_1",
            "right_2",
            "right_3",
        )
    )
    source_path = directory / "source.urdf"
    source_path.write_text(
        f'<robot name="test_robot">{links}{_JOINTS}</robot>', encoding="utf-8"
    )
    return source_path, visual_mesh, collision_mesh


def _origin_transform(origin: ET.Element) -> list[list[float]]:
    """Build a homogeneous transform from a URDF origin using test-local math."""

    xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
    roll, pitch, yaw = [float(value) for value in origin.get("rpy", "0 0 0").split()]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return [
        [rotation[row][0], rotation[row][1], rotation[row][2], xyz[row]]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _multiply_transforms(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    """Multiply homogeneous transforms with independent test-local math."""

    return [
        [sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4)]
        for row in range(4)
    ]


def test_split_resolves_package_and_relative_meshes_without_changing_source(
    tmp_path: Path,
) -> None:
    """Rewrite valid mesh paths in both outputs while preserving the source."""

    source_path, visual_mesh, collision_mesh = _write_source_urdf(tmp_path)
    original_source = source_path.read_bytes()

    result = split_urdf(
        source_path,
        left_first_joint="left_shoulder",
        right_first_joint="right_shoulder",
    )

    assert source_path.read_bytes() == original_source
    source_visual = result.source.tree.getroot().find(
        "./link[@name='base']/visual/geometry/mesh"
    )
    assert source_visual is not None
    assert source_visual.get("filename") == "package://assembly/meshes/visual.stl"

    for generated_arm in (result.left, result.right):
        generated_meshes = generated_arm.tree.getroot().findall(
            "./link[@name='base']/visual/geometry/mesh"
        ) + generated_arm.tree.getroot().findall(
            "./link[@name='base']/collision/geometry/mesh"
        )
        assert {mesh.get("filename") for mesh in generated_meshes} == {
            str(visual_mesh.resolve()),
            str(collision_mesh.resolve()),
        }


def test_explicit_package_root_outside_source_directory_is_used(
    tmp_path: Path,
) -> None:
    """Resolve a package URI from its explicit external package-root mapping."""

    source_directory = tmp_path / "urdf"
    source_path, _, collision_mesh = _write_source_urdf(source_directory)
    package_root = tmp_path / "installed" / "assembly"
    package_mesh = package_root / "meshes" / "visual.stl"
    package_mesh.parent.mkdir(parents=True)
    package_mesh.write_text("solid installed\nendsolid installed\n", encoding="utf-8")

    result = split_urdf(
        source_path,
        left_first_joint="left_shoulder",
        right_first_joint="right_shoulder",
        package_roots={"assembly": package_root},
    )

    assert result.source.package_roots == (("assembly", package_root.resolve()),)
    for generated_arm in (result.left, result.right):
        generated_meshes = generated_arm.tree.getroot().findall(
            "./link[@name='base']/visual/geometry/mesh"
        ) + generated_arm.tree.getroot().findall(
            "./link[@name='base']/collision/geometry/mesh"
        )
        assert {mesh.get("filename") for mesh in generated_meshes} == {
            str(package_mesh.resolve()),
            str(collision_mesh.resolve()),
        }


@pytest.mark.parametrize(
    ("visual_filename", "expected_diagnostic"),
    [
        ("package://assembly/meshes/absent.stl", "no supported candidate"),
        ("package://assembly/meshes/visual.stl", "ambiguous"),
    ],
)
def test_inferred_package_root_errors_explain_missing_or_ambiguous_candidates(
    tmp_path: Path,
    visual_filename: str,
    expected_diagnostic: str,
) -> None:
    """Explain when adjacent package roots cannot be inferred unambiguously."""

    source_directory = tmp_path / "urdf"
    source_path, _, _ = _write_source_urdf(
        source_directory,
        visual_filename=visual_filename,
    )
    if expected_diagnostic == "ambiguous":
        other_candidate_mesh = tmp_path / "meshes" / "visual.stl"
        other_candidate_mesh.parent.mkdir(parents=True)
        other_candidate_mesh.write_text(
            "solid second candidate\nendsolid second candidate\n", encoding="utf-8"
        )

    with pytest.raises(SplitError) as error:
        validate_urdf(source_path)

    message = str(error.value)
    assert expected_diagnostic in message
    assert "assembly" in message
    assert "--package-root assembly=PATH" in message


def test_split_rejects_unsupported_mesh_uris_with_xml_context(tmp_path: Path) -> None:
    """Identify the owning mesh element when its URI scheme is unsupported."""

    source_path, _, _ = _write_source_urdf(
        tmp_path,
        visual_filename="https://assets.example/visual.stl",
    )

    with pytest.raises(SplitError) as error:
        validate_urdf(source_path)

    message = str(error.value)
    assert "link 'base' <visual>[1] <mesh>" in message
    assert "unsupported mesh URI" in message


@pytest.mark.parametrize(
    ("include_visual", "include_collision", "missing_kind"),
    [(False, True, "visual"), (True, False, "collision")],
)
def test_mesh_validation_requires_global_visual_and_collision_meshes(
    tmp_path: Path,
    include_visual: bool,
    include_collision: bool,
    missing_kind: str,
) -> None:
    """Require at least one mesh-backed geometry of each required kind."""

    source_path, _, _ = _write_source_urdf(
        tmp_path,
        include_visual=include_visual,
        include_collision=include_collision,
    )

    with pytest.raises(SplitError) as error:
        validate_urdf(source_path)

    assert f"no mesh-backed <{missing_kind}> geometry" in str(error.value)


@pytest.mark.parametrize("invalid_topology", ["disconnected", "multiple_parents"])
def test_validation_rejects_disconnected_or_multi_parent_topology(
    tmp_path: Path,
    invalid_topology: str,
) -> None:
    """Reject source graphs that cannot represent a single rooted tree."""

    source_path, _, _ = _write_source_urdf(tmp_path)
    source = source_path.read_text(encoding="utf-8")
    if invalid_topology == "disconnected":
        source = source.replace("</robot>", '<link name="orphan"/></robot>')
        diagnostic = "one connected tree"
    else:
        extra_joint = (
            '<joint name="second_parent" type="fixed">'
            '<parent link="base"/><child link="left_1"/></joint>'
        )
        source = source.replace("</robot>", f"{extra_joint}</robot>")
        diagnostic = "more than one parent joint"
    source_path.write_text(source, encoding="utf-8")

    with pytest.raises(SplitError, match=diagnostic):
        validate_urdf(source_path)


def test_split_rejects_overlapping_arm_subtrees(tmp_path: Path) -> None:
    """Require left and right first joints to root independent branches."""

    source_path, _, _ = _write_source_urdf(tmp_path)

    with pytest.raises(SplitError) as error:
        split_urdf(
            source_path,
            left_first_joint="left_shoulder",
            right_first_joint="left_elbow",
        )

    message = str(error.value)
    assert "overlapping joints" in message
    assert "left_elbow" in message
    assert "left_slide" in message


@pytest.mark.parametrize(
    ("joint_name", "lower_limit"),
    [("right_shoulder", "0.1"), ("right_slide", "0.1")],
)
def test_split_rejects_inactive_q_zero_outside_joint_limits(
    tmp_path: Path,
    joint_name: str,
    lower_limit: str,
) -> None:
    """Report the joint and limits when freezing it would violate its bounds."""

    source_path, _, _ = _write_source_urdf(tmp_path)
    tree = ET.parse(source_path)
    joint = tree.getroot().find(f"./joint[@name='{joint_name}']")
    assert joint is not None
    limit = joint.find("limit")
    assert limit is not None
    limit.set("lower", lower_limit)
    tree.write(source_path, encoding="utf-8")

    with pytest.raises(SplitError) as error:
        split_urdf(
            source_path,
            left_first_joint="left_shoulder",
            right_first_joint="right_shoulder",
        )

    message = str(error.value)
    assert joint_name in message
    assert "limit" in message.lower()
    assert lower_limit in message


def test_split_preserves_active_order_and_freezes_inactive_joints_at_zero(
    tmp_path: Path,
) -> None:
    """Keep path order and preserve each inactive joint's exact q=0 origin."""

    source_path, _, _ = _write_source_urdf(tmp_path)
    result = split_urdf(
        source_path,
        left_first_joint="left_shoulder",
        right_first_joint="right_shoulder",
    )

    assert result.left.selection.active_joints == (
        "left_shoulder",
        "left_elbow",
        "left_slide",
    )
    assert result.right.selection.active_joints == (
        "right_shoulder",
        "right_elbow",
        "right_slide",
    )
    active_joint_types = (
        (
            result.left,
            {
                "left_shoulder": "revolute",
                "left_elbow": "continuous",
                "left_slide": "prismatic",
            },
        ),
        (
            result.right,
            {
                "right_shoulder": "revolute",
                "right_elbow": "continuous",
                "right_slide": "prismatic",
            },
        ),
    )
    for generated_arm, expected_active_types in active_joint_types:
        for joint_name, expected_type in expected_active_types.items():
            joint = generated_arm.tree.getroot().find(f"./joint[@name='{joint_name}']")
            assert joint is not None
            assert joint.get("type") == expected_type

    source_robot = result.source.tree.getroot()
    for generated_arm, inactive_names in (
        (result.left, ("right_shoulder", "right_elbow", "right_slide")),
        (result.right, ("left_shoulder", "left_elbow", "left_slide")),
    ):
        generated_robot = generated_arm.tree.getroot()
        for joint_name in inactive_names:
            source_joint = source_robot.find(f"./joint[@name='{joint_name}']")
            generated_joint = generated_robot.find(f"./joint[@name='{joint_name}']")
            assert source_joint is not None
            assert generated_joint is not None
            assert generated_joint.get("type") == "fixed"
            source_origin = source_joint.find("origin")
            generated_origin = generated_joint.find("origin")
            assert source_origin is not None
            assert generated_origin is not None
            assert generated_origin.attrib == source_origin.attrib
            assert generated_joint.find("axis") is None
            assert generated_joint.find("limit") is None
            assert generated_joint.find("dynamics") is None


def test_split_root_places_a_nontrivial_actuator_at_zero_with_axis_minus_z(
    tmp_path: Path,
) -> None:
    """Verify root-frame position and axis using only output XML and local math."""

    source_path, _, _ = _write_source_urdf(tmp_path)
    source_tree = ET.parse(source_path)
    actuator = source_tree.getroot().find("./joint[@name='left_shoulder']")
    assert actuator is not None
    axis = actuator.find("axis")
    assert axis is not None
    axis.set("xyz", "0.3 -0.4 0.5")
    source_tree.write(source_path, encoding="utf-8")

    result = split_urdf(
        source_path,
        left_first_joint="left_shoulder",
        right_first_joint="right_shoulder",
    )
    generated_robot = result.left.tree.getroot()
    actuator = generated_robot.find("./joint[@name='left_shoulder']")
    assert actuator is not None
    source_root_joint = next(
        joint
        for joint in generated_robot.findall("joint")
        if joint.find("child") is not None and joint.find("child").get("link") == "base"
    )
    root_origin = source_root_joint.find("origin")
    actuator_origin = actuator.find("origin")
    assert root_origin is not None
    assert actuator_origin is not None
    actuator_in_split_root = _multiply_transforms(
        _origin_transform(root_origin),
        _origin_transform(actuator_origin),
    )

    assert [actuator_in_split_root[row][3] for row in range(3)] == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-9
    )

    actuator_axis = actuator.find("axis")
    assert actuator_axis is not None
    axis_values = [float(value) for value in actuator_axis.get("xyz", "1 0 0").split()]
    axis_norm = math.sqrt(sum(value * value for value in axis_values))
    normalized_axis = [value / axis_norm for value in axis_values]
    axis_in_split_root = [
        sum(
            actuator_in_split_root[row][column] * normalized_axis[column]
            for column in range(3)
        )
        for row in range(3)
    ]
    assert axis_in_split_root == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
