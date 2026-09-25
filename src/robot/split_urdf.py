#!/usr/bin/env python3
"""Validate and split a bimanual URDF into one model per arm.

The operator identifies the first actuator joint for each arm and, when a
subtree branches, the final joint on each selected arm path. Each generated
URDF retains the complete source robot, keeps the selected arm's supported
one-degree-of-freedom joints movable, and fixes every other movable joint at
the URDF zero configuration.

Generated models are rooted at the selected first actuator. Relative and
``package://`` mesh references are resolved against the source URDF and are
rewritten as absolute paths only in generated trees.
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias
from xml.etree import ElementTree as ET

Matrix4: TypeAlias = list[list[float]]

_ACTIVE_JOINT_TYPES = {"continuous", "prismatic", "revolute"}
_KNOWN_JOINT_TYPES = _ACTIVE_JOINT_TYPES | {"fixed", "floating", "planar"}
_UNSUPPORTED_JOINT_TYPES = {"floating", "planar"}
_FIXED_JOINT_FIELDS = {
    "axis",
    "calibration",
    "dynamics",
    "limit",
    "mimic",
    "safety_controller",
}


class SplitError(ValueError):
    """Indicate that a source URDF or arm selection cannot be split safely."""


@dataclass(frozen=True)
class UrdfGraph:
    """Store the validated link/joint topology needed by the splitter."""

    root_link: str
    links: dict[str, ET.Element]
    joints: dict[str, ET.Element]
    outgoing: dict[str, list[ET.Element]]
    incoming: dict[str, ET.Element]


@dataclass(frozen=True)
class MeshReference:
    """Describe one validated mesh reference and its resolved filesystem path."""

    link_name: str
    geometry_kind: str
    geometry_index: int
    filename: str
    resolved_path: Path

    @property
    def context(self) -> str:
        """Return a human-readable XML location for diagnostics."""

        return (
            f"link {self.link_name!r} <{self.geometry_kind}>"
            f"[{self.geometry_index}] <mesh>"
        )


@dataclass(frozen=True)
class ValidationReport:
    """Contain a parsed source URDF and all successful validation results."""

    source_path: Path
    tree: ET.ElementTree
    graph: UrdfGraph
    mesh_references: tuple[MeshReference, ...]
    package_roots: tuple[tuple[str, Path], ...]


@dataclass(frozen=True)
class ArmSelection:
    """Describe the resolved kinematic path for one generated arm model."""

    label: str
    first_joint: str
    path: tuple[str, ...]
    active_joints: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedArm:
    """Contain one generated arm tree and its deliberate topology changes."""

    selection: ArmSelection
    tree: ET.ElementTree
    frozen_joints: tuple[str, ...]
    removed_transmissions: tuple[str, ...]


@dataclass(frozen=True)
class SplitResult:
    """Contain validated left and right outputs generated from one source URDF."""

    source: ValidationReport
    left: GeneratedArm
    right: GeneratedArm


def _joint_parent(joint: ET.Element) -> str:
    """Return a joint's parent link name after validating the element."""

    parent = joint.find("parent")
    if parent is None or not parent.get("link"):
        raise SplitError(f"Joint {joint.get('name')!r} has no valid <parent link=...>.")
    return parent.get("link", "")


def _joint_child(joint: ET.Element) -> str:
    """Return a joint's child link name after validating the element."""

    child = joint.find("child")
    if child is None or not child.get("link"):
        raise SplitError(f"Joint {joint.get('name')!r} has no valid <child link=...>.")
    return child.get("link", "")


def _build_graph(robot: ET.Element) -> UrdfGraph:
    """Build and validate a connected, acyclic URDF link/joint tree."""

    links: dict[str, ET.Element] = {}
    for link in robot.findall("link"):
        name = link.get("name")
        if not name:
            raise SplitError("Every <link> must have a name.")
        if name in links:
            raise SplitError(f"Duplicate link name: {name!r}.")
        links[name] = link
    if not links:
        raise SplitError("The URDF must contain at least one <link>.")

    joints: dict[str, ET.Element] = {}
    outgoing: dict[str, list[ET.Element]] = defaultdict(list)
    incoming: dict[str, ET.Element] = {}
    for joint in robot.findall("joint"):
        name = joint.get("name")
        if not name:
            raise SplitError("Every <joint> must have a name.")
        if name in joints:
            raise SplitError(f"Duplicate joint name: {name!r}.")
        joint_type = joint.get("type")
        if joint_type not in _KNOWN_JOINT_TYPES:
            raise SplitError(
                f"Joint {name!r} has unknown or missing type {joint_type!r}."
            )
        if joint_type in _UNSUPPORTED_JOINT_TYPES:
            raise SplitError(
                f"Joint {name!r} uses unsupported {joint_type!r} motion; "
                "only fixed and one-degree-of-freedom joints are supported."
            )

        parent = _joint_parent(joint)
        child = _joint_child(joint)
        if parent not in links:
            raise SplitError(
                f"Joint {name!r} references missing parent link {parent!r}."
            )
        if child not in links:
            raise SplitError(f"Joint {name!r} references missing child link {child!r}.")
        if child in incoming:
            other = incoming[child].get("name")
            raise SplitError(
                f"Link {child!r} has more than one parent joint: {other!r} and {name!r}."
            )
        joints[name] = joint
        outgoing[parent].append(joint)
        incoming[child] = joint

    roots = [name for name in links if name not in incoming]
    if len(roots) != 1:
        raise SplitError(
            "The URDF must contain one connected tree; "
            f"found {len(roots)} candidate root links: {roots}."
        )

    visited_links: set[str] = set()
    stack = [roots[0]]
    while stack:
        link_name = stack.pop()
        if link_name in visited_links:
            raise SplitError(f"The URDF joint graph contains a cycle at {link_name!r}.")
        visited_links.add(link_name)
        stack.extend(_joint_child(joint) for joint in outgoing.get(link_name, ()))
    if visited_links != set(links):
        missing = sorted(set(links) - visited_links)
        raise SplitError(f"Links are disconnected from root {roots[0]!r}: {missing}.")

    return UrdfGraph(
        root_link=roots[0],
        links=links,
        joints=joints,
        outgoing=dict(outgoing),
        incoming=incoming,
    )


def _iter_mesh_elements(
    robot: ET.Element,
) -> Iterator[tuple[str, str, int, ET.Element]]:
    """Yield mesh elements with their owning link and geometry context."""

    for link in robot.findall("link"):
        link_name = link.get("name", "<unnamed>")
        for geometry_kind in ("visual", "collision"):
            for index, geometry_owner in enumerate(
                link.findall(geometry_kind), start=1
            ):
                for mesh in geometry_owner.findall("./geometry/mesh"):
                    yield link_name, geometry_kind, index, mesh


def _parse_package_uri(filename: str) -> tuple[str, Path] | None:
    """Parse a package URI into its package name and relative mesh path."""

    if not filename.startswith("package://"):
        return None
    package_reference = filename.removeprefix("package://")
    package_name, separator, package_relative = package_reference.partition("/")
    if not separator or not package_name or not package_relative:
        raise SplitError(f"malformed package URI {filename!r}")
    relative_path = Path(package_relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SplitError(f"package URI escapes its package root: {filename!r}")
    return package_name, relative_path


def _resolve_package_roots(
    robot: ET.Element,
    source_path: Path,
    configured_roots: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    """Bind every referenced package name to one unambiguous package root.

    Explicit mappings take precedence. Otherwise, the source directory and its
    parent are considered supported layouts. An inferred root is accepted only
    when exactly one candidate contains every referenced path for that package.

    Args:
        robot: Parsed URDF root element.
        source_path: Absolute source URDF path.
        configured_roots: Optional package-name to package-root mapping.

    Returns:
        Resolved package roots keyed by package name.

    Raises:
        SplitError: If a mapping is invalid or an inferred binding is missing
            or ambiguous.
    """

    explicit: dict[str, Path] = {}
    for raw_name, raw_root in (configured_roots or {}).items():
        package_name = raw_name.strip()
        if not package_name or "/" in package_name:
            raise SplitError(
                f"Invalid package name in package-root mapping: {raw_name!r}."
            )
        package_root = Path(raw_root).expanduser().resolve()
        if not package_root.is_dir():
            raise SplitError(
                f"Package root for {package_name!r} is not a directory: {package_root}"
            )
        explicit[package_name] = package_root

    references: dict[str, set[Path]] = defaultdict(set)
    malformed: list[str] = []
    for link_name, geometry_kind, index, mesh in _iter_mesh_elements(robot):
        filename = mesh.get("filename", "").strip()
        if not filename.startswith("package://"):
            continue
        try:
            parsed = _parse_package_uri(filename)
        except SplitError as exc:
            context = f"link {link_name!r} <{geometry_kind}>[{index}] <mesh>"
            malformed.append(f"{context}: {exc}")
            continue
        if parsed is not None:
            package_name, relative_path = parsed
            references[package_name].add(relative_path)
    if malformed:
        formatted = "\n".join(f"  - {issue}" for issue in malformed)
        raise SplitError(f"URDF package URI validation failed:\n{formatted}")

    candidate_roots: list[Path] = []
    for candidate in (source_path.parent, source_path.parent.parent):
        resolved_candidate = candidate.resolve()
        if resolved_candidate not in candidate_roots:
            candidate_roots.append(resolved_candidate)

    bindings: dict[str, Path] = {}
    for package_name, relative_paths in sorted(references.items()):
        if package_name in explicit:
            bindings[package_name] = explicit[package_name]
            continue
        matches = [
            candidate
            for candidate in candidate_roots
            if all(
                (candidate / relative_path).is_file()
                for relative_path in relative_paths
            )
        ]
        option = f"--package-root {package_name}=PATH"
        if not matches:
            searched = ", ".join(str(candidate) for candidate in candidate_roots)
            raise SplitError(
                f"Could not infer a root for package {package_name!r}; no supported "
                f"candidate contains all referenced meshes. Searched: {searched}. "
                f"Pass {option}."
            )
        if len(matches) > 1:
            choices = ", ".join(str(candidate) for candidate in matches)
            raise SplitError(
                f"Package {package_name!r} is ambiguous because every referenced "
                f"mesh exists beneath multiple candidate roots: {choices}. Pass {option}."
            )
        bindings[package_name] = matches[0]
    return bindings


def _resolve_mesh_path(
    filename: str,
    source_path: Path,
    package_roots: Mapping[str, Path],
) -> Path:
    """Resolve one mesh filename using source-relative and package bindings.

    Args:
        filename: Mesh filename exactly as stored in the URDF.
        source_path: Absolute path of the source URDF.
        package_roots: Validated package-name to root bindings.

    Returns:
        The absolute, normalized mesh path.

    Raises:
        SplitError: If the filename is empty, malformed, unsupported, or names
            a package without a resolved binding.
    """

    filename = filename.strip()
    if not filename:
        raise SplitError("the filename attribute is missing or empty")

    package_reference = _parse_package_uri(filename)
    if package_reference is not None:
        package_name, package_relative = package_reference
        package_root = package_roots.get(package_name)
        if package_root is None:
            raise SplitError(
                f"package {package_name!r} has no resolved root; pass "
                f"--package-root {package_name}=PATH"
            )
        candidate = package_root / package_relative
    else:
        if "://" in filename:
            raise SplitError(f"unsupported mesh URI {filename!r}")
        candidate = Path(filename).expanduser()
        if not candidate.is_absolute():
            candidate = source_path.parent / candidate
    return candidate.resolve()


def _validate_meshes(
    robot: ET.Element,
    source_path: Path,
    package_roots: Mapping[str, Path],
) -> tuple[MeshReference, ...]:
    """Validate all meshes and the global visual/collision mesh requirements."""

    references: list[MeshReference] = []
    issues: list[str] = []
    mesh_counts = {"visual": 0, "collision": 0}

    for link_name, geometry_kind, index, mesh in _iter_mesh_elements(robot):
        mesh_counts[geometry_kind] += 1
        filename = mesh.get("filename", "")
        context = f"link {link_name!r} <{geometry_kind}>[{index}] <mesh>"
        try:
            resolved_path = _resolve_mesh_path(filename, source_path, package_roots)
        except SplitError as exc:
            issues.append(f"{context}: {exc}")
            continue
        if not resolved_path.is_file():
            issues.append(
                f"{context}: {filename!r} resolves to missing file {resolved_path}"
            )
            continue
        references.append(
            MeshReference(
                link_name=link_name,
                geometry_kind=geometry_kind,
                geometry_index=index,
                filename=filename,
                resolved_path=resolved_path,
            )
        )

    for geometry_kind in ("visual", "collision"):
        if mesh_counts[geometry_kind] == 0:
            issues.append(
                f"the URDF has no mesh-backed <{geometry_kind}> geometry on any link"
            )
    if issues:
        formatted = "\n".join(f"  - {issue}" for issue in issues)
        raise SplitError(f"URDF mesh validation failed:\n{formatted}")
    return tuple(references)


def _load_urdf(path: Path) -> ET.ElementTree:
    """Parse a URDF XML file while preserving comments."""

    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(path, parser=parser)
    except (OSError, ET.ParseError) as exc:
        raise SplitError(f"Could not parse URDF {path}: {exc}") from exc
    if tree.getroot().tag != "robot":
        raise SplitError(f"Expected a <robot> root element in {path}.")
    return tree


def _validate_joint_kinematics(graph: UrdfGraph) -> None:
    """Validate every joint origin, axis, and q=0 position limit.

    Revolute and prismatic joints must contain finite lower and upper limits
    that include zero because generated inactive arms are fixed at URDF q=0.
    Continuous joints have no position-limit requirement.
    """

    for joint_name, joint in graph.joints.items():
        _origin_transform(joint)
        joint_type = joint.get("type")
        if joint_type in _ACTIVE_JOINT_TYPES:
            _joint_axis(joint)
        if joint_type not in {"prismatic", "revolute"}:
            continue

        limit = joint.find("limit")
        if limit is None:
            raise SplitError(
                f"Joint {joint_name!r} of type {joint_type!r} requires a "
                "<limit lower=... upper=...> element."
            )
        lower_text = limit.get("lower")
        upper_text = limit.get("upper")
        if lower_text is None or upper_text is None:
            raise SplitError(
                f"Joint {joint_name!r} of type {joint_type!r} requires both "
                "lower and upper position limits."
            )
        try:
            lower = float(lower_text)
            upper = float(upper_text)
        except ValueError as exc:
            raise SplitError(
                f"Joint {joint_name!r} has non-numeric position limits: "
                f"lower={lower_text!r}, upper={upper_text!r}."
            ) from exc
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise SplitError(
                f"Joint {joint_name!r} must have finite position limits; "
                f"found lower={lower_text!r}, upper={upper_text!r}."
            )
        if lower > upper:
            raise SplitError(
                f"Joint {joint_name!r} has reversed position limits "
                f"[{lower}, {upper}]."
            )
        if not lower <= 0.0 <= upper:
            raise SplitError(
                f"Joint {joint_name!r} cannot use URDF q=0 as its inactive "
                f"rest position because zero is outside its position limits "
                f"[{lower}, {upper}]."
            )


def validate_urdf(
    source_path: str | Path,
    *,
    package_roots: Mapping[str, str | Path] | None = None,
) -> ValidationReport:
    """Load and validate a source URDF without modifying it.

    Args:
        source_path: Source URDF path. Relative paths are resolved from the
            current working directory.
        package_roots: Optional explicit roots keyed by package name. Missing
            package roots are inferred only when one supported candidate
            satisfies every reference for that package.

    Returns:
        Parsed topology, resolved package bindings, and mesh references.

    Raises:
        SplitError: If the source is not a file or fails XML, topology, joint,
            package-root, or mesh validation.
    """

    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise SplitError(f"The source URDF does not exist or is not a file: {path}")
    tree = _load_urdf(path)
    graph = _build_graph(tree.getroot())
    _validate_joint_kinematics(graph)
    resolved_package_roots = _resolve_package_roots(tree.getroot(), path, package_roots)
    mesh_references = _validate_meshes(tree.getroot(), path, resolved_package_roots)
    return ValidationReport(
        source_path=path,
        tree=tree,
        graph=graph,
        mesh_references=mesh_references,
        package_roots=tuple(sorted(resolved_package_roots.items())),
    )


def _find_descendant_joint_path(
    graph: UrdfGraph, start_joint: str, end_joint: str
) -> list[ET.Element] | None:
    """Return the unique tree path from a start joint to its descendant."""

    start = graph.joints[start_joint]

    def visit(joint: ET.Element) -> list[ET.Element] | None:
        """Search the already validated tree below one joint."""

        if joint.get("name") == end_joint:
            return [joint]
        for child_joint in graph.outgoing.get(_joint_child(joint), ()):
            suffix = visit(child_joint)
            if suffix is not None:
                return [joint, *suffix]
        return None

    return visit(start)


def _format_branch(last_joint: ET.Element, choices: list[ET.Element]) -> str:
    """Format a branch point and its candidate outgoing joints."""

    child_link = _joint_child(last_joint)
    lines = [
        f"Branch encountered after joint {last_joint.get('name')!r} at link {child_link!r}:"
    ]
    for joint in choices:
        lines.append(
            f"  - {joint.get('name')} [{joint.get('type')}]: "
            f"{_joint_parent(joint)} -> {_joint_child(joint)}"
        )
    return "\n".join(lines)


def _resolve_arm_selection(
    graph: UrdfGraph,
    *,
    label: str,
    first_joint_name: str,
    end_joint_name: str | None,
) -> ArmSelection:
    """Resolve one arm path without prompting or other CLI side effects."""

    if first_joint_name not in graph.joints:
        raise SplitError(f"Unknown {label} first actuator joint: {first_joint_name!r}.")
    first_joint = graph.joints[first_joint_name]
    if first_joint.get("type") not in _ACTIVE_JOINT_TYPES:
        raise SplitError(
            f"The {label} first actuator {first_joint_name!r} must be revolute, "
            f"continuous, or prismatic; found {first_joint.get('type')!r}."
        )

    if end_joint_name is not None:
        if end_joint_name not in graph.joints:
            raise SplitError(f"Unknown {label} end joint: {end_joint_name!r}.")
        path = _find_descendant_joint_path(graph, first_joint_name, end_joint_name)
        if path is None:
            raise SplitError(
                f"Joint {end_joint_name!r} is not a descendant of {first_joint_name!r}."
            )
    else:
        path = []
        current = first_joint
        while True:
            path.append(current)
            choices = graph.outgoing.get(_joint_child(current), [])
            if not choices:
                break
            if len(choices) > 1:
                option = f"--{label}-end-joint"
                raise SplitError(
                    f"Cannot infer the {label} arm endpoint. "
                    f"{_format_branch(current, choices)}\nPass {option} with the "
                    "last joint belonging to the arm."
                )
            current = choices[0]

    path_names = tuple(joint.get("name", "") for joint in path)
    active_joints = tuple(
        joint.get("name", "")
        for joint in path
        if joint.get("type") in _ACTIVE_JOINT_TYPES
    )
    if not active_joints:
        raise SplitError(f"The selected {label} arm path has no movable joints.")
    return ArmSelection(
        label=label,
        first_joint=first_joint_name,
        path=path_names,
        active_joints=active_joints,
    )


def _subtree_joint_names(graph: UrdfGraph, first_joint_name: str) -> set[str]:
    """Return every joint in the subtree rooted at a selected first joint."""

    names: set[str] = set()
    stack = [graph.joints[first_joint_name]]
    while stack:
        joint = stack.pop()
        name = joint.get("name", "")
        names.add(name)
        stack.extend(graph.outgoing.get(_joint_child(joint), ()))
    return names


def _validate_arm_pair(
    graph: UrdfGraph, left: ArmSelection, right: ArmSelection
) -> None:
    """Require the two selected arms to occupy independent topology branches."""

    left_subtree = _subtree_joint_names(graph, left.first_joint)
    right_subtree = _subtree_joint_names(graph, right.first_joint)
    overlap = sorted(left_subtree & right_subtree)
    if overlap:
        raise SplitError(
            "Left and right first actuators must root independent subtrees; "
            f"overlapping joints: {overlap}."
        )


def _identity() -> Matrix4:
    """Return a four-dimensional identity transform."""

    return [[1.0 if row == col else 0.0 for col in range(4)] for row in range(4)]


def _multiply(a: Matrix4, b: Matrix4) -> Matrix4:
    """Multiply two homogeneous transforms."""

    return [
        [sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def _origin_transform(joint: ET.Element) -> Matrix4:
    """Convert a joint's URDF origin into a homogeneous transform."""

    origin = joint.find("origin")
    xyz_text = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    rpy_text = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    try:
        xyz = [float(value) for value in xyz_text.split()]
        rpy = [float(value) for value in rpy_text.split()]
    except ValueError as exc:
        raise SplitError(
            f"Joint {joint.get('name')!r} has a non-numeric origin."
        ) from exc
    if (
        len(xyz) != 3
        or len(rpy) != 3
        or not all(math.isfinite(value) for value in [*xyz, *rpy])
    ):
        raise SplitError(
            f"Joint {joint.get('name')!r} must have finite three-value xyz/rpy."
        )

    roll, pitch, yaw = rpy
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


def _inverse_rigid(transform: Matrix4) -> Matrix4:
    """Invert a homogeneous rigid transform."""

    rotation_t = [[transform[col][row] for col in range(3)] for row in range(3)]
    translation = [transform[row][3] for row in range(3)]
    inverse_translation = [
        -sum(rotation_t[row][k] * translation[k] for k in range(3)) for row in range(3)
    ]
    return [[*rotation_t[row], inverse_translation[row]] for row in range(3)] + [
        [0.0, 0.0, 0.0, 1.0]
    ]


def _joint_axis(joint: ET.Element) -> list[float]:
    """Return a normalized local axis for a one-degree-of-freedom joint."""

    if joint.get("type") not in _ACTIVE_JOINT_TYPES:
        raise SplitError(
            f"Joint {joint.get('name')!r} must be revolute, continuous, or "
            f"prismatic; found {joint.get('type')!r}."
        )
    axis = joint.find("axis")
    axis_text = axis.get("xyz", "1 0 0") if axis is not None else "1 0 0"
    try:
        values = [float(value) for value in axis_text.split()]
    except ValueError as exc:
        raise SplitError(
            f"Joint {joint.get('name')!r} has a non-numeric axis."
        ) from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise SplitError(
            f"Joint {joint.get('name')!r} must have a finite three-value axis."
        )
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise SplitError(f"Joint {joint.get('name')!r} has a zero-length axis.")
    return [value / norm for value in values]


def _cross(a: list[float], b: list[float]) -> list[float]:
    """Return the three-dimensional cross product of two vectors."""

    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _rotation_aligning_axis_with_world_z(axis: list[float]) -> Matrix4:
    """Map the positive first-actuator axis to world negative Z."""

    target = [0.0, 0.0, -1.0]
    cross = _cross(axis, target)
    sine = math.sqrt(sum(value * value for value in cross))
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(axis, target))))

    if sine <= 1e-12:
        if cosine > 0.0:
            return _identity()
        reference = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
        rotation_axis = _cross(axis, reference)
        axis_norm = math.sqrt(sum(value * value for value in rotation_axis))
        rotation_axis = [value / axis_norm for value in rotation_axis]
        rotation = [
            [
                2.0 * rotation_axis[row] * rotation_axis[col]
                - (1.0 if row == col else 0.0)
                for col in range(3)
            ]
            for row in range(3)
        ]
    else:
        x, y, z = cross
        skew = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
        skew_squared = [
            [sum(skew[row][k] * skew[k][col] for k in range(3)) for col in range(3)]
            for row in range(3)
        ]
        scale = (1.0 - cosine) / (sine * sine)
        rotation = [
            [
                (1.0 if row == col else 0.0)
                + skew[row][col]
                + skew_squared[row][col] * scale
                for col in range(3)
            ]
            for row in range(3)
        ]
    return [[*rotation[row], 0.0] for row in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _matrix_to_xyz_rpy(transform: Matrix4) -> tuple[list[float], list[float]]:
    """Convert a homogeneous transform to URDF XYZ and fixed-axis RPY."""

    xyz = [transform[row][3] for row in range(3)]
    r20 = max(-1.0, min(1.0, transform[2][0]))
    pitch = math.asin(-r20)
    if abs(math.cos(pitch)) > 1e-10:
        roll = math.atan2(transform[2][1], transform[2][2])
        yaw = math.atan2(transform[1][0], transform[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-transform[0][1], transform[1][1])
    return xyz, [roll, pitch, yaw]


def _root_to_joint_frame(graph: UrdfGraph, joint_name: str) -> Matrix4:
    """Calculate source-root to joint-frame FK at the URDF zero configuration."""

    selected = graph.joints[joint_name]
    parent_link = _joint_parent(selected)
    ancestors: list[ET.Element] = []
    cursor = parent_link
    while cursor != graph.root_link:
        incoming = graph.incoming.get(cursor)
        if incoming is None:
            raise SplitError(
                f"Could not trace link {cursor!r} back to root {graph.root_link!r}."
            )
        ancestors.append(incoming)
        cursor = _joint_parent(incoming)
    ancestors.reverse()

    transform = _identity()
    for joint in [*ancestors, selected]:
        transform = _multiply(transform, _origin_transform(joint))
    return transform


def _unique_name(preferred: str, existing: set[str]) -> str:
    """Return a deterministic name not already present in a URDF."""

    if preferred not in existing:
        return preferred
    index = 2
    while f"{preferred}_{index}" in existing:
        index += 1
    return f"{preferred}_{index}"


def _freeze_joint_at_zero(joint: ET.Element) -> None:
    """Convert a supported movable joint to fixed at its URDF q=0 pose."""

    if joint.get("type") not in _ACTIVE_JOINT_TYPES:
        raise SplitError(
            f"Cannot freeze unsupported joint {joint.get('name')!r} of type "
            f"{joint.get('type')!r}."
        )
    # At q=0, revolute/continuous rotation and prismatic translation are both
    # identity transforms, so retaining the exact origin preserves the pose.
    _origin_transform(joint)
    _joint_axis(joint)
    joint.set("type", "fixed")
    for child in list(joint):
        if child.tag in _FIXED_JOINT_FIELDS:
            joint.remove(child)


def _remove_inactive_transmissions(
    robot: ET.Element, frozen_joint_names: set[str]
) -> list[str]:
    """Remove standard transmissions that reference newly fixed joints."""

    removed: list[str] = []
    for transmission in list(robot.findall("transmission")):
        referenced = {joint.get("name", "") for joint in transmission.findall("joint")}
        if referenced & frozen_joint_names:
            removed.append(transmission.get("name", "<unnamed>"))
            robot.remove(transmission)
    return removed


def _format_number(value: float) -> str:
    """Format a stable finite floating-point value for URDF XML."""

    if abs(value) < 1e-14:
        value = 0.0
    return f"{value:.17g}"


def _format_vector(values: list[float]) -> str:
    """Format a numeric vector for a URDF attribute."""

    return " ".join(_format_number(value) for value in values)


def _element_signature(element: ET.Element) -> tuple[object, ...]:
    """Return a whitespace-insensitive structural XML signature."""

    text = (element.text or "").strip()
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        text,
        tuple(_element_signature(child) for child in element),
    )


def _matrix_error(a: Matrix4, b: Matrix4) -> float:
    """Return the maximum absolute entry difference between transforms."""

    return max(abs(a[row][col] - b[row][col]) for row in range(4) for col in range(4))


def _validate_generated_tree(
    *,
    source_tree: ET.ElementTree,
    generated_tree: ET.ElementTree,
    selection: ArmSelection,
    expected_root: str,
    source_root_to_actuator: Matrix4,
    new_root_joint: ET.Element,
    frozen_joint_names: set[str],
) -> None:
    """Validate topology, q=0 freezing, geometry, and generated root invariants."""

    source_graph = _build_graph(source_tree.getroot())
    generated_graph = _build_graph(generated_tree.getroot())

    if generated_graph.root_link != expected_root:
        raise SplitError(
            f"Generated root is {generated_graph.root_link!r}, expected {expected_root!r}."
        )
    movable = {
        name
        for name, joint in generated_graph.joints.items()
        if joint.get("type") in _ACTIVE_JOINT_TYPES
    }
    if movable != set(selection.active_joints):
        raise SplitError(
            f"Generated movable joints {sorted(movable)} do not equal selected "
            f"joints {sorted(selection.active_joints)}."
        )

    for name, source_link in source_graph.links.items():
        generated_link = generated_graph.links.get(name)
        if generated_link is None:
            raise SplitError(f"Generated URDF is missing original link {name!r}.")
        if _element_signature(generated_link) != _element_signature(source_link):
            raise SplitError(f"Generated URDF modified original link {name!r}.")

    for joint_name in selection.active_joints:
        if generated_graph.joints[joint_name].get("type") != source_graph.joints[
            joint_name
        ].get("type"):
            raise SplitError(f"Generated URDF changed active joint {joint_name!r}.")

    for joint_name in frozen_joint_names:
        source_joint = source_graph.joints[joint_name]
        generated_joint = generated_graph.joints[joint_name]
        if generated_joint.get("type") != "fixed":
            raise SplitError(f"Generated URDF did not freeze joint {joint_name!r}.")
        origin_error = _matrix_error(
            _origin_transform(source_joint), _origin_transform(generated_joint)
        )
        if origin_error > 1e-12:
            raise SplitError(
                f"Freezing joint {joint_name!r} changed its q=0 origin "
                f"(transform error {origin_error:.3g})."
            )
        retained_fields = [
            child.tag for child in generated_joint if child.tag in _FIXED_JOINT_FIELDS
        ]
        if retained_fields:
            raise SplitError(
                f"Fixed joint {joint_name!r} retains movable-only fields: "
                f"{retained_fields}."
            )

    split_root_to_actuator = _multiply(
        _origin_transform(new_root_joint), source_root_to_actuator
    )
    position_error = max(abs(split_root_to_actuator[row][3]) for row in range(3))
    actuator_axis = _joint_axis(source_graph.joints[selection.first_joint])
    axis_in_split_root = [
        sum(split_root_to_actuator[row][col] * actuator_axis[col] for col in range(3))
        for row in range(3)
    ]
    axis_error = max(
        abs(actual - expected)
        for actual, expected in zip(axis_in_split_root, (0.0, 0.0, -1.0))
    )
    if position_error > 1e-9 or axis_error > 1e-9:
        raise SplitError(
            "Generated base does not place the first actuator at its origin "
            "with its positive axis along world -Z; "
            f"position error is {position_error:.3g}, axis error is {axis_error:.3g}."
        )


def _rewrite_generated_mesh_paths(
    tree: ET.ElementTree,
    source_path: Path,
    package_roots: Mapping[str, Path],
) -> None:
    """Rewrite generated mesh filenames to validated absolute filesystem paths."""

    for _, _, _, mesh in _iter_mesh_elements(tree.getroot()):
        filename = mesh.get("filename", "")
        mesh.set(
            "filename",
            str(_resolve_mesh_path(filename, source_path, package_roots)),
        )
    _validate_meshes(tree.getroot(), source_path, package_roots)


def _generate_arm_tree(
    source: ValidationReport, selection: ArmSelection
) -> GeneratedArm:
    """Generate and validate one selected-arm URDF tree at inactive q=0."""

    generated_tree = copy.deepcopy(source.tree)
    robot = generated_tree.getroot()
    generated_graph = _build_graph(robot)
    active = set(selection.active_joints)
    frozen: list[str] = []
    for name, joint in generated_graph.joints.items():
        if name not in active and joint.get("type") in _ACTIVE_JOINT_TYPES:
            _freeze_joint_at_zero(joint)
            frozen.append(name)

    removed_transmissions = _remove_inactive_transmissions(robot, set(frozen))
    root_to_actuator = _root_to_joint_frame(source.graph, selection.first_joint)
    actuator_axis = _joint_axis(source.graph.joints[selection.first_joint])
    actuator_in_split_root = _rotation_aligning_axis_with_world_z(actuator_axis)
    split_root_to_source_root = _multiply(
        actuator_in_split_root, _inverse_rigid(root_to_actuator)
    )
    xyz, rpy = _matrix_to_xyz_rpy(split_root_to_source_root)

    link_names = set(generated_graph.links)
    joint_names = set(generated_graph.joints)
    safe_label = re.sub(r"[^A-Za-z0-9_]", "_", selection.label)
    base_link_name = _unique_name(f"__split_{safe_label}_arm_base", link_names)
    base_joint_name = _unique_name(
        f"__split_{safe_label}_arm_base_to_original_root", joint_names
    )

    new_base = ET.Element("link", {"name": base_link_name})
    first_joint_index = next(
        (index for index, child in enumerate(robot) if child.tag == "joint"),
        len(robot),
    )
    robot.insert(first_joint_index, new_base)

    root_joint = ET.Element("joint", {"name": base_joint_name, "type": "fixed"})
    ET.SubElement(
        root_joint,
        "origin",
        {"xyz": _format_vector(xyz), "rpy": _format_vector(rpy)},
    )
    ET.SubElement(root_joint, "parent", {"link": base_link_name})
    ET.SubElement(root_joint, "child", {"link": source.graph.root_link})
    robot.append(root_joint)

    _validate_generated_tree(
        source_tree=source.tree,
        generated_tree=generated_tree,
        selection=selection,
        expected_root=base_link_name,
        source_root_to_actuator=root_to_actuator,
        new_root_joint=root_joint,
        frozen_joint_names=set(frozen),
    )
    _rewrite_generated_mesh_paths(
        generated_tree, source.source_path, dict(source.package_roots)
    )
    return GeneratedArm(
        selection=selection,
        tree=generated_tree,
        frozen_joints=tuple(frozen),
        removed_transmissions=tuple(removed_transmissions),
    )


def split_urdf(
    source_path: str | Path,
    *,
    left_first_joint: str,
    right_first_joint: str,
    left_end_joint: str | None = None,
    right_end_joint: str | None = None,
    package_roots: Mapping[str, str | Path] | None = None,
) -> SplitResult:
    """Validate and generate left/right URDF trees without writing files.

    Inactive revolute, continuous, and prismatic joints are frozen at URDF
    q=0, which preserves their existing origins. Active joints of those same
    types remain movable when they occur on a selected path.

    Args:
        source_path: Source bimanual URDF path.
        left_first_joint: First actuator joint of the left arm.
        right_first_joint: First actuator joint of the right arm.
        left_end_joint: Optional last joint on the left path when it branches.
        right_end_joint: Optional last joint on the right path when it branches.
        package_roots: Optional explicit roots keyed by package name.

    Returns:
        Validated generated trees. Mesh paths are absolute in these trees; the
        source file and parsed source tree remain unchanged.

    Raises:
        SplitError: If validation fails or the selections are ambiguous or do
            not root independent arm subtrees.
    """

    if left_first_joint == right_first_joint:
        raise SplitError("Left and right first actuator joints must be different.")
    source = validate_urdf(source_path, package_roots=package_roots)
    left_selection = _resolve_arm_selection(
        source.graph,
        label="left",
        first_joint_name=left_first_joint,
        end_joint_name=left_end_joint,
    )
    right_selection = _resolve_arm_selection(
        source.graph,
        label="right",
        first_joint_name=right_first_joint,
        end_joint_name=right_end_joint,
    )
    _validate_arm_pair(source.graph, left_selection, right_selection)
    return SplitResult(
        source=source,
        left=_generate_arm_tree(source, left_selection),
        right=_generate_arm_tree(source, right_selection),
    )


def default_output_path(source_path: str | Path, label: str) -> Path:
    """Return the conventional ``SOURCE-STEM-LABEL.urdf`` output path."""

    source = Path(source_path).expanduser().resolve()
    return source.with_name(f"{source.stem}-{label}{source.suffix}")


def _write_tree(tree: ET.ElementTree, destination: Path, *, force: bool) -> None:
    """Write one generated tree, refusing an overwrite unless requested."""

    if destination.exists() and not force:
        raise SplitError(
            f"Refusing to overwrite {destination}; pass --force to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_tree = copy.deepcopy(tree)
    ET.indent(output_tree, space="    ")
    output_tree.write(
        destination,
        encoding="utf-8",
        xml_declaration=False,
        short_empty_elements=True,
    )


def write_split_urdfs(
    result: SplitResult,
    *,
    left_output: str | Path | None = None,
    right_output: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write generated left/right trees using explicit or default paths.

    Args:
        result: Generated split returned by :func:`split_urdf`.
        left_output: Optional left destination. Defaults beside the source.
        right_output: Optional right destination. Defaults beside the source.
        force: Whether existing output files may be replaced.

    Returns:
        Absolute left and right output paths.

    Raises:
        SplitError: If outputs collide with each other or the source, or an
            existing destination is not allowed to be overwritten.
    """

    source_path = result.source.source_path
    left_path = (
        Path(left_output).expanduser().resolve()
        if left_output is not None
        else default_output_path(source_path, "left")
    )
    right_path = (
        Path(right_output).expanduser().resolve()
        if right_output is not None
        else default_output_path(source_path, "right")
    )
    if left_path == right_path:
        raise SplitError("Left and right output paths must be different.")
    if source_path in {left_path, right_path}:
        raise SplitError("An output path must not overwrite the source URDF.")

    _write_tree(result.left.tree, left_path, force=force)
    try:
        _write_tree(result.right.tree, right_path, force=force)
    except Exception:
        print(
            f"Left output was written before the right output failed: {left_path}",
            file=sys.stderr,
        )
        raise
    return left_path, right_path


def _prompt_joint(value: str | None, prompt: str) -> str:
    """Return a supplied joint name or obtain it interactively for the CLI."""

    if value:
        return value
    if not sys.stdin.isatty():
        raise SplitError(f"Missing required input: {prompt.rstrip(': ')}.")
    entered = input(prompt).strip()
    if not entered:
        raise SplitError("A joint name is required.")
    return entered


def _print_summary(arm: GeneratedArm, output: Path) -> None:
    """Print a concise CLI summary for one generated arm."""

    selection = arm.selection
    print(f"\n{selection.label.capitalize()} output: {output}")
    print(f"  First actuator frame: {selection.first_joint}")
    print("  First actuator axis: parallel to world Z (positive direction -Z)")
    print(
        f"  Active joints ({len(selection.active_joints)}): "
        + (", ".join(selection.active_joints) or "none")
    )
    print(
        f"  Frozen at URDF q=0 ({len(arm.frozen_joints)}): "
        + (", ".join(arm.frozen_joints) or "none")
    )
    if arm.removed_transmissions:
        print(
            "  Removed transmissions for frozen joints: "
            + ", ".join(arm.removed_transmissions)
        )


def _parse_package_root_args(values: list[str]) -> dict[str, Path]:
    """Parse repeatable CLI package-root bindings in PACKAGE=PATH form."""

    package_roots: dict[str, Path] = {}
    for value in values:
        package_name, separator, root_text = value.partition("=")
        package_name = package_name.strip()
        root_text = root_text.strip()
        if not separator or not package_name or not root_text:
            raise SplitError(
                f"Invalid --package-root {value!r}; expected PACKAGE=PATH."
            )
        if package_name in package_roots:
            raise SplitError(
                f"Package root for {package_name!r} was provided more than once."
            )
        package_roots[package_name] = Path(root_text)
    return package_roots


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the production split command without draft compatibility modes."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate a bimanual URDF and generate one complete-geometry URDF "
            "per arm."
        )
    )
    parser.add_argument("urdf", type=Path, help="Source bimanual URDF.")
    parser.add_argument("--left-first-joint", help="First left-arm actuator joint.")
    parser.add_argument("--right-first-joint", help="First right-arm actuator joint.")
    parser.add_argument(
        "--left-end-joint",
        help="Last left-arm joint; required when its selected subtree branches.",
    )
    parser.add_argument(
        "--right-end-joint",
        help="Last right-arm joint; required when its selected subtree branches.",
    )
    parser.add_argument("--left-output", type=Path, help="Left output URDF path.")
    parser.add_argument("--right-output", type=Path, help="Right output URDF path.")
    parser.add_argument(
        "--package-root",
        action="append",
        default=[],
        metavar="PACKAGE=PATH",
        help=(
            "Bind a package:// name to a package root. Repeat for multiple "
            "packages; unambiguous source-adjacent layouts are inferred."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the one-shot split CLI and return its process status."""

    args = _parse_args(argv)
    left_first_joint = _prompt_joint(
        args.left_first_joint, "First actuator joint of the left arm: "
    )
    right_first_joint = _prompt_joint(
        args.right_first_joint, "First actuator joint of the right arm: "
    )
    package_roots = _parse_package_root_args(args.package_root)
    result = split_urdf(
        args.urdf,
        left_first_joint=left_first_joint,
        right_first_joint=right_first_joint,
        left_end_joint=args.left_end_joint,
        right_end_joint=args.right_end_joint,
        package_roots=package_roots,
    )
    left_output, right_output = write_split_urdfs(
        result,
        left_output=args.left_output,
        right_output=args.right_output,
        force=args.force,
    )
    _print_summary(result.left, left_output)
    _print_summary(result.right, right_output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
