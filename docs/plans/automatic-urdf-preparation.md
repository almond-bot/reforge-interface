# One-Time URDF Preparation and `robot_interface` Runtime Selection Plan

## Status and scope

This plan targets `/home/reforge/Desktop/almond/reforge-interface`. Phase 1 is
implemented and verified, and is awaiting the user review gate. Later phases
have not started.

The work validates one externally supplied source URDF, creates any required
per-arm URDFs, persists runtime metadata, and lets `robot_interface.py` select
the prepared arm. Multi-arm Joint Tracker calibration-model storage and
selection remain a separate task. No `reforge_core` source is changed.

The operator runs one preparation command before any
`python3 -m robot.run ...` command. Normal robot commands never prompt,
regenerate assets, or write URDF state.

This plan remains the implementation record through Phase 4. Phase 5 deletes
it, together with every other task-generated planning or review document,
after the durable operator guidance has been incorporated into the existing
robot README.

## Goal

Given one source URDF, the one-time setup command:

- validates the URDF topology, joint definitions, and required mesh-backed
  visual and collision geometry;
- resolves both relative and `package://` mesh references without modifying
  the source URDF;
- determines whether one runtime profile or separate left/right profiles are
  required;
- obtains only split boundaries that cannot be derived from topology;
- generates and validates all required runtime URDFs, freezing inactive joints
  at the URDF-defined `q=0` configuration;
- derives `FULL_STRETCH_JOINTS` from each generated URDF;
- evaluates FK at those joints to derive `FULL_STRETCH_XYZ` and
  `FULL_STRETCH_QUAT`; and
- stores the runtime values in a versioned sidecar manifest.

After setup, the existing `USE_LEFT` switch in `robot_interface.py` selects the
entire left or right runtime configuration. `run.py` remains unchanged.

## Settled decisions

- Rename `src/robot/split_urdf_draft.py` to
  `src/robot/split_urdf.py`; no production path retains the draft name.
- Keep CLI orchestration, reusable validation/splitting, later manifest logic,
  and runtime profile loading in that one module. Do not add separate
  `prepare_urdf.py` or `urdf_assets.py` modules.
- Use this one-time setup command:

  ```text
  python3 -m robot.split_urdf /path/to/source.urdf
  ```

- Generate every profile required by the robot in one invocation. For Axol,
  setup produces both left and right URDFs and profiles.
- Treat the configuration represented by the source URDF as the rest pose.
  In URDF terms, inactive movable joints are frozen at `q=0`; their existing
  `<origin>` transform is preserved exactly. There are no rest-position maps,
  rest-position arguments, or rest-position prompts. A bounded revolute or
  prismatic joint whose limits exclude zero is rejected because that source
  cannot represent the promised rest configuration at `q=0`.
- Validate `package://PACKAGE/PATH` references and automatically rewrite them
  in generated URDFs to resolved filesystem paths. Infer a package root only
  when there is one valid candidate. A repeatable explicit
  `--package-root PACKAGE=PATH` override resolves missing or ambiguous package
  roots. The source URDF is always left byte-for-byte unchanged.
- Resolve ordinary relative mesh paths from the source URDF directory. Reject
  unsupported URI schemes and missing files with element-specific diagnostics.
- Require at least one mesh-backed `<visual>` and one mesh-backed `<collision>`
  globally. Validate every mesh reference, but do not require both tags on
  every link.
- Keep `USE_LEFT` as the source-level selector. A changed value takes effect in
  a fresh Python process, matching normal `python3 -m robot.run ...` usage.
- Do not add `--robot-side`, alter route parsing, or edit `run.py`.
- Preserve the exported names `URDF_PATH`, `FULL_STRETCH_XYZ`,
  `FULL_STRETCH_QUAT`, and `FULL_STRETCH_JOINTS`.
- Use `URDF_MANIFEST_PATH` for the sidecar location.
- Derive verification from manifest versions and fingerprints. Do not persist
  a standalone `URDF_VERIFIED` Boolean that can become stale.
- Never rewrite Python source or constants during setup.
- Target a writable source checkout in v1. Generated URDFs and the manifest
  live under `src/robot/urdf/`.
- Do not use an inter-process lock. Concurrent setup invocations are
  unsupported. Unique staging paths plus manifest-last publication prevent
  runtime code from accepting a partial generation.

## Runtime architecture

### Stable selection in `robot_interface.py`

`robot_interface.py` retains the user-facing switch and names the manifest:

```python
USE_LEFT = False
ROBOT_SIDE = RobotSide.LEFT if USE_LEFT else RobotSide.RIGHT
URDF_MANIFEST_PATH = "urdf/axol.reforge-urdf.json"
```

`RobotSide` is a `StrEnum`. CAN channel, SDK arm, TCP link, joint names, and
manifest profile selection all derive from `ROBOT_SIDE`, avoiding independent
Boolean branches. One process selects one side; flipping `USE_LEFT` requires a
restart.

### Versioned sidecar manifest

```text
src/robot/urdf/
├── axol.urdf
├── axol.reforge-urdf.json
├── axol-left.urdf
└── axol-right.urdf
```

The manifest records:

- schema and preparation-pipeline versions;
- source path and SHA-256;
- original mesh identifiers, resolved paths, and SHA-256 values;
- inferred or explicitly overridden package roots needed for reproducibility;
- robot topology, required profiles, and split boundaries;
- package-relative generated URDF paths and SHA-256 values;
- active joint names in model order;
- full-stretch joints in that order, in radians;
- FK-derived XYZ in metres and XYZW quaternion; and
- a transformed pose override, or explicit `null` when none applies.

There are no stored inactive-rest maps: `q=0` is the source-defined rest
configuration, and each generated fixed joint preserves the source origin.

Verification is derived as:

```text
urdf_verified =
    supported schema and pipeline versions
    AND source and mesh fingerprints match
    AND every required generated output exists and hash-matches
    AND the selected profile is internally consistent
```

Missing or stale state raises an actionable error directing the operator to
rerun `python3 -m robot.split_urdf ...`. Runtime loading never prompts or
writes files.

### Immutable selected profile

`split_urdf.py` exposes an immutable runtime value:

```python
@dataclass(frozen=True)
class RobotAssetProfile:
    robot_side: RobotSide
    urdf_path: str
    active_joint_order: tuple[str, ...]
    full_stretch_joints_rad: tuple[float, ...]
    full_stretch_xyz_m: tuple[float, float, float]
    full_stretch_quaternion_xyzw: tuple[float, float, float, float]
    full_stretch_pose_override: tuple[float, ...] | None
```

At module import, `robot_interface.py` validates and selects one profile:

```python
_ACTIVE_PROFILE = load_robot_asset_profile(URDF_MANIFEST_PATH, ROBOT_SIDE)

URDF_PATH = _ACTIVE_PROFILE.urdf_path
FULL_STRETCH_JOINTS = list(_ACTIVE_PROFILE.full_stretch_joints_rad)
FULL_STRETCH_XYZ = list(_ACTIVE_PROFILE.full_stretch_xyz_m)
FULL_STRETCH_QUAT = list(_ACTIVE_PROFILE.full_stretch_quaternion_xyzw)
FULL_STRETCH_POSE_OVERRIDE = _ACTIVE_PROFILE.full_stretch_pose_override
URDF_VERIFIED = True
```

These values are process-local snapshots. Their existing names preserve the
interface imported by `run.py`. `RobotInterface.__init__()` consumes the same
selected profile and cheaply revalidates it before model loading or hardware
connection.

## Mesh resolution and generated-output conversion

Validation separates the source identifier from its resolved path. Ordinary
relative paths resolve from the source URDF directory. For each package URI,
the resolver finds package-root candidates from the source layout and accepts
an inferred root only if exactly one candidate makes the referenced asset
exist. `--package-root PACKAGE=PATH` takes precedence and must point to a
directory that resolves every referenced path for that package. Unknown,
ambiguous, malformed, or incomplete mappings fail with the package name and
owning XML element in the diagnostic.

Generated trees replace all accepted relative and package URIs with normalized
resolved paths. Validation and conversion operate on parsed copies; the source
file and its parsed source tree retain their original filenames. The manifest
preserves the original identifiers and resolution metadata for staleness
checks and diagnostics.

## Inactive-joint rest behavior

For the generated profile, every movable joint outside the selected arm path
is changed to `fixed`. At `q=0`, revolute and continuous motion contribute an
identity rotation, and prismatic motion contributes an identity translation,
so retaining the joint's exact source `<origin>` preserves the promised source
configuration. Motion-only children such as `<axis>`, `<limit>`, and
`<dynamics>` are removed from the fixed joint.

The validator rejects floating and planar joints in v1 and checks that zero is
within every bounded revolute or prismatic joint's finite limits. Generated
tree validation compares each frozen joint's transform with the source `q=0`
transform. No rest values are requested from the operator or stored in the
manifest.

## Generating all `FULL_STRETCH_*` values from the URDF

`FULL_STRETCH_JOINTS` is generated rather than requested from the operator. It
cannot be obtained by FK alone, so setup defines “full stretch” as a
deterministic optimization over the generated URDF:

1. Read active joints in model order and build the base-to-TCP chain.
2. Search within joint limits for the collision-free configuration maximizing
   TCP distance from the chain root.
3. Resolve equal-reach solutions by preferring greater `+Z` in the generated
   frame, then the smallest squared displacement from `q=0`.
4. Reject out-of-limit, self-colliding, or nonconverged solutions.
5. Evaluate FK at the selected joint vector to produce XYZ and XYZW quaternion.
6. Persist the joint vector and pose together in the manifest.

The repository has no `FULL_STRETCH_TCP` symbol. The TCP pose remains
`FULL_STRETCH_XYZ` plus `FULL_STRETCH_QUAT`, avoiding a `run.py` change.

## Single-module preparation flow

`split_urdf.py` contains guarded CLI `main()` and reusable functions used by
tests and, later, `robot_interface.py`. Imports have no CLI side effects. Its
cohesive boundaries include:

```python
validate_urdf(source_path, package_roots=None) -> ValidationReport
split_urdf(source_path, ..., package_roots=None) -> SplitResult
prepare_robot_assets(source_path, robot_spec, package_roots=None) -> PreparationResult
load_robot_asset_profile(manifest_path, robot_side) -> RobotAssetProfile
validate_robot_asset_profile(profile, manifest_path) -> None
```

`prepare_robot_assets()` performs:

1. Parse the source and validate topology, joints, and all mesh references.
2. Infer package roots or apply explicit `PACKAGE=PATH` overrides.
3. Detect or confirm topology. A unimanual source produces one profile; Axol
   produces left and right profiles.
4. Obtain only split boundaries that cannot be derived. Reusable functions
   never call `input()`.
5. Generate outputs into unique staging paths under `src/robot/urdf/`, freeze
   inactive joints at `q=0`, and rewrite generated mesh filenames.
6. Validate generated topology, active order, frozen transforms, and meshes.
7. Generate full-stretch joints and derive their TCP poses.
8. Replace validated output files and write the manifest last.

If publication is interrupted, the old manifest cannot validate a mixed
generation. Rerunning setup repairs the assets. There is deliberately no lock
or concurrency machinery.

## Why `run.py` does not change

`run.py` imports `URDF_PATH` and the `FULL_STRETCH_*` values from
`robot_interface.py`, then builds simulator configuration from those values.
Because `robot_interface.py` resolves `_ACTIVE_PROFILE` before those imports
complete, simulation and live construction receive the same side without
route changes.

The deliberate tradeoff is that side selection is not a command-line choice,
and one process cannot construct clients for both sides. That matches the
current one-arm-at-a-time workflow. A future per-command side option would
require passing side/profile state through `run.py` and is outside this task.

## Expected files

- `src/robot/split_urdf.py`: renamed production module containing the CLI,
  validation, splitting, path conversion, later full-stretch generation,
  manifest publication, and runtime profile loading.
- `src/robot/robot_interface.py`: later `USE_LEFT`-selected profile loading,
  existing exports, and constructor revalidation.
- `src/robot/README.md`: later concise one-time setup and side-switch guidance.
- `src/robot/tests/test_split_urdf.py`: risk-focused split, validation,
  conversion, pose, persistence, and profile-loading tests as phases advance.
- Focused robot-interface tests for later left/right selection.

`src/robot/split_urdf_draft.py` no longer exists after Phase 1. `src/robot/run.py`
is not an implementation target. No `/home/reforge/reforge-core` source or
visualizer file is changed.

## Acceptance criteria

- [ ] One setup command accepts an external source URDF and creates every
      required runtime profile.
- [ ] CLI and reusable logic live in `split_urdf.py`; no `prepare_urdf.py` or
      `urdf_assets.py` exists.
- [ ] Unimanual and Axol inputs use the same validator; unimanual preparation
      does not produce unnecessary left/right outputs.
- [ ] Every mesh reference resolves. Package URIs are inferred uniquely or
      resolved by explicit `PACKAGE=PATH` overrides and converted only in
      generated outputs; source URDFs remain unchanged.
- [ ] Global visual/collision mesh requirements produce useful diagnostics.
- [ ] Axol preparation publishes both side URDFs and one manifest, with the
      manifest written last.
- [ ] Every inactive movable joint is fixed at source `q=0`, preserves its
      exact origin, and requires no rest map or prompt.
- [ ] Full-stretch joints are generated deterministically from each generated
      URDF, respect limits, and are collision-free.
- [ ] XYZ and quaternion equal FK evaluated at the stored full-stretch joints.
- [ ] `USE_LEFT = True` loads only the left profile in a fresh process;
      `USE_LEFT = False` loads only the right profile.
- [ ] `URDF_PATH`, all `FULL_STRETCH_*` exports, CAN/SDK/TCP selection, joint
      names, and constructor fields derive from the same selected side.
- [ ] Every `RobotInterface` construction revalidates assets before model load,
      connection, or motion.
- [ ] Missing, stale, or corrupt assets fail with a rerun-setup instruction;
      normal `robot.run` commands never prompt, split, or write URDF state.
- [ ] Existing `python3 -m robot.run ...` commands work without new arguments
      or edits to `run.py`.
- [ ] Interrupted publication cannot be accepted as valid; rerunning setup
      repairs it. Concurrent setup is documented as unsupported.
- [ ] No core SDK source or Python configuration source is rewritten.
- [ ] All task-generated planning/review documents are deleted before final
      handoff; durable operator guidance lives only in the existing README.

## Implementation phases and review gates

GPT-5.6 Sol at extra-high reasoning owns task management, architecture, and
difficult code. GPT-5.6 Luna is unavailable, so GPT-6 Luna at extra-high
reasoning is the disclosed nearest substitute for bounded tests,
documentation, and mechanical tasks. GPT-5.6 Terra at extra-high reasoning
independently reviews completed code. Sol/Luna address every finding and Terra
rereviews until no actionable findings remain.

After every phase, update this plan with completion notes, run target checks
and relevant tests through the target repository's `.venv`, report results,
and stop for user approval.

### Phase 1 — Rename and productionize the split module

- [x] Rename `split_urdf_draft.py` to `split_urdf.py` and remove draft/viewer
      execution modes.
- [x] Keep guarded CLI orchestration and reusable validation/splitting in the
      same module.
- [x] Add source, topology, joint, and mesh validation with element-specific
      diagnostics.
- [x] Freeze inactive joints at source `q=0`, preserving each exact origin;
      remove explicit nonzero-rest inputs and maps.
- [x] Resolve relative and `package://` meshes, support unique package-root
      inference plus explicit `PACKAGE=PATH` overrides, rewrite only generated
      trees, and leave the source untouched.
- [x] Add risk-focused tests for mesh handling (including explicit, missing,
      and ambiguous package roots), malformed topology, arm selection,
      zero-limit validation, independent root-frame placement, active ordering,
      and inactive-origin preservation.

Phase 1 implementation notes:

- The production API validates without writing, returns parsed validation and
  split results, and keeps filesystem publication in a separate explicit call
  within the same module.
- Supported movable joints are revolute, continuous, and prismatic. Floating
  and planar joints fail with a clear v1 limitation.
- Generated arm URDFs retain full robot geometry, keep only the selected path
  movable, remove transmissions for frozen joints, and preserve the original
  source tree.
- Tests cover package and relative path conversion without source mutation;
  explicit package-root overrides; missing and ambiguous package roots;
  missing/unsupported path diagnostics; global visual/collision requirements;
  disconnected and multi-parent topology; overlapping arms; `q=0` outside
  limits; independent selected-arm root frames; path order; active joint types;
  and exact frozen-origin retention.
- Focused Phase 1 pytest result through the target `.venv`: **14 passed**.
- Ruff and Black both pass after the final import-grouping correction.
- The target repository has no `scripts/check.sh`; therefore no repository check
  script was available to run for this phase.
- A smoke run against the real Axol source generated **7 active and 7 frozen
  joints per side**, converted **36 mesh references**, and left the source URDF
  unchanged.
- Terra's first review used the stale, pre-user-decision plan. Its actionable
  findings about explicit package-root handling and independent-root-frame test
  coverage were addressed. Global `q=0` limit validation was deliberately
  retained because every movable joint is frozen in the opposite output of the
  paired left/right generation.

Review gate: approve the single-module, hardware-independent split engine.

### Phase 2 — Add full-stretch generation and durable manifest state

- [ ] Implement deterministic, limit-aware, collision-aware full-stretch joint
      optimization and FK-derived TCP pose generation.
- [ ] Add the versioned manifest, source/mesh/output fingerprints, unique
      staging paths, and manifest-last publication without a lock.
- [ ] Add one-time CLI prompts only for topology values not derivable from the
      URDF. Do not prompt for rest positions.
- [ ] Test deterministic optimization, FK agreement, invalidation, package-root
      persistence, unimanual and Axol output sets, interruption states, and
      idempotent reruns.

Review gate: approve the generated pose and persistent state model.

### Phase 3 — Integrate selection in `robot_interface.py`

- [ ] Add `RobotSide`, select it from `USE_LEFT`, and load one immutable profile
      through `URDF_MANIFEST_PATH` at module import.
- [ ] Populate existing `URDF_PATH` and `FULL_STRETCH_*` exports from that
      profile without changing `run.py`.
- [ ] Derive CAN, SDK arm, TCP link, joint mappings, and constructor state from
      the same selected side.
- [ ] Revalidate in every constructor before model or hardware work.
- [ ] Test fresh-process left/right imports, constructor consistency, stale
      state failure, and unchanged run integration.
- [ ] Add concise setup and restart-after-`USE_LEFT` guidance to the existing
      `src/robot/README.md`.

Review gate: approve runtime selection and unchanged `run.py` behavior.

### Phase 4 — Independent Terra review and iteration

- [ ] Terra reviews kinematic-objective correctness, transforms, package
      resolution, invalidation, import-time behavior, publication safety,
      simplicity, and tests.
- [ ] Sol handles architecture/correctness findings; Luna handles bounded
      mechanical, test, and README findings.
- [ ] Repeat review until every actionable finding is resolved or raised as a
      genuine user decision.

Review gate: present the review record.

### Phase 5 — Productionization, cleanup, and final documentation

- [ ] Review the whole change and remove exploratory, debugging, dead,
      duplicated, research-only, and temporary code.
- [ ] Remove trivial tests while retaining mesh/path, transform, optimization,
      FK, persistence, publication, and lifecycle protections.
- [ ] Ensure no production path or reference retains the `draft` name.
- [ ] Keep durable operator documentation concise and confined to the existing
      `src/robot/README.md`.
- [ ] Delete this implementation plan and every other task-generated planning,
      review, or scratch document before final handoff.
- [ ] Run `scripts/check.sh` if present and all relevant target tests through
      `/home/reforge/Desktop/almond/reforge-interface/.venv`.
- [ ] Have Terra perform a final production-diff review, address every
      actionable finding, then delete its generated review artifacts.

Review gate: present the final diff, checks, risk-focused tests, and follow-ups.

