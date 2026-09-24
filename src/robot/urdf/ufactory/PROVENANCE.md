# UFACTORY xArm UF850 asset provenance

This integration records the pinned upstream source and the local adaptation
of its UF850 description assets. The upstream source is
[`xArm-Developer/xarm_ros@aad7e1611c9c46eb719045414394bfdd42dcb0f8`](https://github.com/xArm-Developer/xarm_ros/tree/aad7e1611c9c46eb719045414394bfdd42dcb0f8).

- All 14 canonical expanded-binary UF850 STL blobs in `meshes/uf850` were
  hash-matched to the corresponding upstream files in
  [`xarm_description/meshes/uf850`](https://github.com/xArm-Developer/xarm_ros/tree/aad7e1611c9c46eb719045414394bfdd42dcb0f8/xarm_description/meshes/uf850).
- `urdf/uf850.urdf` is a locally derived and adapted plain URDF, not an exact
  copy of an upstream generated file. It derives its mechanical values from
  the official parameterized
  [`uf850.urdf.xacro`](https://github.com/xArm-Developer/xarm_ros/blob/aad7e1611c9c46eb719045414394bfdd42dcb0f8/xarm_description/urdf/uf850/uf850.urdf.xacro).
- The local URDF enables the `link_eef` link and fixed joint that are commented
  in the upstream xacro, and its mesh references use the public package's
  relative paths such as `../meshes/uf850/visual/link1.stl`.
- The pinned upstream [`LICENSE`](https://github.com/xArm-Developer/xarm_ros/blob/aad7e1611c9c46eb719045414394bfdd42dcb0f8/LICENSE)
  notice is retained byte-for-byte in `LICENSE` beside these adapted
  integration assets (SHA256
  `685e81e0cd30124e24e834ea2bf39bfead4efc9907d021bc9d1ae0f2dcea1431`).

This is a factual source and modification record. It is not a legal opinion,
redistribution clearance, or hardware-readiness qualification.
