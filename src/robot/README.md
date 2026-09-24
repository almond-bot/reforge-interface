# Public robot package

This directory is an installable `robot` package containing one selected
adapter and its public resources. Install it from this directory with:

The COBOTTA PRO 900 URDF and 14 DAE meshes are bundled. The vendored ORiN b-CAP client retains its MIT license and pinned provenance; inclusion does not claim live-hardware or KineCal qualification.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run commands from the repository root. The adapter writes calibration and
identification data below `src/robot/data/` and generated model artifacts below
`src/robot/models/`. Copy configuration files to another writable location
when a run needs local changes; do not edit installed package resources.

For offline command discovery, use:

```bash
python -m robot.run --help
```

This package and its example assets do not by themselves claim hardware or
vendor compatibility. Hardware connection and motion qualification require the
selected adapter's separately reviewed prerequisites.
