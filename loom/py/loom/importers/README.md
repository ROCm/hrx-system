# Loom Importers

Loom importers convert external compiler and kernel program surfaces into Loom
IR. They are optional because their frontend packages can be large,
platform-specific, or faster-moving than the base developer environment.

## Position

Importers are onramps from existing program corpora and compiler ecosystems
into Loom. Their job is to preserve useful kernel intent, lowering evidence,
and engineering experience without turning every source ecosystem into a
permanent product surface. A supported importer should have cost proportional
to the value it is currently carrying: reproducible setup, focused fixtures,
clear diagnostics, and explicit build selection, but no default dependency
weight for developers who are not working on that bridge.

The build selection and Python package environment are separate on purpose.
Bazel and CMake can expose importer targets without pulling frontend packages
into the base module lock, while checked-in importer environment locks give CI
and local developers a reproducible package surface when an importer needs one.

## Build Selection

Bazel importer targets are disabled until the corresponding import config is
selected:

```bash
iree-bazel-configure -DLOOM_IMPORT_TILELANG=ON
iree-bazel-configure -DLOOM_IMPORT_MLIR=ON
iree-bazel-test --importer-env tilelang <targets>
iree-bazel-test --importer-env mlir <targets>
iree-bazel-test --importer-env mlir --importer-env tilelang <targets>
```

The native Bazel setting behind those shorthands is:

```bash
--//loom/config/import:enable=tilelang
--//loom/config/import:enable=mlir,tilelang
```

CMake uses matching project options:

```bash
iree-cmake-configure -DLOOM_IMPORT_TILELANG=ON
iree-cmake-configure -DLOOM_IMPORT_MLIR=ON
```

## Managed Python Environments

Managed importer environments are selected through `dev.py importers`:

```bash
python dev.py importers setup tilelang
python dev.py importers setup mlir
python dev.py importers doctor tilelang
python dev.py importers doctor mlir
python dev.py importers env tilelang --format=shell
python dev.py importers env mlir --format=shell
```

`setup` materializes the environment under `.tmp/importers/<name>/` from a
checked-in lock file named `requirements-importers-<name>.lock.txt`. `doctor`
re-runs the import probes and updates the environment manifest. `env` prints
the manifest or shell-style `PYTHONPATH` value.

Build-system commands accept `--importer-env <name>`, which selects both the
build toggle and the locked Python environment:

```bash
iree-bazel-test --config=asan \
    --importer-env tilelang \
    //loom/py/loom/importers/tilelang:tilelang_import_test
iree-bazel-test --config=asan \
    --importer-env mlir \
    //loom/py/loom/importers/mlir:mlir_import_test

iree-cmake-configure --importer-env tilelang
iree-cmake-configure --importer-env mlir
iree-cmake-build --importer-env tilelang loom_tools_loom-opt_loom-opt
iree-cmake-test --importer-env tilelang -R tilelang
iree-cmake-test --importer-env mlir -R mlir
```

The flag refuses missing, stale, or failed environments instead of silently
running against whatever packages happen to be importable.

## CI

Importer CI is intentionally separate from the base Bazel and CMake lanes. The
`CI Importers` workflow is path-scoped to importer code, importer environment
locks, and devtools/CI machinery. Its TileLang matrix entry installs the
hash-locked environment, prints the environment manifest, and runs the
dedicated `iree-importers-tilelang` command profile across Bazel and CMake.
Checked-in importer tests use fail-on-skip semantics so missing frontend
packages fail the selected importer lane instead of reporting a false green.

## Importer Map

- `tilelang/` imports TileLang kernels. It has a managed locked environment and
  Bazel/CMake test coverage. Start with `tilelang/README.md`.
- `mlir/` imports IREE HAL executable MLIR through IREE's Python MLIR bindings.
  It is a narrow continuity bridge, not a general upstream MLIR frontend. It
  has a managed locked environment and Bazel/CMake test coverage. Start with
  `mlir/README.md`.
- `check/` owns the shared golden fixture runner. Its README is for fixture
  authoring, update mode, diagnostics, and oracle sidecar capture after an
  importer has already been enabled.

New managed importer environments follow the same shape: a root lock file named
`requirements-importers-<name>.lock.txt`, a `dev.py importers` environment
entry, Bazel and CMake enablement through `//loom/config/import/...` and
`LOOM_IMPORT_<NAME>`, and a README under the importer directory with setup,
test, and package-boundary notes.
