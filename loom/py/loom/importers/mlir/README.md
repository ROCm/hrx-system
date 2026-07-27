# MLIR Importer

The MLIR importer converts IREE HAL executable MLIR into Loom IR. It is a
continuity bridge for existing IREE-lowered kernels, lowering archaeology,
backend comparison, and importer coverage that starts from already-lowered
compiler output instead of a source-level kernel DSL.

The supported entry shape is intentionally narrow: checked IREE HAL executable
MLIR that carries HAL binding metadata, executable exports, target attributes,
and IREE codegen attributes. This importer is not a general upstream MLIR
frontend and does not imply support for arbitrary dialect mixes or every MLIR
producer that can print a `func.func`.

The importer consumes IREE's Python MLIR bindings through `iree.compiler`. Use
the managed importer environment for normal local and CI work:

```bash
python dev.py importers setup mlir
python dev.py importers doctor mlir
```

The practical release package for this importer is `iree-base-compiler`. It
provides both the upstream MLIR dialect modules used here (`arith`, `func`,
`gpu`, `memref`, `scf`, `vector`, and related dialects) and the IREE dialect
modules needed by HAL executable inputs (`hal`, `iree_codegen`, `iree_gpu`).
The PyPI package named `mlir` is not this binding surface; it does not provide
`mlir.ir` or dialect modules.

## Bazel

Enable MLIR importer targets with the managed environment:

```bash
iree-bazel-test --config=asan \
    --importer-env mlir \
    //loom/py/loom/importers/mlir:mlir_test \
    //loom/py/loom/importers/mlir:mlir_import_test
```

`--importer-env mlir` selects the dedicated Bazel config and forwards the locked
site-packages path through Bazel `test_env`. The native build setting behind
that shorthand is `--//loom/config/import:enable=mlir`.

For direct fixture inspection:

```bash
iree-bazel-run --config=asan \
    --importer-env mlir \
    //loom/py/loom/importers/check:loom_import_check -- \
    mlir loom/py/loom/importers/mlir/testdata/vector_reduce_amdgpu.mlir
```

`loom-import-check mlir --prefer-abi3-extensions ...` prefers freshly built
stable-ABI extension modules when local build trees expose both ABI-tagged and
non-ABI-tagged modules.

## CMake

CMake uses the matching managed environment:

```bash
iree-cmake-configure --importer-env mlir
iree-cmake-test --importer-env mlir -R mlir
```

`--importer-env mlir` adds `-DLOOM_IMPORT_MLIR=ON` at configure time and runs
CMake commands with the locked importer site-packages path on `PYTHONPATH`.

## Package Boundary

MLIR is build-selectable in both build systems, and its normal package
environment is managed through `requirements-importers-mlir.lock.txt` and
`python dev.py importers`.

The managed package exists to keep the bridge reproducible without adding LLVM
or MLIR to the core repository dependency graph. Updating this lock should be a
deliberate compatibility check against the checked fixtures and any imported
customer or research kernels that are meant to stay live.

An upstream LLVM/MLIR binding environment would need to expose `mlir.ir` and
the upstream dialect modules on `PYTHONPATH`. That can parse and walk upstream
dialects, but it does not provide the IREE HAL dialects or HAL executable ABI
metadata this importer currently uses to discover kernel bindings, target
format, and launch configuration. Supporting that lane is a separate generic
MLIR entry-envelope importer rather than a package swap.
