# TileLang Importer

The TileLang importer converts selected TileLang kernel programs into Loom IR.
Its product value is not blanket TileLang compatibility; it is a controlled
path for ported TileLang programs where TileLang/TVM can also act as a local
oracle for source and code-object comparison.

Unsupported TileLang constructs are reported as structured diagnostics. Oracle
artifacts are sidecar evidence for comparison; they are not part of the Loom IR
that the importer produces.

## Environment

TileLang has a managed importer environment. The lock lives at the repository
root:

```text
requirements-importers-tilelang.lock.txt
```

The input file is `requirements-importers-tilelang.in`; it records the lock
refresh command and intentionally keeps TileLang's Python dependency closure
outside the base developer environment.

Set up and probe the environment with:

```bash
python dev.py importers setup tilelang
python dev.py importers doctor tilelang
python dev.py importers env tilelang --format=shell
```

The setup command installs from the hash-locked requirements into
`.tmp/importers/tilelang/venv/` and writes a manifest under
`.tmp/importers/tilelang/environment.json`.

## Bazel

Use `--importer-env tilelang` for normal local and CI verification. It selects
`--config=loom-importer-tilelang` and forwards the locked site-packages path to
tests:

```bash
iree-bazel-test --config=asan \
    --importer-env tilelang \
    //loom/py/loom/importers/check/tilelang:tilelang_test \
    //loom/py/loom/importers/tilelang:tilelang_test \
    //loom/py/loom/importers/tilelang:tilelang_import_test
```

When the frontend packages are already importable without the managed
environment, the raw build selection is:

```bash
iree-bazel-test --config=asan \
    --config=loom-importer-tilelang \
    //loom/py/loom/importers/tilelang:tilelang_import_test
```

The native build setting is `--//loom/config/import:enable=tilelang`.

## CMake

CMake uses the same optionality boundary through `LOOM_IMPORT_TILELANG`:

```bash
iree-cmake-configure --importer-env tilelang
iree-cmake-build --importer-env tilelang loom_tools_loom-opt_loom-opt
iree-cmake-test --importer-env tilelang -R tilelang
```

`--importer-env tilelang` adds `-DLOOM_IMPORT_TILELANG=ON` at configure time
and runs CMake build/test commands with the locked importer environment on
`PYTHONPATH`.

## Fixture Checks

The importer golden tests are ordinary Bazel and CMake tests. Fixture files
live under `testdata/`, and checked output is stored inline in the source file
after each `# ----` marker.

Update intentional output changes through the checked-in importer checker
binary:

```bash
iree-bazel-build //loom/src/loom/tools/loom-opt
iree-bazel-run \
    --importer-env tilelang \
    //loom/py/loom/importers/check:loom_import_check -- \
    tilelang --update \
    --loom-opt=bazel-bin/loom/src/loom/tools/loom-opt/loom-opt \
    loom/py/loom/importers/tilelang/testdata/tileops.py
```

The shared fixture runner and source annotation format are documented in the
[importer check README](/loom/py/loom/importers/check/README.md).

## Cooperative Grid Boundary

TileLang kernels that use workgroup-local synchronization, workgroup-local
`T.cumsum`, warp shuffles, warp reductions, `T.match_any_sync`, and lowered
`__match_any_sync` calls can import through the normal kernel/subgroup/workgroup
Loom operations when the target subgroup width matches TileLang's 32-lane warp
contract.

`T.sync_grid` is a different contract. It requires a cooperative-grid launch
that guarantees all participating workgroups are resident and able to make
forward progress together. The importer recognizes `tl.sync_grid` and reports a
cooperative-grid diagnostic instead of translating it to `kernel.barrier`, since
a workgroup barrier would describe a different program. Kernels that contain
this phase boundary should be treated as requiring a future Loom grid
synchronization operation, a multi-dispatch decomposition, or an explicit
target/runtime cooperative-launch contract.

## CI

The dedicated importer CI profile is:

```bash
python3 build_tools/devtools/ci.py iree-importers-tilelang --keep-going
```

The GitHub `CI Importers` workflow runs that profile when TileLang importer
code, importer locks, importer configuration, or devtools CI machinery changes.
It installs the locked TileLang environment, reports the package manifest, and
runs both the Bazel and CMake test surfaces with skip failures enabled.

## Oracle Capture

TileLang checks can optionally ask TileLang/TVM for generated artifacts. Source
oracle mode captures generated device source; code-object mode additionally
compiles, unbundles, and disassembles when the local ROCm and LLVM tools are
available.

```bash
iree-bazel-run --config=asan \
    --importer-env tilelang \
    //loom/py/loom/importers/check:loom_import_check -- \
    tilelang \
    --oracle=source \
    --oracle-output-dir=build/tilelang-oracle \
    loom/py/loom/importers/tilelang/testdata/tilekernels/transpose.py
```

The checked stdout remains imported Loom IR. Oracle output is retained only
when `--oracle-output-dir` or `--dump-temp-dir` is supplied.

## Shared-Memory Feedback

TileLang and Triton kernels are valuable shared-memory corpus inputs because
they often encode padding, swizzling, staging, and vectorization choices that
came from real tuning work. The importer path should preserve that useful
layout intent where it can, then use Loom's AMDGPU feedback to explain the
selected LDS packets and bank pattern. That is an onramp for ported kernels and
kernel authors, not a promise that every TileLang surface becomes a permanent
Loom product API.

The canonical workflow and classification table live in the authoring guide's
[AMDGPU shared-memory feedback](/loom/src/loom/test/corpus/authoring/README.md#amdgpu-shared-memory-feedback)
section. For imported TileLang-style kernels, run the normal importer check or
differential capture, keep the Loom compile report, and inspect the
`source_low.memory_rows` array when source layout intent and selected packet
consequence do not line up. Runtime profiling still decides final performance,
but the compile report moves a large part of the tuning loop before
code-object profiling and source reverse-mapping.

## AMDGPU Differential Reports

TileLang AMDGPU differential reporting compares two real code objects: the
TileLang/TVM code-object oracle and the Loom artifact produced through
`loom-compile --backend=amdgpu-hal`. The comparison is intentionally at the
external disassembly summary level. Raw source, code objects, manifests,
compile reports, and disassembly stay available for inspection, while the
durable signal groups instruction-family mismatches into categories such as
ABI/prologue overhead, memory-addressing shape, wait planning, instruction
selection, LDS traffic, register pressure, and target coverage.

Build the production Loom compiler before running differential experiments:

```bash
iree-bazel-build --config=asan //loom/src/loom/tools/loom-compile
```

Capture the TileLang side with code-object oracle mode:

```bash
iree-bazel-run --config=asan \
    --importer-env tilelang \
    //loom/py/loom/importers/check:loom_import_check -- \
    tilelang \
    --filter tileop_copy_1d \
    --oracle=code-object \
    --dump-temp-dir .tmp/tilelang-oracle \
    --loom-opt=bazel-bin/loom/src/loom/tools/loom-opt/loom-opt \
    loom/py/loom/importers/tilelang/testdata/tileops.py
```

Run the side-by-side capture with differential oracle mode:

```bash
iree-bazel-run --config=asan \
    --importer-env tilelang \
    //loom/py/loom/importers/check:loom_import_check -- \
    tilelang \
    --filter tileop_copy_1d \
    --oracle=differential \
    --dump-temp-dir .tmp/tilelang-differential \
    --loom-compile=bazel-bin/loom/src/loom/tools/loom-compile/loom-compile \
    --loom-opt=bazel-bin/loom/src/loom/tools/loom-opt/loom-opt \
    loom/py/loom/importers/tilelang/testdata/tileops.py
```

The reusable Loom-side API lives in
`loom.importers.tilelang.differential.capture_loom_amdgpu_artifact`. The checker
uses it to write the imported Loom module, compile through the AMDGPU HAL
backend, emit the target HSACO and VMFB, record the compile report and artifact
manifest, disassemble with `llvm-objdump`, and return an
`AmdgpuDifferentialArtifact` ready for `compare_amdgpu_artifacts`.
