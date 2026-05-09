# HRX Runtime

HRX is a shared-library runtime stack for ROCm-oriented execution. `libhrx.so`
exports the public `hrx_*` C ABI plus C++ RAII helpers, and layers a few related
tools around that ABI.

The core API is explicit: callers manage devices, allocators, buffers,
semaphores, fences, streams, queue operations, executable/module loading,
dispatch, and compiler output. Higher-level HIP-facing code maps stream, memory,
module, graph capture, and replay-oriented behavior onto the same runtime
concepts. CPU devices use IREE local-task; GPU devices use IREE AMDGPU when
enabled. HRX also wires in profile capture, remote HAL support from IREE's
runtime libraries, record/replay, and HIP passthrough/tracing tools for comparison
work.

HRX vendors selected IREE runtime sources and a small set of IREE runtime
dependencies as implementation libraries. Treat that as a build/input detail:
the public integration point for consumers is `hrx::hrx` and the installed HRX
headers.

> [!CAUTION]
> This release is an early-access software technology preview. Running
> production workloads is not recommended.

## Build And Test

Run from `sources/hrx`.

### Bolt-On Build Scripts

The bolt-on path bootstraps a shared ROCm root from TheRock nightly artifacts,
then installs the current HRX core scope into the same tree. By default this
root is `../build/rocm-root`, so future HRX components can share it as both
their dependency prefix and install prefix.

The artifact fetcher follows the public S3 artifact layout described in
`sources/TheRock/RELEASES.md` and mirrors a small, stable subset of
`BUILD_TOPOLOGY.toml`: sysdeps, base ROCm files, LLVM runtime bits, ROCr/HSA,
AMD SMI, and AQL profile. It intentionally does not fetch upstream HIP by
default because HRX core owns `libamdhip64.so` in the final tree.

```bash
python build_tools/fetch_rocm_artifacts.py \
  --release-type nightly --latest --set core

python build_tools/build_core.py
python build_tools/test_core.py
python build_tools/package_core.py
```

The package script emits a unified ROCm-style install tree and tarball under
`../build/hrx-core/dist/`. Source `env.sh` from the packaged tree to use it as
an alternate ROCm universe:

```bash
source ../build/hrx-core/dist/hrx-core/env.sh
hrx-info --device=cpu:0
```

To reproduce the Linux CI environment locally with TheRock's manylinux build
container:

```bash
python build_tools/linux_core_build.py
```

Requires: clang/clang++, ccache, lld, CMake 3.20+, Ninja, ROCm's
`hsa-runtime64` CMake package, and Catch2 v3 for CTS. GTest is only required
when `IREE_BUILD_TESTS=ON`.

```bash
cmake -S . -B build/hrx \
  -DCMAKE_PREFIX_PATH=/srv/vm-shared/projects/pyre-workspace/rocm \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja

cmake --build build/hrx
ctest --test-dir build/hrx --output-on-failure
```

Useful targets:

```bash
cmake --build build/hrx --target hrx
cmake --build build/hrx --target hrx-info
cmake --build build/hrx --target hip_intercept hip_logging hip_buffer_tracer
```

Quick smoke runs:

```bash
build/hrx/hrx-info
build/hrx/hrx-info --device=cpu:0
build/hrx/hrx-info --device=gpu:0
build/hrx/hrx-info --test=all
```

### CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `HRX_IREE_SOURCE_DIR` | `third_party/iree-runtime` | IREE source tree used as embedded runtime libraries. |
| `HRX_ENABLE_IREE_AMDGPU` | ON | Build GPU support through IREE's AMDGPU HAL driver. |
| `HRX_ENABLE_TRACY` | OFF | Enable IREE Tracy runtime tracing support. |
| `HRX_BUILD_CTS` | ON | Build Catch2 CTS binaries. |
| `IREE_BUILD_TESTS` | OFF | Build embedded IREE testing support used by HRX developer tests. Requires GTest. |
| `HRX_BUILD_PASSTHROUGH` | ON | Build HIP passthrough/interception libraries. |
| `HRX_ENABLE_ASAN` | OFF | Enable address sanitizer for HRX and propagate to IREE. |
| `HRX_ENABLE_MSAN` | OFF | Enable memory sanitizer for HRX and propagate to IREE. |
| `HRX_ENABLE_TSAN` | OFF | Enable thread sanitizer for HRX and propagate to IREE. |
| `HRX_ENABLE_UBSAN` | OFF | Enable undefined behavior sanitizer for HRX and propagate to IREE. |

## Consuming With CMake

`hrx` exports `hrx::hrx` from both the build tree and an install prefix.

```bash
# Build-tree package.
cmake -S cts/package_smoke \
  -B build/hrx-package-smoke \
  -Dhrx_DIR="$PWD/build/hrx/cmake/hrx"
cmake --build build/hrx-package-smoke

# Install-tree package.
cmake --install build/hrx --prefix build/hrx-install
cmake -S cts/package_smoke \
  -B build/hrx-package-smoke-install \
  -DCMAKE_PREFIX_PATH="$PWD/build/hrx-install"
cmake --build build/hrx-package-smoke-install

# If ROCm shared libraries are not in the default loader path:
LD_LIBRARY_PATH=/srv/vm-shared/projects/pyre-workspace/rocm/lib \
  ./build/hrx-package-smoke-install/hrx_package_smoke
```

## Runtime Concepts

- **Public ABI**: `hrx_*` symbols only; headers live under `include/`.
- **Backends**: CPU is local-task; GPU is AMDGPU when `HRX_ENABLE_IREE_AMDGPU`
  is enabled; remote HAL support is compiled through the embedded IREE runtime.
- **Explicit execution model**: devices own allocators; streams own timeline
  semaphores and pending command buffers; direct queue ops submit one-shot
  barriers, fills, copies, and dispatches.
- **Memory model**: buffers support explicit allocation/import, map/unmap,
  stream-ordered allocation, buffer views, pools, and host allocators.
- **Modules and dispatch**: VMFB data/files can be loaded as modules,
  functions can be looked up/invoked, and executables can be dispatched through
  queue or stream APIs.
- **HIP-facing layer**: `src/binding/hip` and `src/streaming` provide the
  stream, event, memory, module, graph capture, graph execution, symbol
  registry, and replay-oriented infrastructure used by higher-level API work.
- **Profiling and tracing**: HRX can emit IREE HAL profile files; passthrough
  libraries can log HIP calls and trace buffers around kernel launches.
- **Compiler integration**: HRX can use the IREE compiler dylib or an explicit
  `iree-compile` CLI fallback to produce VMFB artifacts. This is a convenience
  facility and may be re-evaluated later.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HRX_GPU_DRIVER` | `amdgpu` | Selects the GPU HAL driver name. Currently expects `amdgpu`. |
| `HRX_GPU_DEBUG` | unset | Prints IREE status details from GPU init/profiling when nonzero. |
| `HRX_PROFILE_FILE` | unset | Enables GPU HAL profile capture to the given path. |
| `HRX_PROFILE_MODE` | `queue` | Profile families: `queue`, `dispatch`, `executable`, or `all`. |
| `HRX_IREE_COMPILE` | `libIREECompiler.so` | Overrides the IREE compiler dylib path. |
| `HRX_IREE_COMPILER_CLI` | unset | Explicit `iree-compile` fallback path for compiler tests/debugging. |
| `HRX_LIBRARY` | test target path | CTS dlopen path for `libhrx.so`. Usually set by CTest. |
| `HIP_PASSTHROUGH_BACKEND_LIB` | required by passthrough | Real `libamdhip64.so` used by HIP passthrough libraries. |
| `HIP_PASSTHROUGH_INTERCEPTOR` | unset | Interceptor library for `src/passthrough/passthrough.c`. |
| `HIP_INTERCEPTION_LIBRARY` | unset | Legacy interceptor variable used by `hip_intercept.c`. |
| `HIP_LOG_FILE` | stderr | HIP passthrough/logging output path. |
| `HIP_LOG_LEVEL` | `2` | Logging level: `0` off, `1` errors, `2` calls, `3` verbose. |
| `HIP_TRACE_FILE` | stderr | Buffer tracer output path. |
| `HIP_TRACE_LEVEL` | `2` | Trace level: `0` off, `1` errors, `2` calls, `3` buffers, `4` verbose. |
| `HIP_TRACE_SYNC` | `0` | Synchronize before/after kernel launches when set to `1`. |
| `HIP_TRACE_DUMP` | `0` | Buffer dump mode: `1` hex bytes, `2` FNV-1a hash. |
| `HIP_TRACE_DUMP_MAX` | `1024` | Maximum bytes dumped per buffer. |
| `HIP_TRACE_KERNEL_FILTER` | all | Only trace kernels whose name contains this substring. |
| `HIP_TRACE_KERNEL_COUNT` | `0` | Limit traced kernel launches; `0` means unlimited. |
| `HIP_TRACE_KERNEL_FULL_DUMP` | unset | Colon-separated kernel-name filters for unlimited buffer dumps. |

Profile example:

```bash
HRX_PROFILE_FILE=/tmp/hrx.ireeprof \
HRX_PROFILE_MODE=queue \
  build/hrx/hrx-info --device=gpu:0
iree-profile summary /tmp/hrx.ireeprof
```

## Key Code Locations

| Path | Notes |
|------|-------|
| `include/hrx_runtime.h` | Public C runtime ABI. |
| `include/hrx_runtime_cxx.h` | C++ status/RAII helpers over retained HRX handles. |
| `include/hrx_compiler.h` | Compiler API for MLIR-to-VMFB output. |
| `src/libhrx/` | `libhrx.so` implementation: status, runtime, devices, memory, streams, queues, modules, compiler, fences, pools. |
| `src/streaming/` | HIP-oriented streaming substrate: context, devices, events, memory, graph capture/execution, module registry, peer, buffer table. |
| `src/binding/hip/` | HIP API surface implemented on the streaming substrate. |
| `src/passthrough/` | HIP passthrough, logging, and buffer tracing shared libraries. |
| `tools/hrx_info.c` | CLI for enumeration, smoke tests, and profile capture checks. |
| `cts/` | Catch2 CTS, optional compiler tests, and `find_package(hrx)` smoke consumer. |
| `scripts/` | IREE runtime importer and patch queue tools. |
| `third_party/iree-runtime/` | Generated vendored IREE runtime source tree. |

## Build Outputs

- `libhrx.so`: shared library with only `hrx_*` symbols exported by version
  script; selected IREE runtime libraries are linked into this implementation.
- `hrx-info`: CLI linked against the public HRX API.
- `cts/hrx_cts_*`: CTS binaries. Most dlopen `libhrx.so` through `HRX_LIBRARY`.
- `libhip_intercept.so`, `libhip_logging.so`, `libhip_noop.so`,
  `libhip_buffer_tracer.so`: passthrough/interception libraries when
  `HRX_BUILD_PASSTHROUGH=ON`.
- `build/hrx/cmake/hrx`: build-tree CMake package exporting `hrx::hrx`.

## Vendored IREE Runtime

HRX vendors a generated IREE runtime source tree under
`third_party/iree-runtime`. The tree is replayable: generated HRX metadata is
kept beside it in `third_party/iree-runtime.HRX_VENDOR.json`, and HRX-local IREE
changes live as patch files under `scripts/iree-runtime-patches`.

Current upstream IREE commit:

```text
6e34728ce15889222b73d0348932882f6fef54bd
```

The importer vendors selected IREE runtime paths plus `benchmark`, `flatcc`,
`tracy`, `mimalloc`, and `libbacktrace`. It intentionally does not vendor
`third_party/hsa-runtime-headers`; ROCm supplies HSA headers/libraries through
`find_package(hsa-runtime64)`.

### Normal Update Flow

Create the pristine import commit:

```bash
python scripts/vendor_iree_runtime.py \
  --iree-repo ../iree \
  --ref 6e34728ce15889222b73d0348932882f6fef54bd \
  import-pristine
```

Apply HRX-local IREE patches, one commit per patch:

```bash
python scripts/vendor_iree_runtime.py apply-patches
```

After editing `third_party/iree-runtime` in follow-up commits, refresh the patch
directory from the commits after the pristine import:

```bash
python scripts/vendor_iree_runtime.py \
  dump-patches --diffbase <pristine-import-commit>
```

Validate that the committed tree still matches the importer and patch queue:

```bash
python scripts/vendor_iree_runtime.py \
  --iree-repo ../iree \
  --ref 6e34728ce15889222b73d0348932882f6fef54bd \
  check
```

Use `update` and `diff` only for local validation. The normal history shape is:

```text
<HRX tooling/docs commit>
<pristine IREE import commit>
<one commit per IREE patch>
```
