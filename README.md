# HRX Runtime

HRX (HIP Runtime Extended) is a standalone shared library (`libhrx.so`)
wrapping IREE's HAL for device management, timeline synchronization, memory
allocation, and command dispatch.

## Building

Requires: clang/clang++, ccache, lld, cmake 3.20+, ninja.

```bash
cmake -B build sources/hrx \
  -DHRX_IREE_SOURCE_DIR=/path/to/iree \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja

cmake --build build
```

## Consuming with CMake

`hrx` exports `hrx::hrx` from both the build tree and an install
prefix.

```bash
# Build-tree package
cmake -S sources/hrx/cts/package_smoke \
  -B build/hrx-package-smoke \
  -Dhrx_DIR=build/cmake/hrx
cmake --build build/hrx-package-smoke

# Install-tree package
cmake --install build --prefix build/hrx-install
cmake -S sources/hrx/cts/package_smoke \
  -B build/hrx-package-smoke-install \
  -DCMAKE_PREFIX_PATH=build/hrx-install
cmake --build build/hrx-package-smoke-install

# If ROCm shared libraries are not in the default loader path:
LD_LIBRARY_PATH=/path/to/rocm/lib ./build/hrx-package-smoke-install/hrx_package_smoke
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `HRX_IREE_SOURCE_DIR` | (required) | Path to IREE source tree |
| `HRX_ENABLE_TRACY` | OFF | Enable IREE Tracy tracing integration |
| `HRX_BUILD_CTS` | ON | Build conformance test suite |

## Build Outputs

- `libhrx.so` — LTO'd shared library. Only `hrx_*` symbols exported (version
  script enforced). IREE is statically linked and hidden.
- `libhrx_internal.a` — Static library with no visibility hiding. For in-tree
  tools that need both hrx API and IREE internals.
- `hrx-info` — CLI tool. Links libhrx.so (public API) + IREE flags (static).
  Validates dual-linkage: two copies of IREE coexist without symbol conflicts.
- `cts/hrx_cts_*` — Conformance test binaries (Catch2 v3).

## Running

```bash
# Device enumeration
hrx-info

# Smoke test on specific device
hrx-info --device=cpu:0
hrx-info --device=gpu:0

# Full smoke test across all devices
hrx-info --test=all

# CTS
ctest --test-dir build --output-on-failure
```

### Profiling Environment

```bash
# Capture an IREE HAL profile for GPU work.
HRX_PROFILE_FILE=/tmp/hrx.ireeprof hrx-info --device=gpu:0
iree-profile summary /tmp/hrx.ireeprof
```

| Variable | Default | Description |
|----------|---------|-------------|
| `HRX_PROFILE_FILE` | unset | Enables IREE HAL profiling for GPU devices and writes the profile to this path. |
| `HRX_PROFILE_MODE` | `queue` | Profiling mode: `queue`, `dispatch`, `executable`, or `all`. Use `queue` for normal llama.cpp runs on the current AMDGPU branch; `dispatch`/`all` exercise newer counter paths and may perturb some backend unit tests. |
| `HRX_GPU_DRIVER` | `amdgpu` | GPU HAL driver name. |
| `HRX_GPU_DEBUG` | unset | Print IREE status details from GPU initialization and profiling setup. |

## Architecture

```
┌─────────────────────────────────────────────┐
│              libhrx.so (LTO'd)             │
│                                             │
│  hrx_* API  ←── only exported symbols      │
│      │                                      │
│      ▼                                      │
│  IREE HAL/VM  (static, hidden)              │
│  local-task   (static, hidden)              │
│  local-sync   (static, hidden)              │
│                                             │
└─────────────────────────────────────────────┘
```

### API Surface

- **Status**: error codes, messages, ignore
- **C++ helpers**: `hrx_runtime_cxx.h` provides status formatting and RAII
  wrappers for all retained handles.
- **Accelerator namespaces**: GPU and CPU init/shutdown/enumerate independently
- **Device**: properties, sync, type query. Generic once obtained.
- **Semaphores**: timeline synchronization primitives (create/signal/wait/query)
- **Streams**: high-level execution contexts with pending command buffer
- **Buffers**: allocate, map/unmap, device pointer
- **Stream ops**: fill, copy, update (batched via pending CB)
- **Queue ops**: fill, copy, barrier (direct submission, no CB)
- **Retain/release**: all handle types are reference-counted

### Current State (Initial Spike)

GPU init uses IREE's AMDGPU HAL driver when HRX is built with
`HRX_ENABLE_IREE_AMDGPU=ON`. CPU init uses the local-task driver.

## Source Layout

```
include/hrx_runtime.h     Public C API
src/                        Implementation
  status.c                  Status API, IREE→hrx error mapping
  runtime.c                 Global state, accelerator init/shutdown
  device.c                  Device properties and sync
  semaphore.c               Timeline semaphore operations
  stream.c                  Stream + pending CB + stream ops
  buffer.c                  Allocation, mapping
  queue_ops.c               Direct queue fill/copy/barrier
tools/hrx_info.c           CLI tool
cmake/                      Build config, version script, find_package
cts/                        Conformance test suite (Catch2 v3)
  package_smoke/            Standalone find_package(hrx) smoke consumer
docs/                       SOURCE_PROVENANCE.md
```
