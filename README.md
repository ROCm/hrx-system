# Pyre Runtime

Standalone shared library (`libpyre.so`) wrapping IREE's HAL for device
management, timeline synchronization, memory allocation, and command dispatch.

## Building

Requires: clang/clang++, ccache, lld, cmake 3.20+, ninja.

```bash
cmake -B build sources/pyre-runtime \
  -DPYRE_IREE_SOURCE_DIR=/path/to/iree \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja

cmake --build build
```

## Consuming with CMake

`pyre-runtime` exports `pyre::pyre` from both the build tree and an install
prefix.

```bash
# Build-tree package
cmake -S sources/pyre-runtime/cts/package_smoke \
  -B build/pyre-package-smoke \
  -Dpyre_DIR=build/cmake/pyre
cmake --build build/pyre-package-smoke

# Install-tree package
cmake --install build --prefix build/pyre-runtime-install
cmake -S sources/pyre-runtime/cts/package_smoke \
  -B build/pyre-package-smoke-install \
  -DCMAKE_PREFIX_PATH=build/pyre-runtime-install
cmake --build build/pyre-package-smoke-install

# If libhsa-runtime64.so.1 is not in the default loader path:
LD_LIBRARY_PATH=/path/to/rocm/lib ./build/pyre-package-smoke-install/pyre_package_smoke
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `PYRE_IREE_SOURCE_DIR` | (required) | Path to IREE source tree |
| `PYRE_ENABLE_TRACY` | OFF | Enable IREE Tracy tracing integration |
| `PYRE_BUILD_CTS` | ON | Build conformance test suite |

## Build Outputs

- `libpyre.so` — LTO'd shared library. Only `pyre_*` symbols exported (version
  script enforced). IREE is statically linked and hidden.
- `libpyre_internal.a` — Static library with no visibility hiding. For in-tree
  tools that need both pyre API and IREE internals.
- `pyre-info` — CLI tool. Links libpyre.so (public API) + IREE flags (static).
  Validates dual-linkage: two copies of IREE coexist without symbol conflicts.
- `cts/pyre_cts_*` — Conformance test binaries (Catch2 v3).

## Running

```bash
# Device enumeration
pyre-info

# Smoke test on specific device
pyre-info --device=cpu:0
pyre-info --device=gpu:0

# Full smoke test across all devices
pyre-info --test=all

# CTS
ctest --test-dir build --output-on-failure
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              libpyre.so (LTO'd)             │
│                                             │
│  pyre_* API  ←── only exported symbols      │
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
- **C++ helpers**: `pyre_runtime_cxx.h` provides status formatting and RAII
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

GPU init uses local-task driver (no HSA required). This validates the API
surface and build architecture. Real HSA/AMDGPU driver integration comes
in a later phase.

## Source Layout

```
include/pyre_runtime.h     Public C API
src/                        Implementation
  status.c                  Status API, IREE→pyre error mapping
  runtime.c                 Global state, accelerator init/shutdown
  device.c                  Device properties and sync
  semaphore.c               Timeline semaphore operations
  stream.c                  Stream + pending CB + stream ops
  buffer.c                  Allocation, mapping
  queue_ops.c               Direct queue fill/copy/barrier
tools/pyre_info.c           CLI tool
cmake/                      Build config, version script, find_package
cts/                        Conformance test suite (Catch2 v3)
  package_smoke/            Standalone find_package(pyre) smoke consumer
docs/                       SOURCE_PROVENANCE.md
```
