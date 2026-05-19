# Source Provenance

Documents the origin of all code in `sources/hrx-runtime/`. Required for
coordination with Andrew (iree-hal-streaming author).

## Vendored IREE Runtime

`third_party/iree-runtime/` starts as a generated, pruned import of IREE runtime
sources. The pristine import is committed first, and HRX-local IREE changes are
then layered as one commit per patch. Regenerate the pristine import with:

```bash
python scripts/vendor_iree_runtime.py \
  --iree-repo ../iree \
  --ref 05110733b50c2c0faafbe7452ab77f6e8088d33b \
  import-pristine
```

The exact upstream commit, copied paths, submodule commits, and external
dependencies are recorded beside the import in
`third_party/iree-runtime.HRX_VENDOR.json`. HRX-generated provenance stays out
of `third_party/iree-runtime/` so that tree remains a replayable import.
Patch files are dumped from the commits after the pristine import with
`vendor_iree_runtime.py dump-patches --diffbase <pristine-import-commit>`.

The initial snapshot is based on IREE commit
`05110733b50c2c0faafbe7452ab77f6e8088d33b`. It vendors `third_party/flatcc`
from IREE's recorded submodule commit and intentionally excludes
`third_party/hsa-runtime-headers`; HSA is provided by ROCm's
`hsa-runtime64` CMake package. HRX-local IREE changes belong in
`scripts/iree-runtime-patches/*.patch` so they remain replayable across future
imports.

## Files Written from Scratch

All files in this initial spike are **new code**, not copies or adaptations
of iree-hal-streaming. The API design and architecture are informed by the
design document (`docs/design/hrx_runtime_extraction.md`) and by studying
iree-hal-streaming's patterns, but no source code was copied.

### Public API

| File | Origin | Notes |
|------|--------|-------|
| `include/hrx_runtime.h` | New | C API from design doc. Opaque handles, status pattern from IREE conventions. |

### Implementation

| File | Origin | Notes |
|------|--------|-------|
| `src/hrx_internal.h` | New | Internal struct definitions. Struct layout influenced by iree-hal-streaming's `internal.h` (device registry, stream with timeline semaphore + pending CB, buffer with hal_buffer). |
| `src/status.c` | New | Status API. Follows IREE's NULL=OK pattern. `hrx_status_from_iree()` maps IREE status codes. |
| `src/runtime.c` | New | Global state + accelerator init. Device creation pattern adapted from PyTorch's `HrxRuntime::initialize()` (driver-based creation via `iree_hal_task_driver_create` + `iree_hal_driver_create_default_device`). |
| `src/device.c` | New | Device property queries and sync. |
| `src/semaphore.c` | New | Timeline semaphore wrapper. Follows IREE retain/release pattern (atomic ref count, `== 1` check for last release). |
| `src/stream.c` | New | Stream with pending command buffer. The stream-owns-semaphore + pending-CB pattern is inspired by iree-hal-streaming's `stream.c`, but the implementation is independent. |
| `src/buffer.c` | New | Buffer allocation via IREE HAL allocator. Memory type mapping to `iree_hal_buffer_params_t` is new. |
| `src/queue_ops.c` | New | Direct queue operations (fill, copy, barrier). Each creates a one-shot CB, submits with explicit semaphore lists. |

### Tools

| File | Origin | Notes |
|------|--------|-------|
| `tools/hrx_info.c` | New | CLI tool. Dual-links libhrx.so + IREE static (for `iree_flags_parse`). |

### CTS

| File | Origin | Notes |
|------|--------|-------|
| `cts/core/hrx_loader.hpp` | New, pattern from hip-cts | dlopen-based loader. Mirrors hip-cts `HipLoader` pattern (Meyers singleton, loadSymbol). |
| `cts/core/hrx_loader.cpp` | New, pattern from hip-cts | Symbol loading implementation. |
| `cts/core/hrx_test_fixture.hpp` | New, pattern from hip-cts | Catch2 fixture. Mirrors hip-cts `HipTestFixture`. |
| `cts/core/main.cpp` | New, pattern from hip-cts | Custom Catch2 main with `--hrx-library` and `--hrx-device` args. |
| `cts/tests/*/` | New | All test files are new. |
| `cts/third_party/Catch2/` | Symlink | Points to `hip-cts/third_party/Catch2` (same version). |

### Build

| File | Origin | Notes |
|------|--------|-------|
| `CMakeLists.txt` | New | IREE integration pattern adapted from `pytorch/cmake/public/hrx.cmake`. |
| `cmake/hrx_exports.lds` | New | Version script limiting exports to `hrx_*`. |
| `cmake/hrx-config.cmake.in` | New | find_package template. |
| `cts/CMakeLists.txt` | New | Standalone/in-tree dual-mode CTS build. |

## Patterns Adapted from iree-hal-streaming

The following architectural patterns were studied in iree-hal-streaming and
informed the hrx-runtime design, without copying code:

1. **Global device registry** — iree-hal-streaming uses a global
   `iree_hal_streaming_device_registry_t` singleton. HRX uses
   `hrx_gpu_state_t` / `hrx_cpu_state_t` statics with per-accelerator
   namespace isolation.

2. **Stream with timeline semaphore** — iree-hal-streaming's `stream.c`
   tracks `timeline_semaphore` + `command_buffer` + `recorded_events`.
   HRX's `hrx_stream_s` tracks `semaphore` + `pending_cb` + `timepoint`.

3. **Buffer table / pointer mapping** — iree-hal-streaming's
   `buffer_table.{h,c}` maps device pointers to HAL buffers. HRX stores
   the HAL buffer directly in `hrx_buffer_s` (no global table yet).

4. **Reference counting** — Both use `iree_atomic_ref_count_t` with
   retain/release semantics.

5. **Context / primary context** — iree-hal-streaming has per-device
   contexts with lazy initialization. HRX's devices are simpler (no
   context layer yet).

## What Was NOT Taken

- Graph execution (`graph.c`, `graph_exec.c`, `graph_analysis.c`)
- Module/symbol registry (`module.c`, `registry.c`)
- Memory pool management (`mem_pool.c`)
- HIP/CUDA binding layer (`binding/hip/`, `binding/cuda/`)
- Reference HSA driver (`hsa_driver/`); HRX now uses IREE's AMDGPU HAL driver
- Event system (`event.c`)
- Peer-to-peer (`peer.c`, P2P topology)
- Pointer tagging for symbol identification
- Thread-local context stack
