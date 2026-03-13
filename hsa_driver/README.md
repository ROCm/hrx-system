# HSA HAL Driver (Experimental)

> **⚠ Experimental — not intended for production use.**
>
> This driver is under active development and may have incomplete features,
> missing error handling, or performance characteristics that are not
> representative of a production-quality implementation.

## Overview

The HSA (Heterogeneous System Architecture) HAL driver is an IREE backend
that targets AMD GPUs directly through the HSA runtime API
(`libhsa-runtime64.so`), bypassing the HIP runtime entirely. It implements
the `iree_hal_driver_t` / `iree_hal_device_t` / `iree_hal_executable_t`
interfaces so that the IREE streaming layer can dispatch GPU kernels on
AMD hardware without any dependency on `libamdhip64.so`.

### Why a separate HSA driver?

The primary HIP backend (`third_party/iree/runtime/src/iree/hal/drivers/hip/`)
works well for IREE-compiled modules but relies on the HIP runtime for kernel
launches (`hipModuleLaunchKernel`). When the streaming layer intercepts HIP API
calls from frameworks like PyTorch, the HIP runtime is already loaded as a
shared library. However, certain scenarios benefit from a driver that talks
directly to HSA:

- **Reduced layering** — eliminates one level of indirection (HIP → HSA) for
  kernel dispatch, memory allocation, and synchronization.
- **Teardown ordering** — avoids atexit conflicts between the real HIP runtime
  and the streaming layer's intercepted state (see `.docs/direct-passthrough.md`
  for details on the `hipModuleUnload` crash during shutdown).
- **Lower overhead** — HSA AQL packet dispatch is a single memory-mapped write
  to a hardware queue; there is no driver-side marshalling beyond filling the
  packet and ringing the doorbell.

## Architecture

```
┌──────────────────────────────────────┐
│          Streaming Layer             │
│  (intercepts hipLaunchKernel, etc.)  │
└──────────────┬───────────────────────┘
               │  iree_hal_command_buffer_dispatch()
               ▼
┌──────────────────────────────────────┐
│          HSA HAL Driver              │
│  ┌────────────────────────────────┐  │
│  │  hsa_driver.c     — driver    │  │
│  │  hsa_device.c     — device    │  │
│  │  hsa_allocator.c  — allocator │  │
│  │  hsa_buffer.c     — buffers   │  │
│  │  native_executable.c          │  │
│  │  stream_command_buffer.c      │  │
│  │  hsa_semaphore.c              │  │
│  └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
               │  hsa_* API calls
               ▼
┌──────────────────────────────────────┐
│       libhsa-runtime64.so           │
│     (AMD ROCr runtime library)      │
└──────────────────────────────────────┘
```

## Components

### Driver (`hsa_driver.c`)

Entry point for the HAL driver. Initialises the HSA runtime via `hsa_init()`,
enumerates GPU and CPU agents, and creates `iree_hal_hsa_device_t` instances.
The HSA runtime symbols are loaded dynamically from `libhsa-runtime64.so`
through `iree_hal_hsa_dynamic_symbols_t`.

### Device (`hsa_device.c`)

Implements `iree_hal_device_t`. On creation it:

1. Discovers memory pools on the GPU agent (device-local / coarse-grained,
   host-visible / fine-grained, and kernarg pools).
2. Creates an HSA hardware queue (`hsa_queue_create`) for AQL packet dispatch.
3. Creates a completion signal for synchronisation.
4. Creates the device memory allocator.

The device exposes standard HAL queries (total memory, wavefront size, compute
unit count, architecture name) and supports host-to-device / device-to-host
transfers via `hsa_memory_copy`.

Command buffer execution follows the deferred-replay pattern: commands are
recorded into a `iree_hal_deferred_command_buffer_t`, then replayed into a
`iree_hal_hsa_stream_command_buffer_t` at queue-execute time.

### Memory Allocator (`hsa_allocator.c`) and Buffers (`hsa_buffer.c`)

Wraps `hsa_amd_memory_pool_allocate` / `hsa_amd_memory_pool_free` for
device-local and host-visible allocations. Buffers track both a device pointer
and an optional host pointer, and support mapping for persistent host access
when the memory type allows it.

### Native Executable (`native_executable.c`)

Loads GPU executables in one of two formats:

- **`rocm-hsaco-fb`** — IREE's flatbuffer-wrapped HSACO format (produced by
  the IREE compiler). Contains one or more code objects plus export metadata
  (binding counts, constant counts, block dimensions).
- **Native fat binaries** — Clang offload bundles (`__CLANG_OFFLOAD_BUNDLE__`)
  or raw ELF HSACO files produced by the HIP/ROCm toolchain. These are the
  format used by PyTorch, Triton, rocBLAS, etc.

For native fat binaries the driver:

1. Parses the bundle to extract the ELF matching the device's ISA triple
   (e.g. `hip-amdgcn-amd-amdhsa--gfx942`).
2. Parses ELF symbol tables and AMDGPU metadata notes to extract kernel names,
   parameter layouts (offsets, sizes, types), and hidden-argument offsets.
3. Loads the code object via `hsa_code_object_reader_create_from_memory` →
   `hsa_executable_load_agent_code_object` → `hsa_executable_freeze`.
4. Resolves kernel symbols to obtain `kernel_object` handles, kernarg segment
   sizes, and group/private segment sizes.

### Stream Command Buffer (`stream_command_buffer.c`)

Implements `iree_hal_command_buffer_t` for direct kernel dispatch. The
dispatch path:

1. Allocates a kernarg buffer from the kernarg memory pool.
2. Populates explicit kernel arguments:
   - **With parameter metadata** — iterates the parameter list, writing
     bindings (resolved device pointers) and constants at their ABI offsets.
   - **CUSTOM_DIRECT_ARGUMENTS** — copies the pre-packed constants buffer
     directly into the kernarg region.
   - **Without metadata** — falls back to sequential binding/constant layout
     (used for IREE-compiled kernels).
3. Fills in COv5 implicit (hidden) arguments (block counts, group sizes,
   grid dimensions) at the implicit-args offset.
4. Constructs an `hsa_kernel_dispatch_packet_t` with workgroup sizes, grid
   sizes, and the kernarg address.
5. Writes the packet to the hardware queue and rings the doorbell signal.
6. Optionally waits for completion via `hsa_signal_wait_scacquire`.

Memory operations (fill, copy, update) are implemented with
`hsa_amd_memory_fill`, `hsa_amd_memory_async_copy`, and `hsa_memory_copy`
respectively.

### Semaphores (`hsa_semaphore.c`)

Host-only timeline semaphores backed by `iree_notification_t`. These provide
ordering between command buffer submissions but do not interact with GPU
signals. Multi-wait and multi-signal operations are supported.

### Fat Binary Parser (`fat_binary.c`, `fat_binary.h`)

Standalone parser for AMD GPU fat binaries. Handles:

- **Compressed bundles** (`CCOB` magic) — header parsing is supported but
  decompression is not yet implemented.
- **Uncompressed bundles** (`__CLANG_OFFLOAD_BUNDLE__`) — fully supported,
  extracts ELF binaries by target triple.
- **Raw ELF** — direct HSACO files.

Within each ELF the parser extracts:

- Kernel function symbols (from `.symtab` / `.dynsym`).
- AMDGPU metadata notes (`.note` sections) containing parameter types, sizes,
  offsets, and hidden-argument information.
- Kernel descriptors for kernarg size, group segment size, etc.

### Dynamic Symbols (`dynamic_symbols.c`, `dynamic_symbol_tables.h`)

Loads `libhsa-runtime64.so` at runtime and resolves all required HSA API
function pointers. Required symbols include core HSA operations (init,
agents, queues, signals, memory, executables) and AMD extensions (memory
pools, async copy, profiling).

## Limitations

This driver is experimental. Known limitations include:

- **No async queue operations** — `queue_alloca`, `queue_dealloca`,
  `queue_read`, and `queue_write` are not implemented.
- **No collective channels** — `create_channel` returns `UNIMPLEMENTED`.
- **No events** — `create_event` returns `UNIMPLEMENTED`.
- **Host-only semaphores** — semaphore waits and signals happen on the host;
  there is no GPU-side timeline semaphore integration.
- **Single-queue execution** — the device creates one hardware queue; there
  is no multi-queue scheduling.
- **No compressed bundle decompression** — fat binaries compressed with
  zstd/zlib require decompression support that is not yet linked.
- **No profiling integration** — `profiling_begin`/`end`/`flush` are no-ops.

## File Reference

| File | Purpose |
|---|---|
| `api.h` | Public API: driver/device option structs and creation functions |
| `hsa_driver.c` | Driver implementation (init, agent enumeration, device creation) |
| `hsa_device.c` | Device implementation (pools, queues, allocator, HAL vtable) |
| `hsa_allocator.c` | Memory allocator (pool-based alloc/free) |
| `hsa_buffer.c` | Buffer wrapper (device + host pointers, mapping) |
| `native_executable.c` | Executable loading (HSACO-FB + native fat binaries) |
| `stream_command_buffer.c` | Command buffer (dispatch, memory ops, AQL packets) |
| `hsa_semaphore.c` | Host-only timeline semaphores |
| `fat_binary.c/h` | Fat binary and ELF parser |
| `dynamic_symbols.c/h` | Dynamic HSA symbol loading |
| `dynamic_symbol_tables.h` | Table of required/optional HSA API symbols |
| `per_device_information.h` | Per-device state (agent, pools, queue, signal) |
| `native_executable.h` | Kernel params struct and executable API |
| `native_executable_hsaf.h` | ELF/fat-binary format constants and structs |
| `status_util.c/h` | HSA status → IREE status conversion helpers |
| `hsa_headers.h` | Consolidated HSA header includes |
| `nop_executable_cache.c/h` | Pass-through executable cache (no caching) |
