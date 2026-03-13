# IREE Streaming

> **⚠ Experimental — not intended for production use.**
>
> This project is under active development. APIs, architecture, and
> functionality may change without notice. Features may be incomplete and
> error handling may be insufficient for production workloads.

## Overview

IREE Streaming is a collection of libraries for intercepting, re-targeting,
and instrumenting GPU compute workloads on AMD hardware. It has three main
components:

1. **Streaming Library** (`streaming/`) — A CUDA/HIP-compatible driver API
   implementation built on top of [IREE](https://github.com/iree-org/iree)'s
   HAL (Hardware Abstraction Layer). Applications (or frameworks like PyTorch)
   that call CUDA or HIP driver APIs can be redirected through this library
   to run on any IREE-supported backend — including AMD GPUs via HSA, NVIDIA
   GPUs via CUDA, Vulkan devices, or even CPUs — without source changes.

2. **HIP Passthrough Library** (`passthrough/`) — A drop-in replacement for
   `libamdhip64.so` that transparently forwards all HIP calls to the real HIP
   runtime. Pluggable interceptor libraries can be loaded at runtime to log
   API calls, trace memory allocations, dump buffer contents around kernel
   launches, or implement custom instrumentation — all without modifying the
   target application.

3. **HSA HAL Driver** (`hsa_driver/`) — An IREE HAL backend that targets AMD
   GPUs directly through the HSA runtime API (`libhsa-runtime64.so`),
   bypassing the HIP runtime entirely. This reduces layering overhead and
   avoids teardown-ordering issues that arise when the streaming library
   intercepts HIP calls in the same process as the real HIP runtime.

Together these components enable workflows such as:

- **Re-targeting** — Run a PyTorch model's HIP kernels through IREE on
  different hardware backends without recompiling.
- **Graph capture and replay** — Capture a sequence of GPU operations as a
  graph, analyse dependencies, and replay the graph efficiently.
- **API tracing and debugging** — Log every HIP call, track allocations, and
  dump buffer contents before/after kernel launches to diagnose correctness
  or performance issues.
- **Low-overhead dispatch** — Bypass the HIP runtime and dispatch kernels
  directly to AMD GPU hardware queues via HSA AQL packets.

## Project Structure

```
iree-stream/
├── CMakeLists.txt             # Top-level build configuration
├── README.md                  # This file
├── streaming/                 # Streaming library (CUDA/HIP API on IREE HAL)
│   ├── internal.h             # Internal types and data structures
│   ├── context.c              # Context management
│   ├── device.c               # Device enumeration and properties
│   ├── memory.c               # Memory allocation and transfers
│   ├── module.c               # Kernel module loading (fat binaries, HSACO, PTX)
│   ├── stream.c               # Stream/queue management
│   ├── event.c                # Synchronisation events
│   ├── graph.c                # Graph capture and construction
│   ├── graph_analysis.c       # Graph dependency analysis
│   ├── graph_exec.c           # Graph instantiation and replay
│   ├── mem_pool.c             # Stream-ordered memory pool
│   ├── peer.c                 # Peer-to-peer access
│   ├── init.c                 # Global initialisation
│   ├── registry.c             # Device/driver registry
│   ├── binding/
│   │   ├── cuda/              # CUDA Driver API compatibility layer
│   │   └── hip/               # HIP Driver API compatibility layer
│   ├── test/                  # Tests and demos
│   └── docs/                  # Detailed documentation
├── passthrough/               # HIP passthrough interception library
│   ├── hip_intercept.c        # Full-API passthrough with interceptor plugin support
│   ├── hip_function_table.h   # Function table and interceptor interface
│   ├── hip_logging.c          # Logging interceptor
│   ├── hip_buffer_tracer.c    # Buffer-tracing interceptor
│   └── interceptors/          # Example/template interceptors
├── hsa_driver/                # IREE HAL driver for direct HSA dispatch
│   ├── hsa_driver.c           # Driver init, agent enumeration
│   ├── hsa_device.c           # Device: pools, queues, allocator
│   ├── hsa_allocator.c        # HSA memory pool allocator
│   ├── native_executable.c    # HSACO / fat binary loader & kernel resolver
│   ├── stream_command_buffer.c # AQL packet dispatch
│   ├── fat_binary.c           # Clang offload bundle & ELF parser
│   └── dynamic_symbols.c      # Runtime HSA symbol loading
└── third_party/
    ├── iree/                  # IREE submodule (runtime only)
    └── hip-cts/               # HIP conformance test suite
```

## Prerequisites

- CMake 3.21 or later
- C11/C++17 compatible compiler
- Git (for submodules)

For HIP/ROCm support:
- ROCm installation (tested with ROCm 6.x)
- `libhsa-runtime64.so` for the HSA driver

For CUDA support:
- CUDA Toolkit

## Building

### Clone with Submodules

```bash
git clone --recursive <repo-url>
cd iree-stream
```

Or if you've already cloned:

```bash
git submodule update --init --recursive
```

### Configure and Build

```bash
mkdir build && cd build

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release

cmake --build .
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `IREE_STREAMING_BUILD_TESTS` | ON | Build tests and benchmarks |
| `IREE_STREAMING_BUILD_CUDA_BINDING` | ON | Build CUDA compatibility binding |
| `IREE_STREAMING_BUILD_HIP_BINDING` | ON | Build HIP compatibility binding |
| `IREE_STREAMING_BUILD_HSA_DRIVER` | ON | Build the external HSA HAL driver |

### HAL Driver Options

The IREE HAL drivers can be configured via:

| Option | Default | Description |
|--------|---------|-------------|
| `IREE_HAL_DRIVER_CUDA` | ON | Enable CUDA HAL driver |
| `IREE_HAL_DRIVER_HIP` | ON | Enable HIP HAL driver |
| `IREE_HAL_DRIVER_LOCAL_SYNC` | ON | Enable local sync driver |
| `IREE_HAL_DRIVER_LOCAL_TASK` | ON | Enable local task driver |

### Example Build Commands

#### HIP/ROCm build (most common):

```bash
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_HAL_DRIVER_CUDA=OFF \
    -DIREE_STREAMING_BUILD_CUDA_BINDING=OFF \
    -DIREE_ROCM_PATH=/opt/rocm
```

#### CUDA-only build:

```bash
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_HAL_DRIVER_HIP=OFF \
    -DIREE_STREAMING_BUILD_HIP_BINDING=OFF \
    -DIREE_STREAMING_BUILD_HSA_DRIVER=OFF
```

#### Minimal build (no GPU backends):

```bash
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_HAL_DRIVER_CUDA=OFF \
    -DIREE_HAL_DRIVER_HIP=OFF \
    -DIREE_STREAMING_BUILD_CUDA_BINDING=OFF \
    -DIREE_STREAMING_BUILD_HIP_BINDING=OFF \
    -DIREE_STREAMING_BUILD_HSA_DRIVER=OFF
```

## Component Details

### Streaming Library

The streaming library implements a large subset of the CUDA and HIP driver
APIs on top of IREE's HAL. It supports:

- **Device management** — enumeration, properties, peer-to-peer access.
- **Context and stream management** — context stacks, stream creation with
  priorities, host callbacks.
- **Memory management** — device/host/managed allocation, host registration,
  pitched allocation, memset variants (D8/D16/D32), async transfers.
- **Module loading** — load pre-compiled GPU binaries (HSACO fat binaries,
  PTX, SPIR-V) and resolve kernel functions and global variables.
- **Kernel launch** — `cuLaunchKernel` / `hipModuleLaunchKernel` with full
  grid/block/shared-memory configuration.
- **Events** — creation, recording, synchronisation, elapsed-time queries.
- **Graph capture and replay** — `cuStreamBeginCapture` / `hipStreamBeginCapture`,
  explicit graph construction with kernel/memcpy/memset/host nodes,
  instantiation, and launch.
- **Occupancy queries** — max active blocks, potential block size.
- **Memory pools** — stream-ordered allocation.

See [`streaming/README.md`](streaming/README.md) for the full API support
matrix and usage examples.

### HIP Passthrough Library

The passthrough library is a transparent proxy for `libamdhip64.so`. It
dynamically loads the real HIP library and forwards every call through a
function table. An optional interceptor plugin can wrap any subset of
functions to add logging, tracing, or custom behaviour.

Included interceptors:
- **`libhip_logging.so`** — timestamped logging of all HIP API calls.
- **`libhip_buffer_tracer.so`** — tracks allocations and dumps buffer
  contents (hex or FNV-1a hash) before/after kernel launches.
- **`libhip_noop.so`** — no-op passthrough, useful as a template.

See [`passthrough/README.md`](passthrough/README.md) for setup instructions,
environment variables, and how to write a custom interceptor.

### HSA HAL Driver

The HSA driver is an IREE HAL backend that dispatches GPU kernels directly
via HSA AQL packets, bypassing the HIP runtime. It parses native fat binaries
(Clang offload bundles) and ELF HSACO files to extract kernel metadata
and parameter layouts, then dispatches work by writing AQL packets to
hardware queues.

See [`hsa_driver/README.md`](hsa_driver/README.md) for architecture details,
component descriptions, and known limitations.

## Documentation

- [Streaming API reference and support matrix](streaming/README.md)
- [HIP Passthrough setup and interceptor guide](passthrough/README.md)
- [HSA HAL Driver architecture](hsa_driver/README.md)
- [Streaming internal docs](streaming/docs/)

## License

Licensed under the Apache License v2.0 with LLVM Exceptions.
See [LICENSE](LICENSE) for details.
