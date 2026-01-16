# IREE Streaming

A standalone streaming/graph execution library built on top of IREE's runtime.

## Overview

This project provides a CUDA/HIP-compatible streaming API that leverages IREE's
HAL (Hardware Abstraction Layer) for portable GPU execution. It implements graph
capture, analysis, and execution functionality similar to CUDA Graphs and HIP
Graphs.

## Prerequisites

- CMake 3.21 or later
- C11/C++17 compatible compiler
- Git (for submodules)

For CUDA support:
- CUDA Toolkit

For HIP/ROCm support:
- ROCm installation
- Set `IREE_ROCM_PATH` to your ROCm installation path

## Building

### Clone with Submodules

```bash
git clone --recursive https://github.com/your-org/iree-stream.git
cd iree-stream
```

Or if you've already cloned:

```bash
git submodule update --init --recursive
```

### Configure and Build

```bash
# Create build directory
mkdir build && cd build

# Configure (Release build)
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build .
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `IREE_STREAMING_BUILD_TESTS` | ON | Build tests and benchmarks |
| `IREE_STREAMING_BUILD_CUDA_BINDING` | ON | Build CUDA compatibility binding |
| `IREE_STREAMING_BUILD_HIP_BINDING` | ON | Build HIP compatibility binding |

### HAL Driver Options

The IREE HAL drivers can be configured via:

| Option | Default | Description |
|--------|---------|-------------|
| `IREE_HAL_DRIVER_CUDA` | ON | Enable CUDA HAL driver |
| `IREE_HAL_DRIVER_HIP` | ON | Enable HIP HAL driver |
| `IREE_HAL_DRIVER_LOCAL_SYNC` | ON | Enable local sync driver |
| `IREE_HAL_DRIVER_LOCAL_TASK` | ON | Enable local task driver |

### Example Build Commands

#### Minimal build (no GPU backends):

```bash
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_HAL_DRIVER_CUDA=OFF \
    -DIREE_HAL_DRIVER_HIP=OFF \
    -DIREE_STREAMING_BUILD_CUDA_BINDING=OFF \
    -DIREE_STREAMING_BUILD_HIP_BINDING=OFF
```

#### CUDA-only build:

```bash
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_HAL_DRIVER_HIP=OFF \
    -DIREE_STREAMING_BUILD_HIP_BINDING=OFF
```

#### HIP/ROCm build:

```bash
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_HAL_DRIVER_CUDA=OFF \
    -DIREE_STREAMING_BUILD_CUDA_BINDING=OFF \
    -DIREE_ROCM_PATH=/opt/rocm
```

## Project Structure

```
iree-stream/
├── CMakeLists.txt           # Top-level build configuration
├── README.md                # This file
├── streaming/               # Streaming library source
│   ├── internal.h          # Internal API and data structures
│   ├── *.c                 # Core implementation
│   ├── binding/            # API compatibility layers
│   │   ├── cuda/          # CUDA API binding
│   │   └── hip/           # HIP API binding
│   ├── test/              # Tests and demos
│   │   └── kernels/       # Test kernel sources
│   ├── util/              # Utility libraries
│   └── docs/              # Documentation
└── third_party/
    └── iree/              # IREE submodule
```

## Usage

### CUDA Compatibility API

```c
#include "streaming/binding/cuda/api.h"

// Initialize
cuInit(0);

// Create context and stream
CUcontext ctx;
cuCtxCreate(&ctx, 0, 0);

CUstream stream;
cuStreamCreate(&stream, 0);

// Use CUDA-like API...
```

### HIP Compatibility API

```c
#include "streaming/binding/hip/api.h"

// Initialize
hipInit(0);

// Create stream
hipStream_t stream;
hipStreamCreate(&stream);

// Use HIP-like API...
```

## Documentation

See the [docs](streaming/docs/) directory for detailed documentation:
- [Streaming API](streaming/docs/streaming_api.md)
- [Runtime Architecture](streaming/docs/runtime.md)

## License

Licensed under the Apache License v2.0 with LLVM Exceptions.
See [LICENSE](third_party/iree/LICENSE) for details.

