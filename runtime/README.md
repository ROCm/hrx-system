# Shared runtime libraries

`runtime/` contains the execution libraries used by Loom, libhrx, and applications
that integrate them directly. It provides device execution, memory management,
CPU scheduling, asynchronous I/O, and a VM for host programs. These facilities
are reusable independently of the compiler, model services, and HTTP server.

The runtime is derived from [IREE](https://github.com/iree-org/iree) and retains
the `iree/` source layout and API names. The implementation here evolves with
the rest of this repository; use its headers and tests for the current contracts.

## Components

| Component | Responsibility | Entry point |
| --- | --- | --- |
| [HAL](src/iree/hal/README.md) | Device discovery, executable loading, memory, queues, and explicit synchronization | [API](src/iree/hal/api.h), [drivers](src/iree/hal/driver/) |
| [VM](src/iree/vm/README.md) | Link bytecode and native modules, retain program state, and invoke host programs | [Program](src/iree/vm/program.h), [process](src/iree/vm/process.h), [invocation](src/iree/vm/invocation.h) |
| [Task system](src/iree/task/README.md) | Shared CPU workers and cooperative progress for compute and runtime work | [API](src/iree/task/api.h) |
| [Asynchronous I/O](src/iree/async/README.md) | Completion-based file, socket, timer, and notification operations | [API](src/iree/async/api.h) |
| [Networking](src/iree/net/) | Transports, connections, message endpoints, and sessions over asynchronous I/O | [API](src/iree/net/api.h) |
| [Storage](src/iree/io/) | File handles, parameter indexes, archives, and parameter access | [File handles](src/iree/io/file_handle.h), [parameter indexes](src/iree/io/parameter_index.h) |
| [Tokenizer](src/iree/tokenizer/README.md) | Streaming text tokenization and model-format loading | [API](src/iree/tokenizer/tokenizer.h) |
| [Base](src/iree/base/) | Allocators, status, threading, data structures, and platform utilities | [API](src/iree/base/api.h) |

Each library keeps its tests beside its implementation. Public headers describe
ownership, threading, and failure behavior; component guides explain the larger
execution model.

## Device execution

The HAL gives callers explicit devices, memory resources, queues, and dependency
edges. Native AMD GPU and XDNA implementations share
[libamdf](../libamdf/README.md) for device access and memory backing. Their
[ownership guide](src/iree/hal/driver/amd/README.md) separates that foundation
from executable loading, command construction, and completion.

The [task driver](src/iree/hal/driver/task/) executes native CPU kernels, and
the [Vulkan driver](src/iree/hal/driver/vulkan/) executes SPIR-V through the
system Vulkan implementation. The
[driver configuration guide](../BUILDING.md#runtime-libraries-and-drivers)
describes build selection and runtime device names.

Loom adds [program preparation and model execution facilities](../loom/src/loom/runtime/)
above these libraries. Its [services](../loom/src/loom/serve/README.md) supply
modality behavior and HTTP integration. [libhrx](../libhrx/README.md) is another
consumer, exposing native HRX and HIP-compatible APIs.

## Build, test, and integrate

Start with the [repository build guide](../BUILDING.md) for source setup,
target selection, and build-system embedding. Component BUILD files name the
individual libraries and tests. From a configured repository root:

```bash
python dev.py bazel test --config=asan //runtime/src/iree/base/...
```

The [HAL conformance tests](src/iree/hal/cts/) exercise device behavior through
the shared interface. Hardware tests require the selected driver and a compatible
execution device. [Contributing](../CONTRIBUTING.md) covers development checks;
the [tracing guide](src/iree/base/tracing/README.md) covers runtime profiling.
