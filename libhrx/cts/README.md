# HRX Conformance Test Suite

The CTS exercises the public HRX library and HIP frontend through their shared
library entry points. HRX tests use the Catch2 harness in `core/`; HIP tests load
the exact built frontend DSO and use GoogleTest. Kernel fixtures are compiled for
the AMD GPU targets enabled in the build and selected against the actual device.

## Coverage

| Surface | Tests | Behavior exercised |
|---------|-------|--------------------|
| Runtime and devices | `lifecycle`, `status`, `device`, `group` | Explicit runtime/group ownership, passive discovery, shared CPU/GPU pools, device properties, and synchronization |
| Allocation and ownership | `allocator`, `host_allocator`, `memory`, `refcount` | Borrowed allocators, host imports, stream-ordered allocation, mapping, and retain/release |
| Queues and streams | `queue_ops`, `stream`, `stream_ops`, `semaphore` | Host calls, fills, copies, updates, execution barriers, timeline signals, and completion |
| Transfers | `transfer` | Host/device copies, stream ordering, numerical output, and range rejection |
| Executables | `executable` | Loading configured HSACO, function lookup, dispatch, and exact output |
| Virtual memory | `virtual_memory`, `virtual_memory_gpu` | Scope-qualified capabilities, batched mapping, aliases, permissions, and backing ownership |
| Legacy virtual memory ABI | `legacy_virtual_memory`, `legacy_virtual_memory_gpu` | Scalar reservation/physical-handle operations and explicit unsupported behavior on CPU |
| C++ bindings | `cxx_api` | Public C++ wrapper ownership and calls |
| HIP devices | `hip/device_selection_test.cpp`, `hip/occupancy_test.cpp` | Compatibility visibility, selected device identity, and launch occupancy |
| HIP modules and graphs | `hip/module_execution_test.cpp`, `hip/module_library_execution_test.cpp` | Module/library ownership, arguments, globals, captured execution, and dependencies |
| HIP progress | `hip/stream_progress_test.cpp` | Consumer completion after ordinary producer operations, without a later producer flush |
| HIP events | `hip/event_test.cpp`, `hip/event_timing_test.cpp` | Event ownership, stream ordering, native timestamps, and elapsed-time queries |
| HIP profiling | `hip/profile_test.cpp` | A finalized dispatch capture containing direct and graph dispatches with valid timestamps and no dropped records |

Additional public HIP API tests live beside the adapter under
[`src/binding/hip`](../src/binding/hip/BUILD.bazel), including arrays, registered
memory, graph mutation and ownership, memory pools, virtual memory, cooperative
launches, execution resources, and stream-value operations. The common adapter's
CPU tests exercise the same graph and stream machinery through the task HAL.

Coverage depends on the selected backend and its reported capabilities. A skipped
case does not qualify that feature. Multi-device cases require multiple visible
compatible GPUs; a single-device run cannot establish their behavior. Fork safety
after runtime initialization has no CTS witness. Kernel-initiated HIP
hostcall/printf is not implemented by the native GPU HAL; its explicit tests are
separate from ordinary dispatch qualification.

## Selecting the runtime and device

The shared HRX harness owns one explicit runtime and device group for its test
process. `--hrx-device` accepts a HAL URI, such as `task://`, `rdna://0`, or
`xdna://0`. `HRX_CTS_DEVICE` supplies the same selection when the argument is
absent. The harness also accepts `cpu:N` and `gpu:N` as ordinal-selection
shorthands and defaults to `cpu:0`. Tests that own their complete group lifecycle,
such as the mixed CPU/GPU pool test, use `none` for the harness device.

Native applications discover URI/name metadata with
`hrx_runtime_query_available_devices`, then select their devices through
`hrx_device_group_create`. Discovery does not create HAL devices or task workers.
Groups retain their runtime, and runtime events remain observable through final
synchronous group destruction. There is no global CPU or GPU initialization API.

HIP tests select GPUs through the frontend and do not use `HRX_CTS_DEVICE`.
`HRX_GPU_DRIVER` selects the frontend implementation and the harness's `gpu:N`
namespace; exact native HRX URIs select their driver directly:

| Value | Implementation |
|-------|----------------|
| `amdgpu` | Default: native AMD GPU HAL, with RDNA and CDNA enumeration |
| `rdna` / `cdna` | Native AMD GPU HAL, restricted to that architecture family |
| `legacy-amdgpu` | Explicit opt-in to the legacy ROCr-based HAL |

The requested driver must be compiled into the library. Backend selection is
explicit in comparisons so that a default change cannot silently change the
measured implementation. Compatibility visibility variables are processed before
the frontend creates its device group.

Bazel targets supply `--hrx-library` or `HRX_TEST_LIBAMDHIP64` for the exact library
under test. A manually staged HIP executable needs `HRX_TEST_LIBAMDHIP64` set to
that same build's frontend DSO; an ambient system HIP library is a different
implementation.

## Initialization diagnostics and process concurrency

Failed `hipInit` calls and implicit initialization through APIs such as
`hipGetDeviceCount` print the underlying runtime status to stderr, including the
failing operation and native error when available. `hipGetErrorString` continues
to describe the HIP error enum, so `hipErrorNotInitialized` still maps to
`initialization error`. Detailed failures do not require `HRX_GPU_DEBUG`.
Successful initialization and queries with all devices explicitly hidden remain
quiet.

HIP initialization creates execution resources for every selected GPU, even when
the caller only requests a device count. Native HRX discovery is passive and
creates execution resources only when the caller constructs a group. On the
native AMD GPU backend this includes an SDMA transfer queue where the device
exposes that queue family. These queues
come from a finite pool shared with other processes, with some slots reserved by
the kernel. Consequently there is no portable fixed process-concurrency limit:
the selected devices, driver, and other users determine available capacity.
CPU thread and file-descriptor limits do not establish GPU queue capacity.

For example, `gpu.user_queue_create(SDMA)` with an ERRNO status of `0x0000000C`
is `ENOMEM`. A matching kernel message about having no more SDMA queues identifies
queue-slot exhaustion; the errno alone does not establish system RAM exhaustion.
Releasing initialized processes makes their queue slots available again. Retrying
inside a failed process also depends on the native connection lifetime: KFD's
process-lifetime VM binding can survive destruction of the runtime's driver,
causing a new connection to report `EBUSY`. The native lifetime contract is
documented in [libamdf's memory fabric guide](../../libamdf/docs/memory.md#fabric-discovery).

## Running focused checks

From a configured repository root:

```bash
# Public HRX allocation and stream behavior on CPU.
iree-bazel-test --config=asan --test_env=HRX_CTS_DEVICE=cpu:0 \
  //libhrx/cts/tests/allocator:hrx_cts_allocator \
  //libhrx/cts/tests/stream_ops:hrx_cts_stream_ops

# The same public HRX paths on the native AMD GPU HAL.
iree-bazel-test --config=asan --test_env=HRX_CTS_DEVICE=gpu:0 \
  --test_env=HRX_GPU_DRIVER=amdgpu \
  //libhrx/cts/tests/allocator:hrx_cts_allocator \
  //libhrx/cts/tests/stream_ops:hrx_cts_stream_ops \
  //libhrx/cts/tests/executable:hrx_cts_executable

# Public HIP dispatch, independent stream progress, and dispatch capture.
iree-bazel-test --config=asan --test_env=HRX_GPU_DRIVER=amdgpu \
  //libhrx/cts/tests/hip:hip_module_execution_test \
  //libhrx/cts/tests/hip:hip_stream_progress_test \
  //libhrx/cts/tests/hip:hip_profile_test
```

GPU targets declare their hardware requirements and need a compatible runner.
The HIP DSO CTS targets currently select Linux; portable adapter tests and public
HRX tests also cover Windows. With CMake, `LIBHRX_BUILD_CTS=ON` includes the CTS
in the configured build, and CTest discovers its tests.

## Submission measurements

`//libhrx/cts/tests/hip:hip_dispatch_benchmark` measures public HIP direct launches
and captured graphs with a reused runtime, module, and stream. Its output labels
identify the selected driver, architecture, PCI device, and wait policy.

`Enqueue` excludes the completion wait; `EndToEnd` includes it. `EventInterval`
reports GPU timestamps around a graph, with encompassing host time in `host_ns`.
The zero-dispatch interval measures the event pair itself. These intervals include
timestamp-operation overhead and are distinct from ordinary completion latency.
Every nonempty row checks actual kernel output.

Performance runs use the canonical `--config=opt` build on a qualified, exclusive
runner. Fixed iteration counts and repetitions bound hidden completion work in
enqueue-only rows. `--hip_wait_mode=auto|spin|yield|blocking` selects the public HIP
scheduling policy; matching this policy, device identity, CPU locality, and power
settings is necessary when comparing backends. ASAN runs qualify correctness,
not performance.

## Download measurements

`//libhrx/cts/tests/hip:hip_transfer_benchmark` compares device-to-host copies into
ordinary CPU memory and the same kind of allocation registered with
`hipHostRegister`. Each iteration includes the public copy call and stream
completion. Contiguous and pitched rows cover 4 KiB, 1 MiB, and 64 MiB of useful
data; pitched rows have 4 KiB payloads and distinct source and destination
padding. The reported bandwidth counts payload bytes. Allocation, registration,
source initialization, and full output/padding validation stay outside timing.

All rows reuse one frontend, device, and stream. The benchmark has the same
backend labels and `--hip_wait_mode` choices as the submission benchmark. These
serial measurements establish copy latency and bandwidth; they do not establish
progress or interference between concurrent streams.
