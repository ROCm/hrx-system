# HRX Conformance Test Suite

The CTS exercises the public HRX library and HIP frontend through their shared
library entry points. HRX tests use the Catch2 harness in `core/`; HIP tests load
the exact built frontend DSO and use GoogleTest. Kernel fixtures are compiled for
the AMD GPU targets enabled in the build and selected against the actual device.

## Coverage

| Surface | Tests | Behavior exercised |
|---------|-------|--------------------|
| Runtime and devices | `lifecycle`, `status`, `device`, `group` | Explicit runtime/group ownership, passive discovery, shared CPU/GPU pools, device properties, and synchronization |
| Heterogeneous execution | `heterogeneous` | Shared CPU/GPU/NPU storage, NPU arithmetic, cross-device queue dependencies, deferred allocation, backing reuse, and trim |
| Allocation and ownership | `allocator`, `host_allocator`, `memory`, `refcount` | Borrowed allocators, host imports, stream-ordered allocation, mapping, and retain/release |
| Prepared transitions | `group`, `heterogeneous`, `memory_transition_test` | Borrowed pool/buffer contracts, scope-pair qualification, prepared queue barriers, and explicit host maintenance across a GPU copy with mappings held open |
| Queues and streams | `queue_ops`, `stream`, `stream_ops`, `semaphore` | Host calls, fills, copies, updates, execution and visibility barriers, timeline signals, and completion |
| Transfers | `transfer` | Host/device copies, stream ordering, numerical output, and range rejection |
| Executables | `executable` | Loading configured HSACO, function lookup, dispatch, and exact output |
| Virtual memory | `virtual_memory`, `virtual_memory_gpu` | Scope-qualified capabilities, batched mapping, aliases, permissions, and backing ownership |
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

The `group` target defaults to a task CPU producer and the first GPU consumer.
`HRX_CTS_GROUP_PRODUCER` and `HRX_CTS_GROUP_CONSUMER` select exact participant
URIs, allowing the same shared-backing, mapped-access, reuse, and trim checks to
run across two GPUs. Host access has its own scope; neither participant needs
to execute on the CPU. The selected devices must admit the test's shared,
host-visible memory contract.

The `heterogeneous` target constructs one task CPU, AMD GPU, and XDNA NPU group.
CPU fills feed an NPU multiply, GPU copies move the result, and a CPU copy makes
the final output available for inspection. All operations use public HRX entry
points, with semaphore edges between devices and no intervening host waits.
A host gate holds the first allocation pending until its consumers are queued.
The test releases its application runtime/group references while dependent
resources remain live, returns storage after its last use, reuses it through a
sibling pool, and trims the shared source. Runtime errors during final teardown
fail the process as well as errors during execution.

This target requires both GPU and NPU hardware and uses the committed NPU
arithmetic fixtures for the supported execution profiles. It qualifies output,
ownership, and ordering with both HAL's conservative visibility defaults and
explicit prepared barriers. The recorder prepares scope pairs once, combines
effects from the participating buffers, and supplies any required resource
recipes with their concrete ranges. NULL queue boundaries retain default
visibility; supplied empty lists request no optional payload barriers. Neither
changes semaphore dependencies or mandatory native control publication.
Cross-device waits may require host admission; passing this test does not imply
that the devices exchange every dependency entirely through firmware.

HIP tests select GPUs through the frontend and do not use `HRX_CTS_DEVICE`.
`HRX_GPU_DRIVER` selects the frontend implementation and the harness's `gpu:N`
namespace; exact native HRX URIs select their driver directly:

| Value | Implementation |
|-------|----------------|
| `amdgpu` | Default: native AMD GPU HAL, with RDNA and CDNA enumeration |
| `rdna` / `cdna` | Native AMD GPU HAL, restricted to that architecture family |

The native AMD GPU driver must be compiled into the library. Compatibility
visibility variables are processed before the frontend creates its device group.

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

# Public HRX execution through one shared CPU/GPU/NPU group.
iree-bazel-test --config=asan \
  //libhrx/cts/tests/device:hrx_cts_heterogeneous

# Shared backing across two selected GPUs on a runner exposing both devices.
iree-bazel-test --config=asan \
  --test_env=HRX_CTS_GROUP_PRODUCER=amdgpu://0 \
  --test_env=HRX_CTS_GROUP_CONSUMER=amdgpu://1 \
  //libhrx/cts/tests/device:hrx_cts_group
```

GPU targets declare their hardware requirements and need a compatible runner.
The HIP DSO CTS targets currently select Linux; portable adapter tests and public
HRX tests also cover Windows. With CMake, `LIBHRX_BUILD_CTS=ON` includes the CTS
in the configured build, and CTest discovers its tests.

## Submission measurements

`//libhrx/cts/tests/executable:queue_benchmark` compares public HRX queue calls
with direct HAL calls using the same group, queue, executable, buffers, pool,
and semaphore. `Fill` and `Dispatch` retain default visibility; `PreparedFill`
and `PreparedDispatch` use the same prepared acquire/release effects and concrete
ranges on both paths. Native descriptors are prepared once, so the difference
includes HRX's per-call translation. `TransitionQuery` measures the cached
scope-pair lookup separately. Static linkage is identified in the output.

`Submit` excludes final completion, and `Complete` includes it. Burst rows
report time per batch; divide by the batch size to compare per-operation costs.
Allocation rows include both acquire and return submissions, with final
completion and handle release excluded from `Submit`. Source acquisition/byte
counters and provisioned queue counts expose backing growth and queue inventory.
Every queue row verifies changing kernel/fill output outside timing.

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

## Dependent resident-kernel measurements

`//libhrx/cts/tests/executable:dependent_chain_benchmark` measures an ordered
producer/consumer chain with a resident working set. Each dispatch reads two
values per element and writes one value; the neighboring read crosses workgroup
boundaries. An immutable input and two alternating outputs occupy 48 KiB through
24 MiB. Chain lengths include 1, 8, 32, 33, 64, 128, 129, and 512 dispatches,
covering short sequences, command-preparation crossovers, and longer recordings
that batch many small operations.

The kernel is authored in Loom. HRX and HIP load the same generated code object,
so the comparison isolates runtime behavior for this program. An independent
CPU recurrence checks the entire output before and after timing. Allocation,
upload, graph construction, and final readback are outside the timed region;
intermediate results stay on the device. This approximates the dependency and
working-set shape of small stencil or simulation kernels without reproducing a
particular application's equations.

| Mode | HRX submission | HIP comparison |
|------|----------------|----------------|
| `StreamFlushEach` | Flush each dispatch, wait after the whole chain | Ordinary ordered launches |
| `StreamFlushEnd` | Record the chain, flush once, then wait | Ordinary ordered launches |
| `GraphReplay` | Replay a previously instantiated chain | Replay a previously captured chain |

HIP eagerly submits ordinary launches, so both stream rows use the same HIP
program. HRX's per-dispatch flush also publishes a timeline point after each
operation; this row exposes that cost. `DependentChain` includes final
completion. `DependentChainSubmit` excludes the final wait while still retiring
every iteration before the next reuses storage. Times are per chain; divide by
`dispatches` for a per-dispatch figure. `resident_bytes` counts the three
allocations, and `logical_bytes` counts shader reads and writes across the chain,
not measured HBM or bus traffic.

`--chain_runtime=hrx|hip` selects the runtime. HRX takes an exact
`--chain_device` URI; HIP takes an explicit `--chain_hip_library` DSO and
`--chain_hip_device` ordinal. Both runs must select the same physical GPU. The
output label records the architecture and device selection. By default, both
paths poll their public stream-query API before joining completion.
`--chain_active_wait=false` uses blocking completion instead.

For example, after building the target with `--config=opt`, run the executable
under the qualified runner's measurement lease with:

```text
--chain_runtime=hrx --chain_device=amdgpu://0
--benchmark_filter=StreamFlushEnd/elements:131072/dispatches:64/
--benchmark_min_time=100x --benchmark_repetitions=5
```

The HIP counterpart uses `--chain_runtime=hip`,
`--chain_hip_library=/path/to/libamdhip64.so`, and the corresponding device
ordinal. Keep the filter, completion policy, and fixed iteration count identical,
and interleave the runtime comparisons. The generated benchmark smoke test runs
one small case from each mode through HRX and checks its output; comparing HIP
requires explicitly selecting the implementation DSO.

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
