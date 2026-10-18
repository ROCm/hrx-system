# libamdf conformance

The conformance suites exercise the public [libamdf API](../README.md), native
resource ownership and caller-visible results. The
[command and interop coverage map](gpu/coverage.md) connects behaviors to case
sources, deployment paths and required-case invocations. Ordinary corpora use one
runtime-loaded shared-library executable. The API-negotiation corpus also
covers static and link-time shared binding. Source membership, successful
compilation and an observed native result establish different facts.

## Corpora and owners

[GPU command conformance](gpu/README.md) describes PM4, SDMA, AQL, memory recipes,
compiler fixtures and manual lifecycle cases. The
[standalone XDNA recipes](xdna/recipes/README.md) cover finite image loading,
binding, CPU publication, arithmetic results, full guards and retirement with
CTS-only fixtures compiled from [Loom source](xdna/programs/mul_i32.loom)
during the build. The [GPU/XDNA recipes](interop/gpu/xdna/recipes/README.md)
cover finite GPU transfer/shader compositions and resident GPU/NPU exchanges.
Resident cases exercise both dataflow initiators, one/two credits, independently
progressing workers and explicit startup/drain, with no intermediate host relay.
Finite cases join native phases on the host without accessing shared payloads
between those phases. The
[XDNA HAL native suite](../../runtime/src/iree/hal/drivers/amd/xdna/README.md#verification)
separately exercises the production image loader, pools, dispatch and completion.
The XDNA API
[queue cases](xdna/kernel_queue_test.cc) cover immutable ranges, capacity,
notifications and explicit batch retirement. In particular,
`XdnaKernelQueueTest.RefreshRetiresBatchesAndReusesPacketStorage` distinguishes
read-only status from checked progress and transport-slot reuse.

Process and instance native lifetimes are separate invocations of the same
binary. For example, `//libamdf/cts/core:memory_dynamic` and
`:memory_dynamic_instance` use different lifetime arguments. Cases query the
actual capabilities: registration requires that service, and native-owner
recreation requires reclaimable acquisition. Ordinary memory, queue and interop
cases share one cached device per endpoint for the test process. Workload
allocations, mappings, queues and completion storage remain case-owned.

A failed oracle still reaches the case's required completion and retirement
before any reuse. Cleanup follows the native ownership dependencies and checks
release results. Failure to remove a native owner preserves storage that may
remain reachable. A passing payload alone is not successful cleanup.

The [Windows file corpus](gpu/windows/file_io_test.cc) composes IOCP transfers
with allocated and registered GPU backing in buffered and unbuffered modes.
Native file queries determine transfer geometry; libamdf profiles determine
backing geometry and memory-pair queries determine visibility operations.
PM4 or SDMA swaps file blocks between allocations, and IOCP writes the
GPU-produced bytes before any CPU payload inspection. An independent reopen
checks the entire file, while full memory guards check transfer boundaries.
The corpus selects advertised queue families; Windows SDMA requires an
HWS-enabled engine. File handles, completion ports and native I/O remain owned
by the caller, outside libamdf.

The [Linux file-pool corpus](gpu/linux/file_pool_test.cc) uses the owned
`HOST_FILE_IO` allocation contract through PM4, AQL, and SDMA. Buffered and direct
file requests use a fixed io_uring registration of the original host view while
GPU aliases rebind independent slots. Source-only and default-address cases
check repeated load/compute/write generations, full file and memory guards, and
terminal cancellation before reuse. A separate ownership case accesses one
pool through two logical devices on the same endpoint, binds independent slabs
into one stable GPU virtual region, and checks both native attachment releases.
The ordinary corpus shares its cached physical device; only that ownership
case creates a second logical device.

Linux file-pool execution requires access to udmabuf and its size-limit query,
plus permission and locked-memory budget for io_uring buffer registration.
Direct cases require a disk-backed `TEST_TMPDIR` and compatible file geometry;
the case records that geometry and filesystem type. On Btrfs its private,
unlinked test file disables copy-on-write before allocating extents so that
checksummed-write fallback does not turn the direct case into buffered I/O.
Missing optional services produce an explicit skip on CI and user machines.
An unavailable allocation profile retains its native reason; the corpus checks
that allocation reports that reason without publishing an output. Available
profiles proceed through the full workload, where allocation, data, and cleanup
failures fail the test. Positive qualification runs on provisioned machines name
required cases so that absence cannot pass as execution. The native file helper
owns file/ring setup and cleanup, outside the library under test.

## Build and resource selection

Shader-bearing corpora compile their fixtures with Loom during the build.
GPU fixtures require its AMDGPU target/emitter, XDNA fixtures require its XDNA
target/emitter, and GPU/XDNA recipes require both. SDMA and host encoding tests
have no shader compiler dependency. Production library builds remain independent
of Loom, the runtime and their image loaders.

Hardware-backed suites declare `libamdf.resource.amd_gpu` or
`libamdf.resource.xdna`. CMake exposes `runtime-resource=amd-gpu` and
`runtime-resource=amd-xdna`. These suites remain discoverable by wildcard
selection; host-only runs exclude their requirements. GPU/XDNA interoperability
requires both families and resources. Native and interop cases share the AMD
hardware resource group.

[Physical peer-GPU cases](gpu/peer/README.md) also declare
`libamdf.resource.amd_gpu_peers` and require both exact endpoint selectors.
Their runner reserves two distinct physical GPUs; a single-GPU CI profile
does not admit that resource. Missing peer selection skips before activation.

A host-only invocation without the ROCr-backed AMDGPU HAL is:

```sh
iree-bazel-configure -DAMDF_BUILD=ON -DIREE_HAL_DRIVER_AMDGPU=OFF
iree-bazel-test --config=asan \
  --test_tag_filters=-iree-run-requirement=libamdf.resource.amd_gpu,-iree-run-requirement=libamdf.resource.xdna \
  //libamdf/...
```

The corresponding CMake selection is:

```sh
iree-cmake-configure -DAMDF_BUILD=ON -DIREE_HAL_DRIVER_AMDGPU=OFF -DLIBHRX_BUILD=OFF -DLOOM_BUILD=OFF
iree-cmake-test -R '^libamdf/' -LE 'manual|runtime-resource='
```

The XDNA CI entry points configure, build and run the native suites and HAL
consumers:

```sh
python build_tools/devtools/ci.py iree-bazel-xdna-asan
python build_tools/devtools/ci.py iree-cmake-xdna-asan
```

The client configurations compile RDNA and XDNA together, including both Loom
emitters for authored fixtures. Linux selects host and XDNA execution; Windows
also admits native GPU, Vulkan and D3D12 resources. The jobs cover
`//libamdf/...` and `//runtime/src/iree/hal/drivers/amd/...` without enabling the
legacy AMDGPU HAL:

```sh
# Linux client configuration.
python build_tools/devtools/ci.py iree-bazel-amd-client-asan
# Windows, in a configured native clang-cl environment.
python build_tools/devtools/ci.py iree-bazel-amd-client
```

The [Loom workflow](../../.github/workflows/ci_iree.yml) owns Linux and Windows
platform setup and hardware assignment. Compile-time family enablement does not
declare a native device available. Linux XDNA needs access to the `amdxdna`
driver and `/dev/accel`; `/dev/kfd` and `/dev/dri` access alone supplies no NPU
execution interface.

The ordinary Linux AMDGPU jobs build and run the GPU-only libamdf slice
alongside the AMDGPU HAL. They enable RDNA/CDNA families, the authored GPU
fixtures and `libamdf.resource.amd_gpu`, with XDNA and external-API suites
excluded by their separate requirements:

```sh
python build_tools/devtools/ci.py iree-bazel-amdgpu --amdgpu-target "${GPU_TARGET}"
python build_tools/devtools/ci.py iree-cmake-amdgpu --amdgpu-target "${GPU_TARGET}"
```

The runner target controls HAL compilation and the descriptor sets linked into
Loom. The command corpora build the physical kernel variants supported by those
descriptor sets and select the discovered device at execution. The default Loom
configuration includes the full physical-target catalog. Bazel's
explicit `--target` override preserves a narrower package selection. CMake
builds `libamdf/all` and includes its GPU resource labels in hardware execution.
Manual lifecycle cases remain outside these ordinary jobs. Resource admission
permits execution; required-case arguments below establish which individual
behaviors a deployment must actually run.

## Required cases and result records

Qualification invocations name required cases with repeatable
`--amdf_require_test=Suite.Case`. An absent, filtered, skipped, unexecuted or
failed required case fails the process. Discovery invocations can report an
unavailable optional service without claiming its coverage.

GPU selection accepts an exact endpoint and an expected compiler target before
activation. XML records bind the selected identity and case-specific parameters
to the result. The [GPU qualification method](gpu/qualification.md) explains
native identity, source/artifact records and independent output observation.
A frozen run's coverage does not follow subsequent source changes.

Runners can set `AMDF_CTS_GPU_NATIVE_IDENTITY` to `linux_device:MAJOR:MINOR`
or `windows_adapter:LUID:PHYSICAL_INDEX` (hexadecimal LUID, decimal indices).
The CTS resolves this public native identity to its opaque primary endpoint
before activating a GPU. An explicit `--amdf_gpu_endpoint_id` must agree with
that allocation. Malformed or conflicting selections, and identities absent
from discovery, fail the run.
ROCr/HIP device ordinals do not select libamdf endpoints. CI correlates its
visible ROCr GPU with KFD and DRM metadata and forwards the native identity
to both Bazel and CMake test processes. Vulkan CI forwards its independently
verified render identity through the same CTS selection.

XDNA hardware jobs require successful activation and the allocated execution
path with a compatible image. Missing hardware, activation failure or a missing
required image fails that job. Registration, import and placement cases use
the live device's capabilities. A shared architecture name does not replace
image/profile admission.

## Correctness and measurement

Functional conformance and performance answer different questions. Source
review follows hot entry points, failure paths and first-use paths to establish
where allocation, synchronization and native calls occur. Native oracles check
public behavior, payloads, guards, immutable inputs and complete retirement.
Neither replaces the other's evidence.

The [benchmark guide](../benchmarks/README.md) separates memory lifecycle,
publication-only and completed-operation intervals. Generated memory benchmark
smoke tests use the shared hardware resource group and a single iteration;
capability absence is a skip, while native or byte-check failures terminate the
run. The XDNA HAL queue benchmark's smoke invocation is documented in the
[HAL benchmark guide](../../runtime/src/iree/hal/drivers/amd/xdna/README.md#queue-benchmarks). Instrumented and smoke
runs do not supply performance numbers. Optimized measurements retain their
exact timed boundary, raw repetitions, build identity and machine policy.
