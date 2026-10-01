# GPU command conformance

The command CTS exercises the public libamdf ABI through one runtime-loaded
shared-library executable per native corpus. An operation or field combination
gets a named GoogleTest case, not a separate executable. Sources remain
separate translation units so the build can compile them concurrently.

The [hardware reference](../../../docs/reference/amd/gpu/README.md) describes
native command semantics. The [library queue guide](../../docs/gpu.md) describes
the public caller contract. The [coverage map](coverage.md) connects implemented
behaviors, deployment paths and exact required-case invocations. Neither source
membership nor packet definitions alone establish a deployment's executed
behavior.

## Physical boundaries

```text
gpu/
  gpu_device_fixture.h       # Passive selection and the existing device cache.
  util/                      # Case-owned memory and native queue transports.
  pm4/                       # One native PM4 corpus.
    encoding/                # Host-only wire-format checks and test encoder.
  sdma/                      # One native SDMA corpus.
    encoding/
  aql/                       # One native AQL corpus and publication helper.
    encoding/
  recipes/                   # One single-GPU composition corpus; N queues.
  kernels/                   # Compiler fixture/provenance boundary.
  peer/                      # One explicitly reserved physical multi-GPU corpus.
  lifecycle/                 # Completed resources and opt-in device recreation.
```

[GPU/XDNA recipes](../interop/gpu/xdna/recipes/README.md) cover finite GPU
transfer/shader compositions and resident bidirectional exchange with the NPU.
Resident programs own intermediate payload and credit handoffs; the host owns
startup and terminal joins. Platform memory, kernel-queue and external-API suites
retain their separate dependencies. The `kernels` package owns authored Loom GPU
fixtures and their generated products; `xdna/programs` owns the finite and
resident array fixtures. The [peer corpus](peer/README.md) requires explicit
primary and peer endpoint selection under a two-physical-GPU reservation.

`CtsDeviceCache` creates one instance and one device per endpoint and engine
kind for an executable, caching activation failures as well. GPU queue families
on the same endpoint share that cached GPU device. `GpuCommandTest` borrows it
and owns stable case-local memory and queue collections. Each case can create
N queues without creating N devices. Explicit cleanup precedes provider unload,
including partial initialization and failed native removal paths.

The shared utilities expose memory contracts and native command storage.
`GpuCommandQueue` publishes finite PM4/SDMA streams through either a mapped
USER ring or executable KERNEL command buffers. Publishing owned KERNEL storage
submits only each newly appended range. PM4 control flow preserves the entry context:
USER calls a first-level IB from the primary ring, while KERNEL submits that
IB directly. Compute IB2 call-and-return is unsupported.
The IB cases carry acquisition, dispatch, shader join and completion inside
the referenced body, with the same payload and full-backing oracles. AQL and
USER lifecycle cases use the mapped producer state in `GpuUserQueue` directly.
The utilities publish command bytes without inserting payload cache commands
or completion markers. Engine fixtures own exact completion semantics. Resource
setup is test infrastructure; these helpers are separate from production queue
scheduling machinery.

Dataflow cases observe their payloads after the named completion and before
retiring command storage. Independent CPU expectations cover the initialized
payloads and guards; input preservation is checked separately. Oracle failures
still reach retirement before teardown or any reuse. USER consumption and
KERNEL submission retirement retain their distinct native meanings; neither
replaces the case's earlier payload observation.

## Build and execution

`//libamdf/cts/gpu/{pm4,sdma,aql,recipes,peer}` each uses `amdf_cts_test_suite` with
the default dynamic provider. Process/instance native lifetimes are two test
invocations of the same binary. Only `//libamdf/cts/core:query` explicitly
retains the three binding modes.

Native corpora require x86-64 and Linux or Windows at compile time, inherit
the `libamdf.resource.amd_gpu` execution requirement and share the AMD GPU
resource group. Physical peer cases additionally declare
`libamdf.resource.amd_gpu_peers`; single-GPU jobs cannot admit them. Encoder
target predicates and family capabilities select the actual native queue service:

| Platform | PM4 | AQL | SDMA |
| --- | --- | --- | --- |
| Linux CDNA | Outside this corpus's compute matrix. | USER | USER |
| Linux RDNA | USER | USER | USER |
| Windows RDNA | KERNEL | Unavailable. | KERNEL on HWS-enabled engines. |

Tests specifically exercising mapped USER state require that service
independently of ordinary command behavior. Ordinary Linux RDNA AQL dispatch
does not imply support for the optional vendor barrier-value packet. A case
using that packet requires its separate family capability before activation.
SYSTEM-memory cases require their actual backing and visibility contract;
only LOCAL-memory cases require a device-local heap.

Windows SDMA admission requires a hardware-scheduled (HWS) engine. The provider
does not expose the software-scheduled (SWS) submission path; deployments with
only that path report no SDMA family. Ordinary discovery skips those cases,
while an explicitly required SDMA case fails qualification.

The CPU/PM4 atomic cases cover naturally aligned 32-bit and 64-bit STORE to
owned coherent SYSTEM memory. They require matching prospective and achieved
memory operation masks, queue operation support, and SYSTEM atomic reach in
both directions. Other operations, backings, and mapping policies require
their own contracts; payload visibility remains independent of atomic reach.

Each `encoding/` package has one plain host-test binary. Package policy removes
the GPU execution requirement for these exact packages, so byte-layout checks
run without a GPU and do not reserve a GPU slot. GPU-family build enablement
still applies. Shader-bearing corpora build their Loom fixtures by default;
SDMA and host encoding targets have no shader compiler dependency. Native
command corpora acquire no Vulkan/D3D12 dependencies.

Bazel declarations are authoritative; generated CMake targets preserve the
same corpus, dynamic loading, requirements and resource group. For example,
`libamdf_cts_gpu_recipes_recipes_dynamic_bin` is the CMake build target, and
`libamdf/cts/gpu/recipes/recipes_dynamic` is its process-lifetime CTest name.
The [qualification invocation](qualification.md#required-witnesses-and-deployment-identity)
adds an exact endpoint ID, required case names and an expected compiler target.

The manual lifecycle corpus has ordinary PM4/SDMA cases that copy between
exact-access attachments on the cached device, observe both complete pages
before retirement, and release their queue before its backing. SDMA uses the
coherent SYSTEM COPY/FENCE recipe; PM4 retains its explicit cache transitions.
SDMA emits explicit GCR acquire/release when the family provides that format,
or uses the family's per-command scope fields. The dependency NOP and timestamp
cases keep cache commands outside the dependent-copy sequence so they cannot
replace the ordering operation under test. Query-driven compute compositions
place required SDMA cache operations at each upload/download boundary; explicit
GCR variants also exercise that stream when the backing permits a no-op.

PM4/SDMA batches publish four prepared upload/compute/download graphs with one
publication per queue. Each graph has separate payload, arguments and progress
records. With a shared transfer queue, SDMA observes one graph's shader
completion before uploading the next; the next PM4 upload wait therefore also
joins the preceding shader. Independent upload and download queues allow
uploads to advance without waiting for downloads. In that layout, the ordered
data-acquire case explicitly waits for the preceding shader's completion before
acquiring data or rebinding compute registers. Both layouts preserve the
target's data-cache actions and reuse already-published immutable code; their
full-barrier controls retain shader-idle and instruction invalidation.

Each batch case snapshots all graph readbacks after the final download, then
independently joins compute and upload before observing other backing. All
queues retire before the second batch rewrites inputs. Complete payload and
guard checks cover each graph and every allocation. These device chains qualify
host-independent batch advancement with shared or separate transfer queues.
Queue count alone establishes neither physical engine assignment nor overlap
between transfer and compute.

The [bounded streaming cases](recipes/pm4_sdma_streaming_test.cc) reuse
1/2/4/8 payload slots across a longer sequence of graphs. They require mapped
USER publication on one PM4 queue and two independent SDMA queues. The CPU
services source refill and readback consumption while the GPU queues exchange
their own dependencies; no host completion wait separates upload, compute and
download. Upload completion protects source refill, shader completion protects
input reuse, download completion protects output reuse, and CPU consumption
protects readback reuse. Command-ring capacity and final native consumption
remain independent of those payload acknowledgments.

Named cases exercise full or ordered acquisition and 32-bit control tokens
crossing the high bit or wrapping through zero. Producer-closure cases stop
after complete graph groups, including zero work and partial slot windows.
One variant leaves accepted ingress pending until closure; another prepares
an extra input with no submitted consumer. Draining supplies all accepted
inputs and consumes their outputs, using each slot's actual accepted generation
without inventing native work for unused preparation. Full backing, guard and
control checks include inactive slots and unaccepted result sentinels. Every
stream retires all three native frontiers before its backing is reset.

The lifecycle cases exercise the same resource helper as the `DISABLED_`
peer-device recreation scenarios, without creating extra devices. Recreation requires
`--gtest_also_run_disabled_tests` and is a separate qualification. The manual
corpus is not part of the ordinary command aggregate. Other API/interop suites
can have intentional owner-lifetime tests of their own.

## Category and source map

The table links the authoritative case lists in each corpus's Bazel declaration.
Additional translation units enter its explicit `srcs`; adding a file does not
create another executable. Source membership is separate from the recorded
native coverage of each deployment. The operation references describe the
remaining field, composition and architecture boundaries within each group.

| Corpus | Authoritative case sources | Other qualification boundaries |
| --- | --- | --- |
| PM4 | [pm4/BUILD.bazel](pm4/BUILD.bazel) | Publication, additional atomic operations/backing, cache policies, indirect-buffer lifetime and counter ownership. |
| SDMA | [sdma/BUILD.bazel](sdma/BUILD.bazel) | Additional fence/poll fields, atomic reach, cache policies, indirect buffers and counter ownership. |
| AQL | [aql/BUILD.bazel](aql/BUILD.bazel) | Signal reach, additional executable lifecycles, profiling, counters and metadata. |
| Recipes | [recipes/BUILD.bazel](recipes/BUILD.bazel) | Additional backing classes, producer/consumer compositions and executable visibility. |
| Manual lifecycle | [lifecycle/BUILD.bazel](lifecycle/BUILD.bazel) | Ordinary same-device copies are enabled; peer-device recreation remains disabled. |
| Physical peers | [peer/BUILD.bazel](peer/BUILD.bazel) | Explicit endpoint reservation, joint construction and device-driven SYSTEM SDMA round trips; peer-local dataflow, atomic reach and topology remain separate witnesses. |
| GPU/NPU recipes | [interop/gpu/xdna/recipes/BUILD.bazel](../interop/gpu/xdna/recipes/BUILD.bazel) | Finite transfer/shader chains and resident exchanges cover both initiators, credits, independent workers and startup/drain. Cross-output-channel publication, other imported backing and simultaneous independent traffic require separate witnesses. |

Cases use real commands and changing exact data. They do not exhaust their
opcodes' fields. New cases add independent oracles, legal field partitions and
compositions to the corresponding engine group. Shared testbench changes affect every corpus and need
their own caller/ownership review before dependent cases consume them.

Timing cases qualify the observation itself: sampling point, ordering, result
width and visibility. Clock calibration, hardware-counter ownership and
optimized latency/throughput experiments have their own
[measurement contracts](../../../docs/reference/amd/gpu/observability.md). Raw timestamp properties
from correctness tests carry no nanosecond or performance interpretation.
