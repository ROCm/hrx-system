# Physical peer-GPU conformance

This package is reserved for tests whose producer and consumer are different
physical GPUs. Two queues or two device handles for the same endpoint do not
satisfy that condition. Each grouped dynamic corpus shares one cached device
per selected endpoint; each case owns its allocations and native commands.

## Construction and SDMA

The construction cases cover owned SYSTEM memory, registered host storage and
LOCAL memory with two explicit access attachments. Each queries the joint
profile, native extent/alignment, device addresses and achieved backing facts.
Registered storage checks the original allocation and its guards after native
release. A successful attachment establishes neither executed dataflow nor
mutually atomic reach.

`PeerSdmaSystemTest.DeviceDrivenRoundTrips` exchanges payload through one owned,
coherent SYSTEM allocation. It queries every host/device and device/device
visibility direction before creating the two SDMA queues. The case requires
USER publication and NONE payload cache transitions; each device uses its own
access address and packet-format features.

Four prepublished round trips run without a CPU dependency or payload relay
between stages, then the initiating roles reverse with new data. Aligned,
single-writer DWORD fences and equality polls carry the dependencies. Those
completion cells stay stable through all readers and rearm only after both
streams retire; the case requires no atomic read-modify-write capability.
The host checks every input, intermediate, output, guard and control word,
plus both immutable command rings, before native retirement. Queue-first
cleanup precedes backing release. This SYSTEM path does not exercise direct
peer-local-memory transfer or choose an xGMI engine.

The hardware reference separates [completion and signal lifetime](../../../../docs/reference/amd/gpu/sdma/atomics.md)
from [payload cache visibility](../../../../docs/reference/amd/gpu/sdma/cache.md).
The case queries the latter explicitly; the fence/poll protocol supplies the
former.

## AQL through peer-local memory

The separate `aql/` corpus uses native AQL signals and existing Loom-generated
transform kernels. Each endpoint selects its own exact-target image. A
signal-only case establishes both completing-device directions; four payload
cases vary the LOCAL memory owner independently of the producer. The other GPU
consumes the producer's output.

Each payload case attaches both GPUs to device-only LOCAL memory, without
mapping that memory on the host. An owner dispatch initializes its complete
allocation-rounded extent. The producer then changes a 257-word subspan crossing
the allocation's 4 KiB offset boundary, and the consumer transforms the entire
extent into coherent SYSTEM output. Two generations change the input and both
transforms. A CPU oracle checks all output and guards, untouched LOCAL memory
through the consumer's transform, preserved input, immutable argument pages
and native signal fields.

Prospective and concrete pair queries must agree for every host/device and
device/device direction used by the recipe. The AQL dispatches execute the
queried SYSTEM release/acquire transitions. Their dependencies use complete
native signals, independently addressed by each GPU; this does not require or
establish generic atomic reach for the backing.

The consumer is published first. A separate arrival completion establishes
progress before the producer is published. From producer publication through
consumer completion, the CPU relays neither dependencies nor payloads. The
host snapshots output after consumer completion and before independently
joining the producer or retiring either queue. Signals, images, arguments and
payloads remain alive through all readers and queue removal. The
[AQL barrier reference](../../../../docs/reference/amd/gpu/aql/barriers.md)
separates dependency completion from payload visibility; the
[local-memory recipe](../../../../docs/reference/amd/gpu/recipes/local-memory.md)
describes staging and ownership obligations.

## Selection and execution

The runner provisions and exclusively reserves two physical GPUs, then passes
their exact contemporaneous endpoint IDs:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  --//libamdf/config:enabled=true \
  //libamdf/cts/gpu/peer:peer_dynamic \
  --test_arg="--amdf_gpu_endpoint_id=${GPU_ENDPOINT_ID}" \
  --test_arg="--amdf_gpu_peer_endpoint_id=${PEER_GPU_ENDPOINT_ID}" \
  --test_arg=--amdf_require_test=Acquisition/GpuMemoryGroupTest.OneBackingForTwoPhysicalConsumers/SystemCreate \
  --test_arg=--amdf_require_test=PeerSdmaSystemTest.DeviceDrivenRoundTrips
```

The AQL corpus has its own invocation and required cases:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  --//libamdf/config:enabled=true \
  //libamdf/cts/gpu/peer/aql:aql_dynamic \
  --test_arg="--amdf_gpu_endpoint_id=${GPU_ENDPOINT_ID}" \
  --test_arg="--amdf_gpu_peer_endpoint_id=${PEER_GPU_ENDPOINT_ID}" \
  --test_arg=--amdf_require_test=PeerAqlMemoryTest.NativeSignalsAcrossBothDevices \
  --test_arg=--amdf_require_test=Placement/PeerAqlLocalMemoryTest.TransformAcrossPhysicalDevices/Owner0Producer0 \
  --test_arg=--amdf_require_test=Placement/PeerAqlLocalMemoryTest.TransformAcrossPhysicalDevices/Owner0Producer1 \
  --test_arg=--amdf_require_test=Placement/PeerAqlLocalMemoryTest.TransformAcrossPhysicalDevices/Owner1Producer0 \
  --test_arg=--amdf_require_test=Placement/PeerAqlLocalMemoryTest.TransformAcrossPhysicalDevices/Owner1Producer1
```

The [enumeration example](../../../examples/enumerate.c) reports the IDs and
native correlation identities. The runner connects those identities to its
physical inventory and reservation. On Linux, distinct render nodes alone do
not prove separate physical packaging; compute partitions can expose multiple
endpoints. Topology and access reach remain separate facts.

Both selectors are checked before activation. Missing peer selection skips
before device discovery; a peer without a primary, duplicate selectors or equal
IDs fail argument parsing. Explicit unavailable identities fail instead of
selecting another GPU. Cases match both endpoint-local services passively and
record the primary and peer identities/targets separately. Required-case
selection turns a missing, filtered, skipped or failed witness into failure.

Both corpora declare `libamdf.resource.amd_gpu` and
`libamdf.resource.amd_gpu_peers`. Single-GPU CI profiles do not admit the peer
requirement. Both Bazel and CTest use the existing GPU resource group to
serialize with other device users in the same invocation; assignment across
independent jobs remains the runner's responsibility. Construction and SDMA
compilation need only the enabled GPU library family. The `aql/` corpus also
requires Loom's AMDGPU target/emitter. Both build on Linux and Windows; native
cases passively require their actual queue services before activation. Windows
does not expose the AQL service, so those cases skip unless required, in which
case qualification fails.
