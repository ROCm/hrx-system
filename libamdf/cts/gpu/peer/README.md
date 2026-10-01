# Physical peer-GPU conformance

This package is reserved for tests whose producer and consumer are different
physical GPUs. Two queues or two device handles for the same endpoint do not
satisfy that condition. The grouped dynamic corpus shares one cached device
per selected endpoint; each case owns its allocations and native commands.

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

The corpus declares both `libamdf.resource.amd_gpu` and
`libamdf.resource.amd_gpu_peers`. Single-GPU CI profiles do not admit the peer
requirement. Both Bazel and CTest use the existing GPU resource group to
serialize with other device users in the same invocation; assignment across
independent jobs remains the runner's responsibility. Compilation needs only
the enabled GPU library family, without a shader compiler dependency.
