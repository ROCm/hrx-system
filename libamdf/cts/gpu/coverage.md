# Command and interop coverage

The grouped corpora provide executable witnesses for the behaviors below.
Each result applies to the selected device, native transport, memory profile,
program images and source revision. A case's presence in this map does not
establish that it ran on a particular deployment. The
[qualification model](qualification.md) describes required cases, result
identity, independent observation and checked retirement.

## Find a behavior

| Behavior | Case sources | What the oracle observes |
| --- | --- | --- |
| USER ring capacity and reuse | [requested capacities](user_queue_test.cc) | PM4/SDMA default and explicit 4/16/64 KiB capacities, confirmed copies throughout each half-ring, three complete wraps, unchanged source and command backing, destination/completion guards, and consumption before reuse. Requires the selected family's USER service and capacity range. |
| PM4 memory operations | [write](pm4/write_test.cc), [copy](pm4/copy_test.cc), [atomic store](pm4/atomic_store_test.cc) | Changed data, selected widths and extents, surrounding bytes, producer/consumer ordering and CPU handoffs. Atomic-store coverage is distinct from read-modify-write operations. |
| PM4 memory dependencies | [wait](pm4/wait_test.cc) | All six conditional comparisons across 32/64-bit operands and full/partial masks, already-satisfied and consumer-before-producer publication, and always-pass below/equal/above operands. Complete backing checks precede native retirement; masked relational values separate ordering from ignored bits. |
| PM4 command-processor DMA | [DMA](pm4/dma_test.cc) | Incrementing byte copies, dependent two-copy chains and immediate DWORD fills consumed by DMA copies. RAW_WAIT compositions cover byte, DWORD/QWORD, 32-byte alignment, page and driver-chunk extents; fills cover zero/all-one, single-bit and byte-distinct patterns. Changed epochs check complete payload/control backing and immutable commands before native retirement. |
| PM4 execution and dependencies | [dispatch](pm4/dispatch_test.cc), [cross-queue handoff](pm4/handoff_test.cc), [indirect dispatch](pm4/indirect_test.cc), [command buffers](pm4/command_buffer_test.cc) | Shader outputs across generations, device-produced dispatch counts, immutable indirect buffers and completed-use command rebuilding. |
| PM4 conditional execution | [conditional dispatch](pm4/conditional_test.cc), [literal encoding](pm4/encoding/commands_test.cc) | Shader-produced DWORD predicates and normal/inverted DWORD/QWORD normalization select complete dispatch or dispatch-plus-marker ranges. Immutable replay changes both predicate halves, initializes private state on the device each time, preserves skipped output and checks complete backing before retirement. QWORD predicates use stable split reads, not atomic snapshots. |
| PM4 conditional command graphs | [branches](pm4/branch_test.cc), [literal encoding](pm4/encoding/commands_test.cc) | A shader-produced masked QWORD selects one of two retained shader bodies, each terminally chained to common completion. All seven comparisons, both predicate alignments, carry/sign-bit boundaries and partial/zero masks use pre-masked references (`reference & ~mask == 0`). Immutable replay preserves the unselected arm and all backing before native retirement. |
| PM4 shader resources | [LDS](pm4/lds_test.cc), [resource changes](pm4/resource_test.cc) | Cross-wave exchange through fixed workgroup storage and transitions between distinct resource configurations. |
| SDMA transfers | [copy](sdma/copy_test.cc), [fill](sdma/fill_test.cc) | Linear copies, byte tails/page crossings, DWORD fills, NOP-separated copy and fill-to-copy dependencies, changing-pattern payload reuse, immutable commands and fence-visible output. |
| SDMA inline updates | [write](sdma/write_test.cc) | Command-carried DWORD values, NOP-separated copy consumers, page crossings, final output before command retirement, and complete backing across changing epochs. |
| SDMA rectangular transfers | [rectangles](sdma/rectangular_copy_test.cc) | All five element sizes, independent source/destination coordinates and pitches, page crossings, the advertised Z-coordinate/depth widths, and complete source/destination/control backing across changed epochs. |
| SDMA completion-value dependencies | [memory waits](sdma/wait_test.cc) | All six conditional comparisons with full/partial masks, prepublished and CPU-after-arrival operands, and always-pass below/equal/above operands. Changing dependent-copy output, full backing guards and immutable commands precede completed-use reuse. Compared relational values stay below bit 31; signed ordering and wrap are separate contracts. |
| SDMA data visibility | [cache commands](sdma/cache_test.cc), [SDMA/AQL](recipes/copy_dispatch_test.cc), [PM4/SDMA](recipes/pm4_sdma_test.cc) | Explicit USER_GCR acquire/release, changing payloads, immutable commands, and queried cache operations at upload and download boundaries. |
| AQL publication and dependencies | [publication](aql/publication_test.cc), [barriers](aql/barrier_test.cc), [epochs](aql/epoch_test.cc), [fan-in](aql/fanin_test.cc), [scope](aql/scope_test.cc) | Slot reuse, independent producer publication, AND/OR/value dependencies, complete shader payloads and AGENT-to-SYSTEM scope composition. AND payload fan-in covers 1–11 producers across full and partial five-slot groups, changed epochs and opposing publication order. OR payload readiness covers every nonempty five-slot subset and each eligible ready slot, with other producers retained behind a gate until the consumer output is captured. Independent producer joins and native retirement precede changed-epoch reuse. |
| AQL execution and resources | [dispatch](aql/dispatch_test.cc), [geometry](aql/geometry_test.cc), [scratch](aql/private_test.cc), [LDS](aql/lds_test.cc), [resource changes](aql/resource_test.cc) | Complete/partial multidimensional grids, caller-owned private storage, fixed group storage and resource rebinding. |
| AQL transfer and reuse | [carriers](aql/transfer_test.cc), [byte copy](aql/byte_copy_test.cc), [pattern fill](aql/pattern_fill_test.cc), [executable reuse](aql/executable_test.cc), [worksets](aql/workset_test.cc) | PM4-carried copies, shader subspan operations, completed-use code replacement and independent final-use obligations. |
| CPU/GPU and cross-engine memory edges | [memory pairs](recipes/memory_pair_test.cc), [SDMA/AQL](recipes/copy_dispatch_test.cc), [PM4/SDMA](recipes/pm4_sdma_test.cc) | Queried concrete/profile policies, coherent SYSTEM and staged LOCAL payloads, upload/compute/download and final consumer output. PM4/SDMA also covers device-driven batches with shared or independent upload/download queues, full barriers or ordered data acquisition, and immutable code reuse. |
| Finite device-driven transfer streams | [PM4/SDMA recorded streams](recipes/pm4_sdma_finite_test.cc) | All commands resident before publication; 64 graphs reuse 1/2/4/8 payload slots without host refill. Separate, shared alternating and shared credit-grouped transfers retain every output, immutable owners, inactive slots and ring history across wrap and retired reuse. |
| Bounded CPU/GPU streaming and closure | [PM4/SDMA host streams](recipes/pm4_sdma_streaming_test.cc) | Reusable payload and command storage, source/readback acknowledgments, retained outputs, 32-bit control-token boundaries, and exact per-slot/native retirement. Closure preserves pending accepted ingress and distinguishes unused preparation from submitted work. |
| Device-generated SDMA | [GPU-selected dependent copies](recipes/device_sdma_test.cc) | One shader computes COPY operands from previous copied data, authors packets and publishes its owning SDMA queue. Exact transcripts, all transfer lengths from one to 64 words, ring wrap, destination reuse and checked teardown run without intermediate host joins. |
| Batched device-generated SDMA | [Command and payload credits](recipes/device_sdma_test.cc) | Cooperative GPU publication/consumption with asynchronous SDMA, independent command-space and payload admission, actual per-reservation frontiers, copied-data-dependent batches, append with live payloads, empty work and exact final drain across credit and batch boundaries. |
| Device-generated staged transfers | [Separate compute consumers](recipes/device_sdma_test.cc) | GPU-selected SYSTEM uploads into SYSTEM or LOCAL slots, a separate multi-workgroup consumer, GPU-published download and copied-result-dependent selection. One complete finite AQL batch, explicit dispatch completion ordering, 1/2/4 reusable slot pairs, row/block extents through 1 MiB, per-job full-slot and descriptor oracles. |
| Physical peer construction and SDMA | [joint memory](peer/memory_group_test.cc), [SYSTEM round trips](peer/sdma_system_test.cc) | Explicit endpoint selection, joint attachment, changing bidirectional device-driven payloads and complete backing/command checks before retirement. |
| Physical peer AQL | [native signals and peer-local memory](peer/aql/local_memory_test.cc) | Both signal directions, both LOCAL memory owners and producers, queried SYSTEM scopes, consumer-first dependency, changing transforms and complete output before independent joins. |
| GPU initiated file transfers | [Linux io_uring](linux/io_uring/README.md) | One-credit causal round trips and three-credit concurrent gathers, key-based deduplication, retained final readers, independent peer slot reuse, CQ ticket identity, scattered write/reload, short reads/EOF and stop-and-drain errors. Complete file/payload/guard oracles cover buffered/direct paths. Logical consumers share one invocation; durability and independently dispatched matmuls are separate contracts. |
| Matched file I/O latency | [Completion-driven comparison](linux/io_uring/file_latency_test.cc) | Identical native/host-relayed programs, one/four causal streams, 4 KiB through 4 MiB transfers, sparse arrivals, payload/file oracles and shader/DRM clock brackets. Optimized broker-managed samples are distinct from CTS timing. |
| CPU/NPU execution | [CPU/XDNA recipes](../xdna/recipes/README.md) | Allocated/registered backing, queried host publication/acquisition, changed arithmetic outputs, full guards and native retirement. |
| Finite GPU/NPU execution | [GPU/XDNA recipes](../interop/gpu/xdna/recipes/README.md#finite-recipes) | GPU transfers or shaders produce NPU inputs and consume its output; native phases are joined on the host without intermediate CPU payload access. |
| Resident GPU/NPU execution | [resident recipes](../interop/gpu/xdna/recipes/README.md#resident-exchange) | Both dataflow initiators, one/two credits, independently progressing workers, complete per-generation transcripts, backing/payload layouts, startup abort and final drain. The host does not relay intermediate work. |
| Timestamp observations | [PM4 command processor](pm4/timestamp_test.cc), [PM4 shader](pm4/dispatch_test.cc), [AQL](aql/timestamp_test.cc), [SDMA](sdma/timestamp_test.cc) | Sampling order, written result extents, visibility and completed-use reuse. Resident recipes also retain raw GPU clock observations. |

The [wire-format tests](README.md#build-and-execution) check packet encodings
without activating hardware. They do not replace native field or composition
cases. Hardware counters and calibrated performance are separate observation
surfaces; correctness timestamps alone supply neither.

The COPY_DATA width cases check the destination extent and every source word,
including explicitly owned trailing source backing for the 32-bit form. They
retain the final destination DWORD at a mapping boundary. They do not assert
that the result width bounds the native source read footprint; the
[source-read observation](../../../docs/reference/amd/gpu/pm4/memory-commands.md#copy-source-backing)
describes why those are separate contracts.

## Deployment paths

The existing case fixtures distinguish these paths. Exact admission also checks
queue capabilities, cache operations, backing roles and compiled program ABI.
The target name alone does not select a valid recipe.

| Deployment | Corpus and native path | Backing and lifetime distinctions |
| --- | --- | --- |
| Linux RDNA | PM4, AQL and SDMA USER queues; both compute/SDMA recipes. | SYSTEM cases work independently of a LOCAL heap. LOCAL cases require that heap's advertised backing contract. |
| Linux CDNA | AQL and SDMA USER queues; SDMA/AQL recipes. | PM4 compute is outside the required matrix. Fixed-function AQL carriers account for single- and multi-XCC topologies. |
| Windows RDNA | PM4 KERNEL queues; SDMA and PM4/SDMA recipes require an advertised HWS SDMA family. | Ordinary command cases use executable command buffers and native submission retirement. SWS SDMA submission, mapped USER queue state and AQL are unavailable services. |
| Linux RDNA + XDNA | GPU/XDNA recipes, USER PM4 and native XDNA kernel submissions. | Joint allocation and registration select their actual common memory contract. Caller-page registration requires PROCESS lifetime on this KFD path. |
| Windows RDNA + XDNA | GPU/XDNA recipes, KERNEL PM4 and native XDNA kernel submissions. | Joint registration can use either lifetime. A missing common allocation/export route is not substituted with registration. |
| Linux NPU5 / Windows NPU4 | CPU/XDNA recipes through native kernel submissions. | Allocation and registration are independently selected from live capabilities. Both image profiles are built from the same Loom source. |

The kernel catalog selects the physical GPU image and instruction overlay;
behavioral cases retain the same input/output oracles. Native engine format
features select packet fields independently of that image. Explicit GCR cases
require USER_GCR, while scoped SDMA data commands carry system scope themselves.
Ordinary transfers bracket their payloads with GCR when that format is present.
Query-driven recipes emit the backing's returned NONE or GLOBAL transitions;
neither HOST_COHERENT nor a compiler target name substitutes for those answers.

Rectangular cases require `COPY_LINEAR_RECT`; its `EXTENDED_Z` and `WIDE`
modifiers select geometry without inspecting the compute target. The five
element-size cases each exercise a single element, a pitched volume and a
page-crossing volume in two epochs. Separate cases use the largest admitted Z
coordinate and depth count within owned 64 KiB source and destination buffers.
Every case checks the entire backing, including untouched rows, slices and
completion storage, before retiring commands. Host-only literal checks cover
the additional maximum row/slice-pitch encodings, including a positive wide
slice pitch of 2^32 elements; they do not establish native access to those
large layouts. Overlapping backing, tiled images and optional placement hints
need different cases.

Windows compilation does not establish a USER queue service or native execution
result. [Physical peer-GPU conformance](peer/README.md) has separate corpora
with explicit primary/peer selection and a two-GPU run requirement. Its SYSTEM
SDMA round trips use changing data, stable one-writer completion cells and full
backing/command checks before retirement. The shader-bearing AQL corpus covers
device-only peer-local memory with both placements and producer directions,
native signal dependencies and independently selected images and addresses. Joint
allocation and address queries remain distinct from those executed dataflows;
generic atomic RMW reach and SDMA access to another GPU's LOCAL memory require
their own cases.

Per-dispatch LDS capacity changes, additional packet fields, SDMA atomics,
general poll/cache controls, command-buffer variants,
additional physical-peer routes and hardware-counter collection need their own
native witnesses. The corresponding
[hardware reference](../../../docs/reference/amd/gpu/README.md) has a broader
semantic scope than the implemented CTS. A new HAL recipe is qualified by its
complete producer/dependency/consumer/reuse behavior, not by finding its opcode
in an encoder.

## Build and select required cases

Ordinary GPU corpora have the following process-lifetime test targets. Appending
`_instance` selects a second invocation of the same dynamic executable.

| Corpus | Bazel target |
| --- | --- |
| GPU services and USER ring capacity | `//libamdf/cts/gpu:gpu_extension_dynamic` |
| PM4 | `//libamdf/cts/gpu/pm4:pm4_dynamic` |
| SDMA | `//libamdf/cts/gpu/sdma:sdma_dynamic` |
| AQL | `//libamdf/cts/gpu/aql:aql_dynamic` |
| Single-GPU recipes | `//libamdf/cts/gpu/recipes:recipes_dynamic` |
| Peer construction and SDMA | `//libamdf/cts/gpu/peer:peer_dynamic` |
| Peer AQL | `//libamdf/cts/gpu/peer/aql:aql_dynamic` |
| Linux GPU file transfers | `//libamdf/cts/gpu/linux/io_uring:file_io_dynamic` |
| CPU/NPU recipes | `//libamdf/cts/xdna/recipes:execution_dynamic` |
| GPU/NPU recipes | `//libamdf/cts/interop/gpu/xdna/recipes:execution_dynamic` |

Shader-bearing GPU corpora require Loom's AMDGPU target/emitter; CPU/NPU recipes
require its XDNA target/emitter, and GPU/NPU recipes require both. The SDMA and
host encoding corpora have no shader compiler dependency. The
[source fixture guide](kernels/README.md) describes generated image identities.

A short RDNA PM4 execution witness supplies its exact physical target:

```sh
iree-bazel-test --config=asan \
  //libamdf/cts/gpu/pm4:pm4_dynamic \
  --test_arg="--amdf_gpu_endpoint_id=${GPU_ENDPOINT_ID}" \
  --test_arg="--amdf_gpu_target=${GPU_TARGET}" \
  --test_arg=--gtest_filter=Pm4DispatchTest.CoherentSystemPayloadChangesAcrossEpochs \
  --test_arg=--amdf_require_test=Pm4DispatchTest.CoherentSystemPayloadChangesAcrossEpochs
```

For a resident GPU/NPU path with registered backing:

```sh
iree-bazel-test --config=asan \
  //libamdf/cts/interop/gpu/xdna/recipes:execution_dynamic \
  --test_arg="--amdf_gpu_endpoint_id=${GPU_ENDPOINT_ID}" \
  --test_arg="--amdf_gpu_target=${GPU_TARGET}" \
  --test_arg=--gtest_filter=ResidentGpuXdnaTest.RegisteredCausalRoundTrip \
  --test_arg=--amdf_require_test=ResidentGpuXdnaTest.RegisteredCausalRoundTrip
```

The Windows resident path uses the same case and its own physical target. A
parameterized NPU-initiated witness has the full name
`NpuInitiated/ResidentNpuInitiatedTest.ReturnsEveryWord/Rounds17Words16`.
The same suite's `GpuOnlyAbort` and `NpuOnlyAbort` parameters exercise accepted
work draining after the startup decision without a submitted peer.

`--gtest_list_tests` lists the executable's exact names without activating a
device. Each `--amdf_require_test` names one complete case, including its
parameter suffix; it is not a wildcard. A filtered, absent, skipped, unexecuted
or failed required case fails the process. Requirements belong to one corpus
invocation, so different corpora use separate invocations and required lists.
Unfiltered discovery may contain capability skips and is not a substitute for
a required deployment witness.

The examples demonstrate individual behaviors. A qualification record retains
the full selected case list and result XML, native identity, source and binary
identity, generated image hashes, build configuration and checked cleanup.
PROCESS and INSTANCE are separate records. Neither a successful build nor an
older revision's native result certifies the current executable.
