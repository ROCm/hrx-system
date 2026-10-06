# Compiled GPU fixtures

All GPU fixture images are compiled from Loom source by the ordinary CTS
build. The same sources feed Bazel and CMake; kernel compilation and image
extraction require no LLVM libraries or tools.

## Source-built images

[transform.loom](transform.loom) computes `output[i] = input[i] * 3 + addend`
for `i < count`, with arithmetic modulo 2^32. It is consumed by
[`AqlDispatchTest.CoherentSystemPayloadChangesAcrossEpochs`](../aql/dispatch_test.cc)
and the [SDMA/AQL composition](../recipes/copy_dispatch_test.cc). They share
the cold publication fixture and exercise caller-owned executable memory,
code publication, kernargs, dispatch completion and exact output through the
public AQL queue ABI.

[transform_alternate.loom](transform_alternate.loom) changes the multiplier to
five while preserving the transform's argument and memory-access contract. Its
[executable replacement case](../aql/executable_test.cc) retains each program's
descriptor placement, resource requirements and complete fetch extent. It
uploads A → B → A to one retained code allocation sized for both programs after
each prior use has completed, with fixed inputs and arguments distinguishing the
programs. Both programs have valid bounded accesses if instruction fetch
observes the preceding image.

The [PM4 dispatch case](../pm4/dispatch_test.cc) uses the same source and
[typed argument layout](transform.h) through ordinary PM4 dispatch. Its
caller binds the emitted resource words and actual
allocation addresses directly. The caller checks the required wave32,
kernarg-pointer, group-X and local-X initial-register contract, with no
private or group storage. [PM4 dispatch contract](../../../../docs/reference/amd/gpu/pm4/dispatch.md)

[file_exchange.loom](file_exchange.loom) owns an initially empty Linux io_uring
SQ/CQ for one finite invocation. It reads a response-selected file block,
transforms its words, writes another block, and reloads that block into a
different payload window. The [native file I/O cases](../linux/io_uring/README.md)
check complete transcripts, payload guards, native error results and final
file contents. The kernel uses system release/acquire for intermediate ring
and payload handoffs; its outer PM4 stream owns startup and final completion.

[file_gather.loom](file_gather.loom) extends that native boundary to three I/O
credits, a key-based duplicate join and a delayed final reader. Two independent
streams recycle their own slots while the shared source stays live; each
consumer writes scattered file blocks and reloads into a different pool bank.
Native CQ tags identify slot and ticket, while per-request completion frontiers
let the [concurrent oracle](../linux/io_uring/file_gather_test.cc) check reuse
and error drain without assuming FIFO completion. The
[typed state and arguments](file_gather.h) define bounded transcript extents;
full journals stop issuance and drain instead of overrunning oracle storage.
Logical consumers remain in one invocation, not separate matmul workgroups.

[file_latency.loom](file_latency.loom) provides a matched native/host-relayed
measurement workload with up to four completion-driven streams. It probes
returned bytes to choose subsequent file keys, records reference-clock
intervals, and supports read-only or scattered read/write/reload chains.
The [latency oracle](../linux/io_uring/file_latency_test.cc) checks causal
results, complete final buffers and file contents, and same-shader-clock
interval bounds. Full-width endpoint samples disambiguate compact request
timestamps without assuming an epoch relationship with the driver's clock.
Its [measurement contract](../linux/io_uring/README.md#completion-driven-latency-comparison)
separates ordinary correctness runs from optimized, isolated comparisons.

[file_demand.loom](file_demand.loom) separates scheduled logical arrivals from
finite payload credits. One GPU owner hashes keys, coalesces immutable reads,
publishes native requests and retires ready readers without a batch join.
Scattered KV chains retain a credit through writeback and reload. Its journal
records offered arrival, admission, usable-data probing and final release;
generation checks prevent reusing backing before the final reader retires.
The [scheduled-demand cases](../linux/io_uring/README.md#scheduled-demand-and-bounded-backing)
offer bursts of 32 through 256 requests and compare four transports plus a
paced native wake policy. A separately submitted instance can perform
independent arithmetic while I/O progresses. It is a bounded ownership and
latency experiment, not a model kernel or production storage scheduler.

The [GPU/XDNA shader recipe](../../interop/gpu/xdna/recipes/README.md) selects
the same transform set by physical endpoint identity and uses the selected
resources unchanged through USER or KERNEL PM4
publication. Two GPU transforms produce NPU inputs; a third consumes the NPU
output. All three use the same typed argument contract, independently checked
against every compiled variant.

Both sources take a host workload `%groups_x` and require a `64,1,1` local
shape. The workload keeps group X dynamic without becoming a device argument;
`count`, represented as a nonnegative `i32`, bounds accesses within the grid.
The shared typed ABI contains input/output addresses at byte offsets 0/8 and
32-bit count/addend at offsets 16/20. Its 24 semantic bytes occupy a
16-aligned, 32-byte host object; caller initialization owns the padding.
The [host product tests](kernel_test.cc) compare every variant's generated
argument offsets, lengths and kinds with that layout and check compiler
alignment and launch requirements.

The [build declarations](BUILD.bazel) and [CMake equivalent](CMakeLists.txt)
produce each ordinary kernel for the exact physical targets and encoding
overlays supported by the descriptor sets linked into Loom. The default build
includes the full target catalog; a target-limited compiler builds the matching
subset. Unsupported products are incompatible build targets and are absent from
the embedded set. The CTS [build rule](../../../build_tools/bazel/cts_gpu_kernel.bzl)
uses ordinary `loom_kernel_binary` compilation followed by [embed.py](embed.py)
on the actual HSACO. A small generated header declares each behavior's
immutable [kernel set](kernel.h); one generated implementation stores all
variants' code and metadata. These are build outputs, with no checked-in
binary products or manual regeneration step for the Loom programs. This path
has no LLVM tool dependency; Loom and embedding are CTS build dependencies only.

The embedder admits one self-contained AMDHSA V6 kernel, validates its ELF,
descriptor and AMDGPU MessagePack metadata, and rejects relocations,
undefined dependencies and kernel data beyond the descriptor and text. The
image preserves their relative addresses, alignment and complete text padding.
Generated immutable metadata carries argument, geometry and resource facts
and HSACO/image hashes. Text extents, entry placement and resource words come
from the compiled product. Behavioral tests select the exact physical product
once, retain its metadata through launch, and compare results with independent
CPU oracles. The host product tests retain semantic ABI and initial-register
checks without fixing compiler instruction sizes or register counts. There is
no runtime ELF parser or descriptor patching.

### Private memory, geometry and transfers

[private_roundtrip.loom](private_roundtrip.loom) initializes nine volatile private
words per workitem, then reads them in a runtime-selected permutation into
global output. The [fixed-scratch case](../aql/private_test.cc) uses its
compiler-generated frame to exercise caller-owned queue scratch across two
completed dispatches.

[geometry_ids.loom](geometry_ids.loom) records raw group XYZ, local XYZ and an epoch
token at each global position. The [geometry cases](../aql/geometry_test.cc)
change both workgroup and grid shapes while keeping the nominal flat
workgroup size at 64. Complete and partial final groups use the same image,
whose stores have no shader bounds check. Unequal axes expose swapped
coordinate interpretations; padded output pitches distinguish inactive edge
coordinates from active records.

[byte_copy_unaligned.loom](byte_copy_unaligned.loom) retains the HAL's packed
16-byte block-copy algorithm and byte tail, with exact packed-element and
tail views over explicit global buffers. The [AQL byte-copy case](../aql/byte_copy_test.cc)
uses odd source/destination offsets and changes a 15-byte tail to one byte
while keeping a complete 64-workitem group. This fixture specializes the
selected algorithm; it has no build dependency on the HAL or its copy planner.

[pattern_fill_unaligned.loom](pattern_fill_unaligned.loom) retains the HAL's
unaligned fill algorithm, with four 16-byte vectors per block, a partial block
and byte tails. The [AQL pattern-fill case](../aql/pattern_fill_test.cc) fills
an odd subspan with a four-byte pattern, then changes the pattern and shrinks
the range to a tail-only fill. The standalone launch of 64 workitems preserves
the algorithm's bounds; the HAL's exact planner geometry is a separate contract.

The private program requires a full `64,1,1` group; geometry keeps all three
counts and local sizes dynamic, with no required local shape in the compiled
metadata. Its caller chooses nominal groups of 64 workitems and checks the
compiler's maximum group size. The copy/fill sources require `64,1,1` groups
and take dynamic X/Y group counts as workload inputs. Their six device
arguments retain 36 semantic bytes; callers initialize the complete 64-byte
slot, check the compiler's rounded segment fits, and copy only typed fields.
Alignment padding never becomes another argument or uninitialized input.

## AQL publication and observation

The shared fixture copies the exact target image into coherent system memory
with GPU READ|EXECUTE access. Its allocation includes speculative fetch padding
and the compiler's target-specific prefetch extent. Before dispatch it executes
`ACQUIRE_MEM` through an AQL vendor PM4-IB packet: seven-dword COHER_CNTL on
CDNA or eight-dword GCR with the selected RDNA generation's cache fields.
It waits for that packet's native completion. This follows ROCr's
[code freezing][freeze], [cache invalidation][invalidate] and
[ExecutePM4][execute] paths. Both vendor packet fence scopes are NONE,
matching the [ExecutePM4 defaults][defaults].
The explicit cache command owns the instruction publication transition.

Ordinary dispatch and standard AND/OR barriers require the AQL queue contract.
The [epoch case](../aql/epoch_test.cc) additionally requires the optional
`AMDF_GPU_AQL_FORMAT_FEATURE_BARRIER_VALUE` capability before activation.
Its full-width masked comparison is independent of basic AQL support.

The transform payload case uses SYSTEM acquire/release scopes. Its two
completed epochs change the input, addend and count, launch 1024 workitems,
and compare every output word with an independently computed host result.
Prefix and suffix guards, inactive tail lanes and unchanged input are checked.
Code, IB, kernarg, signal and data storage remain alive until queue destruction;
completed kernargs and data are reused only after execution completion and ring
consumption have both been observed.

The [composed recipe](../recipes/copy_dispatch_test.cc) adds SDMA upload and
download, device-side dependencies, and repeated signal and queue-ring reuse.
Its coherent SYSTEM and staged LOCAL payload cases consume distinct queried
visibility policies; code, arguments and control remain in coherent SYSTEM
memory. The [completed-use replacement case](../aql/executable_test.cc)
separately exercises explicit publication at a retained executed address.
Concurrent replacement and LOCAL code upload need their own witnesses.
Cold publication alone does not demonstrate instruction refresh after a
deliberate replacement. The public
[dispatch contract](../../../../docs/reference/amd/gpu/aql/dispatch.md) distinguishes compiler
metadata, memory publication and completion ownership.

## PM4 publication and observation

The PM4 cases copy the selected target's image into a page-rounded coherent
SYSTEM allocation with READ|EXECUTE access. The allocation covers both the
entry-prefetch extent and fetch padding after the complete compiler image.
The [command profile](../pm4/encoding/profile.h) owns the native register and
cache fields used by the writer and the code allocation:

| Compiler target family | Compute/cache representation |
| --- | --- |
| GFX11.0, GFX11.5, GFX11.7 | Metadata/shared-L1 invalidation and six-bit instruction prefetch. |
| GFX12.0 | Reserved metadata/shared-L1 actions remain clear; eight-bit prefetch. |
| GFX12.5 | GC12.1 register placement, explicit GL2 scope and forward vector-to-GL2 writeback. |

Explicit CS_PARTIAL_FLUSH and the selected whole-cache GCR operations publish
code/arguments/data and release completed shader writes. The hardware
[cache field reference](../../../../docs/reference/amd/gpu/pm4/cache.md)
details the generation differences.
A separate confirmed completion marker precedes the independent full-buffer
oracle; native command-storage retirement follows before data or argument
reuse. The same PM4 stream uses USER publication on Linux and KERNEL
publication on Windows, without adding a host wait to the payload edge.
Every queue is destroyed before referenced allocations. Two completed epochs
exercise changing inputs, count and addend; they do not qualify hot code
replacement or runtime instrumentation policy.

## Resident GPU/XDNA programs

[resident_exchange.loom](resident_exchange.loom) maintains one or two paired
request/response slots through system-scope release/acquire operations. Each
slot's next request depends on its preceding actual NPU response. The GPU
records every response before returning that slot's credit. The separate
[resident_channels.loom](resident_channels.loom) program uses one slot per
independent NPU worker and carries a single causal value through A1, B's
exchanges, then A2; its runtime arguments also support the mirrored order.

Both programs use one wave32 workitem, explicit 64-byte argument layouts and
no private or group memory. The [NPU-initiated program](resident_npu_initiated.loom)
uses the same startup and terminal protocol while reversing the first producer.
Its eight arguments contain 52 semantic bytes in a padded 56-byte segment.
Each program builds for every RDNA target used by the PM4 path and supplies its
own resource fields to the [resident recipe caller](../../interop/gpu/xdna/recipes/resident_test.cc).
The programs wait for a separate startup decision and release their final
acknowledgements only after response reads and transcript writes finish.
Prestart ABORT acknowledges without accessing the request, response or
transcript allocations. The recipe checks full payloads, immutable storage,
guards and final drain; raw device-clock observations accompany each exchange.

The shared [clock module](completed_tick.loom) uses ordinary Loom template
specialization to keep the GFX11, GFX12, GFX12.5 and RDNA4m providers in one
authored source. The selected provider drains the resident program's vector
loads and stores before sampling the reference clock, then waits for the
message result. Compact request samples use the low 32 bits; full-width interval
endpoints establish their wrap bound. These are raw ticks, independent of the
caller's release/acquire visibility operations. The complete program is linked
from authored source; runtime selection never patches code.

Source admission parses every target Low fragment before template selection.
A compiler configured with only one exact descriptor therefore omits the
dependent CTS products instead of constructing a descriptor-specific source
module. The default compiler carries the four source representations and emits
the selected physical products from the same module.

## Fixed private storage

The private source keeps both nine-iteration loops volatile. Every load selects
an initialized slot within the nine-word array and contributes to global
output. The descriptor owns the exact frame requirement; the queue rounds
`private_bytes * 64` up to a 1024-byte wave allocation unit. Scratch backing
covers the queried CU count rounded up to a whole number of shader engines,
times scratch slots per CU, across all XCCs. This includes harvested-slot
padding and remains exclusive through successful queue destruction.

The case launches eight full workgroups and checks all 4608 output words per
epoch, with changed seeds/rotations, complement poison and distinct allocation
guards. Its output witnesses the private accesses that actually ran and reuse
of the same scratch backing. It does not establish execution on every physical
slot or XCC. [Fixed scratch case](../aql/private_test.cc)

## LDS exchange images

[lds_exchange.loom](lds_exchange.loom) exchanges tagged fixed group-memory
values between partner lanes in different waves. Each output pair contains the
exchanged word and a stamp identifying the global workitem position.
The [AQL](../aql/lds_test.cc) and [PM4](../pm4/lds_test.cc) cases use four
complete 128-workitem groups and compare every result against the shared
independent [host oracle](lds_exchange.h). Repeated dispatches change the seed
on the same image and queue. Completion-visible payloads are captured before
queue retirement.

The LDS product set covers every exact physical target and encoding overlay
available in the configured compiler. Each variant retains its compiled wave
size, 512 fixed LDS bytes and zero private bytes. [Kernel metadata](kernel.h) travels with the
selected image; the [host product test](kernel_test.cc) checks every variant
against the typed ABI. Physical selection preserves the gfx1250 A0 overlay
instead of treating its base code-object name as a sufficient identity. The
source uses ordinary
`buffer.alloca<workgroup>` storage. Native callers supply the compiler-declared
capacity through the AQL packet or PM4 binding, while compiler descriptors
remain immutable. These cases qualify fixed storage; changing per-dispatch LDS
capacity requires a separate fixture.
The [PM4 group-memory contract](../../../../docs/reference/amd/gpu/pm4/lds.md) describes
the launch allocation and scheduling fields.

The two semantic arguments occupy 12 bytes of the 16-byte argument segment:
output address at byte 0 and seed at 8. Callers zero the segment and copy only
the semantic bytes. The LDS base is a compiler-resolved address, with
no per-dispatch pointer argument or code patch. Callers validate generated ABI
metadata, initialize the complete argument slot, and retain the image, data
and arguments through execution completion and ring retirement. Mixed-resource
cases additionally switch between private storage and LDS on AQL, and between
the transform and LDS programs on PM4.

## Rebuilding images

The authored `.loom` file is the source of truth. Building a consuming CTS
corpus recompiles its images when the source, target profile or compiler changes.
A kernel set can also be built and inspected directly:

```sh
iree-bazel-build //libamdf/cts/gpu/kernels:lds_exchange_kernels_embed
iree-cmake-build libamdf_cts_gpu_kernels_lds_exchange_kernels_embed
```

The generated header, implementation and target `.hsaco` files live in the
build output tree. The small header declares the immutable kernel set; one
implementation owns the bytes, argument layouts, resources and content hashes
for all variants. Callers select once from the GPU endpoint identity. No generated
kernel binaries, headers or JSON records are checked into this directory.

[freeze]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_loader_context.cpp#L347-L373
[invalidate]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L3437-L3497
[execute]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp#L1613-L1762
[defaults]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/inc/amd_aql_queue.h#L225-L229
