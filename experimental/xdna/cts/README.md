# Native XDNA execution coverage

`execution_test` and `execution_test_instance` exercise image loading, binding,
queue submission, readback and resource retirement through libamdf. They select
the appropriate image for the available device family.

## Execution ownership and invocation

The [library execution guide](../../../libamdf/docs/xdna.md) describes public
ownership and submission. The [native reference](../../../docs/reference/amd/xdna/execution.md)
describes placement, native results and program quiescence.

The adapter tests use real image parsing and caller-owned storage without a
fake native provider. CLI tests exercise host argument and file ownership
without creating a device.

The native consumer tests in this directory select the matching canonical compiler
image, allocate data and instruction backing through the public memory scopes,
and execute three different inputs through one retained native allocation. They
check all 48 integer products, immutable command bytes, native retirement and
caller-ordered teardown. A second case alternates two retained contexts across
three producer/consumer pairs with changing inputs, then releases the producer
and continues using shared data in the consumer. Providers advertising fixed
full-array backing are checked to place both contexts on the same physical
array. Each output is poisoned before its submission so missing writes cannot
pass. Both process- and instance-scoped native lifetimes use the shared CTS
device owner.

A full-width interleave prepares two contexts once and alternates three
producer/consumer pairs. The consumer reads the producer's shared-DRAM result
without a host payload operation between them. Changed inputs, poisoned output,
exact products, guard checks, and immutable instruction bytes establish that
independent commands work even when another context uses the entire array.

```sh
iree-bazel-test --config=asan //experimental/xdna/cts/...
iree-cmake-test -R '^iree/experimental/xdna/cts/'
```

These tests carry the XDNA hardware requirement and share the AMDGPU resource
group with native and interop CTS. The same sources run on Linux and Windows;
hosts without an XDNA endpoint or a matching compiler fixture report a skip.

The named execution cases make the composition boundary explicit:

| Case | Contract exercised |
| --- | --- |
| `XdnaExecutionTest.ReusesImmutableInstructionsWithChangingInputs` | Complete establishing instruction reuse with changing arithmetic inputs. |
| `XdnaExecutionTest.SharesDataAcrossIndependentContextLifetimes` | Ordinary shared backing outlives one of its context users. |
| `XdnaExecutionTest.ReestablishesStateAcrossFullWidthContextSwitches` | Complete setup across contexts unable to occupy disjoint physical columns; the case requires full-width logical contexts. |
| `XdnaConcurrentQueuesTest.RotatesProgramsAndBindingsAcrossPendingRuns` | Separate immutable program/binding ranges remain valid while runs are pending; addition consumes multiplication output. |
| `XdnaPoolVisibilityTest.ReplaysQualifiedGpuXdnaGpuTransitions` | Host-sequenced GPU ingress → NPU command → GPU egress, with a host completion join at each transition. |

These [case bodies](execution_test.cc) retain their instruction/data owners and
check output guards and cleanup. Complete-command handoff is distinct from
live-tile checkpointing, persistent placement, program-owned timer/counter
protocols and device-only cross-engine scheduling. An execution record names
the actual selected cases, image, target and native transport.

`pipeline_completion_test` compiles [owned-channel programs](testdata/pipeline_completion.loom)
for both NPU4 and NPU5 and selects the live device's image. Its seven cases cover
strided autonomous ingress, a single admission into a multi-slot channel, fixed
and rotating projected records, reverse publication order, a read held through
DMA, and loop-carried read ownership. Three generations vary input values or
output binding assignments. The [case bodies](pipeline_completion_test.cc)
check exact results, poisoned output
replacement, untouched binding bytes, guards, command immutability during
execution and native retirement. The finite publication/retirement cases return
completed prefixes before admitting more work; they do not require independent
reuse of a later slot while its predecessor remains owned.

Compiler-owned execution coverage uses the normal HAL-backed Loom tooling.
The [channel lifetime scenarios](../../../loom/src/loom/tooling/target/amd/xdna/test/channel_lifetimes.loom)
check shared-buffer route replacement and interleaved rings with a held bias.
The [signed-I4 panel scenarios](../../../loom/src/loom/tooling/target/amd/xdna/test/signed_i4_panels.loom)
compare resident matrix arithmetic against an independent scalar oracle.
Both run through `iree-test-loom --device=xdna`, with resource ownership and
completion supplied by the XDNA HAL instead of a separate native runner.

### Benchmark smoke invocation

The generated smoke test uses the XDNA hardware requirement and shared AMDGPU
resource group, like the native CTS:

```sh
iree-bazel-test --config=asan \
  //experimental/xdna/benchmarks:execution_benchmark_test
iree-cmake-test -R '^iree/experimental/xdna/benchmarks/'
```

Smoke execution checks ordinary completion and cleanup. Its duration is not a
performance result; the [benchmark account](../README.md#independent-execution-benchmarks)
defines the optimized intervals.

## Compiler execution cases

`predicate_select_npu2_test` checks selection between independently computed
predicates in full 64-element and partial 3x3 carriers. Every lane sees all
eight Boolean input triples, with different patterns in the two mask halves.
An independent conditional oracle checks selected bytes, partial-store tails,
binding guards and unchanged inputs through three establishing invocations.
It uses the explicit hardware configuration below.

`bitcast_npu2_test` imports scalar/vector C++ bit casts and executes a streamed
pipeline on Strix Halo. An independent byte oracle checks wrapping byte
arithmetic, FP8/FP16/BF16 lane permutations and 64-bit payload transport through
F64. Each of six packets covers every byte pattern in every lane. Per-word and
binding guards, unchanged input bytes and three independent submissions verify
native execution. It uses the explicit hardware configuration below.

`assembly_pack_npu2_test` imports C++ functions containing descriptor-backed
assembly, links them into a streaming pipeline, and executes on Strix Halo.
Each `vpack.x.signed` packs 128 signed bytes into 64 bytes of INT4. Alternating
saturation-on and saturation-off fragments consume the same input vectors,
exercising the compiler's handling of control-register effects. An independent
integer oracle checks both results for all 256 byte values in each of six
records, along with per-record guards, binding tails, and unchanged inputs.
This is an explicit hardware test with the same configuration as the copy
test below.

`vector_copy_npu2_test` exercises C++ import, bytecode linking, compilation and
native execution on NPU2 hardware using the Strix Halo profile. It is an explicit
hardware test (`manual` in Bazel). Its compiler and importer dependencies are
optional; the common importer package has no dependency on this native consumer.

The same C++ source supplies scalar and 512-bit vector copy controls. Two workers
process three records each, covering zero, one, four, nineteen and twenty blocks,
nonzero byte origins, and different source/destination strides. An independent
scalar oracle checks every output word, untouched gaps, a trailing binding guard
and unchanged input bytes. These checks qualify correctness, not performance.

The [entry wrapper](testdata/vector_copy.loom) supplies an explicit 64-byte
alignment contract for its packet-buffer bases before calling the imported
[C++ worker](testdata/vector_copy.cxx). The sixteen-word input header and vector
block strides preserve that alignment. This makes the guarantee visible through
the imported call and enables native 512-bit memory operations. The
[memory guide](../../../loom/docs/src/guide/buffers-views-memory.md#aligned-bases-enable-wide-transfers)
shows the same pattern for standalone buffer arguments. Adapt the assumption
when changing the buffers supplied by the embedding.

```sh
iree-bazel-build --config=asan --config=loom-importer-cxx \
  --//loom/config/target:enable=xdna --//loom/config/emit:enable=xdna \
  --//libamdf/config:enabled=true \
  //experimental/xdna/cts:vector_copy_npu2_test

benchmark-lock --label=xdna-copy -- \
  iree-bazel-test --config=asan --config=loom-importer-cxx \
  --//loom/config/target:enable=xdna --//loom/config/emit:enable=xdna \
  --//libamdf/config:enabled=true \
  //experimental/xdna/cts:vector_copy_npu2_test
```

`integer_shifts_npu2_test` checks scalar word and split-word shifts using streamed
packets. An independent integer oracle covers all legal dynamic counts, bounded
count ranges, constant word boundaries, logical and signed right shifts, and
retained high words. The 37,888 results include edge bit patterns and seeded
random inputs; binding tails and unchanged inputs are checked as well. This test
uses the same native execution path and resource lease without requiring the
C++ importer.

`integer_widening_npu2_test` checks 17-lane signed and unsigned i32-to-i64
vectors alongside a nine-lane table quantization result. The first widening
lane beyond the ordinary vector boundary forces each result into a partial
accumulator carrier while preserving the established three-threshold quantize
oracle. Rotated high-bit patterns exercise every logical lane, and an
independent scalar oracle checks sign extension, zero extension, ordered
quantization, exact stores, destination padding, binding guards and unchanged
inputs.

`read_only_data_npu2_test` checks that immutable bytes retained by a worker are
initialized in tile-local data memory before core activation. One nonzero table
word contributes to the native result; an independent scalar oracle checks that
result, the unchanged input and both binding guards. The table's core-visible
self aperture differs from its owner-local initialization address, so the case
also covers the address-space boundary between relocation and product loading.

`immutable_gather_npu2_test` checks 16, 32 and 64-bit parallel lookup from one
worker-local table. Four boundary records cover both 32-byte table blocks,
repeated and reversed indices, and ignored high index bits. BF16 and F64 values
are checked as raw bits, while packed BF16 pairs feed a dot with an independent
arithmetic oracle. The compile report proves that each native lookup plan was
selected without spills and remained within code and bank capacity; unchanged
input and output guards bracket the device results.

`transpose_npu2_test` checks ordinary High BF16 8x8 transposition as raw bit
transport. Its 1,056 packets include every 16-bit pattern and signed zeros,
subnormals, infinities and NaN payloads rotated through every lane. Independent
scalar coordinate indexing checks all output bytes, both binding guards and
unchanged inputs. Original, changed and original inputs each run through three
complete invocations in fresh processes; capacity-two rings wrap repeatedly.
The test asserts no floating-point arithmetic or timing property.

`multicast_npu2_test` distributes eight A streams to two consumers each and two
distinct B streams to eight consumers each. Sixteen workers span eight columns
and two rows, retaining separate capacity-two or capacity-three receiver rings.
Each consumes 33 records, repeatedly wrapping both input rings and its
capacity-two output ring. An input-dependent modulo-2^32 recurrence varies the
work across consumers and records; its final state contributes to every output
word. The independent integer oracle checks all 8,448 output words, unchanged
inputs and head/tail guards through three establishing invocations. This is
finite multicast/backpressure coverage, not a timing or changed-image lifecycle
test.

`temporal_fold_npu2_test` covers first-copy bits independently from subsequent
addition: a one-record 1024-element F32 fold preserves negative zero, while
three-record 1024- and 80-element folds check ordered cancellation, exactly
representable finite sums, and full and partial fragments. Eight outputs wrap
capacity-two input and output rings repeatedly. Two runtime control-flow paths
overwrite and reload the output, then return through distinct exits; only the
final contribution may enter the fold. An independent scalar F32 oracle checks
every output byte, both binding guards and unchanged input bytes. Each case uses
three complete establishing invocations through the public runner's image and
buffers. This is fold storage and ordering coverage, not general native floating
point conformance.
