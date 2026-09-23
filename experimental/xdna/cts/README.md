# Native XDNA execution coverage

`execution_test` and `execution_test_instance` exercise image loading, binding,
queue submission, readback and resource retirement through libamdf. They select
the appropriate image for the available device family.

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
