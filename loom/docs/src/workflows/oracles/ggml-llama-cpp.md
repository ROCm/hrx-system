# Port GGML and llama.cpp workloads

GGML and llama.cpp provide three different kinds of evidence during a Loom
port:

- GGUF model files provide real parameter bytes and persistent tensor metadata;
- GGML source provides concrete tensor-format and operator semantics; and
- an optimized llama.cpp backend provides a production graph, selected kernels,
  and a performance reference.

Those roles remain separate. The fastest backend need not be the independent
numerical oracle, and the current GGML operator boundary need not become the
best Loom semantic cut.

## Pin model, source, and runtime independently

Begin with one exact model invocation and retain the identities another agent
needs to reproduce it:

```shell
git -C llama.cpp rev-parse HEAD
sha256sum llama.cpp/build/bin/llama-cli
sha256sum model.gguf
llama.cpp/build/bin/llama-cli --version
llama.cpp/build/bin/llama-cli --help >llama-cli.help.txt
```

The witness also records tokenizer identity, exact input bytes or token IDs,
context policy, batch and sequence dimensions, sampling policy and seed,
generated-token count, backend, loaded-library hashes, device, and reference
output. A pinned source revision does not identify a locally patched binary;
retain the patch and executable hash.

Use greedy decoding for the first end-to-end gate when the model supports it.
It removes sampling noise while preserving the complete parameter, graph,
cache, synchronization, and output path. Seeded stochastic or distributional
tests can follow under an explicit comparison policy.

## Build a model boundary ledger

Translate the reference graph into semantic rows before copying kernels. Each
row names:

| Contract | Evidence |
| --- | --- |
| Inputs and outputs | Logical tensors, shapes, element types, layouts, tolerated numerical behavior, and publication rule. |
| Parameters | GGUF tensor names, raw byte ranges, tensor types, quantization version, dimensions, strides, and alignment. |
| Reference graph | Every GGML operation and transform between the boundary tensors. |
| Selected execution | Backend pipelines, helper packing kernels, physical dispatch order, launch geometry, and intermediate formats. |
| Dynamic state | Active tokens, positions, route indices, cache rows, and other values that remain workload or device state. |
| Specialization | Model constants and target facts that genuinely change executable code. |

A semantic row may contract several GGML operations. For example, a reference
path may quantize F32 activations to a transient Q8 format and then contract
them with persistent Q4 weights. A Loom candidate may consume the same F32
activations and raw Q4 bytes directly. The comparable boundary begins before
the pack and ends after the contraction, so the reference score includes both
dispatches and their intermediate traffic.

This is the central porting freedom: GGML describes one decomposition of the
model, not the required Loom ABI or dispatch graph.

Fusion is a primary candidate whenever it removes a transient packing,
materialization, epilogue, or host-visible decision. Measure the complete
semantic cut: fewer dispatches can still lose when longer live ranges, LDS, or
barriers reduce residency. Keep a split candidate when complementary kernels
can overlap and use different machine resources; the reference graph is neither
a fusion ceiling nor a dispatch-count target.

## Treat GGUF and GGML formats as physical contracts

GGUF is the container and metadata format. GGML tensor types describe the
physical blocks stored inside it. Retain the file hash, byte order, alignment,
tensor metadata, and `general.quantization_version`; a label such as `Q4_K`
without those facts is not a complete storage identity.

The llama.cpp
[GGUF header](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h)
documents the active container implementation. The GGML
[quantization source](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
is the concrete reference for block layouts and scalar encoders or decoders at
the pinned revision. Record permalinks to the revision used by the port rather
than assuming `master` will retain the same contract.

Qualify each carrier from the bottom up:

1. Decode one hand-inspectable block with a scalar reference and a Loom motif.
2. Check exact block bytes produced by the reference encoder where encoding is
   part of the contract.
3. Decode raw blocks sliced directly from the pinned GGUF tensor.
4. Run an operator differential over multiple blocks, rows, tails, and strides.
5. Run the complete semantic cut with captured model tensors.
6. Reach a deterministic layer or model output.

The raw model bytes matter. Repacking them into an easier private layout before
the first differential can make a correct kernel for the wrong deployment
contract. A service-owned repack may still be worthwhile, but it is a separate
format, storage-cost, load-time, memory-residency, and performance decision.

## Separate motifs from compatibility kernels

Reusable Loom components naturally occupy three layers:

| Layer | Contract |
| --- | --- |
| Format motif | `func.def` or `template.def` helpers for block addressing, scale extraction, decoding, encoding, and typed interpretation; no launch ABI. |
| GGML compatibility kernel | A `kernel.def` implementing one concrete GGML operation name, bindings, output layout, and overwrite or accumulation behavior. |
| Model kernel or command program | A model-level semantic cut that composes format motifs directly and may fuse, reorder, or remove GGML operator boundaries. |

This split supports bottom-up adoption without creating islands. A runtime can
first replace one GGML kernel through its existing ABI. Later model code can
reuse the qualified format motifs without carrying the compatibility ABI or
transient packing decisions forward.

Templates specialize storage interpretation and schedules from preserved
facts. They do not require one source copy per model shape. Active token counts,
positions, routes, and tails remain workload values unless they select a small,
deliberate schedule class.

## Capture the implementation actually selected

llama.cpp can select a different backend operation, shader, or algorithm at a
single shape boundary. Capture the exact production shape and record:

- the GGML graph nodes inside the semantic cut;
- backend operation and pipeline identities;
- helper packs, copies, transposes, and epilogues;
- physical dispatch grids, workgroup and subgroup sizes, and order; and
- native programs and device timing for every included dispatch.

Temporary reference instrumentation belongs at the pipeline-selection and
dispatch callsites. Preserve its patch and build identity. The instrumented
binary establishes selection and provenance; an uninstrumented binary with the
same selection state supplies the score.

For a Vulkan backend, continue with the
[RADV oracle workflow](radv.md) to join the selected pipeline to its SPIR-V and
ACO program. The reference source kernel is insufficient when the runtime did
not select it for the witness.

An algorithm cliff in llama.cpp is evidence about that implementation. Test
both sides because they are valuable workload buckets, but do not reproduce
the threshold unless Loom's own compiler and physical evidence select it. A
single specialized Loom family may cover the entire domain more smoothly.

## Use numerical fixtures that exercise the representation

Small correctness cases use distinct, finite, nonuniform values for every
binding. Packed weights, activations, scales, routes, residuals, and output
sentinels receive different patterns so a swapped binding or omitted write
cannot pass accidentally.

Arbitrary packed bytes are not automatically finite data. In formats whose
blocks embed F16 or F32 scale metadata, an unconstrained byte pattern can create
NaNs or infinities and turn the case into a test of exceptional arithmetic.
Generate blocks with the pinned reference quantizer or choose deterministic
byte ranges whose interpreted metadata is known to be finite.

Performance fixtures obey the same rule. All-zero arithmetic operands can:

- select a zero-scale quantization branch;
- collapse nonlinear stages;
- reuse operating-system zero pages;
- make every gather read row zero; and
- hide missing writes when output storage also begins at zero.

A deterministic periodic packed fixture is useful for timing after its metadata
is qualified, but it is not a numerical oracle. Scalar differentials and raw
GGUF samples continue to own format correctness. Protocol state remains zero
only where zero is its real production initial value.

Run the ordinary Loom gates against every carrier and semantic cut:

```shell
loom-format --check kernel.loom

iree-test-loom kernel.loom \
  --case=@raw_gguf_differential \
  --device=amdgpu \
  --sanitizer=access

iree-benchmark-loom kernel.loom \
  --benchmark=@production_bucket \
  --dry-run \
  --output=plan.json
```

Access sanitization proves the lowered physical memory envelope. The scalar
reference, captured tensors, and final model output prove the mathematics.

## Match data reuse before comparing scores

A tight reference microbenchmark may execute tens of thousands of iterations
against one weight allocation. A full model may evict that projection's weights
between uses. Comparing the cache-hot reference with a Loom benchmark that
rotates a model-sized binding ring answers no causal question.

Name the intended reuse contract first:

- cache-hot latency reuses the same bindings on both sides;
- streaming latency rotates enough independent bytes to exceed the relevant
  cache on both sides; and
- model-resident latency measures the complete layer or model schedule whose
  natural accesses create the cache state.

Retain binding-ring count and bytes, parameter residency, helper dispatches,
batching, warmup, time domain, and device-completion policy in every row. The
[benchmark workflow](../benchmark.md) owns the corresponding Loom controls and
structured result fields.

## Prove a vertical model slice

The first model milestone crosses the hardest real ownership boundary and
reaches a visible result: a complete layer against captured tensors, or a
reduced model from embeddings through logits and selected token. Hundreds of
isolated operator kernels cannot prove parameter slicing, transient reuse,
queue dependencies, KV-cache state, or final publication.

Command programs keep that graph-level slice in Loom. The model root links
reusable format motifs and kernels, specializes model and target facts, and
leaves dynamic request state in the program ABI or device-visible storage. A
new model architecture then becomes mostly composition and genuinely new
motifs rather than a forked host runtime plus a copied kernel catalog.

The correctness ladder ends at deterministic model output. The performance
ladder ends at the same user-visible boundary: dispatch-device time explains
kernel mechanisms, command-buffer device time exposes gaps and overlap, queue
completion includes submission and synchronization, and first-token or token
throughput includes the application path appropriate to the claim.

## Preserve the port packet

The durable packet contains:

- reference source, patch, executable, loaded-library, model, and tokenizer
  identities;
- the semantic-boundary ledger and captured tensors;
- raw GGUF carrier cases and pinned format-source permalinks;
- the selected reference graph, pipelines, dispatches, native artifacts, and
  uninstrumented scores;
- Loom source, cases, benchmark rows, reports, artifacts, and integrated
  output; and
- rejected schedules, corrected mechanisms, and standalone compiler blockers.

That packet lets another agent port the next model by reusing qualified motifs
and experiment policy. The reusable product is not just a faster kernel; it is
the smaller body of storage, algorithm, and integration knowledge the next
architecture no longer has to rediscover.
