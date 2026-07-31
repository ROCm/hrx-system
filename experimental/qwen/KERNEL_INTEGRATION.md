# Qwen Kernel Integration Map

This document is the shortest path from the authored Loom kernel corpus to the
working Qwen3-30B-A3B-Q4_K_M prefill and decode programs in this experiment.
It is intended for compiler and runtime engineers harvesting kernels or
reconstructing the model routing. It is not a generic GGML graph-lowering
design.

The distinction between the two experimental subtrees is important:

- `experimental/qwen_moe/kernels/` owns the canonical, editable Loom sources.
  Those files retain their mathematical contracts, launch configurations,
  correctness cases, and benchmark rows.
- `experimental/qwen/` owns one fixed-model integration. It supplies runtime
  exports, selectively links only the functions it issues, embeds the linked
  source, specializes it against a live HAL device, records the model schedule,
  and gathers unmodified GGUF payloads into resident storage.
- `experimental/qwen/kernels/*_bringup_workaround.loom` and
  `flash_attention_bringup_workaround.py` are transient diagnostic artifacts.
  They expose compiler facts or missing corpus endpoints that blocked the
  vertical slice. They are not alternate canonical kernels or a supported Loom
  generation mechanism.

## Start here

The relevant ownership boundaries are concentrated in a few files:

- [`experimental/qwen_moe/kernels/BUILD.bazel`](../qwen_moe/kernels/BUILD.bazel)
  links each authored kernel with its library dependencies and defines its
  plan, HAL, and benchmark coverage.
- [`experimental/qwen/kernels/BUILD.bazel`](kernels/BUILD.bazel) shows the exact
  selective-link roots used by the model runtime.
- [`experimental/qwen/kernels/runtime_exports.loom`](kernels/runtime_exports.loom)
  is the HAL-facing ABI added to linked runtime copies. The canonical corpus
  does not need to carry a particular embedding runtime's export policy.
- [`experimental/qwen/runtime/loom_source.c`](runtime/loom_source.c) maps stable
  runtime module paths to generated linked modules.
- [`experimental/qwen/runtime/loom_jit.c`](runtime/loom_jit.c) specializes one
  linked module, function, config set, and workload against the selected HAL
  device.
- [`experimental/qwen/runtime/program.c`](runtime/program.c) is the executable
  routing table and the exact dispatch order.
- [`experimental/qwen/runtime/parameters.c`](runtime/parameters.c) validates
  the fixed GGUF schema and maps each tensor to a subrange of one resident
  allocation.

The generated selectively linked `.loom` modules are build artifacts. They are
useful for inspecting the compiler input seen by the runtime, but the authored
files under `experimental/qwen_moe/kernels/` remain the sources to modify or
port.

## Fixed model contract

The integration intentionally replaces graph discovery with compile-time model
facts:

| Property | Value |
| --- | ---: |
| Transformer layers | 48 |
| Hidden width | 2048 |
| Query heads | 32 |
| Key/value heads | 4 |
| Head width | 128 |
| Routed experts | 128 |
| Selected experts per token | 8 |
| Expert intermediate width | 768 |
| Vocabulary rows | 151936 |
| Largest physical token chunk | 2048 |
| Largest context | 32768 |

The parameter index must contain exactly 579 known tensors. F32 normalization
and router payloads, Q4_K projections, and the model's Q6_K value/down/output
payloads are copied byte-for-byte from GGUF into a deterministic resident slab.
There is no weight repacking. Each recorded dispatch binds the slab subrange
for its layer and tensor, so all 48 layers reuse the same compiled executable
specializations.

## Current full-model routing

The runtime records embedding, attention metadata, 48 transformer layers,
final RMSNorm, Q6_K vocabulary projection with per-workgroup maximum
publication, and compact greedy finalization into one reusable indirect command
buffer. Explicit dispatch barriers carry producer/consumer visibility. Queue
issue allocates transient storage, initializes schedule-owned synchronization
state, executes the command buffer, deallocates the transient allocation, and
publishes completion through explicit semaphore edges.

The token-count routes below are the ones in the working prefill-512 and
decode-513 programs:

| Stage | Prefill, 512 rows | Decode, 1 row | Canonical source |
| --- | --- | --- | --- |
| Token embedding | Fixed Q4_K row decoder | Same | No corpus provider yet; owned bring-up kernel |
| Attention metadata | Positions, direct K/V cache rows, dense causal mask | Same with nonzero context base | No corpus provider yet; owned bring-up kernel |
| Attention normalization | F32 RMSNorm | Fused RMSNorm and Q8_1 x4 packing | `qwen3_moe/attention_prepare_quantized.loom` |
| Q/K/V projection | Separate Q4_K query/key and storage-selected Q4_K or Q6_K value WMMA dispatches | One Q8_1 x4 Q/K/V dispatch whose last-arriving head tiles also publish the postprocessed attention inputs | `qwen3_moe/dense_linear_quantized_f16_wmma.loom`, `qwen3_moe/attention_qkv_quantized.loom`, and `qwen3_moe/attention_qkv_postprocess_fused.loom` |
| RoPE and cache publication | One combined postprocess dispatch | Published by the fused Q/K/V projection | `qwen3_moe/attention_postprocess_f32_f16.loom` and `qwen3_moe/attention_qkv_postprocess_fused.loom` |
| FlashAttention | General grouped-query prefill kernel | Fused split-K decode kernel with last-arrival reduction | `qwen3_moe/flash_attention_f32_f16_wmma.loom` and `qwen3_moe/flash_attention_decode_split_f32_f16_wmma.loom` |
| Attention output | Q4_K F16 WMMA projection with residual accumulation | Q8_1 x4 pack followed by direct Q4_K row contraction with residual accumulation; the last-arriving output workgroup also publishes the following feed-forward normalized F32 and Q8_1 x4 rows | `qwen3_moe/dense_linear_quantized_f16_wmma.loom`, `qwen3_moe/attention_prepare_quantized.loom`, and `ggml/quantize_q8_1_x4.loom` |
| Feed-forward RMSNorm | F32 RMSNorm | Published by the fused attention-output projection | `qwen3_moe/dense_linear_quantized_f16_wmma.loom` and `qwen3_moe/attention_prepare_quantized.loom` |
| Router projection | Four-row wave32 schedule | Fused four-row wave64 projection with last-arrival top-8 selection | `qwen3_moe/router_projection_f32.loom` and `qwen3_moe/router_projection_top8_fused_f32.loom` |
| Top-8 routing | Normalized compact `[token, 8]` rows | Published by the fused projection's last-arriving wave | `qwen3_moe/router_top8_f32.loom` and `qwen3_moe/router_projection_top8_fused_f32.loom` |
| Expert tables | One fused assignment/partition publication in layers 0-46; separate terminal-row tables | None; compact route IDs directly select expert rows | `qwen3_moe/routed_gate_up_swiglu_q4k.loom` and `qwen3_moe/expert_table_partition_fused.loom` |
| Gate/up, SwiGLU, and Q8 packing | Grouped Q4_K F16 WMMA producing F32 SwiGLU rows | Raw Q4_K by Q8_1 x4 direct contraction publishing both F32 SwiGLU rows and their packed physical groups | `qwen3_moe/routed_linear_q4k_f16_wmma.loom`, `qwen3_moe/routed_gate_up_swiglu_q4k.loom`, and `ggml/quantize_q8_1_x4.loom` |
| Routed down and reduction | Grouped Q4_K or Q6_K F16 WMMA; layers 0-46 fuse weighted reduction, residual update, and next-layer attention normalization | Storage-selected direct Q4_K or Q6_K contraction with route weighting, reduction, and residual update fused; the last-arriving workgroup also publishes the next layer's normalized Q8_1 x4 row | `qwen3_moe/routed_down_quantized_f16_wmma.loom`, `qwen3_moe/routed_down_q4k.loom`, `qwen3_moe/routed_down_q6k.loom`, and `qwen3_moe/routed_down_weighted_reduce_next_rmsnorm_f32.loom` |
| Vocabulary endpoint | Fused RMSNorm/Q8_1 x4 pack, raw Q6_K contraction publishing one maximum pair per eight logits, then compact finite-logit finalization | The final routed-down dispatch publishes the normalized Q8_1 x4 row; the endpoint performs the same partial Q6_K contraction and compact finalization | `qwen3_moe/routed_down_q4k.loom`, `qwen3_moe/routed_down_q6k.loom`, `ggml/linear_q6k_q8_1_x4.loom`, and an owned partial-argmax bring-up kernel |

## GGML kernel harvest list

Both kernel directories participate in the executable, but they are not two
competing source libraries. Port the canonical functions below from
`experimental/qwen_moe/kernels/`. When the last column names a Qwen adapter,
use it to recover the exact fixed-model ABI and structural facts exercised by
the full-model run; do not carry the adapter or its workaround mechanism into
GGML.

The routed-expert families are the highest-value starting point. In the
matched Prefill-512 trace, gate/up plus routed down account for 64.394 ms of
device-time advantage over the Vulkan cuts. In Decode-513, their current
dispatches occupy 3.533 ms, or 40.6 percent of the profiled command-buffer
envelope.

| Order | Semantic cut and exported function | Canonical source | Executed adapter, if any |
| ---: | --- | --- | --- |
| 1 | Grouped prefill gate/up plus SwiGLU: `qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma` | `qwen3_moe/routed_linear_q4k_f16_wmma.loom` with `qwen3_moe/routed_gate_up_swiglu_q4k.loom` | None |
| 2 | Grouped prefill Q4_K/Q6_K down and weighted residual: `qwen3_moe_routed_down_q4k_f16_wmma_grouped`, `qwen3_moe_routed_down_q6k_f16_wmma_grouped`, and `qwen3_moe_routed_down_weighted_reduce_f16_f32` | `qwen3_moe/routed_down_quantized_f16_wmma.loom`, with the raw-layout bodies in `qwen3_moe/routed_down_q4k.loom` and `qwen3_moe/routed_down_q6k.loom` | None |
| 3 | Nonterminal prefill down reduction, residual, and next RMSNorm: `qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32` | `qwen3_moe/routed_down_weighted_reduce_next_rmsnorm_f32.loom` | None |
| 4 | Direct decode gate/up, SwiGLU, and Q8_1 x4 publication: `qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8` | `qwen3_moe/routed_gate_up_swiglu_q4k.loom` and `ggml/quantize_q8_1_x4.loom` | `routed_gate_up_next_q8_bringup_workaround.loom` |
| 5 | Direct decode Q4_K/Q6_K down, route weighting, residual, and next RMSNorm/Q8_1 x4 publication: `qwen3_moe_routed_down_q4k_q8_1_x4_next_q8` and `qwen3_moe_routed_down_q6k_q8_1_x4_next_q8` | `qwen3_moe/routed_down_q4k.loom` and `qwen3_moe/routed_down_q6k.loom` | `routed_down_q4_next_q8_bringup_workaround.loom` and `routed_down_q6_next_q8_bringup_workaround.loom` |

The exact full-model Decode-513 layer is seven dispatches. After one initial
`qwen3_moe_attention_rmsnorm_quantize_q8_1_x4` producer from
`qwen3_moe/attention_prepare_quantized.loom`, each layer issues the following
cuts in order:

| Layer order | Exported function | Canonical source | Executed adapter, if any |
| ---: | --- | --- | --- |
| 1 | `qwen3_moe_attention_qkv_postprocess_fused_decode` | `qwen3_moe/attention_qkv_postprocess_fused.loom`, using `qwen3_moe/attention_qkv_quantized.loom` and `qwen3_moe/attention_postprocess_f32_f16.loom` | None |
| 2 | `qwen3_moe_flash_attention_decode_split_f32_f16_wmma` | `qwen3_moe/flash_attention_decode_split_f32_f16_wmma.loom` | `flash_attention_decode_split_bringup_workaround.loom` carries the exact context-513 topology |
| 3 | `ggml_quantize_q8_1_x4_f32` for the attention result | `ggml/quantize_q8_1_x4.loom` | `quantize_q8_1_x4_bringup_workaround.loom` |
| 4 | `qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8`, including output residual and feed-forward RMSNorm/Q8_1 x4 publication | `qwen3_moe/dense_linear_quantized_f16_wmma.loom` and `qwen3_moe/attention_prepare_quantized.loom` | None |
| 5 | `qwen3_moe_router_projection_top8_fused_decode_f32` | `qwen3_moe/router_projection_top8_fused_f32.loom`, using `qwen3_moe/router_projection_f32.loom` and `qwen3_moe/router_top8_f32.loom` | `router_projection_top8_fused_bringup_workaround.loom` |
| 6 | `qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8` | `qwen3_moe/routed_gate_up_swiglu_q4k.loom` and `ggml/quantize_q8_1_x4.loom` | `routed_gate_up_next_q8_bringup_workaround.loom` |
| 7 | Storage-selected `qwen3_moe_routed_down_q4k_q8_1_x4_next_q8` or `qwen3_moe_routed_down_q6k_q8_1_x4_next_q8` | `qwen3_moe/routed_down_q4k.loom` or `qwen3_moe/routed_down_q6k.loom` | Matching `routed_down_*_next_q8_bringup_workaround.loom` |

To reproduce the full Prefill-512 schedule after the routed-expert cuts, add
these canonical families:

- `qwen3_moe_rmsnorm_f32` from
  `qwen3_moe/attention_prepare_quantized.loom`;
- `qwen3_moe_dense_linear_q4k_f16_wmma` and
  `qwen3_moe_dense_linear_q6k_f16_wmma` from
  `qwen3_moe/dense_linear_quantized_f16_wmma.loom` for Q/K/V and attention
  output projection;
- `qwen3_moe_attention_postprocess_f32_f16` from
  `qwen3_moe/attention_postprocess_f32_f16.loom`;
- `qwen3_moe_flash_attention_f32_f16_wmma` from
  `qwen3_moe/flash_attention_f32_f16_wmma.loom`; the current runtime applies
  `flash_attention_bringup_workaround.py`, which is not a porting source;
- `qwen3_moe_router_projection_f32_four_row_wave32` and
  `qwen3_moe_router_top8_f32` from their correspondingly named canonical
  files; the current runtime uses the correspondingly named Qwen workaround
  forks only to carry missing compiler facts; and
- `qwen3_moe_build_expert_table_partition_prefill_512` from
  `qwen3_moe/expert_table_partition_fused.loom` for the exact headline shape.

The Qwen-owned embedding, attention-metadata, and final partial-argmax kernels
are endpoint or request-policy glue rather than the source of the model-level
speedup. A GGML backend can initially retain its existing equivalents. The
canonical `ggml_linear_q6k_q8_1_x4` contraction is useful for Q6_K vocabulary
or dense projections; the owned `ggml_linear_q6k_q8_1_x4_partial_argmax`
extension deliberately changes the endpoint to compact maxima and is not
required for a full-logit GGML graph.

Several other corpus files are valid alternatives or experiments but are not
part of the quoted full-model route. In particular,
`attention_qkv_same_format_prefill.loom`,
`flash_attention_decode_f32_f16_wmma.loom`, and
`flash_attention_decode_q128_f32_f16_wmma.loom` did not produce the current
631-dispatch prefill or 341-dispatch Decode-513 measurements.

Full-model prefill records 631 dispatches. Layers 0 through 46 execute every
stage over all 512 rows. Layer 47 executes attention over all rows so its cache
is complete, then executes feed-forward and the vocabulary endpoint only for
the final row needed to select the next token. The 47 nonterminal layers fuse
routed-down reduction with next-layer attention normalization and fuse expert
assignment with partition publication. The terminal one-row feed-forward path
retains the ordinary boundaries because it has no next full layer.

Decode records 341 dispatches: two request-setup dispatches, one initial
attention normalization/Q8_1 x4 producer, seven fused attention/feed-forward
dispatches per layer, and two endpoint dispatches. Each Q/K/V dispatch also
publishes RoPE and K/V cache results. Each routed-down dispatch publishes the
normalized Q8_1 x4 row consumed by the following layer or vocabulary
projection. The reusable dispatch-only command buffer has 341
barrier-delimited segments and one terminal return; selected-token publication
does not add a transfer operation.

The endpoint stores 18,992 F32 partial logits and their 18,992 I32 vocabulary
IDs in two independently aligned spans containing 151,936 bytes. The previous
607,744-byte full vocabulary row is not materialized. This reduces complete
prefill transient storage to 65,854,912 bytes and Decode-513 transient storage
to 566,208 bytes.

Decode owns partial-maximum, partial-sum, and partial-output regions sized to
its exact context. Four split-attention counters, 48 gate/up physical-group
counters, and one shared fused-stage counter occupy one contiguous 212-byte
initialization span. Existing 256-byte span alignment absorbs the added
counters without increasing the complete transient allocation. All 48
sequential layers reuse those spans. One queue fill initializes all 53 counters
after transient allocation and signals reusable command-buffer execution; each
split-attention and gate/up dispatch returns its owned counters to zero, while
the fused attention-output, router, and routed-down dispatches sequentially
return the shared counter to zero before its next use. The direct Q8_1 x4
gate/up and fused direct down families are selected for full-model decode;
grouped F16 WMMA remains the prefill and layer-program route.

## Specialization and ABI

The host chooses semantic kernel families and passes model facts; target launch
geometry remains in Loom. `program.c` currently selects five
shape-dependent families:

- Q/K/V uses separate F32-input WMMA projections at 128 or more rows. The
  full-model one-token route fuses Q8_1 x4 Q/K/V projection with Q/K
  normalization, RoPE, and K/V cache publication; ordinary one-token layer
  programs retain separate quantized projection and postprocess dispatches.
- FlashAttention uses fused split-K execution for full-model decode and the
  general grouped-query kernel for prefill and layer programs.
- Attention output uses the direct Q8_1 x4 Q4_K contraction for one-token
  shapes and the F16 WMMA contraction for larger shapes. Full-model decode
  extends that direct contraction with last-arrival publication of the
  following feed-forward normalized F32 and Q8_1 x4 rows; one-token layer
  programs retain the ordinary direct output ABI.
- Router projection publishes normalized top-8 rows from the last arriving
  projection wave for full-model decode. Prefill and layer programs retain
  separate projection and top-8 dispatches.
- Feed-forward consumes the normalized F32 and Q8_1 rows published by attention
  output, then uses fused router selection, direct raw-Q4_K gate/up with Q8_1
  publication, and fused direct down/reduction for full-model decode. Prefill
  and layer programs retain the grouped F16 WMMA route.

Each `qwen_program_prepare_kernel` call supplies:

1. a stable embedded module path;
2. a semantic function name;
3. exact model config bindings such as hidden width, head counts, expert
   count, route count, and projection dimensions; and
4. the exact workload arguments used to evaluate launch configuration.

`runtime_exports.loom` then gives the selectively linked definition an
explicit gfx11 target, export name, scalar constant order, and buffer-binding
order. A graph integration should preserve this separation: graph routing owns
semantic buffer edges and workload values, while the authored Loom source owns
its physical target schedule. The JIT evaluates both workgroup count and size,
verifies that the static size agrees with the loaded executable's function
metadata, and then leaves static workgroup-size ownership with the executable.
Recorded dispatches carry only the evaluated workgroup count.

## What the bring-up workarounds prove

The workaround files are valuable compiler reproducers because each has a
small, named missing fact. They should be compared with the canonical source,
not copied wholesale into a second kernel corpus.

| Workaround | Canonical source or missing provider | Exact pressure exposed |
| --- | --- | --- |
| `router_projection_f32_bringup_workaround.loom` | `qwen3_moe/router_projection_f32.loom` | Counted-loop propagation does not retain the four-element packet bound through all vector loads. The fork narrows an equivalent exclusive bound and carries it to those loads. |
| `router_top8_f32_bringup_workaround.loom` | `qwen3_moe/router_top8_f32.loom` | The generic kernel lacks `route_count <= route_id_stride`, while this model has compact `8 == 8` rows. Workload-evaluated decode/prefill workgroup choice also does not reach source-to-low compilation, so the fork pins the wide 256-thread geometry. |
| `router_projection_top8_fused_bringup_workaround.loom` | `qwen3_moe/router_projection_top8_fused_f32.loom` and `qwen3_moe/router_top8_f32.loom` | The shared generic top-8 helper has the same missing route-count/physical-stride relation. The one-function fork retains the canonical projection and last-arrival schedule, then passes the configured route count as the physical stride for this model's exact compact `8 == 8` rows. |
| `expert_table_bringup_workaround.loom` | `qwen3_moe/routed_gate_up_swiglu_q4k.loom` | AMDGPU division/remainder lowering needs the exact route-count divisor. The JIT workload already knows eight, but that value does not reach target lowering, so the fork makes eight structural. |
| `quantize_q8_1_x4_bringup_workaround.loom` | `ggml/quantize_q8_1_x4.loom` | Launch and loop facts do not prove the four-element `vector.load` footprint. The fork adds the equivalent packet-end relation. |
| `routed_gate_up_next_q8_bringup_workaround.loom` | `qwen3_moe/routed_gate_up_swiglu_q4k.loom` and `ggml/quantize_q8_1_x4.loom` | Exact one-token, eight-route, 128-expert, and 768-channel workload facts do not reach scalar footprint analysis. Nested device-template requirements also prevent factoring the completion epilogue, and divergent integer comparison lowering cannot consume its workgroup reduction. The fork makes only model dimensions structural, delegates the contraction and physical-group pack to canonical bodies, and uses an exact-range floating-point comparison until those compiler gaps close. |
| `routed_down_q4_next_q8_bringup_workaround.loom` | `qwen3_moe/routed_down_q4k.loom` | Exact decode dimensions do not reach source-to-low, so the eight-iteration route loop cannot be unrolled. The one-function fork delegates contraction and residual arithmetic to the canonical body, then retains the canonical last-arrival next-row publication. |
| `routed_down_q6_next_q8_bringup_workaround.loom` | `qwen3_moe/routed_down_q6k.loom` | The same source-to-low specialization gap blocks the storage-selected Q6_K direct route. The one-function fork delegates the contraction while making only the model dimensions structural and retaining next-row publication. |
| `linear_q6k_q8_1_x4_bringup_workaround.loom` | `ggml/linear_q6k_q8_1_x4.loom` | A guarded tail store loses its channel bound during address planning. The one-row fork uses a safe masked weight channel and narrows generic maxima to hidden width 2048 and vocabulary size 151936. Its endpoint export shares that contraction and publishes one deterministic maximum pair per eight logits. |
| `flash_attention_decode_split_bringup_workaround.loom` | `qwen3_moe/flash_attention_decode_split_f32_f16_wmma.loom` | Exact context 513 does not reach reducer-template selection or launch topology analysis. The fork carries that exact range to the canonical template applications and makes only the `9 x 4 x 1` launch structural. |
| `flash_attention_bringup_workaround.py` | `qwen3_moe/flash_attention_f32_f16_wmma.loom` | Bounded subtraction facts are lost in full-tile and tail paths, and a dynamic zero-or-capacity workgroup allocation violates the fixed-frame contract. The exact-text patch expresses tails as remainders, introduces the path-local lower bound, and reserves the full 8192-byte tail stage inside the resulting 23808-byte LDS frame. |
| `token_embedding_bringup_workaround.loom` | No corpus provider yet | Fixed Q4_K GGUF row decoding into the 2048-wide F32 hidden layout. This is endpoint glue, not an assumption repair. |
| `attention_metadata_bringup_workaround.loom` | No corpus provider yet | Direct no-ring K/V indices, positions, and dense causal-mask construction from one context-base word. This is request-policy glue, not an assumption repair. |
| `greedy_argmax_bringup_workaround.loom` | No corpus provider yet | Compact finalization over 18992 finite F32/I32 maximum pairs with lowest-token tie breaking. This is endpoint glue, not an assumption repair. |

The Python FlashAttention patch is especially unsuitable for reuse. It is a
fail-on-drift source interception around two known compiler/source defects, not
a kernel generator. Its useful content is the exact before/after fact pattern
and the fixed-frame requirement.

## Building and inspecting

With an AMDGPU-enabled worktree, the host-only plan tests exercise parsing,
linking, config binding, and launch evaluation without issuing device work:

```sh
iree-bazel-test \
  //experimental/qwen_moe/kernels:router_projection_f32_plan_test \
  //experimental/qwen_moe/kernels:router_projection_top8_fused_f32_plan_test \
  //experimental/qwen_moe/kernels:attention_qkv_quantized_plan_test \
  //experimental/qwen_moe/kernels:routed_gate_up_swiglu_q4k_f16_wmma_plan_test \
  //experimental/qwen_moe/kernels:flash_attention_decode_f32_f16_wmma_plan_test
```

Build the complete canonical corpus and the two end-to-end process owners with:

```sh
iree-bazel-build \
  //experimental/qwen_moe/kernels:tests \
  //experimental/qwen/binding/cli:qwen-prefill-cli \
  //experimental/qwen/binding/benchmark:qwen-prefill-benchmark
```

The full-model smoke requires the exact GGUF model and a raw file containing
512 little-endian I32 token IDs:

```sh
iree-bazel-run //experimental/qwen/binding/cli:qwen-prefill-cli -- \
  --device=amdgpu://0 \
  --parameters=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
  --tokens=/path/to/prefill_512_i32.bin \
  --expected_token=264 \
  --decode_one \
  --expected_decode_token=<oracle-token>
```

The benchmark executable owns the same model, request, and reusable program
objects independently of the CLI:

```sh
iree-bazel-run \
  //experimental/qwen/binding/benchmark:qwen-prefill-benchmark -- \
  --device=amdgpu://0 \
  --parameters=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
  --tokens=/path/to/prefill_512_i32.bin \
  --expected_decode_token=<oracle-token> \
  --benchmark_filter='Qwen/FullModel/(Prefill/512|Decode/513)'
```

Model files and token fixtures are intentionally not repository inputs. The
authored kernel cases use bounded synthetic packed data, so individual kernels
remain inspectable and testable without those private full-model artifacts.
