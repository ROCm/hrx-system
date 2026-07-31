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
final RMSNorm, Q6_K vocabulary projection, and greedy selection into one
reusable indirect command buffer. Explicit dispatch barriers carry
producer/consumer visibility. Queue issue allocates transient storage, executes
the command buffer, deallocates the transient allocation, and publishes
completion through explicit semaphore edges.

The token-count routes below are the ones in the working prefill-512 and
decode-513 programs:

| Stage | Prefill, 512 rows | Decode, 1 row | Canonical source |
| --- | --- | --- | --- |
| Token embedding | Fixed Q4_K row decoder | Same | No corpus provider yet; owned bring-up kernel |
| Attention metadata | Positions, direct K/V cache rows, dense causal mask | Same with nonzero context base | No corpus provider yet; owned bring-up kernel |
| Attention normalization | F32 RMSNorm | Fused RMSNorm and Q8_1 x4 packing | `qwen3_moe/attention_prepare_quantized.loom` |
| Q/K/V projection | Separate Q4_K query/key and storage-selected Q4_K or Q6_K value WMMA dispatches | One fused Q8_1 x4 Q/K/V dispatch with storage-selected value decoding | `qwen3_moe/dense_linear_quantized_f16_wmma.loom` and `qwen3_moe/attention_qkv_quantized.loom` |
| RoPE and cache publication | One combined postprocess dispatch | Same | `qwen3_moe/attention_postprocess_f32_f16.loom` |
| FlashAttention | General grouped-query prefill kernel | Currently the same general kernel | `qwen3_moe/flash_attention_f32_f16_wmma.loom` |
| Attention output | Q4_K F16 WMMA projection with residual accumulation | Q8_1 x4 pack followed by direct Q4_K row contraction with residual accumulation | `qwen3_moe/dense_linear_quantized_f16_wmma.loom` and `ggml/quantize_q8_1_x4.loom` |
| Feed-forward RMSNorm | F32 RMSNorm | Same | `qwen3_moe/attention_prepare_quantized.loom` |
| Router projection | Four-row wave32 schedule | Same current schedule | `qwen3_moe/router_projection_f32.loom` |
| Top-8 routing | Normalized compact `[token, 8]` rows | Same current wide launch | `qwen3_moe/router_top8_f32.loom` |
| Expert tables | Assignment table then 32-row partition table | Same | `qwen3_moe/routed_gate_up_swiglu_q4k.loom` |
| Gate/up and SwiGLU | Grouped Q4_K F16 WMMA | Currently the same grouped kernel | `qwen3_moe/routed_linear_q4k_f16_wmma.loom` |
| Routed down | Storage-selected grouped Q4_K or Q6_K F16 WMMA | Same | `qwen3_moe/routed_down_quantized_f16_wmma.loom` |
| Route reduction | F16 routed rows, F32 route weights, residual update | Same | `qwen3_moe/routed_down_quantized_f16_wmma.loom` |
| Vocabulary endpoint | Q8_1 x4 pack, raw Q6_K contraction, finite-logit argmax | Same | `ggml/quantize_q8_1_x4.loom`, `ggml/linear_q6k_q8_1_x4.loom`, and an owned argmax bring-up kernel |

Full-model prefill records 726 dispatches. Layers 0 through 46 execute every
stage over all 512 rows. Layer 47 executes attention over all rows so its cache
is complete, then executes feed-forward and the vocabulary endpoint only for
the final row needed to select the next token. Decode records 678 dispatches;
its one active row passes through all 48 layers and the endpoint.

Two route gaps are deliberately visible instead of being hidden behind a
generic fallback. The canonical corpus already contains three decode-specific
attention families:

- `qwen3_moe/flash_attention_decode_q128_f32_f16_wmma.loom` for the exact
  128-row schedule;
- `qwen3_moe/flash_attention_decode_f32_f16_wmma.loom` for ordinary context
  lengths; and
- `qwen3_moe/flash_attention_decode_split_f32_f16_wmma.loom` for long-context
  split-K execution.

The owned runtime at this revision still issues the general prefill-capable
FlashAttention kernel for decode. Likewise, its gate/up route always selects
the grouped F16 WMMA family. The corpus also contains the Q8_1 x4
`qwen3_moe_routed_gate_up_swiglu_q4k_q8` small-token schedule in
`qwen3_moe/routed_gate_up_swiglu_q4k.loom`; it is not yet selected by the
owned runtime. These are integration choices, not missing kernel sources.

## Specialization and ABI

The host chooses semantic kernel families and passes model facts; target launch
geometry remains in Loom. `program.c` currently selects only two
token-count-dependent families:

- Q/K/V uses separate F32-input WMMA projections at 128 or more rows and the
  fused Q8_1 x4 row kernel below 128 rows.
- Attention output uses the direct Q8_1 x4 Q4_K contraction for one-token
  decode and the F16 WMMA contraction for larger shapes.

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
its physical target schedule.

## What the bring-up workarounds prove

The workaround files are valuable compiler reproducers because each has a
small, named missing fact. They should be compared with the canonical source,
not copied wholesale into a second kernel corpus.

| Workaround | Canonical source or missing provider | Exact pressure exposed |
| --- | --- | --- |
| `router_projection_f32_bringup_workaround.loom` | `qwen3_moe/router_projection_f32.loom` | Counted-loop propagation does not retain the four-element packet bound through all vector loads. The fork narrows an equivalent exclusive bound and carries it to those loads. |
| `router_top8_f32_bringup_workaround.loom` | `qwen3_moe/router_top8_f32.loom` | The generic kernel lacks `route_count <= route_id_stride`, while this model has compact `8 == 8` rows. Workload-evaluated decode/prefill workgroup choice also does not reach source-to-low compilation, so the fork pins the wide 256-thread geometry. |
| `expert_table_bringup_workaround.loom` | `qwen3_moe/routed_gate_up_swiglu_q4k.loom` | AMDGPU division/remainder lowering needs the exact route-count divisor. The JIT workload already knows eight, but that value does not reach target lowering, so the fork makes eight structural. |
| `quantize_q8_1_x4_bringup_workaround.loom` | `ggml/quantize_q8_1_x4.loom` | Launch and loop facts do not prove the four-element `vector.load` footprint. The fork adds the equivalent packet-end relation. |
| `linear_q6k_q8_1_x4_bringup_workaround.loom` | `ggml/linear_q6k_q8_1_x4.loom` | A guarded tail store loses its channel bound during address planning. The fork uses a safe masked weight channel, repeats the guarded output relation, and narrows generic maxima to hidden width 2048 and vocabulary size 151936. |
| `flash_attention_bringup_workaround.py` | `qwen3_moe/flash_attention_f32_f16_wmma.loom` | Bounded subtraction facts are lost in full-tile and tail paths, and a dynamic zero-or-capacity workgroup allocation violates the fixed-frame contract. The exact-text patch expresses tails as remainders, introduces the path-local lower bound, and reserves the full 8192-byte tail stage inside the resulting 23808-byte LDS frame. |
| `token_embedding_bringup_workaround.loom` | No corpus provider yet | Fixed Q4_K GGUF row decoding into the 2048-wide F32 hidden layout. This is endpoint glue, not an assumption repair. |
| `attention_metadata_bringup_workaround.loom` | No corpus provider yet | Direct no-ring K/V indices, positions, and dense causal-mask construction from one context-base word. This is request-policy glue, not an assumption repair. |
| `greedy_argmax_bringup_workaround.loom` | No corpus provider yet | One fixed 151936-element finite F32 row with lowest-token tie breaking. This is endpoint glue, not an assumption repair. |

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
