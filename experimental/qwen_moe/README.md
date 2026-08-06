# Qwen MoE Loom Kernels

This experiment develops a compact library of fused Loom kernels for the hot
paths in Qwen mixture-of-experts models. The initial reference is
Qwen3-30B-A3B Q4_K_M executed by llama.cpp's Vulkan backend.

The deliverable is a kernel corpus and an integration contract, not another
model runtime. Kernels consume the packed parameter layouts exposed by GGUF,
preserve the graph boundaries needed by a GGML router, and carry their
correctness cases, benchmark declarations, launch configuration, and target
specialization in the authored Loom source.

The routed-expert baseline uses two dispatches:

1. Q4_K gate and up contractions fused through SwiGLU;
2. Q4_K or Q6_K down contraction fused with route weighting, expert reduction,
   and the residual update.

This avoids materializing gate and up projections independently or producing a
`[token, route, hidden]` down-projection tensor.

The attention corpus covers grouped-query prefill and decode. Decode includes
an exact-q128 schedule, a register-carried schedule for ordinary runtime
lengths, and a split-K schedule for long contexts. These are model-facing
alternatives selected through JIT configuration: the host does not encode
target-specific tiles, workgroup geometry, or llama.cpp's dispatch routing.

llama.cpp is the numerical and performance oracle for the original GGUF model.
Checked-in cases use small synthetic packed blocks that expose quantization
semantics directly. Full model captures and weight slices are intentionally not
repository inputs.

## Integration shape

The corpus preserves GGUF's packed Q4_K and Q6_K weight storage. Dense and
routed contractions either consume F32 activations directly or consume the
GGML-compatible Q8_1 x4 transient format. No persistent weight repack is part of
the contract.

The principal decode dataflow through one transformer layer is:

```text
residual
   |-- RMSNorm + Q8_1 x4 ------------------------------.
   |                                                    |
   |   Q/K/V contraction -> head postprocess -> attention
   |                                                    |
   `-> router projection + top-8 -> gate/up + SwiGLU -> down
                                                        |
                          route weighting + reduction + residual
                                                        |
                              next RMSNorm + Q8_1 x4 ---'
```

Prefill substitutes token-row and expert-grouped schedules at the same semantic
boundaries. Fused `*_next_q8` roots publish the transient activation needed by
the following contraction rather than materializing an intermediate F32 row and
quantizing it in another dispatch.

The host supplies semantic dimensions and storage bindings. `model_config.loom`
owns model-wide facts such as hidden size, head geometry, route count, and RMS
epsilon. Source modules own target selection, tiles, workgroup geometry, and
schedule alternatives. A host integration should select a semantic kernel root;
it should not duplicate those authored launch decisions.

## Kernel harvest map

The table identifies the production boundaries to harvest into another graph
runtime. Names beginning with `@` are public `kernel.def` roots. Reference and
inspection roots used only by checked cases are omitted.

| Model boundary | Source | Principal roots and contract |
| --- | --- | --- |
| GGML Q8_1 x4 packing | `kernels/ggml/quantize_q8_1_x4.loom` | `@ggml_quantize_q8_1_x4_f32`; packs F32 rows into the transient layout shared with GGML matrix kernels. |
| Standalone Q6_K contraction | `kernels/ggml/linear_q6k_f32.loom`, `kernels/ggml/linear_q6k_q8_1_x4.loom` | `@ggml_linear_q6k_f32_wave32`, `@ggml_linear_q6k_f32_wave64`, `@ggml_linear_q6k_q8_1_x4`, and `@ggml_linear_q6k_q8_1_x4_wave64_block4`; direct packed-weight schedules for F32 or Q8_1 x4 activations, including the gfx1151 vocabulary schedule. |
| RMSNorm and activation preparation | `kernels/qwen3_moe/attention_prepare_quantized.loom` | `@qwen3_moe_rmsnorm_f32`, `@qwen3_moe_attention_rmsnorm_quantize_q8_1_x4`, and `@qwen3_moe_rmsnorm_f32_quantize_q8_1_x4`; the fused roots avoid rereading the residual row. |
| Router projection and top-8 | `kernels/qwen3_moe/router_projection_f32.loom`, `kernels/qwen3_moe/router_top8_f32.loom`, `kernels/qwen3_moe/router_projection_top8_fused_f32.loom` | One-row decode and four-row projection schedules, deterministic normalized top-8 selection, and `@qwen3_moe_router_projection_top8_fused_decode_f32` for the dispatch-contracted decode boundary. |
| Expert assignment metadata | `kernels/qwen3_moe/routed_gate_up_swiglu_q4k.loom`, `kernels/qwen3_moe/expert_table_partition_fused.loom` | `@qwen3_moe_build_expert_table`, `@qwen3_moe_build_expert_partition_table`, and the Prefill-512 fused table/partition root; converts token routes into expert-grouped work. |
| Routed gate/up and SwiGLU | `kernels/qwen3_moe/routed_gate_up_swiglu_q4k.loom`, `kernels/qwen3_moe/routed_linear_q4k_f16_wmma.loom` | Direct Q4_K decode roots, grouped Q8_1 x4 roots, a fused next-Q8 root, and FP16-WMMA prefill roots. Gate and up share activation traffic and publish only the SwiGLU result. |
| Routed down and residual | `kernels/qwen3_moe/routed_down_q4k.loom`, `kernels/qwen3_moe/routed_down_q6k.loom`, `kernels/qwen3_moe/routed_down_quantized_f16_wmma.loom` | Direct Q4_K/Q6_K decode roots and expert-grouped WMMA prefill roots. The decode roots fuse route weighting, top-8 reduction, and residual publication. |
| Next-layer preparation | `kernels/qwen3_moe/routed_down_next_q8.loom`, `kernels/qwen3_moe/routed_down_weighted_reduce_next_rmsnorm_f32.loom`, `kernels/qwen3_moe/routed_down_weighted_reduce_next_rmsnorm_q8_1_x4.loom` | Shared completion protocol plus fused reduction/residual/RMSNorm roots, optionally publishing next-layer Q8_1 x4. |
| Dense attention projections | `kernels/qwen3_moe/dense_linear_quantized_f16_wmma.loom`, `kernels/qwen3_moe/attention_qkv_quantized.loom`, `kernels/qwen3_moe/attention_qkv_same_format_prefill.loom` | Direct Q4_K/Q6_K dense roots, co-scheduled Q/K/V decode, and fixed-shape Q4_K Prefill-512 aggregation. Mixed Q4_K/Q6_K value storage is selected at JIT time. |
| Q/K/V head publication | `kernels/qwen3_moe/attention_postprocess_f32_f16.loom`, `kernels/qwen3_moe/attention_qkv_postprocess_fused.loom` | Applies Q/K RMSNorm, NeoX rotary embedding, and direct F16 KV-cache publication; the decode root contracts projection and postprocessing through last-arrival completion. |
| Prefill attention | `kernels/qwen3_moe/flash_attention_f32_f16_wmma.loom` | `@qwen3_moe_flash_attention_f32_f16_wmma`; grouped-query causal FlashAttention over F32 query and F16 KV storage. |
| Decode attention | `kernels/qwen3_moe/flash_attention_decode_f32_f16_wmma.loom`, `kernels/qwen3_moe/flash_attention_decode_q128_f32_f16_wmma.loom`, `kernels/qwen3_moe/flash_attention_decode_split_f32_f16_wmma.loom` | Register-carried ordinary decode, exact-q128, and long-context split schedules. Split completion can directly publish next-Q8. |
| Batched decode research | `kernels/qwen3_moe/batched_decode_expert_dispatch.loom`, `kernels/qwen3_moe/batched_decode_gate_up_q4k.loom` | GPU-resident expert queues and a two-row grouped consumer. These roots study multi-session route coalescing and are not required by the batch-one layer path. |

`func.template` bodies are reusable implementation units, not host-visible entry
points. `check.case` operations carry numerical and access-sanitizer oracles;
`check.benchmark` operations carry workload shapes and binding plans. Consumers
should link the source modules they need with `model_config.loom` and their
declared dependencies instead of copying template bodies into new files.

## Target and qualification contract

The production corpus is authored primarily for `gfx11-generic`, with explicit
wave32 or wave64 providers where the ownership schedule requires one. Router
projection additionally contains a `gfx1151` storage specialization. That
single provider does not imply that the complete model path is qualified on
gfx1151.

Each linked source boundary has a host-only plan test and manual AMDGPU
differential or access-sanitizer coverage in `kernels/BUILD.bazel`. The current
corpus is qualified end to end at the kernel-test boundary on gfx1100. New
targets and model geometries should retain the semantic roots and add providers
only when an interleaved production-shaped benchmark demonstrates a distinct
schedule requirement.

The standalone owned runtime's
[kernel integration map](../qwen/KERNEL_INTEGRATION.md) records the exact
prefill/decode routing, selective-link and HAL export boundary, fixed GGUF
layout, and the transient compiler-fact workarounds required by the current
end-to-end vertical slice.
