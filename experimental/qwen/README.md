# Qwen3 MoE Owned C Runtime

This experiment runs the fixed Qwen3-30B-A3B-Q4_K_M model directly through
IREE HAL and JIT-compiled Loom kernels. It owns the complete model lifecycle:
GGUF parameter indexing and gathering, resident device storage, kernel
specialization, reusable command-buffer recording, request state, queue
submission, correctness witnesses, a generation CLI, and a separate Google
Benchmark runner.

The implementation is intentionally model-specific. The tensor schema, layer
count, layouts, and kernel schedule are compile-time contracts rather than a
generic graph or model format. Raw F32, Q4_K, and Q6_K parameter payloads are
gathered into one deterministic resident slab without repacking. Kernel
identity is independent of weight identity: all 48 layers reuse prepared
executables and select their weights through recorded slab subranges.

`experimental/qwen_moe` remains the source of truth for authored Loom kernels.
Qwen-owned declaration modules add the runtime HAL export contracts, and the
runtime embeds the resulting selectively linked source. `experimental/id4` is
reference evidence only; this package has no source, build, or ABI dependency
on it.

See [KERNEL_INTEGRATION.md](KERNEL_INTEGRATION.md) for the exact authored-source
map, prefill/decode routing, JIT and export boundary, and the compiler facts
exposed by each transient bring-up workaround.

The first completed integration milestone was one complete layer-0 prefill-512
execution:

1. validate and gather the real GGUF parameter set;
2. specialize and load the required Loom exports for the live HAL device;
3. record the attention and routed-expert schedule into one reusable indirect
   command buffer;
4. submit through explicit queue dependencies and compare captured semantic
   outputs; and
5. exercise that same runtime path from a CLI smoke and a filterable
   `Qwen/Layer0/Prefill/512` benchmark row.

## Temporary bring-up workarounds

`kernels/flash_attention_bringup_workaround.py` is an explicitly
non-sanctioned unblocker, not a Loom kernel generator or an alternate authoring
path. It patches exact text in the selectively linked FlashAttention module for
two preserved upstream defects: insufficient relational facts for bounded
subtraction and a tail allocation that violates `buffer.alloca`'s fixed-frame
contract. The patch rejects any source drift instead of trying to understand or
regenerate Loom.

Nothing else should depend on this tool or copy its mechanism. Delete the tool,
its test, and the linked-module interception as soon as the unmodified
upstream module compiles. Until then, performance results must identify the
workaround's 23,808-byte LDS frame rather than treating that resource shape as
the intended zero-tail kernel.

`kernels/router_projection_f32_bringup_workaround.loom` is a separate,
one-function fork for a counted-loop footprint-analysis defect. It changes only
the four-row router's exclusive channel bound and carries only the target and
config declarations required to verify that function. Delete it when the
compiler can prove the original stepped loop. It is not a new production kernel
variant and must not accumulate unrelated router work.

`kernels/router_top8_f32_bringup_workaround.loom` makes the owned model's
compact route-ID rows structural while the generic authored kernel omits its
required `route_count <= route_id_stride` relation. The Qwen program fixes both
values to eight. This fork retains that exact behavior, carries one function,
and is deleted when the upstream kernel establishes and validates its generic
stride contract. The same fork temporarily selects the authored wide
four-subgroup geometry because source-to-low compilation cannot see the
workload-evaluated decode-or-prefill launch choice. Prefill measurements remain
representative of that geometry; decode measurements do not until the fork is
deleted.

`kernels/router_projection_top8_fused_bringup_workaround.loom` is the fused
decode router's one-function instance of the same route-stride defect. It
retains the canonical projection, device-scope last-arrival sequence, and
shared top-8 row helper, but passes the configured route count as that helper's
physical stride because this model fixes compact `8 == 8` rows. It intentionally
ignores the public stride argument and is valid only for this owned model.
Delete it with the standalone top-8 fork when the generic helper represents
and verifies `route_count <= route_id_stride`.

`kernels/expert_table_bringup_workaround.loom` makes Qwen's fixed route count
of eight structural in expert-assignment enumeration. AMDGPU lowering requires
an exact divisor, but the compile boundary currently loses the exact workload
value already used during launch evaluation. This one-function fork is deleted
when workload specialization reaches target lowering.

`kernels/attention_metadata_bringup_workaround.loom` is a temporary
one-function device-state producer while the Qwen kernel corpus has no
canonical attention-metadata kernel. It derives positions, separate K/V cache
indices, and dense causal-mask bits from one compact context-base control word.
It is not a kernel framework or a generator and must not accumulate alternate
indexing policies. Delete it when the canonical producer lands.

`kernels/flash_attention_decode_split_bringup_workaround.loom` retains the
canonical split-K decode ABI and both canonical template applications. It makes
the exact context-513 range and `9 x 4 x 1` launch structural because those
workload facts do not reach template selection or topology analysis. It
contains no attention algorithm body and is deleted when the unmodified public
kernel specializes correctly.

The three direct feed-forward forks
`kernels/routed_gate_up_next_q8_bringup_workaround.loom`,
`kernels/routed_down_q4_next_q8_bringup_workaround.loom`, and
`kernels/routed_down_q6_next_q8_bringup_workaround.loom` retain the canonical
decode ABIs and device bodies while making this model's exact one-token,
eight-route dimensions structural. The gate/up variant publishes its F32
SwiGLU rows and packed Q8_1 x4 groups together; the routed-down variants
publish the following layer's normalized Q8_1 x4 row. They exist only because
workload facts and nested device-template requirements do not yet cross every
source-to-low boundary. Delete all three when those compiler boundaries are
repaired; they are not model-generated kernel variants.

`kernels/linear_q6k_q8_1_x4_bringup_workaround.loom` is the vocabulary
projection's one-row specialization. It retains the raw Q6_K by Q8_1 x4
contraction while exposing the guarded output-channel bound needed by address
planning. Its fixed vocabulary and hidden dimensions are model facts, not a
general vocabulary-kernel authoring mechanism.

The working full-model programs now record all 48 layers, embedding, final
normalization, vocabulary projection, and greedy selection into reusable
command buffers. Prefill retains the grouped F16 WMMA feed-forward route;
decode uses fused split-K attention, fused normalization and Q8_1 packing,
direct attention output with fused feed-forward F32/Q8_1 publication, fused
router projection and normalized top-8 selection, direct raw-Q4_K gate/up with
fused SwiGLU Q8_1 publication, and direct Q4_K or Q6_K down contraction with
route weighting, reduction, and residual publication fused.
The layer runner remains a first-class optimization surface so model work can
be isolated to a stable benchmark row without forking execution logic.
