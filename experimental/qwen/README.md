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

The first integration milestone is one complete layer-0 prefill-512 execution:

1. validate and gather the real GGUF parameter set;
2. specialize and load the required Loom exports for the live HAL device;
3. record the attention and routed-expert schedule into one reusable indirect
   command buffer;
4. submit through explicit queue dependencies and compare captured semantic
   outputs; and
5. exercise that same runtime path from a CLI smoke and a filterable
   `Qwen/Layer0/Prefill/512` benchmark row.

## Temporary FlashAttention bring-up workaround

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

Later milestones repeat the proven layer shape across the model, add embedding
and vocabulary projection, and introduce reusable prefill and decode programs.
The layer runner remains a first-class optimization surface so model work can
be isolated to a stable benchmark row without forking execution logic.
