# Qwen Command Program Experiments

This standalone experiment compiles complete Qwen model schedules into
reusable Loom command programs and executes them through IREE HAL. Its current
production witness is the text-only Qwen3.8-27B UD-Q5_K_XL GGUF: 64 base
transformer blocks, one embedded MTP block, mixed Q5_K/Q6_K/Q8_0/F32 weights,
48 gated-delta-network blocks, and 16 periodic full-attention blocks.

The authored Loom program is the model. It owns parameter requirements,
specialization arguments, tensor views, transient storage, barriers, fusion,
and kernel choice. The C host is intentionally generic: it indexes a GGUF,
gathers the parameter requirements published by the compiled package, loads
the independently compiled executable products, materializes command buffers,
and binds request state at issue time.

The first implementation gate is one exact GDN layer and its recurrent-state
transition. The kernel vocabulary starts with GGML's Q8_1 x4 activation layout
and direct Q5_K/Q6_K contractions so the same physical model bytes can be used
without repacking. Full prefill, decode, tokenizer-backed chat, and MTP expand
from that vertical slice only after it matches a canonical execution oracle.

This code is experimental and is not a dependency of shipping Loom, HAL, or
other experimental runtimes.
