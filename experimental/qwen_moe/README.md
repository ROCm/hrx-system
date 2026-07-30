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
