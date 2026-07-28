# Qwen MoE Loom Kernels

This experiment develops a compact library of fused Loom kernels for the hot
paths in Qwen mixture-of-experts models. The initial reference is
Qwen3-30B-A3B Q4_K_M executed by llama.cpp's Vulkan backend.

The deliverable is a kernel corpus and an integration contract, not another
model runtime. Kernels consume the packed parameter layouts exposed by GGUF,
preserve the graph boundaries needed by a GGML router, and carry their
correctness cases, benchmark declarations, launch configuration, and target
specialization in the authored Loom source.

The first production baseline uses two routed-expert dispatches:

1. Q4_K gate and up contractions fused through SwiGLU;
2. Q4_K or Q6_K down contraction fused with route weighting, expert reduction,
   and the residual update.

This avoids materializing gate and up projections independently or producing a
`[token, route, hidden]` down-projection tensor. Persistent kernels spanning
both contractions are a later experiment after the ordinary dispatch baseline
is correct and competitive.

llama.cpp is the numerical and performance oracle for the original GGUF model.
Checked-in cases use small synthetic packed blocks that expose quantization
semantics directly. Full model captures and weight slices are intentionally not
repository inputs.
