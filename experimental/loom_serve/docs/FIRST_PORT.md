# Candidate for a first independent model port

**Candidate, not implemented:**
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
It is a small Apache-2.0 Llama-family text model with BF16 safetensors. A compact
instruction-following demo exercises a complete model without first inheriting
Qwen27B's recurrent layers, quantized weight formats, or MTP machinery. It is
not a proposed replacement for a capable coding model.

The proposed input revision is
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, as reported by the publisher's
[model metadata](https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M-Instruct).
The first port pins checkpoint, configuration, and tokenizer files to that same
revision. The choice is a bounded portability experiment, not an assertion that
an arbitrary Hugging Face model can already be loaded by this runner.

## Numerical contract

The publisher's [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json)
specifies:

| Property | Value |
| --- | --- |
| Architecture | `LlamaForCausalLM`, 30 layers |
| Hidden / feed-forward width | 576 / 1536 |
| Query / KV heads | 9 / 3; derived head width 64 |
| Vocabulary / maximum positions | 49,152 / 8,192 |
| Feed-forward activation | SiLU |
| RMS normalization epsilon | `1e-5` |
| RoPE | theta 100,000; non-interleaved; no scaling |
| Attention / MLP bias | Absent |
| Embedding/output weights | Tied |
| Checkpoint dtype / EOS | BF16 / token 2 |

The model metadata reports 134,515,008 BF16 parameters. Two-byte parameter
storage is therefore about 256.57 MiB before alignment and runtime overhead.
Derived dense two-byte KV storage is
`30 * 2 * 3 * 64 * 2 = 23,040` bytes per retained token: 180 MiB for one full
8K row, or 1.40625 GiB for eight. Even a tiny weight set can become a cache-memory
exercise. These are storage calculations, not an allocation measurement or
throughput estimate.

## What is reusable, and what changes

The source index, public JIT embedding, command recorder, queue/timeline owner,
and coarse VM module have no Qwen arithmetic. They are the template to reuse.
The model adapter and model sources are the port.

| New-model responsibility | Existing starting point and mismatch |
| --- | --- |
| Checkpoint tensor mapping | IREE already parses [safetensors](../../../runtime/src/iree/io/formats/safetensors/safetensors_parser.h); the Qwen names, encoded byte views, and mixed quantization are different |
| Dense projections | Loom's [authoring corpus](../../../loom/src/loom/test/corpus/authoring/README.md) includes BF16 and packed-dot motifs; Qwen's GGML contractions are not raw-BF16 matrix kernels |
| Attention and RoPE | Qwen's span/row separation is reusable design; dimensions, head grouping, positional convention, normalization, and cache layout must follow this model |
| Weight sharing | Tied embedding/output storage has one owner with two consumers, rather than a duplicate vocabulary matrix |
| Tokenization and prompt format | IREE tokenizer loading is reusable; this model's template and special tokens replace Qwen chat/XML-tool translation |
| Persistent state | Dense attention KV and row position only; there is no reason to carry Qwen GDN state or its speculative replay buffer |

BF16 checkpoint bytes are not F16 values with the same storage size. A first
numerical baseline can widen BF16 exactly to F32. A matrix path using BF16
storage/operands needs its actual target support and error checked; an F16
conversion needs range/error evidence from the real weights and activations.
That choice is part of the port, not an implicit reinterpretation by a loader.

## Bounded implementation sequence

The first gate inventories the pinned tensors and renders/tokenizes a short
prompt with the reference implementation. It records exact IDs, special-token
policy, projection orientation, and representative layer values. This is where
a template or tied-weight misunderstanding is cheapest to discover.

Next, one real-weight transformer block runs through the actual source catalog,
live-profile JIT, parameter loading, command recording, and retained KV. An
independent reference checks both outputs and cache contents. One-token and
multi-token cases, nonzero positions, head grouping, and padded tails exercise
the hardest interpretation boundaries. Unsupported operations or unacceptable
numerical drift stop expansion at this gate with a concrete reproducer.

After that gate, the same composition covers all layers, tied output projection,
and device greedy selection. A short reference continuation establishes final
user-visible output. Two interleaved retained rows then compare against their
independent continuations, including reset and a second turn. Only then does
the HTTP service need a model-specific chat adapter.

This model has no checkpoint MTP head. Its initial comparison is target-only.
Speculation would need a separately selected draft model and acceptance
contract; copying Qwen's `--mtp` option would not supply either.

## Readiness of the handoff

The infrastructure path is demonstrated, and the proposed model uses a modest
set of dense-transformer operations. The remaining risk is in the unimplemented
model-specific numerical/layout choices, not in a missing deployment compiler.
Near-certain reproducibility for this second model requires the real-weight
block and full continuation gates above. Until those run, Qwen27B is the
already demonstrated model and SmolLM2 is the approachable next experiment.
