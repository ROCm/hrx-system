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

## Obtain the exact inputs

The pinned repository has a single `model.safetensors`, 269,060,552 bytes,
with SHA-256
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
The [publisher's file metadata](https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M-Instruct/revision/12fd25f77366fa6b3b4b768ec3050bf629380bac?blobs=true)
provides that identity. The complete download below is about 272 MB, retained
once on persistent storage; it does not require a second checkpoint or a clone
of the model repository.

```sh
small_model_dir=/path/to/models/smollm2-135m-instruct
(
  set -eu
  mkdir -p "$small_model_dir"
  model_url=https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac
  for filename in config.json generation_config.json model.safetensors \
      tokenizer.json tokenizer_config.json special_tokens_map.json; do
    curl --fail --location --continue-at - \
      --output "$small_model_dir/$filename" "$model_url/$filename"
  done
  cd "$small_model_dir"
  sha256sum --check <<'CHECKSUMS'
5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c  model.safetensors
CHECKSUMS
)
```

The checkpoint checksum must report `OK`. Tokenizer configuration and special
tokens are inputs to the independent prompt/reference path, even though the
IREE tokenizer loader consumes `tokenizer.json` itself. The publisher's
configuration records Transformers 4.42.3; its
[Llama implementation](https://github.com/huggingface/transformers/blob/v4.42.3/src/transformers/models/llama/modeling_llama.py)
is a concrete mathematical reference. A reference run records the actual
framework versions, dtype and attention implementation, token IDs, and greedy
outputs. That development oracle is separate from the shipped Loom runner.

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

The pinned safetensors header was inspected directly. It stores these BF16
tensors, with the same naming pattern across layers 0 through 29:

| Tensor name | Stored shape |
| --- | --- |
| `model.embed_tokens.weight` | `[49152, 576]` |
| `model.layers.0.input_layernorm.weight`, `post_attention_layernorm.weight` | `[576]` each; the latter has the same layer prefix |
| `model.layers.0.self_attn.q_proj.weight`, `o_proj.weight` | `[576, 576]` each; the latter has the same attention prefix |
| `model.layers.0.self_attn.k_proj.weight`, `v_proj.weight` | `[192, 576]` each; the latter has the same attention prefix |
| `model.layers.0.mlp.gate_proj.weight`, `up_proj.weight` | `[1536, 576]` each; the latter has the same MLP prefix |
| `model.layers.0.mlp.down_proj.weight` | `[576, 1536]` |
| `model.norm.weight` | `[576]` |

There is no separate `lm_head.weight` in this checkpoint: the tied output uses
the embedding tensor. Dense projection storage is `[output, input]`, consumed
as `X * W^T`. An equal byte length does not establish that orientation. Qwen's
extra query/key normalization, gating, recurrent state, and quantization are
not part of this model's transformer block.

## What is reusable, and what changes

The source index, public JIT embedding, command recorder, queue/timeline owner,
and coarse VM module have no Qwen arithmetic. They are the template to reuse.
The [text runner](../text/README.md) supplies the bounded residency and HTTP
consumer. Bootstrap, model math, storage layout, chat and token policy are the
source port; they do not require a new model-specific native adapter.

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

A useful first layer caller loads the embedding plus layer-zero tensors,
embeds a short token sequence, and runs pre-attention RMS normalization,
Q/K/V projection, RoPE, causal grouped-query attention, output projection and
residual, then post-attention RMS normalization, SwiGLU and the second residual.
Its explicit inputs include token positions and cache origins; its outputs are
the hidden states and updated K/V. Comparing those intermediates locates a
wrong rotation, head mapping, or transpose before thirty layers amplify it.

Next, one real-weight transformer block runs through the actual source catalog,
live-profile JIT, parameter loading, command recording, and retained KV. An
independent reference checks both outputs and cache contents. One-token and
multi-token cases, nonzero positions, head grouping, and padded tails exercise
the hardest interpretation boundaries. Unsupported operations or unacceptable
numerical drift stop expansion at this gate with a concrete reproducer.

After that gate, the same composition covers all layers, tied output projection,
and device greedy selection. A short reference continuation establishes final
user-visible output. Two interleaved retained rows then compare against their
independent continuations, including reset and a second turn. The model's source
chat entries then establish HTTP framing and canonical history through the
same native service. Its attention-only storage plan must account explicitly
for the text ABI's resettable private-state lane; no Gated DeltaNet layout is
required, but the present materializer requires a nonempty view.

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
