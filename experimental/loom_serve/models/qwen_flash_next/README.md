# Qwen3.8 Flash-Next

This package runs all 48 text layers of Qwen3.8 Flash-Next through `loom-serve`.
The model traversal, original-format weight reads, recurrent state, sparse
attention, and sampling are Loom programs. The shared server supplies hybrid
batch scheduling, physical cache pages, retained endpoints, and HTTP ownership.
There is no model-specific native runner or separate prefill/decode execution
path.

## Assets and deployment

Supply the original Unsloth `UD-Q4_K_XL` text checkpoint and its matching
`tokenizer.json`. The `weights` argument is the directory containing these four
files:

```text
Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
```

Preparation explicitly declares all four files in one parameter domain. The
first contains metadata, and the other three contain tensor payloads. All
specialized batch shapes share that domain. Renaming a differently quantized or
modified model does not make its tensor layouts compatible. Vision is separate;
the optional MTP predictor uses the additional asset described below.

The parameter bank occupies approximately **103.69 GiB**, including the
**26.82 GiB** IQ4_NL n-gram table. Original packed weights remain packed; there
is no whole-model dequantized copy. A 192 GB unified-memory machine has room for
the resident model, cache pools, recurrent endpoints, and working storage.
Account for the device's actual memory budget and other applications when
choosing capacities.

From the repository root, create a deployment document that preserves the source
provider list:

```sh
flash_weights=/path/to/UD-Q4_K_XL
flash_tokenizer=/path/to/tokenizer.json
jq --arg directory "$PWD/experimental/loom_serve/models/qwen_flash_next" \
   --arg weights "$flash_weights" --arg tokenizer "$flash_tokenizer" \
   '.base_directory = $directory | .definition += {
      weights: $weights, tokenizer: $tokenizer
    }' experimental/loom_serve/models/qwen_flash_next/model.json \
   > /tmp/flash-next-deployment.json
```

After building the [supported server](../../README.md#build-and-load):

```sh
bazel-bin/loom/src/loom/tools/loom-serve/loom-serve \
  --device=amdgpu --listen=127.0.0.1:8080 \
  --model=/tmp/flash-next-deployment.json

# From another terminal:
curl --fail --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Loom-Session: example' \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"What is 2+2?"}],"enable_thinking":false,"temperature":0,"max_tokens":64}'
```

Registration compiles the source programs; parameter activation loads the
checkpoint on first use. `/v1/status` distinguishes those operations. Reusing
`X-Loom-Session` with the complete matching conversation retains model state.
The shared [Qwen chat policy](../qwen/README.md#device-sampling) supplies thinking,
tool formatting, and device sampling, including `definition.options.sampling`.

Copy this directory together with `../qwen/` and `../../motifs/`,
preserving their relative layout. Model changes do not require rebuilding the
server.

## Multi-token prediction

The optional predictor uses the original `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`
sidecar. Enable three recursive proposals by extending the deployment document:

```sh
flash_draft=/path/to/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
jq --arg draft "$flash_draft" \
   '.definition.draft_depth = 3 |
    .definition.options.draft_weights = $draft' \
   /tmp/flash-next-deployment.json > /tmp/flash-next-mtp-deployment.json
```

Use that document with `--model`. The sidecar's embedding and vocabulary tensors
are identical to the target's tensors and share their device allocations. Only
2,775,621,632 additional parameter bytes are needed; allocation rounding brings
the observed total parameter commitment to approximately **106.31 GiB**. The
predictor reads its Q8_0 expert and injection weights directly.

The predictor proposes greedily, and the target verifies under the request's
sampling policy. Accepted prefixes publish target state and refresh the
predictor with accepted target hidden rows. Rejected proposals remain outside
the committed cache frontier. Verification retains compact transition operands
and replays the accepted prefix through the existing recurrent kernels; it does
not copy a complete recurrent endpoint for each candidate. Predictor carry adds
21,248 bytes per retained owner, and predictor cache adds 2,112 bytes per pooled
token, plus a 768-byte working tail per execution row.

The deployment uses `continuation_epochs: 1`. MTP shares the hybrid scheduler's
mixed prompt and generated spans, checkpoint forks, EOS, and cancellation.

The default remains `draft_depth: 0`: the gain depends on accepted proposals
repaying proposal, verification, and catch-up work. Compare the same prompts,
sampling policy, output budget, and retained state with and without MTP. Server
epoch records expose proposed and accepted draft-input counts independently of
selected output tokens.

## Packed execution and retained state

Every epoch carries one immutable table of active token spans. Prompt chunks,
single-token continuations, and independent sessions use the same program.
`prefill_capacity` bounds an epoch, not a prompt: longer inputs are chunked.
The package requires pooled cache storage, a context capacity divisible by 64,
at most 512 tokens per epoch, and draft depth zero or three.

The default document provides four execution rows, a 16K per-session context
ceiling, a shared 32K-token page pool, 32-token epochs, and four checkpoint
slots. KV storage is 25,344 bytes per pool token, including compressed sparse
attention indexes. A recurrent endpoint is 115,651,840 bytes: 36 GDN states,
PLE token and convolution history, and incomplete sparse-index blocks. Endpoint
capacity therefore matters independently of KV capacity.

Pinned readers and private writers have distinct recurrent origins. At a fork,
the first token reads its immutable anchor; subsequent tokens read the writer.
Only that writer is updated. Main K/V and index keys use the scheduler's physical
page IDs. Partial four-token index blocks remain with the recurrent endpoint,
so a fork at an arbitrary token boundary preserves both histories. Fresh spans
synthesize empty state instead of depending on cleared recurrent storage.

## Source map

| Source | Responsibility |
|---|---|
| `prepare.loom`, `weights.loom` | Stage specialization, explicit checkpoint files, and storage geometry. |
| `../qwen/control.loom` | Chat policy, sampling policy, epoch submission, and feedback publication. |
| `epoch.loom` | Four residual streams through 48 layers, including PLE at layer 1 and QSA every fourth layer. |
| `propose.loom`, `verify.loom`, `programs/mtp.loom` | Recursive proposals, target verification, and accepted-prefix predictor catch-up. |
| `programs/` | Hyperconnection, recurrent attention, sparse attention, routed/shared experts, and output-head composition. |
| `kernels/gdn.loom`, `kernels/work.loom` | Recurrent updates and immutable span ownership. |
| `kernels/speculation.loom`, `kernels/mtp_state.loom` | Compact target transition capture, accepted-prefix replay, and predictor carry ownership. |
| `kernels/qsa_*.loom`, `kernels/pages.loom` | Paged K/V, four-token compression, exact top-512 block selection, and causal attention. |
| `kernels/ple_*.loom`, `kernels/iq4.loom` | EOS-aware bigram/trigram hashes, IQ4_NL reads, and dilated convolution history. |
| `kernels/quantized.loom`, `kernels/experts.loom` | Q4_K, Q5_K, Q5_1, and Q8_0 expert projections in the checkpoint's actual encodings. |

The text model uses 36 GDN layers and 12 sparse-attention layers, with 512
experts and ten selected experts plus a shared expert in each layer. Dense
projections and embeddings use Q8_0; index projections use BF16. Accumulation is
F32, with explicit BF16 activation publication boundaries. The output head
projects only rows that require token selection.

## Qualification

The checkpoint-free kernel suite uses independent scalar arithmetic and
ownership checks, including partitioned recurrent execution, shared-anchor
forks, incomplete index blocks, EOS history, inactive storage guards, IQ4_NL
decoding, and stable top-512 selection:

```sh
iree-bazel-test --config=asan //experimental/loom_serve/models/qwen_flash_next/...
```

The real-checkpoint service check exercises useful text, concurrent branches,
retained continuation, fresh replay, and weight reactivation through HTTP:

```sh
PYTHONPATH=loom/src/loom/serve/testing python3 -B \
  -m experimental.loom_serve.models.qwen_flash_next.check_service \
  --url=http://127.0.0.1:8080 --model=flash-next
```

Run it against a dedicated model with at least two rows, a 256-token context,
and one checkpoint slot. It temporarily drains and reactivates that model. This
is a correctness check, not a performance benchmark or broad language-quality
evaluation. Numerical differences near an expert-selection cutoff can change
the routed expert set; full-model evaluation must measure output distributions
and task quality in addition to individual-kernel error.
