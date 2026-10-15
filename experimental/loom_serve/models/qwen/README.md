# Qwen3.8-27B

This package runs the Qwen3.8-27B UD-Q5_K_XL GGUF through the supported
`loom-serve` text service. Preparation, chat formatting, command programs and
numerical kernels are Loom source. One prepared model and weight bank serve many
sessions; sessions own pooled state, not separate VM processes.

The deployment accepts text input. Vision numerical sources and their kernel
checks are included, but the deployment does not register a vision component
or accept images. Image input requires the separately developed batched media
service and its corresponding source-policy and device-packet integration.

## Assets and deployment

Supply `Qwen3.8-27B-UD-Q5_K_XL.gguf` and its matching `tokenizer.json`.
The weight policy expects this model's tensor keys and mixed GGML encodings;
renaming another checkpoint does not make it compatible. MTP additionally
requires the checkpoint's prediction block. Weights are read and prepared into
the model's final parameter storage during activation, then reused by requests.

From the repository root, set the two asset paths and produce one deployment
document. `jq` preserves the complete source provider list from `model.json`:

```sh
qwen_weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf
qwen_tokenizer=/path/to/tokenizer.json
jq --arg directory "$PWD/experimental/loom_serve/models/qwen" \
   --arg weights "$qwen_weights" --arg tokenizer "$qwen_tokenizer" \
   '.base_directory = $directory | .definition += {
      weights: $weights, tokenizer: $tokenizer,
      rows: 8, context_capacity: 32768, pool_capacity: 65536,
      prefill_capacity: 512, draft_depth: 3, continuation_epochs: 2
    }' experimental/loom_serve/models/qwen/model.json \
   > /tmp/qwen-deployment.json
```

After building the [supported server](../../README.md#build-and-load):

```sh
bazel-bin/loom/src/loom/tools/loom-serve/loom-serve \
  --device=amdgpu --listen=127.0.0.1:8080 \
  --model=/tmp/qwen-deployment.json

# From another terminal:
curl --fail --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Loom-Session: example' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is 2+2?"}],"enable_thinking":false,"temperature":0,"max_tokens":64}'
```

The registration name is the HTTP routing alias. Reusing `X-Loom-Session` with
the complete matching conversation allows the server to reuse retained state.
Thinking is enabled by default, using the checkpoint's xhigh reasoning
instruction. `enable_thinking: false` selects the closed, empty reasoning prefix
used in the short example above. Streamed deltas and complete messages carry
reasoning in `reasoning_content`, independently of answer `content` and tool calls.
Include that reasoning field in subsequent assistant history to retain the exact
conversation. A generation truncated inside its reasoning block remains valid
output but requires replay instead of reusing a falsely closed device prefix.
Completion also compares the canonical generated suffix with the original
decoded response. Whitespace trimming or tool-call normalization that changes
that suffix requires replay; byte-identical history remains reusable.
Tool definitions and tool-result turns use the chat endpoint;
the source policy formats and parses the model's XML tool-call protocol.

## Device sampling

Requests support `temperature`, `top_p`, `top_k` and a signed 64-bit `seed`.
Omitted values come from `definition.options.sampling`, for example:

```json
{"sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20}}
```

Without deployment overrides the defaults are greedy temperature zero,
`top_p: 1` and `top_k: 0` (unbounded vocabulary). Positive temperature applies
temperature scaling, then top-k, then nucleus selection. Top-p is in `(0,1]`;
top-k is a nonnegative signed 32-bit integer. Positive temperature and top-p
must be representable as normal F32 values. Temperature zero always retains
the greedy lowest-ID tie rule. Unsupported fields and duplicate defaults reject.

A seed fixes the random stream by absolute target-token position and token ID,
independently of packed row placement or speculative depth. An omitted seed
uses a distinct request nonce; this is not a cryptographic random source.
Floating-point differences between device kernels can still change a draw near
a probability boundary. Selection stays on device, including target sampling
inside MTP/DFlash verification. Draft proposals do not consume target randomness;
acceptance commits only the prefix matching sampled target tokens.

## Scheduling and state

`rows` permits 1–16 independently advancing execution rows. `context_capacity`
is the per-session logical ceiling; `pool_capacity` is the shared token capacity.
Short sessions do not reserve that ceiling. Admission reserves completion space,
and requests that cannot fit wait under the configured admission policy.

KV uses 64-token pages. Recurrent state has an independently pooled lifetime,
allowing retained readers and private writers to share KV prefixes without
copying an entire recurrent history before every fork. Explicit checkpoints,
additional idle endpoints and intermediate prefix caching are opt-in capacities in
the [text definition](../../../../loom/docs/src/serving/model-documents.md#text-definitions).
They retain recurrent endpoints as well as KV references, so their memory cost
is larger than a page-table entry.

`prefill_capacity` bounds the largest automatically specialized packed epoch,
currently at most 512 tokens. It does not cap prompt length: larger inputs are
chunked and can share epochs with decode and verifier work. Token capacity and
independently advancing span capacity are separate. All bucket variants share
the prepared weights and the compilation cache.

`draft_depth: 0` selects target-only execution; use `continuation_epochs: 1`.
Depth 3 selects MTP. Depth 7 selects the DFlash2 source path and requires
`definition.options.draft_weights` to name its matching GGUF draft checkpoint.
Draft weights are additional model storage. Two-epoch continuation lets selected
device output feed the next prepared epoch without a host token round trip.
These choices change the executed workload; throughput comparisons need matching
drafting and retention settings.

## Source map

`prepare.loom` declares stage roots, specialization and storage geometry;
`weights.loom` and `dflash_weights.loom` describe checkpoint preparation.
`control.loom` owns request framing, state-table encoding, submission and progress
publication. `prefill.loom`, `epoch.loom`, `mtp.loom`, `verify.loom` and
`continue.loom` compose the model traversal. `programs/qwen38/` composes layers;
`kernels/qwen38/` supplies concrete numerical leaves. Generic quantization and
projection helpers live in `../../motifs/ggml/`.

`model.json` supplies providers to Loom's linker, not an ordered list of host
operations. Configuration binds model-specific leaves; helper motifs receive
their dimensions and layout choices as SSA arguments or template parameters.

## Exercise concurrent sessions

With the optimized server running, this client sends distinct two-turn sessions
and records completed output, TTFT, token counts and prefix reuse as JSONL:

```sh
PYTHONPATH=loom/src/loom/serve/testing python3 -B \
  -m experimental.loom_serve.models.qwen.benchmark_service \
  --url=http://127.0.0.1:8080 --model=qwen --clients=8 \
  --long-lines=128 --max-tokens=64 --retention=required
```

Adding `--workload=experimental/loom_serve/models/qwen/testdata/source_review.json`
selects the bundled varied-turn corpus. `--retention=observe` records cache
misses instead of rejecting them during intentional memory-pressure experiments.
The `benchmark_cohort` tool additionally checks matching tokenized inputs and
responses across its supported HTTP profiles before reporting throughput.

Correctness checks are under `tests/` and in `request_check_test.py`; the latter
uses a small real tokenizer and the production source chat policy without loading
27B weights. `check_router.py` deploys the real model twice with different
capacities and verifies independent conversations across pressure-driven model
handoffs. The Krea guide exercises the corresponding cross-modality handoff.
