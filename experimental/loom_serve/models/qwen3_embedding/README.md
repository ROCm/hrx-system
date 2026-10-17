# Qwen3-Embedding-0.6B on XDNA

This package serves normalized document and query embeddings on the Strix Halo
NPU through `loom-serve`. All 28 transformer layers, embedding and rotary
gathers, and output normalization execute in one invocation across 32 AIE2P
cores. The host handles tokenization, request scheduling, and HTTP.

Each cohort holds 128 token slots and at most 16 independent sequences. A
sequence consumes its token count rounded up to eight slots, so four 32-token
inputs or sixteen inputs of at most eight tokens share one weight pass. Inputs
longer than 128 tokens and requests exceeding one cohort are rejected. Text
tokenization includes the publisher's end token. Explicit token IDs are used
as supplied. Output dimensions can be any prefix length from 1 to 1024; the NPU
normalizes the selected prefix.

## Prepare and serve

Asset preparation downloads the pinned publisher checkpoint, packs native
BFP16EBS8 weight panels, and writes an 808,069,120-byte resident bank. It also
prepares the BF16 rotary and SiLU tables. Python and PyTorch are preparation
dependencies; the server executes the authored Loom programs directly.

```sh
uv run --script experimental/loom_serve/models/qwen3_embedding/prepare_assets.py \
  --output /srv/models/qwen3-embedding-0.6b

iree-bazel-run --config=opt //loom/src/loom/tools/loom-serve -- \
  '--device=xdna://0?columns=8' --pool_backing=fixed \
  --listen=127.0.0.1:8080 \
  --model=/srv/models/qwen3-embedding-0.6b/model.json
```

The output directory must be new. `--offline` uses the existing Hugging Face
cache. `--name` selects the deployment name. The model document is published
after the bank matches its qualified digest. `provenance.json` records the
checkpoint revision, input hashes, converter, source providers, and output hash.
The deployment refers to this source directory and the cached tokenizer, so
both remain available while the model is registered.

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-0.6b","input":["The library is open on Saturday.","A document about library hours."],"dimensions":256}'

iree-bazel-run //experimental/loom_serve/models/qwen3_embedding:check_service -- \
  --url http://127.0.0.1:8080
```

For retrieval queries, the caller supplies the publisher's instruction format,
for example `Instruct: Given a question, retrieve passages that answer it.\nQuery: When is the library open?`.
Documents use their ordinary text. The package preserves supplied text and
counts instruction tokens against the same 128-token limit.

The service checker uses a ready deployment and exercises input forms,
independent rows, row order, prefix normalization, encodings, usage counts, and
rejection boundaries. It neither downloads a checkpoint nor starts a server.

## Measured serving behavior

Warm Strix Halo measurements compare this package with the pinned Halogen
0.18.4 engine and its HTTP adapter. Both use the same texts, token counts,
1024-dimensional outputs, and persistent loopback HTTP completion boundary.
The NPU stays powered during measurement. Five warmups precede 30 requests per
shape in each round of a Loom/Halogen/Loom/Halogen/Loom sequence; the table
reports pooled medians. Tokenization and response serialization are included.

| Inputs per request | Loom | Halogen | Halogen / Loom |
| --- | ---: | ---: | ---: |
| 1 × 128 tokens | 60.4 ms | 60.3 ms | 1.00× |
| 4 × 32 tokens | 60.3 ms | 240.5 ms | 3.99× |
| 16 × 8 tokens | 60.2 ms | 641.2 ms | 10.64× |

The fixed 128-slot execution makes batching the useful throughput advantage.
Single-input latency is approximately equal; short inputs still execute the
complete cohort. These warm, isolated measurements exclude arrival queueing,
cold wakeup, and model preparation.

## Execution and ownership

`array.loom` owns physical placement, storage, FIFO fanout, and stream lengths.
Four row groups each own 32 token slots. Eight columns divide the model's
projection panels. Each column reads its 120 packed records per layer once and
fans them out to its four compute cores. Paired query/key and probability/value
roles share local storage. `protocol.loom` orders layer completion before reuse
of global state. The worker modules split ingress, query/key, value, and dense
epilogue ownership; those distinctions also keep every core within its code
and SRAM limits.

`tile_math.loom` owns the native packing/MAC register interface and BF16
arithmetic. Products use multiple independent accumulators with load/MAC
overlap. Residual limbs compensate activation packing and selected BF16
arithmetic. Softmax retains the qualified exponential calculation; SiLU uses
native vector table loads. The complete model has no register spills, with a
maximum of 16,342 code bytes and 65,440 local-memory bytes per core.

The command program binds one immutable parameter bank and allocates one
2,924,544-byte mutable state buffer. The control program constructs an
8,192-byte token/sequence/position/dimension buffer and selects each sequence's
last token from the 524,288-byte output. Different sequences have independent
attention masks and rotary positions. Every global binding starts at offset
zero. Packed weight records retain 64 bytes of required FIFO lookahead.

The parameter tensor has this fixed byte layout:

| Region | Offset | Bytes |
| --- | ---: | ---: |
| Weights `[8 columns, 28 layers, 120 records, 18496 bytes]` | 0 | 497172480 |
| Per-layer Q/K and RMS scales `[28, 8192]` | 497172480 | 229376 |
| Final RMS scale block | 497401856 | 8192 |
| SiLU table | 497410048 | 8192 |
| Token embeddings `[151669, 1024]` BF16 | 497418240 | 310618112 |
| Rotary cosine/sine `[128, 128]` BF16 | 808036352 | 32768 |

The resident invocation still transfers intermediates through global memory;
invocation residency does not imply all intermediates stay on the array. The
qualified traffic model is 748,797,952 logical bytes per cohort, including
497,172,480 bytes of dense weights. Physical memory-bus traffic is not measured.
Batching amortizes this traffic over useful inputs; queueing remains part of
the serving workload's latency budget.

## Numerical qualification

The asset source is
[`Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3).
The converter checks its complete resident tensor against SHA-256
`14122288fb4a329131d3dbde969fac3d139b5c89811b56ac1cc245bbf5e66579`.
The numerical contract is approximate agreement with the publisher model;
BFP weight packing and BF16 arithmetic change the output bits. Deployment
qualification includes independent retrieval judgments, publisher token IDs,
full-model output comparison, reordered and individual/cohort equivalence,
and cancellation and retirement through the real HTTP service.
