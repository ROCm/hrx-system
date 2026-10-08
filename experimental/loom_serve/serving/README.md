# Serving multiple source models

`//experimental/loom_serve/serving:server` serves named text and image deployments through
one loopback HTTP listener. All deployments borrow one device, parameter
residency cache and private command workspace backing. Each owns its work and
feedback timelines, source-JIT programs and weights. Text deployments own their
logical KV pool, sessions and checkpoints; image deployments own their finite
request storage. Registering the same checkpoint twice creates two
prepared parameter residencies; it does not deduplicate those weights.

The application owns one [host](../runtime/host.h) with shared IREE task workers
and asynchronous I/O services. Each catalog's JIT queue uses those workers;
adding a model does not create another compiler thread pool. Compilation has
an asynchronous completion API; startup constructors join on the application
thread. HTTP admission and model scheduling run as a cooperative task process:
bounded serialized turns preserve shared memory-policy ownership, while CPU
preparation/output and device work proceed independently. Network, task release,
model frontiers and heartbeat deadlines wake the owner directly. Only the outer
application caller joins final scheduler release; no task waits for device work.

The catalog selects source packages and deployment options. Model math,
checkpoint interpretation, storage geometry and chat policy remain in the
Loom sources described by the [text model contract](../text/README.md) and
[image model contract](../image/model.h). Neither modality requires model-specific
C in the executable.

## Launch

Create a JSON or JSONC catalog, for example:

~~~json
{
  "models": [
    {
      "kind": "text",
      "name": "scout",
      "source": "experimental/loom_serve/models/qwen",
      "weights": "/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf",
      "tokenizer": "/path/to/tokenizer.json",
      "rows": 4,
      "context_capacity": 8192,
      "pool_capacity": 32768,
      "prefill_capacity": 128,
      "mtp_depth": 3,
      "continuation_epochs": 2,
      "checkpoint_capacity": 2
    },
    {
      "kind": "text",
      "name": "researcher",
      "source": "experimental/loom_serve/models/qwen",
      "weights": "/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf",
      "tokenizer": "/path/to/tokenizer.json",
      "rows": 2,
      "context_capacity": 131072,
      "pool_capacity": 65536,
      "prefill_capacity": 512,
      "mtp_depth": 0
    },
    {
      "kind": "image",
      "name": "illustrator",
      "source": "experimental/loom_serve/models/krea",
      "checkpoint": "/path/to/krea-2-turbo",
      "adapter": "/path/to/style.safetensors",
      "width": 384,
      "height": 384,
      "text_tokens": 512,
      "pending_requests": 4
    }
  ]
}
~~~

Paths resolve from the process working directory. Names are nonempty visible
ASCII identifiers, so the same value fits JSON requests and HTTP headers.
Unknown, duplicate and wrong-modality fields reject instead of silently changing
configuration. Image `checkpoint` is an asset directory, whereas text `weights`
is a checkpoint file; an image's optional `adapter` is interpreted by its source.
A model source can impose additional shape/storage constraints.

~~~sh
build_tools/bin/iree-bazel-run --config=opt \
  //experimental/loom_serve/serving:server -- \
  --models=/path/to/models.json --port=8080 \
  --pool_backing=elastic --memory_bytes=38654705664
~~~

The repository's build and hardware-execution policy applies. `--device`,
`--slab_bytes` and `--memory_bytes` configure the shared device. `--workers`
sets the shared CPU worker maximum (eight by default). Connections
and HTTP body storage have independent bounds through `--connections` and
`--request_body_bytes`. Each model's `pending_requests` defaults to 32 and
text `max_tokens` to 512. Image defaults are 384-square pixels and 512 text tokens.
The budget must fit each model's weights plus live retained state; this example's
36 GiB fits either model separately, not all deployments together.
Other defaults and accepted ranges are in
[configuration.c](configuration.c).

## Requests and retained state

`POST /tokenize` and `POST /detokenize` use a deployment's already loaded input
tokenizer without activating weights, taking a model row, or waiting for an
inference epoch. Independent finite CPU tasks share the application executor;
one model's device work does not hold these utilities behind its completion.
This raw-text profile uses llama.cpp's field names and defaults, plus the
required catalog `model` selector:

~~~json
{"model":"scout","content":"Hello!","add_special":false,"parse_special":true}
~~~

The response is `{"tokens":[...]}`. Detokenization accepts
`{"model":"scout","tokens":[...]}` and returns `{"content":"..."}`, using the
tokenizer's decoder and preserving special tokens. It can also be used with an
image deployment's prompt tokenizer. These operations do not apply chat
templates or model-specific prompt framing. Empty text/arrays are valid.

Each utility request allows at most 262,144 tokens and 1 MiB of decoded text,
independently of inference context capacity and the transport JSON body limit.
Limits include inserted special tokens and decoded output; exceeding one returns
413 without a truncated success. Tokens must be integer IDs present in the
selected vocabulary. Unknown/duplicate/wrongly typed fields return 400. An
unknown model or absent tokenizer returns 404; a tokenizer execution error
(including an unconfigured decoder) returns 500 for that request without
stopping other models.

The optional llama.cpp `with_pieces=true` mode is not supported: that protocol
requires raw token bytes, whereas the public tokenizer decoder combines partial
UTF-8 and replaces incomplete sequences on finalization. It returns 400 instead
of substituting vocabulary spellings or silently changing the response shape.
`with_pieces=false` is accepted. vLLM's `prompt`/`add_special_tokens` profile and
chat-tokenization fields are not aliases for this raw profile.

`GET /v1/models` lists deployment names. Chat requests use those names, not
the model source's intrinsic identity:

~~~json
{
  "model": "scout",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "stream": true,
  "max_tokens": 16,
  "temperature": 0
}
~~~

SSE responses report the selected deployment name. `X-Loom-Session` and
`X-Loom-Checkpoint` are scoped to that deployment, so two models can use the
same session/checkpoint string without sharing state. Explicit checkpoint
creation, deletion and suspension select their model with
`X-Loom-Model: scout`. That header may be omitted when exactly one deployment
is registered. Unknown model names return 404; ambiguous checkpoint selectors
return 400. `GET /healthz` does not activate a model.

Images select their deployment with the same `model` field at
`POST /v1/images/generations`:

~~~json
{
  "model": "illustrator",
  "prompt": "a red fox in the snow",
  "seed": "0",
  "strength": 0,
  "response_format": "b64_json"
}
~~~

The response contains one base64 PNG. The configured image shape is fixed;
`strength` selects the loaded adapter's weighting (zero uses base weights).
Without an adapter only the default strength of one is accepted. A text model
does not accept image requests, or vice versa.

## Scheduling and memory ownership

The transport proactor handles network I/O independently. One application
owner accepts a bounded batch of requests, then advances each service, rotating
priority. A text turn submits at most one scheduling cohort, which may contain
two device-fed epochs when configured. Model calls and residency transitions
are never concurrent host operations.

Models have independent semaphore timelines even when they use the same exact
queues. Their command allocations share HAL TLSF backing; safe reuse follows
dependency frontiers, not a device-wide execution chain. Registering larger
workspace requirements creates backing lazily without draining other models.
Older pool geometry remains accounted until its roots and queue captures retire.
Model device work is submitted without host completion waits. Successful model
frontier readiness schedules a later owner turn, and finite CPU preparation
and output work releases its task before model reuse. Cold platform memory
mapping calls still execute synchronously on the owner.

Text admission pins parameters and physically reserves its completion high-water.
Image preparation retains host input without pins while waiting for admission;
an admitted invocation pins parameters through successful model completion.
Other models retry queued admission as the owner advances. If the
budget fits only one model, another model waits until those parameter pins
release; the shared LRU can then evict idle weights and stream the requested
model back into its stable virtual roots. Retained KV and explicit checkpoints
remain model-owned. Weight eviction is not KV eviction.

`--memory_bytes` bounds retained parameter/state VMM commitment, not total
device memory. Heartbeats separately report retained commitment, workspace
commitment/live reservations and workspace reuse. Driver overhead, fixed
buffers and host snapshots are outside that budget. Text JSONL events carry
the deployment name; request/epoch counters remain local to each model.

## Verification

The router tests use actual TCP carriers and borrowed request ownership.
Configuration tests exercise parsing and partial-initialization cleanup;
source-policy tests check that deployment aliases preserve rendered prompts.
The real-weight witness is `models/qwen:check_router`, supplied with the built
shared server, source package, checkpoint, tokenizer and an empty output path.
It compares sequential and overlapping final SSE, independent same-named
sessions/checkpoints, parameter eviction/reload and shared workspace reuse
under a one-model retained budget. Its timings are not a performance baseline.

The mixed-modality witness is
[`models/krea/check_router.py`](../models/krea/check_router.py). It accepts
`--server`, `--text-source`, `--weights`, `--tokenizer`, `--image-source`,
`--checkpoint`, `--adapter`, `--isolated` and `--output`. Run it from the
repository root with `python -m experimental.loom_serve.models.krea.check_router`
on a qualified GPU runner; Pillow decodes the PNG independently. The isolated
directory contains `check_model.py`'s `isolated-0.ppm` for the same Krea assets.
Its default one-worker, 36-GiB retained-budget run compares text-only control
with mixed execution, preserving a pinned conversation through image weight
activation and subsequent text reload. Prepared and queued image cancellation,
health progress, exact native pixels, exact retained text and shared workspace
reuse are checked separately. It does not measure comparative performance.
