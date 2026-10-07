# Serving multiple source models

`//experimental/loom_serve/serving:server` serves named text deployments through
one loopback HTTP listener. All deployments borrow one device, execution
timeline, parameter residency cache and private command workspace pool.
Each keeps its own source-JIT programs, weights, logical KV pool, sessions and
checkpoints. Registering the same checkpoint twice currently creates two
prepared parameter residencies; it does not deduplicate those weights.

The catalog selects source packages and deployment options. Model math,
checkpoint interpretation, storage geometry and chat policy remain in the
Loom sources described by the [text model contract](../text/README.md).
This executable currently accepts `kind: "text"`; the separate
[image service](../image/service.h) has not been connected to this owner loop.

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
    }
  ]
}
~~~

Paths resolve from the process working directory. Names are nonempty visible
ASCII identifiers, so the same value fits JSON requests and HTTP headers.
Unknown and duplicate catalog fields reject instead of silently changing
configuration. A model source can impose additional shape/storage constraints.

~~~sh
build_tools/bin/iree-bazel-run --config=opt \
  //experimental/loom_serve/serving:server -- \
  --models=/path/to/models.json --port=8080 \
  --pool_backing=elastic --memory_bytes=23622320128
~~~

The repository's build and hardware-execution policy applies. `--device`,
`--slab_bytes` and `--memory_bytes` configure the shared device. Connections
and HTTP body storage have independent bounds through `--connections` and
`--request_body_bytes`. Each model's `pending_requests` defaults to 32 and
`max_tokens` to 512. Other defaults and accepted ranges are in
[configuration.c](configuration.c).

## Requests and retained state

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

## Scheduling and memory ownership

The transport proactor handles network I/O independently. One application
owner accepts a bounded batch of requests, then advances each text service by
at most one scheduling cohort, rotating service priority. A cohort may contain
two device-fed epochs when configured. Model calls and residency transitions
are never concurrent host operations.

Admission pins a request's parameters and physically reserves its completion
high-water. Other models retry queued admission as the owner advances. If the
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
