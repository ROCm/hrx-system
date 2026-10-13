# Serving HTTP profiles

All TCP and Unix listeners expose the same catalog and HTTP endpoints.
Inference requests select a registered deployment through the JSON `model`
field. Names come from [model documents](serving-models.md), not checkpoint
filenames or an inferred architecture. An absent model returns 404.

The text, image and embedding profiles use familiar OpenAI response shapes,
but are not the entire OpenAI API. Supported fields are explicit; unsupported
execution options reject rather than being silently ignored. The server has
no built-in authentication or TLS termination.

## Discovery and administration

| Method and path | Input or result |
| --- | --- |
| `GET /healthz`, `GET /readyz` | Server availability: `{"status":"ready"}`. |
| `GET /v1/models` | Prepared deployments not being removed, with input/output modalities and a `loom` capability object. |
| `GET /v1/status` | Every registration, progress/errors, admission, scheduler state and shared server resources. |
| `POST /v1/models/register` | Complete model document; HTTP 202 accepts registration intent. |
| `POST /v1/models/drain`, `POST /v1/models/undrain` | `{"model":"name"}`; HTTP 200 acknowledges the gate change. |
| `POST /v1/models/unregister` | `{"model":"name","mode":"drain"}`; mode may instead be `"abort"`; HTTP 202 accepts removal. |
| `POST /v1/models/activate`, `POST /v1/models/deactivate` | Model-only body; HTTP 202 accepts a residency operation on a healthy drained registration. |
| `POST /v1/models/pin`, `POST /v1/models/unpin` | Model-only body; HTTP 200 changes parameter eviction policy. |
| `POST /v1/devices/trim` | `{"device":0,"target_bytes":0}`; HTTP 200 returns actual resource accounting after eligible reclamation. |
| `GET /metrics` | Prometheus text counters and ownership gauges. |
| `GET /logz`, `GET /logz/stream` | Bounded JSONL polling or SSE with independent cursors. |
| `GET /v1/requests`, `GET /v1/sessions` | Bounded live owner inspection; sessions require a retained-text service. |

Prepared but drained or weight-evicted models remain discoverable.
`created` is the registration timestamp; `owned_by` is `"local"`.
`loom.kind`, `loom.endpoint` and `loom.tokenize` describe the modality and
loaded tokenizer. Text also reports configured `context_length` and
`limits.max_output_tokens`, not training context or total cache-pool capacity.
Capabilities do not establish learned quality or support for every optional field.

[Operation](../workflows/operate-model-server.md) specifies asynchronous control
completion, conflict and overload behavior. [Observation](../workflows/observe-model-server.md)
specifies model selection, cursors, SSE resume and inspection pagination.

## Chat completions

`POST /v1/chat/completions` accepts a text deployment and a message history:

```json
{
  "model": "scout",
  "messages": [{"role":"user","content":"What is 2+2?"}],
  "stream": true,
  "stream_options": {"include_usage": true},
  "max_tokens": 16,
  "temperature": 0
}
```

Omitted, false or null `stream` selects one `chat.completion` JSON response with
the complete assistant message, finish reason and usage. `stream:true` selects
incremental `chat.completion.chunk` SSE events. `stream_options.include_usage`
adds a final usage-only chunk and requires streaming. Both modes use the same
generation and retained-session path.

Either `max_tokens` or `max_completion_tokens` sets the output limit, from 1
through 16384; supplying both rejects. Omission uses the deployment default.
The limit counts selected tokens, including terminal EOS when generated, so
completion-token usage can exceed the number of visibly decoded tokens.

The current execution profile is greedy: `temperature` may be omitted or zero.
Nonzero temperature and other sampling controls are rejected. This is separate
from thinking: `enable_thinking` and the `model_options` object are passed to
source chat policy, which validates their meaning and renders the model's
framing. Their presence does not add a native sampling mode.

`tools` accepts up to 16 function declarations. Omitted `tool_choice` or
`"auto"` is supported; other choices reject. `store` may be omitted or false.
The source policy owns tool grammar, safe output boundaries and canonical
history. Generated calls become structured `tool_calls`; the server never runs
the tools. Invalid generated tool syntax fails rather than publishing a
partially executable call. The client executes tools and returns their results
in the next full conversation history.

### Retained conversations

`X-Loom-Session: SESSION` requests best-effort reuse of a named history. Requests
still supply full history. On an ordinary cache miss the server can replay that
history; a session name is not a durable chat database. Streaming and JSON turns
can alternate without changing the continuation contract.

`X-Loom-Checkpoint: NAME` requests an explicit immutable endpoint for the
destination session. It is not best effort: an unknown name returns 404 and a
history mismatch returns 400 before replacing the destination continuation.
Session and checkpoint names are scoped to the selected deployment, so equal
names on two models do not share state.

### Named checkpoints

The deployment needs nonzero `checkpoint_capacity`. Control requests carry
`X-Loom-Model: scout`; this selector may be omitted only when exactly one
deployment is registered. Names contain 1–64 ASCII letters, digits, underscores,
dots or hyphens. They are identifiers, not authentication tokens.

| Request with empty body | Effect |
| --- | --- |
| `POST /v1/checkpoints/NAME` with `X-Loom-Session: SESSION` | Pin an idle completed session; HTTP 201 reports its consumed token position. |
| `POST /v1/checkpoints/NAME/suspend` | With elastic backing, park the endpoint in host memory; HTTP 200 reports `host_snapshot_bytes`. |
| `DELETE /v1/checkpoints/NAME` | Release the explicit pin; HTTP 200 after accepted release. |

Absent sessions or checkpoint names return 404; busy sessions, duplicate names
or queued borrowers preventing deletion return 409. Exhausted pin capacity
returns 503. Fixed backing rejects suspension with 400.

Pin completion publishes an immutable consumed frontier and the exact pending
token. The next completion still sends full history extending the endpoint's
canonical transcript. Another destination session forks that endpoint; the
same destination rewinds it. This supports keeping a tool-output summary while
discarding the verbose tool text from subsequent context.

Suspension moves the endpoint's placement without resetting live branches.
Its host image is process-local, not a file or durable disk cache. Resuming
warms it into new pool IDs. On unified-memory hardware, host storage does not
create additional physical RAM. Explicit pins stay protected until released;
automatic idle caches may yield under pressure.

Accepted pin/suspend work survives client disconnect through actual readiness.
Deleting a pin does not invalidate active branches that already own independent
references. An impossible reservation or refused restore returns overload
without silently replacing the destination's continuation.

## Embeddings

`POST /v1/embeddings` accepts:

```json
{"model":"vectors","input":["a","bbb"],"dimensions":2,"encoding_format":"float"}
```

`input` may be one string, an ordered string batch, a token-ID sequence or a batch
of token-ID sequences. `encoding_format` is `"float"` by default or `"base64"`;
base64 encodes little-endian IEEE754 F32 values. `dimensions` requests a
source-supported width, not arbitrary host truncation. Optional `user` is a
string accepted as client metadata. Unknown fields reject.

The response is an `object:"list"` with ordered embedding objects, original
per-request indices and token usage. Several requests can share one dispatch
cohort without sharing response identities. Source programs own tokenization,
pooling, projection, normalization and capacity. Cancellation suppresses that
client's response without invalidating peers in the cohort.

## Image generation

`POST /v1/images/generations` accepts a fixed-shape image deployment:

```json
{"model":"illustrator","prompt":"a red fox in the snow","seed":"0","response_format":"b64_json"}
```

The response has one `data[0].b64_json` PNG. Only `n:1` and
`response_format:"b64_json"` are supported. Optional `size` must match the
deployment's configured width and height; it does not trigger a new shape.
`seed` defaults to zero and accepts an unsigned integer or decimal string.

`strength` is a finite adapter multiplier, default one. With a loaded adapter,
zero selects base weights. Without an adapter only the default strength is
accepted. This endpoint is text-to-image; it does not accept reference-image
uploads or imply image-editing support. Unknown fields reject.

## Finite decisions

`POST /v1/systemone` accepts a decision deployment, string/object/array `state`,
and a nonempty map of typed questions:

```json
{
  "model": "judge",
  "state": "The customer cannot log in.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose a team.",
      "criteria": {"support":"Account access","billing":"Payments"}
    },
    "urgency": {"type":"score","instructions":"Rate urgency.","criteria":["Low","Medium","High"]},
    "needs_reply": {"type":"noul","instructions":"Is a reply needed?"}
  }
}
```

Each question requires a type and instructions. Instructions accept a string,
object or array. Choice has 1–255 named options and returns the winning ID plus all probabilities.
Score has 2–10 ordered levels and returns an expected zero-based score,
probabilities and a legend retaining typed criteria. Noul returns the probability
of true. Question and option identities are preserved. The native bound is
1024 questions; source capacity can be smaller.

The source produces normalized F32 vectors. The host validates and encodes them;
it does not silently normalize or truncate malformed outputs. Usage reports
source input tokens and zero output tokens: this is finite evaluation, not
autoregressive generation. Schema/source-input failures return 422. Strict JSON
is required; comments, trailing commas, malformed Unicode, duplicate decoded
IDs, unknown fields and image-input extensions reject.

## Raw tokenization

`POST /tokenize` uses the selected deployment's loaded tokenizer:

```json
{"model":"scout","content":"Hello!","add_special":false,"parse_special":true}
```

It returns `{"tokens":[...]}`. The shown special-token options are defaults.
`POST /detokenize` accepts `{"model":"scout","tokens":[...]}` and returns
`{"content":"..."}`, preserving special tokens. Neither operation applies
chat templates or model-specific prompt framing. Empty text and arrays are valid.

These CPU utilities do not activate weights, occupy execution rows or wait for
an inference epoch. They still obey model admission and retain their tokenizer
borrow through actual task completion. An unknown model or absent tokenizer
returns 404. Unknown, duplicate, wrongly typed or invalid-token fields return 400.
A tokenizer execution failure returns 500 for that request.

Each utility has a 262144-token and 1 MiB decoded-text bound, independent of the
model context and transport body limits. Exceeding a bound returns 413 rather
than a truncated success. `with_pieces:false` is accepted; true rejects because
this decoder contract does not expose raw token bytes. `prompt`,
`add_special_tokens` and chat-tokenization fields are not aliases for this profile.
