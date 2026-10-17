# Serving HTTP profiles

All TCP and Unix listeners expose the same catalog and HTTP endpoints.
Inference requests select a registered deployment through the `model` field
in JSON or multipart form data. Names come from
[model documents](model-documents.md), not checkpoint filenames or an inferred
architecture. An absent model returns 404.

The text, image, embedding, speech and transcription profiles use familiar
OpenAI response shapes, but are not the entire OpenAI API. Supported fields are
explicit; unsupported execution options reject rather than being silently
ignored. The server has no built-in authentication or TLS termination.

## Request framing

Request bodies accept either `Content-Length` or HTTP/1.1
`Transfer-Encoding: chunked`. Supplying both rejects. Chunked transfer is decoded
incrementally into the same bounded request body; the complete request is
validated before routing. `--request_body_bytes` bounds decoded body bytes,
independently of bounded HTTP/chunk framing and multipart metadata. This permits
upload clients that do not know the body length in advance; it does not turn an
inference endpoint into a continuing microphone-input session.

Multipart fields retain their names, ordering and binary contents through
request ownership. Consumers use those parsed fields without interpreting MIME
boundaries again. Part order does not control routing. Unsupported transfer
encodings, ambiguous framing and malformed/truncated multipart bodies reject.
`Expect: 100-continue` is unsupported; upload examples explicitly clear that
header.

## Discovery and administration

| Method and path | Input or result |
| --- | --- |
| `GET /healthz`, `GET /readyz` | Server availability: `{"status":"ready"}`. |
| `GET /v1/models` | Prepared deployments not being removed, with input/output modalities and a `loom` capability object. |
| `GET /v1/status` | Every registration, progress/errors, admission, scheduler state and shared server resources. |
| `POST /v1/models/register` | Complete model document, optionally with uploaded source files; HTTP 202 accepts registration intent. |
| `POST /v1/models/drain`, `POST /v1/models/undrain` | `{"model":"name"}`; HTTP 200 acknowledges the gate change. |
| `POST /v1/models/unregister` | `{"model":"name","mode":"drain"}`; mode may instead be `"abort"`; HTTP 202 accepts removal. |
| `POST /v1/models/activate`, `POST /v1/models/deactivate` | `{"model":"name","component":"primary"}`; HTTP 202 accepts a residency operation on a healthy drained registration. |
| `POST /v1/models/pin`, `POST /v1/models/unpin` | Same component selector; HTTP 200 changes parameter eviction policy. |
| `GET /v1/models/variants` | Native executable inventory or one candidate's retained compilation diagnostics. |
| `POST /v1/models/variants` | Stage native providers against a ready base without selecting them. |
| `POST /v1/models/variants/select` | Select a ready candidate with expected-current protection. |
| `POST /v1/models/variants/remove` | Remove an unused terminal candidate. |
| `POST /v1/devices/trim` | `{"device":0,"target_bytes":0}`; HTTP 200 schedules eligible reclamation and returns current resource accounting. |
| `GET /metrics` | Prometheus text counters and ownership gauges. |
| `GET /logz`, `GET /logz/stream` | Bounded JSONL polling or SSE with independent cursors. |
| `GET /v1/requests`, `GET /v1/sessions` | Bounded live owner inspection; sessions require a retained-text service. |

Prepared but drained or weight-evicted models remain discoverable.
`created` is the registration timestamp; `owned_by` is `"local"`.
`loom.kind`, `loom.endpoint` and `loom.tokenize` describe the modality and
loaded tokenizer. Text also reports configured `context_length` and
`limits.max_output_tokens`, not training context or total cache-pool capacity.
Capabilities do not establish learned quality or support for every optional field.

Residency controls default to `component:"primary"`. A text deployment with an
image-input component also accepts `component:"image"`; an image-generation
deployment's normal computation is still `primary`. An undeclared component
returns 404; an invalid selector returns 400. Drain and unregister apply to the
whole registration and do not accept a component selector.

[Operation](workflows/operate-model-server.md) specifies asynchronous control
completion, conflict and overload behavior. [Observation](workflows/observe-model-server.md)
specifies model selection, cursors, SSE resume and inspection pagination.

### Native executable variants

A prepared service supporting native updates reports
`loom.variants:{"selected":12,"count":2}` in model discovery and status.
Text, image, embedding and decision services enable this capability through
nonzero `variant_capacity`. Disabled or custom services without the capability
return 501. A registration must have a healthy prepared service; drained
admission is permitted, removal is not.

`GET /v1/models/variants` requires `X-Loom-Model: scout` and an empty body.
It returns `object:"list"`, `model`, `selected`, `count`, `offset`, `data` and
`next_offset`. Optional `X-Loom-Offset` selects a zero-based index; each page
contains at most 64 entries in creation order. This is live inventory, not a
stable snapshot across concurrent removals. Instead, `X-Loom-Variant: 13`
selects one full record, including compilation diagnostics. The variant and
offset headers are mutually exclusive.

Staging accepts:

```json
{"model":"scout","base":12,"base_directory":"/srv/models/scout/candidate","sources":["config.loom","advance.loombc"],"wait":true}
```

`base` identifies a ready choice in this registration. It asserts that the
new providers preserve its resource contract; it does not merge in the base's
providers. `sources` supplies the complete native provider set. With a JSON
request, relative paths require an absolute `base_directory`; otherwise paths
must be absolute. Files are read by server-side compilation workers. A multipart
request instead supplies an immutable source package, described below.
`prepare`, `control` and other model fields reject.

Omitted or true `wait` completes the HTTP request after JIT with 200 for ready
or 422 for failed compilation. False returns 202 after accepting construction.
Both responses contain `selected` and `variant`. A variant record contains
`object:"variant"`, `id`, `base`, `phase` (`compiling`, `ready` or `failed`) and
`users` (explicit consumer references). Terminal failures retain an `error`
with code/message. Detailed records also include available `compilation`
evidence. Staging never changes the preferred choice, including when JIT fails.

Selection accepts `{"model":"scout","variant":13,"expected":12}`. The request
completes at the service's safe execution boundary and returns
`{"selected":13}`. A stale expected ID, pending selection, compiling/failed
candidate or candidate from another registration rejects without switching.
Selection failure leaves the previous preference intact. Query inventory after
an error or interrupted connection to inspect the actual preference.

Execution replacement preserves preparation/control VM programs, stage
declarations and specialization values, parameter placement, and retained state.
It can replace command programs and their reachable kernels, or direct-dispatch
implementations including resident pipelines. It does not rerun weight
preparation. Text selects between retired epochs; embedding and decision
services select outside their active cohort; image services finish the whole
image and CPU encoder before selection. Prepared but unsubmitted work may use
the new execution because its request/result layout is unchanged. Image command
part counts and transient requirements may change without relinking the control
VM. Changing preserved model contracts requires a new registration.

Removal accepts `{"model":"scout","variant":12}` and returns `{"removed":12}`.
Compiling or consumer-retained entries cannot be removed. Missing entries return
404; conflicts return 409; exhausted pool or management-observation credit
returns 429. Unknown, duplicate or wrongly typed fields return 400.

IDs never recycle within the server process, including after unregister/register.
They are not persistent identifiers across restarts. Compilation results remain
until explicit removal or model retirement. Disconnect withdraws the reader,
not accepted compilation or selection. Unregister withdraws waiting compilation
readers with 410 while still joining the compiler's actual model ownership.

### Uploaded source packages

Registration and native staging also accept `multipart/form-data`: exactly one
`request` part contains the same JSON as the corresponding endpoint, and repeated
`file` parts carry programs and metadata. Each file's `filename` is a
case-sensitive UTF-8 path inside the package, not a server filesystem
destination. The request filename defaults to
`request.json`; its directory is the base for relative source references.

For a local `candidate.json` containing
`{"model":"scout","base":12,"sources":["config.loom","advance.loombc"]}`:

```bash
curl --fail-with-body -H 'Expect:' \
  http://127.0.0.1:8080/v1/models/variants \
  -F 'request=@candidate.json;type=application/json;filename=model/candidate.json' \
  -F 'file=@config.loom;filename=model/config.loom' \
  -F 'file=@advance.loombc;filename=model/advance.loombc'
```

Package filenames use `/` separators and contain no empty, `.` or `..`
components, absolute paths, backslashes, drive prefixes or control characters.
Authored relative references may use `..` to reach a shared provider inside the
package, but cannot escape its root. Missing members fail without falling back
to server-local files. Explicit absolute source paths still name server-local
providers, so an upload may combine changed files with existing local inputs.
Filenames are literal UTF-8, not URL-decoded.

Parts may arrive in any order; duplicates and malformed/truncated bodies reject
before registration or staging. File bytes are retained in memory independently
of the request connection, including binary `.loombc`. Native staging releases
its source ownership after constructing independent executables and copied
diagnostics. Registration retains the package until the model retires, including
when `admission:"drained"` defers preparation. These are code packages, not
checkpoint transfers or a persistent package catalog.

For `POST /v1/models/register`, the `request` part is the complete
[model document](model-documents.md). Relative `prepare`, `control`, native
providers, tokenizer metadata and source-declared weight-preparation policies
resolve inside the package. An image or finite model's relative asset directory
also applies within that namespace when loading metadata. Absolute locators
still explicitly select server-local files.

Parameter checkpoint bytes always use the server filesystem, independently of
the package namespace. An absolute `base_directory` in the model document binds
relative checkpoint paths there; it does not redirect relative code or metadata
away from the package. For example, `checkpoint:"."` or `assets:"."` can let a
model read uploaded tokenizer metadata while its source-declared weight files
resolve under `base_directory:"/srv/checkpoints/model"`.

The normal `--request_body_bytes` limit covers the complete multipart body;
there is no implicit upload exception. Both content-length and chunked framing
are accepted. The empty `Expect` header above suppresses an unsupported
100-continue handshake. This endpoint loads executable code and has the same
trusted-network boundary as model registration.

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

`temperature`, `top_p`, `top_k` and `seed` are numeric request fields interpreted
by the model's source policy. Omission uses source-defined deployment defaults;
unsupported values reject before admission. The server carries the validated
policy to the model program without reading logits back to the host. Supported
sampling algorithms and value ranges belong to the model package.

Sampling is separate from thinking: `enable_thinking` and the `model_options`
object are passed to source chat policy, which validates their meaning and
renders the model's framing. Reasoning-aware models return `reasoning_content`
separately from answer `content`, in both SSE deltas and complete messages.

`tools` accepts up to 4096 function declarations. Omitted `tool_choice` or
`"auto"` is supported; other choices reject. `store` may be omitted or false.
The source policy owns tool grammar, safe output boundaries and canonical
history. Generated calls become structured `tool_calls`; the server never runs
the tools. Invalid generated tool syntax fails rather than publishing a
partially executable call. The client executes tools and returns their results
in the next full conversation history.

### Image input

An image-capable text deployment reports both `text` and `image` input
modalities. Message `content` may be an ordered array of `text` and `image_url`
parts. A text part contains `type:"text"` and `text`; an image part contains
`type:"image_url"` and an `image_url` object with `url` and optional `detail`.
Detail is `auto` by default, or `low` or `high`, with interpretation owned by
the model's source policy.

Image URLs must be base64 data URLs. The server does not fetch network URLs or
read image paths from the request. Image-enabled builds always accept static
PAM (`image/x-portable-arbitrarymap`) without third-party codecs. PNG, JPEG,
GIF, and WebP are enabled by default and can be omitted individually without
removing image input or other serving modalities. PAM supports standard
black-and-white, grayscale, and RGB tuples, with optional straight alpha and
sample maxima up to 65535, normalized to RGB8 or RGBA8. Animated or multi-image
inputs, MIME/signature mismatches, and images exceeding source-declared byte,
dimension, or pixel limits reject.
The normal HTTP request-body limit still applies.

Multiple images and concurrent requests can share projection cohorts. Features
remain device-resident through language input; a disconnected peer does not
change the surviving images' association. Full conversation history remains
required for retained reuse. Image identity includes the compressed bytes,
ordered placement, detail and resolved source metadata, so reuse cannot cross
a change in those semantics. A complete retained media prefix skips pixel
decoding and projection; it does not require a separate image-feature cache.

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

## Speech synthesis

`POST /v1/audio/speech` accepts a speech deployment:

```json
{"model":"speaker","input":"Good day.","voice":"alba","response_format":"pcm"}
```

The response streams signed little-endian PCM16 at 24 kHz, mono, with
`Content-Type: audio/pcm;rate=24000;channels=1`. Completion closes the response.
Network reads need not coincide with waveform blocks; consumers retain any
partial sample between reads. Disconnection suppresses further generation
while already accepted work retires.

Requests have independent voice, progress and output credit. Compatible next
operations may share numerical batches, including requests arriving after
generation has begun. A stalled reader excludes its own request from further
work while peers continue. Batch boundaries are internal execution boundaries;
they do not add a multiplexing protocol to the client's PCM stream. The
[streaming walkthrough](workflows/stream-speech.md) shows complete requests and
playback, and the [source contract](model-programs.md#stream-speech-with-independent-requests)
explains how packages define compatible work.

`model`, nonempty UTF-8 `input`, `voice` and explicit `response_format: "pcm"`
are required. `voice` is a package-supported identifier or an object containing
an `id` string. Optional `instructions`, `speed` and `seed` reach source request
preparation, which rejects controls it cannot implement. Speed defaults to 1
and accepts 0.25–4. The seed extension accepts an unsigned 64-bit integer or
decimal string and defaults to zero.

`stream_format` may be omitted or set to `"audio"`. Encoded formats, SSE and an
omitted response format return an input error. Voice registration and sample
persistence are not supplied by this endpoint; conditioning assets belong to
the loaded package. File-based players require an encoded-audio profile rather
than raw PCM.

## Audio transcription

`POST /v1/audio/transcriptions` accepts a recognition deployment and a completed
recording as `multipart/form-data`. Builds with audio codecs accept WAV, MP3,
and FLAC, converting their sample rate and channel layout to the model's input
geometry:

```sh
curl --fail-with-body -H 'Expect:' \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=recognizer \
  -F 'file=@recording.wav;type=audio/wav'
```

The default response is `{"text":"Recognized speech."}`. `response_format=text`
returns UTF-8 plain text. Independent finalized segments appear in input-time
order, separated by newlines; segment identities break equal-time ties.

| Field | Contract |
| --- | --- |
| `model` | Required registered recognition deployment. |
| `file` | Required nonempty recording in an enabled format, or PCM16LE with explicit geometry; the filename never names a server path. |
| `language` | Optional UTF-8 language hint interpreted by the package. |
| `prompt` | Optional UTF-8 context interpreted by the package. |
| `temperature` | Finite number from 0 through 1, default 0; source policy may restrict it. |
| `response_format` | `json` by default, or `text`. |
| `stream` | Omitted or `false` for this finite-upload profile. |

Container bytes select the decoder, independently of the filename or MIME
label. `audio/wav`, `audio/mpeg`, and `audio/flac` are conventional labels;
`application/octet-stream` also works for a supported container. A disabled
codec or unsupported container returns an error. Direct WebM/Opus and MP4/AAC
uploads are not accepted; clients must convert them to an enabled format.

Raw PCM16LE works in every build. Its file part explicitly declares `audio/pcm`
with the sample rate and interleaved channel count, matching the selected model.
For a 16 kHz mono recognizer:

```sh
curl --fail-with-body -H 'Expect:' \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=recognizer \
  -F 'file=@recording.pcm;headers="Content-Type: audio/pcm;rate=16000;channels=1"'
```

These bytes contain whole PCM16 frames without a WAV header. Missing, duplicate,
or mismatched geometry and partial frames reject. The raw path performs no
resampling or channel conversion. PCM input and PCM speech output require no
audio codec dependency.

The request-body byte limit and source-defined input-frame limit apply
independently. The raw PCM frame count is checked before model admission;
encoded recordings are checked as decoded frames reach recognition. An
overlong recording rejects without truncation or a partial success response.
Decoding and conversion stage a bounded number of frames on the shared CPU
executor, retaining each recording's codec and resampling history across blocks.

Uploads may share numerical batches while retaining independent sample clocks,
model histories, and transcript state. A disconnect cancels that request;
upload bytes, decoder state, and model storage remain owned until accepted
work retires.
Concurrent requests continue through their own input and output credit.

Unknown or duplicate fields, unsupported response formats and `stream=true`
reject. The finite profile does not emit provisional hypotheses, word timing or
speaker labels. The [recognition source contract](model-programs.md#recognize-continuing-audio)
preserves revisions and optional alignment/diarization evidence for typed
consumers. A chunked HTTP upload still completes before recognition begins;
it is not an appendable microphone session.

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

`POST /v1/images/generations` generates one image at the requested size:

```json
{"model":"illustrator","prompt":"a red fox in the snow","size":"1536x1024","seed":"0","response_format":"b64_json"}
```

The response has one `data[0].b64_json` PNG. Only `n:1` and
`response_format:"b64_json"` are supported. Optional `size` is `WIDTHxHEIGHT`;
omitting it uses the deployment's configured width and height. The model's
source validates supported geometry. Sizes are exact, never silently rounded.
A new shape can compile asynchronously on demand while sharing resident weights;
later requests reuse its code within the deployment's specialization-cache
limits. Larger images require more work and memory, and a first request can
include compilation time. Request sizes do not change the deployment default.
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
