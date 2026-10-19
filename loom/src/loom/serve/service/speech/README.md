# Stream source-defined speech

A `speech` deployment serves `POST /v1/audio/speech` from a source model package.
An explicit `response_format: "pcm"` produces signed little-endian PCM16 at
24 kHz, mono, declared as `Content-Type: audio/pcm;rate=24000;channels=1`.
Audio is delivered incrementally; request completion closes the response.
TCP read boundaries need not coincide with waveform blocks.

```json
{
  "model": "speech",
  "input": "Good day.",
  "voice": "alba",
  "response_format": "pcm"
}
```

Voice identifiers belong to the loaded package. A custom-voice object
`{"id":"alba"}` passes the same identifier to source preparation. Optional
`instructions`, `speed` (0.25–4, default 1) and `seed` are also passed to source;
the package rejects controls it cannot implement. `stream_format` accepts
`"audio"`. Encoded-file formats, SSE and an omitted response format reject;
they are not silently converted to raw PCM. This endpoint does not register or
persist voice samples. Package assets supply its conditioning inputs.

The [waveform fixture](../../fixture/speech/README.md) is an executable example
of the model contract. It generates deterministic samples, not learned speech.

## Native waveform consumers

An in-process serving owner uses `loom_serve_registry_try_open_speech` with an
initialized request and a deployment name. A successful admission moves the
request and returns an independently owned [stream handle](stream.h). NULL
success reports bounded admission pressure and leaves the request with its
caller. Native requests share HTTP's queue, numerical cohorts, residency, and
retirement; opening a handle does not expose a model or service pointer.

The stream reports its sample geometry and publishes one copied F32LE block at
a time, with an absolute frame position and explicit completion. Consuming the
block returns output credit. A stalled reader makes only its own request
ineligible for the next compatible batch. Copies run in the existing shared
CPU collection task. HTTP consumers encode directly into their PCM buffer.

Closing a native handle cancels caller interest, while accepted CPU and device
work retains its storage through retirement. Final blocks and terminal status
can outlive model removal, same-name replacement, and registry destruction.
Holding that result keeps no model pin or source row.

`loom_serve_speech_request_initialize_text` supplies finite text at admission.
`loom_serve_speech_request_initialize_stream` starts an empty continuing request:
write text through `loom_serve_speech_stream_write`, honor the accepted byte
count, and call `loom_serve_speech_stream_finish` when the final text arrives.
Writes may split UTF-8 characters; only complete scalars reach source. Invalid
input accepts no bytes, and finishing inside a scalar leaves input open so the
caller can complete it. The model declares both a bounded input window and a
total text limit. Input credit returns after the CPU accepts text, independently
of numerical completion and waveform consumption. HTTP still supplies a complete
JSON request; continuing text uses the native serving interface.

## Model ownership

The [model ABI](model.h) separates cold preparation, CPU request preparation,
continuing input acceptance, next-work selection, batch packing, numerical
execution, and retired-feedback interpretation. Source preparation chooses
checkpoint domains, stages, capacities, waveform geometry and initial
conditioning. The native model does
not encode a particular vocoder, speaker vector, latent width, number of
diffusion steps or waveform block size.

Each stage binds its own checkpoint roots. Voice encoding, text prefill and
waveform decoding can share a backbone while loading different phase weights.
Roots with the same ordered checkpoint file list and preparation policy share
compatible tensor placement and prepare each tensor once. Distinct file lists
or policies own independent storage, so a voice archive can reuse a tensor name
from the base model without aliasing it. Incompatible overlapping layouts reject
during preparation, before checkpoint payloads are loaded.

Five dynamic roots carry compact batch work, immutable request rows, reusable
model conditioning, mutable stream rows and bounded device output. Request and
stream rows remain stable across changes in batch membership. State crossing
commands lives in the stream bank; compiler-private workspace belongs to one
complete command and is released by the shared runner. Each request retains its
own source state, including cache positions and RNG progression.

CPU request state is an ordered set of typed scalar and reference values chosen
by `prepare_request`. Its final result is the device upload buffer; the preceding
values remain owned by that request. Every consuming entry declares the exact
same types. `prepare_input` and `complete` return replacement state before their
acceptance or waveform fields, so source can grow a text buffer while retaining
independent tokenizer or decoder objects. Binding checks those signatures before
service publication. State never
passes through the packed numerical upload or moves when batch membership
changes.

The synchronous [`input` capability](../../runtime/program/input.h) supplies
managed `input.encoder` and `input.decoder` objects for source-owned text
processing. Each object retains its own tokenizer state across calls. Source
passes reusable buffer ranges, receives consumed/written counts, and drains
final output with an explicit completion result. Reset restores independent
first-token behavior, so a continuing prefix decoder and a fresh suffix decoder
can coexist without rebuilding tokenizer state for every token. The capability
owns byte conversion and tokenizer mechanics; normalization, sentence boundaries,
packet construction, and buffering limits remain source policy.

`prepare_input` receives an immutable text window and its unconsumed byte offset.
Source can accept a complete-scalar prefix, retain the window through ordinary
VM ownership, and continue CPU processing over later turns. The adapter copies
each window once and reuses it until drained; source never sees a mutable input
ring alias. A delayed end-of-input signal reaches the current window when it
covers the final bytes.

`prepare_work` describes the request's next operation without advancing it:
numerical work, another bounded CPU step, waiting for input, or completion.
Waiting sleeps until a complete text prefix or EOF arrives. Completion can
publish an empty final waveform block without submitting numerical work.
Equal compatibility tags and additive work units within the declared batch
capacity permit a cohort. Source owns phase and shape compatibility; differing
request progress alone need not separate requests. `prepare_batch` packs the
selected work while retaining its resident-row mapping.

`generate` executes the batch. After retirement, `complete` runs for every
original batch ordinal, including phases without audio or readback. It returns
an absolute frame position, byte range, live frame count and EOF. Zero frames
are independent of EOF. Native code validates the external range once before
encoding; subsequent consumers use those facts. Every output copy finishes
before the batch releases its feedback. Conditioning becomes reusable only
after its producer retires; cancelled partial work cannot publish a valid cache
entry.

The [generator](generator.h) carries native sample rate and channel count.
PCM HTTP requests require 24 kHz mono; a package with different geometry remains
usable through native handles and rejects incompatible HTTP requests. This wire
format does not redefine the model ABI's geometry. HTTP conversion for another
native geometry requires a continuous converter with retained filter history
and final tail handling.

## Scheduling and retirement

Each registration admits independently resident requests up to its source-defined
capacity; a bounded host queue owns pending requests. One model CPU operation
and one accepted numerical batch serialize shared state. Source preparation,
batch selection/packing and feedback interpretation/encoding run on the shared
CPU executor. Queue submission and admission run on the serialized service
owner. Device work and feedback have independent completion frontiers, waited
together by the shared proactor. No worker waits for a device or socket.

CPU candidates and numerical members have separate ownership records. Every
successful CPU acceptance returns that request's input credit, including offers
that do not fit the selected numerical cohort. Request-local source rejection
retires only that row; accepted borrowers retain it through their actual release.
No-input waits, CPU continuations, and zero-work completion remain ordinary
scheduling outcomes rather than artificial device batches.

Each resident request owns at most one encoded block outside copied transport
sends. Its next model unit becomes eligible after that block acquires send
credit. A slow peer therefore bounds its own waveform queue while other requests
continue. Peer reset and shutdown suppress subsequent work for affected requests
while preserving every original member of an accepted batch through collection.
Pins, row storage and request handles release only after their accepted work and
CPU borrowers retire. Failed readiness leaves ownership with the model's actual
buffer-retirement path.

Source completion and waveform encoding failures terminate only their request;
collection continues for every other original batch member. A partially encoded
block is never published. HTTP returns a JSON error before the first audio
headers, or closes an already streaming response without inserting error text
into its PCM. Native consumers retain the request error after retirement. Both
paths return the row's capacity without poisoning peer or subsequent requests.

Dynamic registration and removal use the same constructors and teardown as
startup model documents. Protocol rejection preserves the prepared service;
terminal execution failures retain accepted consumers until registration
retirement. Discovery reports text input and audio output. Common telemetry
reports request outcomes and completed model units without treating waveform
frames as text tokens.
