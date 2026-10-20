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

## Model ownership

The [model ABI](model.h) separates cold preparation, CPU request preparation,
next-work selection, batch packing, numerical execution and retired-feedback
interpretation. Source preparation chooses checkpoint domains, stages,
capacities, waveform geometry and initial conditioning. The native model does
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
same types. `complete` returns replacement state before its waveform fields, so
source can grow a text buffer while retaining independent tokenizer or decoder
objects. Binding checks those signatures before service publication. State never
passes through the packed numerical upload or moves when batch membership
changes.

`prepare_work` describes the request's next operation without advancing it.
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
Native streams preserve that geometry. The HTTP consumer requires 24 kHz mono
and rejects incompatible geometry before publishing audio. This wire format
does not redefine the model ABI's geometry.

## Native consumers

Applications can initialize owned text, voice, and instruction fields with
`loom_serve_speech_request_initialize_text`, then admit the request through
`loom_serve_registry_try_open_speech`. Native and HTTP requests share one model,
admission queue, compatible execution cohorts, and task executor. Acceptance
moves the request into the service; queue pressure leaves it with the caller.

The returned stream exposes copied F32 waveform blocks with the source's
sample rate, channel count, and absolute frame position. `peek` borrows the
current block until `consume` returns its output credit. Holding a block makes
only that request ineligible for more source work. The bounded stream storage
is allocated once per request, and each block is copied at the existing CPU
collection boundary. HTTP encodes directly into its own bounded response
storage at that same boundary.

The caller closes its stream handle independently of service retirement. Close
withdraws publication interest immediately, while accepted CPU and device work
keep their owners until completion. An unread final block and terminal status
can outlive removal of the model registration. Caller operations run on the
serialized serving owner and never wait for execution or transport progress.

## Scheduling and retirement

Each registration admits independently resident requests up to its source-defined
capacity; a bounded host queue owns pending requests. One model CPU operation
and one accepted numerical batch serialize shared state. Source preparation,
batch selection/packing and feedback interpretation/encoding run on the shared
CPU executor. Queue submission and admission run on the serialized service
owner. Device work and feedback have independent completion frontiers, waited
together by the shared proactor. No worker waits for a device or socket.

Each resident request owns at most one encoded block outside copied transport
sends. Its next model unit becomes eligible after that block acquires send
credit. A slow peer therefore bounds its own waveform queue while other requests
continue. Peer reset and shutdown suppress subsequent work for affected requests
while preserving every original member of an accepted batch through collection.
Pins, row storage and request handles release only after their accepted work and
CPU borrowers retire. Failed readiness leaves ownership with the model's actual
buffer-retirement path.

A failure while interpreting retired source output or encoding one request's
block terminates that request without publishing the partial block. Every
original batch ordinal is still collected, including cancelled peers. Healthy
requests continue, and failed rows become reusable after ordinary retirement.
An HTTP failure before the first audio byte returns a JSON error; a failure
after audio begins closes the response without appending another wire format.

Dynamic registration and removal use the same constructors and teardown as
startup model documents. Protocol rejection preserves the prepared service;
terminal execution failures retain accepted consumers until registration
retirement. Discovery reports text input and audio output. Common telemetry
reports request outcomes and completed model units without treating waveform
frames as text tokens.
