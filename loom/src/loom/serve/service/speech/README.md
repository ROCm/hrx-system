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

Voice identifiers select the loaded package's voices or recordings registered
with a model supporting voice uploads. A voice object `{"id":"alba"}` selects
the same identifier as the string `"alba"`. Optional
`instructions`, `speed` (0.25–4, default 1) and `seed` are also passed to source;
the package rejects controls it cannot implement. `stream_format` accepts
`"audio"`. Encoded-file formats, SSE and an omitted response format reject;
they are not silently converted to raw PCM.

The [waveform fixture](../../fixture/speech/README.md) is an executable example
of the model contract. It generates deterministic samples, not learned speech.

## Register a voice recording

Speech ABI2 models accept a recording through `POST /v1/audio/voices`. The file
uses the same audio input path as dictation: WAV, MP3, and FLAC when those codecs
are enabled, or raw PCM with explicit sample rate and channels. Filenames are
opaque; container metadata or the raw PCM content type identifies the format.

```sh
curl --fail http://127.0.0.1:8080/v1/audio/voices \
  -F model=speech -F id=my-voice -F 'file=@recording.wav;type=audio/wav'
# {"id":"my-voice"}

curl --fail http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"speech","input":"Good day.","voice":"my-voice","response_format":"pcm"}' \
  --output speech.pcm
```

A successful upload response means preparation has finished and the voice is
ready. The model processes the recording once, including its channel mixing,
resampling, and learned voice encoding. Later speech requests reuse the resulting
conditioning. Registration lasts for the loaded model's lifetime; it does not
write a prepared checkpoint or change the model card.

Raw input uses `audio/pcm;rate=44100;channels=2`, with optional `format=s16le`
(the default) or `format=f32le`. F32 input contains finite normalized samples
between -1 and 1. Both formats are little-endian and interleaved. The model
bounds decoded recording storage and may reject recordings outside its duration
or content requirements. The server also bounds the original HTTP body.

An existing or preparing ID returns HTTP 409. When all voice slots are occupied,
an upload returns 503. Removing a voice releases its name immediately:

```sh
curl --fail http://127.0.0.1:8080/v1/audio/voices/remove \
  -H 'Content-Type: application/json' \
  -d '{"model":"speech","id":"my-voice"}'
```

Accepted speech, including queued requests, keeps its original voice through
completion. A replacement upload with the same name uses another free slot until
those captures retire. Removing an unknown or unfinished voice returns 404.
Disconnecting an upload before successful completion cancels registration;
accepted processing still retires safely. Voice removal also works while model
admission is drained. ABI1 models retain their package-defined voices and reject
uploads with 501.

Native applications use `loom_serve_speech_voice_request_initialize` and
`loom_serve_registry_try_upload_voice`. Successful admission moves the owned
recording into the service and returns a completion handle; pressure preserves
caller ownership. Check `loom_serve_speech_voice_upload_is_complete` and its
terminal status before selecting the ID in a speech request. Closing the handle
before completion cancels publication without waiting for CPU or device work.
`loom_serve_registry_remove_voice` hides a published ID on the serving owner.
The independent completion handle remains valid after model removal.

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
retirement. Discovery reports text input and audio output, plus audio input for
models supporting uploaded voices. Common telemetry
reports request outcomes and completed model units without treating waveform
frames as text tokens.
