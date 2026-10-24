# Stream speech to independent clients

`loom-serve` delivers audio while a source-defined speech model is generating
it. Several clients can share the same prepared model and numerical batches
while keeping different voices, text lengths, playback positions and completion
times. Each connection receives its own continuous PCM stream.

This workflow first exercises the complete path with a small deterministic
package, then shows how to consume a learned model's audio. It requires an
installed `loom-serve` with the CPU target and task HAL enabled, plus `curl`.
The example package produces checkable samples without weights; it is useful
for client integration and model authoring, and does not produce spoken words.

## Start a speech deployment

Download the maintained [speech example](https://github.com/ROCm/hrx-system/tree/main/loom/src/loom/serve/fixture/speech)
into a new directory on the server machine:

```sh
mkdir speech
source_url=https://raw.githubusercontent.com/ROCm/hrx-system/main/loom/src/loom/serve/fixture/speech
for name in model.json prepare.loom control.loom generate.loom; do
  curl --fail --silent --show-error --location "$source_url/$name" --output "speech/$name"
done

loom-serve --device=task --pool_backing=fixed \
  --model=speech/model.json --listen=127.0.0.1:8080
```

Leave the server running and use another terminal for requests. Its `ready`
record reports the bound address. The deployment is named `speech`; its two
voices, `positive` and `negative`, select opposite waveform gains. The package
declares four resident request slots and numerical batches of up to four
compatible requests. These capacities belong to source preparation.

The same [model document](../model-documents.md#speech-definitions) can be
registered on an already running server. A package may target CPU, AMDGPU,
Vulkan or another supported execution path; its numerical programs must support
the selected device. Changing the HTTP client does not port a model's kernels.

## Receive incremental audio

```sh
curl --fail --silent --show-error --no-buffer \
  http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"speech","input":"hi","voice":"positive","seed":3,"response_format":"pcm"}' \
  --output positive.pcm
```

The response contains 17 signed little-endian PCM16 samples: 34 bytes at 24 kHz,
mono. Sample `i` is `((i + 3) % 8) * 4096`. Python's standard library can check
the complete result independently:

```sh
python3 - <<'PY'
from pathlib import Path
import struct

data = Path("positive.pcm").read_bytes()
assert len(data) == 34
samples = struct.unpack("<17h", data)
assert samples == tuple(((i + 3) % 8) * 4096 for i in range(17))
print("17 exact PCM16 samples")
PY
```

Audio starts when the model first publishes samples, without waiting for its
final block. Some numerical steps may produce no samples; an empty intermediate
block is not completion. The response closes when the request finishes.

HTTP reads may split a sample or combine several model blocks. An audio client
retains an incomplete two-byte sample across reads and feeds complete samples
to its playback queue. It does not use network packet boundaries as playback
timestamps. The [HTTP profile](../http-profiles.md#speech-synthesis) specifies
the exact fields, format and error behavior.

## Let requests advance independently

Two clients can request different voices and different amounts of output:

```sh
curl --fail --silent --show-error --no-buffer \
  http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"speech","input":"hi","voice":"positive","seed":3,"response_format":"pcm"}' \
  --output positive.pcm &
positive_request=$!

curl --fail --silent --show-error --no-buffer \
  http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"speech","input":"a","voice":"negative","seed":3,"response_format":"pcm"}' \
  --output negative.pcm &
negative_request=$!

wait "$positive_request" && wait "$negative_request"
wc -c positive.pcm negative.pcm
```

The files contain 34 and 18 bytes respectively. The scheduler can combine
compatible ready work; it starts available work without waiting for a batch to
fill, so concurrent submission does not guarantee that these short requests
share a particular batch.

For longer conversations, membership can change at every retired numerical
boundary. This is an illustrative sequence; conditioning steps are omitted:

| Selected batch | What changed |
| --- | --- |
| A | The first ready request starts without waiting for peers. |
| A, B | A later arrival joins compatible work. |
| A, B, C, D | Four voices share execution despite different request progress. |
| A, C | B has no output credit; D has cancelled and its accepted work has retired. |
| C | A has finished; C continues. |
| B, C | B's reader resumes, making B eligible again with its retained state. |

The service batches **compatible next work**. A package supplies compatibility
tags and additive work costs; it can pack different cache positions or prefix
lengths when its kernels support them. Different numerical phases or incompatible
shapes can require separate batches. A request keeps its state when its batch
neighbors change, without copying its complete history into a new cohort.

Each request has one bounded encoded block outside the transport's copied sends.
A stopped reader makes only that request ineligible for further generation.
Disconnecting a client cancels its future work; an already accepted batch keeps
its original members and storage until execution and output borrowers retire.
The freed request slot can then serve another client safely.

## Play a learned model's output

For a registered learned package named `speaker` that supports voice `alba` and
24 kHz mono output, the client protocol is the same. With FFmpeg's `ffplay`
installed, audio can flow directly into a player:

```sh
curl --fail --silent --show-error --no-buffer \
  http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"speaker","input":"Good day. Each voice has its own stream.","voice":"alba","response_format":"pcm"}' |
  ffplay -nodisp -autoexit -f s16le -sample_rate 24000 -ch_layout mono -i pipe:0
```

The package determines available voices, voice conditioning, text preparation,
sampling and EOS. A voice selector may also be an object such as `{"id":"alba"}`.
Speech ABI 2 models also accept [reference uploads](../http-profiles.md#voice-registration)
using the same audio formats as dictation. An upload prepares a named voice once;
subsequent synthesis requests reuse it. Playback buffering belongs
to the client, so incremental HTTP output alone does not establish audible
latency for a particular player.

The initial profile requires explicit `response_format:"pcm"`. Encoded file
formats, transcription and a conversational realtime session have separate
protocol requirements. Open WebUI's ordinary TTS path buffers the provider
response and returns a cached audio file; the
[client guide](connect-model-clients.md#speech-clients) explains its media-type
requirements and the effect on playback latency.

## Keep voice weights available

Speech deployments use the same residency controls as other models:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/pin \
  -H 'Content-Type: application/json' -d '{"model":"speaker"}'
```

Pinning retains activated weights against idle eviction. Working-state admission,
compute scheduling and client playback remain separate concerns; a parameter pin
does not reserve execution time or promise an audio-delivery deadline.
[Model operation](operate-model-server.md) covers activation, pressure and drain,
and [observation](observe-model-server.md) covers request progress and counters.

To author a learned package, start from the example's working execution and
lifetime contract, then replace its arithmetic with independently checked
numerical stages. The [speech authoring flow](../model-programs.md#stream-speech-with-independent-requests)
explains retained rows, compatible work, conditioning publication and retirement.
