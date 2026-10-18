# Run a native voice conversation

`loom-serve` can transcribe microphone audio, generate a response, and stream
speech over one WebSocket connection. Recognition, text, and speech use their
native services inside the same process. Each deployment keeps its own device,
residency policy, and compatible numerical batches.

Turns can be committed by the client or detected automatically. The explicit
workflow requires three registered model packages: a recognition
model, a retained-text model, and a speech model accepting continuing text input.
The text deployment needs retained-session storage and two checkpoint slots per
live conversation, with sufficient shared cache storage for those prefixes.

## Configure an assistant

An assistant names the composition exposed through the connection's `model`
query. For deployments named `listener`, `writer`, and `speaker`, create
`assistant.json`:

```json
{
  "name": "assistant",
  "recognition": "listener",
  "text": "writer",
  "speech": "speaker",
  "voice": "alba",
  "instructions": "Answer briefly and clearly.",
  "max_output_tokens": 256,
  "seed": 42
}
```

The speech package must provide the selected voice. Deployment names refer to
the names in the three [model documents](../model-documents.md); they do not
select model architectures or imply device placement. Start the server with
those documents and the composition:

```sh
loom-serve --device=task --pool_backing=fixed \
  --model=listener/model.json --model=writer/model.json \
  --model=speaker/model.json --realtime_assistant=assistant.json \
  --listen=127.0.0.1:8080 --realtime_origin=http://localhost:3000
```

This command requires CPU-capable packages. Other device selections follow the
normal [deployment workflow](serve-models.md). Repeat `--realtime_assistant` to
expose several compositions sharing any of the same deployments. Repeat
`--realtime_origin` for each browser application's exact Origin. Native clients
that send no Origin header do not require an allowlist entry.

## Send an utterance and receive speech

Connect to `ws://127.0.0.1:8080/v1/realtime?model=assistant` using a WebSocket
client. The server sends `session.created` with its effective settings. Audio in
both directions is mono signed little-endian PCM16 at 24 kHz, base64-encoded in
JSON text events. PCM conversion requires no optional audio codecs.

Send `input_audio_buffer.append` events containing an `audio` string with the
base64 bytes. Each append contains complete two-byte samples. Network packet
boundaries do not reset the rate converter or define a recognition interval.
At the end of the utterance, send:

```json
{"type":"input_audio_buffer.commit"}
```

Commit seals the utterance, flushes conversion lookahead, and finishes recognition
without requiring another append. `input_audio_buffer.committed` identifies the
new conversation item. Its final transcript arrives as
`conversation.item.input_audio_transcription.completed`. A new append starts a
separate input item and can arrive while the previous utterance finishes.

Request an answer explicitly:

```json
{"type":"response.create"}
```

The response waits for preceding committed utterances to finish. Text content
feeds continuing speech directly; reasoning spans are not spoken. Consume
`response.output_audio_transcript.delta` for the response text and independently
base64-decode each `response.output_audio.delta` into a continuous playback queue.
`response.done` describes generation completion, not how much the listener heard.
An output-token cutoff reports `incomplete` with reason `max_output_tokens`.

A text or speech request failure reports `response.done` with status `failed`
and the original diagnostic in `status_details.error`. Microphone capture and
the connection remain available. Correct the request settings or conversation
input before requesting another response. Failed audio contributes no unaligned
assistant transcript to subsequent model context.

`session.update` changes subsequent instructions, voice, transcription hints,
output modality, or token budget at an idle response boundary. Selecting
`output_modalities:["text"]` produces `response.output_text.delta` without speech.
The [Realtime profile](../http-profiles.md#realtime-conversations) defines the
supported event and settings surface.

To customize one reply, put its overrides in `response.create`:

```json
{
  "type": "response.create",
  "response": {
    "audio": {"output": {"voice": "alba"}},
    "max_output_tokens": 128,
    "metadata": {"input_item_id": "utterance-7"}
  }
}
```

Use a voice supplied by the selected speech package and your own correlation
values. The response owns its settings and metadata through completion,
cancellation, or failure. Both `response.created` and `response.done` echo the
metadata; it is not passed to the model. The next reply inherits session defaults
again. Voice and metadata changes preserve compatible text checkpoints.

## Detect turns automatically

Register an activity-capable model and add `"activity":"detector"` to the
assistant document, using that deployment's registered name. Include its model
document with another `--model=detector/model.json` argument. The detector can
use a different device and sample rate from recognition, text, and speech.

Enable automatic turns after connecting:

```json
{
  "type": "session.update",
  "session": {
    "audio": {
      "input": {
        "turn_detection": {
          "type": "server_vad",
          "threshold": 0.5,
          "prefix_padding_ms": 300,
          "silence_duration_ms": 500,
          "create_response": true,
          "interrupt_response": true
        }
      }
    }
  }
}
```

Continue sending microphone appends. Speech onset emits
`input_audio_buffer.speech_started`; enough consecutive silence emits
`input_audio_buffer.speech_stopped` and commits the detected interval. The
transcript and response then arrive without explicit commit or response commands.
The detector stays continuous while earlier transcripts finish. Another onset
cancels an active answer when `interrupt_response` is enabled.

For a client that orchestrates responses, set `create_response:false`. It still
receives onset, stop, and final transcription, and can explicitly request the
answer. With `interrupt_response:false`, an ongoing answer finishes before a
pending automatic answer begins. A newer onset replaces pending automatic
response intent, so the next answer includes the latest committed speech.

The assistant's optional `input_capacity` bounds retained original-rate PCM in
frames; its default is 48000 frames, or two seconds at 24 kHz. This covers
pre-roll and pipeline storage, rather than limiting utterance duration. Larger
source windows can require a larger capacity. Quiet audio older than pre-roll
is discarded. Commit or clear establishes an empty microphone boundary before
changing turn policy or transcription hints. Explicit commit never requests
an automatic response. Setting `turn_detection:null` selects manual turns again.

## Interrupt and reconcile playback

Send `response.cancel` to stop future generation. Its optional `response_id`
identifies the active response. Already admitted CPU or device work retains its
storage until retirement; other requests can continue independently.

The client knows the actual playback cursor. After stopping playback, send
`conversation.item.truncate` with the assistant item's ID, `content_index:0`, and
the heard duration in `audio_end_ms`. The server acknowledges that duration and
removes the item's entire transcript from model-visible history. It does not
guess which words correspond to the playback cursor. A compatible checkpoint
before that turn preserves prefix reuse for the next response. Changing only
the voice also preserves text history.

Automatic interruption already removes the cancelled audio response's unaligned
text. A later playback report for that item is idempotent, including while a
successor response runs. The server cannot infer queued client playback for an
already completed response; the client's reported cursor remains authoritative.

`input_audio_buffer.clear` discards the uncommitted microphone item.
`conversation.item.delete` removes a committed item after its response ownership
has retired. Disconnect cancels delivery and joins accepted conversion work and
checkpoint cleanup, including when model inference has been drained. Removing
and replacing a deployment never redirects an old conversation to the replacement.
