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

Size the text deployment's
[`context_capacity`](../model-documents.md#text-definitions) for instructions,
tool definitions, conversation history, and generated output. A client may send
a large tool catalogue or chat snapshot even for a short spoken question.
Clients that delegate to a separate chat completion need to budget that
request's complete context too. The assistant's `max_output_tokens` controls
generated output; the underlying text deployment owns the total context limit
and shared cache capacity.

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

`response.created` reserves the reply's position in the conversation. A microphone
utterance committed afterward belongs to the next turn, even while the reply
waits for earlier transcription. Output-item events begin when the reply has
content. Empty completions, failures before content, and cancellations before
content report an empty `output` array without adding an assistant message.

A text or speech request failure reports `response.done` with status `failed`
and the original diagnostic in `status_details.error`. Microphone capture and
the connection remain available. Correct the request settings or conversation
input before requesting another response. Failed audio contributes no unaligned
assistant transcript to subsequent model context.

`session.update` changes subsequent instructions, voice, transcription hints,
output modality, or token budget when the default reply is idle. Already accepted
independent responses keep their captured settings. Selecting
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

## Generate an independent reply

An auxiliary response can speak a status update or answer a separate question
while the main conversation continues. Set `conversation:"none"` to keep its
output out of default history. For a short status announcement with no prior
conversation input:

```json
{
  "type": "response.create",
  "response": {
    "conversation": "none",
    "input": [],
    "instructions": "Briefly tell the listener that the image is being generated.",
    "tools": [],
    "tool_choice": "none",
    "metadata": {"purpose": "status"}
  }
}
```

This inherits the session's voice and output modality. Match subsequent output
by `response_id`; events from several responses may interleave. The client
chooses whether to queue, replace, or mix their audio. To cancel this reply,
send `response.cancel` with the ID from its `response.created`. Cancellation
without an ID targets only the default reply.

Omitting `input` uses a snapshot of current default history. An explicit array
instead selects exactly the listed items, combining new text and references:

```json
{
  "type": "response.create",
  "response": {
    "conversation": "none",
    "input": [
      {"type": "item_reference", "id": "utterance-7"},
      {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Summarize that in one sentence."}]
      }
    ],
    "metadata": {"purpose": "summary"}
  }
}
```

Replace `utterance-7` with an actual committed item ID. A microphone reference
waits for that utterance's final transcription. Deleting the original item
after response acceptance leaves the selected input intact. Completed independent
output is also referenceable by ID, but does not become default history by itself.

One default response and multiple independent responses can be live together,
up to four records including retiring work. Each keeps its own voice, native
request, conversion cursor, and cancellation. Native services continue selecting
compatible work for batching across requests and sessions. A slow socket still
shares one output-credit budget across its responses. Independent responses
neither consume a pending automatic voice reply nor change its history.

## Call tools and speak the result

With a text package that supports function calling, declare the client's tools
in session settings. Realtime uses flat function objects:

```json
{
  "type": "session.update",
  "session": {
    "tools": [{
      "type": "function",
      "name": "lookup_weather",
      "description": "Look up the current weather for a city.",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }],
    "tool_choice": "auto"
  }
}
```

Commit the user's utterance and create a response as above. A response can
contain spoken content, several function calls, or calls alone. Tool-only
responses acquire no speech request. After receiving a
`response.output_item.done` event whose item is a `function_call` with
`status:"completed"`, execute that function in the client. Its `arguments`
field contains the complete JSON argument string; `call_id` identifies the
result's destination. Arguments-done events can also occur during cancellation,
so completed output-item status is the execution boundary.

Return each result immediately, using the call ID supplied by the server:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "function_call_output",
    "call_id": "call_7_0",
    "output": "Seattle: 16 degrees Celsius, light rain."
  }
}
```

Results may arrive before the original `response.done`. Once that response has
finished and the required results are acknowledged, send another
`response.create`. The text model receives its calls and the returned results
in canonical history, and its answer streams through the selected speech model.
Set `response.tool_choice:"none"` for a follow-up that should generate content
without further calls; this leaves the session catalogue intact.

Cancellation preserves calls already published as completed. Their results
remain valid; unfinished calls never enter subsequent model input. The server
reserves the full generated call set and checks its serialized size before any
call becomes executable. A response that exceeds those bounds fails locally,
leaving the connection available for a corrected request.

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

Send `response.cancel` to stop future generation of the default reply, or supply
`response_id` to select any live response. Already admitted CPU or device work retains its
storage until retirement; other requests can continue independently.
If the response has already stopped, the server reports
`response_cancel_not_active`; the call can continue on the same connection.

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
Reconnect to use the replacement. A response using a removed text or speech
deployment reports `failed`. Microphone input using a removed recognition or
activity deployment reports an `error` event and closes the connection. Fresh
conversations retain their own history while old conversations release their
resources.
