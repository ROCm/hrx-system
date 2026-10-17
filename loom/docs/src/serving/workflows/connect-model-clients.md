# Connect model clients

`loom-serve` can back Pi, Oh My Pi and Open WebUI through their existing
OpenAI-compatible configuration. No client plugin or Loom-specific HTTP route
is required. The server continues to publish one complete catalog containing
text, image, embedding, decision, speech and recognition deployments; each
client selects the operations it understands.

This workflow assumes that the server and its model packages are already
running. [Serve source-defined models](serve-models.md) establishes that
baseline. The examples use `scout` for text, `illustrator` for image generation
and `vectors` for embeddings; replace those names with IDs from the live
catalog.

The [speech workflow](stream-speech.md) covers clients that consume raw PCM
incrementally. The audio settings below connect completed-recording dictation
and buffered playback to separate recognition and speech deployments.

## Inspect the client-visible contract

Set the base URL to the address visible from the client process and inspect the
catalog before configuring a client:

```sh
loom_server_url=http://127.0.0.1:8080
loom_base_url=$loom_server_url/v1
curl --fail --silent --show-error "$loom_base_url/models" |
  jq '.data[] | {
    id,
    input_modalities,
    output_modalities,
    context_length,
    max_output_tokens: .limits.max_output_tokens,
    endpoint: .loom.endpoint
  }'
```

A chat client selects deployments whose output includes `text` and whose
endpoint is `/v1/chat/completions`. Image and embedding integrations use
`/v1/images/generations` and `/v1/embeddings` respectively. The model ID and
limits come from this response; a checkpoint filename or architecture name is
not a substitute for the deployment contract.

`loom-serve` has no built-in authentication or TLS termination. Keep a direct
listener on a trusted network, or put an authenticated proxy or an SSH tunnel
in front of it. For example, this exposes a remote loopback listener only on
the client's loopback interface:

```sh
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18080:127.0.0.1:8080 server.example
```

The client base URL is then `http://127.0.0.1:18080/v1`. Loopback inside a
container refers to that container. A container therefore needs host networking
or another hostname that reaches the host-side listener; publishing the Loom
server publicly is not a substitute for that routing.

## Configure Pi with an explicit text model

Pi's generic OpenAI-compatible provider reads `~/.pi/agent/models.json`. It
does not discover generic providers from `/v1/models`, so copy the selected
text deployment and its limits into that file. The following command prints a
complete configuration for one catalog row:

```sh
text_model=scout
curl --fail --silent --show-error "$loom_base_url/models" |
  jq --arg base_url "$loom_base_url" --arg model "$text_model" '
    (.data | map(select(.id == $model)) |
      if length == 1 then .[0]
      else error("expected exactly one catalog row for \($model)") end) as $row |
    if ($row.output_modalities | index("text")) == null then
      error("selected deployment does not produce text")
    else {
      "$schema": "https://pi.dev/schemas/models.schema.json",
      "providers": {
        "loom": {
          "baseUrl": $base_url,
          "api": "openai-completions",
          "apiKey": "loom",
          "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
            "maxTokensField": "max_completion_tokens"
          },
          "models": [{
            "id": $row.id,
            "name": $row.id,
            "reasoning": false,
            "input": ($row.input_modalities |
              map(select(. == "text" or . == "image"))),
            "contextWindow": $row.context_length,
            "maxTokens": $row.limits.max_output_tokens,
            "samplingParams": {"temperature": 0}
          }]
        }
      }
    } end'
```

Use the printed document as `models.json` for a new configuration. If the file
already contains providers, merge the `loom` member into its existing
`providers` object. The dummy API key makes the provider available to Pi; Loom
does not authenticate it. Opening Pi's `/model` picker reloads the file.

The compatibility block describes the current Loom chat profile. It keeps Pi
on the `system` instruction role because Loom's public profile does not promise
OpenAI's `developer` role; it selects `max_completion_tokens` and suppresses
OpenAI's `reasoning_effort`. Streaming usage, `store:false` and ordinary
function tools already match Pi's OpenAI-completions transport.

For a Qwen package whose source chat policy actually supports thinking, extend
the model entry with:

```json
{
  "reasoning": true,
  "compat": {"thinkingFormat": "qwen"}
}
```

Pi then sends the top-level `enable_thinking` boolean that Loom passes to source
policy. A model name alone does not establish that capability.

Pi's native llama.cpp integration is a different protocol. It owns router
discovery, load, unload and download operations that Loom does not expose. Use
the generic `openai-completions` provider above rather than configuring Loom as
a llama.cpp router.

## Let Oh My Pi discover the mixed catalog

Oh My Pi can discover a generic OpenAI model list. Add this provider to
`~/.omp/agent/models.yml`:

```yaml
providers:
  loom:
    baseUrl: http://127.0.0.1:8080/v1
    auth: none
    api: openai-completions
    compat:
      supportsDeveloperRole: false
      supportsReasoningEffort: false
      maxTokensField: max_completion_tokens
    discovery:
      type: openai-models-list
    modelOverrides:
      scout:
        samplingParams:
          temperature: 0
```

Change the URL and `scout` to match the client-visible server and selected text
deployment. `omp models --kind all` refreshes and displays every discovered
kind. Oh My Pi consumes Loom's existing modality and limit fields: text-output
rows become chat models, image-only rows become image runners and
embedding-only rows become embedding runners. The ordinary model picker remains
a chat-model view rather than presenting every server deployment as chat.

For a Qwen package that supports Loom's `enable_thinking` option, extend its
override with `reasoning: true` and this model-local compatibility setting:

```yaml
        compat:
          thinkingFormat: qwen
```

Keep `temperature: 0`; the current Loom execution profile is greedy and rejects
nonzero temperature and other sampling controls.

## Give Open WebUI one model for each operation

Open WebUI uses separate OpenAI settings for chat connections, image generation
and RAG embeddings. Its chat discovery does not interpret Loom's modality
fields, so explicitly allow only text deployments on the OpenAI connection.
For a fresh Open WebUI data volume, these environment values establish a
Loom-only configuration:

```text
ENABLE_OLLAMA_API=false
ENABLE_OPENAI_API=true
OPENAI_API_BASE_URL=http://loom-host:8080/v1
OPENAI_API_KEY=loom
OPENAI_API_CONFIGS={"0":{"enable":true,"model_ids":["scout"]}}
DEFAULT_MODEL_METADATA={"capabilities":{"vision":false,"usage":true}}

ENABLE_IMAGE_GENERATION=true
IMAGE_GENERATION_ENGINE=openai
IMAGE_GENERATION_MODEL=illustrator
IMAGE_SIZE=1024x1024
IMAGES_OPENAI_API_BASE_URL=http://loom-host:8080/v1
IMAGES_OPENAI_API_KEY=loom

RAG_EMBEDDING_ENGINE=openai
RAG_EMBEDDING_MODEL=vectors
RAG_OPENAI_API_BASE_URL=http://loom-host:8080/v1
RAG_OPENAI_API_KEY=loom
```

Replace `loom-host`, the three model IDs, and the default image dimensions with
your deployment values. Open WebUI can request other sizes supported by the
image model; Loom specializes them on demand and reuses resident weights.
Omit the image or RAG block when no deployment provides that operation;
Open WebUI can retain its local
embedding engine instead.

`OPENAI_API_CONFIGS` key `0` describes the first OpenAI base URL. Its
`model_ids` list is the chat allowlist: `illustrator` and `vectors` remain
available to their operation-specific integrations without appearing in the
chat selector. The dummy keys satisfy Open WebUI's provider configuration and
are not an authentication boundary.

The shown default metadata is suitable for a Loom-only installation with a
text-only chat model. It prevents image attachments from being presented as a
model input and asks Open WebUI to retain streamed usage. If the selected text
deployment accepts image input, set `vision` to true. If one Open WebUI
instance has chat models with different capabilities, configure each model in
Open WebUI's workspace instead of applying one global default.

Open WebUI persists administrator changes. On an existing data volume, set the
same values in its OpenAI connection, Images and Documents settings; a stored
value takes precedence over a changed environment default. Keep nonzero
sampling controls disabled in model settings.

## Speech clients

A client that accepts raw audio can use `POST /v1/audio/speech` directly with an
explicit `response_format:"pcm"`. Each response is a continuous stream of
signed little-endian PCM16 at 24 kHz, mono. The
[speech walkthrough](stream-speech.md#play-a-learned-models-output) shows a
complete request and a streaming player. It also explains independent requests,
backpressure and cancellation. Available voice IDs and conditioning behavior
belong to the selected package.

Open WebUI's [ordinary TTS adapter](https://github.com/open-webui/open-webui/blob/f8ae8a6c328946dc62cd0712961a0632574581bf/backend/open_webui/routers/audio/__init__.py#L431)
awaits the complete provider response before returning a cached audio file.
Its full installation transcodes Loom's
`audio/pcm;rate=24000;channels=1` response to browser-playable MP3 using FFmpeg.
For a prepared speech deployment named `speaker` with a voice named `alba`,
set these values under **Admin Settings → Audio → Text-to-Speech**:

| Setting | Value |
| --- | --- |
| Engine | OpenAI |
| API Base URL | Your server URL followed by `/v1` |
| API Key | `local`, or the key required by your deployment proxy |
| Model | `speaker` |
| Voice | `alba` |
| Additional Parameters | `{"response_format":"pcm"}` |

The voice field accepts a package's voice ID directly. Choose a voice the
selected deployment accepts; OpenAI's built-in voice names are not aliases for
package voices. The additional format parameter is required because the speech
endpoint accepts explicit PCM requests.

This path requires the full Open WebUI installation with FFmpeg available and
`BYPASS_PYDUB_PREPROCESSING=false`. Slim installations reject raw PCM, and
bypassing preprocessing leaves a headerless file that browsers cannot play.
The adapter buffers each complete provider response before playback, so
incremental server output does not start playback within that request.

Its [realtime voice path](https://github.com/open-webui/open-webui/blob/f8ae8a6c328946dc62cd0712961a0632574581bf/backend/open_webui/routers/audio/realtime.py#L403)
additionally needs session events, microphone input, transcription, turn
detection and interruption. Configuring a speech model on the ordinary OpenAI
chat connection does not supply those capabilities.

## Transcription and dictation

A client can send a completed recording to a recognition deployment through
`POST /v1/audio/transcriptions` and read the JSON `text` result or request plain
text. The [transcription profile](../http-profiles.md#audio-transcription) gives
a copyable curl request and accepted fields.

With audio codecs enabled, upload WAV, MP3, or FLAC directly; the server
converts sample rate and channel layout to the model's requirements. A PCM-only
build accepts signed PCM16LE with explicit matching `rate` and `channels` in
the file part's `Content-Type`. Changing the filename or MIME label does not
convert unsupported bytes into an accepted format.

For a prepared recognition deployment named `recognizer`, set these values
under **Admin Settings → Audio → Speech-to-Text** in Open WebUI:

| Setting | Value |
| --- | --- |
| Engine | OpenAI |
| API Base URL | Your server URL followed by `/v1` |
| API Key | `local`, or the key required by your deployment proxy |
| STT Model | `recognizer` |
| Request Format | Multipart Upload |

Open WebUI's [transcription adapter](https://github.com/open-webui/open-webui/blob/f8ae8a6c328946dc62cd0712961a0632574581bf/backend/open_webui/routers/audio/__init__.py#L817)
sends multipart audio and reads the returned `text`. Its full installation
uses FFmpeg to prepare browser recordings as MP3 before forwarding them. Keep
`BYPASS_PYDUB_PREPROCESSING=false` and enable MP3 input in the Loom server.
The server itself does not require FFmpeg or Python. This configures the
dictation microphone; read-aloud uses the separate TTS settings above.

Chunked multipart transfer is accepted, but recognition begins after the upload
completes. The endpoint returns finalized text; appendable microphone sessions,
provisional transcripts, turn detection, and interruption require a realtime
session protocol in addition to this upload endpoint.

## Diagnose the boundary

Test from the same network namespace as the client. A successful request from
the host does not prove that a container can reach the URL:

```sh
curl --fail --silent --show-error "$loom_server_url/readyz"
curl --fail --silent --show-error "$loom_base_url/models"
```

Then test the operation directly with the exact model ID. The
[HTTP profile reference](../http-profiles.md) gives accepted request
fields for chat, image, embedding, speech and transcription operations. Loom
deliberately rejects unknown or unsupported options instead of silently changing
their meaning.
Common configuration failures are a nonzero sampling parameter, an image size
unsupported by the model's source, an operation using a model of another kind,
or a client base URL that duplicates or omits `/v1`.
