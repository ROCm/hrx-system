# Connect model clients

`loom-serve` can back Pi, Oh My Pi, and Open WebUI through their existing
OpenAI-compatible configuration. Pi and Oh My Pi need no compatibility flags,
request-body overrides, or client plugin. The server publishes one complete
catalog containing text, image, embedding, decision, speech, and recognition
deployments; each client selects the operations it understands.

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
    reasoning,
    supported_reasoning_levels,
    default_reasoning_level,
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
    def thinking_levels($supported):
      ["off", "minimal", "low", "medium", "high", "xhigh", "max"] |
      map(. as $level | (if . == "off" then "none" else . end) as $effort |
        {key: $level, value:
          (if $supported | index($effort) then $effort else null end)}) |
      from_entries;
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
          "models": [{
            "id": $row.id,
            "name": $row.id,
            "reasoning": ($row.reasoning // false),
            "input": ($row.input_modalities |
              map(select(. == "text" or . == "image"))),
            "contextWindow": $row.context_length,
            "maxTokens": $row.limits.max_output_tokens
          } + (if $row.supported_reasoning_levels then {
            "thinkingLevelMap": thinking_levels($row.supported_reasoning_levels)
          } else {} end)]
        }
      }
    } end'
```

Use the printed document as `models.json` for a new configuration. If the file
already contains providers, merge the `loom` member into its existing
`providers` object. The dummy API key makes the provider available to Pi; Loom
does not authenticate it. Opening Pi's `/model` picker reloads the file.

Select the deployment and use Pi's `/thinking` picker to change reasoning:

```sh
pi --model loom/scout
```

The generated `thinkingLevelMap` describes the model's supported choices;
it does not change Pi's transport. For Qwen, the picker offers off, low,
medium, and xhigh. Pi sends the selected `reasoning_effort`, and the model's
source policy applies its corresponding prompt. Streaming usage, leading
`developer` instructions, and ordinary function tools work with the same
configuration. These examples use Pi 1.4 or newer.

Pi's native llama.cpp integration is a different protocol. It owns router
discovery, load, unload and download operations that Loom does not expose. Use
the generic `openai-completions` provider above rather than configuring Loom as
a llama.cpp router.

## Let Oh My Pi discover the mixed catalog

Oh My Pi discovers model modalities, limits, and reasoning choices from the
catalog. Use version 18.13.2 or newer for discovered reasoning levels, and add
this provider to `~/.omp/agent/models.yml`:

```yaml
providers:
  loom:
    baseUrl: http://127.0.0.1:8080/v1
    auth: none
    api: openai-completions
    discovery:
      type: openai-models-list
```

Change the URL to match the client-visible server. `omp models refresh`
refreshes discovery; `omp models --kind all` displays every discovered kind.
Oh My Pi consumes Loom's modality and limit
fields: text-output rows become chat models, image-only rows become image runners,
and embedding-only rows become embedding runners. The ordinary model picker
remains a chat-model view rather than presenting every server deployment as chat.

Select a text model and choose a reasoning level in the client:

```sh
omp --model loom/scout
```

Qwen advertises off (`none`), low, medium, and xhigh. These select its trained
prompt policies, not fixed thinking-token budgets. OMP also sends its ordinary
reasoning-history controls; the source policy preserves thoughts through tool
cycles. No Qwen-specific override is needed. The catalog's default describes
requests that omit a choice; an explicit client preference takes precedence.

Omitted sampling fields use model defaults; supported
`temperature`, `top_p`, `top_k`, and `seed` values belong to the package's
[chat policy](../http-profiles.md#chat-completions).

### Select exact models and enable image tools

OMP selectors include the provider: a server deployment named
`Qwen/Qwen3.8-27B` has selector `loom/Qwen/Qwen3.8-27B`, but its HTTP `model`
value and `modelOverrides` key remain `Qwen/Qwen3.8-27B`. A `loom/*` glob does
not span the extra slash. Explicit selectors also distinguish identical IDs
served by different providers.

With that text deployment and `illustrator` registered, these settings in
`~/.omp/agent/config.yml` select chat and image generation independently:

```yaml
enabledModels:
  - loom/Qwen/Qwen3.8-27B
modelRoles:
  default: loom/Qwen/Qwen3.8-27B
  image: loom/illustrator
generate_image:
  enabled: true
```

Provider definitions belong in `models.yml`; role selection and tool settings
belong in `config.yml`. `enabledModels` scopes chat selection, not the image
runner. See OMP's [model configuration](https://github.com/can1357/oh-my-pi/blob/main/docs/models.md)
and [settings](https://github.com/can1357/oh-my-pi/blob/main/docs/settings.md).

Discovery alone does not give an agent a tool or permission to execute it.
OMP's [image tool](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/generate_image.md)
is disabled by default; a session with an explicit tool allowlist must also
include `generate_image`. Its existing approval policy still applies. An
explicit tool `model` selects one image runner, so a useful first request is:

```text
Use generate_image with model loom/illustrator to create a red fox in snow
at 1024x1024. Return the saved image path from the tool result.
```

The `image` role generates pictures; a `vision` role analyzes images using an
image-capable chat deployment. A text description extracted by vision and then
sent to image generation is not image editing. Reference-image or edit inputs
require a compatible endpoint, client transport, and model package.

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
value takes precedence over a changed environment default. Sampling settings
follow the selected model's source policy, independently of thinking controls.

## Speech clients

A client that accepts raw audio can use `POST /v1/audio/speech` directly with an
explicit `response_format:"pcm"`. Each response is a continuous stream of
signed little-endian PCM16 at 24 kHz, mono. The
[speech walkthrough](stream-speech.md#play-a-learned-models-output) shows a
complete request and a streaming player. It also explains independent requests,
backpressure and cancellation. Available voice IDs and conditioning behavior
belong to the selected package.

Open WebUI's [ordinary TTS adapter](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/routers/audio/__init__.py)
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

Its [realtime voice path](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/routers/audio/realtime.py)
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
build accepts signed PCM16LE with explicit `rate` and `channels` in the file
part's `Content-Type`. It converts the sample rate; channels must match the model.
Changing the filename or MIME label does not convert unsupported bytes into an
accepted format.

For a prepared recognition deployment named `recognizer`, set these values
under **Admin Settings → Audio → Speech-to-Text** in Open WebUI:

| Setting | Value |
| --- | --- |
| Engine | OpenAI |
| API Base URL | Your server URL followed by `/v1` |
| API Key | `local`, or the key required by your deployment proxy |
| STT Model | `recognizer` |
| Request Format | Multipart Upload |

Open WebUI's [transcription adapter](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/routers/audio/__init__.py)
sends multipart audio and reads the returned `text`. Its full installation
uses FFmpeg to prepare browser recordings as MP3 before forwarding them. Keep
`BYPASS_PYDUB_PREPROCESSING=false` and enable MP3 input in the Loom server.
The server itself does not require FFmpeg or Python. This configures the
dictation microphone; read-aloud uses the separate TTS settings above.

Chunked multipart transfer is accepted, but recognition begins after the upload
completes. The endpoint returns finalized text; appendable microphone sessions,
provisional transcripts, turn detection, and interruption require a realtime
session protocol in addition to this upload endpoint.

## Verify the completed workflow

A catalog entry, an assistant's success claim, and a zero client exit code
describe different things. A useful end-to-end check retains the actual tool
results and output artifacts:

| Boundary | Evidence of completion |
| --- | --- |
| Chat transport | Complete JSON, or SSE terminal events with no error object. HTTP 200 alone does not prove a stream succeeded; an error can precede `[DONE]`. |
| Tool execution | A result for each executed call, with no tool error or cancellation. The assistant's prose is not a substitute for that result. |
| Generated media | Nonempty returned image/audio data or a readable saved artifact. For OMP image generation, check `details.imageCount` and `details.imagePaths`; a zero-image result is not necessarily a tool error. |
| Task outcome | Inspect the artifact against the requested content. Successful execution does not prove that an image followed every instruction. |

An image saved before a later chat failure remains a successful tool result.
For example, a `loom_output_error` while the assistant asks to inspect that
image does not invalidate the image or authorize another generation. Retaining
the tool result and artifact path lets the client resume inspection instead of
repeating completed side effects. The [chat error contract](../http-profiles.md#chat-completions)
distinguishes malformed generated output from rejected client input.

## Diagnose the boundary

Test from the same network namespace as the client. A successful request from
the host does not prove that a container can reach the URL:

```sh
curl --fail --silent --show-error "$loom_server_url/readyz"
curl --fail --silent --show-error "$loom_base_url/models"
```

Then test the operation directly with the exact model ID. The
[HTTP profile reference](../http-profiles.md) gives accepted request
fields for chat, image, embedding, speech and transcription operations. Chat
accepts client extensions and nullable native defaults; the model program owns
sampling and template behavior. The [compatibility table](../http-profiles.md#request-compatibility)
distinguishes implemented controls from fields that are accepted without an
effect on generation. Other operation profiles retain their documented schemas.

Common configuration failures are conflicting reasoning/template controls, a
sampling value or image size rejected by the model's source, an operation using
a model of another kind, or a client base URL that duplicates or omits `/v1`.
