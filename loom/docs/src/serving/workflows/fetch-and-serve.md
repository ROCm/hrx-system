# Fetch and serve a model

`loom-fetch` installs the files for one model preset. Pass the resulting
`model.json` to `loom-serve` to run it. The card connects the model's programs,
parameters, tokenizer, and service contract; a separate server configuration is
optional.

This walkthrough transcribes a short English recording with
[Moonshine Streaming Tiny](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny).
It uses the original checkpoint and the Loom library's AMD GPU implementation.
Start with [the SDK installed](../../getting-started/acquiring-loom.md), including
the `hf` extra, and a supported AMD GPU. For a CPU-only example without learned
weights, use the [arithmetic embedding walkthrough](serve-models.md).

## Inspect the available choices

The example pins a published library commit so the card, lock, programs, and
sample recording come from the same revision. The following commands use a POSIX
shell; the Windows acquisition command follows below.

```sh
model_url=https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny
loom-fetch "$model_url/model.json" --list
```

The choices include the `original` parameter variant, `transcription` service,
and `streaming` preset. Inspect the same facts as JSON, optionally with `jq`:

```sh
loom-fetch "$model_url/model.json" --list --json | jq '{presets, variants, origins}'
```

Listing reads the card without downloading weights or programs. Presets combine
a variant, service, program set, and canonical execution settings. Here,
`streaming` selects the original weights, source programs, and four pending
requests. The preset name describes the model's execution configuration; it
does not select a file format.

## Fetch the selected files

```sh
loom-fetch "$model_url/model.json" --preset=streaming --output=./moonshine
```

On Windows PowerShell, the same acquisition is:

```powershell
$modelUrl = 'https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny'
loom-fetch "$modelUrl/model.json" --preset=streaming --output=./moonshine
```

Fetch reads the adjacent `model.json.lock`, resolves the selected dependencies,
and verifies their identities before publishing the installed card. The output
contains local program and asset paths and retains the chosen preset. Supply a
new output directory for another selection; a changed selection never silently
replaces an installed model.

The default shared materialization reuses cached weights. Two installations
that differ only in execution settings can share the same parameter files.
`--materialize=copy` requests independent files when moving an installation or
using a filesystem that cannot share them. Keep the backing cache and local
source files available while using a shared installation.

Hugging Face gating is handled during acquisition. If access is required,
fetch displays the repository's access URL and authentication guidance. Use
`hf auth login` when credentials are required. Complete
the publisher's approval process in your browser, then recheck from the prompt;
approval can be delayed. Fetch never accepts terms on your behalf. In scripts,
`--no-input` fails with the required action instead of waiting for a terminal.
`--wait-for-access` explicitly enables automatic rechecking for unattended use.

## Start the server

```sh
loom-serve --device=amdgpu2 --workers=4 --listen=127.0.0.1:8080 --model=./moonshine/model.json
```

Leave this terminal running. The server loads only local files; it does not
contact Hugging Face or interpret the acquisition lock. Preparation, linking,
and JIT compilation run on workers. A JSON `ready` message reports that the
HTTP listener is available. Model preparation progresses separately:

```sh
curl --fail http://127.0.0.1:8080/v1/status
curl --fail http://127.0.0.1:8080/v1/models
```

`/v1/models` lists prepared models. If this model does not appear, inspect
`/v1/status`: its `loom.registration` progresses to `ready`, or `loom.error`
explains a preparation failure. An incompatible SDK/model contract is reported
here instead of becoming an inference result.

Use `curl.exe` in Windows PowerShell to select curl rather than a shell alias.
The request's model name is the card's ID,
`moonshine-ai/moonshine-streaming-tiny`. Fetch has already retained `streaming`,
so no `--preset` or separate `serve.json` is needed here.

## Transcribe a recording

In another terminal, download the small reference recording and submit it:

```sh
curl --fail --location https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny/test/short.pcm --output=short.pcm
curl --fail-with-body -H 'Expect:' http://127.0.0.1:8080/v1/audio/transcriptions -F model=moonshine-ai/moonshine-streaming-tiny -F 'file=@short.pcm;headers="Content-Type: audio/pcm;rate=16000;channels=1"'
```

On Windows PowerShell, download the same recording and preserve the multipart
header's quotes with the `--%` stop-parsing token:

```powershell
curl.exe --fail --location https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny/test/short.pcm --output=short.pcm
curl.exe --% --fail-with-body -H "Expect:" http://127.0.0.1:8080/v1/audio/transcriptions -F model=moonshine-ai/moonshine-streaming-tiny -F "file=@short.pcm;headers=\"Content-Type: audio/pcm;rate=16000;channels=1\""
```

The response contains:

```json
{"text":"Good day, streaming speech should remain clear when the listener interrupts."}
```

Replace `short.pcm` with your recording to transcribe other audio. This raw
format is signed little-endian PCM16, mono, at 16 kHz. Renaming a WAV or MP3 file
does not convert it to PCM. The [audio transcription reference](../http-profiles.md#audio-transcription)
describes encoded uploads, supported content types, and response options. This
Moonshine preset supports English recordings up to 40.96 seconds and greedy
decoding; it does not provide speaker diarization or word alignment.

Ctrl-C in the server terminal stops admission and joins accepted work before
exiting. The fetched model remains available for the next run.

## Discover other models and services

A release SDK can name a compatible library catalog. With that association,
`loom-fetch --list` shows its models and bare `loom-fetch` opens the terminal
picker. A development SDK requires a card URL or explicit catalog:

```sh
loom-fetch --catalog=github:ROCm/hrx-loom-library@4317508bf5adb251a284ea8dc4e5a24a11bd1c17 --list --json
```

This snapshot includes Moonshine speech recognition, Pocket speech synthesis,
Qwen3.8 Flash-Next text generation, Qwen3.8-27B text and vision, Qwen3 Embedding,
and Krea image generation. Pocket's `streaming` preset uses Alba, Marius, and
Anna; `streaming-noncommercial` adds Jean.

Each catalog entry identifies a runnable package with published programs and
asset sources. The
[model map](https://github.com/ROCm/hrx-loom-library/tree/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model)
records each package's supported devices and presets.

Acquisition and startup are the same across modalities. The selected service
determines the request contract:

| Service | HTTP request |
| --- | --- |
| Text, including optional image input | [Chat completions](../http-profiles.md#chat-completions) |
| Speech synthesis | [Audio speech](../http-profiles.md#speech-synthesis) |
| Speech recognition | [Audio transcription](../http-profiles.md#audio-transcription) |
| Embeddings | [Embeddings](../http-profiles.md#embeddings) |
| Image generation | [Image generation](../http-profiles.md#image-generation) |
| Known-token sequences | [Sequences](../http-profiles.md#known-token-sequences) |
| Finite decisions | [SystemOne](../http-profiles.md#finite-decisions) |

A program set can contain several Loom text or bytecode files and libraries.
The server links the selected providers together and specializes them for the
device. A publisher may offer source, bytecode, or mixed sets; `--programs`
overrides a preset only when another compatible set is actually published.
The source checkout in this example publishes the source set. Its bytecode
choices require a separately built publication.

## Reuse local assets and configure a deployment

After a successful fetch, `--offline` forbids all network requests and reuses
cached documents and payloads. An origin override can use an already provisioned
checkpoint directory while retaining lock verification:

```sh
loom-fetch "$model_url/model.json" --preset=streaming --offline --origin=checkpoint=/srv/checkpoints/moonshine --output=./moonshine-local
```

That directory must supply the paths relative to the named origin, including
`model.safetensors` and `tokenizer.json`. A missing file fails; the override
does not fall back to the network. `--no-lock` is an explicit development mode
for experiments with modified inputs. Publishers use `--update-lock` to record
the exact selected set, or `--update-lock --all-variants` for a complete
publication. Adding `--programs=source` locks all compatible parameter variants
of the source publication without acquiring generated bytecode. The [model format guide](https://github.com/ROCm/hrx-system/tree/main/loom/py/loom/model)
defines those authoring workflows and transformations in detail.

For a routing alias, device assignment, or different server admission settings,
write a small [serving document](../model-documents.md#portable-model-cards)
referencing the fetched card. The model remains reusable by other hosts; HTTP
listeners and deployment policy belong to the server configuration.
