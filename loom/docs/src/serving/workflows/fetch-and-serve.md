# Fetch and serve a model

`loom-fetch` installs the files for one model preset. Pass the installation
directory to `loom-serve` to run it. Its `model.json` connects the model's programs,
parameters, tokenizer, and service contract; a separate server configuration is
optional.

## Quickstart

With [the SDK installed](../../getting-started/acquiring-loom.md), including
the `hf` extra, fetch Moonshine Streaming Tiny and start it on an AMD GPU:

<div class="loom-quickstart" markdown>

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny/model.json --preset=streaming --output=./moonshine
loom-serve --device=amdgpu --workers=4 --listen=127.0.0.1:8080 --model=./moonshine/model.json
```

</div>

Leave the server running. In another terminal,
[transcribe a recording](#transcribe-a-recording) or
[check startup status](#start-the-server). The sections below explain source
selection, acquisition, and serving in detail; the installed `model.json` is
all the server needs from fetch.

To share your own programs and model cards, the
[Hugging Face publishing guide](../sharing/hugging-face.md) provides a small
copyable repository, explicit metadata refresh, and an end-to-end smoke test.

## Fetch a published repository

A publisher can share a repository URL, without a `model.json` suffix or an
installation-directory flag:

```sh
loom-fetch https://huggingface.co/benvanik/hrx-loom-serve-example
```

This arithmetic text example installs into `./example/tiny-text`, matching its
model ID. The equivalent provider-explicit command is:

```sh
loom-fetch huggingface:benvanik/hrx-loom-serve-example
```

Both providers use the same selectors:

| Source | Selection |
| --- | --- |
| `huggingface:OWNER/REPO` | Hugging Face repository's default branch. |
| `github:OWNER/REPO` | GitHub repository's default branch. |
| `PROVIDER:OWNER/REPO:PRESET@REVISION` | A named model preset at a branch, tag, or commit. Both suffixes are optional. |
| `https://github.com/OWNER/REPO` or `https://huggingface.co/OWNER/REPO` | Repository URL; append `/tree/REVISION` to select a revision. |

`:PRESET` means exactly `--preset=PRESET`. It names a publisher-authored
configuration, not a checkpoint filename or a special quantization convention.
Bare `OWNER/REPO` remains a local path; it never implicitly selects a provider.

Repository discovery reads root `model.json`, or root `catalog.json` when the
card is absent. One catalog model and one preset select automatically. With
multiple choices, use `--list`, `--model=ID`, and `--preset=NAME`, or the terminal
picker. An access failure or malformed card stops discovery instead of hiding
the error by trying another document. `--catalog=PROVIDER:OWNER/REPO` explicitly
selects the catalog even when a root card is also present.

Fetch pins the repository to one commit before reading relative cards, locks,
and programs. A branch advancing during acquisition cannot mix revisions.
The output defaults to `./<model.id>`, preserving namespaces. `--output=DIR`
selects another location. An ID that is not a portable relative path requires
this explicit override; a default path also cannot traverse symlinks or
junctions. Fetch reports the destination before payload transfer and never
silently overwrites a conflicting installation.

The remaining walkthrough transcribes a short English recording with
[Moonshine Streaming Tiny](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny).
It uses the original checkpoint and the Loom library's AMD GPU implementation.
A supported AMD GPU is required to serve it. For a CPU-only example without
learned weights, use the [arithmetic embedding walkthrough](serve-models.md).

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

Fetch reports its current phase, file counts, and available byte counts on
stderr. Known byte totals include a compact bar, percentage, and estimated time
remaining once progress establishes a rate. These updates also appear in
redirected logs and with `--no-input`; they require no terminal.
For example, a large file can produce these status lines (illustrative counts):

```text
loom-fetch  Download  [###.........]  25%  1.0 GiB / 4.0 GiB  34.1 MiB/s  ETA 1m 30s  received 512.0 MiB (17.1 MiB/s)  model.safetensors  2/3 files
loom-fetch  Verify    [############] 100%  4.0 GiB / 4.0 GiB  model.safetensors  2/3 files
loom-fetch  Complete  3/3 files
```

stdout contains only the installed card path, or the final JSON result with
`--json`:

```sh
loom-fetch "$model_url/model.json" --preset=streaming --output=./moonshine --no-input --json >installed.json 2>fetch.log
```

Use `--quiet` to suppress progress while retaining errors and access guidance.
Download progress describes reconstructed file bytes. Hugging Face's network
transfer bytes appear on the same row as `received`; these may differ from file
bytes because of cached or compressed chunks. ETA uses the file's observed
average progress, excluding bytes already present at resume, not the network
counter. It estimates this file stage, not the remaining installation. Before
progress establishes a rate, it reads `ETA estimating`; unknown file totals have
no ETA. Verification reports bytes checked, including cache hits. Ongoing
updates are limited to one line every thirty seconds, while new phases and final
completion appear promptly. A complete installation is reported only after its
model card has been published.

On Windows PowerShell, the same acquisition is:

```powershell
$modelUrl = 'https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny'
loom-fetch "$modelUrl/model.json" --preset=streaming --output=./moonshine
```

Fetch reads the adjacent `model.json.lock`, resolves the selected dependencies,
and verifies identities within the lock's declared scope before publishing the
installed card. An asset-scoped lock protects weights, tokenizers, and other
assets while leaving programs editable. A complete publication lock additionally
checks program bytes. The output
contains local program and asset paths and retains the chosen preset. Supply a
new output directory for another selection; a changed selection never silently
replaces an installed model.

The default `--materialize=auto` copies programs and their auxiliary files so
you can edit them independently. It shares cached weights and other assets;
two installations that differ only in execution settings can share the same
parameter files. Fetch tries hardlinks, then readable symlinks. When neither is
available, automatic mode warns and copies the affected files instead; allow
enough space for those independent copies. The warning also appears with
`--quiet` and stays on stderr in JSON mode.

`--materialize=copy` makes every file independent when moving an installation
or storing it on removable media. Copies need no link support and report byte
progress. A failed or cancelled copy does not publish a partial model card;
completed cached downloads remain reusable.
`--materialize=share` also shares programs with their source or cache, so edits
can affect other installations. This explicit mode fails if sharing is
unavailable instead of silently copying. Keep the backing cache and local
source files available while using shared files.

Hardlinks cannot cross filesystems. Symlinks can, but Windows requires symlink
creation privileges or Developer Mode, and network paths have separate
link-evaluation policies. Fetch probes the selected paths without changing
those settings. Copying cannot bypass disk quotas, read-only storage, or
per-file size limits: FAT32, for example, cannot hold a single file of 4 GiB or
more. Use a large-file filesystem such as exFAT or NTFS for those weights.

With an asset-scoped lock, editing a program requires no lock update, and a
repeat fetch into the same installation preserves that edit. Git can version
and share the model's source files. Program origin aliases, such as `@library`,
do not require per-file hashes in an asset-scoped lock. `--offline` reuses cached
assets and the last fetched source bytes for a new installation, without
network requests. The [model format reference](https://github.com/ROCm/hrx-system/tree/main/loom/py/loom/model#acquisition)
describes lock updates and complete publication coverage.

Hugging Face gating is handled during acquisition. If access is required,
fetch displays the repository's access URL and authentication guidance. Use
`hf auth login` when credentials are required. Complete
the publisher's approval process in your browser, then recheck from the prompt;
approval can be delayed. Fetch never accepts terms on your behalf. In scripts,
`--no-input` fails with the required action instead of waiting for a terminal.
`--wait-for-access` explicitly enables automatic rechecking for unattended use.

### Resolve publication and access failures

A card URL, publisher approval, and downloader credentials are separate inputs.
If a GitHub card or its adjacent lock returns HTTP 404, check that both files
are published at the selected path and revision. A local checkout or unmerged
branch can contain a model that is not on the published branch. The pinned
example above keeps the card, lock, programs, and recording together; changing
only the card's revision does not update the rest of that example.

For a Hugging Face access request, approval belongs to the account downloading
the files. Browser approval with one account does not authorize another account's
credential. In the environment performing acquisition, these commands identify
the effective account and establish a saved login:

```sh
hf auth whoami
hf auth login
```

An `HF_TOKEN` environment variable overrides the saved login. With `--json`,
the fetch access diagnostic reports the credential source, not its value. After
changing an environment override, restart fetch so it sees the new environment.
A container or agent sandbox may intentionally have no access to the host's
saved login.
Acquisition can instead run in an authorized host shell with
`--materialize=copy`, followed by exposing that self-contained installation to
the server. Serving local files requires no Hugging Face credential.

Without `--wait-for-access`, `--no-input` reports an unmet access requirement and
exits unsuccessfully. It does not accept terms for the user. `--wait-for-access`
is the explicit waiting policy, not a remedy for a missing published file. The
publisher's [gated-model instructions](https://huggingface.co/docs/hub/models-gated)
explain approval, and the [authentication guide](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-auth-login)
describes credential setup.

## Start the server

```sh
loom-serve --device=amdgpu --workers=4 --listen=127.0.0.1:8080 --model=./moonshine
```

`--model=./moonshine` selects `./moonshine/model.json`. An explicit model-card
or deployment-document path is also accepted. Relative paths inside the
document resolve from its directory.

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
device. A publisher may offer source or bytecode sets; `--programs`
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
