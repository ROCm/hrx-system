# Serving model documents

A portable model card describes identity, programs, variants, services, and
canonical presets. `loom-fetch` installs its selected dependencies, and
`loom-serve --model=installed/model.json` loads that local selection directly.
[Fetch and serve a model](workflows/fetch-and-serve.md) walks through acquisition
and a complete request.

A deployment document gives a model its server name, device, admission policy,
and execution capacities. The same object is accepted by
`--model=path/serve.json` and `POST /v1/models/register`. Registration requires
no model-specific native code or server rebuild.

A deployment can describe its programs directly or reference a portable model
card. A card owns model identity, services, program sets, parameter variants,
and presets independently of the server.

After fetching a model, pass its card directly to the server:

```sh
loom-fetch model.json --preset=interactive --output=./installed
loom-serve --device=task --model=./installed/model.json
```

This example selects the `interactive` preset and uses the task device; select
a device supported by the model's programs. Fetch retains the selected preset,
so serving needs no separate deployment file. Requests use the card's `model.id`
as their model name. Programs and weights are prepared on worker threads.

An authored card with several presets requires a choice, for example
`loom-serve --device=task --model=model.json --preset=interactive`.
`--preset` applies to one card. For multiple deployments, custom routing aliases,
device placement, or execution overrides, use the referencing documents below.

This document describes the arithmetic embedding package in the
[serving walkthrough](workflows/serve-models.md):

```json
{
  "name": "vectors",
  "definition": {
    "kind": "embedding",
    "prepare": "prepare.loom",
    "control": "control.loom",
    "sources": ["config.loom", "embed.loom"],
    "assets": ".",
    "options": {"batch_units": 64},
    "pending_requests": 16
  }
}
```

## Portable model cards

A text deployment can select a local card and override its execution defaults:

```json
{
  "name": "chat",
  "model": {
    "path": "model.json",
    "preset": "interactive",
    "programs": "mixed"
  },
  "definition": {"kind": "text", "rows": 2}
}
```

The referenced `model.json` can contain:

```json
{
  "schema_version": 1,
  "model": {"id": "example/model", "name": "Example model"},
  "programs": {
    "mixed": {
      "inputs": ["prepare.loom", "control.loombc"],
      "libraries": ["kernel.loombc"],
      "config": {"model.width": 64}
    }
  },
  "services": {"chat": {"kind": "text", "version": 1}},
  "variants": {
    "small": {
      "services": {
        "chat": {
          "programs": ["mixed"],
          "config": {"model.width": 32},
          "parameters": {
            "model": {
              "files": [
                "weights-00001.safetensors",
                {"path": "weights-00002.data", "format": "safetensors"}
              ]
            }
          },
          "tokenizers": {
            "text": {"path": "vocabulary.data", "format": "huggingface-json"}
          }
        }
      }
    }
  },
  "presets": {
    "interactive": {
      "service": "chat",
      "variant": "small",
      "programs": "mixed",
      "execution": {"rows": 4, "context_capacity": 128, "prefill_capacity": 32}
    }
  }
}
```

These programs implement the text service ABI, including `text_prepare_v1` and
the corresponding `text_<role>_v1` control exports. Service `version` is
independent of `schema_version` and the bytecode encoding version. Omitting it
selects the unversioned development names, `text_<role>`. A missing versioned
export is an error; bare exports are not substituted. Several services and ABI
revisions can coexist in one provider set. The host checks the selected service.
Native card execution supports service ABI 1 for text, sequence, image,
embedding, decision, speech, recognition, and activity, plus their unversioned
development contracts. Each uses its own `<kind>_<role>_v1` exports for both
preparation and control. Selecting a different service does not rename exports
or substitute another service's ABI.

Text services can declare an independently versioned image projector:

```json
{
  "services": {
    "chat": {
      "kind": "text",
      "version": 1,
      "components": {"image": {"version": 1}}
    }
  }
}
```

The component uses `text_image_prepare_v1`, `text_image_prepare_request_v1`,
`text_image_prepare_batch_v1`, and `text_image_run_v1`. These exports may share
one bytecode bundle with the primary text exports. Both use the card's selected
providers and named parameter scopes. The server supports image component ABI 1
and the unversioned `text_image_<role>` development contract. Other selected
component roles or revisions fail before program loading.

Presets can supply `components.image.options` for option names declared by the
component, and `components.image.execution.specialization_cache` for its idle
request-code cache. A serving deployment's `definition.image` can override
`options` and `specialization_cache`; program and parameter paths remain card
facts. Referenced cards resolve on a worker even for drained registrations,
making component metadata available without preparing programs or loading
weights. Component management becomes available when that resolution completes.
Image preparation and parameter activation remain lazy and independent of the
text model. Evicting image parameters preserves text sessions and checkpoints.

Program set names are opaque: `mixed` could be renamed `source` without changing
its contents. Each input and library contributes to the link index with its
declared role. A provider object can specify `path`, `format`, importer
`options`, and auxiliary `data` files outside the link index. The native loader
accepts Loom text and bytecode providers; unsupported importer formats or
options fail explicitly. File names are relative to the card directory,
including parameter shards and tokenizer assets. `directory`, when present on
a program set, supplies the source program's execution context and does not
rebase the declared file paths.

Each `parameters` key names an independent parameter index. Source preparation
uses `prepare.source_scope(scope)` to select that index; its ordered typed files
populate one shared checkpoint source. `prepare.parameter(stage, binding, input,
source, policy)` binds that source to a resident policy input. Several stages
can share the same source, and demand reads can use it without a resident binding.
Text service uses the `text` tokenizer identity. Image, embedding, decision,
speech, recognition, and activity preparation return a tokenizer identity;
services supporting inputs without tokenization may return an empty string.
Sequence consumes token IDs
directly and needs no tokenizer. File names need not identify their role or
encoding:
an explicit `format` supports renamed assets, while shorthand strings use the
loader's format inference. Ambiguous or unsupported encodings fail.

Speech service variants may declare original recordings under `voices`. Each
key is the ID used by speech requests; `file` follows the same destination and
origin rules as other assets. Optional `content_type` uses the upload media-type
syntax. Encoded recordings can omit it and use container detection; raw PCM
requires its rate, channels, and optional `format=f32le` (`s16le` is the default).
The `file` itself has no `format`, since the media type owns that distinction:

```json
"voices": {
  "narrator": {
    "file": {"path": "voice/narrator.wav", "origin": "@recordings/original.wav"},
    "content_type": "audio/wav"
  }
}
```

`loom-fetch` acquires and locks these original bytes. Speech ABI 2 prepares each
recording through the native upload lifecycle before the model becomes ready;
failed preparation fails model registration. Models can also accept
[uploaded voices](http-profiles.md#voice-registration) after startup. Prepared
conditioning belongs to the loaded model and is not a parameter archive.

Variant `config` values override program defaults and reach both preparation
and generated native stages. Values are compiler scalars: booleans, finite
numbers, or strings. Source preparation may specialize individual stage keys.
Presets supply execution defaults and optional source-owned
`options`; accepted option names are declared by the service. A deployment's
explicit settings override canonical defaults. Explicit service, variant, and
program selections must remain compatible; an ambiguous selection fails.
Checkpoint paths, tokenizer paths, providers, and asset directories belong to
the card and cannot be overridden through the deployment definition. Image
adapters are selected per request, independently of the base model card.

Presets may include informational `memory` estimates:
`weights_bytes` for resident prepared parameters, `transient_bytes` for additional
request/state/workspace storage, and a required `workload` description stating
assumptions and coverage. Unknown counts are omitted. The fetch picker and
generated model reference display this guidance; native serving uses its actual
allocation and scheduling contracts. Changing the selected variant, partition,
service, or program set drops the estimate from a fetched or built card.

A partitioned variant declares `partitions` instead of `services`. Each named
partition supplies its own complete service, provider, configuration, and asset
choices. The deployment's `model.partition` selects one, overriding a canonical
configuration when specified. Names carry no host or device assignment; the
application owns communication and placement between separately loaded models.

Admission records the reference without opening the card. Preparation resolves
it on a worker before constructing the native model. The server reads local
files and does not fetch origins or inspect integrity locks. Acquisition and
materialization happen before registration. A parameter descriptor may include
an acquisition `transform`; `loom-fetch` produces the declared local file, and
the server reads its `path` and `format` without interpreting that transform.

## Deployment fields

| Field | Contract |
| --- | --- |
| `name` | Required nonempty visible-ASCII routing alias, independent of source symbols. |
| `definition` | Required object specifying a modality, programs, assets and capacities. |
| `model` | Optional local card reference with `path` and optional `preset`, `service`, `variant`, `partition`, and `programs` selectors. Programs and assets then come from the card. |
| `admission` | `"open"` by default; `"drained"` registers metadata without reading programs, loading weights or allocating model storage. |
| `device` | Nonnegative index into the ordered CLI device group; required when the group has more than one member. |
| `base_directory` | Optional local file base; relative values resolve from the containing document. |

Placement is a deployment property, outside the reusable `definition`. Drained
registrations still require a valid device selection. Indices are checked
before publication or model I/O; `/v1/status` publishes the ordered group under
`server.resources.devices`. A deployment executes on one selected device.

Unknown fields, duplicate keys, wrong JSON types and wrong-modality fields
reject. Names already registered return a conflict instead of being replaced.
Source programs validate architecture-specific options and constraints during
preparation. Opening admission starts asynchronous preparation; explicit
activation can prepare a model while its admission stays closed.

## Programs and dependencies

Every direct definition contains `kind`, `prepare`, `control` and a nonempty
`sources` array. Kinds are `text`, `sequence`, `image`, `embedding`, `decision`, `speech`,
`recognition`, and `activity`.

| Field | Role |
| --- | --- |
| `prepare` | Cold VM program declaring stages, specialization, parameter domains and model metadata. |
| `control` | Warm VM program implementing the modality's request and execution contract. |
| `sources` | Shared VM functions, native commands, kernels, configuration and motif providers. |
| `options` | JSON object passed to source preparation unchanged; defaults to `{}`. |
| `pending_requests` | Request credit, default `32`; positive integer. Text/sequence/image/speech/recognition/activity count queued requests, while embedding/decision also count the active cohort. |
| `variant_capacity` | Text/sequence/image/embedding/decision retained native executable choices, default `16`, including the original, pending compilations and failed attempts; zero disables native reconfiguration. |

Programs accept text or portable Loom bytecode (`.loom` or `.loombc`), detected
from contents. Providers may mix formats. Bytecode is serialized IR, not a
precompiled device executable: the server specializes and JITs it for the live
device. Source-declared parameter preparation policies accept the same formats.

Provider order is not an execution schedule. Loom's linker indexes symbols,
resolves imported declarations and selects definitions reachable from named
roots. The document supplies inputs to that linker; it does not describe a
second layer graph. Preparation/control VM roles link their helpers from this
catalog and compile separately from native commands. Each VM entry file is a
primary input; library-private symbols stay private. An entry file may also
appear in `sources`. [Module composition](../integration/module-composition.md) explains
the underlying source and dependency contract.

Native configuration does not interpret `options` keys or rewrite strings
inside that object. Their meaning belongs to the source program.
Text control programs also receive this object to establish request defaults;
model packages can define sampling defaults without native per-model flags.

## Text definitions

Direct text definitions require `weights`, a checkpoint file, and `tokenizer`,
a tokenizer JSON file. Card references supply their typed inputs instead.
Native capacity fields are:

| Field | Default | Meaning |
| --- | --- | --- |
| `rows` | `4` | Concurrent execution rows, from 1 through 16. |
| `context_capacity` | `16384` | Per-row logical context limit in token positions, at most 262144. |
| `pool_capacity` | `65536` | Shared logical cache token positions, in whole source-defined pages. |
| `prefill_capacity` | `512` | Largest automatically specialized packed token shape; not a prompt-length limit. |
| `checkpoint_capacity` | `0` | Explicit pinned endpoint slots, independent of execution rows. |
| `session_cache_capacity` | `0` | Evictable idle endpoints beyond execution rows; nonzero values also reserve one admission-exchange endpoint. |
| `prefix_cache_capacity` | `0` | Evictable exact intermediate-prefix endpoints; zero disables capture. |
| `admission_overtakes` | `rows` | Successful fitting admissions allowed to pass a blocked oldest request; zero selects FIFO. |
| `max_tokens` | `512` | Default request output limit, from 1 through 16384. |
| `draft_depth` | `0` | Proposed tokens after each anchor, from 0 through 7; requires matching source support. |
| `continuation_epochs` | `1` | Device-fed epochs per scheduling turn, either 1 or 2. |

Context, pool and prefill capacities are positive. Prefill capacity cannot
exceed context capacity and must fit an anchor plus every draft proposal.
Two-epoch continuation requires nonzero draft depth. A source can impose tighter
shape limits or reject unsupported drafting modes. Endpoint capacities are
nonnegative and their combined native index space must fit signed 32-bit indices.

The shared pool can grow independently of the per-session context limit and
prefill shape. Its token count must fit both host indexing and a signed 64-bit
integer. Source preparation declares the page size and storage layout; the
resulting pool must contain at most 4,294,967,295 physical pages, and each region
must fit its declared allocation. These representation limits do not guarantee
that every page can be backed within the server's physical memory budget.

Logical capacity is not eagerly committed physical storage under elastic
backing. Sessions grow by blocks; admission reserves their requested completion
high-water. Shared prefixes can reference the same resident blocks. Changing
deployment capacities requires a new registration; they are not process flags.

### Image input

A text definition can include an optional `image` object with required
`prepare`, `control` and `assets` paths, plus source-owned `options` defaulting
to `{}`. It uses the enclosing definition's `sources`, uploaded source package,
path base and device. This is an input projector, distinct from an
image-generation deployment.

The optional `image.specialization_cache` object controls idle request-shape
executable reuse, with the same fields and defaults as finite models below.
It does not pin vision weights or allocate image workspace at registration.

The component implements the finite request/cohort contract and owns independent
preparation and parameter residency. Registration and ordinary text generation
leave it dormant; an image not covered by retained state or an explicit
[`component:"image"` activation](workflows/operate-model-server.md#control-weight-residency-independently)
starts preparation. Primary warmup and pinning do not implicitly warm or pin it.
The text source must also declare image limits and implement ordered media
framing and mixed input. A declaration alone does not give a text program
vision capability. See [image-conditioned text](model-programs.md#image-conditioned-text)
for the source boundaries.

## Known-token sequence definitions

`kind:"sequence"` advances retained model state using exact caller-supplied token
IDs and returns source-defined opaque bytes. It requires `weights` but no
tokenizer, chat template, vocabulary head, or token sampler. This supports a
stateful stage inside a larger inference pipeline without making it imitate a
chat model.

Sequence definitions share text's `rows`, `context_capacity`, `pool_capacity`,
`prefill_capacity`, `checkpoint_capacity`, `session_cache_capacity`, and
`admission_overtakes` fields and defaults. Admission reserves only the supplied
known positions, rounded to cache blocks; it does not reserve generated tokens.
Execution uses packed epochs and the same row/cache/checkpoint ownership.

`tokenizer`, `max_tokens`, `draft_depth`, `continuation_epochs`,
`prefix_cache_capacity`, and `image` are not sequence fields. Reuse is explicit
through a retained sequence handle or a named checkpoint, not automatic chat
prefix matching. Source-defined request policy belongs in `options`.
See the [sequence HTTP profile](http-profiles.md#known-token-sequences) and
[source contract](model-programs.md#known-token-stages).

## Image and finite-vector definitions

Image requires `checkpoint`, an asset directory interpreted by source
preparation. `adapter` optionally supplies another asset path. `width` and
`height` default to 384 pixels, and `text_tokens` defaults to 512; all are
positive, with architecture constraints checked by the source. Width and height
are defaults, not the only accepted request size. An explicit image `size`
reaches source preparation and can select or declare shape-specialized code
without changing checkpoint residency.

`command_quantum` defaults to zero, meaning one whole-program submission.
A positive value is a soft command-count target for full-barrier-aligned
submission parts. It is not a wall-clock limit. Parts retain the invocation's
workspace and weights until final completion.

Embedding and decision definitions require an `assets` directory. Their
architecture and batching settings belong in source-owned `options`, not new
native per-model fields.

For image, embedding, and decision deployments, `specialization_cache` bounds
executable choices declared by request preparation:

```json
"specialization_cache": {"capacity": 16, "max_bytes": 16777216}
```

These are the defaults. `capacity` counts idle choices and `max_bytes` counts
their retained declaration, entry, portable-program, and native-key/artifact
storage. Least-recently-used choices retire first. Either limit can be zero to
disable idle retention. A single larger request can still compile and execute;
its choice retires afterward instead of remaining cached. Reusing shapes shares
the immutable source index and checkpoint residency. Fixed source/index capacity,
temporary compiler storage, and driver-private allocations are separate from
this byte budget.

This cache is independent of `variant_capacity`, which controls explicitly
registered native replacements. Requests use the source generation selected
when their image or cohort is prepared; selecting another replacement affects
subsequent requests without changing one already being compiled or executed.

The referenced assets must match the package's preparation policy. Loading
checks its declared parameter keys and extents; matching a filename alone does
not establish checkpoint identity or original tensor dtype/shape compatibility.

## Speech definitions

Speech requires an `assets` directory. Source-owned `options` select model
capacities, voices and preparation policy. The package declares reusable voice
conditioning, separate mutable state for each request, waveform geometry and a
bounded output block. A model may perform several stages before producing audio.

For example, the [speech walkthrough](workflows/stream-speech.md) uses:

```json
{
  "name": "speech",
  "definition": {
    "kind": "speech",
    "prepare": "prepare.loom",
    "control": "control.loom",
    "sources": ["generate.loom"],
    "assets": ".",
    "pending_requests": 4
  }
}
```

`pending_requests` bounds additional host-queued requests. Resident request
capacity and numerical batch capacity are separate source declarations; they
are not inferred from this queue limit. Each request retains its row across
changing compatible batches, and different voices or playback positions need
not require separate registrations. A package supplies compatibility tags and
additive work units for its next operations.

The [speech HTTP profile](http-profiles.md#speech-synthesis) exposes incremental
24 kHz mono PCM; a package producing other geometry rejects during preparation.
Voice identifiers and conditioning assets belong to the package. The service
keeps request state alive until accepted device work and output copies retire,
allowing subsequent requests to reuse the model safely.

The normal pin/unpin, activation, drain and removal controls also apply to
speech. Pinning governs parameter residency, independently of mutable-state
admission and compute scheduling. Speech does not currently accept
`variant_capacity`; replacing its execution contract requires a new registration.

## Recognition and activity definitions

Recognition requires `kind:"recognition"` and an `assets` directory. It uses the
same `prepare`, `control`, `sources` and source-owned `options` fields as other
modalities. The package declares input sample rate and channels, bounded input
credit, maximum recording length, supported event types, event count and byte
bounds, persistent request storage and numerical batch capacity. Native requests
explicitly select transcript or activity output, or both when the source supports
them. HTTP transcription selects transcripts.

An `activity` deployment uses the same fields and continuing-input execution
contract for native speech activity detection. The source must support activity
events; a `recognition` source must support transcripts. Either can additionally
support the other event family, with each native request selecting its outputs.
These are deployment roles, not fixed model architectures or turn policies.
An activity deployment has no HTTP inference endpoint. Its `/v1/models` entry
has `loom.kind:"activity"`, omits `loom.endpoint`, and reports the prepared
source's actual output modalities, including `activity`.

`pending_requests` bounds the additional host queue. Each admitted request owns
its input clock, unconsumed PCM, model state and transcript revisions. Source
compatibility tags group next numerical operations; different recording lengths
and input positions do not require different registrations. Source programs own
language, prompting, decoding, segment boundaries and any speaker/alignment
evidence.

The [transcription profile](http-profiles.md#audio-transcription) accepts uploaded
WAV, MP3, or FLAC when enabled, and always accepts PCM16LE with an explicit
sample rate and matching channel count. Raw rate conversion works without audio
codecs. It returns final JSON or text. The underlying continuing-input service
also accepts bounded PCM with explicit end-of-input, independently of HTTP.
The normal pin/unpin, activation, drain and removal controls apply to both roles.
Neither recognition nor activity accepts `variant_capacity`; changing its
execution contract requires a new registration.

## Local path binding

A file document resolves relative references from its own directory. For
`--model=/srv/models/vectors/model.json`, `prepare.loom` means
`/srv/models/vectors/prepare.loom`. References can reach shared providers outside
that directory; this is not a filesystem sandbox.

An absolute `base_directory` overrides the containing directory. A relative value
such as `program/model` selects that directory inside the document's directory.
A JSON-only HTTP
request has no containing file, so relative references need an explicit absolute
base on the server machine. A document with only absolute references needs no
base; source programs must also return absolute parameter and policy paths in
that case.

Authored references are retained separately from the base and bound at file I/O.
The loader does not rewrite JSON to machine-absolute paths, resolve symlinks or
infer remote mounts. Source-returned parameter/policy references use the same
base. Tokenizer paths returned relative to an asset directory use that directory.

An [uploaded source package](http-profiles.md#uploaded-source-packages) supplies
a separate namespace for relative programs and metadata, based on the `request`
part's logical filename. It supports preparation/control VM programs, native
providers, tokenizers and weight-preparation policies without placing them on the
server filesystem. Missing members fail without a local-file fallback. Parameter
checkpoints still use the filesystem base; explicit absolute program or metadata
paths also select server-local files. Registration retains uploaded bytes through
delayed preparation and final retirement.

HTTP registration changes live state only. It does not rewrite model documents
or flagfiles; persistence of operator changes belongs to the management client.
