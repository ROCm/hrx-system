# Serving model documents

A model package is one JSON document plus the programs and assets it references.
The same object is accepted by `--model=path/model.json` and
`POST /v1/models/register`. Registration requires no model-specific native code
or server rebuild.

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

## Deployment fields

| Field | Contract |
| --- | --- |
| `name` | Required nonempty visible-ASCII routing alias, independent of source symbols. |
| `definition` | Required object specifying a modality, programs, assets and capacities. |
| `admission` | `"open"` by default; `"drained"` registers metadata without reading programs, loading weights or allocating model storage. |
| `device` | Nonnegative index into the ordered CLI device group; required when the group has more than one member. |
| `base_directory` | Optional absolute server-local base for relative file references. |

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

Every definition contains `kind`, `prepare`, `control` and a nonempty `sources`
array. Kinds are `text`, `image`, `embedding`, `decision`, `speech` and `recognition`.

| Field | Role |
| --- | --- |
| `prepare` | Cold VM program declaring stages, specialization, parameter domains and model metadata. |
| `control` | Warm VM program implementing the modality's request and execution contract. |
| `sources` | Shared VM functions, native commands, kernels, configuration and motif providers. |
| `options` | JSON object passed to source preparation unchanged; defaults to `{}`. |
| `pending_requests` | Request credit, default `32`; positive integer. Text/image/speech/recognition count queued requests, while embedding/decision also count the active cohort. |
| `variant_capacity` | Text/image/embedding/decision retained native executable choices, default `16`, including the original, pending compilations and failed attempts; zero disables native reconfiguration. |

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

Text requires `weights`, a checkpoint file, and `tokenizer`, a tokenizer JSON
file. Native capacity fields are:

| Field | Default | Meaning |
| --- | --- | --- |
| `rows` | `4` | Concurrent execution rows, from 1 through 16. |
| `context_capacity` | `16384` | Per-row logical context limit in token positions, at most 262144. |
| `pool_capacity` | `65536` | Shared logical cache token positions, at most 4194304. |
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

## Recognition definitions

Recognition requires `kind:"recognition"` and an `assets` directory. It uses the
same `prepare`, `control`, `sources` and source-owned `options` fields as other
modalities. The package declares input sample rate and channels, bounded input
credit, maximum recording length, transcript-event capacity, persistent request
storage and numerical batch capacity.

`pending_requests` bounds the additional host queue. Each admitted request owns
its input clock, unconsumed PCM, model state and transcript revisions. Source
compatibility tags group next numerical operations; different recording lengths
and input positions do not require different registrations. Source programs own
language, prompting, decoding, segment boundaries and any speaker/alignment
evidence.

The [transcription profile](http-profiles.md#audio-transcription) accepts uploaded
WAV, MP3, or FLAC when enabled, and always accepts PCM16LE with matching sample
geometry. It returns final JSON or text. The underlying continuing-input service
also accepts bounded PCM with explicit end-of-input, independently of HTTP.
The normal pin/unpin, activation, drain and removal controls apply. Recognition
does not accept `variant_capacity`; changing its execution contract requires a
new registration.

## Local path binding

A file document resolves relative references from its own directory. For
`--model=/srv/models/vectors/model.json`, `prepare.loom` means
`/srv/models/vectors/prepare.loom`. References can reach shared providers outside
that directory; this is not a filesystem sandbox.

An absolute `base_directory` overrides the containing directory. A JSON-only HTTP
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
