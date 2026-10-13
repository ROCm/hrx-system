# Serving model documents

A model package is one JSON document plus the programs and assets it references.
The same object is accepted by `--model=path/model.json` and
`POST /v1/models/register`. Registration requires no model-specific native code
or server rebuild.

This document describes the arithmetic embedding package in the
[serving walkthrough](../workflows/serve-models.md):

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
array. Kinds are `text`, `image`, `embedding` and `decision`.

| Field | Role |
| --- | --- |
| `prepare` | Cold VM program declaring stages, specialization, parameter domains and model metadata. |
| `control` | Warm VM program implementing the modality's request and execution contract. |
| `sources` | Native command, kernel, configuration and motif providers. |
| `options` | JSON object passed to source preparation unchanged; defaults to `{}`. |
| `pending_requests` | Request credit, default `32`; positive integer. Text/image count queued requests, while embedding/decision also count the active cohort. |

Programs accept text or portable Loom bytecode (`.loom` or `.loombc`), detected
from contents. Providers may mix formats. Bytecode is serialized IR, not a
precompiled device executable: the server specializes and JITs it for the live
device. Source-declared parameter preparation policies accept the same formats.

Provider order is not an execution schedule. Loom's linker indexes symbols,
resolves imported declarations and selects definitions reachable from named
roots. The document supplies inputs to that linker; it does not describe a
second layer graph. Preparation/control VM roles compile separately from native
providers. [Module composition](../integration/module-composition.md) explains
the underlying source and dependency contract.

Native configuration does not interpret `options` keys or rewrite strings
inside that object. Their meaning belongs to the source program.

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

## Image and finite-vector definitions

Image requires `checkpoint`, an asset directory interpreted by source
preparation. `adapter` optionally supplies another asset path. `width` and
`height` default to 384 pixels, and `text_tokens` defaults to 512; all are
positive, with architecture constraints checked by the source.

`command_quantum` defaults to zero, meaning one whole-program submission.
A positive value is a soft command-count target for full-barrier-aligned
submission parts. It is not a wall-clock limit. Parts retain the invocation's
workspace and weights until final completion.

Embedding and decision definitions require an `assets` directory. Their
architecture and batching settings belong in source-owned `options`, not new
native per-model fields.

The referenced assets must match the package's preparation policy. Loading
checks its declared parameter keys and extents; matching a filename alone does
not establish checkpoint identity or original tensor dtype/shape compatibility.

## Local path binding

A file document resolves relative references from its own directory. For
`--model=/srv/models/vectors/model.json`, `prepare.loom` means
`/srv/models/vectors/prepare.loom`. References can reach shared providers outside
that directory; this is not a filesystem sandbox.

An absolute `base_directory` overrides the containing directory. HTTP has no
containing file, so relative references need an explicit absolute base on the
server machine. A document with only absolute references needs no base; source
programs must also return absolute parameter and policy paths in that case.

Authored references are retained separately from the base and bound at file I/O.
The loader does not rewrite JSON to machine-absolute paths, resolve symlinks or
infer remote mounts. Source-returned parameter/policy references use the same
base. Tokenizer paths returned relative to an asset directory use that directory.

HTTP registration changes live state only. It does not rewrite model documents
or flagfiles; persistence of operator changes belongs to the management client.
