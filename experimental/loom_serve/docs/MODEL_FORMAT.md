# Model documents

A model is one JSON document plus its referenced programs and assets. The
document is both a startup input to `--model=path/model.json` and the body of
`POST /v1/models/register`. Registration does not require rebuilding the server.

This complete example is the [embedding fixture](../models/fixtures/embedding/model.json):

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

`name` is the public routing alias, independent of names declared by the model
program. It must be nonempty visible ASCII. An optional `admission` field is
`"open"` by default; `"drained"` registers metadata without reading programs,
preparing weights or allocating model storage. Opening admission starts normal
asynchronous preparation. Requests can wait for that preparation within the
server's bounded holding budget. Duplicate aliases reject instead of replacing
a live registration.

An optional top-level `device` is a nonnegative index into the ordered
`--device` group, also shown in `/v1/status` under `server.resources.devices`.
It may be omitted only when the group contains exactly one device. Multiple
devices require explicit placement, including for drained registrations;
out-of-range indices reject before publication or model I/O. For example,
`"device": 1` selects the second CLI device. Placement belongs to the deployment,
not its reusable `definition`. Each model currently executes on one selected
device; registration does not imply cross-device weight or state sharing.

## Programs and dependency resolution

Every definition contains `kind`, `prepare`, `control` and a nonempty `sources`
array. Kinds are `text`, `image`, `embedding` and `decision`.

* `prepare` names the cold VM program declaring stages, specialization,
  parameter domains and model metadata.
* `control` names the warm VM program implementing the modality's request and
  execution contract.
* `sources` lists native command, kernel, configuration and motif providers.

Each program accepts Loom text or portable Loom bytecode (`.loom` or
`.loombc`), detected from its contents. Providers may mix formats. Bytecode is
serialized IR, not a precompiled device executable: the server still specializes
and JITs for its live device. The same format support applies to source-declared
weight preparation policies.

Provider order is not a dependency schedule. The public LoomC linker indexes
symbols; provider declarations describe imports, and named roots select
reachable definitions. The JSON file supplies files, not another linker or a
layer graph. Cold/warm VM roles compile separately from the native providers.

## Modality settings

Settings under `definition` select native capacity and asset inputs:

| Kind | Required asset inputs | Other settings |
| --- | --- | --- |
| `text` | `weights` checkpoint file, `tokenizer` JSON file | `rows`, `context_capacity`, `pool_capacity`, `prefill_capacity`, `checkpoint_capacity`, `session_cache_capacity`, `prefix_cache_capacity`, `admission_overtakes`, `max_tokens`, `draft_depth`, `continuation_epochs` |
| `image` | `checkpoint` asset directory | `adapter`, `width`, `height`, `text_tokens`, `command_quantum` |
| `embedding`, `decision` | `assets` directory | Model-specific behavior in `options` |

All kinds accept `pending_requests` and an `options` object. Options default
to `{}` and belong to the cold source program; native code does not interpret
their keys or rewrite strings within them. The
[configuration contract](../../../loom/src/loom/serve/server/configuration.h) documents ownership and
units; the [operator guide](OPERATING.md#configuration-and-resource-boundaries)
explains capacity/admission behavior. The parser rejects unknown, duplicate,
wrongly typed and wrong-modality native fields. Source programs validate
additional architecture-specific constraints.

The learned [text](../models/qwen/model.json) and
[image](../models/krea/model.json) documents show complete provider closures.
Their asset paths must point at the intended checkpoint before use.
A matching filename is not proof of checkpoint compatibility: current loading
uses source preparation policy and reflected parameter keys/extents, not a
complete original tensor dtype/shape or model-identity declaration.

## Local path binding

For a document loaded from a file, relative references use that file's
directory. For example, `--model=/srv/models/vectors/model.json` resolves
`prepare.loom` to `/srv/models/vectors/prepare.loom`. Paths may point outside
that directory to shared providers. The document is not a filesystem sandbox.

An optional top-level `base_directory` overrides the file directory. It must
be an absolute local directory. An inline HTTP document has no containing file:
relative references require that explicit base. Paths refer to the server's
filesystem, not the HTTP client's machine. A document with all absolute
references needs no base; source programs must also return absolute parameter
and policy paths when no base exists.

The loader retains authored locators separately from their local base and binds
them only at file I/O. It does not rewrite model JSON into machine-absolute
paths, resolve symlinks or infer remote mounts. Parameter/policy references
returned by the source use the same base. Tokenizer assets returned relative to
an asset directory resolve within that directory.

To register the example over HTTP after copying its complete closure into
`/srv/models/vectors`:

```sh
curl http://127.0.0.1:8080/v1/models/register \
  -H 'Content-Type: application/json' \
  -d '{"name":"vectors","base_directory":"/srv/models/vectors",
       "definition":{"kind":"embedding","prepare":"prepare.loom",
       "control":"control.loom","sources":["config.loom","embed.loom"],
       "assets":".","options":{"batch_units":64}}}'
```

## Startup lists

`--model` is repeatable. Omitting it starts an empty server. Ordinary flagfiles
can keep model registrations alongside process settings without introducing
another JSON catalog:

```text
--device=task
--pool_backing=fixed
--listen=127.0.0.1:8080
--model=/srv/models/vectors/model.json
--model=/srv/models/decisions/model.json
```

Pass `--flagfile=/path/to/serve.flags`. Each line is a literal flag, with no
shell expansion. Relative `--model` and `--flagfile` arguments use the process
working directory, including arguments inside a flagfile; the references
inside each model document use its own directory. Repeated model entries on
the command line and in flagfiles combine in encounter order.

HTTP registration changes live state only. It does not rewrite model files or
flagfiles. The management client owns persistence of operator changes.

The single-modality authoring executables also take a model document for program
roles, providers and source-owned options. Their explicit shape and asset flags
still drive individual experiments; they are not the shared server's startup
configuration interface.
