# Model cards

A model card describes a published model independently of its host application.
`ModelCard.parse` validates plain JSON, and `select` resolves one service,
variant, optional named partition, and provider set. Neither operation reads
weights, invokes a compiler, or performs network access. The native `loom-serve`
consumer loads a selected card directly with `--model=model.json`. A deployment
document's `model` reference supplies explicit routing, placement, or execution
overrides when needed.

| Section | Meaning |
| --- | --- |
| `model` | Original training-source identity and implementation provenance. |
| `origins` | Shared acquisition roots, referenced as `@name/path`. |
| `programs` | Named sets of linker inputs, libraries, auxiliary data, and default JIT configuration. |
| `services` | Named service contracts with independent ABI versions. |
| `variants` | Compatible provider sets, fixed JIT configuration, scoped parameters, and named tokenizers. |
| `presets` | Canonical selections with execution, request, workload, and preparation options. |

`schema_version: 1` selects the bundled [editor schema](model-card.v1.schema.json).
`$schema` associates it with an editor; consumers never download that URI to
validate a card. JSON comments, duplicate object keys, nonfinite numbers,
unknown structural fields, missing references, incompatible selections, and
escaping destination paths are errors. The Moonshine metadata example declares
inputs and service geometry; execution evidence belongs to the model
implementation and its tests.

Provider-set names have no encoding semantics. A set named `source` may include
bytecode libraries. `inputs` and `libraries` are arrays of path strings or
objects with `path`, optional `format`, importer `options`, and auxiliary `data`.
Auxiliary files remain outside the linker index. A program's `directory` names
its execution context without rebasing the card's file paths.

Every file path describes local placement relative to the card. Its `origin`
describes acquisition separately. For example:

```json
{
  "path": "vocabulary.data",
  "origin": "@checkpoint/tokenizer/tokenizer.json",
  "format": "huggingface-json"
}
```

A program set or parameter scope can declare an origin directory such as
`@checkpoint/`. Entries without an explicit origin append their destination path
exactly once. Without either declaration, a file's origin is its card-relative
path. Model provenance is never an implicit acquisition default. Hashes and
resolved source identities belong to acquisition locks, outside the model card.

`parameters.<scope>.files` is an ordered shard list populating one parameter
index. `tokenizers.<identity>` declares a tokenizer, independently of its
filename. Both accept strings or explicit file descriptors. Variant `config`
scalars override program defaults and reach preparation and native JITs.
Canonical `options` may use only names accepted by the selected service.

A canonical preset can carry coarse guidance in its optional `memory`
object:

```json
{
  "weights_bytes": 180000000,
  "workload": "Original F32 parameters; state, workspace, and runtime overhead excluded."
}
```

`weights_bytes` estimates resident prepared parameters. `transient_bytes`, when
known, estimates additional request, retained state, and workspace storage for
the stated workload. At least one count and a nonempty `workload` description
are required. Counts are unsigned 64-bit integer bytes; omitted components are
unknown. These author estimates identify no device, predict no hardware fit,
and have no scheduling authority. They are independent of download sizes and
acquisition locks.

The picker, list output, fetch plans, and generated reference display the same
guidance. Selecting a different variant, partition, service, or program set
drops it from the selected settings and projected card. Origin overrides and
host execution changes do not turn published estimates into new measurements;
the stated workload and coverage still apply.

A variant declares either `services` directly or `partitions`, whose named
entries each contain their own complete `services` map. Partition names are
arbitrary: `node-0` and `other side` in the
[partitioned fixture](../../../src/loom/serve/fixture/text/partition.card.json)
select distinct weights at the same local path while sharing program files.
Programs, scoped parameters, tokenizers, and JIT values come entirely from the
selected partition; there is no inheritance or automatic union. Machine
assignment and communication between partitions belong to the application.

Canonical presets can name a `partition`. An explicit `--partition`
overrides it; an ambiguous partition requires a choice. Selection preserves only
that partition in the fetched or built card. `ModelCard.choices()` enumerates
the compatible variant, partition, service, and program combinations without
reading files. An all-variant publication lock covers all these combinations;
ordinary acquisition still reads only one selected closure.

A service's kind and version define exported roles, such as `text_prepare_v1`.
An omitted version selects development names such as `text_prepare`; it does
not select the latest ABI. Several services and revisions may share a provider
set. Schema, service, and bytecode revisions have independent compatibility
checks. Native support is documented in the
[serving model contract](../../../docs/src/serving/model-documents.md).

A service may declare components whose roles and lifecycle it owns. Text's
`image` component projects images for text requests:

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

The projector exports `text_image_prepare_v1`,
`text_image_prepare_request_v1`, `text_image_prepare_batch_v1`, and
`text_image_run_v1`. Its revision is independent of the parent text revision;
omitting it selects `text_image_<role>` development names. Components share
the selected providers, JIT `config`, parameter scopes, and tokenizers. They
introduce no separate program or variant choice. A single bytecode library can
provide both contracts; source and mixed-provider selections work the same way.
Declaring a parameter scope makes it available to preparation, without loading
it until that preparation asks for it.

Component `options` lists accepted model option names. Presets may provide their
values and host execution defaults under `components.image.options` and
`components.image.execution`. The native text host prepares its image component
on demand and manages its parameter residency independently. A text-only
selection uses a service without the image component and a file closure without
its additional assets. The
[weighted multimodal fixture](../../../src/loom/serve/fixture/text/image.card.json)
shows a complete card, including shared parameter files and a shared bundle.

## Bazel integration

`loom_model`, exported by `//loom/build_tools/bazel:defs.bzl`, takes a card,
canonical preset, optional provider-set and partition overrides, output
encoding, and explicit Bazel inputs. `files` maps targets to the selected linker-provider
paths. `data` maps exactly their auxiliary inputs. `assets` optionally supplies
runtime parameters or tokenizers, principally for small fixtures and swatches.
Each target supplies exactly one file; missing compiler inputs, extra,
overlapping, and conflicting declarations fail. Runtime assets omitted from
`assets` keep their acquisition origins and transforms in the generated card.
Full model checkpoints belong to `loom-fetch`, outside Bazel's repository,
action, remote execution, and output caches.

The rule performs no acquisition. With `encoding="text"` or `encoding="bc"`, it
normalizes each provider through the configured public `loom-link` toolchain,
preserving provider order and roles. With `encoding="preserve"`, it links each
declared provider into the package unchanged, retaining mixed formats and
importer options without running `loom-link`.
It does not choose service roots or specialize model geometry. Providers and
auxiliary data enter compilation under the card's relative namespace. Supplied
runtime fixtures appear as symlink artifacts and test runfiles; their bytes
never become linker inputs.

The output contains the selected portable `model.json`, normalized providers,
assets, and a separate `serve.json` referencing that card. The default output is
`serve.json`; the `model_card` and `model_files` output groups expose the card
and built files for publication. A package with external runtime assets is
completed by `loom-fetch` before serving. For local development, acquire from
the built card with `--update-lock`, then serve the installed `model.json`
directly. Public releases supply the corresponding lock. Compiler source inputs
are not needed to run a normalized package.

The optional `lockfile` attribute declares the authored acquisition lock as a
test dependency. The original card and lock are carried in runfiles, alongside
the built package, so edits to either invalidate dependent tests. The lock does
not enter the linker action or the `model_files` publication output group: a
normalized publication needs a lock describing its own output bytes.

Model regression tests use current built providers and independently provisioned
parameters and tokenizers. The test-only
[`model_staging`](../../../src/loom/serve/testing/README.md) helper shares these
files into a fresh model directory without copying or hashing weights. The
runner must already have the assets at its explicitly configured path. Their
bytes stay outside Bazel's inputs, runfiles, and caches; the checked-in lock is
their trusted cache identity. Changing external assets without updating that
lock intentionally permits reuse of a cached result. `loom-fetch` verification
is an acquisition concern, independent of the test's output assertions.

Normal Bazel content identity determines test reuse. Changes to the card, lock,
tested program output, runner, runtime, oracle inputs, or execution configuration
invalidate the test; rerunning a build that produces identical outputs does not.
The weighted text fixture checks an actual response through this path. The
recognition fixture checks activity and transcription as independent services
sharing one provider universe, including concurrent streams and both bytecode
and text build outputs.
Image, embedding, decision, and speech fixtures check complete HTTP outputs from
both encodings, including explicitly typed tokenizer files named
`vocabulary.data`. The weighted text card also exposes an independent sequence
service through the same providers; its checks append to retained sessions and
compare activation bytes with a dense arithmetic oracle.

## Generated references

The SDK combines a publisher-authored overview with generated card facts to
produce a standalone model reference. The overview carries intended use,
supported behavior, limitations, licensing, and an installed-tool workflow;
repository build and migration notes stay in developer documentation.

```sh
python -m loom.model.documentation --card model.json --overview OVERVIEW.md \
  --output model.md
python -m loom.model.documentation --card model.json --lockfile model.json.lock \
  --card-url=https://example.org/models/speech/model.json --output model.md
```

The reference lists every compatible variant, partition, service, and program
choice, canonical presets, effective JIT values, scoped parameter shards,
tokenizers, provider order, and importer options. Local destinations and upstream
locations appear separately. Transforms identify both the source archive and the
operation producing the local file. Inline data is summarized by byte count.

A supplied lock must cover every displayed choice. Its immutable repository
revisions determine source links, and its sizes and SHA-256 values describe
output files. Without a lock, links retain the card's authored revisions.
Card-relative sources remain labelled relative paths unless `--card-url` supplies
the publication URL. Rendering reads only the card, optional overview, and
optional lock; it does not acquire files, verify payloads, compile programs,
or establish qualification.
Upstream Hugging Face repositories and HTTP origins are clickable links. The
overview is publisher-authored Markdown, included verbatim in the result;
absolute links keep it independent of a source checkout or neighboring README.
The publisher owns Markdown/HTML sanitization when hosting the generated page.

The same operation is available as `render_markdown` in
`loom.model.documentation`, or as the public Bazel rule:

```starlark
load("@hrx//loom/build_tools/bazel:defs.bzl", "loom_model_documentation")

loom_model_documentation(
    name = "reference",
    card = "model.json",
    overview = "OVERVIEW.md",
)
```

This produces `reference.md` using only declared metadata inputs. Optional
`overview`, `lockfile`, and `card_url` attributes have the same meanings as the
CLI flags.
No weight or provider targets belong in this rule's dependency closure.

## Acquisition

The [Python SDK wheel](../../../packaging/python/README.md) installs the
`loom-fetch` command and includes an optional `hf` provider extra.
`//loom/py/loom/model:loom-fetch` installs the selected closure from a local card
or raw HTTP(S) card URL. The Python API exposes `plan_fetch` and `install`;
`ContentCache` owns document reads and verified payload storage. Listing reads
only the card. Named maps remain available to ordinary JSON tools:

```sh
loom-fetch model.json --list --json | jq '.variants | keys[]'
loom-fetch model.json --preset=streaming --output=./installed --dry-run
loom-fetch model.json --preset=streaming --output=./installed
loom-serve --device=task --model=./installed/model.json
loom-fetch partitioned.json --preset=check --partition=node-0 --output=./node-0
```

Select a device supported by the model; the example uses the task device.
The fetched card retains its preset, and requests use its `model.id` as the
model name. An authored card with multiple presets can be served directly with
`--preset=NAME` alongside one `--model` argument.

On Windows and Unix terminals, missing selections open a keyboard picker. Type
to filter, use arrows or Home/End/Page Up/Page Down to move, and press Enter to
select. Ctrl-U clears the filter; Escape or Ctrl-C cancels. Service, preset,
variant, and partition choices come from the selected card. A preset supplies
its choices directly; explicit flags override them. Program sets follow the
preset or the sole compatible choice. An ambiguous card needs a named preset
or an explicit developer `--programs` override.

The picker asks for an installation directory when `--output` is absent. Screen
controls and input modes are restored on completion or cancellation. Unsupported
terminals use named line input, including `/text` filtering and `/cancel`.
`--no-input`, `--json`, or redirected input/status output resolve choices from
flags and card defaults. They require `--output` for acquisition and never open
the picker.

The acquisition path requires a complete matching `model.json.lock`, or an
explicit `--lockfile` path/URL. A publication lock may cover more variants;
fetch reads only the selected files. `--update-lock` computes a new exact selected set,
replacing the local card's sibling lock or an explicit local `--lockfile`.
For a remote card without an explicit local lock destination, the new lock is
written only beside the installed card. `--no-lock` neither reads nor writes
locks and conflicts with lock update or an explicit lockfile. `--offline`
forbids network access, including document requests; cached documents and locked
payloads remain usable. A floating mutable URL without a lock needs a local
override in offline mode because a historical payload is not a fresh resolution.

For developer swatches, `--no-verify-parameters` skips comparison with the
publisher's parameter byte expectations. It still requires a complete matching
lock and retains its origin revisions. Programs, libraries, auxiliary inputs,
and tokenizers keep their ordinary checks, including a logical source also used
as a parameter at another local path. Format validity and exact range bounds
remain enforced. This mode conflicts with `--no-lock` and `--update-lock`.

Use a new installation directory for this mode. The source lock stays unchanged;
the installed lock records the bytes actually acquired, and `.loom-fetch.json`
lists `unverified_parameters`. JSON output exposes that same list; terminal
output identifies the skipped expectations. Content storage still hashes bytes
as they stream, so the local result has a reproducible identity without another
pass over a remote parent checkpoint. For example, a compact mirror can supply
different swatch values while keeping the publisher's code:

```sh
loom-fetch model.json --preset=swatch --origin=checkpoint=./fixture \
  --offline --no-verify-parameters --output=./experiment
```

Authors can publish a lock covering every declared compatible variant, partition,
service, and program set with `--update-lock --all-variants`. This operation
acquires each unique output once and replaces the lock after every read succeeds.
It creates no installation. `--programs=NAME` restricts the union to compatible
choices using that program set; other selection and output flags do not apply.
Ordinary fetches still download only their selected files, even when variants place
different sources at the same local path. Previously recorded sources outside
the current card are removed from the replacement lock.

```sh
loom-fetch model.json --update-lock --all-variants --dry-run
loom-fetch model.json --update-lock --all-variants
loom-fetch model.json --update-lock --all-variants --programs=source
loom-fetch https://example.org/model.json --update-lock --all-variants --lockfile=publication.lock
```

Publication acquires and hashes the complete output union, using the normal cache;
`--dry-run` resolves metadata without downloading payloads or changing the lock.
A remote card requires an explicit local `--lockfile` destination. Origin
overrides and `--offline` apply to both publication and selected installation.
The program filter lets a source checkout lock every parameter variant without
requiring its generated bytecode. Variants that do not support the selected set
are excluded. An unknown or unused program set fails before source acquisition.
The resulting lock contains exactly that union; generated representations receive
their own lock when their bytes are published.

Locks use a separate [versioned editor schema](model-lock.v1.schema.json).
`origins` records requested roots and resolved repository commits. Each `files`
record contains its logical `origin`, final whole-file `size`, and `sha256`.
Transformed parameter records also contain the normalized `transform`; the hash
identifies the resulting archive, without requiring a read of unused source bytes.
A file's destination name, parameter scope, and JIT configuration are independent
of that identity. `--origin=checkpoint=/local/mirror` replaces transport while retaining
the original lock identity; it never falls back to the original host. Unknown
aliases, missing local files, and mismatched locked bytes fail explicitly.

Whole-file acquisition supports HTTP(S), local, inline, and Hugging Face sources.
HF support is loaded only when needed and requires the optional
`huggingface_hub>=2.2,<3` package. An explicit local/HTTP origin override needs no
HF package or upstream access check. HF downloads use the SDK's ordinary cache;
Loom verifies and shares those bytes without another payload cache. HTTP transfers
validate status, framing, byte count, and locked hash; redirect destinations never
replace logical source coordinates.

HF `blob` and `resolve` model-card URLs resolve one immutable repository snapshot.
The sibling lock and card-relative files retain that snapshot, including cached
offline reads. Named floating HF origins resolve once per operation, and lock
updates record the selected immutable revisions. Cached files avoid authentication
requests. Missing content checks repository access and selected file metadata
before payload transfer.

The Python provider accepts a `wait_for_access(action, delay, cancelled)` policy.
Without it, confirmed gates and credential/permission/rate-limit failures return
`AccessRequired` with structured remediation. A confirmed gate supplies the model
page and effective account; it never accepts terms or requests approval itself.
One coordinator refreshes credentials and waits across all shard workers. Generic
permission and inaccessible-repository failures retain their distinct meanings.

On an interactive terminal, `loom-fetch` displays the approval URL and effective
account, then waits and rechecks with backoff. `o` opens the page, Enter checks
again, and `q` or Ctrl-C cancels. An approval on the website resumes the selected
command; the downloader reloads saved credentials between checks. An environment
token still overrides that saved login, so changing it requires restarting the
command with the variable updated or unset.

`--no-input` and `--json` disable terminal interaction. A gate returns a nonzero
status immediately; JSON mode emits an `action_required` record containing the
repository, reason, URL, account, and credential source name. Explicit
`--wait-for-access` enables automatic rechecking in these modes, with status on
stderr and the final result on stdout. Offline operation never checks access.

Ctrl-C also stops active metadata requests and file transfers. Cancellation joins
the operation's writers before releasing the installation, keeps completed cache
files reusable, and removes its incomplete ordinary HTTP staging files. Hugging
Face retains ownership of its cache and resumable partial downloads. The Python
API exposes the same operation cancellation through `ContentCache.cancel()`.

Blocking remote IO uses at most four reusable transfer interpreters per Python
process. Reuse retains SDK imports between calls; each call exclusively owns its
worker until it finishes. Cancellation terminates and joins that worker while
other callers continue. A changed environment, import path, or working directory
starts a fresh worker, and Python shutdown joins the remaining idle workers.
Forked callers create their own workers without taking ownership of the parent's
transfers.

Files stream into verified content storage with bounded buffers and concurrent
transfers. The default `--materialize=share` links cache or local-source files
into the installation, so changing only preset does not copy weights.
Filesystems without sharing require an explicit `--materialize=copy` choice.
Shared files remain shared: editing one view can alter another, and a later
locked fetch detects that modification. Verified existing destinations can be
reused; conflicts fail before payload transfer. Fetch never prunes an output.

The selected `model.json` is published after the complete closure verifies.
An installation lock is a projection of those verified commitments. Relative
runtime paths are retained, while original card-relative acquisition sources
are anchored to a document origin. A serving deployment references the local
card separately; native serving performs no network access or lock processing.
The fetch-to-serve fixture checks source, bytecode, mixed, and offline packages
against the same weighted response, and rejects substituted code before
publication.

## Streaming parameter materialization

A parameter file can select complete tensors from a safetensors checkpoint and
align the resulting archive's payload base during acquisition:

```json
{
  "path": "weights.data",
  "format": "safetensors",
  "origin": "@checkpoint/model.safetensors",
  "transform": {
    "kind": "safetensors",
    "version": 1,
    "alignment": 4096,
    "tensors": ["encoder.layers.0.weight", "encoder.layers.0.bias"]
  }
}
```

This descriptor belongs in `parameters.<scope>.files`. Its explicit `format`
identifies the local archive consumed by the runtime. The operation preserves
tensor names, dtypes, shapes, bytes, and string metadata. Omit `tensors` to retain
the complete checkpoint. Alignment defaults to 8 and accepts powers of two
through 1048576; it aligns the payload base by padding the JSON header. Individual
tensors retain the format's contiguous layout. Partial tensors, quantization,
and arbitrary conversion commands are not part of this operation.

For a selected tensor set, acquisition reads the eight-byte prefix, bounded
header, and exactly the named tensor ranges. Each HTTP response must honor the
range with status 206, matching byte coordinates, and identity encoding. A
server rejecting ranges fails with a diagnostic; no whole-file retry occurs.
A full repack reads the payload in one streaming range, preserving its order.
Both paths use 1 MiB transfer chunks and a 16 MiB header limit per worker, with
at most four acquisition workers. An uncached HF transform uses the optional
SDK's HTTP client, authentication, and gate handling without placing the whole
checkpoint in the HF cache.

By default, only the resulting archive is retained. `--preserve-sources` also
retains the remote intervals actually read, with their offsets in an acquisition
receipt. It does not fetch omitted tensors. Existing local files and complete HF
cache entries are reused in place and remain owned by their original storage.
An existing verified result is shared across presets; adding preservation
reads the missing source intervals without writing a second result archive.
Preservation requires additional storage for the selected source bytes.

```sh
loom-fetch model.json --preset=swatch --output=./swatch
loom-fetch model.json --preset=swatch --output=./swatch --preserve-sources
loom-fetch model.json --preset=swatch --output=./another --offline
```

The logical acquisition identity includes the origin and normalized transform,
independently of local placement or JIT configuration. A compact local
`--origin` override with the same selected tensor facts, bytes, and string metadata
produces the same locked result even when its source offsets differ. Matching
cached results avoid source access. Native serving reads only the resulting `path` and `format`;
the Bazel model rule likewise consumes already materialized assets. Neither
executes acquisition transforms.

## Catalog discovery

An SDK wheel can associate itself with a compatible catalog tag independently of
its Python package version. `loom-fetch --list --json` lists that catalog's named
`models` map. Select an entry with `--model=MODEL_ID`, then use the same variant,
preset, provider-set, and output flags as a direct card. A direct card
path or URL does not consult the catalog.

`--catalog=./catalog.json` selects a local publication for authoring or CI.
`--catalog=github:OWNER/REPOSITORY@TAG_OR_COMMIT` selects a GitHub publication.
The client resolves a tag once, including annotated tags, then reads
`catalog.json`, its relative cards, and their sibling locks at that commit.
An advancing tag affects the next discovery operation. It cannot change a
selection already being acquired. Missing tags fail explicitly; the client
never substitutes `main` or an older online result.

`--offline` reuses the last successfully loaded catalog snapshot. A failed
refresh leaves that snapshot intact. Listing reads only discovery metadata;
it does not read every card or check access to model weights. The selected card
still owns service ABI and model-format admission.

Generate a catalog from model cards without reading their payloads:

```sh
python -m loom.model.catalog --output=catalog.json \
  --card=model/moonshine-ai/moonshine-streaming-tiny/model.json=model/moonshine-ai/moonshine-streaming-tiny/model.json
loom-fetch --catalog=./catalog.json --list --json | jq '.models | keys[]'
loom-fetch --catalog=./catalog.json --model=moonshine-ai/moonshine-streaming-tiny \
  --preset=streaming --output=./installed
```

Each `--card` maps its publication-relative path to a local input. Publication
must place that card at the declared relative path. The generated index contains
names, service versions, variants, and canonical preset names. It is a
discovery index, not qualification evidence. Its
[versioned schema](model-catalog.v1.schema.json) is bundled with the Python SDK.

Catalog acquisitions publish `.loom-fetch.json` before publishing `model.json`.
This receipt records the source catalog, resolved commit, selected card location,
and card digest. It participates in the same conflict checks and rollback as the
rest of the installation. Native hosts do not consume acquisition receipts.
