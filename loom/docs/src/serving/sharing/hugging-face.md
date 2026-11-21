# Share models on Hugging Face

A Hugging Face model repository can publish Loom programs, model cards, and a
catalog together. Users select a model with `loom-fetch`, then pass the installed
card to `loom-serve`. The publisher needs the installed SDK and ordinary files;
there is no required build system, native extension, or server rebuild.

This guide uses a tiny text model to prove the whole path before introducing
large checkpoints. It performs deterministic arithmetic through real JIT kernels
and the text service. It is not a trained chatbot.

## Start from a working publication

Install the [Loom SDK](../../getting-started/acquiring-loom.md), including the
`hf` extra for Hugging Face acquisition. The example's smoke test requires
`loom-serve` with the x86 CPU JIT and `task` device enabled.

The publication template in each release implements that SDK's text service
ABI. Select the release tag from the installed wheel and check out just the
template directory.

```sh
sdk_tag=$(python -c 'from importlib.metadata import version; print("v" + version("hrx-loom").split("+")[0])')
git clone --depth=1 --filter=blob:none --sparse --branch="$sdk_tag" https://github.com/ROCm/hrx-system.git loom-template
git -C loom-template sparse-checkout set loom/docs/examples/serving/model-publication
cd loom-template/loom/docs/examples/serving/model-publication
python smoke.py
```

For an SDK built from the same revision as this documentation, the
[template ZIP](../../generated/examples/serving/model-publication.zip) packages
the same directory. Extract it and run `python smoke.py` from `model-publication`.
The final JSON reports `"status": "ok"` and
includes the checked completion. Diagnostics and fetch progress use stderr.
The script starts a private CPU server on an automatically selected loopback
port, submits a request, then stops that disposable server.

The downloaded folder contains only the publication:

```text
README.md                      this model's behavior and requirements
AGENTS.md                      links to the authoritative Loom guides
LICENSE
catalog.json                   discovery index
refresh.py                     explicit metadata refresh
smoke.py                       acquisition and expected-output check
model/tiny-text/
  model.json                   portable model definition
  model.json.lock              asset identities
  tokenizer.json
  program/*.loom               preparation, control, storage, and kernels
```

There are no build files or checkout-relative dependencies. The example embeds
its 76-byte safetensors checkpoint in the card; fetch materializes it as an
ordinary parameter file. Larger checkpoints can be local files or references
to another Hugging Face repository, as shown below.

## Edit the model; refresh metadata explicitly

The [portable model card](../model-documents.md#portable-model-cards) connects
programs, services, parameter variants, tokenizers, and presets. Its paths are
relative to the card, not the repository root. Device placement belongs to the
user's deployment, not the published model definition.

| File | Maintained by | Changes that require attention |
| --- | --- | --- |
| `model/*/model.json` | Model author | Program roles, checkpoint sources, service contracts, and presets. |
| `model.json.lock` | `refresh.py` / `loom-fetch` | Asset contents, origins, or the set of selected dependencies. |
| `catalog.json` | `refresh.py` / catalog generator | Model IDs, names, descriptions, services, variants, and presets. |
| `smoke.py` | Model author | The expected response and device requirements of the published example. |

One command refreshes all cards in this template's `model/*/` layout:

```sh
python refresh.py
python smoke.py
```

Refresh deliberately replaces asset commitments. Review its diff alongside the
authored change. Smoke consumes the existing locks: changed assets fail rather
than having their new hashes accepted. It also compares the local catalog to
the cards, so stale discovery metadata fails before serving. The smoke's
expected text and token counts belong to this arithmetic model; a learned port
supplies its own numerical or behavioral checks.

An asset-scoped lock leaves `.loom`, `.loombc`, and auxiliary program files
editable. A kernel edit alone needs no asset-lock refresh. A complete publication
lock can also seal program bytes using `--lock-scope=all`; the
[acquisition reference](https://github.com/ROCm/hrx-system/tree/main/loom/py/loom/model#acquisition)
defines both scopes. Portable bytecode still JITs for the selected target;
[linking and packaging](../../workflows/link-and-package.md) explains how to
produce it.

Both scripts have `--help` and accept `--cache` for reusing acquired assets.
They wrap existing installed commands, which agents can also invoke directly:

```sh
loom-fetch model/tiny-text/model.json --update-lock --all-variants --no-input
python -m loom.model.catalog --output=catalog.json --card=model/tiny-text/model.json=model/tiny-text/model.json
```

`--all-variants` refreshes one lock covering every declared selection without
creating an installation. The catalog command accepts repeated `--card` options.
Additional models can live in sibling directories; each owns its own card and
lock. The template's smoke checks `example/tiny-text`, not every future model
added to a catalog.

## Publish bytecode as XZ

The SDK can stage one built model selection for upload with each bytecode
provider in its own XZ stream. Supply the built card and the authored asset lock.

```sh
python -m loom.model.release build/model.json --asset-lock=model/tiny-text/model.json.lock --output=dist/tiny-text
```

The staged directory contains a selected `model.json`, a complete
`model.json.lock`, one XZ file per bytecode provider, and ordinary copies of
the other program inputs. Upload its contents beside the runtime assets the
card references by relative path. Checkpoints and tokenizers hosted elsewhere
keep their pinned origins and identities from the supplied lock.

`loom-fetch` decodes each provider during installation and verifies its
installed size and SHA-256. Serving, linking, and subsequent offline installs
use the ordinary bytecode. XZ support uses Python's standard `lzma` module.

## Upload and check the published copy

Replace `YOUR_NAME/loom-models` with a repository you own. Authenticate and
create the repository using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
hf auth login
hf repos create YOUR_NAME/loom-models
hf upload YOUR_NAME/loom-models . . --include="model/*" --include="catalog.json" --include="README.md" --include="AGENTS.md" --include="LICENSE" --include=".gitignore" --include="refresh.py" --include="smoke.py" --exclude="*/.loom-fetch.lock" --exclude="*/__pycache__/*"
```

The allowlist publishes the model and author tools without an installed model,
cache, virtual environment, or local log directory. The excluded file is a
local acquisition mutex, not `model.json.lock`. Large folder uploads can span
several commits, so share the catalog only after the upload and smoke check
complete. For updates to an already shared publication, upload and check a new
branch before sharing its release tag or commit.

Now exercise acquisition from the uploaded catalog:

```sh
python smoke.py --catalog=huggingface:YOUR_NAME/loom-models
```

This uses the same result check as the local smoke, but fetches the published
files through Hugging Face. It does not substitute local model files. Access
approval and credentials are handled by `loom-fetch`; `--no-input` keeps the
script from waiting for an interactive prompt.

For releases, append `@REVISION` with a tag or full repository commit to the
repository reference. Fetch resolves a branch or tag once, then acquires
relative cards, locks, and programs at that commit. A branch changing during acquisition cannot
mix publication revisions. An offline fetch uses the cached snapshot; it does
not discover newer uploads.

## Give users a repository URL

The example has one catalog model and one preset, so users can fetch it with:

```sh
loom-fetch https://huggingface.co/YOUR_NAME/loom-models
loom-serve --device=task --pool_backing=fixed --model=./example/tiny-text/model.json --listen=127.0.0.1:8080
```

Fetch uses `./<model.id>` by default, preserving the model's namespace; here
that is `./example/tiny-text`. `--output` selects another installation location.
A repository may instead publish a root `model.json` without a catalog. The
card takes precedence for repository discovery; an explicit `--catalog`
selects the catalog even when both exist. Authentication and malformed metadata
are errors, not reasons to try another entry point.

Users can list a multi-model catalog without downloading payloads, then select
a model and preset without terminal interaction:

```sh
loom-fetch huggingface:YOUR_NAME/loom-models --list --json
loom-fetch huggingface:YOUR_NAME/loom-models:default --model=example/tiny-text --output=./installed --no-input --json
```

The fetch result includes the installed card path and selected catalog
revision. Scripts use its exit status for success and its JSON for the result
or access/error diagnostic. `:default` is the example's preset name, not a
special default-selection rule: only unique choices select automatically.
Append `@REVISION` to pin a branch, tag, or commit, for example
`huggingface:YOUR_NAME/loom-models:default@v1`. The same selector syntax works
with `github:OWNER/REPO`. Direct `model.json` URLs and nested catalog file URLs
also work when root discovery is unnecessary.
The [fetch-and-serve workflow](../workflows/fetch-and-serve.md)
covers caches, materialization, gated checkpoints, and diagnosing failures.

Serving uses the acquired local files and needs no Hugging Face credential.
Model programs are executable code; hashes identify bytes, not publisher
trust. The [sandboxing guide](../sandboxing.md) covers isolation.

## Publish your own model

Replace the example's programs, tokenizer, and parameters with a working
[model package](../model-programs.md), and give the card its own model ID.
Its README describes the model, supported devices, presets, provenance,
license, and output checks. `AGENTS.md` can continue routing agents to the Loom
website rather than copying these instructions into each repository.

A parameter file stored beside a card needs only its relative path. To reuse
an upstream checkpoint without uploading another copy, a card can name an
origin and refer to it from its parameter scope. These fragments belong in
`origins` and the selected variant's service respectively:

```json
"origins": {
  "checkpoint": {
    "kind": "hf",
    "repository": "UPSTREAM/MODEL",
    "revision": "main"
  }
}
```

```json
"parameters": {
  "model": {
    "files": [{
      "path": "parameter/model.safetensors",
      "origin": "@checkpoint/model.safetensors"
    }]
  }
}
```

Refresh resolves the upstream revision and records the acquired asset's
identity. Fetch later uses that lock; it does not silently accept newer weights.
The authored parameter keys, encodings, and layouts must match the preparation
and kernel programs. Referencing a checkpoint does not port its architecture.
Upstream licensing and access requirements still apply.
