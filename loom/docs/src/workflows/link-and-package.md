# Link and package modules

`loom-link` combines text and bytecode modules, applies configuration bindings,
and retains either an archive or the dependency closure of selected roots. Its
inputs are explicit: it never searches a directory or pulls in modules that the
invocation did not provide.

## Inputs and libraries

Positional inputs are primary modules. Repeated `--library=path` options add
provider modules searched after the primary inputs. Every supplied path also
publishes its exact string as an opaque `module.import` provider key. The root
below declares `@project_layer` locally and constrains its definition to the
provider bound as `"layer.loom"`:

```loom title="root.loom"
--8<-- "examples/module-composition/root.loom"
```

```shell
loom-link root.loom \
  --library=layer.loom \
  --library=kernels.loom \
  --root=@entry \
  --mode=link \
  --to=bc \
  --output=entry.loombc
```

Inputs may mix `.loom` text and `.loombc` bytecode. `--from=auto`, the default,
detects their encoding independently. `--to=text|bc` controls the one linked
module written to `--output` or standard output.

The provider key is an exact string match. `loom-link` does not canonicalize the
path or search for imported files. Embeddings may bind other logical keys
through the LoomC API. `module.import` is optional: without one, a compatible
exact definition can satisfy a declaration from the explicit input universe.
Primary versus library is a lookup priority, not a reachability rule: a symbol
becomes live because it is a root or reachable dependency, not because it came
from a primary input. Provider module order also does not override template
matching; contracts, signatures, facts, requirements, and explicit provider
priority decide which template is eligible.

## Inspect before linking

List indexed symbols without materializing an output module:

```shell
loom-link root.loom --library=providers.loom --list-symbols
```

Print the selective link plan for one root:

```shell
loom-link root.loom \
  --library=providers.loom \
  --root=@entry \
  --print-plan
```

The plan shows why each symbol is live before the linker streams the selected
definitions. This is the first place to inspect an unexpected dependency,
missing declaration target, or provider candidate set. It is cheaper and more
direct than debugging target lowering after the wrong source program was
selected.

List the configuration schema visible to the input program:

```shell
loom-link root.loom --print-config-schema
```

This exposes the declared keys, types, and constraints an application or build
must bind without scraping source text.

## Select one program

Selective mode starts from explicit roots and retains their reachable program:

```shell
loom-link model.loom \
  --library=kernel.loom \
  --library=motif.loom \
  --mode=link \
  --root=@elementwise_transform \
  --to=text \
  --output=elementwise-transform.loom
```

This produces a closed program for the facts known at the link boundary. In
the targetless `elementwise-transform` example it selects the portable provider
and discards the unresolved wave32 alternative. When target facts arrive only
at compilation, the [source-to-artifacts
walkthrough](../getting-started/source-to-artifacts.md#follow-one-composition-to-low)
uses archive mode instead and lets `loom-compile` perform the target-aware
selection. In either mode, unrelated private symbols can disappear as soon as
the chosen boundary no longer needs them.

Repeated `--root=@symbol` options select several roots from one catalog. Add
`--include-input-exports` when exported symbols from the requester inputs
should join the explicit root set. Library exports remain resolution candidates
and never become roots merely because they are public in their source archive.

Configuration bindings are applied to the composed analysis module before each
reachability and template-selection step. This lets newly reachable code expose
additional demands while value and target predicates eliminate provider
alternatives that are impossible for the selected configuration:

```shell
loom-link root.loom \
  --library=providers.loom \
  --root=@entry \
  --config=model.hidden_size=4096 \
  --require-resolved-config \
  --output=entry.loom
```

`--require-resolved-config` turns a remaining required configuration declaration
into a link failure. Without it, a partial module may retain unresolved config
for a later composition or JIT boundary.

## Package an archive

Archive mode preserves every non-stripped symbol in deterministic input order:

```shell
loom-link motif.loom kernel.loom \
  --mode=archive \
  --to=bc \
  --output=kernel-library.loombc
```

Use archive mode for reusable catalogs that should retain several helpers,
templates, kernels, or configurations for later root selection. The archive is
still Loom bytecode, not target-native code; later link or compile invocations
can select a much smaller reachable program from it.

Remove test- and benchmark-only symbols from a deployment archive with
`--strip-check`:

```shell
loom-link library.loom \
  --mode=archive \
  --strip-check \
  --to=bc \
  --output=runtime-library.loombc
```

Keeping checks is useful for an authoring or qualification package. Stripping
them is useful when the embedding owns validation or when executable
specifications should not ship in the runtime artifact. It does not change the
production symbols' semantics.

## Link transitive dependencies incrementally

An exact call or launch is legal in an independently verified module only when
that module contains the compatible definition or declaration. Libraries may
satisfy those declared dependencies when linked. They may not retroactively
make an undeclared call valid.

For a declaration with no imports, selective linking considers compatible
definitions from the explicitly supplied universe. If `module.import` names the
declaration, selection is restricted to providers bound to those keys. A strict
link rejects a reachable imported declaration that remains unresolved. Add
`--allow-unresolved` to produce a reusable partial artifact instead.

The checked composition example has three independently verifiable modules:

```text
root.loom  --@project_layer-->  layer.loom  --@scale-->  kernels.loom
```

The layer provider introduces its own kernel demand:

```loom title="layer.loom"
--8<-- "examples/module-composition/layer.loom"
```

Supplying only the layer consumes the root's satisfied import and retains the
new transitive import in standalone bytecode:

```shell
loom-link root.loom \
  --library=layer.loom \
  --root=@entry \
  --allow-unresolved \
  --to=bc \
  --output=partial.loombc
```

The generated partial module is ordinary Loom IR. It contains the selected
layer body, the reachable root, and the unsatisfied kernel contract:

```loom title="partial.loom"
--8<-- "generated/examples/module-composition/partial.loom"
```

A later invocation can reload that artifact in a fresh process and supply only
the remaining provider:

```shell
loom-link partial.loombc \
  --library=kernels.loom \
  --root=@entry \
  --to=text \
  --output=linked.loom
```

```loom title="linked.loom"
--8<-- "generated/examples/module-composition/linked.loom"
```

The final module contains each reachable definition once and no longer needs
an import or declaration. The linker did not chase the layer's key during the
first invocation, and it did not require the root to name the layer's transitive
dependencies. A diamond of callers reaching the same global declaration still
selects one compatible definition by symbol identity; path reachability does
not clone one copy per route.

Run the exact formatting, partial-link, bytecode-reload, and final-link sequence
with:

```shell
loom/docs/examples/module-composition/run.sh build/module-composition
```

`template.apply` follows the parallel family-provider model. The using module
declares the family with `template.decl`; each explicitly supplied library may
repeat that declaration and contribute providers. Specialization selects an
eligible provider from that explicit universe. `module.import` is not used for
template families, and source path and library order never stand in for
matching rules.

## Choose the output for the next boundary

| Output | Use |
| --- | --- |
| Linked `.loom` | Review the selected source, inspect provider reachability, or feed a text-oriented tool. |
| Linked `.loombc` | Cache or distribute a compact linkable program for later specialization and compilation. |
| Archive `.loombc` | Package a reusable catalog whose roots will be selected later. |
| Stripped archive | Package production symbols without executable checks. |

The next boundary decides when remaining facts become known. A build may invoke
`loom-compile` immediately. A model loader may combine the bytecode with more
libraries and configuration. A JIT embedding may keep one indexed archive and
select different roots and targets per workload.
