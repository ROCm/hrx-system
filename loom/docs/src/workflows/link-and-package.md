# Link and package modules

`loom-link` combines text and bytecode modules, applies configuration bindings,
and retains either an archive or the dependency closure of selected roots. Its
inputs are explicit: it never searches a directory or pulls in modules that the
invocation did not provide.

## Inputs and libraries

Positional inputs are primary modules. Repeated `--library=path` options add
provider modules searched after the primary inputs:

```shell
loom-link root.loom \
  --library=providers.loombc \
  --root=@entry \
  --mode=link \
  --to=bc \
  --output=entry.loombc
```

Inputs may mix `.loom` text and `.loombc` bytecode. `--from=auto`, the default,
detects their encoding independently. `--to=text|bc` controls the one linked
module written to `--output` or standard output.

Primary versus library is a lookup priority, not a reachability rule. A symbol
becomes live because it is a root or reachable dependency, not because it came
from a primary input. Provider module order also does not override template
matching: contracts, signatures, facts, requirements, and explicit provider
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
  --root=@transform \
  --config=guide.element_count=1009 \
  --config=guide.workgroup_size=64 \
  --to=text \
  --output=transform.loom
```

This is the exact composition exercised by the checked [mental-model
example](../getting-started/mental-model.md#follow-one-composition-to-low).
`@transform` reaches its declared kernel and the template providers needed for
`guide.transform`; unrelated private symbols can disappear.

Repeated `--root=@symbol` options select several roots from one catalog. Add
`--include-exported-roots` when the module's exported symbols should join the
explicit root set.

Configuration bindings are applied to each materialized input before dependency
walking. This lets value and target predicates eliminate provider alternatives
whose requirements are impossible for the selected configuration:

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

## Declarations make partial linking explicit

An exact call or launch is legal in an independently verified module only when
that module contains the compatible definition or declaration. Libraries may
satisfy those declared dependencies when linked. They may not retroactively
make an undeclared call valid.

Selective linking resolves declarations that the supplied modules can satisfy
and preserves unresolved declarations in a valid partial result. For example, a
root that declares dependencies from `module_a` and `module_b`, linked with only
`module_a`, retains the `module_b` declaration. The linker does not chase a
filesystem import graph or silently add the missing module. A later closed
artifact boundary must supply or reject that dependency.

`func.apply` follows the parallel contract-provider model. A library makes
provider candidates visible; specialization selects an eligible provider from
the explicit set. The source path and library order never stand in for matching
rules.

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
