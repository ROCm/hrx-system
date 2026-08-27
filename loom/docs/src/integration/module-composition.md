# Compose modules in memory

**Example source:** [`loom/binding/c/example/link_modules.c`](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/link_modules.c)

The public `loomc` API links source bytes already owned by an application. It
does not require source paths, a filesystem resolver, or process-global state.
Applications assemble an explicit source universe, freeze its metadata in a
reusable link index, and select roots from that universe for each invocation.

## Keep labels separate from linkage

An in-memory source carries two labels:

| Label | Example | Meaning |
| --- | --- | --- |
| Source identifier | `library.loom` | Diagnostic and source-provenance label passed to `loomc_source_create`. |
| Provider name | `library` | Stable source label used by the frozen index for diagnostics and private-name determinism. |

Neither label participates in symbol resolution or opens a file. Exact linkage
uses authored declarations and exported definitions. Template selection uses
template family contracts, facts, requirements, and priority. The application
chooses the available sources explicitly instead of encoding build-system or
filesystem identity into Loom IR.

The application creates each `loomc_source_t` from borrowed, copied, or
externally owned text or bytecode. The example adds its library source to the
index as follows:

```c
--8<-- "generated/examples/integration/module-composition/link_modules.c:library-source"
```

An input source is part of the module currently being assembled. A library
source contributes exported exact definitions and template implementations,
but its public symbols do not become output roots merely because the library
was supplied.

## Link an explicit source universe

The example creates a harness containing `func.decl @identity` and a library
containing the compatible public definition. It freezes both sources in one
index and asks `loomc_link_module` for the `@caller` root.

The declaration is the semantic contract. The linker resolves it to the unique
eligible definition in the supplied universe, retains the dependency closure,
and leaves unrelated library definitions out of the product. When no input
definition owns the name, ambiguous exported library definitions are an error;
input order is not an overload rule.

```shell
python dev.py bazel run //loom/binding/c/example:link_modules
```

The checked output is a closed program:

```loom title="linked.loom"
--8<-- "generated/examples/integration/module-composition/linked.loom"
```

The documentation build runs the public executable, checks its output with
`loom-format`, and stages both the example source and canonical linked module.

## Reuse the index without retaining host state in products

`loomc_link_index_builder_finish` produces an immutable index that can be
shared by independent workers. Each active link uses its own
`loomc_workspace_t` and materializes only the selected closure into an
invocation-owned module.

When a link permits unresolved symbols, the output retains reachable
declarations in ordinary `.loom` or `.loombc`. That relocatable artifact can be
serialized, reloaded in another process, and linked against another explicitly
supplied library without access to the original index or application state. The
[linking and packaging
workflow](../workflows/link-and-package.md#link-transitive-dependencies-incrementally)
demonstrates that lifecycle across two independent invocations.

Compile-time exact declarations such as `func.decl`, `kernel.decl`, and
`command.program.decl` use this symbol-linkage model. Runtime function imports
remain external ABI contracts rather than library requests. Template families
compose from the supplied library universe and match through `template.decl`,
facts, requirements, and priority. The [source-module
chapter](../guide/source-modules.md#definitions-declarations-and-roots) defines
those contracts, and the generated [`loomc` C API
reference](../reference/c-api/index.md) defines the ownership and failure
semantics of every handle used here.
