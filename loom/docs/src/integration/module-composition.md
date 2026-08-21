# Compose modules in memory

**Example source:** [`loom/binding/c/example/link_modules.c`](https://github.com/ROCm/hrx-system/blob/main/loom/binding/c/example/link_modules.c)

The public `loomc` API links source bytes already owned by an application. It
does not require source paths, a filesystem resolver, or `module.import`.
Applications assemble an explicit source universe, freeze its metadata in a
reusable link index, and select roots from that universe for each invocation.

`module.import` adds one optional constraint: a declaration named by the import
can resolve only from a source bound to one of its opaque keys. This is useful
when a large application chooses provider packages by logical identity, but it
does not change the definition's symbol or ABI.

## Keep the three identities separate

An in-memory source can carry three independent names:

| Identity | Example | Meaning |
| --- | --- | --- |
| Source identifier | `library.loom` | Diagnostic and source-provenance label passed to `loomc_source_create`. |
| Provider name | `library` | Stable provider label used by the frozen link index for diagnostics and private-name determinism. |
| Import key | `identity-library` | Opaque availability key that authored `module.import` operations may reference. |

None of these names opens a file. The application creates every
`loomc_source_t` from borrowed, copied, or externally owned text or bytecode
and explicitly adds it to a `loomc_link_index_builder_t`.

The checked command-line composition example uses path-shaped keys because
`loom-link` binds each supplied input spelling as a key:

```loom title="root.loom"
--8<-- "examples/module-composition/root.loom"
```

An embedding is free to use stable package identities instead. The C example's
imported harness names `"identity-library"`, while the provider source retains
the unrelated identifier `"library.loom"` and provider name `"library"`.

## Link an explicit universe without imports

The example's default mode creates a harness containing `func.decl @identity`
and a library containing the compatible definition. It adds both sources to
one builder without any import keys, freezes the index, and asks
`loomc_link_module` for the `@caller` root.

The declaration is the semantic contract. Because the application has already
chosen the complete source universe, the linker can satisfy it from a
compatible definition in that universe. This is the compact path for generated
programs, tests, and embeddings whose surrounding API already determines the
available libraries.

```shell
python dev.py bazel run //loom/binding/c/example:link_modules
```

## Constrain one declaration to a provider key

Passing `--provider-import` selects a harness with
`module.import "identity-library" [@identity]`. The application binds the
same logical key to the library source when adding its slot:

```c
--8<-- "generated/examples/integration/module-composition/link_modules.c:provider-key"
```

The builder copies the key array and its strings. After
`loomc_link_index_builder_finish`, the immutable index can be shared by
independent workers while each active link uses its own `loomc_workspace_t`.
The import affects candidate availability only; normal signature compatibility,
reachability, and specialization rules still decide what enters the result.

```shell
python dev.py bazel run //loom/binding/c/example:link_modules -- \
  --provider-import
```

Both modes produce the same closed program:

```loom title="linked.loom"
--8<-- "generated/examples/integration/module-composition/linked.loom"
```

The documentation build runs both invocations, checks each output with
`loom-format`, and requires the resulting modules to be byte-for-byte equal.
Separate Bazel tests run the two public executable modes directly.

## Preserve the link boundary

The frozen index owns source metadata and can be reused across many root and
configuration selections. A link operation materializes only the selected
closure into an invocation-owned module. Supplying a library therefore makes
its definitions available; it does not make the entire library live.

A satisfied import disappears from the linked module. An unresolved import may
remain in a partial `.loombc` artifact when the link options permit unresolved
demands, and that artifact can be reloaded without any index or host state from
the creating process. The [linking and packaging
workflow](../workflows/link-and-package.md#link-transitive-dependencies-incrementally)
follows that partial lifecycle across two independent invocations.

Imports constrain exact declarations such as `func.decl`, `kernel.decl`, and
`command.program.decl`. Template families compose from the explicitly supplied
library universe and match through `template.decl`, facts, requirements, and
priority instead of source keys. The [source-module
chapter](../guide/source-modules.md#declarations-state-contracts-imports-state-availability)
defines both contracts, and the generated [`loomc` C API
reference](../reference/c-api/index.md) defines the ownership and failure
semantics of every handle used here.
