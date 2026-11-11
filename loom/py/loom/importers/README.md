# Loom Importer Construction

`loom.importers` is the dependency-free Python construction surface for tools
that translate another language or compiler IR into Loom. It ships with the
base Loom SDK and provides kernel module shells, source-name allocation,
diagnostics, verification, printing, and bytecode serialization on top of the
canonical `LoomBuilder` API.

Foreign frontends live in their own repositories and version against the Loom
SDK. This keeps their parser and compiler dependencies, release cadence, and
compatibility matrix with the integration that owns them. An AOT deployment
that consumes `.loombc` therefore carries neither those frontend dependencies
nor an unused parser.

The native C/C++ importer is the exception: it is a core Loom authoring surface
and remains in-tree, but is independently selectable so AOT-only tools and
deployed runtimes can omit it.

## Construction contract

The public `loom.importers` package re-exports the source-independent pieces an
external frontend needs:

- `KernelModuleSpec` and `create_kernel_module` establish the target, export,
  launch configuration, and kernel body boundary.
- `SourceImportSession` carries source names, mapped values, conversion records,
  and diagnostics while a frontend translates its own nodes.
- `kernel_module_ops` and `print_loom_module` connect the selected target's
  declarations to Loom's canonical verification and serialization APIs.

The [Python construction guide](https://rocm.github.io/hrx-system/loom/integration/python-importer/)
contains a checked executable example with text and bytecode roundtrips. An
external importer owns focused fixtures for its source language and qualifies
the emitted Loom module through the same public verification path.
