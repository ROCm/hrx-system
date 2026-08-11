# Reference

The reference is assembled from the same contracts used to build Loom:

- The dialect reference is generated from the canonical Python declaration
  model that drives parsing, printing, verification, builders, and bytecode.
- The embedding reference is generated from the public `loomc` headers with
  strict documentation diagnostics.
- Tool pages describe only supported public command-line interfaces. Internal
  repository helpers remain contributor infrastructure rather than becoming
  accidental product APIs.

Generated output is rebuilt for every documentation publication. Nothing in
the published reference is a manually maintained inventory of operations or
headers.

- [Dialect reference](dialects/index.md)
- [Type reference](types/index.md)
- [Attribute reference](attributes/index.md)
- [Encoding reference](encodings/index.md)
- [C API reference](c-api/index.md)
