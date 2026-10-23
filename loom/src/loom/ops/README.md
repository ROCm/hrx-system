# Native dialect construction

Each dialect package exposes three dependency surfaces:

| Target | Consumer | Contents |
| --- | --- | --- |
| `:op_defs` | Inspect existing operations | Generated declarations, accessors, and their type/enum vocabulary |
| `:builders` | Construct operations and attributes | Declarations plus generated constructors and the dialect's required handwritten/type constructors |
| The package's combined library | Implement or register compiler behavior | Construction plus verification, inference, canonicalization, and semantic tables |

The same surfaces are available for common and target-owned dialects. A native
caller depends on `//loom/src/loom/ir:builders` for module construction and on
the dialect builders it uses, for example `//loom/src/loom/ops/scalar:builders`.
Common parameterized types, including channels and views, are provided by
`//loom/src/loom/ops:type_builders`; dialect-owned parameterized types are
included in that dialect's `:builders` surface.

Construction uses the caller's module and its context-owned metadata. These
libraries do not register dialects, define another descriptor owner, or supply
verification and optimization pipelines. The caller obtains its configured
module from the compiler it uses. This is an exact-build native C interface,
not an extension of the `loomc` exported ABI.

Low construction includes `loom/codegen/low/builder.h` alongside the generated
structural constructors. Descriptor-backed Low operations require resolved
descriptors; their handwritten constructors preserve that contract. The combined
Low semantic library keeps its original generated constructors but does not
bring descriptor-aware construction into read-only emitter dependencies.

Each dialect still has one batched generator action. Its existing `builders.c`
output is compiled once into the construction library and reused by the combined
library. The generated `ops.h`/`ops.inc` declaration surface is shared rather
than regenerated for each consumer. Semantic callback declarations remain in
that surface; depending on it does not link their implementations. Enum headers
likewise expose vocabulary without requiring target record tables.

The declaration boundary is transitive. Generic operation declarations use
`//loom/src/loom/ops:headers`; shared atomic, cache, and combining vocabulary
uses `:vocabulary`. Beneath them, `//loom/src/loom/ir:type_headers` supplies
value/type vocabulary, while `:headers` supplies graph representation and
metadata declarations without IR storage or interning implementations. These
are dependency capabilities, not new C APIs or alternative representations.
The existing `:ir` and `:builders` aggregates keep their implementation roles.

An out-of-line call still requires its implementation dependency. In particular,
generated attribute setters invoke generic operation mutation, and rewrite
setters invoke the rewriter; including their declarations does not link those
algorithms. Keyword and assembly-format lookups belong to `:format`, which text
parsing and printing select explicitly rather than through construction.

`declaration_dependency_boundary` checks all configured dialect declarations
together, admitting only Loom representation, metadata, and vocabulary providers.
The underlying IREE base/arena runtime dependencies remain explicit.
`construction_dependency_boundary` checks the complete configured dialect
family together. It keeps compiler registries, semantic dialect libraries,
analysis, rewriting, and format implementations out of the construction closure.
Region builders instantiate dependent argument types through the IR construction
API, without depending on the general operation remapper.
