# Loom Dialect Authoring

This directory is the source of truth for Python declarations of Loom dialects,
ops, attributes, types, and assembly formats. The declarations feed the Python
builder surface, text parser/printer, bytecode tests, C op tables, generated C
builders, editor metadata, and migration tooling.

The common editing path starts in a dialect `defs.py` file:

- `Dialect(...)` names the dialect, assigns its bytecode dialect id, and carries
  broad phase/category defaults.
- `Op(...)` declares the operation surface: fields, traits, constraints,
  effects, verifier hooks, fact hooks, textual format, and examples.
- `format=[...]` is the canonical textual spelling. The same sequence drives
  parser/printer behavior and public builder parameter order.
- `examples=[...]` should show the spelling a human or agent should copy.

The synthetic test dialect in
[loom/py/loom/dialect/test/defs.py](/loom/py/loom/dialect/test/defs.py)
exercises the DSL surface. Existing production dialects are usually better
examples for production semantics; the test dialect is best when checking how a
particular format element, trait, or constraint is spelled.

## Op Declarations

A minimal op has fields, a result, and a format:

```python
from loom.assembly import COLON, COMMA, Ref, TypeOf
from loom.dsl import INTEGER, PURE, Op, Operand, Result, SameType

integer_add = Op(
    name="example.addi",
    group=example_ops,
    doc="Add two integer scalar values.",
    operands=[
        Operand("lhs", INTEGER, doc="Left operand."),
        Operand("rhs", INTEGER, doc="Right operand."),
    ],
    results=[Result("result", INTEGER, doc="Integer sum.")],
    constraints=[SameType("lhs", "rhs", "result")],
    traits=[PURE],
    format=[Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")],
    examples=["%sum = example.addi %lhs, %rhs : i32"],
)
```

Field names are the stable bridge between the op declaration and the format
elements. If the format references `Ref("lhs")`, the op needs an operand,
result, attribute, successor, region, or implicit format field named `lhs`.

The `format` order is part of the API exposed to generated Python and C builder
surfaces. Reordering format fields changes the way generated builders ask for
arguments even when the underlying IR fields are unchanged.

### Public Field Semantics

Every declared operation field is public Loom IR, including optional,
backend-specific, generated, or lowering-only attributes, operands, results,
types, regions, and enum cases. Each field has a stable meaning that can be
authored, documented, verified, round-tripped through text and bytecode,
cloned, and transformed without compiler-private producer context.

This semantic contract is distinct from a typed attribute's private physical
layout. A `scoped_enum` field carries the stable key encoded in text and
bytecode even though canonical C IR stores a dense ordinal interpreted by the
function's representation contract. String-, symbol-, and type-table indexes
follow the same rule: changing their private layout cannot change the authored
field value or meaning.

Caches, analysis results, resolved facts, capability masks, provider records,
pointer-like handles, fingerprints, and pass-local state remain in
compiler-owned objects. Private indexes and dense ordinals are valid physical
payloads only behind a typed public attribute whose format codecs preserve the
stable semantic value. An opaque integer or string does not create public
semantics; `contract_feature_bits` is invalid because generated provider bit
positions become the authored meaning and a table-layout change reinterprets
the IR.

Target information represented in IR follows the same rule without presuming
that target records, authored feature sets, or compatibility witnesses should
exist. Each construct still needs a real caller and semantic owner, with an
explicit relation to environment facts, lowering choices, and artifact
requirements; invocation-owned profiles or program-derived facts may be the
correct owner instead. Corpus fixtures reach target-dependent behavior through
the same public IR and compiler APIs as shipping callers rather than injecting
internal masks, IDs, cached facts, or provider records.

Every proposed field satisfies all of these review conditions:

- Its meaning does not rely on a compiler-private data structure or table.
- A user can select or understand it from program or compilation semantics.
- Parsing, bytecode round-trip, cloning, canonicalization, and independent
  transformation preserve it using only declared public context.
- Verification depends only on public IR and declared public contracts.
- Internal representation changes leave the authored value and meaning
  unchanged.

A field that fails any condition belongs in analysis, plan, target facts, or
pipeline-owned state outside IR.

## Descriptor-Backed Attributes and Types

`ParameterizedAttrDef` declares a closed, namespaced attribute family with an
ordered parameter schema. Use it when the family name and parameter meanings
are public IR semantics and generic dictionary keys would lose the family
identity or permit unsupported fields.

```python
from loom.dsl import AttrDef, ParameterizedAttrDef

tile_attr = ParameterizedAttrDef(
    "example.tile",
    group=example_ops,
    parameters=[
        AttrDef("width", "i64", doc="Tile width in elements."),
        AttrDef("transpose", "bool", optional=True),
    ],
    primary_parameter="width",
    doc="A statically selected tile layout.",
)

tile = tile_attr(width=16, transpose=True)
```

The descriptor-declared primary parameter prints positionally while additional
parameters remain named:

```loom
#example.tile<16, transpose = true>
```

Families without `primary_parameter` name every present parameter, even when
the schema currently has only one field. Compact syntax is therefore a stable
family contract rather than an arity heuristic; adding an optional named field
does not change the primary spelling.

Parameters are stored in declaration order and exposed through generated C
accessors without runtime name lookup. Bytecode carries the stable family and
parameter names rather than dense positions. Text carries the family name and
every non-primary parameter name; the descriptor makes the primary position a
family-level syntax contract. Renaming any parameter changes the bytecode and
generated API contract, while the canonical positional spelling survives a
primary-parameter rename. Reordering parameters changes generated storage and
builder order, but the primary still prints first and bytecode still resolves
each present value by name.

Optional absence is distinct from a present false scalar, empty array, byte
span, or dictionary. Python values expose that distinction with `has()` and
`get()`; generated C builders use `HAS_*` build flags. This lets a dialect
assign different semantics to omission and an explicit value without sentinel
payloads:

```python
absent = tile_attr(width=16)
present_false = tile_attr(width=16, transpose=False)

assert not absent.has("transpose")
assert present_false.has("transpose")
```

A parameter whose kind is `ATTR_TYPE_PARAMETERIZED` declares its exact nested
family. The parser, builders, and bytecode reader reject values from another
family instead of accepting any attribute with a similar physical shape:

```python
from loom.dsl import ATTR_TYPE_PARAMETERIZED

AttrDef(
    "tile",
    ATTR_TYPE_PARAMETERIZED,
    optional=True,
    parameterized_attr=tile_attr,
)
```

Symbol parameters retain the `SymbolReference` contract declared by their
`AttrDef`. References nested in parameterized attributes and types participate
in ordinary symbol verification, remapping, and compaction, and text and
bytecode preserve their stable symbol names.

A `TypeDef` whose parameters are all `AttrDef` values declares a
descriptor-backed type family with the same immutable named-slot contract.
Its assembly format controls only how each named parameter is spelled. `Param`
may be positional or surrounded by keywords and punctuation, and every
parameter appears exactly once:

```python
from loom.assembly import COMMA, EQUALS, OptionalGroup, Param, kw
from loom.dsl import AttrDef, TypeDef

array_type = TypeDef(
    "example.array",
    params=[
        AttrDef("element_type", "type"),
        AttrDef("alignment", "i64", optional=True),
    ],
    format=[
        Param("element_type"),
        OptionalGroup(
            [COMMA, kw("alignment"), EQUALS, Param("alignment")],
            anchor="alignment",
        ),
    ],
    doc="An element type with optional storage alignment.",
)
```

That declaration accepts both forms while preserving the same parameter
identity in memory and bytecode:

```loom
example.array<bf16>
example.array<bf16, alignment = 16>
```

Descriptor-backed and representation-specific type parameters are deliberately
separate models. A descriptor-backed `TypeDef` cannot mix `AttrDef` parameters
with `ShapeParam`, `TypeParam`, or other physical type representation fields.
This keeps the generic family self-describing and lets every parser, printer,
builder, remapper, and serializer consume the same schema.

Both family forms currently allow at most 255 total parameters and 64 optional
parameters. Generation rejects larger schemas instead of widening every IR
value or generated builder for a pathological family.

## Assembly Formats

Assembly format elements live in
[loom/py/loom/assembly.py](/loom/py/loom/assembly.py). They are small
declarative objects rather than callbacks. The parser, printer, builders, and C
generators all expect the complete textual grammar to be visible from the format
list.

Common elements:

- `Ref("field")`, `Refs("field")`, and `TypedRefs("field")` reference SSA
  operands or use-side values.
- `Attr("field")` prints one attribute value.
- `AttrDict()` prints uncovered declared attributes in a dictionary.
- `Param("field")` prints one typed parameter of a descriptor-backed type.
- `TypeOf("field")`, `TypesOf("field")`, `ResultType("field")`, and
  `ResultTypeList("field")` print type information.
- `TemplateParam("field")` prints an op template parameter such as
  `buffer.assume.memory_space<global>`.
- `AttrParams("field")` prints only the parameter payload of an exact-family
  parameterized attribute, such as `encoding.matches<element_format = u4>`.
- `Region("field")`, `BlockRef("field")`, and related CFG/region elements
  describe structured control flow.

The parser/printer path has no custom per-op escape hatch. When an op needs a
new shape, extend the declarative format vocabulary or reshape the op
declaration so the existing vocabulary can express it.

## Source Format Changes

Changing the canonical `format` of an existing op changes checked-in `.loom`
and `.loom-test` source. Stable source-format changes need a legacy format
adapter so users and agents can migrate source mechanically.

The normal breadcrumb in an op declaration is a `legacy_formats=[...]` entry:

```python
from loom.assembly import (
    COLON,
    EQUALS,
    LBRACE,
    RBRACE,
    Attr,
    Ref,
    TemplateParam,
    TypeOf,
    kw,
)
from loom.dsl import LegacyFormat

buffer_assume_memory_space = Op(
    name="buffer.assume.memory_space",
    format=[TemplateParam("memory_space"), Ref("buffer"), COLON, TypeOf("result")],
    legacy_formats=[
        LegacyFormat(
            "buffer.assume.memory_space.attr_dict",
            format=[
                Ref("buffer"),
                LBRACE,
                kw("memory_space"),
                EQUALS,
                Attr("memory_space"),
                RBRACE,
                COLON,
                TypeOf("result"),
            ],
            replaced_by="loom-source-format-2026-06-09",
        )
    ],
)
```

That is enough for syntax-only migrations where fields from the old spelling map
directly into the current spelling. Semantic migrations and currently
unsupported format elements can name a rewrite hook, but the migration policy
and hook mechanics live with the migration tool, not in the dialect authoring
flow. See
[loom/py/loom/migration/README.md](/loom/py/loom/migration/README.md).

## In-Tree Developer Checks

When editing op declarations inside the Loom source tree, the tight validation
loop is:

```bash
python dev.py bazel test //loom/py/loom:core_test
python dev.py bazel test //loom/py/loom/dialect/<dialect>:<dialect>_ops_test
python dev.py bazel test //loom/src/loom/ops/<dialect>/...
```

When generated C tables or builders change, run the relevant generated C tests
or the owning dialect tests under `loom/src/loom/ops/<dialect>/...`.

When checked `.loom-test` expectations need intentional updates, route them
through `loom-check --update`; expectation blocks are generated test output, not
hand-edited migration input. The tool may come from an installed Loom release,
a CMake build, or an in-tree Bazel build. The `loom-check` workflow is
documented in
[loom/src/loom/tools/loom-check/README.md](/loom/src/loom/tools/loom-check/README.md).

## Pointers

- [loom/py/loom/dsl.py](/loom/py/loom/dsl.py): op/type/attribute DSL data model
  and validation.
- [loom/py/loom/assembly.py](/loom/py/loom/assembly.py): declarative format
  element vocabulary.
- [loom/py/loom/migration/README.md](/loom/py/loom/migration/README.md): source
  migration policy, baselines, legacy format rules, hook support, and rule
  tests.
- [loom/py/loom/dialect/test/defs.py](/loom/py/loom/dialect/test/defs.py):
  synthetic DSL coverage dialect.
- [loom/src/loom/test/corpus/authoring/README.md](/loom/src/loom/test/corpus/authoring/README.md):
  hand-authored `.loom` program examples.
