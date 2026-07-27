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
- `TypeOf("field")`, `TypesOf("field")`, `ResultType("field")`, and
  `ResultTypeList("field")` print type information.
- `TemplateParam("field")` prints an op template parameter such as
  `buffer.assume.memory_space<global>`.
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
