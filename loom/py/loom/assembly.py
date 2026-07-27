# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Assembly format elements for loom op declarations.

Format elements describe how an op's textual assembly is structured.
They are the single source of truth for:

  - The text printer (walks elements left-to-right, emits tokens).
  - The text parser (walks elements left-to-right, consumes tokens).
  - Builder parameter ordering (format order = parameter order).
  - C code generation (same format elements drive C printer/parser).

An op's format spec is a list of FormatElement instances. The printer
and parser walk the list in order. Every token in the textual
representation is accounted for by exactly one element.

Example format specs:

  # scalar.addi: %r = scalar.addi %a, %b : f32
  [Ref("lhs"), COMMA, Ref("rhs"), COLON, TypeOf("result")]

  # scalar.sitofp: %r = scalar.sitofp %x : i32 to f32
  [Ref("input"), COLON, TypeOf("input"), kw("to"), TypeOf("result")]

  # func.call: %r = func.call @f(%a, %b) : (t0, t1) -> (t0, t1)
  [SymbolRef("callee"),
   LPAREN, Refs("operands"), RPAREN,
   COLON, LPAREN, TypesOf("operands"), RPAREN,
   ARROW, ResultTypeList("results")]

  # cfg.br: cfg.br ^join(%a: i32, %b: i32)
  [BlockRef("dest"), GLUE, LPAREN, TypedRefs("args"), RPAREN]

  # tile.elementwise: %r = tile.elementwise(%e = %x : type) { ... } -> (type)
  [BindingList("inputs"), Region("body"), ARROW, ResultTypeList("result")]

Design rules:
  - No "Custom" escape hatch. Every op is fully declarative.
  - Variadic-ness is explicit: Ref (singular), Refs (variadic values), and
    TypedRefs (variadic value/type pairs).
  - Result types always use parens: -> (type) not -> type.
  - Tied results are sparse (TiedResult list, no sentinels).
  - Format element order = builder parameter order.

Dialect files import the specific format elements they need.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    # Element types.
    "Ref",
    "Refs",
    "TypedRefs",
    "BlockRef",
    "Attr",
    "SymbolRef",
    "TypeOf",
    "TypesOf",
    "ResultType",
    "ResultTypeList",
    "Keyword",
    "Clause",
    "AttrDict",
    "AttrTable",
    "RegionTable",
    "Region",
    "IndexList",
    "OperandDict",
    "BindingList",
    "BlockArgs",
    "FuncArgs",
    "PredicateList",
    "OptionalGroup",
    "Scope",
    "Glue",
    "GLUE",
    # Angle-bracket elements.
    "Flags",
    "OpRef",
    "DescriptorRef",
    "StableKeyRef",
    "TemplateParam",
    "TemplateParamFlags",
    # Type-interior format elements.
    "ShapeOf",
    "ScalarOf",
    "EncodingOf",
    # Union type.
    "FormatElement",
    # Convenience constructors.
    "kw",
    # Binding kinds.
    "BINDING_CAPTURE",
    "BINDING_ELEMENT",
    # Common keywords.
    "COMMA",
    "COLON",
    "ARROW",
    "LPAREN",
    "RPAREN",
    "LBRACKET",
    "RBRACKET",
    "LBRACE",
    "RBRACE",
    "EQUALS",
    "TO",
    "FOR",
]


# ============================================================================
# Value reference elements
# ============================================================================


@dataclass(frozen=True, slots=True)
class Ref:
    """A single SSA value reference.

    Prints/parses: %name

    The field names an operand, result, or special value (like an
    induction variable) on the op declaration. The printer emits the
    value's SSA name. The parser consumes an SSA value token and
    resolves it in the current scope.

    For builders: maps to a single Value parameter.
    """

    field: str


@dataclass(frozen=True, slots=True)
class Refs:
    """Variadic SSA value references, comma-separated.

    Prints/parses: %a, %b, %c

    The field names a variadic operand on the op declaration. The
    printer emits each value's SSA name separated by commas. The
    parser consumes SSA value tokens until a non-value token.

    For builders: maps to a list[Value] parameter.
    """

    field: str


@dataclass(frozen=True, slots=True)
class TypedRefs:
    """Variadic SSA value references with adjacent type annotations.

    Prints/parses: %a: type, %b: type

    The field names a variadic operand on the op declaration. Unlike a separate
    Refs + TypesOf pair, each SSA value carries its type next to its name. This
    is the canonical shape for block-edge payloads where the payload list reads
    like a use-side argument list.

    For builders: maps to a list[Value] parameter.
    """

    field: str


@dataclass(frozen=True, slots=True)
class BlockRef:
    """A single CFG successor block reference.

    Prints/parses: ^label

    The field names a successor on the op declaration. Labels are only the
    textual spelling: after parsing, operations store direct Block references
    so CFG analyses and rewrites do not depend on display names.

    For builders: maps to a single Block parameter.
    """

    field: str


@dataclass(frozen=True, slots=True)
class Attr:
    """An attribute value (compile-time constant).

    Prints/parses: 42, 3.14, "hello", slt, true, false

    The field names an attribute on the op declaration. The printer
    emits the attribute value in its canonical form. The parser
    consumes the appropriate literal token.

    For builders: maps to a Python literal parameter (int, float,
    str, bool, or enum member).
    """

    field: str


@dataclass(frozen=True, slots=True)
class SymbolRef:
    """A symbol reference.

    Prints/parses: @name

    The field names a symbol reference attribute on the op declaration.

    For builders: maps to a str or Symbol parameter.
    """

    field: str


# ============================================================================
# Type elements
# ============================================================================


@dataclass(frozen=True, slots=True)
class TypeOf:
    """The type of a single field.

    Prints/parses: f32, tile<4x4xf32>, tensor<[%M]xf32>, etc.

    The field names an operand or result whose type should be printed.
    For operands, the type is known from the Value. For results, the
    type is part of the operation's result declaration.

    For builders: not a parameter (type is carried by the Value or
    inferred from context).
    """

    field: str


@dataclass(frozen=True, slots=True)
class TypesOf:
    """Types of a variadic field, comma-separated.

    Prints/parses: f32, tensor<[%M]xf32>, i32

    The variadic counterpart of TypeOf. Prints one type per value
    in the variadic field, separated by commas.

    For builders: not a parameter.
    """

    field: str


@dataclass(frozen=True, slots=True)
class ResultTypeList:
    """Result type list with tied-result handling.

    Prints/parses: (type) or (%operand as type, type)

    By default uses parentheses (even for single results). Pass
    parens=False for bare comma-separated types without parentheses,
    useful for ops like global.load where the result type annotation
    is a single bare type: ``global.load @name : type``.

    Each result is either a plain type (fresh allocation) or a tied
    reference (%operand_name as type, consuming the operand).

    The element reads the Operation's tied_results list (sparse,
    no sentinels) to determine which results are tied. The printer
    builds a {result_index: TiedResult} lookup, then emits each
    result in order.

    This element does NOT include the '->' arrow — that's a separate
    Keyword element in the format spec, so the arrow is visible and
    the separator choice is explicit.

    For builders: maps to a result_types parameter and an optional
    tied parameter for specifying ties.
    """

    field: str
    parens: bool = True


@dataclass(frozen=True, slots=True)
class ResultType:
    """A single result type without parentheses.

    Prints/parses: type (bare, no parens)

    For ops with exactly one non-variadic result where parenthesized
    list syntax would be misleading. Does not support tied results —
    use ResultTypeList for ops that need tied-result syntax.

    Like ResultTypeList, this element does NOT include the '->' arrow.

    For builders: maps to a single result_type parameter.
    """

    field: str


# ============================================================================
# Structural elements
# ============================================================================


@dataclass(frozen=True, slots=True)
class Keyword:
    """A literal token in the assembly.

    Prints/parses the exact text. Used for punctuation (,  :  ->)
    and keyword tokens (to, step, else, iter_args, etc.).

    For builders: not a parameter (structural syntax).
    """

    text: str

    def __repr__(self) -> str:
        return f"kw({self.text!r})"


@dataclass(frozen=True, slots=True)
class Clause:
    """A named parenthesized clause.

    Prints/parses: name(payload...)

    Clauses are structural syntax for fixed semantic slots such as
    ``source(%src)``, ``target(%dst)``, ``range(0 to 16)``, or
    ``path("input.npy")``. The clause name is syntax, not data; the
    payload is still described by ordinary format elements and backed by
    ordinary op fields.

    For builders: no parameter for the clause wrapper itself; builder
    parameters come from the payload elements.
    """

    name: str
    elements: tuple[FormatElement, ...]

    def __init__(self, name: str, *elements: FormatElement) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "elements", tuple(elements))


@dataclass(frozen=True, slots=True)
class AttrDict:
    """An attribute dictionary.

    Prints/parses: {key = value, key = value, ...}

    When |field| is set, the dictionary is backed by that attr of type
    "dict". When |field| is empty, the dictionary contains uncovered
    declared attributes from the op itself. Empty dictionaries print nothing.

    For builders: maps to either an optional dict parameter or the declared
    attrs covered by the dictionary.
    """

    field: str = ""


@dataclass(frozen=True, slots=True)
class AttrTable:
    """A static-attribute-keyed SSA value table.

    Prints/parses: {0 = (%a, %b), 1 = (%c, %d)} default(%x, %y)

    |keys| names an attribute field containing the static row keys. |values|
    names a variadic operand field containing row payloads flattened in
    row-major order, with the final row reserved for the default payload.
    """

    keys: str
    values: str


@dataclass(frozen=True, slots=True)
class RegionTable:
    """A static-attribute-keyed region table with a default branch.

    Prints/parses::

      {case 0 { ... } case 1 { ... } default { ... }}

    |keys| names an i64-array attribute field containing sorted case keys.
    |case_regions| names a variadic region field containing one region per key.
    |default_region| names the fixed fallback region field. The default region
    is stored separately from the variadic case tail even though it prints last,
    so generated accessors can expose both concepts directly.
    """

    keys: str
    case_regions: str
    default_region: str


@dataclass(frozen=True, slots=True)
class OperandDict:
    """A keyed SSA operand dictionary.

    Prints/parses: {key = %value : type, key = %value : type, ...}

    |operands| names a variadic operand field. |names| names an optional dict
    attribute whose keys are the operand dictionary keys and whose values are
    operand ordinals relative to the variadic field start. The dictionary never
    stores SSA value IDs; the SSA values are ordinary op operands.
    """

    operands: str
    names: str


@dataclass(frozen=True, slots=True)
class Region:
    """A nested region containing blocks of operations.

    Prints/parses: { block+ }

    The field names a region on the op declaration. Regions contain
    one or more blocks, each with optional block arguments and a
    sequence of operations.

    The optional syntax name selects an alternate parser/printer for the
    region's surface spelling while preserving ordinary region storage. An
    empty syntax name uses the canonical braced region form.

    For builders: maps to a region-building callback or context
    manager parameter.
    """

    field: str
    syntax: str = ""


# ============================================================================
# Composite elements
# ============================================================================


@dataclass(frozen=True, slots=True)
class IndexList:
    """A mixed static/dynamic index list in bracket notation.

    Prints/parses: [0, %x, 4] or [%i, %j] or [0, 0]. When used after
    another format element, `glue` controls whether the opening bracket
    attaches to the previous token.

    Static values come from the static attribute (an integer array).
    Dynamic values come from the dynamic operand list. The two are
    interleaved according to a sentinel convention: in the static
    array, a sentinel value (e.g., min_int) means "this position
    is dynamic, take the next value from the dynamic list."

    For builders: maps to a list parameter accepting int | Value
    entries (static or dynamic per element).
    """

    dynamic: str  # Operand field for dynamic values.
    static: str  # Attribute field for static values.
    glue: bool = True  # Suppress space before the list when non-leading.


@dataclass(frozen=True, slots=True)
class BindingList:
    """Named value bindings with types.

    Prints/parses: (%a = %x : type, %b = %y : type). The text parser also
    accepts the grouped type-list spelling: (%a = %x, %b = %y : type0, type1).

    Each binding maps a new name (block argument in the region body)
    to an existing value (operand) with its type. The `kind` parameter
    determines how the block arg type relates to the operand type:

      kind="capture" — block arg has the SAME type as the operand.
        Used by scf.for iter_args, dispatch.region. The block arg
        is a re-binding of the operand for the region body.

      kind="element" — block arg has the ELEMENT TYPE of the operand.
        Used by tile.elementwise. The block arg receives one scalar
        element from the shaped operand.

    For custom types, the type extraction is driven by the type's
    TypeDef — custom types can define how "element" extraction works.

    For builders: maps to a list of (name, value) pairs.
    """

    field: str
    kind: str = "capture"  # "capture" or "element"


@dataclass(frozen=True, slots=True)
class BlockArgs:
    """Region entry block argument definitions.

    Prints/parses: (%a: type, %b: type)

    Unlike BindingList, BlockArgs does not bind each argument to an existing
    operand in the surface syntax. It only names and types the entry block
    arguments of the referenced region. The op verifier owns the semantic
    relationship between those block arguments and any operands, terminator
    operands, or result fields.
    """

    region: str


@dataclass(frozen=True, slots=True)
class FuncArgs:
    """Function argument definitions with types.

    Prints/parses: (%a: type, %b: type)

    Unlike BindingList, FuncArgs defines new SSA values (the function
    parameters). There is no '= %existing_value' part — the names
    and types are the definitions themselves.

    For builders: maps to a list of (name, type) pairs.
    """

    field: str


@dataclass(frozen=True, slots=True)
class PredicateList:
    """Where-clause predicate list.

    Prints/parses: [mul(%M, 16), lt(%K, 1024), range(%N, 32, 512)]

    Predicates constrain dynamic dimension values. Each predicate is
    a named function applied to SSA values and/or integer constants.
    Used in function where clauses and assume ops.

    For builders: maps to a list of Predicate instances.
    """

    field: str


# ============================================================================
# Modifiers
# ============================================================================


@dataclass(frozen=True, slots=True)
class OptionalGroup:
    """A conditional group of format elements.

    The elements are printed/parsed only when the anchor field is
    present (non-empty, non-None, non-zero-length). Used for optional
    parts of an op's syntax:

      - else region on scf.if
      - iter_args on scf.for
      - where clause on func.def/func.decl
      - visibility/calling convention modifiers
      - body region (present for definitions, absent for declarations)

    For builders: the anchor field maps to an optional parameter
    (with a default of None or empty).
    """

    elements: tuple[FormatElement, ...]
    anchor: str

    def __init__(
        self, elements: list[FormatElement] | tuple[FormatElement, ...], anchor: str
    ) -> None:
        # Accept list for ergonomics, store as tuple for immutability.
        object.__setattr__(self, "elements", tuple(elements))
        object.__setattr__(self, "anchor", anchor)


@dataclass(frozen=True, slots=True)
class Scope:
    """A scoped group of format elements.

    Pushes one temporary declaration name scope before processing children and
    pops it after. Within the scope, type parsing uses definition mode:
    [%name] in types creates new index-typed SSA values (like
    function argument dims) rather than requiring existing names.

    Used for function/global signatures where type annotations introduce named
    type variables that later result types and predicates can reference:

      global.constant @weights : Scope([ tile<[%m]xf32> where [...] ])

    The scope ensures dim names are local to each definition and
    that created values get proper value table entries (enabling
    value facts attachment).

    Nested Scope(...) is intentionally unsupported; region/block lexical
    nesting is represented by Region and BindingList instead.
    """

    elements: tuple[FormatElement, ...]

    def __init__(
        self,
        elements: list[FormatElement] | tuple[FormatElement, ...],
    ) -> None:
        object.__setattr__(self, "elements", tuple(elements))


@dataclass(frozen=True, slots=True)
class Glue:
    """Suppress the space before the next token.

    Placed between two format elements to make them print without
    a space between them. Used when a bare Keyword like LPAREN
    should attach to the preceding token:

        SymbolRef("callee"), Glue, LPAREN, Refs("operands"), RPAREN

    Produces: @compute(%x, %y) — no space between @compute and (.

    Most composite elements have built-in glue behavior and don't
    need an explicit Glue:
      - IndexList: glues when following another format element (always %ref[...]),
        but not when it is the first element after the op name
      - BindingList: always glues (always keyword(...) or opname(...))
      - BlockArgs: always glues (always keyword(...))
      - FuncArgs: always glues (always @name(...))

    Elements that never glue (space before them is always present):
      - ResultTypeList: always follows -> with space
      - PredicateList: always follows 'where' with space

    For builders: Glue is invisible (not a parameter).
    """


# Singleton for use in format specs.
GLUE = Glue()


# ============================================================================
# Angle-bracket flags
# ============================================================================


@dataclass(frozen=True, slots=True)
class Flags:
    """Per-op-instance flags in angle brackets after the op name.

    Prints/parses: <flag1|flag2> (glued to the op name) or nothing
    when the attribute value is absent or zero.

    The field names a "flags"-typed attribute on the op declaration.
    The attribute's enum_def defines the valid flag keywords and their
    bitmask values. The attribute stores flags as a "|"-separated
    string in Python ("nuw|nsw") and as a uint8_t bitmask in C
    (stored in loom_op_t.instance_flags, not the attribute array).

    The '|' separator was chosen over ',' because flags are OR'd
    together, and '|' is visually distinct from comma-separated
    lists used everywhere else.

    Examples:
        scalar.addi<nuw> %a, %b : i32
        scalar.addf<reassoc|nnan> %a, %b : f32
        scalar.addf %a, %b : f32          (no flags = strict)
    """

    field: str


@dataclass(frozen=True, slots=True)
class OpRef:
    """Symbolic operation or implementation key in angle brackets.

    Prints/parses: <key.name> (glued to the op name) where the value
    is a dotted symbolic key like "tile.contract", "tile.reduce", or
    "qwen.q4.matmul".

    Used by func.template<T> and func.ukernel<T> to declare which
    implementation contract key they provide, and by func.apply<T> to
    demand a provider for that same contract key.

    The field names a string attribute storing the key.

    Examples:
        func.template<tile.contract> device @name(...)
        func.ukernel<tile.reduce> device @name(...)
        func.apply<qwen.q4.matmul>(%weights, %input) : (...) -> (...)
    """

    field: str


@dataclass(frozen=True, slots=True)
class DescriptorRef:
    """Descriptor key reference resolved to a dense row ordinal by targets.

    Prints/parses: <target.descriptor> glued to the op name.

    The ``key`` field names a string attribute used for text and diagnostics.
    The ``ordinal`` field names an i64 attribute storing the descriptor-set
    local row ordinal when the builder already has a selected descriptor set.
    Text parsing has no target context, so parsed descriptor refs use the
    unresolved ordinal sentinel and target binding resolves the key spelling at
    that boundary.
    """

    key: str
    ordinal: str


@dataclass(frozen=True, slots=True)
class StableKeyRef:
    """Symbolic key reference resolved to a stable numeric key identity.

    Prints/parses: <target.key> glued to the op name.

    Use this only for non-descriptor symbolic domains that are still keyed by
    stable string identity. Descriptor-backed low packets use DescriptorRef and
    resolve through the active descriptor set instead.
    """

    key: str
    stable_id: str


@dataclass(frozen=True, slots=True)
class TemplateParam:
    """Required compile-time op parameter in angle brackets after the op name.

    Prints/parses: <addf> or <some_enum_case>, glued to the op name.

    The field names an ordinary attribute. The attribute descriptor still owns
    parsing and validation, so enum parameters use the same case table and
    diagnostics as attrs in positional syntax.

    Examples:
        vector.reduce<addf> %v, %zero : vector<16xf32>, f32
    """

    field: str


@dataclass(frozen=True, slots=True)
class TemplateParamFlags:
    """Required op parameter plus optional instance flags in one angle group.

    Prints/parses: <addf> when no flags are present, or
    <addf, reassoc|nnan|nsz> when instance flags are set. The parameter field
    names an ordinary attribute parsed through its descriptor, while the flags field
    names the flags-typed attribute stored in op instance flags.

    Use this for ops whose primary specialization is a template-like enum and
    whose optional flags refine that specialization. Plain Flags remains the
    spelling for ops whose only angle-bracket payload is flags.

    Examples:
        vector.reduce<addf> %v, %zero : vector<16xf32>, f32
        vector.reduce<addf, reassoc|nnan|nsz> %v, %zero : vector<16xf32>, f32
    """

    param: str
    flags: str


# ============================================================================
# Type-interior format elements
# ============================================================================
#
# These elements are used inside TypeDef format specs to describe the
# interior of parameterized types (the content between < and >).
# They share the FormatElement union with op format elements.


@dataclass(frozen=True, slots=True)
class ShapeOf:
    """Shape dimensions in a shaped type interior.

    Prints/parses: 4x4 or [%M]x4 or [%M]x[%K] (dims separated by 'x').

    The field names a ShapeParam on the TypeDef. Each dim is either
    a static integer or a dynamic dim reference [%name].

    The 'x' separator between dims is implicit in this element —
    the format spec does NOT include separate Keyword("x") between
    dims. The 'x' between the last dim and the element type IS in
    the format spec as a Keyword.
    """

    field: str


@dataclass(frozen=True, slots=True)
class ScalarOf:
    """Scalar element type name in a shaped type interior.

    Prints/parses: f32, i8, bf16, index, etc.

    The field names a ScalarParam on the TypeDef.
    """

    field: str


@dataclass(frozen=True, slots=True)
class EncodingOf:
    """Encoding reference in a shaped type interior.

    Prints/parses: #q8_0 or #q8_0<block=32>

    The field names an EncodingParam on the TypeDef. The printer
    uses the module's encoding table to resolve the encoding name
    and parameters. The parser looks up or creates encoding instances.
    """

    field: str


# Binding kind constants for BindingList.
BINDING_CAPTURE = "capture"
BINDING_ELEMENT = "element"


# ============================================================================
# Union type
# ============================================================================

type FormatElement = (
    Ref
    | Refs
    | TypedRefs
    | BlockRef
    | Attr
    | SymbolRef
    | TypeOf
    | TypesOf
    | ResultType
    | ResultTypeList
    | Keyword
    | Clause
    | AttrDict
    | AttrTable
    | RegionTable
    | Region
    | IndexList
    | OperandDict
    | BindingList
    | BlockArgs
    | FuncArgs
    | PredicateList
    | OptionalGroup
    | Scope
    | Glue
    | Flags
    | OpRef
    | DescriptorRef
    | StableKeyRef
    | TemplateParam
    | TemplateParamFlags
    | ShapeOf
    | ScalarOf
    | EncodingOf
)


# ============================================================================
# Convenience constructors
# ============================================================================


def kw(text: str) -> Keyword:
    """Shorthand for Keyword(text)."""
    return Keyword(text)


# Common keywords — pre-built singletons for format spec readability.
COMMA = Keyword(",")
COLON = Keyword(":")
ARROW = Keyword("->")
LPAREN = Keyword("(")
RPAREN = Keyword(")")
LBRACKET = Keyword("[")
RBRACKET = Keyword("]")
LBRACE = Keyword("{")
RBRACE = Keyword("}")
EQUALS = Keyword("=")
TO = Keyword("to")
FOR = Keyword("for")
