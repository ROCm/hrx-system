# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Func dialect op definitions.

Operations for runtime program structure and source provenance:

Top-level (module-level symbols):
  func.def       — Function definition (has body, callable by name).
  func.decl      — External function declaration (no body, callable by name).
Body ops:
  func.call      — Runtime function call.
  func.return    — Return values from function body.
  func.location  — Materialize captured source provenance as a buffer.
"""

from typing import Any

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LPAREN,
    RPAREN,
    Attr,
    AttrDict,
    FormatElement,
    FuncArgs,
    OptionalGroup,
    PredicateList,
    Refs,
    Region,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TypesOf,
    kw,
)
from loom.dialect.target.defs import ExportAbiKind
from loom.dsl import (
    ANY,
    BUFFER,
    ISOLATED_FROM_ABOVE,
    POISON_BOUNDARY,
    PURE,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    AttrDef,
    CallLikeInterface,
    CallLikeKind,
    Dialect,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    InlinePolicy,
    Op,
    Operand,
    OpPhase,
    ParameterizedAttrDef,
    RegionDef,
    Result,
    SymbolDefinition,
    SymbolDefinitionFlag,
    SymbolReference,
)

# ============================================================================
# Op group and shared enums
# ============================================================================

func_ops = Dialect("func", dialect_id=0x06, doc="Program structure operations.")

Visibility = EnumDef(
    "Visibility",
    [
        # Value 0 is reserved for "absent" (private) in optional enum attrs.
        EnumCase("public", 1, doc="Visible outside the module (exported)."),
    ],
    doc="Function visibility. Absent (0) means private (module-internal).",
)

Retain = EnumDef(
    "Retain",
    [
        # Value 0 is reserved for "absent" (ordinary DCE may erase).
        EnumCase("retain", 1, doc="Preserve the symbol across ordinary DCE."),
    ],
    doc="Private symbol retention policy. Absent (0) permits ordinary DCE.",
)

CallingConv = EnumDef(
    "CallingConv",
    [
        # Value 0 is reserved for "absent" (default/host) in optional enum attrs.
        EnumCase("host", 1, doc="Host calling convention."),
        EnumCase("device", 2, doc="Device calling convention."),
        EnumCase("initializer", 3, doc="Module initialization function."),
        EnumCase("deinitializer", 4, doc="Module deinitialization function."),
    ],
    doc="Function calling convention. Absent (0) means host.",
)

Purity = EnumDef(
    "Purity",
    [
        EnumCase("pure", 1, doc="No memory effects, deterministic."),
    ],
    doc="Function purity. Absent (0) means unspecified (conservative).",
)

Temperature = EnumDef(
    "Temperature",
    [
        EnumCase("hot", 1, doc="Expected to execute on a hot path."),
        EnumCase("cold", 2, doc="Expected to execute on a cold path."),
    ],
    doc="Execution temperature hint. Absent (0) means unspecified.",
)

InlinePolicyAttr = EnumDef(
    "InlinePolicy",
    [
        EnumCase(
            "inline",
            InlinePolicy.INLINE.value,
            doc="Require inlining at the current IR stage.",
        ),
        EnumCase(
            "noinline",
            InlinePolicy.NOINLINE.value,
            doc="Preserve the callable boundary.",
        ),
    ],
    doc="Author inline policy. Absent (0) leaves the edge to the current pass.",
    c_type="loom_inline_policy_t",
    c_const_prefix="LOOM_INLINE_POLICY",
    c_include="loom/ir/ir.h",
)

_RETAIN_ATTR = AttrDef("retain", "enum", enum_def=Retain, optional=True)

# ============================================================================
# Shared format fragments
# ============================================================================

# Modifiers appear after the op name, before the symbol:
#   func.def public pure @name(...)
#   func.decl import("module") @name(...)
#   func.decl public import("module", "original") @alias(...)
# Modifiers shared by all func-like ops:
#   func.def public pure @name(...)
_MODIFIER_FORMAT: list[FormatElement] = [
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    OptionalGroup([Attr("retain")], anchor="retain"),
    OptionalGroup([Attr("cc")], anchor="cc"),
    OptionalGroup([Attr("purity")], anchor="purity"),
    OptionalGroup([Attr("temperature")], anchor="temperature"),
    OptionalGroup([Attr("inline_policy")], anchor="inline_policy"),
]

_TARGET_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [kw("target"), GLUE, LPAREN, SymbolRef("target"), GLUE, RPAREN],
        anchor="target",
    ),
]

_ABI_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("abi"),
            GLUE,
            LPAREN,
            Attr("abi"),
            OptionalGroup([COMMA, AttrDict("abi_attrs")], anchor="abi_attrs"),
            GLUE,
            RPAREN,
        ],
        anchor="abi",
    ),
]

_EXPORT_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("export"),
            GLUE,
            LPAREN,
            Attr("export_symbol"),
            OptionalGroup([COMMA, AttrDict("export_attrs")], anchor="export_attrs"),
            GLUE,
            RPAREN,
        ],
        anchor="export_symbol",
    ),
]

# Additional import modifier for func.decl only:
#   func.decl import("module") @name(...)
#   func.decl public import("module", "original") @alias(...)
# Templates and ukernels are discovered by specialization passes, not imported.
_IMPORT_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("import"),
            GLUE,
            LPAREN,
            GLUE,
            Attr("import_module"),
            OptionalGroup([COMMA, Attr("import_symbol")], anchor="import_symbol"),
            GLUE,
            RPAREN,
        ],
        anchor="import_module",
    ),
]

# Signature: @name(%a: type, ...) -> (type, ...) where [...]
_SIGNATURE_FORMAT: list[FormatElement] = [
    SymbolRef("callee"),
    Scope(
        [
            FuncArgs("args"),
            OptionalGroup(
                [ARROW, ResultTypeList("results")],
                anchor="results",
            ),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")],
                anchor="predicates",
            ),
        ]
    ),
]

_MODIFIER_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef("cc", "enum", enum_def=CallingConv, optional=True),
    AttrDef("purity", "enum", enum_def=Purity, optional=True),
    AttrDef("temperature", "enum", enum_def=Temperature, optional=True),
    AttrDef("inline_policy", "enum", enum_def=InlinePolicyAttr, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]

_CONTRACT_ATTRS = [
    AttrDef(
        "target",
        "symbol",
        optional=True,
        symbol_ref=SymbolReference("target", ["target"]),
    ),
    AttrDef(
        "abi",
        "enum",
        enum_def=ExportAbiKind,
        optional=True,
        open_enum=True,
    ),
    AttrDef("abi_attrs", "dict", optional=True),
    AttrDef("export_symbol", "string", optional=True),
    AttrDef("export_attrs", "dict", optional=True),
]

# func.decl adds import attrs to the shared modifier set.
_DECL_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef("import_module", "string", optional=True),
    AttrDef("import_symbol", "string", optional=True),
    AttrDef("cc", "enum", enum_def=CallingConv, optional=True),
    AttrDef("purity", "enum", enum_def=Purity, optional=True),
    AttrDef("temperature", "enum", enum_def=Temperature, optional=True),
    AttrDef("inline_policy", "enum", enum_def=InlinePolicyAttr, optional=True),
    *_CONTRACT_ATTRS,
    AttrDef("predicates", "predicate_list", optional=True),
    _RETAIN_ATTR,
]

# ============================================================================
# FuncLike interface declarations
# ============================================================================

# Shared interface fields for all func-like ops.
_FUNC_LIKE_COMMON: dict[str, Any] = dict(
    callee="callee",
    visibility="visibility",
    cc="cc",
    purity="purity",
    temperature="temperature",
    inline_policy="inline_policy",
    predicates="predicates",
)

_FUNC_LIKE_CONTRACT: dict[str, Any] = dict(
    **_FUNC_LIKE_COMMON,
    target="target",
    abi="abi",
    abi_attrs="abi_attrs",
    export_symbol="export_symbol",
    export_attrs="export_attrs",
)

_FUNC_LIKE_DECL_CONTRACT: dict[str, Any] = dict(
    **_FUNC_LIKE_CONTRACT,
    import_module="import_module",
    import_symbol="import_symbol",
)

# ============================================================================
# func.def — function definition
# ============================================================================

func_def = Op(
    "func.def",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Function definition. Callable by name via func.call. Function arguments "
        "retain their initial values throughout the body, including in dependent "
        "result types and predicates. The entry block has no internal "
        "predecessors; CFG loops carry changing values on separate header blocks."
    ),
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=[*_MODIFIER_ATTRS, *_CONTRACT_ATTRS, _RETAIN_ATTR],
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like", "callable"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
    ),
    results=[Result("results", ANY, variadic=True)],
    regions=[RegionDef("body", doc="Function body.", terminator="func.return")],
    interfaces=[FuncLikeInterface(**_FUNC_LIKE_CONTRACT, body="body")],
    verify="loom_func_def_verify",
    format=[
        *_MODIFIER_FORMAT,
        *_TARGET_FORMAT,
        *_ABI_FORMAT,
        *_EXPORT_FORMAT,
        *_SIGNATURE_FORMAT,
        Region("body"),
    ],
    examples=[
        "func.def @negate(%input: f32) -> (f32) {\n  func.return %input : f32\n}",
        "func.def public device @entry(%a: f32) -> (f32) {\n  func.return %a : f32\n}",
        "func.def public pure @add(%a: f32, %b: f32) -> (f32) {\n  func.return %a : f32\n}",
        "func.def cold noinline @serializer(%a: f32) -> (f32) {\n  func.return %a : f32\n}",
    ],
)

# ============================================================================
# func.decl — function declaration
# ============================================================================

func_decl = Op(
    "func.decl",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="External function declaration. Callable by name via func.call.",
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", ANY, variadic=True)],
    attrs=list(_DECL_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like", "callable"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
        flags=[SymbolDefinitionFlag.DECLARATION],
    ),
    results=[Result("results", ANY, variadic=True)],
    interfaces=[FuncLikeInterface(**_FUNC_LIKE_DECL_CONTRACT, args="args")],
    verify="loom_func_decl_verify",
    format=[
        OptionalGroup([Attr("visibility")], anchor="visibility"),
        OptionalGroup([Attr("retain")], anchor="retain"),
        *_IMPORT_FORMAT,
        OptionalGroup([Attr("cc")], anchor="cc"),
        OptionalGroup([Attr("purity")], anchor="purity"),
        OptionalGroup([Attr("temperature")], anchor="temperature"),
        OptionalGroup([Attr("inline_policy")], anchor="inline_policy"),
        *_TARGET_FORMAT,
        *_ABI_FORMAT,
        *_EXPORT_FORMAT,
        *_SIGNATURE_FORMAT,
    ],
    examples=[
        "func.decl @extern_matmul(%a: tensor<[%M]xf32>, %b: tensor<[%K]xf32>) -> (tensor<[%M]xf32>)",
        "func.decl public @exported(%a: f32) -> (f32)",
        "func.decl hot inline @tiny(%a: f32) -> (f32)",
        'func.decl public import("hal") @hal_buffer_view_create(%a: i32) -> (i64)',
        'func.decl import("hal", "buffer_view.create") @hal_buffer_view_create(%a: i32) -> (i64)',
    ],
)

# ============================================================================
# func.call — function-like symbol call
# ============================================================================

func_call = Op(
    "func.call",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Function-like symbol call. Runtime calls target func.def/func.decl; required-inline exact template calls are consumed before executable lowering.",
    operands=[
        Operand("operands", ANY, variadic=True),
    ],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        ),
        AttrDef("purity", "enum", enum_def=Purity, optional=True),
        AttrDef("temperature", "enum", enum_def=Temperature, optional=True),
        AttrDef("inline_policy", "enum", enum_def=InlinePolicyAttr, optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="operands",
            results="results",
            purity="purity",
            temperature="temperature",
            inline_policy="inline_policy",
            kind=CallLikeKind.SEMANTIC,
        ),
    ],
    verify="loom_func_call_verify",
    canonicalize="loom_func_call_canonicalize",
    effective_traits="loom_func_call_effective_traits",
    format=[
        OptionalGroup([Attr("purity")], anchor="purity"),
        OptionalGroup([Attr("temperature")], anchor="temperature"),
        OptionalGroup([Attr("inline_policy")], anchor="inline_policy"),
        SymbolRef("callee"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "%r = func.call @add(%a, %b) : (f32, f32) -> (f32)",
        "%r = func.call pure @add(%a, %b) : (f32, f32) -> (f32)",
        "%r = func.call hot inline @add(%a, %b) : (f32, f32) -> (f32)",
        "%r = func.call inline @specific_template(%a, %b) : (f32, f32) -> (f32)",
        "%out, %count = func.call @process(%a, %b) : (tensor<[%M]xf32>, index) -> (%a as tensor<[%M]xf32>, index)",
    ],
)

# ============================================================================
# func.return — return from function body
# ============================================================================

func_return = Op(
    "func.return",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Return values from function body. Types must match enclosing function's result types.",
    operands=[Operand("operands", ANY, variadic=True)],
    traits=[TERMINATOR, POISON_BOUNDARY],
    format=[
        OptionalGroup(
            [Refs("operands"), COLON, TypesOf("operands")],
            anchor="operands",
        ),
    ],
    examples=[
        "func.return",
        "func.return %r : f32",
        "func.return %a, %b : tensor<[%M]xf32>, index",
    ],
)

# ============================================================================
# func.location — captured source provenance
# ============================================================================

# Captured locations are semantic values. A flat postorder node array keeps
# shared provenance graphs independent of debug-location tables and aggregate
# attribute nesting limits. Child indices always refer to preceding nodes.
func_location_unknown_attr = ParameterizedAttrDef(
    "func.location.unknown",
    group=func_ops,
    parameters=[],
    doc="An explicitly unknown captured location.",
)

LocationFieldKind = EnumDef(
    "LocationFieldKind",
    [
        EnumCase("operand", 0),
        EnumCase("result", 1),
        EnumCase("attribute", 2),
        EnumCase("region", 3),
        EnumCase("successor", 4),
    ],
    c_type="loom_location_field_kind_t",
    c_const_prefix="LOOM_LOCATION_FIELD",
    c_include="loom/ir/location.h",
    doc="Source field category in a captured file range.",
)

func_location_field_attr = ParameterizedAttrDef(
    "func.location.field",
    group=func_ops,
    parameters=[
        AttrDef("kind", "enum", enum_def=LocationFieldKind, doc="Source field category."),
        AttrDef("index", "i64", doc="Zero-based field index."),
        AttrDef("range", "i64_array", doc="Start line/column and exclusive end line/column."),
    ],
    primary_parameter="kind",
    doc="A captured source field range in its file node's coordinate space.",
)

func_location_file_attr = ParameterizedAttrDef(
    "func.location.file",
    group=func_ops,
    parameters=[
        AttrDef("source", "string", doc="Original source name."),
        AttrDef("range", "i64_array", doc="Four unsigned 32-bit coordinates: one-based lines and Unicode code-point columns, with an exclusive end."),
        AttrDef("synthetic", "bool", optional=True, doc="The source marks this location as compiler-generated."),
        AttrDef("fields", "parameterized_array", parameterized_attr=func_location_field_attr, optional=True),
        AttrDef("text", "bytes", optional=True, doc="Exact source lines spanning the range; absent when unavailable."),
    ],
    primary_parameter="source",
    doc="Captured file range and optional original source text beginning at the start line.",
)

func_location_fused_attr = ParameterizedAttrDef(
    "func.location.fused",
    group=func_ops,
    parameters=[
        AttrDef("children", "i64_array", doc="Ordered indices of preceding location nodes."),
        AttrDef("synthetic", "bool", optional=True, doc="The source marks this location as compiler-generated."),
    ],
    primary_parameter="children",
    doc="Provenance derived from several captured locations.",
)

func_location_opaque_attr = ParameterizedAttrDef(
    "func.location.opaque",
    group=func_ops,
    parameters=[
        AttrDef("source", "string", doc="External provenance namespace."),
        AttrDef("data", "bytes", doc="Uninterpreted external provenance bytes."),
        AttrDef("synthetic", "bool", optional=True, doc="The source marks this location as compiler-generated."),
    ],
    primary_parameter="source",
    doc="Captured external source identity and uninterpreted payload.",
)

func_location_tagged_attr = ParameterizedAttrDef(
    "func.location.tagged",
    group=func_ops,
    parameters=[
        AttrDef("tag", "i64", doc="Stable nonzero location tag."),
        AttrDef("data", "bytes", doc="Uninterpreted tag payload."),
        AttrDef("child", "i64", optional=True, doc="Index of a preceding child location, when present."),
        AttrDef("synthetic", "bool", optional=True, doc="The source marks this location as compiler-generated."),
    ],
    primary_parameter="tag",
    doc="Tagged provenance with an optional captured child.",
)

ALL_FUNC_PARAMETERIZED_ATTRS = (
    func_location_unknown_attr,
    func_location_field_attr,
    func_location_file_attr,
    func_location_fused_attr,
    func_location_opaque_attr,
    func_location_tagged_attr,
)

func_location = Op(
    "func.location",
    group=func_ops,
    doc=(
        "Materialize immutable captured source provenance as a read-only buffer. "
        "The required postorder node array ends with the root location; children "
        "refer to earlier nodes. All captured data is semantic and survives "
        "debug stripping. This operation never reads its own debug annotation. "
        "Equal complete captures may share executable rodata."
    ),
    attrs=[AttrDef("nodes", "parameterized_array")],
    results=[Result("result", BUFFER)],
    traits=[PURE],
    format=[Attr("nodes"), COLON, ResultType("result")],
    verify="loom_func_location_verify",
    examples=['%site = func.location [#func.location.file<"example.cc", range = [12, 3, 12, 28]>] : buffer'],
)

# ============================================================================
# All ops
# ============================================================================

ALL_FUNC_OPS: tuple[Op, ...] = (
    func_def,
    func_decl,
    func_call,
    func_return,
    func_location,
)
