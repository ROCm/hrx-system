# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Func dialect op definitions.

Seven ops for program structure:

Top-level (module-level symbols):
  func.def       — Function definition (has body, callable by name).
  func.decl      — External function declaration (no body, callable by name).

Body ops (inside function/template bodies):
  func.call      — Runtime function call.
  func.apply     — Compile-time contract-key implementation demand.
  func.return    — Return values from function body.

func.template<T> and func.ukernel<T> provide implementation contract keys.
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
    OpRef,
    OptionalGroup,
    PredicateList,
    Refs,
    Region,
    ResultTypeList,
    Scope,
    SymbolRef,
    TypesOf,
    kw,
)
from loom.dialect.target.defs import ExportAbiKind
from loom.dsl import (
    ANY,
    ISOLATED_FROM_ABOVE,
    POISON_BOUNDARY,
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
    Op,
    Operand,
    OpPhase,
    RegionDef,
    Result,
    SymbolDefinition,
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

InlinePolicy = EnumDef(
    "InlinePolicy",
    [
        EnumCase("inline", 1, doc="Require inlining at the current IR stage."),
        EnumCase("noinline", 2, doc="Preserve the callable boundary."),
    ],
    doc="Author inline policy. Absent (0) leaves the edge to the current pass.",
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
    AttrDef("inline_policy", "enum", enum_def=InlinePolicy, optional=True),
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

_PROVIDER_ATTRS = [
    AttrDef(
        "target",
        "symbol",
        optional=True,
        symbol_ref=SymbolReference("target", ["target"]),
    ),
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
    AttrDef("inline_policy", "enum", enum_def=InlinePolicy, optional=True),
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

_FUNC_LIKE_PROVIDER: dict[str, Any] = dict(
    **_FUNC_LIKE_COMMON,
    target="target",
)

# ============================================================================
# func.def — function definition
# ============================================================================

func_def = Op(
    "func.def",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Function definition. Callable by name via func.call.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=[*_MODIFIER_ATTRS, *_CONTRACT_ATTRS, _RETAIN_ATTR],
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
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
    attrs=list(_DECL_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
    ),
    results=[Result("results", ANY, variadic=True)],
    interfaces=[FuncLikeInterface(**_FUNC_LIKE_DECL_CONTRACT, args_as_operands=True)],
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
# func.template<T> — constraint-matched visible implementation
# ============================================================================

func_template = Op(
    "func.template",
    group=func_ops,
    doc="Constraint-matched visible implementation of an abstract op.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=[
        AttrDef("implements", "string"),
        *_MODIFIER_ATTRS,
        *_PROVIDER_ATTRS,
        AttrDef("priority", "i64", optional=True),
        _RETAIN_ATTR,
    ],
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_TEMPLATE",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
    ),
    results=[Result("results", ANY, variadic=True)],
    regions=[RegionDef("body", doc="Template body.", terminator="func.return")],
    interfaces=[
        FuncLikeInterface(
            **_FUNC_LIKE_PROVIDER,
            body="body",
            implements="implements",
            priority="priority",
        )
    ],
    format=[
        OpRef("implements"),
        *_MODIFIER_FORMAT,
        *_TARGET_FORMAT,
        OptionalGroup(
            [kw("priority"), GLUE, LPAREN, GLUE, Attr("priority"), GLUE, RPAREN],
            anchor="priority",
        ),
        *_SIGNATURE_FORMAT,
        Region("body"),
    ],
    examples=[
        "func.template<tile.contract> device @vnni_q8(%w: tensor<[%M]xi8>, %x: tensor<[%K]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)] {\n  func.return %x : tensor<[%K]xf32>\n}",
        "func.template<tile.contract> target(@gfx1100) priority(20) @gfx11_q8(%a: tile<4xf32>) -> (tile<4xf32>) {\n  func.return %a : tile<4xf32>\n}",
        "func.template<tile.contract> priority(10) @high_priority(%a: tile<4xf32>) -> (tile<4xf32>) {\n  func.return %a : tile<4xf32>\n}",
    ],
)

# ============================================================================
# func.ukernel<T> — constraint-matched opaque implementation
# ============================================================================

func_ukernel = Op(
    "func.ukernel",
    group=func_ops,
    doc="Constraint-matched opaque implementation of an abstract op.",
    traits=[SYMBOL_DEFINE],
    attrs=[
        AttrDef("implements", "string"),
        *_MODIFIER_ATTRS,
        *_PROVIDER_ATTRS,
        AttrDef("priority", "i64", optional=True),
        _RETAIN_ATTR,
    ],
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_UKERNEL",
        fact_domain="loom_func_symbol_fact_domain",
        retain="retain",
    ),
    results=[Result("results", ANY, variadic=True)],
    interfaces=[
        FuncLikeInterface(
            **_FUNC_LIKE_PROVIDER,
            implements="implements",
            priority="priority",
            args_as_operands=True,
        )
    ],
    format=[
        OpRef("implements"),
        *_MODIFIER_FORMAT,
        *_TARGET_FORMAT,
        OptionalGroup(
            [kw("priority"), GLUE, LPAREN, GLUE, Attr("priority"), GLUE, RPAREN],
            anchor="priority",
        ),
        *_SIGNATURE_FORMAT,
    ],
    examples=[
        "func.ukernel<tile.contract> device @vnni_q8_asm(%w: tensor<[%M]xi8>, %x: tensor<[%K]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16)]",
        "func.ukernel<tile.contract> target(@gfx1100) priority(20) @gfx11_q8_asm(%a: tile<4xf32>) -> (tile<4xf32>)",
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
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
        AttrDef("purity", "enum", enum_def=Purity, optional=True),
        AttrDef("temperature", "enum", enum_def=Temperature, optional=True),
        AttrDef("inline_policy", "enum", enum_def=InlinePolicy, optional=True),
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
# func.apply — compile-time contract-key implementation demand
# ============================================================================

func_apply = Op(
    "func.apply",
    group=func_ops,
    doc="Compile-time implementation demand. Contract key must be selected before executable lowering.",
    operands=[
        Operand("operands", ANY, variadic=True),
    ],
    attrs=[
        AttrDef("contract", "string"),
        AttrDef("purity", "enum", enum_def=Purity, optional=True),
        AttrDef("temperature", "enum", enum_def=Temperature, optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    canonicalize="loom_func_apply_canonicalize",
    effective_traits="loom_func_apply_effective_traits",
    format=[
        OpRef("contract"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        OptionalGroup([Attr("purity")], anchor="purity"),
        OptionalGroup([Attr("temperature")], anchor="temperature"),
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
        "%r = func.apply<qwen.q4.matmul>(%w, %x) : (tensor<16x32xi8>, tensor<32xf32>) -> (tensor<16xf32>)",
        "%r = func.apply<qwen.q4.matmul>(%w, %x) pure : (tensor<16x32xi8>, tensor<32xf32>) -> (tensor<16xf32>)",
        "%r = func.apply<qwen.q4.matmul>(%w, %x) hot : (tensor<16x32xi8>, tensor<32xf32>) -> (tensor<16xf32>)",
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
# All ops
# ============================================================================

ALL_FUNC_OPS: tuple[Op, ...] = (
    func_def,
    func_decl,
    func_template,
    func_ukernel,
    func_call,
    func_apply,
    func_return,
)
