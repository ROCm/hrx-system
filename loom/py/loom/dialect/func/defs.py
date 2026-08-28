# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Func dialect type and op definitions.

Eleven ops for runtime program structure and first-class function values:

Top-level (module-level symbols):
  func.def       — Function definition (has body, callable by name).
  func.decl      — External function declaration (no body, callable by name).
Body ops:
  func.call      — Runtime function call.
  func.call.indirect — Runtime call through a first-class function value.
  func.return    — Return values from function body.
  func.fail      — Terminate with an explicit program status.
  func.null      — Null first-class function value.
  func.compare.null — Test a first-class function value for null.
  func.address   — Address a local or imported function.
  func.ref.cast  — Widen a synchronous function reference to yieldable.
  func.import.resolved — Test whether an optional import resolved.
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
    Param,
    PredicateList,
    Ref,
    Refs,
    Region,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.target.defs import ExportAbiKind
from loom.dsl import (
    ANY,
    BUFFER,
    I1,
    ISOLATED_FROM_ABOVE,
    NO_RETURN,
    POISON_BOUNDARY,
    PURE,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    VALUE_ALIAS,
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
    RegionDef,
    Result,
    SymbolDefinition,
    SymbolDefinitionFlag,
    SymbolReference,
    TypeDef,
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

ImportPolicy = EnumDef(
    "ImportPolicy",
    [
        # Value 0 is reserved for required imports.
        EnumCase(
            "optional",
            1,
            doc="Permit module linking to leave the import unresolved.",
        ),
    ],
    doc="Import resolution policy. Absent (0) means required.",
)

Yieldability = EnumDef(
    "Yieldability",
    [
        # Value 0 is reserved for the default synchronous contract.
        EnumCase(
            "yieldable",
            1,
            doc="Permit the referenced function to suspend and later resume.",
        ),
    ],
    doc="Function-reference yieldability. Absent (0) guarantees synchronous execution.",
)

func_ref_type = TypeDef(
    "func.ref",
    params=[
        AttrDef(
            "yieldability",
            "enum",
            enum_def=Yieldability,
            optional=True,
            doc="Optional permission for the referenced function to yield.",
        ),
        AttrDef(
            "signature",
            "type",
            doc="Exact structural argument and result signature.",
        ),
    ],
    format=[
        OptionalGroup([Param("yieldability")], anchor="yieldability"),
        Param("signature"),
    ],
    doc=(
        "First-class function reference. A synchronous reference guarantees "
        "that calls return without yielding; a yieldable reference permits "
        "suspension. The nested function type is the exact callable signature."
    ),
)

StatusCode = EnumDef(
    "StatusCode",
    [
        EnumCase("cancelled", 1, doc="Operation was cancelled."),
        EnumCase("unknown", 2, doc="Unknown failure."),
        EnumCase("invalid_argument", 3, doc="Invalid argument."),
        EnumCase("deadline_exceeded", 4, doc="Deadline exceeded."),
        EnumCase("not_found", 5, doc="Requested entity was not found."),
        EnumCase("already_exists", 6, doc="Entity already exists."),
        EnumCase("permission_denied", 7, doc="Permission was denied."),
        EnumCase("resource_exhausted", 8, doc="Resource was exhausted."),
        EnumCase("failed_precondition", 9, doc="Precondition was not met."),
        EnumCase("aborted", 10, doc="Operation was aborted."),
        EnumCase("out_of_range", 11, doc="Value was out of range."),
        EnumCase("unimplemented", 12, doc="Operation is not implemented."),
        EnumCase("internal", 13, doc="Internal failure."),
        EnumCase("unavailable", 14, doc="Service is unavailable."),
        EnumCase("data_loss", 15, doc="Unrecoverable data loss."),
        EnumCase("unauthenticated", 16, doc="Authentication is required."),
        EnumCase("incompatible", 18, doc="Contracts are incompatible."),
    ],
    doc="Non-OK program status returned by func.fail.",
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

_EXPORT_METADATA_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("export_metadata"),
            GLUE,
            LPAREN,
            AttrDict("export_metadata"),
            GLUE,
            RPAREN,
        ],
        anchor="export_metadata",
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

_IMPORT_METADATA_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("import_metadata"),
            GLUE,
            LPAREN,
            AttrDict("import_metadata"),
            GLUE,
            RPAREN,
        ],
        anchor="import_metadata",
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
    AttrDef(
        "export_metadata",
        "dict",
        optional=True,
        doc="Stable typed metadata owned by this export declaration.",
    ),
]

# func.decl adds import attrs to the shared modifier set.
_DECL_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef("import_module", "string", optional=True),
    AttrDef("import_symbol", "string", optional=True),
    AttrDef("import_policy", "enum", enum_def=ImportPolicy, optional=True),
    AttrDef(
        "import_metadata",
        "dict",
        optional=True,
        doc="Stable typed metadata owned by this import declaration.",
    ),
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
    export_metadata="export_metadata",
)

_FUNC_LIKE_DECL_CONTRACT: dict[str, Any] = dict(
    **_FUNC_LIKE_CONTRACT,
    import_module="import_module",
    import_symbol="import_symbol",
    import_policy="import_policy",
    import_metadata="import_metadata",
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
        *_EXPORT_METADATA_FORMAT,
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
        OptionalGroup([Attr("import_policy")], anchor="import_policy"),
        *_IMPORT_FORMAT,
        *_IMPORT_METADATA_FORMAT,
        OptionalGroup([Attr("cc")], anchor="cc"),
        OptionalGroup([Attr("purity")], anchor="purity"),
        OptionalGroup([Attr("temperature")], anchor="temperature"),
        OptionalGroup([Attr("inline_policy")], anchor="inline_policy"),
        *_TARGET_FORMAT,
        *_ABI_FORMAT,
        *_EXPORT_FORMAT,
        *_EXPORT_METADATA_FORMAT,
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
# First-class function values
# ============================================================================

func_null = Op(
    "func.null",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Produce a null first-class function value of the declared type.",
    results=[Result("result", ANY)],
    traits=[PURE],
    verify="loom_func_null_verify",
    format=[COLON, ResultType("result")],
    examples=["%null = func.null : func.ref<(i32) -> (i32)>"],
)

func_compare_null = Op(
    "func.compare.null",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Return true when a first-class function value is null.",
    operands=[Operand("function", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    verify="loom_func_compare_null_verify",
    format=[Ref("function"), COLON, TypeOf("function")],
    examples=["%is_null = func.compare.null %function : func.ref<(i32) -> (i32)>"],
)

func_address = Op(
    "func.address",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Produce a first-class function value addressing a callable symbol.",
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        ),
    ],
    results=[Result("result", ANY)],
    traits=[PURE],
    verify="loom_func_address_verify",
    format=[SymbolRef("callee"), COLON, ResultType("result")],
    examples=[
        "%function = func.address @callee : func.ref<(i32) -> (i32)>",
        "%function = func.address @callee : func.ref<yieldable (i32) -> (i32)>",
    ],
)

func_ref_cast = Op(
    "func.ref.cast",
    group=func_ops,
    builder_name="ref_cast",
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Widen a synchronous function reference to a yieldable reference with "
        "the same structural signature. The result aliases the same function "
        "value and only forgets the synchronous-call guarantee."
    ),
    operands=[Operand("source", ANY)],
    results=[Result("result", ANY)],
    traits=[PURE, VALUE_ALIAS],
    verify="loom_func_ref_cast_verify",
    format=[
        Ref("source"),
        COLON,
        TypeOf("source"),
        kw("to"),
        ResultType("result"),
    ],
    examples=["%yieldable = func.ref.cast %sync : func.ref<(i32) -> (i32)> to func.ref<yieldable (i32) -> (i32)>"],
)

func_import_resolved = Op(
    "func.import.resolved",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Return true when an optional imported function resolved during linking.",
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["callable"]),
        ),
    ],
    results=[Result("result", I1)],
    traits=[PURE],
    verify="loom_func_import_resolved_verify",
    format=[SymbolRef("callee")],
    examples=["%available = func.import.resolved @optional_feature"],
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
# func.call.indirect — first-class function call
# ============================================================================

func_call_indirect = Op(
    "func.call.indirect",
    group=func_ops,
    builder_name="call_indirect",
    phase=OpPhase.EXECUTABLE,
    doc="Call a first-class function value with an exact structural signature.",
    operands=[
        Operand("target", ANY),
        Operand("operands", ANY, variadic=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_func_call_indirect_verify",
    format=[
        Ref("target"),
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
        "%result = func.call.indirect %target(%value) : (i32) -> (i32)",
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
# func.fail — fail the current invocation
# ============================================================================

func_fail = Op(
    "func.fail",
    group=func_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Terminate the current invocation with a status and diagnostic message.",
    operands=[
        Operand(
            "message",
            BUFFER,
            doc="Diagnostic message captured before the invocation unwinds.",
        ),
    ],
    attrs=[
        AttrDef(
            "status",
            "enum",
            enum_def=StatusCode,
            doc="Non-OK status returned by the invocation.",
        ),
    ],
    traits=[TERMINATOR, NO_RETURN, POISON_BOUNDARY, UNKNOWN_EFFECTS],
    format=[
        Attr("status"),
        COMMA,
        Ref("message"),
        COLON,
        TypeOf("message"),
    ],
    examples=["func.fail invalid_argument, %message : buffer"],
)

# ============================================================================
# All ops
# ============================================================================

ALL_FUNC_OPS: tuple[Op, ...] = (
    func_def,
    func_decl,
    func_call,
    func_call_indirect,
    func_return,
    func_fail,
    func_null,
    func_compare_null,
    func_address,
    func_ref_cast,
    func_import_resolved,
)

ALL_FUNC_TYPES: tuple[TypeDef, ...] = (func_ref_type,)
