# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-low structural dialect op definitions.

The low dialect is a target-bound virtual-register IR that stays inside the
normal Loom module, symbol, bytecode, diagnostic, and pass infrastructure. Its
instruction semantics are descriptor-backed: the IR carries a small set of
structural ops, while target packages define opcode tables, operand classes,
scheduling facts, and emission recipes.
"""

from typing import Any

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    Attr,
    AttrDict,
    BlockArgs,
    BlockRef,
    DescriptorRef,
    FormatElement,
    FuncArgs,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    Region,
    ResultType,
    ResultTypeList,
    Scope,
    StableKeyRef,
    SymbolRef,
    TemplateParam,
    TypedRefs,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.func.defs import CallingConv, Purity, Visibility
from loom.dialect.target.defs import ExportAbiKind, ExportLinkage
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_TYPE,
    ISOLATED_FROM_ABOVE,
    PURE,
    REGISTER,
    STORAGE,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    AttrDef,
    BlockArgsSatisfy,
    CallLikeInterface,
    CallLikeKind,
    Dialect,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    ImplicitTerminator,
    IterArgsMatchResults,
    LoopLikeInterface,
    NoAncestor,
    Op,
    Operand,
    OpPhase,
    RegionBranchInterface,
    RegionDef,
    RegisterUnitsSumTo,
    Result,
    SameRegisterClass,
    SameType,
    Successor,
    SymbolDefinition,
    SymbolReference,
    YieldCountMatchesResults,
    YieldTypesMatchResults,
)

# ============================================================================
# Dialect
# ============================================================================

low_ops = Dialect(
    "low",
    dialect_id=0x14,
    doc="Target-low virtual-register operations.",
)

# ============================================================================
# Shared enums
# ============================================================================

LowResourceImportKind = EnumDef(
    "LowResourceImportKind",
    [
        EnumCase(
            "native_pointer",
            1,
            doc="Native object-function pointer argument materialized as a register value.",
        ),
        EnumCase(
            "vm_state",
            2,
            doc="IREE VM module state or context handle materialized as a register value.",
        ),
        EnumCase(
            "vm_import",
            3,
            doc="IREE VM imported-function or imported-resource handle materialized as a register value.",
        ),
        EnumCase(
            "hal_binding",
            4,
            doc="IREE HAL dispatch binding payload materialized as a register value.",
        ),
    ],
    doc="Target-provided ABI resource imported into a low function body.",
)

LowAllocationMode = EnumDef(
    "LowAllocationMode",
    [
        EnumCase(
            "virtual",
            1,
            doc="SSA values name virtual registers; allocation is still open.",
        ),
        EnumCase(
            "assigned",
            2,
            doc="Allocation tables assign physical registers, but rewriting may still repair copies/spills.",
        ),
        EnumCase(
            "fixed",
            3,
            doc="Physical register assignment is part of the low-function contract.",
        ),
    ],
    doc="Register allocation exactness mode for a low function. Absent means virtual.",
)

LowScheduleMode = EnumDef(
    "LowScheduleMode",
    [
        EnumCase("free", 1, doc="Instruction order may be scheduled."),
        EnumCase(
            "constrained",
            2,
            doc="Instruction order carries target constraints, but legal scheduling may still move packets.",
        ),
        EnumCase(
            "locked",
            3,
            doc="Instruction order is part of the low-function contract.",
        ),
    ],
    doc="Instruction scheduling exactness mode for a low function. Absent means free.",
)

LowScfForUnrollPolicy = EnumDef(
    "LowScfForUnrollPolicy",
    [
        EnumCase(
            "unroll",
            1,
            doc="Require full unrolling when the consuming target transform runs.",
        ),
    ],
    doc="Local low.scf.for unroll policy.",
)

LowCodeImportKind = EnumDef(
    "LowCodeImportKind",
    [
        EnumCase("vm", 1, doc="Imported implementation is an IREE VM symbol."),
        EnumCase("native", 2, doc="Imported implementation is a native callable symbol."),
        EnumCase("rocasm", 3, doc="Imported implementation is an AMDGPU rocasm symbol."),
        EnumCase("object", 4, doc="Imported implementation is a linked object-file symbol."),
    ],
    doc="External code source kind for an imported low function declaration.",
)

# ============================================================================
# Shared fragments
# ============================================================================

_FUNC_COMMON_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef(
        "target",
        "symbol",
        symbol_ref=SymbolReference("target", ["target"]),
    ),
    AttrDef(
        "abi",
        ATTR_TYPE_ENUM,
        enum_def=ExportAbiKind,
        optional=True,
        open_enum=True,
    ),
    AttrDef("abi_attrs", "dict", optional=True),
    AttrDef("abi_layout", "dict", optional=True),
    AttrDef("export_symbol", "string", optional=True),
    AttrDef("export_attrs", "dict", optional=True),
    AttrDef("visibility", "enum", enum_def=Visibility, optional=True),
    AttrDef("cc", "enum", enum_def=CallingConv, optional=True),
    AttrDef("purity", "enum", enum_def=Purity, optional=True),
    AttrDef("allocation", "enum", enum_def=LowAllocationMode, optional=True),
    AttrDef("schedule", "enum", enum_def=LowScheduleMode, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]

_KERNEL_COMMON_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef(
        "target",
        "symbol",
        symbol_ref=SymbolReference("target", ["target"]),
    ),
    AttrDef("abi_layout", "dict", optional=True),
    AttrDef("export_symbol", "string", optional=True),
    AttrDef("export_linkage", "enum", enum_def=ExportLinkage, optional=True),
    AttrDef("workgroup_size_x", ATTR_TYPE_I64, optional=True),
    AttrDef("workgroup_size_y", ATTR_TYPE_I64, optional=True),
    AttrDef("workgroup_size_z", ATTR_TYPE_I64, optional=True),
    AttrDef("allocation", "enum", enum_def=LowAllocationMode, optional=True),
    AttrDef("schedule", "enum", enum_def=LowScheduleMode, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]

_FUNC_DECL_IMPORT_ATTRS = [
    AttrDef("import_kind", "enum", enum_def=LowCodeImportKind, optional=True),
    AttrDef("code_symbol", "string", optional=True),
]

_FUNC_MODIFIER_FORMAT: list[FormatElement] = [
    OptionalGroup([Attr("visibility")], anchor="visibility"),
    OptionalGroup([Attr("cc")], anchor="cc"),
    OptionalGroup([Attr("purity")], anchor="purity"),
    OptionalGroup(
        [kw("allocation"), GLUE, LPAREN, Attr("allocation"), GLUE, RPAREN],
        anchor="allocation",
    ),
    OptionalGroup(
        [kw("schedule"), GLUE, LPAREN, Attr("schedule"), GLUE, RPAREN],
        anchor="schedule",
    ),
]

_LOW_EXACTNESS_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [kw("allocation"), GLUE, LPAREN, Attr("allocation"), GLUE, RPAREN],
        anchor="allocation",
    ),
    OptionalGroup(
        [kw("schedule"), GLUE, LPAREN, Attr("schedule"), GLUE, RPAREN],
        anchor="schedule",
    ),
]

_FUNC_TARGET_FORMAT: list[FormatElement] = [
    kw("target"),
    GLUE,
    LPAREN,
    SymbolRef("target"),
    GLUE,
    RPAREN,
]

_FUNC_ABI_FORMAT: list[FormatElement] = [
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

_FUNC_EXPORT_FORMAT: list[FormatElement] = [
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

_KERNEL_EXPORT_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [kw("export"), GLUE, LPAREN, Attr("export_symbol"), GLUE, RPAREN],
        anchor="export_symbol",
    ),
    OptionalGroup(
        [kw("linkage"), GLUE, LPAREN, Attr("export_linkage"), GLUE, RPAREN],
        anchor="export_linkage",
    ),
]

_KERNEL_ABI_LAYOUT_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [kw("abi_layout"), GLUE, LPAREN, AttrDict("abi_layout"), GLUE, RPAREN],
        anchor="abi_layout",
    ),
]

_KERNEL_WORKGROUP_SIZE_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("workgroup_size"),
            GLUE,
            LPAREN,
            Attr("workgroup_size_x"),
            COMMA,
            Attr("workgroup_size_y"),
            COMMA,
            Attr("workgroup_size_z"),
            GLUE,
            RPAREN,
        ],
        anchor="workgroup_size_x",
    ),
]

_FUNC_IMPORT_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [
            kw("import"),
            GLUE,
            LPAREN,
            Attr("import_kind"),
            COMMA,
            Attr("code_symbol"),
            GLUE,
            RPAREN,
        ],
        anchor="import_kind",
    ),
]

_FUNC_SIGNATURE_FORMAT: list[FormatElement] = [
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

_KERNEL_SIGNATURE_FORMAT: list[FormatElement] = [
    SymbolRef("callee"),
    Scope(
        [
            FuncArgs("args"),
            OptionalGroup(
                [kw("where"), PredicateList("predicates")],
                anchor="predicates",
            ),
        ]
    ),
]

_FUNC_LIKE_COMMON: dict[str, Any] = dict(
    callee="callee",
    target="target",
    abi="abi",
    abi_attrs="abi_attrs",
    export_symbol="export_symbol",
    export_attrs="export_attrs",
    visibility="visibility",
    cc="cc",
    purity="purity",
    predicates="predicates",
)

_KERNEL_FUNC_LIKE_COMMON: dict[str, Any] = dict(
    callee="callee",
    target="target",
    export_symbol="export_symbol",
    export_linkage="export_linkage",
    predicates="predicates",
)

# ============================================================================
# low.func.def — target-bound low function definition
# ============================================================================

low_func_def = Op(
    "low.func.def",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Target-bound low function definition with register-typed signature values.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=list(_FUNC_COMMON_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
        fact_domain="loom_func_symbol_fact_domain",
    ),
    results=[Result("results", REGISTER, variadic=True)],
    regions=[RegionDef("body", doc="Low function body.", terminator="low.return")],
    interfaces=[FuncLikeInterface(**_FUNC_LIKE_COMMON, body="body")],
    verify="loom_low_func_def_verify",
    constraints=[
        BlockArgsSatisfy("body", REGISTER),
        YieldCountMatchesResults("body", "results"),
        YieldTypesMatchResults("body", "results"),
    ],
    format=[
        *_FUNC_MODIFIER_FORMAT,
        *_FUNC_TARGET_FORMAT,
        *_FUNC_ABI_FORMAT,
        *_KERNEL_ABI_LAYOUT_FORMAT,
        *_FUNC_EXPORT_FORMAT,
        *_FUNC_SIGNATURE_FORMAT,
        Region("body", syntax="low.asm.optional"),
    ],
    examples=[
        "low.func.def target(@gfx1100) @add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n  %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>\n  low.return %sum : reg<amdgpu.vgpr x1>\n}",
        "low.func.def allocation(fixed) schedule(locked) target(@gfx1100) @agent_authored(%lhs: reg<amdgpu.vgpr x1>) {\n  low.return\n}",
    ],
)

# ============================================================================
# low.kernel.def — target-bound low kernel entry definition
# ============================================================================

low_kernel_def = Op(
    "low.kernel.def",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Target-bound low kernel entry with register-typed launch ABI values. Helper calls stay in low.func.def; kernel launch/export contracts live on this entry op."),
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=list(_KERNEL_COMMON_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
        fact_domain="loom_func_symbol_fact_domain",
    ),
    regions=[RegionDef("body", doc="Low kernel body.", terminator="low.return")],
    interfaces=[FuncLikeInterface(**_KERNEL_FUNC_LIKE_COMMON, body="body")],
    verify="loom_low_kernel_def_verify",
    constraints=[
        BlockArgsSatisfy("body", REGISTER),
    ],
    format=[
        *_LOW_EXACTNESS_FORMAT,
        *_FUNC_TARGET_FORMAT,
        *_KERNEL_ABI_LAYOUT_FORMAT,
        *_KERNEL_EXPORT_FORMAT,
        *_KERNEL_WORKGROUP_SIZE_FORMAT,
        *_KERNEL_SIGNATURE_FORMAT,
        Region("body", syntax="low.asm.optional"),
    ],
    examples=[
        'low.kernel.def target(@gfx1100) export("matmul") workgroup_size(16, 4, 1) @matmul(%lhs: reg<amdgpu.sgpr x4>, %rhs: reg<amdgpu.sgpr x4>, %out: reg<amdgpu.sgpr x4>) {\n  low.return\n}',
    ],
)

# ============================================================================
# low.func.decl — target-bound low function declaration
# ============================================================================

low_func_decl = Op(
    "low.func.decl",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Target-bound low function declaration with register-typed signature values.",
    traits=[SYMBOL_DEFINE],
    operands=[Operand("args", REGISTER, variadic=True)],
    attrs=[*_FUNC_COMMON_ATTRS, *_FUNC_DECL_IMPORT_ATTRS],
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DECL",
        fact_domain="loom_func_symbol_fact_domain",
    ),
    results=[Result("results", REGISTER, variadic=True)],
    interfaces=[FuncLikeInterface(**_FUNC_LIKE_COMMON, args_as_operands=True)],
    verify="loom_low_func_decl_verify",
    format=[
        *_FUNC_MODIFIER_FORMAT,
        *_FUNC_IMPORT_FORMAT,
        *_FUNC_TARGET_FORMAT,
        *_FUNC_ABI_FORMAT,
        *_KERNEL_ABI_LAYOUT_FORMAT,
        *_FUNC_EXPORT_FORMAT,
        *_FUNC_SIGNATURE_FORMAT,
    ],
    examples=[
        "low.func.decl target(@gfx1100) @extern_add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)",
        'low.func.decl allocation(fixed) schedule(locked) import(rocasm, "mfma_16x16_seq") target(@gfx1100) @mfma_rocasm(%acc: reg<amdgpu.vgpr x4>) -> (reg<amdgpu.vgpr x4>)',
    ],
)

# ============================================================================
# low.return — low function return terminator
# ============================================================================

low_return = Op(
    "low.return",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Return register values from a low function.",
    operands=[Operand("values", REGISTER, variadic=True)],
    traits=[TERMINATOR],
    format=[
        OptionalGroup(
            [Refs("values"), COLON, TypesOf("values")],
            anchor="values",
        ),
    ],
    examples=[
        "low.return",
        "low.return %value : reg<amdgpu.vgpr x1>",
    ],
)

# ============================================================================
# low.func.call — direct low function call
# ============================================================================

low_func_call = Op(
    "low.func.call",
    group=low_ops,
    doc="Direct call from one low function body to another same-target low function.",
    operands=[Operand("operands", REGISTER, variadic=True)],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
        AttrDef("purity", "enum", enum_def=Purity, optional=True),
    ],
    results=[Result("results", REGISTER, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="operands",
            results="results",
            purity="purity",
            kind=CallLikeKind.LOW_INTERNAL,
        ),
    ],
    effective_traits="loom_low_func_call_effective_traits",
    verify="loom_low_func_call_verify",
    format=[
        OptionalGroup([Attr("purity")], anchor="purity"),
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
        "%result = low.func.call @extern_add(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)",
        "%result = low.func.call pure @extern_add(%lhs) : (reg<vm.i32>) -> (reg<vm.i32>)",
    ],
)

# ============================================================================
# low.br — low unconditional branch
# ============================================================================

low_br = Op(
    "low.br",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Unconditional branch to a low successor block, forwarding register values.",
    operands=[
        Operand(
            "args",
            REGISTER,
            variadic=True,
            doc="Register values forwarded to the destination block arguments.",
        )
    ],
    successors=[Successor("dest", doc="Destination low block.")],
    traits=[TERMINATOR],
    verify="loom_low_br_verify",
    format=[
        BlockRef("dest"),
        OptionalGroup(
            [GLUE, LPAREN, TypedRefs("args"), RPAREN],
            anchor="args",
        ),
    ],
    examples=[
        "low.br ^done",
        "low.br ^join(%value: reg<vm.i32>)",
    ],
)

# ============================================================================
# low.cond_br — low conditional branch
# ============================================================================

low_cond_br = Op(
    "low.cond_br",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Conditional branch to one of two low successor blocks based on a register predicate.",
    operands=[Operand("condition", REGISTER, doc="Register predicate controlling the branch.")],
    successors=[
        Successor("true_dest", doc="Destination block when the predicate is true."),
        Successor("false_dest", doc="Destination block when the predicate is false."),
    ],
    successor_selector="condition",
    traits=[TERMINATOR],
    verify="loom_low_cond_br_verify",
    format=[
        Ref("condition"),
        COMMA,
        BlockRef("true_dest"),
        COMMA,
        BlockRef("false_dest"),
        COLON,
        TypeOf("condition"),
    ],
    examples=[
        "low.cond_br %condition, ^then, ^else : reg<vm.i32>",
    ],
)

# ============================================================================
# low.scf.yield — low structured-control region terminator
# ============================================================================

low_scf_yield = Op(
    "low.scf.yield",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Forward register values from a low structured-control region.",
    operands=[Operand("values", REGISTER, variadic=True)],
    traits=[TERMINATOR],
    format=[
        OptionalGroup(
            [Refs("values"), COLON, TypesOf("values")],
            anchor="values",
        ),
    ],
    examples=[
        "low.scf.yield",
        "low.scf.yield %value : reg<amdgpu.vgpr x1>",
    ],
)

# ============================================================================
# low.scf.if — low conditional structured control flow
# ============================================================================

low_scf_if = Op(
    "low.scf.if",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Conditional execution over target-low register values.",
    verify="loom_low_scf_if_verify",
    operands=[Operand("condition", REGISTER, doc="Register predicate controlling the branch.")],
    results=[Result("results", REGISTER, variadic=True)],
    regions=[
        RegionDef(
            "then_region",
            doc="Executed when condition is true. Terminated by low.scf.yield.",
            single_block=True,
            terminator="low.scf.yield",
        ),
        RegionDef(
            "else_region",
            doc="Executed when condition is false. Terminated by low.scf.yield.",
            single_block=True,
            optional=True,
            terminator="low.scf.yield",
        ),
    ],
    interfaces=[
        RegionBranchInterface(selector="condition"),
    ],
    constraints=[
        BlockArgsSatisfy("then_region", REGISTER),
        BlockArgsSatisfy("else_region", REGISTER),
        YieldCountMatchesResults("then_region", "results"),
        YieldTypesMatchResults("then_region", "results"),
        YieldCountMatchesResults("else_region", "results"),
        YieldTypesMatchResults("else_region", "results"),
    ],
    traits=[ImplicitTerminator("low.scf.yield")],
    format=[
        Ref("condition"),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("then_region", syntax="low.asm.optional"),
        OptionalGroup(
            [kw("else"), Region("else_region", syntax="low.asm.optional")],
            anchor="else_region",
        ),
    ],
    examples=[
        "low.scf.if %cond {\n  low.scf.yield\n}",
        "low.scf.if %cond {\n  low.scf.yield\n} else {\n  low.scf.yield\n}",
        "%result = low.scf.if %cond -> (reg<amdgpu.vgpr x1>) {\n  low.scf.yield %a : reg<amdgpu.vgpr x1>\n} else {\n  low.scf.yield %b : reg<amdgpu.vgpr x1>\n}",
    ],
)

# ============================================================================
# low.scf.for — low counted structured loop
# ============================================================================

low_scf_for = Op(
    "low.scf.for",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Bounded counted target-low loop with optional loop-carried register state.",
    verify="loom_low_scf_for_verify",
    operands=[
        Operand("lower_bound", REGISTER, doc="Inclusive lower bound register."),
        Operand("upper_bound", REGISTER, doc="Exclusive upper bound register."),
        Operand("step", REGISTER, doc="Positive step register."),
        Operand(
            "iter_args",
            REGISTER,
            variadic=True,
            doc="Initial loop-carried register values.",
        ),
        Operand(
            "unroll_factor",
            REGISTER,
            optional=True,
            doc="Optional dynamic target-domain unroll factor.",
        ),
    ],
    attrs=[
        AttrDef(
            "unroll_policy",
            ATTR_TYPE_ENUM,
            enum_def=LowScfForUnrollPolicy,
            optional=True,
            doc="Optional bare unroll policy for required full unroll.",
        ),
    ],
    results=[Result("results", REGISTER, variadic=True)],
    regions=[
        RegionDef(
            "body",
            doc="Loop body. Terminated by low.scf.yield.",
            single_block=True,
            terminator="low.scf.yield",
            implicit_args=(("iv", "type_of:lower_bound"),),
            arg_source="iter_args",
        ),
    ],
    interfaces=[
        LoopLikeInterface(
            body="body",
            iter_args="iter_args",
            iv="iv",
            lower_bound="lower_bound",
            upper_bound="upper_bound",
            step="step",
        ),
    ],
    constraints=[
        SameType("lower_bound", "upper_bound"),
        SameType("lower_bound", "step"),
        IterArgsMatchResults("iter_args", "results"),
        BlockArgsSatisfy("body", REGISTER),
        YieldCountMatchesResults("body", "results"),
        YieldTypesMatchResults("body", "results"),
    ],
    traits=[ImplicitTerminator("low.scf.yield")],
    format=[
        LBRACKET,
        Ref("lower_bound"),
        kw("to"),
        Ref("upper_bound"),
        kw("step"),
        Ref("step"),
        RBRACKET,
        OptionalGroup(
            [kw("iter_args"), GLUE, LPAREN, TypedRefs("iter_args"), RPAREN],
            anchor="iter_args",
        ),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        OptionalGroup(
            [kw("unroll"), GLUE, LPAREN, Ref("unroll_factor"), GLUE, RPAREN],
            anchor="unroll_factor",
        ),
        OptionalGroup(
            [Attr("unroll_policy")],
            anchor="unroll_policy",
        ),
        kw("do"),
        BlockArgs("body"),
        Region("body", syntax="low.asm.optional"),
    ],
    examples=[
        "low.scf.for [%lo to %hi step %step] do(%iv: reg<amdgpu.sgpr x1>) {\n  low.scf.yield\n}",
        "%result = low.scf.for [%lo to %hi step %step] iter_args(%acc0: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) do(%iv: reg<amdgpu.sgpr x1>, %acc: reg<amdgpu.vgpr x1>) {\n  low.scf.yield %acc : reg<amdgpu.vgpr x1>\n}",
    ],
)

# ============================================================================
# low.op — descriptor-backed target instruction
# ============================================================================

low_op = Op(
    "low.op",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Descriptor-backed target instruction over virtual registers.",
    operands=[Operand("operands", REGISTER, variadic=True)],
    attrs=[
        AttrDef("opcode", "string"),
        AttrDef("descriptor_ordinal", "i64"),
        AttrDef("attrs", "dict", optional=True),
    ],
    results=[Result("results", REGISTER, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_op_verify",
    format=[
        DescriptorRef("opcode", "descriptor_ordinal"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        AttrDict("attrs"),
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results", parens=False)],
            anchor="results",
        ),
    ],
    examples=[
        "%sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>",
        "low.op<amdgpu.s_waitcnt>() {vmcnt = 0} : ()",
    ],
)

# ============================================================================
# low.const — descriptor-backed register constant/materialization
# ============================================================================

low_const = Op(
    "low.const",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Descriptor-backed constant or immediate materialization into a register.",
    attrs=[
        AttrDef("opcode", "string"),
        AttrDef("descriptor_ordinal", "i64"),
        AttrDef("attrs", "dict", optional=True),
    ],
    results=[Result("result", REGISTER)],
    traits=[PURE],
    verify="loom_low_const_verify",
    facts="loom_low_const_facts",
    format=[
        DescriptorRef("opcode", "descriptor_ordinal"),
        AttrDict("attrs"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%c0 = low.const<amdgpu.s_mov_b32> {imm = 0} : reg<amdgpu.sgpr x1>",
    ],
)

# ============================================================================
# low.copy — explicit virtual-register copy/coalesce boundary
# ============================================================================

low_copy = Op(
    "low.copy",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Explicit virtual-register copy used by lowering and allocation. Each copy produces a fresh virtual-register identity."),
    operands=[Operand("source", REGISTER)],
    results=[Result("result", REGISTER, allocates=True)],
    constraints=[
        SameRegisterClass("source", "result"),
    ],
    verify="loom_low_copy_verify",
    facts="loom_low_copy_facts",
    format=[
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%copy = low.copy %value : reg<amdgpu.vgpr x1> -> reg<amdgpu.vgpr x1>",
    ],
)

# ============================================================================
# low.slice — project a contiguous subrange from a register range
# ============================================================================

low_slice = Op(
    "low.slice",
    group=low_ops,
    doc="Project a contiguous subrange from a register-range value.",
    attrs=[
        AttrDef("offset", "i64"),
    ],
    operands=[Operand("source", REGISTER)],
    results=[Result("result", REGISTER)],
    traits=[PURE],
    constraints=[
        SameRegisterClass("source", "result"),
    ],
    canonicalize="loom_low_slice_canonicalize",
    verify="loom_low_slice_verify",
    facts="loom_low_slice_facts",
    format=[
        Ref("source"),
        GLUE,
        LBRACKET,
        Attr("offset"),
        RBRACKET,
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%lane = low.slice %quad[2] : reg<amdgpu.vgpr x4> -> reg<amdgpu.vgpr>",
        "%pair = low.slice %quad[1] : reg<amdgpu.vgpr x4> -> reg<amdgpu.vgpr x2>",
    ],
)

# ============================================================================
# low.concat — compose a register range from ordered subranges
# ============================================================================

low_concat = Op(
    "low.concat",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Compose one fresh register-range identity from ordered register subranges."),
    operands=[Operand("sources", REGISTER, variadic=True)],
    results=[Result("result", REGISTER, allocates=True)],
    constraints=[
        SameRegisterClass("sources", "result"),
        RegisterUnitsSumTo("sources", "result"),
    ],
    facts="loom_low_concat_facts",
    format=[
        GLUE,
        LPAREN,
        Refs("sources"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("sources"),
        RPAREN,
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%pair = low.concat(%lo, %hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>",
        "%resource = low.concat(%ptr, %limit, %flags) : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>",
    ],
)

# ============================================================================
# low.live_in — target ABI live-in register value
# ============================================================================

low_live_in = Op(
    "low.live_in",
    group=low_ops,
    doc="Import a target-provided ABI live-in register value at low-function entry.",
    attrs=[
        AttrDef("source", "string"),
        AttrDef("source_id", "i64"),
        AttrDef("attrs", "dict", optional=True),
    ],
    results=[Result("result", REGISTER)],
    verify="loom_low_live_in_verify",
    format=[
        StableKeyRef("source", "source_id"),
        AttrDict("attrs"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%kernarg = low.live_in<amdgpu.kernarg_segment_ptr> : reg<amdgpu.sgpr x2>",
    ],
)

# ============================================================================
# low.storage.reserve — reserve function-local byte storage
# ============================================================================

low_storage_reserve = Op(
    "low.storage.reserve",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Reserve target-low function-local storage and preserve its segment footprint.",
    attrs=[
        AttrDef("byte_length", ATTR_TYPE_I64),
        AttrDef("byte_alignment", ATTR_TYPE_I64),
    ],
    results=[Result("storage", STORAGE, allocates=True)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_storage_reserve_verify",
    facts="loom_low_storage_reserve_facts",
    format=[
        AttrDict(),
        COLON,
        ResultType("storage"),
    ],
    examples=[
        "%slot = low.storage.reserve {byte_alignment = 4, byte_length = 16} : low.storage<private>",
    ],
)

# ============================================================================
# low.storage.view — project a byte subspan from low storage
# ============================================================================

low_storage_view = Op(
    "low.storage.view",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Project a byte subspan from function-local storage.",
    operands=[Operand("source", STORAGE)],
    attrs=[
        AttrDef("offset", ATTR_TYPE_I64, default=0, elide_default=True),
        AttrDef("byte_length", ATTR_TYPE_I64),
    ],
    results=[Result("result", STORAGE)],
    traits=[PURE],
    constraints=[
        SameType("source", "result"),
    ],
    verify="loom_low_storage_view_verify",
    facts="loom_low_storage_view_facts",
    format=[
        Ref("source"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%tile = low.storage.view %scratch {offset = 128, byte_length = 64} : low.storage<workgroup> -> low.storage<workgroup>",
    ],
)

# ============================================================================
# low.spill — explicit store from a register into low storage
# ============================================================================

low_spill = Op(
    "low.spill",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Explicit spill store from a register value into low storage.",
    operands=[
        Operand("value", REGISTER),
        Operand("storage", STORAGE),
    ],
    attrs=[
        AttrDef("offset", ATTR_TYPE_I64, default=0, elide_default=True),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_spill_verify",
    format=[
        Ref("value"),
        COMMA,
        Ref("storage"),
        AttrDict(),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("storage"),
    ],
    examples=[
        "low.spill %value, %slot : reg<amdgpu.vgpr x4>, low.storage<private>",
    ],
)

# ============================================================================
# low.reload — explicit load from low storage into a register
# ============================================================================

low_reload = Op(
    "low.reload",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Explicit reload from low storage into a register value.",
    operands=[Operand("storage", STORAGE)],
    attrs=[
        AttrDef("offset", ATTR_TYPE_I64, default=0, elide_default=True),
    ],
    results=[Result("result", REGISTER)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_low_reload_verify",
    format=[
        Ref("storage"),
        AttrDict(),
        COLON,
        TypeOf("storage"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%reload = low.reload %slot : low.storage<private> -> reg<amdgpu.vgpr x4>",
    ],
)

# ============================================================================
# low.storage.address — materialize a storage address
# ============================================================================

low_storage_address = Op(
    "low.storage.address",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Materialize a target address for function-local storage.",
    operands=[Operand("storage", STORAGE)],
    attrs=[
        AttrDef("offset", ATTR_TYPE_I64, default=0, elide_default=True),
    ],
    results=[Result("result", REGISTER)],
    traits=[PURE],
    verify="loom_low_storage_address_verify",
    format=[
        Ref("storage"),
        AttrDict(),
        COLON,
        TypeOf("storage"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%addr = low.storage.address %slot : low.storage<workgroup> -> reg<amdgpu.vgpr>",
    ],
)

# ============================================================================
# low.resource — import a function-local target resource into a register value
# ============================================================================

low_resource = Op(
    "low.resource",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Import a function-local target resource into a low register value.",
    operands=[
        Operand(
            "extent_value",
            REGISTER,
            optional=True,
            doc="Optional dynamic byte-addressable extent guaranteed valid for the imported resource.",
        ),
    ],
    attrs=[
        AttrDef("import_kind", ATTR_TYPE_ENUM, enum_def=LowResourceImportKind),
        AttrDef("index", ATTR_TYPE_I64),
        AttrDef("source_type", ATTR_TYPE_TYPE),
        AttrDef(
            "extent",
            ATTR_TYPE_I64,
            optional=True,
            doc="Optional static byte-addressable extent guaranteed valid for the imported resource.",
        ),
        AttrDef(
            "cache_swizzle_stride",
            ATTR_TYPE_I64,
            optional=True,
            doc="Optional byte stride that enables target resource-level cache swizzling.",
        ),
    ],
    results=[Result("result", REGISTER)],
    traits=[PURE],
    verify="loom_low_resource_verify",
    format=[
        TemplateParam("import_kind"),
        OptionalGroup(
            [kw("extent"), GLUE, LPAREN, Ref("extent_value"), GLUE, RPAREN],
            anchor="extent_value",
        ),
        AttrDict(),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%state = low.resource<vm_state> {index = 0, source_type = i64} : reg<vm.i64>",
        "%binding = low.resource<hal_binding> {index = 0, source_type = hal.buffer} : reg<amdgpu.sgpr x2>",
        "%dynamic = low.resource<hal_binding> extent(%extent) {index = 0, source_type = hal.buffer} : reg<amdgpu.sgpr x2>",
        "%swizzled = low.resource<hal_binding> {index = 1, source_type = hal.buffer, cache_swizzle_stride = 64} : reg<amdgpu.sgpr x2>",
    ],
)

# ============================================================================
# low.invoke — semantic interop edge to an explicit low function symbol
# ============================================================================

low_invoke = Op(
    "low.invoke",
    group=low_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Invoke an explicitly selected translated low function from non-low IR.",
    operands=[Operand("operands", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "callee",
            "symbol",
            symbol_ref=SymbolReference("function", ["func_like"]),
        ),
        AttrDef("purity", "enum", enum_def=Purity, optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[
        UNKNOWN_EFFECTS,
        NoAncestor("low.func.def"),
        NoAncestor("low.kernel.def"),
    ],
    interfaces=[
        CallLikeInterface(
            callee="callee",
            operands="operands",
            results="results",
            purity="purity",
            kind=CallLikeKind.LOW_INVOKE,
        ),
    ],
    effective_traits="loom_low_invoke_effective_traits",
    verify="loom_low_invoke_verify",
    format=[
        OptionalGroup([Attr("purity")], anchor="purity"),
        SymbolRef("callee"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        AttrDict(),
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
        "%result = low.invoke @extern_add(%lhs, %rhs) : (i32, i32) -> (i32)",
        "%result = low.invoke pure @extern_add(%lhs, %rhs) : (i32, i32) -> (i32)",
    ],
)

ALL_LOW_OPS: tuple[Op, ...] = (
    low_func_def,
    low_kernel_def,
    low_func_decl,
    low_return,
    low_func_call,
    low_op,
    low_const,
    low_copy,
    low_slice,
    low_concat,
    low_invoke,
    low_storage_reserve,
    low_storage_view,
    low_spill,
    low_reload,
    low_storage_address,
    low_br,
    low_cond_br,
    low_resource,
    low_live_in,
    low_scf_yield,
    low_scf_if,
    low_scf_for,
)
