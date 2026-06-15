# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Kernel dialect type and op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LPAREN,
    RPAREN,
    Attr,
    AttrDict,
    BlockArgs,
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
    SymbolRef,
    TemplateParam,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dialect.atomic import AtomicOrdering, AtomicScope
from loom.dialect.cache import CacheScope, CacheTemporal
from loom.dialect.combining import CombiningKind
from loom.dialect.memory import MemorySpace
from loom.dialect.target.defs import ExportLinkage
from loom.dsl import (
    ANY,
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    ATTR_TYPE_STRING,
    CONVERGENT,
    I1,
    INDEX,
    INTEGER,
    ISOLATED_FROM_ABOVE,
    PURE,
    SCALAR,
    SYMBOL_DEFINE,
    TERMINATOR,
    UNKNOWN_EFFECTS,
    VECTOR,
    VIEW,
    AttrDef,
    ContractFamily,
    Dialect,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    HasAncestor,
    HasParent,
    ImplicitTerminator,
    Op,
    Operand,
    OpPhase,
    Reads,
    RegionDef,
    Result,
    SameType,
    SymbolDefinition,
    SymbolReference,
    TypeDef,
    TypeSemantic,
    Writes,
)

_KERNEL_CONVERGENT_TRAITS = [CONVERGENT, HasAncestor("kernel.def")]

# ============================================================================
# Dialect
# ============================================================================

kernel_ops = Dialect(
    "kernel",
    dialect_id=0x10,
    doc="Kernel execution and synchronization operations.",
)

# ============================================================================
# Types
# ============================================================================

kernel_async_token_type = TypeDef(
    name="kernel.async.token",
    doc=("Opaque token for one initiated asynchronous memory transfer. A token must be committed to exactly one kernel.async.group."),
    semantic=TypeSemantic.CONTROL_TOKEN,
    contracts=[ContractFamily.KERNEL_ASYNC],
)

kernel_async_group_type = TypeDef(
    name="kernel.async.group",
    doc=("Opaque handle for one ordered asynchronous copy group. A group must be waited before leaving the kernel async-copy stream."),
    semantic=TypeSemantic.CONTROL_TOKEN,
    contracts=[ContractFamily.KERNEL_ASYNC],
)

kernel_tensor_lds_descriptor_type = TypeDef(
    name="kernel.tensor.lds.descriptor",
    semantic=TypeSemantic.TARGET_CONTRACT_VALUE,
    contracts=[ContractFamily.TENSOR_MEMORY],
    doc=(
        "Opaque descriptor grouping AMDGPU tensor-memory dgroups for one "
        "global/LDS tensor transfer. The descriptor contains the low-level "
        "dgroup values; the consuming async op still carries the source and "
        "destination views so layout, extent, and memory-space facts remain "
        "visible to Loom analyses."
    ),
)

# ============================================================================
# Shared attrs
# ============================================================================

KernelScope = AtomicScope
KernelMemorySpace = MemorySpace
KernelOrdering = AtomicOrdering

KernelDimension = EnumDef(
    "KernelDimension",
    [
        EnumCase("x", 0, doc="X dimension."),
        EnumCase("y", 1, doc="Y dimension."),
        EnumCase("z", 2, doc="Z dimension."),
    ],
    doc="Three-dimensional kernel coordinate axis.",
)

KernelAsyncDirection = EnumDef(
    "KernelAsyncDirection",
    [
        EnumCase(
            "global_to_workgroup",
            0,
            doc="Copy from global-like memory into workgroup/shared memory.",
        ),
        EnumCase(
            "workgroup_to_global",
            1,
            doc="Copy from workgroup/shared memory back to global-like memory.",
        ),
    ],
    doc="Required async copy direction.",
)

KernelSubgroupShuffleMode = EnumDef(
    "KernelSubgroupShuffleMode",
    [
        EnumCase("xor", 0, doc="Exchange with lane id xor offset."),
        EnumCase("up", 1, doc="Read from a lower-numbered lane by offset."),
        EnumCase("down", 2, doc="Read from a higher-numbered lane by offset."),
        EnumCase("index", 3, doc="Read from the lane named by offset."),
    ],
    doc="Subgroup lane shuffle addressing mode.",
)

KernelScanMode = EnumDef(
    "KernelScanMode",
    [
        EnumCase("inclusive", 0, doc="Include the current invocation value."),
        EnumCase("exclusive", 1, doc="Exclude the current invocation value."),
    ],
    doc="Subgroup scan inclusivity.",
)

KernelScanDirection = EnumDef(
    "KernelScanDirection",
    [
        EnumCase("forward", 0, doc="Scan from lower lane ids toward higher lane ids."),
        EnumCase("reverse", 1, doc="Scan from higher lane ids toward lower lane ids."),
    ],
    doc="Subgroup scan lane order.",
)

KernelWorkgroupScanMode = EnumDef(
    "KernelWorkgroupScanMode",
    [
        EnumCase("inclusive", 0, doc="Include the current invocation value."),
        EnumCase("exclusive", 1, doc="Exclude the current invocation value."),
    ],
    doc="Workgroup scan inclusivity.",
)

KernelWorkgroupScanDirection = EnumDef(
    "KernelWorkgroupScanDirection",
    [
        EnumCase(
            "forward",
            0,
            doc="Scan from lower workitem ids toward higher workitem ids.",
        ),
        EnumCase(
            "reverse",
            1,
            doc="Scan from higher workitem ids toward lower workitem ids.",
        ),
    ],
    doc="Workgroup scan workitem order.",
)

# ============================================================================
# Shared format fragments
# ============================================================================

_ENTRY_TARGET_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [kw("target"), GLUE, LPAREN, SymbolRef("target"), GLUE, RPAREN],
        anchor="target",
    ),
]

_ENTRY_EXPORT_FORMAT: list[FormatElement] = [
    OptionalGroup(
        [kw("export"), GLUE, LPAREN, Attr("export_symbol"), GLUE, RPAREN],
        anchor="export_symbol",
    ),
    OptionalGroup(
        [kw("linkage"), GLUE, LPAREN, Attr("export_linkage"), GLUE, RPAREN],
        anchor="export_linkage",
    ),
]

_ENTRY_CONFIG_FORMAT: list[FormatElement] = [
    SymbolRef("callee"),
    BlockArgs("config"),
]

_ENTRY_LAUNCH_FORMAT: list[FormatElement] = [
    kw("launch"),
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

_ENTRY_ATTRS = [
    AttrDef("callee", "symbol"),
    AttrDef(
        "target",
        "symbol",
        optional=True,
        symbol_ref=SymbolReference("target", ["target"]),
    ),
    AttrDef("export_symbol", "string", optional=True),
    AttrDef("export_linkage", "enum", enum_def=ExportLinkage, optional=True),
    AttrDef("predicates", "predicate_list", optional=True),
]


kernel_def = Op(
    "kernel.def",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Dispatchable source-level kernel entry. Kernel entries own launch and export contracts; ordinary func.def bodies remain helper/callable code."),
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    attrs=list(_ENTRY_ATTRS),
    symbol_def=SymbolDefinition(
        field="callee",
        name="function",
        interfaces=["func_like"],
        bytecode_kind="LOOM_SYMBOL_FUNC_DEF",
        fact_domain="loom_func_symbol_fact_domain",
    ),
    regions=[
        RegionDef(
            "config",
            doc=("Launch configuration region. The region owns host launch inputs and must terminate with kernel.launch.config."),
            single_block=True,
            terminator="kernel.launch.config",
            buffer_arg_memory_space="global",
        ),
        RegionDef(
            "body",
            doc="Kernel body.",
            terminator="kernel.return",
            buffer_arg_memory_space="global",
        ),
    ],
    interfaces=[
        FuncLikeInterface(
            callee="callee",
            target="target",
            export_symbol="export_symbol",
            export_linkage="export_linkage",
            predicates="predicates",
            body="body",
        )
    ],
    verify="loom_kernel_def_verify",
    format=[
        *_ENTRY_TARGET_FORMAT,
        *_ENTRY_EXPORT_FORMAT,
        *_ENTRY_CONFIG_FORMAT,
        Region("config"),
        *_ENTRY_LAUNCH_FORMAT,
        Region("body"),
    ],
    examples=[
        "kernel.def @entry() {\n  %one = index.constant 1 : index\n  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index\n} launch(%buffer: buffer) {\n  kernel.return\n}",
        'kernel.def target(@gfx1100) export("matmul") @matmul(%m: index, %n: index) {\n  %one = index.constant 1 : index\n  %threads = index.constant 256 : index\n  kernel.launch.config workgroups(%m, %n, %one) workgroup_size(%threads, %one, %one) : index\n} launch(%lhs: buffer, %rhs: buffer, %out: buffer) {\n  kernel.return\n}',
    ],
)


kernel_launch_config = Op(
    "kernel.launch.config",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Terminate a kernel launch configuration region with the computed workgroup grid and required workgroup size."),
    operands=[
        Operand(
            "workgroup_count_x",
            INDEX,
            doc="Number of workgroups to launch in the x dimension.",
        ),
        Operand(
            "workgroup_count_y",
            INDEX,
            doc="Number of workgroups to launch in the y dimension.",
        ),
        Operand(
            "workgroup_count_z",
            INDEX,
            doc="Number of workgroups to launch in the z dimension.",
        ),
        Operand(
            "workgroup_size_x",
            INDEX,
            doc="Required workgroup size in the x dimension.",
        ),
        Operand(
            "workgroup_size_y",
            INDEX,
            doc="Required workgroup size in the y dimension.",
        ),
        Operand(
            "workgroup_size_z",
            INDEX,
            doc="Required workgroup size in the z dimension.",
        ),
    ],
    traits=[TERMINATOR, HasParent("kernel.def")],
    format=[
        kw("workgroups"),
        GLUE,
        LPAREN,
        Ref("workgroup_count_x"),
        COMMA,
        Ref("workgroup_count_y"),
        COMMA,
        Ref("workgroup_count_z"),
        GLUE,
        RPAREN,
        kw("workgroup_size"),
        GLUE,
        LPAREN,
        Ref("workgroup_size_x"),
        COMMA,
        Ref("workgroup_size_y"),
        COMMA,
        Ref("workgroup_size_z"),
        GLUE,
        RPAREN,
        COLON,
        TypeOf("workgroup_count_x"),
    ],
    examples=[
        "kernel.launch.config workgroups(%gx, %gy, %gz) workgroup_size(%sx, %sy, %sz) : index",
    ],
)


kernel_return = Op(
    "kernel.return",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Return from a dispatchable kernel entry.",
    traits=[TERMINATOR, CONVERGENT],
    examples=["kernel.return"],
)


kernel_exit = Op(
    "kernel.exit",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Conditionally leaves the current kernel before executing the following top-level kernel-body operations."),
    operands=[Operand("condition", I1)],
    regions=[
        RegionDef(
            "body",
            doc="Optional work to run before leaving the kernel.",
            single_block=True,
            optional=True,
            terminator="kernel.return",
        ),
    ],
    traits=[
        UNKNOWN_EFFECTS,
        HasParent("kernel.def"),
        ImplicitTerminator("kernel.return"),
    ],
    canonicalize="loom_kernel_exit_canonicalize",
    format=[
        Ref("condition"),
        COLON,
        TypeOf("condition"),
        OptionalGroup(
            [Region("body")],
            anchor="body",
        ),
    ],
    examples=[
        "kernel.exit %done : i1",
        "kernel.exit %done : i1 {\n  kernel.return\n}",
    ],
)


kernel_assert = Op(
    "kernel.assert",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Runtime assertion inside a dispatchable kernel. The condition is "
        "expected to be true; if it is false, target lowering must preserve "
        "runtime failure semantics through a trap/assert path or reject the "
        "kernel when assertions cannot be represented. This is not an "
        "optimization assume."
    ),
    operands=[Operand("condition", I1, doc="Predicate that must hold.")],
    attrs=[
        AttrDef(
            "message",
            ATTR_TYPE_STRING,
            optional=True,
            doc="Optional human-readable assertion message.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS, HasAncestor("kernel.def")],
    format=[
        Ref("condition"),
        OptionalGroup(
            [Attr("message")],
            anchor="message",
        ),
        COLON,
        TypeOf("condition"),
    ],
    examples=[
        "kernel.assert %ok : i1",
        'kernel.assert %ok "finite input required" : i1',
    ],
)


def _async_cache_attrs() -> list[AttrDef]:
    return [
        AttrDef(
            "cache_scope",
            ATTR_TYPE_ENUM,
            enum_def=CacheScope,
            doc="Required cache/coherency scope for the transfer.",
        ),
        AttrDef(
            "cache_temporal",
            ATTR_TYPE_ENUM,
            enum_def=CacheTemporal,
            doc="Required temporal cache hint for the transfer.",
        ),
    ]


# ============================================================================
# kernel.workitem.id — current invocation coordinate within the workgroup
# ============================================================================

kernel_workitem_id = Op(
    name="kernel.workitem.id",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Read one coordinate of the current invocation within its workgroup. "
        "The result is a logical index value, not a byte offset; target "
        "lowering decides whether the coordinate is carried in scalar, vector, "
        "or dedicated target registers."
    ),
    results=[
        Result("result", INDEX, doc="Current workitem coordinate in the selected dimension."),
    ],
    attrs=[
        AttrDef(
            "dimension",
            ATTR_TYPE_ENUM,
            enum_def=KernelDimension,
            doc="Coordinate axis to read.",
        ),
    ],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_workitem_id_facts",
    format=[
        TemplateParam("dimension"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%tid = kernel.workitem.id<x> : index",
    ],
)


# ============================================================================
# kernel.workgroup.id — current workgroup coordinate within the dispatch grid
# ============================================================================

kernel_workgroup_id = Op(
    name="kernel.workgroup.id",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Read one coordinate of the current workgroup within the dispatch "
        "grid. The result is a logical index value; target lowering decides "
        "whether the coordinate is carried in scalar registers, ABI state, or "
        "target-specific builtin values."
    ),
    results=[
        Result("result", INDEX, doc="Current workgroup coordinate in the selected dimension."),
    ],
    attrs=[
        AttrDef(
            "dimension",
            ATTR_TYPE_ENUM,
            enum_def=KernelDimension,
            doc="Coordinate axis to read.",
        ),
    ],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_workgroup_id_facts",
    format=[
        TemplateParam("dimension"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%bid = kernel.workgroup.id<x> : index",
    ],
)


# ============================================================================
# Kernel launch query ops
# ============================================================================

kernel_workgroup_size = Op(
    name="kernel.workgroup.size",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=("Read the selected workgroup size dimension. A launch configuration contract can make this an exact fact; otherwise target facts bound the dynamic launch value."),
    results=[Result("result", INDEX, doc="Workgroup size in the selected dimension.")],
    attrs=[
        AttrDef(
            "dimension",
            ATTR_TYPE_ENUM,
            enum_def=KernelDimension,
            doc="Coordinate axis to read.",
        ),
    ],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_workgroup_size_facts",
    format=[TemplateParam("dimension"), COLON, ResultType("result")],
    examples=["%size = kernel.workgroup.size<x> : index"],
)

kernel_workgroup_count = Op(
    name="kernel.workgroup.count",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Read the dispatched workgroup count in one grid dimension.",
    results=[Result("result", INDEX, doc="Dispatched workgroup count in the selected dimension.")],
    attrs=[
        AttrDef(
            "dimension",
            ATTR_TYPE_ENUM,
            enum_def=KernelDimension,
            doc="Coordinate axis to read.",
        ),
    ],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_workgroup_count_facts",
    format=[TemplateParam("dimension"), COLON, ResultType("result")],
    examples=["%count = kernel.workgroup.count<x> : index"],
)

kernel_workitem_dispatch_id = Op(
    name="kernel.workitem.dispatch.id",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Read one coordinate of the current invocation in the whole dispatch. "
        "This is the logical coordinate formed from workgroup id, workgroup "
        "size, and workitem id; target lowering may materialize it directly or "
        "derive it from lower-level launch registers."
    ),
    results=[Result("result", INDEX, doc="Current workitem coordinate in the selected dispatch dimension.")],
    attrs=[
        AttrDef(
            "dimension",
            ATTR_TYPE_ENUM,
            enum_def=KernelDimension,
            doc="Coordinate axis to read.",
        ),
    ],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_workitem_dispatch_id_facts",
    format=[TemplateParam("dimension"), COLON, ResultType("result")],
    examples=["%gid = kernel.workitem.dispatch.id<x> : index"],
)


# ============================================================================
# Kernel subgroup query ops
# ============================================================================

kernel_subgroup_id = Op(
    name="kernel.subgroup.id",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Read the current subgroup coordinate within the workgroup.",
    results=[Result("result", INDEX, doc="Current subgroup id within the workgroup.")],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_subgroup_id_facts",
    format=[COLON, ResultType("result")],
    examples=["%sg = kernel.subgroup.id : index"],
)

kernel_subgroup_count = Op(
    name="kernel.subgroup.count",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Read the number of subgroups in the current workgroup.",
    results=[Result("result", INDEX, doc="Subgroup count in the workgroup.")],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_subgroup_count_facts",
    format=[COLON, ResultType("result")],
    examples=["%count = kernel.subgroup.count : index"],
)

kernel_subgroup_size = Op(
    name="kernel.subgroup.size",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Read the invocation count of the current subgroup.",
    results=[Result("result", INDEX, doc="Current subgroup size.")],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_subgroup_size_facts",
    format=[COLON, ResultType("result")],
    examples=["%size = kernel.subgroup.size : index"],
)

kernel_subgroup_lane_id = Op(
    name="kernel.subgroup.lane.id",
    group=kernel_ops,
    phase=OpPhase.EXECUTABLE,
    doc="Read the current invocation coordinate within its subgroup.",
    results=[Result("result", INDEX, doc="Current subgroup lane id.")],
    traits=[PURE, HasAncestor("kernel.def")],
    facts="loom_kernel_subgroup_lane_id_facts",
    format=[COLON, ResultType("result")],
    examples=["%lane = kernel.subgroup.lane.id : index"],
)


# ============================================================================
# Kernel subgroup collectives
# ============================================================================

kernel_subgroup_shuffle = Op(
    name="kernel.subgroup.shuffle",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Move a scalar or rank-1 vector value across lanes of the current "
        "subgroup. The result value has the same type as the input value, and "
        "the valid result reports whether the named source lane participated."
    ),
    operands=[
        Operand("value", ANY, doc="Per-invocation value to move."),
        Operand("offset", INTEGER, doc="i32 lane offset or lane index interpreted by mode."),
        Operand("width", INTEGER, doc="i32 active subgroup width."),
    ],
    results=[
        Result("result", ANY, doc="Value read from the selected source lane."),
        Result("valid", I1, doc="True when the selected source lane is valid."),
    ],
    attrs=[
        AttrDef("mode", ATTR_TYPE_ENUM, enum_def=KernelSubgroupShuffleMode, doc="Lane addressing mode."),
    ],
    constraints=[SameType("value", "result")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_shuffle_verify",
    format=[
        TemplateParam("mode"),
        Ref("value"),
        COMMA,
        Ref("offset"),
        COMMA,
        Ref("width"),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("offset"),
        COMMA,
        TypeOf("width"),
    ],
    examples=["%r, %valid = kernel.subgroup.shuffle<xor> %v, %offset, %width : f32, i32, i32"],
)

kernel_subgroup_broadcast = Op(
    name="kernel.subgroup.broadcast",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Broadcast a scalar or rank-1 vector value from one named subgroup lane.",
    operands=[
        Operand("value", ANY, doc="Per-invocation source value."),
        Operand("lane", INTEGER, doc="i32 lane id to broadcast from."),
    ],
    results=[Result("result", ANY, doc="Broadcast value.")],
    constraints=[SameType("value", "result")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_broadcast_verify",
    format=[
        Ref("value"),
        kw("from"),
        Ref("lane"),
        COLON,
        TypeOf("value"),
        COMMA,
        TypeOf("lane"),
    ],
    examples=["%r = kernel.subgroup.broadcast %v from %lane : f32, i32"],
)

kernel_subgroup_broadcast_first = Op(
    name="kernel.subgroup.broadcast.first",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Broadcast a scalar or rank-1 vector value from the first active subgroup lane.",
    operands=[Operand("value", ANY, doc="Per-invocation source value.")],
    results=[Result("result", ANY, doc="Broadcast value.")],
    constraints=[SameType("value", "result")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_value_result_verify",
    format=[Ref("value"), COLON, TypeOf("value")],
    examples=["%r = kernel.subgroup.broadcast.first %v : f32"],
)


def _collective_combining_attrs() -> list[AttrDef]:
    return [AttrDef("kind", ATTR_TYPE_ENUM, enum_def=CombiningKind, doc="Combining operation.")]


def _clustered_combining_attrs() -> list[AttrDef]:
    return [
        *_collective_combining_attrs(),
        AttrDef("cluster_size", ATTR_TYPE_I64, optional=True, doc="Optional clustered subgroup size."),
        AttrDef("cluster_stride", ATTR_TYPE_I64, optional=True, doc="Optional clustered subgroup stride."),
    ]


kernel_subgroup_reduce = Op(
    name="kernel.subgroup.reduce",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Reduce a scalar or rank-1 vector value across the current subgroup.",
    operands=[Operand("value", ANY, doc="Per-invocation value to reduce.")],
    results=[Result("result", ANY, doc="Reduced value.")],
    attrs=_clustered_combining_attrs(),
    constraints=[SameType("value", "result")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_reduce_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        AttrDict(),
        COLON,
        TypeOf("value"),
    ],
    examples=["%sum = kernel.subgroup.reduce<addf> %v : f32"],
)

kernel_subgroup_scan = Op(
    name="kernel.subgroup.scan",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Prefix-scan a scalar or rank-1 vector value across the current subgroup.",
    operands=[Operand("value", ANY, doc="Per-invocation value to scan.")],
    results=[Result("result", ANY, doc="Scanned value.")],
    attrs=[
        *_clustered_combining_attrs(),
        AttrDef("mode", ATTR_TYPE_ENUM, enum_def=KernelScanMode, doc="Inclusive or exclusive scan."),
        AttrDef("direction", ATTR_TYPE_ENUM, enum_def=KernelScanDirection, doc="Lane order to scan."),
    ],
    constraints=[SameType("value", "result")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_scan_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        AttrDict(),
        COLON,
        TypeOf("value"),
    ],
    examples=["%prefix = kernel.subgroup.scan<addf> %v {mode = inclusive, direction = forward} : f32"],
)

kernel_subgroup_vote_any = Op(
    name="kernel.subgroup.vote.any",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Return true when any active subgroup lane has a true predicate.",
    operands=[Operand("predicate", I1, doc="Per-invocation predicate.")],
    results=[Result("result", I1, doc="Subgroup-uniform vote result.")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    facts="loom_kernel_subgroup_vote_any_facts",
    format=[Ref("predicate"), COLON, TypeOf("predicate")],
    examples=["%any = kernel.subgroup.vote.any %p : i1"],
)

kernel_subgroup_vote_all = Op(
    name="kernel.subgroup.vote.all",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Return true when all active subgroup lanes have a true predicate.",
    operands=[Operand("predicate", I1, doc="Per-invocation predicate.")],
    results=[Result("result", I1, doc="Subgroup-uniform vote result.")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    facts="loom_kernel_subgroup_vote_all_facts",
    format=[Ref("predicate"), COLON, TypeOf("predicate")],
    examples=["%all = kernel.subgroup.vote.all %p : i1"],
)

kernel_subgroup_vote_ballot = Op(
    name="kernel.subgroup.vote.ballot",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Return an integer mask of active subgroup lanes whose predicate is true.",
    operands=[Operand("predicate", I1, doc="Per-invocation predicate.")],
    results=[Result("mask", INTEGER, doc="Integer subgroup lane mask.")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_mask_result_verify",
    facts="loom_kernel_subgroup_vote_ballot_facts",
    format=[Ref("predicate"), COLON, TypeOf("predicate"), ARROW, ResultType("mask")],
    examples=["%mask = kernel.subgroup.vote.ballot %p : i1 -> i64"],
)

kernel_subgroup_active_mask = Op(
    name="kernel.subgroup.active.mask",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Return an integer mask of the currently active subgroup lanes.",
    results=[Result("mask", INTEGER, doc="Integer active-lane mask.")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_mask_result_verify",
    facts="loom_kernel_subgroup_active_mask_facts",
    format=[COLON, ResultType("mask")],
    examples=["%mask = kernel.subgroup.active.mask : i64"],
)

kernel_subgroup_match_any = Op(
    name="kernel.subgroup.match.any",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Return a lane mask of active subgroup invocations with the same scalar value.",
    operands=[Operand("value", SCALAR, doc="Value to compare across active subgroup lanes.")],
    results=[Result("mask", INTEGER, doc="Integer lane-equality mask.")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_mask_result_verify",
    facts="loom_kernel_subgroup_match_any_facts",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("mask")],
    examples=["%mask = kernel.subgroup.match.any %v : i32 -> i64"],
)

kernel_subgroup_match_all = Op(
    name="kernel.subgroup.match.all",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Return a lane mask and predicate describing whether all active subgroup lanes hold the same scalar value.",
    operands=[Operand("value", SCALAR, doc="Value to compare across active subgroup lanes.")],
    results=[
        Result("mask", INTEGER, doc="Integer lane-equality mask."),
        Result("all_equal", I1, doc="True when all active lanes match."),
    ],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_subgroup_match_all_verify",
    facts="loom_kernel_subgroup_match_all_facts",
    format=[
        Ref("value"),
        COLON,
        TypeOf("value"),
        ARROW,
        ResultTypeList("mask", parens=False),
    ],
    examples=["%mask, %all = kernel.subgroup.match.all %v : i32 -> i64, i1"],
)


# ============================================================================
# Kernel workgroup collectives
# ============================================================================

kernel_workgroup_reduce = Op(
    name="kernel.workgroup.reduce",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Reduce a scalar or rank-1 vector value across the current workgroup.",
    operands=[Operand("value", ANY, doc="Per-invocation value to reduce.")],
    results=[Result("result", ANY, doc="Reduced value.")],
    attrs=_collective_combining_attrs(),
    constraints=[SameType("value", "result")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_workgroup_reduce_verify",
    format=[TemplateParam("kind"), Ref("value"), COLON, TypeOf("value")],
    examples=["%sum = kernel.workgroup.reduce<addf> %v : f32"],
)

kernel_workgroup_scan = Op(
    name="kernel.workgroup.scan",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Prefix-scan a scalar or rank-1 vector value across the current workgroup.",
    operands=[Operand("value", ANY, doc="Per-invocation value to scan.")],
    results=[Result("result", ANY, doc="Scanned value.")],
    attrs=[
        *_collective_combining_attrs(),
        AttrDef("mode", ATTR_TYPE_ENUM, enum_def=KernelWorkgroupScanMode, doc="Inclusive or exclusive scan."),
        AttrDef("direction", ATTR_TYPE_ENUM, enum_def=KernelWorkgroupScanDirection, doc="Workitem order to scan."),
    ],
    constraints=[SameType("value", "result")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_workgroup_scan_verify",
    format=[
        TemplateParam("kind"),
        Ref("value"),
        AttrDict(),
        COLON,
        TypeOf("value"),
    ],
    examples=["%prefix = kernel.workgroup.scan<addf> %v {mode = inclusive, direction = forward} : f32"],
)

kernel_workgroup_vote_any = Op(
    name="kernel.workgroup.vote.any",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Return true when any workgroup invocation has a true predicate.",
    operands=[Operand("predicate", I1, doc="Per-invocation predicate.")],
    results=[Result("result", I1, doc="Workgroup-uniform vote result.")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    format=[Ref("predicate"), COLON, TypeOf("predicate")],
    examples=["%any = kernel.workgroup.vote.any %p : i1"],
)

kernel_workgroup_vote_all = Op(
    name="kernel.workgroup.vote.all",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Return true when all workgroup invocations have a true predicate.",
    operands=[Operand("predicate", I1, doc="Per-invocation predicate.")],
    results=[Result("result", I1, doc="Workgroup-uniform vote result.")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    format=[Ref("predicate"), COLON, TypeOf("predicate")],
    examples=["%all = kernel.workgroup.vote.all %p : i1"],
)

kernel_workgroup_vote_count = Op(
    name="kernel.workgroup.vote.count",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    phase=OpPhase.EXECUTABLE,
    doc="Count workgroup invocations with a true predicate.",
    operands=[Operand("predicate", I1, doc="Per-invocation predicate.")],
    results=[Result("result", INTEGER, doc="Integer true-predicate count.")],
    traits=_KERNEL_CONVERGENT_TRAITS,
    verify="loom_kernel_workgroup_vote_count_verify",
    format=[Ref("predicate"), COLON, TypeOf("predicate"), ARROW, ResultType("result")],
    examples=["%count = kernel.workgroup.vote.count %p : i1 -> i32"],
)


# ============================================================================
# kernel.tensor.lds.descriptor — AMDGPU tensor-memory dgroup bundle
# ============================================================================

kernel_tensor_lds_descriptor = Op(
    name="kernel.tensor.lds.descriptor",
    group=kernel_ops,
    contracts=[ContractFamily.TENSOR_MEMORY],
    doc=(
        "Bundle AMDGPU tensor-memory descriptor groups into one typed SSA "
        "value. The dgroups are the exact operands lowered to "
        "llvm.amdgcn.tensor.load.to.lds or "
        "llvm.amdgcn.tensor.store.from.lds: D0 is vector<4xi32>, D1 is "
        "vector<8xi32>, and optional D2/D3 are vector<4xi32>. Gfx1250 uses "
        "two-group and four-group descriptor forms; the LLVM intrinsic's fifth "
        "D4 operand is lowered as zero because gfx1250 ignores it. The op is "
        "pure and contains no memory endpoints; endpoint views are operands of "
        "the async tensor ops so fact propagation and alias analysis do not "
        "need to decode hardware bitfields."
    ),
    operands=[
        Operand("dgroups", VECTOR, variadic=True, doc="AMDGPU tensor-memory descriptor groups D0..D3."),
    ],
    results=[
        Result("descriptor", ANY, doc="Typed tensor LDS descriptor value."),
    ],
    verify="loom_kernel_tensor_lds_descriptor_verify",
    format=[
        kw("dgroups"),
        GLUE,
        LPAREN,
        Refs("dgroups"),
        RPAREN,
        COLON,
        TypesOf("dgroups"),
        ARROW,
        ResultType("descriptor"),
    ],
    examples=[
        "%desc = kernel.tensor.lds.descriptor dgroups(%d0, %d1) : vector<4xi32>, vector<8xi32> -> kernel.tensor.lds.descriptor",
        "%desc = kernel.tensor.lds.descriptor dgroups(%d0, %d1, %d2, %d3) : vector<4xi32>, vector<8xi32>, vector<4xi32>, vector<4xi32> -> kernel.tensor.lds.descriptor",
    ],
)

# ============================================================================
# kernel.barrier — scoped execution barrier with an explicit memory fence
# ============================================================================

kernel_barrier = Op(
    name="kernel.barrier",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_SYNCHRONIZATION],
    doc=(
        "Synchronize invocations in an explicit execution scope and fence a "
        "named memory space with a required ordering. Supported source-level "
        "kernel barriers synchronize either the current subgroup or workgroup "
        "while fencing workgroup memory with acquire-release ordering. "
        "Async-copy completion is modeled by kernel.async.wait; use "
        "kernel.barrier only when invocations must rendezvous before "
        "consuming shared memory."
    ),
    attrs=[
        AttrDef(
            "memory_space",
            ATTR_TYPE_ENUM,
            enum_def=KernelMemorySpace,
            doc="Memory space whose accesses are fenced by the barrier.",
        ),
        AttrDef(
            "ordering",
            ATTR_TYPE_ENUM,
            enum_def=KernelOrdering,
            doc="Memory ordering applied to fenced accesses.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=KernelScope,
            doc="Execution scope synchronized by the barrier.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS, CONVERGENT],
    verify="loom_kernel_barrier_verify",
    format=[TemplateParam("memory_space"), AttrDict()],
    examples=[
        "kernel.barrier<workgroup> {ordering = acq_rel, scope = subgroup}",
        "kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}",
    ],
)

# ============================================================================
# kernel.async.copy — initiate a view-to-view asynchronous copy
# ============================================================================


def _async_copy_attrs() -> list[AttrDef]:
    return [
        *_async_cache_attrs(),
        AttrDef(
            "direction",
            ATTR_TYPE_ENUM,
            enum_def=KernelAsyncDirection,
            doc="Required memory-space direction for the transfer.",
        ),
    ]


kernel_async_copy = Op(
    name="kernel.async.copy",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Initiate an asynchronous byte-for-byte transfer between two already "
        "originated views. The source and destination view types may use "
        "different logical element types or shapes, but they must describe the "
        "same static byte footprint. The direction attribute makes the "
        "required memory-space flow explicit. The returned token must be "
        "committed to exactly one kernel.async.group before the copied bytes "
        "are waited or consumed."
    ),
    operands=[
        Operand("source", VIEW, doc="Typed source view whose base is copied from."),
        Operand("dest", VIEW, doc="Typed destination view whose base is copied to."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the initiated copy."),
    ],
    attrs=_async_copy_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_copy_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.copy %src to %dst {cache_scope = cu, cache_temporal = regular, direction = global_to_workgroup} : view<16xi8> to view<16xi8> -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.copy.mask — predicated view-to-view asynchronous copy
# ============================================================================

kernel_async_copy_mask = Op(
    name="kernel.async.copy.mask",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Predicated form of kernel.async.copy. When predicate is true, the op "
        "initiates the same transfer as kernel.async.copy. When predicate is "
        "false, the op performs no memory access and produces an already "
        "complete token so grouping and waiting remain structurally uniform."
    ),
    operands=[
        Operand("source", VIEW, doc="Typed source view whose base is copied from."),
        Operand("dest", VIEW, doc="Typed destination view whose base is copied to."),
        Operand("predicate", I1, doc="Scalar predicate controlling this invocation's copy."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the predicated copy."),
    ],
    attrs=_async_copy_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_copy_mask_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        COMMA,
        Ref("predicate"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("predicate"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.copy.mask %src to %dst, %in_bounds {cache_scope = cu, cache_temporal = non_temporal, direction = global_to_workgroup} : view<16xi8> to view<16xi8>, i1 -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.gather — subgroup gather into a workgroup destination tile
# ============================================================================

kernel_async_gather = Op(
    name="kernel.async.gather",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Initiate a subgroup-collective asynchronous gather from each "
        "invocation's source view into a lane-contiguous workgroup destination "
        "view. The destination view has one leading subgroup-lane axis and "
        "a trailing lane slot with enough static bytes to hold one source "
        "payload. If the lane slot is larger than the source footprint, the "
        "extra destination bytes are padding bytes with unspecified contents. "
        "The destination denotes the subgroup-uniform base tile; the current "
        "subgroup lane is applied by the op semantics and must not be "
        "pre-applied by forming a lane subview. This directly represents AMDGPU "
        "global_load_lds-style staging, including padded narrow loads, without "
        "requiring a later pass to rediscover that a set of per-lane copies was "
        "really one subgroup LDS DMA operation."
    ),
    operands=[
        Operand("source", VIEW, doc="Per-invocation global-like source fragment."),
        Operand("dest", VIEW, doc="Subgroup-uniform workgroup destination tile with a leading subgroup-lane axis."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the subgroup gather."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_gather_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.gather %src_lane to %lds_tile {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<[%wave]x4xi8> -> kernel.async.token",
        "%copy = kernel.async.gather %src_lane to %lds_tile {cache_scope = cu, cache_temporal = regular} : view<12xi8> to view<64x16xi8> -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.gather.mask — predicated subgroup gather
# ============================================================================

kernel_async_gather_mask = Op(
    name="kernel.async.gather.mask",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Predicated form of kernel.async.gather. False predicates perform no "
        "source or destination access for the current invocation but still "
        "produce a completed token, preserving a uniform async group shape for "
        "tails and guarded tiles."
    ),
    operands=[
        Operand("source", VIEW, doc="Per-invocation global-like source fragment."),
        Operand("dest", VIEW, doc="Subgroup-uniform workgroup destination tile with a leading subgroup-lane axis."),
        Operand("predicate", I1, doc="Scalar predicate controlling this invocation's gather."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the predicated gather."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_gather_mask_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        COMMA,
        Ref("predicate"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("predicate"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.gather.mask %src_lane to %lds_tile, %in_bounds {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<[%wave]x4xi8>, i1 -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.cluster.gather — AMDGPU cluster broadcast into LDS
# ============================================================================

kernel_async_cluster_gather = Op(
    name="kernel.async.cluster.gather",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Initiate an AMDGPU gfx1250+ cluster asynchronous load from a "
        "global-like source view into a workgroup/LDS destination view. The "
        "required i32 cluster_mask is the hardware workgroup broadcast mask "
        "loaded through M0. Source and destination must have the same static "
        "byte footprint, and that footprint must be exactly 1, 4, 8, or 16 "
        "bytes; target lowering maps those widths to "
        "llvm.amdgcn.cluster.load.async.to.lds.b8/b32/b64/b128. The LLVM "
        "offset and cache-policy immediate operands are lowering choices "
        "derived from the view address and cache attributes, not separate Loom "
        "semantics. The returned token must be committed to exactly one "
        "kernel.async.group."
    ),
    operands=[
        Operand("source", VIEW, doc="Global-like source fragment broadcast across the cluster."),
        Operand("dest", VIEW, doc="Workgroup/LDS destination fragment for this workgroup."),
        Operand("cluster_mask", INTEGER, doc="i32 workgroup-cluster broadcast mask consumed by the target M0 operand."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the cluster gather."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_cluster_gather_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        kw("using"),
        Ref("cluster_mask"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("cluster_mask"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.cluster.gather %src to %lds using %mask {cache_scope = se, cache_temporal = high_temporal} : view<16xi8> to view<16xi8>, i32 -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.cluster.gather.mask — predicated AMDGPU cluster gather
# ============================================================================

kernel_async_cluster_gather_mask = Op(
    name="kernel.async.cluster.gather.mask",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Predicated form of kernel.async.cluster.gather. False predicates "
        "perform no source or destination access for the current invocation "
        "but still produce a completed token, preserving a uniform async group "
        "shape for tails and guarded tiles. The cluster_mask remains the "
        "target workgroup broadcast mask and is distinct from the scalar i1 "
        "predicate."
    ),
    operands=[
        Operand("source", VIEW, doc="Global-like source fragment broadcast across the cluster."),
        Operand("dest", VIEW, doc="Workgroup/LDS destination fragment for this workgroup."),
        Operand("cluster_mask", INTEGER, doc="i32 workgroup-cluster broadcast mask consumed by the target M0 operand."),
        Operand("predicate", I1, doc="Scalar predicate controlling this invocation's cluster gather."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the predicated cluster gather."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_cluster_gather_mask_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        kw("using"),
        Ref("cluster_mask"),
        COMMA,
        Ref("predicate"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("cluster_mask"),
        COMMA,
        TypeOf("predicate"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.cluster.gather.mask %src to %lds using %mask, %in_bounds {cache_scope = cu, cache_temporal = regular} : view<4xi8> to view<4xi8>, i32, i1 -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.tensor.load.to.lds — AMDGPU tensor-memory global-to-LDS transfer
# ============================================================================

kernel_async_tensor_load_to_lds = Op(
    name="kernel.async.tensor.load.to.lds",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC, ContractFamily.TENSOR_MEMORY],
    doc=(
        "Initiate an AMDGPU gfx1250+ tensor-memory load from a global-like "
        "source view into a workgroup/LDS destination view using an explicit "
        "kernel.tensor.lds.descriptor. The descriptor supplies the exact "
        "hardware dgroups, while the source and destination views keep the "
        "logical rank, element type, layout, and memory-space facts visible. "
        "The endpoints must have the same rank in [1, 5], the same 1/2/4/8 "
        "byte element type, and memory spaces global/constant/descriptor to "
        "workgroup. "
        "The returned token must be committed to exactly one kernel.async.group."
    ),
    operands=[
        Operand("source", VIEW, doc="Global-like tensor-memory source view."),
        Operand("dest", VIEW, doc="Workgroup/LDS tensor-memory destination view."),
        Operand("descriptor", ANY, doc="Tensor LDS descriptor supplying AMDGPU dgroups."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the tensor load."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_tensor_load_to_lds_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        kw("using"),
        Ref("descriptor"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("descriptor"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.tensor.load.to.lds %global_tile to %lds_tile using %desc {cache_scope = cu, cache_temporal = regular} : view<64x64xf32> to view<64x64xf32>, kernel.tensor.lds.descriptor -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.tensor.store.from.lds — AMDGPU tensor-memory LDS-to-global transfer
# ============================================================================

kernel_async_tensor_store_from_lds = Op(
    name="kernel.async.tensor.store.from.lds",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC, ContractFamily.TENSOR_MEMORY],
    doc=(
        "Initiate an AMDGPU gfx1250+ tensor-memory store from a workgroup/LDS "
        "source view into a global-like destination view using an explicit "
        "kernel.tensor.lds.descriptor. The descriptor supplies the exact "
        "hardware dgroups, while the source and destination views keep the "
        "logical rank, element type, layout, and memory-space facts visible. "
        "The endpoints must have the same rank in [1, 5], the same 1/2/4/8 "
        "byte element type, and memory spaces workgroup to global/descriptor. "
        "The "
        "returned token must be committed to exactly one kernel.async.group."
    ),
    operands=[
        Operand("source", VIEW, doc="Workgroup/LDS tensor-memory source view."),
        Operand("dest", VIEW, doc="Global-like tensor-memory destination view."),
        Operand("descriptor", ANY, doc="Tensor LDS descriptor supplying AMDGPU dgroups."),
    ],
    results=[
        Result("token", ANY, doc="Opaque async-copy token for the tensor store."),
    ],
    attrs=_async_cache_attrs(),
    effects=[Reads("source"), Writes("dest")],
    verify="loom_kernel_async_tensor_store_from_lds_verify",
    format=[
        Ref("source"),
        kw("to"),
        Ref("dest"),
        kw("using"),
        Ref("descriptor"),
        AttrDict(),
        COLON,
        TypeOf("source"),
        kw("to"),
        TypeOf("dest"),
        COMMA,
        TypeOf("descriptor"),
        ARROW,
        ResultType("token"),
    ],
    examples=[
        "%copy = kernel.async.tensor.store.from.lds %lds_tile to %global_tile using %desc {cache_scope = device, cache_temporal = non_temporal_writeback} : view<64x64xf32> to view<64x64xf32>, kernel.tensor.lds.descriptor -> kernel.async.token",
    ],
)

# ============================================================================
# kernel.async.group — commit async-copy tokens into one ordered group
# ============================================================================

kernel_async_group = Op(
    name="kernel.async.group",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Commit zero or more async copy/gather/cluster/tensor tokens into the "
        "ordered async stream. Empty groups are valid pipeline markers. The "
        "resulting group completes after all committed transfers complete. "
        "Groups are ordered by program order; waiting a group also completes "
        "older groups in the same stream."
    ),
    operands=[
        Operand(
            "tokens",
            ANY,
            variadic=True,
            doc="Async copy, gather, cluster-gather, or tensor-memory tokens to commit into this group.",
        ),
    ],
    results=[
        Result("group", ANY, doc="Opaque async group token to wait."),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_kernel_async_group_verify",
    format=[
        OptionalGroup(
            [Refs("tokens"), COLON, TypesOf("tokens")],
            anchor="tokens",
        ),
        ARROW,
        ResultType("group"),
    ],
    examples=[
        "%empty = kernel.async.group -> kernel.async.group",
        "%group = kernel.async.group %copy0, %copy1 : kernel.async.token, kernel.async.token -> kernel.async.group",
    ],
)

# ============================================================================
# kernel.async.wait — wait for an ordered async-copy group
# ============================================================================

kernel_async_wait = Op(
    name="kernel.async.wait",
    group=kernel_ops,
    contracts=[ContractFamily.KERNEL_ASYNC],
    doc=(
        "Wait until a committed async-copy group has completed. This completes "
        "the named group and all older groups in the same ordered async stream. "
        "The required newer_groups value states how many younger groups are "
        "allowed to remain outstanding after the wait, matching AMDGPU "
        "wait.asyncmark and NVVM wait_group count semantics without making "
        "lowering rediscover the software-pipeline distance from scratch. "
        "It is not a workgroup barrier; use kernel.barrier separately when "
        "other invocations must observe the copied destination data."
    ),
    operands=[
        Operand("group", ANY, doc="Async group token to wait."),
    ],
    attrs=[
        AttrDef(
            "newer_groups",
            ATTR_TYPE_I64,
            doc="Maximum number of younger async groups allowed to remain outstanding.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_kernel_async_wait_verify",
    format=[
        Ref("group"),
        AttrDict(),
        COLON,
        TypeOf("group"),
    ],
    examples=[
        "kernel.async.wait %group {newer_groups = 0} : kernel.async.group",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_KERNEL_TYPES: tuple[TypeDef, ...] = (
    kernel_async_group_type,
    kernel_async_token_type,
    kernel_tensor_lds_descriptor_type,
)

ALL_KERNEL_OPS: tuple[Op, ...] = (
    kernel_def,
    kernel_launch_config,
    kernel_return,
    kernel_exit,
    kernel_barrier,
    kernel_async_copy,
    kernel_async_copy_mask,
    kernel_async_gather,
    kernel_async_gather_mask,
    kernel_async_group,
    kernel_async_wait,
    kernel_tensor_lds_descriptor,
    kernel_async_tensor_load_to_lds,
    kernel_async_tensor_store_from_lds,
    kernel_async_cluster_gather,
    kernel_async_cluster_gather_mask,
    kernel_workitem_id,
    kernel_workgroup_id,
    kernel_workgroup_size,
    kernel_workgroup_count,
    kernel_workitem_dispatch_id,
    kernel_subgroup_id,
    kernel_subgroup_count,
    kernel_subgroup_size,
    kernel_subgroup_lane_id,
    kernel_subgroup_shuffle,
    kernel_subgroup_broadcast,
    kernel_subgroup_broadcast_first,
    kernel_subgroup_reduce,
    kernel_subgroup_scan,
    kernel_subgroup_vote_any,
    kernel_subgroup_vote_all,
    kernel_subgroup_vote_ballot,
    kernel_subgroup_active_mask,
    kernel_subgroup_match_any,
    kernel_subgroup_match_all,
    kernel_workgroup_reduce,
    kernel_workgroup_scan,
    kernel_workgroup_vote_any,
    kernel_workgroup_vote_all,
    kernel_workgroup_vote_count,
    kernel_assert,
)
