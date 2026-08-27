# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V target-family records and semantic value types."""

from loom.assembly import COMMA, GLUE, AttrDict, Param, SymbolRef, TemplateParam, kw
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ATTR_TYPE_ENUM,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    OpPhase,
    SymbolDefinition,
    TargetLikeInterface,
    TypeDef,
)

spirv_ops = Dialect(
    "spirv",
    dialect_id=0x1B,
    doc="SPIR-V target-family records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/arch/spirv/ops",
    register_by_default=False,
)

SpirvTargetKind = EnumDef(
    "SpirvTargetKind",
    [
        EnumCase("vulkan1_3", 1, doc="Vulkan 1.3 logical SPIR-V module row."),
    ],
    doc="SPIR-V target row selected by spirv.target.",
    c_type="loom_spirv_target_kind_t",
    c_const_prefix="LOOM_SPIRV_TARGET_KIND",
    c_include="loom/target/arch/spirv/facts.h",
)

SpirvScalarType = EnumDef(
    "SpirvScalarType",
    [
        EnumCase("f16", 1, doc="IEEE binary16 floating-point component."),
        EnumCase("f32", 2, doc="IEEE binary32 floating-point component."),
        EnumCase("f64", 3, doc="IEEE binary64 floating-point component."),
        EnumCase("bf16", 4, doc="KHR bfloat16 floating-point component."),
        EnumCase("s8", 5, doc="Signed 8-bit integer component."),
        EnumCase("s16", 6, doc="Signed 16-bit integer component."),
        EnumCase("s32", 7, doc="Signed 32-bit integer component."),
        EnumCase("s64", 8, doc="Signed 64-bit integer component."),
        EnumCase("u8", 9, doc="Unsigned 8-bit integer component."),
        EnumCase("u16", 10, doc="Unsigned 16-bit integer component."),
        EnumCase("u32", 11, doc="Unsigned 32-bit integer component."),
        EnumCase("u64", 12, doc="Unsigned 64-bit integer component."),
    ],
    doc=(
        "SPIR-V semantic scalar type. Integer signedness is explicit because "
        "SPIR-V type declarations preserve a distinction absent from Loom's "
        "signless source integer types."
    ),
    c_type="loom_spirv_scalar_type_t",
    c_const_prefix="LOOM_SPIRV_SCALAR_TYPE",
    c_include="loom/target/arch/spirv/scalar_types.h",
)

SpirvScope = EnumDef(
    "SpirvScope",
    [
        EnumCase("cross_device", 0),
        EnumCase("device", 1),
        EnumCase("workgroup", 2),
        EnumCase("subgroup", 3),
        EnumCase("invocation", 4),
        EnumCase("queue_family", 5),
        EnumCase("shader_call", 6),
    ],
    doc="Execution scope encoded by a SPIR-V Scope operand.",
    c_type="loom_spirv_scope_t",
    c_const_prefix="LOOM_SPIRV_SCOPE",
    c_include="loom/target/arch/spirv/isa.h",
)

SpirvCooperativeMatrixUse = EnumDef(
    "SpirvCooperativeMatrixUse",
    [
        EnumCase("matrix_a", 0),
        EnumCase("matrix_b", 1),
        EnumCase("matrix_accumulator", 2),
    ],
    doc="Operand role encoded by a SPIR-V CooperativeMatrixUse operand.",
    c_type="loom_spirv_cooperative_matrix_use_t",
    c_const_prefix="LOOM_SPIRV_COOPERATIVE_MATRIX_USE",
    c_include="loom/target/arch/spirv/isa.h",
)

spirv_cooperative_matrix_type = TypeDef(
    "spirv.cooperative_matrix",
    params=[
        AttrDef("rows", "i64", doc="Static matrix row count."),
        AttrDef("columns", "i64", doc="Static matrix column count."),
        AttrDef(
            "component_type",
            ATTR_TYPE_ENUM,
            enum_def=SpirvScalarType,
            doc="Signedness-preserving SPIR-V component type.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=SpirvScope,
            doc="Execution scope shared by matrix invocations.",
        ),
        AttrDef(
            "use",
            ATTR_TYPE_ENUM,
            enum_def=SpirvCooperativeMatrixUse,
            doc="Matrix operand role.",
        ),
    ],
    format=[
        Param("rows"),
        GLUE,
        kw("x"),
        GLUE,
        Param("columns"),
        GLUE,
        kw("x"),
        GLUE,
        Param("component_type"),
        COMMA,
        Param("scope"),
        COMMA,
        Param("use"),
    ],
    doc=(
        "Exact semantic payload of an OpTypeCooperativeMatrixKHR value. The "
        "type travels inside a typed spirv.id register so control-flow and "
        "function boundaries retain every operand needed by SPIR-V emission."
    ),
)

spirv_target = Op(
    "spirv.target",
    group=spirv_ops,
    doc=(
        "SPIR-V target-family record. The selector chooses an authored SPIR-V "
        "row; optional attrs structurally override common target fields."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_spirv_target_bundles",
            fact_type="loom_spirv_target_fact_type",
            fact_projector="loom_spirv_target_fact_projector",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=target_record_attrs(SpirvTargetKind),
    verify="loom_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "spirv.target<vulkan1_3> @spv",
    ],
)

ALL_SPIRV_OPS: tuple[Op, ...] = (spirv_target,)

ALL_SPIRV_TYPES: tuple[TypeDef, ...] = (spirv_cooperative_matrix_type,)
