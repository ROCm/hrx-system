# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU target-family record dialect."""

from build_tools.amdgpu.target_map_data import AMDGPU_TARGET_ID_FEATURE_ORDER

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    AttrDict,
    Ref,
    ResultType,
    SymbolRef,
    TemplateParam,
    TypeOf,
)
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ATTR_TYPE_I64,
    ATTR_TYPE_SIGNED_ENUM_SET,
    PURE,
    REGISTER,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    OpPhase,
    Result,
    SameRegisterClass,
    SameType,
    SymbolDefinition,
    TargetLikeInterface,
)
from loom.target.arch.amdgpu.target_info import sorted_target_infos

amdgpu_ops = Dialect(
    "amdgpu",
    dialect_id=0x17,
    doc="AMDGPU processor target records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/arch/amdgpu/ops",
    register_by_default=False,
)

AmdgpuTargetKind = EnumDef(
    "AmdgpuTargetKind",
    [
        EnumCase(info.target, info.enum_value, doc=info.doc)
        for info in sorted_target_infos()
    ],
    doc="AMDGPU target row selected by amdgpu.target.",
)

AmdgpuTargetIdFeature = EnumDef(
    "AmdgpuTargetIdFeature",
    [
        EnumCase(
            feature,
            ordinal,
            doc=f"AMDHSA '{feature}' target-ID feature.",
        )
        for ordinal, feature in enumerate(AMDGPU_TARGET_ID_FEATURE_ORDER)
    ],
    doc="Configurable AMDHSA target-ID feature.",
)

amdgpu_target = Op(
    "amdgpu.target",
    group=amdgpu_ops,
    doc=(
        "AMDGPU target record. The selector chooses one exact, generic, or "
        "overlay target row; optional attrs preserve authored common facts "
        "and target-ID feature assertions."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_amdgpu_target_bundles",
            fact_type="loom_amdgpu_target_fact_type",
            fact_projector="loom_amdgpu_target_fact_projector",
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=[
        *target_record_attrs(AmdgpuTargetKind),
        AttrDef(
            "features",
            ATTR_TYPE_SIGNED_ENUM_SET,
            enum_def=AmdgpuTargetIdFeature,
            optional=True,
            doc=(
                "Explicit AMDHSA target-ID feature assertions. Bare members "
                "require enabled features and negative members require "
                "disabled features."
            ),
        ),
    ],
    verify="loom_amdgpu_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "amdgpu.target<gfx11-generic> @gfx11_generic",
        "amdgpu.target<gfx942> @gfx942 {subgroup_size = 64}",
        "amdgpu.target<gfx950> @gfx950 {subgroup_size = 64}",
        "amdgpu.target<gfx1250-a0> @gfx1250_a0",
        "amdgpu.target<gfx942> @gfx942_features {features = [sramecc, -xnack]}",
    ],
)

amdgpu_address_add_scaled_u32 = Op(
    "amdgpu.address.add_scaled_u32",
    group=amdgpu_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Add a scaled unsigned 32-bit scalar offset to a 64-bit scalar "
        "address. Scaling shifts in the 32-bit domain, the scaled offset is "
        "zero-extended, and the address addition wraps modulo 2^64."
    ),
    operands=[
        Operand(
            "base",
            REGISTER,
            doc="Base address in two consecutive AMDGPU scalar registers.",
        ),
        Operand(
            "offset",
            REGISTER,
            doc="Unsigned offset in one AMDGPU scalar register.",
        ),
    ],
    attrs=[
        AttrDef(
            "byte_shift",
            ATTR_TYPE_I64,
            doc="Left shift in the inclusive range [0, 31].",
        ),
    ],
    results=[
        Result(
            "result",
            REGISTER,
            doc="Scaled address in two consecutive AMDGPU scalar registers.",
        ),
    ],
    traits=[PURE],
    constraints=[
        SameType("base", "result"),
        SameRegisterClass("base", "offset", "result"),
    ],
    verify="loom_amdgpu_address_add_scaled_u32_verify",
    format=[
        Ref("base"),
        COMMA,
        Ref("offset"),
        AttrDict(),
        COLON,
        TypeOf("base"),
        COMMA,
        TypeOf("offset"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%address = amdgpu.address.add_scaled_u32 %base, %index "
        "{byte_shift = 2} : reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr> -> "
        "reg<amdgpu.sgpr x2>",
    ],
)

ALL_AMDGPU_OPS: tuple[Op, ...] = (
    amdgpu_target,
    amdgpu_address_add_scaled_u32,
)
