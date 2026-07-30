# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU target-family record dialect."""

from loom.assembly import AttrDict, SymbolRef, TemplateParam
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

AmdgpuTargetFeatureState = EnumDef(
    "AmdgpuTargetFeatureState",
    [
        EnumCase("any", 0, doc="Feature state is unconstrained."),
        EnumCase(
            "unsupported",
            1,
            doc="Feature is unsupported by the selected processor.",
        ),
        EnumCase("off", 2, doc="Feature is explicitly disabled."),
        EnumCase("on", 3, doc="Feature is explicitly enabled."),
    ],
    doc="Normalized state of one AMDHSA target-ID feature.",
    c_type="loom_amdgpu_target_feature_state_t",
    c_const_prefix="LOOM_AMDGPU_TARGET_FEATURE",
    c_include="loom/target/arch/amdgpu/target_info.h",
)

amdgpu_target = Op(
    "amdgpu.target",
    group=amdgpu_ops,
    doc=(
        "AMDGPU target record. The selector chooses one exact, generic, or "
        "overlay target row; optional attrs preserve authored common facts "
        "and target-ID feature states."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_amdgpu_target_bundles",
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
            "sramecc",
            ATTR_TYPE_ENUM,
            enum_def=AmdgpuTargetFeatureState,
            optional=True,
            doc="Required AMDHSA SRAM ECC target-ID feature state.",
        ),
        AttrDef(
            "xnack",
            ATTR_TYPE_ENUM,
            enum_def=AmdgpuTargetFeatureState,
            optional=True,
            doc="Required AMDHSA XNACK target-ID feature state.",
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
        "amdgpu.target<gfx942> @gfx942_xnack {xnack = on}",
    ],
)

ALL_AMDGPU_OPS: tuple[Op, ...] = (amdgpu_target,)
