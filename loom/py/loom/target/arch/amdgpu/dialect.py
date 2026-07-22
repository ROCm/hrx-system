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
from loom.target.arch.amdgpu.target_info import sorted_target_record_infos

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
        EnumCase(info.processor, info.enum_value, doc=info.doc)
        for info in sorted_target_record_infos()
    ],
    doc="AMDGPU target row selected by amdgpu.target.",
)

AmdgpuGfx1250Revision = EnumDef(
    "AmdgpuGfx1250Revision",
    [
        EnumCase("a0", 1, doc="gfx1250 A0 silicon behavior and errata."),
        EnumCase("b0", 2, doc="gfx1250 B0 silicon behavior."),
    ],
    doc="gfx1250 silicon revision selected by amdgpu.target.",
    c_type="loom_amdgpu_gfx1250_revision_t",
    c_const_prefix="LOOM_AMDGPU_GFX1250_REVISION",
    c_include="loom/target/arch/amdgpu/target_info.h",
)

amdgpu_target = Op(
    "amdgpu.target",
    group=amdgpu_ops,
    doc=(
        "AMDGPU processor target record. The selector chooses one exact or "
        "generic processor row; optional attrs structurally override authored "
        "common target fields."
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
            "gfx1250_revision",
            ATTR_TYPE_ENUM,
            enum_def=AmdgpuGfx1250Revision,
            optional=True,
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
        "amdgpu.target<gfx1250> @gfx1250 {gfx1250_revision = a0}",
    ],
)

ALL_AMDGPU_OPS: tuple[Op, ...] = (amdgpu_target,)
