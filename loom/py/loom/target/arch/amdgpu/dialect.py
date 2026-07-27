# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU target-family record dialect."""

from loom.assembly import AttrDict, SymbolRef, TemplateParam
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ATTR_TYPE_STRING,
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
    doc="AMDGPU target-family records.",
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

amdgpu_target = Op(
    "amdgpu.target",
    group=amdgpu_ops,
    doc=(
        "AMDGPU target-family record. The selector chooses a generated "
        "processor/family row; optional attrs structurally override authored "
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
        AttrDef("processor", ATTR_TYPE_STRING, optional=True),
    ],
    verify="loom_amdgpu_target_record_verify",
    format=[
        TemplateParam("kind"),
        SymbolRef("symbol"),
        AttrDict(),
    ],
    examples=[
        "amdgpu.target<gfx1100> @gfx11",
        "amdgpu.target<gfx942> @gfx942 {subgroup_size = 64}",
        "amdgpu.target<gfx950> @gfx950 {subgroup_size = 64}",
    ],
)

ALL_AMDGPU_OPS: tuple[Op, ...] = (amdgpu_target,)
