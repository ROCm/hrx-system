# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V target-family record dialect."""

from loom.assembly import AttrDict, SymbolRef, TemplateParam
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    SYMBOL_DEFINE,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    OpPhase,
    SymbolDefinition,
    TargetLikeInterface,
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
