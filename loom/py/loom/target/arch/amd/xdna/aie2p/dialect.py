# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P target record dialect."""

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
    TargetFactSpecialization,
    TargetLikeInterface,
)

aie2p_ops = Dialect(
    "aie2p",
    dialect_id=0x21,
    doc="AMD XDNA AIE2P target records.",
    default_phase=OpPhase.MODULE_METADATA,
    c_path="target/arch/amd/xdna/aie2p/ops",
    register_by_default=False,
    checked_in_headers=False,
)

Aie2pTargetKind = EnumDef(
    "Aie2pTargetKind",
    [EnumCase("core", 1, doc="One AIE2P compute-tile core target row.")],
    doc="AIE2P target row selected by aie2p.target.",
)

aie2p_target = Op(
    "aie2p.target",
    group=aie2p_ops,
    doc=(
        "AMD XDNA AIE2P compute-tile target record. The selector chooses the "
        "owned core ISA, scheduling, allocation, and object-emission contract."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_aie2p_target_bundles",
            fact_specialization=TargetFactSpecialization.STRUCTURAL,
        )
    ],
    symbol_def=SymbolDefinition(
        field="symbol",
        name="target",
        interfaces=["target", "record"],
        bytecode_kind="LOOM_SYMBOL_RECORD",
        fact_domain="loom_target_symbol_fact_domain",
    ),
    attrs=target_record_attrs(Aie2pTargetKind),
    verify="loom_target_record_verify",
    format=[TemplateParam("kind"), SymbolRef("symbol"), AttrDict()],
    examples=["aie2p.target<core> @tile"],
)

ALL_AIE2P_OPS: tuple[Op, ...] = (aie2p_target,)
