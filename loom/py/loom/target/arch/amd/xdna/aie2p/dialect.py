# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P target record dialect."""

from loom.assembly import AttrDict, SymbolRef, TemplateParam
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ATTR_TYPE_BOOL,
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
    [
        EnumCase("core", 1, doc="One AIE2P compute-tile core target row."),
        EnumCase("array", 2, doc="One AIE2P logical-array program target row."),
    ],
    doc="AIE2P target row selected by aie2p.target.",
)

aie2p_target = Op(
    "aie2p.target",
    group=aie2p_ops,
    doc=(
        "AMD XDNA AIE2P target record. The selector chooses either the owned "
        "core ISA contract or the logical-array program contract."
    ),
    traits=[SYMBOL_DEFINE],
    interfaces=[
        TargetLikeInterface(
            symbol="symbol",
            selector="kind",
            bundle_table="loom_aie2p_target_bundles",
            fact_type="loom_aie2p_target_fact_type",
            fact_projector="loom_aie2p_target_fact_projector",
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
        *target_record_attrs(Aie2pTargetKind),
        AttrDef(
            "device_profile",
            ATTR_TYPE_STRING,
            optional=True,
            doc="Exact physical XDNA deployment profile key.",
        ),
        AttrDef(
            "trace",
            ATTR_TYPE_BOOL,
            optional=True,
            doc=(
                "Opt in to hardware event trace for this array program's core "
                "tiles. Absent or false compiles without trace resources. "
                "Trace routes share the per-link stream-switch capacity pool "
                "with data routes, so a densely packed multi-worker array "
                "(e.g. 2 workers/column) can exhaust it; planning fails with "
                "RESOURCE_EXHAUSTED rather than silently dropping trace."
            ),
        ),
    ],
    verify="loom_aie2p_target_record_verify",
    format=[TemplateParam("kind"), SymbolRef("symbol"), AttrDict()],
    examples=[
        "aie2p.target<core> @tile",
        "aie2p.target<array> @array",
        "aie2p.target<array> @array_traced trace=true",
    ],
)

ALL_AIE2P_OPS: tuple[Op, ...] = (aie2p_target,)
