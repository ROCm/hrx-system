# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P target records and source state."""

from loom.assembly import COLON, COMMA, AttrDict, ResultType, SymbolRef, TemplateParam
from loom.dialect.target import target_record_attrs
from loom.dsl import (
    ANY,
    ATTR_TYPE_STRING,
    PURE,
    SYMBOL_DEFINE,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    HasParent,
    Op,
    OpPhase,
    Result,
    SymbolDefinition,
    TargetLikeInterface,
    TypeDef,
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
        EnumCase(
            "configuration",
            3,
            doc=(
                "Physical array initialization and invocation commands for "
                "complete core programs."
            ),
        ),
    ],
    doc="AIE2P target row selected by aie2p.target.",
    c_type="loom_aie2p_target_kind_t",
    c_const_prefix="LOOM_AIE2P_TARGET_KIND",
    c_include="loom/target/arch/amd/xdna/aie2p/records/target_records.h",
)

aie2p_load_fifo_type = TypeDef(
    "aie2p.load_fifo",
    doc=(
        "One architectural AIE2P load-FIFO state chain. Its entry payload is "
        "unspecified; a FIFO fill establishes the bytes observed by later pops."
    ),
)

aie2p_load_fifo_state = Op(
    "aie2p.load_fifo.state",
    group=aie2p_ops,
    phase=OpPhase.EXECUTABLE,
    doc=(
        "Read the two architectural load-FIFO states at AIE core-function "
        "entry. The results are explicit state chains carried through source "
        "control flow and target-Low helper calls."
    ),
    results=[
        Result("fifo0", ANY, doc="Architectural LF0 state chain."),
        Result("fifo1", ANY, doc="Architectural LF1 state chain."),
    ],
    traits=[PURE, HasParent("func.def")],
    verify="loom_aie2p_load_fifo_state_verify",
    format=[
        COLON,
        ResultType("fifo0"),
        COMMA,
        ResultType("fifo1"),
    ],
    examples=[
        "%fifo0, %fifo1 = aie2p.load_fifo.state : aie2p.load_fifo, aie2p.load_fifo",
    ],
)

aie2p_target = Op(
    "aie2p.target",
    group=aie2p_ops,
    doc=(
        "AMD XDNA AIE2P target record. The selector chooses the owned "
        "core ISA, logical-array, or physical configuration program contract."
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
    ],
    verify="loom_aie2p_target_record_verify",
    format=[TemplateParam("kind"), SymbolRef("symbol"), AttrDict()],
    examples=["aie2p.target<core> @tile", "aie2p.target<array> @array"],
)

ALL_AIE2P_OPS: tuple[Op, ...] = (aie2p_target, aie2p_load_fifo_state)

ALL_AIE2P_TYPES: tuple[TypeDef, ...] = (aie2p_load_fifo_type,)
