# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stable resident-array Low contract for AMD XDNA AIE2P programs."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Descriptor,
    DescriptorFlag,
    DescriptorOpKind,
    DescriptorSet,
    Effect,
    EffectKind,
    EnumDomain,
    EnumValue,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandRole,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    SpillSlotSpace,
)

_TARGET_KEY = "amd.xdna.aie2p"
_DESCRIPTOR_SET_KEY = f"{_TARGET_KEY}.array"

_REG_SCALAR = "aie2p.array.scalar"
_REG_BINDING = "aie2p.array.binding"
_REG_GROUP = "aie2p.array.group"
_REG_WORKER = "aie2p.array.worker"
_REG_SENDER = "aie2p.array.sender"
_REG_RECEIVER = "aie2p.array.receiver"
_REG_CHANNEL = "aie2p.array.channel"

_RESOURCE_GRAPH = "aie2p.array.graph"
_SCHEDULE_GRAPH = "aie2p.array.schedule.graph"
_BINDING_ACCESS_DOMAIN = "aie2p.array.binding_access"

_REFERENCE_BANK = 1


def _reference_reg_class(name: str) -> RegClass:
    return RegClass(
        name,
        32,
        SpillSlotSpace.PRIVATE,
        flags=(
            RegClassFlag.VIRTUAL_ONLY,
            RegClassFlag.REFERENCE,
            RegClassFlag.UNSPILLABLE,
        ),
        target_bank_id=_REFERENCE_BANK,
        alias_set_id=_REFERENCE_BANK,
    )


def _result(reg_class: str, field_name: str = "result") -> Operand:
    return Operand(
        field_name,
        OperandRole.RESULT,
        (RegClassAlt(reg_class),),
    )


def _operand(
    reg_classes: str | tuple[str, ...],
    field_name: str,
    *,
    variadic: bool = False,
) -> Operand:
    if isinstance(reg_classes, str):
        reg_classes = (reg_classes,)
    return Operand(
        field_name,
        OperandRole.OPERAND,
        tuple(RegClassAlt(reg_class) for reg_class in reg_classes),
        flags=(OperandFlag.VARIADIC,) if variadic else (),
    )


def _u32(field_name: str) -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=32,
        unsigned_max=(2**32) - 1,
    )


def _asm(
    mnemonic: str,
    *,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            mnemonic=mnemonic,
            results=results,
            operands=operands,
            immediates=tuple(AsmImmediate(name) for name in immediates),
        ),
    )


# Array declarations materialize persistent device resources outside the tile
# instruction stream. CALL is the existing conservative Low effect for such
# target-owned state until the graph is consumed by the array planner.
_TOPOLOGY_EFFECT = Effect(EffectKind.CALL)

_DESCRIPTORS = (
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.constant.u32",
        mnemonic="constant.u32",
        semantic_tag="array.constant.u32",
        operands=(_result(_REG_SCALAR),),
        op_kind=DescriptorOpKind.CONST,
        immediates=(_u32("value"),),
        asm_forms=_asm(
            "constant.u32",
            results=("result",),
            immediates=("value",),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.binding",
        mnemonic="binding",
        semantic_tag="array.binding",
        operands=(_result(_REG_BINDING),),
        immediates=(
            _u32("ordinal"),
            Immediate(
                "access",
                ImmediateKind.ENUM,
                enum_domain=_BINDING_ACCESS_DOMAIN,
            ),
        ),
        asm_forms=_asm(
            "binding",
            results=("result",),
            immediates=("ordinal", "access"),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.group",
        mnemonic="group",
        semantic_tag="array.worker_group",
        operands=(
            _result(_REG_GROUP),
            _operand(_REG_SCALAR, "lanes"),
        ),
        asm_forms=_asm(
            "group",
            results=("result",),
            operands=("lanes",),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.worker",
        mnemonic="worker",
        semantic_tag="array.resident_worker",
        operands=(
            _result(_REG_WORKER),
            _operand(_REG_GROUP, "group"),
            _operand(_REG_SCALAR, "lane"),
        ),
        immediates=(
            Immediate(
                "entry",
                ImmediateKind.ORDINAL,
                flags=(ImmediateFlag.SYMBOLIC,),
                bit_width=32,
                unsigned_max=(2**32) - 1,
            ),
        ),
        asm_forms=_asm(
            "worker",
            results=("result",),
            operands=("group", "lane"),
            immediates=("entry",),
        ),
        effects=(_TOPOLOGY_EFFECT,),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.sender",
        mnemonic="sender",
        semantic_tag="array.sender",
        operands=(
            _result(_REG_SENDER),
            _operand((_REG_BINDING, _REG_WORKER), "owner"),
        ),
        immediates=(_u32("port"),),
        asm_forms=_asm(
            "sender",
            results=("result",),
            operands=("owner",),
            immediates=("port",),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.receiver",
        mnemonic="receiver",
        semantic_tag="array.receiver",
        operands=(
            _result(_REG_RECEIVER),
            _operand((_REG_BINDING, _REG_WORKER), "owner"),
        ),
        immediates=(_u32("port"),),
        asm_forms=_asm(
            "receiver",
            results=("result",),
            operands=("owner",),
            immediates=("port",),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.partition",
        mnemonic="partition",
        semantic_tag="array.partition",
        operands=(
            _result(_REG_SENDER),
            _operand(_REG_SENDER, "source"),
            _operand(_REG_SCALAR, "lane"),
            _operand(_REG_SCALAR, "lanes"),
        ),
        asm_forms=_asm(
            "partition",
            results=("result",),
            operands=("source", "lane", "lanes"),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.channel",
        mnemonic="channel",
        semantic_tag="array.channel",
        operands=(
            _result(_REG_CHANNEL),
            _operand(_REG_SENDER, "sender"),
            _operand(_REG_RECEIVER, "receiver"),
            _operand(_REG_SCALAR, "capacity"),
        ),
        asm_forms=_asm(
            "channel",
            results=("result",),
            operands=("sender", "receiver", "capacity"),
        ),
        effects=(_TOPOLOGY_EFFECT,),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.constrain.location",
        mnemonic="constrain.location",
        semantic_tag="array.constraint.location",
        operands=(
            _operand(_REG_WORKER, "worker"),
            _operand(_REG_SCALAR, "column"),
            _operand(_REG_SCALAR, "row"),
        ),
        asm_forms=_asm(
            "constrain.location",
            operands=("worker", "column", "row"),
        ),
        effects=(_TOPOLOGY_EFFECT,),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
)


AIE2P_ARRAY_DESCRIPTOR_SET = DescriptorSet(
    key=_DESCRIPTOR_SET_KEY,
    target_key=_TARGET_KEY,
    feature_key=f"{_DESCRIPTOR_SET_KEY}.v2",
    c_header_path=Path(
        "loom/src/loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"
    ),
    c_source_path=Path(
        "loom/src/loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.c"
    ),
    header_guard=("LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_ARRAY_DESCRIPTORS_H_"),
    public_header=("loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"),
    function_name="loom_aie2p_array_descriptor_set",
    c_table_prefix="Aie2pArray",
    c_enum_prefix="AIE2P_ARRAY",
    generator_version=2,
    reg_classes=(
        _reference_reg_class(_REG_SCALAR),
        _reference_reg_class(_REG_BINDING),
        _reference_reg_class(_REG_GROUP),
        _reference_reg_class(_REG_WORKER),
        _reference_reg_class(_REG_SENDER),
        _reference_reg_class(_REG_RECEIVER),
        _reference_reg_class(_REG_CHANNEL),
    ),
    resources=(
        Resource(
            _RESOURCE_GRAPH,
            capacity_per_cycle=1,
            kind=ResourceKind.CONTROL,
        ),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_GRAPH,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_GRAPH, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
    ),
    enum_domains=(
        EnumDomain(
            _BINDING_ACCESS_DOMAIN,
            values=(
                EnumValue("read", 1),
                EnumValue("write", 2),
                EnumValue("read_write", 3),
            ),
        ),
    ),
    descriptors=_DESCRIPTORS,
    requires_explicit_asm_surface=True,
)
