# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stable logical-array Low contract for AMD XDNA AIE2P programs."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    AsmOperandSegment,
    AsmOperandSegmentDelimiter,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    EnumDomain,
    EnumValue,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    MemorySpace,
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
    ScheduleClassFlag,
    SpillSlotSpace,
)

_TARGET_KEY = "amd.xdna.aie2p"
_DESCRIPTOR_SET_KEY = f"{_TARGET_KEY}.array"

_REG_BINDING = "aie2p.array.binding"
_REG_WORKER = "aie2p.array.worker"
_REG_BUFFER = "aie2p.array.buffer"
_REG_FLOW = "aie2p.array.flow"
_REG_TOKEN = "aie2p.array.token"

_RESOURCE_GRAPH = "aie2p.array.graph"
_RESOURCE_ASYNC = "aie2p.array.async"
_RESOURCE_SYNC = "aie2p.array.sync"

_SCHEDULE_GRAPH = "aie2p.array.schedule.graph"
_SCHEDULE_ASYNC = "aie2p.array.schedule.async"
_SCHEDULE_SYNC = "aie2p.array.schedule.sync"

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


def _u64(field_name: str) -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.UNSIGNED,
        bit_width=64,
        unsigned_max=(2**64) - 1,
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


_ORDERED_ASYNC_EFFECT = Effect(
    EffectKind.CALL,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_SYNC_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_DESCRIPTORS = (
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
        key=f"{_DESCRIPTOR_SET_KEY}.worker",
        mnemonic="worker",
        semantic_tag="array.worker",
        operands=(_result(_REG_WORKER),),
        immediates=(
            _u32("ordinal"),
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
            immediates=("ordinal", "entry"),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.buffer",
        mnemonic="buffer",
        semantic_tag="array.logical_buffer",
        operands=(
            _result(_REG_BUFFER),
            _operand(_REG_WORKER, "owner"),
        ),
        immediates=(
            _u32("ordinal"),
            _u64("byte_length"),
            _u32("byte_alignment"),
        ),
        asm_forms=_asm(
            "buffer",
            results=("result",),
            operands=("owner",),
            immediates=("ordinal", "byte_length", "byte_alignment"),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.flow",
        mnemonic="flow",
        semantic_tag="array.flow",
        operands=(
            _result(_REG_FLOW),
            _operand((_REG_BINDING, _REG_BUFFER), "source"),
            _operand((_REG_BINDING, _REG_BUFFER), "target"),
        ),
        asm_forms=_asm(
            "flow",
            results=("result",),
            operands=("source", "target"),
        ),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.transfer",
        mnemonic="transfer",
        semantic_tag="array.async_transfer",
        operands=(
            _result(_REG_TOKEN),
            _operand(_REG_FLOW, "flow"),
        ),
        asm_forms=_asm(
            "transfer",
            results=("result",),
            operands=("flow",),
        ),
        effects=(_ORDERED_ASYNC_EFFECT,),
        schedule_class=_SCHEDULE_ASYNC,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        instruction_classes=(InstructionClass.GENERIC_MEMORY,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.invoke",
        mnemonic="invoke",
        semantic_tag="array.worker_invoke",
        operands=(
            _result(_REG_TOKEN),
            _operand(_REG_WORKER, "worker"),
            _operand(_REG_BUFFER, "arguments", variadic=True),
        ),
        asm_forms=(
            AsmForm(
                mnemonic="invoke",
                results=("result",),
                operand_segments=(
                    AsmOperandSegment(
                        AsmOperandSegmentDelimiter.ANGLE,
                        ("worker",),
                    ),
                    AsmOperandSegment(
                        AsmOperandSegmentDelimiter.PAREN,
                        ("arguments",),
                    ),
                ),
            ),
        ),
        effects=(_ORDERED_ASYNC_EFFECT,),
        schedule_class=_SCHEDULE_ASYNC,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        instruction_classes=(InstructionClass.CONTROL,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.await",
        mnemonic="await",
        semantic_tag="array.await",
        operands=(
            _operand(_REG_TOKEN, "token"),
            _operand(_REG_TOKEN, "additional_tokens", variadic=True),
        ),
        asm_forms=(
            AsmForm(
                mnemonic="await",
                operand_segments=(
                    AsmOperandSegment(
                        AsmOperandSegmentDelimiter.PAREN,
                        ("token", "additional_tokens"),
                    ),
                ),
            ),
        ),
        effects=(_SYNC_EFFECT,),
        schedule_class=_SCHEDULE_SYNC,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.BARRIER),
        instruction_classes=(InstructionClass.BARRIER,),
    ),
    Descriptor(
        key=f"{_DESCRIPTOR_SET_KEY}.constrain.location",
        mnemonic="constrain.location",
        semantic_tag="array.constraint.location",
        operands=(_operand(_REG_WORKER, "worker"),),
        immediates=(_u32("column"), _u32("row")),
        asm_forms=_asm(
            "constrain.location",
            operands=("worker",),
            immediates=("column", "row"),
        ),
        effects=(_ORDERED_ASYNC_EFFECT,),
        schedule_class=_SCHEDULE_GRAPH,
        flags=(DescriptorFlag.SIDE_EFFECTING,),
    ),
)


AIE2P_ARRAY_DESCRIPTOR_SET = DescriptorSet(
    key=_DESCRIPTOR_SET_KEY,
    target_key=_TARGET_KEY,
    feature_key=f"{_DESCRIPTOR_SET_KEY}.v1",
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
    generator_version=1,
    reg_classes=(
        _reference_reg_class(_REG_BINDING),
        _reference_reg_class(_REG_WORKER),
        _reference_reg_class(_REG_BUFFER),
        _reference_reg_class(_REG_FLOW),
        _reference_reg_class(_REG_TOKEN),
    ),
    resources=(
        Resource(
            _RESOURCE_GRAPH,
            capacity_per_cycle=1,
            kind=ResourceKind.CONTROL,
        ),
        Resource(
            _RESOURCE_ASYNC,
            capacity_per_cycle=1,
            kind=ResourceKind.LOAD,
        ),
        Resource(
            _RESOURCE_SYNC,
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
        ScheduleClass(
            _SCHEDULE_ASYNC,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_ASYNC, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_CALL,),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_SYNC,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_SYNC, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
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
