# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor input data for the portable command-machine low target."""

from __future__ import annotations

from pathlib import Path

from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    AsmOperandSegment,
    AsmOperandSegmentDelimiter,
    Descriptor,
    DescriptorCarrier,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    Immediate,
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

_REG_U32 = "cmd.u32"
_REG_U64 = "cmd.u64"
_REG_B8 = "cmd.b8"
_REG_B16 = "cmd.b16"
_REG_B32 = "cmd.b32"
_REG_B64 = "cmd.b64"
_REG_BUFFER = "cmd.buffer"
_REG_BINDING = "cmd.binding"
_REG_BUFFER_REF = "cmd.buffer_ref"
_REG_EXECUTABLE = "cmd.executable"
_REG_ENTRY = "cmd.entry"

_SCHEDULE_PURE = "cmd.pure"
_SCHEDULE_RECORD = "cmd.record"
_SCHEDULE_BARRIER = "cmd.barrier"

_RESOURCE_PURE = "cmd.pure"
_RESOURCE_RECORD = "cmd.record"
_RESOURCE_CONTROL = "cmd.control"

_SCALAR_BANK = 1
_REFERENCE_BANK = 2


def _asm(
    *,
    results: tuple[str, ...] = (),
    operands: tuple[str, ...] = (),
    immediates: tuple[str, ...] = (),
) -> tuple[AsmForm, ...]:
    return (
        AsmForm(
            results=results,
            operands=operands,
            immediates=tuple(AsmImmediate(name) for name in immediates),
        ),
    )


_ALT_BY_CLASS = {
    reg_class: (RegClassAlt(reg_class),)
    for reg_class in (
        _REG_U32,
        _REG_U64,
        _REG_B8,
        _REG_B16,
        _REG_B32,
        _REG_B64,
        _REG_BUFFER,
        _REG_BINDING,
        _REG_BUFFER_REF,
        _REG_EXECUTABLE,
        _REG_ENTRY,
    )
}


def _value(reg_class: str, role: OperandRole, field_name: str) -> Operand:
    return Operand(field_name, role, _ALT_BY_CLASS[reg_class])


def _result(reg_class: str, field_name: str = "result") -> Operand:
    return _value(reg_class, OperandRole.RESULT, field_name)


def _operand(reg_class: str, field_name: str) -> Operand:
    return _value(reg_class, OperandRole.OPERAND, field_name)


def _variadic_operand(reg_classes: tuple[str, ...], field_name: str) -> Operand:
    return Operand(
        field_name,
        OperandRole.OPERAND,
        tuple(RegClassAlt(reg_class) for reg_class in reg_classes),
        flags=(OperandFlag.VARIADIC,),
    )


_RECORD_EFFECT = Effect(
    EffectKind.CALL,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)

_BARRIER_EFFECT = Effect(
    EffectKind.BARRIER,
    memory_space=MemorySpace.GENERIC,
    flags=(EffectFlag.ORDERED, EffectFlag.DEPENDENCY),
)


def _constant_descriptor(reg_class: str, spelling: str, bit_width: int) -> Descriptor:
    immediate = Immediate(
        "value",
        ImmediateKind.UNSIGNED,
        bit_width=bit_width,
        unsigned_max=(2**bit_width) - 1,
    )
    return Descriptor(
        key=f"cmd.constant.{spelling}",
        mnemonic=f"cmd.constant.{spelling}",
        semantic_tag=f"constant.{spelling}",
        operands=(_result(reg_class),),
        immediates=(immediate,),
        carrier=DescriptorCarrier.CONST,
        asm_forms=_asm(results=("result",), immediates=("value",)),
        schedule_class=_SCHEDULE_PURE,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


_CONSTANT_DESCRIPTORS = (
    _constant_descriptor(_REG_U32, "u32", 32),
    _constant_descriptor(_REG_U64, "u64", 64),
    _constant_descriptor(_REG_B8, "b8", 8),
    _constant_descriptor(_REG_B16, "b16", 16),
    _constant_descriptor(_REG_B32, "b32", 32),
    _constant_descriptor(_REG_B64, "b64", 64),
)

_VALUE_DESCRIPTORS = (
    Descriptor(
        key="cmd.buffer.ref.direct",
        mnemonic="cmd.buffer.ref.direct",
        semantic_tag="command.buffer_ref.direct",
        operands=(
            _result(_REG_BUFFER_REF),
            _operand(_REG_BUFFER, "buffer"),
            _operand(_REG_U64, "offset"),
            _operand(_REG_U64, "length"),
        ),
        asm_forms=_asm(results=("result",), operands=("buffer", "offset", "length")),
        schedule_class=_SCHEDULE_PURE,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
    Descriptor(
        key="cmd.buffer.ref.binding",
        mnemonic="cmd.buffer.ref.binding",
        semantic_tag="command.buffer_ref.binding",
        operands=(
            _result(_REG_BUFFER_REF),
            _operand(_REG_BINDING, "binding"),
            _operand(_REG_U64, "offset"),
            _operand(_REG_U64, "length"),
        ),
        asm_forms=_asm(results=("result",), operands=("binding", "offset", "length")),
        schedule_class=_SCHEDULE_PURE,
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    ),
)


def _dispatch_descriptor(*, indirect_mode: str | None, barrier: bool) -> Descriptor:
    if indirect_mode is None:
        key = "cmd.dispatch.direct"
        workgroup_operands = (
            _operand(_REG_U32, "workgroup_count_x"),
            _operand(_REG_U32, "workgroup_count_y"),
            _operand(_REG_U32, "workgroup_count_z"),
        )
        workgroup_names = (
            "workgroup_count_x",
            "workgroup_count_y",
            "workgroup_count_z",
        )
    else:
        key = f"cmd.dispatch.indirect.{indirect_mode}"
        workgroup_operands = (_operand(_REG_BUFFER_REF, "workgroup_count"),)
        workgroup_names = ("workgroup_count",)
    if barrier:
        key = f"{key}.barrier"
    return Descriptor(
        key=key,
        mnemonic=key,
        semantic_tag=key.replace("cmd.", "command."),
        operands=(
            _operand(_REG_EXECUTABLE, "executable"),
            _operand(_REG_ENTRY, "entry"),
            *workgroup_operands,
            _variadic_operand(
                (
                    _REG_B8,
                    _REG_B16,
                    _REG_B32,
                    _REG_B64,
                    _REG_BUFFER,
                    _REG_BINDING,
                    _REG_U64,
                ),
                "arguments",
            ),
        ),
        asm_forms=(
            AsmForm(
                operand_segments=(
                    AsmOperandSegment(
                        AsmOperandSegmentDelimiter.ANGLE,
                        ("executable", "entry"),
                    ),
                    AsmOperandSegment(
                        AsmOperandSegmentDelimiter.SQUARE, workgroup_names
                    ),
                    AsmOperandSegment(AsmOperandSegmentDelimiter.PAREN, ("arguments",)),
                ),
            ),
        ),
        effects=(_RECORD_EFFECT, _BARRIER_EFFECT) if barrier else (_RECORD_EFFECT,),
        schedule_class=_SCHEDULE_BARRIER if barrier else _SCHEDULE_RECORD,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.BARRIER)
        if barrier
        else (DescriptorFlag.SIDE_EFFECTING,),
        instruction_classes=(InstructionClass.CONTROL, InstructionClass.BARRIER)
        if barrier
        else (InstructionClass.CONTROL,),
    )


def _fill_descriptor(*, barrier: bool) -> Descriptor:
    key = "cmd.fill.barrier" if barrier else "cmd.fill"
    return Descriptor(
        key=key,
        mnemonic=key,
        semantic_tag=key.replace("cmd.", "command."),
        operands=(
            _operand(_REG_BUFFER_REF, "target"),
            _operand(_REG_U32, "pattern"),
            _operand(_REG_U32, "pattern_length"),
        ),
        asm_forms=_asm(operands=("target", "pattern", "pattern_length")),
        effects=(_RECORD_EFFECT, _BARRIER_EFFECT) if barrier else (_RECORD_EFFECT,),
        schedule_class=_SCHEDULE_BARRIER if barrier else _SCHEDULE_RECORD,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.BARRIER)
        if barrier
        else (DescriptorFlag.SIDE_EFFECTING,),
        instruction_classes=(
            InstructionClass.GENERIC_MEMORY,
            InstructionClass.BARRIER,
        )
        if barrier
        else (InstructionClass.GENERIC_MEMORY,),
    )


def _copy_descriptor(*, barrier: bool) -> Descriptor:
    key = "cmd.copy.barrier" if barrier else "cmd.copy"
    return Descriptor(
        key=key,
        mnemonic=key,
        semantic_tag=key.replace("cmd.", "command."),
        operands=(
            _operand(_REG_BUFFER_REF, "source"),
            _operand(_REG_BUFFER_REF, "target"),
        ),
        asm_forms=_asm(operands=("source", "target")),
        effects=(_RECORD_EFFECT, _BARRIER_EFFECT) if barrier else (_RECORD_EFFECT,),
        schedule_class=_SCHEDULE_BARRIER if barrier else _SCHEDULE_RECORD,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.BARRIER)
        if barrier
        else (DescriptorFlag.SIDE_EFFECTING,),
        instruction_classes=(
            InstructionClass.GENERIC_MEMORY,
            InstructionClass.BARRIER,
        )
        if barrier
        else (InstructionClass.GENERIC_MEMORY,),
    )


_COMMAND_DESCRIPTORS = (
    _fill_descriptor(barrier=False),
    _fill_descriptor(barrier=True),
    _copy_descriptor(barrier=False),
    _copy_descriptor(barrier=True),
    _dispatch_descriptor(indirect_mode=None, barrier=False),
    _dispatch_descriptor(indirect_mode=None, barrier=True),
    _dispatch_descriptor(indirect_mode="static", barrier=False),
    _dispatch_descriptor(indirect_mode="static", barrier=True),
    _dispatch_descriptor(indirect_mode="dynamic", barrier=False),
    _dispatch_descriptor(indirect_mode="dynamic", barrier=True),
    Descriptor(
        key="cmd.barrier.execution",
        mnemonic="cmd.barrier.execution",
        semantic_tag="command.barrier.execution",
        operands=(),
        asm_forms=_asm(),
        effects=(_BARRIER_EFFECT,),
        schedule_class=_SCHEDULE_BARRIER,
        flags=(DescriptorFlag.SIDE_EFFECTING, DescriptorFlag.BARRIER),
        instruction_classes=(InstructionClass.BARRIER,),
    ),
)


def _scalar_reg_class(name: str, bit_width: int) -> RegClass:
    return RegClass(
        name,
        bit_width,
        SpillSlotSpace.PRIVATE,
        flags=(RegClassFlag.VIRTUAL_ONLY, RegClassFlag.UNSPILLABLE),
        target_bank_id=_SCALAR_BANK,
        alias_set_id=_SCALAR_BANK,
    )


def _reference_reg_class(name: str) -> RegClass:
    return RegClass(
        name,
        64,
        SpillSlotSpace.PRIVATE,
        flags=(
            RegClassFlag.VIRTUAL_ONLY,
            RegClassFlag.REFERENCE,
            RegClassFlag.UNSPILLABLE,
        ),
        target_bank_id=_REFERENCE_BANK,
        alias_set_id=_REFERENCE_BANK,
    )


CMD_CORE_DESCRIPTOR_SET = DescriptorSet(
    key="cmd.core",
    target_key="cmd",
    feature_key="cmd.v1",
    c_header_path=Path("loom/src/loom/target/arch/cmd/descriptors.h"),
    c_source_path=Path("loom/src/loom/target/arch/cmd/descriptors.c"),
    header_guard="LOOM_TARGET_ARCH_CMD_DESCRIPTORS_H_",
    public_header="loom/target/arch/cmd/descriptors/descriptors.h",
    function_name="loom_cmd_core_descriptor_set",
    c_table_prefix="CmdCore",
    c_enum_prefix="CMD_CORE",
    generator_version=1,
    reg_classes=(
        _scalar_reg_class(_REG_U32, 32),
        _scalar_reg_class(_REG_U64, 64),
        _scalar_reg_class(_REG_B8, 8),
        _scalar_reg_class(_REG_B16, 16),
        _scalar_reg_class(_REG_B32, 32),
        _scalar_reg_class(_REG_B64, 64),
        _reference_reg_class(_REG_BUFFER),
        _scalar_reg_class(_REG_BINDING, 32),
        _reference_reg_class(_REG_BUFFER_REF),
        _reference_reg_class(_REG_EXECUTABLE),
        _scalar_reg_class(_REG_ENTRY, 64),
    ),
    resources=(
        Resource(_RESOURCE_PURE, capacity_per_cycle=1, kind=ResourceKind.SCALAR_ALU),
        Resource(_RESOURCE_RECORD, capacity_per_cycle=1, kind=ResourceKind.STORE),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        ScheduleClass(
            _SCHEDULE_PURE,
            latency_kind=LatencyKind.EXACT,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_PURE, cycles=1, units=1),),
            model_quality=ModelQuality.EXACT,
        ),
        ScheduleClass(
            _SCHEDULE_RECORD,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_RECORD, cycles=1, units=1),),
            flags=(ScheduleClassFlag.MAY_CALL,),
            model_quality=ModelQuality.FALLBACK,
            instruction_classes=(InstructionClass.CONTROL,),
        ),
        ScheduleClass(
            _SCHEDULE_BARRIER,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
            instruction_classes=(InstructionClass.BARRIER,),
        ),
    ),
    descriptors=(
        *_CONSTANT_DESCRIPTORS,
        *_VALUE_DESCRIPTORS,
        *_COMMAND_DESCRIPTORS,
    ),
)
