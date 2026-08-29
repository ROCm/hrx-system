# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Initial AIE2P target-low closure derived from the owned machine tables."""

from __future__ import annotations

from pathlib import Path

from loom.target.arch.amd.xdna.aie.machine import (
    MachineForm,
    MachineOperand,
    MachineOperandKind,
    has_property,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import CORE_ENCODING_TABLE
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Descriptor,
    DescriptorFlag,
    DescriptorSet,
    Effect,
    EffectFlag,
    EffectKind,
    EventSeparation,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandRole,
    PhysicalRegister,
    RegClass,
    RegClassAlt,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
    TimingEvent,
)

_TARGET_KEY = "amd.xdna.aie2p"

_SEED_FORMS = (
    "ADD_add_r_ri",
    "LDA_dms_lda_idx_imm",
    "NOP",
    "RET",
    "ST_dms_sts_idx_imm",
    "VADD_16",
    "VADD_32",
    "VADD_8",
    "VLDA_dmx_lda_x_idx_imm",
    "VLDB_dmx_ldb_x_idx_imm",
    "VST_dmx_sts_x_idx_imm",
)

_DESCRIPTOR_KEYS = {
    "ADD_add_r_ri": f"{_TARGET_KEY}.add.i32.immediate",
    "LDA_dms_lda_idx_imm": f"{_TARGET_KEY}.load.scalar.indexed.immediate",
    "NOP": f"{_TARGET_KEY}.nop",
    "RET": f"{_TARGET_KEY}.return",
    "ST_dms_sts_idx_imm": f"{_TARGET_KEY}.store.scalar.indexed.immediate",
    "VADD_8": f"{_TARGET_KEY}.add.i8x64",
    "VADD_16": f"{_TARGET_KEY}.add.i16x32",
    "VADD_32": f"{_TARGET_KEY}.add.i32x16",
    "VLDA_dmx_lda_x_idx_imm": f"{_TARGET_KEY}.load.a.i8x64.indexed.immediate",
    "VLDB_dmx_ldb_x_idx_imm": f"{_TARGET_KEY}.load.b.i8x64.indexed.immediate",
    "VST_dmx_sts_x_idx_imm": f"{_TARGET_KEY}.store.i8x64.indexed.immediate",
}

_SEMANTIC_TAGS = {
    "ADD_add_r_ri": "integer.add.i32",
    "LDA_dms_lda_idx_imm": "memory.load.indexed.i32",
    "NOP": "control.nop",
    "RET": "control.return",
    "ST_dms_sts_idx_imm": "memory.store.indexed.i32",
    "VADD_8": "integer.add.i8x64",
    "VADD_16": "integer.add.i16x32",
    "VADD_32": "integer.add.i32x16",
    "VLDA_dmx_lda_x_idx_imm": "memory.load.indexed.i8x64",
    "VLDB_dmx_ldb_x_idx_imm": "memory.load.indexed.i8x64",
    "VST_dmx_sts_x_idx_imm": "memory.store.indexed.i8x64",
}

_RESOURCE_ALU = f"{_TARGET_KEY}.slot.alu"
_RESOURCE_LDA = f"{_TARGET_KEY}.slot.lda"
_RESOURCE_LDB = f"{_TARGET_KEY}.slot.ldb"
_RESOURCE_MV = f"{_TARGET_KEY}.slot.mv"
_RESOURCE_NOP = f"{_TARGET_KEY}.slot.nop"
_RESOURCE_ST = f"{_TARGET_KEY}.slot.st"

_SCHEDULE_ALU = f"{_TARGET_KEY}.scalar.alu"
_SCHEDULE_CONTROL = f"{_TARGET_KEY}.control"
_SCHEDULE_LOAD_A = f"{_TARGET_KEY}.load.a"
_SCHEDULE_LOAD_B = f"{_TARGET_KEY}.load.b"
_SCHEDULE_NOP = f"{_TARGET_KEY}.nop"
_SCHEDULE_STORE = f"{_TARGET_KEY}.store"
_SCHEDULE_VECTOR_ALU = f"{_TARGET_KEY}.vector.alu"

_TIMING_LOAD_WRITE_CYCLE_7 = f"{_TARGET_KEY}.load.write.cycle_7"
_TIMING_MEMORY_READ_CYCLE_5 = f"{_TARGET_KEY}.memory.read.cycle_5"
_TIMING_MEMORY_WRITE_CYCLE_5 = f"{_TARGET_KEY}.memory.write.cycle_5"
_TIMING_SCALAR_READ_CYCLE_1 = f"{_TARGET_KEY}.scalar.read.cycle_1"
_TIMING_SCALAR_WRITE_CYCLE_1 = f"{_TARGET_KEY}.scalar.write.cycle_1"
_TIMING_STORE_READ_CYCLE_1 = f"{_TARGET_KEY}.store.read.cycle_1"
_TIMING_VECTOR_READ_CYCLE_1 = f"{_TARGET_KEY}.vector.mv.read.cycle_1"
_TIMING_VECTOR_WRITE_CYCLE_2 = f"{_TARGET_KEY}.vector.mv.write.cycle_2"

_SCHEDULE_BY_FORM = {
    "ADD_add_r_ri": _SCHEDULE_ALU,
    "LDA_dms_lda_idx_imm": _SCHEDULE_LOAD_A,
    "NOP": _SCHEDULE_NOP,
    "RET": _SCHEDULE_CONTROL,
    "ST_dms_sts_idx_imm": _SCHEDULE_STORE,
    "VADD_8": _SCHEDULE_VECTOR_ALU,
    "VADD_16": _SCHEDULE_VECTOR_ALU,
    "VADD_32": _SCHEDULE_VECTOR_ALU,
    "VLDA_dmx_lda_x_idx_imm": _SCHEDULE_LOAD_A,
    "VLDB_dmx_ldb_x_idx_imm": _SCHEDULE_LOAD_B,
    "VST_dmx_sts_x_idx_imm": _SCHEDULE_STORE,
}

_MACHINE_FORMS = {form.name: form for form in CORE_MACHINE_TABLE.forms}
_MACHINE_CLASSES = {
    register_class.name: register_class
    for register_class in CORE_MACHINE_TABLE.register_classes
}
_MACHINE_ADAPTERS = {
    adapter.name: adapter for adapter in CORE_MACHINE_TABLE.register_adapters
}
_MACHINE_IMMEDIATES = {
    immediate.name: immediate for immediate in CORE_MACHINE_TABLE.immediates
}
_INSTRUCTION_ENCODINGS = {
    instruction.name: instruction for instruction in CORE_ENCODING_TABLE.instructions
}
_INSTRUCTION_IDS = {
    instruction.name: index
    for index, instruction in enumerate(
        sorted(CORE_ENCODING_TABLE.instructions, key=lambda row: row.name),
        start=1,
    )
}
_ENCODING_FIELD_IDS = {
    name: index
    for index, name in enumerate(
        sorted(
            {
                field.name
                for instruction in CORE_ENCODING_TABLE.instructions
                for field in instruction.fields
            }
        ),
        start=1,
    )
}
_ADAPTER_IDS = {
    adapter.name: index
    for index, adapter in enumerate(
        sorted(CORE_MACHINE_TABLE.register_adapters, key=lambda row: row.name),
        start=1,
    )
}
_IMMEDIATE_IDS = {
    immediate.name: index
    for index, immediate in enumerate(
        sorted(CORE_MACHINE_TABLE.immediates, key=lambda row: row.name),
        start=1,
    )
}


def _operand_register_class(operand: MachineOperand) -> str:
    if operand.kind is MachineOperandKind.REGISTER_CLASS:
        return operand.type_name
    if operand.kind is MachineOperandKind.REGISTER_ADAPTER:
        return _MACHINE_ADAPTERS[operand.type_name].register_class
    raise ValueError(f"{operand.name}: immediate is not a register operand")


_SEED_REGISTER_CLASS_NAMES = tuple(
    sorted(
        {
            _operand_register_class(operand)
            for form_name in _SEED_FORMS
            for operand in (
                *_MACHINE_FORMS[form_name].outputs,
                *_MACHINE_FORMS[form_name].inputs,
            )
            if operand.kind is not MachineOperandKind.IMMEDIATE
        }
    )
)


def _reg_classes() -> tuple[RegClass, ...]:
    result = []
    for target_bank_id, name in enumerate(_SEED_REGISTER_CLASS_NAMES, start=1):
        machine_class = _MACHINE_CLASSES[name]
        result.append(
            RegClass(
                name=name,
                alloc_unit_bits=machine_class.layout.register_size_bits,
                spill_slot_space=SpillSlotSpace.PRIVATE,
                flags=(
                    RegClassFlag.PHYSICAL,
                    RegClassFlag.UNSPILLABLE,
                    RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS,
                ),
                target_bank_id=target_bank_id,
                physical_registers=machine_class.candidates,
            )
        )
    return tuple(result)


def _physical_registers() -> tuple[PhysicalRegister, ...]:
    return tuple(
        PhysicalRegister(register.name, register.atomic_units)
        for register in CORE_MACHINE_TABLE.physical_registers
    )


def _operand_stages(form_name: str, operand: MachineOperand) -> tuple[int, int]:
    if operand in _MACHINE_FORMS[form_name].outputs:
        if has_property(_MACHINE_FORMS[form_name], "mayLoad"):
            return 0, 7
        if form_name.startswith("VADD_"):
            return 0, 2
        if form_name == "ADD_add_r_ri":
            return 0, 1
        return 0, 0
    if form_name.startswith("VADD_") or form_name in (
        "ADD_add_r_ri",
        "ST_dms_sts_idx_imm",
    ):
        return 1, 0
    return 0, 0


def _operand_timing_events(
    form_name: str,
    operand: MachineOperand,
    role: OperandRole,
) -> tuple[str | None, str | None]:
    form = _MACHINE_FORMS[form_name]
    if role is OperandRole.RESULT:
        if has_property(form, "mayLoad"):
            return None, _TIMING_LOAD_WRITE_CYCLE_7
        if form_name.startswith("VADD_"):
            return None, _TIMING_VECTOR_WRITE_CYCLE_2
        if form_name == "ADD_add_r_ri":
            return None, _TIMING_SCALAR_WRITE_CYCLE_1
    elif form_name.startswith("VADD_"):
        return _TIMING_VECTOR_READ_CYCLE_1, None
    elif form_name == "ADD_add_r_ri":
        return _TIMING_SCALAR_READ_CYCLE_1, None
    elif has_property(form, "mayStore") and operand.name == "src":
        return _TIMING_STORE_READ_CYCLE_1, None
    return None, None


def _low_operand(
    form_name: str,
    operand: MachineOperand,
    role: OperandRole,
) -> Operand:
    read_stage, ready_stage = _operand_stages(form_name, operand)
    read_event, write_event = _operand_timing_events(form_name, operand, role)
    encoded_field_names = {
        field.name for field in _INSTRUCTION_ENCODINGS[form_name].fields
    }
    encoding_field_id = (
        _ENCODING_FIELD_IDS[operand.name] if operand.name in encoded_field_names else 0
    )
    encoding_adapter_id = (
        _ADAPTER_IDS[operand.type_name]
        if encoding_field_id and operand.kind is MachineOperandKind.REGISTER_ADAPTER
        else 0
    )
    return Operand(
        field_name=operand.name,
        role=role,
        reg_alts=(RegClassAlt(_operand_register_class(operand)),),
        encoding_field_id=encoding_field_id,
        encoding_adapter_id=encoding_adapter_id,
        read_stage=read_stage,
        ready_stage=ready_stage,
        read_event=read_event,
        write_event=write_event,
    )


def _immediate(form_name: str, operand: MachineOperand) -> Immediate:
    immediate = _MACHINE_IMMEDIATES[operand.type_name]
    if immediate.allows_symbol_reference:
        kind = ImmediateKind.ORDINAL
        flags = (ImmediateFlag.SYMBOLIC,)
    elif immediate.is_signed:
        kind = ImmediateKind.SIGNED
        flags = ()
    else:
        kind = ImmediateKind.UNSIGNED
        flags = ()
    fixed_zero_bits = immediate.step.bit_length() - 1
    if immediate.is_negative:
        minimum = -(1 << (immediate.encoded_width_bits + fixed_zero_bits))
        maximum = -immediate.step
    elif immediate.is_signed:
        semantic_bits = immediate.encoded_width_bits + fixed_zero_bits
        minimum = -(1 << (semantic_bits - 1))
        maximum = (1 << (semantic_bits - 1)) - immediate.step
    else:
        minimum = 0
        maximum = (
            1 << (immediate.encoded_width_bits + fixed_zero_bits)
        ) - immediate.step
    return Immediate(
        field_name=operand.name,
        kind=kind,
        flags=flags,
        bit_width=immediate.semantic_width_bits,
        value_step=immediate.step,
        encoding_field_id=(
            _ENCODING_FIELD_IDS[operand.name]
            if operand.name
            in {field.name for field in _INSTRUCTION_ENCODINGS[form_name].fields}
            else 0
        ),
        encoding_id=_IMMEDIATE_IDS[operand.type_name],
        signed_min=minimum,
        unsigned_max=maximum,
    )


def _effects(form: MachineForm) -> tuple[Effect, ...]:
    result = []
    if has_property(form, "mayLoad"):
        result.append(
            Effect(
                EffectKind.READ,
                MemorySpace.WORKGROUP,
                flags=(EffectFlag.DEPENDENCY,),
                timing_event=_TIMING_MEMORY_READ_CYCLE_5,
            )
        )
    if has_property(form, "mayStore"):
        result.append(
            Effect(
                EffectKind.WRITE,
                MemorySpace.WORKGROUP,
                flags=(EffectFlag.DEPENDENCY,),
                timing_event=_TIMING_MEMORY_WRITE_CYCLE_5,
            )
        )
    if form.control_flow_kind is not None:
        result.append(Effect(EffectKind.CONTROL))
    return tuple(result)


def _descriptor_flags(form: MachineForm) -> tuple[DescriptorFlag, ...]:
    result = []
    if (
        has_property(form, "hasSideEffects")
        or has_property(form, "mayStore")
        or form.control_flow_kind is not None
    ):
        result.append(DescriptorFlag.SIDE_EFFECTING)
    if form.control_flow_kind == "return" or has_property(form, "isTerminator"):
        result.append(DescriptorFlag.TERMINATOR)
    if not _effects(form) and form.name != "RET":
        result.append(DescriptorFlag.DEAD_REMOVABLE)
    return tuple(result)


def _instruction_classes(form: MachineForm) -> tuple[InstructionClass, ...]:
    if has_property(form, "mayLoad"):
        return (InstructionClass.LOCAL_MEMORY,)
    if has_property(form, "mayStore"):
        return (InstructionClass.LOCAL_MEMORY,)
    if form.control_flow_kind is not None:
        return (InstructionClass.CONTROL,)
    if form.name.startswith("VADD_"):
        return (InstructionClass.VECTOR_ALU,)
    if form.name == "ADD_add_r_ri":
        return (InstructionClass.SCALAR_ALU,)
    if form.name == "NOP":
        return (InstructionClass.CONTROL,)
    return (InstructionClass.OTHER,)


def _descriptor(form_name: str) -> Descriptor:
    form = _MACHINE_FORMS[form_name]
    register_outputs = tuple(
        operand
        for operand in form.outputs
        if operand.kind is not MachineOperandKind.IMMEDIATE
    )
    register_inputs = tuple(
        operand
        for operand in form.inputs
        if operand.kind is not MachineOperandKind.IMMEDIATE
    )
    immediate_inputs = tuple(
        operand
        for operand in form.inputs
        if operand.kind is MachineOperandKind.IMMEDIATE
    )
    mnemonic = form.assembly.split("\t", 1)[0].strip()
    descriptor = Descriptor(
        key=_DESCRIPTOR_KEYS[form_name],
        mnemonic=mnemonic,
        semantic_tag=_SEMANTIC_TAGS[form_name],
        operands=(
            *(
                _low_operand(form_name, operand, OperandRole.RESULT)
                for operand in register_outputs
            ),
            *(
                _low_operand(form_name, operand, OperandRole.OPERAND)
                for operand in register_inputs
            ),
        ),
        immediates=tuple(
            _immediate(form_name, operand) for operand in immediate_inputs
        ),
        schedule_class=_SCHEDULE_BY_FORM[form_name],
        asm_forms=(
            AsmForm(
                mnemonic=mnemonic,
                results=tuple(operand.name for operand in register_outputs),
                operands=tuple(operand.name for operand in register_inputs),
                immediates=tuple(
                    AsmImmediate(operand.name) for operand in immediate_inputs
                ),
            ),
        ),
        effects=_effects(form),
        encoding_id=_INSTRUCTION_IDS[form_name],
        flags=_descriptor_flags(form),
        instruction_classes=_instruction_classes(form),
    )
    expected_fields = {field.name for field in _INSTRUCTION_ENCODINGS[form_name].fields}
    encoded_fields = {
        row.field_name
        for row in (*descriptor.operands, *descriptor.immediates)
        if row.encoding_field_id
    }
    if encoded_fields != expected_fields:
        raise ValueError(
            f"{form_name}: Low descriptor fields {sorted(encoded_fields)} "
            f"do not match instruction fields {sorted(expected_fields)}"
        )
    if len(expected_fields) > 16:
        raise ValueError(f"{form_name}: instruction field count exceeds native storage")
    return descriptor


_RESOURCES = (
    Resource(_RESOURCE_ALU, 1, ResourceKind.SCALAR_ALU),
    Resource(_RESOURCE_LDA, 1, ResourceKind.LOAD),
    Resource(_RESOURCE_LDB, 1, ResourceKind.LOAD),
    Resource(_RESOURCE_MV, 1, ResourceKind.VECTOR_ALU),
    Resource(_RESOURCE_NOP, 1, ResourceKind.CONTROL),
    Resource(_RESOURCE_ST, 1, ResourceKind.STORE),
)

_TIMING_EVENTS = tuple(
    TimingEvent(name)
    for name in (
        _TIMING_LOAD_WRITE_CYCLE_7,
        _TIMING_MEMORY_READ_CYCLE_5,
        _TIMING_MEMORY_WRITE_CYCLE_5,
        _TIMING_SCALAR_READ_CYCLE_1,
        _TIMING_SCALAR_WRITE_CYCLE_1,
        _TIMING_STORE_READ_CYCLE_1,
        _TIMING_VECTOR_READ_CYCLE_1,
        _TIMING_VECTOR_WRITE_CYCLE_2,
    )
)

_EVENT_SEPARATIONS = tuple(
    EventSeparation(producer, consumer, cycles, ModelQuality.CALIBRATED)
    for producer, consumer, cycles in (
        (_TIMING_LOAD_WRITE_CYCLE_7, _TIMING_SCALAR_READ_CYCLE_1, 7),
        (_TIMING_LOAD_WRITE_CYCLE_7, _TIMING_VECTOR_READ_CYCLE_1, 7),
        (_TIMING_MEMORY_WRITE_CYCLE_5, _TIMING_MEMORY_READ_CYCLE_5, 1),
        (_TIMING_SCALAR_READ_CYCLE_1, _TIMING_SCALAR_WRITE_CYCLE_1, 0),
        (_TIMING_SCALAR_WRITE_CYCLE_1, _TIMING_SCALAR_READ_CYCLE_1, 1),
        (_TIMING_SCALAR_WRITE_CYCLE_1, _TIMING_SCALAR_WRITE_CYCLE_1, 1),
        (_TIMING_SCALAR_WRITE_CYCLE_1, _TIMING_STORE_READ_CYCLE_1, 1),
        (_TIMING_VECTOR_READ_CYCLE_1, _TIMING_VECTOR_WRITE_CYCLE_2, 0),
        (_TIMING_VECTOR_WRITE_CYCLE_2, _TIMING_STORE_READ_CYCLE_1, 2),
        (_TIMING_VECTOR_WRITE_CYCLE_2, _TIMING_VECTOR_READ_CYCLE_1, 1),
        (_TIMING_VECTOR_WRITE_CYCLE_2, _TIMING_VECTOR_WRITE_CYCLE_2, 1),
    )
)

_SCHEDULE_CLASSES = (
    ScheduleClass(
        _SCHEDULE_ALU,
        LatencyKind.EXACT,
        ModelQuality.CALIBRATED,
        latency_cycles=1,
        issue_uses=(IssueUse(_RESOURCE_ALU, 1, 1),),
        instruction_classes=(InstructionClass.SCALAR_ALU,),
    ),
    ScheduleClass(
        _SCHEDULE_CONTROL,
        LatencyKind.EXACT,
        ModelQuality.EXACT,
        latency_cycles=1,
        issue_uses=(IssueUse(_RESOURCE_ALU, 1, 1),),
        flags=(ScheduleClassFlag.CONTROL,),
        instruction_classes=(InstructionClass.CONTROL,),
    ),
    ScheduleClass(
        _SCHEDULE_LOAD_A,
        LatencyKind.EXACT,
        ModelQuality.CALIBRATED,
        latency_cycles=7,
        minimum_issue_separation_cycles=7,
        issue_uses=(IssueUse(_RESOURCE_LDA, 1, 1),),
        flags=(ScheduleClassFlag.MAY_LOAD,),
        instruction_classes=(InstructionClass.LOCAL_MEMORY,),
    ),
    ScheduleClass(
        _SCHEDULE_LOAD_B,
        LatencyKind.EXACT,
        ModelQuality.CALIBRATED,
        latency_cycles=7,
        minimum_issue_separation_cycles=7,
        issue_uses=(IssueUse(_RESOURCE_LDB, 1, 1),),
        flags=(ScheduleClassFlag.MAY_LOAD,),
        instruction_classes=(InstructionClass.LOCAL_MEMORY,),
    ),
    ScheduleClass(
        _SCHEDULE_NOP,
        LatencyKind.EXACT,
        ModelQuality.EXACT,
        latency_cycles=1,
        issue_uses=(IssueUse(_RESOURCE_NOP, 1, 1),),
        instruction_classes=(InstructionClass.CONTROL,),
    ),
    ScheduleClass(
        _SCHEDULE_STORE,
        LatencyKind.EXACT,
        ModelQuality.CALIBRATED,
        latency_cycles=1,
        issue_uses=(IssueUse(_RESOURCE_ST, 1, 1),),
        flags=(ScheduleClassFlag.MAY_STORE,),
        instruction_classes=(InstructionClass.LOCAL_MEMORY,),
    ),
    ScheduleClass(
        _SCHEDULE_VECTOR_ALU,
        LatencyKind.EXACT,
        ModelQuality.CALIBRATED,
        latency_cycles=1,
        minimum_issue_separation_cycles=2,
        issue_uses=(IssueUse(_RESOURCE_MV, 1, 1),),
        instruction_classes=(InstructionClass.VECTOR_ALU,),
    ),
)


AIE2P_CORE_DESCRIPTOR_SET = DescriptorSet(
    key=f"{_TARGET_KEY}.core",
    target_key=_TARGET_KEY,
    feature_key=f"{_TARGET_KEY}.core.v1",
    c_header_path=Path(
        "loom/src/loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
    ),
    c_source_path=Path(
        "loom/src/loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.c"
    ),
    header_guard=("LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_DESCRIPTORS_CORE_DESCRIPTORS_H_"),
    public_header=("loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"),
    function_name="loom_aie2p_core_descriptor_set",
    c_table_prefix="Aie2pCore",
    c_enum_prefix="AIE2P_CORE",
    generator_version=1,
    reg_classes=_reg_classes(),
    physical_registers=_physical_registers(),
    timing_events=_TIMING_EVENTS,
    event_separations=_EVENT_SEPARATIONS,
    resources=_RESOURCES,
    schedule_classes=_SCHEDULE_CLASSES,
    descriptors=tuple(_descriptor(form_name) for form_name in _SEED_FORMS),
    requires_explicit_asm_surface=True,
)
