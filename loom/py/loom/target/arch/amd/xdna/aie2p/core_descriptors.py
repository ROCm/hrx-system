# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P target-low closure derived from owned machine and schedule tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loom.target.arch.amd.xdna.aie.machine import (
    MachineForm,
    MachineOperand,
    MachineOperandKind,
    has_property,
)
from loom.target.arch.amd.xdna.aie.schedule import (
    NO_ITINERARY,
    DependencyKind,
    Itinerary,
    MemoryCycles,
    PipelineStageKind,
    bypass_class,
    dependency_separation,
    itinerary_payload,
    memory_separation,
    pipeline_uses,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import CORE_ENCODING_TABLE
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.arch.amd.xdna.aie2p.core_schedule_data import CORE_SCHEDULE_TABLE
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    DescriptorOpKind,
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
    IssueUseKind,
    LatencyKind,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandRole,
    PhysicalRegister,
    RegClass,
    RegClassAlt,
    RegClassAltFlag,
    RegClassFlag,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
    TimingEvent,
)

_TARGET_KEY = "amd.xdna.aie2p"


@dataclass(frozen=True, slots=True)
class _DescriptorSpec:
    """Semantic selection of one physical form and its exact itinerary."""

    form_name: str
    key: str
    semantic_tag: str
    itinerary: str
    storage_overrides: tuple[tuple[str, str], ...] = ()
    op_kind: DescriptorOpKind = DescriptorOpKind.OP


_DESCRIPTOR_SPECS = (
    _DescriptorSpec(
        "ADD_add_r_ri",
        f"{_TARGET_KEY}.add.i32.immediate",
        "integer.add.i32",
        "II_ADD_add_r_ri",
    ),
    _DescriptorSpec(
        "LDA_dms_lda_idx_imm",
        f"{_TARGET_KEY}.load.scalar.indexed.immediate",
        "memory.load.indexed.i32",
        "II_LDA_dms_lda_idx_imm",
    ),
    _DescriptorSpec(
        "NOP",
        f"{_TARGET_KEY}.nop",
        "control.nop",
        "NoItinerary",
    ),
    _DescriptorSpec(
        "RET",
        f"{_TARGET_KEY}.return",
        "control.return",
        "II_RET",
    ),
    _DescriptorSpec(
        "ST_dms_sts_idx_imm",
        f"{_TARGET_KEY}.store.scalar.indexed.immediate",
        "memory.store.indexed.i32",
        "II_ST_dms_sts_idx_imm",
    ),
    _DescriptorSpec(
        "VADD_8",
        f"{_TARGET_KEY}.add.i8x64",
        "integer.add.i8x64",
        "II_VADD_8",
    ),
    _DescriptorSpec(
        "VADD_16",
        f"{_TARGET_KEY}.add.i16x32",
        "integer.add.i16x32",
        "II_VADD_16",
    ),
    _DescriptorSpec(
        "VADD_32",
        f"{_TARGET_KEY}.add.i32x16",
        "integer.add.i32x16",
        "II_VADD_32",
    ),
    _DescriptorSpec(
        "VSUB_8",
        f"{_TARGET_KEY}.sub.i8x64",
        "integer.sub.i8x64",
        "II_VSUB_8",
    ),
    _DescriptorSpec(
        "VSUB_16",
        f"{_TARGET_KEY}.sub.i16x32",
        "integer.sub.i16x32",
        "II_VSUB_16",
    ),
    _DescriptorSpec(
        "VSUB_32",
        f"{_TARGET_KEY}.sub.i32x16",
        "integer.sub.i32x16",
        "II_VSUB_32",
    ),
    _DescriptorSpec(
        "VBAND",
        f"{_TARGET_KEY}.and.bits512",
        "integer.and.bits512",
        "II_VBAND",
    ),
    _DescriptorSpec(
        "VBOR",
        f"{_TARGET_KEY}.or.bits512",
        "integer.or.bits512",
        "II_VBOR",
    ),
    _DescriptorSpec(
        "VBCST_8",
        f"{_TARGET_KEY}.splat.i8x64",
        "integer.splat.i8x64",
        "II_VBCST_8",
    ),
    _DescriptorSpec(
        "VBCST_16",
        f"{_TARGET_KEY}.splat.i16x32",
        "integer.splat.i16x32",
        "II_VBCST_16",
    ),
    _DescriptorSpec(
        "VBCST_32",
        f"{_TARGET_KEY}.splat.i32x16",
        "integer.splat.i32x16",
        "II_VBCST_32",
    ),
    _DescriptorSpec(
        "VLDA_dmx_lda_x_idx_imm",
        f"{_TARGET_KEY}.load.a.i8x64.indexed.immediate",
        "memory.load.indexed.i8x64",
        "II_VLDA_dmx_lda_x_idx_imm",
    ),
    _DescriptorSpec(
        "VLDB_dmx_ldb_x_idx_imm",
        f"{_TARGET_KEY}.load.b.i8x64.indexed.immediate",
        "memory.load.indexed.i8x64",
        "II_VLDB_dmx_ldb_x_idx_imm",
    ),
    _DescriptorSpec(
        "VST_dmx_sts_x_idx_imm",
        f"{_TARGET_KEY}.store.i8x64.indexed.immediate",
        "memory.store.indexed.i8x64",
        "II_VST_dmx_sts_x_idx_imm",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_cg",
        f"{_TARGET_KEY}.constant.i32.short",
        "integer.const.i32",
        "II_MOV_alu_mv_mv_mv_cg_eR",
        (("dst", "eR"),),
        DescriptorOpKind.CONST,
    ),
    _DescriptorSpec(
        "MOVXM",
        f"{_TARGET_KEY}.constant.i32",
        "integer.const.i32",
        "II_MOVXM_eR",
        (("dst", "eR"),),
        DescriptorOpKind.CONST,
    ),
    _DescriptorSpec(
        "ADD_alu_r_rr",
        f"{_TARGET_KEY}.add.i32",
        "integer.add.i32",
        "II_ADD_alu_r_rr",
    ),
    _DescriptorSpec("SUB", f"{_TARGET_KEY}.sub.i32", "integer.sub.i32", "II_SUB"),
    _DescriptorSpec("MUL", f"{_TARGET_KEY}.mul.i32", "integer.mul.i32", "II_MUL"),
    _DescriptorSpec("AND", f"{_TARGET_KEY}.and.i32", "integer.and.i32", "II_AND"),
    _DescriptorSpec("OR", f"{_TARGET_KEY}.or.i32", "integer.or.i32", "II_OR"),
    _DescriptorSpec("XOR", f"{_TARGET_KEY}.xor.i32", "integer.xor.i32", "II_XOR"),
    _DescriptorSpec("ASHL", f"{_TARGET_KEY}.ashl.i32", "integer.ashl.i32", "II_ASHL"),
    _DescriptorSpec("LSHL", f"{_TARGET_KEY}.lshl.i32", "integer.lshl.i32", "II_LSHL"),
    _DescriptorSpec("EQ", f"{_TARGET_KEY}.cmp.eq.i32", "integer.cmp.eq.i32", "II_EQ"),
    _DescriptorSpec("NE", f"{_TARGET_KEY}.cmp.ne.i32", "integer.cmp.ne.i32", "II_NE"),
    _DescriptorSpec(
        "EQZ",
        f"{_TARGET_KEY}.cmp.eqz.i32",
        "integer.cmp.eq.i32",
        "II_EQZ",
    ),
    _DescriptorSpec(
        "NEZ",
        f"{_TARGET_KEY}.cmp.nez.i32",
        "integer.cmp.ne.i32",
        "II_NEZ",
    ),
    _DescriptorSpec(
        "LT",
        f"{_TARGET_KEY}.cmp.slt.i32",
        "integer.cmp.slt.i32",
        "II_LT",
    ),
    _DescriptorSpec(
        "GE",
        f"{_TARGET_KEY}.cmp.sge.i32",
        "integer.cmp.sge.i32",
        "II_GE",
    ),
    _DescriptorSpec(
        "LTU",
        f"{_TARGET_KEY}.cmp.ult.i32",
        "integer.cmp.ult.i32",
        "II_LTU",
    ),
    _DescriptorSpec(
        "GEU",
        f"{_TARGET_KEY}.cmp.uge.i32",
        "integer.cmp.uge.i32",
        "II_GEU",
    ),
)

_ASM_MNEMONIC_BY_FORM = {
    "MOV_alu_mv_mv_mv_cg": "mov.short",
    "MOVXM": "mov.i32",
    "ADD_alu_r_rr": "add.rr",
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

# LLVM's mXa/mXb/mXm/mXn/mXs names describe instruction-operand encoding
# roles, not distinct storage domains. They have the same 512-bit layout and
# ordered X-register candidates as the canonical VEC512 class. Low values use
# that canonical storage class while each operand retains its role-specific
# adapter below, allowing loads, arithmetic, and stores to compose without
# erasing the encoding distinction.
_LOW_REGISTER_CLASS_BY_MACHINE_CLASS = {
    "mXa": "VEC512",
    "mXb": "VEC512",
    "mXm": "VEC512",
    "mXn": "VEC512",
    "mXs": "VEC512",
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


def _operand_storage_machine_class(
    spec: _DescriptorSpec,
    operand: MachineOperand,
) -> str:
    if operand.kind is MachineOperandKind.REGISTER_CLASS:
        machine_class = operand.type_name
    elif operand.kind is MachineOperandKind.REGISTER_ADAPTER:
        machine_class = _MACHINE_ADAPTERS[operand.type_name].register_class
    else:
        raise ValueError(f"{operand.name}: immediate is not a register operand")
    storage_overrides = dict(spec.storage_overrides)
    unknown_overrides = storage_overrides.keys() - {
        row.name
        for row in (
            *_MACHINE_FORMS[spec.form_name].outputs,
            *_MACHINE_FORMS[spec.form_name].inputs,
        )
    }
    if unknown_overrides:
        raise ValueError(
            f"{spec.form_name}: storage overrides name unknown operands "
            f"{sorted(unknown_overrides)}"
        )
    low_class = storage_overrides.get(
        operand.name,
        _LOW_REGISTER_CLASS_BY_MACHINE_CLASS.get(machine_class, machine_class),
    )
    source = _MACHINE_CLASSES[machine_class]
    target = _MACHINE_CLASSES[low_class]
    selected_candidates = tuple(
        candidate
        for candidate in source.candidates
        if candidate in set(target.candidates)
    )
    if source.layout != target.layout or selected_candidates != target.candidates:
        raise ValueError(
            f"{spec.form_name}.{operand.name}: Low class {low_class} is not an "
            f"ordered storage subset of {machine_class}"
        )
    if operand.kind is MachineOperandKind.REGISTER_ADAPTER:
        encoded_registers = {
            register
            for register, _ in _MACHINE_ADAPTERS[
                operand.type_name
            ].effective_register_encodings
        }
        if not set(target.candidates).issubset(encoded_registers):
            raise ValueError(
                f"{spec.form_name}.{operand.name}: adapter {operand.type_name} "
                f"does not encode Low class {low_class}"
            )
    return low_class


def _low_register_class_name(machine_class: str) -> str:
    return f"aie2p.{machine_class.lower()}"


def _operand_register_class(spec: _DescriptorSpec, operand: MachineOperand) -> str:
    return _low_register_class_name(_operand_storage_machine_class(spec, operand))


_EXPLICIT_STORAGE_MACHINE_CLASS_NAMES = tuple(
    sorted(
        {
            _operand_storage_machine_class(spec, operand)
            for spec in _DESCRIPTOR_SPECS
            for operand in (
                *_MACHINE_FORMS[spec.form_name].outputs,
                *_MACHINE_FORMS[spec.form_name].inputs,
            )
            if operand.kind is not MachineOperandKind.IMMEDIATE
        }
    )
)

_IMPLICIT_REGISTER_MACHINE_CLASSES = {
    "lr": "mLRa",
    "srCarry": "mSRCarry",
}


def _implicit_register_class_name(register_name: str) -> str:
    return f"aie2p.state.{register_name.lower()}"


def _reg_classes() -> tuple[RegClass, ...]:
    result: list[RegClass] = []
    for target_bank_id, machine_name in enumerate(
        _EXPLICIT_STORAGE_MACHINE_CLASS_NAMES, start=1
    ):
        machine_class = _MACHINE_CLASSES[machine_name]
        result.append(
            RegClass(
                name=_low_register_class_name(machine_name),
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
    next_bank_id = len(result) + 1
    selected_implicit_registers = sorted(
        {
            register_name
            for spec in _DESCRIPTOR_SPECS
            for register_name in (
                *_MACHINE_FORMS[spec.form_name].implicit_defs,
                *_MACHINE_FORMS[spec.form_name].implicit_uses,
            )
        }
    )
    unknown_implicit_registers = (
        set(selected_implicit_registers) - _IMPLICIT_REGISTER_MACHINE_CLASSES.keys()
    )
    if unknown_implicit_registers:
        raise ValueError(
            "selected AIE2P forms require unclassified implicit registers: "
            f"{sorted(unknown_implicit_registers)}"
        )
    for register_name in selected_implicit_registers:
        machine_class = _MACHINE_CLASSES[
            _IMPLICIT_REGISTER_MACHINE_CLASSES[register_name]
        ]
        if machine_class.candidates != (register_name,):
            raise ValueError(
                f"{machine_class.name}: implicit state class must name only "
                f"{register_name}"
            )
        result.append(
            RegClass(
                name=_implicit_register_class_name(register_name),
                alloc_unit_bits=machine_class.layout.register_size_bits,
                spill_slot_space=SpillSlotSpace.PRIVATE,
                flags=(
                    RegClassFlag.PHYSICAL,
                    RegClassFlag.UNSPILLABLE,
                    RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS,
                ),
                target_bank_id=next_bank_id,
                physical_registers=machine_class.candidates,
            )
        )
        next_bank_id += 1
    return tuple(result)


def _physical_registers() -> tuple[PhysicalRegister, ...]:
    return tuple(
        PhysicalRegister(register.name, register.atomic_units)
        for register in CORE_MACHINE_TABLE.physical_registers
    )


_ITINERARIES = {
    NO_ITINERARY.name: NO_ITINERARY,
    **{itinerary.name: itinerary for itinerary in CORE_SCHEDULE_TABLE.itineraries},
}


def _itinerary(spec: _DescriptorSpec) -> Itinerary:
    form = _MACHINE_FORMS[spec.form_name]
    itinerary = _ITINERARIES[spec.itinerary]
    expected_count = (
        len(form.outputs)
        + len(form.inputs)
        + len(form.implicit_defs)
        + len(form.implicit_uses)
    )
    if len(itinerary.operand_cycles) != expected_count:
        raise ValueError(
            f"{spec.form_name}: itinerary {spec.itinerary} has "
            f"{len(itinerary.operand_cycles)} operand cycles for "
            f"{expected_count} machine operands"
        )
    return itinerary


def _operand_ordinal(form: MachineForm, operand: MachineOperand) -> int:
    return (*form.outputs, *form.inputs).index(operand)


def _register_event_name(
    access: str,
    cycle: int,
    bypass: str | None,
) -> str:
    bypass_segment = bypass.lower() if bypass is not None else "none"
    return f"{_TARGET_KEY}.operand.{access}.c{cycle}.{bypass_segment}"


def _register_timing_event(
    spec: _DescriptorSpec,
    operand_ordinal: int,
    access: str,
) -> str:
    itinerary = _itinerary(spec)
    cycle = itinerary.operand_cycles[operand_ordinal]
    bypass = bypass_class(itinerary, operand_ordinal)
    return _register_event_name(access, cycle, bypass)


def _operand_stages(
    spec: _DescriptorSpec,
    operand: MachineOperand,
) -> tuple[int, int]:
    form = _MACHINE_FORMS[spec.form_name]
    operand_ordinal = _operand_ordinal(form, operand)
    cycle = _itinerary(spec).operand_cycles[operand_ordinal]
    return (0, cycle) if operand_ordinal < len(form.outputs) else (cycle, 0)


def _implicit_operand_stage(
    spec: _DescriptorSpec,
    register_name: str,
    *,
    is_definition: bool,
) -> tuple[int, int]:
    form = _MACHINE_FORMS[spec.form_name]
    if is_definition:
        ordinal = (
            len(form.outputs)
            + len(form.inputs)
            + form.implicit_defs.index(register_name)
        )
    else:
        ordinal = (
            len(form.outputs)
            + len(form.inputs)
            + len(form.implicit_defs)
            + form.implicit_uses.index(register_name)
        )
    cycle = _itinerary(spec).operand_cycles[ordinal]
    return (0, cycle) if is_definition else (cycle, 0)


def _operand_timing_events(
    spec: _DescriptorSpec,
    operand: MachineOperand,
    role: OperandRole,
) -> tuple[str | None, str | None]:
    form = _MACHINE_FORMS[spec.form_name]
    operand_ordinal = _operand_ordinal(form, operand)
    if role is OperandRole.RESULT:
        return None, _register_timing_event(spec, operand_ordinal, "write")
    return _register_timing_event(spec, operand_ordinal, "read"), None


def _low_operand(
    spec: _DescriptorSpec,
    operand: MachineOperand,
    role: OperandRole,
) -> Operand:
    read_stage, ready_stage = _operand_stages(spec, operand)
    read_event, write_event = _operand_timing_events(spec, operand, role)
    encoded_field_names = {
        field.name for field in _INSTRUCTION_ENCODINGS[spec.form_name].fields
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
        reg_alts=(RegClassAlt(_operand_register_class(spec, operand)),),
        encoding_field_id=encoding_field_id,
        encoding_adapter_id=encoding_adapter_id,
        read_stage=read_stage,
        ready_stage=ready_stage,
        read_event=read_event,
        write_event=write_event,
    )


def _implicit_operands(spec: _DescriptorSpec) -> tuple[Operand, ...]:
    form = _MACHINE_FORMS[spec.form_name]
    result: list[Operand] = []
    for register_name in form.implicit_defs:
        operand_ordinal = (
            len(form.outputs)
            + len(form.inputs)
            + form.implicit_defs.index(register_name)
        )
        read_stage, ready_stage = _implicit_operand_stage(
            spec, register_name, is_definition=True
        )
        result.append(
            Operand(
                field_name=f"implicit_def_{register_name.lower()}",
                role=OperandRole.IMPLICIT,
                reg_alts=(
                    RegClassAlt(
                        _implicit_register_class_name(register_name),
                        flags=(RegClassAltFlag.PHYSICAL_ONLY,),
                    ),
                ),
                flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
                read_stage=read_stage,
                ready_stage=ready_stage,
                write_event=_register_timing_event(spec, operand_ordinal, "write"),
            )
        )
    for register_name in form.implicit_uses:
        operand_ordinal = (
            len(form.outputs)
            + len(form.inputs)
            + len(form.implicit_defs)
            + form.implicit_uses.index(register_name)
        )
        read_stage, ready_stage = _implicit_operand_stage(
            spec, register_name, is_definition=False
        )
        result.append(
            Operand(
                field_name=f"implicit_use_{register_name.lower()}",
                role=OperandRole.IMPLICIT,
                reg_alts=(
                    RegClassAlt(
                        _implicit_register_class_name(register_name),
                        flags=(RegClassAltFlag.PHYSICAL_ONLY,),
                    ),
                ),
                flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_READ),
                read_stage=read_stage,
                ready_stage=ready_stage,
                read_event=_register_timing_event(spec, operand_ordinal, "read"),
            )
        )
    return tuple(result)


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


def _memory_event_name(access: str, memory: MemoryCycles) -> str:
    cycle_segment = "_".join(str(cycle) for cycle in memory.cycles)
    return f"{_TARGET_KEY}.memory.{access}.c{cycle_segment}"


def _memory_timing_event(spec: _DescriptorSpec, access: str) -> str:
    memory = _itinerary(spec).memory
    if memory is None:
        raise ValueError(
            f"{spec.form_name}: memory operation uses non-memory itinerary "
            f"{spec.itinerary}"
        )
    return _memory_event_name(access, memory)


def _effects(spec: _DescriptorSpec, form: MachineForm) -> tuple[Effect, ...]:
    result = []
    if has_property(form, "mayLoad"):
        result.append(
            Effect(
                EffectKind.READ,
                MemorySpace.WORKGROUP,
                flags=(EffectFlag.DEPENDENCY,),
                timing_event=_memory_timing_event(spec, "read"),
            )
        )
    if has_property(form, "mayStore"):
        result.append(
            Effect(
                EffectKind.WRITE,
                MemorySpace.WORKGROUP,
                flags=(EffectFlag.DEPENDENCY,),
                timing_event=_memory_timing_event(spec, "write"),
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
    if (
        not has_property(form, "mayLoad")
        and not has_property(form, "mayStore")
        and form.control_flow_kind is None
        and form.name != "RET"
    ):
        result.append(DescriptorFlag.DEAD_REMOVABLE)
    return tuple(result)


def _instruction_classes(
    spec: _DescriptorSpec,
    form: MachineForm,
) -> tuple[InstructionClass, ...]:
    if has_property(form, "mayLoad"):
        return (InstructionClass.LOCAL_MEMORY,)
    if has_property(form, "mayStore"):
        return (InstructionClass.LOCAL_MEMORY,)
    if form.control_flow_kind is not None:
        return (InstructionClass.CONTROL,)
    if form.name == "NOP":
        return (InstructionClass.CONTROL,)
    register_operands = tuple(
        operand
        for operand in (*form.outputs, *form.inputs)
        if operand.kind is not MachineOperandKind.IMMEDIATE
    )
    is_vector = any(
        _MACHINE_CLASSES[
            _operand_storage_machine_class(spec, operand)
        ].layout.register_size_bits
        >= 128
        for operand in register_operands
    )
    result = [InstructionClass.VECTOR_ALU if is_vector else InstructionClass.SCALAR_ALU]
    if has_property(form, "isMoveImm"):
        result.append(InstructionClass.REGISTER_MOVE)
    return tuple(result)


def _constraints(
    form: MachineForm,
    explicit_operands: tuple[MachineOperand, ...],
) -> tuple[Constraint, ...]:
    operand_indices = {
        operand.name: operand_index
        for operand_index, operand in enumerate(explicit_operands)
    }
    return tuple(
        Constraint(
            ConstraintKind.TIED,
            operand_indices[tie.definition],
            operand_indices[tie.use],
        )
        for tie in form.ties
    )


def _descriptor(spec: _DescriptorSpec) -> Descriptor:
    form = _MACHINE_FORMS[spec.form_name]
    if spec.form_name != form.name:
        raise ValueError(f"descriptor spec selected the wrong machine form {form.name}")
    if spec.itinerary != form.itinerary and not spec.storage_overrides:
        raise ValueError(
            f"{form.name}: itinerary override {spec.itinerary} requires a storage "
            "specialization"
        )
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
    explicit_register_operands = (*register_outputs, *register_inputs)
    mnemonic = _ASM_MNEMONIC_BY_FORM.get(
        spec.form_name, form.assembly.split("\t", 1)[0].strip()
    )
    descriptor = Descriptor(
        key=spec.key,
        mnemonic=mnemonic,
        semantic_tag=spec.semantic_tag,
        operands=(
            *(
                _low_operand(spec, operand, OperandRole.RESULT)
                for operand in register_outputs
            ),
            *(
                _low_operand(spec, operand, OperandRole.OPERAND)
                for operand in register_inputs
            ),
            *_implicit_operands(spec),
        ),
        immediates=tuple(
            _immediate(spec.form_name, operand) for operand in immediate_inputs
        ),
        schedule_class=_SCHEDULE_CLASS_NAMES[(spec.form_name, spec.itinerary)],
        op_kind=spec.op_kind,
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
        effects=_effects(spec, form),
        constraints=_constraints(form, explicit_register_operands),
        encoding_id=_INSTRUCTION_IDS[spec.form_name],
        flags=_descriptor_flags(form),
        instruction_classes=_instruction_classes(spec, form),
    )
    expected_fields = {
        field.name for field in _INSTRUCTION_ENCODINGS[spec.form_name].fields
    }
    encoded_fields = {
        row.field_name
        for row in (*descriptor.operands, *descriptor.immediates)
        if row.encoding_field_id
    }
    if encoded_fields != expected_fields:
        raise ValueError(
            f"{spec.form_name}: Low descriptor fields {sorted(encoded_fields)} "
            f"do not match instruction fields {sorted(expected_fields)}"
        )
    if len(expected_fields) > 16:
        raise ValueError(
            f"{spec.form_name}: instruction field count exceeds native storage"
        )
    return descriptor


_SLOT_RESOURCE_KINDS = {
    "alu": ResourceKind.SCALAR_ALU,
    "lda": ResourceKind.LOAD,
    "ldb": ResourceKind.LOAD,
    "lng": ResourceKind.SCALAR_ALU,
    "mv": ResourceKind.VECTOR_ALU,
    "nop": ResourceKind.CONTROL,
    "st": ResourceKind.STORE,
    "vec": ResourceKind.VECTOR_ALU,
}


def _slot_resource_name(slot: str) -> str:
    return f"{_TARGET_KEY}.slot.{slot}"


def _pipeline_resource_name(resource: str) -> str:
    return f"{_TARGET_KEY}.pipeline.{resource.lower()}"


_RESOURCES = (
    *(
        Resource(_slot_resource_name(slot), 1, kind)
        for slot, kind in sorted(_SLOT_RESOURCE_KINDS.items())
    ),
    *(
        Resource(
            _pipeline_resource_name(resource),
            1,
            ResourceKind.PIPELINE,
        )
        for resource in CORE_SCHEDULE_TABLE.resources
    ),
)

# The complete endpoint domains make descriptor growth table-selective: adding
# a form cannot require another hand-authored dependency-timing case.
_REGISTER_ENDPOINTS = tuple(
    sorted(
        {
            (cycle, bypass_class(itinerary, operand_index))
            for itinerary in CORE_SCHEDULE_TABLE.itineraries
            for operand_index, cycle in enumerate(itinerary.operand_cycles)
        },
        key=lambda endpoint: (endpoint[0], endpoint[1] or ""),
    )
)
_MEMORY_ENDPOINTS = tuple(
    MemoryCycles(cycles)
    for cycles in sorted(
        {
            itinerary.memory.cycles
            for itinerary in CORE_SCHEDULE_TABLE.itineraries
            if itinerary.memory is not None
        }
    )
)

_TIMING_EVENTS = tuple(
    TimingEvent(name)
    for name in sorted(
        {
            *(
                _register_event_name(access, cycle, bypass)
                for cycle, bypass in _REGISTER_ENDPOINTS
                for access in ("read", "write")
            ),
            *(
                _memory_event_name(access, memory)
                for memory in _MEMORY_ENDPOINTS
                for access in ("read", "write")
            ),
        }
    )
)


def _endpoint_itinerary(endpoint: tuple[int, str | None]) -> Itinerary:
    cycle, bypass = endpoint
    return Itinerary(
        name="endpoint",
        stages=(),
        operand_cycles=(cycle,),
        bypasses=(bypass or "NoBypass",),
    )


def _event_separations() -> tuple[EventSeparation, ...]:
    result = []
    endpoint_itineraries = {
        endpoint: _endpoint_itinerary(endpoint) for endpoint in _REGISTER_ENDPOINTS
    }
    for producer in _REGISTER_ENDPOINTS:
        producer_itinerary = endpoint_itineraries[producer]
        for consumer in _REGISTER_ENDPOINTS:
            consumer_itinerary = endpoint_itineraries[consumer]
            result.extend(
                (
                    EventSeparation(
                        _register_event_name("write", *producer),
                        _register_event_name("read", *consumer),
                        dependency_separation(
                            producer_itinerary,
                            0,
                            consumer_itinerary,
                            0,
                            DependencyKind.RAW,
                        ),
                        ModelQuality.EXACT,
                    ),
                    EventSeparation(
                        _register_event_name("read", *producer),
                        _register_event_name("write", *consumer),
                        dependency_separation(
                            producer_itinerary,
                            0,
                            consumer_itinerary,
                            0,
                            DependencyKind.WAR,
                        ),
                        ModelQuality.EXACT,
                    ),
                    EventSeparation(
                        _register_event_name("write", *producer),
                        _register_event_name("write", *consumer),
                        dependency_separation(
                            producer_itinerary,
                            0,
                            consumer_itinerary,
                            0,
                            DependencyKind.WAW,
                        ),
                        ModelQuality.EXACT,
                    ),
                )
            )
    for producer in _MEMORY_ENDPOINTS:
        producer_itinerary = Itinerary(
            name="memory_producer",
            stages=(),
            operand_cycles=(),
            bypasses=(),
            memory=producer,
        )
        for consumer in _MEMORY_ENDPOINTS:
            consumer_itinerary = Itinerary(
                name="memory_consumer",
                stages=(),
                operand_cycles=(),
                bypasses=(),
                memory=consumer,
            )
            cycles = memory_separation(producer_itinerary, consumer_itinerary)
            result.extend(
                EventSeparation(
                    _memory_event_name(producer_access, producer),
                    _memory_event_name(consumer_access, consumer),
                    cycles,
                    ModelQuality.EXACT,
                )
                for producer_access in ("read", "write")
                for consumer_access in ("read", "write")
            )
    return tuple(
        sorted(
            result,
            key=lambda row: (row.producer_event, row.consumer_event),
        )
    )


_EVENT_SEPARATIONS = _event_separations()


def _canonical_itinerary_names() -> dict[tuple[object, ...], str]:
    """Names each deduplicated payload by its first sorted source alias."""

    result = {itinerary_payload(NO_ITINERARY): NO_ITINERARY.name}
    for itinerary in CORE_SCHEDULE_TABLE.itineraries:
        result.setdefault(itinerary_payload(itinerary), itinerary.name)
    return result


_CANONICAL_ITINERARY_NAMES = _canonical_itinerary_names()


def _schedule_flags(form: MachineForm) -> tuple[ScheduleClassFlag, ...]:
    result = []
    if has_property(form, "mayLoad"):
        result.append(ScheduleClassFlag.MAY_LOAD)
    if has_property(form, "mayStore"):
        result.append(ScheduleClassFlag.MAY_STORE)
    if has_property(form, "isCall"):
        result.append(ScheduleClassFlag.MAY_CALL)
    if form.control_flow_kind is not None:
        result.append(ScheduleClassFlag.CONTROL)
    return tuple(result)


def _schedule_issue_uses(
    spec: _DescriptorSpec,
    itinerary: Itinerary,
) -> tuple[IssueUse, ...]:
    slot = _INSTRUCTION_ENCODINGS[spec.form_name].slot
    result = [IssueUse(_slot_resource_name(slot), 1, 1)]
    for use in pipeline_uses(itinerary):
        if len(use.resources) != 1:
            raise ValueError(
                f"{itinerary.name}: AIE2P pipeline stage must name exactly "
                "one physical resource"
            )
        result.append(
            IssueUse(
                _pipeline_resource_name(use.resources[0]),
                use.cycles,
                1,
                stage=use.start_cycle,
                kind=(
                    IssueUseKind.REQUIRED
                    if use.kind is PipelineStageKind.REQUIRED
                    else IssueUseKind.RESERVED
                ),
            )
        )
    return tuple(result)


def _schedule_class(spec: _DescriptorSpec) -> ScheduleClass:
    form = _MACHINE_FORMS[spec.form_name]
    itinerary = _itinerary(spec)
    has_memory_effect = has_property(form, "mayLoad") or has_property(form, "mayStore")
    if has_memory_effect != (itinerary.memory is not None):
        raise ValueError(
            f"{form.name}: memory properties disagree with itinerary {itinerary.name}"
        )
    flags = _schedule_flags(form)
    instruction_classes = _instruction_classes(spec, form)
    slot = _INSTRUCTION_ENCODINGS[form.name].slot
    canonical_itinerary = _CANONICAL_ITINERARY_NAMES[
        itinerary_payload(itinerary)
    ].lower()
    qualifiers = (
        slot,
        *(flag.name.lower() for flag in flags),
        *(instruction_class.name.lower() for instruction_class in instruction_classes),
    )
    name = ".".join(
        (
            _TARGET_KEY,
            "schedule",
            canonical_itinerary,
            *qualifiers,
        )
    )
    issue_uses = _schedule_issue_uses(spec, itinerary)
    overall_cycles = [
        1,
        *itinerary.operand_cycles,
        *(use.stage + use.cycles for use in issue_uses),
    ]
    if itinerary.memory is not None:
        overall_cycles.extend(itinerary.memory.cycles)
    implicit_def_start = len(form.outputs) + len(form.inputs)
    definition_cycles = (
        *itinerary.operand_cycles[: len(form.outputs)],
        *itinerary.operand_cycles[
            implicit_def_start : implicit_def_start + len(form.implicit_defs)
        ],
    )
    return ScheduleClass(
        name,
        LatencyKind.EXACT,
        ModelQuality.EXACT,
        latency_cycles=max(overall_cycles),
        minimum_issue_separation_cycles=max((1, *definition_cycles)),
        issue_uses=issue_uses,
        flags=flags,
        instruction_classes=instruction_classes,
    )


def _schedule_classes() -> tuple[
    dict[tuple[str, str], str],
    tuple[ScheduleClass, ...],
]:
    names = {}
    classes = {}
    for spec in _DESCRIPTOR_SPECS:
        schedule_class = _schedule_class(spec)
        key = (spec.form_name, spec.itinerary)
        names[key] = schedule_class.name
        previous = classes.setdefault(schedule_class.name, schedule_class)
        if previous != schedule_class:
            raise ValueError(f"{schedule_class.name}: schedule-class name collision")
    return names, tuple(classes[name] for name in sorted(classes))


_SCHEDULE_CLASS_NAMES, _SCHEDULE_CLASSES = _schedule_classes()


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
    descriptors=tuple(_descriptor(spec) for spec in _DESCRIPTOR_SPECS),
    requires_explicit_asm_surface=True,
)
