# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P Low descriptors derived from semantic, machine, and schedule tables."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from loom.target.arch.amd.xdna.aie.machine import (
    MachineForm,
    MachineOperand,
    MachineOperandKind,
    RegisterLayout,
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
from loom.target.arch.amd.xdna.aie2p import core_descriptor_specs as descriptor_specs
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
    EncodingFieldValue,
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
    PhysicalRegisterView,
    RegClass,
    RegClassAlt,
    RegClassAltFlag,
    RegClassFlag,
    RegisterPackingResource,
    RegisterPackingResourceMember,
    Resource,
    ResourceKind,
    ScheduleClass,
    ScheduleClassFlag,
    SpillSlotSpace,
    TimingEvent,
)

_MACHINE_REGISTERS = {
    register.name: register for register in CORE_MACHINE_TABLE.physical_registers
}
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

# LLVM's mW*/mX* names describe instruction-operand encoding roles, not
# distinct storage domains. W registers are the architectural 256-bit storage
# units. Each X register is an ordered pair of W subregisters and remains the
# aggregate encoding domain for 512-bit instructions.
_LOW_REGISTER_CLASS_BY_MACHINE_CLASS = {
    "eL": "eR",
    "mWa": "VEC256",
    "mWb": "VEC256",
    "mWs": "VEC256",
    "mXa": "VEC256",
    "mXb": "VEC256",
    "mXm": "VEC256",
    "mXn": "VEC256",
    "mXs": "VEC256",
    "mXv": "VEC256",
    "mXw": "VEC256",
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


def _operand_override_map(
    spec: descriptor_specs._DescriptorSpec,
    rows: tuple[tuple[str, str], ...],
    description: str,
) -> dict[str, str]:
    names = [name for name, _ in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"{spec.form_name}: {description} names must be unique")
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
    explicit_names = {operand.name for operand in (*form.outputs, *form.inputs)}
    unknown_names = set(names) - explicit_names
    if unknown_names:
        raise ValueError(
            f"{spec.form_name}: {description} name unknown operands "
            f"{sorted(unknown_names)}"
        )
    return dict(rows)


def _operand_machine_class(operand: MachineOperand) -> str:
    if operand.kind is MachineOperandKind.REGISTER_CLASS:
        return operand.type_name
    if operand.kind is MachineOperandKind.REGISTER_ADAPTER:
        return _MACHINE_ADAPTERS[operand.type_name].register_class
    raise ValueError(f"{operand.name}: immediate is not a register operand")


def _operand_encoding_machine_class(
    spec: descriptor_specs._DescriptorSpec, operand: MachineOperand
) -> str:
    """Returns the physical domain encoded by one selected descriptor operand."""

    adapter_overrides = _operand_override_map(
        spec, spec.encoding_adapter_overrides, "encoding-adapter overrides"
    )
    adapter_name = adapter_overrides.get(operand.name)
    if adapter_name is None:
        return _operand_machine_class(operand)
    adapter = _MACHINE_ADAPTERS.get(adapter_name)
    if adapter is None:
        raise ValueError(
            f"{spec.form_name}.{operand.name}: unknown encoding adapter {adapter_name}"
        )
    native_class_name = _operand_machine_class(operand)
    native_encoding_values = (
        {
            value
            for _, value in _MACHINE_ADAPTERS[
                operand.type_name
            ].effective_register_encodings
        }
        if operand.kind is MachineOperandKind.REGISTER_ADAPTER
        else {
            _MACHINE_REGISTERS[register].hardware_encoding
            for register in _MACHINE_CLASSES[native_class_name].candidates
        }
    )
    override_encoding_values = {
        value for _, value in adapter.effective_register_encodings
    }
    if not override_encoding_values <= native_encoding_values:
        raise ValueError(
            f"{spec.form_name}.{operand.name}: adapter {adapter_name} emits values "
            f"outside the native {operand.type_name} encoding domain"
        )
    return adapter.register_class


def _storage_unit_count(source: str, target: str) -> int:
    source_bits = _MACHINE_CLASSES[source].layout.register_size_bits
    target_bits = _MACHINE_CLASSES[target].layout.register_size_bits
    if source_bits < target_bits or source_bits % target_bits:
        raise ValueError(
            f"{source}: {source_bits}-bit native storage cannot use "
            f"{target_bits}-bit Low units from {target}"
        )
    return source_bits // target_bits


def _is_ordered_candidate_subset(source: str, target: str) -> bool:
    source_candidates = _MACHINE_CLASSES[source].candidates
    target_candidates = _MACHINE_CLASSES[target].candidates
    selected_candidates = tuple(
        candidate
        for candidate in source_candidates
        if candidate in set(target_candidates)
    )
    return selected_candidates == target_candidates


def _aggregate_register_units(
    source_class_name: str,
    target_class_name: str,
    register_name: str,
) -> tuple[str, ...]:
    unit_count = _storage_unit_count(source_class_name, target_class_name)
    target_candidates = set(_MACHINE_CLASSES[target_class_name].candidates)

    def collect_units(name: str) -> tuple[str, ...]:
        if name in target_candidates:
            return (name,)
        register = _MACHINE_REGISTERS[name]
        if not register.subregisters:
            raise ValueError(
                f"{register_name}: {name} cannot be decomposed into storage "
                f"class {target_class_name}"
            )
        units = tuple(
            unit
            for subregister_name in register.subregisters
            for unit in collect_units(subregister_name)
        )
        covered_atomic_units = tuple(
            sorted(
                atomic_unit
                for unit_name in units
                for atomic_unit in _MACHINE_REGISTERS[unit_name].atomic_units
            )
        )
        if covered_atomic_units != register.atomic_units:
            raise ValueError(
                f"{register_name}: recursive subregisters do not exactly cover "
                f"{name} atomic storage"
            )
        return units

    units = collect_units(register_name)
    if len(units) != unit_count:
        raise ValueError(
            f"{register_name}: expected {unit_count} named subregister units for "
            f"{source_class_name} as {target_class_name}, found "
            f"{len(units)}"
        )
    if len(units) != len(set(units)) or not set(units) <= target_candidates:
        raise ValueError(
            f"{register_name}: subregisters {units} are outside "
            f"storage class {target_class_name}"
        )
    register = _MACHINE_REGISTERS[register_name]
    covered_units = tuple(
        sorted(
            atomic_unit
            for unit_name in units
            for atomic_unit in _MACHINE_REGISTERS[unit_name].atomic_units
        )
    )
    if covered_units != register.atomic_units:
        raise ValueError(
            f"{register_name}: named subregisters do not exactly cover aggregate "
            "atomic storage"
        )
    return units


def _validate_aggregate_storage_domain(source: str, target: str) -> None:
    source_class = _MACHINE_CLASSES[source]
    target_class = _MACHINE_CLASSES[target]
    flattened_units = tuple(
        unit
        for register_name in source_class.candidates
        for unit in _aggregate_register_units(source, target, register_name)
    )
    if len(flattened_units) != len(set(flattened_units)):
        raise ValueError(f"{source}: aggregate candidates reuse {target} units")
    if set(flattened_units) != set(target_class.candidates):
        raise ValueError(
            f"{source}: aggregate candidates do not exactly cover {target}"
        )


def _operand_storage_machine_class(
    spec: descriptor_specs._DescriptorSpec,
    operand: MachineOperand,
) -> str:
    machine_class = _operand_encoding_machine_class(spec, operand)
    storage_overrides = _operand_override_map(
        spec, spec.storage_overrides, "storage overrides"
    )
    register_parts = _operand_override_map(
        spec, spec.operand_register_parts, "register-part overrides"
    )
    adapter_overrides = _operand_override_map(
        spec, spec.encoding_adapter_overrides, "encoding-adapter overrides"
    )
    if not set(register_parts) <= set(adapter_overrides):
        raise ValueError(
            f"{spec.form_name}: register-part and encoding-adapter overrides "
            "must encode every projected operand"
        )
    low_class = storage_overrides.get(
        operand.name,
        _LOW_REGISTER_CLASS_BY_MACHINE_CLASS.get(machine_class, machine_class),
    )
    source = _MACHINE_CLASSES[machine_class]
    target = _MACHINE_CLASSES[low_class]
    has_register_projection = operand.name in register_parts
    unit_count = (
        1
        if has_register_projection
        or (
            operand.name in storage_overrides
            and _is_ordered_candidate_subset(machine_class, low_class)
        )
        else _storage_unit_count(machine_class, low_class)
    )
    if unit_count == 1:
        selected_candidates = tuple(
            candidate
            for candidate in source.candidates
            if candidate in set(target.candidates)
        )
        if selected_candidates != target.candidates and not has_register_projection:
            raise ValueError(
                f"{spec.form_name}.{operand.name}: Low class {low_class} is not "
                f"an ordered storage subset of {machine_class}"
            )
    else:
        if has_register_projection:
            raise ValueError(
                f"{spec.form_name}.{operand.name}: aggregate storage cannot also "
                "use a register-part projection"
            )
        _validate_aggregate_storage_domain(machine_class, low_class)
    if has_register_projection:
        part_name = register_parts[operand.name]
        part = descriptor_specs._REGISTER_PARTS_BY_NAME.get(part_name)
        if part is None:
            raise ValueError(
                f"{spec.form_name}.{operand.name}: unknown register part {part_name}"
            )
        if part.reg_class != _low_register_class_name(low_class):
            raise ValueError(
                f"{spec.form_name}.{operand.name}: register part {part_name} does "
                f"not belong to Low class {low_class}"
            )
        adapter_name = adapter_overrides[operand.name]
        adapter = _MACHINE_ADAPTERS.get(adapter_name)
        if adapter is None:
            raise ValueError(
                f"{spec.form_name}.{operand.name}: unknown encoding adapter "
                f"{adapter_name}"
            )
        adapter_class = _MACHINE_CLASSES[adapter.register_class]
        if adapter_class.candidates != target.candidates:
            raise ValueError(
                f"{spec.form_name}.{operand.name}: adapter {adapter_name} encodes "
                f"{adapter.register_class}, not the {low_class} candidate domain"
            )
        encoded_registers = {
            register for register, _ in adapter.effective_register_encodings
        }
        if set(target.candidates) != encoded_registers:
            raise ValueError(
                f"{spec.form_name}.{operand.name}: adapter {adapter_name} does not "
                f"exactly encode Low class {low_class}"
            )
    else:
        adapter_name = adapter_overrides.get(
            operand.name,
            operand.type_name
            if operand.kind is MachineOperandKind.REGISTER_ADAPTER
            else None,
        )
        if adapter_name is None:
            return low_class
        encoded_registers = {
            register
            for register, _ in _MACHINE_ADAPTERS[
                adapter_name
            ].effective_register_encodings
        }
        required_registers = (
            set(target.candidates) if unit_count == 1 else set(source.candidates)
        )
        if not required_registers.issubset(encoded_registers):
            raise ValueError(
                f"{spec.form_name}.{operand.name}: adapter {adapter_name} "
                f"does not encode the physical domain for {low_class} x{unit_count}"
            )
    return low_class


def _operand_unit_count(
    spec: descriptor_specs._DescriptorSpec, operand: MachineOperand
) -> int:
    register_parts = _operand_override_map(
        spec, spec.operand_register_parts, "register-part overrides"
    )
    if operand.name in register_parts:
        return 1
    machine_class = _operand_encoding_machine_class(spec, operand)
    storage_class = _operand_storage_machine_class(spec, operand)
    storage_overrides = _operand_override_map(
        spec, spec.storage_overrides, "storage overrides"
    )
    if operand.name in storage_overrides and _is_ordered_candidate_subset(
        machine_class, storage_class
    ):
        return 1
    return _storage_unit_count(machine_class, storage_class)


def _low_register_class_name(machine_class: str) -> str:
    return f"aie2p.{machine_class.lower()}"


def _operand_register_class(
    spec: descriptor_specs._DescriptorSpec, operand: MachineOperand
) -> str:
    return _low_register_class_name(_operand_storage_machine_class(spec, operand))


_EXPLICIT_STORAGE_MACHINE_CLASS_NAMES = tuple(
    sorted(
        {
            _operand_storage_machine_class(spec, operand)
            for spec in descriptor_specs._DESCRIPTOR_SPECS
            for operand in (
                *descriptor_specs._MACHINE_FORMS[spec.form_name].outputs,
                *descriptor_specs._MACHINE_FORMS[spec.form_name].inputs,
            )
            if operand.kind is not MachineOperandKind.IMMEDIATE
        }
    )
)

_IMPLICIT_REGISTER_LAYOUT_OVERRIDES = {
    # The link register participates in both the 20-bit address and 32-bit
    # scalar domains. RET consumes the architectural return address.
    "lr": "mLRa",
}

_IMPLICIT_REGISTER_LAYOUTS = {
    # Part-word stores read and rewrite their containing word. LLVM represents
    # the resulting address-data-store dependency with a non-allocatable
    # singleton register that has no RegisterClass or data payload.
    "pe2_ads": RegisterLayout(1, 1, 1, 1),
}


def _selected_singleton_machine_class_name(register_name: str) -> str | None:
    matches = tuple(
        machine_class_name
        for machine_class_name in _EXPLICIT_STORAGE_MACHINE_CLASS_NAMES
        if _MACHINE_CLASSES[machine_class_name].candidates == (register_name,)
    )
    if len(matches) > 1:
        raise ValueError(
            f"implicit register {register_name} has ambiguous selected singleton "
            f"classes {list(matches)}"
        )
    return matches[0] if matches else None


def _implicit_register_class_name(register_name: str) -> str:
    selected_class = _selected_singleton_machine_class_name(register_name)
    if selected_class is not None:
        return _low_register_class_name(selected_class)
    return f"aie2p.state.{register_name.lower()}"


def _implicit_register_layout(register_name: str) -> RegisterLayout:
    selected_class = _selected_singleton_machine_class_name(register_name)
    if selected_class is not None:
        return _MACHINE_CLASSES[selected_class].layout
    direct_layout = _IMPLICIT_REGISTER_LAYOUTS.get(register_name)
    if direct_layout is not None:
        return direct_layout
    override = _IMPLICIT_REGISTER_LAYOUT_OVERRIDES.get(register_name)
    if override is not None:
        machine_class = _MACHINE_CLASSES[override]
        if machine_class.candidates != (register_name,):
            raise ValueError(
                f"{override}: implicit state layout override must name only "
                f"{register_name}"
            )
        return machine_class.layout

    candidate_classes = tuple(
        machine_class
        for machine_class in CORE_MACHINE_TABLE.register_classes
        if register_name in machine_class.candidates
    )
    if not candidate_classes:
        raise ValueError(
            f"implicit register {register_name} is absent from every machine class"
        )
    layouts = {machine_class.layout for machine_class in candidate_classes}
    if len(layouts) != 1:
        raise ValueError(
            f"implicit register {register_name} has ambiguous machine layouts in "
            f"{[machine_class.name for machine_class in candidate_classes]}"
        )
    return next(iter(layouts))


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
                full_register_part_mask=(
                    0x3 if machine_name in ("eLPredicate", "eWL", "VEC256") else 0x1
                ),
                physical_registers=machine_class.candidates,
            )
        )
    next_bank_id = len(result) + 1
    selected_implicit_registers = sorted(
        {
            register_name
            for spec in descriptor_specs._DESCRIPTOR_SPECS
            for register_name in (
                *descriptor_specs._MACHINE_FORMS[spec.form_name].implicit_defs,
                *descriptor_specs._MACHINE_FORMS[spec.form_name].implicit_uses,
            )
        }
    )
    for register_name in selected_implicit_registers:
        register_class_name = _implicit_register_class_name(register_name)
        if any(row.name == register_class_name for row in result):
            continue
        layout = _implicit_register_layout(register_name)
        result.append(
            RegClass(
                name=register_class_name,
                alloc_unit_bits=layout.register_size_bits,
                spill_slot_space=SpillSlotSpace.PRIVATE,
                flags=(
                    RegClassFlag.PHYSICAL,
                    RegClassFlag.UNSPILLABLE,
                    RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS,
                ),
                target_bank_id=next_bank_id,
                physical_registers=(register_name,),
            )
        )
        next_bank_id += 1
    return tuple(result)


def _physical_registers() -> tuple[PhysicalRegister, ...]:
    return tuple(
        PhysicalRegister(register.name, register.atomic_units)
        for register in CORE_MACHINE_TABLE.physical_registers
    )


def _physical_register_views() -> tuple[PhysicalRegisterView, ...]:
    views: dict[tuple[str, str], PhysicalRegisterView] = {}
    for spec in descriptor_specs._DESCRIPTOR_SPECS:
        form = descriptor_specs._MACHINE_FORMS[spec.form_name]
        for operand in (*form.outputs, *form.inputs):
            if operand.kind is MachineOperandKind.IMMEDIATE:
                continue
            source_class_name = _operand_encoding_machine_class(spec, operand)
            target_class_name = _operand_storage_machine_class(spec, operand)
            if _operand_unit_count(spec, operand) == 1:
                continue
            reg_class_name = _low_register_class_name(target_class_name)
            for register_name in _MACHINE_CLASSES[source_class_name].candidates:
                view = PhysicalRegisterView(
                    physical_register=register_name,
                    reg_class=reg_class_name,
                    units=_aggregate_register_units(
                        source_class_name,
                        target_class_name,
                        register_name,
                    ),
                )
                key = (view.physical_register, view.reg_class)
                previous = views.setdefault(key, view)
                if previous != view:
                    raise ValueError(
                        f"{register_name}: inconsistent aggregate register views "
                        f"{previous.units} and {view.units}"
                    )
    return tuple(views[key] for key in sorted(views))


def _register_packing_resources() -> tuple[RegisterPackingResource, ...]:
    """Describes instantaneous capacity shared across register classes."""

    x_register_count = len(_MACHINE_CLASSES["mXm"].candidates)
    if len(_MACHINE_CLASSES["VEC256"].candidates) != x_register_count * 2:
        raise ValueError(
            "AIE2P VEC256 does not cover both W halves of every X register"
        )
    if len(_MACHINE_CLASSES["eWL"].candidates) != x_register_count:
        raise ValueError("AIE2P eWL does not cover one W half of every X register")

    physical_registers = {
        register.name: register for register in CORE_MACHINE_TABLE.physical_registers
    }
    scalar_atomic_units = frozenset(
        atomic_unit
        for register_name in _MACHINE_CLASSES["eR"].candidates
        for atomic_unit in physical_registers[register_name].atomic_units
    )
    scalar_members: list[RegisterPackingResourceMember] = []
    for machine_class_name in _EXPLICIT_STORAGE_MACHINE_CLASS_NAMES:
        machine_class = _MACHINE_CLASSES[machine_class_name]
        candidate_atomic_unit_counts = {
            len(physical_registers[register_name].atomic_units)
            for register_name in machine_class.candidates
        }
        if len(candidate_atomic_unit_counts) != 1 or not all(
            set(physical_registers[register_name].atomic_units) <= scalar_atomic_units
            for register_name in machine_class.candidates
        ):
            continue
        scalar_members.append(
            RegisterPackingResourceMember(
                _low_register_class_name(machine_class_name),
                resource_unit_count=next(iter(candidate_atomic_unit_counts)),
            )
        )
    if not scalar_members:
        raise ValueError("AIE2P scalar register packing resource has no members")
    return (
        RegisterPackingResource(
            name=f"{descriptor_specs._TARGET_KEY}.register.x.pairs",
            capacity=x_register_count,
            members=(
                RegisterPackingResourceMember("aie2p.ewl"),
                RegisterPackingResourceMember(
                    "aie2p.vec256",
                    register_unit_count=2,
                ),
            ),
        ),
        RegisterPackingResource(
            name=f"{descriptor_specs._TARGET_KEY}.register.scalar.units",
            capacity=len(scalar_atomic_units),
            members=tuple(scalar_members),
        ),
    )


_ITINERARIES = {
    NO_ITINERARY.name: NO_ITINERARY,
    **{itinerary.name: itinerary for itinerary in CORE_SCHEDULE_TABLE.itineraries},
}


def _itinerary(spec: descriptor_specs._DescriptorSpec) -> Itinerary:
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
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
    return f"{descriptor_specs._TARGET_KEY}.operand.{access}.c{cycle}.{bypass_segment}"


def _register_timing_event(
    spec: descriptor_specs._DescriptorSpec,
    operand_ordinal: int,
    access: str,
) -> str:
    itinerary = _itinerary(spec)
    cycle = itinerary.operand_cycles[operand_ordinal]
    bypass = bypass_class(itinerary, operand_ordinal)
    return _register_event_name(access, cycle, bypass)


def _operand_stages(
    spec: descriptor_specs._DescriptorSpec,
    operand: MachineOperand,
) -> tuple[int, int]:
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
    operand_ordinal = _operand_ordinal(form, operand)
    cycle = _itinerary(spec).operand_cycles[operand_ordinal]
    return (0, cycle) if operand_ordinal < len(form.outputs) else (cycle, 0)


def _implicit_operand_stage(
    spec: descriptor_specs._DescriptorSpec,
    register_name: str,
    *,
    is_definition: bool,
) -> tuple[int, int]:
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
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
    spec: descriptor_specs._DescriptorSpec,
    operand: MachineOperand,
    role: OperandRole,
) -> tuple[str | None, str | None]:
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
    operand_ordinal = _operand_ordinal(form, operand)
    if role is OperandRole.RESULT:
        return None, _register_timing_event(spec, operand_ordinal, "write")
    return _register_timing_event(spec, operand_ordinal, "read"), None


def _low_operand(
    spec: descriptor_specs._DescriptorSpec,
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
    adapter_overrides = _operand_override_map(
        spec, spec.encoding_adapter_overrides, "encoding-adapter overrides"
    )
    adapter_name = adapter_overrides.get(
        operand.name,
        operand.type_name
        if operand.kind is MachineOperandKind.REGISTER_ADAPTER
        else None,
    )
    encoding_adapter_id = (
        _ADAPTER_IDS[adapter_name] if encoding_field_id and adapter_name else 0
    )
    register_parts = _operand_override_map(
        spec, spec.operand_register_parts, "register-part overrides"
    )
    return Operand(
        field_name=operand.name,
        role=role,
        reg_alts=(RegClassAlt(_operand_register_class(spec, operand)),),
        unit_count=_operand_unit_count(spec, operand),
        encoding_field_id=encoding_field_id,
        encoding_adapter_id=encoding_adapter_id,
        register_part=register_parts.get(operand.name),
        read_stage=read_stage,
        ready_stage=ready_stage,
        read_event=read_event,
        write_event=write_event,
    )


def _storage_continuation_operand(
    spec: descriptor_specs._DescriptorSpec,
    register_outputs: tuple[MachineOperand, ...],
) -> Operand | None:
    part_name = spec.storage_continuation_part
    if part_name is None:
        return None
    if len(register_outputs) != 1:
        raise ValueError(
            f"{spec.form_name}: storage continuation requires exactly one result"
        )
    part = descriptor_specs._REGISTER_PARTS_BY_NAME.get(part_name)
    if part is None:
        raise ValueError(
            f"{spec.form_name}: unknown storage-continuation part {part_name}"
        )
    result_part_name = _operand_override_map(
        spec, spec.operand_register_parts, "register-part overrides"
    ).get(register_outputs[0].name)
    if result_part_name is None:
        raise ValueError(
            f"{spec.form_name}: storage continuation requires a partial result"
        )
    result_part = descriptor_specs._REGISTER_PARTS_BY_NAME[result_part_name]
    if result_part.reg_class != part.reg_class or result_part.mask & part.mask:
        raise ValueError(
            f"{spec.form_name}: storage continuation must preserve a disjoint "
            "part of the result register"
        )
    return Operand(
        field_name="storage",
        role=OperandRole.OPERAND,
        reg_alts=(RegClassAlt(part.reg_class),),
        flags=(OperandFlag.IMPLICIT, OperandFlag.STORAGE_CONTINUATION),
        register_part=part.name,
    )


def _implicit_output_storage(
    spec: descriptor_specs._DescriptorSpec,
    operand: MachineOperand,
) -> tuple[str, str]:
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
    if operand not in form.outputs:
        raise ValueError(
            f"{spec.form_name}.{operand.name}: implicit output is not a machine output"
        )
    if operand.kind is MachineOperandKind.IMMEDIATE:
        raise ValueError(
            f"{spec.form_name}.{operand.name}: implicit output must be a register"
        )
    machine_class_name = _operand_storage_machine_class(spec, operand)
    machine_class = _MACHINE_CLASSES[machine_class_name]
    if len(machine_class.candidates) != 1:
        raise ValueError(
            f"{spec.form_name}.{operand.name}: implicit output class "
            f"{machine_class_name} must name exactly one physical register"
        )
    return machine_class_name, machine_class.candidates[0]


def _implicit_output_operand(
    spec: descriptor_specs._DescriptorSpec,
    operand: MachineOperand,
) -> Operand:
    """Models one fixed architectural output without producing Low SSA."""

    machine_class_name, _ = _implicit_output_storage(spec, operand)
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
    operand_ordinal = _operand_ordinal(form, operand)
    read_stage, ready_stage = _operand_stages(spec, operand)
    return Operand(
        field_name=f"implicit_output_{operand.name}",
        role=OperandRole.IMPLICIT,
        reg_alts=(
            RegClassAlt(
                _low_register_class_name(machine_class_name),
                flags=(RegClassAltFlag.PHYSICAL_ONLY,),
            ),
        ),
        flags=(OperandFlag.IMPLICIT, OperandFlag.STATE_WRITE),
        read_stage=read_stage,
        ready_stage=ready_stage,
        write_event=_register_timing_event(spec, operand_ordinal, "write"),
    )


def _implicit_output_encoding_field_values(
    spec: descriptor_specs._DescriptorSpec,
    operands: tuple[MachineOperand, ...],
) -> tuple[EncodingFieldValue, ...]:
    instruction = _INSTRUCTION_ENCODINGS[spec.form_name]
    fields = {field.name: field for field in instruction.fields}
    result = []
    for operand in operands:
        field = fields.get(operand.name)
        if field is None:
            continue
        _, register_name = _implicit_output_storage(spec, operand)
        if operand.kind is not MachineOperandKind.REGISTER_ADAPTER:
            raise ValueError(
                f"{spec.form_name}.{operand.name}: encoded implicit output must "
                "use a register adapter"
            )
        adapter = _MACHINE_ADAPTERS[operand.type_name]
        register_encodings = dict(adapter.effective_register_encodings)
        encoded_register = register_encodings[register_name]
        if encoded_register & ~field.value_mask:
            raise ValueError(
                f"{spec.form_name}.{operand.name}: adapted register value "
                f"{encoded_register} exceeds encoding field mask "
                f"{field.value_mask:#x}"
            )
        result.append(
            EncodingFieldValue(
                _ENCODING_FIELD_IDS[operand.name],
                encoded_register,
            )
        )
    return tuple(result)


def _implicit_operands(spec: descriptor_specs._DescriptorSpec) -> tuple[Operand, ...]:
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
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
    return f"{descriptor_specs._TARGET_KEY}.memory.{access}.c{cycle_segment}"


def _memory_timing_event(spec: descriptor_specs._DescriptorSpec, access: str) -> str:
    memory = _itinerary(spec).memory
    if memory is None:
        raise ValueError(
            f"{spec.form_name}: memory operation uses non-memory itinerary "
            f"{spec.itinerary}"
        )
    return _memory_event_name(access, memory)


def _effects(
    spec: descriptor_specs._DescriptorSpec, form: MachineForm
) -> tuple[Effect, ...]:
    has_memory_effect = has_property(form, "mayLoad") or has_property(form, "mayStore")
    if spec.ordered_memory and not has_memory_effect:
        raise ValueError(f"{form.name}: ordered memory alias is not a memory form")
    register_width_bits = spec.memory_width_bits
    if register_width_bits is None:
        register_width_bits = max(
            (
                _MACHINE_CLASSES[
                    _operand_storage_machine_class(spec, operand)
                ].layout.register_size_bits
                * _operand_unit_count(spec, operand)
                for operand in (*form.outputs, *form.inputs)
                if operand.kind is not MachineOperandKind.IMMEDIATE
            ),
            default=0,
        )
    elif not has_memory_effect:
        raise ValueError(
            f"{form.name}: explicit memory width requires a memory instruction"
        )
    result = []
    memory_flags = (
        (EffectFlag.ORDERED, EffectFlag.DEPENDENCY)
        if spec.ordered_memory
        else (EffectFlag.DEPENDENCY,)
    )
    if has_property(form, "mayLoad"):
        if register_width_bits == 0:
            raise ValueError(f"{form.name}: load has no register payload width")
        timing_event = _memory_timing_event(spec, "read")
        result.append(
            Effect(
                EffectKind.READ,
                MemorySpace.WORKGROUP,
                flags=memory_flags,
                width_bits=register_width_bits,
                producer_event=timing_event,
                consumer_event=timing_event,
            )
        )
    if has_property(form, "mayStore"):
        if register_width_bits == 0:
            raise ValueError(f"{form.name}: store has no register payload width")
        timing_event = _memory_timing_event(spec, "write")
        result.append(
            Effect(
                EffectKind.WRITE,
                MemorySpace.WORKGROUP,
                flags=memory_flags,
                width_bits=register_width_bits,
                producer_event=timing_event,
                consumer_event=timing_event,
            )
        )
    if form.control_flow_kind is not None:
        result.append(Effect(EffectKind.CONTROL))
    result.extend(spec.effects)
    return tuple(result)


def _descriptor_flags(
    spec: descriptor_specs._DescriptorSpec,
    form: MachineForm,
    *,
    owns_implicit_state: bool,
) -> tuple[DescriptorFlag, ...]:
    result = []
    side_effecting = (
        has_property(form, "hasSideEffects")
        or has_property(form, "mayStore")
        or form.control_flow_kind is not None
        or owns_implicit_state
    )
    if side_effecting:
        result.append(DescriptorFlag.SIDE_EFFECTING)
    if (
        form.control_flow_kind == "return"
        or (form.control_flow_kind or "").startswith("branch_")
        or has_property(form, "isTerminator")
    ):
        result.append(DescriptorFlag.TERMINATOR)
    if (
        not side_effecting
        and not has_property(form, "mayLoad")
        and not has_property(form, "mayStore")
        and form.control_flow_kind is None
        and form.name != "RET"
        and not owns_implicit_state
    ):
        result.append(DescriptorFlag.DEAD_REMOVABLE)
    if spec.allocation_move:
        result.append(DescriptorFlag.ALLOCATION_MOVE)
    return tuple(result)


def _instruction_classes(
    spec: descriptor_specs._DescriptorSpec,
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
    if has_property(form, "isMoveImm") or spec.semantic_tag.startswith(
        "register.move."
    ):
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


def _descriptor(spec: descriptor_specs._DescriptorSpec) -> Descriptor:
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
    if spec.form_name != form.name:
        raise ValueError(f"descriptor spec selected the wrong machine form {form.name}")
    if spec.itinerary != form.itinerary and not spec.storage_overrides:
        raise ValueError(
            f"{form.name}: itinerary override {spec.itinerary} requires a storage "
            "specialization"
        )
    if len(set(spec.implicit_outputs)) != len(spec.implicit_outputs):
        raise ValueError(f"{form.name}: implicit output names must be unique")
    machine_output_names = {operand.name for operand in form.outputs}
    unknown_implicit_outputs = set(spec.implicit_outputs) - machine_output_names
    if unknown_implicit_outputs:
        raise ValueError(
            f"{form.name}: implicit outputs name unknown machine outputs "
            f"{sorted(unknown_implicit_outputs)}"
        )
    implicit_outputs = tuple(
        operand for operand in form.outputs if operand.name in spec.implicit_outputs
    )
    register_outputs = tuple(
        operand
        for operand in form.outputs
        if operand.kind is not MachineOperandKind.IMMEDIATE
        and operand.name not in spec.implicit_outputs
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
    storage_continuation = _storage_continuation_operand(spec, register_outputs)
    tied_implicit_outputs = {
        name
        for tie in form.ties
        for name in (tie.definition, tie.use)
        if name in spec.implicit_outputs
    }
    if tied_implicit_outputs:
        raise ValueError(
            f"{form.name}: tied outputs cannot be implicit architectural outputs "
            f"{sorted(tied_implicit_outputs)}"
        )
    physical_mnemonic = spec.asm_mnemonic or descriptor_specs._ASM_MNEMONIC_BY_FORM.get(
        spec.form_name, form.assembly.split("\t", 1)[0].strip()
    )
    mnemonic = (
        f"{physical_mnemonic}.volatile" if spec.ordered_memory else physical_mnemonic
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
            *((storage_continuation,) if storage_continuation is not None else ()),
            *(_implicit_output_operand(spec, operand) for operand in implicit_outputs),
            *_implicit_operands(spec),
        ),
        immediates=tuple(
            _immediate(spec.form_name, operand) for operand in immediate_inputs
        ),
        encoding_field_values=_implicit_output_encoding_field_values(
            spec, implicit_outputs
        ),
        schedule_class=_SCHEDULE_CLASS_NAMES[(spec.form_name, spec.itinerary)],
        schedule_alternatives=spec.schedule_alternatives,
        op_kind=spec.op_kind,
        asm_forms=(
            AsmForm(
                mnemonic=mnemonic,
                native_assembly_mnemonic=(
                    physical_mnemonic if spec.ordered_memory else None
                ),
                results=tuple(operand.name for operand in register_outputs),
                operands=(
                    *(operand.name for operand in register_inputs),
                    *(
                        (storage_continuation.field_name,)
                        if storage_continuation
                        else ()
                    ),
                ),
                immediates=tuple(
                    AsmImmediate(operand.name) for operand in immediate_inputs
                ),
            ),
        ),
        effects=_effects(spec, form),
        constraints=(
            *_constraints(form, explicit_register_operands),
            *(
                (Constraint(ConstraintKind.REMATERIALIZABLE, 0),)
                if spec.op_kind is DescriptorOpKind.CONST
                else ()
            ),
            *(
                (
                    Constraint(
                        ConstraintKind.TIED,
                        0,
                        len(explicit_register_operands),
                    ),
                )
                if storage_continuation is not None
                else ()
            ),
        ),
        encoding_id=_INSTRUCTION_IDS[spec.form_name],
        flags=_descriptor_flags(
            spec,
            form,
            owns_implicit_state=bool(implicit_outputs) and not register_outputs,
        ),
        instruction_classes=_instruction_classes(spec, form),
    )
    expected_fields = {
        field.name for field in _INSTRUCTION_ENCODINGS[spec.form_name].fields
    }
    encoded_field_ids = {
        row.encoding_field_id
        for row in (*descriptor.operands, *descriptor.immediates)
        if row.encoding_field_id
    }
    fixed_field_ids = {
        row.encoding_field_id for row in descriptor.encoding_field_values
    }
    if encoded_field_ids & fixed_field_ids:
        raise ValueError(f"{spec.form_name}: dynamic and fixed encoding fields overlap")
    field_names_by_id = {
        field_id: field_name for field_name, field_id in _ENCODING_FIELD_IDS.items()
    }
    encoded_fields = {
        field_names_by_id[field_id] for field_id in encoded_field_ids | fixed_field_ids
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


# Physical bundle slots constrain issue but do not classify instructions. For
# example, the lda slot also carries MOVA register-immediate instructions.
_SLOT_RESOURCE_KINDS = {
    slot: ResourceKind.PIPELINE
    for slot in sorted(
        {instruction.slot for instruction in CORE_ENCODING_TABLE.instructions}
    )
}


def _slot_resource_name(slot: str) -> str:
    return f"{descriptor_specs._TARGET_KEY}.slot.{slot}"


def _pipeline_resource_name(resource: str) -> str:
    return f"{descriptor_specs._TARGET_KEY}.pipeline.{resource.lower()}"


def _bundle_exclusion_resource_name(slots: tuple[str, ...]) -> str:
    return f"{descriptor_specs._TARGET_KEY}.bundle.exclusion.{'.'.join(slots)}"


def _bundle_slot_exclusions() -> tuple[tuple[str, ...], ...]:
    """Derives minimal slot sets no physical bundle can contain."""

    legal_signatures = {
        frozenset(field.slot for field in bundle_format.fields)
        for bundle_format in CORE_ENCODING_TABLE.bundle_formats
    }
    extendable_signatures: set[frozenset[str]] = set()
    for signature in legal_signatures:
        ordered_signature = tuple(sorted(signature))
        for subset_count in range(1, len(ordered_signature) + 1):
            for subset in combinations(ordered_signature, subset_count):
                extendable_signatures.add(frozenset(subset))

    slots = tuple(_SLOT_RESOURCE_KINDS)
    result = []
    for slot_count in range(2, len(slots) + 1):
        for candidate in combinations(slots, slot_count):
            signature = frozenset(candidate)
            if signature in extendable_signatures:
                continue
            if all(
                frozenset(subset) in extendable_signatures
                for subset in combinations(candidate, slot_count - 1)
            ):
                result.append(candidate)
    return tuple(result)


_BUNDLE_SLOT_EXCLUSIONS = _bundle_slot_exclusions()


_RESOURCES = (
    *(
        Resource(_slot_resource_name(slot), 1, kind)
        for slot, kind in sorted(_SLOT_RESOURCE_KINDS.items())
    ),
    *(
        Resource(
            _bundle_exclusion_resource_name(slots),
            len(slots) - 1,
            ResourceKind.PIPELINE,
        )
        for slots in _BUNDLE_SLOT_EXCLUSIONS
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
    spec: descriptor_specs._DescriptorSpec,
    itinerary: Itinerary,
) -> tuple[IssueUse, ...]:
    slot = _INSTRUCTION_ENCODINGS[spec.form_name].slot
    result = [
        IssueUse(_slot_resource_name(slot), 1, 1),
        *(
            IssueUse(_bundle_exclusion_resource_name(exclusion), 1, 1)
            for exclusion in _BUNDLE_SLOT_EXCLUSIONS
            if slot in exclusion
        ),
    ]
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


def _schedule_class(spec: descriptor_specs._DescriptorSpec) -> ScheduleClass:
    form = descriptor_specs._MACHINE_FORMS[spec.form_name]
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
            descriptor_specs._TARGET_KEY,
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
    for spec in descriptor_specs._DESCRIPTOR_SPECS:
        schedule_class = _schedule_class(spec)
        key = (spec.form_name, spec.itinerary)
        names[key] = schedule_class.name
        previous = classes.setdefault(schedule_class.name, schedule_class)
        if previous != schedule_class:
            raise ValueError(f"{schedule_class.name}: schedule-class name collision")
    return names, tuple(classes[name] for name in sorted(classes))


_SCHEDULE_CLASS_NAMES, _SCHEDULE_CLASSES = _schedule_classes()


AIE2P_CORE_DESCRIPTOR_SET = DescriptorSet(
    key=f"{descriptor_specs._TARGET_KEY}.core",
    target_key=descriptor_specs._TARGET_KEY,
    feature_key=f"{descriptor_specs._TARGET_KEY}.core.v1",
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
    register_parts=descriptor_specs._REGISTER_PARTS,
    physical_registers=_physical_registers(),
    physical_register_views=_physical_register_views(),
    register_packing_resources=_register_packing_resources(),
    timing_events=_TIMING_EVENTS,
    event_separations=_EVENT_SEPARATIONS,
    resources=_RESOURCES,
    schedule_classes=_SCHEDULE_CLASSES,
    descriptors=tuple(_descriptor(spec) for spec in descriptor_specs._DESCRIPTOR_SPECS),
    requires_explicit_asm_surface=True,
)
