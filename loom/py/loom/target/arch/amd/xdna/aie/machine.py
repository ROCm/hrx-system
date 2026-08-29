# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Physical-register and machine-form schema shared by AMD AIE targets.

The schema deliberately models arbitrary physical-register unit sets. A
register class is an ordered color domain whose candidates are pairwise
disjoint, while registers in different classes may alias through their atomic
units. Register adapters remain distinct from register hardware encodings:
each instruction operand can map the same physical register differently before
the instruction field consumes its declared low bits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from loom.target.arch.amd.xdna.aie.encoding import EncodingTable


class MachineOperandKind(Enum):
    """Source kind of one explicit machine-form operand."""

    REGISTER_CLASS = "register_class"
    REGISTER_ADAPTER = "register_adapter"
    IMMEDIATE = "immediate"


@dataclass(frozen=True, slots=True)
class RegisterLayout:
    """Register and spill layout shared by register classes."""

    register_size_bits: int
    alignment_bits: int
    spill_size_bits: int
    spill_alignment_bits: int


@dataclass(frozen=True, slots=True)
class PhysicalRegister:
    """One named physical register in a target-global atomic-unit namespace."""

    name: str
    assembly_name: str
    hardware_encoding: int
    atomic_units: tuple[int, ...]
    subregisters: tuple[str, ...] = ()
    subregister_indices: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RegisterClass:
    """One ordered, pairwise-disjoint physical-register color domain."""

    name: str
    layout: RegisterLayout
    value_types: tuple[str, ...]
    candidates: tuple[str, ...]
    is_allocatable: bool = True
    consider_in_pre_ra_scheduling: bool = True
    generate_pressure_set: bool = True


@dataclass(frozen=True, slots=True)
class RegisterAdapter:
    """Operand-local physical-register encoding and any architectural repair.

    ``register_encodings`` preserves the exact value produced by the pinned
    source compiler's custom operand encoder. ``architectural_encodings`` is
    normally empty; when populated, it replaces a source-compiler mapping that
    conflicts with the instruction field and decoder. The final instruction
    field still truncates either mapping to its declared encoded width.
    """

    name: str
    register_class: str
    register_encodings: tuple[tuple[str, int], ...]
    architectural_encodings: tuple[tuple[str, int], ...] = ()

    @property
    def effective_register_encodings(self) -> tuple[tuple[str, int], ...]:
        """Returns the register mapping used by the architectural encoder."""
        return self.architectural_encodings or self.register_encodings


@dataclass(frozen=True, slots=True)
class ImmediateEncoding:
    """One scaled immediate domain and its native encoded width."""

    name: str
    semantic_width_bits: int
    encoded_width_bits: int
    step: int
    is_signed: bool
    is_negative: bool = False
    allows_symbol_reference: bool = False


@dataclass(frozen=True, slots=True)
class MachineOperand:
    """One explicit definition or use in a machine form."""

    name: str
    type_name: str
    kind: MachineOperandKind


@dataclass(frozen=True, slots=True)
class MachineTie:
    """One definition/use pair constrained to the same physical register."""

    definition: str
    use: str


@dataclass(frozen=True, slots=True)
class MachineForm:
    """Selection and allocation facts for one physical instruction form."""

    name: str
    itinerary: str
    property_bits: int
    control_flow_kind: str | None
    outputs: tuple[MachineOperand, ...]
    inputs: tuple[MachineOperand, ...]
    implicit_defs: tuple[str, ...]
    implicit_uses: tuple[str, ...]
    ties: tuple[MachineTie, ...]
    assembly: str


@dataclass(frozen=True, slots=True)
class MachineTable:
    """Complete physical register and instruction-selection domain."""

    atomic_unit_names: tuple[str, ...]
    physical_registers: tuple[PhysicalRegister, ...]
    register_classes: tuple[RegisterClass, ...]
    register_adapters: tuple[RegisterAdapter, ...]
    immediates: tuple[ImmediateEncoding, ...]
    forms: tuple[MachineForm, ...]


MACHINE_PROPERTY_NAMES = (
    "hasDelaySlot",
    "hasExtraDefRegAllocReq",
    "hasExtraSrcRegAllocReq",
    "hasSideEffects",
    "isBarrier",
    "isBranch",
    "isCall",
    "isCodeGenOnly",
    "isCommutable",
    "isMoveImm",
    "isPseudo",
    "isReMaterializable",
    "isReturn",
    "isTerminator",
    "mayLoad",
    "mayStore",
)


def property_bits(properties: Mapping[str, int | bool]) -> int:
    """Packs the canonical machine property domain into stable bits."""
    unknown = set(properties) - set(MACHINE_PROPERTY_NAMES)
    if unknown:
        raise ValueError(f"unknown machine properties: {sorted(unknown)}")
    return sum(
        (1 << index) if properties.get(name, False) else 0
        for index, name in enumerate(MACHINE_PROPERTY_NAMES)
    )


def has_property(form: MachineForm, name: str) -> bool:
    """Returns whether ``form`` has one canonical machine property."""
    try:
        index = MACHINE_PROPERTY_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown machine property {name!r}") from exc
    return bool(form.property_bits & (1 << index))


def encode_immediate(immediate: ImmediateEncoding, value: int) -> int:
    """Validates and encodes one concrete immediate value."""
    if value % immediate.step:
        raise ValueError(
            f"{immediate.name}: value {value} is not a multiple of {immediate.step}"
        )
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
    if value < minimum or value > maximum:
        raise ValueError(
            f"{immediate.name}: value {value} is outside [{minimum}, {maximum}]"
        )
    return (value >> fixed_zero_bits) & ((1 << immediate.encoded_width_bits) - 1)


def decode_immediate(immediate: ImmediateEncoding, encoded_value: int) -> int:
    """Decodes one native immediate field value."""
    if encoded_value < 0 or encoded_value >= 1 << immediate.encoded_width_bits:
        raise ValueError(
            f"{immediate.name}: encoded value {encoded_value} exceeds "
            f"{immediate.encoded_width_bits} bits"
        )
    fixed_zero_bits = immediate.step.bit_length() - 1
    if immediate.is_negative:
        semantic_bits = immediate.encoded_width_bits + fixed_zero_bits + 1
        value = (encoded_value << fixed_zero_bits) | (
            1 << (immediate.encoded_width_bits + fixed_zero_bits)
        )
    else:
        semantic_bits = immediate.encoded_width_bits + fixed_zero_bits
        value = encoded_value << fixed_zero_bits
        if not immediate.is_signed:
            return value
    sign_bit = 1 << (semantic_bits - 1)
    return value - (1 << semantic_bits) if value & sign_bit else value


def _validate_unique(values: Sequence[str], description: str) -> None:
    if len(values) == len(set(values)):
        return
    duplicates = sorted(value for value in set(values) if values.count(value) > 1)
    raise ValueError(f"{description} contains duplicates: {duplicates}")


def _validate_registers(table: MachineTable) -> dict[str, PhysicalRegister]:
    _validate_unique(table.atomic_unit_names, "atomic unit names")
    _validate_unique(
        [register.name for register in table.physical_registers],
        "physical register names",
    )
    atomic_unit_count = len(table.atomic_unit_names)
    registers = {register.name: register for register in table.physical_registers}
    for register in table.physical_registers:
        if not register.name or not register.assembly_name:
            raise ValueError("physical register names must not be empty")
        if register.hardware_encoding < 0 or register.hardware_encoding > 0xFFFF:
            raise ValueError(f"{register.name}: hardware encoding does not fit uint16")
        if not register.atomic_units:
            raise ValueError(f"{register.name}: atomic unit set is empty")
        if register.atomic_units != tuple(sorted(set(register.atomic_units))):
            raise ValueError(f"{register.name}: atomic units must be sorted and unique")
        if (
            register.atomic_units[0] < 0
            or register.atomic_units[-1] >= atomic_unit_count
        ):
            raise ValueError(f"{register.name}: atomic unit is out of range")
        if len(register.subregisters) != len(register.subregister_indices):
            raise ValueError(f"{register.name}: subregister and index counts differ")
        unknown_subregisters = set(register.subregisters) - registers.keys()
        if unknown_subregisters:
            raise ValueError(
                f"{register.name}: unknown subregisters {sorted(unknown_subregisters)}"
            )
        if register.subregisters:
            covered_units = {
                atomic_unit
                for subregister in register.subregisters
                for atomic_unit in registers[subregister].atomic_units
            }
            if covered_units != set(register.atomic_units):
                raise ValueError(
                    f"{register.name}: subregisters do not cover its atomic units"
                )
    return registers


def _validate_register_classes(
    table: MachineTable,
    registers: Mapping[str, PhysicalRegister],
) -> dict[str, RegisterClass]:
    _validate_unique(
        [register_class.name for register_class in table.register_classes],
        "register class names",
    )
    register_classes = {
        register_class.name: register_class for register_class in table.register_classes
    }
    for register_class in table.register_classes:
        layout = register_class.layout
        for value, description in (
            (layout.register_size_bits, "register size"),
            (layout.alignment_bits, "alignment"),
            (layout.spill_size_bits, "spill size"),
            (layout.spill_alignment_bits, "spill alignment"),
        ):
            if value <= 0 or value > 0xFFFF:
                raise ValueError(
                    f"{register_class.name}: {description} does not fit uint16"
                )
        if not register_class.candidates:
            raise ValueError(f"{register_class.name}: candidate set is empty")
        if not register_class.value_types:
            raise ValueError(f"{register_class.name}: value type set is empty")
        _validate_unique(
            register_class.value_types,
            f"{register_class.name} value types",
        )
        _validate_unique(
            register_class.candidates,
            f"{register_class.name} candidates",
        )
        unknown_candidates = set(register_class.candidates) - registers.keys()
        if unknown_candidates:
            raise ValueError(
                f"{register_class.name}: unknown candidates "
                f"{sorted(unknown_candidates)}"
            )
        occupied_units: dict[int, str] = {}
        for candidate in register_class.candidates:
            for atomic_unit in registers[candidate].atomic_units:
                previous = occupied_units.get(atomic_unit)
                if previous is not None:
                    raise ValueError(
                        f"{register_class.name}: candidates {previous} and "
                        f"{candidate} overlap atomic unit {atomic_unit}"
                    )
                occupied_units[atomic_unit] = candidate
    return register_classes


def _validate_adapters(
    table: MachineTable,
    register_classes: Mapping[str, RegisterClass],
) -> dict[str, RegisterAdapter]:
    _validate_unique(
        [adapter.name for adapter in table.register_adapters],
        "register adapter names",
    )
    adapters = {adapter.name: adapter for adapter in table.register_adapters}
    for adapter in table.register_adapters:
        register_class = register_classes.get(adapter.register_class)
        if register_class is None:
            raise ValueError(
                f"{adapter.name}: unknown register class {adapter.register_class}"
            )
        for map_name, register_encodings in (
            ("source", adapter.register_encodings),
            ("architectural", adapter.architectural_encodings),
        ):
            if not register_encodings:
                continue
            encoded_registers = tuple(
                register_name for register_name, _ in register_encodings
            )
            _validate_unique(
                encoded_registers, f"{adapter.name} {map_name} register map"
            )
            if set(encoded_registers) != set(register_class.candidates):
                raise ValueError(
                    f"{adapter.name}: {map_name} register map does not match "
                    f"class {register_class.name}"
                )
            encoded_values = [value for _, value in register_encodings]
            if len(encoded_values) != len(set(encoded_values)):
                raise ValueError(f"{adapter.name}: {map_name} encoded values collide")
            if encoded_values and (
                min(encoded_values) < 0 or max(encoded_values) > 127
            ):
                raise ValueError(
                    f"{adapter.name}: {map_name} encoded value exceeds seven bits"
                )
    return adapters


def _validate_immediates(table: MachineTable) -> dict[str, ImmediateEncoding]:
    _validate_unique(
        [immediate.name for immediate in table.immediates],
        "immediate names",
    )
    immediates = {immediate.name: immediate for immediate in table.immediates}
    for immediate in table.immediates:
        if (
            immediate.step <= 0
            or immediate.step & (immediate.step - 1)
            or immediate.encoded_width_bits <= 0
            or immediate.encoded_width_bits > 64
        ):
            raise ValueError(f"{immediate.name}: invalid encoded immediate layout")
        fixed_zero_bits = immediate.step.bit_length() - 1
        expected_semantic_width = (
            immediate.encoded_width_bits
            + fixed_zero_bits
            + (1 if immediate.is_negative else 0)
        )
        if immediate.semantic_width_bits != expected_semantic_width:
            raise ValueError(
                f"{immediate.name}: semantic width "
                f"{immediate.semantic_width_bits} does not match "
                f"{expected_semantic_width}"
            )
        if immediate.is_negative and not immediate.is_signed:
            raise ValueError(f"{immediate.name}: negative domain is not signed")
    return immediates


def _validate_forms(
    table: MachineTable,
    encoding_table: EncodingTable,
    registers: Mapping[str, PhysicalRegister],
    register_classes: Mapping[str, RegisterClass],
    adapters: Mapping[str, RegisterAdapter],
    immediates: Mapping[str, ImmediateEncoding],
) -> None:
    _validate_unique([form.name for form in table.forms], "machine form names")
    encoding_names = tuple(row.name for row in encoding_table.instructions)
    form_names = tuple(form.name for form in table.forms)
    if form_names != encoding_names:
        raise ValueError("machine forms do not align with instruction encodings")
    known_control_flow_kinds = {
        None,
        "branch_conditional_decrement",
        "branch_conditional_nonzero",
        "branch_conditional_zero",
        "branch_direct",
        "branch_indirect",
        "call_direct",
        "call_indirect",
        "return",
    }
    invalid_encoding_domains: list[str] = []
    for form, encoding in zip(table.forms, encoding_table.instructions, strict=True):
        if not form.itinerary or not form.assembly:
            raise ValueError(f"{form.name}: itinerary and assembly must not be empty")
        if form.property_bits & ~((1 << len(MACHINE_PROPERTY_NAMES)) - 1):
            raise ValueError(f"{form.name}: unknown machine property bits")
        if form.control_flow_kind not in known_control_flow_kinds:
            raise ValueError(
                f"{form.name}: unknown control-flow kind {form.control_flow_kind}"
            )
        operands = (*form.outputs, *form.inputs)
        _validate_unique(
            [operand.name for operand in operands],
            f"{form.name} operand names",
        )
        for operand in operands:
            if operand.kind is MachineOperandKind.REGISTER_CLASS:
                known = operand.type_name in register_classes
            elif operand.kind is MachineOperandKind.REGISTER_ADAPTER:
                known = operand.type_name in adapters
            else:
                known = operand.type_name in immediates
            if not known:
                raise ValueError(
                    f"{form.name}.{operand.name}: unknown "
                    f"{operand.kind.value} {operand.type_name}"
                )
        unknown_implicit_registers = (
            set(form.implicit_defs) | set(form.implicit_uses)
        ) - registers.keys()
        if unknown_implicit_registers:
            raise ValueError(
                f"{form.name}: unknown implicit registers "
                f"{sorted(unknown_implicit_registers)}"
            )
        _validate_unique(form.implicit_defs, f"{form.name} implicit definitions")
        _validate_unique(form.implicit_uses, f"{form.name} implicit uses")
        output_names = {operand.name for operand in form.outputs}
        input_names = {operand.name for operand in form.inputs}
        _validate_unique(
            [tie.definition for tie in form.ties],
            f"{form.name} tied definitions",
        )
        _validate_unique([tie.use for tie in form.ties], f"{form.name} tied uses")
        for tie in form.ties:
            if tie.definition not in output_names or tie.use not in input_names:
                raise ValueError(f"{form.name}: invalid tie {tie.definition}={tie.use}")
        explicit_operand_names = {operand.name for operand in operands}
        encoding_field_names = {field.name for field in encoding.fields}
        if not encoding_field_names.issubset(explicit_operand_names):
            raise ValueError(f"{form.name}: encoding fields are not explicit operands")
        operands_by_name = {operand.name: operand for operand in operands}
        for field in encoding.fields:
            operand = operands_by_name[field.name]
            if operand.kind is MachineOperandKind.REGISTER_CLASS:
                register_class = register_classes[operand.type_name]
                register_values = tuple(
                    (
                        candidate,
                        registers[candidate].hardware_encoding & field.value_mask,
                    )
                    for candidate in register_class.candidates
                )
            elif operand.kind is MachineOperandKind.REGISTER_ADAPTER:
                adapter = adapters[operand.type_name]
                register_values = tuple(
                    (register_name, encoded_value & field.value_mask)
                    for register_name, encoded_value in (
                        adapter.effective_register_encodings
                    )
                )
            else:
                immediate = immediates[operand.type_name]
                maximum_encoded_value = (1 << immediate.encoded_width_bits) - 1
                if maximum_encoded_value > field.value_mask:
                    invalid_encoding_domains.append(
                        f"{form.name}.{field.name} {operand.type_name} "
                        f"maximum 0x{maximum_encoded_value:x} exceeds "
                        f"field mask 0x{field.value_mask:x}"
                    )
                continue
            encoded_registers_by_value: dict[int, list[str]] = {}
            for register_name, encoded_value in register_values:
                encoded_registers_by_value.setdefault(encoded_value, []).append(
                    register_name
                )
            collisions = {
                encoded_value: register_names
                for encoded_value, register_names in encoded_registers_by_value.items()
                if len(register_names) > 1
            }
            if collisions:
                invalid_encoding_domains.append(
                    f"{form.name}.{field.name} {operand.type_name} "
                    f"collides after field mask 0x{field.value_mask:x}: "
                    f"{collisions}"
                )
    if invalid_encoding_domains:
        raise ValueError(
            "operand domains exceed encoding fields:\n"
            + "\n".join(invalid_encoding_domains)
        )


def validate_machine_table(
    table: MachineTable,
    encoding_table: EncodingTable,
) -> None:
    """Validates every invariant consumed unchecked by generated native tables."""
    registers = _validate_registers(table)
    register_classes = _validate_register_classes(table, registers)
    adapters = _validate_adapters(table, register_classes)
    immediates = _validate_immediates(table)
    _validate_forms(
        table,
        encoding_table,
        registers,
        register_classes,
        adapters,
        immediates,
    )
