# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Instruction and bundle encoding schema for AMD XDNA AIE targets.

The native encoder consumes flattened versions of these records. Keeping the
authoritative representation here lets generation prove bit coverage,
fixed-pattern validity, and bundle decode uniqueness before those facts reach
the compiler hot path. Sharing this representation does not imply physical ISA
compatibility or couple the independently enabled targets that use it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BitMapping:
    """Maps one source value bit into one encoded target bit."""

    target_bit: int
    value_bit: int


def bit_range(target_bit: int, value_bit: int, count: int) -> tuple[BitMapping, ...]:
    """Returns a contiguous run of bit mappings."""
    if count <= 0:
        raise ValueError("bit mapping range must not be empty")
    return tuple(
        BitMapping(target_bit + index, value_bit + index) for index in range(count)
    )


@dataclass(frozen=True, slots=True)
class InstructionFieldEncoding:
    """One target-owned low operand field within an instruction encoding."""

    name: str
    mappings: tuple[BitMapping, ...]

    @property
    def value_mask(self) -> int:
        mask = 0
        for mapping in self.mappings:
            mask |= 1 << mapping.value_bit
        return mask


@dataclass(frozen=True, slots=True)
class InstructionEncoding:
    """One named instruction encoding in a physical AIE slot."""

    name: str
    slot: str
    bit_count: int
    fixed_mask: int
    fixed_value: int
    fields: tuple[InstructionFieldEncoding, ...] = ()
    delay_slot_count: int = 0


@dataclass(frozen=True, slots=True)
class BundleFieldEncoding:
    """Places one physical slot instruction into a bundle format."""

    slot: str
    mappings: tuple[BitMapping, ...]

    @property
    def value_mask(self) -> int:
        mask = 0
        for mapping in self.mappings:
            mask |= 1 << mapping.value_bit
        return mask


@dataclass(frozen=True, slots=True)
class BundleFormatEncoding:
    """One variable-width AIE VLIW bundle format."""

    name: str
    bit_count: int
    fixed_mask: int
    fixed_value: int
    fields: tuple[BundleFieldEncoding, ...]


@dataclass(frozen=True, slots=True)
class EncodingTable:
    """Instruction and bundle encoding domain for one physical target."""

    instructions: tuple[InstructionEncoding, ...]
    bundle_formats: tuple[BundleFormatEncoding, ...]


@dataclass(frozen=True, slots=True)
class InstructionInstance:
    """One instruction and its already-adapted target field values."""

    instruction_name: str
    field_values: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class BundleInstance:
    """One bundle format populated by physical-slot instructions."""

    format_name: str
    instructions: tuple[InstructionInstance, ...]


@dataclass(frozen=True, slots=True)
class EncodingWitness:
    """An independently retained byte sequence and its packet plan."""

    symbol: str
    expected_bytes: bytes
    bundles: tuple[BundleInstance, ...]


def _validate_unique(values: Sequence[str], context: str) -> None:
    if len(values) != len(set(values)):
        duplicates = sorted(value for value in set(values) if values.count(value) > 1)
        raise ValueError(f"{context}: duplicate values: {', '.join(duplicates)}")


def _validate_encoding(
    context: str,
    bit_count: int,
    fixed_mask: int,
    fixed_value: int,
    fields: Sequence[InstructionFieldEncoding | BundleFieldEncoding],
) -> None:
    if not 1 <= bit_count <= 128:
        raise ValueError(f"{context}: bit count {bit_count} is outside [1, 128]")
    encoding_mask = (1 << bit_count) - 1
    if fixed_mask & ~encoding_mask:
        raise ValueError(f"{context}: fixed mask exceeds the encoding width")
    if fixed_value & ~fixed_mask:
        raise ValueError(f"{context}: fixed value sets non-fixed bits")

    covered_bits = {bit for bit in range(bit_count) if fixed_mask & (1 << bit)}
    field_ids: list[str] = []
    for field in fields:
        field_id = (
            field.name if isinstance(field, InstructionFieldEncoding) else field.slot
        )
        field_ids.append(field_id)
        if not field.mappings:
            raise ValueError(f"{context}.{field_id}: bit mapping is empty")
        value_bits = [mapping.value_bit for mapping in field.mappings]
        if len(value_bits) != len(set(value_bits)):
            duplicates = sorted(
                value_bit
                for value_bit in set(value_bits)
                if value_bits.count(value_bit) > 1
            )
            raise ValueError(f"{context}.{field_id}: repeated value bits: {duplicates}")
        value_bits.sort()
        if value_bits != list(range(value_bits[-1] + 1)):
            raise ValueError(
                f"{context}.{field_id}: mapped value bits are not contiguous from zero"
            )
        for mapping in field.mappings:
            if not 0 <= mapping.target_bit < bit_count:
                raise ValueError(
                    f"{context}.{field_id}: target bit {mapping.target_bit} "
                    f"is outside [0, {bit_count})"
                )
            if not 0 <= mapping.value_bit < 64:
                raise ValueError(
                    f"{context}.{field_id}: value bit {mapping.value_bit} "
                    "is outside [0, 64)"
                )
            if mapping.target_bit in covered_bits:
                raise ValueError(
                    f"{context}.{field_id}: target bit {mapping.target_bit} "
                    "is covered twice"
                )
            covered_bits.add(mapping.target_bit)
    _validate_unique(field_ids, f"{context} fields")

    expected_bits = set(range(bit_count))
    if covered_bits != expected_bits:
        missing_bits = sorted(expected_bits - covered_bits)
        raise ValueError(f"{context}: uncovered target bits: {missing_bits}")


def fixed_patterns_overlap(
    left_mask: int,
    left_value: int,
    right_mask: int,
    right_value: int,
) -> bool:
    """Returns whether some bit pattern satisfies both fixed-bit predicates."""
    return ((left_value ^ right_value) & left_mask & right_mask) == 0


def instruction_fixed_pattern_overlaps(
    table: EncodingTable,
) -> tuple[tuple[str, str], ...]:
    """Returns instruction pairs requiring operand-domain disambiguation."""
    overlaps: list[tuple[str, str]] = []
    for left_index, left in enumerate(table.instructions):
        for right in table.instructions[left_index + 1 :]:
            if left.slot != right.slot or left.bit_count != right.bit_count:
                continue
            if fixed_patterns_overlap(
                left.fixed_mask,
                left.fixed_value,
                right.fixed_mask,
                right.fixed_value,
            ):
                overlaps.append((left.name, right.name))
    return tuple(overlaps)


def validate_encoding_table(
    table: EncodingTable,
    slot_bit_counts: Mapping[str, int],
) -> None:
    """Validates every invariant consumed without checks by generated tables."""
    for slot, bit_count in slot_bit_counts.items():
        if not 1 <= bit_count <= 64:
            raise ValueError(f"{slot}: slot width {bit_count} is outside [1, 64]")

    _validate_unique([row.name for row in table.instructions], "instruction names")
    _validate_unique([row.name for row in table.bundle_formats], "bundle names")

    for instruction in table.instructions:
        expected_bit_count = slot_bit_counts.get(instruction.slot)
        if expected_bit_count is None:
            raise ValueError(
                f"{instruction.name}: unknown physical slot {instruction.slot}"
            )
        if instruction.bit_count != expected_bit_count:
            raise ValueError(
                f"{instruction.name}: {instruction.bit_count} bits do not match "
                f"{instruction.slot} width {expected_bit_count}"
            )
        if not 0 <= instruction.delay_slot_count <= 255:
            raise ValueError(f"{instruction.name}: delay slot count does not fit uint8")
        _validate_encoding(
            instruction.name,
            instruction.bit_count,
            instruction.fixed_mask,
            instruction.fixed_value,
            instruction.fields,
        )

    bundle_slot_signatures: dict[tuple[str, ...], str] = {}
    for bundle_format in table.bundle_formats:
        if bundle_format.bit_count % 8:
            raise ValueError(f"{bundle_format.name}: bundle width is not byte aligned")
        for field in bundle_format.fields:
            if field.slot not in slot_bit_counts:
                raise ValueError(
                    f"{bundle_format.name}: unknown physical slot {field.slot}"
                )
            expected_value_mask = (1 << slot_bit_counts[field.slot]) - 1
            if field.value_mask != expected_value_mask:
                raise ValueError(
                    f"{bundle_format.name}.{field.slot}: mapping covers "
                    f"{field.value_mask:#x}, expected {expected_value_mask:#x}"
                )
        _validate_encoding(
            bundle_format.name,
            bundle_format.bit_count,
            bundle_format.fixed_mask,
            bundle_format.fixed_value,
            bundle_format.fields,
        )
        slot_signature = tuple(sorted(field.slot for field in bundle_format.fields))
        previous_format = bundle_slot_signatures.get(slot_signature)
        if previous_format is not None:
            raise ValueError(
                f"bundle slot signature {slot_signature} is shared by "
                f"{previous_format} and {bundle_format.name}"
            )
        bundle_slot_signatures[slot_signature] = bundle_format.name

    for left_index, left in enumerate(table.bundle_formats):
        for right in table.bundle_formats[left_index + 1 :]:
            common_bit_count = min(left.bit_count, right.bit_count)
            common_mask = (1 << common_bit_count) - 1
            if fixed_patterns_overlap(
                left.fixed_mask & common_mask,
                left.fixed_value & common_mask,
                right.fixed_mask & common_mask,
                right.fixed_value & common_mask,
            ):
                raise ValueError(
                    f"bundle prefix decode is ambiguous between "
                    f"{left.name} and {right.name}"
                )


def scatter_bits(container: int, mappings: Sequence[BitMapping], value: int) -> int:
    """Scatters a value through an arbitrary target bit mapping."""
    for mapping in mappings:
        target_mask = 1 << mapping.target_bit
        if value & (1 << mapping.value_bit):
            container |= target_mask
        else:
            container &= ~target_mask
    return container


def gather_bits(container: int, mappings: Sequence[BitMapping]) -> int:
    """Gathers one invertible field mapping."""
    result = 0
    for mapping in mappings:
        bit = (container >> mapping.target_bit) & 1
        result |= bit << mapping.value_bit
    return result


def _instruction_by_name(table: EncodingTable, name: str) -> InstructionEncoding:
    return next(row for row in table.instructions if row.name == name)


def _bundle_by_name(table: EncodingTable, name: str) -> BundleFormatEncoding:
    return next(row for row in table.bundle_formats if row.name == name)


def encode_instruction(
    table: EncodingTable,
    instance: InstructionInstance,
) -> tuple[InstructionEncoding, int]:
    """Encodes one instruction from target-adapted field values."""
    instruction = _instruction_by_name(table, instance.instruction_name)
    field_values = dict(instance.field_values)
    if len(field_values) != len(instance.field_values):
        raise ValueError(f"{instruction.name}: repeated field value")
    expected_fields = {field.name for field in instruction.fields}
    if set(field_values) != expected_fields:
        raise ValueError(
            f"{instruction.name}: fields {sorted(field_values)} do not match "
            f"{sorted(expected_fields)}"
        )

    result = instruction.fixed_value
    for field in instruction.fields:
        value = field_values[field.name]
        if value < 0 or value & ~field.value_mask:
            raise ValueError(
                f"{instruction.name}.{field.name}: value {value} exceeds "
                f"mask {field.value_mask:#x}"
            )
        result = scatter_bits(result, field.mappings, value)
    return instruction, result


def decode_instruction_fields(
    instruction: InstructionEncoding,
    encoded_value: int,
) -> dict[str, int]:
    """Decodes fields for a known instruction, including domain constraints."""
    if encoded_value & instruction.fixed_mask != instruction.fixed_value:
        raise ValueError(f"{instruction.name}: fixed bits do not match")
    return {
        field.name: gather_bits(encoded_value, field.mappings)
        for field in instruction.fields
    }


def instruction_candidates(
    table: EncodingTable,
    slot: str,
    encoded_value: int,
) -> tuple[InstructionEncoding, ...]:
    """Returns all fixed-pattern decode candidates for a physical slot."""
    candidates: list[InstructionEncoding] = []
    for instruction in table.instructions:
        if instruction.slot != slot:
            continue
        if encoded_value & instruction.fixed_mask != instruction.fixed_value:
            continue
        candidates.append(instruction)
    return tuple(candidates)


def encode_bundle(table: EncodingTable, instance: BundleInstance) -> bytes:
    """Encodes one variable-width VLIW bundle."""
    bundle_format = _bundle_by_name(table, instance.format_name)
    fields_by_slot = {field.slot: field for field in bundle_format.fields}
    if len(fields_by_slot) != len(bundle_format.fields):
        raise ValueError(f"{bundle_format.name}: repeated physical slot")

    encoded_slots: dict[str, int] = {}
    for instruction_instance in instance.instructions:
        instruction, encoded_value = encode_instruction(table, instruction_instance)
        if instruction.slot in encoded_slots:
            raise ValueError(
                f"{bundle_format.name}: repeated {instruction.slot} instruction"
            )
        if instruction.slot not in fields_by_slot:
            raise ValueError(
                f"{bundle_format.name}: no {instruction.slot} bundle field"
            )
        encoded_slots[instruction.slot] = encoded_value
    if set(encoded_slots) != set(fields_by_slot):
        raise ValueError(
            f"{bundle_format.name}: occupied slots {sorted(encoded_slots)} do not "
            f"match required slots {sorted(fields_by_slot)}"
        )

    result = bundle_format.fixed_value
    for field in bundle_format.fields:
        result = scatter_bits(result, field.mappings, encoded_slots[field.slot])
    return result.to_bytes(bundle_format.bit_count // 8, "little")


def encode_witness(table: EncodingTable, witness: EncodingWitness) -> bytes:
    """Encodes all packets in one retained target witness."""
    return b"".join(encode_bundle(table, bundle) for bundle in witness.bundles)
