# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU ISA XML importer for target-low descriptor source data.

The vendor XML is a fact source, not a Loom descriptor language. This module
parses architecture, encoding, instruction, operand, and functional-group facts
into a small normalized model. Target-low semantics, scheduling, register
allocation policy, and descriptor selection belong in separate Loom-owned
overlays keyed by the normalized instruction facts produced here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree


class AmdgpuIsaXmlError(ValueError):
    """Raised when AMDGPU ISA XML is malformed or missing required facts."""


@dataclass(frozen=True, slots=True)
class AmdgpuIsaBitRange:
    order: int
    bit_count: int
    bit_offset: int
    padding_bit_count: int = 0
    padding_value: int = 0


@dataclass(frozen=True, slots=True)
class AmdgpuIsaEncodingField:
    name: str
    is_conditional: bool
    ranges: tuple[AmdgpuIsaBitRange, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaEncoding:
    name: str
    order: int
    bit_count: int
    identifier_mask: int
    identifier_values: tuple[int, ...]
    fields: tuple[AmdgpuIsaEncodingField, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaInstructionFlags:
    is_branch: bool
    is_conditional_branch: bool
    is_indirect_branch: bool
    is_program_terminator: bool
    is_immediately_executed: bool


@dataclass(frozen=True, slots=True)
class AmdgpuIsaOperand:
    order: int
    field_name: str | None
    data_format_name: str | None
    operand_type: str
    size_bits: int
    is_input: bool
    is_output: bool
    is_implicit: bool
    is_binary_microcode_required: bool


@dataclass(frozen=True, slots=True)
class AmdgpuIsaPredefinedValue:
    name: str
    value: int


@dataclass(frozen=True, slots=True)
class AmdgpuIsaOperandType:
    name: str
    is_partitioned: bool
    predefined_values: tuple[AmdgpuIsaPredefinedValue, ...]
    fields: tuple[AmdgpuIsaEncodingField, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaInstructionEncoding:
    encoding_name: str
    condition_name: str
    opcode: int
    operands: tuple[AmdgpuIsaOperand, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaFunctionalGroup:
    name: str
    subgroups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaInstruction:
    name: str
    aliases: tuple[str, ...]
    flags: AmdgpuIsaInstructionFlags
    encodings: tuple[AmdgpuIsaInstructionEncoding, ...]
    functional_groups: tuple[AmdgpuIsaFunctionalGroup, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuIsaInstructionEncodingSummary:
    instruction_name: str
    aliases: tuple[str, ...]
    encoding_name: str
    condition_name: str
    opcode: int
    operand_types: tuple[str, ...]
    data_format_names: tuple[str, ...]
    functional_groups: tuple[str, ...]
    functional_subgroups: tuple[str, ...]
    explicit_operand_count: int
    implicit_operand_count: int


@dataclass(frozen=True, slots=True)
class AmdgpuIsaEncodingFieldSummary:
    encoding_name: str
    field_name: str
    is_conditional: bool
    ranges: tuple[AmdgpuIsaBitRange, ...]
    total_bit_count: int
    total_padding_bit_count: int


class AmdgpuIsaFactSource(Protocol):
    @property
    def source_name(self) -> str: ...

    @property
    def architecture_name(self) -> str: ...

    @property
    def architecture_id(self) -> int: ...

    @property
    def encodings(self) -> tuple[AmdgpuIsaEncoding, ...]: ...

    @property
    def instructions(self) -> tuple[AmdgpuIsaInstruction, ...]: ...

    @property
    def operand_types(self) -> tuple[AmdgpuIsaOperandType, ...]: ...

    def encoding_map(self) -> dict[str, AmdgpuIsaEncoding]: ...

    def select_encodings(
        self, names: Iterable[str]
    ) -> tuple[AmdgpuIsaEncoding, ...]: ...

    def encoding_field_summaries(
        self, names: Iterable[str] | None = None
    ) -> tuple[AmdgpuIsaEncodingFieldSummary, ...]: ...

    def instruction_map(
        self, *, include_aliases: bool = False
    ) -> dict[str, AmdgpuIsaInstruction]: ...

    def select_instructions(
        self,
        names: Iterable[str],
        *,
        include_aliases: bool = True,
    ) -> tuple[AmdgpuIsaInstruction, ...]: ...

    def operand_type_map(self) -> dict[str, AmdgpuIsaOperandType]: ...

    def operand_predefined_value(
        self, operand_type_name: str, value_name: str
    ) -> int: ...

    def instruction_encoding_summaries(
        self,
        names: Iterable[str] | None = None,
        *,
        include_aliases: bool = True,
    ) -> tuple[AmdgpuIsaInstructionEncodingSummary, ...]: ...


@dataclass(frozen=True, slots=True)
class AmdgpuIsaSpec:
    source_name: str
    architecture_name: str
    architecture_id: int
    encodings: tuple[AmdgpuIsaEncoding, ...]
    instructions: tuple[AmdgpuIsaInstruction, ...]
    operand_types: tuple[AmdgpuIsaOperandType, ...]
    _encoding_lookup_cache: dict[str, AmdgpuIsaEncoding] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _instruction_lookup_cache: dict[bool, dict[str, AmdgpuIsaInstruction]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _operand_type_lookup_cache: dict[str, AmdgpuIsaOperandType] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def _encoding_lookup(self) -> dict[str, AmdgpuIsaEncoding]:
        lookup = self._encoding_lookup_cache
        if lookup is None:
            lookup = {encoding.name: encoding for encoding in self.encodings}
            object.__setattr__(self, "_encoding_lookup_cache", lookup)
        return lookup

    def encoding_map(self) -> dict[str, AmdgpuIsaEncoding]:
        return dict(self._encoding_lookup())

    def select_encodings(self, names: Iterable[str]) -> tuple[AmdgpuIsaEncoding, ...]:
        requested_names = tuple(names)
        encodings_by_name = self._encoding_lookup()
        selected: dict[str, AmdgpuIsaEncoding] = {}
        missing_names: list[str] = []
        for name in requested_names:
            encoding = encodings_by_name.get(name)
            if encoding is None:
                missing_names.append(name)
            else:
                selected[encoding.name] = encoding
        if missing_names:
            missing_text = ", ".join(sorted(missing_names))
            raise AmdgpuIsaXmlError(
                f"{self.source_name}: unknown AMDGPU ISA encoding(s): {missing_text}"
            )
        return tuple(
            selected[encoding.name]
            for encoding in self.encodings
            if encoding.name in selected
        )

    def encoding_field_summaries(
        self, names: Iterable[str] | None = None
    ) -> tuple[AmdgpuIsaEncodingFieldSummary, ...]:
        if names is None:
            encodings = self.encodings
        else:
            encodings = self.select_encodings(names)

        summaries = [
            _summarize_encoding_field(encoding, field)
            for encoding in encodings
            for field in encoding.fields
        ]
        return tuple(
            sorted(
                summaries,
                key=lambda summary: (
                    summary.encoding_name,
                    summary.field_name,
                ),
            )
        )

    def _instruction_lookup(
        self, *, include_aliases: bool = False
    ) -> dict[str, AmdgpuIsaInstruction]:
        cache = self._instruction_lookup_cache
        lookup = cache.get(include_aliases)
        if lookup is not None:
            return lookup
        instructions: dict[str, AmdgpuIsaInstruction] = {}
        for instruction in self.instructions:
            _insert_unique_instruction_name(
                instructions, instruction.name, instruction, self.source_name
            )
            if include_aliases:
                for alias in instruction.aliases:
                    _insert_unique_instruction_name(
                        instructions, alias, instruction, self.source_name
                    )
        cache[include_aliases] = instructions
        return instructions

    def instruction_map(
        self, *, include_aliases: bool = False
    ) -> dict[str, AmdgpuIsaInstruction]:
        return dict(self._instruction_lookup(include_aliases=include_aliases))

    def select_instructions(
        self,
        names: Iterable[str],
        *,
        include_aliases: bool = True,
    ) -> tuple[AmdgpuIsaInstruction, ...]:
        requested_names = tuple(names)
        instructions_by_name = self._instruction_lookup(include_aliases=include_aliases)
        selected: dict[str, AmdgpuIsaInstruction] = {}
        missing_names: list[str] = []
        for name in requested_names:
            instruction = instructions_by_name.get(name)
            if instruction is None:
                missing_names.append(name)
            else:
                selected[instruction.name] = instruction
        if missing_names:
            missing_text = ", ".join(sorted(missing_names))
            raise AmdgpuIsaXmlError(
                f"{self.source_name}: unknown AMDGPU ISA instruction(s): {missing_text}"
            )
        return tuple(selected[name] for name in sorted(selected))

    def _operand_type_lookup(self) -> dict[str, AmdgpuIsaOperandType]:
        lookup = self._operand_type_lookup_cache
        if lookup is None:
            lookup = {
                operand_type.name: operand_type for operand_type in self.operand_types
            }
            object.__setattr__(self, "_operand_type_lookup_cache", lookup)
        return lookup

    def operand_type_map(self) -> dict[str, AmdgpuIsaOperandType]:
        return dict(self._operand_type_lookup())

    def operand_predefined_value(self, operand_type_name: str, value_name: str) -> int:
        operand_type = self._operand_type_lookup().get(operand_type_name)
        if operand_type is None:
            raise AmdgpuIsaXmlError(
                f"{self.source_name}: unknown AMDGPU ISA operand type "
                f"'{operand_type_name}'"
            )
        predefined_values = {
            predefined_value.name: predefined_value.value
            for predefined_value in operand_type.predefined_values
        }
        value = predefined_values.get(value_name)
        if value is None:
            raise AmdgpuIsaXmlError(
                f"{self.source_name}: AMDGPU ISA operand type "
                f"'{operand_type_name}' has no predefined value '{value_name}'"
            )
        return value

    def instruction_encoding_summaries(
        self,
        names: Iterable[str] | None = None,
        *,
        include_aliases: bool = True,
    ) -> tuple[AmdgpuIsaInstructionEncodingSummary, ...]:
        if names is None:
            instructions = self.instructions
        else:
            instructions = self.select_instructions(
                names, include_aliases=include_aliases
            )

        summaries = [
            _summarize_instruction_encoding(instruction, encoding)
            for instruction in instructions
            for encoding in instruction.encodings
        ]
        return tuple(
            sorted(
                summaries,
                key=lambda summary: (
                    summary.instruction_name,
                    summary.encoding_name,
                    summary.condition_name,
                    summary.opcode,
                ),
            )
        )


def parse_amdgpu_isa_xml_path(path: str | Path) -> AmdgpuIsaSpec:
    xml_path = Path(path)
    try:
        tree = ElementTree.parse(xml_path)
    except ElementTree.ParseError as exc:
        raise AmdgpuIsaXmlError(f"{xml_path}: malformed AMDGPU ISA XML: {exc}") from exc
    return _parse_spec_root(tree.getroot(), str(xml_path))


def parse_amdgpu_isa_xml_text(
    xml_text: str,
    *,
    source_name: str = "<string>",
) -> AmdgpuIsaSpec:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise AmdgpuIsaXmlError(
            f"{source_name}: malformed AMDGPU ISA XML: {exc}"
        ) from exc
    return _parse_spec_root(root, source_name)


def compose_amdgpu_isa_partitioned_field(
    base_field: AmdgpuIsaEncodingField,
    operand_field: AmdgpuIsaEncodingField,
) -> AmdgpuIsaEncodingField | None:
    """Returns the operand subfield mapped into the carrier field, if encodable."""

    base_segments = _base_field_source_segments(base_field)
    if base_segments is None:
        return None
    composed_ranges: list[AmdgpuIsaBitRange] = []
    for operand_range in operand_field.ranges:
        remaining_bits = operand_range.bit_count
        source_bit_offset = operand_range.bit_offset
        pending_padding_bit_count = operand_range.padding_bit_count
        while remaining_bits > 0:
            matching_segment = None
            for segment in base_segments:
                source_start, source_end, target_bit_offset = segment
                if source_start <= source_bit_offset < source_end:
                    matching_segment = segment
                    break
            if matching_segment is None:
                return None
            source_start, source_end, target_bit_offset = matching_segment
            chunk_bit_count = min(remaining_bits, source_end - source_bit_offset)
            composed_ranges.append(
                AmdgpuIsaBitRange(
                    order=len(composed_ranges),
                    bit_count=chunk_bit_count,
                    bit_offset=target_bit_offset + source_bit_offset - source_start,
                    padding_bit_count=pending_padding_bit_count,
                    padding_value=operand_range.padding_value,
                )
            )
            source_bit_offset += chunk_bit_count
            remaining_bits -= chunk_bit_count
            pending_padding_bit_count = 0
    return AmdgpuIsaEncodingField(
        name=operand_field.name,
        is_conditional=operand_field.is_conditional,
        ranges=tuple(composed_ranges),
    )


def _base_field_source_segments(
    base_field: AmdgpuIsaEncodingField,
) -> tuple[tuple[int, int, int], ...] | None:
    source_bit_offset = 0
    segments = []
    for bit_range in base_field.ranges:
        if bit_range.padding_bit_count != 0:
            return None
        source_end = source_bit_offset + bit_range.bit_count
        segments.append((source_bit_offset, source_end, bit_range.bit_offset))
        source_bit_offset = source_end
    return tuple(segments)


def _parse_spec_root(root: ElementTree.Element, source_name: str) -> AmdgpuIsaSpec:
    if root.tag != "Spec":
        raise AmdgpuIsaXmlError(
            f"{source_name}: expected root element <Spec>, found <{root.tag}>"
        )

    isa_element = _required_child(root, "ISA", source_name)
    architecture_element = _required_child(isa_element, "Architecture", source_name)
    architecture_name = _required_text(
        architecture_element, "ArchitectureName", source_name
    )
    architecture_id = _required_integer(
        architecture_element, "ArchitectureId", source_name
    )

    encodings_element = _required_child(isa_element, "Encodings", source_name)
    encodings = tuple(
        sorted(
            (
                _parse_encoding(encoding_element, source_name)
                for encoding_element in encodings_element.findall("Encoding")
            ),
            key=lambda encoding: (encoding.order, encoding.name),
        )
    )
    if not encodings:
        raise AmdgpuIsaXmlError(f"{source_name}: <Encodings> has no <Encoding> rows")
    _ensure_unique_names(
        (encoding.name for encoding in encodings),
        "encoding",
        source_name,
    )

    instructions_element = _required_child(isa_element, "Instructions", source_name)
    instructions = tuple(
        sorted(
            (
                _parse_instruction(instruction_element, source_name)
                for instruction_element in instructions_element.findall("Instruction")
            ),
            key=lambda instruction: instruction.name,
        )
    )
    if not instructions:
        raise AmdgpuIsaXmlError(
            f"{source_name}: <Instructions> has no <Instruction> rows"
        )
    _ensure_unique_names(
        (instruction.name for instruction in instructions),
        "instruction",
        source_name,
    )

    operand_types_element = _required_child(isa_element, "OperandTypes", source_name)
    operand_types = tuple(
        sorted(
            (
                _parse_operand_type(operand_type_element, source_name)
                for operand_type_element in operand_types_element.findall("OperandType")
            ),
            key=lambda operand_type: operand_type.name,
        )
    )
    if not operand_types:
        raise AmdgpuIsaXmlError(
            f"{source_name}: <OperandTypes> has no <OperandType> rows"
        )
    _ensure_unique_names(
        (operand_type.name for operand_type in operand_types),
        "operand type",
        source_name,
    )

    return AmdgpuIsaSpec(
        source_name=source_name,
        architecture_name=architecture_name,
        architecture_id=architecture_id,
        encodings=encodings,
        instructions=instructions,
        operand_types=operand_types,
    )


def _parse_encoding(
    encoding_element: ElementTree.Element, source_name: str
) -> AmdgpuIsaEncoding:
    context = _context("Encoding", encoding_element.get("Order"))
    name = _required_text(encoding_element, "EncodingName", source_name)
    order = _required_integer_attribute(encoding_element, "Order", source_name, context)
    bit_count = _required_integer(encoding_element, "BitCount", source_name)
    identifier_mask_element = _required_child(
        encoding_element, "EncodingIdentifierMask", source_name
    )
    identifier_mask = _parse_integer_element(
        identifier_mask_element, source_name, f"{context}/EncodingIdentifierMask"
    )
    identifier_values_element = _required_child(
        encoding_element, "EncodingIdentifiers", source_name
    )
    identifier_values = tuple(
        _parse_integer_element(
            identifier_element,
            source_name,
            f"{context}/EncodingIdentifiers/EncodingIdentifier",
        )
        for identifier_element in identifier_values_element.findall(
            "EncodingIdentifier"
        )
    )
    if not identifier_values:
        raise AmdgpuIsaXmlError(f"{source_name}: {context} has no identifiers")
    fields = _parse_encoding_fields(encoding_element, source_name, context)
    return AmdgpuIsaEncoding(
        name=name,
        order=order,
        bit_count=bit_count,
        identifier_mask=identifier_mask,
        identifier_values=identifier_values,
        fields=fields,
    )


def _parse_encoding_fields(
    encoding_element: ElementTree.Element,
    source_name: str,
    encoding_context: str,
) -> tuple[AmdgpuIsaEncodingField, ...]:
    microcode_format_element = _required_child(
        encoding_element, "MicrocodeFormat", source_name
    )
    bitmap_element = _required_child(microcode_format_element, "BitMap", source_name)
    fields = tuple(
        _parse_encoding_field(field_element, source_name, encoding_context)
        for field_element in bitmap_element.findall("Field")
    )
    if not fields:
        raise AmdgpuIsaXmlError(f"{source_name}: {encoding_context} has no fields")
    _ensure_unique_names(
        (field.name for field in fields),
        f"{encoding_context} field",
        source_name,
    )
    return fields


def _parse_encoding_field(
    field_element: ElementTree.Element,
    source_name: str,
    encoding_context: str,
) -> AmdgpuIsaEncodingField:
    name = _required_text(field_element, "FieldName", source_name)
    context = f"{encoding_context}/Field({name})"
    bit_layout_element = _required_child(field_element, "BitLayout", source_name)
    expected_range_count = _required_integer_attribute(
        bit_layout_element, "RangeCount", source_name, context
    )
    ranges = tuple(
        sorted(
            (
                _parse_bit_range(range_element, source_name, context)
                for range_element in bit_layout_element.findall("Range")
            ),
            key=lambda bit_range: bit_range.order,
        )
    )
    if expected_range_count != len(ranges):
        raise AmdgpuIsaXmlError(
            f"{source_name}: {context} declares RangeCount={expected_range_count} "
            f"but has {len(ranges)} ranges"
        )
    if not ranges:
        raise AmdgpuIsaXmlError(f"{source_name}: {context} has no bit ranges")
    _ensure_unique_names(
        (str(bit_range.order) for bit_range in ranges),
        f"{context} range order",
        source_name,
    )
    return AmdgpuIsaEncodingField(
        name=name,
        is_conditional=_required_boolean_attribute(
            field_element, "IsConditional", source_name, context
        ),
        ranges=ranges,
    )


def _parse_bit_range(
    range_element: ElementTree.Element,
    source_name: str,
    field_context: str,
) -> AmdgpuIsaBitRange:
    order = _required_integer_attribute(
        range_element, "Order", source_name, f"{field_context}/Range"
    )
    context = f"{field_context}/Range({order})"
    padding_bit_count = 0
    padding_value = 0
    padding_elements = range_element.findall("Padding")
    if len(padding_elements) > 1:
        raise AmdgpuIsaXmlError(f"{source_name}: {context} has multiple paddings")
    if padding_elements:
        padding_element = padding_elements[0]
        padding_bit_count = _required_integer(padding_element, "BitCount", source_name)
        padding_value_element = _required_child(padding_element, "Value", source_name)
        padding_value = _parse_integer_element(
            padding_value_element, source_name, f"{context}/Padding/Value"
        )
    return AmdgpuIsaBitRange(
        order=order,
        bit_count=_required_integer(range_element, "BitCount", source_name),
        bit_offset=_required_integer(range_element, "BitOffset", source_name),
        padding_bit_count=padding_bit_count,
        padding_value=padding_value,
    )


def _parse_instruction(
    instruction_element: ElementTree.Element, source_name: str
) -> AmdgpuIsaInstruction:
    name = _required_text(instruction_element, "InstructionName", source_name)
    context = _context("Instruction", name)
    aliases_element = instruction_element.find("AliasedInstructionNames")
    aliases: tuple[str, ...]
    if aliases_element is None:
        aliases = ()
    else:
        aliases = tuple(
            _required_element_text(alias_element, source_name, f"{context}/alias")
            for alias_element in aliases_element.findall("InstructionName")
        )
    flags = _parse_instruction_flags(
        _required_child(instruction_element, "InstructionFlags", source_name),
        source_name,
        context,
    )
    instruction_encodings_element = _required_child(
        instruction_element, "InstructionEncodings", source_name
    )
    encodings = tuple(
        _parse_instruction_encoding(encoding_element, source_name, context)
        for encoding_element in instruction_encodings_element.findall(
            "InstructionEncoding"
        )
    )
    if not encodings:
        raise AmdgpuIsaXmlError(f"{source_name}: {context} has no encodings")

    functional_groups = tuple(
        _parse_functional_group(functional_group_element, source_name, context)
        for functional_group_element in instruction_element.findall("FunctionalGroup")
    )
    if not functional_groups:
        raise AmdgpuIsaXmlError(f"{source_name}: {context} has no functional group")

    return AmdgpuIsaInstruction(
        name=name,
        aliases=aliases,
        flags=flags,
        encodings=encodings,
        functional_groups=functional_groups,
    )


def _parse_instruction_flags(
    flags_element: ElementTree.Element,
    source_name: str,
    context: str,
) -> AmdgpuIsaInstructionFlags:
    return AmdgpuIsaInstructionFlags(
        is_branch=_required_boolean(flags_element, "IsBranch", source_name, context),
        is_conditional_branch=_required_boolean(
            flags_element, "IsConditionalBranch", source_name, context
        ),
        is_indirect_branch=_required_boolean(
            flags_element, "IsIndirectBranch", source_name, context
        ),
        is_program_terminator=_required_boolean(
            flags_element, "IsProgramTerminator", source_name, context
        ),
        is_immediately_executed=_required_boolean(
            flags_element, "IsImmediatelyExecuted", source_name, context
        ),
    )


def _parse_instruction_encoding(
    encoding_element: ElementTree.Element,
    source_name: str,
    instruction_context: str,
) -> AmdgpuIsaInstructionEncoding:
    encoding_name = _required_text(encoding_element, "EncodingName", source_name)
    context = f"{instruction_context}/InstructionEncoding({encoding_name})"
    condition_name = _required_text(encoding_element, "EncodingCondition", source_name)
    opcode = _required_integer(encoding_element, "Opcode", source_name)
    operands_element = _required_child(encoding_element, "Operands", source_name)
    operands = tuple(
        sorted(
            (
                _parse_operand(operand_element, source_name, context)
                for operand_element in operands_element.findall("Operand")
            ),
            key=lambda operand: operand.order,
        )
    )
    _ensure_unique_names(
        (str(operand.order) for operand in operands),
        f"{context} operand order",
        source_name,
    )
    return AmdgpuIsaInstructionEncoding(
        encoding_name=encoding_name,
        condition_name=condition_name,
        opcode=opcode,
        operands=operands,
    )


def _parse_operand(
    operand_element: ElementTree.Element,
    source_name: str,
    encoding_context: str,
) -> AmdgpuIsaOperand:
    order = _required_integer_attribute(
        operand_element, "Order", source_name, f"{encoding_context}/Operand"
    )
    context = f"{encoding_context}/Operand({order})"
    return AmdgpuIsaOperand(
        order=order,
        field_name=_optional_text(operand_element, "FieldName"),
        data_format_name=_optional_text(operand_element, "DataFormatName"),
        operand_type=_required_text(operand_element, "OperandType", source_name),
        size_bits=_required_integer(operand_element, "OperandSize", source_name),
        is_input=_required_boolean_attribute(
            operand_element, "Input", source_name, context
        ),
        is_output=_required_boolean_attribute(
            operand_element, "Output", source_name, context
        ),
        is_implicit=_required_boolean_attribute(
            operand_element, "IsImplicit", source_name, context
        ),
        is_binary_microcode_required=_required_boolean_attribute(
            operand_element,
            "IsBinaryMicrocodeRequired",
            source_name,
            context,
        ),
    )


def _parse_operand_type(
    operand_type_element: ElementTree.Element,
    source_name: str,
) -> AmdgpuIsaOperandType:
    name = _required_text(operand_type_element, "OperandTypeName", source_name)
    context = _context("OperandType", name)
    is_partitioned = _required_boolean_attribute(
        operand_type_element, "IsPartitioned", source_name, context
    )
    microcode_format_element = operand_type_element.find("MicrocodeFormat")
    fields: tuple[AmdgpuIsaEncodingField, ...]
    if microcode_format_element is None:
        fields = ()
    else:
        bitmap_element = _required_child(
            microcode_format_element, "BitMap", source_name
        )
        fields = tuple(
            _parse_encoding_field(field_element, source_name, context)
            for field_element in bitmap_element.findall("Field")
        )
        if not fields:
            raise AmdgpuIsaXmlError(f"{source_name}: {context} has no fields")
        _ensure_unique_names(
            (field.name for field in fields),
            f"{context} field",
            source_name,
        )
    if is_partitioned and not fields:
        raise AmdgpuIsaXmlError(
            f"{source_name}: partitioned {context} has no microcode fields"
        )
    if not is_partitioned and fields:
        raise AmdgpuIsaXmlError(
            f"{source_name}: non-partitioned {context} has microcode fields"
        )
    predefined_values_element = operand_type_element.find("OperandPredefinedValues")
    predefined_values: tuple[AmdgpuIsaPredefinedValue, ...]
    if predefined_values_element is None:
        predefined_values = ()
    else:
        predefined_values = _normalize_predefined_values(
            (
                _parse_predefined_value(predefined_value_element, source_name, context)
                for predefined_value_element in predefined_values_element.findall(
                    "PredefinedValue"
                )
            ),
            source_name,
            context,
        )
    return AmdgpuIsaOperandType(
        name=name,
        is_partitioned=is_partitioned,
        predefined_values=predefined_values,
        fields=fields,
    )


def _parse_predefined_value(
    predefined_value_element: ElementTree.Element,
    source_name: str,
    operand_type_context: str,
) -> AmdgpuIsaPredefinedValue:
    name = _required_text(predefined_value_element, "Name", source_name)
    return AmdgpuIsaPredefinedValue(
        name=name,
        value=_required_integer(predefined_value_element, "Value", source_name),
    )


def _normalize_predefined_values(
    predefined_values: Iterable[AmdgpuIsaPredefinedValue],
    source_name: str,
    operand_type_context: str,
) -> tuple[AmdgpuIsaPredefinedValue, ...]:
    values_by_name: dict[str, int] = {}
    for predefined_value in predefined_values:
        existing_value = values_by_name.get(predefined_value.name)
        if existing_value is not None:
            if existing_value != predefined_value.value:
                raise AmdgpuIsaXmlError(
                    f"{source_name}: {operand_type_context} has conflicting "
                    f"predefined value '{predefined_value.name}' values "
                    f"{existing_value} and {predefined_value.value}"
                )
            continue
        values_by_name[predefined_value.name] = predefined_value.value
    return tuple(
        AmdgpuIsaPredefinedValue(name, value)
        for name, value in sorted(values_by_name.items())
    )


def _parse_functional_group(
    functional_group_element: ElementTree.Element,
    source_name: str,
    instruction_context: str,
) -> AmdgpuIsaFunctionalGroup:
    name = _required_text(functional_group_element, "Name", source_name)
    subgroups_element = _required_child(
        functional_group_element, "FunctionalSubgroups", source_name
    )
    subgroups = tuple(
        _required_element_text(
            subgroup_element,
            source_name,
            f"{instruction_context}/FunctionalGroup({name})/Subgroup",
        )
        for subgroup_element in subgroups_element.findall("Subgroup")
    )
    if not subgroups:
        raise AmdgpuIsaXmlError(
            f"{source_name}: {instruction_context}/FunctionalGroup({name}) "
            "has no subgroups"
        )
    return AmdgpuIsaFunctionalGroup(name=name, subgroups=subgroups)


def _summarize_encoding_field(
    encoding: AmdgpuIsaEncoding,
    field: AmdgpuIsaEncodingField,
) -> AmdgpuIsaEncodingFieldSummary:
    return AmdgpuIsaEncodingFieldSummary(
        encoding_name=encoding.name,
        field_name=field.name,
        is_conditional=field.is_conditional,
        ranges=field.ranges,
        total_bit_count=sum(bit_range.bit_count for bit_range in field.ranges),
        total_padding_bit_count=sum(
            bit_range.padding_bit_count for bit_range in field.ranges
        ),
    )


def _summarize_instruction_encoding(
    instruction: AmdgpuIsaInstruction,
    encoding: AmdgpuIsaInstructionEncoding,
) -> AmdgpuIsaInstructionEncodingSummary:
    explicit_operand_count = sum(
        1 for operand in encoding.operands if not operand.is_implicit
    )
    implicit_operand_count = sum(
        1 for operand in encoding.operands if operand.is_implicit
    )
    functional_group_names = tuple(
        sorted({group.name for group in instruction.functional_groups})
    )
    functional_subgroup_names = tuple(
        sorted(
            {
                subgroup
                for group in instruction.functional_groups
                for subgroup in group.subgroups
            }
        )
    )
    return AmdgpuIsaInstructionEncodingSummary(
        instruction_name=instruction.name,
        aliases=instruction.aliases,
        encoding_name=encoding.encoding_name,
        condition_name=encoding.condition_name,
        opcode=encoding.opcode,
        operand_types=tuple(
            sorted({operand.operand_type for operand in encoding.operands})
        ),
        data_format_names=tuple(
            sorted(
                {
                    operand.data_format_name
                    for operand in encoding.operands
                    if operand.data_format_name is not None
                }
            )
        ),
        functional_groups=functional_group_names,
        functional_subgroups=functional_subgroup_names,
        explicit_operand_count=explicit_operand_count,
        implicit_operand_count=implicit_operand_count,
    )


def _insert_unique_instruction_name(
    instructions: dict[str, AmdgpuIsaInstruction],
    name: str,
    instruction: AmdgpuIsaInstruction,
    source_name: str,
) -> None:
    existing_instruction = instructions.get(name)
    if (
        existing_instruction is not None
        and existing_instruction.name != instruction.name
    ):
        raise AmdgpuIsaXmlError(
            f"{source_name}: AMDGPU ISA instruction name or alias '{name}' is "
            f"ambiguous between '{existing_instruction.name}' and "
            f"'{instruction.name}'"
        )
    instructions[name] = instruction


def _ensure_unique_names(names: Iterable[str], noun: str, source_name: str) -> None:
    seen_names: set[str] = set()
    for name in names:
        if name in seen_names:
            raise AmdgpuIsaXmlError(
                f"{source_name}: duplicate AMDGPU ISA {noun} '{name}'"
            )
        seen_names.add(name)


def _required_child(
    element: ElementTree.Element, tag: str, source_name: str
) -> ElementTree.Element:
    child = element.find(tag)
    if child is None:
        raise AmdgpuIsaXmlError(
            f"{source_name}: expected <{tag}> under <{element.tag}>"
        )
    return child


def _optional_text(element: ElementTree.Element, tag: str) -> str | None:
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    if not text:
        return None
    return text


def _required_text(element: ElementTree.Element, tag: str, source_name: str) -> str:
    child = _required_child(element, tag, source_name)
    return _required_element_text(child, source_name, f"{element.tag}/{tag}")


def _required_element_text(
    element: ElementTree.Element,
    source_name: str,
    context: str,
) -> str:
    if element.text is None:
        raise AmdgpuIsaXmlError(f"{source_name}: missing text for {context}")
    text = element.text.strip()
    if not text:
        raise AmdgpuIsaXmlError(f"{source_name}: empty text for {context}")
    return text


def _required_integer(element: ElementTree.Element, tag: str, source_name: str) -> int:
    child = _required_child(element, tag, source_name)
    return _parse_integer_element(child, source_name, f"{element.tag}/{tag}")


def _required_integer_attribute(
    element: ElementTree.Element,
    attribute_name: str,
    source_name: str,
    context: str,
) -> int:
    text = _required_attribute(element, attribute_name, source_name, context)
    return _parse_integer_text(text, 10, source_name, f"{context}@{attribute_name}")


def _parse_integer_element(
    element: ElementTree.Element,
    source_name: str,
    context: str,
) -> int:
    text = _required_element_text(element, source_name, context)
    radix_text = element.get("Radix")
    radix = (
        10
        if radix_text is None
        else _parse_integer_text(radix_text, 10, source_name, f"{context}@Radix")
    )
    return _parse_integer_text(text, radix, source_name, context)


def _parse_integer_text(
    text: str,
    radix: int,
    source_name: str,
    context: str,
) -> int:
    try:
        return int(text, radix)
    except ValueError as exc:
        raise AmdgpuIsaXmlError(
            f"{source_name}: expected base-{radix} integer for {context}, "
            f"found {text!r}"
        ) from exc


def _required_boolean(
    element: ElementTree.Element,
    tag: str,
    source_name: str,
    context: str,
) -> bool:
    text = _required_text(element, tag, source_name)
    return _parse_boolean_text(text, source_name, f"{context}/{tag}")


def _required_boolean_attribute(
    element: ElementTree.Element,
    attribute_name: str,
    source_name: str,
    context: str,
) -> bool:
    text = _required_attribute(element, attribute_name, source_name, context)
    return _parse_boolean_text(text, source_name, f"{context}@{attribute_name}")


def _parse_boolean_text(text: str, source_name: str, context: str) -> bool:
    normalized_text = text.lower()
    if normalized_text == "true":
        return True
    if normalized_text == "false":
        return False
    raise AmdgpuIsaXmlError(
        f"{source_name}: expected boolean for {context}, found {text!r}"
    )


def _required_attribute(
    element: ElementTree.Element,
    attribute_name: str,
    source_name: str,
    context: str,
) -> str:
    value = element.get(attribute_name)
    if value is None:
        raise AmdgpuIsaXmlError(
            f"{source_name}: expected attribute {attribute_name!r} on {context}"
        )
    return value


def _context(noun: str, name: str | None) -> str:
    if name is None:
        return noun
    return f"{noun}({name})"
