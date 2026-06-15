# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU ISA XML overlays for target-low descriptor materialization.

The ISA XML importer owns vendor facts. This module owns the Loom-specific
overlay that turns an allowlisted instruction encoding into a low descriptor.
Keeping those layers separate lets the vendor parser remain fact-only while
making semantic choices explicit, reviewable, and testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FORMAT_XML_NAMES_BY_ID,
    amdgpu_encoding_field_id,
    amdgpu_encoding_field_name,
    amdgpu_encoding_format_id,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaEncodingField,
    AmdgpuIsaFactSource,
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
    AmdgpuIsaOperand,
    AmdgpuIsaXmlError,
    compose_amdgpu_isa_partitioned_field,
)
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorAsmSurface,
    DescriptorFlag,
    Effect,
    EncodingFieldValue,
    Immediate,
    Operand,
    OperandFlag,
    OperandForm,
    OperandRole,
)


class AmdgpuDescriptorOverlayError(ValueError):
    """Raised when a Loom overlay does not match parsed AMDGPU ISA facts."""


_REGISTER_WIDTH_BITS = {
    "amdgpu.sgpr": 32,
    "amdgpu.vgpr": 32,
    "amdgpu.agpr": 32,
    "amdgpu.m0": 32,
    "amdgpu.scc": 1,
    "amdgpu.exec": 64,
    "amdgpu.mode": 32,
}

_REGISTER_PART_WIDTH_BITS = {
    "amdgpu.vgpr.low16": 16,
    "amdgpu.vgpr.high16": 16,
}


@dataclass(frozen=True, slots=True)
class AmdgpuOperandOverlay:
    xml_field_name: str
    descriptor_operand: Operand
    role_exception_reason: str | None = None
    size_exception_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AmdgpuOperandPredefinedValueRef:
    value_name: str
    operand_type: str | None = None


@dataclass(frozen=True, slots=True)
class AmdgpuEncodingFieldAllOnes:
    pass


AmdgpuFixedEncodingValue = (
    int | AmdgpuOperandPredefinedValueRef | AmdgpuEncodingFieldAllOnes
)


@dataclass(frozen=True, slots=True)
class AmdgpuIgnoredOperandOverlay:
    xml_field_name: str
    ignore_reason: str
    fixed_encoding_value: AmdgpuFixedEncodingValue | None = None


@dataclass(frozen=True, slots=True)
class AmdgpuImplicitOperandOverlay:
    operand_type: str
    descriptor_operand: Operand | None = None
    ignore_reason: str | None = None
    data_format_name: str | None = None
    size_bits: int | None = None
    is_input: bool | None = None
    is_output: bool | None = None
    xml_operand_required: bool = True


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorOverlay:
    descriptor_key: str
    instruction_name: str
    encoding_name: str
    semantic_tag: str
    schedule_class: str
    operands: tuple[AmdgpuOperandOverlay, ...]
    ignored_operands: tuple[AmdgpuIgnoredOperandOverlay, ...] = ()
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = ()
    encoding_condition: str = "default"
    mnemonic: str | None = None
    encoding_format_id: int | None = None
    encoding_id: int | None = None
    immediate_fields: tuple[str, ...] = ()
    immediates: tuple[Immediate, ...] = ()
    fixed_encoding_fields: tuple[tuple[str, AmdgpuFixedEncodingValue], ...] = ()
    effects: tuple[Effect, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    operand_forms: tuple[OperandForm, ...] = ()
    feature_mask_words: tuple[int, ...] = ()
    flags: tuple[DescriptorFlag, ...] = ()
    asm_forms: tuple[AsmForm, ...] | None = None
    asm_surface: DescriptorAsmSurface = DescriptorAsmSurface.AUTHORABLE
    asm_surface_reason: str = ""


def _asm_forms_for_overlay(overlay: AmdgpuDescriptorOverlay) -> tuple[AsmForm, ...]:
    if overlay.asm_forms is not None:
        return overlay.asm_forms

    results = []
    operands = []
    for operand_overlay in overlay.operands:
        operand = operand_overlay.descriptor_operand
        if operand.role is OperandRole.RESULT:
            results.append(operand.field_name)
        elif OperandFlag.IMPLICIT in operand.flags:
            continue
        elif operand.role in (
            OperandRole.OPERAND,
            OperandRole.PREDICATE,
            OperandRole.RESOURCE,
        ):
            operands.append(operand.field_name)
    return (
        AsmForm(
            results=tuple(results),
            operands=tuple(operands),
            immediates=tuple(
                AsmImmediate(immediate.field_name, name=immediate.field_name)
                for immediate in overlay.immediates
            ),
        ),
    )


def materialize_amdgpu_descriptor_overlay(
    spec: AmdgpuIsaFactSource, overlay: AmdgpuDescriptorOverlay
) -> Descriptor:
    instruction = _select_instruction(spec, overlay)
    encoding = _select_instruction_encoding(instruction, overlay)
    _validate_operand_overlay(spec, overlay, instruction, encoding)
    operands = tuple(
        _materialize_operand_overlay(operand_overlay)
        for operand_overlay in overlay.operands
    ) + tuple(
        implicit_overlay.descriptor_operand
        for implicit_overlay in overlay.implicit_operands
        if implicit_overlay.descriptor_operand is not None
    )
    return Descriptor(
        key=overlay.descriptor_key,
        mnemonic=overlay.mnemonic or overlay.instruction_name.lower(),
        semantic_tag=overlay.semantic_tag,
        operands=tuple(
            operand for operand in operands if operand.role is OperandRole.RESULT
        )
        + tuple(
            operand for operand in operands if operand.role is not OperandRole.RESULT
        ),
        immediates=_materialize_immediates(overlay),
        encoding_field_values=_materialize_encoding_field_values(
            spec, overlay, encoding
        ),
        asm_forms=_asm_forms_for_overlay(overlay),
        asm_surface=overlay.asm_surface,
        asm_surface_reason=overlay.asm_surface_reason,
        effects=overlay.effects,
        constraints=overlay.constraints,
        operand_forms=overlay.operand_forms,
        feature_mask_words=overlay.feature_mask_words,
        encoding_format_id=_encoding_format_id(overlay),
        encoding_id=encoding.opcode
        if overlay.encoding_id is None
        else overlay.encoding_id,
        schedule_class=overlay.schedule_class,
        flags=overlay.flags,
    )


def _encoding_format_id(overlay: AmdgpuDescriptorOverlay) -> int:
    if overlay.encoding_format_id is not None:
        return overlay.encoding_format_id
    try:
        return amdgpu_encoding_format_id(overlay.encoding_name)
    except KeyError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' references "
            f"unmapped AMDGPU encoding format '{overlay.encoding_name}'"
        ) from exc


def _immediate_field_is_synthetic_literal(
    overlay: AmdgpuDescriptorOverlay, immediate_field: str
) -> bool:
    if immediate_field != "LITERAL":
        return False
    format_name = AMDGPU_ENCODING_FORMAT_XML_NAMES_BY_ID.get(
        _encoding_format_id(overlay)
    )
    return format_name is not None and "INST_LITERAL" in format_name


def _materialize_operand_overlay(operand_overlay: AmdgpuOperandOverlay) -> Operand:
    try:
        return replace(
            operand_overlay.descriptor_operand,
            encoding_field_id=amdgpu_encoding_field_id(operand_overlay.xml_field_name),
        )
    except KeyError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"AMDGPU operand overlay references unmapped encoding field "
            f"'{operand_overlay.xml_field_name}'"
        ) from exc


def _materialize_immediates(
    overlay: AmdgpuDescriptorOverlay,
) -> tuple[Immediate, ...]:
    if len(overlay.immediate_fields) != len(overlay.immediates):
        return overlay.immediates
    try:
        return tuple(
            replace(immediate, encoding_field_id=amdgpu_encoding_field_id(field_name))
            for field_name, immediate in zip(
                overlay.immediate_fields, overlay.immediates, strict=False
            )
        )
    except KeyError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' references unmapped "
            "immediate encoding field"
        ) from exc


def _materialize_encoding_field_values(
    spec: AmdgpuIsaFactSource,
    overlay: AmdgpuDescriptorOverlay,
    encoding: AmdgpuIsaInstructionEncoding,
) -> tuple[EncodingFieldValue, ...]:
    xml_operands = _explicit_xml_operands_by_field_name(encoding, overlay)
    encoding_fields, _partition_carriers = _encoding_fields_and_partition_carriers(
        spec, overlay, xml_operands
    )
    field_values = [
        _materialize_encoding_field_value(
            spec, overlay, xml_operands, encoding_fields, field_name, value
        )
        for field_name, value in overlay.fixed_encoding_fields
    ]
    field_values.extend(
        _materialize_encoding_field_value(
            spec,
            overlay,
            xml_operands,
            encoding_fields,
            ignored_operand.xml_field_name,
            ignored_operand.fixed_encoding_value,
        )
        for ignored_operand in overlay.ignored_operands
        if ignored_operand.fixed_encoding_value is not None
    )
    return tuple(field_values)


def _encoding_fields_and_partition_carriers(
    spec: AmdgpuIsaFactSource,
    overlay: AmdgpuDescriptorOverlay,
    xml_operands: dict[str, AmdgpuIsaOperand],
) -> tuple[dict[str, AmdgpuIsaEncodingField], dict[str, str]]:
    microcode_encoding = spec.encoding_map().get(overlay.encoding_name)
    encoding_fields = (
        {field.name: field for field in microcode_encoding.fields}
        if microcode_encoding is not None
        else {}
    )
    partition_carriers: dict[str, str] = {}
    operand_types = spec.operand_type_map()
    for xml_operand in xml_operands.values():
        if not xml_operand.is_binary_microcode_required:
            continue
        operand_type = operand_types.get(xml_operand.operand_type)
        if operand_type is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                f"missing XML operand type '{xml_operand.operand_type}'"
            )
        if not operand_type.is_partitioned:
            continue
        if xml_operand.field_name is None:
            continue
        base_field = encoding_fields.get(xml_operand.field_name)
        if base_field is None:
            continue
        for field in operand_type.fields:
            existing_field = encoding_fields.get(field.name)
            if existing_field is not None:
                continue
            composed_field = compose_amdgpu_isa_partitioned_field(base_field, field)
            if composed_field is None:
                continue
            encoding_fields[field.name] = composed_field
            partition_carriers[field.name] = xml_operand.field_name
    return encoding_fields, partition_carriers


def _materialize_encoding_field_value(
    spec: AmdgpuIsaFactSource,
    overlay: AmdgpuDescriptorOverlay,
    xml_operands: dict[str, AmdgpuIsaOperand],
    encoding_fields: dict[str, AmdgpuIsaEncodingField],
    field_name: str,
    value: AmdgpuFixedEncodingValue,
) -> EncodingFieldValue:
    try:
        field_id = amdgpu_encoding_field_id(field_name)
    except KeyError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' references unmapped "
            f"encoding field '{field_name}'"
        ) from exc
    resolved_value = _resolve_fixed_encoding_value(
        spec, overlay, xml_operands, encoding_fields, field_name, value
    )
    _validate_fixed_encoding_field_width(
        overlay, encoding_fields, field_name, resolved_value
    )
    return EncodingFieldValue(field_id, resolved_value)


def _validate_fixed_encoding_field_width(
    overlay: AmdgpuDescriptorOverlay,
    encoding_fields: dict[str, AmdgpuIsaEncodingField],
    field_name: str,
    value: int,
) -> None:
    bit_count = _encoding_field_bit_count(encoding_fields, field_name)
    if bit_count is None:
        return
    if value < 0 or value >= (1 << bit_count):
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' fixed field "
            f"'{field_name}' value {value} does not fit in {bit_count} bits"
        )


def _resolve_fixed_encoding_value(
    spec: AmdgpuIsaFactSource,
    overlay: AmdgpuDescriptorOverlay,
    xml_operands: dict[str, AmdgpuIsaOperand],
    encoding_fields: dict[str, AmdgpuIsaEncodingField],
    field_name: str,
    value: AmdgpuFixedEncodingValue,
) -> int:
    if isinstance(value, int):
        return value

    if isinstance(value, AmdgpuEncodingFieldAllOnes):
        bit_count = _encoding_field_bit_count(encoding_fields, field_name)
        if bit_count is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' fixed field "
                f"'{field_name}' uses all-ones without an XML encoding field"
            )
        return (1 << bit_count) - 1

    operand_type = value.operand_type
    if operand_type is None:
        xml_operand = xml_operands.get(field_name)
        if xml_operand is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' fixed field "
                f"'{field_name}' uses predefined value '{value.value_name}' without "
                "an XML operand to infer the operand type"
            )
        operand_type = xml_operand.operand_type

    try:
        return spec.operand_predefined_value(operand_type, value.value_name)
    except AmdgpuIsaXmlError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' fixed field "
            f"'{field_name}' references AMDGPU predefined value "
            f"'{operand_type}.{value.value_name}'"
        ) from exc


def _encoding_field_bit_count(
    encoding_fields: dict[str, AmdgpuIsaEncodingField],
    field_name: str,
) -> int | None:
    encoding_field = encoding_fields.get(field_name)
    if encoding_field is None:
        return None
    return sum(bit_range.bit_count for bit_range in encoding_field.ranges)


def materialize_amdgpu_descriptor_overlays(
    spec: AmdgpuIsaFactSource, overlays: Iterable[AmdgpuDescriptorOverlay]
) -> tuple[Descriptor, ...]:
    return tuple(
        materialize_amdgpu_descriptor_overlay(spec, overlay) for overlay in overlays
    )


def _select_instruction(
    spec: AmdgpuIsaFactSource, overlay: AmdgpuDescriptorOverlay
) -> AmdgpuIsaInstruction:
    try:
        return spec.select_instructions(
            [overlay.instruction_name], include_aliases=True
        )[0]
    except AmdgpuIsaXmlError as exc:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' references unknown "
            f"AMDGPU instruction '{overlay.instruction_name}'"
        ) from exc


def _select_instruction_encoding(
    instruction: AmdgpuIsaInstruction, overlay: AmdgpuDescriptorOverlay
) -> AmdgpuIsaInstructionEncoding:
    matches = tuple(
        encoding
        for encoding in instruction.encodings
        if encoding.encoding_name == overlay.encoding_name
        and encoding.condition_name == overlay.encoding_condition
    )
    if not matches:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' could not find "
            f"encoding '{overlay.encoding_name}' condition "
            f"'{overlay.encoding_condition}' on instruction '{instruction.name}'"
        )
    if len(matches) > 1:
        first_match = matches[0]
        if any(match != first_match for match in matches[1:]):
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' found "
                f"conflicting duplicate encoding '{overlay.encoding_name}' "
                f"condition '{overlay.encoding_condition}' on instruction "
                f"'{instruction.name}'"
            )
        return first_match
    return matches[0]


def _validate_operand_overlay(
    spec: AmdgpuIsaFactSource,
    overlay: AmdgpuDescriptorOverlay,
    instruction: AmdgpuIsaInstruction,
    encoding: AmdgpuIsaInstructionEncoding,
) -> None:
    _validate_unique_overlay_fields(overlay)
    xml_operands = _explicit_xml_operands_by_field_name(encoding, overlay)
    encoding_fields, partition_carriers = _encoding_fields_and_partition_carriers(
        spec, overlay, xml_operands
    )
    covered_fields: set[str] = set()
    for operand_overlay in overlay.operands:
        xml_operand = xml_operands.get(operand_overlay.xml_field_name)
        if xml_operand is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                f"missing XML operand field '{operand_overlay.xml_field_name}' "
                f"on instruction '{instruction.name}' encoding "
                f"'{encoding.encoding_name}'"
            )
        _validate_operand_role(overlay, operand_overlay, xml_operand)
        _validate_operand_width(overlay, operand_overlay, xml_operand)
        covered_fields.add(operand_overlay.xml_field_name)

    for ignored_operand in overlay.ignored_operands:
        xml_operand = xml_operands.get(ignored_operand.xml_field_name)
        if xml_operand is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' ignores missing "
                f"XML operand field '{ignored_operand.xml_field_name}' on "
                f"instruction '{instruction.name}' encoding '{encoding.encoding_name}'"
            )
        if not ignored_operand.ignore_reason:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' ignores XML "
                f"operand field '{ignored_operand.xml_field_name}' without a "
                "named reason"
            )
        if (
            xml_operand.is_binary_microcode_required
            and ignored_operand.fixed_encoding_value is None
        ):
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' ignores binary "
                f"microcode field '{ignored_operand.xml_field_name}' without a "
                "fixed encoding value"
            )
        covered_fields.add(ignored_operand.xml_field_name)

    for immediate_field in overlay.immediate_fields:
        xml_operand = xml_operands.get(immediate_field)
        if xml_operand is not None:
            if not xml_operand.is_input or xml_operand.is_output:
                raise AmdgpuDescriptorOverlayError(
                    f"descriptor overlay '{overlay.descriptor_key}' immediate field "
                    f"'{immediate_field}' does not describe an input operand"
                )
            covered_fields.add(immediate_field)
        elif immediate_field in partition_carriers:
            covered_fields.add(partition_carriers[immediate_field])
        elif immediate_field not in encoding_fields:
            if _immediate_field_is_synthetic_literal(overlay, immediate_field):
                continue
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                f"missing immediate encoding field '{immediate_field}' on instruction "
                f"'{instruction.name}' encoding '{encoding.encoding_name}'"
            )
    for immediate in overlay.immediates:
        for encoding_slice in immediate.encoding_slices:
            try:
                field_name = amdgpu_encoding_field_name(
                    encoding_slice.encoding_field_id
                )
            except KeyError as exc:
                raise AmdgpuDescriptorOverlayError(
                    f"descriptor overlay '{overlay.descriptor_key}' immediate "
                    f"'{immediate.field_name}' references unmapped sliced "
                    f"encoding field id {encoding_slice.encoding_field_id}"
                ) from exc
            if field_name not in encoding_fields:
                raise AmdgpuDescriptorOverlayError(
                    f"descriptor overlay '{overlay.descriptor_key}' immediate "
                    f"'{immediate.field_name}' references missing sliced "
                    f"encoding field '{field_name}' on instruction "
                    f"'{instruction.name}' encoding '{encoding.encoding_name}'"
                )
            bit_count = _encoding_field_bit_count(encoding_fields, field_name)
            if bit_count is not None and encoding_slice.bit_count > bit_count:
                raise AmdgpuDescriptorOverlayError(
                    f"descriptor overlay '{overlay.descriptor_key}' immediate "
                    f"'{immediate.field_name}' slice for field '{field_name}' "
                    f"copies {encoding_slice.bit_count} bits into {bit_count}-bit field"
                )
    for field_name, _value in overlay.fixed_encoding_fields:
        xml_operand = xml_operands.get(field_name)
        if field_name not in encoding_fields and xml_operand is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                f"missing fixed encoding field '{field_name}' on instruction "
                f"'{instruction.name}' encoding '{encoding.encoding_name}'"
            )
        if xml_operand is not None:
            if not xml_operand.is_input or xml_operand.is_output:
                raise AmdgpuDescriptorOverlayError(
                    f"descriptor overlay '{overlay.descriptor_key}' fixed field "
                    f"'{field_name}' does not describe an input operand"
                )
            covered_fields.add(field_name)
        elif field_name in partition_carriers:
            covered_fields.add(partition_carriers[field_name])

    missing_fields = sorted(set(xml_operands) - covered_fields)
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' does not cover XML "
            f"operand field(s): {missing_text}"
        )
    _validate_implicit_operand_overlays(overlay, instruction, encoding)


def _validate_implicit_operand_overlays(
    overlay: AmdgpuDescriptorOverlay,
    instruction: AmdgpuIsaInstruction,
    encoding: AmdgpuIsaInstructionEncoding,
) -> None:
    xml_backed_overlays = tuple(
        implicit_overlay
        for implicit_overlay in overlay.implicit_operands
        if implicit_overlay.xml_operand_required
    )
    implicit_xml_operands = tuple(
        xml_operand for xml_operand in encoding.operands if xml_operand.is_implicit
    )
    if not implicit_xml_operands:
        if xml_backed_overlays:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' has implicit "
                f"operand decision(s), but instruction '{instruction.name}' "
                f"encoding '{encoding.encoding_name}' has no implicit operands"
            )
        return

    covered_orders: set[int] = set()
    for implicit_overlay in xml_backed_overlays:
        matches = tuple(
            xml_operand
            for xml_operand in implicit_xml_operands
            if _implicit_overlay_matches_xml_operand(implicit_overlay, xml_operand)
        )
        if not matches:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' references "
                "missing implicit XML operand "
                f"'{_format_implicit_overlay_key(implicit_overlay)}' on "
                f"instruction '{instruction.name}' encoding "
                f"'{encoding.encoding_name}'"
            )
        if len(matches) > 1:
            match_text = ", ".join(
                _format_implicit_xml_operand(xml_operand) for xml_operand in matches
            )
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' implicit "
                f"operand decision '{_format_implicit_overlay_key(implicit_overlay)}' "
                f"matches multiple XML operands: {match_text}"
            )
        xml_operand = matches[0]
        if xml_operand.order in covered_orders:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats "
                f"implicit XML operand {_format_implicit_xml_operand(xml_operand)}"
            )
        covered_orders.add(xml_operand.order)
        _validate_implicit_operand_decision(overlay, implicit_overlay, xml_operand)

    missing_operands = tuple(
        xml_operand
        for xml_operand in implicit_xml_operands
        if xml_operand.order not in covered_orders
    )
    if missing_operands:
        missing_text = ", ".join(
            _format_implicit_xml_operand(xml_operand)
            for xml_operand in missing_operands
        )
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' does not cover "
            f"implicit XML operand(s): {missing_text}"
        )


def _implicit_overlay_matches_xml_operand(
    implicit_overlay: AmdgpuImplicitOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
) -> bool:
    if implicit_overlay.operand_type != xml_operand.operand_type:
        return False
    if (
        implicit_overlay.data_format_name is not None
        and implicit_overlay.data_format_name != xml_operand.data_format_name
    ):
        return False
    if (
        implicit_overlay.size_bits is not None
        and implicit_overlay.size_bits != xml_operand.size_bits
    ):
        return False
    if (
        implicit_overlay.is_input is not None
        and implicit_overlay.is_input != xml_operand.is_input
    ):
        return False
    if (
        implicit_overlay.is_output is not None
        and implicit_overlay.is_output != xml_operand.is_output
    ):
        return False
    return True


def _validate_implicit_operand_decision(
    overlay: AmdgpuDescriptorOverlay,
    implicit_overlay: AmdgpuImplicitOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
) -> None:
    if implicit_overlay.descriptor_operand is None:
        if (
            not implicit_overlay.ignore_reason
            or not implicit_overlay.ignore_reason.strip()
        ):
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' ignores "
                f"implicit XML operand {_format_implicit_xml_operand(xml_operand)} "
                "without a named reason"
            )
        return

    if implicit_overlay.ignore_reason is not None:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps implicit XML "
            f"operand {_format_implicit_xml_operand(xml_operand)} and also "
            "provides an ignore reason"
        )
    descriptor_operand = implicit_overlay.descriptor_operand
    mapped_low_roles = (
        OperandRole.RESULT,
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
        OperandRole.IMPLICIT,
    )
    if descriptor_operand.role not in mapped_low_roles:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps implicit XML "
            f"operand {_format_implicit_xml_operand(xml_operand)} to low operand "
            f"'{descriptor_operand.field_name}' with invalid low role"
        )
    if descriptor_operand.role is OperandRole.RESULT and not xml_operand.is_output:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps input-only "
            f"implicit XML operand {_format_implicit_xml_operand(xml_operand)} "
            f"to result '{descriptor_operand.field_name}'"
        )
    if (
        descriptor_operand.role
        in (
            OperandRole.OPERAND,
            OperandRole.PREDICATE,
            OperandRole.RESOURCE,
        )
        and not xml_operand.is_input
    ):
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps output-only "
            f"implicit XML operand {_format_implicit_xml_operand(xml_operand)} "
            f"to packet operand '{descriptor_operand.field_name}'"
        )
    if OperandFlag.IMPLICIT not in descriptor_operand.flags:
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps implicit XML "
            f"operand {_format_implicit_xml_operand(xml_operand)} to low operand "
            f"'{descriptor_operand.field_name}' without implicit flag"
        )


def _format_implicit_overlay_key(
    implicit_overlay: AmdgpuImplicitOperandOverlay,
) -> str:
    parts = [f"type={implicit_overlay.operand_type}"]
    if implicit_overlay.data_format_name is not None:
        parts.append(f"format={implicit_overlay.data_format_name}")
    if implicit_overlay.size_bits is not None:
        parts.append(f"bits={implicit_overlay.size_bits}")
    if implicit_overlay.is_input is not None:
        parts.append(f"input={implicit_overlay.is_input}")
    if implicit_overlay.is_output is not None:
        parts.append(f"output={implicit_overlay.is_output}")
    return ",".join(parts)


def _format_implicit_xml_operand(xml_operand: AmdgpuIsaOperand) -> str:
    format_name = (
        "<none>"
        if xml_operand.data_format_name is None
        else xml_operand.data_format_name
    )
    return (
        f"order={xml_operand.order},type={xml_operand.operand_type},"
        f"format={format_name},input={xml_operand.is_input},"
        f"output={xml_operand.is_output}"
    )


def _validate_unique_overlay_fields(overlay: AmdgpuDescriptorOverlay) -> None:
    covered_fields: set[str] = set()
    operand_fields: dict[str, list[int]] = {}
    for operand_index, operand_overlay in enumerate(overlay.operands):
        operand_fields.setdefault(operand_overlay.xml_field_name, []).append(
            operand_index
        )
        covered_fields.add(operand_overlay.xml_field_name)
    for field_name, operand_indices in operand_fields.items():
        if len(operand_indices) == 1:
            continue
        if not _is_tied_repeated_operand_field(overlay, operand_indices):
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{field_name}' without a tied result/input pair"
            )
    for ignored_operand in overlay.ignored_operands:
        if ignored_operand.xml_field_name in covered_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{ignored_operand.xml_field_name}' across operands and ignored "
                "operands"
            )
        covered_fields.add(ignored_operand.xml_field_name)
    for immediate_field in overlay.immediate_fields:
        if immediate_field in covered_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{immediate_field}' across operands and immediates"
            )
        covered_fields.add(immediate_field)
    for fixed_field, _value in overlay.fixed_encoding_fields:
        if fixed_field in covered_fields:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' repeats XML field "
                f"'{fixed_field}' across variable and fixed encoding fields"
            )
        covered_fields.add(fixed_field)


def _explicit_xml_operands_by_field_name(
    encoding: AmdgpuIsaInstructionEncoding,
    overlay: AmdgpuDescriptorOverlay,
) -> dict[str, AmdgpuIsaOperand]:
    operands: dict[str, AmdgpuIsaOperand] = {}
    for xml_operand in encoding.operands:
        if xml_operand.is_implicit:
            continue
        if xml_operand.field_name is None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' found explicit "
                "XML operand without a field name"
            )
        if xml_operand.field_name in operands:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' found duplicate "
                f"XML operand field '{xml_operand.field_name}'"
            )
        operands[xml_operand.field_name] = xml_operand
    return operands


def _operand_role_is_packet_input(role: OperandRole) -> bool:
    return role in (
        OperandRole.OPERAND,
        OperandRole.PREDICATE,
        OperandRole.RESOURCE,
    )


def _has_tied_constraint(
    overlay: AmdgpuDescriptorOverlay, result_index: int, operand_index: int
) -> bool:
    return any(
        constraint.kind is ConstraintKind.TIED
        and constraint.lhs_operand_index == result_index
        and constraint.rhs_operand_index == operand_index
        for constraint in overlay.constraints
    )


def _is_tied_repeated_operand_field(
    overlay: AmdgpuDescriptorOverlay, operand_indices: list[int]
) -> bool:
    if len(operand_indices) != 2:
        return False
    lhs_index, rhs_index = operand_indices
    lhs = overlay.operands[lhs_index].descriptor_operand
    rhs = overlay.operands[rhs_index].descriptor_operand
    if lhs.role is OperandRole.RESULT and _operand_role_is_packet_input(rhs.role):
        return _has_tied_constraint(overlay, lhs_index, rhs_index)
    if rhs.role is OperandRole.RESULT and _operand_role_is_packet_input(lhs.role):
        return _has_tied_constraint(overlay, rhs_index, lhs_index)
    return False


def _validate_operand_role(
    overlay: AmdgpuDescriptorOverlay,
    operand_overlay: AmdgpuOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
) -> None:
    descriptor_operand = operand_overlay.descriptor_operand
    match descriptor_operand.role:
        case OperandRole.RESULT:
            if not xml_operand.is_output or xml_operand.is_input:
                _raise_role_mismatch(overlay, operand_overlay, xml_operand, "output")
            _reject_unnecessary_role_exception(overlay, operand_overlay)
        case OperandRole.OPERAND | OperandRole.RESOURCE | OperandRole.PREDICATE:
            if not xml_operand.is_input or xml_operand.is_output:
                if _operand_role_mismatch_is_explicitly_allowed(
                    operand_overlay, xml_operand
                ):
                    return
                _raise_role_mismatch(overlay, operand_overlay, xml_operand, "input")
            _reject_unnecessary_role_exception(overlay, operand_overlay)
        case OperandRole.IMPLICIT:
            if not xml_operand.is_implicit:
                _raise_role_mismatch(overlay, operand_overlay, xml_operand, "implicit")
            _reject_unnecessary_role_exception(overlay, operand_overlay)
        case OperandRole.OPERAND_RESULT:
            if not xml_operand.is_input or not xml_operand.is_output:
                _raise_role_mismatch(
                    overlay, operand_overlay, xml_operand, "input/output"
                )
            _reject_unnecessary_role_exception(overlay, operand_overlay)


def _validate_operand_width(
    overlay: AmdgpuDescriptorOverlay,
    operand_overlay: AmdgpuOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
) -> None:
    size_exception_reason = operand_overlay.size_exception_reason
    low_width_bits = _operand_low_width_bits(operand_overlay.descriptor_operand)
    if low_width_bits is None:
        if size_exception_reason is not None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' maps XML field "
                f"'{operand_overlay.xml_field_name}' to low operand "
                f"'{operand_overlay.descriptor_operand.field_name}' with a size "
                "exception, but the low operand width is unknown"
            )
        return
    if low_width_bits == xml_operand.size_bits:
        if size_exception_reason is not None:
            raise AmdgpuDescriptorOverlayError(
                f"descriptor overlay '{overlay.descriptor_key}' maps XML field "
                f"'{operand_overlay.xml_field_name}' to low operand "
                f"'{operand_overlay.descriptor_operand.field_name}' with an "
                "unnecessary size exception"
            )
        return
    if size_exception_reason is not None and size_exception_reason.strip():
        return
    raise AmdgpuDescriptorOverlayError(
        f"descriptor overlay '{overlay.descriptor_key}' maps XML field "
        f"'{operand_overlay.xml_field_name}' to low operand "
        f"'{operand_overlay.descriptor_operand.field_name}' with "
        f"{low_width_bits}-bit low width, but XML operand size is "
        f"{xml_operand.size_bits} bits"
    )


def _operand_low_width_bits(operand: Operand) -> int | None:
    if operand.register_part is not None:
        register_part_width = _REGISTER_PART_WIDTH_BITS.get(operand.register_part)
        if register_part_width is None:
            return None
        return register_part_width * operand.unit_count

    alt_widths = {
        _REGISTER_WIDTH_BITS[reg_alt.reg_class]
        for reg_alt in operand.reg_alts
        if reg_alt.reg_class in _REGISTER_WIDTH_BITS
    }
    if len(alt_widths) != 1:
        return None
    return next(iter(alt_widths)) * operand.unit_count


def _operand_role_mismatch_is_explicitly_allowed(
    operand_overlay: AmdgpuOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
) -> bool:
    reason = operand_overlay.role_exception_reason
    return bool(
        reason
        and reason.strip()
        and xml_operand.is_output
        and not xml_operand.is_input
        and not xml_operand.is_implicit
    )


def _reject_unnecessary_role_exception(
    overlay: AmdgpuDescriptorOverlay,
    operand_overlay: AmdgpuOperandOverlay,
) -> None:
    reason = operand_overlay.role_exception_reason
    if reason is None:
        return
    if reason.strip():
        raise AmdgpuDescriptorOverlayError(
            f"descriptor overlay '{overlay.descriptor_key}' maps XML field "
            f"'{operand_overlay.xml_field_name}' to low operand "
            f"'{operand_overlay.descriptor_operand.field_name}' with an "
            "unnecessary role exception"
        )
    raise AmdgpuDescriptorOverlayError(
        f"descriptor overlay '{overlay.descriptor_key}' maps XML field "
        f"'{operand_overlay.xml_field_name}' to low operand "
        f"'{operand_overlay.descriptor_operand.field_name}' with an empty role "
        "exception reason"
    )


def _raise_role_mismatch(
    overlay: AmdgpuDescriptorOverlay,
    operand_overlay: AmdgpuOperandOverlay,
    xml_operand: AmdgpuIsaOperand,
    expected_role: str,
) -> None:
    raise AmdgpuDescriptorOverlayError(
        f"descriptor overlay '{overlay.descriptor_key}' maps XML field "
        f"'{operand_overlay.xml_field_name}' to low operand "
        f"'{operand_overlay.descriptor_operand.field_name}' as {expected_role}, "
        f"but XML has input={xml_operand.is_input} output={xml_operand.is_output} "
        f"implicit={xml_operand.is_implicit}"
    )
