# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Derives canonical physical instruction text from the VM ISA."""

from __future__ import annotations

import re

from iree.vm.bytecode.spec import isa
from iree.vm.bytecode.spec.isa.core import rules
from iree.vm.bytecode.spec.schema import (
    NumericKind,
    NumericTable,
    NumericValue,
    is_name,
    require,
)
from iree.vm.bytecode.spec.specification import Specification
from iree.vm.bytecode.spec.text import (
    DirectTargetKind,
    InstructionFieldKind,
    InstructionPosition,
    InstructionText,
    InstructionTextField,
    InstructionTextFormat,
    LaneFormat,
    MnemonicProjection,
    SymbolDomain,
)

_ENCODING_SUFFIX = re.compile(r"_(?:[uifs]?\d+|[vrf]\d+)(?:_nullable)?$")

_REGISTER_KINDS = {
    rules.FieldRule.REGISTER_VALUE: InstructionFieldKind.VALUE_REGISTER,
    rules.FieldRule.REGISTER_REF: InstructionFieldKind.REF_REGISTER,
    rules.FieldRule.REGISTER_FUNCTION: InstructionFieldKind.FUNCTION_REGISTER,
}

_BLOCK_TARGET_KINDS = {
    rules.FieldRule.CONTROL_TARGET_S16,
    rules.FieldRule.CONTROL_TARGET_S32,
}

_SYMBOL_DOMAINS = {
    rules.FieldRule.CONSTANT_POOL_ORDINAL: SymbolDomain.CONSTANT,
    rules.FieldRule.IMPORT_ORDINAL_OPTIONAL: SymbolDomain.IMPORT,
    rules.FieldRule.RODATA_ORDINAL: SymbolDomain.RODATA,
}

_PLAIN_ORDINAL_KINDS = {
    rules.FieldRule.ABI_SLOT,
    rules.FieldRule.FUNCTION_LOCAL_ORDINAL,
}

_DIRECT_TARGET_DOMAINS = {
    "local": SymbolDomain.FUNCTION,
    "required_import": SymbolDomain.IMPORT,
    "optional_import": SymbolDomain.IMPORT,
}


def _display_name(name: str) -> str:
    return _ENCODING_SUFFIX.sub("", name)


def _global_domain(mnemonic: str) -> SymbolDomain:
    if mnemonic.startswith("global.value."):
        return SymbolDomain.GLOBAL_VALUE
    if mnemonic.startswith("global.ref."):
        return SymbolDomain.GLOBAL_REF
    if mnemonic.startswith("global.func."):
        return SymbolDomain.GLOBAL_FUNCTION
    raise ValueError(f"{mnemonic}: global ordinal has no symbol domain")


def _numeric_kind(field: isa.InstructionField) -> InstructionFieldKind:
    if field.field.encoding.name.startswith("i"):
        return InstructionFieldKind.SIGNED
    display_name = _display_name(field.field.name)
    if display_name == "bits" or "mask" in display_name:
        return InstructionFieldKind.HEX
    return InstructionFieldKind.UNSIGNED


def _is_positional_immediate(
    field: isa.InstructionField, kind: InstructionFieldKind
) -> bool:
    if kind in {
        InstructionFieldKind.BLOCK_TARGET,
        InstructionFieldKind.ORDINAL,
        InstructionFieldKind.LOCAL_BYTES,
        InstructionFieldKind.LOCAL_BYTES_RANGE,
        InstructionFieldKind.LOCAL_BYTES_REPEATED_RANGE,
        InstructionFieldKind.LOCAL_BYTES_FIXED_RANGE,
        InstructionFieldKind.REF_SLOT,
        InstructionFieldKind.MODULE_SYMBOL,
        InstructionFieldKind.RODATA_RANGE,
        InstructionFieldKind.DIRECT_TARGET,
        InstructionFieldKind.SWITCH_SLICE,
        InstructionFieldKind.COMBINED_HEX,
    }:
        return True
    if field.field.element_count > 1:
        return True
    if _display_name(field.field.name) in {"bits", "immediate"}:
        return True
    return (
        kind == InstructionFieldKind.SELECTOR
        and field.rule.data.name == "control.status"
    )


def _position(
    field: isa.InstructionField, kind: InstructionFieldKind
) -> InstructionPosition:
    if field.role == isa.FieldRole.RESULT:
        return InstructionPosition.RESULT
    if field.role == isa.FieldRole.OPERAND or _is_positional_immediate(field, kind):
        return InstructionPosition.OPERAND
    return InstructionPosition.ATTRIBUTE


def _lane_formats(table: NumericTable) -> tuple[LaneFormat, ...]:
    require(table.kind == NumericKind.SELECTOR, "memory format must be a selector")
    formats = []
    for value in table.values:
        element_name, separator, lane_text = value.name.partition(".x")
        if not separator or not lane_text.isdecimal():
            raise ValueError(f"{table.name}.{value.name}: invalid lane format name")
        lane_count = int(lane_text)
        if lane_count not in (1, 2, 4, 8):
            raise ValueError(f"{table.name}.{value.name}: invalid lane count")
        formats.append(LaneFormat(value.value, element_name, lane_count))
    if len({(item.element_name, item.lane_count) for item in formats}) != len(formats):
        raise ValueError(f"{table.name}: duplicate lane text form")
    return tuple(formats)


def _numeric_value(
    specification: Specification, table_name: str, value_name: str
) -> int:
    tables = specification.module_format.numeric_tables + specification.selectors
    table = next(table for table in tables if table.name == table_name)
    return next(value.value for value in table.values if value.name == value_name)


def _direct_target_kind(
    specification: Specification, encoded_value: NumericValue
) -> DirectTargetKind:
    domain = _DIRECT_TARGET_DOMAINS[encoded_value.name]
    if domain == SymbolDomain.FUNCTION:
        return DirectTargetKind(encoded_value.value, domain)
    optional_flag = _numeric_value(specification, "import_flag", "optional")
    return DirectTargetKind(
        encoded_value.value,
        domain,
        optional_flag,
        optional_flag if encoded_value.name == "optional_import" else 0,
    )


def _instruction_text(
    specification: Specification, instruction: isa.Instruction
) -> InstructionText:
    fields = {item.field.name: item for item in instruction.fields}
    field_ordinals = {
        item.field.name: ordinal for ordinal, item in enumerate(instruction.fields)
    }
    record_rules = {rule.kind: rule for rule in instruction.rules}

    direct_target_fields = None
    direct_target_kinds = ()
    if rules.RecordRuleKind.CALL in record_rules:
        direct_target_fields = record_rules[rules.RecordRuleKind.CALL].fields[:2]
    elif rules.RecordRuleKind.FUNCTION_ADDRESS in record_rules:
        direct_target_fields = record_rules[
            rules.RecordRuleKind.FUNCTION_ADDRESS
        ].fields[:2]
    if direct_target_fields:
        selector = fields[direct_target_fields[0]].rule.data
        direct_target_kinds = tuple(
            _direct_target_kind(specification, value) for value in selector.values
        )
    switch_fields = (
        record_rules[rules.RecordRuleKind.SWITCH_TARGETS].fields
        if rules.RecordRuleKind.SWITCH_TARGETS in record_rules
        else None
    )
    native_call_field = (
        record_rules[rules.RecordRuleKind.NATIVE_CALL].fields[0]
        if rules.RecordRuleKind.NATIVE_CALL in record_rules
        else None
    )

    value_ranges = {}
    for record_rule in instruction.rules:
        if record_rule.kind in {
            rules.RecordRuleKind.VALUE_REGISTER_RANGE,
            rules.RecordRuleKind.VALUE_REGISTER_FORMAT_RANGE,
        }:
            value_ranges[record_rule.fields[0]] = record_rule.fields[1]

    mnemonic_projection = None
    format_field_name = None
    for item in instruction.fields:
        if (
            item.rule.kind == rules.FieldRule.SELECTOR
            and item.rule.data.name == "memory.format"
        ):
            format_field_name = item.field.name
            mnemonic_projection = MnemonicProjection(
                format_field_name, _lane_formats(item.rule.data)
            )
            break

    rodata_offsets = {
        item.rule.fields[0]: item.field.name
        for item in instruction.fields
        if item.rule.kind
        in {rules.FieldRule.RODATA_OFFSET, rules.FieldRule.RODATA_STATIC_OFFSET}
    }

    text_fields = []
    skipped_fields = set()
    for field in instruction.fields:
        name = field.field.name
        if field.role == isa.FieldRole.PADDING or name in skipped_fields:
            continue

        kind = None
        related_field = None
        numeric_table = None
        symbol_domain = None
        bit_offset = 0
        bit_length = 0
        fixed_value = 0
        display_name = _display_name(name)

        if direct_target_fields and name == direct_target_fields[0]:
            kind = InstructionFieldKind.DIRECT_TARGET
            related_field = direct_target_fields[1]
            symbol_domain = SymbolDomain.DIRECT_TARGET
            display_name = "target"
            skipped_fields.add(related_field)
        elif switch_fields and name == switch_fields[0]:
            kind = InstructionFieldKind.SWITCH_SLICE
            related_field = switch_fields[1]
            symbol_domain = SymbolDomain.SWITCH_TARGET
            display_name = "targets"
            skipped_fields.add(related_field)
        elif name == "bits_low_u32" and "bits_high_u32" in fields:
            kind = InstructionFieldKind.COMBINED_HEX
            related_field = "bits_high_u32"
            display_name = "bits"
            skipped_fields.add(related_field)
        elif field.rule.kind == rules.FieldRule.PACKED_SELECTORS:
            for component in field.rule.data:
                text_fields.append(
                    InstructionTextField(
                        component.name,
                        InstructionPosition.ATTRIBUTE,
                        InstructionFieldKind.PACKED_SELECTOR,
                        name,
                        numeric_table=component.table,
                        bit_offset=component.bit_offset,
                        bit_length=component.bit_length,
                    )
                )
            continue
        elif name in value_ranges:
            related_field = value_ranges[name]
            kind = (
                InstructionFieldKind.VALUE_REGISTER_FORMAT_RANGE
                if related_field == format_field_name
                else InstructionFieldKind.VALUE_REGISTER_RANGE
            )
            skipped_fields.add(related_field)
        elif field.rule.kind in _REGISTER_KINDS:
            kind = _REGISTER_KINDS[field.rule.kind]
        elif field.rule.kind in _BLOCK_TARGET_KINDS:
            kind = InstructionFieldKind.BLOCK_TARGET
            symbol_domain = SymbolDomain.BLOCK
            display_name = "target"
        elif field.rule.kind == rules.FieldRule.LOCAL_BYTES_RANGE_BASE:
            kind = InstructionFieldKind.LOCAL_BYTES_RANGE
            related_field = field.rule.fields[0]
            display_name = display_name.removesuffix("_base")
            skipped_fields.add(related_field)
        elif field.rule.kind == rules.FieldRule.LOCAL_BYTES_REPEATED_BASE:
            kind = InstructionFieldKind.LOCAL_BYTES_REPEATED_RANGE
            related_field = field.rule.fields[0]
            fixed_value = field.rule.values[0]
            skipped_fields.add(related_field)
        elif field.rule.kind == rules.FieldRule.LOCAL_BYTES_FIXED_BASE:
            kind = InstructionFieldKind.LOCAL_BYTES_FIXED_RANGE
            fixed_value = field.rule.values[0]
        elif field.rule.kind == rules.FieldRule.LOCAL_BYTES_RANGE_MEMORY_FORMAT:
            kind = InstructionFieldKind.LOCAL_BYTES
            related_field = field.rule.fields[0]
        elif field.rule.kind == rules.FieldRule.REF_SLOT:
            kind = InstructionFieldKind.REF_SLOT
        elif field.rule.kind in _SYMBOL_DOMAINS:
            kind = InstructionFieldKind.MODULE_SYMBOL
            symbol_domain = _SYMBOL_DOMAINS[field.rule.kind]
            if name in rodata_offsets:
                kind = InstructionFieldKind.RODATA_RANGE
                related_field = rodata_offsets[name]
                skipped_fields.add(related_field)
        elif field.rule.kind == rules.FieldRule.GLOBAL_ORDINAL:
            kind = InstructionFieldKind.MODULE_SYMBOL
            symbol_domain = _global_domain(instruction.mnemonic)
        elif name == native_call_field:
            kind = InstructionFieldKind.MODULE_SYMBOL
            symbol_domain = SymbolDomain.NATIVE_CALLSITE
        elif field.rule.kind in _PLAIN_ORDINAL_KINDS:
            kind = InstructionFieldKind.ORDINAL
        elif field.rule.kind == rules.FieldRule.SELECTOR:
            if name == format_field_name:
                skipped_fields.add(name)
                continue
            kind = InstructionFieldKind.SELECTOR
            numeric_table = field.rule.data
        elif (
            field.rule.kind == rules.FieldRule.CONSTRAINT_MEMBER
            and name == "callable_type_ordinal_u16"
        ):
            kind = InstructionFieldKind.MODULE_SYMBOL
            symbol_domain = SymbolDomain.CALLABLE
            display_name = "type"
        else:
            kind = _numeric_kind(field)

        text_fields.append(
            InstructionTextField(
                display_name,
                _position(field, kind),
                kind,
                name,
                related_field,
                numeric_table,
                symbol_domain,
                bit_offset,
                bit_length,
                fixed_value,
            )
        )

    text_fields.sort(
        key=lambda item: (
            item.position,
            field_ordinals[item.primary_field],
        )
    )
    return InstructionText(
        instruction,
        tuple(text_fields),
        mnemonic_projection,
        direct_target_kinds,
    )


def _validate_instruction_text(text: InstructionText) -> None:
    instruction = text.instruction
    fields = {item.field.name: item for item in instruction.fields}
    padding_fields = {
        item.field.name
        for item in instruction.fields
        if item.role == isa.FieldRole.PADDING
    }
    primary_fields = []
    related_fields = []
    for item in text.fields:
        require(is_name(item.name), f"{instruction.mnemonic}: invalid text field name")
        require(
            item.primary_field in fields and item.primary_field not in padding_fields,
            f"{instruction.mnemonic}: invalid primary text field",
        )
        primary_fields.append(item.primary_field)
        if item.related_field:
            require(
                item.related_field in fields
                and item.related_field not in padding_fields,
                f"{instruction.mnemonic}: invalid related text field",
            )
            related_fields.append(item.related_field)
        if item.kind in {
            InstructionFieldKind.SELECTOR,
            InstructionFieldKind.PACKED_SELECTOR,
        }:
            require(
                item.numeric_table is not None,
                f"{instruction.mnemonic}: selector has no numeric table",
            )
        else:
            require(
                item.numeric_table is None,
                f"{instruction.mnemonic}: unexpected numeric table",
            )
        if item.kind in {
            InstructionFieldKind.MODULE_SYMBOL,
            InstructionFieldKind.RODATA_RANGE,
            InstructionFieldKind.DIRECT_TARGET,
            InstructionFieldKind.SWITCH_SLICE,
            InstructionFieldKind.BLOCK_TARGET,
        }:
            require(
                item.symbol_domain is not None,
                f"{instruction.mnemonic}: symbolic field has no domain",
            )
        else:
            require(
                item.symbol_domain is None,
                f"{instruction.mnemonic}: unexpected symbol domain",
            )

    duplicate_primary = {
        name for name in primary_fields if primary_fields.count(name) > 1
    }
    packed_primary = {
        item.primary_field
        for item in text.fields
        if item.kind == InstructionFieldKind.PACKED_SELECTOR
    }
    require(
        duplicate_primary <= packed_primary,
        f"{instruction.mnemonic}: physical field has multiple text owners",
    )
    projected_fields = set()
    format_ranges = tuple(
        item
        for item in text.fields
        if item.kind == InstructionFieldKind.VALUE_REGISTER_FORMAT_RANGE
    )
    if text.mnemonic_projection:
        projection = text.mnemonic_projection
        require(
            projection.field_name in fields,
            f"{instruction.mnemonic}: invalid mnemonic projection field",
        )
        projected_fields.add(projection.field_name)
        encoded_values = {item.encoded_value for item in projection.formats}
        table = fields[projection.field_name].rule.data
        require(
            encoded_values == {item.value for item in table.values},
            f"{instruction.mnemonic}: mnemonic projection is incomplete",
        )
        require(
            len(format_ranges) == 1
            and format_ranges[0].related_field == projection.field_name,
            f"{instruction.mnemonic}: mnemonic projection has no unique lane range",
        )
    else:
        require(
            not format_ranges,
            f"{instruction.mnemonic}: lane range has no mnemonic projection",
        )

    represented_fields = set(primary_fields) | set(related_fields) | projected_fields
    expected_fields = set(fields) - padding_fields
    require(
        represented_fields == expected_fields,
        f"{instruction.mnemonic}: text does not cover every physical field",
    )
    require(
        not (set(primary_fields) & set(related_fields)),
        f"{instruction.mnemonic}: related field also has a text owner",
    )

    positions = tuple(item.position for item in text.fields)
    require(
        positions == tuple(sorted(positions)),
        f"{instruction.mnemonic}: text fields are not canonically ordered",
    )
    direct_target_fields = tuple(
        item for item in text.fields if item.kind == InstructionFieldKind.DIRECT_TARGET
    )
    require(
        bool(direct_target_fields) == bool(text.direct_target_kinds),
        f"{instruction.mnemonic}: inconsistent direct-target presentation",
    )
    if direct_target_fields:
        selector = fields[direct_target_fields[0].primary_field].rule.data
        require(
            {item.encoded_value for item in text.direct_target_kinds}
            == {item.value for item in selector.values},
            f"{instruction.mnemonic}: direct-target presentation is incomplete",
        )
        for target in text.direct_target_kinds:
            require(
                0 <= target.symbol_flags_mask <= 0xFFFF
                and target.symbol_flags_value & ~target.symbol_flags_mask == 0,
                f"{instruction.mnemonic}: invalid direct-target flag match",
            )
        for ordinal, lhs in enumerate(text.direct_target_kinds):
            for rhs in text.direct_target_kinds[ordinal + 1 :]:
                if lhs.symbol_domain != rhs.symbol_domain:
                    continue
                common_mask = lhs.symbol_flags_mask & rhs.symbol_flags_mask
                require(
                    (lhs.symbol_flags_value & common_mask)
                    != (rhs.symbol_flags_value & common_mask),
                    f"{instruction.mnemonic}: ambiguous direct-target flag match",
                )


def build_instruction_text_format(
    specification: Specification,
) -> InstructionTextFormat:
    """Builds and validates one complete instruction text projection."""

    instructions = tuple(
        _instruction_text(specification, instruction)
        for instruction in specification.instructions
    )
    for text in instructions:
        _validate_instruction_text(text)
    require(
        len({item.instruction.mnemonic for item in instructions}) == len(instructions),
        "duplicate textual instruction mnemonic",
    )
    return InstructionTextFormat(instructions)
