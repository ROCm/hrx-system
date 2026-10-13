# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders private lookup programs for the canonical VM assembler."""

from __future__ import annotations

import enum

from iree.vm.bytecode.spec import isa
from iree.vm.bytecode.spec.isa.core import rules as instruction_rules
from iree.vm.bytecode.spec.module import WireRecord
from iree.vm.bytecode.spec.module import rules as module_rules
from iree.vm.bytecode.spec.render.c_tooling import (
    BStringPool,
    generated_preamble,
    intern_row_group,
    render_bstring_pool,
    render_byte_array,
)
from iree.vm.bytecode.spec.schema import NumericKind, NumericTable
from iree.vm.bytecode.spec.specification import Specification
from iree.vm.bytecode.spec.text import (
    InstructionFieldKind,
    InstructionTextFormat,
    SymbolDomain,
)
from iree.vm.bytecode.spec.text.module import (
    AlignmentLayout,
    ConstantField,
    DeclareSymbol,
    FixedLayoutCount,
    FixedRunCount,
    FunctionBody,
    GrammarCountDerivation,
    HexBlob,
    MetadataValue,
    ModuleTextFormat,
    NativeCallBody,
    NativeLayoutBody,
    NestedBody,
    NullableSymbolReference,
    NumericField,
    QuotedBlob,
    RecordCountDerivation,
    RecordGrammar,
    RecordLayout,
    RowMode,
    ScalarField,
    ScalarRadix,
    SectionDirectoryFieldDerivation,
    SignatureBody,
    StatisticDerivation,
    SymbolReference,
    TailLayout,
)


class _FormatOpcode(enum.IntEnum):
    END = 0x00
    LITERAL = 0xE0
    DECLARE = 0xE1
    SYMBOL = 0xE2
    NULLABLE_SYMBOL = 0xE3
    SCALAR = 0xE4
    NUMERIC = 0xE5
    CONSTANT = 0xE6
    BODY = 0xE7
    QUOTED_BLOB = 0xE8
    SIGNATURE = 0xE9
    FUNCTION_BODY = 0xEA
    HEX_BLOB = 0xEB
    METADATA_VALUE = 0xEC
    DECLARE_FLAGS = 0xED
    NATIVE_LAYOUT = 0xEE
    NATIVE_CALL = 0xEF


class _LayoutKind(enum.IntEnum):
    RECORD = 0
    ALIGN = 1
    TAIL = 2


class _DerivationKind(enum.IntEnum):
    RECORD_COUNT = 0
    GRAMMAR_COUNT = 1
    STATISTIC = 2


class _InstructionRelationKind(enum.IntEnum):
    NONE = 0
    ABI_SLOT = 1
    RODATA_OFFSET = 2
    RODATA_STATIC_OFFSET = 3
    FIELDS_DISTINCT = 4
    INTEGER_BITSTREAM_SHAPE = 5
    PACKED_SELECTOR_PAIRS = 6
    CALL = 7
    CALL_INDIRECT = 8
    FUNCTION_ADDRESS = 9
    NATIVE_CALL = 10


_NO_FIELD = 0xFF


_INSTRUCTION_FIELD_RULES = {
    InstructionFieldKind.VALUE_REGISTER: (instruction_rules.FieldRule.REGISTER_VALUE,),
    InstructionFieldKind.VALUE_REGISTER_RANGE: (
        instruction_rules.FieldRule.REGISTER_VALUE,
    ),
    InstructionFieldKind.VALUE_REGISTER_FORMAT_RANGE: (
        instruction_rules.FieldRule.REGISTER_VALUE,
    ),
    InstructionFieldKind.REF_REGISTER: (instruction_rules.FieldRule.REGISTER_REF,),
    InstructionFieldKind.FUNCTION_REGISTER: (
        instruction_rules.FieldRule.REGISTER_FUNCTION,
    ),
    InstructionFieldKind.SIGNED: (instruction_rules.FieldRule.ANY_BITS,),
    InstructionFieldKind.UNSIGNED: (
        instruction_rules.FieldRule.ANY_BITS,
        instruction_rules.FieldRule.ALLOWED_RANGE,
        instruction_rules.FieldRule.ALLOWED_VALUES,
    ),
    InstructionFieldKind.HEX: (
        instruction_rules.FieldRule.ANY_BITS,
        instruction_rules.FieldRule.CONSTRAINT_MEMBER,
    ),
    InstructionFieldKind.COMBINED_HEX: (instruction_rules.FieldRule.ANY_BITS,),
    InstructionFieldKind.SELECTOR: (instruction_rules.FieldRule.SELECTOR,),
    InstructionFieldKind.PACKED_SELECTOR: (
        instruction_rules.FieldRule.PACKED_SELECTORS,
    ),
    InstructionFieldKind.BLOCK_TARGET: instruction_rules.DIRECT_TARGET_RULES,
    InstructionFieldKind.ORDINAL: (
        instruction_rules.FieldRule.ABI_SLOT,
        instruction_rules.FieldRule.FUNCTION_LOCAL_ORDINAL,
    ),
    InstructionFieldKind.LOCAL_BYTES: (
        instruction_rules.FieldRule.LOCAL_BYTES_RANGE_MEMORY_FORMAT,
    ),
    InstructionFieldKind.LOCAL_BYTES_RANGE: (
        instruction_rules.FieldRule.LOCAL_BYTES_RANGE_BASE,
    ),
    InstructionFieldKind.LOCAL_BYTES_REPEATED_RANGE: (
        instruction_rules.FieldRule.LOCAL_BYTES_REPEATED_BASE,
    ),
    InstructionFieldKind.LOCAL_BYTES_FIXED_RANGE: (
        instruction_rules.FieldRule.LOCAL_BYTES_FIXED_BASE,
    ),
    InstructionFieldKind.REF_SLOT: (instruction_rules.FieldRule.REF_SLOT,),
    InstructionFieldKind.MODULE_SYMBOL: (
        instruction_rules.FieldRule.CONSTANT_POOL_ORDINAL,
        instruction_rules.FieldRule.CONSTRAINT_MEMBER,
        instruction_rules.FieldRule.GLOBAL_ORDINAL,
        instruction_rules.FieldRule.IMPORT_ORDINAL_OPTIONAL,
        instruction_rules.FieldRule.RODATA_ORDINAL,
    ),
    InstructionFieldKind.RODATA_RANGE: (instruction_rules.FieldRule.RODATA_ORDINAL,),
    InstructionFieldKind.DIRECT_TARGET: (instruction_rules.FieldRule.SELECTOR,),
    InstructionFieldKind.SWITCH_SLICE: (instruction_rules.FieldRule.CONSTRAINT_MEMBER,),
}

_INSTRUCTION_RELATED_FIELD_RULES = {
    InstructionFieldKind.VALUE_REGISTER_RANGE: (
        instruction_rules.FieldRule.ALLOWED_RANGE,
    ),
    InstructionFieldKind.VALUE_REGISTER_FORMAT_RANGE: (
        instruction_rules.FieldRule.SELECTOR,
    ),
    InstructionFieldKind.COMBINED_HEX: (instruction_rules.FieldRule.ANY_BITS,),
    InstructionFieldKind.LOCAL_BYTES: (instruction_rules.FieldRule.SELECTOR,),
    InstructionFieldKind.LOCAL_BYTES_RANGE: (
        instruction_rules.FieldRule.LOCAL_BYTES_RANGE_LENGTH,
    ),
    InstructionFieldKind.LOCAL_BYTES_REPEATED_RANGE: (
        instruction_rules.FieldRule.LOCAL_BYTES_REPEATED_COUNT,
    ),
    InstructionFieldKind.RODATA_RANGE: (
        instruction_rules.FieldRule.RODATA_OFFSET,
        instruction_rules.FieldRule.RODATA_STATIC_OFFSET,
    ),
    InstructionFieldKind.DIRECT_TARGET: (
        instruction_rules.FieldRule.CONSTRAINT_MEMBER,
    ),
    InstructionFieldKind.SWITCH_SLICE: (instruction_rules.FieldRule.CONSTRAINT_MEMBER,),
}


class _ScalarConstraints:
    """Interns the two scalar predicates required by canonical assembly."""

    def __init__(self):
        self.rows: list[tuple[int, int, bool]] = []
        self.ordinals: dict[tuple[int, int, bool], int] = {}

    def intern(self, minimum: int, maximum: int, power_of_two: bool = False) -> int:
        row = (minimum, maximum, power_of_two)
        ordinal = self.ordinals.get(row)
        if ordinal is not None:
            return ordinal
        if not (0 <= minimum <= maximum <= 0xFFFFFFFF):
            raise ValueError(f"scalar constraint exceeds u32: {row}")
        ordinal = len(self.rows) + 1
        if ordinal > 32:
            raise ValueError(
                "assembler scalar constraints exceed the u32 predicate mask"
            )
        self.rows.append(row)
        self.ordinals[row] = ordinal
        return ordinal

    def for_rule(self, rule, field_width: int) -> int:
        if rule.kind in {
            instruction_rules.FieldRule.ANY_BITS,
            module_rules.FieldRule.ANY_BITS,
            module_rules.FieldRule.PAGE_MAJOR,
            module_rules.FieldRule.PAGE_REQUIRED_MINOR,
        }:
            return 0
        if rule.kind in {
            instruction_rules.FieldRule.ALLOWED_RANGE,
            module_rules.FieldRule.ALLOWED_RANGE,
        }:
            return self.intern(rule.values[0], rule.values[1])
        if rule.kind == instruction_rules.FieldRule.ALLOWED_VALUES:
            values = tuple(rule.values)
            if not values:
                raise ValueError("allowed values cannot be empty")
            expected_values = tuple(
                1 << bit
                for bit in range(values[0].bit_length() - 1, values[-1].bit_length())
            )
            if values != expected_values:
                raise ValueError(
                    f"allowed values are not a power-of-two range: {values}"
                )
            return self.intern(values[0], values[-1], True)
        if rule.kind == module_rules.FieldRule.BYTE_ALIGNMENT:
            maximum = (1 << (field_width * 8)) - 1
            return self.intern(rule.values[0], maximum, True)
        if rule.kind == module_rules.FieldRule.NONCORE_PAGE:
            return self.intern(0xF0, 0xFD)
        raise ValueError(f"unsupported scalar constraint {rule.kind.name}")

    def for_numeric_rule(self, rule, table: NumericTable) -> int:
        """Returns the runtime constraint not already proven by a numeric table."""

        if rule.kind == module_rules.FieldRule.ALLOWED_BITS:
            return 0
        return self.for_rule(rule, table.encoding.byte_length)


def _record_id(records: tuple[WireRecord, ...], record: WireRecord) -> int:
    return records.index(record)


def _field(record: WireRecord, name: str) -> tuple[int, int]:
    for wire_field, offset in zip(record.fields, record.field_offsets, strict=True):
        if wire_field.field.name == name:
            if offset > 0xFF or wire_field.field.byte_length > 0xFF:
                raise ValueError(f"{record.name}.{name}: field layout exceeds u8")
            return offset, wire_field.field.byte_length
    raise ValueError(f"{record.name}: unknown field {name}")


def _wire_field(record: WireRecord, name: str):
    return next(field for field in record.fields if field.field.name == name)


def _encoded_instruction_field(instruction: isa.Instruction, name: str) -> int:
    """Packs one instruction field offset and log2 byte width into u8."""

    offset, width = _field(instruction, name)
    if offset > 0x3F or width not in (1, 2, 4, 8):
        raise ValueError(f"{instruction.mnemonic}.{name}: relation field exceeds u8")
    return offset | ((width.bit_length() - 1) << 6)


def _validate_instruction_field_projection(text, field, physical_fields) -> None:
    """Rejects text fields whose semantic rules have no assembler owner."""

    primary_rule = physical_fields[field.primary_field].rule.kind
    native_symbol = (
        field.kind == InstructionFieldKind.MODULE_SYMBOL
        and field.symbol_domain == SymbolDomain.NATIVE_CALLSITE
        and any(
            rule.kind == instruction_rules.RecordRuleKind.NATIVE_CALL
            and rule.fields == (field.primary_field,)
            for rule in text.instruction.rules
        )
    )
    allowed_rules = (
        (instruction_rules.FieldRule.ANY_BITS,)
        if native_symbol
        else _INSTRUCTION_FIELD_RULES[field.kind]
    )
    if primary_rule not in allowed_rules:
        raise ValueError(
            f"{text.instruction.mnemonic}.{field.primary_field}: "
            f"{field.kind.name} does not own {primary_rule.name}"
        )
    if field.related_field:
        related_rule = physical_fields[field.related_field].rule.kind
        if related_rule not in _INSTRUCTION_RELATED_FIELD_RULES.get(field.kind, ()):
            raise ValueError(
                f"{text.instruction.mnemonic}.{field.related_field}: "
                f"{field.kind.name} does not own {related_rule.name}"
            )


def _instruction_relation(instruction: isa.Instruction):
    """Returns the one compact semantic relationship owned by an instruction."""

    relations = []
    for field in instruction.fields:
        rule = field.rule
        if rule.kind == instruction_rules.FieldRule.ABI_SLOT:
            relations.append(
                (_InstructionRelationKind.ABI_SLOT, (field.field.name,), rule.values[0])
            )
        elif rule.kind == instruction_rules.FieldRule.RODATA_OFFSET:
            relations.append(
                (
                    _InstructionRelationKind.RODATA_OFFSET,
                    (field.field.name, *rule.fields),
                    0,
                )
            )
        elif rule.kind == instruction_rules.FieldRule.RODATA_STATIC_OFFSET:
            relations.append(
                (
                    _InstructionRelationKind.RODATA_STATIC_OFFSET,
                    (field.field.name, *rule.fields),
                    0,
                )
            )
    for rule in instruction.rules:
        if rule.kind == instruction_rules.RecordRuleKind.FIELDS_DISTINCT:
            relations.append((_InstructionRelationKind.FIELDS_DISTINCT, rule.fields, 0))
        elif rule.kind == instruction_rules.RecordRuleKind.INTEGER_BITSTREAM_SHAPE:
            is_pack, maximum_bits = rule.values
            relations.append(
                (
                    _InstructionRelationKind.INTEGER_BITSTREAM_SHAPE,
                    rule.fields,
                    int(maximum_bits) << 1 | int(is_pack == 0),
                )
            )
        elif rule.kind == instruction_rules.RecordRuleKind.PACKED_SELECTOR_PAIRS:
            first, second = rule.data
            for component in (first, second):
                if not (0 < component.bit_length <= 0xF):
                    raise ValueError(
                        f"{instruction.mnemonic}: selector length exceeds u4"
                    )
                if not (0 <= component.bit_offset <= 0xF):
                    raise ValueError(
                        f"{instruction.mnemonic}: selector offset exceeds u4"
                    )
            pair_mask = 0
            for first_value, second_value in zip(
                rule.values[::2], rule.values[1::2], strict=True
            ):
                pair = first_value | second_value << first.bit_length
                if pair >= 48:
                    raise ValueError(
                        f"{instruction.mnemonic}: selector pair exceeds u48 mask"
                    )
                pair_mask |= 1 << pair
            component_data = (
                first.bit_offset
                | (first.bit_length << 4)
                | (second.bit_offset << 8)
                | (second.bit_length << 12)
            )
            relations.append(
                (
                    _InstructionRelationKind.PACKED_SELECTOR_PAIRS,
                    rule.fields,
                    pair_mask | (component_data << 48),
                )
            )
        elif rule.kind == instruction_rules.RecordRuleKind.CALL:
            relations.append((_InstructionRelationKind.CALL, rule.fields, 0))
        elif rule.kind == instruction_rules.RecordRuleKind.CALL_INDIRECT:
            relations.append((_InstructionRelationKind.CALL_INDIRECT, rule.fields, 0))
        elif rule.kind == instruction_rules.RecordRuleKind.FUNCTION_ADDRESS:
            relations.append(
                (_InstructionRelationKind.FUNCTION_ADDRESS, rule.fields, 0)
            )
        elif rule.kind == instruction_rules.RecordRuleKind.NATIVE_CALL:
            relations.append((_InstructionRelationKind.NATIVE_CALL, rule.fields, 0))
        elif rule.kind in {
            instruction_rules.RecordRuleKind.SWITCH_TARGETS,
            instruction_rules.RecordRuleKind.VALUE_REGISTER_RANGE,
            instruction_rules.RecordRuleKind.VALUE_REGISTER_FORMAT_RANGE,
        }:
            pass
        else:
            raise ValueError(
                f"{instruction.mnemonic}: unowned record rule {rule.kind.name}"
            )
    if len(relations) > 1:
        raise ValueError(f"{instruction.mnemonic}: multiple semantic relationships")
    if not relations:
        return None
    kind, fields, data = relations[0]
    if len(fields) > 5:
        raise ValueError(f"{instruction.mnemonic}: relationship exceeds five fields")
    encoded_fields = tuple(
        _encoded_instruction_field(instruction, name) for name in fields
    )
    return (data, *encoded_fields, *([0] * (5 - len(encoded_fields))), int(kind))


def _numeric_tables(
    specification: Specification,
    instruction_text: InstructionTextFormat,
    module_text: ModuleTextFormat,
    strings: BStringPool,
):
    tables: dict[str, NumericTable] = {}
    spellings: dict[tuple[str, str], str] = {}
    for instruction in instruction_text.instructions:
        for field in instruction.fields:
            if field.numeric_table:
                tables[field.numeric_table.name] = field.numeric_table
    for grammar in module_text.grammars:
        for element in grammar.elements:
            if isinstance(element, (SignatureBody, NativeLayoutBody)):
                tables[element.kind_table.name] = element.kind_table
            elif isinstance(element, NumericField):
                tables[element.table.name] = element.table
                for override in element.spelling_overrides:
                    spellings[(element.table.name, override.source_name)] = (
                        override.text
                    )

    table_order = tuple(
        table
        for table in specification.module_format.numeric_tables
        + specification.selectors
        if table.name in tables
    )
    table_ids = {table.name: ordinal for ordinal, table in enumerate(table_order)}
    values = []
    rows = []
    for table in table_order:
        value_base = len(values)
        table_values = []
        for value in table.values:
            spelling = spellings.get((table.name, value.name), value.name)
            table_values.append((spelling, value.value))
        table_values.sort(key=lambda item: item[0].encode("utf-8"))
        values.extend((strings.intern(name), value) for name, value in table_values)
        rows.append(
            (
                value_base,
                len(table_values),
                int(table.kind == NumericKind.FLAGS),
                int(table.unknown_policy.value == "preserve_nonzero"),
            )
        )
    return table_ids, rows, values


def _encode_grammar_program(
    grammar: RecordGrammar,
    records: tuple[WireRecord, ...],
    grammar_ids: dict[str, int],
    tail_ids: dict[str, int],
    domain_ids: dict[SymbolDomain, int],
    numeric_ids: dict[str, int],
    strings: BStringPool,
    scalar_constraints: _ScalarConstraints,
) -> bytes:
    program = bytearray()

    def append_field(
        record: WireRecord, field_name: str, *, with_constraint: bool = False
    ) -> None:
        program.extend(_field(record, field_name))
        if with_constraint:
            wire_field = _wire_field(record, field_name)
            program.append(
                scalar_constraints.for_rule(
                    wire_field.rule, wire_field.field.byte_length
                )
            )

    for element in grammar.elements:
        if isinstance(element, str):
            offset = strings.intern(element)
            program.extend((_FormatOpcode.LITERAL, offset & 0xFF, offset >> 8))
        elif isinstance(element, DeclareSymbol):
            if element.flags_field_name:
                program.extend(
                    (_FormatOpcode.DECLARE_FLAGS, domain_ids[element.domain])
                )
                append_field(grammar.record, element.flags_field_name)
            else:
                program.extend((_FormatOpcode.DECLARE, domain_ids[element.domain]))
        elif isinstance(element, (SymbolReference, NullableSymbolReference)):
            program.extend(
                (
                    _FormatOpcode.NULLABLE_SYMBOL
                    if isinstance(element, NullableSymbolReference)
                    else _FormatOpcode.SYMBOL,
                    domain_ids[element.domain],
                )
            )
            append_field(element.record, element.field_name)
        elif isinstance(element, ScalarField):
            program.append(_FormatOpcode.SCALAR)
            append_field(element.record, element.field_name, with_constraint=True)
            program.append(int(element.radix))
        elif isinstance(element, NumericField):
            program.extend((_FormatOpcode.NUMERIC, numeric_ids[element.table.name]))
            append_field(element.record, element.field_name)
            program.append(
                scalar_constraints.for_numeric_rule(
                    _wire_field(element.record, element.field_name).rule,
                    element.table,
                )
            )
        elif isinstance(element, ConstantField):
            offset, width = _field(element.record, element.field_name)
            program.extend((_FormatOpcode.CONSTANT, offset, width))
            program.extend(element.value.to_bytes(width, "little"))
        elif isinstance(element, NestedBody):
            program.extend(
                (
                    _FormatOpcode.BODY,
                    grammar_ids[element.grammar_name],
                    _record_id(records, element.child_record),
                )
            )
            for field_name in (element.base_field_name, element.count_field_name):
                if field_name:
                    append_field(
                        element.parent_record, field_name, with_constraint=True
                    )
                else:
                    program.extend((_NO_FIELD, 0, 0))
        elif isinstance(element, QuotedBlob):
            program.extend(
                (
                    _FormatOpcode.QUOTED_BLOB,
                    tail_ids[element.tail_name],
                    _record_id(records, element.offset_record),
                )
            )
            append_field(element.offset_record, "byte_offset_u32")
        elif isinstance(element, NativeCallBody):
            program.append(_FormatOpcode.NATIVE_CALL)
        elif isinstance(element, NativeLayoutBody):
            program.extend(
                (
                    _FormatOpcode.NATIVE_LAYOUT,
                    numeric_ids[element.kind_table.name],
                    domain_ids[SymbolDomain.REF_TYPE],
                    domain_ids[SymbolDomain.NATIVE_LAYOUT],
                )
            )
        elif isinstance(element, SignatureBody):
            program.extend(
                (
                    _FormatOpcode.SIGNATURE,
                    _record_id(records, element.descriptor_record),
                    numeric_ids[element.kind_table.name],
                    domain_ids[SymbolDomain.REF_TYPE],
                    domain_ids[SymbolDomain.CALLABLE],
                )
            )
            append_field(element.row_record, "descriptor_base_u32")
            for name in (
                "argument_value_count_u16",
                "argument_ref_count_u16",
                "argument_function_count_u16",
                "result_value_count_u16",
                "result_ref_count_u16",
                "result_function_count_u16",
            ):
                append_field(element.row_record, name)
            append_field(element.descriptor_record, "kind_u16")
            append_field(element.descriptor_record, "type_ordinal_u16")
        elif isinstance(element, FunctionBody):
            program.extend(
                (
                    _FormatOpcode.FUNCTION_BODY,
                    tail_ids[element.tail_name],
                    _record_id(records, element.switch_record),
                    domain_ids[SymbolDomain.SWITCH_TARGET],
                    domain_ids[SymbolDomain.BLOCK],
                )
            )
            for name in (
                "bytecode_offset_u32",
                "bytecode_length_u32",
                "switch_target_base_u32",
                "switch_target_entry_count_u32",
                "block_count_u32",
            ):
                append_field(element.row_record, name, with_constraint=True)
            append_field(element.switch_record, "target_word_offset_u32")
        elif isinstance(element, HexBlob):
            program.extend((_FormatOpcode.HEX_BLOB, tail_ids[element.tail_name]))
            append_field(element.record, element.length_field_name)
            append_field(element.record, element.alignment_field_name)
        elif isinstance(element, MetadataValue):
            program.extend(
                (
                    _FormatOpcode.METADATA_VALUE,
                    tail_ids[element.tail_name],
                    _record_id(records, element.offset_record),
                )
            )
            append_field(element.entry_record, element.type_field_name)
            append_field(element.offset_record, "byte_offset_u64")
        else:
            raise TypeError(f"unsupported grammar element {type(element)}")
    program.append(_FormatOpcode.END)
    return bytes(program)


def _render_instruction_tables(
    instruction_text: InstructionTextFormat,
    strings: BStringPool,
    numeric_ids: dict[str, int],
    scalar_constraints: _ScalarConstraints,
):
    domain_ids = {domain: ordinal for ordinal, domain in enumerate(SymbolDomain)}
    fields = []
    field_group_bases = {}
    direct_targets = []
    direct_target_group_bases = {}
    lanes = []
    lane_group_bases = {}
    relations = []
    relation_ordinals = {}
    global_partitions = []
    global_partition_ordinals = {}
    instructions = []
    for text in instruction_text.instructions:
        physical_fields = {field.field.name: field for field in text.instruction.fields}
        variants = ((text.instruction.mnemonic, 0, 0),)
        if text.mnemonic_projection:
            groups = {}
            for lane in text.mnemonic_projection.formats:
                groups.setdefault(lane.element_name, []).append(
                    (lane.encoded_value, lane.lane_count)
                )
            variants = []
            for element_name in sorted(groups, key=lambda value: value.encode("utf-8")):
                group = tuple(sorted(groups[element_name], key=lambda value: value[1]))
                lane_base = intern_row_group(group, lanes, lane_group_bases)
                if lane_base > 0xFF:
                    raise ValueError("assembler lane table base exceeds u8")
                variants.append(
                    (
                        f"{text.instruction.mnemonic}.{element_name}",
                        lane_base,
                        len(group),
                    )
                )

        target_base = intern_row_group(
            tuple(
                (
                    item.encoded_value,
                    domain_ids[item.symbol_domain],
                    item.symbol_flags_mask,
                    item.symbol_flags_value,
                )
                for item in text.direct_target_kinds
            ),
            direct_targets,
            direct_target_group_bases,
        )
        instruction_fields = []
        for item in text.fields:
            _validate_instruction_field_projection(text, item, physical_fields)
            primary_offset, primary_width = _field(text.instruction, item.primary_field)
            related_offset, related_width = _NO_FIELD, 0
            if item.related_field:
                related_offset, related_width = _field(
                    text.instruction, item.related_field
                )
            physical_field = physical_fields[item.primary_field]
            data = item.fixed_value
            data_count = 0
            if item.numeric_table:
                data = numeric_ids[item.numeric_table.name]
            elif item.symbol_domain:
                data = domain_ids[item.symbol_domain]
            if item.kind == InstructionFieldKind.DIRECT_TARGET:
                data = target_base
                data_count = len(text.direct_target_kinds)
            rule = physical_field.rule
            if item.kind == InstructionFieldKind.UNSIGNED and rule.kind in {
                instruction_rules.FieldRule.ALLOWED_RANGE,
                instruction_rules.FieldRule.ALLOWED_VALUES,
            }:
                data = scalar_constraints.for_rule(
                    rule, physical_field.field.byte_length
                )
            elif item.kind in {
                InstructionFieldKind.LOCAL_BYTES_FIXED_RANGE,
                InstructionFieldKind.LOCAL_BYTES_REPEATED_RANGE,
            }:
                data, data_count = rule.values
            elif (
                item.kind == InstructionFieldKind.ORDINAL
                and rule.kind == instruction_rules.FieldRule.FUNCTION_LOCAL_ORDINAL
            ):
                data_count = 1
            elif item.kind == InstructionFieldKind.MODULE_SYMBOL:
                if rule.kind == instruction_rules.FieldRule.GLOBAL_ORDINAL:
                    partition = rule.values[0]
                    ordinal = global_partition_ordinals.get(partition)
                    if ordinal is None:
                        ordinal = len(global_partitions)
                        global_partition_ordinals[partition] = ordinal
                        global_partitions.append(partition)
                    data_count = ordinal + 1
                elif rule.kind == instruction_rules.FieldRule.IMPORT_ORDINAL_OPTIONAL:
                    data_count = 0x80
            if item.bit_offset > 0xF or item.bit_length > 0xF:
                raise ValueError(
                    f"{text.instruction.name}.{item.name}: bit range exceeds u4"
                )
            instruction_fields.append(
                (
                    strings.intern(item.name),
                    data,
                    primary_offset,
                    related_offset,
                    int(item.kind),
                    primary_width,
                    related_width,
                    physical_field.field.element_count,
                    item.bit_offset | (item.bit_length << 4),
                    data_count,
                )
            )
        field_base = intern_row_group(instruction_fields, fields, field_group_bases)

        counts = tuple(
            sum(field.position == position for field in text.fields)
            for position in range(3)
        )
        if counts[0] > 0xF or counts[1] > 0xF:
            raise ValueError(
                f"{text.instruction.mnemonic}: result or operand count exceeds u4"
            )
        relation = _instruction_relation(text.instruction)
        relation_ordinal = 0
        if relation is not None:
            relation_ordinal = relation_ordinals.get(relation, 0)
            if relation_ordinal == 0:
                relations.append(relation)
                relation_ordinal = len(relations)
                relation_ordinals[relation] = relation_ordinal
            if relation_ordinal > 0xFF:
                raise ValueError("assembler instruction relations exceed u8")
        for name, lane_base, lane_count in variants:
            instructions.append(
                (
                    name,
                    (
                        strings.intern(name),
                        field_base,
                        text.instruction.opcode,
                        text.instruction.byte_length,
                        counts[0] | (counts[1] << 4),
                        counts[2],
                        lane_base,
                        lane_count,
                        list(type(text.instruction.control_flow)).index(
                            text.instruction.control_flow
                        ),
                        relation_ordinal,
                    ),
                )
            )
    instructions.sort(key=lambda item: item[0].encode("utf-8"))
    return (
        [row for _, row in instructions],
        fields,
        direct_targets,
        lanes,
        relations,
        global_partitions,
    )


def _render_module_tables(
    specification: Specification,
    module_text: ModuleTextFormat,
    strings: BStringPool,
    numeric_ids: dict[str, int],
    scalar_constraints: _ScalarConstraints,
):
    records = specification.module_format.records
    record_ids = {record: ordinal for ordinal, record in enumerate(records)}
    grammar_ids = {
        grammar.name: ordinal for ordinal, grammar in enumerate(module_text.grammars)
    }
    tail_ids = {name: ordinal for ordinal, name in enumerate(module_text.tail_names)}
    domain_ids = {domain: ordinal for ordinal, domain in enumerate(SymbolDomain)}

    program = bytearray()
    grammar_rows = []
    terminal_offsets = set()
    for grammar in module_text.grammars:
        program_offset = len(program)
        program.extend(
            _encode_grammar_program(
                grammar,
                records,
                grammar_ids,
                tail_ids,
                domain_ids,
                numeric_ids,
                strings,
                scalar_constraints,
            )
        )
        grammar_rows.append(
            (
                program_offset,
                strings.intern(grammar.keyword),
                record_ids[grammar.record],
                int(grammar.row_mode),
            )
        )
        for element in grammar.elements:
            if isinstance(element, (QuotedBlob, MetadataValue)):
                terminal_offsets.add(
                    (
                        tail_ids[element.tail_name],
                        record_ids[element.offset_record],
                    )
                )

    layout_rows = []
    run_rows = []
    section_rows = []
    for section_text in module_text.sections:
        run_base = len(run_rows)
        for run in section_text.runs:
            run_rows.append(
                (
                    grammar_ids[run.grammar_name],
                    int(isinstance(run.count, FixedRunCount)),
                )
            )
        layout_base = len(layout_rows)
        for item in section_text.layout:
            if isinstance(item, RecordLayout):
                adjustment = (
                    item.count.adjustment if hasattr(item.count, "adjustment") else 0
                )
                fixed_count = (
                    item.count.count if isinstance(item.count, FixedLayoutCount) else 0
                )
                layout_rows.append(
                    (
                        _LayoutKind.RECORD,
                        record_ids[item.record],
                        adjustment,
                        fixed_count,
                    )
                )
            elif isinstance(item, AlignmentLayout):
                layout_rows.append((_LayoutKind.ALIGN, 0, 0, item.alignment))
            elif isinstance(item, TailLayout):
                layout_rows.append((_LayoutKind.TAIL, tail_ids[item.name], 0, 0))
            else:
                raise TypeError(f"unsupported layout item {type(item)}")
        section = section_text.section
        values = (
            run_base,
            len(section_text.runs),
            layout_base,
            len(section_text.layout),
        )
        if any(value > 0xFF for value in values):
            raise ValueError(f"{section.name}: table span exceeds u8")
        section_rows.append(
            (
                strings.intern(section.name),
                section.section_type,
                section.required_flags,
                run_base,
                len(section_text.runs),
                layout_base,
                len(section_text.layout),
            )
        )

    for kind, target, adjustment, value in layout_rows:
        if kind > 0xFF or target > 0xFF or value > 0xFF:
            raise ValueError("assembler layout row exceeds u8")
        if adjustment < -0x80 or adjustment > 0x7F:
            raise ValueError("assembler layout adjustment exceeds i8")

    derivations = []
    statistic_ids = {
        name: ordinal for ordinal, name in enumerate(module_text.statistic_names)
    }
    for item in module_text.derived_fields:
        if isinstance(item, SectionDirectoryFieldDerivation):
            continue
        if isinstance(item, RecordCountDerivation):
            kind = _DerivationKind.RECORD_COUNT
            source = record_ids[item.source_record]
        elif isinstance(item, GrammarCountDerivation):
            kind = _DerivationKind.GRAMMAR_COUNT
            source = grammar_ids[item.grammar_name]
        elif isinstance(item, StatisticDerivation):
            kind = _DerivationKind.STATISTIC
            source = statistic_ids[item.statistic_name]
        else:
            raise TypeError(f"unsupported derivation {type(item)}")
        field_offset, field_width = _field(item.record, item.field_name)
        constraint = scalar_constraints.for_rule(
            _wire_field(item.record, item.field_name).rule, field_width
        )
        derivations.append(
            (
                record_ids[item.record],
                field_offset,
                field_width,
                kind,
                source,
                constraint,
            )
        )

    shell = module_text.shell
    return (
        program,
        grammar_rows,
        run_rows,
        layout_rows,
        section_rows,
        derivations,
        tuple(sorted(terminal_offsets)),
        tuple(
            strings.intern(value)
            for value in (
                shell.module_prefix,
                shell.version_separator,
                shell.module_open,
                shell.section_prefix,
                shell.section_alignment_prefix,
                shell.section_open,
                shell.close,
            )
        ),
        (
            *_field(shell.core_major.record, shell.core_major.field_name),
            *_field(
                shell.core_required_minor.record, shell.core_required_minor.field_name
            ),
            *_field(shell.section_alignment.record, shell.section_alignment.field_name),
        ),
        record_ids[shell.core_major.record],
        record_ids[shell.section_alignment.record],
        tuple(strings.intern(domain.value) for domain in SymbolDomain),
    )


def render_assembler_data(
    specification: Specification,
    instruction_text: InstructionTextFormat,
    module_text: ModuleTextFormat,
) -> str:
    """Renders all static data privately consumed by the assembler."""

    strings = BStringPool("assembler text")
    scalar_constraints = _ScalarConstraints()
    numeric_ids, numeric_rows, numeric_values = _numeric_tables(
        specification, instruction_text, module_text, strings
    )
    (
        instruction_rows,
        field_rows,
        direct_target_rows,
        lane_rows,
        relation_rows,
        global_partitions,
    ) = _render_instruction_tables(
        instruction_text, strings, numeric_ids, scalar_constraints
    )
    (
        program,
        grammar_rows,
        run_rows,
        layout_rows,
        section_rows,
        derivations,
        terminal_offsets,
        shell_literals,
        shell_fields,
        image_record,
        section_record,
        domain_names,
    ) = _render_module_tables(
        specification, module_text, strings, numeric_ids, scalar_constraints
    )

    lines = [generated_preamble(), ""]
    for prefix, values in (
        ("FORMAT", _FormatOpcode),
        ("FIELD", InstructionFieldKind),
        ("LAYOUT", _LayoutKind),
        ("DERIVATION", _DerivationKind),
        ("RELATION", _InstructionRelationKind),
        ("ROW", RowMode),
        ("SCALAR", ScalarRadix),
    ):
        lines.extend(
            f"#define IREE_VM_BYTECODE_ASSEMBLER_{prefix}_{item.name} {item.value}u"
            for item in values
        )
    lines.extend(
        (
            f"#define IREE_VM_BYTECODE_ASSEMBLER_NO_FIELD {_NO_FIELD}u",
            f"#define IREE_VM_BYTECODE_ASSEMBLER_RECORD_COUNT {len(specification.module_format.records)}u",
            f"#define IREE_VM_BYTECODE_ASSEMBLER_MAX_RECORD_SIZE {max(record.byte_length for record in specification.module_format.records)}u",
            f"#define IREE_VM_BYTECODE_ASSEMBLER_GRAMMAR_COUNT {len(module_text.grammars)}u",
            f"#define IREE_VM_BYTECODE_ASSEMBLER_TAIL_COUNT {len(module_text.tail_names)}u",
            f"#define IREE_VM_BYTECODE_ASSEMBLER_STATISTIC_COUNT {len(module_text.statistic_names)}u",
            "",
        )
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_ASSEMBLER_DOMAIN_{domain.name} {ordinal}u"
        for ordinal, domain in enumerate(SymbolDomain)
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_ASSEMBLER_CONTROL_{flow.name} {ordinal}u"
        for ordinal, flow in enumerate(isa.ControlFlow)
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_ASSEMBLER_STATISTIC_{name.upper()} {ordinal}u"
        for ordinal, name in enumerate(module_text.statistic_names)
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_ASSEMBLER_TAIL_{name.upper()} {ordinal}u"
        for ordinal, name in enumerate(module_text.tail_names)
    )
    lines.extend(
        (
            "#define IREE_VM_BYTECODE_ASSEMBLER_FIELD_DATA_FUNCTION_LOCAL 1u",
            "#define IREE_VM_BYTECODE_ASSEMBLER_FIELD_DATA_OPTIONAL_IMPORT 128u",
        )
    )
    lines.append("")
    lines.extend(
        render_bstring_pool(
            "static const uint8_t iree_vm_bytecode_assembler_strings[]", strings
        )
    )
    lines.extend(
        [
            "",
            "static const uint8_t iree_vm_bytecode_assembler_record_lengths[] = {",
            *(
                f"    {record.byte_length}u,"
                for record in specification.module_format.records
            ),
            "};",
            "",
            "static const uint16_t iree_vm_bytecode_assembler_domain_name_offsets[] = {",
            *(f"    {offset}u," for offset in domain_names),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_numeric_value_t iree_vm_bytecode_assembler_numeric_values[] = {",
            *(f"    {{{name}u, {value}u}}," for name, value in numeric_values),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_numeric_table_t iree_vm_bytecode_assembler_numeric_tables[] = {",
            *(
                f"    {{{base}u, {count}u, {flags}u, {preserve}u}},"
                for base, count, flags, preserve in numeric_rows
            ),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_direct_target_t iree_vm_bytecode_assembler_direct_targets[] = {",
            *(
                f"    {{{value}u, {domain}u, {flags_mask}u, {flags_value}u}},"
                for value, domain, flags_mask, flags_value in direct_target_rows
            ),
            "};",
            "",
            "static const uint32_t iree_vm_bytecode_assembler_scalar_minimums[] = {",
            *(
                f"    UINT32_C(0x{minimum:08X}),"
                for minimum, _, _ in scalar_constraints.rows
            ),
            "};",
            "",
            "static const uint32_t iree_vm_bytecode_assembler_scalar_maximums[] = {",
            *(
                f"    UINT32_C(0x{maximum:08X}),"
                for _, maximum, _ in scalar_constraints.rows
            ),
            "};",
            "",
            "static const uint32_t iree_vm_bytecode_assembler_scalar_power_of_two_mask = "
            "UINT32_C(0x%08X);"
            % sum(
                (1 << ordinal) if power_of_two else 0
                for ordinal, (_, _, power_of_two) in enumerate(scalar_constraints.rows)
            ),
            "",
            "static const uint32_t iree_vm_bytecode_assembler_global_partitions[] = {",
            *(f"    UINT32_C(0x{value:08X})," for value in global_partitions),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_field_t iree_vm_bytecode_assembler_fields[] = {",
            *(
                "    {%uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu}," % row
                for row in field_rows
            ),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_lane_t iree_vm_bytecode_assembler_lanes[] = {",
            *("    {%uu, %uu}," % row for row in lane_rows),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_instruction_t iree_vm_bytecode_assembler_instructions[] = {",
            *(
                "    {%uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu}," % row
                for row in instruction_rows
            ),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_relation_t iree_vm_bytecode_assembler_relations[] = {",
            *(
                "    {UINT64_C(0x%016X), {%s}, %uu},"
                % (row[0], ", ".join(f"{value}u" for value in row[1:6]), row[6])
                for row in relation_rows
            ),
            "};",
            "",
        ]
    )
    lines.extend(
        render_byte_array(
            "static const uint8_t iree_vm_bytecode_assembler_program[]", program
        )
    )
    lines.extend(
        [
            "",
            "static const iree_vm_bytecode_assembler_grammar_t iree_vm_bytecode_assembler_grammars[] = {",
            *("    {%uu, %uu, %uu, %uu}," % row for row in grammar_rows),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_run_t iree_vm_bytecode_assembler_runs[] = {",
            *("    {%uu, %uu}," % row for row in run_rows),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_layout_t iree_vm_bytecode_assembler_layouts[] = {",
            *("    {%uu, %uu, %d, %uu}," % row for row in layout_rows),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_section_t iree_vm_bytecode_assembler_sections[] = {",
            *("    {%uu, %uu, %uu, %uu, %uu, %uu, %uu}," % row for row in section_rows),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_derivation_t iree_vm_bytecode_assembler_derivations[] = {",
            *("    {%uu, %uu, %uu, %uu, %uu, %uu}," % row for row in derivations),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_terminal_offset_t iree_vm_bytecode_assembler_terminal_offsets[] = {",
            *("    {%uu, %uu}," % row for row in terminal_offsets),
            "};",
            "",
            "static const iree_vm_bytecode_assembler_shell_t iree_vm_bytecode_assembler_shell = {",
            "    {%s}," % ", ".join(f"{value}u" for value in shell_literals),
            f"    {image_record}u, {section_record}u,",
            "    %s," % ", ".join(f"{value}u" for value in shell_fields),
            "};",
            "",
        ]
    )
    return "\n".join(lines)
