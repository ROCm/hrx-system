# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders private table programs for the canonical VM disassembler."""

from __future__ import annotations

import enum

from iree.vm.bytecode.spec.module import WireRecord
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
    FinalByteExtent,
    FinalExtentLayoutCount,
    FixedLayoutCount,
    FixedRunCount,
    FunctionBody,
    HeaderFieldLayoutCount,
    HeaderFieldRunCount,
    HexBlob,
    MatchingFieldRunCount,
    MetadataValue,
    ModuleTextFormat,
    NativeCallBody,
    NativeLayoutBody,
    NestedBody,
    NullableSymbolReference,
    NumericField,
    QuotedBlob,
    RecordLayout,
    RecordRunCount,
    RemainingByteLength,
    RemainingLayoutCount,
    RowMode,
    ScalarField,
    ScalarRadix,
    SignatureBody,
    SummedFieldLayoutCount,
    SymbolReference,
    TailLayout,
)


class _FormatOpcode(enum.IntEnum):
    END = 0x00
    DECLARE = 0xE0
    SYMBOL = 0xE1
    NULLABLE_SYMBOL = 0xE2
    SCALAR = 0xE3
    NUMERIC = 0xE4
    CONSTANT = 0xE5
    BODY = 0xE6
    QUOTED_BLOB = 0xE7
    SIGNATURE = 0xE8
    FUNCTION_BODY = 0xE9
    HEX_BLOB = 0xEA
    METADATA_VALUE = 0xEB
    NATIVE_LAYOUT = 0xEC
    NATIVE_CALL = 0xED


class _LayoutKind(enum.IntEnum):
    RECORD_FIXED = 0
    RECORD_REMAINING = 1
    RECORD_FIELD = 2
    RECORD_SUM = 3
    RECORD_FINAL_EXTENT = 4
    ALIGN = 5
    TAIL_REMAINING = 6
    TAIL_FINAL_EXTENT = 7


class _RunKind(enum.IntEnum):
    FIXED = 0
    RECORD = 1
    HEADER_FIELD = 2
    MATCH_FIELD = 3


_NO_FIELD = 0xFF
_NO_RECORD = 0xFF


def _record_id(records: tuple[WireRecord, ...], record: WireRecord) -> int:
    return records.index(record)


def _field(record: WireRecord, name: str) -> tuple[int, int]:
    for wire_field, offset in zip(record.fields, record.field_offsets, strict=True):
        if wire_field.field.name == name:
            if offset > 0xFF or wire_field.field.byte_length > 0xFF:
                raise ValueError(f"{record.name}.{name}: field layout exceeds u8")
            return offset, wire_field.field.byte_length
    raise ValueError(f"{record.name}: unknown field {name}")


def _append_literal(program: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    while encoded:
        fragment, encoded = encoded[:0x7F], encoded[0x7F:]
        program.append(len(fragment))
        program.extend(fragment)


def _append_field(program: bytearray, record: WireRecord, field_name: str) -> None:
    program.extend(_field(record, field_name))


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
                continue
            if not isinstance(element, NumericField):
                continue
            tables[element.table.name] = element.table
            for override in element.spelling_overrides:
                key = (element.table.name, override.source_name)
                previous = spellings.setdefault(key, override.text)
                if previous != override.text:
                    raise ValueError(f"conflicting spelling for {key}")

    table_order = tuple(
        table
        for table in specification.module_format.numeric_tables
        + specification.selectors
        if table.name in tables
    )
    table_ids = {table.name: ordinal for ordinal, table in enumerate(table_order)}
    value_rows = []
    table_rows = []
    for table in table_order:
        value_base = len(value_rows)
        for value in table.values:
            spelling = spellings.get((table.name, value.name), value.name)
            value_rows.append((strings.intern(spelling), value.value))
        table_rows.append(
            (
                value_base,
                len(table.values),
                int(table.kind == NumericKind.FLAGS),
                int(table.unknown_policy.value == "preserve_nonzero"),
            )
        )
    return table_ids, table_rows, value_rows


def _render_instruction_tables(
    instruction_text: InstructionTextFormat,
    names: BStringPool,
    strings: BStringPool,
    numeric_ids: dict[str, int],
):
    instruction_rows = [(0, 0, 0, 0, 0, 0, _NO_FIELD, 0)] * 256
    field_rows = []
    field_group_bases = {}
    lane_rows = []
    lane_group_bases = {}
    direct_target_rows = []
    direct_target_group_bases = {}
    domain_ids = {domain: ordinal for ordinal, domain in enumerate(SymbolDomain)}
    instruction_name_offsets = [0] * 256

    for text in instruction_text.instructions:
        instruction = text.instruction
        instruction_name_offsets[instruction.opcode] = names.intern(
            instruction.mnemonic
        )
        lane_base = len(lane_rows)
        projection_offset = _NO_FIELD
        if text.mnemonic_projection:
            projection_offset = _field(
                instruction, text.mnemonic_projection.field_name
            )[0]
            lane_base = intern_row_group(
                tuple(
                    (
                        strings.intern(item.element_name),
                        item.encoded_value,
                        item.lane_count,
                    )
                    for item in text.mnemonic_projection.formats
                ),
                lane_rows,
                lane_group_bases,
            )

        target_base = intern_row_group(
            tuple(
                (item.encoded_value, domain_ids[item.symbol_domain])
                for item in text.direct_target_kinds
            ),
            direct_target_rows,
            direct_target_group_bases,
        )
        instruction_fields = []
        for item in text.fields:
            primary_offset, primary_width = _field(instruction, item.primary_field)
            related_offset, related_width = _NO_FIELD, 0
            if item.related_field:
                related_offset, related_width = _field(instruction, item.related_field)
            physical_field = next(
                field
                for field in instruction.fields
                if field.field.name == item.primary_field
            )
            data = item.fixed_value
            data_count = 0
            if item.numeric_table:
                data = numeric_ids[item.numeric_table.name]
            elif item.symbol_domain:
                data = domain_ids[item.symbol_domain]
            if item.kind == InstructionFieldKind.DIRECT_TARGET:
                data = target_base
                data_count = len(text.direct_target_kinds)
            if item.bit_offset > 0xF or item.bit_length > 0xF:
                raise ValueError(
                    f"{instruction.name}.{item.name}: bit range exceeds u4"
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
        field_base = intern_row_group(instruction_fields, field_rows, field_group_bases)

        counts = [
            sum(field.position == position for field in text.fields)
            for position in range(3)
        ]
        instruction_rows[instruction.opcode] = (
            field_base,
            lane_base,
            instruction.byte_length,
            counts[0],
            counts[1],
            len(text.fields),
            projection_offset,
            len(text.mnemonic_projection.formats) if text.mnemonic_projection else 0,
        )
    return (
        instruction_name_offsets,
        instruction_rows,
        field_rows,
        lane_rows,
        direct_target_rows,
    )


def _encode_grammar_program(
    grammar,
    records: tuple[WireRecord, ...],
    grammar_ids: dict[str, int],
    tail_ids: dict[str, int],
    domain_ids: dict[SymbolDomain, int],
    numeric_ids: dict[str, int],
) -> bytes:
    program = bytearray()
    for element in grammar.elements:
        if isinstance(element, str):
            _append_literal(program, element)
        elif isinstance(element, DeclareSymbol):
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
            _append_field(program, element.record, element.field_name)
        elif isinstance(element, ScalarField):
            program.append(_FormatOpcode.SCALAR)
            _append_field(program, element.record, element.field_name)
            program.append(int(element.radix))
        elif isinstance(element, NumericField):
            program.extend((_FormatOpcode.NUMERIC, numeric_ids[element.table.name]))
            _append_field(program, element.record, element.field_name)
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
                    _append_field(program, element.parent_record, field_name)
                else:
                    program.extend((_NO_FIELD, 0))
        elif isinstance(element, QuotedBlob):
            program.extend(
                (
                    _FormatOpcode.QUOTED_BLOB,
                    tail_ids[element.tail_name],
                    _record_id(records, element.offset_record),
                )
            )
            offset_field = element.offset_record.fields[0].field
            program.extend(
                (
                    element.offset_record.field_offsets[0],
                    offset_field.encoding.byte_length,
                )
            )
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
            _append_field(program, element.row_record, "descriptor_base_u32")
            for name in (
                "argument_value_count_u16",
                "argument_ref_count_u16",
                "argument_function_count_u16",
                "result_value_count_u16",
                "result_ref_count_u16",
                "result_function_count_u16",
            ):
                _append_field(program, element.row_record, name)
            _append_field(program, element.descriptor_record, "kind_u16")
            _append_field(program, element.descriptor_record, "type_ordinal_u16")
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
                _append_field(program, element.row_record, name)
            _append_field(program, element.switch_record, "target_word_offset_u32")
        elif isinstance(element, HexBlob):
            program.extend((_FormatOpcode.HEX_BLOB, tail_ids[element.tail_name]))
            _append_field(program, element.record, element.length_field_name)
            _append_field(program, element.record, element.alignment_field_name)
        elif isinstance(element, MetadataValue):
            program.extend(
                (
                    _FormatOpcode.METADATA_VALUE,
                    tail_ids[element.tail_name],
                    _record_id(records, element.offset_record),
                )
            )
            _append_field(program, element.entry_record, element.type_field_name)
            offset_field = element.offset_record.fields[0].field
            program.extend(
                (
                    element.offset_record.field_offsets[0],
                    offset_field.encoding.byte_length,
                )
            )
        else:
            raise TypeError(f"unsupported grammar element {type(element)}")
    program.append(_FormatOpcode.END)
    return bytes(program)


def _render_module_tables(
    specification: Specification,
    module_text: ModuleTextFormat,
    strings: BStringPool,
    numeric_ids: dict[str, int],
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
            )
        )
        grammar_rows.append(
            (
                program_offset,
                record_ids[grammar.record],
                int(grammar.row_mode),
            )
        )

    layout_fields = []
    layout_rows = []
    run_rows = []
    section_rows = [(0, 0, 0, 0, 0, 0)] * (
        max(section.section_type for section in specification.module_format.sections)
        + 1
    )
    for section_text in module_text.sections:
        run_base = len(run_rows)
        for run in section_text.runs:
            count = run.count
            source = _NO_RECORD
            field_offset = _NO_FIELD
            field_width = 0
            value = 0
            if isinstance(count, FixedRunCount):
                kind, value = _RunKind.FIXED, count.count
            elif isinstance(count, RecordRunCount):
                kind, source = _RunKind.RECORD, record_ids[count.record]
            elif isinstance(count, HeaderFieldRunCount):
                kind, source = _RunKind.HEADER_FIELD, record_ids[count.record]
                field_offset, field_width = _field(count.record, count.field_name)
            elif isinstance(count, MatchingFieldRunCount):
                kind, source, value = (
                    _RunKind.MATCH_FIELD,
                    record_ids[count.record],
                    count.value,
                )
                field_offset, field_width = _field(count.record, count.field_name)
            else:
                raise TypeError(f"unsupported grammar run {type(count)}")
            run_rows.append(
                (
                    grammar_ids[run.grammar_name],
                    kind,
                    source,
                    field_offset,
                    field_width,
                    value,
                )
            )

        layout_base = len(layout_rows)
        for item in section_text.layout:
            source = _NO_RECORD
            target = _NO_RECORD
            field_base = len(layout_fields)
            field_count = 0
            adjustment = 0
            value = 0
            if isinstance(item, RecordLayout):
                target = record_ids[item.record]
                count = item.count
                if isinstance(count, FixedLayoutCount):
                    kind, value = _LayoutKind.RECORD_FIXED, count.count
                elif isinstance(count, RemainingLayoutCount):
                    kind = _LayoutKind.RECORD_REMAINING
                elif isinstance(count, HeaderFieldLayoutCount):
                    kind, source, adjustment = (
                        _LayoutKind.RECORD_FIELD,
                        record_ids[count.record],
                        count.adjustment,
                    )
                    layout_fields.append(_field(count.record, count.field_name))
                elif isinstance(count, SummedFieldLayoutCount):
                    kind, source = _LayoutKind.RECORD_SUM, record_ids[count.record]
                    layout_fields.extend(
                        _field(count.record, name) for name in count.field_names
                    )
                elif isinstance(count, FinalExtentLayoutCount):
                    kind, source = (
                        _LayoutKind.RECORD_FINAL_EXTENT,
                        record_ids[count.record],
                    )
                    layout_fields.extend(
                        (
                            _field(count.record, count.base_field_name),
                            _field(count.record, count.count_field_name),
                        )
                    )
                else:
                    raise TypeError(f"unsupported layout count {type(count)}")
                field_count = len(layout_fields) - field_base
            elif isinstance(item, AlignmentLayout):
                kind, value = _LayoutKind.ALIGN, item.alignment
            elif isinstance(item, TailLayout):
                target = tail_ids[item.name]
                if isinstance(item.byte_length, RemainingByteLength):
                    kind = _LayoutKind.TAIL_REMAINING
                elif isinstance(item.byte_length, FinalByteExtent):
                    kind, source = (
                        _LayoutKind.TAIL_FINAL_EXTENT,
                        record_ids[item.byte_length.record],
                    )
                    layout_fields.extend(
                        (
                            _field(
                                item.byte_length.record,
                                item.byte_length.base_field_name,
                            ),
                            _field(
                                item.byte_length.record,
                                item.byte_length.length_field_name,
                            ),
                        )
                    )
                    field_count = 2
                else:
                    raise TypeError(f"unsupported tail length {type(item.byte_length)}")
            else:
                raise TypeError(f"unsupported layout item {type(item)}")
            layout_rows.append(
                (
                    value,
                    field_base,
                    kind,
                    target,
                    source,
                    field_count,
                    adjustment,
                )
            )

        section = section_text.section
        section_rows[section.section_type] = (
            strings.intern(section.name),
            section.section_type,
            run_base,
            layout_base,
            len(section_text.runs),
            len(section_text.layout),
        )

    shell = module_text.shell
    shell_literals = tuple(
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
    )
    shell_fields = (
        *_field(shell.core_major.record, shell.core_major.field_name),
        *_field(
            shell.core_required_minor.record,
            shell.core_required_minor.field_name,
        ),
        *_field(shell.section_alignment.record, shell.section_alignment.field_name),
    )
    return (
        program,
        grammar_rows,
        run_rows,
        layout_fields,
        layout_rows,
        section_rows,
        shell_literals,
        shell_fields,
        record_ids[shell.core_major.record],
        record_ids[shell.section_alignment.record],
        tuple(strings.intern(domain.value) for domain in SymbolDomain),
    )


def render_disassembler_data(
    specification: Specification,
    instruction_text: InstructionTextFormat,
    module_text: ModuleTextFormat,
) -> str:
    """Renders all static data privately consumed by the disassembler."""

    names = BStringPool("disassembler name")
    strings = BStringPool("disassembler text")
    numeric_ids, numeric_rows, numeric_values = _numeric_tables(
        specification, instruction_text, module_text, strings
    )
    (
        instruction_names,
        instruction_rows,
        field_rows,
        lane_rows,
        direct_target_rows,
    ) = _render_instruction_tables(instruction_text, names, strings, numeric_ids)
    record_names = [
        names.intern(record.name) for record in specification.module_format.records
    ]
    (
        program,
        grammar_rows,
        run_rows,
        layout_fields,
        layout_rows,
        section_rows,
        shell_literals,
        shell_fields,
        image_record,
        section_record,
        domain_names,
    ) = _render_module_tables(specification, module_text, strings, numeric_ids)

    lines = [generated_preamble(), ""]
    lines.extend(
        f"#define IREE_VM_BYTECODE_DISASSEMBLER_FORMAT_{item.name} 0x{item.value:02X}u"
        for item in _FormatOpcode
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_DISASSEMBLER_FIELD_{item.name} {item.value}u"
        for item in InstructionFieldKind
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_DISASSEMBLER_LAYOUT_{item.name} {item.value}u"
        for item in _LayoutKind
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_DISASSEMBLER_RUN_{item.name} {item.value}u"
        for item in _RunKind
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_DISASSEMBLER_ROW_{item.name} {item.value}u"
        for item in RowMode
    )
    lines.extend(
        f"#define IREE_VM_BYTECODE_DISASSEMBLER_SCALAR_{item.name} {item.value}u"
        for item in ScalarRadix
    )
    lines.extend(
        (
            f"#define IREE_VM_BYTECODE_DISASSEMBLER_NO_FIELD {_NO_FIELD}u",
            f"#define IREE_VM_BYTECODE_DISASSEMBLER_TAIL_COUNT {len(module_text.tail_names)}u",
            f"#define IREE_VM_BYTECODE_DISASSEMBLER_DOMAIN_COUNT {len(SymbolDomain)}u",
            "",
        )
    )
    lines.extend(
        render_bstring_pool(
            "static const uint8_t iree_vm_bytecode_disassembler_names[]", names
        )
    )
    lines.extend(
        [
            "",
            "static const uint16_t iree_vm_bytecode_instruction_name_offsets[256] = {",
            *(f"    {offset}u," for offset in instruction_names),
            "};",
            "",
            "static const uint16_t iree_vm_bytecode_module_record_name_offsets[] = {",
            *(f"    {offset}u," for offset in record_names),
            "};",
            "",
            "static const uint8_t iree_vm_bytecode_disassembler_record_lengths[] = {",
            *(
                f"    {record.byte_length}u,"
                for record in specification.module_format.records
            ),
            "};",
            "",
        ]
    )
    lines.extend(
        render_byte_array(
            "static const uint8_t iree_vm_bytecode_disassembler_strings[]",
            strings.data,
        )
    )
    lines.extend(
        [
            "",
            "static const uint16_t iree_vm_bytecode_disassembler_domain_name_offsets[] = {",
            *(f"    {offset}u," for offset in domain_names),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_numeric_value_t iree_vm_bytecode_disassembler_numeric_values[] = {",
            *(f"    {{{name}u, {value}u}}," for name, value in numeric_values),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_numeric_table_t iree_vm_bytecode_disassembler_numeric_tables[] = {",
            *(
                f"    {{{base}u, {count}u, {flags}u, {preserve}u}},"
                for base, count, flags, preserve in numeric_rows
            ),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_lane_t iree_vm_bytecode_disassembler_lanes[] = {",
            *(
                f"    {{{name}u, {value}u, {count}u}},"
                for name, value, count in lane_rows
            ),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_direct_target_t iree_vm_bytecode_disassembler_direct_targets[] = {",
            *(f"    {{{value}u, {domain}u}}," for value, domain in direct_target_rows),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_field_t iree_vm_bytecode_disassembler_fields[] = {",
            *(
                "    {%uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu}," % row
                for row in field_rows
            ),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_instruction_t iree_vm_bytecode_disassembler_instructions[256] = {",
            *(
                "    {%uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu}," % row
                for row in instruction_rows
            ),
            "};",
            "",
        ]
    )
    lines.extend(
        render_byte_array(
            "static const uint8_t iree_vm_bytecode_disassembler_program[]",
            program,
        )
    )
    lines.extend(
        [
            "",
            "static const iree_vm_bytecode_disassembler_grammar_t iree_vm_bytecode_disassembler_grammars[] = {",
            *("    {%uu, %uu, %uu}," % row for row in grammar_rows),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_run_t iree_vm_bytecode_disassembler_runs[] = {",
            *("    {%uu, %uu, %uu, %uu, %uu, %uu}," % row for row in run_rows),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_layout_field_t iree_vm_bytecode_disassembler_layout_fields[] = {",
            *("    {%uu, %uu}," % row for row in layout_fields),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_layout_item_t iree_vm_bytecode_disassembler_layout_items[] = {",
            *("    {%uu, %uu, %uu, %uu, %uu, %uu, %d}," % row for row in layout_rows),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_section_t iree_vm_bytecode_disassembler_sections[] = {",
            *("    {%uu, %uu, %uu, %uu, %uu, %uu}," % row for row in section_rows),
            "};",
            "",
            "static const iree_vm_bytecode_disassembler_shell_t iree_vm_bytecode_disassembler_shell = {",
            "    {%s}," % ", ".join(f"{value}u" for value in shell_literals),
            f"    {image_record}u, {section_record}u,",
            "    %s," % ", ".join(f"{value}u" for value in shell_fields),
            "};",
            "",
        ]
    )
    return "\n".join(lines)
