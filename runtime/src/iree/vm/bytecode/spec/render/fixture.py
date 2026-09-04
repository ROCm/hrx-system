# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders the canonical structural-verification module fixture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from iree.vm.bytecode.spec import isa, module
from iree.vm.bytecode.spec.isa.core import rules as core_rules
from iree.vm.bytecode.spec.module import records
from iree.vm.bytecode.spec.specification import Specification


def _encode_fields(
    fields: Sequence,
    offsets: Sequence[int],
    byte_length: int,
    values: Mapping[str, int | bytes],
) -> bytes:
    unknown_names = values.keys() - {item.field.name for item in fields}
    if unknown_names:
        raise ValueError(f"unknown fixture fields: {sorted(unknown_names)}")
    data = bytearray(byte_length)
    for item, offset in zip(fields, offsets, strict=True):
        value = values.get(item.field.name)
        if value is None and isinstance(item.rule.data, bytes):
            value = item.rule.data
        if value is None:
            continue
        field_length = item.field.byte_length
        if isinstance(value, int):
            encoded = value.to_bytes(field_length, "little", signed=value < 0)
        else:
            encoded = value
        if len(encoded) != field_length:
            raise ValueError(f"{item.field.name}: fixture value has wrong length")
        data[offset : offset + field_length] = encoded
    return bytes(data)


def _record(record: module.WireRecord, **values: int | bytes) -> bytes:
    return _encode_fields(
        record.fields, record.field_offsets, record.byte_length, values
    )


def _instruction(instruction: isa.Instruction, **values: int | bytes) -> bytes:
    data = bytearray(
        _encode_fields(
            instruction.fields,
            instruction.field_offsets,
            instruction.byte_length,
            values,
        )
    )
    data[0] = instruction.opcode
    return bytes(data)


def _align(data: bytearray, alignment: int) -> None:
    data.extend(bytes((-len(data)) % alignment))


def _number(specification: Specification, table_name: str, value_name: str) -> int:
    tables = specification.module_format.numeric_tables + specification.selectors
    table = next(item for item in tables if item.name == table_name)
    return next(item.value for item in table.values if item.name == value_name)


def _strings(values: tuple[str, ...]) -> tuple[bytes, dict[str, int]]:
    encoded = tuple(value.encode("utf-8") for value in values)
    offsets = [0]
    for value in encoded:
        offsets.append(offsets[-1] + len(value))
    payload = bytearray(_record(records.STRINGS_HEADER, string_count_u32=len(values)))
    payload.extend(
        b"".join(
            _record(records.STRING_OFFSET, byte_offset_u32=offset) for offset in offsets
        )
    )
    payload.extend(b"".join(encoded))
    return bytes(payload), {value: i for i, value in enumerate(values)}


def _valid_instruction_values(
    instruction: isa.Instruction,
) -> dict[str, int]:
    """Returns deterministic field values satisfying declarative local rules."""

    values: dict[str, int] = {}
    for instruction_field in instruction.fields:
        field_name = instruction_field.field.name
        rule = instruction_field.rule
        if rule.kind == core_rules.FieldRule.ALLOWED_RANGE:
            values[field_name] = rule.values[0]
        elif rule.kind == core_rules.FieldRule.ALLOWED_VALUES:
            values[field_name] = rule.values[0]
        elif rule.kind == core_rules.FieldRule.SELECTOR:
            values[field_name] = rule.data.values[0].value
        elif rule.kind == core_rules.FieldRule.PACKED_SELECTORS:
            packed_value = 0
            for component in rule.data:
                allowed_values = component.allowed_values or tuple(
                    item.value for item in component.table.values
                )
                packed_value |= allowed_values[0] << component.bit_offset
            values[field_name] = packed_value
        elif rule.kind == core_rules.FieldRule.GLOBAL_ORDINAL:
            # The fixture defines one immutable and one mutable global in every
            # storage domain. A nonzero encoded lower-bound offset selects the
            # mutable suffix.
            values[field_name] = 1 if rule.values[0] >> 16 else 0
        elif rule.kind == core_rules.FieldRule.LOCAL_BYTES_RANGE_BASE:
            values[rule.fields[0]] = 1
        elif rule.kind == core_rules.FieldRule.LOCAL_BYTES_REPEATED_BASE:
            values[rule.fields[0]] = 1
        elif rule.kind == core_rules.FieldRule.RODATA_STATIC_OFFSET:
            values[rule.fields[1]] = 1

    for record_rule in instruction.rules:
        if record_rule.kind == core_rules.RecordRuleKind.FIELDS_DISTINCT:
            values[record_rule.fields[1]] = 1
        elif record_rule.kind == core_rules.RecordRuleKind.SWITCH_TARGETS:
            values[record_rule.fields[0]] = 1
            values[record_rule.fields[1]] = 0
        elif record_rule.kind == core_rules.RecordRuleKind.FUNCTION_ADDRESS:
            values[record_rule.fields[0]] = 0
            values[record_rule.fields[1]] = 0
            values[record_rule.fields[2]] = 1
        elif record_rule.kind == core_rules.RecordRuleKind.INTEGER_BITSTREAM_SHAPE:
            field_width, source_count, result_count, source_width, result_width = (
                record_rule.fields
            )
            values[field_width] = 8
            values[source_count] = 1
            values[result_count] = 1
            values[source_width] = 8
            values[result_width] = 8
    return values


def _all_core_bytecode(specification: Specification) -> bytes:
    """Returns one structurally valid stream containing every Core opcode."""

    instruction_by_name = {item.mnemonic: item for item in specification.instructions}
    block = instruction_by_name["control.block"]
    return_instruction = instruction_by_name["control.return"]
    bytecode = bytearray(_instruction(block))
    for instruction in sorted(specification.instructions, key=lambda item: item.opcode):
        if instruction is block:
            continue
        record_offset = len(bytecode)
        values = _valid_instruction_values(instruction)
        for instruction_field in instruction.fields:
            if instruction_field.rule.kind not in core_rules.DIRECT_TARGET_RULES:
                continue
            values[instruction_field.field.name] = -(
                (record_offset + instruction.byte_length) // 4
            )
        bytecode.extend(_instruction(instruction, **values))
    bytecode.extend(_instruction(return_instruction))
    return bytes(bytecode)


def render_structural_module_fixture(specification: Specification) -> bytes:
    """Returns one valid image containing every Core module section."""

    string_values = (
        "dep",
        "entry",
        "run",
        "vm",
        "buffer",
        "Fixture function.",
        "(i32, vm.ref<vm, buffer>, func.ref<() -> ()>) -> i64",
        "value",
        "i32",
        "payload",
        "vm.ref<vm, buffer>",
        "callback",
        "func.ref<() -> ()>",
        "result",
        "i64",
        "author",
        "feature",
        "revision",
    )
    strings, string_ordinal = _strings(string_values)

    signature_i32 = _number(specification, "signature_kind", "i32")
    signature_i64 = _number(specification, "signature_kind", "i64")
    signature_ref = _number(specification, "signature_kind", "ref")
    signature_function = _number(specification, "signature_kind", "function")
    import_optional = _number(specification, "import_flag", "optional")
    callable_may_yield = _number(specification, "callable_type_flag", "may_yield")
    function_may_yield = _number(specification, "function_flag", "may_yield")
    function_has_call = _number(specification, "function_flag", "has_call")
    global_ref_nullable = _number(specification, "global_ref_flag", "nullable")
    global_function_nullable = _number(
        specification, "global_function_flag", "nullable"
    )
    presentation_import = _number(
        specification, "presentation_declaration_kind", "import"
    )
    presentation_export = _number(
        specification, "presentation_declaration_kind", "export"
    )
    metadata_bool = _number(specification, "metadata_value_type", "bool")
    metadata_u64 = _number(specification, "metadata_value_type", "u64")
    metadata_utf8 = _number(specification, "metadata_value_type", "utf8")

    ref_types = b"".join(
        (
            _record(records.REF_TYPES_HEADER, group_count_u32=1),
            _record(
                records.REF_TYPE_GROUP_ROW,
                namespace_string_u16=string_ordinal["vm"],
                entry_count_u32=1,
            ),
            _record(
                records.REF_TYPE_ENTRY_ROW,
                type_name_string_u16=string_ordinal["buffer"],
            ),
        )
    )
    signature_descriptors = b"".join(
        (
            _record(
                records.SIGNATURE_DESCRIPTOR_ROW,
                kind_u16=signature_i32,
                type_ordinal_u16=0,
            ),
            _record(
                records.SIGNATURE_DESCRIPTOR_ROW,
                kind_u16=signature_ref,
                type_ordinal_u16=0,
            ),
            _record(
                records.SIGNATURE_DESCRIPTOR_ROW,
                kind_u16=signature_function,
                type_ordinal_u16=0,
            ),
            _record(
                records.SIGNATURE_DESCRIPTOR_ROW,
                kind_u16=signature_i64,
                type_ordinal_u16=0,
            ),
        )
    )
    broad_signature_descriptors = b"".join(
        _record(
            records.SIGNATURE_DESCRIPTOR_ROW,
            kind_u16=kind,
            type_ordinal_u16=0,
        )
        for kind in (
            *([signature_i64] * 17),
            *([signature_ref] * 17),
            *([signature_function] * 17),
            *([signature_i64] * 17),
            *([signature_ref] * 17),
            *([signature_function] * 17),
        )
    )
    signatures = b"".join(
        (
            _record(records.SIGNATURES_HEADER, signature_count_u32=3),
            _record(records.SIGNATURE_ROW, descriptor_base_u32=0),
            _record(
                records.SIGNATURE_ROW,
                descriptor_base_u32=0,
                argument_value_count_u16=1,
                result_value_count_u16=1,
                argument_ref_count_u16=1,
                argument_function_count_u16=1,
            ),
            _record(
                records.SIGNATURE_ROW,
                descriptor_base_u32=4,
                argument_value_count_u16=17,
                result_value_count_u16=17,
                argument_ref_count_u16=17,
                result_ref_count_u16=17,
                argument_function_count_u16=17,
                result_function_count_u16=17,
            ),
            signature_descriptors,
            broad_signature_descriptors,
        )
    )
    callable_types = b"".join(
        (
            _record(records.CALLABLE_TYPES_HEADER, callable_type_count_u32=3),
            _record(
                records.CALLABLE_TYPE_ROW,
                signature_ordinal_u16=0,
                nesting_depth_u16=0,
            ),
            _record(
                records.CALLABLE_TYPE_ROW,
                signature_ordinal_u16=1,
                nesting_depth_u16=1,
            ),
            _record(
                records.CALLABLE_TYPE_ROW,
                signature_ordinal_u16=2,
                flags_u16=callable_may_yield,
                nesting_depth_u16=1,
            ),
        )
    )
    imports = b"".join(
        (
            _record(records.IMPORTS_HEADER, group_count_u32=1),
            _record(
                records.IMPORT_GROUP_ROW,
                module_name_string_u16=string_ordinal["dep"],
                entry_count_u32=1,
            ),
            _record(
                records.IMPORT_ENTRY_ROW,
                symbol_name_string_u16=string_ordinal["entry"],
                callable_type_ordinal_u16=0,
                flags_u16=import_optional,
            ),
        )
    )
    exports = b"".join(
        (
            _record(records.EXPORTS_HEADER, export_count_u32=1),
            _record(
                records.EXPORT_ROW,
                name_string_u16=string_ordinal["run"],
                callable_type_ordinal_u16=1,
                function_ordinal_u16=0,
            ),
        )
    )
    instruction_by_name = {item.mnemonic: item for item in specification.instructions}
    bytecode = b"".join(
        (
            _instruction(instruction_by_name["control.block"]),
            _instruction(instruction_by_name["control.return"]),
        )
    )
    all_core_bytecode = _all_core_bytecode(specification)
    functions = b"".join(
        (
            _record(records.FUNCTIONS_HEADER, function_count_u32=2),
            _record(
                records.FUNCTION_ROW,
                callable_type_ordinal_u16=1,
                bytecode_length_u32=len(bytecode),
                value_register_count_u16=1,
                ref_register_count_u16=1,
                function_register_count_u16=1,
                block_count_u32=1,
            ),
            _record(
                records.FUNCTION_ROW,
                callable_type_ordinal_u16=2,
                flags_u16=function_may_yield | function_has_call,
                bytecode_offset_u32=len(bytecode),
                bytecode_length_u32=len(all_core_bytecode),
                switch_target_entry_count_u32=1,
                local_byte_length_u16=64,
                value_register_count_u16=16,
                ref_register_count_u16=16,
                function_register_count_u16=16,
                local_ref_count_u32=1,
                local_function_count_u32=1,
                block_count_u32=1,
            ),
            _record(records.SWITCH_TARGET_ENTRY, target_word_offset_u32=0),
            bytecode,
            all_core_bytecode,
        )
    )
    constants = _record(records.CONSTANT_CELL, bits_u64=0x0123456789ABCDEF)
    globals_payload = b"".join(
        (
            _record(
                records.GLOBALS_HEADER,
                value_count_u32=2,
                immutable_value_count_u32=1,
                ref_count_u32=2,
                immutable_ref_count_u32=1,
                function_count_u32=2,
                immutable_function_count_u32=1,
            ),
            _record(
                records.GLOBAL_REF_DESCRIPTOR_ROW,
                ref_type_ordinal_u16=0,
                flags_u16=global_ref_nullable,
            ),
            _record(
                records.GLOBAL_REF_DESCRIPTOR_ROW,
                ref_type_ordinal_u16=0,
                flags_u16=global_ref_nullable,
            ),
            _record(
                records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
                callable_type_ordinal_u16=0,
                flags_u16=global_function_nullable,
            ),
            _record(
                records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
                callable_type_ordinal_u16=0,
                flags_u16=global_function_nullable,
            ),
        )
    )
    rodata = bytearray(
        b"".join(
            (
                _record(records.RODATA_HEADER, block_count_u32=1),
                _record(
                    records.RODATA_BLOCK_DESCRIPTOR,
                    byte_length_u64=5,
                    minimum_alignment_u32=64,
                ),
            )
        )
    )
    _align(rodata, 64)
    rodata.extend(b"\x01\x02\x03\x04\x05")
    presentation = b"".join(
        (
            _record(records.PRESENTATION_HEADER, entry_count_u32=2),
            _record(
                records.PRESENTATION_ENTRY_ROW,
                declaration_ordinal_u16=0,
                declaration_kind_u16=presentation_import,
                documentation_string_u16=string_ordinal["Fixture function."],
                authored_type_string_u16=0xFFFF,
                field_base_u32=0,
            ),
            _record(
                records.PRESENTATION_ENTRY_ROW,
                declaration_ordinal_u16=0,
                declaration_kind_u16=presentation_export,
                documentation_string_u16=string_ordinal["Fixture function."],
                authored_type_string_u16=string_ordinal[
                    "(i32, vm.ref<vm, buffer>, func.ref<() -> ()>) -> i64"
                ],
                field_base_u32=0,
            ),
            _record(
                records.PRESENTATION_FIELD_ROW,
                name_string_u16=string_ordinal["value"],
                authored_type_string_u16=string_ordinal["i32"],
            ),
            _record(
                records.PRESENTATION_FIELD_ROW,
                name_string_u16=string_ordinal["payload"],
                authored_type_string_u16=string_ordinal["vm.ref<vm, buffer>"],
            ),
            _record(
                records.PRESENTATION_FIELD_ROW,
                name_string_u16=string_ordinal["callback"],
                authored_type_string_u16=string_ordinal["func.ref<() -> ()>"],
            ),
            _record(
                records.PRESENTATION_FIELD_ROW,
                name_string_u16=string_ordinal["result"],
                authored_type_string_u16=string_ordinal["i64"],
            ),
        )
    )
    metadata_values = b"loom" + b"\x01" + (4).to_bytes(8, "little")
    metadata = bytearray(
        b"".join(
            (
                _record(
                    records.METADATA_HEADER,
                    module_entry_count_u32=1,
                    import_scope_count_u32=1,
                    export_scope_count_u32=1,
                    total_entry_count_u32=3,
                ),
                _record(
                    records.METADATA_SCOPE_ROW,
                    declaration_ordinal_u16=0,
                    entry_count_u16=1,
                    entry_base_u32=1,
                ),
                _record(
                    records.METADATA_SCOPE_ROW,
                    declaration_ordinal_u16=0,
                    entry_count_u16=1,
                    entry_base_u32=2,
                ),
                _record(
                    records.METADATA_ENTRY_ROW,
                    key_string_u16=string_ordinal["author"],
                    value_type_u16=metadata_utf8,
                ),
                _record(
                    records.METADATA_ENTRY_ROW,
                    key_string_u16=string_ordinal["feature"],
                    value_type_u16=metadata_bool,
                ),
                _record(
                    records.METADATA_ENTRY_ROW,
                    key_string_u16=string_ordinal["revision"],
                    value_type_u16=metadata_u64,
                ),
            )
        )
    )
    _align(metadata, 8)
    metadata.extend(
        b"".join(
            _record(records.METADATA_VALUE_OFFSET, byte_offset_u64=offset)
            for offset in (0, 4, 5, len(metadata_values))
        )
    )
    metadata.extend(metadata_values)

    payload_by_name = {
        "requirements": _record(
            records.REQUIREMENT_ROW,
            page_id_u16=0xF0,
            major_u16=0,
            required_minor_u16=0,
        ),
        "strings": strings,
        "ref_types": ref_types,
        "signatures": signatures,
        "callable_types": callable_types,
        "imports": imports,
        "exports": exports,
        "functions": functions,
        "constants": constants,
        "globals": globals_payload,
        "rodata": bytes(rodata),
        "presentation": presentation,
        "metadata": bytes(metadata),
    }
    section_entries = [
        (
            section.section_type,
            section.required_flags,
            64 if section.name == "rodata" else 8,
            payload_by_name[section.name],
        )
        for section in specification.module_format.sections
    ]
    section_entries.append((0xF001, 1, 8, b"extension"))

    image = bytearray(
        _record(
            records.IMAGE_HEADER,
            core_major_u16=specification.version.major,
            core_required_minor_u16=specification.version.minor,
            section_count_u16=len(section_entries),
        )
    )
    image.extend(
        b"".join(
            _record(
                records.SECTION_DIRECTORY_ROW,
                section_type_u16=section_type,
                section_flags_u16=flags,
                payload_alignment_u32=alignment,
                byte_length_u64=len(payload),
            )
            for section_type, flags, alignment, payload in section_entries
        )
    )
    for _, _, alignment, payload in section_entries:
        _align(image, alignment)
        image.extend(payload)
    return bytes(image)
