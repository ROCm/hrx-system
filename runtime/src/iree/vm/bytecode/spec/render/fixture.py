# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders canonical VM module fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from iree.vm.bytecode.spec import isa, module
from iree.vm.bytecode.spec.isa.core import rules as core_rules
from iree.vm.bytecode.spec.module import records
from iree.vm.bytecode.spec.specification import Specification

_FixtureInstruction = str | bytes


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


class _InstructionEncoder:
    """Encodes mnemonic-addressed instructions from authoritative layouts."""

    def __init__(self, specification: Specification):
        self._by_name = {
            instruction.mnemonic: instruction
            for instruction in specification.instructions
        }

    def __call__(self, mnemonic: str, **values: int | bytes) -> bytes:
        return _instruction(self._by_name[mnemonic], **values)

    def sequence(self, *instructions: _FixtureInstruction) -> bytes:
        """Encodes a declarative sequence using authoritative field layouts."""

        result = bytearray()
        for instruction in instructions:
            if isinstance(instruction, str):
                result.extend(self(instruction))
            else:
                result.extend(instruction)
        return bytes(result)


def _align(data: bytearray, alignment: int) -> None:
    data.extend(bytes((-len(data)) % alignment))


def _number(specification: Specification, table_name: str, value_name: str) -> int:
    tables = specification.module_format.numeric_tables + specification.selectors
    table = next(item for item in tables if item.name == table_name)
    return next(item.value for item in table.values if item.name == value_name)


class _FixtureSignature(NamedTuple):
    arguments: tuple[tuple[int, int], ...] = ()
    results: tuple[tuple[int, int], ...] = ()


def _signatures(
    signatures: Sequence[_FixtureSignature],
    ref_kind: int,
    function_kind: int,
) -> bytes:
    """Serializes source-ordered signatures with derived physical bank counts."""

    rows = []
    descriptors = []
    descriptor_base = 0
    for signature in signatures:
        arguments = signature.arguments
        results = signature.results
        rows.append(
            _record(
                records.SIGNATURE_ROW,
                descriptor_base_u32=descriptor_base,
                argument_value_count_u16=sum(
                    kind not in (ref_kind, function_kind) for kind, _ in arguments
                ),
                result_value_count_u16=sum(
                    kind not in (ref_kind, function_kind) for kind, _ in results
                ),
                argument_ref_count_u16=sum(kind == ref_kind for kind, _ in arguments),
                result_ref_count_u16=sum(kind == ref_kind for kind, _ in results),
                argument_function_count_u16=sum(
                    kind == function_kind for kind, _ in arguments
                ),
                result_function_count_u16=sum(
                    kind == function_kind for kind, _ in results
                ),
            )
        )
        fields = arguments + results
        descriptors.extend(
            _record(
                records.SIGNATURE_DESCRIPTOR_ROW,
                kind_u16=kind,
                type_ordinal_u16=type_ordinal,
            )
            for kind, type_ordinal in fields
        )
        descriptor_base += len(fields)
    return b"".join(
        (
            _record(
                records.SIGNATURES_HEADER,
                signature_count_u32=len(signatures),
            ),
            *rows,
            *descriptors,
        )
    )


class _FixtureFunction(NamedTuple):
    callable_type_ordinal: int
    bytecode: bytes
    flags: int = 0
    local_byte_length: int = 0
    value_register_count: int = 0
    ref_register_count: int = 0
    function_register_count: int = 0
    local_ref_count: int = 0
    local_function_count: int = 0
    block_count: int = 1
    switch_target_word_offsets: tuple[int, ...] = ()


def _functions(functions: Sequence[_FixtureFunction]) -> bytes:
    """Serializes functions while deriving canonical bytecode offsets."""

    rows = []
    bytecode_offset = 0
    switch_target_base = 0
    for function in functions:
        rows.append(
            _record(
                records.FUNCTION_ROW,
                callable_type_ordinal_u16=function.callable_type_ordinal,
                flags_u16=function.flags,
                bytecode_offset_u32=bytecode_offset,
                bytecode_length_u32=len(function.bytecode),
                switch_target_base_u32=switch_target_base,
                switch_target_entry_count_u32=len(function.switch_target_word_offsets),
                local_byte_length_u16=function.local_byte_length,
                value_register_count_u16=function.value_register_count,
                ref_register_count_u16=function.ref_register_count,
                function_register_count_u16=function.function_register_count,
                local_ref_count_u32=function.local_ref_count,
                local_function_count_u32=function.local_function_count,
                block_count_u32=function.block_count,
            )
        )
        bytecode_offset += len(function.bytecode)
        switch_target_base += len(function.switch_target_word_offsets)
    return b"".join(
        (
            _record(
                records.FUNCTIONS_HEADER,
                function_count_u32=len(functions),
                maximum_block_count_u32=max(
                    function.block_count for function in functions
                ),
            ),
            *rows,
            *(
                _record(
                    records.SWITCH_TARGET_ENTRY,
                    target_word_offset_u32=target_word_offset,
                )
                for function in functions
                for target_word_offset in function.switch_target_word_offsets
            ),
            *(function.bytecode for function in functions),
        )
    )


def _target_word_offset(
    record_offset: int, record_length: int, target_offset: int
) -> int:
    """Returns one four-byte-word displacement from a record end."""

    byte_offset = target_offset - record_offset - record_length
    if byte_offset % 4:
        raise ValueError("control target is not four-byte aligned")
    return byte_offset // 4


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


class _FixtureRefTypeGroup(NamedTuple):
    namespace_string_ordinal: int
    type_name_string_ordinals: tuple[int, ...]


def _ref_types(groups: Sequence[_FixtureRefTypeGroup]) -> bytes:
    return b"".join(
        (
            _record(records.REF_TYPES_HEADER, group_count_u32=len(groups)),
            *(
                _record(
                    records.REF_TYPE_GROUP_ROW,
                    namespace_string_u16=group.namespace_string_ordinal,
                    entry_count_u32=len(group.type_name_string_ordinals),
                )
                for group in groups
            ),
            *(
                _record(
                    records.REF_TYPE_ENTRY_ROW,
                    type_name_string_u16=type_name_string_ordinal,
                )
                for group in groups
                for type_name_string_ordinal in group.type_name_string_ordinals
            ),
        )
    )


class _FixtureCallableType(NamedTuple):
    signature_ordinal: int
    flags: int = 0
    nesting_depth: int = 0


def _callable_types(callable_types: Sequence[_FixtureCallableType]) -> bytes:
    return b"".join(
        (
            _record(
                records.CALLABLE_TYPES_HEADER,
                callable_type_count_u32=len(callable_types),
            ),
            *(
                _record(
                    records.CALLABLE_TYPE_ROW,
                    signature_ordinal_u16=callable_type.signature_ordinal,
                    flags_u16=callable_type.flags,
                    nesting_depth_u16=callable_type.nesting_depth,
                )
                for callable_type in callable_types
            ),
        )
    )


class _FixtureImport(NamedTuple):
    symbol_name_string_ordinal: int
    callable_type_ordinal: int
    flags: int = 0


class _FixtureImportGroup(NamedTuple):
    module_name_string_ordinal: int
    imports: tuple[_FixtureImport, ...]


def _imports(groups: Sequence[_FixtureImportGroup]) -> bytes:
    return b"".join(
        (
            _record(records.IMPORTS_HEADER, group_count_u32=len(groups)),
            *(
                _record(
                    records.IMPORT_GROUP_ROW,
                    module_name_string_u16=group.module_name_string_ordinal,
                    entry_count_u32=len(group.imports),
                )
                for group in groups
            ),
            *(
                _record(
                    records.IMPORT_ENTRY_ROW,
                    symbol_name_string_u16=import_.symbol_name_string_ordinal,
                    callable_type_ordinal_u16=import_.callable_type_ordinal,
                    flags_u16=import_.flags,
                )
                for group in groups
                for import_ in group.imports
            ),
        )
    )


class _FixtureExport(NamedTuple):
    name_string_ordinal: int
    callable_type_ordinal: int
    function_ordinal: int


def _exports(exports: Sequence[_FixtureExport]) -> bytes:
    return b"".join(
        (
            _record(records.EXPORTS_HEADER, export_count_u32=len(exports)),
            *(
                _record(
                    records.EXPORT_ROW,
                    name_string_u16=export.name_string_ordinal,
                    callable_type_ordinal_u16=export.callable_type_ordinal,
                    function_ordinal_u16=export.function_ordinal,
                )
                for export in exports
            ),
        )
    )


class _FixtureGlobalRef(NamedTuple):
    ref_type_ordinal: int
    flags: int = 0


class _FixtureGlobalFunction(NamedTuple):
    callable_type_ordinal: int
    flags: int = 0


def _globals(
    value_count: int,
    immutable_value_count: int,
    refs: Sequence[_FixtureGlobalRef],
    immutable_ref_count: int,
    functions: Sequence[_FixtureGlobalFunction],
    immutable_function_count: int,
) -> bytes:
    return b"".join(
        (
            _record(
                records.GLOBALS_HEADER,
                value_count_u32=value_count,
                immutable_value_count_u32=immutable_value_count,
                ref_count_u32=len(refs),
                immutable_ref_count_u32=immutable_ref_count,
                function_count_u32=len(functions),
                immutable_function_count_u32=immutable_function_count,
            ),
            *(
                _record(
                    records.GLOBAL_REF_DESCRIPTOR_ROW,
                    ref_type_ordinal_u16=ref.ref_type_ordinal,
                    flags_u16=ref.flags,
                )
                for ref in refs
            ),
            *(
                _record(
                    records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
                    callable_type_ordinal_u16=function.callable_type_ordinal,
                    flags_u16=function.flags,
                )
                for function in functions
            ),
        )
    )


class _FixtureRodataBlock(NamedTuple):
    data: bytes
    minimum_alignment: int


def _rodata(blocks: Sequence[_FixtureRodataBlock]) -> bytes:
    payload = bytearray(
        b"".join(
            (
                _record(records.RODATA_HEADER, block_count_u32=len(blocks)),
                *(
                    _record(
                        records.RODATA_BLOCK_DESCRIPTOR,
                        byte_length_u64=len(block.data),
                        minimum_alignment_u32=block.minimum_alignment,
                    )
                    for block in blocks
                ),
            )
        )
    )
    for block in blocks:
        _align(payload, block.minimum_alignment)
        payload.extend(block.data)
    return bytes(payload)


class _FixturePresentationField(NamedTuple):
    name_string_ordinal: int
    authored_type_string_ordinal: int


class _FixturePresentationEntry(NamedTuple):
    declaration_ordinal: int
    declaration_kind: int
    documentation_string_ordinal: int = 0xFFFF
    authored_type_string_ordinal: int = 0xFFFF
    fields: tuple[_FixturePresentationField, ...] = ()


def _presentation(entries: Sequence[_FixturePresentationEntry]) -> bytes:
    field_base = 0
    rows = []
    for entry in entries:
        rows.append(
            _record(
                records.PRESENTATION_ENTRY_ROW,
                declaration_ordinal_u16=entry.declaration_ordinal,
                declaration_kind_u16=entry.declaration_kind,
                documentation_string_u16=entry.documentation_string_ordinal,
                authored_type_string_u16=entry.authored_type_string_ordinal,
                field_base_u32=field_base,
            )
        )
        field_base += len(entry.fields)
    return b"".join(
        (
            _record(records.PRESENTATION_HEADER, entry_count_u32=len(entries)),
            *rows,
            *(
                _record(
                    records.PRESENTATION_FIELD_ROW,
                    name_string_u16=field.name_string_ordinal,
                    authored_type_string_u16=field.authored_type_string_ordinal,
                )
                for entry in entries
                for field in entry.fields
            ),
        )
    )


class _FixtureMetadataEntry(NamedTuple):
    key_string_ordinal: int
    value_type: int
    value: bytes


class _FixtureMetadataScope(NamedTuple):
    declaration_ordinal: int
    entries: tuple[_FixtureMetadataEntry, ...]


def _metadata(
    module_entries: Sequence[_FixtureMetadataEntry],
    import_scopes: Sequence[_FixtureMetadataScope] = (),
    export_scopes: Sequence[_FixtureMetadataScope] = (),
) -> bytes:
    entries = (
        *module_entries,
        *(entry for scope in import_scopes for entry in scope.entries),
        *(entry for scope in export_scopes for entry in scope.entries),
    )
    entry_base = len(module_entries)
    scope_rows = []
    for scope in (*import_scopes, *export_scopes):
        scope_rows.append(
            _record(
                records.METADATA_SCOPE_ROW,
                declaration_ordinal_u16=scope.declaration_ordinal,
                entry_count_u16=len(scope.entries),
                entry_base_u32=entry_base,
            )
        )
        entry_base += len(scope.entries)
    payload = bytearray(
        b"".join(
            (
                _record(
                    records.METADATA_HEADER,
                    module_entry_count_u32=len(module_entries),
                    import_scope_count_u32=len(import_scopes),
                    export_scope_count_u32=len(export_scopes),
                    total_entry_count_u32=len(entries),
                ),
                *scope_rows,
                *(
                    _record(
                        records.METADATA_ENTRY_ROW,
                        key_string_u16=entry.key_string_ordinal,
                        value_type_u16=entry.value_type,
                    )
                    for entry in entries
                ),
            )
        )
    )
    _align(payload, 8)
    byte_offset = 0
    for entry in entries:
        payload.extend(
            _record(records.METADATA_VALUE_OFFSET, byte_offset_u64=byte_offset)
        )
        byte_offset += len(entry.value)
    payload.extend(_record(records.METADATA_VALUE_OFFSET, byte_offset_u64=byte_offset))
    payload.extend(b"".join(entry.value for entry in entries))
    return bytes(payload)


def _render_image(
    specification: Specification,
    payload_by_name: Mapping[str, bytes],
    alignment_by_name: Mapping[str, int],
    additional_sections: Sequence[tuple[int, int, int, bytes]] = (),
) -> bytes:
    """Wraps selected section payloads in one canonical image envelope."""

    section_entries = [
        (
            section.section_type,
            section.required_flags,
            alignment_by_name.get(section.name, 8),
            payload_by_name[section.name],
        )
        for section in specification.module_format.sections
        if section.name in payload_by_name
    ]
    section_entries.extend(additional_sections)
    section_entries.sort(key=lambda item: item[0])

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

    ref_types = _ref_types(
        (_FixtureRefTypeGroup(string_ordinal["vm"], (string_ordinal["buffer"],)),)
    )
    broad_arguments = (
        *((signature_i64, 0),) * 17,
        *((signature_ref, 0),) * 17,
        *((signature_function, 0),) * 17,
    )
    signatures = _signatures(
        (
            _FixtureSignature(),
            _FixtureSignature(
                arguments=(
                    (signature_i32, 0),
                    (signature_ref, 0),
                    (signature_function, 0),
                ),
                results=((signature_i64, 0),),
            ),
            _FixtureSignature(
                arguments=broad_arguments,
                results=broad_arguments,
            ),
        ),
        signature_ref,
        signature_function,
    )
    callable_types = _callable_types(
        (
            _FixtureCallableType(0),
            _FixtureCallableType(1, nesting_depth=1),
            _FixtureCallableType(2, callable_may_yield, 1),
        )
    )
    imports = _imports(
        (
            _FixtureImportGroup(
                string_ordinal["dep"],
                (_FixtureImport(string_ordinal["entry"], 0, import_optional),),
            ),
        )
    )
    exports = _exports((_FixtureExport(string_ordinal["run"], 1, 0),))
    op = _InstructionEncoder(specification)
    bytecode = op.sequence("control.block", "control.return")
    all_core_bytecode = _all_core_bytecode(specification)
    functions = _functions(
        (
            _FixtureFunction(
                callable_type_ordinal=1,
                bytecode=bytecode,
                value_register_count=1,
                ref_register_count=1,
                function_register_count=1,
            ),
            _FixtureFunction(
                callable_type_ordinal=2,
                bytecode=all_core_bytecode,
                flags=function_may_yield | function_has_call,
                local_byte_length=64,
                value_register_count=16,
                ref_register_count=16,
                function_register_count=16,
                local_ref_count=1,
                local_function_count=1,
                switch_target_word_offsets=(0,),
            ),
        )
    )
    constants = _record(records.CONSTANT_CELL, bits_u64=0x0123456789ABCDEF)
    globals_payload = _globals(
        value_count=2,
        immutable_value_count=1,
        refs=(
            _FixtureGlobalRef(0, global_ref_nullable),
            _FixtureGlobalRef(0, global_ref_nullable),
        ),
        immutable_ref_count=1,
        functions=(
            _FixtureGlobalFunction(0, global_function_nullable),
            _FixtureGlobalFunction(0, global_function_nullable),
        ),
        immutable_function_count=1,
    )
    rodata = _rodata((_FixtureRodataBlock(b"\x01\x02\x03\x04\x05", 64),))
    presentation = _presentation(
        (
            _FixturePresentationEntry(
                0,
                presentation_import,
                documentation_string_ordinal=string_ordinal["Fixture function."],
            ),
            _FixturePresentationEntry(
                0,
                presentation_export,
                string_ordinal["Fixture function."],
                string_ordinal["(i32, vm.ref<vm, buffer>, func.ref<() -> ()>) -> i64"],
                (
                    _FixturePresentationField(
                        string_ordinal["value"], string_ordinal["i32"]
                    ),
                    _FixturePresentationField(
                        string_ordinal["payload"],
                        string_ordinal["vm.ref<vm, buffer>"],
                    ),
                    _FixturePresentationField(
                        string_ordinal["callback"],
                        string_ordinal["func.ref<() -> ()>"],
                    ),
                    _FixturePresentationField(
                        string_ordinal["result"], string_ordinal["i64"]
                    ),
                ),
            ),
        )
    )
    metadata = _metadata(
        (_FixtureMetadataEntry(string_ordinal["author"], metadata_utf8, b"loom"),),
        import_scopes=(
            _FixtureMetadataScope(
                0,
                (
                    _FixtureMetadataEntry(
                        string_ordinal["feature"], metadata_bool, b"\x01"
                    ),
                ),
            ),
        ),
        export_scopes=(
            _FixtureMetadataScope(
                0,
                (
                    _FixtureMetadataEntry(
                        string_ordinal["revision"],
                        metadata_u64,
                        (4).to_bytes(8, "little"),
                    ),
                ),
            ),
        ),
    )

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
        "rodata": rodata,
        "presentation": presentation,
        "metadata": metadata,
    }
    return _render_image(
        specification,
        payload_by_name,
        {"rodata": 64},
        ((0xF001, 1, 8, b"extension"),),
    )


def render_launch_config_module_fixture(specification: Specification) -> bytes:
    """Returns a minimal executable module with one scalar launch function."""

    strings, string_ordinal = _strings(("initialize", "launch_config"))
    signature_i32 = _number(specification, "signature_kind", "i32")
    signature_ref = _number(specification, "signature_kind", "ref")
    signature_function = _number(specification, "signature_kind", "function")
    signatures = _signatures(
        (
            _FixtureSignature(),
            _FixtureSignature(((signature_i32, 0),), ((signature_i32, 0),)),
        ),
        signature_ref,
        signature_function,
    )
    callable_types = _callable_types(
        (
            _FixtureCallableType(0),
            _FixtureCallableType(1),
        )
    )
    exports = _exports(
        (
            _FixtureExport(string_ordinal["initialize"], 0, 0),
            _FixtureExport(string_ordinal["launch_config"], 1, 1),
        )
    )

    op = _InstructionEncoder(specification)
    initialize_bytecode = op.sequence("control.block", "control.return")
    launch_bytecode = op.sequence(
        "control.block",
        op("integer.neg.i32", destination_v8=0, source_v8=0),
        op("integer.neg.i32", destination_v8=0, source_v8=0),
        op("integer.neg.i32", destination_v8=0, source_v8=0),
        "control.return",
    )
    functions = _functions(
        (
            _FixtureFunction(0, initialize_bytecode),
            _FixtureFunction(1, launch_bytecode, value_register_count=1),
        )
    )
    return _render_image(
        specification,
        {
            "strings": strings,
            "signatures": signatures,
            "callable_types": callable_types,
            "exports": exports,
            "functions": functions,
        },
        {},
    )


def render_core_execution_module_fixture(specification: Specification) -> bytes:
    """Returns a Core-only module covering the execution state machine."""

    export_names = (
        "call_import",
        "call_indirect",
        "call_local",
        "call_optional",
        "call_overflow_refs",
        "call_overflow_values",
        "call_ref_borrow",
        "call_ref_drop",
        "call_ref_move",
        "control_flow",
        "echo_ref",
        "fail",
        "global_state",
        "initialize",
        "leaf",
        "memory_roundtrip",
        "run",
        "switch",
        "yield_once",
        "yield_ref",
    )
    string_values = (
        *export_names,
        "vm",
        "buffer",
        "Initializes the immutable process bias.",
        "Adds the process bias and returns module read-only data.",
        "() -> ()",
        "(i32) -> (i32, vm.ref<vm, buffer>)",
        "value",
        "i32",
        "sum",
        "payload",
        "vm.ref<vm, buffer>",
        "author",
        "purpose",
        "invocation.test",
        "add",
        "missing",
    )
    strings, string_ordinal = _strings(string_values)

    signature_i32 = _number(specification, "signature_kind", "i32")
    signature_i64 = _number(specification, "signature_kind", "i64")
    signature_ref = _number(specification, "signature_kind", "ref")
    signature_function = _number(specification, "signature_kind", "function")
    callable_may_yield = _number(specification, "callable_type_flag", "may_yield")
    function_may_yield = _number(specification, "function_flag", "may_yield")
    function_has_call = _number(specification, "function_flag", "has_call")
    call_target_local = _number(specification, "control.call.target", "local")
    call_target_required_import = _number(
        specification, "control.call.target", "required_import"
    )
    call_target_optional_import = _number(
        specification, "control.call.target", "optional_import"
    )
    import_optional = _number(specification, "import_flag", "optional")
    global_ref_nullable = _number(specification, "global_ref_flag", "nullable")
    status_aborted = _number(specification, "control.status", "aborted")
    memory_i64_x1 = _number(specification, "memory.format", "i64.x1")
    presentation_export = _number(
        specification, "presentation_declaration_kind", "export"
    )
    metadata_utf8 = _number(specification, "metadata_value_type", "utf8")

    ref_types = _ref_types(
        (_FixtureRefTypeGroup(string_ordinal["vm"], (string_ordinal["buffer"],)),)
    )
    i32_field = (signature_i32, 0)
    ref_field = (signature_ref, 0)
    signature_definitions = (
        _FixtureSignature(),
        _FixtureSignature((i32_field,), (i32_field,)),
        _FixtureSignature((i32_field,), (i32_field, ref_field)),
        _FixtureSignature((ref_field,), ()),
        _FixtureSignature((ref_field,), (ref_field,)),
        _FixtureSignature((i32_field, i32_field), (i32_field,)),
        _FixtureSignature((i32_field,) * 17, (i32_field,) * 17),
        _FixtureSignature((ref_field,) * 17, (ref_field,) * 17),
        _FixtureSignature((ref_field,), ((signature_i64, 0), i32_field)),
    )
    signatures = _signatures(signature_definitions, signature_ref, signature_function)

    callable_types = _callable_types(
        (
            _FixtureCallableType(0),
            _FixtureCallableType(1),
            _FixtureCallableType(1, callable_may_yield),
            _FixtureCallableType(2),
            _FixtureCallableType(3),
            _FixtureCallableType(4),
            _FixtureCallableType(4, callable_may_yield),
            _FixtureCallableType(8),
            _FixtureCallableType(5),
            _FixtureCallableType(6),
            _FixtureCallableType(7),
        )
    )
    imports = _imports(
        (
            _FixtureImportGroup(
                string_ordinal["invocation.test"],
                (
                    _FixtureImport(string_ordinal["add"], 8),
                    _FixtureImport(string_ordinal["missing"], 8, import_optional),
                ),
            ),
        )
    )

    function_ordinal = {
        "initialize": 0,
        "run": 1,
        "leaf": 2,
        "add": 3,
        "call_local": 4,
        "call_indirect": 5,
        "echo_ref": 6,
        "call_ref_borrow": 7,
        "call_ref_move": 8,
        "drop_ref": 9,
        "call_ref_drop": 10,
        "yield_once": 11,
        "yield_ref": 12,
        "fail": 13,
        "call_import": 14,
        "call_optional": 15,
        "overflow_values": 16,
        "call_overflow_values": 17,
        "overflow_refs": 18,
        "call_overflow_refs": 19,
        "control_flow": 20,
        "switch": 21,
        "global_state": 22,
        "memory_roundtrip": 23,
    }
    export_declarations = (
        ("call_import", 8),
        ("call_indirect", 8),
        ("call_local", 8),
        ("call_optional", 8),
        ("call_overflow_refs", 10),
        ("call_overflow_values", 9),
        ("call_ref_borrow", 5),
        ("call_ref_drop", 4),
        ("call_ref_move", 5),
        ("control_flow", 1),
        ("echo_ref", 5),
        ("fail", 1),
        ("global_state", 1),
        ("initialize", 0),
        ("leaf", 1),
        ("memory_roundtrip", 7),
        ("run", 3),
        ("switch", 1),
        ("yield_once", 2),
        ("yield_ref", 6),
    )
    exports = _exports(
        tuple(
            _FixtureExport(
                string_ordinal[name], callable_type_ordinal, function_ordinal[name]
            )
            for name, callable_type_ordinal in export_declarations
        )
    )
    export_ordinal = {
        name: ordinal for ordinal, (name, _) in enumerate(export_declarations)
    }

    op = _InstructionEncoder(specification)
    initialize_bytecode = op.sequence(
        "control.block",
        op("constant.s16", destination_v8=0, immediate_i16=7),
        op("global.value.immutable.store", source_v8=0, global_u16=0),
        op("constant.s16", destination_v8=0, immediate_i16=8),
        op(
            "buffer.allocate",
            destination_r8=0,
            length_v8=0,
            minimum_alignment_log2_u8=0,
        ),
        op("global.ref.immutable.store.move", source_r8=0, global_u16=0),
        op(
            "func.address",
            destination_f8=0,
            target_kind_u8=call_target_local,
            target_ordinal_u16=function_ordinal["leaf"],
            callable_type_ordinal_u16=1,
        ),
        op("global.func.immutable.store", source_f8=0, global_u16=0),
        op("global.func.mutable.store", source_f8=0, global_u16=1),
        "control.return",
    )
    run_bytecode = op.sequence(
        "control.block",
        op("global.value.immutable.load", destination_v8=1, global_u16=0),
        op("integer.add.i32", destination_v8=0, left_v8=0, right_v8=1),
        op("buffer.rodata.load", destination_r8=0, rodata_u16=0),
        "control.return",
    )
    leaf_bytecode = op.sequence(
        "control.block",
        op("integer.neg.i32", destination_v8=0, source_v8=0),
        op("integer.neg.i32", destination_v8=0, source_v8=0),
        op("integer.neg.i32", destination_v8=0, source_v8=0),
        "control.return",
    )
    add_bytecode = op.sequence(
        "control.block",
        op("integer.add.i32", destination_v8=0, left_v8=0, right_v8=1),
        "control.return",
    )
    call_local_bytecode = op.sequence(
        "control.block",
        op(
            "control.call",
            target_kind_u8=call_target_local,
            target_ordinal_u16=function_ordinal["add"],
        ),
        "control.return",
    )
    call_indirect_bytecode = op.sequence(
        "control.block",
        op(
            "func.address",
            destination_f8=0,
            target_kind_u8=call_target_local,
            target_ordinal_u16=function_ordinal["add"],
            callable_type_ordinal_u16=8,
        ),
        op("control.call.indirect", target_f8=0, callable_type_ordinal_u16=8),
        "control.return",
    )
    call_import_bytecode = op.sequence(
        "control.block",
        op(
            "control.call",
            target_kind_u8=call_target_required_import,
            target_ordinal_u16=0,
        ),
        "control.return",
    )
    call_optional_bytecode = op.sequence(
        "control.block",
        op(
            "control.call",
            target_kind_u8=call_target_optional_import,
            target_ordinal_u16=1,
        ),
        "control.return",
    )
    control_flow_bytecode = op.sequence(
        # Entry selects one of two paths with the compact conditional forms.
        "control.block",
        op(
            "control.branch.if.s16",
            condition_v8=0,
            target_word_offset_s16=_target_word_offset(4, 4, 12),
        ),
        op(
            "control.branch.unless.s16",
            condition_v8=0,
            target_word_offset_s16=_target_word_offset(8, 4, 24),
        ),
        # Both paths use one unconditional encoding before their checks.
        "control.block",
        op("control.branch.s32", target_word_offset_s32=_target_word_offset(16, 8, 32)),
        "control.block",
        op("control.branch.s16", target_word_offset_s16=_target_word_offset(28, 4, 60)),
        # Nonzero path: unless.s32 must branch over the bad result.
        "control.block",
        op("constant.s16", destination_v8=0, immediate_i16=10),
        op("constant.zero", destination_v8=1),
        op(
            "control.branch.unless.s32",
            condition_v8=1,
            target_word_offset_s32=_target_word_offset(44, 8, 92),
        ),
        op("constant.s16", destination_v8=0, immediate_i16=-1),
        op("control.branch.s16", target_word_offset_s16=_target_word_offset(56, 4, 92)),
        # Zero path: if.s32 must branch over the bad result.
        "control.block",
        op("constant.s16", destination_v8=0, immediate_i16=20),
        op("constant.s16", destination_v8=1, immediate_i16=1),
        op(
            "control.branch.if.s32",
            condition_v8=1,
            target_word_offset_s32=_target_word_offset(72, 8, 92),
        ),
        op("constant.s16", destination_v8=0, immediate_i16=-1),
        op("control.branch.s32", target_word_offset_s32=_target_word_offset(84, 8, 92)),
        "control.block",
        "control.return",
    )
    switch_bytecode = op.sequence(
        "control.block",
        op("control.switch", selector_v8=0, target_count_u16=3, target_base_u32=0),
        "control.block",
        op("constant.s16", destination_v8=0, immediate_i16=99),
        "control.return",
        "control.block",
        op("constant.s16", destination_v8=0, immediate_i16=10),
        "control.return",
        "control.block",
        op("constant.s16", destination_v8=0, immediate_i16=12),
        "control.return",
    )
    overflow_values_bytecode = op.sequence(
        "control.block",
        op("value.abi.argument.load", destination_v8=16, slot_u16=0),
        op("value.abi.result.store", source_v8=16, slot_u16=0),
        "control.return",
    )
    call_overflow_values_bytecode = op.sequence(
        "control.block",
        op("value.abi.argument.load", destination_v8=16, slot_u16=0),
        op("stack.store", base_u16=0, source_v8=16, format_u8=memory_i64_x1),
        op(
            "control.call",
            target_kind_u8=call_target_local,
            target_ordinal_u16=function_ordinal["overflow_values"],
        ),
        op("stack.load", destination_v8=16, base_u16=8, format_u8=memory_i64_x1),
        op("value.abi.result.store", source_v8=16, slot_u16=0),
        "control.return",
    )
    overflow_refs_bytecode = op.sequence(
        "control.block",
        op("ref.abi.argument.load.move", destination_r8=16, slot_u16=0),
        op("ref.abi.result.store.move", source_r8=16, slot_u16=0),
        "control.return",
    )
    call_overflow_refs_bytecode = op.sequence(
        "control.block",
        op("ref.abi.argument.load.move", destination_r8=16, slot_u16=0),
        op("ref.stack.store.move", source_r8=16, slot_u16=0),
        op(
            "control.call",
            target_kind_u8=call_target_local,
            target_ordinal_u16=function_ordinal["overflow_refs"],
            direct_ref_move_mask_u16=65535,
        ),
        op("ref.stack.load.move", destination_r8=16, slot_u16=1),
        op("ref.abi.result.store.move", source_r8=16, slot_u16=0),
        "control.return",
    )
    echo_ref_bytecode = op.sequence("control.block", "control.return")
    call_ref_borrow_bytecode = op.sequence(
        "control.block",
        op(
            "control.call",
            target_kind_u8=call_target_local,
            target_ordinal_u16=function_ordinal["echo_ref"],
            direct_ref_move_mask_u16=0,
        ),
        "control.return",
    )
    call_ref_move_bytecode = op.sequence(
        "control.block",
        op(
            "control.call",
            target_kind_u8=call_target_local,
            target_ordinal_u16=function_ordinal["echo_ref"],
            direct_ref_move_mask_u16=1,
        ),
        "control.return",
    )
    drop_ref_bytecode = op.sequence("control.block", "control.return")
    call_ref_drop_bytecode = op.sequence(
        "control.block",
        op(
            "control.call",
            target_kind_u8=call_target_local,
            target_ordinal_u16=function_ordinal["drop_ref"],
            direct_ref_move_mask_u16=1,
        ),
        "control.return",
    )
    yield_once_bytecode = op.sequence(
        "control.block",
        op("control.yield.s32", target_word_offset_s32=0),
        "control.block",
        op("constant.s16", destination_v8=1, immediate_i16=1),
        op("integer.add.i32", destination_v8=0, left_v8=0, right_v8=1),
        "control.return",
    )
    yield_ref_bytecode = op.sequence(
        "control.block",
        op("control.yield.s32", target_word_offset_s32=0),
        "control.block",
        "control.return",
    )
    fail_bytecode = op.sequence(
        "control.block",
        op("buffer.rodata.load", destination_r8=0, rodata_u16=0),
        op("control.fail", status_u8=status_aborted, message_r8_nullable=0),
    )
    global_state_bytecode = op.sequence(
        "control.block",
        op("global.value.mutable.store", source_v8=0, global_u16=1),
        op("global.value.mutable.load", destination_v8=0, global_u16=1),
        op("global.ref.immutable.load.borrow", destination_r8=0, global_u16=0),
        op("buffer.length", destination_v8=1, buffer_r8=0),
        op("integer.add.i32", destination_v8=0, left_v8=0, right_v8=1),
        op("global.ref.mutable.store.move", source_r8=0, global_u16=1),
        op("global.ref.mutable.load.retain", destination_r8=0, global_u16=1),
        op("ref.discard", source_r8=0),
        op("global.func.immutable.load", destination_f8=0, global_u16=0),
        op("global.func.mutable.store", source_f8=0, global_u16=1),
        op("global.func.mutable.load", destination_f8=0, global_u16=1),
        op("control.call.indirect", target_f8=0, callable_type_ordinal_u16=1),
        "control.return",
    )
    memory_roundtrip_bytecode = op.sequence(
        "control.block",
        op("constant.zero", destination_v8=0),
        op("constant.s16", destination_v8=1, immediate_i16=8),
        op("constant.s16", destination_v8=2, immediate_i16=8721),
        op(
            "buffer.fill",
            buffer_r8=0,
            offset_v8=0,
            length_v8=1,
            pattern_v8=2,
            pattern_width_u8=2,
        ),
        op(
            "buffer.load",
            destination_v8=3,
            buffer_r8=0,
            base_v8=0,
            index_v8=0,
            scale_u8=1,
            format_u8=memory_i64_x1,
        ),
        op(
            "buffer.allocate",
            destination_r8=1,
            length_v8=1,
            minimum_alignment_log2_u8=0,
        ),
        op(
            "buffer.copy",
            target_r8=1,
            target_offset_v8=0,
            source_r8=0,
            source_offset_v8=0,
            length_v8=1,
        ),
        op(
            "buffer.compare",
            destination_v8=4,
            left_r8=0,
            left_offset_v8=0,
            right_r8=1,
            right_offset_v8=0,
            length_v8=1,
        ),
        op(
            "stack.copy.from.buffer",
            target_u16=0,
            buffer_r8=1,
            source_offset_v8=0,
            length_u16=8,
        ),
        op("stack.copy", target_u16=8, source_u16=0, length_u16=8),
        op("stack.compare", destination_v8=4, left_u16=0, right_u16=8, length_u16=8),
        op("stack.load", destination_v8=3, base_u16=8, format_u8=memory_i64_x1),
        op("stack.store", base_u16=0, source_v8=3, format_u8=memory_i64_x1),
        op(
            "stack.copy.to.buffer",
            buffer_r8=1,
            target_offset_v8=0,
            source_u16=0,
            length_u16=8,
        ),
        op(
            "buffer.store",
            buffer_r8=1,
            base_v8=0,
            index_v8=0,
            scale_u8=1,
            source_v8=3,
            format_u8=memory_i64_x1,
        ),
        op("value.copy", destination_v8=0, source_v8=3),
        op("value.copy", destination_v8=1, source_v8=4),
        "control.return",
    )
    functions = _functions(
        (
            _FixtureFunction(
                0,
                initialize_bytecode,
                value_register_count=1,
                ref_register_count=1,
                function_register_count=1,
            ),
            _FixtureFunction(
                3, run_bytecode, value_register_count=2, ref_register_count=1
            ),
            _FixtureFunction(1, leaf_bytecode, value_register_count=1),
            _FixtureFunction(8, add_bytecode, value_register_count=2),
            _FixtureFunction(
                8,
                call_local_bytecode,
                flags=function_has_call,
                value_register_count=2,
            ),
            _FixtureFunction(
                8,
                call_indirect_bytecode,
                flags=function_has_call,
                value_register_count=2,
                function_register_count=1,
            ),
            _FixtureFunction(5, echo_ref_bytecode, ref_register_count=1),
            _FixtureFunction(
                5,
                call_ref_borrow_bytecode,
                flags=function_has_call,
                ref_register_count=1,
            ),
            _FixtureFunction(
                5,
                call_ref_move_bytecode,
                flags=function_has_call,
                ref_register_count=1,
            ),
            _FixtureFunction(4, drop_ref_bytecode, ref_register_count=1),
            _FixtureFunction(
                4,
                call_ref_drop_bytecode,
                flags=function_has_call,
                ref_register_count=1,
            ),
            _FixtureFunction(
                2,
                yield_once_bytecode,
                flags=function_may_yield,
                value_register_count=2,
                block_count=2,
            ),
            _FixtureFunction(
                6,
                yield_ref_bytecode,
                flags=function_may_yield,
                ref_register_count=1,
                block_count=2,
            ),
            _FixtureFunction(
                1, fail_bytecode, value_register_count=1, ref_register_count=1
            ),
            _FixtureFunction(
                8,
                call_import_bytecode,
                flags=function_has_call,
                value_register_count=2,
            ),
            _FixtureFunction(
                8,
                call_optional_bytecode,
                flags=function_has_call,
                value_register_count=2,
            ),
            _FixtureFunction(9, overflow_values_bytecode, value_register_count=17),
            _FixtureFunction(
                9,
                call_overflow_values_bytecode,
                flags=function_has_call,
                local_byte_length=16,
                value_register_count=17,
            ),
            _FixtureFunction(10, overflow_refs_bytecode, ref_register_count=17),
            _FixtureFunction(
                10,
                call_overflow_refs_bytecode,
                flags=function_has_call,
                ref_register_count=17,
                local_ref_count=2,
            ),
            _FixtureFunction(
                1,
                control_flow_bytecode,
                value_register_count=2,
                block_count=6,
            ),
            _FixtureFunction(
                1,
                switch_bytecode,
                value_register_count=1,
                block_count=4,
                switch_target_word_offsets=(6, 3, 9),
            ),
            _FixtureFunction(
                1,
                global_state_bytecode,
                flags=function_has_call,
                value_register_count=2,
                ref_register_count=1,
                function_register_count=1,
            ),
            _FixtureFunction(
                7,
                memory_roundtrip_bytecode,
                local_byte_length=16,
                value_register_count=5,
                ref_register_count=2,
            ),
        )
    )
    globals_payload = _globals(
        value_count=2,
        immutable_value_count=1,
        refs=(
            _FixtureGlobalRef(0),
            _FixtureGlobalRef(0, global_ref_nullable),
        ),
        immutable_ref_count=1,
        functions=(
            _FixtureGlobalFunction(1),
            _FixtureGlobalFunction(1),
        ),
        immutable_function_count=1,
    )
    rodata = _rodata((_FixtureRodataBlock(b"loom-vm-v1", 8),))

    presentation = _presentation(
        (
            _FixturePresentationEntry(
                export_ordinal["initialize"],
                presentation_export,
                string_ordinal["Initializes the immutable process bias."],
                string_ordinal["() -> ()"],
            ),
            _FixturePresentationEntry(
                export_ordinal["run"],
                presentation_export,
                string_ordinal[
                    "Adds the process bias and returns module read-only data."
                ],
                string_ordinal["(i32) -> (i32, vm.ref<vm, buffer>)"],
                (
                    _FixturePresentationField(
                        string_ordinal["value"], string_ordinal["i32"]
                    ),
                    _FixturePresentationField(
                        string_ordinal["sum"], string_ordinal["i32"]
                    ),
                    _FixturePresentationField(
                        string_ordinal["payload"],
                        string_ordinal["vm.ref<vm, buffer>"],
                    ),
                ),
            ),
        )
    )
    metadata = _metadata(
        (_FixtureMetadataEntry(string_ordinal["author"], metadata_utf8, b"loom"),),
        export_scopes=(
            _FixtureMetadataScope(
                export_ordinal["run"],
                (
                    _FixtureMetadataEntry(
                        string_ordinal["purpose"], metadata_utf8, b"execution-test"
                    ),
                ),
            ),
        ),
    )

    return _render_image(
        specification,
        {
            "strings": strings,
            "ref_types": ref_types,
            "signatures": signatures,
            "callable_types": callable_types,
            "imports": imports,
            "exports": exports,
            "functions": functions,
            "globals": globals_payload,
            "rodata": rodata,
            "presentation": presentation,
            "metadata": metadata,
        },
        {"rodata": 8},
    )
