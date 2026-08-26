# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 fixed module-format wire records."""

from __future__ import annotations

from model.module.validation import (
    CALLABLE_TYPES,
    CORE_MAJOR,
    CORE_REQUIRED_MINOR,
    FUNCTIONS,
    NONCORE_PAGE,
    NONEMPTY_STRINGS,
    ORDINAL,
    ORDINAL_OR_NULL,
    PAGE_MAJOR,
    PAGE_REQUIRED_MINOR,
    REF_TYPES,
    SECTION_BYTE_LENGTH,
    SECTION_FLAGS,
    SECTION_TYPE,
    SIGNATURE_DESCRIPTOR,
    SIGNATURES,
    STRING_OFFSET,
    STRINGS,
    SWITCH_TARGET,
)
from model.schema import (
    ALLOWED_BITS,
    ALLOWED_RANGE,
    ANY_BITS,
    EXACT_BYTES,
    MULTIPLE,
    U8,
    U16,
    U32,
    U64,
    ZERO,
    EntityReference,
    RuleUse,
    WireField,
    WireRecord,
    WireRecordLayout,
)
from model.specification import CORE_0


def _record(
    key: str,
    c_type: str,
    byte_length: int,
    alignment: int,
    fields: tuple[WireField, ...],
    *,
    scalar_alias: bool = False,
) -> tuple[WireRecord, WireRecordLayout]:
    record_id = f"core.module.record.{key}"
    record = WireRecord(
        entity_id=record_id,
        since=CORE_0,
        summary=f"Fixed module wire record {key}.",
        c_type=c_type,
    )
    layout = WireRecordLayout(
        entity_id=f"{record_id}.layout_0",
        since=CORE_0,
        summary=f"Core 0.0 layout of {key}.",
        record_id=record_id,
        byte_length=byte_length,
        alignment=alignment,
        fields=fields,
        scalar_alias=scalar_alias,
    )
    return record, layout


_RECORD_DEFINITIONS = (
    _record(
        "image_header",
        "iree_vm_bytecode_v0_image_header_t",
        16,
        2,
        (
            WireField(
                name="magic_u8",
                offset=0,
                encoding_id=U8.entity_id,
                description="Exact eight-byte IREE VM image magic.",
                validation=(
                    RuleUse(
                        EXACT_BYTES.entity_id, (bytes.fromhex("49524545564d0000"),)
                    ),
                ),
                array_length=8,
            ),
            WireField(
                name="core_major_u16",
                offset=8,
                encoding_id=U16.entity_id,
                description="Incompatible core container and ISA version.",
                validation=(RuleUse(CORE_MAJOR.entity_id, ()),),
            ),
            WireField(
                name="core_required_minor_u16",
                offset=10,
                encoding_id=U16.entity_id,
                description="Minimum compatible core minor version.",
                validation=(RuleUse(CORE_REQUIRED_MINOR.entity_id, ()),),
            ),
            WireField(
                name="section_count_u16",
                offset=12,
                encoding_id=U16.entity_id,
                description="Number of section-directory rows.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="zero_padding_u16",
                offset=14,
                encoding_id=U16.entity_id,
                description="Canonical zero padding.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "section_directory_row",
        "iree_vm_bytecode_v0_section_directory_row_t",
        16,
        8,
        (
            WireField(
                name="section_type_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Architectural section identifier.",
                validation=(RuleUse(SECTION_TYPE.entity_id, ()),),
            ),
            WireField(
                name="section_flags_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Flags interpreted by the owning section authority.",
                validation=(RuleUse(SECTION_FLAGS.entity_id, ()),),
            ),
            WireField(
                name="reserved_u32",
                offset=4,
                encoding_id=U32.entity_id,
                description="Reserved zero word preserving native u64 alignment.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
            WireField(
                name="byte_length_u64",
                offset=8,
                encoding_id=U64.entity_id,
                description="Exact section payload length in bytes.",
                validation=(RuleUse(SECTION_BYTE_LENGTH.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "requirement_row",
        "iree_vm_bytecode_v0_requirement_row_t",
        6,
        2,
        (
            WireField(
                name="page_id_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Non-core architectural page identifier.",
                validation=(RuleUse(NONCORE_PAGE.entity_id, ()),),
            ),
            WireField(
                name="major_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Exact incompatible page version.",
                validation=(RuleUse(PAGE_MAJOR.entity_id, ()),),
            ),
            WireField(
                name="required_minor_u16",
                offset=4,
                encoding_id=U16.entity_id,
                description="Minimum compatible page minor version.",
                validation=(RuleUse(PAGE_REQUIRED_MINOR.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "strings_header",
        "iree_vm_bytecode_v0_strings_header_t",
        4,
        4,
        (
            WireField(
                name="string_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of length-delimited strings.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65535)),),
            ),
        ),
    ),
    _record(
        "string_offset",
        "iree_vm_bytecode_v0_string_offset_t",
        4,
        4,
        (
            WireField(
                name="byte_offset_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Byte offset into the trailing UTF-8 area.",
                validation=(RuleUse(STRING_OFFSET.entity_id, ()),),
            ),
        ),
        scalar_alias=True,
    ),
    _record(
        "ref_types_header",
        "iree_vm_bytecode_v0_ref_types_header_t",
        4,
        4,
        (
            WireField(
                name="group_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of namespace groups.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65535)),),
            ),
        ),
    ),
    _record(
        "ref_type_group_row",
        "iree_vm_bytecode_v0_ref_type_group_row_t",
        8,
        4,
        (
            WireField(
                name="namespace_string_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Nonempty type-namespace string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id,
                        (EntityReference(NONEMPTY_STRINGS.entity_id),),
                    ),
                ),
            ),
            WireField(
                name="zero_padding_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Canonical zero padding.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
            WireField(
                name="entry_count_u32",
                offset=4,
                encoding_id=U32.entity_id,
                description="Nonzero count of local type entries.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 4294967295)),),
            ),
        ),
    ),
    _record(
        "ref_type_entry_row",
        "iree_vm_bytecode_v0_ref_type_entry_row_t",
        4,
        2,
        (
            WireField(
                name="type_name_string_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Nonempty local type-name string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id,
                        (EntityReference(NONEMPTY_STRINGS.entity_id),),
                    ),
                ),
            ),
            WireField(
                name="required_flags_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Required type features; zero in version zero.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "signatures_header",
        "iree_vm_bytecode_v0_signatures_header_t",
        4,
        4,
        (
            WireField(
                name="signature_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of source-ordered logical signatures.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65536)),),
            ),
        ),
    ),
    _record(
        "signature_row",
        "iree_vm_bytecode_v0_signature_row_t",
        16,
        4,
        (
            WireField(
                name="descriptor_base_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Canonical running base in the descriptor array.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="argument_value_count_u16",
                offset=4,
                encoding_id=U16.entity_id,
                description="Value argument count.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="result_value_count_u16",
                offset=6,
                encoding_id=U16.entity_id,
                description="Value result count.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="argument_ref_count_u16",
                offset=8,
                encoding_id=U16.entity_id,
                description="Ref argument count.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="result_ref_count_u16",
                offset=10,
                encoding_id=U16.entity_id,
                description="Ref result count.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="argument_function_count_u16",
                offset=12,
                encoding_id=U16.entity_id,
                description="Function argument count.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="result_function_count_u16",
                offset=14,
                encoding_id=U16.entity_id,
                description="Function result count.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "signature_descriptor_row",
        "iree_vm_bytecode_v0_signature_descriptor_row_t",
        4,
        2,
        (
            WireField(
                name="kind_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Architectural scalar, REF, or FUNCTION kind.",
                validation=(RuleUse(SIGNATURE_DESCRIPTOR.entity_id, ()),),
            ),
            WireField(
                name="type_ordinal_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Exact ref/callable type ordinal, or zero for scalars.",
                validation=(RuleUse(SIGNATURE_DESCRIPTOR.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "callable_types_header",
        "iree_vm_bytecode_v0_callable_types_header_t",
        4,
        4,
        (
            WireField(
                name="callable_type_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of structural callable declarations.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65536)),),
            ),
        ),
    ),
    _record(
        "callable_type_row",
        "iree_vm_bytecode_v0_callable_type_row_t",
        8,
        2,
        (
            WireField(
                name="signature_ordinal_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Exact source-ordered signature.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id, (EntityReference(SIGNATURES.entity_id),)
                    ),
                ),
            ),
            WireField(
                name="flags_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Callable behavior permission flags.",
                validation=(RuleUse(ALLOWED_BITS.entity_id, (1,)),),
            ),
            WireField(
                name="nesting_depth_u16",
                offset=4,
                encoding_id=U16.entity_id,
                description="Maximum nested callable depth; leaf signatures are zero.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="reserved_u16",
                offset=6,
                encoding_id=U16.entity_id,
                description="Reserved zero bits.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "imports_header",
        "iree_vm_bytecode_v0_imports_header_t",
        4,
        4,
        (
            WireField(
                name="group_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of target-module groups.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65535)),),
            ),
        ),
    ),
    _record(
        "import_group_row",
        "iree_vm_bytecode_v0_import_group_row_t",
        8,
        4,
        (
            WireField(
                name="module_name_string_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Nonempty target-module name string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id,
                        (EntityReference(NONEMPTY_STRINGS.entity_id),),
                    ),
                ),
            ),
            WireField(
                name="zero_padding_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Canonical zero padding.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
            WireField(
                name="entry_count_u32",
                offset=4,
                encoding_id=U32.entity_id,
                description="Nonzero count of imported symbols.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 4294967295)),),
            ),
        ),
    ),
    _record(
        "import_entry_row",
        "iree_vm_bytecode_v0_import_entry_row_t",
        8,
        2,
        (
            WireField(
                name="symbol_name_string_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Nonempty target export-name string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id,
                        (EntityReference(NONEMPTY_STRINGS.entity_id),),
                    ),
                ),
            ),
            WireField(
                name="callable_type_ordinal_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Exact local callable-type requirement.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id, (EntityReference(CALLABLE_TYPES.entity_id),)
                    ),
                ),
            ),
            WireField(
                name="flags_u16",
                offset=4,
                encoding_id=U16.entity_id,
                description="Import declaration flags.",
                validation=(RuleUse(ALLOWED_BITS.entity_id, (1,)),),
            ),
            WireField(
                name="zero_padding_u16",
                offset=6,
                encoding_id=U16.entity_id,
                description="Canonical zero padding.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "exports_header",
        "iree_vm_bytecode_v0_exports_header_t",
        4,
        4,
        (
            WireField(
                name="export_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of public export rows.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65535)),),
            ),
        ),
    ),
    _record(
        "export_row",
        "iree_vm_bytecode_v0_export_row_t",
        8,
        2,
        (
            WireField(
                name="name_string_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Nonempty public export-name string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id,
                        (EntityReference(NONEMPTY_STRINGS.entity_id),),
                    ),
                ),
            ),
            WireField(
                name="callable_type_ordinal_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Exact public callable type.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id, (EntityReference(CALLABLE_TYPES.entity_id),)
                    ),
                ),
            ),
            WireField(
                name="function_ordinal_u16",
                offset=4,
                encoding_id=U16.entity_id,
                description="Valid module-local function ordinal.",
                validation=(
                    RuleUse(ORDINAL.entity_id, (EntityReference(FUNCTIONS.entity_id),)),
                ),
            ),
            WireField(
                name="zero_padding_u16",
                offset=6,
                encoding_id=U16.entity_id,
                description="Canonical zero padding.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "functions_header",
        "iree_vm_bytecode_v0_functions_header_t",
        4,
        4,
        (
            WireField(
                name="function_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of bytecode functions.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65536)),),
            ),
        ),
    ),
    _record(
        "function_row",
        "iree_vm_bytecode_v0_function_row_t",
        48,
        4,
        (
            WireField(
                name="callable_type_ordinal_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Canonical callable type implemented by the function.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id,
                        (EntityReference(CALLABLE_TYPES.entity_id),),
                    ),
                ),
            ),
            WireField(
                name="flags_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Function behavior flags.",
                validation=(RuleUse(ALLOWED_BITS.entity_id, (3,)),),
            ),
            WireField(
                name="bytecode_offset_u32",
                offset=4,
                encoding_id=U32.entity_id,
                description="Canonical byte offset from the bytecode payload base.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="bytecode_length_u32",
                offset=8,
                encoding_id=U32.entity_id,
                description="Nonzero record-stream length in bytes.",
                validation=(
                    RuleUse(ALLOWED_RANGE.entity_id, (1, 4294967295)),
                    RuleUse(MULTIPLE.entity_id, (4,)),
                ),
            ),
            WireField(
                name="switch_target_base_u32",
                offset=12,
                encoding_id=U32.entity_id,
                description="Canonical entry base in the switch-target array.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="switch_target_entry_count_u32",
                offset=16,
                encoding_id=U32.entity_id,
                description="Aggregate switch-target entries owned by the function.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="local_byte_length_u16",
                offset=20,
                encoding_id=U16.entity_id,
                description="Complete function-local byte storage extent.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="value_register_count_u16",
                offset=22,
                encoding_id=U16.entity_id,
                description="Value-register count.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 256)),),
            ),
            WireField(
                name="ref_register_count_u16",
                offset=24,
                encoding_id=U16.entity_id,
                description="Ref-register count.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 256)),),
            ),
            WireField(
                name="function_register_count_u16",
                offset=26,
                encoding_id=U16.entity_id,
                description="Function-register count.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 256)),),
            ),
            WireField(
                name="local_ref_count_u32",
                offset=28,
                encoding_id=U32.entity_id,
                description="Function-local owning ref-slot count.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="local_function_count_u32",
                offset=32,
                encoding_id=U32.entity_id,
                description="Function-local non-owning function-slot count.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="block_count_u32",
                offset=36,
                encoding_id=U32.entity_id,
                description="Exact number of decoded control.block records.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65536)),),
            ),
            WireField(
                name="reserved_u32",
                offset=40,
                encoding_id=U32.entity_id,
                description="Reserved zero words.",
                validation=(RuleUse(ZERO.entity_id, ()),),
                array_length=2,
            ),
        ),
    ),
    _record(
        "switch_target_entry",
        "iree_vm_bytecode_v0_switch_target_entry_t",
        4,
        4,
        (
            WireField(
                name="target_word_offset_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Function-relative target offset in four-byte words.",
                validation=(RuleUse(SWITCH_TARGET.entity_id, ()),),
            ),
        ),
        scalar_alias=True,
    ),
    _record(
        "constant_cell",
        "iree_vm_bytecode_v0_constant_cell_t",
        8,
        8,
        (
            WireField(
                name="bits_u64",
                offset=0,
                encoding_id=U64.entity_id,
                description="Arbitrary little-endian value-cell bits.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
        ),
        scalar_alias=True,
    ),
    _record(
        "globals_header",
        "iree_vm_bytecode_v0_globals_header_t",
        32,
        4,
        (
            WireField(
                name="value_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Total module-local value-global count.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="immutable_value_count_u32",
                offset=4,
                encoding_id=U32.entity_id,
                description="Dense immutable value-global prefix length.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="ref_count_u32",
                offset=8,
                encoding_id=U32.entity_id,
                description="Total module-local ref-global count.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="immutable_ref_count_u32",
                offset=12,
                encoding_id=U32.entity_id,
                description="Dense immutable ref-global prefix length.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="function_count_u32",
                offset=16,
                encoding_id=U32.entity_id,
                description="Total module-local function-global count.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="immutable_function_count_u32",
                offset=20,
                encoding_id=U32.entity_id,
                description="Dense immutable function-global prefix length.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="reserved_u32",
                offset=24,
                encoding_id=U32.entity_id,
                description="Reserved zero words.",
                validation=(RuleUse(ZERO.entity_id, ()),),
                array_length=2,
            ),
        ),
    ),
    _record(
        "global_ref_descriptor_row",
        "iree_vm_bytecode_v0_global_ref_descriptor_row_t",
        4,
        2,
        (
            WireField(
                name="ref_type_ordinal_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Exact module-local ref-type ordinal.",
                validation=(
                    RuleUse(ORDINAL.entity_id, (EntityReference(REF_TYPES.entity_id),)),
                ),
            ),
            WireField(
                name="flags_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Ref-global behavior flags.",
                validation=(RuleUse(ALLOWED_BITS.entity_id, (1,)),),
            ),
        ),
    ),
    _record(
        "global_function_descriptor_row",
        "iree_vm_bytecode_v0_global_function_descriptor_row_t",
        4,
        2,
        (
            WireField(
                name="callable_type_ordinal_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Exact module-local callable-type ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id, (EntityReference(CALLABLE_TYPES.entity_id),)
                    ),
                ),
            ),
            WireField(
                name="flags_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Function-global behavior flags.",
                validation=(RuleUse(ALLOWED_BITS.entity_id, (1,)),),
            ),
        ),
    ),
    _record(
        "rodata_header",
        "iree_vm_bytecode_v0_rodata_header_t",
        8,
        4,
        (
            WireField(
                name="block_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of direct rodata ordinals.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65536)),),
            ),
            WireField(
                name="zero_padding_u32",
                offset=4,
                encoding_id=U32.entity_id,
                description="Canonical zero padding.",
                validation=(RuleUse(ZERO.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "rodata_block_length",
        "iree_vm_bytecode_v0_rodata_block_length_t",
        8,
        8,
        (
            WireField(
                name="byte_length_u64",
                offset=0,
                encoding_id=U64.entity_id,
                description="Exact byte length of one rodata block.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
        ),
        scalar_alias=True,
    ),
    _record(
        "presentation_header",
        "iree_vm_bytecode_v0_presentation_header_t",
        4,
        4,
        (
            WireField(
                name="entry_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of sparse import/export presentation rows.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 131071)),),
            ),
        ),
    ),
    _record(
        "presentation_entry_row",
        "iree_vm_bytecode_v0_presentation_entry_row_t",
        12,
        4,
        (
            WireField(
                name="declaration_ordinal_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Valid ordinal in the selected declaration domain.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="declaration_kind_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Import or export declaration kind.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 2)),),
            ),
            WireField(
                name="documentation_string_u16",
                offset=4,
                encoding_id=U16.entity_id,
                description="Nullable documentation string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL_OR_NULL.entity_id,
                        (EntityReference(STRINGS.entity_id), 65535),
                    ),
                ),
            ),
            WireField(
                name="authored_type_string_u16",
                offset=6,
                encoding_id=U16.entity_id,
                description="Nullable authored function-type string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL_OR_NULL.entity_id,
                        (EntityReference(STRINGS.entity_id), 65535),
                    ),
                ),
            ),
            WireField(
                name="field_base_u32",
                offset=8,
                encoding_id=U32.entity_id,
                description="Canonical running base in the presentation field array.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "presentation_field_row",
        "iree_vm_bytecode_v0_presentation_field_row_t",
        4,
        2,
        (
            WireField(
                name="name_string_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Nullable argument or result name string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL_OR_NULL.entity_id,
                        (EntityReference(STRINGS.entity_id), 65535),
                    ),
                ),
            ),
            WireField(
                name="authored_type_string_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Nullable authored field-type string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL_OR_NULL.entity_id,
                        (EntityReference(STRINGS.entity_id), 65535),
                    ),
                ),
            ),
        ),
    ),
    _record(
        "metadata_header",
        "iree_vm_bytecode_v0_metadata_header_t",
        16,
        4,
        (
            WireField(
                name="module_entry_count_u32",
                offset=0,
                encoding_id=U32.entity_id,
                description="Number of module-scope metadata entries.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65535)),),
            ),
            WireField(
                name="import_scope_count_u32",
                offset=4,
                encoding_id=U32.entity_id,
                description="Number of nonempty import metadata scopes.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65536)),),
            ),
            WireField(
                name="export_scope_count_u32",
                offset=8,
                encoding_id=U32.entity_id,
                description="Number of nonempty export metadata scopes.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (0, 65535)),),
            ),
            WireField(
                name="total_entry_count_u32",
                offset=12,
                encoding_id=U32.entity_id,
                description="Total metadata entry count across all scopes.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 4294967295)),),
            ),
        ),
    ),
    _record(
        "metadata_scope_row",
        "iree_vm_bytecode_v0_metadata_scope_row_t",
        8,
        4,
        (
            WireField(
                name="declaration_ordinal_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Valid import or export declaration ordinal.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
            WireField(
                name="entry_count_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Nonzero number of entries in the scope.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65535)),),
            ),
            WireField(
                name="entry_base_u32",
                offset=4,
                encoding_id=U32.entity_id,
                description="Canonical running base in the metadata entry array.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
        ),
    ),
    _record(
        "metadata_entry_row",
        "iree_vm_bytecode_v0_metadata_entry_row_t",
        4,
        2,
        (
            WireField(
                name="key_string_u16",
                offset=0,
                encoding_id=U16.entity_id,
                description="Nonempty metadata-key string ordinal.",
                validation=(
                    RuleUse(
                        ORDINAL.entity_id,
                        (EntityReference(NONEMPTY_STRINGS.entity_id),),
                    ),
                ),
            ),
            WireField(
                name="value_type_u16",
                offset=2,
                encoding_id=U16.entity_id,
                description="Nonzero open metadata value-type identifier.",
                validation=(RuleUse(ALLOWED_RANGE.entity_id, (1, 65535)),),
            ),
        ),
    ),
    _record(
        "metadata_value_offset",
        "iree_vm_bytecode_v0_metadata_value_offset_t",
        8,
        8,
        (
            WireField(
                name="byte_offset_u64",
                offset=0,
                encoding_id=U64.entity_id,
                description="Byte offset into the trailing metadata value area.",
                validation=(RuleUse(ANY_BITS.entity_id, ()),),
            ),
        ),
        scalar_alias=True,
    ),
)

RECORDS = tuple(definition[0] for definition in _RECORD_DEFINITIONS)
RECORD_LAYOUTS = tuple(definition[1] for definition in _RECORD_DEFINITIONS)
RECORDS_BY_KEY = {
    record.entity_id.removeprefix("core.module.record."): record for record in RECORDS
}
RECORD_LAYOUTS_BY_KEY = {
    layout.record_id.removeprefix("core.module.record."): layout
    for layout in RECORD_LAYOUTS
}
ENTITIES = (*RECORDS, *RECORD_LAYOUTS)
