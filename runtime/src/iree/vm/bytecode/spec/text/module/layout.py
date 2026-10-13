# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Physical section traversal for canonical VM module text."""

from iree.vm.bytecode.spec.module import records
from iree.vm.bytecode.spec.specification import Specification
from iree.vm.bytecode.spec.text.module import (
    AlignmentLayout,
    DerivedField,
    FinalByteExtent,
    FinalExtentLayoutCount,
    FixedLayoutCount,
    FixedRunCount,
    GrammarCountDerivation,
    GrammarRun,
    HeaderFieldLayoutCount,
    HeaderFieldRunCount,
    MatchingFieldRunCount,
    RecordCountDerivation,
    RecordLayout,
    RecordRunCount,
    RemainingByteLength,
    RemainingLayoutCount,
    SectionDirectoryFieldDerivation,
    SectionDirectoryValue,
    SectionText,
    StatisticDerivation,
    SummedFieldLayoutCount,
    TailLayout,
)


def _header_and_rows(header, count_field_name: str, row):
    return (
        RecordLayout(header, FixedLayoutCount(1)),
        RecordLayout(row, HeaderFieldLayoutCount(header, count_field_name)),
    )


def build_sections(specification: Specification) -> tuple[SectionText, ...]:
    """Builds section grammars and wire traversal in physical order."""

    section = {item.name: item for item in specification.module_format.sections}
    sections = (
        SectionText(
            section["requirements"],
            (GrammarRun("requirement", RecordRunCount(records.REQUIREMENT_ROW)),),
            (RecordLayout(records.REQUIREMENT_ROW, RemainingLayoutCount()),),
        ),
        SectionText(
            section["strings"],
            (
                GrammarRun(
                    "string",
                    HeaderFieldRunCount(records.STRINGS_HEADER, "string_count_u32"),
                ),
            ),
            (
                RecordLayout(records.STRINGS_HEADER, FixedLayoutCount(1)),
                RecordLayout(
                    records.STRING_OFFSET,
                    HeaderFieldLayoutCount(
                        records.STRINGS_HEADER, "string_count_u32", 1
                    ),
                ),
                TailLayout("strings", RemainingByteLength()),
            ),
        ),
        SectionText(
            section["ref_types"],
            (GrammarRun("ref_group", RecordRunCount(records.REF_TYPE_GROUP_ROW)),),
            (
                *_header_and_rows(
                    records.REF_TYPES_HEADER,
                    "group_count_u32",
                    records.REF_TYPE_GROUP_ROW,
                ),
                RecordLayout(
                    records.REF_TYPE_ENTRY_ROW,
                    SummedFieldLayoutCount(
                        records.REF_TYPE_GROUP_ROW, ("entry_count_u32",)
                    ),
                ),
            ),
        ),
        SectionText(
            section["signatures"],
            (GrammarRun("signature", RecordRunCount(records.SIGNATURE_ROW)),),
            (
                *_header_and_rows(
                    records.SIGNATURES_HEADER,
                    "signature_count_u32",
                    records.SIGNATURE_ROW,
                ),
                RecordLayout(
                    records.SIGNATURE_DESCRIPTOR_ROW,
                    SummedFieldLayoutCount(
                        records.SIGNATURE_ROW,
                        (
                            "argument_value_count_u16",
                            "result_value_count_u16",
                            "argument_ref_count_u16",
                            "result_ref_count_u16",
                            "argument_function_count_u16",
                            "result_function_count_u16",
                        ),
                    ),
                ),
            ),
        ),
        SectionText(
            section["callable_types"],
            (GrammarRun("callable", RecordRunCount(records.CALLABLE_TYPE_ROW)),),
            _header_and_rows(
                records.CALLABLE_TYPES_HEADER,
                "callable_type_count_u32",
                records.CALLABLE_TYPE_ROW,
            ),
        ),
        SectionText(
            section["imports"],
            (GrammarRun("import_group", RecordRunCount(records.IMPORT_GROUP_ROW)),),
            (
                *_header_and_rows(
                    records.IMPORTS_HEADER,
                    "group_count_u32",
                    records.IMPORT_GROUP_ROW,
                ),
                RecordLayout(
                    records.IMPORT_ENTRY_ROW,
                    SummedFieldLayoutCount(
                        records.IMPORT_GROUP_ROW, ("entry_count_u32",)
                    ),
                ),
            ),
        ),
        SectionText(
            section["exports"],
            (GrammarRun("export", RecordRunCount(records.EXPORT_ROW)),),
            _header_and_rows(
                records.EXPORTS_HEADER, "export_count_u32", records.EXPORT_ROW
            ),
        ),
        SectionText(
            section["functions"],
            (GrammarRun("function", RecordRunCount(records.FUNCTION_ROW)),),
            (
                *_header_and_rows(
                    records.FUNCTIONS_HEADER,
                    "function_count_u32",
                    records.FUNCTION_ROW,
                ),
                RecordLayout(
                    records.SWITCH_TARGET_ENTRY,
                    FinalExtentLayoutCount(
                        records.FUNCTION_ROW,
                        "switch_target_base_u32",
                        "switch_target_entry_count_u32",
                    ),
                ),
                TailLayout(
                    "bytecode",
                    FinalByteExtent(
                        records.FUNCTION_ROW,
                        "bytecode_offset_u32",
                        "bytecode_length_u32",
                    ),
                ),
            ),
        ),
        SectionText(
            section["constants"],
            (GrammarRun("constant", RecordRunCount(records.CONSTANT_CELL)),),
            (RecordLayout(records.CONSTANT_CELL, RemainingLayoutCount()),),
        ),
        SectionText(
            section["globals"],
            (
                GrammarRun("global_values", FixedRunCount(1)),
                GrammarRun("global_refs", FixedRunCount(1)),
                GrammarRun("global_functions", FixedRunCount(1)),
            ),
            (
                RecordLayout(records.GLOBALS_HEADER, FixedLayoutCount(1)),
                RecordLayout(
                    records.GLOBAL_REF_DESCRIPTOR_ROW,
                    HeaderFieldLayoutCount(records.GLOBALS_HEADER, "ref_count_u32"),
                ),
                RecordLayout(
                    records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
                    HeaderFieldLayoutCount(
                        records.GLOBALS_HEADER, "function_count_u32"
                    ),
                ),
            ),
        ),
        SectionText(
            section["rodata"],
            (GrammarRun("rodata", RecordRunCount(records.RODATA_BLOCK_DESCRIPTOR)),),
            (
                *_header_and_rows(
                    records.RODATA_HEADER,
                    "block_count_u32",
                    records.RODATA_BLOCK_DESCRIPTOR,
                ),
                TailLayout("rodata", RemainingByteLength()),
            ),
        ),
        SectionText(
            section["presentation"],
            (
                GrammarRun(
                    "presentation_import",
                    MatchingFieldRunCount(
                        records.PRESENTATION_ENTRY_ROW, "declaration_kind_u16", 1
                    ),
                ),
                GrammarRun(
                    "presentation_export",
                    MatchingFieldRunCount(
                        records.PRESENTATION_ENTRY_ROW, "declaration_kind_u16", 2
                    ),
                ),
            ),
            (
                *_header_and_rows(
                    records.PRESENTATION_HEADER,
                    "entry_count_u32",
                    records.PRESENTATION_ENTRY_ROW,
                ),
                RecordLayout(records.PRESENTATION_FIELD_ROW, RemainingLayoutCount()),
            ),
        ),
        SectionText(
            section["metadata"],
            (
                GrammarRun("metadata_module", FixedRunCount(1)),
                GrammarRun(
                    "metadata_import",
                    HeaderFieldRunCount(
                        records.METADATA_HEADER, "import_scope_count_u32"
                    ),
                ),
                GrammarRun(
                    "metadata_export",
                    HeaderFieldRunCount(
                        records.METADATA_HEADER, "export_scope_count_u32"
                    ),
                ),
            ),
            (
                RecordLayout(records.METADATA_HEADER, FixedLayoutCount(1)),
                RecordLayout(
                    records.METADATA_SCOPE_ROW,
                    SummedFieldLayoutCount(
                        records.METADATA_HEADER,
                        ("import_scope_count_u32", "export_scope_count_u32"),
                    ),
                ),
                RecordLayout(
                    records.METADATA_ENTRY_ROW,
                    HeaderFieldLayoutCount(
                        records.METADATA_HEADER, "total_entry_count_u32"
                    ),
                ),
                AlignmentLayout(8),
                RecordLayout(
                    records.METADATA_VALUE_OFFSET,
                    HeaderFieldLayoutCount(
                        records.METADATA_HEADER, "total_entry_count_u32", 1
                    ),
                ),
                TailLayout("metadata", RemainingByteLength()),
            ),
        ),
    )
    return sections + (
        SectionText(
            section["native_calls"],
            (
                GrammarRun("native_layout", RecordRunCount(records.NATIVE_LAYOUT_ROW)),
                GrammarRun("native_import", RecordRunCount(records.NATIVE_IMPORT_ROW)),
                GrammarRun(
                    "native_callsite", RecordRunCount(records.NATIVE_CALLSITE_ROW)
                ),
            ),
            (
                RecordLayout(records.NATIVE_CALLS_HEADER, FixedLayoutCount(1)),
                *(
                    RecordLayout(
                        record,
                        HeaderFieldLayoutCount(records.NATIVE_CALLS_HEADER, field),
                    )
                    for record, field in (
                        (records.NATIVE_LAYOUT_ROW, "layout_count_u32"),
                        (records.NATIVE_FIELD_ROW, "field_count_u32"),
                        (records.NATIVE_IMPORT_ROW, "import_count_u32"),
                        (records.NATIVE_CALLSITE_ROW, "callsite_count_u32"),
                    )
                ),
                RecordLayout(records.NATIVE_CALL_WORD, RemainingLayoutCount()),
            ),
        ),
    )


def build_derived_fields() -> tuple[DerivedField, ...]:
    """Builds record fields mechanically derived during assembly."""

    return (
        RecordCountDerivation(
            records.NATIVE_CALLS_HEADER, "layout_count_u32", records.NATIVE_LAYOUT_ROW
        ),
        RecordCountDerivation(
            records.NATIVE_CALLS_HEADER, "field_count_u32", records.NATIVE_FIELD_ROW
        ),
        RecordCountDerivation(
            records.NATIVE_CALLS_HEADER, "import_count_u32", records.NATIVE_IMPORT_ROW
        ),
        RecordCountDerivation(
            records.NATIVE_CALLS_HEADER,
            "callsite_count_u32",
            records.NATIVE_CALLSITE_ROW,
        ),
        RecordCountDerivation(
            records.IMAGE_HEADER,
            "section_count_u16",
            records.SECTION_DIRECTORY_ROW,
        ),
        SectionDirectoryFieldDerivation(
            records.SECTION_DIRECTORY_ROW,
            "section_type_u16",
            SectionDirectoryValue.TYPE,
        ),
        SectionDirectoryFieldDerivation(
            records.SECTION_DIRECTORY_ROW,
            "section_flags_u16",
            SectionDirectoryValue.FLAGS,
        ),
        SectionDirectoryFieldDerivation(
            records.SECTION_DIRECTORY_ROW,
            "byte_length_u64",
            SectionDirectoryValue.BYTE_LENGTH,
        ),
        RecordCountDerivation(
            records.STRINGS_HEADER, "string_count_u32", records.STRING_OFFSET
        ),
        RecordCountDerivation(
            records.REF_TYPES_HEADER,
            "group_count_u32",
            records.REF_TYPE_GROUP_ROW,
        ),
        RecordCountDerivation(
            records.SIGNATURES_HEADER,
            "signature_count_u32",
            records.SIGNATURE_ROW,
        ),
        RecordCountDerivation(
            records.CALLABLE_TYPES_HEADER,
            "callable_type_count_u32",
            records.CALLABLE_TYPE_ROW,
        ),
        RecordCountDerivation(
            records.IMPORTS_HEADER, "group_count_u32", records.IMPORT_GROUP_ROW
        ),
        RecordCountDerivation(
            records.EXPORTS_HEADER, "export_count_u32", records.EXPORT_ROW
        ),
        RecordCountDerivation(
            records.FUNCTIONS_HEADER, "function_count_u32", records.FUNCTION_ROW
        ),
        StatisticDerivation(
            records.FUNCTIONS_HEADER,
            "maximum_block_count_u32",
            "maximum_block_count",
        ),
        RecordCountDerivation(
            records.RODATA_HEADER,
            "block_count_u32",
            records.RODATA_BLOCK_DESCRIPTOR,
        ),
        RecordCountDerivation(
            records.PRESENTATION_HEADER,
            "entry_count_u32",
            records.PRESENTATION_ENTRY_ROW,
        ),
        GrammarCountDerivation(
            records.METADATA_HEADER,
            "import_scope_count_u32",
            "metadata_import",
        ),
        GrammarCountDerivation(
            records.METADATA_HEADER,
            "export_scope_count_u32",
            "metadata_export",
        ),
        RecordCountDerivation(
            records.METADATA_HEADER,
            "total_entry_count_u32",
            records.METADATA_ENTRY_ROW,
        ),
    )
