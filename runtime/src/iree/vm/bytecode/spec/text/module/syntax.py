# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical source grammar for physical VM module declarations."""

from iree.vm.bytecode.spec.module import WireRecord, records
from iree.vm.bytecode.spec.schema import NumericTable
from iree.vm.bytecode.spec.specification import Specification
from iree.vm.bytecode.spec.text import SymbolDomain
from iree.vm.bytecode.spec.text.module import (
    ConstantField,
    DeclareSymbol,
    FunctionBody,
    GrammarElement,
    HexBlob,
    MetadataValue,
    ModuleShell,
    NativeCallBody,
    NativeLayoutBody,
    NestedBody,
    NullableSymbolReference,
    NumericField,
    NumericSpelling,
    QuotedBlob,
    RecordGrammar,
    RowMode,
    ScalarField,
    ScalarRadix,
    SignatureBody,
    SymbolReference,
)


def _numeric_table(specification: Specification, name: str) -> NumericTable:
    return next(
        table
        for table in specification.module_format.numeric_tables
        if table.name == name
    )


def _grammar(
    name: str,
    record: WireRecord,
    *elements: GrammarElement,
    keyword: str | None = None,
    row_mode: RowMode = RowMode.APPEND,
) -> RecordGrammar:
    return RecordGrammar(
        name,
        keyword or name,
        record,
        row_mode,
        elements,
    )


def _symbol(
    record: WireRecord, field_name: str, domain: SymbolDomain
) -> SymbolReference:
    return SymbolReference(record, field_name, domain)


def _nullable_symbol(
    record: WireRecord, field_name: str, domain: SymbolDomain
) -> NullableSymbolReference:
    return NullableSymbolReference(record, field_name, domain)


def _scalar(
    record: WireRecord,
    field_name: str,
    radix: ScalarRadix = ScalarRadix.DECIMAL,
) -> ScalarField:
    return ScalarField(record, field_name, radix)


def _numeric(
    specification: Specification,
    record: WireRecord,
    field_name: str,
    table_name: str,
    spelling_overrides: tuple[NumericSpelling, ...] = (),
) -> NumericField:
    return NumericField(
        record,
        field_name,
        _numeric_table(specification, table_name),
        spelling_overrides,
    )


def _body(
    grammar_name: str,
    child_record: WireRecord,
    parent_record: WireRecord,
    *,
    base_field_name: str | None = None,
    count_field_name: str | None = None,
) -> NestedBody:
    return NestedBody(
        grammar_name,
        child_record,
        parent_record,
        base_field_name,
        count_field_name,
    )


def build_shell() -> ModuleShell:
    """Builds the canonical module and section wrappers."""

    return ModuleShell(
        module_prefix="vm.module core(",
        core_major=_scalar(records.IMAGE_HEADER, "core_major_u16"),
        version_separator=".",
        core_required_minor=_scalar(records.IMAGE_HEADER, "core_required_minor_u16"),
        module_open=") {",
        section_prefix="section ",
        section_alignment_prefix=" alignment(",
        section_alignment=_scalar(
            records.SECTION_DIRECTORY_ROW, "payload_alignment_u32"
        ),
        section_open=") {",
        close="}",
    )


def build_grammars(specification: Specification) -> tuple[RecordGrammar, ...]:
    """Builds the canonical declaration grammars in parser table order."""

    return (
        _grammar(
            "native_layout",
            records.NATIVE_LAYOUT_ROW,
            "native_layout @",
            DeclareSymbol(SymbolDomain.NATIVE_LAYOUT),
            " ",
            NativeLayoutBody(
                records.NATIVE_LAYOUT_ROW,
                records.NATIVE_FIELD_ROW,
                _numeric_table(specification, "signature_kind"),
            ),
        ),
        _grammar(
            "native_import",
            records.NATIVE_IMPORT_ROW,
            "native_import @",
            DeclareSymbol(SymbolDomain.NATIVE_IMPORT),
            " module(@",
            _symbol(records.NATIVE_IMPORT_ROW, "module_name_u16", SymbolDomain.STRING),
            ") export(@",
            _symbol(records.NATIVE_IMPORT_ROW, "export_name_u16", SymbolDomain.STRING),
            ") version(",
            _scalar(records.NATIVE_IMPORT_ROW, "version_u32"),
            ") : @",
            _symbol(
                records.NATIVE_IMPORT_ROW,
                "layout_ordinal_u16",
                SymbolDomain.NATIVE_LAYOUT,
            ),
        ),
        _grammar(
            "native_callsite",
            records.NATIVE_CALLSITE_ROW,
            "native_callsite @",
            DeclareSymbol(SymbolDomain.NATIVE_CALLSITE),
            " function(@",
            _symbol(
                records.NATIVE_CALLSITE_ROW,
                "function_ordinal_u16",
                SymbolDomain.FUNCTION,
            ),
            ") callee(@",
            _symbol(
                records.NATIVE_CALLSITE_ROW,
                "import_ordinal_u16",
                SymbolDomain.NATIVE_IMPORT,
            ),
            ") ",
            NativeCallBody(records.NATIVE_CALLSITE_ROW, records.NATIVE_CALL_WORD),
        ),
        _grammar(
            "requirement",
            records.REQUIREMENT_ROW,
            "require page(",
            _scalar(
                records.REQUIREMENT_ROW,
                "page_id_u16",
                ScalarRadix.HEXADECIMAL,
            ),
            ") version(",
            _scalar(records.REQUIREMENT_ROW, "major_u16"),
            ".",
            _scalar(records.REQUIREMENT_ROW, "required_minor_u16"),
            ")",
            keyword="require",
        ),
        _grammar(
            "string",
            records.STRING_OFFSET,
            "string @",
            DeclareSymbol(SymbolDomain.STRING),
            " = ",
            QuotedBlob("strings", records.STRING_OFFSET),
        ),
        _grammar(
            "ref_group",
            records.REF_TYPE_GROUP_ROW,
            "namespace @",
            DeclareSymbol(SymbolDomain.REF_GROUP),
            " name(@",
            _symbol(
                records.REF_TYPE_GROUP_ROW,
                "namespace_string_u16",
                SymbolDomain.STRING,
            ),
            ") ",
            _body(
                "ref_type",
                records.REF_TYPE_ENTRY_ROW,
                records.REF_TYPE_GROUP_ROW,
                count_field_name="entry_count_u32",
            ),
            keyword="namespace",
        ),
        _grammar(
            "ref_type",
            records.REF_TYPE_ENTRY_ROW,
            "ref_type @",
            DeclareSymbol(SymbolDomain.REF_TYPE),
            " name(@",
            _symbol(
                records.REF_TYPE_ENTRY_ROW,
                "type_name_string_u16",
                SymbolDomain.STRING,
            ),
            ")",
        ),
        _grammar(
            "signature",
            records.SIGNATURE_ROW,
            "signature @",
            DeclareSymbol(SymbolDomain.SIGNATURE),
            " ",
            SignatureBody(
                records.SIGNATURE_ROW,
                records.SIGNATURE_DESCRIPTOR_ROW,
                _numeric_table(specification, "signature_kind"),
            ),
        ),
        _grammar(
            "callable",
            records.CALLABLE_TYPE_ROW,
            "callable @",
            DeclareSymbol(SymbolDomain.CALLABLE),
            " flags(",
            _numeric(
                specification,
                records.CALLABLE_TYPE_ROW,
                "flags_u16",
                "callable_type_flag",
                (NumericSpelling("may_yield", "yieldable"),),
            ),
            ") depth(",
            _scalar(records.CALLABLE_TYPE_ROW, "nesting_depth_u16"),
            ") : @",
            _symbol(
                records.CALLABLE_TYPE_ROW,
                "signature_ordinal_u16",
                SymbolDomain.SIGNATURE,
            ),
        ),
        _grammar(
            "import_group",
            records.IMPORT_GROUP_ROW,
            "group @",
            DeclareSymbol(SymbolDomain.IMPORT_GROUP),
            " module_name(@",
            _symbol(
                records.IMPORT_GROUP_ROW,
                "module_name_string_u16",
                SymbolDomain.STRING,
            ),
            ") ",
            _body(
                "import",
                records.IMPORT_ENTRY_ROW,
                records.IMPORT_GROUP_ROW,
                count_field_name="entry_count_u32",
            ),
            keyword="group",
        ),
        _grammar(
            "import",
            records.IMPORT_ENTRY_ROW,
            "import @",
            DeclareSymbol(SymbolDomain.IMPORT, "flags_u16"),
            " name(@",
            _symbol(
                records.IMPORT_ENTRY_ROW,
                "symbol_name_string_u16",
                SymbolDomain.STRING,
            ),
            ") flags(",
            _numeric(
                specification,
                records.IMPORT_ENTRY_ROW,
                "flags_u16",
                "import_flag",
            ),
            ") : @",
            _symbol(
                records.IMPORT_ENTRY_ROW,
                "callable_type_ordinal_u16",
                SymbolDomain.CALLABLE,
            ),
        ),
        _grammar(
            "export",
            records.EXPORT_ROW,
            "export @",
            DeclareSymbol(SymbolDomain.EXPORT),
            " name(@",
            _symbol(records.EXPORT_ROW, "name_string_u16", SymbolDomain.STRING),
            ") : @",
            _symbol(
                records.EXPORT_ROW,
                "callable_type_ordinal_u16",
                SymbolDomain.CALLABLE,
            ),
            " = @",
            _symbol(
                records.EXPORT_ROW,
                "function_ordinal_u16",
                SymbolDomain.FUNCTION,
            ),
        ),
        _grammar(
            "function",
            records.FUNCTION_ROW,
            "func @",
            DeclareSymbol(SymbolDomain.FUNCTION),
            " : @",
            _symbol(
                records.FUNCTION_ROW,
                "callable_type_ordinal_u16",
                SymbolDomain.CALLABLE,
            ),
            " flags(",
            _numeric(
                specification,
                records.FUNCTION_ROW,
                "flags_u16",
                "function_flag",
                (NumericSpelling("may_yield", "yieldable"),),
            ),
            ") frame(local_bytes = ",
            _scalar(records.FUNCTION_ROW, "local_byte_length_u16"),
            ", value_registers = ",
            _scalar(records.FUNCTION_ROW, "value_register_count_u16"),
            ", ref_registers = ",
            _scalar(records.FUNCTION_ROW, "ref_register_count_u16"),
            ", function_registers = ",
            _scalar(records.FUNCTION_ROW, "function_register_count_u16"),
            ", local_refs = ",
            _scalar(records.FUNCTION_ROW, "local_ref_count_u32"),
            ", local_functions = ",
            _scalar(records.FUNCTION_ROW, "local_function_count_u32"),
            ") ",
            FunctionBody(records.FUNCTION_ROW, records.SWITCH_TARGET_ENTRY, "bytecode"),
            keyword="func",
        ),
        _grammar(
            "constant",
            records.CONSTANT_CELL,
            "constant @",
            DeclareSymbol(SymbolDomain.CONSTANT),
            " = ",
            _scalar(records.CONSTANT_CELL, "bits_u64", ScalarRadix.HEXADECIMAL),
        ),
        _grammar(
            "global_values",
            records.GLOBALS_HEADER,
            "values count(",
            _scalar(records.GLOBALS_HEADER, "value_count_u32"),
            ") immutable(",
            _scalar(records.GLOBALS_HEADER, "immutable_value_count_u32"),
            ")",
            keyword="values",
            row_mode=RowMode.SINGLETON,
        ),
        _grammar(
            "global_refs",
            records.GLOBALS_HEADER,
            "refs immutable(",
            _scalar(records.GLOBALS_HEADER, "immutable_ref_count_u32"),
            ") ",
            _body(
                "global_ref",
                records.GLOBAL_REF_DESCRIPTOR_ROW,
                records.GLOBALS_HEADER,
                count_field_name="ref_count_u32",
            ),
            keyword="refs",
            row_mode=RowMode.SINGLETON,
        ),
        _grammar(
            "global_functions",
            records.GLOBALS_HEADER,
            "functions immutable(",
            _scalar(records.GLOBALS_HEADER, "immutable_function_count_u32"),
            ") ",
            _body(
                "global_function",
                records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
                records.GLOBALS_HEADER,
                count_field_name="function_count_u32",
            ),
            keyword="functions",
            row_mode=RowMode.SINGLETON,
        ),
        _grammar(
            "global_ref",
            records.GLOBAL_REF_DESCRIPTOR_ROW,
            "global.ref @",
            DeclareSymbol(SymbolDomain.GLOBAL_REF),
            " : @",
            _symbol(
                records.GLOBAL_REF_DESCRIPTOR_ROW,
                "ref_type_ordinal_u16",
                SymbolDomain.REF_TYPE,
            ),
            " flags(",
            _numeric(
                specification,
                records.GLOBAL_REF_DESCRIPTOR_ROW,
                "flags_u16",
                "global_ref_flag",
            ),
            ")",
            keyword="global.ref",
        ),
        _grammar(
            "global_function",
            records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
            "global.func @",
            DeclareSymbol(SymbolDomain.GLOBAL_FUNCTION),
            " : @",
            _symbol(
                records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
                "callable_type_ordinal_u16",
                SymbolDomain.CALLABLE,
            ),
            " flags(",
            _numeric(
                specification,
                records.GLOBAL_FUNCTION_DESCRIPTOR_ROW,
                "flags_u16",
                "global_function_flag",
            ),
            ")",
            keyword="global.func",
        ),
        _grammar(
            "rodata",
            records.RODATA_BLOCK_DESCRIPTOR,
            "rodata @",
            DeclareSymbol(SymbolDomain.RODATA),
            " alignment(",
            _scalar(records.RODATA_BLOCK_DESCRIPTOR, "minimum_alignment_u32"),
            ") = ",
            HexBlob(
                "rodata",
                records.RODATA_BLOCK_DESCRIPTOR,
                "byte_length_u64",
                "minimum_alignment_u32",
            ),
        ),
        _grammar(
            "presentation_import",
            records.PRESENTATION_ENTRY_ROW,
            "import @",
            _symbol(
                records.PRESENTATION_ENTRY_ROW,
                "declaration_ordinal_u16",
                SymbolDomain.IMPORT,
            ),
            ConstantField(records.PRESENTATION_ENTRY_ROW, "declaration_kind_u16", 1),
            " documentation(",
            _nullable_symbol(
                records.PRESENTATION_ENTRY_ROW,
                "documentation_string_u16",
                SymbolDomain.STRING,
            ),
            ") authored_type(",
            _nullable_symbol(
                records.PRESENTATION_ENTRY_ROW,
                "authored_type_string_u16",
                SymbolDomain.STRING,
            ),
            ") ",
            _body(
                "presentation_field",
                records.PRESENTATION_FIELD_ROW,
                records.PRESENTATION_ENTRY_ROW,
                base_field_name="field_base_u32",
            ),
            keyword="import",
        ),
        _grammar(
            "presentation_export",
            records.PRESENTATION_ENTRY_ROW,
            "export @",
            _symbol(
                records.PRESENTATION_ENTRY_ROW,
                "declaration_ordinal_u16",
                SymbolDomain.EXPORT,
            ),
            ConstantField(records.PRESENTATION_ENTRY_ROW, "declaration_kind_u16", 2),
            " documentation(",
            _nullable_symbol(
                records.PRESENTATION_ENTRY_ROW,
                "documentation_string_u16",
                SymbolDomain.STRING,
            ),
            ") authored_type(",
            _nullable_symbol(
                records.PRESENTATION_ENTRY_ROW,
                "authored_type_string_u16",
                SymbolDomain.STRING,
            ),
            ") ",
            _body(
                "presentation_field",
                records.PRESENTATION_FIELD_ROW,
                records.PRESENTATION_ENTRY_ROW,
                base_field_name="field_base_u32",
            ),
            keyword="export",
        ),
        _grammar(
            "presentation_field",
            records.PRESENTATION_FIELD_ROW,
            "field name(",
            _nullable_symbol(
                records.PRESENTATION_FIELD_ROW,
                "name_string_u16",
                SymbolDomain.STRING,
            ),
            ") authored_type(",
            _nullable_symbol(
                records.PRESENTATION_FIELD_ROW,
                "authored_type_string_u16",
                SymbolDomain.STRING,
            ),
            ")",
            keyword="field",
        ),
        _grammar(
            "metadata_module",
            records.METADATA_HEADER,
            "module ",
            _body(
                "metadata_entry",
                records.METADATA_ENTRY_ROW,
                records.METADATA_HEADER,
                count_field_name="module_entry_count_u32",
            ),
            keyword="module",
            row_mode=RowMode.SINGLETON,
        ),
        _grammar(
            "metadata_import",
            records.METADATA_SCOPE_ROW,
            "import @",
            _symbol(
                records.METADATA_SCOPE_ROW,
                "declaration_ordinal_u16",
                SymbolDomain.IMPORT,
            ),
            " ",
            _body(
                "metadata_entry",
                records.METADATA_ENTRY_ROW,
                records.METADATA_SCOPE_ROW,
                base_field_name="entry_base_u32",
                count_field_name="entry_count_u16",
            ),
            keyword="import",
        ),
        _grammar(
            "metadata_export",
            records.METADATA_SCOPE_ROW,
            "export @",
            _symbol(
                records.METADATA_SCOPE_ROW,
                "declaration_ordinal_u16",
                SymbolDomain.EXPORT,
            ),
            " ",
            _body(
                "metadata_entry",
                records.METADATA_ENTRY_ROW,
                records.METADATA_SCOPE_ROW,
                base_field_name="entry_base_u32",
                count_field_name="entry_count_u16",
            ),
            keyword="export",
        ),
        _grammar(
            "metadata_entry",
            records.METADATA_ENTRY_ROW,
            "metadata @",
            _symbol(
                records.METADATA_ENTRY_ROW,
                "key_string_u16",
                SymbolDomain.STRING,
            ),
            " : ",
            _numeric(
                specification,
                records.METADATA_ENTRY_ROW,
                "value_type_u16",
                "metadata_value_type",
            ),
            " = ",
            MetadataValue(
                "metadata",
                records.METADATA_ENTRY_ROW,
                "value_type_u16",
                records.METADATA_VALUE_OFFSET,
            ),
            keyword="metadata",
        ),
    )
