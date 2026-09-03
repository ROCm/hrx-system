# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core VM module wire record declarations."""

from __future__ import annotations

from iree.vm.bytecode.spec.module import FieldRuleUse, WireField, WireRecord
from iree.vm.bytecode.spec.module.rules import FieldRule, OrdinalDomain
from iree.vm.bytecode.spec.schema import (
    U8,
    U16,
    U32,
    U64,
    Field,
    RuleKind,
    ScalarEncoding,
)
from iree.vm.bytecode.spec.version import CORE_0


def _field(
    name: str,
    encoding: ScalarEncoding,
    summary: str,
    rule: RuleKind | FieldRuleUse,
    *,
    element_count: int = 1,
) -> WireField:
    return WireField(Field(name, encoding, summary, element_count), rule)


def _record(
    name: str,
    summary: str,
    contract: str,
    fields: tuple[WireField, ...],
) -> WireRecord:
    return WireRecord(
        name,
        f"iree_vm_bytecode_v0_{name}_t",
        CORE_0,
        summary,
        contract,
        fields,
    )


IMAGE_HEADER = _record(
    "image_header",
    "Identifies a VM image and its required Core version.",
    (
        "This fixed prefix is read before any section-dependent allocation or "
        "pointer formation. The exact magic rejects unrelated inputs. The major "
        "version must match the loader, the required minor must not exceed loader "
        "support, and the section count sizes the immediately following directory."
    ),
    (
        _field(
            "magic_u8",
            U8,
            "Exact eight-byte IREE VM image magic.",
            FieldRuleUse(FieldRule.EXACT_BYTES, data=b"IREEVM\x00\x00"),
            element_count=8,
        ),
        _field(
            "core_major_u16",
            U16,
            "Incompatible Core container and ISA version.",
            FieldRule.CORE_MAJOR,
        ),
        _field(
            "core_required_minor_u16",
            U16,
            "Minimum compatible Core minor version.",
            FieldRule.CORE_REQUIRED_MINOR,
        ),
        _field(
            "section_count_u16",
            U16,
            "Number of section-directory rows.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "zero_padding_u16",
            U16,
            "Canonical zero padding.",
            FieldRule.ZERO,
        ),
    ),
)

SECTION_DIRECTORY_ROW = _record(
    "section_directory_row",
    "Describes one ordered module section payload.",
    (
        "Rows immediately follow the image header in strictly increasing section "
        "type order. Payload offsets are derived by aligning a checked running "
        "cursor; they are never serialized."
    ),
    (
        _field(
            "section_type_u16",
            U16,
            "Architectural section identifier.",
            FieldRule.SECTION_TYPE,
        ),
        _field(
            "section_flags_u16",
            U16,
            "Flags interpreted by the owning section authority.",
            FieldRule.SECTION_FLAGS,
        ),
        _field(
            "payload_alignment_u32",
            U32,
            "Image-relative minimum alignment of the payload.",
            FieldRuleUse(FieldRule.BYTE_ALIGNMENT, values=(8,)),
        ),
        _field(
            "byte_length_u64",
            U64,
            "Exact section payload length in bytes.",
            FieldRule.SECTION_BYTE_LENGTH,
        ),
    ),
)

REQUIREMENT_ROW = _record(
    "requirement_row",
    "Declares one required architectural extension page version.",
    "Rows are strictly ordered by page ID and describe non-Core page authorities.",
    (
        _field(
            "page_id_u16",
            U16,
            "Non-Core architectural page identifier.",
            FieldRule.NONCORE_PAGE,
        ),
        _field(
            "major_u16",
            U16,
            "Exact incompatible page version.",
            FieldRule.PAGE_MAJOR,
        ),
        _field(
            "required_minor_u16",
            U16,
            "Minimum compatible page minor version.",
            FieldRule.PAGE_REQUIRED_MINOR,
        ),
    ),
)

STRINGS_HEADER = _record(
    "strings_header",
    "Counts the module's length-delimited UTF-8 strings.",
    "The count sizes the immediately following count-plus-one offset array.",
    (
        _field(
            "string_count_u32",
            U32,
            "Number of length-delimited strings.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65535)),
        ),
    ),
)

STRING_OFFSET = _record(
    "string_offset",
    "Locates one boundary in the trailing UTF-8 byte area.",
    (
        "The count-plus-one offsets are monotonic, begin at zero, and end at the "
        "exact trailing byte length."
    ),
    (
        _field(
            "byte_offset_u32",
            U32,
            "Byte offset into the trailing UTF-8 area.",
            FieldRule.STRING_OFFSET,
        ),
    ),
)

REF_TYPES_HEADER = _record(
    "ref_types_header",
    "Counts canonical reference-type namespace groups.",
    "The count sizes the namespace-group row array.",
    (
        _field(
            "group_count_u32",
            U32,
            "Number of namespace groups.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65535)),
        ),
    ),
)

REF_TYPE_GROUP_ROW = _record(
    "ref_type_group_row",
    "Names one reference-type namespace and its local entries.",
    "Groups and their entries are nonempty and strictly ordered by raw UTF-8 bytes.",
    (
        _field(
            "namespace_string_u16",
            U16,
            "Nonempty type-namespace string ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.STRING_NONEMPTY),
        ),
        _field(
            "zero_padding_u16",
            U16,
            "Canonical zero padding.",
            FieldRule.ZERO,
        ),
        _field(
            "entry_count_u32",
            U32,
            "Nonzero count of local type entries.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 0xFFFFFFFF)),
        ),
    ),
)

REF_TYPE_ENTRY_ROW = _record(
    "ref_type_entry_row",
    "Names one reference type within its namespace group.",
    "Traversal order across all groups defines the flat module-local type ordinal.",
    (
        _field(
            "type_name_string_u16",
            U16,
            "Nonempty local type-name string ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.STRING_NONEMPTY),
        ),
        _field(
            "required_flags_u16",
            U16,
            "Required type features; zero in Core 0.0.",
            FieldRule.ZERO,
        ),
    ),
)

SIGNATURES_HEADER = _record(
    "signatures_header",
    "Counts source-ordered logical function signatures.",
    (
        "The count sizes the signature-row array. Signature ordinal order is stable "
        "within the module and is used directly by functions and callable types."
    ),
    (
        _field(
            "signature_count_u32",
            U32,
            "Number of source-ordered logical signatures.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65536)),
        ),
    ),
)

SIGNATURE_ROW = _record(
    "signature_row",
    "Locates and partitions one logical signature's descriptors.",
    (
        "descriptor_base_u32 is the canonical running base in the section-wide "
        "descriptor array. The six counts partition arguments and results by "
        "physical value, ref, and function carriers while preserving logical order."
    ),
    (
        _field(
            "descriptor_base_u32",
            U32,
            "Canonical running base in the descriptor array.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "argument_value_count_u16", U16, "Value argument count.", FieldRule.ANY_BITS
        ),
        _field(
            "result_value_count_u16", U16, "Value result count.", FieldRule.ANY_BITS
        ),
        _field(
            "argument_ref_count_u16", U16, "Ref argument count.", FieldRule.ANY_BITS
        ),
        _field("result_ref_count_u16", U16, "Ref result count.", FieldRule.ANY_BITS),
        _field(
            "argument_function_count_u16",
            U16,
            "Function argument count.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "result_function_count_u16",
            U16,
            "Function result count.",
            FieldRule.ANY_BITS,
        ),
    ),
)

SIGNATURE_DESCRIPTOR_ROW = _record(
    "signature_descriptor_row",
    "Declares one logical signature field's kind and exact type.",
    (
        "Scalar kinds require a zero type ordinal. Ref and function kinds require an "
        "in-range module-local type ordinal from their respective type table."
    ),
    (
        _field(
            "kind_u16",
            U16,
            "Architectural scalar, ref, or function kind.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "type_ordinal_u16",
            U16,
            "Exact ref or callable type ordinal, or zero for scalars.",
            FieldRuleUse(FieldRule.SIGNATURE_DESCRIPTOR, fields=("kind_u16",)),
        ),
    ),
)

CALLABLE_TYPES_HEADER = _record(
    "callable_types_header",
    "Counts canonical structural callable types.",
    "The count sizes the immediately following callable-type row array.",
    (
        _field(
            "callable_type_count_u32",
            U32,
            "Number of structural callable declarations.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65536)),
        ),
    ),
)

CALLABLE_TYPE_ROW = _record(
    "callable_type_row",
    "Defines one canonical callable contract.",
    (
        "Rows are unique and strictly ordered by nesting depth, structural "
        "signature, then flags. Nested callable types may name only earlier rows."
    ),
    (
        _field(
            "signature_ordinal_u16",
            U16,
            "Exact source-ordered signature.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.SIGNATURE),
        ),
        _field(
            "flags_u16",
            U16,
            "Callable behavior permission flags.",
            FieldRuleUse(FieldRule.ALLOWED_BITS, values=(1,)),
        ),
        _field(
            "nesting_depth_u16",
            U16,
            "Exact maximum nested callable depth.",
            FieldRule.ANY_BITS,
        ),
        _field("reserved_u16", U16, "Reserved zero bits.", FieldRule.ZERO),
    ),
)

IMPORTS_HEADER = _record(
    "imports_header",
    "Counts target-module import groups.",
    "The count sizes the target-module group row array.",
    (
        _field(
            "group_count_u32",
            U32,
            "Number of target-module groups.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65535)),
        ),
    ),
)

IMPORT_GROUP_ROW = _record(
    "import_group_row",
    "Names one target module and its imported symbols.",
    "Groups are nonempty and strictly ordered by target-module UTF-8 bytes.",
    (
        _field(
            "module_name_string_u16",
            U16,
            "Nonempty target-module name string ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.STRING_NONEMPTY),
        ),
        _field("zero_padding_u16", U16, "Canonical zero padding.", FieldRule.ZERO),
        _field(
            "entry_count_u32",
            U32,
            "Nonzero count of imported symbols.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 0xFFFFFFFF)),
        ),
    ),
)

IMPORT_ENTRY_ROW = _record(
    "import_entry_row",
    "Declares one imported callable symbol.",
    (
        "Traversal order across all groups defines the flat import ordinal. Optional "
        "imports permit absence but never relax callable-type compatibility."
    ),
    (
        _field(
            "symbol_name_string_u16",
            U16,
            "Nonempty target export-name string ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.STRING_NONEMPTY),
        ),
        _field(
            "callable_type_ordinal_u16",
            U16,
            "Exact local callable-type requirement.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.CALLABLE_TYPE),
        ),
        _field(
            "flags_u16",
            U16,
            "Import declaration flags.",
            FieldRuleUse(FieldRule.ALLOWED_BITS, values=(1,)),
        ),
        _field("zero_padding_u16", U16, "Canonical zero padding.", FieldRule.ZERO),
    ),
)

EXPORTS_HEADER = _record(
    "exports_header",
    "Counts public export declarations.",
    "The count sizes the immediately following export row array.",
    (
        _field(
            "export_count_u32",
            U32,
            "Number of public export rows.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65535)),
        ),
    ),
)

EXPORT_ROW = _record(
    "export_row",
    "Publishes one named module-local function.",
    (
        "Rows are strictly ordered by public-name bytes. Multiple names may alias "
        "one function while retaining independent public metadata."
    ),
    (
        _field(
            "name_string_u16",
            U16,
            "Nonempty public export-name string ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.STRING_NONEMPTY),
        ),
        _field(
            "callable_type_ordinal_u16",
            U16,
            "Exact public callable type.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.CALLABLE_TYPE),
        ),
        _field(
            "function_ordinal_u16",
            U16,
            "Valid module-local function ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.FUNCTION),
        ),
        _field("zero_padding_u16", U16, "Canonical zero padding.", FieldRule.ZERO),
    ),
)

FUNCTIONS_HEADER = _record(
    "functions_header",
    "Counts bytecode function declarations.",
    "The count sizes the immediately following function row array.",
    (
        _field(
            "function_count_u32",
            U32,
            "Number of bytecode functions.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65536)),
        ),
    ),
)

FUNCTION_ROW = _record(
    "function_row",
    "Locates one function stream and declares its complete frame high waters.",
    (
        "Switch-target and bytecode ranges are canonical running prefixes in "
        "function ordinal order. Register and local counts cover the signature, all "
        "decoded instructions, and the largest outgoing call packet."
    ),
    (
        _field(
            "callable_type_ordinal_u16",
            U16,
            "Canonical callable type implemented by the function.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.CALLABLE_TYPE),
        ),
        _field(
            "flags_u16",
            U16,
            "Function behavior flags.",
            FieldRuleUse(FieldRule.ALLOWED_BITS, values=(3,)),
        ),
        _field(
            "bytecode_offset_u32",
            U32,
            "Canonical byte offset from the bytecode payload base.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "bytecode_length_u32",
            U32,
            "Nonzero four-byte-framed record-stream length.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 0x7FFFFFFF)),
        ),
        _field(
            "switch_target_base_u32",
            U32,
            "Canonical entry base in the switch-target array.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "switch_target_entry_count_u32",
            U32,
            "Aggregate switch-target entries owned by the function.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "local_byte_length_u16",
            U16,
            "Complete function-local byte storage extent.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "value_register_count_u16",
            U16,
            "Value-register count.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 256)),
        ),
        _field(
            "ref_register_count_u16",
            U16,
            "Ref-register count.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 256)),
        ),
        _field(
            "function_register_count_u16",
            U16,
            "Function-register count.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 256)),
        ),
        _field(
            "local_ref_count_u32",
            U32,
            "Function-local owning ref-slot count.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "local_function_count_u32",
            U32,
            "Function-local non-owning function-slot count.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "block_count_u32",
            U32,
            "Exact number of decoded control.block records.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65536)),
        ),
        _field(
            "reserved_u32",
            U32,
            "Reserved zero words.",
            FieldRule.ZERO,
            element_count=2,
        ),
    ),
)

SWITCH_TARGET_ENTRY = _record(
    "switch_target_entry",
    "Names one function-local switch target.",
    "The four-byte-word offset must resolve to an exact control.block record.",
    (
        _field(
            "target_word_offset_u32",
            U32,
            "Function-relative target offset in four-byte words.",
            FieldRule.SWITCH_TARGET,
        ),
    ),
)

CONSTANT_CELL = _record(
    "constant_cell",
    "Stores one untyped constant value cell.",
    "The cell carries arbitrary little-endian bits and has no implicit type.",
    (
        _field(
            "bits_u64",
            U64,
            "Arbitrary little-endian value-cell bits.",
            FieldRule.ANY_BITS,
        ),
    ),
)

GLOBALS_HEADER = _record(
    "globals_header",
    "Declares all three private process-global domains.",
    (
        "Each domain's immutable globals form a dense prefix of its total count. Ref "
        "and function descriptor rows follow this header in domain order."
    ),
    (
        _field(
            "value_count_u32",
            U32,
            "Total module-local value-global count.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "immutable_value_count_u32",
            U32,
            "Dense immutable value-global prefix length.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "ref_count_u32",
            U32,
            "Total module-local ref-global count.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "immutable_ref_count_u32",
            U32,
            "Dense immutable ref-global prefix length.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "function_count_u32",
            U32,
            "Total module-local function-global count.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "immutable_function_count_u32",
            U32,
            "Dense immutable function-global prefix length.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "reserved_u32",
            U32,
            "Reserved zero words.",
            FieldRule.ZERO,
            element_count=2,
        ),
    ),
)

GLOBAL_REF_DESCRIPTOR_ROW = _record(
    "global_ref_descriptor_row",
    "Declares one typed reference global.",
    "Rows occur in ref-global ordinal order and carry the exact local ref type.",
    (
        _field(
            "ref_type_ordinal_u16",
            U16,
            "Exact module-local ref-type ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.REF_TYPE),
        ),
        _field(
            "flags_u16",
            U16,
            "Ref-global behavior flags.",
            FieldRuleUse(FieldRule.ALLOWED_BITS, values=(1,)),
        ),
    ),
)

GLOBAL_FUNCTION_DESCRIPTOR_ROW = _record(
    "global_function_descriptor_row",
    "Declares one typed function global.",
    "Rows occur in function-global ordinal order and carry the exact callable type.",
    (
        _field(
            "callable_type_ordinal_u16",
            U16,
            "Exact module-local callable-type ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.CALLABLE_TYPE),
        ),
        _field(
            "flags_u16",
            U16,
            "Function-global behavior flags.",
            FieldRuleUse(FieldRule.ALLOWED_BITS, values=(1,)),
        ),
    ),
)

RODATA_HEADER = _record(
    "rodata_header",
    "Counts module read-only data blocks.",
    "The count sizes the immediately following block descriptor array.",
    (
        _field(
            "block_count_u32",
            U32,
            "Number of direct rodata ordinals.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65536)),
        ),
        _field("zero_padding_u32", U32, "Canonical zero padding.", FieldRule.ZERO),
    ),
)

RODATA_BLOCK_DESCRIPTOR = _record(
    "rodata_block_descriptor",
    "Declares one aligned read-only data block.",
    (
        "Block payloads follow all descriptors in ordinal order. Each is placed by "
        "aligning a checked running cursor to its declared minimum alignment."
    ),
    (
        _field(
            "byte_length_u64",
            U64,
            "Exact byte length of one rodata block.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "minimum_alignment_u32",
            U32,
            "Minimum host-address alignment of the block.",
            FieldRuleUse(FieldRule.BYTE_ALIGNMENT, values=(1,)),
        ),
        _field("reserved_u32", U32, "Reserved zero word.", FieldRule.ZERO),
    ),
)

PRESENTATION_HEADER = _record(
    "presentation_header",
    "Counts sparse import and export presentation entries.",
    "The count sizes the declaration presentation row array.",
    (
        _field(
            "entry_count_u32",
            U32,
            "Number of sparse import and export presentation rows.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 131071)),
        ),
    ),
)

PRESENTATION_ENTRY_ROW = _record(
    "presentation_entry_row",
    "Associates human-facing function data with one public declaration.",
    (
        "Rows are ordered by declaration kind and ordinal. field_base_u32 is the "
        "canonical running base into signature-derived field rows."
    ),
    (
        _field(
            "declaration_ordinal_u16",
            U16,
            "Valid ordinal in the selected declaration domain.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "declaration_kind_u16",
            U16,
            "Import or export declaration kind.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 2)),
        ),
        _field(
            "documentation_string_u16",
            U16,
            "Nullable documentation string ordinal.",
            FieldRuleUse(
                FieldRule.ORDINAL_OR_NULL,
                values=(0xFFFF,),
                data=OrdinalDomain.STRING,
            ),
        ),
        _field(
            "authored_type_string_u16",
            U16,
            "Nullable authored function-type string ordinal.",
            FieldRuleUse(
                FieldRule.ORDINAL_OR_NULL,
                values=(0xFFFF,),
                data=OrdinalDomain.STRING,
            ),
        ),
        _field(
            "field_base_u32",
            U32,
            "Canonical running base in the presentation field array.",
            FieldRule.ANY_BITS,
        ),
    ),
)

PRESENTATION_FIELD_ROW = _record(
    "presentation_field_row",
    "Carries optional authored data for one machine argument or result.",
    "Continuation fields of an expanded authored aggregate use canonical null strings.",
    (
        _field(
            "name_string_u16",
            U16,
            "Nullable anchored argument or result name string ordinal.",
            FieldRuleUse(
                FieldRule.ORDINAL_OR_NULL,
                values=(0xFFFF,),
                data=OrdinalDomain.STRING,
            ),
        ),
        _field(
            "authored_type_string_u16",
            U16,
            "Nullable anchored authored field-type string ordinal.",
            FieldRuleUse(
                FieldRule.ORDINAL_OR_NULL,
                values=(0xFFFF,),
                data=OrdinalDomain.STRING,
            ),
        ),
    ),
)

METADATA_HEADER = _record(
    "metadata_header",
    "Partitions module, import, and export metadata entries.",
    (
        "Module entries occur first. Nonempty import and export scope rows partition "
        "the remainder through canonical running entry bases."
    ),
    (
        _field(
            "module_entry_count_u32",
            U32,
            "Number of module-scope metadata entries.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65535)),
        ),
        _field(
            "import_scope_count_u32",
            U32,
            "Number of nonempty import metadata scopes.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65536)),
        ),
        _field(
            "export_scope_count_u32",
            U32,
            "Number of nonempty export metadata scopes.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 65535)),
        ),
        _field(
            "total_entry_count_u32",
            U32,
            "Total metadata entry count across all scopes.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 0xFFFFFFFF)),
        ),
    ),
)

METADATA_SCOPE_ROW = _record(
    "metadata_scope_row",
    "Locates one nonempty import or export metadata scope.",
    "Scope rows are ordered by declaration ordinal within their declaration domain.",
    (
        _field(
            "declaration_ordinal_u16",
            U16,
            "Valid import or export declaration ordinal.",
            FieldRule.ANY_BITS,
        ),
        _field(
            "entry_count_u16",
            U16,
            "Nonzero number of entries in the scope.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65535)),
        ),
        _field(
            "entry_base_u32",
            U32,
            "Canonical running base in the metadata entry array.",
            FieldRule.ANY_BITS,
        ),
    ),
)

METADATA_ENTRY_ROW = _record(
    "metadata_entry_row",
    "Declares one keyed typed metadata value.",
    "Keys are strictly byte-ordered within each scope; unknown nonzero types are valid.",
    (
        _field(
            "key_string_u16",
            U16,
            "Nonempty metadata-key string ordinal.",
            FieldRuleUse(FieldRule.ORDINAL, data=OrdinalDomain.STRING_NONEMPTY),
        ),
        _field(
            "value_type_u16",
            U16,
            "Nonzero open metadata value-type identifier.",
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 65535)),
        ),
    ),
)

METADATA_VALUE_OFFSET = _record(
    "metadata_value_offset",
    "Locates one boundary in the trailing metadata value area.",
    "The count-plus-one offsets are monotonic, begin at zero, and end at the value tail.",
    (
        _field(
            "byte_offset_u64",
            U64,
            "Byte offset into the trailing metadata value area.",
            FieldRule.ANY_BITS,
        ),
    ),
)
