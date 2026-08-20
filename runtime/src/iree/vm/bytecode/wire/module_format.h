// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Module-container constants and natural-layout wire declarations.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_MODULE_FORMAT_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_MODULE_FORMAT_H_

#include <stdint.h>

enum {
  IREE_VM_BYTECODE_CORE_MAJOR = 0,
  IREE_VM_BYTECODE_CORE_MINOR = 0,
  IREE_VM_BYTECODE_IMAGE_ALIGNMENT = 8,
  IREE_VM_BYTECODE_SECTION_ALIGNMENT = 8,
};

typedef uint16_t iree_vm_bytecode_section_type_t;
enum {
  IREE_VM_BYTECODE_SECTION_REQUIREMENTS = 0x0001,
  IREE_VM_BYTECODE_SECTION_STRINGS = 0x0002,
  IREE_VM_BYTECODE_SECTION_REF_TYPES = 0x0003,
  IREE_VM_BYTECODE_SECTION_SIGNATURES = 0x0004,
  IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES = 0x0005,
  IREE_VM_BYTECODE_SECTION_IMPORTS = 0x0006,
  IREE_VM_BYTECODE_SECTION_EXPORTS = 0x0007,
  IREE_VM_BYTECODE_SECTION_FUNCTIONS = 0x0008,
  IREE_VM_BYTECODE_SECTION_CONSTANTS = 0x0009,
  IREE_VM_BYTECODE_SECTION_GLOBALS = 0x000A,
  IREE_VM_BYTECODE_SECTION_RODATA = 0x000B,
  IREE_VM_BYTECODE_SECTION_PRESENTATION = 0x000C,
  IREE_VM_BYTECODE_SECTION_METADATA = 0x000D,
};

enum {
  IREE_VM_BYTECODE_SECTION_REQUIREMENTS_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_REQUIREMENTS_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_STRINGS_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_STRINGS_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_REF_TYPES_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_REF_TYPES_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_SIGNATURES_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_SIGNATURES_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_IMPORTS_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_IMPORTS_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_EXPORTS_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_EXPORTS_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_FUNCTIONS_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_FUNCTIONS_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_CONSTANTS_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_CONSTANTS_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_GLOBALS_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_GLOBALS_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_RODATA_REQUIRED_FLAGS = 0x0000,
  IREE_VM_BYTECODE_SECTION_RODATA_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_PRESENTATION_REQUIRED_FLAGS = 0x0001,
  IREE_VM_BYTECODE_SECTION_PRESENTATION_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SECTION_METADATA_REQUIRED_FLAGS = 0x0001,
  IREE_VM_BYTECODE_SECTION_METADATA_SINCE_MINOR = 0,
};

// Module-format signature kind values.
typedef uint16_t iree_vm_bytecode_signature_kind_t;
enum {
  // Invalid in a descriptor.
  IREE_VM_BYTECODE_SIGNATURE_KIND_INVALID = 0x0000,
  // Low eight integer bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_I8 = 0x0001,
  // Low 16 integer bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_I16 = 0x0002,
  // Low 32 integer bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_I32 = 0x0003,
  // Complete 64 integer bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_I64 = 0x0004,
  // Low eight floating-point bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_F8E4M3FN = 0x0005,
  // Low eight floating-point bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_F8E5M2 = 0x0006,
  // Low 16 IEEE binary16 bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_F16 = 0x0007,
  // Low 16 bfloat16 bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_BF16 = 0x0008,
  // Low 32 IEEE binary32 bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_F32 = 0x0009,
  // Complete 64 IEEE binary64 bits.
  IREE_VM_BYTECODE_SIGNATURE_KIND_F64 = 0x000A,
  // Exact ref type in the second field.
  IREE_VM_BYTECODE_SIGNATURE_KIND_REF = 0x0100,
  // Exact callable type in the second field.
  IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION = 0x0200,
};

enum {
  IREE_VM_BYTECODE_SIGNATURE_KIND_INVALID_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_I8_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_I16_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_I32_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_I64_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_F8E4M3FN_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_F8E5M2_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_F16_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_BF16_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_F32_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_F64_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_REF_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION_SINCE_MINOR = 0,
};

// Module-format section flag values.
typedef uint16_t iree_vm_bytecode_section_flags_t;
enum iree_vm_bytecode_section_flag_bits_e {
  // Unknown readers may skip the section.
  IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE = 0x0001,
};

enum {
  IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE_SINCE_MINOR = 0,
};

// Module-format callable type flag values.
typedef uint16_t iree_vm_bytecode_callable_type_flags_t;
enum iree_vm_bytecode_callable_type_flag_bits_e {
  // The callable contract permits suspension.
  IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD = 0x0001,
};

enum {
  IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD_SINCE_MINOR = 0,
};

// Module-format import flag values.
typedef uint16_t iree_vm_bytecode_import_flags_t;
enum iree_vm_bytecode_import_flag_bits_e {
  // An absent target module or export is permitted.
  IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL = 0x0001,
};

enum {
  IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL_SINCE_MINOR = 0,
};

// Module-format function flag values.
typedef uint16_t iree_vm_bytecode_function_flags_t;
enum iree_vm_bytecode_function_flag_bits_e {
  // The function may yield.
  IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD = 0x0001,
};

enum {
  IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD_SINCE_MINOR = 0,
};

// Module-format global ref flag values.
typedef uint16_t iree_vm_bytecode_global_ref_flags_t;
enum iree_vm_bytecode_global_ref_flag_bits_e {
  // The ref global may remain canonical null.
  IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE = 0x0001,
};

enum {
  IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE_SINCE_MINOR = 0,
};

// Module-format global function flag values.
typedef uint16_t iree_vm_bytecode_global_function_flags_t;
enum iree_vm_bytecode_global_function_flag_bits_e {
  // The function global may remain canonical null.
  IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE = 0x0001,
};

enum {
  IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE_SINCE_MINOR = 0,
};

// Module-format presentation declaration kind values.
typedef uint16_t iree_vm_bytecode_presentation_declaration_kind_t;
enum {
  // Invalid in a presentation row.
  IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_INVALID = 0x0000,
  // An import declaration.
  IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT = 0x0001,
  // An export declaration.
  IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_EXPORT = 0x0002,
};

enum {
  IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_INVALID_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_IMPORT_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_EXPORT_SINCE_MINOR = 0,
};

// Module-format metadata value type values.
typedef uint16_t iree_vm_bytecode_metadata_value_type_t;
enum {
  // Invalid in a metadata entry.
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_INVALID = 0x0000,
  // One canonical Boolean byte.
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL = 0x0001,
  // Little-endian signed 64-bit bits.
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64 = 0x0002,
  // Little-endian unsigned 64-bit bits.
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64 = 0x0003,
  // Little-endian IEEE binary64 bits with every bit pattern preserved.
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_F64 = 0x0004,
  // Length-delimited UTF-8 bytes.
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8 = 0x0005,
  // Opaque bytes.
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BYTES = 0x0006,
};

enum {
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_INVALID_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BOOL_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_I64_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_U64_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_F64_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8_SINCE_MINOR = 0,
  IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BYTES_SINCE_MINOR = 0,
};

// Exact image_header.magic_u8 bytes; compare only the declared length because
// the C literal has its normal terminating zero.
#define IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_BYTES "\x49\x52\x45\x45\x56\x4D\x00\x00"
#define IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_LENGTH 8

// Fixed module wire record image_header.
typedef struct {
  // Exact eight-byte IREE VM image magic.
  uint8_t magic_u8[8];
  // Incompatible core container and ISA version.
  uint16_t core_major_u16;
  // Minimum compatible core minor version.
  uint16_t core_required_minor_u16;
  // Number of section-directory rows.
  uint16_t section_count_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_bytecode_v0_image_header_t;

// Fixed module wire record section_directory_row.
typedef struct {
  // Architectural section identifier.
  uint16_t section_type_u16;
  // Flags interpreted by the owning section authority.
  uint16_t section_flags_u16;
  // Reserved zero word preserving native u64 alignment.
  uint32_t reserved_u32;
  // Exact section payload length in bytes.
  uint64_t byte_length_u64;
} iree_vm_bytecode_v0_section_directory_row_t;

// Fixed module wire record requirement_row.
typedef struct {
  // Non-core architectural page identifier.
  uint16_t page_id_u16;
  // Exact incompatible page version.
  uint16_t major_u16;
  // Minimum compatible page minor version.
  uint16_t required_minor_u16;
} iree_vm_bytecode_v0_requirement_row_t;

// Fixed module wire record strings_header.
typedef struct {
  // Number of length-delimited strings.
  uint32_t string_count_u32;
} iree_vm_bytecode_v0_strings_header_t;

// Fixed module wire record string_offset.
// Byte offset into the trailing UTF-8 area.
typedef uint32_t iree_vm_bytecode_v0_string_offset_t;

// Fixed module wire record ref_types_header.
typedef struct {
  // Number of namespace groups.
  uint32_t group_count_u32;
} iree_vm_bytecode_v0_ref_types_header_t;

// Fixed module wire record ref_type_group_row.
typedef struct {
  // Nonempty type-namespace string ordinal.
  uint16_t namespace_string_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // Nonzero count of local type entries.
  uint32_t entry_count_u32;
} iree_vm_bytecode_v0_ref_type_group_row_t;

// Fixed module wire record ref_type_entry_row.
typedef struct {
  // Nonempty local type-name string ordinal.
  uint16_t type_name_string_u16;
  // Required type features; zero in version zero.
  uint16_t required_flags_u16;
} iree_vm_bytecode_v0_ref_type_entry_row_t;

// Fixed module wire record signatures_header.
typedef struct {
  // Number of source-ordered logical signatures.
  uint32_t signature_count_u32;
} iree_vm_bytecode_v0_signatures_header_t;

// Fixed module wire record signature_row.
typedef struct {
  // Canonical running base in the descriptor array.
  uint32_t descriptor_base_u32;
  // Value argument count.
  uint16_t argument_value_count_u16;
  // Value result count.
  uint16_t result_value_count_u16;
  // Ref argument count.
  uint16_t argument_ref_count_u16;
  // Ref result count.
  uint16_t result_ref_count_u16;
  // Function argument count.
  uint16_t argument_function_count_u16;
  // Function result count.
  uint16_t result_function_count_u16;
} iree_vm_bytecode_v0_signature_row_t;

// Fixed module wire record signature_descriptor_row.
typedef struct {
  // Architectural scalar, REF, or FUNCTION kind.
  uint16_t kind_u16;
  // Exact ref/callable type ordinal, or zero for scalars.
  uint16_t type_ordinal_u16;
} iree_vm_bytecode_v0_signature_descriptor_row_t;

// Fixed module wire record callable_types_header.
typedef struct {
  // Number of structural callable declarations.
  uint32_t callable_type_count_u32;
} iree_vm_bytecode_v0_callable_types_header_t;

// Fixed module wire record callable_type_row.
typedef struct {
  // Exact source-ordered signature.
  uint16_t signature_ordinal_u16;
  // Callable behavior permission flags.
  uint16_t flags_u16;
} iree_vm_bytecode_v0_callable_type_row_t;

// Fixed module wire record imports_header.
typedef struct {
  // Number of target-module groups.
  uint32_t group_count_u32;
} iree_vm_bytecode_v0_imports_header_t;

// Fixed module wire record import_group_row.
typedef struct {
  // Nonempty target-module name string ordinal.
  uint16_t module_name_string_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // Nonzero count of imported symbols.
  uint32_t entry_count_u32;
} iree_vm_bytecode_v0_import_group_row_t;

// Fixed module wire record import_entry_row.
typedef struct {
  // Nonempty target export-name string ordinal.
  uint16_t symbol_name_string_u16;
  // Exact local callable-type requirement.
  uint16_t callable_type_ordinal_u16;
  // Import declaration flags.
  uint16_t flags_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_bytecode_v0_import_entry_row_t;

// Fixed module wire record exports_header.
typedef struct {
  // Number of public export rows.
  uint32_t export_count_u32;
} iree_vm_bytecode_v0_exports_header_t;

// Fixed module wire record export_row.
typedef struct {
  // Nonempty public export-name string ordinal.
  uint16_t name_string_u16;
  // Exact public callable type.
  uint16_t callable_type_ordinal_u16;
  // Valid module-local function ordinal.
  uint16_t function_ordinal_u16;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_bytecode_v0_export_row_t;

// Fixed module wire record functions_header.
typedef struct {
  // Number of bytecode functions.
  uint32_t function_count_u32;
} iree_vm_bytecode_v0_functions_header_t;

// Fixed module wire record function_row.
typedef struct {
  // Exact function signature ordinal.
  uint16_t signature_ordinal_u16;
  // Function behavior flags.
  uint16_t flags_u16;
  // Canonical byte offset from the bytecode payload base.
  uint32_t bytecode_offset_u32;
  // Nonzero record-stream length in bytes.
  uint32_t bytecode_length_u32;
  // Canonical entry base in the switch-target array.
  uint32_t switch_target_base_u32;
  // Aggregate switch-target entries owned by the function.
  uint32_t switch_target_entry_count_u32;
  // Complete function-local byte storage extent.
  uint16_t local_byte_length_u16;
  // Value-register count.
  uint16_t value_register_count_u16;
  // Ref-register count.
  uint16_t ref_register_count_u16;
  // Function-register count.
  uint16_t function_register_count_u16;
  // Function-local owning ref-slot count.
  uint32_t local_ref_count_u32;
  // Function-local non-owning function-slot count.
  uint32_t local_function_count_u32;
  // Reserved zero words.
  uint32_t reserved_u32[3];
} iree_vm_bytecode_v0_function_row_t;

// Fixed module wire record switch_target_entry.
// Function-relative target offset in four-byte words.
typedef uint32_t iree_vm_bytecode_v0_switch_target_entry_t;

// Fixed module wire record constant_cell.
// Arbitrary little-endian value-cell bits.
typedef uint64_t iree_vm_bytecode_v0_constant_cell_t;

// Fixed module wire record globals_header.
typedef struct {
  // Total module-local value-global count.
  uint32_t value_count_u32;
  // Dense immutable value-global prefix length.
  uint32_t immutable_value_count_u32;
  // Total module-local ref-global count.
  uint32_t ref_count_u32;
  // Dense immutable ref-global prefix length.
  uint32_t immutable_ref_count_u32;
  // Total module-local function-global count.
  uint32_t function_count_u32;
  // Dense immutable function-global prefix length.
  uint32_t immutable_function_count_u32;
  // Reserved zero words.
  uint32_t reserved_u32[2];
} iree_vm_bytecode_v0_globals_header_t;

// Fixed module wire record global_ref_descriptor_row.
typedef struct {
  // Exact module-local ref-type ordinal.
  uint16_t ref_type_ordinal_u16;
  // Ref-global behavior flags.
  uint16_t flags_u16;
} iree_vm_bytecode_v0_global_ref_descriptor_row_t;

// Fixed module wire record global_function_descriptor_row.
typedef struct {
  // Exact module-local callable-type ordinal.
  uint16_t callable_type_ordinal_u16;
  // Function-global behavior flags.
  uint16_t flags_u16;
} iree_vm_bytecode_v0_global_function_descriptor_row_t;

// Fixed module wire record rodata_header.
typedef struct {
  // Number of direct rodata ordinals.
  uint32_t block_count_u32;
  // Canonical zero padding.
  uint32_t zero_padding_u32;
} iree_vm_bytecode_v0_rodata_header_t;

// Fixed module wire record rodata_block_length.
// Exact byte length of one rodata block.
typedef uint64_t iree_vm_bytecode_v0_rodata_block_length_t;

// Fixed module wire record presentation_header.
typedef struct {
  // Number of sparse import/export presentation rows.
  uint32_t entry_count_u32;
} iree_vm_bytecode_v0_presentation_header_t;

// Fixed module wire record presentation_entry_row.
typedef struct {
  // Valid ordinal in the selected declaration domain.
  uint16_t declaration_ordinal_u16;
  // Import or export declaration kind.
  uint16_t declaration_kind_u16;
  // Nullable documentation string ordinal.
  uint16_t documentation_string_u16;
  // Nullable authored function-type string ordinal.
  uint16_t authored_type_string_u16;
  // Canonical running base in the presentation field array.
  uint32_t field_base_u32;
} iree_vm_bytecode_v0_presentation_entry_row_t;

// Fixed module wire record presentation_field_row.
typedef struct {
  // Nullable argument or result name string ordinal.
  uint16_t name_string_u16;
  // Nullable authored field-type string ordinal.
  uint16_t authored_type_string_u16;
} iree_vm_bytecode_v0_presentation_field_row_t;

// Fixed module wire record metadata_header.
typedef struct {
  // Number of module-scope metadata entries.
  uint32_t module_entry_count_u32;
  // Number of nonempty import metadata scopes.
  uint32_t import_scope_count_u32;
  // Number of nonempty export metadata scopes.
  uint32_t export_scope_count_u32;
  // Total metadata entry count across all scopes.
  uint32_t total_entry_count_u32;
} iree_vm_bytecode_v0_metadata_header_t;

// Fixed module wire record metadata_scope_row.
typedef struct {
  // Valid import or export declaration ordinal.
  uint16_t declaration_ordinal_u16;
  // Nonzero number of entries in the scope.
  uint16_t entry_count_u16;
  // Canonical running base in the metadata entry array.
  uint32_t entry_base_u32;
} iree_vm_bytecode_v0_metadata_scope_row_t;

// Fixed module wire record metadata_entry_row.
typedef struct {
  // Nonempty metadata-key string ordinal.
  uint16_t key_string_u16;
  // Nonzero open metadata value-type identifier.
  uint16_t value_type_u16;
} iree_vm_bytecode_v0_metadata_entry_row_t;

// Fixed module wire record metadata_value_offset.
// Byte offset into the trailing metadata value area.
typedef uint64_t iree_vm_bytecode_v0_metadata_value_offset_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_MODULE_FORMAT_H_
// clang-format on
