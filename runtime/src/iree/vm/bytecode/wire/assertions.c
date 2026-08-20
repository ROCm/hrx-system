// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Compile-time checks for module-container wire declarations.
// clang-format off

#include "iree/base/alignment.h"
#include "iree/vm/bytecode/wire/module_format.h"

static_assert(sizeof(iree_vm_bytecode_v0_image_header_t) == 16, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_image_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_image_header_t, magic_u8) == 0, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_image_header_t, core_major_u16) == 8, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_image_header_t, core_required_minor_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_image_header_t, section_count_u16) == 12, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_image_header_t, zero_padding_u16) == 14, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_section_directory_row_t) == 16, "wire size");
static_assert(8 % iree_alignof(iree_vm_bytecode_v0_section_directory_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_section_directory_row_t, section_type_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_section_directory_row_t, section_flags_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_section_directory_row_t, reserved_u32) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_section_directory_row_t, byte_length_u64) == 8,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_requirement_row_t) == 6, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_requirement_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_requirement_row_t, page_id_u16) == 0, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_requirement_row_t, major_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_requirement_row_t, required_minor_u16) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_strings_header_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_strings_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_strings_header_t, string_count_u32) == 0, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_string_offset_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_string_offset_t) == 0, "wire alignment");

static_assert(sizeof(iree_vm_bytecode_v0_ref_types_header_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_ref_types_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_ref_types_header_t, group_count_u32) == 0,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_ref_type_group_row_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_ref_type_group_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_ref_type_group_row_t, namespace_string_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_ref_type_group_row_t, zero_padding_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_ref_type_group_row_t, entry_count_u32) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_ref_type_entry_row_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_ref_type_entry_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_ref_type_entry_row_t, type_name_string_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_ref_type_entry_row_t, required_flags_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_signatures_header_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_signatures_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_signatures_header_t, signature_count_u32) == 0,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_signature_row_t) == 16, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_signature_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_signature_row_t, descriptor_base_u32) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_signature_row_t, argument_value_count_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_signature_row_t, result_value_count_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_signature_row_t, argument_ref_count_u16) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_signature_row_t, result_ref_count_u16) == 10,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_signature_row_t, argument_function_count_u16) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_signature_row_t, result_function_count_u16) == 14,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_signature_descriptor_row_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_signature_descriptor_row_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_signature_descriptor_row_t, kind_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_signature_descriptor_row_t, type_ordinal_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_callable_types_header_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_callable_types_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_callable_types_header_t, callable_type_count_u32) == 0,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_callable_type_row_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_callable_type_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_callable_type_row_t, signature_ordinal_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_callable_type_row_t, flags_u16) == 2, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_imports_header_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_imports_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_imports_header_t, group_count_u32) == 0, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_import_group_row_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_import_group_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_import_group_row_t, module_name_string_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_import_group_row_t, zero_padding_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_import_group_row_t, entry_count_u32) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_import_entry_row_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_import_entry_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_import_entry_row_t, symbol_name_string_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_import_entry_row_t, callable_type_ordinal_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_import_entry_row_t, flags_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_import_entry_row_t, zero_padding_u16) == 6,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_exports_header_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_exports_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_exports_header_t, export_count_u32) == 0, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_export_row_t) == 8, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_export_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_export_row_t, name_string_u16) == 0, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_export_row_t, callable_type_ordinal_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_export_row_t, function_ordinal_u16) == 4, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_export_row_t, zero_padding_u16) == 6, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_functions_header_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_functions_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_functions_header_t, function_count_u32) == 0,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_function_row_t) == 48, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_function_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, signature_ordinal_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, flags_u16) == 2, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, bytecode_offset_u32) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, bytecode_length_u32) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, switch_target_base_u32) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, switch_target_entry_count_u32) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, local_byte_length_u16) == 20,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, value_register_count_u16) == 22,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, ref_register_count_u16) == 24,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, function_register_count_u16) == 26,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, local_ref_count_u32) == 28,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, local_function_count_u32) == 32,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_function_row_t, reserved_u32) == 36, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_switch_target_entry_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_switch_target_entry_t) == 0, "wire alignment");

static_assert(sizeof(iree_vm_bytecode_v0_constant_cell_t) == 8, "wire size");
static_assert(8 % iree_alignof(iree_vm_bytecode_v0_constant_cell_t) == 0, "wire alignment");

static_assert(sizeof(iree_vm_bytecode_v0_globals_header_t) == 32, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_globals_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_globals_header_t, value_count_u32) == 0, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_globals_header_t, immutable_value_count_u32) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_globals_header_t, ref_count_u32) == 8, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_globals_header_t, immutable_ref_count_u32) == 12,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_globals_header_t, function_count_u32) == 16,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_globals_header_t, immutable_function_count_u32) == 20,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_globals_header_t, reserved_u32) == 24, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_global_ref_descriptor_row_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_global_ref_descriptor_row_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_global_ref_descriptor_row_t, ref_type_ordinal_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_global_ref_descriptor_row_t, flags_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_global_function_descriptor_row_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_global_function_descriptor_row_t) == 0,
              "wire alignment");
static_assert(
    offsetof(iree_vm_bytecode_v0_global_function_descriptor_row_t, callable_type_ordinal_u16) == 0,
    "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_global_function_descriptor_row_t, flags_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_rodata_header_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_rodata_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_rodata_header_t, block_count_u32) == 0, "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_rodata_header_t, zero_padding_u32) == 4, "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_rodata_block_length_t) == 8, "wire size");
static_assert(8 % iree_alignof(iree_vm_bytecode_v0_rodata_block_length_t) == 0, "wire alignment");

static_assert(sizeof(iree_vm_bytecode_v0_presentation_header_t) == 4, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_presentation_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_presentation_header_t, entry_count_u32) == 0,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_presentation_entry_row_t) == 12, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_presentation_entry_row_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_presentation_entry_row_t, declaration_ordinal_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_presentation_entry_row_t, declaration_kind_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_presentation_entry_row_t, documentation_string_u16) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_presentation_entry_row_t, authored_type_string_u16) == 6,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_presentation_entry_row_t, field_base_u32) == 8,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_presentation_field_row_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_presentation_field_row_t) == 0,
              "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_presentation_field_row_t, name_string_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_presentation_field_row_t, authored_type_string_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_metadata_header_t) == 16, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_metadata_header_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_header_t, module_entry_count_u32) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_header_t, import_scope_count_u32) == 4,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_header_t, export_scope_count_u32) == 8,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_header_t, total_entry_count_u32) == 12,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_metadata_scope_row_t) == 8, "wire size");
static_assert(4 % iree_alignof(iree_vm_bytecode_v0_metadata_scope_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_scope_row_t, declaration_ordinal_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_scope_row_t, entry_count_u16) == 2,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_scope_row_t, entry_base_u32) == 4,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_metadata_entry_row_t) == 4, "wire size");
static_assert(2 % iree_alignof(iree_vm_bytecode_v0_metadata_entry_row_t) == 0, "wire alignment");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_entry_row_t, key_string_u16) == 0,
              "wire offset");
static_assert(offsetof(iree_vm_bytecode_v0_metadata_entry_row_t, value_type_u16) == 2,
              "wire offset");

static_assert(sizeof(iree_vm_bytecode_v0_metadata_value_offset_t) == 8, "wire size");
static_assert(8 % iree_alignof(iree_vm_bytecode_v0_metadata_value_offset_t) == 0, "wire alignment");

// clang-format on
