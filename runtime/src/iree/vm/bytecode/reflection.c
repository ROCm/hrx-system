// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/reflection.h"

// Wire signature descriptors intentionally overlay generic module signature
// types so queries can return stable image-backed declaration spans.
static_assert(sizeof(iree_vm_bytecode_v0_signature_descriptor_row_t) ==
                  sizeof(iree_vm_module_signature_type_t),
              "wire and generic signature fields must have matching layouts");
static_assert(offsetof(iree_vm_bytecode_v0_signature_descriptor_row_t,
                       kind_u16) ==
                  offsetof(iree_vm_module_signature_type_t, kind),
              "wire and generic signature kinds must have matching offsets");
static_assert(offsetof(iree_vm_bytecode_v0_signature_descriptor_row_t,
                       type_ordinal_u16) ==
                  offsetof(iree_vm_module_signature_type_t, type_ordinal),
              "wire and generic signature ordinals must have matching offsets");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_I8,
                           IREE_VM_SCALAR_TYPE_I8,
                           "wire and generic i8 kinds must match");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_F64,
                           IREE_VM_SCALAR_TYPE_F64,
                           "wire and generic f64 kinds must match");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_REF,
                           IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF,
                           "wire and generic ref kinds must match");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION,
                           IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION,
                           "wire and generic function kinds must match");

static uint32_t iree_vm_bytecode_import_group_base(
    const iree_vm_bytecode_image_t* image, uint32_t group_ordinal) {
  uint32_t base = 0;
  for (uint32_t i = 0; i < group_ordinal; ++i) {
    base += image->layout.imports.groups[i].entry_count_u32;
  }
  return base;
}

static uint32_t iree_vm_bytecode_find_import_group(
    const iree_vm_bytecode_image_t* image, uint32_t import_ordinal) {
  uint32_t base = 0;
  for (uint32_t i = 0; i < image->layout.imports.group_count; ++i) {
    const uint32_t count = image->layout.imports.groups[i].entry_count_u32;
    if (import_ordinal >= base && import_ordinal - base < count) return i;
    base += count;
  }
  IREE_ASSERT_UNREACHABLE("validated import ordinal is not in a group");
  return 0;
}

static const iree_vm_bytecode_v0_metadata_scope_row_t*
iree_vm_bytecode_find_metadata_scope(
    const iree_vm_bytecode_v0_metadata_scope_row_t* scopes,
    uint32_t scope_count, uint16_t declaration_ordinal) {
  uint32_t low = 0;
  uint32_t high = scope_count;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    if (scopes[middle].declaration_ordinal_u16 < declaration_ordinal) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < scope_count &&
                 scopes[low].declaration_ordinal_u16 == declaration_ordinal
             ? &scopes[low]
             : NULL;
}

static iree_host_size_t iree_vm_bytecode_import_metadata_count(
    const iree_vm_bytecode_image_t* image, uint16_t ordinal) {
  if (!image->layout.metadata.header) return 0;
  const iree_vm_bytecode_v0_metadata_scope_row_t* scope =
      iree_vm_bytecode_find_metadata_scope(
          image->layout.metadata.import_scopes,
          image->layout.metadata.header->import_scope_count_u32, ordinal);
  return scope ? scope->entry_count_u16 : 0;
}

static iree_host_size_t iree_vm_bytecode_export_metadata_count(
    const iree_vm_bytecode_image_t* image, uint16_t ordinal) {
  if (!image->layout.metadata.header) return 0;
  const iree_vm_bytecode_v0_metadata_scope_row_t* scope =
      iree_vm_bytecode_find_metadata_scope(
          image->layout.metadata.export_scopes,
          image->layout.metadata.header->export_scope_count_u32, ordinal);
  return scope ? scope->entry_count_u16 : 0;
}

void iree_vm_bytecode_reflection_query_import_group(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group) {
  const iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module_const(base_module);
  const iree_vm_bytecode_v0_import_group_row_t* row =
      &image->layout.imports.groups[ordinal];
  const iree_vm_module_import_group_t group = {
      iree_vm_bytecode_string_at(&image->layout.strings,
                                 row->module_name_string_u16),
      iree_vm_bytecode_import_group_base(image, (uint32_t)ordinal),
      row->entry_count_u32,
  };
  *out_group = group;
}

void iree_vm_bytecode_reflection_query_import(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import) {
  const iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module_const(base_module);
  const uint32_t group_ordinal =
      iree_vm_bytecode_find_import_group(image, (uint32_t)ordinal);
  const iree_vm_bytecode_v0_import_group_row_t* group =
      &image->layout.imports.groups[group_ordinal];
  const iree_vm_bytecode_v0_import_entry_row_t* entry =
      &image->layout.imports.entries[ordinal];
  const iree_vm_module_import_declaration_t import_declaration = {
      iree_vm_bytecode_string_at(&image->layout.strings,
                                 group->module_name_string_u16),
      iree_vm_bytecode_string_at(&image->layout.strings,
                                 entry->symbol_name_string_u16),
      entry->callable_type_ordinal_u16,
      entry->flags_u16,
      iree_vm_bytecode_import_metadata_count(image, (uint16_t)ordinal),
  };
  *out_import = import_declaration;
}

void iree_vm_bytecode_reflection_query_export(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export) {
  const iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module_const(base_module);
  const iree_vm_bytecode_v0_export_row_t* row =
      &image->layout.exports.rows[ordinal];
  const iree_vm_module_export_declaration_t export_declaration = {
      iree_vm_bytecode_string_at(&image->layout.strings, row->name_string_u16),
      row->callable_type_ordinal_u16,
      row->function_ordinal_u16,
      iree_vm_bytecode_export_metadata_count(image, (uint16_t)ordinal),
  };
  *out_export = export_declaration;
}

void iree_vm_bytecode_reflection_query_callable_type(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type) {
  const iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module_const(base_module);
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      &image->layout.callable_types.rows[ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &image->layout.signatures.rows[callable_type->signature_ordinal_u16];
  const iree_vm_module_signature_type_t* types =
      (const iree_vm_module_signature_type_t*)
          iree_vm_bytecode_signature_descriptors(
              &image->layout.signatures, callable_type->signature_ordinal_u16);
  const iree_host_size_t argument_count =
      iree_vm_bytecode_signature_argument_count(signature);
  const iree_host_size_t result_count =
      iree_vm_bytecode_signature_result_count(signature);
  const iree_vm_module_callable_type_declaration_t declaration = {
      {{types, (uint16_t)argument_count, signature->argument_value_count_u16,
        signature->argument_ref_count_u16,
        signature->argument_function_count_u16},
       {types + argument_count, (uint16_t)result_count,
        signature->result_value_count_u16, signature->result_ref_count_u16,
        signature->result_function_count_u16}},
      callable_type->flags_u16,
      callable_type->nesting_depth_u16,
      callable_type->reserved_u16,
  };
  *out_callable_type = declaration;
}

static const iree_vm_bytecode_v0_presentation_entry_row_t*
iree_vm_bytecode_find_presentation(const iree_vm_bytecode_image_t* image,
                                   uint16_t declaration_kind,
                                   uint16_t declaration_ordinal) {
  uint32_t low = 0;
  uint32_t high = image->layout.presentation.entry_count;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    const iree_vm_bytecode_v0_presentation_entry_row_t* row =
        &image->layout.presentation.entries[middle];
    if (row->declaration_kind_u16 < declaration_kind ||
        (row->declaration_kind_u16 == declaration_kind &&
         row->declaration_ordinal_u16 < declaration_ordinal)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low >= image->layout.presentation.entry_count) return NULL;
  const iree_vm_bytecode_v0_presentation_entry_row_t* row =
      &image->layout.presentation.entries[low];
  return row->declaration_kind_u16 == declaration_kind &&
                 row->declaration_ordinal_u16 == declaration_ordinal
             ? row
             : NULL;
}

void iree_vm_bytecode_reflection_query_presentation(
    const iree_vm_module_t* base_module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation) {
  const iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module_const(base_module);
  uint16_t callable_type_ordinal = 0;
  if (query->declaration.kind == IREE_VM_MODULE_DECLARATION_KIND_IMPORT) {
    callable_type_ordinal =
        image->layout.imports.entries[query->declaration.ordinal]
            .callable_type_ordinal_u16;
  } else {
    callable_type_ordinal =
        image->layout.exports.rows[query->declaration.ordinal]
            .callable_type_ordinal_u16;
  }
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      &image->layout.callable_types.rows[callable_type_ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &image->layout.signatures.rows[callable_type->signature_ordinal_u16];
  const uint32_t field_count =
      iree_vm_bytecode_signature_argument_count(signature) +
      iree_vm_bytecode_signature_result_count(signature);
  *out_presentation = (iree_vm_module_presentation_t){
      0,
      iree_string_view_empty(),
      iree_string_view_empty(),
  };
  if (query->fields.count != field_count ||
      (field_count != 0 && !query->fields.data)) {
    return;
  }

  for (uint32_t i = 0; i < field_count; ++i) {
    query->fields.data[i].name = iree_string_view_empty();
    query->fields.data[i].authored_type = iree_string_view_empty();
  }
  const iree_vm_bytecode_v0_presentation_entry_row_t* row =
      iree_vm_bytecode_find_presentation(image,
                                         (uint16_t)query->declaration.kind,
                                         (uint16_t)query->declaration.ordinal);
  if (!row) return;

  for (uint32_t i = 0; i < field_count; ++i) {
    const iree_vm_bytecode_v0_presentation_field_row_t* field =
        &image->layout.presentation.fields[row->field_base_u32 + i];
    query->fields.data[i].name = iree_vm_bytecode_nullable_string_at(
        &image->layout.strings, field->name_string_u16);
    query->fields.data[i].authored_type = iree_vm_bytecode_nullable_string_at(
        &image->layout.strings, field->authored_type_string_u16);
  }
  *out_presentation = (iree_vm_module_presentation_t){
      0,
      iree_vm_bytecode_nullable_string_at(&image->layout.strings,
                                          row->documentation_string_u16),
      iree_vm_bytecode_nullable_string_at(&image->layout.strings,
                                          row->authored_type_string_u16),
  };
}

void iree_vm_bytecode_reflection_metadata_by_ordinal(
    const iree_vm_module_t* base_module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry) {
  const iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module_const(base_module);
  uint32_t entry_base = 0;
  if (query->scope.kind == IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT) {
    entry_base = iree_vm_bytecode_find_metadata_scope(
                     image->layout.metadata.import_scopes,
                     image->layout.metadata.header->import_scope_count_u32,
                     (uint16_t)query->scope.ordinal)
                     ->entry_base_u32;
  } else if (query->scope.kind == IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT) {
    entry_base = iree_vm_bytecode_find_metadata_scope(
                     image->layout.metadata.export_scopes,
                     image->layout.metadata.header->export_scope_count_u32,
                     (uint16_t)query->scope.ordinal)
                     ->entry_base_u32;
  }
  const uint32_t entry_ordinal = entry_base + (uint32_t)query->ordinal;
  const iree_vm_bytecode_v0_metadata_entry_row_t* entry =
      &image->layout.metadata.entries[entry_ordinal];
  const iree_host_size_t value_begin =
      (iree_host_size_t)image->layout.metadata.value_offsets[entry_ordinal]
          .byte_offset_u64;
  const iree_host_size_t value_end =
      (iree_host_size_t)image->layout.metadata.value_offsets[entry_ordinal + 1]
          .byte_offset_u64;
  *out_entry = (iree_vm_metadata_entry_t){
      iree_vm_bytecode_string_at(&image->layout.strings, entry->key_string_u16),
      {entry->value_type_u16,
       iree_make_const_byte_span(
           image->layout.metadata.value_data + value_begin,
           value_end - value_begin)},
  };
}
