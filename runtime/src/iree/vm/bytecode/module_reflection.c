// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/module_reflection.h"

#include "iree/vm/bytecode/module_reader.h"

static uint32_t iree_vm_bytecode_import_group_base(
    const iree_vm_bytecode_module_t* module, uint32_t group_ordinal) {
  uint32_t base = 0;
  for (uint32_t i = 0; i < group_ordinal; ++i) {
    base += module->layout.imports.groups[i].entry_count_u32;
  }
  return base;
}

static uint32_t iree_vm_bytecode_find_import_group(
    const iree_vm_bytecode_module_t* module, uint32_t import_ordinal,
    uint32_t* out_entry_ordinal) {
  uint32_t base = 0;
  for (uint32_t i = 0; i < module->layout.imports.group_count; ++i) {
    const uint32_t count = module->layout.imports.groups[i].entry_count_u32;
    if (import_ordinal - base < count) {
      *out_entry_ordinal = import_ordinal;
      return i;
    }
    base += count;
  }
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
    const iree_vm_bytecode_module_t* module, uint16_t ordinal) {
  if (!module->layout.metadata.header) return 0;
  const iree_vm_bytecode_v0_metadata_scope_row_t* scope =
      iree_vm_bytecode_find_metadata_scope(
          module->layout.metadata.import_scopes,
          module->layout.metadata.header->import_scope_count_u32, ordinal);
  return scope ? scope->entry_count_u16 : 0;
}

static iree_host_size_t iree_vm_bytecode_export_metadata_count(
    const iree_vm_bytecode_module_t* module, uint16_t ordinal) {
  if (!module->layout.metadata.header) return 0;
  const iree_vm_bytecode_v0_metadata_scope_row_t* scope =
      iree_vm_bytecode_find_metadata_scope(
          module->layout.metadata.export_scopes,
          module->layout.metadata.header->export_scope_count_u32, ordinal);
  return scope ? scope->entry_count_u16 : 0;
}

void iree_vm_bytecode_module_query_import_group(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group) {
  const iree_vm_bytecode_module_t* module =
      iree_vm_bytecode_module_cast_const(base_module);
  const iree_vm_bytecode_v0_import_group_row_t* row =
      &module->layout.imports.groups[ordinal];
  const iree_vm_module_import_group_t group = {
      iree_vm_bytecode_string_at(&module->layout.strings,
                                 row->module_name_string_u16),
      iree_vm_bytecode_import_group_base(module, (uint32_t)ordinal),
      row->entry_count_u32,
  };
  *out_group = group;
}

void iree_vm_bytecode_module_query_import(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import) {
  const iree_vm_bytecode_module_t* module =
      iree_vm_bytecode_module_cast_const(base_module);
  uint32_t entry_ordinal = 0;
  const uint32_t group_ordinal = iree_vm_bytecode_find_import_group(
      module, (uint32_t)ordinal, &entry_ordinal);
  const iree_vm_bytecode_v0_import_group_row_t* group =
      &module->layout.imports.groups[group_ordinal];
  const iree_vm_bytecode_v0_import_entry_row_t* entry =
      &module->layout.imports.entries[entry_ordinal];
  const iree_vm_module_import_declaration_t import_declaration = {
      iree_vm_bytecode_string_at(&module->layout.strings,
                                 group->module_name_string_u16),
      iree_vm_bytecode_string_at(&module->layout.strings,
                                 entry->symbol_name_string_u16),
      entry->callable_type_ordinal_u16,
      entry->flags_u16,
      iree_vm_bytecode_import_metadata_count(module, (uint16_t)ordinal),
  };
  *out_import = import_declaration;
}

void iree_vm_bytecode_module_query_export(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export) {
  const iree_vm_bytecode_module_t* module =
      iree_vm_bytecode_module_cast_const(base_module);
  const iree_vm_bytecode_v0_export_row_t* row =
      &module->layout.exports.rows[ordinal];
  const iree_vm_module_export_declaration_t export_declaration = {
      iree_vm_bytecode_string_at(&module->layout.strings, row->name_string_u16),
      row->callable_type_ordinal_u16,
      row->function_ordinal_u16,
      iree_vm_bytecode_export_metadata_count(module, (uint16_t)ordinal),
  };
  *out_export = export_declaration;
}

void iree_vm_bytecode_module_query_callable_type(
    const iree_vm_module_t* base_module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type) {
  const iree_vm_bytecode_module_t* module =
      iree_vm_bytecode_module_cast_const(base_module);
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      &module->layout.callable_types.rows[ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &module->layout.signatures.rows[callable_type->signature_ordinal_u16];
  const iree_vm_module_signature_type_t* descriptors =
      (const iree_vm_module_signature_type_t*)
          iree_vm_bytecode_signature_descriptors(
              &module->layout.signatures, callable_type->signature_ordinal_u16);
  const iree_host_size_t argument_count =
      iree_vm_bytecode_signature_argument_count(signature);
  const iree_host_size_t result_count =
      iree_vm_bytecode_signature_result_count(signature);
  const iree_vm_module_callable_type_declaration_t declaration = {
      {{descriptors, argument_count},
       {descriptors + argument_count, result_count}},
      callable_type->flags_u16,
  };
  *out_callable_type = declaration;
}

static const iree_vm_bytecode_v0_presentation_entry_row_t*
iree_vm_bytecode_find_presentation(const iree_vm_bytecode_module_t* module,
                                   uint16_t declaration_kind,
                                   uint16_t declaration_ordinal) {
  uint32_t low = 0;
  uint32_t high = module->layout.presentation.entry_count;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    const iree_vm_bytecode_v0_presentation_entry_row_t* row =
        &module->layout.presentation.entries[middle];
    if (row->declaration_kind_u16 < declaration_kind ||
        (row->declaration_kind_u16 == declaration_kind &&
         row->declaration_ordinal_u16 < declaration_ordinal)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low >= module->layout.presentation.entry_count) return NULL;
  const iree_vm_bytecode_v0_presentation_entry_row_t* row =
      &module->layout.presentation.entries[low];
  return row->declaration_kind_u16 == declaration_kind &&
                 row->declaration_ordinal_u16 == declaration_ordinal
             ? row
             : NULL;
}

void iree_vm_bytecode_module_query_presentation(
    const iree_vm_module_t* base_module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation) {
  const iree_vm_bytecode_module_t* module =
      iree_vm_bytecode_module_cast_const(base_module);
  uint16_t callable_type_ordinal = 0;
  if (query->declaration.kind == IREE_VM_MODULE_DECLARATION_KIND_IMPORT) {
    callable_type_ordinal =
        module->layout.imports.entries[query->declaration.ordinal]
            .callable_type_ordinal_u16;
  } else {
    callable_type_ordinal =
        module->layout.exports.rows[query->declaration.ordinal]
            .callable_type_ordinal_u16;
  }
  const iree_vm_bytecode_v0_callable_type_row_t* callable_type =
      &module->layout.callable_types.rows[callable_type_ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      &module->layout.signatures.rows[callable_type->signature_ordinal_u16];
  const uint32_t field_count =
      iree_vm_bytecode_signature_argument_count(signature) +
      iree_vm_bytecode_signature_result_count(signature);
  const iree_vm_module_presentation_t empty_presentation = {
      0,
      iree_string_view_empty(),
      iree_string_view_empty(),
  };
  *out_presentation = empty_presentation;
  if (query->fields.count != field_count ||
      (field_count != 0 && !query->fields.data)) {
    return;
  }

  for (uint32_t i = 0; i < field_count; ++i) {
    query->fields.data[i].name = iree_string_view_empty();
    query->fields.data[i].authored_type = iree_string_view_empty();
  }
  const iree_vm_bytecode_v0_presentation_entry_row_t* row =
      iree_vm_bytecode_find_presentation(module,
                                         (uint16_t)query->declaration.kind,
                                         (uint16_t)query->declaration.ordinal);
  if (!row) return;

  for (uint32_t i = 0; i < field_count; ++i) {
    const iree_vm_bytecode_v0_presentation_field_row_t* field =
        &module->layout.presentation.fields[row->field_base_u32 + i];
    query->fields.data[i].name = iree_vm_bytecode_nullable_string_at(
        &module->layout.strings, field->name_string_u16);
    query->fields.data[i].authored_type = iree_vm_bytecode_nullable_string_at(
        &module->layout.strings, field->authored_type_string_u16);
  }
  const iree_vm_module_presentation_t presentation = {
      0,
      iree_vm_bytecode_nullable_string_at(&module->layout.strings,
                                          row->documentation_string_u16),
      iree_vm_bytecode_nullable_string_at(&module->layout.strings,
                                          row->authored_type_string_u16),
  };
  *out_presentation = presentation;
}

void iree_vm_bytecode_module_metadata_by_ordinal(
    const iree_vm_module_t* base_module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry) {
  const iree_vm_bytecode_module_t* module =
      iree_vm_bytecode_module_cast_const(base_module);
  uint32_t entry_base = 0;
  if (query->scope.kind == IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT) {
    const iree_vm_bytecode_v0_metadata_scope_row_t* scope =
        iree_vm_bytecode_find_metadata_scope(
            module->layout.metadata.import_scopes,
            module->layout.metadata.header->import_scope_count_u32,
            (uint16_t)query->scope.ordinal);
    entry_base = scope->entry_base_u32;
  } else if (query->scope.kind == IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT) {
    const iree_vm_bytecode_v0_metadata_scope_row_t* scope =
        iree_vm_bytecode_find_metadata_scope(
            module->layout.metadata.export_scopes,
            module->layout.metadata.header->export_scope_count_u32,
            (uint16_t)query->scope.ordinal);
    entry_base = scope->entry_base_u32;
  }
  const uint32_t entry_ordinal = entry_base + (uint32_t)query->ordinal;
  const iree_vm_bytecode_v0_metadata_entry_row_t* entry =
      &module->layout.metadata.entries[entry_ordinal];
  const iree_host_size_t value_begin =
      (iree_host_size_t)module->layout.metadata.value_offsets[entry_ordinal];
  const iree_host_size_t value_end =
      (iree_host_size_t)
          module->layout.metadata.value_offsets[entry_ordinal + 1];
  const iree_vm_metadata_entry_t result = {
      iree_vm_bytecode_string_at(&module->layout.strings,
                                 entry->key_string_u16),
      {entry->value_type_u16,
       iree_make_const_byte_span(
           module->layout.metadata.value_data + value_begin,
           value_end - value_begin)},
  };
  *out_entry = result;
}
