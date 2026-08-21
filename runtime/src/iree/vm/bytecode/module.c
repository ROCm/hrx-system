// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/module.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/vm/bytecode/interpreter.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/module_storage.h"
#include "iree/vm/bytecode/verification.h"

//===----------------------------------------------------------------------===//
// Image and rodata lifetime
//===----------------------------------------------------------------------===//

static void iree_vm_bytecode_image_retain(iree_vm_bytecode_image_t* image) {
  if (!image) return;
  iree_atomic_ref_count_inc(&image->ref_count);
}

static void iree_vm_bytecode_image_release(iree_vm_bytecode_image_t* image) {
  if (!image) return;
  if (iree_atomic_ref_count_dec(&image->ref_count) != 1) return;
  const iree_vm_bytecode_module_storage_t storage = image->storage;
  const iree_allocator_t host_allocator = image->host_allocator;
  iree_allocator_free(storage.deallocator, (void*)storage.contents.data);
  iree_allocator_free(host_allocator, image);
}

static void iree_vm_bytecode_rodata_release(void* user_data,
                                            iree_byte_span_t storage) {
  (void)storage;
  iree_vm_bytecode_image_release((iree_vm_bytecode_image_t*)user_data);
}

static void iree_vm_bytecode_module_destroy(iree_vm_module_t* base_module) {
  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  for (uint32_t i = 0; i < module->layout.rodata.count; ++i) {
    iree_vm_buffer_release(&module->rodata_roots[i]);
  }
  iree_vm_bytecode_image_release(module->image);
}

//===----------------------------------------------------------------------===//
// Process state
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_bytecode_module_attach_state(
    iree_vm_module_t* base_module, iree_byte_span_t zeroed_storage,
    iree_allocator_t host_allocator) {
  (void)base_module;
  (void)zeroed_storage;
  (void)host_allocator;
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_module_seal_state(
    iree_vm_module_t* base_module, iree_byte_span_t storage) {
  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  const iree_vm_bytecode_v0_globals_header_t* globals =
      module->layout.globals.header;
  if (!globals) return iree_ok_status();

  iree_vm_bytecode_process_state_t* state =
      iree_vm_bytecode_process_state(storage.data);
  if (state->construction_state != IREE_VM_BYTECODE_CONSTRUCTION_STATE_OPEN) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "bytecode process state is already sealed");
  }

  const uint64_t* value_set_bits =
      iree_vm_bytecode_process_value_set_bits(module, storage.data);
  for (uint32_t i = 0; i < globals->immutable_value_count_u32; ++i) {
    if (!iree_vm_bytecode_bit_test(value_set_bits, i)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "immutable value global %" PRIu32 " is unset", i);
    }
  }

  const iree_vm_ref_t* refs =
      iree_vm_bytecode_process_refs(module, storage.data);
  const uint64_t* ref_set_bits =
      iree_vm_bytecode_process_ref_set_bits(module, storage.data);
  for (uint32_t i = 0; i < globals->ref_count_u32; ++i) {
    const iree_vm_bytecode_v0_global_ref_descriptor_row_t* descriptor =
        &module->layout.globals.refs[i];
    const bool is_nullable = iree_any_bit_set(
        descriptor->flags_u16, IREE_VM_BYTECODE_GLOBAL_REF_FLAG_NULLABLE);
    const bool is_set = i >= globals->immutable_ref_count_u32 ||
                        iree_vm_bytecode_bit_test(ref_set_bits, i);
    if (!is_set && (!is_nullable || !iree_vm_ref_is_null(refs[i]))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "immutable ref global %" PRIu32
          " is unset without a nullable canonical-null value",
          i);
    }
    if (iree_vm_ref_is_null(refs[i])) {
      if (!is_nullable) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "required ref global %" PRIu32 " is null", i);
      }
    } else if (!iree_vm_ref_isa(
                   refs[i],
                   module->resolved_ref_types[descriptor
                                                  ->ref_type_ordinal_u16])) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "ref global %" PRIu32 " has the wrong type", i);
    }
  }

  const iree_vm_function_ref_t* functions =
      iree_vm_bytecode_process_functions(module, storage.data);
  const uint64_t* function_set_bits =
      iree_vm_bytecode_process_function_set_bits(module, storage.data);
  for (uint32_t i = 0; i < globals->function_count_u32; ++i) {
    const bool is_nullable =
        iree_any_bit_set(module->layout.globals.functions[i].flags_u16,
                         IREE_VM_BYTECODE_GLOBAL_FUNCTION_FLAG_NULLABLE);
    const bool is_set = i >= globals->immutable_function_count_u32 ||
                        iree_vm_bytecode_bit_test(function_set_bits, i);
    if (!is_set &&
        (!is_nullable || !iree_vm_function_ref_is_null(functions[i]))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "immutable function global %" PRIu32
          " is unset without a nullable canonical-null value",
          i);
    }
    if (iree_vm_function_ref_is_null(functions[i]) && !is_nullable) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "required function global %" PRIu32 " is null",
                              i);
    }
  }

  state->construction_state = IREE_VM_BYTECODE_CONSTRUCTION_STATE_SEALED;
  return iree_ok_status();
}

static void iree_vm_bytecode_module_detach_state(iree_vm_module_t* base_module,
                                                 iree_byte_span_t storage) {
  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  const iree_vm_bytecode_v0_globals_header_t* globals =
      module->layout.globals.header;
  if (!globals) return;
  iree_vm_ref_t* refs = iree_vm_bytecode_process_refs(module, storage.data);
  for (uint32_t i = 0; i < globals->ref_count_u32; ++i) {
    iree_vm_ref_reset(&refs[i]);
  }
}

//===----------------------------------------------------------------------===//
// Declaration queries
//===----------------------------------------------------------------------===//

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

static void iree_vm_bytecode_module_query_import_group(
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

static void iree_vm_bytecode_module_query_import(
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

static void iree_vm_bytecode_module_query_export(
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

static void iree_vm_bytecode_module_query_callable_type(
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

static void iree_vm_bytecode_module_query_presentation(
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

static void iree_vm_bytecode_module_metadata_by_ordinal(
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

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_bytecode_resolve_ref_types(
    iree_vm_environment_t* environment,
    const iree_vm_bytecode_module_layout_t* layout,
    iree_vm_ref_type_t* out_types) {
  uint32_t entry_base = 0;
  for (uint32_t group_i = 0; group_i < layout->ref_types.group_count;
       ++group_i) {
    const iree_vm_bytecode_v0_ref_type_group_row_t* group =
        &layout->ref_types.groups[group_i];
    const iree_string_view_t namespace_name = iree_vm_bytecode_string_at(
        &layout->strings, group->namespace_string_u16);
    const iree_vm_ref_type_table_t* table =
        iree_vm_environment_lookup_ref_type_table(environment, namespace_name);
    if (!table) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "ref-type namespace '%.*s' is unavailable",
                              (int)namespace_name.size, namespace_name.data);
    }
    for (uint32_t entry_i = 0; entry_i < group->entry_count_u32; ++entry_i) {
      const iree_vm_bytecode_v0_ref_type_entry_row_t* entry =
          &layout->ref_types.entries[entry_base + entry_i];
      const iree_string_view_t type_name = iree_vm_bytecode_string_at(
          &layout->strings, entry->type_name_string_u16);
      iree_vm_ref_type_t resolved_type = NULL;
      for (iree_host_size_t provider_i = 0; provider_i < table->types.count;
           ++provider_i) {
        const iree_vm_ref_type_t candidate =
            iree_vm_ref_type_storage_at(table->types, provider_i);
        if (iree_string_view_equal(candidate->type_name, type_name)) {
          resolved_type = candidate;
          break;
        }
      }
      if (!resolved_type) {
        return iree_make_status(IREE_STATUS_NOT_FOUND,
                                "ref type '%.*s.%.*s' is unavailable",
                                (int)namespace_name.size, namespace_name.data,
                                (int)type_name.size, type_name.data);
      }
      out_types[entry_base + entry_i] = resolved_type;
    }
    entry_base += group->entry_count_u32;
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_resolve_buffer_type(
    iree_vm_environment_t* environment, iree_vm_ref_type_t* out_buffer_type) {
  const iree_vm_ref_type_table_t* table =
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm"));
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(table, &types));
  *out_buffer_type = types.buffer;
  return iree_ok_status();
}

static void iree_vm_bytecode_initialize_rodata_roots(
    iree_vm_bytecode_image_t* image) {
  iree_vm_bytecode_module_t* module = &image->module;
  iree_host_size_t block_offset = 0;
  for (uint32_t i = 0; i < module->layout.rodata.count; ++i) {
    block_offset =
        iree_host_align(block_offset, IREE_VM_BYTECODE_SECTION_ALIGNMENT);
    const iree_host_size_t block_length =
        (iree_host_size_t)module->layout.rodata.lengths[i];
    const iree_const_byte_span_t block = iree_make_const_byte_span(
        module->layout.rodata.blocks_begin + block_offset, block_length);
    iree_vm_bytecode_image_retain(image);
    const iree_vm_buffer_release_callback_t callback = {
        iree_vm_bytecode_rodata_release,
        image,
    };
    iree_vm_buffer_initialize_embedded_read_only(block, callback,
                                                 &module->rodata_roots[i]);
    block_offset += block_length;
  }
}

static const iree_vm_module_vtable_t iree_vm_bytecode_module_vtable;

IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create(
    iree_vm_environment_t* environment, iree_string_view_t module_name,
    iree_vm_bytecode_module_storage_t storage, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  if (!out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_module is required");
  }
  *out_module = NULL;
  if (!environment || !module_name.data || module_name.size == 0 ||
      iree_string_view_find_char(module_name, '\0', 0) !=
          IREE_STRING_VIEW_NPOS ||
      !iree_unicode_utf8_validate(module_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "environment and valid module name are required");
  }
  if (!storage.contents.data || storage.contents.data_length == 0 ||
      !iree_host_ptr_has_alignment(storage.contents.data,
                                   IREE_VM_BYTECODE_IMAGE_ALIGNMENT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bytecode image storage must be nonempty and eight-byte aligned");
  }

  iree_vm_bytecode_module_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_verify(storage.contents, &plan));

  iree_host_size_t total_size = 0;
  iree_host_size_t name_offset = 0;
  iree_host_size_t ref_types_offset = 0;
  iree_host_size_t rodata_roots_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_vm_bytecode_image_t), &total_size,
      IREE_STRUCT_FIELD(module_name.size, char, &name_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          plan.layout.ref_types.entry_count, iree_vm_ref_type_t,
          iree_alignof(iree_vm_ref_type_t), &ref_types_offset),
      IREE_STRUCT_FIELD_ALIGNED(plan.layout.rodata.count, iree_vm_buffer_t,
                                iree_alignof(iree_vm_buffer_t),
                                &rodata_roots_offset)));

  iree_vm_bytecode_image_t* image = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&image));
  iree_atomic_ref_count_init(&image->ref_count);
  image->host_allocator = host_allocator;
  image->storage.contents = storage.contents;
  image->storage.deallocator = iree_allocator_null();

  iree_vm_bytecode_module_t* module = &image->module;
  module->image = image;
  module->layout = plan.layout;
  module->process_layout = plan.process_layout;
  module->resolved_ref_types =
      plan.layout.ref_types.entry_count == 0
          ? NULL
          : (iree_vm_ref_type_t*)((uint8_t*)image + ref_types_offset);
  module->rodata_roots =
      plan.layout.rodata.count == 0
          ? NULL
          : (iree_vm_buffer_t*)((uint8_t*)image + rodata_roots_offset);
  char* cloned_name = (char*)image + name_offset;
  memcpy(cloned_name, module_name.data, module_name.size);
  module->descriptor.name =
      iree_make_string_view(cloned_name, module_name.size);
  module->descriptor.flags = IREE_VM_MODULE_FLAG_LINKABLE;
  module->descriptor.ref_types.data = module->resolved_ref_types;
  module->descriptor.ref_types.count = plan.layout.ref_types.entry_count;
  module->descriptor.counts.function_count = plan.layout.functions.count;
  module->descriptor.counts.callable_type_count =
      plan.layout.callable_types.count;
  module->descriptor.counts.import_group_count =
      plan.layout.imports.group_count;
  module->descriptor.counts.import_count = plan.layout.imports.entry_count;
  module->descriptor.counts.export_count = plan.layout.exports.count;
  module->descriptor.counts.metadata_count =
      plan.layout.metadata.header
          ? plan.layout.metadata.header->module_entry_count_u32
          : 0;
  module->descriptor.process_storage_size = plan.process_layout.total_size;

  iree_status_t status = iree_vm_bytecode_resolve_ref_types(
      environment, &module->layout, module->resolved_ref_types);
  uint32_t initialized_rodata_root_count = 0;
  if (iree_status_is_ok(status) && module->layout.rodata.count != 0) {
    status =
        iree_vm_bytecode_resolve_buffer_type(environment, &module->buffer_type);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_bytecode_initialize_rodata_roots(image);
    initialized_rodata_root_count = module->layout.rodata.count;
    status = iree_vm_module_initialize(&iree_vm_bytecode_module_vtable,
                                       &module->descriptor, &module->base);
  }

  if (iree_status_is_ok(status)) {
    image->storage.deallocator = storage.deallocator;
    *out_module = &module->base;
  } else {
    for (uint32_t i = 0; i < initialized_rodata_root_count; ++i) {
      iree_vm_buffer_release(&module->rodata_roots[i]);
    }
    iree_vm_bytecode_image_release(image);
  }
  return status;
}

static const iree_vm_module_vtable_t iree_vm_bytecode_module_vtable = {
    sizeof(iree_vm_bytecode_module_vtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    iree_vm_bytecode_module_destroy,
    iree_vm_bytecode_function_start,
    iree_vm_module_function_resume_unreachable,
    iree_vm_bytecode_module_attach_state,
    iree_vm_bytecode_module_seal_state,
    iree_vm_bytecode_module_detach_state,
    iree_vm_bytecode_module_query_import_group,
    iree_vm_bytecode_module_query_import,
    iree_vm_bytecode_module_query_export,
    iree_vm_bytecode_module_query_callable_type,
    iree_vm_bytecode_module_query_presentation,
    iree_vm_bytecode_module_metadata_by_ordinal,
};
