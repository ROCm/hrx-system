// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/inspection.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/vm/bytecode/image.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/module_reflection.h"
#include "iree/vm/bytecode/module_storage.h"
#include "iree/vm/bytecode/verification.h"

static void iree_vm_bytecode_inspection_module_destroy(
    iree_vm_module_t* base_module) {
  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  iree_vm_bytecode_image_release(module->image);
}

static iree_status_t iree_vm_bytecode_inspection_function_start(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  (void)base_module;
  (void)params;
  (void)out_outcome;
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "inspection modules cannot execute functions");
}

static void iree_vm_bytecode_initialize_reflection_types(
    const iree_vm_bytecode_module_layout_t* layout,
    iree_vm_ref_type_table_t* tables,
    iree_vm_ref_type_descriptor_t* descriptors, iree_vm_ref_type_t* out_types) {
  uint32_t entry_base = 0;
  for (uint32_t group_i = 0; group_i < layout->ref_types.group_count;
       ++group_i) {
    const iree_vm_bytecode_v0_ref_type_group_row_t* group =
        &layout->ref_types.groups[group_i];
    iree_vm_ref_type_table_t* table = &tables[group_i];
    table->structure_size = sizeof(*table);
    table->flags = IREE_VM_REF_TYPE_TABLE_FLAG_REFLECTION_ONLY;
    table->namespace_name = iree_vm_bytecode_string_at(
        &layout->strings, group->namespace_string_u16);
    table->types.data = out_types + entry_base;
    table->types.count = group->entry_count_u32;
    for (uint32_t entry_i = 0; entry_i < group->entry_count_u32; ++entry_i) {
      const uint32_t entry_ordinal = entry_base + entry_i;
      const iree_vm_bytecode_v0_ref_type_entry_row_t* entry =
          &layout->ref_types.entries[entry_ordinal];
      iree_vm_ref_type_descriptor_t* descriptor = &descriptors[entry_ordinal];
      descriptor->destroy = NULL;
      descriptor->table = table;
      descriptor->type_name = iree_vm_bytecode_string_at(
          &layout->strings, entry->type_name_string_u16);
      out_types[entry_ordinal] = descriptor;
    }
    entry_base += group->entry_count_u32;
  }
}

static const iree_vm_module_vtable_t iree_vm_bytecode_inspection_module_vtable;

IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create_for_inspection(
    iree_string_view_t module_name, iree_vm_bytecode_module_storage_t storage,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  if (!out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_module is required");
  }
  *out_module = NULL;
  if (!module_name.data || module_name.size == 0 ||
      iree_string_view_find_char(module_name, '\0', 0) !=
          IREE_STRING_VIEW_NPOS ||
      !iree_unicode_utf8_validate(module_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "valid module name is required");
  }
  if (!storage.contents.data || storage.contents.data_length == 0 ||
      !iree_host_ptr_has_alignment(storage.contents.data,
                                   IREE_VM_BYTECODE_IMAGE_ALIGNMENT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bytecode image storage must be nonempty and eight-byte aligned");
  }

  iree_vm_bytecode_module_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_module_verify_structure(storage.contents, &plan));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_verify_inspectable(&plan));

  iree_host_size_t total_size = 0;
  iree_host_size_t name_offset = 0;
  iree_host_size_t ref_types_offset = 0;
  iree_host_size_t type_tables_offset = 0;
  iree_host_size_t type_descriptors_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_vm_bytecode_image_t), &total_size,
      IREE_STRUCT_FIELD(module_name.size, char, &name_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          plan.layout.ref_types.entry_count, iree_vm_ref_type_t,
          iree_alignof(iree_vm_ref_type_t), &ref_types_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          plan.layout.ref_types.group_count, iree_vm_ref_type_table_t,
          iree_alignof(iree_vm_ref_type_table_t), &type_tables_offset),
      IREE_STRUCT_FIELD_ALIGNED(plan.layout.ref_types.entry_count,
                                iree_vm_ref_type_descriptor_t,
                                iree_alignof(iree_vm_ref_type_descriptor_t),
                                &type_descriptors_offset)));

  iree_vm_bytecode_image_t* image = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(host_allocator, total_size,
                                                     0, 0, (void**)&image));
  iree_atomic_ref_count_init(&image->ref_count);
  image->host_allocator = host_allocator;
  image->storage.contents = storage.contents;
  image->storage.deallocator = iree_allocator_null();

  iree_vm_bytecode_module_t* module = &image->module;
  module->image = image;
  module->layout = plan.layout;
  memset(&module->process_layout, 0, sizeof(module->process_layout));
  module->resolved_ref_types =
      plan.layout.ref_types.entry_count == 0
          ? NULL
          : (iree_vm_ref_type_t*)((uint8_t*)image + ref_types_offset);
  module->buffer_type = NULL;
  module->rodata_roots = NULL;
  iree_vm_ref_type_table_t* type_tables =
      plan.layout.ref_types.group_count == 0
          ? NULL
          : (iree_vm_ref_type_table_t*)((uint8_t*)image + type_tables_offset);
  iree_vm_ref_type_descriptor_t* type_descriptors =
      plan.layout.ref_types.entry_count == 0
          ? NULL
          : (iree_vm_ref_type_descriptor_t*)((uint8_t*)image +
                                             type_descriptors_offset);
  iree_vm_bytecode_initialize_reflection_types(&module->layout, type_tables,
                                               type_descriptors,
                                               module->resolved_ref_types);

  char* cloned_name = (char*)image + name_offset;
  memcpy(cloned_name, module_name.data, module_name.size);
  module->descriptor.name =
      iree_make_string_view(cloned_name, module_name.size);
  module->descriptor.flags = IREE_VM_MODULE_FLAG_NONE;
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
  module->descriptor.process_storage_size = 0;

  iree_status_t status =
      iree_vm_module_initialize(&iree_vm_bytecode_inspection_module_vtable,
                                &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    image->storage.deallocator = storage.deallocator;
    *out_module = &module->base;
  } else {
    iree_vm_bytecode_image_release(image);
  }
  return status;
}

static const iree_vm_module_vtable_t iree_vm_bytecode_inspection_module_vtable =
    {
        sizeof(iree_vm_bytecode_inspection_module_vtable),
        IREE_VM_MODULE_ABI_VERSION_0,
        iree_vm_bytecode_inspection_module_destroy,
        iree_vm_bytecode_inspection_function_start,
        iree_vm_module_function_resume_unreachable,
        NULL,
        NULL,
        NULL,
        iree_vm_bytecode_module_query_import_group,
        iree_vm_bytecode_module_query_import,
        iree_vm_bytecode_module_query_export,
        iree_vm_bytecode_module_query_callable_type,
        iree_vm_bytecode_module_query_presentation,
        iree_vm_bytecode_module_metadata_by_ordinal,
};
