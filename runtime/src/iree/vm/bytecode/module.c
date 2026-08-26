// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/module.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/vm/bytecode/image.h"
#include "iree/vm/bytecode/interpreter.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/module_reflection.h"
#include "iree/vm/bytecode/module_storage.h"
#include "iree/vm/bytecode/verification.h"

//===----------------------------------------------------------------------===//
// Image and rodata lifetime
//===----------------------------------------------------------------------===//

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
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_module_verify_structure(storage.contents, &plan));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_module_verify_executable(&plan, host_allocator));

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
  if (iree_status_is_ok(status)) {
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
