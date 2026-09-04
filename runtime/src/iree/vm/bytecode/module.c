// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/module.h"

#include <string.h>

#include "iree/vm/buffer_provider.h"
#include "iree/vm/bytecode/image.h"
#include "iree/vm/bytecode/interpreter.h"
#include "iree/vm/bytecode/process.h"
#include "iree/vm/bytecode/reflection.h"
#include "iree/vm/bytecode/verifier.h"

//===----------------------------------------------------------------------===//
// Module plans
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_bytecode_module_check_runtime_compatibility(
    const iree_vm_bytecode_module_plan_t* plan) {
  if (plan->layout.requirements.count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "bytecode architectural extension pages are not supported");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_module_build_verified_plan(
    iree_const_byte_span_t contents, iree_allocator_t scratch_allocator,
    iree_vm_bytecode_module_plan_t* out_plan) {
  iree_vm_bytecode_module_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_plan_build(contents, &plan));
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_verify_module_layout(&plan));
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_module_check_runtime_compatibility(&plan));

  // Keeps small function CFGs allocation-free with one cache line of scratch.
  uint32_t inline_block_offsets[16];
  const uint32_t maximum_block_count =
      plan.layout.functions.maximum_block_count;
  uint32_t* block_offsets =
      maximum_block_count == 0 ? NULL : inline_block_offsets;
  iree_status_t status = iree_ok_status();
  if (maximum_block_count > IREE_ARRAYSIZE(inline_block_offsets)) {
    status = iree_allocator_malloc_array(scratch_allocator, maximum_block_count,
                                         sizeof(*block_offsets),
                                         (void**)&block_offsets);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_verify_module_instructions(&plan, block_offsets);
  }
  if (block_offsets != inline_block_offsets) {
    iree_allocator_free(scratch_allocator, block_offsets);
  }
  if (iree_status_is_ok(status)) *out_plan = plan;
  return status;
}

static iree_status_t iree_vm_bytecode_module_build_trusted_plan(
    iree_const_byte_span_t contents, iree_vm_bytecode_module_plan_t* out_plan) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_plan_build(contents, out_plan));
  return iree_vm_bytecode_module_check_runtime_compatibility(out_plan);
}

IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_verify(
    iree_const_byte_span_t contents, iree_allocator_t scratch_allocator) {
  iree_vm_bytecode_module_plan_t plan;
  return iree_vm_bytecode_module_build_verified_plan(contents,
                                                     scratch_allocator, &plan);
}

//===----------------------------------------------------------------------===//
// Type resolution
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_bytecode_module_resolve_ref_types(
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

//===----------------------------------------------------------------------===//
// Image-backed rodata
//===----------------------------------------------------------------------===//

static void iree_vm_bytecode_rodata_release(void* user_data,
                                            iree_byte_span_t storage) {
  (void)storage;
  iree_vm_bytecode_image_release((iree_vm_bytecode_image_t*)user_data);
}

static void iree_vm_bytecode_module_initialize_rodata(
    iree_vm_bytecode_image_t* image, uint8_t* copy_storage) {
  iree_host_size_t section_offset = image->layout.rodata.blocks_offset;
  iree_host_size_t copy_offset = 0;
  for (uint32_t i = 0; i < image->layout.rodata.count; ++i) {
    const iree_vm_bytecode_v0_rodata_block_descriptor_t* descriptor =
        &image->layout.rodata.descriptors[i];
    section_offset =
        iree_host_align(section_offset, descriptor->minimum_alignment_u32);
    const iree_host_size_t block_length =
        (iree_host_size_t)descriptor->byte_length_u64;
    const uint8_t* block_data =
        image->layout.rodata.section_begin + section_offset;
    if (!iree_host_ptr_has_alignment(block_data,
                                     descriptor->minimum_alignment_u32)) {
      copy_offset =
          iree_host_align(copy_offset, descriptor->minimum_alignment_u32);
      if (block_length != 0) {
        memcpy(copy_storage + copy_offset, block_data, block_length);
      }
      block_data = copy_storage + copy_offset;
      copy_offset += block_length;
    }

    iree_vm_bytecode_image_retain(image);
    const iree_vm_buffer_release_callback_t release_callback = {
        iree_vm_bytecode_rodata_release,
        image,
    };
    iree_vm_buffer_initialize_embedded_read_only(
        iree_make_const_byte_span(block_data, block_length), release_callback,
        &image->rodata_roots[i]);
    section_offset += block_length;
  }
}

//===----------------------------------------------------------------------===//
// Module lifetime
//===----------------------------------------------------------------------===//

static void iree_vm_bytecode_module_destroy(iree_vm_module_t* base_module) {
  iree_vm_bytecode_image_t* image =
      iree_vm_bytecode_image_from_module(base_module);
  for (uint32_t i = 0; i < image->layout.rodata.count; ++i) {
    iree_vm_buffer_release(&image->rodata_roots[i]);
  }
  iree_vm_bytecode_image_release(image);
}

static const iree_vm_module_vtable_t iree_vm_bytecode_module_vtable;

static iree_status_t iree_vm_bytecode_module_validate_create_arguments(
    iree_vm_environment_t* environment, iree_string_view_t module_name,
    iree_vm_module_t** out_module) {
  if (!out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_module is required");
  }
  *out_module = NULL;
  if (!environment || !module_name.data || module_name.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "environment and module name are required");
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_bytecode_module_create_from_plan(
    iree_vm_environment_t* environment, iree_string_view_t module_name,
    iree_vm_bytecode_module_storage_t storage,
    const iree_vm_bytecode_module_plan_t* plan, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  iree_host_size_t total_size = 0;
  iree_host_size_t name_offset = 0;
  iree_host_size_t ref_types_offset = 0;
  iree_host_size_t rodata_roots_offset = 0;
  iree_host_size_t rodata_copy_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_vm_bytecode_image_t), &total_size,
      IREE_STRUCT_FIELD(module_name.size, char, &name_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          plan->layout.ref_types.entry_count, iree_vm_ref_type_t,
          iree_alignof(iree_vm_ref_type_t), &ref_types_offset),
      IREE_STRUCT_FIELD_ALIGNED(plan->layout.rodata.count, iree_vm_buffer_t,
                                iree_alignof(iree_vm_buffer_t),
                                &rodata_roots_offset),
      IREE_STRUCT_FIELD_ALIGNED(plan->rodata_storage.copy_length, uint8_t,
                                iree_max_align_t, &rodata_copy_offset)));

  const iree_host_size_t slab_alignment =
      iree_max(iree_max_align_t, plan->rodata_storage.copy_alignment);
  iree_vm_bytecode_image_t* image = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_aligned(host_allocator, total_size, slab_alignment,
                                    rodata_copy_offset, (void**)&image));
  iree_atomic_ref_count_init(&image->ref_count);
  image->host_allocator = host_allocator;
  image->storage.contents = storage.contents;
  image->storage.deallocator = iree_allocator_null();
  image->layout = plan->layout;
  image->process_layout = plan->process_layout;
  image->resolved_ref_types =
      plan->layout.ref_types.entry_count == 0
          ? NULL
          : (iree_vm_ref_type_t*)((uint8_t*)image + ref_types_offset);
  image->buffer_type = iree_vm_ref_type_storage_at(
      iree_vm_buffer_provider_table()->types, IREE_VM_REF_TYPE_BUFFER);
  image->rodata_roots =
      plan->layout.rodata.count == 0
          ? NULL
          : (iree_vm_buffer_t*)((uint8_t*)image + rodata_roots_offset);
  uint8_t* rodata_copy_storage = plan->rodata_storage.copy_alignment == 0
                                     ? NULL
                                     : (uint8_t*)image + rodata_copy_offset;

  char* cloned_name = (char*)image + name_offset;
  memcpy(cloned_name, module_name.data, module_name.size);
  image->descriptor.name = iree_make_string_view(cloned_name, module_name.size);
  image->descriptor.flags = IREE_VM_MODULE_FLAG_LINKABLE;
  image->descriptor.ref_types.data = image->resolved_ref_types;
  image->descriptor.ref_types.count = plan->layout.ref_types.entry_count;
  image->descriptor.counts.function_count = plan->layout.functions.count;
  image->descriptor.counts.callable_type_count =
      plan->layout.callable_types.count;
  image->descriptor.counts.import_group_count =
      plan->layout.imports.group_count;
  image->descriptor.counts.import_count = plan->layout.imports.entry_count;
  image->descriptor.counts.export_count = plan->layout.exports.count;
  image->descriptor.counts.metadata_count =
      plan->layout.metadata.header
          ? plan->layout.metadata.header->module_entry_count_u32
          : 0;
  image->descriptor.counts.callable_fields = plan->callable_fields;
  image->descriptor.process_storage_size = plan->process_layout.total_size;

  iree_status_t status = iree_vm_bytecode_module_resolve_ref_types(
      environment, &image->layout, image->resolved_ref_types);
  uint32_t initialized_rodata_count = 0;
  if (iree_status_is_ok(status)) {
    iree_vm_bytecode_module_initialize_rodata(image, rodata_copy_storage);
    initialized_rodata_count = image->layout.rodata.count;
    status = iree_vm_module_initialize(&iree_vm_bytecode_module_vtable,
                                       &image->descriptor, &image->base_module);
  }

  if (iree_status_is_ok(status)) {
    image->storage.deallocator = storage.deallocator;
    *out_module = &image->base_module;
  } else {
    for (uint32_t i = 0; i < initialized_rodata_count; ++i) {
      iree_vm_buffer_release(&image->rodata_roots[i]);
    }
    iree_vm_bytecode_image_release(image);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create(
    iree_vm_environment_t* environment, iree_string_view_t module_name,
    iree_vm_bytecode_module_storage_t storage, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_validate_create_arguments(
      environment, module_name, out_module));
  iree_vm_bytecode_module_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_build_verified_plan(
      storage.contents, host_allocator, &plan));
  return iree_vm_bytecode_module_create_from_plan(
      environment, module_name, storage, &plan, host_allocator, out_module);
}

IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_create_trusted(
    iree_vm_environment_t* environment, iree_string_view_t module_name,
    iree_vm_bytecode_module_storage_t storage, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_validate_create_arguments(
      environment, module_name, out_module));
  iree_vm_bytecode_module_plan_t plan;
  IREE_RETURN_IF_ERROR(
      iree_vm_bytecode_module_build_trusted_plan(storage.contents, &plan));
  return iree_vm_bytecode_module_create_from_plan(
      environment, module_name, storage, &plan, host_allocator, out_module);
}

static const iree_vm_module_vtable_t iree_vm_bytecode_module_vtable = {
    sizeof(iree_vm_bytecode_module_vtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    iree_vm_bytecode_module_destroy,
    iree_vm_bytecode_interpreter_start,
    iree_vm_bytecode_interpreter_resume,
    NULL,
    iree_vm_bytecode_process_seal_state,
    iree_vm_bytecode_process_detach_state,
    iree_vm_bytecode_reflection_query_import_group,
    iree_vm_bytecode_reflection_query_import,
    iree_vm_bytecode_reflection_query_export,
    iree_vm_bytecode_reflection_query_callable_type,
    iree_vm_bytecode_reflection_query_presentation,
    iree_vm_bytecode_reflection_metadata_by_ordinal,
};
