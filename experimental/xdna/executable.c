// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/executable.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/amd/xdna/image/aie2p/native_image.h"

// Immutable metadata and native streams selected by one exported function.
typedef struct iree_hal_amd_xdna_executable_function_t {
  // Reflected HAL function information.
  iree_hal_executable_function_info_t info;
  // Native streams and cold relocations used to instantiate this function.
  iree_hal_amd_xdna_executable_entry_t entry;
  // First global image binding owned by this function.
  uint32_t first_binding_ordinal;
} iree_hal_amd_xdna_executable_function_t;

typedef struct iree_hal_amd_xdna_executable_t {
  // Common HAL executable state.
  iree_hal_executable_t base;
  // Host allocator owning this executable.
  iree_allocator_t host_allocator;
  // Qualified image retaining source and reflected metadata storage.
  iree_hal_amd_xdna_image_t* image;
  // Native transaction storage retained by function entries.
  iree_hal_amd_xdna_aie2p_native_image_t* native_image;
  // Number of exported functions in |functions|.
  iree_host_size_t function_count;
  // Immutable function metadata in export ordinal order.
  iree_hal_amd_xdna_executable_function_t* functions;
} iree_hal_amd_xdna_executable_t;

static const iree_hal_executable_vtable_t iree_hal_amd_xdna_executable_vtable;

static iree_hal_amd_xdna_executable_t* iree_hal_amd_xdna_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amd_xdna_executable_vtable);
  return (iree_hal_amd_xdna_executable_t*)base_value;
}

static iree_status_t iree_hal_amd_xdna_executable_validate_bindings(
    const iree_hal_amd_xdna_image_t* image) {
  const iree_host_size_t function_count =
      iree_hal_amd_xdna_image_entry_count(image);
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const iree_hal_amd_xdna_elf_entry_record_t* entry =
        iree_hal_amd_xdna_image_entry(image, i);
    if (entry->binding_count > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA entry[%" PRIhsz
                              "] declares %u bindings, exceeding the HAL "
                              "reflection limit of %u",
                              i, entry->binding_count, (uint32_t)UINT16_MAX);
    }
    const uint64_t binding_end =
        (uint64_t)entry->first_binding_ordinal + entry->binding_count;
    for (uint64_t j = entry->first_binding_ordinal; j < binding_end; ++j) {
      const iree_hal_amd_xdna_elf_binding_record_t* binding =
          iree_hal_amd_xdna_image_binding(image, (iree_host_size_t)j);
      if (binding == NULL ||
          binding->kind != IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "XDNA entry[%" PRIhsz "] requires scalar binding support", i);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_initialize_functions(
    iree_hal_amd_xdna_executable_t* executable) {
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    const iree_hal_amd_xdna_elf_entry_record_t* image_entry =
        iree_hal_amd_xdna_image_entry(executable->image, i);
    iree_hal_amd_xdna_aie2p_native_entry_t native_entry;
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_native_image_query_entry(
        executable->native_image, i, &native_entry));
    iree_const_byte_span_t array;
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_xdna_aie2p_native_image_query_array_configuration(
            executable->native_image, native_entry.array_ordinal, &array));
    executable->functions[i] = (iree_hal_amd_xdna_executable_function_t){
        .info =
            {
                .name =
                    iree_hal_amd_xdna_image_entry_name(executable->image, i),
                .binding_count = (uint16_t)image_entry->binding_count,
                .parameter_count = (uint16_t)image_entry->binding_count,
                .maximum_workgroup_invocations = 1,
                .workgroup_size = {1, 1, 1},
            },
        .entry =
            {
                .array = array,
                .native = native_entry,
                .binding_count = image_entry->binding_count,
            },
        .first_binding_ordinal = image_entry->first_binding_ordinal,
    };
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_executable_create(
    const iree_hal_queue_family_t* queue_family,
    iree_byte_sequence_t* source_sequence,
    const iree_hal_amd_xdna_aie2p_target_t* target,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(out_executable);
  if (source_sequence == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA image source is required");
  }

  iree_hal_amd_xdna_image_target_t image_target;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_target_initialize_image_target(
      target, &image_target));
  iree_hal_amd_xdna_image_t* image = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_create(
      source_sequence, &image_target, host_allocator, &image));

  iree_hal_amd_xdna_aie2p_native_image_t* native_image = NULL;
  iree_status_t status = iree_hal_amd_xdna_aie2p_native_image_create(
      image, target, host_allocator, &native_image);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_executable_validate_bindings(image);
  }

  const iree_host_size_t function_count =
      iree_status_is_ok(status)
          ? iree_hal_amd_xdna_aie2p_native_image_entry_count(native_image)
          : 0;
  iree_host_size_t functions_offset = 0;
  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_amd_xdna_executable_t), &total_size,
        IREE_STRUCT_FIELD_ALIGNED(
            function_count, iree_hal_amd_xdna_executable_function_t,
            iree_alignof(iree_hal_amd_xdna_executable_function_t),
            &functions_offset));
  }

  iree_hal_amd_xdna_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  }
  if (iree_status_is_ok(status)) {
    memset(executable, 0, total_size);
    iree_hal_executable_initialize(
        queue_family, &iree_hal_amd_xdna_executable_vtable, &executable->base);
    executable->host_allocator = host_allocator;
    executable->image = image;
    executable->native_image = native_image;
    executable->function_count = function_count;
    executable->functions =
        (iree_hal_amd_xdna_executable_function_t*)((uint8_t*)executable +
                                                   functions_offset);
    image = NULL;
    native_image = NULL;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_executable_initialize_functions(executable);
  }

  iree_hal_amd_xdna_aie2p_native_image_destroy(native_image);
  iree_hal_amd_xdna_image_destroy(image);
  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else if (executable != NULL) {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  return status;
}

bool iree_hal_amd_xdna_executable_isa(iree_hal_executable_t* executable) {
  return iree_hal_resource_is((const iree_hal_resource_t*)executable,
                              &iree_hal_amd_xdna_executable_vtable);
}

iree_status_t iree_hal_amd_xdna_executable_query_entry(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_amd_xdna_executable_entry_t* out_entry) {
  IREE_ASSERT_ARGUMENT(out_entry);
  iree_hal_amd_xdna_executable_t* executable =
      iree_hal_amd_xdna_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->function_count);
  }
  const iree_hal_amd_xdna_executable_function_t* executable_function =
      &executable->functions[iree_hal_executable_function_index(function)];
  *out_entry = executable_function->entry;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_executable_query_binding(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t binding_ordinal,
    iree_hal_amd_xdna_elf_binding_record_t* out_binding) {
  IREE_ASSERT_ARGUMENT(out_binding);
  iree_hal_amd_xdna_executable_t* executable =
      iree_hal_amd_xdna_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA executable function is out of range");
  }
  const iree_hal_amd_xdna_executable_function_t* executable_function =
      &executable->functions[iree_hal_executable_function_index(function)];
  if (binding_ordinal >= executable_function->info.binding_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA executable binding is out of range");
  }
  const iree_host_size_t image_binding_ordinal =
      executable_function->first_binding_ordinal + binding_ordinal;
  const iree_hal_amd_xdna_elf_binding_record_t* binding =
      iree_hal_amd_xdna_image_binding(executable->image, image_binding_ordinal);
  if (binding == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "qualified XDNA executable binding metadata is inconsistent");
  }
  *out_binding = *binding;
  return iree_ok_status();
}

static void iree_hal_amd_xdna_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_amd_xdna_executable_t* executable =
      iree_hal_amd_xdna_executable_cast(base_executable);
  const iree_allocator_t host_allocator = executable->host_allocator;
  iree_hal_amd_xdna_aie2p_native_image_destroy(executable->native_image);
  iree_hal_amd_xdna_image_destroy(executable->image);
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t iree_hal_amd_xdna_executable_function_count(
    iree_hal_executable_t* base_executable) {
  return iree_hal_amd_xdna_executable_cast(base_executable)->function_count;
}

static iree_status_t iree_hal_amd_xdna_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_amd_xdna_executable_t* executable =
      iree_hal_amd_xdna_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA executable function is out of range");
  }
  *out_info =
      executable->functions[iree_hal_executable_function_index(function)].info;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_amd_xdna_executable_t* executable =
      iree_hal_amd_xdna_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA executable function is out of range");
  }
  const iree_hal_amd_xdna_executable_function_t* executable_function =
      &executable->functions[iree_hal_executable_function_index(function)];
  const iree_host_size_t copy_count =
      iree_min(capacity, executable_function->info.parameter_count);
  for (iree_host_size_t i = 0; i < copy_count; ++i) {
    out_parameters[i] = (iree_hal_executable_function_parameter_t){
        .type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
        .offset = (uint16_t)i,
    };
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_amd_xdna_executable_t* executable =
      iree_hal_amd_xdna_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->functions[i].info.name, name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "XDNA executable function '%.*s' was not found",
                          (int)name.size, name.data);
}

static iree_status_t iree_hal_amd_xdna_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  (void)base_executable;
  (void)global;
  (void)out_info;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid XDNA executable global");
}

static iree_status_t iree_hal_amd_xdna_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)global;
  (void)out_buffer;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid XDNA executable global");
}

static const iree_hal_executable_vtable_t iree_hal_amd_xdna_executable_vtable =
    {
        .destroy = iree_hal_amd_xdna_executable_destroy,
        .function_count = iree_hal_amd_xdna_executable_function_count,
        .function_info = iree_hal_amd_xdna_executable_function_info,
        .function_parameters = iree_hal_amd_xdna_executable_function_parameters,
        .lookup_function_by_name =
            iree_hal_amd_xdna_executable_lookup_function_by_name,
        .try_lookup_global_by_name =
            iree_hal_amd_xdna_executable_try_lookup_global_by_name,
        .global_info = iree_hal_amd_xdna_executable_global_info,
        .global_buffer = iree_hal_amd_xdna_executable_global_buffer,
};
