// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_executable.h"

#include "iree/hal/drivers/webgpu/webgpu_executable_format.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// Per-function entry
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_executable_entry_t {
  // Bridge handle for the compute pipeline.
  iree_hal_webgpu_handle_t pipeline_handle;
  // Bridge handle for bind group layout 0 of the compute pipeline.
  iree_hal_webgpu_handle_t bind_group_layout_handle;
  // Function name used by the command buffer and reflection APIs.
  iree_string_view_t name;
  // Static workgroup size for the function.
  uint32_t workgroup_size[3];
  // Number of resource bindings declared by the function.
  uint16_t binding_count;
} iree_hal_webgpu_executable_entry_t;

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_executable_t {
  // Common executable state.
  iree_hal_executable_t base;
  // Host allocator used to allocate the executable and entry storage.
  iree_allocator_t host_allocator;
  // Number of functions in the executable.
  iree_host_size_t function_count;
  // Per-function entry table with |function_count| entries.
  iree_hal_webgpu_executable_entry_t entries[];
} iree_hal_webgpu_executable_t;

static const iree_hal_executable_vtable_t iree_hal_webgpu_executable_vtable;

static iree_hal_webgpu_executable_t* iree_hal_webgpu_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_executable_vtable);
  return (iree_hal_webgpu_executable_t*)base_value;
}

static iree_status_t iree_hal_webgpu_executable_calculate_name_storage_size(
    const iree_hal_webgpu_executable_format_t* executable_format,
    iree_host_size_t* out_name_storage_size) {
  iree_host_size_t name_storage_size = 0;
  for (iree_host_size_t i = 0; i < executable_format->export_count; ++i) {
    iree_hal_webgpu_executable_export_t export_def;
    IREE_RETURN_IF_ERROR(iree_hal_webgpu_executable_format_read_export(
        executable_format, i, &export_def));
    if (!iree_host_size_checked_add(name_storage_size,
                                    export_def.entry_point.size,
                                    &name_storage_size) ||
        !iree_host_size_checked_add(name_storage_size, 1, &name_storage_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "WebGPU executable export name storage exceeds "
                              "host size limits");
    }
  }
  *out_name_storage_size = name_storage_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_executable_initialize_export(
    iree_hal_webgpu_handle_t device_handle,
    const iree_hal_webgpu_executable_export_t* export_def,
    char** inout_name_storage, iree_hal_webgpu_executable_entry_t* out_entry) {
  memcpy(*inout_name_storage, export_def->entry_point.data,
         export_def->entry_point.size);
  (*inout_name_storage)[export_def->entry_point.size] = '\0';
  out_entry->name =
      iree_make_string_view(*inout_name_storage, export_def->entry_point.size);
  *inout_name_storage += export_def->entry_point.size + 1;

  out_entry->binding_count = export_def->binding_count;
  memcpy(out_entry->workgroup_size, export_def->workgroup_size,
         sizeof(out_entry->workgroup_size));

  out_entry->pipeline_handle =
      iree_hal_webgpu_import_device_create_compute_pipeline(
          device_handle, /*layout_handle=*/0,
          (uint32_t)(uintptr_t)export_def->wgsl_source.data,
          (uint32_t)export_def->wgsl_source.size,
          (uint32_t)(uintptr_t)out_entry->name.data,
          (uint32_t)out_entry->name.size);
  if (out_entry->pipeline_handle == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "failed to create WebGPU compute pipeline for entry point '%.*s'",
        (int)out_entry->name.size, out_entry->name.data);
  }

  out_entry->bind_group_layout_handle =
      iree_hal_webgpu_import_pipeline_get_bind_group_layout(
          out_entry->pipeline_handle, /*index=*/0);
  if (out_entry->bind_group_layout_handle == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "failed to query WebGPU bind group layout for entry point '%.*s'",
        (int)out_entry->name.size, out_entry->name.data);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_webgpu_executable_create(
    iree_hal_webgpu_handle_t device_handle,
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_UNLIKELY(load_params->constant_count != 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "WebGPU executable specialization constants are not supported");
  }

  iree_hal_webgpu_executable_format_t executable_format;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_webgpu_executable_format_parse(load_params->executable_data,
                                                  &executable_format));
  const iree_host_size_t export_count = executable_format.export_count;

  iree_host_size_t name_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_webgpu_executable_calculate_name_storage_size(
              &executable_format, &name_storage_size));

  iree_host_size_t entry_storage_size = 0;
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(export_count,
                                  sizeof(iree_hal_webgpu_executable_entry_t),
                                  &entry_storage_size) ||
      !iree_host_size_checked_add(sizeof(iree_hal_webgpu_executable_t),
                                  entry_storage_size, &total_size) ||
      !iree_host_size_checked_add(total_size, name_storage_size, &total_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebGPU executable metadata storage exceeds host "
                            "size limits");
  }
  iree_hal_webgpu_executable_t* executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_executable_initialize(
      queue_family, &iree_hal_webgpu_executable_vtable, &executable->base);
  executable->host_allocator = host_allocator;
  executable->function_count = export_count;
  char* name_storage =
      (char*)executable + sizeof(*executable) + entry_storage_size;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_webgpu_executable_export_t export_def;
    status = iree_hal_webgpu_executable_format_read_export(&executable_format,
                                                           i, &export_def);
    if (iree_status_is_ok(status)) {
      status = iree_hal_webgpu_executable_initialize_export(
          device_handle, &export_def, &name_storage, &executable->entries[i]);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_executable_pipeline_handle(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  IREE_ASSERT(iree_hal_executable_function_is_index_in_range(
      function, executable->function_count));
  const uint32_t function_index = iree_hal_executable_function_index(function);
  return executable->entries[function_index].pipeline_handle;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_executable_bind_group_layout_handle(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  IREE_ASSERT(iree_hal_executable_function_is_index_in_range(
      function, executable->function_count));
  const uint32_t function_index = iree_hal_executable_function_index(function);
  return executable->entries[function_index].bind_group_layout_handle;
}

static void iree_hal_webgpu_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    iree_hal_webgpu_executable_entry_t* entry = &executable->entries[i];
    iree_hal_webgpu_import_handle_release(entry->pipeline_handle);
    iree_hal_webgpu_import_handle_release(entry->bind_group_layout_handle);
  }

  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_webgpu_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  return executable->function_count;
}

static iree_status_t iree_hal_webgpu_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range (count=%" PRIhsz ")",
                            function.value, executable->function_count);
  }
  const uint32_t function_index = iree_hal_executable_function_index(function);
  const iree_hal_webgpu_executable_entry_t* entry =
      &executable->entries[function_index];
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = entry->name;
  out_info->binding_count = entry->binding_count;
  out_info->workgroup_size[0] = entry->workgroup_size[0];
  out_info->workgroup_size[1] = entry->workgroup_size[1];
  out_info->workgroup_size[2] = entry->workgroup_size[2];
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  (void)base_executable;
  (void)function;
  (void)capacity;
  (void)out_parameters;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU executables do not support parameter "
                          "reflection; WGSL shader metadata does not carry "
                          "per-parameter name/description information");
}

static iree_status_t iree_hal_webgpu_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_webgpu_executable_t* executable =
      iree_hal_webgpu_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->entries[i].name, name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no function named '%.*s' in executable",
                          (int)name.size, name.data);
}

static iree_status_t iree_hal_webgpu_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  (void)base_executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid WebGPU executable global");
}

static iree_status_t iree_hal_webgpu_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)global;
  (void)out_buffer;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid WebGPU executable global");
}

static const iree_hal_executable_vtable_t iree_hal_webgpu_executable_vtable = {
    .destroy = iree_hal_webgpu_executable_destroy,
    .function_count = iree_hal_webgpu_executable_function_count,
    .function_info = iree_hal_webgpu_executable_function_info,
    .function_parameters = iree_hal_webgpu_executable_function_parameters,
    .lookup_function_by_name =
        iree_hal_webgpu_executable_lookup_function_by_name,
    .try_lookup_global_by_name =
        iree_hal_webgpu_executable_try_lookup_global_by_name,
    .global_info = iree_hal_webgpu_executable_global_info,
    .global_buffer = iree_hal_webgpu_executable_global_buffer,
};
