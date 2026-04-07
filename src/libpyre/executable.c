// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include <stdio.h>
#include <string.h>

static pyre_status_t pyre_executable_wrap(
    pyre_device_t device, iree_hal_executable_cache_t* hal_executable_cache,
    iree_hal_executable_t* hal_executable, pyre_executable_t* executable) {
  pyre_executable_t value = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(*value), (void**)&value);
  if (!iree_status_is_ok(alloc_status)) {
    iree_hal_executable_release(hal_executable);
    iree_hal_executable_cache_release(hal_executable_cache);
    return pyre_status_from_iree(alloc_status);
  }

  memset(value, 0, sizeof(*value));
  iree_atomic_ref_count_init(&value->ref_count);
  value->hal_executable_cache = hal_executable_cache;
  value->hal_executable = hal_executable;
  value->device = device;
  pyre_device_retain(device);
  *executable = value;
  return pyre_ok_status();
}

pyre_status_t pyre_executable_load_data(
    pyre_device_t device, const void* executable_data,
    size_t executable_data_size, const char* executable_format,
    pyre_executable_t* executable) {
  if (!device || !executable || !executable_data || executable_data_size == 0) {
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT,
        "device, executable_data, or executable is invalid");
  }
  *executable = NULL;

  iree_hal_executable_cache_t* hal_executable_cache = NULL;
  iree_status_t status = iree_hal_executable_cache_create(
      device->hal_device, IREE_SV("pyre"), &hal_executable_cache);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  char inferred_format[16] = {0};
  iree_host_size_t inferred_size = executable_data_size;
  if (!executable_format || executable_format[0] == '\0') {
    status = iree_hal_executable_cache_infer_format(
        hal_executable_cache, /*caching_mode=*/0,
        iree_make_const_byte_span(executable_data, executable_data_size),
        sizeof(inferred_format), inferred_format, &inferred_size);
    if (!iree_status_is_ok(status)) {
      iree_hal_executable_cache_release(hal_executable_cache);
      return pyre_status_from_iree(status);
    }
    executable_format = inferred_format;
  }

  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.executable_format = iree_make_cstring_view(executable_format);
  params.executable_data =
      iree_make_const_byte_span(executable_data, inferred_size);
  params.caching_mode = 0;

  iree_hal_executable_t* hal_executable = NULL;
  status = iree_hal_executable_cache_prepare_executable(
      hal_executable_cache, &params, &hal_executable);
  if (!iree_status_is_ok(status)) {
    iree_hal_executable_cache_release(hal_executable_cache);
    return pyre_status_from_iree(status);
  }

  return pyre_executable_wrap(
      device, hal_executable_cache, hal_executable, executable);
}

pyre_status_t pyre_executable_load_file(
    pyre_device_t device, const char* path, const char* executable_format,
    pyre_executable_t* executable) {
  if (!device || !path || !executable) {
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT, "device, path, or executable is NULL");
  }
  *executable = NULL;

  FILE* file = fopen(path, "rb");
  if (!file) {
    return pyre_make_status(
        PYRE_STATUS_NOT_FOUND, "failed to open executable file");
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return pyre_make_status(
        PYRE_STATUS_INTERNAL, "failed to seek executable file");
  }
  long file_size = ftell(file);
  if (file_size <= 0) {
    fclose(file);
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT, "empty executable file");
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return pyre_make_status(
        PYRE_STATUS_INTERNAL, "failed to rewind executable file");
  }

  void* file_data = NULL;
  pyre_status_t status = pyre_host_allocator_malloc_uninitialized(
      pyre_host_allocator_system(), (size_t)file_size, &file_data);
  if (!pyre_status_is_ok(status)) {
    fclose(file);
    return status;
  }

  size_t read_size = fread(file_data, 1, (size_t)file_size, file);
  fclose(file);
  if (read_size != (size_t)file_size) {
    pyre_host_allocator_free(pyre_host_allocator_system(), file_data);
    return pyre_make_status(
        PYRE_STATUS_DATA_LOSS, "short read while loading executable file");
  }

  status = pyre_executable_load_data(
      device, file_data, (size_t)file_size, executable_format, executable);
  pyre_host_allocator_free(pyre_host_allocator_system(), file_data);
  return status;
}

void pyre_executable_retain(pyre_executable_t executable) {
  if (!executable) return;
  iree_hal_executable_cache_retain(executable->hal_executable_cache);
  iree_hal_executable_retain(executable->hal_executable);
  pyre_device_retain(executable->device);
  iree_atomic_ref_count_inc(&executable->ref_count);
}

void pyre_executable_release(pyre_executable_t executable) {
  if (!executable) return;
  iree_hal_executable_cache_t* hal_executable_cache =
      executable->hal_executable_cache;
  iree_hal_executable_t* hal_executable = executable->hal_executable;
  pyre_device_t device = executable->device;
  if (iree_atomic_ref_count_dec(&executable->ref_count) == 1) {
    iree_allocator_free(iree_allocator_system(), executable);
  }
  iree_hal_executable_release(hal_executable);
  iree_hal_executable_cache_release(hal_executable_cache);
  pyre_device_release(device);
}

pyre_status_t pyre_executable_export_count(
    pyre_executable_t executable, size_t* count) {
  if (!executable || !count) {
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT, "executable or count is NULL");
  }
  *count = iree_hal_executable_export_count(executable->hal_executable);
  return pyre_ok_status();
}

pyre_status_t pyre_executable_export_info(
    pyre_executable_t executable, uint32_t export_ordinal,
    pyre_executable_export_info_t* out_info) {
  if (!executable || !out_info) {
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT, "executable or out_info is NULL");
  }

  iree_hal_executable_export_info_t hal_info;
  iree_status_t status = iree_hal_executable_export_info(
      executable->hal_executable,
      (iree_hal_executable_export_ordinal_t)export_ordinal, &hal_info);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  memset(out_info, 0, sizeof(*out_info));
  out_info->name = hal_info.name.data;
  out_info->flags = (uint32_t)hal_info.flags;
  out_info->constant_count = hal_info.constant_count;
  out_info->binding_count = hal_info.binding_count;
  out_info->parameter_count = hal_info.parameter_count;
  out_info->workgroup_size[0] = hal_info.workgroup_size[0];
  out_info->workgroup_size[1] = hal_info.workgroup_size[1];
  out_info->workgroup_size[2] = hal_info.workgroup_size[2];
  return pyre_ok_status();
}

pyre_status_t pyre_executable_lookup_export_by_name(
    pyre_executable_t executable, const char* name,
    uint32_t* export_ordinal) {
  if (!executable || !name || !export_ordinal) {
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT,
        "executable, name, or export_ordinal is NULL");
  }

  iree_hal_executable_export_ordinal_t ordinal = 0;
  iree_status_t status = iree_hal_executable_lookup_export_by_name(
      executable->hal_executable, iree_make_cstring_view(name), &ordinal);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  *export_ordinal = ordinal;
  return pyre_ok_status();
}
