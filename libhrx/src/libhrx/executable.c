// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <string.h>

#include "hrx_internal.h"
#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
#include "iree/hal/drivers/amdgpu/target/selection.h"
#endif  // HRX_HAS_IREE_AMDGPU_DRIVER

static iree_status_t hrx_executable_select_target(
    const iree_hal_device_spec_t* device_spec, iree_string_view_t target_family,
    iree_string_view_t artifact_target_key,
    iree_hal_executable_target_selection_result_t* out_result) {
#ifdef HRX_HAS_IREE_AMDGPU_DRIVER
  if (iree_string_view_equal(target_family, IREE_SV("amdgpu"))) {
    return iree_hal_amdgpu_device_spec_select_executable_target(
        device_spec, artifact_target_key,
        /*physical_device_affinity=*/0, out_result);
  }
#endif  // HRX_HAS_IREE_AMDGPU_DRIVER

  const iree_hal_executable_target_selection_t selection = {
      .family = target_family,
      .target_key = artifact_target_key,
  };
  *out_result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  return iree_ok_status();
}

static iree_status_t hrx_executable_snapshot_export_names(
    iree_hal_executable_t* hal_executable, iree_host_size_t* out_export_count,
    const char*** out_export_names) {
  *out_export_count = 0;
  *out_export_names = NULL;

  const iree_host_size_t export_count =
      iree_hal_executable_export_count(hal_executable);
  if (export_count == 0) return iree_ok_status();
  if (export_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable export count exceeds HRX ordinal range");
  }

  iree_host_size_t storage_size = 0;
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_executable_export_info_t hal_info;
    IREE_RETURN_IF_ERROR(iree_hal_executable_export_info(
        hal_executable, (iree_hal_executable_export_ordinal_t)i, &hal_info));
    if (hal_info.name.size > 0 && !hal_info.name.data) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "executable export name is missing storage");
    }
    iree_host_size_t name_storage_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(hal_info.name.size, /*NUL=*/1,
                                                  &name_storage_size) ||
                      !iree_host_size_checked_add(
                          storage_size, name_storage_size, &storage_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable export name storage overflow");
    }
  }

  iree_host_size_t pointer_table_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(export_count,
                                                sizeof(const char*),
                                                &pointer_table_size) ||
                    !iree_host_size_checked_add(pointer_table_size,
                                                storage_size, &total_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable export name table overflow");
  }

  const char** export_names = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), total_size, (void**)&export_names));

  char* name_storage = (char*)export_names + pointer_table_size;
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_executable_export_info_t hal_info;
    iree_status_t status = iree_hal_executable_export_info(
        hal_executable, (iree_hal_executable_export_ordinal_t)i, &hal_info);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), export_names);
      return status;
    }
    export_names[i] = name_storage;
    if (hal_info.name.size > 0) {
      memcpy(name_storage, hal_info.name.data, hal_info.name.size);
    }
    name_storage[hal_info.name.size] = '\0';
    name_storage += hal_info.name.size + 1;
  }

  *out_export_count = export_count;
  *out_export_names = export_names;
  return iree_ok_status();
}

static hrx_status_t hrx_executable_wrap(hrx_device_t device,
                                        iree_hal_executable_t* hal_executable,
                                        hrx_executable_t* executable) {
  iree_host_size_t export_count = 0;
  const char** export_names = NULL;
  iree_status_t status = hrx_executable_snapshot_export_names(
      hal_executable, &export_count, &export_names);
  if (!iree_status_is_ok(status)) {
    iree_hal_executable_release(hal_executable);
    return hrx_status_from_iree(status);
  }

  hrx_executable_t value = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(*value),
                                 (void**)&value);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), export_names);
    iree_hal_executable_release(hal_executable);
    return hrx_status_from_iree(status);
  }

  memset(value, 0, sizeof(*value));
  iree_atomic_ref_count_init(&value->ref_count);
  value->hal_executable = hal_executable;
  value->device = device;
  value->export_count = export_count;
  value->export_names = export_names;
  hrx_device_retain(device);
  *executable = value;
  return hrx_ok_status();
}

hrx_status_t hrx_executable_load_data(hrx_device_t device,
                                      const void* executable_data,
                                      size_t executable_data_size,
                                      const char* target_family,
                                      const char* target_key,
                                      hrx_executable_t* executable) {
  if (!device || !executable || !executable_data || executable_data_size == 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device, executable_data, or executable is invalid");
  }
  *executable = NULL;
  if (!target_family || target_family[0] == '\0' || !target_key ||
      target_key[0] == '\0') {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "target_family and target_key are required");
  }

  iree_hal_executable_target_selection_result_t selection_result;
  HRX_RETURN_IF_IREE_ERROR(hrx_executable_select_target(
      iree_hal_device_spec(device->hal_device),
      iree_make_cstring_view(target_family), iree_make_cstring_view(target_key),
      &selection_result));
  if (selection_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "executable target is not supported by the device");
  } else if (selection_result.outcome ==
             IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return hrx_make_status(HRX_STATUS_FAILED_PRECONDITION,
                           "executable target is ambiguous for the device");
  }

  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data =
      iree_make_const_byte_span(executable_data, executable_data_size);

  iree_hal_executable_t* hal_executable = NULL;
  iree_status_t status = iree_hal_device_load_executable(
      device->hal_device, IREE_HAL_QUEUE_AFFINITY_ANY, selection_result.target,
      &load_params, &hal_executable);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  return hrx_executable_wrap(device, hal_executable, executable);
}

hrx_status_t hrx_executable_load_file(hrx_device_t device, const char* path,
                                      const char* target_family,
                                      const char* target_key,
                                      hrx_executable_t* executable) {
  if (!device || !path || !executable) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device, path, or executable is NULL");
  }
  *executable = NULL;

  FILE* file = fopen(path, "rb");
  if (!file) {
    return hrx_make_status(HRX_STATUS_NOT_FOUND,
                           "failed to open executable file");
  }

  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return hrx_make_status(HRX_STATUS_INTERNAL,
                           "failed to seek executable file");
  }
  long file_size = ftell(file);
  if (file_size <= 0) {
    fclose(file);
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "empty executable file");
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return hrx_make_status(HRX_STATUS_INTERNAL,
                           "failed to rewind executable file");
  }

  void* file_data = NULL;
  hrx_status_t status = hrx_host_allocator_malloc_uninitialized(
      hrx_host_allocator_system(), (size_t)file_size, &file_data);
  if (!hrx_status_is_ok(status)) {
    fclose(file);
    return status;
  }

  size_t read_size = fread(file_data, 1, (size_t)file_size, file);
  fclose(file);
  if (read_size != (size_t)file_size) {
    hrx_host_allocator_free(hrx_host_allocator_system(), file_data);
    return hrx_make_status(HRX_STATUS_DATA_LOSS,
                           "short read while loading executable file");
  }

  status = hrx_executable_load_data(device, file_data, (size_t)file_size,
                                    target_family, target_key, executable);
  hrx_host_allocator_free(hrx_host_allocator_system(), file_data);
  return status;
}

void hrx_executable_retain(hrx_executable_t executable) {
  if (!executable) return;
  iree_hal_executable_retain(executable->hal_executable);
  hrx_device_retain(executable->device);
  iree_atomic_ref_count_inc(&executable->ref_count);
}

void hrx_executable_release(hrx_executable_t executable) {
  if (!executable) return;
  iree_hal_executable_t* hal_executable = executable->hal_executable;
  hrx_device_t device = executable->device;
  if (iree_atomic_ref_count_dec(&executable->ref_count) == 1) {
    iree_allocator_free(iree_allocator_system(), executable->export_names);
    iree_allocator_free(iree_allocator_system(), executable);
  }
  iree_hal_executable_release(hal_executable);
  hrx_device_release(device);
}

hrx_status_t hrx_executable_export_count(hrx_executable_t executable,
                                         size_t* count) {
  if (!executable || !count) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "executable or count is NULL");
  }
  *count = executable->export_count;
  return hrx_ok_status();
}

hrx_status_t hrx_executable_export_info(
    hrx_executable_t executable, uint32_t export_ordinal,
    hrx_executable_export_info_t* out_info) {
  if (!executable || !out_info) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "executable or out_info is NULL");
  }
  if (export_ordinal >= executable->export_count) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "executable export ordinal is out of range");
  }

  iree_hal_executable_export_info_t hal_info;
  iree_status_t status = iree_hal_executable_export_info(
      executable->hal_executable,
      (iree_hal_executable_export_ordinal_t)export_ordinal, &hal_info);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  memset(out_info, 0, sizeof(*out_info));
  out_info->name = executable->export_names[export_ordinal];
  out_info->flags = (uint32_t)hal_info.flags;
  out_info->constant_byte_length = hal_info.constant_byte_length;
  out_info->binding_count = hal_info.binding_count;
  out_info->parameter_count = hal_info.parameter_count;
  out_info->workgroup_size[0] = hal_info.workgroup_size[0];
  out_info->workgroup_size[1] = hal_info.workgroup_size[1];
  out_info->workgroup_size[2] = hal_info.workgroup_size[2];
  return hrx_ok_status();
}

hrx_status_t hrx_executable_lookup_export_by_name(hrx_executable_t executable,
                                                  const char* name,
                                                  uint32_t* export_ordinal) {
  if (!executable || !name || !export_ordinal) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "executable, name, or export_ordinal is NULL");
  }

  iree_hal_executable_export_ordinal_t ordinal = 0;
  iree_status_t status = iree_hal_executable_lookup_export_by_name(
      executable->hal_executable, iree_make_cstring_view(name), &ordinal);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }

  *export_ordinal = ordinal;
  return hrx_ok_status();
}
