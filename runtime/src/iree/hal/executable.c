// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/executable.h"

#include <string.h>

#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(executable, method_name) \
  IREE_HAL_VTABLE_DISPATCH(executable, iree_hal_executable, method_name)

IREE_HAL_API_RETAIN_RELEASE(executable);

IREE_API_EXPORT const iree_hal_queue_family_t* iree_hal_executable_queue_family(
    const iree_hal_executable_t* executable) {
  IREE_ASSERT_ARGUMENT(executable);
  return executable->queue_family;
}

IREE_API_EXPORT void iree_hal_executable_initialize(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_vtable_t* vtable,
    iree_hal_executable_t* out_executable) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(out_executable);
  iree_hal_resource_initialize(vtable, &out_executable->resource);
  out_executable->queue_family = queue_family;
}

IREE_API_EXPORT void iree_hal_executable_load_params_initialize(
    iree_hal_executable_load_params_t* out_params) {
  IREE_ASSERT_ARGUMENT(out_params);
  memset(out_params, 0, sizeof(*out_params));
  out_params->flags = IREE_HAL_EXECUTABLE_LOAD_FLAG_ALLOW_OPTIMIZATION;
}

IREE_API_EXPORT iree_host_size_t
iree_hal_executable_function_count(iree_hal_executable_t* executable) {
  IREE_ASSERT_ARGUMENT(executable);
  return _VTABLE_DISPATCH(executable, function_count)(executable);
}

IREE_API_EXPORT iree_status_t iree_hal_executable_function_info(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_info);
  memset(out_info, 0, sizeof(*out_info));
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, function_info)(
      executable, function, out_info);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_function_parameters(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_parameters);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, function_parameters)(
      executable, function, capacity, out_parameters);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_lookup_function_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_function);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, lookup_function_by_name)(
      executable, name, out_function);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_try_lookup_global_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name, bool* out_found,
    iree_hal_executable_global_t* out_global) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_found);
  IREE_ASSERT_ARGUMENT(out_global);
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return _VTABLE_DISPATCH(executable, try_lookup_global_by_name)(
      executable, name, out_found, out_global);
}

IREE_API_EXPORT iree_status_t iree_hal_executable_lookup_global_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_executable_global_t* out_global) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_global);
  bool found = false;
  IREE_RETURN_IF_ERROR(iree_hal_executable_try_lookup_global_by_name(
      executable, name, &found, out_global));
  if (!found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "executable global `%.*s` not found",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_executable_global_info(
    iree_hal_executable_t* executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_info);
  memset(out_info, 0, sizeof(*out_info));
  iree_status_t status =
      _VTABLE_DISPATCH(executable, global_info)(executable, global, out_info);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_global_buffer(
    iree_hal_executable_t* executable, iree_hal_executable_global_t global,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_buffer);
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status =
      _VTABLE_DISPATCH(executable, global_buffer)(executable, global, &buffer);
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  }
  return status;
}
