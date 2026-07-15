// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_null_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_null_executable_t;

static const iree_hal_executable_vtable_t iree_hal_null_executable_vtable;

static iree_hal_null_executable_t* iree_hal_null_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_executable_vtable);
  return (iree_hal_null_executable_t*)base_value;
}

iree_status_t iree_hal_null_executable_create(
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_executable = NULL;

  // Allocate storage for the executable and its associated data structures.
  iree_hal_null_executable_t* executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable),
                                (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_null_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;

  // TODO(null): load executable module(s) for |target|. The input data is
  // untrusted and requires the strongest verification the native artifact
  // representation provides. For JIT-style implementations as much work as
  // possible belongs here so failures propagate synchronously to users instead
  // of appearing during dispatch.
  //
  // In general the executable should only retain information required to
  // service the command buffer implementation that will be dispatching entry
  // points within it. Optionally information can be retained for tracing and
  // debugging.
  //
  // Native containers such as ELF can carry reflection and ABI metadata in
  // their normal sections. Drivers loading another container retain only the
  // target-native metadata required for dispatch, tracing, and debugging.
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "executable not implemented");

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_null_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): release any implementation resources.

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_null_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  // TODO(null): return the total number of exports in the executable.
  (void)executable;
  return 0;
}

static iree_status_t iree_hal_null_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t export_ordinal,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  (void)executable;
  // TODO(null): return export information.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reflection not implemented");
}

static iree_status_t iree_hal_null_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t export_ordinal, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  (void)executable;
  // TODO(null): return export parameter information.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "parameter reflection not implemented");
}

static iree_status_t iree_hal_null_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_export_ordinal) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  (void)executable;
  // TODO(null): lookup the export ordinal by name.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reflection not implemented");
}

static iree_status_t iree_hal_null_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  (void)executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_null_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  (void)executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid null executable global");
}

static iree_status_t iree_hal_null_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  (void)executable;
  (void)global;
  (void)queue_affinity;
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid null executable global");
}

static const iree_hal_executable_vtable_t iree_hal_null_executable_vtable = {
    .destroy = iree_hal_null_executable_destroy,
    .function_count = iree_hal_null_executable_export_count,
    .function_info = iree_hal_null_executable_export_info,
    .function_parameters = iree_hal_null_executable_export_parameters,
    .lookup_function_by_name = iree_hal_null_executable_lookup_export_by_name,
    .try_lookup_global_by_name =
        iree_hal_null_executable_try_lookup_global_by_name,
    .global_info = iree_hal_null_executable_global_info,
    .global_buffer = iree_hal_null_executable_global_buffer,
};
