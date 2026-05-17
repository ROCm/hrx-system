// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/executable.h"

#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"

static const iree_hal_executable_vtable_t
    iree_hal_remote_client_executable_vtable;

typedef struct iree_hal_remote_client_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_remote_client_device_t* device;
  iree_hal_remote_resource_id_t resource_id;
  iree_host_size_t export_count;
} iree_hal_remote_client_executable_t;

static void iree_hal_remote_client_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_ignore(iree_hal_remote_client_device_release_resource(
      executable->device, executable->resource_id));

  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_remote_client_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  return executable->export_count;
}

static iree_status_t iree_hal_remote_client_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->export_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function index %u >= function count %" PRIhsz,
                            iree_hal_executable_function_index(function),
                            executable->export_count);
  }
  // Return default metadata. Full function info would require an
  // EXECUTABLE_QUERY_EXPORT RPC to the server.
  memset(out_info, 0, sizeof(*out_info));
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "function name lookup requires "
                          "EXECUTABLE_QUERY_EXPORT RPC (not yet implemented)");
}

iree_status_t iree_hal_remote_client_executable_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id, iree_host_size_t export_count,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  iree_hal_remote_client_executable_t* executable = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable);
  if (iree_status_is_ok(status)) {
    memset(executable, 0, sizeof(*executable));
    iree_hal_resource_initialize(&iree_hal_remote_client_executable_vtable,
                                 &executable->resource);
    executable->host_allocator = host_allocator;
    executable->device = device;
    executable->resource_id = resource_id;
    executable->export_count = export_count;

    *out_executable = (iree_hal_executable_t*)executable;
  }
  return status;
}

iree_hal_remote_resource_id_t iree_hal_remote_client_executable_resource_id(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  return executable->resource_id;
}

static const iree_hal_executable_vtable_t
    iree_hal_remote_client_executable_vtable = {
        .destroy = iree_hal_remote_client_executable_destroy,
        .function_count = iree_hal_remote_client_executable_function_count,
        .function_info = iree_hal_remote_client_executable_function_info,
        .function_parameters =
            iree_hal_remote_client_executable_function_parameters,
        .lookup_function_by_name =
            iree_hal_remote_client_executable_lookup_function_by_name,
};
