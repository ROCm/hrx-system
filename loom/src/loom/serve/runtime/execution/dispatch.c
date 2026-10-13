// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/dispatch.h"

struct loom_serve_dispatch_t {
  // Shared ownership between the compiler stage and retained model module.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this descriptor and its trailing argument map.
  iree_allocator_t allocator;
  // Retained native executable loaded for the model's exact queue family.
  iree_hal_executable_t* executable;
  // Executable-local entry resolved at preparation.
  iree_hal_executable_function_t function;
  // Concrete launch geometry produced by the matching compiler invocation.
  iree_hal_dispatch_config_t config;
  // Number of logical buffer arguments and native binding slots.
  uint16_t binding_count;
  // Native binding ordinal for each logical argument.
  uint16_t bindings[];
};

void loom_serve_dispatch_retain(loom_serve_dispatch_t* dispatch) {
  if (dispatch) {
    iree_atomic_ref_count_inc(&dispatch->ref_count);
  }
}

void loom_serve_dispatch_release(loom_serve_dispatch_t* dispatch) {
  if (dispatch && iree_atomic_ref_count_dec(&dispatch->ref_count) == 1) {
    iree_hal_executable_release(dispatch->executable);
    iree_allocator_free(dispatch->allocator, dispatch);
  }
}

iree_status_t loom_serve_dispatch_create(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_hal_dispatch_config_t config, loom_serve_dispatch_t** out_dispatch,
    iree_allocator_t host_allocator) {
  *out_dispatch = NULL;
  iree_hal_executable_function_info_t info;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_function_info(executable, function, &info));
  if (info.constant_byte_length || info.parameter_count != info.binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "direct stage requires a buffer-only signature");
  }
  loom_serve_dispatch_t* dispatch = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*dispatch), info.binding_count,
      sizeof(*dispatch->bindings), (void**)&dispatch));
  iree_atomic_ref_count_init(&dispatch->ref_count);
  dispatch->allocator = host_allocator;
  dispatch->executable = executable;
  iree_hal_executable_retain(executable);
  dispatch->function = function;
  dispatch->config = config;
  dispatch->binding_count = info.binding_count;
  iree_hal_executable_function_parameter_t* parameters = NULL;
  iree_status_t status = iree_ok_status();
  if (info.parameter_count) {
    status =
        iree_allocator_malloc_array(host_allocator, info.parameter_count,
                                    sizeof(*parameters), (void**)&parameters);
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_function_parameters(
          executable, function, info.parameter_count, parameters);
    }
  }
  for (uint16_t i = 0; i < info.parameter_count && iree_status_is_ok(status);
       ++i) {
    if (parameters[i].type !=
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "direct stage argument %u is not a buffer", i);
    } else {
      dispatch->bindings[i] = parameters[i].offset;
    }
  }
  iree_allocator_free(host_allocator, parameters);
  if (iree_status_is_ok(status)) {
    *out_dispatch = dispatch;
  } else {
    loom_serve_dispatch_release(dispatch);
  }
  return status;
}

uint16_t loom_serve_dispatch_binding_count(
    const loom_serve_dispatch_t* dispatch) {
  return dispatch->binding_count;
}

iree_status_t loom_serve_dispatch_issue(
    const loom_serve_dispatch_t* dispatch, loom_serve_execution_t* execution,
    iree_hal_buffer_binding_table_t bindings, iree_hal_buffer_ref_t* scratch,
    uint64_t* out_value) {
  for (uint16_t i = 0; i < dispatch->binding_count; ++i) {
    const iree_hal_buffer_binding_t* binding = &bindings.bindings[i];
    scratch[dispatch->bindings[i]] = iree_hal_make_buffer_ref(
        binding->buffer, binding->offset, binding->length);
  }
  return loom_serve_execution_dispatch(
      execution, dispatch->executable, dispatch->function, dispatch->config,
      (iree_hal_buffer_ref_list_t){dispatch->binding_count, scratch},
      out_value);
}
