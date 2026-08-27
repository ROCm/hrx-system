// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/function_attributes.h"

#include <limits.h>
#include <string.h>

static iree_status_t iree_hal_streaming_function_attribute_int_status(
    iree_string_view_t function_name, iree_string_view_t attribute_name,
    uint64_t value) {
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "function '%.*s' %.*s value %" PRIu64
                          " cannot be represented by the compatibility API",
                          (int)function_name.size, function_name.data,
                          (int)attribute_name.size, attribute_name.data, value);
}

iree_status_t iree_hal_streaming_function_attributes_initialize(
    const iree_hal_device_spec_t* device_spec,
    const iree_hal_executable_function_info_t* function_info,
    iree_hal_streaming_function_attributes_t* out_attributes) {
  memset(out_attributes, 0, sizeof(*out_attributes));
  iree_atomic_store(&out_attributes->configured_dynamic_shared_memory_size, 0,
                    iree_memory_order_relaxed);

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  const iree_hal_device_launch_spec_t* launch =
      dispatch ? &dispatch->launch : NULL;
  const iree_hal_device_execution_spec_t* execution =
      dispatch ? &dispatch->execution : NULL;

  const uint32_t device_maximum_threads =
      launch ? launch->maximum_workgroup_invocations : 0;
  const uint32_t function_maximum_threads =
      function_info->maximum_workgroup_invocations;
  if (device_maximum_threads != 0 && function_maximum_threads != 0 &&
      function_maximum_threads > device_maximum_threads) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function '%.*s' maximum workgroup invocation count %u exceeds "
        "device maximum %u",
        (int)function_info->name.size, function_info->name.data,
        function_maximum_threads, device_maximum_threads);
  }
  const uint32_t maximum_threads = function_maximum_threads != 0
                                       ? function_maximum_threads
                                       : device_maximum_threads;
  if (maximum_threads > INT_MAX) {
    return iree_hal_streaming_function_attribute_int_status(
        function_info->name, IREE_SV("maximum thread count"), maximum_threads);
  }
  if (maximum_threads != 0) {
    out_attributes->provided_flags |=
        IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK;
    out_attributes->maximum_threads_per_block = maximum_threads;
  }

  const iree_hal_executable_function_resource_usage_t* resource_usage =
      &function_info->resource_usage;
  if (iree_all_bits_set(
          resource_usage->provided_flags,
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_WORKGROUP_LOCAL_MEMORY)) {
    if (resource_usage->fixed_workgroup_local_memory_size > INT_MAX) {
      return iree_hal_streaming_function_attribute_int_status(
          function_info->name, IREE_SV("fixed shared-memory"),
          resource_usage->fixed_workgroup_local_memory_size);
    }
    out_attributes->provided_flags |=
        IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_SHARED_MEMORY;
    out_attributes->fixed_shared_memory_size =
        resource_usage->fixed_workgroup_local_memory_size;
  }
  if (iree_all_bits_set(
          resource_usage->provided_flags,
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_PRIVATE_MEMORY)) {
    if (resource_usage->fixed_private_memory_size > INT_MAX) {
      return iree_hal_streaming_function_attribute_int_status(
          function_info->name, IREE_SV("fixed local-memory"),
          resource_usage->fixed_private_memory_size);
    }
    out_attributes->provided_flags |=
        IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_LOCAL_MEMORY;
    out_attributes->fixed_local_memory_size =
        resource_usage->fixed_private_memory_size;
  }
  if (iree_all_bits_set(
          resource_usage->provided_flags,
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_INVOCATION_REGISTERS)) {
    if (resource_usage->invocation_register_count > INT_MAX) {
      return iree_hal_streaming_function_attribute_int_status(
          function_info->name, IREE_SV("register count"),
          resource_usage->invocation_register_count);
    }
    out_attributes->provided_flags |=
        IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_REGISTER_COUNT;
    out_attributes->register_count = resource_usage->invocation_register_count;
  }

  if (!execution ||
      !iree_all_bits_set(
          resource_usage->provided_flags,
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_WORKGROUP_LOCAL_MEMORY)) {
    return iree_ok_status();
  }

  const uint64_t default_capacity =
      execution->maximum_workgroup_local_memory_size;
  const uint64_t optin_capacity =
      execution->maximum_workgroup_local_memory_size_optin != 0
          ? execution->maximum_workgroup_local_memory_size_optin
          : default_capacity;
  if (optin_capacity == 0) return iree_ok_status();
  if (default_capacity > optin_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device default workgroup-local memory capacity %" PRIu64
        " exceeds opt-in capacity %" PRIu64,
        default_capacity, optin_capacity);
  }

  const uint64_t fixed_size = resource_usage->fixed_workgroup_local_memory_size;
  if (fixed_size > optin_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function '%.*s' fixed shared-memory size %" PRIu64
                            " exceeds device opt-in capacity %" PRIu64,
                            (int)function_info->name.size,
                            function_info->name.data, fixed_size,
                            optin_capacity);
  }
  const uint64_t maximum_configurable_size = optin_capacity - fixed_size;
  const uint64_t configured_size =
      default_capacity > fixed_size ? default_capacity - fixed_size : 0;
  if (maximum_configurable_size > INT_MAX) {
    return iree_hal_streaming_function_attribute_int_status(
        function_info->name,
        IREE_SV("maximum configurable dynamic shared-memory"),
        maximum_configurable_size);
  }
  if (configured_size > INT_MAX) {
    return iree_hal_streaming_function_attribute_int_status(
        function_info->name, IREE_SV("dynamic shared-memory"), configured_size);
  }

  out_attributes->provided_flags |=
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY;
  out_attributes->maximum_configurable_dynamic_shared_memory_size =
      (uint32_t)maximum_configurable_size;
  iree_atomic_store(&out_attributes->configured_dynamic_shared_memory_size,
                    (uint32_t)configured_size, iree_memory_order_relaxed);
  return iree_ok_status();
}
