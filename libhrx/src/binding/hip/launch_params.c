// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/launch_params.h"

#include <stdint.h>

static bool iree_hip_launch_extra_value_is_marker(void* value) {
  return value == HIP_LAUNCH_PARAM_BUFFER_POINTER ||
         value == HIP_LAUNCH_PARAM_BUFFER_SIZE || value == HIP_LAUNCH_PARAM_END;
}

hipError_t iree_hip_parse_launch_extra(void** extra, void** out_buffer,
                                       size_t* out_buffer_size) {
  if (!extra || !out_buffer || !out_buffer_size) return hipErrorInvalidValue;
  *out_buffer = NULL;
  *out_buffer_size = 0;

  bool saw_buffer = false;
  bool saw_size = false;
  for (iree_host_size_t i = 0; extra[i] != HIP_LAUNCH_PARAM_END; i += 2) {
    void* key = extra[i];
    void* value = extra[i + 1];
    if (value == HIP_LAUNCH_PARAM_END) return hipErrorInvalidValue;

    if (key == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
      if (saw_buffer || iree_hip_launch_extra_value_is_marker(value)) {
        return hipErrorInvalidValue;
      }
      *out_buffer = value;
      saw_buffer = true;
    } else if (key == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
      if (saw_size || !value || iree_hip_launch_extra_value_is_marker(value)) {
        return hipErrorInvalidValue;
      }
      *out_buffer_size = *(const size_t*)value;
      saw_size = true;
    } else {
      return hipErrorInvalidValue;
    }
  }

  if (saw_buffer != saw_size) return hipErrorInvalidValue;
  if (*out_buffer_size != 0 && !*out_buffer) return hipErrorInvalidValue;
  return hipSuccess;
}

hipError_t iree_hip_validate_launch_block_configuration(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, size_t shared_memory_bytes) {
  if (!device) return hipErrorInvalidDevice;
  if (shared_memory_bytes > UINT32_MAX) return hipErrorInvalidConfiguration;

  const unsigned int block_dim[3] = {block_dim_x, block_dim_y, block_dim_z};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(block_dim); ++i) {
    if (block_dim[i] == 0) return hipErrorInvalidConfiguration;
    if (device->max_block_dim[i] != 0 &&
        block_dim[i] > device->max_block_dim[i]) {
      return hipErrorInvalidConfiguration;
    }
  }

  uint64_t threads_per_block = block_dim[0];
  if (block_dim[1] != 0 && threads_per_block > UINT64_MAX / block_dim[1]) {
    return hipErrorInvalidConfiguration;
  }
  threads_per_block *= block_dim[1];
  if (block_dim[2] != 0 && threads_per_block > UINT64_MAX / block_dim[2]) {
    return hipErrorInvalidConfiguration;
  }
  threads_per_block *= block_dim[2];
  const uint32_t device_max_threads = device->max_threads_per_block;
  const uint32_t symbol_max_threads =
      symbol ? symbol->function_attributes.maximum_threads_per_block : 0;
  if ((device_max_threads != 0 && threads_per_block > device_max_threads) ||
      (symbol_max_threads != 0 && threads_per_block > symbol_max_threads)) {
    return hipErrorInvalidConfiguration;
  }

  if (symbol &&
      iree_all_bits_set(
          symbol->function_attributes.provided_flags,
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY)) {
    const uint32_t configured_limit =
        iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
            &symbol->function_attributes);
    if (shared_memory_bytes > configured_limit) {
      return hipErrorInvalidConfiguration;
    }
  } else if (device->max_shared_memory_per_block != 0 &&
             shared_memory_bytes > device->max_shared_memory_per_block) {
    return hipErrorInvalidConfiguration;
  }
  return hipSuccess;
}

hipError_t iree_hip_validate_launch_configuration(
    iree_hal_streaming_device_t* device, iree_hal_streaming_symbol_t* symbol,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, size_t shared_memory_bytes) {
  if (!device) return hipErrorInvalidDevice;

  const unsigned int grid_dim[3] = {grid_dim_x, grid_dim_y, grid_dim_z};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(grid_dim); ++i) {
    if (grid_dim[i] == 0 || (device->max_grid_dim[i] != 0 &&
                             grid_dim[i] > device->max_grid_dim[i])) {
      return hipErrorInvalidConfiguration;
    }
  }

  return iree_hip_validate_launch_block_configuration(
      device, symbol, block_dim_x, block_dim_y, block_dim_z,
      shared_memory_bytes);
}
