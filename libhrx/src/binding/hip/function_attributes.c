// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/function_attributes.h"

#include <limits.h>
#include <stdint.h>

#include "common/context.h"
#include "common/device.h"
#include "iree/base/api.h"

static hipError_t iree_hip_function_binary_version(
    const iree_hal_streaming_symbol_t* symbol, int* out_version) {
  const iree_hal_streaming_device_t* device =
      symbol->module->context->device_entry;
  const uint64_t version = (uint64_t)device->compute_capability_major * 10 +
                           device->compute_capability_minor;
  if (version > INT_MAX) {
    return hipErrorInvalidValue;
  }
  *out_version = (int)version;
  return hipSuccess;
}

hipError_t iree_hip_function_get_attribute(
    const iree_hal_streaming_symbol_t* symbol, hipFuncAttribute_t attribute,
    int* out_value) {
  const iree_hal_streaming_function_attributes_t* attributes =
      &symbol->function_attributes;
  iree_hal_streaming_function_attribute_flags_t required_flags = 0;
  int value = 0;
  switch (attribute) {
    case hipFuncAttributeMaxThreadsPerBlock:
      required_flags =
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK;
      value = (int)attributes->maximum_threads_per_block;
      break;
    case hipFuncAttributeSharedSizeBytes:
      required_flags =
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_SHARED_MEMORY;
      value = (int)attributes->fixed_shared_memory_size;
      break;
    case hipFuncAttributeConstSizeBytes:
      // AMDHSA has no per-kernel constant-memory resource. Match AMD HIP's
      // unavailable value instead of interpreting argument or image bytes.
      value = -1;
      break;
    case hipFuncAttributeLocalSizeBytes:
      required_flags =
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_LOCAL_MEMORY;
      value = (int)attributes->fixed_local_memory_size;
      break;
    case hipFuncAttributeNumRegs:
      required_flags =
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_REGISTER_COUNT;
      value = (int)attributes->register_count;
      break;
    case hipFuncAttributePtxVersion:
    case hipFuncAttributeBinaryVersion:
      return iree_hip_function_binary_version(symbol, out_value);
    case hipFuncAttributeCacheModeCA:
      break;
    case hipFuncAttributeMaxDynamicSharedSizeBytes:
      required_flags =
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY;
      value = (int)
          iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
              attributes);
      break;
    case hipFuncAttributePreferredSharedMemoryCarveout:
      value = iree_atomic_load(&symbol->preferred_shared_memory_carveout,
                               iree_memory_order_relaxed);
      break;
    default:
      return hipErrorInvalidValue;
  }
  if (!iree_all_bits_set(attributes->provided_flags, required_flags)) {
    return hipErrorNotSupported;
  }
  *out_value = value;
  return hipSuccess;
}

hipError_t iree_hip_function_get_attributes(
    const iree_hal_streaming_symbol_t* symbol,
    hipFuncAttributes* out_attributes) {
  const iree_hal_streaming_function_attributes_t* attributes =
      &symbol->function_attributes;
  const iree_hal_streaming_function_attribute_flags_t required_flags =
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK |
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_SHARED_MEMORY |
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_LOCAL_MEMORY |
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_REGISTER_COUNT |
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY;
  if (!iree_all_bits_set(attributes->provided_flags, required_flags)) {
    return hipErrorNotSupported;
  }
  int binary_version = 0;
  const hipError_t result =
      iree_hip_function_binary_version(symbol, &binary_version);
  if (result != hipSuccess) {
    return result;
  }
  *out_attributes = (hipFuncAttributes){
      .binaryVersion = binary_version,
      .cacheModeCA = 0,
      .constSizeBytes = SIZE_MAX,
      .localSizeBytes = attributes->fixed_local_memory_size,
      .maxDynamicSharedSizeBytes = (int)
          iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
              attributes),
      .maxThreadsPerBlock = (int)attributes->maximum_threads_per_block,
      .numRegs = (int)attributes->register_count,
      .preferredShmemCarveout = iree_atomic_load(
          &symbol->preferred_shared_memory_carveout, iree_memory_order_relaxed),
      .ptxVersion = binary_version,
      .sharedSizeBytes = attributes->fixed_shared_memory_size,
  };
  return hipSuccess;
}
