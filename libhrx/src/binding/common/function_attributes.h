// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_FUNCTION_ATTRIBUTES_H_
#define LIBHRX_SRC_BINDING_COMMON_FUNCTION_ATTRIBUTES_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Indicates which compatibility function attributes are available.
typedef enum iree_hal_streaming_function_attribute_flag_bits_e {
  IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_NONE = 0u,
  IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK = 1u
                                                                         << 0,
  IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_SHARED_MEMORY = 1u << 1,
  IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_LOCAL_MEMORY = 1u << 2,
  IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_REGISTER_COUNT = 1u << 3,
  IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY = 1u << 4,
} iree_hal_streaming_function_attribute_flag_bits_t;
typedef uint32_t iree_hal_streaming_function_attribute_flags_t;

// Function attributes cached for HIP and CUDA compatibility APIs.
typedef struct iree_hal_streaming_function_attributes_t {
  // Attributes populated from generic HAL function and device information.
  iree_hal_streaming_function_attribute_flags_t provided_flags;
  // Maximum number of threads accepted in one block.
  uint32_t maximum_threads_per_block;
  // Fixed shared-memory bytes allocated per block.
  uint32_t fixed_shared_memory_size;
  // Fixed thread-local memory bytes allocated per thread.
  uint32_t fixed_local_memory_size;
  // Number of 32-bit registers allocated per thread.
  uint32_t register_count;
  // Maximum dynamic shared-memory limit accepted by a compatibility setter.
  uint32_t maximum_configurable_dynamic_shared_memory_size;
  // Current dynamic shared-memory limit applied to launches.
  iree_atomic_uint32_t configured_dynamic_shared_memory_size;
} iree_hal_streaming_function_attributes_t;

// Initializes compatibility attributes from generic loaded-function and device
// information.
iree_status_t iree_hal_streaming_function_attributes_initialize(
    const iree_hal_device_spec_t* device_spec,
    const iree_hal_executable_function_info_t* function_info,
    iree_hal_streaming_function_attributes_t* out_attributes);

// Returns the current dynamic shared-memory limit.
static inline uint32_t
iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
    const iree_hal_streaming_function_attributes_t* attributes) {
  return iree_atomic_load(&attributes->configured_dynamic_shared_memory_size,
                          iree_memory_order_relaxed);
}

// Updates the dynamic shared-memory limit when |value| is supported.
// Returns false when the generic HAL facts do not provide a limit or |value|
// exceeds the immutable configurable maximum.
static inline bool
iree_hal_streaming_function_attributes_try_set_dynamic_shared_memory_size(
    iree_hal_streaming_function_attributes_t* attributes, uint32_t value) {
  if (!iree_all_bits_set(
          attributes->provided_flags,
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY) ||
      value > attributes->maximum_configurable_dynamic_shared_memory_size) {
    return false;
  }
  iree_atomic_store(&attributes->configured_dynamic_shared_memory_size, value,
                    iree_memory_order_relaxed);
  return true;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_FUNCTION_ATTRIBUTES_H_
