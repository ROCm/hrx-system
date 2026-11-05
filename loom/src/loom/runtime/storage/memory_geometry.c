// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_geometry.h"

iree_status_t loom_runtime_memory_pool_query_geometry(
    const iree_hal_virtual_buffer_capabilities_t* capabilities,
    iree_hal_pool_t* storage_pool, iree_device_size_t backing_size,
    loom_runtime_memory_geometry_t* out_geometry) {
  *out_geometry = (loom_runtime_memory_geometry_t){0};
  const iree_hal_memory_mapping_operations_t required =
      (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_BIND) |
      (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND);
  if (!capabilities->max_byte_length ||
      !iree_all_bits_set(capabilities->operations, required)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "elastic backing requires bind and unbind support");
  }
  const iree_hal_pool_mapping_requirements_t requirements =
      iree_hal_pool_query_mapping_requirements(storage_pool);
  const iree_device_size_t minimum =
      iree_max(capabilities->reservation_granularity,
               iree_max(capabilities->mapping_granularity,
                        requirements.minimum_granularity));
  iree_hal_pool_capabilities_t source_capabilities;
  iree_hal_pool_query_capabilities(storage_pool, &source_capabilities);
  const iree_device_size_t minimum_size = iree_device_align(
      iree_max(minimum, source_capabilities.min_allocation_size), minimum);
  if (!backing_size) {
    backing_size = iree_max(minimum_size, requirements.recommended_granularity);
  }
  if (!iree_device_size_is_valid_alignment(backing_size) ||
      backing_size > INT64_MAX || backing_size % minimum) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "slab size must be a power-of-two multiple of %" PRIdsz, minimum);
  }
  if (source_capabilities.max_allocation_size &&
      backing_size > source_capabilities.max_allocation_size) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "slab size %" PRIdsz " exceeds backing source limit %" PRIdsz,
        backing_size, source_capabilities.max_allocation_size);
  }
  if (minimum > source_capabilities.max_allocation_alignment) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "backing source cannot satisfy slab alignment");
  }
  if (backing_size < minimum_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "slab size is smaller than source minimum %" PRIdsz,
                            minimum_size);
  }
  *out_geometry = (loom_runtime_memory_geometry_t){
      .mapping_size = backing_size,
      .backing_size = backing_size,
      .granularity = minimum,
      .minimum_size = minimum_size,
  };
  return iree_ok_status();
}
