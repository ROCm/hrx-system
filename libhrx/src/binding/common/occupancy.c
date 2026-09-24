// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/occupancy.h"

#include <limits.h>

iree_status_t iree_hal_streaming_query_dispatch_occupancy(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, uint32_t block_size,
    uint32_t dynamic_workgroup_local_memory,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  const iree_hal_queue_dispatch_concurrency_params_t params = {
      .workgroup_size = {block_size, 1, 1},
      .dynamic_workgroup_local_memory = dynamic_workgroup_local_memory,
  };
  iree_hal_queue_dispatch_concurrency_t concurrency;
  iree_status_t status = iree_hal_queue_query_dispatch_concurrency(
      queue, executable, function, params,
      IREE_HAL_QUEUE_DISPATCH_CONCURRENCY_FLAG_NONE, &concurrency);
  if (iree_status_is_ok(status)) {
    *out_concurrency = concurrency;
  }
  return status;
}

iree_status_t iree_hal_streaming_check_cooperative_dispatch_residency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, iree_hal_dispatch_config_t config,
    bool* out_exceeds_residency) {
  uint64_t block_size_xy = 0;
  uint64_t block_size = 0;
  if (IREE_UNLIKELY(!iree_checked_mul_u64(config.workgroup_size[0],
                                          config.workgroup_size[1],
                                          &block_size_xy) ||
                    !iree_checked_mul_u64(
                        block_size_xy, config.workgroup_size[2], &block_size) ||
                    block_size == 0 || block_size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cooperative workgroup size is invalid");
  }

  iree_hal_queue_dispatch_concurrency_t concurrency;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_query_dispatch_occupancy(
      queue, executable, function, (uint32_t)block_size,
      config.dynamic_workgroup_local_memory, &concurrency));
  const uint64_t maximum_workgroup_count =
      iree_hal_queue_dispatch_concurrency_total_workgroup_count(concurrency);

  uint64_t grid_size_xy = 0;
  uint64_t grid_size = 0;
  const bool grid_size_valid =
      iree_checked_mul_u64(config.workgroup_count[0], config.workgroup_count[1],
                           &grid_size_xy) &&
      iree_checked_mul_u64(grid_size_xy, config.workgroup_count[2], &grid_size);
  *out_exceeds_residency =
      !grid_size_valid || grid_size > maximum_workgroup_count;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_select_optimal_dispatch_occupancy(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, uint32_t maximum_block_size,
    uint32_t maximum_dynamic_workgroup_local_memory,
    iree_host_size_t fixed_dynamic_workgroup_local_memory,
    iree_hal_streaming_block_size_to_dynamic_memory_fn_t dynamic_memory_fn,
    uint32_t block_size_limit, uint32_t* out_block_size,
    uint64_t* out_minimum_grid_size) {
  if (IREE_UNLIKELY(maximum_block_size == 0)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "function does not report a maximum block size");
  }
  if (IREE_UNLIKELY(maximum_block_size > INT_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "maximum block size %u exceeds the compatibility API range",
        maximum_block_size);
  }

  uint32_t search_block_size = maximum_block_size;
  if (block_size_limit != 0) {
    search_block_size = iree_min(search_block_size, block_size_limit);
  }

  uint32_t best_block_size = 0;
  uint64_t best_resident_invocation_count = 0;
  iree_hal_queue_dispatch_concurrency_t best_concurrency = {0};
  for (uint32_t candidate_block_size = search_block_size;
       candidate_block_size != 0; --candidate_block_size) {
    const iree_host_size_t dynamic_workgroup_local_memory =
        dynamic_memory_fn ? dynamic_memory_fn((int)candidate_block_size)
                          : fixed_dynamic_workgroup_local_memory;
    if (dynamic_workgroup_local_memory >
            maximum_dynamic_workgroup_local_memory ||
        dynamic_workgroup_local_memory > UINT32_MAX) {
      continue;
    }

    iree_hal_queue_dispatch_concurrency_t concurrency;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_query_dispatch_occupancy(
        queue, executable, function, candidate_block_size,
        (uint32_t)dynamic_workgroup_local_memory, &concurrency));
    const uint64_t resident_invocation_count =
        (uint64_t)candidate_block_size *
        concurrency.maximum_concurrent_workgroup_count_per_domain;
    if (resident_invocation_count > best_resident_invocation_count) {
      best_block_size = candidate_block_size;
      best_resident_invocation_count = resident_invocation_count;
      best_concurrency = concurrency;
    }
  }

  const uint64_t minimum_grid_size =
      best_block_size == 0
          ? 0
          : iree_hal_queue_dispatch_concurrency_total_workgroup_count(
                best_concurrency);
  *out_block_size = best_block_size;
  *out_minimum_grid_size = minimum_grid_size;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_query_available_dynamic_workgroup_local_memory(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, uint32_t block_size,
    uint32_t required_workgroup_count_per_domain,
    uint32_t maximum_dynamic_workgroup_local_memory,
    uint32_t* out_dynamic_workgroup_local_memory) {
  if (IREE_UNLIKELY(block_size == 0 ||
                    required_workgroup_count_per_domain == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "block size and required workgroup count must be positive");
  }

  iree_hal_queue_dispatch_concurrency_t concurrency;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_query_dispatch_occupancy(
      queue, executable, function, block_size,
      /*dynamic_workgroup_local_memory=*/0, &concurrency));
  if (concurrency.maximum_concurrent_workgroup_count_per_domain <
      required_workgroup_count_per_domain) {
    *out_dynamic_workgroup_local_memory = 0;
    return iree_ok_status();
  }

  uint32_t lower_bound = 0;
  uint32_t upper_bound = maximum_dynamic_workgroup_local_memory;
  // Increasing per-workgroup local memory cannot increase residency, so the
  // exact query provides a monotonic predicate over the byte range.
  while (lower_bound < upper_bound) {
    const uint32_t candidate =
        lower_bound + (uint32_t)(((uint64_t)upper_bound - lower_bound + 1) / 2);
    IREE_RETURN_IF_ERROR(iree_hal_streaming_query_dispatch_occupancy(
        queue, executable, function, block_size, candidate, &concurrency));
    if (concurrency.maximum_concurrent_workgroup_count_per_domain >=
        required_workgroup_count_per_domain) {
      lower_bound = candidate;
    } else {
      upper_bound = candidate - 1;
    }
  }

  *out_dynamic_workgroup_local_memory = lower_bound;
  return iree_ok_status();
}
