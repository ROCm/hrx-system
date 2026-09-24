// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_OCCUPANCY_H_
#define LIBHRX_SRC_BINDING_COMMON_OCCUPANCY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns dynamic workgroup-local memory bytes required by |block_size|.
typedef iree_host_size_t (
    *iree_hal_streaming_block_size_to_dynamic_memory_fn_t)(int block_size);

// Queries the exact concurrent residency of a one-dimensional workgroup on
// |queue|. The executable must have been loaded for the queue family containing
// |queue|. The query enqueues no work and |out_concurrency| is unchanged on
// failure.
iree_status_t iree_hal_streaming_query_dispatch_occupancy(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, uint32_t block_size,
    uint32_t dynamic_workgroup_local_memory,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency);

// Checks whether |config| fits the concurrent residency of |queue|.
// Workgroup and grid products are computed with checked arithmetic and the
// occupancy query is issued against the exact queue and executable entry point.
// |out_exceeds_residency| is unchanged on failure.
iree_status_t iree_hal_streaming_check_cooperative_dispatch_residency(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, iree_hal_dispatch_config_t config,
    bool* out_exceeds_residency);

// Selects the block size with the greatest resident invocation count per
// scheduling domain. Equal scores select the larger block. The returned grid
// size is the total number of concurrently resident workgroups across the exact
// |queue|.
//
// |maximum_block_size| and |maximum_dynamic_workgroup_local_memory| carry the
// compatibility-visible function limits. |block_size_limit| further restricts
// the search when non-zero. |dynamic_memory_fn| replaces
// |fixed_dynamic_workgroup_local_memory| when provided. Both outputs are
// unchanged on failure.
iree_status_t iree_hal_streaming_select_optimal_dispatch_occupancy(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, uint32_t maximum_block_size,
    uint32_t maximum_dynamic_workgroup_local_memory,
    iree_host_size_t fixed_dynamic_workgroup_local_memory,
    iree_hal_streaming_block_size_to_dynamic_memory_fn_t dynamic_memory_fn,
    uint32_t block_size_limit, uint32_t* out_block_size,
    uint64_t* out_minimum_grid_size);

// Finds the greatest dynamic workgroup-local memory size that preserves at
// least |required_workgroup_count_per_domain| resident workgroups on the exact
// |queue|. Returns zero when the requested residency cannot be achieved even
// without dynamic memory. |out_dynamic_workgroup_local_memory| is unchanged on
// failure.
iree_status_t iree_hal_streaming_query_available_dynamic_workgroup_local_memory(
    iree_hal_queue_t* queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, uint32_t block_size,
    uint32_t required_workgroup_count_per_domain,
    uint32_t maximum_dynamic_workgroup_local_memory,
    uint32_t* out_dynamic_workgroup_local_memory);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_OCCUPANCY_H_
