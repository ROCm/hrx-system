// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_WINDOW_STATISTICS_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_WINDOW_STATISTICS_H_

#include "experimental/id4/pipeline/parameter_window.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Options for querying compact parameter materialization resource usage.
typedef struct id4_pipeline_parameter_window_statistics_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Plan whose independently materialized region windows are measured.
  const id4_pipeline_plan_t* plan;
  // Maximum number of adjacent region windows live due to prefetch.
  iree_host_size_t concurrent_window_count;
  // Maximum provider source bytes packed into one encoder staging chunk.
  iree_device_size_t encoder_staging_chunk_byte_capacity;
} id4_pipeline_parameter_window_statistics_options_t;

// Exact resource and transfer statistics for compact parameter windows.
typedef struct id4_pipeline_parameter_window_statistics_t {
  // Maximum number of adjacent independently allocated windows considered.
  iree_host_size_t concurrent_window_count;
  // Number of independently materialized single-region windows in the plan.
  iree_host_size_t window_count;
  // Full target bytes across all resident parameter slabs.
  iree_device_size_t full_slab_target_byte_length;
  // Largest simultaneously live compact target allocation set.
  iree_device_size_t peak_target_byte_length;
  // Stage-owned encoder staging bytes shared by all compact region windows.
  iree_device_size_t encoder_staging_byte_length;
  // Largest simultaneously live target plus staging allocation set.
  iree_device_size_t peak_live_byte_length;
  // Largest source transfer volume represented by one concurrent window set.
  iree_device_size_t peak_source_transfer_byte_length;
  // Sum of compact target allocation bytes across all region windows.
  iree_device_size_t total_target_byte_length;
  // Sum of provider source transfer bytes across all region windows.
  iree_device_size_t total_source_transfer_byte_length;
  // Maximum load-group count represented by one concurrent window set.
  iree_host_size_t peak_load_group_count;
  // Sum of load groups submitted across all region windows.
  iree_host_size_t total_load_group_count;
  // Maximum encoded load-step count in one concurrent window set.
  iree_host_size_t peak_encode_load_step_count;
  // Sum of encoded load steps submitted across all region windows.
  iree_host_size_t total_encode_load_step_count;
  // Target payload byte length of the largest single load group.
  iree_device_size_t largest_load_group_target_byte_length;
  // Plan-local index of the largest single load group.
  iree_host_size_t largest_load_group_index;
  // Target payload byte length of the largest single parameter request.
  iree_device_size_t largest_request_target_byte_length;
  // Global plan request index of the largest single parameter request.
  iree_host_size_t largest_request_index;
  // Plan-local load group containing the largest single parameter request.
  iree_host_size_t largest_request_load_group_index;
} id4_pipeline_parameter_window_statistics_t;

// Exact resources and traffic required by one explicit compact window.
typedef struct id4_pipeline_parameter_window_resource_statistics_t {
  // Compact target allocation bytes across all selected semantic slabs.
  iree_device_size_t target_byte_length;
  // Shared encoder staging bytes required while populating the window.
  iree_device_size_t encoder_staging_byte_length;
  // Provider source bytes transferred while populating the window.
  iree_device_size_t source_transfer_byte_length;
  // Number of compact load groups submitted for the window.
  iree_host_size_t load_group_count;
  // Number of encoded load steps submitted for the window.
  iree_host_size_t encode_load_step_count;
} id4_pipeline_parameter_window_resource_statistics_t;

// Queries exact resources and traffic for one explicit compact window.
iree_status_t id4_pipeline_parameter_window_query_resource_statistics(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    id4_pipeline_parameter_window_source_kind_t source_kind,
    iree_device_size_t encoder_staging_chunk_byte_capacity,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_resource_statistics_t* out_statistics);

// Queries exact compact-window allocations and transfer volumes using the same
// packing and encoder-staging calculations as warm issue.
iree_status_t id4_pipeline_parameter_window_query_statistics(
    const id4_pipeline_parameter_window_statistics_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_statistics_t* out_statistics);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_WINDOW_STATISTICS_H_
