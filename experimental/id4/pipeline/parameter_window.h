// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_WINDOW_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_WINDOW_H_

#include "experimental/id4/pipeline/plan.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque compact parameter materialization plan for a contiguous region window.
typedef struct id4_pipeline_parameter_window_t id4_pipeline_parameter_window_t;

// Opaque compact parameter loading schedule derived from a window.
typedef struct id4_pipeline_parameter_window_schedule_t
    id4_pipeline_parameter_window_schedule_t;

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

// Compact target slab represented inside one parameter materialization window.
typedef struct id4_pipeline_parameter_window_slab_t {
  // Original parameter slab index represented by this compact window slab.
  iree_host_size_t original_slab_index;
  // Issue-time binding-table slot used by this compact slab.
  uint32_t binding_slot;
  // HAL buffer parameters inherited from the original parameter slab.
  iree_hal_buffer_params_t target_params;
  // Compact slab byte length after packing window-local requests.
  iree_device_size_t byte_length;
  // Compact slab base and request packing alignment.
  iree_device_size_t alignment;
  // First compact request ordinal belonging to this slab.
  iree_host_size_t request_offset;
  // Number of compact requests belonging to this slab.
  iree_host_size_t request_count;
} id4_pipeline_parameter_window_slab_t;

// Compact target request represented inside one parameter materialization
// window.
typedef struct id4_pipeline_parameter_window_request_t {
  // Original parameter slab index containing this request.
  iree_host_size_t original_slab_index;
  // Original request index within original_slab_index.
  iree_host_size_t original_request_index;
  // Global request ordinal in the containing plan.
  iree_host_size_t global_request_index;
  // Source parameter span and compact target buffer span.
  iree_io_parameter_span_t span;
} id4_pipeline_parameter_window_request_t;

// Options for creating a compact parameter materialization window.
typedef struct id4_pipeline_parameter_window_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Plan whose parameter load groups define the window.
  const id4_pipeline_plan_t* plan;
  // First contiguous region represented by the window.
  iree_host_size_t region_offset;
  // Number of contiguous regions represented by the window.
  iree_host_size_t region_count;
} id4_pipeline_parameter_window_create_options_t;

// Options for deriving compact parameter loading descriptors from a window.
typedef struct id4_pipeline_parameter_window_schedule_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Plan that owns the original slab, request, and loading descriptors.
  id4_pipeline_plan_t* plan;
  // Window selecting the region-local materialization scope.
  const id4_pipeline_parameter_window_t* window;
} id4_pipeline_parameter_window_schedule_create_options_t;

// Queries exact compact-window allocations and transfer volumes using the same
// packing and encoder-staging calculations as warm issue.
iree_status_t id4_pipeline_parameter_window_query_statistics(
    const id4_pipeline_parameter_window_statistics_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_statistics_t* out_statistics);

// Creates a compact parameter materialization window from existing plan
// metadata.
iree_status_t id4_pipeline_parameter_window_create(
    const id4_pipeline_parameter_window_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_t** out_window);

// Releases a compact parameter materialization window.
void id4_pipeline_parameter_window_release(
    id4_pipeline_parameter_window_t* window);

// Returns the first contiguous region represented by |window|.
iree_host_size_t id4_pipeline_parameter_window_region_offset(
    const id4_pipeline_parameter_window_t* window);

// Returns the number of contiguous regions represented by |window|.
iree_host_size_t id4_pipeline_parameter_window_region_count(
    const id4_pipeline_parameter_window_t* window);

// Returns the number of compact slabs in |window|.
iree_host_size_t id4_pipeline_parameter_window_slab_count(
    const id4_pipeline_parameter_window_t* window);

// Returns compact slab |index| or NULL when out of range.
const id4_pipeline_parameter_window_slab_t*
id4_pipeline_parameter_window_slab_at(
    const id4_pipeline_parameter_window_t* window, iree_host_size_t index);

// Returns the number of compact requests in |window|.
iree_host_size_t id4_pipeline_parameter_window_request_count(
    const id4_pipeline_parameter_window_t* window);

// Returns compact request |index| or NULL when out of range.
const id4_pipeline_parameter_window_request_t*
id4_pipeline_parameter_window_request_at(
    const id4_pipeline_parameter_window_t* window, iree_host_size_t index);

// Returns the number of original parameter load groups used by |window|.
iree_host_size_t id4_pipeline_parameter_window_load_group_count(
    const id4_pipeline_parameter_window_t* window);

// Returns original load group ordinal |index| or IREE_HOST_SIZE_MAX when out of
// range.
iree_host_size_t id4_pipeline_parameter_window_load_group_at(
    const id4_pipeline_parameter_window_t* window, iree_host_size_t index);

// Resolves a global plan request ordinal to its compact window request, or NULL
// when the request is not materialized by |window|.
const id4_pipeline_parameter_window_request_t*
id4_pipeline_parameter_window_resolve_request(
    const id4_pipeline_parameter_window_t* window,
    iree_host_size_t global_request_index);

// Creates compact parameter slab loads and load steps for |window|.
iree_status_t id4_pipeline_parameter_window_schedule_create(
    const id4_pipeline_parameter_window_schedule_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_schedule_t** out_schedule);

// Releases a compact parameter loading schedule.
void id4_pipeline_parameter_window_schedule_release(
    id4_pipeline_parameter_window_schedule_t* schedule);

// Returns the number of compact parameter slab loads in |schedule|.
iree_host_size_t id4_pipeline_parameter_window_schedule_load_count(
    const id4_pipeline_parameter_window_schedule_t* schedule);

// Returns compact parameter slab loads owned by |schedule|.
const id4_pipeline_parameter_slab_load_t*
id4_pipeline_parameter_window_schedule_loads(
    const id4_pipeline_parameter_window_schedule_t* schedule);

// Returns the number of compact load steps in |schedule|.
iree_host_size_t id4_pipeline_parameter_window_schedule_load_step_count(
    const id4_pipeline_parameter_window_schedule_t* schedule);

// Returns compact parameter load steps owned by |schedule|.
const id4_pipeline_parameter_load_step_t*
id4_pipeline_parameter_window_schedule_load_steps(
    const id4_pipeline_parameter_window_schedule_t* schedule);

// Returns the number of compact load groups in |schedule|.
iree_host_size_t id4_pipeline_parameter_window_schedule_load_group_count(
    const id4_pipeline_parameter_window_schedule_t* schedule);

// Returns the original plan load group ordinal for compact load group |index|.
iree_host_size_t id4_pipeline_parameter_window_schedule_original_load_group_at(
    const id4_pipeline_parameter_window_schedule_t* schedule,
    iree_host_size_t index);

// Resolves an original plan load group ordinal to its compact schedule ordinal.
iree_host_size_t id4_pipeline_parameter_window_schedule_compact_load_group(
    const id4_pipeline_parameter_window_schedule_t* schedule,
    iree_host_size_t original_load_group_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_WINDOW_H_
