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
