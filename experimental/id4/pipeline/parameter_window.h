// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_WINDOW_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_WINDOW_H_

#include <stdint.h>

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

// Source representation used to populate compact parameter windows.
typedef uint32_t id4_pipeline_parameter_window_source_kind_t;

// Compact parameter-window source representation values.
typedef enum id4_pipeline_parameter_window_source_kind_e {
  // Invalid or unspecified parameter-window source.
  ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_INVALID = 0,
  // Plan-directed checkpoint gathers and encoders populate each window.
  ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_CHECKPOINT = 1,
  // Validated baked execution-layout entries are gathered directly.
  ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_EXECUTION_LAYOUT = 2,
} id4_pipeline_parameter_window_source_kind_e;

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
  // Plan parameter tensor index owning this request.
  iree_host_size_t parameter_tensor_index;
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
  // Plan whose logical parameter tensors are compacted into the window.
  const id4_pipeline_plan_t* plan;
  // Number of selected semantic program parameter tensor ordinals.
  iree_host_size_t parameter_tensor_count;
  // Selected semantic program parameter tensor ordinals.
  const uint32_t* parameter_tensor_ordinals;
} id4_pipeline_parameter_window_create_options_t;

// Options for deriving compact parameter loading descriptors from a window.
typedef struct id4_pipeline_parameter_window_schedule_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Plan that owns the original slab, request, and loading descriptors.
  id4_pipeline_plan_t* plan;
  // Window selecting the compact parameter materialization scope.
  const id4_pipeline_parameter_window_t* window;
} id4_pipeline_parameter_window_schedule_create_options_t;

// Options for loading a compact window from a baked execution-layout archive.
typedef struct
    id4_pipeline_parameter_window_execution_layout_schedule_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Plan that defines archive entry identity and compact target layouts.
  id4_pipeline_plan_t* plan;
  // Window selecting the compact parameter materialization scope.
  const id4_pipeline_parameter_window_t* window;
  // Provider scope containing the validated execution-layout archive.
  iree_string_view_t source_scope;
} id4_pipeline_parameter_window_execution_layout_schedule_create_options_t;

// Creates a compact parameter materialization window from existing plan
// metadata.
iree_status_t id4_pipeline_parameter_window_create(
    const id4_pipeline_parameter_window_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_t** out_window);

// Creates a compact window for every parameter tensor referenced by one plan
// region. This is the current semantic-region residency policy; constrained
// policies select tensors directly with |id4_pipeline_parameter_window_create|.
iree_status_t id4_pipeline_parameter_window_create_for_region(
    const id4_pipeline_plan_t* plan, iree_host_size_t region_index,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_t** out_window);

// Releases a compact parameter materialization window.
void id4_pipeline_parameter_window_release(
    id4_pipeline_parameter_window_t* window);

// Returns the number of selected semantic program parameter tensors.
iree_host_size_t id4_pipeline_parameter_window_parameter_tensor_count(
    const id4_pipeline_parameter_window_t* window);

// Returns selected semantic program parameter tensor ordinal |index| or
// UINT32_MAX when out of range.
uint32_t id4_pipeline_parameter_window_parameter_tensor_ordinal_at(
    const id4_pipeline_parameter_window_t* window, iree_host_size_t index);

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

// Creates compact parameter slab loads and plan-directed checkpoint load steps
// for |window|.
iree_status_t id4_pipeline_parameter_window_schedule_create(
    const id4_pipeline_parameter_window_schedule_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_schedule_t** out_schedule);

// Creates compact gather-only loads from baked execution-layout archive
// entries into |window|.
iree_status_t id4_pipeline_parameter_window_execution_layout_schedule_create(
    const id4_pipeline_parameter_window_execution_layout_schedule_create_options_t*
        options,
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
