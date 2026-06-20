// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PLAN_H_
#define EXPERIMENTAL_ID4_PIPELINE_PLAN_H_

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/parameter_slab.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque planned execution object.
typedef struct id4_pipeline_plan_t id4_pipeline_plan_t;

// Device placement selected during planning.
typedef struct id4_pipeline_device_placement_t {
  // Human-readable role for diagnostics.
  iree_string_view_t role;
  // Device index within the retained HAL device group.
  iree_host_size_t device_index;
  // Queue affinity used for work submitted on this placement.
  iree_hal_queue_affinity_t queue_affinity;
} id4_pipeline_device_placement_t;

// Options for creating an inspectable plan.
typedef struct id4_pipeline_plan_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stage name used in diagnostics.
  iree_string_view_t stage_name;
  // Device group retained by the plan.
  iree_hal_device_group_t* device_group;
  // Default device index used when no placement array is provided.
  iree_host_size_t default_device_index;
  // Default queue affinity used when no placement array is provided.
  iree_hal_queue_affinity_t default_queue_affinity;
  // Number of explicit placements.
  iree_host_size_t placement_count;
  // Explicit placements to copy into the plan.
  const id4_pipeline_device_placement_t* placements;
  // Number of parameter slabs.
  iree_host_size_t parameter_slab_count;
  // Parameter slabs to copy into the plan.
  const id4_pipeline_parameter_slab_plan_t* parameter_slabs;
  // Optional diagnostics sink for plan creation events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_plan_create_options_t;

// Creates a plan by copying placement and parameter-slab metadata.
iree_status_t id4_pipeline_plan_create(
    const id4_pipeline_plan_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan);

// Retains |plan| for the caller.
void id4_pipeline_plan_retain(id4_pipeline_plan_t* plan);

// Releases |plan| from the caller.
void id4_pipeline_plan_release(id4_pipeline_plan_t* plan);

// Returns the stage name copied into |plan|.
iree_string_view_t id4_pipeline_plan_stage_name(
    const id4_pipeline_plan_t* plan);

// Returns the retained device group associated with |plan|.
iree_hal_device_group_t* id4_pipeline_plan_device_group(
    const id4_pipeline_plan_t* plan);

// Returns the number of placements in |plan|.
iree_host_size_t id4_pipeline_plan_placement_count(
    const id4_pipeline_plan_t* plan);

// Returns placement |index| or NULL when out of range.
const id4_pipeline_device_placement_t* id4_pipeline_plan_placement_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Returns the number of parameter slabs in |plan|.
iree_host_size_t id4_pipeline_plan_parameter_slab_count(
    const id4_pipeline_plan_t* plan);

// Returns parameter slab |index| or NULL when out of range.
const id4_pipeline_parameter_slab_plan_t* id4_pipeline_plan_parameter_slab_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Loads all planned parameter slabs through |provider|.
iree_status_t id4_pipeline_plan_load_parameter_slabs(
    const id4_pipeline_plan_t* plan, iree_io_parameter_provider_t* provider,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set);

// Appends deterministic JSON describing the plan.
iree_status_t id4_pipeline_plan_format_json(const id4_pipeline_plan_t* plan,
                                            iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PLAN_H_
