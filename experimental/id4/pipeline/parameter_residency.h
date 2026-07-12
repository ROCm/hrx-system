// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_RESIDENCY_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_RESIDENCY_H_

#include <stdint.h>

#include "experimental/id4/pipeline/parameter_window_statistics.h"
#include "experimental/id4/pipeline/plan.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque barrier-aligned compact parameter residency plan.
typedef struct id4_pipeline_parameter_residency_plan_t
    id4_pipeline_parameter_residency_plan_t;

// One contiguous source-program interval with a fixed compact parameter set.
typedef struct id4_pipeline_parameter_residency_segment_t {
  // Semantic region containing this segment.
  uint32_t semantic_region_id;
  // First source-program operation represented by this segment.
  iree_host_size_t source_operation_offset;
  // Number of contiguous source-program operations in this segment.
  iree_host_size_t source_operation_count;
  // Number of Loom dispatches represented by this segment.
  iree_host_size_t dispatch_count;
  // Number of authored barriers represented by this segment.
  iree_host_size_t barrier_count;
  // Number of logical parameter tensors selected by this segment.
  iree_host_size_t parameter_tensor_count;
  // Program tensor ordinals in containing-plan parameter order.
  const uint32_t* parameter_tensor_ordinals;
  // Compact physical parameter window owned by the residency plan.
  const id4_pipeline_parameter_window_t* window;
  // Exact resources and traffic required to populate the compact window.
  id4_pipeline_parameter_window_resource_statistics_t resource_statistics;
} id4_pipeline_parameter_residency_segment_t;

// Aggregate statistics for a barrier-aligned parameter residency plan.
typedef struct id4_pipeline_parameter_residency_statistics_t {
  // Number of semantic regions represented by the source plan.
  iree_host_size_t semantic_region_count;
  // Number of constrained residency segments produced across all regions.
  iree_host_size_t segment_count;
  // Maximum compact target bytes accepted by the planner.
  iree_device_size_t maximum_target_byte_length;
  // Target bytes required by the fully resident parameter slabs.
  iree_device_size_t resident_target_byte_length;
  // Largest compact target allocation required by one segment.
  iree_device_size_t peak_segment_target_byte_length;
  // Largest encoder staging allocation required by one segment.
  iree_device_size_t peak_encoder_staging_byte_length;
  // Largest compact target plus encoder staging live set for one segment.
  iree_device_size_t peak_segment_live_byte_length;
  // Compact target bytes for the union of all parameters used by any segment.
  iree_device_size_t unique_target_byte_length;
  // Provider source bytes for the union of all parameters used by any segment.
  iree_device_size_t unique_source_transfer_byte_length;
  // Sum of compact target allocation bytes across all segments.
  iree_device_size_t total_target_byte_length;
  // Sum of provider source bytes transferred across all segments.
  iree_device_size_t total_source_transfer_byte_length;
  // Additional target allocation bytes caused by residency-segment repetition.
  iree_device_size_t duplicated_target_byte_length;
  // Additional source transfer bytes caused by residency-segment repetition.
  iree_device_size_t duplicated_source_transfer_byte_length;
  // Sum of compact parameter load groups across all segments.
  iree_host_size_t total_load_group_count;
  // Sum of encoded load steps across all segments.
  iree_host_size_t total_encode_load_step_count;
} id4_pipeline_parameter_residency_statistics_t;

// Options for planning barrier-aligned compact parameter residency segments.
typedef struct id4_pipeline_parameter_residency_plan_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Program-backed source plan whose semantic regions are segmented.
  const id4_pipeline_plan_t* plan;
  // Source representation used to populate planned compact windows.
  id4_pipeline_parameter_window_source_kind_t source_kind;
  // Maximum compact target allocation bytes permitted in one segment.
  iree_device_size_t maximum_target_byte_length;
  // Maximum provider source bytes packed into one encoder staging chunk.
  iree_device_size_t encoder_staging_chunk_byte_capacity;
} id4_pipeline_parameter_residency_plan_create_options_t;

// Creates a deterministic greedy residency plan without issuing device work.
iree_status_t id4_pipeline_parameter_residency_plan_create(
    const id4_pipeline_parameter_residency_plan_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_residency_plan_t** out_residency_plan);

// Retains |residency_plan| for the caller.
void id4_pipeline_parameter_residency_plan_retain(
    id4_pipeline_parameter_residency_plan_t* residency_plan);

// Releases a residency plan and its compact parameter windows.
void id4_pipeline_parameter_residency_plan_release(
    id4_pipeline_parameter_residency_plan_t* residency_plan);

// Returns the source plan retained by |residency_plan|.
const id4_pipeline_plan_t* id4_pipeline_parameter_residency_plan_source_plan(
    const id4_pipeline_parameter_residency_plan_t* residency_plan);

// Returns aggregate statistics derived from the fixed segment plan.
id4_pipeline_parameter_residency_statistics_t
id4_pipeline_parameter_residency_plan_statistics(
    const id4_pipeline_parameter_residency_plan_t* residency_plan);

// Returns the number of ordered residency segments.
iree_host_size_t id4_pipeline_parameter_residency_plan_segment_count(
    const id4_pipeline_parameter_residency_plan_t* residency_plan);

// Returns ordered segment |index| or NULL when out of range.
const id4_pipeline_parameter_residency_segment_t*
id4_pipeline_parameter_residency_plan_segment_at(
    const id4_pipeline_parameter_residency_plan_t* residency_plan,
    iree_host_size_t index);

// Queries exact live parameter resources for the issue prefetch distance.
iree_status_t id4_pipeline_parameter_residency_plan_query_live_statistics(
    const id4_pipeline_parameter_residency_plan_t* residency_plan,
    iree_host_size_t parameter_load_prefetch_segment_distance,
    id4_pipeline_parameter_window_statistics_t* out_statistics);

// Appends deterministic JSON describing the residency plan and its segments.
iree_status_t id4_pipeline_parameter_residency_plan_format_json(
    const id4_pipeline_parameter_residency_plan_t* residency_plan,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_RESIDENCY_H_
