// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PROGRAM_PLAN_H_
#define EXPERIMENTAL_ID4_PIPELINE_PROGRAM_PLAN_H_

#include <stdint.h>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/program.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Options for deriving a pipeline plan from a semantic program.
typedef struct id4_pipeline_program_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Semantic program to lower into plan metadata.
  const id4_pipeline_program_t* program;
  // Device group retained by the created plan.
  iree_hal_device_group_t* device_group;
  // Number of explicit placements.
  iree_host_size_t placement_count;
  // Explicit placements copied into the created plan.
  const id4_pipeline_device_placement_t* placements;
  // Provider scope containing all parameter keys in the program.
  iree_string_view_t parameter_scope;
  // Placement id used for the packed parameter slab.
  id4_pipeline_device_placement_id_t parameter_slab_placement_id;
  // HAL buffer parameters used for the packed parameter slab.
  iree_hal_buffer_params_t parameter_slab_target_params;
  // Required packed parameter slab base alignment.
  iree_device_size_t parameter_slab_alignment;
  // Alignment used when packing each parameter request into the slab.
  iree_device_size_t parameter_request_alignment;
  // Placement id used for all planned Loom kernel specializations.
  id4_pipeline_device_placement_id_t kernel_placement_id;
  // Placement id used for the derived executable region.
  id4_pipeline_device_placement_id_t region_placement_id;
  // Exact issue-time binding-table capacity for the derived region.
  iree_host_size_t region_binding_capacity;
  // Binding-table slot reserved for the derived region local slab.
  uint32_t region_local_binding_slot;
  // Diagnostics sink for plan creation events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_program_plan_options_t;

// Derives an inspectable plan from a semantic program.
iree_status_t id4_pipeline_program_create_plan(
    const id4_pipeline_program_plan_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PROGRAM_PLAN_H_
