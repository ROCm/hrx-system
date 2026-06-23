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

// Program planning behavior flags.
typedef uint32_t id4_pipeline_program_plan_flags_t;

// Program planning behavior flag bits.
typedef enum id4_pipeline_program_plan_flag_bits_e {
  // Includes diagnostic tap copies in the planned executable region.
  ID4_PIPELINE_PROGRAM_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS = 1u << 0,
} id4_pipeline_program_plan_flag_bits_t;

// Options for deriving a pipeline plan from a semantic program.
typedef struct id4_pipeline_program_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Planning behavior flags.
  id4_pipeline_program_plan_flags_t flags;
  // Stage name copied into the created plan.
  iree_string_view_t stage_name;
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
  // Binding-table slot used for the packed parameter slab.
  uint32_t parameter_slab_binding_slot;
  // HAL buffer parameters used for the packed parameter slab.
  iree_hal_buffer_params_t parameter_slab_target_params;
  // Required packed parameter slab base alignment.
  iree_device_size_t parameter_slab_alignment;
  // Alignment used when packing each parameter request into the slab.
  iree_device_size_t parameter_request_alignment;
  // Placement id used for the packed constant slab.
  id4_pipeline_device_placement_id_t constant_slab_placement_id;
  // Binding-table slot used for the packed constant slab.
  uint32_t constant_slab_binding_slot;
  // HAL buffer parameters used for the packed constant slab.
  iree_hal_buffer_params_t constant_slab_target_params;
  // Required packed constant slab base alignment.
  iree_device_size_t constant_slab_alignment;
  // Alignment used when packing each constant into the slab.
  iree_device_size_t constant_request_alignment;
  // Placement id used for all planned Loom kernel specializations.
  id4_pipeline_device_placement_id_t kernel_placement_id;
  // Placement id used for the derived executable region.
  id4_pipeline_device_placement_id_t region_placement_id;
  // HAL buffer parameters used for the region local transient slab.
  iree_hal_buffer_params_t region_local_slab_params;
  // Required region local transient slab base alignment.
  iree_device_size_t region_local_slab_alignment;
  // Required alignment for local tensor suballocations in the region slab.
  iree_device_size_t region_local_tensor_alignment;
  // Exact issue-time binding-table capacity for the derived region.
  iree_host_size_t region_binding_capacity;
  // Binding-table slot reserved for the derived region local slab.
  uint32_t region_local_binding_slot;
  // First binding-table slot assigned to external boundary tensors.
  uint32_t region_boundary_binding_slot_base;
  // Caller-owned diagnostic tap names to capture.
  iree_string_view_list_t diagnostic_tap_names;
  // First binding-table slot assigned to diagnostic tap tensors.
  uint32_t diagnostic_tap_binding_slot_base;
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
