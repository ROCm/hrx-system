// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_IDEOGRAM4_LORA_BAKE_PLAN_H_
#define EXPERIMENTAL_ID4_IDEOGRAM4_LORA_BAKE_PLAN_H_

#include "experimental/id4/ideogram4/lora.h"
#include "experimental/id4/pipeline/plan.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable schedule for baking one LoRA topology into compact DiT weights.
typedef struct id4_ideogram4_lora_bake_plan_t id4_ideogram4_lora_bake_plan_t;

// Target-relative byte range in the LoRA-patchable parameter slab.
typedef struct id4_ideogram4_lora_bake_parameter_range_t {
  // Byte offset from the start of the patchable parameter slab.
  iree_device_size_t offset;
  // Byte length of the parameter tensor.
  iree_device_size_t length;
} id4_ideogram4_lora_bake_parameter_range_t;

// Packed device working-set layout reused across one target's bake windows.
typedef struct id4_ideogram4_lora_bake_working_set_t {
  // F32 adapter strengths ordered by topology adapter ordinal.
  id4_ideogram4_lora_bake_parameter_range_t strengths;
  // Reusable source-layout BF16 down projection [segment_rank, input].
  id4_ideogram4_lora_bake_parameter_range_t down_source;
  // Concatenated compact BF16 RHS tiles, padded per segment rank.
  id4_ideogram4_lora_bake_parameter_range_t down;
  // Reusable BF16 up-projection window [output_rows, maximum_segment_rank].
  id4_ideogram4_lora_bake_parameter_range_t up;
  // F32 effective base-plus-adapter weight window [output_rows, input].
  id4_ideogram4_lora_bake_parameter_range_t effective_weight;
  // Total bytes required by the packed working set.
  iree_device_size_t byte_length;
} id4_ideogram4_lora_bake_working_set_t;

// Planned bounded bake work for one compact FP8 linear weight.
typedef struct id4_ideogram4_lora_bake_target_t {
  // Canonical base parameter key patched by this target.
  iree_string_view_t base_parameter_key;
  // Compact FP8 weight range in the patchable parameter slab.
  id4_ideogram4_lora_bake_parameter_range_t weight_range;
  // F32 output-row scale range in the patchable parameter slab.
  id4_ideogram4_lora_bake_parameter_range_t scale_range;
  // Input feature count of the effective linear weight.
  uint32_t input_size;
  // Output feature count of the effective linear weight.
  uint32_t output_size;
  // Sum of adapter ranks applied to this target.
  uint32_t total_rank;
  // Largest individual adapter rank applied to this target.
  uint32_t maximum_segment_rank;
  // Number of output rows processed by each full bake window.
  uint32_t output_rows_per_window;
  // Number of bounded windows required to bake the full target.
  uint32_t window_count;
  // Packed device working-set layout reused across all target windows.
  id4_ideogram4_lora_bake_working_set_t working_set;
} id4_ideogram4_lora_bake_target_t;

// Options for deriving a compact-weight LoRA bake schedule.
typedef struct id4_ideogram4_lora_bake_plan_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Exact-base conditioned-DiT plan whose patchable slab will be replaced.
  const id4_pipeline_plan_t* base_plan;
  // Immutable ordered LoRA topology whose targets will be baked.
  id4_ideogram4_lora_topology_t* topology;
  // Maximum device working bytes available to one target bake window.
  iree_device_size_t working_set_byte_capacity;
} id4_ideogram4_lora_bake_plan_create_options_t;

// Derives a bounded bake schedule from a base conditioned-DiT plan.
//
// The base plan must use compact FP8 linear weights and contain exactly one
// `lora_patchable` parameter slab. It must not already contain dynamic LoRA
// execution, because baking and dynamic residual application are mutually
// exclusive execution policies.
iree_status_t id4_ideogram4_lora_bake_plan_create(
    const id4_ideogram4_lora_bake_plan_create_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_lora_bake_plan_t** out_plan);

// Retains |plan| for the caller.
void id4_ideogram4_lora_bake_plan_retain(id4_ideogram4_lora_bake_plan_t* plan);

// Releases |plan| from the caller.
void id4_ideogram4_lora_bake_plan_release(id4_ideogram4_lora_bake_plan_t* plan);

// Returns the retained exact-base conditioned-DiT plan.
const id4_pipeline_plan_t* id4_ideogram4_lora_bake_plan_base_plan(
    const id4_ideogram4_lora_bake_plan_t* plan);

// Returns the retained immutable LoRA topology.
id4_ideogram4_lora_topology_t* id4_ideogram4_lora_bake_plan_topology(
    const id4_ideogram4_lora_bake_plan_t* plan);

// Returns the slab index of the replaceable `lora_patchable` domain.
iree_host_size_t id4_ideogram4_lora_bake_plan_patchable_slab_index(
    const id4_ideogram4_lora_bake_plan_t* plan);

// Returns the byte length of the replaceable patchable slab.
iree_device_size_t id4_ideogram4_lora_bake_plan_patchable_slab_byte_length(
    const id4_ideogram4_lora_bake_plan_t* plan);

// Returns the maximum working-set byte length used by any target window.
iree_device_size_t id4_ideogram4_lora_bake_plan_working_set_high_water_mark(
    const id4_ideogram4_lora_bake_plan_t* plan);

// Returns the total source BF16 adapter bytes consumed across all targets.
iree_device_size_t id4_ideogram4_lora_bake_plan_adapter_byte_length(
    const id4_ideogram4_lora_bake_plan_t* plan);

// Returns the number of planned compact weight targets.
iree_host_size_t id4_ideogram4_lora_bake_plan_target_count(
    const id4_ideogram4_lora_bake_plan_t* plan);

// Returns planned target |index| or NULL when it is out of range.
const id4_ideogram4_lora_bake_target_t* id4_ideogram4_lora_bake_plan_target_at(
    const id4_ideogram4_lora_bake_plan_t* plan, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_IDEOGRAM4_LORA_BAKE_PLAN_H_
