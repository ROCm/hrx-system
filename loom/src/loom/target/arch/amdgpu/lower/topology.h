// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU launch topology lowering helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_TOPOLOGY_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_TOPOLOGY_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the exact wavefront size selected by the active target bundle.
uint32_t loom_amdgpu_target_wavefront_size(const loom_target_bundle_t* bundle);

// Returns the native execution partition width available to subgroup
// communication for |source_wavefront_size| on |target_facts|.
//
// Source target records may request a kernel wavefront size that is wider than
// the processor's default/native execution partition. Workgroup collectives can
// stitch native partitions together through LDS, but direct subgroup operations
// must not claim semantic communication wider than this value.
uint32_t loom_amdgpu_target_native_subgroup_width(
    const loom_amdgpu_target_facts_t* target_facts,
    uint32_t source_wavefront_size);

// Returns whether a direct subgroup operation with |required_width| lanes can
// be represented by native subgroup communication for the selected target.
bool loom_amdgpu_target_supports_direct_subgroup_width(
    const loom_amdgpu_target_facts_t* target_facts,
    uint32_t source_wavefront_size, uint32_t required_width);

// Selects the active target wavefront size when it is valid for native
// subgroup lowering.
bool loom_amdgpu_select_subgroup_wavefront_size(
    loom_low_lower_context_t* context, uint32_t* out_wavefront_size);

// Selects whether a direct subgroup operation with |required_width| semantic
// lanes can be represented by native subgroup communication for the selected
// target.
bool loom_amdgpu_select_direct_subgroup_width(loom_low_lower_context_t* context,
                                              uint32_t source_wavefront_size,
                                              uint32_t required_width);

// Selects the active target wavefront size when full-wave native subgroup
// communication is available.
bool loom_amdgpu_select_full_wave_direct_subgroup_width(
    loom_low_lower_context_t* context, uint32_t* out_wavefront_size);

// Returns the fixed per-dimension workgroup size required by the source
// function or target ABI.
bool loom_amdgpu_required_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, loom_target_workgroup_size_t* out_size);

// Returns the fixed per-dimension workgroup size required by the source
// function or target ABI, using |fact_table| to prove launch-config values
// that are not literal constants in the current IR.
bool loom_amdgpu_required_workgroup_size_from_facts(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    const loom_value_fact_table_t* fact_table,
    loom_target_workgroup_size_t* out_size);

// Returns the fixed flat workgroup size required by the source function or
// target ABI.
bool loom_amdgpu_required_flat_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t* out_flat_size);

// Returns the fixed flat workgroup size required by the source function or
// target ABI, using |fact_table| to prove launch-config values that are not
// literal constants in the current IR.
bool loom_amdgpu_required_flat_workgroup_size_from_facts(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    const loom_value_fact_table_t* fact_table, uint32_t* out_flat_size);

typedef struct loom_amdgpu_subgroup_wave_shape_t {
  // Exact flattened workgroup size selected by launch configuration.
  uint32_t flat_workgroup_size;
  // Maximum number of live lanes in any source subgroup wave.
  uint32_t active_lane_count;
  // Number of live lanes in the final source subgroup wave.
  uint32_t tail_lane_count;
  // Whether full waves precede a shorter final wave.
  bool has_partial_tail;
} loom_amdgpu_subgroup_wave_shape_t;

// Resolves the exact live-lane shape of source subgroup waves. A workgroup
// smaller than one wave has one statically short wave; |has_partial_tail| is
// reserved for workgroups whose waves have differing live-lane counts.
bool loom_amdgpu_resolve_subgroup_wave_shape(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t wavefront_size,
    loom_amdgpu_subgroup_wave_shape_t* out_shape);

// Emits the current invocation lane id within its subgroup as a VGPR value.
iree_status_t loom_amdgpu_emit_current_subgroup_lane_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_lane_id);

// Emits the flattened local workitem id for the active kernel launch.
iree_status_t loom_amdgpu_emit_current_workitem_linear_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_linear_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_TOPOLOGY_H_
