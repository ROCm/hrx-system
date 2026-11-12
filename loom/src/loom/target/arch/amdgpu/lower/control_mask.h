// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native EXEC-mask algebra and masked value merging shared by AMDGPU control
// planners. These helpers operate only on already-selected Low carriers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_MASK_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_MASK_H_

#include "loom/codegen/low/lower/control_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_mask_merge_flag_bits_e {
  // Exact logical values replace incoming native lane masks on these arms.
  LOOM_AMDGPU_MASK_MERGE_FALSE_ZERO = 1u << 0,
  LOOM_AMDGPU_MASK_MERGE_FALSE_ONE = 1u << 1,
  LOOM_AMDGPU_MASK_MERGE_TRUE_ZERO = 1u << 2,
  LOOM_AMDGPU_MASK_MERGE_TRUE_ONE = 1u << 3,
  // SGPR words hold uniform payload bits instead of per-lane predicate bits.
  LOOM_AMDGPU_MASK_MERGE_UNIFORM = 1u << 4,
} loom_amdgpu_mask_merge_flag_bits_t;
typedef uint8_t loom_amdgpu_mask_merge_flags_t;

// Tests the physical pair shape, independently of its logical meaning.
bool loom_amdgpu_low_type_is_sgpr_pair(loom_low_lower_context_t* context,
                                       loom_type_t low_type);

// Uses the retained authored signature to distinguish logical lane masks from
// numeric pairs. Supplemental carriers have no authored merge semantics.
bool loom_amdgpu_control_argument_is_lane_mask(
    loom_low_lower_context_t* context, loom_low_lower_block_ref_t block,
    uint16_t argument_index);

iree_status_t loom_amdgpu_prepare_masked_merge(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_block_t* source_merge_dest,
    loom_low_lower_block_ref_t merge_low_dest, loom_value_slice_t false_args,
    loom_value_slice_t true_args,
    const loom_amdgpu_mask_merge_flags_t** out_flags);

iree_status_t loom_amdgpu_emit_zero_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value);

iree_status_t loom_amdgpu_emit_zero_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value);

iree_status_t loom_amdgpu_emit_native_i1_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_result);

// Uniform arguments share one SCC projection per merge tuple. Initialize
// |inout_uniform_condition| to INVALID before emitting the tuple; nonuniform
// payloads do not access it and may pass NULL.
iree_status_t loom_amdgpu_emit_masked_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t participating_lanes, loom_amdgpu_mask_merge_flags_t flags,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_value_id_t* inout_uniform_condition,
    loom_type_t value_type, loom_value_id_t* out_merged_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_MASK_H_
