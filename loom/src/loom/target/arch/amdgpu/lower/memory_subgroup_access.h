// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU active-subgroup proofs and fragment memory address geometry.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_SUBGROUP_ACCESS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_SUBGROUP_ACCESS_H_

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_memory_lane_source_e {
  // Lane identity is the X workitem coordinate and must not wrap in a wave.
  LOOM_AMDGPU_MEMORY_LANE_SOURCE_WORKITEM_X = 0,
  // Lane identity is the target subgroup lane ID.
  LOOM_AMDGPU_MEMORY_LANE_SOURCE_SUBGROUP_LANE = 1,
} loom_amdgpu_memory_lane_source_t;

// Proof that one source operation executes with a complete active subgroup.
typedef struct loom_amdgpu_memory_full_subgroup_proof_t {
  // Whether every lane in the target subgroup is proven active.
  bool is_full_subgroup;
  // Stable proof key when |is_full_subgroup| is true.
  iree_string_view_t proof;
  // Stable reason key when |is_full_subgroup| is false.
  iree_string_view_t unknown_reason;
} loom_amdgpu_memory_full_subgroup_proof_t;

// Proves that |source_op| executes with every target subgroup lane active.
//
// Status reports only analysis allocation failures. An unproven active set is
// returned as ordinary structured evidence in |out_proof|.
iree_status_t loom_amdgpu_memory_prove_full_subgroup(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint8_t subgroup_size, loom_amdgpu_memory_lane_source_t lane_source,
    loom_amdgpu_memory_full_subgroup_proof_t* out_proof);

// Populates exact or explicitly unknown fragment subgroup address geometry.
//
// The compiled fragment plan is the sole source of lane-address terms. The
// selected descriptor effect supplies the per-lane packet width. Analysis runs
// only from the existing detail-report path.
iree_status_t loom_amdgpu_fragment_memory_report_subgroup_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index,
    const loom_low_descriptor_memory_effect_summary_t* issued,
    loom_low_lower_memory_subgroup_access_report_t* out_report);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_SUBGROUP_ACCESS_H_
