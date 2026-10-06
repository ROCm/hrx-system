// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for register-structural vector source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_STRUCTURAL_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_STRUCTURAL_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_amdgpu_vector_shuffle_kind_t;
enum loom_amdgpu_vector_shuffle_kind_e {
  LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_REGISTER_UNITS = 1,
  LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_PACKED_BYTES = 2,
};

// Immutable storage-shaped permutation selected for one vector.shuffle.
typedef struct loom_amdgpu_vector_shuffle_plan_t {
  // Source vector value supplying every selected logical lane.
  loom_value_id_t source;
  // Result vector value receiving the shuffled payload.
  loom_value_id_t result;
  // Best available V_PERM_B32 form for packed sub-dword payloads.
  loom_amdgpu_byte_permute_plan_t byte_permute;
  // Selected lowering strategy for the source and lane map.
  loom_amdgpu_vector_shuffle_kind_t kind;
  // Number of logical elements shared by the source and result.
  uint8_t element_count;
  // Number of 32-bit register units occupied by one logical element.
  uint8_t element_register_count;
  // Number of payload bits occupied by one logical element.
  uint8_t element_bit_count;
  // Source logical element selected for each result logical element.
  uint8_t source_element_indices[LOOM_AMDGPU_MAX_VECTOR_STORAGE_ELEMENTS];
} loom_amdgpu_vector_shuffle_plan_t;
static_assert(sizeof(loom_amdgpu_vector_shuffle_plan_t) == 80,
              "AMDGPU vector shuffle plans must stay cache dense");
static_assert(LOOM_AMDGPU_MAX_VECTOR_STORAGE_ELEMENTS <= UINT8_MAX,
              "vector element counts must fit compact shuffle plans");

// Selects an AMDGPU vector.bitcast register reinterpretation plan.
iree_status_t loom_amdgpu_select_vector_bitcast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_bitcast_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitcast as a register reinterpretation, materializing
// an SGPR payload in VGPRs when the result's consumers require vector storage.
iree_status_t loom_amdgpu_lower_vector_bitcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bitcast_plan_t* plan);

// Selects an AMDGPU vector.concat register concatenation plan.
iree_status_t loom_amdgpu_select_vector_concat_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_concat_plan_t* out_plan, bool* out_selected);

// Lowers vector.concat by composing its existing Low register tuples.
iree_status_t loom_amdgpu_lower_vector_concat(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_concat_plan_t* plan);

// Lowers a source vector structural op with a static 32-bit register map.
iree_status_t loom_amdgpu_lower_vector_register_map(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_register_map_plan_t* plan);

// Selects an AMDGPU vector.deinterleave register split plan.
iree_status_t loom_amdgpu_select_vector_deinterleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_deinterleave_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.deinterleave op as AMDGPU register splitting.
iree_status_t loom_amdgpu_lower_vector_deinterleave(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_deinterleave_plan_t* plan);

// Selects an AMDGPU vector.interleave register merge plan.
iree_status_t loom_amdgpu_select_vector_interleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_interleave_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.interleave op as AMDGPU register merging.
iree_status_t loom_amdgpu_lower_vector_interleave(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_interleave_plan_t* plan);

// Selects an AMDGPU vector.shuffle register permutation plan.
iree_status_t loom_amdgpu_select_vector_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_shuffle_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.shuffle with its selected register permutation.
iree_status_t loom_amdgpu_lower_vector_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_shuffle_plan_t* plan);

// Returns true when vector.shuffle has a native plan for the target descriptor
// set and therefore needs no source legalization.
bool loom_amdgpu_vector_shuffle_can_lower(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op);

// Selects an AMDGPU vector.transpose flattened register permutation plan.
iree_status_t loom_amdgpu_select_vector_transpose_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_register_map_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU vector.slice register slicing plan.
iree_status_t loom_amdgpu_select_vector_slice_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_slice_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.slice op as AMDGPU register slicing.
iree_status_t loom_amdgpu_lower_vector_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* plan);

// Verifies source vector structural op legality for AMDGPU target-low
// selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_structural(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_STRUCTURAL_H_
