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
  LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_REGISTER_MAP = 1,
  LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_PACKED_BYTE_PERMUTE = 2,
};

// Immutable register permutation selected for one vector.shuffle operation.
typedef struct loom_amdgpu_vector_shuffle_plan_t {
  // Source vector value supplying every selected logical lane.
  loom_value_id_t source;
  // Result vector value receiving the shuffled payload.
  loom_value_id_t result;
  // Selected lowering strategy for the source and lane map.
  loom_amdgpu_vector_shuffle_kind_t kind;
  // Static 32-bit backing register count shared by source and result.
  uint8_t register_count;
  // Strategy-specific plan data selected by |kind|.
  union {
    // Whole-register source selection for register-map mode.
    struct {
      // Source register selected for each result register.
      uint8_t source_register_indices[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    } register_map;
    // Packed-byte permutation data for packed-byte-permute mode.
    struct {
      // Best available V_PERM_B32 selector representation.
      loom_amdgpu_byte_permute_plan_t packet;
      // Source registers selected for the two V_PERM_B32 inputs.
      uint8_t source_register_indices[2]
                                     [LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
      // V_PERM_B32 byte selector literal for each result register.
      uint32_t selectors[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
    } packed_bytes;
  } strategy;
} loom_amdgpu_vector_shuffle_plan_t;
static_assert(sizeof(loom_amdgpu_vector_shuffle_plan_t) == 112,
              "AMDGPU vector shuffle plans must stay cache dense");
static_assert(LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS <= UINT8_MAX,
              "packed vector register counts must fit compact plans");

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
