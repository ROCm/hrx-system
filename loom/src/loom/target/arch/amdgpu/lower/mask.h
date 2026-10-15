// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private AMDGPU predicate and payload selection contracts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MASK_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MASK_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/ir.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_amdgpu_cndmask_b32_descriptor_flags_t;

enum {
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER = 1u << 0,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_INLINE = 1u << 1,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_INLINE = 1u << 2,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL = 1u << 3,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL = 1u << 4,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL_SRC1_INLINE = 1u << 5,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL_SRC0_INLINE = 1u << 6,
  LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_ALL =
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_REGISTER |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_INLINE |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_INLINE |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC0_LITERAL_SRC1_INLINE |
      LOOM_AMDGPU_CNDMASK_B32_DESCRIPTOR_SRC1_LITERAL_SRC0_INLINE,
};

typedef struct loom_amdgpu_cndmask_b32_descriptors_t {
  // Descriptor row selected for register-register lane selects.
  loom_low_lower_resolved_descriptor_t register_descriptor;
  // Optional descriptor row selected when the false lane is an inline source.
  loom_low_lower_resolved_descriptor_t src0_inline_descriptor;
  // Optional descriptor row selected when the true lane is an inline source.
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor;
  // Optional descriptor row selected when the false lane is a literal source.
  loom_low_lower_resolved_descriptor_t src0_literal_descriptor;
  // Optional descriptor row selected when the true lane is a literal source.
  loom_low_lower_resolved_descriptor_t src1_literal_descriptor;
  // Optional descriptor row selected when false is literal and true is inline.
  loom_low_lower_resolved_descriptor_t src0_literal_src1_inline_descriptor;
  // Optional descriptor row selected when true is literal and false is inline.
  loom_low_lower_resolved_descriptor_t src1_literal_src0_inline_descriptor;
} loom_amdgpu_cndmask_b32_descriptors_t;

typedef enum loom_amdgpu_select_condition_kind_e {
  LOOM_AMDGPU_SELECT_CONDITION_KIND_NONE = 0,
  LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC = 1,
  LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK = 2,
  LOOM_AMDGPU_SELECT_CONDITION_KIND_VECTOR_MASK = 3,
  LOOM_AMDGPU_SELECT_CONDITION_KIND_SGPR_BOOL = 4,
} loom_amdgpu_select_condition_kind_t;

typedef enum loom_amdgpu_select_payload_kind_e {
  LOOM_AMDGPU_SELECT_PAYLOAD_KIND_NONE = 0,
  LOOM_AMDGPU_SELECT_PAYLOAD_KIND_DATA = 1,
  LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_MASK = 2,
  LOOM_AMDGPU_SELECT_PAYLOAD_KIND_PACKED_DATA = 3,
  LOOM_AMDGPU_SELECT_PAYLOAD_KIND_I1_BOOL = 4,
} loom_amdgpu_select_payload_kind_t;

// Boolean operands either reuse their selected carrier or materialize the
// active mask at this use. EXEC is read here, not captured at the definition.
typedef enum loom_amdgpu_mask_operand_kind_e {
  LOOM_AMDGPU_MASK_OPERAND_VALUE = 0,
  LOOM_AMDGPU_MASK_OPERAND_ZERO = 1,
  LOOM_AMDGPU_MASK_OPERAND_EXEC = 2,
} loom_amdgpu_mask_operand_kind_t;

// Boolean identities selected once from source facts and operand identities.
typedef enum loom_amdgpu_mask_select_recipe_e {
  LOOM_AMDGPU_MASK_SELECT_TRUE_VALUE = 0,
  LOOM_AMDGPU_MASK_SELECT_UNIFORM = 1,
  LOOM_AMDGPU_MASK_SELECT_CONDITION = 2,
  LOOM_AMDGPU_MASK_SELECT_INVERSE_CONDITION = 3,
  LOOM_AMDGPU_MASK_SELECT_CONDITION_OR_FALSE = 4,
  LOOM_AMDGPU_MASK_SELECT_INVERSE_AND_FALSE = 5,
  LOOM_AMDGPU_MASK_SELECT_TRUE_OR_INVERSE = 6,
  LOOM_AMDGPU_MASK_SELECT_CONDITION_AND_TRUE = 7,
  LOOM_AMDGPU_MASK_SELECT_MERGE = 8,
} loom_amdgpu_mask_select_recipe_t;

// Sparse ordered lane recipes retained only for immediates or equal inputs.
typedef struct loom_amdgpu_select_lane_plan_t loom_amdgpu_select_lane_plan_t;

// Retained state for independently selected packed payload elements.
typedef struct loom_amdgpu_packed_select_payload_t {
  // Bitfield insertion with a literal mask when the target supports it.
  loom_low_lower_resolved_descriptor_t merge_descriptor;
  // Materializes an SGPR mask for a register-form merge; empty for literals.
  loom_low_lower_resolved_descriptor_t mask_constant_descriptor;
  // Interned immediate name used by mask constants.
  loom_string_id_t imm32_attr_name_id;
  // Number of logical payload elements, excluding physical tail padding.
  uint32_t element_count;
  // Number of bits selected by each independent predicate.
  uint32_t element_bit_count;
} loom_amdgpu_packed_select_payload_t;

typedef struct loom_amdgpu_vector_select_plan_t {
  // Source condition selecting true lanes.
  loom_value_id_t condition;
  // Source vector used when the corresponding condition lane is true.
  loom_value_id_t true_value;
  // Source vector used when the corresponding condition lane is false.
  loom_value_id_t false_value;
  // Selected loom_amdgpu_select_payload_kind_t of the payload.
  uint8_t payload_kind;
  // Selected loom_amdgpu_select_condition_kind_t of the condition.
  uint8_t condition_kind;
  // Descriptor row selected for SCC-controlled scalar selects.
  loom_low_lower_resolved_descriptor_t scc_descriptor;
  // Descriptor row rematerializing SCC from an SGPR boolean condition.
  loom_low_lower_resolved_descriptor_t sgpr_bool_compare_descriptor;
  // Descriptor rows selected for scalar-mask v_cndmask_b32 lane selects.
  loom_amdgpu_cndmask_b32_descriptors_t cndmask_descriptors;
  // Additional emission state selected by payload_kind.
  union {
    // Register data uses ordinary selects except for these selected lanes.
    struct {
      // Function-plan-owned recipes, sorted by physical register lane.
      const loom_amdgpu_select_lane_plan_t* lanes;
      // Number of retained lane recipes; zero requires no allocation.
      uint8_t lane_count;
    } data;
    // Boolean payloads combine native per-workitem masks.
    struct {
      // Descriptor row selected to read EXEC for i1 mask selection.
      loom_low_lower_resolved_descriptor_t exec_read_descriptor;
      // Descriptor row selected to AND i1 mask payloads.
      loom_low_lower_resolved_descriptor_t and_descriptor;
      // Descriptor row selected to OR i1 mask payloads.
      loom_low_lower_resolved_descriptor_t or_descriptor;
      // Descriptor row selected to XOR i1 mask payloads.
      loom_low_lower_resolved_descriptor_t xor_descriptor;
      // Selected loom_amdgpu_mask_select_recipe_t.
      uint8_t recipe;
      // Selected loom_amdgpu_mask_operand_kind_t for each source operand.
      struct {
        // Condition carrier or use-local constant mask.
        uint8_t condition;
        // True payload carrier or use-local constant mask.
        uint8_t true_value;
        // False payload carrier or use-local constant mask.
        uint8_t false_value;
      } operands;
    } mask;
    // Independent element choices merged into their packed payload words.
    loom_amdgpu_packed_select_payload_t packed;
  } payload;
  // Result vector value.
  loom_value_id_t result;
  // Static number of selected 32-bit register units.
  uint32_t lane_count;
  // Number of selected register units controlled by one vector mask lane.
  uint32_t registers_per_condition_lane;
} loom_amdgpu_vector_select_plan_t;

static_assert(sizeof(loom_amdgpu_vector_select_plan_t) == 144,
              "select recipes must fit the existing plan layout");

// Selects each hardware work-item's bit between two already-materialized
// SGPR-pair masks. All operands and the result have |mask_type|; |condition|
// carries one selector bit per hardware work-item, not a uniform Boolean.
iree_status_t loom_amdgpu_emit_i1_mask_select(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_value_id_t false_value,
                                              loom_value_id_t true_value,
                                              loom_value_id_t condition,
                                              loom_type_t mask_type,
                                              loom_value_id_t* out_value);

// Selects the AMDGPU vector.select plan using explicit SGPR-pair masks and b32
// cndmask packets.
iree_status_t loom_amdgpu_select_vector_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected);

// Verifies AMDGPU low legality for vector.select packed payload forms.
iree_status_t loom_amdgpu_low_legality_verify_vector_select(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects an AMDGPU scf.select plan using either SCC-controlled scalar selects
// or explicit SGPR-pair masks and b32 cndmask packets.
iree_status_t loom_amdgpu_select_scf_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.select or scf.select op from its selected AMDGPU plan.
iree_status_t loom_amdgpu_lower_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MASK_H_
