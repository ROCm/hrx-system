// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_repack.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/float16.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_repack_packed_b16.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_fragment_repack_shape_value_matches(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source_value,
    loom_value_id_t result_value) {
  if (source_value == result_value) {
    return true;
  }
  int64_t source_exact = 0;
  int64_t result_exact = 0;
  return loom_amdgpu_value_facts_as_exact_non_negative_i64(
             loom_value_fact_table_lookup(fact_table, source_value),
             &source_exact) &&
         loom_amdgpu_value_facts_as_exact_non_negative_i64(
             loom_value_fact_table_lookup(fact_table, result_value),
             &result_exact) &&
         source_exact == result_exact;
}

static bool loom_amdgpu_fragment_repack_shape_matches(
    const loom_value_fact_table_t* fact_table,
    loom_vector_fragment_fact_t source_fact, loom_value_id_t blocks,
    loom_value_id_t rows, loom_value_id_t columns) {
  const uint8_t expected_rank = blocks == LOOM_VALUE_ID_INVALID ? 2 : 3;
  return source_fact.shape_rank == expected_rank &&
         (blocks == LOOM_VALUE_ID_INVALID ||
          loom_amdgpu_fragment_repack_shape_value_matches(
              fact_table, loom_vector_fragment_fact_block_value(source_fact),
              blocks)) &&
         loom_amdgpu_fragment_repack_shape_value_matches(
             fact_table, loom_vector_fragment_fact_row_value(source_fact),
             rows) &&
         loom_amdgpu_fragment_repack_shape_value_matches(
             fact_table, loom_vector_fragment_fact_column_value(source_fact),
             columns);
}

static iree_string_view_t loom_amdgpu_fragment_repack_role_flags_key(
    loom_vector_fragment_role_flags_t role_flags) {
  switch (role_flags) {
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS:
      return IREE_SV("lhs");
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS:
      return IREE_SV("rhs");
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT:
      return IREE_SV("init");
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT:
      return IREE_SV("result");
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
        LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT:
      return IREE_SV("accumulator_result");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_amdgpu_fragment_repack_reason_key(
    loom_amdgpu_fragment_repack_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SOURCE_FACTS:
      return IREE_SV("source_fragment_facts");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SHAPE:
      return IREE_SV("fragment_shape");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TRANSITION:
      return IREE_SV("role_transition");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TYPE_TRANSITION:
      return IREE_SV("type_transition");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TYPE_TRANSITION:
      return IREE_SV("role_type_transition");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_LAYOUT:
      return IREE_SV("target_layout");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_LAYOUT_STRATEGY:
      return IREE_SV("layout_strategy");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_PACKETS:
      return IREE_SV("target_packets");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_amdgpu_fragment_repack_plan_key(
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  switch (plan->strategy) {
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_ALIAS:
      return IREE_SV("amdgpu.fragment_repack.strategy.alias");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_PACKED_BPERMUTE:
      return IREE_SV(
          "amdgpu.fragment_repack.strategy.result_to_lhs_bf16_packed_bpermute");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_BPERMUTE:
      return IREE_SV(
          "amdgpu.fragment_repack.strategy.result_to_lhs_bf16_bpermute");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_TRANSPOSE_BPERMUTE:
      return IREE_SV(
          "amdgpu.fragment_repack.strategy."
          "result_to_lhs_bf16_transpose_bpermute");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_RHS_PACKED_B16_XOR_PERMUTE:
      return IREE_SV(
          "amdgpu.fragment_repack.strategy."
          "result_to_rhs_packed_b16_xor_permute");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC:
      return IREE_SV("amdgpu.fragment_repack.strategy.diagnostic");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_NONE:
    default:
      return iree_string_view_empty();
  }
}

static loom_amdgpu_fragment_repack_reason_t
loom_amdgpu_fragment_repack_transition_reason(
    loom_vector_fragment_role_flags_t source_role_flags,
    loom_vector_fragment_role_flags_t result_role_flags,
    loom_type_t source_type, loom_type_t result_type) {
  const bool role_matches = source_role_flags == result_role_flags;
  const bool type_matches = loom_type_equal(source_type, result_type);
  if (!role_matches && !type_matches) {
    return LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TYPE_TRANSITION;
  }
  if (!role_matches) {
    return LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TRANSITION;
  }
  if (!type_matches) {
    return LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TYPE_TRANSITION;
  }
  return LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE;
}

static bool loom_amdgpu_fragment_repack_contract_role(
    loom_vector_fragment_role_flags_t role_flags,
    loom_contract_operand_role_t* out_role) {
  *out_role = LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
  switch (role_flags) {
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
      return true;
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
      return true;
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
      return true;
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT:
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
        LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_RESULT;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_repack_uses_source_register_bit_tree(
    uint16_t source_register_count) {
  return source_register_count >= 4 &&
         loom_amdgpu_u32_is_power_of_two(source_register_count);
}

static bool loom_amdgpu_fragment_repack_has_static_zero_source_byte_base(
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  return plan->source_lane_group.and_mask == 0;
}

static bool loom_amdgpu_fragment_repack_can_emit_lane_recipe(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_fragment_repack_lane_recipe_t recipe) {
  if (recipe.and_mask == 0) {
    return true;
  }
  return (recipe.and_mask == UINT16_MAX ||
          loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
              recipe.and_mask)) &&
         (recipe.right_shift == 0 ||
          loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
              recipe.right_shift));
}

static bool loom_amdgpu_fragment_repack_can_emit_packed_pair(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  switch (plan->packed_pair.kind) {
    case LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_NONE:
    case LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_EXCHANGE_THEN_PACK:
      return loom_amdgpu_bf16_descriptor_set_can_emit_f32_pair_to_packed_bf16(
          descriptor_set);
    case LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_U16:
      return loom_amdgpu_descriptor_set_has_ref(
                 descriptor_set, plan->packed_pair.descriptor_ref) &&
             loom_amdgpu_bf16_descriptor_set_can_emit_f32_to_bf16_lane(
                 descriptor_set);
    case LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_BF16:
      return loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, plan->packed_pair.descriptor_ref);
    default:
      IREE_ASSERT_UNREACHABLE();
      return false;
  }
}

static bool
loom_amdgpu_fragment_repack_has_result_to_lhs_bf16_bpermute_descriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  static const loom_amdgpu_descriptor_ref_t kRequiredDescriptorRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
  };
  if (!loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, kRequiredDescriptorRefs,
          IREE_ARRAYSIZE(kRequiredDescriptorRefs)) ||
      !loom_amdgpu_fragment_repack_can_emit_packed_pair(descriptor_set, plan)) {
    return false;
  }
  if ((plan->source_register_count > 1 &&
       !loom_amdgpu_fragment_repack_can_emit_lane_recipe(
           descriptor_set, plan->source_register_selector)) ||
      !loom_amdgpu_fragment_repack_can_emit_lane_recipe(
          descriptor_set, plan->source_lane_group)) {
    return false;
  }
  if (plan->source_lane_group.and_mask != 0 &&
      !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          plan->source_lane_group_byte_shift)) {
    return false;
  }
  if (loom_amdgpu_fragment_repack_has_static_zero_source_byte_base(plan) &&
      plan->result_lane_div_byte_shift == 0 &&
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32)) {
    return false;
  }
  if (plan->result_lane_div_byte_shift != 0 &&
      !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          plan->result_lane_div_byte_shift)) {
    return false;
  }
  if (plan->result_lane_div_byte_shift != 0 &&
      !loom_amdgpu_fragment_repack_has_static_zero_source_byte_base(plan) &&
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32)) {
    return false;
  }
  if (loom_amdgpu_fragment_repack_uses_source_register_bit_tree(
          plan->source_register_count)) {
    if (!loom_amdgpu_descriptor_set_can_emit_vgpr_compare_immediate(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
            kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
                [LOOM_VECTOR_CMPI_PREDICATE_NE]
                    .src1_inline_descriptor_ref,
            0)) {
      return false;
    }
    const uint16_t bit_count = (uint16_t)iree_math_count_trailing_zeros_u32(
        plan->source_register_count);
    for (uint16_t i = 0; i < bit_count; ++i) {
      if (!loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
              UINT32_C(1) << i)) {
        return false;
      }
    }
    return true;
  }
  for (uint16_t i = 1; i < plan->source_register_count; ++i) {
    if (!loom_amdgpu_descriptor_set_can_emit_vgpr_compare_immediate(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
            kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
                [LOOM_VECTOR_CMPI_PREDICATE_EQ]
                    .src1_inline_descriptor_ref,
            i)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_repack_can_pack_adjacent_source_lanes(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_fragment_repack_plan_t* plan) {
  const uint16_t required_even_lane_byte_base_shift = 3;
  if (!loom_matrix_fragment_role_has_contiguous_lane_xor1_columns(
          layout, plan->source_role) ||
      plan->source_lane_group_byte_shift < required_even_lane_byte_base_shift ||
      (plan->result_lane_div_byte_shift != 0 &&
       plan->result_lane_div_byte_shift < required_even_lane_byte_base_shift)) {
    return false;
  }
  plan->packed_pair = (loom_amdgpu_fragment_repack_packed_pair_recipe_t){
      .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32_DPP16)) {
    plan->packed_pair = (loom_amdgpu_fragment_repack_packed_pair_recipe_t){
        .kind = LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_BF16,
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32_DPP16,
        .immediate = LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1,
        .crosslane_kind = LOOM_AMDGPU_CROSSLANE_DPP,
    };
    return true;
  }
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32_DPP16) &&
      loom_amdgpu_bf16_descriptor_set_can_emit_f32_to_bf16_lane(
          descriptor_set)) {
    plan->packed_pair = (loom_amdgpu_fragment_repack_packed_pair_recipe_t){
        .kind = LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_U16,
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32_DPP16,
        .immediate = LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1,
        .crosslane_kind = LOOM_AMDGPU_CROSSLANE_DPP,
    };
    return true;
  }

  loom_amdgpu_direct_xor_lane_recipe_t exchange = {0};
  if (!loom_amdgpu_select_direct_xor_lane_recipe(
          descriptor_set, LOOM_AMDGPU_DPP_ROW_LANE_COUNT, 1, &exchange)) {
    return false;
  }
  plan->packed_pair = (loom_amdgpu_fragment_repack_packed_pair_recipe_t){
      .kind = LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_EXCHANGE_THEN_PACK,
      .descriptor_ref = exchange.descriptor_ref,
      .immediate = exchange.immediate,
      .crosslane_kind = exchange.crosslane_kind,
  };
  return true;
}

typedef struct loom_amdgpu_fragment_repack_transpose_lane_bit_t {
  // Predicate selecting wave32 lanes whose lane-id bit is set.
  uint32_t wave32_mask;
  // DPP destination banks containing lanes whose lane-id bit is set.
  uint8_t dpp_bank_mask;
} loom_amdgpu_fragment_repack_transpose_lane_bit_t;

static const loom_amdgpu_fragment_repack_transpose_lane_bit_t
    kLoomAmdgpuFragmentRepackTransposeLaneBits[] = {
        [2] = {.wave32_mask = UINT32_C(0xCCCCCCCC)},
        [4] = {.wave32_mask = UINT32_C(0xF0F0F0F0), .dpp_bank_mask = 0xAu},
        [8] = {.wave32_mask = UINT32_C(0xFF00FF00), .dpp_bank_mask = 0xCu},
};

static bool loom_amdgpu_fragment_repack_select_transpose_strategy(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t wave_size,
    loom_amdgpu_fragment_repack_plan_t* plan) {
  const uint16_t transpose_bit_count = iree_min(
      (uint16_t)iree_math_count_trailing_zeros_u32(plan->source_register_count),
      (uint16_t)iree_math_count_trailing_zeros_u32(
          plan->result_register_count));
  if (transpose_bit_count == 0 ||
      transpose_bit_count >
          LOOM_AMDGPU_FRAGMENT_REPACK_TRANSPOSE_STAGE_CAPACITY) {
    return false;
  }

  if (!loom_amdgpu_fragment_repack_can_emit_lane_recipe(
          descriptor_set, plan->source_register_selector)) {
    return false;
  }

  loom_amdgpu_fragment_repack_transpose_stage_t
      transpose_stages[LOOM_AMDGPU_FRAGMENT_REPACK_TRANSPOSE_STAGE_CAPACITY] = {
          0};
  bool all_stages_have_conditional_exchange = true;
  for (uint16_t bit_index = 0; bit_index < transpose_bit_count; ++bit_index) {
    const uint32_t lane_xor = UINT32_C(2) << bit_index;
    if (!loom_amdgpu_select_direct_xor_lane_recipe(
            descriptor_set, LOOM_AMDGPU_DPP_ROW_LANE_COUNT, lane_xor,
            &transpose_stages[bit_index].exchange)) {
      return false;
    }
    all_stages_have_conditional_exchange &=
        transpose_stages[bit_index].exchange.conditional_ref !=
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  }

  const bool use_wave32_constant_predicates =
      wave_size == 32 && all_stages_have_conditional_exchange &&
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_VCC_IMM);
  const bool use_vcc_compare_predicates =
      !use_wave32_constant_predicates && all_stages_have_conditional_exchange &&
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC0_INLINE_VCC) &&
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32_SRC0_INLINE_VCC);
  const bool use_fused_conditional_exchange =
      use_wave32_constant_predicates || use_vcc_compare_predicates;

  for (uint16_t bit_index = 0; bit_index < transpose_bit_count; ++bit_index) {
    const uint32_t lane_xor = UINT32_C(2) << bit_index;
    if (lane_xor >=
        IREE_ARRAYSIZE(kLoomAmdgpuFragmentRepackTransposeLaneBits)) {
      return false;
    }
    const loom_amdgpu_fragment_repack_transpose_lane_bit_t* lane_bit =
        &kLoomAmdgpuFragmentRepackTransposeLaneBits[lane_xor];
    if (use_wave32_constant_predicates) {
      transpose_stages[bit_index].lane_bit_set_mask = lane_bit->wave32_mask;
    } else if (!loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
                   descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                   lane_xor)) {
      return false;
    }
    if (use_fused_conditional_exchange && lane_bit->dpp_bank_mask != 0 &&
        transpose_stages[bit_index].exchange.masked_move_ref !=
            LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      transpose_stages[bit_index].lane_bit_set_bank_mask =
          lane_bit->dpp_bank_mask;
    }
  }
  if (!use_fused_conditional_exchange &&
      !loom_amdgpu_descriptor_set_can_emit_vgpr_compare_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
          kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
              [LOOM_VECTOR_CMPI_PREDICATE_NE]
                  .src1_inline_descriptor_ref,
          0)) {
    return false;
  }

  const uint16_t candidate_count =
      (uint16_t)(plan->source_register_count >> transpose_bit_count);
  loom_amdgpu_fragment_repack_plan_t candidate_plan = *plan;
  candidate_plan.source_register_count = candidate_count;
  candidate_plan.source_register_selector.right_shift =
      (uint16_t)(candidate_plan.source_register_selector.right_shift +
                 transpose_bit_count);
  if (!loom_amdgpu_fragment_repack_has_result_to_lhs_bf16_bpermute_descriptors(
          descriptor_set, &candidate_plan)) {
    return false;
  }

  const uint32_t transpose_mask = (UINT32_C(1) << transpose_bit_count) - 1u;
  if ((candidate_count > 1 &&
       !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
           descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
           transpose_mask)) ||
      !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 3) ||
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32)) {
    return false;
  }

  plan->strategy =
      LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_TRANSPOSE_BPERMUTE;
  plan->transpose_bit_count = transpose_bit_count;
  plan->transposed_source_register_candidate_count = candidate_count;
  memcpy(plan->strategy_payload.transpose_stages, transpose_stages,
         sizeof(plan->strategy_payload.transpose_stages));
  if (use_wave32_constant_predicates) {
    plan->transpose_predicate.constant =
        LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_VCC_IMM;
  } else if (use_vcc_compare_predicates) {
    plan->transpose_predicate.equal_zero =
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC0_INLINE_VCC;
    plan->transpose_predicate.not_equal_zero =
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32_SRC0_INLINE_VCC;
  }
  return true;
}

static bool
loom_amdgpu_fragment_repack_select_result_to_lhs_bf16_bpermute_strategy(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* source_role_layout,
    const loom_matrix_fragment_role_layout_t* result_role_layout,
    loom_amdgpu_fragment_repack_plan_t* plan) {
  if (plan->source_role != LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
      plan->result_role != LOOM_CONTRACT_OPERAND_ROLE_LHS ||
      loom_type_element_type(plan->source_type) != LOOM_SCALAR_TYPE_F32 ||
      loom_type_element_type(plan->result_type) != LOOM_SCALAR_TYPE_BF16 ||
      source_role_layout->element_bit_count != 32 ||
      loom_amdgpu_matrix_fragment_payload_elements_per_register(
          result_role_layout) !=
          LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT ||
      result_role_layout->element_bit_count !=
          LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT ||
      source_role_layout->coordinate_flags !=
          (LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
           LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN) ||
      result_role_layout->coordinate_flags !=
          (LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
           LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION) ||
      source_role_layout->register_count == 0 ||
      result_role_layout->register_count == 0) {
    return false;
  }

  const loom_matrix_fragment_axis_layout_t* source_row_layout =
      &source_role_layout->axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW];
  const loom_matrix_fragment_axis_layout_t* source_column_layout =
      &source_role_layout->axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN];
  const loom_matrix_fragment_axis_layout_t* result_row_layout =
      &result_role_layout->axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW];
  const loom_matrix_fragment_axis_layout_t* result_reduction_layout =
      &result_role_layout->axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION];
  const bool result_uses_lane_div_reduction =
      result_reduction_layout->thread_count > 1;

  if (!loom_amdgpu_u32_is_power_of_two(layout->tile_shape.result_row_count) ||
      !loom_amdgpu_u32_is_power_of_two(
          layout->tile_shape.result_column_count) ||
      !loom_amdgpu_u32_is_power_of_two(layout->tile_shape.reduction_count)) {
    return false;
  }

  const uint16_t lane_group_count =
      (uint16_t)(layout->wave_size / layout->tile_shape.result_column_count);
  if (lane_group_count == 0 ||
      (layout->wave_size % layout->tile_shape.result_column_count) != 0 ||
      !loom_amdgpu_u32_is_power_of_two(lane_group_count) ||
      (layout->tile_shape.result_row_count % lane_group_count) != 0 ||
      source_role_layout->register_count !=
          layout->tile_shape.result_row_count / lane_group_count ||
      source_role_layout->register_count >
          LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES ||
      (layout->tile_shape.reduction_count %
       LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT) != 0 ||
      result_role_layout->register_count >
          LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES ||
      source_row_layout->thread_count != lane_group_count ||
      source_row_layout->thread_stride !=
          layout->tile_shape.result_column_count ||
      source_column_layout->thread_count !=
          layout->tile_shape.result_column_count ||
      source_column_layout->thread_stride != 1 ||
      result_row_layout->thread_count != layout->tile_shape.result_row_count ||
      result_row_layout->thread_stride != 1) {
    return false;
  }
  const uint16_t lane_div_count =
      (uint16_t)(layout->wave_size / layout->tile_shape.result_row_count);
  if (lane_div_count == 0 ||
      (layout->wave_size % layout->tile_shape.result_row_count) != 0 ||
      !loom_amdgpu_u32_is_power_of_two(lane_div_count) ||
      (result_uses_lane_div_reduction &&
       (result_reduction_layout->thread_count != lane_div_count ||
        result_reduction_layout->thread_stride !=
            layout->tile_shape.result_row_count)) ||
      (!result_uses_lane_div_reduction &&
       result_reduction_layout->thread_count != 1)) {
    return false;
  }
  const uint16_t expected_result_register_count =
      (uint16_t)(layout->tile_shape.reduction_count /
                 (LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT *
                  (result_uses_lane_div_reduction ? lane_div_count : 1u)));
  if (expected_result_register_count == 0 ||
      result_role_layout->register_count != expected_result_register_count) {
    return false;
  }

  const uint32_t source_lane_group_byte_count =
      (uint32_t)layout->tile_shape.result_column_count *
      LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT;
  if (!loom_amdgpu_u32_is_power_of_two(source_lane_group_byte_count)) {
    return false;
  }
  const uint32_t result_lane_div_byte_count =
      result_uses_lane_div_reduction
          ? (uint32_t)result_role_layout->register_count *
                LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT *
                LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT
          : 0u;
  if (result_lane_div_byte_count != 0 &&
      !loom_amdgpu_u32_is_power_of_two(result_lane_div_byte_count)) {
    return false;
  }

  plan->strategy =
      LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_BPERMUTE;
  plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE;
  plan->layout_kind = (loom_amdgpu_matrix_fragment_layout_kind_t)layout->kind;
  plan->source_register_count = source_role_layout->register_count;
  plan->result_register_count = result_role_layout->register_count;
  plan->lane_group_count = lane_group_count;
  plan->lane_divisor = layout->tile_shape.result_row_count;
  plan->source_lane_group_byte_shift =
      (uint16_t)iree_math_count_trailing_zeros_u32(
          source_lane_group_byte_count);
  plan->result_lane_div_byte_shift =
      result_lane_div_byte_count == 0
          ? 0
          : (uint16_t)iree_math_count_trailing_zeros_u32(
                result_lane_div_byte_count);
  if (source_row_layout->element_count == source_role_layout->register_count &&
      source_row_layout->outer_count == 1) {
    plan->source_register_selector =
        (loom_amdgpu_fragment_repack_lane_recipe_t){
            .and_mask = (uint16_t)(source_role_layout->register_count - 1u),
        };
    plan->source_lane_group =
        lane_group_count == 1
            ? (loom_amdgpu_fragment_repack_lane_recipe_t){0}
            : (loom_amdgpu_fragment_repack_lane_recipe_t){
                  .and_mask = UINT16_MAX,
                  .right_shift = (uint16_t)iree_math_count_trailing_zeros_u32(
                      source_role_layout->register_count),
              };
  } else if (source_row_layout->outer_count ==
                 source_role_layout->register_count &&
             source_row_layout->element_count == 1) {
    plan->source_register_selector =
        (loom_amdgpu_fragment_repack_lane_recipe_t){
            .and_mask = UINT16_MAX,
            .right_shift =
                (uint16_t)iree_math_count_trailing_zeros_u32(lane_group_count),
        };
    plan->source_lane_group =
        lane_group_count == 1
            ? (loom_amdgpu_fragment_repack_lane_recipe_t){0}
            : (loom_amdgpu_fragment_repack_lane_recipe_t){
                  .and_mask = (uint16_t)(lane_group_count - 1u),
              };
  } else {
    return false;
  }

  if (loom_amdgpu_fragment_repack_can_pack_adjacent_source_lanes(
          descriptor_set, layout, plan)) {
    if (loom_amdgpu_fragment_repack_select_transpose_strategy(
            descriptor_set, layout->wave_size, plan)) {
      return true;
    }
    if (loom_amdgpu_fragment_repack_has_result_to_lhs_bf16_bpermute_descriptors(
            descriptor_set, plan)) {
      plan->strategy =
          LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_PACKED_BPERMUTE;
      return true;
    }
  }

  if (!loom_amdgpu_fragment_repack_has_result_to_lhs_bf16_bpermute_descriptors(
          descriptor_set, plan)) {
    plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_PACKETS;
    plan->strategy = LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC;
    return false;
  }

  return true;
}

static bool loom_amdgpu_fragment_repack_select_target_strategy(
    const loom_module_t* module, const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_contract_candidates_t*
        contract_candidates,
    const loom_value_fact_table_t* fact_table, loom_value_id_t blocks,
    loom_value_id_t rows, loom_value_id_t columns,
    const loom_amdgpu_target_facts_t* target_facts,
    loom_amdgpu_fragment_repack_plan_t* plan) {
  if (bundle == NULL || bundle->snapshot == NULL || descriptor_set == NULL) {
    plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_LAYOUT;
    return false;
  }

  if (!loom_amdgpu_fragment_repack_contract_role(plan->source_role_flags,
                                                 &plan->source_role) ||
      !loom_amdgpu_fragment_repack_contract_role(plan->result_role_flags,
                                                 &plan->result_role)) {
    return false;
  }

  const loom_amdgpu_matrix_feature_bits_t feature_bits =
      contract_candidates != NULL
          ? contract_candidates->feature_bits
          : loom_amdgpu_matrix_fragment_feature_bits(target_facts);
  const uint32_t wave_size = contract_candidates != NULL
                                 ? contract_candidates->wave_size
                                 : bundle->snapshot->subgroup_size;
  const iree_host_size_t descriptor_count =
      loom_amdgpu_matrix_fragment_contract_candidate_count(contract_candidates);
  bool found_storage_candidate = false;
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_fragment_contract_candidate_at(contract_candidates,
                                                          i);
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    if (layout == NULL ||
        (contract_candidates == NULL &&
         !loom_amdgpu_matrix_fragment_contract_is_available(
             descriptor, descriptor_set, feature_bits, wave_size))) {
      continue;
    }
    if (!loom_amdgpu_matrix_fragment_shape_matches(
            fact_table, layout, plan->source_role, blocks, rows, columns) ||
        !loom_amdgpu_matrix_fragment_shape_matches(
            fact_table, layout, plan->result_role, blocks, rows, columns)) {
      continue;
    }

    loom_amdgpu_matrix_fragment_role_storage_t source_storage = {0};
    loom_amdgpu_matrix_fragment_role_storage_t result_storage = {0};
    if (!loom_amdgpu_matrix_fragment_descriptor_role_storage(
            descriptor, plan->source_role, &source_storage) ||
        !loom_amdgpu_matrix_fragment_descriptor_role_storage(
            descriptor, plan->result_role, &result_storage)) {
      continue;
    }

    const loom_matrix_fragment_role_layout_t* source_role_layout =
        loom_matrix_fragment_role_layout(layout, plan->source_role);
    const loom_matrix_fragment_role_layout_t* result_role_layout =
        loom_matrix_fragment_role_layout(layout, plan->result_role);
    if (!loom_amdgpu_matrix_fragment_payload_matches_role_storage(
            plan->source_type, source_storage.element_type,
            source_role_layout) ||
        !loom_amdgpu_matrix_fragment_payload_matches_role_storage(
            plan->result_type, result_storage.element_type,
            result_role_layout)) {
      continue;
    }
    found_storage_candidate = true;

    if (loom_amdgpu_select_result_to_rhs_packed_b16_fragment_repack_plan(
            descriptor_set, layout, source_role_layout, result_role_layout,
            plan)) {
      return true;
    }
    if (loom_amdgpu_fragment_repack_select_result_to_lhs_bf16_bpermute_strategy(
            descriptor_set, layout, source_role_layout, result_role_layout,
            plan)) {
      return true;
    }
  }

  if (plan->reason != LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_PACKETS) {
    plan->reason = found_storage_candidate
                       ? LOOM_AMDGPU_FRAGMENT_REPACK_REASON_LAYOUT_STRATEGY
                       : LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_LAYOUT;
  }
  return false;
}

iree_status_t loom_amdgpu_select_vector_fragment_repack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_repack_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  const loom_value_id_t source = loom_vector_fragment_repack_source(source_op);
  const loom_value_id_t result = loom_vector_fragment_repack_result(source_op);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_vector_fragment_role_flags_t result_role_flags =
      loom_vector_fragment_fact_role_flags(
          loom_vector_fragment_repack_role(source_op));
  *out_plan = (loom_amdgpu_fragment_repack_plan_t){
      .source = source,
      .result = result,
      .strategy = LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC,
      .reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SOURCE_FACTS,
      .source_role_flags = 0,
      .result_role_flags = result_role_flags,
      .transpose_predicate =
          {
              .constant = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
              .equal_zero = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
              .not_equal_zero = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
          },
      .packed_pair =
          {
              .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
          },
      .source_type = source_type,
      .result_type = result_type,
  };

  loom_vector_fragment_fact_t source_fact = {0};
  if (fact_table == NULL ||
      !loom_vector_fragment_fact_query_value_facts(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, source), &source_fact) ||
      source_fact.role_flags == 0 || result_role_flags == 0) {
    *out_selected = true;
    return iree_ok_status();
  }
  out_plan->source_role_flags = source_fact.role_flags;

  if (!loom_amdgpu_fragment_repack_shape_matches(
          fact_table, source_fact,
          loom_vector_fragment_repack_blocks(source_op),
          loom_vector_fragment_repack_rows(source_op),
          loom_vector_fragment_repack_columns(source_op))) {
    out_plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SHAPE;
    *out_selected = true;
    return iree_ok_status();
  }

  const loom_amdgpu_fragment_repack_reason_t reason =
      loom_amdgpu_fragment_repack_transition_reason(
          source_fact.role_flags, result_role_flags, source_type, result_type);
  if (reason == LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE) {
    out_plan->strategy = LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_ALIAS;
  } else {
    out_plan->reason = reason;
    const loom_amdgpu_matrix_fragment_contract_candidates_t*
        contract_candidates = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_fragment_contract_candidates(
        context, &contract_candidates));
    (void)loom_amdgpu_fragment_repack_select_target_strategy(
        module, loom_low_lower_context_bundle(context),
        loom_low_lower_context_descriptor_set(context), contract_candidates,
        fact_table, loom_vector_fragment_repack_blocks(source_op),
        loom_vector_fragment_repack_rows(source_op),
        loom_vector_fragment_repack_columns(source_op),
        loom_amdgpu_target_facts_cast(
            loom_low_lower_context_target_facts(context)),
        out_plan);
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(
          loom_amdgpu_fragment_repack_role_flags_key(plan->source_role_flags)),
      loom_param_string(
          loom_amdgpu_fragment_repack_role_flags_key(plan->result_role_flags)),
      loom_param_type(plan->source_type),
      loom_param_type(plan->result_type),
      loom_param_string(loom_amdgpu_fragment_repack_plan_key(plan)),
      loom_param_string(loom_amdgpu_fragment_repack_reason_key(plan->reason)),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_041_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_emit_fragment_repack_source_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan, loom_value_id_t low_source,
    loom_type_t vgpr_type, loom_value_id_t* out_source_registers) {
  if (plan->source_register_count == 1) {
    out_source_registers[0] = low_source;
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                    low_source, i, vgpr_type,
                                                    &out_source_registers[i]));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_narrow_source_registers_to_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    loom_type_t vgpr_type, loom_value_id_t* inout_source_registers) {
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
        context, source_op, bf16_pack_descriptors, inout_source_registers[i],
        vgpr_type, &inout_source_registers[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_lane_recipe(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_repack_lane_recipe_t recipe,
    loom_value_id_t low_lane_coordinate, loom_type_t vgpr_type,
    loom_value_id_t* out_low_coordinate) {
  *out_low_coordinate = LOOM_VALUE_ID_INVALID;
  if (recipe.and_mask == 0) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, out_low_coordinate);
  }

  loom_value_id_t low_masked_coordinate = low_lane_coordinate;
  if (recipe.and_mask != UINT16_MAX) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        low_lane_coordinate, recipe.and_mask, vgpr_type,
        &low_masked_coordinate));
  }
  if (recipe.right_shift == 0) {
    *out_low_coordinate = low_masked_coordinate;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      recipe.right_shift, low_masked_coordinate, vgpr_type, out_low_coordinate);
}

static iree_status_t loom_amdgpu_emit_fragment_repack_source_register_selector(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_value_id_t* out_low_source_selector) {
  *out_low_source_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_mod(
      context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
  return loom_amdgpu_emit_fragment_repack_lane_recipe(
      context, source_op, plan->source_register_selector, lane_ids->lane_mod,
      vgpr_type, out_low_source_selector);
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_exact_source_register_masks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_selector, uint16_t source_register_count,
    loom_type_t vgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_source_register_masks) {
  if (source_register_count <= 1) {
    return iree_ok_status();
  }

  for (uint16_t i = 1; i < source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_compare_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
            [LOOM_VECTOR_CMPI_PREDICATE_EQ]
                .src1_inline_descriptor_ref,
        low_source_selector, i, vgpr_type, mask_type,
        &out_source_register_masks[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_source_register_bit_masks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_selector, uint16_t source_register_count,
    loom_type_t vgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_source_register_bit_masks) {
  if (source_register_count <= 1) {
    return iree_ok_status();
  }

  const uint16_t bit_count =
      (uint16_t)iree_math_count_trailing_zeros_u32(source_register_count);
  for (uint16_t i = 0; i < bit_count; ++i) {
    loom_value_id_t low_selector_bit = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        low_source_selector, UINT32_C(1) << i, vgpr_type, &low_selector_bit));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_compare_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
            [LOOM_VECTOR_CMPI_PREDICATE_NE]
                .src1_inline_descriptor_ref,
        low_selector_bit, 0, vgpr_type, mask_type,
        &out_source_register_bit_masks[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_lane_group_byte_base(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_value_id_t* out_low_byte_base) {
  *out_low_byte_base = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_fragment_repack_has_static_zero_source_byte_base(plan)) {
    if (plan->result_lane_div_byte_shift == 0) {
      return loom_amdgpu_emit_const_u32(context, source_op,
                                        LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                        vgpr_type, out_low_byte_base);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_div(
        context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        plan->result_lane_div_byte_shift, lane_ids->lane_div, vgpr_type,
        out_low_byte_base);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_mod(
      context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
  loom_value_id_t source_lane_group = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_lane_recipe(
      context, source_op, plan->source_lane_group, lane_ids->lane_mod,
      vgpr_type, &source_lane_group));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      plan->source_lane_group_byte_shift, source_lane_group, vgpr_type,
      out_low_byte_base));
  if (plan->result_lane_div_byte_shift == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_div(
      context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
  loom_value_id_t low_lane_div_byte_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      plan->result_lane_div_byte_shift, lane_ids->lane_div, vgpr_type,
      &low_lane_div_byte_base));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      *out_low_byte_base, low_lane_div_byte_base, vgpr_type, out_low_byte_base);
}

static iree_status_t loom_amdgpu_emit_fragment_repack_bpermute_candidates(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* bpermute_descriptor,
    const loom_value_id_t* source_registers, uint16_t source_register_count,
    loom_value_id_t low_source_byte_offset, uint32_t static_byte_offset,
    loom_type_t vgpr_type, loom_value_id_t* out_low_candidates) {
  for (uint16_t i = 0; i < source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, bpermute_descriptor, low_source_byte_offset,
        static_byte_offset, source_registers[i], vgpr_type,
        &out_low_candidates[i]));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_repack_packed_pair_is_fused(
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  return plan->packed_pair.kind ==
             LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_U16 ||
         plan->packed_pair.kind ==
             LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_BF16;
}

static bool loom_amdgpu_fragment_repack_must_pre_narrow_source_registers(
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors) {
  switch (plan->packed_pair.kind) {
    case LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_U16:
      return true;
    case LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_DPP_PACK_BF16:
      return false;
    default:
      return !iree_any_bit_set(
          bf16_pack_descriptors->flags,
          LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE);
  }
}

static iree_status_t loom_amdgpu_emit_fragment_repack_dpp_packed_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t source_register, uint32_t dpp_ctrl, loom_type_t vgpr_type,
    loom_value_id_t* out_packed_register) {
  *out_packed_register = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("dpp_ctrl"), dpp_ctrl, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {source_register, source_register};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &vgpr_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_packed_register = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_packed_source_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* packed_pair_descriptor,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    const loom_value_id_t* source_registers, bool pre_narrow_source_registers,
    loom_type_t vgpr_type, loom_value_id_t* out_packed_source_registers) {
  if (loom_amdgpu_fragment_repack_packed_pair_is_fused(plan)) {
    for (uint16_t i = 0; i < plan->source_register_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_dpp_packed_pair(
          context, source_op, packed_pair_descriptor, source_registers[i],
          plan->packed_pair.immediate, vgpr_type,
          &out_packed_source_registers[i]));
    }
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(plan->packed_pair.kind,
                 LOOM_AMDGPU_FRAGMENT_REPACK_PACKED_PAIR_EXCHANGE_THEN_PACK);
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(bf16_pack_descriptors->flags,
                       LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)
          ? &bf16_pack_descriptors->pack_u16_descriptor
          : NULL;
  loom_value_id_t
      paired_source_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
          LOOM_VALUE_ID_INVALID};
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_direct_crosslane_register(
        context, source_op, packed_pair_descriptor,
        plan->packed_pair.crosslane_kind, source_registers[i],
        plan->packed_pair.immediate, vgpr_type, &paired_source_registers[i]));
  }
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    if (pre_narrow_source_registers) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_u16_lane_pair(
          context, source_op, pack_u16_descriptor, source_registers[i],
          paired_source_registers[i], vgpr_type,
          &out_packed_source_registers[i]));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, bf16_pack_descriptors, source_registers[i],
              paired_source_registers[i], vgpr_type,
              &out_packed_source_registers[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_linear_bpermute_element(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* bpermute_descriptor,
    const loom_value_id_t* source_registers, uint16_t source_register_count,
    const loom_value_id_t* source_register_masks,
    loom_value_id_t low_source_byte_offset, uint32_t static_byte_offset,
    loom_type_t vgpr_type, loom_value_id_t* out_low_element) {
  *out_low_element = LOOM_VALUE_ID_INVALID;
  loom_value_id_t candidates[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_bpermute_candidates(
      context, source_op, bpermute_descriptor, source_registers,
      source_register_count, low_source_byte_offset, static_byte_offset,
      vgpr_type, candidates));
  *out_low_element = candidates[0];
  for (uint16_t i = 1; i < source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
        context, source_op, *out_low_element, candidates[i],
        source_register_masks[i], vgpr_type, out_low_element));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_tree_bpermute_element(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* bpermute_descriptor,
    const loom_value_id_t* source_registers, uint16_t source_register_count,
    const loom_value_id_t* source_register_bit_masks,
    loom_value_id_t low_source_byte_offset, uint32_t static_byte_offset,
    loom_type_t vgpr_type, loom_value_id_t* out_low_element) {
  *out_low_element = LOOM_VALUE_ID_INVALID;
  loom_value_id_t selected[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_bpermute_candidates(
      context, source_op, bpermute_descriptor, source_registers,
      source_register_count, low_source_byte_offset, static_byte_offset,
      vgpr_type, selected));
  uint16_t selected_count = source_register_count;
  for (uint16_t bit_index = 0; selected_count > 1; ++bit_index) {
    uint16_t next_count = 0;
    for (uint16_t i = 0; i < selected_count; i += 2) {
      if (i + 1 == selected_count) {
        selected[next_count++] = selected[i];
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
          context, source_op, selected[i], selected[i + 1],
          source_register_bit_masks[bit_index], vgpr_type,
          &selected[next_count++]));
    }
    selected_count = next_count;
  }
  *out_low_element = selected[0];
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_repack_uses_packed_source_pairs(
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  return plan->strategy ==
             LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_PACKED_BPERMUTE ||
         plan->strategy ==
             LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_TRANSPOSE_BPERMUTE;
}

static iree_status_t loom_amdgpu_emit_fragment_repack_transpose_lane_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_value_id_t* out_lane_bits) {
  bool has_dynamic_predicate = false;
  for (uint16_t bit_index = 0; bit_index < plan->transpose_bit_count;
       ++bit_index) {
    has_dynamic_predicate |=
        plan->strategy_payload.transpose_stages[bit_index].lane_bit_set_mask ==
        0;
  }
  if (!has_dynamic_predicate) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_matrix_fragment_lane_mod(
      context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
  for (uint16_t bit_index = 0; bit_index < plan->transpose_bit_count;
       ++bit_index) {
    if (plan->strategy_payload.transpose_stages[bit_index].lane_bit_set_mask !=
        0) {
      continue;
    }
    const uint32_t lane_bit = UINT32_C(2) << bit_index;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        lane_ids->lane_mod, lane_bit, vgpr_type, &out_lane_bits[bit_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_transpose_stage_masks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_value_id_t* lane_bits, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_stage_masks) {
  for (uint16_t bit_index = 0; bit_index < plan->transpose_bit_count;
       ++bit_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_compare_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
            [LOOM_VECTOR_CMPI_PREDICATE_NE]
                .src1_inline_descriptor_ref,
        lane_bits[bit_index], 0, vgpr_type, mask_type,
        &out_stage_masks[bit_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_vcc_compare_zero(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t lane_bits, loom_type_t vcc_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("lhs"), 0, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, &lane_bits, 1,
      loom_make_named_attr_slice(attrs, attr_count), &vcc_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_dpp_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t false_value, loom_value_id_t true_value,
    loom_value_id_t condition, uint32_t dpp_ctrl, loom_type_t vgpr_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("dpp_ctrl"), dpp_ctrl, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {false_value, true_value, condition};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &vgpr_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_vcc_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, uint32_t mask,
    loom_type_t vcc_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), mask, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* low_const = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, descriptor, loom_make_named_attr_slice(attrs, attr_count),
      vcc_type, source_op->location, &low_const));
  *out_mask = loom_low_const_result(low_const);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_masked_dpp_update(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_old_value, loom_value_id_t low_source_value,
    uint32_t dpp_ctrl, uint8_t bank_mask, loom_type_t vgpr_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[2];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("dpp_ctrl"), dpp_ctrl, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("bank_mask"), bank_mask,
                                  attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {low_old_value, low_source_value};
  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &vgpr_type, 1,
      &tied_result, 1, source_op->location, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_fused_transposed_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_value_id_t* source_registers, const loom_value_id_t* lane_bits,
    loom_type_t vgpr_type, loom_value_id_t* out_transposed_registers) {
  loom_low_lower_resolved_descriptor_t constant_descriptor = {0};
  loom_low_lower_resolved_descriptor_t equal_zero_descriptor = {0};
  loom_low_lower_resolved_descriptor_t not_equal_zero_descriptor = {0};
  if (plan->transpose_predicate.constant != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->transpose_predicate.constant, &constant_descriptor));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->transpose_predicate.equal_zero, &equal_zero_descriptor));
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->transpose_predicate.not_equal_zero,
        &not_equal_zero_descriptor));
  }
  loom_type_t vcc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vcc_type(context, &vcc_type));

  loom_value_id_t current_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  loom_value_id_t next_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  memcpy(current_registers, source_registers,
         plan->source_register_count * sizeof(current_registers[0]));

  for (uint16_t bit_index = 0; bit_index < plan->transpose_bit_count;
       ++bit_index) {
    const uint16_t register_bit = (uint16_t)(UINT16_C(1) << bit_index);
    const loom_amdgpu_fragment_repack_transpose_stage_t* stage =
        &plan->strategy_payload.transpose_stages[bit_index];
    const loom_amdgpu_direct_xor_lane_recipe_t* recipe = &stage->exchange;
    loom_low_lower_resolved_descriptor_t conditional_descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, recipe->conditional_ref, &conditional_descriptor));

    const loom_low_lower_resolved_descriptor_t* predicate_descriptors[] = {
        &equal_zero_descriptor,
        &not_equal_zero_descriptor,
    };
    const uint16_t conditional_register_bit_value_count =
        stage->lane_bit_set_bank_mask == 0 ? 2 : 1;
    for (uint16_t register_bit_value = 0;
         register_bit_value < conditional_register_bit_value_count;
         ++register_bit_value) {
      loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
      if (stage->lane_bit_set_mask != 0) {
        const uint32_t predicate_mask = register_bit_value == 0
                                            ? ~stage->lane_bit_set_mask
                                            : stage->lane_bit_set_mask;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_vcc_constant(
            context, source_op, &constant_descriptor, predicate_mask, vcc_type,
            &low_predicate));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_vcc_compare_zero(
            context, source_op, predicate_descriptors[register_bit_value],
            lane_bits[bit_index], vcc_type, &low_predicate));
      }
      const uint16_t register_group_size = (uint16_t)(register_bit * 2u);
      for (uint16_t register_group_base =
               (uint16_t)(register_bit_value * register_bit);
           register_group_base < plan->source_register_count;
           register_group_base =
               (uint16_t)(register_group_base + register_group_size)) {
        for (uint16_t register_offset = 0; register_offset < register_bit;
             ++register_offset) {
          const uint16_t register_index =
              (uint16_t)(register_group_base + register_offset);
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_dpp_select(
              context, source_op, &conditional_descriptor,
              current_registers[register_index ^ register_bit],
              current_registers[register_index], low_predicate,
              recipe->immediate, vgpr_type, &next_registers[register_index]));
        }
      }
    }
    if (stage->lane_bit_set_bank_mask != 0) {
      loom_low_lower_resolved_descriptor_t masked_move_descriptor = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
          context, recipe->masked_move_ref, &masked_move_descriptor));
      const uint8_t bank_mask = (uint8_t)(stage->lane_bit_set_bank_mask ^ 0xFu);
      const uint16_t register_group_size = (uint16_t)(register_bit * 2u);
      for (uint16_t register_group_base = register_bit;
           register_group_base < plan->source_register_count;
           register_group_base =
               (uint16_t)(register_group_base + register_group_size)) {
        for (uint16_t register_offset = 0; register_offset < register_bit;
             ++register_offset) {
          const uint16_t register_index =
              (uint16_t)(register_group_base + register_offset);
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_emit_fragment_repack_masked_dpp_update(
                  context, source_op, &masked_move_descriptor,
                  current_registers[register_index],
                  current_registers[register_index ^ register_bit],
                  recipe->immediate, bank_mask, vgpr_type,
                  &next_registers[register_index]));
        }
      }
    }
    memcpy(current_registers, next_registers,
           plan->source_register_count * sizeof(current_registers[0]));
  }

  memcpy(out_transposed_registers, current_registers,
         plan->source_register_count * sizeof(current_registers[0]));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_transposed_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_value_id_t* source_registers, const loom_value_id_t* stage_masks,
    loom_type_t vgpr_type, loom_value_id_t* out_transposed_registers) {
  loom_value_id_t current_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  loom_value_id_t next_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  memcpy(current_registers, source_registers,
         plan->source_register_count * sizeof(current_registers[0]));

  for (uint16_t bit_index = 0; bit_index < plan->transpose_bit_count;
       ++bit_index) {
    const uint16_t register_bit = (uint16_t)(UINT16_C(1) << bit_index);
    const loom_amdgpu_direct_xor_lane_recipe_t* recipe =
        &plan->strategy_payload.transpose_stages[bit_index].exchange;
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, recipe->descriptor_ref, &descriptor));
    loom_value_id_t peer_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
        LOOM_VALUE_ID_INVALID};
    for (uint16_t source_register_index = 0;
         source_register_index < plan->source_register_count;
         ++source_register_index) {
      const uint16_t register_index =
          (uint16_t)(source_register_index ^ register_bit);
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_direct_crosslane_register(
          context, source_op, &descriptor, recipe->crosslane_kind,
          current_registers[source_register_index], recipe->immediate,
          vgpr_type, &peer_registers[register_index]));
    }
    for (uint16_t output_index = 0; output_index < plan->source_register_count;
         ++output_index) {
      const uint16_t register_index =
          bit_index + 1 == plan->transpose_bit_count
              ? (uint16_t)(output_index ^ register_bit)
              : output_index;
      const bool register_bit_set = (register_index & register_bit) != 0;
      const loom_value_id_t low_false_value =
          register_bit_set ? peer_registers[register_index]
                           : current_registers[register_index];
      const loom_value_id_t low_true_value =
          register_bit_set ? current_registers[register_index]
                           : peer_registers[register_index];
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
          context, source_op, low_false_value, low_true_value,
          stage_masks[bit_index], vgpr_type, &next_registers[register_index]));
    }
    memcpy(current_registers, next_registers,
           plan->source_register_count * sizeof(current_registers[0]));
  }

  memcpy(out_transposed_registers, current_registers,
         plan->source_register_count * sizeof(current_registers[0]));
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_transposed_lane_group_byte_base(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    loom_value_id_t low_lane_group_byte_base,
    loom_value_id_t low_source_selector, loom_type_t vgpr_type,
    loom_value_id_t* out_low_byte_base) {
  const uint32_t transpose_mask =
      (UINT32_C(1) << plan->transpose_bit_count) - 1u;
  loom_value_id_t low_transposed_register_bits = low_source_selector;
  if (plan->transposed_source_register_candidate_count > 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        low_source_selector, transpose_mask, vgpr_type,
        &low_transposed_register_bits));
  }
  loom_value_id_t low_transposed_register_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 3,
      low_transposed_register_bits, vgpr_type,
      &low_transposed_register_byte_offset));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      low_lane_group_byte_base, low_transposed_register_byte_offset, vgpr_type,
      out_low_byte_base);
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_result_to_lhs_bf16_bpermute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_source_registers(
      context, source_op, plan, low_source, vgpr_type, source_registers));

  const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_bf16_pack_descriptors(context, &bf16_pack_descriptors));
  const bool pre_narrow_source_registers =
      loom_amdgpu_fragment_repack_must_pre_narrow_source_registers(
          plan, bf16_pack_descriptors);
  if (pre_narrow_source_registers) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_narrow_source_registers_to_bf16(
            context, source_op, plan, bf16_pack_descriptors, vgpr_type,
            source_registers));
  }

  loom_amdgpu_matrix_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_matrix_fragment_lane_ids(
      context, source_op, plan->lane_divisor, vgpr_type, &lane_ids));
  const bool use_transpose =
      plan->strategy ==
      LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_TRANSPOSE_BPERMUTE;
  const uint16_t selected_source_register_count =
      use_transpose ? plan->transposed_source_register_candidate_count
                    : plan->source_register_count;
  loom_value_id_t low_source_selector = LOOM_VALUE_ID_INVALID;
  if (plan->source_register_count > 1) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_source_register_selector(
            context, source_op, plan, &lane_ids, vgpr_type,
            &low_source_selector));
  }
  loom_value_id_t low_selected_source_register = low_source_selector;
  if (use_transpose && selected_source_register_count > 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        plan->transpose_bit_count, low_source_selector, vgpr_type,
        &low_selected_source_register));
  }
  loom_value_id_t
      source_register_masks[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
          LOOM_VALUE_ID_INVALID};
  const bool use_source_register_bit_tree =
      loom_amdgpu_fragment_repack_uses_source_register_bit_tree(
          selected_source_register_count);
  if (use_source_register_bit_tree) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_source_register_bit_masks(
            context, source_op, low_selected_source_register,
            selected_source_register_count, vgpr_type, mask_type,
            source_register_masks));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_exact_source_register_masks(
            context, source_op, low_selected_source_register,
            selected_source_register_count, vgpr_type, mask_type,
            source_register_masks));
  }

  loom_value_id_t low_lane_group_byte_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_lane_group_byte_base(
      context, source_op, plan, &lane_ids, vgpr_type,
      &low_lane_group_byte_base));
  if (use_transpose) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_transposed_lane_group_byte_base(
            context, source_op, plan, low_lane_group_byte_base,
            low_source_selector, vgpr_type, &low_lane_group_byte_base));
  }

  loom_low_lower_resolved_descriptor_t bpermute_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &bpermute_descriptor));
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(bf16_pack_descriptors->flags,
                       LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)
          ? &bf16_pack_descriptors->pack_u16_descriptor
          : NULL;

  const bool use_packed_source_pairs =
      loom_amdgpu_fragment_repack_uses_packed_source_pairs(plan);
  const uint32_t packed_pair_source_lane_byte_offset =
      loom_amdgpu_fragment_repack_packed_pair_is_fused(plan)
          ? LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT
          : 0u;
  loom_value_id_t
      packed_source_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
          LOOM_VALUE_ID_INVALID};
  if (use_packed_source_pairs) {
    loom_low_lower_resolved_descriptor_t packed_pair_descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->packed_pair.descriptor_ref, &packed_pair_descriptor));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_packed_source_registers(
            context, source_op, plan, &packed_pair_descriptor,
            bf16_pack_descriptors, source_registers,
            pre_narrow_source_registers, vgpr_type, packed_source_registers));
  }

  loom_value_id_t
      transposed_source_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
          LOOM_VALUE_ID_INVALID};
  if (use_transpose) {
    loom_value_id_t transpose_lane_bits
        [LOOM_AMDGPU_FRAGMENT_REPACK_TRANSPOSE_STAGE_CAPACITY] = {
            LOOM_VALUE_ID_INVALID};
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_transpose_lane_bits(
        context, source_op, plan, &lane_ids, vgpr_type, transpose_lane_bits));
    if (plan->transpose_predicate.constant != LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
        plan->transpose_predicate.equal_zero !=
            LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_repack_fused_transposed_registers(
              context, source_op, plan, packed_source_registers,
              transpose_lane_bits, vgpr_type, transposed_source_registers));
    } else {
      loom_value_id_t transpose_stage_masks
          [LOOM_AMDGPU_FRAGMENT_REPACK_TRANSPOSE_STAGE_CAPACITY] = {
              LOOM_VALUE_ID_INVALID};
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_repack_transpose_stage_masks(
              context, source_op, plan, transpose_lane_bits, vgpr_type,
              mask_type, transpose_stage_masks));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_repack_transposed_registers(
              context, source_op, plan, packed_source_registers,
              transpose_stage_masks, vgpr_type, transposed_source_registers));
    }
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  for (uint16_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (use_transpose) {
      const uint16_t transpose_mask =
          (uint16_t)((UINT16_C(1) << plan->transpose_bit_count) - 1u);
      loom_value_id_t
          candidate_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
              LOOM_VALUE_ID_INVALID};
      for (uint16_t candidate_index = 0;
           candidate_index < selected_source_register_count;
           ++candidate_index) {
        const uint16_t source_register_index =
            (uint16_t)((candidate_index << plan->transpose_bit_count) |
                       (register_index & transpose_mask));
        candidate_registers[candidate_index] =
            transposed_source_registers[source_register_index];
      }
      const uint32_t static_byte_offset =
          (uint32_t)(register_index & (uint16_t)~transpose_mask) *
              LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT *
              LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT +
          packed_pair_source_lane_byte_offset;
      if (use_source_register_bit_tree) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_tree_bpermute_element(
                context, source_op, &bpermute_descriptor, candidate_registers,
                selected_source_register_count, source_register_masks,
                low_lane_group_byte_base, static_byte_offset, vgpr_type,
                &result_registers[register_index]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_linear_bpermute_element(
                context, source_op, &bpermute_descriptor, candidate_registers,
                selected_source_register_count, source_register_masks,
                low_lane_group_byte_base, static_byte_offset, vgpr_type,
                &result_registers[register_index]));
      }
      continue;
    }

    if (use_packed_source_pairs) {
      const uint32_t static_byte_offset =
          (uint32_t)register_index * LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT *
              LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT +
          packed_pair_source_lane_byte_offset;
      if (use_source_register_bit_tree) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_tree_bpermute_element(
                context, source_op, &bpermute_descriptor,
                packed_source_registers, selected_source_register_count,
                source_register_masks, low_lane_group_byte_base,
                static_byte_offset, vgpr_type,
                &result_registers[register_index]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_linear_bpermute_element(
                context, source_op, &bpermute_descriptor,
                packed_source_registers, selected_source_register_count,
                source_register_masks, low_lane_group_byte_base,
                static_byte_offset, vgpr_type,
                &result_registers[register_index]));
      }
      continue;
    }

    loom_value_id_t elements[LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT] = {
        LOOM_VALUE_ID_INVALID};
    for (uint16_t element_index = 0;
         element_index < LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
         ++element_index) {
      const uint16_t reduction =
          (uint16_t)(register_index *
                         LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT +
                     element_index);
      const uint32_t static_byte_offset =
          (uint32_t)reduction * LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT;
      if (use_source_register_bit_tree) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_tree_bpermute_element(
                context, source_op, &bpermute_descriptor, source_registers,
                selected_source_register_count, source_register_masks,
                low_lane_group_byte_base, static_byte_offset, vgpr_type,
                &elements[element_index]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_linear_bpermute_element(
                context, source_op, &bpermute_descriptor, source_registers,
                selected_source_register_count, source_register_masks,
                low_lane_group_byte_base, static_byte_offset, vgpr_type,
                &elements[element_index]));
      }
    }
    if (pre_narrow_source_registers) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_u16_lane_pair(
          context, source_op, pack_u16_descriptor, elements[0], elements[1],
          vgpr_type, &result_registers[register_index]));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, bf16_pack_descriptors, elements[0],
              elements[1], vgpr_type, &result_registers[register_index]));
    }
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_fragment_repack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  switch (plan->strategy) {
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_ALIAS:
      return loom_low_lower_bind_value_alias(context, plan->source,
                                             plan->result);
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_PACKED_BPERMUTE:
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_BPERMUTE:
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_TRANSPOSE_BPERMUTE:
      return loom_amdgpu_emit_fragment_repack_result_to_lhs_bf16_bpermute(
          context, source_op, plan);
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_RHS_PACKED_B16_XOR_PERMUTE:
      return loom_amdgpu_lower_result_to_rhs_packed_b16_fragment_repack(
          context, source_op, plan);
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC:
      return loom_amdgpu_emit_fragment_repack_diagnostic(context, source_op,
                                                         plan);
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_NONE:
    default:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU fragment repack strategy");
  IREE_BUILTIN_UNREACHABLE();
}
