// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_state.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"

static int loom_amdgpu_matrix_fragment_state_key;

iree_status_t loom_amdgpu_matrix_fragment_state(
    loom_low_lower_context_t* context,
    loom_amdgpu_matrix_fragment_state_t** out_cache) {
  return loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_matrix_fragment_state_key, sizeof(**out_cache),
      (void**)out_cache);
}

iree_status_t loom_amdgpu_matrix_fragment_contract_candidates(
    loom_low_lower_context_t* context,
    const loom_amdgpu_matrix_fragment_contract_candidates_t** out_candidates) {
  *out_candidates = NULL;
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (bundle == NULL || bundle->snapshot == NULL || descriptor_set == NULL) {
    return iree_ok_status();
  }

  const loom_amdgpu_matrix_feature_bits_t feature_bits =
      loom_amdgpu_matrix_fragment_feature_bits(loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context)));
  const uint32_t wave_size = bundle->snapshot->subgroup_size;
  loom_amdgpu_matrix_fragment_state_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_fragment_state(context, &cache));
  loom_amdgpu_matrix_fragment_contract_candidates_t* candidates =
      &cache->contract_candidates;
  if (candidates->descriptor_set == descriptor_set &&
      candidates->feature_bits == feature_bits &&
      candidates->wave_size == wave_size) {
    *out_candidates = candidates;
    return iree_ok_status();
  }

  const iree_host_size_t descriptor_count =
      loom_amdgpu_matrix_contract_descriptor_count();
  uint16_t* descriptor_ordinals = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, descriptor_count, sizeof(*descriptor_ordinals),
      (void**)&descriptor_ordinals));
  iree_host_size_t candidate_count = 0;
  uint64_t canonical_result_representation_bits = 0;
  uint64_t exact_result_representation_bits = 0;
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    if (!loom_amdgpu_matrix_fragment_contract_is_available(
            descriptor, descriptor_set, feature_bits, wave_size)) {
      continue;
    }
    IREE_ASSERT_LE(i, UINT16_MAX);
    descriptor_ordinals[candidate_count++] = (uint16_t)i;
    const loom_amdgpu_matrix_contract_realization_choices_t* choices =
        loom_amdgpu_matrix_contract_realization_choices_at(i);
    if (choices->canonical_result_representation_id !=
        LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE) {
      canonical_result_representation_bits |=
          UINT64_C(1) << choices->canonical_result_representation_id;
      exact_result_representation_bits |=
          UINT64_C(1) << choices->canonical_result_representation_id;
    }
    if (choices->operand_exchanged_result_representation_id !=
            LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE &&
        choices->operand_exchanged_contract_ordinal !=
            LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE) {
      const loom_amdgpu_matrix_contract_descriptor_t* exchanged_descriptor =
          loom_amdgpu_matrix_contract_descriptor_at(
              choices->operand_exchanged_contract_ordinal);
      if (loom_amdgpu_matrix_fragment_contract_is_available(
              exchanged_descriptor, descriptor_set, feature_bits, wave_size)) {
        exact_result_representation_bits |=
            UINT64_C(1) << choices->operand_exchanged_result_representation_id;
      }
    }
  }
  *candidates = (loom_amdgpu_matrix_fragment_contract_candidates_t){
      .descriptor_set = descriptor_set,
      .feature_bits = feature_bits,
      .wave_size = wave_size,
      .descriptor_ordinals = descriptor_ordinals,
      .descriptor_count = candidate_count,
      .canonical_result_representation_bits =
          canonical_result_representation_bits,
      .exact_result_representation_bits = exact_result_representation_bits,
  };
  *out_candidates = candidates;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_matrix_fragment_record_contract_ordinal(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    uint16_t contract_ordinal) {
  loom_amdgpu_matrix_fragment_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_fragment_state(context, &state));
  const loom_local_value_domain_t* value_domain =
      loom_low_lower_context_value_domain(context);
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, source_value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "matrix result is outside the source value domain");
  }
  if (state->contract_ordinals_by_value_ordinal == NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
        context, value_domain->value_count,
        sizeof(*state->contract_ordinals_by_value_ordinal),
        (void**)&state->contract_ordinals_by_value_ordinal));
    memset(state->contract_ordinals_by_value_ordinal, 0xFF,
           value_domain->value_count *
               sizeof(*state->contract_ordinals_by_value_ordinal));
    state->contract_ordinal_count = value_domain->value_count;
  }
  IREE_ASSERT_EQ(state->contract_ordinal_count, value_domain->value_count);
  uint16_t* recorded_ordinal =
      &state->contract_ordinals_by_value_ordinal[value_ordinal];
  if (*recorded_ordinal != LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE &&
      *recorded_ordinal != contract_ordinal) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "matrix result has conflicting source contracts");
  }
  *recorded_ordinal = contract_ordinal;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_matrix_fragment_query_contract_ordinal(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id, uint16_t* out_contract_ordinal) {
  *out_contract_ordinal = LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE;
  loom_amdgpu_matrix_fragment_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_target_contract_query_get_or_allocate_target_state(
      environment, &loom_amdgpu_matrix_fragment_state_key, sizeof(*state),
      (void**)&state));
  if (state == NULL || state->contract_ordinals_by_value_ordinal == NULL ||
      environment->value_domain == NULL) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(environment->value_domain,
                                          source_value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= state->contract_ordinal_count) {
    return iree_ok_status();
  }
  *out_contract_ordinal =
      state->contract_ordinals_by_value_ordinal[value_ordinal];
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_memory_lane_id_cache_matches(
    loom_low_lower_context_t* context,
    const loom_amdgpu_matrix_fragment_state_t* cache, uint16_t lane_divisor,
    loom_type_t vgpr_type) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  return builder->ip.block != NULL && builder->ip.before_op == NULL &&
         cache->lane_ids.block == builder->ip.block &&
         cache->lane_ids.lane_ids.lane != LOOM_VALUE_ID_INVALID &&
         loom_type_equal(cache->lane_ids.vgpr_type, vgpr_type) &&
         cache->lane_ids.lane_divisor == lane_divisor;
}

static iree_status_t loom_amdgpu_update_fragment_memory_lane_id_cache(
    loom_low_lower_context_t* context, uint16_t lane_divisor,
    loom_type_t vgpr_type,
    const loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  if (builder->ip.block == NULL || builder->ip.before_op != NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_matrix_fragment_state_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_fragment_state(context, &cache));
  if (loom_amdgpu_fragment_memory_lane_id_cache_matches(
          context, cache, lane_divisor, vgpr_type)) {
    cache->lane_ids.lane_ids = *lane_ids;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_ensure_matrix_fragment_lane_mod(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_matrix_fragment_lane_ids_t* inout_lane_ids) {
  if (inout_lane_ids->lane_mod != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (lane_divisor == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
        &inout_lane_ids->lane_mod));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        inout_lane_ids->lane, lane_divisor - 1u, vgpr_type,
        &inout_lane_ids->lane_mod));
  }
  return loom_amdgpu_update_fragment_memory_lane_id_cache(
      context, lane_divisor, vgpr_type, inout_lane_ids);
}

iree_status_t loom_amdgpu_ensure_matrix_fragment_lane_div(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_matrix_fragment_lane_ids_t* inout_lane_ids) {
  if (inout_lane_ids->lane_div != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      (uint16_t)iree_math_count_trailing_zeros_u32(lane_divisor),
      inout_lane_ids->lane, vgpr_type, &inout_lane_ids->lane_div));
  return loom_amdgpu_update_fragment_memory_lane_id_cache(
      context, lane_divisor, vgpr_type, inout_lane_ids);
}

iree_status_t loom_amdgpu_emit_matrix_fragment_lane_ids(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_matrix_fragment_lane_ids_t* out_lane_ids) {
  IREE_ASSERT_GT(lane_divisor, 0u);
  IREE_ASSERT_EQ(lane_divisor & (lane_divisor - 1u), 0u);
  *out_lane_ids = (loom_amdgpu_matrix_fragment_lane_ids_t){
      .lane = LOOM_VALUE_ID_INVALID,
      .lane_mod = LOOM_VALUE_ID_INVALID,
      .lane_div = LOOM_VALUE_ID_INVALID,
  };
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_amdgpu_matrix_fragment_state_t* cache = NULL;
  if (builder->ip.block != NULL && builder->ip.before_op == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_fragment_state(context, &cache));
    if (loom_amdgpu_fragment_memory_lane_id_cache_matches(
            context, cache, lane_divisor, vgpr_type)) {
      *out_lane_ids = cache->lane_ids.lane_ids;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, vgpr_type, &out_lane_ids->lane));
  if (cache != NULL) {
    cache->lane_ids.block = builder->ip.block;
    cache->lane_ids.vgpr_type = vgpr_type;
    cache->lane_ids.lane_divisor = lane_divisor;
    cache->lane_ids.lane_ids = *out_lane_ids;
  }
  return iree_ok_status();
}
