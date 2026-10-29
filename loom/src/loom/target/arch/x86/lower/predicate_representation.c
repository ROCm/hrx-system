// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/predicate_representation.h"

#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/feature_bits.h"

typedef enum loom_x86_predicate_representation_action_e {
  LOOM_X86_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT = 0,
  LOOM_X86_PREDICATE_REPRESENTATION_ACTION_EXTRACT = 1,
  LOOM_X86_PREDICATE_REPRESENTATION_ACTION_INSERT = 2,
  LOOM_X86_PREDICATE_REPRESENTATION_ACTION_SELECT = 3,
  LOOM_X86_PREDICATE_REPRESENTATION_ACTION_COMPARE = 4,
  LOOM_X86_PREDICATE_REPRESENTATION_ACTION_EXTEND_INPUT = 5,
  LOOM_X86_PREDICATE_REPRESENTATION_ACTION_TRUNCATE_RESULT = 6,
  LOOM_X86_PREDICATE_REPRESENTATION_ACTION_SLICE = 7,
} loom_x86_predicate_representation_action_t;

typedef enum loom_x86_predicate_profile_e {
  LOOM_X86_PREDICATE_PROFILE_AVX2 = 0,
  LOOM_X86_PREDICATE_PROFILE_AVX512 = 1,
} loom_x86_predicate_profile_t;

typedef struct loom_x86_predicate_representation_policy_t {
  // Instruction profile providing predicate operations.
  loom_x86_predicate_profile_t profile;
  // Largest logical predicate shape handled by this profile.
  uint32_t maximum_lane_count;
} loom_x86_predicate_representation_policy_t;

static bool loom_x86_predicate_type(loom_type_t source_type,
                                    uint32_t maximum_lane_count,
                                    uint32_t* out_lane_count) {
  if (out_lane_count != NULL) {
    *out_lane_count = 0;
  }
  if (!loom_type_is_vector(source_type) || loom_type_rank(source_type) != 1 ||
      !loom_type_is_all_static(source_type) ||
      loom_type_element_type(source_type) != LOOM_SCALAR_TYPE_I1) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(source_type, 0);
  if ((lane_count != 2 && lane_count != 4 && lane_count != 8 &&
       lane_count != 16 && lane_count != 32 && lane_count != 64) ||
      lane_count > maximum_lane_count) {
    return false;
  }
  if (out_lane_count != NULL) {
    *out_lane_count = (uint32_t)lane_count;
  }
  return true;
}

bool loom_x86_avx2_predicate_type(loom_type_t source_type,
                                  uint32_t* out_lane_count) {
  return loom_x86_predicate_type(source_type, 32, out_lane_count);
}

bool loom_x86_avx512_predicate_type(loom_type_t source_type,
                                    uint32_t* out_lane_count) {
  return loom_x86_predicate_type(source_type, 64, out_lane_count);
}

#define LOOM_X86_PREDICATE_MAX_CANDIDATE_COUNT 5u

static bool loom_x86_predicate_simd_bit_width(
    uint32_t lane_count, loom_low_representation_id_t representation,
    uint32_t* out_bit_width) {
  switch (representation) {
    case LOOM_X86_PREDICATE_REPRESENTATION_SIMD8:
    case LOOM_X86_PREDICATE_REPRESENTATION_SIMD16:
    case LOOM_X86_PREDICATE_REPRESENTATION_SIMD32:
    case LOOM_X86_PREDICATE_REPRESENTATION_SIMD64:
      *out_bit_width = lane_count * (uint32_t)representation;
      return true;
    default:
      *out_bit_width = 0;
      return false;
  }
}

static bool loom_x86_predicate_representation_available(
    const loom_x86_predicate_representation_policy_t* policy,
    uint32_t lane_count, loom_low_representation_id_t representation) {
  if (representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
    return policy->profile == LOOM_X86_PREDICATE_PROFILE_AVX512;
  }
  uint32_t bit_width = 0;
  if (!loom_x86_predicate_simd_bit_width(lane_count, representation,
                                         &bit_width)) {
    return false;
  }
  if (bit_width == 16 || bit_width == 32 || bit_width == 64 ||
      bit_width == 128 || bit_width == 256) {
    return true;
  }
  return policy->profile == LOOM_X86_PREDICATE_PROFILE_AVX512 &&
         lane_count == 64 && bit_width == 512;
}

loom_low_representation_id_t loom_x86_predicate_callable_representation(
    uint32_t lane_count) {
  const uint32_t bit_width = lane_count == 64   ? 512
                             : lane_count == 32 ? 256
                                                : 128;
  return (loom_low_representation_id_t)(bit_width / lane_count);
}

static bool loom_x86_predicate_simd_register_class(
    uint32_t lane_count, loom_low_representation_id_t representation,
    loom_x86_register_class_t* out_register_class) {
  uint32_t bit_width = 0;
  if (!loom_x86_predicate_simd_bit_width(lane_count, representation,
                                         &bit_width)) {
    return false;
  }
  if (bit_width == 16 || bit_width == 32 || bit_width == 64 ||
      bit_width == 128) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_XMM;
    return true;
  }
  if (bit_width == 256) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_YMM;
    return true;
  }
  if (bit_width == 512) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_ZMM;
    return true;
  }
  return false;
}

bool loom_x86_avx2_predicate_register_class(
    loom_type_t source_type, loom_low_representation_id_t representation,
    loom_x86_register_class_t* out_register_class) {
  uint32_t lane_count = 0;
  if (!loom_x86_avx2_predicate_type(source_type, &lane_count)) {
    return false;
  }
  if (representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    representation = loom_x86_predicate_callable_representation(lane_count);
  }
  const loom_x86_predicate_representation_policy_t policy = {
      .profile = LOOM_X86_PREDICATE_PROFILE_AVX2,
      .maximum_lane_count = 32,
  };
  if (!loom_x86_predicate_representation_available(&policy, lane_count,
                                                   representation)) {
    return false;
  }
  return loom_x86_predicate_simd_register_class(lane_count, representation,
                                                out_register_class);
}

bool loom_x86_avx512_predicate_register_class(
    loom_type_t source_type, loom_low_representation_id_t representation,
    loom_x86_register_class_t* out_register_class) {
  uint32_t lane_count = 0;
  if (!loom_x86_avx512_predicate_type(source_type, &lane_count)) {
    return false;
  }
  if (representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    representation = loom_x86_predicate_callable_representation(lane_count);
  }
  if (representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_K;
    return true;
  }
  const loom_x86_predicate_representation_policy_t policy = {
      .profile = LOOM_X86_PREDICATE_PROFILE_AVX512,
      .maximum_lane_count = 64,
  };
  if (!loom_x86_predicate_representation_available(&policy, lane_count,
                                                   representation)) {
    return false;
  }
  return loom_x86_predicate_simd_register_class(lane_count, representation,
                                                out_register_class);
}

static loom_low_representation_cost_t loom_x86_avx2_predicate_conversion_cost(
    uint32_t lane_count, loom_low_representation_id_t source,
    loom_low_representation_id_t destination) {
  if (source == destination) {
    return (loom_low_representation_cost_t){0};
  }
  uint32_t source_bit_width = 0;
  uint32_t destination_bit_width = 0;
  if (!loom_x86_predicate_simd_bit_width(lane_count, source,
                                         &source_bit_width) ||
      !loom_x86_predicate_simd_bit_width(lane_count, destination,
                                         &destination_bit_width)) {
    return (loom_low_representation_cost_t){.runtime = 1, .code_size = 6};
  }
  if ((uint32_t)source < (uint32_t)destination) {
    return (loom_low_representation_cost_t){.runtime = 1, .code_size = 6};
  }
  uint16_t instruction_count = 1;
  const bool source_has_packed_conversion = source_bit_width == 64 ||
                                            source_bit_width == 128 ||
                                            source_bit_width == 256;
  const bool destination_has_packed_conversion = destination_bit_width == 64 ||
                                                 destination_bit_width == 128 ||
                                                 destination_bit_width == 256;
  if (!source_has_packed_conversion || !destination_has_packed_conversion) {
    instruction_count = source_bit_width <= 128 ? 2 : 5;
  } else if (source_bit_width == 256 && destination_bit_width == 128) {
    instruction_count = 3;
  } else if (source_bit_width == 256 && destination_bit_width == 64) {
    instruction_count = 4;
  }
  return (loom_low_representation_cost_t){
      .runtime = instruction_count,
      .code_size = (uint16_t)(instruction_count * 5u),
  };
}

static loom_low_representation_cost_t loom_x86_predicate_conversion_cost(
    const loom_x86_predicate_representation_policy_t* policy,
    uint32_t lane_count, loom_low_representation_id_t source,
    loom_low_representation_id_t destination) {
  if (source == destination) {
    return (loom_low_representation_cost_t){0};
  }
  if (policy->profile == LOOM_X86_PREDICATE_PROFILE_AVX512 &&
      (source == LOOM_X86_PREDICATE_REPRESENTATION_MASK ||
       destination == LOOM_X86_PREDICATE_REPRESENTATION_MASK)) {
    return (loom_low_representation_cost_t){.runtime = 1, .code_size = 6};
  }
  return loom_x86_avx2_predicate_conversion_cost(lane_count, source,
                                                 destination);
}

static loom_low_representation_id_t loom_x86_predicate_native_representation(
    const loom_x86_predicate_representation_policy_t* policy,
    loom_low_lower_context_t* context, uint32_t lane_count,
    uint32_t vector_bit_width) {
  if (vector_bit_width != 16 && vector_bit_width != 32 &&
      vector_bit_width != 64 && vector_bit_width != 128 &&
      vector_bit_width != 256 && vector_bit_width != 512) {
    return LOOM_LOW_REPRESENTATION_ID_NONE;
  }
  if (policy->profile == LOOM_X86_PREDICATE_PROFILE_AVX512 &&
      vector_bit_width > 64 &&
      (vector_bit_width == 512 ||
       (loom_low_lower_context_bundle(context)->config->contract_feature_bits &
        LOOM_X86_FEATURE_AVX512_VL) != 0)) {
    return LOOM_X86_PREDICATE_REPRESENTATION_MASK;
  }
  return (loom_low_representation_id_t)(vector_bit_width / lane_count);
}

static iree_host_size_t loom_x86_predicate_candidates(
    const loom_x86_predicate_representation_policy_t* policy,
    uint32_t lane_count, loom_low_representation_id_t native_representation,
    loom_low_representation_candidate_t
        candidates[LOOM_X86_PREDICATE_MAX_CANDIDATE_COUNT]) {
  static const loom_low_representation_id_t kRepresentations[] = {
      LOOM_X86_PREDICATE_REPRESENTATION_MASK,
      LOOM_X86_PREDICATE_REPRESENTATION_SIMD8,
      LOOM_X86_PREDICATE_REPRESENTATION_SIMD16,
      LOOM_X86_PREDICATE_REPRESENTATION_SIMD32,
      LOOM_X86_PREDICATE_REPRESENTATION_SIMD64,
  };
  iree_host_size_t candidate_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kRepresentations); ++i) {
    const loom_low_representation_id_t representation = kRepresentations[i];
    if (!loom_x86_predicate_representation_available(policy, lane_count,
                                                     representation)) {
      continue;
    }
    if (policy->profile == LOOM_X86_PREDICATE_PROFILE_AVX512 &&
        native_representation != LOOM_LOW_REPRESENTATION_ID_NONE) {
      const bool native_is_mask =
          native_representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK;
      if (native_is_mask &&
          representation != LOOM_X86_PREDICATE_REPRESENTATION_MASK &&
          representation !=
              loom_x86_predicate_callable_representation(lane_count)) {
        continue;
      }
      if (!native_is_mask &&
          representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
        continue;
      }
    }
    candidates[candidate_count++] = (loom_low_representation_candidate_t){
        .representation = representation,
        .cost = native_representation == LOOM_LOW_REPRESENTATION_ID_NONE
                    ? (loom_low_representation_cost_t){0}
                    : loom_x86_predicate_conversion_cost(policy, lane_count,
                                                         native_representation,
                                                         representation),
    };
  }
  return candidate_count;
}

static void loom_x86_predicate_constrain_flexible(
    const loom_x86_predicate_representation_policy_t* policy,
    const loom_module_t* module, loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  uint32_t lane_count = 0;
  if (!loom_x86_predicate_type(loom_module_value_type(module, value_id),
                               policy->maximum_lane_count, &lane_count)) {
    return;
  }
  loom_low_representation_candidate_t
      candidates[LOOM_X86_PREDICATE_MAX_CANDIDATE_COUNT];
  const iree_host_size_t candidate_count = loom_x86_predicate_candidates(
      policy, lane_count, LOOM_LOW_REPRESENTATION_ID_NONE, candidates);
  loom_low_lower_representation_record_candidates(recorder, value_id,
                                                  candidates, candidate_count);
}

static void loom_x86_predicate_constrain_callable_value(
    const loom_x86_predicate_representation_policy_t* policy,
    const loom_module_t* module, loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  uint32_t lane_count = 0;
  if (!loom_x86_predicate_type(loom_module_value_type(module, value_id),
                               policy->maximum_lane_count, &lane_count)) {
    return;
  }
  const loom_low_representation_candidate_t candidate = {
      .representation = loom_x86_predicate_callable_representation(lane_count),
  };
  loom_low_lower_representation_record_candidates(recorder, value_id,
                                                  &candidate, 1);
}

static void loom_x86_predicate_constrain_callable_values(
    const loom_x86_predicate_representation_policy_t* policy,
    const loom_module_t* module, const loom_value_id_t* value_ids,
    iree_host_size_t value_count,
    loom_low_lower_representation_recorder_t* recorder) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_x86_predicate_constrain_callable_value(policy, module, value_ids[i],
                                                recorder);
  }
}

static bool loom_x86_predicate_representation_relation(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, const loom_value_relation_t* relation,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_x86_predicate_representation_policy_t* policy = user_data;
  (void)source_op;
  (void)recorder;
  if (iree_any_bit_set(relation->flags, LOOM_VALUE_RELATION_FLAG_TYPE_CHANGE)) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, relation->source_value_id);
  return loom_type_equal(
             source_type,
             loom_module_value_type(module, relation->destination_value_id)) &&
         loom_x86_predicate_type(source_type, policy->maximum_lane_count, NULL);
}

static void loom_x86_predicate_observe_compare(
    const loom_x86_predicate_representation_policy_t* policy,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const bool is_integer = loom_vector_cmpi_isa(source_op);
  const loom_value_id_t lhs = is_integer ? loom_vector_cmpi_lhs(source_op)
                                         : loom_vector_cmpf_lhs(source_op);
  const loom_value_id_t result = is_integer
                                     ? loom_vector_cmpi_result(source_op)
                                     : loom_vector_cmpf_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint32_t lane_count = 0;
  if (!loom_x86_predicate_type(loom_module_value_type(module, result),
                               policy->maximum_lane_count, &lane_count)) {
    return;
  }
  const loom_type_t operand_type = loom_module_value_type(module, lhs);
  if (!loom_type_is_vector(operand_type) || loom_type_rank(operand_type) != 1 ||
      !loom_type_is_all_static(operand_type)) {
    return;
  }
  const int32_t element_bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(operand_type));
  if (element_bit_width <= 0) {
    return;
  }
  const uint32_t vector_bit_width = lane_count * (uint32_t)element_bit_width;
  const loom_low_representation_id_t native_representation =
      loom_x86_predicate_native_representation(policy, context, lane_count,
                                               vector_bit_width);
  if (native_representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    return;
  }
  loom_low_representation_candidate_t
      candidates[LOOM_X86_PREDICATE_MAX_CANDIDATE_COUNT];
  const iree_host_size_t candidate_count = loom_x86_predicate_candidates(
      policy, lane_count, native_representation, candidates);
  loom_low_lower_representation_record_candidates(recorder, result, candidates,
                                                  candidate_count);
}

static bool loom_x86_integer_vector_shape(loom_type_t type,
                                          uint32_t expected_lane_count,
                                          uint32_t* out_element_bit_width,
                                          uint32_t* out_vector_bit_width) {
  if (out_element_bit_width != NULL) {
    *out_element_bit_width = 0;
  }
  if (out_vector_bit_width != NULL) {
    *out_vector_bit_width = 0;
  }
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type) ||
      loom_type_dim_static_size_at(type, 0) != expected_lane_count) {
    return false;
  }
  const int32_t element_bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (element_bit_width != 8 && element_bit_width != 16 &&
      element_bit_width != 32 && element_bit_width != 64) {
    return false;
  }
  if (out_element_bit_width != NULL) {
    *out_element_bit_width = (uint32_t)element_bit_width;
  }
  if (out_vector_bit_width != NULL) {
    *out_vector_bit_width = expected_lane_count * (uint32_t)element_bit_width;
  }
  return true;
}

static loom_low_representation_cost_t
loom_x86_predicate_integer_conversion_cost(
    uint32_t lane_count, loom_low_representation_id_t source_element_bit_width,
    uint32_t result_element_bit_width) {
  return loom_x86_avx2_predicate_conversion_cost(
      lane_count, source_element_bit_width,
      (loom_low_representation_id_t)result_element_bit_width);
}

static void loom_x86_predicate_observe_extend(
    const loom_x86_predicate_representation_policy_t* policy,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const bool is_signed = loom_vector_extsi_isa(source_op);
  const loom_value_id_t input = is_signed ? loom_vector_extsi_input(source_op)
                                          : loom_vector_extui_input(source_op);
  const loom_value_id_t result = is_signed
                                     ? loom_vector_extsi_result(source_op)
                                     : loom_vector_extui_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint32_t lane_count = 0;
  if (!loom_x86_predicate_type(loom_module_value_type(module, input),
                               policy->maximum_lane_count, &lane_count)) {
    return;
  }
  uint32_t result_element_bit_width = 0;
  if (!loom_x86_integer_vector_shape(loom_module_value_type(module, result),
                                     lane_count, &result_element_bit_width,
                                     NULL)) {
    return;
  }

  loom_low_representation_candidate_t
      candidates[LOOM_X86_PREDICATE_MAX_CANDIDATE_COUNT];
  const iree_host_size_t candidate_count = loom_x86_predicate_candidates(
      policy, lane_count, LOOM_LOW_REPRESENTATION_ID_NONE, candidates);
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (candidates[i].representation ==
        LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
      candidates[i].cost =
          (loom_low_representation_cost_t){.runtime = 1, .code_size = 6};
    } else {
      candidates[i].cost = loom_x86_predicate_integer_conversion_cost(
          lane_count, candidates[i].representation, result_element_bit_width);
    }
  }
  loom_low_lower_representation_record_costs(recorder, input, candidates,
                                             candidate_count);
}

static void loom_x86_predicate_observe_truncate(
    const loom_x86_predicate_representation_policy_t* policy,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t input = loom_vector_trunci_input(source_op);
  const loom_value_id_t result = loom_vector_trunci_result(source_op);
  uint32_t lane_count = 0;
  if (!loom_x86_predicate_type(loom_module_value_type(module, result),
                               policy->maximum_lane_count, &lane_count)) {
    return;
  }
  uint32_t input_element_bit_width = 0;
  if (!loom_x86_integer_vector_shape(loom_module_value_type(module, input),
                                     lane_count, &input_element_bit_width,
                                     NULL)) {
    return;
  }
  loom_low_representation_candidate_t
      candidates[LOOM_X86_PREDICATE_MAX_CANDIDATE_COUNT];
  const iree_host_size_t candidate_count = loom_x86_predicate_candidates(
      policy, lane_count, LOOM_LOW_REPRESENTATION_ID_NONE, candidates);
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (policy->profile == LOOM_X86_PREDICATE_PROFILE_AVX512) {
      candidates[i].cost =
          candidates[i].representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK
              ? (loom_low_representation_cost_t){0}
              : (loom_low_representation_cost_t){.runtime = 1, .code_size = 6};
    } else {
      candidates[i].cost = loom_x86_predicate_integer_conversion_cost(
          lane_count, (loom_low_representation_id_t)input_element_bit_width,
          (uint32_t)candidates[i].representation);
    }
  }
  loom_low_lower_representation_record_candidates(recorder, result, candidates,
                                                  candidate_count);
}

static void loom_x86_predicate_observe_select(
    const loom_x86_predicate_representation_policy_t* policy,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t condition = loom_vector_select_condition(source_op);
  uint32_t lane_count = 0;
  if (!loom_x86_predicate_type(loom_module_value_type(module, condition),
                               policy->maximum_lane_count, &lane_count)) {
    return;
  }
  const loom_type_t value_type =
      loom_module_value_type(module, loom_vector_select_true_value(source_op));
  if (!loom_type_is_vector(value_type) || loom_type_rank(value_type) != 1 ||
      !loom_type_is_all_static(value_type)) {
    return;
  }
  const int32_t element_bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(value_type));
  if (element_bit_width <= 0) {
    return;
  }
  const uint32_t vector_bit_width = lane_count * (uint32_t)element_bit_width;
  const loom_low_representation_id_t native_representation =
      loom_x86_predicate_native_representation(policy, context, lane_count,
                                               vector_bit_width);
  if (native_representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    return;
  }
  loom_low_representation_candidate_t
      candidates[LOOM_X86_PREDICATE_MAX_CANDIDATE_COUNT];
  const iree_host_size_t candidate_count = loom_x86_predicate_candidates(
      policy, lane_count, native_representation, candidates);
  // These rows are the select's accepted representations as well as their
  // conversion costs. An omitted representation cannot consume this payload.
  loom_low_lower_representation_record_candidates(recorder, condition,
                                                  candidates, candidate_count);
}

static void loom_x86_predicate_observe_lane_access(
    const loom_x86_predicate_representation_policy_t* policy,
    loom_low_lower_context_t* context, loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  uint32_t lane_count = 0;
  if (!loom_x86_predicate_type(loom_module_value_type(module, value_id),
                               policy->maximum_lane_count, &lane_count)) {
    return;
  }
  loom_low_representation_candidate_t
      candidates[LOOM_X86_PREDICATE_MAX_CANDIDATE_COUNT];
  const iree_host_size_t candidate_count = loom_x86_predicate_candidates(
      policy, lane_count, LOOM_LOW_REPRESENTATION_ID_NONE, candidates);
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    if (candidates[i].representation ==
        LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
      candidates[i].cost =
          lane_count == 64
              ? (loom_low_representation_cost_t){0}
              : (loom_low_representation_cost_t){.runtime = 3, .code_size = 12};
      continue;
    }
    uint32_t bit_width = 0;
    loom_x86_predicate_simd_bit_width(lane_count, candidates[i].representation,
                                      &bit_width);
    if (bit_width >= 256) {
      candidates[i].cost =
          (loom_low_representation_cost_t){.runtime = 1, .code_size = 6};
    }
  }
  loom_low_lower_representation_record_costs(recorder, value_id, candidates,
                                             candidate_count);
}

static void loom_x86_predicate_observe_insert(
    const loom_x86_predicate_representation_policy_t* policy,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t destination = loom_vector_insert_dest(source_op);
  const loom_value_id_t result = loom_vector_insert_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_x86_predicate_type(loom_module_value_type(module, destination),
                               policy->maximum_lane_count, NULL)) {
    return;
  }
  loom_low_lower_representation_record_union(recorder, destination, result);
  loom_x86_predicate_observe_lane_access(policy, context, destination,
                                         recorder);
}

static void loom_x86_predicate_observe_slice(
    const loom_x86_predicate_representation_policy_t* policy,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source = loom_vector_slice_source(source_op);
  const loom_value_id_t result = loom_vector_slice_result(source_op);
  loom_x86_predicate_constrain_flexible(policy, module, source, recorder);
  loom_x86_predicate_constrain_flexible(policy, module, result, recorder);
}

static void loom_x86_predicate_observe_boundary(
    void* user_data, uint8_t action,
    loom_low_lower_representation_boundary_flags_t flags,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_x86_predicate_representation_policy_t* policy = user_data;
  (void)flags;
  switch ((loom_x86_predicate_representation_action_t)action) {
    case LOOM_X86_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT:
      loom_x86_predicate_constrain_flexible(
          policy, loom_low_lower_context_module(context),
          loom_op_const_results(source_op)[0], recorder);
      return;
    case LOOM_X86_PREDICATE_REPRESENTATION_ACTION_EXTRACT:
      loom_x86_predicate_observe_lane_access(
          policy, context, loom_vector_extract_source(source_op), recorder);
      loom_low_lower_representation_claim_operand(recorder, 0);
      return;
    case LOOM_X86_PREDICATE_REPRESENTATION_ACTION_INSERT:
      loom_x86_predicate_observe_insert(policy, context, source_op, recorder);
      loom_low_lower_representation_claim_operand(recorder, 0);
      return;
    case LOOM_X86_PREDICATE_REPRESENTATION_ACTION_SLICE:
      loom_x86_predicate_observe_slice(policy, context, source_op, recorder);
      loom_low_lower_representation_claim_operand(recorder, 0);
      return;
    case LOOM_X86_PREDICATE_REPRESENTATION_ACTION_SELECT:
      loom_x86_predicate_observe_select(policy, context, source_op, recorder);
      loom_low_lower_representation_claim_operand(recorder, 0);
      return;
    case LOOM_X86_PREDICATE_REPRESENTATION_ACTION_COMPARE:
      loom_x86_predicate_observe_compare(policy, context, source_op, recorder);
      return;
    case LOOM_X86_PREDICATE_REPRESENTATION_ACTION_EXTEND_INPUT:
      loom_x86_predicate_observe_extend(policy, context, source_op, recorder);
      loom_low_lower_representation_claim_operand(recorder, 0);
      return;
    case LOOM_X86_PREDICATE_REPRESENTATION_ACTION_TRUNCATE_RESULT:
      loom_x86_predicate_observe_truncate(policy, context, source_op, recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown x86 predicate representation action");
}

static void loom_x86_predicate_observe_callable_boundary(
    void* user_data,
    loom_low_lower_representation_callable_boundary_kind_t kind,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_x86_predicate_representation_policy_t* policy = user_data;
  const loom_module_t* module = loom_low_lower_context_module(context);
  switch (kind) {
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_DEFINITION: {
      const loom_func_like_t function =
          loom_func_like_const_cast(module, source_op);
      uint16_t argument_count = 0;
      const loom_value_id_t* argument_ids =
          loom_func_like_arg_ids(function, &argument_count);
      loom_x86_predicate_constrain_callable_values(policy, module, argument_ids,
                                                   argument_count, recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_CALL: {
      const loom_call_like_t call =
          loom_call_like_const_cast(module, source_op);
      const loom_value_slice_t operands = loom_call_like_operands(call);
      const loom_value_slice_t results = loom_call_like_results(call);
      loom_x86_predicate_constrain_callable_values(
          policy, module, operands.values, operands.count, recorder);
      loom_x86_predicate_constrain_callable_values(
          policy, module, results.values, results.count, recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_EXIT:
      loom_x86_predicate_constrain_callable_values(
          policy, module, loom_op_const_operands(source_op),
          source_op->operand_count, recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown callable representation boundary kind");
}

static void loom_x86_predicate_observe_unclaimed_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, uint16_t source_operand_index,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_x86_predicate_representation_policy_t* policy = user_data;
  loom_x86_predicate_constrain_callable_value(
      policy, loom_low_lower_context_module(context),
      loom_op_const_operands(source_op)[source_operand_index], recorder);
}

static const loom_low_lower_representation_boundary_t
    kX86PredicateRepresentationBoundaries[] = {
        {LOOM_OP_VECTOR_SPLATI,
         LOOM_X86_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_SPLAT,
         LOOM_X86_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_FROM_ELEMENTS,
         LOOM_X86_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_EXTRACT,
         LOOM_X86_PREDICATE_REPRESENTATION_ACTION_EXTRACT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_INSERT, LOOM_X86_PREDICATE_REPRESENTATION_ACTION_INSERT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS |
             LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_SLICE, LOOM_X86_PREDICATE_REPRESENTATION_ACTION_SLICE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS |
             LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_SELECT, LOOM_X86_PREDICATE_REPRESENTATION_ACTION_SELECT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_CMPI, LOOM_X86_PREDICATE_REPRESENTATION_ACTION_COMPARE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_CMPF, LOOM_X86_PREDICATE_REPRESENTATION_ACTION_COMPARE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_EXTSI,
         LOOM_X86_PREDICATE_REPRESENTATION_ACTION_EXTEND_INPUT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_EXTUI,
         LOOM_X86_PREDICATE_REPRESENTATION_ACTION_EXTEND_INPUT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_TRUNCI,
         LOOM_X86_PREDICATE_REPRESENTATION_ACTION_TRUNCATE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
};
static const loom_low_lower_representation_boundary_span_t
    kX86PredicateRepresentationBoundarySpans[] = {
        {0, IREE_ARRAYSIZE(kX86PredicateRepresentationBoundaries)},
};
static_assert((loom_op_kind_t)LOOM_OP_VECTOR_SPLATI <
                      (loom_op_kind_t)LOOM_OP_VECTOR_SPLAT &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_SPLAT <
                      (loom_op_kind_t)LOOM_OP_VECTOR_FROM_ELEMENTS &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_FROM_ELEMENTS <
                      (loom_op_kind_t)LOOM_OP_VECTOR_EXTRACT &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_EXTRACT <
                      (loom_op_kind_t)LOOM_OP_VECTOR_INSERT &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_INSERT <
                      (loom_op_kind_t)LOOM_OP_VECTOR_SLICE &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_SLICE <
                      (loom_op_kind_t)LOOM_OP_VECTOR_SELECT &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_SELECT <
                      (loom_op_kind_t)LOOM_OP_VECTOR_CMPI &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_CMPI <
                      (loom_op_kind_t)LOOM_OP_VECTOR_CMPF &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_CMPF <
                      (loom_op_kind_t)LOOM_OP_VECTOR_EXTSI &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_EXTSI <
                      (loom_op_kind_t)LOOM_OP_VECTOR_EXTUI &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_EXTUI <
                      (loom_op_kind_t)LOOM_OP_VECTOR_TRUNCI,
              "x86 predicate representation boundaries must remain ordered");

static const loom_x86_predicate_representation_policy_t
    kX86Avx2PredicateRepresentationPolicy = {
        .profile = LOOM_X86_PREDICATE_PROFILE_AVX2,
        .maximum_lane_count = 32,
};

static const loom_x86_predicate_representation_policy_t
    kX86Avx512PredicateRepresentationPolicy = {
        .profile = LOOM_X86_PREDICATE_PROFILE_AVX512,
        .maximum_lane_count = 64,
};

static const loom_low_lower_representation_provider_t
    kX86Avx2PredicateRepresentationProvider = {
        .relation = loom_x86_predicate_representation_relation,
        .observe_boundary = loom_x86_predicate_observe_boundary,
        .observe_callable_boundary =
            loom_x86_predicate_observe_callable_boundary,
        .observe_unclaimed_operand =
            loom_x86_predicate_observe_unclaimed_operand,
        .boundaries = kX86PredicateRepresentationBoundaries,
        .boundary_spans = kX86PredicateRepresentationBoundarySpans,
        .boundary_count = IREE_ARRAYSIZE(kX86PredicateRepresentationBoundaries),
        .boundary_dialect_base_id = LOOM_DIALECT_VECTOR,
        .boundary_dialect_count =
            IREE_ARRAYSIZE(kX86PredicateRepresentationBoundarySpans),
        .relation_mask = LOOM_VALUE_RELATION_MASK_ALL,
        .user_data = (void*)&kX86Avx2PredicateRepresentationPolicy,
};

static const loom_low_lower_representation_provider_t
    kX86Avx512PredicateRepresentationProvider = {
        .relation = loom_x86_predicate_representation_relation,
        .observe_boundary = loom_x86_predicate_observe_boundary,
        .observe_callable_boundary =
            loom_x86_predicate_observe_callable_boundary,
        .observe_unclaimed_operand =
            loom_x86_predicate_observe_unclaimed_operand,
        .boundaries = kX86PredicateRepresentationBoundaries,
        .boundary_spans = kX86PredicateRepresentationBoundarySpans,
        .boundary_count = IREE_ARRAYSIZE(kX86PredicateRepresentationBoundaries),
        .boundary_dialect_base_id = LOOM_DIALECT_VECTOR,
        .boundary_dialect_count =
            IREE_ARRAYSIZE(kX86PredicateRepresentationBoundarySpans),
        .relation_mask = LOOM_VALUE_RELATION_MASK_ALL,
        .user_data = (void*)&kX86Avx512PredicateRepresentationPolicy,
};

const loom_low_lower_source_plan_observer_t
    loom_x86_avx2_predicate_representation_observer = {
        .begin = loom_low_lower_representation_observer_begin,
        .observe = loom_low_lower_representation_observer_observe,
        .end = loom_low_lower_representation_observer_end,
        .user_data = (void*)&kX86Avx2PredicateRepresentationProvider,
};

const loom_low_lower_source_plan_observer_t
    loom_x86_avx512_predicate_representation_observer = {
        .begin = loom_low_lower_representation_observer_begin,
        .observe = loom_low_lower_representation_observer_observe,
        .end = loom_low_lower_representation_observer_end,
        .user_data = (void*)&kX86Avx512PredicateRepresentationProvider,
};
