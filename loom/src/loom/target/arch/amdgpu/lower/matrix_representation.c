// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_representation.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/contract_vector.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/matrix.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_plan.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_state.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"

typedef enum loom_amdgpu_matrix_representation_action_e {
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FUNCTION_BOUNDARY = 0,
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_PIN_VALUE = 1,
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_STORE = 2,
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_MMA = 3,
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FRAGMENT = 4,
} loom_amdgpu_matrix_representation_action_t;

IREE_ATTRIBUTE_NOINLINE static bool
loom_amdgpu_matrix_representation_accumulator_fact(
    loom_low_lower_context_t* context, loom_value_id_t value_id,
    loom_vector_fragment_fact_t* out_fragment) {
  loom_vector_fragment_fact_initialize(out_fragment);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  return fact_table != NULL &&
         loom_vector_fragment_fact_query_value_facts(
             &fact_table->context,
             loom_value_fact_table_lookup(fact_table, value_id),
             out_fragment) &&
         loom_vector_fragment_fact_is_accumulator_like(*out_fragment);
}

static bool loom_amdgpu_matrix_representation_matches_fragment(
    const loom_value_fact_table_t* fact_table, loom_type_t payload_type,
    loom_vector_fragment_fact_t fragment,
    loom_amdgpu_matrix_result_representation_id_t representation_id) {
  const loom_amdgpu_matrix_result_representation_t* representation =
      loom_amdgpu_matrix_result_representation_at(representation_id);
  if (representation == NULL || fact_table == NULL) return false;
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(
          (loom_amdgpu_matrix_fragment_layout_kind_t)
              representation->fragment_layout_kind);
  if (layout == NULL) return false;
  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout,
                                       LOOM_CONTRACT_OPERAND_ROLE_RESULT);
  loom_scalar_type_t element_type = LOOM_SCALAR_TYPE_NONE;
  if (role_layout == NULL ||
      !loom_amdgpu_matrix_fragment_scalar_type_from_numeric(
          (loom_amdgpu_matrix_numeric_type_t)representation->numeric_type,
          &element_type) ||
      !loom_amdgpu_matrix_fragment_payload_matches_role_storage(
          payload_type, element_type, role_layout)) {
    return false;
  }
  return loom_amdgpu_matrix_fragment_tile_shape_matches(
      fact_table,
      loom_amdgpu_matrix_fragment_source_tile_shape(
          layout, LOOM_CONTRACT_OPERAND_ROLE_RESULT, representation->flags),
      LOOM_CONTRACT_OPERAND_ROLE_RESULT,
      loom_vector_fragment_fact_block_value(fragment),
      loom_vector_fragment_fact_row_value(fragment),
      loom_vector_fragment_fact_column_value(fragment));
}

static void loom_amdgpu_matrix_representation_record_available(
    loom_low_lower_context_t* context, loom_value_id_t value_id,
    loom_vector_fragment_fact_t fragment, uint64_t available_bits,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_low_representation_candidate_t
      candidates[LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_MAX_ID];
  iree_host_size_t candidate_count = 0;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_type_t payload_type =
      loom_module_value_type(loom_low_lower_context_module(context), value_id);
  available_bits &= ~UINT64_C(1);
  while (available_bits != 0) {
    const loom_amdgpu_matrix_result_representation_id_t representation_id =
        (loom_amdgpu_matrix_result_representation_id_t)
            iree_math_count_trailing_zeros_u64(available_bits);
    available_bits &= available_bits - 1u;
    if (!loom_amdgpu_matrix_representation_matches_fragment(
            fact_table, payload_type, fragment, representation_id)) {
      continue;
    }
    candidates[candidate_count++] = (loom_low_representation_candidate_t){
        .representation = representation_id,
    };
  }
  if (candidate_count != 0) {
    loom_low_lower_representation_record_candidates(
        recorder, value_id, candidates, candidate_count);
  }
}

static void loom_amdgpu_matrix_representation_constrain_canonical(
    loom_low_lower_context_t* context, loom_value_id_t value_id,
    loom_vector_fragment_fact_t fragment,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_amdgpu_matrix_fragment_contract_candidates_t* contracts = NULL;
  iree_status_t status =
      loom_amdgpu_matrix_fragment_contract_candidates(context, &contracts);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
    return;
  }
  if (contracts != NULL) {
    loom_amdgpu_matrix_representation_record_available(
        context, value_id, fragment,
        contracts->canonical_result_representation_bits, recorder);
  }
}

static void loom_amdgpu_matrix_representation_pin_value(
    loom_low_lower_context_t* context, loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_vector_fragment_fact_t fragment;
  if (loom_amdgpu_matrix_representation_accumulator_fact(context, value_id,
                                                         &fragment)) {
    loom_amdgpu_matrix_representation_constrain_canonical(context, value_id,
                                                          fragment, recorder);
  }
}

IREE_ATTRIBUTE_NOINLINE static void
loom_amdgpu_matrix_representation_pin_values(
    loom_low_lower_context_t* context, const loom_value_id_t* value_ids,
    iree_host_size_t value_count,
    loom_low_lower_representation_recorder_t* recorder) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_amdgpu_matrix_representation_pin_value(context, value_ids[i],
                                                recorder);
  }
}

static bool loom_amdgpu_matrix_representation_relation(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, const loom_value_relation_t* relation) {
  (void)user_data;
  // Matrix instructions define their result representation together with the
  // selected descriptor. Their tied accumulator/result relation is only exact
  // when the matrix boundary below admits a generated realization.
  if (source_op->kind == LOOM_OP_VECTOR_MMA &&
      relation->source_value_id == loom_vector_mma_init(source_op) &&
      relation->destination_value_id == loom_vector_mma_result(source_op)) {
    return false;
  }
  if (iree_any_bit_set(relation->flags, LOOM_VALUE_RELATION_FLAG_TYPE_CHANGE)) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_type_equal(
          loom_module_value_type(module, relation->source_value_id),
          loom_module_value_type(module, relation->destination_value_id))) {
    return false;
  }
  switch ((loom_value_relation_kind_t)relation->kind) {
    case LOOM_VALUE_RELATION_CFG_ARGUMENT:
    case LOOM_VALUE_RELATION_LOOP_CARRIED:
    case LOOM_VALUE_RELATION_LOOP_BYPASS:
    case LOOM_VALUE_RELATION_REGION_RESULT:
      // Control-flow transport may temporarily lose fragment facts at raw
      // carrier values. Equal source types preserve the representation; an
      // unrelated carrier remains absent from the sparse plan until a matrix
      // boundary constrains its component.
      return true;
    default:
      break;
  }
  loom_vector_fragment_fact_t source_fragment;
  loom_vector_fragment_fact_t destination_fragment;
  return loom_amdgpu_matrix_representation_accumulator_fact(
             context, relation->source_value_id, &source_fragment) &&
         loom_amdgpu_matrix_representation_accumulator_fact(
             context, relation->destination_value_id, &destination_fragment) &&
         loom_vector_fragment_facts_match_accumulator_contract(
             source_fragment, destination_fragment);
}

static void loom_amdgpu_matrix_representation_observe_mma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_amdgpu_matrix_target_configuration_t configuration = {0};
  loom_amdgpu_matrix_target_configuration_initialize(
      loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context)),
      loom_low_lower_context_bundle(context), &configuration);
  loom_contract_request_t request = {0};
  loom_contract_diagnostic_t contract_diagnostic = {0};
  if (!loom_contract_request_from_vector_mma_op(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op,
          &configuration.options, &request, &contract_diagnostic)) {
    return;
  }
  const loom_amdgpu_matrix_contract_descriptor_t* descriptor = NULL;
  uint16_t descriptor_ordinal = LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE;
  loom_amdgpu_matrix_contract_match_diagnostic_t match_diagnostic = {0};
  if (!loom_amdgpu_matrix_select_contract(
          &request, &configuration, &descriptor, &descriptor_ordinal,
          &contract_diagnostic, &match_diagnostic)) {
    return;
  }
  const loom_amdgpu_matrix_contract_realization_choices_t* choices =
      &descriptor->realization;
  if (choices->canonical_result_representation_id ==
      LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE) {
    return;
  }

  const loom_amdgpu_matrix_fragment_contract_candidates_t* contracts = NULL;
  iree_status_t status =
      loom_amdgpu_matrix_fragment_contract_candidates(context, &contracts);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
    return;
  }
  loom_low_representation_candidate_t candidates[2] = {
      {
          .representation = choices->canonical_result_representation_id,
      },
  };
  iree_host_size_t candidate_count = 1;
  const loom_amdgpu_matrix_result_representation_id_t alternative =
      choices->operand_exchanged_result_representation_id;
  if (alternative != LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE &&
      contracts != NULL &&
      (contracts->exact_result_representation_bits &
       (UINT64_C(1) << alternative)) != 0) {
    candidates[candidate_count++].representation = alternative;
  }
  const loom_value_id_t result = loom_vector_mma_result(source_op);
  loom_low_lower_representation_record_candidates(recorder, result, candidates,
                                                  candidate_count);
  loom_low_lower_representation_record_union(
      recorder, loom_vector_mma_init(source_op), result);
  status = loom_amdgpu_matrix_fragment_record_contract_ordinal(
      context, result, descriptor_ordinal);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
  }
}

static void loom_amdgpu_matrix_representation_observe_fragment(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t data = loom_vector_fragment_data(source_op);
  const loom_value_id_t result = loom_vector_fragment_result(source_op);
  loom_vector_fragment_fact_t data_fragment;
  loom_vector_fragment_fact_t result_fragment;
  if (!loom_amdgpu_matrix_representation_accumulator_fact(context, result,
                                                          &result_fragment)) {
    return;
  }
  if (loom_amdgpu_matrix_representation_accumulator_fact(context, data,
                                                         &data_fragment) &&
      loom_vector_fragment_facts_match_accumulator_contract(data_fragment,
                                                            result_fragment)) {
    return;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  if (fact_table != NULL &&
      loom_value_facts_query_all_equal_element(
          &fact_table->context, loom_value_fact_table_lookup(fact_table, data),
          &element_facts)) {
    return;
  }
  if (loom_low_lower_representation_component_is_constrained(recorder, data)) {
    // Raw control-flow carriers may lose fragment facts at a meet while still
    // belonging to a component constrained by matrix operations. Preserve
    // that established identity through the fragment value alias.
    loom_low_lower_representation_record_union(recorder, data, result);
    return;
  }
  loom_amdgpu_matrix_representation_pin_value(context, result, recorder);
}

static void loom_amdgpu_matrix_representation_observe_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t payload = loom_vector_fragment_store_value(source_op);
  loom_vector_fragment_fact_t fragment;
  if (!loom_amdgpu_matrix_representation_accumulator_fact(context, payload,
                                                          &fragment)) {
    return;
  }
  if (loom_vector_fragment_store_role(source_op) != LOOM_VECTOR_ROLE_RESULT) {
    loom_amdgpu_matrix_representation_constrain_canonical(context, payload,
                                                          fragment, recorder);
    return;
  }
  loom_low_representation_candidate_t
      candidates[LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_MAX_ID];
  iree_host_size_t candidate_count = 0;
  iree_status_t status =
      loom_amdgpu_query_accumulator_fragment_store_representations(
          context, source_op, candidates, &candidate_count);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
    return;
  }
  if (candidate_count != 0) {
    loom_low_lower_representation_record_candidates(
        recorder, payload, candidates, candidate_count);
  } else {
    loom_amdgpu_matrix_representation_constrain_canonical(context, payload,
                                                          fragment, recorder);
  }
}

static void loom_amdgpu_matrix_representation_observe_boundary(
    void* user_data, uint8_t action,
    loom_low_lower_representation_boundary_flags_t flags,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  switch ((loom_amdgpu_matrix_representation_action_t)action) {
    case LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FUNCTION_BOUNDARY: {
      loom_func_like_t function = loom_func_like_const_cast(
          loom_low_lower_context_module(context), source_op);
      uint16_t argument_count = 0;
      const loom_value_id_t* argument_ids =
          loom_func_like_arg_ids(function, &argument_count);
      loom_amdgpu_matrix_representation_pin_values(context, argument_ids,
                                                   argument_count, recorder);
      return;
    }
    case LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_PIN_VALUE:
      if (iree_any_bit_set(
              flags, LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS)) {
        loom_amdgpu_matrix_representation_pin_values(
            context, loom_op_operands(source_op), source_op->operand_count,
            recorder);
      }
      if (iree_any_bit_set(
              flags, LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS)) {
        loom_amdgpu_matrix_representation_pin_values(
            context, loom_op_results(source_op), source_op->result_count,
            recorder);
      }
      return;
    case LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_STORE:
      loom_amdgpu_matrix_representation_observe_store(context, source_op,
                                                      recorder);
      return;
    case LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_MMA:
      loom_amdgpu_matrix_representation_observe_mma(context, source_op,
                                                    recorder);
      return;
    case LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FRAGMENT:
      loom_amdgpu_matrix_representation_observe_fragment(context, source_op,
                                                         recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU matrix representation action");
}

static const loom_low_lower_representation_boundary_t
    kAmdgpuMatrixRepresentationBoundaries[] = {
        {LOOM_OP_FUNC_DEF,
         LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FUNCTION_BOUNDARY,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_NONE},
        {LOOM_OP_FUNC_CALL, LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_PIN_VALUE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS |
             LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_FUNC_RETURN,
         LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_PIN_VALUE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_FRAGMENT_LOAD,
         LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_PIN_VALUE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_FRAGMENT_STORE,
         LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_STORE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_NONE},
        {LOOM_OP_VECTOR_MMA, LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_MMA,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_NONE},
        {LOOM_OP_VECTOR_FRAGMENT,
         LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FRAGMENT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_NONE},
        {LOOM_OP_VECTOR_FRAGMENT_REPACK,
         LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_PIN_VALUE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS |
             LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_KERNEL_DEF,
         LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FUNCTION_BOUNDARY,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_NONE},
};
static_assert((loom_op_kind_t)LOOM_OP_FUNC_DEF <
                      (loom_op_kind_t)LOOM_OP_FUNC_CALL &&
                  (loom_op_kind_t)LOOM_OP_FUNC_CALL <
                      (loom_op_kind_t)LOOM_OP_FUNC_RETURN &&
                  (loom_op_kind_t)LOOM_OP_FUNC_RETURN <
                      (loom_op_kind_t)LOOM_OP_VECTOR_FRAGMENT_LOAD &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_FRAGMENT_LOAD <
                      (loom_op_kind_t)LOOM_OP_VECTOR_FRAGMENT_STORE &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_FRAGMENT_STORE <
                      (loom_op_kind_t)LOOM_OP_VECTOR_MMA &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_MMA <
                      (loom_op_kind_t)LOOM_OP_VECTOR_FRAGMENT &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_FRAGMENT <
                      (loom_op_kind_t)LOOM_OP_VECTOR_FRAGMENT_REPACK &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_FRAGMENT_REPACK <
                      (loom_op_kind_t)LOOM_OP_KERNEL_DEF,
              "matrix representation boundaries must remain ordered");

static const loom_low_lower_representation_provider_t
    kAmdgpuMatrixRepresentationProvider = {
        .relation = loom_amdgpu_matrix_representation_relation,
        .observe_boundary = loom_amdgpu_matrix_representation_observe_boundary,
        .boundaries = kAmdgpuMatrixRepresentationBoundaries,
        .boundary_count = IREE_ARRAYSIZE(kAmdgpuMatrixRepresentationBoundaries),
        .relation_mask = LOOM_VALUE_RELATION_MASK_ALL,
};

const loom_low_lower_source_plan_observer_t
    loom_amdgpu_matrix_representation_observer = {
        .begin = loom_low_lower_representation_observer_begin,
        .observe = loom_low_lower_representation_observer_observe,
        .end = loom_low_lower_representation_observer_end,
        .user_data = (void*)&kAmdgpuMatrixRepresentationProvider,
};
