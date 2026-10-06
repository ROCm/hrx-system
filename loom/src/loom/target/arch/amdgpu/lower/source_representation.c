// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/source_representation.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/contract_vector.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/plan.h"
#include "loom/target/arch/amdgpu/lower/matrix.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_state.h"
#include "loom/target/arch/amdgpu/lower/source_integer_representation.h"
#include "loom/target/arch/amdgpu/lower/source_value_analysis.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"

static_assert(LOOM_AMDGPU_ADDRESS_REPRESENTATION_NARROW >
                  LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_MAX_ID,
              "AMDGPU representation namespaces must not overlap");
static_assert(LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS >
                  LOOM_AMDGPU_ADDRESS_REPRESENTATION_WIDE_VGPR,
              "AMDGPU representation namespaces must not overlap");

typedef enum loom_amdgpu_matrix_representation_action_e {
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_PIN_VALUE = 0,
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_STORE = 1,
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_MMA = 2,
  LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FRAGMENT = 3,
} loom_amdgpu_matrix_representation_action_t;
static_assert(
    (uint8_t)LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT >
        (uint8_t)LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FRAGMENT,
    "AMDGPU representation action namespaces must not overlap");

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
  if (representation == NULL || fact_table == NULL) {
    return false;
  }
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(
          (loom_amdgpu_matrix_fragment_layout_kind_t)
              representation->fragment_layout_kind);
  if (layout == NULL) {
    return false;
  }
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

static bool loom_amdgpu_address_representation_requires_wide_vgpr(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (loom_amdgpu_source_value_facts_prefer_vgpr(
          module, loom_low_lower_context_fact_table(context), value_id)) {
    return true;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op != NULL &&
         iree_any_bit_set(loom_amdgpu_source_producer_flags(defining_op->kind),
                          LOOM_AMDGPU_SOURCE_PRODUCER_ADDRESS_64BIT);
}

static void loom_amdgpu_address_representation_constrain_value(
    loom_low_lower_context_t* context, loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_amdgpu_type_is_address_scalar(type) ||
      loom_low_lower_representation_component_is_constrained(recorder,
                                                             value_id)) {
    return;
  }

  const bool needs_wide = loom_amdgpu_source_address_value_needs_64bit(
      module, loom_low_lower_context_fact_table(context), value_id, type);
  const bool wide_requires_vgpr =
      loom_amdgpu_address_representation_requires_wide_vgpr(context, value_id);
  loom_low_representation_candidate_t candidates[3];
  iree_host_size_t candidate_count = 0;
  if (!needs_wide) {
    candidates[candidate_count++] = (loom_low_representation_candidate_t){
        .representation = LOOM_AMDGPU_ADDRESS_REPRESENTATION_NARROW,
    };
  }
  if (!wide_requires_vgpr) {
    candidates[candidate_count++] = (loom_low_representation_candidate_t){
        .representation = LOOM_AMDGPU_ADDRESS_REPRESENTATION_WIDE_SGPR,
        .cost = {.runtime = needs_wide ? 0u : 1u},
    };
  }
  candidates[candidate_count++] = (loom_low_representation_candidate_t){
      .representation = LOOM_AMDGPU_ADDRESS_REPRESENTATION_WIDE_VGPR,
      .cost = {.runtime = wide_requires_vgpr ? 0u : (needs_wide ? 1u : 2u)},
  };
  loom_low_lower_representation_record_candidates(recorder, value_id,
                                                  candidates, candidate_count);
}

static void loom_amdgpu_address_representation_observe_fixed_cast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source = loom_index_cast_input(source_op);
  const loom_value_id_t result = loom_index_cast_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const bool source_is_address =
      loom_amdgpu_type_is_address_scalar(source_type);
  const bool result_is_address =
      loom_amdgpu_type_is_address_scalar(result_type);
  if (source_is_address == result_is_address) {
    return;
  }

  const loom_value_id_t fixed_value = source_is_address ? result : source;
  const loom_type_t fixed_type = source_is_address ? result_type : source_type;
  const loom_scalar_type_t fixed_scalar_type =
      loom_type_element_type(fixed_type);
  if (!loom_scalar_type_is_integer(fixed_scalar_type) ||
      fixed_scalar_type == LOOM_SCALAR_TYPE_I1) {
    return;
  }

  bool fixed_prefers_vgpr = false;
  iree_status_t status = loom_amdgpu_context_value_prefers_vgpr(
      context, fixed_value, &fixed_prefers_vgpr);
  if (!iree_status_is_ok(status)) {
    loom_low_lower_representation_record_failure(recorder, status);
    return;
  }

  const uint32_t fixed_unit_count =
      ((uint32_t)loom_scalar_type_bitwidth(fixed_scalar_type) + 31u) / 32u;
  // The cast materializes one move for each word that crosses register banks.
  // Charge that cost to the address component so exact address relations can
  // retain the fixed side's bank across a cast cycle.
  loom_low_representation_candidate_t bank_crossing_costs[2];
  iree_host_size_t cost_count = 0;
  if (fixed_prefers_vgpr) {
    bank_crossing_costs[cost_count++] = (loom_low_representation_candidate_t){
        .representation = LOOM_AMDGPU_ADDRESS_REPRESENTATION_NARROW,
        .cost = {.runtime = 1, .code_size = 1},
    };
    bank_crossing_costs[cost_count++] = (loom_low_representation_candidate_t){
        .representation = LOOM_AMDGPU_ADDRESS_REPRESENTATION_WIDE_SGPR,
        .cost = {.runtime = fixed_unit_count, .code_size = fixed_unit_count},
    };
  } else {
    bank_crossing_costs[cost_count++] = (loom_low_representation_candidate_t){
        .representation = LOOM_AMDGPU_ADDRESS_REPRESENTATION_WIDE_VGPR,
        .cost = {.runtime = fixed_unit_count, .code_size = fixed_unit_count},
    };
  }
  const loom_value_id_t address_value = source_is_address ? source : result;
  loom_low_lower_representation_record_costs(recorder, address_value,
                                             bank_crossing_costs, cost_count);
}

static bool loom_amdgpu_address_representation_relation_is_exact(
    loom_value_relation_kind_t kind) {
  switch (kind) {
    case LOOM_VALUE_RELATION_TIED_RESULT:
    case LOOM_VALUE_RELATION_VALUE_ALIAS:
    case LOOM_VALUE_RELATION_CFG_ARGUMENT:
    case LOOM_VALUE_RELATION_LOOP_CARRIED:
    case LOOM_VALUE_RELATION_LOOP_BYPASS:
    case LOOM_VALUE_RELATION_REGION_RESULT:
      return true;
    case LOOM_VALUE_RELATION_UNKNOWN:
      return false;
    case LOOM_VALUE_RELATION_FACT_IDENTITY:
      // Fact identity is a refinement boundary, not physical transport.
      // index.assume may intentionally narrow its result while retaining a
      // wide source value for other uses.
      return false;
    case LOOM_VALUE_RELATION_SELECT_PAYLOAD:
      // Select lowering materializes each payload into the result carrier.
      // A shared payload may therefore feed independently placed selects.
      return false;
    case LOOM_VALUE_RELATION_ELEMENTWISE:
    case LOOM_VALUE_RELATION_COUNT_:
      return false;
  }
  return false;
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

static bool loom_amdgpu_source_representation_relation(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, const loom_value_relation_t* relation,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  // Matrix instructions define their result representation together with the
  // selected descriptor. Their tied accumulator/result relation is only exact
  // when the matrix boundary below admits a generated realization.
  if (source_op->kind == LOOM_OP_VECTOR_MMA &&
      relation->kind == LOOM_VALUE_RELATION_TIED_RESULT) {
    return false;
  }
  if (iree_any_bit_set(relation->flags, LOOM_VALUE_RELATION_FLAG_TYPE_CHANGE)) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, relation->source_value_id);
  if (!loom_type_equal(
          source_type,
          loom_module_value_type(module, relation->destination_value_id))) {
    return false;
  }
  if (loom_amdgpu_type_is_address_scalar(source_type) &&
      loom_amdgpu_address_representation_relation_is_exact(relation->kind)) {
    loom_amdgpu_address_representation_constrain_value(
        context, relation->source_value_id, recorder);
    loom_amdgpu_address_representation_constrain_value(
        context, relation->destination_value_id, recorder);
    return true;
  }
  switch ((loom_value_relation_kind_t)relation->kind) {
    case LOOM_VALUE_RELATION_FACT_IDENTITY:
    case LOOM_VALUE_RELATION_VALUE_ALIAS:
    case LOOM_VALUE_RELATION_SELECT_PAYLOAD:
      if (loom_amdgpu_source_integer_representation_type_is_narrow(
              source_type)) {
        return true;
      }
      break;
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

static void loom_amdgpu_source_representation_observe_unclaimed_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, uint16_t source_operand_index,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  const loom_value_id_t value_id =
      loom_op_const_operands(source_op)[source_operand_index];
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), value_id);
  if (loom_type_is_vector(type)) {
    loom_amdgpu_matrix_representation_pin_value(context, value_id, recorder);
  }
}

static void loom_amdgpu_source_representation_observe_boundary(
    void* user_data, uint8_t action,
    loom_low_lower_representation_boundary_flags_t flags,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  if (source_op->kind == LOOM_OP_INDEX_CAST) {
    loom_amdgpu_address_representation_observe_fixed_cast(context, source_op,
                                                          recorder);
  }
  if (action >=
      LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT) {
    loom_amdgpu_source_integer_representation_observe_boundary(
        (loom_amdgpu_source_integer_representation_action_t)action, context,
        source_op, recorder);
    return;
  }
  switch ((loom_amdgpu_matrix_representation_action_t)action) {
    case LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_PIN_VALUE:
      if (iree_any_bit_set(
              flags, LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS)) {
        loom_amdgpu_matrix_representation_pin_values(
            context, loom_op_operands(source_op), source_op->operand_count,
            recorder);
        for (uint16_t i = 0; i < source_op->operand_count; ++i) {
          loom_low_lower_representation_claim_operand(recorder, i);
        }
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
      loom_low_lower_representation_claim_operand(recorder, 0);
      return;
    case LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_MMA:
      loom_amdgpu_matrix_representation_observe_mma(context, source_op,
                                                    recorder);
      loom_low_lower_representation_claim_operand(recorder, 2);
      return;
    case LOOM_AMDGPU_MATRIX_REPRESENTATION_ACTION_FRAGMENT:
      loom_amdgpu_matrix_representation_observe_fragment(context, source_op,
                                                         recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU matrix representation action");
}

static void loom_amdgpu_source_representation_observe_callable_boundary(
    void* user_data,
    loom_low_lower_representation_callable_boundary_kind_t kind,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  loom_amdgpu_source_integer_representation_observe_callable_boundary(
      kind, context, source_op, recorder);
  const loom_module_t* module = loom_low_lower_context_module(context);
  switch (kind) {
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_DEFINITION: {
      const loom_func_like_t function =
          loom_func_like_const_cast(module, source_op);
      uint16_t argument_count = 0;
      const loom_value_id_t* argument_ids =
          loom_func_like_arg_ids(function, &argument_count);
      loom_amdgpu_matrix_representation_pin_values(context, argument_ids,
                                                   argument_count, recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_CALL: {
      const loom_call_like_t call =
          loom_call_like_const_cast(module, source_op);
      const loom_value_slice_t operands = loom_call_like_operands(call);
      const loom_value_slice_t results = loom_call_like_results(call);
      loom_amdgpu_matrix_representation_pin_values(context, operands.values,
                                                   operands.count, recorder);
      loom_amdgpu_matrix_representation_pin_values(context, results.values,
                                                   results.count, recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_EXIT:
      loom_amdgpu_matrix_representation_pin_values(
          context, loom_op_const_operands(source_op), source_op->operand_count,
          recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown callable representation boundary kind");
}

static const loom_low_lower_representation_boundary_t
    kAmdgpuSourceRepresentationBoundaries[] = {
        {LOOM_OP_SCALAR_SITOFP,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGNED_CONVERSION,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_SCALAR_UITOFP,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_UNSIGNED_CONVERSION,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_SCALAR_FPTOSI,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_SCALAR_FPTOUI,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_SCALAR_EXTSI,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGNED_CONVERSION,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_SCALAR_EXTUI,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_UNSIGNED_CONVERSION,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_SCALAR_TRUNCI,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_SCALAR_BITCAST,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_SCALAR_CONSTANT,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_CONSTANT_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VIEW_LOAD,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGN_EXTENDED_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VIEW_ATOMIC_CMPXCHG,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGN_EXTENDED_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VIEW_ATOMIC_LOAD,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGN_EXTENDED_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_EXTRACT,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGN_EXTENDED_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
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
        {LOOM_OP_INDEX_CAST,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_SIGN_EXTENDED_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_KERNEL_SUBGROUP_SHUFFLE,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_TRANSPORT_PAYLOAD,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_KERNEL_SUBGROUP_BROADCAST,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_TRANSPORT_PAYLOAD,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST,
         LOOM_AMDGPU_SOURCE_INTEGER_REPRESENTATION_ACTION_TRANSPORT_PAYLOAD,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
};
static const loom_low_lower_representation_boundary_span_t
    kAmdgpuSourceRepresentationBoundarySpans[LOOM_DIALECT_KERNEL -
                                             LOOM_DIALECT_SCALAR + 1] = {
        [LOOM_DIALECT_SCALAR - LOOM_DIALECT_SCALAR] = {0, 9},
        [LOOM_DIALECT_VIEW - LOOM_DIALECT_SCALAR] = {9, 3},
        [LOOM_DIALECT_VECTOR - LOOM_DIALECT_SCALAR] = {12, 6},
        [LOOM_DIALECT_INDEX - LOOM_DIALECT_SCALAR] = {18, 1},
        [LOOM_DIALECT_KERNEL - LOOM_DIALECT_SCALAR] = {19, 3},
};
static_assert(9 + 3 + 6 + 1 + 3 ==
                  IREE_ARRAYSIZE(kAmdgpuSourceRepresentationBoundaries),
              "AMDGPU representation spans must cover every boundary");
static_assert((loom_op_kind_t)LOOM_OP_SCALAR_SITOFP <
                      (loom_op_kind_t)LOOM_OP_SCALAR_UITOFP &&
                  (loom_op_kind_t)LOOM_OP_SCALAR_UITOFP <
                      (loom_op_kind_t)LOOM_OP_SCALAR_FPTOSI &&
                  (loom_op_kind_t)LOOM_OP_SCALAR_FPTOSI <
                      (loom_op_kind_t)LOOM_OP_SCALAR_FPTOUI &&
                  (loom_op_kind_t)LOOM_OP_SCALAR_FPTOUI <
                      (loom_op_kind_t)LOOM_OP_SCALAR_EXTSI &&
                  (loom_op_kind_t)LOOM_OP_SCALAR_EXTSI <
                      (loom_op_kind_t)LOOM_OP_SCALAR_EXTUI &&
                  (loom_op_kind_t)LOOM_OP_SCALAR_EXTUI <
                      (loom_op_kind_t)LOOM_OP_SCALAR_TRUNCI &&
                  (loom_op_kind_t)LOOM_OP_SCALAR_TRUNCI <
                      (loom_op_kind_t)LOOM_OP_SCALAR_BITCAST &&
                  (loom_op_kind_t)LOOM_OP_SCALAR_BITCAST <
                      (loom_op_kind_t)LOOM_OP_SCALAR_CONSTANT &&
                  (loom_op_kind_t)LOOM_OP_SCALAR_CONSTANT <
                      (loom_op_kind_t)LOOM_OP_VIEW_LOAD &&
                  (loom_op_kind_t)LOOM_OP_VIEW_LOAD <
                      (loom_op_kind_t)LOOM_OP_VIEW_ATOMIC_CMPXCHG &&
                  (loom_op_kind_t)LOOM_OP_VIEW_ATOMIC_CMPXCHG <
                      (loom_op_kind_t)LOOM_OP_VIEW_ATOMIC_LOAD &&
                  (loom_op_kind_t)LOOM_OP_VIEW_ATOMIC_LOAD <
                      (loom_op_kind_t)LOOM_OP_VECTOR_EXTRACT &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_EXTRACT <
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
                      (loom_op_kind_t)LOOM_OP_INDEX_CAST &&
                  (loom_op_kind_t)LOOM_OP_INDEX_CAST <
                      (loom_op_kind_t)LOOM_OP_KERNEL_SUBGROUP_SHUFFLE &&
                  (loom_op_kind_t)LOOM_OP_KERNEL_SUBGROUP_SHUFFLE <
                      (loom_op_kind_t)LOOM_OP_KERNEL_SUBGROUP_BROADCAST &&
                  (loom_op_kind_t)LOOM_OP_KERNEL_SUBGROUP_BROADCAST <
                      (loom_op_kind_t)LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST,
              "AMDGPU representation boundaries must remain ordered");

static const loom_low_lower_representation_provider_t
    kAmdgpuSourceRepresentationProvider = {
        .relation = loom_amdgpu_source_representation_relation,
        .observe_boundary = loom_amdgpu_source_representation_observe_boundary,
        .observe_callable_boundary =
            loom_amdgpu_source_representation_observe_callable_boundary,
        .observe_unclaimed_operand =
            loom_amdgpu_source_representation_observe_unclaimed_operand,
        .boundaries = kAmdgpuSourceRepresentationBoundaries,
        .boundary_spans = kAmdgpuSourceRepresentationBoundarySpans,
        .boundary_count = IREE_ARRAYSIZE(kAmdgpuSourceRepresentationBoundaries),
        .boundary_dialect_base_id = LOOM_DIALECT_SCALAR,
        .boundary_dialect_count =
            IREE_ARRAYSIZE(kAmdgpuSourceRepresentationBoundarySpans),
        .relation_mask = LOOM_VALUE_RELATION_MASK_ALL,
};

const loom_low_lower_source_plan_observer_t
    loom_amdgpu_source_representation_observer = {
        .begin = loom_low_lower_representation_observer_begin,
        .observe = loom_low_lower_representation_observer_observe,
        .end = loom_low_lower_representation_observer_end,
        .user_data = (void*)&kAmdgpuSourceRepresentationProvider,
};
