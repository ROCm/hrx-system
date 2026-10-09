// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/lower/predicate_representation.h"

#include "loom/ops/vector/ops.h"

typedef enum loom_wasm_predicate_representation_action_e {
  LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT = 0,
  LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_EXTRACT = 1,
  LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_INSERT = 2,
  LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_PRESERVE_STRUCTURE = 3,
  LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_SELECT = 4,
  LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_COMPARE = 5,
} loom_wasm_predicate_representation_action_t;

static const loom_low_representation_id_t kWasmPredicateRepresentations[] = {
    LOOM_WASM_PREDICATE_REPRESENTATION_I8X16,
    LOOM_WASM_PREDICATE_REPRESENTATION_I16X8,
    LOOM_WASM_PREDICATE_REPRESENTATION_I32X4,
    LOOM_WASM_PREDICATE_REPRESENTATION_I64X2,
};
static_assert(IREE_ARRAYSIZE(kWasmPredicateRepresentations) ==
                  LOOM_WASM_PREDICATE_REPRESENTATION_COUNT,
              "predicate representation count must match its table");

bool loom_wasm_predicate_type(loom_type_t source_type,
                              uint32_t* out_lane_count) {
  if (out_lane_count != NULL) {
    *out_lane_count = 0;
  }
  if (!loom_type_is_vector(source_type) ||
      !loom_type_is_all_static(source_type) ||
      loom_type_element_type(source_type) != LOOM_SCALAR_TYPE_I1) {
    return false;
  }
  uint64_t lane_count = 0;
  if (!loom_type_static_element_count(source_type, &lane_count) ||
      lane_count < 1 || lane_count > 16) {
    return false;
  }
  if (out_lane_count != NULL) {
    *out_lane_count = (uint32_t)lane_count;
  }
  return true;
}

bool loom_wasm_predicate_representation_available(
    uint32_t lane_count, loom_low_representation_id_t representation) {
  if (lane_count < 1 || lane_count > 16) {
    return false;
  }
  switch (representation) {
    case LOOM_WASM_PREDICATE_REPRESENTATION_I8X16:
    case LOOM_WASM_PREDICATE_REPRESENTATION_I16X8:
    case LOOM_WASM_PREDICATE_REPRESENTATION_I32X4:
    case LOOM_WASM_PREDICATE_REPRESENTATION_I64X2:
      return lane_count * representation <= 128;
    default:
      return false;
  }
}

loom_low_representation_id_t loom_wasm_predicate_default_representation(
    loom_type_t source_type) {
  uint32_t lane_count = 0;
  if (!loom_wasm_predicate_type(source_type, &lane_count)) {
    return LOOM_LOW_REPRESENTATION_ID_NONE;
  }
  for (iree_host_size_t i = IREE_ARRAYSIZE(kWasmPredicateRepresentations);
       i > 0; --i) {
    const loom_low_representation_id_t representation =
        kWasmPredicateRepresentations[i - 1];
    if (loom_wasm_predicate_representation_available(lane_count,
                                                     representation)) {
      return representation;
    }
  }
  IREE_BUILTIN_UNREACHABLE();
}

loom_wasm_vector_carrier_t loom_wasm_predicate_carrier(
    loom_type_t source_type, loom_low_representation_id_t representation) {
  uint32_t lane_count = 0;
  if (!loom_wasm_predicate_type(source_type, &lane_count)) {
    return (loom_wasm_vector_carrier_t){0};
  }
  if (representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    representation = loom_wasm_predicate_default_representation(source_type);
  }
  if (!loom_wasm_predicate_representation_available(lane_count,
                                                    representation)) {
    return (loom_wasm_vector_carrier_t){0};
  }
  return loom_wasm_vector_carrier_for_physical_element(source_type,
                                                       representation);
}

iree_host_size_t loom_wasm_predicate_representation_candidates(
    loom_type_t source_type, loom_low_representation_id_t native_representation,
    loom_low_representation_candidate_t
        out_candidates[LOOM_WASM_PREDICATE_REPRESENTATION_COUNT]) {
  uint32_t lane_count = 0;
  if (!loom_wasm_predicate_type(source_type, &lane_count)) {
    return 0;
  }
  iree_host_size_t candidate_count = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kWasmPredicateRepresentations); ++i) {
    const loom_low_representation_id_t representation =
        kWasmPredicateRepresentations[i];
    if (!loom_wasm_predicate_representation_available(lane_count,
                                                      representation)) {
      continue;
    }
    out_candidates[candidate_count++] = (loom_low_representation_candidate_t){
        .representation = representation,
        .cost = native_representation == LOOM_LOW_REPRESENTATION_ID_NONE ||
                        native_representation == representation
                    ? (loom_low_representation_cost_t){0}
                    : (loom_low_representation_cost_t){.runtime = 1,
                                                       .code_size = 18},
    };
  }
  return candidate_count;
}

static void loom_wasm_predicate_constrain_value(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_low_representation_id_t native_representation,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_low_representation_candidate_t
      candidates[LOOM_WASM_PREDICATE_REPRESENTATION_COUNT];
  const iree_host_size_t candidate_count =
      loom_wasm_predicate_representation_candidates(
          loom_module_value_type(module, value_id), native_representation,
          candidates);
  if (candidate_count != 0) {
    loom_low_lower_representation_record_candidates(
        recorder, value_id, candidates, candidate_count);
  }
}

static void loom_wasm_predicate_constrain_default_value(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_type_t source_type = loom_module_value_type(module, value_id);
  const loom_low_representation_id_t representation =
      loom_wasm_predicate_default_representation(source_type);
  if (representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    return;
  }
  const loom_low_representation_candidate_t candidate = {
      .representation = representation,
  };
  loom_low_lower_representation_record_candidates(recorder, value_id,
                                                  &candidate, 1);
}

static void loom_wasm_predicate_contribute_costs(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_low_representation_id_t native_representation,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_low_representation_candidate_t
      candidates[LOOM_WASM_PREDICATE_REPRESENTATION_COUNT];
  const iree_host_size_t candidate_count =
      loom_wasm_predicate_representation_candidates(
          loom_module_value_type(module, value_id), native_representation,
          candidates);
  if (candidate_count != 0) {
    loom_low_lower_representation_record_costs(recorder, value_id, candidates,
                                               candidate_count);
  }
}

static bool loom_wasm_predicate_representation_relation(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, const loom_value_relation_t* relation,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
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
         loom_wasm_predicate_type(source_type, NULL);
}

static void loom_wasm_predicate_observe_lane_access(
    loom_low_lower_context_t* context, loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_wasm_predicate_contribute_costs(
      loom_low_lower_context_module(context), value_id,
      LOOM_LOW_REPRESENTATION_ID_NONE, recorder);
}

static void loom_wasm_predicate_observe_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t destination = loom_vector_insert_dest(source_op);
  const loom_value_id_t result = loom_vector_insert_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_wasm_predicate_type(loom_module_value_type(module, destination),
                                NULL)) {
    return;
  }
  loom_low_lower_representation_record_union(recorder, destination, result);
  loom_wasm_predicate_observe_lane_access(context, destination, recorder);
}

static void loom_wasm_predicate_observe_structure(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_value_id_t representative = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    if (!loom_wasm_predicate_type(loom_module_value_type(module, operands[i]),
                                  NULL)) {
      continue;
    }
    loom_wasm_predicate_constrain_value(
        module, operands[i], LOOM_LOW_REPRESENTATION_ID_NONE, recorder);
    if (representative == LOOM_VALUE_ID_INVALID) {
      representative = operands[i];
    } else {
      loom_low_lower_representation_record_union(recorder, representative,
                                                 operands[i]);
    }
    loom_low_lower_representation_claim_operand(recorder, i);
  }
  const loom_value_id_t* results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    if (!loom_wasm_predicate_type(loom_module_value_type(module, results[i]),
                                  NULL)) {
      continue;
    }
    loom_wasm_predicate_constrain_value(
        module, results[i], LOOM_LOW_REPRESENTATION_ID_NONE, recorder);
    if (representative == LOOM_VALUE_ID_INVALID) {
      representative = results[i];
    } else {
      loom_low_lower_representation_record_union(recorder, representative,
                                                 results[i]);
    }
  }
}

static void loom_wasm_predicate_observe_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t condition = loom_vector_select_condition(source_op);
  if (!loom_wasm_predicate_type(loom_module_value_type(module, condition),
                                NULL)) {
    return;
  }
  const loom_value_id_t true_value = loom_vector_select_true_value(source_op);
  const loom_type_t value_type = loom_module_value_type(module, true_value);
  if (loom_wasm_predicate_type(value_type, NULL)) {
    loom_low_lower_representation_record_union(recorder, condition, true_value);
  } else if (loom_type_is_vector(value_type)) {
    const uint16_t element_bit_count = loom_wasm_scalar_type_physical_bit_count(
        loom_type_element_type(value_type));
    if (element_bit_count != 0) {
      loom_wasm_predicate_contribute_costs(module, condition, element_bit_count,
                                           recorder);
    }
  }
  loom_low_lower_representation_claim_operand(recorder, 0);
}

static void loom_wasm_predicate_observe_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const bool is_integer = loom_vector_cmpi_isa(source_op);
  const loom_value_id_t lhs = is_integer ? loom_vector_cmpi_lhs(source_op)
                                         : loom_vector_cmpf_lhs(source_op);
  const loom_value_id_t result = is_integer
                                     ? loom_vector_cmpi_result(source_op)
                                     : loom_vector_cmpf_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t operand_type = loom_module_value_type(module, lhs);
  if (!loom_type_is_vector(operand_type)) {
    return;
  }
  const uint16_t element_bit_count = loom_wasm_scalar_type_physical_bit_count(
      loom_type_element_type(operand_type));
  if (element_bit_count != 0) {
    loom_wasm_predicate_constrain_value(module, result, element_bit_count,
                                        recorder);
  }
}

static void loom_wasm_predicate_observe_boundary(
    void* user_data, uint8_t action,
    loom_low_lower_representation_boundary_flags_t flags,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  (void)flags;
  switch ((loom_wasm_predicate_representation_action_t)action) {
    case LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT:
      loom_wasm_predicate_constrain_value(
          loom_low_lower_context_module(context),
          loom_op_const_results(source_op)[0], LOOM_LOW_REPRESENTATION_ID_NONE,
          recorder);
      return;
    case LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_EXTRACT:
      loom_wasm_predicate_observe_lane_access(
          context, loom_vector_extract_source(source_op), recorder);
      loom_low_lower_representation_claim_operand(recorder, 0);
      return;
    case LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_INSERT:
      loom_wasm_predicate_observe_insert(context, source_op, recorder);
      loom_low_lower_representation_claim_operand(recorder, 0);
      return;
    case LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_PRESERVE_STRUCTURE:
      loom_wasm_predicate_observe_structure(context, source_op, recorder);
      return;
    case LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_SELECT:
      loom_wasm_predicate_observe_select(context, source_op, recorder);
      return;
    case LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_COMPARE:
      loom_wasm_predicate_observe_compare(context, source_op, recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown Wasm predicate representation action");
}

static void loom_wasm_predicate_constrain_default_values(
    const loom_module_t* module, const loom_value_id_t* value_ids,
    iree_host_size_t value_count,
    loom_low_lower_representation_recorder_t* recorder) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_wasm_predicate_constrain_default_value(module, value_ids[i], recorder);
  }
}

static void loom_wasm_predicate_observe_callable_boundary(
    void* user_data,
    loom_low_lower_representation_callable_boundary_kind_t kind,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  const loom_module_t* module = loom_low_lower_context_module(context);
  switch (kind) {
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_DEFINITION: {
      const loom_func_like_t function =
          loom_func_like_const_cast(module, source_op);
      uint16_t argument_count = 0;
      const loom_value_id_t* argument_ids =
          loom_func_like_arg_ids(function, &argument_count);
      loom_wasm_predicate_constrain_default_values(module, argument_ids,
                                                   argument_count, recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_CALL: {
      const loom_call_like_t call =
          loom_call_like_const_cast(module, source_op);
      const loom_value_slice_t operands = loom_call_like_operands(call);
      const loom_value_slice_t results = loom_call_like_results(call);
      loom_wasm_predicate_constrain_default_values(module, operands.values,
                                                   operands.count, recorder);
      loom_wasm_predicate_constrain_default_values(module, results.values,
                                                   results.count, recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_EXIT:
      loom_wasm_predicate_constrain_default_values(
          module, loom_op_const_operands(source_op), source_op->operand_count,
          recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown callable representation boundary kind");
}

static void loom_wasm_predicate_observe_unclaimed_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, uint16_t source_operand_index,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  loom_wasm_predicate_constrain_default_value(
      loom_low_lower_context_module(context),
      loom_op_const_operands(source_op)[source_operand_index], recorder);
}

static const loom_low_lower_representation_boundary_t
    kWasmPredicateRepresentationBoundaries[] = {
        {LOOM_OP_VECTOR_CONSTANT,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_SPLAT,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_FROM_ELEMENTS,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_FLEXIBLE_RESULT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_EXTRACT,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_EXTRACT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_INSERT,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_INSERT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_SLICE,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_PRESERVE_STRUCTURE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_CONCAT,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_PRESERVE_STRUCTURE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_SHUFFLE,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_PRESERVE_STRUCTURE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_INTERLEAVE,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_PRESERVE_STRUCTURE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_DEINTERLEAVE,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_PRESERVE_STRUCTURE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_SELECT,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_SELECT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_CMPI, LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_COMPARE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_CMPF, LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_COMPARE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_BITCAST,
         LOOM_WASM_PREDICATE_REPRESENTATION_ACTION_PRESERVE_STRUCTURE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
};
static const loom_low_lower_representation_boundary_span_t
    kWasmPredicateRepresentationBoundarySpans[] = {
        {0, IREE_ARRAYSIZE(kWasmPredicateRepresentationBoundaries)},
};
static_assert((loom_op_kind_t)LOOM_OP_VECTOR_CONSTANT <
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
                      (loom_op_kind_t)LOOM_OP_VECTOR_CONCAT &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_CONCAT <
                      (loom_op_kind_t)LOOM_OP_VECTOR_SHUFFLE &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_SHUFFLE <
                      (loom_op_kind_t)LOOM_OP_VECTOR_INTERLEAVE &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_INTERLEAVE <
                      (loom_op_kind_t)LOOM_OP_VECTOR_DEINTERLEAVE &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_DEINTERLEAVE <
                      (loom_op_kind_t)LOOM_OP_VECTOR_SELECT &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_SELECT <
                      (loom_op_kind_t)LOOM_OP_VECTOR_CMPI &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_CMPI <
                      (loom_op_kind_t)LOOM_OP_VECTOR_CMPF &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_CMPF <
                      (loom_op_kind_t)LOOM_OP_VECTOR_BITCAST,
              "Wasm predicate representation boundaries must remain ordered");

static const loom_low_lower_representation_provider_t
    kWasmPredicateRepresentationProvider = {
        .relation = loom_wasm_predicate_representation_relation,
        .observe_boundary = loom_wasm_predicate_observe_boundary,
        .observe_callable_boundary =
            loom_wasm_predicate_observe_callable_boundary,
        .observe_unclaimed_operand =
            loom_wasm_predicate_observe_unclaimed_operand,
        .boundaries = kWasmPredicateRepresentationBoundaries,
        .boundary_spans = kWasmPredicateRepresentationBoundarySpans,
        .boundary_count =
            IREE_ARRAYSIZE(kWasmPredicateRepresentationBoundaries),
        .boundary_dialect_base_id = LOOM_DIALECT_VECTOR,
        .boundary_dialect_count =
            IREE_ARRAYSIZE(kWasmPredicateRepresentationBoundarySpans),
        .relation_mask = LOOM_VALUE_RELATION_MASK_ALL,
};

const loom_low_lower_source_plan_observer_t
    loom_wasm_predicate_representation_observer = {
        .begin = loom_low_lower_representation_observer_begin,
        .observe = loom_low_lower_representation_observer_observe,
        .end = loom_low_lower_representation_observer_end,
        .user_data = (void*)&kWasmPredicateRepresentationProvider,
};
