// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/f32_representation.h"

#include <math.h>

#include "loom/ops/vector/ops.h"

typedef enum loom_aie2p_f32_representation_action_e {
  LOOM_AIE2P_F32_REPRESENTATION_ACTION_CONSTANT = 0,
  LOOM_AIE2P_F32_REPRESENTATION_ACTION_BINARY = 1,
  LOOM_AIE2P_F32_REPRESENTATION_ACTION_NATIVE_NONLINEAR = 2,
  LOOM_AIE2P_F32_REPRESENTATION_ACTION_EXTEND = 3,
  LOOM_AIE2P_F32_REPRESENTATION_ACTION_TRUNCATE = 4,
  LOOM_AIE2P_F32_REPRESENTATION_ACTION_DOT2 = 5,
  LOOM_AIE2P_F32_REPRESENTATION_ACTION_REDUCE_ADD = 6,
  LOOM_AIE2P_F32_REPRESENTATION_ACTION_SELECT = 7,
} loom_aie2p_f32_representation_action_t;

static bool loom_aie2p_packet_type(loom_type_t source_type,
                                   loom_scalar_type_t element_type,
                                   int64_t lane_count) {
  return loom_type_is_vector(source_type) && loom_type_rank(source_type) == 1 &&
         loom_type_is_all_static(source_type) &&
         loom_type_element_type(source_type) == element_type &&
         loom_type_dim_static_size_at(source_type, 0) == lane_count;
}

bool loom_aie2p_f32_packet_type(loom_type_t source_type) {
  return loom_aie2p_packet_type(source_type, LOOM_SCALAR_TYPE_F32, 16);
}

static const loom_low_representation_candidate_t
    kAie2pF32NativeAccumulatorCandidates[] = {
        {
            .representation = LOOM_AIE2P_F32_REPRESENTATION_VECTOR,
            .cost = {.runtime = 1, .code_size = 4},
        },
        {.representation = LOOM_AIE2P_F32_REPRESENTATION_ACCUMULATOR},
};

static const loom_low_representation_candidate_t
    kAie2pF32NativeVectorCandidates[] = {
        {.representation = LOOM_AIE2P_F32_REPRESENTATION_VECTOR},
        {
            .representation = LOOM_AIE2P_F32_REPRESENTATION_ACCUMULATOR,
            .cost = {.runtime = 1, .code_size = 4},
        },
};

static const loom_low_representation_candidate_t
    kAie2pF32BalancedConsumerCandidates[] = {
        {
            .representation = LOOM_AIE2P_F32_REPRESENTATION_VECTOR,
            .cost = {.runtime = 1, .code_size = 4},
        },
        {
            .representation = LOOM_AIE2P_F32_REPRESENTATION_ACCUMULATOR,
            .cost = {.runtime = 1, .code_size = 4},
        },
};

static void loom_aie2p_f32_record_native_accumulator_producer(
    loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_low_lower_representation_record_candidates(
      recorder, value_id, kAie2pF32NativeAccumulatorCandidates,
      IREE_ARRAYSIZE(kAie2pF32NativeAccumulatorCandidates));
}

static void loom_aie2p_f32_record_native_accumulator_consumer(
    loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  loom_low_lower_representation_record_costs(
      recorder, value_id, kAie2pF32NativeAccumulatorCandidates,
      IREE_ARRAYSIZE(kAie2pF32NativeAccumulatorCandidates));
}

static void loom_aie2p_f32_constrain_vector(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_low_lower_representation_recorder_t* recorder) {
  if (!loom_aie2p_f32_packet_type(loom_module_value_type(module, value_id))) {
    return;
  }
  const loom_low_representation_candidate_t candidate = {
      .representation = LOOM_AIE2P_F32_REPRESENTATION_VECTOR,
  };
  loom_low_lower_representation_record_candidates(recorder, value_id,
                                                  &candidate, 1);
}

static void loom_aie2p_f32_constrain_vectors(
    const loom_module_t* module, const loom_value_id_t* value_ids,
    iree_host_size_t value_count,
    loom_low_lower_representation_recorder_t* recorder) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    loom_aie2p_f32_constrain_vector(module, value_ids[i], recorder);
  }
}

static void loom_aie2p_f32_observe_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t result = loom_vector_splatf_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_aie2p_f32_packet_type(loom_module_value_type(module, result))) {
    return;
  }
  const double value = loom_vector_splatf_value(source_op);
  if (value != 0.0 || signbit(value)) {
    loom_aie2p_f32_constrain_vector(module, result, recorder);
    return;
  }
  loom_low_lower_representation_record_candidates(
      recorder, result, kAie2pF32NativeAccumulatorCandidates,
      IREE_ARRAYSIZE(kAie2pF32NativeAccumulatorCandidates));
}

static bool loom_aie2p_f32_representation_relation(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, const loom_value_relation_t* relation,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  (void)source_op;
  (void)recorder;
  if (iree_any_bit_set(relation->flags, LOOM_VALUE_RELATION_FLAG_TYPE_CHANGE)) {
    return false;
  }
  switch (relation->kind) {
    case LOOM_VALUE_RELATION_FACT_IDENTITY:
    case LOOM_VALUE_RELATION_VALUE_ALIAS:
    case LOOM_VALUE_RELATION_CFG_ARGUMENT:
    case LOOM_VALUE_RELATION_LOOP_CARRIED:
    case LOOM_VALUE_RELATION_LOOP_BYPASS:
    case LOOM_VALUE_RELATION_REGION_RESULT:
      break;
    default:
      return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, relation->source_value_id);
  return loom_aie2p_f32_packet_type(source_type) &&
         loom_type_equal(
             source_type,
             loom_module_value_type(module, relation->destination_value_id));
}

static void loom_aie2p_f32_observe_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const bool is_add = loom_vector_addf_isa(source_op);
  const loom_value_id_t lhs = is_add ? loom_vector_addf_lhs(source_op)
                                     : loom_vector_subf_lhs(source_op);
  const loom_value_id_t rhs = is_add ? loom_vector_addf_rhs(source_op)
                                     : loom_vector_subf_rhs(source_op);
  const loom_value_id_t result = is_add ? loom_vector_addf_result(source_op)
                                        : loom_vector_subf_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_aie2p_f32_packet_type(loom_module_value_type(module, result))) {
    return;
  }

  loom_aie2p_f32_record_native_accumulator_consumer(lhs, recorder);
  loom_aie2p_f32_record_native_accumulator_consumer(rhs, recorder);
  loom_aie2p_f32_record_native_accumulator_producer(result, recorder);
  loom_low_lower_representation_claim_operand(recorder, 0);
  loom_low_lower_representation_claim_operand(recorder, 1);
}

static void loom_aie2p_f32_observe_extend(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t input = loom_vector_extf_input(source_op);
  const loom_value_id_t result = loom_vector_extf_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_aie2p_packet_type(loom_module_value_type(module, input),
                              LOOM_SCALAR_TYPE_BF16, 16) ||
      !loom_aie2p_f32_packet_type(loom_module_value_type(module, result))) {
    return;
  }
  loom_aie2p_f32_record_native_accumulator_producer(result, recorder);
}

static void loom_aie2p_f32_observe_native_nonlinear(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t input = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_aie2p_f32_packet_type(loom_module_value_type(module, input)) ||
      !loom_aie2p_f32_packet_type(loom_module_value_type(module, result))) {
    return;
  }
  loom_aie2p_f32_record_native_accumulator_consumer(input, recorder);
  loom_aie2p_f32_record_native_accumulator_producer(result, recorder);
  loom_low_lower_representation_claim_operand(recorder, 0);
}

static void loom_aie2p_f32_observe_truncate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t input = loom_vector_fptrunc_input(source_op);
  const loom_value_id_t result = loom_vector_fptrunc_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_aie2p_f32_packet_type(loom_module_value_type(module, input)) ||
      !loom_aie2p_packet_type(loom_module_value_type(module, result),
                              LOOM_SCALAR_TYPE_BF16, 16)) {
    return;
  }
  loom_aie2p_f32_record_native_accumulator_consumer(input, recorder);
  loom_low_lower_representation_claim_operand(recorder, 0);
}

static void loom_aie2p_f32_observe_dot2(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t lhs = loom_vector_dot2f_lhs(source_op);
  const loom_value_id_t rhs = loom_vector_dot2f_rhs(source_op);
  const loom_value_id_t accumulator = loom_vector_dot2f_acc(source_op);
  const loom_value_id_t result = loom_vector_dot2f_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_aie2p_packet_type(loom_module_value_type(module, lhs),
                              LOOM_SCALAR_TYPE_BF16, 32) ||
      !loom_aie2p_packet_type(loom_module_value_type(module, rhs),
                              LOOM_SCALAR_TYPE_BF16, 32) ||
      !loom_aie2p_f32_packet_type(
          loom_module_value_type(module, accumulator)) ||
      !loom_aie2p_f32_packet_type(loom_module_value_type(module, result))) {
    return;
  }
  loom_aie2p_f32_record_native_accumulator_consumer(accumulator, recorder);
  loom_aie2p_f32_record_native_accumulator_producer(result, recorder);
  loom_low_lower_representation_claim_operand(recorder, 2);
}

static void loom_aie2p_f32_observe_reduce_add(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  if (loom_vector_reduce_kind(source_op) != LOOM_COMBINING_KIND_ADDF) {
    return;
  }
  const loom_value_id_t input = loom_vector_reduce_input(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_aie2p_f32_packet_type(loom_module_value_type(module, input))) {
    return;
  }
  loom_low_lower_representation_record_costs(
      recorder, input, kAie2pF32BalancedConsumerCandidates,
      IREE_ARRAYSIZE(kAie2pF32BalancedConsumerCandidates));
  loom_low_lower_representation_claim_operand(recorder, 0);
}

static void loom_aie2p_f32_observe_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  const loom_value_id_t true_value = loom_vector_select_true_value(source_op);
  const loom_value_id_t false_value = loom_vector_select_false_value(source_op);
  const loom_value_id_t result = loom_vector_select_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_aie2p_f32_packet_type(loom_module_value_type(module, true_value)) ||
      !loom_aie2p_f32_packet_type(
          loom_module_value_type(module, false_value)) ||
      !loom_aie2p_f32_packet_type(loom_module_value_type(module, result))) {
    return;
  }

  loom_low_lower_representation_record_costs(
      recorder, true_value, kAie2pF32NativeVectorCandidates,
      IREE_ARRAYSIZE(kAie2pF32NativeVectorCandidates));
  loom_low_lower_representation_record_costs(
      recorder, false_value, kAie2pF32NativeVectorCandidates,
      IREE_ARRAYSIZE(kAie2pF32NativeVectorCandidates));
  loom_low_lower_representation_record_candidates(
      recorder, result, kAie2pF32NativeVectorCandidates,
      IREE_ARRAYSIZE(kAie2pF32NativeVectorCandidates));
  loom_low_lower_representation_claim_operand(recorder, 1);
  loom_low_lower_representation_claim_operand(recorder, 2);
}

static void loom_aie2p_f32_observe_boundary(
    void* user_data, uint8_t action,
    loom_low_lower_representation_boundary_flags_t flags,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  (void)flags;
  switch ((loom_aie2p_f32_representation_action_t)action) {
    case LOOM_AIE2P_F32_REPRESENTATION_ACTION_CONSTANT:
      loom_aie2p_f32_observe_constant(context, source_op, recorder);
      return;
    case LOOM_AIE2P_F32_REPRESENTATION_ACTION_BINARY:
      loom_aie2p_f32_observe_binary(context, source_op, recorder);
      return;
    case LOOM_AIE2P_F32_REPRESENTATION_ACTION_NATIVE_NONLINEAR:
      loom_aie2p_f32_observe_native_nonlinear(context, source_op, recorder);
      return;
    case LOOM_AIE2P_F32_REPRESENTATION_ACTION_EXTEND:
      loom_aie2p_f32_observe_extend(context, source_op, recorder);
      return;
    case LOOM_AIE2P_F32_REPRESENTATION_ACTION_TRUNCATE:
      loom_aie2p_f32_observe_truncate(context, source_op, recorder);
      return;
    case LOOM_AIE2P_F32_REPRESENTATION_ACTION_DOT2:
      loom_aie2p_f32_observe_dot2(context, source_op, recorder);
      return;
    case LOOM_AIE2P_F32_REPRESENTATION_ACTION_REDUCE_ADD:
      loom_aie2p_f32_observe_reduce_add(context, source_op, recorder);
      return;
    case LOOM_AIE2P_F32_REPRESENTATION_ACTION_SELECT:
      loom_aie2p_f32_observe_select(context, source_op, recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown AIE2P F32 representation action");
}

static void loom_aie2p_f32_observe_callable_boundary(
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
      loom_aie2p_f32_constrain_vectors(module, argument_ids, argument_count,
                                       recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_CALL: {
      const loom_call_like_t call =
          loom_call_like_const_cast(module, source_op);
      const loom_value_slice_t operands = loom_call_like_operands(call);
      const loom_value_slice_t results = loom_call_like_results(call);
      loom_aie2p_f32_constrain_vectors(module, operands.values, operands.count,
                                       recorder);
      loom_aie2p_f32_constrain_vectors(module, results.values, results.count,
                                       recorder);
      return;
    }
    case LOOM_LOW_LOWER_REPRESENTATION_CALLABLE_EXIT:
      loom_aie2p_f32_constrain_vectors(module,
                                       loom_op_const_operands(source_op),
                                       source_op->operand_count, recorder);
      return;
  }
  IREE_ASSERT_UNREACHABLE("unknown callable representation boundary kind");
}

static void loom_aie2p_f32_observe_unclaimed_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, uint16_t source_operand_index,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  loom_aie2p_f32_constrain_vector(
      loom_low_lower_context_module(context),
      loom_op_const_operands(source_op)[source_operand_index], recorder);
}

static const loom_low_lower_representation_boundary_t
    kAie2pF32RepresentationBoundaries[] = {
        {LOOM_OP_VECTOR_SELECT, LOOM_AIE2P_F32_REPRESENTATION_ACTION_SELECT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_ADDF, LOOM_AIE2P_F32_REPRESENTATION_ACTION_BINARY,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_SUBF, LOOM_AIE2P_F32_REPRESENTATION_ACTION_BINARY,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_EXP2F,
         LOOM_AIE2P_F32_REPRESENTATION_ACTION_NATIVE_NONLINEAR,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_TANHF,
         LOOM_AIE2P_F32_REPRESENTATION_ACTION_NATIVE_NONLINEAR,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_EXTF, LOOM_AIE2P_F32_REPRESENTATION_ACTION_EXTEND,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
        {LOOM_OP_VECTOR_FPTRUNC, LOOM_AIE2P_F32_REPRESENTATION_ACTION_TRUNCATE,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_DOT2F, LOOM_AIE2P_F32_REPRESENTATION_ACTION_DOT2,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL},
        {LOOM_OP_VECTOR_REDUCE, LOOM_AIE2P_F32_REPRESENTATION_ACTION_REDUCE_ADD,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS},
        {LOOM_OP_VECTOR_SPLATF, LOOM_AIE2P_F32_REPRESENTATION_ACTION_CONSTANT,
         LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
};

static const loom_low_lower_representation_boundary_span_t
    kAie2pF32RepresentationBoundarySpans[] = {
        {0, IREE_ARRAYSIZE(kAie2pF32RepresentationBoundaries)},
};
static_assert((loom_op_kind_t)LOOM_OP_VECTOR_SELECT <
                      (loom_op_kind_t)LOOM_OP_VECTOR_ADDF &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_ADDF <
                      (loom_op_kind_t)LOOM_OP_VECTOR_SUBF &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_SUBF <
                      (loom_op_kind_t)LOOM_OP_VECTOR_EXP2F &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_EXP2F <
                      (loom_op_kind_t)LOOM_OP_VECTOR_TANHF &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_TANHF <
                      (loom_op_kind_t)LOOM_OP_VECTOR_EXTF &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_EXTF <
                      (loom_op_kind_t)LOOM_OP_VECTOR_FPTRUNC &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_FPTRUNC <
                      (loom_op_kind_t)LOOM_OP_VECTOR_DOT2F &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_DOT2F <
                      (loom_op_kind_t)LOOM_OP_VECTOR_REDUCE &&
                  (loom_op_kind_t)LOOM_OP_VECTOR_REDUCE <
                      (loom_op_kind_t)LOOM_OP_VECTOR_SPLATF,
              "AIE2P f32 representation boundaries must remain ordered");

static const loom_low_lower_representation_provider_t
    kAie2pF32RepresentationProvider = {
        .relation = loom_aie2p_f32_representation_relation,
        .observe_boundary = loom_aie2p_f32_observe_boundary,
        .observe_callable_boundary = loom_aie2p_f32_observe_callable_boundary,
        .observe_unclaimed_operand = loom_aie2p_f32_observe_unclaimed_operand,
        .boundaries = kAie2pF32RepresentationBoundaries,
        .boundary_spans = kAie2pF32RepresentationBoundarySpans,
        .boundary_count = IREE_ARRAYSIZE(kAie2pF32RepresentationBoundaries),
        .boundary_dialect_base_id = LOOM_DIALECT_VECTOR,
        .boundary_dialect_count =
            IREE_ARRAYSIZE(kAie2pF32RepresentationBoundarySpans),
        .relation_mask =
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS) |
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_CFG_ARGUMENT) |
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_CARRIED) |
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_BYPASS) |
            LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REGION_RESULT),
};

const loom_low_lower_source_plan_observer_t
    loom_aie2p_f32_representation_observer = {
        .begin = loom_low_lower_representation_observer_begin,
        .observe = loom_low_lower_representation_observer_observe,
        .end = loom_low_lower_representation_observer_end,
        .user_data = (void*)&kAie2pF32RepresentationProvider,
};
