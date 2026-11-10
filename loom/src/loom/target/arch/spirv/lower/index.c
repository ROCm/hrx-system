// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/index.h"

#include "loom/codegen/low/lower/representation_observer.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/registers.h"
#include "loom/target/registers.h"

typedef enum loom_spirv_index_boundary_action_e {
  LOOM_SPIRV_INDEX_BOUNDARY_RESULT = 0,
  LOOM_SPIRV_INDEX_BOUNDARY_LOOP = 1,
  LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC = 2,
  LOOM_SPIRV_INDEX_BOUNDARY_SHIFT = 3,
} loom_spirv_index_boundary_action_t;

static bool loom_spirv_index_needs_wide(loom_value_facts_t facts) {
  return facts.range_lo >= INT32_MIN && facts.range_hi > INT32_MAX;
}

static bool loom_spirv_is_index(loom_low_lower_context_t* context,
                                loom_value_id_t value) {
  return loom_type_equal(
      loom_module_value_type(loom_low_lower_context_module(context), value),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
}

static void loom_spirv_index_constrain(
    loom_low_lower_context_t* context, loom_value_id_t value,
    loom_low_lower_representation_recorder_t* recorder) {
  if (!loom_spirv_is_index(context, value)) {
    return;
  }
  const loom_value_facts_t facts = loom_value_fact_table_lookup(
      loom_low_lower_context_fact_table(context), value);
  const loom_low_representation_candidate_t candidates[] = {
      {.representation = LOOM_SPIRV_INDEX_REPRESENTATION_NATIVE},
      {.representation = LOOM_SPIRV_INDEX_REPRESENTATION_WIDE,
       .cost = {.runtime = 1}},
  };
  const bool wide = loom_spirv_index_needs_wide(facts);
  if (!wide &&
      loom_low_lower_representation_component_is_constrained(recorder, value)) {
    return;
  }
  loom_low_lower_representation_record_candidates(
      recorder, value, candidates + wide, wide ? 1 : 2);
}

static bool loom_spirv_index_relation(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, const loom_value_relation_t* relation,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  (void)source_op;
  if (!loom_spirv_is_index(context, relation->source_value_id) ||
      !loom_spirv_is_index(context, relation->destination_value_id)) {
    return false;
  }
  switch (relation->kind) {
    case LOOM_VALUE_RELATION_TIED_RESULT:
    case LOOM_VALUE_RELATION_FACT_IDENTITY:
    case LOOM_VALUE_RELATION_VALUE_ALIAS:
    case LOOM_VALUE_RELATION_CFG_ARGUMENT:
    case LOOM_VALUE_RELATION_LOOP_CARRIED:
    case LOOM_VALUE_RELATION_LOOP_BYPASS:
    case LOOM_VALUE_RELATION_REGION_RESULT:
      loom_spirv_index_constrain(context, relation->source_value_id, recorder);
      loom_spirv_index_constrain(context, relation->destination_value_id,
                                 recorder);
      return true;
    default:
      return false;
  }
}

static void loom_spirv_index_observe_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, uint16_t operand,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  loom_spirv_index_constrain(
      context, loom_op_const_operands(source_op)[operand], recorder);
}

static void loom_spirv_index_observe_boundary(
    void* user_data, uint8_t action,
    loom_low_lower_representation_boundary_flags_t flags,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder) {
  (void)user_data;
  if (iree_any_bit_set(flags,
                       LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS)) {
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      loom_spirv_index_constrain(context, loom_op_const_results(source_op)[i],
                                 recorder);
    }
  }
  if (action == LOOM_SPIRV_INDEX_BOUNDARY_RESULT) {
    return;
  }
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  if (action != LOOM_SPIRV_INDEX_BOUNDARY_LOOP) {
    const loom_value_id_t result = loom_op_const_results(source_op)[0];
    const loom_value_fact_table_t* facts =
        loom_low_lower_context_fact_table(context);
    if (!loom_spirv_is_index(context, result) ||
        loom_value_facts_fit_signed_bit_count(
            loom_value_fact_table_lookup(facts, result), 32)) {
      return;
    }
    // A loss of interval precision cannot narrow an established wide carrier.
    // Proven native values terminate propagation; shift counts have their own
    // domain and never establish the width of the shifted data.
    const uint16_t operand_count = action == LOOM_SPIRV_INDEX_BOUNDARY_SHIFT
                                       ? 1
                                       : source_op->operand_count;
    for (uint16_t i = 0; i < operand_count; ++i) {
      if (loom_spirv_is_index(context, operands[i]) &&
          !loom_value_facts_fit_signed_bit_count(
              loom_value_fact_table_lookup(facts, operands[i]), 32)) {
        loom_low_lower_representation_record_union(recorder, result,
                                                   operands[i]);
      }
    }
    return;
  }
  if (!loom_spirv_is_index(context, operands[0])) {
    return;
  }
  const loom_block_t* body =
      loom_region_const_entry_block(loom_scf_for_body(source_op));
  bool wide = false;
  for (uint16_t i = 0; i < 3; ++i) {
    const loom_value_facts_t facts = loom_value_fact_table_lookup(
        loom_low_lower_context_fact_table(context), operands[i]);
    const bool operand_wide = loom_spirv_index_needs_wide(facts);
    wide |= operand_wide;
    // An unresolved bound still carries the width established by its producers.
    // Known ranges select the receiving width directly; they need no equality
    // with the producer's independent transport representation.
    if (!operand_wide && !loom_value_facts_fit_signed_bit_count(facts, 32)) {
      loom_low_lower_representation_record_union(recorder, body->arg_ids[0],
                                                 operands[i]);
    }
  }
  if (wide) {
    const loom_low_representation_candidate_t candidate = {
        .representation = LOOM_SPIRV_INDEX_REPRESENTATION_WIDE};
    loom_low_lower_representation_record_candidates(recorder, body->arg_ids[0],
                                                    &candidate, 1);
  }
}

// Result boundaries also retain carriers for values used only by dependent
// types. Such values have no ordinary operand use to constrain their component.
static const loom_low_lower_representation_boundary_t kIndexBoundaries[] = {
    {LOOM_OP_SCF_FOR, LOOM_SPIRV_INDEX_BOUNDARY_LOOP,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_NONE},
    {LOOM_OP_SCF_SELECT, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_CONSTANT, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_CAST, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_ASSUME, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_ADD, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_SUB, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_MUL, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_SCALE, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_DIV, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_REM, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_DIVREM, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_MIN, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_MAX, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_MADD, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_ANDI, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_ORI, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_XORI, LOOM_SPIRV_INDEX_BOUNDARY_ARITHMETIC,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_SHLI, LOOM_SPIRV_INDEX_BOUNDARY_SHIFT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_SHRSI, LOOM_SPIRV_INDEX_BOUNDARY_SHIFT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_SHRUI, LOOM_SPIRV_INDEX_BOUNDARY_SHIFT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_ROTLI, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_ROTRI, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_CTLZI, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_CTTZI, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_INDEX_CTPOPI, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_KERNEL_WORKITEM_ID, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_KERNEL_WORKGROUP_ID, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_KERNEL_WORKGROUP_COUNT, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_KERNEL_SUBGROUP_ID, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_KERNEL_SUBGROUP_COUNT, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_KERNEL_SUBGROUP_SIZE, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
    {LOOM_OP_KERNEL_SUBGROUP_LANE_ID, LOOM_SPIRV_INDEX_BOUNDARY_RESULT,
     LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS},
};
static const loom_low_lower_representation_boundary_span_t
    kIndexSpans[LOOM_DIALECT_KERNEL - LOOM_DIALECT_SCF + 1] = {
        [LOOM_DIALECT_SCF - LOOM_DIALECT_SCF] = {0, 2},
        [LOOM_DIALECT_INDEX - LOOM_DIALECT_SCF] = {2, 24},
        [LOOM_DIALECT_KERNEL - LOOM_DIALECT_SCF] = {26, 8},
};
static const loom_low_lower_representation_provider_t kIndexProvider = {
    .relation = loom_spirv_index_relation,
    .observe_boundary = loom_spirv_index_observe_boundary,
    .observe_unclaimed_operand = loom_spirv_index_observe_operand,
    .boundaries = kIndexBoundaries,
    .boundary_spans = kIndexSpans,
    .boundary_count = IREE_ARRAYSIZE(kIndexBoundaries),
    .boundary_dialect_base_id = LOOM_DIALECT_SCF,
    .boundary_dialect_count = IREE_ARRAYSIZE(kIndexSpans),
    .relation_mask = LOOM_VALUE_RELATION_MASK_ALL,
};
const loom_low_lower_source_plan_observer_t
    loom_spirv_index_representation_observer = {
        .begin = loom_low_lower_representation_observer_begin,
        .observe = loom_low_lower_representation_observer_observe,
        .end = loom_low_lower_representation_observer_end,
        .user_data = (void*)&kIndexProvider,
};

iree_status_t loom_spirv_map_index_value(loom_low_lower_context_t* context,
                                         loom_value_id_t value,
                                         loom_type_t* out_type) {
  loom_low_representation_id_t representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_lookup_if_ready(context, value,
                                                &representation);
  const bool wide = representation == LOOM_SPIRV_INDEX_REPRESENTATION_WIDE ||
                    loom_spirv_index_needs_wide(loom_value_fact_table_lookup(
                        loom_low_lower_context_fact_table(context), value));
  return loom_low_lower_make_typed_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1,
      loom_type_scalar(wide ? LOOM_SCALAR_TYPE_I64 : LOOM_SCALAR_TYPE_INDEX),
      out_type);
}

static uint32_t loom_spirv_index_carrier_bitwidth(loom_type_t type) {
  if (loom_low_register_type_class_id(type) !=
          SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID ||
      loom_low_register_type_unit_count(type) != 1) {
    return 0;
  }
  const loom_type_t* payload = loom_type_register_value_type(type);
  if (payload == NULL || !loom_type_is_scalar(*payload)) {
    return 0;
  }
  switch (loom_type_element_type(*payload)) {
    case LOOM_SCALAR_TYPE_INDEX:
      return 32;
    case LOOM_SCALAR_TYPE_I64:
      return 64;
    default:
      return 0;
  }
}

loom_type_t loom_spirv_join_index_type(loom_type_t source_type, loom_type_t lhs,
                                       loom_type_t rhs) {
  if (!loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
    return loom_type_none();
  }
  const uint32_t left_width = loom_spirv_index_carrier_bitwidth(lhs);
  const uint32_t right_width = loom_spirv_index_carrier_bitwidth(rhs);
  if (left_width == 0 || right_width == 0) {
    return loom_type_none();
  }
  return right_width > left_width ? rhs : lhs;
}
