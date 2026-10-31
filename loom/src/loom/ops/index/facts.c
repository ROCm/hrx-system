// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the index dialect.

#include "loom/ir/facts.h"

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/cast.h"
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/util/fact_relation.h"
#include "loom/util/fact_table.h"
#include "loom/util/predicate_facts.h"

#define LOOM_INDEX_BINARY_FACTS(name, transfer_fn)                       \
  iree_status_t name(loom_fact_context_t* context,                       \
                     const loom_module_t* module, const loom_op_t* op,   \
                     const loom_value_facts_t* operand_facts,            \
                     loom_value_facts_t* result_facts) {                 \
    transfer_fn(&operand_facts[0], &operand_facts[1], &result_facts[0]); \
    return iree_ok_status();                                             \
  }

iree_status_t loom_index_constant_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_attr_as_i64(loom_index_constant_value(op)));
  return iree_ok_status();
}

static bool loom_index_cast_scalar_type(const loom_module_t* module,
                                        loom_value_id_t value,
                                        loom_scalar_type_t* out_scalar_type) {
  loom_type_t type = loom_module_value_type(module, value);
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  *out_scalar_type = loom_type_element_type(type);
  return true;
}

static void loom_index_wrap_offset_result_facts(
    const loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, loom_value_facts_t* facts) {
  int32_t bit_count =
      loom_index_target_carrier_bitwidth(context, LOOM_SCALAR_TYPE_OFFSET);
  if (bit_count == 0) {
    bit_count = 64;
  }
  // Most address arithmetic remains inside every possible target carrier.
  // Prove that before loading the result's segmented type record; only a
  // mathematical result that may wrap needs to distinguish index from offset.
  if (bit_count > 0 &&
      loom_value_facts_unsigned_wrap_is_identity(*facts, (uint8_t)bit_count)) {
    return;
  }

  const loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  if (loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_OFFSET) {
    return;
  }

  *facts = loom_index_wrap_offset_facts(context, *facts);
}

iree_status_t loom_index_cast_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  loom_scalar_type_t input_scalar_type = LOOM_SCALAR_TYPE_NONE;
  loom_scalar_type_t result_scalar_type = LOOM_SCALAR_TYPE_NONE;
  if (!loom_index_cast_scalar_type(module, loom_index_cast_input(op),
                                   &input_scalar_type) ||
      !loom_index_cast_scalar_type(module, loom_index_cast_result(op),
                                   &result_scalar_type)) {
    result_facts[0] = loom_value_facts_unknown();
    return loom_value_fact_table_define_exact_scalar_origin(
        context->table, loom_index_cast_result(op),
        (loom_value_fact_exact_scalar_origin_t){
            .source_value_id = LOOM_VALUE_ID_INVALID,
        });
  }

  result_facts[0] = loom_index_cast_transfer_facts(
      input_scalar_type, result_scalar_type, operand_facts[0]);
  loom_value_fact_exact_scalar_origin_t exact_origin = {
      .source_value_id = LOOM_VALUE_ID_INVALID,
  };
  if (loom_index_cast_transfer_preserves_value(
          input_scalar_type, result_scalar_type, operand_facts[0])) {
    const loom_value_id_t input = loom_index_cast_input(op);
    if (!loom_value_fact_table_query_exact_scalar_origin(
            context->table, module, input, &exact_origin)) {
      exact_origin.source_value_id =
          loom_value_fact_table_query_identity(context->table, input);
    }
  }
  return loom_value_fact_table_define_exact_scalar_origin(
      context->table, loom_index_cast_result(op), exact_origin);
}

iree_status_t loom_index_assume_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  uint16_t fact_count = op->operand_count < op->result_count ? op->operand_count
                                                             : op->result_count;
  for (uint16_t i = 0; i < fact_count; ++i) {
    result_facts[i] = operand_facts[i];
  }
  for (uint16_t i = fact_count; i < op->result_count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  loom_attribute_t pred_attr = loom_index_assume_predicates(op);
  const loom_predicate_t* predicates = pred_attr.predicate_list;
  uint16_t predicate_count = pred_attr.count;
  loom_value_slice_t values = loom_index_assume_values(op);
  return loom_value_fact_table_apply_alias_predicates(
      context->table, values.values, fact_count, predicates, predicate_count,
      result_facts);
}

iree_status_t loom_index_constexpr_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
  return iree_ok_status();
}

iree_status_t loom_index_add_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  loom_value_facts_addi(&operand_facts[0], &operand_facts[1], &result_facts[0]);
  loom_index_wrap_offset_result_facts(context, module, op, &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_index_sub_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  loom_value_facts_subi(&operand_facts[0], &operand_facts[1], &result_facts[0]);

  loom_value_id_t lhs = loom_index_sub_lhs(op);
  loom_value_id_t rhs = loom_index_sub_rhs(op);
  const loom_integer_order_outcome_bits_t order =
      loom_value_fact_table_query_signed_integer_order(context->table, lhs,
                                                       rhs);
  // Operand order describes the mathematical difference. Its sign survives
  // only when that ordered half of the difference cannot wrap the fact carrier.
  // The opposite endpoint may overflow: the relation excludes that half.
  int64_t difference_bound = 0;
  if (iree_checked_sub_i64(operand_facts[0].range_hi, operand_facts[1].range_lo,
                           &difference_bound)) {
    if ((order & (LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS |
                  LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL)) == 0) {
      result_facts[0].range_lo = iree_max(result_facts[0].range_lo, INT64_C(1));
    } else if ((order & LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS) == 0) {
      result_facts[0].range_lo = iree_max(result_facts[0].range_lo, INT64_C(0));
    }
  }
  if (iree_checked_sub_i64(operand_facts[0].range_lo, operand_facts[1].range_hi,
                           &difference_bound)) {
    if ((order & (LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL |
                  LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER)) == 0) {
      result_facts[0].range_hi =
          iree_min(result_facts[0].range_hi, INT64_C(-1));
    } else if ((order & LOOM_INTEGER_ORDER_OUTCOME_BIT_GREATER) == 0) {
      result_facts[0].range_hi = iree_min(result_facts[0].range_hi, INT64_C(0));
    }
  }
  loom_value_facts_recompute_flags(&result_facts[0]);
  loom_index_wrap_offset_result_facts(context, module, op, &result_facts[0]);
  return iree_ok_status();
}

LOOM_INDEX_BINARY_FACTS(loom_index_mul_facts, loom_value_facts_muli)
iree_status_t loom_index_scale_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_muli(&operand_facts[0], &operand_facts[1], &result_facts[0]);
  loom_index_wrap_offset_result_facts(context, module, op, &result_facts[0]);
  return iree_ok_status();
}

static void loom_index_div_result_facts(const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_positive(operand_facts[1])) {
    *result_facts = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(
        operand_facts[0], operand_facts[1], result_facts);
    return;
  }
  loom_value_facts_divui(&operand_facts[0], &operand_facts[1], 64,
                         result_facts);
  // The raw unsigned transfer establishes divisibility independently of the
  // range. Keep that fact, but do not expose a raw-bit range for an index whose
  // required nonnegative logical-coordinate domain has not been proven.
  if (!loom_value_facts_is_non_negative(operand_facts[0])) {
    const int64_t divisor = result_facts->known_divisor;
    *result_facts = loom_value_facts_make(INT64_MIN, INT64_MAX, divisor);
    loom_value_facts_propagate_binary_distribution(
        operand_facts[0], operand_facts[1], result_facts);
  }
}

iree_status_t loom_index_div_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  loom_index_div_result_facts(operand_facts, &result_facts[0]);
  return iree_ok_status();
}

static void loom_index_rem_result_facts(const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_non_negative(operand_facts[0]) ||
      !loom_value_facts_is_positive(operand_facts[1])) {
    *result_facts = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(
        operand_facts[0], operand_facts[1], result_facts);
    return;
  }
  loom_value_facts_remui(&operand_facts[0], &operand_facts[1], 64,
                         result_facts);
}

iree_status_t loom_index_rem_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  loom_index_rem_result_facts(operand_facts, &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_index_divrem_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_index_div_result_facts(operand_facts, &result_facts[0]);
  loom_index_rem_result_facts(operand_facts, &result_facts[1]);
  return iree_ok_status();
}

typedef void (*loom_index_minmax_transfer_fn_t)(const loom_value_facts_t* lhs,
                                                const loom_value_facts_t* rhs,
                                                int32_t bit_count,
                                                loom_value_facts_t* out_facts);

static iree_status_t loom_index_minmax_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_index_minmax_transfer_fn_t transfer_fn,
    loom_value_facts_t* result_facts) {
  if (loom_index_value_facts_fit_signed_target_carrier(
          context, LOOM_SCALAR_TYPE_INDEX, operand_facts[0]) &&
      loom_index_value_facts_fit_signed_target_carrier(
          context, LOOM_SCALAR_TYPE_INDEX, operand_facts[1])) {
    transfer_fn(&operand_facts[0], &operand_facts[1], 64, &result_facts[0]);
    return iree_ok_status();
  }

  result_facts[0] = loom_value_facts_make_signed_bit_count_range(
      loom_index_target_carrier_bitwidth(context, LOOM_SCALAR_TYPE_INDEX));
  loom_value_facts_propagate_binary_distribution(
      operand_facts[0], operand_facts[1], &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_index_min_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  return loom_index_minmax_facts(context, operand_facts, loom_value_facts_minsi,
                                 result_facts);
}

iree_status_t loom_index_max_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  return loom_index_minmax_facts(context, operand_facts, loom_value_facts_maxsi,
                                 result_facts);
}

iree_status_t loom_index_madd_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  loom_value_facts_fmai(&operand_facts[0], &operand_facts[1], &operand_facts[2],
                        &result_facts[0]);
  return iree_ok_status();
}

LOOM_INDEX_BINARY_FACTS(loom_index_andi_facts, loom_value_facts_andi)
LOOM_INDEX_BINARY_FACTS(loom_index_ori_facts, loom_value_facts_ori)
LOOM_INDEX_BINARY_FACTS(loom_index_xori_facts, loom_value_facts_xori)
LOOM_INDEX_BINARY_FACTS(loom_index_shli_facts, loom_value_facts_shli)
LOOM_INDEX_BINARY_FACTS(loom_index_shrsi_facts, loom_value_facts_shrsi)
iree_status_t loom_index_shrui_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  int32_t bit_count =
      loom_index_target_carrier_bitwidth(context, LOOM_SCALAR_TYPE_INDEX);
  // Non-negative mathematical values shift identically at every carrier width
  // that can represent them. Signed bit patterns require a selected carrier.
  if (bit_count == 0 && operand_facts[0].range_lo >= 0) {
    bit_count = 64;
  }
  if (bit_count <= 0) {
    result_facts[0] = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(
        operand_facts[0], operand_facts[1], &result_facts[0]);
    return iree_ok_status();
  }
  loom_value_facts_shrui(&operand_facts[0], &operand_facts[1], bit_count,
                         &result_facts[0]);
  return iree_ok_status();
}

static iree_status_t loom_index_rotate_facts(
    const loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    bool rotate_left, loom_value_facts_t* result_facts) {
  int32_t bitwidth =
      loom_index_target_carrier_bitwidth(context, LOOM_SCALAR_TYPE_INDEX);
  if (rotate_left) {
    loom_value_facts_rotli(&operand_facts[0], &operand_facts[1], bitwidth,
                           &result_facts[0]);
  } else {
    loom_value_facts_rotri(&operand_facts[0], &operand_facts[1], bitwidth,
                           &result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_index_rotli_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_index_rotate_facts(context, operand_facts,
                                 /*rotate_left=*/true, result_facts);
}

iree_status_t loom_index_rotri_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_index_rotate_facts(context, operand_facts,
                                 /*rotate_left=*/false, result_facts);
}

#define LOOM_INDEX_BIT_COUNT_FACTS(name, fn)                                 \
  iree_status_t name(loom_fact_context_t* context,                           \
                     const loom_module_t* module, const loom_op_t* op,       \
                     const loom_value_facts_t* operand_facts,                \
                     loom_value_facts_t* result_facts) {                     \
    int64_t value = 0;                                                       \
    int32_t bitwidth =                                                       \
        loom_index_target_carrier_bitwidth(context, LOOM_SCALAR_TYPE_INDEX); \
    if (bitwidth == 0) {                                                     \
      result_facts[0] = loom_value_facts_make(                               \
          0, loom_scalar_type_bitwidth(LOOM_SCALAR_TYPE_INDEX), 1);          \
    } else if (bitwidth < 0) {                                               \
      result_facts[0] = loom_value_facts_unknown();                          \
    } else if (loom_value_facts_as_exact_i64(operand_facts[0], &value)) {    \
      result_facts[0] =                                                      \
          loom_value_facts_exact_i64(fn((uint64_t)value, bitwidth));         \
    } else {                                                                 \
      result_facts[0] = loom_value_facts_make(0, bitwidth, 1);               \
    }                                                                        \
    loom_value_facts_propagate_unary_distribution(operand_facts[0],          \
                                                  &result_facts[0]);         \
    return iree_ok_status();                                                 \
  }

LOOM_INDEX_BIT_COUNT_FACTS(loom_index_ctlzi_facts,
                           iree_math_count_leading_zeros_u64_width)
LOOM_INDEX_BIT_COUNT_FACTS(loom_index_cttzi_facts,
                           iree_math_count_trailing_zeros_u64_width)
LOOM_INDEX_BIT_COUNT_FACTS(loom_index_ctpopi_facts,
                           iree_math_count_ones_u64_width)

#undef LOOM_INDEX_BIT_COUNT_FACTS

static void loom_index_mark_compare_distribution(
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts) {
  loom_value_facts_propagate_binary_distribution(
      operand_facts[0], operand_facts[1], result_facts);
  if (loom_value_facts_is_lane_varying(*result_facts)) {
    loom_value_facts_mark_lane_predicate(result_facts);
  }
}

iree_status_t loom_index_cmp_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  if (op->operand_count >= 2 && op->attribute_count >= 1) {
    const loom_value_id_t lhs_identity = loom_value_fact_table_query_identity(
        context->table, loom_index_cmp_lhs(op));
    const loom_value_id_t rhs_identity = loom_value_fact_table_query_identity(
        context->table, loom_index_cmp_rhs(op));
    bool result = false;
    uint8_t predicate = loom_index_cmp_predicate(op);
    loom_scalar_type_t operand_scalar_type = LOOM_SCALAR_TYPE_NONE;
    if ((lhs_identity == rhs_identity &&
         loom_index_cmp_same_value_result(predicate, &result)) ||
        (loom_index_cast_scalar_type(module, loom_index_cmp_lhs(op),
                                     &operand_scalar_type) &&
         loom_index_cmp_result_from_value_facts(
             context, module, operand_scalar_type, predicate,
             loom_index_cmp_lhs(op), &operand_facts[0], loom_index_cmp_rhs(op),
             &operand_facts[1], &result))) {
      result_facts[0] = loom_value_facts_exact_i64(result ? 1 : 0);
      return iree_ok_status();
    }
  }
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  loom_index_mark_compare_distribution(operand_facts, &result_facts[0]);
  return iree_ok_status();
}

#undef LOOM_INDEX_BINARY_FACTS
