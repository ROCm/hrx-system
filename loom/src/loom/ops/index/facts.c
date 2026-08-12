// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the index dialect.

#include "loom/ir/facts.h"

#include "iree/base/internal/math.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/util/fact_table.h"

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
  if (!loom_type_is_scalar(type)) return false;
  *out_scalar_type = loom_type_element_type(type);
  return true;
}

static bool loom_index_cast_to_offset_zero_extends_source(
    loom_scalar_type_t input_scalar_type, loom_scalar_type_t result_scalar_type,
    int32_t input_bitwidth) {
  return result_scalar_type == LOOM_SCALAR_TYPE_OFFSET &&
         loom_scalar_type_is_integer(input_scalar_type) && input_bitwidth < 63;
}

static loom_value_facts_t loom_index_cast_zero_extend_to_offset_facts(
    loom_value_facts_t facts, int32_t input_bitwidth) {
  const int64_t unsigned_extent = INT64_C(1) << input_bitwidth;
  const int64_t unsigned_max = unsigned_extent - 1;
  if (facts.range_lo >= 0) {
    return loom_value_facts_clamp_domain(facts, 0, unsigned_max);
  }
  if (facts.range_hi < 0) {
    return loom_value_facts_make(
        facts.range_lo + unsigned_extent, facts.range_hi + unsigned_extent,
        iree_math_gcd_i64(facts.known_divisor, unsigned_extent));
  }
  return loom_value_facts_make(0, unsigned_max, 1);
}

iree_status_t loom_index_cast_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  loom_scalar_type_t input_scalar_type = LOOM_SCALAR_TYPE_COUNT_;
  loom_scalar_type_t result_scalar_type = LOOM_SCALAR_TYPE_COUNT_;
  if (!loom_index_cast_scalar_type(module, loom_index_cast_input(op),
                                   &input_scalar_type) ||
      !loom_index_cast_scalar_type(module, loom_index_cast_result(op),
                                   &result_scalar_type)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  int64_t input_lo = 0;
  int64_t input_hi = 0;
  int64_t result_lo = 0;
  int64_t result_hi = 0;
  if (!loom_value_facts_scalar_type_domain(input_scalar_type, &input_lo,
                                           &input_hi) ||
      !loom_value_facts_scalar_type_domain(result_scalar_type, &result_lo,
                                           &result_hi)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  int32_t input_bitwidth = loom_scalar_type_bitwidth(input_scalar_type);
  int32_t result_bitwidth = loom_scalar_type_bitwidth(result_scalar_type);
  loom_value_facts_t facts =
      loom_value_facts_clamp_domain(operand_facts[0], input_lo, input_hi);

  if (loom_index_cast_to_offset_zero_extends_source(
          input_scalar_type, result_scalar_type, input_bitwidth)) {
    result_facts[0] =
        loom_index_cast_zero_extend_to_offset_facts(facts, input_bitwidth);
    return iree_ok_status();
  }

  if (input_bitwidth <= result_bitwidth) {
    result_facts[0] =
        loom_value_facts_clamp_domain(facts, result_lo, result_hi);
    return iree_ok_status();
  }

  if (facts.range_lo >= result_lo && facts.range_hi <= result_hi) {
    result_facts[0] = facts;
    return iree_ok_status();
  }

  // Truncation may wrap arbitrary inputs into any value in the result domain.
  // Preserve the original facts only when the source range already proves that
  // truncation is value-preserving.
  result_facts[0] = loom_value_facts_make(result_lo, result_hi, 1);
  return iree_ok_status();
}

static bool loom_index_assume_lookup_facts(void* user_data,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t* out_facts) {
  const loom_value_fact_table_t* fact_table =
      (const loom_value_fact_table_t*)user_data;
  if (!fact_table || !loom_value_fact_table_has_entry(fact_table, value_id)) {
    return false;
  }
  *out_facts = loom_value_fact_table_lookup(fact_table, value_id);
  return true;
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
  loom_attribute_t pred_attr = loom_op_attrs(op)[0];
  const loom_predicate_t* predicates = pred_attr.predicate_list;
  uint16_t predicate_count = pred_attr.count;
  loom_value_slice_t values = loom_index_assume_values(op);
  loom_value_facts_apply_alias_predicates(
      values.values, fact_count, predicates, predicate_count,
      loom_index_assume_lookup_facts, context ? context->table : NULL,
      result_facts);
  return iree_ok_status();
}

LOOM_INDEX_BINARY_FACTS(loom_index_add_facts, loom_value_facts_addi)

typedef enum loom_index_assumed_order_flag_bits_e {
  LOOM_INDEX_ASSUMED_ORDER_NONE = 0,
  LOOM_INDEX_ASSUMED_ORDER_LESS_OR_EQUAL = 1u << 0,
  LOOM_INDEX_ASSUMED_ORDER_LESS = 1u << 1,
  LOOM_INDEX_ASSUMED_ORDER_GREATER_OR_EQUAL = 1u << 2,
  LOOM_INDEX_ASSUMED_ORDER_GREATER = 1u << 3,
} loom_index_assumed_order_flag_bits_t;

typedef uint32_t loom_index_assumed_order_flags_t;

static bool loom_index_assume_source_for_result(const loom_module_t* module,
                                                loom_value_id_t value_id,
                                                const loom_op_t** out_assume_op,
                                                loom_value_id_t* out_source) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_index_assume_isa(defining_op)) return false;

  loom_value_slice_t sources = loom_index_assume_values(defining_op);
  uint16_t result_index = loom_value_def_index(value);
  if (result_index >= sources.count) return false;
  *out_assume_op = defining_op;
  *out_source = sources.values[result_index];
  return true;
}

static loom_value_id_t loom_index_strip_assume_chain(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_op_t* assume_op = NULL;
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  while (loom_index_assume_source_for_result(module, value_id, &assume_op,
                                             &source)) {
    value_id = source;
  }
  return value_id;
}

static loom_index_assumed_order_flags_t loom_index_assumed_order(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t other_value_id) {
  loom_index_assumed_order_flags_t flags = LOOM_INDEX_ASSUMED_ORDER_NONE;
  const loom_value_id_t other_source =
      loom_index_strip_assume_chain(module, other_value_id);
  const loom_op_t* assume_op = NULL;
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  while (loom_index_assume_source_for_result(module, value_id, &assume_op,
                                             &source)) {
    loom_attribute_t predicates_attr = loom_index_assume_predicates(assume_op);
    for (uint16_t i = 0; i < predicates_attr.count; ++i) {
      const loom_predicate_t* predicate = &predicates_attr.predicate_list[i];
      if (predicate->arg_count != 2 ||
          predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
          predicate->arg_tags[1] != LOOM_PRED_ARG_VALUE ||
          loom_index_strip_assume_chain(module,
                                        (loom_value_id_t)predicate->args[0]) !=
              loom_index_strip_assume_chain(module, source) ||
          loom_index_strip_assume_chain(
              module, (loom_value_id_t)predicate->args[1]) != other_source) {
        continue;
      }
      switch ((loom_predicate_kind_t)predicate->kind) {
        case LOOM_PREDICATE_EQ:
          flags |= LOOM_INDEX_ASSUMED_ORDER_LESS_OR_EQUAL |
                   LOOM_INDEX_ASSUMED_ORDER_GREATER_OR_EQUAL;
          break;
        case LOOM_PREDICATE_LT:
          flags |= LOOM_INDEX_ASSUMED_ORDER_LESS |
                   LOOM_INDEX_ASSUMED_ORDER_LESS_OR_EQUAL;
          break;
        case LOOM_PREDICATE_LE:
          flags |= LOOM_INDEX_ASSUMED_ORDER_LESS_OR_EQUAL;
          break;
        case LOOM_PREDICATE_GT:
          flags |= LOOM_INDEX_ASSUMED_ORDER_GREATER |
                   LOOM_INDEX_ASSUMED_ORDER_GREATER_OR_EQUAL;
          break;
        case LOOM_PREDICATE_GE:
          flags |= LOOM_INDEX_ASSUMED_ORDER_GREATER_OR_EQUAL;
          break;
        default:
          break;
      }
    }
    value_id = source;
  }
  return flags;
}

iree_status_t loom_index_sub_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  loom_value_facts_subi(&operand_facts[0], &operand_facts[1], &result_facts[0]);

  loom_value_id_t lhs = loom_index_sub_lhs(op);
  loom_value_id_t rhs = loom_index_sub_rhs(op);
  loom_index_assumed_order_flags_t lhs_order =
      loom_index_assumed_order(module, lhs, rhs);
  loom_index_assumed_order_flags_t rhs_order =
      loom_index_assumed_order(module, rhs, lhs);
  if (iree_any_bit_set(lhs_order, LOOM_INDEX_ASSUMED_ORDER_GREATER) ||
      iree_any_bit_set(rhs_order, LOOM_INDEX_ASSUMED_ORDER_LESS)) {
    result_facts[0].range_lo = iree_max(result_facts[0].range_lo, INT64_C(1));
  } else if (iree_any_bit_set(lhs_order,
                              LOOM_INDEX_ASSUMED_ORDER_GREATER_OR_EQUAL) ||
             iree_any_bit_set(rhs_order,
                              LOOM_INDEX_ASSUMED_ORDER_LESS_OR_EQUAL)) {
    result_facts[0].range_lo = iree_max(result_facts[0].range_lo, INT64_C(0));
  }
  if (iree_any_bit_set(lhs_order, LOOM_INDEX_ASSUMED_ORDER_LESS) ||
      iree_any_bit_set(rhs_order, LOOM_INDEX_ASSUMED_ORDER_GREATER)) {
    result_facts[0].range_hi = iree_min(result_facts[0].range_hi, INT64_C(-1));
  } else if (iree_any_bit_set(lhs_order,
                              LOOM_INDEX_ASSUMED_ORDER_LESS_OR_EQUAL) ||
             iree_any_bit_set(rhs_order,
                              LOOM_INDEX_ASSUMED_ORDER_GREATER_OR_EQUAL)) {
    result_facts[0].range_hi = iree_min(result_facts[0].range_hi, INT64_C(0));
  }
  loom_value_facts_recompute_flags(&result_facts[0]);
  return iree_ok_status();
}

LOOM_INDEX_BINARY_FACTS(loom_index_mul_facts, loom_value_facts_muli)
LOOM_INDEX_BINARY_FACTS(loom_index_scale_facts, loom_value_facts_muli)
LOOM_INDEX_BINARY_FACTS(loom_index_div_facts, loom_value_facts_divui)

iree_status_t loom_index_rem_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_float(operand_facts[0]) &&
      !loom_value_facts_is_float(operand_facts[1]) &&
      loom_value_facts_is_exact(operand_facts[0]) &&
      operand_facts[0].range_lo == 0 &&
      loom_value_facts_is_positive(operand_facts[1])) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }
  loom_value_facts_remui(&operand_facts[0], &operand_facts[1],
                         &result_facts[0]);
  return iree_ok_status();
}

typedef void (*loom_index_minmax_transfer_fn_t)(const loom_value_facts_t* lhs,
                                                const loom_value_facts_t* rhs,
                                                loom_value_facts_t* out_facts);

static iree_status_t loom_index_minmax_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_index_minmax_transfer_fn_t transfer_fn,
    loom_value_facts_t* result_facts) {
  if (loom_index_value_facts_fit_signed_target_carrier(
          context, LOOM_SCALAR_TYPE_INDEX, operand_facts[0]) &&
      loom_index_value_facts_fit_signed_target_carrier(
          context, LOOM_SCALAR_TYPE_INDEX, operand_facts[1])) {
    transfer_fn(&operand_facts[0], &operand_facts[1], &result_facts[0]);
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
LOOM_INDEX_BINARY_FACTS(loom_index_shrui_facts, loom_value_facts_shrui)

static iree_status_t loom_index_rotate_facts(
    const loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    bool rotate_left, loom_value_facts_t* result_facts) {
  int64_t value = 0;
  int64_t amount = 0;
  int32_t bitwidth =
      loom_index_target_carrier_bitwidth(context, LOOM_SCALAR_TYPE_INDEX);
  if (!loom_value_facts_as_exact_i64(operand_facts[0], &value) ||
      !loom_value_facts_as_exact_i64(operand_facts[1], &amount) ||
      bitwidth <= 0 || bitwidth > 64 || amount < 0 || amount >= bitwidth) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  uint64_t mask = bitwidth == 64 ? UINT64_MAX : ((UINT64_C(1) << bitwidth) - 1);
  uint64_t raw_value = (uint64_t)value & mask;
  uint32_t shift = (uint32_t)amount;
  if (shift == 0) {
    result_facts[0] =
        loom_value_facts_make_signed_raw_bits(raw_value, bitwidth);
    return iree_ok_status();
  }
  uint64_t rotated =
      rotate_left
          ? ((raw_value << shift) | (raw_value >> ((uint32_t)bitwidth - shift)))
          : ((raw_value >> shift) |
             (raw_value << ((uint32_t)bitwidth - shift)));
  result_facts[0] =
      loom_value_facts_make_signed_raw_bits(rotated & mask, bitwidth);
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
    bool result = false;
    uint8_t predicate = loom_index_cmp_predicate(op);
    loom_scalar_type_t operand_scalar_type = LOOM_SCALAR_TYPE_COUNT_;
    if ((loom_index_cmp_lhs(op) == loom_index_cmp_rhs(op) &&
         loom_index_cmp_same_value_result(predicate, &result)) ||
        (loom_index_cast_scalar_type(module, loom_index_cmp_lhs(op),
                                     &operand_scalar_type) &&
         loom_index_cmp_result_from_facts(context, operand_scalar_type,
                                          predicate, &operand_facts[0],
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
