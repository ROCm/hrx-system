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
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/util/math.h"

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
        loom_gcd_i64(facts.known_divisor, unsigned_extent));
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
  for (uint16_t predicate_ordinal = 0; predicate_ordinal < predicate_count;
       ++predicate_ordinal) {
    const loom_predicate_t* predicate = &predicates[predicate_ordinal];
    if (predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) continue;
    loom_value_slice_t values = loom_index_assume_values(op);
    loom_value_id_t target_id = (loom_value_id_t)predicate->args[0];
    uint16_t target = 0;
    bool found = false;
    for (uint16_t i = 0; i < values.count; ++i) {
      if (values.values[i] == target_id) {
        target = i;
        found = true;
        break;
      }
    }
    if (!found) continue;
    if (target < fact_count) {
      loom_value_facts_apply_predicate(&result_facts[target], predicate);
    }
  }
  return iree_ok_status();
}

LOOM_INDEX_BINARY_FACTS(loom_index_add_facts, loom_value_facts_addi)
LOOM_INDEX_BINARY_FACTS(loom_index_sub_facts, loom_value_facts_subi)
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

LOOM_INDEX_BINARY_FACTS(loom_index_min_facts, loom_value_facts_minsi)
LOOM_INDEX_BINARY_FACTS(loom_index_max_facts, loom_value_facts_maxsi)

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

static int32_t loom_index_result_bitwidth(const loom_module_t* module,
                                          loom_value_id_t result) {
  loom_type_t result_type = loom_module_value_type(module, result);
  return loom_scalar_type_bitwidth(loom_type_element_type(result_type));
}

static iree_status_t loom_index_rotate_facts(
    const loom_module_t* module, const loom_value_facts_t* operand_facts,
    bool rotate_left, loom_value_id_t result,
    loom_value_facts_t* result_facts) {
  int64_t value = 0;
  int64_t amount = 0;
  int32_t bitwidth = loom_index_result_bitwidth(module, result);
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
    result_facts[0] = loom_value_facts_exact_i64((int64_t)raw_value);
    return iree_ok_status();
  }
  uint64_t rotated =
      rotate_left
          ? ((raw_value << shift) | (raw_value >> ((uint32_t)bitwidth - shift)))
          : ((raw_value >> shift) |
             (raw_value << ((uint32_t)bitwidth - shift)));
  result_facts[0] = loom_value_facts_exact_i64((int64_t)(rotated & mask));
  return iree_ok_status();
}

iree_status_t loom_index_rotli_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_index_rotate_facts(module, operand_facts, /*rotate_left=*/true,
                                 loom_index_rotli_result(op), result_facts);
}

iree_status_t loom_index_rotri_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_index_rotate_facts(module, operand_facts, /*rotate_left=*/false,
                                 loom_index_rotri_result(op), result_facts);
}

#define LOOM_INDEX_BIT_COUNT_FACTS(name, result_accessor, fn)          \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    int64_t value = 0;                                                 \
    int32_t bitwidth =                                                 \
        loom_index_result_bitwidth(module, result_accessor(op));       \
    if (!loom_value_facts_as_exact_i64(operand_facts[0], &value) ||    \
        bitwidth <= 0) {                                               \
      result_facts[0] = loom_value_facts_unknown();                    \
      return iree_ok_status();                                         \
    }                                                                  \
    result_facts[0] =                                                  \
        loom_value_facts_exact_i64(fn((uint64_t)value, bitwidth));     \
    return iree_ok_status();                                           \
  }

LOOM_INDEX_BIT_COUNT_FACTS(loom_index_ctlzi_facts, loom_index_ctlzi_result,
                           loom_count_leading_zeros_u64_width)
LOOM_INDEX_BIT_COUNT_FACTS(loom_index_cttzi_facts, loom_index_cttzi_result,
                           loom_count_trailing_zeros_u64_width)
LOOM_INDEX_BIT_COUNT_FACTS(loom_index_ctpopi_facts, loom_index_ctpopi_result,
                           loom_count_ones_u64_width)

#undef LOOM_INDEX_BIT_COUNT_FACTS

static void loom_index_mark_compare_distribution(
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_lane_predicate(operand_facts[0]) ||
      loom_value_facts_is_lane_predicate(operand_facts[1]) ||
      loom_value_facts_is_lane_varying(operand_facts[0]) ||
      loom_value_facts_is_lane_varying(operand_facts[1])) {
    loom_value_facts_mark_lane_predicate(result_facts);
  } else if (loom_value_facts_is_uniform(operand_facts[0]) &&
             loom_value_facts_is_uniform(operand_facts[1])) {
    loom_value_facts_mark_uniform(result_facts);
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
    if ((loom_index_cmp_lhs(op) == loom_index_cmp_rhs(op) &&
         loom_index_cmp_same_value_result(predicate, &result)) ||
        loom_index_cmp_result_from_facts(predicate, &operand_facts[0],
                                         &operand_facts[1], &result)) {
      result_facts[0] = loom_value_facts_exact_i64(result ? 1 : 0);
      return iree_ok_status();
    }
  }
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  loom_index_mark_compare_distribution(operand_facts, &result_facts[0]);
  return iree_ok_status();
}

#undef LOOM_INDEX_BINARY_FACTS
