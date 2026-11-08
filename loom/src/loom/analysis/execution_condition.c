// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/execution_condition.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/facts.h"
#include "loom/util/fact_extensions.h"
#include "loom/util/fact_relation.h"
#include "loom/util/fact_table.h"

typedef struct loom_execution_integer_facts_t {
  // Inclusive scalar or aggregate-element range.
  loom_value_facts_t bounds;
  // True when the aggregate has no elements and cannot execute an operation.
  bool empty;
} loom_execution_integer_facts_t;

static bool loom_execution_query_integer_facts(
    const loom_value_fact_table_t* fact_table, loom_type_t type,
    loom_value_id_t value_id, loom_execution_integer_facts_t* out_facts) {
  *out_facts = (loom_execution_integer_facts_t){0};
  if (loom_type_is_scalar(type)) {
    out_facts->bounds = loom_value_fact_table_lookup(fact_table, value_id);
    return true;
  }

  IREE_ASSERT(loom_type_is_vector(type));
  const uint64_t maximum_element_count =
      loom_value_fact_table_maximum_element_count(fact_table, type);
  if (maximum_element_count == 0) {
    out_facts->empty = true;
    return true;
  }

  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, value_id);
  loom_value_fact_uniform_element_t uniform;
  if (loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                             &uniform)) {
    // A nonzero element can span both signs. Keep its predicates instead of
    // flattening them to an interval that includes zero.
    out_facts->bounds = uniform.element;
    return true;
  }

  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  if (!loom_value_facts_query_vector_integer_bounds(
          &fact_table->context, facts, maximum_element_count, &lower_bound,
          &upper_bound)) {
    return false;
  }
  if (lower_bound > upper_bound) {
    out_facts->empty = true;
    return true;
  }
  out_facts->bounds =
      loom_value_facts_make(lower_bound, upper_bound, /*known_divisor=*/1);
  return true;
}

static bool loom_execution_integer_excludes_literal(
    const loom_value_fact_table_t* fact_table, loom_type_t type,
    loom_value_id_t value, int64_t literal) {
  if (loom_type_is_scalar(type)) {
    return !(loom_value_fact_table_query_signed_integer_literal_order(
                 fact_table, value, literal) &
             LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL);
  }
  loom_execution_integer_facts_t facts;
  if (!loom_execution_query_integer_facts(fact_table, type, value, &facts)) {
    return false;
  }
  return facts.empty || literal < facts.bounds.range_lo ||
         literal > facts.bounds.range_hi ||
         (literal == 0 && loom_value_facts_is_non_zero(facts.bounds));
}

static bool loom_execution_integer_division_is_safe(
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_fact_table_t* fact_table,
    loom_execution_condition_t condition) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_type_t type = loom_module_value_type(module, operands[0]);
  if (condition == LOOM_EXECUTION_CONDITION_INDEX_DIVISION) {
    const loom_value_facts_t lhs =
        loom_value_fact_table_lookup(fact_table, operands[0]);
    const loom_value_facts_t rhs =
        loom_value_fact_table_lookup(fact_table, operands[1]);
    return lhs.range_lo >= 0 && rhs.range_lo >= 0 &&
           loom_value_facts_is_non_zero(rhs);
  }
  if (!loom_execution_integer_excludes_literal(fact_table, type, operands[1],
                                               0)) {
    return false;
  }
  if (condition == LOOM_EXECUTION_CONDITION_INTEGER_NONZERO_DIVISOR) {
    return true;
  }

  const uint32_t bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  // Boolean facts describe raw one-bit payloads as [0, 1]. Both signed MIN
  // and -1 have the raw value 1 in that domain.
  const int64_t minimum = bit_count == 1    ? 1
                          : bit_count == 64 ? INT64_MIN
                                            : -(INT64_C(1) << (bit_count - 1));
  const int64_t minus_one = bit_count == 1 ? 1 : -1;
  return loom_execution_integer_excludes_literal(fact_table, type, operands[0],
                                                 minimum) ||
         loom_execution_integer_excludes_literal(fact_table, type, operands[1],
                                                 minus_one);
}

static bool loom_execution_integer_shift_is_safe(
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_fact_table_t* fact_table,
    loom_execution_condition_t condition) {
  const loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  const bool is_index = element_type == LOOM_SCALAR_TYPE_INDEX;

  uint32_t bit_count = 0;
  if (is_index) {
    if (!fact_table->context.target_facts) {
      return false;
    }
    bit_count =
        fact_table->context.target_facts->storage.snapshot.index_bitwidth;
  } else {
    IREE_ASSERT(loom_scalar_type_is_integer(element_type));
    bit_count = (uint32_t)loom_scalar_type_bitwidth(element_type);
  }
  if (bit_count == 0 || bit_count > 64) {
    return false;
  }

  loom_execution_integer_facts_t count_facts;
  if (!loom_execution_query_integer_facts(fact_table, result_type,
                                          loom_op_const_operands(op)[1],
                                          &count_facts)) {
    return false;
  }
  if (count_facts.empty) {
    return true;
  }
  if (count_facts.bounds.range_lo < 0 ||
      (uint64_t)count_facts.bounds.range_hi >= bit_count) {
    return false;
  }
  if (condition == LOOM_EXECUTION_CONDITION_INTEGER_SHIFT_RIGHT) {
    return true;
  }

  const uint8_t maximum_shift = (uint8_t)count_facts.bounds.range_hi;
  if (maximum_shift == 0) {
    return true;
  }
  const uint8_t remaining_bit_count = (uint8_t)(bit_count - maximum_shift);

  if (!is_index && op->instance_flags == 0) {
    return true;
  }

  loom_execution_integer_facts_t input_facts;
  if (!loom_execution_query_integer_facts(fact_table, result_type,
                                          loom_op_const_operands(op)[0],
                                          &input_facts)) {
    return false;
  }
  if (input_facts.empty) {
    return true;
  }
  if (is_index) {
    return loom_value_facts_fit_signed_bit_count(input_facts.bounds,
                                                 remaining_bit_count);
  }

  if (iree_any_bit_set(op->instance_flags, LOOM_INTEGER_OVERFLOW_FLAG_NSW) &&
      !loom_value_facts_fit_signed_bit_count(input_facts.bounds,
                                             remaining_bit_count)) {
    return false;
  }
  if (iree_any_bit_set(op->instance_flags, LOOM_INTEGER_OVERFLOW_FLAG_NUW) &&
      !loom_value_facts_fit_unsigned_bit_count(input_facts.bounds,
                                               remaining_bit_count)) {
    return false;
  }
  return true;
}

static bool loom_execution_signed_arithmetic_bounds(
    loom_execution_condition_t condition,
    const loom_execution_integer_facts_t* operand_facts, uint8_t operand_count,
    bool is_i1, int64_t* out_lower, int64_t* out_upper) {
  loom_value_facts_t bounds[3];
  for (uint8_t i = 0; i < operand_count; ++i) {
    bounds[i] = operand_facts[i].bounds;
    if (is_i1) {
      // The fact lattice represents Boolean payloads numerically as [0, 1],
      // while signed one-bit arithmetic interprets raw 1 as -1.
      bounds[i].range_lo = -operand_facts[i].bounds.range_hi;
      bounds[i].range_hi = -operand_facts[i].bounds.range_lo;
    }
  }

  int64_t lower = 0;
  int64_t upper = 0;
  switch (condition) {
    case LOOM_EXECUTION_CONDITION_INTEGER_ADD:
      if (!iree_checked_add_i64(bounds[0].range_lo, bounds[1].range_lo,
                                &lower) ||
          !iree_checked_add_i64(bounds[0].range_hi, bounds[1].range_hi,
                                &upper)) {
        return false;
      }
      break;
    case LOOM_EXECUTION_CONDITION_INTEGER_SUBTRACT:
      if (!iree_checked_sub_i64(bounds[0].range_lo, bounds[1].range_hi,
                                &lower) ||
          !iree_checked_sub_i64(bounds[0].range_hi, bounds[1].range_lo,
                                &upper)) {
        return false;
      }
      break;
    case LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY:
    case LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD: {
      const int64_t products[][2] = {
          {bounds[0].range_lo, bounds[1].range_lo},
          {bounds[0].range_lo, bounds[1].range_hi},
          {bounds[0].range_hi, bounds[1].range_lo},
          {bounds[0].range_hi, bounds[1].range_hi},
      };
      if (!iree_checked_mul_i64(products[0][0], products[0][1], &lower)) {
        return false;
      }
      upper = lower;
      for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(products); ++i) {
        int64_t product = 0;
        if (!iree_checked_mul_i64(products[i][0], products[i][1], &product)) {
          return false;
        }
        lower = iree_min(lower, product);
        upper = iree_max(upper, product);
      }
      if (condition == LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD &&
          (!iree_checked_add_i64(lower, bounds[2].range_lo, &lower) ||
           !iree_checked_add_i64(upper, bounds[2].range_hi, &upper))) {
        return false;
      }
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("not an integer arithmetic condition");
      return false;
  }
  *out_lower = lower;
  *out_upper = upper;
  return true;
}

static bool loom_execution_unsigned_arithmetic_is_safe(
    loom_execution_condition_t condition,
    const loom_execution_integer_facts_t* operand_facts, uint8_t operand_count,
    uint8_t bit_count, loom_integer_order_outcome_bits_t subtraction_order) {
  typedef struct loom_execution_unsigned_bounds_t {
    uint64_t lower;
    uint64_t upper;
  } loom_execution_unsigned_bounds_t;

  const uint64_t maximum =
      bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << bit_count) - 1;
  loom_execution_unsigned_bounds_t bounds[3];
  for (uint8_t i = 0; i < operand_count; ++i) {
    const loom_value_facts_t facts = operand_facts[i].bounds;
    // Fact production clamps fixed-width values to their signed payload
    // domain. Raw unsigned bits stay ordered within either sign half; an
    // interval crossing zero spans both ends of the unsigned domain.
    if (facts.range_lo < 0 && facts.range_hi >= 0) {
      bounds[i] = (loom_execution_unsigned_bounds_t){0, maximum};
    } else {
      bounds[i] = (loom_execution_unsigned_bounds_t){
          (uint64_t)facts.range_lo & maximum,
          (uint64_t)facts.range_hi & maximum,
      };
    }
  }

  uint64_t result_upper = 0;
  switch (condition) {
    case LOOM_EXECUTION_CONDITION_INTEGER_ADD:
      return iree_checked_add_u64(bounds[0].upper, bounds[1].upper,
                                  &result_upper) &&
             result_upper <= maximum;
    case LOOM_EXECUTION_CONDITION_INTEGER_SUBTRACT: {
      if (bounds[0].lower >= bounds[1].upper) {
        return true;
      }
      // A retained signed order also proves unsigned order while both ranges
      // occupy the same signed half. Boolean facts use numeric [0, 1] instead
      // of signed one-bit [0, -1], so only direct bounds prove i1 order.
      const bool same_signed_half =
          bit_count > 1 && ((operand_facts[0].bounds.range_lo >= 0 &&
                             operand_facts[1].bounds.range_lo >= 0) ||
                            (operand_facts[0].bounds.range_hi < 0 &&
                             operand_facts[1].bounds.range_hi < 0));
      return same_signed_half &&
             !(subtraction_order & LOOM_INTEGER_ORDER_OUTCOME_BIT_LESS);
    }
    case LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY:
      return iree_checked_mul_u64(bounds[0].upper, bounds[1].upper,
                                  &result_upper) &&
             result_upper <= maximum;
    case LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD:
      return iree_checked_mul_u64(bounds[0].upper, bounds[1].upper,
                                  &result_upper) &&
             iree_checked_add_u64(result_upper, bounds[2].upper,
                                  &result_upper) &&
             result_upper <= maximum;
    default:
      IREE_ASSERT_UNREACHABLE("not an integer arithmetic condition");
      return false;
  }
}

static bool loom_execution_integer_arithmetic_is_safe(
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_fact_table_t* fact_table,
    loom_execution_condition_t condition) {
  const loom_type_t result_type =
      loom_module_value_type(module, loom_op_const_results(op)[0]);
  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  if (element_type == LOOM_SCALAR_TYPE_OFFSET) {
    return true;
  }

  uint32_t bit_count = 0;
  const bool is_index = element_type == LOOM_SCALAR_TYPE_INDEX;
  if (is_index) {
    if (!fact_table->context.target_facts) {
      return false;
    }
    bit_count =
        fact_table->context.target_facts->storage.snapshot.index_bitwidth;
  } else {
    IREE_ASSERT(loom_scalar_type_is_integer(element_type));
    bit_count = (uint32_t)loom_scalar_type_bitwidth(element_type);
  }
  if (bit_count == 0 || bit_count > 64) {
    return false;
  }

  const uint8_t operand_count =
      condition == LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD ? 3 : 2;
  loom_execution_integer_facts_t operand_facts[3];
  for (uint8_t i = 0; i < operand_count; ++i) {
    if (!loom_execution_query_integer_facts(fact_table, result_type,
                                            loom_op_const_operands(op)[i],
                                            &operand_facts[i])) {
      return false;
    }
    if (operand_facts[i].empty) {
      return true;
    }
  }

  loom_integer_order_outcome_bits_t subtraction_order =
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL;
  if (condition == LOOM_EXECUTION_CONDITION_INTEGER_SUBTRACT) {
    const loom_value_id_t* operands = loom_op_const_operands(op);
    if (loom_type_is_scalar(result_type)) {
      subtraction_order = loom_value_fact_table_query_signed_integer_order(
          fact_table, operands[0], operands[1]);
    }
    const bool operands_equal =
        subtraction_order == LOOM_INTEGER_ORDER_OUTCOME_BIT_EQUAL ||
        (!loom_type_is_scalar(result_type) &&
         loom_value_fact_table_values_equal(fact_table, operands[0],
                                            operands[1]));
    if (operands_equal) {
      return true;
    }
  }

  int64_t signed_lower = 0;
  int64_t signed_upper = 0;
  const bool signed_bounds_valid = loom_execution_signed_arithmetic_bounds(
      condition, operand_facts, operand_count,
      element_type == LOOM_SCALAR_TYPE_I1, &signed_lower, &signed_upper);
  if (is_index) {
    return signed_bounds_valid &&
           loom_value_facts_fit_signed_bit_count(
               loom_value_facts_make(signed_lower, signed_upper,
                                     /*known_divisor=*/1),
               (uint8_t)bit_count);
  }

  const uint8_t overflow_flags = op->instance_flags;
  if (iree_any_bit_set(overflow_flags, LOOM_INTEGER_OVERFLOW_FLAG_NSW)) {
    if (!signed_bounds_valid ||
        !loom_value_facts_fit_signed_bit_count(
            loom_value_facts_make(signed_lower, signed_upper,
                                  /*known_divisor=*/1),
            (uint8_t)bit_count)) {
      return false;
    }
  }
  if (iree_any_bit_set(overflow_flags, LOOM_INTEGER_OVERFLOW_FLAG_NUW) &&
      !loom_execution_unsigned_arithmetic_is_safe(
          condition, operand_facts, operand_count, (uint8_t)bit_count,
          subtraction_order)) {
    return false;
  }
  return true;
}

bool loom_execution_condition_is_satisfied(
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_fact_table_t* fact_table) {
  const loom_execution_condition_t condition =
      loom_op_semantics(module, op).execution_condition;
  if (condition == LOOM_EXECUTION_CONDITION_NONE || !fact_table) {
    return false;
  }
  IREE_ASSERT(!fact_table->has_conditioned_results);

  switch (condition) {
    case LOOM_EXECUTION_CONDITION_INTEGER_SHIFT_LEFT:
    case LOOM_EXECUTION_CONDITION_INTEGER_SHIFT_RIGHT:
      return loom_execution_integer_shift_is_safe(module, op, fact_table,
                                                  condition);
    case LOOM_EXECUTION_CONDITION_INTEGER_ADD:
    case LOOM_EXECUTION_CONDITION_INTEGER_SUBTRACT:
    case LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY:
    case LOOM_EXECUTION_CONDITION_INTEGER_MULTIPLY_ADD:
      return loom_execution_integer_arithmetic_is_safe(module, op, fact_table,
                                                       condition);
    case LOOM_EXECUTION_CONDITION_INTEGER_NONZERO_DIVISOR:
    case LOOM_EXECUTION_CONDITION_INTEGER_SIGNED_QUOTIENT:
    case LOOM_EXECUTION_CONDITION_INDEX_DIVISION:
      return loom_execution_integer_division_is_safe(module, op, fact_table,
                                                     condition);
    default:
      IREE_ASSERT_UNREACHABLE("unimplemented execution condition");
      return false;
  }
}
