// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-width modular integer fact transfers.

#include "loom/ir/facts.h"

typedef enum loom_value_facts_modular_binary_kind_e {
  LOOM_VALUE_FACTS_MODULAR_ADD,
  LOOM_VALUE_FACTS_MODULAR_SUBTRACT,
  LOOM_VALUE_FACTS_MODULAR_MULTIPLY,
} loom_value_facts_modular_binary_kind_t;

static loom_value_facts_t loom_value_facts_modular_exact(uint64_t raw_bits,
                                                         int32_t bit_count) {
  return bit_count == 1
             ? loom_value_facts_exact_i64((int64_t)(raw_bits & 1u))
             : loom_value_facts_make_signed_raw_bits(raw_bits, bit_count);
}

static bool loom_value_facts_try_modular_binary_exact(
    loom_value_facts_modular_binary_kind_t kind, loom_value_facts_t lhs_facts,
    loom_value_facts_t rhs_facts, int32_t bit_count, loom_value_facts_t* out) {
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  if (!loom_value_facts_as_exact_raw_bits(lhs_facts, bit_count, &lhs_bits) ||
      !loom_value_facts_as_exact_raw_bits(rhs_facts, bit_count, &rhs_bits)) {
    return false;
  }

  uint64_t result_bits = 0;
  switch (kind) {
    case LOOM_VALUE_FACTS_MODULAR_ADD:
      result_bits = lhs_bits + rhs_bits;
      break;
    case LOOM_VALUE_FACTS_MODULAR_SUBTRACT:
      result_bits = lhs_bits - rhs_bits;
      break;
    case LOOM_VALUE_FACTS_MODULAR_MULTIPLY:
      result_bits = lhs_bits * rhs_bits;
      break;
  }
  *out = loom_value_facts_modular_exact(result_bits, bit_count);
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
  return true;
}

void loom_value_facts_addi_modular(const loom_value_facts_t* lhs,
                                   const loom_value_facts_t* rhs,
                                   int32_t bit_count, loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if (loom_value_facts_try_modular_binary_exact(
          LOOM_VALUE_FACTS_MODULAR_ADD, lhs_facts, rhs_facts, bit_count, out)) {
    return;
  }
  loom_value_facts_addi(&lhs_facts, &rhs_facts, out);
  *out = loom_value_facts_wrap_integer(*out, bit_count);
}

void loom_value_facts_subi_modular(const loom_value_facts_t* lhs,
                                   const loom_value_facts_t* rhs,
                                   int32_t bit_count, loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if (loom_value_facts_try_modular_binary_exact(
          LOOM_VALUE_FACTS_MODULAR_SUBTRACT, lhs_facts, rhs_facts, bit_count,
          out)) {
    return;
  }
  loom_value_facts_subi(&lhs_facts, &rhs_facts, out);
  *out = loom_value_facts_wrap_integer(*out, bit_count);
}

static void loom_value_facts_unsigned_carry_or_borrow(
    const loom_value_facts_t* lhs, const loom_value_facts_t* rhs,
    int32_t bit_count, bool is_subtraction,
    loom_value_facts_t* out_carry_or_borrow) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  if (loom_value_facts_as_exact_raw_bits(lhs_facts, bit_count, &lhs_bits) &&
      loom_value_facts_as_exact_raw_bits(rhs_facts, bit_count, &rhs_bits)) {
    const bool carry_or_borrow =
        is_subtraction
            ? lhs_bits < rhs_bits
            : (bit_count == 64 ? lhs_bits + rhs_bits < lhs_bits
                               : ((lhs_bits + rhs_bits) >> bit_count) != 0);
    *out_carry_or_borrow = loom_value_facts_exact_i64(carry_or_borrow ? 1 : 0);
  } else {
    *out_carry_or_borrow = loom_value_facts_make(0, 1, 1);
  }
  loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts,
                                                 out_carry_or_borrow);
  if (loom_value_facts_is_lane_varying(*out_carry_or_borrow)) {
    loom_value_facts_mark_lane_predicate(out_carry_or_borrow);
  }
}

void loom_value_facts_addcarryui(const loom_value_facts_t* lhs,
                                 const loom_value_facts_t* rhs,
                                 int32_t bit_count, loom_value_facts_t* out_sum,
                                 loom_value_facts_t* out_carry) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  loom_value_facts_addi_modular(&lhs_facts, &rhs_facts, bit_count, out_sum);
  loom_value_facts_unsigned_carry_or_borrow(
      &lhs_facts, &rhs_facts, bit_count, /*is_subtraction=*/false, out_carry);
}

void loom_value_facts_subborrowui(const loom_value_facts_t* lhs,
                                  const loom_value_facts_t* rhs,
                                  int32_t bit_count,
                                  loom_value_facts_t* out_difference,
                                  loom_value_facts_t* out_borrow) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  loom_value_facts_subi_modular(&lhs_facts, &rhs_facts, bit_count,
                                out_difference);
  loom_value_facts_unsigned_carry_or_borrow(
      &lhs_facts, &rhs_facts, bit_count, /*is_subtraction=*/true, out_borrow);
}

void loom_value_facts_muli_modular(const loom_value_facts_t* lhs,
                                   const loom_value_facts_t* rhs,
                                   int32_t bit_count, loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  if (loom_value_facts_try_modular_binary_exact(
          LOOM_VALUE_FACTS_MODULAR_MULTIPLY, lhs_facts, rhs_facts, bit_count,
          out)) {
    return;
  }
  loom_value_facts_muli(&lhs_facts, &rhs_facts, out);
  *out = loom_value_facts_wrap_integer(*out, bit_count);
}

void loom_value_facts_shli_modular(const loom_value_facts_t* lhs,
                                   const loom_value_facts_t* rhs,
                                   int32_t bit_count, loom_value_facts_t* out) {
  const loom_value_facts_t lhs_facts = *lhs;
  const loom_value_facts_t rhs_facts = *rhs;
  uint64_t lhs_bits = 0;
  int64_t shift = 0;
  const bool shift_is_exact = loom_value_facts_as_exact_i64(rhs_facts, &shift);
  if (shift_is_exact && (shift < 0 || shift >= bit_count)) {
    *out = loom_value_facts_wrap_integer(loom_value_facts_unknown(), bit_count);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  if (shift_is_exact &&
      loom_value_facts_as_exact_raw_bits(lhs_facts, bit_count, &lhs_bits)) {
    *out = loom_value_facts_modular_exact(lhs_bits << shift, bit_count);
    loom_value_facts_propagate_binary_distribution(lhs_facts, rhs_facts, out);
    return;
  }
  loom_value_facts_shli(&lhs_facts, &rhs_facts, out);
  *out = loom_value_facts_wrap_integer(*out, bit_count);
}

void loom_value_facts_negi_modular(const loom_value_facts_t* input,
                                   int32_t bit_count, loom_value_facts_t* out) {
  const loom_value_facts_t input_facts = *input;
  uint64_t input_bits = 0;
  if (loom_value_facts_as_exact_raw_bits(input_facts, bit_count, &input_bits)) {
    *out = loom_value_facts_modular_exact(0u - input_bits, bit_count);
    loom_value_facts_propagate_unary_distribution(input_facts, out);
    return;
  }
  loom_value_facts_negi(&input_facts, out);
  *out = loom_value_facts_wrap_integer(*out, bit_count);
}

void loom_value_facts_absi_modular(const loom_value_facts_t* input,
                                   int32_t bit_count, loom_value_facts_t* out) {
  const loom_value_facts_t input_facts = *input;
  uint64_t input_bits = 0;
  if (loom_value_facts_as_exact_raw_bits(input_facts, bit_count, &input_bits)) {
    const uint64_t sign_bit = UINT64_C(1) << (bit_count - 1);
    const uint64_t result_bits =
        input_bits & sign_bit ? 0u - input_bits : input_bits;
    *out = loom_value_facts_modular_exact(result_bits, bit_count);
    loom_value_facts_propagate_unary_distribution(input_facts, out);
    return;
  }
  loom_value_facts_absi(&input_facts, out);
  *out = loom_value_facts_wrap_integer(*out, bit_count);
}

void loom_value_facts_fmai_modular(const loom_value_facts_t* a,
                                   const loom_value_facts_t* b,
                                   const loom_value_facts_t* c,
                                   int32_t bit_count, loom_value_facts_t* out) {
  const loom_value_facts_t a_facts = *a;
  const loom_value_facts_t b_facts = *b;
  const loom_value_facts_t c_facts = *c;
  uint64_t a_bits = 0;
  uint64_t b_bits = 0;
  uint64_t c_bits = 0;
  if (loom_value_facts_as_exact_raw_bits(a_facts, bit_count, &a_bits) &&
      loom_value_facts_as_exact_raw_bits(b_facts, bit_count, &b_bits) &&
      loom_value_facts_as_exact_raw_bits(c_facts, bit_count, &c_bits)) {
    *out = loom_value_facts_modular_exact(a_bits * b_bits + c_bits, bit_count);
    loom_value_facts_propagate_ternary_distribution(a_facts, b_facts, c_facts,
                                                    out);
    return;
  }
  loom_value_facts_fmai(&a_facts, &b_facts, &c_facts, out);
  *out = loom_value_facts_wrap_integer(*out, bit_count);
}
