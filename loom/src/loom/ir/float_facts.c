// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/float_facts.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/math.h"

static bool loom_float_type_uses_f32_arithmetic(
    loom_scalar_type_t scalar_type) {
  return scalar_type >= LOOM_SCALAR_TYPE_F8E4M3 &&
         scalar_type <= LOOM_SCALAR_TYPE_F32;
}

static bool loom_float_type_is_supported(loom_scalar_type_t scalar_type) {
  return loom_float_type_uses_f32_arithmetic(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_F64;
}

static double loom_float_round_to_type(loom_scalar_type_t scalar_type,
                                       double value) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return iree_math_f8e4m3fn_to_f64(iree_math_f32_to_f8e4m3fn((float)value));
    case LOOM_SCALAR_TYPE_F8E5M2:
      return iree_math_f8e5m2_to_f64(iree_math_f32_to_f8e5m2((float)value));
    case LOOM_SCALAR_TYPE_F16:
      return iree_math_f16_to_f64(iree_math_f32_to_f16((float)value));
    case LOOM_SCALAR_TYPE_BF16:
      return iree_math_bf16_to_f64(iree_math_f32_to_bf16((float)value));
    case LOOM_SCALAR_TYPE_F32:
      return (double)(float)value;
    case LOOM_SCALAR_TYPE_F64:
      return value;
    default:
      return NAN;
  }
}

// Constructs exact facts for a value already rounded to its declared type.
static loom_value_facts_t loom_value_facts_exact_rounded_float(double value) {
  loom_value_facts_t facts = {0};
  memcpy(&facts.range_lo, &value, sizeof(value));
  facts.range_hi = facts.range_lo;
  facts.known_divisor = 1;
  facts.flags = LOOM_VALUE_FACT_EXACT | LOOM_VALUE_FACT_FLOAT;
  if (isnan(value)) {
    facts.flags |= LOOM_VALUE_FACT_NAN | LOOM_VALUE_FACT_NOT_INF;
  } else {
    facts.flags |= LOOM_VALUE_FACT_NOT_NAN;
    if (isinf(value)) {
      facts.flags |= LOOM_VALUE_FACT_INF;
    } else {
      facts.flags |= LOOM_VALUE_FACT_NOT_INF | LOOM_VALUE_FACT_FINITE;
    }
  }
  loom_value_facts_mark_cluster_uniform(&facts);
  return facts;
}

loom_value_facts_t loom_value_facts_exact_float(loom_scalar_type_t scalar_type,
                                                double value) {
  if (!loom_float_type_is_supported(scalar_type)) {
    return loom_value_facts_unknown();
  }
  const double rounded_value = loom_float_round_to_type(scalar_type, value);
  loom_value_facts_t facts =
      loom_value_facts_exact_rounded_float(rounded_value);
  if (scalar_type == LOOM_SCALAR_TYPE_F64 &&
      fpclassify(rounded_value) != FP_SUBNORMAL) {
    facts.flags |= LOOM_VALUE_FACT_NOT_SUBNORMAL;
  }
  return facts;
}

loom_value_facts_t loom_value_facts_known_nan(void) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  facts.flags =
      LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NAN | LOOM_VALUE_FACT_NOT_INF;
  return facts;
}

bool loom_value_facts_as_exact_float(loom_scalar_type_t scalar_type,
                                     loom_value_facts_t facts,
                                     double* out_value) {
  if (!loom_float_type_is_supported(scalar_type) ||
      !loom_value_facts_is_exact(facts) || !loom_value_facts_is_float(facts)) {
    return false;
  }
  memcpy(out_value, &facts.range_lo, sizeof(*out_value));
  return true;
}

float loom_float_logistic_f32(float input) {
  if (input >= 0.0f) {
    const float exponent = expf(-input);
    return 1.0f / (1.0f + exponent);
  }
  const float exponent = expf(input);
  return exponent / (1.0f + exponent);
}

double loom_float_logistic_f64(double input) {
  if (input >= 0.0) {
    const double exponent = exp(-input);
    return 1.0 / (1.0 + exponent);
  }
  const double exponent = exp(input);
  return exponent / (1.0 + exponent);
}

void loom_value_facts_eval_float_unary(loom_scalar_type_t scalar_type,
                                       const loom_value_facts_t* input,
                                       loom_float_unary_f32_fn_t f32_fn,
                                       loom_float_unary_f64_fn_t f64_fn,
                                       loom_value_facts_t* out_facts) {
  double input_value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, *input, &input_value)) {
    *out_facts = loom_value_facts_unknown();
  } else if (loom_float_type_uses_f32_arithmetic(scalar_type)) {
    *out_facts = loom_value_facts_exact_float(
        scalar_type, (double)f32_fn((float)input_value));
  } else {
    *out_facts = loom_value_facts_exact_float(scalar_type, f64_fn(input_value));
  }
  loom_value_facts_propagate_unary_distribution(*input, out_facts);
}

void loom_value_facts_eval_float_unary_data(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* input,
    loom_float_unary_data_f32_fn_t f32_fn,
    loom_float_unary_data_f64_fn_t f64_fn, const void* user_data,
    loom_value_facts_t* out_facts) {
  double input_value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, *input, &input_value)) {
    *out_facts = loom_value_facts_unknown();
  } else if (loom_float_type_uses_f32_arithmetic(scalar_type)) {
    *out_facts = loom_value_facts_exact_float(
        scalar_type, (double)f32_fn((float)input_value, user_data));
  } else {
    *out_facts = loom_value_facts_exact_float(scalar_type,
                                              f64_fn(input_value, user_data));
  }
  loom_value_facts_propagate_unary_distribution(*input, out_facts);
}

void loom_value_facts_eval_float_binary(loom_scalar_type_t scalar_type,
                                        const loom_value_facts_t* lhs,
                                        const loom_value_facts_t* rhs,
                                        loom_float_binary_f32_fn_t f32_fn,
                                        loom_float_binary_f64_fn_t f64_fn,
                                        loom_value_facts_t* out_facts) {
  double lhs_value = 0.0;
  double rhs_value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, *lhs, &lhs_value) ||
      !loom_value_facts_as_exact_float(scalar_type, *rhs, &rhs_value)) {
    *out_facts = loom_value_facts_unknown();
  } else if (loom_float_type_uses_f32_arithmetic(scalar_type)) {
    *out_facts = loom_value_facts_exact_float(
        scalar_type, (double)f32_fn((float)lhs_value, (float)rhs_value));
  } else {
    *out_facts =
        loom_value_facts_exact_float(scalar_type, f64_fn(lhs_value, rhs_value));
  }
  loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out_facts);
}

void loom_value_facts_eval_float_ternary(loom_scalar_type_t scalar_type,
                                         const loom_value_facts_t* a,
                                         const loom_value_facts_t* b,
                                         const loom_value_facts_t* c,
                                         loom_float_ternary_f32_fn_t f32_fn,
                                         loom_float_ternary_f64_fn_t f64_fn,
                                         loom_value_facts_t* out_facts) {
  double a_value = 0.0;
  double b_value = 0.0;
  double c_value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, *a, &a_value) ||
      !loom_value_facts_as_exact_float(scalar_type, *b, &b_value) ||
      !loom_value_facts_as_exact_float(scalar_type, *c, &c_value)) {
    *out_facts = loom_value_facts_unknown();
  } else if (loom_float_type_uses_f32_arithmetic(scalar_type)) {
    *out_facts = loom_value_facts_exact_float(
        scalar_type,
        (double)f32_fn((float)a_value, (float)b_value, (float)c_value));
  } else {
    *out_facts = loom_value_facts_exact_float(
        scalar_type, f64_fn(a_value, b_value, c_value));
  }
  loom_value_facts_propagate_ternary_distribution(*a, *b, *c, out_facts);
}

// Reduces |input| to an angle in [-pi/4, pi/4] and returns its quadrant.
// The quarter-turn divisor is exactly representable, so remquo preserves the
// periodic position of every finite input while providing the low quotient
// bits needed to recover the quadrant.
static float loom_float_reduce_turns_f32(float input, int* out_quadrant) {
  int quotient = 0;
  const float residual = remquof(input, 0.25f, &quotient);
  int quadrant = quotient % 4;
  if (quadrant < 0) quadrant += 4;
  *out_quadrant = quadrant;
  return residual * 6.2831853071795864769252867665590057683943387987502f;
}

static double loom_float_reduce_turns_f64(double input, int* out_quadrant) {
  int quotient = 0;
  const double residual = remquo(input, 0.25, &quotient);
  int quadrant = quotient % 4;
  if (quadrant < 0) quadrant += 4;
  *out_quadrant = quadrant;
  return residual * 6.2831853071795864769252867665590057683943387987502;
}

static float loom_float_eval_turns_f32(float input, const void* user_data) {
  const loom_float_turns_kind_t kind =
      *(const loom_float_turns_kind_t*)user_data;
  int quadrant = 0;
  const float angle = loom_float_reduce_turns_f32(input, &quadrant);
  if (angle == 0.0f) {
    if (kind == LOOM_FLOAT_TURNS_SIN) {
      if (quadrant == 1) return 1.0f;
      if (quadrant == 3) return -1.0f;
      return copysignf(0.0f, input);
    }
    if (quadrant == 0) return 1.0f;
    if (quadrant == 2) return -1.0f;
    return 0.0f;
  }
  if (kind == LOOM_FLOAT_TURNS_SIN) {
    switch (quadrant) {
      case 0:
        return sinf(angle);
      case 1:
        return cosf(angle);
      case 2:
        return -sinf(angle);
      default:
        return -cosf(angle);
    }
  }
  switch (quadrant) {
    case 0:
      return cosf(angle);
    case 1:
      return -sinf(angle);
    case 2:
      return -cosf(angle);
    default:
      return sinf(angle);
  }
}

static double loom_float_eval_turns_f64(double input, const void* user_data) {
  const loom_float_turns_kind_t kind =
      *(const loom_float_turns_kind_t*)user_data;
  int quadrant = 0;
  const double angle = loom_float_reduce_turns_f64(input, &quadrant);
  if (angle == 0.0) {
    if (kind == LOOM_FLOAT_TURNS_SIN) {
      if (quadrant == 1) return 1.0;
      if (quadrant == 3) return -1.0;
      return copysign(0.0, input);
    }
    if (quadrant == 0) return 1.0;
    if (quadrant == 2) return -1.0;
    return 0.0;
  }
  if (kind == LOOM_FLOAT_TURNS_SIN) {
    switch (quadrant) {
      case 0:
        return sin(angle);
      case 1:
        return cos(angle);
      case 2:
        return -sin(angle);
      default:
        return -cos(angle);
    }
  }
  switch (quadrant) {
    case 0:
      return cos(angle);
    case 1:
      return -sin(angle);
    case 2:
      return -cos(angle);
    default:
      return sin(angle);
  }
}

void loom_value_facts_eval_float_turns(loom_scalar_type_t scalar_type,
                                       loom_float_turns_kind_t kind,
                                       const loom_value_facts_t* input,
                                       loom_value_facts_t* out_facts) {
  if (kind != LOOM_FLOAT_TURNS_SIN && kind != LOOM_FLOAT_TURNS_COS) {
    *out_facts = loom_value_facts_unknown();
    loom_value_facts_propagate_unary_distribution(*input, out_facts);
    return;
  }
  if (loom_value_facts_is_nan(*input) || loom_value_facts_is_inf(*input)) {
    *out_facts = loom_value_facts_exact_float(scalar_type, NAN);
    loom_value_facts_propagate_unary_distribution(*input, out_facts);
    return;
  }
  loom_value_facts_eval_float_unary_data(
      scalar_type, input, loom_float_eval_turns_f32, loom_float_eval_turns_f64,
      &kind, out_facts);
}

static float loom_float_negate_f32(float value) { return -value; }
static double loom_float_negate_f64(double value) { return -value; }
static float loom_float_abs_f32(float value) { return fabsf(value); }
static double loom_float_abs_f64(double value) { return fabs(value); }
static float loom_float_copysign_f32(float magnitude, float sign) {
  return copysignf(magnitude, sign);
}
static double loom_float_copysign_f64(double magnitude, double sign) {
  return copysign(magnitude, sign);
}

static void loom_value_facts_make_unmaterializable_nan(
    loom_value_facts_t input, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_known_nan();
  loom_value_facts_propagate_unary_distribution(input, out_facts);
}

void loom_value_facts_eval_float_negate(loom_scalar_type_t scalar_type,
                                        const loom_value_facts_t* input,
                                        loom_value_facts_t* out_facts) {
  if (loom_value_facts_is_nan(*input)) {
    loom_value_facts_make_unmaterializable_nan(*input, out_facts);
    return;
  }
  loom_value_facts_eval_float_unary(scalar_type, input, loom_float_negate_f32,
                                    loom_float_negate_f64, out_facts);
}

void loom_value_facts_eval_float_abs(loom_scalar_type_t scalar_type,
                                     const loom_value_facts_t* input,
                                     loom_value_facts_t* out_facts) {
  if (loom_value_facts_is_nan(*input)) {
    loom_value_facts_make_unmaterializable_nan(*input, out_facts);
    return;
  }
  loom_value_facts_eval_float_unary(scalar_type, input, loom_float_abs_f32,
                                    loom_float_abs_f64, out_facts);
}

void loom_value_facts_eval_float_copysign(loom_scalar_type_t scalar_type,
                                          const loom_value_facts_t* magnitude,
                                          const loom_value_facts_t* sign,
                                          loom_value_facts_t* out_facts) {
  if (loom_value_facts_is_nan(*magnitude)) {
    *out_facts = loom_value_facts_known_nan();
    loom_value_facts_propagate_binary_distribution(*magnitude, *sign,
                                                   out_facts);
    return;
  }
  loom_value_facts_eval_float_binary(scalar_type, magnitude, sign,
                                     loom_float_copysign_f32,
                                     loom_float_copysign_f64, out_facts);
}

static double loom_float_minimum(double lhs, double rhs) {
  if (lhs == rhs) {
    return lhs == 0.0 && (signbit(lhs) || signbit(rhs)) ? -0.0 : lhs;
  }
  return lhs < rhs ? lhs : rhs;
}

static double loom_float_maximum(double lhs, double rhs) {
  if (lhs == rhs) {
    return lhs == 0.0 && !(signbit(lhs) && signbit(rhs)) ? 0.0 : lhs;
  }
  return lhs > rhs ? lhs : rhs;
}

void loom_value_facts_eval_float_minmax(loom_scalar_type_t scalar_type,
                                        loom_float_minmax_kind_t kind,
                                        const loom_value_facts_t* lhs,
                                        const loom_value_facts_t* rhs,
                                        loom_value_facts_t* out_facts) {
  switch (kind) {
    case LOOM_FLOAT_MINMAX_MINIMUM:
    case LOOM_FLOAT_MINMAX_MAXIMUM:
    case LOOM_FLOAT_MINMAX_MINNUM:
    case LOOM_FLOAT_MINMAX_MAXNUM:
      break;
    default:
      *out_facts = loom_value_facts_unknown();
      return;
  }
  const bool lhs_is_nan = loom_value_facts_is_nan(*lhs);
  const bool rhs_is_nan = loom_value_facts_is_nan(*rhs);
  if (kind == LOOM_FLOAT_MINMAX_MINNUM || kind == LOOM_FLOAT_MINMAX_MAXNUM) {
    if (lhs_is_nan && !rhs_is_nan) {
      *out_facts = *rhs;
      loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out_facts);
      return;
    }
    if (!lhs_is_nan && rhs_is_nan) {
      *out_facts = *lhs;
      loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out_facts);
      return;
    }
  }
  if (lhs_is_nan || rhs_is_nan) {
    *out_facts = loom_value_facts_exact_float(scalar_type, NAN);
    loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out_facts);
    return;
  }

  double lhs_value = 0.0;
  double rhs_value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, *lhs, &lhs_value) ||
      !loom_value_facts_as_exact_float(scalar_type, *rhs, &rhs_value)) {
    *out_facts = loom_value_facts_unknown();
    loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out_facts);
    return;
  }
  const bool is_minimum =
      kind == LOOM_FLOAT_MINMAX_MINIMUM || kind == LOOM_FLOAT_MINMAX_MINNUM;
  *out_facts = loom_value_facts_exact_float(
      scalar_type, is_minimum ? loom_float_minimum(lhs_value, rhs_value)
                              : loom_float_maximum(lhs_value, rhs_value));
  loom_value_facts_propagate_binary_distribution(*lhs, *rhs, out_facts);
}

static void loom_value_facts_eval_float_clamp_ordered(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* value,
    const loom_value_facts_t* lower, const loom_value_facts_t* upper,
    loom_value_facts_t* out_facts) {
  if (loom_value_facts_is_nan(*value)) {
    *out_facts = *value;
  } else {
    double result = 0.0;
    if (!loom_value_facts_as_exact_float(scalar_type, *value, &result)) {
      *out_facts = loom_value_facts_unknown();
      loom_value_facts_propagate_ternary_distribution(*value, *lower, *upper,
                                                      out_facts);
      return;
    }
    if (!loom_value_facts_is_nan(*lower)) {
      double lower_value = 0.0;
      if (!loom_value_facts_as_exact_float(scalar_type, *lower, &lower_value)) {
        *out_facts = loom_value_facts_unknown();
        loom_value_facts_propagate_ternary_distribution(*value, *lower, *upper,
                                                        out_facts);
        return;
      }
      if (result < lower_value) result = lower_value;
    }
    if (!loom_value_facts_is_nan(*upper)) {
      double upper_value = 0.0;
      if (!loom_value_facts_as_exact_float(scalar_type, *upper, &upper_value)) {
        *out_facts = loom_value_facts_unknown();
        loom_value_facts_propagate_ternary_distribution(*value, *lower, *upper,
                                                        out_facts);
        return;
      }
      if (result > upper_value) result = upper_value;
    }
    *out_facts = loom_value_facts_exact_float(scalar_type, result);
  }
  loom_value_facts_propagate_ternary_distribution(*value, *lower, *upper,
                                                  out_facts);
}

void loom_value_facts_eval_float_clamp(loom_scalar_type_t scalar_type,
                                       loom_float_clamp_kind_t kind,
                                       const loom_value_facts_t* value,
                                       const loom_value_facts_t* lower,
                                       const loom_value_facts_t* upper,
                                       loom_value_facts_t* out_facts) {
  if (kind == LOOM_FLOAT_CLAMP_ORDERED) {
    loom_value_facts_eval_float_clamp_ordered(scalar_type, value, lower, upper,
                                              out_facts);
    return;
  }
  loom_value_facts_t intermediate = {0};
  if (kind == LOOM_FLOAT_CLAMP_NUMBER) {
    loom_value_facts_eval_float_minmax(scalar_type, LOOM_FLOAT_MINMAX_MAXNUM,
                                       value, lower, &intermediate);
    loom_value_facts_eval_float_minmax(scalar_type, LOOM_FLOAT_MINMAX_MINNUM,
                                       &intermediate, upper, out_facts);
    return;
  }
  if (kind != LOOM_FLOAT_CLAMP_IEEE) {
    *out_facts = loom_value_facts_unknown();
    return;
  }
  loom_value_facts_eval_float_minmax(scalar_type, LOOM_FLOAT_MINMAX_MAXIMUM,
                                     value, lower, &intermediate);
  loom_value_facts_eval_float_minmax(scalar_type, LOOM_FLOAT_MINMAX_MINIMUM,
                                     &intermediate, upper, out_facts);
}

void loom_value_facts_eval_float_to_integer(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_float_integer_conversion_kind_t kind,
    const loom_value_facts_t* source_facts,
    loom_value_facts_t* out_result_facts) {
  *out_result_facts = loom_value_facts_unknown();

  double source_value = 0.0;
  int64_t result_domain_lo = 0;
  int64_t result_domain_hi = 0;
  const int32_t result_bit_count = loom_scalar_type_bitwidth(result_type);
  if (!loom_value_facts_as_exact_float(source_type, *source_facts,
                                       &source_value) ||
      !isfinite(source_value) || !loom_scalar_type_is_integer(result_type) ||
      !loom_value_facts_scalar_type_domain(result_type, &result_domain_lo,
                                           &result_domain_hi) ||
      result_bit_count <= 0 || result_bit_count > 64) {
    return;
  }

  const double truncated_value = trunc(source_value);
  switch (kind) {
    case LOOM_FLOAT_INTEGER_CONVERSION_SIGNED: {
      const double upper_exclusive =
          result_bit_count == 64 ? 0x1p63 : (double)result_domain_hi + 1.0;
      if (truncated_value >= (double)result_domain_lo &&
          truncated_value < upper_exclusive) {
        *out_result_facts =
            loom_value_facts_exact_i64((int64_t)truncated_value);
      }
      break;
    }
    case LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED: {
      const double upper_exclusive = ldexp(1.0, result_bit_count);
      if (truncated_value >= 0.0 && truncated_value < upper_exclusive) {
        const uint64_t result_bits = (uint64_t)truncated_value;
        *out_result_facts =
            result_type == LOOM_SCALAR_TYPE_I1
                ? loom_value_facts_exact_i64(result_bits != 0 ? 1 : 0)
                : loom_value_facts_make_signed_raw_bits(result_bits,
                                                        result_bit_count);
      }
      break;
    }
    default:
      break;
  }
}

bool loom_value_facts_as_exact_float_bits(loom_scalar_type_t scalar_type,
                                          loom_value_facts_t facts,
                                          uint64_t* out_bits) {
  double value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, facts, &value) ||
      isnan(value)) {
    return false;
  }
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_bits = iree_math_f32_to_f8e4m3fn((float)value);
      return true;
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_bits = iree_math_f32_to_f8e5m2((float)value);
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_bits = iree_math_f32_to_f16((float)value);
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_bits = iree_math_f32_to_bf16((float)value);
      return true;
    case LOOM_SCALAR_TYPE_F32: {
      float f32_value = (float)value;
      uint32_t bits = 0;
      memcpy(&bits, &f32_value, sizeof(bits));
      *out_bits = bits;
      return true;
    }
    case LOOM_SCALAR_TYPE_F64:
      memcpy(out_bits, &value, sizeof(value));
      return true;
    default:
      return false;
  }
}

static bool loom_float_bits_are_nan(loom_scalar_type_t scalar_type,
                                    uint64_t bits) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return (bits & UINT64_C(0x7F)) == UINT64_C(0x7F);
    case LOOM_SCALAR_TYPE_F8E5M2:
      return (bits & UINT64_C(0x7C)) == UINT64_C(0x7C) &&
             (bits & UINT64_C(0x03)) != 0;
    case LOOM_SCALAR_TYPE_F16:
      return (bits & UINT64_C(0x7C00)) == UINT64_C(0x7C00) &&
             (bits & UINT64_C(0x03FF)) != 0;
    case LOOM_SCALAR_TYPE_BF16:
      return (bits & UINT64_C(0x7F80)) == UINT64_C(0x7F80) &&
             (bits & UINT64_C(0x007F)) != 0;
    case LOOM_SCALAR_TYPE_F32:
      return (bits & UINT64_C(0x7F800000)) == UINT64_C(0x7F800000) &&
             (bits & UINT64_C(0x007FFFFF)) != 0;
    case LOOM_SCALAR_TYPE_F64:
      return (bits & UINT64_C(0x7FF0000000000000)) ==
                 UINT64_C(0x7FF0000000000000) &&
             (bits & UINT64_C(0x000FFFFFFFFFFFFF)) != 0;
    default:
      return false;
  }
}

bool loom_value_facts_from_float_bits(loom_scalar_type_t scalar_type,
                                      uint64_t bits,
                                      loom_value_facts_t* out_facts) {
  if (!loom_float_type_is_supported(scalar_type)) return false;
  const int32_t bit_count = loom_scalar_type_bitwidth(scalar_type);
  bits = iree_math_mask_low_bits_u64(bits, bit_count);
  if (loom_float_bits_are_nan(scalar_type, bits)) {
    *out_facts = loom_value_facts_known_nan();
    return true;
  }
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_facts = loom_value_facts_exact_rounded_float(
          iree_math_f8e4m3fn_to_f64((uint8_t)bits));
      return true;
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_facts = loom_value_facts_exact_rounded_float(
          iree_math_f8e5m2_to_f64((uint8_t)bits));
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_facts = loom_value_facts_exact_rounded_float(
          iree_math_f16_to_f64((uint16_t)bits));
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_facts = loom_value_facts_exact_rounded_float(
          iree_math_bf16_to_f64((uint16_t)bits));
      return true;
    case LOOM_SCALAR_TYPE_F32: {
      uint32_t f32_bits = (uint32_t)bits;
      *out_facts = loom_value_facts_exact_rounded_float(
          iree_math_make_f64_from_f32_bits(f32_bits));
      return true;
    }
    case LOOM_SCALAR_TYPE_F64: {
      double value = 0.0;
      memcpy(&value, &bits, sizeof(value));
      *out_facts = loom_value_facts_exact_rounded_float(value);
      return true;
    }
    default:
      return false;
  }
}

void loom_value_facts_eval_scalar_bitcast(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    const loom_value_facts_t* source_facts,
    loom_value_facts_t* out_result_facts) {
  const int32_t source_bit_count = loom_scalar_type_bitwidth(source_type);
  const int32_t result_bit_count = loom_scalar_type_bitwidth(result_type);
  uint64_t bits = 0;
  bool has_bits = source_bit_count > 0 && source_bit_count == result_bit_count;
  if (has_bits && loom_scalar_type_is_float(source_type)) {
    has_bits =
        loom_value_facts_as_exact_float_bits(source_type, *source_facts, &bits);
  } else if (has_bits) {
    has_bits = loom_value_facts_as_exact_raw_bits(*source_facts,
                                                  source_bit_count, &bits);
  }
  if (!has_bits) {
    *out_result_facts = loom_value_facts_unknown();
    return;
  }

  bool converted = false;
  if (loom_scalar_type_is_float(result_type)) {
    converted =
        loom_value_facts_from_float_bits(result_type, bits, out_result_facts);
  } else if (result_type == LOOM_SCALAR_TYPE_I1) {
    *out_result_facts = loom_value_facts_exact_i64((bits & 1) != 0 ? 1 : 0);
    converted = true;
  } else if (result_bit_count > 0) {
    *out_result_facts =
        loom_value_facts_make_signed_raw_bits(bits, result_bit_count);
    converted = true;
  }
  if (!converted) {
    *out_result_facts = loom_value_facts_unknown();
    return;
  }
  loom_value_facts_propagate_unary_distribution(*source_facts,
                                                out_result_facts);
}
