// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/float_facts.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/math/trigonometry.h"

static bool loom_float_type_uses_f32_arithmetic(
    loom_scalar_type_t scalar_type) {
  return scalar_type >= LOOM_SCALAR_TYPE_F8E4M3 &&
         scalar_type <= LOOM_SCALAR_TYPE_F32;
}

static bool loom_float_type_is_supported(loom_scalar_type_t scalar_type) {
  return loom_float_type_uses_f32_arithmetic(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_F64;
}

// Returns the formats that contain every finite value of |scalar_type|. This
// is a partial order: F16 and BF16 do not contain one another, nor do the two
// FP8 formats.
static loom_scalar_type_set_t loom_float_type_exact_destinations(
    loom_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F16 |
             LOOM_SCALAR_TYPE_SET_BF16 | LOOM_SCALAR_TYPE_SET_F32 |
             LOOM_SCALAR_TYPE_SET_F64;
    case LOOM_SCALAR_TYPE_F8E5M2:
      return LOOM_SCALAR_TYPE_SET_F8E5M2 | LOOM_SCALAR_TYPE_SET_F16 |
             LOOM_SCALAR_TYPE_SET_BF16 | LOOM_SCALAR_TYPE_SET_F32 |
             LOOM_SCALAR_TYPE_SET_F64;
    case LOOM_SCALAR_TYPE_F16:
      return LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |
             LOOM_SCALAR_TYPE_SET_F64;
    case LOOM_SCALAR_TYPE_BF16:
      return LOOM_SCALAR_TYPE_SET_BF16 | LOOM_SCALAR_TYPE_SET_F32 |
             LOOM_SCALAR_TYPE_SET_F64;
    case LOOM_SCALAR_TYPE_F32:
      return LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64;
    case LOOM_SCALAR_TYPE_F64:
      return LOOM_SCALAR_TYPE_SET_F64;
    default:
      return LOOM_SCALAR_TYPE_SET_NONE;
  }
}

static void loom_value_facts_add_exact_float_types(
    loom_value_facts_t* facts, loom_scalar_type_set_t types) {
  if (types == LOOM_SCALAR_TYPE_SET_NONE) {
    return;
  }
  facts->flags |= LOOM_VALUE_FACT_FLOAT;
  loom_value_facts_set_exact_float_types(
      facts, loom_value_facts_exact_float_types(*facts) | types);
}

static double loom_float_type_minimum_normal(loom_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return 0x1p-6;
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
      return 0x1p-14;
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
      return 0x1p-126;
    case LOOM_SCALAR_TYPE_F64:
      return 0x1p-1022;
    default:
      return NAN;
  }
}

static bool loom_float_is_subnormal(loom_scalar_type_t scalar_type,
                                    double value) {
  const double magnitude = fabs(value);
  return magnitude != 0.0 &&
         magnitude < loom_float_type_minimum_normal(scalar_type);
}

static double loom_float_round_to_type(loom_scalar_type_t scalar_type,
                                       double value) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return iree_math_f8e4m3fn_to_f64(iree_math_f64_to_f8e4m3fn(value));
    case LOOM_SCALAR_TYPE_F8E5M2:
      return iree_math_f8e5m2_to_f64(iree_math_f64_to_f8e5m2(value));
    case LOOM_SCALAR_TYPE_F16:
      return iree_math_f16_to_f64(iree_math_f64_to_f16(value));
    case LOOM_SCALAR_TYPE_BF16:
      return iree_math_bf16_to_f64(iree_math_f64_to_bf16(value));
    case LOOM_SCALAR_TYPE_F32:
      return (double)(float)value;
    case LOOM_SCALAR_TYPE_F64:
      return value;
    default:
      return NAN;
  }
}

typedef struct loom_float_exact_format_t {
  // Number of significand bits, including the implicit leading bit.
  uint8_t precision;
  // Unbiased exponent of the format's smallest normal value.
  int16_t minimum_normal_exponent;
  // Largest finite magnitude represented by the format.
  double maximum_finite_value;
} loom_float_exact_format_t;

static bool loom_float_exact_format(loom_scalar_type_t scalar_type,
                                    loom_float_exact_format_t* out_format) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_format = (loom_float_exact_format_t){4, -6, 0x1.Cp8};
      return true;
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_format = (loom_float_exact_format_t){3, -14, 0x1.Cp15};
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_format = (loom_float_exact_format_t){11, -14, 0x1.FFCp15};
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_format = (loom_float_exact_format_t){8, -126, 0x1.FEp127};
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_format = (loom_float_exact_format_t){24, -126, 0x1.FFFFFEp127};
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_format =
          (loom_float_exact_format_t){53, -1022, 0x1.FFFFFFFFFFFFFp1023};
      return true;
    default:
      return false;
  }
}

// Returns whether the finite binary64 |value| is exactly representable in
// |format|. A binary value is exact when its least significant set bit is no
// smaller than the destination spacing at that magnitude. Working from the
// retained binary64 payload avoids performing six narrowing conversions for
// every constant encountered by fact analysis.
static bool loom_float_value_is_exact_in_format(
    double value, const loom_float_exact_format_t* format) {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  const uint64_t magnitude_bits = bits & UINT64_C(0x7FFFFFFFFFFFFFFF);
  if (magnitude_bits == 0) {
    return true;
  }
  if (!isfinite(value) || fabs(value) > format->maximum_finite_value) {
    return false;
  }

  const uint64_t fraction = magnitude_bits & UINT64_C(0x000FFFFFFFFFFFFF);
  const uint64_t encoded_exponent = magnitude_bits >> 52;
  const uint64_t significand = encoded_exponent == 0
                                   ? fraction
                                   : fraction | UINT64_C(0x0010000000000000);
  const int32_t significand_exponent =
      encoded_exponent == 0 ? -1074 : (int32_t)encoded_exponent - 1023 - 52;
  const int32_t most_significant_exponent =
      significand_exponent + 63 -
      (int32_t)iree_math_count_leading_zeros_u64(significand);
  const int32_t least_significant_exponent =
      significand_exponent +
      (int32_t)iree_math_count_trailing_zeros_u64(significand);
  const int32_t normal_spacing_exponent =
      most_significant_exponent - ((int32_t)format->precision - 1);
  const int32_t subnormal_spacing_exponent =
      format->minimum_normal_exponent - ((int32_t)format->precision - 1);
  return least_significant_exponent >=
         iree_max(normal_spacing_exponent, subnormal_spacing_exponent);
}

static loom_scalar_type_set_t loom_float_exact_value_destinations(
    loom_scalar_type_t source_type, double value) {
  loom_scalar_type_set_t result =
      loom_float_type_exact_destinations(source_type);
  if (!isfinite(value)) {
    // The guarantee covers finite values only. A nonfinite alternative must
    // not weaken the representability of finite values at a later join.
    return LOOM_SCALAR_TYPE_SET_FLOAT;
  }
  if (value == 0.0) {
    return LOOM_SCALAR_TYPE_SET_FLOAT;
  }
  for (loom_scalar_type_t type = LOOM_SCALAR_TYPE_F8E4M3;
       type <= LOOM_SCALAR_TYPE_F64; ++type) {
    if (loom_scalar_type_set_contains(result, type)) {
      continue;
    }
    loom_float_exact_format_t format;
    if (loom_float_exact_format(type, &format) &&
        loom_float_value_is_exact_in_format(value, &format)) {
      result |= (loom_scalar_type_set_t)(1u << type);
    }
  }
  return result;
}

// Converts one signed integer endpoint directly to its declared floating-point
// type. The direct f32 cast matters for i64 values: routing through f64 can
// double-round at an f32 midpoint. Narrow formats use the exact f64 source path
// only while the integer is exactly representable by f64.
static bool loom_signed_integer_to_rounded_float(loom_scalar_type_t result_type,
                                                 int64_t value,
                                                 double* out_value) {
  switch (result_type) {
    case LOOM_SCALAR_TYPE_F32:
      *out_value = (double)(float)value;
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_value = (double)value;
      return true;
    default:
      if (value < -INT64_C(9007199254740992) ||
          value > INT64_C(9007199254740992)) {
        return false;
      }
      *out_value = loom_float_round_to_type(result_type, (double)value);
      return true;
  }
}

// Unsigned counterpart to loom_signed_integer_to_rounded_float.
static bool loom_unsigned_integer_to_rounded_float(
    loom_scalar_type_t result_type, uint64_t value, double* out_value) {
  switch (result_type) {
    case LOOM_SCALAR_TYPE_F32:
      *out_value = (double)(float)value;
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_value = (double)value;
      return true;
    default:
      if (value > UINT64_C(9007199254740992)) {
        return false;
      }
      *out_value = loom_float_round_to_type(result_type, (double)value);
      return true;
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

IREE_ATTRIBUTE_NOINLINE loom_value_facts_t
loom_value_facts_exact_float(loom_scalar_type_t scalar_type, double value) {
  if (!loom_float_type_is_supported(scalar_type)) {
    return loom_value_facts_unknown();
  }
  const double rounded_value = loom_float_round_to_type(scalar_type, value);
  loom_value_facts_t facts =
      loom_value_facts_exact_rounded_float(rounded_value);
  loom_value_facts_add_exact_float_types(
      &facts, loom_float_exact_value_destinations(scalar_type, rounded_value));
  if (!loom_float_is_subnormal(scalar_type, rounded_value)) {
    facts.flags |= LOOM_VALUE_FACT_NOT_SUBNORMAL;
  }
  return facts;
}

loom_value_facts_t loom_value_facts_make_float_range(
    loom_scalar_type_t scalar_type, double lo, double hi) {
  if (!loom_float_type_is_supported(scalar_type) || !isfinite(lo) ||
      !isfinite(hi) || lo > hi) {
    return loom_value_facts_unknown();
  }

  const double rounded_lo = loom_float_round_to_type(scalar_type, lo);
  const double rounded_hi = loom_float_round_to_type(scalar_type, hi);
  int64_t rounded_lo_bits = 0;
  int64_t rounded_hi_bits = 0;
  memcpy(&rounded_lo_bits, &rounded_lo, sizeof(rounded_lo));
  memcpy(&rounded_hi_bits, &rounded_hi, sizeof(rounded_hi));
  if (rounded_lo_bits == rounded_hi_bits) {
    return loom_value_facts_exact_float(scalar_type, rounded_lo);
  }

  loom_value_facts_t facts = loom_value_facts_unknown();
  facts.flags = LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NOT_NAN;
  if (isfinite(rounded_lo) && isfinite(rounded_hi)) {
    facts.range_lo = rounded_lo_bits;
    facts.range_hi = rounded_hi_bits;
    facts.flags |= LOOM_VALUE_FACT_NOT_INF | LOOM_VALUE_FACT_FINITE;
    const double minimum_normal = loom_float_type_minimum_normal(scalar_type);
    if ((rounded_lo == 0.0 && rounded_hi == 0.0) ||
        rounded_hi <= -minimum_normal || rounded_lo >= minimum_normal) {
      facts.flags |= LOOM_VALUE_FACT_NOT_SUBNORMAL;
    }
  }
  loom_value_facts_add_exact_float_types(
      &facts, loom_float_type_exact_destinations(scalar_type));
  return facts;
}

loom_value_facts_t loom_value_facts_known_nan(void) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  facts.flags = LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NAN |
                LOOM_VALUE_FACT_NOT_INF | LOOM_VALUE_FACT_NOT_SUBNORMAL;
  loom_value_facts_set_exact_float_types(&facts, LOOM_SCALAR_TYPE_SET_FLOAT);
  return facts;
}

static void loom_value_facts_add_signed_zero_alternative(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    const loom_value_facts_t* source_facts, loom_value_facts_t* result_facts) {
  double source_value = 0.0;
  if (loom_value_facts_as_exact_float(source_type, *source_facts,
                                      &source_value)) {
    const loom_value_facts_t signed_zero =
        loom_value_facts_exact_float(result_type, copysign(0.0, source_value));
    loom_value_facts_meet(result_facts, &signed_zero, result_facts);
    return;
  }

  double range_lo = 0.0;
  double range_hi = 0.0;
  if (loom_value_facts_as_float_range(source_type, *source_facts, &range_lo,
                                      &range_hi)) {
    if (range_lo > 0.0) {
      const loom_value_facts_t positive_zero =
          loom_value_facts_exact_float(result_type, 0.0);
      loom_value_facts_meet(result_facts, &positive_zero, result_facts);
      return;
    }
    if (range_hi < 0.0) {
      const loom_value_facts_t negative_zero =
          loom_value_facts_exact_float(result_type, -0.0);
      loom_value_facts_meet(result_facts, &negative_zero, result_facts);
      return;
    }
  }

  const loom_value_facts_t positive_zero =
      loom_value_facts_exact_float(result_type, 0.0);
  const loom_value_facts_t negative_zero =
      loom_value_facts_exact_float(result_type, -0.0);
  loom_value_facts_meet(result_facts, &positive_zero, result_facts);
  loom_value_facts_meet(result_facts, &negative_zero, result_facts);
}

void loom_value_facts_eval_float_conversion(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_float_conversion_policy_t policy,
    const loom_value_facts_t* input_facts,
    loom_value_facts_t* out_result_facts) {
  double input_value = 0.0;
  if (loom_value_facts_is_nan(*input_facts)) {
    *out_result_facts = loom_value_facts_known_nan();
  } else if (loom_value_facts_as_exact_float(source_type, *input_facts,
                                             &input_value)) {
    *out_result_facts = loom_value_facts_exact_float(result_type, input_value);
  } else {
    double range_lo = 0.0;
    double range_hi = 0.0;
    if (loom_value_facts_as_float_range(source_type, *input_facts, &range_lo,
                                        &range_hi)) {
      *out_result_facts =
          loom_value_facts_make_float_range(result_type, range_lo, range_hi);
    } else if (loom_scalar_type_bitwidth(source_type) <
               loom_scalar_type_bitwidth(result_type)) {
      *out_result_facts = *input_facts;
    } else {
      *out_result_facts = loom_value_facts_unknown();
    }
  }

  if (policy.may_flush_input_subnormal &&
      !loom_value_facts_is_not_subnormal(*input_facts)) {
    loom_value_facts_add_signed_zero_alternative(source_type, result_type,
                                                 input_facts, out_result_facts);
  }
  if (policy.may_flush_result_subnormal &&
      !loom_value_facts_is_not_subnormal(*out_result_facts)) {
    loom_value_facts_add_signed_zero_alternative(
        result_type, result_type, out_result_facts, out_result_facts);
  }

  loom_scalar_type_set_t result_exact_types =
      loom_value_facts_exact_float_types(*out_result_facts) |
      loom_float_type_exact_destinations(result_type);
  const loom_scalar_type_set_t source_exact_types =
      loom_value_facts_exact_float_types(*input_facts) |
      loom_float_type_exact_destinations(source_type);
  const bool preserves_input_value =
      loom_scalar_type_set_contains(source_exact_types, result_type) &&
      (!policy.may_flush_input_subnormal ||
       loom_value_facts_is_not_subnormal(*input_facts)) &&
      (!policy.may_flush_result_subnormal ||
       loom_value_facts_is_not_subnormal(*out_result_facts));
  if (preserves_input_value) {
    result_exact_types |= source_exact_types;
  }
  loom_value_facts_add_exact_float_types(out_result_facts, result_exact_types);
  loom_value_facts_propagate_unary_distribution(*input_facts, out_result_facts);
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

bool loom_value_facts_as_exact_power_of_two_float(
    loom_scalar_type_t scalar_type, loom_value_facts_t facts,
    int32_t* out_exponent) {
  double value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, facts, &value) ||
      !isfinite(value) || value == 0.0) {
    return false;
  }
  int exponent = 0;
  const double fraction = frexp(fabs(value), &exponent);
  if (fraction != 0.5) {
    return false;
  }
  *out_exponent = (int32_t)(exponent - 1);
  return true;
}

bool loom_value_facts_as_float_range(loom_scalar_type_t scalar_type,
                                     loom_value_facts_t facts, double* out_lo,
                                     double* out_hi) {
  if (!loom_float_type_is_supported(scalar_type) ||
      !loom_value_facts_is_float(facts) || !loom_value_facts_is_finite(facts) ||
      (facts.range_lo == INT64_MIN && facts.range_hi == INT64_MAX)) {
    return false;
  }
  memcpy(out_lo, &facts.range_lo, sizeof(*out_lo));
  memcpy(out_hi, &facts.range_hi, sizeof(*out_hi));
  return true;
}

bool loom_value_facts_float_range_within(loom_scalar_type_t scalar_type,
                                         loom_value_facts_t facts,
                                         double minimum_value,
                                         double maximum_value) {
  double lo = 0.0;
  double hi = 0.0;
  return minimum_value <= maximum_value &&
         loom_value_facts_as_float_range(scalar_type, facts, &lo, &hi) &&
         lo >= minimum_value && hi <= maximum_value;
}

void loom_value_facts_drop_float_range(loom_value_facts_t* facts) {
  facts->range_lo = INT64_MIN;
  facts->range_hi = INT64_MAX;
  facts->known_divisor = 1;
  facts->flags &= ~LOOM_VALUE_FACT_EXACT;
}

void loom_value_facts_assume_float_class(loom_value_facts_t* facts,
                                         bool not_nan, bool not_inf) {
  if ((!not_nan && !not_inf) ||
      (facts->flags & (LOOM_VALUE_FACT_NAN | LOOM_VALUE_FACT_INF))) {
    return;
  }
  facts->flags |= LOOM_VALUE_FACT_FLOAT |
                  (not_nan ? LOOM_VALUE_FACT_NOT_NAN : 0) |
                  (not_inf ? LOOM_VALUE_FACT_NOT_INF : 0);
  if (loom_value_facts_is_not_nan(*facts) &&
      loom_value_facts_is_not_inf(*facts)) {
    facts->flags |= LOOM_VALUE_FACT_FINITE;
  }
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

// Produces an F32 value that rounds to the exact narrow FMA result. For finite
// F8/F16/BF16 operands, the product has at most 22 significant bits and fits
// exactly in F64. A rounded F64 sum alone can still lose a tiny addend that
// changes the side of a narrow rounding midpoint.
static float loom_float_fma_narrow(double a, double b, double c) {
  const double product = a * b;
  const double sum = product + c;
  float rounded = (float)sum;
  uint32_t bits = 0;
  memcpy(&bits, &rounded, sizeof(bits));

  // F16 has the widest narrow significand. Every narrow rounding midpoint,
  // including subnormal and overflow boundaries, has twelve zero low F32 bits.
  if ((bits & UINT32_C(0xFFF)) != 0 || !isfinite(rounded)) {
    return rounded;
  }

  double error = sum - (double)rounded;
  if (error == 0.0) {
    // FastTwoSum recovers the exact residual even when the addend is too small
    // to survive the F64 addition. Neither its product nor residual can exceed
    // F64's exponent range for the supported narrow operand formats.
    error =
        fabs(product) >= fabs(c) ? c - (sum - product) : product - (sum - c);
  }
  if (error != 0.0) {
    // Nudge a midpoint toward the exact result before narrowing. The filter
    // also admits non-midpoints, where one F32 ULP cannot change narrow
    // rounding.
    bits += ((error > 0.0) != ((bits >> 31) != 0)) ? 1u : UINT32_MAX;
    memcpy(&rounded, &bits, sizeof(rounded));
  }
  return rounded;
}

void loom_value_facts_eval_float_fma(loom_scalar_type_t scalar_type,
                                     const loom_value_facts_t* a,
                                     const loom_value_facts_t* b,
                                     const loom_value_facts_t* c,
                                     loom_value_facts_t* out_facts) {
  double a_value = 0.0;
  double b_value = 0.0;
  double c_value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, *a, &a_value) ||
      !loom_value_facts_as_exact_float(scalar_type, *b, &b_value) ||
      !loom_value_facts_as_exact_float(scalar_type, *c, &c_value)) {
    *out_facts = loom_value_facts_unknown();
  } else if (scalar_type == LOOM_SCALAR_TYPE_F32) {
    *out_facts = loom_value_facts_exact_float(
        scalar_type,
        (double)fmaf((float)a_value, (float)b_value, (float)c_value));
  } else if (scalar_type == LOOM_SCALAR_TYPE_F64) {
    *out_facts = loom_value_facts_exact_float(scalar_type,
                                              fma(a_value, b_value, c_value));
  } else {
    *out_facts = loom_value_facts_exact_float(
        scalar_type, (double)loom_float_fma_narrow(a_value, b_value, c_value));
  }
  loom_value_facts_propagate_ternary_distribution(*a, *b, *c, out_facts);
}

// Reduces |input| to an angle in [-pi/4, pi/4] and returns its quadrant.
// The quarter-turn divisor is exactly representable, so remquo preserves the
// periodic position of every finite input while providing the low quotient
// bits needed to recover the quadrant.
static double loom_float_reduce_turns_f64(double input, int* out_quadrant) {
  int quotient = 0;
  const double residual = remquo(input, 0.25, &quotient);
  int quadrant = quotient % 4;
  if (quadrant < 0) {
    quadrant += 4;
  }
  *out_quadrant = quadrant;
  return residual * 6.2831853071795864769252867665590057683943387987502;
}

static float loom_float_eval_turns_f32(float input, const void* user_data) {
  const loom_float_turns_kind_t kind =
      *(const loom_float_turns_kind_t*)user_data;
  return kind == LOOM_FLOAT_TURNS_SIN ? iree_math_sin_turns_f32_approx(input)
                                      : iree_math_cos_turns_f32_approx(input);
}

static double loom_float_eval_turns_f64(double input, const void* user_data) {
  const loom_float_turns_kind_t kind =
      *(const loom_float_turns_kind_t*)user_data;
  int quadrant = 0;
  const double angle = loom_float_reduce_turns_f64(input, &quadrant);
  if (angle == 0.0) {
    if (kind == LOOM_FLOAT_TURNS_SIN) {
      if (quadrant == 1) {
        return 1.0;
      }
      if (quadrant == 3) {
        return -1.0;
      }
      return copysign(0.0, input);
    }
    if (quadrant == 0) {
      return 1.0;
    }
    if (quadrant == 2) {
      return -1.0;
    }
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

// Negation, absolute value, and sign replacement preserve the magnitude's
// floating-point class and exact destination formats. Their generic evaluators
// may otherwise discard these properties when the payload is not exact.
static void loom_value_facts_preserve_float_magnitude_properties(
    loom_value_facts_t magnitude, loom_value_facts_t* out_facts) {
  out_facts->flags |= magnitude.flags & LOOM_VALUE_FACT_FLOAT_PREDICATE_MASK;
  loom_value_facts_add_exact_float_types(
      out_facts, loom_value_facts_exact_float_types(magnitude));
}

void loom_value_facts_eval_float_negate(loom_scalar_type_t scalar_type,
                                        const loom_value_facts_t* input,
                                        loom_value_facts_t* out_facts) {
  if (loom_value_facts_is_nan(*input)) {
    loom_value_facts_make_unmaterializable_nan(*input, out_facts);
    loom_value_facts_preserve_float_magnitude_properties(*input, out_facts);
    return;
  }
  loom_value_facts_eval_float_unary(scalar_type, input, loom_float_negate_f32,
                                    loom_float_negate_f64, out_facts);
  loom_value_facts_preserve_float_magnitude_properties(*input, out_facts);
}

void loom_value_facts_eval_float_abs(loom_scalar_type_t scalar_type,
                                     const loom_value_facts_t* input,
                                     loom_value_facts_t* out_facts) {
  if (loom_value_facts_is_nan(*input)) {
    loom_value_facts_make_unmaterializable_nan(*input, out_facts);
    loom_value_facts_preserve_float_magnitude_properties(*input, out_facts);
    return;
  }
  loom_value_facts_eval_float_unary(scalar_type, input, loom_float_abs_f32,
                                    loom_float_abs_f64, out_facts);
  loom_value_facts_preserve_float_magnitude_properties(*input, out_facts);
}

void loom_value_facts_eval_float_copysign(loom_scalar_type_t scalar_type,
                                          const loom_value_facts_t* magnitude,
                                          const loom_value_facts_t* sign,
                                          loom_value_facts_t* out_facts) {
  if (loom_value_facts_is_nan(*magnitude)) {
    *out_facts = loom_value_facts_known_nan();
    loom_value_facts_preserve_float_magnitude_properties(*magnitude, out_facts);
    loom_value_facts_propagate_binary_distribution(*magnitude, *sign,
                                                   out_facts);
    return;
  }
  loom_value_facts_eval_float_binary(scalar_type, magnitude, sign,
                                     loom_float_copysign_f32,
                                     loom_float_copysign_f64, out_facts);
  loom_value_facts_preserve_float_magnitude_properties(*magnitude, out_facts);
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
    // Every min/max result is selected from its operands. Their meet retains
    // exactly the properties and interval that hold for either selection.
    loom_value_facts_meet_float(lhs, rhs, out_facts);
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
  loom_value_facts_t common_facts = loom_value_facts_unknown();
  loom_value_facts_meet_float(value, lower, &common_facts);
  loom_value_facts_meet_float(&common_facts, upper, &common_facts);
  if (loom_value_facts_is_nan(*value)) {
    *out_facts = *value;
  } else {
    double result = 0.0;
    if (!loom_value_facts_as_exact_float(scalar_type, *value, &result)) {
      *out_facts = common_facts;
      return;
    }
    if (!loom_value_facts_is_nan(*lower)) {
      double lower_value = 0.0;
      if (!loom_value_facts_as_exact_float(scalar_type, *lower, &lower_value)) {
        *out_facts = common_facts;
        return;
      }
      if (result < lower_value) {
        result = lower_value;
      }
    }
    if (!loom_value_facts_is_nan(*upper)) {
      double upper_value = 0.0;
      if (!loom_value_facts_as_exact_float(scalar_type, *upper, &upper_value)) {
        *out_facts = common_facts;
        return;
      }
      if (result > upper_value) {
        result = upper_value;
      }
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

static uint64_t loom_signed_integer_magnitude(int64_t value) {
  return value < 0 ? (uint64_t)(-(value + 1)) + UINT64_C(1) : (uint64_t)value;
}

static uint32_t loom_integer_fact_trailing_zero_count(
    loom_value_facts_t facts) {
  int64_t exact_value = 0;
  if (loom_value_facts_as_exact_i64(facts, &exact_value)) {
    const uint64_t magnitude = loom_signed_integer_magnitude(exact_value);
    return magnitude == 0
               ? 64u
               : (uint32_t)iree_math_count_trailing_zeros_u64(magnitude);
  }
  return (uint32_t)iree_math_count_trailing_zeros_u64(
      (uint64_t)facts.known_divisor);
}

static uint64_t loom_integer_maximum_aligned_magnitude(
    uint64_t maximum_magnitude, uint32_t trailing_zero_count) {
  if (trailing_zero_count >= 64u) {
    return 0;
  }
  return maximum_magnitude & (UINT64_MAX << trailing_zero_count);
}

static bool loom_float_integer_format_properties(
    loom_scalar_type_t scalar_type, uint32_t* out_precision,
    uint64_t* out_maximum_finite_integer) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_precision = 4;
      *out_maximum_finite_integer = 448;
      return true;
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_precision = 3;
      *out_maximum_finite_integer = 57344;
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_precision = 11;
      *out_maximum_finite_integer = 65504;
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_precision = 8;
      *out_maximum_finite_integer = UINT64_MAX;
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_precision = 24;
      *out_maximum_finite_integer = UINT64_MAX;
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_precision = 53;
      *out_maximum_finite_integer = UINT64_MAX;
      return true;
    default:
      return false;
  }
}

static loom_scalar_type_set_t loom_integer_exact_float_destinations(
    uint64_t maximum_magnitude, uint32_t trailing_zero_count) {
  maximum_magnitude = loom_integer_maximum_aligned_magnitude(
      maximum_magnitude, trailing_zero_count);
  loom_scalar_type_set_t result = LOOM_SCALAR_TYPE_SET_NONE;
  for (loom_scalar_type_t type = LOOM_SCALAR_TYPE_F8E4M3;
       type <= LOOM_SCALAR_TYPE_F64; ++type) {
    uint32_t precision = 0;
    uint64_t maximum_finite_integer = 0;
    if (!loom_float_integer_format_properties(type, &precision,
                                              &maximum_finite_integer) ||
        maximum_magnitude > maximum_finite_integer) {
      continue;
    }
    const uint32_t exact_magnitude_exponent = precision + trailing_zero_count;
    if (exact_magnitude_exponent >= 64u ||
        maximum_magnitude <= (UINT64_C(1) << exact_magnitude_exponent)) {
      result |= (loom_scalar_type_set_t)(1u << type);
    }
  }
  return result;
}

static loom_scalar_type_set_t loom_signed_integer_exact_float_destinations(
    loom_value_facts_t facts) {
  const uint32_t trailing_zero_count =
      loom_integer_fact_trailing_zero_count(facts);
  const uint64_t maximum_magnitude =
      iree_max(loom_signed_integer_magnitude(facts.range_lo),
               loom_signed_integer_magnitude(facts.range_hi));
  return loom_integer_exact_float_destinations(maximum_magnitude,
                                               trailing_zero_count);
}

static loom_scalar_type_set_t loom_unsigned_integer_exact_float_destinations(
    loom_value_facts_t facts, int32_t source_bit_count) {
  uint64_t exact_bits = 0;
  if (loom_value_facts_as_exact_raw_bits(facts, source_bit_count,
                                         &exact_bits)) {
    const uint32_t trailing_zero_count =
        exact_bits == 0
            ? 64u
            : (uint32_t)iree_math_count_trailing_zeros_u64(exact_bits);
    return loom_integer_exact_float_destinations(exact_bits,
                                                 trailing_zero_count);
  }

  const uint32_t trailing_zero_count =
      loom_integer_fact_trailing_zero_count(facts);
  uint64_t maximum_value = 0;
  if (facts.range_lo >= 0) {
    maximum_value = (uint64_t)facts.range_hi;
  } else if (facts.range_hi < 0) {
    maximum_value =
        iree_math_mask_low_bits_u64((uint64_t)facts.range_hi, source_bit_count);
  } else {
    maximum_value = iree_math_mask_low_bits_u64(UINT64_MAX, source_bit_count);
  }
  return loom_integer_exact_float_destinations(maximum_value,
                                               trailing_zero_count);
}

void loom_value_facts_eval_integer_to_float(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_float_integer_conversion_kind_t kind,
    const loom_value_facts_t* source_facts,
    loom_value_facts_t* out_result_facts) {
  *out_result_facts = loom_value_facts_unknown();
  if (!loom_scalar_type_is_integer(source_type) ||
      !loom_scalar_type_is_float(result_type)) {
    return;
  }

  int64_t source_domain_lo = 0;
  int64_t source_domain_hi = 0;
  if (!loom_value_facts_scalar_type_domain(source_type, &source_domain_lo,
                                           &source_domain_hi)) {
    return;
  }
  const int32_t source_bit_count = loom_scalar_type_bitwidth(source_type);
  loom_value_facts_t bounded = loom_value_facts_clamp_domain(
      *source_facts, source_domain_lo, source_domain_hi);

  double result_lo = 0.0;
  double result_hi = 0.0;
  bool has_rounded_range = false;
  loom_scalar_type_set_t integer_exact_types = LOOM_SCALAR_TYPE_SET_NONE;
  switch (kind) {
    case LOOM_FLOAT_INTEGER_CONVERSION_SIGNED: {
      bounded = loom_value_facts_sign_extend(bounded, source_bit_count);
      integer_exact_types =
          loom_signed_integer_exact_float_destinations(bounded);
      has_rounded_range = loom_signed_integer_to_rounded_float(
                              result_type, bounded.range_lo, &result_lo) &&
                          loom_signed_integer_to_rounded_float(
                              result_type, bounded.range_hi, &result_hi);
      break;
    }
    case LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED: {
      integer_exact_types = loom_unsigned_integer_exact_float_destinations(
          bounded, source_bit_count);
      uint64_t exact_bits = 0;
      if (loom_value_facts_as_exact_raw_bits(bounded, source_bit_count,
                                             &exact_bits)) {
        has_rounded_range = loom_unsigned_integer_to_rounded_float(
            result_type, exact_bits, &result_lo);
        result_hi = result_lo;
      } else if (bounded.range_lo >= 0) {
        has_rounded_range =
            loom_unsigned_integer_to_rounded_float(
                result_type, (uint64_t)bounded.range_lo, &result_lo) &&
            loom_unsigned_integer_to_rounded_float(
                result_type, (uint64_t)bounded.range_hi, &result_hi);
      } else if (bounded.range_hi < 0) {
        const uint64_t unsigned_lo = iree_math_mask_low_bits_u64(
            (uint64_t)bounded.range_lo, source_bit_count);
        const uint64_t unsigned_hi = iree_math_mask_low_bits_u64(
            (uint64_t)bounded.range_hi, source_bit_count);
        has_rounded_range = loom_unsigned_integer_to_rounded_float(
                                result_type, unsigned_lo, &result_lo) &&
                            loom_unsigned_integer_to_rounded_float(
                                result_type, unsigned_hi, &result_hi);
      } else {
        const uint64_t unsigned_hi =
            iree_math_mask_low_bits_u64(UINT64_MAX, source_bit_count);
        has_rounded_range = loom_unsigned_integer_to_rounded_float(
                                result_type, 0, &result_lo) &&
                            loom_unsigned_integer_to_rounded_float(
                                result_type, unsigned_hi, &result_hi);
      }
      break;
    }
    default:
      return;
  }

  *out_result_facts = has_rounded_range ? loom_value_facts_make_float_range(
                                              result_type, result_lo, result_hi)
                                        : loom_value_facts_unknown();
  if (!loom_value_facts_is_float(*out_result_facts)) {
    // A rounded endpoint may be infinite, or a wide integer may not have a
    // sound direct path to a narrow host representation. The conversion still
    // cannot produce NaN even when no finite interval can be retained.
    out_result_facts->flags = LOOM_VALUE_FACT_FLOAT | LOOM_VALUE_FACT_NOT_NAN;
  }
  loom_scalar_type_set_t result_exact_types =
      loom_float_type_exact_destinations(result_type);
  if (loom_scalar_type_set_contains(integer_exact_types, result_type)) {
    result_exact_types |= integer_exact_types;
  }
  loom_value_facts_add_exact_float_types(out_result_facts, result_exact_types);
  // Integer values are zero or have magnitude at least one, so their rounded
  // floating-point conversions cannot be subnormal.
  out_result_facts->flags |= LOOM_VALUE_FACT_NOT_SUBNORMAL;
  loom_value_facts_propagate_unary_distribution(*source_facts,
                                                out_result_facts);
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
  if (!loom_float_type_is_supported(scalar_type)) {
    return false;
  }
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
