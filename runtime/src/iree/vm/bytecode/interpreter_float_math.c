// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_float_math.h"

#include <math.h>
#include <string.h>

#include "iree/vm/bytecode/wire/core/float.h"

static float iree_vm_bytecode_float_math_f32_from_bits(uint32_t bits) {
  float value = 0.0f;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t iree_vm_bytecode_float_math_f32_to_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static double iree_vm_bytecode_float_math_f64_from_bits(uint64_t bits) {
  double value = 0.0;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint64_t iree_vm_bytecode_float_math_f64_to_bits(double value) {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static uint32_t iree_vm_bytecode_float_math_f32_canonicalize_nan(
    uint32_t bits) {
  const uint32_t magnitude = bits & UINT32_C(0x7FFFFFFF);
  return IREE_UNLIKELY(magnitude > UINT32_C(0x7F800000)) ? UINT32_C(0x7FC00000)
                                                         : bits;
}

static uint64_t iree_vm_bytecode_float_math_f64_canonicalize_nan(
    uint64_t bits) {
  const uint64_t magnitude = bits & UINT64_C(0x7FFFFFFFFFFFFFFF);
  return IREE_UNLIKELY(magnitude > UINT64_C(0x7FF0000000000000))
             ? UINT64_C(0x7FF8000000000000)
             : bits;
}

static float iree_vm_bytecode_float_math_logistic_f32(float value) {
  if (value >= 0.0f) {
    const float exponent = expf(-value);
    return 1.0f / (1.0f + exponent);
  }
  const float exponent = expf(value);
  return exponent / (1.0f + exponent);
}

static double iree_vm_bytecode_float_math_logistic_f64(double value) {
  if (value >= 0.0) {
    const double exponent = exp(-value);
    return 1.0 / (1.0 + exponent);
  }
  const double exponent = exp(value);
  return exponent / (1.0 + exponent);
}

static float iree_vm_bytecode_float_math_turns_f32(float value,
                                                   bool selects_sine) {
  const uint32_t value_bits = iree_vm_bytecode_float_math_f32_to_bits(value);
  if ((value_bits & UINT32_C(0x7FFFFFFF)) >= UINT32_C(0x7F800000)) {
    return iree_vm_bytecode_float_math_f32_from_bits(UINT32_C(0x7FC00000));
  }
  int quotient = 0;
  const float residual = remquof(value, 0.25f, &quotient);
  int quadrant = quotient % 4;
  if (quadrant < 0) quadrant += 4;
  const float angle =
      residual * 6.2831853071795864769252867665590057683943387987502f;
  if (angle == 0.0f) {
    if (selects_sine) {
      if (quadrant == 1) return 1.0f;
      if (quadrant == 3) return -1.0f;
      return copysignf(0.0f, value);
    }
    if (quadrant == 0) return 1.0f;
    if (quadrant == 2) return -1.0f;
    return 0.0f;
  }
  if (selects_sine) {
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

static double iree_vm_bytecode_float_math_turns_f64(double value,
                                                    bool selects_sine) {
  const uint64_t value_bits = iree_vm_bytecode_float_math_f64_to_bits(value);
  if ((value_bits & UINT64_C(0x7FFFFFFFFFFFFFFF)) >=
      UINT64_C(0x7FF0000000000000)) {
    return iree_vm_bytecode_float_math_f64_from_bits(
        UINT64_C(0x7FF8000000000000));
  }
  int quotient = 0;
  const double residual = remquo(value, 0.25, &quotient);
  int quadrant = quotient % 4;
  if (quadrant < 0) quadrant += 4;
  const double angle =
      residual * 6.2831853071795864769252867665590057683943387987502;
  if (angle == 0.0) {
    if (selects_sine) {
      if (quadrant == 1) return 1.0;
      if (quadrant == 3) return -1.0;
      return copysign(0.0, value);
    }
    if (quadrant == 0) return 1.0;
    if (quadrant == 2) return -1.0;
    return 0.0;
  }
  if (selects_sine) {
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

static float iree_vm_bytecode_float_math_gelu_erf_f32(float value) {
  const float inverse_sqrt2 =
      iree_vm_bytecode_float_math_f32_from_bits(UINT32_C(0x3F3504F3));
  const float scaled = value * inverse_sqrt2;
  const float erf_value = erff(scaled);
  const float one_plus_erf = 1.0f + erf_value;
  const float half_value = 0.5f * value;
  return half_value * one_plus_erf;
}

static double iree_vm_bytecode_float_math_gelu_erf_f64(double value) {
  const double inverse_sqrt2 =
      iree_vm_bytecode_float_math_f64_from_bits(UINT64_C(0x3FE6A09E667F3BCD));
  const double scaled = value * inverse_sqrt2;
  const double erf_value = erf(scaled);
  const double one_plus_erf = 1.0 + erf_value;
  const double half_value = 0.5 * value;
  return half_value * one_plus_erf;
}

static float iree_vm_bytecode_float_math_gelu_tanh_f32(float value) {
  const float cubic_coefficient =
      iree_vm_bytecode_float_math_f32_from_bits(UINT32_C(0x3D372713));
  const float sqrt_2_over_pi =
      iree_vm_bytecode_float_math_f32_from_bits(UINT32_C(0x3F4C422A));
  const float value_squared = value * value;
  const float value_cubed = value_squared * value;
  const float cubic_term = cubic_coefficient * value_cubed;
  const float inner_sum = value + cubic_term;
  const float scaled = sqrt_2_over_pi * inner_sum;
  const float tanh_value = tanhf(scaled);
  const float one_plus_tanh = 1.0f + tanh_value;
  const float half_value = 0.5f * value;
  return half_value * one_plus_tanh;
}

static double iree_vm_bytecode_float_math_gelu_tanh_f64(double value) {
  const double cubic_coefficient =
      iree_vm_bytecode_float_math_f64_from_bits(UINT64_C(0x3FA6E4E26D4801F7));
  const double sqrt_2_over_pi =
      iree_vm_bytecode_float_math_f64_from_bits(UINT64_C(0x3FE9884533D43651));
  const double value_squared = value * value;
  const double value_cubed = value_squared * value;
  const double cubic_term = cubic_coefficient * value_cubed;
  const double inner_sum = value + cubic_term;
  const double scaled = sqrt_2_over_pi * inner_sum;
  const double tanh_value = tanh(scaled);
  const double one_plus_tanh = 1.0 + tanh_value;
  const double half_value = 0.5 * value;
  return half_value * one_plus_tanh;
}

IREE_ATTRIBUTE_NOINLINE uint32_t
iree_vm_bytecode_float_math_unary_f32(uint8_t selector, uint32_t source_bits) {
  const float source = iree_vm_bytecode_float_math_f32_from_bits(source_bits);
  float result = 0.0f;
  switch (selector) {
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXP:
      result = expf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2:
      result = exp2f(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1:
      result = expm1f(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG:
      result = logf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG2:
      result = log2f(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG10:
      result = log10f(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P:
      result = log1pf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT:
      result = sqrtf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT: {
      const float root = sqrtf(source);
      result = 1.0f / root;
      break;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT:
      result = cbrtf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SIN:
      result = sinf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COS:
      result = cosf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS:
      result = iree_vm_bytecode_float_math_turns_f32(source, true);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS:
      result = iree_vm_bytecode_float_math_turns_f32(source, false);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TAN:
      result = tanf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ASIN:
      result = asinf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ACOS:
      result = acosf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN:
      result = atanf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SINH:
      result = sinhf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COSH:
      result = coshf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TANH:
      result = tanhf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH:
      result = asinhf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ACOSH:
      result = acoshf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH:
      result = atanhf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ERF:
      result = erff(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ERFC:
      result = erfcf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC:
      result = iree_vm_bytecode_float_math_logistic_f32(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SILU: {
      const float logistic = iree_vm_bytecode_float_math_logistic_f32(source);
      result = source * logistic;
      break;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS: {
      const float positive = source > 0.0f ? source : 0.0f;
      const float exponent = expf(-fabsf(source));
      const float correction = log1pf(exponent);
      result = positive + correction;
      break;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_CEIL:
      result = ceilf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_FLOOR:
      result = floorf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ROUND:
      result = roundf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ROUNDEVEN:
      result = nearbyintf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TRUNC:
      result = truncf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN: {
      const uint32_t magnitude = source_bits & UINT32_C(0x7FFFFFFF);
      if (magnitude == 0 || magnitude > UINT32_C(0x7F800000)) return 0;
      return source_bits & UINT32_C(0x80000000) ? UINT32_C(0xBF800000)
                                                : UINT32_C(0x3F800000);
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_ERF:
      result = iree_vm_bytecode_float_math_gelu_erf_f32(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH:
      result = iree_vm_bytecode_float_math_gelu_tanh_f32(source);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("float.math.unary selector was not verified");
      return 0;
  }
  return iree_vm_bytecode_float_math_f32_canonicalize_nan(
      iree_vm_bytecode_float_math_f32_to_bits(result));
}

IREE_ATTRIBUTE_NOINLINE uint64_t
iree_vm_bytecode_float_math_unary_f64(uint8_t selector, uint64_t source_bits) {
  const double source = iree_vm_bytecode_float_math_f64_from_bits(source_bits);
  double result = 0.0;
  switch (selector) {
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXP:
      result = exp(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2:
      result = exp2(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1:
      result = expm1(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG:
      result = log(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG2:
      result = log2(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG10:
      result = log10(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P:
      result = log1p(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT:
      result = sqrt(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT: {
      const double root = sqrt(source);
      result = 1.0 / root;
      break;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT:
      result = cbrt(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SIN:
      result = sin(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COS:
      result = cos(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS:
      result = iree_vm_bytecode_float_math_turns_f64(source, true);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS:
      result = iree_vm_bytecode_float_math_turns_f64(source, false);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TAN:
      result = tan(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ASIN:
      result = asin(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ACOS:
      result = acos(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN:
      result = atan(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SINH:
      result = sinh(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COSH:
      result = cosh(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TANH:
      result = tanh(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH:
      result = asinh(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ACOSH:
      result = acosh(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH:
      result = atanh(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ERF:
      result = erf(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ERFC:
      result = erfc(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC:
      result = iree_vm_bytecode_float_math_logistic_f64(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SILU: {
      const double logistic = iree_vm_bytecode_float_math_logistic_f64(source);
      result = source * logistic;
      break;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS: {
      const double positive = source > 0.0 ? source : 0.0;
      const double exponent = exp(-fabs(source));
      const double correction = log1p(exponent);
      result = positive + correction;
      break;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_CEIL:
      result = ceil(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_FLOOR:
      result = floor(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ROUND:
      result = round(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ROUNDEVEN:
      result = nearbyint(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TRUNC:
      result = trunc(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN: {
      const uint64_t magnitude = source_bits & UINT64_C(0x7FFFFFFFFFFFFFFF);
      if (magnitude == 0 || magnitude > UINT64_C(0x7FF0000000000000)) return 0;
      return source_bits & UINT64_C(0x8000000000000000)
                 ? UINT64_C(0xBFF0000000000000)
                 : UINT64_C(0x3FF0000000000000);
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_ERF:
      result = iree_vm_bytecode_float_math_gelu_erf_f64(source);
      break;
    case IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH:
      result = iree_vm_bytecode_float_math_gelu_tanh_f64(source);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("float.math.unary selector was not verified");
      return 0;
  }
  return iree_vm_bytecode_float_math_f64_canonicalize_nan(
      iree_vm_bytecode_float_math_f64_to_bits(result));
}

IREE_ATTRIBUTE_NOINLINE uint32_t iree_vm_bytecode_float_math_binary_f32(
    uint8_t selector, uint32_t lhs_bits, uint32_t rhs_bits) {
  const float lhs = iree_vm_bytecode_float_math_f32_from_bits(lhs_bits);
  const float rhs = iree_vm_bytecode_float_math_f32_from_bits(rhs_bits);
  float result = 0.0f;
  switch (selector) {
    case IREE_VM_ISA_FLOAT_MATH_BINARY_POW:
      result = powf(lhs, rhs);
      break;
    case IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2:
      result = atan2f(lhs, rhs);
      break;
    case IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC: {
      const float scaled = rhs * lhs;
      const float logistic = iree_vm_bytecode_float_math_logistic_f32(scaled);
      result = lhs * logistic;
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("float.math.binary selector was not verified");
      return 0;
  }
  return iree_vm_bytecode_float_math_f32_canonicalize_nan(
      iree_vm_bytecode_float_math_f32_to_bits(result));
}

IREE_ATTRIBUTE_NOINLINE uint64_t iree_vm_bytecode_float_math_binary_f64(
    uint8_t selector, uint64_t lhs_bits, uint64_t rhs_bits) {
  const double lhs = iree_vm_bytecode_float_math_f64_from_bits(lhs_bits);
  const double rhs = iree_vm_bytecode_float_math_f64_from_bits(rhs_bits);
  double result = 0.0;
  switch (selector) {
    case IREE_VM_ISA_FLOAT_MATH_BINARY_POW:
      result = pow(lhs, rhs);
      break;
    case IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2:
      result = atan2(lhs, rhs);
      break;
    case IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC: {
      const double scaled = rhs * lhs;
      const double logistic = iree_vm_bytecode_float_math_logistic_f64(scaled);
      result = lhs * logistic;
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("float.math.binary selector was not verified");
      return 0;
  }
  return iree_vm_bytecode_float_math_f64_canonicalize_nan(
      iree_vm_bytecode_float_math_f64_to_bits(result));
}

IREE_ATTRIBUTE_NOINLINE uint32_t iree_vm_bytecode_float_math_ternary_f32(
    uint8_t selector, uint32_t a_bits, uint32_t b_bits, uint32_t c_bits) {
  IREE_ASSERT_EQ(selector, IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA);
  const float a = iree_vm_bytecode_float_math_f32_from_bits(a_bits);
  const float b = iree_vm_bytecode_float_math_f32_from_bits(b_bits);
  const float c = iree_vm_bytecode_float_math_f32_from_bits(c_bits);
  return iree_vm_bytecode_float_math_f32_canonicalize_nan(
      iree_vm_bytecode_float_math_f32_to_bits(fmaf(a, b, c)));
}

IREE_ATTRIBUTE_NOINLINE uint64_t iree_vm_bytecode_float_math_ternary_f64(
    uint8_t selector, uint64_t a_bits, uint64_t b_bits, uint64_t c_bits) {
  IREE_ASSERT_EQ(selector, IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA);
  const double a = iree_vm_bytecode_float_math_f64_from_bits(a_bits);
  const double b = iree_vm_bytecode_float_math_f64_from_bits(b_bits);
  const double c = iree_vm_bytecode_float_math_f64_from_bits(c_bits);
  return iree_vm_bytecode_float_math_f64_canonicalize_nan(
      iree_vm_bytecode_float_math_f64_to_bits(fma(a, b, c)));
}
