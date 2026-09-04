// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Instantiates one exact-width implementation of the Core transcendental and
// compound floating-point operations. The including translation unit supplies
// the matching C type, bit constants, and libm suffix.

static IREE_VM_BYTECODE_FLOAT_MATH_TYPE IREE_VM_BYTECODE_FLOAT_MATH_NAME(
    from_bits)(IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE bits) {
  IREE_VM_BYTECODE_FLOAT_MATH_TYPE value =
      IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0);
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE IREE_VM_BYTECODE_FLOAT_MATH_NAME(
    to_bits)(IREE_VM_BYTECODE_FLOAT_MATH_TYPE value) {
  IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static IREE_VM_BYTECODE_FLOAT_MATH_TYPE IREE_VM_BYTECODE_FLOAT_MATH_NAME(
    logistic)(IREE_VM_BYTECODE_FLOAT_MATH_TYPE value) {
  if (value >= IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0)) {
    const IREE_VM_BYTECODE_FLOAT_MATH_TYPE exponent =
        IREE_VM_BYTECODE_FLOAT_MATH_LIBM(exp)(-value);
    return IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0) /
           (IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0) + exponent);
  }
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE exponent =
      IREE_VM_BYTECODE_FLOAT_MATH_LIBM(exp)(value);
  return exponent / (IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0) + exponent);
}

static IREE_VM_BYTECODE_FLOAT_MATH_TYPE IREE_VM_BYTECODE_FLOAT_MATH_NAME(turns)(
    IREE_VM_BYTECODE_FLOAT_MATH_TYPE value, bool selects_sine) {
  const IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE value_bits =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(to_bits)(value);
  if ((value_bits & IREE_VM_BYTECODE_FLOAT_MATH_MAGNITUDE_MASK) >=
      IREE_VM_BYTECODE_FLOAT_MATH_INFINITY_BITS) {
    return IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(
        IREE_VM_BYTECODE_FLOAT_MATH_QUIET_NAN_BITS);
  }
  int quotient = 0;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE residual =
      IREE_VM_BYTECODE_FLOAT_MATH_LIBM(remquo)(
          value, IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.25), &quotient);
  int quadrant = quotient % 4;
  if (quadrant < 0) quadrant += 4;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE angle =
      residual * IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(
                     6.2831853071795864769252867665590057683943387987502);
  if (angle == IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0)) {
    if (selects_sine) {
      if (quadrant == 1) return IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0);
      if (quadrant == 3) return -IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0);
      return IREE_VM_BYTECODE_FLOAT_MATH_LIBM(copysign)(
          IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0), value);
    }
    if (quadrant == 0) return IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0);
    if (quadrant == 2) return -IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0);
    return IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0);
  }
  if (selects_sine) {
    switch (quadrant) {
      case 0:
        return IREE_VM_BYTECODE_FLOAT_MATH_LIBM(sin)(angle);
      case 1:
        return IREE_VM_BYTECODE_FLOAT_MATH_LIBM(cos)(angle);
      case 2:
        return -IREE_VM_BYTECODE_FLOAT_MATH_LIBM(sin)(angle);
      default:
        return -IREE_VM_BYTECODE_FLOAT_MATH_LIBM(cos)(angle);
    }
  }
  switch (quadrant) {
    case 0:
      return IREE_VM_BYTECODE_FLOAT_MATH_LIBM(cos)(angle);
    case 1:
      return -IREE_VM_BYTECODE_FLOAT_MATH_LIBM(sin)(angle);
    case 2:
      return -IREE_VM_BYTECODE_FLOAT_MATH_LIBM(cos)(angle);
    default:
      return IREE_VM_BYTECODE_FLOAT_MATH_LIBM(sin)(angle);
  }
}

static IREE_VM_BYTECODE_FLOAT_MATH_TYPE IREE_VM_BYTECODE_FLOAT_MATH_NAME(
    gelu_erf)(IREE_VM_BYTECODE_FLOAT_MATH_TYPE value) {
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE inverse_sqrt2 =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(
          IREE_VM_BYTECODE_FLOAT_MATH_INVERSE_SQRT2_BITS);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE scaled = value * inverse_sqrt2;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE erf_value =
      IREE_VM_BYTECODE_FLOAT_MATH_LIBM(erf)(scaled);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE one_plus_erf =
      IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0) + erf_value;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE half_value =
      IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.5) * value;
  return half_value * one_plus_erf;
}

static IREE_VM_BYTECODE_FLOAT_MATH_TYPE IREE_VM_BYTECODE_FLOAT_MATH_NAME(
    gelu_tanh)(IREE_VM_BYTECODE_FLOAT_MATH_TYPE value) {
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE cubic_coefficient =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(
          IREE_VM_BYTECODE_FLOAT_MATH_GELU_CUBIC_BITS);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE sqrt_2_over_pi =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(
          IREE_VM_BYTECODE_FLOAT_MATH_SQRT_2_OVER_PI_BITS);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE value_squared = value * value;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE value_cubed = value_squared * value;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE cubic_term =
      cubic_coefficient * value_cubed;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE inner_sum = value + cubic_term;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE scaled = sqrt_2_over_pi * inner_sum;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE tanh_value =
      IREE_VM_BYTECODE_FLOAT_MATH_LIBM(tanh)(scaled);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE one_plus_tanh =
      IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0) + tanh_value;
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE half_value =
      IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.5) * value;
  return half_value * one_plus_tanh;
}

IREE_ATTRIBUTE_NOINLINE IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE
IREE_VM_BYTECODE_FLOAT_MATH_NAME(unary)(
    uint8_t selector, IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE source_bits) {
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE source =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(source_bits);
  IREE_VM_BYTECODE_FLOAT_MATH_TYPE result =
      IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0);
  switch (selector) {
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_EXP:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(exp)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_EXP2:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(exp2)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_EXPM1:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(expm1)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_LOG:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(log)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_LOG2:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(log2)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_LOG10:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(log10)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_LOG1P:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(log1p)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_SQRT:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(sqrt)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_RSQRT: {
      const IREE_VM_BYTECODE_FLOAT_MATH_TYPE root =
          IREE_VM_BYTECODE_FLOAT_MATH_LIBM(sqrt)(source);
      result = IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(1.0) / root;
      break;
    }
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_CBRT:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(cbrt)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_SIN:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(sin)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_COS:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(cos)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_SINTURNS:
      result = IREE_VM_BYTECODE_FLOAT_MATH_NAME(turns)(source, true);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_COSTURNS:
      result = IREE_VM_BYTECODE_FLOAT_MATH_NAME(turns)(source, false);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_TAN:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(tan)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ASIN:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(asin)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ACOS:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(acos)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ATAN:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(atan)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_SINH:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(sinh)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_COSH:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(cosh)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_TANH:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(tanh)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ASINH:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(asinh)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ACOSH:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(acosh)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ATANH:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(atanh)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ERF:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(erf)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ERFC:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(erfc)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_LOGISTIC:
      result = IREE_VM_BYTECODE_FLOAT_MATH_NAME(logistic)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_SILU: {
      const IREE_VM_BYTECODE_FLOAT_MATH_TYPE logistic =
          IREE_VM_BYTECODE_FLOAT_MATH_NAME(logistic)(source);
      result = source * logistic;
      break;
    }
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_SOFTPLUS: {
      const IREE_VM_BYTECODE_FLOAT_MATH_TYPE positive =
          source > IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0)
              ? source
              : IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0);
      const IREE_VM_BYTECODE_FLOAT_MATH_TYPE exponent =
          IREE_VM_BYTECODE_FLOAT_MATH_LIBM(exp)(
              -IREE_VM_BYTECODE_FLOAT_MATH_LIBM(fabs)(source));
      const IREE_VM_BYTECODE_FLOAT_MATH_TYPE correction =
          IREE_VM_BYTECODE_FLOAT_MATH_LIBM(log1p)(exponent);
      result = positive + correction;
      break;
    }
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_CEIL:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(ceil)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_FLOOR:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(floor)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ROUND:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(round)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_ROUNDEVEN:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(nearbyint)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_TRUNC:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(trunc)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_SIGN: {
      const IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE magnitude =
          source_bits & IREE_VM_BYTECODE_FLOAT_MATH_MAGNITUDE_MASK;
      if (magnitude == 0 ||
          magnitude > IREE_VM_BYTECODE_FLOAT_MATH_INFINITY_BITS) {
        return 0;
      }
      return source_bits & IREE_VM_BYTECODE_FLOAT_MATH_SIGN_MASK
                 ? IREE_VM_BYTECODE_FLOAT_MATH_NEGATIVE_ONE_BITS
                 : IREE_VM_BYTECODE_FLOAT_MATH_POSITIVE_ONE_BITS;
    }
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_GELU_ERF:
      result = IREE_VM_BYTECODE_FLOAT_MATH_NAME(gelu_erf)(source);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_UNARY_GELU_TANH:
      result = IREE_VM_BYTECODE_FLOAT_MATH_NAME(gelu_tanh)(source);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("float.math.unary selector was not verified");
      return 0;
  }
  return IREE_VM_BYTECODE_FLOAT_MATH_NAME(to_bits)(result);
}

IREE_ATTRIBUTE_NOINLINE IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE
IREE_VM_BYTECODE_FLOAT_MATH_NAME(binary)(
    uint8_t selector, IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE lhs_bits,
    IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE rhs_bits) {
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE lhs =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(lhs_bits);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE rhs =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(rhs_bits);
  IREE_VM_BYTECODE_FLOAT_MATH_TYPE result =
      IREE_VM_BYTECODE_FLOAT_MATH_LITERAL(0.0);
  switch (selector) {
    case IREE_VM_BYTECODE_FLOAT_MATH_BINARY_POW:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(pow)(lhs, rhs);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_BINARY_ATAN2:
      result = IREE_VM_BYTECODE_FLOAT_MATH_LIBM(atan2)(lhs, rhs);
      break;
    case IREE_VM_BYTECODE_FLOAT_MATH_BINARY_GELU_LOGISTIC: {
      const IREE_VM_BYTECODE_FLOAT_MATH_TYPE scaled = rhs * lhs;
      const IREE_VM_BYTECODE_FLOAT_MATH_TYPE logistic =
          IREE_VM_BYTECODE_FLOAT_MATH_NAME(logistic)(scaled);
      result = lhs * logistic;
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("float.math.binary selector was not verified");
      return 0;
  }
  return IREE_VM_BYTECODE_FLOAT_MATH_NAME(to_bits)(result);
}

IREE_ATTRIBUTE_NOINLINE IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE
IREE_VM_BYTECODE_FLOAT_MATH_NAME(ternary)(
    uint8_t selector, IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE a_bits,
    IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE b_bits,
    IREE_VM_BYTECODE_FLOAT_MATH_BITS_TYPE c_bits) {
  IREE_ASSERT_EQ(selector, IREE_VM_BYTECODE_FLOAT_MATH_TERNARY_FMA);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE a =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(a_bits);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE b =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(b_bits);
  const IREE_VM_BYTECODE_FLOAT_MATH_TYPE c =
      IREE_VM_BYTECODE_FLOAT_MATH_NAME(from_bits)(c_bits);
  return IREE_VM_BYTECODE_FLOAT_MATH_NAME(to_bits)(
      IREE_VM_BYTECODE_FLOAT_MATH_LIBM(fma)(a, b, c));
}
