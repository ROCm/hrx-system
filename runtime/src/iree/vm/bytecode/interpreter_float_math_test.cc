// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_float_math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "iree/base/internal/fpu_state.h"
#include "iree/testing/gtest.h"
#include "iree/vm/bytecode/wire/core/float.h"

namespace iree::vm::bytecode::testing {
namespace {

template <typename Float>
struct MathTraits;

template <>
struct MathTraits<float> {
  using Bits = uint32_t;

  static Bits Unary(uint8_t selector, Bits source) {
    return iree_vm_bytecode_float_math_unary_f32(selector, source);
  }
  static Bits Binary(uint8_t selector, Bits lhs, Bits rhs) {
    return iree_vm_bytecode_float_math_binary_f32(selector, lhs, rhs);
  }
  static Bits Ternary(uint8_t selector, Bits a, Bits b, Bits c) {
    return iree_vm_bytecode_float_math_ternary_f32(selector, a, b, c);
  }

  static constexpr Bits kSign = UINT32_C(0x80000000);
  static constexpr Bits kExponent = UINT32_C(0x7F800000);
  static constexpr Bits kSignificand = UINT32_C(0x007FFFFF);
  static constexpr Bits kQuiet = UINT32_C(0x00400000);
  static constexpr Bits kSignalingNaN = UINT32_C(0x7F800001);
  static constexpr Bits kInverseSqrt2 = UINT32_C(0x3F3504F3);
  static constexpr Bits kGeluCubicCoefficient = UINT32_C(0x3D372713);
  static constexpr Bits kSqrtTwoOverPi = UINT32_C(0x3F4C422A);
};

template <>
struct MathTraits<double> {
  using Bits = uint64_t;

  static Bits Unary(uint8_t selector, Bits source) {
    return iree_vm_bytecode_float_math_unary_f64(selector, source);
  }
  static Bits Binary(uint8_t selector, Bits lhs, Bits rhs) {
    return iree_vm_bytecode_float_math_binary_f64(selector, lhs, rhs);
  }
  static Bits Ternary(uint8_t selector, Bits a, Bits b, Bits c) {
    return iree_vm_bytecode_float_math_ternary_f64(selector, a, b, c);
  }

  static constexpr Bits kSign = UINT64_C(0x8000000000000000);
  static constexpr Bits kExponent = UINT64_C(0x7FF0000000000000);
  static constexpr Bits kSignificand = UINT64_C(0x000FFFFFFFFFFFFF);
  static constexpr Bits kQuiet = UINT64_C(0x0008000000000000);
  static constexpr Bits kSignalingNaN = UINT64_C(0x7FF0000000000001);
  static constexpr Bits kInverseSqrt2 = UINT64_C(0x3FE6A09E667F3BCD);
  static constexpr Bits kGeluCubicCoefficient = UINT64_C(0x3FA6E4E26D4801F7);
  static constexpr Bits kSqrtTwoOverPi = UINT64_C(0x3FE9884533D43651);
};

template <typename Float>
using FloatBits = typename MathTraits<Float>::Bits;

template <typename Float>
FloatBits<Float> ToBits(Float value) {
  FloatBits<Float> bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename Float>
Float FromBits(FloatBits<Float> bits) {
  Float value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename Float>
Float EvaluateUnary(uint8_t selector, Float source) {
  return FromBits<Float>(MathTraits<Float>::Unary(selector, ToBits(source)));
}

template <typename Float>
Float EvaluateBinary(uint8_t selector, Float lhs, Float rhs) {
  return FromBits<Float>(
      MathTraits<Float>::Binary(selector, ToBits(lhs), ToBits(rhs)));
}

template <typename Float>
Float EvaluateTernary(uint8_t selector, Float a, Float b, Float c) {
  return FromBits<Float>(
      MathTraits<Float>::Ternary(selector, ToBits(a), ToBits(b), ToBits(c)));
}

template <typename Float>
void ExpectBits(Float actual, Float expected) {
  EXPECT_EQ(ToBits(actual), ToBits(expected));
}

template <typename Float>
void ExpectHostMathNear(Float actual, Float expected) {
  const Float scale = std::max(std::fabs(expected), Float{1});
  const Float tolerance =
      Float{4} * std::numeric_limits<Float>::epsilon() * scale;
  EXPECT_NEAR(actual, expected, tolerance);
}

bool IsWidthMatchedHostMathSelector(uint8_t selector) {
  return selector <= IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT ||
         selector == IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT ||
         (selector >= IREE_VM_ISA_FLOAT_MATH_UNARY_SIN &&
          selector <= IREE_VM_ISA_FLOAT_MATH_UNARY_COS) ||
         (selector >= IREE_VM_ISA_FLOAT_MATH_UNARY_TAN &&
          selector <= IREE_VM_ISA_FLOAT_MATH_UNARY_ERFC);
}

template <typename Float>
void ExpectArithmeticNaN(Float actual) {
  using Traits = MathTraits<Float>;
  const FloatBits<Float> bits = ToBits(actual);
  EXPECT_EQ(bits & Traits::kExponent, Traits::kExponent);
  EXPECT_NE(bits & Traits::kSignificand, 0u);
  EXPECT_NE(bits & Traits::kQuiet, 0u);
}

template <typename Float>
void ExpectCanonicalArithmeticNaN(Float actual) {
  using Traits = MathTraits<Float>;
  const FloatBits<Float> bits = ToBits(actual);
  EXPECT_EQ(bits & ~Traits::kSign, Traits::kExponent | Traits::kQuiet);
}

template <typename Float>
Float ReferenceLogistic(Float value) {
  if (value >= Float{0}) {
    const Float exponent = std::exp(-value);
    return Float{1} / (Float{1} + exponent);
  }
  const Float exponent = std::exp(value);
  return exponent / (Float{1} + exponent);
}

template <typename Float>
Float ReferenceTurns(Float value, bool selects_sine) {
  int quotient = 0;
  const Float residual = std::remquo(value, Float{0.25}, &quotient);
  int quadrant = quotient % 4;
  if (quadrant < 0) quadrant += 4;
  const Float angle =
      residual * static_cast<Float>(6.283185307179586476925286766559005768L);
  if (angle == Float{0}) {
    if (selects_sine) {
      if (quadrant == 1) return Float{1};
      if (quadrant == 3) return Float{-1};
      return std::copysign(Float{0}, value);
    }
    if (quadrant == 0) return Float{1};
    if (quadrant == 2) return Float{-1};
    return Float{0};
  }
  if (selects_sine) {
    switch (quadrant) {
      case 0:
        return std::sin(angle);
      case 1:
        return std::cos(angle);
      case 2:
        return -std::sin(angle);
      default:
        return -std::cos(angle);
    }
  }
  switch (quadrant) {
    case 0:
      return std::cos(angle);
    case 1:
      return -std::sin(angle);
    case 2:
      return -std::cos(angle);
    default:
      return std::sin(angle);
  }
}

template <typename Float>
Float ReferenceGeluErf(Float value) {
  const Float inverse_sqrt2 = FromBits<Float>(MathTraits<Float>::kInverseSqrt2);
  const Float scaled = value * inverse_sqrt2;
  const Float erf_value = std::erf(scaled);
  const Float one_plus_erf = Float{1} + erf_value;
  const Float half_value = Float{0.5} * value;
  return half_value * one_plus_erf;
}

template <typename Float>
Float ReferenceGeluTanh(Float value) {
  const Float cubic_coefficient =
      FromBits<Float>(MathTraits<Float>::kGeluCubicCoefficient);
  const Float sqrt_two_over_pi =
      FromBits<Float>(MathTraits<Float>::kSqrtTwoOverPi);
  const Float value_squared = value * value;
  const Float value_cubed = value_squared * value;
  const Float cubic_term = cubic_coefficient * value_cubed;
  const Float inner_sum = value + cubic_term;
  const Float scaled = sqrt_two_over_pi * inner_sum;
  const Float tanh_value = std::tanh(scaled);
  const Float one_plus_tanh = Float{1} + tanh_value;
  const Float half_value = Float{0.5} * value;
  return half_value * one_plus_tanh;
}

template <typename Float>
Float ReferenceUnary(uint8_t selector, Float value) {
  switch (selector) {
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXP:
      return std::exp(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2:
      return std::exp2(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1:
      return std::expm1(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG:
      return std::log(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG2:
      return std::log2(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG10:
      return std::log10(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P:
      return std::log1p(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT:
      return std::sqrt(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT: {
      const Float root = std::sqrt(value);
      return Float{1} / root;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT:
      return std::cbrt(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SIN:
      return std::sin(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COS:
      return std::cos(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS:
      return ReferenceTurns(value, true);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS:
      return ReferenceTurns(value, false);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TAN:
      return std::tan(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ASIN:
      return std::asin(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ACOS:
      return std::acos(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN:
      return std::atan(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SINH:
      return std::sinh(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_COSH:
      return std::cosh(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TANH:
      return std::tanh(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH:
      return std::asinh(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ACOSH:
      return std::acosh(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH:
      return std::atanh(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ERF:
      return std::erf(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ERFC:
      return std::erfc(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC:
      return ReferenceLogistic(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SILU: {
      const Float logistic = ReferenceLogistic(value);
      return value * logistic;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS: {
      const Float positive = value > Float{0} ? value : Float{0};
      const Float exponent = std::exp(-std::fabs(value));
      const Float correction = std::log1p(exponent);
      return positive + correction;
    }
    case IREE_VM_ISA_FLOAT_MATH_UNARY_CEIL:
      return std::ceil(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_FLOOR:
      return std::floor(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ROUND:
      return std::round(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_ROUNDEVEN:
      return std::nearbyint(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_TRUNC:
      return std::trunc(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN:
      if (std::isnan(value) || value == Float{0}) return Float{0};
      return std::signbit(value) ? Float{-1} : Float{1};
    case IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_ERF:
      return ReferenceGeluErf(value);
    case IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH:
      return ReferenceGeluTanh(value);
  }
  return std::numeric_limits<Float>::quiet_NaN();
}

template <typename Float>
void CheckEverySelector() {
  constexpr Float kInput = Float{0.5};
  for (uint8_t selector = IREE_VM_ISA_FLOAT_MATH_UNARY_EXP;
       selector <= IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH; ++selector) {
    SCOPED_TRACE(static_cast<unsigned>(selector));
    const Float expected = ReferenceUnary(selector, kInput);
    const Float actual = EvaluateUnary(selector, kInput);
    if (std::isnan(expected)) {
      ExpectArithmeticNaN(actual);
    } else if (IsWidthMatchedHostMathSelector(selector)) {
      ExpectHostMathNear(actual, expected);
    } else {
      ExpectBits(actual, expected);
    }
  }

  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW,
                                   Float{0.5}, Float{3}),
             std::pow(Float{0.5}, Float{3}));
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2,
                                   Float{0.5}, Float{3}),
             std::atan2(Float{0.5}, Float{3}));
  const Float scaled = Float{1.702} * kInput;
  const Float logistic = ReferenceLogistic(scaled);
  const Float gelu_logistic = kInput * logistic;
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC,
                                   kInput, Float{1.702}),
             gelu_logistic);
  ExpectBits(EvaluateTernary<Float>(IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA,
                                    Float{0.5}, Float{3}, Float{-1}),
             std::fma(Float{0.5}, Float{3}, Float{-1}));
}

template <typename Float>
void CheckExceptionalValues() {
  const Float zero = Float{0};
  const Float negative_zero = -Float{0};
  const Float infinity = std::numeric_limits<Float>::infinity();
  const Float quiet_nan = std::numeric_limits<Float>::quiet_NaN();

  ExpectCanonicalArithmeticNaN(
      EvaluateUnary<Float>(IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT, Float{-1}));
  ExpectCanonicalArithmeticNaN(
      EvaluateUnary<Float>(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP, quiet_nan));

  const auto exact = [](uint8_t selector, Float input, Float expected) {
    ExpectBits(EvaluateUnary(selector, input), expected);
  };
  const auto arithmetic_nan = [](uint8_t selector, Float input) {
    ExpectArithmeticNaN(EvaluateUnary(selector, input));
  };

  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP, -infinity, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP, infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP, zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP, negative_zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2, -infinity, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2, infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2, zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2, negative_zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1, -infinity, Float{-1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1, infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1, negative_zero, negative_zero);

  for (uint8_t selector :
       {IREE_VM_ISA_FLOAT_MATH_UNARY_LOG, IREE_VM_ISA_FLOAT_MATH_UNARY_LOG2,
        IREE_VM_ISA_FLOAT_MATH_UNARY_LOG10}) {
    exact(selector, zero, -infinity);
    exact(selector, negative_zero, -infinity);
    exact(selector, infinity, infinity);
    arithmetic_nan(selector, Float{-1});
  }
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P, Float{-1}, -infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P, infinity, infinity);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P, Float{-2});

  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT, infinity, infinity);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT, Float{-1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT, zero, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT, negative_zero, -infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT, infinity, zero);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT, Float{-1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT, infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT, -infinity, -infinity);

  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SIN, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SIN, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COS, zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COS, negative_zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_TAN, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_TAN, negative_zero, negative_zero);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_SIN, infinity);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_COS, infinity);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_TAN, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ASIN, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ASIN, negative_zero, negative_zero);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_ASIN, Float{2});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ACOS, Float{1}, zero);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_ACOS, Float{2});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN, infinity, std::atan(infinity));
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN, -infinity, std::atan(-infinity));

  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINH, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINH, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINH, infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINH, -infinity, -infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSH, zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSH, negative_zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSH, infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSH, -infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_TANH, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_TANH, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_TANH, infinity, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_TANH, -infinity, Float{-1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH, infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH, -infinity, -infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ACOSH, Float{1}, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ACOSH, infinity, infinity);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_ACOSH, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH, Float{-1}, -infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH, Float{1}, infinity);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH, Float{2});

  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ERF, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ERF, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ERF, infinity, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ERF, -infinity, Float{-1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ERFC, -infinity, Float{2});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ERFC, infinity, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC, -infinity, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC, zero, Float{0.5});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC, negative_zero, Float{0.5});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC, infinity, Float{1});
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_SILU, -infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SILU, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SILU, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SILU, infinity, infinity);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS, -infinity, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS, zero, std::log1p(Float{1}));
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS, negative_zero,
        std::log1p(Float{1}));
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS, infinity, infinity);

  for (uint8_t selector :
       {IREE_VM_ISA_FLOAT_MATH_UNARY_CEIL, IREE_VM_ISA_FLOAT_MATH_UNARY_FLOOR,
        IREE_VM_ISA_FLOAT_MATH_UNARY_ROUND,
        IREE_VM_ISA_FLOAT_MATH_UNARY_ROUNDEVEN,
        IREE_VM_ISA_FLOAT_MATH_UNARY_TRUNC}) {
    exact(selector, zero, zero);
    exact(selector, negative_zero, negative_zero);
    exact(selector, infinity, infinity);
    exact(selector, -infinity, -infinity);
  }
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ROUND, Float{2.5}, Float{3});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ROUND, Float{-2.5}, Float{-3});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ROUNDEVEN, Float{2.5}, Float{2});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_ROUNDEVEN, Float{3.5}, Float{4});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN, quiet_nan, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN, negative_zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN, Float{-2}, Float{-1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN, infinity, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN, -infinity, Float{-1});

  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS, zero, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS, negative_zero, negative_zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS, zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS, negative_zero, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS, Float{0.25}, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS, Float{0.5}, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS, Float{-0.5}, negative_zero);
  const Float large_turn =
      std::ldexp(Float{1}, std::numeric_limits<Float>::digits);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS, large_turn, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS, large_turn, Float{1});
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS, Float{0.25}, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS, Float{-0.25}, zero);
  exact(IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS, Float{0.5}, Float{-1});
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS, infinity);
  arithmetic_nan(IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS, -infinity);

  for (uint8_t selector : {IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_ERF,
                           IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH}) {
    arithmetic_nan(selector, -infinity);
    exact(selector, negative_zero, negative_zero);
    exact(selector, zero, zero);
    exact(selector, infinity, infinity);
  }

  const Float signaling_nan = FromBits<Float>(MathTraits<Float>::kSignalingNaN);
  for (Float nan : {quiet_nan, signaling_nan}) {
    for (uint8_t selector = IREE_VM_ISA_FLOAT_MATH_UNARY_EXP;
         selector <= IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH; ++selector) {
      if (selector == IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN) {
        exact(selector, nan, zero);
      } else {
        arithmetic_nan(selector, nan);
      }
    }
  }

  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW, Float{2},
                                   Float{3}),
             Float{8});
  ExpectBits(
      EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW, quiet_nan, zero),
      Float{1});
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW, Float{1},
                                   quiet_nan),
             Float{1});
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW,
                                   negative_zero, Float{3}),
             negative_zero);
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW,
                                   negative_zero, Float{-3}),
             -infinity);
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW, -infinity,
                                   Float{3}),
             -infinity);
  ExpectArithmeticNaN(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW,
                                            Float{-2}, Float{0.5}));
  ExpectArithmeticNaN(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_POW,
                                            signaling_nan, Float{1}));
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2,
                                   negative_zero, Float{1}),
             negative_zero);
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2, zero,
                                   negative_zero),
             std::atan2(zero, negative_zero));
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2,
                                   negative_zero, negative_zero),
             std::atan2(negative_zero, negative_zero));
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2,
                                   infinity, -infinity),
             std::atan2(infinity, -infinity));
  ExpectArithmeticNaN(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2,
                                            signaling_nan, Float{1}));

  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC,
                                   negative_zero, Float{1.702}),
             negative_zero);
  ExpectBits(EvaluateBinary<Float>(IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC,
                                   infinity, Float{1.702}),
             infinity);
  ExpectArithmeticNaN(EvaluateBinary<Float>(
      IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC, -infinity, Float{1.702}));
  ExpectArithmeticNaN(EvaluateBinary<Float>(
      IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC, Float{1}, signaling_nan));

  ExpectArithmeticNaN(EvaluateTernary<Float>(IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA,
                                             infinity, zero, Float{1}));
  ExpectBits(EvaluateTernary<Float>(IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA,
                                    negative_zero, Float{1}, negative_zero),
             negative_zero);
  ExpectBits(EvaluateTernary<Float>(IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA,
                                    infinity, Float{1}, Float{1}),
             infinity);
  ExpectArithmeticNaN(EvaluateTernary<Float>(IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA,
                                             infinity, Float{1}, -infinity));
  ExpectArithmeticNaN(EvaluateTernary<Float>(
      IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA, signaling_nan, Float{1}, Float{1}));
}

template <typename Float>
void CheckFusedRounding() {
  constexpr int kSignificandBits = std::numeric_limits<Float>::digits;
  const Float epsilon = std::ldexp(Float{1}, 1 - kSignificandBits);
  const Float a = Float{1} + epsilon;
  const Float b = Float{1} - epsilon;
  const Float c = Float{-1};
  const Float fused =
      EvaluateTernary<Float>(IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA, a, b, c);
  ExpectBits(fused, std::fma(a, b, c));
  EXPECT_NE(ToBits(fused), ToBits(Float{0}));
}

class VMBytecodeInterpreterFloatMathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fpu_state_ = iree_fpu_state_push(IREE_FPU_STATE_FLAG_MASK_EXCEPTIONS |
                                     IREE_FPU_STATE_FLAG_ROUND_TO_NEAREST);
  }

  void TearDown() override { iree_fpu_state_pop(fpu_state_); }

  // Caller floating-point control state restored after each test.
  iree_fpu_state_t fpu_state_ = {};
};

TEST_F(VMBytecodeInterpreterFloatMathTest, EvaluatesEverySelector) {
  CheckEverySelector<float>();
  CheckEverySelector<double>();
}

TEST_F(VMBytecodeInterpreterFloatMathTest, PreservesExceptionalValues) {
  CheckExceptionalValues<float>();
  CheckExceptionalValues<double>();
}

TEST_F(VMBytecodeInterpreterFloatMathTest, FmaRoundsOnce) {
  CheckFusedRounding<float>();
  CheckFusedRounding<double>();
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
