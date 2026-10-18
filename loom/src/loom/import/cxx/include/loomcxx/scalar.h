// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.cxx.intrinsics.
// Regenerate: python3 loom/py/loom/gen/run.py cxx_intrinsics --in-place
// clang-format off
#ifndef LOOMCXX_SCALAR_H_
#define LOOMCXX_SCALAR_H_

// Fixed scalar projections of the canonical Loom operation contracts.
// The approximate namespace explicitly permits AFN and no other flags.
// Its preserved trigonometry retains source-defined zero signs even
// when the calling scope permits NSZ. Nonzero results remain approximate.
namespace loom::scalar {

// Floating-point absolute value.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.absf")]] Float absf(Float input);

// Arccosine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.acosf")]] Float acosf(Float input);

// Inverse hyperbolic cosine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.acoshf")]] Float acoshf(Float input);

// Floating-point addition.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.addf")]] Float addf(Float lhs, Float rhs);

// Arcsine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.asinf")]] Float asinf(Float input);

// Inverse hyperbolic sine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.asinhf")]] Float asinhf(Float input);

// Two-argument arctangent: atan2(y, x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.atan2f")]] Float atan2f(Float lhs, Float rhs);

// Arctangent.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.atanf")]] Float atanf(Float input);

// Inverse hyperbolic tangent.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.atanhf")]] Float atanhf(Float input);

// Cube root.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.cbrtf")]] Float cbrtf(Float input);

// Round toward positive infinity.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.ceilf")]] Float ceilf(Float input);

// Copy sign of rhs onto magnitude of lhs.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.copysignf")]] Float copysignf(Float lhs, Float rhs);

// Cosine in radians. AFN permits native approximation with target-dependent
// zero signs; preserve requires source-defined zero signs.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.cosf")]] Float cosf(Float input);

// Hyperbolic cosine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.coshf")]] Float coshf(Float input);

// Cosine over turns: cos(2*pi*x), preserving finite-input periodicity and
// exact quarter-turn cardinal values. Non-finite inputs produce NaN. AFN
// permits target-dependent zero signs; preserve requires positive zero at odd
// quarter-turns.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.costurnsf")]] Float costurnsf(Float input);

// Count leading zeros.
template <class Integer> requires (__is_integral(Integer) && !__is_same(Integer, bool))
[[loom::op("scalar.ctlzi")]] Integer ctlzi(Integer input);

// Population count (number of set bits).
template <class Integer> requires (__is_integral(Integer) && !__is_same(Integer, bool))
[[loom::op("scalar.ctpopi")]] Integer ctpopi(Integer input);

// Count trailing zeros.
template <class Integer> requires (__is_integral(Integer) && !__is_same(Integer, bool))
[[loom::op("scalar.cttzi")]] Integer cttzi(Integer input);

// Floating-point division.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.divf")]] Float divf(Float lhs, Float rhs);

// Complementary error function: 1 - erf(x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.erfcf")]] Float erfcf(Float input);

// Error function (used in GeLU activation).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.erff")]] Float erff(Float input);

// Base-2 exponential: 2^x.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.exp2f")]] Float exp2f(Float input);

// Exponential: e^x.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.expf")]] Float expf(Float input);

// Exponential minus one: e^x - 1 (numerically stable near 0).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.expm1f")]] Float expm1f(Float input);

// Round toward negative infinity.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.floorf")]] Float floorf(Float input);

// Fused multiply-add: a*b + c with single rounding. AFN permits a target-
// native fused form with weaker subnormal handling.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.fmaf")]] Float fmaf(Float a, Float b, Float c);

// Base-10 logarithm.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.log10f")]] Float log10f(Float input);

// Natural logarithm of 1+x: ln(1+x) (numerically stable near 0).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.log1pf")]] Float log1pf(Float input);

// Base-2 logarithm.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.log2f")]] Float log2f(Float input);

// Natural logarithm: ln(x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.logf")]] Float logf(Float input);

// Logistic sigmoid: 1 / (1 + exp(-x)).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.logisticf")]] Float logisticf(Float input);

// IEEE 754 maximum (NaN propagates).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.maximumf")]] Float maximumf(Float lhs, Float rhs);

// C99 fmax (NaN ignored, returns the non-NaN operand).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.maxnumf")]] Float maxnumf(Float lhs, Float rhs);

// IEEE 754 minimum (NaN propagates).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.minimumf")]] Float minimumf(Float lhs, Float rhs);

// C99 fmin (NaN ignored, returns the non-NaN operand).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.minnumf")]] Float minnumf(Float lhs, Float rhs);

// Floating-point multiplication.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.mulf")]] Float mulf(Float lhs, Float rhs);

// Floating-point negation.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.negf")]] Float negf(Float input);

// Power: x^y.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.powf")]] Float powf(Float lhs, Float rhs);

// Floating-point remainder (C fmod semantics).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.remf")]] Float remf(Float lhs, Float rhs);

// Round to nearest, ties to even (IEEE 754 default rounding).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.roundevenf")]] Float roundevenf(Float input);

// Round to nearest, ties away from zero.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.roundf")]] Float roundf(Float input);

// Reciprocal square root: 1/sqrt(x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.rsqrtf")]] Float rsqrtf(Float input);

// Floating-point sign: returns -1.0, 0.0, or 1.0.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.signf")]] Float signf(Float input);

// SiLU activation: x * logistic(x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.siluf")]] Float siluf(Float input);

// Sine in radians. AFN permits native approximation with target-dependent zero
// signs; preserve requires source-defined zero signs.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sinf")]] Float sinf(Float input);

// Hyperbolic sine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sinhf")]] Float sinhf(Float input);

// Sine over turns: sin(2*pi*x), preserving finite-input periodicity and exact
// quarter-turn cardinal values. Non-finite inputs produce NaN. AFN permits
// target-dependent zero signs; preserve requires the input sign at zero and
// integer or half-integer turns.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sinturnsf")]] Float sinturnsf(Float input);

// Softplus activation: log(1 + exp(x)).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.softplusf")]] Float softplusf(Float input);

// Square root.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sqrtf")]] Float sqrtf(Float input);

// Floating-point subtraction.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.subf")]] Float subf(Float lhs, Float rhs);

// Tangent.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.tanf")]] Float tanf(Float input);

// Hyperbolic tangent.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.tanhf")]] Float tanhf(Float input);

// Round toward zero (C trunc).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.truncf")]] Float truncf(Float input);

namespace approximate {

// Floating-point absolute value.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.absf", "afn")]] Float absf(Float input);

// Arccosine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.acosf", "afn")]] Float acosf(Float input);

// Inverse hyperbolic cosine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.acoshf", "afn")]] Float acoshf(Float input);

// Floating-point addition.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.addf", "afn")]] Float addf(Float lhs, Float rhs);

// Arcsine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.asinf", "afn")]] Float asinf(Float input);

// Inverse hyperbolic sine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.asinhf", "afn")]] Float asinhf(Float input);

// Two-argument arctangent: atan2(y, x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.atan2f", "afn")]] Float atan2f(Float lhs, Float rhs);

// Arctangent.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.atanf", "afn")]] Float atanf(Float input);

// Inverse hyperbolic tangent.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.atanhf", "afn")]] Float atanhf(Float input);

// Cube root.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.cbrtf", "afn")]] Float cbrtf(Float input);

// Round toward positive infinity.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.ceilf", "afn")]] Float ceilf(Float input);

// Copy sign of rhs onto magnitude of lhs.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.copysignf", "afn")]] Float copysignf(Float lhs, Float rhs);

// Cosine in radians. AFN permits native approximation with target-dependent
// zero signs; preserve requires source-defined zero signs.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.cosf", "afn")]] Float cosf(Float input);

// Hyperbolic cosine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.coshf", "afn")]] Float coshf(Float input);

// Cosine over turns: cos(2*pi*x), preserving finite-input periodicity and
// exact quarter-turn cardinal values. Non-finite inputs produce NaN. AFN
// permits target-dependent zero signs; preserve requires positive zero at odd
// quarter-turns.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.costurnsf", "afn")]] Float costurnsf(Float input);

// Floating-point division.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.divf", "afn")]] Float divf(Float lhs, Float rhs);

// Complementary error function: 1 - erf(x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.erfcf", "afn")]] Float erfcf(Float input);

// Error function (used in GeLU activation).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.erff", "afn")]] Float erff(Float input);

// Base-2 exponential: 2^x.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.exp2f", "afn")]] Float exp2f(Float input);

// Exponential: e^x.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.expf", "afn")]] Float expf(Float input);

// Exponential minus one: e^x - 1 (numerically stable near 0).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.expm1f", "afn")]] Float expm1f(Float input);

// Round toward negative infinity.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.floorf", "afn")]] Float floorf(Float input);

// Fused multiply-add: a*b + c with single rounding. AFN permits a target-
// native fused form with weaker subnormal handling.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.fmaf", "afn")]] Float fmaf(Float a, Float b, Float c);

// Base-10 logarithm.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.log10f", "afn")]] Float log10f(Float input);

// Natural logarithm of 1+x: ln(1+x) (numerically stable near 0).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.log1pf", "afn")]] Float log1pf(Float input);

// Base-2 logarithm.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.log2f", "afn")]] Float log2f(Float input);

// Natural logarithm: ln(x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.logf", "afn")]] Float logf(Float input);

// Logistic sigmoid: 1 / (1 + exp(-x)).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.logisticf", "afn")]] Float logisticf(Float input);

// IEEE 754 maximum (NaN propagates).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.maximumf", "afn")]] Float maximumf(Float lhs, Float rhs);

// C99 fmax (NaN ignored, returns the non-NaN operand).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.maxnumf", "afn")]] Float maxnumf(Float lhs, Float rhs);

// IEEE 754 minimum (NaN propagates).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.minimumf", "afn")]] Float minimumf(Float lhs, Float rhs);

// C99 fmin (NaN ignored, returns the non-NaN operand).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.minnumf", "afn")]] Float minnumf(Float lhs, Float rhs);

// Floating-point multiplication.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.mulf", "afn")]] Float mulf(Float lhs, Float rhs);

// Floating-point negation.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.negf", "afn")]] Float negf(Float input);

// Power: x^y.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.powf", "afn")]] Float powf(Float lhs, Float rhs);

// Floating-point remainder (C fmod semantics).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.remf", "afn")]] Float remf(Float lhs, Float rhs);

// Round to nearest, ties to even (IEEE 754 default rounding).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.roundevenf", "afn")]] Float roundevenf(Float input);

// Round to nearest, ties away from zero.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.roundf", "afn")]] Float roundf(Float input);

// Reciprocal square root: 1/sqrt(x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.rsqrtf", "afn")]] Float rsqrtf(Float input);

// SiLU activation: x * logistic(x).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.siluf", "afn")]] Float siluf(Float input);

// Sine in radians. AFN permits native approximation with target-dependent zero
// signs; preserve requires source-defined zero signs.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sinf", "afn")]] Float sinf(Float input);

// Hyperbolic sine.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sinhf", "afn")]] Float sinhf(Float input);

// Sine over turns: sin(2*pi*x), preserving finite-input periodicity and exact
// quarter-turn cardinal values. Non-finite inputs produce NaN. AFN permits
// target-dependent zero signs; preserve requires the input sign at zero and
// integer or half-integer turns.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sinturnsf", "afn")]] Float sinturnsf(Float input);

// Softplus activation: log(1 + exp(x)).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.softplusf", "afn")]] Float softplusf(Float input);

// Square root.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sqrtf", "afn")]] Float sqrtf(Float input);

// Floating-point subtraction.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.subf", "afn")]] Float subf(Float lhs, Float rhs);

// Tangent.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.tanf", "afn")]] Float tanf(Float input);

// Hyperbolic tangent.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.tanhf", "afn")]] Float tanhf(Float input);

// Round toward zero (C trunc).
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.truncf", "afn")]] Float truncf(Float input);

}  // namespace approximate

namespace approximate::preserved {

// Cosine in radians. AFN permits native approximation with target-dependent
// zero signs; preserve requires source-defined zero signs.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.cosf", "afn", "preserve")]] Float cosf(Float input);

// Cosine over turns: cos(2*pi*x), preserving finite-input periodicity and
// exact quarter-turn cardinal values. Non-finite inputs produce NaN. AFN
// permits target-dependent zero signs; preserve requires positive zero at odd
// quarter-turns.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.costurnsf", "afn", "preserve")]] Float costurnsf(Float input);

// Sine in radians. AFN permits native approximation with target-dependent zero
// signs; preserve requires source-defined zero signs.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sinf", "afn", "preserve")]] Float sinf(Float input);

// Sine over turns: sin(2*pi*x), preserving finite-input periodicity and exact
// quarter-turn cardinal values. Non-finite inputs produce NaN. AFN permits
// target-dependent zero signs; preserve requires the input sign at zero and
// integer or half-integer turns.
template <class Float> requires (__is_floating_point(Float))
[[loom::op("scalar.sinturnsf", "afn", "preserve")]] Float sinturnsf(Float input);

}  // namespace approximate::preserved

}  // namespace loom::scalar

#endif  // LOOMCXX_SCALAR_H_
