// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_ROOTS_H_
#define IREE_MATH_ROOTS_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns one frozen machine-like approximation of 1/value.
// Normal inputs use one f32 division. Zero and subnormal inputs return signed
// infinity, infinity returns signed zero, and subnormal results flush to signed
// zero. Every NaN returns the canonical positive quiet NaN. The caller must
// establish non-trapping round-to-nearest-even arithmetic.
IREE_API_EXPORT float iree_math_reciprocal_f32_approx(float value);

// Returns one frozen machine-like approximation of 1/sqrt(value).
// Positive normal inputs use one f32 square root followed by one f32 division,
// with a rounding point after each operation. Signed zero and subnormal inputs
// return signed infinity, positive infinity returns positive zero, and every
// other negative input and every NaN return the canonical positive quiet NaN.
// Finite results differ from the correctly rounded reciprocal square root by
// at most one adjacent f32 value. The caller must establish non-trapping
// round-to-nearest-even arithmetic.
IREE_API_EXPORT float iree_math_rsqrt_f32_approx(float value);

// Returns one frozen machine-like approximation of sqrt(value).
// Positive normal inputs use one correctly rounded f32 square root. Signed zero
// and subnormal inputs return signed zero, positive infinity is preserved, and
// every other negative input and every NaN return the canonical positive quiet
// NaN. The caller must establish non-trapping round-to-nearest-even
// arithmetic.
IREE_API_EXPORT float iree_math_sqrt_f32_approx(float value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_ROOTS_H_
