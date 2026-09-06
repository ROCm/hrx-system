// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_TRIGONOMETRY_H_
#define IREE_MATH_TRIGONOMETRY_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the correctly rounded sin(2*pi*turns) result for every f32 payload.
// Cardinal values are structural: signed zero is preserved, positive and
// negative half-integers return their matching signed zero, and odd
// quarter-turns return exact signed one. Every nonfinite input returns the
// canonical positive quiet NaN. The caller must establish non-trapping
// round-to-nearest-even arithmetic with input and output subnormals preserved.
IREE_API_EXPORT float iree_math_sin_turns_f32_approx(float turns);

// Returns the correctly rounded cos(2*pi*turns) result for every f32 payload.
// Cardinal values are structural: integers return one, half-integers return
// exact signed one, and odd quarter-turns return positive zero. Every
// nonfinite input returns the canonical positive quiet NaN. The caller must
// establish non-trapping round-to-nearest-even arithmetic with input and
// output subnormals preserved.
IREE_API_EXPORT float iree_math_cos_turns_f32_approx(float turns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_TRIGONOMETRY_H_
