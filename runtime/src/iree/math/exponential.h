// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_EXPONENTIAL_H_
#define IREE_MATH_EXPONENTIAL_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns one frozen machine-like approximation of 2^value.
// Finite normal-path results differ from the correctly rounded result by at
// most one adjacent f32 value. Values below -126 flush to positive zero and
// values at least 128 overflow to positive infinity. Either zero returns one,
// infinities map to their limiting results, and every NaN returns the canonical
// positive quiet NaN. The caller must establish non-trapping
// round-to-nearest-even arithmetic.
IREE_API_EXPORT float iree_math_exp2_f32_approx(float value);

// Returns one frozen machine-like approximation of log2(value).
// Finite normal-path results differ from the correctly rounded result by at
// most one adjacent f32 value. Zeros and subnormals flush to zero and return
// negative infinity. Negative normal values and every NaN return the canonical
// positive quiet NaN, and positive infinity is preserved. The caller must
// establish non-trapping round-to-nearest-even arithmetic.
IREE_API_EXPORT float iree_math_log2_f32_approx(float value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_EXPONENTIAL_H_
