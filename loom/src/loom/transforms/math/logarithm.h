// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_MATH_LOGARITHM_H_
#define LOOM_TRANSFORMS_MATH_LOGARITHM_H_

#include "loom/transforms/math/evaluation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Evaluates the AFN logarithm family in F32 or F64, preserving exceptional
// values and cancellation near one (near zero for log1p).
iree_status_t loom_math_build_logarithm(
    loom_target_math_op_t math_op, loom_math_evaluation_t* evaluation,
    loom_math_evaluation_value_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_MATH_LOGARITHM_H_
