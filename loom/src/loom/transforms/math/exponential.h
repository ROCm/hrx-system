// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_MATH_EXPONENTIAL_H_
#define LOOM_TRANSFORMS_MATH_EXPONENTIAL_H_

#include "loom/transforms/math/evaluation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Evaluates the AFN exponential family in F32 or F64. Split range reduction
// and normal scale factors retain gradual underflow and finite overflow-edge
// results; expm1 reconstructs the difference without cancellation near zero.
iree_status_t loom_math_build_exponential(
    loom_target_math_op_t math_op, loom_math_evaluation_t* evaluation,
    loom_math_evaluation_value_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_MATH_EXPONENTIAL_H_
