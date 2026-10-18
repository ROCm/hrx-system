// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_MATH_TRIGONOMETRY_H_
#define LOOM_TRANSFORMS_MATH_TRIGONOMETRY_H_

#include "loom/transforms/math/evaluation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Evaluates sine or cosine using the target's native radian operation in f32
// lanes. Turn inputs are reduced before scaling and retain exact cardinals.
// The evaluation owns source import/export, including narrow source formats.
iree_status_t loom_math_build_trigonometric(
    loom_target_math_op_t math_op, loom_math_evaluation_t* evaluation,
    loom_math_evaluation_value_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_MATH_TRIGONOMETRY_H_
