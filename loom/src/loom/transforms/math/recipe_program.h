// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact evaluation of generated target-neutral math recipes.

#ifndef LOOM_TRANSFORMS_MATH_RECIPE_PROGRAM_H_
#define LOOM_TRANSFORMS_MATH_RECIPE_PROGRAM_H_

#include "iree/base/api.h"
#include "loom/transforms/math/evaluation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_math_recipe_program_id_e {
  LOOM_MATH_RECIPE_PROGRAM_EXP2_F32 = 0,
  LOOM_MATH_RECIPE_PROGRAM_SIN_TURNS_F32 = 1,
  LOOM_MATH_RECIPE_PROGRAM_COS_TURNS_F32 = 2,
  LOOM_MATH_RECIPE_PROGRAM_COUNT_ = 3,
} loom_math_recipe_program_id_t;

// Evaluates a generated straight-line program in the scalar or vector source
// domain selected by |evaluation|.
iree_status_t loom_math_recipe_program_evaluate(
    loom_math_evaluation_t* evaluation,
    loom_math_recipe_program_id_t program_id,
    const loom_math_evaluation_value_t* input,
    loom_math_evaluation_value_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_MATH_RECIPE_PROGRAM_H_
