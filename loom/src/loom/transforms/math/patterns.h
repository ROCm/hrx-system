// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Legalize-math recipe handlers.

#ifndef LOOM_TRANSFORMS_MATH_PATTERNS_H_
#define LOOM_TRANSFORMS_MATH_PATTERNS_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/math_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_math_legalize_recipe_context_t {
  // Pass invocation using the recipe.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // Target-policy query that selected this recipe.
  loom_target_math_query_t query;
  // Target-policy decision that selected this recipe.
  loom_target_math_policy_decision_t decision;
} loom_math_legalize_recipe_context_t;

// Attempts to rewrite |op| using the recipe selected in |context|. Returns
// |out_rewritten| false when no linked recipe shard owns the recipe.
iree_status_t loom_math_legalize_rewrite_recipe(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_rewritten);

// Attempts elementwise scalar/vector recipes such as exp, erf, logistic, SiLU,
// GELU, and softplus. This shard is intentionally separate from the pass
// wrapper so the recipe set can grow by math family without creating a
// monolithic pass file.
iree_status_t loom_math_legalize_rewrite_elementwise_recipe(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_rewritten);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_MATH_PATTERNS_H_
