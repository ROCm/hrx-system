// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/patterns.h"

iree_status_t loom_math_legalize_rewrite_recipe(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_rewritten) {
  *out_rewritten = false;

  IREE_RETURN_IF_ERROR(loom_math_legalize_rewrite_elementwise_recipe(
      context, op, rewriter, out_rewritten));
  if (*out_rewritten) {
    return iree_ok_status();
  }

  return iree_ok_status();
}
