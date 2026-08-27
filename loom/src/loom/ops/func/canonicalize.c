// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/callable_effects.h"
#include "loom/ops/func/ops.h"

iree_status_t loom_func_call_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  return loom_callable_effects_propagate_purity(
      op, loom_func_call_callee(op), loom_func_call_purity_ATTR_INDEX,
      rewriter);
}

loom_trait_flags_t loom_func_call_effective_traits(const loom_op_t* op) {
  return loom_callable_effects_traits(op, loom_func_call_purity_ATTR_INDEX);
}
