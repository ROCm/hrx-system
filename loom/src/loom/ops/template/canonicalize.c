// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/callable_effects.h"
#include "loom/ops/template/ops.h"

iree_status_t loom_template_apply_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  return loom_callable_effects_propagate_purity(
      op, loom_template_apply_family(op), loom_template_apply_purity_ATTR_INDEX,
      rewriter);
}

iree_status_t loom_template_call_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  return loom_callable_effects_propagate_purity(
      op, loom_template_call_callee(op), loom_template_call_purity_ATTR_INDEX,
      rewriter);
}

loom_trait_flags_t loom_template_apply_effective_traits(const loom_op_t* op) {
  return loom_callable_effects_traits(op,
                                      loom_template_apply_purity_ATTR_INDEX);
}

loom_trait_flags_t loom_template_call_effective_traits(const loom_op_t* op) {
  return loom_callable_effects_traits(op, loom_template_call_purity_ATTR_INDEX);
}
