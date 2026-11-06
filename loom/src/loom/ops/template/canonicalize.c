// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/callable_effects.h"
#include "loom/ops/template/ops.h"
#include "loom/rewrite/rewriter.h"

iree_status_t loom_template_apply_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  if (loom_template_apply_effect_contract(op) != 0) {
    return iree_ok_status();
  }
  const loom_callable_effect_contract_t contract =
      loom_callable_effects_callee_contract(rewriter->module,
                                            loom_template_apply_family(op));
  return contract == 0 ? iree_ok_status()
                       : loom_template_apply_rewrite_effect_contract(
                             rewriter, op, loom_attr_enum(contract));
}

iree_status_t loom_template_call_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  if (loom_template_call_effect_contract(op) != 0) {
    return iree_ok_status();
  }
  const loom_callable_effect_contract_t contract =
      loom_callable_effects_callee_contract(rewriter->module,
                                            loom_template_call_callee(op));
  return contract == 0 ? iree_ok_status()
                       : loom_template_call_rewrite_effect_contract(
                             rewriter, op, loom_attr_enum(contract));
}

loom_trait_flags_t loom_template_apply_effective_traits(
    const loom_module_t* module, const loom_op_t* op) {
  return LOOM_TRAIT_CONTEXTUAL |
         loom_callable_effects_traits(module, op,
                                      loom_template_apply_effect_contract(op));
}

loom_trait_flags_t loom_template_call_effective_traits(
    const loom_module_t* module, const loom_op_t* op) {
  return LOOM_TRAIT_CONTEXTUAL |
         loom_callable_effects_traits(module, op,
                                      loom_template_call_effect_contract(op));
}
