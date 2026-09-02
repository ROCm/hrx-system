// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/callable_effects.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

bool loom_callable_effects_is_pure(loom_func_like_t function) {
  if (!loom_func_like_isa(function)) {
    return false;
  }
  loom_region_t* body = loom_func_like_body(function);
  if (body) {
    return !loom_region_has_read_effects(body) &&
           !loom_region_has_write_effects(body) &&
           !loom_region_has_convergent_effects(body) &&
           !loom_region_has_observable_effects(body);
  }
  return loom_func_like_purity(function) != 0;
}

iree_status_t loom_callable_effects_propagate_purity(
    loom_op_t* op, loom_symbol_ref_t callee, uint8_t purity_attr_index,
    loom_rewriter_t* rewriter) {
  if (loom_attr_as_enum(loom_op_attrs(op)[purity_attr_index]) != 0 ||
      !loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= rewriter->module->symbols.count) {
    return iree_ok_status();
  }

  const loom_symbol_t* symbol =
      &rewriter->module->symbols.entries[callee.symbol_id];
  if (!symbol->defining_op) {
    return iree_ok_status();
  }
  loom_func_like_t function =
      loom_func_like_cast(rewriter->module, symbol->defining_op);
  if (!loom_callable_effects_is_pure(function)) {
    return iree_ok_status();
  }
  return loom_rewriter_set_attr(rewriter, op, purity_attr_index,
                                loom_attr_enum(1));
}

loom_trait_flags_t loom_callable_effects_traits(const loom_op_t* op,
                                                uint8_t purity_attr_index) {
  if (loom_attr_as_enum(loom_op_const_attrs(op)[purity_attr_index]) != 0) {
    return LOOM_TRAIT_PURE;
  }
  return LOOM_TRAIT_UNKNOWN_EFFECTS;
}
