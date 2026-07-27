// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonicalization and effective trait implementations for the func
// dialect. These are hand-written (not generated) and linked into the
// func dialect library so the vtable function pointers resolve.
//
// Purity propagation: the func.call canonicalizer reads the callee's explicit
// purity or body effect summary from the symbol table and sets the call-site
// purity attr to match. The effective_traits callback then reports
// LOOM_TRAIT_PURE instead of LOOM_TRAIT_UNKNOWN_EFFECTS, enabling CSE and DCE
// across call boundaries.

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"

static bool loom_func_like_is_pure(const loom_func_like_t func) {
  if (!loom_func_like_isa(func)) return false;
  loom_region_t* body = loom_func_like_body(func);
  if (body) {
    return !loom_region_has_read_effects(body) &&
           !loom_region_has_write_effects(body) &&
           !loom_region_has_convergent_effects(body);
  }
  return loom_func_like_purity(func) != 0;
}

// Propagates callee purity to the call site. If the callee is a pure
// function and the call doesn't already have the purity attr set,
// sets it via the rewriter.
static iree_status_t loom_func_propagate_callee_purity(
    loom_op_t* op, loom_rewriter_t* rewriter, uint16_t purity_attr_index) {
  // Already marked pure — nothing to propagate.
  if (loom_attr_as_enum(loom_op_attrs(op)[purity_attr_index]) != 0) {
    return iree_ok_status();
  }

  // Look up the callee symbol.
  loom_symbol_ref_t callee =
      loom_attr_as_symbol(loom_op_attrs(op)[0]);  // callee is always attr 0.
  if (!loom_symbol_ref_is_valid(callee)) return iree_ok_status();
  if (callee.symbol_id >= rewriter->module->symbols.count) {
    return iree_ok_status();
  }

  const loom_symbol_t* symbol =
      &rewriter->module->symbols.entries[callee.symbol_id];
  loom_func_like_t callee_func =
      loom_func_like_cast(rewriter->module, symbol->defining_op);

  if (loom_func_like_is_pure(callee_func)) {
    return loom_rewriter_set_attr(rewriter, op, purity_attr_index,
                                  loom_attr_enum(LOOM_FUNC_PURITY_PURE));
  }

  return iree_ok_status();
}

iree_status_t loom_func_call_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  return loom_func_propagate_callee_purity(op, rewriter,
                                           loom_func_call_purity_ATTR_INDEX);
}

iree_status_t loom_func_apply_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  (void)op;
  (void)rewriter;
  return iree_ok_status();
}

loom_trait_flags_t loom_func_call_effective_traits(const loom_op_t* op) {
  if (loom_func_call_purity(op) != 0) return LOOM_TRAIT_PURE;
  return LOOM_TRAIT_UNKNOWN_EFFECTS;
}

loom_trait_flags_t loom_func_apply_effective_traits(const loom_op_t* op) {
  if (loom_func_apply_purity(op) != 0) return LOOM_TRAIT_PURE;
  return LOOM_TRAIT_UNKNOWN_EFFECTS;
}
