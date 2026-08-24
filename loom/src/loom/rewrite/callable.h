// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Callable materialization adapters.
//
// These helpers adapt generic IR remapping/materialization to function-like
// operations. They do not decide inline profitability or import policy; callers
// choose a call site and these helpers perform the checked mutation.

#ifndef LOOM_REWRITE_CALLABLE_H_
#define LOOM_REWRITE_CALLABLE_H_

#include "iree/base/api.h"
#include "loom/analysis/availability.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves the direct symbol target of a direct call-like op.
iree_status_t loom_callable_resolve_direct_callee(const loom_module_t* module,
                                                  const loom_op_t* call_op,
                                                  loom_func_like_t* out_callee);

// Inlines |callee| at |call_op| using clone materialization.
//
// The callee must be a same-module function-like op with a single-block body
// ending in the terminator declared by its body region. Function entry block
// arguments are bound to call operands, body ops are cloned before the call,
// terminator operands become the call result replacements, and the original
// call-like op is erased through the rewriter.
iree_status_t loom_callable_inline_call(loom_rewriter_t* rewriter,
                                        loom_op_t* call_op,
                                        loom_func_like_t callee);

// Resolves |call_op|'s direct callee and then inlines it.
iree_status_t loom_callable_inline_direct_call(loom_rewriter_t* rewriter,
                                               loom_op_t* call_op);

// Inlines |callee| at |call_op| using same-module move materialization.
//
// This consumes the callee definition. The caller must have an exact immutable
// reference plan proving that the callee is private and that the selected call
// will be the final live reference outside of the callee's own defining
// attribute. This helper does not rediscover that global ownership by scanning
// the module. |availability| must cover the caller region and remain valid
// across the topology-preserving body move. The callee body must satisfy the
// same single-block terminator shape as clone inlining. On success body ops are
// moved before the call, call results are replaced by remapped terminator
// operands, the call is erased, and the consumed callee op is erased.
iree_status_t loom_callable_inline_consuming_call(
    loom_rewriter_t* rewriter, const loom_availability_analysis_t* availability,
    loom_op_t* call_op, loom_func_like_t callee);

// Clones one same-module function-like definition as |target_ref|.
//
// |target_ref| must name an existing symbol without a defining op. The cloned
// function's defining symbol and recursive self-references are rewritten to
// |target_ref| while references to every other same-module symbol are
// preserved.
iree_status_t loom_callable_clone_definition(
    loom_builder_t* builder, loom_func_like_t source,
    loom_symbol_ref_t target_ref, loom_func_like_t* out_cloned,
    iree_arena_allocator_t* scratch_arena);

// Result handles produced by callable outlining.
typedef struct loom_callable_outline_result_t {
  // Function-like definition that owns the outlined body.
  loom_func_like_t outlined;
  // Call op inserted at the original range position.
  loom_op_t* call_op;
} loom_callable_outline_result_t;

// Outlines a contiguous same-block op range into a new private func.def.
//
// |first_op| is included. |after_last_op| is excluded and may be NULL to
// outline through the end of the block. All selected root ops must be live,
// linked, non-terminator ops in the same block. |outlined_ref| must name an
// existing target-module symbol with no defining op; the helper binds it to the
// created func.def instead of inventing or renaming symbols.
//
// Captures and live-outs are derived structurally from SSA operands, dynamic
// type references, value-bearing attributes, predicate lists, nested regions,
// and type-use lists. The replacement func.call returns every selected value
// needed outside the range, including values needed only by dynamic result
// types, so erasing the original range leaves no dangling SSA or type refs.
// Ranges inside representation-bound functions are rejected because the
// generic func.def created by this helper cannot preserve their contract.
iree_status_t loom_callable_outline_range(
    loom_rewriter_t* rewriter, loom_op_t* first_op, loom_op_t* after_last_op,
    loom_symbol_ref_t outlined_ref, loom_callable_outline_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_CALLABLE_H_
