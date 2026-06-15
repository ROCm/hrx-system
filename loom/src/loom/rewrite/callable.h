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
// ending in func.return. Function entry block arguments are bound to call
// operands, body ops are cloned before the call, func.return operands become
// the call result replacements, and the original call-like op is erased through
// the rewriter.
iree_status_t loom_callable_inline_call(loom_rewriter_t* rewriter,
                                        loom_op_t* call_op,
                                        loom_func_like_t callee);

// Resolves |call_op|'s direct callee and then inlines it.
iree_status_t loom_callable_inline_direct_call(loom_rewriter_t* rewriter,
                                               loom_op_t* call_op);

// Inlines |callee| at |call_op| using same-module move materialization.
//
// This consumes the callee definition. The callee must be private, the selected
// call must be the only live symbol reference to the callee outside of the
// callee's own defining attribute, and the callee body must satisfy the same
// single-block func.return shape as clone inlining. On success body ops are
// moved before the call, call results are replaced by remapped func.return
// operands, the call is erased, and the consumed callee op is erased.
iree_status_t loom_callable_inline_consuming_call(loom_rewriter_t* rewriter,
                                                  loom_op_t* call_op,
                                                  loom_func_like_t callee);

// Resolves |call_op|'s direct callee and then consuming-inlines it.
iree_status_t loom_callable_inline_consuming_direct_call(
    loom_rewriter_t* rewriter, loom_op_t* call_op);

// Options for importing one function-like definition across modules.
typedef struct loom_callable_import_options_t {
  // Optional policy for non-callee symbol references in the imported body.
  // The imported callee's own defining symbol is handled by the import helper.
  // If NULL, any other symbol reference in the source definition is rejected.
  loom_ir_remap_symbol_callback_t external_symbol_remap;
} loom_callable_import_options_t;

// Result handles produced by callable outlining.
typedef struct loom_callable_outline_result_t {
  // Function-like definition that owns the outlined body.
  loom_func_like_t outlined;
  // Call op inserted at the original range position.
  loom_op_t* call_op;
} loom_callable_outline_result_t;

// Clones |source| from |source_module| into |builder|'s target module.
//
// The source callee symbol is recreated in the target module with the same name
// and bound to the cloned function-like op. Target name collisions are rejected
// instead of renamed implicitly. References to other source symbols require an
// explicit external symbol remap policy in |options|; without one, the import
// fails before mutating the target module.
iree_status_t loom_callable_import_definition(
    loom_builder_t* builder, const loom_module_t* source_module,
    loom_func_like_t source, const loom_callable_import_options_t* options,
    loom_func_like_t* out_imported, iree_arena_allocator_t* scratch_arena);

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
iree_status_t loom_callable_outline_range(
    loom_rewriter_t* rewriter, loom_op_t* first_op, loom_op_t* after_last_op,
    loom_symbol_ref_t outlined_ref, loom_callable_outline_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_CALLABLE_H_
