// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Programmatic construction helpers for pass IR.
//
// These helpers are the C-side equivalent of writing a pass.pipeline in Loom
// text. They centralize the small amount of symbol/string plumbing needed by
// tools and production pipeline contributors without inventing a second pass
// representation. Execution still goes through pass.program compilation and the
// interpreter.

#ifndef LOOM_PASS_BUILDER_H_
#define LOOM_PASS_BUILDER_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/pass/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*loom_pass_ir_body_build_fn_t)(loom_builder_t* builder,
                                                      void* user_data);

// Open nested pass control region. Callers that need to stream multiple ops
// into a region over time should hold this scope and close it with
// loom_pass_ir_end_scope. Prefer the callback-based helpers below when the
// nested body can be built in one call.
typedef struct loom_pass_ir_scope_t {
  // Pass control op owning the open region.
  loom_op_t* owner_op;
  // Insertion point outside the open region.
  loom_builder_ip_t saved_insertion_point;
} loom_pass_ir_scope_t;

// Builds a named pass.pipeline in |module| and invokes |build_body| while the
// builder insertion point is inside the pipeline body. The helper appends the
// required pass.yield terminator after the body callback succeeds.
iree_status_t loom_pass_ir_build_pipeline(
    loom_module_t* module, iree_string_view_t name, loom_pass_anchor_t anchor,
    loom_pass_ir_body_build_fn_t build_body, void* user_data,
    loom_op_t** out_pipeline_op);

// Appends a pass.run with the canonical pass |key| and typed option attrs.
// |options| must already be owned by |builder->module|.
iree_status_t loom_pass_ir_build_run(loom_builder_t* builder,
                                     iree_string_view_t key,
                                     loom_named_attr_slice_t options,
                                     loom_op_t** out_run_op);

// Appends a pass.for and enters its body region. The caller must eventually
// call loom_pass_ir_end_scope.
iree_status_t loom_pass_ir_begin_for(loom_builder_t* builder,
                                     loom_pass_anchor_t anchor,
                                     loom_pass_ir_scope_t* out_scope);

// Appends a pass.for and invokes |build_body| inside the body region.
iree_status_t loom_pass_ir_build_for(loom_builder_t* builder,
                                     loom_pass_anchor_t anchor,
                                     loom_pass_ir_body_build_fn_t build_body,
                                     void* user_data, loom_op_t** out_for_op);

// Appends a pass.where and enters its body region. The caller must eventually
// call loom_pass_ir_end_scope.
iree_status_t loom_pass_ir_begin_where(loom_builder_t* builder,
                                       iree_string_view_t predicate,
                                       loom_named_attr_slice_t attrs,
                                       loom_pass_ir_scope_t* out_scope);

// Appends a pass.where and invokes |build_body| inside the body region.
iree_status_t loom_pass_ir_build_where(loom_builder_t* builder,
                                       iree_string_view_t predicate,
                                       loom_named_attr_slice_t attrs,
                                       loom_pass_ir_body_build_fn_t build_body,
                                       void* user_data,
                                       loom_op_t** out_where_op);

// Appends a pass.repeat and enters its body region. The caller must eventually
// call loom_pass_ir_end_scope.
iree_status_t loom_pass_ir_begin_repeat(
    loom_builder_t* builder, loom_pass_repeat_build_flags_t build_flags,
    loom_pass_repeat_mode_t mode, int64_t count, int64_t max_iterations,
    loom_pass_ir_scope_t* out_scope);

// Appends a pass.repeat and invokes |build_body| inside the body region.
iree_status_t loom_pass_ir_build_repeat(
    loom_builder_t* builder, loom_pass_repeat_build_flags_t build_flags,
    loom_pass_repeat_mode_t mode, int64_t count, int64_t max_iterations,
    loom_pass_ir_body_build_fn_t build_body, void* user_data,
    loom_op_t** out_repeat_op);

// Appends a pass.call to |callee|.
iree_status_t loom_pass_ir_build_call(loom_builder_t* builder,
                                      loom_symbol_ref_t callee,
                                      loom_op_t** out_call_op);

// Appends a pass.yield terminator at the current insertion point.
iree_status_t loom_pass_ir_build_yield(loom_builder_t* builder);

// Appends pass.yield to the current pass control region and restores the
// insertion point captured by |scope|.
iree_status_t loom_pass_ir_end_scope(loom_builder_t* builder,
                                     const loom_pass_ir_scope_t* scope);

// Restores the insertion point captured by |scope| without inserting a
// terminator. This is only for aborting construction after a prior error.
void loom_pass_ir_abandon_scope(loom_builder_t* builder,
                                const loom_pass_ir_scope_t* scope);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_BUILDER_H_
