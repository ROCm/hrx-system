// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural rewrites for template applications.

#ifndef LOOM_TRANSFORMS_SYMBOL_TEMPLATE_REWRITE_H_
#define LOOM_TRANSFORMS_SYMBOL_TEMPLATE_REWRITE_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Carries newly proven scalar identities into shaped operand types using
// explicit carrier refinements. Non-shaped operands and unaffected types are
// unchanged. The caller separately replaces direct scalar operands.
iree_status_t loom_template_rewrite_assumed_inputs(
    loom_rewriter_t* rewriter, const loom_op_t* application,
    const loom_value_id_t* source_values, const loom_value_id_t* assumed_values,
    uint16_t value_count, loom_value_id_t* operands);

// Rewrites |apply_op| as an exact template.call to |callee|.
//
// |operands| must contain one replacement for every application operand. The
// rewrite preserves result types, tied results, effect contract, temperature,
// source location, and authored result names before replacing all result uses
// and erasing the application. On success, writes the replacement to
// |out_call_op| when non-NULL, allowing its selection owner to retain the
// expansion plan.
iree_status_t loom_template_rewrite_apply_as_exact_call(
    loom_rewriter_t* rewriter, loom_op_t* apply_op, loom_symbol_ref_t callee,
    const loom_value_id_t* operands, loom_op_t** out_call_op);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_SYMBOL_TEMPLATE_REWRITE_H_
