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

// Rewrites |apply_op| as an exact template.call to |callee|.
//
// |operands| must contain one replacement for every application operand. The
// rewrite preserves result types, tied results, purity, temperature, source
// location, and authored result names before replacing all result uses and
// erasing the application.
iree_status_t loom_template_rewrite_apply_as_exact_call(
    loom_rewriter_t* rewriter, loom_op_t* apply_op, loom_symbol_ref_t callee,
    const loom_value_id_t* operands);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TRANSFORMS_SYMBOL_TEMPLATE_REWRITE_H_
