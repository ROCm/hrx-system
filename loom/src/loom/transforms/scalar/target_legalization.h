// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared scalar target rewrites and generic reference legalizers.

#ifndef LOOM_TRANSFORMS_SCALAR_TARGET_LEGALIZATION_H_
#define LOOM_TRANSFORMS_SCALAR_TARGET_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_scalar_multiply_add_match_t {
  // Unflagged multiply whose result is owned by add_op.
  loom_op_t* multiply_op;
  // Unflagged add consuming the multiply result.
  loom_op_t* add_op;
  // Other add operand accumulated into the product.
  loom_value_id_t addend;
} loom_scalar_multiply_add_match_t;

// Matches an unflagged |multiply_op| owned solely by an unflagged scalar.addi.
// Attribute and type observers retain the multiply and prevent a match.
bool loom_scalar_match_multiply_add(
    const loom_module_t* module, const loom_op_t* multiply_op,
    loom_scalar_multiply_add_match_t* out_match);

// Replaces a multiply-add match with scalar.fmai.
iree_status_t loom_scalar_fuse_multiply_add_match(
    loom_rewriter_t* rewriter, const loom_scalar_multiply_add_match_t* match);

// Applies the exact shared floating-point clamp recipe.
iree_status_t loom_scalar_legalize_clamp(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Returns the generic scalar legalizer provider. Pipelines should compose this
// after target-specific providers so native target rewrites win before scalar
// reference decomposition.
const loom_target_legalizer_provider_t* loom_scalar_target_legalizer_provider(
    void);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SCALAR_TARGET_LEGALIZATION_H_
