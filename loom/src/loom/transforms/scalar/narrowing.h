// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SCALAR_NARROWING_H_
#define LOOM_TRANSFORMS_SCALAR_NARROWING_H_

#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materialization of an arithmetic operand's low 32 bits.
typedef enum loom_scalar_narrowing_operand_kind_e {
  LOOM_SCALAR_NARROWING_OPERAND_REUSE,
  LOOM_SCALAR_NARROWING_OPERAND_CONSTANT,
  LOOM_SCALAR_NARROWING_OPERAND_ADDRESS,
  LOOM_SCALAR_NARROWING_OPERAND_TRUNCATE,
} loom_scalar_narrowing_operand_kind_t;

typedef struct loom_scalar_narrowing_operand_t {
  // Materialization selected before any IR mutation.
  loom_scalar_narrowing_operand_kind_t kind;
  // Existing value for reuse, address conversion, or residual truncation.
  loom_value_id_t value;
  // Signed i32 payload for a constant materialization.
  int64_t constant;
} loom_scalar_narrowing_operand_t;

// Stack-local selection for one owned, unflagged integer arithmetic producer.
// At least one operand narrows without a residual truncation. Values remain
// borrowed from the module until the caller materializes the selected rewrite.
typedef struct loom_scalar_narrowing_plan_t {
  // Scalar operation preserving the low word, or index.madd to decompose.
  loom_op_kind_t kind;
  // Operand recipes in the original arithmetic order.
  loom_scalar_narrowing_operand_t operands[3];
} loom_scalar_narrowing_plan_t;

// Selects a profitable low-word replacement for |producer|, whose result is
// i64 or an address domain whose selected carrier is at least 32 bits.
// Requires one ordinary use and no embedded type uses. Attribute references
// may belong only to |attribute_owner|, or must be absent when it is
// NULL. The caller owns retirement of that observer and the producer after
// replacing the demand. Selection inspects only this producer, its operands,
// and indexed facts; rejected selections allocate no IR and do not
// walk producer chains.
bool loom_scalar_narrowing_select(loom_rewriter_t* rewriter,
                                  loom_op_t* producer,
                                  loom_op_t* attribute_owner,
                                  loom_scalar_narrowing_plan_t* out_plan);

// Builds the selected i32 arithmetic at the rewriter's insertion point.
// Does not replace or erase the producer or its observer.
iree_status_t loom_scalar_narrowing_build(
    loom_rewriter_t* rewriter, const loom_scalar_narrowing_plan_t* plan,
    loom_location_id_t location, loom_op_t** out_op);

// Combines an i64-to-i32 truncation with its already-resolved defining
// operation. Retains the truncation for a single residual operand when
// same-block placement and ordinary-only result uses allow it. Retires the wide
// producer immediately so the existing worklist can continue narrowing its
// newly unshared inputs.
iree_status_t loom_scalar_narrowing_truncate(loom_rewriter_t* rewriter,
                                             loom_op_t* truncation,
                                             loom_op_t* producer,
                                             bool* out_changed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCALAR_NARROWING_H_
