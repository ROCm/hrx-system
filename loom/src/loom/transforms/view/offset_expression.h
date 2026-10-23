// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VIEW_OFFSET_EXPRESSION_H_
#define LOOM_TRANSFORMS_VIEW_OFFSET_EXPRESSION_H_

#include "loom/analysis/symbolic_expr.h"
#include "loom/ops/builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materializes an analyzed linear byte expression after anchor_value_id's
// definition, or at block entry for a block argument. A zero displacement
// reuses a supplied base_value_id without changing the builder position.
// base_value_id is an optional offset added to the expression, or INVALID for a
// complete root-relative expression. A retained equal SSA value is reused for
// a complete expression so producer-established aggregate facts remain
// attached. Otherwise, sign-known translations preserve the offset domain and
// complete expressions use full-width signed arithmetic with the normal
// index.cast legality requirements. The caller owns the dominance of retained
// relations and canonical terms at the anchor, recipe lifetime, and reuse of
// the returned value.
iree_status_t loom_view_materialize_offset_expression(
    loom_builder_t* builder, const loom_symbolic_expr_t* expression,
    loom_value_id_t base_value_id, loom_value_id_t anchor_value_id,
    loom_value_id_t* out_value);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VIEW_OFFSET_EXPRESSION_H_
