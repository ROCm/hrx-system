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

// Materializes an analyzed linear byte expression in result_type (index or
// offset) after anchor_value_id's definition, or at entry for a block argument.
// base_value_id is an optional byte position added to the expression, or
// INVALID for a complete root-relative expression. Identity recipes reuse a
// matching base without changing the builder position. Retained equal SSA
// values preserve producer-established aggregate facts instead of
// reconstructing their terms. Signed index results use target-selected index
// arithmetic. Offset results enter the target's modular physical address domain
// with index.scale and preserve its full width across subsequent translations.
// The caller owns dominance of retained relations and terms at the anchor,
// recipe lifetime, the result domain, and reuse of the returned value.
iree_status_t loom_view_materialize_offset_expression(
    loom_builder_t* builder, const loom_symbolic_expr_t* expression,
    loom_value_id_t base_value_id, loom_type_t result_type,
    loom_value_id_t anchor_value_id, loom_value_id_t* out_value);

// Emits the same retained recipe at the current insertion point. Callers that
// project an invocation already own capture dominance and need no source
// anchor.
iree_status_t loom_view_emit_offset_expression(
    loom_builder_t* builder, const loom_symbolic_expr_t* expression,
    loom_value_id_t base_value_id, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VIEW_OFFSET_EXPRESSION_H_
