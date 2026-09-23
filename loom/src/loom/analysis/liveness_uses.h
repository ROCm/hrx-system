// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Semantic reads used by liveness transfer and interval construction.

#ifndef LOOM_ANALYSIS_LIVENESS_USES_H_
#define LOOM_ANALYSIS_LIVENESS_USES_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*loom_liveness_value_fn_t)(void* user_data,
                                                  loom_value_id_t value_id);

typedef struct loom_liveness_value_callback_t {
  // Function invoked for each visited value.
  loom_liveness_value_fn_t fn;
  // Opaque callback payload passed to |fn|.
  void* user_data;
} loom_liveness_value_callback_t;

static inline loom_liveness_value_callback_t loom_liveness_value_callback_make(
    loom_liveness_value_fn_t fn, void* user_data) {
  return (loom_liveness_value_callback_t){
      .fn = fn,
      .user_data = user_data,
  };
}

// Visits SSA values referenced by a type. Callback errors stop traversal.
iree_status_t loom_liveness_for_each_type_ref(
    const loom_module_t* module, loom_type_t type,
    loom_liveness_value_callback_t visitor);

// Visits reads captured from nested regions, excluding values defined within
// the owning operation's subtree. Repeated reads may produce repeated visits.
iree_status_t loom_liveness_for_each_nested_external_use(
    const loom_module_t* module, const loom_op_t* owner_op,
    loom_liveness_value_callback_t visitor);

// Visits operands and SSA references in operand/result types, excluding the
// operation's own results. Nested-region captures are visited separately.
iree_status_t loom_liveness_for_each_op_direct_use(
    const loom_module_t* module, const loom_op_t* op,
    loom_liveness_value_callback_t visitor);

// Visits direct reads followed by external reads captured in nested regions.
iree_status_t loom_liveness_for_each_op_use(
    const loom_module_t* module, const loom_op_t* op,
    loom_liveness_value_callback_t visitor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_LIVENESS_USES_H_
