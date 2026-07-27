// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared verifier helpers for ops that name logical indices into a view.

#ifndef LOOM_OPS_VIEW_ACCESS_VERIFIER_H_
#define LOOM_OPS_VIEW_ACCESS_VERIFIER_H_

#include "loom/error/emitter.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies that a static/dynamic index list has one position per view rank.
//
// The static list uses INT64_MIN sentinels for dynamic operands. The dynamic
// operand count must match the sentinel count, and the static index list length
// must match the rank of |view_type| when |view_type| is a view.
iree_status_t loom_view_verify_index_list_rank(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t view_field_name,
    loom_type_t view_type, loom_attribute_t static_indices,
    uint16_t dynamic_index_count);

// Verifies that a logical element access is structurally valid.
//
// In addition to the rank contract, statically known indices must be in bounds
// for statically known view dimensions.
iree_status_t loom_view_verify_element_access(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t view_field_name,
    loom_type_t view_type, loom_attribute_t static_indices,
    uint16_t dynamic_index_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_VIEW_ACCESS_VERIFIER_H_
