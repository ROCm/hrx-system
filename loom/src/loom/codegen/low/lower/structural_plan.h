// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained type and control decisions for shared structural lowering.

#ifndef LOOM_CODEGEN_LOW_LOWER_STRUCTURAL_PLAN_H_
#define LOOM_CODEGEN_LOW_LOWER_STRUCTURAL_PLAN_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_structural_plan_t {
  // Per-block exact branch choices: zero for dynamic, one for false, and two
  // for true. NULL when all edges are dynamic or unconditional.
  uint8_t* branch_conditions;
} loom_low_lower_structural_plan_t;

// Retains an exact conditional edge before structural storage demand is marked.
// Both demand analysis and emission consume this same decision.
iree_status_t loom_low_lower_structural_plan_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op);

// Returns the retained condition when planning proved an exact branch choice.
bool loom_low_lower_structural_branch_exact_bool(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_condition);

// Maps a source block's arguments during planning. Top-level CFG carriers are
// retained for block creation; structured region arguments are validated here
// and are supplied by their owning structured operation during emission.
iree_status_t loom_low_lower_structural_plan_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_block_t* source_block);

// Returns the retained native type of a non-entry CFG block argument.
loom_type_t loom_low_lower_structural_block_argument_type(
    const loom_low_lower_context_t* context, uint16_t block_index,
    uint16_t argument_index);

// Retains result carriers for structured control,
// including the distinct header tuple of scf.while. Rejects unconsumed source
// pipeline annotations before any Low operation is constructed.
iree_status_t loom_low_lower_structural_plan_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op);

// Expands the operation's selected result bindings into emission scratch.
// Zero-sized tuples produce NULL. The header output is used only for scf.while;
// other callers pass NULL.
iree_status_t loom_low_lower_structural_take_types(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t** out_result_types, loom_type_t** out_header_types);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_STRUCTURAL_PLAN_H_
