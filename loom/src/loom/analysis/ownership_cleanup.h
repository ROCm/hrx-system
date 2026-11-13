// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Planned ownership releases and their dialect-specific materialization.
#ifndef LOOM_ANALYSIS_OWNERSHIP_CLEANUP_H_
#define LOOM_ANALYSIS_OWNERSHIP_CLEANUP_H_

#include "loom/analysis/ownership_lifetime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_ownership_cleanup_action_t {
  // Block receiving the operation when no edge split is required.
  loom_block_t* block;
  // Operation to insert before when no edge split is required.
  loom_op_t* before_op;
  // Terminator owning the successor edge for edge-local actions.
  loom_op_t* terminator;
  // Successor ordinal on |terminator|, or NONE for block-local actions.
  uint16_t successor_index;
  // Resource value whose lifetime action is materialized.
  loom_value_id_t value_id;
  // Index into the active materialization policy table.
  uint16_t policy_index;
  // Source location to stamp on the materialized operation.
  loom_location_id_t location;
} loom_ownership_cleanup_action_t;

// Applies a complete release plan after analysis stops observing source IR.
iree_status_t loom_ownership_cleanup_apply(
    loom_module_t* module, loom_func_like_t function,
    const loom_ownership_lifetime_options_t* options,
    const loom_ownership_cleanup_action_t* actions,
    iree_host_size_t action_count, loom_ownership_lifetime_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_ANALYSIS_OWNERSHIP_CLEANUP_H_
