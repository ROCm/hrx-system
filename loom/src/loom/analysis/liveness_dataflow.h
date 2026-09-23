// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact live-in and live-out relations for one region CFG.

#ifndef LOOM_ANALYSIS_LIVENESS_DATAFLOW_H_
#define LOOM_ANALYSIS_LIVENESS_DATAFLOW_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Block-local transfer facts over region-local value ordinals.
typedef struct loom_liveness_block_transfer_t {
  // Unique upward-exposed uses in this block.
  const loom_value_ordinal_t* use_ordinals;
  // Number of entries in |use_ordinals|.
  iree_host_size_t use_count;
  // Unique values defined in this block.
  const loom_value_ordinal_t* definition_ordinals;
  // Number of entries in |definition_ordinals|.
  iree_host_size_t definition_count;
} loom_liveness_block_transfer_t;

// Exact solved liveness relation for one block.
typedef struct loom_liveness_block_relation_t {
  // Values live at block entry in region-local ordinal order.
  const loom_value_id_t* live_in_values;
  // Number of entries in |live_in_values|.
  iree_host_size_t live_in_count;
  // Values live at block exit in region-local ordinal order.
  const loom_value_id_t* live_out_values;
  // Number of entries in |live_out_values|.
  iree_host_size_t live_out_count;
} loom_liveness_block_relation_t;

// Canonical block-boundary facts for one immutable region snapshot. Legal
// reordering within a block preserves these relations. Moving operations across
// blocks, changing uses or definitions, or rewriting CFG edges invalidates
// them. The value lists borrow SSA identities from the analyzed module and are
// owned by the result arena; they do not borrow the acquired ordinal scratch
// map.
typedef struct loom_liveness_dataflow_t {
  // Exact live-in/live-out relations in region block order.
  const loom_liveness_block_relation_t* blocks;
  // Number of block relations in |blocks|.
  iree_host_size_t block_count;
} loom_liveness_dataflow_t;

// Collects canonical block transfers and solves their boundary relations once.
// The acquired |value_domain| and |cfg_graph| must describe the same immutable
// region. Non-CFG regions use local block relations and an identity-only graph.
// |arena| must outlive every consumer of |out_dataflow|.
iree_status_t loom_liveness_dataflow_analyze(
    const loom_local_value_domain_t* value_domain,
    const loom_cfg_graph_t* cfg_graph, iree_arena_allocator_t* arena,
    loom_liveness_dataflow_t* out_dataflow);

// Solves the least live-in/live-out fixed point for |block_transfers|.
//
// Values are separable in the canonical transfer equations. The solver walks
// predecessors from each upward-exposed use and stores only relations that
// exist. Passing NULL for |cfg_graph| gives every block an empty predecessor
// set, which is the local-region form. A non-NULL graph must describe the same
// |block_count| blocks as |block_transfers|.
//
// The caller provides |out_block_relations| with |block_count| records. Value
// lists are allocated in |result_arena| and remain valid with it. All indexes,
// marks, and worklists are allocated in |scratch_arena| and may be discarded
// as soon as this function returns.
iree_status_t loom_liveness_dataflow_solve(
    const loom_cfg_graph_t* cfg_graph, const loom_value_id_t* value_ids,
    loom_value_ordinal_t value_count,
    const loom_liveness_block_transfer_t* block_transfers, uint16_t block_count,
    loom_liveness_block_relation_t* out_block_relations,
    iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_LIVENESS_DATAFLOW_H_
