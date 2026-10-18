// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/block_fusion.h"

#include <string.h>

#include "loom/ops/cfg/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/transforms/cfg/block_arguments.h"

static bool loom_cfg_find_fusable_predecessor(const loom_cfg_graph_t* graph,
                                              uint16_t block_index,
                                              loom_op_t** out_predecessor_br,
                                              loom_value_slice_t* out_args) {
  *out_predecessor_br = NULL;
  *out_args = (loom_value_slice_t){0};
  if (block_index == 0) {
    return false;
  }

  const loom_block_t* block = graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  if (predecessors.count != 1) {
    return false;
  }

  const loom_block_t* predecessor = graph->blocks[predecessors.values[0]].block;
  if (predecessor == block) {
    return false;
  }

  loom_op_t* terminator = ((loom_block_t*)predecessor)->last_op;
  if (!loom_cfg_br_isa(terminator) && !loom_low_br_isa(terminator)) {
    return false;
  }
  *out_predecessor_br = terminator;
  *out_args = loom_cfg_br_isa(terminator) ? loom_cfg_br_args(terminator)
                                          : loom_low_br_args(terminator);
  return true;
}

static iree_status_t loom_cfg_move_block_ops_before(loom_rewriter_t* rewriter,
                                                    loom_block_t* block,
                                                    loom_op_t* before_op) {
  loom_op_t* op = block->first_op;
  while (op) {
    loom_op_t* next_op = op->next_op;
    IREE_RETURN_IF_ERROR(loom_rewriter_move_before(rewriter, op, before_op));
    op = next_op;
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_fuse_single_predecessor_blocks(
    loom_rewriter_t* rewriter, const loom_cfg_graph_t* graph,
    const loom_dominance_info_t* dominance, iree_arena_allocator_t* arena,
    uint16_t* out_fused_count) {
  *out_fused_count = 0;
  if (graph->malformed) {
    return iree_ok_status();
  }
  loom_op_t** predecessor_branches = NULL;
  bool* remove_blocks = NULL;
  uint16_t fusion_count = 0;
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    uint16_t block_index = graph->reverse_postorder.values[i];
    const loom_block_t* block = graph->blocks[block_index].block;
    loom_op_t* predecessor_br = NULL;
    loom_value_slice_t replacements = {0};
    if (!loom_cfg_find_fusable_predecessor(graph, block_index, &predecessor_br,
                                           &replacements)) {
      continue;
    }
    if (!loom_cfg_block_arguments_can_replace(
            rewriter->module, dominance, block, replacements, predecessor_br)) {
      continue;
    }
    if (!predecessor_branches) {
      // The branch table and removal mask share one lazy edit allocation.
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, graph->block_count,
          sizeof(*predecessor_branches) + sizeof(*remove_blocks),
          (void**)&predecessor_branches));
      remove_blocks = (bool*)(predecessor_branches + graph->block_count);
      memset(predecessor_branches, 0,
             graph->block_count *
                 (sizeof(*predecessor_branches) + sizeof(*remove_blocks)));
    }
    predecessor_branches[block_index] = predecessor_br;
    remove_blocks[block_index] = true;
    ++fusion_count;
  }
  if (fusion_count == 0) {
    return iree_ok_status();
  }

  // Every selected predecessor dominates its block, so graph-owned reverse
  // postorder contracts chains from their surviving head. Incoming branches
  // move with their predecessor but remain live until their own contraction.
  // Their maintained operands compose argument substitutions without keeping
  // untracked value IDs or querying dominance during the structural edit.
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    uint16_t block_index = graph->reverse_postorder.values[i];
    loom_op_t* predecessor_br = predecessor_branches[block_index];
    if (!predecessor_br) {
      continue;
    }
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    loom_value_slice_t replacements = loom_cfg_br_isa(predecessor_br)
                                          ? loom_cfg_br_args(predecessor_br)
                                          : loom_low_br_args(predecessor_br);
    for (uint16_t argument_index = 0; argument_index < block->arg_count;
         ++argument_index) {
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
          rewriter, loom_block_arg_id(block, argument_index),
          replacements.values[argument_index]));
    }
    IREE_RETURN_IF_ERROR(
        loom_cfg_move_block_ops_before(rewriter, block, predecessor_br));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, predecessor_br));
  }
  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_rewriter_remove_blocks(
      rewriter, (loom_region_t*)graph->region, remove_blocks,
      graph->block_count, arena, &removed_count));
  *out_fused_count = removed_count;
  return iree_ok_status();
}
