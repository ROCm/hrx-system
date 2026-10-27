// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/entry_materialization.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/cfg_dominance.h"
#include "loom/util/cfg_graph.h"

iree_status_t loom_low_entry_materialization_plan(const loom_module_t* module,
                                                  const loom_region_t* body,
                                                  const loom_value_id_t* values,
                                                  iree_host_size_t value_count,
                                                  iree_arena_allocator_t* arena,
                                                  loom_op_t** out_before_ops) {
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_op_t* entry_anchor = loom_region_const_entry_block(body)->first_op;
  while (loom_low_function_entry_preamble_op(entry_anchor)) {
    entry_anchor = entry_anchor->next_op;
  }

  loom_cfg_graph_t graph = {0};
  IREE_RETURN_IF_ERROR(loom_cfg_graph_build(module, body, arena, &graph));
  loom_cfg_dominance_t dominance = {0};
  IREE_RETURN_IF_ERROR(loom_cfg_dominance_build(&graph, arena, &dominance));
  uint16_t* acyclic_dominators = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, graph.block_count,
                                                 sizeof(*acyclic_dominators),
                                                 (void**)&acyclic_dominators));
  // Dominance preorder visits each parent before its children. SCC membership
  // covers both natural and irreducible cycles without reconstructing loops.
  for (iree_host_size_t i = 0; i < dominance.preorder.count; ++i) {
    const uint16_t block = dominance.preorder.values[i];
    acyclic_dominators[block] =
        block != 0 && graph.blocks[block].component_is_cyclic
            ? acyclic_dominators[dominance.immediate_dominators[block]]
            : block;
  }

  for (iree_host_size_t i = 0; i < value_count; ++i) {
    out_before_ops[i] = NULL;
    if (values[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, values[i]);
    if (loom_module_value_has_type_uses(module, values[i]) ||
        loom_value_has_attribute_uses(value)) {
      out_before_ops[i] = entry_anchor;
      continue;
    }
    uint16_t common_block = LOOM_CFG_DOMINATOR_INVALID;
    loom_op_t* anchor = NULL;
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t j = 0; j < value->use_count; ++j) {
      loom_op_t* user = loom_use_user_op(uses[j]);
      while (user->parent_block->parent_region != body) {
        user = user->parent_op;
      }
      const uint16_t block =
          (uint16_t)loom_cfg_graph_block_index(&graph, user->parent_block);
      if (!graph.blocks[block].reachable ||
          loom_low_function_entry_preamble_op(user)) {
        common_block = 0;
        anchor = entry_anchor;
        break;
      }
      if (common_block == LOOM_CFG_DOMINATOR_INVALID) {
        common_block = block;
        anchor = user;
      } else {
        const uint16_t previous_block = common_block;
        while (!loom_cfg_dominance_block_dominates(&dominance, common_block,
                                                   block)) {
          common_block = dominance.immediate_dominators[common_block];
        }
        if (common_block != previous_block) {
          anchor = (loom_op_t*)graph.blocks[common_block].block->last_op;
        }
        if (common_block == block &&
            user->block_ordinal < anchor->block_ordinal) {
          anchor = user;
        }
      }
    }
    if (common_block != LOOM_CFG_DOMINATOR_INVALID &&
        acyclic_dominators[common_block] != common_block) {
      common_block = acyclic_dominators[common_block];
      anchor = (loom_op_t*)graph.blocks[common_block].block->last_op;
    }
    out_before_ops[i] = anchor;
  }
  return iree_ok_status();
}
