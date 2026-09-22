// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/dominance.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_dominance.h"
#include "loom/util/cfg_graph.h"

struct loom_cfg_dominance_region_t {
  // Region described by graph and dominator arrays.
  const loom_region_t* region;
  // Dense control-flow graph for region.
  loom_cfg_graph_t graph;
  // Indexed dominator tree for the graph snapshot.
  loom_cfg_dominance_t dominance;
  // Next cached CFG region in loom_dominance_info_t::cfg_regions.
  struct loom_cfg_dominance_region_t* next;
};

//===----------------------------------------------------------------------===//
// Dominance info
//===----------------------------------------------------------------------===//

static iree_status_t loom_dominance_info_build_region(
    loom_dominance_info_t* info, const loom_region_t* region);

iree_status_t loom_dominance_info_initialize(const loom_module_t* module,
                                             iree_arena_allocator_t* arena,
                                             loom_dominance_info_t* out_info) {
  if (!module) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dominance analysis requires a module, region, arena, and output info");
  }
  return loom_dominance_info_initialize_region(module, module->body, arena,
                                               out_info);
}

iree_status_t loom_dominance_info_initialize_region(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_dominance_info_t* out_info) {
  if (!module || !region || !arena || !out_info) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dominance analysis requires a module, region, arena, and output info");
  }
  memset(out_info, 0, sizeof(*out_info));
  out_info->module = module;
  out_info->arena = arena;
  return loom_dominance_info_build_region(out_info, region);
}

//===----------------------------------------------------------------------===//
// Ancestry helpers
//===----------------------------------------------------------------------===//

uint16_t loom_op_nesting_depth(const loom_op_t* op) {
  uint16_t depth = 0;
  const loom_op_t* current = op->parent_op;
  while (current) {
    ++depth;
    current = current->parent_op;
  }
  return depth;
}

const loom_op_t* loom_op_ancestor_at_depth(const loom_op_t* op,
                                           uint16_t target_depth) {
  uint16_t depth = loom_op_nesting_depth(op);
  IREE_ASSERT(target_depth <= depth);
  while (depth > target_depth) {
    op = op->parent_op;
    --depth;
  }
  return op;
}

//===----------------------------------------------------------------------===//
// CFG dominance construction
//===----------------------------------------------------------------------===//

iree_status_t loom_dominance_info_add_cfg_graph(
    loom_dominance_info_t* info, const loom_cfg_graph_t* graph,
    const loom_cfg_dominance_t* dominance) {
  loom_cfg_dominance_region_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(info->arena, sizeof(*cache), (void**)&cache));
  *cache = (loom_cfg_dominance_region_t){
      .region = graph->region,
      .graph = *graph,
      .next = info->cfg_regions,
  };
  if (dominance) {
    cache->dominance = *dominance;
  } else {
    IREE_RETURN_IF_ERROR(loom_cfg_dominance_build(&cache->graph, info->arena,
                                                  &cache->dominance));
  }
  info->cfg_regions = cache;
  return iree_ok_status();
}

static iree_status_t loom_dominance_info_build_region(
    loom_dominance_info_t* info, const loom_region_t* region) {
  if (!region) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG) ||
      region->block_count > 1) {
    loom_cfg_graph_t graph;
    IREE_RETURN_IF_ERROR(
        loom_cfg_graph_build(info->module, region, info->arena, &graph));
    IREE_RETURN_IF_ERROR(loom_dominance_info_add_cfg_graph(info, &graph, NULL));
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = block->first_op;
    while (op) {
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_dominance_info_build_region(info, regions[i]));
      }
      op = op->next_op;
    }
  }
  return iree_ok_status();
}

static const loom_cfg_dominance_region_t* loom_dominance_lookup_cfg_region(
    const loom_dominance_info_t* info, const loom_region_t* region) {
  for (const loom_cfg_dominance_region_t* cache = info->cfg_regions; cache;
       cache = cache->next) {
    if (cache->region == region) {
      return cache;
    }
  }
  return NULL;
}

bool loom_dominates_block(const loom_dominance_info_t* info,
                          const loom_block_t* dominator_block,
                          const loom_block_t* dominated_block) {
  if (!dominator_block || !dominated_block) {
    return false;
  }
  if (dominator_block == dominated_block) {
    return true;
  }
  const loom_region_t* region = dominator_block->parent_region;
  if (!region || region != dominated_block->parent_region) {
    return false;
  }
  if (!iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG) &&
      region->block_count <= 1) {
    return false;
  }

  const loom_cfg_dominance_region_t* cache =
      loom_dominance_lookup_cfg_region(info, region);
  if (!cache || !cache->dominance.available) {
    return false;
  }
  iree_host_size_t dominator_index =
      loom_cfg_graph_block_index(&cache->graph, dominator_block);
  iree_host_size_t dominated_index =
      loom_cfg_graph_block_index(&cache->graph, dominated_block);
  if (dominator_index == IREE_HOST_SIZE_MAX ||
      dominated_index == IREE_HOST_SIZE_MAX) {
    return false;
  }
  return loom_cfg_dominance_block_dominates(
      &cache->dominance, (uint16_t)dominator_index, (uint16_t)dominated_index);
}

const loom_block_t* loom_dominance_immediate_dominator_block(
    const loom_dominance_info_t* info, const loom_block_t* block) {
  if (!info || !block) {
    return NULL;
  }
  const loom_region_t* region = block->parent_region;
  if (!region) {
    return NULL;
  }

  const loom_block_t* entry_block = loom_region_const_entry_block(region);
  if (block == entry_block) {
    return NULL;
  }

  if (!iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG) &&
      region->block_count <= 1) {
    return NULL;
  }

  const loom_cfg_dominance_region_t* cache =
      loom_dominance_lookup_cfg_region(info, region);
  if (!cache || !cache->dominance.available) {
    return NULL;
  }
  iree_host_size_t block_index =
      loom_cfg_graph_block_index(&cache->graph, block);
  if (block_index == IREE_HOST_SIZE_MAX) {
    return NULL;
  }
  uint16_t immediate_dominator =
      cache->dominance.immediate_dominators[block_index];
  if (immediate_dominator == LOOM_CFG_DOMINATOR_INVALID ||
      immediate_dominator == block_index ||
      immediate_dominator >= cache->graph.block_count) {
    return NULL;
  }
  return cache->graph.blocks[immediate_dominator].block;
}

static bool loom_op_crosses_isolation_boundary(
    const loom_dominance_info_t* info, const loom_op_t* op) {
  (void)info;
  return loom_traits_is_isolated(op->traits);
}

//===----------------------------------------------------------------------===//
// Dominance queries
//===----------------------------------------------------------------------===//

bool loom_dominates_op(const loom_dominance_info_t* info, const loom_op_t* a,
                       const loom_op_t* b) {
  if (!info || !info->module || !a || !b) {
    return false;
  }

  // Self-dominance.
  if (a == b) {
    return true;
  }

  // Same block: compare sparse block ordinals.
  if (a->parent_block == b->parent_block) {
    return a->block_ordinal < b->block_ordinal;
  }

  // Same parent op, different blocks: region block dominance.
  if (a->parent_op == b->parent_op && a->parent_op != NULL) {
    return loom_dominates_block(info, a->parent_block, b->parent_block);
  }

  // Different nesting depths or different parent ops.
  // Walk ancestry chains to find the relationship.
  uint16_t depth_a = loom_op_nesting_depth(a);
  uint16_t depth_b = loom_op_nesting_depth(b);

  // If a is deeper than b, a cannot dominate b (inner doesn't
  // dominate outer).
  if (depth_a > depth_b) {
    return false;
  }

  // Check if a is an ancestor of b: walk b up to a's depth and
  // see if we reach a's parent_op at a's level. If a's parent_op
  // is on b's ancestry chain and a comes before the descendant
  // in the same block, a dominates b.
  //
  // Walk b up to depth_a. At each step, the op we're looking at
  // is the child at that nesting level.
  const loom_op_t* b_at_a_depth = b;
  uint16_t current_depth = depth_b;
  while (current_depth > depth_a) {
    // Check for isolation boundary on b's path.
    if (b_at_a_depth->parent_op &&
        loom_op_crosses_isolation_boundary(info, b_at_a_depth->parent_op)) {
      return false;
    }
    b_at_a_depth = b_at_a_depth->parent_op;
    --current_depth;
  }

  // Now b_at_a_depth is at the same depth as a.
  //
  // If a == b_at_a_depth, then a is the op whose region contains b.
  // An op dominates everything in its own regions, so a dominates b.
  // (This is distinct from self-dominance, which was handled above —
  // here a != b but a is b's ancestor.)
  if (a == b_at_a_depth) {
    return true;
  }

  // Different ops at the same depth in the same block: compare sparse block
  // ordinals.
  if (a->parent_block == b_at_a_depth->parent_block) {
    return a->block_ordinal < b_at_a_depth->block_ordinal;
  }

  // Same parent op at the same depth, different blocks: entry block
  // or CFG block dominance.
  if (a->parent_op == b_at_a_depth->parent_op && a->parent_op != NULL) {
    return loom_dominates_block(info, a->parent_block,
                                b_at_a_depth->parent_block);
  }

  // Different parent ops at the same depth: neither dominates.
  return false;
}

bool loom_dominates_value(const loom_dominance_info_t* info,
                          loom_value_id_t value_id, const loom_op_t* use_op) {
  if (!info || !info->module || !use_op ||
      value_id >= info->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(info->module, value_id);

  if (value->flags & LOOM_VALUE_FLAG_BLOCK_ARG) {
    // Block argument: dominates all ops in dominated blocks and all ops in
    // nested regions reached without crossing an isolation boundary.
    const loom_block_t* def_block = loom_value_def_block(value);
    if (!def_block) {
      return false;
    }

    const loom_op_t* current = use_op;
    while (current) {
      if (loom_dominates_block(info, def_block, current->parent_block)) {
        return true;
      }
      if (!current->parent_op) {
        return false;
      }
      if (loom_op_crosses_isolation_boundary(info, current->parent_op)) {
        return false;
      }
      current = current->parent_op;
    }
    return false;
  }

  // Op result: the defining op must dominate the use op.
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) {
    return false;
  }
  return loom_dominates_op(info, def_op, use_op);
}

bool loom_value_is_available_before_op(const loom_dominance_info_t* info,
                                       loom_value_id_t value_id,
                                       const loom_op_t* before_op) {
  if (!info || !info->module || !before_op ||
      value_id >= info->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(info->module, value_id);
  if (!loom_value_is_block_arg(value) &&
      loom_value_def_op(value) == before_op) {
    return false;
  }
  return loom_dominates_value(info, value_id, before_op);
}
