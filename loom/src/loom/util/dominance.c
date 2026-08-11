// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/dominance.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

#define LOOM_CFG_DOMINATOR_INVALID UINT16_MAX
#define LOOM_CFG_DOMINATOR_INTERVAL_INVALID UINT32_MAX

struct loom_cfg_dominance_region_t {
  // Region described by graph and dominator arrays.
  const loom_region_t* region;
  // Dense control-flow graph for region.
  loom_cfg_graph_t graph;
  // Immediate dominator per dense block index.
  uint16_t* immediate_dominators;
  // Packed inclusive dominator-tree preorder ranges per dense block index.
  uint32_t* dominator_intervals;
  // True when graph is well-formed enough for CFG dominance queries.
  bool available;
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
  if (!module || !arena || !out_info) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dominance analysis requires a module, arena, and output info");
  }
  memset(out_info, 0, sizeof(*out_info));
  out_info->module = module;
  out_info->arena = arena;
  return loom_dominance_info_build_region(out_info, module->body);
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

typedef struct loom_cfg_traversal_frame_t {
  // Block currently on the iterative traversal stack.
  uint16_t block_index;
  // Next adjacent block position to visit from block_index.
  iree_host_size_t next_position;
} loom_cfg_traversal_frame_t;

typedef struct loom_cfg_dominance_working_set_t {
  // Reachable block indices in reverse postorder.
  uint16_t* rpo_order;
  // Number of reachable entries in rpo_order.
  iree_host_size_t rpo_count;
  // Reverse-postorder position per dense block index.
  iree_host_size_t* rpo_numbers;
  // Reusable stack with graph->block_count entries.
  loom_cfg_traversal_frame_t* traversal_stack;
} loom_cfg_dominance_working_set_t;

static iree_status_t loom_cfg_dominance_compute_rpo(
    const loom_cfg_graph_t* graph, iree_arena_allocator_t* arena,
    loom_cfg_dominance_working_set_t* out_working_set) {
  memset(out_working_set, 0, sizeof(*out_working_set));
  if (graph->block_count == 0) return iree_ok_status();

  bool* visited = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*visited), (void**)&visited));
  memset(visited, 0, graph->block_count * sizeof(*visited));

  loom_cfg_traversal_frame_t* traversal_stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, graph->block_count,
                                                 sizeof(*traversal_stack),
                                                 (void**)&traversal_stack));

  uint16_t* rpo_order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*rpo_order), (void**)&rpo_order));

  iree_host_size_t* rpo_numbers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*rpo_numbers), (void**)&rpo_numbers));

  iree_host_size_t stack_count = 0;
  iree_host_size_t rpo_count = 0;
  visited[0] = true;
  traversal_stack[stack_count++] = (loom_cfg_traversal_frame_t){
      .block_index = 0,
      .next_position = 0,
  };
  while (stack_count > 0) {
    loom_cfg_traversal_frame_t* frame = &traversal_stack[stack_count - 1];
    loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, frame->block_index);
    if (frame->next_position < successors.count) {
      uint16_t successor_index = successors.values[frame->next_position++];
      if (!visited[successor_index]) {
        visited[successor_index] = true;
        traversal_stack[stack_count++] = (loom_cfg_traversal_frame_t){
            .block_index = successor_index,
            .next_position = 0,
        };
      }
      continue;
    }
    rpo_order[rpo_count++] = frame->block_index;
    --stack_count;
  }

  // Reverse the completed postorder in place instead of retaining a second
  // block-index array for the dominance computation.
  for (iree_host_size_t i = 0; i < rpo_count / 2; ++i) {
    uint16_t block_index = rpo_order[i];
    rpo_order[i] = rpo_order[rpo_count - i - 1];
    rpo_order[rpo_count - i - 1] = block_index;
  }
  for (iree_host_size_t i = 0; i < rpo_count; ++i) {
    rpo_numbers[rpo_order[i]] = i;
  }
  out_working_set->rpo_order = rpo_order;
  out_working_set->rpo_count = rpo_count;
  out_working_set->rpo_numbers = rpo_numbers;
  out_working_set->traversal_stack = traversal_stack;
  return iree_ok_status();
}

static uint16_t loom_cfg_dominance_intersect(
    const uint16_t* immediate_dominators, const iree_host_size_t* rpo_numbers,
    uint16_t lhs, uint16_t rhs) {
  while (lhs != rhs) {
    while (rpo_numbers[lhs] > rpo_numbers[rhs]) {
      lhs = immediate_dominators[lhs];
    }
    while (rpo_numbers[rhs] > rpo_numbers[lhs]) {
      rhs = immediate_dominators[rhs];
    }
  }
  return lhs;
}

static void loom_cfg_dominance_compute_intervals(
    loom_cfg_dominance_region_t* cache,
    loom_cfg_dominance_working_set_t* working_set) {
  const iree_host_size_t block_count = cache->graph.block_count;
  if (working_set->rpo_count == 0) return;

  // Reuse the RPO-number array as child counts and then child starts. RPO order
  // is no longer needed after immediate dominators have converged, so reuse it
  // as the compact child table.
  memset(working_set->rpo_numbers, 0,
         block_count * sizeof(*working_set->rpo_numbers));
  for (uint16_t block_index = 1; block_index < block_count; ++block_index) {
    uint16_t immediate_dominator = cache->immediate_dominators[block_index];
    if (immediate_dominator != LOOM_CFG_DOMINATOR_INVALID) {
      ++working_set->rpo_numbers[immediate_dominator];
    }
  }
  iree_host_size_t child_count = 0;
  for (iree_host_size_t block_index = 0; block_index < block_count;
       ++block_index) {
    iree_host_size_t block_child_count = working_set->rpo_numbers[block_index];
    working_set->rpo_numbers[block_index] = child_count;
    child_count += block_child_count;
    cache->dominator_intervals[block_index] =
        (uint32_t)working_set->rpo_numbers[block_index];
  }
  for (uint16_t block_index = 1; block_index < block_count; ++block_index) {
    uint16_t immediate_dominator = cache->immediate_dominators[block_index];
    if (immediate_dominator == LOOM_CFG_DOMINATOR_INVALID) continue;
    uint32_t child_position = cache->dominator_intervals[immediate_dominator]++;
    working_set->rpo_order[child_position] = block_index;
  }
  for (iree_host_size_t block_index = 0; block_index < block_count;
       ++block_index) {
    cache->dominator_intervals[block_index] =
        LOOM_CFG_DOMINATOR_INTERVAL_INVALID;
  }

  iree_host_size_t stack_count = 0;
  uint32_t preorder = 0;
  cache->dominator_intervals[0] = preorder++;
  working_set->traversal_stack[stack_count++] = (loom_cfg_traversal_frame_t){
      .block_index = 0,
      .next_position = working_set->rpo_numbers[0],
  };
  while (stack_count > 0) {
    loom_cfg_traversal_frame_t* frame =
        &working_set->traversal_stack[stack_count - 1];
    iree_host_size_t child_end =
        (iree_host_size_t)frame->block_index + 1 < block_count
            ? working_set->rpo_numbers[frame->block_index + 1]
            : child_count;
    if (frame->next_position < child_end) {
      uint16_t child_index = working_set->rpo_order[frame->next_position++];
      cache->dominator_intervals[child_index] = preorder++;
      working_set->traversal_stack[stack_count++] =
          (loom_cfg_traversal_frame_t){
              .block_index = child_index,
              .next_position = working_set->rpo_numbers[child_index],
          };
      continue;
    }

    uint16_t block_preorder =
        (uint16_t)cache->dominator_intervals[frame->block_index];
    uint16_t block_last_preorder = (uint16_t)(preorder - 1);
    cache->dominator_intervals[frame->block_index] =
        (uint32_t)block_preorder | ((uint32_t)block_last_preorder << 16);
    --stack_count;
  }
}

static iree_status_t loom_cfg_dominance_compute(
    loom_cfg_dominance_region_t* cache, iree_arena_allocator_t* arena) {
  iree_host_size_t block_count = cache->graph.block_count;
  if (block_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, block_count, sizeof(*cache->immediate_dominators),
      (void**)&cache->immediate_dominators));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, block_count, sizeof(*cache->dominator_intervals),
      (void**)&cache->dominator_intervals));
  for (iree_host_size_t i = 0; i < block_count; ++i) {
    cache->immediate_dominators[i] = LOOM_CFG_DOMINATOR_INVALID;
    cache->dominator_intervals[i] = LOOM_CFG_DOMINATOR_INTERVAL_INVALID;
  }

  if (cache->graph.malformed) return iree_ok_status();

  loom_cfg_dominance_working_set_t working_set;
  IREE_RETURN_IF_ERROR(
      loom_cfg_dominance_compute_rpo(&cache->graph, arena, &working_set));
  if (working_set.rpo_count == 0) return iree_ok_status();

  cache->immediate_dominators[0] = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (iree_host_size_t rpo_index = 1; rpo_index < working_set.rpo_count;
         ++rpo_index) {
      uint16_t block_index = working_set.rpo_order[rpo_index];
      uint16_t new_idom = LOOM_CFG_DOMINATOR_INVALID;
      loom_cfg_block_index_span_t predecessors =
          loom_cfg_graph_predecessors(&cache->graph, block_index);
      for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
        uint16_t predecessor_index = predecessors.values[i];
        if (cache->immediate_dominators[predecessor_index] ==
            LOOM_CFG_DOMINATOR_INVALID) {
          continue;
        }
        new_idom =
            new_idom == LOOM_CFG_DOMINATOR_INVALID
                ? predecessor_index
                : loom_cfg_dominance_intersect(cache->immediate_dominators,
                                               working_set.rpo_numbers,
                                               predecessor_index, new_idom);
      }
      if (cache->immediate_dominators[block_index] != new_idom) {
        cache->immediate_dominators[block_index] = new_idom;
        changed = true;
      }
    }
  }

  loom_cfg_dominance_compute_intervals(cache, &working_set);
  cache->available = true;
  return iree_ok_status();
}

static iree_status_t loom_dominance_info_add_cfg_region(
    loom_dominance_info_t* info, const loom_region_t* region) {
  loom_cfg_dominance_region_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(info->arena, sizeof(*cache), (void**)&cache));
  memset(cache, 0, sizeof(*cache));
  cache->region = region;
  IREE_RETURN_IF_ERROR(
      loom_cfg_graph_build(info->module, region, info->arena, &cache->graph));
  IREE_RETURN_IF_ERROR(loom_cfg_dominance_compute(cache, info->arena));
  cache->next = info->cfg_regions;
  info->cfg_regions = cache;
  return iree_ok_status();
}

static iree_status_t loom_dominance_info_build_region(
    loom_dominance_info_t* info, const loom_region_t* region) {
  if (!region) return iree_ok_status();
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG) ||
      region->block_count > 1) {
    IREE_RETURN_IF_ERROR(loom_dominance_info_add_cfg_region(info, region));
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
    if (cache->region == region) return cache;
  }
  return NULL;
}

static bool loom_cfg_region_block_dominates(
    const loom_cfg_dominance_region_t* cache, uint16_t dominator_index,
    uint16_t dominated_index) {
  if (!cache || !cache->available) return false;
  if (dominator_index == dominated_index) return true;
  uint32_t dominator_interval = cache->dominator_intervals[dominator_index];
  uint32_t dominated_interval = cache->dominator_intervals[dominated_index];
  if (dominator_interval == LOOM_CFG_DOMINATOR_INTERVAL_INVALID ||
      dominated_interval == LOOM_CFG_DOMINATOR_INTERVAL_INVALID) {
    return false;
  }
  uint16_t dominator_preorder = (uint16_t)dominator_interval;
  uint16_t dominator_last_preorder = (uint16_t)(dominator_interval >> 16);
  uint16_t dominated_preorder = (uint16_t)dominated_interval;
  return dominator_preorder <= dominated_preorder &&
         dominated_preorder <= dominator_last_preorder;
}

bool loom_dominates_block(const loom_dominance_info_t* info,
                          const loom_block_t* dominator_block,
                          const loom_block_t* dominated_block) {
  if (!dominator_block || !dominated_block) return false;
  if (dominator_block == dominated_block) return true;
  const loom_region_t* region = dominator_block->parent_region;
  if (!region || region != dominated_block->parent_region) return false;
  if (!iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG) &&
      region->block_count <= 1) {
    return false;
  }

  const loom_cfg_dominance_region_t* cache =
      loom_dominance_lookup_cfg_region(info, region);
  if (!cache) return false;
  iree_host_size_t dominator_index =
      loom_cfg_graph_block_index(&cache->graph, dominator_block);
  iree_host_size_t dominated_index =
      loom_cfg_graph_block_index(&cache->graph, dominated_block);
  if (dominator_index == IREE_HOST_SIZE_MAX ||
      dominated_index == IREE_HOST_SIZE_MAX) {
    return false;
  }
  return loom_cfg_region_block_dominates(cache, (uint16_t)dominator_index,
                                         (uint16_t)dominated_index);
}

const loom_block_t* loom_dominance_immediate_dominator_block(
    const loom_dominance_info_t* info, const loom_block_t* block) {
  if (!info || !block) return NULL;
  const loom_region_t* region = block->parent_region;
  if (!region) return NULL;

  const loom_block_t* entry_block = loom_region_const_entry_block(region);
  if (block == entry_block) return NULL;

  if (!iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG) &&
      region->block_count <= 1) {
    return NULL;
  }

  const loom_cfg_dominance_region_t* cache =
      loom_dominance_lookup_cfg_region(info, region);
  if (!cache || !cache->available) return NULL;
  iree_host_size_t block_index =
      loom_cfg_graph_block_index(&cache->graph, block);
  if (block_index == IREE_HOST_SIZE_MAX) return NULL;
  uint16_t immediate_dominator = cache->immediate_dominators[block_index];
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
  if (!info || !info->module || !a || !b) return false;

  // Self-dominance.
  if (a == b) return true;

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
  if (depth_a > depth_b) return false;

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
  if (a == b_at_a_depth) return true;

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
    if (!def_block) return false;

    const loom_op_t* current = use_op;
    while (current) {
      if (loom_dominates_block(info, def_block, current->parent_block)) {
        return true;
      }
      if (!current->parent_op) return false;
      if (loom_op_crosses_isolation_boundary(info, current->parent_op)) {
        return false;
      }
      current = current->parent_op;
    }
    return false;
  }

  // Op result: the defining op must dominate the use op.
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) return false;
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

typedef struct loom_type_availability_query_t {
  // Dominance information used for value-availability queries.
  const loom_dominance_info_t* info;
  // Op before which the type would be materialized.
  const loom_op_t* before_op;
  // Cleared when any embedded SSA reference is unavailable.
  bool available;
} loom_type_availability_query_t;

static iree_status_t loom_type_availability_check_ref(loom_value_id_t value_id,
                                                      void* user_data) {
  loom_type_availability_query_t* query =
      (loom_type_availability_query_t*)user_data;
  if (!loom_value_is_available_before_op(query->info, value_id,
                                         query->before_op)) {
    query->available = false;
  }
  return iree_ok_status();
}

bool loom_type_is_available_before_op(const loom_dominance_info_t* info,
                                      loom_type_t type,
                                      const loom_op_t* before_op) {
  if (!info || !info->module || !before_op) return false;
  loom_type_availability_query_t query = {
      .info = info,
      .before_op = before_op,
      .available = true,
  };
  iree_status_t status = loom_type_walk_value_refs(
      info->module, type, loom_type_availability_check_ref, &query);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return false;
  }
  return query.available;
}

bool loom_value_type_is_available_before_op(const loom_dominance_info_t* info,
                                            loom_value_id_t value_id,
                                            const loom_op_t* before_op) {
  if (!info || !info->module || value_id >= info->module->values.count) {
    return false;
  }
  return loom_type_is_available_before_op(
      info, loom_module_value_type(info->module, value_id), before_op);
}
