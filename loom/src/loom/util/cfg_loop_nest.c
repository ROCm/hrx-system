// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_loop_nest.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/util/cfg_dominance.h"

typedef struct loom_cfg_loop_build_state_t {
  // Outermost discovered loop in the path-compressed contraction forest.
  uint16_t root;
  // First child used when numbering the completed loop tree.
  uint16_t first_child;
  // Next sibling used when numbering the completed loop tree.
  uint16_t next_sibling;
  // Outgoing minus incoming edges, accumulated through the loop tree.
  int32_t boundary_balance;
  // XOR of boundary edge indices plus one, accumulated through the loop tree.
  uint32_t boundary_xor;
  // XOR of external header entry edge indices plus one.
  uint32_t entry_xor;
  // Sum of boundary edge target indices, with entries contributing negatively.
  int64_t boundary_target_sum;
  // Sum of squared boundary target indices with the same signed convention.
  int64_t boundary_target_square_sum;
} loom_cfg_loop_build_state_t;

static uint16_t loom_cfg_loop_find_root(loom_cfg_loop_build_state_t* states,
                                        uint16_t loop_index) {
  uint16_t root = loop_index;
  while (states[root].root != root) {
    root = states[root].root;
  }
  while (states[loop_index].root != loop_index) {
    uint16_t next = states[loop_index].root;
    states[loop_index].root = root;
    loop_index = next;
  }
  return root;
}

static void loom_cfg_loop_add_edge(loom_cfg_loop_edge_summary_t* summary,
                                   loom_cfg_edge_index_t edge_index) {
  summary->unique_index =
      ++summary->count == 1 ? edge_index : LOOM_CFG_EDGE_INDEX_INVALID;
}

// Every ordinary block is mapped once. When a completed subloop is absorbed,
// only its external header predecessors remain to visit: its interior cannot
// contribute another entry to an enclosing natural loop.
static iree_host_size_t loom_cfg_loop_discover(
    const loom_cfg_graph_t* graph, const loom_cfg_dominance_t* dominance,
    uint16_t* stack, uint16_t* innermost, loom_cfg_loop_build_state_t* states,
    loom_cfg_natural_loop_t* loops) {
  iree_host_size_t loop_count = 0;
  for (iree_host_size_t h = dominance->preorder.count; h > 0; --h) {
    uint16_t header = dominance->preorder.values[h - 1];
    loom_cfg_natural_loop_t loop = {
        .header_index = header,
        .parent_loop_index = LOOM_CFG_LOOP_NEST_NONE,
        .continuation_index = LOOM_CFG_LOOP_CONTINUATION_NONE,
        .entries.unique_index = LOOM_CFG_EDGE_INDEX_INVALID,
        .backedges.unique_index = LOOM_CFG_EDGE_INDEX_INVALID,
        .exits.unique_index = LOOM_CFG_EDGE_INDEX_INVALID,
    };
    uint32_t entry_xor = 0;
    iree_host_size_t stack_count = 0;
    loom_cfg_edge_index_span_t incoming =
        loom_cfg_graph_predecessor_edges(graph, header);
    for (iree_host_size_t i = 0; i < incoming.count; ++i) {
      const loom_cfg_edge_info_t* edge = &graph->edges[incoming.values[i]];
      uint16_t source = edge->source_block_index;
      if (!graph->blocks[source].reachable) {
        continue;
      }
      if (loom_cfg_dominance_block_dominates(dominance, header, source)) {
        loom_cfg_loop_add_edge(&loop.backedges, incoming.values[i]);
        stack[stack_count++] = source;
      } else {
        loom_cfg_loop_add_edge(&loop.entries, incoming.values[i]);
        entry_xor ^= incoming.values[i] + 1;
      }
    }
    if (loop.backedges.count == 0) {
      continue;
    }
    const uint16_t loop_index = (uint16_t)loop_count++;
    loops[loop_index] = loop;
    states[loop_index] = (loom_cfg_loop_build_state_t){
        .root = loop_index,
        .first_child = LOOM_CFG_LOOP_NEST_NONE,
        .next_sibling = LOOM_CFG_LOOP_NEST_NONE,
        .entry_xor = entry_xor,
    };
    innermost[header] = loop_index;
    while (stack_count > 0) {
      uint16_t block = stack[--stack_count];
      if (!graph->blocks[block].reachable) {
        continue;
      }
      uint16_t subloop = innermost[block];
      if (subloop != LOOM_CFG_LOOP_NEST_NONE) {
        subloop = loom_cfg_loop_find_root(states, subloop);
        if (subloop == loop_index) {
          continue;
        }
        loops[subloop].parent_loop_index = loop_index;
        states[subloop].root = loop_index;
        block = loops[subloop].header_index;
      } else {
        innermost[block] = loop_index;
      }
      loom_cfg_block_index_span_t predecessors =
          loom_cfg_graph_predecessors(graph, block);
      for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
        uint16_t source = predecessors.values[i];
        if (subloop == LOOM_CFG_LOOP_NEST_NONE ||
            !loom_cfg_dominance_block_dominates(dominance, block, source)) {
          stack[stack_count++] = source;
        }
      }
    }
  }
  return loop_count;
}

static void loom_cfg_loop_link_tree(iree_host_size_t loop_count,
                                    loom_cfg_loop_build_state_t* states,
                                    const loom_cfg_natural_loop_t* loops) {
  for (iree_host_size_t i = 0; i < loop_count; ++i) {
    uint16_t parent = loops[i].parent_loop_index;
    if (parent == LOOM_CFG_LOOP_NEST_NONE) {
      continue;
    }
    states[i].next_sibling = states[parent].first_child;
    states[parent].first_child = (uint16_t)i;
  }
}

static bool loom_cfg_loop_contains_innermost(
    const loom_cfg_natural_loop_t* loops, uint16_t loop_index,
    uint16_t innermost_loop_index) {
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE ||
      innermost_loop_index == LOOM_CFG_LOOP_NEST_NONE) {
    return false;
  }
  uint32_t interval = loops[loop_index].interval;
  uint16_t position = (uint16_t)loops[innermost_loop_index].interval;
  return (uint16_t)interval <= position && position <= (interval >> 16);
}

// Edge balances cancel at the least common ancestor of their endpoints. A
// natural loop has no side entries, so its external header entries account for
// every remaining negative contribution. Their count, edge XOR, and target
// moments recover the exit summary without walking each edge through all
// enclosing loops.
static bool loom_cfg_loop_summarize_boundaries(
    const loom_cfg_graph_t* graph, const loom_cfg_dominance_t* dominance,
    const uint16_t* innermost, iree_host_size_t loop_count,
    loom_cfg_loop_build_state_t* states, loom_cfg_natural_loop_t* loops) {
  bool reducible = true;
  for (iree_host_size_t i = 0; i < graph->edge_count; ++i) {
    const loom_cfg_edge_info_t* edge = &graph->edges[i];
    const loom_cfg_block_info_t* source =
        &graph->blocks[edge->source_block_index];
    const loom_cfg_block_info_t* target_block =
        &graph->blocks[edge->target_block_index];
    if (!source->reachable) {
      continue;
    }
    if (target_block->preorder <= source->preorder &&
        source->preorder < target_block->preorder_end &&
        !loom_cfg_dominance_block_dominates(dominance, edge->target_block_index,
                                            edge->source_block_index)) {
      reducible = false;
    }
    uint16_t from = innermost[edge->source_block_index];
    uint16_t to = innermost[edge->target_block_index];
    if (from == to) {
      continue;
    }
    const int64_t target_index = edge->target_block_index;
    const int64_t target_square = target_index * target_index;
    if (from != LOOM_CFG_LOOP_NEST_NONE) {
      ++states[from].boundary_balance;
      states[from].boundary_xor ^= (uint32_t)i + 1;
      states[from].boundary_target_sum += target_index;
      states[from].boundary_target_square_sum += target_square;
      if (!loom_cfg_loop_contains_innermost(loops, from, to)) {
        ++loops[from].direct_exit_count;
      }
    }
    if (to != LOOM_CFG_LOOP_NEST_NONE) {
      --states[to].boundary_balance;
      states[to].boundary_xor ^= (uint32_t)i + 1;
      states[to].boundary_target_sum -= target_index;
      states[to].boundary_target_square_sum -= target_square;
    }
  }
  for (iree_host_size_t i = 0; i < loop_count; ++i) {
    loom_cfg_natural_loop_t* loop = &loops[i];
    loom_cfg_loop_build_state_t* state = &states[i];
    if (loop->parent_loop_index != LOOM_CFG_LOOP_NEST_NONE) {
      loom_cfg_loop_build_state_t* parent = &states[loop->parent_loop_index];
      parent->boundary_balance += state->boundary_balance;
      parent->boundary_xor ^= state->boundary_xor;
      parent->boundary_target_sum += state->boundary_target_sum;
      parent->boundary_target_square_sum += state->boundary_target_square_sum;
    }
    loop->exits.count = state->boundary_balance + loop->entries.count;
    if (loop->exits.count == 1) {
      loop->exits.unique_index = (state->boundary_xor ^ state->entry_xor) - 1;
    }
    if (loop->exits.count != 0) {
      const int64_t header = loop->header_index;
      const int64_t exit_target_sum =
          state->boundary_target_sum + loop->entries.count * header;
      const int64_t exit_target_square_sum =
          state->boundary_target_square_sum +
          loop->entries.count * header * header;
      const int64_t exit_count = loop->exits.count;
      if (exit_target_sum % exit_count == 0) {
        const int64_t candidate = exit_target_sum / exit_count;
        if (candidate >= 0 && candidate < graph->block_count &&
            exit_target_square_sum == exit_count * candidate * candidate) {
          loop->continuation_index = (uint16_t)candidate;
        }
      }
    }
  }
  return reducible;
}

static void loom_cfg_loop_number_tree(iree_host_size_t loop_count,
                                      uint16_t* stack,
                                      loom_cfg_loop_build_state_t* states,
                                      loom_cfg_natural_loop_t* loops) {
  uint32_t position = 0;
  for (iree_host_size_t i = loop_count; i > 0; --i) {
    if (loops[i - 1].parent_loop_index != LOOM_CFG_LOOP_NEST_NONE) {
      continue;
    }
    iree_host_size_t stack_count = 1;
    stack[0] = (uint16_t)(i - 1);
    loops[i - 1].interval = position++;
    while (stack_count > 0) {
      uint16_t loop_index = stack[stack_count - 1];
      uint16_t child = states[loop_index].first_child;
      if (child != LOOM_CFG_LOOP_NEST_NONE) {
        states[loop_index].first_child = states[child].next_sibling;
        loops[child].interval = position++;
        stack[stack_count++] = child;
      } else {
        loops[loop_index].interval |= (position - 1) << 16;
        --stack_count;
      }
    }
  }
}

static iree_status_t loom_cfg_loop_nest_build_impl(
    const loom_cfg_graph_t* graph, const loom_cfg_dominance_t* dominance,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    loom_cfg_loop_nest_t* out_nest) {
  uint16_t* stack = NULL;
  uint16_t* innermost = NULL;
  loom_cfg_loop_build_state_t* states = NULL;
  loom_cfg_natural_loop_t* loops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, graph->edge_count, sizeof(*stack), (void**)&stack));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, graph->block_count,
                                sizeof(*innermost), (void**)&innermost));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, graph->block_count, sizeof(*states), (void**)&states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, graph->block_count, sizeof(*loops), (void**)&loops));
  memset(innermost, 0xFF, graph->block_count * sizeof(*innermost));
  iree_host_size_t loop_count =
      loom_cfg_loop_discover(graph, dominance, stack, innermost, states, loops);
  loom_cfg_loop_link_tree(loop_count, states, loops);
  loom_cfg_loop_number_tree(loop_count, stack, states, loops);
  out_nest->reducible = loom_cfg_loop_summarize_boundaries(
      graph, dominance, innermost, loop_count, states, loops);
  if (loop_count == 0) {
    return iree_ok_status();
  }
  loom_cfg_natural_loop_t* retained_loops = NULL;
  uint16_t* retained_innermost = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, loop_count, sizeof(*retained_loops), (void**)&retained_loops));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, graph->block_count,
                                                 sizeof(*retained_innermost),
                                                 (void**)&retained_innermost));
  memcpy(retained_loops, loops, loop_count * sizeof(*retained_loops));
  memcpy(retained_innermost, innermost,
         graph->block_count * sizeof(*retained_innermost));
  out_nest->loops = retained_loops;
  out_nest->innermost_loop_indices = retained_innermost;
  out_nest->loop_count = loop_count;
  return iree_ok_status();
}

iree_status_t loom_cfg_loop_nest_build(const loom_cfg_graph_t* graph,
                                       const loom_cfg_dominance_t* dominance,
                                       iree_arena_allocator_t* arena,
                                       loom_cfg_loop_nest_t* out_nest) {
  *out_nest = (loom_cfg_loop_nest_t){
      .graph = graph,
      .reducible = !graph->malformed,
  };
  if (graph->malformed || !graph->has_cycles) {
    return iree_ok_status();
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  iree_status_t status = loom_cfg_loop_nest_build_impl(
      graph, dominance, &scratch_arena, arena, out_nest);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static bool loom_cfg_loop_nest_has_counted_header(
    const loom_cfg_loop_nest_t* nest, const loom_cfg_natural_loop_t* loop) {
  return loop->entries.count == 1 && loop->backedges.count == 1 &&
         loop->exits.count == 1 &&
         nest->graph->edges[loop->exits.unique_index].source_block_index ==
             loop->header_index;
}

static uint64_t loom_cfg_loop_nest_bound_product(uint64_t lhs, uint64_t rhs) {
  uint64_t product;
  return iree_checked_mul_u64(lhs, rhs, &product) ? product : UINT64_MAX;
}

bool loom_cfg_loop_nest_calculate_bounds(const loom_cfg_loop_nest_t* nest,
                                         const uint64_t* trip_bounds,
                                         uint64_t* out_block_bounds,
                                         bool* out_exact) {
  const loom_cfg_graph_t* graph = nest->graph;
  bool bounded = nest->reducible;
  *out_exact = true;
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    out_block_bounds[i] = !graph->blocks[i].reachable            ? 0
                          : graph->blocks[i].component_is_cyclic ? UINT64_MAX
                                                                 : 1;
    if (!graph->blocks[i].reachable || graph->blocks[i].successor_count <= 1) {
      continue;
    }
    const uint16_t loop_index = loom_cfg_loop_nest_innermost(nest, (uint16_t)i);
    if (graph->blocks[i].successor_count != 2 ||
        loop_index == LOOM_CFG_LOOP_NEST_NONE ||
        nest->loops[loop_index].header_index != i) {
      *out_exact = false;
    }
  }
  if (!nest->reducible) {
    return false;
  }
  // Header slots first hold body bounds, with parents preceding children.
  for (iree_host_size_t i = nest->loop_count; i > 0; --i) {
    const loom_cfg_natural_loop_t* loop = &nest->loops[i - 1];
    const uint64_t trips = loom_cfg_loop_nest_has_counted_header(nest, loop)
                               ? trip_bounds[i - 1]
                               : UINT64_MAX;
    const uint64_t parent =
        loop->parent_loop_index == LOOM_CFG_LOOP_NEST_NONE
            ? 1
            : out_block_bounds[nest->loops[loop->parent_loop_index]
                                   .header_index];
    out_block_bounds[loop->header_index] =
        loom_cfg_loop_nest_bound_product(parent, trips);
    bounded &= trips != UINT64_MAX &&
               out_block_bounds[loop->header_index] != UINT64_MAX;
  }
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    const uint16_t loop = loom_cfg_loop_nest_innermost(nest, (uint16_t)i);
    if (loop != LOOM_CFG_LOOP_NEST_NONE &&
        nest->loops[loop].header_index != i) {
      out_block_bounds[i] = out_block_bounds[nest->loops[loop].header_index];
    }
  }
  // Child headers consume their parent's body bound before it is overwritten.
  for (iree_host_size_t i = 0; i < nest->loop_count; ++i) {
    const loom_cfg_natural_loop_t* loop = &nest->loops[i];
    const uint64_t trips = loom_cfg_loop_nest_has_counted_header(nest, loop)
                               ? trip_bounds[i]
                               : UINT64_MAX;
    const uint64_t parent =
        loop->parent_loop_index == LOOM_CFG_LOOP_NEST_NONE
            ? 1
            : out_block_bounds[nest->loops[loop->parent_loop_index]
                                   .header_index];
    out_block_bounds[loop->header_index] = loom_cfg_loop_nest_bound_product(
        parent, trips == UINT64_MAX ? UINT64_MAX : trips + 1);
    bounded &= out_block_bounds[loop->header_index] != UINT64_MAX;
  }
  return bounded;
}

bool loom_cfg_loop_nest_calculate_block_multipliers(
    const loom_cfg_loop_nest_t* nest, const uint64_t* trip_counts,
    uint64_t* out_block_multipliers) {
  bool exact;
  return loom_cfg_loop_nest_calculate_bounds(nest, trip_counts,
                                             out_block_multipliers, &exact);
}

bool loom_cfg_loop_nest_calculate_block_execution_counts(
    const loom_cfg_loop_nest_t* nest, const uint64_t* trip_counts,
    uint64_t* out_block_counts) {
  return loom_cfg_loop_nest_calculate_block_execution_bounds(nest, trip_counts,
                                                             out_block_counts);
}

bool loom_cfg_loop_nest_calculate_block_execution_bounds(
    const loom_cfg_loop_nest_t* nest, const uint64_t* trip_bounds,
    uint64_t* out_block_bounds) {
  bool exact;
  const bool bounded = loom_cfg_loop_nest_calculate_bounds(
      nest, trip_bounds, out_block_bounds, &exact);
  return bounded && exact;
}

void loom_cfg_loop_nest_order_blocks(const loom_cfg_loop_nest_t* nest,
                                     uint16_t* out_blocks) {
  const loom_cfg_graph_t* graph = nest->graph;
  uint16_t block = 0;
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    out_blocks[i] = block;
    const loom_cfg_edge_index_span_t successors =
        loom_cfg_graph_successor_edges(graph, block);
    if (!successors.count) {
      break;
    }
    const uint16_t loop_index = loom_cfg_loop_nest_innermost(nest, block);
    const loom_cfg_natural_loop_t* loop =
        loop_index == LOOM_CFG_LOOP_NEST_NONE ? NULL : &nest->loops[loop_index];
    loom_cfg_edge_index_t edge =
        loop && loop->header_index == block
            ? successors
                  .values[successors.values[0] == loop->exits.unique_index]
            : successors.values[0];
    // A backedge closes this one structural visit, including a loop whose
    // header is also its latch. An inner exit may itself be an outer backedge;
    // each closed loop contributes its retained exit edge exactly once.
    for (;;) {
      block = graph->edges[edge].target_block_index;
      const uint16_t target_loop = loom_cfg_loop_nest_innermost(nest, block);
      if (target_loop == LOOM_CFG_LOOP_NEST_NONE ||
          edge != nest->loops[target_loop].backedges.unique_index) {
        break;
      }
      edge = nest->loops[target_loop].exits.unique_index;
    }
  }
}
