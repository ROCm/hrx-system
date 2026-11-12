// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/cfg_converge.h"

#include "loom/ir/float_facts.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/value_refs.h"
#include "loom/ir/value_replacement.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/special_values.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/util/fact_cfg.h"

#define LOOM_CFG_CONVERGE_STATISTICS(V, statistics_type)                  \
  V(statistics_type, decisions_factored, "decisions-factored",            \
    "Number of overlapping decisions factored through a predicate join.") \
  V(statistics_type, loop_exit_frontiers_isolated,                        \
    "loop-exit-frontiers-isolated",                                       \
    "Number of loop exit frontiers isolated from enclosing decisions.")   \
  V(statistics_type, downstream_frontiers_factored,                       \
    "downstream-frontiers-factored",                                      \
    "Number of downstream shared bodies given a predicate join.")         \
  V(statistics_type, values_carried, "values-carried",                    \
    "Number of scalar live-ins and payloads carried across predicate joins.")

LOOM_PASS_STATISTICS_DEFINE(loom_cfg_converge_statistics,
                            loom_cfg_converge_statistics_t,
                            LOOM_CFG_CONVERGE_STATISTICS)

static const loom_pass_info_t loom_cfg_converge_pass_info_storage = {
    .name = IREE_SVL("cfg-converge"),
    .description =
        IREE_SVL("Converge overlapping CFG decisions and loop frontiers."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_cfg_converge_statistics_layout,
};

const loom_pass_info_t* loom_cfg_converge_pass_info(void) {
  return &loom_cfg_converge_pass_info_storage;
}

typedef struct loom_cfg_converge_capture_t {
  // Original scalar live-in whose definition ceases to dominate its users.
  loom_value_id_t value;
  // Join argument replacing only affected ordinary uses.
  loom_value_id_t argument;
  // Definition's block index in the immutable pre-edit snapshot.
  uint16_t definition;
  // Successor dominance subtrees containing affected users, one bit per arm.
  uint8_t arms;
} loom_cfg_converge_capture_t;

typedef struct loom_cfg_converge_replacement_t {
  // Ordinary use recorded before block insertion changes dense indices.
  loom_use_t use;
  // Index of the capture supplying this operand's join argument.
  iree_host_size_t capture;
} loom_cfg_converge_replacement_t;

typedef struct loom_cfg_converge_plan_t {
  // Module containing the selected decision.
  loom_module_t* module;
  // Immutable graph, dominance and postdominance used during preflight.
  const loom_value_fact_cfg_region_t* facts;
  // Scratch storage released after completing or declining this candidate.
  iree_arena_allocator_t* arena;
  // Function-local value ordinals for deduplicating captures.
  loom_local_value_domain_t domain;
  // Conditional terminator whose decision moves to the join.
  loom_op_t* branch;
  // Original decision block index.
  uint16_t source;
  // Original true and false successor indices.
  uint16_t destinations[2];
  // Immediate dominator of the new join, derived from shared successors.
  uint16_t parent;
  // External incoming edges redirected through constant predicate payloads.
  loom_cfg_edge_index_t* incoming;
  // Number of populated incoming edges.
  iree_host_size_t incoming_count;
  // Captures in join argument order, owned by arena.
  loom_cfg_converge_capture_t* captures;
  // Number of populated captures.
  iree_host_size_t capture_count;
  // Capture index per local value ordinal, or UINT32_MAX when absent.
  uint32_t* capture_indices;
  // Exact ordinary operands that need substitution after inserting the join.
  loom_cfg_converge_replacement_t* replacements;
  // Number of populated replacement operands.
  iree_host_size_t replacement_count;
  // Reserved replacement operand capacity.
  iree_host_size_t replacement_capacity;
  // Whether a direct true continuation needs its own single-entry arm.
  bool isolate_true_passthrough;
  // Successor ordinal whose subtree is being inspected by the reference walk.
  uint8_t arm;
  // False when a capture cannot travel through an ordinary scalar argument.
  bool representable;
} loom_cfg_converge_plan_t;

// Returns the semantic continuation shared by every supported exit from a
// natural loop. Direct destinations are retained when they agree; otherwise
// the loop header's immediate postdominator names the continuation beyond the
// exit regions.
static uint16_t loom_cfg_converge_loop_continuation(
    const loom_value_fact_cfg_region_t* facts,
    const loom_cfg_natural_loop_t* loop) {
  if (loop->continuation_index != LOOM_CFG_LOOP_CONTINUATION_NONE) {
    return loop->continuation_index;
  }
  if (!facts->control_structure.postdominance.available) {
    return LOOM_CFG_LOOP_CONTINUATION_NONE;
  }
  uint32_t continuation =
      facts->control_structure.postdominance.nodes[loop->header_index]
          .immediate_postdominator;
  return continuation < facts->graph.block_count
             ? (uint16_t)continuation
             : LOOM_CFG_LOOP_CONTINUATION_NONE;
}

// One payload-preserving gateway planned against an immutable graph snapshot.
typedef struct loom_cfg_converge_frontier_t {
  // Next disjoint frontier in natural-loop order.
  struct loom_cfg_converge_frontier_t* next;
  // Original continuation supplying the gateway signature and live position.
  loom_block_t* continuation;
  // Original graph edges redirected together; owned by the plan arena.
  loom_cfg_edge_index_span_t edges;
} loom_cfg_converge_frontier_t;

// Plans disjoint loop exit frontiers before inserting any gateway. Each gateway
// preserves every path and payload between original blocks, so these plans
// preserve each other's dominance, reachability, and loop membership. Shared
// edges wait for the next snapshot instead of being redirected twice.
static iree_status_t loom_cfg_converge_plan_loop_exit_frontiers(
    const loom_value_fact_cfg_region_t* facts,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* plan_arena,
    loom_cfg_converge_frontier_t** out_frontiers) {
  *out_frontiers = NULL;
  const loom_cfg_graph_t* graph = &facts->graph;
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  if (!loops->reducible) {
    return iree_ok_status();
  }

  uint8_t* claimed_edges = NULL;
  loom_cfg_converge_frontier_t** next = out_frontiers;
  for (iree_host_size_t loop_index = 0; loop_index < loops->loop_count;
       ++loop_index) {
    const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
    if (loop->exits.count == 0 ||
        loop->direct_exit_count != loop->exits.count) {
      continue;
    }
    const uint16_t continuation =
        loom_cfg_converge_loop_continuation(facts, loop);
    if (continuation == LOOM_CFG_LOOP_CONTINUATION_NONE ||
        loom_cfg_loop_nest_contains(loops, (uint16_t)loop_index,
                                    continuation)) {
      continue;
    }

    // A loop header that dominates its continuation already owns the complete
    // frontier. This is the ordinary-loop fast path: no edge walk or transient
    // allocation is needed unless an enclosing decision bypasses the loop.
    if (loom_cfg_dominance_block_dominates(&facts->dominance,
                                           loop->header_index, continuation)) {
      continue;
    }

    iree_arena_reset(scratch_arena);
    loom_cfg_edge_index_t* direct_exits = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, loop->direct_exit_count, sizeof(*direct_exits),
        (void**)&direct_exits));
    iree_host_size_t direct_exit_count = 0;
    for (iree_host_size_t edge_index = 0; edge_index < graph->edge_count;
         ++edge_index) {
      const loom_cfg_edge_info_t* edge = &graph->edges[edge_index];
      if (!graph->blocks[edge->source_block_index].reachable ||
          loom_cfg_loop_nest_innermost(loops, edge->source_block_index) !=
              loop_index ||
          loom_cfg_loop_nest_contains(loops, (uint16_t)loop_index,
                                      edge->target_block_index)) {
        continue;
      }
      direct_exits[direct_exit_count++] = (loom_cfg_edge_index_t)edge_index;
    }
    IREE_ASSERT_EQ(direct_exit_count, loop->direct_exit_count);

    uint8_t* visited_targets = NULL;
    uint8_t* visited_edges = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, graph->block_count, sizeof(*visited_targets),
        (void**)&visited_targets));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, graph->edge_count, sizeof(*visited_edges),
        (void**)&visited_edges));
    memset(visited_targets, 0, graph->block_count * sizeof(*visited_targets));
    memset(visited_edges, 0, graph->edge_count * sizeof(*visited_edges));

    iree_host_size_t frontier_capacity = 0;
    bool supported = true;
    for (iree_host_size_t i = 0; i < direct_exit_count && supported; ++i) {
      const loom_cfg_edge_info_t* edge = &graph->edges[direct_exits[i]];
      const uint16_t target = edge->target_block_index;
      if (target == continuation) {
        ++frontier_capacity;
        continue;
      }
      const loom_cfg_region_t* region = &facts->regions.blocks[target];
      // An externally entered exit region would leave the new gateway shared
      // too, so it cannot isolate this loop's frontier.
      if (visited_targets[target] ||
          !loom_cfg_dominance_block_dominates(&facts->dominance,
                                              loop->header_index, target) ||
          region->continuation_index != continuation ||
          region->exit_edges.count == 0) {
        supported = false;
        break;
      }
      visited_targets[target] = 1;
      frontier_capacity += region->exit_edges.count;
    }
    if (!supported || frontier_capacity == 0) {
      continue;
    }

    loom_cfg_edge_index_t* frontier = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, frontier_capacity,
                                  sizeof(*frontier), (void**)&frontier));
    iree_host_size_t frontier_count = 0;
    for (iree_host_size_t i = 0; i < direct_exit_count && supported; ++i) {
      const loom_cfg_edge_info_t* exit = &graph->edges[direct_exits[i]];
      loom_cfg_edge_index_span_t edges =
          exit->target_block_index == continuation
              ? (loom_cfg_edge_index_span_t){
                    .values = &direct_exits[i],
                    .count = 1,
                }
              : facts->regions.blocks[exit->target_block_index].exit_edges;
      for (iree_host_size_t j = 0; j < edges.count; ++j) {
        const loom_cfg_edge_index_t edge_index = edges.values[j];
        const loom_cfg_edge_info_t* edge = &graph->edges[edge_index];
        if (visited_edges[edge_index] ||
            edge->target_block_index != continuation) {
          supported = false;
          break;
        }
        visited_edges[edge_index] = 1;
        frontier[frontier_count++] = edge_index;
      }
    }
    if (!supported || frontier_count != frontier_capacity) {
      continue;
    }

    if (!claimed_edges) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          plan_arena, graph->edge_count, sizeof(*claimed_edges),
          (void**)&claimed_edges));
      memset(claimed_edges, 0, graph->edge_count * sizeof(*claimed_edges));
    }
    bool overlaps = false;
    for (iree_host_size_t i = 0; i < frontier_count; ++i) {
      overlaps |= claimed_edges[frontier[i]] != 0;
    }
    if (overlaps) {
      continue;
    }
    loom_cfg_converge_frontier_t* plan = NULL;
    loom_cfg_edge_index_t* edges = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(plan_arena, sizeof(*plan), (void**)&plan));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan_arena, frontier_count, sizeof(*edges), (void**)&edges));
    memcpy(edges, frontier, frontier_count * sizeof(*edges));
    for (iree_host_size_t i = 0; i < frontier_count; ++i) {
      claimed_edges[frontier[i]] = 1;
    }
    *plan = (loom_cfg_converge_frontier_t){
        .continuation = (loom_block_t*)graph->blocks[continuation].block,
        .edges = {.values = edges, .count = frontier_count},
    };
    *next = plan;
    next = &plan->next;
  }
  return iree_ok_status();
}

// Isolates a loop's result-producing exits from an enclosing decision sharing
// its continuation. The gateway retains the continuation's exact signature, so
// each control owner receives a disjoint frontier with unchanged exit payloads.
static iree_status_t loom_cfg_converge_apply_loop_exit_frontier(
    loom_module_t* module, loom_region_t* body, loom_rewriter_t* rewriter,
    const loom_cfg_graph_t* graph,
    const loom_cfg_converge_frontier_t* frontier) {
  loom_block_t* continuation_block = frontier->continuation;
  loom_block_t* gateway = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      module, body, continuation_block->region_index, &gateway));
  loom_builder_t* builder = &rewriter->builder;
  builder->ip.parent_op =
      graph->edges[frontier->edges.values[0]].terminator->parent_op;
  loom_builder_set_block(builder, gateway);
  for (uint16_t i = 0; i < continuation_block->arg_count; ++i) {
    loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, gateway,
        loom_module_value_type(module, continuation_block->arg_ids[i]),
        &argument));
  }
  IREE_RETURN_IF_ERROR(loom_module_assign_value_types(
      module, continuation_block->arg_ids, gateway->arg_ids,
      continuation_block->arg_count));
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_br_build(
      builder, continuation_block, gateway->arg_ids, gateway->arg_count,
      graph->edges[frontier->edges.values[0]].terminator->location, &branch));
  for (iree_host_size_t i = 0; i < frontier->edges.count; ++i) {
    const loom_cfg_edge_info_t* edge = &graph->edges[frontier->edges.values[i]];
    loom_op_successors(edge->terminator)[edge->successor_index] = gateway;
  }

  return iree_ok_status();
}

static iree_status_t loom_cfg_converge_isolate_loop_exit_frontiers(
    loom_pass_t* pass, loom_module_t* module, loom_region_t* body,
    loom_rewriter_t* rewriter, iree_arena_allocator_t* arena,
    const loom_value_fact_cfg_region_t* facts, bool* out_changed) {
  *out_changed = false;
  iree_arena_reset(arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_cfg_converge_frontier_t* frontiers = NULL;
  iree_status_t status = loom_cfg_converge_plan_loop_exit_frontiers(
      facts, &scratch_arena, arena, &frontiers);
  iree_arena_deinitialize(&scratch_arena);
  for (const loom_cfg_converge_frontier_t* frontier = frontiers;
       frontier && iree_status_is_ok(status); frontier = frontier->next) {
    status = loom_cfg_converge_apply_loop_exit_frontier(
        module, body, rewriter, &facts->graph, frontier);
    if (iree_status_is_ok(status)) {
      ++loom_cfg_converge_statistics(pass)->loop_exit_frontiers_isolated;
      *out_changed = true;
    }
  }
  if (iree_status_is_ok(status) && *out_changed) {
    status = loom_rewriter_refresh_cfg_facts(rewriter, body);
  }
  return status;
}

static bool loom_cfg_converge_is_forwarding(const loom_block_t* block,
                                            const loom_block_t* continuation) {
  return block->op_count == 1 && loom_cfg_br_isa(block->last_op) &&
         loom_cfg_br_dest(block->last_op) == continuation;
}

// Component order and the earliest reachable DFS node provide independent
// exclusion bounds. Neither needs a walk through the successor's region.
static bool loom_cfg_converge_may_reach(const loom_cfg_graph_t* graph,
                                        uint16_t source, uint16_t destination) {
  const loom_cfg_block_info_t* source_info = &graph->blocks[source];
  const loom_cfg_block_info_t* destination_info = &graph->blocks[destination];
  return source_info->component >= destination_info->component &&
         graph->blocks[source_info->reachability_root].preorder <=
             destination_info->preorder;
}

// A decision with an externally shared arm moves behind that arm's incoming
// alternatives. Retained reachability bounds prove the redirected edges cannot
// create a cycle. A false forwarding passthrough already has the canonical
// conditional body shape and remains untouched. A direct false continuation
// contributes bypasses only after another shared arm requires a join; those
// bypasses can close an enclosing decision. A true passthrough has inverted
// sense and must itself be inspected for external entries.
static iree_status_t loom_cfg_converge_select(loom_cfg_converge_plan_t* plan,
                                              bool* out_selected) {
  *out_selected = false;
  const loom_cfg_graph_t* graph = &plan->facts->graph;
  const loom_cfg_dominance_t* dominance = &plan->facts->dominance;
  const loom_cfg_postdominance_t* postdominance =
      &plan->facts->control_structure.postdominance;
  uint32_t continuation =
      postdominance->nodes[plan->source].immediate_postdominator;
  if (continuation == postdominance->exit_node ||
      plan->destinations[0] == plan->destinations[1]) {
    return iree_ok_status();
  }
  plan->isolate_true_passthrough = plan->destinations[0] == continuation;
  const loom_cfg_loop_nest_t* loops = &plan->facts->loops;
  uint16_t loop_index = loom_cfg_loop_nest_innermost(loops, plan->source);
  if (graph->blocks[plan->source].component_is_cyclic) {
    if (!loops->reducible || loop_index == LOOM_CFG_LOOP_NEST_NONE) {
      return iree_ok_status();
    }
    uint16_t header = loops->loops[loop_index].header_index;
    if (continuation == header ||
        loom_cfg_loop_nest_innermost(loops, (uint16_t)continuation) !=
            loop_index) {
      return iree_ok_status();
    }
    for (uint8_t arm = 0; arm < 2; ++arm) {
      uint16_t entry = plan->destinations[arm];
      // Every path leaving this arm reaches the continuation. Keeping that
      // boundary and both arms inside one loop preserves iteration ownership.
      if (entry == header ||
          loom_cfg_loop_nest_innermost(loops, entry) != loop_index ||
          (entry != continuation &&
           plan->facts->regions.blocks[entry].continuation_index !=
               continuation)) {
        return iree_ok_status();
      }
    }
  }
  iree_host_size_t capacity =
      graph->blocks[plan->destinations[0]].predecessor_count +
      graph->blocks[plan->destinations[1]].predecessor_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, capacity, sizeof(*plan->incoming), (void**)&plan->incoming));
  plan->parent = plan->source;
  // A false continuation contributes upstream bypasses only when a real shared
  // arm already requires a join. Visit the non-continuation first, regardless
  // of predicate sense. True passthroughs are always inspected because their
  // inverted sense cannot form a target masked region while externally shared.
  uint8_t arms[2] = {0, 1};
  if (plan->destinations[0] == continuation) {
    arms[0] = 1;
    arms[1] = 0;
  }
  for (uint8_t order = 0; order < 2; ++order) {
    uint8_t arm = arms[order];
    uint16_t destination = plan->destinations[arm];
    // A dominated successor has no reachable incoming edge outside this
    // decision and needs no predecessor scan.
    if (loom_cfg_dominance_block_dominates(dominance, plan->source,
                                           destination)) {
      continue;
    }
    if (arm == 1 &&
        ((destination == continuation && !plan->incoming_count) ||
         loom_cfg_converge_is_forwarding(graph->blocks[destination].block,
                                         graph->blocks[continuation].block))) {
      continue;
    }
    loom_cfg_edge_index_span_t incoming =
        loom_cfg_graph_predecessor_edges(graph, destination);
    for (iree_host_size_t i = 0; i < incoming.count; ++i) {
      loom_cfg_edge_index_t index = incoming.values[i];
      const loom_cfg_edge_info_t* edge = &graph->edges[index];
      const loom_cfg_block_info_t* source =
          &graph->blocks[edge->source_block_index];
      if (!source->reachable ||
          loom_cfg_dominance_block_dominates(dominance, plan->source,
                                             edge->source_block_index)) {
        continue;
      }
      bool upstream = true;
      if (loop_index != LOOM_CFG_LOOP_NEST_NONE) {
        upstream = loom_cfg_loop_nest_innermost(
                       loops, edge->source_block_index) == loop_index;
        for (uint8_t other_arm = 0; other_arm < 2; ++other_arm) {
          uint16_t entry = plan->destinations[other_arm];
          // The predecessor is outside both arm subtrees, so neither arm can
          // reach it before their shared continuation, even through a backedge.
          if (entry != continuation &&
              loom_cfg_dominance_block_dominates(dominance, entry,
                                                 edge->source_block_index)) {
            upstream = false;
          }
        }
      } else {
        upstream = !loom_cfg_converge_may_reach(graph, plan->destinations[0],
                                                edge->source_block_index) &&
                   !loom_cfg_converge_may_reach(graph, plan->destinations[1],
                                                edge->source_block_index);
      }
      if (!upstream) {
        // Internal arm exits keep their continuation. Only bypasses that can
        // precede both arms participate in the new predicate join.
        if (destination == continuation) {
          continue;
        }
        return iree_ok_status();
      }
      plan->incoming[plan->incoming_count++] = index;
      // The branch directly reaches each successor, so each successor's
      // immediate dominator is an ancestor of the branch. The outermost of
      // these ancestors is the incoming edges' common dominator; no ancestry
      // search or new dominance solve is needed to locate the join's parent.
      uint16_t parent = dominance->immediate_dominators[destination];
      if ((uint16_t)dominance->intervals[parent] <
          (uint16_t)dominance->intervals[plan->parent]) {
        plan->parent = parent;
      }
    }
  }
  *out_selected = plan->incoming_count != 0 &&
                  graph->block_count + plan->incoming_count + 1 <= UINT16_MAX;
  return iree_ok_status();
}

static iree_status_t loom_cfg_converge_capture(loom_value_id_t value_id,
                                               loom_value_id_t type_carrier,
                                               void* user_data) {
  (void)type_carrier;
  loom_cfg_converge_plan_t* plan = user_data;
  if (!plan->representable) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(plan->module, value_id);
  const loom_block_t* definition = NULL;
  if (loom_value_is_block_arg(value)) {
    definition = loom_value_def_block(value);
  } else {
    const loom_op_t* op = loom_value_def_op(value);
    definition = op ? op->parent_block : NULL;
  }
  if (!definition || definition->parent_region != plan->facts->graph.region) {
    return iree_ok_status();
  }
  const loom_cfg_dominance_t* dominance = &plan->facts->dominance;
  uint16_t index = definition->region_index;
  if (loom_cfg_dominance_block_dominates(
          dominance, plan->destinations[plan->arm], index) ||
      loom_cfg_dominance_block_dominates(dominance, index, plan->parent)) {
    return iree_ok_status();
  }
  // Moving only the branch would break this reference's dominance. Scalar
  // carriers have a defined inactive value; opaque/linear carriers and type
  // or attribute identities require a different ownership-preserving rewrite.
  if (!loom_type_is_scalar(value->type) ||
      loom_module_value_has_type_uses(plan->module, value_id) ||
      loom_value_has_attribute_uses(value) ||
      plan->capture_count == UINT16_MAX - 1) {
    plan->representable = false;
    return iree_ok_status();
  }
  loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(&plan->domain, value_id);
  uint32_t capture = plan->capture_indices[ordinal];
  if (capture == UINT32_MAX) {
    capture = (uint32_t)plan->capture_count++;
    plan->capture_indices[ordinal] = capture;
    plan->captures[capture] = (loom_cfg_converge_capture_t){
        .value = value_id,
        .definition = index,
    };
  }
  plan->captures[capture].arms |= 1u << plan->arm;
  return iree_ok_status();
}

static bool loom_cfg_converge_use_is_affected(
    const loom_cfg_converge_plan_t* plan,
    const loom_cfg_converge_capture_t* capture, loom_use_t use) {
  const loom_op_t* op = loom_use_user_op(use);
  // Nested-region operands inherit the containing CFG block's availability.
  while (op && op->parent_block &&
         op->parent_block->parent_region != plan->facts->graph.region) {
    op = op->parent_op;
  }
  if (!op || !op->parent_block) {
    return false;
  }
  for (uint8_t arm = 0; arm < 2; ++arm) {
    if ((capture->arms & (1u << arm)) &&
        loom_cfg_dominance_block_dominates(&plan->facts->dominance,
                                           plan->destinations[arm],
                                           op->parent_block->region_index)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_cfg_converge_preflight(
    loom_cfg_converge_plan_t* plan) {
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      plan->module, plan->facts->graph.region, plan->arena, &plan->domain));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->domain.value_count, sizeof(*plan->captures),
      (void**)&plan->captures));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->domain.value_count, sizeof(*plan->capture_indices),
      (void**)&plan->capture_indices));
  memset(plan->capture_indices, 0xFF,
         plan->domain.value_count * sizeof(*plan->capture_indices));
  plan->representable = true;
  const loom_cfg_dominance_t* dominance = &plan->facts->dominance;
  for (plan->arm = 0; plan->arm < 2 && plan->representable; ++plan->arm) {
    uint32_t interval = dominance->intervals[plan->destinations[plan->arm]];
    for (uint32_t i = (uint16_t)interval;
         i <= (interval >> 16) && plan->representable; ++i) {
      const loom_block_t* block =
          plan->facts->graph.blocks[dominance->preorder.values[i]].block;
      for (uint16_t j = 0; j < block->arg_count; ++j) {
        loom_type_use_iterator_t dependencies;
        loom_module_value_type_dependencies(
            plan->module, loom_block_arg_id(block, j), &dependencies);
        for (loom_value_id_t dependency =
                 loom_type_dependencies_next(&dependencies);
             dependency != LOOM_VALUE_ID_INVALID;
             dependency = loom_type_dependencies_next(&dependencies)) {
          IREE_RETURN_IF_ERROR(loom_cfg_converge_capture(
              dependency, LOOM_VALUE_ID_INVALID, plan));
        }
      }
      const loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        IREE_RETURN_IF_ERROR(loom_op_walk_subtree_value_refs(
            plan->module, op, loom_cfg_converge_capture, plan));
      }
    }
  }
  for (iree_host_size_t i = 0; i < plan->capture_count && plan->representable;
       ++i) {
    const loom_cfg_converge_capture_t* capture = &plan->captures[i];
    const loom_value_t* value = loom_module_value(plan->module, capture->value);
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t j = 0; j < value->use_count; ++j) {
      if (!loom_cfg_converge_use_is_affected(plan, capture, uses[j])) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          plan->arena, plan->replacement_count, plan->replacement_count + 1,
          sizeof(*plan->replacements), &plan->replacement_capacity,
          (void**)&plan->replacements));
      plan->replacements[plan->replacement_count++] =
          (loom_cfg_converge_replacement_t){.use = uses[j], .capture = i};
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_converge_build_payload(
    loom_cfg_converge_plan_t* plan, loom_builder_t* builder, uint16_t source,
    loom_value_id_t* payload) {
  for (iree_host_size_t i = 0; i < plan->capture_count; ++i) {
    const loom_cfg_converge_capture_t* capture = &plan->captures[i];
    loom_value_id_t value = capture->value;
    if (!loom_cfg_dominance_block_dominates(&plan->facts->dominance,
                                            capture->definition, source)) {
      loom_type_t type = loom_module_value_type(plan->module, value);
      loom_scalar_type_t element = loom_type_element_type(type);
      if (element != LOOM_SCALAR_TYPE_INDEX &&
          element != LOOM_SCALAR_TYPE_OFFSET) {
        loom_op_t* constant = NULL;
        IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
            builder,
            loom_scalar_type_is_float(element) ? loom_attr_f64(0)
                                               : loom_attr_i64(0),
            type, plan->branch->location, &constant));
        value = loom_scalar_constant_result(constant);
      } else {
        IREE_RETURN_IF_ERROR(
            loom_constant_build(builder, loom_value_facts_exact_i64(0), type,
                                plan->branch->location, &value));
      }
    }
    payload[i + 1] = value;
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_converge_apply(loom_cfg_converge_plan_t* plan,
                                             loom_rewriter_t* rewriter) {
  loom_builder_t* builder = &rewriter->builder;
  loom_region_t* body = plan->branch->parent_block->parent_region;
  loom_block_t* destinations[] = {loom_cfg_cond_br_true_dest(plan->branch),
                                  loom_cfg_cond_br_false_dest(plan->branch)};
  loom_value_id_t selector = loom_cfg_cond_br_condition(plan->branch);
  loom_location_id_t location = plan->branch->location;
  builder->ip.parent_op = plan->branch->parent_op;
  loom_op_t* op = NULL;
  if (plan->isolate_true_passthrough) {
    loom_block_t* passthrough = NULL;
    IREE_RETURN_IF_ERROR(loom_region_insert_block(
        plan->module, body, destinations[0]->region_index, &passthrough));
    loom_builder_set_block(builder, passthrough);
    IREE_RETURN_IF_ERROR(loom_cfg_br_build(builder, destinations[0],
                                           /*operands=*/NULL,
                                           /*operand_count=*/0, location, &op));
    destinations[0] = passthrough;
  }
  loom_block_t* join = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      plan->module, body,
      iree_min(destinations[0]->region_index, destinations[1]->region_index),
      &join));
  loom_value_id_t choice = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, join, loom_type_scalar(LOOM_SCALAR_TYPE_I1), &choice));
  for (iree_host_size_t i = 0; i < plan->capture_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, join,
        loom_module_value_type(plan->module, plan->captures[i].value),
        &plan->captures[i].argument));
  }
  loom_builder_set_block(builder, join);
  IREE_RETURN_IF_ERROR(loom_cfg_cond_br_build(builder, choice, destinations[0],
                                              destinations[1], location, &op));
  uint16_t payload_count = (uint16_t)(plan->capture_count + 1);
  loom_value_id_t* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, payload_count, sizeof(*payload), (void**)&payload));
  loom_builder_set_before(builder, plan->branch);
  payload[0] = selector;
  IREE_RETURN_IF_ERROR(
      loom_cfg_converge_build_payload(plan, builder, plan->source, payload));
  IREE_RETURN_IF_ERROR(
      loom_cfg_br_build(builder, join, payload, payload_count, location, &op));
  for (iree_host_size_t i = 0; i < plan->incoming_count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &plan->facts->graph.edges[plan->incoming[i]];
    loom_block_t* forwarding = NULL;
    IREE_RETURN_IF_ERROR(loom_region_insert_block(
        plan->module, body, join->region_index, &forwarding));
    loom_builder_set_block(builder, forwarding);
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder,
        loom_attr_i64(edge->target_block_index == plan->destinations[0]),
        loom_type_scalar(LOOM_SCALAR_TYPE_I1), location, &op));
    payload[0] = loom_scalar_constant_result(op);
    IREE_RETURN_IF_ERROR(loom_cfg_converge_build_payload(
        plan, builder, edge->source_block_index, payload));
    IREE_RETURN_IF_ERROR(loom_cfg_br_build(builder, join, payload,
                                           payload_count, location, &op));
    loom_op_successors(edge->terminator)[edge->successor_index] = forwarding;
  }
  for (iree_host_size_t i = 0; i < plan->replacement_count; ++i) {
    const loom_cfg_converge_replacement_t* replacement = &plan->replacements[i];
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        rewriter, loom_use_user_op(replacement->use),
        loom_use_operand_index(replacement->use),
        plan->captures[replacement->capture].argument));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, plan->branch));
  return loom_rewriter_refresh_cfg_facts(rewriter, body);
}

typedef struct loom_cfg_converge_downstream_t {
  // Next independent frontier retained from the same immutable CFG snapshot.
  struct loom_cfg_converge_downstream_t* next;
  // Shared body and final continuation, in predicate-true/false order.
  loom_block_t* destinations[2];
  // Body predecessors and exits bypassing the body from the selected subtree.
  loom_cfg_edge_index_span_t edges[2];
  // Shared body's original dense index, used to exclude its internal backedges.
  uint16_t entry;
  // Location of the authored branch whose overlap requires convergence.
  loom_location_id_t location;
} loom_cfg_converge_downstream_t;

static bool loom_cfg_converge_downstream_edge(
    const loom_value_fact_cfg_region_t* facts,
    const loom_cfg_converge_downstream_t* plan, uint8_t arm,
    const loom_cfg_edge_info_t* edge) {
  return facts->graph.blocks[edge->source_block_index].reachable &&
         (arm != 0 ||
          !loom_cfg_dominance_block_dominates(&facts->dominance, plan->entry,
                                              edge->source_block_index));
}

// The branch subtree exits to a shared body or bypasses it to the final
// continuation. The body's own subtree exits only to that continuation. Its
// immediate dominator must dominate the branch: adding a bypass-to-body path
// then preserves every original dominance relationship. No live-in capture
// scan is needed, and effects stay in their original blocks.
static bool loom_cfg_converge_downstream_select(
    const loom_value_fact_cfg_region_t* facts, uint16_t source, uint16_t entry,
    uint16_t continuation, loom_cfg_converge_downstream_t* out_plan) {
  const loom_cfg_graph_t* graph = &facts->graph;
  const loom_cfg_dominance_t* dominance = &facts->dominance;
  const loom_cfg_regions_t* regions = &facts->regions;
  if (entry == LOOM_CFG_REGION_CONTINUATION_NONE || entry == continuation ||
      regions->blocks[entry].continuation_index != continuation ||
      loom_cfg_dominance_block_dominates(dominance, source, entry) ||
      loom_cfg_dominance_block_dominates(dominance, entry, source) ||
      loom_cfg_dominance_block_dominates(dominance, source, continuation) ||
      !loom_cfg_dominance_block_dominates(
          dominance, dominance->immediate_dominators[entry], source)) {
    return false;
  }
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  if (graph->blocks[source].component_is_cyclic ||
      graph->blocks[entry].component_is_cyclic) {
    if (!loops->reducible) {
      return false;
    }
    uint16_t loop = loom_cfg_loop_nest_innermost(loops, source);
    if (loom_cfg_loop_nest_innermost(loops, continuation) != loop ||
        (loop != LOOM_CFG_LOOP_NEST_NONE &&
         (loops->loops[loop].header_index == continuation ||
          !loom_cfg_loop_nest_contains(loops, loop, entry)))) {
      return false;
    }
  }
  loom_cfg_edge_index_span_t to_body =
      loom_cfg_regions_edges_to(graph, dominance, regions, source, entry);
  loom_cfg_edge_index_span_t bypass = loom_cfg_regions_edges_to(
      graph, dominance, regions, source, continuation);
  if (to_body.count == 0 || bypass.count == 0 ||
      to_body.count + bypass.count != regions->blocks[source].exit_count) {
    return false;
  }
  loom_block_t* body = (loom_block_t*)graph->blocks[entry].block;
  loom_block_t* end = (loom_block_t*)graph->blocks[continuation].block;
  if ((uint32_t)body->arg_count + end->arg_count + 1 > UINT16_MAX) {
    return false;
  }
  loom_block_t* destinations[] = {body, end};
  for (uint8_t arm = 0; arm < 2; ++arm) {
    for (uint16_t i = 0; i < destinations[arm]->arg_count; ++i) {
      if (!loom_type_has_constant_materializer(loom_module_value_type(
              graph->module, destinations[arm]->arg_ids[i]))) {
        return false;
      }
    }
  }
  *out_plan = (loom_cfg_converge_downstream_t){
      .destinations = {body, end},
      .edges = {loom_cfg_graph_predecessor_edges(graph, entry), bypass},
      .entry = entry,
      .location = graph->blocks[source].block->last_op->location,
  };
  return true;
}

// Plans disjoint redirected edge sets. Each rewrite preserves the old blocks'
// dominance and iteration boundaries, so independent plans can share one fact
// refresh. An overlapping set waits for the next snapshot.
static iree_status_t loom_cfg_converge_plan_downstream(
    loom_rewriter_t* rewriter, const loom_value_fact_cfg_region_t* facts,
    iree_arena_allocator_t* arena, loom_cfg_converge_downstream_t** out_plans,
    iree_host_size_t* out_added_block_count) {
  *out_plans = NULL;
  *out_added_block_count = 0;
  const loom_cfg_graph_t* graph = &facts->graph;
  const loom_cfg_postdominance_t* postdominance =
      &facts->control_structure.postdominance;
  uint8_t* claimed_edges = NULL;
  uint8_t* claimed_entries = NULL;
  iree_host_size_t block_count = graph->block_count;
  loom_cfg_converge_downstream_t** next = out_plans;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < graph->reverse_postorder.count && iree_status_is_ok(status); ++i) {
    uint16_t source = graph->reverse_postorder.values[i];
    const loom_op_t* branch = graph->blocks[source].block->last_op;
    if (!loom_cfg_cond_br_isa(branch)) {
      continue;
    }
    uint32_t continuation =
        postdominance->nodes[source].immediate_postdominator;
    if (continuation == postdominance->exit_node) {
      continue;
    }
    uint16_t destinations[] = {
        loom_cfg_cond_br_true_dest(branch)->region_index,
        loom_cfg_cond_br_false_dest(branch)->region_index};
    for (uint8_t arm = 0; arm < 2 && iree_status_is_ok(status); ++arm) {
      uint16_t entry =
          facts->regions.blocks[destinations[arm]].continuation_index;
      // Immediate shared successors belong to ordinary decision factoring.
      if (entry == destinations[0] || entry == destinations[1] ||
          entry == LOOM_CFG_REGION_CONTINUATION_NONE ||
          (claimed_entries && claimed_entries[entry])) {
        continue;
      }
      loom_cfg_converge_downstream_t candidate;
      if (!loom_cfg_converge_downstream_select(
              facts, source, entry, (uint16_t)continuation, &candidate)) {
        continue;
      }
      if (loom_value_facts_is_uniform_at_scope(
              loom_value_fact_control_execution(facts->control, source),
              LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP) &&
          loom_value_facts_is_uniform_at_scope(
              loom_value_fact_table_lookup(rewriter->fact_table,
                                           loom_cfg_cond_br_condition(branch)),
              LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP)) {
        continue;
      }
      iree_host_size_t added_blocks =
          1 + (candidate.destinations[0]->arg_count != 0) +
          (candidate.destinations[1]->arg_count != 0);
      bool supported = true;
      for (uint8_t side = 0; side < 2 && supported; ++side) {
        for (iree_host_size_t j = 0; j < candidate.edges[side].count; ++j) {
          loom_cfg_edge_index_t index = candidate.edges[side].values[j];
          const loom_cfg_edge_info_t* edge = &graph->edges[index];
          if (!loom_cfg_converge_downstream_edge(facts, &candidate, side,
                                                 edge)) {
            continue;
          }
          if ((claimed_edges && claimed_edges[index]) ||
              (!loom_cfg_br_isa(edge->terminator) &&
               !loom_cfg_cond_br_isa(edge->terminator))) {
            supported = false;
            break;
          }
          added_blocks += loom_cfg_cond_br_isa(edge->terminator);
        }
      }
      if (!supported || block_count + added_blocks > UINT16_MAX) {
        continue;
      }
      if (!claimed_edges) {
        iree_host_size_t claim_count = graph->edge_count + graph->block_count;
        status = iree_arena_allocate_array(
            arena, claim_count, sizeof(*claimed_edges), (void**)&claimed_edges);
        if (iree_status_is_ok(status)) {
          memset(claimed_edges, 0, claim_count * sizeof(*claimed_edges));
          claimed_entries = claimed_edges + graph->edge_count;
        }
      }
      loom_cfg_converge_downstream_t* plan = NULL;
      if (iree_status_is_ok(status)) {
        status = iree_arena_allocate(arena, sizeof(*plan), (void**)&plan);
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      *plan = candidate;
      *next = plan;
      next = &plan->next;
      claimed_entries[entry] = 1;
      block_count += added_blocks;
      for (uint8_t side = 0; side < 2; ++side) {
        for (iree_host_size_t j = 0; j < plan->edges[side].count; ++j) {
          loom_cfg_edge_index_t index = plan->edges[side].values[j];
          if (loom_cfg_converge_downstream_edge(facts, plan, side,
                                                &graph->edges[index])) {
            claimed_edges[index] = 1;
          }
        }
      }
      break;
    }
  }
  *out_added_block_count = block_count - graph->block_count;
  return status;
}

// New blocks are built at the end of the region, then placed before their
// original anchors in one stable counting pass. This preserves layout tie
// breaks without moving and renumbering the suffix after every insertion.
typedef struct loom_cfg_converge_placement_t {
  // Newly appended block to insert into the final table.
  loom_block_t* block;
  // Original block index immediately after this insertion.
  uint16_t anchor;
} loom_cfg_converge_placement_t;

typedef struct loom_cfg_converge_order_t {
  // Per-original-block insertion counts, then next insertion positions.
  uint16_t* positions;
  // New blocks in insertion order, bounded by the selected batch's block count.
  loom_cfg_converge_placement_t* placements;
  // Number of populated placement records.
  iree_host_size_t placement_count;
  // Scratch destination table for the complete reordered region.
  loom_block_t** blocks;
} loom_cfg_converge_order_t;

static void loom_cfg_converge_place_before(loom_cfg_converge_order_t* order,
                                           loom_block_t* block,
                                           const loom_block_t* anchor) {
  order->placements[order->placement_count++] = (loom_cfg_converge_placement_t){
      .block = block, .anchor = anchor->region_index};
  ++order->positions[anchor->region_index];
}

static void loom_cfg_converge_finish_order(loom_cfg_converge_order_t* order,
                                           const loom_cfg_graph_t* graph) {
  iree_host_size_t position = 0;
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    uint16_t count = order->positions[i];
    order->positions[i] = (uint16_t)position;
    position += count;
    order->blocks[position++] = (loom_block_t*)graph->blocks[i].block;
  }
  for (iree_host_size_t i = 0; i < order->placement_count; ++i) {
    const loom_cfg_converge_placement_t* placement = &order->placements[i];
    order->blocks[order->positions[placement->anchor]++] = placement->block;
  }
  loom_region_t* region = (loom_region_t*)graph->region;
  for (iree_host_size_t i = 0; i < region->block_count; ++i) {
    region->blocks[i] = order->blocks[i];
    region->blocks[i]->region_index = (uint16_t)i;
  }
}

// Redirects one original edge with its selected tuple and defined inactive
// values for the other tuple. Only conditional edges need an adapter block;
// direct branches retain their original source block and effect ordering.
static iree_status_t loom_cfg_converge_downstream_incoming(
    loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_cfg_converge_downstream_t* plan, uint8_t arm,
    const loom_cfg_edge_info_t* edge, loom_block_t* join,
    const uint16_t* offsets, loom_value_id_t* payload,
    loom_cfg_converge_order_t* order) {
  loom_builder_t* builder = &rewriter->builder;
  loom_op_t* terminator = (loom_op_t*)edge->terminator;
  bool direct = loom_cfg_br_isa(terminator);
  if (direct) {
    loom_builder_set_before(builder, terminator);
  } else {
    loom_block_t* incoming = NULL;
    IREE_RETURN_IF_ERROR(
        loom_region_append_block(module, join->parent_region, &incoming));
    loom_cfg_converge_place_before(order, incoming, plan->destinations[0]);
    loom_op_successors(terminator)[edge->successor_index] = incoming;
    loom_builder_set_block(builder, incoming);
  }
  IREE_RETURN_IF_ERROR(
      loom_constant_build(builder, loom_value_facts_exact_i64(arm == 0),
                          loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                          terminator->location, &payload[0]));
  iree_status_t status = iree_ok_status();
  for (uint8_t side = 0; side < 2 && iree_status_is_ok(status); ++side) {
    for (uint16_t j = 0;
         j < plan->destinations[side]->arg_count && iree_status_is_ok(status);
         ++j) {
      if (side == arm) {
        payload[offsets[side] + j] = loom_op_operands(terminator)[j];
      } else {
        loom_type_t type = loom_module_value_type(
            module, plan->destinations[side]->arg_ids[j]);
        loom_scalar_type_t element = loom_type_element_type(type);
        loom_value_facts_t zero = loom_scalar_type_is_float(element)
                                      ? loom_value_facts_exact_float(element, 0)
                                      : loom_value_facts_exact_i64(0);
        status = loom_constant_build(builder, zero, type, terminator->location,
                                     &payload[offsets[side] + j]);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch = NULL;
    status = loom_cfg_br_build(builder, join, payload, join->arg_count,
                               terminator->location, &branch);
  }
  if (iree_status_is_ok(status) && direct) {
    status = loom_rewriter_erase(rewriter, terminator);
  }
  return status;
}

static iree_status_t loom_cfg_converge_apply_downstream(
    loom_module_t* module, loom_rewriter_t* rewriter,
    const loom_value_fact_cfg_region_t* facts, iree_arena_allocator_t* arena,
    const loom_cfg_converge_downstream_t* plan,
    loom_cfg_converge_order_t* order) {
  loom_region_t* region = plan->destinations[0]->parent_region;
  loom_builder_t* builder = &rewriter->builder;
  builder->ip.parent_op = plan->destinations[0]->first_op->parent_op;
  // Appending keeps original block indices stable and avoids shifting the
  // region pointer table once for every frontier in the batch.
  loom_block_t* join = NULL;
  IREE_RETURN_IF_ERROR(loom_region_append_block(module, region, &join));
  loom_value_id_t choice = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, join, loom_type_scalar(LOOM_SCALAR_TYPE_I1), &choice));
  uint16_t offsets[2] = {1, (uint16_t)(1 + plan->destinations[0]->arg_count)};
  loom_block_t* destinations[2] = {plan->destinations[0],
                                   plan->destinations[1]};
  iree_status_t status = iree_ok_status();
  for (uint8_t arm = 0; arm < 2 && iree_status_is_ok(status); ++arm) {
    loom_block_t* destination = plan->destinations[arm];
    for (uint16_t i = 0;
         i < destination->arg_count && iree_status_is_ok(status); ++i) {
      loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
      status = loom_builder_define_block_arg(
          builder, join,
          loom_module_value_type(module, destination->arg_ids[i]), &argument);
    }
    if (iree_status_is_ok(status) && destination->arg_count != 0) {
      status = loom_region_append_block(module, region, &destinations[arm]);
      if (iree_status_is_ok(status)) {
        loom_builder_set_block(builder, destinations[arm]);
        loom_op_t* forwarding = NULL;
        status = loom_cfg_br_build(
            builder, destination, join->arg_ids + offsets[arm],
            destination->arg_count, plan->location, &forwarding);
        if (arm == 1) {
          loom_cfg_converge_place_before(order, destinations[arm], destination);
        }
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  loom_builder_set_block(builder, join);
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_cond_br_build(builder, choice, destinations[0],
                                              destinations[1], plan->location,
                                              &branch));
  loom_value_id_t* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, join->arg_count, sizeof(*payload), (void**)&payload));
  for (uint8_t arm = 0; arm < 2 && iree_status_is_ok(status); ++arm) {
    for (iree_host_size_t i = 0;
         i < plan->edges[arm].count && iree_status_is_ok(status); ++i) {
      const loom_cfg_edge_info_t* edge =
          &facts->graph.edges[plan->edges[arm].values[i]];
      if (loom_cfg_converge_downstream_edge(facts, plan, arm, edge)) {
        status = loom_cfg_converge_downstream_incoming(
            module, rewriter, plan, arm, edge, join, offsets, payload, order);
      }
    }
  }
  loom_cfg_converge_place_before(order, join, plan->destinations[0]);
  if (destinations[0] != plan->destinations[0]) {
    loom_cfg_converge_place_before(order, destinations[0],
                                   plan->destinations[0]);
  }
  return status;
}

static iree_status_t loom_cfg_converge_factor_downstream(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    iree_arena_allocator_t* arena, const loom_value_fact_cfg_region_t* facts,
    bool* out_changed) {
  *out_changed = false;
  iree_arena_reset(arena);
  loom_cfg_converge_downstream_t* plans = NULL;
  iree_host_size_t added_block_count = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_converge_plan_downstream(
      rewriter, facts, arena, &plans, &added_block_count));
  if (!plans) {
    return iree_ok_status();
  }
  loom_cfg_converge_order_t order = {0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, facts->graph.block_count, sizeof(*order.positions),
      (void**)&order.positions));
  memset(order.positions, 0,
         facts->graph.block_count * sizeof(*order.positions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, added_block_count,
                                                 sizeof(*order.placements),
                                                 (void**)&order.placements));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, facts->graph.block_count + added_block_count,
      sizeof(*order.blocks), (void**)&order.blocks));
  iree_status_t status = iree_ok_status();
  for (const loom_cfg_converge_downstream_t* plan = plans;
       plan && iree_status_is_ok(status); plan = plan->next) {
    status = loom_cfg_converge_apply_downstream(module, rewriter, facts, arena,
                                                plan, &order);
    if (iree_status_is_ok(status)) {
      loom_cfg_converge_statistics_t* statistics =
          loom_cfg_converge_statistics(pass);
      ++statistics->downstream_frontiers_factored;
      statistics->values_carried +=
          plan->destinations[0]->arg_count + plan->destinations[1]->arg_count;
      *out_changed = true;
    }
  }
  if (iree_status_is_ok(status) && *out_changed) {
    loom_cfg_converge_finish_order(&order, &facts->graph);
    status = loom_rewriter_refresh_cfg_facts(
        rewriter, plans->destinations[0]->parent_region);
  }
  return status;
}

static iree_status_t loom_cfg_converge_once(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    iree_arena_allocator_t* arena, const loom_value_fact_cfg_region_t* facts,
    bool* out_changed) {
  *out_changed = false;
  const loom_cfg_graph_t* graph = &facts->graph;
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    uint16_t source = graph->reverse_postorder.values[i];
    const loom_cfg_block_info_t* block = &graph->blocks[source];
    loom_op_t* branch = (loom_op_t*)block->block->last_op;
    if (!loom_cfg_cond_br_isa(branch)) {
      continue;
    }
    // Uniform participation and a uniform selector need no lane convergence.
    // A uniform selector inside varying control still needs factoring: its
    // shared successor can also be reached by lanes in the enclosing bypass.
    if (loom_value_facts_is_uniform_at_scope(
            loom_value_fact_control_execution(facts->control, source),
            LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP) &&
        loom_value_facts_is_uniform_at_scope(
            loom_value_fact_table_lookup(rewriter->fact_table,
                                         loom_cfg_cond_br_condition(branch)),
            LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP)) {
      continue;
    }
    iree_arena_reset(arena);
    loom_cfg_converge_plan_t plan = {
        .module = module,
        .facts = facts,
        .arena = arena,
        .branch = branch,
        .source = source,
        .destinations = {loom_cfg_cond_br_true_dest(branch)->region_index,
                         loom_cfg_cond_br_false_dest(branch)->region_index},
    };
    bool selected = false;
    iree_status_t status = loom_cfg_converge_select(&plan, &selected);
    if (iree_status_is_ok(status) && selected) {
      status = loom_cfg_converge_preflight(&plan);
      if (iree_status_is_ok(status) && plan.representable) {
        status = loom_cfg_converge_apply(&plan, rewriter);
        *out_changed = true;
        loom_cfg_converge_statistics_t* statistics =
            loom_cfg_converge_statistics(pass);
        ++statistics->decisions_factored;
        statistics->values_carried += plan.capture_count;
      }
    }
    loom_local_value_domain_release(&plan.domain);
    if (!iree_status_is_ok(status) || *out_changed) {
      return status;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_converge_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body || body->block_count < 4) {
    return iree_ok_status();
  }
  loom_rewriter_t rewriter = {0};
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_arena_allocator_t arena;
  iree_arena_initialize(pass->arena->block_pool, &arena);
  loom_value_fact_table_t* facts = NULL;
  iree_status_t status = loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(
          function,
          loom_target_function_version_target_facts(pass->function_version)),
      &facts);
  loom_rewriter_attach_value_facts(&rewriter, facts);
  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    const loom_value_fact_cfg_region_t* facts = NULL;
    status = loom_rewriter_propagate_value_facts(&rewriter);
    if (iree_status_is_ok(status)) {
      status = loom_value_fact_table_get_or_build_cfg_region(
          rewriter.fact_table, module, body, &facts);
    }
    if (iree_status_is_ok(status)) {
      status = loom_cfg_converge_isolate_loop_exit_frontiers(
          pass, module, body, &rewriter, &arena, facts, &changed);
    }
    if (iree_status_is_ok(status) && !changed) {
      status = loom_cfg_converge_once(pass, module, &rewriter, &arena, facts,
                                      &changed);
    }
    if (iree_status_is_ok(status) && !changed) {
      status = loom_cfg_converge_factor_downstream(pass, module, &rewriter,
                                                   &arena, facts, &changed);
    }
    any_changed |= changed;
  }
  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (any_changed || !iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  iree_arena_deinitialize(&arena);
  return status;
}
