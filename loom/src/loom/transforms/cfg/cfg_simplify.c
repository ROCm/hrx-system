// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/cfg_simplify.h"

#include <string.h>

#include "loom/analysis/cfg_condition_facts.h"
#include "loom/analysis/cfg_value_identity.h"
#include "loom/analysis/condition_fact_scope.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/transforms/cfg/block_arguments.h"
#include "loom/transforms/cfg/block_forwarding.h"
#include "loom/transforms/cfg/block_fusion.h"
#include "loom/transforms/cfg/branch_folding.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_cfg.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_CFG_SIMPLIFY_STATISTICS(V, statistics_type)                 \
  V(statistics_type, iterations, "iterations",                           \
    "Number of fixed-point simplification iterations.")                  \
  V(statistics_type, branches_folded, "branches-folded",                 \
    "Number of conditional branches folded to direct branches.")         \
  V(statistics_type, edges_forwarded, "edges-forwarded",                 \
    "Number of predecessor edges forwarded through trivial blocks.")     \
  V(statistics_type, blocks_removed, "blocks-removed",                   \
    "Number of unreachable CFG blocks removed.")                         \
  V(statistics_type, block_args_removed, "block-args-removed",           \
    "Number of redundant CFG block arguments removed.")                  \
  V(statistics_type, blocks_fused, "blocks-fused",                       \
    "Number of single-predecessor CFG blocks fused into their "          \
    "predecessors.")                                                     \
  V(statistics_type, duplicate_blocks_merged, "duplicate-blocks-merged", \
    "Number of equivalent CFG blocks merged.")

LOOM_PASS_STATISTICS_DEFINE(loom_cfg_simplify_statistics,
                            loom_cfg_simplify_statistics_t,
                            LOOM_CFG_SIMPLIFY_STATISTICS)

static const loom_pass_info_t loom_cfg_simplify_pass_info_storage = {
    .name = IREE_SVL("cfg-simplify"),
    .description = IREE_SVL("Simplify explicit CFG block structure."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_cfg_simplify_statistics_layout,
};

const loom_pass_info_t* loom_cfg_simplify_pass_info(void) {
  return &loom_cfg_simplify_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Worklist
//===----------------------------------------------------------------------===//

#define LOOM_CFG_SIMPLIFY_INITIAL_REGION_STACK_CAPACITY 16

typedef struct loom_cfg_simplify_region_stack_t {
  // Regions still waiting for simplification.
  loom_region_t** regions;
  // Number of queued regions.
  iree_host_size_t count;
  // Capacity of regions.
  iree_host_size_t capacity;
} loom_cfg_simplify_region_stack_t;

typedef struct loom_cfg_simplify_state_t {
  // Active pass instance for statistics and scratch allocation.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_cfg_simplify_statistics_t* statistics;
  // Module being rewritten.
  loom_module_t* module;
  // Rewriter used for use-list preserving IR edits.
  loom_rewriter_t* rewriter;
  // Per-iteration analysis and temporary allocation arena.
  iree_arena_allocator_t* analysis_arena;
  // Value facts maintained across rewrites by the shared rewriter.
  const loom_value_fact_table_t* fact_table;
  // Dominance computed for the current fixed-point iteration.
  const loom_dominance_info_t* dominance;
  // Function-local value domain acquired for the current iteration.
  loom_local_value_domain_t value_domain;
  // Exact CFG forwarding identities accumulated for the current iteration.
  loom_cfg_value_identity_table_t value_identities;
  // Reusable condition traversal state for the current fixed-point iteration.
  loom_condition_query_t condition_query;
  // DFS stack for nested regions.
  loom_cfg_simplify_region_stack_t region_stack;
} loom_cfg_simplify_state_t;

static iree_status_t loom_cfg_simplify_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_cfg_simplify_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_CFG_SIMPLIFY_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(*stack->regions), (void**)&stack->regions);
}

static iree_status_t loom_cfg_simplify_region_stack_push(
    iree_arena_allocator_t* arena, loom_cfg_simplify_region_stack_t* stack,
    loom_region_t* region) {
  if (!region) {
    return iree_ok_status();
  }
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(*stack->regions),
        &stack->capacity, (void**)&stack->regions));
  }
  stack->regions[stack->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_cfg_simplify_region_stack_pop(
    loom_cfg_simplify_region_stack_t* stack) {
  return stack->count == 0 ? NULL : stack->regions[--stack->count];
}

static iree_status_t loom_cfg_simplify_push_child_regions(
    loom_cfg_simplify_state_t* state, const loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_region_stack_push(
        state->pass->arena, &state->region_stack, regions[region_index]));
  }
  return iree_ok_status();
}

static bool loom_cfg_simplify_region_is_cfg_shaped(loom_region_t* region) {
  if (!region) {
    return false;
  }
  if (region->block_count > 1) {
    return true;
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (op->successor_count > 0) {
        return true;
      }
    }
  }
  return false;
}

static iree_status_t loom_cfg_simplify_mark_cfg_regions(
    loom_region_t* root_region, iree_arena_allocator_t* arena) {
  if (!root_region) {
    return iree_ok_status();
  }

  iree_host_size_t stack_capacity =
      LOOM_CFG_SIMPLIFY_INITIAL_REGION_STACK_CAPACITY;
  loom_region_t** stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, stack_capacity, sizeof(*stack), (void**)&stack));
  iree_host_size_t stack_count = 0;
  stack[stack_count++] = root_region;

  while (stack_count > 0) {
    loom_region_t* region = stack[--stack_count];
    if (loom_cfg_simplify_region_is_cfg_shaped(region)) {
      region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    } else {
      region->flags &= ~LOOM_REGION_INSTANCE_FLAG_CFG;
    }

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t region_index = 0; region_index < op->region_count;
             ++region_index) {
          if (!regions[region_index]) {
            continue;
          }
          if (stack_count >= stack_capacity) {
            IREE_RETURN_IF_ERROR(iree_arena_grow_array(
                arena, stack_count, stack_count + 1, sizeof(*stack),
                &stack_capacity, (void**)&stack));
          }
          stack[stack_count++] = regions[region_index];
        }
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Branch rewriting
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_replace_br(
    loom_cfg_simplify_state_t* state, loom_op_t* old_br, loom_block_t* dest,
    const loom_value_id_t* args, iree_host_size_t arg_count) {
  loom_builder_ip_t saved_ip = loom_builder_save(&state->rewriter->builder);
  loom_builder_set_before(&state->rewriter->builder, old_br);
  loom_op_t* new_br = NULL;
  iree_status_t status =
      loom_cfg_br_build(&state->rewriter->builder, dest, args, arg_count,
                        old_br->location, &new_br);
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  return loom_rewriter_erase(state->rewriter, old_br);
}

static iree_status_t loom_cfg_simplify_replace_direct_br(
    loom_cfg_simplify_state_t* state, loom_op_t* old_br, loom_block_t* dest,
    const loom_value_id_t* args, iree_host_size_t arg_count) {
  if (loom_cfg_br_isa(old_br)) {
    return loom_cfg_simplify_replace_br(state, old_br, dest, args, arg_count);
  }

  if (!loom_low_br_isa(old_br)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cfg-simplify direct branch rewrite expected a "
                            "cfg.br or low.br");
  }

  loom_builder_ip_t saved_ip = loom_builder_save(&state->rewriter->builder);
  loom_builder_set_before(&state->rewriter->builder, old_br);
  loom_op_t* new_br = NULL;
  iree_status_t status =
      loom_low_br_build(&state->rewriter->builder, dest, args, arg_count,
                        old_br->location, &new_br);
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  return loom_rewriter_erase(state->rewriter, old_br);
}

//===----------------------------------------------------------------------===//
// Direct branch access
//===----------------------------------------------------------------------===//

static bool loom_cfg_simplify_direct_branch(const loom_op_t* op,
                                            loom_block_t** out_dest,
                                            loom_value_slice_t* out_args) {
  if (loom_cfg_br_isa(op)) {
    *out_dest = loom_cfg_br_dest(op);
    *out_args = loom_cfg_br_args(op);
    return true;
  }
  if (loom_low_br_isa(op)) {
    *out_dest = loom_low_br_dest(op);
    *out_args = loom_low_br_args(op);
    return true;
  }
  *out_dest = NULL;
  *out_args = (loom_value_slice_t){0};
  return false;
}

//===----------------------------------------------------------------------===//
// Unreachable block removal
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_remove_unreachable_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed || graph->block_count <= 1) {
    return iree_ok_status();
  }

  bool* remove_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->analysis_arena, graph->block_count, sizeof(*remove_blocks),
      (void**)&remove_blocks));
  memset(remove_blocks, 0, graph->block_count * sizeof(*remove_blocks));

  bool any_removed = false;
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (loom_cfg_graph_block_is_reachable(graph, block_index)) {
      continue;
    }
    remove_blocks[block_index] = true;
    any_removed = true;
  }
  if (!any_removed) {
    return iree_ok_status();
  }

  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_region_remove_blocks(
      state->module, (loom_region_t*)graph->region, remove_blocks,
      graph->block_count, state->analysis_arena, &removed_count));
  state->statistics->blocks_removed += removed_count;
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  *out_changed = removed_count != 0;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Path-sensitive branch facts
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_entry_facts_prove_bool(
    loom_cfg_simplify_state_t* state,
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view, loom_value_id_t condition,
    bool* out_value, bool* out_proven) {
  loom_condition_fact_scope_t scope;
  const loom_condition_fact_scope_t* scope_ptr = NULL;
  if (view != NULL) {
    loom_condition_fact_scope_initialize_indexed(NULL, table, view, &scope);
    scope_ptr = &scope;
  }
  return loom_condition_fact_scope_proves_condition(
      &state->condition_query, state->fact_table, scope_ptr, condition,
      out_value, out_proven);
}

static bool loom_cfg_simplify_value_uses_stay_in_block(
    const loom_cfg_simplify_state_t* state, loom_value_id_t value_id,
    const loom_block_t* block) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count ||
      loom_module_value_has_type_uses(state->module, value_id)) {
    return false;
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (loom_value_has_attribute_uses(value)) {
    return false;
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    loom_op_t* user = loom_use_user_op(*use);
    if (!user || user->parent_block != block) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_can_skip_block_prefix(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    const loom_op_t* terminator) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (!loom_cfg_simplify_value_uses_stay_in_block(
            state, loom_block_arg_id(block, i), block)) {
      return false;
    }
  }

  for (const loom_op_t* op = block->first_op; op && op != terminator;
       op = op->next_op) {
    loom_trait_flags_t traits = loom_op_effective_traits(state->module, op);
    if (!iree_all_bits_set(traits, LOOM_TRAIT_PURE) ||
        loom_traits_are_convergent(traits) || op->successor_count != 0 ||
        op->region_count != 0) {
      return false;
    }
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (!loom_cfg_simplify_value_uses_stay_in_block(state, results[i],
                                                      block)) {
        return false;
      }
    }
  }
  return true;
}

static iree_status_t loom_cfg_simplify_thread_fact_known_branches(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_condition_relation_table_t* table, bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }
  // Prove every bypass against the original CFG before changing any edge.
  // Destinations stay alive throughout application; each redirect skips only
  // pure local definitions, even when another edit bypasses its source block.
  uint16_t* destinations = NULL;
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) {
      continue;
    }
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    loom_op_t* terminator = block ? block->last_op : NULL;
    if (!terminator || !loom_cfg_cond_br_isa(terminator) ||
        !loom_cfg_simplify_can_skip_block_prefix(state, block, terminator)) {
      continue;
    }

    const loom_cfg_edge_index_span_t predecessor_edges =
        loom_cfg_graph_predecessor_edges(graph, block_index);
    for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
      const loom_cfg_edge_index_t edge_index = predecessor_edges.values[i];
      const loom_cfg_edge_info_t* predecessor_edge =
          loom_cfg_graph_edge(graph, edge_index);
      const loom_op_t* predecessor_terminator = predecessor_edge->terminator;
      if (!loom_cfg_br_isa(predecessor_terminator) &&
          (!loom_cfg_cond_br_isa(predecessor_terminator) ||
           block->arg_count != 0)) {
        continue;
      }
      const uint16_t predecessor_index = predecessor_edge->source_block_index;
      if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index) ||
          predecessor_index == block_index) {
        continue;
      }
      loom_block_t* predecessor =
          (loom_block_t*)graph->blocks[predecessor_index].block;
      if (!predecessor || !predecessor->last_op) {
        continue;
      }
      if (block->first_op &&
          loom_dominates_op(state->dominance, block->first_op,
                            predecessor->last_op)) {
        continue;
      }

      bool condition = false;
      bool condition_proven = false;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_entry_facts_prove_bool(
          state, table,
          loom_cfg_condition_relation_table_edge(table, edge_index),
          loom_cfg_cond_br_condition(terminator), &condition,
          &condition_proven));
      if (!condition_proven) {
        continue;
      }

      loom_block_t* new_dest = condition
                                   ? loom_cfg_cond_br_true_dest(terminator)
                                   : loom_cfg_cond_br_false_dest(terminator);
      if (new_dest == block || new_dest->arg_count != 0) {
        continue;
      }
      if (!destinations) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            state->analysis_arena, graph->edge_count, sizeof(*destinations),
            (void**)&destinations));
        memset(destinations, 0xFF, graph->edge_count * sizeof(*destinations));
      }
      destinations[edge_index] = new_dest->region_index;
    }
  }
  if (!destinations) {
    return iree_ok_status();
  }

  // Original edges now identify edit locations only. A direct branch owns one
  // slot and may be rebuilt to drop its payload; conditional branches remain
  // intact so both successor slots can be updated. Queued users run only after
  // the driver refreshes the completed CFG edit.
  for (iree_host_size_t i = 0; i < graph->edge_count; ++i) {
    if (destinations[i] == UINT16_MAX) {
      continue;
    }
    const loom_cfg_edge_info_t* edge = &graph->edges[i];
    loom_op_t* terminator = (loom_op_t*)edge->terminator;
    loom_block_t* destination =
        (loom_block_t*)graph->blocks[destinations[i]].block;
    if (loom_cfg_br_isa(terminator) && terminator->operand_count != 0) {
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_replace_br(state, terminator,
                                                        destination, NULL, 0));
    } else {
      loom_op_successors(terminator)[edge->successor_index] = destination;
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_to_worklist(state->rewriter, terminator));
      state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    }
    ++state->statistics->edges_forwarded;
  }
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_cfg_simplify_is_i1_value(const loom_module_t* module,
                                          loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static bool loom_cfg_simplify_can_replace_with_constant(
    const loom_cfg_simplify_state_t* state, const loom_op_t* op,
    loom_value_id_t result) {
  if (op->result_count != 1 ||
      !loom_cfg_simplify_is_i1_value(state->module, result)) {
    return false;
  }
  loom_trait_flags_t traits = loom_op_effective_traits(state->module, op);
  return iree_all_bits_set(traits, LOOM_TRAIT_PURE) &&
         !iree_any_bit_set(traits, LOOM_TRAIT_CONSTANT_LIKE) &&
         !loom_traits_are_convergent(traits) && op->successor_count == 0 &&
         op->region_count == 0;
}

static iree_status_t loom_cfg_simplify_replace_with_bool_constant(
    loom_cfg_simplify_state_t* state, loom_op_t* op, bool value) {
  loom_value_id_t result = loom_op_const_results(op)[0];
  loom_type_t result_type = loom_module_value_type(state->module, result);

  loom_builder_ip_t saved_ip = loom_builder_save(&state->rewriter->builder);
  loom_builder_set_before(&state->rewriter->builder, op);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(state->rewriter);

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  iree_status_t status = loom_constant_build(
      &state->rewriter->builder, loom_value_facts_exact_i64(value ? 1 : 0),
      result_type, op->location, &replacement);
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_preserve_result_names_on_new_values(
        state->rewriter, op, &replacement, 1, value_checkpoint);
  }
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  return loom_rewriter_replace_all_uses_and_erase(state->rewriter, op,
                                                  &replacement, 1);
}

static iree_status_t loom_cfg_simplify_fold_path_sensitive_i1_ops(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_condition_relation_table_t* table, bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }
  // Exact substitutions preserve CFG shape and existing value meanings. The
  // rewriter queues affected users without executing them, so retained entry
  // relations stay valid throughout this scan. New constants have facts, and
  // the condition query extends its value domain on demand.
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) {
      continue;
    }
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (!block) {
      continue;
    }

    loom_op_t* op = block->first_op;
    while (op) {
      loom_op_t* next_op = op->next_op;
      if (op->result_count == 1) {
        loom_value_id_t result = loom_op_const_results(op)[0];
        if (loom_cfg_simplify_can_replace_with_constant(state, op, result)) {
          bool value = false;
          bool proven = false;
          IREE_RETURN_IF_ERROR(loom_cfg_simplify_entry_facts_prove_bool(
              state, table,
              loom_cfg_condition_relation_table_block(table, block_index),
              result, &value, &proven));
          if (proven) {
            IREE_RETURN_IF_ERROR(
                loom_cfg_simplify_replace_with_bool_constant(state, op, value));
            *out_changed = true;
          }
        }
      }
      op = next_op;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Duplicate block merging
//===----------------------------------------------------------------------===//

#define LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_OPS 8
#define LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES 32

typedef struct loom_cfg_simplify_block_hash_entry_t {
  // Structural fingerprint used to reject non-equivalent blocks cheaply.
  uint32_t fingerprint;
  // Dense graph block index, or zero when the entry is empty.
  uint16_t block_index;
} loom_cfg_simplify_block_hash_entry_t;

static_assert(sizeof(loom_cfg_simplify_block_hash_entry_t) == 8,
              "block hash entries must remain compact");

typedef struct loom_cfg_simplify_block_hash_table_t {
  // Open-addressed entries with capacity slots.
  loom_cfg_simplify_block_hash_entry_t* entries;
  // Power-of-two entry capacity.
  iree_host_size_t capacity;
} loom_cfg_simplify_block_hash_table_t;

typedef struct loom_cfg_simplify_value_map_t {
  // Original block-local definitions in the equivalence comparison.
  loom_value_id_t source_values[LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES];
  // Corresponding definitions in the surviving candidate block.
  loom_value_id_t target_values[LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES];
  // Number of established source-to-target value pairs.
  iree_host_size_t count;
} loom_cfg_simplify_value_map_t;

typedef struct loom_cfg_simplify_local_values_t {
  // Block-local values in argument/result definition order.
  loom_value_id_t values[LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES];
  // Number of block-local values recorded in values.
  iree_host_size_t count;
} loom_cfg_simplify_local_values_t;

// Extends an FNV-1a fingerprint with one byte span.
static uint32_t loom_cfg_simplify_hash_bytes(const void* data,
                                             iree_host_size_t length,
                                             uint32_t fingerprint) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    fingerprint ^= bytes[i];
    fingerprint *= 16777619u;
  }
  return fingerprint;
}

static uint32_t loom_cfg_simplify_hash_u32(uint32_t value,
                                           uint32_t fingerprint) {
  return loom_cfg_simplify_hash_bytes(&value, sizeof(value), fingerprint);
}

static iree_status_t loom_cfg_simplify_block_hash_table_initialize(
    iree_arena_allocator_t* arena, iree_host_size_t block_count,
    loom_cfg_simplify_block_hash_table_t* table) {
  iree_host_size_t minimum_capacity =
      iree_max((block_count * 4 + 2) / 3, (iree_host_size_t)2);
  table->capacity = iree_host_size_next_power_of_two(minimum_capacity);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, table->capacity,
                                                 sizeof(*table->entries),
                                                 (void**)&table->entries));
  memset(table->entries, 0, table->capacity * sizeof(*table->entries));
  return iree_ok_status();
}

static bool loom_cfg_simplify_value_map_append(
    loom_cfg_simplify_value_map_t* map, loom_value_id_t source_value,
    loom_value_id_t target_value) {
  if (source_value == LOOM_VALUE_ID_INVALID ||
      target_value == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  for (iree_host_size_t i = 0; i < map->count; ++i) {
    if (map->source_values[i] == source_value) {
      return map->target_values[i] == target_value;
    }
    if (map->target_values[i] == target_value) {
      return false;
    }
  }
  if (map->count >= LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES) {
    return false;
  }
  map->source_values[map->count] = source_value;
  map->target_values[map->count] = target_value;
  ++map->count;
  return true;
}

static loom_value_id_t loom_cfg_simplify_value_map_lookup(
    const loom_cfg_simplify_value_map_t* map, loom_value_id_t source_value) {
  for (iree_host_size_t i = 0; i < map->count; ++i) {
    if (map->source_values[i] == source_value) {
      return map->target_values[i];
    }
  }
  return source_value;
}

static bool loom_cfg_simplify_values_equal_after_map(
    const loom_cfg_simplify_value_map_t* map, loom_value_id_t source_value,
    loom_value_id_t target_value) {
  return loom_cfg_simplify_value_map_lookup(map, source_value) == target_value;
}

static bool loom_cfg_simplify_types_equal_after_map(
    const loom_module_t* module, const loom_cfg_simplify_value_map_t* map,
    loom_type_t source_type, loom_type_t target_type) {
  loom_type_value_remap_t remap = {
      .source_values = map->source_values,
      .target_values = map->target_values,
      .count = (uint16_t)map->count,
  };
  return loom_type_equal_after_value_remap(module, source_type, target_type,
                                           &remap);
}

static bool loom_cfg_simplify_op_is_alpha_mergeable(
    const loom_cfg_simplify_state_t* state, const loom_op_t* op) {
  if (!op || op->region_count != 0 || op->tied_result_count != 0) {
    return false;
  }
  loom_trait_flags_t traits = loom_op_effective_traits(state->module, op);
  if (!iree_all_bits_set(traits, LOOM_TRAIT_PURE)) {
    return false;
  }
  if (loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
      loom_traits_has_unique_identity(traits) ||
      loom_traits_are_convergent(traits) ||
      iree_any_bit_set(traits, LOOM_TRAIT_HINT)) {
    return false;
  }
  return true;
}

static bool loom_cfg_simplify_block_values_stay_in_block(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (!loom_cfg_simplify_value_uses_stay_in_block(
            state, loom_block_arg_id(block, i), block)) {
      return false;
    }
  }
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (!loom_cfg_simplify_value_uses_stay_in_block(state, results[i],
                                                      block)) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_cfg_simplify_is_alpha_merge_candidate(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index) {
  if (block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return false;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  if (!block || block->op_count == 0 ||
      block->op_count > LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_OPS ||
      !loom_cfg_simplify_block_values_stay_in_block(state, block)) {
    return false;
  }
  loom_block_t* branch_dest = NULL;
  loom_value_slice_t branch_args = {0};
  if (block->first_op == block->last_op &&
      loom_cfg_simplify_direct_branch(block->first_op, &branch_dest,
                                      &branch_args)) {
    return false;
  }
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (!loom_cfg_simplify_op_is_alpha_mergeable(state, op)) {
      return false;
    }
  }
  return true;
}

// Acyclic successors finish before their predecessors in DFS postorder, so
// their planned destinations are final. Cyclic successors retain original
// identities: their destinations may still change after this key is inserted.
static uint16_t loom_cfg_simplify_successor_key(const loom_cfg_graph_t* graph,
                                                const uint16_t* destinations,
                                                const loom_block_t* successor) {
  const uint16_t block_index = successor->region_index;
  if (!destinations || graph->blocks[block_index].component_is_cyclic ||
      destinations[block_index] == 0) {
    return block_index;
  }
  return destinations[block_index];
}

static bool loom_cfg_simplify_successors_equal_after_map(
    const loom_cfg_graph_t* graph, const uint16_t* destinations,
    const loom_block_t* source_block, const loom_block_t* target_block,
    const loom_op_t* source_op, const loom_op_t* target_op) {
  loom_block_t* const* source_successors = loom_op_const_successors(source_op);
  loom_block_t* const* target_successors = loom_op_const_successors(target_op);
  for (uint8_t i = 0; i < source_op->successor_count; ++i) {
    loom_block_t* source_successor = source_successors[i];
    loom_block_t* target_successor = target_successors[i];
    if (source_successor == source_block || source_successor == target_block ||
        target_successor == source_block || target_successor == target_block) {
      return false;
    }
    if (source_successor != target_successor &&
        loom_cfg_simplify_successor_key(graph, destinations,
                                        source_successor) !=
            loom_cfg_simplify_successor_key(graph, destinations,
                                            target_successor)) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_attributes_equal(const loom_op_t* source_op,
                                               const loom_op_t* target_op) {
  const loom_attribute_t* source_attrs = loom_op_attrs((loom_op_t*)source_op);
  const loom_attribute_t* target_attrs = loom_op_attrs((loom_op_t*)target_op);
  for (uint8_t i = 0; i < source_op->attribute_count; ++i) {
    if (!loom_attribute_equal(&source_attrs[i], &target_attrs[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_ops_equal_after_map(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const uint16_t* destinations, const loom_block_t* source_block,
    const loom_block_t* target_block, const loom_op_t* source_op,
    const loom_op_t* target_op, loom_cfg_simplify_value_map_t* map) {
  if (source_op->kind != target_op->kind ||
      source_op->operand_count != target_op->operand_count ||
      source_op->result_count != target_op->result_count ||
      source_op->tied_result_count != target_op->tied_result_count ||
      source_op->region_count != target_op->region_count ||
      source_op->successor_count != target_op->successor_count ||
      source_op->attribute_count != target_op->attribute_count ||
      source_op->instance_flags != target_op->instance_flags ||
      !loom_cfg_simplify_op_is_alpha_mergeable(state, source_op) ||
      !loom_cfg_simplify_op_is_alpha_mergeable(state, target_op)) {
    return false;
  }

  const loom_value_id_t* source_operands =
      loom_op_operands((loom_op_t*)source_op);
  const loom_value_id_t* target_operands =
      loom_op_operands((loom_op_t*)target_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    if (!loom_cfg_simplify_values_equal_after_map(map, source_operands[i],
                                                  target_operands[i])) {
      return false;
    }
  }

  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  const loom_value_id_t* target_results = loom_op_const_results(target_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    if (!loom_cfg_simplify_value_map_append(map, source_results[i],
                                            target_results[i])) {
      return false;
    }
  }
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    loom_type_t source_type =
        loom_module_value_type(state->module, source_results[i]);
    loom_type_t target_type =
        loom_module_value_type(state->module, target_results[i]);
    if (!loom_cfg_simplify_types_equal_after_map(state->module, map,
                                                 source_type, target_type)) {
      return false;
    }
  }

  return loom_cfg_simplify_successors_equal_after_map(
             graph, destinations, source_block, target_block, source_op,
             target_op) &&
         loom_cfg_simplify_attributes_equal(source_op, target_op);
}

static bool loom_cfg_simplify_block_args_equal_after_map(
    const loom_cfg_simplify_state_t* state, const loom_block_t* source_block,
    const loom_block_t* target_block, loom_cfg_simplify_value_map_t* map) {
  if (source_block->arg_count != target_block->arg_count) {
    return false;
  }
  for (uint16_t i = 0; i < source_block->arg_count; ++i) {
    if (!loom_cfg_simplify_value_map_append(
            map, loom_block_arg_id(source_block, i),
            loom_block_arg_id(target_block, i))) {
      return false;
    }
  }
  for (uint16_t i = 0; i < source_block->arg_count; ++i) {
    loom_type_t source_type = loom_module_value_type(
        state->module, loom_block_arg_id(source_block, i));
    loom_type_t target_type = loom_module_value_type(
        state->module, loom_block_arg_id(target_block, i));
    if (!loom_cfg_simplify_types_equal_after_map(state->module, map,
                                                 source_type, target_type)) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_blocks_alpha_equivalent(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const uint16_t* destinations, const loom_block_t* source_block,
    const loom_block_t* target_block) {
  if (source_block->op_count != target_block->op_count) {
    return false;
  }

  loom_cfg_simplify_value_map_t map = {0};
  if (!loom_cfg_simplify_block_args_equal_after_map(state, source_block,
                                                    target_block, &map)) {
    return false;
  }

  const loom_op_t* source_op = source_block->first_op;
  const loom_op_t* target_op = target_block->first_op;
  while (source_op && target_op) {
    if (!loom_cfg_simplify_ops_equal_after_map(state, graph, destinations,
                                               source_block, target_block,
                                               source_op, target_op, &map)) {
      return false;
    }
    source_op = source_op->next_op;
    target_op = target_op->next_op;
  }
  return !source_op && !target_op;
}

static bool loom_cfg_simplify_local_values_append(
    loom_cfg_simplify_local_values_t* local_values, loom_value_id_t value) {
  if (value == LOOM_VALUE_ID_INVALID ||
      local_values->count >= LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES) {
    return false;
  }
  local_values->values[local_values->count++] = value;
  return true;
}

static uint32_t loom_cfg_simplify_hash_alpha_operand(
    const loom_cfg_simplify_local_values_t* local_values, loom_value_id_t value,
    uint32_t fingerprint) {
  for (iree_host_size_t i = 0; i < local_values->count; ++i) {
    if (local_values->values[i] != value) {
      continue;
    }
    fingerprint = loom_cfg_simplify_hash_u32(0, fingerprint);
    return loom_cfg_simplify_hash_u32((uint32_t)i, fingerprint);
  }
  fingerprint = loom_cfg_simplify_hash_u32(1, fingerprint);
  return loom_cfg_simplify_hash_u32(value, fingerprint);
}

// Computes an alpha-invariant fingerprint for cheap candidate rejection.
// Types are intentionally omitted because their dynamic fields may reference
// block-local SSA values. Exact comparison checks every type after remapping.
static bool loom_cfg_simplify_alpha_block_fingerprint(
    const loom_cfg_graph_t* graph, const uint16_t* destinations,
    const loom_block_t* block, uint32_t* out_fingerprint) {
  uint32_t fingerprint = 2166136261u;
  fingerprint = loom_cfg_simplify_hash_u32(block->arg_count, fingerprint);
  fingerprint = loom_cfg_simplify_hash_u32(block->op_count, fingerprint);

  loom_cfg_simplify_local_values_t local_values = {0};
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (!loom_cfg_simplify_local_values_append(&local_values,
                                               loom_block_arg_id(block, i))) {
      return false;
    }
  }

  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    fingerprint = loom_cfg_simplify_hash_u32(op->kind, fingerprint);
    fingerprint = loom_cfg_simplify_hash_u32(op->operand_count, fingerprint);
    fingerprint = loom_cfg_simplify_hash_u32(op->result_count, fingerprint);
    fingerprint = loom_cfg_simplify_hash_u32(op->successor_count, fingerprint);
    fingerprint = loom_cfg_simplify_hash_u32(op->attribute_count, fingerprint);
    fingerprint = loom_cfg_simplify_hash_u32(op->instance_flags, fingerprint);

    const loom_value_id_t* operands = loom_op_const_operands(op);
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      fingerprint = loom_cfg_simplify_hash_alpha_operand(
          &local_values, operands[i], fingerprint);
    }

    const loom_attribute_t* attributes = loom_op_const_attrs(op);
    for (uint8_t i = 0; i < op->attribute_count; ++i) {
      uint32_t attribute_hash = loom_attribute_hash(&attributes[i]);
      fingerprint = loom_cfg_simplify_hash_u32(attribute_hash, fingerprint);
    }

    loom_block_t* const* successors = loom_op_const_successors(op);
    for (uint8_t i = 0; i < op->successor_count; ++i) {
      fingerprint = loom_cfg_simplify_hash_u32(
          loom_cfg_simplify_successor_key(graph, destinations, successors[i]),
          fingerprint);
    }

    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (!loom_cfg_simplify_local_values_append(&local_values, results[i])) {
        return false;
      }
    }
  }

  *out_fingerprint = fingerprint;
  return true;
}

static bool loom_cfg_simplify_can_redirect_successor(
    const loom_cfg_simplify_state_t* state, const loom_op_t* terminator,
    const loom_block_t* old_dest, const loom_block_t* new_dest) {
  if (!new_dest) {
    return false;
  }
  if (loom_cfg_br_isa(terminator)) {
    loom_value_slice_t args = loom_cfg_br_args((loom_op_t*)terminator);
    return loom_cfg_br_dest(terminator) == old_dest &&
           args.count == old_dest->arg_count &&
           args.count == new_dest->arg_count &&
           loom_cfg_block_arguments_can_replace(state->module, state->dominance,
                                                new_dest, args, terminator);
  }
  if (loom_cfg_cond_br_isa(terminator)) {
    return old_dest->arg_count == 0 && new_dest->arg_count == 0 &&
           (loom_cfg_cond_br_true_dest(terminator) == old_dest ||
            loom_cfg_cond_br_false_dest(terminator) == old_dest);
  }
  return false;
}

static bool loom_cfg_simplify_can_redirect_block_predecessors(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, loom_block_t* new_dest) {
  loom_block_t* old_dest = (loom_block_t*)graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  if (predecessors.count == 0) {
    return false;
  }
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const loom_block_t* predecessor =
        graph->blocks[predecessors.values[i]].block;
    if (!predecessor) {
      return false;
    }
    if (predecessor == old_dest || predecessor == new_dest) {
      return false;
    }
    loom_op_t* terminator = ((loom_block_t*)predecessor)->last_op;
    if (!terminator || !loom_cfg_simplify_can_redirect_successor(
                           state, terminator, old_dest, new_dest)) {
      return false;
    }
  }
  return true;
}

// Retains only surviving representatives in the existing candidate table.
// A matched block never becomes a representative, so destinations need no
// transitive resolution and remain live throughout the structural edit.
static uint16_t loom_cfg_simplify_find_equivalent_block(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    loom_cfg_simplify_block_hash_table_t* table, const uint16_t* destinations,
    uint16_t block_index) {
  if (!loom_cfg_simplify_is_alpha_merge_candidate(state, graph, block_index)) {
    return 0;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  uint32_t fingerprint = 0;
  if (!loom_cfg_simplify_alpha_block_fingerprint(graph, destinations, block,
                                                 &fingerprint)) {
    return 0;
  }
  iree_host_size_t slot = fingerprint & (table->capacity - 1);
  while (table->entries[slot].block_index != 0) {
    const loom_cfg_simplify_block_hash_entry_t* entry = &table->entries[slot];
    if (entry->fingerprint == fingerprint) {
      loom_block_t* canonical_block =
          (loom_block_t*)graph->blocks[entry->block_index].block;
      if (loom_cfg_simplify_blocks_alpha_equivalent(state, graph, destinations,
                                                    block, canonical_block) &&
          loom_cfg_simplify_can_redirect_block_predecessors(
              state, graph, block_index, canonical_block)) {
        return entry->block_index;
      }
    }
    slot = (slot + 1) & (table->capacity - 1);
  }
  table->entries[slot] = (loom_cfg_simplify_block_hash_entry_t){
      .fingerprint = fingerprint,
      .block_index = block_index,
  };
  return 0;
}

static iree_status_t loom_cfg_simplify_redirect_block_predecessors(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, loom_block_t* new_dest) {
  const loom_cfg_edge_index_span_t edges =
      loom_cfg_graph_predecessor_edges(graph, block_index);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < edges.count && iree_status_is_ok(status);
       ++i) {
    const loom_cfg_edge_info_t* edge =
        loom_cfg_graph_edge(graph, edges.values[i]);
    loom_op_t* terminator = (loom_op_t*)edge->terminator;
    // Argument replacement was proved before mutation. Keeping the operation
    // and its operands preserves every original edge's edit location.
    loom_op_successors(terminator)[edge->successor_index] = new_dest;
    status = loom_rewriter_add_to_worklist(state->rewriter, terminator);
  }
  return status;
}

static iree_status_t loom_cfg_simplify_merge_equivalent_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }
  loom_cfg_simplify_block_hash_table_t table = {0};
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_block_hash_table_initialize(
      state->analysis_arena, graph->block_count, &table));

  uint16_t* destinations = NULL;
  bool* remove_blocks = NULL;
  uint16_t merge_count = 0;
  iree_status_t status = iree_ok_status();
  // Reuse the graph owner's DFS completion order. Each acyclic successor's
  // merge is decided before its predecessors are compared, so a complete
  // equivalent tail can share this edit without another analysis refresh.
  for (iree_host_size_t i = graph->reverse_postorder.count;
       i > 0 && iree_status_is_ok(status); --i) {
    const uint16_t block_index = graph->reverse_postorder.values[i - 1];
    uint16_t destination = loom_cfg_simplify_find_equivalent_block(
        state, graph, &table, destinations, block_index);
    if (destination == 0) {
      continue;
    }
    if (!destinations) {
      // The destinations and the removal API's mask share one lazy allocation.
      // Zero destinations mean no edit; entry block zero is never a candidate.
      status = iree_arena_allocate_array(
          state->analysis_arena, graph->block_count,
          sizeof(*destinations) + sizeof(*remove_blocks),
          (void**)&destinations);
      if (iree_status_is_ok(status)) {
        remove_blocks = (bool*)(destinations + graph->block_count);
        memset(destinations, 0,
               graph->block_count *
                   (sizeof(*destinations) + sizeof(*remove_blocks)));
      }
    }
    if (iree_status_is_ok(status)) {
      destinations[block_index] = destination;
      remove_blocks[block_index] = true;
      ++merge_count;
    }
  }
  if (!iree_status_is_ok(status) || merge_count == 0) {
    return status;
  }

  // Definitions in discarded blocks have no nonlocal uses. Each destination
  // survives, and equivalent blocks have identical outgoing successors after
  // applying the already-proved successor merges. Redirecting all incoming
  // edges therefore preserves each group's equivalence, even when the
  // predecessor belongs to another merge group.
  // During mutation, original edge records serve only as stable edit locations.
  // The caller refreshes analyses after the complete structural edit.
  for (uint16_t block_index = 1;
       block_index < graph->block_count && iree_status_is_ok(status);
       ++block_index) {
    if (destinations[block_index] == 0) {
      continue;
    }
    status = loom_cfg_simplify_redirect_block_predecessors(
        state, graph, block_index,
        (loom_block_t*)graph->blocks[destinations[block_index]].block);
  }
  if (iree_status_is_ok(status)) {
    uint16_t removed_count = 0;
    status = loom_region_remove_blocks(
        state->module, (loom_region_t*)graph->region, remove_blocks,
        graph->block_count, state->analysis_arena, &removed_count);
    if (iree_status_is_ok(status)) {
      state->statistics->duplicate_blocks_merged += removed_count;
      state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
      *out_changed = true;
    }
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Block argument removal
//===----------------------------------------------------------------------===//

static bool loom_cfg_simplify_type_allows_replacement(
    const loom_module_t* module, loom_value_id_t old_value,
    loom_value_id_t replacement) {
  loom_type_t old_type = loom_module_value_type(module, old_value);
  loom_type_t replacement_type = loom_module_value_type(module, replacement);
  loom_type_value_remap_t remap = {
      .source_values = &old_value,
      .target_values = &replacement,
      .count = 1,
  };
  return loom_type_equal_after_value_remap(module, old_type, replacement_type,
                                           &remap);
}

static bool loom_cfg_simplify_pred_branches_to_block(
    const loom_cfg_graph_t* graph, uint16_t block_index,
    loom_op_t** pred_branches) {
  const loom_block_t* block = graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const loom_block_t* predecessor =
        graph->blocks[predecessors.values[i]].block;
    loom_op_t* terminator = ((loom_block_t*)predecessor)->last_op;
    loom_block_t* dest = NULL;
    loom_value_slice_t args = {0};
    if (!terminator ||
        !loom_cfg_simplify_direct_branch(terminator, &dest, &args) ||
        dest != block) {
      return false;
    }
    if (args.count != block->arg_count) {
      return false;
    }
    pred_branches[i] = terminator;
  }
  return true;
}

// Returns the argument payload of a direct branch already qualified by
// loom_cfg_simplify_pred_branches_to_block.
static loom_value_slice_t loom_cfg_simplify_pred_branch_args(loom_op_t* br) {
  return loom_cfg_br_isa(br) ? loom_cfg_br_args(br) : loom_low_br_args(br);
}

static bool loom_cfg_simplify_incoming_slots_match(
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t lhs_slot, uint16_t rhs_slot) {
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    const loom_value_slice_t args =
        loom_cfg_simplify_pred_branch_args(pred_branches[i]);
    if (args.values[lhs_slot] != args.values[rhs_slot]) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_find_identity_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    uint16_t arg_index, loom_value_id_t* out_replacement) {
  const loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  const loom_value_id_t replacement =
      loom_cfg_value_identity_table_lookup(&state->value_identities, old_arg);
  if (replacement == old_arg || !loom_cfg_simplify_type_allows_replacement(
                                    state->module, old_arg, replacement)) {
    return false;
  }
  const loom_value_t* replacement_value =
      loom_module_value(state->module, replacement);
  IREE_ASSERT(!loom_value_is_block_arg(replacement_value) ||
              loom_value_def_block(replacement_value) != block ||
              loom_value_def_index(replacement_value) < arg_index);
  *out_replacement = replacement;
  return true;
}

// Handles the asymmetric dependent-type case excluded from the identity
// partition: the source type may already name the only valid replacement.
// The retained dependency set enumerates those candidates directly.
static bool loom_cfg_simplify_find_type_dependency_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t arg_index, loom_value_id_t* out_replacement) {
  const loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  const loom_value_id_t old_identity =
      loom_cfg_value_identity_table_lookup(&state->value_identities, old_arg);
  uint16_t best_candidate_index = UINT16_MAX;
  loom_type_use_iterator_t dependencies;
  loom_module_value_type_dependencies(state->module, old_arg, &dependencies);
  for (loom_value_id_t candidate = loom_type_dependencies_next(&dependencies);
       candidate != LOOM_VALUE_ID_INVALID;
       candidate = loom_type_dependencies_next(&dependencies)) {
    const loom_value_t* candidate_value =
        loom_module_value(state->module, candidate);
    if (!loom_value_is_block_arg(candidate_value) ||
        loom_value_def_block(candidate_value) != block) {
      continue;
    }
    const uint16_t candidate_index = loom_value_def_index(candidate_value);
    if (candidate_index >= arg_index ||
        candidate_index >= best_candidate_index) {
      continue;
    }
    const bool same_identity =
        old_identity == loom_cfg_value_identity_table_lookup(
                            &state->value_identities, candidate);
    if (!same_identity &&
        !loom_cfg_simplify_incoming_slots_match(
            pred_branches, predecessor_count, arg_index, candidate_index)) {
      continue;
    }
    if (!loom_cfg_simplify_type_allows_replacement(state->module, old_arg,
                                                   candidate)) {
      continue;
    }
    best_candidate_index = candidate_index;
  }
  if (best_candidate_index == UINT16_MAX) {
    return false;
  }
  *out_replacement = loom_block_arg_id(block, best_candidate_index);
  return true;
}

static bool loom_cfg_simplify_find_forwarded_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t arg_index, loom_value_id_t* out_replacement) {
  if (predecessor_count == 0) {
    return false;
  }
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    const loom_value_slice_t args =
        loom_cfg_simplify_pred_branch_args(pred_branches[i]);
    loom_value_id_t incoming = args.values[arg_index];
    if (incoming == old_arg) {
      continue;
    }
    if (replacement == LOOM_VALUE_ID_INVALID) {
      replacement = incoming;
      continue;
    }
    if (incoming != replacement) {
      return false;
    }
  }
  if (replacement == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (!loom_cfg_simplify_type_allows_replacement(state->module, old_arg,
                                                 replacement)) {
    return false;
  }

  const loom_op_t* anchor = block->first_op ? block->first_op : block->last_op;
  if (!anchor) {
    return false;
  }
  if (!loom_value_is_available_before_op(state->dominance, replacement,
                                         anchor) ||
      !loom_value_type_is_available_before_op(state->dominance, replacement,
                                              anchor)) {
    return false;
  }
  *out_replacement = replacement;
  return true;
}

static iree_status_t loom_cfg_simplify_rebuild_br_for_block_args(
    loom_cfg_simplify_state_t* state, loom_op_t* br, loom_block_t* dest,
    const uint16_t* retained_slots, uint16_t retained_count,
    loom_value_id_t* rebuilt_args) {
  const loom_value_slice_t args = loom_cfg_simplify_pred_branch_args(br);
  IREE_ASSERT_EQ(dest->arg_count, retained_count);
  for (uint16_t arg_index = 0; arg_index < retained_count; ++arg_index) {
    IREE_ASSERT_LT(retained_slots[arg_index], args.count);
    rebuilt_args[arg_index] = args.values[retained_slots[arg_index]];
  }
  return loom_cfg_simplify_replace_direct_br(state, br, dest, rebuilt_args,
                                             retained_count);
}

static bool loom_cfg_simplify_block_arg_has_no_direct_uses(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    uint16_t arg_index) {
  const loom_value_id_t arg = loom_block_arg_id(block, arg_index);
  const loom_value_t* value = loom_module_value(state->module, arg);
  return value->use_count == 0 && !loom_value_has_attribute_uses(value);
}

// Resolves an argument replacement through a previously replaced argument.
// Identity replacements point to earlier arguments, so one lookup reaches the
// final replacement recorded by the ascending argument walk.
static loom_value_id_t loom_cfg_simplify_resolve_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    const loom_value_id_t* replacements, loom_value_id_t replacement) {
  const loom_value_t* value = loom_module_value(state->module, replacement);
  if (!loom_value_is_block_arg(value) || loom_value_def_block(value) != block) {
    return replacement;
  }
  const loom_value_id_t resolved = replacements[loom_value_def_index(value)];
  return resolved == LOOM_VALUE_ID_INVALID ? replacement : resolved;
}

// Computes the greatest closed set of directly unused arguments. An argument
// with an active type user can be removed exactly when that carrier is also in
// the set. Starting with every directly unused argument and propagating each
// retained carrier through its outgoing dependencies visits every relevant
// type edge at most twice.
static uint16_t loom_cfg_simplify_select_unused_block_args(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    bool* remove_args, uint16_t* worklist) {
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    remove_args[arg_index] =
        loom_cfg_simplify_block_arg_has_no_direct_uses(state, block, arg_index);
  }

  uint16_t worklist_count = 0;
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    if (!remove_args[arg_index]) {
      continue;
    }
    loom_type_use_iterator_t users;
    loom_module_value_type_users(state->module,
                                 loom_block_arg_id(block, arg_index), &users);
    for (loom_value_id_t carrier = loom_type_users_next(&users);
         carrier != LOOM_VALUE_ID_INVALID;
         carrier = loom_type_users_next(&users)) {
      const loom_value_t* carrier_value =
          loom_module_value(state->module, carrier);
      if (loom_value_is_block_arg(carrier_value) &&
          loom_value_def_block(carrier_value) == block &&
          remove_args[loom_value_def_index(carrier_value)]) {
        continue;
      }
      remove_args[arg_index] = false;
      worklist[worklist_count++] = arg_index;
      break;
    }
  }

  while (worklist_count > 0) {
    const uint16_t carrier_index = worklist[--worklist_count];
    loom_type_use_iterator_t dependencies;
    loom_module_value_type_dependencies(
        state->module, loom_block_arg_id(block, carrier_index), &dependencies);
    for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&dependencies)) {
      const loom_value_t* provider_value =
          loom_module_value(state->module, provider);
      if (!loom_value_is_block_arg(provider_value) ||
          loom_value_def_block(provider_value) != block) {
        continue;
      }
      const uint16_t provider_index = loom_value_def_index(provider_value);
      if (!remove_args[provider_index]) {
        continue;
      }
      remove_args[provider_index] = false;
      worklist[worklist_count++] = provider_index;
    }
  }

  uint16_t removed_count = 0;
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    removed_count += remove_args[arg_index] ? 1 : 0;
  }
  return removed_count;
}

static iree_status_t loom_cfg_simplify_remove_redundant_block_args_from_block(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, iree_arena_allocator_t* scratch_arena,
    bool* out_changed) {
  *out_changed = false;
  if (!loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return iree_ok_status();
  }
  loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
  if (block->arg_count == 0) {
    return iree_ok_status();
  }

  const loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  if (predecessors.count == 0) {
    return iree_ok_status();
  }
  loom_op_t** pred_branches = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, predecessors.count, sizeof(*pred_branches),
      (void**)&pred_branches));
  if (!loom_cfg_simplify_pred_branches_to_block(graph, block_index,
                                                pred_branches)) {
    return iree_ok_status();
  }

  const uint16_t initial_arg_count = block->arg_count;
  // Reused first as the closure worklist and then as retained incoming slots.
  uint16_t* index_scratch = NULL;
  // Reused first for resolved replacements and then rebuilt branch payloads.
  loom_value_id_t* argument_scratch = NULL;
  bool* remove_args = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, initial_arg_count, sizeof(*index_scratch),
      (void**)&index_scratch));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, initial_arg_count, sizeof(*argument_scratch),
      (void**)&argument_scratch));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, initial_arg_count,
                                sizeof(*remove_args), (void**)&remove_args));
  for (uint16_t arg_index = 0; arg_index < initial_arg_count; ++arg_index) {
    argument_scratch[arg_index] = LOOM_VALUE_ID_INVALID;
  }

  bool block_changed = false;
  for (uint16_t arg_index = 0; arg_index < initial_arg_count; ++arg_index) {
    loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
    const bool found_replacement =
        loom_cfg_simplify_find_identity_arg_replacement(state, block, arg_index,
                                                        &replacement) ||
        loom_cfg_simplify_find_type_dependency_arg_replacement(
            state, block, pred_branches, predecessors.count, arg_index,
            &replacement);
    if (!found_replacement) {
      continue;
    }
    const loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
    const loom_value_id_t resolved = loom_cfg_simplify_resolve_arg_replacement(
        state, block, argument_scratch, replacement);
    if (resolved == old_arg ||
        (resolved != replacement && !loom_cfg_simplify_type_allows_replacement(
                                        state->module, old_arg, resolved))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        state->rewriter, old_arg, resolved));
    argument_scratch[arg_index] = resolved;
    block_changed = true;
  }

  for (uint16_t arg_index = 0; arg_index < initial_arg_count; ++arg_index) {
    if (argument_scratch[arg_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
    if (!loom_cfg_simplify_find_forwarded_arg_replacement(
            state, block, pred_branches, predecessors.count, arg_index,
            &replacement)) {
      continue;
    }
    const loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
    const loom_value_id_t resolved = loom_cfg_simplify_resolve_arg_replacement(
        state, block, argument_scratch, replacement);
    if (resolved == old_arg) {
      continue;
    }
    if (resolved != replacement &&
        (!loom_cfg_simplify_type_allows_replacement(state->module, old_arg,
                                                    resolved) ||
         !loom_value_is_available_before_op(
             state->dominance, resolved,
             block->first_op ? block->first_op : block->last_op) ||
         !loom_value_type_is_available_before_op(
             state->dominance, resolved,
             block->first_op ? block->first_op : block->last_op))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        state->rewriter, old_arg, resolved));
    argument_scratch[arg_index] = resolved;
    block_changed = true;
  }

  const uint16_t removed_count = loom_cfg_simplify_select_unused_block_args(
      state, block, remove_args, index_scratch);
  if (removed_count == 0) {
    *out_changed = block_changed;
    return iree_ok_status();
  }

  uint16_t retained_count = 0;
  for (uint16_t arg_index = 0; arg_index < initial_arg_count; ++arg_index) {
    if (!remove_args[arg_index]) {
      index_scratch[retained_count++] = arg_index;
    }
  }
  IREE_ASSERT_EQ((uint16_t)(retained_count + removed_count), initial_arg_count);
  const uint16_t actual_removed_count = loom_block_remove_args(
      state->module, block, remove_args, initial_arg_count);
  IREE_ASSERT_EQ(actual_removed_count, removed_count);
  state->statistics->block_args_removed += removed_count;
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;

  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_rebuild_br_for_block_args(
        state, pred_branches[i], block, index_scratch, retained_count,
        argument_scratch));
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_remove_redundant_block_args(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(state->analysis_arena->block_pool, &scratch_arena);
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 1;
       iree_status_is_ok(status) && block_index < graph->block_count;
       ++block_index) {
    iree_arena_reset(&scratch_arena);
    bool block_changed = false;
    status = loom_cfg_simplify_remove_redundant_block_args_from_block(
        state, graph, block_index, &scratch_arena, &block_changed);
    *out_changed |= block_changed;
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_process_cfg_region(
    loom_cfg_simplify_state_t* state, loom_region_t* region,
    bool* out_changed) {
  // The fact owner publishes current structure after each completed edit.
  // This borrow ends before the driver refreshes that structural snapshot.
  const loom_value_fact_cfg_region_t* structure = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
      state->rewriter->fact_table, state->module, region, &structure));
  const loom_cfg_graph_t* graph = &structure->graph;
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_remove_unreachable_blocks(state, graph, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_value_identity_table_update(
      &state->value_identities, structure, state->dominance,
      state->analysis_arena));
  loom_cfg_condition_relation_table_t path_fact_table = {0};
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_table_compute(
      state->module, graph, state->fact_table, state->dominance,
      &state->value_domain, &state->value_identities, state->analysis_arena,
      &path_fact_table));
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_thread_fact_known_branches(
      state, graph, &path_fact_table, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  uint16_t branches_folded = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_fold_path_sensitive_branches(
      state->rewriter, graph, &path_fact_table, &state->condition_query,
      state->analysis_arena, &branches_folded));
  if (branches_folded != 0) {
    state->statistics->branches_folded += branches_folded;
    *out_changed = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_fold_path_sensitive_i1_ops(
      state, graph, &path_fact_table, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  uint16_t fused_count = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_fuse_single_predecessor_blocks(
      state->rewriter, graph, state->dominance, state->analysis_arena,
      &fused_count));
  if (fused_count != 0) {
    state->statistics->blocks_fused += fused_count;
    *out_changed = true;
    return iree_ok_status();
  }

  iree_host_size_t forwarded_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_cfg_forward_empty_blocks(state->rewriter, graph, state->dominance,
                                    state->analysis_arena, &forwarded_count));
  if (forwarded_count != 0) {
    state->statistics->edges_forwarded += forwarded_count;
    *out_changed = true;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_merge_equivalent_blocks(state, graph, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  return loom_cfg_simplify_remove_redundant_block_args(state, graph,
                                                       out_changed);
}

static iree_status_t loom_cfg_simplify_process_function_once(
    loom_cfg_simplify_state_t* state, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  state->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_region_stack_push(
      state->pass->arena, &state->region_stack, body));
  while (true) {
    loom_region_t* region =
        loom_cfg_simplify_region_stack_pop(&state->region_stack);
    if (!region) {
      break;
    }

    uint16_t branches_folded = 0;
    IREE_RETURN_IF_ERROR(loom_cfg_fold_constant_branches(
        state->rewriter, region, state->analysis_arena, &branches_folded));
    if (branches_folded != 0) {
      state->statistics->branches_folded += branches_folded;
      *out_changed = true;
      return loom_rewriter_refresh_cfg_facts(state->rewriter, region);
    }

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        IREE_RETURN_IF_ERROR(loom_cfg_simplify_push_child_regions(state, op));
      }
    }

    if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
      IREE_RETURN_IF_ERROR(
          loom_cfg_simplify_process_cfg_region(state, region, out_changed));
      if (*out_changed) {
        return loom_rewriter_refresh_cfg_facts(state->rewriter, region);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_simplify_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter = {0};
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  IREE_RETURN_IF_ERROR(loom_rewriter_enable_worklist(&rewriter));

  iree_arena_allocator_t analysis_arena = {0};
  iree_arena_initialize(pass->arena->block_pool, &analysis_arena);

  loom_cfg_simplify_state_t state = {
      .pass = pass,
      .statistics = loom_cfg_simplify_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
      .analysis_arena = &analysis_arena,
  };

  iree_status_t status = loom_cfg_simplify_region_stack_initialize(
      pass->arena, &state.region_stack);
  if (iree_status_is_ok(status)) {
    status = loom_cfg_simplify_mark_cfg_regions(body, &analysis_arena);
  }
  loom_value_fact_table_t* fact_table = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_function_for_target(
            function,
            loom_target_function_version_target_facts(pass->function_version)),
        &fact_table);
  }
  loom_rewriter_attach_value_facts(&rewriter, fact_table);
  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    ++state.statistics->iterations;
    changed = false;
    loom_local_value_domain_release(&state.value_domain);
    iree_arena_reset(&analysis_arena);

    loom_op_t* pending_op = NULL;
    while (iree_status_is_ok(status) &&
           (pending_op = loom_rewriter_pop(&rewriter)) != NULL) {
      bool folded = false;
      status = loom_rewriter_try_fold(&rewriter, pending_op, &folded);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }

    status = loom_cfg_simplify_mark_cfg_regions(body, &analysis_arena);
    if (!iree_status_is_ok(status)) {
      break;
    }

    status = loom_local_value_domain_acquire_for_region_tree(
        module, body, &analysis_arena, &state.value_domain);
    if (!iree_status_is_ok(status)) {
      break;
    }
    status = loom_cfg_value_identity_table_initialize(
        &state.value_domain, &analysis_arena, &state.value_identities);
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_condition_query_initialize(module, &state.value_domain,
                                    &analysis_arena, &state.condition_query);

    loom_dominance_info_t dominance = {0};
    status = loom_dominance_info_initialize_region(module, body,
                                                   &analysis_arena, &dominance);
    if (!iree_status_is_ok(status)) {
      break;
    }

    state.fact_table = fact_table;
    state.dominance = &dominance;
    status =
        loom_cfg_simplify_process_function_once(&state, function, &changed);
    if (changed) {
      any_changed = true;
    }
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_invalidate(pass->value_facts);
  loom_local_value_domain_release(&state.value_domain);
  iree_arena_deinitialize(&analysis_arena);
  return status;
}
