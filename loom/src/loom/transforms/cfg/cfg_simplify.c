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
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_cfg.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_CFG_SIMPLIFY_STATISTICS(V, statistics_type)                       \
  V(statistics_type, iterations, "iterations",                                 \
    "Number of fixed-point simplification iterations.")                        \
  V(statistics_type, branches_folded, "branches-folded",                       \
    "Number of conditional branches folded to direct branches.")               \
  V(statistics_type, edges_forwarded, "edges-forwarded",                       \
    "Number of predecessor edges forwarded through trivial blocks.")           \
  V(statistics_type, blocks_removed, "blocks-removed",                         \
    "Number of unreachable CFG blocks removed.")                               \
  V(statistics_type, block_args_removed, "block-args-removed",                 \
    "Number of redundant CFG block arguments removed.")                        \
  V(statistics_type, blocks_fused, "blocks-fused",                             \
    "Number of single-predecessor CFG blocks fused into their "                \
    "predecessors.")                                                           \
  V(statistics_type, duplicate_blocks_merged, "duplicate-blocks-merged",       \
    "Number of duplicate terminal CFG blocks merged.")                         \
  V(statistics_type, terminal_blocks_duplicated, "terminal-blocks-duplicated", \
    "Number of direct branches replaced by duplicated terminal "               \
    "successors.")

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

static iree_status_t loom_cfg_simplify_replace_cond_br_with_br(
    loom_cfg_simplify_state_t* state, loom_op_t* cond_br, loom_block_t* dest) {
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_replace_br(state, cond_br, dest, NULL, 0));
  ++state->statistics->branches_folded;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_fold_cond_br(
    loom_cfg_simplify_state_t* state, loom_op_t* op, bool* out_changed) {
  if (!loom_cfg_cond_br_isa(op)) {
    return iree_ok_status();
  }
  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(op);
  if (true_dest == false_dest && true_dest && true_dest->arg_count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_replace_cond_br_with_br(state, op, true_dest));
    *out_changed = true;
    return iree_ok_status();
  }

  bool condition = false;
  loom_value_facts_t facts = loom_value_fact_table_lookup(
      state->fact_table, loom_cfg_cond_br_condition(op));
  if (!loom_value_facts_as_exact_bool(facts, &condition)) {
    return iree_ok_status();
  }
  loom_block_t* chosen_dest = condition ? true_dest : false_dest;
  if (!chosen_dest || chosen_dest->arg_count != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_replace_cond_br_with_br(state, op, chosen_dest));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_fold_block_branches(
    loom_cfg_simplify_state_t* state, loom_block_t* block, bool* out_changed) {
  loom_op_t* op = block->first_op;
  while (op) {
    loom_op_t* next_op = op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_fold_cond_br(state, op, out_changed));
    if (*out_changed) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_push_child_regions(state, op));
    op = next_op;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Terminal block duplication
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

static bool loom_cfg_simplify_operandless_terminal_block(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    const loom_op_t** out_terminator) {
  *out_terminator = NULL;
  if (!block || block->arg_count != 0 || block->first_op != block->last_op ||
      !block->first_op) {
    return false;
  }

  const loom_op_t* terminator = block->first_op;
  loom_trait_flags_t traits =
      loom_op_effective_traits(state->module, terminator);
  if (!iree_all_bits_set(traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_PURE) ||
      loom_traits_are_convergent(traits) || terminator->operand_count != 0 ||
      terminator->successor_count != 0 || terminator->result_count != 0 ||
      terminator->region_count != 0) {
    return false;
  }
  *out_terminator = terminator;
  return true;
}

static bool loom_cfg_simplify_can_duplicate_terminal_successor(
    const loom_cfg_graph_t* graph, uint16_t dest_block_index) {
  // A shared terminal block is a real reconvergence point even when the
  // terminator itself is cheap to clone.
  if (graph->blocks[dest_block_index].predecessor_count > 1) {
    return false;
  }
  return true;
}

static iree_status_t loom_cfg_simplify_clone_terminal_before_branch(
    loom_cfg_simplify_state_t* state, loom_op_t* branch_op,
    const loom_op_t* terminator) {
  loom_builder_ip_t saved_ip = loom_builder_save(&state->rewriter->builder);
  loom_builder_set_before(&state->rewriter->builder, branch_op);

  loom_ir_remap_t remap = {0};
  const loom_ir_remap_options_t remap_options = {
      .allow_unmapped_values = true,
  };
  iree_status_t status =
      loom_ir_remap_initialize(state->module, state->module,
                               state->analysis_arena, &remap_options, &remap);
  if (iree_status_is_ok(status)) {
    loom_op_t* cloned_op = NULL;
    status = loom_ir_clone_op(&state->rewriter->builder, terminator, &remap,
                              &cloned_op);
    (void)cloned_op;
  }

  loom_builder_restore(&state->rewriter->builder, saved_ip);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  return loom_rewriter_erase(state->rewriter, branch_op);
}

static iree_status_t loom_cfg_simplify_duplicate_terminal_successors(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (!block || !block->last_op) {
      continue;
    }

    loom_block_t* dest = NULL;
    loom_value_slice_t args = {0};
    if (!loom_cfg_simplify_direct_branch(block->last_op, &dest, &args) ||
        args.count != 0 || dest == block) {
      continue;
    }

    const loom_op_t* terminator = NULL;
    if (!loom_cfg_simplify_operandless_terminal_block(state, dest,
                                                      &terminator)) {
      continue;
    }
    const iree_host_size_t dest_block_index =
        loom_cfg_graph_block_index(graph, dest);
    if (dest_block_index == IREE_HOST_SIZE_MAX ||
        !loom_cfg_simplify_can_duplicate_terminal_successor(
            graph, (uint16_t)dest_block_index)) {
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_cfg_simplify_clone_terminal_before_branch(
        state, block->last_op, terminator));
    ++state->statistics->terminal_blocks_duplicated;
    state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
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

static iree_status_t loom_cfg_simplify_fold_path_sensitive_cond_br(
    loom_cfg_simplify_state_t* state,
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view, loom_op_t* op,
    bool* out_changed) {
  if (!loom_cfg_cond_br_isa(op)) {
    return iree_ok_status();
  }

  loom_value_id_t condition_value = loom_cfg_cond_br_condition(op);
  bool condition = false;
  bool condition_proven = false;
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_entry_facts_prove_bool(
      state, table, view, condition_value, &condition, &condition_proven));
  if (!condition_proven) {
    return iree_ok_status();
  }

  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(op);
  loom_block_t* chosen_dest = condition ? true_dest : false_dest;
  if (!chosen_dest || chosen_dest->arg_count != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_replace_cond_br_with_br(state, op, chosen_dest));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_thread_predecessor_to_block(
    loom_cfg_simplify_state_t* state, loom_block_t* old_dest,
    loom_block_t* new_dest, const loom_cfg_edge_info_t* predecessor_edge,
    bool* out_changed) {
  if (!new_dest || new_dest->arg_count != 0) {
    return iree_ok_status();
  }
  loom_op_t* predecessor_terminator = (loom_op_t*)predecessor_edge->terminator;
  if (loom_cfg_br_isa(predecessor_terminator)) {
    if (predecessor_edge->successor_index != 0 ||
        loom_cfg_br_dest(predecessor_terminator) != old_dest) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_replace_br(
        state, predecessor_terminator, new_dest, NULL, 0));
    ++state->statistics->edges_forwarded;
    *out_changed = true;
    return iree_ok_status();
  }

  if (!loom_cfg_cond_br_isa(predecessor_terminator) ||
      old_dest->arg_count != 0) {
    return iree_ok_status();
  }
  loom_block_t** successors = loom_op_successors(predecessor_terminator);
  const uint16_t successor_index = predecessor_edge->successor_index;
  if (successor_index >= predecessor_terminator->successor_count ||
      successors[successor_index] != old_dest) {
    return iree_ok_status();
  }
  successors[successor_index] = new_dest;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_to_worklist(state->rewriter, predecessor_terminator));
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  ++state->statistics->edges_forwarded;
  *out_changed = true;
  return iree_ok_status();
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
      if (!new_dest || new_dest->arg_count != 0) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_thread_predecessor_to_block(
          state, block, new_dest, predecessor_edge, out_changed));
      if (*out_changed) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_fold_path_sensitive_branches(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_condition_relation_table_t* table, bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }
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
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_fold_path_sensitive_cond_br(
          state, table,
          loom_cfg_condition_relation_table_block(table, block_index), op,
          out_changed));
      if (*out_changed) {
        return iree_ok_status();
      }
      op = next_op;
    }
  }
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
            return iree_ok_status();
          }
        }
      }
      op = next_op;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block removal
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_remove_cfg_block(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index) {
  bool* remove_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->analysis_arena, graph->block_count, sizeof(*remove_blocks),
      (void**)&remove_blocks));
  memset(remove_blocks, 0, graph->block_count * sizeof(*remove_blocks));
  remove_blocks[block_index] = true;

  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_region_remove_blocks(
      state->module, (loom_region_t*)graph->region, remove_blocks,
      (uint16_t)graph->block_count, state->analysis_arena, &removed_count));
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
  loom_value_id_t source_values[LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES];
  loom_value_id_t target_values[LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES];
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

static void loom_cfg_simplify_block_hash_table_reset(
    loom_cfg_simplify_block_hash_table_t* table) {
  memset(table->entries, 0, table->capacity * sizeof(*table->entries));
}

static bool loom_cfg_simplify_is_mergeable_terminal_block(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index) {
  if (block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return false;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  if (!block || block->arg_count != 0 || block->first_op != block->last_op ||
      !block->first_op) {
    return false;
  }

  const loom_op_t* terminator = block->first_op;
  loom_trait_flags_t traits =
      loom_op_effective_traits(state->module, terminator);
  return iree_all_bits_set(traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_PURE) &&
         !loom_traits_are_convergent(traits) &&
         terminator->successor_count == 0 && terminator->result_count == 0 &&
         terminator->region_count == 0;
}

static bool loom_cfg_simplify_terminal_ops_equal(const loom_op_t* lhs,
                                                 const loom_op_t* rhs) {
  if (lhs->kind != rhs->kind || lhs->operand_count != rhs->operand_count ||
      lhs->attribute_count != rhs->attribute_count ||
      lhs->instance_flags != rhs->instance_flags) {
    return false;
  }

  const loom_value_id_t* lhs_operands = loom_op_operands((loom_op_t*)lhs);
  const loom_value_id_t* rhs_operands = loom_op_operands((loom_op_t*)rhs);
  if (memcmp(lhs_operands, rhs_operands,
             (iree_host_size_t)lhs->operand_count * sizeof(*lhs_operands)) !=
      0) {
    return false;
  }

  const loom_attribute_t* lhs_attrs = loom_op_attrs((loom_op_t*)lhs);
  const loom_attribute_t* rhs_attrs = loom_op_attrs((loom_op_t*)rhs);
  for (uint8_t i = 0; i < lhs->attribute_count; ++i) {
    if (!loom_attribute_equal(&lhs_attrs[i], &rhs_attrs[i])) {
      return false;
    }
  }
  return true;
}

static uint32_t loom_cfg_simplify_terminal_op_fingerprint(const loom_op_t* op) {
  uint32_t fingerprint = 2166136261u;
  fingerprint = loom_cfg_simplify_hash_u32(op->kind, fingerprint);
  fingerprint = loom_cfg_simplify_hash_u32(op->operand_count, fingerprint);
  fingerprint = loom_cfg_simplify_hash_u32(op->attribute_count, fingerprint);
  fingerprint = loom_cfg_simplify_hash_u32(op->instance_flags, fingerprint);
  fingerprint = loom_cfg_simplify_hash_bytes(
      loom_op_const_operands(op),
      (iree_host_size_t)op->operand_count * sizeof(loom_value_id_t),
      fingerprint);
  const loom_attribute_t* attributes = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    uint32_t attribute_hash = loom_attribute_hash(&attributes[i]);
    fingerprint = loom_cfg_simplify_hash_u32(attribute_hash, fingerprint);
  }
  return fingerprint;
}

static bool loom_cfg_simplify_find_duplicate_terminal_block(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    loom_cfg_simplify_block_hash_table_t* table, uint16_t block_index,
    loom_block_t** out_canonical_block) {
  *out_canonical_block = NULL;
  if (!loom_cfg_simplify_is_mergeable_terminal_block(state, graph,
                                                     block_index)) {
    return false;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  const loom_op_t* terminator = block->first_op;
  uint32_t fingerprint = loom_cfg_simplify_terminal_op_fingerprint(terminator);
  iree_host_size_t slot = fingerprint & (table->capacity - 1);
  while (table->entries[slot].block_index != 0) {
    const loom_cfg_simplify_block_hash_entry_t* entry = &table->entries[slot];
    if (!*out_canonical_block && entry->fingerprint == fingerprint) {
      loom_block_t* canonical_block =
          (loom_block_t*)graph->blocks[entry->block_index].block;
      if (loom_cfg_simplify_terminal_ops_equal(canonical_block->first_op,
                                               terminator)) {
        *out_canonical_block = canonical_block;
      }
    }
    slot = (slot + 1) & (table->capacity - 1);
  }
  table->entries[slot] = (loom_cfg_simplify_block_hash_entry_t){
      .fingerprint = fingerprint,
      .block_index = block_index,
  };
  return *out_canonical_block != NULL;
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

static bool loom_cfg_simplify_successors_equal_after_map(
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
    if (source_successor != target_successor) {
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
    const loom_cfg_simplify_state_t* state, const loom_block_t* source_block,
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
             source_block, target_block, source_op, target_op) &&
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
    const loom_cfg_simplify_state_t* state, const loom_block_t* source_block,
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
    if (!loom_cfg_simplify_ops_equal_after_map(
            state, source_block, target_block, source_op, target_op, &map)) {
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
      fingerprint =
          loom_cfg_simplify_hash_u32(successors[i]->region_index, fingerprint);
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

static bool loom_cfg_simplify_find_alpha_equivalent_block(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    loom_cfg_simplify_block_hash_table_t* table, uint16_t block_index,
    loom_block_t** out_canonical_block) {
  *out_canonical_block = NULL;
  if (!loom_cfg_simplify_is_alpha_merge_candidate(state, graph, block_index)) {
    return false;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  uint32_t fingerprint = 0;
  if (!loom_cfg_simplify_alpha_block_fingerprint(block, &fingerprint)) {
    return false;
  }
  iree_host_size_t slot = fingerprint & (table->capacity - 1);
  while (table->entries[slot].block_index != 0) {
    const loom_cfg_simplify_block_hash_entry_t* entry = &table->entries[slot];
    if (!*out_canonical_block && entry->fingerprint == fingerprint) {
      loom_block_t* canonical_block =
          (loom_block_t*)graph->blocks[entry->block_index].block;
      if (loom_cfg_simplify_blocks_alpha_equivalent(state, block,
                                                    canonical_block)) {
        *out_canonical_block = canonical_block;
      }
    }
    slot = (slot + 1) & (table->capacity - 1);
  }
  table->entries[slot] = (loom_cfg_simplify_block_hash_entry_t){
      .fingerprint = fingerprint,
      .block_index = block_index,
  };
  return *out_canonical_block != NULL;
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

static iree_status_t loom_cfg_simplify_redirect_successor(
    loom_cfg_simplify_state_t* state, loom_op_t* terminator,
    loom_block_t* old_dest, loom_block_t* new_dest) {
  if (loom_cfg_br_isa(terminator)) {
    loom_value_slice_t args = loom_cfg_br_args(terminator);
    return loom_cfg_simplify_replace_br(state, terminator, new_dest,
                                        args.values, args.count);
  }

  loom_block_t** successors = loom_op_successors(terminator);
  bool changed = false;
  for (uint8_t successor_index = 0;
       successor_index < terminator->successor_count; ++successor_index) {
    if (successors[successor_index] != old_dest) {
      continue;
    }
    successors[successor_index] = new_dest;
    changed = true;
  }
  if (!changed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_to_worklist(state->rewriter, terminator));
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_redirect_block_predecessors(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, loom_block_t* new_dest) {
  loom_block_t* old_dest = (loom_block_t*)graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    loom_block_t* predecessor =
        (loom_block_t*)graph->blocks[predecessors.values[i]].block;
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_redirect_successor(
        state, predecessor->last_op, old_dest, new_dest));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_merge_alpha_equivalent_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    loom_cfg_simplify_block_hash_table_t* table, bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* canonical_block = NULL;
    if (!loom_cfg_simplify_find_alpha_equivalent_block(
            state, graph, table, block_index, &canonical_block)) {
      continue;
    }
    if (!loom_cfg_simplify_can_redirect_block_predecessors(
            state, graph, block_index, canonical_block)) {
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_cfg_simplify_redirect_block_predecessors(
        state, graph, block_index, canonical_block));
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_remove_cfg_block(state, graph, block_index));
    ++state->statistics->duplicate_blocks_merged;
    state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_merge_duplicate_terminal_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    loom_cfg_simplify_block_hash_table_t* table, bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* canonical_block = NULL;
    if (!loom_cfg_simplify_find_duplicate_terminal_block(
            state, graph, table, block_index, &canonical_block)) {
      continue;
    }

    if (!loom_cfg_simplify_can_redirect_block_predecessors(
            state, graph, block_index, canonical_block)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_redirect_block_predecessors(
        state, graph, block_index, canonical_block));
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_remove_cfg_block(state, graph, block_index));
    ++state->statistics->duplicate_blocks_merged;
    state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
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

static bool loom_cfg_simplify_incoming_slots_match(
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t lhs_slot, uint16_t rhs_slot) {
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    loom_block_t* dest = NULL;
    loom_value_slice_t args = {0};
    if (!loom_cfg_simplify_direct_branch(pred_branches[i], &dest, &args)) {
      return false;
    }
    if (args.values[lhs_slot] != args.values[rhs_slot]) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_find_duplicate_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    const uint16_t* incoming_slot_by_arg, uint16_t arg_index,
    loom_value_id_t* out_replacement) {
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  for (uint16_t candidate_index = 0; candidate_index < arg_index;
       ++candidate_index) {
    loom_value_id_t candidate = loom_block_arg_id(block, candidate_index);
    if (!loom_cfg_simplify_incoming_slots_match(
            pred_branches, predecessor_count, incoming_slot_by_arg[arg_index],
            incoming_slot_by_arg[candidate_index])) {
      continue;
    }
    if (!loom_cfg_simplify_type_allows_replacement(state->module, old_arg,
                                                   candidate)) {
      continue;
    }
    *out_replacement = candidate;
    return true;
  }
  return false;
}

static bool loom_cfg_simplify_find_forwarded_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    const uint16_t* incoming_slot_by_arg, uint16_t arg_index,
    loom_value_id_t* out_replacement) {
  if (predecessor_count == 0) {
    return false;
  }
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    loom_block_t* dest = NULL;
    loom_value_slice_t args = {0};
    if (!loom_cfg_simplify_direct_branch(pred_branches[i], &dest, &args)) {
      return false;
    }
    loom_value_id_t incoming = args.values[incoming_slot_by_arg[arg_index]];
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
    loom_cfg_simplify_state_t* state, loom_op_t* br,
    const uint16_t* incoming_slot_by_arg, uint16_t arg_count,
    loom_value_id_t* rebuilt_args) {
  loom_block_t* dest = NULL;
  loom_value_slice_t args = {0};
  if (!loom_cfg_simplify_direct_branch(br, &dest, &args)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cfg-simplify block argument removal expected a "
                            "cfg.br or low.br predecessor");
  }
  for (uint16_t arg_index = 0; arg_index < arg_count; ++arg_index) {
    rebuilt_args[arg_index] = args.values[incoming_slot_by_arg[arg_index]];
  }
  return loom_cfg_simplify_replace_direct_br(state, br, dest, rebuilt_args,
                                             arg_count);
}

static iree_status_t loom_cfg_simplify_remove_block_arg(
    loom_cfg_simplify_state_t* state, loom_block_t* block, uint16_t arg_index,
    loom_value_id_t replacement) {
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      state->rewriter, old_arg, replacement));
  IREE_RETURN_IF_ERROR(loom_block_remove_arg(state->module, block, arg_index));
  ++state->statistics->block_args_removed;
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static bool loom_cfg_simplify_block_arg_unused(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    uint16_t arg_index) {
  loom_value_id_t arg = loom_block_arg_id(block, arg_index);
  if (arg == LOOM_VALUE_ID_INVALID || arg >= state->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(state->module, arg);
  return value->use_count == 0 && !loom_value_has_attribute_uses(value) &&
         !loom_module_value_has_type_uses(state->module, arg);
}

static iree_status_t loom_cfg_simplify_remove_unused_block_arg(
    loom_cfg_simplify_state_t* state, loom_block_t* block, uint16_t arg_index) {
  IREE_RETURN_IF_ERROR(loom_block_remove_arg(state->module, block, arg_index));
  ++state->statistics->block_args_removed;
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static void loom_cfg_simplify_remove_incoming_slot_for_arg(
    uint16_t* incoming_slot_by_arg, uint16_t arg_count,
    uint16_t removed_arg_index) {
  for (uint16_t arg_index = (uint16_t)(removed_arg_index + 1);
       arg_index < arg_count; ++arg_index) {
    incoming_slot_by_arg[arg_index - 1] = incoming_slot_by_arg[arg_index];
  }
}

static iree_status_t loom_cfg_simplify_remove_redundant_block_args(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) {
      continue;
    }
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (block->arg_count == 0) {
      continue;
    }

    loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    if (predecessors.count == 0) {
      continue;
    }
    loom_op_t** pred_branches = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->analysis_arena, predecessors.count, sizeof(*pred_branches),
        (void**)&pred_branches));
    if (!loom_cfg_simplify_pred_branches_to_block(graph, block_index,
                                                  pred_branches)) {
      continue;
    }

    const uint16_t initial_arg_count = block->arg_count;
    uint16_t* incoming_slot_by_arg = NULL;
    loom_value_id_t* rebuilt_args = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->analysis_arena, initial_arg_count, sizeof(*incoming_slot_by_arg),
        (void**)&incoming_slot_by_arg));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->analysis_arena, initial_arg_count, sizeof(*rebuilt_args),
        (void**)&rebuilt_args));
    for (uint16_t arg_index = 0; arg_index < initial_arg_count; ++arg_index) {
      incoming_slot_by_arg[arg_index] = arg_index;
    }

    bool block_changed = false;
    uint16_t arg_index = 0;
    while (arg_index < block->arg_count) {
      if (loom_cfg_simplify_block_arg_unused(state, block, arg_index)) {
        const uint16_t previous_arg_count = block->arg_count;
        IREE_RETURN_IF_ERROR(
            loom_cfg_simplify_remove_unused_block_arg(state, block, arg_index));
        loom_cfg_simplify_remove_incoming_slot_for_arg(
            incoming_slot_by_arg, previous_arg_count, arg_index);
        block_changed = true;
        continue;
      }

      loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
      bool found_replacement =
          loom_cfg_simplify_find_duplicate_arg_replacement(
              state, block, pred_branches, predecessors.count,
              incoming_slot_by_arg, arg_index, &replacement) ||
          loom_cfg_simplify_find_forwarded_arg_replacement(
              state, block, pred_branches, predecessors.count,
              incoming_slot_by_arg, arg_index, &replacement);
      if (!found_replacement) {
        ++arg_index;
        continue;
      }
      const uint16_t previous_arg_count = block->arg_count;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_remove_block_arg(
          state, block, arg_index, replacement));
      loom_cfg_simplify_remove_incoming_slot_for_arg(
          incoming_slot_by_arg, previous_arg_count, arg_index);
      block_changed = true;
    }

    // Removing a later argument may release the final type use of an earlier
    // argument. References between block argument types can only point to
    // earlier arguments, so one reverse sweep removes the newly unused
    // dependency chain without rebuilding function-scoped facts.
    for (arg_index = block->arg_count; arg_index > 0;) {
      --arg_index;
      if (!loom_cfg_simplify_block_arg_unused(state, block, arg_index)) {
        continue;
      }
      const uint16_t previous_arg_count = block->arg_count;
      IREE_RETURN_IF_ERROR(
          loom_cfg_simplify_remove_unused_block_arg(state, block, arg_index));
      loom_cfg_simplify_remove_incoming_slot_for_arg(
          incoming_slot_by_arg, previous_arg_count, arg_index);
      block_changed = true;
    }

    if (!block_changed) {
      continue;
    }
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_rebuild_br_for_block_args(
          state, pred_branches[i], incoming_slot_by_arg, block->arg_count,
          rebuilt_args));
    }
    *out_changed = true;
  }
  return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_fold_path_sensitive_branches(
      state, graph, &path_fact_table, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_fold_path_sensitive_i1_ops(
      state, graph, &path_fact_table, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_duplicate_terminal_successors(
      state, graph, out_changed));
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

  loom_cfg_simplify_block_hash_table_t block_hash_table = {0};
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_block_hash_table_initialize(
      state->analysis_arena, graph->block_count, &block_hash_table));
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_merge_duplicate_terminal_blocks(
      state, graph, &block_hash_table, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  loom_cfg_simplify_block_hash_table_reset(&block_hash_table);
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_merge_alpha_equivalent_blocks(
      state, graph, &block_hash_table, out_changed));
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

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_cfg_simplify_fold_block_branches(state, block, out_changed));
      if (*out_changed) {
        return loom_rewriter_refresh_cfg_facts(state->rewriter, region);
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
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

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
