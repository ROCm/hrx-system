// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/cfg_simplify.h"

#include <string.h>

#include "loom/analysis/condition_facts.h"
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
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_CFG_SIMPLIFY_STATISTICS(V, statistics_type)                       \
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
  // Facts computed for the current fixed-point iteration.
  const loom_value_fact_table_t* fact_table;
  // Dominance computed for the current fixed-point iteration.
  const loom_dominance_info_t* dominance;
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
  if (!region) return iree_ok_status();
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
  if (!root_region) return iree_ok_status();

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
  if (!loom_cfg_cond_br_isa(op)) return iree_ok_status();
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
  if (!chosen_dest || chosen_dest->arg_count != 0) return iree_ok_status();
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
    if (*out_changed) return iree_ok_status();
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
  if (graph->malformed || graph->block_count <= 1) return iree_ok_status();

  bool* remove_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->analysis_arena, graph->block_count, sizeof(*remove_blocks),
      (void**)&remove_blocks));
  memset(remove_blocks, 0, graph->block_count * sizeof(*remove_blocks));

  bool any_removed = false;
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    remove_blocks[block_index] = true;
    any_removed = true;
  }
  if (!any_removed) return iree_ok_status();

  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_region_remove_blocks(
      state->module, (loom_region_t*)graph->region, remove_blocks,
      graph->block_count, &removed_count));
  state->statistics->blocks_removed += removed_count;
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  *out_changed = removed_count != 0;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Path-sensitive branch facts
//===----------------------------------------------------------------------===//

#define LOOM_CFG_SIMPLIFY_EDGE_RELATION_CAPACITY 32

typedef struct loom_cfg_simplify_block_entry_path_facts_t {
  // SSA condition proven by every reachable predecessor edge into the block.
  loom_value_id_t condition;
  // Boolean value proven for condition at block entry.
  bool condition_value;
  // True when condition and condition_value contain a usable entry fact.
  bool condition_known;
  // Integer relations proven by every reachable predecessor edge.
  loom_condition_integer_relation_t* integer_relations;
  // Number of valid entries in integer_relations.
  iree_host_size_t integer_relation_count;
} loom_cfg_simplify_block_entry_path_facts_t;

static bool loom_cfg_simplify_edge_implies_bool(const loom_block_t* target,
                                                loom_op_t* terminator,
                                                loom_value_id_t* out_condition,
                                                bool* out_value) {
  if (!terminator || !loom_cfg_cond_br_isa(terminator)) {
    return false;
  }

  bool is_true_edge = loom_cfg_cond_br_true_dest(terminator) == target;
  bool is_false_edge = loom_cfg_cond_br_false_dest(terminator) == target;
  if (is_true_edge == is_false_edge) return false;
  *out_condition = loom_cfg_cond_br_condition(terminator);
  *out_value = is_true_edge;
  return true;
}

static bool loom_cfg_simplify_condition_operands_equal(
    loom_condition_integer_operand_t lhs,
    loom_condition_integer_operand_t rhs) {
  if (lhs.kind != rhs.kind) return false;
  switch (lhs.kind) {
    case LOOM_CONDITION_INTEGER_OPERAND_VALUE:
      return lhs.value_id == rhs.value_id;
    case LOOM_CONDITION_INTEGER_OPERAND_CONSTANT:
      return lhs.constant == rhs.constant;
    default:
      return false;
  }
}

static loom_condition_integer_operand_t
loom_cfg_simplify_condition_value_operand(loom_value_id_t value_id) {
  return (loom_condition_integer_operand_t){
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value_id,
      .constant = 0,
  };
}

static bool loom_cfg_simplify_condition_relation_implies(
    const loom_condition_integer_relation_t* known,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  if (loom_cfg_simplify_condition_operands_equal(known->left, queried->left) &&
      loom_cfg_simplify_condition_operands_equal(known->right,
                                                 queried->right)) {
    return loom_symbolic_integer_relation_implies(
        known->relation, queried->relation, out_result);
  }

  if (loom_cfg_simplify_condition_operands_equal(known->left, queried->right) &&
      loom_cfg_simplify_condition_operands_equal(known->right, queried->left)) {
    return loom_symbolic_integer_relation_implies(
        loom_symbolic_integer_relation_swap(known->relation), queried->relation,
        out_result);
  }

  return false;
}

static bool loom_cfg_simplify_condition_relations_equivalent(
    const loom_condition_integer_relation_t* lhs,
    const loom_condition_integer_relation_t* rhs) {
  bool lhs_implies_rhs = false;
  if (!loom_cfg_simplify_condition_relation_implies(lhs, rhs,
                                                    &lhs_implies_rhs) ||
      !lhs_implies_rhs) {
    return false;
  }

  bool rhs_implies_lhs = false;
  return loom_cfg_simplify_condition_relation_implies(rhs, lhs,
                                                      &rhs_implies_lhs) &&
         rhs_implies_lhs;
}

static bool loom_cfg_simplify_condition_relation_meet(
    const loom_condition_integer_relation_t* existing,
    const loom_condition_integer_relation_t* edge,
    loom_condition_integer_relation_t* out_relation) {
  bool implication_result = false;
  if (loom_cfg_simplify_condition_relation_implies(edge, existing,
                                                   &implication_result) &&
      implication_result) {
    *out_relation = *existing;
    return true;
  }

  if (loom_cfg_simplify_condition_relation_implies(existing, edge,
                                                   &implication_result) &&
      implication_result) {
    *out_relation = *edge;
    return true;
  }

  return false;
}

static void loom_cfg_simplify_append_condition_relation(
    loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_capacity, iree_host_size_t* inout_relation_count,
    loom_condition_integer_relation_t relation) {
  for (iree_host_size_t i = 0; i < *inout_relation_count; ++i) {
    if (loom_cfg_simplify_condition_relations_equivalent(&relations[i],
                                                         &relation)) {
      return;
    }
  }
  if (*inout_relation_count >= relation_capacity) {
    return;
  }
  relations[(*inout_relation_count)++] = relation;
}

static iree_host_size_t loom_cfg_simplify_intersect_condition_relations(
    loom_condition_integer_relation_t* inout_relations,
    iree_host_size_t inout_relation_count,
    const loom_condition_integer_relation_t* edge_relations,
    iree_host_size_t edge_relation_count) {
  iree_host_size_t new_relation_count = 0;
  for (iree_host_size_t existing_index = 0;
       existing_index < inout_relation_count; ++existing_index) {
    loom_condition_integer_relation_t common_relation = {0};
    bool found_common_relation = false;
    for (iree_host_size_t edge_index = 0; edge_index < edge_relation_count;
         ++edge_index) {
      if (!loom_cfg_simplify_condition_relation_meet(
              &inout_relations[existing_index], &edge_relations[edge_index],
              &common_relation)) {
        continue;
      }
      found_common_relation = true;
      break;
    }
    if (!found_common_relation) continue;
    inout_relations[new_relation_count++] = common_relation;
  }
  return new_relation_count;
}

static iree_status_t loom_cfg_simplify_copy_condition_relations(
    loom_cfg_simplify_state_t* state,
    const loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_count,
    loom_cfg_simplify_block_entry_path_facts_t* fact) {
  if (relation_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->analysis_arena, relation_count, sizeof(*fact->integer_relations),
      (void**)&fact->integer_relations));
  memcpy(fact->integer_relations, relations,
         relation_count * sizeof(*fact->integer_relations));
  fact->integer_relation_count = relation_count;
  return iree_ok_status();
}

static bool loom_cfg_simplify_block_entry_path_facts_equal(
    const loom_cfg_simplify_block_entry_path_facts_t* lhs,
    const loom_cfg_simplify_block_entry_path_facts_t* rhs) {
  if (lhs->condition != rhs->condition ||
      lhs->condition_value != rhs->condition_value ||
      lhs->condition_known != rhs->condition_known ||
      lhs->integer_relation_count != rhs->integer_relation_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < lhs->integer_relation_count; ++i) {
    if (!loom_cfg_simplify_condition_relations_equivalent(
            &lhs->integer_relations[i], &rhs->integer_relations[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_value_available_at_block_entry(
    const loom_cfg_simplify_state_t* state, loom_value_id_t value_id,
    const loom_block_t* block) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count || !block || !block->first_op) {
    return false;
  }
  return loom_value_is_available_before_op(state->dominance, value_id,
                                           block->first_op) &&
         loom_value_type_is_available_before_op(state->dominance, value_id,
                                                block->first_op);
}

static bool loom_cfg_simplify_try_map_branch_arg_to_block_arg(
    const loom_block_t* block, loom_value_slice_t branch_args,
    loom_value_id_t value_id, loom_value_id_t* out_block_arg,
    bool* out_ambiguous) {
  *out_block_arg = LOOM_VALUE_ID_INVALID;
  *out_ambiguous = false;
  if (!block || branch_args.count != block->arg_count) return false;

  bool found_mapping = false;
  for (uint16_t i = 0; i < branch_args.count; ++i) {
    if (branch_args.values[i] != value_id) continue;
    if (found_mapping) {
      *out_ambiguous = true;
      *out_block_arg = LOOM_VALUE_ID_INVALID;
      return false;
    }
    *out_block_arg = loom_block_arg_id(block, i);
    found_mapping = true;
  }
  return found_mapping;
}

static bool loom_cfg_simplify_remap_condition_operand_to_block_entry(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* predecessor_terminator, loom_condition_integer_operand_t operand,
    loom_condition_integer_operand_t* out_operand) {
  *out_operand = operand;
  if (operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) return true;

  if (loom_cfg_br_isa(predecessor_terminator) &&
      loom_cfg_br_dest(predecessor_terminator) == block) {
    loom_value_id_t block_arg = LOOM_VALUE_ID_INVALID;
    bool ambiguous_mapping = false;
    bool found_mapping = loom_cfg_simplify_try_map_branch_arg_to_block_arg(
        block, loom_cfg_br_args(predecessor_terminator), operand.value_id,
        &block_arg, &ambiguous_mapping);
    if (ambiguous_mapping) return false;
    if (found_mapping) {
      *out_operand = loom_cfg_simplify_condition_value_operand(block_arg);
      return true;
    }
  }

  if (!loom_cfg_simplify_value_available_at_block_entry(state, operand.value_id,
                                                        block)) {
    return false;
  }
  return true;
}

static bool loom_cfg_simplify_remap_value_to_block_entry(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* predecessor_terminator, loom_value_id_t value_id,
    loom_value_id_t* out_value_id) {
  *out_value_id = value_id;
  if (loom_cfg_br_isa(predecessor_terminator) &&
      loom_cfg_br_dest(predecessor_terminator) == block) {
    loom_value_id_t block_arg = LOOM_VALUE_ID_INVALID;
    bool ambiguous_mapping = false;
    bool found_mapping = loom_cfg_simplify_try_map_branch_arg_to_block_arg(
        block, loom_cfg_br_args(predecessor_terminator), value_id, &block_arg,
        &ambiguous_mapping);
    if (ambiguous_mapping) return false;
    if (found_mapping) {
      *out_value_id = block_arg;
      return true;
    }
  }
  return loom_cfg_simplify_value_available_at_block_entry(state, value_id,
                                                          block);
}

static bool loom_cfg_simplify_remap_condition_relation_to_block_entry(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* predecessor_terminator,
    const loom_condition_integer_relation_t* relation,
    loom_condition_integer_relation_t* out_relation) {
  *out_relation = *relation;
  if (!loom_cfg_simplify_remap_condition_operand_to_block_entry(
          state, block, predecessor_terminator, relation->left,
          &out_relation->left)) {
    return false;
  }
  if (!loom_cfg_simplify_remap_condition_operand_to_block_entry(
          state, block, predecessor_terminator, relation->right,
          &out_relation->right)) {
    return false;
  }
  return true;
}

static void loom_cfg_simplify_append_remapped_condition_relations(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* predecessor_terminator,
    const loom_condition_integer_relation_t* source_relations,
    iree_host_size_t source_relation_count,
    loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_capacity,
    iree_host_size_t* inout_relation_count) {
  for (iree_host_size_t i = 0; i < source_relation_count; ++i) {
    loom_condition_integer_relation_t remapped_relation = {0};
    if (!loom_cfg_simplify_remap_condition_relation_to_block_entry(
            state, block, predecessor_terminator, &source_relations[i],
            &remapped_relation)) {
      continue;
    }
    loom_cfg_simplify_append_condition_relation(
        relations, relation_capacity, inout_relation_count, remapped_relation);
  }
}

static bool loom_cfg_simplify_set_edge_bool_fact(
    loom_value_id_t condition, bool value, loom_value_id_t* inout_condition,
    bool* inout_value, bool* inout_known) {
  if (!*inout_known) {
    *inout_condition = condition;
    *inout_value = value;
    *inout_known = true;
    return true;
  }
  return *inout_condition == condition && *inout_value == value;
}

static void loom_cfg_simplify_compute_predecessor_edge_path_facts(
    loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* predecessor_terminator, uint16_t predecessor_index,
    const loom_cfg_simplify_block_entry_path_facts_t* current_facts,
    loom_condition_integer_relation_t* relation_storage,
    iree_host_size_t relation_capacity,
    loom_cfg_simplify_block_entry_path_facts_t* out_fact) {
  *out_fact = (loom_cfg_simplify_block_entry_path_facts_t){
      .condition = LOOM_VALUE_ID_INVALID,
      .integer_relations = relation_storage,
  };

  loom_value_id_t edge_condition = LOOM_VALUE_ID_INVALID;
  bool edge_value = false;
  bool edge_has_condition = loom_cfg_simplify_edge_implies_bool(
      block, predecessor_terminator, &edge_condition, &edge_value);
  if (edge_has_condition) {
    out_fact->condition = edge_condition;
    out_fact->condition_value = edge_value;
    out_fact->condition_known = true;
  } else if (current_facts &&
             current_facts[predecessor_index].condition_known) {
    loom_value_id_t remapped_condition = LOOM_VALUE_ID_INVALID;
    if (loom_cfg_simplify_remap_value_to_block_entry(
            state, block, predecessor_terminator,
            current_facts[predecessor_index].condition, &remapped_condition)) {
      out_fact->condition = remapped_condition;
      out_fact->condition_value =
          current_facts[predecessor_index].condition_value;
      out_fact->condition_known = true;
    }
  }

  loom_condition_fact_set_t edge_facts;
  loom_condition_fact_set_initialize(relation_storage, relation_capacity,
                                     &edge_facts);
  if (edge_has_condition) {
    loom_condition_facts_query(state->module, state->fact_table, edge_condition,
                               edge_value, &edge_facts);
  }
  if (current_facts) {
    const loom_cfg_simplify_block_entry_path_facts_t* predecessor_facts =
        &current_facts[predecessor_index];
    loom_cfg_simplify_append_remapped_condition_relations(
        state, block, predecessor_terminator,
        predecessor_facts->integer_relations,
        predecessor_facts->integer_relation_count, relation_storage,
        relation_capacity, &edge_facts.integer_relation_count);
  }

  out_fact->integer_relation_count = edge_facts.integer_relation_count;
}

static void loom_cfg_simplify_compute_block_entry_path_facts(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_simplify_block_entry_path_facts_t* current_facts,
    uint16_t block_index, loom_condition_integer_relation_t* relation_storage,
    loom_cfg_simplify_block_entry_path_facts_t* fact) {
  *fact = (loom_cfg_simplify_block_entry_path_facts_t){
      .condition = LOOM_VALUE_ID_INVALID,
  };
  // The region entry also has an implicit caller edge with no CFG fact.
  if (block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return;
  }

  loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
  if (!block) return;

  bool saw_reachable_predecessor = false;
  bool condition_candidate_initialized = false;
  bool condition_candidate_valid = true;
  iree_host_size_t relation_count = 0;
  bool relation_meet_initialized = false;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    uint16_t predecessor_index = predecessors.values[i];
    if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) continue;
    loom_block_t* predecessor =
        (loom_block_t*)graph->blocks[predecessor_index].block;
    if (!predecessor) return;
    saw_reachable_predecessor = true;

    loom_condition_integer_relation_t
        edge_relation_storage[LOOM_CFG_SIMPLIFY_EDGE_RELATION_CAPACITY];
    loom_cfg_simplify_block_entry_path_facts_t edge_fact = {0};
    loom_cfg_simplify_compute_predecessor_edge_path_facts(
        state, block, predecessor->last_op, predecessor_index, current_facts,
        edge_relation_storage, IREE_ARRAYSIZE(edge_relation_storage),
        &edge_fact);
    if (!edge_fact.condition_known ||
        !loom_cfg_simplify_set_edge_bool_fact(
            edge_fact.condition, edge_fact.condition_value, &fact->condition,
            &fact->condition_value, &condition_candidate_initialized)) {
      condition_candidate_valid = false;
    }

    if (!relation_meet_initialized) {
      relation_count = edge_fact.integer_relation_count;
      memcpy(relation_storage, edge_fact.integer_relations,
             relation_count * sizeof(relation_storage[0]));
      relation_meet_initialized = true;
      continue;
    }

    relation_count = loom_cfg_simplify_intersect_condition_relations(
        relation_storage, relation_count, edge_fact.integer_relations,
        edge_fact.integer_relation_count);
  }

  if (!saw_reachable_predecessor) return;
  fact->condition_known =
      condition_candidate_valid && condition_candidate_initialized;
  if (!fact->condition_known) {
    fact->condition = LOOM_VALUE_ID_INVALID;
  }
  fact->integer_relations = relation_storage;
  fact->integer_relation_count = relation_count;
}

static iree_status_t loom_cfg_simplify_compute_block_entry_path_fact_table(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    loom_cfg_simplify_block_entry_path_facts_t** out_facts) {
  *out_facts = NULL;
  if (graph->malformed || graph->block_count == 0) return iree_ok_status();

  loom_cfg_simplify_block_entry_path_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->analysis_arena, graph->block_count,
                                sizeof(*facts), (void**)&facts));
  memset(facts, 0, graph->block_count * sizeof(*facts));

  iree_host_size_t max_iterations = (iree_host_size_t)graph->block_count + 1;
  for (iree_host_size_t iteration = 0; iteration < max_iterations;
       ++iteration) {
    bool changed = false;
    for (uint16_t block_index = 0; block_index < graph->block_count;
         ++block_index) {
      loom_condition_integer_relation_t
          relation_storage[LOOM_CFG_SIMPLIFY_EDGE_RELATION_CAPACITY];
      loom_cfg_simplify_block_entry_path_facts_t computed_facts = {0};
      loom_cfg_simplify_compute_block_entry_path_facts(
          state, graph, facts, block_index, relation_storage, &computed_facts);
      if (loom_cfg_simplify_block_entry_path_facts_equal(&facts[block_index],
                                                         &computed_facts)) {
        continue;
      }
      loom_cfg_simplify_block_entry_path_facts_t copied_facts = computed_facts;
      copied_facts.integer_relations = NULL;
      copied_facts.integer_relation_count = 0;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_copy_condition_relations(
          state, computed_facts.integer_relations,
          computed_facts.integer_relation_count, &copied_facts));
      facts[block_index] = copied_facts;
      changed = true;
    }
    if (!changed) {
      *out_facts = facts;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_cfg_simplify_path_fact_dominates_op(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, loom_op_t* op) {
  if (!loom_cfg_graph_block_is_reachable(graph, block_index)) return false;
  loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
  if (!block || !block->first_op) return false;
  return loom_dominates_op(state->dominance, block->first_op, op);
}

static bool loom_cfg_simplify_dominating_exact_bool(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_simplify_block_entry_path_facts_t* facts, loom_op_t* op,
    loom_value_id_t condition, bool* out_value) {
  if (!facts) return false;
  bool found_fact = false;
  bool dominating_value = false;
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    const loom_cfg_simplify_block_entry_path_facts_t* fact =
        &facts[block_index];
    if (!fact->condition_known || fact->condition != condition) continue;
    if (!loom_cfg_simplify_path_fact_dominates_op(state, graph, block_index,
                                                  op)) {
      continue;
    }

    if (!found_fact) {
      found_fact = true;
      dominating_value = fact->condition_value;
      continue;
    }
    if (dominating_value != fact->condition_value) return false;
  }

  if (!found_fact) return false;
  *out_value = dominating_value;
  return true;
}

static bool loom_cfg_simplify_dominating_relation_proves(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_simplify_block_entry_path_facts_t* facts, loom_op_t* op,
    const loom_condition_integer_relation_t* queried_relation,
    bool* out_result) {
  if (!facts) return false;
  bool found_relation = false;
  bool proven_result = false;
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    const loom_cfg_simplify_block_entry_path_facts_t* fact =
        &facts[block_index];
    if (fact->integer_relation_count == 0) continue;
    if (!loom_cfg_simplify_path_fact_dominates_op(state, graph, block_index,
                                                  op)) {
      continue;
    }

    for (iree_host_size_t i = 0; i < fact->integer_relation_count; ++i) {
      bool relation_result = false;
      if (!loom_cfg_simplify_condition_relation_implies(
              &fact->integer_relations[i], queried_relation,
              &relation_result)) {
        continue;
      }
      if (!found_relation) {
        found_relation = true;
        proven_result = relation_result;
        continue;
      }
      if (proven_result != relation_result) return false;
    }
  }

  if (!found_relation) return false;
  *out_result = proven_result;
  return true;
}

static bool loom_cfg_simplify_entry_relation_proves(
    const loom_cfg_simplify_block_entry_path_facts_t* facts,
    const loom_condition_integer_relation_t* queried_relation,
    bool* out_result) {
  bool found_relation = false;
  bool proven_result = false;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    bool relation_result = false;
    if (!loom_cfg_simplify_condition_relation_implies(
            &facts->integer_relations[i], queried_relation, &relation_result)) {
      continue;
    }
    if (!found_relation) {
      found_relation = true;
      proven_result = relation_result;
      continue;
    }
    if (proven_result != relation_result) return false;
  }

  if (!found_relation) return false;
  *out_result = proven_result;
  return true;
}

static void loom_cfg_simplify_prove_condition_fact_set(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_simplify_block_entry_path_facts_t* facts, loom_op_t* op,
    const loom_condition_fact_set_t* queried_facts, bool* out_proven,
    bool* out_contradicted) {
  *out_proven = false;
  *out_contradicted = false;
  if (queried_facts->integer_relation_count == 0) return;

  *out_proven = true;
  for (iree_host_size_t i = 0; i < queried_facts->integer_relation_count; ++i) {
    bool relation_result = false;
    if (!loom_cfg_simplify_dominating_relation_proves(
            state, graph, facts, op, &queried_facts->integer_relations[i],
            &relation_result)) {
      *out_proven = false;
      continue;
    }
    if (!relation_result) {
      *out_proven = false;
      *out_contradicted = true;
      return;
    }
  }
}

static void loom_cfg_simplify_entry_facts_prove_condition_fact_set(
    const loom_cfg_simplify_block_entry_path_facts_t* facts,
    const loom_condition_fact_set_t* queried_facts, bool* out_proven,
    bool* out_contradicted) {
  *out_proven = false;
  *out_contradicted = false;
  if (queried_facts->integer_relation_count == 0) return;

  *out_proven = true;
  for (iree_host_size_t i = 0; i < queried_facts->integer_relation_count; ++i) {
    bool relation_result = false;
    if (!loom_cfg_simplify_entry_relation_proves(
            facts, &queried_facts->integer_relations[i], &relation_result)) {
      *out_proven = false;
      continue;
    }
    if (!relation_result) {
      *out_proven = false;
      *out_contradicted = true;
      return;
    }
  }
}

static void loom_cfg_simplify_dominating_relations_prove_bool(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_simplify_block_entry_path_facts_t* facts, loom_op_t* op,
    loom_value_id_t condition, bool* out_value, bool* out_proven) {
  *out_value = false;
  *out_proven = false;

  loom_condition_integer_relation_t
      true_relation_storage[LOOM_CFG_SIMPLIFY_EDGE_RELATION_CAPACITY];
  loom_condition_fact_set_t true_facts;
  loom_condition_fact_set_initialize(true_relation_storage,
                                     IREE_ARRAYSIZE(true_relation_storage),
                                     &true_facts);
  bool true_facts_complete = loom_condition_facts_query(
      state->module, state->fact_table, condition, true, &true_facts);

  loom_condition_integer_relation_t
      false_relation_storage[LOOM_CFG_SIMPLIFY_EDGE_RELATION_CAPACITY];
  loom_condition_fact_set_t false_facts;
  loom_condition_fact_set_initialize(false_relation_storage,
                                     IREE_ARRAYSIZE(false_relation_storage),
                                     &false_facts);
  bool false_facts_complete = loom_condition_facts_query(
      state->module, state->fact_table, condition, false, &false_facts);

  bool true_proven = false;
  bool true_contradicted = false;
  if (true_facts_complete) {
    loom_cfg_simplify_prove_condition_fact_set(
        state, graph, facts, op, &true_facts, &true_proven, &true_contradicted);
  }

  bool false_proven = false;
  bool false_contradicted = false;
  if (false_facts_complete) {
    loom_cfg_simplify_prove_condition_fact_set(state, graph, facts, op,
                                               &false_facts, &false_proven,
                                               &false_contradicted);
  }

  bool proves_true = true_proven || false_contradicted;
  bool proves_false = false_proven || true_contradicted;
  if (proves_true == proves_false) return;
  *out_value = proves_true;
  *out_proven = true;
}

static void loom_cfg_simplify_entry_facts_prove_bool(
    loom_cfg_simplify_state_t* state,
    const loom_cfg_simplify_block_entry_path_facts_t* facts,
    loom_value_id_t condition, bool* out_value, bool* out_proven) {
  *out_value = false;
  *out_proven = false;
  if (facts->condition_known && facts->condition == condition) {
    *out_value = facts->condition_value;
    *out_proven = true;
    return;
  }

  loom_condition_integer_relation_t
      true_relation_storage[LOOM_CFG_SIMPLIFY_EDGE_RELATION_CAPACITY];
  loom_condition_fact_set_t true_facts;
  loom_condition_fact_set_initialize(true_relation_storage,
                                     IREE_ARRAYSIZE(true_relation_storage),
                                     &true_facts);
  bool true_facts_complete = loom_condition_facts_query(
      state->module, state->fact_table, condition, true, &true_facts);

  loom_condition_integer_relation_t
      false_relation_storage[LOOM_CFG_SIMPLIFY_EDGE_RELATION_CAPACITY];
  loom_condition_fact_set_t false_facts;
  loom_condition_fact_set_initialize(false_relation_storage,
                                     IREE_ARRAYSIZE(false_relation_storage),
                                     &false_facts);
  bool false_facts_complete = loom_condition_facts_query(
      state->module, state->fact_table, condition, false, &false_facts);

  bool true_proven = false;
  bool true_contradicted = false;
  if (true_facts_complete) {
    loom_cfg_simplify_entry_facts_prove_condition_fact_set(
        facts, &true_facts, &true_proven, &true_contradicted);
  }

  bool false_proven = false;
  bool false_contradicted = false;
  if (false_facts_complete) {
    loom_cfg_simplify_entry_facts_prove_condition_fact_set(
        facts, &false_facts, &false_proven, &false_contradicted);
  }

  bool proves_true = true_proven || false_contradicted;
  bool proves_false = false_proven || true_contradicted;
  if (proves_true == proves_false) return;
  *out_value = proves_true;
  *out_proven = true;
}

static void loom_cfg_simplify_path_facts_prove_bool(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_simplify_block_entry_path_facts_t* facts, loom_op_t* op,
    loom_value_id_t condition_value, bool* out_value, bool* out_proven) {
  *out_value = false;
  *out_proven = false;
  if (loom_cfg_simplify_dominating_exact_bool(state, graph, facts, op,
                                              condition_value, out_value)) {
    *out_proven = true;
    return;
  }
  loom_cfg_simplify_dominating_relations_prove_bool(
      state, graph, facts, op, condition_value, out_value, out_proven);
}

static iree_status_t loom_cfg_simplify_fold_path_sensitive_cond_br(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_simplify_block_entry_path_facts_t* facts, loom_op_t* op,
    bool* out_changed) {
  if (!loom_cfg_cond_br_isa(op)) return iree_ok_status();

  loom_value_id_t condition_value = loom_cfg_cond_br_condition(op);
  bool condition = false;
  bool condition_proven = false;
  loom_cfg_simplify_path_facts_prove_bool(
      state, graph, facts, op, condition_value, &condition, &condition_proven);
  if (!condition_proven) return iree_ok_status();

  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(op);
  loom_block_t* chosen_dest = condition ? true_dest : false_dest;
  if (!chosen_dest || chosen_dest->arg_count != 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_replace_cond_br_with_br(state, op, chosen_dest));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_thread_predecessor_to_block(
    loom_cfg_simplify_state_t* state, loom_block_t* old_dest,
    loom_block_t* new_dest, loom_op_t* predecessor_terminator,
    bool* out_changed) {
  if (!new_dest || new_dest->arg_count != 0) return iree_ok_status();
  if (loom_cfg_br_isa(predecessor_terminator)) {
    if (loom_cfg_br_dest(predecessor_terminator) != old_dest) {
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
  bool changed = false;
  for (uint8_t successor_index = 0;
       successor_index < predecessor_terminator->successor_count;
       ++successor_index) {
    if (successors[successor_index] != old_dest) continue;
    successors[successor_index] = new_dest;
    changed = true;
  }
  if (!changed) return iree_ok_status();
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
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    loom_op_t* user = loom_use_user_op(*use);
    if (!user || user->parent_block != block) return false;
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
    const loom_cfg_simplify_block_entry_path_facts_t* facts,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    loom_op_t* terminator = block ? block->last_op : NULL;
    if (!terminator || !loom_cfg_cond_br_isa(terminator) ||
        !loom_cfg_simplify_can_skip_block_prefix(state, block, terminator)) {
      continue;
    }

    loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      uint16_t predecessor_index = predecessors.values[i];
      if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index) ||
          predecessor_index == block_index) {
        continue;
      }
      loom_block_t* predecessor =
          (loom_block_t*)graph->blocks[predecessor_index].block;
      if (!predecessor || !predecessor->last_op) continue;
      if (block->first_op &&
          loom_dominates_op(state->dominance, block->first_op,
                            predecessor->last_op)) {
        continue;
      }

      loom_condition_integer_relation_t
          edge_relation_storage[LOOM_CFG_SIMPLIFY_EDGE_RELATION_CAPACITY];
      loom_cfg_simplify_block_entry_path_facts_t edge_facts = {0};
      loom_cfg_simplify_compute_predecessor_edge_path_facts(
          state, block, predecessor->last_op, predecessor_index, facts,
          edge_relation_storage, IREE_ARRAYSIZE(edge_relation_storage),
          &edge_facts);

      bool condition = false;
      bool condition_proven = false;
      loom_cfg_simplify_entry_facts_prove_bool(
          state, &edge_facts, loom_cfg_cond_br_condition(terminator),
          &condition, &condition_proven);
      if (!condition_proven) continue;

      loom_block_t* new_dest = condition
                                   ? loom_cfg_cond_br_true_dest(terminator)
                                   : loom_cfg_cond_br_false_dest(terminator);
      if (!new_dest || new_dest->arg_count != 0) continue;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_thread_predecessor_to_block(
          state, block, new_dest, predecessor->last_op, out_changed));
      if (*out_changed) return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_fold_path_sensitive_branches(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    const loom_cfg_simplify_block_entry_path_facts_t* facts,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (!block) continue;

    loom_op_t* op = block->first_op;
    while (op) {
      loom_op_t* next_op = op->next_op;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_fold_path_sensitive_cond_br(
          state, graph, facts, op, out_changed));
      if (*out_changed) return iree_ok_status();
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
    const loom_cfg_simplify_block_entry_path_facts_t* facts,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (!block) continue;

    loom_op_t* op = block->first_op;
    while (op) {
      loom_op_t* next_op = op->next_op;
      if (op->result_count == 1) {
        loom_value_id_t result = loom_op_const_results(op)[0];
        if (loom_cfg_simplify_can_replace_with_constant(state, op, result)) {
          bool value = false;
          bool proven = false;
          loom_cfg_simplify_path_facts_prove_bool(state, graph, facts, op,
                                                  result, &value, &proven);
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
// Trivial block forwarding
//===----------------------------------------------------------------------===//

static bool loom_cfg_simplify_match_trivial_forward_block(
    const loom_block_t* block, loom_block_t** out_dest,
    loom_value_slice_t* out_args) {
  *out_dest = NULL;
  *out_args = (loom_value_slice_t){0};
  if (!block || block->first_op != block->last_op || !block->first_op ||
      !loom_cfg_br_isa(block->first_op)) {
    return false;
  }
  loom_block_t* dest = loom_cfg_br_dest(block->first_op);
  if (!dest || dest == block) return false;
  *out_dest = dest;
  *out_args = loom_cfg_br_args(block->first_op);
  if (out_args->count != dest->arg_count) return false;
  return true;
}

static bool loom_cfg_simplify_map_forward_arg(
    const loom_cfg_simplify_state_t* state, const loom_block_t* forward_block,
    loom_value_slice_t predecessor_args, loom_value_id_t forward_arg,
    loom_value_id_t* out_arg) {
  *out_arg = LOOM_VALUE_ID_INVALID;
  if (forward_arg == LOOM_VALUE_ID_INVALID ||
      forward_arg >= state->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(state->module, forward_arg);
  if (!loom_value_is_block_arg(value) ||
      loom_value_def_block(value) != forward_block) {
    *out_arg = forward_arg;
    return true;
  }

  uint16_t arg_index = loom_value_def_index(value);
  if (arg_index >= predecessor_args.count) return false;
  *out_arg = predecessor_args.values[arg_index];
  return *out_arg != LOOM_VALUE_ID_INVALID;
}

static bool loom_cfg_simplify_branch_args_available_before(
    const loom_cfg_simplify_state_t* state, loom_value_slice_t args,
    const loom_op_t* before_op) {
  for (uint16_t i = 0; i < args.count; ++i) {
    loom_value_id_t arg = args.values[i];
    if (!loom_value_is_available_before_op(state->dominance, arg, before_op) ||
        !loom_value_type_is_available_before_op(state->dominance, arg,
                                                before_op)) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_branch_args_match_dest(
    const loom_cfg_simplify_state_t* state, const loom_block_t* dest,
    loom_value_slice_t args) {
  if (!dest || args.count != dest->arg_count) return false;
  loom_type_value_remap_t remap = {
      .source_values = dest->arg_ids,
      .target_values = args.values,
      .count = dest->arg_count,
  };
  for (uint16_t i = 0; i < dest->arg_count; ++i) {
    loom_value_id_t actual_id = args.values[i];
    loom_value_id_t expected_id = loom_block_arg_id(dest, i);
    if (actual_id == LOOM_VALUE_ID_INVALID ||
        expected_id == LOOM_VALUE_ID_INVALID ||
        actual_id >= state->module->values.count ||
        expected_id >= state->module->values.count) {
      return false;
    }
    loom_type_t actual_type = loom_module_value_type(state->module, actual_id);
    loom_type_t expected_type =
        loom_module_value_type(state->module, expected_id);
    if (!loom_type_equal_after_value_remap(expected_type, actual_type,
                                           &remap)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_cfg_simplify_validate_block_arg_replacements(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* predecessor_br, loom_value_slice_t replacements,
    bool* out_valid) {
  *out_valid = false;
  if (replacements.count != block->arg_count) return iree_ok_status();
  if (block->arg_count == 0) {
    *out_valid = true;
    return iree_ok_status();
  }

  loom_value_id_t* old_args = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->analysis_arena, block->arg_count,
                                sizeof(*old_args), (void**)&old_args));
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    old_args[i] = loom_block_arg_id(block, i);
  }

  loom_type_value_remap_t remap = {
      .source_values = old_args,
      .target_values = replacements.values,
      .count = block->arg_count,
  };
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t old_arg = old_args[i];
    loom_value_id_t replacement = replacements.values[i];
    if (old_arg == LOOM_VALUE_ID_INVALID ||
        replacement == LOOM_VALUE_ID_INVALID ||
        old_arg >= state->module->values.count ||
        replacement >= state->module->values.count) {
      return iree_ok_status();
    }
    if (!loom_value_is_available_before_op(state->dominance, replacement,
                                           predecessor_br) ||
        !loom_value_type_is_available_before_op(state->dominance, replacement,
                                                predecessor_br)) {
      return iree_ok_status();
    }
    loom_type_t old_type = loom_module_value_type(state->module, old_arg);
    loom_type_t replacement_type =
        loom_module_value_type(state->module, replacement);
    if (!loom_type_equal_after_value_remap(old_type, replacement_type,
                                           &remap)) {
      return iree_ok_status();
    }
  }

  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_replace_block_args(
    loom_cfg_simplify_state_t* state, loom_block_t* block,
    loom_value_slice_t replacements) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        state->rewriter, loom_block_arg_id(block, i), replacements.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_compose_forward_args(
    const loom_cfg_simplify_state_t* state, const loom_block_t* forward_block,
    loom_op_t* predecessor_br, const loom_block_t* new_dest,
    loom_value_slice_t forward_args, loom_value_slice_t* out_args,
    bool* out_valid) {
  *out_args = (loom_value_slice_t){0};
  *out_valid = false;

  loom_value_slice_t predecessor_args = loom_cfg_br_args(predecessor_br);
  if (predecessor_args.count != forward_block->arg_count) {
    return iree_ok_status();
  }

  loom_value_id_t* composed_args = NULL;
  if (forward_args.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->analysis_arena, forward_args.count, sizeof(*composed_args),
        (void**)&composed_args));
  }
  for (uint16_t i = 0; i < forward_args.count; ++i) {
    if (!loom_cfg_simplify_map_forward_arg(
            state, forward_block, predecessor_args, forward_args.values[i],
            &composed_args[i])) {
      return iree_ok_status();
    }
  }

  loom_value_slice_t composed = {
      .values = composed_args,
      .count = forward_args.count,
  };
  if (!loom_cfg_simplify_branch_args_available_before(state, composed,
                                                      predecessor_br) ||
      !loom_cfg_simplify_branch_args_match_dest(state, new_dest, composed)) {
    return iree_ok_status();
  }

  *out_args = composed;
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_forward_branch_edge(
    loom_cfg_simplify_state_t* state, loom_op_t* predecessor_br,
    loom_block_t* old_dest, loom_block_t* new_dest, loom_value_slice_t new_args,
    bool* out_changed) {
  if (!loom_cfg_br_isa(predecessor_br) ||
      loom_cfg_br_dest(predecessor_br) != old_dest) {
    return iree_ok_status();
  }

  loom_value_slice_t composed_args = {0};
  bool valid_args = false;
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_compose_forward_args(
      state, old_dest, predecessor_br, new_dest, new_args, &composed_args,
      &valid_args));
  if (!valid_args) {
    return iree_ok_status();
  }
  if (old_dest->arg_count != 0) {
    loom_value_slice_t replacements = loom_cfg_br_args(predecessor_br);
    bool valid_replacements = false;
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_validate_block_arg_replacements(
        state, old_dest, predecessor_br, replacements, &valid_replacements));
    if (!valid_replacements) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_replace_block_args(state, old_dest, replacements));
  }
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_replace_br(state, predecessor_br, new_dest,
                                   composed_args.values, composed_args.count));
  ++state->statistics->edges_forwarded;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_forward_cond_br_edge(
    loom_cfg_simplify_state_t* state, loom_op_t* predecessor_cond_br,
    loom_block_t* old_dest, loom_block_t* new_dest, loom_value_slice_t new_args,
    bool* out_changed) {
  if (!loom_cfg_cond_br_isa(predecessor_cond_br) || old_dest->arg_count != 0 ||
      new_args.count != 0) {
    return iree_ok_status();
  }
  if (!new_dest || new_dest->arg_count != 0) return iree_ok_status();
  loom_block_t** successors = loom_op_successors(predecessor_cond_br);
  for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
    if (successors[successor_index] != old_dest) continue;
    successors[successor_index] = new_dest;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_to_worklist(state->rewriter, predecessor_cond_br));
    state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    ++state->statistics->edges_forwarded;
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_forward_trivial_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    loom_block_t* dest = NULL;
    loom_value_slice_t args = {0};
    if (!loom_cfg_simplify_match_trivial_forward_block(block, &dest, &args)) {
      continue;
    }

    loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    if (block->arg_count != 0 && predecessors.count > 1) {
      continue;
    }
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      const loom_cfg_block_info_t* predecessor_info =
          &graph->blocks[predecessors.values[i]];
      loom_op_t* terminator = ((loom_block_t*)predecessor_info->block)->last_op;
      if (!terminator) continue;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_forward_branch_edge(
          state, terminator, block, dest, args, out_changed));
      if (*out_changed) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_forward_cond_br_edge(
          state, terminator, block, dest, args, out_changed));
      if (*out_changed) return iree_ok_status();
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block fusion
//===----------------------------------------------------------------------===//

static bool loom_cfg_simplify_find_fusable_predecessor(
    const loom_cfg_graph_t* graph, uint16_t block_index,
    loom_op_t** out_predecessor_br, loom_value_slice_t* out_args) {
  *out_predecessor_br = NULL;
  *out_args = (loom_value_slice_t){0};
  if (block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return false;
  }

  const loom_block_t* block = graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  if (predecessors.count != 1) return false;

  const loom_block_t* predecessor = graph->blocks[predecessors.values[0]].block;
  if (!predecessor || predecessor == block) return false;

  loom_op_t* terminator = ((loom_block_t*)predecessor)->last_op;
  if (!terminator || !loom_cfg_br_isa(terminator) ||
      loom_cfg_br_dest(terminator) != block) {
    return false;
  }

  loom_value_slice_t args = loom_cfg_br_args(terminator);
  if (args.count != block->arg_count) return false;
  *out_predecessor_br = terminator;
  *out_args = args;
  return true;
}

static iree_status_t loom_cfg_simplify_move_block_ops_before(
    loom_cfg_simplify_state_t* state, loom_block_t* block,
    loom_op_t* before_op) {
  loom_op_t* op = block->first_op;
  while (op) {
    loom_op_t* next_op = op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(state->rewriter, op, before_op));
    op = next_op;
  }
  return iree_ok_status();
}

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
      (uint16_t)graph->block_count, &removed_count));
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_fuse_single_predecessor_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (!block || !block->first_op) continue;

    loom_op_t* predecessor_br = NULL;
    loom_value_slice_t replacements = {0};
    if (!loom_cfg_simplify_find_fusable_predecessor(
            graph, block_index, &predecessor_br, &replacements)) {
      continue;
    }

    bool valid_replacements = false;
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_validate_block_arg_replacements(
        state, block, predecessor_br, replacements, &valid_replacements));
    if (!valid_replacements) continue;

    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_replace_block_args(state, block, replacements));
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_move_block_ops_before(state, block, predecessor_br));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(state->rewriter, predecessor_br));
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_remove_cfg_block(state, graph, block_index));
    ++state->statistics->blocks_fused;
    state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Duplicate block merging
//===----------------------------------------------------------------------===//

#define LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_OPS 8
#define LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES 32

typedef struct loom_cfg_simplify_value_map_t {
  loom_value_id_t source_values[LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES];
  loom_value_id_t target_values[LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES];
  iree_host_size_t count;
} loom_cfg_simplify_value_map_t;

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
    if (!loom_attribute_equal(&lhs_attrs[i], &rhs_attrs[i])) return false;
  }
  return true;
}

static bool loom_cfg_simplify_find_duplicate_terminal_block(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, loom_block_t** out_canonical_block) {
  *out_canonical_block = NULL;
  if (!loom_cfg_simplify_is_mergeable_terminal_block(state, graph,
                                                     block_index)) {
    return false;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  const loom_op_t* terminator = block->first_op;
  for (uint16_t canonical_index = 1; canonical_index < block_index;
       ++canonical_index) {
    if (!loom_cfg_simplify_is_mergeable_terminal_block(state, graph,
                                                       canonical_index)) {
      continue;
    }
    loom_block_t* canonical_block =
        (loom_block_t*)graph->blocks[canonical_index].block;
    if (!loom_cfg_simplify_terminal_ops_equal(canonical_block->first_op,
                                              terminator)) {
      continue;
    }
    *out_canonical_block = canonical_block;
    return true;
  }
  return false;
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
    if (map->target_values[i] == target_value) return false;
  }
  if (map->count >= LOOM_CFG_SIMPLIFY_ALPHA_EQUIV_MAX_VALUES) return false;
  map->source_values[map->count] = source_value;
  map->target_values[map->count] = target_value;
  ++map->count;
  return true;
}

static loom_value_id_t loom_cfg_simplify_value_map_lookup(
    const loom_cfg_simplify_value_map_t* map, loom_value_id_t source_value) {
  for (iree_host_size_t i = 0; i < map->count; ++i) {
    if (map->source_values[i] == source_value) return map->target_values[i];
  }
  return source_value;
}

static bool loom_cfg_simplify_values_equal_after_map(
    const loom_cfg_simplify_value_map_t* map, loom_value_id_t source_value,
    loom_value_id_t target_value) {
  return loom_cfg_simplify_value_map_lookup(map, source_value) == target_value;
}

static bool loom_cfg_simplify_types_equal_after_map(
    const loom_cfg_simplify_value_map_t* map, loom_type_t source_type,
    loom_type_t target_type) {
  loom_type_value_remap_t remap = {
      .source_values = map->source_values,
      .target_values = map->target_values,
      .count = map->count,
  };
  return loom_type_equal_after_value_remap(source_type, target_type, &remap);
}

static bool loom_cfg_simplify_op_is_alpha_mergeable(
    const loom_cfg_simplify_state_t* state, const loom_op_t* op) {
  if (!op || op->region_count != 0 || op->tied_result_count != 0) {
    return false;
  }
  loom_trait_flags_t traits = loom_op_effective_traits(state->module, op);
  if (!iree_all_bits_set(traits, LOOM_TRAIT_PURE)) return false;
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
    if (!loom_cfg_simplify_op_is_alpha_mergeable(state, op)) return false;
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
    if (source_successor != target_successor) return false;
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
    if (!loom_cfg_simplify_types_equal_after_map(map, source_type,
                                                 target_type)) {
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
  if (source_block->arg_count != target_block->arg_count) return false;
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
    if (!loom_cfg_simplify_types_equal_after_map(map, source_type,
                                                 target_type)) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_blocks_alpha_equivalent(
    const loom_cfg_simplify_state_t* state, const loom_block_t* source_block,
    const loom_block_t* target_block) {
  if (source_block->op_count != target_block->op_count) return false;

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

static bool loom_cfg_simplify_find_alpha_equivalent_block(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, loom_block_t** out_canonical_block) {
  *out_canonical_block = NULL;
  if (!loom_cfg_simplify_is_alpha_merge_candidate(state, graph, block_index)) {
    return false;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  for (uint16_t canonical_index = 1; canonical_index < block_index;
       ++canonical_index) {
    if (!loom_cfg_simplify_is_alpha_merge_candidate(state, graph,
                                                    canonical_index)) {
      continue;
    }
    loom_block_t* canonical_block =
        (loom_block_t*)graph->blocks[canonical_index].block;
    if (!loom_cfg_simplify_blocks_alpha_equivalent(state, block,
                                                   canonical_block)) {
      continue;
    }
    *out_canonical_block = canonical_block;
    return true;
  }
  return false;
}

static bool loom_cfg_simplify_can_redirect_successor(
    const loom_cfg_simplify_state_t* state, const loom_op_t* terminator,
    const loom_block_t* old_dest, const loom_block_t* new_dest) {
  if (!new_dest) return false;
  if (loom_cfg_br_isa(terminator)) {
    loom_value_slice_t args = loom_cfg_br_args((loom_op_t*)terminator);
    return loom_cfg_br_dest(terminator) == old_dest &&
           args.count == old_dest->arg_count &&
           args.count == new_dest->arg_count &&
           loom_cfg_simplify_branch_args_available_before(state, args,
                                                          terminator) &&
           loom_cfg_simplify_branch_args_match_dest(state, new_dest, args);
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
  if (predecessors.count == 0) return false;
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const loom_block_t* predecessor =
        graph->blocks[predecessors.values[i]].block;
    if (!predecessor) return false;
    if (predecessor == old_dest || predecessor == new_dest) return false;
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
    if (successors[successor_index] != old_dest) continue;
    successors[successor_index] = new_dest;
    changed = true;
  }
  if (!changed) return iree_ok_status();
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
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* canonical_block = NULL;
    if (!loom_cfg_simplify_find_alpha_equivalent_block(
            state, graph, block_index, &canonical_block)) {
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
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* canonical_block = NULL;
    if (!loom_cfg_simplify_find_duplicate_terminal_block(
            state, graph, block_index, &canonical_block)) {
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
  return loom_type_equal_after_value_remap(old_type, replacement_type, &remap);
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
    if (args.count != block->arg_count) return false;
    pred_branches[i] = terminator;
  }
  return true;
}

static bool loom_cfg_simplify_incoming_slots_match(
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t lhs_index, uint16_t rhs_index) {
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    loom_block_t* dest = NULL;
    loom_value_slice_t args = {0};
    if (!loom_cfg_simplify_direct_branch(pred_branches[i], &dest, &args)) {
      return false;
    }
    if (args.values[lhs_index] != args.values[rhs_index]) return false;
  }
  return true;
}

static bool loom_cfg_simplify_find_duplicate_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t arg_index, loom_value_id_t* out_replacement) {
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  for (uint16_t candidate_index = 0; candidate_index < arg_index;
       ++candidate_index) {
    loom_value_id_t candidate = loom_block_arg_id(block, candidate_index);
    if (!loom_cfg_simplify_incoming_slots_match(
            pred_branches, predecessor_count, arg_index, candidate_index)) {
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
    uint16_t arg_index, loom_value_id_t* out_replacement) {
  if (predecessor_count == 0) return false;
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    loom_block_t* dest = NULL;
    loom_value_slice_t args = {0};
    if (!loom_cfg_simplify_direct_branch(pred_branches[i], &dest, &args)) {
      return false;
    }
    loom_value_id_t incoming = args.values[arg_index];
    if (incoming == old_arg) continue;
    if (replacement == LOOM_VALUE_ID_INVALID) {
      replacement = incoming;
      continue;
    }
    if (incoming != replacement) {
      return false;
    }
  }
  if (replacement == LOOM_VALUE_ID_INVALID) return false;
  if (!loom_cfg_simplify_type_allows_replacement(state->module, old_arg,
                                                 replacement)) {
    return false;
  }

  const loom_op_t* anchor = block->first_op ? block->first_op : block->last_op;
  if (!anchor) return false;
  if (!loom_value_is_available_before_op(state->dominance, replacement,
                                         anchor) ||
      !loom_value_type_is_available_before_op(state->dominance, replacement,
                                              anchor)) {
    return false;
  }
  *out_replacement = replacement;
  return true;
}

static iree_status_t loom_cfg_simplify_rebuild_br_without_arg(
    loom_cfg_simplify_state_t* state, loom_op_t* br, uint16_t removed_index) {
  loom_block_t* dest = NULL;
  loom_value_slice_t args = {0};
  if (!loom_cfg_simplify_direct_branch(br, &dest, &args)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cfg-simplify block argument removal expected a "
                            "cfg.br or low.br predecessor");
  }
  loom_value_id_t* kept_args = NULL;
  iree_host_size_t kept_count = args.count - 1;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->analysis_arena, kept_count,
                                  sizeof(*kept_args), (void**)&kept_args));
  }
  iree_host_size_t kept_index = 0;
  for (uint16_t i = 0; i < args.count; ++i) {
    if (i == removed_index) continue;
    kept_args[kept_index++] = args.values[i];
  }
  return loom_cfg_simplify_replace_direct_br(state, br, dest, kept_args,
                                             kept_count);
}

static iree_status_t loom_cfg_simplify_remove_block_arg(
    loom_cfg_simplify_state_t* state, loom_block_t* block,
    loom_op_t** pred_branches, iree_host_size_t predecessor_count,
    uint16_t arg_index, loom_value_id_t replacement) {
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      state->rewriter, old_arg, replacement));
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_rebuild_br_without_arg(
        state, pred_branches[i], arg_index));
  }
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
  return value->use_count == 0 &&
         !loom_module_value_has_type_uses(state->module, arg);
}

static iree_status_t loom_cfg_simplify_remove_unused_block_arg(
    loom_cfg_simplify_state_t* state, loom_block_t* block,
    loom_op_t** pred_branches, iree_host_size_t predecessor_count,
    uint16_t arg_index) {
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_rebuild_br_without_arg(
        state, pred_branches[i], arg_index));
  }
  IREE_RETURN_IF_ERROR(loom_block_remove_arg(state->module, block, arg_index));
  ++state->statistics->block_args_removed;
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_remove_redundant_block_args(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (block->arg_count == 0) continue;

    loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    if (predecessors.count == 0) continue;
    loom_op_t** pred_branches = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->analysis_arena, predecessors.count, sizeof(*pred_branches),
        (void**)&pred_branches));
    if (!loom_cfg_simplify_pred_branches_to_block(graph, block_index,
                                                  pred_branches)) {
      continue;
    }

    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      if (loom_cfg_simplify_block_arg_unused(state, block, arg_index)) {
        IREE_RETURN_IF_ERROR(loom_cfg_simplify_remove_unused_block_arg(
            state, block, pred_branches, predecessors.count, arg_index));
        *out_changed = true;
        return iree_ok_status();
      }

      loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
      bool found_replacement =
          loom_cfg_simplify_find_duplicate_arg_replacement(
              state, block, pred_branches, predecessors.count, arg_index,
              &replacement) ||
          loom_cfg_simplify_find_forwarded_arg_replacement(
              state, block, pred_branches, predecessors.count, arg_index,
              &replacement);
      if (!found_replacement) continue;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_remove_block_arg(
          state, block, pred_branches, predecessors.count, arg_index,
          replacement));
      *out_changed = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_process_cfg_region(
    loom_cfg_simplify_state_t* state, loom_region_t* region,
    bool* out_changed) {
  loom_cfg_graph_t graph = {0};
  IREE_RETURN_IF_ERROR(loom_cfg_graph_build(state->module, region,
                                            state->analysis_arena, &graph));
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_remove_unreachable_blocks(state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  loom_cfg_simplify_block_entry_path_facts_t* path_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_compute_block_entry_path_fact_table(
      state, &graph, &path_facts));
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_thread_fact_known_branches(
      state, &graph, path_facts, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_fold_path_sensitive_branches(
      state, &graph, path_facts, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_fold_path_sensitive_i1_ops(
      state, &graph, path_facts, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_duplicate_terminal_successors(
      state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_forward_trivial_blocks(state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_fuse_single_predecessor_blocks(
      state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_merge_duplicate_terminal_blocks(
      state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_merge_alpha_equivalent_blocks(
      state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  return loom_cfg_simplify_remove_redundant_block_args(state, &graph,
                                                       out_changed);
}

static iree_status_t loom_cfg_simplify_process_function_once(
    loom_cfg_simplify_state_t* state, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  state->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_region_stack_push(
      state->pass->arena, &state->region_stack, body));
  while (true) {
    loom_region_t* region =
        loom_cfg_simplify_region_stack_pop(&state->region_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_cfg_simplify_fold_block_branches(state, block, out_changed));
      if (*out_changed) return iree_ok_status();
    }

    if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
      IREE_RETURN_IF_ERROR(
          loom_cfg_simplify_process_cfg_region(state, region, out_changed));
      if (*out_changed) return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_simplify_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

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
  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    iree_arena_reset(&analysis_arena);

    loom_value_fact_table_t* fact_table = NULL;
    status = loom_pass_value_facts_acquire(
        pass, module, loom_pass_value_fact_scope_function(function),
        &fact_table);
    if (!iree_status_is_ok(status)) break;

    status = loom_cfg_simplify_mark_cfg_regions(loom_func_like_body(function),
                                                &analysis_arena);
    if (!iree_status_is_ok(status)) break;

    loom_dominance_info_t dominance = {0};
    status =
        loom_dominance_info_initialize(module, &analysis_arena, &dominance);
    if (!iree_status_is_ok(status)) break;

    state.fact_table = fact_table;
    state.dominance = &dominance;
    status =
        loom_cfg_simplify_process_function_once(&state, function, &changed);
    if (changed) {
      any_changed = true;
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&analysis_arena);
  return status;
}
