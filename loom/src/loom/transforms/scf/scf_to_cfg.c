// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_to_cfg.h"

#include <stdint.h>
#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_SCF_TO_CFG_STATISTICS(V, statistics_type)     \
  V(statistics_type, ifs_lowered, "ifs-lowered",           \
    "Number of scf.if ops lowered.")                       \
  V(statistics_type, fors_lowered, "fors-lowered",         \
    "Number of scf.for ops lowered.")                      \
  V(statistics_type, whiles_lowered, "whiles-lowered",     \
    "Number of scf.while ops lowered.")                    \
  V(statistics_type, switches_lowered, "switches-lowered", \
    "Number of scf.switch ops lowered.")

LOOM_PASS_STATISTICS_DEFINE(loom_scf_to_cfg_statistics,
                            loom_scf_to_cfg_statistics_t,
                            LOOM_SCF_TO_CFG_STATISTICS)

static const loom_pass_info_t loom_scf_to_cfg_pass_info_storage = {
    .name = IREE_SVL("scf-to-cfg"),
    .description = IREE_SVL("Lower structured SCF control flow to CFG blocks."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_scf_to_cfg_statistics_layout,
};

const loom_pass_info_t* loom_scf_to_cfg_pass_info(void) {
  return &loom_scf_to_cfg_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

#define LOOM_SCF_TO_CFG_INITIAL_REGION_STACK_CAPACITY 16

typedef struct loom_scf_to_cfg_region_stack_t {
  // Region pointers waiting to be processed.
  loom_region_t** regions;
  // Number of queued region pointers.
  iree_host_size_t count;
  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_scf_to_cfg_region_stack_t;

typedef struct loom_scf_to_cfg_state_t {
  // Current pass instance used for diagnostics and statistics.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_scf_to_cfg_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for use-list and type-use-aware mutations.
  loom_rewriter_t* rewriter;
  // Transient arena reset around each structured-op lowering.
  iree_arena_allocator_t* lowering_arena;
  // Value facts computed before mutation for structured legality checks.
  loom_value_fact_table_t* fact_table;
  // Region DFS stack for finding structured control flow.
  loom_scf_to_cfg_region_stack_t region_stack;
} loom_scf_to_cfg_state_t;

typedef struct loom_scf_to_cfg_block_insertion_t {
  // Region receiving blocks created for one structured op lowering.
  loom_region_t* region;
  // Next block-table index for a generated CFG block.
  uint16_t next_index;
} loom_scf_to_cfg_block_insertion_t;

static iree_status_t loom_scf_to_cfg_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_scf_to_cfg_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_SCF_TO_CFG_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_region_t*), (void**)&stack->regions);
}

static iree_status_t loom_scf_to_cfg_region_stack_push(
    iree_arena_allocator_t* arena, loom_scf_to_cfg_region_stack_t* stack,
    loom_region_t* region) {
  if (!region || region->block_count == 0) return iree_ok_status();
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(loom_region_t*),
        &stack->capacity, (void**)&stack->regions));
  }
  stack->regions[stack->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_scf_to_cfg_region_stack_pop(
    loom_scf_to_cfg_region_stack_t* stack) {
  return stack->count > 0 ? stack->regions[--stack->count] : NULL;
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

static iree_status_t loom_scf_to_cfg_emit(loom_scf_to_cfg_state_t* state,
                                          const loom_op_t* op,
                                          const loom_error_def_t* error) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(state->module, op)),
      loom_param_string(state->pass->info->name),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission);
}

//===----------------------------------------------------------------------===//
// IR mutation helpers
//===----------------------------------------------------------------------===//

static iree_status_t loom_scf_to_cfg_initialize_remap(
    loom_scf_to_cfg_state_t* state, loom_ir_remap_t* remap) {
  return loom_ir_remap_initialize(state->module, state->module,
                                  state->lowering_arena,
                                  &(loom_ir_remap_options_t){
                                      .allow_unmapped_values = true,
                                  },
                                  remap);
}

static void loom_scf_to_cfg_record_subtree_effects(loom_module_t* module,
                                                   loom_op_t* op) {
  loom_module_record_op_effects(module, op);
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        loom_scf_to_cfg_record_subtree_effects(module, child_op);
      }
    }
  }
}

static iree_status_t loom_scf_to_cfg_move_op_to_block_end(
    loom_scf_to_cfg_state_t* state, loom_op_t* op, loom_block_t* target_block,
    loom_op_t* parent_op) {
  loom_module_t* module = state->module;
  loom_block_unlink_op(module, op);
  op->parent_block = NULL;
  op->parent_op = parent_op;
  IREE_RETURN_IF_ERROR(loom_block_append_op(module, target_block, op));
  loom_scf_to_cfg_record_subtree_effects(module, op);
  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(state->rewriter, op));
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_move_tail_after_op(
    loom_scf_to_cfg_state_t* state, loom_op_t* op, loom_block_t* target_block) {
  loom_op_t* parent_op = op->parent_op;
  while (op->next_op) {
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_op_to_block_end(
        state, op->next_op, target_block, parent_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_move_block_body_to_end(
    loom_scf_to_cfg_state_t* state, loom_block_t* source_block,
    loom_op_t* terminator, loom_block_t* target_block, loom_op_t* parent_op,
    loom_ir_remap_t* remap) {
  while (source_block->first_op && source_block->first_op != terminator) {
    loom_op_t* child_op = source_block->first_op;
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_op_references(state->rewriter, child_op, remap));
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_op_to_block_end(
        state, child_op, target_block, parent_op));
  }
  return iree_ok_status();
}

static loom_builder_ip_t loom_scf_to_cfg_set_block_end(
    loom_scf_to_cfg_state_t* state, loom_block_t* block, loom_op_t* parent_op) {
  loom_builder_t* builder = &state->rewriter->builder;
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  builder->ip.block = block;
  builder->ip.parent_op = parent_op;
  builder->ip.before_op = NULL;
  return saved_ip;
}

static loom_scf_to_cfg_block_insertion_t loom_scf_to_cfg_insert_blocks_after(
    loom_block_t* source_block) {
  return (loom_scf_to_cfg_block_insertion_t){
      .region = source_block->parent_region,
      .next_index = (uint16_t)(loom_block_region_index(source_block) + 1),
  };
}

static iree_status_t loom_scf_to_cfg_insert_block(
    loom_scf_to_cfg_state_t* state,
    loom_scf_to_cfg_block_insertion_t* insertion, loom_block_t** out_block) {
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      state->module, insertion->region, insertion->next_index, out_block));
  ++insertion->next_index;
  insertion->region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_define_block_arg(
    loom_scf_to_cfg_state_t* state, loom_block_t* block, loom_type_t type,
    loom_string_id_t name_id, loom_value_id_t* out_arg) {
  IREE_RETURN_IF_ERROR(loom_module_define_value(state->module, type, out_arg));
  IREE_RETURN_IF_ERROR(loom_block_add_arg(state->module, block, *out_arg));
  if (name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_name(state->module, *out_arg, name_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_define_remapped_block_args(
    loom_scf_to_cfg_state_t* state, loom_block_t* block,
    const loom_value_id_t* source_values, uint16_t source_value_count,
    loom_value_id_t** out_args) {
  *out_args = NULL;
  if (source_value_count == 0) return iree_ok_status();

  loom_value_id_t* args = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lowering_arena, source_value_count, sizeof(*args), (void**)&args));
  for (uint16_t i = 0; i < source_value_count; ++i) {
    loom_string_id_t name_id =
        loom_module_value(state->module, source_values[i])->name_id;
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_define_block_arg(
        state, block, loom_type_none(), name_id, &args[i]));
  }

  loom_ir_remap_t type_remap = {0};
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_initialize_remap(state, &type_remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(&type_remap, source_values,
                                                args, source_value_count));
  loom_type_t* arg_types = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_value_types(
      &type_remap, source_values, source_value_count, &arg_types));
  for (uint16_t i = 0; i < source_value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(state->module, args[i], arg_types[i]));
  }

  *out_args = args;
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_resolve_values(
    const loom_ir_remap_t* remap, const loom_value_id_t* source_values,
    uint16_t source_value_count, loom_value_id_t* target_values) {
  for (uint16_t i = 0; i < source_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(remap, source_values[i],
                                                     &target_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_replace_results(
    loom_scf_to_cfg_state_t* state, loom_op_t* op,
    const loom_value_id_t* replacements) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        state->rewriter, results[i], replacements[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_build_br(
    loom_scf_to_cfg_state_t* state, loom_block_t* block, loom_op_t* parent_op,
    loom_block_t* dest, const loom_value_id_t* args, uint16_t arg_count,
    loom_location_id_t location) {
  loom_builder_ip_t saved_ip =
      loom_scf_to_cfg_set_block_end(state, block, parent_op);
  loom_op_t* branch_op = NULL;
  iree_status_t status = loom_cfg_br_build(
      &state->rewriter->builder, dest, args, arg_count, location, &branch_op);
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  return status;
}

static iree_status_t loom_scf_to_cfg_build_cond_br(
    loom_scf_to_cfg_state_t* state, loom_block_t* block, loom_op_t* parent_op,
    loom_value_id_t condition, loom_block_t* true_dest,
    loom_block_t* false_dest, loom_location_id_t location) {
  loom_builder_ip_t saved_ip =
      loom_scf_to_cfg_set_block_end(state, block, parent_op);
  loom_op_t* branch_op = NULL;
  iree_status_t status =
      loom_cfg_cond_br_build(&state->rewriter->builder, condition, true_dest,
                             false_dest, location, &branch_op);
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  return status;
}

static void loom_scf_to_cfg_read_single_block_region(
    loom_region_t* region, loom_block_t** out_block,
    loom_op_t** out_terminator) {
  loom_block_t* block = loom_region_entry_block(region);
  *out_block = block;
  *out_terminator = block->last_op;
}

static iree_status_t loom_scf_to_cfg_verify_op_preconditions(
    loom_scf_to_cfg_state_t* state, loom_op_t* op) {
  if (op->tied_result_count == 0) return iree_ok_status();
  return loom_scf_to_cfg_emit(state, op, LOOM_ERR_STRUCTURE_036);
}

//===----------------------------------------------------------------------===//
// scf.if
//===----------------------------------------------------------------------===//

static iree_status_t loom_scf_to_cfg_lower_if(loom_scf_to_cfg_state_t* state,
                                              loom_op_t* op) {
  iree_arena_reset(state->lowering_arena);

  loom_block_t* then_source_block = NULL;
  loom_op_t* then_yield = NULL;
  loom_scf_to_cfg_read_single_block_region(loom_scf_if_then_region(op),
                                           &then_source_block, &then_yield);
  loom_block_t* else_source_block = NULL;
  loom_op_t* else_yield = NULL;
  loom_region_t* else_region = loom_scf_if_else_region(op);
  if (else_region) {
    loom_scf_to_cfg_read_single_block_region(else_region, &else_source_block,
                                             &else_yield);
  }
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_verify_op_preconditions(state, op));

  loom_block_t* source_block = op->parent_block;
  loom_scf_to_cfg_block_insertion_t insertion =
      loom_scf_to_cfg_insert_blocks_after(source_block);

  loom_block_t* then_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &then_block));
  loom_block_t* else_block = NULL;
  if (else_region) {
    IREE_RETURN_IF_ERROR(
        loom_scf_to_cfg_insert_block(state, &insertion, &else_block));
  }
  loom_block_t* join_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &join_block));
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_move_tail_after_op(state, op, join_block));

  loom_value_id_t* join_args = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_define_remapped_block_args(
      state, join_block, loom_op_const_results(op), op->result_count,
      &join_args));

  loom_ir_remap_t then_remap = {0};
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_initialize_remap(state, &then_remap));
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_block_body_to_end(
      state, then_source_block, then_yield, then_block, op->parent_op,
      &then_remap));
  loom_value_id_t* then_yield_values = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lowering_arena, op->result_count, sizeof(*then_yield_values),
        (void**)&then_yield_values));
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_resolve_values(
        &then_remap, loom_op_const_operands(then_yield), op->result_count,
        then_yield_values));
  }
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(
      state, then_block, op->parent_op, join_block, then_yield_values,
      op->result_count, then_yield->location));

  if (else_region) {
    loom_ir_remap_t else_remap = {0};
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_initialize_remap(state, &else_remap));
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_block_body_to_end(
        state, else_source_block, else_yield, else_block, op->parent_op,
        &else_remap));
    loom_value_id_t* else_yield_values = NULL;
    if (op->result_count > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->lowering_arena, op->result_count, sizeof(*else_yield_values),
          (void**)&else_yield_values));
      IREE_RETURN_IF_ERROR(loom_scf_to_cfg_resolve_values(
          &else_remap, loom_op_const_operands(else_yield), op->result_count,
          else_yield_values));
    }
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(
        state, else_block, op->parent_op, join_block, else_yield_values,
        op->result_count, else_yield->location));
  }

  loom_builder_t* builder = &state->rewriter->builder;
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_before(builder, op);
  loom_op_t* cond_br_op = NULL;
  iree_status_t status = loom_cfg_cond_br_build(
      builder, loom_scf_if_condition(op), then_block,
      else_block ? else_block : join_block, op->location, &cond_br_op);
  loom_builder_restore(builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_replace_results(state, op, join_args));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(state->rewriter, op));
  ++state->statistics->ifs_lowered;
  iree_arena_reset(state->lowering_arena);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// scf.for
//===----------------------------------------------------------------------===//

// Keep materialized IV facts compact. Very wide dynamic ranges remain true in
// the structured source, but printing them after CFG lowering would obscure the
// control flow without helping target address legality.
#define LOOM_SCF_TO_CFG_MATERIALIZED_IV_RANGE_MAX UINT32_MAX
#define LOOM_SCF_TO_CFG_FOR_IV_PREDICATE_CAPACITY 4

static uint16_t loom_scf_to_cfg_for_iv_predicates_from_facts(
    loom_value_id_t value, loom_value_facts_t facts,
    loom_predicate_t* predicates) {
  if (value == LOOM_VALUE_ID_INVALID || loom_value_facts_is_float(facts) ||
      loom_value_facts_is_unknown(facts)) {
    return 0;
  }

  uint16_t count = 0;
  if (loom_value_facts_is_exact(facts)) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_EQ,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {value, facts.range_lo, 0},
    };
  } else if (facts.range_hi >= facts.range_lo &&
             facts.range_hi <= LOOM_SCF_TO_CFG_MATERIALIZED_IV_RANGE_MAX) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_RANGE,
        .arg_count = 3,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_CONST},
        .args = {value, facts.range_lo, facts.range_hi},
    };
  }

  if (count < LOOM_SCF_TO_CFG_FOR_IV_PREDICATE_CAPACITY &&
      facts.known_divisor > 1 && !loom_value_facts_is_exact(facts)) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_MUL,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {value, facts.known_divisor, 0},
    };
  }

  return count;
}

static bool loom_scf_to_cfg_for_iv_facts_are_materializable(
    loom_value_facts_t facts) {
  loom_predicate_t predicates[LOOM_SCF_TO_CFG_FOR_IV_PREDICATE_CAPACITY];
  return loom_scf_to_cfg_for_iv_predicates_from_facts((loom_value_id_t)0, facts,
                                                      predicates) > 0;
}

static bool loom_scf_to_cfg_value_has_exact_integer_facts(
    loom_value_fact_table_t* fact_table, loom_value_id_t value) {
  if (!fact_table || value == LOOM_VALUE_ID_INVALID) return false;
  int64_t exact_value = 0;
  return loom_value_facts_as_exact_i64(
      loom_value_fact_table_lookup(fact_table, value), &exact_value);
}

static bool loom_scf_to_cfg_for_iv_bound_predicate_is_needed(
    loom_value_fact_table_t* fact_table, loom_value_id_t bound) {
  return bound != LOOM_VALUE_ID_INVALID &&
         !loom_scf_to_cfg_value_has_exact_integer_facts(fact_table, bound);
}

static uint16_t loom_scf_to_cfg_for_iv_bound_predicate_count(
    loom_value_fact_table_t* fact_table, loom_value_id_t lower_bound,
    loom_value_id_t upper_bound) {
  uint16_t count = 0;
  if (loom_scf_to_cfg_for_iv_bound_predicate_is_needed(fact_table,
                                                       lower_bound)) {
    ++count;
  }
  if (loom_scf_to_cfg_for_iv_bound_predicate_is_needed(fact_table,
                                                       upper_bound)) {
    ++count;
  }
  return count;
}

static uint16_t loom_scf_to_cfg_for_iv_bound_predicates(
    loom_value_fact_table_t* fact_table, loom_value_id_t value,
    loom_value_id_t lower_bound, loom_value_id_t upper_bound,
    loom_predicate_t* predicates) {
  if (value == LOOM_VALUE_ID_INVALID) return 0;
  uint16_t count = 0;
  if (loom_scf_to_cfg_for_iv_bound_predicate_is_needed(fact_table,
                                                       lower_bound)) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_GE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_NONE},
        .args = {value, lower_bound, 0},
    };
  }
  if (loom_scf_to_cfg_for_iv_bound_predicate_is_needed(fact_table,
                                                       upper_bound)) {
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_LT,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_NONE},
        .args = {value, upper_bound, 0},
    };
  }
  return count;
}

static bool loom_scf_to_cfg_for_iv_assume_is_materializable(
    loom_value_fact_table_t* fact_table, loom_op_t* op,
    loom_value_facts_t facts) {
  if (loom_scf_to_cfg_for_iv_facts_are_materializable(facts)) return true;
  return loom_scf_to_cfg_for_iv_bound_predicate_count(
             fact_table, loom_scf_for_lower_bound(op),
             loom_scf_for_upper_bound(op)) > 0;
}

static iree_status_t loom_scf_to_cfg_define_for_header_args(
    loom_scf_to_cfg_state_t* state, loom_op_t* op, loom_block_t* header_block,
    loom_block_t* body_source_block, bool reserve_iv_name_for_assume,
    loom_value_id_t* out_iv_arg, loom_value_id_t** out_iter_args) {
  *out_iv_arg = LOOM_VALUE_ID_INVALID;
  *out_iter_args = NULL;

  loom_string_id_t iv_name_id =
      !reserve_iv_name_for_assume && body_source_block->arg_count > 0
          ? loom_module_value(state->module, body_source_block->arg_ids[0])
                ->name_id
          : LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_define_block_arg(
      state, header_block,
      loom_module_value_type(state->module, loom_scf_for_lower_bound(op)),
      iv_name_id, out_iv_arg));

  loom_value_slice_t results = loom_scf_for_results(op);
  const loom_value_id_t* iter_arg_names = results.values;
  if (body_source_block->arg_count > 1) {
    iter_arg_names = body_source_block->arg_ids + 1;
  }
  return loom_scf_to_cfg_define_remapped_block_args(
      state, header_block, iter_arg_names, results.count, out_iter_args);
}

static iree_status_t loom_scf_to_cfg_build_for_iv_assume(
    loom_scf_to_cfg_state_t* state, loom_op_t* op, loom_block_t* body_block,
    loom_value_id_t header_iv, loom_value_facts_t original_iv_facts,
    loom_string_id_t result_name_id, loom_value_id_t* out_iv) {
  *out_iv = header_iv;

  loom_predicate_t predicate_storage[LOOM_SCF_TO_CFG_FOR_IV_PREDICATE_CAPACITY];
  uint16_t predicate_count = loom_scf_to_cfg_for_iv_predicates_from_facts(
      header_iv, original_iv_facts, predicate_storage);
  predicate_count += loom_scf_to_cfg_for_iv_bound_predicates(
      state->fact_table, header_iv, loom_scf_for_lower_bound(op),
      loom_scf_for_upper_bound(op), predicate_storage + predicate_count);
  if (predicate_count == 0) {
    return iree_ok_status();
  }

  loom_predicate_t* predicates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&state->module->arena, predicate_count,
                                sizeof(*predicates), (void**)&predicates));
  memcpy(predicates, predicate_storage, predicate_count * sizeof(*predicates));

  loom_type_t result_type =
      loom_module_value_type(state->module, loom_scf_for_lower_bound(op));
  const loom_value_id_t value = header_iv;
  loom_builder_ip_t saved_ip =
      loom_scf_to_cfg_set_block_end(state, body_block, op->parent_op);
  loom_op_t* assume_op = NULL;
  iree_status_t status = loom_index_assume_build(
      &state->rewriter->builder, &value, 1, predicates, predicate_count,
      &result_type, 1, op->location, &assume_op);
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  *out_iv = loom_index_assume_results(assume_op).values[0];
  if (result_name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_name(state->module, *out_iv, result_name_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_lower_for(loom_scf_to_cfg_state_t* state,
                                               loom_op_t* op) {
  iree_arena_reset(state->lowering_arena);

  loom_value_facts_t step_facts =
      loom_value_fact_table_lookup(state->fact_table, loom_scf_for_step(op));
  if (loom_value_facts_is_float(step_facts) ||
      !loom_value_facts_is_positive(step_facts)) {
    return loom_scf_to_cfg_emit(state, op, LOOM_ERR_STRUCTURE_035);
  }

  loom_block_t* body_source_block = NULL;
  loom_op_t* yield_op = NULL;
  loom_scf_to_cfg_read_single_block_region(loom_scf_for_body(op),
                                           &body_source_block, &yield_op);
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_verify_op_preconditions(state, op));

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_string_id_t original_iv_name_id =
      body_source_block->arg_count > 0
          ? loom_module_value(state->module, body_source_block->arg_ids[0])
                ->name_id
          : LOOM_STRING_ID_INVALID;
  loom_value_facts_t original_iv_facts =
      body_source_block->arg_count > 0
          ? loom_value_fact_table_lookup(state->fact_table,
                                         body_source_block->arg_ids[0])
          : loom_value_facts_unknown();
  bool reserve_iv_name_for_assume =
      original_iv_name_id != LOOM_STRING_ID_INVALID &&
      loom_scf_to_cfg_for_iv_assume_is_materializable(state->fact_table, op,
                                                      original_iv_facts);

  loom_block_t* source_block = op->parent_block;
  loom_scf_to_cfg_block_insertion_t insertion =
      loom_scf_to_cfg_insert_blocks_after(source_block);

  loom_block_t* header_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &header_block));
  loom_block_t* body_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &body_block));
  loom_block_t* exit_block = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_scf_to_cfg_insert_block(state, &insertion, &exit_block));
  }
  loom_block_t* join_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &join_block));
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_move_tail_after_op(state, op, join_block));

  loom_value_id_t* join_args = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_define_remapped_block_args(
      state, join_block, loom_op_const_results(op), op->result_count,
      &join_args));

  loom_value_id_t header_iv = LOOM_VALUE_ID_INVALID;
  loom_value_id_t* header_iter_args = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_define_for_header_args(
      state, op, header_block, body_source_block, reserve_iv_name_for_assume,
      &header_iv, &header_iter_args));

  loom_value_id_t body_iv = header_iv;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_for_iv_assume(
      state, op, body_block, header_iv, original_iv_facts, original_iv_name_id,
      &body_iv));

  loom_ir_remap_t body_remap = {0};
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_initialize_remap(state, &body_remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
      &body_remap, body_source_block->arg_ids[0], body_iv));
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &body_remap, body_source_block->arg_ids[i + 1], header_iter_args[i]));
  }

  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_block_body_to_end(
      state, body_source_block, yield_op, body_block, op->parent_op,
      &body_remap));

  loom_value_id_t* yielded_values = NULL;
  if (iter_args.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lowering_arena, iter_args.count, sizeof(*yielded_values),
        (void**)&yielded_values));
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_resolve_values(
        &body_remap, loom_op_const_operands(yield_op), iter_args.count,
        yielded_values));
  }

  loom_value_id_t* initial_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lowering_arena, (iree_host_size_t)1 + iter_args.count,
      sizeof(*initial_values), (void**)&initial_values));
  initial_values[0] = loom_scf_for_lower_bound(op);
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    initial_values[i + 1] = iter_args.values[i];
  }
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(
      state, source_block, op->parent_op, header_block, initial_values,
      (uint16_t)(1 + iter_args.count), op->location));

  loom_builder_ip_t header_ip =
      loom_scf_to_cfg_set_block_end(state, header_block, op->parent_op);
  loom_op_t* compare_op = NULL;
  iree_status_t status = loom_index_cmp_build(
      &state->rewriter->builder, LOOM_INDEX_CMP_PREDICATE_SLT, header_iv,
      loom_scf_for_upper_bound(op),
      loom_module_value_type(state->module, loom_scf_for_lower_bound(op)),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), op->location, &compare_op);
  if (iree_status_is_ok(status)) {
    status = loom_cfg_cond_br_build(
        &state->rewriter->builder, loom_index_cmp_result(compare_op),
        body_block, exit_block ? exit_block : join_block, op->location,
        &compare_op);
  }
  loom_builder_restore(&state->rewriter->builder, header_ip);
  IREE_RETURN_IF_ERROR(status);

  if (exit_block) {
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(
        state, exit_block, op->parent_op, join_block, header_iter_args,
        op->result_count, op->location));
  }

  loom_builder_ip_t body_ip =
      loom_scf_to_cfg_set_block_end(state, body_block, op->parent_op);
  loom_op_t* next_iv_op = NULL;
  status = loom_index_add_build(
      &state->rewriter->builder, body_iv, loom_scf_for_step(op),
      loom_module_value_type(state->module, loom_scf_for_lower_bound(op)),
      op->location, &next_iv_op);
  loom_builder_restore(&state->rewriter->builder, body_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_value_id_t* backedge_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lowering_arena, (iree_host_size_t)1 + iter_args.count,
      sizeof(*backedge_values), (void**)&backedge_values));
  backedge_values[0] = loom_index_add_result(next_iv_op);
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    backedge_values[i + 1] = yielded_values[i];
  }
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(
      state, body_block, op->parent_op, header_block, backedge_values,
      (uint16_t)(1 + iter_args.count), yield_op->location));

  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_replace_results(state, op, join_args));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(state->rewriter, op));
  ++state->statistics->fors_lowered;
  iree_arena_reset(state->lowering_arena);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// scf.while
//===----------------------------------------------------------------------===//

static iree_status_t loom_scf_to_cfg_lower_while(loom_scf_to_cfg_state_t* state,
                                                 loom_op_t* op) {
  iree_arena_reset(state->lowering_arena);

  loom_block_t* before_source_block = NULL;
  loom_op_t* condition_op = NULL;
  loom_scf_to_cfg_read_single_block_region(loom_scf_while_before(op),
                                           &before_source_block, &condition_op);
  loom_block_t* after_source_block = NULL;
  loom_op_t* yield_op = NULL;
  loom_scf_to_cfg_read_single_block_region(loom_scf_while_after(op),
                                           &after_source_block, &yield_op);
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_verify_op_preconditions(state, op));

  loom_value_slice_t iter_args = loom_scf_while_iter_args(op);
  loom_value_slice_t forwarded = loom_scf_condition_forwarded(condition_op);

  loom_block_t* source_block = op->parent_block;
  loom_scf_to_cfg_block_insertion_t insertion =
      loom_scf_to_cfg_insert_blocks_after(source_block);

  loom_block_t* before_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &before_block));
  loom_block_t* after_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &after_block));
  loom_block_t* exit_block = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_scf_to_cfg_insert_block(state, &insertion, &exit_block));
  }
  loom_block_t* join_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &join_block));
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_move_tail_after_op(state, op, join_block));

  loom_value_id_t* join_args = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_define_remapped_block_args(
      state, join_block, loom_op_const_results(op), op->result_count,
      &join_args));

  loom_value_id_t* before_args = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_define_remapped_block_args(
      state, before_block, before_source_block->arg_ids, iter_args.count,
      &before_args));

  loom_ir_remap_t before_remap = {0};
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_initialize_remap(state, &before_remap));
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &before_remap, before_source_block->arg_ids[i], before_args[i]));
  }
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_block_body_to_end(
      state, before_source_block, condition_op, before_block, op->parent_op,
      &before_remap));

  loom_value_id_t condition_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
      &before_remap, loom_scf_condition_condition(condition_op),
      &condition_value));
  loom_value_id_t* forwarded_values = NULL;
  if (forwarded.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lowering_arena, forwarded.count, sizeof(*forwarded_values),
        (void**)&forwarded_values));
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_resolve_values(
        &before_remap, forwarded.values, forwarded.count, forwarded_values));
  }

  loom_ir_remap_t after_remap = {0};
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_initialize_remap(state, &after_remap));
  for (uint16_t i = 0; i < forwarded.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &after_remap, after_source_block->arg_ids[i], forwarded_values[i]));
  }
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_block_body_to_end(
      state, after_source_block, yield_op, after_block, op->parent_op,
      &after_remap));

  loom_value_id_t* yielded_values = NULL;
  if (yield_op->operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lowering_arena, yield_op->operand_count, sizeof(*yielded_values),
        (void**)&yielded_values));
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_resolve_values(
        &after_remap, loom_op_const_operands(yield_op), yield_op->operand_count,
        yielded_values));
  }

  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(
      state, source_block, op->parent_op, before_block, iter_args.values,
      iter_args.count, op->location));
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_cond_br(
      state, before_block, op->parent_op, condition_value, after_block,
      exit_block ? exit_block : join_block, condition_op->location));
  if (exit_block) {
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(
        state, exit_block, op->parent_op, join_block, forwarded_values,
        op->result_count, condition_op->location));
  }
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(
      state, after_block, op->parent_op, before_block, yielded_values,
      yield_op->operand_count, yield_op->location));

  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_replace_results(state, op, join_args));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(state->rewriter, op));
  ++state->statistics->whiles_lowered;
  iree_arena_reset(state->lowering_arena);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// scf.switch
//===----------------------------------------------------------------------===//

static iree_status_t loom_scf_to_cfg_move_switch_case(
    loom_scf_to_cfg_state_t* state, loom_op_t* op, loom_region_t* source_region,
    loom_block_t* target_block, loom_block_t* join_block) {
  loom_block_t* source_block = NULL;
  loom_op_t* yield_op = NULL;
  loom_scf_to_cfg_read_single_block_region(source_region, &source_block,
                                           &yield_op);

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_initialize_remap(state, &remap));
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_block_body_to_end(
      state, source_block, yield_op, target_block, op->parent_op, &remap));

  loom_value_id_t* yielded_values = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lowering_arena, op->result_count, sizeof(*yielded_values),
        (void**)&yielded_values));
    IREE_RETURN_IF_ERROR(
        loom_scf_to_cfg_resolve_values(&remap, loom_op_const_operands(yield_op),
                                       op->result_count, yielded_values));
  }
  return loom_scf_to_cfg_build_br(state, target_block, op->parent_op,
                                  join_block, yielded_values, op->result_count,
                                  yield_op->location);
}

static iree_status_t loom_scf_to_cfg_build_switch_dispatch(
    loom_scf_to_cfg_state_t* state, loom_op_t* op, loom_block_t* block,
    int64_t case_key, loom_block_t* true_dest, loom_block_t* false_dest) {
  loom_builder_ip_t saved_ip =
      loom_scf_to_cfg_set_block_end(state, block, op->parent_op);
  if (block == op->parent_block) {
    loom_builder_set_before(&state->rewriter->builder, op);
  }

  loom_op_t* key_op = NULL;
  iree_status_t status = loom_index_constant_build(
      &state->rewriter->builder, loom_attr_i64(case_key),
      loom_module_value_type(state->module, loom_scf_switch_selector(op)),
      op->location, &key_op);
  loom_op_t* cmp_op = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_index_cmp_build(
        &state->rewriter->builder, LOOM_INDEX_CMP_PREDICATE_EQ,
        loom_scf_switch_selector(op), loom_index_constant_result(key_op),
        loom_module_value_type(state->module, loom_scf_switch_selector(op)),
        loom_type_scalar(LOOM_SCALAR_TYPE_I1), op->location, &cmp_op);
  }
  loom_op_t* cond_br_op = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_cfg_cond_br_build(&state->rewriter->builder,
                                    loom_index_cmp_result(cmp_op), true_dest,
                                    false_dest, op->location, &cond_br_op);
  }

  loom_builder_restore(&state->rewriter->builder, saved_ip);
  return status;
}

static iree_status_t loom_scf_to_cfg_lower_switch(
    loom_scf_to_cfg_state_t* state, loom_op_t* op) {
  iree_arena_reset(state->lowering_arena);

  loom_attribute_t case_keys_attr = loom_scf_switch_case_keys(op);
  loom_region_slice_t case_regions = loom_scf_switch_case_regions(op);
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_verify_op_preconditions(state, op));

  loom_block_t* source_block = op->parent_block;
  loom_scf_to_cfg_block_insertion_t insertion =
      loom_scf_to_cfg_insert_blocks_after(source_block);

  loom_block_t** dispatch_blocks = NULL;
  iree_host_size_t dispatch_block_count =
      case_regions.count > 1 ? (iree_host_size_t)(case_regions.count - 1) : 0;
  if (dispatch_block_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lowering_arena, dispatch_block_count, sizeof(*dispatch_blocks),
        (void**)&dispatch_blocks));
    for (iree_host_size_t i = 0; i < dispatch_block_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_scf_to_cfg_insert_block(state, &insertion, &dispatch_blocks[i]));
    }
  }

  loom_block_t** case_blocks = NULL;
  if (case_regions.count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->lowering_arena, case_regions.count,
                                  sizeof(*case_blocks), (void**)&case_blocks));
    for (uint8_t i = 0; i < case_regions.count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_scf_to_cfg_insert_block(state, &insertion, &case_blocks[i]));
    }
  }
  loom_block_t* default_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &default_block));
  loom_block_t* join_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_insert_block(state, &insertion, &join_block));
  IREE_RETURN_IF_ERROR(
      loom_scf_to_cfg_move_tail_after_op(state, op, join_block));

  loom_value_id_t* join_args = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_define_remapped_block_args(
      state, join_block, loom_op_const_results(op), op->result_count,
      &join_args));

  for (uint8_t i = 0; i < case_regions.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_switch_case(
        state, op, case_regions.regions[i], case_blocks[i], join_block));
  }
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_move_switch_case(
      state, op, loom_scf_switch_default_region(op), default_block,
      join_block));

  if (case_regions.count == 0) {
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_br(state, source_block,
                                                  op->parent_op, default_block,
                                                  NULL, 0, op->location));
  } else {
    for (uint8_t i = 0; i < case_regions.count; ++i) {
      loom_block_t* dispatch_block =
          i == 0 ? source_block : dispatch_blocks[i - 1];
      loom_block_t* false_dest = (uint8_t)(i + 1) < case_regions.count
                                     ? dispatch_blocks[i]
                                     : default_block;
      IREE_RETURN_IF_ERROR(loom_scf_to_cfg_build_switch_dispatch(
          state, op, dispatch_block, case_keys_attr.i64_array[i],
          case_blocks[i], false_dest));
    }
  }

  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_replace_results(state, op, join_args));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(state->rewriter, op));
  ++state->statistics->switches_lowered;
  iree_arena_reset(state->lowering_arena);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Traversal
//===----------------------------------------------------------------------===//

static iree_status_t loom_scf_to_cfg_push_child_regions(
    loom_scf_to_cfg_state_t* state, loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_region_stack_push(
        state->pass->arena, &state->region_stack, regions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_process_block(
    loom_scf_to_cfg_state_t* state, loom_block_t* block, bool* out_changed) {
  loom_op_t* op = block->first_op;
  while (op) {
    loom_op_t* next_op = op->next_op;
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      op = next_op;
      continue;
    }

    if (loom_scf_if_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_scf_to_cfg_lower_if(state, op));
      if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
      *out_changed = true;
      return iree_ok_status();
    }
    if (loom_scf_for_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_scf_to_cfg_lower_for(state, op));
      if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
      *out_changed = true;
      return iree_ok_status();
    }
    if (loom_scf_while_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_scf_to_cfg_lower_while(state, op));
      if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
      *out_changed = true;
      return iree_ok_status();
    }
    if (loom_scf_switch_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_scf_to_cfg_lower_switch(state, op));
      if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
      *out_changed = true;
      return iree_ok_status();
    }

    IREE_RETURN_IF_ERROR(loom_scf_to_cfg_push_child_regions(state, op));
    op = next_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_to_cfg_process_function_once(
    loom_scf_to_cfg_state_t* state, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  state->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_to_cfg_region_stack_push(
      state->pass->arena, &state->region_stack, body));

  while (true) {
    loom_region_t* region =
        loom_scf_to_cfg_region_stack_pop(&state->region_stack);
    if (!region || loom_pass_has_error_diagnostics(state->pass)) break;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_scf_to_cfg_process_block(state, block, out_changed));
      if (*out_changed || loom_pass_has_error_diagnostics(state->pass)) {
        return iree_ok_status();
      }
    }
  }

  return iree_ok_status();
}

iree_status_t loom_scf_to_cfg_run(loom_pass_t* pass, loom_module_t* module,
                                  loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  iree_arena_allocator_t lowering_arena = {0};
  iree_arena_initialize(pass->arena->block_pool, &lowering_arena);

  loom_scf_to_cfg_state_t state = {
      .pass = pass,
      .statistics = loom_scf_to_cfg_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
      .lowering_arena = &lowering_arena,
  };
  iree_status_t status =
      loom_scf_to_cfg_region_stack_initialize(pass->arena, &state.region_stack);

  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed &&
         !loom_pass_has_error_diagnostics(pass)) {
    changed = false;
    status = loom_pass_value_facts_acquire(
        pass, module, loom_pass_value_fact_scope_function(function),
        &state.fact_table);
    if (!iree_status_is_ok(status)) break;
    status = loom_scf_to_cfg_process_function_once(&state, function, &changed);
    if (changed) {
      any_changed = true;
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&lowering_arena);
  return status;
}
