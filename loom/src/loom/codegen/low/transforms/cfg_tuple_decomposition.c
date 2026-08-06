// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/cfg_tuple_decomposition.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/registers.h"

#define LOOM_LOW_DECOMPOSE_CFG_TUPLES_STATISTICS(V, statistics_type)       \
  V(statistics_type, block_args_decomposed, "block-args-decomposed",       \
    "Number of tuple-valued low CFG block arguments decomposed.")          \
  V(statistics_type, lane_block_args_inserted, "lane-block-args-inserted", \
    "Number of scalar lane block arguments inserted.")                     \
  V(statistics_type, branch_edges_decomposed, "branch-edges-decomposed",   \
    "Number of low.br edges rewritten to forward scalar lanes.")           \
  V(statistics_type, slices_removed, "slices-removed",                     \
    "Number of low.slice projections replaced by lane block arguments.")

LOOM_PASS_STATISTICS_DEFINE(loom_low_decompose_cfg_tuples_statistics,
                            loom_low_decompose_cfg_tuples_statistics_t,
                            LOOM_LOW_DECOMPOSE_CFG_TUPLES_STATISTICS)

static const loom_pass_info_t loom_low_decompose_cfg_tuples_pass_info_storage =
    {
        .name = IREE_SVL("low-decompose-cfg-tuples"),
        .description =
            IREE_SVL("Decompose lane-local low CFG register tuples."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_layout = &loom_low_decompose_cfg_tuples_statistics_layout,
};

const loom_pass_info_t* loom_low_decompose_cfg_tuples_pass_info(void) {
  return &loom_low_decompose_cfg_tuples_pass_info_storage;
}

typedef struct loom_low_decompose_cfg_tuples_state_t {
  // Active pass instance for statistics and scratch allocation.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_low_decompose_cfg_tuples_statistics_t* statistics;
  // Module being rewritten.
  loom_module_t* module;
  // Rewriter used for use-list preserving IR edits.
  loom_rewriter_t* rewriter;
  // Per-candidate temporary allocation arena.
  iree_arena_allocator_t* analysis_arena;
} loom_low_decompose_cfg_tuples_state_t;

typedef struct loom_low_cfg_tuple_candidate_t {
  // Block containing the tuple argument.
  loom_block_t* block;
  // Current block argument index of the tuple argument.
  uint16_t arg_index;
  // Tuple argument value ID.
  loom_value_id_t arg_id;
  // Original tuple type.
  loom_type_t tuple_type;
  // Scalar lane type used for all decomposed lane arguments.
  loom_type_t lane_type;
  // Number of register allocation units in the tuple.
  uint32_t lane_count;
  // Direct low.br predecessor terminators targeting |block|.
  loom_op_t** predecessor_branches;
  // Number of predecessor branches.
  iree_host_size_t predecessor_count;
  // low.slice projections using |arg_id|.
  loom_op_t** slice_ops;
  // Number of slice projections.
  uint32_t slice_count;
} loom_low_cfg_tuple_candidate_t;

static bool loom_low_decompose_cfg_tuples_get_register_tuple_type(
    loom_type_t type, loom_type_t* out_lane_type, uint32_t* out_lane_count) {
  *out_lane_type = loom_type_none();
  *out_lane_count = 0;
  if (!loom_type_is_register(type) || loom_type_register_has_value_type(type)) {
    return false;
  }

  const uint32_t lane_count = loom_low_register_type_unit_count(type);
  if (lane_count <= 1 || lane_count > UINT16_MAX) {
    return false;
  }

  *out_lane_type = loom_low_register_carrier_type_with_unit_count(type, 1);
  *out_lane_count = lane_count;
  return true;
}

static iree_status_t loom_low_decompose_cfg_tuples_collect_predecessors(
    loom_low_decompose_cfg_tuples_state_t* state, loom_region_t* body,
    loom_block_t* block, uint16_t arg_index, loom_type_t arg_type,
    loom_op_t*** out_predecessor_branches,
    iree_host_size_t* out_predecessor_count) {
  *out_predecessor_branches = NULL;
  *out_predecessor_count = 0;

  iree_host_size_t predecessor_count = 0;
  loom_block_t* candidate_block = NULL;
  loom_region_for_each_block(body, candidate_block) {
    loom_op_t* terminator = candidate_block->last_op;
    if (terminator && loom_low_br_isa(terminator) &&
        loom_low_br_dest(terminator) == block) {
      ++predecessor_count;
    }
  }
  if (predecessor_count == 0) {
    return iree_ok_status();
  }

  loom_op_t** predecessor_branches = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->analysis_arena, predecessor_count, sizeof(*predecessor_branches),
      (void**)&predecessor_branches));

  iree_host_size_t predecessor_index = 0;
  loom_region_for_each_block(body, candidate_block) {
    loom_op_t* terminator = candidate_block->last_op;
    if (!terminator || !loom_low_br_isa(terminator) ||
        loom_low_br_dest(terminator) != block) {
      continue;
    }
    loom_value_slice_t args = loom_low_br_args(terminator);
    if (args.count != block->arg_count) {
      *out_predecessor_branches = NULL;
      *out_predecessor_count = 0;
      return iree_ok_status();
    }
    if (!loom_type_equal(
            loom_module_value_type(state->module, args.values[arg_index]),
            arg_type)) {
      *out_predecessor_branches = NULL;
      *out_predecessor_count = 0;
      return iree_ok_status();
    }
    predecessor_branches[predecessor_index++] = terminator;
  }

  *out_predecessor_branches = predecessor_branches;
  *out_predecessor_count = predecessor_count;
  return iree_ok_status();
}

static bool loom_low_decompose_cfg_tuples_slice_matches_candidate(
    loom_low_decompose_cfg_tuples_state_t* state,
    const loom_low_cfg_tuple_candidate_t* candidate, const loom_op_t* op,
    uint32_t* out_lane_index) {
  *out_lane_index = 0;
  if (!loom_low_slice_isa(op)) {
    return false;
  }

  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0 || (uint64_t)offset >= candidate->lane_count) {
    return false;
  }

  const loom_type_t result_type = loom_module_value_type(
      state->module, loom_low_slice_result((loom_op_t*)op));
  if (!loom_type_equal(result_type, candidate->lane_type)) {
    return false;
  }

  *out_lane_index = (uint32_t)offset;
  return true;
}

static iree_status_t loom_low_decompose_cfg_tuples_collect_slices(
    loom_low_decompose_cfg_tuples_state_t* state,
    loom_low_cfg_tuple_candidate_t* candidate, bool* out_eligible) {
  *out_eligible = false;
  loom_value_t* value = loom_module_value(state->module, candidate->arg_id);
  if (value->use_count == 0 ||
      loom_module_value_has_predicate_attribute_uses(state->module,
                                                     candidate->arg_id) ||
      loom_module_value_has_type_uses(state->module, candidate->arg_id)) {
    return iree_ok_status();
  }

  loom_op_t** slice_ops = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->analysis_arena, value->use_count,
                                sizeof(*slice_ops), (void**)&slice_ops));

  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_use_t use = uses[i];
    if (loom_use_operand_index(use) != 0) {
      return iree_ok_status();
    }

    loom_op_t* slice_op = loom_use_user_op(use);
    uint32_t lane_index = 0;
    if (!loom_low_decompose_cfg_tuples_slice_matches_candidate(
            state, candidate, slice_op, &lane_index)) {
      return iree_ok_status();
    }
    slice_ops[candidate->slice_count++] = slice_op;
  }

  candidate->slice_ops = slice_ops;
  *out_eligible = true;
  return iree_ok_status();
}

static bool loom_low_decompose_cfg_tuples_try_concat_lane(
    loom_low_decompose_cfg_tuples_state_t* state, loom_value_id_t value_id,
    uint32_t lane_index, loom_type_t lane_type, loom_value_id_t* out_source,
    uint32_t* out_source_offset) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_source_offset = 0;

  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }

  loom_op_t* op = loom_value_def_op(value);
  if (!loom_low_concat_isa(op) || loom_low_concat_result(op) != value_id) {
    return false;
  }

  uint32_t offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_value_id_t source = sources.values[i];
    const loom_type_t source_type =
        loom_module_value_type(state->module, source);
    if (!loom_type_is_register(source_type)) {
      return false;
    }

    const uint32_t source_lane_count =
        loom_low_register_type_unit_count(source_type);
    if (source_lane_count > UINT32_MAX - offset) {
      return false;
    }
    if (lane_index >= offset && lane_index < offset + source_lane_count) {
      *out_source = source;
      *out_source_offset = lane_index - offset;
      return source_lane_count == 1 && *out_source_offset == 0 &&
             loom_type_equal(source_type, lane_type);
    }
    offset += source_lane_count;
  }
  return false;
}

static iree_status_t loom_low_decompose_cfg_tuples_materialize_lane_before(
    loom_low_decompose_cfg_tuples_state_t* state, loom_op_t* before_op,
    loom_value_id_t tuple_value, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  uint32_t source_offset = 0;
  if (loom_low_decompose_cfg_tuples_try_concat_lane(
          state, tuple_value, lane_index, lane_type, &source, &source_offset)) {
    *out_lane = source;
    return iree_ok_status();
  }

  if (source == LOOM_VALUE_ID_INVALID) {
    source = tuple_value;
    source_offset = lane_index;
  }

  loom_builder_ip_t saved_ip = loom_builder_save(&state->rewriter->builder);
  loom_builder_set_before(&state->rewriter->builder, before_op);
  loom_op_t* slice_op = NULL;
  iree_status_t status =
      loom_low_slice_build(&state->rewriter->builder, source, source_offset,
                           lane_type, before_op->location, &slice_op);
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  *out_lane = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_low_decompose_cfg_tuples_rebuild_branch(
    loom_low_decompose_cfg_tuples_state_t* state,
    const loom_low_cfg_tuple_candidate_t* candidate, loom_op_t* branch_op) {
  loom_value_slice_t old_args = loom_low_br_args(branch_op);
  const iree_host_size_t new_arg_count =
      old_args.count - 1 + candidate->lane_count;
  loom_value_id_t* new_args = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->analysis_arena, new_arg_count,
                                sizeof(*new_args), (void**)&new_args));

  iree_host_size_t new_arg_index = 0;
  for (uint16_t i = 0; i < old_args.count; ++i) {
    if (i != candidate->arg_index) {
      new_args[new_arg_index++] = old_args.values[i];
      continue;
    }

    const loom_value_id_t tuple_value = old_args.values[i];
    const loom_type_t tuple_type =
        loom_module_value_type(state->module, tuple_value);
    if (!loom_type_equal(tuple_type, candidate->tuple_type)) {
      return iree_ok_status();
    }
    for (uint32_t lane_index = 0; lane_index < candidate->lane_count;
         ++lane_index) {
      IREE_RETURN_IF_ERROR(
          loom_low_decompose_cfg_tuples_materialize_lane_before(
              state, branch_op, tuple_value, lane_index, candidate->lane_type,
              &new_args[new_arg_index++]));
    }
  }

  loom_builder_ip_t saved_ip = loom_builder_save(&state->rewriter->builder);
  loom_builder_set_before(&state->rewriter->builder, branch_op);
  loom_op_t* new_branch_op = NULL;
  iree_status_t status = loom_low_br_build(
      &state->rewriter->builder, loom_low_br_dest(branch_op), new_args,
      new_arg_count, branch_op->location, &new_branch_op);
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  return loom_rewriter_erase(state->rewriter, branch_op);
}

static iree_status_t loom_low_decompose_cfg_tuples_define_lane_block_args(
    loom_low_decompose_cfg_tuples_state_t* state,
    const loom_low_cfg_tuple_candidate_t* candidate,
    loom_value_id_t* lane_args) {
  for (uint32_t i = 0; i < candidate->lane_count; ++i) {
    loom_value_id_t lane_arg = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_define_value(
        state->module, candidate->lane_type, &lane_arg));
    const uint16_t insertion_index =
        (uint16_t)((uint32_t)candidate->arg_index + i);
    IREE_RETURN_IF_ERROR(loom_block_insert_arg(state->module, candidate->block,
                                               insertion_index, lane_arg));
    lane_args[i] = lane_arg;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_decompose_cfg_tuples_replace_slices(
    loom_low_decompose_cfg_tuples_state_t* state,
    const loom_low_cfg_tuple_candidate_t* candidate,
    const loom_value_id_t* lane_args) {
  for (uint32_t i = 0; i < candidate->slice_count; ++i) {
    loom_op_t* slice_op = candidate->slice_ops[i];
    uint32_t lane_index = 0;
    if (!loom_low_decompose_cfg_tuples_slice_matches_candidate(
            state, candidate, slice_op, &lane_index)) {
      continue;
    }
    loom_value_id_t replacement = lane_args[lane_index];
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        state->rewriter, slice_op, &replacement, 1));
    ++state->statistics->slices_removed;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_decompose_cfg_tuples_apply_candidate(
    loom_low_decompose_cfg_tuples_state_t* state,
    const loom_low_cfg_tuple_candidate_t* candidate) {
  loom_value_id_t* lane_args = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->analysis_arena, candidate->lane_count,
                                sizeof(*lane_args), (void**)&lane_args));

  IREE_RETURN_IF_ERROR(loom_low_decompose_cfg_tuples_define_lane_block_args(
      state, candidate, lane_args));
  IREE_RETURN_IF_ERROR(loom_low_decompose_cfg_tuples_replace_slices(
      state, candidate, lane_args));
  const uint16_t shifted_tuple_index =
      (uint16_t)((uint32_t)candidate->arg_index + candidate->lane_count);
  IREE_RETURN_IF_ERROR(loom_block_remove_arg(state->module, candidate->block,
                                             shifted_tuple_index));

  for (iree_host_size_t i = 0; i < candidate->predecessor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_decompose_cfg_tuples_rebuild_branch(
        state, candidate, candidate->predecessor_branches[i]));
  }

  ++state->statistics->block_args_decomposed;
  state->statistics->lane_block_args_inserted += candidate->lane_count;
  state->statistics->branch_edges_decomposed += candidate->predecessor_count;
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static iree_status_t loom_low_decompose_cfg_tuples_try_block_arg(
    loom_low_decompose_cfg_tuples_state_t* state, loom_region_t* body,
    loom_block_t* block, uint16_t arg_index, bool* out_changed) {
  *out_changed = false;
  const loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
  const loom_type_t arg_type = loom_module_value_type(state->module, arg_id);

  loom_type_t lane_type = loom_type_none();
  uint32_t lane_count = 0;
  if (!loom_low_decompose_cfg_tuples_get_register_tuple_type(
          arg_type, &lane_type, &lane_count)) {
    return iree_ok_status();
  }
  if ((uint64_t)block->arg_count - 1 + lane_count > UINT16_MAX) {
    return iree_ok_status();
  }

  loom_low_cfg_tuple_candidate_t candidate = {
      .block = block,
      .arg_index = arg_index,
      .arg_id = arg_id,
      .tuple_type = arg_type,
      .lane_type = lane_type,
      .lane_count = lane_count,
  };

  IREE_RETURN_IF_ERROR(loom_low_decompose_cfg_tuples_collect_predecessors(
      state, body, block, arg_index, arg_type, &candidate.predecessor_branches,
      &candidate.predecessor_count));
  if (candidate.predecessor_count == 0) {
    return iree_ok_status();
  }

  bool eligible = false;
  IREE_RETURN_IF_ERROR(loom_low_decompose_cfg_tuples_collect_slices(
      state, &candidate, &eligible));
  if (!eligible) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_low_decompose_cfg_tuples_apply_candidate(state, &candidate));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_low_decompose_cfg_tuples_process_once(
    loom_low_decompose_cfg_tuples_state_t* state, loom_func_like_t function,
    bool* out_changed) {
  *out_changed = false;
  loom_region_t* body = loom_func_like_body(function);
  for (uint16_t block_index = 1; block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = body->blocks[block_index];
    for (uint16_t reverse_index = block->arg_count; reverse_index > 0;
         --reverse_index) {
      const uint16_t arg_index = (uint16_t)(reverse_index - 1);
      iree_arena_reset(state->analysis_arena);
      bool changed = false;
      IREE_RETURN_IF_ERROR(loom_low_decompose_cfg_tuples_try_block_arg(
          state, body, block, arg_index, &changed));
      *out_changed |= changed;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_decompose_cfg_tuples_run(loom_pass_t* pass,
                                                loom_module_t* module,
                                                loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op) ||
      !loom_func_like_body(function)) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  iree_arena_allocator_t analysis_arena = {0};
  iree_arena_initialize(pass->arena->block_pool, &analysis_arena);

  loom_low_decompose_cfg_tuples_state_t state = {
      .pass = pass,
      .statistics = loom_low_decompose_cfg_tuples_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
      .analysis_arena = &analysis_arena,
  };

  bool any_changed = false;
  iree_status_t status = loom_low_decompose_cfg_tuples_process_once(
      &state, function, &any_changed);

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&analysis_arena);
  return status;
}
