// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/scf/canonicalize.h"
#include "loom/ops/scf/forwarding_equivalence.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"

static iree_status_t loom_scf_preserve_value_name(
    loom_module_t* module, loom_value_id_t old_value_id,
    loom_value_id_t new_value_id) {
  if (old_value_id == LOOM_VALUE_ID_INVALID ||
      new_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_module_copy_value_name(module, old_value_id, new_value_id);
}

//===----------------------------------------------------------------------===//
// scf.for
//===----------------------------------------------------------------------===//

static bool loom_scf_for_has_zero_trip_count(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  if (loom_scf_for_lower_bound(op) == loom_scf_for_upper_bound(op)) {
    return true;
  }
  loom_value_facts_t lower_bound =
      loom_rewriter_value_facts(rewriter, loom_scf_for_lower_bound(op));
  loom_value_facts_t upper_bound =
      loom_rewriter_value_facts(rewriter, loom_scf_for_upper_bound(op));
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound)) {
    return false;
  }
  if (loom_value_facts_is_exact(lower_bound) &&
      loom_value_facts_is_exact(upper_bound) &&
      lower_bound.range_lo == upper_bound.range_lo) {
    return true;
  }

  loom_value_facts_t step =
      loom_rewriter_value_facts(rewriter, loom_scf_for_step(op));
  if (!loom_value_facts_is_positive(step)) {
    return false;
  }
  return lower_bound.range_lo >= upper_bound.range_hi;
}

static bool loom_scf_for_step_is_positive(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_facts_t step =
      loom_rewriter_value_facts(rewriter, loom_scf_for_step(op));
  return !loom_value_facts_is_float(step) && loom_value_facts_is_positive(step);
}

static bool loom_scf_for_has_single_trip_count(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  loom_value_facts_t lower_bound =
      loom_rewriter_value_facts(rewriter, loom_scf_for_lower_bound(op));
  loom_value_facts_t upper_bound =
      loom_rewriter_value_facts(rewriter, loom_scf_for_upper_bound(op));
  loom_value_facts_t step =
      loom_rewriter_value_facts(rewriter, loom_scf_for_step(op));
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return false;
  }

  if (lower_bound.range_hi >= upper_bound.range_lo) {
    return false;
  }
  int64_t next_iv_lower_bound = 0;
  if (!iree_checked_add_i64(lower_bound.range_lo, step.range_lo,
                            &next_iv_lower_bound)) {
    return false;
  }
  return next_iv_lower_bound >= upper_bound.range_hi;
}

static bool loom_scf_for_yields_loop_carried_args(loom_op_t* op) {
  loom_region_t* body = loom_scf_for_body(op);
  loom_op_t* yield = loom_scf_region_terminator(body);
  if (!yield) {
    return false;
  }

  loom_block_t* block = loom_region_entry_block(body);
  if (block->first_op != yield) {
    return false;
  }
  if (op->result_count == 0) {
    return true;
  }

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return false;
  }
  if (block->arg_count < 1 + op->result_count) {
    return false;
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (yielded_values.values[i] != loom_block_arg_id(block, 1 + i)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_scf_for_inline_single_trip(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (!loom_scf_for_has_single_trip_count(op, rewriter)) {
    return iree_ok_status();
  }

  loom_region_t* body = loom_scf_for_body(op);
  loom_op_t* yield = loom_scf_region_terminator(body);
  if (!yield) {
    return iree_ok_status();
  }

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return iree_ok_status();
  }

  loom_block_t* block = loom_region_entry_block(body);
  if (block->arg_count < 1 + iter_args.count) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      rewriter, loom_block_arg_id(block, 0), loom_scf_for_lower_bound(op)));
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, loom_block_arg_id(block, (uint16_t)(1 + i)),
        iter_args.values[i]));
  }

  IREE_RETURN_IF_ERROR(
      loom_scf_move_region_body_before_op(rewriter, body, yield, op));

  loom_value_id_t* replacements = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, op->result_count, sizeof(*replacements),
        (void**)&replacements));
    yielded_values = loom_scf_yield_values(yield);
    memcpy(replacements, yielded_values.values,
           (iree_host_size_t)op->result_count * sizeof(*replacements));
  }
  return loom_scf_replace_results_and_erase(op, rewriter, replacements,
                                            op->result_count);
}

static iree_status_t loom_scf_for_adjust_tied_results(
    loom_op_t* op, const uint16_t* result_map, const uint16_t* iter_arg_map,
    uint16_t old_iter_arg_offset, uint16_t new_iter_arg_offset,
    iree_arena_allocator_t* scratch_arena,
    loom_tied_result_t** out_tied_results, uint16_t* out_tied_result_count,
    bool* out_supported) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  *out_supported = true;
  if (op->tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  uint16_t* result_tie_indices = NULL;
  uint16_t* operand_tie_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, op->tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, op->result_count, sizeof(*result_tie_indices),
      (void**)&result_tie_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, op->operand_count, sizeof(*operand_tie_indices),
      (void**)&operand_tie_indices));
  memset(result_tie_indices, 0xFF,
         (iree_host_size_t)op->result_count * sizeof(*result_tie_indices));
  memset(operand_tie_indices, 0xFF,
         (iree_host_size_t)op->operand_count * sizeof(*operand_tie_indices));

  const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
  uint16_t tied_result_count = 0;
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    loom_tied_result_t tied_result = old_tied_results[i];
    IREE_ASSERT_LT(tied_result.result_index, op->result_count);
    IREE_ASSERT_LT(tied_result.operand_index, op->operand_count);

    uint16_t new_result_index = result_map[tied_result.result_index];
    if (new_result_index == UINT16_MAX) {
      new_result_index = iter_arg_map[tied_result.result_index];
      if (new_result_index == UINT16_MAX) {
        // An unchanged result is replaced directly by its initial operand, so
        // there is no remaining operation result whose tie must be retained.
        continue;
      }
    }

    uint16_t new_operand_index = tied_result.operand_index;
    if (new_operand_index >= old_iter_arg_offset) {
      const uint16_t old_iter_arg_index =
          (uint16_t)(new_operand_index - old_iter_arg_offset);
      IREE_ASSERT_LT(old_iter_arg_index, op->result_count);
      const uint16_t new_iter_arg_index = iter_arg_map[old_iter_arg_index];
      if (new_iter_arg_index == UINT16_MAX) {
        *out_supported = false;
        return iree_ok_status();
      }
      new_operand_index = (uint16_t)(new_iter_arg_offset + new_iter_arg_index);
    }

    const loom_tied_result_t adjusted_tie = {
        .result_index = new_result_index,
        .operand_index = new_operand_index,
        .has_type_change = tied_result.has_type_change,
    };
    const uint16_t result_tie_index = result_tie_indices[new_result_index];
    if (result_tie_index != UINT16_MAX) {
      const loom_tied_result_t existing_tie = tied_results[result_tie_index];
      if (existing_tie.operand_index != adjusted_tie.operand_index ||
          existing_tie.has_type_change != adjusted_tie.has_type_change) {
        *out_supported = false;
        return iree_ok_status();
      }
      continue;
    }
    if (operand_tie_indices[new_operand_index] != UINT16_MAX) {
      *out_supported = false;
      return iree_ok_status();
    }
    result_tie_indices[new_result_index] = tied_result_count;
    operand_tie_indices[new_operand_index] = tied_result_count;
    tied_results[tied_result_count++] = adjusted_tie;
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

// Rebuilds an scf.for after a simultaneous carried-state compaction.
// |result_map| names kept columns and leaves removed results invalid.
// |iter_arg_map| names the kept column replacing each body argument; an invalid
// entry replaces that body argument and result with its original iter operand.
static iree_status_t loom_scf_for_rebuild_carried_state(
    loom_op_t* op, const uint16_t* result_map, const uint16_t* iter_arg_map,
    uint16_t kept_count, iree_arena_allocator_t* scratch_arena,
    loom_rewriter_t* rewriter, bool* out_rewritten) {
  *out_rewritten = false;
  loom_region_t* body = loom_scf_for_body(op);
  loom_block_t* old_block = loom_region_entry_block(body);
  loom_op_t* old_yield = loom_scf_region_terminator(body);
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(old_yield);

  const uint16_t old_iter_arg_offset =
      (uint16_t)(iter_args.values - loom_op_const_operands(op));
  const uint16_t new_iter_arg_offset = 3;

  // Builder inputs are copied into the new operations. Discard them as soon as
  // both operations exist so result replacement cannot increase peak scratch.
  iree_arena_checkpoint_t build_checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  loom_value_id_t* kept_iter_args = NULL;
  loom_value_id_t* kept_yielded_values = NULL;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, kept_count,
                                                   sizeof(*kept_iter_args),
                                                   (void**)&kept_iter_args));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, kept_count, sizeof(*kept_yielded_values),
        (void**)&kept_yielded_values));
  }

  for (uint16_t i = 0; i < op->result_count; ++i) {
    const uint16_t kept_ordinal = result_map[i];
    if (kept_ordinal == UINT16_MAX) {
      continue;
    }
    kept_iter_args[kept_ordinal] = iter_args.values[i];
    kept_yielded_values[kept_ordinal] = yielded_values.values[i];
  }

  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  bool tied_results_supported = false;
  IREE_RETURN_IF_ERROR(loom_scf_for_adjust_tied_results(
      op, result_map, iter_arg_map, old_iter_arg_offset, new_iter_arg_offset,
      scratch_arena, &tied_results, &tied_result_count,
      &tied_results_supported));
  if (!tied_results_supported) {
    iree_arena_checkpoint_restore(&build_checkpoint);
    return iree_ok_status();
  }

  loom_scf_for_build_flags_t build_flags = 0;
  loom_value_id_t pipeline_depth = LOOM_VALUE_ID_INVALID;
  if (loom_scf_for_pipeline_depth_is_present(op)) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_PIPELINE_DEPTH;
    pipeline_depth = loom_scf_for_pipeline_depth(op);
  }
  loom_value_id_t unroll_factor = LOOM_VALUE_ID_INVALID;
  if (loom_scf_for_unroll_factor_is_present(op)) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    unroll_factor = loom_scf_for_unroll_factor(op);
  }
  loom_scf_for_unroll_policy_t unroll_policy = 0;
  if (loom_scf_for_has_unroll_policy(op)) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    unroll_policy = loom_scf_for_unroll_policy(op);
  }
  loom_scf_for_unroll_schedule_t unroll_schedule = 0;
  if (loom_scf_for_has_unroll_schedule(op)) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_SCHEDULE;
    unroll_schedule = loom_scf_for_unroll_schedule(op);
  }

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_loop = NULL;
  iree_status_t status = loom_scf_for_build(
      &rewriter->builder, build_flags, loom_scf_for_lower_bound(op),
      loom_scf_for_upper_bound(op), loom_scf_for_step(op), kept_iter_args,
      kept_count, /*result_types=*/NULL, tied_results, tied_result_count,
      pipeline_depth, unroll_factor, unroll_policy, unroll_schedule,
      op->location, &new_loop);
  loom_op_t* new_yield = NULL;
  if (iree_status_is_ok(status)) {
    loom_region_t* new_body = loom_scf_for_body(new_loop);
    loom_builder_ip_t saved_ip =
        loom_builder_enter_region(&rewriter->builder, new_loop, new_body);
    status = loom_scf_yield_build(&rewriter->builder, kept_yielded_values,
                                  kept_count, op->location, &new_yield);
    loom_builder_restore(&rewriter->builder, saved_ip);
  }
  iree_arena_checkpoint_restore(&build_checkpoint);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  loom_block_t* new_block =
      loom_region_entry_block(loom_scf_for_body(new_loop));
  IREE_RETURN_IF_ERROR(loom_scf_preserve_value_name(
      rewriter->module, loom_block_arg_id(old_block, 0),
      loom_block_arg_id(new_block, 0)));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const uint16_t kept_ordinal = result_map[i];
    if (kept_ordinal == UINT16_MAX) {
      continue;
    }
    // Preserve the declared tuple before replacing its identities. Initial
    // operands can have different dependent dimensions from the recurring
    // body and result values. Indexed replacement below updates every peer
    // reference, including references to removed invariant columns.
    const loom_value_id_t new_argument =
        loom_block_arg_id(new_block, (uint16_t)(1 + kept_ordinal));
    const loom_type_t argument_type = loom_module_value_type(
        rewriter->module, loom_block_arg_id(old_block, (uint16_t)(1 + i)));
    if (!loom_type_equal(loom_module_value_type(rewriter->module, new_argument),
                         argument_type)) {
      IREE_RETURN_IF_ERROR(loom_module_set_value_type(
          rewriter->module, new_argument, argument_type));
    }
    const loom_value_id_t new_result = loom_op_results(new_loop)[kept_ordinal];
    const loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_op_results(op)[i]);
    if (!loom_type_equal(loom_module_value_type(rewriter->module, new_result),
                         result_type)) {
      IREE_RETURN_IF_ERROR(loom_module_set_value_type(rewriter->module,
                                                      new_result, result_type));
    }
    IREE_RETURN_IF_ERROR(loom_scf_preserve_value_name(
        rewriter->module, loom_block_arg_id(old_block, (uint16_t)(1 + i)),
        loom_block_arg_id(new_block, (uint16_t)(1 + kept_ordinal))));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      rewriter, loom_block_arg_id(old_block, 0),
      loom_block_arg_id(new_block, 0)));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t old_arg =
        loom_block_arg_id(old_block, (uint16_t)(1 + i));
    const uint16_t replacement_ordinal =
        result_map[i] != UINT16_MAX ? result_map[i] : iter_arg_map[i];
    const loom_value_id_t replacement =
        replacement_ordinal == UINT16_MAX
            ? iter_args.values[i]
            : loom_block_arg_id(new_block, (uint16_t)(1 + replacement_ordinal));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_with(rewriter, old_arg, replacement));
  }

  loom_op_t* child_op = old_block->first_op;
  while (child_op && child_op != old_yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, child_op, new_yield));
    child_op = next_child_op;
  }

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  loom_value_slice_t new_results = loom_scf_for_results(new_loop);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const uint16_t replacement_ordinal =
        result_map[i] != UINT16_MAX ? result_map[i] : iter_arg_map[i];
    replacements[i] = replacement_ordinal == UINT16_MAX
                          ? iter_args.values[i]
                          : new_results.values[replacement_ordinal];
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, op->result_count));
  *out_rewritten = true;
  return iree_ok_status();
}

static bool loom_scf_for_carried_slot_types_equal(
    const loom_module_t* module, const loom_block_t* body_block,
    loom_value_slice_t iter_args, loom_value_slice_t results, uint16_t lhs,
    uint16_t rhs) {
  return loom_type_equal(
             loom_module_value_type(module, iter_args.values[lhs]),
             loom_module_value_type(module, iter_args.values[rhs])) &&
         loom_type_equal(loom_block_arg_type(module, body_block, 1 + lhs),
                         loom_block_arg_type(module, body_block, 1 + rhs)) &&
         loom_type_equal(loom_module_value_type(module, results.values[lhs]),
                         loom_module_value_type(module, results.values[rhs]));
}

static bool loom_scf_for_carried_slot_is_unchanged(
    const loom_block_t* body_block, loom_value_slice_t yielded_values,
    uint16_t index) {
  return yielded_values.values[index] ==
         loom_block_arg_id(body_block, (uint16_t)(1 + index));
}

// Compacts recurrence columns that are exactly equal for every possible trip
// count. Initial identities color states, direct body-argument yields form
// recurrence edges, and every other yield is an opaque terminal identity.
static iree_status_t loom_scf_for_compact_equivalent_carried_state(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_rewritten) {
  *out_rewritten = false;
  if (op->result_count < 2) {
    return iree_ok_status();
  }

  loom_region_t* body = loom_scf_for_body(op);
  loom_op_t* yield = loom_scf_region_terminator(body);
  loom_block_t* block = loom_region_entry_block(body);
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  loom_value_slice_t results = loom_scf_for_results(op);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(rewriter->arena->block_pool, &scratch_arena);
  uint16_t* successors = NULL;
  uint16_t* representatives = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&scratch_arena, op->result_count,
                                sizeof(*successors), (void**)&successors);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, op->result_count,
                                       sizeof(*representatives),
                                       (void**)&representatives);
  }

  if (iree_status_is_ok(status)) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      successors[i] = LOOM_SCF_FORWARDING_TERMINAL;
      const loom_value_t* yielded =
          loom_module_value(rewriter->module, yielded_values.values[i]);
      if (!loom_value_is_block_arg(yielded) ||
          loom_value_def_block(yielded) != block) {
        continue;
      }
      const uint16_t argument_index = loom_value_def_index(yielded);
      if (argument_index > 0 && argument_index <= op->result_count) {
        successors[i] = (uint16_t)(argument_index - 1);
      }
    }

    iree_arena_checkpoint_t proof_checkpoint =
        iree_arena_checkpoint_save(&scratch_arena);
    uint16_t class_count = 0;
    status = loom_scf_forwarding_equivalence_partition(
        (loom_scf_forwarding_equivalence_problem_t){
            .fact_table = rewriter->fact_table,
            .initial_values = iter_args.values,
            .yielded_values = yielded_values.values,
            .successors = successors,
            .count = op->result_count,
        },
        &scratch_arena, representatives, &class_count);
    iree_arena_checkpoint_restore(&proof_checkpoint);

    if (iree_status_is_ok(status) && class_count < op->result_count) {
      // The consumed successor array becomes the kept-result map.
      uint16_t* result_map = successors;
      uint16_t* iter_arg_map = NULL;
      status = iree_arena_allocate_array(&scratch_arena, op->result_count,
                                         sizeof(*iter_arg_map),
                                         (void**)&iter_arg_map);
      if (iree_status_is_ok(status)) {
        for (uint16_t i = 0; i < op->result_count; ++i) {
          result_map[i] = UINT16_MAX;
          iter_arg_map[i] = UINT16_MAX;
        }

        uint16_t kept_count = 0;
        uint16_t duplicate_count = 0;
        for (uint16_t i = 0; i < op->result_count; ++i) {
          const uint16_t representative = representatives[i];
          // Let the direct self-forward cleanup delete an entire unchanged
          // class in one rebuild instead of first merging and then deleting it.
          const bool unchanged_class =
              loom_scf_for_carried_slot_is_unchanged(block, yielded_values,
                                                     representative) &&
              loom_scf_for_carried_slot_is_unchanged(block, yielded_values, i);
          if (representative == i || unchanged_class ||
              !loom_scf_for_carried_slot_types_equal(rewriter->module, block,
                                                     iter_args, results, i,
                                                     representative)) {
            result_map[i] = kept_count;
            iter_arg_map[i] = kept_count++;
          } else {
            iter_arg_map[i] = result_map[representative];
            ++duplicate_count;
          }
        }

        if (duplicate_count > 0) {
          status = loom_scf_for_rebuild_carried_state(
              op, result_map, iter_arg_map, kept_count, &scratch_arena,
              rewriter, out_rewritten);
        }
      }
    }
  }

  iree_arena_deinitialize(&scratch_arena);
  return status;
}

// Removes carried columns whose yield is their own body argument. Kept columns
// can still reference removed columns through ordinary captures; a kept tied
// result makes the rewrite inapplicable when its storage operand is removed.
static iree_status_t loom_scf_for_remove_unchanged_carried_state(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (!loom_scf_for_step_is_positive(op, rewriter) || op->result_count == 0) {
    return iree_ok_status();
  }

  loom_region_t* body = loom_scf_for_body(op);
  loom_op_t* yield = loom_scf_region_terminator(body);
  if (!yield) {
    return iree_ok_status();
  }
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return iree_ok_status();
  }

  loom_block_t* block = loom_region_entry_block(body);
  if (block->arg_count < 1 + op->result_count) {
    return iree_ok_status();
  }

  uint16_t removed_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (loom_scf_for_carried_slot_is_unchanged(block, yielded_values, i)) {
      ++removed_count;
    }
  }
  if (removed_count == 0) {
    return iree_ok_status();
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(rewriter->arena->block_pool, &scratch_arena);
  uint16_t* result_map = NULL;
  uint16_t* iter_arg_map = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&scratch_arena, op->result_count,
                                sizeof(*result_map), (void**)&result_map);
  if (iree_status_is_ok(status)) {
    status =
        iree_arena_allocate_array(&scratch_arena, op->result_count,
                                  sizeof(*iter_arg_map), (void**)&iter_arg_map);
  }

  if (iree_status_is_ok(status)) {
    uint16_t kept_count = 0;
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_map[i] = UINT16_MAX;
      iter_arg_map[i] = UINT16_MAX;
      if (!loom_scf_for_carried_slot_is_unchanged(block, yielded_values, i)) {
        result_map[i] = kept_count;
        iter_arg_map[i] = kept_count++;
      }
    }
    bool rewritten = false;
    status = loom_scf_for_rebuild_carried_state(
        op, result_map, iter_arg_map,
        (uint16_t)(op->result_count - removed_count), &scratch_arena, rewriter,
        &rewritten);
  }

  iree_arena_deinitialize(&scratch_arena);
  return status;
}

iree_status_t loom_scf_for_canonicalize(loom_op_t* op,
                                        loom_rewriter_t* rewriter) {
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  if (loom_scf_for_has_zero_trip_count(op, rewriter)) {
    return loom_scf_replace_results_and_erase(op, rewriter, iter_args.values,
                                              iter_args.count);
  }
  IREE_RETURN_IF_ERROR(loom_scf_for_inline_single_trip(op, rewriter));
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  if (loom_scf_for_step_is_positive(op, rewriter) &&
      loom_scf_for_yields_loop_carried_args(op)) {
    return loom_scf_replace_results_and_erase(op, rewriter, iter_args.values,
                                              iter_args.count);
  }
  if (loom_scf_for_step_is_positive(op, rewriter)) {
    bool rewritten = false;
    IREE_RETURN_IF_ERROR(loom_scf_for_compact_equivalent_carried_state(
        op, rewriter, &rewritten));
    if (rewritten) {
      return iree_ok_status();
    }
  }
  return loom_scf_for_remove_unchanged_carried_state(op, rewriter);
}

//===----------------------------------------------------------------------===//
// scf.while
//===----------------------------------------------------------------------===//

iree_status_t loom_scf_while_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_region_t* before = loom_scf_while_before(op);
  loom_block_t* block = loom_region_entry_block(before);
  loom_op_t* condition = block->last_op;
  if (!condition || !loom_scf_condition_isa(condition)) {
    return iree_ok_status();
  }
  bool continues = true;
  if (!loom_value_facts_as_exact_bool(
          loom_rewriter_value_facts(rewriter,
                                    loom_scf_condition_condition(condition)),
          &continues) ||
      continues) {
    return iree_ok_status();
  }

  // The before region executes once even when the body is unreachable. Its
  // forwarded values can differ from the initial tuple or depend on effects.
  const loom_value_slice_t initial = loom_scf_while_iter_args(op);
  for (uint16_t i = 0; i < initial.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, loom_block_arg_id(block, i), initial.values[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_scf_move_region_body_before_op(rewriter, before, condition, op));
  const loom_value_slice_t forwarded = loom_scf_condition_forwarded(condition);
  return loom_scf_replace_results_and_erase(op, rewriter, forwarded.values,
                                            forwarded.count);
}
