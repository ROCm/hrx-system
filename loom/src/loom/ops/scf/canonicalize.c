// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/analysis/availability.h"
#include "loom/ir/attribute.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_scf_value_facts_are_exact_i64(loom_rewriter_t* rewriter,
                                               loom_value_id_t value,
                                               int64_t* out_value) {
  return loom_value_facts_as_exact_i64(
      loom_rewriter_value_facts(rewriter, value), out_value);
}

static bool loom_scf_lookup_selected_row(loom_rewriter_t* rewriter,
                                         loom_op_t* op,
                                         iree_host_size_t* out_row_index) {
  int64_t selector = 0;
  if (!loom_scf_value_facts_are_exact_i64(
          rewriter, loom_scf_lookup_selector(op), &selector)) {
    return false;
  }
  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] == selector) {
      *out_row_index = i;
      return true;
    }
  }
  *out_row_index = case_keys.count;
  return true;
}

static bool loom_scf_lookup_has_well_formed_table_shape(loom_op_t* op) {
  if (op->result_count == 0) return false;
  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY ||
      (case_keys.count > 0 && !case_keys.i64_array)) {
    return false;
  }
  loom_value_slice_t values = loom_scf_lookup_values(op);
  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  return values.count == row_count * (iree_host_size_t)op->result_count;
}

static bool loom_scf_lookup_key_matches_selector_facts(
    loom_value_facts_t selector_facts, int64_t key) {
  if (loom_value_facts_is_float(selector_facts)) return true;
  if (key < selector_facts.range_lo || key > selector_facts.range_hi) {
    return false;
  }
  int64_t divisor = selector_facts.known_divisor;
  return divisor <= 1 || key % divisor == 0;
}

static bool loom_scf_lookup_key_is_explicit(loom_attribute_t case_keys,
                                            int64_t key) {
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] == key) return true;
    if (case_keys.i64_array[i] > key) return false;
  }
  return false;
}

static bool loom_scf_lookup_default_row_may_match(
    loom_value_facts_t selector_facts, loom_attribute_t case_keys) {
  if (loom_value_facts_is_float(selector_facts)) return true;
  if (loom_value_facts_is_exact(selector_facts)) {
    return !loom_scf_lookup_key_is_explicit(case_keys, selector_facts.range_lo);
  }

  int64_t span = 0;
  if (!loom_checked_sub_i64(selector_facts.range_hi, selector_facts.range_lo,
                            &span) ||
      span < 0 || span > 4096) {
    return true;
  }

  for (int64_t offset = 0; offset <= span; ++offset) {
    int64_t candidate = selector_facts.range_lo + offset;
    if (!loom_scf_lookup_key_matches_selector_facts(selector_facts,
                                                    candidate)) {
      continue;
    }
    if (!loom_scf_lookup_key_is_explicit(case_keys, candidate)) return true;
  }
  return false;
}

static void loom_scf_lookup_mark_reachable_rows(
    loom_op_t* op, loom_value_facts_t selector_facts, bool* reachable_rows,
    iree_host_size_t* out_reachable_count,
    iree_host_size_t* out_reachable_explicit_count,
    iree_host_size_t* out_first_reachable_row) {
  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    reachable_rows[i] = false;
  }

  *out_reachable_count = 0;
  *out_reachable_explicit_count = 0;
  *out_first_reachable_row = row_count;
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    if (!loom_scf_lookup_key_matches_selector_facts(selector_facts,
                                                    case_keys.i64_array[i])) {
      continue;
    }
    reachable_rows[i] = true;
    if (*out_reachable_count == 0) *out_first_reachable_row = i;
    ++*out_reachable_count;
    ++*out_reachable_explicit_count;
  }

  iree_host_size_t default_row = case_keys.count;
  if (loom_scf_lookup_default_row_may_match(selector_facts, case_keys)) {
    reachable_rows[default_row] = true;
    if (*out_reachable_count == 0) *out_first_reachable_row = default_row;
    ++*out_reachable_count;
  }
}

static loom_op_t* loom_scf_region_terminator(loom_region_t* region) {
  if (!region || region->block_count != 1) return NULL;
  loom_block_t* block = loom_region_entry_block(region);
  if (!block || !block->last_op || !loom_scf_yield_isa(block->last_op)) {
    return NULL;
  }
  return block->last_op;
}

static iree_status_t loom_scf_move_region_body_before_op(
    loom_rewriter_t* rewriter, loom_region_t* region, loom_op_t* old_yield,
    loom_op_t* before_op) {
  loom_block_t* block = loom_region_entry_block(region);
  if (!block) return iree_ok_status();
  loom_op_t* child_op = block->first_op;
  while (child_op && child_op != old_yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, child_op, before_op));
    child_op = next_child_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_replace_results_and_erase(
    loom_op_t* op, loom_rewriter_t* rewriter,
    const loom_value_id_t* replacements, uint16_t replacement_count) {
  if (replacement_count != op->result_count) return iree_ok_status();
  if (op->result_count == 0) return loom_rewriter_erase(rewriter, op);
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                  replacement_count);
}

static bool loom_scf_lookup_reachable_column_is_uniform(
    loom_op_t* op, const bool* reachable_rows, uint16_t column,
    loom_value_id_t* out_value) {
  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  loom_value_slice_t values = loom_scf_lookup_values(op);
  uint16_t result_count = op->result_count;
  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  loom_value_id_t first_value = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t row = 0; row < row_count; ++row) {
    if (!reachable_rows[row]) continue;
    loom_value_id_t value = values.values[row * result_count + column];
    if (first_value == LOOM_VALUE_ID_INVALID) {
      first_value = value;
      continue;
    }
    if (value != first_value) {
      return false;
    }
  }
  if (first_value == LOOM_VALUE_ID_INVALID) return false;
  *out_value = first_value;
  return true;
}

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
// scf.select
//===----------------------------------------------------------------------===//

iree_status_t loom_scf_select_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_value_id_t true_value = loom_scf_select_true_value(op);
  loom_value_id_t false_value = loom_scf_select_false_value(op);
  if (true_value == false_value) {
    return loom_scf_replace_results_and_erase(op, rewriter, &true_value, 1);
  }

  bool condition = false;
  if (!loom_value_facts_as_exact_bool(
          loom_rewriter_value_facts(rewriter, loom_scf_select_condition(op)),
          &condition)) {
    loom_value_id_t result = loom_scf_select_result(op);
    loom_type_t result_type = loom_module_value_type(rewriter->module, result);
    int64_t true_i64 = 0;
    int64_t false_i64 = 0;
    if (loom_type_is_scalar(result_type) &&
        loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1 &&
        loom_scf_value_facts_are_exact_i64(rewriter, true_value, &true_i64) &&
        true_i64 == 1 &&
        loom_scf_value_facts_are_exact_i64(rewriter, false_value, &false_i64) &&
        false_i64 == 0) {
      loom_value_id_t condition_value = loom_scf_select_condition(op);
      return loom_scf_replace_results_and_erase(op, rewriter, &condition_value,
                                                1);
    }
    return iree_ok_status();
  }
  loom_value_id_t replacement = condition ? true_value : false_value;
  return loom_scf_replace_results_and_erase(op, rewriter, &replacement, 1);
}

//===----------------------------------------------------------------------===//
// scf.lookup
//===----------------------------------------------------------------------===//

static iree_status_t loom_scf_lookup_replace_with_row(
    loom_op_t* op, iree_host_size_t row_index, loom_rewriter_t* rewriter) {
  loom_value_slice_t values = loom_scf_lookup_values(op);
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  iree_host_size_t row_offset = row_index * op->result_count;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    replacements[i] = values.values[row_offset + i];
  }
  return loom_scf_replace_results_and_erase(op, rewriter, replacements,
                                            op->result_count);
}

static iree_status_t loom_scf_lookup_compact_uniform_columns(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  loom_value_slice_t old_values = loom_scf_lookup_values(op);
  const loom_value_id_t* old_results = loom_op_const_results(op);
  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;

  bool* reachable_rows = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, row_count,
                                                 sizeof(*reachable_rows),
                                                 (void**)&reachable_rows));
  loom_value_facts_t selector_facts =
      loom_rewriter_value_facts(rewriter, loom_scf_lookup_selector(op));
  iree_host_size_t reachable_count = 0;
  iree_host_size_t reachable_explicit_count = 0;
  iree_host_size_t first_reachable_row = row_count;
  loom_scf_lookup_mark_reachable_rows(
      op, selector_facts, reachable_rows, &reachable_count,
      &reachable_explicit_count, &first_reachable_row);
  if (reachable_count == 0) return iree_ok_status();

  bool* forwarded_results = NULL;
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*forwarded_results),
      (void**)&forwarded_results));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));

  uint16_t forwarded_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    forwarded_results[i] = loom_scf_lookup_reachable_column_is_uniform(
        op, reachable_rows, i, &replacements[i]);
    if (forwarded_results[i]) ++forwarded_count;
  }
  bool default_reachable = reachable_rows[case_keys.count];
  bool default_row_matches_dead_source = true;
  if (!default_reachable) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (forwarded_results[i]) continue;
      loom_value_id_t default_value =
          old_values.values[case_keys.count * op->result_count + i];
      loom_value_id_t source_value =
          old_values.values[first_reachable_row * op->result_count + i];
      if (default_value != source_value) {
        default_row_matches_dead_source = false;
        break;
      }
    }
  }
  if (forwarded_count == 0 && reachable_explicit_count == case_keys.count &&
      (default_reachable || default_row_matches_dead_source)) {
    return iree_ok_status();
  }
  if (forwarded_count == op->result_count) {
    return loom_scf_replace_results_and_erase(op, rewriter, replacements,
                                              op->result_count);
  }

  uint16_t kept_count = (uint16_t)(op->result_count - forwarded_count);
  int64_t* kept_case_keys = NULL;
  loom_type_t* kept_result_types = NULL;
  loom_value_id_t* kept_values = NULL;
  if (reachable_explicit_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, reachable_explicit_count, sizeof(*kept_case_keys),
        (void**)&kept_case_keys));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                 sizeof(*kept_result_types),
                                                 (void**)&kept_result_types));
  iree_host_size_t kept_row_count = reachable_explicit_count + 1;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, kept_row_count * kept_count,
                                sizeof(*kept_values), (void**)&kept_values));

  uint16_t kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (forwarded_results[i]) continue;
    kept_result_types[kept_ordinal] =
        loom_module_value_type(rewriter->module, old_results[i]);
    ++kept_ordinal;
  }

  iree_host_size_t new_row = 0;
  for (uint16_t row = 0; row < case_keys.count; ++row) {
    if (!reachable_rows[row]) continue;
    kept_case_keys[new_row] = case_keys.i64_array[row];
    kept_ordinal = 0;
    for (uint16_t column = 0; column < op->result_count; ++column) {
      if (forwarded_results[column]) continue;
      kept_values[new_row * kept_count + kept_ordinal] =
          old_values.values[row * op->result_count + column];
      ++kept_ordinal;
    }
    ++new_row;
  }

  iree_host_size_t default_source_row =
      reachable_rows[case_keys.count] ? case_keys.count : first_reachable_row;
  kept_ordinal = 0;
  for (uint16_t column = 0; column < op->result_count; ++column) {
    if (forwarded_results[column]) continue;
    kept_values[new_row * kept_count + kept_ordinal] =
        old_values.values[default_source_row * op->result_count + column];
    ++kept_ordinal;
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_lookup = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_lookup_build(
      &rewriter->builder, loom_scf_lookup_selector(op), kept_case_keys,
      reachable_explicit_count, kept_values, kept_row_count * kept_count,
      kept_result_types, kept_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, op->location, &new_lookup));

  kept_ordinal = 0;
  loom_value_slice_t new_results = loom_scf_lookup_results(new_lookup);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (!forwarded_results[i]) {
      replacements[i] = new_results.values[kept_ordinal++];
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                  op->result_count);
}

iree_status_t loom_scf_lookup_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  if (op->tied_result_count != 0 ||
      !loom_scf_lookup_has_well_formed_table_shape(op)) {
    return iree_ok_status();
  }

  iree_host_size_t selected_row = 0;
  if (loom_scf_lookup_selected_row(rewriter, op, &selected_row)) {
    return loom_scf_lookup_replace_with_row(op, selected_row, rewriter);
  }
  return loom_scf_lookup_compact_uniform_columns(op, rewriter);
}

//===----------------------------------------------------------------------===//
// RegionBranch tail factoring
//===----------------------------------------------------------------------===//

static loom_op_t* loom_scf_region_branch_tail_op_from_yielded_value(
    const loom_module_t* module, loom_block_t* block, loom_op_t* yield_op,
    loom_value_id_t yielded_value) {
  if (yielded_value == LOOM_VALUE_ID_INVALID ||
      yielded_value >= module->values.count) {
    return NULL;
  }
  const loom_value_t* value = loom_module_value(module, yielded_value);
  if (loom_value_is_block_arg(value) || loom_value_def_index(value) != 0) {
    return NULL;
  }
  loom_op_t* tail_op = loom_value_def_op(value);
  if (!tail_op || tail_op->parent_block != block ||
      tail_op->next_op != yield_op) {
    return NULL;
  }
  return tail_op;
}

static bool loom_scf_region_branch_tail_result_has_single_yield_use(
    const loom_module_t* module, const loom_op_t* tail_op) {
  loom_value_id_t result = loom_op_const_results(tail_op)[0];
  if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, result);
  return loom_value_has_single_use(value) &&
         !loom_module_value_has_type_uses(module, result);
}

static bool loom_scf_region_branch_tail_attrs_match(const loom_op_t* lhs,
                                                    const loom_op_t* rhs) {
  if (lhs->attribute_count != rhs->attribute_count) return false;
  const loom_attribute_t* lhs_attrs = loom_op_const_attrs(lhs);
  const loom_attribute_t* rhs_attrs = loom_op_const_attrs(rhs);
  for (uint8_t i = 0; i < lhs->attribute_count; ++i) {
    if (!loom_attribute_equal(&lhs_attrs[i], &rhs_attrs[i])) return false;
  }
  return true;
}

static bool loom_scf_region_branch_tail_op_can_factor(const loom_op_t* op) {
  // Common-tail factoring is not speculation: exactly one matching tail already
  // executed on every branch path, and branch-local work remains before the new
  // yield. Result-producing side-effecting tails are therefore legal here.
  return op && !iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) &&
         op->result_count == 1 && op->region_count == 0 &&
         op->tied_result_count == 0;
}

static bool loom_scf_type_is_storage_root_or_projection(loom_type_t type) {
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_VIEW:
    case LOOM_TYPE_BUFFER:
    case LOOM_TYPE_STORAGE:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_scf_region_branch_tail_ops_match(
    loom_rewriter_t* rewriter, const loom_op_t* branch_op,
    const loom_op_t* reference_tail, const loom_op_t* candidate_tail,
    const loom_availability_analysis_t* availability, bool* out_match) {
  *out_match = false;
  if (reference_tail->kind != candidate_tail->kind ||
      reference_tail->instance_flags != candidate_tail->instance_flags ||
      reference_tail->operand_count != candidate_tail->operand_count ||
      !loom_scf_region_branch_tail_op_can_factor(reference_tail) ||
      !loom_scf_region_branch_tail_op_can_factor(candidate_tail)) {
    return iree_ok_status();
  }
  if (!loom_scf_region_branch_tail_result_has_single_yield_use(
          rewriter->module, reference_tail) ||
      !loom_scf_region_branch_tail_result_has_single_yield_use(
          rewriter->module, candidate_tail)) {
    return iree_ok_status();
  }
  if (!loom_scf_region_branch_tail_attrs_match(reference_tail,
                                               candidate_tail)) {
    return iree_ok_status();
  }
  bool attrs_available = false;
  IREE_RETURN_IF_ERROR(loom_availability_op_attrs_are_available_before_op(
      availability, /*moving_root_op=*/NULL, branch_op, reference_tail,
      &attrs_available));
  if (!attrs_available) return iree_ok_status();

  loom_value_id_t branch_result = loom_op_const_results(branch_op)[0];
  if (branch_result == LOOM_VALUE_ID_INVALID ||
      branch_result >= rewriter->module->values.count) {
    return iree_ok_status();
  }
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, branch_result);
  if (!loom_type_equal(
          result_type,
          loom_module_value_type(rewriter->module,
                                 loom_op_const_results(reference_tail)[0])) ||
      !loom_type_equal(
          result_type,
          loom_module_value_type(rewriter->module,
                                 loom_op_const_results(candidate_tail)[0]))) {
    return iree_ok_status();
  }

  const loom_value_id_t* reference_operands =
      loom_op_const_operands(reference_tail);
  const loom_value_id_t* candidate_operands =
      loom_op_const_operands(candidate_tail);
  for (uint16_t i = 0; i < reference_tail->operand_count; ++i) {
    if (reference_operands[i] == LOOM_VALUE_ID_INVALID ||
        reference_operands[i] >= rewriter->module->values.count ||
        candidate_operands[i] == LOOM_VALUE_ID_INVALID ||
        candidate_operands[i] >= rewriter->module->values.count) {
      return iree_ok_status();
    }
    loom_type_t reference_type =
        loom_module_value_type(rewriter->module, reference_operands[i]);
    if (!loom_type_equal(
            reference_type,
            loom_module_value_type(rewriter->module, candidate_operands[i]))) {
      return iree_ok_status();
    }
    // Selecting between distinct storage roots/projections erases the
    // single-root facts needed by later memory lowering. Keep those accesses
    // branch-local unless the storage operand is already common to every
    // branch.
    if (reference_operands[i] != candidate_operands[i] &&
        loom_scf_type_is_storage_root_or_projection(reference_type)) {
      return iree_ok_status();
    }
    bool type_available = false;
    IREE_RETURN_IF_ERROR(loom_availability_type_is_available_before_op(
        availability, /*moving_root_op=*/NULL, branch_op, reference_type,
        &type_available));
    if (!type_available) return iree_ok_status();
  }
  *out_match = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_region_branch_build_empty_clone(
    loom_rewriter_t* rewriter, const loom_op_t* branch_op,
    const loom_type_t* result_types, uint16_t result_count,
    loom_op_t** out_new_branch) {
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &rewriter->builder, branch_op->kind, branch_op->operand_count,
      result_count, branch_op->region_count, /*tied_result_count=*/0,
      branch_op->attribute_count, branch_op->location, out_new_branch));
  (*out_new_branch)->instance_flags = branch_op->instance_flags;
  (*out_new_branch)->traits = branch_op->traits;

  if (branch_op->operand_count > 0) {
    memcpy(
        loom_op_operands(*out_new_branch), loom_op_const_operands(branch_op),
        (iree_host_size_t)branch_op->operand_count * sizeof(loom_value_id_t));
  }
  if (branch_op->attribute_count > 0) {
    memcpy(loom_op_attrs(*out_new_branch), loom_op_const_attrs(branch_op),
           (iree_host_size_t)branch_op->attribute_count *
               sizeof(loom_attribute_t));
  }

  loom_region_t** new_regions = loom_op_regions(*out_new_branch);
  for (uint8_t i = 0; i < branch_op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_module_allocate_region(rewriter->module, 1, &new_regions[i]));
  }

  loom_value_id_t* new_results = loom_op_results(*out_new_branch);
  for (uint16_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_value(
        &rewriter->builder, result_types[i], &new_results[i]));
  }
  return loom_builder_finalize_op(&rewriter->builder, *out_new_branch);
}

static iree_status_t loom_scf_region_branch_build_operand_yield(
    loom_rewriter_t* rewriter, loom_region_branch_t branch,
    uint8_t region_index, const loom_value_id_t* values, uint16_t count,
    loom_location_id_t location, loom_op_t** out_yield) {
  loom_region_t* region =
      loom_region_branch_region(rewriter->module, branch, region_index);
  if (!region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region branch operand yield has no region");
  }
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&rewriter->builder, branch.op, region);
  iree_status_t status = loom_region_branch_build_region_terminator(
      &rewriter->builder, rewriter->module, branch, region_index, values, count,
      location, out_yield);
  loom_builder_restore(&rewriter->builder, saved_ip);
  return status;
}

static iree_status_t loom_scf_region_branch_clone_tail_after_branch(
    loom_rewriter_t* rewriter, const loom_op_t* branch_op,
    const loom_op_t* tail_op, loom_op_t* new_branch, loom_type_t result_type,
    loom_op_t** out_cloned_tail) {
  loom_builder_set_after(&rewriter->builder, new_branch);
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      &rewriter->builder, tail_op->kind, tail_op->operand_count,
      tail_op->result_count, 0, 0, tail_op->attribute_count, tail_op->location,
      out_cloned_tail));
  (*out_cloned_tail)->instance_flags = tail_op->instance_flags;
  (*out_cloned_tail)->traits = tail_op->traits;

  loom_value_slice_t new_branch_results = {
      .values = loom_op_results(new_branch),
      .count = new_branch->result_count,
  };
  if (tail_op->operand_count > 0) {
    memcpy(loom_op_operands(*out_cloned_tail), new_branch_results.values,
           (iree_host_size_t)tail_op->operand_count * sizeof(loom_value_id_t));
  }
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_define_value(&rewriter->builder, result_type, &result));
  loom_op_results(*out_cloned_tail)[0] = result;
  if (tail_op->attribute_count > 0) {
    memcpy(
        loom_op_attrs(*out_cloned_tail), loom_op_const_attrs(tail_op),
        (iree_host_size_t)tail_op->attribute_count * sizeof(loom_attribute_t));
  }
  return loom_builder_finalize_op(&rewriter->builder, *out_cloned_tail);
}

static iree_status_t loom_scf_region_branch_factor_common_tail(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (op->result_count != 1 || op->tied_result_count != 0) {
    return iree_ok_status();
  }
  loom_region_branch_t branch = loom_region_branch_cast(rewriter->module, op);
  if (!loom_region_branch_isa(branch) || op->region_count == 0) {
    return iree_ok_status();
  }

  loom_op_t** tail_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->region_count, sizeof(*tail_ops), (void**)&tail_ops));

  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region =
        loom_region_branch_region(rewriter->module, branch, i);
    if (!region || region->block_count != 1) return iree_ok_status();
    loom_block_t* block = loom_region_entry_block(region);
    loom_op_t* terminator =
        loom_region_branch_region_terminator(rewriter->module, branch, i);
    if (!block || !terminator || terminator->operand_count != 1) {
      return iree_ok_status();
    }
    tail_ops[i] = loom_scf_region_branch_tail_op_from_yielded_value(
        rewriter->module, block, terminator, loom_op_operands(terminator)[0]);
    if (!tail_ops[i]) return iree_ok_status();
  }

  loom_availability_analysis_t availability;
  IREE_RETURN_IF_ERROR(loom_availability_analysis_initialize(
      rewriter->module, rewriter->arena, &availability));
  loom_op_t* reference_tail = tail_ops[0];
  for (uint8_t i = 0; i < op->region_count; ++i) {
    bool tail_ops_match = false;
    IREE_RETURN_IF_ERROR(loom_scf_region_branch_tail_ops_match(
        rewriter, op, reference_tail, tail_ops[i], &availability,
        &tail_ops_match));
    if (!tail_ops_match) return iree_ok_status();
  }

  loom_type_t* operand_types = NULL;
  const loom_value_id_t* reference_tail_operands =
      loom_op_const_operands(reference_tail);
  if (reference_tail->operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, reference_tail->operand_count, sizeof(*operand_types),
        (void**)&operand_types));
    for (uint16_t i = 0; i < reference_tail->operand_count; ++i) {
      operand_types[i] =
          loom_module_value_type(rewriter->module, reference_tail_operands[i]);
    }
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_region_branch_build_empty_clone(
      rewriter, op, operand_types, reference_tail->operand_count,
      &new_branch_op));
  loom_region_branch_t new_branch =
      loom_region_branch_cast(rewriter->module, new_branch_op);

  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_op_t* new_terminator = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_region_branch_build_operand_yield(
        rewriter, new_branch, i, loom_op_const_operands(tail_ops[i]),
        reference_tail->operand_count, op->location, &new_terminator));
    if (!new_terminator) return iree_ok_status();
    loom_region_t* old_region =
        loom_region_branch_region(rewriter->module, branch, i);
    IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
        rewriter, old_region, tail_ops[i], new_terminator));
  }

  loom_type_t old_result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  loom_op_t* cloned_tail = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_region_branch_clone_tail_after_branch(
      rewriter, op, reference_tail, new_branch_op, old_result_type,
      &cloned_tail));

  loom_value_id_t replacement = loom_op_results(cloned_tail)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

//===----------------------------------------------------------------------===//
// scf.switch
//===----------------------------------------------------------------------===//

static bool loom_scf_switch_selected_region(loom_rewriter_t* rewriter,
                                            loom_op_t* op,
                                            loom_region_t** out_region,
                                            uint8_t* out_region_index) {
  *out_region = NULL;
  *out_region_index = UINT8_MAX;
  int64_t selector = 0;
  if (!loom_scf_value_facts_are_exact_i64(
          rewriter, loom_scf_switch_selector(op), &selector)) {
    return false;
  }

  loom_attribute_t case_keys = loom_scf_switch_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY ||
      (case_keys.count > 0 && !case_keys.i64_array)) {
    return false;
  }
  loom_region_slice_t case_regions = loom_scf_switch_case_regions(op);
  if (case_regions.count != case_keys.count) return false;

  for (uint16_t i = 0; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] == selector) {
      *out_region = case_regions.regions[i];
      *out_region_index = (uint8_t)(i + 1);
      return true;
    }
  }
  *out_region = loom_scf_switch_default_region(op);
  *out_region_index = 0;
  return true;
}

static bool loom_scf_switch_yielded_value_types_match_results(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_slice_t yielded_values) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        results[i] >= module->values.count) {
      return false;
    }
    loom_value_id_t yielded_value = yielded_values.values[i];
    if (yielded_value == LOOM_VALUE_ID_INVALID ||
        yielded_value >= module->values.count) {
      return false;
    }
    loom_type_t result_type = loom_module_value_type(module, results[i]);
    loom_type_t yielded_type = loom_module_value_type(module, yielded_value);
    if (!loom_type_equal(result_type, yielded_type)) return false;
  }
  return true;
}

static iree_status_t loom_scf_switch_selectify_yield_only(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (op->result_count == 0 || op->tied_result_count != 0) {
    return iree_ok_status();
  }
  loom_region_branch_t branch = loom_region_branch_cast(rewriter->module, op);
  if (!loom_region_branch_isa(branch)) return iree_ok_status();

  loom_attribute_t case_keys = loom_scf_switch_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY ||
      (case_keys.count > 0 && !case_keys.i64_array)) {
    return iree_ok_status();
  }
  loom_region_slice_t case_regions = loom_scf_switch_case_regions(op);
  if (case_regions.count != case_keys.count) return iree_ok_status();

  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  iree_host_size_t value_count = 0;
  if (!iree_host_size_checked_mul(row_count, op->result_count, &value_count)) {
    return iree_ok_status();
  }
  if (value_count > UINT16_MAX - 1) return iree_ok_status();

  loom_value_slice_t yielded_values = {0};
  for (uint16_t row = 0; row < case_regions.count; ++row) {
    if (!loom_region_branch_region_yield_only_operands(
            rewriter->module, branch, (uint8_t)(row + 1), op->result_count,
            &yielded_values) ||
        !loom_scf_switch_yielded_value_types_match_results(rewriter->module, op,
                                                           yielded_values)) {
      return iree_ok_status();
    }
  }

  if (!loom_region_branch_region_yield_only_operands(
          rewriter->module, branch, /*region_index=*/0, op->result_count,
          &yielded_values) ||
      !loom_scf_switch_yielded_value_types_match_results(rewriter->module, op,
                                                         yielded_values)) {
    return iree_ok_status();
  }

  loom_value_id_t* lookup_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, value_count,
                                                 sizeof(*lookup_values),
                                                 (void**)&lookup_values));
  for (uint16_t row = 0; row < case_regions.count; ++row) {
    if (!loom_region_branch_region_yield_only_operands(
            rewriter->module, branch, (uint8_t)(row + 1), op->result_count,
            &yielded_values)) {
      return iree_ok_status();
    }
    memcpy(lookup_values + (iree_host_size_t)row * op->result_count,
           yielded_values.values,
           (iree_host_size_t)op->result_count * sizeof(*lookup_values));
  }
  if (!loom_region_branch_region_yield_only_operands(
          rewriter->module, branch, /*region_index=*/0, op->result_count,
          &yielded_values)) {
    return iree_ok_status();
  }
  memcpy(lookup_values + (iree_host_size_t)case_keys.count * op->result_count,
         yielded_values.values,
         (iree_host_size_t)op->result_count * sizeof(*lookup_values));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*result_types), (void**)&result_types));
  const loom_value_id_t* old_results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (old_results[i] == LOOM_VALUE_ID_INVALID ||
        old_results[i] >= rewriter->module->values.count) {
      return iree_ok_status();
    }
    result_types[i] = loom_module_value_type(rewriter->module, old_results[i]);
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* lookup_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_lookup_build(
      &rewriter->builder, loom_scf_switch_selector(op), case_keys.i64_array,
      case_keys.count, lookup_values, value_count, result_types,
      op->result_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
      op->location, &lookup_op));

  loom_value_slice_t replacements = loom_scf_lookup_results(lookup_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements.values, replacements.count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements.values, replacements.count);
}

iree_status_t loom_scf_switch_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_region_t* selected_region = NULL;
  uint8_t selected_region_index = UINT8_MAX;
  if (!loom_scf_switch_selected_region(rewriter, op, &selected_region,
                                       &selected_region_index)) {
    IREE_RETURN_IF_ERROR(
        loom_scf_region_branch_factor_common_tail(op, rewriter));
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      return iree_ok_status();
    }
    return loom_scf_switch_selectify_yield_only(op, rewriter);
  }

  loom_region_branch_t branch = loom_region_branch_cast(rewriter->module, op);
  loom_op_t* yield = loom_region_branch_region_terminator(
      rewriter->module, branch, selected_region_index);
  if (!yield) return iree_ok_status();

  loom_value_slice_t yielded_values = {
      .values = loom_op_operands(yield),
      .count = yield->operand_count,
  };
  if (yielded_values.count != op->result_count) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      rewriter, selected_region, yield, op));

  return loom_scf_replace_results_and_erase(op, rewriter, yielded_values.values,
                                            yielded_values.count);
}

//===----------------------------------------------------------------------===//
// scf.if
//===----------------------------------------------------------------------===//

static bool loom_scf_if_regions_are_discardable(const loom_module_t* module,
                                                loom_op_t* op) {
  // Read-only branch work with no yielded observer is dead. Writes,
  // convergence, and hints are retained because they affect memory,
  // participant-set semantics, or requested code-generation shape.
  if (loom_op_regions_have_write_effects(op)) return false;
  if (loom_op_regions_have_convergent_effects(op)) return false;
  return !loom_op_regions_have_hints(module, op);
}

static iree_status_t loom_scf_if_fold_exact_boolean_yields(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_folded) {
  *out_folded = false;
  if (op->result_count != 1 || op->tied_result_count != 0) {
    return iree_ok_status();
  }
  if (!loom_scf_if_regions_are_discardable(rewriter->module, op)) {
    return iree_ok_status();
  }

  loom_value_id_t result = loom_op_const_results(op)[0];
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);
  if (!loom_type_is_scalar(result_type) ||
      loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_I1) {
    return iree_ok_status();
  }

  loom_op_t* then_yield =
      loom_scf_region_terminator(loom_scf_if_then_region(op));
  loom_region_t* else_region = loom_scf_if_else_region(op);
  if (!else_region) return iree_ok_status();
  loom_op_t* else_yield = loom_scf_region_terminator(else_region);
  if (!then_yield || !else_yield) return iree_ok_status();

  loom_value_slice_t then_values = loom_scf_yield_values(then_yield);
  loom_value_slice_t else_values = loom_scf_yield_values(else_yield);
  if (then_values.count != 1 || else_values.count != 1) {
    return iree_ok_status();
  }

  bool then_value = false;
  bool else_value = false;
  if (!loom_value_facts_as_exact_bool(
          loom_rewriter_value_facts(rewriter, then_values.values[0]),
          &then_value) ||
      !loom_value_facts_as_exact_bool(
          loom_rewriter_value_facts(rewriter, else_values.values[0]),
          &else_value)) {
    return iree_ok_status();
  }

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (then_value && !else_value) {
    replacement = loom_scf_if_condition(op);
  } else if (then_value == else_value) {
    loom_builder_set_before(&rewriter->builder, op);
    loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
    IREE_RETURN_IF_ERROR(loom_rewriter_build_constant(
        rewriter, loom_value_facts_exact_i64(then_value ? 1 : 0), result_type,
        op->location, &replacement));
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, &replacement, 1, value_checkpoint));
  } else {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_folded = true;
  return iree_ok_status();
}

static bool loom_scf_value_has_no_uses(const loom_module_t* module,
                                       loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  return loom_value_has_no_uses(loom_module_value(module, value_id)) &&
         !loom_module_value_has_type_uses(module, value_id);
}

static iree_status_t loom_scf_if_erase_if_effect_free_resultless(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_erased) {
  *out_erased = false;
  if (op->result_count != 0) return iree_ok_status();
  if (!loom_scf_if_regions_are_discardable(rewriter->module, op)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_erased = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_if_compact_results(loom_op_t* op,
                                                 loom_rewriter_t* rewriter) {
  if (op->result_count == 0 || op->tied_result_count != 0) {
    return iree_ok_status();
  }

  loom_op_t* then_yield =
      loom_scf_region_terminator(loom_scf_if_then_region(op));
  loom_region_t* else_region = loom_scf_if_else_region(op);
  if (!else_region) return iree_ok_status();
  loom_op_t* else_yield = loom_scf_region_terminator(else_region);
  if (!then_yield || !else_yield) return iree_ok_status();

  loom_value_slice_t then_values = loom_scf_yield_values(then_yield);
  loom_value_slice_t else_values = loom_scf_yield_values(else_yield);
  if (then_values.count != op->result_count ||
      else_values.count != op->result_count) {
    return iree_ok_status();
  }

  const loom_value_id_t* old_results = loom_op_const_results(op);
  bool* forwarded_results = NULL;
  bool* dropped_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*forwarded_results),
      (void**)&forwarded_results));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*dropped_results),
      (void**)&dropped_results));
  uint16_t dropped_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    forwarded_results[i] = then_values.values[i] == else_values.values[i];
    dropped_results[i] =
        forwarded_results[i] ||
        loom_scf_value_has_no_uses(rewriter->module, old_results[i]);
    if (dropped_results[i]) ++dropped_count;
  }
  if (dropped_count == 0) return iree_ok_status();

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));

  if (dropped_count == op->result_count &&
      loom_scf_if_regions_are_discardable(rewriter->module, op)) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      replacements[i] =
          forwarded_results[i] ? then_values.values[i] : old_results[i];
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, op, replacements, op->result_count));
    return iree_ok_status();
  }

  uint16_t kept_count = (uint16_t)(op->result_count - dropped_count);
  loom_type_t* kept_result_types = NULL;
  loom_value_id_t* kept_then_values = NULL;
  loom_value_id_t* kept_else_values = NULL;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_result_types),
                                                   (void**)&kept_result_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_then_values),
                                                   (void**)&kept_then_values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_else_values),
                                                   (void**)&kept_else_values));
  }

  uint16_t kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (dropped_results[i]) continue;
    kept_result_types[kept_ordinal] =
        loom_module_value_type(rewriter->module, old_results[i]);
    kept_then_values[kept_ordinal] = then_values.values[i];
    kept_else_values[kept_ordinal] = else_values.values[i];
    ++kept_ordinal;
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_if = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &rewriter->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      loom_scf_if_condition(op), kept_result_types, kept_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, op->location, &new_if));

  loom_region_t* new_then_region = loom_scf_if_then_region(new_if);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&rewriter->builder, new_if, new_then_region);
  loom_op_t* new_then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder,
                                            kept_then_values, kept_count,
                                            op->location, &new_then_yield));
  loom_builder_restore(&rewriter->builder, saved_ip);

  loom_region_t* new_else_region = loom_scf_if_else_region(new_if);
  saved_ip =
      loom_builder_enter_region(&rewriter->builder, new_if, new_else_region);
  loom_op_t* new_else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder,
                                            kept_else_values, kept_count,
                                            op->location, &new_else_yield));
  loom_builder_restore(&rewriter->builder, saved_ip);

  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      rewriter, loom_scf_if_then_region(op), then_yield, new_then_yield));
  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      rewriter, else_region, else_yield, new_else_yield));

  kept_ordinal = 0;
  loom_value_slice_t new_results = loom_scf_if_results(new_if);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (!dropped_results[i]) {
      replacements[i] = new_results.values[kept_ordinal++];
    } else if (forwarded_results[i]) {
      replacements[i] = then_values.values[i];
    } else {
      replacements[i] = old_results[i];
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, op->result_count));
  return iree_ok_status();
}

static iree_status_t loom_scf_if_build_select(loom_op_t* op, loom_type_t type,
                                              loom_value_id_t true_value,
                                              loom_value_id_t false_value,
                                              loom_rewriter_t* rewriter,
                                              loom_value_id_t* out_result) {
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      &rewriter->builder, loom_scf_if_condition(op), true_value, false_value,
      type, op->location, &select_op));
  *out_result = loom_scf_select_result(select_op);
  return iree_ok_status();
}

static iree_status_t loom_scf_if_selectify_yield_only(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (op->result_count == 0 || op->tied_result_count != 0) {
    return iree_ok_status();
  }

  loom_region_t* then_region = loom_scf_if_then_region(op);
  loom_region_t* else_region = loom_scf_if_else_region(op);
  if (!else_region) return iree_ok_status();
  loom_op_t* then_yield = loom_scf_region_terminator(then_region);
  loom_op_t* else_yield = loom_scf_region_terminator(else_region);
  if (!then_yield || !else_yield) return iree_ok_status();

  loom_block_t* then_block = loom_region_entry_block(then_region);
  loom_block_t* else_block = loom_region_entry_block(else_region);
  if (then_block->first_op != then_yield ||
      else_block->first_op != else_yield) {
    return iree_ok_status();
  }

  loom_value_slice_t then_values = loom_scf_yield_values(then_yield);
  loom_value_slice_t else_values = loom_scf_yield_values(else_yield);
  if (then_values.count != op->result_count ||
      else_values.count != op->result_count) {
    return iree_ok_status();
  }

  const loom_value_id_t* old_results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, old_results[i]);
    if (!loom_type_equal(
            result_type,
            loom_module_value_type(rewriter->module, then_values.values[i])) ||
        !loom_type_equal(
            result_type,
            loom_module_value_type(rewriter->module, else_values.values[i]))) {
      return iree_ok_status();
    }
  }

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, old_results[i]);
    IREE_RETURN_IF_ERROR(loom_scf_if_build_select(
        op, result_type, then_values.values[i], else_values.values[i], rewriter,
        &replacements[i]));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                  op->result_count);
}

iree_status_t loom_scf_if_canonicalize(loom_op_t* op,
                                       loom_rewriter_t* rewriter) {
  bool condition = false;
  if (!loom_value_facts_as_exact_bool(
          loom_rewriter_value_facts(rewriter, loom_scf_if_condition(op)),
          &condition)) {
    bool erased = false;
    IREE_RETURN_IF_ERROR(
        loom_scf_if_erase_if_effect_free_resultless(op, rewriter, &erased));
    if (erased) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_scf_if_compact_results(op, rewriter));
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      return iree_ok_status();
    }
    if (loom_scf_if_else_region(op)) {
      IREE_RETURN_IF_ERROR(
          loom_scf_region_branch_factor_common_tail(op, rewriter));
      if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
        return iree_ok_status();
      }
    }
    bool exact_boolean_folded = false;
    IREE_RETURN_IF_ERROR(loom_scf_if_fold_exact_boolean_yields(
        op, rewriter, &exact_boolean_folded));
    if (exact_boolean_folded) return iree_ok_status();
    return loom_scf_if_selectify_yield_only(op, rewriter);
  }

  loom_region_t* selected_region =
      condition ? loom_scf_if_then_region(op) : loom_scf_if_else_region(op);
  if (!selected_region) {
    if (op->result_count == 0) {
      return loom_rewriter_erase(rewriter, op);
    }
    return iree_ok_status();
  }
  loom_op_t* yield = loom_scf_region_terminator(selected_region);
  if (!yield) return iree_ok_status();

  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (yielded_values.count != op->result_count) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      rewriter, selected_region, yield, op));

  return loom_scf_replace_results_and_erase(op, rewriter, yielded_values.values,
                                            yielded_values.count);
}

//===----------------------------------------------------------------------===//
// scf.for
//===----------------------------------------------------------------------===//

static bool loom_scf_for_has_zero_trip_count(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
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
  if (!loom_value_facts_is_positive(step)) return false;
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

  if (lower_bound.range_hi >= upper_bound.range_lo) return false;
  int64_t next_iv_lower_bound = 0;
  if (!loom_checked_add_i64(lower_bound.range_lo, step.range_lo,
                            &next_iv_lower_bound)) {
    return false;
  }
  return next_iv_lower_bound >= upper_bound.range_hi;
}

static bool loom_scf_for_yields_loop_carried_args(loom_op_t* op) {
  loom_region_t* body = loom_scf_for_body(op);
  loom_op_t* yield = loom_scf_region_terminator(body);
  if (!yield) return false;

  loom_block_t* block = loom_region_entry_block(body);
  if (block->first_op != yield) return false;
  if (op->result_count == 0) return true;

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return false;
  }
  if (block->arg_count < 1 + op->result_count) return false;
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
  if (!yield) return iree_ok_status();

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return iree_ok_status();
  }

  loom_block_t* block = loom_region_entry_block(body);
  if (block->arg_count < 1 + iter_args.count) return iree_ok_status();

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
    loom_rewriter_t* rewriter, loom_tied_result_t** out_tied_results,
    uint16_t* out_tied_result_count, bool* out_supported) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  *out_supported = true;
  if (op->tied_result_count == 0) return iree_ok_status();

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));

  const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
  uint16_t tied_result_count = 0;
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    loom_tied_result_t tied_result = old_tied_results[i];
    if (tied_result.result_index >= op->result_count) {
      *out_supported = false;
      return iree_ok_status();
    }
    if (tied_result.operand_index >= op->operand_count) {
      *out_supported = false;
      return iree_ok_status();
    }
    uint16_t new_result_index = result_map[tied_result.result_index];
    if (new_result_index == UINT16_MAX) continue;

    uint16_t new_operand_index = tied_result.operand_index;
    if (new_operand_index >= old_iter_arg_offset) {
      uint16_t old_iter_arg_index =
          (uint16_t)(new_operand_index - old_iter_arg_offset);
      if (old_iter_arg_index >= op->result_count) {
        *out_supported = false;
        return iree_ok_status();
      }
      uint16_t new_iter_arg_index = iter_arg_map[old_iter_arg_index];
      if (new_iter_arg_index == UINT16_MAX) {
        *out_supported = false;
        return iree_ok_status();
      }
      new_operand_index = (uint16_t)(new_iter_arg_offset + new_iter_arg_index);
    }

    tied_results[tied_result_count++] = (loom_tied_result_t){
        .result_index = new_result_index,
        .operand_index = new_operand_index,
        .has_type_change = tied_result.has_type_change,
    };
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

// Compacts loop-carried slots whose yield operand is exactly the corresponding
// iter_arg block argument. The loop is rebuilt because scf.for operands, body
// block arguments, results, and tied-result metadata all share the carried slot
// numbering.
static iree_status_t loom_scf_for_forward_loop_carried_results(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (!loom_scf_for_step_is_positive(op, rewriter) || op->result_count == 0) {
    return iree_ok_status();
  }

  loom_region_t* body = loom_scf_for_body(op);
  loom_op_t* yield = loom_scf_region_terminator(body);
  if (!yield) return iree_ok_status();

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return iree_ok_status();
  }

  loom_scf_for_build_flags_t build_flags = 0;
  loom_value_id_t unroll_factor = LOOM_VALUE_ID_INVALID;
  bool has_unroll_factor = loom_scf_for_unroll_factor_is_present(op);
  if (has_unroll_factor) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    unroll_factor = loom_scf_for_unroll_factor(op);
  }
  loom_scf_for_unroll_policy_t unroll_policy = 0;
  if (!loom_attr_is_absent(
          loom_op_attrs(op)[loom_scf_for_unroll_policy_ATTR_INDEX])) {
    build_flags |= LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    unroll_policy = loom_scf_for_unroll_policy(op);
  }
  uint16_t old_iter_arg_offset =
      (uint16_t)(iter_args.values - loom_op_const_operands(op));
  uint16_t new_iter_arg_offset = 3;

  loom_block_t* old_block = loom_region_entry_block(body);
  if (old_block->arg_count < 1 + op->result_count) return iree_ok_status();

  bool* forwarded_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*forwarded_results),
      (void**)&forwarded_results));
  uint16_t forwarded_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t carried_arg =
        loom_block_arg_id(old_block, (uint16_t)(1 + i));
    forwarded_results[i] = yielded_values.values[i] == carried_arg;
    if (forwarded_results[i]) ++forwarded_count;
  }
  if (forwarded_count == 0) return iree_ok_status();

  uint16_t kept_count = (uint16_t)(op->result_count - forwarded_count);
  uint16_t* result_map = NULL;
  uint16_t* iter_arg_map = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*result_map), (void**)&result_map));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*iter_arg_map), (void**)&iter_arg_map));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    result_map[i] = UINT16_MAX;
    iter_arg_map[i] = UINT16_MAX;
  }

  loom_value_id_t* kept_iter_args = NULL;
  loom_type_t* kept_result_types = NULL;
  loom_value_id_t* kept_yielded_values = NULL;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_iter_args),
                                                   (void**)&kept_iter_args));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_result_types),
                                                   (void**)&kept_result_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, kept_count, sizeof(*kept_yielded_values),
        (void**)&kept_yielded_values));
  }

  const loom_value_id_t* old_results = loom_op_const_results(op);
  uint16_t kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (forwarded_results[i]) continue;
    result_map[i] = kept_ordinal;
    iter_arg_map[i] = kept_ordinal;
    kept_iter_args[kept_ordinal] = iter_args.values[i];
    kept_result_types[kept_ordinal] =
        loom_module_value_type(rewriter->module, old_results[i]);
    kept_yielded_values[kept_ordinal] = yielded_values.values[i];
    ++kept_ordinal;
  }

  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  bool tied_results_supported = false;
  IREE_RETURN_IF_ERROR(loom_scf_for_adjust_tied_results(
      op, result_map, iter_arg_map, old_iter_arg_offset, new_iter_arg_offset,
      rewriter, &tied_results, &tied_result_count, &tied_results_supported));
  if (!tied_results_supported) return iree_ok_status();

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &rewriter->builder, build_flags, loom_scf_for_lower_bound(op),
      loom_scf_for_upper_bound(op), loom_scf_for_step(op), kept_iter_args,
      kept_count, kept_result_types, kept_count, tied_results,
      tied_result_count, unroll_factor, unroll_policy, op->location,
      &new_loop));

  loom_region_t* new_body = loom_scf_for_body(new_loop);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&rewriter->builder, new_loop, new_body);
  loom_op_t* new_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder,
                                            kept_yielded_values, kept_count,
                                            op->location, &new_yield));
  loom_builder_restore(&rewriter->builder, saved_ip);

  loom_block_t* new_block = loom_region_entry_block(new_body);
  IREE_RETURN_IF_ERROR(loom_scf_preserve_value_name(
      rewriter->module, loom_block_arg_id(old_block, 0),
      loom_block_arg_id(new_block, 0)));
  kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (forwarded_results[i]) continue;
    IREE_RETURN_IF_ERROR(loom_scf_preserve_value_name(
        rewriter->module, loom_block_arg_id(old_block, (uint16_t)(1 + i)),
        loom_block_arg_id(new_block, (uint16_t)(1 + kept_ordinal++))));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      rewriter, loom_block_arg_id(old_block, 0),
      loom_block_arg_id(new_block, 0)));
  kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t old_arg = loom_block_arg_id(old_block, (uint16_t)(1 + i));
    loom_value_id_t replacement =
        forwarded_results[i]
            ? iter_args.values[i]
            : loom_block_arg_id(new_block, (uint16_t)(1 + kept_ordinal++));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_with(rewriter, old_arg, replacement));
  }

  loom_op_t* child_op = old_block->first_op;
  while (child_op && child_op != yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, child_op, new_yield));
    child_op = next_child_op;
  }

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  kept_ordinal = 0;
  loom_value_slice_t new_results = loom_scf_for_results(new_loop);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    replacements[i] = forwarded_results[i] ? iter_args.values[i]
                                           : new_results.values[kept_ordinal++];
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                  op->result_count);
}

iree_status_t loom_scf_for_canonicalize(loom_op_t* op,
                                        loom_rewriter_t* rewriter) {
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  if (loom_scf_for_has_zero_trip_count(op, rewriter)) {
    return loom_scf_replace_results_and_erase(op, rewriter, iter_args.values,
                                              iter_args.count);
  }
  IREE_RETURN_IF_ERROR(loom_scf_for_inline_single_trip(op, rewriter));
  if (op->flags & LOOM_OP_FLAG_DEAD) return iree_ok_status();
  if (loom_scf_for_step_is_positive(op, rewriter) &&
      loom_scf_for_yields_loop_carried_args(op)) {
    return loom_scf_replace_results_and_erase(op, rewriter, iter_args.values,
                                              iter_args.count);
  }
  return loom_scf_for_forward_loop_carried_results(op, rewriter);
}
