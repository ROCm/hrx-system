// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/lookup.h"

#include <string.h>

#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

// Three equality decisions amortize each range branch over four dense rows.
enum { LOOM_SCF_LOOKUP_LEAF_ROWS = 4 };

static bool loom_scf_lookup_can_partition(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL ||
      loom_scf_lookup_case_keys(op).count < LOOM_SCF_LOOKUP_LEAF_ROWS) {
    return false;
  }
  // Only executable callable and structured control regions admit this
  // choice. Other single-block regions can still predicate the whole table.
  const loom_region_t* region = op->parent_block->parent_region;
  if (region != loom_func_like_body(context->function) &&
      loom_op_dialect_id(region->parent_op->kind) != LOOM_DIALECT_SCF) {
    return false;
  }
  const loom_target_bundle_t* target =
      loom_target_legalization_context_bundle(context);
  return target->snapshot->subgroup_size == 1 ||
         loom_value_facts_is_subgroup_uniform(loom_value_fact_table_lookup(
             context->fact_table, loom_scf_lookup_selector(op)));
}

typedef struct loom_scf_lookup_projection_t {
  // Builder positioned in the current selection arm.
  loom_builder_t* builder;
  // Verified table whose last row is the default.
  loom_op_t* lookup;
  // One type per result column, shared by all joins.
  loom_type_t* types;
  // Scratch row, consumed by each yield before the next arm overwrites it.
  loom_value_id_t* row;
} loom_scf_lookup_projection_t;

static iree_status_t loom_scf_lookup_compare(
    const loom_scf_lookup_projection_t* projection, uint16_t row,
    loom_index_cmp_predicate_t predicate, loom_value_id_t* out_condition) {
  loom_op_t* lookup = projection->lookup;
  loom_value_id_t selector = loom_scf_lookup_selector(lookup);
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      projection->builder,
      loom_attr_i64(loom_scf_lookup_case_keys(lookup).i64_array[row]),
      loom_module_value_type(projection->builder->module, selector),
      lookup->location, &constant));
  loom_op_t* compare = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      projection->builder, predicate, selector,
      loom_index_constant_result(constant), lookup->location, &compare));
  *out_condition = loom_index_cmp_result(compare);
  return iree_ok_status();
}

static iree_status_t loom_scf_lookup_project_leaf(
    loom_scf_lookup_projection_t* projection, uint16_t begin, uint16_t end) {
  loom_op_t* lookup = projection->lookup;
  const loom_attribute_t keys = loom_scf_lookup_case_keys(lookup);
  const uint16_t columns = lookup->result_count;
  const loom_value_id_t* cells = loom_scf_lookup_values(lookup).values;
  uint16_t fallback = keys.count;
  if (end > keys.count) {
    end = keys.count;
  } else if ((begin || keys.i64_array[0] == 0 ||
              keys.i64_array[0] == INT64_MIN) &&
             (uint64_t)keys.i64_array[end] - (uint64_t)keys.i64_array[begin] ==
                 (uint64_t)(end - begin)) {
    // Both bounds come from the search path. A contiguous interval covering
    // that entire domain can use its final row for the remaining selector.
    fallback = --end;
  }
  memcpy(projection->row, cells + (size_t)fallback * columns,
         columns * sizeof(*projection->row));
  for (uint16_t i = end; i > begin; --i) {
    const loom_value_id_t* row = cells + (size_t)(i - 1) * columns;
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    for (uint16_t column = 0; column < columns; ++column) {
      if (row[column] == projection->row[column]) {
        continue;
      }
      if (condition == LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_scf_lookup_compare(
            projection, i - 1, LOOM_INDEX_CMP_PREDICATE_EQ, &condition));
      }
      loom_op_t* select = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(
          projection->builder, condition, row[column], projection->row[column],
          projection->types[column], lookup->location, &select));
      projection->row[column] = loom_scf_select_result(select);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_lookup_project_range(
    loom_scf_lookup_projection_t* projection, uint16_t begin, uint16_t end) {
  if (end - begin <= LOOM_SCF_LOOKUP_LEAF_ROWS) {
    return loom_scf_lookup_project_leaf(projection, begin, end);
  }
  loom_op_t* lookup = projection->lookup;
  const uint16_t middle = begin + (end - begin) / 2;
  // For nonnegative keys use unsigned order so every negative selector
  // naturally reaches the open-ended default arm.
  const bool nonnegative_keys =
      loom_scf_lookup_case_keys(lookup).i64_array[0] >= 0;
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_scf_lookup_compare(projection, middle,
                              nonnegative_keys ? LOOM_INDEX_CMP_PREDICATE_ULT
                                               : LOOM_INDEX_CMP_PREDICATE_SLT,
                              &condition));
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      projection->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, condition,
      projection->types, lookup->result_count, NULL, 0, lookup->location,
      &branch));
  for (uint16_t arm = 0; arm < 2; ++arm) {
    loom_builder_ip_t parent =
        loom_builder_enter_region(projection->builder, branch,
                                  arm ? loom_scf_if_else_region(branch)
                                      : loom_scf_if_then_region(branch));
    IREE_RETURN_IF_ERROR(loom_scf_lookup_project_range(
        projection, arm ? middle : begin, arm ? end : middle));
    loom_op_t* yield = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scf_yield_build(projection->builder, projection->row,
                             lookup->result_count, lookup->location, &yield));
    loom_builder_restore(projection->builder, parent);
  }
  memcpy(projection->row, loom_op_const_results(branch),
         lookup->result_count * sizeof(*projection->row));
  return iree_ok_status();
}

static iree_status_t loom_scf_lookup_project(loom_rewriter_t* rewriter,
                                             iree_arena_allocator_t* arena,
                                             loom_op_t* lookup,
                                             bool partition_rows) {
  loom_scf_lookup_projection_t projection = {.builder = &rewriter->builder,
                                             .lookup = lookup};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, lookup->result_count,
                                                 sizeof(*projection.types),
                                                 (void**)&projection.types));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, lookup->result_count,
                                                 sizeof(*projection.row),
                                                 (void**)&projection.row));
  for (uint16_t column = 0; column < lookup->result_count; ++column) {
    projection.types[column] = loom_module_value_type(
        rewriter->module, loom_op_const_results(lookup)[column]);
  }

  loom_builder_set_before(&rewriter->builder, lookup);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const uint16_t row_count = loom_scf_lookup_case_keys(lookup).count + 1;
  if (partition_rows) {
    IREE_RETURN_IF_ERROR(
        loom_scf_lookup_project_range(&projection, 0, row_count));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_scf_lookup_project_leaf(&projection, 0, row_count));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, lookup, projection.row, lookup->result_count, checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(
      rewriter, lookup, projection.row, lookup->result_count);
}

iree_status_t loom_scf_lookup_project_selects(loom_rewriter_t* rewriter,
                                              iree_arena_allocator_t* arena,
                                              loom_op_t* lookup) {
  return loom_scf_lookup_project(rewriter, arena, lookup, false);
}

static iree_status_t loom_scf_lookup_partition(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* lookup,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  IREE_RETURN_IF_ERROR(
      loom_scf_lookup_project(context->rewriter, context->arena, lookup, true));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kLookupRules[] = {{
    .root_kind = LOOM_OP_SCF_LOOKUP,
    .match = loom_scf_lookup_can_partition,
    .legalize = loom_scf_lookup_partition,
}};

const loom_target_legalizer_provider_t* loom_scf_lookup_legalizer_provider(
    void) {
  static const loom_target_legalizer_provider_t provider = {
      .name = IREE_SVL("scf-lookup"),
      .scope = LOOM_LOW_DESCRIPTOR_SCOPE_SHARED,
      .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
      .rules = kLookupRules,
      .rule_count = IREE_ARRAYSIZE(kLookupRules),
  };
  return &provider;
}
