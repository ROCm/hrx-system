// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower_report.h"

#include <stdint.h>
#include <string.h>

#include "loom/codegen/low/lower/lower_context.h"
#include "loom/codegen/low/lower/source_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/cfg_loop.h"

enum {
  // Default allocation block size for source-to-Low selection report rows.
  LOOM_LOW_LOWER_REPORT_ROW_BLOCK_BYTE_LENGTH = 4096,
  // Initial allocation size for contiguous source-memory report rows.
  LOOM_LOW_LOWER_MEMORY_REPORT_ROW_ARRAY_BYTE_LENGTH = 4096,
};

typedef struct loom_low_lower_memory_expression_term_t {
  // Source SSA value multiplied into this symbolic byte expression.
  loom_value_id_t value_id;
  // Signed byte coefficient applied to |value_id|.
  int64_t coefficient;
} loom_low_lower_memory_expression_term_t;

typedef struct loom_low_lower_memory_expression_key_t {
  // Static byte constant added to all dynamic terms.
  int64_t constant;
  // Number of populated entries in |terms|.
  uint8_t term_count;
  // Sorted symbolic byte terms.
  loom_low_lower_memory_expression_term_t
      terms[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
} loom_low_lower_memory_expression_key_t;

struct loom_low_lower_memory_expression_entry_t {
  // Comparable symbolic expression key interned for report-only accounting.
  loom_low_lower_memory_expression_key_t key;
};

static iree_string_view_t loom_low_lower_report_descriptor_string(
    const loom_low_lower_context_t* context,
    const loom_low_descriptor_t* descriptor,
    loom_bstring_table_offset_t string_offset) {
  if (descriptor == NULL || string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return iree_string_view_empty();
  }
  return loom_low_descriptor_set_string(context->descriptor_set, string_offset);
}

static void loom_low_lower_report_populate_descriptor(
    const loom_low_lower_context_t* context,
    const loom_low_descriptor_t* descriptor, loom_low_lower_report_row_t* row) {
  if (descriptor == NULL) {
    return;
  }
  row->descriptor_key = loom_low_lower_report_descriptor_string(
      context, descriptor, descriptor->key_string_offset);
  row->descriptor_semantic_tag = loom_low_lower_report_descriptor_string(
      context, descriptor, descriptor->semantic_tag_string_offset);
}

static void loom_low_lower_report_row_list_deinitialize(
    iree_allocator_t allocator, loom_low_lower_report_row_list_t* list) {
  loom_low_lower_report_row_vec_t* vec = list->head;
  while (vec != NULL) {
    loom_low_lower_report_row_vec_t* next = vec->next;
    iree_allocator_free(allocator, vec);
    vec = next;
  }
  *list = (loom_low_lower_report_row_list_t){0};
}

static void loom_low_lower_memory_report_row_list_deinitialize(
    iree_allocator_t allocator, loom_low_lower_memory_report_row_list_t* list) {
  iree_allocator_free(allocator, list->rows);
  *list = (loom_low_lower_memory_report_row_list_t){0};
}

void loom_low_lower_result_deinitialize(loom_low_lower_result_t* result) {
  if (result == NULL) {
    return;
  }
  loom_low_lower_report_row_list_deinitialize(result->report_allocator,
                                              &result->report_rows);
  loom_low_lower_memory_report_row_list_deinitialize(
      result->memory_report_row_allocator, &result->memory_report_rows);
  result->report_allocator = iree_allocator_null();
  result->memory_report_row_allocator = iree_allocator_null();
}

static iree_status_t loom_low_lower_report_row_list_append(
    loom_low_lower_report_row_list_t* list, iree_allocator_t allocator,
    const loom_low_lower_report_row_t* row) {
  if (list->tail == NULL || list->tail->count == list->tail->capacity) {
    iree_host_size_t capacity = (LOOM_LOW_LOWER_REPORT_ROW_BLOCK_BYTE_LENGTH -
                                 sizeof(loom_low_lower_report_row_vec_t)) /
                                sizeof(*row);
    capacity = iree_max((iree_host_size_t)1, capacity);
    loom_low_lower_report_row_vec_t* vec = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
        allocator, sizeof(*vec), capacity, sizeof(*row), (void**)&vec));
    *vec = (loom_low_lower_report_row_vec_t){
        .capacity = capacity,
    };
    if (list->tail != NULL) {
      list->tail->next = vec;
    } else {
      list->head = vec;
    }
    list->tail = vec;
  }
  loom_low_lower_report_row_t* rows =
      loom_low_lower_report_row_vec_rows(list->tail);
  rows[list->tail->count++] = *row;
  ++list->count;
  return iree_ok_status();
}

iree_status_t loom_low_lower_report_record_selected_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan,
    uint32_t emitted_low_op_count) {
  loom_low_lower_result_t* result = context->result;
  ++result->selected_source_op_count;
  result->emitted_low_op_count += emitted_low_op_count;

  loom_low_lower_report_row_t row = {
      .function_name = loom_low_lower_context_function_name(context),
      .source_op_name = loom_op_name(context->module, selected_plan->source_op),
      .source_op_kind = selected_plan->source_op->kind,
      .selection_kind = LOOM_LOW_LOWER_REPORT_SELECTION_PLAN,
      .rule_set_index = UINT16_MAX,
      .rule_index = UINT16_MAX,
      .plan_id = selected_plan->plan.id,
      .plan_key = iree_string_view_empty(),
      .native_contraction_facts = NULL,
      .native_transition_facts = NULL,
      .descriptor_key = iree_string_view_empty(),
      .descriptor_semantic_tag = iree_string_view_empty(),
      .emitted_low_op_count = emitted_low_op_count,
      .execution_count_plus_one =
          LOOM_LOW_LOWER_REPORT_EXECUTION_COUNT_PLUS_ONE_UNKNOWN,
  };
  if (selected_plan->rule != NULL) {
    row.selection_kind = LOOM_LOW_LOWER_REPORT_SELECTION_RULE;
    row.rule_set_index = selected_plan->rule_set_index;
    row.rule_index = selected_plan->rule_index;
    row.plan_id = LOOM_LOW_LOWER_PLAN_ID_NONE;
    if (selected_plan->rule->report_key_ordinal !=
        LOOM_LOW_LOWER_RULE_REPORT_KEY_NONE) {
      const uint16_t report_key_index =
          selected_plan->rule->report_key_ordinal - 1u;
      IREE_ASSERT_LT(report_key_index,
                     selected_plan->rule_set->report_key_count);
      row.plan_key = loom_low_lower_rule_set_string(
          selected_plan->rule_set,
          selected_plan->rule_set->report_key_string_offsets[report_key_index]);
    }
    if (selected_plan->rule->emit_count != 0 &&
        selected_plan->resolved_emits != NULL) {
      loom_low_lower_report_populate_descriptor(
          context, selected_plan->resolved_emits[0].descriptor.descriptor,
          &row);
    }
  } else if (selected_plan->kind ==
             LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX) {
    const loom_low_lower_descriptor_matrix_plan_t* plan =
        (const loom_low_lower_descriptor_matrix_plan_t*)
            selected_plan->plan.target_data;
    loom_low_lower_report_populate_descriptor(
        context, plan->descriptor.descriptor, &row);
    row.native_contraction_facts = plan->native_contraction_facts;
  }
  if (selected_plan->rule == NULL &&
      context->policy->describe_plan.fn != NULL) {
    loom_low_lower_plan_report_t plan_report = {0};
    context->policy->describe_plan.fn(context->policy->describe_plan.user_data,
                                      context, selected_plan->source_op,
                                      selected_plan->plan, &plan_report);
    row.plan_key = plan_report.plan_key;
    if (plan_report.native_contraction_facts != NULL) {
      row.native_contraction_facts = plan_report.native_contraction_facts;
    }
    row.native_transition_facts = plan_report.native_transition_facts;
    row.native_transition_source_type =
        plan_report.native_transition_source_type;
    row.native_transition_destination_type =
        plan_report.native_transition_destination_type;
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_source_op_execution_count_plus_one(
      context, selected_plan->source_op, &row.execution_count_plus_one));
  return loom_low_lower_report_row_list_append(&result->report_rows,
                                               result->report_allocator, &row);
}

static iree_status_t loom_low_lower_memory_report_row_list_append(
    loom_low_lower_memory_report_row_list_t* list, iree_allocator_t allocator,
    const loom_low_lower_memory_report_row_t* row) {
  if (list->count == list->capacity) {
    iree_host_size_t minimum_capacity =
        LOOM_LOW_LOWER_MEMORY_REPORT_ROW_ARRAY_BYTE_LENGTH / sizeof(*row);
    minimum_capacity = iree_max((iree_host_size_t)1, minimum_capacity);
    iree_host_size_t capacity = list->capacity;
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(allocator, minimum_capacity,
                                                   sizeof(*row), &capacity,
                                                   (void**)&list->rows));
    list->capacity = capacity;
  }
  list->rows[list->count++] = *row;
  return iree_ok_status();
}

static bool loom_low_lower_report_multiply_u64(uint64_t lhs, uint64_t rhs,
                                               uint64_t* out_result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs) {
    return false;
  }
  *out_result = lhs * rhs;
  return true;
}

static bool loom_low_lower_memory_expression_keys_equal(
    const loom_low_lower_memory_expression_key_t* left,
    const loom_low_lower_memory_expression_key_t* right) {
  if (left->constant != right->constant ||
      left->term_count != right->term_count) {
    return false;
  }
  for (uint8_t i = 0; i < left->term_count; ++i) {
    if (left->terms[i].value_id != right->terms[i].value_id ||
        left->terms[i].coefficient != right->terms[i].coefficient) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_memory_expression_key_append_term(
    loom_low_lower_memory_expression_key_t* key, loom_value_id_t value_id,
    int64_t coefficient) {
  if (coefficient == 0) {
    return true;
  }
  uint8_t insert_index = 0;
  while (insert_index < key->term_count &&
         key->terms[insert_index].value_id < value_id) {
    ++insert_index;
  }
  if (insert_index < key->term_count &&
      key->terms[insert_index].value_id == value_id) {
    int64_t combined_coefficient = 0;
    if (!iree_checked_add_i64(key->terms[insert_index].coefficient, coefficient,
                              &combined_coefficient)) {
      return false;
    }
    if (combined_coefficient == 0) {
      memmove(&key->terms[insert_index], &key->terms[insert_index + 1],
              (key->term_count - insert_index - 1) * sizeof(key->terms[0]));
      --key->term_count;
      return true;
    }
    key->terms[insert_index].coefficient = combined_coefficient;
    return true;
  }
  if (key->term_count >= IREE_ARRAYSIZE(key->terms)) {
    return false;
  }
  memmove(&key->terms[insert_index + 1], &key->terms[insert_index],
          (key->term_count - insert_index) * sizeof(key->terms[0]));
  key->terms[insert_index] = (loom_low_lower_memory_expression_term_t){
      .value_id = value_id,
      .coefficient = coefficient,
  };
  ++key->term_count;
  return true;
}

static bool loom_low_lower_memory_expression_key_from_source_plan(
    const loom_low_source_memory_access_plan_t* source_plan,
    int64_t lane_offset, loom_low_lower_memory_expression_key_t* out_key) {
  *out_key = (loom_low_lower_memory_expression_key_t){0};
  if (source_plan->dynamic_term_count == 0) {
    return false;
  }
  if (!iree_checked_add_i64(source_plan->static_byte_offset, lane_offset,
                            &out_key->constant)) {
    return false;
  }
  for (uint8_t i = 0; i < source_plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source_plan->dynamic_terms[i];
    if (term->index == LOOM_VALUE_ID_INVALID || term->stride_value_count != 0 ||
        !loom_low_lower_memory_expression_key_append_term(out_key, term->index,
                                                          term->byte_stride)) {
      return false;
    }
  }
  return out_key->term_count != 0;
}

static iree_status_t loom_low_lower_memory_expression_ensure_capacity(
    loom_low_lower_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <=
      context->lowering.report.memory_expression_entry_capacity) {
    return iree_ok_status();
  }
  void* entries = context->lowering.report.memory_expression_entries;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      &context->function_arena,
      context->lowering.report.memory_expression_entry_capacity,
      minimum_capacity,
      sizeof(*context->lowering.report.memory_expression_entries),
      &context->lowering.report.memory_expression_entry_capacity, &entries));
  context->lowering.report.memory_expression_entries =
      (loom_low_lower_memory_expression_entry_t*)entries;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_memory_expression_intern(
    loom_low_lower_context_t* context,
    const loom_low_lower_memory_expression_key_t* key,
    loom_low_memory_expr_id_t* out_expression_id) {
  *out_expression_id = LOOM_LOW_MEMORY_EXPR_ID_NONE;
  for (iree_host_size_t i = 0;
       i < context->lowering.report.memory_expression_entry_count; ++i) {
    if (loom_low_lower_memory_expression_keys_equal(
            &context->lowering.report.memory_expression_entries[i].key, key)) {
      *out_expression_id = (loom_low_memory_expr_id_t)i;
      return iree_ok_status();
    }
  }
  if (context->lowering.report.memory_expression_entry_count >=
      (iree_host_size_t)LOOM_LOW_MEMORY_EXPR_ID_NONE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many source-memory report expressions");
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_memory_expression_ensure_capacity(
      context, context->lowering.report.memory_expression_entry_count + 1));
  const iree_host_size_t entry_index =
      context->lowering.report.memory_expression_entry_count++;
  context->lowering.report.memory_expression_entries[entry_index] =
      (loom_low_lower_memory_expression_entry_t){.key = *key};
  *out_expression_id = (loom_low_memory_expr_id_t)entry_index;
  return iree_ok_status();
}

iree_status_t loom_low_lower_memory_report_row_populate_source_interval(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_plan,
    loom_low_lower_memory_report_row_t* row) {
  loom_low_byte_interval_t byte_interval = {0};
  loom_low_memory_access_summary_t summary = {0};
  loom_low_source_memory_access_plan_make_summary(source_plan, &byte_interval,
                                                  &summary);
  if (summary.byte_interval == NULL) {
    return iree_ok_status();
  }
  row->source_interval = *summary.byte_interval;

  int64_t lane_begin_offset = 0;
  int64_t lane_end_offset = 0;
  if (!loom_low_source_memory_access_plan_lane_byte_envelope(
          source_plan, &lane_begin_offset, &lane_end_offset)) {
    return iree_ok_status();
  }
  loom_low_lower_memory_expression_key_t begin_key = {0};
  loom_low_lower_memory_expression_key_t end_key = {0};
  if (!loom_low_lower_memory_expression_key_from_source_plan(
          source_plan, lane_begin_offset, &begin_key) ||
      !loom_low_lower_memory_expression_key_from_source_plan(
          source_plan, lane_end_offset, &end_key)) {
    return iree_ok_status();
  }

  loom_low_memory_expr_id_t begin_expression_id = LOOM_LOW_MEMORY_EXPR_ID_NONE;
  loom_low_memory_expr_id_t end_expression_id = LOOM_LOW_MEMORY_EXPR_ID_NONE;
  IREE_RETURN_IF_ERROR(loom_low_lower_memory_expression_intern(
      context, &begin_key, &begin_expression_id));
  IREE_RETURN_IF_ERROR(loom_low_lower_memory_expression_intern(
      context, &end_key, &end_expression_id));
  row->source_interval.begin_expr_id = begin_expression_id;
  row->source_interval.end_expr_id = end_expression_id;
  row->source_interval.precision_flags |=
      LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_EXPR |
      LOOM_LOW_BYTE_INTERVAL_PRECISION_END_EXPR;
  return iree_ok_status();
}

static bool loom_low_lower_report_exact_trip_count(
    const loom_value_fact_table_t* fact_table, loom_loop_like_t loop,
    uint64_t* out_trip_count) {
  *out_trip_count = 0;
  if (fact_table == NULL || !loom_loop_like_isa(loop) ||
      !loom_loop_like_has_counted_range(loop)) {
    return false;
  }
  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  int64_t step = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table,
                                       loom_loop_like_lower_bound(loop)),
          &lower_bound) ||
      !loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table,
                                       loom_loop_like_upper_bound(loop)),
          &upper_bound) ||
      !loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, loom_loop_like_step(loop)),
          &step) ||
      step <= 0) {
    return false;
  }
  if (upper_bound <= lower_bound) {
    return true;
  }
  int64_t span = 0;
  if (!iree_checked_sub_i64(upper_bound, lower_bound, &span)) {
    return false;
  }
  const uint64_t unsigned_span = (uint64_t)span;
  const uint64_t unsigned_step = (uint64_t)step;
  *out_trip_count = ((unsigned_span - 1) / unsigned_step) + 1;
  return true;
}

static bool loom_low_lower_report_value_exact_i64(
    const loom_low_lower_context_t* context, loom_value_id_t value_id,
    int64_t* out_value) {
  *out_value = 0;
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  const loom_value_fact_table_t* fact_table = context->lowering.fact_table;
  if (fact_table != NULL) {
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(fact_table, value_id);
    if (loom_value_facts_as_exact_i64(facts, out_value)) {
      return true;
    }
    loom_value_facts_t element_facts = loom_value_facts_unknown();
    if (loom_value_facts_query_all_equal_element(&fact_table->context, facts,
                                                 &element_facts) &&
        loom_value_facts_as_exact_i64(element_facts, out_value)) {
      return true;
    }
  }
  if (value_id >= context->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || !loom_index_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_index_constant_value(defining_op);
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static loom_value_id_t loom_low_lower_report_identity_value(
    const loom_low_lower_context_t* context, loom_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < context->module->values.count; ++i) {
    if (value_id == LOOM_VALUE_ID_INVALID ||
        value_id >= context->module->values.count) {
      return value_id;
    }
    const loom_value_t* value = loom_module_value(context->module, value_id);
    if (loom_value_is_block_arg(value)) {
      return value_id;
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (defining_op == NULL) {
      return value_id;
    }
    const loom_trait_flags_t traits =
        loom_op_effective_traits(context->module, defining_op);
    const uint16_t result_index = loom_value_def_index(value);
    loom_value_id_t next_value_id = value_id;
    if (loom_traits_are_fact_identity(traits)) {
      if (result_index >= defining_op->operand_count) {
        return value_id;
      }
      next_value_id = loom_op_const_operands(defining_op)[result_index];
    } else if (loom_traits_are_value_alias(traits)) {
      if (result_index != 0 || defining_op->operand_count == 0) {
        return value_id;
      }
      next_value_id = loom_op_const_operands(defining_op)[0];
    } else {
      return value_id;
    }
    if (next_value_id == value_id) {
      return value_id;
    }
    value_id = next_value_id;
  }
  return value_id;
}

static bool loom_low_lower_report_block_branches_to(const loom_block_t* block,
                                                    const loom_block_t* dest,
                                                    const loom_op_t** out_br) {
  const loom_op_t* terminator = block != NULL ? block->last_op : NULL;
  if (terminator == NULL || !loom_cfg_br_isa(terminator) ||
      loom_cfg_br_dest(terminator) != dest) {
    return false;
  }
  *out_br = terminator;
  return true;
}

static bool loom_low_lower_report_branch_argument(
    const loom_op_t* branch_op, const loom_block_t* dest, uint16_t arg_index,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  if (branch_op == NULL || !loom_cfg_br_isa(branch_op) ||
      arg_index >= dest->arg_count) {
    return false;
  }
  const loom_value_slice_t args = loom_cfg_br_args(branch_op);
  if (arg_index >= args.count) {
    return false;
  }
  *out_value_id = args.values[arg_index];
  return true;
}

static bool loom_low_lower_report_add_step(
    const loom_low_lower_context_t* context, loom_value_id_t value_id,
    loom_value_id_t iv_id, int64_t* out_step) {
  if (value_id >= context->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || !loom_index_add_isa(defining_op)) {
    return false;
  }
  const loom_value_id_t lhs = loom_low_lower_report_identity_value(
      context, loom_index_add_lhs(defining_op));
  const loom_value_id_t rhs = loom_low_lower_report_identity_value(
      context, loom_index_add_rhs(defining_op));
  if (lhs == iv_id) {
    return loom_low_lower_report_value_exact_i64(context, rhs, out_step);
  }
  if (rhs == iv_id) {
    return loom_low_lower_report_value_exact_i64(context, lhs, out_step);
  }
  return false;
}

static bool loom_low_lower_report_header_upper_bound(
    const loom_low_lower_context_t* context, const loom_op_t* cond_br_op,
    loom_value_id_t iv_id, int64_t* out_upper_bound,
    bool* out_inclusive_upper_bound) {
  *out_inclusive_upper_bound = false;
  if (cond_br_op == NULL || !loom_cfg_cond_br_isa(cond_br_op)) {
    return false;
  }
  const loom_value_id_t condition = loom_cfg_cond_br_condition(cond_br_op);
  if (condition >= context->module->values.count) {
    return false;
  }
  const loom_value_t* condition_value =
      loom_module_value(context->module, condition);
  if (loom_value_is_block_arg(condition_value)) {
    return false;
  }
  const loom_op_t* compare_op = loom_value_def_op(condition_value);
  if (compare_op == NULL || !loom_index_cmp_isa(compare_op)) {
    return false;
  }
  const bool exclusive_upper =
      loom_index_cmp_predicate(compare_op) == LOOM_INDEX_CMP_PREDICATE_SLT ||
      loom_index_cmp_predicate(compare_op) == LOOM_INDEX_CMP_PREDICATE_ULT;
  const bool inclusive_upper =
      loom_index_cmp_predicate(compare_op) == LOOM_INDEX_CMP_PREDICATE_SLE ||
      loom_index_cmp_predicate(compare_op) == LOOM_INDEX_CMP_PREDICATE_ULE;
  if (!exclusive_upper && !inclusive_upper) {
    return false;
  }
  if (loom_index_cmp_lhs(compare_op) != iv_id) {
    return false;
  }
  if (!loom_low_lower_report_value_exact_i64(
          context, loom_index_cmp_rhs(compare_op), out_upper_bound)) {
    return false;
  }
  *out_inclusive_upper_bound = inclusive_upper;
  return true;
}

static bool loom_low_lower_report_compute_trip_count(int64_t lower_bound,
                                                     int64_t upper_bound,
                                                     bool inclusive_upper_bound,
                                                     int64_t step,
                                                     uint64_t* out_trip_count) {
  *out_trip_count = 0;
  if (step <= 0) {
    return false;
  }
  if (inclusive_upper_bound) {
    if (upper_bound == INT64_MAX) {
      return false;
    }
    ++upper_bound;
  }
  if (upper_bound <= lower_bound) {
    return true;
  }
  int64_t span = 0;
  if (!iree_checked_sub_i64(upper_bound, lower_bound, &span)) {
    return false;
  }
  const uint64_t unsigned_span = (uint64_t)span;
  const uint64_t unsigned_step = (uint64_t)step;
  *out_trip_count = ((unsigned_span - 1) / unsigned_step) + 1;
  return true;
}

static bool loom_low_lower_report_try_counted_cfg_loop(
    const loom_low_lower_context_t* context, const loom_cfg_graph_t* graph,
    const loom_cfg_loop_interval_t* interval, uint64_t* out_trip_count) {
  *out_trip_count = 0;
  const loom_block_t* header = graph->blocks[interval->header_index].block;
  if (header == NULL || header->arg_count == 0 || header->last_op == NULL ||
      !loom_cfg_cond_br_isa(header->last_op)) {
    return false;
  }
  const loom_value_id_t iv_id = loom_block_arg_id(header, 0);
  const loom_cfg_edge_info_t* entry_edge =
      loom_cfg_graph_edge(graph, interval->entry_edge_index);
  const loom_cfg_edge_info_t* backedge =
      loom_cfg_graph_edge(graph, interval->backedge_edge_index);
  if (entry_edge == NULL || backedge == NULL) {
    return false;
  }

  const loom_op_t* initial_branch_op = NULL;
  if (!loom_low_lower_report_block_branches_to(
          graph->blocks[entry_edge->source_block_index].block, header,
          &initial_branch_op)) {
    return false;
  }
  const loom_op_t* body_backedge_op = NULL;
  if (!loom_low_lower_report_block_branches_to(
          graph->blocks[backedge->source_block_index].block, header,
          &body_backedge_op)) {
    return false;
  }

  loom_value_id_t body_backedge_arg = LOOM_VALUE_ID_INVALID;
  if (!loom_low_lower_report_branch_argument(
          body_backedge_op, header, /*arg_index=*/0, &body_backedge_arg)) {
    return false;
  }

  loom_value_id_t initial_arg = LOOM_VALUE_ID_INVALID;
  if (!loom_low_lower_report_branch_argument(initial_branch_op, header,
                                             /*arg_index=*/0, &initial_arg)) {
    return false;
  }

  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  int64_t step = 0;
  bool inclusive_upper_bound = false;
  bool lower_ok =
      loom_low_lower_report_value_exact_i64(context, initial_arg, &lower_bound);
  bool step_ok =
      loom_low_lower_report_add_step(context, body_backedge_arg, iv_id, &step);
  bool upper_ok = loom_low_lower_report_header_upper_bound(
      context, header->last_op, iv_id, &upper_bound, &inclusive_upper_bound);
  bool trip_ok = loom_low_lower_report_compute_trip_count(
      lower_bound, upper_bound, inclusive_upper_bound, step, out_trip_count);
  return lower_ok && step_ok && upper_ok && trip_ok;
}

static iree_status_t loom_low_lower_report_calculate_source_block_counts(
    loom_low_lower_context_t* context, loom_region_t* body,
    iree_arena_allocator_t* analysis_arena) {
  loom_cfg_graph_t graph = {0};
  IREE_RETURN_IF_ERROR(
      loom_cfg_graph_build(context->module, body, analysis_arena, &graph));
  if (graph.malformed || graph.block_count != body->block_count) {
    context->lowering.report.source_block_execution_counts_exact = false;
    return iree_ok_status();
  }
  loom_cfg_loop_forest_t loop_forest = {0};
  IREE_RETURN_IF_ERROR(
      loom_cfg_loop_forest_build(&graph, analysis_arena, &loop_forest));
  uint64_t* trip_counts = NULL;
  if (loop_forest.interval_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(analysis_arena, loop_forest.interval_count,
                                  sizeof(*trip_counts), (void**)&trip_counts));
  }
  for (iree_host_size_t i = 0; i < loop_forest.interval_count; ++i) {
    if (!loom_low_lower_report_try_counted_cfg_loop(
            context, &graph, &loop_forest.intervals[i], &trip_counts[i])) {
      context->lowering.report.source_block_execution_counts_exact = false;
      return iree_ok_status();
    }
  }
  context->lowering.report.source_block_execution_counts_exact =
      loom_cfg_loop_forest_calculate_block_execution_counts(
          &loop_forest, &graph, trip_counts,
          context->lowering.report.source_block_execution_counts);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_report_ensure_source_block_counts(
    loom_low_lower_context_t* context) {
  if (context->lowering.report.source_block_execution_counts_initialized) {
    return iree_ok_status();
  }
  context->lowering.report.source_block_execution_counts_initialized = true;
  context->lowering.report.source_block_execution_counts_exact = true;
  loom_region_t* body = loom_func_like_body(context->source_function);
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, body->block_count,
      sizeof(*context->lowering.report.source_block_execution_counts),
      (void**)&context->lowering.report.source_block_execution_counts));

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(context->module->arena.block_pool, &analysis_arena);
  iree_status_t status = loom_low_lower_report_calculate_source_block_counts(
      context, body, &analysis_arena);
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

static const loom_block_t* loom_low_lower_source_op_function_block(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_block_t* block = source_op ? source_op->parent_block : NULL;
  for (const loom_op_t* parent = source_op ? source_op->parent_op : NULL;
       parent != NULL && parent != context->source_function.op;
       parent = parent->parent_op) {
    if (parent->parent_block != NULL) {
      block = parent->parent_block;
    }
  }
  return block;
}

iree_status_t loom_low_lower_source_op_execution_count_plus_one(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t* out_execution_count_plus_one) {
  uint64_t execution_count = 1;
  for (const loom_op_t* parent = source_op ? source_op->parent_op : NULL;
       parent; parent = parent->parent_op) {
    loom_loop_like_t loop =
        loom_loop_like_cast(context->module, (loom_op_t*)parent);
    if (!loom_loop_like_isa(loop)) {
      continue;
    }
    uint64_t trip_count = 0;
    if (!loom_low_lower_report_exact_trip_count(context->lowering.fact_table,
                                                loop, &trip_count) ||
        !loom_low_lower_report_multiply_u64(execution_count, trip_count,
                                            &execution_count) ||
        execution_count == UINT64_MAX) {
      *out_execution_count_plus_one =
          LOOM_LOW_LOWER_MEMORY_REPORT_EXECUTION_COUNT_PLUS_ONE_UNKNOWN;
      return iree_ok_status();
    }
  }

  const loom_block_t* function_block =
      loom_low_lower_source_op_function_block(context, source_op);
  loom_region_t* body = loom_func_like_body(context->source_function);
  uint16_t block_index = 0;
  if (body != NULL && function_block != NULL &&
      loom_region_try_block_index(body, function_block, &block_index)) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_report_ensure_source_block_counts(context));
    if (!context->lowering.report.source_block_execution_counts_exact ||
        block_index >= body->block_count ||
        !loom_low_lower_report_multiply_u64(
            execution_count,
            context->lowering.report.source_block_execution_counts[block_index],
            &execution_count)) {
      *out_execution_count_plus_one =
          LOOM_LOW_LOWER_MEMORY_REPORT_EXECUTION_COUNT_PLUS_ONE_UNKNOWN;
      return iree_ok_status();
    }
  }

  *out_execution_count_plus_one = execution_count + 1;
  return iree_ok_status();
}

iree_status_t loom_low_lower_record_memory_report_row(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_memory_report_row_t* row) {
  if (!loom_low_lower_context_wants_report_rows(context)) {
    return iree_ok_status();
  }
  loom_low_lower_memory_report_row_t counted_row = *row;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_op_execution_count_plus_one(
      context, source_op, &counted_row.execution_count_plus_one));
  return loom_low_lower_memory_report_row_list_append(
      &context->result->memory_report_rows,
      context->result->memory_report_row_allocator, &counted_row);
}
