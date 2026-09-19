// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/report.h"

#include <stdint.h>
#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/rule_emit.h"
#include "loom/codegen/low/lower/source_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/util/cfg_loop_nest.h"
#include "loom/util/fact_cfg.h"

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
      .plan_id = selected_plan->kind == LOOM_LOW_LOWER_SELECTED_PLAN_RULE
                     ? LOOM_LOW_LOWER_PLAN_ID_NONE
                     : selected_plan->data.target_plan.id,
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
        selected_plan->rule->metadata.emit.primary_emit_ordinal !=
            LOOM_LOW_LOWER_RULE_PRIMARY_EMIT_NONE &&
        selected_plan->resolved_emits != NULL) {
      const uint16_t primary_emit_ordinal =
          selected_plan->rule->metadata.emit.primary_emit_ordinal;
      IREE_ASSERT_LT(primary_emit_ordinal, selected_plan->rule->emit_count);
      loom_low_lower_report_populate_descriptor(
          context,
          selected_plan->resolved_emits[primary_emit_ordinal]
              .descriptor.descriptor,
          &row);
    }
  } else if (selected_plan->kind ==
             LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX) {
    const loom_low_lower_descriptor_matrix_plan_t* plan =
        (const loom_low_lower_descriptor_matrix_plan_t*)
            selected_plan->data.target_plan.target_data;
    loom_low_lower_report_populate_descriptor(
        context, plan->descriptor.descriptor, &row);
    row.native_contraction_facts = plan->native_contraction_facts;
  }
  if (selected_plan->rule == NULL &&
      context->policy->describe_plan.fn != NULL) {
    loom_low_lower_plan_report_t plan_report = {0};
    context->policy->describe_plan.fn(context->policy->describe_plan.user_data,
                                      context, selected_plan->source_op,
                                      selected_plan->data.target_plan,
                                      &plan_report);
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

static iree_status_t loom_low_lower_report_calculate_source_block_counts(
    loom_low_lower_context_t* context, loom_region_t* body,
    iree_arena_allocator_t* analysis_arena) {
  if (!iree_any_bit_set(body->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    context->lowering.report.source_block_execution_counts[0] = 1;
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* region =
      loom_value_fact_table_lookup_cfg_region(context->lowering.fact_table,
                                              body);
  const loom_cfg_loop_nest_t* loops = &region->loops;
  uint64_t* trip_counts = NULL;
  if (loops->loop_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(analysis_arena, loops->loop_count,
                                  sizeof(*trip_counts), (void**)&trip_counts));
  }
  for (iree_host_size_t i = 0; i < loops->loop_count; ++i) {
    const loom_loop_recurrence_facts_t recurrence =
        loom_value_fact_cfg_induction_facts(context->lowering.fact_table,
                                            context->module,
                                            &region->inductions[i]);
    if (!recurrence.trip_count_known) {
      context->lowering.report.source_block_execution_counts_exact = false;
      return iree_ok_status();
    }
    trip_counts[i] = recurrence.trip_count;
  }
  context->lowering.report.source_block_execution_counts_exact =
      loom_cfg_loop_nest_calculate_block_execution_counts(
          loops, trip_counts,
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
