// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr.h"

#include <string.h>

#include "loom/analysis/condition_facts.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/adaptive_sort.h"

//===----------------------------------------------------------------------===//
// Context storage
//===----------------------------------------------------------------------===//

enum loom_symbolic_expr_memo_state_e {
  LOOM_SYMBOLIC_EXPR_MEMO_EMPTY = 0,
  LOOM_SYMBOLIC_EXPR_MEMO_VISITING = 1,
  LOOM_SYMBOLIC_EXPR_MEMO_READY = 2,
};

struct loom_symbolic_expr_memo_entry_t {
  // Current memo state for this value ID.
  uint8_t state;

  // Cached expression when state is LOOM_SYMBOLIC_EXPR_MEMO_READY.
  loom_symbolic_expr_t expression;
};

static loom_value_facts_t loom_symbolic_expr_intersect_integer_facts(
    loom_value_facts_t lhs, loom_value_facts_t rhs) {
  if (loom_value_facts_is_unknown(lhs) || loom_value_facts_is_float(lhs)) {
    return rhs;
  }
  if (loom_value_facts_is_unknown(rhs) || loom_value_facts_is_float(rhs)) {
    return lhs;
  }
  int64_t lower_bound = iree_max(lhs.range_lo, rhs.range_lo);
  int64_t upper_bound = iree_min(lhs.range_hi, rhs.range_hi);
  if (lower_bound > upper_bound) {
    return loom_value_facts_unknown();
  }
  return loom_value_facts_make(lower_bound, upper_bound, 1);
}

static const loom_op_t* loom_symbolic_expr_value_defining_op(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  if (!context->module || value_id >= context->module->values.count) {
    return NULL;
  }
  const loom_value_t* value = loom_module_value(context->module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

void loom_symbolic_expr_context_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_symbolic_expr_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->module = module;
  out_context->fact_table = fact_table;
  out_context->arena = arena;
  out_context->maximum_term_count = LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT;
}

void loom_symbolic_expr_context_reset(loom_symbolic_expr_context_t* context) {
  if (context->memo_entries && context->memo_capacity > 0) {
    memset(context->memo_entries, 0,
           context->memo_capacity * sizeof(*context->memo_entries));
  }
  if (context->condition_fact_memo_entries &&
      context->condition_fact_memo_capacity > 0) {
    memset(context->condition_fact_memo_entries, 0,
           context->condition_fact_memo_capacity *
               sizeof(*context->condition_fact_memo_entries));
  }
}

static iree_status_t loom_symbolic_expr_ensure_memo_capacity(
    loom_symbolic_expr_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= context->memo_capacity) return iree_ok_status();
  iree_host_size_t old_capacity = context->memo_capacity;
  void* entries = context->memo_entries;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->arena, context->memo_capacity, minimum_capacity,
      sizeof(*context->memo_entries), &context->memo_capacity, &entries));
  context->memo_entries = (loom_symbolic_expr_memo_entry_t*)entries;
  memset(
      context->memo_entries + old_capacity, 0,
      (context->memo_capacity - old_capacity) * sizeof(*context->memo_entries));
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_ensure_scratch_terms(
    loom_symbolic_expr_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= context->scratch_term_capacity) {
    return iree_ok_status();
  }
  void* terms = context->scratch_terms;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->arena, 0, minimum_capacity, sizeof(*context->scratch_terms),
      &context->scratch_term_capacity, &terms));
  context->scratch_terms = (loom_symbolic_term_t*)terms;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Constructors and normalization
//===----------------------------------------------------------------------===//

static bool loom_symbolic_term_less(const loom_symbolic_term_t* lhs,
                                    const loom_symbolic_term_t* rhs) {
  return lhs->value_id < rhs->value_id;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_symbolic_expr_sort_terms, loom_symbolic_term_t,
                          loom_symbolic_term_less)

loom_value_facts_t loom_symbolic_expr_context_lookup_facts(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  loom_value_facts_t facts =
      context->fact_table
          ? loom_value_fact_table_lookup(context->fact_table, value_id)
          : loom_value_facts_unknown();
  if (context->condition_facts && context->fact_table) {
    loom_condition_fact_set_apply_to_value_facts(
        context->condition_facts, context->fact_table, value_id, &facts);
    while (context->module && value_id < context->module->values.count) {
      const loom_op_t* defining_op =
          loom_symbolic_expr_value_defining_op(context, value_id);
      bool condition = false;
      if (!defining_op || !loom_scf_select_isa(defining_op) ||
          !loom_condition_fact_set_proves_condition(
              context->module, context->fact_table, context->condition_facts,
              loom_scf_select_condition(defining_op), &condition)) {
        break;
      }
      value_id = condition ? loom_scf_select_true_value(defining_op)
                           : loom_scf_select_false_value(defining_op);
      loom_value_facts_t selected_facts =
          loom_value_fact_table_lookup(context->fact_table, value_id);
      loom_condition_fact_set_apply_to_value_facts(context->condition_facts,
                                                   context->fact_table,
                                                   value_id, &selected_facts);
      facts = loom_symbolic_expr_intersect_integer_facts(facts, selected_facts);
    }
  }
  return facts;
}

static bool loom_symbolic_expr_exact_integer_facts(loom_value_facts_t facts,
                                                   int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_symbolic_expr_projected_func_arg_representative(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_id_t* out_representative) {
  *out_representative = value_id;
  if (!context->module || !context->fact_table ||
      value_id >= context->module->values.count) {
    return false;
  }
  loom_func_like_t function = context->fact_table->context.function;
  if (!loom_func_like_isa(function)) {
    return false;
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (arguments[i] == value_id) {
      *out_representative = value_id;
      return true;
    }
  }

  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (!loom_value_is_block_arg(value)) {
    return false;
  }
  loom_block_t* block = loom_value_def_block(value);
  if (!block || !block->parent_region ||
      block != loom_region_entry_block(block->parent_region)) {
    return false;
  }
  uint16_t arg_index = loom_value_def_index(value);
  if (arg_index >= argument_count) {
    return false;
  }

  for (uint8_t i = 0; i < loom_func_like_region_count(function); ++i) {
    if (loom_func_like_region(function, i) != block->parent_region) {
      continue;
    }
    if (!loom_func_like_region_projects_args(context->module, function, i)) {
      return false;
    }
    *out_representative = arguments[arg_index];
    return true;
  }
  return false;
}

static bool loom_symbolic_expr_checked_term_count(iree_host_size_t left_count,
                                                  iree_host_size_t right_count,
                                                  iree_host_size_t* out_count) {
  if (left_count > (iree_host_size_t)-1 - right_count) return false;
  *out_count = left_count + right_count;
  return true;
}

void loom_symbolic_expr_unknown(loom_value_facts_t facts,
                                loom_symbolic_expr_t* out_expression) {
  *out_expression = (loom_symbolic_expr_t){
      .constant = 0,
      .terms = NULL,
      .term_count = 0,
      .facts = facts,
      .flags = 0,
  };
}

void loom_symbolic_expr_constant(int64_t value,
                                 loom_symbolic_expr_t* out_expression) {
  *out_expression = (loom_symbolic_expr_t){
      .constant = value,
      .terms = NULL,
      .term_count = 0,
      .facts = loom_value_facts_exact_i64(value),
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
  };
}

static iree_status_t loom_symbolic_expr_make_linear(
    loom_symbolic_expr_context_t* context, int64_t constant,
    loom_symbolic_term_t* terms, iree_host_size_t term_count,
    loom_value_facts_t facts, loom_symbolic_expr_t* out_expression) {
  if (term_count > context->maximum_term_count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  if (term_count > 1) {
    loom_symbolic_expr_sort_terms(terms, term_count);
  }

  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < term_count;) {
    loom_value_id_t value_id = terms[read_index].value_id;
    loom_value_id_t relation_value_id = terms[read_index].relation_value_id;
    if (relation_value_id == LOOM_VALUE_ID_INVALID) {
      relation_value_id = value_id;
    }
    int64_t coefficient = 0;
    while (read_index < term_count && terms[read_index].value_id == value_id) {
      loom_value_id_t term_relation_value_id =
          terms[read_index].relation_value_id;
      if (term_relation_value_id == LOOM_VALUE_ID_INVALID) {
        term_relation_value_id = value_id;
      }
      if (term_relation_value_id != relation_value_id) {
        relation_value_id = value_id;
      }
      int64_t new_coefficient = 0;
      if (!iree_checked_add_i64(coefficient, terms[read_index].coefficient,
                                &new_coefficient)) {
        loom_symbolic_expr_unknown(facts, out_expression);
        return iree_ok_status();
      }
      coefficient = new_coefficient;
      ++read_index;
    }
    if (coefficient == 0) continue;
    terms[write_index++] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = value_id,
        .relation_value_id = relation_value_id,
    };
  }

  const loom_symbolic_term_t* retained_terms = NULL;
  if (write_index > 0) {
    loom_symbolic_term_t* copied_terms = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, write_index,
                                                   sizeof(*copied_terms),
                                                   (void**)&copied_terms));
    memcpy(copied_terms, terms, write_index * sizeof(*copied_terms));
    retained_terms = copied_terms;
  }

  *out_expression = (loom_symbolic_expr_t){
      .constant = constant,
      .terms = retained_terms,
      .term_count = write_index,
      .facts = facts,
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
  };
  return iree_ok_status();
}

iree_status_t loom_symbolic_expr_value(loom_symbolic_expr_context_t* context,
                                       loom_value_id_t value_id,
                                       loom_symbolic_expr_t* out_expression) {
  loom_value_facts_t facts =
      loom_symbolic_expr_context_lookup_facts(context, value_id);
  if (!context->module || value_id >= context->module->values.count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  loom_value_id_t representative = value_id;
  (void)loom_symbolic_expr_projected_func_arg_representative(context, value_id,
                                                             &representative);
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_ensure_scratch_terms(context, 1));
  context->scratch_terms[0] = (loom_symbolic_term_t){
      .coefficient = 1,
      .value_id = representative,
      .relation_value_id = value_id,
  };
  return loom_symbolic_expr_make_linear(context, 0, context->scratch_terms, 1,
                                        facts, out_expression);
}

static void loom_symbolic_expr_override_facts(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_t* expression) {
  loom_value_facts_t facts =
      loom_symbolic_expr_context_lookup_facts(context, value_id);
  if (!loom_value_facts_is_unknown(facts)) expression->facts = facts;
}

static iree_status_t loom_symbolic_expr_add_or_sub(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, bool subtract,
    loom_symbolic_expr_t* out_expression) {
  loom_value_facts_t facts = {0};
  if (subtract) {
    loom_value_facts_subi(&left_expression->facts, &right_expression->facts,
                          &facts);
  } else {
    loom_value_facts_addi(&left_expression->facts, &right_expression->facts,
                          &facts);
  }
  if (!loom_symbolic_expr_is_linear(left_expression) ||
      !loom_symbolic_expr_is_linear(right_expression)) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }

  int64_t constant = 0;
  bool constant_ok =
      subtract ? iree_checked_sub_i64(left_expression->constant,
                                      right_expression->constant, &constant)
               : iree_checked_add_i64(left_expression->constant,
                                      right_expression->constant, &constant);
  if (!constant_ok) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }

  iree_host_size_t term_count = 0;
  if (!loom_symbolic_expr_checked_term_count(left_expression->term_count,
                                             right_expression->term_count,
                                             &term_count)) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  if (term_count > context->maximum_term_count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_ensure_scratch_terms(context, term_count));
  iree_host_size_t term_ordinal = 0;
  for (iree_host_size_t i = 0; i < left_expression->term_count; ++i) {
    context->scratch_terms[term_ordinal++] = left_expression->terms[i];
  }
  for (iree_host_size_t i = 0; i < right_expression->term_count; ++i) {
    int64_t coefficient = right_expression->terms[i].coefficient;
    if (subtract) {
      if (coefficient == INT64_MIN) {
        loom_symbolic_expr_unknown(facts, out_expression);
        return iree_ok_status();
      }
      coefficient = -coefficient;
    }
    context->scratch_terms[term_ordinal++] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = right_expression->terms[i].value_id,
        .relation_value_id = right_expression->terms[i].relation_value_id,
    };
  }
  return loom_symbolic_expr_make_linear(context, constant,
                                        context->scratch_terms, term_ordinal,
                                        facts, out_expression);
}

iree_status_t loom_symbolic_expr_add(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_add_or_sub(context, left_expression,
                                       right_expression, false, out_expression);
}

iree_status_t loom_symbolic_expr_sub(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_add_or_sub(context, left_expression,
                                       right_expression, true, out_expression);
}

iree_status_t loom_symbolic_expr_mul_i64(loom_symbolic_expr_context_t* context,
                                         const loom_symbolic_expr_t* expression,
                                         int64_t multiplier,
                                         loom_symbolic_expr_t* out_expression) {
  loom_value_facts_t multiplier_facts = loom_value_facts_exact_i64(multiplier);
  loom_value_facts_t facts = {0};
  loom_value_facts_muli(&expression->facts, &multiplier_facts, &facts);
  if (!loom_symbolic_expr_is_linear(expression)) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }

  int64_t constant = 0;
  if (!iree_checked_mul_i64(expression->constant, multiplier, &constant)) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  if (expression->term_count > context->maximum_term_count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_ensure_scratch_terms(context, expression->term_count));
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    int64_t coefficient = 0;
    if (!iree_checked_mul_i64(expression->terms[i].coefficient, multiplier,
                              &coefficient)) {
      loom_symbolic_expr_unknown(facts, out_expression);
      return iree_ok_status();
    }
    context->scratch_terms[i] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = expression->terms[i].value_id,
        .relation_value_id = expression->terms[i].relation_value_id,
    };
  }
  return loom_symbolic_expr_make_linear(
      context, constant, context->scratch_terms, expression->term_count, facts,
      out_expression);
}

static bool loom_symbolic_expr_constant_value(
    const loom_symbolic_expr_t* expression, int64_t* out_value) {
  if (!loom_symbolic_expr_is_constant(expression)) return false;
  *out_value = expression->constant;
  return true;
}

//===----------------------------------------------------------------------===//
// Value expansion
//===----------------------------------------------------------------------===//

static iree_status_t loom_symbolic_expr_from_binary_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value, bool subtract,
    loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t left_expression = {0};
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));
  return loom_symbolic_expr_add_or_sub(
      context, &left_expression, &right_expression, subtract, out_expression);
}

static iree_status_t loom_symbolic_expr_from_mul_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t result_value,
    loom_value_id_t left_value, loom_value_id_t right_value,
    loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t left_expression = {0};
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));
  int64_t left_constant = 0;
  int64_t right_constant = 0;
  if (loom_symbolic_expr_constant_value(&left_expression, &left_constant)) {
    return loom_symbolic_expr_mul_i64(context, &right_expression, left_constant,
                                      out_expression);
  }
  if (loom_symbolic_expr_constant_value(&right_expression, &right_constant)) {
    return loom_symbolic_expr_mul_i64(context, &left_expression, right_constant,
                                      out_expression);
  }
  return loom_symbolic_expr_value(context, result_value, out_expression);
}

static iree_status_t loom_symbolic_expr_from_madd_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t result_value,
    loom_value_id_t a_value, loom_value_id_t b_value, loom_value_id_t c_value,
    loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t product_expression = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_mul_value(
      context, result_value, a_value, b_value, &product_expression));
  if (product_expression.term_count == 1 &&
      product_expression.terms[0].value_id == result_value &&
      product_expression.constant == 0) {
    return loom_symbolic_expr_value(context, result_value, out_expression);
  }
  loom_symbolic_expr_t c_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, c_value, &c_expression));
  return loom_symbolic_expr_add(context, &product_expression, &c_expression,
                                out_expression);
}

static iree_status_t loom_symbolic_expr_from_shli_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t result_value,
    loom_value_id_t left_value, loom_value_id_t right_value,
    loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));
  int64_t shift_amount = 0;
  if (!loom_symbolic_expr_constant_value(&right_expression, &shift_amount) ||
      shift_amount < 0 || shift_amount > 62) {
    return loom_symbolic_expr_value(context, result_value, out_expression);
  }
  loom_symbolic_expr_t left_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  return loom_symbolic_expr_mul_i64(context, &left_expression,
                                    INT64_C(1) << shift_amount, out_expression);
}

static iree_status_t loom_symbolic_expr_from_select_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t result_value,
    loom_value_id_t condition_value, loom_value_id_t true_value,
    loom_value_id_t false_value, loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t condition_expression = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(context, condition_value,
                                                     &condition_expression));
  int64_t condition = 0;
  if (loom_symbolic_expr_constant_value(&condition_expression, &condition)) {
    return loom_symbolic_expr_from_value(
        context, condition ? true_value : false_value, out_expression);
  }
  return loom_symbolic_expr_value(context, result_value, out_expression);
}

static iree_status_t loom_symbolic_expr_attach_identity_relation_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t relation_value,
    loom_symbolic_expr_t* expression) {
  if (!loom_symbolic_expr_is_linear(expression) || expression->constant != 0 ||
      expression->term_count != 1 || expression->terms[0].coefficient != 1) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_ensure_scratch_terms(context, 1));
  context->scratch_terms[0] = expression->terms[0];
  context->scratch_terms[0].relation_value_id = relation_value;
  return loom_symbolic_expr_make_linear(context, expression->constant,
                                        context->scratch_terms, 1,
                                        expression->facts, expression);
}

static iree_status_t loom_symbolic_expr_from_assume_value(
    loom_symbolic_expr_context_t* context, const loom_value_slice_t values,
    loom_value_id_t result_value, uint16_t result_index,
    loom_symbolic_expr_t* out_expression) {
  if (result_index >= values.count) {
    loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      context, values.values[result_index], out_expression));
  return loom_symbolic_expr_attach_identity_relation_value(
      context, result_value, out_expression);
}

static iree_status_t loom_symbolic_expr_from_value_uncached(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_t* out_expression) {
  loom_value_facts_t facts =
      loom_symbolic_expr_context_lookup_facts(context, value_id);
  int64_t exact_value = 0;
  if (loom_symbolic_expr_exact_integer_facts(facts, &exact_value)) {
    loom_symbolic_expr_constant(exact_value, out_expression);
    out_expression->facts = facts;
    return iree_ok_status();
  }
  if (!context->module || value_id >= context->module->values.count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_symbolic_expr_value(context, value_id, out_expression);
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return loom_symbolic_expr_value(context, value_id, out_expression);
  }

  iree_status_t status = iree_ok_status();
  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CONSTANT: {
      loom_attribute_t value_attr = loom_index_constant_value(defining_op);
      if (value_attr.kind == LOOM_ATTR_I64) {
        loom_symbolic_expr_constant(loom_attr_as_i64(value_attr),
                                    out_expression);
      } else {
        status = loom_symbolic_expr_value(context, value_id, out_expression);
      }
      break;
    }
    case LOOM_OP_INDEX_CAST:
      status = loom_symbolic_expr_from_value(
          context, loom_index_cast_input(defining_op), out_expression);
      break;
    case LOOM_OP_INDEX_ASSUME:
      status = loom_symbolic_expr_from_assume_value(
          context, loom_index_assume_values(defining_op), value_id,
          loom_value_def_index(value), out_expression);
      break;
    case LOOM_OP_INDEX_ADD:
      status = loom_symbolic_expr_from_binary_value(
          context, loom_index_add_lhs(defining_op),
          loom_index_add_rhs(defining_op), false, out_expression);
      break;
    case LOOM_OP_INDEX_SUB:
      status = loom_symbolic_expr_from_binary_value(
          context, loom_index_sub_lhs(defining_op),
          loom_index_sub_rhs(defining_op), true, out_expression);
      break;
    case LOOM_OP_INDEX_MUL:
      status = loom_symbolic_expr_from_mul_value(
          context, value_id, loom_index_mul_lhs(defining_op),
          loom_index_mul_rhs(defining_op), out_expression);
      break;
    case LOOM_OP_INDEX_SCALE:
      status = loom_symbolic_expr_from_mul_value(
          context, value_id, loom_index_scale_index(defining_op),
          loom_index_scale_stride(defining_op), out_expression);
      break;
    case LOOM_OP_INDEX_MADD:
      status = loom_symbolic_expr_from_madd_value(
          context, value_id, loom_index_madd_a(defining_op),
          loom_index_madd_b(defining_op), loom_index_madd_c(defining_op),
          out_expression);
      break;
    case LOOM_OP_INDEX_SHLI:
      status = loom_symbolic_expr_from_shli_value(
          context, value_id, loom_index_shli_lhs(defining_op),
          loom_index_shli_rhs(defining_op), out_expression);
      break;
    case LOOM_OP_SCALAR_CONSTANT: {
      loom_attribute_t value_attr = loom_scalar_constant_value(defining_op);
      if (value_attr.kind == LOOM_ATTR_I64) {
        loom_symbolic_expr_constant(loom_attr_as_i64(value_attr),
                                    out_expression);
      } else {
        status = loom_symbolic_expr_value(context, value_id, out_expression);
      }
      break;
    }
    case LOOM_OP_SCALAR_ADDI:
      status = loom_symbolic_expr_from_binary_value(
          context, loom_scalar_addi_lhs(defining_op),
          loom_scalar_addi_rhs(defining_op), false, out_expression);
      break;
    case LOOM_OP_SCALAR_SUBI:
      status = loom_symbolic_expr_from_binary_value(
          context, loom_scalar_subi_lhs(defining_op),
          loom_scalar_subi_rhs(defining_op), true, out_expression);
      break;
    case LOOM_OP_SCALAR_MULI:
      status = loom_symbolic_expr_from_mul_value(
          context, value_id, loom_scalar_muli_lhs(defining_op),
          loom_scalar_muli_rhs(defining_op), out_expression);
      break;
    case LOOM_OP_SCALAR_NEGI: {
      loom_symbolic_expr_t input_expression = {0};
      status = loom_symbolic_expr_from_value(
          context, loom_scalar_negi_input(defining_op), &input_expression);
      if (iree_status_is_ok(status)) {
        status = loom_symbolic_expr_mul_i64(context, &input_expression, -1,
                                            out_expression);
      }
      break;
    }
    case LOOM_OP_SCALAR_FMAI:
      status = loom_symbolic_expr_from_madd_value(
          context, value_id, loom_scalar_fmai_a(defining_op),
          loom_scalar_fmai_b(defining_op), loom_scalar_fmai_c(defining_op),
          out_expression);
      break;
    case LOOM_OP_SCALAR_ASSUME:
      status = loom_symbolic_expr_from_assume_value(
          context, loom_scalar_assume_values(defining_op), value_id,
          loom_value_def_index(value), out_expression);
      break;
    case LOOM_OP_SCF_SELECT:
      status = loom_symbolic_expr_from_select_value(
          context, value_id, loom_scf_select_condition(defining_op),
          loom_scf_select_true_value(defining_op),
          loom_scf_select_false_value(defining_op), out_expression);
      break;
    default:
      status = loom_symbolic_expr_value(context, value_id, out_expression);
      break;
  }
  if (iree_status_is_ok(status)) {
    if (!loom_symbolic_expr_is_linear(out_expression)) {
      status = loom_symbolic_expr_value(context, value_id, out_expression);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_symbolic_expr_override_facts(context, value_id, out_expression);
  }
  return status;
}

iree_status_t loom_symbolic_expr_from_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_t* out_expression) {
  if (!context->module || value_id >= context->module->values.count) {
    loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_ensure_memo_capacity(context, value_id + 1));
  loom_symbolic_expr_memo_entry_t* entry = &context->memo_entries[value_id];
  if (entry->state == LOOM_SYMBOLIC_EXPR_MEMO_READY) {
    *out_expression = entry->expression;
    return iree_ok_status();
  }
  if (entry->state == LOOM_SYMBOLIC_EXPR_MEMO_VISITING) {
    return loom_symbolic_expr_value(context, value_id, out_expression);
  }

  entry->state = LOOM_SYMBOLIC_EXPR_MEMO_VISITING;
  iree_status_t status =
      loom_symbolic_expr_from_value_uncached(context, value_id, out_expression);
  // Recursive expansion can grow and move the memo table when a lower value ID
  // expands through a producer with a higher value ID. Reacquire the entry
  // before publishing the completed expression.
  entry = &context->memo_entries[value_id];
  if (iree_status_is_ok(status)) {
    entry->expression = *out_expression;
    entry->state = LOOM_SYMBOLIC_EXPR_MEMO_READY;
  } else {
    entry->state = LOOM_SYMBOLIC_EXPR_MEMO_EMPTY;
  }
  return status;
}
