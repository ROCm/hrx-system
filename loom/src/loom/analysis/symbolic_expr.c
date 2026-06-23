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
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Context storage
//===----------------------------------------------------------------------===//

enum loom_symbolic_expr_memo_state_e {
  LOOM_SYMBOLIC_EXPR_MEMO_EMPTY = 0,
  LOOM_SYMBOLIC_EXPR_MEMO_VISITING = 1,
  LOOM_SYMBOLIC_EXPR_MEMO_READY = 2,
};

#define LOOM_SYMBOLIC_EXPR_CONDITION_FACT_INFER_DEPTH_LIMIT 16
#define LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_LIMIT 64
#define LOOM_SYMBOLIC_EXPR_SELECT_CASE_CONDITION_LIMIT 8
#define LOOM_SYMBOLIC_EXPR_SELECT_CASE_DEPTH_LIMIT 2
#define LOOM_SYMBOLIC_EXPR_SELECT_DEPENDENCY_SEARCH_LIMIT 64

struct loom_symbolic_expr_memo_entry_t {
  // Current memo state for this value ID.
  uint8_t state;

  // Cached expression when state is LOOM_SYMBOLIC_EXPR_MEMO_READY.
  loom_symbolic_expr_t expression;
};

static loom_value_facts_t loom_symbolic_expr_intersect_integer_facts(
    loom_value_facts_t lhs, loom_value_facts_t rhs);
static const loom_op_t* loom_symbolic_expr_value_defining_op(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id);
static iree_status_t
loom_symbolic_expr_apply_identity_chain_predicates_to_value_facts(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_facts_t* inout_facts);
static iree_status_t loom_symbolic_expr_values_match(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value, bool* out_match);

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

static iree_status_t loom_symbolic_expr_stabilize_scratch_terms(
    loom_symbolic_expr_context_t* context, iree_host_size_t term_count,
    loom_symbolic_term_t* stack_terms, iree_host_size_t stack_term_capacity,
    const loom_symbolic_term_t** out_terms) {
  if (term_count == 0) {
    *out_terms = NULL;
    return iree_ok_status();
  }
  loom_symbolic_term_t* stable_terms = stack_terms;
  if (term_count > stack_term_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, term_count,
                                                   sizeof(*stable_terms),
                                                   (void**)&stable_terms));
  }
  memcpy(stable_terms, context->scratch_terms,
         term_count * sizeof(*stable_terms));
  *out_terms = stable_terms;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Constructors and normalization
//===----------------------------------------------------------------------===//

static loom_value_facts_t loom_symbolic_expr_lookup_facts(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  loom_value_facts_t facts =
      context->fact_table
          ? loom_value_fact_table_lookup(context->fact_table, value_id)
          : loom_value_facts_unknown();
  if (context->condition_facts && context->fact_table) {
    loom_condition_fact_set_apply_to_value_facts(
        context->condition_facts, context->fact_table, value_id, &facts);
    if (context->module && value_id < context->module->values.count) {
      const loom_op_t* defining_op =
          loom_symbolic_expr_value_defining_op(context, value_id);
      bool condition = false;
      if (defining_op && loom_scf_select_isa(defining_op) &&
          loom_condition_fact_set_proves_condition(
              context->module, context->fact_table, context->condition_facts,
              loom_scf_select_condition(defining_op), &condition)) {
        loom_value_id_t selected_value =
            condition ? loom_scf_select_true_value(defining_op)
                      : loom_scf_select_false_value(defining_op);
        facts = loom_symbolic_expr_intersect_integer_facts(
            facts, loom_symbolic_expr_lookup_facts(context, selected_value));
      }
    }
  }
  return facts;
}

static iree_status_t loom_symbolic_expr_lookup_condition_refined_facts(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    uint8_t remaining_depth, loom_value_facts_t* out_facts) {
  loom_value_facts_t facts = loom_symbolic_expr_lookup_facts(context, value_id);
  if (!context->condition_facts || !context->module || !context->fact_table ||
      remaining_depth == 0 || value_id >= context->module->values.count) {
    *out_facts = facts;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_apply_identity_chain_predicates_to_value_facts(
          context, value_id, &facts));

  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) {
    *out_facts = facts;
    return iree_ok_status();
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  const loom_op_vtable_t* vtable =
      defining_op ? loom_op_vtable(context->module, defining_op) : NULL;
  if (!vtable || !vtable->infer_facts) {
    *out_facts = facts;
    return iree_ok_status();
  }

  const uint16_t result_index = loom_value_def_index(value);
  if (result_index >= defining_op->result_count ||
      !loom_type_is_scalar(loom_module_value_type(context->module, value_id))) {
    *out_facts = facts;
    return iree_ok_status();
  }

  loom_value_facts_t* operand_facts = NULL;
  if (defining_op->operand_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        context->arena, defining_op->operand_count * sizeof(*operand_facts),
        (void**)&operand_facts));
  }
  const loom_value_id_t* operands = loom_op_const_operands(defining_op);
  for (uint16_t i = 0; i < defining_op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_lookup_condition_refined_facts(
        context, operands[i], (uint8_t)(remaining_depth - 1),
        &operand_facts[i]));
  }

  loom_value_facts_t* result_facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      context->arena, defining_op->result_count * sizeof(*result_facts),
      (void**)&result_facts));
  for (uint16_t i = 0; i < defining_op->result_count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }

  loom_fact_context_t fact_context = context->fact_table->context;
  IREE_RETURN_IF_ERROR(vtable->infer_facts(&fact_context, context->module,
                                           defining_op, operand_facts,
                                           result_facts));

  loom_value_facts_t inferred_facts = result_facts[result_index];
  loom_condition_fact_set_apply_to_value_facts(
      context->condition_facts, context->fact_table, value_id, &inferred_facts);
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_apply_identity_chain_predicates_to_value_facts(
          context, value_id, &inferred_facts));
  *out_facts =
      loom_symbolic_expr_intersect_integer_facts(facts, inferred_facts);
  return iree_ok_status();
}

static bool loom_symbolic_expr_exact_integer_facts(loom_value_facts_t facts,
                                                   int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

//===----------------------------------------------------------------------===//
// Bounded expression summary
//===----------------------------------------------------------------------===//

#define LOOM_SYMBOLIC_EXPR_BOUNDED_DEPTH_LIMIT 16
#define LOOM_SYMBOLIC_EXPR_BOUNDED_FRAME_CAPACITY 64
#define LOOM_SYMBOLIC_EXPR_BOUNDED_SCRATCH_TERM_CAPACITY 16

static loom_value_facts_t loom_symbolic_expr_bounded_lookup_facts(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id) {
  return fact_table ? loom_value_fact_table_lookup(fact_table, value_id)
                    : loom_value_facts_unknown();
}

static const loom_op_t* loom_symbolic_expr_bounded_defining_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) return NULL;
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

static bool loom_symbolic_expr_bounded_constant_attr(const loom_op_t* op,
                                                     int64_t* out_value) {
  if (op == NULL) return false;
  loom_attribute_t value_attr = {0};
  switch (op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      value_attr = loom_index_constant_value(op);
      break;
    case LOOM_OP_SCALAR_CONSTANT:
      value_attr = loom_scalar_constant_value(op);
      break;
    default:
      return false;
  }
  if (value_attr.kind != LOOM_ATTR_I64) return false;
  *out_value = loom_attr_as_i64(value_attr);
  return true;
}

static bool loom_symbolic_expr_bounded_exact_i64(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, int64_t* out_value) {
  if (loom_symbolic_expr_exact_integer_facts(
          loom_symbolic_expr_bounded_lookup_facts(fact_table, value_id),
          out_value)) {
    return true;
  }
  return loom_symbolic_expr_bounded_constant_attr(
      loom_symbolic_expr_bounded_defining_op(module, value_id), out_value);
}

static bool loom_symbolic_expr_bounded_add_constant(int64_t scaled_value,
                                                    int64_t* inout_constant) {
  int64_t new_constant = 0;
  if (!loom_checked_add_i64(*inout_constant, scaled_value, &new_constant)) {
    return false;
  }
  *inout_constant = new_constant;
  return true;
}

static bool loom_symbolic_expr_bounded_accumulate_constant(
    int64_t value, int64_t coefficient, int64_t* inout_constant) {
  int64_t scaled_value = 0;
  return loom_checked_mul_i64(value, coefficient, &scaled_value) &&
         loom_symbolic_expr_bounded_add_constant(scaled_value, inout_constant);
}

static bool loom_symbolic_expr_bounded_append_term(
    loom_symbolic_term_t* terms, iree_host_size_t term_capacity,
    iree_host_size_t* inout_term_count, loom_value_id_t value_id,
    loom_value_id_t relation_value_id, int64_t coefficient) {
  if (coefficient == 0) return true;
  iree_host_size_t insert_index = 0;
  while (insert_index < *inout_term_count &&
         terms[insert_index].value_id < value_id) {
    ++insert_index;
  }
  if (insert_index < *inout_term_count &&
      terms[insert_index].value_id == value_id) {
    int64_t combined_coefficient = 0;
    if (!loom_checked_add_i64(terms[insert_index].coefficient, coefficient,
                              &combined_coefficient)) {
      return false;
    }
    if (combined_coefficient == 0) {
      memmove(&terms[insert_index], &terms[insert_index + 1],
              (*inout_term_count - insert_index - 1) * sizeof(*terms));
      *inout_term_count -= 1;
      return true;
    }
    terms[insert_index].coefficient = combined_coefficient;
    if (terms[insert_index].relation_value_id != relation_value_id) {
      terms[insert_index].relation_value_id = value_id;
    }
    return true;
  }
  if (*inout_term_count >= term_capacity) return false;
  memmove(&terms[insert_index + 1], &terms[insert_index],
          (*inout_term_count - insert_index) * sizeof(*terms));
  terms[insert_index] = (loom_symbolic_term_t){
      .coefficient = coefficient,
      .value_id = value_id,
      .relation_value_id = relation_value_id,
  };
  *inout_term_count += 1;
  return true;
}

typedef enum loom_symbolic_expr_bounded_frame_kind_e {
  LOOM_SYMBOLIC_EXPR_BOUNDED_FRAME_EXPAND_VALUE = 0,
  LOOM_SYMBOLIC_EXPR_BOUNDED_FRAME_FINISH_ASSUME = 1,
} loom_symbolic_expr_bounded_frame_kind_t;

typedef struct loom_symbolic_expr_bounded_frame_t {
  // Frame category.
  loom_symbolic_expr_bounded_frame_kind_t kind;
  union {
    // Value expansion waiting to be summarized.
    struct {
      // SSA value being expanded.
      loom_value_id_t value_id;
      // Coefficient applied to the expanded value.
      int64_t coefficient;
      // Remaining producer depth before the value becomes opaque.
      uint8_t remaining_depth;
    } expand;
    // Assume-source expansion continuation.
    struct {
      // Assumed result to preserve if the source is an opaque identity.
      loom_value_id_t result_value;
      // Coefficient applied to the assumed result or expanded source.
      int64_t coefficient;
      // Raw term count before expanding the assume source.
      iree_host_size_t term_count;
      // Constant value before expanding the assume source.
      int64_t constant;
    } assume;
  };
} loom_symbolic_expr_bounded_frame_t;

static bool loom_symbolic_expr_bounded_raw_append_term(
    loom_symbolic_term_t* terms, iree_host_size_t term_capacity,
    iree_host_size_t* inout_term_count, loom_value_id_t value_id,
    loom_value_id_t relation_value_id, int64_t coefficient) {
  if (coefficient == 0) return true;
  if (*inout_term_count >= term_capacity) return false;
  terms[(*inout_term_count)++] = (loom_symbolic_term_t){
      .coefficient = coefficient,
      .value_id = value_id,
      .relation_value_id = relation_value_id,
  };
  return true;
}

static bool loom_symbolic_expr_bounded_normalize_raw_terms(
    loom_symbolic_term_t* terms, iree_host_size_t* inout_term_count) {
  const iree_host_size_t term_count = *inout_term_count;
  for (iree_host_size_t i = 1; i < term_count; ++i) {
    const loom_symbolic_term_t term = terms[i];
    iree_host_size_t insertion_index = i;
    while (insertion_index > 0 &&
           terms[insertion_index - 1].value_id > term.value_id) {
      terms[insertion_index] = terms[insertion_index - 1];
      --insertion_index;
    }
    terms[insertion_index] = term;
  }

  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < term_count;) {
    const loom_value_id_t value_id = terms[read_index].value_id;
    loom_value_id_t relation_value_id = terms[read_index].relation_value_id;
    int64_t coefficient = 0;
    while (read_index < term_count && terms[read_index].value_id == value_id) {
      if (terms[read_index].relation_value_id != relation_value_id) {
        relation_value_id = value_id;
      }
      int64_t next_coefficient = 0;
      if (!loom_checked_add_i64(coefficient, terms[read_index].coefficient,
                                &next_coefficient)) {
        return false;
      }
      coefficient = next_coefficient;
      ++read_index;
    }
    if (coefficient == 0) continue;
    terms[write_index++] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = value_id,
        .relation_value_id = relation_value_id,
    };
  }
  *inout_term_count = write_index;
  return true;
}

static bool loom_symbolic_expr_bounded_push_frame(
    loom_symbolic_expr_bounded_frame_t* frames,
    iree_host_size_t* inout_frame_count,
    loom_symbolic_expr_bounded_frame_t frame) {
  if (*inout_frame_count >= LOOM_SYMBOLIC_EXPR_BOUNDED_FRAME_CAPACITY) {
    return false;
  }
  frames[(*inout_frame_count)++] = frame;
  return true;
}

static bool loom_symbolic_expr_bounded_push_expand(
    loom_symbolic_expr_bounded_frame_t* frames,
    iree_host_size_t* inout_frame_count, loom_value_id_t value_id,
    int64_t coefficient, uint8_t remaining_depth) {
  if (coefficient == 0) return true;
  return loom_symbolic_expr_bounded_push_frame(
      frames, inout_frame_count,
      (loom_symbolic_expr_bounded_frame_t){
          .kind = LOOM_SYMBOLIC_EXPR_BOUNDED_FRAME_EXPAND_VALUE,
          .expand =
              {
                  .value_id = value_id,
                  .coefficient = coefficient,
                  .remaining_depth = remaining_depth,
              },
      });
}

static bool loom_symbolic_expr_bounded_push_mul(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t result_value, loom_value_id_t left_value,
    loom_value_id_t right_value, int64_t coefficient, uint8_t remaining_depth,
    loom_symbolic_expr_bounded_frame_t* frames,
    iree_host_size_t* inout_frame_count, loom_symbolic_term_t* raw_terms,
    iree_host_size_t raw_term_capacity,
    iree_host_size_t* inout_raw_term_count) {
  int64_t left_constant = 0;
  if (loom_symbolic_expr_bounded_exact_i64(module, fact_table, left_value,
                                           &left_constant)) {
    int64_t scaled_coefficient = 0;
    if (!loom_checked_mul_i64(coefficient, left_constant,
                              &scaled_coefficient)) {
      return false;
    }
    return loom_symbolic_expr_bounded_push_expand(
        frames, inout_frame_count, right_value, scaled_coefficient,
        remaining_depth);
  }
  int64_t right_constant = 0;
  if (loom_symbolic_expr_bounded_exact_i64(module, fact_table, right_value,
                                           &right_constant)) {
    int64_t scaled_coefficient = 0;
    if (!loom_checked_mul_i64(coefficient, right_constant,
                              &scaled_coefficient)) {
      return false;
    }
    return loom_symbolic_expr_bounded_push_expand(
        frames, inout_frame_count, left_value, scaled_coefficient,
        remaining_depth);
  }
  return loom_symbolic_expr_bounded_raw_append_term(
      raw_terms, raw_term_capacity, inout_raw_term_count, result_value,
      result_value, coefficient);
}

static bool loom_symbolic_expr_bounded_push_shli(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t result_value, loom_value_id_t left_value,
    loom_value_id_t right_value, int64_t coefficient, uint8_t remaining_depth,
    loom_symbolic_expr_bounded_frame_t* frames,
    iree_host_size_t* inout_frame_count, loom_symbolic_term_t* raw_terms,
    iree_host_size_t raw_term_capacity,
    iree_host_size_t* inout_raw_term_count) {
  int64_t shift_amount = 0;
  if (!loom_symbolic_expr_bounded_exact_i64(module, fact_table, right_value,
                                            &shift_amount) ||
      shift_amount < 0 || shift_amount > 62) {
    return loom_symbolic_expr_bounded_raw_append_term(
        raw_terms, raw_term_capacity, inout_raw_term_count, result_value,
        result_value, coefficient);
  }
  int64_t scaled_coefficient = 0;
  if (!loom_checked_mul_i64(coefficient, INT64_C(1) << shift_amount,
                            &scaled_coefficient)) {
    return false;
  }
  return loom_symbolic_expr_bounded_push_expand(frames, inout_frame_count,
                                                left_value, scaled_coefficient,
                                                remaining_depth);
}

static bool loom_symbolic_expr_bounded_push_assume(
    const loom_value_slice_t values, loom_value_id_t result_value,
    uint16_t result_index, int64_t coefficient, uint8_t remaining_depth,
    loom_symbolic_expr_bounded_frame_t* frames,
    iree_host_size_t* inout_frame_count, iree_host_size_t raw_term_count,
    int64_t constant) {
  if (result_index >= values.count) return false;
  return loom_symbolic_expr_bounded_push_frame(
             frames, inout_frame_count,
             (loom_symbolic_expr_bounded_frame_t){
                 .kind = LOOM_SYMBOLIC_EXPR_BOUNDED_FRAME_FINISH_ASSUME,
                 .assume =
                     {
                         .result_value = result_value,
                         .coefficient = coefficient,
                         .term_count = raw_term_count,
                         .constant = constant,
                     },
             }) &&
         loom_symbolic_expr_bounded_push_expand(frames, inout_frame_count,
                                                values.values[result_index], 1,
                                                remaining_depth);
}

static bool loom_symbolic_expr_bounded_finish_assume(
    const loom_symbolic_expr_bounded_frame_t* frame,
    loom_symbolic_term_t* raw_terms, iree_host_size_t raw_term_capacity,
    iree_host_size_t* inout_raw_term_count, int64_t* inout_constant) {
  const iree_host_size_t source_term_count =
      *inout_raw_term_count - frame->assume.term_count;
  int64_t source_constant = 0;
  if (!loom_checked_sub_i64(*inout_constant, frame->assume.constant,
                            &source_constant)) {
    return false;
  }
  const bool is_opaque_identity =
      source_constant == 0 && source_term_count == 1 &&
      raw_terms[frame->assume.term_count].coefficient == 1;
  if (is_opaque_identity) {
    *inout_raw_term_count = frame->assume.term_count;
    *inout_constant = frame->assume.constant;
    return loom_symbolic_expr_bounded_raw_append_term(
        raw_terms, raw_term_capacity, inout_raw_term_count,
        frame->assume.result_value, frame->assume.result_value,
        frame->assume.coefficient);
  }

  int64_t scaled_constant = 0;
  if (!loom_checked_mul_i64(source_constant, frame->assume.coefficient,
                            &scaled_constant) ||
      !loom_checked_add_i64(frame->assume.constant, scaled_constant,
                            inout_constant)) {
    return false;
  }
  for (iree_host_size_t i = frame->assume.term_count; i < *inout_raw_term_count;
       ++i) {
    int64_t scaled_coefficient = 0;
    if (!loom_checked_mul_i64(raw_terms[i].coefficient,
                              frame->assume.coefficient, &scaled_coefficient)) {
      return false;
    }
    raw_terms[i].coefficient = scaled_coefficient;
  }
  return true;
}

static bool loom_symbolic_expr_bounded_accumulate_value(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, int64_t coefficient, uint8_t remaining_depth,
    loom_symbolic_term_t* terms, iree_host_size_t term_capacity,
    iree_host_size_t* inout_term_count, int64_t* inout_constant) {
  loom_symbolic_term_t
      raw_terms[LOOM_SYMBOLIC_EXPR_BOUNDED_SCRATCH_TERM_CAPACITY] = {0};
  iree_host_size_t raw_term_count = 0;
  int64_t constant = 0;
  loom_symbolic_expr_bounded_frame_t
      frames[LOOM_SYMBOLIC_EXPR_BOUNDED_FRAME_CAPACITY] = {0};
  iree_host_size_t frame_count = 0;
  if (!loom_symbolic_expr_bounded_push_expand(frames, &frame_count, value_id,
                                              coefficient, remaining_depth)) {
    return false;
  }

  while (frame_count > 0) {
    loom_symbolic_expr_bounded_frame_t frame = frames[--frame_count];
    if (frame.kind == LOOM_SYMBOLIC_EXPR_BOUNDED_FRAME_FINISH_ASSUME) {
      if (!loom_symbolic_expr_bounded_finish_assume(
              &frame, raw_terms, IREE_ARRAYSIZE(raw_terms), &raw_term_count,
              &constant)) {
        return false;
      }
      continue;
    }

    value_id = frame.expand.value_id;
    coefficient = frame.expand.coefficient;
    remaining_depth = frame.expand.remaining_depth;
    if (coefficient == 0) continue;
    int64_t exact_value = 0;
    if (loom_symbolic_expr_bounded_exact_i64(module, fact_table, value_id,
                                             &exact_value)) {
      if (!loom_symbolic_expr_bounded_accumulate_constant(
              exact_value, coefficient, &constant)) {
        return false;
      }
      continue;
    }
    if (remaining_depth == 0 || !module || value_id >= module->values.count) {
      if (!loom_symbolic_expr_bounded_raw_append_term(
              raw_terms, IREE_ARRAYSIZE(raw_terms), &raw_term_count, value_id,
              value_id, coefficient)) {
        return false;
      }
      continue;
    }
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) {
      if (!loom_symbolic_expr_bounded_raw_append_term(
              raw_terms, IREE_ARRAYSIZE(raw_terms), &raw_term_count, value_id,
              value_id, coefficient)) {
        return false;
      }
      continue;
    }
    const loom_op_t* op = loom_value_def_op(value);
    if (op == NULL) {
      if (!loom_symbolic_expr_bounded_raw_append_term(
              raw_terms, IREE_ARRAYSIZE(raw_terms), &raw_term_count, value_id,
              value_id, coefficient)) {
        return false;
      }
      continue;
    }

    const uint8_t next_depth = (uint8_t)(remaining_depth - 1);
    bool handled = true;
    switch (op->kind) {
      case LOOM_OP_INDEX_CONSTANT:
      case LOOM_OP_SCALAR_CONSTANT: {
        int64_t op_constant = 0;
        handled = loom_symbolic_expr_bounded_constant_attr(op, &op_constant) &&
                  loom_symbolic_expr_bounded_accumulate_constant(
                      op_constant, coefficient, &constant);
        break;
      }
      case LOOM_OP_INDEX_CAST:
        handled = loom_symbolic_expr_bounded_push_expand(
            frames, &frame_count, loom_index_cast_input(op), coefficient,
            next_depth);
        break;
      case LOOM_OP_INDEX_ASSUME:
        handled = loom_symbolic_expr_bounded_push_assume(
            loom_index_assume_values(op), value_id, loom_value_def_index(value),
            coefficient, next_depth, frames, &frame_count, raw_term_count,
            constant);
        break;
      case LOOM_OP_INDEX_ADD:
        handled = loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_index_add_rhs(op), coefficient,
                      next_depth) &&
                  loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_index_add_lhs(op), coefficient,
                      next_depth);
        break;
      case LOOM_OP_INDEX_SUB: {
        int64_t rhs_coefficient = 0;
        handled = loom_checked_sub_i64(0, coefficient, &rhs_coefficient) &&
                  loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_index_sub_rhs(op),
                      rhs_coefficient, next_depth) &&
                  loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_index_sub_lhs(op), coefficient,
                      next_depth);
        break;
      }
      case LOOM_OP_INDEX_MUL:
        handled = loom_symbolic_expr_bounded_push_mul(
            module, fact_table, value_id, loom_index_mul_lhs(op),
            loom_index_mul_rhs(op), coefficient, next_depth, frames,
            &frame_count, raw_terms, IREE_ARRAYSIZE(raw_terms),
            &raw_term_count);
        break;
      case LOOM_OP_INDEX_SCALE:
        handled = loom_symbolic_expr_bounded_push_mul(
            module, fact_table, value_id, loom_index_scale_index(op),
            loom_index_scale_stride(op), coefficient, next_depth, frames,
            &frame_count, raw_terms, IREE_ARRAYSIZE(raw_terms),
            &raw_term_count);
        break;
      case LOOM_OP_INDEX_MADD:
        handled = loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_index_madd_c(op), coefficient,
                      next_depth) &&
                  loom_symbolic_expr_bounded_push_mul(
                      module, fact_table, value_id, loom_index_madd_a(op),
                      loom_index_madd_b(op), coefficient, next_depth, frames,
                      &frame_count, raw_terms, IREE_ARRAYSIZE(raw_terms),
                      &raw_term_count);
        break;
      case LOOM_OP_INDEX_SHLI:
        handled = loom_symbolic_expr_bounded_push_shli(
            module, fact_table, value_id, loom_index_shli_lhs(op),
            loom_index_shli_rhs(op), coefficient, next_depth, frames,
            &frame_count, raw_terms, IREE_ARRAYSIZE(raw_terms),
            &raw_term_count);
        break;
      case LOOM_OP_SCALAR_ADDI:
        handled = loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_scalar_addi_rhs(op),
                      coefficient, next_depth) &&
                  loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_scalar_addi_lhs(op),
                      coefficient, next_depth);
        break;
      case LOOM_OP_SCALAR_SUBI: {
        int64_t rhs_coefficient = 0;
        handled = loom_checked_sub_i64(0, coefficient, &rhs_coefficient) &&
                  loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_scalar_subi_rhs(op),
                      rhs_coefficient, next_depth) &&
                  loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_scalar_subi_lhs(op),
                      coefficient, next_depth);
        break;
      }
      case LOOM_OP_SCALAR_MULI:
        handled = loom_symbolic_expr_bounded_push_mul(
            module, fact_table, value_id, loom_scalar_muli_lhs(op),
            loom_scalar_muli_rhs(op), coefficient, next_depth, frames,
            &frame_count, raw_terms, IREE_ARRAYSIZE(raw_terms),
            &raw_term_count);
        break;
      case LOOM_OP_SCALAR_NEGI: {
        int64_t negated_coefficient = 0;
        handled = loom_checked_sub_i64(0, coefficient, &negated_coefficient) &&
                  loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_scalar_negi_input(op),
                      negated_coefficient, next_depth);
        break;
      }
      case LOOM_OP_SCALAR_FMAI:
        handled = loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count, loom_scalar_fmai_c(op), coefficient,
                      next_depth) &&
                  loom_symbolic_expr_bounded_push_mul(
                      module, fact_table, value_id, loom_scalar_fmai_a(op),
                      loom_scalar_fmai_b(op), coefficient, next_depth, frames,
                      &frame_count, raw_terms, IREE_ARRAYSIZE(raw_terms),
                      &raw_term_count);
        break;
      case LOOM_OP_SCALAR_ASSUME:
        handled = loom_symbolic_expr_bounded_push_assume(
            loom_scalar_assume_values(op), value_id,
            loom_value_def_index(value), coefficient, next_depth, frames,
            &frame_count, raw_term_count, constant);
        break;
      case LOOM_OP_SCF_SELECT: {
        int64_t condition = 0;
        handled = loom_symbolic_expr_bounded_exact_i64(
                      module, fact_table, loom_scf_select_condition(op),
                      &condition) &&
                  loom_symbolic_expr_bounded_push_expand(
                      frames, &frame_count,
                      condition ? loom_scf_select_true_value(op)
                                : loom_scf_select_false_value(op),
                      coefficient, next_depth);
        break;
      }
      default:
        handled = false;
        break;
    }
    if (handled) continue;
    if (!loom_symbolic_expr_bounded_raw_append_term(
            raw_terms, IREE_ARRAYSIZE(raw_terms), &raw_term_count, value_id,
            value_id, coefficient)) {
      return false;
    }
  }

  if (!loom_symbolic_expr_bounded_accumulate_constant(constant, 1,
                                                      inout_constant)) {
    return false;
  }
  if (!loom_symbolic_expr_bounded_normalize_raw_terms(raw_terms,
                                                      &raw_term_count)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < raw_term_count; ++i) {
    if (!loom_symbolic_expr_bounded_append_term(
            terms, term_capacity, inout_term_count, raw_terms[i].value_id,
            raw_terms[i].relation_value_id, raw_terms[i].coefficient)) {
      return false;
    }
  }
  return true;
}

void loom_symbolic_expr_from_value_bounded(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_symbolic_term_t* terms,
    iree_host_size_t term_capacity, loom_symbolic_expr_t* out_expression) {
  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  const bool expanded = terms != NULL && term_capacity > 0 &&
                        loom_symbolic_expr_bounded_accumulate_value(
                            module, fact_table, value_id, 1,
                            LOOM_SYMBOLIC_EXPR_BOUNDED_DEPTH_LIMIT, terms,
                            term_capacity, &term_count, &constant);
  if (!expanded) {
    if (terms != NULL && term_capacity > 0) {
      terms[0] = (loom_symbolic_term_t){
          .coefficient = 1,
          .value_id = value_id,
          .relation_value_id = value_id,
      };
      term_count = 1;
    } else {
      loom_symbolic_expr_unknown(
          loom_symbolic_expr_bounded_lookup_facts(fact_table, value_id),
          out_expression);
      return;
    }
    constant = 0;
  }
  *out_expression = (loom_symbolic_expr_t){
      .constant = constant,
      .terms = term_count == 0 ? NULL : terms,
      .term_count = term_count,
      .facts = loom_symbolic_expr_bounded_lookup_facts(fact_table, value_id),
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
  };
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

static void loom_symbolic_expr_sort_terms(loom_symbolic_term_t* terms,
                                          iree_host_size_t term_count) {
  if (term_count <= 16) {
    for (iree_host_size_t i = 1; i < term_count; ++i) {
      const loom_symbolic_term_t term = terms[i];
      iree_host_size_t insertion_index = i;
      while (insertion_index > 0 &&
             terms[insertion_index - 1].value_id > term.value_id) {
        terms[insertion_index] = terms[insertion_index - 1];
        --insertion_index;
      }
      terms[insertion_index] = term;
    }
    return;
  }

  for (iree_host_size_t gap = term_count / 2; gap > 0; gap /= 2) {
    for (iree_host_size_t i = gap; i < term_count; ++i) {
      const loom_symbolic_term_t term = terms[i];
      iree_host_size_t insertion_index = i;
      while (insertion_index >= gap &&
             terms[insertion_index - gap].value_id > term.value_id) {
        terms[insertion_index] = terms[insertion_index - gap];
        insertion_index -= gap;
      }
      terms[insertion_index] = term;
    }
  }
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
      if (!loom_checked_add_i64(coefficient, terms[read_index].coefficient,
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
  loom_value_facts_t facts = loom_symbolic_expr_lookup_facts(context, value_id);
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
  loom_value_facts_t facts = loom_symbolic_expr_lookup_facts(context, value_id);
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
      subtract ? loom_checked_sub_i64(left_expression->constant,
                                      right_expression->constant, &constant)
               : loom_checked_add_i64(left_expression->constant,
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
  if (!loom_checked_mul_i64(expression->constant, multiplier, &constant)) {
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
    if (!loom_checked_mul_i64(expression->terms[i].coefficient, multiplier,
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
  loom_value_facts_t facts = loom_symbolic_expr_lookup_facts(context, value_id);
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

//===----------------------------------------------------------------------===//
// Proofs
//===----------------------------------------------------------------------===//

static loom_symbolic_proof_result_t loom_symbolic_expr_prove_le_by_facts(
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression) {
  if (loom_value_facts_is_float(left_expression->facts) ||
      loom_value_facts_is_float(right_expression->facts)) {
    return LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  if (left_expression->facts.range_hi <= right_expression->facts.range_lo) {
    return LOOM_SYMBOLIC_PROOF_TRUE;
  }
  if (left_expression->facts.range_lo > right_expression->facts.range_hi) {
    return LOOM_SYMBOLIC_PROOF_FALSE;
  }
  return LOOM_SYMBOLIC_PROOF_UNKNOWN;
}

static bool loom_symbolic_expr_accumulate_checked(int64_t term_min,
                                                  int64_t term_max,
                                                  int64_t* inout_min,
                                                  int64_t* inout_max) {
  int64_t new_min = 0;
  int64_t new_max = 0;
  if (!loom_checked_add_i64(*inout_min, term_min, &new_min)) return false;
  if (!loom_checked_add_i64(*inout_max, term_max, &new_max)) return false;
  *inout_min = new_min;
  *inout_max = new_max;
  return true;
}

static loom_value_facts_t loom_symbolic_expr_intersect_integer_facts(
    loom_value_facts_t lhs, loom_value_facts_t rhs) {
  if (loom_value_facts_is_unknown(lhs) || loom_value_facts_is_float(lhs)) {
    return rhs;
  }
  if (loom_value_facts_is_unknown(rhs) || loom_value_facts_is_float(rhs)) {
    return lhs;
  }
  int64_t lower_bound = loom_max_i64(lhs.range_lo, rhs.range_lo);
  int64_t upper_bound = loom_min_i64(lhs.range_hi, rhs.range_hi);
  if (lower_bound > upper_bound) {
    return loom_value_facts_unknown();
  }
  return loom_value_facts_make(lower_bound, upper_bound, 1);
}

static iree_status_t loom_symbolic_expr_term_facts(
    loom_symbolic_expr_context_t* context, const loom_symbolic_term_t term,
    loom_value_facts_t* out_facts) {
  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_lookup_condition_refined_facts(
      context, term.value_id,
      LOOM_SYMBOLIC_EXPR_CONDITION_FACT_INFER_DEPTH_LIMIT, &facts));
  if (term.relation_value_id == LOOM_VALUE_ID_INVALID ||
      term.relation_value_id == term.value_id) {
    *out_facts = facts;
    return iree_ok_status();
  }
  loom_value_facts_t relation_facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_lookup_condition_refined_facts(
      context, term.relation_value_id,
      LOOM_SYMBOLIC_EXPR_CONDITION_FACT_INFER_DEPTH_LIMIT, &relation_facts));
  facts = loom_symbolic_expr_intersect_integer_facts(facts, relation_facts);
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_apply_identity_chain_predicates_to_value_facts(
          context, term.relation_value_id, &facts));
  *out_facts = facts;
  return iree_ok_status();
}

static void loom_symbolic_expr_mul_interval_bound(int64_t lhs, int64_t rhs,
                                                  int64_t* out_product) {
  if (loom_checked_mul_i64(lhs, rhs, out_product)) {
    return;
  }
  *out_product = (lhs < 0) != (rhs < 0) ? INT64_MIN : INT64_MAX;
}

static iree_status_t loom_symbolic_expr_term_interval(
    loom_symbolic_expr_context_t* context, const loom_symbolic_term_t term,
    int64_t* out_min, int64_t* out_max, bool* out_known) {
  *out_known = false;
  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_term_facts(context, term, &facts));
  if (loom_value_facts_is_unknown(facts) || loom_value_facts_is_float(facts)) {
    return iree_ok_status();
  }
  int64_t lower_product = 0;
  int64_t upper_product = 0;
  if (term.coefficient >= 0) {
    loom_symbolic_expr_mul_interval_bound(term.coefficient, facts.range_lo,
                                          &lower_product);
    loom_symbolic_expr_mul_interval_bound(term.coefficient, facts.range_hi,
                                          &upper_product);
  } else {
    loom_symbolic_expr_mul_interval_bound(term.coefficient, facts.range_hi,
                                          &lower_product);
    loom_symbolic_expr_mul_interval_bound(term.coefficient, facts.range_lo,
                                          &upper_product);
  }
  *out_min = lower_product;
  *out_max = upper_product;
  *out_known = true;
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_normalize_difference_into_scratch(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, int64_t* out_constant,
    iree_host_size_t* out_term_count, bool* out_linear) {
  *out_constant = 0;
  *out_term_count = 0;
  *out_linear = false;
  if (!loom_symbolic_expr_is_linear(left_expression) ||
      !loom_symbolic_expr_is_linear(right_expression)) {
    return iree_ok_status();
  }

  if (!loom_checked_sub_i64(left_expression->constant,
                            right_expression->constant, out_constant)) {
    return iree_ok_status();
  }

  iree_host_size_t term_count = 0;
  if (!loom_symbolic_expr_checked_term_count(left_expression->term_count,
                                             right_expression->term_count,
                                             &term_count)) {
    return iree_ok_status();
  }
  if (term_count > context->maximum_term_count) {
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
    if (coefficient == INT64_MIN) {
      return iree_ok_status();
    }
    context->scratch_terms[term_ordinal++] = (loom_symbolic_term_t){
        .coefficient = -coefficient,
        .value_id = right_expression->terms[i].value_id,
        .relation_value_id = right_expression->terms[i].relation_value_id,
    };
  }
  if (term_ordinal > 1) {
    loom_symbolic_expr_sort_terms(context->scratch_terms, term_ordinal);
  }

  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < term_ordinal;) {
    loom_value_id_t value_id = context->scratch_terms[read_index].value_id;
    loom_value_id_t relation_value_id =
        context->scratch_terms[read_index].relation_value_id;
    if (relation_value_id == LOOM_VALUE_ID_INVALID) {
      relation_value_id = value_id;
    }
    int64_t coefficient = 0;
    while (read_index < term_ordinal &&
           context->scratch_terms[read_index].value_id == value_id) {
      loom_value_id_t term_relation_value_id =
          context->scratch_terms[read_index].relation_value_id;
      if (term_relation_value_id == LOOM_VALUE_ID_INVALID) {
        term_relation_value_id = value_id;
      }
      if (term_relation_value_id != relation_value_id) {
        relation_value_id = value_id;
      }
      int64_t new_coefficient = 0;
      if (!loom_checked_add_i64(coefficient,
                                context->scratch_terms[read_index].coefficient,
                                &new_coefficient)) {
        return iree_ok_status();
      }
      coefficient = new_coefficient;
      ++read_index;
    }
    if (coefficient == 0) continue;
    context->scratch_terms[write_index++] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = value_id,
        .relation_value_id = relation_value_id,
    };
  }
  *out_term_count = write_index;
  *out_linear = true;
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_le_linear(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  bool linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, left_expression, right_expression, &constant, &term_count,
      &linear));
  if (!linear) {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    return iree_ok_status();
  }

  if (term_count == 0) {
    *out_result =
        constant <= 0 ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
    return iree_ok_status();
  }

  loom_symbolic_term_t stack_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
  const loom_symbolic_term_t* terms = NULL;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_stabilize_scratch_terms(
      context, term_count, stack_terms, IREE_ARRAYSIZE(stack_terms), &terms));

  int64_t minimum = constant;
  int64_t maximum = constant;
  for (iree_host_size_t i = 0; i < term_count; ++i) {
    int64_t term_minimum = 0;
    int64_t term_maximum = 0;
    bool term_interval_known = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_term_interval(
        context, terms[i], &term_minimum, &term_maximum, &term_interval_known));
    if (!term_interval_known ||
        !loom_symbolic_expr_accumulate_checked(term_minimum, term_maximum,
                                               &minimum, &maximum)) {
      *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      return iree_ok_status();
    }
  }
  if (maximum <= 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
  } else if (minimum > 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
  } else {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  return iree_ok_status();
}

iree_status_t loom_symbolic_expr_simplify_value_difference(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value,
    loom_symbolic_value_difference_t* out_difference) {
  *out_difference = (loom_symbolic_value_difference_t){
      .kind = LOOM_SYMBOLIC_VALUE_DIFFERENCE_UNKNOWN,
      .constant = 0,
      .value_id = LOOM_VALUE_ID_INVALID,
  };
  loom_symbolic_expr_t left_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));

  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  bool linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, &left_expression, &right_expression, &constant, &term_count,
      &linear));
  if (!linear) return iree_ok_status();

  if (term_count == 0) {
    *out_difference = (loom_symbolic_value_difference_t){
        .kind = LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT,
        .constant = constant,
        .value_id = LOOM_VALUE_ID_INVALID,
    };
    return iree_ok_status();
  }
  if (constant == 0 && term_count == 1 &&
      context->scratch_terms[0].coefficient == 1) {
    *out_difference = (loom_symbolic_value_difference_t){
        .kind = LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE,
        .constant = 0,
        .value_id = context->scratch_terms[0].value_id,
    };
  }
  return iree_ok_status();
}

static bool loom_symbolic_expr_predicate_relation(
    const loom_predicate_t* predicate,
    loom_symbolic_integer_relation_t* out_relation) {
  switch ((loom_predicate_kind_t)predicate->kind) {
    case LOOM_PREDICATE_EQ:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_PREDICATE_NE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_PREDICATE_LT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_PREDICATE_LE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_PREDICATE_GT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_PREDICATE_GE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static bool loom_symbolic_expr_relation_predicate_kind(
    loom_symbolic_integer_relation_t relation, uint8_t* out_kind) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      *out_kind = LOOM_PREDICATE_EQ;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
      *out_kind = LOOM_PREDICATE_NE;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      *out_kind = LOOM_PREDICATE_LT;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      *out_kind = LOOM_PREDICATE_LE;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      *out_kind = LOOM_PREDICATE_GT;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      *out_kind = LOOM_PREDICATE_GE;
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_symbolic_expr_predicate_arg_exact_integer(
    loom_symbolic_expr_context_t* context, loom_predicate_arg_tag_t arg_tag,
    int64_t arg, int64_t* out_value, bool* out_known) {
  *out_value = 0;
  *out_known = false;
  switch (arg_tag) {
    case LOOM_PRED_ARG_CONST:
      *out_value = arg;
      *out_known = true;
      return iree_ok_status();
    case LOOM_PRED_ARG_VALUE: {
      if (arg < 0) return iree_ok_status();
      loom_symbolic_expr_t expression = {0};
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
          context, (loom_value_id_t)arg, &expression));
      *out_known = loom_symbolic_expr_constant_value(&expression, out_value);
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_symbolic_expr_predicate_apply_to_value_facts(
    loom_symbolic_expr_context_t* context, const loom_predicate_t* predicate,
    loom_value_id_t value_id, loom_value_facts_t* inout_facts) {
  if (predicate->arg_count != 2) return iree_ok_status();

  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!loom_symbolic_expr_predicate_relation(predicate, &relation)) {
    return iree_ok_status();
  }

  bool value_is_left = false;
  if (predicate->arg_tags[0] == LOOM_PRED_ARG_VALUE &&
      predicate->args[0] >= 0) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
        context, (loom_value_id_t)predicate->args[0], value_id,
        &value_is_left));
  }
  bool value_is_right = false;
  if (predicate->arg_tags[1] == LOOM_PRED_ARG_VALUE &&
      predicate->args[1] >= 0) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
        context, (loom_value_id_t)predicate->args[1], value_id,
        &value_is_right));
  }
  if (!value_is_left && !value_is_right) return iree_ok_status();

  uint8_t other_arg_index = value_is_left ? 1 : 0;
  int64_t other_value = 0;
  bool other_known = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_arg_exact_integer(
      context, (loom_predicate_arg_tag_t)predicate->arg_tags[other_arg_index],
      predicate->args[other_arg_index], &other_value, &other_known));
  if (!other_known) return iree_ok_status();

  if (value_is_right) {
    relation = loom_symbolic_integer_relation_swap(relation);
  }
  uint8_t predicate_kind = 0;
  if (!loom_symbolic_expr_relation_predicate_kind(relation, &predicate_kind)) {
    return iree_ok_status();
  }

  loom_predicate_t normalized = {
      .kind = predicate_kind,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_NONE},
      .args = {value_id, other_value, 0},
  };
  loom_value_facts_apply_predicate(inout_facts, &normalized);
  return iree_ok_status();
}

static iree_status_t
loom_symbolic_expr_apply_identity_chain_predicates_to_value_facts(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_facts_t* inout_facts) {
  if (!context->module || start_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_value_id_t current_value = start_value;
  uint8_t remaining_steps = LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0) {
    if (current_value >= context->module->values.count) {
      return iree_ok_status();
    }
    const loom_value_t* value =
        loom_module_value(context->module, current_value);
    if (loom_value_is_block_arg(value)) return iree_ok_status();
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) return iree_ok_status();

    if (loom_index_cast_isa(defining_op)) {
      current_value = loom_index_cast_input(defining_op);
      continue;
    }

    loom_value_slice_t values = {.values = NULL, .count = 0};
    if (loom_index_assume_isa(defining_op)) {
      values = loom_index_assume_values(defining_op);
    } else if (loom_scalar_assume_isa(defining_op)) {
      values = loom_scalar_assume_values(defining_op);
    } else {
      return iree_ok_status();
    }

    uint16_t result_index = loom_value_def_index(value);
    if (result_index >= values.count) return iree_ok_status();
    if (defining_op->attribute_count > 0) {
      loom_attribute_t predicates_attr = loom_op_attrs(defining_op)[0];
      if (predicates_attr.kind == LOOM_ATTR_PREDICATE_LIST &&
          (predicates_attr.count == 0 || predicates_attr.predicate_list)) {
        for (uint16_t i = 0; i < predicates_attr.count; ++i) {
          IREE_RETURN_IF_ERROR(
              loom_symbolic_expr_predicate_apply_to_value_facts(
                  context, &predicates_attr.predicate_list[i], current_value,
                  inout_facts));
        }
      }
    }
    current_value = values.values[result_index];
  }
  return iree_ok_status();
}

static bool loom_symbolic_expr_value_is_integer_domain(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  if (!context->module || value_id >= context->module->values.count) {
    return false;
  }
  loom_type_t type = loom_module_value_type(context->module, value_id);
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
         loom_scalar_type_is_integer(scalar_type);
}

static iree_status_t loom_symbolic_expr_values_match(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value, bool* out_match) {
  *out_match = false;
  if (left_value == right_value) {
    *out_match = true;
    return iree_ok_status();
  }

  int64_t left_exact = 0;
  int64_t right_exact = 0;
  if (loom_symbolic_expr_exact_integer_facts(
          loom_symbolic_expr_lookup_facts(context, left_value), &left_exact) &&
      loom_symbolic_expr_exact_integer_facts(
          loom_symbolic_expr_lookup_facts(context, right_value),
          &right_exact)) {
    *out_match = left_exact == right_exact;
    return iree_ok_status();
  }

  loom_symbolic_value_difference_t difference = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_simplify_value_difference(
      context, left_value, right_value, &difference));
  *out_match = difference.kind == LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT &&
               difference.constant == 0;
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_value_matches_constant(
    loom_symbolic_expr_context_t* context, loom_value_id_t value,
    int64_t constant, bool* out_match) {
  *out_match = false;
  loom_symbolic_expr_t expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, value, &expression));
  int64_t expression_constant = 0;
  *out_match =
      loom_symbolic_expr_constant_value(&expression, &expression_constant) &&
      expression_constant == constant;
  return iree_ok_status();
}

static loom_value_id_t loom_symbolic_expr_assumption_source_value(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  if (!context->module) return value_id;
  uint8_t remaining_steps = LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0) {
    if (value_id >= context->module->values.count) return value_id;
    const loom_op_t* defining_op =
        loom_symbolic_expr_value_defining_op(context, value_id);
    if (!defining_op) return value_id;
    loom_value_slice_t values = {.values = NULL, .count = 0};
    if (loom_index_assume_isa(defining_op)) {
      values = loom_index_assume_values(defining_op);
    } else if (loom_scalar_assume_isa(defining_op)) {
      values = loom_scalar_assume_values(defining_op);
    } else {
      return value_id;
    }
    const loom_value_t* value = loom_module_value(context->module, value_id);
    uint16_t result_index = loom_value_def_index(value);
    if (result_index >= values.count) return value_id;
    value_id = values.values[result_index];
  }
  return value_id;
}

typedef enum loom_symbolic_kernel_bound_kind_e {
  LOOM_SYMBOLIC_KERNEL_BOUND_WORKGROUP_COUNT = 0,
  LOOM_SYMBOLIC_KERNEL_BOUND_WORKGROUP_SIZE = 1,
} loom_symbolic_kernel_bound_kind_t;

static const loom_op_t* loom_symbolic_expr_value_defining_op(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  if (!context->module || value_id >= context->module->values.count) {
    return NULL;
  }
  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return NULL;
  }
  return loom_value_def_op(value);
}

static bool loom_symbolic_expr_kernel_coordinate_bound(
    const loom_op_t* op, loom_symbolic_kernel_bound_kind_t* out_kind,
    loom_kernel_dimension_t* out_dimension) {
  if (loom_kernel_workgroup_id_isa(op)) {
    *out_kind = LOOM_SYMBOLIC_KERNEL_BOUND_WORKGROUP_COUNT;
    *out_dimension = loom_kernel_workgroup_id_dimension(op);
    return true;
  }
  if (loom_kernel_workitem_id_isa(op)) {
    *out_kind = LOOM_SYMBOLIC_KERNEL_BOUND_WORKGROUP_SIZE;
    *out_dimension = loom_kernel_workitem_id_dimension(op);
    return true;
  }
  return false;
}

static bool loom_symbolic_expr_kernel_query_matches_bound(
    const loom_op_t* op, loom_symbolic_kernel_bound_kind_t kind,
    loom_kernel_dimension_t dimension) {
  switch (kind) {
    case LOOM_SYMBOLIC_KERNEL_BOUND_WORKGROUP_COUNT:
      return loom_kernel_workgroup_count_isa(op) &&
             loom_kernel_workgroup_count_dimension(op) == dimension;
    case LOOM_SYMBOLIC_KERNEL_BOUND_WORKGROUP_SIZE:
      return loom_kernel_workgroup_size_isa(op) &&
             loom_kernel_workgroup_size_dimension(op) == dimension;
    default:
      return false;
  }
}

static loom_value_id_t loom_symbolic_expr_kernel_launch_bound_operand(
    const loom_op_t* launch_config, loom_symbolic_kernel_bound_kind_t kind,
    loom_kernel_dimension_t dimension) {
  switch (kind) {
    case LOOM_SYMBOLIC_KERNEL_BOUND_WORKGROUP_COUNT:
      return loom_kernel_launch_config_workgroup_count_operand(launch_config,
                                                               dimension);
    case LOOM_SYMBOLIC_KERNEL_BOUND_WORKGROUP_SIZE:
      return loom_kernel_launch_config_workgroup_size_operand(launch_config,
                                                              dimension);
    default:
      return LOOM_VALUE_ID_INVALID;
  }
}

static iree_status_t loom_symbolic_expr_value_matches_kernel_bound(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_kernel_bound_kind_t kind, loom_kernel_dimension_t dimension,
    bool* out_match) {
  *out_match = false;
  const loom_op_t* defining_op =
      loom_symbolic_expr_value_defining_op(context, value_id);
  if (defining_op && loom_symbolic_expr_kernel_query_matches_bound(
                         defining_op, kind, dimension)) {
    *out_match = true;
    return iree_ok_status();
  }

  if (!context->fact_table) {
    return iree_ok_status();
  }
  loom_func_like_t function = context->fact_table->context.function;
  if (!loom_kernel_def_isa(function.op)) {
    return iree_ok_status();
  }
  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(function.op);
  const loom_value_id_t bound_value =
      loom_symbolic_expr_kernel_launch_bound_operand(launch_config, kind,
                                                     dimension);
  if (bound_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_symbolic_expr_values_match(context, value_id, bound_value,
                                         out_match);
}

static iree_status_t loom_symbolic_expr_kernel_coordinate_proves_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  *out_matched = false;

  const loom_op_t* left_op = loom_symbolic_expr_value_defining_op(
      context, loom_symbolic_expr_assumption_source_value(context, left_value));
  loom_symbolic_kernel_bound_kind_t left_kind = 0;
  loom_kernel_dimension_t left_dimension = 0;
  if (left_op && loom_symbolic_expr_kernel_coordinate_bound(left_op, &left_kind,
                                                            &left_dimension)) {
    bool right_matches_bound = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_value_matches_kernel_bound(
        context, right_value, left_kind, left_dimension, &right_matches_bound));
    if (right_matches_bound) {
      bool result = false;
      *out_matched = loom_symbolic_integer_relation_implies(
          LOOM_SYMBOLIC_INTEGER_RELATION_LT, relation, &result);
      if (*out_matched) {
        *out_result =
            result ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
      }
      return iree_ok_status();
    }
  }

  const loom_op_t* right_op = loom_symbolic_expr_value_defining_op(
      context,
      loom_symbolic_expr_assumption_source_value(context, right_value));
  loom_symbolic_kernel_bound_kind_t right_kind = 0;
  loom_kernel_dimension_t right_dimension = 0;
  if (right_op && loom_symbolic_expr_kernel_coordinate_bound(
                      right_op, &right_kind, &right_dimension)) {
    bool left_matches_bound = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_value_matches_kernel_bound(
        context, left_value, right_kind, right_dimension, &left_matches_bound));
    if (left_matches_bound) {
      bool result = false;
      *out_matched = loom_symbolic_integer_relation_implies(
          LOOM_SYMBOLIC_INTEGER_RELATION_GT, relation, &result);
      if (*out_matched) {
        *out_result =
            result ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
      }
    }
  }
  return iree_ok_status();
}

static bool loom_symbolic_expr_unsigned_quotient_def(
    const loom_op_t* op, loom_value_id_t* out_dividend,
    loom_value_id_t* out_divisor) {
  if (loom_index_div_isa(op)) {
    *out_dividend = loom_index_div_lhs(op);
    *out_divisor = loom_index_div_rhs(op);
    return true;
  }
  if (loom_scalar_divui_isa(op)) {
    *out_dividend = loom_scalar_divui_lhs(op);
    *out_divisor = loom_scalar_divui_rhs(op);
    return true;
  }
  return false;
}

static bool loom_symbolic_expr_value_facts_are_non_negative(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  return loom_value_facts_is_non_negative(
      loom_symbolic_expr_lookup_facts(context, value_id));
}

static bool loom_symbolic_expr_value_facts_are_positive(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  return loom_value_facts_is_positive(
      loom_symbolic_expr_lookup_facts(context, value_id));
}

static bool loom_symbolic_expr_multiply_def(const loom_op_t* op,
                                            loom_value_id_t* out_lhs,
                                            loom_value_id_t* out_rhs) {
  if (loom_index_mul_isa(op)) {
    *out_lhs = loom_index_mul_lhs(op);
    *out_rhs = loom_index_mul_rhs(op);
    return true;
  }
  if (loom_scalar_muli_isa(op)) {
    *out_lhs = loom_scalar_muli_lhs(op);
    *out_rhs = loom_scalar_muli_rhs(op);
    return true;
  }
  return false;
}

static iree_status_t loom_symbolic_expr_value_matches_product(
    loom_symbolic_expr_context_t* context, loom_value_id_t product_value,
    loom_value_id_t left_factor, loom_value_id_t right_factor,
    bool* out_match) {
  *out_match = false;
  loom_value_id_t product_source =
      loom_symbolic_expr_assumption_source_value(context, product_value);
  const loom_op_t* product_op =
      loom_symbolic_expr_value_defining_op(context, product_source);
  loom_value_id_t product_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t product_rhs = LOOM_VALUE_ID_INVALID;
  if (!product_op || !loom_symbolic_expr_multiply_def(product_op, &product_lhs,
                                                      &product_rhs)) {
    return iree_ok_status();
  }

  bool lhs_matches_left = false;
  bool rhs_matches_right = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
      context, product_lhs, left_factor, &lhs_matches_left));
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
      context, product_rhs, right_factor, &rhs_matches_right));
  if (lhs_matches_left && rhs_matches_right) {
    *out_match = true;
    return iree_ok_status();
  }

  bool lhs_matches_right = false;
  bool rhs_matches_left = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
      context, product_lhs, right_factor, &lhs_matches_right));
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
      context, product_rhs, left_factor, &rhs_matches_left));
  *out_match = lhs_matches_right && rhs_matches_left;
  return iree_ok_status();
}

static bool loom_symbolic_expr_kernel_coordinate_launch_bound_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_id_t* out_bound_value) {
  *out_bound_value = LOOM_VALUE_ID_INVALID;
  const loom_op_t* op = loom_symbolic_expr_value_defining_op(
      context, loom_symbolic_expr_assumption_source_value(context, value_id));
  loom_symbolic_kernel_bound_kind_t kind = 0;
  loom_kernel_dimension_t dimension = 0;
  if (!op ||
      !loom_symbolic_expr_kernel_coordinate_bound(op, &kind, &dimension)) {
    return false;
  }
  if (!context->fact_table) return false;
  loom_func_like_t function = context->fact_table->context.function;
  if (!loom_kernel_def_isa(function.op)) return false;
  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(function.op);
  *out_bound_value = loom_symbolic_expr_kernel_launch_bound_operand(
      launch_config, kind, dimension);
  return *out_bound_value != LOOM_VALUE_ID_INVALID;
}

static bool loom_symbolic_expr_quotient_bound_relation(
    loom_symbolic_integer_relation_t relation,
    loom_symbolic_proof_result_t product_relation_result,
    loom_symbolic_proof_result_t* out_result) {
  loom_symbolic_integer_relation_t implied_relation =
      LOOM_SYMBOLIC_INTEGER_RELATION_LT;
  if (product_relation_result == LOOM_SYMBOLIC_PROOF_FALSE) {
    implied_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
  } else if (product_relation_result != LOOM_SYMBOLIC_PROOF_TRUE) {
    return false;
  }

  bool result = false;
  if (!loom_symbolic_integer_relation_implies(implied_relation, relation,
                                              &result)) {
    return false;
  }
  *out_result = result ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
  return true;
}

static iree_status_t loom_symbolic_expr_quotient_launch_bound_proves_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t dividend,
    loom_value_id_t divisor, loom_value_id_t bound_value, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  *out_matched = false;
  if (!loom_symbolic_expr_value_facts_are_non_negative(context, dividend) ||
      !loom_symbolic_expr_value_facts_are_positive(context, divisor) ||
      !loom_symbolic_expr_value_facts_are_non_negative(context, bound_value)) {
    return iree_ok_status();
  }

  loom_value_id_t product_bound = LOOM_VALUE_ID_INVALID;
  if (!loom_symbolic_expr_kernel_coordinate_launch_bound_value(
          context, dividend, &product_bound)) {
    return iree_ok_status();
  }

  bool product_matches = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_value_matches_product(
      context, product_bound, divisor, bound_value, &product_matches));
  if (!product_matches) return iree_ok_status();

  loom_symbolic_proof_result_t product_relation = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  bool product_relation_matched = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_kernel_coordinate_proves_relation(
      context, LOOM_SYMBOLIC_INTEGER_RELATION_LT, dividend, product_bound,
      &product_relation_matched, &product_relation));
  if (!product_relation_matched) return iree_ok_status();
  if (loom_symbolic_expr_quotient_bound_relation(relation, product_relation,
                                                 out_result)) {
    *out_matched = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_quotient_bound_proves_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t quotient_value,
    loom_value_id_t bound_value, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  *out_matched = false;
  loom_value_id_t quotient_source =
      loom_symbolic_expr_assumption_source_value(context, quotient_value);
  const loom_op_t* quotient_op =
      loom_symbolic_expr_value_defining_op(context, quotient_source);
  loom_value_id_t dividend = LOOM_VALUE_ID_INVALID;
  loom_value_id_t divisor = LOOM_VALUE_ID_INVALID;
  if (!quotient_op || !loom_symbolic_expr_unsigned_quotient_def(
                          quotient_op, &dividend, &divisor)) {
    return iree_ok_status();
  }

  return loom_symbolic_expr_quotient_launch_bound_proves_relation(
      context, relation, dividend, divisor, bound_value, out_matched,
      out_result);
}

static bool loom_symbolic_expr_unsigned_remainder_def(
    const loom_op_t* op, loom_value_id_t* out_dividend,
    loom_value_id_t* out_divisor) {
  if (loom_index_rem_isa(op)) {
    *out_dividend = loom_index_rem_lhs(op);
    *out_divisor = loom_index_rem_rhs(op);
    return true;
  }
  if (loom_scalar_remui_isa(op)) {
    *out_dividend = loom_scalar_remui_lhs(op);
    *out_divisor = loom_scalar_remui_rhs(op);
    return true;
  }
  return false;
}

static bool loom_symbolic_expr_remainder_bound_relation(
    loom_symbolic_integer_relation_t relation, bool swapped,
    loom_symbolic_proof_result_t* out_result) {
  bool result = false;
  loom_symbolic_integer_relation_t implied_relation =
      swapped ? LOOM_SYMBOLIC_INTEGER_RELATION_GT
              : LOOM_SYMBOLIC_INTEGER_RELATION_LT;
  if (!loom_symbolic_integer_relation_implies(implied_relation, relation,
                                              &result)) {
    return false;
  }
  *out_result = result ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
  return true;
}

static iree_status_t loom_symbolic_expr_remainder_bound_proves_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  *out_matched = false;
  loom_value_id_t left_source =
      loom_symbolic_expr_assumption_source_value(context, left_value);
  const loom_op_t* left_op =
      loom_symbolic_expr_value_defining_op(context, left_source);
  loom_value_id_t dividend = LOOM_VALUE_ID_INVALID;
  loom_value_id_t divisor = LOOM_VALUE_ID_INVALID;
  if (left_op &&
      loom_symbolic_expr_unsigned_remainder_def(left_op, &dividend, &divisor)) {
    bool right_matches_divisor = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
        context, right_value, divisor, &right_matches_divisor));
    if (right_matches_divisor &&
        loom_symbolic_expr_value_facts_are_non_negative(context, dividend) &&
        loom_symbolic_expr_value_facts_are_positive(context, divisor)) {
      *out_matched = loom_symbolic_expr_remainder_bound_relation(
          relation, /*swapped=*/false, out_result);
      return iree_ok_status();
    }
  }

  loom_value_id_t right_source =
      loom_symbolic_expr_assumption_source_value(context, right_value);
  const loom_op_t* right_op =
      loom_symbolic_expr_value_defining_op(context, right_source);
  if (right_op && loom_symbolic_expr_unsigned_remainder_def(right_op, &dividend,
                                                            &divisor)) {
    bool left_matches_divisor = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
        context, left_value, divisor, &left_matches_divisor));
    if (left_matches_divisor &&
        loom_symbolic_expr_value_facts_are_non_negative(context, dividend) &&
        loom_symbolic_expr_value_facts_are_positive(context, divisor)) {
      *out_matched = loom_symbolic_expr_remainder_bound_relation(
          relation, /*swapped=*/true, out_result);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_predicate_proves_relation(
    loom_symbolic_expr_context_t* context, const loom_predicate_t* predicate,
    loom_symbolic_integer_relation_t queried_relation,
    loom_value_id_t left_value, loom_value_id_t right_value, bool* out_matched,
    bool* out_result) {
  *out_matched = false;
  *out_result = false;
  if (predicate->arg_count != 2 ||
      predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) {
    return iree_ok_status();
  }

  loom_value_id_t predicate_left = (loom_value_id_t)predicate->args[0];
  loom_symbolic_integer_relation_t implied_relation =
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!loom_symbolic_expr_predicate_relation(predicate, &implied_relation)) {
    return iree_ok_status();
  }

  bool ordered_left_match = false;
  bool swapped_right_match = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
      context, predicate_left, left_value, &ordered_left_match));
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
      context, predicate_left, right_value, &swapped_right_match));

  bool ordered_right_match = false;
  bool swapped_left_match = false;
  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[1]) {
    case LOOM_PRED_ARG_VALUE: {
      loom_value_id_t predicate_right = (loom_value_id_t)predicate->args[1];
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
          context, predicate_right, right_value, &ordered_right_match));
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_values_match(
          context, predicate_right, left_value, &swapped_left_match));
      break;
    }
    case LOOM_PRED_ARG_CONST: {
      int64_t constant = predicate->args[1];
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_value_matches_constant(
          context, right_value, constant, &ordered_right_match));
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_value_matches_constant(
          context, left_value, constant, &swapped_left_match));
      break;
    }
    default:
      return iree_ok_status();
  }
  if (ordered_left_match && ordered_right_match) {
    *out_matched = loom_symbolic_integer_relation_implies(
        implied_relation, queried_relation, out_result);
    return iree_ok_status();
  }
  if (swapped_right_match && swapped_left_match) {
    *out_matched = loom_symbolic_integer_relation_implies(
        loom_symbolic_integer_relation_swap(implied_relation), queried_relation,
        out_result);
    return iree_ok_status();
  }
  return iree_ok_status();
}

typedef iree_status_t (*loom_symbolic_expr_predicate_proof_fn_t)(
    loom_symbolic_expr_context_t* context, const loom_predicate_t* predicate,
    const void* user_data, bool* out_matched,
    loom_symbolic_proof_result_t* out_result);

typedef struct loom_symbolic_expr_relation_proof_t {
  loom_symbolic_integer_relation_t relation;
  loom_value_id_t left_value;
  loom_value_id_t right_value;
} loom_symbolic_expr_relation_proof_t;

static iree_status_t loom_symbolic_expr_relation_predicate_proof(
    loom_symbolic_expr_context_t* context, const loom_predicate_t* predicate,
    const void* user_data, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  const loom_symbolic_expr_relation_proof_t* proof =
      (const loom_symbolic_expr_relation_proof_t*)user_data;
  bool relation_result = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_proves_relation(
      context, predicate, proof->relation, proof->left_value,
      proof->right_value, out_matched, &relation_result));
  if (*out_matched) {
    *out_result =
        relation_result ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_predicate_arg_matches_value(
    loom_symbolic_expr_context_t* context, loom_predicate_arg_tag_t arg_tag,
    int64_t arg, loom_value_id_t value_id, bool* out_match) {
  *out_match = false;
  switch (arg_tag) {
    case LOOM_PRED_ARG_VALUE:
      if (arg < 0) return iree_ok_status();
      return loom_symbolic_expr_values_match(context, (loom_value_id_t)arg,
                                             value_id, out_match);
    case LOOM_PRED_ARG_CONST:
      return loom_symbolic_expr_value_matches_constant(context, value_id, arg,
                                                       out_match);
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_symbolic_expr_predicate_arg_difference_from_value(
    loom_symbolic_expr_context_t* context, loom_predicate_arg_tag_t arg_tag,
    int64_t arg, loom_value_id_t value_id, int64_t* out_difference,
    bool* out_known) {
  *out_difference = 0;
  *out_known = false;
  switch (arg_tag) {
    case LOOM_PRED_ARG_VALUE: {
      if (arg < 0) return iree_ok_status();
      loom_symbolic_value_difference_t difference = {0};
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_simplify_value_difference(
          context, (loom_value_id_t)arg, value_id, &difference));
      if (difference.kind == LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT) {
        *out_difference = difference.constant;
        *out_known = true;
      }
      return iree_ok_status();
    }
    case LOOM_PRED_ARG_CONST: {
      loom_symbolic_expr_t value_expression = {0};
      IREE_RETURN_IF_ERROR(
          loom_symbolic_expr_from_value(context, value_id, &value_expression));
      int64_t value_constant = 0;
      if (loom_symbolic_expr_constant_value(&value_expression,
                                            &value_constant) &&
          loom_checked_sub_i64(arg, value_constant, out_difference)) {
        *out_known = true;
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static bool loom_symbolic_expr_scaled_upper_bound_proves_le(
    int64_t scale, int64_t constant,
    loom_symbolic_integer_relation_t upper_relation,
    int64_t upper_minus_negative) {
  if (upper_relation != LOOM_SYMBOLIC_INTEGER_RELATION_LT &&
      upper_relation != LOOM_SYMBOLIC_INTEGER_RELATION_LE) {
    return false;
  }
  int64_t effective_upper_delta = upper_minus_negative;
  if (upper_relation == LOOM_SYMBOLIC_INTEGER_RELATION_LT &&
      !loom_checked_sub_i64(effective_upper_delta, 1, &effective_upper_delta)) {
    return false;
  }
  int64_t scaled_upper_delta = 0;
  if (!loom_checked_mul_i64(scale, effective_upper_delta,
                            &scaled_upper_delta)) {
    return false;
  }
  int64_t maximum_difference = 0;
  return loom_checked_add_i64(constant, scaled_upper_delta,
                              &maximum_difference) &&
         maximum_difference <= 0;
}

static bool loom_symbolic_expr_scaled_static_upper_bound_proves_le(
    int64_t scale, int64_t constant,
    loom_symbolic_integer_relation_t upper_relation, int64_t upper_bound) {
  if (upper_relation != LOOM_SYMBOLIC_INTEGER_RELATION_LT &&
      upper_relation != LOOM_SYMBOLIC_INTEGER_RELATION_LE) {
    return false;
  }
  int64_t effective_upper_bound = upper_bound;
  if (upper_relation == LOOM_SYMBOLIC_INTEGER_RELATION_LT &&
      !loom_checked_sub_i64(effective_upper_bound, 1, &effective_upper_bound)) {
    return false;
  }
  int64_t scaled_upper_bound = 0;
  if (!loom_checked_mul_i64(scale, effective_upper_bound,
                            &scaled_upper_bound)) {
    return false;
  }
  int64_t maximum_difference = 0;
  return loom_checked_add_i64(constant, scaled_upper_bound,
                              &maximum_difference) &&
         maximum_difference <= 0;
}

typedef struct loom_symbolic_expr_scaled_le_proof_t {
  loom_value_id_t positive_relation_value;
  loom_value_id_t negative_relation_value;
  int64_t scale;
  int64_t constant;
} loom_symbolic_expr_scaled_le_proof_t;

typedef struct loom_symbolic_expr_scaled_static_le_proof_t {
  // SSA value whose identity-chain predicates constrain the positive term.
  loom_value_id_t positive_relation_value;

  // Positive coefficient multiplying positive_relation_value.
  int64_t scale;

  // Constant residual in the normalized less-or-equal comparison.
  int64_t constant;
} loom_symbolic_expr_scaled_static_le_proof_t;

static iree_status_t loom_symbolic_expr_predicate_arg_upper_bound(
    loom_symbolic_expr_context_t* context, loom_predicate_arg_tag_t arg_tag,
    int64_t arg, int64_t* out_upper_bound, bool* out_known) {
  *out_upper_bound = 0;
  *out_known = false;
  switch (arg_tag) {
    case LOOM_PRED_ARG_VALUE: {
      if (arg < 0) return iree_ok_status();
      loom_symbolic_expr_t expression = {0};
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
          context, (loom_value_id_t)arg, &expression));
      if (loom_value_facts_is_float(expression.facts) ||
          expression.facts.range_hi == INT64_MAX) {
        return iree_ok_status();
      }
      *out_upper_bound = expression.facts.range_hi;
      *out_known = true;
      return iree_ok_status();
    }
    case LOOM_PRED_ARG_CONST:
      *out_upper_bound = arg;
      *out_known = true;
      return iree_ok_status();
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_symbolic_expr_scaled_le_predicate_proof(
    loom_symbolic_expr_context_t* context, const loom_predicate_t* predicate,
    const void* user_data, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  const loom_symbolic_expr_scaled_le_proof_t* proof =
      (const loom_symbolic_expr_scaled_le_proof_t*)user_data;
  *out_matched = false;
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (predicate->arg_count != 2) return iree_ok_status();

  loom_symbolic_integer_relation_t predicate_relation =
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!loom_symbolic_expr_predicate_relation(predicate, &predicate_relation)) {
    return iree_ok_status();
  }

  bool positive_is_left = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_arg_matches_value(
      context, (loom_predicate_arg_tag_t)predicate->arg_tags[0],
      predicate->args[0], proof->positive_relation_value, &positive_is_left));
  bool positive_is_right = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_arg_matches_value(
      context, (loom_predicate_arg_tag_t)predicate->arg_tags[1],
      predicate->args[1], proof->positive_relation_value, &positive_is_right));

  loom_symbolic_integer_relation_t upper_relation = predicate_relation;
  uint8_t upper_arg_index = 1;
  if (positive_is_left) {
    upper_arg_index = 1;
  } else if (positive_is_right) {
    upper_relation = loom_symbolic_integer_relation_swap(predicate_relation);
    upper_arg_index = 0;
  } else {
    return iree_ok_status();
  }

  int64_t upper_minus_negative = 0;
  bool upper_difference_known = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_arg_difference_from_value(
      context, (loom_predicate_arg_tag_t)predicate->arg_tags[upper_arg_index],
      predicate->args[upper_arg_index], proof->negative_relation_value,
      &upper_minus_negative, &upper_difference_known));
  if (!upper_difference_known) return iree_ok_status();

  if (loom_symbolic_expr_scaled_upper_bound_proves_le(
          proof->scale, proof->constant, upper_relation,
          upper_minus_negative)) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
    *out_matched = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_scaled_static_le_predicate_proof(
    loom_symbolic_expr_context_t* context, const loom_predicate_t* predicate,
    const void* user_data, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  const loom_symbolic_expr_scaled_static_le_proof_t* proof =
      (const loom_symbolic_expr_scaled_static_le_proof_t*)user_data;
  *out_matched = false;
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (predicate->arg_count != 2) return iree_ok_status();

  loom_symbolic_integer_relation_t predicate_relation =
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!loom_symbolic_expr_predicate_relation(predicate, &predicate_relation)) {
    return iree_ok_status();
  }

  bool positive_is_left = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_arg_matches_value(
      context, (loom_predicate_arg_tag_t)predicate->arg_tags[0],
      predicate->args[0], proof->positive_relation_value, &positive_is_left));
  bool positive_is_right = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_arg_matches_value(
      context, (loom_predicate_arg_tag_t)predicate->arg_tags[1],
      predicate->args[1], proof->positive_relation_value, &positive_is_right));

  loom_symbolic_integer_relation_t upper_relation = predicate_relation;
  uint8_t upper_arg_index = 1;
  if (positive_is_left) {
    upper_arg_index = 1;
  } else if (positive_is_right) {
    upper_relation = loom_symbolic_integer_relation_swap(predicate_relation);
    upper_arg_index = 0;
  } else {
    return iree_ok_status();
  }

  int64_t upper_bound = 0;
  bool upper_bound_known = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_arg_upper_bound(
      context, (loom_predicate_arg_tag_t)predicate->arg_tags[upper_arg_index],
      predicate->args[upper_arg_index], &upper_bound, &upper_bound_known));
  if (!upper_bound_known) return iree_ok_status();

  if (loom_symbolic_expr_scaled_static_upper_bound_proves_le(
          proof->scale, proof->constant, upper_relation, upper_bound)) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
    *out_matched = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_identity_chain_predicates(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_symbolic_expr_predicate_proof_fn_t proof_fn,
    const void* proof_user_data, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  *out_matched = false;
  if (!context->module) return iree_ok_status();
  loom_value_id_t current_value = start_value;
  uint8_t remaining_steps = LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0) {
    if (current_value >= context->module->values.count) {
      return iree_ok_status();
    }
    const loom_value_t* value =
        loom_module_value(context->module, current_value);
    if (loom_value_is_block_arg(value)) return iree_ok_status();
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) return iree_ok_status();

    if (loom_index_cast_isa(defining_op)) {
      current_value = loom_index_cast_input(defining_op);
      continue;
    }

    loom_value_slice_t values = {.values = NULL, .count = 0};
    if (loom_index_assume_isa(defining_op)) {
      values = loom_index_assume_values(defining_op);
    } else if (loom_scalar_assume_isa(defining_op)) {
      values = loom_scalar_assume_values(defining_op);
    } else {
      return iree_ok_status();
    }

    uint16_t result_index = loom_value_def_index(value);
    if (result_index >= values.count) return iree_ok_status();
    if (defining_op->attribute_count > 0) {
      loom_attribute_t predicates_attr = loom_op_attrs(defining_op)[0];
      if (predicates_attr.kind == LOOM_ATTR_PREDICATE_LIST &&
          (predicates_attr.count == 0 || predicates_attr.predicate_list)) {
        for (uint16_t i = 0; i < predicates_attr.count; ++i) {
          IREE_RETURN_IF_ERROR(
              proof_fn(context, &predicates_attr.predicate_list[i],
                       proof_user_data, out_matched, out_result));
          if (*out_matched) return iree_ok_status();
        }
      }
    }
    current_value = values.values[result_index];
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_identity_chain_assumption(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t start_value,
    loom_value_id_t left_value, loom_value_id_t right_value, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  loom_symbolic_expr_relation_proof_t proof = {
      .relation = relation,
      .left_value = left_value,
      .right_value = right_value,
  };
  return loom_symbolic_expr_prove_identity_chain_predicates(
      context, start_value, loom_symbolic_expr_relation_predicate_proof, &proof,
      out_matched, out_result);
}

static iree_status_t loom_symbolic_expr_prove_identity_chain_scaled_assumption(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    const loom_symbolic_expr_scaled_le_proof_t* proof, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  return loom_symbolic_expr_prove_identity_chain_predicates(
      context, start_value, loom_symbolic_expr_scaled_le_predicate_proof, proof,
      out_matched, out_result);
}

static iree_status_t
loom_symbolic_expr_prove_identity_chain_scaled_static_assumption(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    const loom_symbolic_expr_scaled_static_le_proof_t* proof, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  return loom_symbolic_expr_prove_identity_chain_predicates(
      context, start_value, loom_symbolic_expr_scaled_static_le_predicate_proof,
      proof, out_matched, out_result);
}

static iree_status_t loom_symbolic_expr_prove_assumed_value_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  *out_matched = false;
  if (!loom_symbolic_expr_value_is_integer_domain(context, left_value) ||
      !loom_symbolic_expr_value_is_integer_domain(context, right_value)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_identity_chain_assumption(
      context, relation, left_value, left_value, right_value, out_matched,
      out_result));
  if (*out_matched) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_identity_chain_assumption(
      context, relation, right_value, left_value, right_value, out_matched,
      out_result));
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_direct_value_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  bool assumed_relation_matched = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_assumed_value_relation(
      context, relation, left_value, right_value, &assumed_relation_matched,
      out_result));
  if (assumed_relation_matched) return iree_ok_status();

  bool kernel_relation_matched = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_kernel_coordinate_proves_relation(
      context, relation, left_value, right_value, &kernel_relation_matched,
      out_result));
  if (kernel_relation_matched) return iree_ok_status();

  bool quotient_bound_matched = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_quotient_bound_proves_relation(
      context, relation, left_value, right_value, &quotient_bound_matched,
      out_result));
  if (quotient_bound_matched) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_quotient_bound_proves_relation(
      context, loom_symbolic_integer_relation_swap(relation), right_value,
      left_value, &quotient_bound_matched, out_result));
  if (quotient_bound_matched) return iree_ok_status();

  bool remainder_bound_matched = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_remainder_bound_proves_relation(
      context, relation, left_value, right_value, &remainder_bound_matched,
      out_result));
  return iree_ok_status();
}

static bool loom_symbolic_expr_scaled_pair_terms(
    const loom_symbolic_term_t* terms, iree_host_size_t term_count,
    int64_t* out_scale, loom_value_id_t* out_positive_relation_value,
    loom_value_id_t* out_negative_relation_value) {
  if (term_count != 2) return false;
  const loom_symbolic_term_t* positive_term = NULL;
  const loom_symbolic_term_t* negative_term = NULL;
  if (terms[0].coefficient > 0 && terms[1].coefficient < 0) {
    positive_term = &terms[0];
    negative_term = &terms[1];
  } else if (terms[1].coefficient > 0 && terms[0].coefficient < 0) {
    positive_term = &terms[1];
    negative_term = &terms[0];
  } else {
    return false;
  }
  if (negative_term->coefficient == INT64_MIN ||
      positive_term->coefficient != -negative_term->coefficient) {
    return false;
  }
  *out_scale = positive_term->coefficient;
  *out_positive_relation_value = positive_term->relation_value_id;
  if (*out_positive_relation_value == LOOM_VALUE_ID_INVALID) {
    *out_positive_relation_value = positive_term->value_id;
  }
  *out_negative_relation_value = negative_term->relation_value_id;
  if (*out_negative_relation_value == LOOM_VALUE_ID_INVALID) {
    *out_negative_relation_value = negative_term->value_id;
  }
  return true;
}

static bool loom_symbolic_expr_scaled_pair_term_indices(
    const loom_symbolic_term_t* terms, iree_host_size_t positive_index,
    iree_host_size_t negative_index, int64_t* out_scale,
    loom_value_id_t* out_positive_relation_value,
    loom_value_id_t* out_negative_relation_value) {
  const loom_symbolic_term_t* positive_term = &terms[positive_index];
  const loom_symbolic_term_t* negative_term = &terms[negative_index];
  if (positive_term->coefficient <= 0 || negative_term->coefficient >= 0 ||
      negative_term->coefficient == INT64_MIN ||
      positive_term->coefficient != -negative_term->coefficient) {
    return false;
  }
  *out_scale = positive_term->coefficient;
  *out_positive_relation_value = positive_term->relation_value_id;
  if (*out_positive_relation_value == LOOM_VALUE_ID_INVALID) {
    *out_positive_relation_value = positive_term->value_id;
  }
  *out_negative_relation_value = negative_term->relation_value_id;
  if (*out_negative_relation_value == LOOM_VALUE_ID_INVALID) {
    *out_negative_relation_value = negative_term->value_id;
  }
  return true;
}

static bool loom_symbolic_expr_scaled_pair_from_indices(
    const loom_symbolic_term_t* terms, iree_host_size_t first_index,
    iree_host_size_t second_index, int64_t* out_scale,
    loom_value_id_t* out_positive_relation_value,
    loom_value_id_t* out_negative_relation_value) {
  if (terms[first_index].coefficient > 0) {
    return loom_symbolic_expr_scaled_pair_term_indices(
        terms, first_index, second_index, out_scale,
        out_positive_relation_value, out_negative_relation_value);
  }
  return loom_symbolic_expr_scaled_pair_term_indices(
      terms, second_index, first_index, out_scale, out_positive_relation_value,
      out_negative_relation_value);
}

static iree_status_t loom_symbolic_expr_residual_interval_excluding_pair(
    loom_symbolic_expr_context_t* context, const loom_symbolic_term_t* terms,
    iree_host_size_t term_count, iree_host_size_t first_index,
    iree_host_size_t second_index, int64_t constant, int64_t* out_maximum,
    bool* out_known) {
  *out_known = false;
  int64_t minimum = constant;
  int64_t maximum = constant;
  for (iree_host_size_t i = 0; i < term_count; ++i) {
    if (i == first_index || i == second_index) continue;
    int64_t term_minimum = 0;
    int64_t term_maximum = 0;
    bool term_interval_known = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_term_interval(
        context, terms[i], &term_minimum, &term_maximum, &term_interval_known));
    if (!term_interval_known ||
        !loom_symbolic_expr_accumulate_checked(term_minimum, term_maximum,
                                               &minimum, &maximum)) {
      return iree_ok_status();
    }
  }
  *out_maximum = maximum;
  *out_known = true;
  return iree_ok_status();
}

static iree_status_t
loom_symbolic_expr_prove_le_by_scaled_relation_with_residual(
    loom_symbolic_expr_context_t* context, int64_t constant,
    const loom_symbolic_term_t* normalized_terms, iree_host_size_t term_count,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (term_count < 2 || term_count > LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT) {
    return iree_ok_status();
  }

  loom_symbolic_term_t terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
  memcpy(terms, normalized_terms, term_count * sizeof(*terms));

  for (iree_host_size_t i = 0; i < term_count; ++i) {
    for (iree_host_size_t j = i + 1; j < term_count; ++j) {
      int64_t scale = 0;
      loom_value_id_t positive_relation_value = LOOM_VALUE_ID_INVALID;
      loom_value_id_t negative_relation_value = LOOM_VALUE_ID_INVALID;
      if (!loom_symbolic_expr_scaled_pair_from_indices(
              terms, i, j, &scale, &positive_relation_value,
              &negative_relation_value)) {
        continue;
      }

      int64_t residual_maximum = 0;
      bool residual_interval_known = false;
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_residual_interval_excluding_pair(
          context, terms, term_count, i, j, constant, &residual_maximum,
          &residual_interval_known));
      if (!residual_interval_known) {
        continue;
      }

      loom_symbolic_proof_result_t strict_relation =
          LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_direct_value_relation(
          context, LOOM_SYMBOLIC_INTEGER_RELATION_LT, positive_relation_value,
          negative_relation_value, &strict_relation));
      if (strict_relation == LOOM_SYMBOLIC_PROOF_TRUE &&
          residual_maximum <= scale) {
        *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
        return iree_ok_status();
      }

      loom_symbolic_proof_result_t nonstrict_relation =
          LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_direct_value_relation(
          context, LOOM_SYMBOLIC_INTEGER_RELATION_LE, positive_relation_value,
          negative_relation_value, &nonstrict_relation));
      if (nonstrict_relation == LOOM_SYMBOLIC_PROOF_TRUE &&
          residual_maximum <= 0) {
        *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_le_by_scaled_relation(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  bool linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, left_expression, right_expression, &constant, &term_count,
      &linear));
  if (!linear) return iree_ok_status();

  loom_symbolic_term_t stable_terms[LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT];
  const loom_symbolic_term_t* terms = NULL;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_stabilize_scratch_terms(
      context, term_count, stable_terms, IREE_ARRAYSIZE(stable_terms), &terms));

  if (term_count == 1 && terms[0].coefficient > 0) {
    loom_value_id_t positive_relation_value = terms[0].relation_value_id;
    if (positive_relation_value == LOOM_VALUE_ID_INVALID) {
      positive_relation_value = terms[0].value_id;
    }
    bool scaled_static_assumption_matched = false;
    const loom_symbolic_expr_scaled_static_le_proof_t
        scaled_static_assumption_proof = {
            .positive_relation_value = positive_relation_value,
            .scale = terms[0].coefficient,
            .constant = constant,
        };
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_prove_identity_chain_scaled_static_assumption(
            context, positive_relation_value, &scaled_static_assumption_proof,
            &scaled_static_assumption_matched, out_result));
    if (scaled_static_assumption_matched) return iree_ok_status();
  }

  int64_t scale = 0;
  loom_value_id_t positive_relation_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t negative_relation_value = LOOM_VALUE_ID_INVALID;
  if (!loom_symbolic_expr_scaled_pair_terms(terms, term_count, &scale,
                                            &positive_relation_value,
                                            &negative_relation_value)) {
    IREE_RETURN_IF_ERROR(
        loom_symbolic_expr_prove_le_by_scaled_relation_with_residual(
            context, constant, terms, term_count, out_result));
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t strict_relation = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_direct_value_relation(
      context, LOOM_SYMBOLIC_INTEGER_RELATION_LT, positive_relation_value,
      negative_relation_value, &strict_relation));
  if (strict_relation == LOOM_SYMBOLIC_PROOF_TRUE && constant <= scale) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
    return iree_ok_status();
  }
  if (strict_relation == LOOM_SYMBOLIC_PROOF_FALSE && constant > 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
    return iree_ok_status();
  }

  bool scaled_assumption_matched = false;
  const loom_symbolic_expr_scaled_le_proof_t scaled_assumption_proof = {
      .positive_relation_value = positive_relation_value,
      .negative_relation_value = negative_relation_value,
      .scale = scale,
      .constant = constant,
  };
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_identity_chain_scaled_assumption(
          context, positive_relation_value, &scaled_assumption_proof,
          &scaled_assumption_matched, out_result));
  if (scaled_assumption_matched) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_identity_chain_scaled_assumption(
          context, negative_relation_value, &scaled_assumption_proof,
          &scaled_assumption_matched, out_result));
  if (scaled_assumption_matched) return iree_ok_status();

  loom_symbolic_proof_result_t nonstrict_relation = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_direct_value_relation(
      context, LOOM_SYMBOLIC_INTEGER_RELATION_LE, positive_relation_value,
      negative_relation_value, &nonstrict_relation));
  if (nonstrict_relation == LOOM_SYMBOLIC_PROOF_TRUE && constant <= 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
    return iree_ok_status();
  }
  int64_t false_lower_bound = 0;
  if (nonstrict_relation == LOOM_SYMBOLIC_PROOF_FALSE &&
      loom_checked_add_i64(scale, constant, &false_lower_bound) &&
      false_lower_bound > 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_le_by_scaled_relation_with_residual(
          context, constant, terms, term_count, out_result));
  return iree_ok_status();
}

static bool loom_symbolic_expr_select_condition_from_value(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_id_t* out_condition) {
  loom_value_id_t source_value =
      loom_symbolic_expr_assumption_source_value(context, value_id);
  const loom_op_t* defining_op =
      loom_symbolic_expr_value_defining_op(context, source_value);
  if (!defining_op || !loom_scf_select_isa(defining_op)) return false;
  *out_condition = loom_scf_select_condition(defining_op);
  return true;
}

static bool loom_symbolic_expr_predicate_arg_select_condition(
    const loom_symbolic_expr_context_t* context, loom_predicate_arg_tag_t tag,
    int64_t arg, loom_value_id_t* out_condition) {
  if (tag != LOOM_PRED_ARG_VALUE || arg < 0) return false;
  return loom_symbolic_expr_select_condition_from_value(
      context, (loom_value_id_t)arg, out_condition);
}

static bool loom_symbolic_expr_predicate_select_condition(
    const loom_symbolic_expr_context_t* context,
    const loom_predicate_t* predicate, loom_value_id_t* out_condition) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (loom_symbolic_expr_predicate_arg_select_condition(
            context, (loom_predicate_arg_tag_t)predicate->arg_tags[i],
            predicate->args[i], out_condition)) {
      return true;
    }
  }
  return false;
}

static void loom_symbolic_expr_append_select_condition(
    loom_value_id_t condition, loom_value_id_t* conditions,
    uint8_t* inout_condition_count) {
  if (condition == LOOM_VALUE_ID_INVALID ||
      *inout_condition_count >=
          LOOM_SYMBOLIC_EXPR_SELECT_CASE_CONDITION_LIMIT) {
    return;
  }
  for (uint8_t i = 0; i < *inout_condition_count; ++i) {
    if (conditions[i] == condition) return;
  }
  conditions[(*inout_condition_count)++] = condition;
}

static void loom_symbolic_expr_collect_predicate_select_conditions(
    const loom_symbolic_expr_context_t* context,
    const loom_predicate_t* predicate, loom_value_id_t* conditions,
    uint8_t* inout_condition_count) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    if (loom_symbolic_expr_predicate_arg_select_condition(
            context, (loom_predicate_arg_tag_t)predicate->arg_tags[i],
            predicate->args[i], &condition)) {
      loom_symbolic_expr_append_select_condition(condition, conditions,
                                                 inout_condition_count);
    }
  }
}

static bool loom_symbolic_expr_identity_chain_select_condition(
    const loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_id_t* out_condition) {
  if (!context->module || start_value == LOOM_VALUE_ID_INVALID) return false;

  loom_value_id_t current_value = start_value;
  uint8_t remaining_steps = LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0) {
    if (current_value >= context->module->values.count) return false;
    const loom_value_t* value =
        loom_module_value(context->module, current_value);
    if (loom_value_is_block_arg(value)) return false;
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) return false;

    if (loom_index_cast_isa(defining_op)) {
      current_value = loom_index_cast_input(defining_op);
      continue;
    }

    loom_value_slice_t values = {.values = NULL, .count = 0};
    if (loom_index_assume_isa(defining_op)) {
      values = loom_index_assume_values(defining_op);
    } else if (loom_scalar_assume_isa(defining_op)) {
      values = loom_scalar_assume_values(defining_op);
    } else {
      return false;
    }

    uint16_t result_index = loom_value_def_index(value);
    if (result_index >= values.count) return false;
    if (defining_op->attribute_count > 0) {
      loom_attribute_t predicates_attr = loom_op_attrs(defining_op)[0];
      if (predicates_attr.kind == LOOM_ATTR_PREDICATE_LIST &&
          (predicates_attr.count == 0 || predicates_attr.predicate_list)) {
        for (uint16_t i = 0; i < predicates_attr.count; ++i) {
          if (loom_symbolic_expr_predicate_select_condition(
                  context, &predicates_attr.predicate_list[i], out_condition)) {
            return true;
          }
        }
      }
    }
    current_value = values.values[result_index];
  }
  return false;
}

static void loom_symbolic_expr_collect_identity_chain_select_conditions(
    const loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_id_t* conditions, uint8_t* inout_condition_count) {
  if (!context->module || start_value == LOOM_VALUE_ID_INVALID) return;

  loom_value_id_t current_value = start_value;
  uint8_t remaining_steps = LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0 &&
         *inout_condition_count <
             LOOM_SYMBOLIC_EXPR_SELECT_CASE_CONDITION_LIMIT) {
    if (current_value >= context->module->values.count) return;
    const loom_value_t* value =
        loom_module_value(context->module, current_value);
    if (loom_value_is_block_arg(value)) return;
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) return;

    if (loom_index_cast_isa(defining_op)) {
      current_value = loom_index_cast_input(defining_op);
      continue;
    }

    loom_value_slice_t values = {.values = NULL, .count = 0};
    if (loom_index_assume_isa(defining_op)) {
      values = loom_index_assume_values(defining_op);
    } else if (loom_scalar_assume_isa(defining_op)) {
      values = loom_scalar_assume_values(defining_op);
    } else {
      return;
    }

    uint16_t result_index = loom_value_def_index(value);
    if (result_index >= values.count) return;
    if (defining_op->attribute_count > 0) {
      loom_attribute_t predicates_attr = loom_op_attrs(defining_op)[0];
      if (predicates_attr.kind == LOOM_ATTR_PREDICATE_LIST &&
          (predicates_attr.count == 0 || predicates_attr.predicate_list)) {
        for (uint16_t i = 0; i < predicates_attr.count; ++i) {
          loom_symbolic_expr_collect_predicate_select_conditions(
              context, &predicates_attr.predicate_list[i], conditions,
              inout_condition_count);
        }
      }
    }
    current_value = values.values[result_index];
  }
}

static void loom_symbolic_expr_push_dependency_value(
    loom_value_id_t value_id, loom_value_id_t* worklist,
    iree_host_size_t* inout_worklist_count) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      *inout_worklist_count >=
          LOOM_SYMBOLIC_EXPR_SELECT_DEPENDENCY_SEARCH_LIMIT) {
    return;
  }
  worklist[(*inout_worklist_count)++] = value_id;
}

static void loom_symbolic_expr_collect_dependency_select_conditions(
    const loom_symbolic_expr_context_t* context, loom_value_id_t root_value_id,
    iree_host_size_t* inout_remaining_value_count, loom_value_id_t* conditions,
    uint8_t* inout_condition_count) {
  if (!context->module || *inout_remaining_value_count == 0) return;

  loom_value_id_t worklist[LOOM_SYMBOLIC_EXPR_SELECT_DEPENDENCY_SEARCH_LIMIT];
  iree_host_size_t worklist_count = 0;
  loom_symbolic_expr_push_dependency_value(root_value_id, worklist,
                                           &worklist_count);
  while (worklist_count > 0 && *inout_remaining_value_count > 0 &&
         *inout_condition_count <
             LOOM_SYMBOLIC_EXPR_SELECT_CASE_CONDITION_LIMIT) {
    loom_value_id_t value_id = worklist[--worklist_count];
    if (value_id >= context->module->values.count) continue;
    --*inout_remaining_value_count;

    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    if (loom_symbolic_expr_select_condition_from_value(context, value_id,
                                                       &condition)) {
      loom_symbolic_expr_append_select_condition(condition, conditions,
                                                 inout_condition_count);
    }
    loom_symbolic_expr_collect_identity_chain_select_conditions(
        context, value_id, conditions, inout_condition_count);

    const loom_value_t* value = loom_module_value(context->module, value_id);
    if (loom_value_is_block_arg(value)) continue;
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) continue;

    const loom_value_id_t* operands = loom_op_const_operands(defining_op);
    for (uint16_t i = defining_op->operand_count; i > 0; --i) {
      loom_symbolic_expr_push_dependency_value(operands[i - 1], worklist,
                                               &worklist_count);
    }
  }
}

static iree_status_t loom_symbolic_expr_collect_select_conditions_for_le(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, loom_value_id_t* conditions,
    uint8_t* out_condition_count) {
  *out_condition_count = 0;

  int64_t constant = 0;
  iree_host_size_t term_count = 0;
  bool linear = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_normalize_difference_into_scratch(
      context, left_expression, right_expression, &constant, &term_count,
      &linear));
  (void)constant;
  if (!linear) return iree_ok_status();

  for (iree_host_size_t i = 0; i < term_count; ++i) {
    const loom_symbolic_term_t term = context->scratch_terms[i];
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    if (loom_symbolic_expr_select_condition_from_value(context, term.value_id,
                                                       &condition)) {
      loom_symbolic_expr_append_select_condition(condition, conditions,
                                                 out_condition_count);
    }
    loom_symbolic_expr_collect_identity_chain_select_conditions(
        context, term.relation_value_id, conditions, out_condition_count);
  }

  iree_host_size_t remaining_value_count =
      LOOM_SYMBOLIC_EXPR_SELECT_DEPENDENCY_SEARCH_LIMIT;
  for (iree_host_size_t i = 0;
       i < term_count && remaining_value_count > 0 &&
       *out_condition_count < LOOM_SYMBOLIC_EXPR_SELECT_CASE_CONDITION_LIMIT;
       ++i) {
    const loom_symbolic_term_t term = context->scratch_terms[i];
    loom_symbolic_expr_collect_dependency_select_conditions(
        context, term.value_id, &remaining_value_count, conditions,
        out_condition_count);
    loom_symbolic_expr_collect_dependency_select_conditions(
        context, term.relation_value_id, &remaining_value_count, conditions,
        out_condition_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_le_with_condition_facts(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, loom_value_id_t condition,
    bool assumed_truth, loom_symbolic_proof_result_t* out_result) {
  loom_condition_integer_relation_t relation_storage[32];
  loom_condition_fact_set_t condition_facts;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &condition_facts);
  const loom_condition_fact_set_t* previous_condition_facts =
      context->condition_facts;
  if (previous_condition_facts) {
    for (iree_host_size_t i = 0;
         i < previous_condition_facts->integer_relation_count &&
         condition_facts.integer_relation_count <
             condition_facts.integer_relation_capacity;
         ++i) {
      condition_facts
          .integer_relations[condition_facts.integer_relation_count++] =
          previous_condition_facts->integer_relations[i];
    }
  }
  (void)loom_condition_facts_query_into(context->module, context->fact_table,
                                        condition, assumed_truth,
                                        &condition_facts);

  uint8_t previous_condition_proof_depth = context->condition_proof_depth;
  context->condition_facts = &condition_facts;
  context->condition_proof_depth =
      (uint8_t)(previous_condition_proof_depth + 1);
  loom_symbolic_expr_context_reset(context);
  iree_status_t status = loom_symbolic_expr_prove_le(
      context, left_expression, right_expression, out_result);
  context->condition_facts = previous_condition_facts;
  context->condition_proof_depth = previous_condition_proof_depth;
  loom_symbolic_expr_context_reset(context);
  return status;
}

static iree_status_t loom_symbolic_expr_prove_le_by_select_cases(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (context->condition_proof_depth >=
      LOOM_SYMBOLIC_EXPR_SELECT_CASE_DEPTH_LIMIT) {
    return iree_ok_status();
  }

  loom_value_id_t conditions[LOOM_SYMBOLIC_EXPR_SELECT_CASE_CONDITION_LIMIT] = {
      LOOM_VALUE_ID_INVALID};
  uint8_t condition_count = 0;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_collect_select_conditions_for_le(
      context, left_expression, right_expression, conditions,
      &condition_count));
  for (uint8_t i = 0; i < condition_count; ++i) {
    bool proven_condition = false;
    if (context->condition_facts &&
        loom_condition_fact_set_proves_condition(
            context->module, context->fact_table, context->condition_facts,
            conditions[i], &proven_condition)) {
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_with_condition_facts(
          context, left_expression, right_expression, conditions[i],
          proven_condition, out_result));
      if (*out_result == LOOM_SYMBOLIC_PROOF_TRUE) {
        return iree_ok_status();
      }
      continue;
    }

    loom_symbolic_proof_result_t true_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_with_condition_facts(
        context, left_expression, right_expression, conditions[i],
        /*assumed_truth=*/true, &true_result));
    loom_symbolic_proof_result_t false_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_with_condition_facts(
        context, left_expression, right_expression, conditions[i],
        /*assumed_truth=*/false, &false_result));
    if (true_result == LOOM_SYMBOLIC_PROOF_TRUE &&
        false_result == LOOM_SYMBOLIC_PROOF_TRUE) {
      *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_symbolic_expr_prove_le(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (loom_symbolic_expr_is_linear(left_expression) &&
      loom_symbolic_expr_is_linear(right_expression)) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_linear(
        context, left_expression, right_expression, out_result));
    if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_by_scaled_relation(
        context, left_expression, right_expression, out_result));
    if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_by_select_cases(
        context, left_expression, right_expression, out_result));
    if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) return iree_ok_status();
  }
  // Expanded expressions may still carry stronger facts from their defining
  // SSA value, such as index.assume range facts on a value that algebraically
  // expands back to its unconstrained source.
  *out_result =
      loom_symbolic_expr_prove_le_by_facts(left_expression, right_expression);
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_equal(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  loom_symbolic_proof_result_t left_le_right = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
      context, left_expression, right_expression, &left_le_right));
  loom_symbolic_proof_result_t right_le_left = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
      context, right_expression, left_expression, &right_le_left));
  if (left_le_right == LOOM_SYMBOLIC_PROOF_TRUE &&
      right_le_left == LOOM_SYMBOLIC_PROOF_TRUE) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
  } else if (left_le_right == LOOM_SYMBOLIC_PROOF_FALSE ||
             right_le_left == LOOM_SYMBOLIC_PROOF_FALSE) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
  } else {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_prove_less_than(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  loom_symbolic_proof_result_t right_le_left = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le(
      context, right_expression, left_expression, &right_le_left));
  if (right_le_left == LOOM_SYMBOLIC_PROOF_TRUE) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
  } else if (right_le_left == LOOM_SYMBOLIC_PROOF_FALSE) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
  } else {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  return iree_ok_status();
}

iree_status_t loom_symbolic_expr_prove_value_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_direct_value_relation(
      context, relation, left_value, right_value, out_result));
  if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) return iree_ok_status();

  loom_symbolic_expr_t left_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));

  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      return loom_symbolic_expr_prove_equal(context, &left_expression,
                                            &right_expression, out_result);
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE: {
      loom_symbolic_proof_result_t equal = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_equal(
          context, &left_expression, &right_expression, &equal));
      if (equal == LOOM_SYMBOLIC_PROOF_TRUE) {
        *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
      } else if (equal == LOOM_SYMBOLIC_PROOF_FALSE) {
        *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
      }
      return iree_ok_status();
    }
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      return loom_symbolic_expr_prove_less_than(context, &left_expression,
                                                &right_expression, out_result);
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      return loom_symbolic_expr_prove_le(context, &left_expression,
                                         &right_expression, out_result);
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      return loom_symbolic_expr_prove_less_than(context, &right_expression,
                                                &left_expression, out_result);
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      return loom_symbolic_expr_prove_le(context, &right_expression,
                                         &left_expression, out_result);
    default:
      return iree_ok_status();
  }
}
