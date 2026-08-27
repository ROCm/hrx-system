// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_value.h"

#include <string.h>

#include "loom/analysis/condition_facts.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"

#define LOOM_SYMBOLIC_VALUE_CONDITION_FACT_INFER_DEPTH_LIMIT 16
#define LOOM_SYMBOLIC_VALUE_CONDITION_FACT_STACK_VALUE_CAPACITY 8
#define LOOM_SYMBOLIC_VALUE_IDENTITY_CHAIN_LIMIT 64
#define LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_DEPTH_LIMIT 16
#define LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_LIMIT 64

//===----------------------------------------------------------------------===//
// Condition-refined facts
//===----------------------------------------------------------------------===//

enum loom_symbolic_value_condition_fact_memo_state_e {
  LOOM_SYMBOLIC_VALUE_CONDITION_FACT_MEMO_EMPTY = 0,
  LOOM_SYMBOLIC_VALUE_CONDITION_FACT_MEMO_VISITING = 1,
  LOOM_SYMBOLIC_VALUE_CONDITION_FACT_MEMO_READY = 2,
};

static iree_status_t loom_symbolic_value_ensure_condition_fact_memo_capacity(
    loom_symbolic_expr_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= context->condition_fact_memo_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t old_capacity = context->condition_fact_memo_capacity;
  void* entries = context->condition_fact_memo_entries;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->arena, context->condition_fact_memo_capacity, minimum_capacity,
      sizeof(*context->condition_fact_memo_entries),
      &context->condition_fact_memo_capacity, &entries));
  context->condition_fact_memo_entries =
      (loom_symbolic_expr_condition_fact_memo_entry_t*)entries;
  memset(context->condition_fact_memo_entries + old_capacity, 0,
         (context->condition_fact_memo_capacity - old_capacity) *
             sizeof(*context->condition_fact_memo_entries));
  return iree_ok_status();
}

static loom_value_facts_t loom_symbolic_value_intersect_integer_facts(
    loom_value_facts_t lhs, loom_value_facts_t rhs) {
  if (loom_value_facts_is_unknown(lhs) || loom_value_facts_is_float(lhs)) {
    return rhs;
  }
  if (loom_value_facts_is_unknown(rhs) || loom_value_facts_is_float(rhs)) {
    return lhs;
  }
  int64_t lower_bound = iree_max(lhs.range_lo, rhs.range_lo);
  int64_t upper_bound = iree_min(lhs.range_hi, rhs.range_hi);
  if (lower_bound > upper_bound) return loom_value_facts_unknown();
  return loom_value_facts_make(lower_bound, upper_bound, 1);
}

static bool loom_symbolic_value_exact_integer_facts(loom_value_facts_t facts,
                                                    int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_symbolic_value_constant_expression(
    const loom_symbolic_expr_t* expression, int64_t* out_value) {
  if (!loom_symbolic_expr_is_constant(expression)) return false;
  *out_value = expression->constant;
  return true;
}

typedef struct loom_symbolic_expr_condition_fact_frame_t {
  // SSA value whose facts this frame computes.
  loom_value_id_t value_id;

  // Remaining producer depth available to operand frames.
  uint8_t remaining_depth;

  // Next producer operand requiring condition-refined facts.
  uint16_t next_operand;

  // Defining operation whose transfer function computes the result facts.
  const loom_op_t* defining_op;

  // Result ordinal represented by value_id.
  uint16_t result_index;

  // Facts known before recursively applying the defining transfer function.
  loom_value_facts_t base_facts;

  // Destination receiving this frame's completed facts.
  loom_value_facts_t* out_facts;

  // Operand facts passed to the defining transfer function.
  loom_value_facts_t* operand_facts;

  // Inline operand-fact storage for ordinary scalar operations.
  loom_value_facts_t operand_facts_storage
      [LOOM_SYMBOLIC_VALUE_CONDITION_FACT_STACK_VALUE_CAPACITY];
} loom_symbolic_expr_condition_fact_frame_t;

static iree_status_t loom_symbolic_expr_condition_fact_frame_begin(
    loom_symbolic_expr_context_t* context,
    iree_arena_allocator_t* transient_arena, loom_value_id_t value_id,
    uint8_t remaining_depth, loom_value_facts_t* out_facts,
    loom_symbolic_expr_condition_fact_frame_t* out_frame, bool* out_pending) {
  *out_pending = false;
  if (!context->condition_facts || !context->module || !context->fact_table ||
      remaining_depth == 0 || value_id >= context->module->values.count) {
    return loom_symbolic_expr_context_lookup_facts(context, value_id,
                                                   out_facts);
  }

  IREE_RETURN_IF_ERROR(loom_symbolic_value_ensure_condition_fact_memo_capacity(
      context, value_id + 1));
  loom_symbolic_expr_condition_fact_memo_entry_t* memo_entry =
      &context->condition_fact_memo_entries[value_id];
  if (memo_entry->state == LOOM_SYMBOLIC_VALUE_CONDITION_FACT_MEMO_READY &&
      memo_entry->depth >= remaining_depth) {
    *out_facts = memo_entry->facts;
    return iree_ok_status();
  }

  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_context_lookup_facts(context, value_id, &facts));
  IREE_RETURN_IF_ERROR(
      loom_symbolic_value_apply_identity_chain_predicates_to_facts(
          context, value_id, &facts));
  if (memo_entry->state == LOOM_SYMBOLIC_VALUE_CONDITION_FACT_MEMO_VISITING) {
    *out_facts = facts;
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(context->module, value_id);
  const loom_op_t* defining_op =
      loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
  const loom_op_vtable_t* vtable =
      defining_op ? loom_op_vtable(context->module, defining_op) : NULL;
  const uint16_t result_index =
      defining_op ? loom_value_def_index(value) : UINT16_MAX;
  if (!vtable || !vtable->infer_facts ||
      result_index >= defining_op->result_count ||
      !loom_type_is_scalar(loom_module_value_type(context->module, value_id))) {
    memo_entry->facts = facts;
    memo_entry->depth = remaining_depth;
    memo_entry->state = LOOM_SYMBOLIC_VALUE_CONDITION_FACT_MEMO_READY;
    *out_facts = facts;
    return iree_ok_status();
  }

  memo_entry->state = LOOM_SYMBOLIC_VALUE_CONDITION_FACT_MEMO_VISITING;

  *out_frame = (loom_symbolic_expr_condition_fact_frame_t){
      .value_id = value_id,
      .remaining_depth = remaining_depth,
      .next_operand = 0,
      .defining_op = defining_op,
      .result_index = result_index,
      .base_facts = facts,
      .out_facts = out_facts,
      .operand_facts = NULL,
  };
  if (defining_op->operand_count <=
      IREE_ARRAYSIZE(out_frame->operand_facts_storage)) {
    if (defining_op->operand_count != 0) {
      out_frame->operand_facts = out_frame->operand_facts_storage;
    }
  } else {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        transient_arena, defining_op->operand_count,
        sizeof(*out_frame->operand_facts), (void**)&out_frame->operand_facts));
  }
  *out_pending = true;
  return iree_ok_status();
}

static iree_status_t loom_symbolic_value_lookup_condition_refined_facts_bounded(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    uint8_t remaining_depth, loom_value_facts_t* out_facts) {
  iree_arena_allocator_t transient_arena;
  iree_arena_initialize(context->arena->block_pool, &transient_arena);

  loom_symbolic_expr_condition_fact_frame_t
      frames[LOOM_SYMBOLIC_VALUE_CONDITION_FACT_INFER_DEPTH_LIMIT + 1];
  iree_host_size_t frame_count = 0;
  bool root_pending = false;
  iree_status_t status = loom_symbolic_expr_condition_fact_frame_begin(
      context, &transient_arena, value_id, remaining_depth, out_facts,
      &frames[0], &root_pending);
  if (iree_status_is_ok(status) && root_pending) {
    frame_count = 1;
  }

  while (iree_status_is_ok(status) && frame_count > 0) {
    loom_symbolic_expr_condition_fact_frame_t* frame = &frames[frame_count - 1];
    if (frame->next_operand < frame->defining_op->operand_count) {
      const uint16_t operand_index = frame->next_operand++;
      const loom_value_id_t operand =
          loom_op_const_operands(frame->defining_op)[operand_index];
      bool operand_pending = false;
      status = loom_symbolic_expr_condition_fact_frame_begin(
          context, &transient_arena, operand,
          (uint8_t)(frame->remaining_depth - 1),
          &frame->operand_facts[operand_index], &frames[frame_count],
          &operand_pending);
      if (iree_status_is_ok(status) && operand_pending) {
        ++frame_count;
      }
      continue;
    }

    loom_value_facts_t result_facts_storage
        [LOOM_SYMBOLIC_VALUE_CONDITION_FACT_STACK_VALUE_CAPACITY];
    loom_value_facts_t* result_facts = result_facts_storage;
    if (frame->defining_op->result_count >
        IREE_ARRAYSIZE(result_facts_storage)) {
      status = iree_arena_allocate_array(
          &transient_arena, frame->defining_op->result_count,
          sizeof(*result_facts), (void**)&result_facts);
      if (!iree_status_is_ok(status)) continue;
    }
    for (uint16_t i = 0; i < frame->defining_op->result_count; ++i) {
      result_facts[i] = loom_value_facts_unknown();
    }
    const loom_op_vtable_t* vtable =
        loom_op_vtable(context->module, frame->defining_op);
    loom_fact_context_t fact_context = context->fact_table->context;
    status =
        vtable->infer_facts(&fact_context, context->module, frame->defining_op,
                            frame->operand_facts, result_facts);
    if (!iree_status_is_ok(status)) continue;

    loom_value_facts_t inferred_facts = result_facts[frame->result_index];
    loom_condition_fact_set_apply_to_value_facts(
        context->condition_facts, context->fact_table, frame->value_id,
        &inferred_facts);
    status = loom_symbolic_value_apply_identity_chain_predicates_to_facts(
        context, frame->value_id, &inferred_facts);
    if (!iree_status_is_ok(status)) continue;

    *frame->out_facts = loom_symbolic_value_intersect_integer_facts(
        frame->base_facts, inferred_facts);
    // Preparing operand frames can grow and move the memo table. Reacquire the
    // entry before publishing the completed facts.
    loom_symbolic_expr_condition_fact_memo_entry_t* memo_entry =
        &context->condition_fact_memo_entries[frame->value_id];
    memo_entry->facts = *frame->out_facts;
    memo_entry->depth = frame->remaining_depth;
    memo_entry->state = LOOM_SYMBOLIC_VALUE_CONDITION_FACT_MEMO_READY;
    --frame_count;
  }

  iree_arena_deinitialize(&transient_arena);
  return status;
}

iree_status_t loom_symbolic_value_lookup_condition_refined_facts(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_value_facts_t* out_facts) {
  return loom_symbolic_value_lookup_condition_refined_facts_bounded(
      context, value_id, LOOM_SYMBOLIC_VALUE_CONDITION_FACT_INFER_DEPTH_LIMIT,
      out_facts);
}

//===----------------------------------------------------------------------===//
// Identity chains and value equivalence
//===----------------------------------------------------------------------===//

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
      *out_known =
          loom_symbolic_value_constant_expression(&expression, out_value);
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
    IREE_RETURN_IF_ERROR(
        loom_symbolic_values_match(context, (loom_value_id_t)predicate->args[0],
                                   value_id, &value_is_left));
  }
  bool value_is_right = false;
  if (predicate->arg_tags[1] == LOOM_PRED_ARG_VALUE &&
      predicate->args[1] >= 0) {
    IREE_RETURN_IF_ERROR(
        loom_symbolic_values_match(context, (loom_value_id_t)predicate->args[1],
                                   value_id, &value_is_right));
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

typedef enum loom_symbolic_expr_identity_chain_flag_bits_e {
  LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_FOLLOW_INDEX_CASTS = 1u << 0,
} loom_symbolic_expr_identity_chain_flag_bits_t;

typedef uint32_t loom_symbolic_expr_identity_chain_flags_t;

typedef struct loom_symbolic_expr_identity_chain_step_t {
  // Next value in the identity chain.
  loom_value_id_t next_value;

  // Optional predicate list carried by the current assume op.
  loom_attribute_t predicates_attr;
} loom_symbolic_expr_identity_chain_step_t;

static bool loom_symbolic_expr_predicate_list_attr(const loom_op_t* op,
                                                   loom_attribute_t* out_attr) {
  *out_attr = (loom_attribute_t){0};
  if (op->attribute_count == 0) return false;
  const loom_attribute_t attr = loom_op_attrs(op)[0];
  if (attr.kind != LOOM_ATTR_PREDICATE_LIST ||
      (attr.count != 0 && attr.predicate_list == NULL)) {
    return false;
  }
  *out_attr = attr;
  return true;
}

static bool loom_symbolic_expr_identity_chain_step(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_identity_chain_flags_t flags,
    loom_symbolic_expr_identity_chain_step_t* out_step) {
  *out_step = (loom_symbolic_expr_identity_chain_step_t){
      .next_value = LOOM_VALUE_ID_INVALID,
      .predicates_attr = {0},
  };
  if (!context->module || value_id >= context->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) return false;

  if (loom_index_cast_isa(defining_op)) {
    if (!iree_all_bits_set(
            flags, LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_FOLLOW_INDEX_CASTS)) {
      return false;
    }
    out_step->next_value = loom_index_cast_input(defining_op);
    return true;
  }

  loom_value_slice_t values = {.values = NULL, .count = 0};
  if (loom_index_assume_isa(defining_op)) {
    values = loom_index_assume_values(defining_op);
  } else if (loom_scalar_assume_isa(defining_op)) {
    values = loom_scalar_assume_values(defining_op);
  } else {
    return false;
  }

  const uint16_t result_index = loom_value_def_index(value);
  if (result_index >= values.count) return false;
  out_step->next_value = values.values[result_index];
  (void)loom_symbolic_expr_predicate_list_attr(defining_op,
                                               &out_step->predicates_attr);
  return true;
}

iree_status_t loom_symbolic_value_apply_identity_chain_predicates_to_facts(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_facts_t* inout_facts) {
  if (!context->module || start_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_value_id_t current_value = start_value;
  uint8_t remaining_steps = LOOM_SYMBOLIC_VALUE_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0) {
    loom_symbolic_expr_identity_chain_step_t step = {0};
    if (!loom_symbolic_expr_identity_chain_step(
            context, current_value,
            LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_FOLLOW_INDEX_CASTS, &step)) {
      return iree_ok_status();
    }
    const loom_attribute_t predicates_attr = step.predicates_attr;
    for (uint16_t i = 0; i < predicates_attr.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_predicate_apply_to_value_facts(
          context, &predicates_attr.predicate_list[i], current_value,
          inout_facts));
    }
    current_value = step.next_value;
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

static loom_value_id_t loom_symbolic_expr_assumption_source_value(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  if (!context->module) return value_id;
  uint8_t remaining_steps = LOOM_SYMBOLIC_VALUE_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0) {
    loom_symbolic_expr_identity_chain_step_t step = {0};
    if (!loom_symbolic_expr_identity_chain_step(context, value_id,
                                                /*flags=*/0, &step)) {
      return value_id;
    }
    value_id = step.next_value;
  }
  return value_id;
}

static bool loom_symbolic_expr_ops_can_structurally_match(
    const loom_module_t* module, const loom_op_t* left_op,
    const loom_op_t* right_op, uint16_t left_result_index,
    uint16_t right_result_index) {
  if (!left_op || !right_op || left_op->kind != right_op->kind ||
      left_result_index != right_result_index ||
      left_op->operand_count != right_op->operand_count ||
      left_op->result_count != right_op->result_count ||
      left_op->attribute_count != right_op->attribute_count ||
      left_op->instance_flags != right_op->instance_flags ||
      left_op->region_count != 0 || right_op->region_count != 0 ||
      left_op->tied_result_count != 0 || right_op->tied_result_count != 0) {
    return false;
  }

  const loom_trait_flags_t prohibited_traits =
      LOOM_TRAIT_NON_DETERMINISTIC | LOOM_TRAIT_UNKNOWN_EFFECTS |
      LOOM_TRAIT_UNIQUE_IDENTITY | LOOM_TRAIT_CONVERGENT |
      LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY |
      LOOM_TRAIT_MEMORY_FENCE;
  const loom_trait_flags_t left_traits =
      loom_op_effective_traits(module, left_op);
  const loom_trait_flags_t right_traits =
      loom_op_effective_traits(module, right_op);
  if (!iree_all_bits_set(left_traits, LOOM_TRAIT_PURE) ||
      !iree_all_bits_set(right_traits, LOOM_TRAIT_PURE) ||
      iree_any_bit_set(left_traits | right_traits, prohibited_traits)) {
    return false;
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, left_op);
  const uint8_t operand_segment_count =
      loom_op_vtable_operand_segment_count(vtable);
  if (operand_segment_count != 0 &&
      memcmp(loom_op_const_operand_segment_counts(left_op),
             loom_op_const_operand_segment_counts(right_op),
             (iree_host_size_t)operand_segment_count * sizeof(uint16_t)) != 0) {
    return false;
  }
  const loom_attribute_t* left_attrs = loom_op_const_attrs(left_op);
  const loom_attribute_t* right_attrs = loom_op_const_attrs(right_op);
  for (uint8_t i = 0; i < left_op->attribute_count; ++i) {
    if (!loom_attribute_equal(&left_attrs[i], &right_attrs[i])) return false;
  }

  const loom_value_id_t left_result =
      loom_op_const_results(left_op)[left_result_index];
  const loom_value_id_t right_result =
      loom_op_const_results(right_op)[right_result_index];
  return loom_type_equal(loom_module_value_type(module, left_result),
                         loom_module_value_type(module, right_result));
}

iree_status_t loom_symbolic_values_match(loom_symbolic_expr_context_t* context,
                                         loom_value_id_t left_value,
                                         loom_value_id_t right_value,
                                         bool* out_match) {
  *out_match = false;
  if (left_value == right_value) {
    *out_match = true;
    return iree_ok_status();
  }

  int64_t left_exact = 0;
  int64_t right_exact = 0;
  loom_value_facts_t left_facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_context_lookup_facts(
      context, left_value, &left_facts));
  loom_value_facts_t right_facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_context_lookup_facts(
      context, right_value, &right_facts));
  if (loom_symbolic_value_exact_integer_facts(left_facts, &left_exact) &&
      loom_symbolic_value_exact_integer_facts(right_facts, &right_exact)) {
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

typedef struct loom_symbolic_semantic_match_frame_t {
  // Left SSA value compared by this frame.
  loom_value_id_t left_value;

  // Right SSA value compared by this frame.
  loom_value_id_t right_value;

  // Remaining producer depth available to operand frames.
  uint8_t remaining_depth;

  // Next producer operand requiring comparison.
  uint16_t next_operand;

  // Structurally matched left producer, or NULL before frame entry.
  const loom_op_t* left_op;

  // Structurally matched right producer, or NULL before frame entry.
  const loom_op_t* right_op;
} loom_symbolic_semantic_match_frame_t;

typedef struct loom_symbolic_semantic_match_pair_t {
  // Canonically ordered lower SSA value ID.
  loom_value_id_t lower_value;

  // Canonically ordered upper SSA value ID.
  loom_value_id_t upper_value;
} loom_symbolic_semantic_match_pair_t;

#define LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_TABLE_CAPACITY \
  (LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_LIMIT * 2)
static_assert((LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_TABLE_CAPACITY &
               (LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_TABLE_CAPACITY - 1)) ==
                  0,
              "semantic match pair table capacity must be a power of two");

// Inserts a canonically ordered value pair and returns true when it was new.
// The table remains at most half full because the proof budget bounds inserts.
static bool loom_symbolic_semantic_match_pair_insert(
    loom_symbolic_semantic_match_pair_t* pair_table, loom_value_id_t left_value,
    loom_value_id_t right_value) {
  const loom_value_id_t lower_value = iree_min(left_value, right_value);
  const loom_value_id_t upper_value = iree_max(left_value, right_value);
  const uint32_t hash =
      lower_value * UINT32_C(0x9E3779B1) ^ upper_value * UINT32_C(0x85EBCA77);
  iree_host_size_t index =
      hash & (LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_TABLE_CAPACITY - 1);
  for (iree_host_size_t probe = 0;
       probe < LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_TABLE_CAPACITY;
       ++probe) {
    loom_symbolic_semantic_match_pair_t* pair = &pair_table[index];
    if (pair->lower_value == LOOM_VALUE_ID_INVALID) {
      *pair = (loom_symbolic_semantic_match_pair_t){
          .lower_value = lower_value,
          .upper_value = upper_value,
      };
      return true;
    }
    if (pair->lower_value == lower_value && pair->upper_value == upper_value) {
      return false;
    }
    index = (index + 1) &
            (LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_TABLE_CAPACITY - 1);
  }
  IREE_ASSERT_UNREACHABLE("symbolic semantic match pair table is full");
  IREE_BUILTIN_UNREACHABLE();
}

// Compares deterministic producer DAGs when ordinary symbolic cancellation
// cannot relate their separately materialized values. The explicit depth-first
// traversal bounds producer depth and distinct pairs while visiting shared DAG
// pairs only once.
iree_status_t loom_symbolic_values_semantically_match(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value, bool* out_match) {
  *out_match = false;

  loom_symbolic_semantic_match_frame_t
      frames[LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_DEPTH_LIMIT + 1] = {
          {
              .left_value = left_value,
              .right_value = right_value,
              .remaining_depth = LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_DEPTH_LIMIT,
          },
      };
  iree_host_size_t frame_count = 1;

  loom_symbolic_semantic_match_pair_t
      pair_table[LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_TABLE_CAPACITY];
  memset(pair_table, 0xFF, sizeof(pair_table));
  uint16_t remaining_pair_count = LOOM_SYMBOLIC_VALUE_SEMANTIC_MATCH_PAIR_LIMIT;

  while (frame_count > 0) {
    loom_symbolic_semantic_match_frame_t* frame = &frames[frame_count - 1];
    if (frame->left_op != NULL) {
      if (frame->next_operand == frame->left_op->operand_count) {
        --frame_count;
        continue;
      }
      const uint16_t operand_index = frame->next_operand++;
      const loom_value_id_t* left_operands =
          loom_op_const_operands(frame->left_op);
      const loom_value_id_t* right_operands =
          loom_op_const_operands(frame->right_op);
      IREE_ASSERT_LT(frame_count, IREE_ARRAYSIZE(frames));
      frames[frame_count++] = (loom_symbolic_semantic_match_frame_t){
          .left_value = left_operands[operand_index],
          .right_value = right_operands[operand_index],
          .remaining_depth = (uint8_t)(frame->remaining_depth - 1),
      };
      continue;
    }

    bool values_match = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
        context, frame->left_value, frame->right_value, &values_match));
    if (values_match) {
      --frame_count;
      continue;
    }
    if (frame->remaining_depth == 0 || remaining_pair_count == 0) {
      return iree_ok_status();
    }

    const loom_value_id_t normalized_left =
        loom_symbolic_expr_assumption_source_value(context, frame->left_value);
    const loom_value_id_t normalized_right =
        loom_symbolic_expr_assumption_source_value(context, frame->right_value);
    if (normalized_left == normalized_right) {
      --frame_count;
      continue;
    }
    if (!context->module || normalized_left >= context->module->values.count ||
        normalized_right >= context->module->values.count) {
      return iree_ok_status();
    }

    const loom_value_t* left =
        loom_module_value(context->module, normalized_left);
    const loom_value_t* right =
        loom_module_value(context->module, normalized_right);
    if (loom_value_is_block_arg(left) || loom_value_is_block_arg(right)) {
      return iree_ok_status();
    }
    const loom_op_t* left_op = loom_value_def_op(left);
    const loom_op_t* right_op = loom_value_def_op(right);
    if (!loom_symbolic_expr_ops_can_structurally_match(
            context->module, left_op, right_op, loom_value_def_index(left),
            loom_value_def_index(right))) {
      return iree_ok_status();
    }
    if (!loom_symbolic_semantic_match_pair_insert(pair_table, normalized_left,
                                                  normalized_right)) {
      --frame_count;
      continue;
    }

    --remaining_pair_count;
    frame->left_value = normalized_left;
    frame->right_value = normalized_right;
    frame->next_operand = 0;
    frame->left_op = left_op;
    frame->right_op = right_op;
  }

  *out_match = true;
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
  *out_match = loom_symbolic_value_constant_expression(&expression,
                                                       &expression_constant) &&
               expression_constant == constant;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Kernel-coordinate and arithmetic relations
//===----------------------------------------------------------------------===//

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
  return loom_symbolic_values_match(context, value_id, bound_value, out_match);
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

iree_status_t loom_symbolic_value_is_non_negative(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    bool* out_is_non_negative) {
  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_context_lookup_facts(context, value_id, &facts));
  *out_is_non_negative = loom_value_facts_is_non_negative(facts);
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_value_facts_are_positive(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    bool* out_is_positive) {
  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_context_lookup_facts(context, value_id, &facts));
  *out_is_positive = loom_value_facts_is_positive(facts);
  return iree_ok_status();
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

bool loom_symbolic_value_product_factors(
    const loom_symbolic_expr_context_t* context, loom_value_id_t product_value,
    loom_value_id_t* out_lhs, loom_value_id_t* out_rhs) {
  loom_value_id_t product_source =
      loom_symbolic_expr_assumption_source_value(context, product_value);
  const loom_op_t* product_op =
      loom_symbolic_expr_value_defining_op(context, product_source);
  return product_op &&
         loom_symbolic_expr_multiply_def(product_op, out_lhs, out_rhs);
}

static iree_status_t loom_symbolic_expr_value_matches_product(
    loom_symbolic_expr_context_t* context, loom_value_id_t product_value,
    loom_value_id_t left_factor, loom_value_id_t right_factor,
    bool* out_match) {
  *out_match = false;
  loom_value_id_t product_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t product_rhs = LOOM_VALUE_ID_INVALID;
  if (!loom_symbolic_value_product_factors(context, product_value, &product_lhs,
                                           &product_rhs)) {
    return iree_ok_status();
  }

  bool lhs_matches_left = false;
  bool rhs_matches_right = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, product_lhs, left_factor, &lhs_matches_left));
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, product_rhs, right_factor, &rhs_matches_right));
  if (lhs_matches_left && rhs_matches_right) {
    *out_match = true;
    return iree_ok_status();
  }

  bool lhs_matches_right = false;
  bool rhs_matches_left = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, product_lhs, right_factor, &lhs_matches_right));
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
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
  bool dividend_non_negative = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_is_non_negative(
      context, dividend, &dividend_non_negative));
  bool divisor_positive = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_value_facts_are_positive(
      context, divisor, &divisor_positive));
  bool bound_non_negative = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_is_non_negative(
      context, bound_value, &bound_non_negative));
  if (!dividend_non_negative || !divisor_positive || !bound_non_negative) {
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
    IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
        context, right_value, divisor, &right_matches_divisor));
    bool dividend_non_negative = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_value_is_non_negative(
        context, dividend, &dividend_non_negative));
    bool divisor_positive = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_value_facts_are_positive(
        context, divisor, &divisor_positive));
    if (right_matches_divisor && dividend_non_negative && divisor_positive) {
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
    IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
        context, left_value, divisor, &left_matches_divisor));
    bool dividend_non_negative = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_value_is_non_negative(
        context, dividend, &dividend_non_negative));
    bool divisor_positive = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_value_facts_are_positive(
        context, divisor, &divisor_positive));
    if (left_matches_divisor && dividend_non_negative && divisor_positive) {
      *out_matched = loom_symbolic_expr_remainder_bound_relation(
          relation, /*swapped=*/true, out_result);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Predicate relation proofs
//===----------------------------------------------------------------------===//

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
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, predicate_left, left_value, &ordered_left_match));
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, predicate_left, right_value, &swapped_right_match));

  bool ordered_right_match = false;
  bool swapped_left_match = false;
  switch ((loom_predicate_arg_tag_t)predicate->arg_tags[1]) {
    case LOOM_PRED_ARG_VALUE: {
      loom_value_id_t predicate_right = (loom_value_id_t)predicate->args[1];
      IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
          context, predicate_right, right_value, &ordered_right_match));
      IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
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
      return loom_symbolic_values_match(context, (loom_value_id_t)arg, value_id,
                                        out_match);
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
      if (loom_symbolic_value_constant_expression(&value_expression,
                                                  &value_constant) &&
          iree_checked_sub_i64(arg, value_constant, out_difference)) {
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
      !iree_checked_sub_i64(effective_upper_delta, 1, &effective_upper_delta)) {
    return false;
  }
  int64_t scaled_upper_delta = 0;
  if (!iree_checked_mul_i64(scale, effective_upper_delta,
                            &scaled_upper_delta)) {
    return false;
  }
  int64_t maximum_difference = 0;
  return iree_checked_add_i64(constant, scaled_upper_delta,
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
      !iree_checked_sub_i64(effective_upper_bound, 1, &effective_upper_bound)) {
    return false;
  }
  int64_t scaled_upper_bound = 0;
  if (!iree_checked_mul_i64(scale, effective_upper_bound,
                            &scaled_upper_bound)) {
    return false;
  }
  int64_t maximum_difference = 0;
  return iree_checked_add_i64(constant, scaled_upper_bound,
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
  uint8_t remaining_steps = LOOM_SYMBOLIC_VALUE_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0) {
    loom_symbolic_expr_identity_chain_step_t step = {0};
    if (!loom_symbolic_expr_identity_chain_step(
            context, current_value,
            LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_FOLLOW_INDEX_CASTS, &step)) {
      return iree_ok_status();
    }
    const loom_attribute_t predicates_attr = step.predicates_attr;
    for (uint16_t i = 0; i < predicates_attr.count; ++i) {
      IREE_RETURN_IF_ERROR(proof_fn(context, &predicates_attr.predicate_list[i],
                                    proof_user_data, out_matched, out_result));
      if (*out_matched) return iree_ok_status();
    }
    current_value = step.next_value;
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

iree_status_t loom_symbolic_value_prove_scaled_assumption(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_id_t positive_value, loom_value_id_t negative_value,
    int64_t scale, int64_t constant, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  const loom_symbolic_expr_scaled_le_proof_t proof = {
      .positive_relation_value = positive_value,
      .negative_relation_value = negative_value,
      .scale = scale,
      .constant = constant,
  };
  return loom_symbolic_expr_prove_identity_chain_predicates(
      context, start_value, loom_symbolic_expr_scaled_le_predicate_proof,
      &proof, out_matched, out_result);
}

iree_status_t loom_symbolic_value_prove_scaled_static_assumption(
    loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_id_t positive_value, int64_t scale, int64_t constant,
    bool* out_matched, loom_symbolic_proof_result_t* out_result) {
  const loom_symbolic_expr_scaled_static_le_proof_t proof = {
      .positive_relation_value = positive_value,
      .scale = scale,
      .constant = constant,
  };
  return loom_symbolic_expr_prove_identity_chain_predicates(
      context, start_value, loom_symbolic_expr_scaled_static_le_predicate_proof,
      &proof, out_matched, out_result);
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

iree_status_t loom_symbolic_value_prove_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t left_value,
    loom_value_id_t right_value, loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  const loom_condition_integer_relation_t queried_relation = {
      .relation = relation,
      .left =
          {
              .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
              .value_id = left_value,
          },
      .right =
          {
              .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
              .value_id = right_value,
          },
  };
  bool condition_result = false;
  if (loom_condition_fact_set_proves_integer_relation(
          context->condition_facts, context->fact_table, &queried_relation,
          &condition_result)) {
    *out_result =
        condition_result ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
    return iree_ok_status();
  }

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

//===----------------------------------------------------------------------===//
// Select conditions
//===----------------------------------------------------------------------===//

bool loom_symbolic_value_select_condition(
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

static bool loom_symbolic_value_predicate_arg_select_condition(
    const loom_symbolic_expr_context_t* context, loom_predicate_arg_tag_t tag,
    int64_t arg, loom_value_id_t* out_condition) {
  if (tag != LOOM_PRED_ARG_VALUE || arg < 0) return false;
  return loom_symbolic_value_select_condition(context, (loom_value_id_t)arg,
                                              out_condition);
}

static void loom_symbolic_value_append_select_condition(
    loom_value_id_t condition, loom_value_id_t* conditions,
    iree_host_size_t condition_capacity,
    iree_host_size_t* inout_condition_count) {
  if (condition == LOOM_VALUE_ID_INVALID ||
      *inout_condition_count >= condition_capacity) {
    return;
  }
  for (iree_host_size_t i = 0; i < *inout_condition_count; ++i) {
    if (conditions[i] == condition) return;
  }
  conditions[(*inout_condition_count)++] = condition;
}

static void loom_symbolic_value_collect_predicate_select_conditions(
    const loom_symbolic_expr_context_t* context,
    const loom_predicate_t* predicate, loom_value_id_t* conditions,
    iree_host_size_t condition_capacity,
    iree_host_size_t* inout_condition_count) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
    if (loom_symbolic_value_predicate_arg_select_condition(
            context, (loom_predicate_arg_tag_t)predicate->arg_tags[i],
            predicate->args[i], &condition)) {
      loom_symbolic_value_append_select_condition(
          condition, conditions, condition_capacity, inout_condition_count);
    }
  }
}

void loom_symbolic_value_collect_identity_chain_select_conditions(
    const loom_symbolic_expr_context_t* context, loom_value_id_t start_value,
    loom_value_id_t* conditions, iree_host_size_t condition_capacity,
    iree_host_size_t* inout_condition_count) {
  if (!context->module || start_value == LOOM_VALUE_ID_INVALID) return;

  loom_value_id_t current_value = start_value;
  uint8_t remaining_steps = LOOM_SYMBOLIC_VALUE_IDENTITY_CHAIN_LIMIT;
  while (remaining_steps-- > 0 && *inout_condition_count < condition_capacity) {
    loom_symbolic_expr_identity_chain_step_t step = {0};
    if (!loom_symbolic_expr_identity_chain_step(
            context, current_value,
            LOOM_SYMBOLIC_EXPR_IDENTITY_CHAIN_FOLLOW_INDEX_CASTS, &step)) {
      return;
    }
    const loom_attribute_t predicates_attr = step.predicates_attr;
    for (uint16_t i = 0; i < predicates_attr.count; ++i) {
      loom_symbolic_value_collect_predicate_select_conditions(
          context, &predicates_attr.predicate_list[i], conditions,
          condition_capacity, inout_condition_count);
    }
    current_value = step.next_value;
  }
}
