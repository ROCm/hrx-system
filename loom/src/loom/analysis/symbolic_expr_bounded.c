// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr_bounded.h"

#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/adaptive_sort.h"

static bool loom_symbolic_expr_bounded_exact_integer_facts(
    loom_value_facts_t facts, int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_symbolic_expr_bounded_term_less(
    const loom_symbolic_term_t* lhs, const loom_symbolic_term_t* rhs) {
  return lhs->value_id < rhs->value_id;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_symbolic_expr_bounded_sort_terms,
                          loom_symbolic_term_t,
                          loom_symbolic_expr_bounded_term_less)

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
  if (loom_symbolic_expr_bounded_exact_integer_facts(
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
  if (!iree_checked_add_i64(*inout_constant, scaled_value, &new_constant)) {
    return false;
  }
  *inout_constant = new_constant;
  return true;
}

static bool loom_symbolic_expr_bounded_accumulate_constant(
    int64_t value, int64_t coefficient, int64_t* inout_constant) {
  int64_t scaled_value = 0;
  return iree_checked_mul_i64(value, coefficient, &scaled_value) &&
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
    if (!iree_checked_add_i64(terms[insert_index].coefficient, coefficient,
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
  if (term_count > 1) {
    loom_symbolic_expr_bounded_sort_terms(terms, term_count);
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
      if (!iree_checked_add_i64(coefficient, terms[read_index].coefficient,
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
    if (!iree_checked_mul_i64(coefficient, left_constant,
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
    if (!iree_checked_mul_i64(coefficient, right_constant,
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
  if (!iree_checked_mul_i64(coefficient, INT64_C(1) << shift_amount,
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
  if (!iree_checked_sub_i64(*inout_constant, frame->assume.constant,
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
  if (!iree_checked_mul_i64(source_constant, frame->assume.coefficient,
                            &scaled_constant) ||
      !iree_checked_add_i64(frame->assume.constant, scaled_constant,
                            inout_constant)) {
    return false;
  }
  for (iree_host_size_t i = frame->assume.term_count; i < *inout_raw_term_count;
       ++i) {
    int64_t scaled_coefficient = 0;
    if (!iree_checked_mul_i64(raw_terms[i].coefficient,
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
        handled = iree_checked_sub_i64(0, coefficient, &rhs_coefficient) &&
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
      case LOOM_OP_INDEX_MADD: {
        int64_t madd_a_constant = 0;
        int64_t madd_b_constant = 0;
        if (!loom_symbolic_expr_bounded_exact_i64(
                module, fact_table, loom_index_madd_a(op), &madd_a_constant) &&
            !loom_symbolic_expr_bounded_exact_i64(
                module, fact_table, loom_index_madd_b(op), &madd_b_constant)) {
          handled = loom_symbolic_expr_bounded_raw_append_term(
              raw_terms, IREE_ARRAYSIZE(raw_terms), &raw_term_count, value_id,
              value_id, coefficient);
        } else {
          handled = loom_symbolic_expr_bounded_push_expand(
                        frames, &frame_count, loom_index_madd_c(op),
                        coefficient, next_depth) &&
                    loom_symbolic_expr_bounded_push_mul(
                        module, fact_table, value_id, loom_index_madd_a(op),
                        loom_index_madd_b(op), coefficient, next_depth, frames,
                        &frame_count, raw_terms, IREE_ARRAYSIZE(raw_terms),
                        &raw_term_count);
        }
        break;
      }
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
        handled = iree_checked_sub_i64(0, coefficient, &rhs_coefficient) &&
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
        handled = iree_checked_sub_i64(0, coefficient, &negated_coefficient) &&
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
