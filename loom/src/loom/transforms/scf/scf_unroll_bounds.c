// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_unroll_bounds.h"

#include "iree/base/internal/arena.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/predicate_facts.h"

loom_value_facts_t loom_scf_unroll_guarded_index_facts(
    loom_value_facts_t source_facts, loom_value_facts_t upper_facts) {
  const int64_t upper =
      upper_facts.range_hi == INT64_MIN ? INT64_MAX : upper_facts.range_hi - 1;
  return loom_value_facts_make(source_facts.range_lo,
                               iree_max(source_facts.range_hi, upper),
                               source_facts.known_divisor);
}

iree_status_t loom_scf_unroll_refine_indices(
    loom_rewriter_t* rewriter, loom_value_facts_t source_facts,
    loom_value_facts_t inferred_facts, loom_value_id_t lower_bound,
    loom_value_id_t upper_bound, uint32_t index_count, loom_value_id_t* indices,
    loom_location_id_t location, iree_arena_allocator_t* scratch_arena) {
  if (index_count == 0) {
    return iree_ok_status();
  }

  loom_predicate_t prototype[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY + 2];
  uint16_t predicates_per_index = loom_value_facts_export_integer_refinements(
      indices[0], inferred_facts, source_facts, prototype,
      LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY);
  if (lower_bound != LOOM_VALUE_ID_INVALID) {
    prototype[predicates_per_index++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_GE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_NONE},
        .args = {indices[0], lower_bound, 0},
    };
  }
  if (upper_bound != LOOM_VALUE_ID_INVALID) {
    prototype[predicates_per_index++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_LT,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                     LOOM_PRED_ARG_NONE},
        .args = {indices[0], upper_bound, 0},
    };
  }
  if (predicates_per_index == 0) {
    return iree_ok_status();
  }

  const uint16_t group_capacity =
      (uint16_t)iree_min(index_count, UINT16_MAX / predicates_per_index);
  const loom_type_t type = loom_module_value_type(rewriter->module, indices[0]);
  loom_type_t singleton_type = type;
  loom_type_t* types = &singleton_type;
  loom_predicate_t* predicates = prototype;
  if (group_capacity > 1) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, group_capacity, sizeof(*types), (void**)&types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, (iree_host_size_t)group_capacity * predicates_per_index,
        sizeof(*predicates), (void**)&predicates));
    for (uint16_t i = 0; i < group_capacity; ++i) {
      types[i] = type;
    }
  }

  iree_status_t status = iree_ok_status();
  for (uint32_t position = 0;
       position < index_count && iree_status_is_ok(status);) {
    const uint16_t count =
        (uint16_t)iree_min(index_count - position, group_capacity);
    for (uint16_t i = 0; i < count; ++i) {
      for (uint16_t j = 0; j < predicates_per_index; ++j) {
        loom_predicate_t predicate = prototype[j];
        predicate.args[0] = indices[position + i];
        predicates[i * predicates_per_index + j] = predicate;
      }
    }
    loom_op_t* assume = NULL;
    status = loom_index_assume_build(
        &rewriter->builder, indices + position, count, predicates,
        count * predicates_per_index, types, count, location, &assume);
    if (iree_status_is_ok(status)) {
      const loom_value_slice_t results = loom_index_assume_results(assume);
      for (uint16_t i = 0; i < count && iree_status_is_ok(status); ++i) {
        status = loom_rewriter_move_value_name(rewriter, indices[position + i],
                                               results.values[i]);
        indices[position + i] = results.values[i];
      }
    }
    position += count;
  }
  return status;
}

static iree_status_t loom_scf_unroll_bound_constant(
    loom_builder_t* builder, const loom_op_t* op, int64_t value,
    loom_type_t type, loom_value_id_t* out_value) {
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(value), type, op->location, &constant));
  *out_value = loom_index_constant_result(constant);
  return iree_ok_status();
}

// The ceiling-count formulation is cheaper when every intermediate and the
// unused terminal IV fit. Consume the existing source ranges once; no IR
// walk or derived bound reconstruction is needed.
static bool loom_scf_unroll_padded_split_fits(
    const loom_value_fact_table_t* facts, const loom_op_t* op, loom_type_t type,
    int64_t step) {
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_INDEX) {
    return false;
  }
  const loom_value_facts_t lower =
      loom_value_fact_table_lookup(facts, loom_scf_for_lower_bound(op));
  const loom_value_facts_t upper =
      loom_value_fact_table_lookup(facts, loom_scf_for_upper_bound(op));
  int64_t minimum_span = 0;
  int64_t maximum_span = 0;
  int64_t maximum_padded_span = 0;
  int64_t maximum_terminal = 0;
  if (!iree_checked_sub_i64(upper.range_lo, lower.range_hi, &minimum_span) ||
      !iree_checked_sub_i64(upper.range_hi, lower.range_lo, &maximum_span) ||
      !iree_checked_add_i64(iree_max(maximum_span, 0), step - 1,
                            &maximum_padded_span) ||
      !iree_checked_add_i64(upper.range_hi, step - 1, &maximum_terminal)) {
    return false;
  }
  const loom_value_facts_t intermediates = loom_value_facts_make(
      iree_min(minimum_span, lower.range_lo),
      iree_max(iree_max(maximum_padded_span, maximum_terminal), lower.range_hi),
      1);
  return loom_index_value_facts_fit_signed_target_carrier(
      &facts->context, LOOM_SCALAR_TYPE_INDEX, intermediates);
}

static iree_status_t loom_scf_unroll_build_padded_split(
    loom_builder_t* builder, const loom_op_t* op, int64_t step,
    uint32_t unroll_factor, loom_type_t index_type, loom_value_id_t scaled_step,
    loom_value_id_t* out_main_upper) {
  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(0), index_type, op->location, &zero_op));
  const loom_value_id_t zero = loom_index_constant_result(zero_op);

  loom_op_t* span_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_sub_build(
      builder, loom_scf_for_upper_bound(op), loom_scf_for_lower_bound(op),
      index_type, op->location, &span_op));
  loom_op_t* non_negative_span_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_max_build(builder, loom_index_sub_result(span_op), zero,
                           op->location, &non_negative_span_op));
  loom_value_id_t trip_count = loom_index_max_result(non_negative_span_op);
  if (step != 1) {
    loom_value_id_t step_minus_one = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scf_unroll_bound_constant(
        builder, op, step - 1, index_type, &step_minus_one));
    loom_op_t* padded_span_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_add_build(builder, trip_count,
                                              step_minus_one, index_type,
                                              op->location, &padded_span_op));
    loom_value_id_t step_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_scf_unroll_bound_constant(
        builder, op, step, index_type, &step_value));
    loom_op_t* trip_count_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_div_build(builder, loom_index_add_result(padded_span_op),
                             step_value, op->location, &trip_count_op));
    trip_count = loom_index_div_result(trip_count_op);
  }

  loom_value_id_t factor_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_bound_constant(
      builder, op, unroll_factor, index_type, &factor_value));
  loom_op_t* tile_count_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_div_build(builder, trip_count, factor_value,
                                            op->location, &tile_count_op));
  loom_op_t* main_span_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_mul_build(builder, loom_index_div_result(tile_count_op),
                           scaled_step, op->location, &main_span_op));
  loom_op_t* main_upper_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_add_build(builder, loom_scf_for_lower_bound(op),
                           loom_index_mul_result(main_span_op), index_type,
                           op->location, &main_upper_op));
  *out_main_upper = loom_index_add_result(main_upper_op);
  return iree_ok_status();
}

// Selects the narrowest integer carrier that preserves the unsigned distance
// between the clamped lower bound and upper bound. Subtraction in a fixed-width
// carrier retains that distance modulo its width even when either address does
// not fit the carrier's signed range. The remainder must also fit the address
// domain when converted back: index interprets i32 as signed while offset zero
// extends its bits.
static loom_type_t loom_scf_unroll_split_integer_type(
    const loom_value_fact_table_t* facts, const loom_op_t* op,
    loom_scalar_type_t address_type, int64_t tile_span,
    int64_t* out_tile_span_literal) {
  *out_tile_span_literal = tile_span;
  const int32_t target_bitwidth =
      loom_index_target_carrier_bitwidth(&facts->context, address_type);
  bool distance_fits_u32 = target_bitwidth > 0 && target_bitwidth <= 32;
  if (!distance_fits_u32) {
    const loom_value_facts_t lower =
        loom_value_fact_table_lookup(facts, loom_scf_for_lower_bound(op));
    const loom_value_facts_t upper =
        loom_value_fact_table_lookup(facts, loom_scf_for_upper_bound(op));
    int64_t maximum_distance = 0;
    distance_fits_u32 = upper.range_hi <= lower.range_lo ||
                        (iree_checked_sub_i64(upper.range_hi, lower.range_lo,
                                              &maximum_distance) &&
                         (uint64_t)maximum_distance <= UINT32_MAX);
  }

  const uint64_t maximum_tile_span =
      address_type == LOOM_SCALAR_TYPE_OFFSET ? UINT32_MAX : INT32_MAX;
  if (!distance_fits_u32 || (uint64_t)tile_span > maximum_tile_span) {
    return loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  }

  // Fixed-width constants use their signed spelling. Unsigned remainder
  // consumes the same i32 bit pattern for offset spans above INT32_MAX.
  *out_tile_span_literal =
      tile_span <= INT32_MAX ? tile_span : tile_span - (INT64_C(1) << 32);
  return loom_type_scalar(LOOM_SCALAR_TYPE_I32);
}

iree_status_t loom_scf_unroll_build_dynamic_split(
    loom_builder_t* builder, const loom_value_fact_table_t* facts,
    const loom_op_t* op, int64_t step, uint32_t factor,
    loom_value_id_t scaled_step, loom_scf_unroll_split_t* out_split) {
  const loom_value_id_t lower = loom_scf_for_lower_bound(op);
  const loom_value_id_t upper = loom_scf_for_upper_bound(op);
  const loom_type_t type = loom_module_value_type(builder->module, lower);
  const loom_value_facts_t lower_facts =
      loom_value_fact_table_lookup(facts, lower);
  const int64_t tile_span = step * factor;
  if (loom_scf_unroll_padded_split_fits(facts, op, type, step)) {
    const int64_t divisor =
        loom_value_facts_is_zero(lower_facts)
            ? tile_span
            : iree_math_gcd_i64(lower_facts.known_divisor, tile_span);
    out_split->facts =
        loom_value_facts_make(lower_facts.range_lo, INT64_MAX, divisor);
    return loom_scf_unroll_build_padded_split(builder, op, step, factor, type,
                                              scaled_step, &out_split->value);
  }

  // Clamp empty domains before subtraction. Fixed-width unsigned arithmetic
  // represents a full signed-address span without assigning it index facts.
  const bool is_offset =
      loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
  loom_op_t* nonempty = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      builder,
      is_offset ? LOOM_INDEX_CMP_PREDICATE_ULT : LOOM_INDEX_CMP_PREDICATE_SLT,
      lower, upper, op->location, &nonempty));
  loom_op_t* bounded_lower = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_select_build(builder, loom_index_cmp_result(nonempty), lower,
                            upper, type, op->location, &bounded_lower));
  const loom_value_id_t begin = loom_scf_select_result(bounded_lower);
  int64_t tile_span_literal = 0;
  const loom_type_t integer_type = loom_scf_unroll_split_integer_type(
      facts, op, loom_type_element_type(type), tile_span, &tile_span_literal);
  loom_op_t* lower_integer = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(builder, begin, type, integer_type,
                                             op->location, &lower_integer));
  loom_op_t* upper_integer = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(builder, upper, type, integer_type,
                                             op->location, &upper_integer));
  loom_op_t* distance = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_subi_build(builder, 0, loom_index_cast_result(upper_integer),
                             loom_index_cast_result(lower_integer),
                             integer_type, op->location, &distance));
  loom_op_t* tile_span_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_constant_build(builder, loom_attr_i64(tile_span_literal),
                                 integer_type, op->location, &tile_span_op));
  loom_op_t* remainder = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_remui_build(builder, loom_scalar_subi_result(distance),
                              loom_scalar_constant_result(tile_span_op),
                              integer_type, op->location, &remainder));
  loom_op_t* remainder_address = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      builder, loom_scalar_remui_result(remainder), integer_type, type,
      op->location, &remainder_address));
  const loom_value_id_t residual = loom_index_cast_result(remainder_address);

  // A remainder greater than the final lane's advance still contains a full
  // tile. It has no tail, so the original upper bound is its safe terminator.
  loom_value_id_t final_advance = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_scf_unroll_bound_constant(
      builder, op, step * (factor - 1), type, &final_advance));
  loom_op_t* has_tail = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cmp_build(builder, LOOM_INDEX_CMP_PREDICATE_ULE, residual,
                           final_advance, op->location, &has_tail));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_scf_unroll_bound_constant(builder, op, 0, type, &zero));
  loom_op_t* drop = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_select_build(builder, loom_index_cmp_result(has_tail), residual,
                            zero, type, op->location, &drop));
  loom_op_t* split = NULL;
  IREE_RETURN_IF_ERROR(loom_index_sub_build(builder, upper,
                                            loom_scf_select_result(drop), type,
                                            op->location, &split));
  const loom_value_id_t split_value = loom_index_sub_result(split);
  // The split producer owns these bounds. Preserve them across the unsigned
  // remainder arithmetic so tail memory origins consume the established proof.
  const loom_predicate_t predicates[] = {
      {.kind = LOOM_PREDICATE_GE,
       .arg_count = 2,
       .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                    LOOM_PRED_ARG_NONE},
       .args = {split_value, begin, 0}},
      {.kind = LOOM_PREDICATE_LE,
       .arg_count = 2,
       .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE,
                    LOOM_PRED_ARG_NONE},
       .args = {split_value, upper, 0}},
  };
  loom_op_t* bounded_split = NULL;
  IREE_RETURN_IF_ERROR(loom_index_assume_build(
      builder, &split_value, 1, predicates, IREE_ARRAYSIZE(predicates), &type,
      1, op->location, &bounded_split));
  out_split->value = loom_index_assume_results(bounded_split).values[0];
  // The unsigned remainder/select expression does not retain the source grid.
  // A split equal to upper can be off-grid when the tail is empty, so its
  // divisibility cannot be strengthened until the tail body is entered.
  const loom_value_facts_t upper_facts =
      loom_value_fact_table_lookup(facts, upper);
  out_split->facts = loom_value_facts_make(
      iree_min(lower_facts.range_lo, upper_facts.range_lo),
      upper_facts.range_hi, 1);
  return iree_ok_status();
}
