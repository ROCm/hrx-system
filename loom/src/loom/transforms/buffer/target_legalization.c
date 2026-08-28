// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/buffer/target_legalization.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"

static iree_status_t loom_buffer_legalize_build_offset_constant(
    loom_builder_t* builder, int64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      location, &op));
  *out_value = loom_index_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_scalar_constant(
    loom_builder_t* builder, loom_scalar_type_t scalar_type, int64_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, loom_attr_i64(value),
                                                  loom_type_scalar(scalar_type),
                                                  location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

// Builds the physical address of one byte in a bulk operation range.
static iree_status_t loom_buffer_legalize_build_byte_offset(
    loom_builder_t* builder, loom_value_id_t base_offset,
    loom_value_id_t relative_offset, loom_location_id_t location,
    loom_value_id_t* out_byte_offset) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_add_build(
      builder, base_offset, relative_offset,
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), location, &op));
  *out_byte_offset = loom_index_add_result(op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_byte_load(
    loom_builder_t* builder, loom_value_id_t source,
    loom_value_id_t source_offset, loom_value_id_t relative_offset,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_offset(
      builder, source_offset, relative_offset, location, &byte_offset));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_buffer_load_i8_u_build(builder, source, byte_offset, location, &op));
  *out_value = loom_buffer_load_i8_u_result(op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_byte_store(
    loom_builder_t* builder, loom_value_id_t value, loom_value_id_t target,
    loom_value_id_t target_offset, loom_value_id_t relative_offset,
    loom_location_id_t location) {
  loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_offset(
      builder, target_offset, relative_offset, location, &byte_offset));
  loom_op_t* op = NULL;
  return loom_buffer_store_i8_build(builder, value, target, byte_offset,
                                    location, &op);
}

static iree_status_t loom_buffer_legalize_build_copy_loop_body(
    loom_builder_t* builder, loom_op_t* loop, loom_value_id_t source,
    loom_value_id_t source_offset, loom_value_id_t target,
    loom_value_id_t target_offset, loom_location_id_t location) {
  const loom_value_id_t relative_offset =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_load(
      builder, source, source_offset, relative_offset, location, &value));
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_store(
      builder, value, target, target_offset, relative_offset, location));
  loom_op_t* yield_op = NULL;
  return loom_scf_yield_build(builder, /*values=*/NULL, /*value_count=*/0,
                              location, &yield_op);
}

static iree_status_t loom_buffer_legalize_copy(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 0, op->location, &zero));
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 1, op->location, &one));
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &rewriter->builder, /*build_flags=*/0, zero,
      loom_buffer_copy_byte_length(op), one, /*iter_args=*/NULL,
      /*iter_args_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, /*unroll_factor=*/LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &loop));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, loop, loom_scf_for_body(loop));
  iree_status_t status = loom_buffer_legalize_build_copy_loop_body(
      &rewriter->builder, loop, loom_buffer_copy_source(op),
      loom_buffer_copy_source_offset(op), loom_buffer_copy_target(op),
      loom_buffer_copy_target_offset(op), op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static loom_scalar_type_t loom_buffer_legalize_pattern_integer_type(
    loom_scalar_type_t pattern_type) {
  switch (loom_scalar_type_bitwidth(pattern_type)) {
    case 8:
      return LOOM_SCALAR_TYPE_I8;
    case 16:
      return LOOM_SCALAR_TYPE_I16;
    case 32:
      return LOOM_SCALAR_TYPE_I32;
    case 64:
      return LOOM_SCALAR_TYPE_I64;
    default:
      IREE_ASSERT_UNREACHABLE(
          "verified buffer.fill pattern must have a supported byte width");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_buffer_legalize_build_pattern_bits(
    loom_builder_t* builder, loom_value_id_t pattern,
    loom_scalar_type_t pattern_type, loom_location_id_t location,
    loom_scalar_type_t* out_working_type, loom_value_id_t* out_pattern_bits) {
  const loom_scalar_type_t integer_type =
      loom_buffer_legalize_pattern_integer_type(pattern_type);
  loom_value_id_t pattern_bits = pattern;
  if (loom_scalar_type_is_float(pattern_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
        builder, pattern, loom_type_scalar(pattern_type),
        loom_type_scalar(integer_type), location, &bitcast_op));
    pattern_bits = loom_scalar_bitcast_result(bitcast_op);
  }

  const uint32_t pattern_bitwidth = loom_scalar_type_bitwidth(integer_type);
  if (pattern_bitwidth < 32) {
    loom_op_t* extend_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        builder, pattern_bits, loom_type_scalar(integer_type),
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &extend_op));
    pattern_bits = loom_scalar_extui_result(extend_op);

    // Repeat a two-byte pattern across the i32 working value so rotating the
    // working value by one byte preserves the authored pattern period.
    if (pattern_bitwidth == 16) {
      loom_value_id_t halfword_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
          builder, LOOM_SCALAR_TYPE_I32, 16, location, &halfword_shift));
      loom_op_t* high_half_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
          builder, /*instance_flags=*/0, pattern_bits, halfword_shift,
          loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &high_half_op));
      loom_op_t* repeated_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_ori_build(
          builder, pattern_bits, loom_scalar_shli_result(high_half_op),
          loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &repeated_op));
      pattern_bits = loom_scalar_ori_result(repeated_op);
    }

    *out_working_type = LOOM_SCALAR_TYPE_I32;
    *out_pattern_bits = pattern_bits;
    return iree_ok_status();
  }

  *out_working_type = integer_type;
  *out_pattern_bits = pattern_bits;
  return iree_ok_status();
}

// Projects the low byte of the rotating pattern into the canonical i32 byte
// carrier used by buffer.store.i8. The store consumes only the low eight bits,
// so an i32 working value needs no separate mask or truncate operation.
static iree_status_t loom_buffer_legalize_build_pattern_byte(
    loom_builder_t* builder, loom_value_id_t pattern_bits,
    loom_scalar_type_t working_type, loom_location_id_t location,
    loom_value_id_t* out_byte) {
  if (working_type == LOOM_SCALAR_TYPE_I32) {
    *out_byte = pattern_bits;
    return iree_ok_status();
  }

  loom_op_t* carrier_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
      builder, pattern_bits, loom_type_scalar(working_type),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &carrier_op));
  *out_byte = loom_scalar_trunci_result(carrier_op);
  return iree_ok_status();
}

// Rotates a multi-byte working pattern right by one byte. Both shifts have
// compile-time-constant amounts, which keeps the reference legalization in the
// scalar baseline shared by targets without a general variable-shift form.
static iree_status_t loom_buffer_legalize_build_rotated_pattern(
    loom_builder_t* builder, loom_value_id_t pattern_bits,
    loom_scalar_type_t working_type, loom_value_id_t byte_shift,
    loom_value_id_t wrap_shift, loom_location_id_t location,
    loom_value_id_t* out_rotated_pattern) {
  const loom_type_t working_loom_type = loom_type_scalar(working_type);
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(
      builder, pattern_bits, byte_shift, working_loom_type, location, &low_op));
  loom_op_t* high_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
      builder, /*instance_flags=*/0, pattern_bits, wrap_shift,
      working_loom_type, location, &high_op));
  loom_op_t* rotated_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_ori_build(builder, loom_scalar_shrui_result(low_op),
                            loom_scalar_shli_result(high_op), working_loom_type,
                            location, &rotated_op));
  *out_rotated_pattern = loom_scalar_ori_result(rotated_op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_fill_loop_body(
    loom_builder_t* builder, loom_op_t* loop, loom_value_id_t target,
    loom_value_id_t target_offset, loom_value_id_t pattern_bits,
    loom_scalar_type_t working_type, loom_value_id_t byte_shift,
    loom_value_id_t wrap_shift, loom_location_id_t location) {
  const loom_region_t* body = loom_scf_for_body(loop);
  const loom_value_id_t relative_offset = loom_region_entry_arg_id(body, 0);
  const bool rotates_pattern = byte_shift != LOOM_VALUE_ID_INVALID;
  const loom_value_id_t current_pattern =
      rotates_pattern ? loom_region_entry_arg_id(body, 1) : pattern_bits;
  loom_value_id_t byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_pattern_byte(
      builder, current_pattern, working_type, location, &byte));
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_store(
      builder, byte, target, target_offset, relative_offset, location));

  if (!rotates_pattern) {
    loom_op_t* yield_op = NULL;
    return loom_scf_yield_build(builder, /*values=*/NULL, /*value_count=*/0,
                                location, &yield_op);
  }

  loom_value_id_t next_pattern = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_rotated_pattern(
      builder, current_pattern, working_type, byte_shift, wrap_shift, location,
      &next_pattern));
  loom_op_t* yield_op = NULL;
  return loom_scf_yield_build(builder, &next_pattern, 1, location, &yield_op);
}

static iree_status_t loom_buffer_legalize_fill(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);

  const loom_scalar_type_t pattern_type = loom_type_element_type(
      loom_module_value_type(context->module, loom_buffer_fill_pattern(op)));
  loom_scalar_type_t working_type = LOOM_SCALAR_TYPE_NONE;
  loom_value_id_t pattern_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_pattern_bits(
      &rewriter->builder, loom_buffer_fill_pattern(op), pattern_type,
      op->location, &working_type, &pattern_bits));

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 0, op->location, &zero));
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 1, op->location, &one));
  loom_value_id_t byte_shift = LOOM_VALUE_ID_INVALID;
  loom_value_id_t wrap_shift = LOOM_VALUE_ID_INVALID;
  const int64_t pattern_byte_count =
      loom_scalar_type_bitwidth(pattern_type) / 8;
  if (pattern_byte_count > 1) {
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
        &rewriter->builder, working_type, 8, op->location, &byte_shift));
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
        &rewriter->builder, working_type,
        loom_scalar_type_bitwidth(working_type) - 8, op->location,
        &wrap_shift));
  }

  const loom_value_id_t initial_pattern =
      pattern_byte_count > 1 ? pattern_bits : LOOM_VALUE_ID_INVALID;
  const loom_value_id_t* iter_args =
      initial_pattern == LOOM_VALUE_ID_INVALID ? NULL : &initial_pattern;
  const iree_host_size_t iter_arg_count = iter_args == NULL ? 0 : 1;
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &rewriter->builder, /*build_flags=*/0, zero,
      loom_buffer_fill_byte_length(op), one, iter_args, iter_arg_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*unroll_factor=*/LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      /*unroll_schedule=*/0, op->location, &loop));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, loop, loom_scf_for_body(loop));
  iree_status_t status = loom_buffer_legalize_build_fill_loop_body(
      &rewriter->builder, loop, loom_buffer_fill_target(op),
      loom_buffer_fill_target_offset(op), pattern_bits, working_type,
      byte_shift, wrap_shift, op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

typedef struct loom_buffer_compare_constants_t {
  // Canonical equal comparison result.
  loom_value_id_t equal;
  // Canonical less-than comparison result.
  loom_value_id_t less;
  // Canonical greater-than comparison result.
  loom_value_id_t greater;
} loom_buffer_compare_constants_t;

static iree_status_t loom_buffer_legalize_build_compare_byte_order(
    loom_builder_t* builder, loom_value_id_t lhs, loom_value_id_t rhs,
    const loom_buffer_compare_constants_t* constants,
    loom_location_id_t location, loom_value_id_t* out_order) {
  loom_op_t* equal_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, lhs, rhs, location, &equal_op));
  loom_op_t* less_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_ULT, lhs, rhs, location, &less_op));
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* unequal_order_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, loom_scalar_cmpi_result(less_op), constants->less,
      constants->greater, i32_type, location, &unequal_order_op));
  loom_op_t* byte_order_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, loom_scalar_cmpi_result(equal_op), constants->equal,
      loom_scf_select_result(unequal_order_op), i32_type, location,
      &byte_order_op));
  *out_order = loom_scf_select_result(byte_order_op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_compare_before_region(
    loom_builder_t* builder, loom_op_t* loop, loom_value_id_t byte_length,
    const loom_buffer_compare_constants_t* constants,
    loom_location_id_t location) {
  const loom_region_t* before = loom_scf_while_before(loop);
  const loom_value_id_t relative_offset = loom_region_entry_arg_id(before, 0);
  const loom_value_id_t current_order = loom_region_entry_arg_id(before, 1);

  loom_op_t* in_range_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      builder, LOOM_INDEX_CMP_PREDICATE_ULT, relative_offset, byte_length,
      location, &in_range_op));
  loom_op_t* unresolved_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, current_order, constants->equal,
      location, &unresolved_op));
  loom_op_t* continue_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      builder, loom_index_cmp_result(in_range_op),
      loom_scalar_cmpi_result(unresolved_op),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), location, &continue_op));

  const loom_value_id_t forwarded[] = {relative_offset, current_order};
  loom_op_t* condition_op = NULL;
  return loom_scf_condition_build(builder, loom_scalar_andi_result(continue_op),
                                  forwarded, IREE_ARRAYSIZE(forwarded),
                                  location, &condition_op);
}

static iree_status_t loom_buffer_legalize_build_compare_after_region(
    loom_builder_t* builder, loom_op_t* loop, loom_value_id_t lhs,
    loom_value_id_t lhs_offset, loom_value_id_t rhs, loom_value_id_t rhs_offset,
    loom_value_id_t one_offset,
    const loom_buffer_compare_constants_t* constants,
    loom_location_id_t location) {
  const loom_region_t* after = loom_scf_while_after(loop);
  const loom_value_id_t relative_offset = loom_region_entry_arg_id(after, 0);
  loom_value_id_t lhs_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_load(
      builder, lhs, lhs_offset, relative_offset, location, &lhs_byte));
  loom_value_id_t rhs_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_load(
      builder, rhs, rhs_offset, relative_offset, location, &rhs_byte));
  loom_value_id_t next_order = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_compare_byte_order(
      builder, lhs_byte, rhs_byte, constants, location, &next_order));
  loom_op_t* next_offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_add_build(
      builder, relative_offset, one_offset,
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), location, &next_offset_op));

  const loom_value_id_t yielded[] = {loom_index_add_result(next_offset_op),
                                     next_order};
  loom_op_t* yield_op = NULL;
  return loom_scf_yield_build(builder, yielded, IREE_ARRAYSIZE(yielded),
                              location, &yield_op);
}

static iree_status_t loom_buffer_legalize_compare(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t zero_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 0, op->location, &zero_offset));
  loom_value_id_t one_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 1, op->location, &one_offset));
  loom_buffer_compare_constants_t constants = {0};
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      &rewriter->builder, LOOM_SCALAR_TYPE_I32, 0, op->location,
      &constants.equal));
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      &rewriter->builder, LOOM_SCALAR_TYPE_I32, -1, op->location,
      &constants.less));
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      &rewriter->builder, LOOM_SCALAR_TYPE_I32, 1, op->location,
      &constants.greater));

  const loom_value_id_t iter_args[] = {zero_offset, constants.equal};
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_while_build(
      &rewriter->builder, iter_args, IREE_ARRAYSIZE(iter_args),
      /*tied_results=*/NULL, /*tied_result_count=*/0, op->location, &loop));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, loop, loom_scf_while_before(loop));
  iree_status_t status = loom_buffer_legalize_build_compare_before_region(
      &rewriter->builder, loop, loom_buffer_compare_byte_length(op), &constants,
      op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  saved_ip = loom_builder_enter_region(&rewriter->builder, loop,
                                       loom_scf_while_after(loop));
  status = loom_buffer_legalize_build_compare_after_region(
      &rewriter->builder, loop, loom_buffer_compare_lhs(op),
      loom_buffer_compare_lhs_offset(op), loom_buffer_compare_rhs(op),
      loom_buffer_compare_rhs_offset(op), one_offset, &constants, op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  const loom_value_id_t order = loom_scf_while_results(loop).values[1];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &order, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &order, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  switch (op->kind) {
    case LOOM_OP_BUFFER_COPY:
      return loom_buffer_legalize_copy(context, op, out_result);
    case LOOM_OP_BUFFER_FILL:
      return loom_buffer_legalize_fill(context, op, out_result);
    case LOOM_OP_BUFFER_COMPARE:
      return loom_buffer_legalize_compare(context, op, out_result);
    default:
      return iree_ok_status();
  }
}

static const loom_target_legalizer_entry_t kBufferLegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_BUFFER_COPY,
        .legalize = loom_buffer_legalize,
    },
    {
        .root_kind = LOOM_OP_BUFFER_FILL,
        .legalize = loom_buffer_legalize,
    },
    {
        .root_kind = LOOM_OP_BUFFER_COMPARE,
        .legalize = loom_buffer_legalize,
    },
};

static const loom_target_legalizer_provider_t kBufferLegalizerProvider = {
    .name = IREE_SVL("buffer"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .entries = kBufferLegalizerEntries,
    .entry_count = IREE_ARRAYSIZE(kBufferLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_buffer_target_legalizer_provider(
    void) {
  return &kBufferLegalizerProvider;
}
