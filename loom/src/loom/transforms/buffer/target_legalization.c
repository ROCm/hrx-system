// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/buffer/target_legalization.h"

#include <stdint.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

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

// Keep the bias in the physical offset domain so the mask conversion preserves
// its unsigned range independently of the target's logical index width.
static iree_status_t loom_buffer_legalize_align_offset(
    loom_builder_t* builder, loom_scalar_type_t carrier_type,
    loom_value_id_t value, int64_t alignment, loom_location_id_t location,
    loom_value_id_t* out_value) {
  if (alignment == 1) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_value_id_t bias = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      builder, alignment - 1, location, &bias));
  loom_op_t* op = NULL;
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_type_t integer_type = loom_type_scalar(carrier_type);
  IREE_RETURN_IF_ERROR(
      loom_index_add_build(builder, value, bias, offset_type, location, &op));
  IREE_RETURN_IF_ERROR(loom_index_cast_build(builder, loom_index_add_result(op),
                                             offset_type, integer_type,
                                             location, &op));
  const loom_value_id_t biased_value = loom_index_cast_result(op);
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      builder, carrier_type, -alignment, location, &mask));
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(builder, biased_value, mask,
                                              integer_type, location, &op));
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(builder, loom_scalar_andi_result(op), integer_type,
                            offset_type, location, &op));
  *out_value = loom_index_cast_result(op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_pack(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_value_slice_t lengths = loom_buffer_pack_byte_lengths(op);
  const loom_attribute_t alignments = loom_buffer_pack_minimum_alignments(op);
  const uint32_t offset_bitwidth =
      loom_target_legalization_context_bundle(context)
          ->snapshot->offset_bitwidth;
  const loom_scalar_type_t carrier_type =
      offset_bitwidth <= 32 ? LOOM_SCALAR_TYPE_I32 : LOOM_SCALAR_TYPE_I64;
  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  loom_value_id_t end = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      builder, 0, op->location, &end));
  int64_t slab_alignment = 1;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < lengths.count && iree_status_is_ok(status); ++i) {
    const int64_t alignment = alignments.i64_array[i];
    if (alignment > slab_alignment) {
      slab_alignment = alignment;
    }
    // Zero already satisfies every alignment for the first range.
    if (i != 0) {
      status = loom_buffer_legalize_align_offset(builder, carrier_type, end,
                                                 alignment, op->location, &end);
    }
    if (iree_status_is_ok(status)) {
      replacements[i + 1] = end;
      loom_op_t* addition = NULL;
      status = loom_index_add_build(builder, end, lengths.values[i],
                                    loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                    op->location, &addition);
      if (iree_status_is_ok(status)) {
        end = loom_index_add_result(addition);
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_align_offset(
      builder, carrier_type, end, slab_alignment, op->location,
      &replacements[0]));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, op->result_count));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
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

// Selects transfers from producer-retained memory-space and alignment facts.
// No stronger alignment requirement is introduced by the generated views.
static uint8_t loom_buffer_legalize_transfer_widths(
    const loom_target_legalization_context_t* context, loom_value_id_t buffer,
    loom_value_id_t offset) {
  const loom_target_bulk_memory_policy_t policy = context->bulk_memory_policy;
  if ((policy.scalar_byte_widths | policy.vector_byte_widths) == 0) {
    return 1;
  }
  uint64_t alignment = 1;
  loom_value_fact_buffer_reference_t reference = {0};
  if (context->fact_table != NULL &&
      loom_value_facts_query_buffer_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table, buffer),
          &reference)) {
    const loom_value_facts_t offset_facts =
        loom_value_fact_table_lookup(context->fact_table, offset);
    alignment = reference.minimum_alignment;
    if (!loom_value_facts_is_zero(offset_facts)) {
      alignment = iree_math_gcd_u64(alignment, offset_facts.known_divisor);
    }
  }
  if (!(policy.memory_spaces & (1u << reference.memory_space))) {
    return 1;
  }
  uint8_t widths = 1;
  for (uint32_t width = 2; width <= 64; width *= 2) {
    const uint32_t required_alignment = iree_min(
        width, (policy.scalar_byte_widths & width) ? policy.scalar_alignment
                                                   : policy.vector_alignment);
    if (((policy.scalar_byte_widths | policy.vector_byte_widths) & width) &&
        alignment >= required_alignment) {
      widths |= width;
    }
  }
  return widths;
}

static loom_type_t loom_buffer_legalize_transfer_type(
    const loom_target_bulk_memory_policy_t* policy, uint32_t width) {
  if (policy->vector_byte_widths & width) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                               width / 4, 0);
  }
  return loom_type_scalar(width == 8   ? LOOM_SCALAR_TYPE_I64
                          : width == 4 ? LOOM_SCALAR_TYPE_I32
                          : width == 2 ? LOOM_SCALAR_TYPE_I16
                                       : LOOM_SCALAR_TYPE_I8);
}

static iree_status_t loom_buffer_legalize_build_transfer_view(
    loom_builder_t* builder, loom_value_id_t buffer, loom_value_id_t offset,
    loom_value_id_t relative_offset, loom_type_t transfer_type,
    loom_location_id_t location, loom_value_id_t* out_view) {
  loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_offset(
      builder, offset, relative_offset, location, &byte_offset));
  const uint32_t lane_count =
      loom_type_is_vector(transfer_type)
          ? loom_type_dim_static_size_at(transfer_type, 0)
          : 1;
  loom_type_t view_type = loom_type_view_with_alignment(
      loom_type_shaped_1d(LOOM_TYPE_VIEW, loom_type_element_type(transfer_type),
                          lane_count, 0),
      1);
  loom_op_t* view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(builder, buffer, byte_offset,
                                              view_type, location, &view_op));
  *out_view = loom_buffer_view_result(view_op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_load(
    loom_builder_t* builder, loom_value_id_t source,
    loom_value_id_t source_offset, loom_value_id_t relative_offset,
    loom_type_t transfer_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  if (loom_type_equal(transfer_type, loom_type_scalar(LOOM_SCALAR_TYPE_I8))) {
    return loom_buffer_legalize_build_byte_load(
        builder, source, source_offset, relative_offset, location, out_value);
  }
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_transfer_view(
      builder, source, source_offset, relative_offset, transfer_type, location,
      &view));
  const int64_t origin = 0;
  loom_op_t* load = NULL;
  if (loom_type_is_vector(transfer_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_load_build(builder, 0, 0, view, NULL, 0,
                                                &origin, 1, 0, 0, transfer_type,
                                                location, &load));
    *out_value = loom_vector_load_result(load);
  } else {
    IREE_RETURN_IF_ERROR(loom_view_load_build(builder, 0, 0, view, NULL, 0,
                                              &origin, 1, 0, 0, transfer_type,
                                              location, &load));
    *out_value = loom_view_load_result(load);
  }
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_store(
    loom_builder_t* builder, loom_value_id_t value, loom_value_id_t target,
    loom_value_id_t target_offset, loom_value_id_t relative_offset,
    loom_type_t transfer_type, loom_location_id_t location) {
  if (loom_type_equal(transfer_type, loom_type_scalar(LOOM_SCALAR_TYPE_I8))) {
    return loom_buffer_legalize_build_byte_store(
        builder, value, target, target_offset, relative_offset, location);
  }
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_transfer_view(
      builder, target, target_offset, relative_offset, transfer_type, location,
      &view));
  const int64_t origin = 0;
  loom_op_t* store = NULL;
  if (loom_type_is_vector(transfer_type)) {
    return loom_vector_store_build(builder, 0, 0, value, view, NULL, 0, &origin,
                                   1, 0, 0, location, &store);
  }
  return loom_view_store_build(builder, 0, 0, value, view, NULL, 0, &origin, 1,
                               0, 0, location, &store);
}

static iree_status_t loom_buffer_legalize_build_copy_chunk(
    loom_builder_t* builder, loom_value_id_t relative_offset,
    loom_type_t transfer_type, loom_value_id_t source,
    loom_value_id_t source_offset, loom_value_id_t target,
    loom_value_id_t target_offset, loom_location_id_t location) {
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_load(
      builder, source, source_offset, relative_offset, transfer_type, location,
      &value));
  return loom_buffer_legalize_build_store(builder, value, target, target_offset,
                                          relative_offset, transfer_type,
                                          location);
}

// Round the physical offset in a carrier that preserves its unsigned range.
static iree_status_t loom_buffer_legalize_build_aligned_end(
    loom_target_legalization_context_t* context, loom_value_id_t byte_length,
    uint32_t width, loom_location_id_t location, loom_value_id_t* out_end) {
  if (width == 1) {
    *out_end = byte_length;
    return iree_ok_status();
  }
  const loom_value_facts_t length_facts =
      loom_value_fact_table_lookup(context->fact_table, byte_length);
  if (loom_value_facts_divisible_by(length_facts, width)) {
    *out_end = byte_length;
    return iree_ok_status();
  }
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const bool wide_length =
      !loom_value_facts_fit_signed_bit_count(length_facts, 32);
  const loom_scalar_type_t scalar_type =
      wide_length ? LOOM_SCALAR_TYPE_I64 : LOOM_SCALAR_TYPE_I32;
  const loom_type_t bits_type = loom_type_scalar(scalar_type);
  loom_op_t* bits = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(builder, byte_length, offset_type,
                                             bits_type, location, &bits));
  int64_t mask_value = -(int64_t)width;
  const int32_t offset_bitwidth =
      loom_target_legalization_context_bundle(context)
          ->snapshot->offset_bitwidth;
  if (offset_bitwidth < loom_scalar_type_bitwidth(scalar_type)) {
    // Make native offset wrapping explicit in the arithmetic. The input may
    // retain a wider source domain; its facts must not be narrowed to justify
    // the conversion of this result back into the physical offset carrier.
    mask_value &= (INT64_C(1) << offset_bitwidth) - 1;
  }
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      builder, scalar_type, mask_value, location, &mask));
  loom_op_t* masked_bits = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_andi_build(builder, loom_index_cast_result(bits), mask,
                             bits_type, location, &masked_bits));
  loom_op_t* masked_offset = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(builder, loom_scalar_andi_result(masked_bits),
                            bits_type, offset_type, location, &masked_offset));
  *out_end = loom_index_cast_result(masked_offset);
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

// Fill patterns are loop invariants. Small patterns repeat through an i32
// carrier; i64 patterns retain their full raw payload.
typedef struct loom_buffer_fill_pattern_t {
  // Repeated raw bits, with no floating-point arithmetic or conversion.
  loom_value_id_t bits;
  // Integer carrier containing at least one complete pattern period.
  loom_scalar_type_t type;
  // Authored repetition period in bytes.
  uint32_t byte_count;
} loom_buffer_fill_pattern_t;

static iree_status_t loom_buffer_legalize_build_integer_cast(
    loom_builder_t* builder, loom_value_id_t value, loom_scalar_type_t source,
    loom_scalar_type_t target, loom_location_id_t location,
    loom_value_id_t* out_value) {
  if (source == target) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_op_t* cast = NULL;
  if (loom_scalar_type_bitwidth(source) < loom_scalar_type_bitwidth(target)) {
    IREE_RETURN_IF_ERROR(
        loom_scalar_extui_build(builder, value, loom_type_scalar(source),
                                loom_type_scalar(target), location, &cast));
    *out_value = loom_scalar_extui_result(cast);
  } else {
    IREE_RETURN_IF_ERROR(
        loom_scalar_trunci_build(builder, value, loom_type_scalar(source),
                                 loom_type_scalar(target), location, &cast));
    *out_value = loom_scalar_trunci_result(cast);
  }
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_repeated_bits(
    loom_builder_t* builder, loom_value_id_t value, loom_scalar_type_t type,
    uint32_t period_bits, loom_location_id_t location,
    loom_value_id_t* out_value) {
  iree_status_t status = iree_ok_status();
  for (uint32_t shift = period_bits;
       shift < loom_scalar_type_bitwidth(type) && iree_status_is_ok(status);
       shift *= 2) {
    loom_value_id_t amount = LOOM_VALUE_ID_INVALID;
    status = loom_buffer_legalize_build_scalar_constant(builder, type, shift,
                                                        location, &amount);
    loom_op_t* high = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_scalar_shli_build(builder, 0, value, amount,
                                      loom_type_scalar(type), location, &high);
    }
    loom_op_t* repeated = NULL;
    if (iree_status_is_ok(status)) {
      status =
          loom_scalar_ori_build(builder, value, loom_scalar_shli_result(high),
                                loom_type_scalar(type), location, &repeated);
    }
    if (iree_status_is_ok(status)) {
      value = loom_scalar_ori_result(repeated);
    }
  }
  *out_value = value;
  return status;
}

static iree_status_t loom_buffer_legalize_prepare_fill(
    loom_builder_t* builder, loom_value_id_t pattern,
    loom_scalar_type_t pattern_type, loom_location_id_t location,
    loom_buffer_fill_pattern_t* out_pattern) {
  const loom_scalar_type_t integer_type =
      loom_buffer_legalize_pattern_integer_type(pattern_type);
  if (loom_scalar_type_is_float(pattern_type)) {
    loom_op_t* bitcast = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
        builder, pattern, loom_type_scalar(pattern_type),
        loom_type_scalar(integer_type), location, &bitcast));
    pattern = loom_scalar_bitcast_result(bitcast);
  }
  out_pattern->byte_count = loom_scalar_type_bitwidth(integer_type) / 8;
  out_pattern->type = out_pattern->byte_count == 8 ? LOOM_SCALAR_TYPE_I64
                                                   : LOOM_SCALAR_TYPE_I32;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_integer_cast(
      builder, pattern, integer_type, out_pattern->type, location, &pattern));
  return loom_buffer_legalize_build_repeated_bits(
      builder, pattern, out_pattern->type,
      out_pattern->byte_count == 1 ? 32 : out_pattern->byte_count * 8, location,
      &out_pattern->bits);
}

// Selects the initial pattern at a range's byte phase. Suffix ranges may
// start partway through a repetition; narrow loops carry subsequent rotations.
static iree_status_t loom_buffer_legalize_build_fill_bits(
    loom_builder_t* builder, const loom_buffer_fill_pattern_t* pattern,
    uint32_t start_alignment, loom_value_id_t range_start,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_value_id_t bits = pattern->bits;
  const loom_type_t working_type = loom_type_scalar(pattern->type);
  if (start_alignment != 0 && start_alignment < pattern->byte_count) {
    loom_op_t* position = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cast_build(
        builder, range_start, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
        working_type, location, &position));
    loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
        builder, pattern->type, pattern->byte_count - 1, location, &mask));
    loom_op_t* phase = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_andi_build(builder, loom_index_cast_result(position), mask,
                               working_type, location, &phase));
    loom_value_id_t three = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
        builder, pattern->type, 3, location, &three));
    loom_op_t* shift = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_shli_build(builder, 0, loom_scalar_andi_result(phase),
                               three, working_type, location, &shift));
    loom_op_t* shifted = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_shrui_build(builder, bits, loom_scalar_shli_result(shift),
                                working_type, location, &shifted));
    bits = loom_scalar_shrui_result(shifted);
  }
  *out_value = bits;
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_fill_value(
    loom_builder_t* builder, const loom_buffer_fill_pattern_t* pattern,
    loom_value_id_t bits, loom_type_t transfer_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  uint32_t repeated_bits =
      pattern->byte_count == 1 ? 8 : loom_scalar_type_bitwidth(pattern->type);
  if (loom_type_is_vector(transfer_type)) {
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_repeated_bits(
        builder, bits, pattern->type, repeated_bits, location, &bits));
    const loom_type_t splat_type =
        loom_type_shaped_1d(LOOM_TYPE_VECTOR, pattern->type,
                            loom_type_dim_static_size_at(transfer_type, 0) *
                                32 / loom_scalar_type_bitwidth(pattern->type),
                            0);
    loom_op_t* splat = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_splat_build(builder, bits, splat_type, location, &splat));
    loom_op_t* bitcast = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder, loom_vector_splat_result(splat), splat_type, transfer_type,
        location, &bitcast));
    *out_value = loom_vector_bitcast_result(bitcast);
    return iree_ok_status();
  }
  // Raw byte stores take the low eight bits of an i32 carrier.
  loom_scalar_type_t scalar_type = loom_type_element_type(transfer_type);
  if (scalar_type == LOOM_SCALAR_TYPE_I8) {
    scalar_type = LOOM_SCALAR_TYPE_I32;
  }
  if (loom_scalar_type_bitwidth(scalar_type) >
      loom_scalar_type_bitwidth(pattern->type)) {
    // Complete the existing carrier before widening so wide transfers share
    // their repeated low word with narrower transfers and vector splats.
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_repeated_bits(
        builder, bits, pattern->type, repeated_bits, location, &bits));
    repeated_bits = loom_scalar_type_bitwidth(pattern->type);
  }
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_integer_cast(
      builder, bits, pattern->type, scalar_type, location, &bits));
  return loom_buffer_legalize_build_repeated_bits(
      builder, bits, scalar_type,
      loom_type_element_type(transfer_type) == LOOM_SCALAR_TYPE_I8
          ? 32
          : repeated_bits,
      location, out_value);
}

// Narrow memory may require several transfers per pattern. Carry one rotated
// pattern through that loop instead of unrolling every byte into live values.
static iree_status_t loom_buffer_legalize_rotate_pattern(
    loom_builder_t* builder, loom_value_id_t bits, loom_scalar_type_t type,
    uint32_t byte_count, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_value_id_t right_shift = LOOM_VALUE_ID_INVALID;
  loom_value_id_t left_shift = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      builder, type, byte_count * 8, location, &right_shift));
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      builder, type, loom_scalar_type_bitwidth(type) - byte_count * 8, location,
      &left_shift));
  const loom_type_t working_type = loom_type_scalar(type);
  loom_op_t* low = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(builder, bits, right_shift,
                                               working_type, location, &low));
  loom_op_t* high = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shli_build(builder, 0, bits, left_shift,
                                              working_type, location, &high));
  loom_op_t* combined = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_ori_build(
      builder, loom_scalar_shrui_result(low), loom_scalar_shli_result(high),
      working_type, location, &combined));
  *out_value = loom_scalar_ori_result(combined);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_transfer(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 0, op->location, &zero));
  const bool is_fill = loom_buffer_fill_isa(op);
  const loom_value_id_t target =
      is_fill ? loom_buffer_fill_target(op) : loom_buffer_copy_target(op);
  const loom_value_id_t target_offset =
      is_fill ? loom_buffer_fill_target_offset(op)
              : loom_buffer_copy_target_offset(op);
  const loom_value_id_t byte_length = is_fill
                                          ? loom_buffer_fill_byte_length(op)
                                          : loom_buffer_copy_byte_length(op);
  uint8_t widths =
      loom_buffer_legalize_transfer_widths(context, target, target_offset);
  loom_buffer_fill_pattern_t pattern = {0};
  if (is_fill) {
    const loom_value_id_t value = loom_buffer_fill_pattern(op);
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_prepare_fill(
        &rewriter->builder, value,
        loom_type_element_type(loom_module_value_type(context->module, value)),
        op->location, &pattern));
  } else {
    widths &= loom_buffer_legalize_transfer_widths(
        context, loom_buffer_copy_source(op),
        loom_buffer_copy_source_offset(op));
  }
  int64_t static_length = 0;
  const bool is_static = loom_value_facts_as_exact_i64(
      loom_value_fact_table_lookup(context->fact_table, byte_length),
      &static_length);
  int64_t static_start = 0;
  loom_value_id_t start = zero;
  uint32_t previous_width = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t width = 64;
       width != 0 && start != byte_length && iree_status_is_ok(status);
       width /= 2) {
    if (!(widths & width)) {
      continue;
    }
    const int64_t static_end = static_length & -(int64_t)width;
    if (is_static && static_end == static_start) {
      continue;
    }
    loom_value_id_t end = LOOM_VALUE_ID_INVALID;
    if (is_static) {
      status = loom_buffer_legalize_build_offset_constant(
          &rewriter->builder, static_end, op->location, &end);
    } else {
      status = loom_buffer_legalize_build_aligned_end(
          context, byte_length, width, op->location, &end);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    const loom_type_t transfer_type =
        loom_buffer_legalize_transfer_type(&context->bulk_memory_policy, width);
    const bool single = is_static && static_end - static_start == width;
    const bool conditional = !is_static && previous_width == 2 * width;
    const bool rotates =
        is_fill && width < pattern.byte_count && !single && !conditional;
    loom_value_id_t pattern_bits = LOOM_VALUE_ID_INVALID;
    loom_value_id_t fill_value = LOOM_VALUE_ID_INVALID;
    if (is_fill) {
      status = loom_buffer_legalize_build_fill_bits(
          &rewriter->builder, &pattern, previous_width, start, op->location,
          &pattern_bits);
      if (iree_status_is_ok(status) && !rotates) {
        status = loom_buffer_legalize_build_fill_value(
            &rewriter->builder, &pattern, pattern_bits, transfer_type,
            op->location, &fill_value);
      }
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
    loom_value_id_t position = start;
    if (!single) {
      loom_op_t* region_op = NULL;
      if (conditional) {
        loom_op_t* condition_op = NULL;
        status = loom_index_cmp_build(&rewriter->builder,
                                      LOOM_INDEX_CMP_PREDICATE_ULT, start, end,
                                      op->location, &condition_op);
        if (iree_status_is_ok(status)) {
          status = loom_scf_if_build(&rewriter->builder, 0,
                                     loom_index_cmp_result(condition_op), NULL,
                                     0, NULL, 0, op->location, &region_op);
        }
        if (iree_status_is_ok(status)) {
          loom_builder_enter_region(&rewriter->builder, region_op,
                                    loom_scf_if_then_region(region_op));
        }
      } else {
        loom_value_id_t step = LOOM_VALUE_ID_INVALID;
        status = loom_buffer_legalize_build_offset_constant(
            &rewriter->builder, width, op->location, &step);
        if (iree_status_is_ok(status)) {
          status = loom_scf_for_build(
              &rewriter->builder, 0, start, end, step,
              rotates ? &pattern_bits : NULL, rotates ? 1 : 0, NULL, NULL, 0,
              LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0, op->location,
              &region_op);
        }
        if (iree_status_is_ok(status)) {
          loom_builder_enter_region(&rewriter->builder, region_op,
                                    loom_scf_for_body(region_op));
          position = loom_region_entry_arg_id(loom_scf_for_body(region_op), 0);
          if (rotates) {
            pattern_bits =
                loom_region_entry_arg_id(loom_scf_for_body(region_op), 1);
            status = loom_buffer_legalize_build_fill_value(
                &rewriter->builder, &pattern, pattern_bits, transfer_type,
                op->location, &fill_value);
          }
        }
      }
    }
    if (iree_status_is_ok(status)) {
      if (is_fill) {
        status = loom_buffer_legalize_build_store(
            &rewriter->builder, fill_value, target, target_offset, position,
            transfer_type, op->location);
      } else {
        status = loom_buffer_legalize_build_copy_chunk(
            &rewriter->builder, position, transfer_type,
            loom_buffer_copy_source(op), loom_buffer_copy_source_offset(op),
            target, target_offset, op->location);
      }
    }
    if (rotates && iree_status_is_ok(status)) {
      status = loom_buffer_legalize_rotate_pattern(
          &rewriter->builder, pattern_bits, pattern.type, width, op->location,
          &pattern_bits);
    }
    if (!single && iree_status_is_ok(status)) {
      loom_op_t* yield_op = NULL;
      status = loom_scf_yield_build(&rewriter->builder,
                                    rotates ? &pattern_bits : NULL,
                                    rotates ? 1 : 0, op->location, &yield_op);
    }
    if (!single) {
      loom_builder_restore(&rewriter->builder, saved_ip);
    }
    start = end;
    static_start = static_end;
    previous_width = width;
  }
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

// Numeric word order is not byte order on a little-endian target. Locate the
// first different byte in the already loaded words, then compare it unsigned.
static iree_status_t loom_buffer_legalize_build_unequal_word_order(
    loom_builder_t* builder, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t type, const loom_buffer_compare_constants_t* constants,
    loom_location_id_t location, loom_value_id_t* out_order) {
  loom_op_t* difference = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_xori_build(builder, lhs, rhs, type, location, &difference));
  loom_op_t* first_bit = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cttzi_build(builder, loom_scalar_xori_result(difference),
                              type, location, &first_bit));
  loom_value_id_t byte_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      builder, loom_type_element_type(type), -8, location, &byte_mask));
  loom_op_t* first_byte = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_andi_build(builder, loom_scalar_cttzi_result(first_bit),
                             byte_mask, type, location, &first_byte));
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      builder, loom_type_element_type(type), 255, location, &mask));
  loom_value_id_t bytes[2];
  const loom_value_id_t words[] = {lhs, rhs};
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    loom_op_t* shifted = NULL;
    status = loom_scalar_shrui_build(builder, words[i],
                                     loom_scalar_andi_result(first_byte), type,
                                     location, &shifted);
    loom_op_t* byte = NULL;
    if (iree_status_is_ok(status)) {
      status =
          loom_scalar_andi_build(builder, loom_scalar_shrui_result(shifted),
                                 mask, type, location, &byte);
    }
    if (iree_status_is_ok(status)) {
      bytes[i] = loom_scalar_andi_result(byte);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  loom_op_t* less = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_ULT, bytes[0],
                             bytes[1], location, &less));
  loom_op_t* order = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, loom_scalar_cmpi_result(less), constants->less,
      constants->greater, loom_type_scalar(LOOM_SCALAR_TYPE_I32), location,
      &order));
  *out_order = loom_scf_select_result(order);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_compare_order(
    loom_builder_t* builder, loom_value_id_t lhs, loom_value_id_t rhs,
    uint32_t width, loom_type_t type,
    const loom_buffer_compare_constants_t* constants,
    loom_location_id_t location, loom_value_id_t* out_order) {
  if (width == 1) {
    return loom_buffer_legalize_build_compare_byte_order(
        builder, lhs, rhs, constants, location, out_order);
  }
  loom_op_t* unequal = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_NE, lhs, rhs, location, &unequal));
  const loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* branch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                        loom_scalar_cmpi_result(unequal), &result_type, 1, NULL,
                        0, location, &branch));
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      builder, branch, loom_scf_if_then_region(branch));
  loom_value_id_t order = LOOM_VALUE_ID_INVALID;
  iree_status_t status = loom_buffer_legalize_build_unequal_word_order(
      builder, lhs, rhs, type, constants, location, &order);
  loom_op_t* yield = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_scf_yield_build(builder, &order, 1, location, &yield);
  }
  loom_builder_restore(builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  saved_ip = loom_builder_enter_region(builder, branch,
                                       loom_scf_if_else_region(branch));
  status =
      loom_scf_yield_build(builder, &constants->equal, 1, location, &yield);
  loom_builder_restore(builder, saved_ip);
  *out_order = loom_scf_if_results(branch).values[0];
  return status;
}

// Returns the last loaded pair, comparing only equality inside the loop.
// Empty scans return two zero words.
static iree_status_t loom_buffer_legalize_build_compare_range(
    loom_builder_t* builder, const loom_op_t* op, uint32_t width, bool single,
    loom_value_id_t start, loom_value_id_t end, loom_value_id_t* out_lhs,
    loom_value_id_t* out_rhs) {
  const loom_scalar_type_t element = width == 8   ? LOOM_SCALAR_TYPE_I64
                                     : width == 4 ? LOOM_SCALAR_TYPE_I32
                                     : width == 2 ? LOOM_SCALAR_TYPE_I16
                                                  : LOOM_SCALAR_TYPE_I8;
  const loom_type_t type = loom_type_scalar(element);
  const loom_type_t loaded_type =
      width == 1 ? loom_type_scalar(LOOM_SCALAR_TYPE_I32) : type;
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_op_t* loop = NULL;
  loom_value_id_t position = start;
  if (!single) {
    loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
        builder, loom_type_element_type(loaded_type), 0, op->location, &zero));
    const loom_value_id_t iter_args[] = {start, zero, zero};
    IREE_RETURN_IF_ERROR(loom_scf_while_build(
        builder, iter_args, IREE_ARRAYSIZE(iter_args), NULL, NULL,
        IREE_ARRAYSIZE(iter_args), NULL, 0, op->location, &loop));
    loom_region_t* before = loom_scf_while_before(loop);
    loom_builder_enter_region(builder, loop, before);
    const loom_value_id_t forwarded[] = {loom_region_entry_arg_id(before, 0),
                                         loom_region_entry_arg_id(before, 1),
                                         loom_region_entry_arg_id(before, 2)};
    loom_op_t* in_range = NULL;
    iree_status_t status =
        loom_index_cmp_build(builder, LOOM_INDEX_CMP_PREDICATE_ULT,
                             forwarded[0], end, op->location, &in_range);
    loom_op_t* equal = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ,
                                      forwarded[1], forwarded[2], op->location,
                                      &equal);
    }
    loom_op_t* condition = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_scalar_andi_build(builder, loom_index_cmp_result(in_range),
                                      loom_scalar_cmpi_result(equal),
                                      loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                      op->location, &condition);
    }
    if (iree_status_is_ok(status)) {
      loom_op_t* condition_op = NULL;
      status = loom_scf_condition_build(
          builder, loom_scalar_andi_result(condition), forwarded,
          IREE_ARRAYSIZE(forwarded), op->location, &condition_op);
    }
    loom_builder_restore(builder, saved_ip);
    IREE_RETURN_IF_ERROR(status);
    loom_builder_enter_region(builder, loop, loom_scf_while_after(loop));
    position = loom_region_entry_arg_id(loom_scf_while_after(loop), 0);
  }
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  iree_status_t status = loom_buffer_legalize_build_load(
      builder, loom_buffer_compare_lhs(op), loom_buffer_compare_lhs_offset(op),
      position, type, op->location, &lhs);
  if (iree_status_is_ok(status)) {
    status = loom_buffer_legalize_build_load(
        builder, loom_buffer_compare_rhs(op),
        loom_buffer_compare_rhs_offset(op), position, type, op->location, &rhs);
  }
  if (!single && iree_status_is_ok(status)) {
    loom_value_id_t step = LOOM_VALUE_ID_INVALID;
    status = loom_buffer_legalize_build_offset_constant(builder, width,
                                                        op->location, &step);
    loom_op_t* next_position = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_index_add_build(builder, position, step,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                    op->location, &next_position);
    }
    if (iree_status_is_ok(status)) {
      const loom_value_id_t yielded[] = {loom_index_add_result(next_position),
                                         lhs, rhs};
      loom_op_t* yield = NULL;
      status = loom_scf_yield_build(builder, yielded, IREE_ARRAYSIZE(yielded),
                                    op->location, &yield);
      lhs = loom_scf_while_results(loop).values[1];
      rhs = loom_scf_while_results(loop).values[2];
    }
  }
  loom_builder_restore(builder, saved_ip);
  *out_lhs = lhs;
  *out_rhs = rhs;
  return status;
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

  // SIMD comparison requires an efficient packed-predicate conversion in
  // addition to vector memory. Scalar words share the ordinary memory policy.
  const uint8_t widths =
      (context->bulk_memory_policy.scalar_byte_widths | 1) &
      loom_buffer_legalize_transfer_widths(context, loom_buffer_compare_lhs(op),
                                           loom_buffer_compare_lhs_offset(op)) &
      loom_buffer_legalize_transfer_widths(context, loom_buffer_compare_rhs(op),
                                           loom_buffer_compare_rhs_offset(op));
  const loom_value_id_t byte_length = loom_buffer_compare_byte_length(op);
  int64_t static_length = 0;
  const bool is_static = loom_value_facts_as_exact_i64(
      loom_value_fact_table_lookup(context->fact_table, byte_length),
      &static_length);
  loom_value_id_t start = zero_offset;
  loom_value_id_t pair[] = {constants.equal, constants.equal};
  // A single word loop plus a bounded byte tail avoids duplicating mismatch
  // resolution for every smaller width. Static short ranges still select the
  // widest word that fits their complete extent.
  uint32_t width = 8;
  while (width > 1 && (!(widths & width) ||
                       (is_static && (uint64_t)static_length < width))) {
    width /= 2;
  }
  const loom_type_t pair_type =
      loom_type_scalar(width == 8   ? LOOM_SCALAR_TYPE_I64
                       : width == 2 ? LOOM_SCALAR_TYPE_I16
                                    : LOOM_SCALAR_TYPE_I32);
  const int64_t static_end = static_length & -(int64_t)width;
  if (width > 1) {
    if (is_static) {
      IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
          &rewriter->builder, static_end, op->location, &start));
    } else {
      IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_aligned_end(
          context, byte_length, width, op->location, &start));
    }
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_compare_range(
        &rewriter->builder, op, width, is_static && static_end == width,
        zero_offset, start, &pair[0], &pair[1]));
  }
  if (start != byte_length &&
      (!is_static ||
       (width == 1 ? static_length != 0 : static_length != static_end))) {
    const loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
    loom_op_t* branch = NULL;
    if (width > 1) {
      loom_op_t* equal = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
          &rewriter->builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, pair[0], pair[1],
          op->location, &equal));
      const loom_type_t result_types[] = {pair_type, pair_type};
      IREE_RETURN_IF_ERROR(loom_scf_if_build(
          &rewriter->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
          loom_scalar_cmpi_result(equal), result_types,
          IREE_ARRAYSIZE(result_types), NULL, 0, op->location, &branch));
      loom_builder_enter_region(&rewriter->builder, branch,
                                loom_scf_if_else_region(branch));
      loom_op_t* yield = NULL;
      iree_status_t status = loom_scf_yield_build(
          &rewriter->builder, pair, IREE_ARRAYSIZE(pair), op->location, &yield);
      loom_builder_restore(&rewriter->builder, saved_ip);
      IREE_RETURN_IF_ERROR(status);
      loom_builder_enter_region(&rewriter->builder, branch,
                                loom_scf_if_then_region(branch));
    }
    const int64_t tail_length =
        width == 1 ? static_length : static_length - static_end;
    iree_status_t status = loom_buffer_legalize_build_compare_range(
        &rewriter->builder, op, 1, is_static && tail_length == 1, start,
        byte_length, &pair[0], &pair[1]);
    if (width > 1) {
      for (uint32_t i = 0;
           i < IREE_ARRAYSIZE(pair) && iree_status_is_ok(status); ++i) {
        status = loom_buffer_legalize_build_integer_cast(
            &rewriter->builder, pair[i], LOOM_SCALAR_TYPE_I32,
            loom_type_element_type(pair_type), op->location, &pair[i]);
      }
      if (iree_status_is_ok(status)) {
        loom_op_t* yield = NULL;
        status =
            loom_scf_yield_build(&rewriter->builder, pair, IREE_ARRAYSIZE(pair),
                                 op->location, &yield);
      }
      loom_builder_restore(&rewriter->builder, saved_ip);
      pair[0] = loom_scf_if_results(branch).values[0];
      pair[1] = loom_scf_if_results(branch).values[1];
    }
    IREE_RETURN_IF_ERROR(status);
  }
  // Resolve order after both scans have finished using address state.
  loom_value_id_t order = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_compare_order(
      &rewriter->builder, pair[0], pair[1], width, pair_type, &constants,
      op->location, &order));
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
    case LOOM_OP_BUFFER_DISTANCE: {
      IREE_RETURN_IF_ERROR(
          loom_buffer_distance_canonicalize(op, context->rewriter));
      out_result->action = iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)
                               ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                               : LOOM_TARGET_LEGALIZER_ACTION_DEFER;
      return iree_ok_status();
    }
    case LOOM_OP_BUFFER_PACK:
      return loom_buffer_legalize_pack(context, op, out_result);
    case LOOM_OP_BUFFER_COPY:
    case LOOM_OP_BUFFER_FILL:
      return loom_buffer_legalize_transfer(context, op, out_result);
    case LOOM_OP_BUFFER_COMPARE:
      return loom_buffer_legalize_compare(context, op, out_result);
    default:
      return iree_ok_status();
  }
}

static const loom_target_legalizer_rule_t kBufferLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_BUFFER_DISTANCE,
        .legalize = loom_buffer_legalize,
    },
    {
        .root_kind = LOOM_OP_BUFFER_PACK,
        .legalize = loom_buffer_legalize,
    },
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
    .scope = LOOM_LOW_DESCRIPTOR_SCOPE_SHARED,
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kBufferLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kBufferLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_buffer_target_legalizer_provider(
    void) {
  return &kBufferLegalizerProvider;
}
