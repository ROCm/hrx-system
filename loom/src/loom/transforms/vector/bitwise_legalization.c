// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/bitwise_legalization.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector/to_scalar.h"

iree_status_t loom_vector_legalize_bitfield_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const bool signed_extract = loom_vector_bitfield_extracts_isa(op);
  loom_value_id_t value = signed_extract
                              ? loom_vector_bitfield_extracts_source(op)
                              : loom_vector_bitfield_extractu_source(op);
  const int64_t offset = signed_extract
                             ? loom_vector_bitfield_extracts_offset(op)
                             : loom_vector_bitfield_extractu_offset(op);
  const int64_t width = signed_extract
                            ? loom_vector_bitfield_extracts_width(op)
                            : loom_vector_bitfield_extractu_width(op);
  const loom_type_t source_type =
      loom_module_value_type(context->module, value);
  // A singleton has no SIMD parallelism. Preserve its scalar lane program
  // instead of introducing a wide-register broadcast and extraction.
  uint64_t element_count = 0;
  if (loom_type_static_element_count(source_type, &element_count) &&
      element_count == 1) {
    bool rewritten = false;
    IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
        context->pass, context->rewriter, op, &rewritten));
    *out_result = (loom_target_legalizer_result_t){
        .action = rewritten ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                            : LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
    };
    return iree_ok_status();
  }
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  const int32_t source_width =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);

  // Retain the logical vector width. Target shift and cast legalizers own
  // packetization and any required widening of these elementwise operations.
  loom_op_t* vector_op = NULL;
  if (signed_extract && source_width - offset - width != 0) {
    loom_op_t* amount = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(source_width - offset - width), source_type,
        op->location, &amount));
    IREE_RETURN_IF_ERROR(
        loom_vector_shli_build(builder, /*instance_flags=*/0, value,
                               loom_vector_constant_result(amount), source_type,
                               op->location, &vector_op));
    value = loom_vector_shli_result(vector_op);
  }
  const int64_t right_shift = signed_extract ? source_width - width : offset;
  if (right_shift != 0) {
    loom_op_t* amount = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_constant_build(builder, loom_attr_i64(right_shift),
                                   source_type, op->location, &amount));
    if (signed_extract) {
      IREE_RETURN_IF_ERROR(loom_vector_shrsi_build(
          builder, value, loom_vector_constant_result(amount), source_type,
          op->location, &vector_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_shrui_build(
          builder, value, loom_vector_constant_result(amount), source_type,
          op->location, &vector_op));
    }
    value = loom_op_results(vector_op)[0];
  }
  if (!signed_extract && width < source_width - offset) {
    loom_op_t* mask = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(UINT64_MAX >> (64 - width)), source_type,
        op->location, &mask));
    IREE_RETURN_IF_ERROR(loom_vector_andi_build(
        builder, value, loom_vector_constant_result(mask), source_type,
        op->location, &vector_op));
    value = loom_vector_andi_result(vector_op);
  }
  if (result_width < source_width) {
    IREE_RETURN_IF_ERROR(loom_vector_trunci_build(
        builder, value, source_type, result_type, op->location, &vector_op));
    value = loom_vector_trunci_result(vector_op);
  } else if (result_width > source_width) {
    if (signed_extract) {
      IREE_RETURN_IF_ERROR(loom_vector_extsi_build(
          builder, value, source_type, result_type, op->location, &vector_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_extui_build(
          builder, value, source_type, result_type, op->location, &vector_op));
    }
    value = loom_op_results(vector_op)[0];
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &value, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &value, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

// Decomposes a rotation while preserving its vector and logical element
// domains. Target packet contracts lower the resulting shifts and bitwise
// operations without exposing authored lanes as scalar IR.
iree_status_t loom_vector_legalize_rotation(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t value = loom_op_operands(op)[0];
  const loom_value_id_t count = loom_op_operands(op)[1];
  const loom_type_t type = loom_module_value_type(context->module, value);
  const int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(type));

  loom_value_id_t replacement = value;
  if (bit_width != 1) {
    loom_op_t* mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(bit_width - 1), type, op->location, &mask_op));
    loom_op_t* amount_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_andi_build(
        builder, count, loom_vector_constant_result(mask_op), type,
        op->location, &amount_op));

    loom_op_t* complement_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_xori_build(builder, loom_vector_andi_result(amount_op),
                               loom_vector_constant_result(mask_op), type,
                               op->location, &complement_op));
    loom_op_t* one_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_constant_build(
        builder, loom_attr_i64(1), type, op->location, &one_op));

    const loom_value_id_t amount = loom_vector_andi_result(amount_op);
    const loom_value_id_t complement = loom_vector_xori_result(complement_op);
    const loom_value_id_t one = loom_vector_constant_result(one_op);
    const bool rotate_left = loom_vector_rotli_isa(op);
    loom_op_t* complementary_op = NULL;
    if (rotate_left) {
      IREE_RETURN_IF_ERROR(loom_vector_shrui_build(
          builder, value, complement, type, op->location, &complementary_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_shli_build(
          builder, /*instance_flags=*/0, value, complement, type, op->location,
          &complementary_op));
    }
    loom_op_t* left_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_shli_build(
        builder, /*instance_flags=*/0,
        rotate_left ? value : loom_vector_shli_result(complementary_op),
        rotate_left ? amount : one, type, op->location, &left_op));
    loom_op_t* right_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_shrui_build(
        builder,
        rotate_left ? loom_vector_shrui_result(complementary_op) : value,
        rotate_left ? one : amount, type, op->location, &right_op));
    loom_op_t* combine_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_ori_build(
        builder, loom_vector_shli_result(left_op),
        loom_vector_shrui_result(right_op), type, op->location, &combine_op));
    replacement = loom_vector_ori_result(combine_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}
