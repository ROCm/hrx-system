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

static iree_status_t loom_vector_build_fixed_shift(
    loom_builder_t* builder, loom_op_kind_t kind, loom_value_id_t value,
    loom_type_t type, int64_t amount, loom_location_id_t location,
    loom_value_id_t* out_result) {
  loom_op_t* amount_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_splati_build(builder, amount, type, location, &amount_op));
  const loom_value_id_t amount_value = loom_vector_splati_result(amount_op);

  loom_op_t* shift_op = NULL;
  switch (kind) {
    case LOOM_OP_VECTOR_SHLI: {
      IREE_RETURN_IF_ERROR(loom_vector_shli_build(builder, /*instance_flags=*/0,
                                                  value, amount_value, type,
                                                  location, &shift_op));
      break;
    }
    case LOOM_OP_VECTOR_SHRSI: {
      IREE_RETURN_IF_ERROR(loom_vector_shrsi_build(builder, value, amount_value,
                                                   type, location, &shift_op));
      break;
    }
    case LOOM_OP_VECTOR_SHRUI: {
      IREE_RETURN_IF_ERROR(loom_vector_shrui_build(builder, value, amount_value,
                                                   type, location, &shift_op));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("invalid fixed vector shift kind");
      IREE_BUILTIN_UNREACHABLE();
  }
  *out_result = loom_op_results(shift_op)[0];
  return iree_ok_status();
}

// Preserves rejected bitfield extraction in the vector domain. Targets can
// select the resulting fixed shifts and conversion independently instead of
// exposing every authored lane to scalar IR.
iree_status_t loom_vector_legalize_bitfield_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const bool signed_extract = loom_vector_bitfield_extracts_isa(op);
  const loom_value_id_t source = loom_op_operands(op)[0];
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
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
  const int64_t offset = signed_extract
                             ? loom_vector_bitfield_extracts_offset(op)
                             : loom_vector_bitfield_extractu_offset(op);
  const int64_t width = signed_extract
                            ? loom_vector_bitfield_extracts_width(op)
                            : loom_vector_bitfield_extractu_width(op);

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);

  const bool full_width = offset == 0 && width == source_width;
  const bool top_widen =
      result_width > source_width && offset + width == source_width;
  loom_value_id_t replacement = source;
  loom_type_t working_type = source_type;
  int32_t working_width = source_width;
  if (result_width > source_width) {
    loom_op_t* extension_op = NULL;
    if (signed_extract && (full_width || top_widen)) {
      IREE_RETURN_IF_ERROR(
          loom_vector_extsi_build(builder, replacement, source_type,
                                  result_type, op->location, &extension_op));
      replacement = loom_vector_extsi_result(extension_op);
    } else {
      // Extraction interprets the source as raw bits. Zero extension retains
      // those bits before the field is aligned in the wider result carrier.
      IREE_RETURN_IF_ERROR(
          loom_vector_extui_build(builder, replacement, source_type,
                                  result_type, op->location, &extension_op));
      replacement = loom_vector_extui_result(extension_op);
    }
    working_type = result_type;
    working_width = result_width;
  }

  if (!full_width) {
    if (top_widen) {
      IREE_RETURN_IF_ERROR(loom_vector_build_fixed_shift(
          builder, signed_extract ? LOOM_OP_VECTOR_SHRSI : LOOM_OP_VECTOR_SHRUI,
          replacement, working_type, offset, op->location, &replacement));
    } else {
      const int64_t left_shift = working_width - offset - width;
      if (left_shift != 0) {
        IREE_RETURN_IF_ERROR(loom_vector_build_fixed_shift(
            builder, LOOM_OP_VECTOR_SHLI, replacement, working_type, left_shift,
            op->location, &replacement));
      }
      const int64_t right_shift = working_width - width;
      if (right_shift != 0) {
        IREE_RETURN_IF_ERROR(loom_vector_build_fixed_shift(
            builder,
            signed_extract ? LOOM_OP_VECTOR_SHRSI : LOOM_OP_VECTOR_SHRUI,
            replacement, working_type, right_shift, op->location,
            &replacement));
      }
    }
  }

  if (result_width < source_width) {
    loom_op_t* truncation_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_trunci_build(builder, replacement, source_type, result_type,
                                 op->location, &truncation_op));
    replacement = loom_vector_trunci_result(truncation_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
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
    IREE_RETURN_IF_ERROR(loom_vector_splati_build(builder, bit_width - 1, type,
                                                  op->location, &mask_op));
    loom_op_t* amount_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_andi_build(
        builder, count, loom_vector_splati_result(mask_op), type, op->location,
        &amount_op));

    loom_op_t* complement_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_xori_build(builder, loom_vector_andi_result(amount_op),
                               loom_vector_splati_result(mask_op), type,
                               op->location, &complement_op));
    loom_op_t* one_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_splati_build(builder, 1, type, op->location, &one_op));

    const loom_value_id_t amount = loom_vector_andi_result(amount_op);
    const loom_value_id_t complement = loom_vector_xori_result(complement_op);
    const loom_value_id_t one = loom_vector_splati_result(one_op);
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
