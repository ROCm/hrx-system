// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/legalization_float.h"

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/transforms/vector/target_legalization.h"
#include "loom/util/numeric_format.h"

static bool loom_wasm_float_legalizer_supports_descriptor_set(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id ==
             loom_wasm_core_simd128_descriptor_set()->target_stable_id;
}

typedef uint8_t loom_wasm_float_extremum_kind_t;
enum loom_wasm_float_extremum_kind_e {
  LOOM_WASM_FLOAT_EXTREMUM_MINIMUM = 0,
  LOOM_WASM_FLOAT_EXTREMUM_MAXIMUM = 1,
};

static bool loom_wasm_scalar_extremum_is_relaxed(const loom_op_t* op) {
  const uint8_t required_flags =
      LOOM_SCALAR_FASTMATHFLAGS_NNAN | LOOM_SCALAR_FASTMATHFLAGS_NSZ;
  return (op->instance_flags & required_flags) == required_flags;
}

static bool loom_wasm_vector_extremum_is_relaxed(const loom_op_t* op) {
  const uint8_t required_flags =
      LOOM_VECTOR_FASTMATHFLAGS_NNAN | LOOM_VECTOR_FASTMATHFLAGS_NSZ;
  return (op->instance_flags & required_flags) == required_flags;
}

// Maps a narrow sign-magnitude scalar encoding to a signed I32 key. Sign
// extension retains the sign while the XOR reverses only a negative value's
// magnitude bits.
static iree_status_t loom_wasm_build_scalar_float_order_key(
    loom_builder_t* builder, loom_location_id_t location, loom_value_id_t value,
    loom_type_t float_type, loom_type_t integer_type,
    loom_value_id_t magnitude_mask, loom_value_id_t sign_shift,
    loom_value_id_t* out_key) {
  const loom_type_t working_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      builder, value, float_type, integer_type, location, &bits_op));
  loom_op_t* extended_bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(
      builder, loom_scalar_bitcast_result(bits_op), integer_type, working_type,
      location, &extended_bits_op));
  loom_op_t* sign_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shrsi_build(
      builder, loom_scalar_extsi_result(extended_bits_op), sign_shift,
      working_type, location, &sign_op));
  loom_op_t* adjustment_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      builder, loom_scalar_shrsi_result(sign_op), magnitude_mask, working_type,
      location, &adjustment_op));
  loom_op_t* key_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_xori_build(
      builder, loom_scalar_extsi_result(extended_bits_op),
      loom_scalar_andi_result(adjustment_op), working_type, location, &key_op));
  *out_key = loom_scalar_xori_result(key_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_build_scalar_relaxed_extremum(
    loom_builder_t* builder, loom_location_id_t location,
    loom_wasm_float_extremum_kind_t kind, uint8_t fastmath, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t type, loom_value_id_t* out_result) {
  const bool is_minimum = kind == LOOM_WASM_FLOAT_EXTREMUM_MINIMUM;
  loom_value_id_t predicate = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                    element_type)) {
    loom_numeric_float_encoding_t encoding = {0};
    const bool has_encoding =
        loom_numeric_float_encoding(element_type, &encoding);
    IREE_ASSERT(has_encoding && "floating extremum type must have an encoding");
    (void)has_encoding;
    const loom_type_t integer_type = loom_type_scalar(encoding.integer_type);
    const loom_type_t working_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* magnitude_mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64((int64_t)encoding.magnitude_mask), working_type,
        location, &magnitude_mask_op));
    loom_op_t* sign_shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64(31), working_type, location, &sign_shift_op));
    loom_value_id_t lhs_key = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_build_scalar_float_order_key(
        builder, location, lhs, type, integer_type,
        loom_scalar_constant_result(magnitude_mask_op),
        loom_scalar_constant_result(sign_shift_op), &lhs_key));
    loom_value_id_t rhs_key = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_build_scalar_float_order_key(
        builder, location, rhs, type, integer_type,
        loom_scalar_constant_result(magnitude_mask_op),
        loom_scalar_constant_result(sign_shift_op), &rhs_key));
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_cmpi_build(builder,
                               is_minimum ? LOOM_SCALAR_CMPI_PREDICATE_SLT
                                          : LOOM_SCALAR_CMPI_PREDICATE_SGT,
                               lhs_key, rhs_key, location, &comparison_op));
    predicate = loom_scalar_cmpi_result(comparison_op);
  } else {
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_cmpf_build(builder, fastmath,
                               is_minimum ? LOOM_SCALAR_CMPF_PREDICATE_OLT
                                          : LOOM_SCALAR_CMPF_PREDICATE_OGT,
                               lhs, rhs, location, &comparison_op));
    predicate = loom_scalar_cmpf_result(comparison_op);
  }
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(builder, predicate, lhs, rhs, type,
                                             location, &select_op));
  *out_result = loom_scf_select_result(select_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_rewrite_scalar_relaxed_extremum(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_rewritten) {
  *out_rewritten = false;
  if ((!loom_scalar_minnumf_isa(op) && !loom_scalar_maxnumf_isa(op) &&
       !loom_scalar_minimumf_isa(op) && !loom_scalar_maximumf_isa(op) &&
       !loom_scalar_clampf_isa(op)) ||
      !loom_wasm_scalar_extremum_is_relaxed(op)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t type =
      loom_module_value_type(rewriter->module, loom_op_results(op)[0]);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (loom_scalar_clampf_isa(op)) {
    loom_value_id_t lower_bounded = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_build_scalar_relaxed_extremum(
        builder, op->location, LOOM_WASM_FLOAT_EXTREMUM_MAXIMUM,
        op->instance_flags, loom_scalar_clampf_value(op),
        loom_scalar_clampf_lower(op), type, &lower_bounded));
    IREE_RETURN_IF_ERROR(loom_wasm_build_scalar_relaxed_extremum(
        builder, op->location, LOOM_WASM_FLOAT_EXTREMUM_MINIMUM,
        op->instance_flags, lower_bounded, loom_scalar_clampf_upper(op), type,
        &replacement));
  } else {
    const bool is_minimum =
        loom_scalar_minnumf_isa(op) || loom_scalar_minimumf_isa(op);
    IREE_RETURN_IF_ERROR(loom_wasm_build_scalar_relaxed_extremum(
        builder, op->location,
        is_minimum ? LOOM_WASM_FLOAT_EXTREMUM_MINIMUM
                   : LOOM_WASM_FLOAT_EXTREMUM_MAXIMUM,
        op->instance_flags, loom_op_operands(op)[0], loom_op_operands(op)[1],
        type, &replacement));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

// Maps narrow sign-magnitude vector payloads to same-width signed keys.
// Positive encodings are unchanged; negative magnitude bits are inverted.
static iree_status_t loom_wasm_build_vector_float_order_key(
    loom_builder_t* builder, loom_location_id_t location, loom_value_id_t value,
    loom_type_t float_type, loom_type_t integer_type,
    loom_value_id_t magnitude_mask, loom_value_id_t sign_shift,
    loom_value_id_t* out_key) {
  loom_op_t* bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
      builder, value, float_type, integer_type, location, &bits_op));
  loom_op_t* sign_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_shrsi_build(builder, loom_vector_bitcast_result(bits_op),
                              sign_shift, integer_type, location, &sign_op));
  loom_op_t* adjustment_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_andi_build(
      builder, loom_vector_shrsi_result(sign_op), magnitude_mask, integer_type,
      location, &adjustment_op));
  loom_op_t* key_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_xori_build(
      builder, loom_vector_bitcast_result(bits_op),
      loom_vector_andi_result(adjustment_op), integer_type, location, &key_op));
  *out_key = loom_vector_xori_result(key_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_build_vector_relaxed_extremum(
    loom_builder_t* builder, loom_location_id_t location,
    loom_wasm_float_extremum_kind_t kind, uint8_t fastmath, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t type, loom_value_id_t* out_result) {
  loom_type_t predicate_type = type;
  predicate_type.header =
      loom_type_make_header(loom_type_kind(type), LOOM_SCALAR_TYPE_I1,
                            loom_type_rank(type), loom_type_flags(type));
  const bool is_minimum = kind == LOOM_WASM_FLOAT_EXTREMUM_MINIMUM;
  loom_value_id_t predicate = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                    element_type)) {
    loom_numeric_float_encoding_t encoding = {0};
    const bool has_encoding =
        loom_numeric_float_encoding(element_type, &encoding);
    IREE_ASSERT(has_encoding && "floating extremum type must have an encoding");
    (void)has_encoding;
    loom_type_t integer_type = type;
    integer_type.header =
        loom_type_make_header(loom_type_kind(type), encoding.integer_type,
                              loom_type_rank(type), loom_type_flags(type));
    loom_op_t* magnitude_mask_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_splati_build(builder, (int64_t)encoding.magnitude_mask,
                                 integer_type, location, &magnitude_mask_op));
    loom_op_t* sign_shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splati_build(
        builder, loom_scalar_type_bitwidth(element_type) - 1, integer_type,
        location, &sign_shift_op));
    loom_value_id_t lhs_key = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_build_vector_float_order_key(
        builder, location, lhs, type, integer_type,
        loom_vector_splati_result(magnitude_mask_op),
        loom_vector_splati_result(sign_shift_op), &lhs_key));
    loom_value_id_t rhs_key = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_build_vector_float_order_key(
        builder, location, rhs, type, integer_type,
        loom_vector_splati_result(magnitude_mask_op),
        loom_vector_splati_result(sign_shift_op), &rhs_key));
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_cmpi_build(builder,
                               is_minimum ? LOOM_VECTOR_CMPI_PREDICATE_SLT
                                          : LOOM_VECTOR_CMPI_PREDICATE_SGT,
                               lhs_key, rhs_key, integer_type, predicate_type,
                               location, &comparison_op));
    predicate = loom_vector_cmpi_result(comparison_op);
  } else {
    loom_op_t* comparison_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
        builder, fastmath,
        is_minimum ? LOOM_VECTOR_CMPF_PREDICATE_OLT
                   : LOOM_VECTOR_CMPF_PREDICATE_OGT,
        lhs, rhs, type, predicate_type, location, &comparison_op));
    predicate = loom_vector_cmpf_result(comparison_op);
  }
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_select_build(builder, predicate, lhs, rhs,
                                                type, location, &select_op));
  *out_result = loom_vector_select_result(select_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_rewrite_vector_relaxed_extremum(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_rewritten) {
  *out_rewritten = false;
  if ((!loom_vector_minnumf_isa(op) && !loom_vector_maxnumf_isa(op) &&
       !loom_vector_minimumf_isa(op) && !loom_vector_maximumf_isa(op) &&
       !loom_vector_clampf_isa(op)) ||
      !loom_wasm_vector_extremum_is_relaxed(op)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t type =
      loom_module_value_type(rewriter->module, loom_op_results(op)[0]);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (loom_vector_clampf_isa(op)) {
    loom_value_id_t lower_bounded = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_build_vector_relaxed_extremum(
        builder, op->location, LOOM_WASM_FLOAT_EXTREMUM_MAXIMUM,
        op->instance_flags, loom_vector_clampf_value(op),
        loom_vector_clampf_lower(op), type, &lower_bounded));
    IREE_RETURN_IF_ERROR(loom_wasm_build_vector_relaxed_extremum(
        builder, op->location, LOOM_WASM_FLOAT_EXTREMUM_MINIMUM,
        op->instance_flags, lower_bounded, loom_vector_clampf_upper(op), type,
        &replacement));
  } else {
    const bool is_minimum =
        loom_vector_minnumf_isa(op) || loom_vector_minimumf_isa(op);
    IREE_RETURN_IF_ERROR(loom_wasm_build_vector_relaxed_extremum(
        builder, op->location,
        is_minimum ? LOOM_WASM_FLOAT_EXTREMUM_MINIMUM
                   : LOOM_WASM_FLOAT_EXTREMUM_MAXIMUM,
        op->instance_flags, loom_op_operands(op)[0], loom_op_operands(op)[1],
        type, &replacement));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

static loom_type_t loom_wasm_vector_type_with_element(
    loom_type_t source_type, loom_scalar_type_t element_type) {
  source_type.header = loom_type_make_header(
      loom_type_kind(source_type), element_type, loom_type_rank(source_type),
      loom_type_flags(source_type));
  return source_type;
}

static iree_status_t loom_wasm_float_comparison_widen_operand(
    loom_builder_t* builder, loom_value_id_t operand, loom_type_t source_type,
    loom_type_t comparison_type, loom_location_id_t location,
    loom_value_id_t* out_operand) {
  if (loom_type_equal(source_type, comparison_type)) {
    *out_operand = operand;
    return iree_ok_status();
  }
  loom_op_t* extension_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extf_build(
      builder, /*instance_flags=*/0, operand, source_type, comparison_type,
      location, &extension_op));
  *out_operand = loom_vector_extf_result(extension_op);
  return iree_ok_status();
}

iree_status_t loom_wasm_legalize_vector_cmpf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_float_legalizer_supports_descriptor_set(
          context->descriptor_set) ||
      !loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t lhs = loom_vector_cmpf_lhs(op);
  const loom_value_id_t rhs = loom_vector_cmpf_rhs(op);
  const loom_type_t source_type = loom_module_value_type(context->module, lhs);
  const loom_type_t comparison_type =
      loom_wasm_vector_type_with_element(source_type, LOOM_SCALAR_TYPE_F32);
  loom_value_id_t comparison_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_float_comparison_widen_operand(
      builder, lhs, source_type, comparison_type, op->location,
      &comparison_lhs));
  loom_value_id_t comparison_rhs = comparison_lhs;
  if (rhs != lhs) {
    IREE_RETURN_IF_ERROR(loom_wasm_float_comparison_widen_operand(
        builder, rhs, source_type, comparison_type, op->location,
        &comparison_rhs));
  }

  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_cmpf_result(op));
  loom_op_t* comparison_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
      builder, loom_vector_cmpf_fastmath(op), loom_vector_cmpf_predicate(op),
      comparison_lhs, comparison_rhs, comparison_type, result_type,
      op->location, &comparison_op));
  const loom_value_id_t replacement = loom_vector_cmpf_result(comparison_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

iree_status_t loom_wasm_legalize_scalar_float_extrema(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_float_legalizer_supports_descriptor_set(
          context->descriptor_set)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_wasm_rewrite_scalar_relaxed_extremum(
      context->rewriter, op, &rewritten));
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}

iree_status_t loom_wasm_legalize_vector_float_extrema(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_wasm_float_legalizer_supports_descriptor_set(
          context->descriptor_set) ||
      !loom_target_legalization_op_has_source_vector_carriers(context, op)) {
    return iree_ok_status();
  }
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_wasm_rewrite_vector_relaxed_extremum(
      context->rewriter, op, &rewritten));
  if (!rewritten) {
    IREE_RETURN_IF_ERROR(loom_vector_float_extrema_rewrite_op(context->rewriter,
                                                              op, &rewritten));
  }
  if (rewritten) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return iree_ok_status();
}
