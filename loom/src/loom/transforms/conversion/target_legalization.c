// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/conversion/target_legalization.h"

#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

// Staged conversions are attribute-free one-operand, one-result operations.
// Float extension/truncation flags remain zero because staging preserves the
// source operation's strict conversion semantics.
static iree_status_t loom_conversion_build_exact(
    loom_builder_t* builder, loom_op_kind_t kind, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  IREE_RETURN_IF_ERROR(
      loom_builder_allocate_op(builder, kind, 1, 1, 0, 0, 0, location, out_op));
  loom_op_operands(*out_op)[0] = input;
  IREE_RETURN_IF_ERROR(loom_builder_define_result(
      builder, result_type, &loom_op_results(*out_op)[0]));
  return loom_builder_finalize_op(builder, *out_op);
}

// Builds a shape-preserving conversion chain through the supplied intermediate
// element types. The caller establishes that every stage preserves the
// original conversion semantics.
static iree_status_t loom_conversion_stage_chain(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_scalar_type_t* intermediate_elements,
    iree_host_size_t intermediate_count, const loom_op_kind_t* stage_kinds,
    loom_target_legalizer_result_t* out_result) {
  const loom_value_id_t input = loom_op_operands(op)[0];
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t stage_input = input;
  for (iree_host_size_t stage_index = 0; stage_index <= intermediate_count;
       ++stage_index) {
    loom_type_t stage_result_type = result_type;
    if (stage_index < intermediate_count) {
      stage_result_type = input_type;
      stage_result_type.header = loom_type_make_header(
          loom_type_kind(input_type), intermediate_elements[stage_index],
          loom_type_rank(input_type), loom_type_flags(input_type));
      IREE_RETURN_IF_ERROR(loom_module_intern_type(
          context->module, stage_result_type, &stage_result_type));
    }
    loom_op_t* stage = NULL;
    IREE_RETURN_IF_ERROR(loom_conversion_build_exact(
        &rewriter->builder, stage_kinds[stage_index], stage_input,
        stage_result_type, op->location, &stage));
    stage_input = loom_op_results(stage)[0];
  }
  const loom_value_id_t replacement = stage_input;
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_conversion_stage_through(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_scalar_type_t intermediate_element, loom_op_kind_t first_kind,
    loom_op_kind_t second_kind, loom_target_legalizer_result_t* out_result) {
  const loom_scalar_type_t intermediate_elements[] = {intermediate_element};
  const loom_op_kind_t stage_kinds[] = {first_kind, second_kind};
  return loom_conversion_stage_chain(context, op, intermediate_elements,
                                     IREE_ARRAYSIZE(intermediate_elements),
                                     stage_kinds, out_result);
}

iree_status_t loom_conversion_legalize_narrow_float_to_i64(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const bool is_signed =
      op->kind == LOOM_OP_SCALAR_FPTOSI || op->kind == LOOM_OP_VECTOR_FPTOSI;
  const bool is_vector = loom_type_is_vector(
      loom_module_value_type(context->module, loom_op_operands(op)[0]));
  const loom_scalar_type_t intermediate_elements[] = {
      LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_I32,
  };
  const loom_op_kind_t stage_kinds[] = {
      is_vector ? LOOM_OP_VECTOR_EXTF : LOOM_OP_SCALAR_EXTF,
      is_vector ? (is_signed ? LOOM_OP_VECTOR_FPTOSI : LOOM_OP_VECTOR_FPTOUI)
                : (is_signed ? LOOM_OP_SCALAR_FPTOSI : LOOM_OP_SCALAR_FPTOUI),
      is_vector ? (is_signed ? LOOM_OP_VECTOR_EXTSI : LOOM_OP_VECTOR_EXTUI)
                : (is_signed ? LOOM_OP_SCALAR_EXTSI : LOOM_OP_SCALAR_EXTUI),
  };
  return loom_conversion_stage_chain(context, op, intermediate_elements,
                                     IREE_ARRAYSIZE(intermediate_elements),
                                     stage_kinds, out_result);
}

static bool loom_conversion_match_integer_to_float(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  const loom_scalar_type_t result_element = loom_type_element_type(
      loom_module_value_type(context->module, loom_op_results(op)[0]));
  // All integers near a finite F16/FP8 rounding boundary are exact in F32.
  // Larger magnitudes overflow IEEE formats or saturate E4M3 identically.
  if (result_element == LOOM_SCALAR_TYPE_F16 ||
      result_element == LOOM_SCALAR_TYPE_F8E4M3 ||
      result_element == LOOM_SCALAR_TYPE_F8E5M2) {
    return true;
  }
  // I1/I8/I16 are exact in F32. Wider integers are not: e.g. 16842753 rounds
  // to BF16 0x4b81 directly, but to 0x4b80 through F32. Keep those
  // conversions single-rounded unless source facts prove an exact F32 stage.
  const loom_scalar_type_t input_element = loom_type_element_type(
      loom_module_value_type(context->module, loom_op_operands(op)[0]));
  const bool narrow_input = loom_scalar_type_set_contains(
      LOOM_SCALAR_TYPE_SET_I1 | LOOM_SCALAR_TYPE_SET_I8 |
          LOOM_SCALAR_TYPE_SET_I16,
      input_element);
  if (!narrow_input && result_element == LOOM_SCALAR_TYPE_BF16) {
    const loom_value_id_t input = loom_op_const_operands(op)[0];
    const loom_float_integer_conversion_kind_t kind =
        op->kind == LOOM_OP_SCALAR_SITOFP || op->kind == LOOM_OP_VECTOR_SITOFP
            ? LOOM_FLOAT_INTEGER_CONVERSION_SIGNED
            : LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED;
    return loom_scalar_type_set_contains(
        loom_value_facts_integer_exact_float_types(
            input_element, kind,
            loom_value_fact_table_lookup(context->fact_table, input)),
        LOOM_SCALAR_TYPE_F32);
  }
  return narrow_input && (result_element == LOOM_SCALAR_TYPE_BF16 ||
                          result_element == LOOM_SCALAR_TYPE_F32 ||
                          result_element == LOOM_SCALAR_TYPE_F64);
}

static iree_status_t loom_conversion_legalize_integer_to_float(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const bool is_signed =
      op->kind == LOOM_OP_SCALAR_SITOFP || op->kind == LOOM_OP_VECTOR_SITOFP;
  const bool is_vector = loom_type_is_vector(
      loom_module_value_type(context->module, loom_op_operands(op)[0]));
  const loom_op_kind_t convert_kind =
      is_vector ? (is_signed ? LOOM_OP_VECTOR_SITOFP : LOOM_OP_VECTOR_UITOFP)
                : (is_signed ? LOOM_OP_SCALAR_SITOFP : LOOM_OP_SCALAR_UITOFP);
  const loom_scalar_type_t result_element = loom_type_element_type(
      loom_module_value_type(context->module, loom_op_results(op)[0]));
  if (result_element == LOOM_SCALAR_TYPE_F32 ||
      result_element == LOOM_SCALAR_TYPE_F64) {
    const loom_op_kind_t extend_kind =
        is_vector ? (is_signed ? LOOM_OP_VECTOR_EXTSI : LOOM_OP_VECTOR_EXTUI)
                  : (is_signed ? LOOM_OP_SCALAR_EXTSI : LOOM_OP_SCALAR_EXTUI);
    return loom_conversion_stage_through(context, op, LOOM_SCALAR_TYPE_I32,
                                         extend_kind, convert_kind, out_result);
  }
  return loom_conversion_stage_through(
      context, op, LOOM_SCALAR_TYPE_F32, convert_kind,
      is_vector ? LOOM_OP_VECTOR_FPTRUNC : LOOM_OP_SCALAR_FPTRUNC, out_result);
}

static iree_status_t loom_conversion_legalize_float_to_integer(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const bool is_signed =
      op->kind == LOOM_OP_SCALAR_FPTOSI || op->kind == LOOM_OP_VECTOR_FPTOSI;
  const bool is_vector = loom_type_is_vector(
      loom_module_value_type(context->module, loom_op_operands(op)[0]));
  const loom_op_kind_t convert_kind =
      is_vector ? (is_signed ? LOOM_OP_VECTOR_FPTOSI : LOOM_OP_VECTOR_FPTOUI)
                : (is_signed ? LOOM_OP_SCALAR_FPTOSI : LOOM_OP_SCALAR_FPTOUI);
  // Every F16/BF16/FP8 source value widens exactly, so the original toward-zero
  // conversion and its defined integer result domain are unchanged.
  return loom_conversion_stage_through(
      context, op, LOOM_SCALAR_TYPE_F32,
      is_vector ? LOOM_OP_VECTOR_EXTF : LOOM_OP_SCALAR_EXTF, convert_kind,
      out_result);
}

static const loom_target_legalizer_rule_t kConversionLegalizerRules[] = {
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_SCALAR_SITOFP,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .match = loom_conversion_match_integer_to_float,
        .legalize = loom_conversion_legalize_integer_to_float,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_SCALAR_UITOFP,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .match = loom_conversion_match_integer_to_float,
        .legalize = loom_conversion_legalize_integer_to_float,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_VECTOR_SITOFP,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .match = loom_conversion_match_integer_to_float,
        .legalize = loom_conversion_legalize_integer_to_float,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_VECTOR_UITOFP,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_INTEGER,
        .match = loom_conversion_match_integer_to_float,
        .legalize = loom_conversion_legalize_integer_to_float,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_SCALAR_FPTOSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_16BIT_FLOAT |
                                       LOOM_SCALAR_TYPE_SET_F8E4M3 |
                                       LOOM_SCALAR_TYPE_SET_F8E5M2,
        .legalize = loom_conversion_legalize_float_to_integer,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_SCALAR_FPTOUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_16BIT_FLOAT |
                                       LOOM_SCALAR_TYPE_SET_F8E4M3 |
                                       LOOM_SCALAR_TYPE_SET_F8E5M2,
        .legalize = loom_conversion_legalize_float_to_integer,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_VECTOR_FPTOSI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_16BIT_FLOAT |
                                       LOOM_SCALAR_TYPE_SET_F8E4M3 |
                                       LOOM_SCALAR_TYPE_SET_F8E5M2,
        .legalize = loom_conversion_legalize_float_to_integer,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_VECTOR_FPTOUI,
        .first_operand_element_types = LOOM_SCALAR_TYPE_SET_16BIT_FLOAT |
                                       LOOM_SCALAR_TYPE_SET_F8E4M3 |
                                       LOOM_SCALAR_TYPE_SET_F8E5M2,
        .legalize = loom_conversion_legalize_float_to_integer,
    },
};

static const loom_target_legalizer_provider_t kConversionLegalizerProvider = {
    .name = IREE_SVL("conversion"),
    .scope = LOOM_LOW_DESCRIPTOR_SCOPE_SHARED,
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kConversionLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kConversionLegalizerRules),
};

const loom_target_legalizer_provider_t*
loom_conversion_target_legalizer_provider(void) {
  return &kConversionLegalizerProvider;
}
