// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/combine_patterns.h"

#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/conversion/chain.h"
#include "loom/transforms/scalar/narrowing.h"
#include "loom/util/fact_table.h"
#include "loom/util/predicate_facts.h"

static loom_conversion_kind_t loom_scalar_combine_conversion_kind(
    const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_EXTF:
      return LOOM_CONVERSION_EXTF;
    case LOOM_OP_SCALAR_FPTRUNC:
      return LOOM_CONVERSION_FPTRUNC;
    case LOOM_OP_SCALAR_EXTSI:
      return LOOM_CONVERSION_EXTSI;
    case LOOM_OP_SCALAR_EXTUI:
      return LOOM_CONVERSION_EXTUI;
    case LOOM_OP_SCALAR_TRUNCI:
      return LOOM_CONVERSION_TRUNCI;
    case LOOM_OP_SCALAR_SITOFP:
      return LOOM_CONVERSION_SITOFP;
    case LOOM_OP_SCALAR_UITOFP:
      return LOOM_CONVERSION_UITOFP;
    default:
      return LOOM_CONVERSION_NONE;
  }
}

static iree_status_t loom_scalar_combine_replace_with_value(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_scalar_combine_replace_with_conversion(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_conversion_kind_t candidate,
    loom_value_id_t input, bool* out_changed) {
  const loom_type_t input_type =
      loom_module_value_type(rewriter->module, input);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  const loom_conversion_kind_t replacement_kind =
      loom_conversion_chain_resolve(candidate, input_type, result_type);
  if (replacement_kind == LOOM_CONVERSION_NONE) {
    return iree_ok_status();
  }
  if (replacement_kind == LOOM_CONVERSION_IDENTITY) {
    IREE_RETURN_IF_ERROR(
        loom_scalar_combine_replace_with_value(rewriter, op, input));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  switch (replacement_kind) {
    case LOOM_CONVERSION_EXTF: {
      IREE_RETURN_IF_ERROR(loom_scalar_extf_build(
          &rewriter->builder, /*instance_flags=*/0, input, input_type,
          result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_FPTRUNC: {
      IREE_RETURN_IF_ERROR(loom_scalar_fptrunc_build(
          &rewriter->builder, /*instance_flags=*/0, input, input_type,
          result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_EXTSI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extsi_build(&rewriter->builder, input, input_type,
                                  result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_EXTUI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extui_build(&rewriter->builder, input, input_type,
                                  result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_TRUNCI: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_trunci_build(&rewriter->builder, input, input_type,
                                   result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_SITOFP:
    case LOOM_CONVERSION_UITOFP: {
      IREE_RETURN_IF_ERROR((replacement_kind == LOOM_CONVERSION_SITOFP
                                ? loom_scalar_sitofp_build
                                : loom_scalar_uitofp_build)(
          &rewriter->builder, input, input_type, result_type, op->location,
          &replacement_op));
      break;
    }
    case LOOM_CONVERSION_EXTRACT_SIGNED:
    case LOOM_CONVERSION_EXTRACT_UNSIGNED: {
      const loom_type_t intermediate_type = loom_module_value_type(
          rewriter->module, loom_op_const_operands(op)[0]);
      const int32_t field_width =
          loom_scalar_type_bitwidth(loom_type_element_type(intermediate_type));
      IREE_RETURN_IF_ERROR((replacement_kind == LOOM_CONVERSION_EXTRACT_SIGNED
                                ? loom_scalar_bitfield_extracts_build
                                : loom_scalar_bitfield_extractu_build)(
          &rewriter->builder, input, 0, field_width, result_type, op->location,
          &replacement_op));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported scalar conversion replacement");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t replacement = loom_op_const_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_scalar_combine_replace_with_value(rewriter, op, replacement));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_scalar_combine_numeric_identity(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_conversion_kind_t outer_kind,
    bool* out_changed) {
  if (outer_kind != LOOM_CONVERSION_EXTSI &&
      outer_kind != LOOM_CONVERSION_EXTUI &&
      outer_kind != LOOM_CONVERSION_TRUNCI) {
    return iree_ok_status();
  }
  const loom_value_id_t outer_input = loom_op_const_operands(op)[0];
  const loom_value_id_t result = loom_op_const_results(op)[0];
  const loom_value_id_t origin = loom_value_fact_table_query_numeric_identity(
      rewriter->fact_table, result);
  if (origin != result &&
      loom_type_equal(loom_module_value_type(rewriter->module, result),
                      loom_module_value_type(rewriter->module, origin))) {
    // Numeric identity survives predicate carriers between conversions. Retain
    // the input's scoped information while returning the same-typed origin.
    const loom_type_t type = loom_module_value_type(rewriter->module, result);
    loom_predicate_t
        predicates[LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY + 1];
    uint16_t count = loom_value_facts_export_integer_refinements(
        origin, loom_rewriter_value_facts(rewriter, origin),
        loom_rewriter_value_facts(rewriter, result), predicates,
        LOOM_VALUE_FACT_INTEGER_REFINEMENT_CAPACITY);
    predicates[count++] = (loom_predicate_t){
        .kind = LOOM_PREDICATE_EQ,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
        .args = {origin, outer_input},
    };
    loom_builder_set_before(&rewriter->builder, op);
    loom_op_t* assume = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_assume_build(&rewriter->builder, &origin,
                                                  1, predicates, count, &type,
                                                  1, op->location, &assume));
    IREE_RETURN_IF_ERROR(loom_scalar_combine_replace_with_value(
        rewriter, op, loom_scalar_assume_results(assume).values[0]));
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_combine_conversion(loom_op_t* op,
                                             loom_rewriter_t* rewriter,
                                             bool allow_bitfield,
                                             bool* out_changed) {
  *out_changed = false;
  const loom_conversion_kind_t outer_kind =
      loom_scalar_combine_conversion_kind(op);
  // Callers supply a verified unary width conversion.
  const loom_value_id_t outer_input = loom_op_const_operands(op)[0];
  loom_op_t* defining_op =
      loom_rewriter_value_defining_op(rewriter, outer_input);
  if (defining_op == NULL) {
    return loom_scalar_combine_numeric_identity(op, rewriter, outer_kind,
                                                out_changed);
  }

  const loom_conversion_kind_t inner_kind =
      loom_scalar_combine_conversion_kind(defining_op);
  if ((loom_conversion_kind_is_float(outer_kind) && op->instance_flags != 0) ||
      (loom_conversion_kind_is_float(inner_kind) &&
       defining_op->instance_flags != 0)) {
    return iree_ok_status();
  }

  const loom_conversion_chain_match_t match =
      loom_conversion_chain_match(outer_kind, inner_kind);
  if (match.candidate == LOOM_CONVERSION_EXTRACT_SIGNED ||
      match.candidate == LOOM_CONVERSION_EXTRACT_UNSIGNED) {
    const loom_value_id_t source = loom_op_const_operands(defining_op)[0];
    // A proved lossless round trip keeps the original numeric value. Lowering
    // it to bit extraction would discard that identity and its retained bounds.
    if (loom_value_fact_table_query_numeric_identity(
            rewriter->fact_table, loom_op_const_results(op)[0]) ==
        loom_value_fact_table_query_numeric_identity(rewriter->fact_table,
                                                     source)) {
      return loom_scalar_combine_replace_with_conversion(
          rewriter, op, outer_kind, source, out_changed);
    }
    if (!allow_bitfield) {
      return iree_ok_status();
    }
  }
  if (match.candidate == LOOM_CONVERSION_NONE) {
    if (outer_kind == LOOM_CONVERSION_TRUNCI) {
      IREE_RETURN_IF_ERROR(loom_scalar_narrowing_truncate(
          rewriter, op, defining_op, out_changed));
      if (*out_changed) {
        return iree_ok_status();
      }
    }
    return loom_scalar_combine_numeric_identity(op, rewriter, outer_kind,
                                                out_changed);
  }
  if (iree_any_bit_set(match.flags, LOOM_CONVERSION_CHAIN_FLAG_NON_NEGATIVE) &&
      !loom_value_facts_is_non_negative(
          loom_rewriter_value_facts(rewriter, outer_input))) {
    return iree_ok_status();
  }
  const loom_value_id_t input = loom_op_const_operands(defining_op)[0];
  if (iree_any_bit_set(match.flags, LOOM_CONVERSION_CHAIN_FLAG_EXACT_INTEGER)) {
    const loom_scalar_type_t input_type =
        loom_type_element_type(loom_module_value_type(rewriter->module, input));
    const loom_scalar_type_t intermediate_type = loom_type_element_type(
        loom_module_value_type(rewriter->module, outer_input));
    const loom_float_integer_conversion_kind_t kind =
        inner_kind == LOOM_CONVERSION_SITOFP
            ? LOOM_FLOAT_INTEGER_CONVERSION_SIGNED
            : LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED;
    if (!loom_scalar_type_set_contains(
            loom_value_facts_integer_exact_float_types(
                input_type, kind, loom_rewriter_value_facts(rewriter, input)),
            intermediate_type)) {
      return iree_ok_status();
    }
  }
  return loom_scalar_combine_replace_with_conversion(
      rewriter, op, match.candidate, input, out_changed);
}

static iree_status_t loom_scalar_conversion_chain_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_scalar_combine_conversion(op, rewriter, false, out_changed);
}

static iree_status_t loom_scalar_shift_narrowing_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  *out_changed = false;
  const loom_value_facts_t amount_facts =
      loom_rewriter_value_facts(rewriter, loom_scalar_shli_rhs(op));
  if (!loom_value_facts_is_exact(amount_facts)) {
    return iree_ok_status();
  }
  loom_op_t* extension =
      loom_rewriter_value_defining_op(rewriter, loom_scalar_shli_lhs(op));
  if (!extension || !loom_scalar_extui_isa(extension)) {
    return iree_ok_status();
  }
  const loom_value_id_t input = loom_scalar_extui_input(extension);
  const loom_type_t input_type =
      loom_module_value_type(rewriter->module, input);
  const int32_t input_bitwidth =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));
  const int64_t amount = amount_facts.range_lo;
  if (amount <= 0 || amount >= input_bitwidth ||
      !loom_value_facts_fit_unsigned_bit_count(
          loom_rewriter_value_facts(rewriter, input),
          (uint8_t)(input_bitwidth - amount))) {
    return iree_ok_status();
  }

  // The unsigned product fits the input width but may set its sign bit.
  // Keep zero extension and drop both redundant wide no-wrap contracts.
  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* narrow_amount = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_constant_build(&rewriter->builder, loom_attr_i64(amount),
                                 input_type, op->location, &narrow_amount));
  loom_op_t* narrow_shift = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_shli_build(&rewriter->builder, /*overflow=*/0, input,
                             loom_scalar_constant_result(narrow_amount),
                             input_type, op->location, &narrow_shift));
  return loom_scalar_combine_replace_with_conversion(
      rewriter, op, LOOM_CONVERSION_EXTUI,
      loom_scalar_shli_result(narrow_shift), out_changed);
}

static const loom_rewrite_pattern_t kScalarSourceCombinePatterns[] = {
    {
        .root_kind = LOOM_OP_SCALAR_EXTF,
        .match_and_rewrite = loom_scalar_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_FPTRUNC,
        .match_and_rewrite = loom_scalar_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTSI,
        .match_and_rewrite = loom_scalar_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTUI,
        .match_and_rewrite = loom_scalar_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_TRUNCI,
        .match_and_rewrite = loom_scalar_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_SHLI,
        .match_and_rewrite = loom_scalar_shift_narrowing_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_scalar_source_combine_pattern_provider = {
        .name = IREE_SVL("scalar"),
        .patterns = kScalarSourceCombinePatterns,
        .pattern_count = IREE_ARRAYSIZE(kScalarSourceCombinePatterns),
};
