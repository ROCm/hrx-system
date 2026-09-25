// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/combine_patterns.h"

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/conversion/chain.h"
#include "loom/transforms/scalar/narrowing.h"

static loom_op_t* loom_scalar_combine_defining_op(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return NULL;
  }
  return loom_value_def_op(value);
}

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
      IREE_RETURN_IF_ERROR(
          loom_scalar_extf_build(&rewriter->builder, input, input_type,
                                 result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_FPTRUNC: {
      IREE_RETURN_IF_ERROR(loom_scalar_fptrunc_build(
          &rewriter->builder, input, input_type, result_type, op->location,
          &replacement_op));
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

static iree_status_t loom_scalar_combine_conversion_chain(
    loom_conversion_kind_t outer_kind, loom_op_t* op, loom_rewriter_t* rewriter,
    bool* out_changed) {
  *out_changed = false;
  // Registered roots are verified unary conversions.
  const loom_value_id_t outer_input = loom_op_const_operands(op)[0];
  loom_op_t* defining_op =
      loom_scalar_combine_defining_op(rewriter, outer_input);
  if (defining_op == NULL) {
    return iree_ok_status();
  }

  const loom_conversion_chain_match_t match = loom_conversion_chain_match(
      outer_kind, loom_scalar_combine_conversion_kind(defining_op));
  if (match.candidate == LOOM_CONVERSION_NONE) {
    if (outer_kind == LOOM_CONVERSION_TRUNCI) {
      return loom_scalar_narrowing_truncate(rewriter, op, defining_op,
                                            out_changed);
    }
    return iree_ok_status();
  }
  if (iree_any_bit_set(match.flags, LOOM_CONVERSION_CHAIN_FLAG_NON_NEGATIVE) &&
      !loom_value_facts_is_non_negative(
          loom_rewriter_value_facts(rewriter, outer_input))) {
    return iree_ok_status();
  }
  return loom_scalar_combine_replace_with_conversion(
      rewriter, op, match.candidate, loom_op_const_operands(defining_op)[0],
      out_changed);
}

static iree_status_t loom_scalar_extf_chain_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_scalar_combine_conversion_chain(LOOM_CONVERSION_EXTF, op,
                                              rewriter, out_changed);
}

static iree_status_t loom_scalar_fptrunc_chain_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_scalar_combine_conversion_chain(LOOM_CONVERSION_FPTRUNC, op,
                                              rewriter, out_changed);
}

static iree_status_t loom_scalar_extsi_chain_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_scalar_combine_conversion_chain(LOOM_CONVERSION_EXTSI, op,
                                              rewriter, out_changed);
}

static iree_status_t loom_scalar_extui_chain_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_scalar_combine_conversion_chain(LOOM_CONVERSION_EXTUI, op,
                                              rewriter, out_changed);
}

static iree_status_t loom_scalar_trunci_chain_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_scalar_combine_conversion_chain(LOOM_CONVERSION_TRUNCI, op,
                                              rewriter, out_changed);
}

static const loom_rewrite_pattern_t kScalarSourceCombinePatterns[] = {
    {
        .root_kind = LOOM_OP_SCALAR_EXTF,
        .match_and_rewrite = loom_scalar_extf_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_FPTRUNC,
        .match_and_rewrite = loom_scalar_fptrunc_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTSI,
        .match_and_rewrite = loom_scalar_extsi_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_EXTUI,
        .match_and_rewrite = loom_scalar_extui_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_SCALAR_TRUNCI,
        .match_and_rewrite = loom_scalar_trunci_chain_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_scalar_source_combine_pattern_provider = {
        .name = IREE_SVL("scalar"),
        .patterns = kScalarSourceCombinePatterns,
        .pattern_count = IREE_ARRAYSIZE(kScalarSourceCombinePatterns),
};
