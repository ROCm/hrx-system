// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/combine_patterns.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/construction.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/table.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/patterns.h"
#include "loom/transforms/conversion/chain.h"

static bool loom_vector_combine_value_is_non_negative(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  if (rewriter->fact_table == NULL) {
    return false;
  }
  const loom_fact_context_t* context = &rewriter->fact_table->context;
  const loom_value_facts_t facts =
      loom_rewriter_value_facts(rewriter, value_id);
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(context, facts, &uniform)) {
    return loom_value_facts_is_non_negative(uniform.element);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(context, facts, &lanes)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    if (!loom_value_facts_is_non_negative(lanes.lanes[i])) {
      return false;
    }
  }
  return true;
}

static loom_conversion_kind_t loom_vector_combine_conversion_kind(
    const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_EXTF:
      return LOOM_CONVERSION_EXTF;
    case LOOM_OP_VECTOR_FPTRUNC:
      return LOOM_CONVERSION_FPTRUNC;
    case LOOM_OP_VECTOR_EXTSI:
      return LOOM_CONVERSION_EXTSI;
    case LOOM_OP_VECTOR_EXTUI:
      return LOOM_CONVERSION_EXTUI;
    case LOOM_OP_VECTOR_TRUNCI:
      return LOOM_CONVERSION_TRUNCI;
    default:
      return LOOM_CONVERSION_NONE;
  }
}

static iree_status_t loom_vector_combine_replace_with_value(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_vector_combine_replace_with_conversion(
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
        loom_vector_combine_replace_with_value(rewriter, op, input));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  switch (replacement_kind) {
    case LOOM_CONVERSION_EXTF: {
      IREE_RETURN_IF_ERROR(loom_vector_extf_build(
          &rewriter->builder, /*instance_flags=*/0, input, input_type,
          result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_FPTRUNC: {
      IREE_RETURN_IF_ERROR(loom_vector_fptrunc_build(
          &rewriter->builder, /*instance_flags=*/0, input, input_type,
          result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_EXTSI: {
      IREE_RETURN_IF_ERROR(
          loom_vector_extsi_build(&rewriter->builder, input, input_type,
                                  result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_EXTUI: {
      IREE_RETURN_IF_ERROR(
          loom_vector_extui_build(&rewriter->builder, input, input_type,
                                  result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_TRUNCI: {
      IREE_RETURN_IF_ERROR(
          loom_vector_trunci_build(&rewriter->builder, input, input_type,
                                   result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_CONVERSION_EXTRACT_SIGNED:
    case LOOM_CONVERSION_EXTRACT_UNSIGNED: {
      const loom_type_t intermediate_type = loom_module_value_type(
          rewriter->module, loom_op_const_operands(op)[0]);
      const int32_t field_width =
          loom_scalar_type_bitwidth(loom_type_element_type(intermediate_type));
      IREE_RETURN_IF_ERROR((replacement_kind == LOOM_CONVERSION_EXTRACT_SIGNED
                                ? loom_vector_bitfield_extracts_build
                                : loom_vector_bitfield_extractu_build)(
          &rewriter->builder, input, 0, field_width, result_type, op->location,
          &replacement_op));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported vector conversion replacement");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t replacement = loom_op_const_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_vector_combine_replace_with_value(rewriter, op, replacement));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_combine_sink_conversion_through_splat(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_op_t* splat_op) {
  const loom_value_id_t scalar = loom_vector_splat_scalar(splat_op);
  const loom_type_t scalar_type =
      loom_module_value_type(rewriter->module, scalar);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  const loom_type_t scalar_result_type =
      loom_type_scalar(loom_type_element_type(result_type));

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* scalar_op = NULL;
  switch (op->kind) {
    case LOOM_OP_VECTOR_EXTF: {
      IREE_RETURN_IF_ERROR(loom_scalar_extf_build(
          &rewriter->builder, op->instance_flags, scalar, scalar_type,
          scalar_result_type, op->location, &scalar_op));
      break;
    }
    case LOOM_OP_VECTOR_FPTRUNC: {
      IREE_RETURN_IF_ERROR(loom_scalar_fptrunc_build(
          &rewriter->builder, op->instance_flags, scalar, scalar_type,
          scalar_result_type, op->location, &scalar_op));
      break;
    }
    case LOOM_OP_VECTOR_EXTSI: {
      IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(
          &rewriter->builder, scalar, scalar_type, scalar_result_type,
          op->location, &scalar_op));
      break;
    }
    case LOOM_OP_VECTOR_EXTUI: {
      IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
          &rewriter->builder, scalar, scalar_type, scalar_result_type,
          op->location, &scalar_op));
      break;
    }
    case LOOM_OP_VECTOR_TRUNCI: {
      IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
          &rewriter->builder, scalar, scalar_type, scalar_result_type,
          op->location, &scalar_op));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported vector splat conversion");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      &rewriter->builder, loom_op_const_results(scalar_op)[0], result_type,
      op->location, &replacement_op));
  loom_value_id_t replacement = loom_op_const_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_combine_replace_with_value(rewriter, op, replacement);
}

iree_status_t loom_vector_combine_conversion(loom_op_t* op,
                                             loom_rewriter_t* rewriter,
                                             bool allow_bitfield,
                                             bool* out_changed) {
  *out_changed = false;
  const loom_conversion_kind_t outer_kind =
      loom_vector_combine_conversion_kind(op);
  // Callers supply a verified unary width conversion.
  const loom_value_id_t outer_input = loom_op_const_operands(op)[0];
  loom_op_t* defining_op =
      loom_rewriter_value_defining_op(rewriter, outer_input);
  if (defining_op == NULL) {
    return iree_ok_status();
  }
  if (loom_vector_splat_isa(defining_op)) {
    IREE_RETURN_IF_ERROR(loom_vector_combine_sink_conversion_through_splat(
        rewriter, op, defining_op));
    *out_changed = true;
    return iree_ok_status();
  }

  const loom_conversion_kind_t inner_kind =
      loom_vector_combine_conversion_kind(defining_op);
  if ((loom_conversion_kind_is_float(outer_kind) && op->instance_flags != 0) ||
      (loom_conversion_kind_is_float(inner_kind) &&
       defining_op->instance_flags != 0)) {
    return iree_ok_status();
  }

  const loom_conversion_chain_match_t match =
      loom_conversion_chain_match(outer_kind, inner_kind);
  if (!allow_bitfield &&
      (match.candidate == LOOM_CONVERSION_EXTRACT_SIGNED ||
       match.candidate == LOOM_CONVERSION_EXTRACT_UNSIGNED)) {
    return iree_ok_status();
  }
  if (match.candidate == LOOM_CONVERSION_NONE) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(match.flags, LOOM_CONVERSION_CHAIN_FLAG_NON_NEGATIVE) &&
      !loom_vector_combine_value_is_non_negative(rewriter, outer_input)) {
    return iree_ok_status();
  }
  return loom_vector_combine_replace_with_conversion(
      rewriter, op, match.candidate, loom_op_const_operands(defining_op)[0],
      out_changed);
}

static iree_status_t loom_vector_conversion_chain_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_vector_combine_conversion(op, rewriter, false, out_changed);
}

static iree_status_t loom_vector_from_elements_combine_lanes_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_vector_from_elements_combine_lanes(op, rewriter, out_changed);
}

static iree_status_t loom_vector_insert_combine_elements_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_vector_insert_combine_elements(op, rewriter, out_changed);
}

static iree_status_t loom_vector_from_elements_to_table_lookup_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_vector_from_elements_to_table_lookup(op, rewriter, out_changed);
}

static iree_status_t loom_vector_table_lookup_simplify_indices_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_vector_table_lookup_simplify_indices(op, rewriter, out_changed);
}

static iree_status_t loom_vector_table_lookup_to_shuffle_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_vector_table_lookup_to_shuffle(op, rewriter, out_changed);
}

static iree_status_t loom_vector_mask_range_relations_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  *out_changed = false;
  const loom_value_id_t lower = loom_vector_mask_range_lower_bound(op);
  const loom_type_t coordinate_type =
      loom_module_value_type(rewriter->module, lower);
  const loom_scalar_type_t element_type =
      loom_type_element_type(coordinate_type);
  int32_t bit_count = loom_scalar_type_bitwidth(element_type);
  if (element_type == LOOM_SCALAR_TYPE_INDEX) {
    bit_count = loom_index_target_carrier_bitwidth(
        &rewriter->fact_table->context, element_type);
    if (bit_count <= 0) {
      return iree_ok_status();
    }
  }
  const loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_mask_range_result(op));
  const uint64_t maximum_lane_count =
      loom_value_fact_table_maximum_element_count(rewriter->fact_table,
                                                  result_type);
  if (maximum_lane_count == 0) {
    return iree_ok_status();
  }
  const loom_value_fact_vector_iota_t coordinates = {
      .base = loom_rewriter_value_facts(rewriter, lower),
      .step =
          loom_rewriter_value_facts(rewriter, loom_vector_mask_range_step(op)),
      .bit_count = (uint8_t)bit_count,
  };
  int64_t minimum = 0;
  int64_t maximum = 0;
  if (!loom_value_fact_vector_iota_bounds(coordinates, maximum_lane_count,
                                          &minimum, &maximum)) {
    return iree_ok_status();
  }
  // The bounds query checked both deltas before adding them to the base.
  const int64_t minimum_delta = minimum - coordinates.base.range_lo;
  const int64_t maximum_delta = maximum - coordinates.base.range_hi;
  loom_cleanup_pattern_context_t* cleanup =
      (loom_cleanup_pattern_context_t*)context;
  loom_symbolic_expr_context_t* expressions =
      cleanup->symbolic_expression_context;
  loom_symbolic_expr_t base = {0};
  loom_symbolic_expr_t bound = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(expressions, lower, &base));
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      expressions, loom_vector_mask_range_upper_bound(op), &bound));

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  loom_symbolic_expr_t extreme = base;
  extreme.facts = loom_value_facts_unknown();
  if (iree_checked_add_i64(base.constant, maximum_delta, &extreme.constant) &&
      iree_checked_add_i64(extreme.constant, 1, &extreme.constant)) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_with_active_facts(
        expressions, &extreme, &bound, &proof));
  }
  const bool all_true = proof == LOOM_SYMBOLIC_PROOF_TRUE;
  if (!all_true) {
    proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    if (iree_checked_add_i64(base.constant, minimum_delta, &extreme.constant)) {
      IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_with_active_facts(
          expressions, &bound, &extreme, &proof));
    }
  }
  if (proof != LOOM_SYMBOLIC_PROOF_TRUE) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_constant_build(&rewriter->builder, loom_attr_bool(all_true),
                                 result_type, op->location, &replacement_op));
  const loom_value_id_t replacement =
      loom_vector_constant_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_vector_combine_replace_with_value(rewriter, op, replacement));
  *out_changed = true;
  return iree_ok_status();
}

static const loom_rewrite_pattern_t kVectorSourceCombinePatterns[] = {
    {
        .root_kind = LOOM_OP_VECTOR_MASK_RANGE,
        .match_and_rewrite = loom_vector_mask_range_relations_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTF,
        .match_and_rewrite = loom_vector_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FPTRUNC,
        .match_and_rewrite = loom_vector_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTSI,
        .match_and_rewrite = loom_vector_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTUI,
        .match_and_rewrite = loom_vector_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRUNCI,
        .match_and_rewrite = loom_vector_conversion_chain_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_INSERT,
        .match_and_rewrite = loom_vector_insert_combine_elements_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FROM_ELEMENTS,
        .match_and_rewrite = loom_vector_from_elements_combine_lanes_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FROM_ELEMENTS,
        .match_and_rewrite = loom_vector_from_elements_to_table_lookup_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_LOOKUP,
        .match_and_rewrite = loom_vector_table_lookup_to_shuffle_pattern,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TABLE_LOOKUP,
        .match_and_rewrite = loom_vector_table_lookup_simplify_indices_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_vector_source_combine_pattern_provider = {
        .name = IREE_SVL("vector"),
        .patterns = kVectorSourceCombinePatterns,
        .pattern_count = IREE_ARRAYSIZE(kVectorSourceCombinePatterns),
};
