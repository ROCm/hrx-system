// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/type_refinement.h"

static bool loom_type_refinement_type_has_dimensions(loom_type_t type) {
  return loom_type_is_shaped(type) || loom_type_is_pool(type);
}

static void loom_type_refinement_merge_analysis_result(
    loom_type_refinement_result_t next, loom_type_refinement_result_t* result) {
  if (*result == LOOM_TYPE_REFINEMENT_CONFLICT) return;
  if (next == LOOM_TYPE_REFINEMENT_CONFLICT ||
      next == LOOM_TYPE_REFINEMENT_NARROWED) {
    *result = next;
  }
}

static bool loom_type_refinement_exact_dimension_fact(
    loom_value_facts_t facts, uint64_t* out_dimension,
    loom_type_refinement_result_t* out_result) {
  if (!loom_value_facts_is_exact(facts)) return false;
  if (loom_value_facts_is_float(facts) || facts.range_lo < 0 ||
      facts.range_lo > LOOM_DIM_MAX_STATIC_SIZE) {
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
    return true;
  }
  *out_dimension = loom_dim_pack_static(facts.range_lo);
  *out_result = LOOM_TYPE_REFINEMENT_NARROWED;
  return true;
}

static bool loom_type_refinement_dimension_value_id(
    uint64_t dimension, loom_value_id_t* out_value_id) {
  uint64_t payload = dimension & LOOM_DIM_PAYLOAD_MASK;
  if (payload > UINT32_MAX) return false;
  *out_value_id = (loom_value_id_t)payload;
  return true;
}

static iree_status_t loom_type_refine_dimensions_with_value_facts(
    loom_type_t current_type, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  *out_type = current_type;
  *out_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  if (!loom_type_refinement_type_has_dimensions(current_type)) {
    return iree_ok_status();
  }

  uint8_t rank = loom_type_rank(current_type);
  uint64_t candidate_dimensions[LOOM_TYPE_MAX_RANK] = {0};
  bool has_candidate_dimension = false;
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t current_dimension = loom_type_dim(current_type, i);
    candidate_dimensions[i] = current_dimension;
    if (!loom_dim_is_dynamic(current_dimension)) continue;

    loom_value_id_t dimension_value = LOOM_VALUE_ID_INVALID;
    if (!loom_type_refinement_dimension_value_id(current_dimension,
                                                 &dimension_value)) {
      continue;
    }
    loom_value_facts_t facts =
        loom_value_fact_table_lookup(fact_table, dimension_value);
    loom_type_refinement_result_t dimension_result =
        LOOM_TYPE_REFINEMENT_UNCHANGED;
    uint64_t candidate_dimension = current_dimension;
    if (!loom_type_refinement_exact_dimension_fact(facts, &candidate_dimension,
                                                   &dimension_result)) {
      continue;
    }
    if (dimension_result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      *out_result = dimension_result;
      return iree_ok_status();
    }
    candidate_dimensions[i] = candidate_dimension;
    has_candidate_dimension = true;
  }

  if (!has_candidate_dimension) return iree_ok_status();
  return loom_type_refine_shape_with_dims(current_type, candidate_dimensions,
                                          rank, arena, out_type, out_result);
}

static iree_status_t loom_type_refine_encoding_with_value_facts(
    loom_type_t current_type, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  *out_type = current_type;
  *out_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  if (!loom_type_has_ssa_encoding(current_type)) return iree_ok_status();

  loom_value_id_t encoding_value =
      (loom_value_id_t)loom_type_encoding_value_id(current_type);
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, encoding_value);
  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(&fact_table->context, facts,
                                               &summary)) {
    return iree_ok_status();
  }
  if (summary.static_spec_encoding_id == 0) return iree_ok_status();

  return loom_type_refine_encoding_with_attachment(
      current_type, summary.static_spec_encoding_id, 0, arena, out_type,
      out_result);
}

iree_status_t loom_type_refine_with_value_facts(
    loom_type_t current_type, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;

  loom_type_t dimension_refined_type = refined_type;
  loom_type_refinement_result_t dimension_result =
      LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_dimensions_with_value_facts(
      refined_type, fact_table, arena, &dimension_refined_type,
      &dimension_result));
  loom_type_refinement_merge_analysis_result(dimension_result, &result);
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    *out_type = current_type;
    *out_result = result;
    return iree_ok_status();
  }
  refined_type = dimension_refined_type;

  loom_type_t encoding_refined_type = refined_type;
  loom_type_refinement_result_t encoding_result =
      LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_encoding_with_value_facts(
      refined_type, fact_table, arena, &encoding_refined_type,
      &encoding_result));
  loom_type_refinement_merge_analysis_result(encoding_result, &result);
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    *out_type = current_type;
    *out_result = result;
    return iree_ok_status();
  }

  *out_type = encoding_refined_type;
  *out_result = result;
  return iree_ok_status();
}
