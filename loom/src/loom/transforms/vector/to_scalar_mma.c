// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_mma.h"

#include <stdint.h>

#include "loom/analysis/contract.h"
#include "loom/analysis/matrix_fragment_layout.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector/to_scalar_encoding.h"

typedef enum loom_vector_to_scalar_mma_numeric_e {
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED = 0,
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT = 1,
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER = 2,
} loom_vector_to_scalar_mma_numeric_t;

typedef enum loom_vector_to_scalar_mma_payload_layout_e {
  LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED = 0,
  LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR = 1,
  LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_MATRIX_ROW_MAJOR = 2,
  LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_LOGICAL_MEMORY = 3,
} loom_vector_to_scalar_mma_payload_layout_t;

typedef struct loom_vector_to_scalar_mma_fragment_t {
  // Dense logical payload value used for scalar lane extraction.
  loom_value_id_t payload;

  // Fragment facts attached to value.
  loom_vector_fragment_fact_t fact;

  // Dense logical payload type.
  loom_type_t type;

  // Logical row count for this fragment role.
  loom_vector_to_scalar_index_term_t rows;

  // Logical column count for this fragment role.
  loom_vector_to_scalar_index_term_t columns;

  // Physical-to-logical lane interpretation for dense fallback extraction.
  loom_vector_to_scalar_mma_payload_layout_t layout;
} loom_vector_to_scalar_mma_fragment_t;

static bool loom_vector_to_scalar_mma_fragment_has_schema(
    const loom_vector_fragment_fact_t* fact) {
  return iree_any_bit_set(fact->flags,
                          LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA);
}

static bool loom_vector_to_scalar_mma_query_fragment(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_fragment_role_flags_t role_flags,
    loom_vector_to_scalar_mma_fragment_t* out_fragment) {
  *out_fragment = (loom_vector_to_scalar_mma_fragment_t){0};
  loom_value_facts_t facts = loom_rewriter_value_facts(state->rewriter, value);
  loom_vector_fragment_fact_t fact;
  if (!loom_vector_fragment_fact_query_value_facts(
          &state->rewriter->fact_table->context, facts, &fact) ||
      !iree_any_bit_set(fact.role_flags, role_flags) || fact.shape_rank != 2 ||
      (fact.auxiliary.present_keys != 0 &&
       !loom_vector_to_scalar_mma_fragment_has_schema(&fact))) {
    return false;
  }

  loom_value_id_t payload = value;
  loom_op_t* def_op =
      loom_vector_to_scalar_value_def_op(state->rewriter->module, value);
  if (def_op != NULL && loom_vector_fragment_isa(def_op)) {
    payload = loom_vector_fragment_data(def_op);
  }

  *out_fragment = (loom_vector_to_scalar_mma_fragment_t){
      .payload = payload,
      .fact = fact,
      .type = loom_module_value_type(state->rewriter->module, payload),
      .rows = loom_vector_to_scalar_value_term(state, fact.shape_value_ids[0]),
      .columns =
          loom_vector_to_scalar_value_term(state, fact.shape_value_ids[1]),
  };
  return true;
}

static uint32_t loom_vector_to_scalar_mma_fragment_query_rejection_bits(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_fragment_role_flags_t role_flags) {
  if (state->rewriter->fact_table == NULL || value == LOOM_VALUE_ID_INVALID) {
    return LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  loom_value_facts_t facts = loom_rewriter_value_facts(state->rewriter, value);
  loom_vector_fragment_fact_t fact;
  if (!loom_vector_fragment_fact_query_value_facts(
          &state->rewriter->fact_table->context, facts, &fact) ||
      !iree_any_bit_set(fact.role_flags, role_flags)) {
    return LOOM_CONTRACT_REJECTION_ROLE;
  }
  if (fact.shape_rank != 2) {
    return LOOM_CONTRACT_REJECTION_SHAPE;
  }
  if (fact.auxiliary.present_keys != 0 &&
      !loom_vector_to_scalar_mma_fragment_has_schema(&fact)) {
    return LOOM_CONTRACT_REJECTION_SCHEMA |
           LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND;
  }
  return LOOM_CONTRACT_REJECTION_NONE;
}

static bool loom_vector_to_scalar_mma_logical_element_count(
    int64_t rows, int64_t columns, uint64_t* out_element_count) {
  if (rows < 0 || columns < 0) {
    return false;
  }
  uint64_t row_count = (uint64_t)rows;
  uint64_t column_count = (uint64_t)columns;
  if (column_count == 0) {
    *out_element_count = 0;
    return true;
  }
  if (row_count > UINT64_MAX / column_count) {
    return false;
  }
  *out_element_count = row_count * column_count;
  return true;
}

static bool loom_vector_to_scalar_mma_product_count(uint64_t lhs, uint64_t rhs,
                                                    uint64_t* out_product) {
  if (rhs == 0) {
    *out_product = 0;
    return true;
  }
  if (lhs > UINT64_MAX / rhs) {
    return false;
  }
  *out_product = lhs * rhs;
  return true;
}

static bool loom_vector_to_scalar_mma_term_is_static_non_negative(
    loom_vector_to_scalar_index_term_t term, int64_t* out_value) {
  if (term.is_dynamic || term.static_value < 0) {
    return false;
  }
  *out_value = term.static_value;
  return true;
}

static bool loom_vector_to_scalar_mma_term_is_static_value(
    loom_vector_to_scalar_index_term_t term, int64_t value) {
  return !term.is_dynamic && term.static_value == value;
}

static bool loom_vector_to_scalar_mma_terms_equal(
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs) {
  if (lhs.is_dynamic || rhs.is_dynamic) {
    return lhs.is_dynamic && rhs.is_dynamic &&
           lhs.dynamic_value == rhs.dynamic_value;
  }
  return lhs.static_value == rhs.static_value;
}

static bool loom_vector_to_scalar_mma_term_matches_product(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t candidate,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs) {
  int64_t lhs_static = 0;
  int64_t rhs_static = 0;
  if (loom_vector_to_scalar_mma_term_is_static_non_negative(lhs, &lhs_static) &&
      loom_vector_to_scalar_mma_term_is_static_non_negative(rhs, &rhs_static)) {
    uint64_t product = 0;
    if (!loom_vector_to_scalar_mma_logical_element_count(lhs_static, rhs_static,
                                                         &product) ||
        product > INT64_MAX) {
      return false;
    }
    return loom_vector_to_scalar_mma_term_is_static_value(candidate,
                                                          (int64_t)product);
  }
  if (!candidate.is_dynamic) {
    return false;
  }
  loom_op_t* def_op = loom_vector_to_scalar_value_def_op(
      state->rewriter->module, candidate.dynamic_value);
  if (!def_op || !loom_index_mul_isa(def_op)) {
    return false;
  }
  loom_vector_to_scalar_index_term_t mul_lhs =
      loom_vector_to_scalar_value_term(state, loom_index_mul_lhs(def_op));
  loom_vector_to_scalar_index_term_t mul_rhs =
      loom_vector_to_scalar_value_term(state, loom_index_mul_rhs(def_op));
  return (loom_vector_to_scalar_mma_terms_equal(mul_lhs, lhs) &&
          loom_vector_to_scalar_mma_terms_equal(mul_rhs, rhs)) ||
         (loom_vector_to_scalar_mma_terms_equal(mul_lhs, rhs) &&
          loom_vector_to_scalar_mma_terms_equal(mul_rhs, lhs));
}

static bool loom_vector_to_scalar_mma_type_is_dense_payload(
    loom_vector_to_scalar_state_t* state, loom_type_t type,
    loom_vector_to_scalar_index_term_t rows,
    loom_vector_to_scalar_index_term_t columns,
    loom_vector_to_scalar_mma_payload_layout_t* out_layout) {
  *out_layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED;
  if (!loom_type_is_vector(type)) {
    return false;
  }
  if (loom_type_rank(type) == 2) {
    if (!loom_vector_to_scalar_mma_terms_equal(
            loom_vector_to_scalar_dim_bound_term(state, type, 0), rows) ||
        !loom_vector_to_scalar_mma_terms_equal(
            loom_vector_to_scalar_dim_bound_term(state, type, 1), columns)) {
      return false;
    }
    *out_layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_MATRIX_ROW_MAJOR;
    return true;
  }
  if (loom_type_rank(type) != 1) {
    return false;
  }
  loom_vector_to_scalar_index_term_t flat_extent =
      loom_vector_to_scalar_dim_bound_term(state, type, 0);
  if ((loom_vector_to_scalar_mma_term_is_static_value(columns, 1) &&
       loom_vector_to_scalar_mma_terms_equal(flat_extent, rows)) ||
      (loom_vector_to_scalar_mma_term_is_static_value(rows, 1) &&
       loom_vector_to_scalar_mma_terms_equal(flat_extent, columns)) ||
      loom_vector_to_scalar_mma_term_matches_product(state, flat_extent, rows,
                                                     columns)) {
    *out_layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR;
    return true;
  }

  int64_t static_rows = 0;
  int64_t static_columns = 0;
  uint64_t physical_element_count = 0;
  uint64_t logical_element_count = 0;
  if (loom_vector_to_scalar_mma_term_is_static_non_negative(rows,
                                                            &static_rows) &&
      loom_vector_to_scalar_mma_term_is_static_non_negative(columns,
                                                            &static_columns) &&
      loom_vector_to_scalar_mma_logical_element_count(
          static_rows, static_columns, &logical_element_count) &&
      loom_type_is_all_static(type) &&
      loom_type_static_element_count(type, &physical_element_count) &&
      physical_element_count == logical_element_count) {
    *out_layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR;
    return true;
  }
  return false;
}

static bool loom_vector_to_scalar_mma_value_is_fragment_load(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value) {
  loom_op_t* def_op =
      loom_vector_to_scalar_value_def_op(state->rewriter->module, value);
  return def_op != NULL && loom_vector_fragment_load_isa(def_op);
}

static loom_vector_to_scalar_encoded_matrix_operand_t
loom_vector_to_scalar_mma_encoded_operand(
    const loom_vector_to_scalar_mma_fragment_t* fragment) {
  return (loom_vector_to_scalar_encoded_matrix_operand_t){
      .schema = fragment->fact.encoded_operand,
      .auxiliary = fragment->fact.auxiliary,
      .rows = fragment->rows,
      .columns = fragment->columns,
  };
}

static bool loom_vector_to_scalar_mma_fragment_supports_logical_lanes(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_fragment_t* fragment) {
  if (loom_vector_to_scalar_mma_fragment_has_schema(&fragment->fact) &&
      fragment->fact.encoded_operand.payload_packing !=
          LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES) {
    return false;
  }
  if (loom_vector_to_scalar_mma_type_is_dense_payload(
          state, fragment->type, fragment->rows, fragment->columns,
          &fragment->layout)) {
    return true;
  }
  if (loom_vector_to_scalar_mma_fragment_has_schema(&fragment->fact)) {
    return false;
  }
  if (loom_vector_to_scalar_mma_value_is_fragment_load(state,
                                                       fragment->payload)) {
    fragment->layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_LOGICAL_MEMORY;
    return true;
  }
  return false;
}

static uint32_t loom_vector_to_scalar_mma_fragment_logical_lane_rejection_bits(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment) {
  if (loom_vector_to_scalar_mma_fragment_has_schema(&fragment->fact) &&
      fragment->fact.encoded_operand.payload_packing !=
          LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES) {
    return LOOM_CONTRACT_REJECTION_SCHEMA;
  }
  loom_vector_to_scalar_mma_payload_layout_t layout =
      LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED;
  if (loom_vector_to_scalar_mma_type_is_dense_payload(
          state, fragment->type, fragment->rows, fragment->columns, &layout)) {
    return LOOM_CONTRACT_REJECTION_NONE;
  }
  if (loom_vector_to_scalar_mma_fragment_has_schema(&fragment->fact)) {
    return LOOM_CONTRACT_REJECTION_SCHEMA | LOOM_CONTRACT_REJECTION_SHAPE;
  }
  if (loom_vector_to_scalar_mma_value_is_fragment_load(state,
                                                       fragment->payload)) {
    return LOOM_CONTRACT_REJECTION_NONE;
  }
  return LOOM_CONTRACT_REJECTION_FRAGMENT;
}

static bool loom_vector_to_scalar_mma_result_is_dense_payload(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* init) {
  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_vector_mma_result(state->op));
  loom_vector_to_scalar_mma_payload_layout_t result_layout =
      LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED;
  return loom_vector_to_scalar_mma_type_is_dense_payload(
             state, result_type, init->rows, init->columns, &result_layout) &&
         init->layout == result_layout &&
         loom_type_equal(init->type, result_type);
}

static bool loom_vector_to_scalar_mma_numeric_kind(
    loom_scalar_type_t lhs_type, loom_scalar_type_t rhs_type,
    loom_scalar_type_t accumulator_type,
    loom_vector_to_scalar_mma_numeric_t* out_numeric) {
  *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (loom_scalar_type_is_float(lhs_type) &&
      loom_scalar_type_is_float(rhs_type) &&
      loom_scalar_type_is_float(accumulator_type)) {
    *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT;
    return true;
  }
  if (loom_scalar_type_is_integer(lhs_type) &&
      loom_scalar_type_is_integer(rhs_type) &&
      loom_scalar_type_is_integer(accumulator_type)) {
    *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER;
    return true;
  }
  return false;
}

static bool loom_vector_to_scalar_mma_cast_is_supported(
    loom_scalar_type_t input_type, loom_scalar_type_t result_type,
    loom_vector_to_scalar_mma_numeric_t numeric) {
  if (input_type == result_type) {
    return true;
  }
  const int32_t input_bitwidth = loom_scalar_type_bitwidth(input_type);
  const int32_t result_bitwidth = loom_scalar_type_bitwidth(result_type);
  if (input_bitwidth <= 0 || result_bitwidth <= 0 ||
      input_bitwidth >= result_bitwidth) {
    return false;
  }
  switch (numeric) {
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT:
      return loom_scalar_type_is_float(input_type) &&
             loom_scalar_type_is_float(result_type);
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER:
      return loom_scalar_type_is_integer(input_type) &&
             loom_scalar_type_is_integer(result_type);
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED:
    default:
      return false;
  }
}

static bool loom_vector_to_scalar_mma_semantics_match(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_fragment_t* lhs,
    loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_mma_numeric_t* out_numeric) {
  *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_vector_mma_result(state->op));
  if (!loom_vector_to_scalar_mma_terms_equal(lhs->columns, rhs->rows) ||
      !loom_vector_to_scalar_mma_terms_equal(lhs->rows, init->rows) ||
      !loom_vector_to_scalar_mma_terms_equal(rhs->columns, init->columns) ||
      !loom_type_is_vector(result_type)) {
    return false;
  }
  if (loom_vector_to_scalar_mma_fragment_has_schema(&init->fact)) {
    return false;
  }

  const loom_scalar_type_t accumulator_element_type =
      loom_type_element_type(init->type);
  if (loom_type_element_type(result_type) != accumulator_element_type) {
    return false;
  }
  const loom_type_t accumulator_type =
      loom_type_scalar(accumulator_element_type);
  const bool lhs_encoded =
      loom_vector_to_scalar_mma_fragment_has_schema(&lhs->fact);
  const bool rhs_encoded =
      loom_vector_to_scalar_mma_fragment_has_schema(&rhs->fact);
  if (lhs_encoded || rhs_encoded) {
    if (!loom_scalar_type_is_float(accumulator_element_type)) {
      return false;
    }
    if (lhs_encoded) {
      const loom_vector_to_scalar_encoded_matrix_operand_t operand =
          loom_vector_to_scalar_mma_encoded_operand(lhs);
      if (!loom_vector_to_scalar_encoded_matrix_operand_is_supported(
              state, &operand, loom_vector_to_scalar_lane_type(lhs->type),
              accumulator_type)) {
        return false;
      }
    }
    if (rhs_encoded) {
      const loom_vector_to_scalar_encoded_matrix_operand_t operand =
          loom_vector_to_scalar_mma_encoded_operand(rhs);
      if (!loom_vector_to_scalar_encoded_matrix_operand_is_supported(
              state, &operand, loom_vector_to_scalar_lane_type(rhs->type),
              accumulator_type)) {
        return false;
      }
    }
  }

  const loom_scalar_type_t lhs_element_type =
      lhs_encoded ? accumulator_element_type
                  : loom_type_element_type(lhs->type);
  const loom_scalar_type_t rhs_element_type =
      rhs_encoded ? accumulator_element_type
                  : loom_type_element_type(rhs->type);
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_numeric_kind(
          lhs_element_type, rhs_element_type, accumulator_element_type,
          &numeric)) {
    return false;
  }
  if (!loom_vector_to_scalar_mma_cast_is_supported(
          lhs_element_type, accumulator_element_type, numeric) ||
      !loom_vector_to_scalar_mma_cast_is_supported(
          rhs_element_type, accumulator_element_type, numeric)) {
    return false;
  }
  *out_numeric = numeric;
  return true;
}

static bool loom_vector_to_scalar_mma_query_semantic_operands(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_fragment_t* lhs,
    loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_mma_numeric_t* out_numeric) {
  if (!loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_lhs(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS, lhs) ||
      !loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_rhs(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS, rhs) ||
      !loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_init(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
              LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT,
          init)) {
    return false;
  }
  return loom_vector_to_scalar_mma_semantics_match(state, lhs, rhs, init,
                                                   out_numeric);
}

static uint32_t loom_vector_to_scalar_mma_semantic_rejection_bits(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init) {
  loom_contract_rejection_bits_t rejection_bits = LOOM_CONTRACT_REJECTION_NONE;
  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_vector_mma_result(state->op));
  if (!loom_vector_to_scalar_mma_terms_equal(lhs->columns, rhs->rows) ||
      !loom_vector_to_scalar_mma_terms_equal(lhs->rows, init->rows) ||
      !loom_vector_to_scalar_mma_terms_equal(rhs->columns, init->columns) ||
      !loom_type_is_vector(result_type)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_SHAPE;
  }
  rejection_bits |=
      loom_vector_to_scalar_mma_fragment_logical_lane_rejection_bits(state,
                                                                     lhs);
  rejection_bits |=
      loom_vector_to_scalar_mma_fragment_logical_lane_rejection_bits(state,
                                                                     rhs);
  rejection_bits |=
      loom_vector_to_scalar_mma_fragment_logical_lane_rejection_bits(state,
                                                                     init);
  if (loom_vector_to_scalar_mma_fragment_has_schema(&init->fact)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_SCHEMA;
  }

  const loom_scalar_type_t accumulator_element_type =
      loom_type_element_type(init->type);
  if (loom_type_element_type(result_type) != accumulator_element_type) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_NUMERIC;
  }
  const loom_type_t accumulator_type =
      loom_type_scalar(accumulator_element_type);
  const bool lhs_encoded =
      loom_vector_to_scalar_mma_fragment_has_schema(&lhs->fact);
  const bool rhs_encoded =
      loom_vector_to_scalar_mma_fragment_has_schema(&rhs->fact);
  if ((lhs_encoded || rhs_encoded) &&
      !loom_scalar_type_is_float(accumulator_element_type)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_NUMERIC;
  }
  if (lhs_encoded) {
    const loom_vector_to_scalar_encoded_matrix_operand_t operand =
        loom_vector_to_scalar_mma_encoded_operand(lhs);
    rejection_bits |=
        loom_vector_to_scalar_encoded_matrix_operand_rejection_bits(
            state, &operand, loom_vector_to_scalar_lane_type(lhs->type),
            accumulator_type);
  }
  if (rhs_encoded) {
    const loom_vector_to_scalar_encoded_matrix_operand_t operand =
        loom_vector_to_scalar_mma_encoded_operand(rhs);
    rejection_bits |=
        loom_vector_to_scalar_encoded_matrix_operand_rejection_bits(
            state, &operand, loom_vector_to_scalar_lane_type(rhs->type),
            accumulator_type);
  }

  const loom_scalar_type_t lhs_element_type =
      lhs_encoded ? accumulator_element_type
                  : loom_type_element_type(lhs->type);
  const loom_scalar_type_t rhs_element_type =
      rhs_encoded ? accumulator_element_type
                  : loom_type_element_type(rhs->type);
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_numeric_kind(
          lhs_element_type, rhs_element_type, accumulator_element_type,
          &numeric) ||
      !loom_vector_to_scalar_mma_cast_is_supported(
          lhs_element_type, accumulator_element_type, numeric) ||
      !loom_vector_to_scalar_mma_cast_is_supported(
          rhs_element_type, accumulator_element_type, numeric)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_NUMERIC;
  }
  return rejection_bits;
}

static iree_status_t loom_vector_to_scalar_mma_cast_float(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_scalar_type_t input_scalar_type, loom_type_t result_type,
    loom_value_id_t* out_result) {
  loom_type_t input_type = loom_type_scalar(input_scalar_type);
  if (loom_type_equal(input_type, result_type)) {
    *out_result = input;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extf_build(&state->rewriter->builder, input,
                                              input_type, result_type,
                                              state->location, &cast_op));
  *out_result = loom_scalar_extf_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_cast_integer(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_scalar_type_t input_scalar_type, loom_type_t result_type,
    loom_value_id_t* out_result) {
  return loom_vector_to_scalar_cast_integer_lane(
      state, input, loom_type_scalar(input_scalar_type), result_type,
      /*signed_extend=*/true, out_result);
}

static iree_status_t loom_vector_to_scalar_mma_build_accumulate(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric, loom_value_id_t lhs,
    loom_scalar_type_t lhs_scalar_type, loom_value_id_t rhs,
    loom_scalar_type_t rhs_scalar_type, loom_value_id_t accumulator,
    loom_type_t accumulator_type, loom_value_id_t* out_next) {
  loom_value_id_t cast_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cast_rhs = LOOM_VALUE_ID_INVALID;
  switch (numeric) {
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT: {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_float(
          state, lhs, lhs_scalar_type, accumulator_type, &cast_lhs));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_float(
          state, rhs, rhs_scalar_type, accumulator_type, &cast_rhs));
      loom_op_t* fma_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_fmaf_build(
          &state->rewriter->builder, 0, cast_lhs, cast_rhs, accumulator,
          accumulator_type, state->location, &fma_op));
      *out_next = loom_scalar_fmaf_result(fma_op);
      return iree_ok_status();
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER: {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_integer(
          state, lhs, lhs_scalar_type, accumulator_type, &cast_lhs));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_integer(
          state, rhs, rhs_scalar_type, accumulator_type, &cast_rhs));
      loom_op_t* fma_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_fmai_build(
          &state->rewriter->builder, 0, cast_lhs, cast_rhs, accumulator,
          accumulator_type, state->location, &fma_op));
      *out_next = loom_scalar_fmai_result(fma_op);
      return iree_ok_status();
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported vector.mma scalar numeric kind");
  }
}

static iree_status_t loom_vector_to_scalar_mma_build_matrix_indices(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_list_t* out_indices) {
  switch (fragment->layout) {
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR: {
      loom_vector_to_scalar_index_term_t ordinal = {0};
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, row, fragment->columns,
          &ordinal));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, ordinal, column,
          &ordinal));
      return loom_vector_to_scalar_terms_to_index_list(state, &ordinal, 1,
                                                       out_indices);
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_MATRIX_ROW_MAJOR: {
      loom_vector_to_scalar_index_term_t terms[2] = {row, column};
      return loom_vector_to_scalar_terms_to_index_list(state, terms, 2,
                                                       out_indices);
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_LOGICAL_MEMORY: {
      loom_vector_to_scalar_index_term_t terms[2] = {row, column};
      return loom_vector_to_scalar_terms_to_index_list(state, terms, 2,
                                                       out_indices);
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported vector.mma dense payload layout");
  }
}

static iree_status_t loom_vector_to_scalar_mma_build_matrix_ordinal(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_term_t* out_ordinal) {
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, row, fragment->columns,
      out_ordinal));
  return loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, *out_ordinal, column,
      out_ordinal);
}

static iree_status_t loom_vector_to_scalar_mma_materialize_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, loom_type_t result_type,
    loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_list_t indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_matrix_indices(
      state, fragment, row, column, &indices));
  loom_value_id_t raw_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, fragment->payload, indices, &raw_lane));
  if (!loom_vector_to_scalar_mma_fragment_has_schema(&fragment->fact)) {
    *out_lane = raw_lane;
    return iree_ok_status();
  }
  loom_vector_to_scalar_index_term_t ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_matrix_ordinal(
      state, fragment, row, column, &ordinal));
  const loom_vector_to_scalar_encoded_matrix_operand_t operand =
      loom_vector_to_scalar_mma_encoded_operand(fragment);
  return loom_vector_to_scalar_build_encoded_matrix_lane(
      state, &operand, raw_lane,
      loom_vector_to_scalar_lane_type(fragment->type), result_type, row, column,
      ordinal, out_lane);
}

static iree_status_t loom_vector_to_scalar_mma_insert_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment, loom_value_id_t lane,
    loom_value_id_t aggregate, loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, loom_value_id_t* out_aggregate) {
  loom_vector_to_scalar_index_list_t indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_matrix_indices(
      state, fragment, row, column, &indices));
  return loom_vector_to_scalar_insert_lane(
      state, lane, aggregate, fragment->type, indices, out_aggregate);
}

static iree_status_t loom_vector_to_scalar_mma_extract_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment, int64_t row,
    int64_t column, loom_type_t result_type, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_mma_materialize_matrix_lane(
      state, fragment, loom_vector_to_scalar_static_term(row),
      loom_vector_to_scalar_static_term(column), result_type, out_lane);
}

static loom_scalar_type_t loom_vector_to_scalar_mma_fragment_scalar_type(
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    loom_scalar_type_t accumulator_element_type) {
  return loom_vector_to_scalar_mma_fragment_has_schema(&fragment->fact)
             ? accumulator_element_type
             : loom_type_element_type(fragment->type);
}

static iree_status_t loom_vector_to_scalar_mma_build_result_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t row,
    int64_t column, int64_t k_count, loom_value_id_t* out_lane) {
  loom_type_t accumulator_type =
      loom_type_scalar(loom_type_element_type(init->type));
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_matrix_lane(
      state, init, row, column, accumulator_type, &accumulator));

  const loom_scalar_type_t accumulator_element_type =
      loom_type_element_type(accumulator_type);
  const loom_scalar_type_t lhs_scalar_type =
      loom_vector_to_scalar_mma_fragment_scalar_type(lhs,
                                                     accumulator_element_type);
  const loom_scalar_type_t rhs_scalar_type =
      loom_vector_to_scalar_mma_fragment_scalar_type(rhs,
                                                     accumulator_element_type);
  for (int64_t k = 0; k < k_count; ++k) {
    loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_matrix_lane(
        state, lhs, row, k, accumulator_type, &lhs_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_matrix_lane(
        state, rhs, k, column, accumulator_type, &rhs_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_accumulate(
        state, numeric, lhs_lane, lhs_scalar_type, rhs_lane, rhs_scalar_type,
        accumulator, accumulator_type, &accumulator));
    loom_vector_to_scalar_record_lane_materialized(state);
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

static bool loom_vector_to_scalar_mma_shapes_are_static(
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t* out_m,
    int64_t* out_n, int64_t* out_k) {
  int64_t lhs_rows = 0;
  int64_t lhs_columns = 0;
  int64_t rhs_rows = 0;
  int64_t rhs_columns = 0;
  int64_t init_rows = 0;
  int64_t init_columns = 0;
  if (!loom_vector_to_scalar_mma_term_is_static_non_negative(lhs->rows,
                                                             &lhs_rows) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(lhs->columns,
                                                             &lhs_columns) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(rhs->rows,
                                                             &rhs_rows) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(rhs->columns,
                                                             &rhs_columns) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(init->rows,
                                                             &init_rows) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(init->columns,
                                                             &init_columns)) {
    return false;
  }
  if (lhs_rows != init_rows || lhs_columns != rhs_rows ||
      rhs_columns != init_columns) {
    return false;
  }
  *out_m = init_rows;
  *out_n = init_columns;
  *out_k = lhs_columns;
  return true;
}

static iree_status_t loom_vector_to_scalar_mma_build_accumulate_at(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_term_t k, loom_value_id_t accumulator,
    loom_value_id_t* out_next) {
  loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
  loom_type_t accumulator_type =
      loom_type_scalar(loom_type_element_type(state->vector_type));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_materialize_matrix_lane(
      state, lhs, row, k, accumulator_type, &lhs_lane));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_materialize_matrix_lane(
      state, rhs, k, column, accumulator_type, &rhs_lane));
  const loom_scalar_type_t accumulator_element_type =
      loom_type_element_type(accumulator_type);
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_accumulate(
      state, numeric, lhs_lane,
      loom_vector_to_scalar_mma_fragment_scalar_type(lhs,
                                                     accumulator_element_type),
      rhs_lane,
      loom_vector_to_scalar_mma_fragment_scalar_type(rhs,
                                                     accumulator_element_type),
      accumulator, accumulator_type, out_next));
  loom_vector_to_scalar_record_lane_materialized(state);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_build_loop_bounds(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t upper_bound_term,
    loom_value_id_t* out_lower_bound, loom_value_id_t* out_upper_bound,
    loom_value_id_t* out_step) {
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, out_lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_term_value(state, upper_bound_term,
                                                        out_upper_bound));
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, out_step);
}

static iree_status_t loom_vector_to_scalar_mma_accumulator_loop(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, loom_value_id_t init_lane,
    loom_value_id_t* out_lane) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_loop_bounds(
      state, lhs->columns, &lower_bound, &upper_bound, &step));

  loom_type_t accumulator_type =
      loom_type_scalar(loom_type_element_type(state->vector_type));
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &init_lane, 1, &accumulator_type, 1, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_vector_to_scalar_index_term_t k = loom_vector_to_scalar_dynamic_term(
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t next_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_accumulate_at(
      state, numeric, lhs, rhs, row, column, k, accumulator_arg,
      &next_accumulator));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &next_accumulator, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_lane = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_build_dynamic_result_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, loom_value_id_t* out_lane) {
  loom_value_id_t init_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_materialize_matrix_lane(
      state, init, row, column,
      loom_type_scalar(loom_type_element_type(state->vector_type)),
      &init_lane));
  return loom_vector_to_scalar_mma_accumulator_loop(
      state, numeric, lhs, rhs, row, column, init_lane, out_lane);
}

static iree_status_t loom_vector_to_scalar_mma_column_loop(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_index_term_t row, loom_value_id_t current_aggregate,
    loom_value_id_t* out_aggregate) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_loop_bounds(
      state, init->columns, &lower_bound, &upper_bound, &step));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &current_aggregate, 1, &init->type, 1, NULL, 0,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_vector_to_scalar_index_term_t column =
      loom_vector_to_scalar_dynamic_term(
          loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_dynamic_result_lane(
      state, numeric, lhs, rhs, init, row, column, &lane));
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_insert_matrix_lane(
      state, init, lane, aggregate_arg, row, column, &yielded_aggregate));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_aggregate, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_aggregate = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_row_loop(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_value_id_t* out_payload) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_loop_bounds(
      state, init->rows, &lower_bound, &upper_bound, &step));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &init->payload, 1, &init->type, 1, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_vector_to_scalar_index_term_t row = loom_vector_to_scalar_dynamic_term(
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_column_loop(
      state, numeric, lhs, rhs, init, row, aggregate_arg, &yielded_aggregate));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_aggregate, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_payload = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_build_result_fragment(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* init, loom_value_id_t payload,
    loom_value_id_t* out_replacement) {
  loom_op_t* fragment_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_fragment_build(
      &state->rewriter->builder, LOOM_VECTOR_ROLE_RESULT, payload,
      init->fact.shape_value_ids[0], init->fact.shape_value_ids[1], NULL, 0,
      NULL, 0, init->type, state->location, &fragment_op));
  *out_replacement = loom_vector_fragment_result(fragment_op);
  return iree_ok_status();
}

static bool loom_vector_to_scalar_mma_physical_element_count(
    const loom_matrix_fragment_role_layout_t* role_layout,
    uint16_t* out_count) {
  *out_count = 0;
  if (role_layout == NULL) {
    return false;
  }
  const uint32_t element_count = (uint32_t)role_layout->register_count *
                                 role_layout->elements_per_register;
  if (element_count == 0 || element_count > UINT16_MAX) {
    return false;
  }
  *out_count = (uint16_t)element_count;
  return true;
}

static bool loom_vector_to_scalar_mma_physical_payload_matches_role(
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    const loom_matrix_fragment_role_layout_t* role_layout) {
  if (loom_vector_to_scalar_mma_fragment_has_schema(&fragment->fact) ||
      fragment->fact.auxiliary.present_keys != 0 ||
      !loom_type_is_vector(fragment->type) ||
      loom_type_rank(fragment->type) != 1 ||
      !loom_type_is_all_static(fragment->type)) {
    return false;
  }
  uint16_t physical_element_count = 0;
  if (!loom_vector_to_scalar_mma_physical_element_count(
          role_layout, &physical_element_count)) {
    return false;
  }
  uint64_t payload_element_count = 0;
  if (!loom_type_static_element_count(fragment->type, &payload_element_count) ||
      payload_element_count != physical_element_count) {
    return false;
  }
  return loom_scalar_type_bitwidth(loom_type_element_type(fragment->type)) ==
         (int32_t)role_layout->element_bit_count;
}

static iree_status_t loom_vector_to_scalar_mma_extract_physical_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_matrix_fragment_physical_element_t element,
    loom_value_id_t* out_lane) {
  const int64_t ordinal =
      (int64_t)element.register_index * role_layout->elements_per_register +
      element.element_index;
  const int64_t static_indices[1] = {ordinal};
  const loom_vector_to_scalar_index_list_t indices = {
      .static_indices = static_indices,
      .rank = 1,
  };
  return loom_vector_to_scalar_materialize_lane(state, fragment->payload,
                                                indices, out_lane);
}

static iree_status_t loom_vector_to_scalar_mma_build_lane_id_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t* out_lane_id) {
  loom_op_t* lane_id_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_lane_id_build(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, &lane_id_op));
  *out_lane_id = loom_vector_to_scalar_dynamic_term(
      loom_kernel_subgroup_lane_id_result(lane_id_op));
  return iree_ok_status();
}

static bool loom_vector_to_scalar_mma_role_layouts_match(
    const loom_matrix_fragment_role_layout_t* lhs,
    const loom_matrix_fragment_role_layout_t* rhs) {
  return lhs->map_kind == rhs->map_kind &&
         lhs->register_count == rhs->register_count &&
         lhs->elements_per_register == rhs->elements_per_register &&
         lhs->element_bit_count == rhs->element_bit_count &&
         lhs->coordinate_flags == rhs->coordinate_flags;
}

static bool loom_vector_to_scalar_mma_result_layout_is_supported(
    const loom_matrix_fragment_layout_t* layout) {
  const loom_matrix_fragment_role_layout_t* result = &layout->result;
  if (result->coordinate_flags != (LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
                                   LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN) ||
      layout->tile_shape.result_column_count == 0) {
    return false;
  }
  switch (result->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
      return true;
    case LOOM_MATRIX_FRAGMENT_MAP_UNKNOWN:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
    default:
      return false;
  }
}

static bool loom_vector_to_scalar_mma_reduction_role_layout_is_supported(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout) {
  const bool carries_row = role_layout->coordinate_flags ==
                           (LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
                            LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION);
  const bool carries_column = role_layout->coordinate_flags ==
                              (LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN |
                               LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION);
  if (!carries_row && !carries_column) {
    return false;
  }
  if (layout->tile_shape.reduction_count == 0) {
    return false;
  }
  const uint32_t reduction_group_span = (uint32_t)role_layout->register_count *
                                        role_layout->elements_per_register;
  if (reduction_group_span == 0) {
    return false;
  }
  switch (role_layout->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
      return carries_row &&
             layout->tile_shape.reduction_count <= reduction_group_span;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
      return carries_column &&
             layout->tile_shape.reduction_count <= reduction_group_span;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
      return carries_row &&
             ((uint32_t)(layout->tile_shape.reduction_count - 1) /
                  reduction_group_span +
              1u) * layout->tile_shape.result_row_count <=
                 layout->wave_size;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
      return carries_column &&
             ((uint32_t)(layout->tile_shape.reduction_count - 1) /
                  reduction_group_span +
              1u) * layout->tile_shape.result_column_count <=
                 layout->wave_size;
    case LOOM_MATRIX_FRAGMENT_MAP_UNKNOWN:
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
    default:
      return false;
  }
}

static bool loom_vector_to_scalar_mma_distributed_static_is_supported(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t* out_k) {
  if (state->matrix_fragment_layout == NULL ||
      !iree_any_bit_set(
          state->flags,
          LOOM_VECTOR_TO_SCALAR_FLAG_ALLOW_SUBGROUP_COMMUNICATION)) {
    return false;
  }
  const loom_matrix_fragment_layout_t* layout = state->matrix_fragment_layout;
  if (layout->wave_size == 0 || layout->tile_shape.reduction_count == 0) {
    return false;
  }
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  if (!loom_vector_to_scalar_mma_shapes_are_static(lhs, rhs, init, &m, &n,
                                                   &k) ||
      m != layout->tile_shape.result_row_count ||
      n != layout->tile_shape.result_column_count ||
      k != layout->tile_shape.reduction_count) {
    return false;
  }
  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_vector_mma_result(state->op));
  if (!loom_type_equal(result_type, init->type)) {
    return false;
  }
  if (!loom_vector_to_scalar_mma_physical_payload_matches_role(lhs,
                                                               &layout->lhs) ||
      !loom_vector_to_scalar_mma_physical_payload_matches_role(rhs,
                                                               &layout->rhs) ||
      !loom_vector_to_scalar_mma_physical_payload_matches_role(
          init, &layout->accumulator) ||
      !loom_vector_to_scalar_mma_physical_payload_matches_role(
          &(loom_vector_to_scalar_mma_fragment_t){.fact = init->fact,
                                                  .type = result_type},
          &layout->result)) {
    return false;
  }
  if (!loom_vector_to_scalar_mma_role_layouts_match(&layout->accumulator,
                                                    &layout->result) ||
      !loom_vector_to_scalar_mma_result_layout_is_supported(layout) ||
      !loom_vector_to_scalar_mma_reduction_role_layout_is_supported(
          layout, &layout->lhs) ||
      !loom_vector_to_scalar_mma_reduction_role_layout_is_supported(
          layout, &layout->rhs)) {
    return false;
  }
  *out_k = k;
  return true;
}

static iree_status_t loom_vector_to_scalar_mma_build_result_coordinate_terms(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t lane_id, uint16_t register_index,
    loom_vector_to_scalar_index_term_t* out_row,
    loom_vector_to_scalar_index_term_t* out_column) {
  const loom_matrix_fragment_layout_t* layout = state->matrix_fragment_layout;
  const loom_matrix_fragment_role_layout_t* result_layout = &layout->result;
  const loom_vector_to_scalar_index_term_t column_count =
      loom_vector_to_scalar_static_term(layout->tile_shape.result_column_count);
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, lane_id, column_count,
      out_column));

  loom_vector_to_scalar_index_term_t lane_group = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, lane_id, column_count,
      &lane_group));
  switch (result_layout->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN: {
      const loom_vector_to_scalar_index_term_t register_base =
          loom_vector_to_scalar_static_term((int64_t)register_index * 2);
      return loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, register_base,
          lane_group, out_row);
    }
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN: {
      const loom_vector_to_scalar_index_term_t register_count =
          loom_vector_to_scalar_static_term(result_layout->register_count);
      loom_vector_to_scalar_index_term_t group_base = {0};
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, lane_group,
          register_count, &group_base));
      const loom_vector_to_scalar_index_term_t register_offset =
          loom_vector_to_scalar_static_term(register_index);
      return loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, group_base,
          register_offset, out_row);
    }
    case LOOM_MATRIX_FRAGMENT_MAP_UNKNOWN:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported matrix result fragment layout");
  }
}

static iree_status_t loom_vector_to_scalar_mma_build_reduction_source_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, uint16_t reduction,
    loom_vector_to_scalar_index_term_t* out_source_lane,
    uint16_t* out_register_index, uint16_t* out_element_index) {
  const uint32_t reduction_group_span = (uint32_t)role_layout->register_count *
                                        role_layout->elements_per_register;
  if (reduction_group_span == 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "matrix reduction fragment layout has no payload "
                            "elements");
  }
  const uint32_t group_relative_reduction =
      (uint32_t)reduction % reduction_group_span;
  *out_register_index =
      (uint16_t)(group_relative_reduction / role_layout->elements_per_register);
  *out_element_index =
      (uint16_t)(group_relative_reduction % role_layout->elements_per_register);

  switch (role_layout->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
      *out_source_lane = row;
      return iree_ok_status();
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
      *out_source_lane = column;
      return iree_ok_status();
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION: {
      const int64_t lane_group =
          (int64_t)((uint32_t)reduction / reduction_group_span);
      const loom_vector_to_scalar_index_term_t lane_group_base =
          loom_vector_to_scalar_static_term(
              lane_group *
              state->matrix_fragment_layout->tile_shape.result_row_count);
      return loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, lane_group_base, row,
          out_source_lane);
    }
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION: {
      const int64_t lane_group =
          (int64_t)((uint32_t)reduction / reduction_group_span);
      const loom_vector_to_scalar_index_term_t lane_group_base =
          loom_vector_to_scalar_static_term(
              lane_group *
              state->matrix_fragment_layout->tile_shape.result_column_count);
      return loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, lane_group_base,
          column, out_source_lane);
    }
    case LOOM_MATRIX_FRAGMENT_MAP_UNKNOWN:
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported matrix reduction fragment layout");
  }
}

static iree_status_t loom_vector_to_scalar_mma_broadcast_physical_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_vector_to_scalar_index_term_t source_lane, uint16_t register_index,
    uint16_t element_index, loom_type_t result_type,
    loom_value_id_t* out_lane) {
  const loom_matrix_fragment_physical_element_t element = {
      .register_index = register_index,
      .element_index = element_index,
  };
  loom_value_id_t raw_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_physical_lane(
      state, fragment, role_layout, element, &raw_lane));

  loom_value_id_t cast_lane = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_t source_scalar_type =
      loom_type_element_type(fragment->type);
  switch (numeric) {
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT: {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_float(
          state, raw_lane, source_scalar_type, result_type, &cast_lane));
      break;
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER: {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_integer(
          state, raw_lane, source_scalar_type, result_type, &cast_lane));
      break;
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported vector.mma scalar numeric kind");
  }

  loom_value_id_t source_lane_i32 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_as_scalar(
      state, source_lane, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      &source_lane_i32));
  loom_op_t* broadcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_broadcast_build(
      &state->rewriter->builder, cast_lane, source_lane_i32, result_type,
      state->location, &broadcast_op));
  *out_lane = loom_kernel_subgroup_broadcast_result(broadcast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_build_distributed_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_index_term_t lane_id, uint16_t register_index,
    uint16_t element_index, int64_t k, loom_value_id_t* out_lane) {
  const loom_matrix_fragment_layout_t* layout = state->matrix_fragment_layout;
  loom_vector_to_scalar_index_term_t row = {0};
  loom_vector_to_scalar_index_term_t column = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_coordinate_terms(
      state, lane_id, register_index, &row, &column));

  loom_type_t accumulator_type =
      loom_type_scalar(loom_type_element_type(init->type));
  const loom_matrix_fragment_physical_element_t accumulator_element = {
      .register_index = register_index,
      .element_index = element_index,
  };
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_physical_lane(
      state, init, &layout->accumulator, accumulator_element, &accumulator));

  const loom_scalar_type_t accumulator_element_type =
      loom_type_element_type(accumulator_type);
  for (int64_t reduction = 0; reduction < k; ++reduction) {
    loom_vector_to_scalar_index_term_t lhs_source_lane = {0};
    uint16_t lhs_register_index = 0;
    uint16_t lhs_element_index = 0;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_reduction_source_lane(
        state, &layout->lhs, row, column, (uint16_t)reduction, &lhs_source_lane,
        &lhs_register_index, &lhs_element_index));
    loom_vector_to_scalar_index_term_t rhs_source_lane = {0};
    uint16_t rhs_register_index = 0;
    uint16_t rhs_element_index = 0;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_reduction_source_lane(
        state, &layout->rhs, row, column, (uint16_t)reduction, &rhs_source_lane,
        &rhs_register_index, &rhs_element_index));

    loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_broadcast_physical_lane(
        state, numeric, lhs, &layout->lhs, lhs_source_lane, lhs_register_index,
        lhs_element_index, accumulator_type, &lhs_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_broadcast_physical_lane(
        state, numeric, rhs, &layout->rhs, rhs_source_lane, rhs_register_index,
        rhs_element_index, accumulator_type, &rhs_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_accumulate(
        state, numeric, lhs_lane, accumulator_element_type, rhs_lane,
        accumulator_element_type, accumulator, accumulator_type, &accumulator));
    loom_vector_to_scalar_record_lane_materialized(state);
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_lower_distributed_static(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t k,
    loom_value_id_t* out_replacement, bool* out_handled) {
  const loom_matrix_fragment_role_layout_t* result_layout =
      &state->matrix_fragment_layout->result;
  uint16_t element_count = 0;
  if (!loom_vector_to_scalar_mma_physical_element_count(result_layout,
                                                        &element_count)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "matrix fragment layout has no physical result "
                            "elements after support was proven");
  }
  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->rewriter->arena, element_count,
                                sizeof(*elements), (void**)&elements));
  loom_vector_to_scalar_index_term_t lane_id = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_mma_build_lane_id_term(state, &lane_id));
  uint16_t ordinal = 0;
  for (uint16_t register_index = 0;
       register_index < result_layout->register_count; ++register_index) {
    for (uint16_t element_index = 0;
         element_index < result_layout->elements_per_register;
         ++element_index) {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_distributed_lane(
          state, numeric, lhs, rhs, init, lane_id, register_index,
          element_index, k, &elements[ordinal]));
      ++ordinal;
    }
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->rewriter->builder, elements, element_count, init->type,
      state->location, &from_elements_op));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_fragment(
      state, init, loom_vector_from_elements_result(from_elements_op),
      out_replacement));
  *out_handled = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_lower_static(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t m, int64_t n,
    int64_t k, loom_value_id_t* out_replacement, bool* out_handled) {
  uint64_t element_count = 0;
  if (!loom_vector_to_scalar_mma_logical_element_count(m, n, &element_count) ||
      element_count > UINT16_MAX) {
    return iree_ok_status();
  }
  uint64_t product_count = 0;
  if (!loom_vector_to_scalar_mma_product_count(element_count, (uint64_t)k,
                                               &product_count) ||
      product_count > UINT16_MAX) {
    return iree_ok_status();
  }

  if (element_count == 0) {
    loom_op_t* empty_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_empty_build(
        &state->rewriter->builder, init->type, state->location, &empty_op));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_fragment(
        state, init, loom_vector_empty_result(empty_op), out_replacement));
    *out_handled = true;
    return iree_ok_status();
  }

  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, (iree_host_size_t)element_count,
      sizeof(*elements), (void**)&elements));
  uint16_t ordinal = 0;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t column = 0; column < n; ++column) {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_lane(
          state, numeric, lhs, rhs, init, row, column, k, &elements[ordinal]));
      ++ordinal;
    }
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->rewriter->builder, elements, (iree_host_size_t)element_count,
      init->type, state->location, &from_elements_op));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_fragment(
      state, init, loom_vector_from_elements_result(from_elements_op),
      out_replacement));
  *out_handled = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_lower_dynamic(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_value_id_t* out_replacement, bool* out_handled) {
  loom_value_id_t payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_row_loop(state, numeric, lhs,
                                                          rhs, init, &payload));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_fragment(
      state, init, payload, out_replacement));
  *out_handled = true;
  return iree_ok_status();
}

static bool loom_vector_to_scalar_mma_query_operands(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_fragment_t* lhs,
    loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_mma_numeric_t* out_numeric) {
  if (!loom_vector_to_scalar_mma_query_semantic_operands(state, lhs, rhs, init,
                                                         out_numeric)) {
    return false;
  }
  return loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state,
                                                                   lhs) &&
         loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state,
                                                                   rhs) &&
         loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, init);
}

bool loom_vector_to_scalar_mma_supports_logical_result_lanes(
    loom_vector_to_scalar_state_t* state, loom_op_t* op) {
  if (!loom_vector_mma_isa(op) || state->rewriter->fact_table == NULL) {
    return false;
  }

  loom_vector_to_scalar_state_t lane_state = *state;
  lane_state.op = op;
  lane_state.vector_type = loom_module_value_type(state->rewriter->module,
                                                  loom_vector_mma_result(op));
  lane_state.result_scalar_type =
      loom_vector_to_scalar_lane_type(lane_state.vector_type);
  lane_state.location = op->location;

  loom_vector_to_scalar_mma_fragment_t lhs = {0};
  loom_vector_to_scalar_mma_fragment_t rhs = {0};
  loom_vector_to_scalar_mma_fragment_t init = {0};
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  return loom_vector_to_scalar_mma_query_operands(&lane_state, &lhs, &rhs,
                                                  &init, &numeric);
}

uint32_t loom_vector_to_scalar_mma_reference_rejection_bits(
    loom_vector_to_scalar_state_t* state) {
  if (!loom_vector_mma_isa(state->op)) {
    return LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  if (state->rewriter->fact_table == NULL) {
    return LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }

  loom_contract_rejection_bits_t rejection_bits = LOOM_CONTRACT_REJECTION_NONE;
  rejection_bits |= loom_vector_to_scalar_mma_fragment_query_rejection_bits(
      state, loom_vector_mma_lhs(state->op),
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS);
  rejection_bits |= loom_vector_to_scalar_mma_fragment_query_rejection_bits(
      state, loom_vector_mma_rhs(state->op),
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS);
  rejection_bits |= loom_vector_to_scalar_mma_fragment_query_rejection_bits(
      state, loom_vector_mma_init(state->op),
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT);
  if (rejection_bits != LOOM_CONTRACT_REJECTION_NONE) {
    return rejection_bits;
  }

  loom_vector_to_scalar_mma_fragment_t lhs = {0};
  loom_vector_to_scalar_mma_fragment_t rhs = {0};
  loom_vector_to_scalar_mma_fragment_t init = {0};
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_query_semantic_operands(state, &lhs, &rhs,
                                                         &init, &numeric)) {
    return loom_vector_to_scalar_mma_semantic_rejection_bits(state, &lhs, &rhs,
                                                             &init);
  }
  if (loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &lhs) &&
      loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &rhs) &&
      loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &init) &&
      loom_vector_to_scalar_mma_result_is_dense_payload(state, &init)) {
    return LOOM_CONTRACT_REJECTION_NONE;
  }
  int64_t k = 0;
  if (!loom_vector_to_scalar_mma_distributed_static_is_supported(
          state, &lhs, &rhs, &init, &k)) {
    return LOOM_CONTRACT_REJECTION_FRAGMENT;
  }
  return LOOM_CONTRACT_REJECTION_NONE;
}

static loom_contract_rejection_detail_t
loom_vector_to_scalar_mma_fragment_rejection_detail(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    loom_contract_rejection_detail_t detail) {
  const loom_contract_rejection_bits_t bits =
      loom_vector_to_scalar_mma_fragment_logical_lane_rejection_bits(state,
                                                                     fragment);
  return iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_FRAGMENT)
             ? detail
             : LOOM_CONTRACT_REJECTION_DETAIL_NONE;
}

uint32_t loom_vector_to_scalar_mma_reference_rejection_detail(
    loom_vector_to_scalar_state_t* state) {
  if (!loom_vector_mma_isa(state->op) || state->rewriter->fact_table == NULL) {
    return LOOM_CONTRACT_REJECTION_DETAIL_NONE;
  }

  loom_contract_rejection_bits_t rejection_bits = LOOM_CONTRACT_REJECTION_NONE;
  rejection_bits |= loom_vector_to_scalar_mma_fragment_query_rejection_bits(
      state, loom_vector_mma_lhs(state->op),
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS);
  rejection_bits |= loom_vector_to_scalar_mma_fragment_query_rejection_bits(
      state, loom_vector_mma_rhs(state->op),
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS);
  rejection_bits |= loom_vector_to_scalar_mma_fragment_query_rejection_bits(
      state, loom_vector_mma_init(state->op),
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT);
  if (rejection_bits != LOOM_CONTRACT_REJECTION_NONE) {
    return LOOM_CONTRACT_REJECTION_DETAIL_NONE;
  }

  loom_vector_to_scalar_mma_fragment_t lhs = {0};
  loom_vector_to_scalar_mma_fragment_t rhs = {0};
  loom_vector_to_scalar_mma_fragment_t init = {0};
  if (!loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_lhs(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS, &lhs) ||
      !loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_rhs(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS, &rhs) ||
      !loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_init(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
              LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT,
          &init)) {
    return LOOM_CONTRACT_REJECTION_DETAIL_NONE;
  }

  loom_contract_rejection_detail_t detail =
      loom_vector_to_scalar_mma_fragment_rejection_detail(
          state, &lhs,
          LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_LHS_FRAGMENT_OWNERSHIP);
  if (detail != LOOM_CONTRACT_REJECTION_DETAIL_NONE) {
    return detail;
  }
  detail = loom_vector_to_scalar_mma_fragment_rejection_detail(
      state, &rhs,
      LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_RHS_FRAGMENT_OWNERSHIP);
  if (detail != LOOM_CONTRACT_REJECTION_DETAIL_NONE) {
    return detail;
  }
  detail = loom_vector_to_scalar_mma_fragment_rejection_detail(
      state, &init,
      LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_INIT_FRAGMENT_OWNERSHIP);
  if (detail != LOOM_CONTRACT_REJECTION_DETAIL_NONE) {
    return detail;
  }

  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_query_semantic_operands(state, &lhs, &rhs,
                                                         &init, &numeric)) {
    return LOOM_CONTRACT_REJECTION_DETAIL_NONE;
  }
  if (loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &lhs) &&
      loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &rhs) &&
      loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &init) &&
      loom_vector_to_scalar_mma_result_is_dense_payload(state, &init)) {
    return LOOM_CONTRACT_REJECTION_DETAIL_NONE;
  }
  int64_t k = 0;
  if (!loom_vector_to_scalar_mma_distributed_static_is_supported(
          state, &lhs, &rhs, &init, &k)) {
    return LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_RESULT_FRAGMENT_PAYLOAD;
  }
  return LOOM_CONTRACT_REJECTION_DETAIL_NONE;
}

static iree_status_t loom_vector_to_scalar_mma_result_lane_terms(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_index_list_t indices, bool* out_supported,
    loom_vector_to_scalar_index_term_t* out_row,
    loom_vector_to_scalar_index_term_t* out_column) {
  *out_supported = false;
  if (indices.rank == 2) {
    *out_row = loom_vector_to_scalar_lane_term(state, indices, 0);
    *out_column = loom_vector_to_scalar_lane_term(state, indices, 1);
    *out_supported = true;
    return iree_ok_status();
  }
  switch (init->layout) {
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_MATRIX_ROW_MAJOR:
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_LOGICAL_MEMORY:
      return iree_ok_status();
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR: {
      if (indices.rank != 1) {
        return iree_ok_status();
      }
      loom_vector_to_scalar_index_term_t ordinal =
          loom_vector_to_scalar_lane_term(state, indices, 0);
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, ordinal, init->columns,
          out_row));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, ordinal, init->columns,
          out_column));
      *out_supported = true;
      return iree_ok_status();
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED:
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_vector_to_scalar_try_materialize_mma_lane(
    loom_vector_to_scalar_state_t* state, loom_op_t* op,
    loom_vector_to_scalar_index_list_t indices, bool* out_materialized,
    loom_value_id_t* out_lane) {
  *out_materialized = false;
  if (!loom_vector_mma_isa(op) || state->rewriter->fact_table == NULL) {
    return iree_ok_status();
  }

  loom_vector_to_scalar_state_t lane_state = *state;
  lane_state.op = op;
  lane_state.vector_type = loom_module_value_type(state->rewriter->module,
                                                  loom_vector_mma_result(op));
  lane_state.result_scalar_type =
      loom_vector_to_scalar_lane_type(lane_state.vector_type);
  lane_state.location = op->location;

  loom_vector_to_scalar_mma_fragment_t lhs = {0};
  loom_vector_to_scalar_mma_fragment_t rhs = {0};
  loom_vector_to_scalar_mma_fragment_t init = {0};
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_query_operands(&lane_state, &lhs, &rhs, &init,
                                                &numeric)) {
    return iree_ok_status();
  }

  loom_vector_to_scalar_index_term_t row = {0};
  loom_vector_to_scalar_index_term_t column = {0};
  bool lane_supported = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_result_lane_terms(
      &lane_state, &init, indices, &lane_supported, &row, &column));
  if (!lane_supported) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_dynamic_result_lane(
      &lane_state, numeric, &lhs, &rhs, &init, row, column, out_lane));
  *out_materialized = true;
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_lower_mma(
    loom_vector_to_scalar_state_t* state, bool* out_handled,
    loom_value_id_t* out_replacement) {
  *out_handled = false;
  *out_replacement = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_mma_isa(state->op) || state->rewriter->fact_table == NULL) {
    return iree_ok_status();
  }

  loom_vector_to_scalar_mma_fragment_t lhs = {0};
  loom_vector_to_scalar_mma_fragment_t rhs = {0};
  loom_vector_to_scalar_mma_fragment_t init = {0};
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_query_semantic_operands(state, &lhs, &rhs,
                                                         &init, &numeric)) {
    return iree_ok_status();
  }

  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  if (loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &lhs) &&
      loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &rhs) &&
      loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &init) &&
      loom_vector_to_scalar_mma_result_is_dense_payload(state, &init) &&
      loom_type_is_all_static(init.type) &&
      loom_vector_to_scalar_mma_shapes_are_static(&lhs, &rhs, &init, &m, &n,
                                                  &k)) {
    return loom_vector_to_scalar_mma_lower_static(state, numeric, &lhs, &rhs,
                                                  &init, m, n, k,
                                                  out_replacement, out_handled);
  }
  if (loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &lhs) &&
      loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &rhs) &&
      loom_vector_to_scalar_mma_fragment_supports_logical_lanes(state, &init) &&
      loom_vector_to_scalar_mma_result_is_dense_payload(state, &init)) {
    return loom_vector_to_scalar_mma_lower_dynamic(
        state, numeric, &lhs, &rhs, &init, out_replacement, out_handled);
  }

  if (loom_vector_to_scalar_mma_distributed_static_is_supported(
          state, &lhs, &rhs, &init, &k)) {
    return loom_vector_to_scalar_mma_lower_distributed_static(
        state, numeric, &lhs, &rhs, &init, k, out_replacement, out_handled);
  }
  return iree_ok_status();
}
