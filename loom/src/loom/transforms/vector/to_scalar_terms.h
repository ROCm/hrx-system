// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Index-list and index-term algebra for vector lane addressing.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_TERMS_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_TERMS_H_

#include "loom/ops/op_defs.h"
#include "loom/transforms/vector/to_scalar_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vector_to_scalar_index_list_t {
  // Dynamic index SSA values, or NULL when all lane indices are static.
  const loom_value_id_t* dynamic_indices;
  // Static index values, or NULL when any lane index is dynamic.
  const int64_t* static_indices;
  // Number of logical vector axes addressed by this index list.
  uint8_t rank;
} loom_vector_to_scalar_index_list_t;

typedef struct loom_vector_to_scalar_index_term_t {
  // SSA index value when |is_dynamic| is true.
  loom_value_id_t dynamic_value;
  // Static index value when |is_dynamic| is false.
  int64_t static_value;
  // Whether this term is represented by |dynamic_value| instead of a constant.
  bool is_dynamic;
} loom_vector_to_scalar_index_term_t;

typedef enum loom_vector_to_scalar_index_binary_e {
  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD = 0,
  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_SUB,
  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL,
  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV,
  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM,
  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MIN,
  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MAX,
  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_COUNT,
} loom_vector_to_scalar_index_binary_t;

bool loom_vector_to_scalar_indices_are_dynamic(
    loom_vector_to_scalar_index_list_t indices);

// Copies static index attribute storage into the builder arena so constructed
// ops do not retain references to pass scratch memory.
iree_status_t loom_vector_to_scalar_copy_static_indices(
    loom_builder_t* builder, const int64_t* indices,
    iree_host_size_t index_count, int64_t** out_indices);

iree_status_t loom_vector_to_scalar_build_index_binary(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_binary_t binary, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_static_term(
    int64_t value);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_dynamic_term(
    loom_value_id_t value);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_lane_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, uint8_t axis);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_value_term(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value);

iree_status_t loom_vector_to_scalar_build_term_binary(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_binary_t binary,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs,
    loom_vector_to_scalar_index_term_t* out_term);

bool loom_vector_to_scalar_terms_equal_static(
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs, bool* out_equal);

iree_status_t loom_vector_to_scalar_build_index_term_cmp(
    loom_vector_to_scalar_state_t* state, uint32_t predicate,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs, loom_value_id_t* out_condition);

iree_status_t loom_vector_to_scalar_terms_to_index_list(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_index_term_t* terms, uint8_t rank,
    loom_vector_to_scalar_index_list_t* out_indices);

iree_status_t loom_vector_to_scalar_terms_from_explicit_indices(
    loom_vector_to_scalar_state_t* state, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices,
    loom_vector_to_scalar_index_term_t** out_terms, uint8_t* out_count);

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_dim_bound_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    uint8_t axis);

iree_status_t loom_vector_to_scalar_linear_ordinal_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    loom_vector_to_scalar_index_list_t indices,
    loom_vector_to_scalar_index_term_t* out_ordinal);

iree_status_t loom_vector_to_scalar_term_value(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t term, loom_value_id_t* out_value);

iree_status_t loom_vector_to_scalar_ordinal_for_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_type_t result_type,
    loom_value_id_t* out_ordinal);

int64_t loom_vector_to_scalar_linear_ordinal_static(loom_type_t vector_type,
                                                    const int64_t* indices);

void loom_vector_to_scalar_indices_from_ordinal(loom_type_t vector_type,
                                                int64_t ordinal,
                                                int64_t* indices);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_TERMS_H_
