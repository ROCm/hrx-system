// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_terms.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/transforms/vector/to_scalar_lanes.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Index op emission
//===----------------------------------------------------------------------===//

static const loom_op_kind_t kVectorToScalarIndexBinaryOps[] = {
    [LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD] = LOOM_OP_INDEX_ADD,
    [LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_SUB] = LOOM_OP_INDEX_SUB,
    [LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL] = LOOM_OP_INDEX_MUL,
    [LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV] = LOOM_OP_INDEX_DIV,
    [LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM] = LOOM_OP_INDEX_REM,
    [LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MIN] = LOOM_OP_INDEX_MIN,
    [LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MAX] = LOOM_OP_INDEX_MAX,
};

static_assert(IREE_ARRAYSIZE(kVectorToScalarIndexBinaryOps) ==
                  LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_COUNT,
              "index binary op table must cover all binary term modes");

iree_status_t loom_vector_to_scalar_build_index_binary(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_binary_t binary, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_generic_lane_op(
      state, kVectorToScalarIndexBinaryOps[binary], 0,
      (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), out_result);
}

//===----------------------------------------------------------------------===//
// Index terms
//===----------------------------------------------------------------------===//

iree_status_t loom_vector_to_scalar_copy_static_indices(
    loom_builder_t* builder, const int64_t* indices,
    iree_host_size_t index_count, int64_t** out_indices) {
  *out_indices = NULL;
  if (index_count == 0) return iree_ok_status();
  int64_t* copied_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, index_count, sizeof(int64_t), (void**)&copied_indices));
  memcpy(copied_indices, indices, index_count * sizeof(int64_t));
  *out_indices = copied_indices;
  return iree_ok_status();
}

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_static_term(
    int64_t value) {
  return (loom_vector_to_scalar_index_term_t){
      .static_value = value,
  };
}

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_dynamic_term(
    loom_value_id_t value) {
  return (loom_vector_to_scalar_index_term_t){
      .dynamic_value = value,
      .is_dynamic = true,
  };
}

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_lane_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, uint8_t axis) {
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    return loom_vector_to_scalar_value_term(state,
                                            indices.dynamic_indices[axis]);
  }
  return loom_vector_to_scalar_static_term(indices.static_indices[axis]);
}

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_value_term(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(state->rewriter, value);
  if (loom_value_facts_is_exact(facts) && !loom_value_facts_is_float(facts)) {
    return loom_vector_to_scalar_static_term(facts.range_lo);
  }
  return loom_vector_to_scalar_dynamic_term(value);
}

static bool loom_vector_to_scalar_term_is_static_value(
    loom_vector_to_scalar_index_term_t term, int64_t value) {
  return !term.is_dynamic && term.static_value == value;
}

static int64_t loom_vector_to_scalar_static_index_divide(int64_t lhs,
                                                         int64_t rhs) {
  // Zero-sized axes have no observable lanes; use the same all-zero coordinate
  // convention as ordinal-to-index expansion.
  if (rhs == 0) return 0;
  if (lhs == INT64_MIN && rhs == -1) return 0;
  return lhs / rhs;
}

static int64_t loom_vector_to_scalar_static_index_remainder(int64_t lhs,
                                                            int64_t rhs) {
  // Zero-sized axes have no observable lanes; use the same all-zero coordinate
  // convention as ordinal-to-index expansion.
  if (rhs == 0) return 0;
  if (lhs == INT64_MIN && rhs == -1) return 0;
  return lhs % rhs;
}

iree_status_t loom_vector_to_scalar_term_value(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t term, loom_value_id_t* out_value) {
  if (term.is_dynamic) {
    *out_value = term.dynamic_value;
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, term.static_value, out_value);
}

iree_status_t loom_vector_to_scalar_build_term_binary(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_binary_t binary,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs,
    loom_vector_to_scalar_index_term_t* out_term) {
  if (!lhs.is_dynamic && !rhs.is_dynamic) {
    int64_t result = 0;
    switch (binary) {
      case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD:
        if (!loom_checked_add_i64(lhs.static_value, rhs.static_value,
                                  &result)) {
          break;
        }
        *out_term = loom_vector_to_scalar_static_term(result);
        return iree_ok_status();
      case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_SUB:
        if (!loom_checked_sub_i64(lhs.static_value, rhs.static_value,
                                  &result)) {
          break;
        }
        *out_term = loom_vector_to_scalar_static_term(result);
        return iree_ok_status();
      case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL:
        if (!loom_checked_mul_i64(lhs.static_value, rhs.static_value,
                                  &result)) {
          break;
        }
        *out_term = loom_vector_to_scalar_static_term(result);
        return iree_ok_status();
      case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV:
        *out_term = loom_vector_to_scalar_static_term(
            loom_vector_to_scalar_static_index_divide(lhs.static_value,
                                                      rhs.static_value));
        return iree_ok_status();
      case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM:
        *out_term = loom_vector_to_scalar_static_term(
            loom_vector_to_scalar_static_index_remainder(lhs.static_value,
                                                         rhs.static_value));
        return iree_ok_status();
      case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MIN:
        *out_term = loom_vector_to_scalar_static_term(
            loom_min_i64(lhs.static_value, rhs.static_value));
        return iree_ok_status();
      case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MAX:
        *out_term = loom_vector_to_scalar_static_term(
            loom_max_i64(lhs.static_value, rhs.static_value));
        return iree_ok_status();
      default:
        IREE_ASSERT_UNREACHABLE("unsupported vector-to-scalar index term op");
        IREE_BUILTIN_UNREACHABLE();
    }
  }

  switch (binary) {
    case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD:
      if (loom_vector_to_scalar_term_is_static_value(lhs, 0)) {
        *out_term = rhs;
        return iree_ok_status();
      }
      if (loom_vector_to_scalar_term_is_static_value(rhs, 0)) {
        *out_term = lhs;
        return iree_ok_status();
      }
      break;
    case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_SUB:
      if (loom_vector_to_scalar_term_is_static_value(rhs, 0)) {
        *out_term = lhs;
        return iree_ok_status();
      }
      break;
    case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL:
      if (loom_vector_to_scalar_term_is_static_value(lhs, 1)) {
        *out_term = rhs;
        return iree_ok_status();
      }
      if (loom_vector_to_scalar_term_is_static_value(rhs, 1)) {
        *out_term = lhs;
        return iree_ok_status();
      }
      break;
    case LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV:
      if (loom_vector_to_scalar_term_is_static_value(rhs, 1)) {
        *out_term = lhs;
        return iree_ok_status();
      }
      break;
    default:
      break;
  }

  loom_value_id_t lhs_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, lhs, &lhs_value));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, rhs, &rhs_value));
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
      state, binary, lhs_value, rhs_value, &result));
  *out_term = loom_vector_to_scalar_value_term(state, result);
  return iree_ok_status();
}

bool loom_vector_to_scalar_terms_equal_static(
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs, bool* out_equal) {
  if (lhs.is_dynamic || rhs.is_dynamic) return false;
  *out_equal = lhs.static_value == rhs.static_value;
  return true;
}

static bool loom_vector_to_scalar_index_predicate_static_result(
    uint32_t predicate, int64_t lhs, int64_t rhs, bool* out_result) {
  switch (predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
      *out_result = lhs == rhs;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      *out_result = lhs < rhs;
      return true;
    default:
      return false;
  }
}

iree_status_t loom_vector_to_scalar_build_index_term_cmp(
    loom_vector_to_scalar_state_t* state, uint32_t predicate,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs, loom_value_id_t* out_condition) {
  if (!lhs.is_dynamic && !rhs.is_dynamic) {
    bool result = false;
    if (loom_vector_to_scalar_index_predicate_static_result(
            predicate, lhs.static_value, rhs.static_value, &result)) {
      return loom_vector_to_scalar_build_scalar_constant(
          &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_I1),
          state->location, result ? 1 : 0, out_condition);
    }
  }

  loom_value_id_t lhs_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, lhs, &lhs_value));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, rhs, &rhs_value));
  loom_op_t* cmp_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      &state->rewriter->builder, predicate, lhs_value, rhs_value,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &cmp_op));
  *out_condition = loom_index_cmp_result(cmp_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_terms_to_index_list(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_index_term_t* terms, uint8_t rank,
    loom_vector_to_scalar_index_list_t* out_indices) {
  bool has_dynamic = false;
  for (uint8_t i = 0; i < rank; ++i) {
    has_dynamic |= terms[i].is_dynamic;
  }
  if (!has_dynamic) {
    int64_t* static_indices = NULL;
    if (rank > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena,
                                                     rank, sizeof(int64_t),
                                                     (void**)&static_indices));
    }
    for (uint8_t i = 0; i < rank; ++i) {
      static_indices[i] = terms[i].static_value;
    }
    *out_indices = (loom_vector_to_scalar_index_list_t){
        .static_indices = static_indices,
        .rank = rank,
    };
    return iree_ok_status();
  }

  loom_value_id_t* dynamic_indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena, rank,
                                                   sizeof(loom_value_id_t),
                                                   (void**)&dynamic_indices));
  }
  for (uint8_t i = 0; i < rank; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_term_value(state, terms[i], &dynamic_indices[i]));
  }
  *out_indices = (loom_vector_to_scalar_index_list_t){
      .dynamic_indices = dynamic_indices,
      .rank = rank,
  };
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_terms_from_explicit_indices(
    loom_vector_to_scalar_state_t* state, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices,
    loom_vector_to_scalar_index_term_t** out_terms, uint8_t* out_count) {
  loom_vector_to_scalar_index_term_t* terms = NULL;
  if (static_indices.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, static_indices.count,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&terms));
  }
  uint16_t dynamic_ordinal = 0;
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index == INT64_MIN) {
      terms[i] = loom_vector_to_scalar_value_term(
          state, loom_value_slice_get(dynamic_indices, dynamic_ordinal++));
    } else {
      terms[i] = loom_vector_to_scalar_static_term(static_index);
    }
  }
  *out_terms = terms;
  *out_count = (uint8_t)static_indices.count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Linear lane coordinates
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_linear_ordinal_dynamic(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_ordinal) {
  if (indices.rank == 1) {
    *out_ordinal = indices.dynamic_indices[0];
    return iree_ok_status();
  }

  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &ordinal));
  for (uint8_t axis = 0; axis < indices.rank; ++axis) {
    if (axis > 0) {
      loom_value_id_t dim = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
          state, state->vector_type, axis, &dim));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, ordinal, dim,
          &ordinal));
    }
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, ordinal,
        indices.dynamic_indices[axis], &ordinal));
  }
  *out_ordinal = ordinal;
  return iree_ok_status();
}

int64_t loom_vector_to_scalar_linear_ordinal_static(loom_type_t vector_type,
                                                    const int64_t* indices) {
  int64_t ordinal = 0;
  uint8_t rank = loom_type_rank(vector_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (axis > 0) {
      ordinal *= (int64_t)loom_type_dim_static_size_at(vector_type, axis);
    }
    ordinal += indices[axis];
  }
  return ordinal;
}

void loom_vector_to_scalar_indices_from_ordinal(loom_type_t vector_type,
                                                int64_t ordinal,
                                                int64_t* indices) {
  uint8_t rank = loom_type_rank(vector_type);
  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    uint8_t axis = (uint8_t)(rank - reverse_axis - 1);
    int64_t dim = loom_type_dim_static_size_at(vector_type, axis);
    indices[axis] = dim == 0 ? 0 : (int64_t)(ordinal % dim);
    if (dim != 0) ordinal /= dim;
  }
}

static iree_status_t loom_vector_to_scalar_indices_from_ordinal_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    loom_vector_to_scalar_index_term_t ordinal,
    loom_vector_to_scalar_index_list_t* out_indices) {
  uint8_t rank = loom_type_rank(vector_type);
  if (!ordinal.is_dynamic && loom_type_is_all_static(vector_type)) {
    int64_t* static_indices = NULL;
    if (rank > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena,
                                                     rank, sizeof(int64_t),
                                                     (void**)&static_indices));
    }
    loom_vector_to_scalar_indices_from_ordinal(
        vector_type, ordinal.static_value, static_indices);
    *out_indices = (loom_vector_to_scalar_index_list_t){
        .static_indices = static_indices,
        .rank = rank,
    };
    return iree_ok_status();
  }

  if (rank == 1) {
    loom_vector_to_scalar_index_term_t* terms = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, 1, sizeof(loom_vector_to_scalar_index_term_t),
        (void**)&terms));
    terms[0] = ordinal;
    return loom_vector_to_scalar_terms_to_index_list(state, terms, 1,
                                                     out_indices);
  }

  loom_vector_to_scalar_index_term_t* terms = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, rank, sizeof(loom_vector_to_scalar_index_term_t),
      (void**)&terms));
  loom_vector_to_scalar_index_term_t remaining = ordinal;
  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    uint8_t axis = (uint8_t)(rank - reverse_axis - 1);
    if (axis == 0) {
      terms[axis] = remaining;
      break;
    }
    loom_vector_to_scalar_index_term_t dim =
        loom_vector_to_scalar_dim_bound_term(state, vector_type, axis);
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, remaining, dim,
        &terms[axis]));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, remaining, dim,
        &remaining));
  }
  return loom_vector_to_scalar_terms_to_index_list(state, terms, rank,
                                                   out_indices);
}

iree_status_t loom_vector_to_scalar_materialize_linear_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t vector_value,
    loom_type_t vector_type, loom_vector_to_scalar_index_term_t ordinal,
    loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_list_t indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_indices_from_ordinal_term(
      state, vector_type, ordinal, &indices));
  return loom_vector_to_scalar_materialize_lane(state, vector_value, indices,
                                                out_lane);
}

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_dim_bound_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    uint8_t axis) {
  if (loom_type_dim_is_dynamic_at(vector_type, axis)) {
    return loom_vector_to_scalar_value_term(
        state, loom_type_dim_value_id_at(vector_type, axis));
  }
  return loom_vector_to_scalar_static_term(
      (int64_t)loom_type_dim_static_size_at(vector_type, axis));
}

iree_status_t loom_vector_to_scalar_linear_ordinal_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    loom_vector_to_scalar_index_list_t indices,
    loom_vector_to_scalar_index_term_t* out_ordinal) {
  loom_vector_to_scalar_index_term_t ordinal =
      loom_vector_to_scalar_static_term(0);
  for (uint8_t axis = 0; axis < indices.rank; ++axis) {
    if (axis > 0) {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, ordinal,
          loom_vector_to_scalar_dim_bound_term(state, vector_type, axis),
          &ordinal));
    }
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, ordinal,
        loom_vector_to_scalar_lane_term(state, indices, axis), &ordinal));
  }
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_cast_index_to_scalar(
    loom_vector_to_scalar_state_t* state, loom_value_id_t index_value,
    loom_type_t result_type, loom_value_id_t* out_value) {
  if (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_INDEX) {
    *out_value = index_value;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(&state->rewriter->builder, index_value,
                            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                            result_type, state->location, &cast_op));
  *out_value = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_ordinal_for_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_type_t result_type,
    loom_value_id_t* out_ordinal) {
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    loom_value_id_t index_ordinal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_dynamic(
        state, indices, &index_ordinal));
    return loom_vector_to_scalar_cast_index_to_scalar(state, index_ordinal,
                                                      result_type, out_ordinal);
  }
  int64_t ordinal = loom_vector_to_scalar_linear_ordinal_static(
      state->vector_type, indices.static_indices);
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, result_type, state->location, ordinal,
      out_ordinal);
}
