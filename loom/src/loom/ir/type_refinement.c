// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_refinement.h"

#include <string.h>

static bool loom_type_refinement_has_dimensions(loom_type_t type) {
  return loom_type_is_shaped(type) || loom_type_is_pool(type);
}

static bool loom_type_refinement_has_element_or_role(loom_type_t type) {
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_SCALAR:
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW:
    case LOOM_TYPE_GROUP:
    case LOOM_TYPE_ENCODING:
      return true;
    default:
      return false;
  }
}

static void loom_type_refinement_merge_result(
    loom_type_refinement_result_t next, loom_type_refinement_result_t* result) {
  if (*result == LOOM_TYPE_REFINEMENT_CONFLICT) return;
  if (next == LOOM_TYPE_REFINEMENT_CONFLICT ||
      next == LOOM_TYPE_REFINEMENT_NARROWED) {
    *result = next;
  }
}

static iree_status_t loom_type_refinement_prepare_outputs(
    loom_type_t current_type, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  if (!out_type || !out_result) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type refinement requires output pointers");
  }
  *out_type = current_type;
  *out_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  return iree_ok_status();
}

static loom_type_refinement_result_t loom_type_refine_dimension(
    uint64_t current_dim, uint64_t candidate_dim, uint64_t* out_dim) {
  *out_dim = current_dim;
  if (current_dim == candidate_dim) {
    return LOOM_TYPE_REFINEMENT_UNCHANGED;
  }

  bool current_dynamic = loom_dim_is_dynamic(current_dim);
  bool candidate_dynamic = loom_dim_is_dynamic(candidate_dim);
  if (current_dynamic && !candidate_dynamic) {
    *out_dim = candidate_dim;
    return LOOM_TYPE_REFINEMENT_NARROWED;
  }
  if (!current_dynamic && !candidate_dynamic) {
    return LOOM_TYPE_REFINEMENT_CONFLICT;
  }
  return current_dynamic ? LOOM_TYPE_REFINEMENT_CONFLICT
                         : LOOM_TYPE_REFINEMENT_UNCHANGED;
}

static iree_status_t loom_type_refinement_rebuild_dimensions(
    loom_type_t current_type, const uint64_t* dimensions,
    iree_arena_allocator_t* arena, loom_type_t* out_type) {
  uint8_t rank = loom_type_rank(current_type);
  bool all_static = true;
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_dim_is_dynamic(dimensions[i])) {
      all_static = false;
      break;
    }
  }

  uint8_t flags = 0;
  if (rank <= 2) flags |= LOOM_TYPE_FLAG_INLINE_DIMS;
  if (all_static) flags |= LOOM_TYPE_FLAG_ALL_STATIC;

  loom_type_t refined = current_type;
  refined.header =
      loom_type_make_header(loom_type_kind(current_type),
                            loom_type_element_type(current_type), rank, flags);
  refined.dims[0] = 0;
  refined.dims[1] = 0;

  if (rank <= 2) {
    for (uint8_t i = 0; i < rank; ++i) {
      refined.dims[i] = dimensions[i];
    }
  } else {
    if (!arena) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "rank-%u type refinement requires an arena for overflow dimensions",
          rank);
    }
    loom_overflow_dim_t* overflow_dimensions = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, rank, sizeof(loom_overflow_dim_t),
                                  (void**)&overflow_dimensions));
    memcpy(overflow_dimensions, dimensions,
           (iree_host_size_t)rank * sizeof(*overflow_dimensions));
    refined.dims[0] = (uint64_t)(uintptr_t)overflow_dimensions;
  }

  *out_type = refined;
  return iree_ok_status();
}

static iree_status_t loom_type_refinement_rebuild_element_or_role(
    loom_type_t current_type, uint8_t element_or_role, loom_type_t* out_type) {
  *out_type = current_type;
  out_type->header = loom_type_make_header(
      loom_type_kind(current_type), (loom_scalar_type_t)element_or_role,
      loom_type_rank(current_type), loom_type_flags(current_type));
  return iree_ok_status();
}

static loom_type_refinement_result_t loom_type_refine_element_or_role_value(
    loom_type_t current_type, loom_type_t candidate_type,
    uint8_t* out_element_or_role) {
  *out_element_or_role = (uint8_t)loom_type_element_type(current_type);
  if (loom_type_kind(current_type) != loom_type_kind(candidate_type)) {
    return LOOM_TYPE_REFINEMENT_CONFLICT;
  }

  if (!loom_type_refinement_has_element_or_role(current_type)) {
    return LOOM_TYPE_REFINEMENT_UNCHANGED;
  }

  uint8_t current_value = (uint8_t)loom_type_element_type(current_type);
  uint8_t candidate_value = (uint8_t)loom_type_element_type(candidate_type);
  if (current_value == candidate_value) {
    return LOOM_TYPE_REFINEMENT_UNCHANGED;
  }

  if (loom_type_is_encoding(current_type)) {
    loom_encoding_role_t current_role = loom_type_encoding_role(current_type);
    loom_encoding_role_t candidate_role =
        loom_type_encoding_role(candidate_type);
    if (current_role == LOOM_ENCODING_ROLE_UNKNOWN &&
        candidate_role != LOOM_ENCODING_ROLE_UNKNOWN) {
      *out_element_or_role = candidate_value;
      return LOOM_TYPE_REFINEMENT_NARROWED;
    }
    if (candidate_role == LOOM_ENCODING_ROLE_UNKNOWN) {
      return LOOM_TYPE_REFINEMENT_UNCHANGED;
    }
  }

  return LOOM_TYPE_REFINEMENT_CONFLICT;
}

iree_status_t loom_type_refine_element_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  (void)arena;
  IREE_RETURN_IF_ERROR(
      loom_type_refinement_prepare_outputs(current_type, out_type, out_result));

  uint8_t element_or_role = 0;
  loom_type_refinement_result_t result = loom_type_refine_element_or_role_value(
      current_type, candidate_type, &element_or_role);
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT ||
      result == LOOM_TYPE_REFINEMENT_UNCHANGED) {
    *out_result = result;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_type_refinement_rebuild_element_or_role(
      current_type, element_or_role, out_type));
  *out_result = result;
  return iree_ok_status();
}

iree_status_t loom_type_refine_shape_with_dims(
    loom_type_t current_type, const uint64_t* candidate_dims,
    uint8_t candidate_rank, iree_arena_allocator_t* arena,
    loom_type_t* out_type, loom_type_refinement_result_t* out_result) {
  IREE_RETURN_IF_ERROR(
      loom_type_refinement_prepare_outputs(current_type, out_type, out_result));

  if (!loom_type_refinement_has_dimensions(current_type)) {
    if (candidate_rank == 0) return iree_ok_status();
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
    return iree_ok_status();
  }
  if (candidate_rank != loom_type_rank(current_type)) {
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
    return iree_ok_status();
  }
  if (candidate_rank > 0 && !candidate_dims) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shape refinement requires candidate dimensions");
  }

  uint64_t refined_dimensions[LOOM_TYPE_MAX_RANK] = {0};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  for (uint8_t i = 0; i < candidate_rank; ++i) {
    loom_type_refinement_result_t dimension_result =
        loom_type_refine_dimension(loom_type_dim(current_type, i),
                                   candidate_dims[i], &refined_dimensions[i]);
    loom_type_refinement_merge_result(dimension_result, &result);
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      *out_result = result;
      return iree_ok_status();
    }
  }

  if (result == LOOM_TYPE_REFINEMENT_NARROWED) {
    IREE_RETURN_IF_ERROR(loom_type_refinement_rebuild_dimensions(
        current_type, refined_dimensions, arena, out_type));
  }
  *out_result = result;
  return iree_ok_status();
}

iree_status_t loom_type_refine_shape_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  IREE_RETURN_IF_ERROR(
      loom_type_refinement_prepare_outputs(current_type, out_type, out_result));
  if (loom_type_kind(current_type) != loom_type_kind(candidate_type)) {
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
    return iree_ok_status();
  }
  uint8_t rank = loom_type_rank(candidate_type);
  uint64_t candidate_dims[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t i = 0; i < rank; ++i) {
    candidate_dims[i] = loom_type_dim(candidate_type, i);
  }
  return loom_type_refine_shape_with_dims(current_type, candidate_dims, rank,
                                          arena, out_type, out_result);
}

iree_status_t loom_type_refine_encoding_with_attachment(
    loom_type_t current_type, uint16_t candidate_encoding_id,
    loom_encoding_flags_t candidate_encoding_flags,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  (void)arena;
  IREE_RETURN_IF_ERROR(
      loom_type_refinement_prepare_outputs(current_type, out_type, out_result));

  bool current_can_have_encoding = loom_type_can_have_encoding(current_type);
  bool current_has_encoding = loom_type_has_encoding(current_type);
  bool candidate_has_encoding =
      candidate_encoding_id != 0 || candidate_encoding_flags != 0;

  if (!current_can_have_encoding) {
    *out_result = candidate_has_encoding ? LOOM_TYPE_REFINEMENT_CONFLICT
                                         : LOOM_TYPE_REFINEMENT_UNCHANGED;
    return iree_ok_status();
  }

  if (current_type.encoding_id == candidate_encoding_id &&
      current_type.encoding_flags == candidate_encoding_flags) {
    return iree_ok_status();
  }

  if (!current_has_encoding || !candidate_has_encoding) {
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
    return iree_ok_status();
  }

  bool current_is_ssa =
      iree_all_bits_set(current_type.encoding_flags, LOOM_ENCODING_FLAG_SSA);
  bool candidate_is_ssa =
      iree_all_bits_set(candidate_encoding_flags, LOOM_ENCODING_FLAG_SSA);
  if (current_is_ssa && !candidate_is_ssa) {
    out_type->encoding_id = candidate_encoding_id;
    out_type->encoding_flags = candidate_encoding_flags;
    *out_result = LOOM_TYPE_REFINEMENT_NARROWED;
    return iree_ok_status();
  }

  if (!current_is_ssa && candidate_is_ssa) {
    return iree_ok_status();
  }

  *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
  return iree_ok_status();
}

iree_status_t loom_type_refine_encoding_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  IREE_RETURN_IF_ERROR(
      loom_type_refinement_prepare_outputs(current_type, out_type, out_result));
  if (loom_type_kind(current_type) != loom_type_kind(candidate_type)) {
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
    return iree_ok_status();
  }
  return loom_type_refine_encoding_with_attachment(
      current_type, candidate_type.encoding_id, candidate_type.encoding_flags,
      arena, out_type, out_result);
}

iree_status_t loom_type_refine_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  IREE_RETURN_IF_ERROR(
      loom_type_refinement_prepare_outputs(current_type, out_type, out_result));

  if (memcmp(&current_type, &candidate_type, sizeof(current_type)) == 0) {
    return iree_ok_status();
  }

  if (loom_type_kind(current_type) != loom_type_kind(candidate_type)) {
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
    return iree_ok_status();
  }

  loom_type_kind_t kind = loom_type_kind(current_type);
  if (kind == LOOM_TYPE_FUNCTION || kind == LOOM_TYPE_DIALECT) {
    *out_result = loom_type_equal(current_type, candidate_type)
                      ? LOOM_TYPE_REFINEMENT_UNCHANGED
                      : LOOM_TYPE_REFINEMENT_CONFLICT;
    return iree_ok_status();
  }

  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;

  loom_type_t element_refined_type = refined_type;
  loom_type_refinement_result_t element_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_element_with_candidate(
      refined_type, candidate_type, arena, &element_refined_type,
      &element_result));
  loom_type_refinement_merge_result(element_result, &result);
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    *out_result = result;
    return iree_ok_status();
  }
  refined_type = element_refined_type;

  if (loom_type_refinement_has_dimensions(refined_type) ||
      loom_type_refinement_has_dimensions(candidate_type)) {
    loom_type_t shape_refined_type = refined_type;
    loom_type_refinement_result_t shape_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    IREE_RETURN_IF_ERROR(loom_type_refine_shape_with_candidate(
        refined_type, candidate_type, arena, &shape_refined_type,
        &shape_result));
    loom_type_refinement_merge_result(shape_result, &result);
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      *out_result = result;
      return iree_ok_status();
    }
    refined_type = shape_refined_type;
  }

  if (loom_type_can_have_encoding(refined_type) ||
      loom_type_has_encoding(candidate_type)) {
    loom_type_t encoding_refined_type = refined_type;
    loom_type_refinement_result_t encoding_result =
        LOOM_TYPE_REFINEMENT_UNCHANGED;
    IREE_RETURN_IF_ERROR(loom_type_refine_encoding_with_candidate(
        refined_type, candidate_type, arena, &encoding_refined_type,
        &encoding_result));
    loom_type_refinement_merge_result(encoding_result, &result);
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      *out_result = result;
      return iree_ok_status();
    }
    refined_type = encoding_refined_type;
  }

  *out_type = refined_type;
  *out_result = result;
  return iree_ok_status();
}
