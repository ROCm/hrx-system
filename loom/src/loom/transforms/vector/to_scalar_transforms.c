// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_transforms.h"

#include <math.h>
#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/numeric_transform.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Numeric transform diagnostics
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_emit_transform_error(
    loom_vector_to_scalar_state_t* state, const loom_error_def_t* error) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(state->rewriter->module, state->op)),
      loom_param_string(state->pass->info->name),
  };
  loom_diagnostic_emission_t emission = {
      .op = state->op,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission);
}

static iree_status_t
loom_vector_to_scalar_emit_undecodable_transform_descriptor(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_emit_transform_error(state,
                                                    LOOM_ERR_ENCODING_017);
}

static iree_status_t loom_vector_to_scalar_emit_dynamic_transform_extent(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_emit_transform_error(state, LOOM_ERR_SHAPE_006);
}

//===----------------------------------------------------------------------===//
// Numeric transform descriptor decoding
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_read_transform_descriptor(
    loom_vector_to_scalar_state_t* state,
    loom_encoding_numeric_transform_descriptor_t* out_descriptor) {
  loom_value_id_t transform_value = loom_vector_transform_transform(state->op);
  loom_encoding_numeric_transform_read_t read =
      loom_encoding_numeric_transform_read_descriptor(state->rewriter->module,
                                                      transform_value);
  switch (read.code) {
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK:
      *out_descriptor = read.descriptor;
      return iree_ok_status();
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_VALUE_OUT_OF_RANGE:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_LOCALLY_DEFINED:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_ENCODING_DEFINE:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NOT_NUMERIC_TRANSFORM:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_MISSING_FAMILY:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NON_STRING_FAMILY:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_UNKNOWN_FAMILY:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_NON_STRING_NORMALIZATION:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_UNKNOWN_NORMALIZATION:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_READ_MALFORMED_DYNAMIC_PARAM:
      return loom_vector_to_scalar_emit_undecodable_transform_descriptor(state);
  }
  return loom_vector_to_scalar_emit_undecodable_transform_descriptor(state);
}

//===----------------------------------------------------------------------===//
// Numeric transform descriptor helpers
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_transform_uses_signs(
    loom_encoding_numeric_transform_family_t family) {
  return family == LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN ||
         family == LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD;
}

static int64_t loom_vector_to_scalar_static_last_axis_extent(loom_type_t type) {
  uint8_t rank = loom_type_rank(type);
  if (rank == 0 || loom_type_dim_is_dynamic_at(type, (uint8_t)(rank - 1))) {
    return -1;
  }
  return (int64_t)loom_type_dim_static_size_at(type, (uint8_t)(rank - 1));
}

//===----------------------------------------------------------------------===//
// Numeric transform validation
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_query_exact_i64(loom_value_facts_t facts,
                                                  int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_vector_to_scalar_exact_leading_element_count(
    const loom_vector_to_scalar_state_t* state, loom_type_t type,
    int64_t* out_count) {
  *out_count = 1;
  uint8_t rank = loom_type_rank(type);
  if (rank <= 1) return true;
  for (uint8_t axis = 0; axis < rank - 1; ++axis) {
    int64_t extent = 0;
    if (loom_type_dim_is_dynamic_at(type, axis)) {
      loom_value_id_t dim_value = loom_type_dim_value_id_at(type, axis);
      loom_value_facts_t facts =
          loom_rewriter_value_facts(state->rewriter, dim_value);
      if (!loom_vector_to_scalar_query_exact_i64(facts, &extent)) {
        return false;
      }
    } else {
      extent = (int64_t)loom_type_dim_static_size_at(type, axis);
    }
    if (!loom_checked_mul_i64(*out_count, extent, out_count)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_to_scalar_emit_unproven_transform_permutation(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_emit_transform_error(state,
                                                    LOOM_ERR_ENCODING_018);
}

static iree_status_t loom_vector_to_scalar_emit_duplicate_transform_permutation(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_emit_transform_error(state,
                                                    LOOM_ERR_ENCODING_019);
}

static iree_status_t
loom_vector_to_scalar_emit_out_of_bounds_transform_permutation(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_emit_transform_error(state,
                                                    LOOM_ERR_ENCODING_020);
}

static iree_status_t loom_vector_to_scalar_validate_permutation_facts(
    loom_vector_to_scalar_state_t* state, const loom_value_facts_t* lane_facts,
    iree_host_size_t lane_count, int64_t source_lane_count) {
  if (source_lane_count <= 0) {
    return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
  }
  iree_host_size_t source_lane_count_size = (iree_host_size_t)source_lane_count;
  if ((int64_t)source_lane_count_size != source_lane_count ||
      lane_count % source_lane_count_size != 0) {
    return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
  }

  uint8_t* seen_lanes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->rewriter->arena, source_lane_count_size,
                                sizeof(uint8_t), (void**)&seen_lanes));
  for (iree_host_size_t lane_ordinal = 0; lane_ordinal < lane_count;
       ++lane_ordinal) {
    iree_host_size_t slice_ordinal = lane_ordinal % source_lane_count_size;
    if (slice_ordinal == 0) {
      memset(seen_lanes, 0, source_lane_count_size * sizeof(uint8_t));
    }

    int64_t source_lane = 0;
    if (!loom_vector_to_scalar_query_exact_i64(lane_facts[lane_ordinal],
                                               &source_lane)) {
      return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
    }
    if (source_lane < 0 || source_lane >= source_lane_count) {
      return loom_vector_to_scalar_emit_out_of_bounds_transform_permutation(
          state);
    }
    iree_host_size_t source_lane_ordinal = (iree_host_size_t)source_lane;
    if (seen_lanes[source_lane_ordinal]) {
      return loom_vector_to_scalar_emit_duplicate_transform_permutation(state);
    }
    seen_lanes[source_lane_ordinal] = 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_validate_uniform_permutation(
    loom_vector_to_scalar_state_t* state, loom_value_facts_t element_facts,
    int64_t source_lane_count) {
  int64_t source_lane = 0;
  if (!loom_vector_to_scalar_query_exact_i64(element_facts, &source_lane)) {
    return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
  }
  if (source_lane < 0 || source_lane >= source_lane_count) {
    return loom_vector_to_scalar_emit_out_of_bounds_transform_permutation(
        state);
  }
  if (source_lane_count <= 1) return iree_ok_status();
  return loom_vector_to_scalar_emit_duplicate_transform_permutation(state);
}

static iree_status_t loom_vector_to_scalar_validate_iota_permutation(
    loom_vector_to_scalar_state_t* state,
    loom_value_fact_vector_iota_t iota_facts, loom_type_t source_type,
    int64_t source_lane_count) {
  int64_t base = 0;
  int64_t step = 0;
  if (!loom_vector_to_scalar_query_exact_i64(iota_facts.base, &base) ||
      !loom_vector_to_scalar_query_exact_i64(iota_facts.step, &step)) {
    return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
  }

  if (source_lane_count == 1 && base == 0 && step == 0) {
    return iree_ok_status();
  }

  int64_t leading_count = 0;
  if (!loom_vector_to_scalar_exact_leading_element_count(state, source_type,
                                                         &leading_count)) {
    return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
  }
  if (leading_count == 0) return iree_ok_status();
  if (leading_count != 1) {
    return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
  }

  if (source_lane_count <= 0) {
    return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
  }
  iree_host_size_t source_lane_count_size = (iree_host_size_t)source_lane_count;
  if ((int64_t)source_lane_count_size != source_lane_count) {
    return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
  }
  loom_value_facts_t* lane_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_facts_scratch(
      state->rewriter->fact_table, source_lane_count_size, &lane_facts));
  int64_t source_lane = base;
  for (iree_host_size_t i = 0; i < source_lane_count_size; ++i) {
    lane_facts[i] = loom_value_facts_exact_i64(source_lane);
    if (i + 1 < source_lane_count_size &&
        !loom_checked_add_i64(source_lane, step, &source_lane)) {
      return loom_vector_to_scalar_emit_out_of_bounds_transform_permutation(
          state);
    }
  }
  return loom_vector_to_scalar_validate_permutation_facts(
      state, lane_facts, source_lane_count_size, source_lane_count);
}

static iree_status_t loom_vector_to_scalar_validate_from_elements_permutation(
    loom_vector_to_scalar_state_t* state, const loom_op_t* from_elements_op,
    int64_t source_lane_count) {
  loom_value_slice_t elements =
      loom_vector_from_elements_elements(from_elements_op);
  loom_value_facts_t* lane_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_facts_scratch(
      state->rewriter->fact_table, elements.count, &lane_facts));
  for (uint16_t i = 0; i < elements.count; ++i) {
    lane_facts[i] =
        loom_rewriter_value_facts(state->rewriter, elements.values[i]);
  }
  return loom_vector_to_scalar_validate_permutation_facts(
      state, lane_facts, elements.count, source_lane_count);
}

static iree_status_t loom_vector_to_scalar_validate_transform_permutation(
    loom_vector_to_scalar_state_t* state, loom_value_id_t permutation,
    loom_type_t source_type, int64_t source_lane_count) {
  const loom_op_t* def_op =
      loom_vector_to_scalar_value_def_op(state->rewriter->module, permutation);
  if (def_op && loom_vector_from_elements_isa(def_op)) {
    return loom_vector_to_scalar_validate_from_elements_permutation(
        state, def_op, source_lane_count);
  }

  loom_value_facts_t facts =
      loom_rewriter_value_facts(state->rewriter, permutation);
  loom_value_fact_vector_iota_t iota = {0};
  if (loom_value_facts_query_vector_iota(&state->rewriter->fact_table->context,
                                         facts, &iota)) {
    return loom_vector_to_scalar_validate_iota_permutation(
        state, iota, source_type, source_lane_count);
  }

  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(
          &state->rewriter->fact_table->context, facts, &uniform)) {
    return loom_vector_to_scalar_validate_uniform_permutation(
        state, uniform.element, source_lane_count);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(
          &state->rewriter->fact_table->context, facts, &lanes)) {
    return loom_vector_to_scalar_validate_permutation_facts(
        state, lanes.lanes, lanes.count, source_lane_count);
  }

  return loom_vector_to_scalar_emit_unproven_transform_permutation(state);
}

iree_status_t loom_vector_to_scalar_validate_transform(
    loom_vector_to_scalar_state_t* state) {
  loom_encoding_numeric_transform_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_read_transform_descriptor(state, &descriptor));
  if (descriptor.family !=
      LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD) {
    return iree_ok_status();
  }
  if (!loom_encoding_numeric_transform_has_permutation(&descriptor)) {
    return iree_ok_status();
  }

  loom_type_t source_type = loom_module_value_type(
      state->rewriter->module, loom_vector_transform_source(state->op));
  int64_t source_lane_count =
      loom_vector_to_scalar_static_last_axis_extent(source_type);
  if (source_lane_count <= 0) {
    return loom_vector_to_scalar_emit_dynamic_transform_extent(state);
  }
  return loom_vector_to_scalar_validate_transform_permutation(
      state, descriptor.permutation, source_type, source_lane_count);
}

//===----------------------------------------------------------------------===//
// Numeric transform lane programs
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_build_negf_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_generic_lane_op(
      state, LOOM_OP_SCALAR_NEGF, 0, &input, 1, NULL, 0,
      state->result_scalar_type, out_result);
}

static iree_status_t loom_vector_to_scalar_apply_sign_bit(
    loom_vector_to_scalar_state_t* state, loom_value_id_t sign,
    loom_value_id_t input, loom_value_id_t* out_result) {
  loom_value_id_t negated = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_negf_lane(state, input, &negated));
  return loom_vector_to_scalar_build_select_lane(state, sign, negated, input,
                                                 out_result);
}

static iree_status_t loom_vector_to_scalar_build_addf_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ADDF, lhs, rhs, state->result_scalar_type,
      out_result);
}

static iree_status_t loom_vector_to_scalar_build_mulf_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_MULF, lhs, rhs, state->result_scalar_type,
      out_result);
}

static iree_status_t loom_vector_to_scalar_apply_orthonormal_scale(
    loom_vector_to_scalar_state_t* state,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    int64_t input_extent, loom_value_id_t input, loom_value_id_t* out_result) {
  if (descriptor->normalization !=
      LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_ORTHONORMAL) {
    *out_result = input;
    return iree_ok_status();
  }

  double scale = 1.0 / sqrt((double)input_extent);
  loom_value_id_t scale_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_attr_constant(
      &state->rewriter->builder, state->result_scalar_type, state->location,
      loom_attr_f64(scale), &scale_value));
  return loom_vector_to_scalar_build_mulf_lane(state, input, scale_value,
                                               out_result);
}

static iree_status_t loom_vector_to_scalar_transform_source_indices_from_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    loom_vector_to_scalar_index_term_t last_index,
    loom_vector_to_scalar_index_list_t* out_indices) {
  uint8_t rank = result_indices.rank;
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, rank, sizeof(loom_vector_to_scalar_index_term_t),
      (void**)&source_terms));
  for (uint8_t axis = 0; axis < rank - 1; ++axis) {
    source_terms[axis] =
        loom_vector_to_scalar_lane_term(state, result_indices, axis);
  }
  source_terms[rank - 1] = last_index;
  return loom_vector_to_scalar_terms_to_index_list(state, source_terms, rank,
                                                   out_indices);
}

static iree_status_t loom_vector_to_scalar_cast_value_to_index(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_value_id_t* out_index) {
  loom_type_t value_type =
      loom_module_value_type(state->rewriter->module, value);
  if (loom_type_element_type(value_type) == LOOM_SCALAR_TYPE_INDEX) {
    *out_index = value;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      &state->rewriter->builder, value, value_type,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &cast_op));
  *out_index = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_permutation_lane_term(
    loom_vector_to_scalar_state_t* state,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    loom_vector_to_scalar_index_list_t indices,
    loom_vector_to_scalar_index_term_t* out_term) {
  loom_value_id_t permutation_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, descriptor->permutation, indices, &permutation_lane));
  loom_value_id_t permutation_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_value_to_index(
      state, permutation_lane, &permutation_index));
  *out_term = loom_vector_to_scalar_value_term(state, permutation_index);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_transform_accumulate(
    loom_vector_to_scalar_state_t* state, loom_value_id_t term,
    loom_value_id_t* accumulator) {
  if (*accumulator == LOOM_VALUE_ID_INVALID) {
    *accumulator = term;
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_addf_lane(state, *accumulator, term,
                                               accumulator);
}

static iree_status_t loom_vector_to_scalar_build_scalar_i64_constant(
    loom_vector_to_scalar_state_t* state, int64_t value,
    loom_value_id_t* out_value) {
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      state->location, value, out_value);
}

static iree_status_t loom_vector_to_scalar_build_scalar_i64_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_scalar_binary(
      state, kind, lhs, rhs, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      out_result);
}

static iree_status_t loom_vector_to_scalar_build_scalar_i64_binary_constant(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, int64_t rhs, loom_value_id_t* out_result) {
  loom_value_id_t rhs_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_scalar_i64_constant(state, rhs, &rhs_value));
  return loom_vector_to_scalar_build_scalar_i64_binary(state, kind, lhs,
                                                       rhs_value, out_result);
}

static iree_status_t loom_vector_to_scalar_build_i64_nonzero(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_value_id_t* out_condition) {
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_scalar_i64_constant(state, 0, &zero));
  loom_op_t* cmp_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      &state->rewriter->builder, LOOM_SCALAR_CMPI_PREDICATE_NE, value, zero,
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &cmp_op));
  *out_condition = loom_scalar_cmpi_result(cmp_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_hadamard_dynamic_phase(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t output_index, int64_t input_index,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  if (input_index == 0) return iree_ok_status();

  loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_value_id_t output_i64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_as_scalar(
      state, output_index, i64_type, &output_i64));

  loom_value_id_t parity = LOOM_VALUE_ID_INVALID;
  for (uint8_t bit = 0; bit < 63; ++bit) {
    int64_t bit_mask = (int64_t)1 << bit;
    if ((input_index & bit_mask) == 0) continue;

    loom_value_id_t shifted = output_i64;
    if (bit != 0) {
      IREE_RETURN_IF_ERROR(
          loom_vector_to_scalar_build_scalar_i64_binary_constant(
              state, LOOM_OP_SCALAR_SHRUI, output_i64, bit, &shifted));
    }
    loom_value_id_t low_bit = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
        state, LOOM_OP_SCALAR_ANDI, shifted, 1, &low_bit));
    if (parity == LOOM_VALUE_ID_INVALID) {
      parity = low_bit;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary(
        state, LOOM_OP_SCALAR_XORI, parity, low_bit, &parity));
  }
  return loom_vector_to_scalar_build_i64_nonzero(state, parity, out_condition);
}

static iree_status_t loom_vector_to_scalar_apply_hadamard_phase(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t output_index, int64_t input_index,
    loom_value_id_t input, loom_value_id_t* out_result) {
  if (!output_index.is_dynamic) {
    if (loom_count_ones_u64_width(
            (uint64_t)(output_index.static_value & input_index), 64) &
        1) {
      return loom_vector_to_scalar_build_negf_lane(state, input, out_result);
    }
    *out_result = input;
    return iree_ok_status();
  }

  loom_value_id_t negate_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_hadamard_dynamic_phase(
      state, output_index, input_index, &negate_condition));
  if (negate_condition == LOOM_VALUE_ID_INVALID) {
    *out_result = input;
    return iree_ok_status();
  }
  return loom_vector_to_scalar_apply_sign_bit(state, negate_condition, input,
                                              out_result);
}

// Seeded signs use the low bit of the SplitMix64 finalizer for
// seed + input lane. This keeps the reference path deterministic and
// expressible entirely in scalar integer IR.
static iree_status_t loom_vector_to_scalar_build_seed_sign_bit(
    loom_vector_to_scalar_state_t* state,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    int64_t input_index, loom_value_id_t* out_sign) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* seed_cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      &state->rewriter->builder, descriptor->seed, index_type, i64_type,
      state->location, &seed_cast_op));
  loom_value_id_t mixed = loom_index_cast_result(seed_cast_op);

  if (input_index != 0) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
        state, LOOM_OP_SCALAR_ADDI, mixed, input_index, &mixed));
  }
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_ADDI, mixed, -7046029254386353131LL, &mixed));

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_SHRUI, mixed, 30, &shifted));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary(
      state, LOOM_OP_SCALAR_XORI, mixed, shifted, &mixed));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_MULI, mixed, -4658895280553007687LL, &mixed));

  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_SHRUI, mixed, 27, &shifted));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary(
      state, LOOM_OP_SCALAR_XORI, mixed, shifted, &mixed));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_MULI, mixed, -7723592293110705685LL, &mixed));

  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_SHRUI, mixed, 31, &shifted));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary(
      state, LOOM_OP_SCALAR_XORI, mixed, shifted, &mixed));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_ANDI, mixed, 1, &mixed));

  return loom_vector_to_scalar_build_i64_nonzero(state, mixed, out_sign);
}

static iree_status_t loom_vector_to_scalar_transform_hadamard_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t source_type = loom_module_value_type(
      state->rewriter->module, loom_vector_transform_source(state->op));
  int64_t input_extent =
      loom_vector_to_scalar_static_last_axis_extent(source_type);
  if (input_extent <= 0) {
    return loom_vector_to_scalar_emit_dynamic_transform_extent(state);
  }

  uint8_t last_axis = (uint8_t)(indices.rank - 1);
  loom_vector_to_scalar_index_term_t output_index =
      loom_vector_to_scalar_lane_term(state, indices, last_axis);
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  for (int64_t input_index = 0; input_index < input_extent; ++input_index) {
    loom_vector_to_scalar_index_list_t transform_indices = {0};
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_transform_source_indices_from_term(
            state, indices, loom_vector_to_scalar_static_term(input_index),
            &transform_indices));

    loom_vector_to_scalar_index_list_t source_indices = transform_indices;
    if (descriptor->family ==
        LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD) {
      loom_vector_to_scalar_index_term_t source_last_index = {0};
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_permutation_lane_term(
          state, descriptor, transform_indices, &source_last_index));
      IREE_RETURN_IF_ERROR(
          loom_vector_to_scalar_transform_source_indices_from_term(
              state, indices, source_last_index, &source_indices));
    }

    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_transform_source(state->op), source_indices,
        &source_lane));

    if (loom_vector_to_scalar_transform_uses_signs(descriptor->family)) {
      loom_value_id_t sign_lane = LOOM_VALUE_ID_INVALID;
      if (loom_encoding_numeric_transform_has_signs(descriptor)) {
        IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
            state, descriptor->signs, source_indices, &sign_lane));
      } else {
        IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_seed_sign_bit(
            state, descriptor, input_index, &sign_lane));
      }
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_apply_sign_bit(
          state, sign_lane, source_lane, &source_lane));
    }

    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_apply_hadamard_phase(
        state, output_index, input_index, source_lane, &source_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_transform_accumulate(
        state, source_lane, &accumulator));
  }

  return loom_vector_to_scalar_apply_orthonormal_scale(
      state, descriptor, input_extent, accumulator, out_lane);
}

static iree_status_t loom_vector_to_scalar_transform_matrix_indices_from_terms(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t output_index,
    loom_vector_to_scalar_index_term_t input_index,
    loom_vector_to_scalar_index_list_t* out_indices) {
  loom_vector_to_scalar_index_term_t terms[2] = {
      output_index,
      input_index,
  };
  return loom_vector_to_scalar_terms_to_index_list(
      state, terms, IREE_ARRAYSIZE(terms), out_indices);
}

static iree_status_t loom_vector_to_scalar_transform_jl_dense_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t source_type = loom_module_value_type(
      state->rewriter->module, loom_vector_transform_source(state->op));
  int64_t input_extent =
      loom_vector_to_scalar_static_last_axis_extent(source_type);
  if (input_extent <= 0) {
    return loom_vector_to_scalar_emit_dynamic_transform_extent(state);
  }

  uint8_t last_axis = (uint8_t)(indices.rank - 1);
  loom_vector_to_scalar_index_term_t output_index =
      loom_vector_to_scalar_lane_term(state, indices, last_axis);
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  for (int64_t input_index = 0; input_index < input_extent; ++input_index) {
    loom_vector_to_scalar_index_list_t source_indices = {0};
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_transform_source_indices_from_term(
            state, indices, loom_vector_to_scalar_static_term(input_index),
            &source_indices));
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_transform_source(state->op), source_indices,
        &source_lane));

    loom_vector_to_scalar_index_list_t matrix_indices = {0};
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_transform_matrix_indices_from_terms(
            state, output_index, loom_vector_to_scalar_static_term(input_index),
            &matrix_indices));
    loom_value_id_t matrix_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, descriptor->matrix, matrix_indices, &matrix_lane));

    loom_value_id_t product = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_mulf_lane(
        state, matrix_lane, source_lane, &product));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_transform_accumulate(
        state, product, &accumulator));
  }

  *out_lane = accumulator;
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_transform_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_encoding_numeric_transform_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_read_transform_descriptor(state, &descriptor));
  switch (descriptor.family) {
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD:
      return loom_vector_to_scalar_transform_hadamard_lane(state, &descriptor,
                                                           indices, out_lane);
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE:
      return loom_vector_to_scalar_transform_jl_dense_lane(state, &descriptor,
                                                           indices, out_lane);
    default:
      return loom_vector_to_scalar_emit_undecodable_transform_descriptor(state);
  }
}
