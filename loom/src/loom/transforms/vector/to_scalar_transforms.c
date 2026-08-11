// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_transforms.h"

#include <math.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/hadamard.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

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

static iree_status_t loom_vector_to_scalar_build_float_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_scalar_binary(
      state, kind, lhs, rhs, state->result_scalar_type, out_result);
}

static iree_status_t loom_vector_to_scalar_apply_hadamard_slice(
    loom_vector_to_scalar_state_t* state, iree_host_size_t slice_offset,
    iree_host_size_t slice_extent, loom_value_id_t* elements) {
  for (iree_host_size_t half_span = 1; half_span < slice_extent;
       half_span <<= 1) {
    const iree_host_size_t span = half_span << 1;
    for (iree_host_size_t base = 0; base < slice_extent; base += span) {
      for (iree_host_size_t lane = 0; lane < half_span; ++lane) {
        const iree_host_size_t lhs_index = slice_offset + base + lane;
        const iree_host_size_t rhs_index = lhs_index + half_span;
        const loom_value_id_t lhs = elements[lhs_index];
        const loom_value_id_t rhs = elements[rhs_index];
        IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_float_binary(
            state, LOOM_OP_SCALAR_ADDF, lhs, rhs, &elements[lhs_index]));
        IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_float_binary(
            state, LOOM_OP_SCALAR_SUBF, lhs, rhs, &elements[rhs_index]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_apply_hadamard_normalization(
    loom_vector_to_scalar_state_t* state, iree_host_size_t element_count,
    iree_host_size_t slice_extent, loom_value_id_t* elements) {
  loom_value_id_t scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_attr_constant(
      &state->rewriter->builder, state->result_scalar_type, state->location,
      loom_attr_f64(1.0 / sqrt((double)slice_extent)), &scale));
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_float_binary(
        state, LOOM_OP_SCALAR_MULF, elements[i], scale, &elements[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_lower_transform(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  const loom_module_t* module = state->rewriter->module;
  loom_encoding_hadamard_descriptor_t descriptor;
  if (!loom_encoding_hadamard_try_read_verified_descriptor(
          module, loom_vector_transform_transform(state->op), &descriptor)) {
    return loom_vector_to_scalar_emit_transform_error(state,
                                                      LOOM_ERR_ENCODING_017);
  }

  const loom_value_id_t source = loom_vector_transform_source(state->op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const uint8_t rank = loom_type_rank(source_type);
  if (rank == 0 ||
      loom_type_dim_is_dynamic_at(source_type, (uint8_t)(rank - 1))) {
    return loom_vector_to_scalar_emit_transform_error(state,
                                                      LOOM_ERR_SHAPE_006);
  }
  const iree_host_size_t slice_extent =
      (iree_host_size_t)loom_type_dim_static_size_at(source_type,
                                                     (uint8_t)(rank - 1));

  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      state, source_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();

  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->rewriter->arena, element_count,
                                sizeof(*elements), (void**)&elements));
  int64_t indices[LOOM_TYPE_MAX_RANK] = {0};
  for (uint16_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(source_type, ordinal, indices);
    const loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, source, index_list, &elements[ordinal]));
    loom_vector_to_scalar_record_lane_materialized(state);
  }

  for (iree_host_size_t offset = 0; offset < element_count;
       offset += slice_extent) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_apply_hadamard_slice(
        state, offset, slice_extent, elements));
  }
  if (descriptor.normalization ==
      LOOM_ENCODING_TRANSFORM_NORMALIZATION_ORTHONORMAL) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_apply_hadamard_normalization(
        state, element_count, slice_extent, elements));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->rewriter->builder, elements, element_count, source_type,
      state->location, &from_elements_op));
  *out_replacement = loom_vector_from_elements_result(from_elements_op);
  return iree_ok_status();
}
