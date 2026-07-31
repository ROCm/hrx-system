// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/structural.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum {
  // V_PERM_B32 selector indices 4..7 read bytes from SRC0 and indices 0..3
  // read bytes from SRC1. These selectors merge either the low or high 16-bit
  // half from each source register into one packed 16-bit result register.
  LOOM_AMDGPU_PACKED_16BIT_PERMUTE_LOW_HALVES_SELECTOR = 0x01000504u,
  LOOM_AMDGPU_PACKED_16BIT_PERMUTE_HIGH_HALVES_SELECTOR = 0x03020706u,
};

static bool loom_amdgpu_vector_bitcast_storage_shape(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;

  const uint32_t register_count = loom_amdgpu_vector_32bit_register_count(type);
  if (register_count != 0) {
    *out_payload_bit_count = 32u * register_count;
    *out_register_count = register_count;
    return true;
  }

  return loom_amdgpu_type_packed_integer_storage(type, out_payload_bit_count,
                                                 out_register_count) ||
         loom_amdgpu_type_packed_8bit_float_storage(type, out_payload_bit_count,
                                                    out_register_count) ||
         loom_amdgpu_type_packed_16bit_float_storage(
             type, out_payload_bit_count, out_register_count);
}

static bool loom_amdgpu_vector_bitcast_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_bitcast_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_bitcast_plan_t){0};
  if (!loom_vector_bitcast_isa(source_op)) {
    return false;
  }

  out_plan->source = loom_vector_bitcast_input(source_op);
  out_plan->result = loom_vector_bitcast_result(source_op);
  const loom_type_t input_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);

  uint32_t input_payload_bit_count = 0;
  uint32_t input_register_count = 0;
  uint32_t result_payload_bit_count = 0;
  uint32_t result_register_count = 0;
  return loom_amdgpu_vector_bitcast_storage_shape(
             input_type, &input_payload_bit_count, &input_register_count) &&
         loom_amdgpu_vector_bitcast_storage_shape(
             result_type, &result_payload_bit_count, &result_register_count) &&
         input_payload_bit_count == result_payload_bit_count &&
         input_register_count == result_register_count;
}

static bool loom_amdgpu_static_rank1_slice_shape(loom_type_t source_type,
                                                 loom_type_t result_type,
                                                 int64_t* out_lane_count) {
  *out_lane_count = 0;
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      loom_type_rank(source_type) != 1 || loom_type_rank(result_type) != 1 ||
      !loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type) ||
      !loom_type_element_type_equals(source_type, result_type)) {
    return false;
  }
  const int64_t source_lane_count =
      loom_type_dim_static_size_at(source_type, 0);
  const int64_t result_lane_count =
      loom_type_dim_static_size_at(result_type, 0);
  if (source_lane_count < 1 || result_lane_count < 1) {
    return false;
  }
  *out_lane_count = result_lane_count;
  return true;
}

static bool loom_amdgpu_static_rank1_32bit_vector_shape(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = loom_amdgpu_vector_32bit_lane_count(type);
  return *out_register_count != 0 &&
         *out_register_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES;
}

static bool loom_amdgpu_static_rank1_register_storage_shape(
    loom_type_t type, loom_amdgpu_vector_storage_t* out_storage) {
  *out_storage = (loom_amdgpu_vector_storage_t){0};
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_amdgpu_type_vector_storage(type, out_storage) ||
      out_storage->register_count == 0 ||
      out_storage->register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return false;
  }
  const uint64_t payload_bit_count =
      (uint64_t)out_storage->element_count * out_storage->element_bit_count;
  const uint64_t storage_bit_count =
      (uint64_t)out_storage->register_count * 32u;
  return payload_bit_count == storage_bit_count;
}

static bool loom_amdgpu_packed_register_slice_storage_shape(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count, uint32_t* out_element_bit_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  *out_element_bit_count = 0;
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(type, &storage)) {
    return false;
  }
  if (!iree_any_bit_set(loom_amdgpu_vector_storage_kind_flags(storage.kind),
                        LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    return false;
  }
  *out_payload_bit_count = storage.element_count * storage.element_bit_count;
  *out_register_count = storage.register_count;
  *out_element_bit_count = storage.element_bit_count;
  return true;
}

static bool loom_amdgpu_static_rank1_even_odd_storage(
    loom_type_t type, loom_amdgpu_vector_storage_t* out_storage) {
  *out_storage = (loom_amdgpu_vector_storage_t){0};
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_amdgpu_type_vector_storage(type, out_storage) ||
      out_storage->register_count == 0 ||
      out_storage->register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return false;
  }
  return out_storage->kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT ||
         out_storage->kind ==
             LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT;
}

static bool loom_amdgpu_vector_even_odd_kind_from_storage(
    loom_amdgpu_vector_storage_kind_t storage_kind,
    loom_amdgpu_vector_even_odd_kind_t* out_kind) {
  *out_kind = LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_NONE;
  switch (storage_kind) {
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
      *out_kind = LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_32BIT_LANES;
      return true;
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT:
      *out_kind = LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_PACKED_16BIT_FLOAT;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_static_32bit_vector_register_shape(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = loom_amdgpu_vector_32bit_register_count(type);
  return *out_register_count != 0 &&
         *out_register_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES;
}

static bool loom_amdgpu_vector_concat_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_register_map_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_register_map_plan_t){0};
  if (!loom_vector_concat_isa(source_op) ||
      loom_vector_concat_axis(source_op) != 0) {
    return false;
  }

  loom_value_slice_t inputs = loom_vector_concat_inputs(source_op);
  if (inputs.count == 0 ||
      inputs.count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return false;
  }

  out_plan->result = loom_vector_concat_result(source_op);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  loom_amdgpu_vector_storage_t result_storage = {0};
  if (!loom_amdgpu_static_rank1_register_storage_shape(result_type,
                                                       &result_storage)) {
    return false;
  }
  out_plan->result_register_count = result_storage.register_count;

  uint32_t total_register_count = 0;
  for (uint16_t i = 0; i < inputs.count; ++i) {
    const loom_value_id_t input = inputs.values[i];
    const loom_type_t input_type = loom_module_value_type(module, input);
    loom_amdgpu_vector_storage_t input_storage = {0};
    if (!loom_type_element_type_equals(input_type, result_type) ||
        !loom_amdgpu_static_rank1_register_storage_shape(input_type,
                                                         &input_storage) ||
        input_storage.kind != result_storage.kind ||
        input_storage.register_count > out_plan->result_register_count ||
        total_register_count >
            out_plan->result_register_count - input_storage.register_count) {
      return false;
    }
    out_plan->sources[i] = input;
    out_plan->source_register_counts[i] = input_storage.register_count;
    for (uint32_t input_register_index = 0;
         input_register_index < input_storage.register_count;
         ++input_register_index) {
      out_plan->result_source_indices[total_register_count] = i;
      out_plan->source_register_indices[total_register_count] =
          input_register_index;
      ++total_register_count;
    }
  }
  if (total_register_count != out_plan->result_register_count) {
    return false;
  }
  out_plan->source_count = inputs.count;
  return true;
}

static bool loom_amdgpu_vector_deinterleave_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_deinterleave_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_deinterleave_plan_t){0};
  if (!loom_vector_deinterleave_isa(source_op) ||
      loom_vector_deinterleave_axis(source_op) != 0) {
    return false;
  }

  loom_value_slice_t results = loom_vector_deinterleave_results(source_op);
  if (results.count != 2) {
    return false;
  }

  out_plan->source = loom_vector_deinterleave_source(source_op);
  out_plan->results[0] = results.values[0];
  out_plan->results[1] = results.values[1];
  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t even_type =
      loom_module_value_type(module, out_plan->results[0]);
  const loom_type_t odd_type =
      loom_module_value_type(module, out_plan->results[1]);
  loom_amdgpu_vector_storage_t source_storage = {0};
  loom_amdgpu_vector_storage_t result_storage = {0};
  if (!loom_type_equal(even_type, odd_type) ||
      !loom_type_element_type_equals(source_type, even_type) ||
      !loom_amdgpu_static_rank1_even_odd_storage(source_type,
                                                 &source_storage) ||
      !loom_amdgpu_static_rank1_even_odd_storage(even_type, &result_storage) ||
      source_storage.kind != result_storage.kind ||
      source_storage.element_type != result_storage.element_type ||
      source_storage.element_count != 2u * result_storage.element_count ||
      !loom_amdgpu_vector_even_odd_kind_from_storage(source_storage.kind,
                                                     &out_plan->kind)) {
    return false;
  }
  out_plan->result_lane_count = result_storage.element_count;
  out_plan->source_register_count = source_storage.register_count;
  out_plan->result_register_count = result_storage.register_count;
  return true;
}

static bool loom_amdgpu_vector_interleave_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_interleave_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_interleave_plan_t){0};
  if (!loom_vector_interleave_isa(source_op) ||
      loom_vector_interleave_axis(source_op) != 0) {
    return false;
  }

  out_plan->sources[0] = loom_vector_interleave_even(source_op);
  out_plan->sources[1] = loom_vector_interleave_odd(source_op);
  out_plan->result = loom_vector_interleave_result(source_op);
  const loom_type_t even_type =
      loom_module_value_type(module, out_plan->sources[0]);
  const loom_type_t odd_type =
      loom_module_value_type(module, out_plan->sources[1]);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  loom_amdgpu_vector_storage_t source_storage = {0};
  loom_amdgpu_vector_storage_t result_storage = {0};
  if (!loom_type_equal(even_type, odd_type) ||
      !loom_type_element_type_equals(even_type, result_type) ||
      !loom_amdgpu_static_rank1_even_odd_storage(even_type, &source_storage) ||
      !loom_amdgpu_static_rank1_even_odd_storage(result_type,
                                                 &result_storage) ||
      source_storage.kind != result_storage.kind ||
      source_storage.element_type != result_storage.element_type ||
      result_storage.element_count != 2u * source_storage.element_count ||
      !loom_amdgpu_vector_even_odd_kind_from_storage(source_storage.kind,
                                                     &out_plan->kind)) {
    return false;
  }
  out_plan->source_register_count = source_storage.register_count;
  out_plan->result_register_count = result_storage.register_count;
  return true;
}

static bool loom_amdgpu_vector_shuffle_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_register_map_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_register_map_plan_t){0};
  if (!loom_vector_shuffle_isa(source_op)) {
    return false;
  }

  out_plan->sources[0] = loom_vector_shuffle_source(source_op);
  out_plan->result = loom_vector_shuffle_result(source_op);
  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->sources[0]);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  uint32_t register_count = 0;
  if (!loom_type_equal(source_type, result_type) ||
      !loom_amdgpu_static_rank1_32bit_vector_shape(source_type,
                                                   &register_count)) {
    return false;
  }

  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(source_op);
  if (source_lanes.kind != LOOM_ATTR_I64_ARRAY ||
      source_lanes.count != register_count) {
    return false;
  }
  out_plan->source_count = 1;
  out_plan->result_register_count = register_count;
  out_plan->source_register_counts[0] = register_count;
  for (uint16_t i = 0; i < source_lanes.count; ++i) {
    if (source_lanes.i64_array[i] < 0 ||
        source_lanes.i64_array[i] >= register_count) {
      return false;
    }
    out_plan->result_source_indices[i] = 0;
    out_plan->source_register_indices[i] = (uint32_t)source_lanes.i64_array[i];
  }
  return true;
}

static void loom_amdgpu_static_vector_indices_from_flat_register(
    loom_type_t type, uint32_t ordinal, int64_t* indices) {
  const uint8_t rank = loom_type_rank(type);
  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    const uint8_t axis = (uint8_t)(rank - reverse_axis - 1);
    const uint32_t dimension_size =
        (uint32_t)loom_type_dim_static_size_at(type, axis);
    indices[axis] =
        dimension_size == 0 ? 0 : (int64_t)(ordinal % dimension_size);
    if (dimension_size != 0) {
      ordinal /= dimension_size;
    }
  }
}

static bool loom_amdgpu_vector_transpose_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_register_map_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_register_map_plan_t){0};
  if (!loom_vector_transpose_isa(source_op)) {
    return false;
  }

  out_plan->sources[0] = loom_vector_transpose_source(source_op);
  out_plan->result = loom_vector_transpose_result(source_op);
  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->sources[0]);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_element_type_equals(source_type, result_type) ||
      loom_type_rank(source_type) != loom_type_rank(result_type) ||
      !loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type)) {
    return false;
  }

  uint32_t source_register_count = 0;
  uint32_t result_register_count = 0;
  if (!loom_amdgpu_static_32bit_vector_register_shape(source_type,
                                                      &source_register_count) ||
      !loom_amdgpu_static_32bit_vector_register_shape(result_type,
                                                      &result_register_count) ||
      source_register_count != result_register_count) {
    return false;
  }

  const uint8_t rank = loom_type_rank(source_type);
  loom_attribute_t permutation = loom_vector_transpose_permutation(source_op);
  if (permutation.kind != LOOM_ATTR_I64_ARRAY || permutation.count != rank) {
    return false;
  }
  uint32_t seen_axes = 0;
  for (uint8_t result_axis = 0; result_axis < rank; ++result_axis) {
    const int64_t source_axis = permutation.i64_array[result_axis];
    if (source_axis < 0 || source_axis >= rank) {
      return false;
    }
    const uint32_t axis_bit = 1u << (uint32_t)source_axis;
    if ((seen_axes & axis_bit) != 0) {
      return false;
    }
    seen_axes |= axis_bit;
  }

  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (uint32_t result_register = 0; result_register < result_register_count;
       ++result_register) {
    loom_amdgpu_static_vector_indices_from_flat_register(
        result_type, result_register, result_indices);
    for (uint8_t result_axis = 0; result_axis < rank; ++result_axis) {
      const int64_t source_axis = permutation.i64_array[result_axis];
      source_indices[source_axis] = result_indices[result_axis];
    }
    if (!loom_amdgpu_static_vector_flat_register_from_indices(
            source_type, source_indices,
            &out_plan->source_register_indices[result_register])) {
      return false;
    }
    out_plan->result_source_indices[result_register] = 0;
  }

  out_plan->source_count = 1;
  out_plan->source_register_counts[0] = source_register_count;
  out_plan->result_register_count = result_register_count;
  return true;
}

static bool loom_amdgpu_packed_register_bit_slice_window_is_supported(
    uint32_t source_bit_offset, uint32_t result_payload_bit_count,
    uint32_t result_register_count) {
  for (uint32_t register_index = 0; register_index < result_register_count;
       ++register_index) {
    const uint32_t result_bit_offset = register_index * 32u;
    const uint32_t result_remaining_bit_count =
        result_payload_bit_count - result_bit_offset;
    const uint32_t needed_bit_count = iree_min(result_remaining_bit_count, 32u);
    const uint32_t source_register_bit_offset =
        (source_bit_offset + result_bit_offset) & 31u;
    if (source_register_bit_offset + needed_bit_count > 32u) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_slice_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_slice_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_slice_plan_t){0};
  if (!loom_vector_slice_isa(source_op) ||
      loom_vector_slice_offsets(source_op).count != 0) {
    return false;
  }
  loom_attribute_t static_offsets = loom_vector_slice_static_offsets(source_op);
  if (static_offsets.kind != LOOM_ATTR_I64_ARRAY || static_offsets.count != 1 ||
      static_offsets.i64_array[0] < 0 ||
      static_offsets.i64_array[0] == INT64_MIN ||
      static_offsets.i64_array[0] > UINT32_MAX) {
    return false;
  }

  out_plan->source = loom_vector_slice_source(source_op);
  out_plan->result = loom_vector_slice_result(source_op);
  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);

  const uint32_t lane_offset = (uint32_t)static_offsets.i64_array[0];
  int64_t lane_count_i64 = 0;
  if (!loom_amdgpu_static_rank1_slice_shape(source_type, result_type,
                                            &lane_count_i64) ||
      lane_count_i64 > UINT32_MAX) {
    return false;
  }
  const uint32_t lane_count = (uint32_t)lane_count_i64;
  if ((uint64_t)lane_offset + lane_count >
      (uint64_t)loom_type_dim_static_size_at(source_type, 0)) {
    return false;
  }

  const uint32_t source_32bit_lane_count =
      loom_amdgpu_vector_32bit_lane_count(source_type);
  const uint32_t result_32bit_lane_count =
      loom_amdgpu_vector_32bit_lane_count(result_type);
  if (source_32bit_lane_count != 0 && result_32bit_lane_count != 0) {
    out_plan->source_register_count = source_32bit_lane_count;
    out_plan->result_register_count = result_32bit_lane_count;
    out_plan->kind = LOOM_AMDGPU_VECTOR_SLICE_KIND_32BIT_LANES;
  } else {
    uint32_t source_payload_bit_count = 0;
    uint32_t result_payload_bit_count = 0;
    uint32_t source_register_count = 0;
    uint32_t result_register_count = 0;
    uint32_t source_element_bit_count = 0;
    uint32_t result_element_bit_count = 0;
    if (!loom_amdgpu_packed_register_slice_storage_shape(
            source_type, &source_payload_bit_count, &source_register_count,
            &source_element_bit_count) ||
        !loom_amdgpu_packed_register_slice_storage_shape(
            result_type, &result_payload_bit_count, &result_register_count,
            &result_element_bit_count) ||
        source_element_bit_count != result_element_bit_count) {
      return false;
    }
    const uint32_t source_bit_offset = lane_offset * source_element_bit_count;
    if (!loom_amdgpu_packed_register_bit_slice_window_is_supported(
            source_bit_offset, result_payload_bit_count,
            result_register_count)) {
      return false;
    }
    out_plan->source_register_count = source_register_count;
    out_plan->result_register_count = result_register_count;
    out_plan->element_bit_count = source_element_bit_count;
    out_plan->kind = LOOM_AMDGPU_VECTOR_SLICE_KIND_PACKED_REGISTER_BITS;
  }

  out_plan->lane_offset = lane_offset;
  out_plan->lane_count = lane_count;
  return true;
}

iree_status_t loom_amdgpu_select_vector_bitcast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_bitcast_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_bitcast_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_bitcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bitcast_plan_t* plan) {
  loom_value_id_t low_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_input));

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, plan->result, &result_low_type));
  const loom_type_t input_low_type =
      loom_module_value_type(loom_low_lower_context_module(context), low_input);
  if (!loom_type_equal(input_low_type, result_low_type)) {
    IREE_ASSERT(loom_amdgpu_low_type_is_register_class(
        context, input_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR));
    IREE_ASSERT(loom_amdgpu_low_type_is_register_class(
        context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR));
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(input_low_type),
                   loom_low_register_type_unit_count(result_low_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_input, &low_input));
  }

  return loom_low_lower_bind_value(context, plan->result, low_input);
}

iree_status_t loom_amdgpu_select_vector_concat_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_register_map_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_concat_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  return iree_ok_status();
}

static bool loom_amdgpu_vector_register_map_is_source_alias(
    const loom_amdgpu_vector_register_map_plan_t* plan) {
  if (plan->source_count != 1 ||
      plan->source_register_counts[0] != plan->result_register_count) {
    return false;
  }
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    if (plan->result_source_indices[i] != 0 ||
        plan->source_register_indices[i] != i) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_amdgpu_lower_vector_register_map(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_register_map_plan_t* plan) {
  IREE_ASSERT_GT(plan->source_count, 0);
  IREE_ASSERT_LE(plan->source_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  IREE_ASSERT_GT(plan->result_register_count, 0);
  IREE_ASSERT_LE(plan->result_register_count,
                 LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  if (loom_amdgpu_vector_register_map_is_source_alias(plan)) {
    return loom_low_lower_bind_value_alias(context, plan->sources[0],
                                           plan->result);
  }

  loom_value_id_t low_sources[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->source_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &low_sources[i]));
  }

  loom_type_t register_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  loom_value_id_t low_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    const uint32_t source_index = plan->result_source_indices[i];
    IREE_ASSERT_LT(source_index, plan->source_count);
    IREE_ASSERT_LT(plan->source_register_indices[i],
                   plan->source_register_counts[source_index]);
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_sources[source_index],
        plan->source_register_counts[source_index],
        plan->source_register_indices[i], register_type, &low_registers[i]));
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             low_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_resolve_packed_16bit_permute_descriptor(
    loom_low_lower_context_t* context, loom_amdgpu_vector_even_odd_kind_t kind,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  if (kind != LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_PACKED_16BIT_FLOAT) {
    return iree_ok_status();
  }
  bool descriptor_present = false;
  return loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC2_LIT, out_descriptor,
      &descriptor_present);
}

iree_status_t loom_amdgpu_select_vector_deinterleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_deinterleave_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_deinterleave_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  if (*out_selected) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_packed_16bit_permute_descriptor(
        context, out_plan->kind, &out_plan->packed_permute_descriptor));
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_packed_16bit_permute_selector(uint32_t lane_index) {
  return (lane_index & 1u) == 0
             ? LOOM_AMDGPU_PACKED_16BIT_PERMUTE_LOW_HALVES_SELECTOR
             : LOOM_AMDGPU_PACKED_16BIT_PERMUTE_HIGH_HALVES_SELECTOR;
}

static iree_status_t loom_amdgpu_try_emit_packed_16bit_permute_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_half_source, uint32_t low_half_source_register_count,
    uint32_t low_lane_index, loom_value_id_t high_half_source,
    uint32_t high_half_source_register_count, uint32_t high_lane_index,
    loom_type_t lane_type, loom_value_id_t* out_register, bool* out_selected) {
  *out_register = LOOM_VALUE_ID_INVALID;
  *out_selected = false;
  if (descriptor->descriptor == NULL || high_lane_index == UINT32_MAX ||
      ((low_lane_index ^ high_lane_index) & 1u) != 0) {
    return iree_ok_status();
  }

  loom_value_id_t low_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_half_source, low_half_source_register_count,
      low_lane_index / 2u, lane_type, &low_register));
  loom_value_id_t high_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, high_half_source, high_half_source_register_count,
      high_lane_index / 2u, lane_type, &high_register));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary_immediate(
      context, source_op, descriptor, low_register, high_register,
      loom_amdgpu_packed_16bit_permute_selector(low_lane_index), lane_type,
      out_register));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_packed_16bit_lane_low_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t source_register_index = lane_index / 2u;
  const uint32_t source_register_bit_offset = (lane_index & 1u) * 16u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, source_register_count,
      source_register_index, lane_type, &source_register));
  if (source_register_bit_offset == 16) {
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
        source_register, lane_type, out_lane);
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      source_register, UINT32_C(0xFFFF), lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_packed_16bit_lane_high_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t source_register_index = lane_index / 2u;
  const uint32_t source_register_bit_offset = (lane_index & 1u) * 16u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, source_register_count,
      source_register_index, lane_type, &source_register));
  if (source_register_bit_offset == 16) {
    return loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        source_register, UINT32_C(0xFFFF0000), lane_type, out_lane);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      source_register, UINT32_C(0xFFFF), lane_type, out_lane));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      *out_lane, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_packed_16bit_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* permute_descriptor,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t low_lane_index, uint32_t high_lane_index, loom_type_t lane_type,
    loom_value_id_t* out_register) {
  *out_register = LOOM_VALUE_ID_INVALID;
  bool selected_permute = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_packed_16bit_permute_register(
      context, source_op, permute_descriptor, low_source, source_register_count,
      low_lane_index, low_source, source_register_count, high_lane_index,
      lane_type, out_register, &selected_permute));
  if (selected_permute) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_16bit_lane_low_bits(
      context, source_op, low_source, source_register_count, low_lane_index,
      lane_type, out_register));
  if (high_lane_index == UINT32_MAX) {
    return iree_ok_status();
  }

  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_16bit_lane_high_bits(
      context, source_op, low_source, source_register_count, high_lane_index,
      lane_type, &high_lane));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, *out_register,
      high_lane, lane_type, out_register);
}

iree_status_t loom_amdgpu_lower_vector_deinterleave(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_deinterleave_plan_t* plan) {
  if (plan->kind != LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_32BIT_LANES) {
    IREE_ASSERT_EQ(plan->kind,
                   LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_PACKED_16BIT_FLOAT);
    loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, plan->source, &low_source));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_source, &low_source));

    loom_type_t lane_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
    for (uint32_t result_index = 0;
         result_index < IREE_ARRAYSIZE(plan->results); ++result_index) {
      loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
      for (uint32_t register_index = 0;
           register_index < plan->result_register_count; ++register_index) {
        const uint32_t low_result_lane = register_index * 2u;
        const uint32_t high_result_lane = low_result_lane + 1u;
        const uint32_t low_source_lane = low_result_lane * 2u + result_index;
        const uint32_t high_source_lane =
            high_result_lane < plan->result_lane_count
                ? high_result_lane * 2u + result_index
                : UINT32_MAX;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_16bit_register(
            context, source_op, &plan->packed_permute_descriptor, low_source,
            plan->source_register_count, low_source_lane, high_source_lane,
            lane_type, &registers[register_index]));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_bind_low_register_range(
          context, source_op, plan->results[result_index], registers,
          plan->result_register_count));
    }
    return iree_ok_status();
  }

  for (uint32_t result_index = 0; result_index < IREE_ARRAYSIZE(plan->results);
       ++result_index) {
    loom_amdgpu_vector_register_map_plan_t map_plan = {
        .sources = {plan->source},
        .result = plan->results[result_index],
        .source_count = 1,
        .result_register_count = plan->result_register_count,
        .source_register_counts = {plan->source_register_count},
    };
    for (uint32_t lane_index = 0; lane_index < plan->result_register_count;
         ++lane_index) {
      map_plan.result_source_indices[lane_index] = 0;
      map_plan.source_register_indices[lane_index] =
          lane_index * 2 + result_index;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_lower_vector_register_map(context, source_op, &map_plan));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_interleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_interleave_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_interleave_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  if (*out_selected) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_packed_16bit_permute_descriptor(
        context, out_plan->kind, &out_plan->packed_permute_descriptor));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_interleave(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_interleave_plan_t* plan) {
  if (plan->kind == LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_32BIT_LANES) {
    loom_amdgpu_vector_register_map_plan_t map_plan = {
        .sources = {plan->sources[0], plan->sources[1]},
        .result = plan->result,
        .source_count = 2,
        .result_register_count = plan->result_register_count,
        .source_register_counts = {plan->source_register_count,
                                   plan->source_register_count},
    };
    for (uint32_t lane_index = 0; lane_index < plan->source_register_count;
         ++lane_index) {
      map_plan.result_source_indices[lane_index * 2] = 0;
      map_plan.source_register_indices[lane_index * 2] = lane_index;
      map_plan.result_source_indices[lane_index * 2 + 1] = 1;
      map_plan.source_register_indices[lane_index * 2 + 1] = lane_index;
    }
    return loom_amdgpu_lower_vector_register_map(context, source_op, &map_plan);
  }

  IREE_ASSERT_EQ(plan->kind,
                 LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_PACKED_16BIT_FLOAT);
  loom_value_id_t low_sources[2] = {LOOM_VALUE_ID_INVALID,
                                    LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(low_sources); ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &low_sources[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_sources[i], &low_sources[i]));
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    bool selected_permute = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_packed_16bit_permute_register(
        context, source_op, &plan->packed_permute_descriptor, low_sources[0],
        plan->source_register_count, register_index, low_sources[1],
        plan->source_register_count, register_index, lane_type,
        &registers[register_index], &selected_permute));
    if (selected_permute) {
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_16bit_register(
        context, source_op, &plan->packed_permute_descriptor, low_sources[0],
        plan->source_register_count, register_index, UINT32_MAX, lane_type,
        &registers[register_index]));
    loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_16bit_lane_high_bits(
        context, source_op, low_sources[1], plan->source_register_count,
        register_index, lane_type, &high_bits));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
        registers[register_index], high_bits, lane_type,
        &registers[register_index]));
  }
  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, registers, plan->result_register_count);
}

iree_status_t loom_amdgpu_select_vector_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_register_map_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_shuffle_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_transpose_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_register_map_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_transpose_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_slice_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_slice_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_slice_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_slice_32bit_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* plan) {
  loom_amdgpu_vector_register_map_plan_t map_plan = {
      .sources = {plan->source},
      .result = plan->result,
      .source_count = 1,
      .result_register_count = plan->result_register_count,
      .source_register_counts = {plan->source_register_count},
  };
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    map_plan.result_source_indices[i] = 0;
    map_plan.source_register_indices[i] = plan->lane_offset + i;
  }
  return loom_amdgpu_lower_vector_register_map(context, source_op, &map_plan);
}

static iree_status_t loom_amdgpu_lower_vector_slice_packed_register_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* select, loom_value_id_t low_source,
    loom_type_t lane_type, loom_value_id_t* low_registers) {
  const uint32_t source_bit_offset =
      select->lane_offset * select->element_bit_count;
  for (uint32_t i = 0; i < select->result_register_count; ++i) {
    const uint32_t register_source_bit_offset = source_bit_offset + i * 32u;
    const uint32_t source_register_index = register_source_bit_offset / 32u;
    const uint32_t source_register_bit_offset =
        register_source_bit_offset & 31u;
    loom_value_id_t source_register = low_source;
    if (select->source_register_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, source_register_index, lane_type,
          &source_register));
    }
    if (source_register_bit_offset == 0) {
      low_registers[i] = source_register;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        source_register_bit_offset, source_register, lane_type,
        &low_registers[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* plan) {
  if (plan->kind == LOOM_AMDGPU_VECTOR_SLICE_KIND_32BIT_LANES) {
    return loom_amdgpu_lower_vector_slice_32bit_lanes(context, source_op, plan);
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_slice_packed_register_bits(
      context, source_op, plan, low_source, lane_type, low_registers));

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             low_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_structural(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  switch (op->kind) {
    case LOOM_OP_VECTOR_BITCAST: {
      loom_amdgpu_vector_bitcast_plan_t plan;
      if (loom_amdgpu_vector_bitcast_plan_from_op(module, op, &plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("bitcast.storage"));
    }
    case LOOM_OP_VECTOR_CONCAT: {
      loom_amdgpu_vector_register_map_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_concat_plan_from_op(module, op, &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("concat.register_storage"));
    }
    case LOOM_OP_VECTOR_DEINTERLEAVE: {
      loom_amdgpu_vector_deinterleave_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_deinterleave_plan_from_op(module, op,
                                                       &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("deinterleave.rank1_even_odd_storage"));
    }
    case LOOM_OP_VECTOR_INTERLEAVE: {
      loom_amdgpu_vector_interleave_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_interleave_plan_from_op(module, op,
                                                     &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("interleave.rank1_even_odd_storage"));
    }
    case LOOM_OP_VECTOR_SHUFFLE: {
      loom_amdgpu_vector_register_map_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_shuffle_plan_from_op(module, op, &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("shuffle.rank1_32bit"));
    }
    case LOOM_OP_VECTOR_TRANSPOSE: {
      loom_amdgpu_vector_register_map_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_transpose_plan_from_op(module, op, &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("transpose.static_32bit"));
    }
    case LOOM_OP_VECTOR_SLICE: {
      loom_amdgpu_vector_slice_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_slice_plan_from_op(module, op, &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("slice.shape"));
    }
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}
