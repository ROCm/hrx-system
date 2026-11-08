// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/structural.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/byte_permute.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_vector_bitcast_storage_shape(
    loom_type_t type, uint64_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;

  loom_amdgpu_vector_storage_t storage = {0};
  if (loom_amdgpu_type_vector_register_storage(type, &storage)) {
    *out_payload_bit_count =
        (uint64_t)storage.element_count * storage.element_bit_count;
    *out_register_count = storage.register_count;
    return true;
  }

  return false;
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

  uint64_t input_payload_bit_count = 0;
  uint32_t input_register_count = 0;
  uint64_t result_payload_bit_count = 0;
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

static bool loom_amdgpu_static_rank1_register_tuple_storage_shape(
    loom_type_t type, loom_amdgpu_vector_storage_t* out_storage) {
  *out_storage = (loom_amdgpu_vector_storage_t){0};
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_amdgpu_type_vector_register_storage(type, out_storage) &&
         out_storage->register_count != 0;
}

static bool loom_amdgpu_vector_storage_fills_registers(
    const loom_amdgpu_vector_storage_t* storage) {
  const uint64_t payload_bit_count =
      (uint64_t)storage->element_count * storage->element_bit_count;
  const uint64_t storage_bit_count = (uint64_t)storage->register_count * 32u;
  return payload_bit_count == storage_bit_count;
}

static bool loom_amdgpu_packed_register_slice_storage_shape(
    loom_type_t type, uint32_t* out_register_count,
    uint32_t* out_element_bit_count) {
  *out_register_count = 0;
  *out_element_bit_count = 0;
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_register_storage(type, &storage)) {
    return false;
  }
  if (!iree_any_bit_set(loom_amdgpu_vector_storage_kind_flags(storage.kind),
                        LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    return false;
  }
  *out_register_count = storage.register_count;
  *out_element_bit_count = storage.element_bit_count;
  return true;
}

static bool loom_amdgpu_vector_even_odd_layout(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t half_type,
    loom_type_t combined_type, int64_t axis, loom_vector_interleave_kind_t kind,
    loom_amdgpu_vector_even_odd_layout_t* out_layout) {
  *out_layout = (loom_amdgpu_vector_even_odd_layout_t){0};
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_register_storage(half_type, &storage)) {
    return false;
  }
  const bool packed =
      iree_any_bit_set(loom_amdgpu_vector_storage_kind_flags(storage.kind),
                       LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD);
  const uint16_t physical_element_bit_count =
      packed ? storage.element_bit_count : storage.element_register_count * 32u;
  if (!loom_vector_interleave_packet_plan_initialize(
          kind, half_type, combined_type, axis, physical_element_bit_count,
          /*packet_byte_count=*/4, /*maximum_packet_count=*/0,
          &out_layout->route)) {
    return false;
  }
  // Complete words retain their source register class and require no native
  // instruction. Subword rows use the shared byte-permutation emitter.
  if (out_layout->route.chunk_byte_count % 4u != 0) {
    loom_amdgpu_select_byte_permute_plan(descriptor_set,
                                         &out_layout->byte_permute);
    if (out_layout->byte_permute.kind == LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_static_32bit_vector_register_shape(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = loom_amdgpu_vector_32bit_register_count(type);
  return *out_register_count != 0 &&
         *out_register_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES;
}

bool loom_amdgpu_vector_concat_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_concat_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_concat_plan_t){0};
  if (!loom_vector_concat_isa(source_op) ||
      loom_vector_concat_axis(source_op) != 0) {
    return false;
  }

  loom_value_slice_t inputs = loom_vector_concat_inputs(source_op);
  if (inputs.count == 0) {
    return false;
  }

  out_plan->result = loom_vector_concat_result(source_op);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  loom_amdgpu_vector_storage_t result_storage = {0};
  if (!loom_amdgpu_static_rank1_register_tuple_storage_shape(result_type,
                                                             &result_storage)) {
    return false;
  }

  uint32_t total_register_count = 0;
  for (uint16_t i = 0; i < inputs.count; ++i) {
    const loom_value_id_t input = inputs.values[i];
    const loom_type_t input_type = loom_module_value_type(module, input);
    loom_amdgpu_vector_storage_t input_storage = {0};
    if (!loom_type_element_type_equals(input_type, result_type) ||
        !loom_amdgpu_static_rank1_register_tuple_storage_shape(
            input_type, &input_storage) ||
        input_storage.kind != result_storage.kind) {
      return false;
    }
    const bool input_is_packed = iree_any_bit_set(
        loom_amdgpu_vector_storage_kind_flags(input_storage.kind),
        LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD);
    // Padding in an interior packed input would leave a gap before the next
    // logical lane. Terminal padding already occupies the result's tail and
    // preserves the concatenated lane order without a subregister shuffle.
    if ((i + 1u < inputs.count && input_is_packed &&
         !loom_amdgpu_vector_storage_fills_registers(&input_storage)) ||
        input_storage.register_count > result_storage.register_count ||
        total_register_count >
            result_storage.register_count - input_storage.register_count) {
      return false;
    }
    total_register_count += input_storage.register_count;
  }
  if (total_register_count != result_storage.register_count) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vector_deinterleave_plan_from_op(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_vector_deinterleave_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_deinterleave_plan_t){0};
  if (!loom_vector_deinterleave_isa(source_op)) {
    return false;
  }

  out_plan->source = loom_vector_deinterleave_source(source_op);
  out_plan->results[0] = loom_vector_deinterleave_even(source_op);
  out_plan->results[1] = loom_vector_deinterleave_odd(source_op);
  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t even_type =
      loom_module_value_type(module, out_plan->results[0]);
  const loom_type_t odd_type =
      loom_module_value_type(module, out_plan->results[1]);
  if (!loom_type_equal(even_type, odd_type) ||
      !loom_type_element_type_equals(source_type, even_type) ||
      !loom_amdgpu_vector_even_odd_layout(
          descriptor_set, even_type, source_type,
          loom_vector_deinterleave_axis(source_op),
          LOOM_VECTOR_INTERLEAVE_KIND_UNZIP, &out_plan->layout)) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vector_interleave_plan_from_op(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_vector_interleave_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_interleave_plan_t){0};
  if (!loom_vector_interleave_isa(source_op)) {
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
  if (!loom_type_equal(even_type, odd_type) ||
      !loom_type_element_type_equals(even_type, result_type) ||
      !loom_amdgpu_vector_even_odd_layout(
          descriptor_set, even_type, result_type,
          loom_vector_interleave_axis(source_op),
          LOOM_VECTOR_INTERLEAVE_KIND_ZIP, &out_plan->layout)) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vector_shuffle_select_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_vector_storage_t* storage,
    loom_amdgpu_vector_shuffle_plan_t* out_plan) {
  const loom_amdgpu_vector_storage_kind_flags_t storage_flags =
      loom_amdgpu_vector_storage_kind_flags(storage->kind);
  if (iree_any_bit_set(storage_flags,
                       LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    if (storage->element_bit_count != 8 && storage->element_bit_count != 16) {
      return false;
    }
    loom_amdgpu_select_byte_permute_plan(descriptor_set,
                                         &out_plan->byte_permute);
    if (out_plan->byte_permute.kind == LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE) {
      return false;
    }
    out_plan->kind = LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_PACKED_BYTES;
    return true;
  }

  switch (storage->kind) {
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
    case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
      out_plan->kind = LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_REGISTER_UNITS;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_vector_shuffle_plan_from_op(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_vector_shuffle_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_shuffle_plan_t){0};
  if (!loom_vector_shuffle_isa(source_op)) {
    return false;
  }

  out_plan->source = loom_vector_shuffle_source(source_op);
  out_plan->result = loom_vector_shuffle_result(source_op);
  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_type_equal(source_type, result_type) ||
      !loom_amdgpu_static_rank1_register_tuple_storage_shape(source_type,
                                                             &storage) ||
      !loom_amdgpu_vector_shuffle_select_kind(descriptor_set, &storage,
                                              out_plan)) {
    return false;
  }
  // Verification owns the map length and lane bounds. The source operation
  // remains alive throughout selection and emission, so retain its map.
  out_plan->element_count = storage.element_count;
  out_plan->element_register_count = (uint8_t)storage.element_register_count;
  out_plan->element_bit_count = (uint8_t)storage.element_bit_count;
  out_plan->source_element_indices =
      loom_vector_shuffle_source_lanes(source_op).i64_array;
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

bool loom_amdgpu_vector_permutation_can_lower(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_SHUFFLE: {
      loom_amdgpu_vector_shuffle_plan_t plan = {0};
      return loom_amdgpu_vector_shuffle_plan_from_op(module, descriptor_set,
                                                     source_op, &plan);
    }
    case LOOM_OP_VECTOR_TRANSPOSE: {
      loom_amdgpu_vector_register_map_plan_t plan = {0};
      return loom_amdgpu_vector_transpose_plan_from_op(module, source_op,
                                                       &plan);
    }
    default:
      return false;
  }
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

  loom_amdgpu_vector_storage_t source_storage = {0};
  loom_amdgpu_vector_storage_t result_storage = {0};
  if (loom_amdgpu_static_rank1_register_tuple_storage_shape(source_type,
                                                            &source_storage) &&
      loom_amdgpu_static_rank1_register_tuple_storage_shape(result_type,
                                                            &result_storage) &&
      source_storage.kind == result_storage.kind &&
      source_storage.element_register_count ==
          result_storage.element_register_count &&
      !iree_any_bit_set(
          loom_amdgpu_vector_storage_kind_flags(source_storage.kind),
          LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    out_plan->source_register_count = source_storage.register_count;
    out_plan->result_register_count = result_storage.register_count;
    out_plan->register_offset =
        lane_offset * source_storage.element_register_count;
    out_plan->kind = LOOM_AMDGPU_VECTOR_SLICE_KIND_REGISTER_RANGE;
  } else {
    uint32_t source_register_count = 0;
    uint32_t result_register_count = 0;
    uint32_t source_element_bit_count = 0;
    uint32_t result_element_bit_count = 0;
    if (!loom_amdgpu_packed_register_slice_storage_shape(
            source_type, &source_register_count, &source_element_bit_count) ||
        !loom_amdgpu_packed_register_slice_storage_shape(
            result_type, &result_register_count, &result_element_bit_count) ||
        result_register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES ||
        source_element_bit_count != result_element_bit_count) {
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
  if (!*out_selected) {
    return iree_ok_status();
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, out_plan->result, &result_type));
  return loom_low_lower_plan_value_type(context, out_plan->result, result_type);
}

iree_status_t loom_amdgpu_lower_vector_bitcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bitcast_plan_t* plan) {
  loom_value_id_t low_input =
      loom_low_lower_lookup_value(context, plan->source);

  const loom_type_t result_low_type =
      loom_low_lower_value_binding_type(context, plan->result);
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
    loom_amdgpu_vector_concat_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_concat_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_concat(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_concat_plan_t* plan) {
  const loom_value_slice_t sources = loom_vector_concat_inputs(source_op);
  IREE_ASSERT_GT(sources.count, 0);
  if (sources.count == 1) {
    return loom_low_lower_bind_value_alias(context, sources.values[0],
                                           plan->result);
  }

  loom_value_id_t* low_sources = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, sources.count, sizeof(*low_sources), (void**)&low_sources));
  for (uint16_t i = 0; i < sources.count; ++i) {
    low_sources[i] = loom_low_lower_lookup_value(context, sources.values[i]);
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             low_sources, sources.count);
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
    low_sources[i] = loom_low_lower_lookup_value(context, plan->sources[i]);
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

iree_status_t loom_amdgpu_select_vector_deinterleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_deinterleave_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_deinterleave_plan_from_op(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan);
  return iree_ok_status();
}

static loom_type_t loom_amdgpu_vector_register_unit_type(
    loom_low_lower_context_t* context, loom_value_id_t low_value,
    uint32_t expected_register_count) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  IREE_ASSERT(loom_low_type_is_register(low_type));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(low_type),
                 expected_register_count);
  return loom_low_register_carrier_type_with_unit_count(low_type, 1);
}

static iree_status_t loom_amdgpu_lower_vector_even_odd(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_even_odd_layout_t* layout,
    const loom_value_id_t* sources, const loom_value_id_t* results) {
  const loom_vector_interleave_packet_plan_t* route = &layout->route;
  const uint32_t source_count =
      route->kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP ? 2u : 1u;
  const uint32_t result_count =
      route->kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP ? 1u : 2u;
  const loom_type_t result_type =
      loom_low_lower_value_binding_type(context, results[0]);
  const loom_type_t register_type =
      loom_low_register_carrier_type_with_unit_count(result_type, 1);
  loom_value_id_t low_sources[2];
  for (uint32_t i = 0; i < source_count; ++i) {
    low_sources[i] = loom_low_lower_lookup_value(context, sources[i]);
    if (loom_amdgpu_low_type_is_register_class(context, result_type,
                                               LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, low_sources[i], &low_sources[i]));
    }
  }
  loom_amdgpu_byte_permute_emitter_t emitter;
  if (layout->byte_permute.kind != LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_initialize(
        context, &layout->byte_permute, &emitter));
  }

  // Emission scratch is proportional to the selected physical words, with
  // one cache shared by both unzip results. No recipe-sized array limits the
  // logical vector or requires scalar reconstruction of an irregular row.
  const uint32_t source_register_count =
      source_count * route->source_packet_count;
  const uint32_t scratch_register_count =
      source_register_count + route->result_packet_count;
  loom_value_id_t inline_registers[2 * LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  loom_value_id_t* registers = inline_registers;
  if (scratch_register_count > IREE_ARRAYSIZE(inline_registers)) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, scratch_register_count, sizeof(*registers),
        (void**)&registers));
  }
  for (uint32_t i = 0; i < source_register_count; ++i) {
    registers[i] = LOOM_VALUE_ID_INVALID;
  }
  loom_value_id_t* result_registers = registers + source_register_count;
  for (uint32_t result_index = 0; result_index < result_count; ++result_index) {
    for (uint32_t packet = 0; packet < route->result_packet_count; ++packet) {
      loom_value_id_t word_sources[4];
      uint32_t word_source_count = 0;
      uint8_t selectors[4] = {0, 1, 2, 3};
      uint16_t position = 0;
      loom_vector_interleave_packet_segment_t segment;
      while (loom_vector_interleave_packet_plan_next_segment(
          route, result_index, packet, &position, &segment)) {
        const uint32_t register_index =
            segment.source_index * route->source_packet_count +
            segment.source_packet;
        if (registers[register_index] == LOOM_VALUE_ID_INVALID) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
              context, source_op, low_sources[segment.source_index],
              route->source_packet_count, segment.source_packet, register_type,
              &registers[register_index]));
        }
        const loom_value_id_t word = registers[register_index];
        uint32_t ordinal = 0;
        while (ordinal < word_source_count && word_sources[ordinal] != word) {
          ++ordinal;
        }
        if (ordinal == word_source_count) {
          word_sources[word_source_count++] = word;
        }
        for (uint32_t byte = 0; byte < segment.byte_count; ++byte) {
          selectors[segment.result_packet_byte_offset + byte] =
              ordinal * 4u + segment.source_packet_byte_offset + byte;
        }
      }
      if (layout->byte_permute.kind == LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE) {
        result_registers[packet] = word_sources[0];
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_gather(
            context, source_op, &emitter, word_sources, word_source_count,
            selectors, register_type, &result_registers[packet]));
      }
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_bind_low_register_range(
        context, source_op, results[result_index], result_registers,
        route->result_packet_count));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_deinterleave(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_deinterleave_plan_t* plan) {
  return loom_amdgpu_lower_vector_even_odd(context, source_op, &plan->layout,
                                           &plan->source, plan->results);
}

iree_status_t loom_amdgpu_select_vector_interleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_interleave_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_interleave_plan_from_op(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_interleave(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_interleave_plan_t* plan) {
  return loom_amdgpu_lower_vector_even_odd(context, source_op, &plan->layout,
                                           plan->sources, &plan->result);
}

iree_status_t loom_amdgpu_select_vector_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_shuffle_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_shuffle_plan_from_op(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan);
  return iree_ok_status();
}

static bool loom_amdgpu_vector_shuffle_is_source_alias(
    const loom_amdgpu_vector_shuffle_plan_t* plan) {
  for (uint32_t i = 0; i < plan->element_count; ++i) {
    if (plan->source_element_indices[i] != i) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_lower_vector_shuffle_register_units(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_shuffle_plan_t* plan) {
  const uint32_t register_count =
      plan->element_count * plan->element_register_count;
  loom_value_id_t low_source =
      loom_low_lower_lookup_value(context, plan->source);
  const loom_type_t register_type = loom_amdgpu_vector_register_unit_type(
      context, low_source, register_count);

  loom_value_id_t
      inline_registers[LOOM_AMDGPU_MAX_VECTOR_STORAGE_REGISTER_UNITS];
  loom_value_id_t* registers = inline_registers;
  if (register_count > IREE_ARRAYSIZE(inline_registers)) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, register_count, sizeof(*registers), (void**)&registers));
  }
  uint32_t result_register = 0;
  for (uint32_t result_element = 0; result_element < plan->element_count;
       ++result_element) {
    const uint32_t source_element =
        plan->source_element_indices[result_element];
    for (uint32_t unit = 0; unit < plan->element_register_count; ++unit) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_source, register_count,
          source_element * plan->element_register_count + unit, register_type,
          &registers[result_register++]));
    }
  }
  IREE_ASSERT_EQ(result_register, register_count);
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             registers, register_count);
}

static iree_status_t loom_amdgpu_lower_vector_shuffle_packed_bytes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_shuffle_plan_t* plan) {
  const uint32_t bytes_per_element = plan->element_bit_count / 8u;
  const uint32_t elements_per_register = 4u / bytes_per_element;
  const uint32_t register_count =
      (plan->element_count * bytes_per_element + 3u) / 4u;
  loom_type_t register_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  loom_value_id_t low_source =
      loom_low_lower_lookup_value(context, plan->source);
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source, &low_source));

  loom_value_id_t inline_registers[2 * LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  loom_value_id_t* registers = inline_registers;
  if (register_count > IREE_ARRAYSIZE(inline_registers) / 2u) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, 2u * register_count, sizeof(*registers), (void**)&registers));
  }
  for (uint32_t i = 0; i < register_count; ++i) {
    registers[i] = LOOM_VALUE_ID_INVALID;
  }
  loom_value_id_t* result_registers = registers + register_count;
  loom_amdgpu_byte_permute_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_initialize(
      context, &plan->byte_permute, &emitter));

  for (uint32_t result_register = 0; result_register < register_count;
       ++result_register) {
    const uint32_t result_element_base =
        result_register * elements_per_register;
    const uint32_t live_byte_count =
        bytes_per_element * iree_min(plan->element_count - result_element_base,
                                     elements_per_register);
    uint32_t source_register_indices[4];
    loom_value_id_t word_sources[4];
    uint32_t word_source_count = 0;
    uint8_t selectors[4] = {0, 1, 2, 3};
    for (uint32_t byte = 0; byte < live_byte_count; ++byte) {
      const uint32_t source_element =
          plan->source_element_indices[result_element_base +
                                       byte / bytes_per_element];
      const uint32_t source_byte =
          source_element * bytes_per_element + byte % bytes_per_element;
      const uint32_t source_register = source_byte / 4u;
      uint32_t ordinal = 0;
      while (ordinal < word_source_count &&
             source_register_indices[ordinal] != source_register) {
        ++ordinal;
      }
      if (ordinal == word_source_count) {
        source_register_indices[word_source_count] = source_register;
        if (registers[source_register] == LOOM_VALUE_ID_INVALID) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
              context, source_op, low_source, register_count, source_register,
              register_type, &registers[source_register]));
        }
        word_sources[word_source_count++] = registers[source_register];
      }
      selectors[byte] = ordinal * 4u + source_byte % 4u;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_gather(
        context, source_op, &emitter, word_sources, word_source_count,
        selectors, register_type, &result_registers[result_register]));
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers, register_count);
}

iree_status_t loom_amdgpu_lower_vector_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_shuffle_plan_t* plan) {
  if (loom_amdgpu_vector_shuffle_is_source_alias(plan)) {
    return loom_low_lower_bind_value_alias(context, plan->source, plan->result);
  }
  switch (plan->kind) {
    case LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_REGISTER_UNITS:
      return loom_amdgpu_lower_vector_shuffle_register_units(context, source_op,
                                                             plan);
    case LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_PACKED_BYTES:
      return loom_amdgpu_lower_vector_shuffle_packed_bytes(context, source_op,
                                                           plan);
    case LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("invalid AMDGPU vector shuffle plan");
      IREE_BUILTIN_UNREACHABLE();
  }
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

static bool loom_amdgpu_vector_slice_is_source_alias(
    const loom_amdgpu_vector_slice_plan_t* plan) {
  if (plan->kind == LOOM_AMDGPU_VECTOR_SLICE_KIND_REGISTER_RANGE) {
    return plan->register_offset == 0 &&
           plan->result_register_count == plan->source_register_count;
  }
  return plan->lane_offset == 0 && plan->source_register_count == 1;
}

static iree_status_t loom_amdgpu_lower_vector_slice_register_range(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* plan) {
  if (loom_amdgpu_vector_slice_is_source_alias(plan)) {
    return loom_low_lower_bind_value_alias(context, plan->source, plan->result);
  }

  const loom_value_id_t low_source =
      loom_low_lower_lookup_value(context, plan->source);
  const loom_type_t result_type =
      loom_low_lower_value_binding_type(context, plan->result);
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source, plan->register_offset, result_type,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

static iree_status_t loom_amdgpu_lower_vector_slice_packed_register_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* select, loom_value_id_t low_source,
    loom_type_t lane_type, loom_value_id_t* low_registers) {
  const uint64_t source_bit_offset =
      (uint64_t)select->lane_offset * select->element_bit_count;
  const uint32_t result_bit_count =
      select->lane_count * select->element_bit_count;
  const uint32_t shift = source_bit_offset & 31u;
  loom_low_lower_resolved_descriptor_t align_descriptor = {0};
  loom_value_id_t shift_value = LOOM_VALUE_ID_INVALID;
  if (shift + iree_min(result_bit_count, 32u) > 32u) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_ALIGNBIT_B32, &align_descriptor));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, shift,
        lane_type, &shift_value));
  }
  for (uint32_t i = 0; i < select->result_register_count; ++i) {
    const uint64_t register_source_bit_offset = source_bit_offset + i * 32u;
    const uint32_t source_register_index =
        (uint32_t)(register_source_bit_offset / 32u);
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
    const uint32_t needed_bit_count = iree_min(result_bit_count - i * 32u, 32u);
    if (source_register_bit_offset + needed_bit_count > 32u) {
      // The verified slice window contains the required bits in the adjacent
      // source dword. A partial final result needs no high word when all of
      // its live bits already fit in the low word.
      loom_value_id_t high_register = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, source_register_index + 1u, lane_type,
          &high_register));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
          context, source_op, &align_descriptor, high_register, source_register,
          shift_value, lane_type, &low_registers[i]));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          source_register_bit_offset, source_register, lane_type,
          &low_registers[i]));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* plan) {
  if (plan->kind == LOOM_AMDGPU_VECTOR_SLICE_KIND_REGISTER_RANGE) {
    return loom_amdgpu_lower_vector_slice_register_range(context, source_op,
                                                         plan);
  }

  loom_value_id_t low_source =
      loom_low_lower_lookup_value(context, plan->source);

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_slice_packed_register_bits(
      context, source_op, plan, low_source, lane_type, low_registers));

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             low_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_finalize_vector_structural_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t selected_plan) {
  const uint64_t descriptor_set_id =
      loom_low_lower_context_descriptor_set(context)->stable_id;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  loom_type_t result_type = loom_type_none();
  switch (selected_plan.id) {
    case LOOM_OP_VECTOR_CONCAT: {
      const loom_amdgpu_vector_concat_plan_t* plan = selected_plan.target_data;
      const loom_value_slice_t sources = loom_vector_concat_inputs(source_op);
      result = plan->result;
      result_type =
          loom_low_lower_value_binding_type(context, sources.values[0]);
      if (sources.count != 1) {
        uint32_t unit_count = loom_low_register_type_unit_count(result_type);
        for (uint16_t i = 1; i < sources.count; ++i) {
          unit_count += loom_low_register_type_unit_count(
              loom_low_lower_value_binding_type(context, sources.values[i]));
        }
        result_type = loom_low_register_carrier_type_with_unit_count(
            result_type, unit_count);
      }
      break;
    }
    case LOOM_OP_VECTOR_TRANSPOSE: {
      const loom_amdgpu_vector_register_map_plan_t* plan =
          selected_plan.target_data;
      result = plan->result;
      if (loom_amdgpu_vector_register_map_is_source_alias(plan)) {
        result_type =
            loom_low_lower_value_binding_type(context, plan->sources[0]);
      } else {
        const uint32_t source_index = plan->result_source_indices[0];
        result_type =
            plan->source_register_counts[source_index] == 1
                ? loom_low_lower_value_binding_type(context,
                                                    plan->sources[source_index])
                : loom_low_register_type(descriptor_set_id,
                                         LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
        if (plan->result_register_count != 1) {
          result_type = loom_low_register_carrier_type_with_unit_count(
              result_type, plan->result_register_count);
        }
      }
      break;
    }
    case LOOM_OP_VECTOR_SLICE: {
      const loom_amdgpu_vector_slice_plan_t* plan = selected_plan.target_data;
      result = plan->result;
      const loom_type_t source_type =
          loom_low_lower_value_binding_type(context, plan->source);
      result_type = loom_amdgpu_vector_slice_is_source_alias(plan)
                        ? source_type
                        : loom_low_register_carrier_type_with_unit_count(
                              source_type, plan->result_register_count);
      break;
    }
    case LOOM_OP_VECTOR_DEINTERLEAVE: {
      const loom_amdgpu_vector_deinterleave_plan_t* plan =
          selected_plan.target_data;
      result_type =
          plan->layout.byte_permute.kind == LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE
              ? loom_low_register_carrier_type_with_unit_count(
                    loom_low_lower_value_binding_type(context, plan->source),
                    plan->layout.route.result_packet_count)
              : loom_low_register_type(descriptor_set_id,
                                       LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                       plan->layout.route.result_packet_count);
      IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
          context, plan->results[0], result_type));
      result = plan->results[1];
      break;
    }
    case LOOM_OP_VECTOR_INTERLEAVE: {
      const loom_amdgpu_vector_interleave_plan_t* plan =
          selected_plan.target_data;
      result = plan->result;
      if (plan->layout.byte_permute.kind ==
          LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE) {
        const loom_type_t source_type = loom_module_value_type(
            loom_low_lower_context_module(context), plan->sources[0]);
        const loom_type_t joined_type = loom_amdgpu_join_result_type(
            source_type,
            loom_low_lower_value_binding_type(context, plan->sources[0]),
            loom_low_lower_value_binding_type(context, plan->sources[1]));
        result_type = loom_low_register_carrier_type_with_unit_count(
            joined_type, plan->layout.route.result_packet_count);
      } else {
        result_type = loom_low_register_type(
            descriptor_set_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
            plan->layout.route.result_packet_count);
      }
      break;
    }
    case LOOM_OP_VECTOR_SHUFFLE: {
      const loom_amdgpu_vector_shuffle_plan_t* plan = selected_plan.target_data;
      result = plan->result;
      if (loom_amdgpu_vector_shuffle_is_source_alias(plan)) {
        result_type = loom_low_lower_value_binding_type(context, plan->source);
      } else if (plan->kind == LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_REGISTER_UNITS) {
        result_type = loom_low_register_carrier_type_with_unit_count(
            loom_low_lower_value_binding_type(context, plan->source),
            plan->element_count * plan->element_register_count);
      } else {
        result_type = loom_low_register_type(
            descriptor_set_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
            (plan->element_count * plan->element_bit_count + 31u) / 32u);
      }
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown structural vector plan");
      IREE_BUILTIN_UNREACHABLE();
  }
  return loom_low_lower_plan_value_type(context, result, result_type);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_structural(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
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
      loom_amdgpu_vector_concat_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_concat_plan_from_op(module, op, &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("concat.register_storage"));
    }
    case LOOM_OP_VECTOR_DEINTERLEAVE: {
      loom_amdgpu_vector_deinterleave_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_deinterleave_plan_from_op(
              module, loom_target_low_legality_descriptor_set(context), op,
              &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("deinterleave.even_odd_storage"));
    }
    case LOOM_OP_VECTOR_INTERLEAVE: {
      loom_amdgpu_vector_interleave_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_interleave_plan_from_op(
              module, loom_target_low_legality_descriptor_set(context), op,
              &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, IREE_SV("interleave.even_odd_storage"));
    }
    case LOOM_OP_VECTOR_SHUFFLE: {
      loom_amdgpu_vector_shuffle_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_shuffle_plan_from_op(
              module, loom_target_low_legality_descriptor_set(context), op,
              &unused_plan)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("shuffle.storage"));
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
