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
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;

  loom_amdgpu_vector_storage_t storage;
  if (loom_amdgpu_type_vector_storage(type, &storage) &&
      (storage.kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT ||
       storage.kind == LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT)) {
    *out_payload_bit_count = storage.element_count * storage.element_bit_count;
    *out_register_count = storage.register_count;
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
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_amdgpu_type_vector_storage(type, out_storage) &&
         out_storage->register_count != 0 &&
         out_storage->register_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES;
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

static bool loom_amdgpu_vector_even_odd_layout(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t half_type,
    loom_type_t combined_type, int64_t axis,
    loom_amdgpu_vector_even_odd_layout_t* out_layout) {
  *out_layout = (loom_amdgpu_vector_even_odd_layout_t){0};
  if (!loom_type_is_vector(half_type) || !loom_type_is_vector(combined_type) ||
      !loom_type_element_type_equals(half_type, combined_type) ||
      loom_type_rank(half_type) != loom_type_rank(combined_type) ||
      !loom_type_is_all_static(half_type) ||
      !loom_type_is_all_static(combined_type) || axis < 0 ||
      axis >= loom_type_rank(half_type)) {
    return false;
  }

  loom_amdgpu_vector_storage_t half_storage = {0};
  loom_amdgpu_vector_storage_t combined_storage = {0};
  if (!loom_amdgpu_type_vector_storage(half_type, &half_storage) ||
      !loom_amdgpu_type_vector_storage(combined_type, &combined_storage) ||
      half_storage.kind != combined_storage.kind ||
      half_storage.element_type != combined_storage.element_type ||
      half_storage.element_bit_count != combined_storage.element_bit_count ||
      half_storage.element_register_count !=
          combined_storage.element_register_count ||
      combined_storage.element_count != 2u * half_storage.element_count ||
      half_storage.register_count > UINT8_MAX ||
      combined_storage.register_count > UINT8_MAX) {
    return false;
  }

  uint64_t outer_group_count = 1;
  uint64_t inner_group_count = 1;
  const uint8_t rank = loom_type_rank(half_type);
  for (uint8_t dimension = 0; dimension < rank; ++dimension) {
    const int64_t half_size =
        loom_type_dim_static_size_at(half_type, dimension);
    const int64_t combined_size =
        loom_type_dim_static_size_at(combined_type, dimension);
    if (half_size < 1 || combined_size < 1 ||
        (dimension == (uint8_t)axis
             ? (half_size > INT64_MAX / 2 || combined_size != half_size * 2)
             : combined_size != half_size)) {
      return false;
    }
    if (dimension < (uint8_t)axis) {
      outer_group_count *= (uint64_t)half_size;
    } else if (dimension > (uint8_t)axis) {
      inner_group_count *= (uint64_t)half_size;
    }
  }
  const int64_t axis_group_count =
      loom_type_dim_static_size_at(half_type, (uint8_t)axis);
  if (outer_group_count > UINT8_MAX || inner_group_count > UINT8_MAX ||
      axis_group_count > UINT8_MAX) {
    return false;
  }

  const loom_amdgpu_vector_storage_kind_flags_t storage_flags =
      loom_amdgpu_vector_storage_kind_flags(half_storage.kind);
  if (iree_any_bit_set(storage_flags,
                       LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    if (rank != 1 || axis != 0 ||
        (half_storage.element_bit_count != 8 &&
         half_storage.element_bit_count != 16)) {
      return false;
    }
    loom_amdgpu_select_byte_permute_plan(descriptor_set,
                                         &out_layout->byte_permute);
    if (out_layout->byte_permute.kind == LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE) {
      return false;
    }
    out_layout->kind = LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_PACKED_BYTES;
  } else {
    switch (half_storage.kind) {
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT:
      case LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT:
        out_layout->kind = LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_REGISTER_UNITS;
        break;
      default:
        return false;
    }
  }

  out_layout->outer_group_count = (uint8_t)outer_group_count;
  out_layout->axis_group_count = (uint8_t)axis_group_count;
  out_layout->inner_group_count = (uint8_t)inner_group_count;
  out_layout->element_register_count =
      (uint8_t)half_storage.element_register_count;
  out_layout->element_bit_count = (uint8_t)half_storage.element_bit_count;
  out_layout->half_register_count = (uint8_t)half_storage.register_count;
  out_layout->combined_register_count =
      (uint8_t)combined_storage.register_count;
  return true;
}

static bool loom_amdgpu_static_32bit_vector_register_shape(
    loom_type_t type, uint32_t* out_register_count) {
  *out_register_count = loom_amdgpu_vector_32bit_register_count(type);
  return *out_register_count != 0 &&
         *out_register_count <= LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES;
}

static bool loom_amdgpu_vector_concat_plan_from_op(
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
  if (!loom_type_equal(even_type, odd_type) ||
      !loom_type_element_type_equals(source_type, even_type) ||
      !loom_amdgpu_vector_even_odd_layout(
          descriptor_set, even_type, source_type,
          loom_vector_deinterleave_axis(source_op), &out_plan->layout)) {
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
          loom_vector_interleave_axis(source_op), &out_plan->layout)) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vector_shuffle_uses_register_map(
    const loom_amdgpu_vector_storage_t* storage, loom_attribute_t source_lanes,
    loom_amdgpu_vector_shuffle_plan_t* out_plan) {
  if (storage->element_register_count != 1 || storage->element_bit_count == 0 ||
      32u % storage->element_bit_count != 0) {
    return false;
  }
  const uint32_t lanes_per_register = 32u / storage->element_bit_count;
  for (uint32_t register_index = 0; register_index < storage->register_count;
       ++register_index) {
    const uint32_t result_lane_base = register_index * lanes_per_register;
    const uint32_t remaining_lane_count =
        storage->element_count - result_lane_base;
    const uint32_t result_lane_count =
        iree_min(remaining_lane_count, lanes_per_register);
    const uint32_t source_lane_base =
        (uint32_t)source_lanes.i64_array[result_lane_base];
    if (source_lane_base % lanes_per_register != 0) {
      return false;
    }
    for (uint32_t lane_offset = 1; lane_offset < result_lane_count;
         ++lane_offset) {
      if ((uint32_t)source_lanes.i64_array[result_lane_base + lane_offset] !=
          source_lane_base + lane_offset) {
        return false;
      }
    }
    out_plan->strategy.register_map.source_register_indices[register_index] =
        (uint8_t)(source_lane_base / lanes_per_register);
  }
  out_plan->kind = LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_REGISTER_MAP;
  return true;
}

static bool loom_amdgpu_vector_shuffle_uses_packed_byte_permute(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_vector_storage_t* storage, loom_attribute_t source_lanes,
    loom_amdgpu_vector_shuffle_plan_t* out_plan) {
  if (storage->element_bit_count != 8 ||
      storage->register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      !iree_any_bit_set(loom_amdgpu_vector_storage_kind_flags(storage->kind),
                        LOOM_AMDGPU_VECTOR_STORAGE_KIND_FLAG_PACKED_PAYLOAD)) {
    return false;
  }
  loom_amdgpu_select_byte_permute_plan(descriptor_set,
                                       &out_plan->strategy.packed_bytes.packet);
  if (out_plan->strategy.packed_bytes.packet.kind ==
      LOOM_AMDGPU_BYTE_PERMUTE_KIND_NONE) {
    return false;
  }

  for (uint32_t result_register_index = 0;
       result_register_index < storage->register_count;
       ++result_register_index) {
    const uint32_t result_lane_base = result_register_index * 4u;
    const uint32_t remaining_lane_count =
        storage->element_count - result_lane_base;
    const uint32_t result_lane_count = iree_min(remaining_lane_count, 4u);
    uint8_t source_register_indices[2] = {0};
    uint32_t source_register_count = 0;
    for (uint32_t byte_index = 0; byte_index < result_lane_count;
         ++byte_index) {
      const uint32_t source_lane =
          (uint32_t)source_lanes.i64_array[result_lane_base + byte_index];
      const uint8_t source_register_index = (uint8_t)(source_lane / 4u);
      uint32_t source_index = 0;
      while (source_index < source_register_count &&
             source_register_indices[source_index] != source_register_index) {
        ++source_index;
      }
      if (source_index == source_register_count) {
        if (source_register_count == IREE_ARRAYSIZE(source_register_indices)) {
          return false;
        }
        source_register_indices[source_register_count++] =
            source_register_index;
      }
    }
    IREE_ASSERT_GT(source_register_count, 0);
    if (source_register_count == 1) {
      source_register_indices[1] = source_register_indices[0];
    }
    out_plan->strategy.packed_bytes
        .source_register_indices[0][result_register_index] =
        source_register_indices[0];
    out_plan->strategy.packed_bytes
        .source_register_indices[1][result_register_index] =
        source_register_indices[1];

    uint32_t selector = 0;
    for (uint32_t byte_index = 0; byte_index < 4u; ++byte_index) {
      const uint32_t source_lane =
          byte_index < result_lane_count
              ? (uint32_t)source_lanes.i64_array[result_lane_base + byte_index]
              : (uint32_t)source_register_indices[0] * 4u;
      const uint32_t source_register_index = source_lane / 4u;
      const uint32_t source_byte_index = source_lane % 4u;
      const uint32_t selector_byte =
          source_byte_index +
          (source_register_index == source_register_indices[0] ? 4u : 0u);
      selector |= selector_byte << (byte_index * 8u);
    }
    out_plan->strategy.packed_bytes.selectors[result_register_index] = selector;
  }
  out_plan->kind = LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_PACKED_BYTE_PERMUTE;
  return true;
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
      !loom_amdgpu_static_rank1_register_storage_shape(source_type, &storage)) {
    return false;
  }

  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(source_op);
  if (source_lanes.kind != LOOM_ATTR_I64_ARRAY ||
      source_lanes.count != storage.element_count) {
    return false;
  }
  for (uint16_t i = 0; i < source_lanes.count; ++i) {
    if (source_lanes.i64_array[i] < 0 ||
        source_lanes.i64_array[i] >= storage.element_count) {
      return false;
    }
  }
  out_plan->register_count = (uint8_t)storage.register_count;
  return loom_amdgpu_vector_shuffle_uses_register_map(&storage, source_lanes,
                                                      out_plan) ||
         loom_amdgpu_vector_shuffle_uses_packed_byte_permute(
             descriptor_set, &storage, source_lanes, out_plan);
}

bool loom_amdgpu_vector_shuffle_can_lower(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op) {
  loom_amdgpu_vector_shuffle_plan_t unused_plan = {0};
  return loom_amdgpu_vector_shuffle_plan_from_op(module, descriptor_set,
                                                 source_op, &unused_plan);
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
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, sources.values[i],
                                                     &low_sources[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_sources, sources.count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
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

static iree_status_t loom_amdgpu_lower_vector_deinterleave_register_units(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_deinterleave_plan_t* plan) {
  const loom_amdgpu_vector_even_odd_layout_t* layout = &plan->layout;
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  const loom_type_t register_type = loom_amdgpu_vector_register_unit_type(
      context, low_source, layout->combined_register_count);

  for (uint32_t result_index = 0; result_index < IREE_ARRAYSIZE(plan->results);
       ++result_index) {
    loom_value_id_t registers[LOOM_AMDGPU_MAX_VECTOR_STORAGE_REGISTER_UNITS];
    uint32_t output_register = 0;
    for (uint32_t outer = 0; outer < layout->outer_group_count; ++outer) {
      for (uint32_t axis = 0; axis < layout->axis_group_count; ++axis) {
        for (uint32_t inner = 0; inner < layout->inner_group_count; ++inner) {
          const uint32_t source_element =
              ((outer * (layout->axis_group_count * 2u) + axis * 2u +
                result_index) *
                   layout->inner_group_count +
               inner);
          for (uint32_t unit = 0; unit < layout->element_register_count;
               ++unit) {
            IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
                context, source_op, low_source, layout->combined_register_count,
                source_element * layout->element_register_count + unit,
                register_type, &registers[output_register++]));
          }
        }
      }
    }
    IREE_ASSERT_EQ(output_register, layout->half_register_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_bind_low_register_range(
        context, source_op, plan->results[result_index], registers,
        output_register));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_interleave_register_units(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_interleave_plan_t* plan) {
  const loom_amdgpu_vector_even_odd_layout_t* layout = &plan->layout;
  loom_value_id_t low_sources[2] = {LOOM_VALUE_ID_INVALID,
                                    LOOM_VALUE_ID_INVALID};
  loom_type_t register_type = loom_type_none();
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(low_sources); ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->sources[i],
                                                     &low_sources[i]));
    const loom_type_t source_register_type =
        loom_amdgpu_vector_register_unit_type(context, low_sources[i],
                                              layout->half_register_count);
    if (i == 0) {
      register_type = source_register_type;
    } else {
      IREE_ASSERT(loom_type_equal(register_type, source_register_type));
    }
  }

  loom_value_id_t registers[LOOM_AMDGPU_MAX_VECTOR_STORAGE_REGISTER_UNITS];
  uint32_t output_register = 0;
  for (uint32_t outer = 0; outer < layout->outer_group_count; ++outer) {
    for (uint32_t axis = 0; axis < layout->axis_group_count; ++axis) {
      for (uint32_t source_index = 0;
           source_index < IREE_ARRAYSIZE(low_sources); ++source_index) {
        for (uint32_t inner = 0; inner < layout->inner_group_count; ++inner) {
          const uint32_t source_element =
              ((outer * layout->axis_group_count + axis) *
                   layout->inner_group_count +
               inner);
          for (uint32_t unit = 0; unit < layout->element_register_count;
               ++unit) {
            IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
                context, source_op, low_sources[source_index],
                layout->half_register_count,
                source_element * layout->element_register_count + unit,
                register_type, &registers[output_register++]));
          }
        }
      }
    }
  }
  IREE_ASSERT_EQ(output_register, layout->combined_register_count);
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             registers, output_register);
}

typedef uint8_t loom_amdgpu_vector_even_odd_direction_t;
enum loom_amdgpu_vector_even_odd_direction_e {
  LOOM_AMDGPU_VECTOR_EVEN_ODD_DIRECTION_INTERLEAVE = 0,
  LOOM_AMDGPU_VECTOR_EVEN_ODD_DIRECTION_DEINTERLEAVE = 1,
};

typedef struct loom_amdgpu_vector_packed_byte_source_t {
  // Source value ordinal within the even/odd source pair.
  uint8_t source_index;
  // Backing register ordinal within the selected source value.
  uint8_t register_index;
  // Byte ordinal within the selected backing register.
  uint8_t byte_index;
} loom_amdgpu_vector_packed_byte_source_t;

typedef struct loom_amdgpu_vector_packed_source_registers_t {
  // Materialized Low register tuple supplying packed payload words.
  loom_value_id_t low_value;
  // Number of backing registers in |low_value|.
  uint32_t register_count;
  // Register words cached after their first use.
  loom_value_id_t registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
} loom_amdgpu_vector_packed_source_registers_t;

static loom_amdgpu_vector_packed_byte_source_t
loom_amdgpu_vector_even_odd_packed_byte_source(
    loom_amdgpu_vector_even_odd_direction_t direction, uint32_t result_index,
    uint32_t result_lane, uint32_t lane_byte, uint32_t bytes_per_element) {
  uint32_t source_index = 0;
  uint32_t source_lane = 0;
  if (direction == LOOM_AMDGPU_VECTOR_EVEN_ODD_DIRECTION_INTERLEAVE) {
    source_index = result_lane & 1u;
    source_lane = result_lane / 2u;
  } else {
    source_lane = result_lane * 2u + result_index;
  }
  const uint32_t source_byte = source_lane * bytes_per_element + lane_byte;
  return (loom_amdgpu_vector_packed_byte_source_t){
      .source_index = (uint8_t)source_index,
      .register_index = (uint8_t)(source_byte / 4u),
      .byte_index = (uint8_t)(source_byte % 4u),
  };
}

static iree_status_t loom_amdgpu_vector_packed_source_registers_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, uint32_t register_count,
    loom_amdgpu_vector_packed_source_registers_t* out_registers) {
  IREE_ASSERT_LE(register_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  *out_registers = (loom_amdgpu_vector_packed_source_registers_t){
      .low_value = LOOM_VALUE_ID_INVALID,
      .register_count = register_count,
  };
  for (uint32_t i = 0; i < register_count; ++i) {
    out_registers->registers[i] = LOOM_VALUE_ID_INVALID;
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, &out_registers->low_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, out_registers->low_value, &out_registers->low_value));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packed_source_registers_get(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_packed_source_registers_t* registers,
    uint32_t register_index, loom_type_t register_type,
    loom_value_id_t* out_register) {
  IREE_ASSERT_LT(register_index, registers->register_count);
  if (registers->registers[register_index] == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, registers->low_value, registers->register_count,
        register_index, register_type, &registers->registers[register_index]));
  }
  *out_register = registers->registers[register_index];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_even_odd_packed_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_even_odd_layout_t* layout,
    loom_amdgpu_vector_even_odd_direction_t direction, uint32_t result_index,
    uint32_t result_register_index,
    loom_amdgpu_vector_packed_source_registers_t source_registers[2],
    loom_type_t register_type, loom_amdgpu_byte_permute_emitter_t* emitter,
    loom_value_id_t* out_register) {
  const uint32_t bytes_per_element = layout->element_bit_count / 8u;
  const uint32_t elements_per_register = 4u / bytes_per_element;
  const uint32_t half_element_count = layout->outer_group_count *
                                      layout->axis_group_count *
                                      layout->inner_group_count;
  const uint32_t result_element_count =
      direction == LOOM_AMDGPU_VECTOR_EVEN_ODD_DIRECTION_INTERLEAVE
          ? half_element_count * 2u
          : half_element_count;
  const uint32_t result_lane_base =
      result_register_index * elements_per_register;

  loom_amdgpu_vector_packed_byte_source_t byte_sources[4];
  loom_amdgpu_vector_packed_byte_source_t register_sources[2];
  uint32_t register_source_count = 0;
  for (uint32_t output_byte = 0; output_byte < IREE_ARRAYSIZE(byte_sources);
       ++output_byte) {
    uint32_t result_lane = result_lane_base + output_byte / bytes_per_element;
    if (result_lane >= result_element_count) {
      result_lane = result_lane_base;
    }
    byte_sources[output_byte] = loom_amdgpu_vector_even_odd_packed_byte_source(
        direction, result_index, result_lane, output_byte % bytes_per_element,
        bytes_per_element);

    uint32_t source_ordinal = 0;
    while (source_ordinal < register_source_count &&
           (register_sources[source_ordinal].source_index !=
                byte_sources[output_byte].source_index ||
            register_sources[source_ordinal].register_index !=
                byte_sources[output_byte].register_index)) {
      ++source_ordinal;
    }
    if (source_ordinal == register_source_count) {
      IREE_ASSERT_LT(register_source_count, IREE_ARRAYSIZE(register_sources));
      register_sources[register_source_count++] = byte_sources[output_byte];
    }
  }
  IREE_ASSERT_GT(register_source_count, 0);
  if (register_source_count == 1) {
    register_sources[1] = register_sources[0];
  }

  uint32_t selector = 0;
  for (uint32_t output_byte = 0; output_byte < IREE_ARRAYSIZE(byte_sources);
       ++output_byte) {
    uint32_t source_ordinal = 0;
    while (source_ordinal < IREE_ARRAYSIZE(register_sources) &&
           (register_sources[source_ordinal].source_index !=
                byte_sources[output_byte].source_index ||
            register_sources[source_ordinal].register_index !=
                byte_sources[output_byte].register_index)) {
      ++source_ordinal;
    }
    IREE_ASSERT_LT(source_ordinal, IREE_ARRAYSIZE(register_sources));
    const uint32_t selector_byte =
        byte_sources[output_byte].byte_index + (source_ordinal == 0 ? 4u : 0u);
    selector |= selector_byte << (output_byte * 8u);
  }

  loom_value_id_t low_sources[2] = {LOOM_VALUE_ID_INVALID,
                                    LOOM_VALUE_ID_INVALID};
  for (uint32_t source_ordinal = 0;
       source_ordinal < IREE_ARRAYSIZE(register_sources); ++source_ordinal) {
    const loom_amdgpu_vector_packed_byte_source_t source =
        register_sources[source_ordinal];
    IREE_ASSERT_LT(source.source_index, 2u);
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packed_source_registers_get(
        context, source_op, &source_registers[source.source_index],
        source.register_index, register_type, &low_sources[source_ordinal]));
  }
  return loom_amdgpu_byte_permute_emitter_emit(
      context, source_op, emitter, low_sources[0], low_sources[1], selector,
      register_type, out_register);
}

static iree_status_t loom_amdgpu_lower_vector_deinterleave_packed_bytes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_deinterleave_plan_t* plan) {
  const loom_amdgpu_vector_even_odd_layout_t* layout = &plan->layout;
  loom_type_t register_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  loom_amdgpu_vector_packed_source_registers_t source_registers[2] = {{0}};
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packed_source_registers_initialize(
      context, source_op, plan->source, layout->combined_register_count,
      &source_registers[0]));
  loom_amdgpu_byte_permute_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_initialize(
      context, &layout->byte_permute, &emitter));

  for (uint32_t result_index = 0; result_index < IREE_ARRAYSIZE(plan->results);
       ++result_index) {
    loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    for (uint32_t register_index = 0;
         register_index < layout->half_register_count; ++register_index) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_even_odd_packed_register(
          context, source_op, layout,
          LOOM_AMDGPU_VECTOR_EVEN_ODD_DIRECTION_DEINTERLEAVE, result_index,
          register_index, source_registers, register_type, &emitter,
          &result_registers[register_index]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_bind_low_register_range(
        context, source_op, plan->results[result_index], result_registers,
        layout->half_register_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_interleave_packed_bytes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_interleave_plan_t* plan) {
  const loom_amdgpu_vector_even_odd_layout_t* layout = &plan->layout;
  loom_type_t register_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  loom_amdgpu_vector_packed_source_registers_t source_registers[2] = {{0}};
  for (uint32_t source_index = 0; source_index < IREE_ARRAYSIZE(plan->sources);
       ++source_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packed_source_registers_initialize(
        context, source_op, plan->sources[source_index],
        layout->half_register_count, &source_registers[source_index]));
  }
  loom_amdgpu_byte_permute_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_initialize(
      context, &layout->byte_permute, &emitter));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t register_index = 0;
       register_index < layout->combined_register_count; ++register_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_even_odd_packed_register(
        context, source_op, layout,
        LOOM_AMDGPU_VECTOR_EVEN_ODD_DIRECTION_INTERLEAVE, 0, register_index,
        source_registers, register_type, &emitter,
        &result_registers[register_index]));
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             layout->combined_register_count);
}

iree_status_t loom_amdgpu_lower_vector_deinterleave(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_deinterleave_plan_t* plan) {
  switch (plan->layout.kind) {
    case LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_REGISTER_UNITS:
      return loom_amdgpu_lower_vector_deinterleave_register_units(
          context, source_op, plan);
    case LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_PACKED_BYTES:
      return loom_amdgpu_lower_vector_deinterleave_packed_bytes(
          context, source_op, plan);
    case LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("invalid AMDGPU vector deinterleave plan");
      IREE_BUILTIN_UNREACHABLE();
  }
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
  switch (plan->layout.kind) {
    case LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_REGISTER_UNITS:
      return loom_amdgpu_lower_vector_interleave_register_units(
          context, source_op, plan);
    case LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_PACKED_BYTES:
      return loom_amdgpu_lower_vector_interleave_packed_bytes(context,
                                                              source_op, plan);
    case LOOM_AMDGPU_VECTOR_EVEN_ODD_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("invalid AMDGPU vector interleave plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_select_vector_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_shuffle_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_vector_shuffle_plan_from_op(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_shuffle_register_map(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_shuffle_plan_t* plan) {
  loom_amdgpu_vector_register_map_plan_t map_plan = {
      .sources = {plan->source},
      .result = plan->result,
      .source_count = 1,
      .result_register_count = plan->register_count,
      .source_register_counts = {plan->register_count},
  };
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    map_plan.result_source_indices[i] = 0;
    map_plan.source_register_indices[i] =
        plan->strategy.register_map.source_register_indices[i];
  }
  return loom_amdgpu_lower_vector_register_map(context, source_op, &map_plan);
}

static iree_status_t loom_amdgpu_lower_vector_shuffle_packed_bytes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_shuffle_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  const loom_type_t low_source_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_source);
  if (!loom_amdgpu_low_type_is_register_class(context, low_source_type,
                                              LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    IREE_ASSERT(loom_amdgpu_low_type_is_register_class(
        context, low_source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_source, &low_source));
  }

  loom_type_t register_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &register_type));
  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->register_count, i, register_type,
        &source_registers[i]));
  }

  loom_amdgpu_byte_permute_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_initialize(
      context, &plan->strategy.packed_bytes.packet, &emitter));

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    const uint8_t source0_index =
        plan->strategy.packed_bytes.source_register_indices[0][i];
    const uint8_t source1_index =
        plan->strategy.packed_bytes.source_register_indices[1][i];
    IREE_ASSERT_LT(source0_index, plan->register_count);
    IREE_ASSERT_LT(source1_index, plan->register_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_byte_permute_emitter_emit(
        context, source_op, &emitter, source_registers[source0_index],
        source_registers[source1_index],
        plan->strategy.packed_bytes.selectors[i], register_type,
        &result_registers[i]));
  }
  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, result_registers, plan->register_count);
}

iree_status_t loom_amdgpu_lower_vector_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_shuffle_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_REGISTER_MAP:
      return loom_amdgpu_lower_vector_shuffle_register_map(context, source_op,
                                                           plan);
    case LOOM_AMDGPU_VECTOR_SHUFFLE_KIND_PACKED_BYTE_PERMUTE:
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
