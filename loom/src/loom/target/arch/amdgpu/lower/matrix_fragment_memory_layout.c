// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_layout.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_address.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_fragment_memory_layout_reject(
    iree_string_view_t constraint_key, iree_string_view_t* out_constraint_key) {
  if (out_constraint_key != NULL &&
      iree_string_view_is_empty(*out_constraint_key)) {
    *out_constraint_key = constraint_key;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_add_scaled_stride(
    uint32_t factor, uint32_t byte_stride, uint32_t* inout_byte_stride) {
  const uint64_t result =
      (uint64_t)*inout_byte_stride + (uint64_t)factor * byte_stride;
  if (result > UINT32_MAX) return false;
  *inout_byte_stride = (uint32_t)result;
  return true;
}

static bool loom_amdgpu_fragment_memory_append_lane_term(
    uint16_t divisor, uint16_t modulus, uint32_t byte_stride,
    loom_amdgpu_fragment_memory_address_layout_t* address_layout) {
  IREE_ASSERT_TRUE(loom_amdgpu_u32_is_power_of_two(divisor));
  IREE_ASSERT_TRUE(modulus == 0 || loom_amdgpu_u32_is_power_of_two(modulus));
  uint8_t insert_index = address_layout->lane_term_count;
  for (uint8_t i = 0; i < address_layout->lane_term_count; ++i) {
    loom_amdgpu_fragment_memory_lane_term_t* term =
        &address_layout->lane_terms[i];
    if (term->divisor == divisor && term->modulus == modulus) {
      return loom_amdgpu_fragment_memory_add_scaled_stride(1, byte_stride,
                                                           &term->byte_stride);
    }
    if (insert_index == address_layout->lane_term_count &&
        (term->divisor > divisor ||
         (term->divisor == divisor && term->modulus > modulus))) {
      insert_index = i;
    }
  }
  IREE_ASSERT_LT(address_layout->lane_term_count,
                 IREE_ARRAYSIZE(address_layout->lane_terms));
  for (uint8_t i = address_layout->lane_term_count; i > insert_index; --i) {
    address_layout->lane_terms[i] = address_layout->lane_terms[i - 1];
  }
  address_layout->lane_terms[insert_index] =
      (loom_amdgpu_fragment_memory_lane_term_t){
          .divisor = divisor,
          .modulus = modulus,
          .byte_stride = byte_stride,
      };
  ++address_layout->lane_term_count;
  return true;
}

static void loom_amdgpu_fragment_memory_register_coordinates(
    const loom_matrix_fragment_role_layout_t* role_layout,
    uint16_t payload_elements_per_register,
    uint16_t payload_registers_per_element, uint16_t register_index,
    uint32_t* out_intra_element_byte_offset, uint32_t* out_coordinates) {
  const uint32_t element_register_index =
      register_index / payload_registers_per_element;
  const uint32_t payload_element_index =
      element_register_index * payload_elements_per_register;
  *out_intra_element_byte_offset =
      (register_index % payload_registers_per_element) *
      LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT;
  IREE_ASSERT_EQ(payload_element_index % role_layout->coordinate_element_stride,
                 0);
  const uint32_t coordinate_element_index =
      payload_element_index / role_layout->coordinate_element_stride;
  IREE_ASSERT_LT(coordinate_element_index,
                 role_layout->coordinate_element_count);
  const loom_matrix_fragment_coordinate_projection_plan_t* projection =
      role_layout->coordinate_projection_plan;
  IREE_ASSERT_TRUE(projection != NULL);
  const uint32_t source_terms[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT] =
      {
          [LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE] =
              coordinate_element_index,
      };
  loom_matrix_fragment_apply_coordinate_projection(
      projection->terms, projection->forward_term_count, source_terms,
      out_coordinates);
}

static uint16_t loom_amdgpu_fragment_memory_primary_lane_divisor(
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axes,
    uint8_t view_rank) {
  for (uint8_t i = 0; i < address_layout->lane_term_count; ++i) {
    const loom_amdgpu_fragment_memory_lane_term_t* term =
        &address_layout->lane_terms[i];
    if (term->divisor == 1 && term->modulus > 1) return term->modulus;
  }
  for (uint8_t view_axis = 0; view_axis < view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* axis =
        &runtime_axes[view_axis];
    if (axis->lane_coordinate_scale != 0 && axis->lane_divisor == 1 &&
        axis->lane_modulus > 1) {
      return axis->lane_modulus;
    }
  }
  for (uint8_t i = 0; i < address_layout->lane_term_count; ++i) {
    if (address_layout->lane_terms[i].divisor > 1) {
      return address_layout->lane_terms[i].divisor;
    }
  }
  for (uint8_t view_axis = 0; view_axis < view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* axis =
        &runtime_axes[view_axis];
    if (axis->lane_coordinate_scale != 0 && axis->lane_divisor > 1) {
      return axis->lane_divisor;
    }
  }
  return 1;
}

static uint32_t loom_amdgpu_fragment_memory_linear_lane_byte_stride(
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout) {
  if (address_layout->lane_term_count == 1) {
    const loom_amdgpu_fragment_memory_lane_term_t* term =
        &address_layout->lane_terms[0];
    return term->divisor == 1 && term->modulus == 0 ? term->byte_stride : 0;
  }
  if (address_layout->lane_term_count != 2) return 0;

  const loom_amdgpu_fragment_memory_lane_term_t* low_term = NULL;
  const loom_amdgpu_fragment_memory_lane_term_t* high_term = NULL;
  for (uint8_t i = 0; i < address_layout->lane_term_count; ++i) {
    const loom_amdgpu_fragment_memory_lane_term_t* term =
        &address_layout->lane_terms[i];
    if (term->divisor == 1 && term->modulus > 1) {
      low_term = term;
    } else if (term->divisor > 1 && term->modulus == 0) {
      high_term = term;
    }
  }
  if (low_term == NULL || high_term == NULL ||
      high_term->divisor != low_term->modulus ||
      (uint64_t)low_term->byte_stride * low_term->modulus !=
          high_term->byte_stride) {
    return 0;
  }
  return low_term->byte_stride;
}

bool loom_amdgpu_fragment_memory_compile_address_layout(
    loom_contract_operand_role_t role,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_matrix_result_representation_flags_t representation_flags,
    uint8_t view_rank,
    const loom_low_source_memory_axis_byte_stride_t* axis_byte_strides,
    loom_amdgpu_fragment_memory_address_layout_t* out_address_layout,
    loom_amdgpu_fragment_memory_runtime_axis_t* out_runtime_axes,
    iree_string_view_t* out_constraint_key) {
  *out_address_layout = (loom_amdgpu_fragment_memory_address_layout_t){0};
  memset(out_runtime_axes, 0,
         LOOM_AMDGPU_FRAGMENT_MEMORY_VIEW_RANK_CAPACITY *
             sizeof(out_runtime_axes[0]));
  const uint16_t payload_elements_per_register =
      loom_amdgpu_matrix_fragment_payload_elements_per_register(role_layout);
  const uint16_t payload_registers_per_element =
      role_layout->element_bit_count > 32 &&
              (role_layout->element_bit_count % 32) == 0
          ? role_layout->element_bit_count / 32
          : 1;
  const uint16_t address_payload_elements_per_register =
      payload_elements_per_register != 0 ? payload_elements_per_register : 1;
  IREE_ASSERT_TRUE(payload_elements_per_register != 0 ||
                   payload_registers_per_element > 1);
  IREE_ASSERT_GT(role_layout->coordinate_element_stride, 0);
  IREE_ASSERT_EQ(address_payload_elements_per_register %
                     role_layout->coordinate_element_stride,
                 0);
  IREE_ASSERT_EQ(
      role_layout->payload_element_count * payload_registers_per_element,
      role_layout->register_count * address_payload_elements_per_register);
  const loom_matrix_fragment_coordinate_projection_plan_t* projection =
      role_layout->coordinate_projection_plan;
  IREE_ASSERT_TRUE(projection != NULL);

  out_address_layout->payload_elements_per_register =
      address_payload_elements_per_register;
  out_address_layout->payload_registers_per_element =
      payload_registers_per_element;
  for (uint8_t view_axis = 0; view_axis < view_rank; ++view_axis) {
    const loom_low_source_memory_axis_byte_stride_t* axis_stride =
        &axis_byte_strides[view_axis];
    if (axis_stride->kind == LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC) {
      out_runtime_axes[view_axis].byte_stride = *axis_stride;
    }
  }
  for (uint8_t i = 0; i < projection->forward_term_count; ++i) {
    const loom_matrix_fragment_coordinate_projection_term_t* term =
        &projection->terms[i];
    if (term->source_dimension !=
        LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT) {
      continue;
    }
    const loom_matrix_fragment_axis_t axis =
        loom_matrix_fragment_coordinate_dimension_axis(
            (loom_matrix_fragment_coordinate_dimension_t)
                term->destination_dimension);
    IREE_ASSERT_NE(axis, LOOM_MATRIX_FRAGMENT_AXIS_COUNT);
    const uint8_t view_axis = loom_amdgpu_matrix_fragment_role_view_axis(
        role, view_rank, representation_flags, axis);
    IREE_ASSERT_NE(view_axis, UINT8_MAX);
    const loom_low_source_memory_axis_byte_stride_t* axis_stride =
        &axis_byte_strides[view_axis];
    if (axis_stride->kind == LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC) {
      loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
          &out_runtime_axes[view_axis];
      IREE_ASSERT_EQ(runtime_axis->lane_coordinate_scale, 0u);
      runtime_axis->lane_divisor = term->source_divisor;
      runtime_axis->lane_modulus = term->source_modulus;
      runtime_axis->lane_coordinate_scale = term->destination_multiplier;
      continue;
    }
    uint32_t lane_byte_stride = 0;
    if (!loom_amdgpu_fragment_memory_add_scaled_stride(
            term->destination_multiplier,
            (uint32_t)axis_stride->static_byte_coefficient,
            &lane_byte_stride) ||
        !loom_amdgpu_fragment_memory_append_lane_term(
            term->source_divisor, term->source_modulus, lane_byte_stride,
            out_address_layout)) {
      return loom_amdgpu_fragment_memory_layout_reject(
          IREE_SV("fragment_memory.address_layout"), out_constraint_key);
    }
  }
  out_address_layout->primary_lane_divisor =
      loom_amdgpu_fragment_memory_primary_lane_divisor(
          out_address_layout, out_runtime_axes, view_rank);
  out_address_layout->linear_lane_byte_stride =
      loom_amdgpu_fragment_memory_linear_lane_byte_stride(out_address_layout);

  for (uint16_t register_index = 0;
       register_index < role_layout->register_count; ++register_index) {
    uint32_t coordinates[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT] = {0};
    uint32_t static_byte_offset = 0;
    loom_amdgpu_fragment_memory_register_coordinates(
        role_layout, address_payload_elements_per_register,
        payload_registers_per_element, register_index, &static_byte_offset,
        coordinates);
    for (iree_host_size_t i = 0; i < LOOM_MATRIX_FRAGMENT_AXIS_COUNT; ++i) {
      const loom_matrix_fragment_axis_t axis = (loom_matrix_fragment_axis_t)i;
      const uint8_t view_axis = loom_amdgpu_matrix_fragment_role_view_axis(
          role, view_rank, representation_flags, axis);
      if (view_axis == UINT8_MAX) continue;
      const loom_low_source_memory_axis_byte_stride_t* axis_stride =
          &axis_byte_strides[view_axis];
      const loom_matrix_fragment_coordinate_dimension_t dimension =
          loom_matrix_fragment_axis_coordinate_dimension(axis);
      const uint32_t coordinate = coordinates[dimension];
      if (axis_stride->kind == LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC) {
        if (!loom_amdgpu_fragment_memory_add_scaled_stride(
                coordinate, (uint32_t)axis_stride->static_byte_coefficient,
                &static_byte_offset)) {
          return loom_amdgpu_fragment_memory_layout_reject(
              IREE_SV("fragment_memory.address_layout"), out_constraint_key);
        }
        continue;
      }
      loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
          &out_runtime_axes[view_axis];
      IREE_ASSERT_EQ(runtime_axis->byte_stride.kind,
                     LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC);
      runtime_axis->register_coordinates[register_index] = coordinate;
    }
    out_address_layout->register_byte_offsets[register_index] =
        static_byte_offset;
  }

  if (role_layout->packed_element_axis != LOOM_MATRIX_FRAGMENT_AXIS_COUNT) {
    const uint8_t packed_view_axis = loom_amdgpu_matrix_fragment_role_view_axis(
        role, view_rank, representation_flags,
        role_layout->packed_element_axis);
    IREE_ASSERT_NE(packed_view_axis, UINT8_MAX);
    IREE_ASSERT_LT(packed_view_axis, view_rank);
    const loom_low_source_memory_axis_byte_stride_t* packed_stride =
        &axis_byte_strides[packed_view_axis];
    if (packed_stride->kind == LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC) {
      out_address_layout->packed_element_byte_stride =
          (uint32_t)packed_stride->static_byte_coefficient;
    } else {
      loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
          &out_runtime_axes[packed_view_axis];
      IREE_ASSERT_EQ(runtime_axis->byte_stride.kind,
                     LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC);
      runtime_axis->packed_element_coordinate_stride = 1;
    }
  }
  return true;
}

bool loom_amdgpu_fragment_memory_select_packetization(
    loom_contract_operand_role_t role,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t payload_form,
    uint16_t element_byte_count,
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axes,
    uint8_t view_rank,
    loom_amdgpu_fragment_memory_packetization_t* out_packetization,
    iree_string_view_t* out_constraint_key) {
  *out_packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE;
  if (payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32) {
    return true;
  }
  if (loom_amdgpu_matrix_fragment_role_layout_uses_low_subword(role_layout)) {
    *out_packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_SCALAR_B16;
    return true;
  }
  if (loom_amdgpu_matrix_fragment_role_layout_uses_packed_b16_elements(
          role, role_layout)) {
    *out_packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_PACKED_B16;
    return true;
  }

  const uint32_t packed_element_byte_stride =
      address_layout->packed_element_byte_stride;
  bool has_runtime_packed_element_stride = false;
  for (uint8_t view_axis = 0; view_axis < view_rank; ++view_axis) {
    has_runtime_packed_element_stride |=
        runtime_axes[view_axis].packed_element_coordinate_stride != 0;
  }
  if (!has_runtime_packed_element_stride &&
      (packed_element_byte_stride == 0 ||
       packed_element_byte_stride == element_byte_count)) {
    return true;
  }
  if (payload_form == LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE &&
      element_byte_count == 2 && role_layout->element_bit_count == 16 &&
      address_layout->payload_elements_per_register == 2) {
    *out_packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_PACKED_B16;
    return true;
  }
  return loom_amdgpu_fragment_memory_layout_reject(
      IREE_SV("fragment_memory.packed_axis_stride"), out_constraint_key);
}

bool loom_amdgpu_fragment_memory_source_plan_supports_addressing(
    const loom_low_source_memory_access_plan_t* source,
    iree_string_view_t* out_constraint_key) {
  if (source->static_byte_offset < 0 ||
      source->static_byte_offset > UINT32_MAX) {
    return loom_amdgpu_fragment_memory_layout_reject(
        IREE_SV("fragment_memory.base_offset"), out_constraint_key);
  }
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source->dynamic_terms[i];
    if (term->byte_stride < 0 || term->byte_stride > UINT32_MAX ||
        !loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(term,
                                                                     32)) {
      return loom_amdgpu_fragment_memory_layout_reject(
          IREE_SV("fragment_memory.dynamic_stride"), out_constraint_key);
    }
  }
  return true;
}

bool loom_amdgpu_fragment_memory_address_range_fits_u32(
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axes,
    uint8_t view_rank, uint16_t wave_size, uint16_t register_count,
    iree_string_view_t* out_constraint_key) {
  uint64_t maximum_static_lane_byte_offset = 0;
  for (uint16_t lane = 0; lane < wave_size; ++lane) {
    maximum_static_lane_byte_offset =
        iree_max(maximum_static_lane_byte_offset,
                 loom_amdgpu_fragment_memory_relative_lane_byte_offset(
                     address_layout, (uint8_t)lane));
  }
  uint64_t maximum_static_register_byte_offset = 0;
  for (uint16_t register_index = 0; register_index < register_count;
       ++register_index) {
    maximum_static_register_byte_offset = iree_max(
        maximum_static_register_byte_offset,
        (uint64_t)address_layout->register_byte_offsets[register_index]);
  }
  const uint64_t maximum_static_packed_element_byte_offset =
      (uint64_t)(address_layout->payload_elements_per_register - 1u) *
      address_layout->packed_element_byte_stride;
  const uint64_t maximum_static_fragment_byte_offset =
      maximum_static_lane_byte_offset + maximum_static_register_byte_offset +
      maximum_static_packed_element_byte_offset;
  if (maximum_static_fragment_byte_offset > INT64_MAX) {
    return loom_amdgpu_fragment_memory_layout_reject(
        IREE_SV("fragment_memory.address_range"), out_constraint_key);
  }

  loom_value_facts_t address_facts =
      loom_low_source_memory_dynamic_offset_facts(source,
                                                  source->static_byte_offset);
  const loom_value_facts_t static_fragment_facts =
      loom_value_facts_exact_i64((int64_t)maximum_static_fragment_byte_offset);
  loom_value_facts_addi(&address_facts, &static_fragment_facts, &address_facts);

  for (uint8_t view_axis = 0; view_axis < view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
        &runtime_axes[view_axis];
    uint64_t maximum_lane_coordinate = 0;
    if (runtime_axis->lane_coordinate_scale != 0) {
      const uint64_t maximum_lane_digit =
          runtime_axis->lane_modulus > 1
              ? (uint64_t)runtime_axis->lane_modulus - 1u
              : ((uint64_t)wave_size - 1u) / runtime_axis->lane_divisor;
      maximum_lane_coordinate =
          maximum_lane_digit * runtime_axis->lane_coordinate_scale;
    }
    uint64_t maximum_register_coordinate = 0;
    for (uint16_t register_index = 0; register_index < register_count;
         ++register_index) {
      maximum_register_coordinate = iree_max(
          maximum_register_coordinate,
          (uint64_t)runtime_axis->register_coordinates[register_index]);
    }
    const uint64_t maximum_packed_element_coordinate =
        (uint64_t)(address_layout->payload_elements_per_register - 1u) *
        runtime_axis->packed_element_coordinate_stride;
    const uint64_t maximum_coordinate = maximum_lane_coordinate +
                                        maximum_register_coordinate +
                                        maximum_packed_element_coordinate;
    if (maximum_coordinate == 0) continue;
    if (maximum_coordinate > INT64_MAX) {
      return loom_amdgpu_fragment_memory_layout_reject(
          IREE_SV("fragment_memory.address_range"), out_constraint_key);
    }
    const loom_value_facts_t coordinate_facts =
        loom_value_facts_exact_i64((int64_t)maximum_coordinate);
    loom_value_facts_t axis_byte_facts = loom_value_facts_unknown();
    loom_value_facts_muli(&runtime_axis->byte_stride.byte_facts,
                          &coordinate_facts, &axis_byte_facts);
    loom_value_facts_addi(&address_facts, &axis_byte_facts, &address_facts);
  }

  if (!loom_value_facts_fit_unsigned_bit_count(address_facts, 32)) {
    return loom_amdgpu_fragment_memory_layout_reject(
        IREE_SV("fragment_memory.address_range"), out_constraint_key);
  }
  return true;
}
