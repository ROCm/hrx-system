// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/fragment_memory/transpose.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/matrix_fragment_layout.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef enum loom_amdgpu_fragment_memory_transpose_destination_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_TRANSPOSE_DESTINATION_RDNA4 = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_TRANSPOSE_DESTINATION_CDNA4 = 1,
} loom_amdgpu_fragment_memory_transpose_destination_t;

enum {
  // Largest issuer domain represented by an AMDGPU transpose geometry.
  LOOM_AMDGPU_TRANSPOSE_MAX_ISSUING_LANE_COUNT = 64,
  // Lane-ID bits needed to represent the largest issuer domain.
  LOOM_AMDGPU_TRANSPOSE_LANE_BIT_COUNT = 6,
};

typedef struct loom_amdgpu_fragment_memory_transpose_geometry_t {
  // Low descriptor implementing one physical transpose packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Packet flags describing issuer participation.
  loom_amdgpu_fragment_memory_packet_flags_t packet_flags;
  // Fragment registers produced by one physical packet.
  uint16_t packet_register_count;
  // Lanes that contribute addresses to the physical packet.
  uint8_t issuing_lane_count;
  // Contiguous view elements read by each issuing lane.
  uint8_t loaded_element_count;
  // Hardware lane/register destination mapping.
  loom_amdgpu_fragment_memory_transpose_destination_t destination;
} loom_amdgpu_fragment_memory_transpose_geometry_t;

static bool loom_amdgpu_fragment_memory_transpose_role_supported(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  return plan->operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD &&
         (plan->role == LOOM_CONTRACT_OPERAND_ROLE_LHS ||
          plan->role == LOOM_CONTRACT_OPERAND_ROLE_RHS) &&
         plan->payload_form ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE &&
         plan->dynamic_base_is_subgroup_uniform && plan->view_rank == 2 &&
         layout->tile_shape.block_count == 1 &&
         role_layout->coordinate_element_stride == 1 &&
         role_layout->reduction_group.storage_element_count == 0 &&
         role_layout->reduction_group.logical_element_count == 0 &&
         role_layout->register_count == plan->register_count &&
         plan->payload_register_count == plan->register_count &&
         ((role_layout->element_bit_count == 8 &&
           plan->element_byte_count == 1) ||
          (role_layout->element_bit_count == 16 &&
           plan->element_byte_count == 2));
}

static bool loom_amdgpu_fragment_memory_transpose_static_strides(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  for (uint8_t axis = 0; axis < plan->view_rank; ++axis) {
    if (plan->runtime_axes[axis].byte_stride.kind !=
        LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_UNAVAILABLE) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_rdna4_transpose_geometry(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_fragment_memory_transpose_geometry_t* out_geometry) {
  *out_geometry = (loom_amdgpu_fragment_memory_transpose_geometry_t){0};
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
      plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) {
    return false;
  }
  const bool wave64 = layout->wave_size == 64;
  if (!wave64 && layout->wave_size != 32) {
    return false;
  }
  switch (role_layout->element_bit_count) {
    case 8:
      out_geometry->descriptor_ref =
          wave64 ? LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_TR_B64_SADDR_W64
                 : LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_TR_B64_SADDR;
      out_geometry->packet_register_count = wave64 ? 1 : 2;
      break;
    case 16:
      out_geometry->descriptor_ref =
          wave64 ? LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_TR_B128_SADDR_W64
                 : LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_TR_B128_SADDR;
      out_geometry->packet_register_count = wave64 ? 2 : 4;
      break;
    default:
      return false;
  }
  out_geometry->packet_flags =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_NATIVE_TRANSPOSE_LOAD |
      (wave64 ? LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FIRST_32_LANES_ONLY
              : 0);
  out_geometry->issuing_lane_count = 32;
  out_geometry->loaded_element_count = 8;
  out_geometry->destination =
      LOOM_AMDGPU_FRAGMENT_MEMORY_TRANSPOSE_DESTINATION_RDNA4;
  return true;
}

static bool loom_amdgpu_fragment_memory_cdna4_transpose_geometry(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_fragment_memory_transpose_geometry_t* out_geometry) {
  *out_geometry = (loom_amdgpu_fragment_memory_transpose_geometry_t){0};
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      plan->source.minimum_alignment < 8 || layout->wave_size != 64 ||
      role_layout->element_bit_count != 16) {
    return false;
  }
  out_geometry->descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64_TR_B16;
  out_geometry->packet_flags =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_NATIVE_TRANSPOSE_LOAD;
  out_geometry->packet_register_count = 2;
  out_geometry->issuing_lane_count = 64;
  out_geometry->loaded_element_count = 4;
  out_geometry->destination =
      LOOM_AMDGPU_FRAGMENT_MEMORY_TRANSPOSE_DESTINATION_CDNA4;
  return true;
}

static bool loom_amdgpu_fragment_memory_transpose_geometry(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_fragment_memory_transpose_geometry_t* out_geometry) {
  return loom_amdgpu_fragment_memory_rdna4_transpose_geometry(
             layout, role_layout, plan, out_geometry) ||
         loom_amdgpu_fragment_memory_cdna4_transpose_geometry(
             layout, role_layout, plan, out_geometry);
}

static void loom_amdgpu_fragment_memory_transpose_destination(
    const loom_amdgpu_fragment_memory_transpose_geometry_t* geometry,
    uint16_t wave_size, uint8_t issuing_lane, uint8_t loaded_element,
    uint8_t* out_participant, uint8_t* out_packet_element) {
  if (geometry->destination ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_TRANSPOSE_DESTINATION_CDNA4) {
    *out_participant = (uint8_t)((issuing_lane / 16u) * 16u +
                                 (issuing_lane % 4u) * 4u + loaded_element);
    *out_packet_element = (issuing_lane % 16u) / 4u;
    return;
  }
  // Both wave sizes use the same 32 issuing lanes. Wave64 distributes the
  // upper four elements of each wave32 destination lane to lane + 32.
  *out_participant = (uint8_t)((issuing_lane / 8u) * 8u + loaded_element);
  *out_packet_element = issuing_lane % 8u;
  if (wave_size == 64) {
    *out_participant += (*out_packet_element / 4u) * 32u;
    *out_packet_element %= 4u;
  }
}

static bool loom_amdgpu_fragment_memory_transpose_element_byte_offset(
    const loom_matrix_fragment_role_layout_t* role_layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint8_t participant,
    uint16_t payload_element, uint64_t* out_byte_offset) {
  uint32_t source_coordinates[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT] =
      {
          [LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT] = participant,
          [LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE] = payload_element,
      };
  uint32_t role_coordinates[LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COUNT];
  const loom_matrix_fragment_coordinate_projection_plan_t* projection =
      role_layout->coordinate_projection_plan;
  if (projection == NULL) {
    return false;
  }
  loom_matrix_fragment_apply_coordinate_projection(
      projection->terms, projection->forward_term_count, source_coordinates,
      role_coordinates);

  uint64_t byte_offset = 0;
  for (uint8_t axis = 0; axis < LOOM_MATRIX_FRAGMENT_AXIS_COUNT; ++axis) {
    const uint8_t view_axis = loom_amdgpu_matrix_fragment_role_view_axis(
        plan->role, plan->view_rank, plan->representation_flags,
        (loom_matrix_fragment_axis_t)axis);
    if (view_axis == UINT8_MAX) {
      continue;
    }
    const loom_matrix_fragment_coordinate_dimension_t dimension =
        loom_matrix_fragment_axis_coordinate_dimension(
            (loom_matrix_fragment_axis_t)axis);
    uint64_t contribution = 0;
    if (!iree_checked_mul_u64(role_coordinates[dimension],
                              plan->static_axis_byte_strides[view_axis],
                              &contribution) ||
        !iree_checked_add_u64(byte_offset, contribution, &byte_offset)) {
      return false;
    }
  }
  *out_byte_offset = byte_offset;
  return true;
}

static bool loom_amdgpu_fragment_memory_transpose_packet_offsets(
    const loom_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_transpose_geometry_t* geometry,
    uint16_t register_index, uint64_t* out_lane_offsets) {
  const uint16_t elements_per_register =
      loom_amdgpu_matrix_fragment_payload_elements_per_register(role_layout);
  const uint16_t packet_element_base = register_index * elements_per_register;
  for (uint8_t issuing_lane = 0; issuing_lane < geometry->issuing_lane_count;
       ++issuing_lane) {
    uint64_t lane_offset = 0;
    for (uint8_t loaded_element = 0;
         loaded_element < geometry->loaded_element_count; ++loaded_element) {
      uint8_t participant = 0;
      uint8_t packet_element = 0;
      loom_amdgpu_fragment_memory_transpose_destination(
          geometry, layout->wave_size, issuing_lane, loaded_element,
          &participant, &packet_element);
      uint64_t byte_offset = 0;
      if (!loom_amdgpu_fragment_memory_transpose_element_byte_offset(
              role_layout, plan, participant,
              (uint16_t)(packet_element_base + packet_element), &byte_offset)) {
        return false;
      }
      if (loaded_element == 0) {
        lane_offset = byte_offset;
      } else if (byte_offset !=
                 lane_offset + loaded_element * plan->element_byte_count) {
        return false;
      }
    }
    out_lane_offsets[issuing_lane] = lane_offset;
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_transpose_lane_terms(
    const loom_amdgpu_fragment_memory_transpose_geometry_t* geometry,
    const uint64_t* lane_offsets,
    loom_amdgpu_fragment_memory_address_layout_t* out_address_layout) {
  const uint64_t base_offset = lane_offsets[0];
  uint32_t bit_byte_strides[LOOM_AMDGPU_TRANSPOSE_LANE_BIT_COUNT] = {0};
  uint8_t lane_bit_count = 0;
  for (uint8_t lane_bit = 0;
       (UINT8_C(1) << lane_bit) < geometry->issuing_lane_count; ++lane_bit) {
    const uint8_t divisor = UINT8_C(1) << lane_bit;
    const uint64_t probe_offset = lane_offsets[divisor];
    if (probe_offset < base_offset || probe_offset - base_offset > UINT32_MAX) {
      return false;
    }
    bit_byte_strides[lane_bit] = (uint32_t)(probe_offset - base_offset);
    lane_bit_count = lane_bit + 1u;
  }

  for (uint8_t lane = 0; lane < geometry->issuing_lane_count; ++lane) {
    uint64_t expected_offset = base_offset;
    for (uint8_t lane_bit = 0; lane_bit < lane_bit_count; ++lane_bit) {
      if ((lane & (UINT8_C(1) << lane_bit)) != 0) {
        expected_offset += bit_byte_strides[lane_bit];
      }
    }
    if (expected_offset != lane_offsets[lane]) {
      return false;
    }
  }

  if (bit_byte_strides[0] != 0) {
    bool linear = true;
    for (uint8_t lane_bit = 1; lane_bit < lane_bit_count; ++lane_bit) {
      linear &= (uint64_t)bit_byte_strides[lane_bit] ==
                (uint64_t)bit_byte_strides[0] * (UINT64_C(1) << lane_bit);
    }
    if (linear) {
      out_address_layout->linear_lane_byte_stride = bit_byte_strides[0];
      out_address_layout->lane_address_cost = 1;
      out_address_layout->primary_lane_divisor = geometry->issuing_lane_count;
      return true;
    }
  }

  for (uint8_t lane_bit = 0; lane_bit < lane_bit_count;) {
    const uint32_t byte_stride = bit_byte_strides[lane_bit];
    if (byte_stride == 0) {
      ++lane_bit;
      continue;
    }
    uint8_t run_bit_count = 1;
    while (lane_bit + run_bit_count < lane_bit_count &&
           (uint64_t)bit_byte_strides[lane_bit + run_bit_count] ==
               (uint64_t)byte_stride * (UINT64_C(1) << run_bit_count)) {
      ++run_bit_count;
    }
    if (out_address_layout->lane_term_count ==
        IREE_ARRAYSIZE(out_address_layout->lane_terms)) {
      return false;
    }
    loom_amdgpu_fragment_memory_lane_term_t* term =
        &out_address_layout->lane_terms[out_address_layout->lane_term_count++];
    *term = (loom_amdgpu_fragment_memory_lane_term_t){
        .divisor = UINT16_C(1) << lane_bit,
        .modulus = lane_bit + run_bit_count == lane_bit_count
                       ? 0
                       : UINT16_C(1) << run_bit_count,
        .byte_stride = byte_stride,
    };
    out_address_layout->lane_address_cost +=
        (uint8_t)(2u + (term->divisor > 1 ? 1u : 0u) +
                  (term->modulus > 1 ? 1u : 0u));
    lane_bit += run_bit_count;
  }
  if (out_address_layout->lane_term_count == 0) {
    out_address_layout->primary_lane_divisor = 1;
    return true;
  }
  const loom_amdgpu_fragment_memory_lane_term_t* first_term =
      &out_address_layout->lane_terms[0];
  out_address_layout->primary_lane_divisor =
      first_term->divisor == 1 && first_term->modulus > 1 ? first_term->modulus
                                                          : first_term->divisor;
  return true;
}

bool loom_amdgpu_fragment_memory_select_transpose_load(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_fragment_memory_transpose_choice_t* out_choice) {
  *out_choice = (loom_amdgpu_fragment_memory_transpose_choice_t){0};
  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, plan->role);
  if (role_layout == NULL ||
      !loom_amdgpu_fragment_memory_transpose_role_supported(layout, role_layout,
                                                            plan) ||
      !loom_amdgpu_fragment_memory_transpose_static_strides(plan)) {
    return false;
  }

  loom_amdgpu_fragment_memory_transpose_geometry_t geometry;
  if (!loom_amdgpu_fragment_memory_transpose_geometry(layout, role_layout, plan,
                                                      &geometry) ||
      role_layout->register_count % geometry.packet_register_count != 0 ||
      !loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                          geometry.descriptor_ref)) {
    return false;
  }

  loom_amdgpu_fragment_memory_address_layout_t address_layout = {
      .packed_element_byte_stride = plan->element_byte_count,
      .payload_elements_per_register =
          loom_amdgpu_matrix_fragment_payload_elements_per_register(
              role_layout),
      .payload_registers_per_element = 1,
  };
  uint64_t canonical_lane_offsets[LOOM_AMDGPU_TRANSPOSE_MAX_ISSUING_LANE_COUNT];
  if (!loom_amdgpu_fragment_memory_transpose_packet_offsets(
          layout, role_layout, plan, &geometry, /*register_index=*/0,
          canonical_lane_offsets) ||
      !loom_amdgpu_fragment_memory_transpose_lane_terms(
          &geometry, canonical_lane_offsets, &address_layout)) {
    return false;
  }

  for (uint16_t register_index = 0;
       register_index < role_layout->register_count;
       register_index += geometry.packet_register_count) {
    uint64_t lane_offsets[LOOM_AMDGPU_TRANSPOSE_MAX_ISSUING_LANE_COUNT];
    if (!loom_amdgpu_fragment_memory_transpose_packet_offsets(
            layout, role_layout, plan, &geometry, register_index,
            lane_offsets) ||
        lane_offsets[0] < canonical_lane_offsets[0] ||
        lane_offsets[0] - canonical_lane_offsets[0] > UINT32_MAX ||
        lane_offsets[0] > UINT32_MAX) {
      return false;
    }
    const uint32_t packet_byte_offset =
        (uint32_t)(lane_offsets[0] - canonical_lane_offsets[0]);
    for (uint8_t lane = 0; lane < geometry.issuing_lane_count; ++lane) {
      if (lane_offsets[lane] !=
          canonical_lane_offsets[lane] + packet_byte_offset) {
        return false;
      }
    }
    for (uint16_t i = 0; i < geometry.packet_register_count; ++i) {
      address_layout.register_byte_offsets[register_index + i] =
          (uint32_t)(canonical_lane_offsets[0] + packet_byte_offset);
    }
    out_choice->packets[out_choice->packet_count++] =
        (loom_amdgpu_fragment_memory_packet_plan_t){
            .flags = geometry.packet_flags,
            .issued_element_count = geometry.loaded_element_count,
            .descriptor_ref = geometry.descriptor_ref,
            .register_index = register_index,
            .result_register_count = geometry.packet_register_count,
            .packet_register_count = geometry.packet_register_count,
        };
  }

  const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
      descriptor_set, geometry.descriptor_ref);
  out_choice->address_layout = address_layout;
  out_choice->packet_address_sources_consumed_at_issue = !iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_ADDRESS_SOURCE_RETAINED);
  return true;
}

void loom_amdgpu_fragment_memory_apply_transpose_load(
    const loom_amdgpu_fragment_memory_transpose_choice_t* choice,
    loom_amdgpu_fragment_memory_plan_t* plan) {
  plan->address_layout = choice->address_layout;
  memset(plan->runtime_axes, 0, sizeof(plan->runtime_axes));
  memcpy(plan->packets, choice->packets,
         choice->packet_count * sizeof(choice->packets[0]));
  plan->packet_count = choice->packet_count;
  plan->packet_address_sources_consumed_at_issue =
      choice->packet_address_sources_consumed_at_issue;
  plan->packetization = LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE;
  plan->packed_b16_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  plan->epilogue_strategy = LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_NONE;
  plan->packed_b16_publication = NULL;
}
