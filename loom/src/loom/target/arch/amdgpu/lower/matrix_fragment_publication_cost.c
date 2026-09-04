// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_publication_cost.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ops/vector/fragment.h"
#include "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory_subgroup_access.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"

// Candidate-differential cost of the descriptor recipe emitted for one
// publication. Setup shared by every candidate is intentionally omitted.
// Runtime accumulates generated schedule-resource occupancy. Code size counts
// the minimum encoded bytes contributed by descriptor packets; exact final
// bytes remain a native-encoding property and are measured by the binary-size
// gate rather than pulling the complete encoding tables into source lowering.
static uint32_t loom_amdgpu_fragment_publication_saturating_multiply_u32(
    uint32_t lhs, uint32_t rhs) {
  const uint64_t product = (uint64_t)lhs * rhs;
  return product > UINT32_MAX ? UINT32_MAX : (uint32_t)product;
}

static const loom_low_schedule_class_t*
loom_amdgpu_fragment_publication_schedule_class(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  if (descriptor == NULL) return NULL;
  const loom_low_descriptor_view_t* descriptor_view =
      loom_low_descriptor_set_descriptor_view(descriptor_set, descriptor);
  return &descriptor_set->schedule_classes[descriptor_view->schedule_class_id];
}

static bool loom_amdgpu_fragment_publication_add_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t count,
    loom_low_representation_cost_t* inout_cost) {
  const loom_low_schedule_class_t* schedule_class =
      loom_amdgpu_fragment_publication_schedule_class(descriptor_set,
                                                      descriptor_ref);
  if (schedule_class == NULL) return false;
  inout_cost->runtime = iree_math_saturating_add_u32(
      inout_cost->runtime,
      loom_amdgpu_fragment_publication_saturating_multiply_u32(
          schedule_class->minimum_issue_cycles, count));
  inout_cost->code_size = iree_math_saturating_add_u32(
      inout_cost->code_size,
      loom_amdgpu_fragment_publication_saturating_multiply_u32(4u, count));
  return true;
}

static uint16_t loom_amdgpu_fragment_publication_schedule_distance(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_schedule_class_t* schedule_class =
      loom_amdgpu_fragment_publication_schedule_class(descriptor_set,
                                                      descriptor_ref);
  return loom_low_schedule_class_schedule_distance_cycles(schedule_class);
}

static bool loom_amdgpu_fragment_publication_add_vgpr_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t immediate,
    uint32_t count, loom_low_representation_cost_t* inout_cost) {
  const loom_amdgpu_descriptor_ref_t selected_descriptor_ref =
      loom_amdgpu_select_vgpr_binary_immediate_descriptor_ref(
          descriptor_set, descriptor_ref, immediate);
  return selected_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
         loom_amdgpu_fragment_publication_add_descriptor(
             descriptor_set, selected_descriptor_ref, count, inout_cost);
}

static bool loom_amdgpu_fragment_publication_add_compare_immediate(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t immediate,
    loom_low_representation_cost_t* inout_cost,
    loom_amdgpu_descriptor_ref_t* out_compare_descriptor_ref) {
  const loom_amdgpu_descriptor_ref_t inline_descriptor_ref =
      kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
          [LOOM_VECTOR_CMPI_PREDICATE_EQ]
              .src1_inline_descriptor_ref;
  if (immediate <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         inline_descriptor_ref)) {
    *out_compare_descriptor_ref = inline_descriptor_ref;
    return loom_amdgpu_fragment_publication_add_descriptor(
        descriptor_set, inline_descriptor_ref, 1, inout_cost);
  }
  *out_compare_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32;
  return loom_amdgpu_fragment_publication_add_descriptor(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 1,
             inout_cost) &&
         loom_amdgpu_fragment_publication_add_descriptor(
             descriptor_set, *out_compare_descriptor_ref, 1, inout_cost);
}

static bool loom_amdgpu_fragment_publication_add_pack_u16(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t count,
    loom_low_representation_cost_t* inout_cost) {
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32)) {
    return loom_amdgpu_fragment_publication_add_descriptor(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32, count,
        inout_cost);
  }
  return loom_amdgpu_fragment_publication_add_vgpr_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
             UINT16_MAX, count, inout_cost) &&
         loom_amdgpu_fragment_publication_add_vgpr_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
             count, inout_cost) &&
         loom_amdgpu_fragment_publication_add_descriptor(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, count,
             inout_cost);
}

static bool loom_amdgpu_fragment_publication_add_bf16_lane(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t count,
    loom_low_representation_cost_t* inout_cost) {
  if (!loom_amdgpu_fragment_publication_add_vgpr_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
          count, inout_cost) ||
      !loom_amdgpu_fragment_publication_add_vgpr_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, 1, count,
          inout_cost)) {
    return false;
  }
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT)) {
    if (!loom_amdgpu_fragment_publication_add_descriptor(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT,
            count, inout_cost)) {
      return false;
    }
  } else if (!loom_amdgpu_fragment_publication_add_vgpr_immediate(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
                 UINT32_C(0x7FFF), count, inout_cost) ||
             !loom_amdgpu_fragment_publication_add_descriptor(
                 descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, count,
                 inout_cost)) {
    return false;
  }
  return loom_amdgpu_fragment_publication_add_vgpr_immediate(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16, count,
      inout_cost);
}

static bool loom_amdgpu_fragment_publication_add_lane_conversion(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    uint16_t register_index, loom_low_representation_cost_t* inout_cost) {
  const loom_low_descriptor_set_t* descriptor_set = query->descriptor_set;
  if (iree_any_bit_set(query->source_flags,
                       LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED)) {
    return (register_index & 1u) == 0 ||
           loom_amdgpu_fragment_publication_add_vgpr_immediate(
               descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
               1, inout_cost);
  }
  if (iree_any_bit_set(query->source_flags,
                       LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_SCALED) &&
      !loom_amdgpu_fragment_publication_add_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, 1,
          inout_cost)) {
    return false;
  }
  if (query->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16) {
    return loom_amdgpu_fragment_publication_add_descriptor(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32, 1,
        inout_cost);
  }
  return loom_amdgpu_fragment_publication_add_bf16_lane(descriptor_set, 1,
                                                        inout_cost);
}

static bool loom_amdgpu_fragment_publication_add_pair_conversion(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    uint32_t pair_count, loom_low_representation_cost_t* inout_cost) {
  if (iree_any_bit_set(query->source_flags,
                       LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED)) {
    return true;
  }
  const loom_low_descriptor_set_t* descriptor_set = query->descriptor_set;
  if (iree_any_bit_set(query->source_flags,
                       LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_SCALED) &&
      !loom_amdgpu_fragment_publication_add_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, 2u * pair_count,
          inout_cost)) {
    return false;
  }
  if (query->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16) {
    if (loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32)) {
      return loom_amdgpu_fragment_publication_add_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32,
          pair_count, inout_cost);
    }
    return loom_amdgpu_fragment_publication_add_bf16_lane(
               descriptor_set, 2u * pair_count, inout_cost) &&
           loom_amdgpu_fragment_publication_add_pack_u16(
               descriptor_set, pair_count, inout_cost);
  }
  return loom_amdgpu_fragment_publication_add_descriptor(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32,
             2u * pair_count, inout_cost) &&
         (loom_amdgpu_descriptor_set_has_ref(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PACK_B32_F16)
              ? loom_amdgpu_fragment_publication_add_descriptor(
                    descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PACK_B32_F16,
                    pair_count, inout_cost)
              : loom_amdgpu_fragment_publication_add_pack_u16(
                    descriptor_set, pair_count, inout_cost));
}

static uint32_t loom_amdgpu_fragment_publication_address_cost(
    const loom_amdgpu_fragment_memory_publication_query_t* query) {
  const loom_amdgpu_fragment_memory_address_layout_t* address_layout =
      query->address_layout;
  if (address_layout->linear_lane_byte_stride != 0) return 1;
  uint32_t cost = 0;
  for (uint8_t i = 0; i < address_layout->lane_term_count; ++i) {
    const loom_amdgpu_fragment_memory_lane_term_t* term =
        &address_layout->lane_terms[i];
    cost += 2;
    cost += term->divisor > 1 ? 1 : 0;
    cost += term->modulus > 1 ? 1 : 0;
  }
  for (uint8_t view_axis = 0; view_axis < query->view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
        &query->runtime_axes[view_axis];
    cost += runtime_axis->lane_coordinate_scale != 0 ? 2 : 0;
  }
  return cost;
}

static uint64_t loom_amdgpu_fragment_publication_active_lane_mask(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    const loom_matrix_fragment_packed_b16_publication_t* publication) {
  const uint8_t wave_size = (uint8_t)query->layout->wave_size;
  if (publication == NULL) {
    return wave_size == 64 ? UINT64_MAX : (UINT64_C(1) << wave_size) - 1u;
  }
  uint64_t active_lane_mask = 0;
  for (uint8_t lane = 0; lane < wave_size; ++lane) {
    if ((lane & publication->publishing_participant_and_mask) ==
        publication->publishing_participant_equal_value) {
      active_lane_mask |= UINT64_C(1) << lane;
    }
  }
  return active_lane_mask;
}

static uint32_t loom_amdgpu_fragment_publication_region_count(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    uint32_t per_lane_packet_byte_count,
    const loom_matrix_fragment_packed_b16_publication_t* publication) {
  const uint64_t active_lane_mask =
      loom_amdgpu_fragment_publication_active_lane_mask(query, publication);
  for (uint8_t view_axis = 0; view_axis < query->view_rank; ++view_axis) {
    if (query->runtime_axes[view_axis].lane_coordinate_scale == 0) continue;
    return (uint32_t)iree_math_count_ones_u64(active_lane_mask);
  }
  loom_low_lower_memory_subgroup_access_report_t geometry = {0};
  loom_amdgpu_memory_calculate_subgroup_geometry(
      query->address_layout, (uint8_t)query->layout->wave_size,
      active_lane_mask, per_lane_packet_byte_count, &geometry);
  return geometry.contiguous_region_count;
}

static loom_low_representation_cost_t
loom_amdgpu_fragment_publication_finalize_cost(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    loom_low_representation_cost_t recipe_cost, uint32_t memory_region_count,
    uint32_t dependency_distance) {
  const uint32_t address_cost =
      loom_amdgpu_fragment_publication_address_cost(query);
  const uint32_t dependency_gap =
      dependency_distance > 1 ? dependency_distance - 1u : 0;
  return (loom_low_representation_cost_t){
      .runtime = iree_math_saturating_add_u32(
          iree_math_saturating_add_u32(recipe_cost.runtime, address_cost),
          iree_math_saturating_add_u32(memory_region_count, dependency_gap)),
      .code_size = iree_math_saturating_add_u32(
          recipe_cost.code_size,
          loom_amdgpu_fragment_publication_saturating_multiply_u32(address_cost,
                                                                   4u)),
  };
}

bool loom_amdgpu_fragment_publication_cost_direct(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    const loom_amdgpu_fragment_memory_packet_plan_t* packets,
    uint16_t packet_count, loom_low_representation_cost_t* out_cost) {
  *out_cost = (loom_low_representation_cost_t){0};
  loom_low_representation_cost_t recipe_cost = {0};
  uint32_t memory_region_count = 0;
  for (uint16_t i = 0; i < packet_count; ++i) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet = &packets[i];
    const bool conversion_available =
        packet->result_register_count == 1
            ? loom_amdgpu_fragment_publication_add_lane_conversion(
                  query, packet->register_index, &recipe_cost)
            : loom_amdgpu_fragment_publication_add_pair_conversion(
                  query, packet->result_register_count / 2u, &recipe_cost);
    if (!conversion_available ||
        !loom_amdgpu_fragment_publication_add_descriptor(
            query->descriptor_set, packet->descriptor_ref, 1, &recipe_cost)) {
      return false;
    }
    memory_region_count = iree_math_saturating_add_u32(
        memory_region_count,
        loom_amdgpu_fragment_publication_region_count(
            query, packet->result_register_count * query->element_byte_count,
            /*publication=*/NULL));
  }
  *out_cost = loom_amdgpu_fragment_publication_finalize_cost(
      query, recipe_cost, memory_region_count, /*dependency_distance=*/0);
  return true;
}

bool loom_amdgpu_fragment_publication_cost_crosslane(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    const loom_matrix_fragment_packed_b16_publication_t* publication,
    loom_amdgpu_fragment_memory_packet_flags_t packet_flags,
    loom_amdgpu_descriptor_ref_t store_descriptor_ref,
    loom_low_representation_cost_t* out_cost) {
  *out_cost = (loom_low_representation_cost_t){0};
  const bool uses_dpp = iree_any_bit_set(
      packet_flags,
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE_DPP);
  const loom_amdgpu_descriptor_ref_t exchange_descriptor_ref =
      uses_dpp ? loom_amdgpu_select_dpp_descriptor_ref(query->descriptor_set)
               : LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32;
  loom_low_representation_cost_t recipe_cost = {0};
  loom_amdgpu_descriptor_ref_t compare_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  bool available =
      loom_amdgpu_fragment_publication_add_vgpr_immediate(
          query->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          publication->publishing_participant_and_mask, 1, &recipe_cost) &&
      loom_amdgpu_fragment_publication_add_compare_immediate(
          query->descriptor_set,
          publication->publishing_participant_equal_value, &recipe_cost,
          &compare_descriptor_ref) &&
      loom_amdgpu_fragment_publication_add_descriptor(
          query->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
          1, &recipe_cost) &&
      loom_amdgpu_fragment_publication_add_descriptor(
          query->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC, 1,
          &recipe_cost);
  if (available && !uses_dpp) {
    available =
        loom_amdgpu_fragment_publication_add_vgpr_immediate(
            query->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
            publication->paired_participant_xor_mask, 1, &recipe_cost) &&
        loom_amdgpu_fragment_publication_add_vgpr_immediate(
            query->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
            2, 1, &recipe_cost);
  }
  const bool pre_narrow_bf16 =
      query->payload_form ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16 &&
      iree_any_bit_set(query->source_flags,
                       LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_ROUNDED) &&
      !loom_amdgpu_descriptor_set_has_ref(
          query->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32);
  for (uint16_t register_index = 0;
       available && register_index < query->register_count; ++register_index) {
    if (iree_any_bit_set(query->source_flags,
                         LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED) &&
        (register_index & 1u) != 0) {
      available = loom_amdgpu_fragment_publication_add_vgpr_immediate(
          query->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          16, 1, &recipe_cost);
    } else if (pre_narrow_bf16) {
      if (iree_any_bit_set(
              query->source_flags,
              LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_SCALED)) {
        available = loom_amdgpu_fragment_publication_add_descriptor(
            query->descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, 1,
            &recipe_cost);
      }
      if (available) {
        available = loom_amdgpu_fragment_publication_add_bf16_lane(
            query->descriptor_set, 1, &recipe_cost);
      }
    }
    if (available) {
      available = loom_amdgpu_fragment_publication_add_descriptor(
          query->descriptor_set, exchange_descriptor_ref, 1, &recipe_cost);
    }
    if (available &&
        (iree_any_bit_set(
             query->source_flags,
             LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED) ||
         pre_narrow_bf16)) {
      available = loom_amdgpu_fragment_publication_add_pack_u16(
          query->descriptor_set, 1, &recipe_cost);
    } else if (available) {
      available = loom_amdgpu_fragment_publication_add_pair_conversion(
          query, 1, &recipe_cost);
    }
    if (available) {
      available = loom_amdgpu_fragment_publication_add_descriptor(
          query->descriptor_set, store_descriptor_ref, 1, &recipe_cost);
    }
  }
  if (!available) return false;

  const uint32_t compare_distance =
      loom_amdgpu_fragment_publication_schedule_distance(
          query->descriptor_set, compare_descriptor_ref);
  const uint32_t exchange_distance =
      loom_amdgpu_fragment_publication_schedule_distance(
          query->descriptor_set, exchange_descriptor_ref);
  const uint32_t dependency_distance = compare_distance > exchange_distance
                                           ? compare_distance
                                           : exchange_distance;
  const uint32_t memory_region_count =
      loom_amdgpu_fragment_publication_saturating_multiply_u32(
          loom_amdgpu_fragment_publication_region_count(
              query,
              LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT *
                  query->element_byte_count,
              publication),
          query->register_count);
  *out_cost = loom_amdgpu_fragment_publication_finalize_cost(
      query, recipe_cost, memory_region_count, dependency_distance);
  return true;
}
