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
IREE_ATTRIBUTE_NOINLINE static const loom_low_schedule_class_t*
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

IREE_ATTRIBUTE_NOINLINE static bool
loom_amdgpu_fragment_publication_add_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t count,
    loom_low_representation_cost_t* inout_cost) {
  if (count == 0) return true;
  const loom_low_schedule_class_t* schedule_class =
      loom_amdgpu_fragment_publication_schedule_class(descriptor_set,
                                                      descriptor_ref);
  if (schedule_class == NULL) return false;
  // Publication queries contain at most 32 registers. Two-lane conversions
  // therefore contribute at most 64 instances of a 16-bit issue bound, so the
  // complete fixed recipe cannot overflow either 32-bit cost component.
  inout_cost->runtime += schedule_class->minimum_issue_cycles * count;
  inout_cost->code_size += 4u * count;
  return true;
}

IREE_ATTRIBUTE_NOINLINE static uint16_t
loom_amdgpu_fragment_publication_schedule_distance(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_schedule_class_t* schedule_class =
      loom_amdgpu_fragment_publication_schedule_class(descriptor_set,
                                                      descriptor_ref);
  return loom_low_schedule_class_schedule_distance_cycles(schedule_class);
}

IREE_ATTRIBUTE_NOINLINE static bool
loom_amdgpu_fragment_publication_add_vgpr_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t immediate,
    uint32_t count, loom_low_representation_cost_t* inout_cost) {
  if (count == 0) return true;
  const loom_amdgpu_descriptor_ref_t selected_descriptor_ref =
      loom_amdgpu_select_vgpr_binary_immediate_descriptor_ref(
          descriptor_set, descriptor_ref, immediate);
  return selected_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
         loom_amdgpu_fragment_publication_add_descriptor(
             descriptor_set, selected_descriptor_ref, count, inout_cost);
}

IREE_ATTRIBUTE_NOINLINE static bool
loom_amdgpu_fragment_publication_add_compare_immediate(
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

IREE_ATTRIBUTE_NOINLINE static bool
loom_amdgpu_fragment_publication_add_conversions(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    uint32_t scalar_lane_count, uint32_t packed_shift_count,
    uint32_t pair_count, uint32_t low_u16_pair_count,
    loom_low_representation_cost_t* inout_cost) {
  const loom_low_descriptor_set_t* descriptor_set = query->descriptor_set;
  if (!loom_amdgpu_fragment_publication_add_vgpr_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
          packed_shift_count, inout_cost)) {
    return false;
  }

  const uint32_t converted_lane_count = scalar_lane_count + 2u * pair_count;
  if (iree_any_bit_set(query->source_flags,
                       LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_SCALED) &&
      !loom_amdgpu_fragment_publication_add_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
          converted_lane_count, inout_cost)) {
    return false;
  }

  uint32_t bf16_lane_count = 0;
  uint32_t pack_u16_count = low_u16_pair_count;
  if (query->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16) {
    if (!loom_amdgpu_fragment_publication_add_descriptor(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32,
            converted_lane_count, inout_cost)) {
      return false;
    }
    if (pair_count != 0) {
      if (loom_amdgpu_descriptor_set_has_ref(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PACK_B32_F16)) {
        if (!loom_amdgpu_fragment_publication_add_descriptor(
                descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PACK_B32_F16,
                pair_count, inout_cost)) {
          return false;
        }
      } else {
        pack_u16_count += pair_count;
      }
    }
  } else {
    bf16_lane_count = scalar_lane_count;
    if (loom_amdgpu_descriptor_set_has_ref(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32)) {
      if (!loom_amdgpu_fragment_publication_add_descriptor(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32,
              pair_count, inout_cost)) {
        return false;
      }
    } else {
      bf16_lane_count += 2u * pair_count;
      pack_u16_count += pair_count;
    }
  }

  if (bf16_lane_count != 0 &&
      (!loom_amdgpu_fragment_publication_add_vgpr_immediate(
           descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
           bf16_lane_count, inout_cost) ||
       !loom_amdgpu_fragment_publication_add_vgpr_immediate(
           descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, 1,
           bf16_lane_count, inout_cost))) {
    return false;
  }
  if (bf16_lane_count != 0 &&
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT)) {
    if (!loom_amdgpu_fragment_publication_add_descriptor(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT,
            bf16_lane_count, inout_cost)) {
      return false;
    }
  } else if (bf16_lane_count != 0 &&
             (!loom_amdgpu_fragment_publication_add_vgpr_immediate(
                  descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
                  UINT32_C(0x7FFF), bf16_lane_count, inout_cost) ||
              !loom_amdgpu_fragment_publication_add_descriptor(
                  descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
                  bf16_lane_count, inout_cost))) {
    return false;
  }
  if (bf16_lane_count != 0 &&
      !loom_amdgpu_fragment_publication_add_vgpr_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
          bf16_lane_count, inout_cost)) {
    return false;
  }

  if (pack_u16_count == 0) return true;
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32)) {
    return loom_amdgpu_fragment_publication_add_descriptor(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32,
        pack_u16_count, inout_cost);
  }
  return loom_amdgpu_fragment_publication_add_vgpr_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
             UINT16_MAX, pack_u16_count, inout_cost) &&
         loom_amdgpu_fragment_publication_add_vgpr_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
             pack_u16_count, inout_cost) &&
         loom_amdgpu_fragment_publication_add_descriptor(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
             pack_u16_count, inout_cost);
}

static const uint64_t kLoomAmdgpuLaneIndexBitMasks[] = {
    UINT64_C(0xAAAAAAAAAAAAAAAA), UINT64_C(0xCCCCCCCCCCCCCCCC),
    UINT64_C(0xF0F0F0F0F0F0F0F0), UINT64_C(0xFF00FF00FF00FF00),
    UINT64_C(0xFFFF0000FFFF0000), UINT64_C(0xFFFFFFFF00000000),
};

static uint64_t loom_amdgpu_fragment_publication_active_lane_mask(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    const loom_matrix_fragment_packed_b16_publication_t* publication) {
  const uint8_t wave_size = (uint8_t)query->layout->wave_size;
  if (publication == NULL) {
    return wave_size == 64 ? UINT64_MAX : (UINT64_C(1) << wave_size) - 1u;
  }
  uint32_t participant_bits = publication->publishing_participant_and_mask;
  uint64_t active_lane_mask = UINT64_MAX;
  while (participant_bits != 0) {
    const uint32_t participant_bit =
        iree_math_count_trailing_zeros_u32(participant_bits);
    IREE_ASSERT_LT(participant_bit,
                   IREE_ARRAYSIZE(kLoomAmdgpuLaneIndexBitMasks));
    const uint64_t participant_mask =
        kLoomAmdgpuLaneIndexBitMasks[participant_bit];
    active_lane_mask &= (publication->publishing_participant_equal_value &
                         (UINT32_C(1) << participant_bit)) != 0
                            ? participant_mask
                            : ~participant_mask;
    participant_bits &= participant_bits - 1u;
  }
  const uint64_t wave_mask =
      wave_size == 64 ? UINT64_MAX : (UINT64_C(1) << wave_size) - 1u;
  return active_lane_mask & wave_mask;
}

IREE_ATTRIBUTE_NOINLINE static uint32_t
loom_amdgpu_fragment_publication_region_count(
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
  const uint32_t address_cost = query->address_layout->lane_address_cost;
  const uint32_t dependency_gap =
      dependency_distance > 1 ? dependency_distance - 1u : 0;
  return (loom_low_representation_cost_t){
      .runtime = recipe_cost.runtime + address_cost + memory_region_count +
                 dependency_gap,
      .code_size = recipe_cost.code_size + address_cost * 4u,
  };
}

bool loom_amdgpu_fragment_publication_cost_direct(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    const loom_amdgpu_fragment_memory_publication_packet_t* packets,
    uint16_t packet_count, loom_low_representation_cost_t* out_cost) {
  loom_low_representation_cost_t recipe_cost = {0};
  const bool source_is_packed = iree_any_bit_set(
      query->source_flags, LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED);
  // Packet widths sum to the result register count, so equality means every
  // packet is scalar and the repeated entry can be costed as one batch.
  if (packet_count == query->register_count) {
    if (!loom_amdgpu_fragment_publication_add_descriptor(
            query->descriptor_set,
            loom_amdgpu_fragment_memory_publication_packet_descriptor_ref(
                packets[0]),
            packet_count, &recipe_cost) ||
        !loom_amdgpu_fragment_publication_add_conversions(
            query, source_is_packed ? 0 : packet_count,
            source_is_packed ? packet_count / 2u : 0, /*pair_count=*/0,
            /*low_u16_pair_count=*/0, &recipe_cost)) {
      return false;
    }
    const uint32_t memory_region_count =
        loom_amdgpu_fragment_publication_region_count(
            query, query->element_byte_count, /*publication=*/NULL) *
        packet_count;
    *out_cost = loom_amdgpu_fragment_publication_finalize_cost(
        query, recipe_cost, memory_region_count, /*dependency_distance=*/0);
    return true;
  }

  uint32_t
      region_counts[2u * LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS + 1u];
  uint8_t packet_counts[2u * LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS +
                        1u] = {0};
  loom_amdgpu_descriptor_ref_t
      descriptor_refs[2u * LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS +
                      1u];
  // Narrowed-store descriptor selection depends only on memory space and
  // result width, so every packet in one width bucket shares a descriptor.
  uint32_t scalar_lane_count = 0;
  uint32_t packed_shift_count = 0;
  uint32_t pair_count = 0;
  uint16_t register_index = 0;
  for (uint16_t i = 0; i < packet_count; ++i) {
    const loom_amdgpu_fragment_memory_publication_packet_t packet = packets[i];
    const uint16_t result_register_count =
        loom_amdgpu_fragment_memory_publication_packet_result_register_count(
            packet);
    if (packet_counts[result_register_count] == 0) {
      descriptor_refs[result_register_count] =
          loom_amdgpu_fragment_memory_publication_packet_descriptor_ref(packet);
      region_counts[result_register_count] =
          loom_amdgpu_fragment_publication_region_count(
              query, result_register_count * query->element_byte_count,
              /*publication=*/NULL);
    }
    ++packet_counts[result_register_count];
    if (result_register_count == 1) {
      if (source_is_packed) {
        packed_shift_count += register_index & 1u;
      } else {
        ++scalar_lane_count;
      }
    } else if (!source_is_packed) {
      pair_count += result_register_count / 2u;
    }
    register_index += result_register_count;
  }
  uint32_t memory_region_count = 0;
  for (uint16_t result_register_count = 1;
       result_register_count < IREE_ARRAYSIZE(packet_counts);
       ++result_register_count) {
    if (packet_counts[result_register_count] == 0) continue;
    if (!loom_amdgpu_fragment_publication_add_descriptor(
            query->descriptor_set, descriptor_refs[result_register_count],
            packet_counts[result_register_count], &recipe_cost)) {
      return false;
    }
    memory_region_count += region_counts[result_register_count] *
                           packet_counts[result_register_count];
  }
  if (!loom_amdgpu_fragment_publication_add_conversions(
          query, scalar_lane_count, packed_shift_count, pair_count,
          /*low_u16_pair_count=*/0, &recipe_cost)) {
    return false;
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
  const bool source_is_packed = iree_any_bit_set(
      query->source_flags, LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED);
  if (available) {
    available = loom_amdgpu_fragment_publication_add_conversions(
        query, pre_narrow_bf16 ? query->register_count : 0,
        source_is_packed ? query->register_count / 2u : 0,
        !source_is_packed && !pre_narrow_bf16 ? query->register_count : 0,
        source_is_packed || pre_narrow_bf16 ? query->register_count : 0,
        &recipe_cost);
  }
  if (available) {
    available = loom_amdgpu_fragment_publication_add_descriptor(
        query->descriptor_set, exchange_descriptor_ref, query->register_count,
        &recipe_cost);
  }
  if (available) {
    available = loom_amdgpu_fragment_publication_add_descriptor(
        query->descriptor_set, store_descriptor_ref, query->register_count,
        &recipe_cost);
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
      loom_amdgpu_fragment_publication_region_count(
          query,
          LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT *
              query->element_byte_count,
          publication) *
      query->register_count;
  *out_cost = loom_amdgpu_fragment_publication_finalize_cost(
      query, recipe_cost, memory_region_count, dependency_distance);
  return true;
}
