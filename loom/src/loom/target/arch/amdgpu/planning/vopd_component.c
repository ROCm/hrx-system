// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/vopd_component.h"

#include "loom/target/arch/amdgpu/planning/vopd_data.h"

static const loom_amdgpu_vopd_component_info_t*
loom_amdgpu_vopd_component_info_for_index_plus_one(
    uint8_t info_index_plus_one) {
  if (info_index_plus_one == 0) {
    return NULL;
  }
  return &loom_amdgpu_vopd_component_infos[info_index_plus_one - 1];
}

const loom_amdgpu_vopd_component_info_t* loom_amdgpu_vopd_component_info_for_op(
    uint16_t op) {
  if (op >= IREE_ARRAYSIZE(loom_amdgpu_vopd_component_info_indices_by_op)) {
    return NULL;
  }
  return loom_amdgpu_vopd_component_info_for_index_plus_one(
      loom_amdgpu_vopd_component_info_indices_by_op[op]);
}

static bool loom_amdgpu_vopd_component_can_use_lane(
    const loom_amdgpu_vopd_component_info_t* info,
    loom_amdgpu_vopd_component_lane_mask_t lane) {
  return info != NULL && iree_all_bits_set(info->lane_mask, lane);
}

static bool loom_amdgpu_vopd_component_can_pair(
    const loom_amdgpu_vopd_component_info_t* info,
    loom_amdgpu_vopd_component_pair_mask_t pairing_mode) {
  return info != NULL && iree_all_bits_set(info->pairing_mask, pairing_mode);
}

bool loom_amdgpu_vopd_component_infos_pair_reason(
    const loom_amdgpu_vopd_component_info_t* first_info,
    const loom_amdgpu_vopd_component_info_t* second_info,
    loom_amdgpu_vopd_pair_reason_t* out_reason) {
  *out_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN;
  if (!loom_amdgpu_vopd_component_can_use_lane(
          first_info, LOOM_AMDGPU_VOPD_COMPONENT_LANE_X) ||
      !loom_amdgpu_vopd_component_can_use_lane(
          second_info, LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y)) {
    return false;
  }
  if (first_info->op == second_info->op) {
    if (!loom_amdgpu_vopd_component_can_pair(
            first_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_SAME_OPCODE)) {
      return false;
    }
    *out_reason = first_info->same_op_reason;
    return *out_reason != LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN;
  }
  if (!loom_amdgpu_vopd_component_can_pair(
          first_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE) ||
      !loom_amdgpu_vopd_component_can_pair(
          second_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE)) {
    return false;
  }
  *out_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_MIXED_COMPONENTS;
  return true;
}

static bool loom_amdgpu_vopd_bank_compatible(uint16_t x_register,
                                             uint16_t y_register,
                                             uint16_t bank_mask) {
  return (x_register & bank_mask) != (y_register & bank_mask);
}

static loom_amdgpu_vopd_register_constraint_flags_t
loom_amdgpu_vopd_register_constraint_flags(
    const loom_amdgpu_vopd_candidate_pair_t* candidate) {
  loom_amdgpu_vopd_register_constraint_flags_t flags = 0;
  if (((candidate->x.vdst ^ candidate->y.vdst) & 1u) == 0) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_DESTINATION_PARITY;
  }
  const uint16_t x_registers[] = {0, candidate->x.src0, candidate->x.vsrc1,
                                  candidate->x.vdst};
  const uint16_t y_registers[] = {0, candidate->y.src0, candidate->y.vsrc1,
                                  candidate->y.vdst};
  const bool dual_move_separate_caches =
      candidate->x.rule->info->op == candidate->y.rule->info->op &&
      iree_all_bits_set(candidate->x.rule->flags & candidate->y.rule->flags,
                        LOOM_AMDGPU_VOPD_COMPONENT_FLAG_DUAL_MOV_SRC2_CACHE);
  const uint8_t source_bank_mask =
      iree_any_bit_set(candidate->x.rule->flags | candidate->y.rule->flags,
                       LOOM_AMDGPU_VOPD_COMPONENT_FLAG_SOURCE_PARITY)
          ? 1
          : 3;
  static const loom_amdgpu_vopd_register_constraint_flags_t cache_flags[] = {
      LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC0_BANK,
      LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_VSRC1_BANK,
      LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC2_BANK};
  for (iree_host_size_t cache = 0; cache < 3; ++cache) {
    uint8_t x_source = candidate->x.rule->cache_registers[cache];
    uint8_t y_source = candidate->y.rule->cache_registers[cache];
    if (dual_move_separate_caches) {
      y_source = cache == 2 ? LOOM_AMDGPU_VOPD_CACHE_REGISTER_SRC0
                            : LOOM_AMDGPU_VOPD_CACHE_REGISTER_NONE;
    }
    if (x_source && y_source &&
        !loom_amdgpu_vopd_bank_compatible(x_registers[x_source],
                                          y_registers[y_source],
                                          cache == 2 ? 1 : source_bank_mask)) {
      flags |= cache_flags[cache];
    }
  }
  if (iree_any_bit_set(candidate->y.rule->source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      candidate->x.vdst == candidate->y.src0) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_SRC0;
  }
  if (iree_any_bit_set(candidate->y.rule->source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1) &&
      candidate->x.vdst == candidate->y.vsrc1) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_VSRC1;
  }
  if (iree_any_bit_set(candidate->x.rule->source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      candidate->y.vdst == candidate->x.src0) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_SRC0;
  }
  if (iree_any_bit_set(candidate->x.rule->source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1) &&
      candidate->y.vdst == candidate->x.vsrc1) {
    flags |= LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_VSRC1;
  }
  return flags;
}

static void loom_amdgpu_vopd_swap_component_sources(
    loom_amdgpu_vopd_candidate_component_t* component) {
  IREE_ASSERT(iree_all_bits_set(component->rule->source_register_mask,
                                LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY));
  const uint16_t src0 = component->src0;
  component->src0 = component->vsrc1;
  component->vsrc1 = src0;
}

static bool loom_amdgpu_vopd_try_register_orientations(
    loom_amdgpu_vopd_candidate_pair_t* candidate,
    loom_amdgpu_vopd_register_constraint_flags_t* out_constraint_flags) {
  *out_constraint_flags = loom_amdgpu_vopd_register_constraint_flags(candidate);
  if (*out_constraint_flags == 0) {
    return true;
  }

  const bool can_swap_x =
      iree_any_bit_set(candidate->x.rule->flags,
                       LOOM_AMDGPU_VOPD_COMPONENT_FLAG_COMMUTABLE_SOURCES);
  const bool can_swap_y =
      iree_any_bit_set(candidate->y.rule->flags,
                       LOOM_AMDGPU_VOPD_COMPONENT_FLAG_COMMUTABLE_SOURCES);
  if (can_swap_x) {
    loom_amdgpu_vopd_swap_component_sources(&candidate->x);
    if (loom_amdgpu_vopd_register_constraint_flags(candidate) == 0) {
      candidate->flags |= LOOM_AMDGPU_VOPD_PAIR_FLAG_X_SOURCES_SWAPPED;
      return true;
    }
    loom_amdgpu_vopd_swap_component_sources(&candidate->x);
  }
  if (can_swap_y) {
    loom_amdgpu_vopd_swap_component_sources(&candidate->y);
    if (loom_amdgpu_vopd_register_constraint_flags(candidate) == 0) {
      candidate->flags |= LOOM_AMDGPU_VOPD_PAIR_FLAG_Y_SOURCES_SWAPPED;
      return true;
    }
    loom_amdgpu_vopd_swap_component_sources(&candidate->y);
  }
  if (can_swap_x && can_swap_y) {
    loom_amdgpu_vopd_swap_component_sources(&candidate->x);
    loom_amdgpu_vopd_swap_component_sources(&candidate->y);
    if (loom_amdgpu_vopd_register_constraint_flags(candidate) == 0) {
      candidate->flags |= LOOM_AMDGPU_VOPD_PAIR_FLAG_X_SOURCES_SWAPPED |
                          LOOM_AMDGPU_VOPD_PAIR_FLAG_Y_SOURCES_SWAPPED;
      return true;
    }
    loom_amdgpu_vopd_swap_component_sources(&candidate->x);
    loom_amdgpu_vopd_swap_component_sources(&candidate->y);
  }
  return false;
}

static bool loom_amdgpu_vopd_resolve_pair_literal(
    const loom_amdgpu_vopd_candidate_component_t* first,
    const loom_amdgpu_vopd_candidate_component_t* second,
    loom_amdgpu_vopd_candidate_pair_t* candidate) {
  const bool first_has_literal =
      iree_any_bit_set(first->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL);
  const bool second_has_literal =
      iree_any_bit_set(second->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL);
  if (first_has_literal && second_has_literal &&
      first->literal_u32 != second->literal_u32) {
    return false;
  }
  candidate->flags = first->flags | second->flags;
  if (first_has_literal) {
    candidate->literal_u32 = first->literal_u32;
  } else if (second_has_literal) {
    candidate->literal_u32 = second->literal_u32;
  }
  return true;
}

loom_amdgpu_vopd_rejection_reason_t loom_amdgpu_vopd_candidate_pair_resolve(
    loom_amdgpu_vopd_candidate_pair_t* candidate,
    loom_amdgpu_vopd_register_constraint_flags_t* out_constraint_flags) {
  *out_constraint_flags = 0;
  if (!loom_amdgpu_vopd_resolve_pair_literal(&candidate->x, &candidate->y,
                                             candidate)) {
    return LOOM_AMDGPU_VOPD_REJECTION_REASON_LITERAL_MISMATCH;
  }
  if (!loom_amdgpu_vopd_try_register_orientations(candidate,
                                                  out_constraint_flags)) {
    return LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS;
  }
  return LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN;
}
