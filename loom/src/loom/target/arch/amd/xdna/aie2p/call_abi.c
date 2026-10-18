// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"

#include "loom/ir/scalar_type.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/registers.h"

static bool loom_aie2p_call_abi_add_count(uint16_t* count, uint16_t increment) {
  if (*count > UINT16_MAX - increment) {
    return false;
  }
  *count += increment;
  return true;
}

bool loom_aie2p_call_abi_add_source_type(
    loom_type_t source_type,
    loom_aie2p_call_abi_requirements_t* inout_requirements) {
  if (loom_type_is_buffer(source_type) || loom_type_is_view(source_type)) {
    return loom_aie2p_call_abi_add_count(&inout_requirements->pointer_count, 1);
  }
  if (!loom_type_is_scalar(source_type)) {
    return false;
  }
  switch (loom_type_element_type(source_type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
      return loom_aie2p_call_abi_add_count(&inout_requirements->scalar_count,
                                           1);
    default:
      return false;
  }
}

bool loom_aie2p_call_abi_add_low_type(
    loom_type_t low_type,
    loom_aie2p_call_abi_requirements_t* inout_requirements) {
  if (!loom_low_type_is_register(low_type) ||
      loom_low_register_type_descriptor_set_stable_id(low_type) !=
          AIE2P_CORE_DESCRIPTOR_SET_ID ||
      loom_low_register_type_unit_count(low_type) != 1) {
    return false;
  }
  switch (loom_low_register_type_class_id(low_type)) {
    case AIE2P_CORE_REG_CLASS_ID_AIE2P_ER:
      return loom_aie2p_call_abi_add_count(&inout_requirements->scalar_count,
                                           1);
    case AIE2P_CORE_REG_CLASS_ID_AIE2P_EP:
      return loom_aie2p_call_abi_add_count(&inout_requirements->pointer_count,
                                           1);
    default:
      return false;
  }
}

bool loom_aie2p_call_abi_requirements_fit(
    loom_aie2p_call_abi_requirements_t requirements) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  const uint16_t pointer_register_count =
      descriptor_set->reg_classes[AIE2P_CORE_REG_CLASS_ID_AIE2P_EP]
          .allocatable_count;
  IREE_ASSERT_GE(pointer_register_count,
                 LOOM_AIE2P_CALL_ABI_POINTER_REGISTER_OFFSET);
  return requirements.scalar_count <=
             LOOM_AIE2P_CALL_ABI_SCALAR_REGISTER_COUNT &&
         requirements.pointer_count <=
             pointer_register_count -
                 LOOM_AIE2P_CALL_ABI_POINTER_REGISTER_OFFSET;
}

uint16_t loom_aie2p_call_abi_preserved_scalar_register(uint16_t ordinal) {
  static const uint16_t candidate_ordinals[] = {15, 18, 19, 20, 21, 22};
  IREE_ASSERT_LT(ordinal, IREE_ARRAYSIZE(candidate_ordinals));
  return loom_low_descriptor_set_physical_register_candidate(
      loom_aie2p_core_descriptor_set(), AIE2P_CORE_REG_CLASS_ID_AIE2P_ER,
      candidate_ordinals[ordinal]);
}

iree_status_t loom_aie2p_call_abi_select_location(
    loom_type_t low_type,
    loom_aie2p_call_abi_requirements_t* inout_requirements,
    loom_low_allocation_abi_location_t* out_location) {
  const loom_aie2p_call_abi_requirements_t prior_requirements =
      *inout_requirements;
  if (!loom_aie2p_call_abi_add_low_type(low_type, inout_requirements)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P direct helper ABI requires one-unit register values");
  }
  if (!loom_aie2p_call_abi_requirements_fit(*inout_requirements)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P direct helper ABI register bank overflow");
  }

  const uint16_t register_class = loom_low_register_type_class_id(low_type);
  uint16_t candidate_ordinal = prior_requirements.scalar_count;
  if (register_class == AIE2P_CORE_REG_CLASS_ID_AIE2P_EP) {
    candidate_ordinal = LOOM_AIE2P_CALL_ABI_POINTER_REGISTER_OFFSET +
                        prior_requirements.pointer_count;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_aie2p_core_descriptor_set();
  *out_location = (loom_low_allocation_abi_location_t){
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .descriptor_reg_class_id = register_class,
      .location_base = loom_low_descriptor_set_physical_register_candidate(
          descriptor_set, register_class, candidate_ordinal),
  };
  return iree_ok_status();
}
