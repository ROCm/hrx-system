// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/abi/call_abi.h"

#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static bool loom_amdgpu_call_abi_add_count(uint16_t* count,
                                           uint32_t increment) {
  if (increment > UINT16_MAX || *count > UINT16_MAX - increment) {
    return false;
  }
  *count += (uint16_t)increment;
  return true;
}

loom_amdgpu_call_fixed_state_mask_t
loom_amdgpu_call_fixed_states_for_register_class(
    uint16_t register_class,
    loom_amdgpu_reg_class_traits_t register_class_traits) {
  if (register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT |
           LOOM_AMDGPU_CALL_FIXED_STATE_MATRIX_RESULT |
           LOOM_AMDGPU_CALL_FIXED_STATE_MATRIX_COEXECUTION |
           LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU;
  }
  if (register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT |
           LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU |
           LOOM_AMDGPU_CALL_FIXED_STATE_MASK_WRITE |
           LOOM_AMDGPU_CALL_FIXED_STATE_SGPR_READ;
  }
  if (register_class == LOOM_AMDGPU_REG_CLASS_ID_SCC) {
    return LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU;
  }
  loom_amdgpu_call_fixed_state_mask_t states = 0;
  if (iree_any_bit_set(register_class_traits,
                       LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR)) {
    states |= LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT |
              LOOM_AMDGPU_CALL_FIXED_STATE_MATRIX_RESULT;
  }
  if (iree_any_bit_set(register_class_traits, LOOM_AMDGPU_REG_CLASS_TRAIT_M0)) {
    states |= LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT |
              LOOM_AMDGPU_CALL_FIXED_STATE_MASK_WRITE;
  }
  if (iree_any_bit_set(register_class_traits,
                       LOOM_AMDGPU_REG_CLASS_TRAIT_VCC)) {
    states |= LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT |
              LOOM_AMDGPU_CALL_FIXED_STATE_MASK_WRITE |
              LOOM_AMDGPU_CALL_FIXED_STATE_SGPR_READ;
  }
  return states;
}

loom_amdgpu_call_state_t loom_amdgpu_call_state_resolve(
    loom_amdgpu_call_state_provider_t provider, loom_symbol_ref_t callee) {
  if (provider.query == NULL) {
    return (loom_amdgpu_call_state_t){
        .counter_mask = UINT8_MAX,
        .fixed_state_mask = LOOM_AMDGPU_CALL_FIXED_STATE_ALL,
    };
  }
  const loom_amdgpu_call_state_t* state =
      provider.query(provider.user_data, callee);
  IREE_ASSERT(state != NULL);
  return *state;
}

bool loom_amdgpu_call_abi_add_low_type(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t low_type,
    loom_amdgpu_call_abi_requirements_t* inout_requirements) {
  if (!descriptor_set || !loom_low_type_is_register(low_type) ||
      loom_low_register_type_descriptor_set_stable_id(low_type) !=
          descriptor_set->stable_id) {
    return false;
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  switch (loom_low_register_type_class_id(low_type)) {
    case LOOM_AMDGPU_REG_CLASS_ID_SGPR:
      return loom_amdgpu_call_abi_add_count(&inout_requirements->sgpr_count,
                                            unit_count);
    case LOOM_AMDGPU_REG_CLASS_ID_VGPR:
      return loom_amdgpu_call_abi_add_count(&inout_requirements->vgpr_count,
                                            unit_count);
    default:
      return false;
  }
}

bool loom_amdgpu_call_abi_requirements_fit(
    loom_amdgpu_call_abi_requirements_t requirements) {
  return requirements.sgpr_count <= LOOM_AMDGPU_CALL_ABI_SGPR_COUNT &&
         requirements.vgpr_count <= LOOM_AMDGPU_CALL_ABI_VGPR_COUNT;
}

iree_status_t loom_amdgpu_call_abi_select_location(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t low_type,
    loom_amdgpu_call_abi_requirements_t* inout_requirements,
    loom_low_allocation_abi_location_t* out_location) {
  const loom_amdgpu_call_abi_requirements_t prior_requirements =
      *inout_requirements;
  loom_amdgpu_call_abi_requirements_t requirements = prior_requirements;
  if (!loom_amdgpu_call_abi_add_low_type(descriptor_set, low_type,
                                         &requirements)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU direct helper ABI requires SGPR or VGPR register values");
  }
  if (!loom_amdgpu_call_abi_requirements_fit(requirements)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU direct helper ABI requires %u SGPR and %u VGPR units; "
        "the fixed carrier banks provide %u and %u",
        requirements.sgpr_count, requirements.vgpr_count,
        LOOM_AMDGPU_CALL_ABI_SGPR_COUNT, LOOM_AMDGPU_CALL_ABI_VGPR_COUNT);
  }

  const uint16_t register_class = loom_low_register_type_class_id(low_type);
  *out_location = (loom_low_allocation_abi_location_t){
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .descriptor_reg_class_id = register_class,
      .location_base =
          register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR
              ? LOOM_AMDGPU_CALL_ABI_SGPR_BASE + prior_requirements.sgpr_count
              : LOOM_AMDGPU_CALL_ABI_VGPR_BASE + prior_requirements.vgpr_count,
  };
  *inout_requirements = requirements;
  return iree_ok_status();
}
