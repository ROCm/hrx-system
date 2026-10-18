// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"

#include "loom/ir/scalar_type.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/registers.h"

enum {
  // The first sixteen ERs form the scalar argument and result bank. Keeping
  // the two sides independent lets a returned value reuse an argument ER.
  LOOM_AIE2P_CALL_ABI_SCALAR_REGISTER_COUNT = 16,
};

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
  return requirements.scalar_count <=
             LOOM_AIE2P_CALL_ABI_SCALAR_REGISTER_COUNT &&
         requirements.pointer_count <=
             descriptor_set->reg_classes[AIE2P_CORE_REG_CLASS_ID_AIE2P_EP]
                 .allocatable_count;
}
