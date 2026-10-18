// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/call_abi.h"

#include "iree/testing/gtest.h"
#include "loom/ir/scalar_type.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

TEST(Aie2pCallAbiTest, ClassifiesCompleteScalarFamily) {
  for (loom_scalar_type_t scalar_type = LOOM_SCALAR_TYPE_INDEX;
       scalar_type < LOOM_SCALAR_TYPE_COUNT_; ++scalar_type) {
    loom_aie2p_call_abi_requirements_t requirements = {};
    const bool supported = loom_aie2p_call_abi_add_source_type(
        loom_type_scalar(scalar_type), &requirements);
    const bool expected = scalar_type != LOOM_SCALAR_TYPE_I64 &&
                          scalar_type != LOOM_SCALAR_TYPE_F64;
    EXPECT_EQ(supported, expected) << static_cast<int>(scalar_type);
    if (expected) {
      EXPECT_EQ(requirements.scalar_count, 1);
      EXPECT_EQ(requirements.pointer_count, 0);
    }
  }
}

TEST(Aie2pCallAbiTest, ClassifiesPointerAndLowCarriers) {
  loom_aie2p_call_abi_requirements_t requirements = {};
  EXPECT_TRUE(
      loom_aie2p_call_abi_add_source_type(loom_type_buffer(), &requirements));
  EXPECT_EQ(requirements.scalar_count, 0);
  EXPECT_EQ(requirements.pointer_count, 1);

  requirements = {};
  EXPECT_TRUE(loom_aie2p_call_abi_add_low_type(
      loom_low_register_type(AIE2P_CORE_DESCRIPTOR_SET_ID,
                             AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1),
      &requirements));
  EXPECT_TRUE(loom_aie2p_call_abi_add_low_type(
      loom_low_register_type(AIE2P_CORE_DESCRIPTOR_SET_ID,
                             AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1),
      &requirements));
  EXPECT_EQ(requirements.scalar_count, 1);
  EXPECT_EQ(requirements.pointer_count, 1);
}

TEST(Aie2pCallAbiTest, RejectsNonAbiLowCarriers) {
  loom_aie2p_call_abi_requirements_t requirements = {};
  EXPECT_FALSE(loom_aie2p_call_abi_add_low_type(
      loom_low_register_type(AIE2P_CORE_DESCRIPTOR_SET_ID,
                             AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 2),
      &requirements));
  EXPECT_FALSE(loom_aie2p_call_abi_add_low_type(
      loom_low_register_type(AIE2P_CORE_DESCRIPTOR_SET_ID,
                             AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 1),
      &requirements));
  EXPECT_FALSE(loom_aie2p_call_abi_add_low_type(
      loom_low_register_type(AIE2P_CORE_DESCRIPTOR_SET_ID + 1,
                             AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1),
      &requirements));
}

TEST(Aie2pCallAbiTest, EnforcesIndependentRegisterBankCapacity) {
  loom_aie2p_call_abi_requirements_t requirements = {
      .scalar_count = 16,
  };
  EXPECT_TRUE(loom_aie2p_call_abi_requirements_fit(requirements));
  ++requirements.scalar_count;
  EXPECT_FALSE(loom_aie2p_call_abi_requirements_fit(requirements));

  requirements = {
      .pointer_count = loom_aie2p_core_descriptor_set()
                           ->reg_classes[AIE2P_CORE_REG_CLASS_ID_AIE2P_EP]
                           .allocatable_count,
  };
  EXPECT_TRUE(loom_aie2p_call_abi_requirements_fit(requirements));
  ++requirements.pointer_count;
  EXPECT_FALSE(loom_aie2p_call_abi_requirements_fit(requirements));
}

}  // namespace
}  // namespace loom
