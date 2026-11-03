// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/abi/call_abi.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

constexpr uint64_t kDescriptorSetId = UINT64_C(0x123456789abcdef0);
const loom_low_descriptor_set_t kDescriptorSet = {
    .stable_id = kDescriptorSetId,
};

const loom_amdgpu_call_state_t* QueryCallState(void* user_data,
                                               loom_symbol_ref_t callee) {
  IREE_ASSERT(!loom_symbol_ref_is_valid(callee));
  return static_cast<const loom_amdgpu_call_state_t*>(user_data);
}

TEST(AmdgpuCallAbiTest, ResolvesCallState) {
  loom_amdgpu_call_state_t exact = {
      .counter_mask = 3,
      .fixed_hazard_mask = LOOM_AMDGPU_CALL_FIXED_HAZARD_DELAY_ALU,
  };
  const loom_amdgpu_call_state_t resolved = loom_amdgpu_call_state_resolve(
      {.query = QueryCallState, .user_data = &exact}, loom_symbol_ref_null());
  EXPECT_EQ(resolved.counter_mask, exact.counter_mask);
  EXPECT_EQ(resolved.fixed_hazard_mask, exact.fixed_hazard_mask);

  const loom_amdgpu_call_state_t conservative =
      loom_amdgpu_call_state_resolve({}, loom_symbol_ref_null());
  EXPECT_EQ(conservative.counter_mask, UINT8_MAX);
  EXPECT_EQ(conservative.fixed_hazard_mask, LOOM_AMDGPU_CALL_FIXED_HAZARD_ALL);
}

TEST(AmdgpuCallAbiTest, ClassifiesFixedHazardRegisterFamilies) {
  struct Case {
    uint16_t register_class;
    loom_amdgpu_reg_class_traits_t traits;
    loom_amdgpu_call_fixed_hazard_mask_t expected;
  };
  const Case cases[] = {
      {
          LOOM_AMDGPU_REG_CLASS_ID_SGPR,
          0,
          LOOM_AMDGPU_CALL_FIXED_HAZARD_SHORT_WAIT |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_DELAY_ALU |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_MASK_WRITE |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_SGPR_READ,
      },
      {
          LOOM_AMDGPU_REG_CLASS_ID_VGPR,
          0,
          LOOM_AMDGPU_CALL_FIXED_HAZARD_SHORT_WAIT |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_MATRIX_RESULT |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_MATRIX_COEXECUTION |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_DELAY_ALU,
      },
      {
          LOOM_AMDGPU_REG_CLASS_ID_SCC,
          0,
          LOOM_AMDGPU_CALL_FIXED_HAZARD_DELAY_ALU,
      },
      {
          LOOM_LOW_REG_CLASS_NONE,
          LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR,
          LOOM_AMDGPU_CALL_FIXED_HAZARD_SHORT_WAIT |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_MATRIX_RESULT,
      },
      {
          LOOM_LOW_REG_CLASS_NONE,
          LOOM_AMDGPU_REG_CLASS_TRAIT_M0,
          LOOM_AMDGPU_CALL_FIXED_HAZARD_SHORT_WAIT |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_MASK_WRITE,
      },
      {
          LOOM_LOW_REG_CLASS_NONE,
          LOOM_AMDGPU_REG_CLASS_TRAIT_VCC,
          LOOM_AMDGPU_CALL_FIXED_HAZARD_SHORT_WAIT |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_MASK_WRITE |
              LOOM_AMDGPU_CALL_FIXED_HAZARD_SGPR_READ,
      },
      {
          LOOM_AMDGPU_REG_CLASS_ID_EXEC,
          0,
          0,
      },
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.register_class);
    EXPECT_EQ(loom_amdgpu_call_fixed_hazards_for_register_class(
                  test_case.register_class, test_case.traits),
              test_case.expected);
  }
}

TEST(AmdgpuCallAbiTest, ClassifiesSgprAndVgprRanges) {
  loom_amdgpu_call_abi_requirements_t requirements = {};
  EXPECT_TRUE(loom_amdgpu_call_abi_add_low_type(
      &kDescriptorSet,
      loom_low_register_type(kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                             2),
      &requirements));
  EXPECT_TRUE(loom_amdgpu_call_abi_add_low_type(
      &kDescriptorSet,
      loom_low_register_type(kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                             4),
      &requirements));
  EXPECT_EQ(requirements.sgpr_count, 2);
  EXPECT_EQ(requirements.vgpr_count, 4);
}

TEST(AmdgpuCallAbiTest, RejectsForeignAndSpecialRegisterCarriers) {
  loom_amdgpu_call_abi_requirements_t requirements = {};
  EXPECT_FALSE(loom_amdgpu_call_abi_add_low_type(
      &kDescriptorSet,
      loom_low_register_type(kDescriptorSetId + 1,
                             LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1),
      &requirements));
  EXPECT_FALSE(loom_amdgpu_call_abi_add_low_type(
      &kDescriptorSet,
      loom_low_register_type(kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_EXEC,
                             1),
      &requirements));
  EXPECT_EQ(requirements.sgpr_count, 0);
  EXPECT_EQ(requirements.vgpr_count, 0);
}

TEST(AmdgpuCallAbiTest, EnforcesIndependentRegisterBankCapacity) {
  loom_amdgpu_call_abi_requirements_t requirements = {
      .sgpr_count = LOOM_AMDGPU_CALL_ABI_SGPR_COUNT,
      .vgpr_count = LOOM_AMDGPU_CALL_ABI_VGPR_COUNT,
  };
  EXPECT_TRUE(loom_amdgpu_call_abi_requirements_fit(requirements));
  ++requirements.sgpr_count;
  EXPECT_FALSE(loom_amdgpu_call_abi_requirements_fit(requirements));
  --requirements.sgpr_count;
  ++requirements.vgpr_count;
  EXPECT_FALSE(loom_amdgpu_call_abi_requirements_fit(requirements));
}

TEST(AmdgpuCallAbiTest, FailedSelectionDoesNotConsumeCarrierRegisters) {
  loom_amdgpu_call_abi_requirements_t requirements = {
      .sgpr_count = LOOM_AMDGPU_CALL_ABI_SGPR_COUNT,
  };
  loom_low_allocation_abi_location_t location = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_amdgpu_call_abi_select_location(
          &kDescriptorSet,
          loom_low_register_type(kDescriptorSetId,
                                 LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1),
          &requirements, &location));
  EXPECT_EQ(requirements.sgpr_count, LOOM_AMDGPU_CALL_ABI_SGPR_COUNT);
  EXPECT_EQ(requirements.vgpr_count, 0);
}

TEST(AmdgpuCallAbiTest, AssignsIndependentArgumentAndResultBanks) {
  const loom_type_t sgpr_x2 = loom_low_register_type(
      kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  const loom_type_t vgpr_x4 = loom_low_register_type(
      kDescriptorSetId, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4);
  loom_amdgpu_call_abi_requirements_t requirements = {};
  loom_low_allocation_abi_location_t location = {};

  IREE_ASSERT_OK(loom_amdgpu_call_abi_select_location(
      &kDescriptorSet, sgpr_x2, &requirements, &location));
  EXPECT_EQ(location.descriptor_reg_class_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  EXPECT_EQ(location.location_base, LOOM_AMDGPU_CALL_ABI_SGPR_BASE);
  IREE_ASSERT_OK(loom_amdgpu_call_abi_select_location(
      &kDescriptorSet, vgpr_x4, &requirements, &location));
  EXPECT_EQ(location.descriptor_reg_class_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  EXPECT_EQ(location.location_base, LOOM_AMDGPU_CALL_ABI_VGPR_BASE);
  IREE_ASSERT_OK(loom_amdgpu_call_abi_select_location(
      &kDescriptorSet, sgpr_x2, &requirements, &location));
  EXPECT_EQ(location.location_base, LOOM_AMDGPU_CALL_ABI_SGPR_BASE + 2);
  IREE_ASSERT_OK(loom_amdgpu_call_abi_select_location(
      &kDescriptorSet, vgpr_x4, &requirements, &location));
  EXPECT_EQ(location.location_base, LOOM_AMDGPU_CALL_ABI_VGPR_BASE + 4);

  requirements = {};
  IREE_ASSERT_OK(loom_amdgpu_call_abi_select_location(
      &kDescriptorSet, sgpr_x2, &requirements, &location));
  EXPECT_EQ(location.location_base, LOOM_AMDGPU_CALL_ABI_SGPR_BASE);
}

TEST(AmdgpuCallAbiTest, KeepsControlStateOutsideTheCarrierBanks) {
  EXPECT_EQ(LOOM_AMDGPU_CALL_ABI_SGPR_BASE + LOOM_AMDGPU_CALL_ABI_SGPR_COUNT,
            LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE);
  EXPECT_EQ(LOOM_AMDGPU_CALL_ABI_RETURN_ADDRESS_BASE,
            LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE);
  EXPECT_EQ(LOOM_AMDGPU_CALL_ABI_RETURN_ADDRESS_BASE % 2, 0);
  EXPECT_EQ(LOOM_AMDGPU_CALL_ABI_CALL_TARGET_BASE % 2, 0);
  EXPECT_EQ(LOOM_AMDGPU_CALL_ABI_CALL_TARGET_BASE,
            LOOM_AMDGPU_CALL_ABI_RETURN_ADDRESS_BASE + 2);
  EXPECT_LT(LOOM_AMDGPU_CALL_ABI_CALL_TARGET_BASE + 1,
            LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE +
                LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT);
}

}  // namespace
}  // namespace loom
