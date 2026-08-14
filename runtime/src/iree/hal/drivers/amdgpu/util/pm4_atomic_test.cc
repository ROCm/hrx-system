// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/pm4_atomic.h"

#include <array>

#include "iree/testing/gtest.h"

namespace {

TEST(PM4AtomicTest, EmitsWaitPredicatesAtBothWidths) {
  struct WaitCase {
    iree_hal_atomic_wait_condition_t condition;
    uint32_t function;
  };
  constexpr WaitCase kCases[] = {
      {IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
       IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_EQUAL},
      {IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
       IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_NOT_EQUAL},
      {IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
       IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT_FUNCTION_UNSIGNED_GREATER_EQUAL},
  };

  for (const WaitCase& test_case : kCases) {
    std::array<uint32_t, IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT64_DWORD_COUNT> dwords =
        {};
    uint32_t dword_count = 0;
    EXPECT_TRUE(iree_hal_amdgpu_pm4_atomic_wait_emit(
        IREE_HAL_ATOMIC_WIDTH_32, test_case.condition,
        /*target_address=*/0x123456789ABCDEF4ull,
        /*value=*/0x11223344u, /*mask=*/0x55667788u, dwords.size(),
        dwords.data(), &dword_count));
    EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT32_DWORD_COUNT);
    EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                             IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_WAIT_REG_MEM,
                             IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT32_DWORD_COUNT));
    EXPECT_EQ(
        dwords[1],
        iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
            test_case.function, IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY,
            IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM));
    EXPECT_EQ(dwords[2], 0x9ABCDEF4u);
    EXPECT_EQ(dwords[3], 0x12345678u);
    EXPECT_EQ(dwords[4], 0x11223344u);
    EXPECT_EQ(dwords[5], 0x55667788u);
    EXPECT_EQ(dwords[6],
              4u | IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE);

    dwords.fill(0);
    EXPECT_TRUE(iree_hal_amdgpu_pm4_atomic_wait_emit(
        IREE_HAL_ATOMIC_WIDTH_64, test_case.condition,
        /*target_address=*/0x123456789ABCDEF8ull,
        /*value=*/0x1122334455667788ull,
        /*mask=*/0xFFEEDDCCBBAA0099ull, dwords.size(), dwords.data(),
        &dword_count));
    EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT64_DWORD_COUNT);
    EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                             IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_WAIT_REG_MEM64,
                             IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT64_DWORD_COUNT));
    EXPECT_EQ(
        dwords[1],
        iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
            test_case.function, IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY,
            IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM));
    EXPECT_EQ(dwords[2], 0x9ABCDEF8u);
    EXPECT_EQ(dwords[3], 0x12345678u);
    EXPECT_EQ(dwords[4], 0x55667788u);
    EXPECT_EQ(dwords[5], 0x11223344u);
    EXPECT_EQ(dwords[6], 0xBBAA0099u);
    EXPECT_EQ(dwords[7], 0xFFEEDDCCu);
    EXPECT_EQ(dwords[8], 4u);
  }
}

TEST(PM4AtomicTest, EmitsAtomicStoresAtBothWidths) {
  std::array<uint32_t, IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT> dwords = {};
  uint32_t dword_count = 0;
  EXPECT_TRUE(iree_hal_amdgpu_pm4_atomic_store_emit(
      IREE_HAL_ATOMIC_WIDTH_32, /*target_address=*/0x123456789ABCDEF4ull,
      /*value=*/0x55667788u, dwords.size(), dwords.data(), &dword_count));
  EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT);
  EXPECT_EQ(dwords[0], iree_hal_amdgpu_pm4_make_header(
                           IREE_HAL_AMDGPU_PM4_ATOMIC_OPCODE_ATOMIC_MEM,
                           IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT));
  EXPECT_EQ(dwords[1], IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_SWAP_32);
  EXPECT_EQ(dwords[2], 0x9ABCDEF4u);
  EXPECT_EQ(dwords[3], 0x12345678u);
  EXPECT_EQ(dwords[4], 0x55667788u);
  EXPECT_EQ(dwords[5], 0u);

  dwords.fill(0);
  EXPECT_TRUE(iree_hal_amdgpu_pm4_atomic_store_emit(
      IREE_HAL_ATOMIC_WIDTH_64, /*target_address=*/0x123456789ABCDEF8ull,
      /*value=*/0x1122334455667788ull, dwords.size(), dwords.data(),
      &dword_count));
  EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT);
  EXPECT_EQ(dwords[1], IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_SWAP_32 +
                           IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_64_OFFSET);
  EXPECT_EQ(dwords[2], 0x9ABCDEF8u);
  EXPECT_EQ(dwords[3], 0x12345678u);
  EXPECT_EQ(dwords[4], 0x55667788u);
  EXPECT_EQ(dwords[5], 0x11223344u);
}

TEST(PM4AtomicTest, EmitsEveryRmwOperationAtBothWidths) {
  struct RmwCase {
    iree_hal_atomic_rmw_operation_t operation;
    uint32_t encoded_operation_32;
  };
  constexpr RmwCase kCases[] = {
      {IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
       IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_ADD_32},
      {IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
       IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_SUBTRACT_32},
      {IREE_HAL_ATOMIC_RMW_OPERATION_AND,
       IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_AND_32},
      {IREE_HAL_ATOMIC_RMW_OPERATION_OR,
       IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_OR_32},
      {IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
       IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_XOR_32},
  };

  for (const RmwCase& test_case : kCases) {
    std::array<uint32_t, IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT> dwords =
        {};
    uint32_t dword_count = 0;
    EXPECT_TRUE(iree_hal_amdgpu_pm4_atomic_rmw_emit(
        IREE_HAL_ATOMIC_WIDTH_32, test_case.operation,
        /*target_address=*/0x123456789ABCDEF4ull,
        /*operand=*/0x55667788u, dwords.size(), dwords.data(), &dword_count));
    EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT);
    EXPECT_EQ(dwords[1], test_case.encoded_operation_32);

    EXPECT_TRUE(iree_hal_amdgpu_pm4_atomic_rmw_emit(
        IREE_HAL_ATOMIC_WIDTH_64, test_case.operation,
        /*target_address=*/0x123456789ABCDEF8ull,
        /*operand=*/0x1122334455667788ull, dwords.size(), dwords.data(),
        &dword_count));
    EXPECT_EQ(dword_count, IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT);
    EXPECT_EQ(dwords[1], test_case.encoded_operation_32 +
                             IREE_HAL_AMDGPU_PM4_ATOMIC_OPERATION_64_OFFSET);
    EXPECT_EQ(dwords[4], 0x55667788u);
    EXPECT_EQ(dwords[5], 0x11223344u);
  }
}

TEST(PM4AtomicTest, RejectsInvalidOrUnencodableOperations) {
  std::array<uint32_t, IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT> dwords = {};
  uint32_t dword_count = 99;
  EXPECT_FALSE(iree_hal_amdgpu_pm4_atomic_wait_emit(
      IREE_HAL_ATOMIC_WIDTH_32, (iree_hal_atomic_wait_condition_t)99,
      /*target_address=*/0x1000, /*value=*/0, /*mask=*/0, dwords.size(),
      dwords.data(), &dword_count));
  EXPECT_EQ(dword_count, 0u);
  EXPECT_FALSE(iree_hal_amdgpu_pm4_atomic_wait_emit(
      IREE_HAL_ATOMIC_WIDTH_64, IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      /*target_address=*/0x1004, /*value=*/0, /*mask=*/0, dwords.size(),
      dwords.data(), &dword_count));
  EXPECT_EQ(dword_count, 0u);
  EXPECT_FALSE(iree_hal_amdgpu_pm4_atomic_store_emit(
      IREE_HAL_ATOMIC_WIDTH_64, /*target_address=*/0x1000, /*value=*/0,
      IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT - 1, dwords.data(),
      &dword_count));
  EXPECT_EQ(dword_count, 0u);
  EXPECT_FALSE(iree_hal_amdgpu_pm4_atomic_rmw_emit(
      IREE_HAL_ATOMIC_WIDTH_32, (iree_hal_atomic_rmw_operation_t)99,
      /*target_address=*/0x1000, /*operand=*/0, dwords.size(), dwords.data(),
      &dword_count));
  EXPECT_EQ(dword_count, 0u);
}

}  // namespace
