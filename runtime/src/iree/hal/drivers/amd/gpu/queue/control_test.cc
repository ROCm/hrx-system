// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/control.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

void ExpectPacket(iree_hal_amd_gpu_control_t command,
                  std::initializer_list<uint32_t> expected) {
  std::array<uint32_t, 12> storage;
  storage.fill(0xCDCDCDCDu);
  auto* end = iree_hal_amd_gpu_control_emit(storage.data() + 1,
                                            UINT64_C(0x1234567890), command);
  EXPECT_EQ(end - storage.data() - 1, expected.size());
  EXPECT_EQ(iree_hal_amd_gpu_control_word_count(command), expected.size());
  EXPECT_EQ(std::vector<uint32_t>(storage.data() + 1, end),
            std::vector<uint32_t>(expected));
  EXPECT_EQ(storage.front(), 0xCDCDCDCDu);
  for (auto* word = end; word != storage.data() + storage.size(); ++word) {
    EXPECT_EQ(*word, 0xCDCDCDCDu);
  }
  const uint32_t offset = iree_hal_amd_gpu_control_address_word(command);
  EXPECT_EQ(storage[offset + 1], 0x34567890u);
  EXPECT_EQ(storage[offset + 2], 0x12u);
}

TEST(ControlTest, WaitWidthsPredicatesAndFullMasks) {
  for (uint8_t condition = 0; condition < 3; ++condition) {
    SCOPED_TRACE(condition);
    auto command = iree_hal_amd_gpu_control_wait({
        .value = 0x81020304,
        .mask = 0xFF0FFFFF,
        .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
        .width = IREE_HAL_ATOMIC_WIDTH_32,
        .condition = condition,
    });
    EXPECT_EQ(command.flags, IREE_HAL_ATOMIC_FLAG_ACQUIRE);
    ExpectPacket(command, {0xc0053c00, 0x13u + condition, 0x34567890, 0x12,
                           0x81020304, 0xFF0FFFFF, 0x80000004});
    command.width = IREE_HAL_ATOMIC_WIDTH_64;
    command.value = UINT64_C(0x8001020381020304);
    command.mask = UINT64_C(0xFF0FFFFF0FFFFFFF);
    ExpectPacket(command, {0xc0079300, 0x13u + condition, 0x34567890, 0x12,
                           0x81020304, 0x80010203, 0x0FFFFFFF, 0xFF0FFFFF, 4});
  }
}

TEST(ControlTest, StoreIsAConfirmedWriteWithoutReadingTheTarget) {
  auto command = iree_hal_amd_gpu_control_store({
      .value = 0x81234567,
      .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
      .width = IREE_HAL_ATOMIC_WIDTH_32,
  });
  EXPECT_EQ(command.flags, IREE_HAL_ATOMIC_FLAG_RELEASE);
  ExpectPacket(command,
               {0xc0044000, 0x100205, 0x81234567, 0, 0x34567890, 0x12});
  command.width = IREE_HAL_ATOMIC_WIDTH_64;
  command.value = UINT64_C(0xFEDCBA9881234567);
  ExpectPacket(command, {0xc0044000, 0x110205, 0x81234567, 0xFEDCBA98,
                         0x34567890, 0x12});
}

TEST(ControlTest, RmwFamilyUsesReturningTcOperations) {
  // TC operation numbers, independently of the HAL operation enum. Subtract
  // is composed using PCIe-supported ADD; INT_MIN wraps without signed UB.
  const uint32_t opcodes[] = {0x0f, 0x0f, 0x15, 0x16, 0x17};
  for (uint8_t operation = 0; operation < 5; ++operation) {
    for (uint8_t width : {32, 64}) {
      SCOPED_TRACE(::testing::Message()
                   << unsigned(operation) << "/" << unsigned(width));
      auto command = iree_hal_amd_gpu_control_rmw({
          .operand = 3,
          .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
          .width = width,
          .operation = operation,
      });
      const bool subtract = operation == IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT;
      ExpectPacket(command,
                   {0xc0071e00, opcodes[operation] + (width == 64 ? 0x20u : 0),
                    0x34567890, 0x12, subtract ? 0xFFFFFFFDu : 3u,
                    subtract ? 0xFFFFFFFFu : 0u, 0, 0, 0});
    }
  }
  ExpectPacket(iree_hal_amd_gpu_control_rmw({
                   .operand = UINT64_C(1) << 63,
                   .width = IREE_HAL_ATOMIC_WIDTH_64,
                   .operation = IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
               }),
               {0xc0071e00, 0x2f, 0x34567890, 0x12, 0, 0x80000000, 0, 0, 0});
}

TEST(ControlTest, RequirementsMatchExactSiteCellsAndErrorModes) {
  std::array<iree_hal_memory_scope_access_t, 4> scopes = {};
  iree_hal_memory_contract_t contract = {
      .scope_count = scopes.size(),
      .scopes = scopes.data(),
  };
  iree_hal_buffer_t buffer = {};
  buffer.memory.contract = &contract;
  iree_hal_queue_family_t family = {};
  family.memory.queue_scope_id = 2;
  for (uint32_t cell = 0; cell < 4; ++cell) {
    for (uint32_t operation = 0; operation < 7; ++operation) {
      auto command = iree_hal_amd_gpu_control_wait({
          .width = static_cast<iree_hal_atomic_width_t>(
              cell & 1 ? IREE_HAL_ATOMIC_WIDTH_64 : IREE_HAL_ATOMIC_WIDTH_32),
      });
      command.flags = cell & 2 ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE : 0;
      command.kind = operation == 0   ? IREE_HAL_AMD_GPU_CONTROL_WAIT
                     : operation == 1 ? IREE_HAL_AMD_GPU_CONTROL_STORE
                                      : IREE_HAL_AMD_GPU_CONTROL_RMW;
      command.operation = operation >= 2 ? operation - 2 : 0;
      const uint32_t required = iree_hal_amd_gpu_control_requirement(command);
      EXPECT_EQ(required, 1u << (cell * 8 + operation));
      scopes[2].atomic_operations = required;
      IREE_EXPECT_OK(iree_hal_amd_gpu_control_validate_target(&family, &buffer,
                                                              required, 0));
      scopes[2].atomic_operations = 0;
      scopes[3].atomic_operations = required;
      IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                            iree_hal_amd_gpu_control_validate_target(
                                &family, &buffer, required, 0));
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                            iree_hal_amd_gpu_control_validate_target(
                                &family, &buffer, 0, required));
      IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                            iree_hal_amd_gpu_control_validate_target(
                                &family, &buffer, required, required));
    }
  }
  EXPECT_EQ(sizeof(iree_hal_amd_gpu_control_t), 24u);
  EXPECT_EQ(iree_hal_amd_gpu_control_requirement(
                iree_hal_amd_gpu_control_timestamp()),
            0u);
}

}  // namespace
