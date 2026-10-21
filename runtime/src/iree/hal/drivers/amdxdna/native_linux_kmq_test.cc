// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>

#include "iree/hal/drivers/amdxdna/native_linux_kmq_internal.h"
#include "iree/hal/drivers/amdxdna/shim/linux/kmq/kernel.h"
#include "iree/testing/gtest.h"

namespace {

TEST(NativeLinuxKmqTest, BoAllocationHeapExhaustionIsRecoverable) {
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(EAGAIN),
            IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(-EAGAIN),
            IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(ENOSPC),
            IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(-ENOSPC),
            IREE_STATUS_UNAVAILABLE);
}

TEST(NativeLinuxKmqTest, BoAllocationOtherErrorsPreserveStatus) {
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(ENOMEM),
            IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(iree_hal_amdxdna_native_linux_bo_allocation_status_code(EINVAL),
            IREE_STATUS_INVALID_ARGUMENT);
}

TEST(NativeLinuxKmqTest, FullElfPacketPayloadMatchesFirmwareAbi) {
  ert_npu_preempt_data data = {};
  shim_xdna::initialize_npu_elf_data(&data, 0x1122334455667788ull, 0x1234u);

  EXPECT_EQ(ERT_START_NPU_PREEMPT_ELF, 22);
  EXPECT_EQ(2u + sizeof(data) / sizeof(uint32_t), 12u);
  EXPECT_EQ(data.instruction_buffer, 0x1122334455667788ull);
  EXPECT_EQ(data.instruction_buffer_size, 0x1234u);
  EXPECT_EQ(data.save_buffer, 0u);
  EXPECT_EQ(data.restore_buffer, 0u);
  EXPECT_EQ(data.save_buffer_size, 0u);
  EXPECT_EQ(data.restore_buffer_size, 0u);
  EXPECT_EQ(data.instruction_prop_count, 0u);
}

TEST(NativeLinuxKmqTest, InstructionOpcodesRetainControlBoDependency) {
  EXPECT_TRUE(shim_xdna::kernel_opcode_requires_instruction_bo_dependency(
      ERT_START_NPU));
  EXPECT_TRUE(shim_xdna::kernel_opcode_requires_instruction_bo_dependency(
      ERT_START_NPU_PREEMPT_ELF));
  EXPECT_TRUE(shim_xdna::kernel_opcode_requires_instruction_bo_dependency(
      ERT_START_DPU));
  EXPECT_FALSE(shim_xdna::kernel_opcode_requires_instruction_bo_dependency(
      ERT_START_CU));
  EXPECT_FALSE(shim_xdna::kernel_opcode_requires_instruction_bo_dependency(
      ERT_CMD_CHAIN));
}

TEST(NativeLinuxKmqTest, FullElfCapabilityRejectsPreProtocolFirmware) {
  EXPECT_FALSE(iree_hal_amdxdna_native_linux_firmware_supports_full_elf(
      false, 0, 0, 0, 0));
  EXPECT_FALSE(iree_hal_amdxdna_native_linux_firmware_supports_full_elf(
      true, 1, 0, 0, 63));
  EXPECT_FALSE(iree_hal_amdxdna_native_linux_firmware_supports_full_elf(
      true, 1, 0, 0, 166));
  EXPECT_FALSE(iree_hal_amdxdna_native_linux_firmware_supports_full_elf(
      true, 1, 0, 20, 30));
}

TEST(NativeLinuxKmqTest, FullElfCapabilityIsMonotonicFromProtocolFloor) {
  // 1.0.20.31 is the oldest public release known to carry protocol 6.12.
  EXPECT_TRUE(iree_hal_amdxdna_native_linux_firmware_supports_full_elf(
      true, 1, 0, 20, 31));
  EXPECT_TRUE(iree_hal_amdxdna_native_linux_firmware_supports_full_elf(
      true, 1, 1, 2, 65));
  EXPECT_TRUE(iree_hal_amdxdna_native_linux_firmware_supports_full_elf(
      true, 2, 0, 0, 0));
}

}  // namespace
