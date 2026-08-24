// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/buffer.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

class VulkanBufferTest : public ::testing::Test {
 protected:
  void TearDown() override { iree_hal_buffer_release(buffer_); }

  void CreateBorrowedBuffer(iree_device_size_t handle_offset,
                            VkDeviceAddress device_address) {
    iree_hal_vulkan_device_syms_t syms = {};
    const iree_hal_buffer_placement_t placement =
        iree_hal_buffer_placement_undefined();
    IREE_ASSERT_OK(iree_hal_vulkan_buffer_create_borrowed(
        &syms, reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1000)),
        placement, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*allocation_size=*/64, handle_offset,
        /*byte_length=*/16, /*memory_property_flags=*/0,
        /*non_coherent_atom_size=*/1, VK_NULL_HANDLE,
        /*mapping_state=*/nullptr,
        reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(0x2000)),
        device_address, iree_hal_buffer_release_callback_null(),
        iree_allocator_system(), &buffer_));
  }

  iree_hal_buffer_t* buffer_ = nullptr;
};

TEST_F(VulkanBufferTest, DeviceAddressIncludesBorrowedHandleOffset) {
  CreateBorrowedBuffer(/*handle_offset=*/4, /*device_address=*/0x10000);

  VkDeviceAddress device_address = 0;
  IREE_ASSERT_OK(
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
  EXPECT_EQ(device_address, 0x10004u);
}

TEST_F(VulkanBufferTest, MissingDeviceAddressRemainsMissing) {
  CreateBorrowedBuffer(/*handle_offset=*/4, /*device_address=*/0);

  VkDeviceAddress device_address = UINT64_MAX;
  IREE_ASSERT_OK(
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
  EXPECT_EQ(device_address, 0u);
}

TEST_F(VulkanBufferTest, DeviceAddressOverflowFails) {
  CreateBorrowedBuffer(/*handle_offset=*/4,
                       /*device_address=*/UINT64_MAX - 3);

  VkDeviceAddress device_address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
}

}  // namespace
}  // namespace iree::hal::vulkan
