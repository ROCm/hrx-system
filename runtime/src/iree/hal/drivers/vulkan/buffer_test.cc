// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/buffer.h"

#include <cstdint>

#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

static VkDeviceAddress g_sparse_device_address = 0;

static VKAPI_ATTR VkDeviceAddress VKAPI_CALL FakeGetBufferDeviceAddress(
    VkDevice device, const VkBufferDeviceAddressInfo* address_info) {
  (void)device;
  (void)address_info;
  return g_sparse_device_address;
}

static VKAPI_ATTR void VKAPI_CALL
FakeDestroyBuffer(VkDevice device, VkBuffer buffer,
                  const VkAllocationCallbacks* allocation_callbacks) {
  (void)device;
  (void)buffer;
  (void)allocation_callbacks;
}

class VulkanBufferTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_hal_buffer_release(buffer_);
    iree_hal_buffer_release(allocated_buffer_);
  }

  iree_status_t CreateDenseBuffer(iree_device_size_t handle_offset,
                                  VkDeviceAddress device_address) {
    iree_hal_vulkan_device_syms_t syms = {};
    return iree_hal_vulkan_buffer_create_borrowed(
        &syms, reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1000)),
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*allocation_size=*/64, handle_offset,
        /*byte_length=*/16, /*memory_property_flags=*/0,
        /*non_coherent_atom_size=*/1, VK_NULL_HANDLE,
        /*mapping_state=*/nullptr,
        reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(0x2000)),
        device_address, iree_hal_buffer_release_callback_null(),
        iree_allocator_system(), &buffer_);
  }

  iree_status_t CreateSparseSubspan(iree_device_size_t byte_offset,
                                    VkDeviceAddress device_address) {
    g_sparse_device_address = device_address;
    iree_hal_vulkan_device_syms_t syms = {};
    syms.vkGetBufferDeviceAddress = FakeGetBufferDeviceAddress;
    syms.vkDestroyBuffer = FakeDestroyBuffer;
    iree_hal_buffer_placement_t placement =
        iree_hal_buffer_placement_undefined();
    placement.device =
        reinterpret_cast<iree_hal_device_t*>(static_cast<uintptr_t>(0x1000));
    VkMemoryRequirements memory_requirements = {};
    memory_requirements.size = 64;
    memory_requirements.alignment = 1;
    memory_requirements.memoryTypeBits = 1;
    iree_status_t status = iree_hal_vulkan_sparse_buffer_create_unbound(
        &syms, reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x2000)),
        placement, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*allocation_size=*/64,
        /*byte_length=*/64,
        reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(0x3000)),
        memory_requirements, iree_allocator_system(), &allocated_buffer_);
    if (iree_status_is_ok(status)) {
      status = iree_hal_buffer_subspan(allocated_buffer_, byte_offset,
                                       /*byte_length=*/16,
                                       iree_allocator_system(), &buffer_);
    }
    return status;
  }

  iree_hal_buffer_t* allocated_buffer_ = nullptr;
  iree_hal_buffer_t* buffer_ = nullptr;
};

TEST_F(VulkanBufferTest, DenseAddressIncludesHandleOffset) {
  IREE_ASSERT_OK(
      CreateDenseBuffer(/*handle_offset=*/4, /*device_address=*/0x10000));

  VkDeviceAddress device_address = 0;
  IREE_ASSERT_OK(
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
  EXPECT_EQ(device_address, 0x10004u);
}

TEST_F(VulkanBufferTest, DenseMissingAddressRemainsMissing) {
  IREE_ASSERT_OK(CreateDenseBuffer(/*handle_offset=*/4, /*device_address=*/0));

  VkDeviceAddress device_address = UINT64_MAX;
  IREE_ASSERT_OK(
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
  EXPECT_EQ(device_address, 0u);
}

TEST_F(VulkanBufferTest, DenseAddressOverflowFails) {
  IREE_ASSERT_OK(CreateDenseBuffer(/*handle_offset=*/4,
                                   /*device_address=*/UINT64_MAX - 3));

  VkDeviceAddress device_address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
}

TEST_F(VulkanBufferTest, SparseAddressIncludesSubspanOffset) {
  IREE_ASSERT_OK(
      CreateSparseSubspan(/*byte_offset=*/4, /*device_address=*/0x10000));

  VkDeviceAddress device_address = 0;
  IREE_ASSERT_OK(
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
  EXPECT_EQ(device_address, 0x10004u);
}

TEST_F(VulkanBufferTest, SparseMissingAddressRemainsMissing) {
  IREE_ASSERT_OK(CreateSparseSubspan(/*byte_offset=*/4, /*device_address=*/0));

  VkDeviceAddress device_address = UINT64_MAX;
  IREE_ASSERT_OK(
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
  EXPECT_EQ(device_address, 0u);
}

TEST_F(VulkanBufferTest, SparseAddressOverflowFails) {
  IREE_ASSERT_OK(CreateSparseSubspan(/*byte_offset=*/4,
                                     /*device_address=*/UINT64_MAX - 3));

  VkDeviceAddress device_address = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_buffer_device_address(buffer_, &device_address));
}

}  // namespace
}  // namespace iree::hal::vulkan
