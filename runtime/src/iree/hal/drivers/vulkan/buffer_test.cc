// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/buffer.h"

#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

class VulkanBufferTest : public ::testing::Test {
 protected:
  void TearDown() override { iree_hal_buffer_release(buffer_); }

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
        device_address, /*host_allocation=*/nullptr,
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &buffer_);
  }

  // Borrowed native storage wrapper owned by this fixture.
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

TEST_F(VulkanBufferTest, DenseRangeAlignmentIncludesHandleOffset) {
  IREE_ASSERT_OK(
      CreateDenseBuffer(/*handle_offset=*/2, /*device_address=*/0x10000));

  EXPECT_EQ(iree_hal_vulkan_buffer_range_dword_alignment(
                buffer_, /*local_byte_offset=*/2, /*local_byte_length=*/4),
            IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_ALIGNED);
  EXPECT_EQ(iree_hal_vulkan_buffer_range_dword_alignment(
                buffer_, /*local_byte_offset=*/0, /*local_byte_length=*/4),
            IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNALIGNED);
  EXPECT_EQ(iree_hal_vulkan_buffer_range_dword_alignment(
                buffer_, /*local_byte_offset=*/2, /*local_byte_length=*/3),
            IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNALIGNED);
}

TEST_F(VulkanBufferTest, DenseRangeAlignmentValidatesRange) {
  IREE_ASSERT_OK(
      CreateDenseBuffer(/*handle_offset=*/0, /*device_address=*/0x10000));

  EXPECT_EQ(iree_hal_vulkan_buffer_range_dword_alignment(
                buffer_, /*local_byte_offset=*/16, /*local_byte_length=*/0),
            IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_ALIGNED);
  EXPECT_EQ(iree_hal_vulkan_buffer_range_dword_alignment(
                buffer_, /*local_byte_offset=*/17, /*local_byte_length=*/0),
            IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN);
  EXPECT_EQ(iree_hal_vulkan_buffer_range_dword_alignment(
                buffer_, /*local_byte_offset=*/12, /*local_byte_length=*/8),
            IREE_HAL_VULKAN_BUFFER_RANGE_ALIGNMENT_UNKNOWN);
}

TEST_F(VulkanBufferTest, NativeBindingsPreserveNestedOffsets) {
  IREE_ASSERT_OK(CreateDenseBuffer(4, 0x10000));
  iree_hal_buffer_t* outer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(buffer_, 4, 12, iree_allocator_system(), &outer));
  iree_hal_buffer_t* inner = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(outer, 4, 8, iree_allocator_system(), &inner));
  iree_hal_buffer_release(outer);

  const auto resource = iree_hal_buffer_native_binding(
      inner, {IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE,
              IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER});
  EXPECT_EQ(resource.vulkan.buffer, 0x2000u);
  EXPECT_EQ(resource.vulkan.offset, 12u);
  const auto address = iree_hal_buffer_native_binding(
      inner, {IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS,
              IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS});
  EXPECT_EQ(address.device_address, 0x1000cu);
  VkDeviceAddress packet_address = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_buffer_device_address(inner, &packet_address));
  EXPECT_EQ(packet_address, address.device_address);
  EXPECT_EQ(inner->memory.bindings, buffer_->memory.bindings);
  iree_hal_buffer_release(inner);
}

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

// Native dependencies are replaced to exercise noncoherent imports even on
// machines exposing only coherent memory. The real buffer implementation owns
// construction, view offsets, public mapping leases and native teardown.
struct HostImportCapture {
  // Distinct native mapping alias; I/O must keep using the original allocation.
  std::array<uint8_t, 128> alias = {};
  // Native mapping result used to exercise construction failure.
  VkResult map_result = VK_SUCCESS;
  // Number of native mapping acquisitions.
  int map_count = 0;
  // Number of native mapping releases.
  int unmap_count = 0;
  // Number of native resource destructions.
  int destroy_count = 0;
  // Number of native memory releases.
  int free_count = 0;
  // Cache-maintenance ranges issued against the retained native mapping.
  std::vector<VkMappedMemoryRange> ranges;
};

static VKAPI_ATTR VkResult VKAPI_CALL
CaptureMapMemory(VkDevice device, VkDeviceMemory, VkDeviceSize offset,
                 VkDeviceSize length, VkMemoryMapFlags, void** pointer) {
  auto* capture = reinterpret_cast<HostImportCapture*>(device);
  ++capture->map_count;
  EXPECT_EQ(offset, 0u);
  EXPECT_EQ(length, capture->alias.size());
  if (capture->map_result == VK_SUCCESS) {
    *pointer = capture->alias.data();
  }
  return capture->map_result;
}

static VKAPI_ATTR void VKAPI_CALL CaptureUnmapMemory(VkDevice device,
                                                     VkDeviceMemory) {
  ++reinterpret_cast<HostImportCapture*>(device)->unmap_count;
}

static VKAPI_ATTR void VKAPI_CALL
CaptureDestroyBuffer(VkDevice device, VkBuffer, const VkAllocationCallbacks*) {
  ++reinterpret_cast<HostImportCapture*>(device)->destroy_count;
}

static VKAPI_ATTR void VKAPI_CALL CaptureFreeMemory(
    VkDevice device, VkDeviceMemory, const VkAllocationCallbacks*) {
  ++reinterpret_cast<HostImportCapture*>(device)->free_count;
}

static VKAPI_ATTR VkResult VKAPI_CALL CaptureMaintenance(
    VkDevice device, uint32_t count, const VkMappedMemoryRange* ranges) {
  auto* capture = reinterpret_cast<HostImportCapture*>(device);
  capture->ranges.insert(capture->ranges.end(), ranges, ranges + count);
  return VK_SUCCESS;
}

static iree_hal_vulkan_device_syms_t HostImportSyms() {
  return {
      .vkDestroyBuffer = CaptureDestroyBuffer,
      .vkFreeMemory = CaptureFreeMemory,
      .vkMapMemory = CaptureMapMemory,
      .vkUnmapMemory = CaptureUnmapMemory,
      .vkInvalidateMappedMemoryRanges = CaptureMaintenance,
      .vkFlushMappedMemoryRanges = CaptureMaintenance,
  };
}

static iree_status_t CreateHostImport(HostImportCapture* capture, void* pointer,
                                      bool coherent,
                                      iree_hal_buffer_t** out_buffer) {
  const auto syms = HostImportSyms();
  return iree_hal_vulkan_buffer_create(
      &syms, reinterpret_cast<VkDevice>(capture),
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      128, 128,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          (coherent ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0),
      0, 16, reinterpret_cast<VkDeviceMemory>(static_cast<uintptr_t>(0x1000)),
      0, 128, reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(0x2000)), 128,
      0, 0, pointer, /*allocation_trace=*/nullptr,
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      out_buffer);
}

TEST(VulkanHostImportTest,
     KeepsOriginalAddressAndPreparedMaintenanceAcrossViews) {
  for (const bool coherent : {false, true}) {
    SCOPED_TRACE(coherent);
    HostImportCapture capture;
    std::array<uint8_t, 128> original = {};
    iree_hal_buffer_t* owner = nullptr;
    IREE_ASSERT_OK(
        CreateHostImport(&capture, original.data(), coherent, &owner));
    EXPECT_EQ(capture.map_count, coherent ? 0 : 1);
    iree_hal_buffer_t* view = nullptr;
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(owner, 17, 64, iree_allocator_system(), &view));
    iree_hal_buffer_t* nested = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(view, 11, 32,
                                           iree_allocator_system(), &nested));
    iree_hal_buffer_release(view);
    iree_hal_vulkan_buffer_host_span_t span;
    IREE_ASSERT_OK(
        iree_hal_vulkan_buffer_resolve_host_span(nested, 3, 8, &span));
    ASSERT_EQ(span.contents.data, original.data() + 31);
    span.contents.data[0] = 0xA5;
    EXPECT_EQ(original[31], 0xA5);
    EXPECT_EQ(capture.alias[31], 0);
    IREE_ASSERT_OK(iree_hal_vulkan_buffer_flush_host_span(&span));
    IREE_ASSERT_OK(iree_hal_vulkan_buffer_invalidate_host_span(&span));
    EXPECT_EQ(capture.map_count, coherent ? 0 : 1);
    ASSERT_EQ(capture.ranges.size(), coherent ? 0u : 2u);
    for (const auto& range : capture.ranges) {
      EXPECT_EQ(range.offset, 16u);
      EXPECT_EQ(range.size, 32u);
    }
    iree_hal_buffer_mapping_t mapping = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        owner, IREE_HAL_MAPPING_MODE_PERSISTENT, IREE_HAL_MEMORY_ACCESS_READ,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 128, &mapping));
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    EXPECT_EQ(capture.map_count, 1);
    EXPECT_EQ(capture.unmap_count, coherent ? 1 : 0);
    iree_hal_buffer_release(nested);
    iree_hal_buffer_release(owner);
    EXPECT_EQ(capture.unmap_count, 1);
    EXPECT_EQ(capture.destroy_count, 1);
    EXPECT_EQ(capture.free_count, 1);
  }
}

TEST(VulkanHostImportTest, FailedMaintenanceMappingDoesNotTakeNativeOwnership) {
  HostImportCapture capture;
  capture.map_result = VK_ERROR_MEMORY_MAP_FAILED;
  std::array<uint8_t, 128> original = {};
  iree_hal_buffer_t* buffer = nullptr;
  Status status(CreateHostImport(&capture, original.data(), false, &buffer));
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(capture.map_count, 1);
  EXPECT_EQ(capture.unmap_count, 0);
  EXPECT_EQ(capture.destroy_count, 0);
  EXPECT_EQ(capture.free_count, 0);
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::vulkan
