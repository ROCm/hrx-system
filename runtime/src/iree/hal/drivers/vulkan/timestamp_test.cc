// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/timestamp.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC
namespace iree::hal::vulkan {
namespace {

// Replaces only native Vulkan recording. The production timestamp component
// owns stage lowering, query delivery, and synchronization.
struct Capture {
  // Native command sequence, including dependencies and compute delivery.
  std::vector<char> commands;
  // Single capture stage required by Vulkan's timestamp API.
  VkPipelineStageFlags2 capture_stage = 0;
  // Destination of the native query copy.
  VkDeviceSize copy_offset = 0;
  // Global execution dependencies emitted after delivery.
  std::vector<VkMemoryBarrier2> execution;
  // Private resolve-buffer dependency emitted before shader consumption.
  std::vector<VkBufferMemoryBarrier2> memory;
  // Shader source address and address of the published destination pointer.
  std::array<uint64_t, 2> constants = {};

  static Capture& Get(VkCommandBuffer commands) {
    return *reinterpret_cast<Capture*>(commands);
  }
  static VKAPI_ATTR void VKAPI_CALL Write(VkCommandBuffer commands,
                                          VkPipelineStageFlags2 stage,
                                          VkQueryPool, uint32_t query) {
    auto& capture = Get(commands);
    capture.commands.push_back('T');
    capture.capture_stage = stage;
    EXPECT_EQ(query, 3u);
  }
  static VKAPI_ATTR void VKAPI_CALL Copy(VkCommandBuffer commands, VkQueryPool,
                                         uint32_t first, uint32_t count,
                                         VkBuffer, VkDeviceSize offset,
                                         VkDeviceSize stride,
                                         VkQueryResultFlags flags) {
    auto& capture = Get(commands);
    capture.commands.push_back('C');
    capture.copy_offset = offset;
    EXPECT_EQ(first, 3u);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(stride, 8u);
    EXPECT_EQ(flags, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  }
  static VKAPI_ATTR void VKAPI_CALL
  Barrier(VkCommandBuffer commands, const VkDependencyInfo* dependency) {
    auto& capture = Get(commands);
    capture.commands.push_back('B');
    for (uint32_t i = 0; i < dependency->memoryBarrierCount; ++i) {
      capture.execution.push_back(dependency->pMemoryBarriers[i]);
    }
    for (uint32_t i = 0; i < dependency->bufferMemoryBarrierCount; ++i) {
      capture.memory.push_back(dependency->pBufferMemoryBarriers[i]);
    }
  }
  static VKAPI_ATTR void VKAPI_CALL Bind(VkCommandBuffer commands,
                                         VkPipelineBindPoint point,
                                         VkPipeline) {
    Get(commands).commands.push_back('P');
    EXPECT_EQ(point, VK_PIPELINE_BIND_POINT_COMPUTE);
  }
  static VKAPI_ATTR void VKAPI_CALL Push(VkCommandBuffer commands,
                                         VkPipelineLayout,
                                         VkShaderStageFlags stages,
                                         uint32_t offset, uint32_t size,
                                         const void* data) {
    auto& capture = Get(commands);
    capture.commands.push_back('A');
    EXPECT_EQ(stages, VK_SHADER_STAGE_COMPUTE_BIT);
    EXPECT_EQ(offset, 0u);
    ASSERT_EQ(size, sizeof(capture.constants));
    std::memcpy(capture.constants.data(), data, size);
  }
  static VKAPI_ATTR void VKAPI_CALL Dispatch(VkCommandBuffer commands,
                                             uint32_t x, uint32_t y,
                                             uint32_t z) {
    Get(commands).commands.push_back('D');
    EXPECT_EQ(x, 1u);
    EXPECT_EQ(y, 1u);
    EXPECT_EQ(z, 1u);
  }
  static iree_hal_vulkan_device_syms_t Syms() {
    return {
        .vkCmdWriteTimestamp2 = Write,
        .vkCmdCopyQueryPoolResults = Copy,
        .vkCmdPipelineBarrier2 = Barrier,
        .vkCmdBindPipeline = Bind,
        .vkCmdPushConstants = Push,
        .vkCmdDispatch = Dispatch,
    };
  }
};

TEST(VulkanTimestampTest, CaptureScopesAndExecutionOnlyDelivery) {
  const auto syms = Capture::Syms();
  const iree_hal_execution_stage_t sources[] = {
      0,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
      IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_ATOMIC,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_ATOMIC,
      IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
  };
  const VkPipelineStageFlags2 stages[] = {
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  for (size_t i = 0; i < IREE_ARRAYSIZE(sources); ++i) {
    SCOPED_TRACE(sources[i]);
    for (bool ordered : {false, true}) {
      Capture capture;
      iree_hal_vulkan_timestamp_record_direct(
          &syms, VK_QUEUE_COMPUTE_BIT,
          reinterpret_cast<VkCommandBuffer>(&capture), VK_NULL_HANDLE, 3,
          sources[i], ordered ? IREE_HAL_EXECUTION_STAGE_TRANSFER : 0,
          VK_NULL_HANDLE, 24);
      EXPECT_EQ(capture.commands, ordered ? std::vector<char>({'T', 'C', 'B'})
                                          : std::vector<char>({'T', 'C'}));
      EXPECT_EQ(capture.capture_stage, stages[i]);
      EXPECT_EQ(capture.copy_offset, 24u);
      EXPECT_TRUE(capture.memory.empty());
      if (ordered) {
        ASSERT_EQ(capture.execution.size(), 1u);
        EXPECT_EQ(capture.execution[0].srcStageMask,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT);
        EXPECT_EQ(capture.execution[0].dstStageMask,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        EXPECT_EQ(capture.execution[0].srcAccessMask, 0u);
        EXPECT_EQ(capture.execution[0].dstAccessMask, 0u);
      }
    }
  }
}

TEST(VulkanTimestampTest, IndirectDeliveryMakesOnlyPrivateScratchVisible) {
  const auto syms = Capture::Syms();
  const iree_hal_vulkan_timestamp_pipeline_t pipeline = {};
  const iree_hal_vulkan_timestamp_state_t state = {
      .resolve_offset = 64,
      .resolve_address = 0x1000,
      .pipeline = &pipeline,
  };
  Capture capture;
  iree_hal_vulkan_timestamp_record_indirect(
      &syms, VK_QUEUE_COMPUTE_BIT, reinterpret_cast<VkCommandBuffer>(&capture),
      &state, 3, 2, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, 0x2000);
  EXPECT_EQ(capture.commands,
            std::vector<char>({'T', 'C', 'B', 'P', 'A', 'D', 'B'}));
  EXPECT_EQ(capture.copy_offset, 80u);
  EXPECT_EQ(capture.constants, (std::array<uint64_t, 2>{0x1010, 0x2000}));
  ASSERT_EQ(capture.memory.size(), 1u);
  EXPECT_EQ(capture.memory[0].offset, 80u);
  EXPECT_EQ(capture.memory[0].size, 8u);
  EXPECT_EQ(capture.memory[0].srcStageMask, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
  EXPECT_EQ(capture.memory[0].srcAccessMask, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  EXPECT_EQ(capture.memory[0].dstStageMask,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(capture.memory[0].dstAccessMask, VK_ACCESS_2_SHADER_READ_BIT);
  ASSERT_EQ(capture.execution.size(), 1u);
  EXPECT_EQ(capture.execution[0].srcStageMask,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(capture.execution[0].dstStageMask,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  EXPECT_EQ(capture.execution[0].srcAccessMask, 0u);
  EXPECT_EQ(capture.execution[0].dstAccessMask, 0u);
}

}  // namespace
}  // namespace iree::hal::vulkan
#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC
