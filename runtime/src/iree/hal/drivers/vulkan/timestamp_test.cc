// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/timestamp.h"

#include <array>
#include <cstring>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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
  // Query range and byte offset of each native delivery batch.
  std::vector<std::array<uint64_t, 3>> copies;
  // Global execution dependencies around capture.
  std::vector<VkMemoryBarrier2> execution;
  // Private resolve-buffer dependency emitted before shader consumption.
  std::vector<VkBufferMemoryBarrier2> memory;
  // Shader source address and address of the published destination pointer.
  std::array<uint64_t, 3> constants = {};

  static Capture& Get(VkCommandBuffer commands) {
    return *reinterpret_cast<Capture*>(commands);
  }
  static VKAPI_ATTR void VKAPI_CALL Write(VkCommandBuffer commands,
                                          VkPipelineStageFlags2 stage,
                                          VkQueryPool, uint32_t query) {
    auto& capture = Get(commands);
    capture.commands.push_back('T');
    capture.capture_stage = stage;
    (void)query;
  }
  static VKAPI_ATTR void VKAPI_CALL Copy(VkCommandBuffer commands, VkQueryPool,
                                         uint32_t first, uint32_t count,
                                         VkBuffer, VkDeviceSize offset,
                                         VkDeviceSize stride,
                                         VkQueryResultFlags flags) {
    auto& capture = Get(commands);
    capture.commands.push_back('C');
    capture.copies.push_back({first, count, offset});
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

TEST(VulkanTimestampTest, CaptureOrdersOnlyRequestedStagesAndNeverDelivers) {
  const auto syms = Capture::Syms();
  const std::pair<iree_hal_execution_stage_t, VkPipelineStageFlags2> stages[] =
      {
          {0, 0},
          {IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT},
          {IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
          {IREE_HAL_EXECUTION_STAGE_DISPATCH,
           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
          {IREE_HAL_EXECUTION_STAGE_ATOMIC,
           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
          {IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_ATOMIC,
           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
          {IREE_HAL_EXECUTION_STAGE_TRANSFER,
           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
               VK_PIPELINE_STAGE_2_TRANSFER_BIT |
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
          {IREE_HAL_EXECUTION_STAGE_DISPATCH |
               IREE_HAL_EXECUTION_STAGE_TRANSFER,
           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
               VK_PIPELINE_STAGE_2_TRANSFER_BIT |
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
          {IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
           VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT},
      };
  for (auto [source, native_source] : stages) {
    SCOPED_TRACE(source);
    for (auto [target, native_target] : stages) {
      SCOPED_TRACE(target);
      Capture capture;
      iree_hal_vulkan_timestamp_record_capture(
          &syms, VK_QUEUE_COMPUTE_BIT,
          reinterpret_cast<VkCommandBuffer>(&capture), VK_NULL_HANDLE, 3,
          source, target);
      std::vector<char> expected;
      if (source & ~IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE) {
        expected.push_back('B');
      }
      expected.push_back('T');
      if (target) {
        expected.push_back('B');
      }
      EXPECT_EQ(capture.commands, expected);
      EXPECT_EQ(capture.capture_stage, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
      EXPECT_TRUE(capture.copies.empty());
      EXPECT_TRUE(capture.memory.empty());
      for (const auto& dependency : capture.execution) {
        EXPECT_EQ(dependency.srcAccessMask, 0u);
        EXPECT_EQ(dependency.dstAccessMask, 0u);
      }
      if (source & ~IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE) {
        EXPECT_EQ(capture.execution.front().srcStageMask, native_source);
        EXPECT_EQ(capture.execution.front().dstStageMask,
                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      }
      if (target) {
        EXPECT_EQ(capture.execution.back().srcStageMask,
                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        EXPECT_EQ(capture.execution.back().dstStageMask, native_target);
      }
    }
  }
}

TEST(VulkanTimestampTest, DeliveryBatchesContiguousQueriesAndPreservesHoles) {
  const auto syms = Capture::Syms();
  const iree_hal_vulkan_timestamp_pipeline_t pipeline = {};
  // Native handles are opaque to the recording-only Vulkan dependency.
  const auto handle = reinterpret_cast<VkBuffer>(uintptr_t{1});
  std::array<uint8_t, 64> storage = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
      storage.size(), iree_make_byte_span(storage.data(), storage.size()),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &buffer));
  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, 8, 32, iree_allocator_system(),
                                         &subspan));
  // A second import can name the same bytes without sharing a HAL allocation
  // object or using the same byte coordinates.
  iree_hal_buffer_t* imported_alias = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER, 32,
      iree_make_byte_span(storage.data() + 8, 32),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &imported_alias));
  for (bool indirect : {false, true}) {
    SCOPED_TRACE(indirect);
    std::array<iree_hal_vulkan_timestamp_pending_t, 4> pending = {};
    for (uint32_t i = 0; i < pending.size(); ++i) {
      pending[i] = {
          .buffer = buffer,
          .offset = i * 8,
          .target_handle = indirect ? VK_NULL_HANDLE : handle,
          .target_offset = (indirect ? 0 : 64) + i * 8,
      };
    }
    const iree_hal_vulkan_timestamp_state_t state = {
        .resolve_offset = 64,
        .resolve_address = 0x1000,
        .pipeline = &pipeline,
        .pending = pending.data(),
    };
    iree_hal_vulkan_timestamp_recording_t recording = {
        .state = &state,
        .target_table = 0x2000,
        .count = 4,
    };
    Capture capture;
    const auto commands = reinterpret_cast<VkCommandBuffer>(&capture);
    // An unrelated range creates no query work. A partial overlap through an
    // aliasing subspan then delivers only queries 1 and 2.
    iree_hal_vulkan_timestamp_record_delivery(
        &syms, commands, &recording, iree_hal_make_buffer_ref(buffer, 40, 8));
    EXPECT_TRUE(capture.commands.empty());
    iree_hal_vulkan_timestamp_record_delivery(
        &syms, commands, &recording, iree_hal_make_buffer_ref(subspan, 4, 8));
    ASSERT_EQ(capture.copies.size(), 1u);
    EXPECT_EQ(capture.copies[0], (std::array<uint64_t, 3>{1, 2, 72}));
    EXPECT_EQ(recording.first_pending, 0u);
    EXPECT_EQ(pending[1].buffer, nullptr);
    EXPECT_EQ(pending[2].buffer, nullptr);
    if (indirect) {
      EXPECT_EQ(capture.commands, std::vector<char>({'C', 'B', 'P', 'A', 'D'}));
      EXPECT_EQ(capture.constants,
                (std::array<uint64_t, 3>{0x1008, 0x2008, 2}));
      ASSERT_EQ(capture.memory.size(), 1u);
      EXPECT_EQ(capture.memory[0].offset, 72u);
      EXPECT_EQ(capture.memory[0].size, 16u);
      EXPECT_EQ(capture.memory[0].srcAccessMask,
                VK_ACCESS_2_TRANSFER_WRITE_BIT);
      EXPECT_EQ(capture.memory[0].dstAccessMask, VK_ACCESS_2_SHADER_READ_BIT);
    } else {
      EXPECT_EQ(capture.commands, std::vector<char>({'C'}));
    }
    EXPECT_TRUE(capture.execution.empty());
    iree_hal_vulkan_timestamp_record_delivery(&syms, commands, &recording, {});
    EXPECT_EQ(capture.copies, (std::vector<std::array<uint64_t, 3>>{
                                  {1, 2, 72}, {0, 1, 64}, {3, 1, 88}}));
    EXPECT_EQ(recording.first_pending, 4u);
    const auto previous = capture.commands;
    iree_hal_vulkan_timestamp_record_delivery(&syms, commands, &recording, {});
    EXPECT_EQ(capture.commands, previous);

    pending[0] = {
        .buffer = buffer,
        .offset = 8,
        .target_handle = indirect ? VK_NULL_HANDLE : handle,
        .target_offset = indirect ? 8u : 72u,
    };
    recording.first_pending = 0;
    recording.count = 1;
    Capture alias_capture;
    const auto alias_commands =
        reinterpret_cast<VkCommandBuffer>(&alias_capture);
    iree_hal_vulkan_timestamp_record_delivery(
        &syms, alias_commands, &recording,
        iree_hal_make_buffer_ref(imported_alias, 0, 0));
    EXPECT_TRUE(alias_capture.commands.empty());
    iree_hal_vulkan_timestamp_record_delivery(
        &syms, alias_commands, &recording,
        iree_hal_make_buffer_ref(imported_alias, 0, 8));
    EXPECT_EQ(alias_capture.commands,
              indirect ? std::vector<char>({'C', 'B', 'P', 'A', 'D'})
                       : std::vector<char>({'C'}));
    EXPECT_EQ(alias_capture.copies,
              (std::vector<std::array<uint64_t, 3>>{{0, 1, 72}}));
    EXPECT_EQ(recording.first_pending, 1u);
  }

  // Mixed recordings use separate query and indirect-result ordinals. The
  // indirect batch starts at query 2 but private result/address-table slot 0.
  std::array<iree_hal_vulkan_timestamp_pending_t, 4> pending = {};
  for (uint32_t i = 0; i < pending.size(); ++i) {
    pending[i] = {
        .buffer = buffer,
        .offset = i * 8,
        .target_handle = i < 2 ? handle : VK_NULL_HANDLE,
        .target_offset = i < 2 ? 64 + i * 8 : (i - 2) * 8,
    };
  }
  const iree_hal_vulkan_timestamp_state_t state = {
      .resolve_offset = 128,
      .resolve_address = 0x1000,
      .pipeline = &pipeline,
      .pending = pending.data(),
  };
  iree_hal_vulkan_timestamp_recording_t recording = {
      .state = &state,
      .target_table = 0x2000,
      .count = 4,
  };
  Capture capture;
  iree_hal_vulkan_timestamp_record_delivery(
      &syms, reinterpret_cast<VkCommandBuffer>(&capture), &recording, {});
  EXPECT_EQ(capture.commands,
            std::vector<char>({'C', 'C', 'B', 'P', 'A', 'D'}));
  EXPECT_EQ(capture.copies,
            (std::vector<std::array<uint64_t, 3>>{{0, 2, 64}, {2, 2, 128}}));
  EXPECT_EQ(capture.constants, (std::array<uint64_t, 3>{0x1000, 0x2000, 2}));
  EXPECT_EQ(recording.first_pending, 4u);
  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(imported_alias);
  iree_hal_buffer_release(buffer);
}

}  // namespace
}  // namespace iree::hal::vulkan
#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC
