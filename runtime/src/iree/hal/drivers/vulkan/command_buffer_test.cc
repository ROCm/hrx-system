// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/command_buffer.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/vulkan/barrier.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

static constexpr iree_hal_queue_priority_t kQueuePriority =
    IREE_HAL_QUEUE_PRIORITY_NORMAL;
static const iree_hal_queue_family_spec_t kQueueFamilySpec = [] {
  iree_hal_queue_family_spec_t spec = {
      .name = IREE_SV("test"),
      .priority_count = 1,
      .priorities = &kQueuePriority,
      .timestamp_valid_bits = 64,
      .timestamp_frequency_hz = 1000000000,
      .physical_device_affinity = 1,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC,
      .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS |
               IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS,
  };
  return spec;
}();

TEST(QueueBarrierTest, ExplicitEmptyReplacesDefaultDependency) {
  auto native = iree_hal_vulkan_queue_barriers_resolve(nullptr);
  EXPECT_FALSE(iree_hal_vulkan_barrier_is_empty(&native.before));
  EXPECT_FALSE(iree_hal_vulkan_barrier_is_empty(&native.after));
  const iree_hal_barrier_list_t empty = {};
  iree_hal_queue_barriers_t barriers = {&empty, nullptr};
  native = iree_hal_vulkan_queue_barriers_resolve(&barriers);
  EXPECT_TRUE(iree_hal_vulkan_barrier_is_empty(&native.before));
  EXPECT_FALSE(iree_hal_vulkan_barrier_is_empty(&native.after));
  barriers.after = &empty;
  native = iree_hal_vulkan_queue_barriers_resolve(&barriers);
  EXPECT_TRUE(iree_hal_vulkan_barrier_is_empty(&native.before));
  EXPECT_TRUE(iree_hal_vulkan_barrier_is_empty(&native.after));

  iree_hal_barrier_t acquire = {};
  acquire.effects.bits = IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM;
  iree_hal_barrier_list_t list = {1, &acquire};
  barriers.after = &list;
  native = iree_hal_vulkan_queue_barriers_resolve(&barriers);
  EXPECT_TRUE(iree_hal_vulkan_barrier_is_empty(&native.before));
  EXPECT_TRUE(native.after.target_access_mask & VK_ACCESS_2_MEMORY_READ_BIT);
}

TEST(QueueBarrierTest, RangedAcquirePromotesToGlobalQueueVisibility) {
  const iree_hal_memory_transition_recipe_info_t operation = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM,
      .range_granularity = 64,
  };
  const iree_hal_memory_transition_recipe_t recipe = {
      .effects = {IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM},
      .operation_count = 1,
      .operations = &operation,
  };
  const iree_hal_buffer_barrier_t buffer_barrier = {
      .source_scope = 0,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
      .buffer_ref = {},
      .recipe = &recipe,
  };
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = 0,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_NONE,
      .effects = recipe.effects,
      .memory_barrier_count = 0,
      .memory_barriers = nullptr,
      .buffer_barrier_count = 1,
      .buffer_barriers = &buffer_barrier,
  };
  const iree_hal_barrier_list_t list = {1, &barrier};
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_queue_barriers_t barriers = {&list, &empty};
  const auto native = iree_hal_vulkan_queue_barriers_resolve(&barriers);
  EXPECT_NE(native.before.source_stage_mask & IREE_HAL_EXECUTION_STAGE_HOST,
            0u);
  EXPECT_NE(native.before.source_access_mask & VK_ACCESS_2_HOST_WRITE_BIT, 0u);
  EXPECT_NE(native.before.target_access_mask & VK_ACCESS_2_SHADER_READ_BIT, 0u);
  EXPECT_TRUE(iree_hal_vulkan_barrier_is_empty(&native.after));
}

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

struct NativeReplayCapture {
  // Count of fake vkBeginCommandBuffer calls.
  int begin_command_buffer_count = 0;

  // Count of fake vkEndCommandBuffer calls.
  int end_command_buffer_count = 0;

  // Count of fake vkCmdBeginDebugUtilsLabelEXT calls.
  int begin_label_count = 0;

  // Count of fake vkCmdEndDebugUtilsLabelEXT calls.
  int end_label_count = 0;

  // Count of fake vkCmdPipelineBarrier2 calls.
  int pipeline_barrier_count = 0;

  // Count of fake vkCmdPushConstants calls.
  int push_constants_count = 0;

  // Count of fake vkCmdBindPipeline calls.
  int bind_pipeline_count = 0;

  // Count of fake vkCmdDispatch calls.
  int dispatch_count = 0;

  // Native ordering of captures, result copies, barriers, and dispatches.
  std::vector<std::pair<char, uint32_t>> events;

  // Query range and destination offset of each delivery batch.
  std::vector<std::array<uint64_t, 3>> query_copies;

  // VkCommandBuffer passed to vkBeginCommandBuffer.
  VkCommandBuffer begin_command_buffer = VK_NULL_HANDLE;

  // VkCommandBuffer passed to vkEndCommandBuffer.
  VkCommandBuffer end_command_buffer = VK_NULL_HANDLE;

  // VkCommandBuffer passed to vkCmdBeginDebugUtilsLabelEXT.
  VkCommandBuffer begin_label_command_buffer = VK_NULL_HANDLE;

  // VkCommandBuffer passed to vkCmdEndDebugUtilsLabelEXT.
  VkCommandBuffer end_label_command_buffer = VK_NULL_HANDLE;

  // Label copied during the fake begin-label entry-point call.
  std::string label;

  // RGBA color copied during the fake begin-label entry-point call.
  float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Memory barriers copied during fake pipeline-barrier entry-point calls.
  std::array<VkMemoryBarrier2, 16> memory_barriers = {};

  // Number of captured global memory barriers.
  uint32_t memory_barrier_count = 0;

  // Buffer barriers copied during fake pipeline-barrier entry-point calls.
  std::array<VkBufferMemoryBarrier2, 4> buffer_barriers = {};

  // Number of captured buffer memory barriers.
  uint32_t buffer_barrier_count = 0;

  // Push constant bytes copied during the fake entry-point call.
  std::array<uint8_t, 32> push_constants = {};

  // Number of valid bytes in push_constants.
  uint32_t push_constants_length = 0;

  // Pipeline passed to the fake bind-pipeline entry-point call.
  VkPipeline pipeline = VK_NULL_HANDLE;
};

static NativeReplayCapture* g_native_replay_capture = nullptr;

static VKAPI_ATTR VkResult VKAPI_CALL
FakeBeginCommandBuffer(VkCommandBuffer command_buffer,
                       const VkCommandBufferBeginInfo* begin_info) {
  ++g_native_replay_capture->begin_command_buffer_count;
  g_native_replay_capture->begin_command_buffer = command_buffer;
  EXPECT_EQ(0u, begin_info->flags);
  return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
FakeEndCommandBuffer(VkCommandBuffer command_buffer) {
  ++g_native_replay_capture->end_command_buffer_count;
  g_native_replay_capture->end_command_buffer = command_buffer;
  return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdBeginDebugUtilsLabelEXT(
    VkCommandBuffer command_buffer, const VkDebugUtilsLabelEXT* label_info) {
  ++g_native_replay_capture->begin_label_count;
  g_native_replay_capture->begin_label_command_buffer = command_buffer;
  g_native_replay_capture->label = label_info->pLabelName;
  for (int i = 0; i < 4; ++i) {
    g_native_replay_capture->color[i] = label_info->color[i];
  }
}

static VKAPI_ATTR void VKAPI_CALL
FakeCmdEndDebugUtilsLabelEXT(VkCommandBuffer command_buffer) {
  ++g_native_replay_capture->end_label_count;
  g_native_replay_capture->end_label_command_buffer = command_buffer;
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdPipelineBarrier2(
    VkCommandBuffer command_buffer, const VkDependencyInfo* dependency_info) {
  (void)command_buffer;
  ++g_native_replay_capture->pipeline_barrier_count;
  g_native_replay_capture->events.emplace_back('B', 0);
  ASSERT_EQ(dependency_info->memoryBarrierCount +
                dependency_info->bufferMemoryBarrierCount,
            1u);
  if (dependency_info->memoryBarrierCount) {
    ASSERT_LT(g_native_replay_capture->memory_barrier_count,
              g_native_replay_capture->memory_barriers.size());
    ASSERT_NE(dependency_info->pMemoryBarriers, nullptr);
    g_native_replay_capture
        ->memory_barriers[g_native_replay_capture->memory_barrier_count++] =
        dependency_info->pMemoryBarriers[0];
  } else {
    ASSERT_LT(g_native_replay_capture->buffer_barrier_count,
              g_native_replay_capture->buffer_barriers.size());
    ASSERT_NE(dependency_info->pBufferMemoryBarriers, nullptr);
    g_native_replay_capture
        ->buffer_barriers[g_native_replay_capture->buffer_barrier_count++] =
        dependency_info->pBufferMemoryBarriers[0];
  }
}

static VKAPI_ATTR void VKAPI_CALL
FakeCmdPushConstants(VkCommandBuffer command_buffer, VkPipelineLayout layout,
                     VkShaderStageFlags stage_flags, uint32_t offset,
                     uint32_t size, const void* values) {
  (void)command_buffer;
  (void)layout;
  EXPECT_EQ(stage_flags, VK_SHADER_STAGE_COMPUTE_BIT);
  EXPECT_EQ(offset, 0u);
  ASSERT_LE(size, g_native_replay_capture->push_constants.size());
  ++g_native_replay_capture->push_constants_count;
  g_native_replay_capture->push_constants_length = size;
  memcpy(g_native_replay_capture->push_constants.data(), values, size);
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdBindPipeline(
    VkCommandBuffer command_buffer, VkPipelineBindPoint pipeline_bind_point,
    VkPipeline pipeline) {
  (void)command_buffer;
  EXPECT_EQ(pipeline_bind_point, VK_PIPELINE_BIND_POINT_COMPUTE);
  ++g_native_replay_capture->bind_pipeline_count;
  g_native_replay_capture->pipeline = pipeline;
}

static VKAPI_ATTR void VKAPI_CALL
FakeCmdDispatch(VkCommandBuffer command_buffer, uint32_t group_count_x,
                uint32_t group_count_y, uint32_t group_count_z) {
  (void)command_buffer;
  EXPECT_EQ(group_count_x, 1u);
  EXPECT_EQ(group_count_y, 1u);
  EXPECT_EQ(group_count_z, 1u);
  ++g_native_replay_capture->dispatch_count;
  g_native_replay_capture->events.emplace_back('D', 0);
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdInsertDebugUtilsLabelEXT(
    VkCommandBuffer command_buffer, const VkDebugUtilsLabelEXT* label_info) {
  (void)command_buffer;
  (void)label_info;
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdResetQueryPool(VkCommandBuffer,
                                                        VkQueryPool,
                                                        uint32_t first,
                                                        uint32_t count) {
  g_native_replay_capture->events.emplace_back('R', count);
  EXPECT_EQ(first, 0u);
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdWriteTimestamp2(
    VkCommandBuffer, VkPipelineStageFlags2 stage, VkQueryPool, uint32_t query) {
  g_native_replay_capture->events.emplace_back('T', query);
  EXPECT_EQ(stage, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdCopyQueryPoolResults(
    VkCommandBuffer, VkQueryPool, uint32_t first, uint32_t count, VkBuffer,
    VkDeviceSize offset, VkDeviceSize stride, VkQueryResultFlags flags) {
  g_native_replay_capture->events.emplace_back('C', first);
  g_native_replay_capture->query_copies.push_back({first, count, offset});
  EXPECT_EQ(stride, 8u);
  EXPECT_EQ(flags, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

static iree_hal_vulkan_device_syms_t MakeNativeReplaySyms() {
  iree_hal_vulkan_device_syms_t syms = {
      .vkBeginCommandBuffer = FakeBeginCommandBuffer,
      .vkEndCommandBuffer = FakeEndCommandBuffer,
      .vkCmdBeginDebugUtilsLabelEXT = FakeCmdBeginDebugUtilsLabelEXT,
      .vkCmdEndDebugUtilsLabelEXT = FakeCmdEndDebugUtilsLabelEXT,
      .vkCmdInsertDebugUtilsLabelEXT = FakeCmdInsertDebugUtilsLabelEXT,
      .vkCmdResetQueryPool = FakeCmdResetQueryPool,
      .vkCmdWriteTimestamp2 = FakeCmdWriteTimestamp2,
      .vkCmdCopyQueryPoolResults = FakeCmdCopyQueryPoolResults,
      .vkCmdPipelineBarrier2 = FakeCmdPipelineBarrier2,
      .vkCmdBindPipeline = FakeCmdBindPipeline,
      .vkCmdPushConstants = FakeCmdPushConstants,
      .vkCmdDispatch = FakeCmdDispatch,
  };
  return syms;
}

TEST(QueueBarrierTest, ExecutionStagesComposeWithoutInventingDependencies) {
  const auto syms = MakeNativeReplaySyms();
  const auto command_buffer = reinterpret_cast<VkCommandBuffer>(uintptr_t{1});
  const std::pair<iree_hal_execution_stage_t, VkPipelineStageFlags2> cases[] = {
      {0, VK_PIPELINE_STAGE_2_NONE},
      {IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT},
      {IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
       VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT},
      {IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
           IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
           VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT},
      {IREE_HAL_EXECUTION_STAGE_DISPATCH |
           IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
           VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT},
      {IREE_HAL_EXECUTION_STAGE_TRANSFER |
           IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
       VK_PIPELINE_STAGE_2_TRANSFER_BIT |
           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT},
  };
  for (auto [stages, native_stages] : cases) {
    const iree_hal_vulkan_barrier_t barrier = {
        .source_stage_mask = stages,
        .target_stage_mask = stages | IREE_HAL_EXECUTION_STAGE_HOST,
    };
    NativeReplayCapture capture;
    g_native_replay_capture = &capture;
    iree_hal_vulkan_barrier_record(&syms, VK_QUEUE_COMPUTE_BIT, command_buffer,
                                   &barrier);
    g_native_replay_capture = nullptr;
    ASSERT_EQ(capture.memory_barrier_count, 1u);
    EXPECT_EQ(capture.memory_barriers[0].srcStageMask, native_stages);
    EXPECT_EQ(capture.memory_barriers[0].dstStageMask,
              native_stages | VK_PIPELINE_STAGE_2_HOST_BIT);
    EXPECT_EQ(capture.memory_barriers[0].srcAccessMask, 0u);
    EXPECT_EQ(capture.memory_barriers[0].dstAccessMask, 0u);
  }
}

TEST(QueueBarrierTest, TransferScopesCoverOnlyAvailableNativeEngines) {
  const auto syms = MakeNativeReplaySyms();
  const auto command_buffer = reinterpret_cast<VkCommandBuffer>(uintptr_t{1});
  const auto buffer = reinterpret_cast<VkBuffer>(uintptr_t{2});
  for (VkQueueFlags queue_flags : {VkQueueFlags{VK_QUEUE_TRANSFER_BIT},
                                   VkQueueFlags{VK_QUEUE_COMPUTE_BIT}}) {
    const bool has_compute = (queue_flags & VK_QUEUE_COMPUTE_BIT) != 0;
    const VkPipelineStageFlags2 stages =
        VK_PIPELINE_STAGE_2_TRANSFER_BIT |
        (has_compute ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : 0);
    for (VkAccessFlags2 target_access :
         {VkAccessFlags2{0}, VK_ACCESS_2_TRANSFER_READ_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT}) {
      const iree_hal_vulkan_barrier_t barrier = {
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .source_access_mask =
              target_access ? VK_ACCESS_2_TRANSFER_WRITE_BIT : 0,
          .target_access_mask = target_access,
      };
      const VkAccessFlags2 writes =
          target_access ? VK_ACCESS_2_TRANSFER_WRITE_BIT |
                              (has_compute ? VK_ACCESS_2_SHADER_WRITE_BIT : 0)
                        : 0;
      VkAccessFlags2 accesses = target_access;
      if (target_access && has_compute) {
        accesses |=
            target_access == VK_ACCESS_2_TRANSFER_READ_BIT
                ? VK_ACCESS_2_SHADER_READ_BIT
                : VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      }
      NativeReplayCapture capture;
      g_native_replay_capture = &capture;
      iree_hal_vulkan_barrier_record(&syms, queue_flags, command_buffer,
                                     &barrier);
      iree_hal_vulkan_buffer_barrier_record(&syms, queue_flags, command_buffer,
                                            &barrier, buffer, 8, 16);
      g_native_replay_capture = nullptr;
      ASSERT_EQ(capture.pipeline_barrier_count, 2);
      ASSERT_EQ(capture.memory_barrier_count, 1u);
      ASSERT_EQ(capture.buffer_barrier_count, 1u);
      const auto& global = capture.memory_barriers[0];
      EXPECT_EQ(global.srcStageMask, stages);
      EXPECT_EQ(global.dstStageMask, stages);
      EXPECT_EQ(global.srcAccessMask, writes);
      EXPECT_EQ(global.dstAccessMask, accesses);
      const auto& ranged = capture.buffer_barriers[0];
      EXPECT_EQ(ranged.srcStageMask, stages);
      EXPECT_EQ(ranged.dstStageMask, stages);
      EXPECT_EQ(ranged.srcAccessMask, writes);
      EXPECT_EQ(ranged.dstAccessMask, accesses);
      EXPECT_EQ(ranged.srcQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
      EXPECT_EQ(ranged.dstQueueFamilyIndex, VK_QUEUE_FAMILY_IGNORED);
      EXPECT_EQ(ranged.buffer, buffer);
      EXPECT_EQ(ranged.offset, 8u);
      EXPECT_EQ(ranged.size, 16u);
    }
  }
}

struct CommandBufferDeleter {
  void operator()(iree_hal_command_buffer_t* command_buffer) const {
    iree_hal_command_buffer_release(command_buffer);
  }
};

using CommandBufferPtr =
    std::unique_ptr<iree_hal_command_buffer_t, CommandBufferDeleter>;

class VulkanCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(block_size_, iree_allocator_system(),
                                     &block_pool_);
    iree_hal_device_queue_spec_t queues = {
        .family_count = 1,
        .families = &kQueueFamilySpec,
    };
    iree_hal_device_spec_params_t spec_params = {
        .queues = &queues,
    };
    iree_hal_device_spec_t* device_spec = nullptr;
    IREE_ASSERT_OK(iree_hal_device_spec_create(
        &spec_params, iree_allocator_system(), &device_spec));
    iree_hal_mock_device_options_t device_options;
    iree_hal_mock_device_options_initialize(&device_options);
    device_options.device_spec = device_spec;
    iree_status_t status = iree_hal_mock_device_create(
        &device_options, iree_allocator_system(), &device_);
    iree_hal_device_spec_release(device_spec);
    IREE_ASSERT_OK(status);
    queue_family_ = iree_hal_device_queue_family(device_, 0);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(iree_allocator_system(),
                                                  &device_allocator_));
    atomic_pipelines_.pipeline_layout =
        reinterpret_cast<VkPipelineLayout>(static_cast<uintptr_t>(0x1111));
    atomic_pipelines_.pipeline_32 =
        reinterpret_cast<VkPipeline>(static_cast<uintptr_t>(0x2222));
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    iree_hal_allocator_release(device_allocator_);
    iree_hal_device_release(device_);
  }

  CommandBufferPtr CreateCommandBuffer(
      iree_host_size_t binding_capacity = 0,
      iree_hal_command_buffer_mode_t mode =
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT) {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_vulkan_command_buffer_create(
        device_allocator_, queue_family_,
        VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, mode,
        IREE_HAL_COMMAND_CATEGORY_ANY, binding_capacity, &atomic_pipelines_,
        &block_pool_, iree_allocator_system(), &command_buffer));
    return CommandBufferPtr(command_buffer);
  }

 private:
  // Canonical family identity borrowed by command buffers in this fixture.
  const iree_hal_queue_family_t* queue_family_ = nullptr;
  // Device owning the canonical family used by this fixture.
  iree_hal_device_t* device_ = nullptr;

  // Test allocator borrowed by command buffers for validation.
  iree_hal_allocator_t* device_allocator_ = nullptr;

  // Command-buffer block size used by this fixture.
  iree_host_size_t block_size_ = 256;

  // Command-payload and resource-set block pool borrowed by command buffers.
  iree_arena_block_pool_t block_pool_;

  // Empty device-owned pipeline table borrowed by command buffers.
  iree_hal_vulkan_atomic_pipelines_t atomic_pipelines_ = {};
};

TEST_F(VulkanCommandBufferTest, FinalAtomicTargetAlignmentFollowsMode) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_vulkan_atomic_validate_target_address(
                            /*target_address=*/0x1001, IREE_HAL_ATOMIC_WIDTH_32,
                            IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_vulkan_atomic_validate_target_address(
                            /*target_address=*/0x1001, IREE_HAL_ATOMIC_WIDTH_32,
                            IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_vulkan_atomic_validate_target_address(
          /*target_address=*/0x1001, static_cast<iree_hal_atomic_width_t>(0),
          IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE));
}

TEST_F(VulkanCommandBufferTest, ReplaysDebugGroupsAsDebugUtilsLabels) {
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);

  iree_hal_label_color_t label_color = iree_hal_label_color_unspecified();
  label_color.r = 0x10;
  label_color.g = 0x20;
  label_color.b = 0x30;
  label_color.a = 0x40;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin_debug_group(
      command_buffer.get(), IREE_SV("dispatch-region"), label_color,
      /*location=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end_debug_group(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  EXPECT_FALSE(iree_hal_vulkan_command_buffer_is_empty(command_buffer.get()));
  EXPECT_TRUE(
      iree_hal_vulkan_command_buffer_has_native_commands(command_buffer.get()));

  NativeReplayCapture capture;
  g_native_replay_capture = &capture;
  iree_hal_vulkan_device_syms_t syms = MakeNativeReplaySyms();
  iree_hal_vulkan_debug_utils_t debug_utils;
  debug_utils.flags = IREE_HAL_VULKAN_DEBUG_UTILS_FLAG_COMMAND_LABELS;
  iree_hal_vulkan_builtins_t builtins = {};
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();
  VkDevice logical_device =
      reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));
  VkCommandBuffer native_command_buffer =
      reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x5678));

  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_record_native(
      command_buffer.get(), &syms, logical_device, &debug_utils, &builtins,
      native_command_buffer, /*usage_flags=*/0, VK_NULL_HANDLE, binding_table,
      /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr,
      /*timestamps=*/nullptr, /*profile_marker=*/nullptr,
      iree_allocator_system()));

  EXPECT_EQ(1, capture.begin_command_buffer_count);
  EXPECT_EQ(1, capture.end_command_buffer_count);
  EXPECT_EQ(1, capture.begin_label_count);
  EXPECT_EQ(1, capture.end_label_count);
  EXPECT_EQ(native_command_buffer, capture.begin_command_buffer);
  EXPECT_EQ(native_command_buffer, capture.end_command_buffer);
  EXPECT_EQ(native_command_buffer, capture.begin_label_command_buffer);
  EXPECT_EQ(native_command_buffer, capture.end_label_command_buffer);
  EXPECT_EQ("dispatch-region", capture.label);
  EXPECT_FLOAT_EQ(16.0f / 255.0f, capture.color[0]);
  EXPECT_FLOAT_EQ(32.0f / 255.0f, capture.color[1]);
  EXPECT_FLOAT_EQ(48.0f / 255.0f, capture.color[2]);
  EXPECT_FLOAT_EQ(64.0f / 255.0f, capture.color[3]);
  g_native_replay_capture = nullptr;
}

TEST_F(VulkanCommandBufferTest, SystemScopeUsesHostMemoryDomain) {
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);

  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  const iree_hal_barrier_t execution_barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
               IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      .effects = {},
      .memory_barrier_count = 1,
      .memory_barriers = &memory_barrier,
      .buffer_barrier_count = 0,
      .buffer_barriers = nullptr,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer.get(),
                                                 &execution_barrier));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  NativeReplayCapture capture;
  g_native_replay_capture = &capture;
  iree_hal_vulkan_device_syms_t syms = MakeNativeReplaySyms();
  iree_hal_vulkan_debug_utils_t debug_utils = {};
  iree_hal_vulkan_builtins_t builtins = {};
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();
  VkDevice logical_device =
      reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));
  VkCommandBuffer native_command_buffer =
      reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x5678));

  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_record_native(
      command_buffer.get(), &syms, logical_device, &debug_utils, &builtins,
      native_command_buffer, /*usage_flags=*/0, VK_NULL_HANDLE, binding_table,
      /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr,
      /*timestamps=*/nullptr, /*profile_marker=*/nullptr,
      iree_allocator_system()));

  EXPECT_EQ(capture.pipeline_barrier_count, 1);
  EXPECT_NE(capture.memory_barriers[0].srcStageMask &
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            0u);
  EXPECT_NE(
      capture.memory_barriers[0].srcStageMask & VK_PIPELINE_STAGE_2_HOST_BIT,
      0u);
  EXPECT_NE(
      capture.memory_barriers[0].srcAccessMask & VK_ACCESS_2_HOST_WRITE_BIT,
      0u);
  EXPECT_NE(capture.memory_barriers[0].dstStageMask &
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            0u);
  EXPECT_NE(
      capture.memory_barriers[0].dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT,
      0u);
  EXPECT_NE(
      capture.memory_barriers[0].dstAccessMask & VK_ACCESS_2_HOST_READ_BIT, 0u);
  g_native_replay_capture = nullptr;
}

TEST_F(VulkanCommandBufferTest,
       GlobalActionsWithoutStageDependenciesCoverLocalCommands) {
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  iree_hal_barrier_t execution_barrier = {};
  execution_barrier.effects.bits =
      IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM |
      IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer.get(),
                                                 &execution_barrier));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  NativeReplayCapture capture;
  g_native_replay_capture = &capture;
  iree_hal_vulkan_device_syms_t syms = MakeNativeReplaySyms();
  iree_hal_vulkan_debug_utils_t debug_utils = {};
  iree_hal_vulkan_builtins_t builtins = {};
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();
  VkDevice logical_device =
      reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));
  VkCommandBuffer native_command_buffer =
      reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x5678));

  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_record_native(
      command_buffer.get(), &syms, logical_device, &debug_utils, &builtins,
      native_command_buffer, /*usage_flags=*/0, VK_NULL_HANDLE, binding_table,
      /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr,
      /*timestamps=*/nullptr, /*profile_marker=*/nullptr,
      iree_allocator_system()));

  EXPECT_EQ(capture.pipeline_barrier_count, 1);
  EXPECT_NE(capture.memory_barriers[0].srcStageMask &
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            0u);
  EXPECT_NE(
      capture.memory_barriers[0].srcStageMask & VK_PIPELINE_STAGE_2_HOST_BIT,
      0u);
  EXPECT_NE(
      capture.memory_barriers[0].srcAccessMask & VK_ACCESS_2_HOST_WRITE_BIT,
      0u);
  EXPECT_NE(capture.memory_barriers[0].dstStageMask &
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            0u);
  EXPECT_NE(
      capture.memory_barriers[0].dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT,
      0u);
  EXPECT_NE(
      capture.memory_barriers[0].dstAccessMask & VK_ACCESS_2_HOST_READ_BIT, 0u);
  g_native_replay_capture = nullptr;
}

TEST_F(VulkanCommandBufferTest, TimestampDeliveryFollowsDataDependencies) {
  const auto syms = MakeNativeReplaySyms();
  const auto handle = reinterpret_cast<VkBuffer>(uintptr_t{0x2468});
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_vulkan_buffer_create_borrowed(
      &syms, reinterpret_cast<VkDevice>(uintptr_t{1}),
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE, 128, 0,
      128, 0, 1, VK_NULL_HANDLE, nullptr, handle, 0x1000, nullptr,
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &buffer));
  const std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
      buffer_owner(buffer, iree_hal_buffer_release);
  enum class References {
    kSameSlot,
    kOtherSlot,
    kDirectCapture,
    kDirectConsumer,
    kDirect
  };
  const std::pair<References, iree_hal_execution_stage_t> cases[] = {
      {References::kSameSlot, IREE_HAL_EXECUTION_STAGE_TRANSFER},
      {References::kOtherSlot, IREE_HAL_EXECUTION_STAGE_TRANSFER},
      {References::kDirectCapture, IREE_HAL_EXECUTION_STAGE_TRANSFER},
      {References::kDirectConsumer, IREE_HAL_EXECUTION_STAGE_TRANSFER},
      {References::kDirect, IREE_HAL_EXECUTION_STAGE_TRANSFER},
      {References::kSameSlot, IREE_HAL_EXECUTION_STAGE_HOST},
      {References::kOtherSlot, IREE_HAL_EXECUTION_STAGE_HOST},
      {References::kDirectCapture, IREE_HAL_EXECUTION_STAGE_HOST},
      {References::kDirectConsumer, IREE_HAL_EXECUTION_STAGE_HOST},
      {References::kDirect, IREE_HAL_EXECUTION_STAGE_HOST},
  };
  for (auto [references, execution_target] : cases) {
    SCOPED_TRACE(static_cast<int>(references));
    SCOPED_TRACE(execution_target);
    const bool direct_capture = references == References::kDirectCapture ||
                                references == References::kDirect;
    const bool direct_consumer = references == References::kDirectConsumer ||
                                 references == References::kDirect;
    const bool alias_slot = references == References::kOtherSlot ||
                            references == References::kDirectCapture;
    const auto capture_ref = [&](uint64_t offset) {
      return direct_capture ? iree_hal_make_buffer_ref(buffer, offset, 8)
                            : iree_hal_make_indirect_buffer_ref(0, offset, 8);
    };
    auto commands =
        CreateCommandBuffer(2, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT);
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands.get()));
    IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(commands.get(), 0, 0,
                                                     capture_ref(0), 0));
    // A wave join orders captures, but does not consume timestamp data.
    const iree_hal_barrier_t execution = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = execution_target,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(commands.get(), &execution));
    IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(commands.get(), 0, 0,
                                                     capture_ref(8), 0));
    const iree_hal_buffer_barrier_t unrelated = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
        .buffer_ref = capture_ref(32),
    };
    iree_hal_barrier_t consume = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .buffer_barrier_count = 1,
        .buffer_barriers = &unrelated,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(commands.get(), &consume));
    const iree_hal_buffer_barrier_t sample = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
        .buffer_ref =
            direct_consumer
                ? iree_hal_make_buffer_ref(buffer, direct_capture ? 0 : 8, 8)
                : iree_hal_make_indirect_buffer_ref(alias_slot ? 1 : 0, 0, 8),
    };
    consume.buffer_barriers = &sample;
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(commands.get(), &consume));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(commands.get()));
    EXPECT_EQ(
        iree_hal_vulkan_command_buffer_native_replay_compatible(commands.get()),
        references == References::kSameSlot);

    // Cross-slot and mixed direct/indirect aliases change between submissions.
    // Resolve exactly the query covered by this invocation's range.
    for (uint64_t alias_offset : {0u, 8u}) {
      SCOPED_TRACE(alias_offset);
      const uint64_t capture_offset =
          references == References::kDirectConsumer ? alias_offset : 0;
      const iree_hal_buffer_binding_t bindings[] = {
          {buffer, capture_offset, 64}, {buffer, alias_offset, 64}};
      std::array<uint64_t, 2> addresses = {};
      const iree_hal_vulkan_command_buffer_bda_publication_t publication = {
          .host_span = iree_make_byte_span(addresses.data(), sizeof(addresses)),
          .device_address = 0x2000,
      };
      std::array<iree_hal_vulkan_timestamp_pending_t, 2> pending = {};
      const iree_hal_vulkan_timestamp_pipeline_t pipeline = {};
      const iree_hal_vulkan_timestamp_state_t timestamps = {
          .resolve_address = 0x3000,
          .pipeline = &pipeline,
          .pending = pending.data(),
      };
      const iree_hal_vulkan_debug_utils_t debug_utils = {};
      const iree_hal_vulkan_builtins_t builtins = {};
      NativeReplayCapture capture;
      g_native_replay_capture = &capture;
      IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_record_native(
          commands.get(), &syms, reinterpret_cast<VkDevice>(uintptr_t{1}),
          &debug_utils, &builtins,
          reinterpret_cast<VkCommandBuffer>(uintptr_t{2}), 0, VK_NULL_HANDLE,
          {2, bindings}, direct_capture ? nullptr : &publication, nullptr,
          &timestamps, nullptr, iree_allocator_system()));
      g_native_replay_capture = nullptr;
      const uint32_t first = references == References::kDirectConsumer
                                 ? 1 - alias_offset / 8
                             : alias_slot ? alias_offset / 8
                                          : 0;
      const std::vector<std::pair<char, uint32_t>> expected_events =
          direct_capture
              ? std::vector<std::pair<char, uint32_t>>{{'R', 2},
                                                       {'T', 0},
                                                       {'B', 0},
                                                       {'T', 1},
                                                       {'B', 0},
                                                       {'C', first},
                                                       {'B', 0},
                                                       {'C', 1 - first}}
              : (std::vector<std::pair<char, uint32_t>>{{'R', 2},
                                                        {'B', 0},
                                                        {'T', 0},
                                                        {'B', 0},
                                                        {'T', 1},
                                                        {'B', 0},
                                                        {'C', first},
                                                        {'B', 0},
                                                        {'D', 0},
                                                        {'B', 0},
                                                        {'C', 1 - first},
                                                        {'B', 0},
                                                        {'D', 0}});
      EXPECT_EQ(capture.events, expected_events);
      EXPECT_EQ(capture.query_copies,
                (std::vector<std::array<uint64_t, 3>>{
                    {first, 1, first * 8}, {1 - first, 1, (1 - first) * 8}}));
      const std::array<uint64_t, 2> expected_addresses =
          direct_capture ? std::array<uint64_t, 2>{}
                         : std::array<uint64_t, 2>{0x1000 + capture_offset,
                                                   0x1008 + capture_offset};
      EXPECT_EQ(addresses, expected_addresses);
      // An ordinary ranged access dependency retains visibility without a
      // recipe.
      ASSERT_EQ(capture.memory_barrier_count, direct_capture ? 3u : 4u);
      EXPECT_NE(capture.memory_barriers[capture.memory_barrier_count - 1]
                    .srcAccessMask,
                0u);
      const size_t execution_index = direct_capture ? 0 : 1;
      EXPECT_NE(capture.memory_barriers[execution_index].srcStageMask &
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                0u);
      EXPECT_EQ(capture.memory_barriers[execution_index].srcAccessMask, 0u);
    }
  }
}

TEST_F(VulkanCommandBufferTest,
       RangedActionCopiesRecipeAndResolvesIndirectBufferAtReplay) {
  CommandBufferPtr command_buffer = CreateCommandBuffer(
      /*binding_capacity=*/1, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT);
  ASSERT_NE(command_buffer, nullptr);

  iree_hal_memory_transition_recipe_info_t operation = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
      .range_granularity = 64,
  };
  const iree_hal_memory_transition_recipe_t recipe = {
      .effects = {IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM},
      .operation_count = 1,
      .operations = &operation,
  };
  const iree_hal_buffer_barrier_t buffer_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_HOST_READ,
      .buffer_ref = iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, /*offset=*/7, /*length=*/13),
      .recipe = &recipe,
  };
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_NONE,
      .effects = recipe.effects,
      .memory_barrier_count = 0,
      .memory_barriers = nullptr,
      .buffer_barrier_count = 1,
      .buffer_barriers = &buffer_barrier,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(
      iree_hal_command_buffer_barrier(command_buffer.get(), &barrier));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  EXPECT_FALSE(iree_hal_vulkan_command_buffer_native_replay_compatible(
      command_buffer.get()));
  // The native command must use the operation copied while recording.
  operation.operation =
      IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM;

  iree_hal_vulkan_device_syms_t syms = MakeNativeReplaySyms();
  iree_hal_buffer_t* buffer = nullptr;
  const VkBuffer native_buffer =
      reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(0x2468));
  IREE_ASSERT_OK(iree_hal_vulkan_buffer_create_borrowed(
      &syms, reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234)),
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE,
      /*allocation_size=*/128, /*handle_offset=*/8, /*byte_length=*/64,
      /*memory_property_flags=*/0, /*non_coherent_atom_size=*/1, VK_NULL_HANDLE,
      /*mapping_state=*/nullptr, native_buffer,
      /*device_address=*/0, /*host_allocation=*/nullptr,
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &buffer));
  const iree_hal_buffer_binding_t binding = {
      .buffer = buffer,
      .offset = 5,
      .length = 40,
  };
  const iree_hal_buffer_binding_table_t binding_table = {1, &binding};

  NativeReplayCapture capture;
  g_native_replay_capture = &capture;
  iree_hal_vulkan_debug_utils_t debug_utils = {};
  iree_hal_vulkan_builtins_t builtins = {};
  const VkCommandBuffer native_command_buffer =
      reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x5678));
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_record_native(
      command_buffer.get(), &syms,
      reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234)), &debug_utils,
      &builtins, native_command_buffer, /*usage_flags=*/0, VK_NULL_HANDLE,
      binding_table, /*bda_publication=*/nullptr,
      /*bda_binding_cache=*/nullptr, /*timestamps=*/nullptr,
      /*profile_marker=*/nullptr, iree_allocator_system()));

  ASSERT_EQ(capture.pipeline_barrier_count, 1);
  ASSERT_EQ(capture.memory_barrier_count, 0u);
  ASSERT_EQ(capture.buffer_barrier_count, 1u);
  const VkBufferMemoryBarrier2& native = capture.buffer_barriers[0];
  EXPECT_EQ(native.buffer, native_buffer);
  EXPECT_EQ(native.offset, 20u);
  EXPECT_EQ(native.size, 13u);
  EXPECT_NE(native.srcStageMask & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0u);
  EXPECT_EQ(native.srcStageMask & VK_PIPELINE_STAGE_2_HOST_BIT, 0u);
  EXPECT_NE(native.srcAccessMask & VK_ACCESS_2_SHADER_WRITE_BIT, 0u);
  EXPECT_NE(native.dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT, 0u);
  EXPECT_NE(native.dstAccessMask & VK_ACCESS_2_HOST_READ_BIT, 0u);
  g_native_replay_capture = nullptr;
  iree_hal_buffer_release(buffer);
}

TEST_F(VulkanCommandBufferTest,
       AtomicReplayPublishesIndirectTargetAndPreservesOrdering) {
  struct TestCase {
    iree_hal_atomic_flags_t flags;
    iree_hal_execution_stage_t source_stage_mask;
    iree_hal_execution_stage_t target_stage_mask;
    VkPipelineStageFlags2 pre_source_stage_mask;
    VkAccessFlags2 pre_source_access_mask;
    VkAccessFlags2 pre_target_access_mask;
    VkAccessFlags2 post_source_access_mask;
    VkPipelineStageFlags2 post_target_stage_mask;
    VkAccessFlags2 post_target_access_mask;
  };
  const TestCase test_cases[] = {
      {
          .flags = IREE_HAL_ATOMIC_FLAG_NONE,
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .pre_source_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .pre_source_access_mask = 0,
          .pre_target_access_mask = 0,
          .post_source_access_mask = 0,
          .post_target_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .post_target_access_mask = 0,
      },
      {
          .flags = IREE_HAL_ATOMIC_FLAG_RELEASE,
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .pre_source_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .pre_source_access_mask =
              VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .pre_target_access_mask =
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .post_source_access_mask = 0,
          .post_target_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .post_target_access_mask = 0,
      },
      {
          .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE,
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .pre_source_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .pre_source_access_mask = 0,
          .pre_target_access_mask = 0,
          .post_source_access_mask =
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .post_target_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .post_target_access_mask =
              VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      },
      {
          .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
          .pre_source_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .pre_source_access_mask =
              VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .pre_target_access_mask =
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .post_source_access_mask =
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .post_target_stage_mask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          .post_target_access_mask =
              VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
      },
      {
          .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
          .source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          .pre_source_stage_mask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
          .pre_source_access_mask = 0,
          .pre_target_access_mask =
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .post_source_access_mask =
              VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          .post_target_stage_mask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
          .post_target_access_mask = 0,
      },
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.flags);
    CommandBufferPtr command_buffer = CreateCommandBuffer(
        /*binding_capacity=*/1, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT);
    ASSERT_NE(command_buffer, nullptr);

    const iree_hal_atomic_rmw_params_t params = {
        .operand = 7,
        .flags = test_case.flags,
        .width = IREE_HAL_ATOMIC_WIDTH_32,
        .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
        .target_error_mode = IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
        .reserved = 0,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
        command_buffer.get(), test_case.source_stage_mask,
        test_case.target_stage_mask,
        iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/4,
                                          /*length=*/4),
        params));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

    iree_device_size_t publication_length = 0;
    IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_native_bda_publication_length(
        command_buffer.get(), &publication_length));
    ASSERT_EQ(publication_length, sizeof(uint64_t));

    NativeReplayCapture capture;
    g_native_replay_capture = &capture;
    iree_hal_vulkan_device_syms_t syms = MakeNativeReplaySyms();
    iree_hal_vulkan_debug_utils_t debug_utils = {};
    iree_hal_vulkan_builtins_t builtins = {};
    builtins.atomic_pipelines.syms = syms;
    builtins.atomic_pipelines.pipeline_layout =
        reinterpret_cast<VkPipelineLayout>(static_cast<uintptr_t>(0x1111));
    builtins.atomic_pipelines.pipeline_32 =
        reinterpret_cast<VkPipeline>(static_cast<uintptr_t>(0x2222));
    const iree_hal_buffer_binding_t binding = {};
    const iree_hal_buffer_binding_table_t binding_table = {
        .count = 1,
        .bindings = &binding,
    };
    iree_hal_vulkan_command_buffer_bda_binding_slot_t cached_slot = {
        .device_address = 0x1000,
        .length = 16,
    };
    iree_hal_vulkan_command_buffer_bda_binding_cache_t binding_cache = {
        .slots = &cached_slot,
        .slot_count = 1,
    };
    uint64_t published_target_address = 0;
    const iree_hal_vulkan_command_buffer_bda_publication_t publication = {
        .host_span = iree_make_byte_span(&published_target_address,
                                         sizeof(published_target_address)),
        .device_address = 0x4000,
    };
    const VkDevice logical_device =
        reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));
    const VkCommandBuffer native_command_buffer =
        reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x5678));

    IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_record_native(
        command_buffer.get(), &syms, logical_device, &debug_utils, &builtins,
        native_command_buffer, /*usage_flags=*/0, VK_NULL_HANDLE, binding_table,
        &publication, &binding_cache, /*timestamps=*/nullptr,
        /*profile_marker=*/nullptr, iree_allocator_system()));

    EXPECT_EQ(published_target_address, 0x1004u);
    ASSERT_EQ(capture.pipeline_barrier_count, 3);
    EXPECT_EQ(capture.memory_barriers[0].srcStageMask,
              VK_PIPELINE_STAGE_2_HOST_BIT);
    EXPECT_EQ(capture.memory_barriers[0].srcAccessMask,
              VK_ACCESS_2_HOST_WRITE_BIT);
    EXPECT_EQ(capture.memory_barriers[0].dstStageMask,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(capture.memory_barriers[0].dstAccessMask,
              VK_ACCESS_2_SHADER_READ_BIT);
    EXPECT_EQ(capture.memory_barriers[1].srcStageMask,
              test_case.pre_source_stage_mask);
    EXPECT_EQ(capture.memory_barriers[1].srcAccessMask,
              test_case.pre_source_access_mask);
    EXPECT_EQ(capture.memory_barriers[1].dstStageMask,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(capture.memory_barriers[1].dstAccessMask,
              test_case.pre_target_access_mask);
    EXPECT_EQ(capture.memory_barriers[2].srcStageMask,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(capture.memory_barriers[2].srcAccessMask,
              test_case.post_source_access_mask);
    EXPECT_EQ(capture.memory_barriers[2].dstStageMask,
              test_case.post_target_stage_mask);
    EXPECT_EQ(capture.memory_barriers[2].dstAccessMask,
              test_case.post_target_access_mask);
    EXPECT_EQ(capture.push_constants_count, 1);
    EXPECT_EQ(capture.bind_pipeline_count, 1);
    EXPECT_EQ(capture.dispatch_count, 1);
    EXPECT_EQ(capture.pipeline, builtins.atomic_pipelines.pipeline_32);
    ASSERT_EQ(capture.push_constants_length, 24u);
    uint64_t pushed_target_address = 0;
    uint32_t pushed_operation = 0;
    uint32_t pushed_flags = 0;
    memcpy(&pushed_target_address, capture.push_constants.data(),
           sizeof(pushed_target_address));
    memcpy(&pushed_operation, capture.push_constants.data() + 16,
           sizeof(pushed_operation));
    memcpy(&pushed_flags, capture.push_constants.data() + 20,
           sizeof(pushed_flags));
    EXPECT_EQ(pushed_target_address, publication.device_address);
    EXPECT_EQ(pushed_operation, IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_ADD);
    EXPECT_EQ(pushed_flags, 1u);

    cached_slot.device_address = 0x3000;
    IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_publish_bda_replay_data(
        command_buffer.get(), binding_table, &publication, &binding_cache));
    EXPECT_EQ(published_target_address, 0x3004u);
    g_native_replay_capture = nullptr;
  }
}

TEST_F(VulkanCommandBufferTest,
       AtomicTargetModeDoesNotReclassifyReplayInfrastructureFailures) {
  CommandBufferPtr command_buffer = CreateCommandBuffer(
      /*binding_capacity=*/1, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT);
  ASSERT_NE(command_buffer, nullptr);

  const iree_hal_atomic_rmw_params_t params = {
      .operand = 7,
      .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
      .width = IREE_HAL_ATOMIC_WIDTH_32,
      .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
      .target_error_mode = IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
      .reserved = 0,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
      command_buffer.get(), IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                        /*length=*/4),
      params));

  // Recording state is an internal command-buffer failure, not a target
  // compatibility failure.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_vulkan_command_buffer_publish_bda_replay_data(
          command_buffer.get(), iree_hal_buffer_binding_table_empty(),
          /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  // Missing or malformed publication storage also retains its existing
  // FAILED_PRECONDITION classification in opt-in mode.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_vulkan_command_buffer_publish_bda_replay_data(
          command_buffer.get(), iree_hal_buffer_binding_table_empty(),
          /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr));
  uint64_t published_target_address = 0;
  const iree_hal_vulkan_command_buffer_bda_publication_t short_publication = {
      .host_span = iree_make_byte_span(&published_target_address,
                                       sizeof(published_target_address) - 1),
      .device_address = 0x4000,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_vulkan_command_buffer_publish_bda_replay_data(
          command_buffer.get(), iree_hal_buffer_binding_table_empty(),
          &short_publication, /*bda_binding_cache=*/nullptr));

  const iree_hal_buffer_binding_t binding = {};
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = 1,
      .bindings = &binding,
  };
  iree_hal_vulkan_command_buffer_bda_binding_slot_t cached_slot = {
      .device_address = 0x1000,
      .length = 16,
  };
  iree_hal_vulkan_command_buffer_bda_binding_cache_t binding_cache = {
      .slots = &cached_slot,
      .slot_count = 1,
  };
  const iree_hal_vulkan_command_buffer_bda_publication_t
      misaligned_publication = {
          .host_span = iree_make_byte_span(&published_target_address,
                                           sizeof(published_target_address)),
          .device_address = 0x4001,
      };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_vulkan_command_buffer_publish_bda_replay_data(
                            command_buffer.get(), binding_table,
                            &misaligned_publication, &binding_cache));

  const iree_hal_vulkan_command_buffer_bda_publication_t publication = {
      .host_span = iree_make_byte_span(&published_target_address,
                                       sizeof(published_target_address)),
      .device_address = 0x4000,
  };
  iree_hal_vulkan_command_buffer_bda_binding_cache_t empty_cache = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_vulkan_command_buffer_publish_bda_replay_data(
          command_buffer.get(), iree_hal_buffer_binding_table_empty(),
          &publication, &empty_cache));

  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_publish_bda_replay_data(
      command_buffer.get(), binding_table, &publication, &binding_cache));
  EXPECT_EQ(published_target_address, 0x1000u);
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::vulkan
