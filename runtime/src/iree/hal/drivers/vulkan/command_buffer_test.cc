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

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

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
  std::array<VkMemoryBarrier2, 4> memory_barriers = {};

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
  ASSERT_LE(g_native_replay_capture->pipeline_barrier_count,
            g_native_replay_capture->memory_barriers.size());
  ASSERT_EQ(dependency_info->memoryBarrierCount, 1u);
  ASSERT_NE(dependency_info->pMemoryBarriers, nullptr);
  g_native_replay_capture
      ->memory_barriers[g_native_replay_capture->pipeline_barrier_count - 1] =
      dependency_info->pMemoryBarriers[0];
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
}

static VKAPI_ATTR void VKAPI_CALL FakeCmdInsertDebugUtilsLabelEXT(
    VkCommandBuffer command_buffer, const VkDebugUtilsLabelEXT* label_info) {
  (void)command_buffer;
  (void)label_info;
}

static iree_hal_vulkan_device_syms_t MakeNativeReplaySyms() {
  iree_hal_vulkan_device_syms_t syms = {};
  syms.vkBeginCommandBuffer = FakeBeginCommandBuffer;
  syms.vkEndCommandBuffer = FakeEndCommandBuffer;
  syms.vkCmdBeginDebugUtilsLabelEXT = FakeCmdBeginDebugUtilsLabelEXT;
  syms.vkCmdEndDebugUtilsLabelEXT = FakeCmdEndDebugUtilsLabelEXT;
  syms.vkCmdInsertDebugUtilsLabelEXT = FakeCmdInsertDebugUtilsLabelEXT;
  syms.vkCmdPipelineBarrier2 = FakeCmdPipelineBarrier2;
  syms.vkCmdPushConstants = FakeCmdPushConstants;
  syms.vkCmdBindPipeline = FakeCmdBindPipeline;
  syms.vkCmdDispatch = FakeCmdDispatch;
  return syms;
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
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        iree_make_cstring_view("vulkan_command_buffer_test"),
        iree_allocator_system(), iree_allocator_system(), &device_allocator_));
    iree_arena_block_pool_initialize(block_size_, iree_allocator_system(),
                                     &block_pool_);
    atomic_pipelines_.pipeline_layout =
        reinterpret_cast<VkPipelineLayout>(static_cast<uintptr_t>(0x1111));
    atomic_pipelines_.pipeline_32 =
        reinterpret_cast<VkPipeline>(static_cast<uintptr_t>(0x2222));
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    iree_hal_allocator_release(device_allocator_);
  }

  CommandBufferPtr CreateCommandBuffer(
      iree_host_size_t binding_capacity = 0,
      iree_hal_command_buffer_mode_t mode =
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT) {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_vulkan_command_buffer_create(
        device_allocator_, mode, IREE_HAL_COMMAND_CATEGORY_ANY,
        IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity, &atomic_pipelines_,
        &block_pool_, iree_allocator_system(), &command_buffer));
    return CommandBufferPtr(command_buffer);
  }

 private:
  // Test allocator borrowed by command buffers for validation.
  iree_hal_allocator_t* device_allocator_ = nullptr;

  // Command-buffer block size used by this fixture.
  iree_host_size_t block_size_ = 256;

  // Command-payload and resource-set block pool borrowed by command buffers.
  iree_arena_block_pool_t block_pool_;

  // Empty device-owned pipeline table borrowed by command buffers.
  iree_hal_vulkan_atomic_pipelines_t atomic_pipelines_ = {};
};

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
      native_command_buffer, /*usage_flags=*/0, binding_table,
      /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr,
      /*profile_marker=*/nullptr, iree_allocator_system()));

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

TEST_F(VulkanCommandBufferTest, IndirectTransferRequiresPerIssueRecording) {
  CommandBufferPtr command_buffer = CreateCommandBuffer(
      /*binding_capacity=*/1, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT);
  ASSERT_NE(command_buffer, nullptr);

  constexpr uint8_t kPattern = 0xA5;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer.get(),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                        /*length=*/4),
      &kPattern, sizeof(kPattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  EXPECT_TRUE(iree_hal_vulkan_command_buffer_requires_per_issue_recording(
      command_buffer.get()));
}

TEST_F(VulkanCommandBufferTest, SystemScopeUsesHostMemoryDomain) {
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);

  const iree_hal_memory_barrier_t memory_barrier = {
      /*.source_scope=*/IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      /*.target_scope=*/IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer.get(), IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
          IREE_HAL_EXECUTION_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      /*memory_barrier_count=*/1, &memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
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
      native_command_buffer, /*usage_flags=*/0, binding_table,
      /*bda_publication=*/nullptr, /*bda_binding_cache=*/nullptr,
      /*profile_marker=*/nullptr, iree_allocator_system()));

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
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_NONE,
          /*.source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER,
          /*.target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER,
          /*.pre_source_stage_mask=*/VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          /*.pre_source_access_mask=*/0,
          /*.pre_target_access_mask=*/0,
          /*.post_source_access_mask=*/0,
          /*.post_target_stage_mask=*/VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          /*.post_target_access_mask=*/0,
      },
      {
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_RELEASE,
          /*.source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER,
          /*.target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER,
          /*.pre_source_stage_mask=*/VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          /*.pre_source_access_mask=*/VK_ACCESS_2_TRANSFER_WRITE_BIT,
          /*.pre_target_access_mask=*/
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          /*.post_source_access_mask=*/0,
          /*.post_target_stage_mask=*/VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          /*.post_target_access_mask=*/0,
      },
      {
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE,
          /*.source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER,
          /*.target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER,
          /*.pre_source_stage_mask=*/VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          /*.pre_source_access_mask=*/0,
          /*.pre_target_access_mask=*/0,
          /*.post_source_access_mask=*/
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          /*.post_target_stage_mask=*/VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          /*.post_target_access_mask=*/
          VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
      },
      {
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE |
              IREE_HAL_ATOMIC_FLAG_RELEASE,
          /*.source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER,
          /*.target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER,
          /*.pre_source_stage_mask=*/VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          /*.pre_source_access_mask=*/VK_ACCESS_2_TRANSFER_WRITE_BIT,
          /*.pre_target_access_mask=*/
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          /*.post_source_access_mask=*/
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          /*.post_target_stage_mask=*/VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          /*.post_target_access_mask=*/
          VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
      },
      {
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE |
              IREE_HAL_ATOMIC_FLAG_RELEASE,
          /*.source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          /*.target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          /*.pre_source_stage_mask=*/VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
          /*.pre_source_access_mask=*/0,
          /*.pre_target_access_mask=*/
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          /*.post_source_access_mask=*/
          VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
          /*.post_target_stage_mask=*/VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
          /*.post_target_access_mask=*/0,
      },
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.flags);
    CommandBufferPtr command_buffer = CreateCommandBuffer(
        /*binding_capacity=*/1, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT);
    ASSERT_NE(command_buffer, nullptr);

    const iree_hal_atomic_rmw_params_t params = {
        /*.operand=*/7,
        /*.flags=*/test_case.flags,
        /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
        /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
        /*.reserved=*/0,
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
        /*.count=*/1,
        /*.bindings=*/&binding,
    };
    iree_hal_vulkan_command_buffer_bda_binding_slot_t cached_slot = {
        /*.device_address=*/0x1000,
        /*.length=*/16,
    };
    iree_hal_vulkan_command_buffer_bda_binding_cache_t binding_cache = {
        /*.slots=*/&cached_slot,
        /*.slot_count=*/1,
    };
    uint64_t published_target_address = 0;
    const iree_hal_vulkan_command_buffer_bda_publication_t publication = {
        /*.host_span=*/iree_make_byte_span(&published_target_address,
                                           sizeof(published_target_address)),
        /*.device_address=*/0x4000,
    };
    const VkDevice logical_device =
        reinterpret_cast<VkDevice>(static_cast<uintptr_t>(0x1234));
    const VkCommandBuffer native_command_buffer =
        reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x5678));

    IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_record_native(
        command_buffer.get(), &syms, logical_device, &debug_utils, &builtins,
        native_command_buffer, /*usage_flags=*/0, binding_table, &publication,
        &binding_cache, /*profile_marker=*/nullptr, iree_allocator_system()));

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

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::vulkan
