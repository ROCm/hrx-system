// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/byte_transfer.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

struct ByteCopyPushConstants {
  uint64_t source_address;
  uint64_t target_address;
  uint32_t source_byte_offset;
  uint32_t target_byte_offset;
  uint32_t byte_length;
  uint32_t target_word_count;
  uint32_t first_target_word;
  uint32_t flags;
};
static_assert(sizeof(ByteCopyPushConstants) == 40);

struct ByteTransferCapture {
  std::vector<ByteCopyPushConstants> push_constants;
  std::vector<uint32_t> dispatch_group_counts;
};

static ByteTransferCapture* g_byte_transfer_capture = nullptr;

static VKAPI_ATTR void VKAPI_CALL FakeCmdBindPipeline(
    VkCommandBuffer command_buffer, VkPipelineBindPoint pipeline_bind_point,
    VkPipeline pipeline) {
  (void)command_buffer;
  (void)pipeline;
  EXPECT_EQ(pipeline_bind_point, VK_PIPELINE_BIND_POINT_COMPUTE);
}

static VKAPI_ATTR void VKAPI_CALL
FakeCmdPushConstants(VkCommandBuffer command_buffer, VkPipelineLayout layout,
                     VkShaderStageFlags stage_flags, uint32_t offset,
                     uint32_t size, const void* values) {
  (void)command_buffer;
  (void)layout;
  EXPECT_EQ(stage_flags, VK_SHADER_STAGE_COMPUTE_BIT);
  EXPECT_EQ(offset, 0u);
  ASSERT_EQ(size, sizeof(ByteCopyPushConstants));
  ByteCopyPushConstants constants;
  memcpy(&constants, values, sizeof(constants));
  g_byte_transfer_capture->push_constants.push_back(constants);
}

static VKAPI_ATTR void VKAPI_CALL
FakeCmdDispatch(VkCommandBuffer command_buffer, uint32_t group_count_x,
                uint32_t group_count_y, uint32_t group_count_z) {
  (void)command_buffer;
  EXPECT_GT(group_count_x, 0u);
  EXPECT_EQ(group_count_y, 1u);
  EXPECT_EQ(group_count_z, 1u);
  g_byte_transfer_capture->dispatch_group_counts.push_back(group_count_x);
}

TEST(ByteTransferTest, CopySegmentsBoundSourceAndTargetByteIndices) {
  iree_hal_vulkan_byte_transfer_pipelines_t pipelines = {};
  pipelines.syms.vkCmdBindPipeline = FakeCmdBindPipeline;
  pipelines.syms.vkCmdPushConstants = FakeCmdPushConstants;
  pipelines.syms.vkCmdDispatch = FakeCmdDispatch;
  pipelines.max_workgroup_count_x = UINT32_MAX;
  pipelines.pipeline_layout =
      reinterpret_cast<VkPipelineLayout>(static_cast<uintptr_t>(0x1111));
  pipelines.copy_pipeline =
      reinterpret_cast<VkPipeline>(static_cast<uintptr_t>(0x2222));
  const VkCommandBuffer command_buffer =
      reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x3333));
  constexpr VkDeviceSize kLength = (VkDeviceSize)UINT32_MAX + 17;

  for (uint32_t source_byte_offset = 0; source_byte_offset < 4;
       ++source_byte_offset) {
    for (uint32_t target_byte_offset = 0; target_byte_offset < 4;
         ++target_byte_offset) {
      SCOPED_TRACE(source_byte_offset);
      SCOPED_TRACE(target_byte_offset);
      const VkDeviceAddress source_address = 0x1000 + source_byte_offset;
      const VkDeviceAddress target_address =
          0x1000000000ull + target_byte_offset;
      ByteTransferCapture capture;
      g_byte_transfer_capture = &capture;
      IREE_ASSERT_OK(iree_hal_vulkan_byte_transfer_record_copy(
          &pipelines, command_buffer, source_address, target_address, kLength));

      ASSERT_EQ(capture.push_constants.size(),
                capture.dispatch_group_counts.size());
      ASSERT_GE(capture.push_constants.size(), 2u);
      VkDeviceSize captured_length = 0;
      uint32_t expected_source_byte_offset = source_byte_offset;
      uint32_t expected_target_byte_offset = target_byte_offset;
      for (iree_host_size_t i = 0; i < capture.push_constants.size(); ++i) {
        const ByteCopyPushConstants& constants = capture.push_constants[i];
        EXPECT_EQ(constants.source_byte_offset, expected_source_byte_offset);
        EXPECT_EQ(constants.target_byte_offset, expected_target_byte_offset);
        EXPECT_LE(
            (uint64_t)constants.source_byte_offset + constants.byte_length,
            (uint64_t)UINT32_MAX);
        EXPECT_LE(
            (uint64_t)constants.target_byte_offset + constants.byte_length,
            (uint64_t)UINT32_MAX);
        EXPECT_EQ(constants.target_word_count,
                  ((uint64_t)constants.target_byte_offset +
                   constants.byte_length + 3) /
                      4);
        captured_length += constants.byte_length;
        if (i + 1 < capture.push_constants.size()) {
          EXPECT_EQ((constants.target_byte_offset + constants.byte_length) % 4,
                    0u);
        }
        expected_source_byte_offset =
            (expected_source_byte_offset + constants.byte_length) % 4;
        expected_target_byte_offset =
            (expected_target_byte_offset + constants.byte_length) % 4;
      }
      EXPECT_EQ(captured_length, kLength);
    }
  }
  g_byte_transfer_capture = nullptr;
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::vulkan
