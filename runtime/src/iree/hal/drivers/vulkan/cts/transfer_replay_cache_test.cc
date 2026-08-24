// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/vulkan/queue_stats.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class VulkanTransferReplayCacheTest : public CtsTestBase<> {
 protected:
  iree_hal_vulkan_native_replay_cache_stats_t NativeReplayCacheStats() {
    iree_hal_vulkan_native_replay_cache_stats_t stats;
    iree_hal_vulkan_logical_device_sample_native_replay_cache_stats(device_,
                                                                    &stats);
    return stats;
  }

  void ExecuteCommandBufferAndWait(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table) {
    IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer, binding_table));
  }
};

TEST_P(VulkanTransferReplayCacheTest,
       RepublishesIndirectFillUpdateAndCopyParameters) {
  ASSERT_GE(NativeReplayCacheStats().max_instance_count, 1);
  ASSERT_EQ(NativeReplayCacheStats().retained_instance_count, 0);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(NativeReplayCacheStats().instance_count, 0);

  constexpr iree_device_size_t kBufferLength = 16;
  constexpr iree_device_size_t kTransferLength = 7;
  constexpr uint8_t kFillPattern = 0xA5;
  const std::array<uint8_t, kTransferLength> update_data = {
      0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37};
  const std::array<uint8_t, kBufferLength> copy_source_a_data = {
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
  const std::array<uint8_t, kBufferLength> copy_source_b_data = {
      0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
      0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F};

  Ref<iree_hal_buffer_t> fill_target_a;
  Ref<iree_hal_buffer_t> update_target_a;
  Ref<iree_hal_buffer_t> copy_source_a;
  Ref<iree_hal_buffer_t> copy_target_a;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferLength, fill_target_a.out()));
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kBufferLength, update_target_a.out()));
  IREE_ASSERT_OK(CreateDeviceBufferWithData(copy_source_a_data.data(),
                                            copy_source_a_data.size(),
                                            copy_source_a.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferLength, copy_target_a.out()));

  Ref<iree_hal_buffer_t> fill_target_b;
  Ref<iree_hal_buffer_t> update_target_b;
  Ref<iree_hal_buffer_t> copy_source_b;
  Ref<iree_hal_buffer_t> copy_target_b;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferLength, fill_target_b.out()));
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kBufferLength, update_target_b.out()));
  IREE_ASSERT_OK(CreateDeviceBufferWithData(copy_source_b_data.data(),
                                            copy_source_b_data.size(),
                                            copy_source_b.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferLength, copy_target_b.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/1,
                                        kTransferLength),
      &kFillPattern, sizeof(kFillPattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, update_data.data(), /*source_offset=*/0,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/2,
                                        kTransferLength),
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/2, /*offset=*/1,
                                        kTransferLength),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/3, /*offset=*/3,
                                        kTransferLength),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_buffer_binding_t binding_entries_a[4] = {
      {fill_target_a.get(), /*offset=*/0, kBufferLength},
      {update_target_a.get(), /*offset=*/0, kBufferLength},
      {copy_source_a.get(), /*offset=*/0, kBufferLength},
      {copy_target_a.get(), /*offset=*/0, kBufferLength},
  };
  const iree_hal_buffer_binding_table_t binding_table_a = {
      IREE_ARRAYSIZE(binding_entries_a), binding_entries_a};
  iree_hal_buffer_binding_t binding_entries_b[4] = {
      {fill_target_b.get(), /*offset=*/1, kBufferLength - 1},
      {update_target_b.get(), /*offset=*/1, kBufferLength - 1},
      {copy_source_b.get(), /*offset=*/1, kBufferLength - 1},
      {copy_target_b.get(), /*offset=*/1, kBufferLength - 1},
  };
  const iree_hal_buffer_binding_table_t binding_table_b = {
      IREE_ARRAYSIZE(binding_entries_b), binding_entries_b};

  const iree_hal_vulkan_native_replay_cache_stats_t initial_stats =
      NativeReplayCacheStats();
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table_a);
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table_b);

  std::vector<uint8_t> expected_fill_a(kBufferLength, 0);
  std::memset(expected_fill_a.data() + 1, kFillPattern, kTransferLength);
  EXPECT_THAT(ReadBufferBytes(fill_target_a.get(), 0, kBufferLength),
              ContainerEq(expected_fill_a));
  std::vector<uint8_t> expected_fill_b(kBufferLength, 0);
  std::memset(expected_fill_b.data() + 2, kFillPattern, kTransferLength);
  EXPECT_THAT(ReadBufferBytes(fill_target_b.get(), 0, kBufferLength),
              ContainerEq(expected_fill_b));

  std::vector<uint8_t> expected_update_a(kBufferLength, 0);
  std::memcpy(expected_update_a.data() + 2, update_data.data(),
              update_data.size());
  EXPECT_THAT(ReadBufferBytes(update_target_a.get(), 0, kBufferLength),
              ContainerEq(expected_update_a));
  std::vector<uint8_t> expected_update_b(kBufferLength, 0);
  std::memcpy(expected_update_b.data() + 3, update_data.data(),
              update_data.size());
  EXPECT_THAT(ReadBufferBytes(update_target_b.get(), 0, kBufferLength),
              ContainerEq(expected_update_b));

  std::vector<uint8_t> expected_copy_a(kBufferLength, 0);
  std::memcpy(expected_copy_a.data() + 3, copy_source_a_data.data() + 1,
              kTransferLength);
  EXPECT_THAT(ReadBufferBytes(copy_target_a.get(), 0, kBufferLength),
              ContainerEq(expected_copy_a));
  std::vector<uint8_t> expected_copy_b(kBufferLength, 0);
  std::memcpy(expected_copy_b.data() + 4, copy_source_b_data.data() + 2,
              kTransferLength);
  EXPECT_THAT(ReadBufferBytes(copy_target_b.get(), 0, kBufferLength),
              ContainerEq(expected_copy_b));

  const iree_hal_vulkan_native_replay_cache_stats_t final_stats =
      NativeReplayCacheStats();
  EXPECT_GE(final_stats.miss_count - initial_stats.miss_count, 1);
  EXPECT_GE(final_stats.create_count - initial_stats.create_count, 1);
  EXPECT_GE(final_stats.hit_count - initial_stats.hit_count, 1);
  EXPECT_GE(final_stats.publication_update_count -
                initial_stats.publication_update_count,
            1);

  command_buffer.reset();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(VulkanTransferReplayCacheTest,
                                  {"vulkan_replay_cache"}, {});

}  // namespace iree::hal::cts
