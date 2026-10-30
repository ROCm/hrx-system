// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HAL command buffer dispatch with multiple workgroups.
// Exercises multi-workgroup scheduling paths that single-workgroup dispatches
// skip entirely.

#include <cstdint>
#include <numeric>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class DispatchMultiWorkgroupTest : public CtsTestBase<> {
 protected:
  static constexpr iree_host_size_t kWorkgroupCount = 32;

  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }

    LoadExecutableOrSkipUnsupported(
        "command_buffer_dispatch_multi_workgroup_test.bin", &executable_);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_hal_executable_t* executable_ = nullptr;
};

// Dispatches a kernel with 32 workgroups where each workgroup writes its
// workgroup_id to the corresponding output element.
// Expected output: [0, 1, 2, ..., 31].
TEST_P(DispatchMultiWorkgroupTest, WriteWorkgroupIds) {
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kWorkgroupCount * sizeof(uint32_t),
                                          &output_buffer));

  iree_hal_buffer_ref_t binding_refs[1];
  iree_hal_buffer_binding_t binding_table_values[1];
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();
  switch (recording_mode()) {
    case RecordingMode::kDirect:
      binding_refs[0] = {
          /*binding=*/0,
          /*buffer_slot=*/0,
          /*buffer=*/output_buffer,
          /*offset=*/0,
          /*length=*/iree_hal_buffer_byte_length(output_buffer),
      };
      break;
    case RecordingMode::kIndirect:
      binding_table.count = IREE_ARRAYSIZE(binding_refs);
      binding_table.bindings = binding_table_values;
      binding_table_values[0] = {
          /*buffer=*/output_buffer,
          /*offset=*/0,
          /*length=*/iree_hal_buffer_byte_length(output_buffer),
      };
      binding_refs[0] = {
          /*binding=*/0,
          /*buffer_slot=*/0,
          /*buffer=*/nullptr,
          /*offset=*/0,
          /*length=*/iree_hal_buffer_byte_length(output_buffer),
      };
      break;
  }
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                     IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                     binding_table.count, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(kWorkgroupCount, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  const iree_hal_barrier_t execution_barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH |
                           IREE_HAL_EXECUTION_STAGE_TRANSFER |
                           IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
                           IREE_HAL_EXECUTION_STAGE_DISPATCH |
                           IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = IREE_HAL_BARRIER_FLAG_NONE,
      .effects = {},
      .memory_barrier_count = 0,
      .memory_barriers = nullptr,
      .buffer_barrier_count = 0,
      .buffer_barriers = nullptr,
  };
  IREE_ASSERT_OK(
      iree_hal_command_buffer_barrier(command_buffer, &execution_barrier));

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer, binding_table));

  std::vector<uint32_t> output_data(kWorkgroupCount);
  IREE_ASSERT_OK(
      DownloadBufferData(output_buffer, /*source_offset=*/0,
                         /*target=*/output_data.data(),
                         /*length=*/output_data.size() * sizeof(uint32_t)));

  std::vector<uint32_t> expected_data(kWorkgroupCount);
  std::iota(expected_data.begin(), expected_data.end(), 0u);
  EXPECT_THAT(output_data, ContainerEq(expected_data));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
}

TEST_P(DispatchMultiWorkgroupTest, TimestampedDispatch) {
  auto* queue = QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH |
                                          IREE_HAL_COMMAND_CATEGORY_TRANSFER);
  ASSERT_NE(nullptr, queue);
  const auto* family = iree_hal_queue_family(queue);
  const bool indirect = recording_mode() == RecordingMode::kIndirect;
  const auto capability =
      indirect ? IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS
               : IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS;
  if (!(family->spec->flags & capability)) {
    GTEST_SKIP() << "Queue family does not support command timestamps";
  }
  Ref<iree_hal_buffer_t> output;
  Ref<iree_hal_buffer_t> timestamps;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kWorkgroupCount * sizeof(uint32_t),
                                          output.out()));
  IREE_ASSERT_OK(
      CreateFilledDeviceBuffer<uint32_t>(32, 0xABABABAB, timestamps.out()));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      family, 0,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      indirect ? 2 : 0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const auto first = indirect ? iree_hal_make_indirect_buffer_ref(1, 8, 8)
                              : iree_hal_make_buffer_ref(timestamps, 8, 8);
  // The end timestamp replaces workgroups 4 and 5's output after an ordinary
  // write-to-write visibility transition, without an intervening submission.
  const auto last = indirect ? iree_hal_make_indirect_buffer_ref(0, 16, 8)
                             : iree_hal_make_buffer_ref(output, 16, 8);
  const auto target = indirect
                          ? iree_hal_make_indirect_buffer_ref(
                                0, 0, kWorkgroupCount * sizeof(uint32_t))
                          : iree_hal_make_buffer_ref(
                                output, 0, kWorkgroupCount * sizeof(uint32_t));
  IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
      command_buffer, 0, IREE_HAL_EXECUTION_STAGE_DISPATCH, first, 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(kWorkgroupCount, 1, 1),
      iree_const_byte_span_empty(), {1, &target}, IREE_HAL_DISPATCH_FLAG_NONE));
  const iree_hal_memory_barrier_t overwrite_memory = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  const iree_hal_barrier_t overwrite = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .memory_barrier_count = 1,
      .memory_barriers = &overwrite_memory,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer, &overwrite));
  IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
      command_buffer, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, last, 0));
  const iree_hal_barrier_t publish = {
      .source_stage_mask =
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer, &publish));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  const iree_hal_buffer_binding_t bindings[] = {
      {output, 0, kWorkgroupCount * sizeof(uint32_t)}, {timestamps, 0, 32}};
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      command_buffer, {indirect ? 2u : 0u, indirect ? bindings : nullptr}));
  std::vector<uint32_t> expected(kWorkgroupCount);
  std::iota(expected.begin(), expected.end(), 0u);
  auto actual = ReadBufferData<uint32_t>(output);
  ASSERT_EQ(actual.size(), kWorkgroupCount);
  const uint64_t end_tick = uint64_t{actual[4]} | (uint64_t{actual[5]} << 32);
  EXPECT_NE(end_tick, UINT64_C(0x0000000500000004));
  actual[4] = 4;
  actual[5] = 5;
  EXPECT_EQ(actual, expected);
  const auto ticks = ReadBufferData<uint64_t>(timestamps);
  ASSERT_EQ(ticks.size(), 4u);
  EXPECT_EQ(ticks[0], UINT64_C(0xABABABABABABABAB));
  EXPECT_EQ(ticks[3], ticks[0]);
  EXPECT_NE(ticks[1], ticks[0]);
  EXPECT_EQ(ticks[2], ticks[0]);
}

CTS_REGISTER_EXECUTABLE_COMMAND_BUFFER_TEST_SUITE(DispatchMultiWorkgroupTest);

}  // namespace iree::hal::cts
