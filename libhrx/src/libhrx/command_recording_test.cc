// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "command_recording.h"

#include <array>

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CommandRecordingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    hrx::runtime::runtime_ptr runtime;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_runtime_create(
        nullptr, hrx_host_allocator_system(), runtime.for_output())));
    const hrx_device_create_params_t params = {.uri = "task://"};
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_device_group_create(runtime, 1, &params, group_.for_output())));
    hrx_device_t device = hrx_device_group_get_device(group_, 0);
    queue_ = hrx_device_get_queue(device, 0, 0);
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_stream_create(queue_, 0, stream_.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_buffer_allocate(stream_, 128, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                            HRX_BUFFER_USAGE_DEFAULT, source_.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_buffer_allocate(stream_, 128, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                            HRX_BUFFER_USAGE_DEFAULT, target_.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream_)));
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue_->hal_queue),
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_ANY, 0,
        &command_buffer_));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_create(device, 0, completion_.for_output())));
  }

  void TearDown() override { iree_hal_command_buffer_release(command_buffer_); }

  void Execute() {
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer_));
    uint64_t value = 1;
    iree_hal_semaphore_t* semaphore = completion_.get()->hal_semaphore;
    const iree_hal_semaphore_list_t signals = {1, &semaphore, &value};
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queue_->hal_queue, iree_hal_semaphore_list_empty(), signals,
        command_buffer_, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_semaphore_wait(completion_, value, UINT64_MAX)));
  }

  // Owns the device and all convenience allocation sources until teardown.
  hrx::runtime::device_group_ptr group_;
  // Borrowed exact queue used for command recording and readback.
  hrx_queue_t queue_ = nullptr;
  // Allocation stream, synchronized before raw command submission.
  hrx::runtime::stream_ptr stream_;
  // Retained source allocation.
  hrx::runtime::buffer_ptr source_;
  // Retained destination allocation.
  hrx::runtime::buffer_ptr target_;
  // Explicit completion of the submitted command buffer.
  hrx::runtime::semaphore_ptr completion_;
  // Open command buffer retained through execution and released in TearDown.
  iree_hal_command_buffer_t* command_buffer_ = nullptr;
};

TEST_F(CommandRecordingTest, StridedCopiesPreserveGapsAndSelectOneWorkstream) {
  hrx_command_fill_t clear = {
      .command = {HRX_COMMAND_FILL},
      .pattern_size = 1,
      .target = iree_hal_make_buffer_ref(target_.get()->hal_buffer, 0, 128),
  };
  hrx_command_fill_t fill = {
      .command = {HRX_COMMAND_FILL},
      .pattern_size = 1,
      .pattern = 0x3C,
      .target = iree_hal_make_buffer_ref(source_.get()->hal_buffer, 0, 128),
  };
  hrx_command_copy_strided_t copy = {
      .copy =
          {
              .command = {HRX_COMMAND_COPY},
              .strided = true,
              .release_to_host = true,
              .source_buffer = source_.get()->hal_buffer,
              .target_buffer = target_.get()->hal_buffer,
              .target_offset = 4,
              .length = 4,
          },
      .row_count = 3,
      .slice_count = 2,
      .source_row_stride = 8,
      .target_row_stride = 12,
      .source_slice_stride = 32,
      .target_slice_stride = 48,
  };
  hrx_command_fill_t other_stream = clear;
  other_stream.pattern = 0xFF;
  void* sources[] = {&clear, &fill, &copy, &other_stream};
  uint32_t order[] = {0, 1, 2, 3};
  uint8_t flags[] = {0, HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE,
                     HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE,
                     1 << HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT};
  const hrx_graph_schedule_t schedule = {
      .node_count = 4,
      .sources = sources,
      .order = order,
      .flags = flags,
  };
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
  };
  iree_hal_barrier_flags_t release_flags = IREE_HAL_BARRIER_FLAG_NONE;
  IREE_ASSERT_OK(hrx_graph_record_commands(&schedule, 0, 4, 0, &barrier,
                                           command_buffer_, &release_flags));
  EXPECT_EQ(release_flags, IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
  Execute();
  std::array<uint8_t, 128> contents;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h(
      queue_, target_, 0, contents.data(), contents.size())));
  std::array<uint8_t, 128> expected = {};
  for (size_t slice = 0; slice < 2; ++slice) {
    for (size_t row = 0; row < 3; ++row) {
      for (size_t byte = 0; byte < 4; ++byte) {
        expected[4 + slice * 48 + row * 12 + byte] = 0x3C;
      }
    }
  }
  EXPECT_EQ(contents, expected);
}

}  // namespace
