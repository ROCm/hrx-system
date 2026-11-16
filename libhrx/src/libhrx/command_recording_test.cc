// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "command_recording.h"

#include <array>
#include <vector>

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum class RecordedKind { kBegin, kEnd, kBarrier, kFill };

struct RecordedCommand {
  // Operation observed at the HAL command-buffer boundary.
  RecordedKind kind;
  // Execution stages preceding a barrier.
  iree_hal_execution_stage_t source_stage = 0;
  // Execution stages following a barrier.
  iree_hal_execution_stage_t target_stage = 0;
  // Explicit visibility flags passed to the backend.
  iree_hal_barrier_flags_t flags = 0;
  // Source access scope of the optional memory dependency.
  iree_hal_access_scope_t source_scope = 0;
  // Target access scope of the optional memory dependency.
  iree_hal_access_scope_t target_scope = 0;

  bool operator==(const RecordedCommand&) const = default;
};

// Captures the real recorder's calls to its HAL dependency. Public command
// validation remains enabled; payload execution is covered separately below.
struct RecordingCommandBuffer {
  explicit RecordingCommandBuffer(const iree_hal_queue_family_t* family)
      : validation((iree_hal_command_buffer_validation_state_size(0, 0) + 7) /
                   8) {
    static const iree_hal_command_buffer_vtable_t vtable = {
        .destroy = +[](iree_hal_command_buffer_t*) {},
        .begin =
            +[](iree_hal_command_buffer_t* command_buffer) {
              Cast(command_buffer)->commands.push_back({RecordedKind::kBegin});
              return iree_ok_status();
            },
        .end =
            +[](iree_hal_command_buffer_t* command_buffer) {
              Cast(command_buffer)->commands.push_back({RecordedKind::kEnd});
              return iree_ok_status();
            },
        .begin_debug_group =
            +[](iree_hal_command_buffer_t*, iree_string_view_t,
                iree_hal_label_color_t,
                const iree_hal_label_location_t*) { return iree_ok_status(); },
        .end_debug_group =
            +[](iree_hal_command_buffer_t*) { return iree_ok_status(); },
        .barrier =
            +[](iree_hal_command_buffer_t* command_buffer,
                const iree_hal_barrier_t* barrier) {
              EXPECT_LE(barrier->memory_barrier_count, 1u);
              const iree_hal_memory_barrier_t memory =
                  barrier->memory_barrier_count ? barrier->memory_barriers[0]
                                                : iree_hal_memory_barrier_t{};
              Cast(command_buffer)
                  ->commands.push_back(
                      {RecordedKind::kBarrier, barrier->source_stage_mask,
                       barrier->target_stage_mask, barrier->flags,
                       memory.source_scope, memory.target_scope});
              return iree_ok_status();
            },
        .fill_buffer =
            +[](iree_hal_command_buffer_t* command_buffer,
                iree_hal_buffer_ref_t, const void*, iree_host_size_t,
                iree_hal_fill_flags_t) {
              Cast(command_buffer)->commands.push_back({RecordedKind::kFill});
              return iree_ok_status();
            },
    };
    iree_hal_command_buffer_initialize(family, 0, IREE_HAL_COMMAND_CATEGORY_ANY,
                                       0, validation.data(), &vtable, &base);
  }

  ~RecordingCommandBuffer() { iree_hal_command_buffer_release(&base); }

  static RecordingCommandBuffer* Cast(iree_hal_command_buffer_t* base) {
    return reinterpret_cast<RecordingCommandBuffer*>(base);
  }

  // HAL resource prefix used by the normal public recording entry points.
  iree_hal_command_buffer_t base = {};
  // Aligned storage for the actual command-buffer validator.
  std::vector<uint64_t> validation;
  // Exact operation and visibility sequence received by the backend.
  std::vector<RecordedCommand> commands;
};

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
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_create(device, 0, completion_.for_output())));
  }

  void TearDown() override { iree_hal_command_buffer_release(command_buffer_); }

  void Execute() {
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
  // Command buffer retained through execution and released in TearDown.
  iree_hal_command_buffer_t* command_buffer_ = nullptr;
};

TEST_F(CommandRecordingTest, PartitionPublishesAcrossExternalBoundaries) {
  hrx_command_fill_t fill = {
      .command = {HRX_COMMAND_FILL},
      .pattern_size = 1,
      .pattern = 0x3C,
      .target = iree_hal_make_buffer_ref(target_.get()->hal_buffer, 0, 128),
  };
  void* sources[] = {&fill, &fill};
  uint32_t order[] = {0, 1};
  uint8_t flags[] = {0, HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE};
  const hrx_graph_schedule_t schedule = {
      .node_count = 2,
      .sources = sources,
      .order = order,
      .flags = flags,
  };
  RecordingCommandBuffer recorded(iree_hal_queue_family(queue_->hal_queue));
  IREE_ASSERT_OK(
      hrx_graph_record_partition(&schedule, 0, 2, 0, &recorded.base));
  constexpr iree_hal_access_scope_t device_access =
      IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE | IREE_HAL_ACCESS_SCOPE_ATOMIC_READ |
      IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE;
  const std::vector<RecordedCommand> expected = {
      {RecordedKind::kBegin},
      {RecordedKind::kBarrier, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
       IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
       IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE},
      {RecordedKind::kFill},
      {RecordedKind::kBarrier, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
       IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, 0, device_access, device_access},
      {RecordedKind::kFill},
      {RecordedKind::kBarrier, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, 0,
       IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE},
      {RecordedKind::kEnd},
  };
  EXPECT_EQ(recorded.commands, expected);
}

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
  IREE_ASSERT_OK(
      hrx_graph_record_partition(&schedule, 0, 4, 0, command_buffer_));
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
