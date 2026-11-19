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

enum class RecordedKind {
  kBegin,
  kEnd,
  kBarrier,
  kDispatch,
  kCopy,
  kFill,
  kAtomic
};

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

struct RecordedCopy {
  // Bounded source window passed to the HAL dependency.
  iree_hal_buffer_ref_t source;
  // Bounded target window passed to the HAL dependency.
  iree_hal_buffer_ref_t target;
  // Geometry captured during the recording call.
  iree_hal_transfer_region_t region;
  // Copy policy passed through from the graph producer.
  iree_hal_copy_flags_t flags;
};

struct RecordedFill {
  // Addressed destination window passed to HAL.
  iree_hal_buffer_ref_t target;
  // Useful byte extent and strides captured during recording.
  iree_hal_fill_region_t region;
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
        .atomic_store =
            +[](iree_hal_command_buffer_t* command_buffer,
                iree_hal_execution_stage_t, iree_hal_execution_stage_t,
                iree_hal_buffer_ref_t, iree_hal_atomic_store_params_t) {
              Cast(command_buffer)->commands.push_back({RecordedKind::kAtomic});
              return iree_ok_status();
            },
        .fill_buffer =
            +[](iree_hal_command_buffer_t* command_buffer,
                iree_hal_buffer_ref_t target, const void*, iree_host_size_t,
                const iree_hal_fill_region_t* region, iree_hal_fill_flags_t) {
              Cast(command_buffer)->commands.push_back({RecordedKind::kFill});
              Cast(command_buffer)->fills.push_back({target, *region});
              return iree_ok_status();
            },
        .copy_buffer =
            +[](iree_hal_command_buffer_t* command_buffer,
                iree_hal_buffer_ref_t source, iree_hal_buffer_ref_t target,
                const iree_hal_transfer_region_t* region,
                iree_hal_copy_flags_t flags) {
              Cast(command_buffer)->commands.push_back({RecordedKind::kCopy});
              Cast(command_buffer)
                  ->copies.push_back({source, target, *region, flags});
              return iree_ok_status();
            },
        .dispatch =
            +[](iree_hal_command_buffer_t* command_buffer,
                iree_hal_executable_t*, iree_hal_executable_function_t,
                iree_hal_dispatch_config_t, iree_const_byte_span_t,
                iree_hal_buffer_ref_list_t, iree_hal_dispatch_flags_t) {
              Cast(command_buffer)
                  ->commands.push_back({RecordedKind::kDispatch});
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
  // Copy windows and geometry received by the backend.
  std::vector<RecordedCopy> copies;
  // Fill windows and geometry received by the backend.
  std::vector<RecordedFill> fills;
};

class CommandRecordingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &analysis_pool_);
    iree_arena_initialize(&analysis_pool_, &analysis_arena_);
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

  void TearDown() override {
    iree_hal_command_buffer_release(command_buffer_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&analysis_pool_);
  }

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
  // Compilation scratch shared by the analyzer and recorder tests.
  iree_arena_block_pool_t analysis_pool_;
  // Reset between schedules and released even after a fatal assertion.
  iree_arena_allocator_t analysis_arena_;
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
  const hrx_graph_partition_t partition = {
      .count = 2,
      .stream_count = 1,
      .command_mask = 1u << HRX_COMMAND_FILL,
  };
  IREE_ASSERT_OK(
      hrx_graph_record_partition(&schedule, &partition, 0, &recorded.base));
  constexpr iree_hal_access_scope_t device_access =
      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;
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

TEST_F(CommandRecordingTest,
       InternalBarriersCoverEveryCommandClassInPartition) {
  const auto* family = iree_hal_queue_family(queue_->hal_queue);
  // Only the executable's family is needed at this recording-only boundary.
  // The HAL dependency captures dispatches instead of executing them.
  static const iree_hal_executable_vtable_t executable_vtable = {
      .destroy = +[](iree_hal_executable_t*) {},
  };
  iree_hal_executable_t executable;
  iree_hal_executable_initialize(family, &executable_vtable, &executable);
  hrx_command_dispatch_t dispatch = {
      .command = {HRX_COMMAND_DISPATCH},
      .executable = &executable,
      .workgroup_count = {1, 1, 1},
      .workgroup_size = {1, 1, 1},
  };
  hrx_command_copy_t copy = {
      .command = {HRX_COMMAND_COPY},
      .source_buffer = source_.get()->hal_buffer,
      .target_buffer = target_.get()->hal_buffer,
      .length = 4,
  };
  hrx_command_fill_t fill = {
      .command = {HRX_COMMAND_FILL},
      .pattern_size = 4,
      .target = iree_hal_make_buffer_ref(target_.get()->hal_buffer, 0, 4),
  };
  hrx_atomic_operation_t operation = {
      .kind = HRX_ATOMIC_OPERATION_STORE,
      .target_buffer = target_.get()->hal_buffer,
  };
  operation.params.store = {.value = 1, .width = IREE_HAL_ATOMIC_WIDTH_32};
  hrx_command_atomic_batch_t atomic = {
      .command = {HRX_COMMAND_ATOMIC_BATCH},
      .operation_count = 1,
      .operations = &operation,
  };
  void* commands[] = {&dispatch, &copy, &fill, &atomic};
  constexpr iree_hal_access_scope_t scopes[] = {
      IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
          IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
          IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
          IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
          IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      IREE_HAL_ACCESS_SCOPE_ATOMIC_READ | IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE,
  };
  // Exercise every pair, including each homogeneous class, through the actual
  // analyzer and recorder. The join must cover both predecessor classes even
  // when the immediately preceding recorded command has only the second class.
  for (uint32_t first = 0; first < HRX_COMMAND_COUNT; ++first) {
    for (uint32_t second = 0; second < HRX_COMMAND_COUNT; ++second) {
      SCOPED_TRACE(testing::Message() << first << " -> " << second);
      hrx_graph_analysis_t analysis;
      IREE_ASSERT_OK(
          hrx_graph_analysis_initialize(3, &analysis_arena_, &analysis));
      analysis.sources[0] = commands[first];
      analysis.sources[1] = commands[second];
      analysis.sources[2] = commands[second];
      analysis.flags[0] = first << HRX_GRAPH_ANALYSIS_NODE_COMMAND_SHIFT;
      analysis.flags[1] = second << HRX_GRAPH_ANALYSIS_NODE_COMMAND_SHIFT;
      analysis.flags[2] = second << HRX_GRAPH_ANALYSIS_NODE_COMMAND_SHIFT;
      analysis.offsets[3] = 2;
      IREE_ASSERT_OK(hrx_graph_analysis_allocate_dependencies(
          &analysis, &analysis_arena_));
      analysis.dependencies[0] = 0;
      analysis.dependencies[1] = 1;
      analysis.is_topologically_ordered = true;
      hrx_graph_schedule_t schedule;
      IREE_ASSERT_OK(hrx_graph_analyze(&analysis, &analysis_arena_, &schedule));
      ASSERT_EQ(schedule.partition_count, 1u);
      RecordingCommandBuffer recorded(family);
      IREE_ASSERT_OK(hrx_graph_record_partition(
          &schedule, &schedule.partitions[0], 0, &recorded.base));
      std::vector<RecordedCommand> barriers;
      for (const auto& command : recorded.commands) {
        if (command.kind == RecordedKind::kBarrier) {
          barriers.push_back(command);
        }
      }
      const auto scope = scopes[first] | scopes[second];
      const std::vector<RecordedCommand> expected = {
          {RecordedKind::kBarrier, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
           IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
           IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE},
          {RecordedKind::kBarrier, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
           IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, 0, scope, scope},
          {RecordedKind::kBarrier, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, 0,
           IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE},
      };
      EXPECT_EQ(barriers, expected);
      iree_arena_reset(&analysis_arena_);
    }
  }
  iree_hal_executable_release(&executable);
}

TEST_F(CommandRecordingTest, StridedFillRecordsOneOperation) {
  hrx_command_fill_strided_t fill = {
      .fill =
          {
              .command = {HRX_COMMAND_FILL},
              .pattern_size = 4,
              .strided = true,
              .pattern = 0x52A13C78u,
              .target =
                  iree_hal_make_buffer_ref(target_.get()->hal_buffer, 8, 8),
          },
      .row_count = 3,
      .slice_count = 2,
      .target_row_stride = 16,
      .target_slice_stride = 64,
  };
  void* sources[] = {&fill};
  uint32_t order[] = {0};
  uint8_t flags[] = {0};
  const hrx_graph_schedule_t schedule = {
      .node_count = 1,
      .sources = sources,
      .order = order,
      .flags = flags,
  };
  const hrx_graph_partition_t partition = {
      .count = 1,
      .stream_count = 1,
      .command_mask = 1u << HRX_COMMAND_FILL,
  };
  RecordingCommandBuffer recorded(iree_hal_queue_family(queue_->hal_queue));
  IREE_ASSERT_OK(
      hrx_graph_record_partition(&schedule, &partition, 0, &recorded.base));
  ASSERT_EQ(recorded.fills.size(), 1u);
  const RecordedFill& observed = recorded.fills.front();
  EXPECT_EQ(observed.target.buffer, target_.get()->hal_buffer);
  EXPECT_EQ(observed.target.offset, 8u);
  EXPECT_EQ(observed.target.length, 104u);
  EXPECT_EQ(observed.region.row_length, 8u);
  EXPECT_EQ(observed.region.row_count, 3u);
  EXPECT_EQ(observed.region.slice_count, 2u);
  EXPECT_EQ(observed.region.target_strides.row_stride, 16u);
  EXPECT_EQ(observed.region.target_strides.slice_stride, 64u);
}

TEST_F(CommandRecordingTest, StridedCopiesPreserveGapsAndSelectOneWorkstream) {
  std::array<uint8_t, 128> source_contents;
  for (size_t i = 0; i < source_contents.size(); ++i) {
    source_contents[i] = static_cast<uint8_t>(i + 1);
  }
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_h2d_linear(
      queue_, source_contents.data(), source_, 0, source_contents.size())));
  hrx_command_fill_t clear = {
      .command = {HRX_COMMAND_FILL},
      .pattern_size = 1,
      .target = iree_hal_make_buffer_ref(target_.get()->hal_buffer, 0, 128),
  };
  hrx_command_copy_strided_t copy = {
      .copy =
          {
              .command = {HRX_COMMAND_COPY},
              .strided = true,
              .source_buffer = source_.get()->hal_buffer,
              .source_offset = 7,
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
  void* sources[] = {&clear, &copy, &other_stream};
  uint32_t order[] = {0, 1, 2};
  uint8_t flags[] = {0, HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE,
                     1 << HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT};
  const hrx_graph_schedule_t schedule = {
      .node_count = 3,
      .sources = sources,
      .order = order,
      .flags = flags,
  };
  const hrx_graph_partition_t partition = {
      .count = 3,
      .stream_count = 2,
      .command_mask = (1u << HRX_COMMAND_COPY) | (1u << HRX_COMMAND_FILL),
  };
  RecordingCommandBuffer recorded(iree_hal_queue_family(queue_->hal_queue));
  IREE_ASSERT_OK(
      hrx_graph_record_partition(&schedule, &partition, 0, &recorded.base));
  ASSERT_EQ(recorded.copies.size(), 1u);
  const RecordedCopy& observed = recorded.copies.front();
  EXPECT_EQ(observed.source.buffer, source_.get()->hal_buffer);
  EXPECT_EQ(observed.source.offset, 7u);
  EXPECT_EQ(observed.source.length, 52u);
  EXPECT_EQ(observed.target.buffer, target_.get()->hal_buffer);
  EXPECT_EQ(observed.target.offset, 4u);
  EXPECT_EQ(observed.target.length, 76u);
  EXPECT_EQ(observed.region.row_length, 4u);
  EXPECT_EQ(observed.region.row_count, 3u);
  EXPECT_EQ(observed.region.slice_count, 2u);
  EXPECT_EQ(observed.region.source_strides.row_stride, 8u);
  EXPECT_EQ(observed.region.source_strides.slice_stride, 32u);
  EXPECT_EQ(observed.region.target_strides.row_stride, 12u);
  EXPECT_EQ(observed.region.target_strides.slice_stride, 48u);
  EXPECT_EQ(observed.flags, IREE_HAL_COPY_FLAG_NONE);
  IREE_ASSERT_OK(
      hrx_graph_record_partition(&schedule, &partition, 0, command_buffer_));
  Execute();
  std::array<uint8_t, 128> contents;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_synchronous_d2h_linear(
      queue_, target_, 0, contents.data(), contents.size())));
  std::array<uint8_t, 128> expected = {};
  for (size_t slice = 0; slice < 2; ++slice) {
    for (size_t row = 0; row < 3; ++row) {
      for (size_t byte = 0; byte < 4; ++byte) {
        expected[4 + slice * 48 + row * 12 + byte] =
            source_contents[7 + slice * 32 + row * 8 + byte];
      }
    }
  }
  EXPECT_EQ(contents, expected);
}

}  // namespace
