// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/direct_transfer.h"

#include <array>
#include <cstring>

#include "common/context.h"
#include "common/device.h"
#include "common/hrx_bridge.h"
#include "common/memory.h"
#include "common/stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

class ContextTransferTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params[2] = {{.uri = "task://"},
                                                  {.uri = "task://"}};
    const hrx_status_t status =
        hrx_device_group_create(runtime, GetParam() ? 2 : 1, params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
    for (size_t i = 0; i < contexts_.size(); ++i) {
      auto& device = devices_[i];
      device.hrx_device =
          hrx_device_group_get_device(group_, GetParam() ? i : 0);
      device.hal_device = hrx_device_hal(device.hrx_device);
      iree_slim_mutex_initialize(&device.primary_context_mutex);
      iree_slim_mutex_initialize(&device.graph_memory_mutex);
      iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                       &device.block_pool);
      ++initialized_devices_;
      const iree_hal_streaming_context_flags_t flags = {
          .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
      };
      IREE_ASSERT_OK(iree_hal_streaming_context_create(
          &device, flags, iree_allocator_system(), &contexts_[i]));
    }
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        contexts_[1], contexts_[1]->queue,
        IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING, 0, iree_allocator_system(),
        &stream_));
    IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_device(
        contexts_[0], 64, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer_));
    std::memset(buffer_->host_ptr, 0xCD, 64);
  }

  void TearDown() override {
    if (stream_) {
      IREE_EXPECT_STATUS_IS(expected_completion_,
                            iree_hal_streaming_stream_synchronize(stream_));
    }
    iree_hal_streaming_memory_release_wrapped_buffer(buffer_);
    iree_hal_streaming_stream_release(stream_);
    for (auto* context : contexts_) {
      iree_hal_streaming_context_release(context);
    }
    for (size_t i = 0; i < initialized_devices_; ++i) {
      iree_arena_block_pool_deinitialize(&devices_[i].block_pool);
      iree_slim_mutex_deinitialize(&devices_[i].graph_memory_mutex);
      iree_slim_mutex_deinitialize(&devices_[i].primary_context_mutex);
    }
    hrx_device_group_release(group_);
  }

  // Owns one or two real task devices through both compatibility contexts.
  hrx_device_group_t group_ = nullptr;
  // Device views and queue acquisition state borrowed by the contexts.
  std::array<iree_hal_streaming_device_t, 2> devices_ = {};
  // Prefix whose mutexes and arena need teardown after partial setup.
  size_t initialized_devices_ = 0;
  // Separate allocation and execution contexts, released before their devices.
  std::array<iree_hal_streaming_context_t*, 2> contexts_ = {};
  // Stream ordering copies whose buffer belongs to the first context.
  iree_hal_streaming_stream_t* stream_ = nullptr;
  // Device allocation retained until the stream is terminal.
  iree_hal_streaming_buffer_t* buffer_ = nullptr;
  // Expected terminal status, including a caller-failed prerequisite.
  iree_status_code_t expected_completion_ = IREE_STATUS_OK;
};

TEST_P(ContextTransferTest, BatchesPreserveOffsetsAndFollowingDownloads) {
  const std::array<uint8_t, 7> first = {1, 19, 3, 67, 5, 241, 7};
  const std::array<uint8_t, 3> second = {17, 85, 241};
  const iree_hal_transfer_operation_t uploads[] = {
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload =
              {
                  .source = first.data(),
                  .target_buffer = buffer_->buffer,
                  .target_offset = 3,
                  .length = first.size(),
              },
      },
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload =
              {
                  .source = second.data(),
                  .target_buffer = buffer_->buffer,
                  .target_offset = 39,
                  .length = second.size(),
              },
      },
  };
  IREE_ASSERT_OK(iree_hal_streaming_direct_transfer_on_context_and_wait(
      contexts_[0], stream_, IREE_ARRAYSIZE(uploads), uploads));
  std::array<uint8_t, 64> result = {};
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = buffer_->buffer,
              .source_offset = 0,
              .target = result.data(),
              .length = result.size(),
          },
  };
  IREE_ASSERT_OK(iree_hal_streaming_direct_transfer_on_context_and_wait(
      contexts_[0], stream_, 1, &download));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  std::array<uint8_t, 64> expected;
  expected.fill(0xCD);
  std::memcpy(expected.data() + 3, first.data(), first.size());
  std::memcpy(expected.data() + 39, second.data(), second.size());
  EXPECT_EQ(result, expected);
}

TEST_P(ContextTransferTest, FailedPredecessorPreservesBothEndpoints) {
  std::array<uint8_t, 16> input;
  input.fill(0x41);
  std::array<uint8_t, 16> output;
  output.fill(0x67);
  expected_completion_ = IREE_STATUS_ABORTED;
  IREE_EXPECT_STATUS_IS(
      expected_completion_,
      iree_hal_streaming_execute_host_operation(
          stream_,
          +[](void*) { return iree_status_from_code(IREE_STATUS_ABORTED); },
          nullptr));
  const iree_hal_transfer_operation_t operations[] = {
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload =
              {
                  .source = input.data(),
                  .target_buffer = buffer_->buffer,
                  .target_offset = 0,
                  .length = input.size(),
              },
      },
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = buffer_->buffer,
                  .source_offset = 0,
                  .target = output.data(),
                  .length = output.size(),
              },
      },
  };
  IREE_EXPECT_STATUS_IS(
      expected_completion_,
      iree_hal_streaming_direct_transfer_on_context_and_wait(
          contexts_[0], stream_, IREE_ARRAYSIZE(operations), operations));
  const auto* device_bytes = static_cast<const uint8_t*>(buffer_->host_ptr);
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(device_bytes[i], 0xCD);
    EXPECT_EQ(output[i], 0x67);
  }
}

INSTANTIATE_TEST_SUITE_P(NativeDevices, ContextTransferTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "DifferentDevices"
                                             : "SameDevice";
                         });

}  // namespace
