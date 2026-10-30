// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream.h"

#include <array>
#include <barrier>
#include <thread>

#include "common/context.h"
#include "common/device.h"
#include "common/hrx_bridge.h"
#include "common/module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

class StreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(0)));
    hrx_device_t device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(0, &device)));
    device_entry_.hrx_device = device;
    device_entry_.hal_device = hrx_device_hal(device);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &device_entry_.block_pool);
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_,
        {.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO},
        iree_allocator_system(), &context_));
  }

  void TearDown() override {
    iree_hal_streaming_context_release(context_);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  // CPU registry entry borrowed by the context until synchronous teardown.
  iree_hal_streaming_device_t device_entry_ = {};
  // Real context shared by independent application threads and streams.
  iree_hal_streaming_context_t* context_ = nullptr;
};

TEST_F(StreamTest, ConcurrentFirstUseAndCompletion) {
  constexpr size_t kThreadCount = 8;
  std::barrier<> start(static_cast<std::ptrdiff_t>(kThreadCount));
  std::array<iree_hal_streaming_stream_t*, kThreadCount> streams = {};
  std::array<std::thread, kThreadCount> threads;
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads[i] = std::thread([&, i] {
      start.arrive_and_wait();
      IREE_ASSERT_OK(iree_hal_streaming_stream_create(
          context_, context_->queue,
          IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING, 0,
          iree_allocator_system(), &streams[i]));

      // A rejected launch still enters the stream's submission policy path.
      // No executable is needed to exercise concurrent launch admission.
      iree_hal_streaming_symbol_t symbol = {
          .type = IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL,
      };
      const iree_hal_streaming_dispatch_params_t params = {};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          iree_hal_streaming_launch_kernel(&symbol, &params, streams[i]));

      // Submit a real command buffer and join its queue completion. Independent
      // streams must not mutate shared launch/flush/synchronization policy.
      IREE_ASSERT_OK(iree_hal_streaming_stream_begin(streams[i]));
      IREE_ASSERT_OK(iree_hal_streaming_stream_flush(streams[i]));
      IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize_flushed(streams[i]));
      EXPECT_EQ(1u, streams[i]->pending_value);
      EXPECT_EQ(streams[i]->pending_value, streams[i]->completed_value);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto* stream : streams) {
    iree_hal_streaming_stream_release(stream);
  }
}

}  // namespace
