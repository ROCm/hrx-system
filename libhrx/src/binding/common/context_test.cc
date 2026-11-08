// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/context.h"

#include <array>
#include <barrier>
#include <thread>

#include "iree/testing/gtest.h"

namespace {

TEST(ContextFlagsTest, PreservesSchedulingAndMemoryFlags) {
  iree_hal_streaming_context_t context = {};
  const std::array<iree_async_wait_flags_t, 4> expected_wait_flags = {
      IREE_ASYNC_WAIT_FLAG_ACTIVE, IREE_ASYNC_WAIT_FLAG_ACTIVE,
      IREE_ASYNC_WAIT_FLAG_YIELD, IREE_ASYNC_WAIT_FLAG_NONE};
  for (uint32_t mode = 0; mode < expected_wait_flags.size(); ++mode) {
    for (uint32_t memory_flags = 0; memory_flags < 4; ++memory_flags) {
      const iree_hal_streaming_context_flags_t flags = {
          .scheduling_mode =
              static_cast<iree_hal_streaming_scheduling_mode_t>(mode),
          .map_host_memory = memory_flags & 1u,
          .resize_local_mem_to_max = memory_flags >> 1,
      };
      iree_hal_streaming_context_set_flags(&context, flags);
      const auto snapshot = iree_hal_streaming_context_flags(&context);
      EXPECT_EQ(flags.scheduling_mode, snapshot.scheduling_mode);
      EXPECT_EQ(flags.map_host_memory, snapshot.map_host_memory);
      EXPECT_EQ(flags.resize_local_mem_to_max,
                snapshot.resize_local_mem_to_max);
      EXPECT_EQ(expected_wait_flags[mode],
                iree_hal_streaming_context_wait_flags(&context));
    }
  }
}

TEST(ContextFlagsTest, ConcurrentUpdatesPublishCoherentSnapshots) {
  iree_hal_streaming_context_t context = {};
  std::array<std::thread, 4> threads;
  std::barrier<> start(threads.size());
  for (uint32_t mode = 0; mode < threads.size(); ++mode) {
    threads[mode] = std::thread([&, mode] {
      const iree_hal_streaming_context_flags_t flags = {
          .scheduling_mode =
              static_cast<iree_hal_streaming_scheduling_mode_t>(mode),
          .map_host_memory = mode & 1u,
          .resize_local_mem_to_max = mode >> 1,
      };
      start.arrive_and_wait();
      for (int i = 0; i < 1000; ++i) {
        iree_hal_streaming_context_set_flags(&context, flags);
        const auto snapshot = iree_hal_streaming_context_flags(&context);
        EXPECT_EQ(snapshot.scheduling_mode & 1u, snapshot.map_host_memory);
        EXPECT_EQ(snapshot.scheduling_mode >> 1,
                  snapshot.resize_local_mem_to_max);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace
