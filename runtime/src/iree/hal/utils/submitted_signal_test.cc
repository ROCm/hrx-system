// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/submitted_signal.h"

#include <atomic>
#include <thread>

#include "iree/testing/gtest.h"

namespace {

static iree_async_axis_t test_queue_axis(uint8_t queue_index) {
  return iree_async_axis_make_queue(/*session_epoch=*/1, /*machine_index=*/0,
                                    /*device_index=*/0, queue_index,
                                    /*queue_incarnation=*/0);
}

TEST(SubmittedSignalTest, ConcurrentLoadReturnsOneCompleteGeneration) {
  // Keep every published snapshot valid so returning the caller's initialized
  // outputs while a writer is active is distinguishable from an empty cache.
  constexpr uint64_t kGenerationCount = 65536;
  iree_hal_submitted_signal_t cache = {};
  auto publish = [&](uint64_t generation) {
    iree_hal_submitted_signal_store(
        &cache, IREE_HAL_SUBMITTED_SIGNAL_FLAG_VALID,
        test_queue_axis(static_cast<uint8_t>(generation % 16)), generation,
        generation * 17 + 5);
  };
  publish(1);

  std::atomic<bool> reader_ready{false};
  std::atomic<bool> writer_ready{false};
  std::thread writer([&] {
    writer_ready.store(true, std::memory_order_release);
    while (!reader_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (uint64_t generation = 2; generation <= kGenerationCount;
         ++generation) {
      publish(generation);
    }
  });
  reader_ready.store(true, std::memory_order_release);
  while (!writer_ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  uint64_t invalid_snapshot_count = 0;
  uint64_t last_epoch = 0;
  for (uint64_t i = 0; i < kGenerationCount; ++i) {
    iree_hal_submitted_signal_flags_t flags =
        IREE_HAL_SUBMITTED_SIGNAL_FLAG_NONE;
    iree_async_axis_t axis = 0;
    uint64_t epoch = 0;
    uint64_t value = 0;
    const bool valid =
        iree_hal_submitted_signal_load(&cache, &flags, &axis, &epoch, &value);
    if (!valid || epoch == 0 || epoch > kGenerationCount ||
        epoch < last_epoch ||
        axis != test_queue_axis(static_cast<uint8_t>(epoch % 16)) ||
        value != epoch * 17 + 5) {
      ++invalid_snapshot_count;
    }
    last_epoch = epoch;
  }
  writer.join();
  EXPECT_EQ(invalid_snapshot_count, 0u);
}

}  // namespace
