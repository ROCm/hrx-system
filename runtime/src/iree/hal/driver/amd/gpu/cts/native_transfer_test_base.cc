// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/cts/native_transfer_test_base.h"

#include <cstring>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::amd::gpu::cts {

void GpuNativeTransferTest::RunConcurrentTransfers(size_t lane_count,
                                                   size_t iterations) {
  using iree::hal::cts::Ref;
  // Exceeds the bounded staging window in both native protocols, with an odd
  // tail to exercise partial publications and unaligned host/device ranges.
  constexpr size_t kTransferLength = 17 * 1024 * 1024 + 37;
  std::vector<Ref<iree_hal_buffer_t>> buffers(lane_count);
  std::vector<Ref<iree_hal_semaphore_t>> timelines(lane_count);
  for (size_t lane = 0; lane < lane_count; ++lane) {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(kTransferLength + 3, 4),
        iree_infinite_timeout(), buffers[lane].out()));
    CreateSemaphore(timelines[lane].out());
  }

  // Keep completed native transfers from returning staging space until all
  // first-batch callers have submitted. Each upload spans multiple staging
  // windows, so none can finish publication while retirement is held.
  uint64_t paused_value = 1;
  CompletionPause pause{reinterpret_cast<iree_async_semaphore_t*>(timeline_)};
  IREE_ASSERT_OK(pause.Arm(paused_value));
  IREE_ASSERT_OK(iree_hal_queue_barrier(
      queues_[1], {}, {1, &timeline_, &paused_value}, nullptr, 0));
  pause.entered.acquire();

  const auto* barriers = Barriers();
  std::counting_semaphore<> initial_submissions{0};
  std::vector<std::thread> threads;
  threads.reserve(lane_count);
  for (size_t lane = 0; lane < lane_count; ++lane) {
    threads.emplace_back([&, lane] {
      std::vector<uint8_t> source(kTransferLength);
      std::vector<uint8_t> target(kTransferLength + 2, 0xCC);
      for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>((i * 29) ^ (i >> 9) ^ lane);
      }
      auto* timeline = timelines[lane].get();
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        SCOPED_TRACE(::testing::Message()
                     << "lane " << lane << ", iteration " << iteration);
        uint64_t before = iteration * 2;
        uint64_t uploaded = before + 1;
        uint64_t downloaded = before + 2;
        source.front() = static_cast<uint8_t>(iteration);
        source.back() = static_cast<uint8_t>(lane ^ iteration);
        auto status = iree_hal_queue_upload(
            queues_[0], {iteration ? 1u : 0u, &timeline, &before},
            {1, &timeline, &uploaded}, source.data(), buffers[lane], 3,
            kTransferLength, barriers);
        uint64_t completion = 0;
        if (iree_status_is_ok(status)) {
          completion = uploaded;
          status = iree_hal_queue_download(
              queues_[0], {1, &timeline, &uploaded},
              {1, &timeline, &downloaded}, buffers[lane], 3, target.data() + 1,
              kTransferLength, barriers);
          if (iree_status_is_ok(status)) {
            completion = downloaded;
          }
        }
        // Failed admission also releases the owner before the assertion exits
        // this thread, so teardown cannot strand the completion callback.
        if (iteration == 0) {
          initial_submissions.release();
        }
        // Even rejected downloads must retain source bytes until the accepted
        // upload retires. Successful batches join the complete round trip.
        if (completion) {
          status = iree_status_join(
              status, iree_hal_semaphore_wait(
                          timeline, completion, iree_infinite_timeout(),
                          (lane + iteration) & 1 ? IREE_ASYNC_WAIT_FLAG_ACTIVE
                                                 : IREE_ASYNC_WAIT_FLAG_NONE));
        }
        IREE_ASSERT_OK(status);
        EXPECT_EQ(target.front(), 0xCC);
        EXPECT_EQ(target.back(), 0xCC);
        EXPECT_EQ(
            std::memcmp(source.data(), target.data() + 1, kTransferLength), 0);
      }
    });
  }
  for (size_t lane = 0; lane < lane_count; ++lane) {
    initial_submissions.acquire();
  }
  pause.Join();
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace iree::hal::amd::gpu::cts
