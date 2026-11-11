// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <semaphore>
#include <utility>

#include "iree/base/threading/processor.h"
#include "iree/hal/driver/amd/gpu/completion_test.h"

namespace {

using ::iree::hal::amd::gpu::testing::CompletionTest;

class CompletionObservationTest : public CompletionTest,
                                  public ::testing::WithParamInterface<bool> {
 protected:
  void AwaitObservation() {
    // Before this fixture requests any stop, replacement, or callout, only
    // observation can publish a nonzero lifecycle state. The complete source
    // scan precedes that publication, including every passive hint snapshot.
    while (!iree_atomic_load(&completion_.state, iree_memory_order_acquire)) {
      iree_processor_yield();
    }
  }
};

TEST_P(CompletionObservationTest, PassivePublicationInterruptsObservation) {
  iree_atomic_int32_t pending_word = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_int32_t word32 = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_int64_t word64 = IREE_ATOMIC_VAR_INIT(0);
  const bool wide = GetParam();
  std::atomic<bool> pending{true};
  std::binary_semaphore retired{0};
  bool first_round = true;
  bool reported = false;
  WaitCallback outstanding = [&](amdf_status_t,
                                 iree_hal_amd_gpu_completion_wait_t* wait) {
    if (pending.load()) {
      *wait = {.address = &pending_word, .value = 0, .pending = true};
    }
    return std::exchange(first_round, false);
  };
  WaitCallback passive = [&](amdf_status_t,
                             iree_hal_amd_gpu_completion_wait_t* wait) {
    const int64_t value =
        wide ? iree_atomic_load(&word64, iree_memory_order_acquire)
             : iree_atomic_load(&word32, iree_memory_order_acquire);
    *wait = {
        .address = wide ? static_cast<const void*>(&word64)
                        : static_cast<const void*>(&word32),
        .value = value,
        .is_64_bit = wide,
    };
    if (value && !std::exchange(reported, true)) {
      retired.release();
      return true;
    }
    return false;
  };
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(outstanding),
                                                    Source(passive)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  AwaitObservation();
  // The initially idle source changes after the worker has captured its hint.
  // Neither the older pending word nor a native event needs to change.
  if (wide) {
    iree_atomic_store(&word64, INT64_C(1) << 40, iree_memory_order_release);
  } else {
    iree_atomic_store(&word32, INT32_MIN, iree_memory_order_release);
  }
  retired.acquire();
  EXPECT_EQ(iree_atomic_load(&pending_word, iree_memory_order_relaxed), 0);
  EXPECT_EQ(event_.signal_count(), 0u);
  pending.store(false);
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  // The passive word remains installed. It cannot burn an infinite idle tail.
  event_.AwaitWaitCount(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
}

TEST_P(CompletionObservationTest, HostWakeInterruptsObservation) {
  iree_atomic_int32_t word32 = IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_int64_t word64 = IREE_ATOMIC_VAR_INIT(0);
  const bool wide = GetParam();
  std::atomic<bool> ready{false};
  std::binary_semaphore retired{0};
  bool first_round = true;
  WaitCallback outstanding = [&](amdf_status_t,
                                 iree_hal_amd_gpu_completion_wait_t* wait) {
    *wait = {
        .address = wide ? static_cast<const void*>(&word64)
                        : static_cast<const void*>(&word32),
        .value = 0,
        .is_64_bit = wide,
        .pending = true,
    };
    return std::exchange(first_round, false);
  };
  Callback other = [&](amdf_status_t) {
    if (ready.exchange(false)) {
      retired.release();
      return true;
    }
    return false;
  };
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(outstanding),
                                                    Source(other)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  AwaitObservation();
  ready.store(true);
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  retired.acquire();
  EXPECT_EQ(iree_atomic_load(&word32, iree_memory_order_relaxed), 0);
  EXPECT_EQ(iree_atomic_load(&word64, iree_memory_order_relaxed), 0);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
}

TEST_F(CompletionTest, HostWakeDuringSourceDrainRemainsObservable) {
  iree_atomic_int32_t word = IREE_ATOMIC_VAR_INIT(0);
  std::atomic<bool> ready{false};
  std::binary_semaphore scanning{0};
  std::binary_semaphore resume{0};
  std::binary_semaphore retired{0};
  uint32_t round = 0;
  Callback other = [&](amdf_status_t) {
    if (ready.exchange(false)) {
      retired.release();
      return true;
    }
    return false;
  };
  WaitCallback outstanding = [&](amdf_status_t,
                                 iree_hal_amd_gpu_completion_wait_t* wait) {
    *wait = {.address = &word, .value = 0, .pending = true};
    const uint32_t current_round = round++;
    if (current_round == 1) {
      // The independent source has been inspected, but observation has not
      // started. A wake in this gap must survive the transition to observation.
      scanning.release();
      resume.acquire();
    }
    return current_round == 0;
  };
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(other),
                                                    Source(outstanding)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  scanning.acquire();
  ready.store(true);
  EXPECT_EQ(iree_hal_amd_gpu_context_signal(&context_), AMDF_STATUS_OK);
  resume.release();
  retired.acquire();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
}

INSTANTIATE_TEST_SUITE_P(WordWidth, CompletionObservationTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Int64" : "Int32";
                         });

}  // namespace
