// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/completion.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Dependency model for the native coalescing event. The actual IREE worker,
// source scheduling, stop protocol, and thread join are the subject under test.
class NativeEvent {
 public:
  amdf_gpu_event_t* handle() {
    return reinterpret_cast<amdf_gpu_event_t*>(this);
  }

  static amdf_status_t AMDF_CALL Wait(amdf_gpu_event_t* event,
                                      uint64_t timeout_nanoseconds) {
    EXPECT_EQ(timeout_nanoseconds, AMDF_TIMEOUT_INFINITE);
    auto* self = reinterpret_cast<NativeEvent*>(event);
    std::unique_lock<std::mutex> lock(self->mutex_);
    ++self->wait_count_;
    self->condition_.notify_all();
    self->condition_.wait(lock, [&] {
      return self->signaled_ || !amdf_status_is_ok(self->wait_status_);
    });
    self->signaled_ = false;
    return self->wait_status_;
  }

  static amdf_status_t AMDF_CALL Signal(amdf_gpu_event_t* event) {
    auto* self = reinterpret_cast<NativeEvent*>(event);
    std::lock_guard<std::mutex> lock(self->mutex_);
    ++self->signal_count_;
    if (amdf_status_is_ok(self->signal_status_)) {
      self->signaled_ = true;
    }
    self->condition_.notify_all();
    return self->signal_status_;
  }

  void AwaitWaitCount(uint32_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return wait_count_ >= count; });
  }

  void AwaitSignalCount(uint32_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return signal_count_ >= count; });
  }

  void SetWaitStatus(amdf_status_t status) {
    std::lock_guard<std::mutex> lock(mutex_);
    wait_status_ = status;
    condition_.notify_all();
  }

  void SetSignalStatus(amdf_status_t status) {
    std::lock_guard<std::mutex> lock(mutex_);
    signal_status_ = status;
  }

  uint32_t signal_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return signal_count_;
  }

 private:
  // Serializes dependency event state and test observation.
  std::mutex mutex_;
  // Native wait and controller readiness, without elapsed-time assumptions.
  std::condition_variable condition_;
  // Pending coalesced wake, consumed by the sole waiter.
  bool signaled_ = false;
  // Number of calls entering the native wait boundary.
  uint32_t wait_count_ = 0;
  // Number of attempted host signals, including failures.
  uint32_t signal_count_ = 0;
  // Native failure delivered to the waiting worker, if any.
  amdf_status_t wait_status_ = AMDF_STATUS_OK;
  // Native failure returned by the host wake boundary, if any.
  amdf_status_t signal_status_ = AMDF_STATUS_OK;
};

class CompletionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    gpu_.event_wait = NativeEvent::Wait;
    gpu_.event_signal = NativeEvent::Signal;
    context_.gpu = &gpu_;
    context_.event = event_.handle();
    context_.event_sink = {
        .fn =
            [](void* user_data, const iree_hal_device_event_t* event) {
              auto* self = static_cast<CompletionTest*>(user_data);
              EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
              const auto* payload = reinterpret_cast<
                  const iree_hal_device_driver_failure_event_t*>(
                  event->payload.data);
              self->reports_.push_back(payload->backend_result_code);
            },
        .user_data = this,
    };
  }

  void TearDown() override {
    IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  }

  using Callback = std::function<bool(amdf_status_t)>;
  iree_hal_amd_gpu_completion_source_t Source(Callback& callback) {
    return {
        .fn =
            [](void* user_data, amdf_status_t status) {
              return (*static_cast<Callback*>(user_data))(status);
            },
        .user_data = &callback,
    };
  }

  // Event dependency outlives the actual worker's final join.
  NativeEvent event_;
  // Native API table with only the dependency boundaries used here.
  amdf_gpu_api_t gpu_ = {};
  // Borrowed context whose native construction is covered by context tests.
  iree_hal_amd_gpu_context_t context_ = {};
  // Actual completion owner, safe for teardown before initialization.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Native failures reported once by the worker before its source callbacks.
  std::vector<uint64_t> reports_;
};

TEST_F(CompletionTest, EmptyDeviceParksAndJoinsWithHostWake) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 0, nullptr, {}, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(event_.signal_count(), 1u);
  EXPECT_EQ(completion_.thread, nullptr);
}

TEST_F(CompletionTest, BoundedSourcesTakeTurnsBeforeParking) {
  std::vector<uint32_t> order;
  std::array<uint32_t, 2> remaining = {3, 2};
  Callback first = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    order.push_back(0);
    if (remaining[0]) {
      --remaining[0];
    }
    return remaining[0] != 0;
  };
  Callback second = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    order.push_back(1);
    if (remaining[1]) {
      --remaining[1];
    }
    return remaining[1] != 0;
  };
  const iree_hal_amd_gpu_completion_source_t sources[] = {Source(first),
                                                          Source(second)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, {}, {},
      iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(order, (std::vector<uint32_t>{0, 1, 0, 1, 0, 1}));
  EXPECT_EQ(remaining, (std::array<uint32_t, 2>{0, 0}));
}

TEST_F(CompletionTest, CoalescedWakeBeforeWaitPreservesTheFinalWork) {
  std::binary_semaphore observed_empty{0};
  std::binary_semaphore finish_observation{0};
  std::atomic<uint32_t> pending = 0;
  uint32_t calls = 0;
  uint32_t completed = 0;
  Callback callback = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    const uint32_t ready = pending.exchange(0, std::memory_order_acquire);
    if (++calls == 1) {
      EXPECT_EQ(ready, 0u);
      observed_empty.release();
      finish_observation.acquire();
    }
    completed += ready;
    return false;
  };
  const auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, {}, {}, iree_allocator_system(), &completion_));
  observed_empty.acquire();
  pending.store(1000, std::memory_order_release);
  for (uint32_t i = 0; i < 1000; ++i) {
    EXPECT_TRUE(amdf_status_is_ok(NativeEvent::Signal(event_.handle())));
  }
  finish_observation.release();
  event_.AwaitWaitCount(2);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(completed, 1000u);
  EXPECT_EQ(calls, 2u);
}

TEST_F(CompletionTest, LaterSourceRetirementUnblocksEarlierSourceBeforeWait) {
  bool producer_retired = false;
  bool consumer_retired = false;
  std::vector<uint32_t> order;
  Callback consumer = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    order.push_back(0);
    if (!producer_retired || consumer_retired) {
      return false;
    }
    consumer_retired = true;
    return true;
  };
  Callback producer = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    order.push_back(1);
    if (producer_retired) {
      return false;
    }
    producer_retired = true;
    // No work remains here, but the earlier source can now retire its payload.
    return true;
  };
  const iree_hal_amd_gpu_completion_source_t sources[] = {Source(consumer),
                                                          Source(producer)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, {}, {},
      iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_TRUE(consumer_retired);
  EXPECT_EQ(order, (std::vector<uint32_t>{0, 1, 0, 1, 0, 1}));
}

TEST_F(CompletionTest, StaleWakeRechecksWithoutInventingCompletion) {
  uint32_t calls = 0;
  Callback callback = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    ++calls;
    return false;
  };
  const auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, {}, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  EXPECT_TRUE(amdf_status_is_ok(NativeEvent::Signal(event_.handle())));
  event_.AwaitWaitCount(2);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(calls, 2u);
}

TEST_F(CompletionTest, DeinitializationJoinsTheLastSourceAccess) {
  std::binary_semaphore entered{0};
  std::binary_semaphore finish{0};
  uint32_t final_access = 0;
  Callback callback = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    entered.release();
    finish.acquire();
    ++final_access;
    return false;
  };
  const auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, {}, {}, iree_allocator_system(), &completion_));
  entered.acquire();
  std::atomic<bool> joined = false;
  std::thread owner([&] {
    IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
    joined.store(true, std::memory_order_release);
  });
  event_.AwaitSignalCount(1);
  EXPECT_FALSE(joined.load(std::memory_order_acquire));
  finish.release();
  owner.join();
  EXPECT_TRUE(joined.load());
  EXPECT_EQ(final_access, 1u);
}

TEST_F(CompletionTest, NativeWaitFailureReachesEverySourceAndStops) {
  const amdf_status_t failure =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  std::array<uint32_t, 2> failures = {};
  std::binary_semaphore delivered{0};
  Callback first = [&](amdf_status_t status) {
    if (amdf_status_is_ok(status)) {
      return false;
    }
    EXPECT_EQ(status, failure);
    ++failures[0];
    return true;
  };
  Callback second = [&](amdf_status_t status) {
    if (amdf_status_is_ok(status)) {
      return false;
    }
    EXPECT_EQ(status, failure);
    ++failures[1];
    delivered.release();
    return true;
  };
  const iree_hal_amd_gpu_completion_source_t sources[] = {Source(first),
                                                          Source(second)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, {}, {},
      iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  event_.SetWaitStatus(failure);
  delivered.acquire();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(failures, (std::array<uint32_t, 2>{1, 1}));
  EXPECT_EQ(reports_, (std::vector<uint64_t>{failure}));
}

TEST_F(CompletionTest, FailedStopWakePreservesTheOwnerForRetry) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 0, nullptr, {}, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  iree_thread_t* thread = completion_.thread;
  event_.SetSignalStatus(
      amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(completion_.thread, thread);
  event_.SetSignalStatus(AMDF_STATUS_OK);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(event_.signal_count(), 2u);
  EXPECT_EQ(completion_.thread, nullptr);
}

TEST_F(CompletionTest, FailedThreadAllocationRunsNoSource) {
  uint32_t calls = 0;
  uint32_t exits = 0;
  Callback callback = [&](amdf_status_t status) {
    ++calls;
    return false;
  };
  const auto source = Source(callback);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_completion_initialize(
          &context_, 1, &source,
          {[](void* data) { ++*static_cast<uint32_t*>(data); }, &exits}, {},
          iree_allocator_null(), &completion_));
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(completion_.thread, nullptr);
  EXPECT_EQ(calls, 0u);
  EXPECT_EQ(exits, 0u);
  EXPECT_EQ(event_.signal_count(), 0u);
}

TEST_F(CompletionTest, ExternalDeinitializationJoinsTheExitCallback) {
  struct State {
    // Controller observes entry after the worker's final source access.
    std::binary_semaphore entered{0};
    // Controller releases the final callback to return to the thread wrapper.
    std::binary_semaphore finish{0};
  } state;
  const iree_hal_amd_gpu_completion_exit_callback_t exit_callback = {
      .fn =
          [](void* data) {
            auto* state = static_cast<State*>(data);
            state->entered.release();
            state->finish.acquire();
          },
      .user_data = &state,
  };
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 0, nullptr, exit_callback, {}, iree_allocator_system(),
      &completion_));
  event_.AwaitWaitCount(1);
  std::atomic<bool> joined = false;
  std::thread owner([&] {
    IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
    joined.store(true, std::memory_order_release);
  });
  state.entered.acquire();
  EXPECT_FALSE(joined.load(std::memory_order_acquire));
  state.finish.release();
  owner.join();
  EXPECT_TRUE(joined.load());
}

TEST_F(CompletionTest, ExitCallbackCanReleaseTheWorkerAndItsAllocation) {
  struct Owner {
    // Embedded worker state released only by its final exit action.
    iree_hal_amd_gpu_completion_t completion;
    // Immutable source retained with the owner through the worker's exit.
    iree_hal_amd_gpu_completion_source_t source;
    // Models the constructor's ownership through initialize returning.
    std::binary_semaphore constructed{0};
  };
  // The detached worker can finish posting without borrowing a test stack.
  static std::binary_semaphore finalized{0};
  auto* owner = new Owner;
  owner->source = {
      .fn =
          [](void* data, amdf_status_t status) {
            auto* owner = static_cast<Owner*>(data);
            EXPECT_TRUE(amdf_status_is_ok(status));
            owner->constructed.acquire();
            EXPECT_TRUE(amdf_status_is_ok(
                iree_hal_amd_gpu_completion_request_stop(&owner->completion)));
            return false;
          },
      .user_data = owner,
  };
  const iree_hal_amd_gpu_completion_exit_callback_t exit_callback = {
      .fn =
          [](void* data) {
            auto* owner = static_cast<Owner*>(data);
            IREE_EXPECT_OK(
                iree_hal_amd_gpu_completion_deinitialize(&owner->completion));
            delete owner;
            finalized.release();
          },
      .user_data = owner,
  };
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &owner->source, exit_callback, {}, iree_allocator_system(),
      &owner->completion));
  owner->constructed.release();
  finalized.acquire();
}

}  // namespace
