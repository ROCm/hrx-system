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
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <utility>
#include <vector>

#include "iree/base/threading/processor.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"
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
            [](void* user_data, amdf_status_t status,
               iree_hal_amd_gpu_completion_wait_t* out_wait,
               iree_hal_amd_gpu_completion_copy_t* out_copy) {
              return (*static_cast<Callback*>(user_data))(status);
            },
        .user_data = &callback,
    };
  }

  using WaitCallback =
      std::function<bool(amdf_status_t, iree_hal_amd_gpu_completion_wait_t*)>;
  iree_hal_amd_gpu_completion_source_t Source(WaitCallback& callback) {
    return {
        .fn =
            [](void* user_data, amdf_status_t status,
               iree_hal_amd_gpu_completion_wait_t* out_wait,
               iree_hal_amd_gpu_completion_copy_t* out_copy) {
              return (*static_cast<WaitCallback*>(user_data))(status, out_wait);
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
      &context_, 0, nullptr, 0, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(event_.signal_count(), 1u);
  EXPECT_EQ(completion_.thread, nullptr);
}

TEST_F(CompletionTest, CoherentChangeProgressesWithoutNativeWake) {
  iree_atomic_int32_t word = IREE_ATOMIC_VAR_INIT(0);
  std::binary_semaphore armed{0};
  uint32_t calls = 0;
  uint32_t other_calls = 0;
  WaitCallback callback = [&](amdf_status_t status,
                              iree_hal_amd_gpu_completion_wait_t* out_wait) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    ++calls;
    if (iree_atomic_load(&word, iree_memory_order_acquire)) {
      EXPECT_TRUE(amdf_status_is_ok(
          iree_hal_amd_gpu_completion_request_stop(&completion_)));
      return false;
    }
    *out_wait = {.address = &word, .value = 0};
    if (calls == 2) {
      armed.release();
    }
    return calls == 1;
  };
  Callback other = [&](amdf_status_t status) {
    ++other_calls;
    return false;
  };
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(callback),
                                                    Source(other)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  armed.acquire();
  iree_atomic_store(&word, 1, iree_memory_order_release);
  // Joining directly avoids a controller stop racing the changed-word drain.
  iree_thread_release(completion_.thread);
  completion_.thread = nullptr;
  iree_slim_mutex_deinitialize(&completion_.drain_mutex);
  EXPECT_EQ(calls, 3u);
  EXPECT_EQ(other_calls, 3u);
  EXPECT_EQ(event_.signal_count(), 1u);  // Only the callback's final stop.
}

// Models preemption inside a bounded CPU copy. The worker, helper, source-table
// handoff, and final join are real; only the CPU transform dependency is gated.
struct StagingCopy {
  // Enables the next copy after the source's initial idle observation.
  std::atomic<bool> ready{true};
  // Observes entry into the transform before changing any owner state.
  std::binary_semaphore entered{0};
  // Holds the transform's borrowed storage across a simulated preemption.
  std::binary_semaphore resume{0};
  // Published only after the transform's final storage access.
  std::atomic<bool> copied{false};
  // Counts source observations independently of transform executions.
  std::atomic<uint32_t> calls{0};
  // Counts actual CPU transforms; helper drains must not increment it.
  std::atomic<uint32_t> copies{0};
  // Counts terminal notifications after the worker joins the transform.
  std::atomic<uint32_t> failures{0};
  // Written before entered is released and immutable through resume.
  std::thread::id copy_thread;

  iree_hal_amd_gpu_completion_source_t source() {
    return {
        .fn =
            [](void* user_data, amdf_status_t status,
               iree_hal_amd_gpu_completion_wait_t*,
               iree_hal_amd_gpu_completion_copy_t* out_copy) {
              auto& copy = *static_cast<StagingCopy*>(user_data);
              ++copy.calls;
              if (!amdf_status_is_ok(status)) {
                EXPECT_TRUE(copy.copied.load());
                ++copy.failures;
              } else if (copy.ready.load() && !copy.copied.load() && out_copy) {
                *out_copy = {
                    .fn =
                        [](void* user_data) {
                          auto& copy = *static_cast<StagingCopy*>(user_data);
                          copy.copy_thread = std::this_thread::get_id();
                          ++copy.copies;
                          copy.entered.release();
                          copy.resume.acquire();
                          copy.copied.store(true);
                        },
                    .user_data = &copy,
                };
              }
              return false;
            },
        .user_data = this,
    };
  }
};

TEST_F(CompletionTest, CpuCopyExcludesOnlyItsOwnSource) {
  StagingCopy copy;
  std::atomic<uint32_t> other_calls{0};
  Callback other = [&](amdf_status_t status) {
    ++other_calls;
    return false;
  };
  iree_hal_amd_gpu_completion_source_t sources[] = {copy.source(),
                                                    Source(other)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, 0, {},
      iree_allocator_system(), &completion_));
  copy.entered.acquire();
  EXPECT_EQ(copy.calls.load(), 1u);
  EXPECT_EQ(other_calls.load(), 1u);
  EXPECT_NE(copy.copy_thread, std::this_thread::get_id());
  EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  EXPECT_EQ(copy.calls.load(), 1u);
  EXPECT_EQ(other_calls.load(), 2u);
  EXPECT_EQ(copy.copies.load(), 1u);
  copy.resume.release();
  event_.AwaitWaitCount(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_TRUE(copy.copied.load());
  EXPECT_EQ(copy.calls.load(), 2u);
}

TEST_F(CompletionTest, ContinuousDownloadsTakeTurnsAcrossSources) {
  struct Copies {
    // Worker stopped by the final bounded transform.
    iree_hal_amd_gpu_completion_t* completion;
    // Copy-source order, written only by the worker before done is released.
    std::array<uint32_t, 6> order = {};
    // Number of completed transforms in order.
    size_t count = 0;
    // Joins all writes to order before caller inspection.
    std::binary_semaphore done{0};
  } copies{&completion_};
  struct SourceState {
    // Shared bounded execution trace, live through worker shutdown.
    Copies* copies;
    // Source identity recorded for each transform.
    uint32_t ordinal;
  } states[] = {{&copies, 0}, {&copies, 1}, {&copies, 2}};
  iree_hal_amd_gpu_completion_source_t sources[IREE_ARRAYSIZE(states)] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(states); ++i) {
    sources[i] = {
        .fn =
            [](void* user_data, amdf_status_t,
               iree_hal_amd_gpu_completion_wait_t*,
               iree_hal_amd_gpu_completion_copy_t* out_copy) {
              if (out_copy) {
                *out_copy = {
                    .fn =
                        [](void* user_data) {
                          auto& source = *static_cast<SourceState*>(user_data);
                          auto& copies = *source.copies;
                          copies.order[copies.count++] = source.ordinal;
                          if (copies.count == copies.order.size()) {
                            EXPECT_TRUE(amdf_status_is_ok(
                                iree_hal_amd_gpu_completion_request_stop(
                                    copies.completion)));
                            copies.done.release();
                          }
                        },
                    .user_data = user_data,
                };
              }
              return false;
            },
        .user_data = &states[i],
    };
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, 0, {},
      iree_allocator_system(), &completion_));
  copies.done.acquire();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(copies.order, (std::array<uint32_t, 6>{0, 1, 2, 0, 1, 2}));
}

TEST_F(CompletionTest, HostHelperLeavesCpuCopyForWorker) {
  StagingCopy copy;
  copy.ready.store(false);
  auto source = copy.source();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  copy.ready.store(true);
  EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  EXPECT_EQ(copy.calls.load(), 2u);
  EXPECT_EQ(copy.copies.load(), 0u);
  EXPECT_TRUE(amdf_status_is_ok(NativeEvent::Signal(event_.handle())));
  copy.entered.acquire();
  EXPECT_NE(copy.copy_thread, std::this_thread::get_id());
  copy.resume.release();
  event_.AwaitWaitCount(2);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(copy.copies.load(), 1u);
}

TEST_F(CompletionTest, SourceReplacementJoinsCpuCopyBorrow) {
  StagingCopy copy;
  auto original =
      std::make_unique<iree_hal_amd_gpu_completion_source_t>(copy.source());
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, original.get(), 0, {}, iree_allocator_system(),
      &completion_));
  copy.entered.acquire();
  auto replacement = copy.source();
  const int32_t copying_state =
      iree_atomic_load(&completion_.state, iree_memory_order_acquire);
  std::binary_semaphore replacing{0};
  std::atomic<bool> replaced{false};
  std::thread owner([&] {
    replacing.release();
    iree_hal_amd_gpu_completion_set_sources(&completion_, 1, &replacement);
    EXPECT_TRUE(copy.copied.load());
    original.reset();
    replaced.store(true);
  });
  replacing.acquire();
  while (iree_atomic_load(&completion_.state, iree_memory_order_acquire) ==
         copying_state) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(replaced.load());
  copy.resume.release();
  owner.join();
  EXPECT_TRUE(replaced.load());
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(copy.copies.load(), 1u);
}

TEST_F(CompletionTest, TerminalFailureJoinsCpuCopyBeforeSourceNotification) {
  StagingCopy copy;
  auto source = copy.source();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
  copy.entered.acquire();
  iree_hal_amd_gpu_completion_request_failure(
      &completion_, amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  EXPECT_EQ(copy.failures.load(), 0u);
  copy.resume.release();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_TRUE(copy.copied.load());
  EXPECT_EQ(copy.failures.load(), 1u);
}

TEST_F(CompletionTest, StopJoinsCpuCopyBorrow) {
  StagingCopy copy;
  auto source = copy.source();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
  copy.entered.acquire();
  const int32_t copying_state =
      iree_atomic_load(&completion_.state, iree_memory_order_acquire);
  std::atomic<bool> joined{false};
  std::thread owner([&] {
    IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
    EXPECT_TRUE(copy.copied.load());
    joined.store(true);
  });
  while (iree_atomic_load(&completion_.state, iree_memory_order_acquire) ==
         copying_state) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(joined.load());
  copy.resume.release();
  owner.join();
  EXPECT_TRUE(joined.load());
  EXPECT_EQ(copy.copies.load(), 1u);
}

TEST_F(CompletionTest, SourceReplacementExcludesNewCpuCopyBorrows) {
  std::binary_semaphore draining{0};
  std::binary_semaphore resume{0};
  uint32_t gate_calls = 0;
  Callback gate = [&](amdf_status_t) {
    if (++gate_calls == 1) {
      draining.release();
      resume.acquire();
    }
    return false;
  };
  StagingCopy copy;
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(gate),
                                                    copy.source()};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, 0, {},
      iree_allocator_system(), &completion_));
  draining.acquire();
  const int32_t draining_state =
      iree_atomic_load(&completion_.state, iree_memory_order_acquire);
  Callback idle = [](amdf_status_t) { return false; };
  auto replacement = Source(idle);
  std::thread owner([&] {
    iree_hal_amd_gpu_completion_set_sources(&completion_, 1, &replacement);
  });
  while (iree_atomic_load(&completion_.state, iree_memory_order_acquire) ==
         draining_state) {
    std::this_thread::yield();
  }
  resume.release();
  owner.join();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(copy.copies.load(), 0u);
}

TEST_F(CompletionTest, EventOnlySourcesParkAfterProgress) {
  uint32_t calls = 0;
  Callback callback = [&](amdf_status_t status) { return ++calls == 1; };
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(calls, 2u);
}

TEST_F(CompletionTest, StopInterruptsActiveObservation) {
  iree_atomic_int32_t word = IREE_ATOMIC_VAR_INIT(0);
  std::binary_semaphore armed{0};
  uint32_t calls = 0;
  WaitCallback callback = [&](amdf_status_t status,
                              iree_hal_amd_gpu_completion_wait_t* out_wait) {
    *out_wait = {.address = &word, .value = 0};
    if (++calls == 2) {
      armed.release();
    }
    return calls == 1;
  };
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  armed.acquire();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(calls, 2u);
}

TEST_F(CompletionTest, ReplacementJoinsLastSourceAccess) {
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  std::binary_semaphore replacing{0};
  auto calls = std::make_unique<uint32_t>(0);
  Callback callback = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    if (*calls == 0) {
      entered.release();
      resume.acquire();
    }
    ++*calls;
    return false;
  };
  auto source =
      std::make_unique<iree_hal_amd_gpu_completion_source_t>(Source(callback));
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, source.get(), 0, {}, iree_allocator_system(),
      &completion_));
  entered.acquire();
  std::atomic<bool> replaced = false;
  std::thread owner([&] {
    replacing.release();
    iree_hal_amd_gpu_completion_set_sources(&completion_, 0, nullptr);
    EXPECT_EQ(*calls, 1u);
    calls.reset();
    source.reset();
    callback = {};
    replaced.store(true, std::memory_order_release);
  });
  replacing.acquire();
  EXPECT_FALSE(replaced.load(std::memory_order_acquire));
  resume.release();
  owner.join();
  EXPECT_TRUE(replaced.load());
  EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
}

TEST_F(CompletionTest, ReplacementInterruptsObservationAndJoinsBorrowedHint) {
  auto word = std::make_unique<iree_atomic_int32_t>();
  iree_atomic_store(word.get(), 0, iree_memory_order_relaxed);
  std::binary_semaphore armed{0};
  uint32_t calls = 0;
  WaitCallback callback = [&](amdf_status_t status,
                              iree_hal_amd_gpu_completion_wait_t* out_wait) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    *out_wait = {.address = word.get(), .value = 0};
    if (++calls == 2) {
      armed.release();
    }
    return calls == 1;
  };
  auto source =
      std::make_unique<iree_hal_amd_gpu_completion_source_t>(Source(callback));
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, source.get(), IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  armed.acquire();
  iree_hal_amd_gpu_completion_set_sources(&completion_, 0, nullptr);
  word.reset();
  source.reset();
  callback = {};
  EXPECT_EQ(calls, 2u);
  event_.AwaitWaitCount(1);

  // The same worker remains available for a newly acquired idle queue.
  uint32_t replacement_calls = 0;
  Callback replacement_callback = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    ++replacement_calls;
    return false;
  };
  auto replacement = Source(replacement_callback);
  iree_hal_amd_gpu_completion_set_sources(&completion_, 1, &replacement);
  EXPECT_TRUE(amdf_status_is_ok(NativeEvent::Signal(event_.handle())));
  event_.AwaitWaitCount(2);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(replacement_calls, 1u);
}

TEST_F(CompletionTest, ReplacementPreservesTerminalStop) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 0, nullptr, 0, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  EXPECT_TRUE(amdf_status_is_ok(
      iree_hal_amd_gpu_completion_request_stop(&completion_)));
  uint32_t calls = 0;
  Callback callback = [&](amdf_status_t status) {
    ++calls;
    return false;
  };
  auto source = Source(callback);
  iree_hal_amd_gpu_completion_set_sources(&completion_, 1, &source);
  EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(calls, 0u);
}

TEST_F(CompletionTest, ReplacementRearmsRemainingSources) {
  iree_atomic_int32_t word = IREE_ATOMIC_VAR_INIT(0);
  std::binary_semaphore armed{0};
  std::binary_semaphore rearmed{0};
  std::binary_semaphore retired{0};
  uint32_t calls = 0;
  WaitCallback callback = [&](amdf_status_t status,
                              iree_hal_amd_gpu_completion_wait_t* out_wait) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    ++calls;
    if (iree_atomic_load(&word, iree_memory_order_acquire)) {
      EXPECT_TRUE(amdf_status_is_ok(
          iree_hal_amd_gpu_completion_request_stop(&completion_)));
      retired.release();
      return false;
    }
    *out_wait = {.address = &word, .value = 0};
    if (calls == 2) {
      armed.release();
    } else if (calls == 3) {
      rearmed.release();
    }
    return calls == 1;
  };
  auto source =
      std::make_unique<iree_hal_amd_gpu_completion_source_t>(Source(callback));
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, source.get(), IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  armed.acquire();

  uint32_t added_calls = 0;
  Callback added = [&](amdf_status_t status) {
    ++added_calls;
    return false;
  };
  iree_hal_amd_gpu_completion_source_t replacement[] = {Source(callback),
                                                        Source(added)};
  iree_hal_amd_gpu_completion_set_sources(
      &completion_, IREE_ARRAYSIZE(replacement), replacement);
  source.reset();
  EXPECT_TRUE(amdf_status_is_ok(NativeEvent::Signal(event_.handle())));
  rearmed.acquire();
  // Existing work continues through its coherent hint after the cold table
  // handoff. This completion itself produces no additional native event.
  iree_atomic_store(&word, 1, iree_memory_order_release);
  retired.acquire();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(calls, 4u);
  EXPECT_EQ(added_calls, 2u);
}

TEST_F(CompletionTest, IdleAndStaleEventsDoNotRenewObservationBudget) {
  iree_atomic_int32_t word = IREE_ATOMIC_VAR_INIT(0);
  uint32_t calls = 0;
  WaitCallback callback = [&](amdf_status_t status,
                              iree_hal_amd_gpu_completion_wait_t* out_wait) {
    *out_wait = {.address = &word, .value = 0};
    return ++calls == 1;
  };
  auto source = Source(callback);
  // This explicitly tests budget expiry, not a deadline for valid GPU work.
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 1, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  // Even an unsatisfied source's changed hint cannot restart an expired budget.
  iree_atomic_store(&word, 1, iree_memory_order_release);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(amdf_status_is_ok(NativeEvent::Signal(event_.handle())));
    event_.AwaitWaitCount(i + 2);
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(calls, 6u);
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
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(first),
                                                    Source(second)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, 0, {},
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
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
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
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(consumer),
                                                    Source(producer)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, 0, {},
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
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
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
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
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
    EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
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
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(first),
                                                    Source(second)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, 0, {},
      iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  event_.SetWaitStatus(failure);
  delivered.acquire();
  EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  // A new cold source owner sees failure even if it missed the terminal round.
  iree_hal_amd_gpu_completion_set_sources(&completion_, 0, nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_completion_query_status(&completion_), failure);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(failures, (std::array<uint32_t, 2>{1, 1}));
  EXPECT_EQ(reports_, (std::vector<uint64_t>{failure}));
}

TEST_F(CompletionTest, FinalReleaseJoinsTerminalOwnerOutsideDrainMutex) {
  struct Terminal {
    // Completion owner whose worker executes this cold terminal join.
    iree_hal_amd_gpu_completion_t* completion;
    // Native stop has begun after the error source round.
    std::binary_semaphore entered{0};
    // Allows the test dependency to finish its synchronous native stop.
    std::binary_semaphore finish{0};
    // Final callback access, observed only after the destruction owner joins.
    bool final_access = false;
  } terminal{.completion = &completion_};
  context_.execution_failure.user_data = &terminal;
  context_.execution_failure.join = [](void* data, amdf_status_t status) {
    auto* terminal = static_cast<Terminal*>(data);
    EXPECT_EQ(status, amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
    const bool acquired =
        iree_slim_mutex_try_lock(&terminal->completion->drain_mutex);
    EXPECT_TRUE(acquired);
    if (acquired) {
      iree_slim_mutex_unlock(&terminal->completion->drain_mutex);
    }
    terminal->entered.release();
    terminal->finish.acquire();
    terminal->final_access = true;
  };
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 0, nullptr, 0, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  event_.SetWaitStatus(amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  terminal.entered.acquire();
  std::atomic<bool> joined = false;
  std::thread owner([&] {
    IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
    joined.store(true, std::memory_order_release);
  });
  event_.AwaitSignalCount(1);
  EXPECT_FALSE(joined.load(std::memory_order_acquire));
  terminal.finish.release();
  owner.join();
  EXPECT_TRUE(joined.load());
  EXPECT_TRUE(terminal.final_access);
}

TEST_F(CompletionTest,
       SourceFailureRequestsTerminalRoundWithoutRecursiveDrain) {
  std::atomic<bool> fail = false;
  uint32_t failures = 0;
  const auto native_failure =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  Callback callback = [&](amdf_status_t status) {
    if (!amdf_status_is_ok(status)) {
      EXPECT_EQ(status, native_failure);
      ++failures;
    } else if (fail.exchange(false)) {
      iree_hal_amd_gpu_completion_request_failure(&completion_, native_failure);
      EXPECT_EQ(iree_hal_amd_gpu_completion_request_stop(&completion_),
                AMDF_STATUS_OK);
    }
    return false;
  };
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  fail.store(true);
  iree_hal_amd_gpu_completion_try_drain(&completion_);
  EXPECT_EQ(iree_hal_amd_gpu_completion_query_status(&completion_),
            native_failure);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(failures, 1u);
}

TEST_F(CompletionTest, CallerRetiresWhileWorkerParks) {
  uint32_t calls = 0;
  uint32_t retired = 0;
  Callback callback = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    // A nested submission or continuation cannot recursively consume sources.
    EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
    if (++calls == 1) {
      return false;
    }
    ++retired;
    return true;
  };
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  EXPECT_TRUE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  EXPECT_EQ(retired, 1u);
  EXPECT_EQ(event_.signal_count(), 0u);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(calls, 2u);
}

// Real signal ownership and trim, with CPU storage standing in for the native
// coherent allocation. One cell makes premature reuse and retained capacity
// observable without allocating additional signal blocks.
struct SignalStorage {
  explicit SignalStorage(
      iree_hal_amd_gpu_signal_pool_notification_t notification = {}) {
    iree_hal_amd_gpu_signal_pool_initialize(notification, &pool);
    iree_hal_amd_gpu_signal_block_initialize(&pool, this, &value, 0x1000, 8, 0,
                                             &signal, 1, 1, frontiers, &block);
  }
  ~SignalStorage() {
    if (block.pool) {
      iree_hal_amd_gpu_signal_block_deinitialize(&block);
    }
    iree_hal_amd_gpu_signal_pool_deinitialize(&pool);
  }

  // Producer-owned native value, observed through the low coherent word.
  uint64_t value = 0;
  // Existing per-cell metadata, including the independently retained borrow.
  iree_hal_amd_gpu_signal_t signal = {};
  // One signal's immutable one-queue frontier plus its header.
  uint64_t frontiers[3] = {};
  // Actual single-cell capacity and concurrent return path.
  iree_hal_amd_gpu_signal_pool_t pool = {};
  // Actual independently trimmable block.
  iree_hal_amd_gpu_signal_block_t block = {};
};

// The producer reference keeps this counter live until it publishes completion.
// Waiting for the real observer pin avoids a test hook or elapsed-time guess.
static void AwaitObservation(iree_hal_amd_gpu_signal_t* signal) {
  while (iree_atomic_ref_count_load(&signal->state.live.ref_count) != 2) {
    iree_processor_yield();
  }
}

TEST_F(CompletionTest, CallerDrainsWhileWorkerObservesRetainedSignal) {
  SignalStorage storage;
  auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 1);
  auto* word = reinterpret_cast<iree_atomic_int32_t*>(signal->host_value);
  std::binary_semaphore retired{0};
  uint32_t calls = 0;
  WaitCallback callback = [&](amdf_status_t status,
                              iree_hal_amd_gpu_completion_wait_t* out_wait) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    ++calls;
    if (!signal) {
      return false;
    }
    if (!iree_atomic_load(word, iree_memory_order_acquire)) {
      iree_hal_amd_gpu_signal_release(signal);
      signal = nullptr;
      retired.release();
      return true;
    }
    *out_wait = {.address = word, .value = 1, .signal = signal};
    return calls == 1;
  };
  std::atomic<bool> other_ready = false;
  Callback other = [&](amdf_status_t status) {
    return other_ready.exchange(false);
  };
  iree_hal_amd_gpu_completion_source_t sources[] = {Source(callback),
                                                    Source(other)};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, IREE_ARRAYSIZE(sources), sources, IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  AwaitObservation(signal);
  other_ready.store(true);
  while (!iree_hal_amd_gpu_completion_try_drain(&completion_)) {
    iree_processor_yield();
  }
  EXPECT_EQ(iree_atomic_ref_count_load(&signal->state.live.ref_count), 2);
  EXPECT_EQ(iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 1),
            nullptr);
  EXPECT_FALSE(iree_hal_amd_gpu_signal_block_try_deinitialize(&storage.block));
  EXPECT_EQ(event_.signal_count(), 0u);

  // Native completion races a helper and the worker. Either can retire the
  // producer; only the observer's final read permits recycling the cell.
  iree_atomic_store(word, 0, iree_memory_order_release);
  iree_hal_amd_gpu_completion_try_drain(&completion_);
  retired.acquire();
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_TRUE(iree_hal_amd_gpu_signal_block_try_deinitialize(&storage.block));
}

TEST_F(CompletionTest, SourceReplacementReturnsObservationBeforeTrim) {
  auto storage = std::make_unique<SignalStorage>();
  auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage->pool, 1);
  uint32_t calls = 0;
  WaitCallback callback = [&](amdf_status_t status,
                              iree_hal_amd_gpu_completion_wait_t* out_wait) {
    *out_wait = {
        .address = reinterpret_cast<iree_atomic_int32_t*>(signal->host_value),
        .value = 1,
        .signal = signal,
    };
    return ++calls == 1;
  };
  auto source =
      std::make_unique<iree_hal_amd_gpu_completion_source_t>(Source(callback));
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, source.get(), IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  AwaitObservation(signal);
  iree_hal_amd_gpu_completion_set_sources(&completion_, 0, nullptr);
  EXPECT_EQ(iree_atomic_ref_count_load(&signal->state.live.ref_count), 1);
  // The source owner has separately joined its producer. Removal joins all
  // observer borrows, so its final signal return can now detach the block.
  iree_hal_amd_gpu_signal_release(signal);
  EXPECT_TRUE(iree_hal_amd_gpu_signal_block_try_deinitialize(&storage->block));
  storage.reset();
  source.reset();
  callback = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
}

TEST_F(CompletionTest, StopAndFailureReturnObservationSignals) {
  for (amdf_status_t failure :
       {AMDF_STATUS_OK, amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL)}) {
    SignalStorage storage;
    auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 1);
    uint32_t calls = 0;
    uint32_t failures = 0;
    WaitCallback callback = [&](amdf_status_t status,
                                iree_hal_amd_gpu_completion_wait_t* out_wait) {
      if (!amdf_status_is_ok(status)) {
        EXPECT_EQ(status, failure);
        EXPECT_EQ(iree_atomic_ref_count_load(&signal->state.live.ref_count), 1);
        ++failures;
        return false;
      }
      *out_wait = {
          .address = reinterpret_cast<iree_atomic_int32_t*>(signal->host_value),
          .value = 1,
          .signal = signal,
      };
      return ++calls == 1;
    };
    auto source = Source(callback);
    IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
        &context_, 1, &source, IREE_DURATION_INFINITE, {},
        iree_allocator_system(), &completion_));
    AwaitObservation(signal);
    if (!amdf_status_is_ok(failure)) {
      iree_hal_amd_gpu_completion_request_failure(&completion_, failure);
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
    EXPECT_EQ(failures, amdf_status_is_ok(failure) ? 0u : 1u);
    EXPECT_EQ(iree_atomic_ref_count_load(&signal->state.live.ref_count), 1);
    iree_hal_amd_gpu_signal_release(signal);
    EXPECT_TRUE(iree_hal_amd_gpu_signal_block_try_deinitialize(&storage.block));
  }
}

TEST_F(CompletionTest, HelperAndWorkerReuseOneSignalWithoutGrowingCapacity) {
  std::binary_semaphore returned{0};
  SignalStorage storage(
      {[](void* user_data) {
         static_cast<std::binary_semaphore*>(user_data)->release();
       },
       &returned});
  std::atomic<iree_hal_amd_gpu_signal_t*> pending = nullptr;
  bool new_signal = true;
  WaitCallback callback = [&](amdf_status_t status,
                              iree_hal_amd_gpu_completion_wait_t* out_wait) {
    auto* signal = pending.load();
    if (!signal) {
      return false;
    }
    auto* word = reinterpret_cast<iree_atomic_int32_t*>(signal->host_value);
    if (!iree_atomic_load(word, iree_memory_order_acquire)) {
      pending.store(nullptr);
      new_signal = true;
      iree_hal_amd_gpu_signal_release(signal);
      return true;
    }
    *out_wait = {.address = word, .value = 1, .signal = signal};
    return std::exchange(new_signal, false);
  };
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, IREE_DURATION_INFINITE, {},
      iree_allocator_system(), &completion_));
  for (uint32_t i = 0; i < 128; ++i) {
    auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 1);
    ASSERT_EQ(signal, &storage.signal);
    pending.store(signal);
    EXPECT_EQ(NativeEvent::Signal(event_.handle()), AMDF_STATUS_OK);
    AwaitObservation(signal);
    iree_atomic_store(
        reinterpret_cast<iree_atomic_int32_t*>(signal->host_value), 0,
        iree_memory_order_release);
    iree_hal_amd_gpu_completion_try_drain(&completion_);
    returned.acquire();
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(storage.pool.block_count, 1u);
  EXPECT_TRUE(iree_hal_amd_gpu_signal_block_try_deinitialize(&storage.block));
}

TEST_F(CompletionTest, StopJoinsCallerDrainBeforeOwnerExit) {
  std::binary_semaphore entered{0};
  std::binary_semaphore finish{0};
  std::atomic<bool> exited = false;
  uint32_t calls = 0;
  Callback callback = [&](amdf_status_t status) {
    EXPECT_TRUE(amdf_status_is_ok(status));
    if (++calls == 1) {
      return false;
    }
    entered.release();
    finish.acquire();
    return false;
  };
  auto source = Source(callback);
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 1, &source, 0, {}, iree_allocator_system(), &completion_));
  event_.AwaitWaitCount(1);
  std::thread caller([&] {
    EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  });
  entered.acquire();
  EXPECT_FALSE(iree_hal_amd_gpu_completion_try_drain(&completion_));
  std::thread owner([&] {
    IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
    exited.store(true, std::memory_order_release);
  });
  event_.AwaitSignalCount(1);
  EXPECT_FALSE(exited.load(std::memory_order_acquire));
  finish.release();
  caller.join();
  owner.join();
  EXPECT_TRUE(exited.load());
  EXPECT_EQ(calls, 2u);
}

TEST_F(CompletionTest, FailedStopWakePreservesTheOwnerForRetry) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context_, 0, nullptr, 0, {}, iree_allocator_system(), &completion_));
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
  Callback callback = [&](amdf_status_t status) {
    ++calls;
    return false;
  };
  auto source = Source(callback);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_completion_initialize(
          &context_, 1, &source, 0, {}, iree_allocator_null(), &completion_));
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  EXPECT_EQ(completion_.thread, nullptr);
  EXPECT_EQ(calls, 0u);
  EXPECT_EQ(event_.signal_count(), 0u);
}

TEST(CompletionNativeEventTest, KernelWakeSharesOneWorkerAndJoinsShutdown) {
  iree_hal_amd_gpu_context_t context = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_context_prepare_completion(
      &context, AMDF_QUEUE_PUBLICATION_MODE_KERNEL));
  ASSERT_NE(context.kernel_notification.type, AMDF_NATIVE_EVENT_TYPE_NONE);
  struct State {
    // Producer-owned work predicate, independent of event coalescing.
    std::atomic<bool> ready{false};
    // Joins each source observation before the next publication.
    std::binary_semaphore observed{0};
  } state;
  iree_hal_amd_gpu_completion_source_t source = {
      .fn =
          [](void* user_data, amdf_status_t status,
             iree_hal_amd_gpu_completion_wait_t* out_wait,
             iree_hal_amd_gpu_completion_copy_t* out_copy) {
            auto* state = static_cast<State*>(user_data);
            EXPECT_TRUE(amdf_status_is_ok(status));
            if (state->ready.exchange(false)) {
              state->observed.release();
            }
            return false;
          },
      .user_data = &state,
  };
  iree_hal_amd_gpu_completion_t completion = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_completion_initialize(
      &context, 1, &source, 0, {}, iree_allocator_system(), &completion));
  for (int i = 0; i < 128; ++i) {
    state.ready.store(true);
    EXPECT_TRUE(amdf_status_is_ok(iree_hal_amd_gpu_context_signal(&context)));
    state.observed.acquire();
  }
  IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion));
  iree_async_event_native_deinitialize(&context.kernel_event);
}

}  // namespace
