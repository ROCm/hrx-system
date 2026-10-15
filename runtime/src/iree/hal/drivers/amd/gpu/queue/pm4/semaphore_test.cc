// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/semaphore.h"

#include <array>
#include <atomic>
#include <barrier>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using Signal = iree_hal_amd_gpu_pm4_signal_t;
using Wait = iree_hal_amd_gpu_pm4_semaphore_wait_t;

// Controls only allocation, leaving semaphore state and synchronization intact.
struct Allocator {
  // Denies cold growth after the owning object has been constructed.
  std::atomic<bool> deny{false};
  // Number of currently owned allocations.
  std::atomic<int> live{0};

  iree_allocator_t get() { return {this, Control}; }
  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto& self = *static_cast<Allocator*>(user_data);
    if (command != IREE_ALLOCATOR_COMMAND_FREE && self.deny.load()) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "cold allocation denied");
    }
    const auto system = iree_allocator_system();
    const bool freeing = command == IREE_ALLOCATOR_COMMAND_FREE && *pointer;
    iree_status_t status =
        system.ctl(system.self, command, parameters, pointer);
    if (iree_status_is_ok(status)) {
      self.live.fetch_add(freeing ? -1 : *pointer ? 1 : 0);
    }
    return status;
  }
};

struct Completion {
  // Joins the service callback's last access to this result.
  std::binary_semaphore done{0};
  // Exactly one callback owns the terminal result.
  std::atomic<uint32_t> calls{0};
  // Owned callback status, read after done.
  iree_status_t status = iree_ok_status();

  ~Completion() { iree_status_free(status); }
  static void Run(void* user_data, iree_status_t status) {
    auto& self = *static_cast<Completion*>(user_data);
    self.status = status;
    self.calls.fetch_add(1);
    self.done.release();
  }
};

class SemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    iree_hal_amd_gpu_pm4_signal_pool_initialize(this, values_.data(), 0x1000,
                                                signals_.data(),
                                                signals_.size(), {}, &pool_);
  }
  void TearDown() override {
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_amd_gpu_pm4_signal_pool_deinitialize(&pool_);
    iree_async_proactor_release(proactor_);
  }
  iree_hal_semaphore_t* Semaphore(uint64_t initial_value = 0) {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_create(
        proactor_, initial_value, 16, iree_allocator_system(), &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }
  Signal* AcquireSignal() {
    return iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&pool_);
  }
  void Publish(iree_hal_semaphore_t* semaphore, uint64_t value, Signal* signal,
               iree_async_axis_t axis = 1) {
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_publish_signal(
        semaphore, value, {axis, value}, signal));
  }
  Wait AcquireWait(iree_hal_semaphore_t* semaphore, uint64_t value) {
    Wait wait;
    IREE_CHECK_OK(
        iree_hal_amd_gpu_pm4_semaphore_acquire_wait(semaphore, value, &wait));
    return wait;
  }
  uint64_t Value(iree_hal_semaphore_t* semaphore) {
    uint64_t value = 0;
    IREE_CHECK_OK(iree_hal_semaphore_query(semaphore, &value));
    return value;
  }
  iree_hal_semaphore_t* Submission(iree_hal_semaphore_t* semaphore) {
    iree_hal_semaphore_t* submission = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(
        semaphore, &submission));
    semaphores_.push_back(submission);
    return submission;
  }
  // Actual platform proactor, borrowed by all semaphores.
  iree_async_proactor_t* proactor_ = nullptr;
  // Host-mapped native control cells supplied by the allocation dependency.
  std::array<uint64_t, 8> values_;
  // Independently retained native signal metadata.
  std::array<Signal, 8> signals_;
  // Finite producer storage, reclaimed only after all uses join.
  iree_hal_amd_gpu_pm4_signal_pool_t pool_;
  // Test-owned references, released before native backing.
  std::vector<iree_hal_semaphore_t*> semaphores_;
};

TEST_F(SemaphoreTest, HostTimelinePreservesStrictSignalsAndWaits) {
  auto* semaphore = Semaphore(2);
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_semaphore_isa(semaphore));
  EXPECT_EQ(Value(semaphore), 2u);
  EXPECT_EQ(AcquireWait(semaphore, 2).state,
            IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_COMPLETED);
  EXPECT_EQ(AcquireWait(semaphore, 3).state,
            IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_DEFERRED);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 3, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, 3, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_semaphore_signal(semaphore, 3, nullptr));
}

TEST_F(SemaphoreTest, SubmittedProducerDoesNotBecomeCompletedHistory) {
  auto* semaphore = Semaphore();
  iree_async_single_frontier_t frontier;
  iree_async_single_frontier_initialize(&frontier, 7, 10);
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_semaphore_publish_completed(
      semaphore, 1, iree_async_single_frontier_as_const_frontier(&frontier)));
  auto* signal = AcquireSignal();
  Publish(semaphore, 2, signal, 9);
  EXPECT_EQ(Value(semaphore), 1u);
  EXPECT_EQ(AcquireWait(semaphore, 1).state,
            IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_COMPLETED);
  auto wait = AcquireWait(semaphore, 2);
  EXPECT_EQ(wait.state, IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_NATIVE);
  EXPECT_EQ(wait.producer.axis, 9u);
  EXPECT_EQ(wait.producer.epoch, 2u);
  EXPECT_EQ(iree_async_semaphore_query_frontier(
                reinterpret_cast<iree_async_semaphore_t*>(semaphore),
                iree_async_single_frontier_as_frontier(&frontier), 1),
            1u);
  EXPECT_EQ(frontier.entries[0].axis, 7u);
  EXPECT_EQ(frontier.entries[0].epoch, 10u);
  iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  iree_hal_amd_gpu_pm4_signal_release(signal);
}

TEST_F(SemaphoreTest, ReaderOutlivesProducerAndSemaphore) {
  auto* semaphore = Semaphore();
  auto* signal = AcquireSignal();
  Publish(semaphore, 1, signal);
  auto wait = AcquireWait(semaphore, 1);
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_semaphore_publish_completed(semaphore, 1, nullptr));
  iree_hal_amd_gpu_pm4_signal_release(signal);
  iree_hal_semaphore_release(semaphores_.back());
  semaphores_.pop_back();
  EXPECT_EQ(pool_.live_count, 1u);
  EXPECT_EQ(wait.signal->device_address, 0x1000u);
  iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  EXPECT_EQ(pool_.live_count, 0u);
}

TEST_F(SemaphoreTest, UnassignedTopologyStillProvidesNativeDependency) {
  auto* semaphore = Semaphore();
  auto* signal = AcquireSignal();
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_semaphore_publish_signal(semaphore, 1, {}, signal));
  auto wait = AcquireWait(semaphore, 1);
  EXPECT_EQ(wait.state, IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_NATIVE);
  EXPECT_EQ(wait.producer.epoch, 0u);
  EXPECT_EQ(wait.signal, signal);
  iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  iree_hal_amd_gpu_pm4_signal_release(signal);
}

TEST_F(SemaphoreTest, ReplacingCachePreservesEveryNativeReader) {
  auto* semaphore = Semaphore();
  auto* first = AcquireSignal();
  auto* second = AcquireSignal();
  Publish(semaphore, 1, first);
  auto first_wait = AcquireWait(semaphore, 1);
  Publish(semaphore, 2, second, 2);
  iree_hal_amd_gpu_pm4_signal_release(first);
  auto second_wait = AcquireWait(semaphore, 1);
  EXPECT_EQ(first_wait.signal, first);
  EXPECT_EQ(second_wait.signal, second);
  EXPECT_EQ(second_wait.producer.axis, 2u);
  EXPECT_EQ(pool_.live_count, 2u);
  iree_hal_amd_gpu_pm4_signal_release(first_wait.signal);
  EXPECT_EQ(pool_.live_count, 1u);
  iree_hal_amd_gpu_pm4_signal_release(second_wait.signal);
  iree_hal_amd_gpu_pm4_signal_release(second);
}

TEST_F(SemaphoreTest, CompletionBeforePublicationCannotInstallStaleCache) {
  auto* semaphore = Semaphore();
  auto* signal = AcquireSignal();
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_semaphore_publish_completed(semaphore, 1, nullptr));
  Publish(semaphore, 1, signal);
  EXPECT_EQ(AcquireWait(semaphore, 1).state,
            IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_COMPLETED);
  iree_hal_amd_gpu_pm4_signal_release(signal);
  EXPECT_EQ(pool_.live_count, 0u);
}

TEST_F(SemaphoreTest, DelayedPublicationCannotReplaceHigherProducer) {
  auto* semaphore = Semaphore();
  auto* first = AcquireSignal();
  auto* second = AcquireSignal();
  Publish(semaphore, 2, second, 2);
  Publish(semaphore, 1, first);
  iree_hal_amd_gpu_pm4_signal_release(first);
  EXPECT_EQ(pool_.live_count, 1u);
  auto wait = AcquireWait(semaphore, 1);
  EXPECT_EQ(wait.signal, second);
  EXPECT_EQ(wait.producer.epoch, 2u);
  iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  iree_hal_amd_gpu_pm4_signal_release(second);
}

TEST_F(SemaphoreTest, CompletionObserversCanArriveInReverseOrder) {
  auto* semaphore = Semaphore();
  auto* signal = AcquireSignal();
  Publish(semaphore, 2, signal);
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_semaphore_publish_completed(semaphore, 2, nullptr));
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_pm4_semaphore_publish_completed(semaphore, 1, nullptr));
  EXPECT_EQ(Value(semaphore), 2u);
  iree_hal_amd_gpu_pm4_signal_release(signal);
  EXPECT_EQ(pool_.live_count, 0u);
}

TEST_F(SemaphoreTest, HostSignalClearsOnlyCoveredCache) {
  auto* semaphore = Semaphore();
  auto* signal = AcquireSignal();
  Publish(semaphore, 2, signal);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));
  auto wait = AcquireWait(semaphore, 2);
  EXPECT_EQ(wait.signal, signal);
  iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 2, nullptr));
  iree_hal_amd_gpu_pm4_signal_release(signal);
  EXPECT_EQ(pool_.live_count, 0u);
}

TEST_F(SemaphoreTest, ColdTimelineStartsFromPublishedOrCompletedValue) {
  auto* completed = Semaphore(4);
  auto* completed_submission = Submission(completed);
  EXPECT_EQ(Value(completed_submission), 4u);
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_semaphore_isa(completed_submission));
  auto* submitted = Semaphore();
  auto* signal = AcquireSignal();
  Publish(submitted, 5, signal);
  auto* submission = Submission(submitted);
  EXPECT_EQ(Value(submission), 5u);
  EXPECT_EQ(Value(submitted), 0u);
  EXPECT_EQ(Submission(submitted), submission);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(submitted, 6, nullptr));
  EXPECT_EQ(Value(submission), 6u);
  iree_hal_amd_gpu_pm4_signal_release(signal);
}

TEST_F(SemaphoreTest, FuturePublicationResumesOnServiceBeforeCompletion) {
  auto* semaphore = Semaphore();
  auto* submission = Submission(semaphore);
  iree_hal_memory_maintenance_t* service = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create(
      {}, iree_allocator_system(), &service));
  iree_async_semaphore_wait_context_t context;
  iree_async_semaphore_wait_context_initialize(&context);
  uint64_t value = 1;
  Completion completion;
  iree_hal_amd_gpu_host_wait_t wait;
  IREE_ASSERT_OK(iree_hal_amd_gpu_host_wait_begin(
      &context, service, {1, &submission, &value}, Completion::Run, &completion,
      iree_allocator_system(), &wait));
  EXPECT_EQ(completion.calls.load(), 0u);
  auto* signal = AcquireSignal();
  Publish(semaphore, value, signal);
  completion.done.acquire();
  EXPECT_EQ(iree_status_code(completion.status), IREE_STATUS_OK);
  EXPECT_EQ(completion.calls.load(), 1u);
  EXPECT_EQ(Value(semaphore), 0u);
  auto native_wait = AcquireWait(semaphore, value);
  EXPECT_EQ(native_wait.signal, signal);
  iree_hal_amd_gpu_pm4_signal_release(native_wait.signal);
  iree_hal_amd_gpu_pm4_signal_release(signal);
  iree_hal_memory_maintenance_release(service);
  iree_async_semaphore_wait_context_deinitialize(&context);
}

TEST_F(SemaphoreTest, FailureDoesNotReclaimNativeReaders) {
  auto* semaphore = Semaphore();
  auto* submission = Submission(semaphore);
  auto* signal = AcquireSignal();
  Publish(semaphore, 1, signal);
  auto wait = AcquireWait(semaphore, 1);
  iree_hal_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "native producer"));
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_query(submission, &value));
  Wait failed_wait;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_amd_gpu_pm4_semaphore_acquire_wait(semaphore, 1, &failed_wait));
  EXPECT_EQ(failed_wait.signal, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_signal(semaphore, 2, nullptr));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_amd_gpu_pm4_semaphore_publish_signal(
                            semaphore, 2, {1, 2}, signal));
  iree_hal_amd_gpu_pm4_signal_release(signal);
  EXPECT_EQ(pool_.live_count, 1u);
  iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  EXPECT_EQ(pool_.live_count, 0u);
}

TEST_F(SemaphoreTest, ColdAllocationFailureLeavesReadyPathUsable) {
  Allocator allocator;
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_semaphore_create(
      proactor_, 0, 4, allocator.get(), &semaphore));
  allocator.deny.store(true);
  iree_hal_semaphore_t* submission = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(semaphore,
                                                                  &submission));
  EXPECT_EQ(submission, nullptr);
  auto* signal = AcquireSignal();
  Publish(semaphore, 1, signal);
  auto wait = AcquireWait(semaphore, 1);
  EXPECT_EQ(wait.signal, signal);
  iree_hal_amd_gpu_pm4_signal_release(wait.signal);
  iree_hal_amd_gpu_pm4_signal_release(signal);
  iree_hal_semaphore_release(semaphore);
  EXPECT_EQ(allocator.live.load(), 0);
}

TEST_F(SemaphoreTest, TerminalFailurePrecedesColdAllocation) {
  Allocator allocator;
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_semaphore_create(
      proactor_, 0, 4, allocator.get(), &semaphore));
  iree_hal_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "native producer"));
  allocator.deny.store(true);
  iree_hal_semaphore_t* submission = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(semaphore,
                                                                  &submission));
  EXPECT_EQ(submission, nullptr);
  EXPECT_EQ(allocator.live.load(), 1);
  iree_hal_semaphore_release(semaphore);
  EXPECT_EQ(allocator.live.load(), 0);
}

TEST_F(SemaphoreTest, FutureWaitFailureAndCancellationJoinOnService) {
  iree_hal_memory_maintenance_t* service = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create(
      {}, iree_allocator_system(), &service));
  iree_async_semaphore_wait_context_t context;
  iree_async_semaphore_wait_context_initialize(&context);
  for (auto expected : {IREE_STATUS_ABORTED, IREE_STATUS_CANCELLED}) {
    auto* semaphore = Semaphore();
    auto* submission = Submission(semaphore);
    uint64_t value = 1;
    Completion completion;
    iree_hal_amd_gpu_host_wait_t wait;
    IREE_ASSERT_OK(iree_hal_amd_gpu_host_wait_begin(
        &context, service, {1, &submission, &value}, Completion::Run,
        &completion, iree_allocator_system(), &wait));
    if (expected == IREE_STATUS_ABORTED) {
      iree_hal_semaphore_fail(semaphore,
                              iree_make_status(expected, "native producer"));
    } else {
      iree_hal_amd_gpu_host_wait_cancel(&wait);
    }
    completion.done.acquire();
    EXPECT_EQ(iree_status_code(completion.status), expected);
    EXPECT_EQ(completion.calls.load(), 1u);
    EXPECT_EQ(
        reinterpret_cast<iree_async_semaphore_t*>(submission)->timepoints_head,
        nullptr);
    if (expected == IREE_STATUS_CANCELLED) {
      auto* signal = AcquireSignal();
      Publish(semaphore, value, signal);
      EXPECT_EQ(Value(submission), value);
      EXPECT_EQ(Value(semaphore), 0u);
      iree_hal_amd_gpu_pm4_signal_release(signal);
    }
  }
  iree_hal_memory_maintenance_release(service);
  iree_async_semaphore_wait_context_deinitialize(&context);
}

TEST_F(SemaphoreTest, ConcurrentColdInstallationKeepsOneChild) {
  Allocator allocator;
  for (uint64_t iteration = 0; iteration < 64; ++iteration) {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_semaphore_create(
        proactor_, iteration, 4, allocator.get(), &semaphore));
    std::array<iree_hal_semaphore_t*, 4> children = {};
    std::array<std::thread, 4> threads;
    std::barrier<> start(4);
    for (size_t i = 0; i < threads.size(); ++i) {
      threads[i] = std::thread([&, i] {
        start.arrive_and_wait();
        IREE_CHECK_OK(
            iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(
                semaphore, &children[i]));
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    EXPECT_EQ(allocator.live.load(), 2);
    for (auto* child : children) {
      EXPECT_EQ(child, children[0]);
      EXPECT_EQ(Value(child), iteration);
      iree_hal_semaphore_release(child);
    }
    iree_hal_semaphore_release(semaphore);
    EXPECT_EQ(allocator.live.load(), 0);
  }
}

TEST_F(SemaphoreTest, PublicationCompletionAndReaderAcquisitionRace) {
  auto* semaphore = Semaphore();
  std::barrier<> start(3);
  std::barrier<> finish(3);
  Signal* signal = nullptr;
  std::thread publisher([&] {
    for (uint64_t value = 1; value <= 2048; ++value) {
      start.arrive_and_wait();
      Publish(semaphore, value, signal);
      finish.arrive_and_wait();
    }
  });
  std::thread observer([&] {
    for (uint64_t value = 1; value <= 2048; ++value) {
      start.arrive_and_wait();
      IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_publish_completed(
          semaphore, value, nullptr));
      finish.arrive_and_wait();
    }
  });
  for (uint64_t value = 1; value <= 2048; ++value) {
    signal = AcquireSignal();
    start.arrive_and_wait();
    auto wait = AcquireWait(semaphore, value);
    if (wait.signal) {
      EXPECT_EQ(wait.signal, signal);
      EXPECT_EQ(wait.producer.epoch, value);
    }
    finish.arrive_and_wait();
    iree_hal_amd_gpu_pm4_signal_release(signal);
    iree_hal_amd_gpu_pm4_signal_release(wait.signal);
    EXPECT_EQ(pool_.live_count, 0u);
  }
  publisher.join();
  observer.join();
}

TEST_F(SemaphoreTest, ConcurrentFirstWaitAndProducerCannotLoseAvailability) {
  std::barrier<> start(2);
  std::barrier<> finish(2);
  iree_hal_semaphore_t* semaphore = nullptr;
  Signal* signal = nullptr;
  std::thread producer([&] {
    for (uint64_t iteration = 0; iteration < 512; ++iteration) {
      start.arrive_and_wait();
      Publish(semaphore, 1, signal);
      finish.arrive_and_wait();
    }
  });
  for (uint64_t iteration = 0; iteration < 512; ++iteration) {
    semaphore = Semaphore();
    signal = AcquireSignal();
    start.arrive_and_wait();
    auto* submission = Submission(semaphore);
    finish.arrive_and_wait();
    EXPECT_EQ(Value(submission), 1u);
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_semaphore_publish_completed(semaphore, 1,
                                                                   nullptr));
    iree_hal_amd_gpu_pm4_signal_release(signal);
  }
  producer.join();
}

}  // namespace
