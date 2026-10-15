// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/host_wait.h"

#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using Wait = iree_hal_amd_gpu_host_wait_t;
using Entry = iree_hal_memory_maintenance_entry_t;

struct Completion {
  // Joins the callback's final access to this test-owned result.
  std::binary_semaphore done{0};
  // Owned terminal status transferred by the actual service callback.
  iree_status_t status = iree_ok_status();
  // Executor identity, published before done.
  std::thread::id thread;
  // Exactly one terminal callback must run for an accepted wait.
  std::atomic<uint32_t> calls{0};

  ~Completion() { iree_status_free(status); }

  static void Run(void* user_data, iree_status_t status) {
    auto& self = *static_cast<Completion*>(user_data);
    self.status = status;
    self.thread = std::this_thread::get_id();
    self.calls.fetch_add(1);
    self.done.release();
  }
};

// Controls an external cold call, without changing the service implementation.
struct Gate : Entry {
  // Signals that the service has entered the controlled call.
  std::binary_semaphore entered{0};
  // Controller-owned permission to return.
  std::binary_semaphore resume{0};

  Gate() {
    fn = [](Entry* entry) {
      auto& self = *static_cast<Gate*>(entry);
      self.entered.release();
      self.resume.acquire();
    };
  }
};

// Observes actual semaphore/tracker destruction through the allocator boundary.
struct Allocator {
  // Controlled allocation failure before the wait accepts ownership.
  bool fail_allocations = false;
  // Number of allocations released by the owner under test.
  std::atomic<uint32_t> frees{0};
  // Final release thread, read only after joining the service.
  std::thread::id thread;

  iree_allocator_t get() { return {this, Control}; }
  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto& self = *static_cast<Allocator*>(user_data);
    if (self.fail_allocations && command != IREE_ALLOCATOR_COMMAND_FREE) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "tracker allocation denied");
    }
    const auto system = iree_allocator_system();
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      self.thread = std::this_thread::get_id();
      self.frees.fetch_add(1);
    }
    return system.ctl(system.self, command, parameters, pointer);
  }
};

// Supplies exactly the queried allocation capacity, with normal allocator
// alignment. Reuse depends on tracker finalization freeing its sole allocation.
class TrackerStorage {
 public:
  explicit TrackerStorage(iree_host_size_t count) {
    IREE_CHECK_OK(iree_async_semaphore_wait_tracker_storage_size(
        count, &storage_.capacity));
    bytes_.resize((storage_.capacity + sizeof(std::max_align_t) - 1) /
                  sizeof(std::max_align_t));
    storage_.buffer = reinterpret_cast<uint8_t*>(bytes_.data());
  }

  iree_allocator_t allocator() {
    return iree_allocator_inline_arena(&storage_);
  }
  iree_host_size_t length() const { return storage_.length; }

 private:
  // Aligned backing, stable through every callback using the tracker.
  std::vector<std::max_align_t> bytes_;
  // Actual allocator passed to the production wait path.
  iree_allocator_inline_storage_t storage_ = {};
};

class HostWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create(
        {}, iree_allocator_system(), &service_));
    iree_async_semaphore_wait_context_initialize(&context_);
    iree_hal_memory_maintenance_call(
        service_,
        [](void* data) {
          *static_cast<std::thread::id*>(data) = std::this_thread::get_id();
        },
        &service_thread_);
  }
  void TearDown() override {
    iree_hal_memory_maintenance_release(service_);
    iree_async_semaphore_wait_context_deinitialize(&context_);
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_async_proactor_release(proactor_);
  }
  iree_hal_semaphore_t* Semaphore(uint64_t value = 0) {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(iree_hal_host_semaphore_create(
        proactor_, value, iree_allocator_system(), &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }
  iree_status_t Begin(iree_hal_semaphore_list_t waits, Completion& completion,
                      Wait& wait,
                      iree_allocator_t allocator = iree_allocator_system()) {
    return iree_hal_amd_gpu_host_wait_begin(&context_, service_, waits,
                                            Completion::Run, &completion,
                                            allocator, &wait);
  }
  void Check(Completion& completion, iree_status_code_t code = IREE_STATUS_OK) {
    completion.done.acquire();
    EXPECT_EQ(iree_status_code(completion.status), code);
    EXPECT_EQ(completion.calls.load(), 1u);
    EXPECT_EQ(completion.thread, service_thread_);
  }
  // Existing service executor, shared by waits and cold work.
  iree_hal_memory_maintenance_t* service_ = nullptr;
  // Platform proactor borrowed by host semaphores; no polling thread is needed.
  iree_async_proactor_t* proactor_ = nullptr;
  // Queue-owned cancellation association, shared by its pending waits.
  iree_async_semaphore_wait_context_t context_;
  // Identity of the actual service worker.
  std::thread::id service_thread_;
  // Test-owned references independent of the wait tracker's references.
  std::vector<iree_hal_semaphore_t*> semaphores_;
};

TEST_F(HostWaitTest, AlreadyReachedWaitRunsOnService) {
  iree_hal_semaphore_t* semaphore = Semaphore(7);
  uint64_t value = 7;
  Completion completion;
  Wait wait;
  IREE_ASSERT_OK(Begin({1, &semaphore, &value}, completion, wait));
  Check(completion);
}

TEST_F(HostWaitTest, InlineTrackerReusesExactStorageAfterFinalization) {
  std::array<iree_hal_semaphore_t*, 4> semaphores = {
      Semaphore(128), Semaphore(128), Semaphore(128), Semaphore(128)};
  for (iree_host_size_t count : {1u, 4u}) {
    TrackerStorage storage(count);
    for (uint64_t value = 1; value <= 128; ++value) {
      std::array<uint64_t, 4> values;
      values.fill(value);
      Completion completion;
      Wait wait;
      IREE_ASSERT_OK(Begin({count, semaphores.data(), values.data()},
                           completion, wait, storage.allocator()));
      Check(completion);
      EXPECT_EQ(storage.length(), 0u);
      EXPECT_EQ(wait.operation.base.next, nullptr);
    }
  }
}

TEST_F(HostWaitTest, TrackerStorageRejectsUnrepresentableCount) {
  for (iree_host_size_t count :
       {iree_host_size_t{INT32_MAX} + 1, IREE_HOST_SIZE_MAX}) {
    iree_host_size_t size = 123;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_async_semaphore_wait_tracker_storage_size(count, &size));
    EXPECT_EQ(size, 0u);
  }
}

TEST_F(HostWaitTest, AllWaitsResolveBeforeContinuation) {
  std::array<iree_hal_semaphore_t*, 3> semaphores = {Semaphore(), Semaphore(),
                                                     Semaphore()};
  std::array<uint64_t, 3> values = {1, 2, 3};
  Completion completion;
  Wait wait;
  IREE_ASSERT_OK(Begin({semaphores.size(), semaphores.data(), values.data()},
                       completion, wait));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[2], 3, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[0], 1, nullptr));
  EXPECT_EQ(completion.calls.load(), 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphores[1], 2, nullptr));
  Check(completion);
}

TEST_F(HostWaitTest, FailureCancelsPeerThatWillNeverSignal) {
  std::array<iree_hal_semaphore_t*, 2> semaphores = {Semaphore(), Semaphore()};
  std::array<uint64_t, 2> values = {1, 1};
  Completion completion;
  Wait wait;
  IREE_ASSERT_OK(Begin({semaphores.size(), semaphores.data(), values.data()},
                       completion, wait));
  iree_hal_semaphore_fail(semaphores[1],
                          iree_make_status(IREE_STATUS_DATA_LOSS, "producer"));
  Check(completion, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(reinterpret_cast<iree_async_semaphore_t*>(semaphores[0])
                ->timepoints.count,
            0u);
}

TEST_F(HostWaitTest, FailureDuringRegistrationJoinsEarlierTimepoint) {
  std::array<iree_hal_semaphore_t*, 3> semaphores = {Semaphore(), Semaphore(),
                                                     Semaphore()};
  std::array<uint64_t, 3> values = {1, 1, 1};
  iree_hal_semaphore_fail(semaphores[1],
                          iree_make_status(IREE_STATUS_ABORTED, "producer"));
  Completion completion;
  Wait wait;
  IREE_ASSERT_OK(Begin({semaphores.size(), semaphores.data(), values.data()},
                       completion, wait));
  Check(completion, IREE_STATUS_ABORTED);
  for (auto* semaphore : semaphores) {
    EXPECT_EQ(
        reinterpret_cast<iree_async_semaphore_t*>(semaphore)->timepoints.count,
        0u);
  }
}

TEST_F(HostWaitTest, IndexGrowthFailureCancelsOnlyRegisteredWaits) {
  Allocator allocator;
  iree_hal_semaphore_t* full_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_host_semaphore_create(proactor_, 0, allocator.get(),
                                                &full_semaphore));
  auto* full_async = reinterpret_cast<iree_async_semaphore_t*>(full_semaphore);
  std::array<iree_async_semaphore_timepoint_t, 4> pending{};
  uint32_t callbacks = 0;
  for (auto& point : pending) {
    point.user_data = &callbacks;
    point.callback = [](void* data, iree_async_semaphore_timepoint_t*,
                        iree_status_t status) {
      IREE_CHECK_OK(status);
      ++*static_cast<uint32_t*>(data);
    };
    IREE_ASSERT_OK(
        iree_async_semaphore_acquire_timepoint(full_async, 10, &point));
  }
  allocator.fail_allocations = true;
  std::array<iree_hal_semaphore_t*, 2> semaphores = {Semaphore(),
                                                     full_semaphore};
  std::array<uint64_t, 2> values = {1, 1};
  Completion completion;
  Wait wait;
  IREE_ASSERT_OK(Begin({semaphores.size(), semaphores.data(), values.data()},
                       completion, wait));
  Check(completion, IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(reinterpret_cast<iree_async_semaphore_t*>(semaphores[0])
                ->timepoints.count,
            0u);
  EXPECT_EQ(full_async->timepoints.count, pending.size());
  EXPECT_EQ(callbacks, 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(full_semaphore, 10, nullptr));
  EXPECT_EQ(callbacks, pending.size());
  iree_hal_semaphore_release(full_semaphore);
}

TEST_F(HostWaitTest, CancellationDoesNotWaitForProducer) {
  iree_hal_semaphore_t* semaphore = Semaphore();
  uint64_t value = 1;
  Completion completion;
  Wait wait;
  IREE_ASSERT_OK(Begin({1, &semaphore, &value}, completion, wait));
  iree_hal_amd_gpu_host_wait_cancel(&wait);
  Check(completion, IREE_STATUS_CANCELLED);
  // The sidecar remains test-owned; cancellation after finalization is benign.
  iree_hal_amd_gpu_host_wait_cancel(&wait);
  EXPECT_EQ(completion.calls.load(), 1u);
}

TEST_F(HostWaitTest, AllocationFailureDoesNotAcceptOrRetain) {
  iree_hal_semaphore_t* semaphore = Semaphore();
  uint64_t value = 1;
  Completion completion;
  Wait wait;
  Allocator allocator;
  allocator.fail_allocations = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amd_gpu_host_wait_begin(&context_, service_,
                                       {1, &semaphore, &value}, Completion::Run,
                                       &completion, allocator.get(), &wait));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore, 1, nullptr));
  EXPECT_EQ(completion.calls.load(), 0u);
  EXPECT_EQ(wait.operation.base.next, nullptr);
}

TEST_F(HostWaitTest, LastSemaphoreAndTrackerReferencesReleaseOnService) {
  Allocator semaphore_allocator;
  Allocator tracker_allocator;
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_host_semaphore_create(
      proactor_, 1, semaphore_allocator.get(), &semaphore));
  Gate gate;
  iree_hal_memory_maintenance_enqueue(service_, &gate);
  gate.entered.acquire();
  uint64_t value = 1;
  Completion completion;
  Wait wait;
  IREE_ASSERT_OK(iree_hal_amd_gpu_host_wait_begin(
      &context_, service_, {1, &semaphore, &value}, Completion::Run,
      &completion, tracker_allocator.get(), &wait));
  iree_hal_semaphore_release(semaphore);
  EXPECT_EQ(semaphore_allocator.frees.load(), 0u);
  EXPECT_EQ(tracker_allocator.frees.load(), 0u);
  gate.resume.release();
  Check(completion);
  EXPECT_EQ(semaphore_allocator.frees.load(), 1u);
  EXPECT_EQ(semaphore_allocator.thread, service_thread_);
  EXPECT_EQ(tracker_allocator.frees.load(), 1u);
  EXPECT_EQ(tracker_allocator.thread, service_thread_);
}

TEST_F(HostWaitTest, TerminalCallbackCanFreeCaptureImmediately) {
  struct Capture {
    // Cold sidecar embedded in the owning capture.
    Wait wait;
    // Captured semaphore pointer array.
    iree_hal_semaphore_t* semaphore;
    // Captured value array.
    uint64_t value;
    // Independent test result, live after capture destruction.
    Completion* completion;
    // Tracker backing destroyed with the enclosing capture in its callback.
    TrackerStorage storage{1};
  };
  auto* semaphore = Semaphore(1024);
  for (uint64_t value = 1; value <= 1024; ++value) {
    Completion completion;
    auto* capture = new Capture{{}, semaphore, value, &completion};
    IREE_ASSERT_OK(iree_hal_amd_gpu_host_wait_begin(
        &context_, service_, {1, &capture->semaphore, &capture->value},
        [](void* user_data, iree_status_t status) {
          auto* capture = static_cast<Capture*>(user_data);
          auto* completion = capture->completion;
          delete capture;
          Completion::Run(completion, status);
        },
        capture, capture->storage.allocator(), &capture->wait));
    Check(completion);
  }
}

TEST_F(HostWaitTest, ConcurrentRegistrationAndSignalsReturnEveryBorrow) {
  std::array<iree_hal_semaphore_t*, 4> semaphores = {Semaphore(), Semaphore(),
                                                     Semaphore(), Semaphore()};
  std::array<uint64_t, 4> values;
  TrackerStorage storage(semaphores.size());
  std::barrier<> start(5);
  std::barrier<> finish(5);
  std::array<std::thread, 4> producers;
  for (size_t i = 0; i < producers.size(); ++i) {
    producers[i] = std::thread([&, i] {
      for (uint64_t value = 1; value <= 512; ++value) {
        start.arrive_and_wait();
        IREE_CHECK_OK(iree_hal_semaphore_signal(semaphores[i], value, nullptr));
        finish.arrive_and_wait();
      }
    });
  }
  for (uint64_t value = 1; value <= 512; ++value) {
    values.fill(value);
    Completion completion;
    Wait wait;
    start.arrive_and_wait();
    IREE_CHECK_OK(Begin({semaphores.size(), semaphores.data(), values.data()},
                        completion, wait, storage.allocator()));
    Check(completion);
    EXPECT_EQ(storage.length(), 0u);
    finish.arrive_and_wait();
  }
  for (auto& producer : producers) {
    producer.join();
  }
}

TEST_F(HostWaitTest, ConcurrentCancellationAndSignalHaveOneTerminalOwner) {
  auto* semaphore = Semaphore();
  TrackerStorage storage(1);
  std::barrier<> start(2);
  std::barrier<> finish(2);
  std::thread producer([&] {
    for (uint64_t value = 1; value <= 1024; ++value) {
      start.arrive_and_wait();
      IREE_CHECK_OK(iree_hal_semaphore_signal(semaphore, value, nullptr));
      finish.arrive_and_wait();
    }
  });
  for (uint64_t value = 1; value <= 1024; ++value) {
    Completion completion;
    Wait wait;
    IREE_CHECK_OK(
        Begin({1, &semaphore, &value}, completion, wait, storage.allocator()));
    start.arrive_and_wait();
    iree_hal_amd_gpu_host_wait_cancel(&wait);
    completion.done.acquire();
    const auto code = iree_status_code(completion.status);
    EXPECT_TRUE(code == IREE_STATUS_OK || code == IREE_STATUS_CANCELLED);
    EXPECT_EQ(completion.calls.load(), 1u);
    EXPECT_EQ(completion.thread, service_thread_);
    EXPECT_EQ(storage.length(), 0u);
    finish.arrive_and_wait();
  }
  producer.join();
}

}  // namespace
