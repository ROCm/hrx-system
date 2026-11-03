// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/semaphore_wait.h"

#include <array>
#include <barrier>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

// A direct terminal consumer, using the same tracker as queued proactors.
// Completion may release or reuse the operation and its tracker allocation.
struct Wait {
  // Borrowed owner of cancellation associations.
  iree_async_semaphore_wait_context_t* context = nullptr;
  // Stable caller-owned input and cancellation identity.
  iree_async_semaphore_wait_operation_t operation = {};
  // Owned until finalization, then immediately reusable by the callback.
  iree_async_semaphore_wait_tracker_t* tracker = nullptr;
  // Allocator used by each registration, including callback-driven reuse.
  iree_allocator_t allocator = iree_allocator_system();
  // Whether producer failure completes immediately or joins all dependencies.
  iree_async_semaphore_wait_flags_t flags = IREE_ASYNC_SEMAPHORE_WAIT_FLAG_NONE;
  // Number of completed registrations, read after producer threads join.
  uint32_t completions = 0;
  // Terminal result of the last registration.
  iree_status_code_t code = IREE_STATUS_OK;
  // Identity of the final borrower performing completion.
  std::thread::id completion_thread;
  // Optional action invoked after the tracker's final access.
  void (*completed)(Wait* wait) = nullptr;
  // Consumer holds finalization after preparing, for the cancellation boundary.
  bool hold_finalization = false;
  // Exclusive finalization ownership, established by a successful prepare.
  bool prepared = false;

  iree_status_t Allocate() {
    return iree_async_semaphore_wait_tracker_allocate(
        context, &operation, {Consume, this}, flags, allocator, &tracker);
  }

  iree_status_t Begin() {
    IREE_RETURN_IF_ERROR(Allocate());
    iree_async_semaphore_wait_tracker_register_timepoints(tracker);
    return iree_ok_status();
  }

  void Finalize() {
    iree_async_semaphore_wait_completion_t completion;
    iree_async_semaphore_wait_tracker_finalize(tracker, &completion);
    tracker = nullptr;
    EXPECT_EQ(completion.operation, &operation);
    EXPECT_EQ(operation.base.next, nullptr);
    code = iree_status_code(completion.status);
    iree_status_free(completion.status);
    ++completions;
    completion_thread = std::this_thread::get_id();
    if (completed) {
      completed(this);
    }
  }

  static void Consume(void* user_data, iree_atomic_slist_entry_t* entry) {
    auto* wait = static_cast<Wait*>(user_data);
    if (!iree_async_semaphore_wait_tracker_try_prepare_completion(
            wait->tracker)) {
      return;
    }
    wait->prepared = true;
    if (!wait->hold_finalization) {
      wait->Finalize();
    }
  }
};

class SemaphoreWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    iree_async_semaphore_wait_context_initialize(&context_);
  }

  void TearDown() override {
    iree_async_semaphore_wait_context_deinitialize(&context_);
    for (auto* semaphore : semaphores_) {
      iree_async_semaphore_release(semaphore);
    }
    iree_async_proactor_release(proactor_);
  }

  iree_async_semaphore_t* Semaphore(uint64_t value = 0) {
    iree_async_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(iree_async_semaphore_create(
        proactor_, value, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
        iree_allocator_system(), &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }

  Wait MakeWait(iree_host_size_t count, iree_async_semaphore_t** semaphores,
                uint64_t* values,
                iree_async_wait_mode_t mode = IREE_ASYNC_WAIT_MODE_ALL) {
    return Wait{
        .context = &context_,
        .operation =
            {
                .semaphores = semaphores,
                .values = values,
                .count = count,
                .mode = mode,
            },
    };
  }

  // Platform service required by real semaphores; no polling thread is started.
  iree_async_proactor_t* proactor_ = nullptr;
  // Shared cancellation association owner, live through every terminal
  // callback.
  iree_async_semaphore_wait_context_t context_;
  // Independent test references, released after all wait borrowers have joined.
  std::vector<iree_async_semaphore_t*> semaphores_;
};

TEST_F(SemaphoreWaitTest, PollEmptyCompletesImmediately) {
  for (const auto mode : {IREE_ASYNC_WAIT_MODE_ALL, IREE_ASYNC_WAIT_MODE_ANY}) {
    auto wait = MakeWait(0, nullptr, nullptr, mode);
    bool ready = false;
    IREE_ASSERT_OK(iree_async_semaphore_wait_poll(&wait.operation, &ready));
    EXPECT_TRUE(ready);
  }
}

TEST_F(SemaphoreWaitTest, PollAllRequiresEveryValue) {
  std::array semaphores = {Semaphore(3), Semaphore(6)};
  std::array<uint64_t, 2> values = {3, 7};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data());
  bool ready = true;
  IREE_ASSERT_OK(iree_async_semaphore_wait_poll(&wait.operation, &ready));
  EXPECT_FALSE(ready);
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores[1], 7, nullptr));
  IREE_ASSERT_OK(iree_async_semaphore_wait_poll(&wait.operation, &ready));
  EXPECT_TRUE(ready);
}

TEST_F(SemaphoreWaitTest, PollAnyReportsFirstSatisfiedIndex) {
  std::array semaphores = {Semaphore(2), Semaphore(6), Semaphore(8)};
  std::array<uint64_t, 3> values = {3, 7, 9};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data(),
                       IREE_ASYNC_WAIT_MODE_ANY);
  bool ready = true;
  IREE_ASSERT_OK(iree_async_semaphore_wait_poll(&wait.operation, &ready));
  EXPECT_FALSE(ready);
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores[1], 7, nullptr));
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores[2], 9, nullptr));
  IREE_ASSERT_OK(iree_async_semaphore_wait_poll(&wait.operation, &ready));
  EXPECT_TRUE(ready);
  EXPECT_EQ(wait.operation.satisfied_index, 1u);
}

TEST_F(SemaphoreWaitTest, PollPreservesFailureAfterPendingOrSatisfiedValue) {
  for (const auto mode : {IREE_ASYNC_WAIT_MODE_ALL, IREE_ASYNC_WAIT_MODE_ANY}) {
    for (uint64_t first_value : {0u, 3u}) {
      for (uint64_t failed_value : {0u, 7u}) {
        SCOPED_TRACE(mode);
        SCOPED_TRACE(first_value);
        SCOPED_TRACE(failed_value);
        std::array semaphores = {Semaphore(first_value),
                                 Semaphore(failed_value)};
        std::array<uint64_t, 2> values = {3, 7};
        auto wait =
            MakeWait(values.size(), semaphores.data(), values.data(), mode);
        iree_async_semaphore_fail(
            semaphores[1], iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "producer allocation failed"));
        bool ready = true;
        Status status(iree_async_semaphore_wait_poll(&wait.operation, &ready));
        EXPECT_FALSE(ready);
        EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
        EXPECT_NE(status.ToString().find("producer allocation failed"),
                  std::string::npos);
      }
    }
  }
}

TEST_F(SemaphoreWaitTest, PollNativeFailureEncodingIsNotSatisfaction) {
  auto* semaphore = Semaphore();
  uint64_t value = 10;
  auto wait = MakeWait(1, &semaphore, &value);
  iree_async_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "native failure"));

  // HAL query overrides may encode failure above every usable timeline value.
  // Keep the real semaphore's failure publication and ownership unchanged.
  const auto* original_vtable = semaphore->vtable;
  auto native_vtable = *original_vtable;
  native_vtable.query = [](iree_async_semaphore_t* semaphore) -> uint64_t {
    return UINT64_MAX;
  };
  semaphore->vtable = &native_vtable;
  bool ready = true;
  Status status(iree_async_semaphore_wait_poll(&wait.operation, &ready));
  semaphore->vtable = original_vtable;
  EXPECT_FALSE(ready);
  EXPECT_EQ(status.code(), StatusCode::kAborted);
  EXPECT_NE(status.ToString().find("native failure"), std::string::npos);
}

TEST_F(SemaphoreWaitTest, AlreadyReachedCompletesOnRegisteringThread) {
  std::array semaphores = {Semaphore(3), Semaphore(7)};
  std::array<uint64_t, 2> values = {3, 7};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data());
  IREE_ASSERT_OK(wait.Begin());
  EXPECT_EQ(wait.completions, 1u);
  EXPECT_EQ(wait.code, IREE_STATUS_OK);
  EXPECT_EQ(wait.completion_thread, std::this_thread::get_id());
}

TEST_F(SemaphoreWaitTest, PendingAllCompletesOnLastSignalingThread) {
  std::array semaphores = {Semaphore(), Semaphore()};
  std::array<uint64_t, 2> values = {3, 7};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data());
  IREE_ASSERT_OK(wait.Begin());
  IREE_ASSERT_OK(
      iree_async_semaphore_signal(semaphores[0], values[0], nullptr));
  EXPECT_EQ(wait.completions, 0u);
  std::thread producer([&] {
    IREE_EXPECT_OK(
        iree_async_semaphore_signal(semaphores[1], values[1], nullptr));
    EXPECT_EQ(wait.completions, 1u);
    EXPECT_EQ(wait.completion_thread, std::this_thread::get_id());
  });
  producer.join();
  EXPECT_EQ(wait.code, IREE_STATUS_OK);
}

TEST_F(SemaphoreWaitTest, AnyCancelsRemainingRegistrations) {
  std::array semaphores = {Semaphore(), Semaphore(), Semaphore()};
  std::array<uint64_t, 3> values = {2, 4, 6};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data(),
                       IREE_ASYNC_WAIT_MODE_ANY);
  IREE_ASSERT_OK(wait.Begin());
  IREE_ASSERT_OK(
      iree_async_semaphore_signal(semaphores[1], values[1], nullptr));
  EXPECT_EQ(wait.completions, 1u);
  EXPECT_EQ(wait.code, IREE_STATUS_OK);
  EXPECT_EQ(wait.operation.satisfied_index, 1u);
  IREE_ASSERT_OK(
      iree_async_semaphore_signal(semaphores[0], values[0], nullptr));
  IREE_ASSERT_OK(
      iree_async_semaphore_signal(semaphores[2], values[2], nullptr));
  EXPECT_EQ(wait.completions, 1u);
}

TEST_F(SemaphoreWaitTest, FailureCancelsRemainingRegistrations) {
  std::array semaphores = {Semaphore(), Semaphore()};
  std::array<uint64_t, 2> values = {2, 4};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data());
  IREE_ASSERT_OK(wait.Begin());
  iree_async_semaphore_fail(
      semaphores[0], iree_make_status(IREE_STATUS_ABORTED, "producer failed"));
  EXPECT_EQ(wait.completions, 1u);
  EXPECT_EQ(wait.code, IREE_STATUS_ABORTED);
  IREE_ASSERT_OK(
      iree_async_semaphore_signal(semaphores[1], values[1], nullptr));
  EXPECT_EQ(wait.completions, 1u);
}

TEST_F(SemaphoreWaitTest, DependencyJoinPreservesEveryInputAfterFailure) {
  for (iree_host_size_t failed_index : {0u, 1u, 2u}) {
    for (bool already_failed : {false, true}) {
      SCOPED_TRACE(failed_index);
      SCOPED_TRACE(already_failed);
      std::array semaphores = {Semaphore(), Semaphore(), Semaphore()};
      std::array<uint64_t, 3> values = {2, 4, 6};
      auto wait = MakeWait(values.size(), semaphores.data(), values.data());
      wait.flags = IREE_ASYNC_SEMAPHORE_WAIT_FLAG_JOIN;
      if (already_failed) {
        iree_async_semaphore_fail(
            semaphores[failed_index],
            iree_make_status(IREE_STATUS_DATA_LOSS, "producer failed"));
      }
      IREE_ASSERT_OK(wait.Begin());
      if (!already_failed) {
        iree_async_semaphore_fail(
            semaphores[failed_index],
            iree_make_status(IREE_STATUS_DATA_LOSS, "producer failed"));
      }
      EXPECT_EQ(wait.completions, 0u);
      const auto second = (failed_index + 1) % values.size();
      const auto last = (failed_index + 2) % values.size();
      IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores[second],
                                                 values[second], nullptr));
      EXPECT_EQ(wait.completions, 0u);
      IREE_ASSERT_OK(
          iree_async_semaphore_signal(semaphores[last], values[last], nullptr));
      EXPECT_EQ(wait.completions, 1u);
      EXPECT_EQ(wait.code, IREE_STATUS_DATA_LOSS);
    }
  }
}

TEST_F(SemaphoreWaitTest, DependencyJoinRetainsFirstFailure) {
  std::array semaphores = {Semaphore(), Semaphore()};
  std::array<uint64_t, 2> values = {1, 1};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data());
  wait.flags = IREE_ASYNC_SEMAPHORE_WAIT_FLAG_JOIN;
  IREE_ASSERT_OK(wait.Begin());
  iree_async_semaphore_fail(
      semaphores[0], iree_make_status(IREE_STATUS_DATA_LOSS, "first producer"));
  EXPECT_EQ(wait.completions, 0u);
  iree_async_semaphore_fail(
      semaphores[1], iree_make_status(IREE_STATUS_ABORTED, "second producer"));
  EXPECT_EQ(wait.completions, 1u);
  EXPECT_EQ(wait.code, IREE_STATUS_DATA_LOSS);
}

TEST_F(SemaphoreWaitTest, DependencyJoinCanCancelAfterProducerFailure) {
  std::array semaphores = {Semaphore(), Semaphore()};
  std::array<uint64_t, 2> values = {1, 1};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data());
  wait.flags = IREE_ASYNC_SEMAPHORE_WAIT_FLAG_JOIN;
  IREE_ASSERT_OK(wait.Begin());
  iree_async_semaphore_fail(
      semaphores[0],
      iree_make_status(IREE_STATUS_DATA_LOSS, "producer failed"));
  EXPECT_EQ(wait.completions, 0u);
  iree_async_semaphore_wait_context_request_cancellation(&context_,
                                                         &wait.operation);
  EXPECT_EQ(wait.completions, 1u);
  EXPECT_EQ(wait.code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(semaphores[1]->timepoints.count, 0u);
}

TEST_F(SemaphoreWaitTest, ConcurrentDependencyFailureAndCancellation) {
  for (int iteration = 0; iteration < 64; ++iteration) {
    std::array semaphores = {Semaphore(), Semaphore(), Semaphore()};
    std::array<uint64_t, 3> values = {1, 1, 1};
    auto wait = MakeWait(values.size(), semaphores.data(), values.data());
    wait.flags = IREE_ASYNC_SEMAPHORE_WAIT_FLAG_JOIN;
    IREE_ASSERT_OK(wait.Begin());
    std::barrier<> start(3);
    std::thread failed([&] {
      start.arrive_and_wait();
      iree_async_semaphore_fail(
          semaphores[0], iree_make_status(IREE_STATUS_DATA_LOSS, "producer"));
      IREE_EXPECT_OK(iree_async_semaphore_signal(semaphores[1], 1, nullptr));
    });
    std::thread completed([&] {
      start.arrive_and_wait();
      IREE_EXPECT_OK(iree_async_semaphore_signal(semaphores[2], 1, nullptr));
    });
    start.arrive_and_wait();
    iree_async_semaphore_wait_context_request_cancellation(&context_,
                                                           &wait.operation);
    failed.join();
    completed.join();
    EXPECT_EQ(wait.completions, 1u);
    EXPECT_TRUE(wait.code == IREE_STATUS_DATA_LOSS ||
                wait.code == IREE_STATUS_CANCELLED);
    EXPECT_EQ(wait.tracker, nullptr);
  }
}

TEST_F(SemaphoreWaitTest, CancellationCompletesOnCancelingThread) {
  std::array semaphores = {Semaphore(), Semaphore()};
  std::array<uint64_t, 2> values = {2, 4};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data());
  IREE_ASSERT_OK(wait.Begin());
  iree_async_semaphore_wait_context_request_cancellation(&context_,
                                                         &wait.operation);
  EXPECT_EQ(wait.completions, 1u);
  EXPECT_EQ(wait.code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(wait.completion_thread, std::this_thread::get_id());
  // The detached operation has no second cancellation delivery.
  iree_async_semaphore_wait_context_request_cancellation(&context_,
                                                         &wait.operation);
  EXPECT_EQ(wait.completions, 1u);
}

TEST_F(SemaphoreWaitTest, CancellationBeforeRegistrationJoinsRegistrar) {
  auto* semaphore = Semaphore();
  uint64_t value = 1;
  auto wait = MakeWait(1, &semaphore, &value);
  IREE_ASSERT_OK(wait.Allocate());
  iree_async_semaphore_wait_context_request_cancellation(&context_,
                                                         &wait.operation);
  EXPECT_EQ(wait.completions, 0u);
  iree_async_semaphore_wait_tracker_register_timepoints(wait.tracker);
  EXPECT_EQ(wait.completions, 1u);
  EXPECT_EQ(wait.code, IREE_STATUS_CANCELLED);
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, value, nullptr));
  EXPECT_EQ(wait.completions, 1u);
}

TEST_F(SemaphoreWaitTest, PreparedCompletionExcludesLaterCancellation) {
  auto* semaphore = Semaphore();
  uint64_t value = 1;
  auto wait = MakeWait(1, &semaphore, &value);
  wait.hold_finalization = true;
  IREE_ASSERT_OK(wait.Begin());
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, value, nullptr));
  ASSERT_TRUE(wait.prepared);
  EXPECT_EQ(wait.operation.base.next, nullptr);
  iree_async_semaphore_wait_context_request_cancellation(&context_,
                                                         &wait.operation);
  EXPECT_EQ(wait.completions, 0u);
  wait.Finalize();
  EXPECT_EQ(wait.completions, 1u);
  EXPECT_EQ(wait.code, IREE_STATUS_OK);
}

TEST_F(SemaphoreWaitTest, CompletionReusesOperationAndTrackerStorage) {
  auto* semaphore = Semaphore(32);
  uint64_t value = 1;
  auto wait = MakeWait(1, &semaphore, &value);
  iree_host_size_t size = 0;
  IREE_ASSERT_OK(iree_async_semaphore_wait_tracker_storage_size(1, &size));
  std::vector<std::max_align_t> bytes((size + sizeof(std::max_align_t) - 1) /
                                      sizeof(std::max_align_t));
  iree_allocator_inline_storage_t storage = {
      .capacity = size,
      .buffer = reinterpret_cast<uint8_t*>(bytes.data()),
  };
  wait.allocator = iree_allocator_inline_arena(&storage);
  wait.completed = [](Wait* wait) {
    if (++wait->operation.values[0] <= 32) {
      IREE_EXPECT_OK(wait->Begin());
    }
  };
  IREE_ASSERT_OK(wait.Begin());
  EXPECT_EQ(wait.completions, 32u);
  EXPECT_EQ(storage.length, 0u);
}

TEST_F(SemaphoreWaitTest, CompletionCanFreeEnclosingOperation) {
  auto* semaphore = Semaphore();
  uint64_t value = 1;
  auto* wait = new Wait(MakeWait(1, &semaphore, &value));
  wait->completed = [](Wait* wait) { delete wait; };
  IREE_ASSERT_OK(wait->Begin());
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, value, nullptr));
}

TEST_F(SemaphoreWaitTest, ConcurrentSignalAndCancellationJoinAllBorrowers) {
  std::array semaphores = {Semaphore(), Semaphore()};
  std::barrier<> start(3);
  std::array<uint64_t, 2> values = {};
  auto wait = MakeWait(values.size(), semaphores.data(), values.data());
  constexpr uint64_t kIterations = 256;
  std::thread producer([&] {
    for (uint64_t i = 1; i <= kIterations; ++i) {
      start.arrive_and_wait();
      IREE_EXPECT_OK(iree_async_semaphore_signal(semaphores[0], i, nullptr));
      IREE_EXPECT_OK(iree_async_semaphore_signal(semaphores[1], i, nullptr));
      start.arrive_and_wait();
    }
  });
  std::thread canceler([&] {
    for (uint64_t i = 1; i <= kIterations; ++i) {
      start.arrive_and_wait();
      iree_async_semaphore_wait_context_request_cancellation(&context_,
                                                             &wait.operation);
      start.arrive_and_wait();
    }
  });
  for (uint64_t i = 1; i <= kIterations; ++i) {
    values = {i, i};
    IREE_CHECK_OK(wait.Begin());
    start.arrive_and_wait();
    start.arrive_and_wait();
    EXPECT_EQ(wait.completions, i);
    EXPECT_TRUE(wait.code == IREE_STATUS_OK ||
                wait.code == IREE_STATUS_CANCELLED);
    EXPECT_EQ(wait.tracker, nullptr);
  }
  producer.join();
  canceler.join();
}

}  // namespace
}  // namespace iree
