// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async semaphore proactor operations.
//
// Tests SEMAPHORE_WAIT and SEMAPHORE_SIGNAL operations submitted via proactor.
// These tests verify that semaphore operations integrate correctly with the
// proactor's async execution model, including linked operation chains.
//
// For direct semaphore API tests (timepoints, frontier, etc.), see
// semaphore_sync_test.cc.

#include <atomic>
#include <thread>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/semaphore.h"

namespace iree::async::cts {

class SemaphoreAsyncTest : public CtsTestBase<> {
 protected:
  // Initializes a SEMAPHORE_SIGNAL operation for a single semaphore.
  static void InitSignalOp(iree_async_semaphore_signal_operation_t* operation,
                           iree_async_semaphore_t* semaphore, uint64_t value,
                           const iree_async_frontier_t* frontier,
                           iree_async_completion_fn_t callback,
                           void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->semaphores = &semaphore;
    operation->values = &value;
    operation->count = 1;
    operation->frontier = frontier;
  }

  // Initializes a SEMAPHORE_WAIT operation for a single semaphore.
  static void InitWaitOp(iree_async_semaphore_wait_operation_t* operation,
                         iree_async_semaphore_t** semaphore_storage,
                         uint64_t* value_storage, iree_async_wait_mode_t mode,
                         iree_async_completion_fn_t callback, void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->semaphores = semaphore_storage;
    operation->values = value_storage;
    operation->count = 1;
    operation->mode = mode;
    operation->satisfied_index = 0;
  }
};

struct FreeingWaitCompletion {
  std::atomic<int> call_count{0};
  std::atomic<int> status_code{IREE_STATUS_UNKNOWN};

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    (void)flags;
    auto* completion = static_cast<FreeingWaitCompletion*>(user_data);
    completion->status_code.store(iree_status_code(status),
                                  std::memory_order_relaxed);
    iree_status_free(status);
    completion->call_count.fetch_add(1, std::memory_order_release);
    iree_allocator_free(iree_allocator_system(), operation);
  }
};

//===----------------------------------------------------------------------===//
// SEMAPHORE_SIGNAL tests
//===----------------------------------------------------------------------===//

// Submit signal operation, completes OK, semaphore advances.
TEST_P(SemaphoreAsyncTest, SignalSingle) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Store value in stable storage (operation points to it).
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t target_value = 10;

  CompletionTracker tracker;
  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = CompletionTracker::Callback;
  signal_op.base.user_data = &tracker;
  signal_op.semaphores = &semaphore_ptr;
  signal_op.values = &target_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &signal_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Verify semaphore was signaled.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 10u);

  iree_async_semaphore_release(semaphore);
}

// Signal operation with multiple semaphores, all advance.
TEST_P(SemaphoreAsyncTest, SignalMultiple) {
  iree_async_semaphore_t* sem1 = nullptr;
  iree_async_semaphore_t* sem2 = nullptr;
  iree_async_semaphore_t* sem3 = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem1));
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem2));
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem3));

  iree_async_semaphore_t* semaphores[] = {sem1, sem2, sem3};
  uint64_t values[] = {10, 20, 30};

  CompletionTracker tracker;
  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = CompletionTracker::Callback;
  signal_op.base.user_data = &tracker;
  signal_op.semaphores = semaphores;
  signal_op.values = values;
  signal_op.count = 3;
  signal_op.frontier = NULL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &signal_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // All semaphores should be signaled.
  EXPECT_EQ(iree_async_semaphore_query(sem1), 10u);
  EXPECT_EQ(iree_async_semaphore_query(sem2), 20u);
  EXPECT_EQ(iree_async_semaphore_query(sem3), 30u);

  iree_async_semaphore_release(sem1);
  iree_async_semaphore_release(sem2);
  iree_async_semaphore_release(sem3);
}

// Signal with non-monotonic value fails with error status.
TEST_P(SemaphoreAsyncTest, SignalNonMonotonicFails) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Signal to same value should fail.
  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t target_value = 10;

  CompletionTracker tracker;
  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = CompletionTracker::Callback;
  signal_op.base.user_data = &tracker;
  signal_op.semaphores = &semaphore_ptr;
  signal_op.values = &target_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &signal_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Signal with frontier merges it into the semaphore.
TEST_P(SemaphoreAsyncTest, SignalWithFrontier) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Create frontier (must be sorted by axis).
  uint8_t frontier_storage[sizeof(iree_async_frontier_t) +
                           2 * sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* frontier = (iree_async_frontier_t*)frontier_storage;
  iree_async_frontier_initialize(frontier, 2);
  frontier->entries[0].axis = 0x1234;
  frontier->entries[0].epoch = 100;
  frontier->entries[1].axis = 0x5678;
  frontier->entries[1].epoch = 200;

  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t target_value = 1;

  CompletionTracker tracker;
  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = CompletionTracker::Callback;
  signal_op.base.user_data = &tracker;
  signal_op.semaphores = &semaphore_ptr;
  signal_op.values = &target_value;
  signal_op.count = 1;
  signal_op.frontier = frontier;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &signal_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Verify frontier was merged.
  uint8_t result_storage[sizeof(iree_async_frontier_t) +
                         4 * sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* result = (iree_async_frontier_t*)result_storage;
  uint8_t count =
      iree_async_semaphore_query_frontier(semaphore, result, /*capacity=*/4);

  EXPECT_EQ(count, 2u);

  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// SEMAPHORE_WAIT tests
//===----------------------------------------------------------------------===//

// Wait on already-satisfied value completes immediately.
TEST_P(SemaphoreAsyncTest, WaitSingleAlreadySatisfied) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/10, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t target_value = 5;

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.semaphores = &semaphore_ptr;
  wait_op.values = &target_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Wait pending, signal from thread, wait completes.
TEST_P(SemaphoreAsyncTest, WaitSinglePending) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t target_value = 10;

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.semaphores = &semaphore_ptr;
  wait_op.values = &target_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  iree_async_proactor_wake(proactor_);
  PollOneProgressEvent();
  EXPECT_EQ(tracker.call_count, 0);

  // Signal from another thread.
  std::thread signaler([semaphore]() {
    iree_status_t status = iree_async_semaphore_signal(semaphore, 10, NULL);
    IREE_CHECK_OK(status);
  });

  PollUntil(/*min_completions=*/1);

  signaler.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Wait ALL mode on 3 semaphores, signal 2 - still pending.
TEST_P(SemaphoreAsyncTest, WaitAllModePartial) {
  iree_async_semaphore_t* sem1 = nullptr;
  iree_async_semaphore_t* sem2 = nullptr;
  iree_async_semaphore_t* sem3 = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem1));
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem2));
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem3));

  iree_async_semaphore_t* semaphores[] = {sem1, sem2, sem3};
  uint64_t values[] = {1, 1, 1};

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.semaphores = semaphores;
  wait_op.values = values;
  wait_op.count = 3;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Signal only 2 of 3.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem1, 1, NULL));
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem2, 1, NULL));

  // Force one proactor progress pass after the partial signal set. The wait
  // must remain pending until every timeline value has been reached.
  iree_async_proactor_wake(proactor_);
  PollOneProgressEvent();
  EXPECT_EQ(tracker.call_count, 0);

  // Signal the third - should complete.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem3, 1, NULL));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_semaphore_release(sem1);
  iree_async_semaphore_release(sem2);
  iree_async_semaphore_release(sem3);
}

// Wait ALL mode, signal all, completes.
TEST_P(SemaphoreAsyncTest, WaitAllModeComplete) {
  iree_async_semaphore_t* sem1 = nullptr;
  iree_async_semaphore_t* sem2 = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem1));
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem2));

  iree_async_semaphore_t* semaphores[] = {sem1, sem2};
  uint64_t values[] = {5, 10};

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.semaphores = semaphores;
  wait_op.values = values;
  wait_op.count = 2;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Signal both.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem1, 5, NULL));
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem2, 10, NULL));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_semaphore_release(sem1);
  iree_async_semaphore_release(sem2);
}

// Wait ANY mode, signal one, completes with satisfied_index.
TEST_P(SemaphoreAsyncTest, WaitAnyMode) {
  iree_async_semaphore_t* sem1 = nullptr;
  iree_async_semaphore_t* sem2 = nullptr;
  iree_async_semaphore_t* sem3 = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem1));
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem2));
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &sem3));

  iree_async_semaphore_t* semaphores[] = {sem1, sem2, sem3};
  uint64_t values[] = {1, 1, 1};

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.semaphores = semaphores;
  wait_op.values = values;
  wait_op.count = 3;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ANY;
  wait_op.satisfied_index = 0;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Signal the middle one.
  IREE_ASSERT_OK(iree_async_semaphore_signal(sem2, 1, NULL));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // satisfied_index should indicate which semaphore was satisfied.
  EXPECT_EQ(wait_op.satisfied_index, 1u);  // Index of sem2.

  iree_async_semaphore_release(sem1);
  iree_async_semaphore_release(sem2);
  iree_async_semaphore_release(sem3);
}

// Wait on failed semaphore completes with failure status.
TEST_P(SemaphoreAsyncTest, WaitSemaphoreFailed) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t target_value = 10;

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.semaphores = &semaphore_ptr;
  wait_op.values = &target_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  iree_async_proactor_wake(proactor_);
  PollOneProgressEvent();
  EXPECT_EQ(tracker.call_count, 0);

  // Fail the semaphore from another thread.
  std::thread failure([semaphore]() {
    iree_async_semaphore_fail(
        semaphore, iree_make_status(IREE_STATUS_ABORTED, "device lost"));
  });

  PollUntil(/*min_completions=*/1);

  failure.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Every timepoint detached by one failure must finish before the operation's
// terminal callback can release shared tracker and operation storage.
TEST_P(SemaphoreAsyncTest, WaitFailureJoinsDetachedTimepoints) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  constexpr iree_host_size_t kTimepointCount = 64;
  iree_async_semaphore_t* semaphores[kTimepointCount];
  uint64_t values[kTimepointCount];
  for (iree_host_size_t i = 0; i < kTimepointCount; ++i) {
    semaphores[i] = semaphore;
    values[i] = 1;
  }

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  iree_async_operation_zero(&wait_op.base, sizeof(wait_op));
  iree_async_operation_initialize(
      &wait_op.base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, CompletionTracker::Callback, &tracker);
  wait_op.semaphores = semaphores;
  wait_op.values = values;
  wait_op.count = kTimepointCount;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  std::thread failure([semaphore]() {
    iree_async_semaphore_fail(
        semaphore, iree_make_status(IREE_STATUS_ABORTED, "device lost"));
  });
  PollUntil(/*min_completions=*/1);
  failure.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// A terminal callback owns the operation and may release it immediately.
TEST_P(SemaphoreAsyncTest, WaitCompletionMayReleaseOperation) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_host_size_t operation_size = 0;
  IREE_ASSERT_OK(iree_async_semaphore_wait_operation_size(1, &operation_size));
  iree_async_semaphore_wait_operation_t* wait_op = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(), operation_size,
                                       reinterpret_cast<void**>(&wait_op)));
  iree_async_operation_zero(&wait_op->base, operation_size);

  FreeingWaitCompletion completion;
  iree_async_operation_initialize(&wait_op->base,
                                  IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  FreeingWaitCompletion::Callback, &completion);
  iree_async_semaphore_wait_operation_initialize(wait_op, /*count=*/1,
                                                 IREE_ASYNC_WAIT_MODE_ALL);
  wait_op->semaphores[0] = semaphore;
  wait_op->values[0] = 1;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op->base));
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 1, nullptr));
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(completion.call_count.load(std::memory_order_acquire), 1);
  EXPECT_EQ(completion.status_code.load(std::memory_order_relaxed),
            IREE_STATUS_OK);

  iree_async_semaphore_release(semaphore);
}

// A deferred wait owns its semaphore references until poll-time finalization.
TEST_P(SemaphoreAsyncTest, WaitRetainsSemaphoreUntilCompletion) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));
  iree_async_semaphore_retain(semaphore);

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  iree_async_semaphore_t* wait_semaphore = semaphore;
  uint64_t wait_value = 1;
  InitWaitOp(&wait_op, &wait_semaphore, &wait_value, IREE_ASYNC_WAIT_MODE_ALL,
             CompletionTracker::Callback, &tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Leave one caller-owned reference for signaling, then release it before the
  // poll thread accesses the completed wait tracker.
  iree_async_semaphore_release(semaphore);
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 1, nullptr));
  iree_async_semaphore_release(semaphore);

  PollUntil(/*min_completions=*/1);
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Wait on already-failed semaphore completes immediately with failure.
TEST_P(SemaphoreAsyncTest, WaitSemaphoreAlreadyFailed) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Fail the semaphore before submitting wait.
  iree_async_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "pre-failed"));

  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t target_value = 10;

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.semaphores = &semaphore_ptr;
  wait_op.values = &target_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Submit wait, cancel, completes with CANCELLED.
TEST_P(SemaphoreAsyncTest, WaitCancellation) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_async_semaphore_t* semaphore_ptr = semaphore;
  uint64_t target_value = 10;

  CompletionTracker tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &tracker;
  wait_op.semaphores = &semaphore_ptr;
  wait_op.values = &target_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

// Cancellation may race both terminal satisfaction and poll-thread
// finalization. The operation remains live until both racing calls return.
TEST_P(SemaphoreAsyncTest, WaitSignalCancelRace) {
  constexpr int kIterationCount = 100;
  for (int iteration = 0; iteration < kIterationCount; ++iteration) {
    iree_async_semaphore_t* semaphore = nullptr;
    IREE_ASSERT_OK(iree_async_semaphore_create(
        proactor_,
        /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
        iree_allocator_system(), &semaphore));

    CompletionTracker tracker;
    iree_async_semaphore_wait_operation_t wait_op;
    iree_async_semaphore_t* wait_semaphore = semaphore;
    uint64_t wait_value = 1;
    InitWaitOp(&wait_op, &wait_semaphore, &wait_value, IREE_ASYNC_WAIT_MODE_ALL,
               CompletionTracker::Callback, &tracker);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

    std::atomic<int> ready_count{0};
    std::atomic<bool> start{false};
    std::atomic<int> cancel_status_code{IREE_STATUS_UNKNOWN};
    std::atomic<int> signal_status_code{IREE_STATUS_UNKNOWN};
    std::thread cancel_thread([&]() {
      ready_count.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      iree_status_t status =
          iree_async_proactor_cancel(proactor_, &wait_op.base);
      cancel_status_code.store(iree_status_code(status),
                               std::memory_order_release);
      iree_status_free(status);
    });
    std::thread signal_thread([&]() {
      ready_count.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      iree_status_t status =
          iree_async_semaphore_signal(semaphore, /*new_value=*/1, nullptr);
      signal_status_code.store(iree_status_code(status),
                               std::memory_order_release);
      iree_status_free(status);
    });

    while (ready_count.load(std::memory_order_acquire) != 2) {
      std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    PollUntil(/*min_completions=*/1);
    cancel_thread.join();
    signal_thread.join();

    EXPECT_EQ(cancel_status_code.load(std::memory_order_acquire),
              IREE_STATUS_OK)
        << "iteration " << iteration;
    EXPECT_EQ(signal_status_code.load(std::memory_order_acquire),
              IREE_STATUS_OK)
        << "iteration " << iteration;
    EXPECT_EQ(tracker.call_count, 1) << "iteration " << iteration;
    iree_status_t status = tracker.ConsumeStatus();
    if (!iree_status_is_ok(status) && !iree_status_is_cancelled(status)) {
      IREE_EXPECT_OK(status) << "iteration " << iteration;
    } else {
      iree_status_free(status);
    }

    iree_async_semaphore_release(semaphore);
  }
}

//===----------------------------------------------------------------------===//
// Signal wakes pending wait test
//===----------------------------------------------------------------------===//

// Submit wait, then submit signal operation, wait completes.
TEST_P(SemaphoreAsyncTest, SignalWakesPendingWait) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_,
      /*initial_value=*/0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Submit wait first.
  iree_async_semaphore_t* wait_sem_ptr = semaphore;
  uint64_t wait_value = 10;

  CompletionTracker wait_tracker;
  iree_async_semaphore_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
  wait_op.base.completion_fn = CompletionTracker::Callback;
  wait_op.base.user_data = &wait_tracker;
  wait_op.semaphores = &wait_sem_ptr;
  wait_op.values = &wait_value;
  wait_op.count = 1;
  wait_op.mode = IREE_ASYNC_WAIT_MODE_ALL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Then submit signal.
  iree_async_semaphore_t* signal_sem_ptr = semaphore;
  uint64_t signal_value = 10;

  CompletionTracker signal_tracker;
  iree_async_semaphore_signal_operation_t signal_op;
  memset(&signal_op, 0, sizeof(signal_op));
  signal_op.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_op.base.completion_fn = CompletionTracker::Callback;
  signal_op.base.user_data = &signal_tracker;
  signal_op.semaphores = &signal_sem_ptr;
  signal_op.values = &signal_value;
  signal_op.count = 1;
  signal_op.frontier = NULL;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &signal_op.base));

  // Both should complete.
  PollUntil(/*min_completions=*/2);

  EXPECT_EQ(wait_tracker.call_count, 1);
  EXPECT_EQ(signal_tracker.call_count, 1);
  IREE_EXPECT_OK(wait_tracker.ConsumeStatus());
  IREE_EXPECT_OK(signal_tracker.ConsumeStatus());

  iree_async_semaphore_release(semaphore);
}

CTS_REGISTER_TEST_SUITE(SemaphoreAsyncTest);

}  // namespace iree::async::cts
