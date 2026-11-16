// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/completion_queue.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct TestState {
  iree_atomic_int32_t action_count;
  iree_atomic_int32_t cleanup_count;
  int32_t cleanup_order[2];
} TestState;

typedef struct OrderedCleanup {
  TestState* state;
  int32_t value;
} OrderedCleanup;

typedef struct BlockingActionState {
  std::mutex mutex;
  std::condition_variable cv;
  bool started = false;
  bool release = false;
  std::atomic<int32_t> action_count{0};
  std::atomic<int32_t> cleanup_count{0};
} BlockingActionState;

static iree_status_t IncrementAction(void* user_data) {
  TestState* state = reinterpret_cast<TestState*>(user_data);
  iree_atomic_fetch_add(&state->action_count, 1, iree_memory_order_acq_rel);
  return iree_ok_status();
}

static void IncrementCleanup(void* user_data) {
  TestState* state = reinterpret_cast<TestState*>(user_data);
  iree_atomic_fetch_add(&state->cleanup_count, 1, iree_memory_order_acq_rel);
}

static void RecordOrderedCleanup(void* user_data) {
  OrderedCleanup* cleanup = reinterpret_cast<OrderedCleanup*>(user_data);
  int32_t index = iree_atomic_fetch_add(&cleanup->state->cleanup_count, 1,
                                        iree_memory_order_acq_rel);
  cleanup->state->cleanup_order[index] = cleanup->value;
}

static iree_status_t BlockingAction(void* user_data) {
  BlockingActionState* state =
      reinterpret_cast<BlockingActionState*>(user_data);
  state->action_count.fetch_add(1, std::memory_order_acq_rel);
  std::unique_lock<std::mutex> lock(state->mutex);
  state->started = true;
  state->cv.notify_all();
  state->cv.wait(lock, [&] { return state->release; });
  return iree_ok_status();
}

static void BlockingCleanup(void* user_data) {
  BlockingActionState* state =
      reinterpret_cast<BlockingActionState*>(user_data);
  state->cleanup_count.fetch_add(1, std::memory_order_acq_rel);
}

static bool WaitForBlockingActionStarted(BlockingActionState* state) {
  std::unique_lock<std::mutex> lock(state->mutex);
  return state->cv.wait_for(lock, std::chrono::seconds(5),
                            [&] { return state->started; });
}

static void ReleaseBlockingAction(BlockingActionState* state) {
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->release = true;
  }
  state->cv.notify_all();
}

static bool WaitForStartedCount(const std::atomic<int32_t>& count,
                                int32_t expected) {
  for (int i = 0; i < 5000; ++i) {
    if (count.load(std::memory_order_acquire) >= expected) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return count.load(std::memory_order_acquire) >= expected;
}

static iree_async_proactor_t* CreateProactor() {
  iree_async_proactor_t* proactor = nullptr;
  IREE_CHECK_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  return proactor;
}

static iree_hal_semaphore_t* CreateSemaphore(iree_async_proactor_t* proactor,
                                             uint64_t initial_value) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_semaphore_create(
      proactor, IREE_HAL_QUEUE_AFFINITY_ANY, initial_value,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, iree_allocator_system(), &semaphore));
  return semaphore;
}

static iree_hal_semaphore_list_t MakeSemaphoreList(
    iree_hal_semaphore_t** semaphores, uint64_t* values,
    iree_host_size_t count) {
  iree_hal_semaphore_list_t list = iree_hal_semaphore_list_empty();
  list.count = count;
  list.semaphores = semaphores;
  list.payload_values = values;
  return list;
}

static bool SubmitBlockingBatch(iree_hal_amdxdna_completion_queue_t* queue,
                                BlockingActionState* state) {
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, iree_hal_semaphore_list_empty(), &batch));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_action(
      batch, BlockingAction, BlockingCleanup, state,
      /*run_on_error=*/false));
  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  bool deferred = iree_status_is_deferred(status);
  iree_status_ignore(status);
  return deferred;
}

TEST(CompletionQueueTest, RunsActionAndCleanupBeforeDestroyReturns) {
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  TestState state = {};
  iree_atomic_store(&state.action_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&state.cleanup_count, 0, iree_memory_order_relaxed);
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, iree_hal_semaphore_list_empty(), &batch));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_action(
      batch, IncrementAction, IncrementCleanup, &state,
      /*run_on_error=*/false));

  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  EXPECT_EQ(iree_atomic_load(&state.action_count, iree_memory_order_acquire),
            1);
  EXPECT_EQ(iree_atomic_load(&state.cleanup_count, iree_memory_order_acquire),
            1);
}

TEST(CompletionQueueTest, CleanupOnlyBatchRunsCleanupWithoutWorkerSubmission) {
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  TestState state = {};
  iree_atomic_store(&state.cleanup_count, 0, iree_memory_order_relaxed);
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, iree_hal_semaphore_list_empty(), &batch));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_cleanup(
      batch, IncrementCleanup, &state));

  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_submit(batch));
  EXPECT_EQ(iree_atomic_load(&state.cleanup_count, iree_memory_order_acquire),
            1);

  iree_hal_amdxdna_completion_queue_destroy(queue);
}

TEST(CompletionQueueTest, CleanupsRunInReverseRegistrationOrder) {
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  TestState state = {};
  iree_atomic_store(&state.cleanup_count, 0, iree_memory_order_relaxed);
  OrderedCleanup first = {&state, 1};
  OrderedCleanup second = {&state, 2};
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, iree_hal_semaphore_list_empty(), &batch));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_cleanup(
      batch, RecordOrderedCleanup, &first));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_cleanup(
      batch, RecordOrderedCleanup, &second));

  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_submit(batch));
  EXPECT_EQ(state.cleanup_order[0], 2);
  EXPECT_EQ(state.cleanup_order[1], 1);

  iree_hal_amdxdna_completion_queue_destroy(queue);
}

TEST(CompletionQueueTest, PublishedEmptyBatchSignalsAfterSubmit) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t values[] = {1};

  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, MakeSemaphoreList(semaphores, values, IREE_ARRAYSIZE(semaphores)),
      &batch));
  iree_hal_amdxdna_completion_batch_publish_signals(batch);

  std::atomic<bool> wait_started{false};
  std::atomic<iree_status_code_t> wait_code{IREE_STATUS_UNKNOWN};
  std::thread waiter([&] {
    wait_started.store(true, std::memory_order_release);
    iree_status_t status = iree_hal_semaphore_wait(
        semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    wait_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_free(status);
  });

  while (!wait_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 0ull);

  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);

  waiter.join();
  EXPECT_EQ(wait_code.load(std::memory_order_acquire), IREE_STATUS_OK);
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 1ull);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(CompletionQueueTest, PublishedEmptyBatchFailsAfterRecordedError) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t values[] = {1};

  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, MakeSemaphoreList(semaphores, values, IREE_ARRAYSIZE(semaphores)),
      &batch));
  iree_hal_amdxdna_completion_batch_publish_signals(batch);
  iree_hal_amdxdna_completion_batch_record_error(
      batch, iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "expected"));

  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);

  status = iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

// A command that fails while it is being built must surface on this timeline
// point without failing the semaphore. A failed semaphore rejects every later
// wait, including the next request's buffer allocation.
TEST(CompletionQueueTest, CommandBuildErrorDoesNotFailLaterWaits) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t values[] = {1};

  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, MakeSemaphoreList(semaphores, values, IREE_ARRAYSIZE(semaphores)),
      &batch));
  iree_hal_amdxdna_completion_batch_publish_signals(batch);
  iree_hal_amdxdna_completion_batch_record_command_error(
      batch, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "amdxdna native BO allocation failed"));

  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);

  status = iree_hal_semaphore_wait(semaphore, 1, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_status_free(status);

  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 1ull);

  status = iree_hal_semaphore_wait(semaphore, 2, iree_immediate_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  EXPECT_TRUE(iree_status_is_deadline_exceeded(status));
  iree_status_free(status);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(CompletionQueueTest, WorkerOwnedBatchWakesMultipleWaitersOnce) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t values[] = {1};

  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));
  BlockingActionState state;
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, MakeSemaphoreList(semaphores, values, IREE_ARRAYSIZE(semaphores)),
      &batch));
  iree_hal_amdxdna_completion_batch_publish_signals(batch);
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_action(
      batch, BlockingAction, BlockingCleanup, &state,
      /*run_on_error=*/false));
  iree_status_t status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);
  ASSERT_TRUE(WaitForBlockingActionStarted(&state));

  constexpr int32_t kWaiterCount = 4;
  std::atomic<int32_t> wait_started{0};
  std::vector<iree_status_code_t> wait_codes(kWaiterCount, IREE_STATUS_UNKNOWN);
  std::vector<std::thread> waiters;
  waiters.reserve(kWaiterCount);
  for (int32_t i = 0; i < kWaiterCount; ++i) {
    waiters.emplace_back([&, i] {
      wait_started.fetch_add(1, std::memory_order_acq_rel);
      iree_status_t wait_status = iree_hal_semaphore_wait(
          semaphore, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
      wait_codes[i] = iree_status_code(wait_status);
      iree_status_free(wait_status);
    });
  }
  EXPECT_TRUE(WaitForStartedCount(wait_started, kWaiterCount));
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(state.action_count.load(std::memory_order_acquire), 1);
  EXPECT_EQ(state.cleanup_count.load(std::memory_order_acquire), 0);

  ReleaseBlockingAction(&state);
  for (std::thread& waiter : waiters) {
    waiter.join();
  }
  for (iree_status_code_t code : wait_codes) {
    EXPECT_EQ(code, IREE_STATUS_OK);
  }
  EXPECT_EQ(state.action_count.load(std::memory_order_acquire), 1);
  EXPECT_EQ(state.cleanup_count.load(std::memory_order_acquire), 1);
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 1ull);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(CompletionQueueTest, ImmediateWaitDoesNotCompletePendingBatch) {
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  BlockingActionState blocker;
  ASSERT_TRUE(SubmitBlockingBatch(queue, &blocker));
  ASSERT_TRUE(WaitForBlockingActionStarted(&blocker));

  TestState pending_state = {};
  iree_atomic_store(&pending_state.action_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pending_state.cleanup_count, 0, iree_memory_order_relaxed);
  iree_hal_amdxdna_completion_batch_t* pending_batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, iree_hal_semaphore_list_empty(), &pending_batch));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_add_action(
      pending_batch, IncrementAction, IncrementCleanup, &pending_state,
      /*run_on_error=*/false));
  iree_hal_amdxdna_completion_batch_retain(pending_batch);
  iree_status_t status =
      iree_hal_amdxdna_completion_batch_submit(pending_batch);
  EXPECT_TRUE(iree_status_is_deferred(status));
  iree_status_ignore(status);

  status = iree_hal_amdxdna_completion_batch_wait(
      pending_batch, iree_immediate_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_DEADLINE_EXCEEDED);
  iree_status_free(status);
  EXPECT_EQ(
      iree_atomic_load(&pending_state.action_count, iree_memory_order_acquire),
      0);
  EXPECT_EQ(
      iree_atomic_load(&pending_state.cleanup_count, iree_memory_order_acquire),
      0);

  ReleaseBlockingAction(&blocker);
  status = iree_hal_amdxdna_completion_batch_wait(
      pending_batch, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OK);
  iree_status_free(status);
  EXPECT_EQ(
      iree_atomic_load(&pending_state.action_count, iree_memory_order_acquire),
      1);
  EXPECT_EQ(
      iree_atomic_load(&pending_state.cleanup_count, iree_memory_order_acquire),
      0);
  iree_hal_amdxdna_completion_batch_destroy(pending_batch);
  EXPECT_EQ(
      iree_atomic_load(&pending_state.cleanup_count, iree_memory_order_acquire),
      1);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  EXPECT_EQ(blocker.cleanup_count.load(std::memory_order_acquire), 1);
}

// Submits one recorded device status and reports how the semaphore came back.
// A command-scoped failure advances the timeline and leaves a later wait able
// to block. Failing the semaphore makes every later wait return that failure.
struct RecordedWait {
  iree_status_code_t wait_code = IREE_STATUS_UNKNOWN;
  iree_status_code_t later_code = IREE_STATUS_UNKNOWN;
  uint64_t timeline = 0;
  bool query_ok = false;
};

static RecordedWait SubmitRecordedStatus(
    iree_hal_amdxdna_completion_queue_t* queue,
    iree_hal_semaphore_t* semaphore, uint64_t signal_value,
    iree_status_t status) {
  iree_hal_semaphore_t* semaphores[] = {semaphore};
  uint64_t values[] = {signal_value};
  iree_hal_amdxdna_completion_batch_t* batch = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_batch_create(
      queue, MakeSemaphoreList(semaphores, values, 1), &batch));
  iree_hal_amdxdna_completion_batch_publish_signals(batch);
  iree_hal_amdxdna_completion_batch_record_error(batch, status);
  iree_status_t submit_status = iree_hal_amdxdna_completion_batch_submit(batch);
  EXPECT_TRUE(iree_status_is_deferred(submit_status));
  iree_status_ignore(submit_status);

  RecordedWait result;
  iree_status_t wait_status = iree_hal_semaphore_wait(
      semaphore, signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE);
  result.wait_code = iree_status_code(wait_status);
  iree_status_free(wait_status);

  iree_status_t query_status =
      iree_hal_semaphore_query(semaphore, &result.timeline);
  result.query_ok = iree_status_is_ok(query_status);
  if (!result.query_ok) {
    result.timeline = 0;
    iree_status_free(query_status);
  }

  iree_status_t later_status = iree_hal_semaphore_wait(
      semaphore, signal_value + 1, iree_immediate_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE);
  result.later_code = iree_status_code(later_status);
  iree_status_free(later_status);
  return result;
}

TEST(CompletionQueueTest, NewStateWithoutTimeoutFailsSemaphore) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  RecordedWait waited = SubmitRecordedStatus(
      queue, semaphore, 1,
      iree_make_status(IREE_STATUS_INTERNAL,
                       "amdxdna dispatch did not complete: ert state 1"));
  EXPECT_EQ(waited.wait_code, IREE_STATUS_INTERNAL);
  EXPECT_FALSE(waited.query_ok);
  EXPECT_EQ(waited.later_code, IREE_STATUS_INTERNAL);
  EXPECT_FALSE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(CompletionQueueTest, StatusTextDoesNotOpenRecoveryWindow) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  RecordedWait timed_out = SubmitRecordedStatus(
      queue, semaphore, 1,
      iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                       "amdxdna dispatch timed out: ert state 8"));
  EXPECT_EQ(timed_out.wait_code, IREE_STATUS_DEADLINE_EXCEEDED);
  EXPECT_TRUE(timed_out.query_ok);
  EXPECT_EQ(timed_out.timeline, 1ull);
  EXPECT_FALSE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  RecordedWait never_started = SubmitRecordedStatus(
      queue, semaphore, 2,
      iree_make_status(IREE_STATUS_INTERNAL,
                       "amdxdna dispatch did not complete: ert state 1"));
  EXPECT_EQ(never_started.wait_code, IREE_STATUS_INTERNAL);
  EXPECT_FALSE(never_started.query_ok);
  EXPECT_EQ(never_started.later_code, IREE_STATUS_INTERNAL);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(CompletionQueueTest, TimeoutThenNewStaysUsableUntilCloserCompletes) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  iree_hal_amdxdna_completion_queue_note_timeout_recovery(queue);
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  RecordedWait timed_out = SubmitRecordedStatus(
      queue, semaphore, 1,
      iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                       "amdxdna dispatch timed out: ert state 8"));
  EXPECT_EQ(timed_out.wait_code, IREE_STATUS_DEADLINE_EXCEEDED);
  EXPECT_TRUE(timed_out.query_ok);
  EXPECT_EQ(timed_out.timeline, 1ull);
  EXPECT_EQ(timed_out.later_code, IREE_STATUS_DEADLINE_EXCEEDED);
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  RecordedWait never_started = SubmitRecordedStatus(
      queue, semaphore, 2,
      iree_make_status(IREE_STATUS_INTERNAL,
                       "amdxdna dispatch did not complete: ert state 1"));
  EXPECT_EQ(never_started.wait_code, IREE_STATUS_INTERNAL);
  EXPECT_TRUE(never_started.query_ok);
  EXPECT_EQ(never_started.timeline, 2ull);
  EXPECT_EQ(never_started.later_code, IREE_STATUS_DEADLINE_EXCEEDED);
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  uint64_t generation = 0;
  EXPECT_TRUE(iree_hal_amdxdna_completion_queue_arm_recovery_closer(
      queue, &generation));
  uint64_t second = 0;
  EXPECT_FALSE(iree_hal_amdxdna_completion_queue_arm_recovery_closer(
      queue, &second));
  iree_hal_amdxdna_completion_queue_note_closer_completed(queue, generation);
  EXPECT_FALSE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  RecordedWait after_recovery = SubmitRecordedStatus(
      queue, semaphore, 3,
      iree_make_status(IREE_STATUS_INTERNAL,
                       "amdxdna dispatch did not complete: ert state 1"));
  EXPECT_EQ(after_recovery.wait_code, IREE_STATUS_INTERNAL);
  EXPECT_FALSE(after_recovery.query_ok);
  EXPECT_EQ(after_recovery.later_code, IREE_STATUS_INTERNAL);

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(CompletionQueueTest, StaleCloserDoesNotCloseANewerTimeout) {
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  iree_hal_amdxdna_completion_queue_note_timeout_recovery(queue);
  uint64_t first = 0;
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_arm_recovery_closer(queue, &first));
  iree_hal_amdxdna_completion_queue_note_timeout_recovery(queue);
  iree_hal_amdxdna_completion_queue_note_closer_completed(queue, first);
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  uint64_t second = 0;
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_arm_recovery_closer(queue, &second));
  EXPECT_NE(first, second);
  iree_hal_amdxdna_completion_queue_disarm_recovery_closer(queue, second);
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));
  uint64_t retried = 0;
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_arm_recovery_closer(queue, &retried));
  EXPECT_EQ(retried, second);
  iree_hal_amdxdna_completion_queue_note_closer_completed(queue, retried);
  EXPECT_FALSE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  iree_hal_amdxdna_completion_queue_destroy(queue);
}

TEST(CompletionQueueTest, RecoveryWindowDoesNotCrossDevices) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_amdxdna_completion_queue_t* timed_out = nullptr;
  iree_hal_amdxdna_completion_queue_t* other = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &timed_out));
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &other));

  iree_hal_amdxdna_completion_queue_note_timeout_recovery(timed_out);
  RecordedWait never_started = SubmitRecordedStatus(
      other, semaphore, 1,
      iree_make_status(IREE_STATUS_INTERNAL,
                       "amdxdna dispatch did not complete: ert state 1"));
  EXPECT_EQ(never_started.wait_code, IREE_STATUS_INTERNAL);
  EXPECT_FALSE(never_started.query_ok);
  EXPECT_EQ(never_started.later_code, IREE_STATUS_INTERNAL);
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(timed_out));
  EXPECT_FALSE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(other));

  iree_hal_amdxdna_completion_queue_destroy(other);
  iree_hal_amdxdna_completion_queue_destroy(timed_out);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(CompletionQueueTest, NoResponseStaysFatalDuringTimeoutRecovery) {
  iree_async_proactor_t* proactor = CreateProactor();
  iree_hal_semaphore_t* semaphore = CreateSemaphore(proactor, 0);
  iree_hal_amdxdna_completion_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_hal_amdxdna_completion_queue_create(
      iree_allocator_system(), &queue));

  iree_hal_amdxdna_completion_queue_note_timeout_recovery(queue);
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  RecordedWait no_response = SubmitRecordedStatus(
      queue, semaphore, 1,
      iree_make_status(IREE_STATUS_INTERNAL,
                       "amdxdna dispatch did not complete: ert state 9"));
  EXPECT_EQ(no_response.wait_code, IREE_STATUS_INTERNAL);
  EXPECT_FALSE(no_response.query_ok);
  EXPECT_EQ(no_response.later_code, IREE_STATUS_INTERNAL);
  EXPECT_TRUE(
      iree_hal_amdxdna_completion_queue_timeout_recovery_pending(queue));

  iree_hal_amdxdna_completion_queue_destroy(queue);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}
}  // namespace
