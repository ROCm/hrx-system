// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_pool.h"

#include <array>
#include <cstring>
#include <thread>

#include "iree/async/operations/scheduling.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/numa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_status_t RejectProactorCreation(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  (void)options;
  (void)allocator;
  *out_proactor = nullptr;
  return iree_make_status(IREE_STATUS_ABORTED, "selected creator invoked");
}

struct TestRunnerState;

struct TestRunner {
  TestRunnerState* state = nullptr;
  bool stop_requested = false;
};

struct TestRunnerState {
  static constexpr iree_host_size_t kMaxRunners = 32;

  std::array<TestRunner, kMaxRunners> runners;
  iree_host_size_t create_attempt_count = 0;
  iree_host_size_t create_count = 0;
  iree_host_size_t request_stop_count = 0;
  iree_host_size_t destroy_count = 0;
  iree_host_size_t destroy_without_stop_count = 0;
  iree_host_size_t destroy_before_all_stopped_count = 0;
  iree_host_size_t expected_stop_count_before_destroy = 0;
  iree_host_size_t remaining_create_failures = 0;
};

static iree_status_t TestRunnerCreate(void* user_data,
                                      iree_async_proactor_t* proactor,
                                      uint32_t node_id,
                                      iree_allocator_t allocator,
                                      void** out_runner) {
  (void)proactor;
  (void)node_id;
  (void)allocator;
  TestRunnerState* state = (TestRunnerState*)user_data;
  ++state->create_attempt_count;
  *out_runner = nullptr;
  if (state->remaining_create_failures > 0) {
    --state->remaining_create_failures;
    return iree_make_status(IREE_STATUS_ABORTED,
                            "injected runner creation failure");
  }
  if (state->create_count >= state->runners.size()) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "test runner capacity exceeded");
  }

  TestRunner* runner = &state->runners[state->create_count];
  runner->state = state;
  runner->stop_requested = false;
  ++state->create_count;
  *out_runner = runner;
  return iree_ok_status();
}

static void TestRunnerRequestStop(void* user_data, void* runner_ptr) {
  (void)user_data;
  TestRunner* runner = (TestRunner*)runner_ptr;
  if (!runner->stop_requested) {
    runner->stop_requested = true;
    ++runner->state->request_stop_count;
  }
}

static void TestRunnerDestroy(void* user_data, void* runner_ptr) {
  (void)user_data;
  TestRunner* runner = (TestRunner*)runner_ptr;
  TestRunnerState* state = runner->state;
  if (!runner->stop_requested) {
    ++state->destroy_without_stop_count;
  }
  if (state->request_stop_count < state->expected_stop_count_before_destroy) {
    ++state->destroy_before_all_stopped_count;
  }
  ++state->destroy_count;
}

struct NopCompletionState {
  // Publishes completion state to the waiting test thread.
  iree_atomic_int32_t completed = IREE_ATOMIC_VAR_INIT(0);
  // Wakes the test thread when the callback completes.
  iree_notification_t notification;
  // Terminal operation status transferred from the callback.
  iree_status_t status = iree_ok_status();
  // Physical NUMA node observed by the actual polling callback.
  iree_numa_node_id_t observed_node = IREE_NUMA_NODE_ANY;
};

static bool NopCompleted(void* user_data) {
  NopCompletionState* state = (NopCompletionState*)user_data;
  return iree_atomic_load(&state->completed, iree_memory_order_acquire) != 0;
}

static void NopCompletion(void* user_data, iree_async_operation_t* operation,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  NopCompletionState* state = (NopCompletionState*)user_data;
  state->status = status;
  state->observed_node = iree_numa_node_for_current_thread();
  iree_atomic_store(&state->completed, 1, iree_memory_order_release);
  iree_notification_post(&state->notification, IREE_ALL_WAITERS);
}

struct ReleaseEntryCompletionState {
  // Entry whose final reference is released by the completion callback.
  iree_async_proactor_pool_entry_t* entry = nullptr;
  // Terminal operation status transferred from the callback.
  iree_status_t status = iree_ok_status();
};

static void ReleaseEntryCompletion(void* user_data,
                                   iree_async_operation_t* operation,
                                   iree_status_t status,
                                   iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  ReleaseEntryCompletionState* state = (ReleaseEntryCompletionState*)user_data;
  state->status = status;
  iree_async_proactor_pool_entry_t* entry = state->entry;
  state->entry = nullptr;
  iree_async_proactor_pool_entry_release(entry);
}

struct AllocationLifetimeState {
  // Number of allocations currently owned by the pool and its runner.
  iree_atomic_int32_t live_count = IREE_ATOMIC_VAR_INIT(0);
  // Wakes the test when the final allocation is released.
  iree_notification_t depleted;
};

static bool NoLiveAllocations(void* user_data) {
  AllocationLifetimeState* state = (AllocationLifetimeState*)user_data;
  return iree_atomic_load(&state->live_count, iree_memory_order_acquire) == 0;
}

static void ReleaseLiveAllocation(AllocationLifetimeState* state) {
  int32_t previous_count =
      iree_atomic_fetch_sub(&state->live_count, 1, iree_memory_order_acq_rel);
  if (previous_count == 1) {
    iree_notification_post(&state->depleted, IREE_ALL_WAITERS);
  }
}

static iree_status_t AllocationLifetimeCtl(void* self,
                                           iree_allocator_command_t command,
                                           const void* params,
                                           void** inout_ptr) {
  AllocationLifetimeState* state = (AllocationLifetimeState*)self;
  void* old_ptr = *inout_ptr;
  iree_allocator_t system_allocator = iree_allocator_system();
  iree_status_t status =
      system_allocator.ctl(system_allocator.self, command, params, inout_ptr);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
      command == IREE_ALLOCATOR_COMMAND_CALLOC ||
      (command == IREE_ALLOCATOR_COMMAND_REALLOC && !old_ptr && *inout_ptr)) {
    iree_atomic_fetch_add(&state->live_count, 1, iree_memory_order_relaxed);
  } else if ((command == IREE_ALLOCATOR_COMMAND_FREE && old_ptr) ||
             (command == IREE_ALLOCATOR_COMMAND_REALLOC && old_ptr &&
              !*inout_ptr)) {
    ReleaseLiveAllocation(state);
  }
  return iree_ok_status();
}

static iree_allocator_t AllocationLifetimeAllocator(
    AllocationLifetimeState* state) {
  iree_allocator_t allocator = {
      /*.self=*/state,
      /*.ctl=*/AllocationLifetimeCtl,
  };
  return allocator;
}

class ProactorPoolTest : public ::testing::Test {
 protected:
  iree_async_proactor_pool_options_t default_options() {
    return iree_async_proactor_pool_options_default();
  }

  iree_async_proactor_pool_options_t test_runner_options(
      TestRunnerState* state) {
    iree_async_proactor_pool_options_t options = default_options();
    memset(&options.runner, 0, sizeof(options.runner));
    options.runner.user_data = state;
    options.runner.create = TestRunnerCreate;
    options.runner.request_stop = TestRunnerRequestStop;
    options.runner.destroy = TestRunnerDestroy;
    return options;
  }
};

TEST_F(ProactorPoolTest, CreateZeroNodesFails) {
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_proactor_pool_create(
                            0, /*node_ids=*/nullptr, default_options(),
                            iree_allocator_system(), &pool));
  EXPECT_EQ(pool, nullptr);
}

TEST_F(ProactorPoolTest, CreateSingleNodeNoAffinity) {
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(), iree_allocator_system(),
      &pool));
  ASSERT_NE(pool, nullptr);

  EXPECT_EQ(iree_async_proactor_pool_count(pool), 1u);
  EXPECT_EQ(iree_async_proactor_pool_node_id(pool, 0), UINT32_MAX);

  // Out of bounds returns error.
  iree_async_proactor_t* proactor = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_async_proactor_pool_get(pool, 1, &proactor));
  EXPECT_EQ(proactor, nullptr);

  iree_async_proactor_pool_entry_t* entry = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_async_proactor_pool_acquire(pool, 1, &entry));
  EXPECT_EQ(entry, nullptr);
  EXPECT_EQ(iree_async_proactor_pool_node_id(pool, 1), UINT32_MAX);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, CreateAndReleaseWithoutGet) {
  // Pool creation is lazy — creating and releasing without ever calling get
  // should be free (no threads spawned).
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(), iree_allocator_system(),
      &pool));
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(iree_async_proactor_pool_count(pool), 1u);
  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, SelectedCreatorIsLazyAndPropagatesFailure) {
  iree_async_proactor_pool_options_t options = default_options();
  options.proactor_create = RejectProactorCreation;
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, options, iree_allocator_system(), &pool));

  iree_async_proactor_t* proactor = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_async_proactor_pool_get(pool, 0, &proactor));
  EXPECT_EQ(proactor, nullptr);

  iree_async_proactor_pool_entry_t* entry = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_async_proactor_pool_acquire(pool, 0, &entry));
  EXPECT_EQ(entry, nullptr);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, OnDemandGet) {
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(), iree_allocator_system(),
      &pool));

  // First get creates the proactor on-demand.
  iree_async_proactor_t* proactor = nullptr;
  iree_status_t status = iree_async_proactor_pool_get(pool, 0, &proactor);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  EXPECT_NE(proactor, nullptr);

  // Second get returns the same proactor (already created).
  iree_async_proactor_t* proactor_again = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 0, &proactor_again));
  EXPECT_EQ(proactor, proactor_again);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, CreateWithNodeIds) {
  uint32_t node_ids[] = {iree_numa_node_for_current_thread(), UINT32_MAX};
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      2, node_ids, default_options(), iree_allocator_system(), &pool));
  ASSERT_NE(pool, nullptr);

  EXPECT_EQ(iree_async_proactor_pool_count(pool), 2u);
  EXPECT_EQ(iree_async_proactor_pool_node_id(pool, 0), node_ids[0]);
  EXPECT_EQ(iree_async_proactor_pool_node_id(pool, 1), node_ids[1]);

  // Each proactor is distinct (on-demand creation).
  iree_async_proactor_t* proactor_0 = nullptr;
  iree_async_proactor_t* proactor_1 = nullptr;
  iree_status_t status = iree_async_proactor_pool_get(pool, 0, &proactor_0);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 1, &proactor_1));
  EXPECT_NE(proactor_0, nullptr);
  EXPECT_NE(proactor_1, nullptr);
  EXPECT_NE(proactor_0, proactor_1);

  // An unconstrained request selects the first entry even when a later entry
  // has explicitly unspecified placement.
  iree_async_proactor_t* proactor_any = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_pool_get_for_node(pool, UINT32_MAX, &proactor_any));
  EXPECT_EQ(proactor_any, proactor_0);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, GetForNodeExactMatch) {
  uint32_t node_ids[] = {UINT32_MAX, iree_numa_node_for_current_thread()};
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      2, node_ids, default_options(), iree_allocator_system(), &pool));

  // Exact match returns the right proactor.
  iree_async_proactor_t* proactor_local = nullptr;
  iree_async_proactor_t* proactor_any = nullptr;
  iree_status_t status =
      iree_async_proactor_pool_get_for_node(pool, node_ids[1], &proactor_local);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(
      iree_async_proactor_pool_get_for_node(pool, UINT32_MAX, &proactor_any));

  iree_async_proactor_t* proactor_0 = nullptr;
  iree_async_proactor_t* proactor_1 = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 0, &proactor_0));
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool, 1, &proactor_1));
  EXPECT_EQ(proactor_local, proactor_1);
  EXPECT_EQ(proactor_any, proactor_0);

  // An unplaced entry cannot satisfy a concrete locality request. Failures
  // clear the output even when the caller provided a non-null value.
  iree_async_proactor_t* proactor_missing = proactor_local;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_async_proactor_pool_get_for_node(
                            pool, UINT32_MAX - 1, &proactor_missing));
  EXPECT_EQ(proactor_missing, nullptr);

  // Entry acquisition follows the same exact and unconstrained mapping while
  // retaining runner ownership for the caller.
  iree_async_proactor_pool_entry_t* entry_local = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_acquire_for_node(pool, node_ids[1],
                                                           &entry_local));
  EXPECT_EQ(iree_async_proactor_pool_entry_node_id(entry_local), node_ids[1]);
  EXPECT_EQ(iree_async_proactor_pool_entry_proactor(entry_local),
            proactor_local);

  iree_async_proactor_pool_entry_t* entry_any = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_pool_acquire_for_node(pool, UINT32_MAX, &entry_any));
  EXPECT_EQ(iree_async_proactor_pool_entry_node_id(entry_any), UINT32_MAX);
  EXPECT_EQ(iree_async_proactor_pool_entry_proactor(entry_any), proactor_any);
  iree_async_proactor_pool_entry_release(entry_any);

  iree_async_proactor_pool_entry_t* entry_missing = entry_local;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_async_proactor_pool_acquire_for_node(
                            pool, UINT32_MAX - 1, &entry_missing));
  EXPECT_EQ(entry_missing, nullptr);
  iree_async_proactor_pool_entry_release(entry_local);

  iree_async_proactor_pool_release(pool);
}

TEST_F(ProactorPoolTest, RetainRelease) {
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(), iree_allocator_system(),
      &pool));

  // Extra retain keeps the pool alive.
  iree_async_proactor_pool_retain(pool);
  iree_async_proactor_pool_release(pool);  // Drops to ref count 1.

  // Pool should still be usable.
  EXPECT_EQ(iree_async_proactor_pool_count(pool), 1u);

  iree_async_proactor_pool_release(pool);  // Final release, destroys.
}

TEST_F(ProactorPoolTest, AcquiredEntryKeepsRunnerAfterPoolRelease) {
  TestRunnerState runner_state;
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, test_runner_options(&runner_state),
      iree_allocator_system(), &pool));

  // Acquiring the entry owns both the proactor and its progress runner.
  iree_async_proactor_pool_entry_t* entry = nullptr;
  iree_status_t status = iree_async_proactor_pool_acquire(pool, 0, &entry);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  ASSERT_NE(entry, nullptr);
  iree_async_proactor_t* proactor =
      iree_async_proactor_pool_entry_proactor(entry);
  ASSERT_NE(proactor, nullptr);
  EXPECT_EQ(iree_async_proactor_pool_entry_node_id(entry), UINT32_MAX);
  EXPECT_EQ(runner_state.create_count, 1u);

  // Releasing the aggregate pool must not stop a consumer-retained entry.
  iree_async_proactor_pool_release(pool);
  EXPECT_EQ(runner_state.request_stop_count, 0u);
  EXPECT_EQ(runner_state.destroy_count, 0u);
  EXPECT_EQ(iree_async_proactor_pool_entry_proactor(entry), proactor);

  // The final entry release owns runner stop and destruction.
  runner_state.expected_stop_count_before_destroy = 1;
  iree_async_proactor_pool_entry_release(entry);
  EXPECT_EQ(runner_state.request_stop_count, 1u);
  EXPECT_EQ(runner_state.destroy_count, 1u);
  EXPECT_EQ(runner_state.destroy_without_stop_count, 0u);
  EXPECT_EQ(runner_state.destroy_before_all_stopped_count, 0u);
}

TEST_F(ProactorPoolTest, AcquiredEntryMakesProgressAfterPoolRelease) {
  uint32_t node_id = iree_numa_node_for_current_thread();
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, &node_id, default_options(), iree_allocator_system(), &pool));

  iree_async_proactor_pool_entry_t* entry = nullptr;
  iree_status_t status = iree_async_proactor_pool_acquire(pool, 0, &entry);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);

  iree_async_proactor_t* proactor =
      iree_async_proactor_pool_entry_proactor(entry);
  iree_async_proactor_pool_release(pool);

  NopCompletionState completion;
  iree_notification_initialize(&completion.notification);
  iree_async_nop_operation_t nop;
  iree_async_operation_zero(&nop.base, sizeof(nop));
  iree_async_operation_initialize(&nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE, NopCompletion,
                                  &completion);

  status = iree_async_proactor_submit_one(proactor, &nop.base);
  if (!iree_status_is_ok(status)) {
    iree_async_proactor_pool_entry_release(entry);
    iree_notification_deinitialize(&completion.notification);
    IREE_ASSERT_OK(status);
  }

  EXPECT_TRUE(iree_notification_await(&completion.notification, NopCompleted,
                                      &completion, iree_infinite_timeout()));

  // Final entry release requests runner stop and joins the polling thread,
  // ensuring the stack operation and callback state are no longer in use.
  iree_async_proactor_pool_entry_release(entry);
  IREE_EXPECT_OK(completion.status);
  if (node_id != IREE_NUMA_NODE_ANY) {
    EXPECT_EQ(completion.observed_node, node_id);
  }
  iree_notification_deinitialize(&completion.notification);
}

TEST_F(ProactorPoolTest, FinalEntryReleaseFromCompletionRetiresRunner) {
  AllocationLifetimeState allocations;
  iree_notification_initialize(&allocations.depleted);
  iree_async_proactor_pool_t* pool = nullptr;
  iree_status_t status = iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, default_options(),
      AllocationLifetimeAllocator(&allocations), &pool);
  if (!iree_status_is_ok(status)) {
    iree_notification_deinitialize(&allocations.depleted);
    IREE_ASSERT_OK(status);
  }

  iree_async_proactor_pool_entry_t* entry = nullptr;
  status = iree_async_proactor_pool_acquire(pool, 0, &entry);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    iree_async_proactor_pool_release(pool);
    iree_notification_deinitialize(&allocations.depleted);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  if (!iree_status_is_ok(status)) {
    iree_async_proactor_pool_release(pool);
    iree_notification_deinitialize(&allocations.depleted);
    IREE_ASSERT_OK(status);
  }

  iree_async_proactor_t* proactor =
      iree_async_proactor_pool_entry_proactor(entry);
  iree_async_proactor_pool_release(pool);

  ReleaseEntryCompletionState completion;
  completion.entry = entry;
  iree_async_nop_operation_t nop;
  iree_async_operation_zero(&nop.base, sizeof(nop));
  iree_async_operation_initialize(&nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  ReleaseEntryCompletion, &completion);

  status = iree_async_proactor_submit_one(proactor, &nop.base);
  if (!iree_status_is_ok(status)) {
    completion.entry = nullptr;
    iree_async_proactor_pool_entry_release(entry);
    iree_notification_deinitialize(&allocations.depleted);
    IREE_ASSERT_OK(status);
  }

  // The tracked allocator reaches zero only after the callback has returned,
  // poll has unwound, and the deferred runner and proactor owners have released
  // all of their storage. This keeps callback state live through teardown
  // without a wall-clock timeout.
  EXPECT_TRUE(iree_notification_await(&allocations.depleted, NoLiveAllocations,
                                      &allocations, iree_infinite_timeout()));
  IREE_EXPECT_OK(completion.status);
  iree_notification_deinitialize(&allocations.depleted);
}

TEST_F(ProactorPoolTest, PoolReleaseStopsAllEntriesBeforeDestroy) {
  constexpr iree_host_size_t kEntryCount = 17;
  TestRunnerState runner_state;
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      kEntryCount, /*node_ids=*/nullptr, test_runner_options(&runner_state),
      iree_allocator_system(), &pool));

  for (iree_host_size_t i = 0; i < kEntryCount; ++i) {
    iree_async_proactor_t* proactor = nullptr;
    iree_status_t status = iree_async_proactor_pool_get(pool, i, &proactor);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      runner_state.expected_stop_count_before_destroy =
          runner_state.create_count;
      iree_async_proactor_pool_release(pool);
      GTEST_SKIP() << "Platform proactor unavailable";
    }
    IREE_ASSERT_OK(status);
    ASSERT_NE(proactor, nullptr);
  }

  runner_state.expected_stop_count_before_destroy = kEntryCount;
  iree_async_proactor_pool_release(pool);

  EXPECT_EQ(runner_state.create_count, kEntryCount);
  EXPECT_EQ(runner_state.request_stop_count, kEntryCount);
  EXPECT_EQ(runner_state.destroy_count, kEntryCount);
  EXPECT_EQ(runner_state.destroy_without_stop_count, 0u);
  EXPECT_EQ(runner_state.destroy_before_all_stopped_count, 0u);
}

TEST_F(ProactorPoolTest, RunnerCreationFailureCanRetry) {
  TestRunnerState runner_state;
  runner_state.remaining_create_failures = 1;
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, test_runner_options(&runner_state),
      iree_allocator_system(), &pool));

  iree_async_proactor_pool_entry_t* entry = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_async_proactor_pool_acquire(pool, 0, &entry));
  EXPECT_EQ(entry, nullptr);
  EXPECT_EQ(runner_state.create_attempt_count, 1u);
  EXPECT_EQ(runner_state.create_count, 0u);

  iree_status_t status = iree_async_proactor_pool_acquire(pool, 0, &entry);
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    iree_async_proactor_pool_release(pool);
    GTEST_SKIP() << "Platform proactor unavailable";
  }
  IREE_ASSERT_OK(status);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(runner_state.create_attempt_count, 2u);
  EXPECT_EQ(runner_state.create_count, 1u);

  iree_async_proactor_pool_entry_release(entry);
  runner_state.expected_stop_count_before_destroy = 1;
  iree_async_proactor_pool_release(pool);
  EXPECT_EQ(runner_state.request_stop_count, 1u);
  EXPECT_EQ(runner_state.destroy_count, 1u);
}

TEST_F(ProactorPoolTest, ConcurrentAcquireInitializesOneEntry) {
  constexpr iree_host_size_t kThreadCount = 8;
  TestRunnerState runner_state;
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, test_runner_options(&runner_state),
      iree_allocator_system(), &pool));

  std::array<iree_async_proactor_pool_entry_t*, kThreadCount> entries = {};
  std::array<iree_status_t, kThreadCount> statuses = {};
  std::array<std::thread, kThreadCount> threads;
  for (iree_host_size_t i = 0; i < kThreadCount; ++i) {
    threads[i] = std::thread([&, i]() {
      statuses[i] = iree_async_proactor_pool_acquire(pool, 0, &entries[i]);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  for (iree_host_size_t i = 0; i < kThreadCount; ++i) {
    IREE_EXPECT_OK(statuses[i]);
    ASSERT_NE(entries[i], nullptr);
    EXPECT_EQ(entries[i], entries[0]);
    iree_async_proactor_pool_entry_release(entries[i]);
  }
  EXPECT_EQ(runner_state.create_attempt_count, 1u);
  EXPECT_EQ(runner_state.create_count, 1u);

  runner_state.expected_stop_count_before_destroy = 1;
  iree_async_proactor_pool_release(pool);
  EXPECT_EQ(runner_state.request_stop_count, 1u);
  EXPECT_EQ(runner_state.destroy_count, 1u);
}

}  // namespace
