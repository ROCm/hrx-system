// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper to allocate a frontier on the stack with a given capacity.
#define FRONTIER_ALLOC(name, capacity)                                  \
  alignas(16) uint8_t                                                   \
      name##_storage[sizeof(iree_async_frontier_t) +                    \
                     (capacity) * sizeof(iree_async_frontier_entry_t)]; \
  iree_async_frontier_t* name =                                         \
      reinterpret_cast<iree_async_frontier_t*>(name##_storage);         \
  memset(name##_storage, 0, sizeof(name##_storage))

// Helper to build a frontier from a list of (axis, epoch) pairs.
static iree_async_frontier_t* BuildFrontier(
    uint8_t* storage, iree_host_size_t storage_size,
    std::initializer_list<iree_async_frontier_entry_t> entries) {
  iree_async_frontier_t* frontier =
      reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_frontier_initialize(frontier,
                                 static_cast<uint8_t>(entries.size()));
  uint8_t i = 0;
  for (const auto& entry : entries) {
    frontier->entries[i++] = entry;
  }
  return frontier;
}

#define MAKE_FRONTIER(name, capacity, ...)                              \
  alignas(16) uint8_t                                                   \
      name##_storage[sizeof(iree_async_frontier_t) +                    \
                     (capacity) * sizeof(iree_async_frontier_entry_t)]; \
  iree_async_frontier_t* name =                                         \
      BuildFrontier(name##_storage, sizeof(name##_storage), {__VA_ARGS__})

// Shorthand for creating frontier entries.
static iree_async_frontier_entry_t E(iree_async_axis_t axis, uint64_t epoch) {
  return {axis, epoch};
}

// Test axes.
static iree_async_axis_t Axis(uint8_t index) {
  return iree_async_axis_make_queue(1, 0, 0, index, 0);
}

// Callback tracking state.
struct CallbackState {
  // Number of times the callback fired.
  std::atomic<int> call_count{0};

  // Status code captured from the most recent callback.
  iree_status_code_t last_status_code{IREE_STATUS_OK};

  void Reset() {
    call_count = 0;
    last_status_code = IREE_STATUS_OK;
  }
};

// Standard callback that tracks invocations.
static void TrackingCallback(void* user_data, iree_status_t status) {
  auto* state = reinterpret_cast<CallbackState*>(user_data);
  state->call_count.fetch_add(1, std::memory_order_relaxed);
  state->last_status_code = iree_status_code(status);
  iree_status_free(status);
}

// RAII wrapper for tracker create/release.
class TrackerFixture {
 public:
  explicit TrackerFixture(uint32_t axis_capacity = 16) {
    iree_async_frontier_tracker_options_t options =
        iree_async_frontier_tracker_options_default();
    options.axis_table_capacity = axis_capacity;
    IREE_CHECK_OK(iree_async_frontier_tracker_create(
        options, iree_allocator_system(), &tracker_));
  }

  ~TrackerFixture() { iree_async_frontier_tracker_release(tracker_); }

  iree_async_frontier_tracker_t* tracker() { return tracker_; }

  void AddAxis(iree_async_axis_t axis,
               iree_async_semaphore_t* semaphore = nullptr) {
    IREE_CHECK_OK(RegisterAxis(axis, semaphore));
  }

  iree_status_t RegisterAxis(iree_async_axis_t axis,
                             iree_async_semaphore_t* semaphore = nullptr) {
    return iree_async_frontier_tracker_register_axis(tracker_, axis, semaphore);
  }

 private:
  iree_async_frontier_tracker_t* tracker_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Section 1: Initialize/Deinitialize
//===----------------------------------------------------------------------===//

TEST(InitializeTest, BasicInitialization) {
  TrackerFixture fixture(16);
  fixture.AddAxis(Axis(0));
  EXPECT_FALSE(
      iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(0), 1));
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 1);
  EXPECT_TRUE(
      iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(0), 1));
}

TEST(InitializeTest, ZeroCapacity) {
  TrackerFixture fixture(0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        fixture.RegisterAxis(Axis(0)));
}

TEST(RegisterAxisTest, DuplicateAxisFails) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        fixture.RegisterAxis(Axis(0)));
}

TEST(RegisterAxisTest, FullTableDuplicateReportsDuplicate) {
  TrackerFixture fixture(1);
  fixture.AddAxis(Axis(0));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        fixture.RegisterAxis(Axis(0)));
}

TEST(DeinitializeTest, NoWaiters) {
  // Just verify no crash when deinitializing with no waiters.
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));
}

TEST(DeinitializeTest, PendingWaitersCancelled) {
  iree_async_frontier_tracker_options_t options =
      iree_async_frontier_tracker_options_default();
  options.axis_table_capacity = 4;
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      options, iree_allocator_system(), &tracker));
  IREE_ASSERT_OK(
      iree_async_frontier_tracker_register_axis(tracker, Axis(0), nullptr));

  // Create a waiter that won't be satisfied.
  MAKE_FRONTIER(f, 1, E(Axis(0), 100));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(tracker, f, TrackingCallback,
                                                  &state, &waiter));
  EXPECT_EQ(state.call_count, 0);  // Not yet satisfied.

  // Releasing the last reference should cancel the waiter.
  iree_async_frontier_tracker_release(tracker);
  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_CANCELLED);
}

//===----------------------------------------------------------------------===//
// Section 2: Advance (basic, no waiters)
//===----------------------------------------------------------------------===//

TEST(AdvanceTest, SingleAxis) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  EXPECT_EQ(dispatched, 0u);

  // Verify epoch was updated.
  EXPECT_TRUE(
      iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(0), 5));
  EXPECT_FALSE(
      iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(0), 6));
}

TEST(AdvanceTest, BelowCurrent) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 10);
  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  EXPECT_EQ(dispatched, 0u);

  // Epoch should still be 10 (monotonic).
  EXPECT_TRUE(
      iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(0), 10));
  EXPECT_FALSE(
      iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(0), 11));
}

TEST(AdvanceTest, EqualToCurrent) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 10);
  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 10);
  EXPECT_EQ(dispatched, 0u);
}

TEST(AdvanceTest, LargeValue) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  uint64_t large = 0xFFFFFFFFFFFFFFFEull;
  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), large);
  EXPECT_EQ(dispatched, 0u);

  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(fixture.tracker(),
                                                      Axis(0), large));
}

TEST(AdvanceTest, UnknownAxis) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(99), 5);
  EXPECT_EQ(dispatched, 0u);
}

//===----------------------------------------------------------------------===//
// Section 3: Wait (immediate satisfaction)
//===----------------------------------------------------------------------===//

TEST(WaitTest, EmptyFrontierImmediatelySatisfied) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 0);  // Empty frontier.
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 1);  // Immediately satisfied.
  EXPECT_EQ(state.last_status_code, IREE_STATUS_OK);
}

TEST(WaitTest, AlreadySatisfied) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 10);

  MAKE_FRONTIER(f, 1, E(Axis(0), 5));  // Needs epoch 5, current is 10.
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_OK);
}

TEST(WaitTest, NotYetSatisfiedPends) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(0), 10));  // Needs epoch 10, current is 0.
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 0);  // Not yet.

  // Advance to satisfy.
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 10);
  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_OK);
}

TEST(WaitTest, UnknownAxisReturnsError) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(99), 5));  // Unknown axis.
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_async_frontier_tracker_wait(fixture.tracker(), f, TrackingCallback,
                                       &state, &waiter));
  EXPECT_EQ(state.call_count, 0);  // Callback not invoked.
}

//===----------------------------------------------------------------------===//
// Section 4: Advance + Wait interaction
//===----------------------------------------------------------------------===//

TEST(AdvanceWaitTest, WaitThenAdvance) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(0), 5));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 0);

  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  EXPECT_EQ(dispatched, 1u);
  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_OK);
}

TEST(AdvanceWaitTest, MultiEntryAdvanceOne) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));
  fixture.AddAxis(Axis(1));

  MAKE_FRONTIER(f, 2, E(Axis(0), 5), E(Axis(1), 5));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 0);

  // Advance only one axis.
  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  EXPECT_EQ(dispatched, 0u);
  EXPECT_EQ(state.call_count, 0);  // Still waiting for Axis(1).

  // Cancel before fixture destruction to avoid use-after-scope.
  EXPECT_TRUE(
      iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiter));
}

TEST(AdvanceWaitTest, MultiEntryAdvanceAll) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));
  fixture.AddAxis(Axis(1));

  MAKE_FRONTIER(f, 2, E(Axis(0), 5), E(Axis(1), 5));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));

  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  EXPECT_EQ(state.call_count, 0);

  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(1), 5);
  EXPECT_EQ(dispatched, 1u);
  EXPECT_EQ(state.call_count, 1);
}

TEST(AdvanceWaitTest, MultipleWaitersSameFrontier) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(0), 5));

  iree_async_frontier_waiter_t waiter1, waiter2;
  CallbackState state1, state2;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state1, &waiter1));
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state2, &waiter2));

  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  EXPECT_EQ(dispatched, 2u);
  EXPECT_EQ(state1.call_count, 1);
  EXPECT_EQ(state2.call_count, 1);
}

TEST(AdvanceWaitTest, MultipleWaitersDifferentFrontiers) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));
  fixture.AddAxis(Axis(1));

  MAKE_FRONTIER(f1, 1, E(Axis(0), 5));
  MAKE_FRONTIER(f2, 1, E(Axis(1), 5));

  iree_async_frontier_waiter_t waiter1, waiter2;
  CallbackState state1, state2;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f1, TrackingCallback, &state1, &waiter1));
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f2, TrackingCallback, &state2, &waiter2));

  // Advance Axis(0); only waiter1 should fire.
  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  EXPECT_EQ(dispatched, 1u);
  EXPECT_EQ(state1.call_count, 1);
  EXPECT_EQ(state2.call_count, 0);

  // Advance Axis(1); waiter2 fires.
  dispatched =
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(1), 5);
  EXPECT_EQ(dispatched, 1u);
  EXPECT_EQ(state2.call_count, 1);
}

//===----------------------------------------------------------------------===//
// Section 5: Cancel
//===----------------------------------------------------------------------===//

TEST(CancelTest, CancelPendingWaiter) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(0), 10));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 0);

  EXPECT_TRUE(
      iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiter));

  // Advance past the frontier; callback should NOT fire.
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 20);
  EXPECT_EQ(state.call_count, 0);
}

TEST(CancelTest, CancelAlreadyDispatched) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(0), 5));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  EXPECT_EQ(state.call_count, 1);

  // Cancel after already dispatched; should be a no-op.
  EXPECT_FALSE(
      iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiter));
  EXPECT_EQ(state.call_count, 1);  // Still 1, not re-invoked or anything weird.
}

TEST(CancelTest, DoubleCancel) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(0), 10));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));

  EXPECT_TRUE(
      iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiter));
  // Second cancel should be safe.
  EXPECT_FALSE(
      iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiter));
  EXPECT_EQ(state.call_count, 0);
}

//===----------------------------------------------------------------------===//
// Section 6: Fail axis
//===----------------------------------------------------------------------===//

TEST(FailAxisTest, FailWithPendingWaiter) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(0), 10));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 0);

  iree_async_frontier_tracker_fail_axis(
      fixture.tracker(), Axis(0),
      iree_make_status(IREE_STATUS_INTERNAL, "device lost"));

  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_INTERNAL);
}

TEST(FailAxisTest, FailThenWait) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  iree_async_frontier_tracker_fail_axis(
      fixture.tracker(), Axis(0),
      iree_make_status(IREE_STATUS_UNAVAILABLE, "GPU gone"));

  MAKE_FRONTIER(f, 1, E(Axis(0), 5));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  // Should fail immediately.
  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_UNAVAILABLE);
}

TEST(FailAxisTest, FailTwice) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  iree_async_frontier_tracker_fail_axis(
      fixture.tracker(), Axis(0),
      iree_make_status(IREE_STATUS_INTERNAL, "first failure"));

  // Second failure should be ignored.
  iree_async_frontier_tracker_fail_axis(
      fixture.tracker(), Axis(0),
      iree_make_status(IREE_STATUS_UNAVAILABLE, "second failure"));

  MAKE_FRONTIER(f, 1, E(Axis(0), 5));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  // Should get the first failure status code.
  EXPECT_EQ(state.last_status_code, IREE_STATUS_INTERNAL);
}

TEST(FailAxisTest, FailOneAxisOfMulti) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));
  fixture.AddAxis(Axis(1));

  MAKE_FRONTIER(f, 2, E(Axis(0), 5), E(Axis(1), 5));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 0);

  // Fail Axis(1); waiter should fail even though Axis(0) is not satisfied.
  iree_async_frontier_tracker_fail_axis(
      fixture.tracker(), Axis(1),
      iree_make_status(IREE_STATUS_INTERNAL, "axis 1 failed"));

  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_INTERNAL);
}

TEST(FailAxisTest, FailUnknownAxis) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  // Failing an unknown axis should be a no-op (no crash, status freed).
  iree_async_frontier_tracker_fail_axis(
      fixture.tracker(), Axis(99),
      iree_make_status(IREE_STATUS_INTERNAL, "unknown axis"));
}

TEST(RetireAxisTest, RetireWithPendingWaiter) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  MAKE_FRONTIER(f, 1, E(Axis(0), 10));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));
  EXPECT_EQ(state.call_count, 0);

  iree_async_frontier_tracker_retire_axis(
      fixture.tracker(), Axis(0),
      iree_make_status(IREE_STATUS_CANCELLED, "device teardown"));

  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_CANCELLED);
}

TEST(RetireAxisTest, RetireThenWait) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  iree_async_frontier_tracker_retire_axis(
      fixture.tracker(), Axis(0),
      iree_make_status(IREE_STATUS_CANCELLED, "device teardown"));

  MAKE_FRONTIER(f, 1, E(Axis(0), 5));
  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));

  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_CANCELLED);
}

//===----------------------------------------------------------------------===//
// Section 7: Semaphore bridging
//===----------------------------------------------------------------------===//

// Note: Full semaphore bridging tests would require a mock semaphore.
// For now, we just verify no crash when semaphore is NULL.

TEST(SemaphoreBridgingTest, AdvanceWithoutSemaphore) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));  // No semaphore.

  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 5);
  // No crash; that's the test.
}

TEST(SemaphoreBridgingTest, FailureCallbackRegistersAxisAndRetiresBridge) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  std::unique_ptr<iree_async_proactor_t, decltype(&iree_async_proactor_release)>
      proactor_owner(proactor, iree_async_proactor_release);
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor, /*initial_value=*/0,
      IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, iree_allocator_system(),
      &semaphore));
  std::unique_ptr<iree_async_semaphore_t,
                  decltype(&iree_async_semaphore_release)>
      semaphore_owner(semaphore, iree_async_semaphore_release);

  TrackerFixture fixture(2);
  fixture.AddAxis(Axis(0), semaphore);
  struct CallbackData {
    // Tracker whose failure dispatch invokes this callback.
    iree_async_frontier_tracker_t* tracker;
    // Caller-owned bridge reference released after retiring the axis.
    decltype(semaphore_owner)* semaphore;
    // Number of synchronous failure notifications received.
    int call_count;
  } callback_data = {fixture.tracker(), &semaphore_owner, 0};
  iree_async_semaphore_timepoint_t timepoint = {};
  timepoint.user_data = &callback_data;
  timepoint.callback = [](void* user_data,
                          iree_async_semaphore_timepoint_t* timepoint,
                          iree_status_t status) {
    auto* data = static_cast<CallbackData*>(user_data);
    ++data->call_count;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
    IREE_EXPECT_OK(iree_async_frontier_tracker_register_axis(data->tracker,
                                                             Axis(1), nullptr));
    iree_async_frontier_tracker_retire_axis(
        data->tracker, Axis(0), iree_status_from_code(IREE_STATUS_CANCELLED));
    data->semaphore->reset();
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      semaphore, /*minimum_value=*/1, &timepoint));

  iree_async_frontier_tracker_fail_axis(
      fixture.tracker(), Axis(0), iree_status_from_code(IREE_STATUS_ABORTED));
  EXPECT_EQ(callback_data.call_count, 1);
  EXPECT_EQ(semaphore_owner.get(), nullptr);
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(1), 1);
  EXPECT_TRUE(
      iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(1), 1));
}

//===----------------------------------------------------------------------===//
// Section 8: Concurrency
//===----------------------------------------------------------------------===//

TEST(ConcurrencyTest, RegisterWhileAdvancingAndQuerying) {
  constexpr int kThreadCount = 4;
  constexpr int kAxesPerThread = 32;
  constexpr int kRegistrationCount = kThreadCount * kAxesPerThread;
  TrackerFixture fixture(kRegistrationCount + 1);
  const iree_async_axis_t running_axis = Axis(kRegistrationCount);
  fixture.AddAxis(running_axis);

  std::atomic<bool> start{false};
  std::atomic<int> remaining{kThreadCount};
  std::vector<std::thread> threads;
  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kAxesPerThread; ++i) {
        const iree_async_axis_t axis = Axis(thread_index * kAxesPerThread + i);
        IREE_EXPECT_OK(fixture.RegisterAxis(axis));
        iree_async_frontier_tracker_advance(fixture.tracker(), axis, 1);
      }
      remaining.fetch_sub(1, std::memory_order_release);
    });
  }

  start.store(true, std::memory_order_release);
  uint64_t epoch = 0;
  do {
    iree_async_frontier_tracker_advance(fixture.tracker(), running_axis,
                                        ++epoch);
    EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(fixture.tracker(),
                                                        running_axis, epoch));
    for (int i = 0; i < kRegistrationCount; ++i) {
      // Unknown axes may become visible during this scan. Exercise publication
      // without an external synchronization edge from the registering thread.
      iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(i), 1);
    }
  } while (remaining.load(std::memory_order_acquire) != 0);

  for (auto& thread : threads) thread.join();
  for (int i = 0; i < kRegistrationCount; ++i) {
    EXPECT_TRUE(
        iree_async_frontier_tracker_query_epoch(fixture.tracker(), Axis(i), 1));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        fixture.RegisterAxis(Axis(kRegistrationCount + 1)));
}

TEST(ConcurrencyTest, ConcurrentDuplicateRegistration) {
  constexpr int kThreadCount = 8;
  TrackerFixture fixture(2);
  std::atomic<bool> start{false};
  std::array<iree_status_code_t, kThreadCount> results;
  std::vector<std::thread> threads;
  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      iree_status_t status = fixture.RegisterAxis(Axis(0));
      results[thread_index] = iree_status_code(status);
      iree_status_free(status);
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  int successful_registrations = 0;
  for (iree_status_code_t result : results) {
    if (result == IREE_STATUS_OK) {
      ++successful_registrations;
    } else {
      EXPECT_EQ(result, IREE_STATUS_ALREADY_EXISTS);
    }
  }
  EXPECT_EQ(successful_registrations, 1);
  IREE_EXPECT_OK(fixture.RegisterAxis(Axis(1)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        fixture.RegisterAxis(Axis(2)));
}

TEST(ConcurrencyTest, ConcurrentAdvanceDifferentAxes) {
  TrackerFixture fixture(16);
  for (int i = 0; i < 8; ++i) {
    fixture.AddAxis(Axis(i));
  }

  // Create persistent frontiers and waiters for the threads.
  struct WaiterData {
    // Waiter storage registered with the tracker.
    iree_async_frontier_waiter_t waiter;

    // Callback state updated when the waiter dispatches.
    CallbackState state;

    // Inline storage for a one-entry frontier.
    alignas(16) uint8_t frontier_storage[sizeof(iree_async_frontier_t) +
                                         sizeof(iree_async_frontier_entry_t)];

    // Frontier view over |frontier_storage|.
    iree_async_frontier_t* frontier;
  };
  std::vector<WaiterData> waiter_data(8);
  for (int i = 0; i < 8; ++i) {
    waiter_data[i].frontier = reinterpret_cast<iree_async_frontier_t*>(
        waiter_data[i].frontier_storage);
    iree_async_frontier_initialize(waiter_data[i].frontier, 1);
    waiter_data[i].frontier->entries[0] = E(Axis(i), 10);
    IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
        fixture.tracker(), waiter_data[i].frontier, TrackingCallback,
        &waiter_data[i].state, &waiter_data[i].waiter));
  }

  // Launch threads to advance each axis.
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&fixture, i]() {
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(i), 10);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // All waiters should have fired exactly once.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(waiter_data[i].state.call_count, 1)
        << "Waiter " << i << " didn't fire exactly once";
    EXPECT_EQ(waiter_data[i].state.last_status_code, IREE_STATUS_OK);
  }
}

TEST(ConcurrencyTest, ConcurrentAdvanceSameAxis) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  // Create a waiter.
  alignas(16) uint8_t frontier_storage[sizeof(iree_async_frontier_t) +
                                       sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage);
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = E(Axis(0), 100);

  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), frontier, TrackingCallback, &state, &waiter));

  // Launch many threads trying to advance the same axis.
  std::vector<std::thread> threads;
  for (int i = 0; i < 16; ++i) {
    threads.emplace_back([&fixture]() {
      iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 100);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // Waiter should have fired exactly once.
  EXPECT_EQ(state.call_count, 1);
  EXPECT_EQ(state.last_status_code, IREE_STATUS_OK);
}

//===----------------------------------------------------------------------===//
// Section 9: Scenarios
//===----------------------------------------------------------------------===//

TEST(ScenarioTest, GpuPipeline) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));  // GPU queue 0.

  // Op1 signals epoch 1.
  // Op2 waits for epoch 1, signals epoch 2.
  // Op3 waits for epoch 2.

  alignas(16) uint8_t f1_storage[sizeof(iree_async_frontier_t) +
                                 sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* f1 =
      reinterpret_cast<iree_async_frontier_t*>(f1_storage);
  iree_async_frontier_initialize(f1, 1);
  f1->entries[0] = E(Axis(0), 1);

  alignas(16) uint8_t f2_storage[sizeof(iree_async_frontier_t) +
                                 sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* f2 =
      reinterpret_cast<iree_async_frontier_t*>(f2_storage);
  iree_async_frontier_initialize(f2, 1);
  f2->entries[0] = E(Axis(0), 2);

  iree_async_frontier_waiter_t waiter1, waiter2;
  CallbackState state1, state2;

  // Op2 waits for epoch 1.
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f1, TrackingCallback, &state1, &waiter1));
  // Op3 waits for epoch 2.
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f2, TrackingCallback, &state2, &waiter2));

  EXPECT_EQ(state1.call_count, 0);
  EXPECT_EQ(state2.call_count, 0);

  // Op1 completes.
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 1);
  EXPECT_EQ(state1.call_count, 1);  // Op2 can proceed.
  EXPECT_EQ(state2.call_count, 0);  // Op3 still waiting.

  // Op2 completes.
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 2);
  EXPECT_EQ(state2.call_count, 1);  // Op3 can proceed.
}

TEST(ScenarioTest, FanOutFanIn) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));  // GPU0.
  fixture.AddAxis(Axis(1));  // GPU1.

  // GPU0 produces, GPU1 processes, then something waits for both.
  alignas(16) uint8_t f_storage[sizeof(iree_async_frontier_t) +
                                2 * sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* f =
      reinterpret_cast<iree_async_frontier_t*>(f_storage);
  iree_async_frontier_initialize(f, 2);
  f->entries[0] = E(Axis(0), 10);
  f->entries[1] = E(Axis(1), 5);

  iree_async_frontier_waiter_t waiter;
  CallbackState state;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f, TrackingCallback, &state, &waiter));

  // GPU0 finishes.
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 10);
  EXPECT_EQ(state.call_count, 0);

  // GPU1 finishes.
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(1), 5);
  EXPECT_EQ(state.call_count, 1);
}

TEST(ScenarioTest, DeviceLost) {
  TrackerFixture fixture(4);
  fixture.AddAxis(Axis(0));

  // Multiple waiters on GPU0.
  alignas(16) uint8_t f1_storage[sizeof(iree_async_frontier_t) +
                                 sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* f1 =
      reinterpret_cast<iree_async_frontier_t*>(f1_storage);
  iree_async_frontier_initialize(f1, 1);
  f1->entries[0] = E(Axis(0), 10);

  alignas(16) uint8_t f2_storage[sizeof(iree_async_frontier_t) +
                                 sizeof(iree_async_frontier_entry_t)];
  iree_async_frontier_t* f2 =
      reinterpret_cast<iree_async_frontier_t*>(f2_storage);
  iree_async_frontier_initialize(f2, 1);
  f2->entries[0] = E(Axis(0), 20);

  iree_async_frontier_waiter_t waiter1, waiter2;
  CallbackState state1, state2;
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f1, TrackingCallback, &state1, &waiter1));
  IREE_EXPECT_OK(iree_async_frontier_tracker_wait(
      fixture.tracker(), f2, TrackingCallback, &state2, &waiter2));

  // GPU0 dies.
  iree_async_frontier_tracker_fail_axis(
      fixture.tracker(), Axis(0),
      iree_make_status(IREE_STATUS_INTERNAL, "GPU0 device lost"));

  // Both waiters should fail.
  EXPECT_EQ(state1.call_count, 1);
  EXPECT_EQ(state1.last_status_code, IREE_STATUS_INTERNAL);
  EXPECT_EQ(state2.call_count, 1);
  EXPECT_EQ(state2.last_status_code, IREE_STATUS_INTERNAL);
}

}  // namespace
