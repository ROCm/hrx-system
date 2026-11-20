// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/pool_wait.h"

#include <atomic>
#include <thread>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/proactor_thread.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct Completion {
  // Final callback count, published after all other callback state accesses.
  std::atomic<int> count{0};
  // Result owned and consumed by the terminal callback.
  iree_status_code_t code = IREE_STATUS_UNKNOWN;

  static void Complete(void* user_data, iree_status_t status) {
    auto* self = static_cast<Completion*>(user_data);
    self->code = iree_status_code(status);
    iree_status_free(status);
    self->count.fetch_add(1, std::memory_order_release);
  }

  static void CompleteOperation(void* user_data, iree_async_operation_t*,
                                iree_status_t status,
                                iree_async_completion_flags_t) {
    Complete(user_data, status);
  }

  iree_hal_pool_wait_callback_t callback() { return {Complete, this}; }

  void Await(iree_status_code_t expected) {
    while (count.load(std::memory_order_acquire) == 0) {
      std::this_thread::yield();
    }
    EXPECT_EQ(count.load(std::memory_order_acquire), 1);
    EXPECT_EQ(code, expected);
  }
};

// Faults only dependency admission. Notification ownership, polling, native
// monitoring and cancellation still use the real platform proactor.
struct SubmissionGate {
  // Active fixture, published before its owner thread starts.
  static SubmissionGate* current;
  // Native implementation used by all operations other than rejected waits.
  const iree_async_proactor_vtable_t* native = nullptr;
  // Fixture-owned vtable with only submit replaced.
  iree_async_proactor_vtable_t vtable = {};
  // Deliberate admission failure, independent of native completion races.
  std::atomic<bool> reject_waits{false};

  static iree_status_t Submit(iree_async_proactor_t* proactor,
                              iree_async_operation_list_t operations,
                              iree_host_size_t* out_retry_batch_size) {
    for (iree_host_size_t i = 0; i < operations.count; ++i) {
      if (operations.values[i]->type ==
              IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT &&
          current->reject_waits.load(std::memory_order_acquire)) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
    }
    return current->native->submit(proactor, operations, out_retry_batch_size);
  }
};
SubmissionGate* SubmissionGate::current = nullptr;

class PoolWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker_));
    auto options = iree_async_proactor_options_default();
    options.threading_mode = IREE_ASYNC_PROACTOR_THREADING_CROSS_THREAD;
    for (int i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_async_proactor_create_platform(
          options, iree_allocator_system(), &proactors_[i]));
      IREE_ASSERT_OK(iree_async_notification_create(
          proactors_[i], IREE_ASYNC_NOTIFICATION_FLAG_NONE,
          &notifications_[i]));
    }
    SubmissionGate::current = &gate_;
    gate_.native = proactors_[1]->vtable;
    gate_.vtable = *gate_.native;
    gate_.vtable.submit = SubmissionGate::Submit;
    proactors_[1]->vtable = &gate_.vtable;
    for (int i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_async_proactor_thread_create(
          proactors_[i], iree_async_proactor_thread_options_default(),
          iree_allocator_system(), &threads_[i]));
    }

    // The waiter consumes only captured pool facts. Actual allocation and
    // materialization are exercised by pool_test and the queue CTS.
    IREE_ASSERT_OK(iree_hal_pool_initialize(
        &pool_vtable_, nullptr, notifications_[0], {2, notifications_},
        tracker_, iree_allocator_system(), &pool_));
    EXPECT_EQ(pool_.wait_sources.count, 2u);
    IREE_ASSERT_OK(
        iree_hal_pool_wait_allocate(&pool_, iree_allocator_system(), &wait_));
  }

  void TearDown() override {
    iree_hal_pool_wait_free(wait_);
    iree_hal_pool_deinitialize(&pool_);
    for (int i = 0; i < 2; ++i) {
      iree_async_notification_release(notifications_[i]);
      iree_async_proactor_thread_request_stop(threads_[i]);
      IREE_ASSERT_OK(
          iree_async_proactor_thread_join(threads_[i], IREE_DURATION_INFINITE));
      IREE_EXPECT_OK(iree_async_proactor_thread_consume_status(threads_[i]));
      iree_async_proactor_thread_release(threads_[i]);
      iree_async_proactor_release(proactors_[i]);
    }
    SubmissionGate::current = nullptr;
    iree_async_frontier_tracker_release(tracker_);
  }

  void DrainOwner(int index) {
    Completion completion;
    iree_async_nop_operation_t marker = {};
    iree_async_operation_initialize(&marker.base, IREE_ASYNC_OPERATION_TYPE_NOP,
                                    IREE_ASYNC_OPERATION_FLAG_NONE,
                                    Completion::CompleteOperation, &completion);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactors_[index], &marker.base));
    completion.Await(IREE_STATUS_OK);
  }

  void ExpectNoObservers() {
    for (auto* notification : notifications_) {
      EXPECT_EQ(iree_atomic_load(&notification->observer_count,
                                 iree_memory_order_acquire),
                0);
    }
  }

  // Real notification owners with independent polling threads.
  iree_async_proactor_t* proactors_[2] = {};
  // Threads remain alive until every wait and captured notification retires.
  iree_async_proactor_thread_t* threads_[2] = {};
  // Capacity sources shared by the helper and unrelated waiters.
  iree_async_notification_t* notifications_[2] = {};
  // The sealed group's tracker is borrowed by the pool's common state.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Dependency admission gate installed before either thread can read it.
  SubmissionGate gate_;
  // No allocation methods are needed by a capacity notification consumer.
  iree_hal_pool_vtable_t pool_vtable_ = {};
  // Common pool capture with the same local source also present in backing.
  iree_hal_pool_t pool_ = {};
  // Reused cold helper under test.
  iree_hal_pool_wait_t* wait_ = nullptr;
};

TEST_F(PoolWaitTest, AbortReleasesAllObservations) {
  for (int i = 0; i < 16; ++i) {
    iree_hal_pool_wait_prepare(wait_);
    for (auto* notification : notifications_) {
      EXPECT_EQ(iree_atomic_load(&notification->observer_count,
                                 iree_memory_order_acquire),
                1);
    }
    iree_hal_pool_wait_abort(wait_);
    ExpectNoObservers();
  }
}

TEST_F(PoolWaitTest, SignalBetweenObservationAndCommitIsNotLost) {
  for (auto* notification : notifications_) {
    Completion completion;
    iree_hal_pool_wait_prepare(wait_);
    iree_async_notification_signal_if_observed(notification, INT32_MAX);
    iree_hal_pool_wait_commit(wait_, iree_infinite_timeout(),
                              completion.callback());
    completion.Await(IREE_STATUS_OK);
    ExpectNoObservers();
  }
}

TEST_F(PoolWaitTest, EitherOwnerWakesAndJoinsTheOther) {
  for (auto* notification : notifications_) {
    Completion completion;
    iree_hal_pool_wait_prepare(wait_);
    iree_hal_pool_wait_commit(wait_, iree_infinite_timeout(),
                              completion.callback());
    iree_async_notification_signal_if_observed(notification, INT32_MAX);
    completion.Await(IREE_STATUS_OK);
    ExpectNoObservers();
  }
}

TEST_F(PoolWaitTest, CancellationDoesNotSignalAnUnrelatedWaiter) {
  iree_hal_pool_wait_t* other = nullptr;
  IREE_ASSERT_OK(
      iree_hal_pool_wait_allocate(&pool_, iree_allocator_system(), &other));
  Completion cancelled;
  Completion survivor;
  iree_hal_pool_wait_prepare(other);
  iree_hal_pool_wait_commit(other, iree_infinite_timeout(),
                            survivor.callback());
  iree_hal_pool_wait_prepare(wait_);
  iree_hal_pool_wait_commit(wait_, iree_infinite_timeout(),
                            cancelled.callback());
  uint32_t tokens[2] = {
      iree_async_notification_query_epoch(notifications_[0]),
      iree_async_notification_query_epoch(notifications_[1]),
  };
  iree_hal_pool_wait_cancel(wait_);
  cancelled.Await(IREE_STATUS_CANCELLED);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(iree_async_notification_query_epoch(notifications_[i]),
              tokens[i]);
  }
  EXPECT_EQ(survivor.count.load(std::memory_order_acquire), 0);
  iree_async_notification_signal(notifications_[1], INT32_MAX);
  survivor.Await(IREE_STATUS_OK);
  iree_hal_pool_wait_free(other);
  ExpectNoObservers();
}

TEST_F(PoolWaitTest, FailedAdmissionJoinsAlreadySubmittedSources) {
  gate_.reject_waits.store(true, std::memory_order_release);
  Completion rejected;
  iree_hal_pool_wait_prepare(wait_);
  iree_hal_pool_wait_commit(wait_, iree_infinite_timeout(),
                            rejected.callback());
  rejected.Await(IREE_STATUS_RESOURCE_EXHAUSTED);
  ExpectNoObservers();
  gate_.reject_waits.store(false, std::memory_order_release);
  Completion accepted;
  iree_hal_pool_wait_prepare(wait_);
  iree_hal_pool_wait_commit(wait_, iree_infinite_timeout(),
                            accepted.callback());
  iree_async_notification_signal(notifications_[1], INT32_MAX);
  accepted.Await(IREE_STATUS_OK);
  ExpectNoObservers();
}

TEST_F(PoolWaitTest, DeadlineJoinsEverySource) {
  Completion completion;
  iree_hal_pool_wait_prepare(wait_);
  iree_hal_pool_wait_commit(wait_, iree_make_timeout_ms(1),
                            completion.callback());
  completion.Await(IREE_STATUS_DEADLINE_EXCEEDED);
  ExpectNoObservers();
}

TEST_F(PoolWaitTest, ImmediateDeadlineJoinsEverySource) {
  Completion completion;
  iree_hal_pool_wait_prepare(wait_);
  iree_hal_pool_wait_commit(wait_, iree_immediate_timeout(),
                            completion.callback());
  completion.Await(IREE_STATUS_DEADLINE_EXCEEDED);
  ExpectNoObservers();
}

TEST_F(PoolWaitTest, WakingBeforeDeadlineJoinsTimerIdentityForReuse) {
  for (int i = 0; i < 64; ++i) {
    Completion completion;
    iree_hal_pool_wait_prepare(wait_);
    iree_hal_pool_wait_commit(wait_,
                              iree_make_deadline(IREE_TIME_INFINITE_FUTURE - 1),
                              completion.callback());
    // NOPs execute in submission order on this owner, so the timer's admission
    // control callback has run before the independent capacity owner wakes us.
    DrainOwner(0);
    iree_async_notification_signal(notifications_[1], INT32_MAX);
    completion.Await(IREE_STATUS_OK);
    ExpectNoObservers();
  }
}

TEST_F(PoolWaitTest, CancellationJoinsFiniteDeadline) {
  Completion completion;
  iree_hal_pool_wait_prepare(wait_);
  iree_hal_pool_wait_commit(wait_,
                            iree_make_deadline(IREE_TIME_INFINITE_FUTURE - 1),
                            completion.callback());
  DrainOwner(0);
  iree_hal_pool_wait_cancel(wait_);
  completion.Await(IREE_STATUS_CANCELLED);
  ExpectNoObservers();
}

TEST_F(PoolWaitTest, TerminalCallbackCanDestroyHelper) {
  struct State {
    // Helper ownership is handed back to this terminal callback.
    iree_hal_pool_wait_t** wait;
    // Completion witness outlives the helper.
    Completion completion;
  } state{&wait_};
  iree_hal_pool_wait_prepare(wait_);
  iree_hal_pool_wait_commit(wait_, iree_infinite_timeout(),
                            {[](void* user_data, iree_status_t status) {
                               auto* state = static_cast<State*>(user_data);
                               iree_hal_pool_wait_free(*state->wait);
                               *state->wait = nullptr;
                               Completion::Complete(&state->completion, status);
                             },
                             &state});
  iree_async_notification_signal(notifications_[1], INT32_MAX);
  state.completion.Await(IREE_STATUS_OK);
  EXPECT_EQ(wait_, nullptr);
  ExpectNoObservers();
}

TEST_F(PoolWaitTest, TerminalCallbackCanRearmHelper) {
  struct State {
    // Shared helper reused only at its terminal callback handoff.
    iree_hal_pool_wait_t* wait;
    // Capacity source on the other progress owner.
    iree_async_notification_t* notification;
    // Sequential rounds completed by the callback chain.
    int round = 0;
    // Witness published after the last round or an unexpected failure.
    Completion completion;

    static void Complete(void* user_data, iree_status_t status) {
      auto* state = static_cast<State*>(user_data);
      if (!iree_status_is_ok(status) || ++state->round == 64) {
        Completion::Complete(&state->completion, status);
        return;
      }
      iree_hal_pool_wait_prepare(state->wait);
      iree_async_notification_signal(state->notification, INT32_MAX);
      // Commit may hand back ownership before returning. There are no further
      // accesses to the state after starting the next round.
      iree_hal_pool_wait_commit(state->wait, iree_infinite_timeout(),
                                {Complete, state});
    }
  } state{wait_, notifications_[1]};
  iree_hal_pool_wait_prepare(wait_);
  iree_hal_pool_wait_commit(wait_, iree_infinite_timeout(),
                            {State::Complete, &state});
  iree_async_notification_signal(notifications_[1], INT32_MAX);
  state.completion.Await(IREE_STATUS_OK);
  EXPECT_EQ(state.round, 64);
  ExpectNoObservers();
}

}  // namespace
