// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/owner.h"

#include <array>
#include <atomic>
#include <thread>
#include <type_traits>

#include "iree/base/threading/notification.h"
#include "iree/task/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class OwnerTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(GetParam(), &topology);
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    auto status = iree_task_executor_create(
        options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    iree_notification_initialize(&notification_);
    iree_task_owner_initialize(executor_, Advance, this, &owner_);
  }

  void TearDown() override {
    if (!executor_) {
      return;
    }
    Call([&] { stopping_ = true; });
    iree_task_owner_await(&owner_);
    iree_task_owner_deinitialize(&owner_);
    iree_notification_deinitialize(&notification_);
    iree_task_executor_release(executor_);
  }

  template <typename Fn>
  void Call(Fn&& callback) {
    using Callback = std::remove_reference_t<Fn>;
    iree_task_owner_call(
        &owner_, [](void* self) { (*static_cast<Callback*>(self))(); },
        &callback);
  }

  void WaitForProcessed(int expected) {
    bool complete = false;
    while (!complete) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      Call([&] { complete = processed_ == expected; });
      if (complete) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  static void Advance(void* self, const iree_task_worker_context_t*,
                      iree_task_process_drain_result_t* result) {
    auto& fixture = *static_cast<OwnerTest*>(self);
    ++fixture.turn_count_;
    if (fixture.pending_) {
      --fixture.pending_;
      ++fixture.processed_;
      result->did_work = true;
      iree_notification_post(&fixture.notification_, IREE_ALL_WAITERS);
    }
    result->completed = fixture.stopping_ && !fixture.pending_;
  }

  // Existing worker pool; its release follows the owner's final release.
  iree_task_executor_t* executor_ = nullptr;
  // Shared serialized access to all non-atomic application fields below.
  iree_task_owner_t owner_;
  // Caller notification for published progress, independent of owner wakes.
  iree_notification_t notification_;
  // Accepted work waiting for a worker turn.
  int pending_ = 0;
  // Work completed under the same exclusion as admission.
  int processed_ = 0;
  // Application callbacks invoked, including idle turns.
  int turn_count_ = 0;
  // Terminal intent; already accepted work still retires first.
  bool stopping_ = false;
};

TEST_P(OwnerTest, ConcurrentCallsAndWorkerTurnsShareOneOwner) {
  constexpr int kCallCount = 256;
  std::array<std::thread, 4> callers;
  for (auto& caller : callers) {
    caller = std::thread([&] {
      for (int i = 0; i < kCallCount; ++i) {
        Call([&] { ++pending_; });
      }
    });
  }
  for (auto& caller : callers) {
    caller.join();
  }
  WaitForProcessed(callers.size() * kCallCount);
  Call([&] {
    EXPECT_EQ(pending_, 0);
    EXPECT_EQ(processed_, callers.size() * kCallCount);
  });
}

TEST_P(OwnerTest, FailedCallStillSchedulesAcceptedWork) {
  struct {
    // Owned call result, consumed by the caller after serialized entry returns.
    iree_status_t status = iree_ok_status();
  } result;
  Call([&] {
    // A partial admission accepted one item before rejecting the next. The
    // caller owns its failure; it cannot abandon accepted asynchronous work.
    ++pending_;
    result.status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  });
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, result.status);
  WaitForProcessed(1);
}

TEST_P(OwnerTest, RetirementJoinsWorkAndLateWakesCannotReenter) {
  Call([&] {
    pending_ = 256;
    stopping_ = true;
  });
  iree_task_owner_await(&owner_);
  EXPECT_EQ(processed_, 256);
  EXPECT_EQ(pending_, 0);
  const int final_turn_count = turn_count_;
  for (int i = 0; i < 256; ++i) {
    iree_task_owner_wake(&owner_);
  }
  Call([&] { EXPECT_EQ(turn_count_, final_turn_count); });
  iree_task_owner_await(&owner_);
}

TEST_P(OwnerTest, InertStorageNeedsNoExecutorActivation) {
  iree_task_owner_t inert;
  iree_task_owner_initialize(executor_, Advance, this, &inert);
  iree_task_owner_deinitialize(&inert);
  EXPECT_EQ(turn_count_, 0);
}

// Observes the actual drain callback without replacing its implementation or
// changing executor scheduling. This supplies an exact contention checkpoint.
struct DrainProbe {
  // Owner whose application call is deliberately paused like a descheduled
  // application thread; executor workers must remain able to yield.
  iree_task_owner_t owner;
  // Original owner implementation invoked by the observation trampoline.
  iree_task_process_drain_fn_t drain;
  // Signals that a worker returned from the contended drain.
  iree_notification_t notification;
  // Release/acquire publication of the worker's first attempted entry.
  std::atomic<bool> attempted{false};
  // Caller publication, read only under owner exclusion.
  bool ready = false;
  // Number of actual application advances, protected by owner exclusion.
  int turn_count = 0;
};

TEST_P(OwnerTest, ContendedWorkerYieldsAndCallExitRestartsIt) {
  DrainProbe probe;
  iree_notification_initialize(&probe.notification);
  iree_task_owner_initialize(
      executor_,
      [](void* self, const iree_task_worker_context_t*,
         iree_task_process_drain_result_t* result) {
        auto& probe = *static_cast<DrainProbe*>(self);
        ++probe.turn_count;
        result->completed = probe.ready;
      },
      &probe, &probe.owner);
  probe.drain = probe.owner.process.drain;
  probe.owner.process.drain = [](iree_task_process_t* process,
                                 const iree_task_worker_context_t* context,
                                 iree_task_process_drain_result_t* result) {
    auto& probe = *static_cast<DrainProbe*>(process->user_data);
    auto status = probe.drain(process, context, result);
    probe.attempted.store(true, std::memory_order_release);
    iree_notification_post(&probe.notification, IREE_ALL_WAITERS);
    return status;
  };
  iree_task_owner_call(
      &probe.owner,
      [](void* self) {
        auto& probe = *static_cast<DrainProbe*>(self);
        iree_task_owner_wake(&probe.owner);
        // The test pauses only the external thread. A worker blocking on the
        // owner mutex would deadlock here instead of publishing attempted.
        iree_notification_await(
            &probe.notification,
            [](void* self) {
              return static_cast<DrainProbe*>(self)->attempted.load(
                  std::memory_order_acquire);
            },
            &probe, iree_infinite_timeout());
        EXPECT_EQ(probe.turn_count, 0);
        probe.ready = true;
      },
      &probe);
  // No additional wake or caller-side progress can rescue a lost handoff.
  iree_task_owner_await(&probe.owner);
  EXPECT_EQ(probe.turn_count, 1);
  iree_task_owner_deinitialize(&probe.owner);
  iree_notification_deinitialize(&probe.notification);
}

INSTANTIATE_TEST_SUITE_P(WorkerCounts, OwnerTest, ::testing::Values(1, 4));

}  // namespace
