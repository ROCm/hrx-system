// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/ipc_event_monitor.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>

#include "iree/hal/drivers/amdgpu/ipc_event_monitor_test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class Completion {
 public:
  void Signal() {
    std::lock_guard<std::mutex> lock(mutex_);
    signaled_ = true;
    condition_.notify_all();
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return signaled_; });
  }

 private:
  // Serializes completion publication and waiting.
  std::mutex mutex_;
  // Wakes waiters after completion publication.
  std::condition_variable condition_;
  // True after completion has been published.
  bool signaled_ = false;
};

class Gate {
 public:
  void EnterAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    condition_.notify_all();
    condition_.wait(lock, [this] { return may_continue_; });
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return entered_; });
  }

  void Continue() {
    std::lock_guard<std::mutex> lock(mutex_);
    may_continue_ = true;
    condition_.notify_all();
  }

 private:
  // Serializes gate entry and release.
  std::mutex mutex_;
  // Wakes either side after the corresponding state transition.
  std::condition_variable condition_;
  // True after the worker has entered the gate.
  bool entered_ = false;
  // True after the test permits the worker to continue.
  bool may_continue_ = false;
};

struct PollOperation {
  // Intrusive monitor operation; must remain the first field.
  iree_hal_amdgpu_ipc_event_monitor_operation_t operation;
  // Becomes true when an ordinary poll should complete the operation.
  std::atomic<bool> ready;
  // Completes on this poll count, or zero to use only |ready|.
  size_t complete_after_poll_count;
  // Counts all worker polls of the operation.
  std::atomic<size_t> poll_count;
  // Counts polls carrying an external readiness request.
  std::atomic<size_t> requested_count;
  // Counts polls that completed through the shutdown path.
  std::atomic<size_t> shutdown_count;
  // Largest per-operation retry delay observed by the poll routine.
  std::atomic<iree_duration_t> maximum_observed_poll_delay_ns;
  // Ensures |first_poll| is signaled at most once.
  std::atomic<bool> first_poll_signaled;
  // Optional notification that the worker has polled at least once.
  Completion* first_poll;
  // Optional notification that the operation has retired.
  Completion* retired;
  // Worker gate entered on |block_on_poll_count|, or NULL for no gate.
  Gate* poll_gate;
  // Poll invocation that enters |poll_gate|, or zero for no gate.
  size_t block_on_poll_count;
  // Optional gate entered after terminal readiness is published.
  Gate* terminal_poll_gate;
};

static bool PollTestOperation(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool was_requested) {
  auto* poll_operation = reinterpret_cast<PollOperation*>(operation);
  const size_t poll_count =
      poll_operation->poll_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (was_requested) {
    poll_operation->requested_count.fetch_add(1, std::memory_order_relaxed);
  }
  iree_duration_t observed_max =
      poll_operation->maximum_observed_poll_delay_ns.load(
          std::memory_order_relaxed);
  while (observed_max < operation->current_poll_delay_ns &&
         !poll_operation->maximum_observed_poll_delay_ns.compare_exchange_weak(
             observed_max, operation->current_poll_delay_ns,
             std::memory_order_relaxed)) {
  }
  if (poll_operation->first_poll &&
      !poll_operation->first_poll_signaled.exchange(
          true, std::memory_order_acq_rel)) {
    poll_operation->first_poll->Signal();
  }
  if (poll_operation->poll_gate &&
      poll_count == poll_operation->block_on_poll_count) {
    poll_operation->poll_gate->EnterAndWait();
  }

  const bool normally_ready =
      poll_operation->ready.load(std::memory_order_acquire) ||
      (poll_operation->complete_after_poll_count != 0 &&
       poll_count >= poll_operation->complete_after_poll_count);
  if (!normally_ready && !is_shutting_down) return false;
  if (!normally_ready) {
    poll_operation->shutdown_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (poll_operation->terminal_poll_gate) {
    poll_operation->terminal_poll_gate->EnterAndWait();
  }
  // This must be the callback's final access. The monitor permits a completed
  // poll to free its containing operation as soon as the callback returns.
  if (poll_operation->retired) poll_operation->retired->Signal();
  return true;
}

static void InitializePollOperation(
    Completion* first_poll, Completion* retired, PollOperation* out_operation,
    iree_duration_t maximum_poll_delay_ns =
        IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_EXTERNAL_MAX_POLL_DELAY_NS,
    size_t complete_after_poll_count = 0) {
  iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
      PollTestOperation, maximum_poll_delay_ns, &out_operation->operation);
  out_operation->ready.store(false, std::memory_order_relaxed);
  out_operation->complete_after_poll_count = complete_after_poll_count;
  out_operation->poll_count.store(0, std::memory_order_relaxed);
  out_operation->requested_count.store(0, std::memory_order_relaxed);
  out_operation->shutdown_count.store(0, std::memory_order_relaxed);
  out_operation->maximum_observed_poll_delay_ns.store(
      0, std::memory_order_relaxed);
  out_operation->first_poll_signaled.store(false, std::memory_order_relaxed);
  out_operation->first_poll = first_poll;
  out_operation->retired = retired;
  out_operation->poll_gate = nullptr;
  out_operation->block_on_poll_count = 0;
  out_operation->terminal_poll_gate = nullptr;
}

struct ImmediateFloodOperation {
  // Intrusive monitor operation; must remain the first field.
  iree_hal_amdgpu_ipc_event_monitor_operation_t operation;
  // Next pre-admitted operation published before this poll returns.
  ImmediateFloodOperation* next;
  // Counts immediate operations polled by the worker.
  std::atomic<size_t>* poll_count;
  // Signals after the final pre-admitted operation retires.
  Completion* drained;
};

static bool PollImmediateFloodOperation(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool was_requested) {
  auto* flood_operation = reinterpret_cast<ImmediateFloodOperation*>(operation);
  (void)is_shutting_down;
  (void)was_requested;
  flood_operation->poll_count->fetch_add(1, std::memory_order_relaxed);
  if (flood_operation->next) {
    // Keep the immediate FIFO nonempty before retiring this node. All chain
    // nodes were admitted before the first publication.
    iree_hal_amdgpu_ipc_event_monitor_operation_publish(
        &flood_operation->next->operation);
  } else {
    flood_operation->drained->Signal();
  }
  return true;
}

struct DueUnderFloodOperation {
  // Intrusive monitor operation; must remain the first field.
  iree_hal_amdgpu_ipc_event_monitor_operation_t operation;
  // Proves the first poll occurred before its finite retry is forced due.
  Completion first_poll;
  // Holds the first poll until the test has published a reschedule barrier.
  Gate* first_poll_gate;
  // Signals when the due retry retires the operation.
  Completion retired;
  // Counts all polls; the second poll is the forced-due retry.
  std::atomic<size_t> poll_count;
  // Shared immediate-poll count sampled on retirement.
  std::atomic<size_t>* immediate_poll_count;
  // Deterministic scheduler-node visit count sampled on retirement.
  std::atomic<int64_t> visits_at_retirement;
  // Immediate polls observed when the due retry was selected.
  std::atomic<size_t> immediate_polls_at_retirement;
};

static bool PollDueUnderFloodOperation(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool was_requested) {
  auto* due_operation = reinterpret_cast<DueUnderFloodOperation*>(operation);
  (void)is_shutting_down;
  (void)was_requested;
  const size_t poll_count =
      due_operation->poll_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (poll_count == 1) {
    due_operation->first_poll.Signal();
    due_operation->first_poll_gate->EnterAndWait();
    return false;
  }
  due_operation->immediate_polls_at_retirement.store(
      due_operation->immediate_poll_count->load(std::memory_order_acquire),
      std::memory_order_release);
  due_operation->visits_at_retirement.store(
      iree_hal_amdgpu_ipc_event_monitor_schedule_node_visit_count_for_testing(),
      std::memory_order_release);
  due_operation->retired.Signal();
  return true;
}

TEST(IpcEventMonitorTest, CallbackRequestBeforePublicationIsObserved) {
  Completion retired;
  PollOperation operation;
  InitializePollOperation(nullptr, &retired, &operation,
                          IREE_DURATION_INFINITE);
  operation.ready.store(true, std::memory_order_release);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation));

  // Models a semaphore timepoint callback that fires synchronously during
  // transaction preparation. Publication later transfers the requested
  // operation to the worker without losing readiness.
  iree_hal_amdgpu_ipc_event_monitor_operation_request_poll(
      &operation.operation);
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&operation.operation);
  retired.Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  EXPECT_EQ(1u, operation.poll_count.load(std::memory_order_acquire));
  EXPECT_EQ(1u, operation.requested_count.load(std::memory_order_acquire));
}

TEST(IpcEventMonitorTest, PollsUntilReadyWithoutExternalCallback) {
  Completion first_poll;
  Completion retired;
  PollOperation operation;
  InitializePollOperation(&first_poll, &retired, &operation);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&operation.operation);

  first_poll.Wait();
  EXPECT_EQ(0u, operation.shutdown_count.load(std::memory_order_relaxed));
  operation.ready.store(true, std::memory_order_release);
  retired.Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  EXPECT_GE(operation.poll_count.load(std::memory_order_acquire), 2u);
}

TEST(IpcEventMonitorTest, RequestedPollWakesNotificationDrivenOperation) {
  Completion first_poll;
  Completion retired;
  PollOperation operation;
  InitializePollOperation(&first_poll, &retired, &operation,
                          IREE_DURATION_INFINITE);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&operation.operation);
  first_poll.Wait();

  operation.ready.store(true, std::memory_order_release);
  iree_hal_amdgpu_ipc_event_monitor_operation_request_poll(
      &operation.operation);
  retired.Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  EXPECT_EQ(2u, operation.poll_count.load(std::memory_order_acquire));
  EXPECT_EQ(1u, operation.requested_count.load(std::memory_order_acquire));
}

struct PollingRequestRaceOperation {
  // Intrusive monitor operation; must remain the first field.
  iree_hal_amdgpu_ipc_event_monitor_operation_t operation;
  // Blocks the first poll while an external request races POLLING.
  Gate first_poll_gate;
  // Signals after the requested second poll retires the operation.
  Completion retired;
  // Counts worker polls.
  std::atomic<size_t> poll_count;
};

static bool PollRequestRaceOperation(
    iree_hal_amdgpu_ipc_event_monitor_operation_t* operation,
    bool is_shutting_down, bool was_requested) {
  auto* race_operation =
      reinterpret_cast<PollingRequestRaceOperation*>(operation);
  const size_t poll_count =
      race_operation->poll_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (poll_count == 1) {
    race_operation->first_poll_gate.EnterAndWait();
    return false;
  }
  if (was_requested || is_shutting_down) {
    race_operation->retired.Signal();
    return true;
  }
  return false;
}

TEST(IpcEventMonitorTest, CallbackRequestRacingPollForcesImmediateRetry) {
  PollingRequestRaceOperation operation;
  operation.poll_count.store(0, std::memory_order_relaxed);
  iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
      PollRequestRaceOperation, IREE_DURATION_INFINITE, &operation.operation);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&operation.operation);
  operation.first_poll_gate.WaitUntilEntered();

  iree_hal_amdgpu_ipc_event_monitor_operation_request_poll(
      &operation.operation);
  operation.first_poll_gate.Continue();
  operation.retired.Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  EXPECT_EQ(2u, operation.poll_count.load(std::memory_order_acquire));
}

TEST(IpcEventMonitorTest, NewPublicationDoesNotPollDormantOperation) {
  Completion dormant_first_poll;
  Completion dormant_retired;
  PollOperation dormant_operation;
  InitializePollOperation(&dormant_first_poll, &dormant_retired,
                          &dormant_operation, IREE_DURATION_INFINITE);
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
      &dormant_operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(
      &dormant_operation.operation);
  dormant_first_poll.Wait();

  // Each new publication wakes and exercises the scheduler. None may query or
  // reset a notification-driven operation whose own deadline is dormant.
  for (size_t i = 0; i < 32; ++i) {
    Completion sentinel_retired;
    PollOperation sentinel_operation;
    InitializePollOperation(nullptr, &sentinel_retired, &sentinel_operation,
                            IREE_DURATION_INFINITE);
    sentinel_operation.ready.store(true, std::memory_order_relaxed);
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
        &sentinel_operation.operation));
    iree_hal_amdgpu_ipc_event_monitor_operation_publish(
        &sentinel_operation.operation);
    sentinel_retired.Wait();
    EXPECT_EQ(1u, dormant_operation.poll_count.load(std::memory_order_acquire));
  }

  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  dormant_retired.Wait();
  EXPECT_EQ(2u, dormant_operation.poll_count.load(std::memory_order_acquire));
  EXPECT_EQ(1u,
            dormant_operation.shutdown_count.load(std::memory_order_acquire));
}

TEST(IpcEventMonitorTest, DormantFloodHasLinearSchedulerNodeVisits) {
  constexpr size_t kDormantCount = 128;
  constexpr size_t kSentinelCount = 256;
  auto dormant_operations = std::make_unique<PollOperation[]>(kDormantCount);
  auto dormant_first_polls = std::make_unique<Completion[]>(kDormantCount);
  for (size_t i = 0; i < kDormantCount; ++i) {
    InitializePollOperation(&dormant_first_polls[i], nullptr,
                            &dormant_operations[i], IREE_DURATION_INFINITE);
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
        &dormant_operations[i].operation));
    iree_hal_amdgpu_ipc_event_monitor_operation_publish(
        &dormant_operations[i].operation);
  }
  for (size_t i = 0; i < kDormantCount; ++i) {
    dormant_first_polls[i].Wait();
  }

  // All node visits through the first dormant poll precede its completion
  // signal. Resetting here makes any revisit caused by sentinel publication
  // directly visible instead of hiding it behind poll-count assertions.
  iree_hal_amdgpu_ipc_event_monitor_reset_schedule_node_visit_count_for_testing();
  for (size_t i = 0; i < kSentinelCount; ++i) {
    Completion retired;
    PollOperation sentinel;
    InitializePollOperation(nullptr, &retired, &sentinel,
                            IREE_DURATION_INFINITE);
    sentinel.ready.store(true, std::memory_order_relaxed);
    IREE_ASSERT_OK(
        iree_hal_amdgpu_ipc_event_monitor_operation_admit(&sentinel.operation));
    iree_hal_amdgpu_ipc_event_monitor_operation_publish(&sentinel.operation);
    retired.Wait();
  }

  const int64_t visits_before_shutdown =
      iree_hal_amdgpu_ipc_event_monitor_schedule_node_visit_count_for_testing();
  EXPECT_LE(visits_before_shutdown, static_cast<int64_t>(2 * kSentinelCount));
  for (size_t i = 0; i < kDormantCount; ++i) {
    EXPECT_EQ(1u,
              dormant_operations[i].poll_count.load(std::memory_order_acquire));
  }

  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  const int64_t total_visits =
      iree_hal_amdgpu_ipc_event_monitor_schedule_node_visit_count_for_testing();
  EXPECT_LE(total_visits,
            static_cast<int64_t>(2 * kSentinelCount + 2 * kDormantCount));
}

TEST(IpcEventMonitorTest, ExternalPollingBackoffIsCappedPerOperation) {
  Completion retired;
  PollOperation operation;
  InitializePollOperation(
      nullptr, &retired, &operation,
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_EXTERNAL_MAX_POLL_DELAY_NS,
      /*complete_after_poll_count=*/8);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&operation.operation);
  retired.Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();

  EXPECT_EQ(8u, operation.poll_count.load(std::memory_order_acquire));
  EXPECT_EQ(
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_EXTERNAL_MAX_POLL_DELAY_NS,
      operation.maximum_observed_poll_delay_ns.load(std::memory_order_acquire));
}

TEST(IpcEventMonitorTest, ExpiredFiniteBatchDoesNotStarveNewPublications) {
  constexpr size_t kFiniteCount = 64;
  constexpr size_t kSentinelCount = 64;

  // First settle a large notification-driven batch in the dormant list. The
  // test hook can then make the entire batch due at a stable worker boundary
  // without depending on a short retry deadline under instrumentation.
  auto finite_operations = std::make_unique<PollOperation[]>(kFiniteCount);
  auto finite_first_polls = std::make_unique<Completion[]>(kFiniteCount);
  for (size_t i = 0; i < kFiniteCount; ++i) {
    InitializePollOperation(&finite_first_polls[i], nullptr,
                            &finite_operations[i], IREE_DURATION_INFINITE);
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
        &finite_operations[i].operation));
    iree_hal_amdgpu_ipc_event_monitor_operation_publish(
        &finite_operations[i].operation);
  }
  for (size_t i = 0; i < kFiniteCount; ++i) {
    finite_first_polls[i].Wait();
  }

  Gate barrier_gate;
  Completion barrier_retired;
  PollOperation barrier;
  InitializePollOperation(nullptr, &barrier_retired, &barrier,
                          IREE_DURATION_INFINITE);
  barrier.poll_gate = &barrier_gate;
  barrier.block_on_poll_count = 1;
  barrier.ready.store(true, std::memory_order_relaxed);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&barrier.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&barrier.operation);
  barrier_gate.WaitUntilEntered();

  EXPECT_EQ(
      kFiniteCount,
      iree_hal_amdgpu_ipc_event_monitor_make_dormant_operations_due_for_testing());

  // A newly published ready operation must run before any member of the due
  // batch is retried.
  auto sentinels = std::make_unique<PollOperation[]>(kSentinelCount);
  auto sentinel_retired = std::make_unique<Completion[]>(kSentinelCount);
  Gate first_sentinel_gate;
  iree_hal_amdgpu_ipc_event_monitor_reset_schedule_node_visit_count_for_testing();
  for (size_t i = 0; i < kSentinelCount; ++i) {
    InitializePollOperation(nullptr, &sentinel_retired[i], &sentinels[i],
                            IREE_DURATION_INFINITE);
    if (i == 0) {
      sentinels[i].terminal_poll_gate = &first_sentinel_gate;
    }
    sentinels[i].ready.store(true, std::memory_order_relaxed);
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
        &sentinels[i].operation));
    iree_hal_amdgpu_ipc_event_monitor_operation_publish(
        &sentinels[i].operation);
  }
  barrier_gate.Continue();
  barrier_retired.Wait();
  first_sentinel_gate.WaitUntilEntered();

  for (size_t i = 0; i < kFiniteCount; ++i) {
    EXPECT_EQ(1u,
              finite_operations[i].poll_count.load(std::memory_order_acquire));
  }
  EXPECT_LE(
      iree_hal_amdgpu_ipc_event_monitor_schedule_node_visit_count_for_testing(),
      static_cast<int64_t>(kSentinelCount + 1));
  first_sentinel_gate.Continue();
  sentinel_retired[0].Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
}

TEST(IpcEventMonitorTest, ContinuousImmediateFloodCannotStarveDueOperation) {
  constexpr size_t kImmediateServiceQuota = 4;
  constexpr size_t kFloodCount = 64;

  std::atomic<size_t> immediate_poll_count{0};
  Gate due_first_poll_gate;
  DueUnderFloodOperation due_operation;
  due_operation.poll_count.store(0, std::memory_order_relaxed);
  due_operation.first_poll_gate = &due_first_poll_gate;
  due_operation.immediate_poll_count = &immediate_poll_count;
  due_operation.visits_at_retirement.store(0, std::memory_order_relaxed);
  due_operation.immediate_polls_at_retirement.store(0,
                                                    std::memory_order_relaxed);
  iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
      PollDueUnderFloodOperation,
      IREE_HAL_AMDGPU_IPC_EVENT_MONITOR_EXTERNAL_MAX_POLL_DELAY_NS,
      &due_operation.operation);
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
      &due_operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&due_operation.operation);
  due_first_poll_gate.WaitUntilEntered();

  Gate reschedule_barrier_gate;
  Completion reschedule_barrier_retired;
  PollOperation reschedule_barrier;
  InitializePollOperation(nullptr, &reschedule_barrier_retired,
                          &reschedule_barrier, IREE_DURATION_INFINITE);
  reschedule_barrier.poll_gate = &reschedule_barrier_gate;
  reschedule_barrier.block_on_poll_count = 1;
  reschedule_barrier.ready.store(true, std::memory_order_relaxed);
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
      &reschedule_barrier.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(
      &reschedule_barrier.operation);
  // Returning from the gated first poll reschedules the finite operation
  // before the worker can activate and enter the immediate barrier.
  due_first_poll_gate.Continue();
  reschedule_barrier_gate.WaitUntilEntered();

  Completion flood_drained;
  auto flood_operations =
      std::make_unique<ImmediateFloodOperation[]>(kFloodCount);
  for (size_t i = 0; i < kFloodCount; ++i) {
    iree_hal_amdgpu_ipc_event_monitor_operation_initialize(
        PollImmediateFloodOperation, IREE_DURATION_INFINITE,
        &flood_operations[i].operation);
    flood_operations[i].next =
        i + 1 < kFloodCount ? &flood_operations[i + 1] : nullptr;
    flood_operations[i].poll_count = &immediate_poll_count;
    flood_operations[i].drained = &flood_drained;
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
        &flood_operations[i].operation));
  }

  // Reset at a stable worker boundary, force the finite retry due, and start a
  // chain that replenishes the immediate queue before each poll returns.
  iree_hal_amdgpu_ipc_event_monitor_reset_schedule_node_visit_count_for_testing();
  iree_hal_amdgpu_ipc_event_monitor_make_finite_operations_due_for_testing();
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(
      &flood_operations[0].operation);
  reschedule_barrier_gate.Continue();
  reschedule_barrier_retired.Wait();
  due_operation.retired.Wait();

  // Selection must reach the due retry within one fixed immediate quota. Each
  // immediate costs one activation visit plus one selection visit, and the due
  // candidate costs one more selection visit.
  EXPECT_LE(due_operation.immediate_polls_at_retirement.load(
                std::memory_order_acquire),
            kImmediateServiceQuota);
  EXPECT_LE(due_operation.visits_at_retirement.load(std::memory_order_acquire),
            static_cast<int64_t>(2 * kImmediateServiceQuota + 1));

  flood_drained.Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
}

TEST(IpcEventMonitorTest,
     UnpublishedCancellationDoesNotWakeWorkerOutsideShutdown) {
  Completion first_poll;
  Completion retired;
  PollOperation dormant_operation;
  InitializePollOperation(&first_poll, &retired, &dormant_operation,
                          IREE_DURATION_INFINITE);
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
      &dormant_operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(
      &dormant_operation.operation);
  first_poll.Wait();

  iree_hal_amdgpu_ipc_event_monitor_reset_cancellation_wake_count_for_testing();
  constexpr size_t kCancellationCount = 64;
  std::array<PollOperation, kCancellationCount> cancelled_operations;
  for (PollOperation& operation : cancelled_operations) {
    InitializePollOperation(nullptr, nullptr, &operation);
    IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
        &operation.operation));
    iree_hal_amdgpu_ipc_event_monitor_operation_cancel(&operation.operation);
  }
  EXPECT_EQ(
      0,
      iree_hal_amdgpu_ipc_event_monitor_cancellation_wake_count_for_testing());

  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  retired.Wait();
}

TEST(IpcEventMonitorTest, PermanentlyPendingOperationCancelsOnShutdown) {
  Completion first_poll;
  Completion retired;
  PollOperation operation;
  InitializePollOperation(&first_poll, &retired, &operation);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&operation.operation);

  first_poll.Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  retired.Wait();
  EXPECT_EQ(1u, operation.shutdown_count.load(std::memory_order_acquire));
}

TEST(IpcEventMonitorTest, ConcurrentPublishAndPoll) {
  constexpr size_t kProducerCount = 8;
  constexpr size_t kOperationCountPerProducer = 128;
  constexpr size_t kOperationCount =
      kProducerCount * kOperationCountPerProducer;

  auto operations = std::make_unique<PollOperation[]>(kOperationCount);
  size_t admitted_count = 0;
  for (; admitted_count < kOperationCount; ++admitted_count) {
    InitializePollOperation(nullptr, nullptr, &operations[admitted_count]);
    operations[admitted_count].ready.store(true, std::memory_order_relaxed);
    iree_status_t status = iree_hal_amdgpu_ipc_event_monitor_operation_admit(
        &operations[admitted_count].operation);
    if (!iree_status_is_ok(status)) {
      for (size_t i = 0; i < admitted_count; ++i) {
        iree_hal_amdgpu_ipc_event_monitor_operation_cancel(
            &operations[i].operation);
      }
      iree_hal_amdgpu_ipc_event_monitor_shutdown();
      IREE_ASSERT_OK(status);
      return;
    }
  }

  std::mutex start_mutex;
  std::condition_variable start_condition;
  size_t ready_count = 0;
  bool start = false;
  std::array<std::thread, kProducerCount> producers;
  for (size_t producer_index = 0; producer_index < kProducerCount;
       ++producer_index) {
    producers[producer_index] = std::thread([&, producer_index] {
      {
        std::unique_lock<std::mutex> lock(start_mutex);
        ++ready_count;
        start_condition.notify_all();
        start_condition.wait(lock, [&] { return start; });
      }
      const size_t begin = producer_index * kOperationCountPerProducer;
      const size_t end = begin + kOperationCountPerProducer;
      for (size_t i = begin; i < end; ++i) {
        iree_hal_amdgpu_ipc_event_monitor_operation_publish(
            &operations[i].operation);
      }
    });
  }
  {
    std::unique_lock<std::mutex> lock(start_mutex);
    start_condition.wait(lock, [&] { return ready_count == kProducerCount; });
    start = true;
    start_condition.notify_all();
  }
  for (std::thread& producer : producers) producer.join();

  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  size_t total_poll_count = 0;
  for (size_t i = 0; i < kOperationCount; ++i) {
    total_poll_count +=
        operations[i].poll_count.load(std::memory_order_acquire);
  }
  EXPECT_EQ(kOperationCount, total_poll_count);
}

TEST(IpcEventMonitorTest, ThreadCreationFailureRollsBackAndCanRetry) {
  PollOperation operation;
  InitializePollOperation(nullptr, nullptr, &operation);

  iree_hal_amdgpu_ipc_event_monitor_set_fail_thread_create_for_testing(true);
  iree_status_t status =
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_free(status);

  iree_hal_amdgpu_ipc_event_monitor_set_fail_thread_create_for_testing(false);
  operation.ready.store(true, std::memory_order_relaxed);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&operation.operation);
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  EXPECT_EQ(1u, operation.poll_count.load(std::memory_order_acquire));
}

TEST(IpcEventMonitorTest, ShutdownRacingCommitClosesAdmissionAndCanRestart) {
  Completion retired;
  PollOperation operation;
  InitializePollOperation(nullptr, &retired, &operation);
  IREE_ASSERT_OK(
      iree_hal_amdgpu_ipc_event_monitor_operation_admit(&operation.operation));

  std::thread shutdown_thread(
      [] { iree_hal_amdgpu_ipc_event_monitor_shutdown(); });

  // The first rejected admission proves shutdown has closed this worker
  // generation. Admissions linearized before closure remain publishable.
  for (;;) {
    PollOperation candidate;
    InitializePollOperation(nullptr, nullptr, &candidate);
    iree_status_t status =
        iree_hal_amdgpu_ipc_event_monitor_operation_admit(&candidate.operation);
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_ipc_event_monitor_operation_cancel(&candidate.operation);
      std::this_thread::yield();
      continue;
    }
    EXPECT_EQ(IREE_STATUS_UNAVAILABLE, iree_status_code(status));
    iree_status_free(status);
    break;
  }

  iree_hal_amdgpu_ipc_event_monitor_operation_publish(&operation.operation);
  shutdown_thread.join();
  retired.Wait();
  EXPECT_EQ(1u, operation.shutdown_count.load(std::memory_order_acquire));

  Completion restarted_retired;
  PollOperation restarted_operation;
  InitializePollOperation(nullptr, &restarted_retired, &restarted_operation);
  restarted_operation.ready.store(true, std::memory_order_relaxed);
  IREE_ASSERT_OK(iree_hal_amdgpu_ipc_event_monitor_operation_admit(
      &restarted_operation.operation));
  iree_hal_amdgpu_ipc_event_monitor_operation_publish(
      &restarted_operation.operation);
  restarted_retired.Wait();
  iree_hal_amdgpu_ipc_event_monitor_shutdown();
  EXPECT_EQ(0u,
            restarted_operation.shutdown_count.load(std::memory_order_acquire));
}

}  // namespace
}  // namespace iree::hal::amdgpu
