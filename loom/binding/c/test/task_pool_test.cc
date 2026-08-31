// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task_pool.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using TaskPoolPtr = HandlePtr<loomc_task_pool_t, loomc_task_pool_free>;

class TaskTracker {
 public:
  explicit TaskTracker(loomc_host_size_t worker_count)
      : active_by_worker_(worker_count, 0) {}

  void BeginExecution(loomc_host_size_t worker_ordinal) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_ordinal >= active_by_worker_.size()) {
      invalid_worker_ordinal_ = true;
    } else if (active_by_worker_[worker_ordinal]++ != 0) {
      concurrent_worker_ordinal_ = true;
    }
    ++execution_count_;
    condition_.notify_all();
  }

  void EndExecution(loomc_host_size_t worker_ordinal) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_ordinal < active_by_worker_.size()) {
      --active_by_worker_[worker_ordinal];
    }
  }

  void RecordDestruction() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++destruction_count_;
    condition_.notify_all();
  }

  void RecordSubmissionFailure(loomc_status_t status) {
    const loomc_status_code_t code = loomc_status_consume_code(status);
    std::lock_guard<std::mutex> lock(mutex_);
    submission_status_code_ = code;
    condition_.notify_all();
  }

  void WaitForExecutions(int expected_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return execution_count_ >= expected_count; });
  }

  void WaitForDestructions(int expected_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return destruction_count_ >= expected_count; });
  }

  int execution_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return execution_count_;
  }

  int destruction_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return destruction_count_;
  }

  loomc_status_code_t submission_status_code() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return submission_status_code_;
  }

  bool has_invalid_worker_ordinal() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return invalid_worker_ordinal_;
  }

  bool has_concurrent_worker_ordinal() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return concurrent_worker_ordinal_;
  }

 private:
  // Serializes all tracked outcome state.
  mutable std::mutex mutex_;

  // Notifies waiters when task execution or destruction advances.
  std::condition_variable condition_;

  // Active callback count indexed by dense worker ordinal.
  std::vector<int> active_by_worker_;

  // Number of task callbacks that began execution.
  int execution_count_ = 0;

  // Number of task objects destroyed.
  int destruction_count_ = 0;

  // First observed recursive submission failure.
  loomc_status_code_t submission_status_code_ = LOOMC_STATUS_OK;

  // Whether a callback observed an out-of-range worker ordinal.
  bool invalid_worker_ordinal_ = false;

  // Whether two callbacks ran concurrently with one worker ordinal.
  bool concurrent_worker_ordinal_ = false;
};

typedef struct tracked_task_t {
  // Generic task base.
  loomc_task_t base;

  // External outcome tracker that outlives the task.
  TaskTracker* tracker;
} tracked_task_t;

static void ExecuteTrackedTask(loomc_task_t* base_task,
                               loomc_host_size_t worker_ordinal) {
  tracked_task_t* task = reinterpret_cast<tracked_task_t*>(base_task);
  task->tracker->BeginExecution(worker_ordinal);
  task->tracker->EndExecution(worker_ordinal);
}

static void DestroyTrackedTask(loomc_task_t* base_task) {
  tracked_task_t* task = reinterpret_cast<tracked_task_t*>(base_task);
  task->tracker->RecordDestruction();
  delete task;
}

static const loomc_task_vtable_t kTrackedTaskVtable = {
    /*.execute=*/ExecuteTrackedTask,
    /*.destroy=*/DestroyTrackedTask,
};

static loomc_task_t* AllocateTrackedTask(TaskTracker* tracker) {
  tracked_task_t* task = new tracked_task_t;
  loomc_task_initialize(&kTrackedTaskVtable, &task->base);
  task->tracker = tracker;
  return &task->base;
}

class ExecutionGate {
 public:
  explicit ExecutionGate(int expected_count)
      : expected_count_(expected_count) {}

  void EnterAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++entered_count_;
    condition_.notify_all();
    condition_.wait(lock, [&] { return released_; });
  }

  void WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_count_ >= expected_count_; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

 private:
  // Serializes entry and release state.
  std::mutex mutex_;

  // Notifies waiters when entrants arrive or the gate releases.
  std::condition_variable condition_;

  // Number of concurrent entrants required before the observer proceeds.
  int expected_count_ = 0;

  // Number of tasks that have entered the gate.
  int entered_count_ = 0;

  // Whether every task waiting at the gate may proceed.
  bool released_ = false;
};

typedef struct gated_task_t {
  // Generic task base.
  loomc_task_t base;

  // External execution gate that outlives the task.
  ExecutionGate* gate;

  // External outcome tracker that outlives the task.
  TaskTracker* tracker;
} gated_task_t;

static void ExecuteGatedTask(loomc_task_t* base_task,
                             loomc_host_size_t worker_ordinal) {
  gated_task_t* task = reinterpret_cast<gated_task_t*>(base_task);
  task->tracker->BeginExecution(worker_ordinal);
  task->gate->EnterAndWait();
  task->tracker->EndExecution(worker_ordinal);
}

static void DestroyGatedTask(loomc_task_t* base_task) {
  gated_task_t* task = reinterpret_cast<gated_task_t*>(base_task);
  task->tracker->RecordDestruction();
  delete task;
}

static const loomc_task_vtable_t kGatedTaskVtable = {
    /*.execute=*/ExecuteGatedTask,
    /*.destroy=*/DestroyGatedTask,
};

static loomc_task_t* AllocateGatedTask(ExecutionGate* gate,
                                       TaskTracker* tracker) {
  gated_task_t* task = new gated_task_t;
  loomc_task_initialize(&kGatedTaskVtable, &task->base);
  task->gate = gate;
  task->tracker = tracker;
  return &task->base;
}

typedef struct recursive_task_t {
  // Generic task base.
  loomc_task_t base;

  // Borrowed sink used to publish child tasks.
  loomc_task_sink_t sink;

  // External child execution gate that outlives the task.
  ExecutionGate* child_gate;

  // External outcome tracker that outlives the task.
  TaskTracker* tracker;

  // Number of child tasks to publish.
  int child_count;
} recursive_task_t;

static void ExecuteRecursiveTask(loomc_task_t* base_task,
                                 loomc_host_size_t worker_ordinal) {
  recursive_task_t* task = reinterpret_cast<recursive_task_t*>(base_task);
  task->tracker->BeginExecution(worker_ordinal);
  for (int i = 0; i < task->child_count; ++i) {
    loomc_task_t* child = AllocateGatedTask(task->child_gate, task->tracker);
    loomc_status_t status = loomc_task_sink_submit(task->sink, child);
    if (!loomc_status_is_ok(status)) {
      task->tracker->RecordSubmissionFailure(status);
      loomc_task_destroy(child);
    }
  }
  task->tracker->EndExecution(worker_ordinal);
}

static void DestroyRecursiveTask(loomc_task_t* base_task) {
  recursive_task_t* task = reinterpret_cast<recursive_task_t*>(base_task);
  task->tracker->RecordDestruction();
  delete task;
}

static const loomc_task_vtable_t kRecursiveTaskVtable = {
    /*.execute=*/ExecuteRecursiveTask,
    /*.destroy=*/DestroyRecursiveTask,
};

static loomc_task_t* AllocateRecursiveTask(loomc_task_sink_t sink,
                                           ExecutionGate* child_gate,
                                           int child_count,
                                           TaskTracker* tracker) {
  recursive_task_t* task = new recursive_task_t;
  loomc_task_initialize(&kRecursiveTaskVtable, &task->base);
  task->sink = sink;
  task->child_gate = child_gate;
  task->tracker = tracker;
  task->child_count = child_count;
  return &task->base;
}

static TaskPoolPtr AllocateTaskPool(loomc_host_size_t max_worker_count) {
  loomc_task_pool_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS,
      /*.structure_size=*/sizeof(loomc_task_pool_options_t),
      /*.next=*/nullptr,
      /*.max_worker_count=*/max_worker_count,
      /*.worker_stack_size=*/0,
  };
  loomc_task_pool_t* pool = nullptr;
  IREE_CHECK_OK(iree_status_from_loomc(
      loomc_task_pool_allocate(&options, loomc_allocator_system(), &pool)));
  return TaskPoolPtr(pool);
}

TEST(TaskPoolTest, DefaultUsesFourWorkerLimit) {
  loomc_task_pool_t* raw_pool = nullptr;
  LOOMC_ASSERT_OK(loomc_task_pool_allocate(
      /*options=*/nullptr, loomc_allocator_system(), &raw_pool));
  TaskPoolPtr pool(raw_pool);
  EXPECT_GE(loomc_task_pool_worker_count(pool.get()), 1u);
  EXPECT_LE(loomc_task_pool_worker_count(pool.get()), 4u);
}

TEST(TaskPoolTest, DrainsAcceptedTasksWithExclusiveWorkerOrdinals) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  const loomc_host_size_t worker_count =
      loomc_task_pool_worker_count(pool.get());
  TaskTracker tracker(worker_count);
  loomc_task_sink_t sink = loomc_task_pool_sink(pool.get());

  constexpr int kTaskCount = 256;
  for (int i = 0; i < kTaskCount; ++i) {
    LOOMC_ASSERT_OK(
        loomc_task_sink_submit(sink, AllocateTrackedTask(&tracker)));
  }
  LOOMC_ASSERT_OK(loomc_task_pool_shutdown(pool.get()));
  LOOMC_ASSERT_OK(loomc_task_pool_await_shutdown(pool.get()));

  EXPECT_EQ(tracker.execution_count(), kTaskCount);
  EXPECT_EQ(tracker.destruction_count(), kTaskCount);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskPoolTest, ConcurrentSubmittersShareOneWorkerPopulation) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  TaskTracker tracker(loomc_task_pool_worker_count(pool.get()));
  loomc_task_sink_t sink = loomc_task_pool_sink(pool.get());

  constexpr int kSubmitterCount = 8;
  constexpr int kTasksPerSubmitter = 128;
  std::vector<std::thread> submitters;
  submitters.reserve(kSubmitterCount);
  for (int i = 0; i < kSubmitterCount; ++i) {
    submitters.emplace_back([&] {
      for (int j = 0; j < kTasksPerSubmitter; ++j) {
        loomc_task_t* task = AllocateTrackedTask(&tracker);
        loomc_status_t status = loomc_task_sink_submit(sink, task);
        if (!loomc_status_is_ok(status)) {
          tracker.RecordSubmissionFailure(status);
          loomc_task_destroy(task);
        }
      }
    });
  }
  for (std::thread& submitter : submitters) submitter.join();

  LOOMC_ASSERT_OK(loomc_task_pool_shutdown(pool.get()));
  LOOMC_ASSERT_OK(loomc_task_pool_await_shutdown(pool.get()));
  EXPECT_EQ(tracker.submission_status_code(), LOOMC_STATUS_OK);
  EXPECT_EQ(tracker.execution_count(), kSubmitterCount * kTasksPerSubmitter);
  EXPECT_EQ(tracker.destruction_count(), kSubmitterCount * kTasksPerSubmitter);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskPoolTest, ReusesWorkersAcrossSubmissionWaves) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  const loomc_host_size_t worker_count =
      loomc_task_pool_worker_count(pool.get());
  TaskTracker tracker(worker_count);
  loomc_task_sink_t sink = loomc_task_pool_sink(pool.get());

  constexpr int kWaveCount = 256;
  int expected_task_count = 0;
  for (int wave = 0; wave < kWaveCount; ++wave) {
    const int wave_task_count = 1 + wave % static_cast<int>(worker_count);
    for (int i = 0; i < wave_task_count; ++i) {
      LOOMC_ASSERT_OK(
          loomc_task_sink_submit(sink, AllocateTrackedTask(&tracker)));
    }
    expected_task_count += wave_task_count;
    tracker.WaitForDestructions(expected_task_count);
  }

  LOOMC_ASSERT_OK(loomc_task_pool_shutdown(pool.get()));
  LOOMC_ASSERT_OK(loomc_task_pool_await_shutdown(pool.get()));
  EXPECT_EQ(tracker.execution_count(), expected_task_count);
  EXPECT_EQ(tracker.destruction_count(), expected_task_count);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskPoolTest, RecursiveSubmissionUsesMultipleWorkers) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  const int worker_count =
      static_cast<int>(loomc_task_pool_worker_count(pool.get()));
  if (worker_count < 2) GTEST_SKIP() << "host exposes only one worker";

  ExecutionGate child_gate(/*expected_count=*/2);
  TaskTracker tracker(worker_count);
  loomc_task_sink_t sink = loomc_task_pool_sink(pool.get());
  const int child_count = worker_count * 2;
  LOOMC_ASSERT_OK(loomc_task_sink_submit(
      sink, AllocateRecursiveTask(sink, &child_gate, child_count, &tracker)));

  child_gate.WaitUntilEntered();
  child_gate.Release();
  tracker.WaitForExecutions(/*expected_count=*/child_count + 1);
  LOOMC_ASSERT_OK(loomc_task_pool_shutdown(pool.get()));
  LOOMC_ASSERT_OK(loomc_task_pool_await_shutdown(pool.get()));

  EXPECT_EQ(tracker.submission_status_code(), LOOMC_STATUS_OK);
  EXPECT_EQ(tracker.execution_count(), child_count + 1);
  EXPECT_EQ(tracker.destruction_count(), child_count + 1);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskPoolTest, ShutdownDrainsAcceptedWorkAndRejectsNewWork) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/1);
  TaskTracker tracker(loomc_task_pool_worker_count(pool.get()));
  ExecutionGate gate(/*expected_count=*/1);
  loomc_task_sink_t sink = loomc_task_pool_sink(pool.get());

  LOOMC_ASSERT_OK(
      loomc_task_sink_submit(sink, AllocateGatedTask(&gate, &tracker)));
  gate.WaitUntilEntered();
  LOOMC_ASSERT_OK(loomc_task_pool_shutdown(pool.get()));

  loomc_task_t* rejected_task = AllocateTrackedTask(&tracker);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION,
                         loomc_task_sink_submit(sink, rejected_task));
  loomc_task_destroy(rejected_task);

  gate.Release();
  LOOMC_ASSERT_OK(loomc_task_pool_await_shutdown(pool.get()));
  EXPECT_EQ(tracker.execution_count(), 1);
  EXPECT_EQ(tracker.destruction_count(), 2);
}

TEST(TaskPoolTest, FreeDrainsAcceptedWork) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  TaskTracker tracker(loomc_task_pool_worker_count(pool.get()));
  loomc_task_sink_t sink = loomc_task_pool_sink(pool.get());

  constexpr int kTaskCount = 256;
  for (int i = 0; i < kTaskCount; ++i) {
    LOOMC_ASSERT_OK(
        loomc_task_sink_submit(sink, AllocateTrackedTask(&tracker)));
  }
  pool.reset();

  EXPECT_EQ(tracker.execution_count(), kTaskCount);
  EXPECT_EQ(tracker.destruction_count(), kTaskCount);
}

TEST(TaskPoolTest, AwaitRequiresShutdown) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/1);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION,
                         loomc_task_pool_await_shutdown(pool.get()));
}

}  // namespace
