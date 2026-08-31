// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task_queue.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "loomc/task_pool.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using TaskPoolPtr = HandlePtr<loomc_task_pool_t, loomc_task_pool_free>;
using TaskQueuePtr = HandlePtr<loomc_task_queue_t, loomc_task_queue_free>;

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

typedef enum pipeline_task_role_e {
  PIPELINE_TASK_ROLE_COMPILE = 0,
  PIPELINE_TASK_ROLE_MATERIALIZE = 1,
} pipeline_task_role_t;

typedef struct pipeline_tracker_t {
  // Number of compile tasks that have executed.
  std::atomic<int> compile_count{0};

  // Compile count observed when the materialization task executed.
  std::atomic<int> compile_count_at_materialization{0};

  // Status from publishing the materialization task.
  std::atomic<loomc_status_code_t> publication_status{LOOMC_STATUS_OK};
} pipeline_tracker_t;

typedef struct pipeline_task_t {
  // Generic task base.
  loomc_task_t base;

  // Scheduling role performed by this task.
  pipeline_task_role_t role;

  // External outcome tracker that outlives the task.
  pipeline_tracker_t* tracker;

  // Borrowed sink receiving `downstream_task` during execution.
  loomc_task_sink_t downstream_sink;

  // Owned downstream task awaiting publication, or NULL.
  loomc_task_t* downstream_task;
} pipeline_task_t;

static void ExecutePipelineTask(loomc_task_t* base_task,
                                loomc_host_size_t worker_ordinal) {
  (void)worker_ordinal;
  pipeline_task_t* task = reinterpret_cast<pipeline_task_t*>(base_task);
  switch (task->role) {
    case PIPELINE_TASK_ROLE_COMPILE:
      task->tracker->compile_count.fetch_add(1, std::memory_order_release);
      if (task->downstream_task != nullptr) {
        loomc_task_t* downstream_task = task->downstream_task;
        task->downstream_task = nullptr;
        loomc_status_t status =
            loomc_task_sink_submit(task->downstream_sink, downstream_task);
        if (!loomc_status_is_ok(status)) {
          task->tracker->publication_status.store(
              loomc_status_consume_code(status), std::memory_order_release);
          loomc_task_destroy(downstream_task);
        }
      }
      break;
    case PIPELINE_TASK_ROLE_MATERIALIZE:
      task->tracker->compile_count_at_materialization.store(
          task->tracker->compile_count.load(std::memory_order_acquire),
          std::memory_order_release);
      break;
  }
}

static void DestroyPipelineTask(loomc_task_t* base_task) {
  pipeline_task_t* task = reinterpret_cast<pipeline_task_t*>(base_task);
  if (task->downstream_task != nullptr) {
    loomc_task_destroy(task->downstream_task);
  }
  delete task;
}

static const loomc_task_vtable_t kPipelineTaskVtable = {
    /*.execute=*/ExecutePipelineTask,
    /*.destroy=*/DestroyPipelineTask,
};

static loomc_task_t* AllocatePipelineTask(pipeline_task_role_t role,
                                          pipeline_tracker_t* tracker) {
  pipeline_task_t* task = new pipeline_task_t;
  loomc_task_initialize(&kPipelineTaskVtable, &task->base);
  task->role = role;
  task->tracker = tracker;
  task->downstream_sink = {};
  task->downstream_task = nullptr;
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

static TaskQueuePtr AllocateTaskQueue(loomc_task_pool_t* pool) {
  loomc_task_queue_t* queue = nullptr;
  IREE_CHECK_OK(iree_status_from_loomc(
      loomc_task_queue_allocate(pool, loomc_allocator_system(), &queue)));
  return TaskQueuePtr(queue);
}

TEST(TaskQueueTest, DrainsAcceptedTasksWithExclusiveWorkerOrdinals) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  TaskQueuePtr queue = AllocateTaskQueue(pool.get());
  const loomc_host_size_t worker_count =
      loomc_task_pool_worker_count(pool.get());
  TaskTracker tracker(worker_count);
  loomc_task_sink_t sink = loomc_task_queue_sink(queue.get());

  constexpr int kTaskCount = 256;
  for (int i = 0; i < kTaskCount; ++i) {
    LOOMC_ASSERT_OK(
        loomc_task_sink_submit(sink, AllocateTrackedTask(&tracker)));
  }
  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(queue.get()));

  EXPECT_EQ(tracker.execution_count(), kTaskCount);
  EXPECT_EQ(tracker.destruction_count(), kTaskCount);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskQueueTest, ConcurrentSubmittersShareOneQueue) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  TaskQueuePtr queue = AllocateTaskQueue(pool.get());
  TaskTracker tracker(loomc_task_pool_worker_count(pool.get()));
  loomc_task_sink_t sink = loomc_task_queue_sink(queue.get());

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

  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(queue.get()));
  EXPECT_EQ(tracker.submission_status_code(), LOOMC_STATUS_OK);
  EXPECT_EQ(tracker.execution_count(), kSubmitterCount * kTasksPerSubmitter);
  EXPECT_EQ(tracker.destruction_count(), kSubmitterCount * kTasksPerSubmitter);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskQueueTest, ReusesWorkersAcrossSubmissionWaves) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  TaskQueuePtr queue = AllocateTaskQueue(pool.get());
  const loomc_host_size_t worker_count =
      loomc_task_pool_worker_count(pool.get());
  TaskTracker tracker(worker_count);
  loomc_task_sink_t sink = loomc_task_queue_sink(queue.get());

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

  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(queue.get()));
  EXPECT_EQ(tracker.execution_count(), expected_task_count);
  EXPECT_EQ(tracker.destruction_count(), expected_task_count);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskQueueTest, RecursiveSubmissionUsesMultipleWorkers) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  TaskQueuePtr queue = AllocateTaskQueue(pool.get());
  const int worker_count =
      static_cast<int>(loomc_task_pool_worker_count(pool.get()));
  if (worker_count < 2) GTEST_SKIP() << "host exposes only one worker";

  ExecutionGate child_gate(/*expected_count=*/2);
  TaskTracker tracker(worker_count);
  loomc_task_sink_t sink = loomc_task_queue_sink(queue.get());
  const int child_count = worker_count * 2;
  LOOMC_ASSERT_OK(loomc_task_sink_submit(
      sink, AllocateRecursiveTask(sink, &child_gate, child_count, &tracker)));

  child_gate.WaitUntilEntered();
  child_gate.Release();
  tracker.WaitForExecutions(/*expected_count=*/child_count + 1);
  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(queue.get()));

  EXPECT_EQ(tracker.submission_status_code(), LOOMC_STATUS_OK);
  EXPECT_EQ(tracker.execution_count(), child_count + 1);
  EXPECT_EQ(tracker.destruction_count(), child_count + 1);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskQueueTest, ShutdownDrainsAcceptedWorkAndRejectsNewWork) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/1);
  TaskQueuePtr queue = AllocateTaskQueue(pool.get());
  TaskTracker tracker(loomc_task_pool_worker_count(pool.get()));
  ExecutionGate gate(/*expected_count=*/1);
  loomc_task_sink_t sink = loomc_task_queue_sink(queue.get());

  LOOMC_ASSERT_OK(
      loomc_task_sink_submit(sink, AllocateGatedTask(&gate, &tracker)));
  gate.WaitUntilEntered();
  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(queue.get()));

  loomc_task_t* rejected_task = AllocateTrackedTask(&tracker);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION,
                         loomc_task_sink_submit(sink, rejected_task));
  loomc_task_destroy(rejected_task);

  gate.Release();
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(queue.get()));
  EXPECT_EQ(tracker.execution_count(), 1);
  EXPECT_EQ(tracker.destruction_count(), 2);
}

TEST(TaskQueueTest, FreeDrainsAcceptedWork) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  TaskQueuePtr queue = AllocateTaskQueue(pool.get());
  TaskTracker tracker(loomc_task_pool_worker_count(pool.get()));
  loomc_task_sink_t sink = loomc_task_queue_sink(queue.get());

  constexpr int kTaskCount = 256;
  for (int i = 0; i < kTaskCount; ++i) {
    LOOMC_ASSERT_OK(
        loomc_task_sink_submit(sink, AllocateTrackedTask(&tracker)));
  }
  queue.reset();

  EXPECT_EQ(tracker.execution_count(), kTaskCount);
  EXPECT_EQ(tracker.destruction_count(), kTaskCount);
}

TEST(TaskQueueTest, AwaitRequiresShutdown) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/1);
  TaskQueuePtr queue = AllocateTaskQueue(pool.get());
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION,
                         loomc_task_queue_await_shutdown(queue.get()));
}

TEST(TaskQueueTest, QueuesRetainAndIndependentlyUseSharedWorkers) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/4);
  const loomc_host_size_t worker_count =
      loomc_task_pool_worker_count(pool.get());
  TaskQueuePtr compile_queue = AllocateTaskQueue(pool.get());
  TaskQueuePtr materialize_queue = AllocateTaskQueue(pool.get());
  TaskTracker tracker(worker_count);
  loomc_task_sink_t compile_sink = loomc_task_queue_sink(compile_queue.get());
  loomc_task_sink_t materialize_sink =
      loomc_task_queue_sink(materialize_queue.get());

  // Queue ownership keeps the shared executor alive independently from the
  // convenience pool handle.
  pool.reset();

  constexpr int kTaskCountPerQueue = 256;
  for (int i = 0; i < kTaskCountPerQueue; ++i) {
    LOOMC_ASSERT_OK(
        loomc_task_sink_submit(compile_sink, AllocateTrackedTask(&tracker)));
    LOOMC_ASSERT_OK(loomc_task_sink_submit(materialize_sink,
                                           AllocateTrackedTask(&tracker)));
  }

  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(compile_queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(compile_queue.get()));

  // Shutting one scheduling domain down does not stop another domain sharing
  // the same workers.
  LOOMC_ASSERT_OK(
      loomc_task_sink_submit(materialize_sink, AllocateTrackedTask(&tracker)));
  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(materialize_queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(materialize_queue.get()));

  EXPECT_EQ(tracker.execution_count(), kTaskCountPerQueue * 2 + 1);
  EXPECT_EQ(tracker.destruction_count(), kTaskCountPerQueue * 2 + 1);
  EXPECT_FALSE(tracker.has_invalid_worker_ordinal());
  EXPECT_FALSE(tracker.has_concurrent_worker_ordinal());
}

TEST(TaskQueueTest, PipelinesDependentWorkAcrossIndependentQueues) {
  TaskPoolPtr pool = AllocateTaskPool(/*max_worker_count=*/1);
  TaskQueuePtr compile_queue = AllocateTaskQueue(pool.get());
  TaskQueuePtr materialize_queue = AllocateTaskQueue(pool.get());
  const loomc_task_sink_t compile_sink =
      loomc_task_queue_sink(compile_queue.get());
  const loomc_task_sink_t materialize_sink =
      loomc_task_queue_sink(materialize_queue.get());
  pipeline_tracker_t tracker;

  loomc_task_t* materialize_task =
      AllocatePipelineTask(PIPELINE_TASK_ROLE_MATERIALIZE, &tracker);
  loomc_task_t* first_compile_task =
      AllocatePipelineTask(PIPELINE_TASK_ROLE_COMPILE, &tracker);
  pipeline_task_t* first_compile =
      reinterpret_cast<pipeline_task_t*>(first_compile_task);
  first_compile->downstream_sink = materialize_sink;
  first_compile->downstream_task = materialize_task;
  LOOMC_ASSERT_OK(loomc_task_sink_submit(compile_sink, first_compile_task));

  constexpr int kCompileTaskCount = 256;
  for (int i = 1; i < kCompileTaskCount; ++i) {
    LOOMC_ASSERT_OK(loomc_task_sink_submit(
        compile_sink,
        AllocatePipelineTask(PIPELINE_TASK_ROLE_COMPILE, &tracker)));
  }

  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(compile_queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(compile_queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_shutdown(materialize_queue.get()));
  LOOMC_ASSERT_OK(loomc_task_queue_await_shutdown(materialize_queue.get()));

  EXPECT_EQ(tracker.publication_status.load(std::memory_order_acquire),
            LOOMC_STATUS_OK);
  EXPECT_EQ(tracker.compile_count.load(std::memory_order_acquire),
            kCompileTaskCount);
  EXPECT_GT(
      tracker.compile_count_at_materialization.load(std::memory_order_acquire),
      0);
  EXPECT_LT(
      tracker.compile_count_at_materialization.load(std::memory_order_acquire),
      kCompileTaskCount);
}

}  // namespace
