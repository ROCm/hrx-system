// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/threading/notification.h"
#include "loomc/iree.h"
#include "loomc/task_pool.h"
#include "loomc/task_queue.h"

namespace {

template <typename T, void (*Free)(T*)>
struct HandleDeleter {
  void operator()(T* value) const { Free(value); }
};

using TaskPoolPtr =
    std::unique_ptr<loomc_task_pool_t,
                    HandleDeleter<loomc_task_pool_t, loomc_task_pool_free>>;
using TaskQueuePtr =
    std::unique_ptr<loomc_task_queue_t,
                    HandleDeleter<loomc_task_queue_t, loomc_task_queue_free>>;

typedef struct task_batch_t {
  // Number of tasks that have not completed execution.
  iree_atomic_int32_t remaining_count;

  // Posted when the final task completes.
  iree_notification_t notification;
} task_batch_t;

typedef struct benchmark_task_t {
  // Generic task base reinitialized before each submission.
  loomc_task_t base;

  // Shared batch completion state.
  task_batch_t* batch;
} benchmark_task_t;

static void CompleteBenchmarkTask(benchmark_task_t* task) {
  const int32_t previous = iree_atomic_fetch_sub(&task->batch->remaining_count,
                                                 1, iree_memory_order_acq_rel);
  if (previous == 1) {
    iree_notification_post(&task->batch->notification, IREE_ALL_WAITERS);
  }
}

static void ExecuteBenchmarkTask(loomc_task_t* base_task,
                                 loomc_host_size_t worker_ordinal) {
  (void)base_task;
  (void)worker_ordinal;
}

static void DestroyBenchmarkTask(loomc_task_t* base_task) {
  CompleteBenchmarkTask(reinterpret_cast<benchmark_task_t*>(base_task));
}

static const loomc_task_vtable_t kBenchmarkTaskVtable = {
    /*.execute=*/ExecuteBenchmarkTask,
    /*.destroy=*/DestroyBenchmarkTask,
};

static bool TaskBatchComplete(void* user_data) {
  const task_batch_t* batch = static_cast<const task_batch_t*>(user_data);
  return iree_atomic_load(&batch->remaining_count, iree_memory_order_acquire) ==
         0;
}

class TaskQueueBenchmarkFixture {
 public:
  TaskQueueBenchmarkFixture(loomc_host_size_t worker_count,
                            loomc_host_size_t task_count)
      : tasks_(task_count) {
    loomc_task_pool_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS,
        /*.structure_size=*/sizeof(loomc_task_pool_options_t),
        /*.next=*/nullptr,
        /*.max_worker_count=*/worker_count,
        /*.worker_stack_size=*/0,
    };
    loomc_task_pool_t* pool = nullptr;
    IREE_CHECK_OK(iree_status_from_loomc(
        loomc_task_pool_allocate(&options, loomc_allocator_system(), &pool)));
    pool_.reset(pool);
    loomc_task_queue_t* queue = nullptr;
    IREE_CHECK_OK(iree_status_from_loomc(loomc_task_queue_allocate(
        pool_.get(), loomc_allocator_system(), &queue)));
    queue_.reset(queue);
    sink_ = loomc_task_queue_sink(queue_.get());
    iree_atomic_store(&batch_.remaining_count, 0, iree_memory_order_relaxed);
    iree_notification_initialize(&batch_.notification);
    for (benchmark_task_t& task : tasks_) task.batch = &batch_;
  }

  ~TaskQueueBenchmarkFixture() {
    queue_.reset();
    iree_notification_deinitialize(&batch_.notification);
  }

  loomc_host_size_t task_count() const { return tasks_.size(); }

  void SubmitBatch() {
    iree_atomic_store(&batch_.remaining_count, (int32_t)tasks_.size(),
                      iree_memory_order_release);
    for (benchmark_task_t& task : tasks_) {
      loomc_task_initialize(&kBenchmarkTaskVtable, &task.base);
      IREE_CHECK_OK(
          iree_status_from_loomc(loomc_task_sink_submit(sink_, &task.base)));
    }
  }

  void AwaitBatch() {
    iree_notification_await(&batch_.notification, TaskBatchComplete, &batch_,
                            iree_infinite_timeout());
  }

 private:
  // Shared worker population under measurement.
  TaskPoolPtr pool_;

  // Task scheduling domain attached to `pool_`.
  TaskQueuePtr queue_;

  // Borrowed sink backed by `queue_`.
  loomc_task_sink_t sink_ = {0};

  // Completion state shared by every task in one iteration.
  task_batch_t batch_;

  // Reusable task storage excluded from iteration allocation costs.
  std::vector<benchmark_task_t> tasks_;
};

// Measures caller time spent initializing and submitting already-allocated
// tasks. Final task completion is kept outside of the timed region.
static void BM_TaskQueueSubmitOnly(benchmark::State& state) {
  TaskQueueBenchmarkFixture fixture((loomc_host_size_t)state.range(0),
                                    (loomc_host_size_t)state.range(1));
  for (auto _ : state) {
    fixture.SubmitBatch();
    state.PauseTiming();
    fixture.AwaitBatch();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * fixture.task_count());
}
BENCHMARK(BM_TaskQueueSubmitOnly)
    ->Args({1, 1})
    ->Args({1, 64})
    ->Args({4, 1})
    ->Args({4, 64})
    ->Args({4, 1024})
    ->Args({8, 1})
    ->Args({8, 64})
    ->Args({8, 1024})
    ->UseRealTime();

// Measures the user-visible round trip from task initialization and submission
// through completion of the entire batch.
static void BM_TaskQueueEndToEnd(benchmark::State& state) {
  TaskQueueBenchmarkFixture fixture((loomc_host_size_t)state.range(0),
                                    (loomc_host_size_t)state.range(1));
  for (auto _ : state) {
    fixture.SubmitBatch();
    fixture.AwaitBatch();
  }
  state.SetItemsProcessed(state.iterations() * fixture.task_count());
}
BENCHMARK(BM_TaskQueueEndToEnd)
    ->Args({1, 1})
    ->Args({1, 64})
    ->Args({4, 1})
    ->Args({4, 64})
    ->Args({4, 1024})
    ->Args({8, 1})
    ->Args({8, 64})
    ->Args({8, 1024})
    ->UseRealTime();

}  // namespace
