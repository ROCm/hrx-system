// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_WORKER_GATE_H_
#define LOOM_SERVE_TESTING_WORKER_GATE_H_

#include <condition_variable>
#include <mutex>

#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom_serve::testing {

// Holds one shared executor worker while the owner accepts and cancels real
// CPU tasks. One instance per worker establishes an exact pre-execution cut.
// Destruction releases and joins the gate even after a failed assertion.
class WorkerGate {
 public:
  explicit WorkerGate(iree_task_executor_t* executor) {
    iree_task_process_initialize(
        [](iree_task_process_t* process, const iree_task_worker_context_t*,
           iree_task_process_drain_result_t* result) {
          auto* gate = static_cast<WorkerGate*>(process->user_data);
          std::unique_lock<std::mutex> lock(gate->mutex_);
          gate->entered_ = true;
          gate->condition_.notify_all();
          gate->condition_.wait(lock, [&] { return gate->open_; });
          result->completed = true;
          result->did_work = true;
          return iree_ok_status();
        },
        0, 1, &process_);
    process_.user_data = this;
    process_.completion_fn = [](iree_task_process_t*, iree_status_t status) {
      IREE_EXPECT_OK(status);
    };
    process_.release_fn = [](iree_task_process_t* process) {
      auto* gate = static_cast<WorkerGate*>(process->user_data);
      std::lock_guard<std::mutex> lock(gate->mutex_);
      gate->released_ = true;
      gate->condition_.notify_all();
    };
    iree_task_executor_schedule_process(executor, &process_);
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }
  ~WorkerGate() { Open(); }
  void Open() {
    std::unique_lock<std::mutex> lock(mutex_);
    open_ = true;
    condition_.notify_all();
    condition_.wait(lock, [&] { return released_; });
  }

 private:
  // Intrusive process storage retained through the final release callback.
  iree_task_process_t process_ = {};
  // Serializes every gate publication, including final destruction.
  std::mutex mutex_;
  // Explicit worker-entry, gate-opening, and retirement readiness.
  std::condition_variable condition_;
  // One application worker has entered this process.
  bool entered_ = false;
  // The test owner has released the worker.
  bool open_ = false;
  // The executor has made its final process access.
  bool released_ = false;
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_WORKER_GATE_H_
