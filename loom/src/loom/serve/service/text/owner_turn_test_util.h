// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_OWNER_TURN_TEST_UTIL_H_
#define LOOM_SERVE_SERVICE_TEXT_OWNER_TURN_TEST_UTIL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <utility>

#include "iree/task/executor.h"

namespace loom_serve::testing::text {

// One finite owner turn on the actual shared executor. With one worker, CPU
// preparation submitted by this turn cannot start until the turn returns.
// Only the outer test waits; final process release returns all borrowed data.
class OwnerTurn {
 public:
  static void Run(iree_task_executor_t* executor, std::function<void()> body) {
    OwnerTurn turn;
    turn.body_ = std::move(body);
    iree_task_process_initialize(Drain, 0, 1, &turn.process_);
    turn.process_.user_data = &turn;
    turn.process_.release_fn = Release;
    iree_task_executor_schedule_process(executor, &turn.process_);
    std::unique_lock<std::mutex> lock(turn.mutex_);
    turn.condition_.wait(lock, [&] { return turn.released_; });
  }

 private:
  static iree_status_t Drain(iree_task_process_t* process,
                             const iree_task_worker_context_t*,
                             iree_task_process_drain_result_t* result) {
    static_cast<OwnerTurn*>(process->user_data)->body_();
    result->completed = true;
    result->did_work = true;
    return iree_ok_status();
  }

  static void Release(iree_task_process_t* process) {
    auto* turn = static_cast<OwnerTurn*>(process->user_data);
    std::lock_guard<std::mutex> lock(turn->mutex_);
    turn->released_ = true;
    turn->condition_.notify_one();
  }

  // Finite service-owner work, borrowing the caller's still-live captures.
  std::function<void()> body_;
  // Intrusive process storage retained through Release.
  iree_task_process_t process_ = {};
  // Joins the executor's final access to this owner.
  std::mutex mutex_;
  // Wakes the outer test after process release.
  std::condition_variable condition_;
  // All process storage can be reclaimed after publication under mutex_.
  bool released_ = false;
};

}  // namespace loom_serve::testing::text

#endif  // LOOM_SERVE_SERVICE_TEXT_OWNER_TURN_TEST_UTIL_H_
