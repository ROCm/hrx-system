// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/loom_compile_pool.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

constexpr iree_host_size_t kWorkerCount = 4;
constexpr iree_host_size_t kJobCount = 37;

struct VisitState {
  std::atomic<uint64_t> visited_jobs{0};
  std::array<std::atomic<int>, kWorkerCount> active_workers{};
  std::atomic<bool> overlapping_worker_ordinal{false};
};

iree_status_t VisitJob(void* user_data, iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) {
  auto* state = static_cast<VisitState*>(user_data);
  if (worker_ordinal >= kWorkerCount || job_ordinal >= kJobCount) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "compile pool produced an invalid ordinal");
  }

  if (state->active_workers[worker_ordinal].fetch_add(1) != 0) {
    state->overlapping_worker_ordinal.store(true);
  }
  std::this_thread::yield();
  state->visited_jobs.fetch_or(uint64_t{1} << job_ordinal);
  state->active_workers[worker_ordinal].fetch_sub(1);
  return iree_ok_status();
}

iree_status_t FailJob(void* user_data, iree_host_size_t worker_ordinal,
                      iree_host_size_t job_ordinal) {
  (void)user_data;
  (void)worker_ordinal;
  if (job_ordinal == 0) {
    return iree_make_status(IREE_STATUS_ABORTED, "requested test failure");
  }
  return iree_ok_status();
}

TEST(QwenLoomCompilePoolTest, VisitsEveryJobWithExclusiveWorkerOrdinals) {
  qwen_loom_compile_pool_t pool;
  IREE_ASSERT_OK(qwen_loom_compile_pool_initialize(
      kWorkerCount, iree_allocator_system(), &pool));

  VisitState state;
  IREE_EXPECT_OK(
      qwen_loom_compile_pool_run_batch(&pool, kJobCount, VisitJob, &state));
  EXPECT_EQ(state.visited_jobs.load(), (uint64_t{1} << kJobCount) - 1);
  EXPECT_FALSE(state.overlapping_worker_ordinal.load());

  qwen_loom_compile_pool_deinitialize(&pool);
}

TEST(QwenLoomCompilePoolTest, PropagatesJobFailureAfterDrainersExit) {
  qwen_loom_compile_pool_t pool;
  IREE_ASSERT_OK(qwen_loom_compile_pool_initialize(
      kWorkerCount, iree_allocator_system(), &pool));

  iree_status_t status =
      qwen_loom_compile_pool_run_batch(&pool, kJobCount, FailJob, nullptr);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kAborted));
  iree_status_free(status);

  qwen_loom_compile_pool_deinitialize(&pool);
}

}  // namespace
