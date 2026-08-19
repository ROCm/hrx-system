// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/tooling/compile_pool.h"

#include <array>
#include <atomic>

#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr iree_host_size_t kWorkerCount = 4;
constexpr iree_host_size_t kJobCount = 32;

struct BlockingBatchContext {
  iree_notification_t release_notification;
  std::atomic<bool> released{false};
  std::array<std::atomic<int>, kJobCount> invocation_counts{};
  std::array<std::atomic<int>, kWorkerCount> worker_counts{};
};

static bool IsBatchReleased(void* user_data) {
  auto* context = static_cast<BlockingBatchContext*>(user_data);
  return context->released.load(std::memory_order_acquire);
}

static iree_status_t RunBlockingJob(void* user_data,
                                    iree_host_size_t worker_ordinal,
                                    iree_host_size_t job_ordinal) {
  auto* context = static_cast<BlockingBatchContext*>(user_data);
  iree_notification_await(&context->release_notification, IsBatchReleased,
                          context, iree_infinite_timeout());
  context->worker_counts[worker_ordinal].fetch_add(1,
                                                   std::memory_order_relaxed);
  context->invocation_counts[job_ordinal].fetch_add(1,
                                                    std::memory_order_relaxed);
  return iree_ok_status();
}

TEST(CompilePoolTest, RunsSubmittedJobsExactlyOnce) {
  qwen_tooling_compile_pool_t* pool = nullptr;
  IREE_ASSERT_OK(qwen_tooling_compile_pool_create(
      kWorkerCount, iree_allocator_system(), &pool));
  EXPECT_EQ(qwen_tooling_compile_pool_worker_count(pool), kWorkerCount);

  BlockingBatchContext context;
  iree_notification_initialize(&context.release_notification);
  qwen_tooling_compile_batch_t batch;
  IREE_ASSERT_OK(qwen_tooling_compile_pool_submit(
      pool, kJobCount, RunBlockingJob, &context, &batch));

  context.released.store(true, std::memory_order_release);
  iree_notification_post(&context.release_notification, IREE_ALL_WAITERS);
  IREE_EXPECT_OK(qwen_tooling_compile_batch_wait(&batch));

  int total_worker_invocations = 0;
  for (const auto& invocation_count : context.invocation_counts) {
    EXPECT_EQ(invocation_count.load(std::memory_order_relaxed), 1);
  }
  for (const auto& worker_count : context.worker_counts) {
    total_worker_invocations += worker_count.load(std::memory_order_relaxed);
  }
  EXPECT_EQ(total_worker_invocations, kJobCount);

  iree_notification_deinitialize(&context.release_notification);
  qwen_tooling_compile_pool_release(pool);
}

static iree_status_t FailFirstJob(void* /*user_data*/,
                                  iree_host_size_t /*worker_ordinal*/,
                                  iree_host_size_t job_ordinal) {
  if (job_ordinal == 0) {
    return iree_make_status(IREE_STATUS_ABORTED, "requested test failure");
  }
  return iree_ok_status();
}

TEST(CompilePoolTest, PropagatesJobFailure) {
  qwen_tooling_compile_pool_t* pool = nullptr;
  IREE_ASSERT_OK(qwen_tooling_compile_pool_create(
      kWorkerCount, iree_allocator_system(), &pool));

  qwen_tooling_compile_batch_t batch;
  IREE_ASSERT_OK(qwen_tooling_compile_pool_submit(
      pool, kJobCount, FailFirstJob, /*user_data=*/nullptr, &batch));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        qwen_tooling_compile_batch_wait(&batch));

  qwen_tooling_compile_pool_release(pool);
}

}  // namespace
