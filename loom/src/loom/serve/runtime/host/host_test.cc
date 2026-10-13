// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/host/host.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loomc/iree/task_pool.h"

namespace {

TEST(HostTest, NullLifetimeOperations) {
  loom_serve_host_retain(nullptr);
  loom_serve_host_release(nullptr);
}

TEST(HostTest, CompilerAdapterUsesApplicationExecutor) {
  loom_serve_host_t* host = nullptr;
  IREE_ASSERT_OK(loom_serve_host_create(1, &host, iree_allocator_system()));
  EXPECT_EQ(
      loom_serve_host_executor(host),
      iree_task_executor_from_loomc_task_pool(loom_serve_host_task_pool(host)));
  EXPECT_EQ(iree_task_executor_worker_count(loom_serve_host_executor(host)),
            1u);
  loom_serve_host_retain(host);
  loom_serve_host_release(host);
  EXPECT_NE(loom_serve_host_proactor_pool(host), nullptr);
  loom_serve_host_release(host);
}

TEST(HostTest, InvalidCapacityAndAllocationFailureReturnNoHost) {
  loom_serve_host_t* host = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_host_create(0, &host, iree_allocator_system()));
  EXPECT_EQ(host, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_host_create(IREE_TASK_EXECUTOR_MAX_WORKER_COUNT + 1, &host,
                             iree_allocator_system()));
  EXPECT_EQ(host, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_host_create(1, &host, iree_allocator_null()));
  EXPECT_EQ(host, nullptr);
}

}  // namespace
