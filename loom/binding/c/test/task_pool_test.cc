// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task_pool.h"

#include "iree/testing/gtest.h"
#include "loomc/iree/task_pool.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using TaskPoolPtr = HandlePtr<loomc_task_pool_t, loomc_task_pool_free>;

TEST(TaskPoolTest, DefaultUsesFourWorkerLimit) {
  loomc_task_pool_t* raw_pool = nullptr;
  LOOMC_ASSERT_OK(loomc_task_pool_allocate(
      /*options=*/nullptr, loomc_allocator_system(), &raw_pool));
  TaskPoolPtr pool(raw_pool);
  EXPECT_GE(loomc_task_pool_worker_count(pool.get()), 1u);
  EXPECT_LE(loomc_task_pool_worker_count(pool.get()), 4u);
}

TEST(TaskPoolTest, WrapperRetainsExistingExecutor) {
  loomc_task_pool_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS,
      /*.structure_size=*/sizeof(loomc_task_pool_options_t),
      /*.next=*/nullptr,
      /*.max_worker_count=*/2,
      /*.worker_stack_size=*/0,
  };
  loomc_task_pool_t* raw_owner = nullptr;
  LOOMC_ASSERT_OK(
      loomc_task_pool_allocate(&options, loomc_allocator_system(), &raw_owner));
  TaskPoolPtr owner(raw_owner);

  iree_task_executor_t* executor =
      iree_task_executor_from_loomc_task_pool(owner.get());
  ASSERT_NE(executor, nullptr);
  const loomc_host_size_t worker_count =
      loomc_task_pool_worker_count(owner.get());

  loomc_task_pool_t* raw_wrapper = nullptr;
  LOOMC_ASSERT_OK(loomc_task_pool_allocate_from_iree_executor(
      executor, loomc_allocator_system(), &raw_wrapper));
  TaskPoolPtr wrapper(raw_wrapper);
  owner.reset();

  EXPECT_EQ(iree_task_executor_from_loomc_task_pool(wrapper.get()), executor);
  EXPECT_EQ(loomc_task_pool_worker_count(wrapper.get()), worker_count);
}

TEST(TaskPoolTest, NullQueriesReturnEmptyValues) {
  EXPECT_EQ(loomc_task_pool_worker_count(nullptr), 0u);
  EXPECT_EQ(iree_task_executor_from_loomc_task_pool(nullptr), nullptr);
}

}  // namespace
