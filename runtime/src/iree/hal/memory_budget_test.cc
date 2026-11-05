// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory_budget.h"

#include <barrier>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(MemoryBudgetTest, CapacityIncludesPendingAndRetiringCredits) {
  iree_hal_memory_budget_t* budget = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_budget_create(
      8192, nullptr, iree_allocator_system(), &budget));
  iree_hal_memory_credit_t batch = {};
  ASSERT_TRUE(iree_hal_memory_budget_try_reserve(budget, 8192, &batch));
  auto first = iree_hal_memory_credit_split(&batch, 4096);
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 8192u);
  iree_hal_memory_credit_t denied = {};
  EXPECT_FALSE(iree_hal_memory_budget_try_reserve(budget, 1, &denied));
  iree_hal_memory_credit_return(&first);
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 4096u);
  ASSERT_TRUE(iree_hal_memory_budget_try_reserve(budget, 4096, &first));

  // Lowering the limit cannot revoke the two accepted owners.
  iree_hal_memory_budget_set_limit(budget, 0);
  iree_hal_memory_credit_return(&batch);
  EXPECT_FALSE(iree_hal_memory_budget_try_reserve(budget, 1, &denied));
  iree_hal_memory_credit_return(&first);
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 0u);
  iree_hal_memory_budget_release(budget);
}

TEST(MemoryBudgetTest, LossPreservesCapacityAndCompleteDiagnostic) {
  iree_hal_memory_budget_t* budget = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_budget_create(
      8192, nullptr, iree_allocator_system(), &budget));
  iree_hal_memory_credit_t credit = {};
  ASSERT_TRUE(iree_hal_memory_budget_try_reserve(budget, 8192, &credit));
  iree_hal_memory_credit_settle(
      &credit, 4096,
      iree_make_status(IREE_STATUS_UNAVAILABLE,
                       "native mapping still owns pages"));
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 4096u);
  EXPECT_EQ(iree_hal_memory_budget_abandoned(budget), 4096u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_memory_budget_query_loss(budget));
  iree_status_t diagnostic = iree_hal_memory_budget_query_loss(budget);
  char message[2048];
  iree_host_size_t length = 0;
  const bool formatted =
      iree_status_format(diagnostic, sizeof(message), message, &length);
  iree_status_free(diagnostic);
  ASSERT_TRUE(formatted);
  EXPECT_NE(
      std::string(message, length).find("native mapping still owns pages"),
      std::string::npos);
  ASSERT_TRUE(iree_hal_memory_budget_try_reserve(budget, 4096, &credit));
  iree_hal_memory_credit_return(&credit);
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 4096u);
  iree_hal_memory_budget_release(budget);
}

TEST(MemoryBudgetTest, SharedSourcesCannotOverdraw) {
  iree_hal_memory_budget_t* budget = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_budget_create(
      4096, nullptr, iree_allocator_system(), &budget));
  std::barrier<> ready(3);
  iree_hal_memory_credit_t credits[2] = {};
  bool admitted[2] = {};
  auto reserve = [&](size_t index) {
    ready.arrive_and_wait();
    admitted[index] =
        iree_hal_memory_budget_try_reserve(budget, 4096, &credits[index]);
  };
  std::thread first(reserve, 0);
  std::thread second(reserve, 1);
  ready.arrive_and_wait();
  first.join();
  second.join();
  EXPECT_NE(admitted[0], admitted[1]);
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 4096u);
  iree_hal_memory_credit_return(&credits[0]);
  iree_hal_memory_credit_return(&credits[1]);
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), 0u);
  iree_hal_memory_budget_release(budget);
}

TEST(MemoryBudgetTest, CreditRetainsAccountAndUnsignedCapacityBoundaries) {
  iree_hal_memory_budget_t* budget = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_budget_create(
      UINT64_MAX, nullptr, iree_allocator_system(), &budget));
  iree_hal_memory_credit_t credit = {};
  ASSERT_TRUE(iree_hal_memory_budget_try_reserve(budget, UINT64_MAX, &credit));
  iree_hal_memory_credit_t denied = {};
  EXPECT_FALSE(iree_hal_memory_budget_try_reserve(budget, 1, &denied));
  EXPECT_EQ(iree_hal_memory_budget_charged(budget), UINT64_MAX);
  auto part = iree_hal_memory_credit_split(&credit, UINT64_MAX - 1);
  iree_hal_memory_budget_release(budget);
  iree_hal_memory_credit_return(&credit);
  EXPECT_EQ(iree_hal_memory_budget_charged(part.budget), UINT64_MAX - 1);
  iree_hal_memory_credit_return(&part);
}

}  // namespace
