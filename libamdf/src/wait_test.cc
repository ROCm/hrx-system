// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/wait.h"

#include "gtest/gtest.h"
#include "libamdf/src/platform/wait.h"

namespace {

// The clock dependency advances only at explicit host/native wait boundaries.
uint64_t now = 0;
// Clock failure is independent of elapsed time and must reach the caller.
amdf_status_t clock_status = AMDF_STATUS_OK;

TEST(WaitTest, HostContentionAndNativeWaitingConsumeOneBudget) {
  now = 100;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(1000, 200, &deadline),
            AMDF_STATUS_OK);
  amdf_wait_budget_t remaining;
  now += 150;
  ASSERT_EQ(amdf_wait_deadline_query_remaining(&deadline, &remaining),
            AMDF_STATUS_OK);
  EXPECT_EQ(remaining.timeout, 850u);
  EXPECT_EQ(remaining.poll, 50u);
  now += 100;
  ASSERT_EQ(amdf_wait_deadline_query_remaining(&deadline, &remaining),
            AMDF_STATUS_OK);
  EXPECT_EQ(remaining.timeout, 750u);
  EXPECT_EQ(remaining.poll, 0u);
  now += 800;
  ASSERT_EQ(amdf_wait_deadline_query_remaining(&deadline, &remaining),
            AMDF_STATUS_OK);
  EXPECT_EQ(remaining.timeout, 0u);
  EXPECT_EQ(remaining.poll, 0u);
}

TEST(WaitTest, ActivePollingCannotExtendFiniteTimeout) {
  now = 50;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(
      amdf_wait_deadline_initialize(100, AMDF_TIMEOUT_INFINITE, &deadline),
      AMDF_STATUS_OK);
  now += 70;
  amdf_wait_budget_t remaining;
  ASSERT_EQ(amdf_wait_deadline_query_remaining(&deadline, &remaining),
            AMDF_STATUS_OK);
  EXPECT_EQ(remaining.timeout, 30u);
  EXPECT_EQ(remaining.poll, 30u);
}

TEST(WaitTest, InfiniteTimeoutDoesNotExtendActivePolling) {
  now = 50;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(
      amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 100, &deadline),
      AMDF_STATUS_OK);
  now += 200;
  amdf_wait_budget_t remaining;
  ASSERT_EQ(amdf_wait_deadline_query_remaining(&deadline, &remaining),
            AMDF_STATUS_OK);
  EXPECT_EQ(remaining.timeout, AMDF_TIMEOUT_INFINITE);
  EXPECT_EQ(remaining.poll, 0u);
}

TEST(WaitTest, FiniteOverflowCannotBecomeInfinite) {
  now = 100;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(UINT64_MAX - 1, 0, &deadline),
            AMDF_STATUS_OK);
  amdf_wait_budget_t remaining;
  ASSERT_EQ(amdf_wait_deadline_query_remaining(&deadline, &remaining),
            AMDF_STATUS_OK);
  EXPECT_NE(remaining.timeout, AMDF_TIMEOUT_INFINITE);
  EXPECT_GT(remaining.timeout, 0u);
  EXPECT_EQ(remaining.poll, 0u);
}

TEST(WaitTest, ClockFailurePreservesOutputs) {
  clock_status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  amdf_wait_deadline_t deadline = {7, 9};
  EXPECT_EQ(amdf_wait_deadline_initialize(100, 10, &deadline), clock_status);
  EXPECT_EQ(deadline.timeout, 7u);
  EXPECT_EQ(deadline.poll, 9u);
  amdf_wait_budget_t remaining = {11, 13};
  EXPECT_EQ(amdf_wait_deadline_query_remaining(&deadline, &remaining),
            clock_status);
  EXPECT_EQ(remaining.timeout, 11u);
  EXPECT_EQ(remaining.poll, 13u);
  clock_status = AMDF_STATUS_OK;
}

}  // namespace

extern "C" amdf_status_t amdf_platform_wait_query_time(
    uint64_t* out_nanoseconds) {
  if (amdf_status_is_ok(clock_status)) *out_nanoseconds = now;
  return clock_status;
}
