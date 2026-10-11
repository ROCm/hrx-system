// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/admission.h"

#include "iree/testing/gtest.h"

namespace {

TEST(TextAdmissionTest, ZeroBudgetPreservesFifo) {
  loom_serve_text_admission_t policy = {};
  EXPECT_FALSE(loom_serve_text_admission_blocked(&policy, 16));
  EXPECT_EQ(policy.cursor, 0u);
}

TEST(TextAdmissionTest, OnlySuccessfulOvertakesSpendTheOldestBudget) {
  loom_serve_text_admission_t policy = {.budget = 2};
  ASSERT_TRUE(loom_serve_text_admission_blocked(&policy, 3));
  ASSERT_TRUE(loom_serve_text_admission_blocked(&policy, 3));
  EXPECT_EQ(policy.cursor, 2u);
  EXPECT_EQ(policy.overtakes, 0u);
  loom_serve_text_admission_acquired(&policy);
  loom_serve_text_admission_remove(&policy, 2);
  EXPECT_EQ(policy.cursor, 0u);
  EXPECT_EQ(policy.overtakes, 1u);
  ASSERT_TRUE(loom_serve_text_admission_blocked(&policy, 2));
  loom_serve_text_admission_remove(&policy, 1);
  EXPECT_EQ(policy.overtakes, 1u);
  ASSERT_TRUE(loom_serve_text_admission_blocked(&policy, 2));
  loom_serve_text_admission_acquired(&policy);
  loom_serve_text_admission_remove(&policy, 1);
  EXPECT_FALSE(loom_serve_text_admission_blocked(&policy, 16));
  EXPECT_EQ(policy.overtakes, 2u);
  loom_serve_text_admission_remove(&policy, 0);
  EXPECT_EQ(policy.overtakes, 0u);
  EXPECT_TRUE(loom_serve_text_admission_blocked(&policy, 2));
}

TEST(TextAdmissionTest, ExhaustedScanRestartsAtOldestWithoutSpendingCredit) {
  loom_serve_text_admission_t policy = {.budget = 4};
  ASSERT_TRUE(loom_serve_text_admission_blocked(&policy, 2));
  EXPECT_FALSE(loom_serve_text_admission_blocked(&policy, 2));
  EXPECT_EQ(policy.cursor, 0u);
  EXPECT_EQ(policy.overtakes, 0u);
  loom_serve_text_admission_acquired(&policy);
  EXPECT_EQ(policy.overtakes, 0u);
}

}  // namespace
