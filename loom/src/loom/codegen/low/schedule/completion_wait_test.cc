// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/completion_wait.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace loom {
namespace {

class ScheduleCompletionWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    descriptor_set_.hazards = hazards_;
    descriptor_set_.hazard_count = IREE_ARRAYSIZE(hazards_);
  }

  // Target hazard rows selected by schedule_class_.
  loom_low_hazard_t hazards_[3] = {};
  // Target descriptor set owning hazards_.
  loom_low_descriptor_set_t descriptor_set_ = {};
  // Schedule class queried by each test.
  loom_low_schedule_class_t schedule_class_ = {};
};

TEST_F(ScheduleCompletionWaitTest, MissingTargetModelHasNoCompletionWait) {
  uint16_t wait_cycles = 7;
  EXPECT_FALSE(loom_low_schedule_class_query_completion_wait(
      nullptr, &schedule_class_, &wait_cycles));
  EXPECT_EQ(wait_cycles, 0u);

  wait_cycles = 7;
  EXPECT_FALSE(loom_low_schedule_class_query_completion_wait(
      &descriptor_set_, nullptr, &wait_cycles));
  EXPECT_EQ(wait_cycles, 0u);
}

TEST_F(ScheduleCompletionWaitTest,
       NonCounterHazardsDoNotRequireCompletionWait) {
  hazards_[0].kind = LOOM_LOW_HAZARD_KIND_MIN_DISTANCE;
  hazards_[0].reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE;
  hazards_[0].distance = 9;
  hazards_[1].kind = LOOM_LOW_HAZARD_KIND_BYPASS;
  hazards_[1].reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_TARGET;
  hazards_[1].distance = 5;
  schedule_class_.hazard_count = 2;

  uint16_t wait_cycles = 7;
  EXPECT_FALSE(loom_low_schedule_class_query_completion_wait(
      &descriptor_set_, &schedule_class_, &wait_cycles));
  EXPECT_EQ(wait_cycles, 0u);
}

TEST_F(ScheduleCompletionWaitTest, ZeroCostCounterStillRequiresCompletion) {
  hazards_[0].kind = LOOM_LOW_HAZARD_KIND_WAIT_COUNTER;
  hazards_[0].reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER;
  hazards_[0].distance = 0;
  schedule_class_.hazard_count = 1;

  uint16_t wait_cycles = 7;
  EXPECT_TRUE(loom_low_schedule_class_query_completion_wait(
      &descriptor_set_, &schedule_class_, &wait_cycles));
  EXPECT_EQ(wait_cycles, 0u);
}

TEST_F(ScheduleCompletionWaitTest, UsesCounterCostInScheduleClassHazardSlice) {
  hazards_[0].kind = LOOM_LOW_HAZARD_KIND_WAIT_COUNTER;
  hazards_[0].reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER;
  hazards_[0].distance = 11;
  hazards_[1].kind = LOOM_LOW_HAZARD_KIND_MIN_DISTANCE;
  hazards_[1].reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE;
  hazards_[1].distance = 13;
  hazards_[2].kind = LOOM_LOW_HAZARD_KIND_WAIT_COUNTER;
  hazards_[2].reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER;
  hazards_[2].distance = 5;
  schedule_class_.hazard_start = 1;
  schedule_class_.hazard_count = 2;

  uint16_t wait_cycles = 0;
  EXPECT_TRUE(loom_low_schedule_class_query_completion_wait(
      &descriptor_set_, &schedule_class_, &wait_cycles));
  EXPECT_EQ(wait_cycles, 5u);
}

}  // namespace
}  // namespace loom
