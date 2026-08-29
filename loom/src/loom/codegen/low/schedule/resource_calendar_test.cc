// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/resource_calendar.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class ScheduleResourceCalendarTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(128 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    descriptor_set_ = loom_test_low_core_descriptor_set();
    IREE_ASSERT_OK(loom_low_schedule_resource_calendar_initialize(
        descriptor_set_, &arena_, &calendar_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  const loom_low_schedule_class_t* ScheduleClass(uint32_t descriptor_ref) {
    const loom_low_descriptor_t* descriptor =
        &descriptor_set_->descriptors[descriptor_ref];
    const loom_low_descriptor_view_t* descriptor_view =
        loom_low_descriptor_set_descriptor_view(descriptor_set_, descriptor);
    const uint16_t schedule_class_id = descriptor_view->schedule_class_id;
    return &descriptor_set_->schedule_classes[schedule_class_id];
  }

  uint32_t FindEarliest(const loom_low_schedule_class_t* schedule_class,
                        uint32_t proposed_issue_cycle,
                        uint16_t* out_bottleneck_resource_id = nullptr) {
    uint16_t bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE;
    const uint32_t issue_cycle =
        loom_low_schedule_resource_calendar_find_earliest_issue_cycle(
            &calendar_, schedule_class, proposed_issue_cycle,
            &bottleneck_resource_id);
    if (out_bottleneck_resource_id != nullptr) {
      *out_bottleneck_resource_id = bottleneck_resource_id;
    }
    return issue_cycle;
  }

  void Commit(const loom_low_schedule_class_t* schedule_class,
              uint32_t issue_cycle) {
    IREE_ASSERT_OK(loom_low_schedule_resource_calendar_commit(
        &calendar_, schedule_class, issue_cycle));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_low_schedule_resource_calendar_t calendar_ = {};
};

TEST_F(ScheduleResourceCalendarTest,
       SharesCapacityAcrossResourcesStagesAndCycles) {
  const loom_low_schedule_class_t* fast =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32);
  const loom_low_schedule_class_t* slow =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_SLOW_I32);
  const loom_low_schedule_class_t* consumer =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_CONSUME_EARLY_I32);

  EXPECT_EQ(FindEarliest(fast, 0), 0u);
  Commit(fast, 0);
  EXPECT_EQ(FindEarliest(slow, 0), 0u);
  Commit(slow, 0);
  EXPECT_EQ(FindEarliest(consumer, 0), 0u);
  Commit(consumer, 0);

  uint16_t bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE;
  EXPECT_EQ(FindEarliest(consumer, 0, &bottleneck_resource_id), 2u);
  EXPECT_NE(bottleneck_resource_id, LOOM_LOW_RESOURCE_NONE);
  Commit(consumer, 2);
  EXPECT_EQ(FindEarliest(consumer, 2), 2u);
  Commit(consumer, 2);
  EXPECT_EQ(FindEarliest(consumer, 2), 3u);
}

TEST_F(ScheduleResourceCalendarTest, ResetRetainsAnEmptyCalendar) {
  const loom_low_schedule_class_t* fast =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32);
  const loom_low_schedule_class_t* slow =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_SLOW_I32);
  const loom_low_schedule_class_t* consumer =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_CONSUME_EARLY_I32);

  Commit(fast, 0);
  Commit(slow, 0);
  Commit(consumer, 0);
  loom_low_schedule_resource_calendar_reset(&calendar_);

  EXPECT_EQ(FindEarliest(fast, 0), 0u);
  EXPECT_EQ(FindEarliest(slow, 0), 0u);
  EXPECT_EQ(FindEarliest(consumer, 0), 0u);
}

TEST_F(ScheduleResourceCalendarTest, LargeIssueCycleUsesACompactRollingWindow) {
  const loom_low_schedule_class_t* fast =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32);

  Commit(fast, 0);
  constexpr uint32_t kLargeIssueCycle = 1000000000u;
  EXPECT_EQ(FindEarliest(fast, kLargeIssueCycle), kLargeIssueCycle);
  Commit(fast, kLargeIssueCycle);

  const uint16_t resource_id =
      descriptor_set_->issue_uses[fast->issue_use_start].resource_id;
  const uint16_t row_index = calendar_.resource_row_indices[resource_id];
  const loom_low_schedule_resource_calendar_row_t* row =
      &calendar_.rows[row_index];
  EXPECT_EQ(row->base_issue_cycle, kLargeIssueCycle);
  EXPECT_EQ(row->occupied_cycle_count, 2u);
  EXPECT_LE(row->occupied_cycle_capacity, 16u);
}

}  // namespace
}  // namespace loom
