// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/event_frontier.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class ScheduleEventFrontierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    descriptors_ = loom_test_low_core_descriptor_set();
    IREE_ASSERT_OK(loom_low_schedule_event_frontier_initialize(
        descriptors_, &arena_, &frontier_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  template <typename T>
  uint16_t NamedIndex(const T* rows, uint32_t count, const char* name) {
    for (uint32_t i = 0; i < count; ++i) {
      if (iree_string_view_equal(loom_low_descriptor_set_string(
                                     descriptors_, rows[i].name_string_ref),
                                 iree_make_cstring_view(name))) {
        return (uint16_t)i;
      }
    }
    ADD_FAILURE() << "Missing test fixture row: " << name;
    return 0;
  }

  uint16_t Register(const char* name) {
    return NamedIndex(descriptors_->physical_registers,
                      descriptors_->physical_register_count, name);
  }

  uint16_t Event(const char* name) {
    return NamedIndex(descriptors_->timing_events,
                      descriptors_->timing_event_count, name);
  }

  iree_arena_block_pool_t pool_ = {};
  iree_arena_allocator_t arena_ = {};
  const loom_low_descriptor_set_t* descriptors_ = nullptr;
  loom_low_schedule_event_frontier_t frontier_ = {};
};

TEST_F(ScheduleEventFrontierTest, AliasesSharePendingReadAndWriteEvents) {
  const auto wide = Register("test.l0");
  const auto low = Register("test.r0");
  const auto high = Register("test.r2");
  const auto disjoint = Register("test.r1");
  const auto write = Event("test.state.write");
  const auto read = Event("test.state.read");
  IREE_ASSERT_OK(
      loom_low_schedule_event_frontier_commit(&frontier_, wide, write, 10));
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, low, read), 13u);
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, high, read),
            13u);
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, disjoint, read),
            0u);
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, low, write),
            11u);
  const auto late_read = Event("test.read.late");
  const auto fast_write = Event("test.write.fast");
  IREE_ASSERT_OK(
      loom_low_schedule_event_frontier_commit(&frontier_, high, late_read, 20));
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, wide, fast_write),
      23u);
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, low, fast_write),
            0u);
}

TEST_F(ScheduleEventFrontierTest, NewerEventsDoNotEraseOlderPendingAccesses) {
  const auto physical = Register("test.r0");
  const auto slow_write = Event("test.write.slow");
  const auto fast_write = Event("test.write.fast");
  const auto early_read = Event("test.read.early");
  const auto late_read = Event("test.read.late");
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         slow_write, 10));
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         fast_write, 11));
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, early_read),
      13u);
  // The signed bound permits native issue before the published writer without
  // dropping the producer's timing obligation.
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, late_read),
      9u);
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         late_read, 20));
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         early_read, 21));
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, fast_write),
      23u);
  EXPECT_EQ(frontier_.quiescent_cycle, 23u);
}

TEST_F(ScheduleEventFrontierTest, NonpositiveDeadlinesRetainOrderAndSaturate) {
  const auto physical = Register("test.r0");
  const auto fast_write = Event("test.write.fast");
  const auto early_read = Event("test.read.early");
  const auto late_read = Event("test.read.late");
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         fast_write, 1));
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, early_read),
      1u);
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, late_read),
      0u);
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         fast_write, 10));
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, early_read),
      10u);
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, late_read),
      8u);
}

TEST_F(ScheduleEventFrontierTest, DelayedOverwriteKeepsOldAndNewValueReads) {
  // The storage write occurs at stage seven; reads and immediate writes occur
  // at stage one. An old read may share the delayed replacement's write cycle,
  // while a new-value read must occur on the following cycle.
  constexpr uint16_t kDelayedWrite = 0;
  constexpr uint16_t kRead = 1;
  constexpr uint16_t kImmediateWrite = 2;
  const loom_low_event_separation_t separations[] = {
      {kDelayedWrite, kDelayedWrite, 1, LOOM_LOW_MODEL_QUALITY_EXACT},
      {kDelayedWrite, kRead, 7, LOOM_LOW_MODEL_QUALITY_EXACT},
      {kDelayedWrite, kImmediateWrite, 7, LOOM_LOW_MODEL_QUALITY_EXACT},
      {kRead, kDelayedWrite, -6, LOOM_LOW_MODEL_QUALITY_EXACT},
      {kRead, kImmediateWrite, 0, LOOM_LOW_MODEL_QUALITY_EXACT},
      {kImmediateWrite, kDelayedWrite, -5, LOOM_LOW_MODEL_QUALITY_EXACT},
      {kImmediateWrite, kRead, 1, LOOM_LOW_MODEL_QUALITY_EXACT},
      {kImmediateWrite, kImmediateWrite, 1, LOOM_LOW_MODEL_QUALITY_EXACT},
  };
  loom_low_timing_event_t events[3] = {};
  events[kDelayedWrite].separation_count = 3;
  events[kDelayedWrite].maximum_issue_separation_cycles = 7;
  events[kRead].separation_start = 3;
  events[kRead].separation_count = 2;
  events[kImmediateWrite].separation_start = 5;
  events[kImmediateWrite].separation_count = 3;
  events[kImmediateWrite].maximum_issue_separation_cycles = 1;
  loom_low_descriptor_set_t descriptors = *descriptors_;
  descriptors.timing_events = events;
  descriptors.timing_event_count = IREE_ARRAYSIZE(events);
  descriptors.event_separations = separations;
  descriptors.event_separation_count = IREE_ARRAYSIZE(separations);
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_initialize(
      &descriptors, &arena_, &frontier_));
  const auto physical = Register("test.r0");
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         kDelayedWrite, 0));
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, physical, kRead),
            7u);
  IREE_ASSERT_OK(
      loom_low_schedule_event_frontier_commit(&frontier_, physical, kRead, 7));

  // Semantic publication is old write, old read, new write. Only the new
  // delayed write may issue at one; an immediate overwrite must wait to seven.
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, physical,
                                                   kDelayedWrite),
            1u);
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, physical,
                                                   kImmediateWrite),
            7u);
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         kDelayedWrite, 1));
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, physical, kRead),
            8u);
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, physical,
                                                   kImmediateWrite),
            8u);
  EXPECT_EQ(frontier_.quiescent_cycle, 8u);
}

TEST_F(ScheduleEventFrontierTest, FixedStorageAcrossLargeCyclesAndOverflow) {
  const auto physical = Register("test.r0");
  const auto write = Event("test.state.write");
  const auto read = Event("test.state.read");
  const auto used_bytes = arena_.used_allocation_size;
  const auto owned_bytes = arena_.total_allocation_size;
  constexpr uint32_t kCycle = 1000000000;
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         write, kCycle));
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, physical, read),
            kCycle + 3);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_low_schedule_event_frontier_commit(
                            &frontier_, physical, write, UINT32_MAX));
  EXPECT_EQ(loom_low_schedule_event_frontier_query(&frontier_, physical, read),
            kCycle + 3);
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
  EXPECT_EQ(arena_.total_allocation_size, owned_bytes);
}

}  // namespace
}  // namespace loom
