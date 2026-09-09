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
                                     descriptors_, rows[i].name_string_offset),
                                 iree_make_cstring_view(name)))
        return (uint16_t)i;
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
  // The negative fast-write/late-read separation imposes no extra issue gap.
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, late_read),
      0u);
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         late_read, 20));
  IREE_ASSERT_OK(loom_low_schedule_event_frontier_commit(&frontier_, physical,
                                                         early_read, 21));
  EXPECT_EQ(
      loom_low_schedule_event_frontier_query(&frontier_, physical, fast_write),
      23u);
  EXPECT_EQ(frontier_.quiescent_cycle, 23u);
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
