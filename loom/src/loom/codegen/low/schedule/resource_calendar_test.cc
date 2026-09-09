// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/resource_calendar.h"

#include <algorithm>
#include <cstdint>
#include <vector>

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
    const loom_low_descriptor_view_t* descriptor_view =
        loom_low_descriptor_set_descriptor_view_at(descriptor_set_,
                                                   descriptor_ref);
    return &descriptor_set_
                ->schedule_classes[descriptor_view->schedule_class_id];
  }

  uint32_t FindEarliest(const loom_low_schedule_class_t* schedule_class,
                        uint32_t proposed_issue_cycle,
                        uint16_t* out_bottleneck_resource_id = nullptr) {
    uint16_t bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE;
    const uint32_t issue_cycle =
        loom_low_schedule_resource_calendar_find_earliest_issue_cycle(
            &calendar_, &schedule_class, 1, proposed_issue_cycle,
            &bottleneck_resource_id);
    if (out_bottleneck_resource_id != nullptr) {
      *out_bottleneck_resource_id = bottleneck_resource_id;
    }
    return issue_cycle;
  }

  void Commit(const loom_low_schedule_class_t* schedule_class,
              uint32_t issue_cycle) {
    IREE_ASSERT_OK(loom_low_schedule_resource_calendar_commit(
        &calendar_, &schedule_class, 1, issue_cycle));
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

TEST_F(ScheduleResourceCalendarTest,
       ReservationsOverlapButExcludeRequiredUses) {
  const loom_low_schedule_class_t* required =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_RESOURCE_REQUIRED_I32);
  const loom_low_schedule_class_t* reserved =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_RESOURCE_RESERVED_I32);

  Commit(reserved, 0);
  EXPECT_EQ(FindEarliest(reserved, 0), 0u);
  Commit(reserved, 0);
  EXPECT_EQ(FindEarliest(required, 0), 0u);
  Commit(required, 0);
  EXPECT_EQ(FindEarliest(required, 0), 1u);

  loom_low_schedule_resource_calendar_reset(&calendar_);
  Commit(required, 0);
  Commit(required, 0);
  EXPECT_EQ(FindEarliest(reserved, 0), 1u);
}

TEST_F(ScheduleResourceCalendarTest, LargeIssueCycleReusesFixedStorage) {
  const loom_low_schedule_class_t* fast =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32);

  const iree_host_size_t used_bytes = arena_.used_allocation_size;
  const iree_host_size_t owned_bytes = arena_.total_allocation_size;
  const auto* slots = calendar_.slots;
  Commit(fast, 0);
  constexpr uint32_t kLargeIssueCycle = 1000000000u;
  EXPECT_EQ(FindEarliest(fast, kLargeIssueCycle), kLargeIssueCycle);
  Commit(fast, kLargeIssueCycle);

  EXPECT_EQ(FindEarliest(fast, kLargeIssueCycle), kLargeIssueCycle);
  Commit(fast, kLargeIssueCycle);
  EXPECT_EQ(FindEarliest(fast, kLargeIssueCycle), kLargeIssueCycle + 2);
  EXPECT_EQ(calendar_.slots, slots);
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
  EXPECT_EQ(arena_.total_allocation_size, owned_bytes);
}

TEST_F(ScheduleResourceCalendarTest, OccupancyMayIncludeTheFinalCycle) {
  const auto* required =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_RESOURCE_REQUIRED_I32);
  EXPECT_EQ(FindEarliest(required, UINT32_MAX), UINT32_MAX);
  Commit(required, UINT32_MAX);
  // The all-ones cycle is real occupancy, not an uninitialized-slot sentinel.
  Commit(required, UINT32_MAX);
  const auto& use = descriptor_set_->issue_uses[required->issue_use_start];
  const auto& resource = descriptor_set_->resources[use.resource_id];
  const auto& slot =
      calendar_.slots[resource.calendar.slot_start +
                      (UINT32_MAX & resource.calendar.slot_mask)];
  EXPECT_EQ(slot.issue_cycle, UINT32_MAX);
  EXPECT_EQ(slot.occupancy.required_units, resource.capacity_per_cycle);
  loom_low_schedule_resource_calendar_reset(&calendar_);
  EXPECT_EQ(FindEarliest(required, 0), 0u);
}

TEST_F(ScheduleResourceCalendarTest, RejectsAnUnrepresentableResourceStage) {
  const auto* fast =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32);
  EXPECT_EQ(FindEarliest(fast, UINT32_MAX), UINT32_MAX);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_low_schedule_resource_calendar_commit(
                            &calendar_, &fast, 1, UINT32_MAX));
}

TEST_F(ScheduleResourceCalendarTest, AdmitsCollectiveInstructionDemand) {
  const auto* required =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_RESOURCE_SERIAL_I32);
  const loom_low_schedule_class_t* pair[] = {required, required};
  // Each instruction alone fills the resource. Delaying the pair cannot make
  // their simultaneous demand legal, so packet formation must split them.
  EXPECT_TRUE(loom_low_schedule_resource_group_fits(descriptor_set_, pair, 1));
  EXPECT_FALSE(loom_low_schedule_resource_group_fits(descriptor_set_, pair, 2));
  const auto* fast =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32);
  const auto* slow =
      ScheduleClass(TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_SLOW_I32);
  const loom_low_schedule_class_t* group[] = {fast, slow};
  ASSERT_TRUE(loom_low_schedule_resource_group_fits(descriptor_set_, group, 2));
  uint16_t bottleneck = LOOM_LOW_RESOURCE_NONE;
  const uint32_t cycle =
      loom_low_schedule_resource_calendar_find_earliest_issue_cycle(
          &calendar_, group, 2, 0, &bottleneck);
  EXPECT_EQ(cycle, 0u);
  IREE_ASSERT_OK(
      loom_low_schedule_resource_calendar_commit(&calendar_, group, 2, cycle));
  EXPECT_GT(calendar_.quiescent_cycle, cycle);
}

TEST_F(ScheduleResourceCalendarTest, MatchesDenseOccupancyAcrossRingWraps) {
  // The oracle records absolute cycles independently for every resource. It
  // derives contention from the authored group IDs, without using generated
  // ring offsets, masks, or the production occupancy queries.
  constexpr uint32_t kCycleCount = 4096;
  const uint32_t descriptors[] = {
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_FAST_I32,
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_SLOW_I32,
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_EVENT_CONSUME_EARLY_I32,
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_RESOURCE_REQUIRED_I32,
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_RESOURCE_RESERVED_I32,
      TEST_LOW_CORE_DESCRIPTOR_REF_TEST_ADD_I32,
  };
  using Occupancy = loom_low_schedule_resource_occupancy_t;
  const uint32_t resource_count = descriptor_set_->resource_count;
  std::vector<Occupancy> occupied(resource_count * kCycleCount);
  std::vector<Occupancy> candidate(resource_count * kCycleCount);
  const auto accumulate = [](Occupancy& target, const Occupancy& use) {
    target.required_units += use.required_units;
    target.reserved_units = std::max(target.reserved_units, use.reserved_units);
  };
  const iree_host_size_t used_bytes = arena_.used_allocation_size;
  const iree_host_size_t owned_bytes = arena_.total_allocation_size;
  uint32_t issue_cycle = 0;
  uint32_t random = 42;
  for (uint32_t step = 0; step < 256; ++step) {
    random = random * 1664525u + 1013904223u;
    const auto* schedule_class =
        ScheduleClass(descriptors[random % IREE_ARRAYSIZE(descriptors)]);
    std::fill(candidate.begin(), candidate.end(), Occupancy{});
    uint32_t duration = 0;
    for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
      const auto& use =
          descriptor_set_->issue_uses[schedule_class->issue_use_start + i];
      const auto& resource = descriptor_set_->resources[use.resource_id];
      const uint32_t end = use.stage + use.cycles;
      duration = std::max(duration, end);
      for (uint32_t resource_id = 0; resource_id < resource_count;
           ++resource_id) {
        if (resource_id != use.resource_id &&
            (resource.contention_group_id == 0 ||
             descriptor_set_->resources[resource_id].contention_group_id !=
                 resource.contention_group_id)) {
          continue;
        }
        Occupancy demand = {};
        if (use.kind == LOOM_LOW_ISSUE_USE_KIND_REQUIRED) {
          demand.required_units = use.units;
        } else {
          demand.reserved_units = use.units;
        }
        for (uint32_t cycle = use.stage; cycle < end; ++cycle) {
          accumulate(candidate[resource_id * kCycleCount + cycle], demand);
        }
      }
    }
    // Include idle gaps, same-cycle issue, and queries which must not reserve
    // anything until their chosen cycle is committed.
    const uint32_t proposed = issue_cycle + ((random >> 24) % 4);
    uint32_t expected = proposed;
    for (;; ++expected) {
      ASSERT_LT(expected + duration, kCycleCount);
      bool fits = true;
      for (uint32_t resource_id = 0; resource_id < resource_count;
           ++resource_id) {
        for (uint32_t cycle = 0; cycle < duration; ++cycle) {
          Occupancy combined =
              occupied[resource_id * kCycleCount + expected + cycle];
          accumulate(combined, candidate[resource_id * kCycleCount + cycle]);
          fits &= static_cast<uint32_t>(combined.required_units) +
                      combined.reserved_units <=
                  descriptor_set_->resources[resource_id].capacity_per_cycle;
        }
      }
      if (fits) break;
    }
    ASSERT_EQ(FindEarliest(schedule_class, proposed), expected)
        << "step " << step;
    EXPECT_EQ(FindEarliest(schedule_class, proposed), expected);
    Commit(schedule_class, expected);
    for (uint32_t resource_id = 0; resource_id < resource_count;
         ++resource_id) {
      for (uint32_t cycle = 0; cycle < duration; ++cycle) {
        accumulate(occupied[resource_id * kCycleCount + expected + cycle],
                   candidate[resource_id * kCycleCount + cycle]);
      }
    }
    issue_cycle = expected;
  }
  EXPECT_EQ(arena_.used_allocation_size, used_bytes);
  EXPECT_EQ(arena_.total_allocation_size, owned_bytes);
}

}  // namespace
}  // namespace loom
