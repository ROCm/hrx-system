// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/write_interference_events.h"

#include <initializer_list>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

void ExpectRetained(const loom_low_write_events_t& events, uint32_t point,
                    std::initializer_list<loom_low_write_retained_t> expected) {
  SCOPED_TRACE(point);
  const auto range = events.points[point].retained;
  ASSERT_EQ(range.count, expected.size());
  uint32_t index = range.start;
  for (const auto& wanted : expected) {
    const auto& actual = events.retained[index++];
    EXPECT_EQ(actual.value, wanted.value);
    EXPECT_EQ(actual.offset, wanted.offset);
    EXPECT_EQ(actual.count, wanted.count);
  }
}

class WriteInterferenceEventsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &scratch_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Blocks shared by the persistent results and disposable solver workspace.
  iree_arena_block_pool_t pool_;
  // Event accesses and immutable snapshots.
  iree_arena_allocator_t arena_;
  // Decoder and dataflow state, reset before querying snapshots.
  iree_arena_allocator_t scratch_;
};

TEST_F(WriteInterferenceEventsTest, PartialOverwriteReleasesOnlyWrittenUnits) {
  for (uint32_t overwritten = 0; overwritten < 2; ++overwritten) {
    SCOPED_TRACE(overwritten);
    const loom_low_write_value_t values[] = {
        {
            .origin = 0,
            .fixed_base = 0,
            .construction = {.width = 2, .retained_start = 0},
        },
        {
            .origin = 1,
            .fixed_base = overwritten,
            .construction = {.width = 1, .retained_start = UINT32_MAX},
        },
        {
            .origin = 2,
            .fixed_base = UINT32_MAX,
            .construction = {.width = 1, .retained_start = UINT32_MAX},
        },
    };
    loom_low_write_event_t points[9] = {};
    for (auto& point : points) {
      point.access = UINT32_MAX;
    }
    loom_low_write_events_t events = {
        .points = points,
        .point_count = IREE_ARRAYSIZE(points),
    };
    IREE_ASSERT_OK(
        loom_low_write_events_append(&events, 1,
                                     {
                                         .value = 0,
                                         .source = LOOM_VALUE_ORDINAL_INVALID,
                                         .count = 2,
                                         .flags = LOOM_LOW_WRITE_ACCESS_RETAIN,
                                     },
                                     &arena_));
    // A memory result overwrites one known unit, without a hazardous ALU write.
    IREE_ASSERT_OK(
        loom_low_write_events_append(&events, 4,
                                     {
                                         .value = 1,
                                         .source = LOOM_VALUE_ORDINAL_INVALID,
                                         .count = 1,
                                         .flags = LOOM_LOW_WRITE_ACCESS_WRITE,
                                     },
                                     &arena_));
    IREE_ASSERT_OK(loom_low_write_events_append(
        &events, 7,
        {
            .value = 2,
            .source = LOOM_VALUE_ORDINAL_INVALID,
            .count = 1,
            .flags =
                LOOM_LOW_WRITE_ACCESS_WRITE | LOOM_LOW_WRITE_ACCESS_INTERFERE,
        },
        &arena_));
    loom_low_write_flow_block_t block = {.begin = 1, .end = 9};
    uint32_t worklist;
    const loom_low_write_flow_t flow = {
        .blocks = &block,
        .block_count = 1,
        .worklist = &worklist,
    };
    iree_host_size_t constraint_count = 0;
    IREE_ASSERT_OK(loom_low_write_events_solve(
        &events, values, IREE_ARRAYSIZE(values), 2, &flow, &arena_, &scratch_,
        &constraint_count));
    iree_arena_reset(&scratch_);

    EXPECT_EQ(constraint_count, 1u);
    EXPECT_EQ(points[1].retained.count, 0u);
    ASSERT_EQ(points[4].retained.count, 1u);
    const auto& before = events.retained[points[4].retained.start];
    EXPECT_EQ(before.value, 0u);
    EXPECT_EQ(before.offset, 0u);
    EXPECT_EQ(before.count, 2u);
    ASSERT_EQ(points[5].retained.count, 1u);
    const auto& after = events.retained[points[5].retained.start];
    EXPECT_EQ(after.value, 0u);
    EXPECT_EQ(after.offset, 1u - overwritten);
    EXPECT_EQ(after.count, 1u);
    // Vector-only gaps and writes at unknown locations reuse the snapshot.
    EXPECT_EQ(points[2].retained.start, points[4].retained.start);
    EXPECT_EQ(points[5].retained.start, points[8].retained.start);
    EXPECT_EQ(events.retained_count, 2u);

    const loom_low_allocation_assignment_t assignments[] = {
        {
            .unit_count = 1,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = overwritten,
            .location_count = 1,
        },
        {
            .unit_count = 2,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 0,
            .location_count = 2,
        },
    };
    const uint32_t assignment_indices[] = {1, 0, UINT32_MAX};
    for (uint32_t location = 0; location < 3; ++location) {
      EXPECT_EQ(loom_low_write_events_temporary_conflicts(
                    &events, assignments, assignment_indices, 4, location),
                location < 2);
      EXPECT_EQ(loom_low_write_events_temporary_conflicts(
                    &events, assignments, assignment_indices, 5, location),
                location == 1u - overwritten);
    }
  }
}

TEST_F(WriteInterferenceEventsTest, ResetsBeforeWritesAndRetainsAfterWrites) {
  for (bool read_first : {false, true}) {
    SCOPED_TRACE(read_first);
    const loom_low_write_value_t values[] = {
        {
            .origin = 0,
            .fixed_base = 0,
            .construction = {.width = 2, .retained_start = 0},
        },
        {
            .origin = 1,
            .fixed_base = 2,
            .construction = {.width = 2, .retained_start = 2},
        },
    };
    loom_low_write_event_t points[8] = {};
    for (auto& point : points) {
      point.access = UINT32_MAX;
    }
    points[3].reset = true;
    points[7].reset = true;
    loom_low_write_events_t events = {
        .points = points,
        .point_count = IREE_ARRAYSIZE(points),
    };
    const loom_low_write_access_t old_read = {
        .value = 0,
        .source = LOOM_VALUE_ORDINAL_INVALID,
        .count = 2,
        .flags = LOOM_LOW_WRITE_ACCESS_RETAIN,
    };
    const loom_low_write_access_t new_read = {
        .value = 1,
        .source = LOOM_VALUE_ORDINAL_INVALID,
        .count = 2,
        .flags = LOOM_LOW_WRITE_ACCESS_RETAIN,
    };
    auto write = old_read;
    write.flags = LOOM_LOW_WRITE_ACCESS_WRITE | LOOM_LOW_WRITE_ACCESS_INTERFERE;
    IREE_ASSERT_OK(loom_low_write_events_append(&events, 1, old_read, &arena_));
    // The replacing reader also writes the old location. Its reset makes that
    // write safe before the new predicate becomes retained.
    IREE_ASSERT_OK(loom_low_write_events_append(&events, 3, write, &arena_));
    IREE_ASSERT_OK(loom_low_write_events_append(&events, 3, new_read, &arena_));
    // A tied read/write must retain the new observation regardless of the
    // producer's operand enumeration order.
    write.value = 1;
    IREE_ASSERT_OK(loom_low_write_events_append(
        &events, 5, read_first ? new_read : write, &arena_));
    IREE_ASSERT_OK(loom_low_write_events_append(
        &events, 5, read_first ? write : new_read, &arena_));
    loom_low_write_flow_block_t block = {.begin = 1, .end = 8};
    uint32_t worklist;
    const loom_low_write_flow_t flow = {
        .blocks = &block,
        .block_count = 1,
        .worklist = &worklist,
    };
    iree_host_size_t constraint_count = 0;
    IREE_ASSERT_OK(loom_low_write_events_solve(
        &events, values, IREE_ARRAYSIZE(values), 4, &flow, &arena_, &scratch_,
        &constraint_count));
    iree_arena_reset(&scratch_);
    EXPECT_EQ(constraint_count, 1u);
    ExpectRetained(events, 1, {});
    ExpectRetained(events, 2, {{0, 0, 2}});
    ExpectRetained(events, 3, {});
    ExpectRetained(events, 4, {{1, 0, 2}});
    ExpectRetained(events, 5, {{1, 0, 2}});
    ExpectRetained(events, 6, {{1, 0, 2}});
    ExpectRetained(events, 7, {});
  }
}

TEST_F(WriteInterferenceEventsTest, CopiesClearOnlyProvedNonidentityWrites) {
  struct CopyCase {
    // Physical destination base, or UINT32_MAX when not yet assigned.
    uint32_t destination_base;
    // Canonical source: one is the destination itself, two is independent.
    uint32_t source;
    // Fixed base of the independent source, or UINT32_MAX when unknown.
    uint32_t source_base;
    // First destination unit in the copy.
    uint32_t destination_offset;
    // First source unit in the copy.
    uint32_t source_offset;
    // Consecutive units transferred by the copy.
    uint32_t count;
    // Expected retained prefix of the two-unit value at physical base four.
    uint32_t remaining;
  };
  const CopyCase cases[] = {
      {4, 2, UINT32_MAX, 1, 0, 1, 2},  // Source may still coalesce.
      {4, 2, 4, 1, 1, 1, 2},           // Same physical source unit.
      {4, 2, 6, 1, 0, 1, 1},           // Different physical source unit.
      {4, 1, UINT32_MAX, 1, 1, 1, 2},  // Same canonical unit.
      {4, 1, UINT32_MAX, 1, 0, 1, 1},  // Different unit of the same origin.
      {UINT32_MAX, 2, 4, 1, 0, 1, 2},  // Destination may still coalesce.
      {3, 2, 6, 2, 0, 1, 1},           // Destination begins before the mask.
      {4, 2, 6, 0, 0, 2, 0},           // Full two-unit overwrite.
      {2, 2, 6, 0, 0, 4, 0},           // Only the upper half overlaps the mask.
  };
  for (const auto& copy : cases) {
    SCOPED_TRACE(&copy - cases);
    const loom_low_write_value_t values[] = {
        {
            .origin = 0,
            .fixed_base = 4,
            .construction = {.width = 2, .retained_start = 0},
        },
        {
            .origin = 1,
            .fixed_base = copy.destination_base,
            .construction = {.width = 4, .retained_start = UINT32_MAX},
        },
        {
            .origin = 2,
            .fixed_base = copy.source_base,
            .construction = {.width = 4, .retained_start = UINT32_MAX},
        },
    };
    loom_low_write_event_t points[4] = {};
    for (auto& point : points) {
      point.access = UINT32_MAX;
    }
    loom_low_write_events_t events = {
        .points = points,
        .point_count = IREE_ARRAYSIZE(points),
    };
    IREE_ASSERT_OK(
        loom_low_write_events_append(&events, 1,
                                     {
                                         .value = 0,
                                         .source = LOOM_VALUE_ORDINAL_INVALID,
                                         .count = 2,
                                         .flags = LOOM_LOW_WRITE_ACCESS_RETAIN,
                                     },
                                     &arena_));
    IREE_ASSERT_OK(loom_low_write_events_append(
        &events, 2,
        {
            .value = 1,
            .source = copy.source,
            .offset = copy.destination_offset,
            .source_offset = copy.source_offset,
            .count = copy.count,
            .flags =
                LOOM_LOW_WRITE_ACCESS_WRITE | LOOM_LOW_WRITE_ACCESS_INTERFERE,
        },
        &arena_));
    loom_low_write_flow_block_t block = {.begin = 1, .end = 4};
    uint32_t worklist;
    const loom_low_write_flow_t flow = {
        .blocks = &block,
        .block_count = 1,
        .worklist = &worklist,
    };
    iree_host_size_t constraint_count = 0;
    IREE_ASSERT_OK(loom_low_write_events_solve(
        &events, values, IREE_ARRAYSIZE(values), 2, &flow, &arena_, &scratch_,
        &constraint_count));
    iree_arena_reset(&scratch_);
    EXPECT_EQ(constraint_count, copy.count);
    ExpectRetained(events, 2, {{0, 0, 2}});
    if (copy.remaining == 0) {
      ExpectRetained(events, 3, {});
    } else {
      ExpectRetained(events, 3, {{0, 0, copy.remaining}});
    }
  }
}

TEST_F(WriteInterferenceEventsTest, JoinsPredecessorsAndPropagatesBackedges) {
  const loom_low_write_value_t values[] = {
      {
          .origin = 0,
          .fixed_base = 0,
          .construction = {.width = 2, .retained_start = 0},
      },
      {
          .origin = 1,
          .fixed_base = 0,
          .construction = {.width = 2, .retained_start = UINT32_MAX},
      },
      {
          .origin = 2,
          .fixed_base = 2,
          .construction = {.width = 1, .retained_start = 2},
      },
      {
          .origin = 3,
          .fixed_base = UINT32_MAX,
          .construction = {.width = 1, .retained_start = UINT32_MAX},
      },
  };
  loom_low_write_event_t points[11] = {};
  for (auto& point : points) {
    point.access = UINT32_MAX;
  }
  points[9].reset = true;
  loom_low_write_events_t events = {
      .points = points,
      .point_count = IREE_ARRAYSIZE(points),
  };
  IREE_ASSERT_OK(
      loom_low_write_events_append(&events, 1,
                                   {
                                       .value = 0,
                                       .source = LOOM_VALUE_ORDINAL_INVALID,
                                       .count = 2,
                                       .flags = LOOM_LOW_WRITE_ACCESS_RETAIN,
                                   },
                                   &arena_));
  for (uint32_t unit = 0; unit < 2; ++unit) {
    IREE_ASSERT_OK(
        loom_low_write_events_append(&events, 3 + 2 * unit,
                                     {
                                         .value = 1,
                                         .source = LOOM_VALUE_ORDINAL_INVALID,
                                         .offset = unit,
                                         .count = 1,
                                         .flags = LOOM_LOW_WRITE_ACCESS_WRITE,
                                     },
                                     &arena_));
  }
  IREE_ASSERT_OK(loom_low_write_events_append(
      &events, 7,
      {
          .value = 3,
          .source = LOOM_VALUE_ORDINAL_INVALID,
          .count = 1,
          .flags =
              LOOM_LOW_WRITE_ACCESS_WRITE | LOOM_LOW_WRITE_ACCESS_INTERFERE,
      },
      &arena_));
  IREE_ASSERT_OK(
      loom_low_write_events_append(&events, 8,
                                   {
                                       .value = 2,
                                       .source = LOOM_VALUE_ORDINAL_INVALID,
                                       .count = 1,
                                       .flags = LOOM_LOW_WRITE_ACCESS_RETAIN,
                                   },
                                   &arena_));
  // The two arms overwrite opposite halves. Their join retains the union,
  // then a new read flows back to the first arm and forward to a resetting
  // exit.
  loom_low_write_flow_block_t blocks[] = {
      {.begin = 1, .end = 3, .successor_start = 0, .successor_count = 2},
      {.begin = 3, .end = 5, .successor_start = 2, .successor_count = 1},
      {.begin = 5, .end = 7, .successor_start = 3, .successor_count = 1},
      {.begin = 7, .end = 9, .successor_start = 4, .successor_count = 2},
      {.begin = 9, .end = 11, .successor_start = 6, .successor_count = 0},
  };
  uint32_t successors[] = {1, 2, 3, 3, 1, 4};
  uint32_t worklist[IREE_ARRAYSIZE(blocks)];
  const loom_low_write_flow_t flow = {
      .blocks = blocks,
      .block_count = IREE_ARRAYSIZE(blocks),
      .successors = successors,
      .worklist = worklist,
  };
  iree_host_size_t constraint_count = 0;
  IREE_ASSERT_OK(loom_low_write_events_solve(
      &events, values, IREE_ARRAYSIZE(values), 3, &flow, &arena_, &scratch_,
      &constraint_count));
  iree_arena_reset(&scratch_);
  EXPECT_EQ(constraint_count, 2u);
  ExpectRetained(events, 2, {{0, 0, 2}});
  ExpectRetained(events, 3, {{0, 0, 2}, {2, 0, 1}});
  ExpectRetained(events, 4, {{0, 1, 1}, {2, 0, 1}});
  ExpectRetained(events, 5, {{0, 0, 2}});
  ExpectRetained(events, 6, {{0, 0, 1}});
  ExpectRetained(events, 7, {{0, 0, 2}, {2, 0, 1}});
  ExpectRetained(events, 9, {});
  ExpectRetained(events, 10, {});
}

}  // namespace
}  // namespace loom
