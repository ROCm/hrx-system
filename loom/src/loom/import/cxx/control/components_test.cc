// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/components.h"

#include <array>

#include "iree/testing/gtest.h"

namespace loom::cxx_import {
namespace {

Components components(unsigned mask) {
  Components result;
  for (size_t index = 0; index < 6; ++index) {
    if (mask & (1u << index)) {
      result.add({index, index + 1});
    }
  }
  return result;
}

TEST(ComponentsTest, SetOperationsMatchIndependentBitSets) {
  for (unsigned left = 0; left < 64; ++left) {
    for (unsigned right = 0; right < 64; ++right) {
      auto original = components(left);
      auto other = components(right);
      auto combined = original;
      EXPECT_EQ(combined.add(other), (right & ~left) != 0);
      EXPECT_EQ(combined, components(left | right));
      auto intersection = original;
      intersection.intersect(other);
      EXPECT_EQ(intersection, components(left & right));
      auto difference = original;
      difference.subtract(other);
      EXPECT_EQ(difference, components(left & ~right));
    }
  }
}

TEST(ComponentsTest, QueriesAndProjectionPreserveIntervals) {
  for (unsigned mask = 0; mask < 64; ++mask) {
    auto source = components(mask);
    for (size_t begin = 0; begin <= 6; ++begin) {
      for (size_t end = begin; end <= 6; ++end) {
        const unsigned interval = ((1u << end) - 1) ^ ((1u << begin) - 1);
        EXPECT_EQ(source.contains({begin, end}), (mask & interval) == interval);
        EXPECT_EQ(source.intersects({begin, end}), (mask & interval) != 0);
        EXPECT_EQ(source.project({begin, end}, 0),
                  components((mask & interval) >> begin));
        auto shifted = source.project({begin, end}, 1000);
        for (size_t index = 0; index < end - begin; ++index) {
          EXPECT_EQ(shifted.contains({1000 + index, 1001 + index}),
                    (mask & (1u << (begin + index))) != 0);
        }
      }
    }
  }
}

TEST(ComponentsTest, CoalescesOutOfOrderAndLargeRanges) {
  Components value;
  EXPECT_TRUE(value.add({1000000, 1000001}));
  EXPECT_TRUE(value.add({3, 4}));
  EXPECT_TRUE(value.add({1, 2}));
  EXPECT_TRUE(value.add({2, 3}));
  EXPECT_FALSE(value.add({2, 4}));
  const std::array<ComponentRange, 2> expected{{{1, 4}, {1000000, 1000001}}};
  ASSERT_EQ(value.ranges().size(), expected.size());
  EXPECT_EQ(value.ranges()[0], expected[0]);
  EXPECT_EQ(value.ranges()[1], expected[1]);
  EXPECT_FALSE(value.add(value));
  value.intersect(value);
  EXPECT_EQ(value.ranges().size(), expected.size());
  value.subtract(value);
  EXPECT_TRUE(value.empty());
}

}  // namespace
}  // namespace loom::cxx_import
