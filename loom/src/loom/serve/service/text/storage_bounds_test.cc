// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/storage_bounds.h"

#include <algorithm>
#include <bit>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

using Bound = loom_serve_text_storage_bound_t;

uint64_t Minimum(std::vector<Bound> bounds) {
  return loom_serve_text_storage_bound(bounds.size(), bounds.data());
}

TEST(StorageBoundsTest, Empty) { EXPECT_EQ(Minimum({}), 0u); }

TEST(StorageBoundsTest, FixedUnionAndIndependentAllocations) {
  EXPECT_EQ(Minimum({{0, 2, 5, 3}, {0, 4, 7, 3}, {0, 9, 11, 2}, {1, 2, 5, 3}}),
            10u);
}

TEST(StorageBoundsTest, MovableDomainsDoNotFreezeCurrentAddresses) {
  // Two retained units may be compacted together. The later fixed view may
  // alias their domain, so adding their independent counts would be unsound.
  EXPECT_EQ(Minimum({{0, 0, 8, 2}, {0, 5, 7, 2}}), 2u);
  EXPECT_EQ(Minimum({{0, 0, 4, 2}, {0, 5, 7, 2}}), 4u);
  // Transitive overlaps also alias; no dependence on input order is allowed.
  std::vector<Bound> bounds = {{0, 0, 4, 2}, {0, 3, 8, 2}, {0, 7, 10, 2}};
  EXPECT_EQ(Minimum(bounds), 2u);
  std::reverse(bounds.begin(), bounds.end());
  EXPECT_EQ(Minimum(bounds), 2u);
}

TEST(StorageBoundsTest, SaturationAcrossAllocations) {
  EXPECT_EQ(Minimum({{0, 0, UINT64_MAX, UINT64_MAX}, {1, 0, 1, 1}}),
            UINT64_MAX);
}

TEST(StorageBoundsTest, ExhaustiveSmallDomainsNeverRejectValidPlacement) {
  // Enumerate independent occupancy masks, not the production interval union
  // algorithm. Every valid placement must cost at least the reported bound.
  // All ordered triples exercise fixed/movable overlap, duplicates, adjacency,
  // nested domains, and every permutation of those constraints.
  constexpr uint32_t kUnits = 4;
  std::vector<Bound> domains;
  for (uint32_t begin = 0; begin < kUnits; ++begin) {
    for (uint32_t end = begin + 1; end <= kUnits; ++end) {
      for (uint32_t minimum = 1; minimum <= end - begin; ++minimum) {
        domains.push_back({0, begin, end, minimum});
      }
    }
  }
  for (const Bound& first : domains) {
    for (const Bound& second : domains) {
      for (const Bound& third : domains) {
        Bound bounds[] = {first, second, third};
        uint32_t exact = kUnits;
        for (uint32_t occupied = 0; occupied < (1u << kUnits); ++occupied) {
          bool valid = true;
          for (const Bound& bound : bounds) {
            const uint32_t mask =
                ((1u << bound.end) - 1) ^ ((1u << bound.begin) - 1);
            valid &= std::popcount(occupied & mask) >= bound.minimum;
          }
          if (valid) {
            exact = std::min(exact, uint32_t(std::popcount(occupied)));
          }
        }
        ASSERT_LE(loom_serve_text_storage_bound(3, bounds), exact);
      }
    }
  }
}

}  // namespace
