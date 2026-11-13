// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/call_aliases.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom::cxx_import {
namespace {

class CallAliasesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  std::vector<uint8_t> Select(std::span<const CallAliasParameter> parameters,
                              std::span<const CallAliasInterval> intervals) {
    iree_arena_reset(&arena_);
    std::span<const uint8_t> selected;
    IREE_EXPECT_OK(
        CallAliases(parameters, intervals).select_storage(&arena_, &selected));
    return selected.empty()
               ? std::vector<uint8_t>(parameters.size(), 0)
               : std::vector<uint8_t>(selected.begin(), selected.end());
  }

  iree_arena_block_pool_t pool_ = {};
  iree_arena_allocator_t arena_ = {};
};

TEST_F(CallAliasesTest, SelectsDirectWritableOverlap) {
  int object;
  const std::array parameters = {
      CallAliasParameter{false, true, false},
      CallAliasParameter{true, true, false},
      CallAliasParameter{false, true, false},
  };
  const std::array intervals = {
      CallAliasInterval{&object, 0, 1, 0},
      CallAliasInterval{&object, 0, 2, 1},
      CallAliasInterval{&object, 1, 2, 2},
  };
  EXPECT_EQ(Select(parameters, intervals), (std::vector<uint8_t>{1, 1, 1}));
}

TEST_F(CallAliasesTest, DoesNotSelectTransitiveReadonlyOverlap) {
  int object;
  const std::array parameters = {
      CallAliasParameter{true, true, false},
      CallAliasParameter{false, true, false},
      CallAliasParameter{false, true, false},
  };
  const std::array intervals = {
      CallAliasInterval{&object, 0, 1, 0},
      CallAliasInterval{&object, 0, 2, 1},
      CallAliasInterval{&object, 1, 2, 2},
  };
  EXPECT_EQ(Select(parameters, intervals), (std::vector<uint8_t>{1, 1, 0}));
}

TEST_F(CallAliasesTest, ExcludesIntervalsFromSameParameter) {
  int object;
  const std::array parameters = {
      CallAliasParameter{true, true, false},
      CallAliasParameter{false, true, false},
  };
  const std::array intervals = {
      CallAliasInterval{&object, 0, SIZE_MAX, 0},
      CallAliasInterval{&object, 0, 1, 0},
      CallAliasInterval{&object, 1, 2, 1},
  };
  EXPECT_EQ(Select(parameters, intervals), (std::vector<uint8_t>{1, 1}));

  const std::array own_intervals = {
      CallAliasInterval{&object, 0, SIZE_MAX, 0},
      CallAliasInterval{&object, 0, 1, 0},
  };
  EXPECT_EQ(Select(parameters, own_intervals), (std::vector<uint8_t>{0, 0}));
}

TEST_F(CallAliasesTest, SelectsUnresolvedIdentityConservatively) {
  const std::array parameters = {
      CallAliasParameter{false, true, true},
      CallAliasParameter{true, true, false},
      CallAliasParameter{false, false, false},
  };
  EXPECT_EQ(Select(parameters, {}), (std::vector<uint8_t>{1, 1, 0}));
}

TEST_F(CallAliasesTest, KeepsWideDisjointArgumentsIndependent) {
  constexpr size_t kCount = 2048;
  std::vector<int> objects(kCount);
  std::vector<CallAliasParameter> parameters(
      kCount, CallAliasParameter{true, true, false});
  std::vector<CallAliasInterval> intervals;
  intervals.reserve(kCount);
  for (size_t index = 0; index < kCount; ++index) {
    intervals.push_back({&objects[index], 0, SIZE_MAX, index});
  }
  EXPECT_EQ(Select(parameters, intervals), std::vector<uint8_t>(kCount, 0));
  EXPECT_LT(arena_.used_allocation_size, 128 * 1024u);
}

TEST_F(CallAliasesTest, MatchesPairwiseSemantics) {
  std::array<int, 4> objects;
  uint32_t random = 0xC001CAFEu;
  auto next = [&] {
    random = random * 1664525u + 1013904223u;
    return random;
  };
  for (size_t iteration = 0; iteration < 1000; ++iteration) {
    const size_t parameter_count = 2 + next() % 11;
    std::vector<CallAliasParameter> parameters(parameter_count);
    std::vector<CallAliasInterval> intervals;
    for (size_t parameter = 0; parameter < parameter_count; ++parameter) {
      auto& facts = parameters[parameter];
      facts.written = next() % 4 == 0;
      facts.unresolved_identity = next() % 8 == 0;
      const size_t interval_count = next() % 4;
      for (size_t index = 0; index < interval_count; ++index) {
        const size_t begin = next() % 8;
        const size_t end =
            next() % 12 == 0 ? SIZE_MAX : begin + 1 + next() % (9 - begin);
        intervals.push_back(
            {&objects[next() % objects.size()], begin, end, parameter});
      }
      facts.has_identity =
          facts.unresolved_identity || interval_count || next() % 8 == 0;
    }

    std::vector<uint8_t> expected(parameter_count, 0);
    for (size_t left = 0; left < parameter_count; ++left) {
      for (size_t right = left + 1; right < parameter_count; ++right) {
        if (!parameters[left].written && !parameters[right].written) {
          continue;
        }
        bool overlaps = (parameters[left].unresolved_identity &&
                         parameters[right].has_identity) ||
                        (parameters[right].unresolved_identity &&
                         parameters[left].has_identity);
        for (const auto& left_interval : intervals) {
          if (left_interval.parameter != left) {
            continue;
          }
          for (const auto& right_interval : intervals) {
            if (right_interval.parameter != right ||
                left_interval.binding != right_interval.binding) {
              continue;
            }
            overlaps |= left_interval.end == SIZE_MAX ||
                        right_interval.end == SIZE_MAX ||
                        (left_interval.begin < right_interval.end &&
                         right_interval.begin < left_interval.end);
          }
        }
        if (overlaps) {
          expected[left] = true;
          expected[right] = true;
        }
      }
    }
    EXPECT_EQ(Select(parameters, intervals), expected)
        << "random case " << iteration;
  }
}

}  // namespace
}  // namespace loom::cxx_import
