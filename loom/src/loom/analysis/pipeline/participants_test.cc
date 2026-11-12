// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/participants.h"

#include <algorithm>
#include <array>
#include <map>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class PipelineParticipantsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  const loom_pipeline_participants_t* Build(
      std::vector<loom_pipeline_participant_t> entries) {
    const loom_pipeline_participants_t* result = nullptr;
    IREE_EXPECT_OK(loom_pipeline_participants_build(
        &arena_, entries.data(), entries.size(), &result, &conflict_));
    EXPECT_EQ(conflict_.second_worker, IREE_HOST_SIZE_MAX);
    return result;
  }
  void Check(const loom_pipeline_participants_t* participants,
             const std::map<size_t, size_t>& expected) {
    EXPECT_EQ(loom_pipeline_participants_count(participants), expected.size());
    loom_pipeline_participant_iterator_t iterator;
    loom_pipeline_participants_begin(participants, &iterator);
    size_t index = 0;
    for (const auto& [engine, worker] : expected) {
      loom_pipeline_participant_t next;
      ASSERT_TRUE(loom_pipeline_participants_next(&iterator, &next));
      EXPECT_EQ(next.engine, engine);
      EXPECT_EQ(next.worker, worker);
      const auto location =
          loom_pipeline_participants_find(participants, engine);
      EXPECT_EQ(location.index, index++);
      EXPECT_EQ(location.worker, worker);
    }
    loom_pipeline_participant_t next;
    EXPECT_FALSE(loom_pipeline_participants_next(&iterator, &next));
  }

  // Backing blocks shared by immutable participant summaries.
  iree_arena_block_pool_t pool_;
  // Owner of all retained and shared nodes in each test.
  iree_arena_allocator_t arena_;
  // Conflict from the most recent concurrent assignment.
  loom_pipeline_execution_conflict_t conflict_ = {};
};

TEST_F(PipelineParticipantsTest, SparseOrdinalsKeepRankAndOrderedIteration) {
  const auto* participants =
      Build({{4096, 3}, {0, 4}, {63, 2}, {IREE_HOST_SIZE_MAX - 1, 9}, {64, 1}});
  Check(participants,
        {{0, 4}, {63, 2}, {64, 1}, {4096, 3}, {IREE_HOST_SIZE_MAX - 1, 9}});
  for (size_t absent : {1u, 62u, 65u, 128u, 4095u}) {
    const auto location = loom_pipeline_participants_find(participants, absent);
    EXPECT_EQ(location.index, IREE_HOST_SIZE_MAX);
    EXPECT_EQ(location.worker, IREE_HOST_SIZE_MAX);
  }
  Check(Build({}), {});
}

TEST_F(PipelineParticipantsTest, OrderedUnionSharesUnchangedMembership) {
  const auto* first = Build({{1, 10}, {63, 11}, {64, 12}, {128, 13}});
  const auto* second = Build({{1, 20}, {64, 21}});
  const loom_pipeline_participants_t* result = nullptr;
  const auto before = arena_.used_allocation_size;
  IREE_ASSERT_OK(loom_pipeline_participants_merge(&arena_, first, second, false,
                                                  nullptr, &result));
  EXPECT_EQ(result, first);
  EXPECT_EQ(arena_.used_allocation_size, before);
  Check(result, {{1, 10}, {63, 11}, {64, 12}, {128, 13}});
  // Membership is independent of representatives and subtree identity.
  const auto* same = Build({{1, 30}, {63, 31}, {64, 32}, {128, 33}});
  EXPECT_EQ(loom_pipeline_participants_compare_members(result, same), 0);
}

TEST_F(PipelineParticipantsTest, ConcurrentUnionReportsLowestSharedEngine) {
  const auto* first = Build({{1, 10}, {128, 11}, {4096, 12}});
  const auto* second = Build({{64, 20}, {128, 21}, {4096, 22}});
  const loom_pipeline_participants_t* result = nullptr;
  IREE_ASSERT_OK(loom_pipeline_participants_merge(&arena_, first, second, true,
                                                  &conflict_, &result));
  EXPECT_EQ(conflict_.first_worker, 11u);
  EXPECT_EQ(conflict_.second_worker, 21u);
  Check(first, {{1, 10}, {128, 11}, {4096, 12}});
  Check(second, {{64, 20}, {128, 21}, {4096, 22}});
}

TEST_F(PipelineParticipantsTest, LocalDuplicatesRemainPlacementConflicts) {
  std::vector<loom_pipeline_participant_t> entries = {{64, 3}, {1, 1}, {64, 2}};
  const loom_pipeline_participants_t* result = nullptr;
  IREE_ASSERT_OK(loom_pipeline_participants_build(
      &arena_, entries.data(), entries.size(), &result, &conflict_));
  EXPECT_EQ(conflict_.first_worker, 2u);
  EXPECT_EQ(conflict_.second_worker, 3u);
}

TEST_F(PipelineParticipantsTest,
       UnionAcrossRadixPartitionsPreservesBothInputs) {
  // Independent selected engine patterns exercise same-leaf, partial-prefix,
  // and disjoint-subtree unions without depending on a particular node layout.
  for (size_t step = 1; step <= 17; step += 4) {
    std::vector<loom_pipeline_participant_t> left, right;
    std::map<size_t, size_t> expected_left, expected_right, expected_union;
    std::array<bool, 321> left_bits = {}, right_bits = {};
    for (size_t engine = 0; engine < left_bits.size(); ++engine) {
      if (engine % step == 0) {
        left.push_back({engine, engine + 1000});
        expected_left[engine] = expected_union[engine] = engine + 1000;
        left_bits[engine] = true;
      }
      if ((engine + 1) % (step + 1) == 0) {
        right.push_back({engine, engine + 2000});
        expected_right[engine] = engine + 2000;
        expected_union.emplace(engine, engine + 2000);
        right_bits[engine] = true;
      }
    }
    const auto* first = Build(left);
    const auto* second = Build(right);
    const loom_pipeline_participants_t* result = nullptr;
    IREE_ASSERT_OK(loom_pipeline_participants_merge(&arena_, first, second,
                                                    false, nullptr, &result));
    Check(result, expected_union);
    Check(first, expected_left);
    Check(second, expected_right);
    const int expected_comparison =
        left_bits == right_bits ? 0
        : std::lexicographical_compare(left_bits.begin(), left_bits.end(),
                                       right_bits.begin(), right_bits.end())
            ? -1
            : 1;
    EXPECT_EQ(loom_pipeline_participants_compare_members(first, second),
              expected_comparison);
    EXPECT_EQ(loom_pipeline_participants_compare_members(second, first),
              -expected_comparison);
  }
}

TEST_F(PipelineParticipantsTest, SelectedControllersCanBeAddedIncrementally) {
  const loom_pipeline_participants_t* participants = nullptr;
  std::map<size_t, size_t> expected;
  for (size_t engine : {4096u, 64u, 129u, 1u, 63u, 128u, 4095u, 0u}) {
    const auto* snapshot = participants;
    IREE_ASSERT_OK(loom_pipeline_participants_insert(
        &arena_, {engine, engine + 1}, &participants));
    Check(snapshot, expected);
    expected[engine] = engine + 1;
    Check(participants, expected);
  }
  const auto* snapshot = participants;
  IREE_ASSERT_OK(
      loom_pipeline_participants_insert(&arena_, {64, 999}, &participants));
  EXPECT_EQ(participants, snapshot);
  Check(participants, expected);
}

}  // namespace
