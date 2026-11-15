// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_relation_set.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

using SetId = loom_condition_relation_set_id_t;

struct CollectState {
  std::vector<uint32_t>* values;
  size_t limit;
};

bool CollectValue(void* user_data, uint32_t value) {
  auto* state = static_cast<CollectState*>(user_data);
  state->values->push_back(value);
  return state->values->size() < state->limit;
}

std::vector<uint32_t> ValuesFromMembers(const std::vector<uint8_t>& members) {
  std::vector<uint32_t> values;
  for (uint32_t value = 0; value < members.size(); ++value) {
    if (members[value]) {
      values.push_back(value);
    }
  }
  return values;
}

SetId Intern(loom_condition_relation_set_builder_t* builder,
             const std::vector<uint8_t>& members) {
  const std::vector<uint32_t> values = ValuesFromMembers(members);
  SetId set = LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_CHECK_OK(loom_condition_relation_set_builder_intern(
      builder, values.data(), values.size(), &set));
  return set;
}

void ExpectBuilderMembers(const loom_condition_relation_set_builder_t* builder,
                          SetId set, const std::vector<uint8_t>& expected) {
  std::vector<uint32_t> actual;
  CollectState state = {&actual, SIZE_MAX};
  EXPECT_TRUE(loom_condition_relation_set_builder_for_each_while(
      builder, set, CollectValue, &state));
  EXPECT_EQ(actual, ValuesFromMembers(expected));
}

void ExpectIndexMembers(const loom_condition_relation_set_index_t& index,
                        SetId set, const std::vector<uint8_t>& expected) {
  for (uint32_t value = 0; value < expected.size(); ++value) {
    EXPECT_EQ(loom_condition_relation_set_index_contains(&index, set, value),
              expected[value] != 0)
        << "value " << value;
  }
  EXPECT_FALSE(loom_condition_relation_set_index_contains(
      &index, set, static_cast<uint32_t>(expected.size())));
  std::vector<uint32_t> actual;
  CollectState state = {&actual, SIZE_MAX};
  EXPECT_TRUE(loom_condition_relation_set_index_for_each_while(
      &index, set, CollectValue, &state));
  EXPECT_EQ(actual, ValuesFromMembers(expected));
}

class ConditionRelationSetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    scratch_arena_initialized_ = true;
    iree_arena_initialize(&block_pool_, &retained_arena_);
  }

  void TearDown() override {
    if (scratch_arena_initialized_) {
      iree_arena_deinitialize(&scratch_arena_);
    }
    iree_arena_deinitialize(&retained_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_condition_relation_set_builder_t* AllocateBuilder(uint32_t value_count) {
    loom_condition_relation_set_builder_t* builder = nullptr;
    IREE_CHECK_OK(loom_condition_relation_set_builder_allocate(
        value_count, &scratch_arena_, &builder));
    return builder;
  }

  void ReleaseScratch() {
    iree_arena_deinitialize(&scratch_arena_);
    scratch_arena_initialized_ = false;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  iree_arena_allocator_t retained_arena_;
  bool scratch_arena_initialized_ = false;
};

TEST_F(ConditionRelationSetTest, EmptyAndSingletonSetsStayInline) {
  auto* builder = AllocateBuilder(256);
  SetId empty = 99;
  IREE_ASSERT_OK(
      loom_condition_relation_set_builder_intern(builder, nullptr, 0, &empty));
  EXPECT_EQ(empty, LOOM_CONDITION_RELATION_SET_EMPTY);

  const uint32_t repeated_values[] = {17, 17, 17};
  SetId singleton = 0;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(
      builder, repeated_values, std::size(repeated_values), &singleton));
  EXPECT_EQ(singleton,
            loom_condition_relation_set_builder_singleton(builder, 17));

  std::array<SetId, 2> roots = {empty, singleton};
  loom_condition_relation_set_index_t index;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
      builder, roots.data(), roots.size(), &retained_arena_, &index));
  EXPECT_EQ(index.node_count, 0u);
  EXPECT_EQ(roots[0], LOOM_CONDITION_RELATION_SET_EMPTY);
  EXPECT_EQ(roots[1], singleton);
  EXPECT_FALSE(
      loom_condition_relation_set_index_contains(&index, roots[0], 17));
  EXPECT_TRUE(loom_condition_relation_set_index_contains(&index, roots[1], 17));
}

TEST_F(ConditionRelationSetTest, EmptyDomainPublishesAnEmptyIndex) {
  auto* builder = AllocateBuilder(0);
  SetId root = 1;
  IREE_ASSERT_OK(
      loom_condition_relation_set_builder_intern(builder, nullptr, 0, &root));
  loom_condition_relation_set_index_t index;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
      builder, &root, 1, &retained_arena_, &index));
  EXPECT_EQ(root, LOOM_CONDITION_RELATION_SET_EMPTY);
  EXPECT_EQ(index.value_count, 0u);
  EXPECT_EQ(index.node_count, 0u);
  EXPECT_FALSE(loom_condition_relation_set_index_contains(&index, root, 0));
}

TEST_F(ConditionRelationSetTest, SingletonLeavesNeedOnlyBranchNodes) {
  auto* builder = AllocateBuilder(256);
  std::vector<uint8_t> members(256);
  for (uint32_t value : std::array<uint32_t, 4>{0, 64, 128, 192}) {
    members[value] = 1;
  }
  SetId root = Intern(builder, members);
  loom_condition_relation_set_index_t index;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
      builder, &root, 1, &retained_arena_, &index));
  EXPECT_EQ(index.node_count, 3u);
  ExpectIndexMembers(index, root, members);
}

TEST_F(ConditionRelationSetTest, IterationStopsWithoutVisitingTheTail) {
  auto* builder = AllocateBuilder(128);
  std::vector<uint8_t> members(128);
  for (uint32_t value : {1u, 3u, 5u, 64u}) {
    members[value] = 1;
  }
  SetId root = Intern(builder, members);

  std::vector<uint32_t> builder_values;
  CollectState builder_state = {&builder_values, 2};
  EXPECT_FALSE(loom_condition_relation_set_builder_for_each_while(
      builder, root, CollectValue, &builder_state));
  EXPECT_EQ(builder_values, (std::vector<uint32_t>{1, 3}));

  loom_condition_relation_set_index_t index;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
      builder, &root, 1, &retained_arena_, &index));
  std::vector<uint32_t> index_values;
  CollectState index_state = {&index_values, 2};
  EXPECT_FALSE(loom_condition_relation_set_index_for_each_while(
      &index, root, CollectValue, &index_state));
  EXPECT_EQ(index_values, (std::vector<uint32_t>{1, 3}));
}

TEST_F(ConditionRelationSetTest, PublicationDropsUnreachableSets) {
  auto* builder = AllocateBuilder(4096);
  std::mt19937 random(0x515E7u);
  for (uint32_t trial = 0; trial < 128; ++trial) {
    std::vector<uint8_t> members(4096);
    for (uint32_t i = 0; i < 64; ++i) {
      members[random() % members.size()] = 1;
    }
    Intern(builder, members);
  }
  const uint32_t singleton_value = 3072;
  SetId root = 0;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(
      builder, &singleton_value, 1, &root));
  loom_condition_relation_set_index_t index;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
      builder, &root, 1, &retained_arena_, &index));
  EXPECT_EQ(index.node_count, 0u);
  EXPECT_TRUE(loom_condition_relation_set_index_contains(&index, root, 3072));
}

TEST_F(ConditionRelationSetTest, RandomizedAlgebraSurvivesScratchDestruction) {
  constexpr uint32_t kValueCount = 4096;
  auto* builder = AllocateBuilder(kValueCount);
  std::mt19937 random(0xC001D00Du);
  std::vector<SetId> retained_roots;
  std::vector<std::vector<uint8_t>> retained_members;

  for (uint32_t trial = 0; trial < 750; ++trial) {
    SCOPED_TRACE(trial);
    std::vector<uint8_t> left(kValueCount);
    std::vector<uint8_t> right(kValueCount);
    const uint32_t left_count = trial % 11 == 0 ? 1024 : random() % 41;
    const uint32_t right_count = trial % 13 == 0 ? 2048 : random() % 41;
    for (uint32_t i = 0; i < left_count; ++i) {
      left[random() % kValueCount] = 1;
    }
    for (uint32_t i = 0; i < right_count; ++i) {
      right[random() % kValueCount] = 1;
    }
    if (trial == 0) {
      for (uint32_t value :
           std::array<uint32_t, 9>{0, 1, 63, 64, 65, 127, 128, 4094, 4095}) {
        left[value] = 1;
      }
    }

    const SetId left_set = Intern(builder, left);
    const SetId right_set = Intern(builder, right);
    ExpectBuilderMembers(builder, left_set, left);
    ExpectBuilderMembers(builder, right_set, right);

    std::array<std::vector<uint8_t>, 3> expected = {
        std::vector<uint8_t>(kValueCount),
        std::vector<uint8_t>(kValueCount),
        std::vector<uint8_t>(kValueCount),
    };
    for (uint32_t value = 0; value < kValueCount; ++value) {
      expected[0][value] = left[value] || right[value];
      expected[1][value] = left[value] && right[value];
      expected[2][value] = left[value] && !right[value];
    }
    std::array<SetId, 3> results;
    IREE_ASSERT_OK(loom_condition_relation_set_builder_union(
        builder, left_set, right_set, &results[0]));
    IREE_ASSERT_OK(loom_condition_relation_set_builder_intersection(
        builder, left_set, right_set, &results[1]));
    IREE_ASSERT_OK(loom_condition_relation_set_builder_difference(
        builder, left_set, right_set, &results[2]));
    for (size_t i = 0; i < results.size(); ++i) {
      ExpectBuilderMembers(builder, results[i], expected[i]);
    }
    if (trial % 37 == 0) {
      retained_roots.insert(retained_roots.end(), results.begin(),
                            results.end());
      retained_members.insert(retained_members.end(), expected.begin(),
                              expected.end());
    }
  }

  const iree_host_size_t scratch_bytes = scratch_arena_.used_allocation_size;
  loom_condition_relation_set_index_t index;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
      builder, retained_roots.data(), retained_roots.size(), &retained_arena_,
      &index));
  EXPECT_GT(index.node_count, 0u);
  ReleaseScratch();

  iree_arena_allocator_t churn_arena;
  iree_arena_initialize(&block_pool_, &churn_arena);
  iree_host_size_t remaining_bytes = scratch_bytes;
  while (remaining_bytes != 0) {
    const iree_host_size_t chunk_size =
        iree_min(remaining_bytes, (iree_host_size_t)16384);
    void* chunk = nullptr;
    IREE_ASSERT_OK(iree_arena_allocate(&churn_arena, chunk_size, &chunk));
    memset(chunk, 0xA5, chunk_size);
    remaining_bytes -= chunk_size;
  }

  ASSERT_EQ(retained_roots.size(), retained_members.size());
  for (size_t i = 0; i < retained_roots.size(); ++i) {
    SCOPED_TRACE(i);
    ExpectIndexMembers(index, retained_roots[i], retained_members[i]);
  }
  iree_arena_deinitialize(&churn_arena);
}

TEST_F(ConditionRelationSetTest, RejectsAnExhaustedIdDomain) {
  loom_condition_relation_set_builder_t* builder = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_condition_relation_set_builder_allocate(
                            UINT32_MAX, &scratch_arena_, &builder));
  EXPECT_EQ(builder, nullptr);
}

TEST_F(ConditionRelationSetTest, AlgebraHasCanonicalIdentityAcrossChunks) {
  for (uint32_t value_count : {31u, 32u, 33u, 64u, 256u}) {
    SCOPED_TRACE(value_count);
    const std::array<uint32_t, 7> kValues =
        value_count <= 64
            ? std::array<uint32_t, 7>{0, 1, 2, 4, 15, 16, value_count - 1}
            : std::array<uint32_t, 7>{0, 1, 63, 64, 127, 128, 255};
    constexpr size_t kSubsetCount = size_t{1} << 7;
    auto* builder = AllocateBuilder(value_count);
    std::array<SetId, kSubsetCount> subsets;
    for (size_t subset = 0; subset < subsets.size(); ++subset) {
      std::vector<uint32_t> values;
      for (size_t bit = 0; bit < kValues.size(); ++bit) {
        if (subset & (size_t{1} << bit)) {
          values.push_back(kValues[bit]);
        }
      }
      IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(
          builder, values.data(), values.size(), &subsets[subset]));
      std::reverse(values.begin(), values.end());
      if (!values.empty()) {
        values.push_back(values.back());
      }
      const auto original = values;
      SetId reordered = 0;
      IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(
          builder, values.data(), values.size(), &reordered));
      EXPECT_EQ(reordered, subsets[subset]);
      EXPECT_EQ(values, original);
    }
    for (size_t left = 0; left < subsets.size(); ++left) {
      for (size_t right = 0; right < subsets.size(); ++right) {
        SCOPED_TRACE(::testing::Message() << left << ", " << right);
        SetId result = 0;
        IREE_ASSERT_OK(loom_condition_relation_set_builder_union(
            builder, subsets[left], subsets[right], &result));
        EXPECT_EQ(result, subsets[left | right]);
        IREE_ASSERT_OK(loom_condition_relation_set_builder_intersection(
            builder, subsets[left], subsets[right], &result));
        EXPECT_EQ(result, subsets[left & right]);
        IREE_ASSERT_OK(loom_condition_relation_set_builder_difference(
            builder, subsets[left], subsets[right], &result));
        EXPECT_EQ(result, subsets[left & ~right]);
      }
    }
  }
}

TEST_F(ConditionRelationSetTest, ImmediateCapacityPreservesSetSemantics) {
  constexpr std::array<uint32_t, 6> kValueCounts = {0, 1, 31, 32, 33, 64};
  std::array<loom_condition_relation_set_index_t, kValueCounts.size()> indexes;
  std::array<std::array<SetId, 2>, kValueCounts.size()> roots;
  for (size_t i = 0; i < kValueCounts.size(); ++i) {
    const uint32_t value_count = kValueCounts[i];
    SCOPED_TRACE(value_count);
    auto* builder = AllocateBuilder(value_count);
    const size_t initial_scratch_bytes = scratch_arena_.used_allocation_size;
    SetId full = 0;
    std::vector<uint8_t> even_members(value_count);
    for (uint32_t value = 0; value < value_count; ++value) {
      const SetId singleton =
          loom_condition_relation_set_builder_singleton(builder, value);
      SetId interned = 0;
      IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(builder, &value,
                                                                1, &interned));
      EXPECT_EQ(singleton, interned);
      IREE_ASSERT_OK(loom_condition_relation_set_builder_union(
          builder, full, singleton, &full));
      even_members[value] = value % 2 == 0;
    }
    const SetId even = Intern(builder, even_members);
    SetId odd = 0;
    IREE_ASSERT_OK(loom_condition_relation_set_builder_difference(builder, full,
                                                                  even, &odd));
    SetId intersection = full;
    IREE_ASSERT_OK(loom_condition_relation_set_builder_intersection(
        builder, even, odd, &intersection));
    EXPECT_EQ(intersection, LOOM_CONDITION_RELATION_SET_EMPTY);
    SetId combined = 0;
    IREE_ASSERT_OK(loom_condition_relation_set_builder_union(builder, even, odd,
                                                             &combined));
    EXPECT_EQ(combined, full);
    ExpectBuilderMembers(builder, full, std::vector<uint8_t>(value_count, 1));

    roots[i] = {full, even};
    const size_t initial_retained_bytes = retained_arena_.used_allocation_size;
    IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
        builder, roots[i].data(), roots[i].size(), &retained_arena_,
        &indexes[i]));
    if (value_count <= 32) {
      EXPECT_EQ(scratch_arena_.used_allocation_size, initial_scratch_bytes);
      EXPECT_EQ(retained_arena_.used_allocation_size, initial_retained_bytes);
      EXPECT_EQ(indexes[i].node_count, 0u);
      EXPECT_EQ(indexes[i].nodes, nullptr);
    }
  }
  ReleaseScratch();
  for (size_t i = 0; i < kValueCounts.size(); ++i) {
    SCOPED_TRACE(kValueCounts[i]);
    std::vector<uint8_t> even_members(kValueCounts[i]);
    for (uint32_t value = 0; value < kValueCounts[i]; ++value) {
      even_members[value] = value % 2 == 0;
    }
    ExpectIndexMembers(indexes[i], roots[i][0],
                       std::vector<uint8_t>(kValueCounts[i], 1));
    ExpectIndexMembers(indexes[i], roots[i][1], even_members);
    std::vector<uint32_t> prefix;
    CollectState state = {&prefix, 2};
    EXPECT_EQ(loom_condition_relation_set_index_for_each_while(
                  &indexes[i], roots[i][0], CollectValue, &state),
              kValueCounts[i] < 2);
    const std::vector<uint32_t> expected_prefix =
        kValueCounts[i] == 0   ? std::vector<uint32_t>{}
        : kValueCounts[i] == 1 ? std::vector<uint32_t>{0}
                               : std::vector<uint32_t>{0, 1};
    EXPECT_EQ(prefix, expected_prefix);
  }
}

TEST_F(ConditionRelationSetTest, UnorderedBatchesPreserveCanonicalMembers) {
  auto* builder = AllocateBuilder(16384);
  std::mt19937 random(0xB47C4u);
  std::vector<SetId> roots;
  std::vector<std::vector<uint32_t>> expected;
  for (uint32_t count : {3u, 65u, 1025u, 17u}) {
    std::vector<uint32_t> values(count);
    for (uint32_t& value : values) {
      value = random() % 16384;
    }
    const auto original = values;
    SetId root = 0;
    IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(
        builder, values.data(), values.size(), &root));
    EXPECT_EQ(values, original);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    SetId ordered = 0;
    IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(
        builder, values.data(), values.size(), &ordered));
    EXPECT_EQ(root, ordered);
    roots.push_back(root);
    expected.push_back(std::move(values));
  }
  loom_condition_relation_set_index_t index;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
      builder, roots.data(), roots.size(), &retained_arena_, &index));
  ReleaseScratch();
  for (size_t i = 0; i < roots.size(); ++i) {
    std::vector<uint32_t> actual;
    CollectState state = {&actual, SIZE_MAX};
    EXPECT_TRUE(loom_condition_relation_set_index_for_each_while(
        &index, roots[i], CollectValue, &state));
    EXPECT_EQ(actual, expected[i]);
  }
}

TEST_F(ConditionRelationSetTest, PublicationPreservesRelativeLeafPositions) {
  auto* builder = AllocateBuilder(4096);
  const uint32_t lower_values[] = {1, 3};
  const uint32_t upper_values[] = {2049, 2051};
  std::array<SetId, 3> roots = {};
  IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(
      builder, lower_values, std::size(lower_values), &roots[0]));
  IREE_ASSERT_OK(loom_condition_relation_set_builder_intern(
      builder, upper_values, std::size(upper_values), &roots[1]));
  IREE_ASSERT_OK(loom_condition_relation_set_builder_union(
      builder, roots[0], roots[1], &roots[2]));
  // Grow construction after retaining the selected roots. Publication must
  // preserve their shared leaf shape at both absolute positions.
  SetId unused = 0;
  for (uint32_t value = 0; value < 4096; ++value) {
    IREE_ASSERT_OK(loom_condition_relation_set_builder_union(
        builder, unused,
        loom_condition_relation_set_builder_singleton(builder, value),
        &unused));
  }
  loom_condition_relation_set_index_t index;
  IREE_ASSERT_OK(loom_condition_relation_set_builder_publish(
      builder, roots.data(), roots.size(), &retained_arena_, &index));
  EXPECT_LE(index.node_count, 3u * (index.root_level + 1u));
  ReleaseScratch();
  for (uint32_t value = 0; value < 4096; ++value) {
    const bool lower = value == 1 || value == 3;
    const bool upper = value == 2049 || value == 2051;
    EXPECT_EQ(
        loom_condition_relation_set_index_contains(&index, roots[0], value),
        lower);
    EXPECT_EQ(
        loom_condition_relation_set_index_contains(&index, roots[1], value),
        upper);
    EXPECT_EQ(
        loom_condition_relation_set_index_contains(&index, roots[2], value),
        lower || upper);
  }
}

}  // namespace
}  // namespace loom
