// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_derivation.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/fact_relation.h"

namespace loom {
namespace {

using Relation = loom_symbolic_integer_relation_t;
using Operand = loom_condition_integer_operand_t;
using Fact = loom_condition_integer_relation_t;
using Row = std::tuple<int, int64_t, int, int64_t, Relation>;
constexpr Relation kRelations[] = {
    LOOM_SYMBOLIC_INTEGER_RELATION_EQ, LOOM_SYMBOLIC_INTEGER_RELATION_NE,
    LOOM_SYMBOLIC_INTEGER_RELATION_LT, LOOM_SYMBOLIC_INTEGER_RELATION_LE,
    LOOM_SYMBOLIC_INTEGER_RELATION_GT, LOOM_SYMBOLIC_INTEGER_RELATION_GE,
};

Operand Value(loom_value_id_t value) {
  return {LOOM_CONDITION_INTEGER_OPERAND_VALUE, value, 0};
}

Operand Literal(int64_t value) {
  return {LOOM_CONDITION_INTEGER_OPERAND_CONSTANT, 0, value};
}

int64_t Payload(Operand operand) {
  return operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE
             ? static_cast<int64_t>(operand.value_id)
             : operand.constant;
}

Row CanonicalRow(Fact fact) {
  if (std::make_pair(fact.right.kind, Payload(fact.right)) <
      std::make_pair(fact.left.kind, Payload(fact.left))) {
    std::swap(fact.left, fact.right);
    fact.relation = loom_symbolic_integer_relation_swap(fact.relation);
  }
  return {fact.left.kind, Payload(fact.left), fact.right.kind,
          Payload(fact.right), fact.relation};
}

bool Holds(Relation relation, int64_t left, int64_t right) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      return left == right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
      return left != right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      return left < right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      return left <= right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      return left > right;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      return left >= right;
  }
  return false;
}

class ConditionDerivationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &scratch_arena_);
    iree_arena_initialize(&pool_, &result_arena_);
    loom_condition_derivation_initialize(&result_arena_, &derivation_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&result_arena_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Add(Relation relation, Operand left, Operand right) {
    IREE_CHECK_OK(loom_condition_derivation_append_integer_relation(
        &derivation_, {relation, left, right}));
  }

  void Complete(const loom_value_fact_table_t* table = nullptr) {
    IREE_CHECK_OK(loom_condition_derivation_complete(table, &scratch_arena_,
                                                     &scratch_, &derivation_));
  }

  std::vector<Row> Rows() const {
    std::vector<Row> rows;
    const auto& facts = derivation_.integer_facts;
    for (size_t i = 0; i < facts.integer_relation_count; ++i) {
      const auto& fact = facts.integer_relations[i];
      const Row row = {fact.left.kind, Payload(fact.left), fact.right.kind,
                       Payload(fact.right), fact.relation};
      EXPECT_EQ(row, CanonicalRow(fact));
      rows.push_back(row);
    }
    EXPECT_TRUE(std::is_sorted(rows.begin(), rows.end()));
    EXPECT_EQ(std::adjacent_find(rows.begin(), rows.end()), rows.end());
    return rows;
  }

  void ExpectFacts(std::initializer_list<Fact> expected) {
    std::vector<Row> rows;
    for (const auto& fact : expected) {
      rows.push_back(CanonicalRow(fact));
    }
    std::sort(rows.begin(), rows.end());
    EXPECT_EQ(Rows(), rows);
  }

  void Chain(uint32_t count, uint32_t base = 0) {
    loom_condition_derivation_reset(&derivation_);
    for (uint32_t i = 0; i < count; ++i) {
      Add(LOOM_SYMBOLIC_INTEGER_RELATION_LE, Value(base + i),
          Value(base + i + 1));
    }
    Complete();
  }

  // Shared pool for independently owned scratch and retained results.
  iree_arena_block_pool_t pool_ = {};
  // Reusable construction state, independent of the published result.
  iree_arena_allocator_t scratch_arena_ = {};
  // Storage retained by the current result.
  iree_arena_allocator_t result_arena_ = {};
  // Lazy construction owner; no semantic state survives completion.
  loom_condition_composition_scratch_t* scratch_ = nullptr;
  // Current complete or in-progress relation result.
  loom_condition_derivation_t derivation_ = {};
};

TEST_F(ConditionDerivationTest, EmptyAndSingletonNeedNoConstructionStorage) {
  Complete();
  EXPECT_TRUE(Rows().empty());
  EXPECT_EQ(scratch_, nullptr);
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_GE, Value(7), Literal(INT64_MIN));
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_LE, Literal(INT64_MIN), Value(7));
  Complete();
  ExpectFacts(
      {{LOOM_SYMBOLIC_INTEGER_RELATION_GE, Value(7), Literal(INT64_MIN)}});
  EXPECT_EQ(scratch_, nullptr);
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
}

TEST_F(ConditionDerivationTest, BooleanTruthIsSortedDeduplicatedAndReset) {
  for (const auto fact : {loom_condition_boolean_fact_t{8, true},
                          {2, false},
                          {8, true},
                          {8, false}}) {
    IREE_ASSERT_OK(loom_condition_derivation_append_boolean_fact(
        &derivation_, fact.value_id, fact.value));
  }
  Complete();
  ASSERT_EQ(derivation_.boolean_fact_count, 3u);
  EXPECT_EQ(derivation_.boolean_facts[0].value_id, 2u);
  EXPECT_FALSE(derivation_.boolean_facts[0].value);
  EXPECT_EQ(derivation_.boolean_facts[1].value_id, 8u);
  EXPECT_FALSE(derivation_.boolean_facts[1].value);
  EXPECT_EQ(derivation_.boolean_facts[2].value_id, 8u);
  EXPECT_TRUE(derivation_.boolean_facts[2].value);
  const auto* storage = derivation_.boolean_facts;
  loom_condition_derivation_reset(&derivation_);
  Complete();
  EXPECT_EQ(derivation_.boolean_fact_count, 0u);
  EXPECT_EQ(derivation_.boolean_facts, storage);
}

TEST_F(ConditionDerivationTest, EveryPredicatePairMatchesIntegerOrderings) {
  // All weak orders of three integers are represented by values in [-1, 1].
  // Enumerating them supplies an independent truth oracle for each composition.
  for (Relation first : kRelations) {
    for (Relation second : kRelations) {
      SCOPED_TRACE(::testing::Message() << first << "/" << second);
      for (bool reverse : {false, true}) {
        loom_condition_derivation_reset(&derivation_);
        const Fact sources[] = {{first, Value(29), Value(4)},
                                {second, Value(4), Value(201)}};
        for (unsigned i = 0; i < 2; ++i) {
          const auto fact = sources[reverse ? 1 - i : i];
          Add(fact.relation, fact.left, fact.right);
          Add(loom_symbolic_integer_relation_swap(fact.relation), fact.right,
              fact.left);
        }
        Complete();
        uint8_t possible = 0;
        for (int64_t x = -1; x <= 1; ++x) {
          for (int64_t y = -1; y <= 1; ++y) {
            for (int64_t z = -1; z <= 1; ++z) {
              if (!Holds(first, x, y) || !Holds(second, y, z)) {
                continue;
              }
              possible |= x < z ? 1u : (x == z ? 2u : 4u);
              for (size_t i = 0;
                   i < derivation_.integer_facts.integer_relation_count; ++i) {
                const auto& fact =
                    derivation_.integer_facts.integer_relations[i];
                const auto value = [&](Operand operand) {
                  return operand.value_id == 29  ? x
                         : operand.value_id == 4 ? y
                                                 : z;
                };
                EXPECT_TRUE(
                    Holds(fact.relation, value(fact.left), value(fact.right)));
              }
            }
          }
        }
        std::vector<Row> expected = {CanonicalRow(sources[0]),
                                     CanonicalRow(sources[1])};
        for (Relation relation : kRelations) {
          uint8_t outcomes = 0;
          if (Holds(relation, 0, 1)) {
            outcomes |= 1;
          }
          if (Holds(relation, 0, 0)) {
            outcomes |= 2;
          }
          if (Holds(relation, 1, 0)) {
            outcomes |= 4;
          }
          if (outcomes == possible) {
            expected.push_back(CanonicalRow({relation, Value(29), Value(201)}));
          }
        }
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(Rows(), expected);
      }
    }
  }
}

TEST_F(ConditionDerivationTest, FullWidthOperandsAndLiteralTableSurviveReuse) {
  for (int64_t literal : {INT64_MIN, INT64_C(-1), INT64_C(0), INT64_MAX}) {
    loom_condition_derivation_reset(&derivation_);
    const auto first = Value(UINT32_MAX - 1);
    const auto second = Value(UINT32_MAX - 2);
    Add(LOOM_SYMBOLIC_INTEGER_RELATION_EQ, first, second);
    Add(LOOM_SYMBOLIC_INTEGER_RELATION_NE, second, Literal(literal));
    Complete();
    ExpectFacts({{LOOM_SYMBOLIC_INTEGER_RELATION_EQ, first, second},
                 {LOOM_SYMBOLIC_INTEGER_RELATION_NE, second, Literal(literal)},
                 {LOOM_SYMBOLIC_INTEGER_RELATION_NE, first, Literal(literal)}});
  }
}

TEST_F(ConditionDerivationTest, SmallGraphsAreSoundAndIndependentOfInputOrder) {
  constexpr unsigned kPairs[3][2] = {{0, 1}, {1, 2}, {0, 2}};
  for (unsigned graph = 0; graph < 343; ++graph) {
    SCOPED_TRACE(graph);
    const unsigned choices[] = {graph % 7, graph / 7 % 7, graph / 49};
    std::array<unsigned, 3> order = {0, 1, 2};
    std::vector<Row> expected;
    do {
      loom_condition_derivation_reset(&derivation_);
      for (unsigned edge : order) {
        if (choices[edge] == 0) {
          continue;
        }
        const auto relation = kRelations[choices[edge] - 1];
        const auto left = Value(kPairs[edge][0]);
        const auto right = Value(kPairs[edge][1]);
        Add(relation, left, right);
        Add(loom_symbolic_integer_relation_swap(relation), right, left);
      }
      Complete();
      if (order == std::array<unsigned, 3>{0, 1, 2}) {
        expected = Rows();
      } else {
        EXPECT_EQ(Rows(), expected);
      }
      for (unsigned edge = 0; edge < 3; ++edge) {
        if (choices[edge] == 0) {
          continue;
        }
        EXPECT_TRUE(std::binary_search(
            expected.begin(), expected.end(),
            CanonicalRow({kRelations[choices[edge] - 1], Value(kPairs[edge][0]),
                          Value(kPairs[edge][1])})));
      }
    } while (std::next_permutation(order.begin(), order.end()));
    for (int64_t x = -1; x <= 1; ++x) {
      for (int64_t y = -1; y <= 1; ++y) {
        for (int64_t z = -1; z <= 1; ++z) {
          const int64_t values[] = {x, y, z};
          bool feasible = true;
          for (unsigned edge = 0; edge < 3; ++edge) {
            feasible &= choices[edge] == 0 ||
                        Holds(kRelations[choices[edge] - 1],
                              values[kPairs[edge][0]], values[kPairs[edge][1]]);
          }
          if (!feasible) {
            continue;
          }
          for (const auto& row : expected) {
            EXPECT_TRUE(Holds(std::get<4>(row), values[std::get<1>(row)],
                              values[std::get<3>(row)]));
          }
        }
      }
    }
  }
}

TEST_F(ConditionDerivationTest, LiteralAnchorsAndDisconnectedComponents) {
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_LT, Literal(-9), Value(17));
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_LE, Value(17), Literal(19));
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(2), Value(3));
  Complete();
  ExpectFacts({{LOOM_SYMBOLIC_INTEGER_RELATION_LT, Literal(-9), Value(17)},
               {LOOM_SYMBOLIC_INTEGER_RELATION_LE, Value(17), Literal(19)},
               {LOOM_SYMBOLIC_INTEGER_RELATION_LT, Literal(-9), Literal(19)},
               {LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(2), Value(3)}});
}

TEST_F(ConditionDerivationTest, OriginalSelfRelationsRemainButClosureAddsNone) {
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(0), Value(0));
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(0), Value(1));
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(1), Value(2));
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(2), Value(0));
  Complete();
  ExpectFacts({{LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(0), Value(0)},
               {LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(0), Value(1)},
               {LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(0), Value(2)},
               {LOOM_SYMBOLIC_INTEGER_RELATION_EQ, Value(1), Value(2)}});
}

TEST_F(ConditionDerivationTest, RetainedAliasesExtendOnlyOriginalAnchors) {
  loom_value_fact_table_t table;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &result_arena_, 8));
  const auto unknown = loom_value_facts_unknown();
  bool changed = false;
  for (loom_value_id_t value : {3u, 2u}) {
    const loom_value_id_t bound = value + 1;
    IREE_ASSERT_OK(loom_value_fact_integer_relations_define_upper_bounds(
        &table, value, &bound, nullptr, &unknown, 1,
        LOOM_SYMBOLIC_INTEGER_RELATION_LE, &unknown, &changed));
  }
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_LT, Value(1), Value(2));
  Complete(&table);
  ExpectFacts({{LOOM_SYMBOLIC_INTEGER_RELATION_LT, Value(1), Value(2)},
               {LOOM_SYMBOLIC_INTEGER_RELATION_LT, Value(1), Value(3)},
               {LOOM_SYMBOLIC_INTEGER_RELATION_LT, Value(1), Value(4)}});
  // Ambient-only edges remain in their owner rather than entering this result.
  loom_condition_derivation_reset(&derivation_);
  Add(LOOM_SYMBOLIC_INTEGER_RELATION_LT, Value(5), Value(6));
  Complete(&table);
  ExpectFacts({{LOOM_SYMBOLIC_INTEGER_RELATION_LT, Value(5), Value(6)}});
}

TEST_F(ConditionDerivationTest, LongChainsProduceEveryPairOnce) {
  for (uint32_t count : {8u, 32u, 128u}) {
    SCOPED_TRACE(count);
    Chain(count);
    ASSERT_EQ(derivation_.integer_facts.integer_relation_count,
              count * (count + 1) / 2);
    size_t index = 0;
    for (uint32_t left = 0; left < count; ++left) {
      for (uint32_t right = left + 1; right <= count; ++right) {
        EXPECT_EQ(
            CanonicalRow(derivation_.integer_facts.integer_relations[index++]),
            CanonicalRow({LOOM_SYMBOLIC_INTEGER_RELATION_LE, Value(left),
                          Value(right)}));
      }
    }
  }
}

TEST_F(ConditionDerivationTest, RepeatedLargeAndSmallQueriesReuseCapacity) {
  Chain(128);
  const auto scratch_used = scratch_arena_.used_allocation_size;
  const auto result_used = result_arena_.used_allocation_size;
  for (uint32_t iteration = 0; iteration < 64; ++iteration) {
    const uint32_t count = iteration % 8 == 0 ? 128 : 2;
    Chain(count, 1000 + iteration * 256);
    EXPECT_EQ(derivation_.integer_facts.integer_relation_count,
              count * (count + 1) / 2);
    EXPECT_EQ(scratch_arena_.used_allocation_size, scratch_used);
    EXPECT_EQ(result_arena_.used_allocation_size, result_used);
    EXPECT_EQ(derivation_.integer_facts.integer_relations[0].left.value_id,
              1000 + iteration * 256);
  }
}

TEST_F(ConditionDerivationTest, GraduallyLargerResultsKeepStorageBounded) {
  for (uint32_t count = 1; count <= 128; ++count) {
    Chain(count);
    ASSERT_EQ(derivation_.integer_facts.integer_relation_count,
              count * (count + 1) / 2);
  }
  // An arena retains prior array generations. Across successively larger
  // queries, geometric growth bounds their total storage by twice the final
  // capacity, which is less than twice the final result size. Exact-sizing
  // each query instead retains the sum of all intermediate closure sizes.
  EXPECT_LE(
      result_arena_.used_allocation_size,
      4 * derivation_.integer_facts.integer_relation_count * sizeof(Fact));
}

TEST_F(ConditionDerivationTest, ScratchDoesNotBorrowReleasedResultArenas) {
  for (uint32_t iteration = 0; iteration < 32; ++iteration) {
    Chain(32, 1000 * iteration);
    const auto expected = Rows();
    iree_arena_reset(&result_arena_);
    loom_condition_derivation_initialize(&result_arena_, &derivation_);
    Chain(32, 1000 * iteration);
    EXPECT_EQ(Rows(), expected);
  }
  const auto expected = Rows();
  iree_arena_reset(&scratch_arena_);
  scratch_ = nullptr;
  EXPECT_EQ(Rows(), expected);
  Chain(32, 1000 * 31);
  EXPECT_EQ(Rows(), expected);
}

class ConditionDerivationFailureTest : public ConditionDerivationTest {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ConditionDerivationFailureTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected derivation allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(512, {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &scratch_arena_);
    iree_arena_initialize(&pool_, &result_arena_);
    loom_condition_derivation_initialize(&result_arena_, &derivation_);
  }

  void Prepare() {
    loom_condition_derivation_reset(&derivation_);
    // Include literals so failure injection reaches the literal dictionary as
    // well as row segments, hash growth, and final result publication.
    for (uint32_t i = 0; i < 32; ++i) {
      Add(LOOM_SYMBOLIC_INTEGER_RELATION_LE, Value(i), Value(i + 1));
      Add(LOOM_SYMBOLIC_INTEGER_RELATION_LT, Value(i), Literal(INT64_MAX - i));
    }
    allocation_count_ = 0;
  }

  void CheckFailures(uint32_t warm_count);

  // Allocations attempted since the current completion began.
  size_t allocation_count_ = 0;
  // Failing allocation ordinal, or SIZE_MAX for normal operation.
  size_t failure_index_ = SIZE_MAX;
};

void ConditionDerivationFailureTest::CheckFailures(uint32_t warm_count) {
  Chain(warm_count);
  Prepare();
  Complete();
  const auto expected = Rows();
  const auto allocation_count = allocation_count_;
  ASSERT_GT(allocation_count, 0u);
  for (size_t failure = 0; failure < allocation_count; ++failure) {
    SCOPED_TRACE(failure);
    TearDown();
    scratch_ = nullptr;
    SetUp();
    Chain(warm_count);
    Prepare();
    failure_index_ = failure;
    IREE_ASSERT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_condition_derivation_complete(nullptr, &scratch_arena_, &scratch_,
                                           &derivation_));
    failure_index_ = SIZE_MAX;
    Prepare();
    Complete();
    EXPECT_EQ(Rows(), expected);
    const auto used = scratch_arena_.used_allocation_size;
    Prepare();
    Complete();
    EXPECT_EQ(Rows(), expected);
    EXPECT_EQ(scratch_arena_.used_allocation_size, used);
  }
}
TEST_F(ConditionDerivationFailureTest, ColdFailureLeavesReusableScratch) {
  CheckFailures(0);
}

TEST_F(ConditionDerivationFailureTest, GrowthFailurePreservesWarmStorage) {
  CheckFailures(8);
}

}  // namespace
}  // namespace loom
