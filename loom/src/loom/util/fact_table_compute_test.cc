// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/condition_edge_projection.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/facts.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_induction.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactTableComputeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_index_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_INDEX, vtables, static_cast<uint16_t>(count)));
    vtables = loom_cfg_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_CFG, vtables, static_cast<uint16_t>(count)));
    vtables = loom_scf_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, vtables, static_cast<uint16_t>(count)));
    vtables = loom_scalar_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCALAR, vtables, static_cast<uint16_t>(count)));
    vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, static_cast<uint16_t>(count)));
    vtables = loom_vector_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_VECTOR, vtables, static_cast<uint16_t>(count)));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("facts"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 4));
    for (loom_value_id_t& input : inputs_) {
      IREE_ASSERT_OK(loom_builder_define_value(&builder_, type_, &input));
      IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input,
                                                  loom_value_facts_unknown()));
    }
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t BuildIdentity(loom_value_id_t input, loom_op_t** out_op) {
    return loom_index_assume_build(&builder_, &input, 1, nullptr, 0, &type_, 1,
                                   LOOM_LOCATION_UNKNOWN, out_op);
  }

  loom_value_id_t DefineValue(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_value(&builder_, type, &value));
    IREE_CHECK_OK(loom_value_fact_table_define(&table_, value,
                                               loom_value_facts_unknown()));
    return value;
  }

  static std::vector<loom_op_t*> PendingExactRelations(
      const loom_value_fact_table_t& table) {
    loom_op_t* const* ops = nullptr;
    iree_host_size_t op_count = 0;
    loom_value_fact_table_pending_exact_relations(&table, &ops, &op_count);
    return op_count ? std::vector<loom_op_t*>(ops, ops + op_count)
                    : std::vector<loom_op_t*>();
  }

  void ExpectSelector(loom_value_id_t value, loom_value_id_t selector) {
    loom_value_set_cursor_t cursor;
    loom_value_fact_table_select_dependencies_begin(&table_, value, &cursor);
    EXPECT_EQ(loom_value_set_cursor_next(&cursor), selector);
    EXPECT_EQ(loom_value_set_cursor_next(&cursor), LOOM_VALUE_ID_INVALID);
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t table_;
  const loom_type_t type_ = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t inputs_[2];
};

// Exercises the immutable replay contract used by nested loop and CFG owners.
class FactTableReplayTest : public FactTableComputeTest {
 protected:
  void SetUp() override {
    FactTableComputeTest::SetUp();
    lhs_ = DefineValue(integer_type_);
    rhs_ = DefineValue(integer_type_);
    SetInput(lhs_, 3);
    SetInput(rhs_, 4);
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &builder_, 0, lhs_, rhs_, integer_type_, LOOM_LOCATION_UNKNOWN, &sum_));
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &builder_, 0, loom_scalar_addi_result(sum_), rhs_, integer_type_,
        LOOM_LOCATION_UNKNOWN, &consumer_));
    IREE_ASSERT_OK(loom_value_fact_table_enable_replay_generations(&table_));
    Expect(sum_, 7);
    Expect(consumer_, 11);
    table_.replaying = true;
  }

  void SetInput(loom_value_id_t value, int64_t number) {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, value, loom_value_facts_exact_i64(number)));
  }

  void Expect(loom_op_t* op, int64_t number) {
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, op));
    const auto facts =
        loom_value_fact_table_lookup(&table_, loom_scalar_addi_result(op));
    ASSERT_TRUE(loom_value_facts_is_exact(facts));
    EXPECT_EQ(facts.range_lo, number);
  }

  // Scalar arithmetic keeps the API fixture independent of origin families.
  const loom_type_t integer_type_ = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  // Input whose numerical publications change between immutable visits.
  loom_value_id_t lhs_;
  // Independent input shared by the producer and its consumer.
  loom_value_id_t rhs_;
  // Producer revisited while an inner solve converges.
  loom_op_t* sum_ = nullptr;
  // Consumer that observes the producer after convergence.
  loom_op_t* consumer_ = nullptr;
};

TEST_F(FactTableReplayTest, InnerConvergencePreservesUnconsumedChanges) {
  SetInput(lhs_, 5);
  Expect(sum_, 9);
  Expect(sum_, 9);
  Expect(consumer_, 13);
  SetInput(lhs_, 3);
  Expect(sum_, 7);
  Expect(sum_, 7);
  Expect(consumer_, 11);
}

TEST_F(FactTableReplayTest, ReplacedAndUndefinedResultsMustRecompute) {
  const auto result = loom_scalar_addi_result(sum_);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, result,
                                              loom_value_facts_unknown()));
  Expect(sum_, 7);
  loom_value_fact_table_undefine(&table_, result);
  Expect(sum_, 7);
}

TEST_F(FactTableReplayTest, RefinementCanReadAnUnlistedValue) {
  auto source = DefineValue(integer_type_);
  table_.context.refine_operands.user_data = &source;
  table_.context.refine_operands.fn =
      [](void* user_data, const loom_value_fact_table_t* table,
         const loom_op_t*, loom_value_facts_t* facts) {
        facts[0] = loom_value_fact_table_lookup(
            table, *static_cast<loom_value_id_t*>(user_data));
      };
  SetInput(source, 11);
  Expect(sum_, 15);
  SetInput(source, 13);
  Expect(sum_, 17);
  table_.context.refine_operands.fn = nullptr;
  table_.context.refine_operands.user_data = nullptr;
}

TEST_F(FactTableReplayTest, CounterRestartExpiresEveryEvaluation) {
  table_.replay.generation = UINT32_MAX - 1;
  SetInput(lhs_, 5);
  Expect(sum_, 9);
  Expect(consumer_, 13);
  SetInput(rhs_, 6);
  Expect(sum_, 11);
  Expect(consumer_, 17);
}

TEST_F(FactTableReplayTest, StorageGrowthAndScopeResetPreserveInvalidation) {
  IREE_ASSERT_OK(loom_value_fact_table_reserve(&table_, 4096));
  SetInput(lhs_, 5);
  Expect(sum_, 9);
  Expect(consumer_, 13);
  loom_value_fact_table_clear_scope(&table_);
  SetInput(lhs_, 11);
  SetInput(rhs_, 7);
  IREE_ASSERT_OK(loom_value_fact_table_enable_replay_generations(&table_));
  Expect(sum_, 18);
  Expect(consumer_, 25);
}

TEST_F(FactTableReplayTest, SemanticEditsStartOutsideReplay) {
  table_.replaying = false;
  IREE_ASSERT_OK(loom_op_set_operand(module_, sum_, 0, rhs_));
  Expect(sum_, 8);
  table_.replaying = true;
  Expect(consumer_, 12);
}

TEST_F(FactTableReplayTest, EveryResultOfAnEquationMustRemainPublished) {
  loom_op_t* division = nullptr;
  const loom_type_t types[] = {integer_type_, integer_type_};
  IREE_ASSERT_OK(loom_scalar_divremsi_build(&builder_, lhs_, rhs_, types,
                                            LOOM_LOCATION_UNKNOWN, &division));
  const auto quotient = loom_scalar_divremsi_quotient(division);
  const auto remainder = loom_scalar_divremsi_remainder(division);
  auto expect = [&](int64_t expected_quotient, int64_t expected_remainder) {
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, division));
    EXPECT_EQ(loom_value_fact_table_lookup(&table_, quotient).range_lo,
              expected_quotient);
    EXPECT_EQ(loom_value_fact_table_lookup(&table_, remainder).range_lo,
              expected_remainder);
  };
  SetInput(lhs_, 11);
  SetInput(rhs_, 3);
  expect(3, 2);
  loom_value_fact_table_undefine(&table_, remainder);
  expect(3, 2);
  SetInput(lhs_, 12);
  expect(4, 0);
}

TEST_F(FactTableReplayTest, ChangedExtensionInvalidatesEqualNumericEnvelope) {
  const auto type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                        loom_dim_pack_static(2), 0);
  const auto input = DefineValue(type);
  loom_op_t* sum = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(&builder_, 0, input, input, type,
                                        LOOM_LOCATION_UNKNOWN, &sum));
  for (int64_t first : {1, 9, 1}) {
    const loom_value_facts_t lanes[] = {
        loom_value_facts_exact_i64(first),
        loom_value_facts_exact_i64(10 - first),
    };
    loom_value_facts_t facts;
    IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
        &table_.context, {lanes, IREE_ARRAYSIZE(lanes)}, &facts));
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input, facts));
    for (unsigned visit = 0; visit < 2; ++visit) {
      IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, sum));
      loom_value_fact_small_static_lanes_t result;
      ASSERT_TRUE(loom_value_facts_query_small_static_lanes(
          &table_.context,
          loom_value_fact_table_lookup(&table_, loom_vector_addi_result(sum)),
          &result));
      ASSERT_EQ(result.count, 2u);
      EXPECT_EQ(result.lanes[0].range_lo, 2 * first);
      EXPECT_EQ(result.lanes[1].range_lo, 2 * (10 - first));
    }
  }
}

TEST_F(FactTableReplayTest, IdentityChangesDoNotRequireNumericChanges) {
  for (auto input : {lhs_, rhs_}) {
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, input,
                                                loom_value_facts_unknown()));
  }
  loom_op_t* comparison = nullptr;
  IREE_ASSERT_OK(
      loom_scalar_cmpi_build(&builder_, LOOM_SCALAR_CMPI_PREDICATE_EQ, lhs_,
                             rhs_, LOOM_LOCATION_UNKNOWN, &comparison));
  const auto result = loom_scalar_cmpi_result(comparison);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&table_, module_, comparison));
  EXPECT_FALSE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table_, result)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define_identity(&table_, rhs_, lhs_, nullptr));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&table_, module_, comparison));
  EXPECT_TRUE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table_, result)));
  EXPECT_EQ(loom_value_fact_table_lookup(&table_, result).range_lo, 1);
  loom_value_fact_table_undefine_identity(&table_, rhs_);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&table_, module_, comparison));
  EXPECT_FALSE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(&table_, result)));
}

class FactTableScratchTest : public FactTableComputeTest {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<FactTableScratchTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      test->failure_active_count_ = test->table_.scratch.regions.count;
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected fact scratch allocation failure");
    }
    const auto allocator = iree_allocator_system();
    const bool had_allocation =
        command == IREE_ALLOCATOR_COMMAND_REALLOC && *pointer != nullptr;
    iree_status_t status =
        allocator.ctl(allocator.self, command, parameters, pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        --test->live_allocation_count_;
      } else if (!had_allocation) {
        ++test->live_allocation_count_;
      }
    }
    return status;
  }

  void SetUp() override {
    FactTableComputeTest::SetUp();
    // Small blocks expose individual scratch and region-table allocations.
    iree_arena_block_pool_initialize(64, {this, Allocate}, &transient_pool_);
    iree_arena_initialize(&transient_pool_, &transient_arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize_with_arenas(
        &table_, &arena_, &transient_arena_, 4));
    SeedInputs();
    condition_ = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  }

  void TearDown() override {
    iree_arena_deinitialize(&transient_arena_);
    iree_arena_block_pool_deinitialize(&transient_pool_);
    FactTableComputeTest::TearDown();
  }

  void SeedInputs() {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, inputs_[0], loom_value_facts_make(2, 4, 2)));
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, inputs_[1], loom_value_facts_make(8, 16, 4)));
  }

  // Each level forwards two different facts through both branches. A nested
  // branch on the true edge requires a continuation and a complete result join.
  loom_op_t* BuildBranches(unsigned depth, loom_op_t** out_leaf = nullptr) {
    const loom_type_t types[] = {type_, type_};
    loom_op_t* branch = nullptr;
    IREE_CHECK_OK(loom_test_branch_build(&builder_, condition_, types,
                                         IREE_ARRAYSIZE(types), nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &branch));
    if (depth == 1 && out_leaf) {
      *out_leaf = branch;
    }
    for (uint8_t i = 0; i < branch->region_count; ++i) {
      const auto saved = loom_builder_enter_region(&builder_, branch,
                                                   loom_op_regions(branch)[i]);
      const loom_value_id_t* yielded = inputs_;
      if (i == 0 && depth > 1) {
        yielded = loom_op_const_results(BuildBranches(depth - 1, out_leaf));
      }
      loom_op_t* terminator = nullptr;
      IREE_CHECK_OK(loom_test_yield_build(&builder_, yielded, 2,
                                          LOOM_LOCATION_UNKNOWN, &terminator));
      loom_builder_restore(&builder_, saved);
    }
    return branch;
  }

  void ExpectResults(loom_op_t* branch) {
    for (uint16_t i = 0; i < branch->result_count; ++i) {
      const auto expected = loom_value_fact_table_lookup(&table_, inputs_[i]);
      const auto actual = loom_value_fact_table_lookup(
          &table_, loom_op_const_results(branch)[i]);
      EXPECT_EQ(actual.range_lo, expected.range_lo);
      EXPECT_EQ(actual.range_hi, expected.range_hi);
      EXPECT_EQ(actual.known_divisor, expected.known_divisor);
    }
    EXPECT_EQ(table_.scratch.regions.count, 0u);
  }

  // Scope-lived extension and scratch backing, separate from retained entries.
  iree_arena_block_pool_t transient_pool_ = {};
  // Reset after clearing the table, matching pass-owned fact scopes.
  iree_arena_allocator_t transient_arena_ = {};
  // Number of attempted transient backing allocations.
  iree_host_size_t allocation_count_ = 0;
  // Backing allocations still owned by a transient arena or its block pool.
  iree_host_size_t live_allocation_count_ = 0;
  // Backing allocation to fail, or SIZE_MAX for unrestricted allocation.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Live continuations when the selected allocation failed.
  iree_host_size_t failure_active_count_ = 0;
  // Unknown selector keeps both branch contributions observable.
  loom_value_id_t condition_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(FactTableScratchTest, FlatRegionNeedsNoContinuationStorage) {
  loom_op_t* identity = nullptr;
  IREE_ASSERT_OK(BuildIdentity(inputs_[0], &identity));
  IREE_ASSERT_OK(loom_value_fact_table_compute_region_tree(
      &table_, module_, module_->body, nullptr));
  EXPECT_EQ(table_.scratch.regions.values, nullptr);
  EXPECT_EQ(table_.scratch.regions.capacity, 0u);
  EXPECT_EQ(
      loom_value_fact_table_lookup(&table_, loom_op_const_results(identity)[0])
          .range_lo,
      2);
}

TEST_F(FactTableScratchTest, RepeatedBranchQueriesReuseScopeStorage) {
  loom_op_t* branch = BuildBranches(4);
  for (unsigned scope = 0; scope < 2; ++scope) {
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, branch));
    ASSERT_GT(table_.scratch.regions.capacity, 0u);
    const auto* frames = table_.scratch.regions.values;
    const auto used_bytes = transient_arena_.used_allocation_size;
    for (unsigned query = 0; query < 16; ++query) {
      IREE_ASSERT_OK(
          loom_value_fact_table_compute_op(&table_, module_, branch));
      ExpectResults(branch);
      EXPECT_EQ(table_.scratch.regions.values, frames);
      EXPECT_EQ(transient_arena_.used_allocation_size, used_bytes);
    }
    loom_value_fact_table_clear_scope(&table_);
    EXPECT_EQ(table_.scratch.regions.values, nullptr);
    EXPECT_EQ(table_.scratch.regions.capacity, 0u);
    iree_arena_reset(&transient_arena_);
    SeedInputs();
  }
}

TEST_F(FactTableScratchTest, LoopReentryGrowsPastSuspendedContinuations) {
  loom_op_t* leaf = nullptr;
  loom_op_t* branch = BuildBranches(4, &leaf);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, branch));
  const auto* frames = table_.scratch.regions.values;
  const auto capacity = table_.scratch.regions.capacity;

  loom_builder_set_before(
      &builder_,
      loom_region_entry_block(loom_test_branch_then_region(leaf))->last_op);
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(&builder_, nullptr, 0, nullptr, nullptr,
                                      0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                      &loop));
  const auto saved =
      loom_builder_enter_region(&builder_, loop, loom_scf_while_before(loop));
  loom_op_t* condition = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(&builder_, condition_, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &condition));
  loom_builder_restore(&builder_, saved);
  const auto body_saved =
      loom_builder_enter_region(&builder_, loop, loom_scf_while_after(loop));
  loom_op_t* inner = BuildBranches(4);
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, body_saved);
  loom_op_t* after_loop = nullptr;
  IREE_ASSERT_OK(BuildIdentity(inputs_[0], &after_loop));

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, branch));
  EXPECT_NE(table_.scratch.regions.values, frames);
  EXPECT_GT(table_.scratch.regions.capacity, capacity);
  ExpectResults(branch);
  ExpectResults(inner);
  const auto resumed = loom_value_fact_table_lookup(
      &table_, loom_op_const_results(after_loop)[0]);
  EXPECT_EQ(resumed.range_lo, 2);
  EXPECT_EQ(resumed.range_hi, 4);
}

TEST_F(FactTableScratchTest, FailedTraversalReleasesActiveReservations) {
  loom_op_t* branch = BuildBranches(4);
  // Visit every failing allocation, including failures after descent has
  // reserved frames. Each failed query is retried on the same table.
  bool failed_during_descent = false;
  for (iree_host_size_t failure = 0;; ++failure) {
    loom_value_fact_table_clear_scope(&table_);
    iree_arena_reset(&transient_arena_);
    SeedInputs();
    allocation_count_ = 0;
    failure_index_ = failure;
    iree_status_t status =
        loom_value_fact_table_compute_op(&table_, module_, branch);
    if (iree_status_is_ok(status)) {
      ExpectResults(branch);
      break;
    }
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    failed_during_descent |= failure_active_count_ > 0;
    EXPECT_EQ(table_.scratch.regions.count, 0u);
    failure_index_ = SIZE_MAX;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, branch));
    ExpectResults(branch);
  }
  EXPECT_TRUE(failed_during_descent);
}

class FactTableLoopScratchTest : public FactTableScratchTest,
                                 public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    FactTableScratchTest::SetUp();
    for (auto& value : carried_) {
      value = DefineValue(vector_type_);
    }
    SeedCarriedFacts();
  }

  void SeedCarriedFacts() {
    for (unsigned i = 0; i < IREE_ARRAYSIZE(carried_); ++i) {
      const loom_value_facts_t lanes[] = {
          loom_value_facts_exact_i64(4 * i + 1),
          loom_value_facts_exact_i64(4 * i + 3),
      };
      loom_value_facts_t facts;
      IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(
          &table_.context, {lanes, IREE_ARRAYSIZE(lanes)}, &facts));
      IREE_ASSERT_OK(loom_value_fact_table_define(&table_, carried_[i], facts));
    }
  }

  // Opposite loop kinds nest while the outer solve still needs its tuple.
  loom_op_t* BuildLoop(bool condition_loop, const loom_value_id_t* initial,
                       bool nested) {
    const loom_type_t types[] = {vector_type_, vector_type_};
    loom_op_t* loop = nullptr;
    if (condition_loop) {
      IREE_CHECK_OK(loom_scf_while_build(&builder_, initial, 2, nullptr, types,
                                         2, nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                         &loop));
      auto* before = loom_scf_while_before(loop);
      const auto saved = loom_builder_enter_region(&builder_, loop, before);
      const loom_value_id_t forwarded[] = {
          loom_region_entry_arg_id(before, 0),
          loom_region_entry_arg_id(before, 1),
      };
      loom_op_t* condition = nullptr;
      IREE_CHECK_OK(loom_scf_condition_build(&builder_, condition_, forwarded,
                                             2, LOOM_LOCATION_UNKNOWN,
                                             &condition));
      loom_builder_restore(&builder_, saved);
    } else {
      IREE_CHECK_OK(loom_scf_for_build(
          &builder_, 0, inputs_[0], inputs_[1], inputs_[0], initial, 2, nullptr,
          nullptr, 0, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0,
          LOOM_LOCATION_UNKNOWN, &loop));
    }
    auto* body =
        condition_loop ? loom_scf_while_after(loop) : loom_scf_for_body(loop);
    const auto saved = loom_builder_enter_region(&builder_, loop, body);
    const uint16_t offset = condition_loop ? 0 : 1;
    const loom_value_id_t arguments[] = {
        loom_region_entry_arg_id(body, offset),
        loom_region_entry_arg_id(body, offset + 1),
    };
    const auto* yielded =
        nested ? loom_op_const_results(BuildLoop(!condition_loop, arguments,
                                                 /*nested=*/false))
               : arguments;
    loom_op_t* yield = nullptr;
    IREE_CHECK_OK(loom_scf_yield_build(&builder_, yielded, 2,
                                       LOOM_LOCATION_UNKNOWN, &yield));
    loom_builder_restore(&builder_, saved);
    return loop;
  }

  void ExpectCarriedFacts(loom_op_t* loop) {
    for (unsigned i = 0; i < IREE_ARRAYSIZE(carried_); ++i) {
      const auto facts =
          loom_value_fact_table_lookup(&table_, loom_op_const_results(loop)[i]);
      loom_value_fact_small_static_lanes_t lanes;
      ASSERT_TRUE(loom_value_facts_query_small_static_lanes(&table_.context,
                                                            facts, &lanes));
      ASSERT_EQ(lanes.count, 2u);
      EXPECT_EQ(lanes.lanes[0].range_lo, 4 * i + 1);
      EXPECT_EQ(lanes.lanes[1].range_lo, 4 * i + 3);
    }
  }

  // Nonuniform lane facts require retained, table-owned extension payloads.
  const loom_type_t vector_type_ = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(2), 0);
  // Independent inputs forwarded through both nested loop tuples.
  loom_value_id_t carried_[2];
};

TEST_P(FactTableLoopScratchTest, RepeatedSolvesReleaseWorkingTuples) {
  loom_op_t* loop = BuildLoop(GetParam(), carried_, /*nested=*/true);
  for (unsigned scope = 0; scope < 2; ++scope) {
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
    const auto used_bytes = transient_arena_.used_allocation_size;
    for (unsigned query = 0; query < 16; ++query) {
      IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
      ExpectCarriedFacts(loop);
      EXPECT_EQ(transient_arena_.used_allocation_size, used_bytes);
    }
    loom_value_fact_table_clear_scope(&table_);
    iree_arena_reset(&transient_arena_);
    SeedInputs();
    SeedCarriedFacts();
  }
}

TEST_P(FactTableLoopScratchTest,
       NestedConstantsRefreshAfterEditsAndScopeReset) {
  loom_op_t* loop = BuildLoop(GetParam(), carried_, /*nested=*/true);
  const auto outer = loom_loop_like_cast(module_, loop);
  const auto inner = loom_loop_like_cast(
      module_, loom_region_entry_block(loom_loop_like_body(outer))->first_op);
  loom_op_t* yield =
      loom_region_entry_block(loom_loop_like_body(inner))->last_op;
  loom_builder_set_before(&builder_, yield);
  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(loom_vector_constant_build(&builder_, loom_attr_i64(7),
                                            vector_type_, LOOM_LOCATION_UNKNOWN,
                                            &constant));
  const auto result = loom_vector_constant_result(constant);
  IREE_ASSERT_OK(loom_op_set_operand(module_, yield, 0, result));

  for (unsigned scope = 0; scope < 2; ++scope) {
    for (int64_t value : {7, 11}) {
      IREE_ASSERT_OK(loom_vector_constant_set_value(module_, constant,
                                                    loom_attr_i64(value)));
      IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
      EXPECT_FALSE(table_.replaying);
      const auto facts = loom_value_fact_table_lookup(&table_, result);
      loom_value_fact_uniform_element_t uniform;
      ASSERT_TRUE(loom_value_facts_query_uniform_element(&table_.context, facts,
                                                         &uniform));
      EXPECT_TRUE(loom_value_facts_is_exact(uniform.element));
      EXPECT_EQ(uniform.element.range_lo, value);
      const auto used_bytes = transient_arena_.used_allocation_size;
      IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
      EXPECT_TRUE(loom_value_facts_equal(
          facts, loom_value_fact_table_lookup(&table_, result)));
      EXPECT_EQ(transient_arena_.used_allocation_size, used_bytes);
    }
    loom_value_fact_table_clear_scope(&table_);
    EXPECT_FALSE(loom_value_fact_table_has_entry(&table_, result));
    iree_arena_reset(&transient_arena_);
    SeedInputs();
    SeedCarriedFacts();
  }
}

TEST_P(FactTableLoopScratchTest, FailedSolvesReleaseWorkingTuples) {
  loom_op_t* loop = BuildLoop(GetParam(), carried_, /*nested=*/true);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  const auto used_bytes = transient_arena_.used_allocation_size;
  iree_arena_block_pool_trim(&transient_pool_);
  const auto live_allocations = live_allocation_count_;
  for (iree_host_size_t failure = 0;; ++failure) {
    allocation_count_ = 0;
    failure_index_ = failure;
    iree_status_t status =
        loom_value_fact_table_compute_op(&table_, module_, loop);
    EXPECT_FALSE(table_.replaying);
    // Returning scratch to the pool is sufficient; trimming exposes any
    // backing still stranded in a failed solve's arena.
    iree_arena_block_pool_trim(&transient_pool_);
    EXPECT_EQ(live_allocation_count_, live_allocations);
    EXPECT_EQ(transient_arena_.used_allocation_size, used_bytes);
    if (iree_status_is_ok(status)) {
      EXPECT_GT(failure, 0u);
      ExpectCarriedFacts(loop);
      break;
    }
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    failure_index_ = SIZE_MAX;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
    ExpectCarriedFacts(loop);
    iree_arena_block_pool_trim(&transient_pool_);
    EXPECT_EQ(live_allocation_count_, live_allocations);
    EXPECT_EQ(transient_arena_.used_allocation_size, used_bytes);
  }
}

TEST_P(FactTableLoopScratchTest, NestedSolvesRefreshStructuralDependencies) {
  loom_op_t* outer = BuildLoop(GetParam(), carried_, /*nested=*/true);
  loom_region_t* outer_body =
      loom_loop_like_body(loom_loop_like_cast(module_, outer));
  loom_op_t* inner = loom_region_entry_block(outer_body)->first_op;
  loom_region_t* inner_body =
      loom_loop_like_body(loom_loop_like_cast(module_, inner));
  loom_region_t* condition = loom_scf_while_before(GetParam() ? outer : inner);

  // Rotating distinct lane facts requires numerical convergence; the selector
  // dependencies are already complete after the first traversal.
  loom_op_t* yield = loom_region_entry_block(inner_body)->last_op;
  const auto first = loom_op_const_operands(yield)[0];
  const auto second = loom_op_const_operands(yield)[1];
  IREE_ASSERT_OK(loom_op_set_operand(module_, yield, 0, second));
  IREE_ASSERT_OK(loom_op_set_operand(module_, yield, 1, first));
  const auto replacement = DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_region_t* regions[] = {condition, inner_body, outer_body};
  loom_op_t* selections[IREE_ARRAYSIZE(regions)] = {};
  for (unsigned i = 0; i < IREE_ARRAYSIZE(regions); ++i) {
    loom_builder_set_before(&builder_,
                            loom_region_entry_block(regions[i])->last_op);
    IREE_ASSERT_OK(
        loom_scf_select_build(&builder_, condition_, inputs_[0], inputs_[1],
                              type_, LOOM_LOCATION_UNKNOWN, &selections[i]));
  }

  for (unsigned scope = 0; scope < 2; ++scope) {
    for (const auto selector : {condition_, replacement}) {
      for (auto* select : selections) {
        IREE_ASSERT_OK(loom_op_set_operand(module_, select, 0, selector));
      }
      IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, outer));
      for (auto* select : selections) {
        ExpectSelector(loom_scf_select_result(select), selector);
        const auto facts = loom_value_fact_table_lookup(
            &table_, loom_scf_select_result(select));
        EXPECT_EQ(facts.range_lo, 2);
        EXPECT_EQ(facts.range_hi, 16);
        bool changed = true;
        IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
            &table_, module_, select, &changed));
        EXPECT_FALSE(changed);
      }
    }
    loom_value_fact_table_clear_scope(&table_);
    iree_arena_reset(&transient_arena_);
    SeedInputs();
    SeedCarriedFacts();
  }
}

INSTANTIATE_TEST_SUITE_P(LoopKinds, FactTableLoopScratchTest,
                         ::testing::Bool());

TEST_F(FactTableComputeTest, InductionStepReadsCurrentFactsAndLiteralBits) {
  loom_value_fact_induction_t induction = {};
  induction.step.value = inputs_[0];
  induction.flags = LOOM_VALUE_FACT_INDUCTION_SUBTRACT_STEP;
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_make(2, 4, 2)));
  loom_value_facts_t delta =
      loom_value_fact_induction_step_facts(&table_, &induction);
  EXPECT_EQ(delta.range_lo, -4);
  EXPECT_EQ(delta.range_hi, -2);
  EXPECT_EQ(delta.known_divisor, 2);

  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_exact_i64(3)));
  delta = loom_value_fact_induction_step_facts(&table_, &induction);
  EXPECT_EQ(delta.range_lo, -3);
  EXPECT_EQ(delta.range_hi, -3);
  induction.flags = LOOM_VALUE_FACT_INDUCTION_NONE;
  delta = loom_value_fact_induction_step_facts(&table_, &induction);
  EXPECT_EQ(delta.range_lo, 3);
  EXPECT_EQ(delta.range_hi, 3);

  induction.flags = LOOM_VALUE_FACT_INDUCTION_SUBTRACT_STEP;
  induction.step.is_literal = true;
  induction.step.literal = INT64_MIN;
  loom_value_fact_table_undefine(&table_, inputs_[0]);
  delta = loom_value_fact_induction_step_facts(&table_, &induction);
  EXPECT_EQ(delta.range_lo, INT64_MIN);
  EXPECT_EQ(delta.range_hi, INT64_MIN);

  induction.step.is_literal = false;
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_induction_step_facts(&table_, &induction)));
}

TEST_F(FactTableComputeTest, RetainsComputedResultTypeKindsUntilScopeClear) {
  EXPECT_FALSE(
      loom_value_fact_table_has_result_type_kind(&table_, LOOM_TYPE_SCALAR));
  const loom_type_t result_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  for (loom_type_t result_type : result_types) {
    loom_op_t* constant = nullptr;
    IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0),
                                            result_type, LOOM_LOCATION_UNKNOWN,
                                            &constant));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, constant));
    EXPECT_TRUE(loom_value_fact_table_has_result_type_kind(
        &table_, loom_type_kind(result_type)));
  }

  loom_value_fact_table_clear_scope(&table_);
  for (loom_type_t result_type : result_types) {
    EXPECT_FALSE(loom_value_fact_table_has_result_type_kind(
        &table_, loom_type_kind(result_type)));
  }
}

TEST_F(FactTableComputeTest, IdentityOnlyMutationReportsChangedFacts) {
  EXPECT_EQ(table_.identities.entries, nullptr);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, inputs_[0]),
            inputs_[0]);
  EXPECT_EQ(loom_value_fact_table_query_identity(nullptr, inputs_[0]),
            inputs_[0]);
  loom_op_t* first = nullptr;
  IREE_ASSERT_OK(BuildIdentity(inputs_[0], &first));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             first, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(table_.select_dependencies.index, nullptr);
  EXPECT_EQ(table_.select_dependencies.pages, nullptr);
  const loom_value_id_t first_result = loom_op_const_results(first)[0];
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(BuildIdentity(first_result, &second));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, second));
  const loom_value_id_t second_result = loom_op_const_results(second)[0];
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, second_result),
            inputs_[0]);
  EXPECT_TRUE(
      loom_value_fact_table_values_equal(&table_, second_result, inputs_[0]));
  EXPECT_TRUE(
      loom_value_fact_table_values_equal(&table_, first_result, second_result));
  EXPECT_FALSE(
      loom_value_fact_table_values_equal(&table_, second_result, inputs_[1]));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             first, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(table_.exact_relations.ops, nullptr);

  IREE_ASSERT_OK(loom_op_set_operand(module_, first, 0, inputs_[1]));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             first, &changed));
  EXPECT_TRUE(changed);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             second, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, second_result),
            inputs_[1]);
  EXPECT_TRUE(
      loom_value_fact_table_values_equal(&table_, second_result, inputs_[1]));
  EXPECT_FALSE(
      loom_value_fact_table_values_equal(&table_, second_result, inputs_[0]));
  EXPECT_TRUE(loom_value_facts_is_unknown(
      loom_value_fact_table_lookup(&table_, second_result)));
}

TEST_F(FactTableComputeTest, ComparisonInferenceUsesRetainedIdentity) {
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(BuildIdentity(inputs_[0], &alias));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, alias));
  const loom_value_id_t refined = loom_op_const_results(alias)[0];
  for (const auto predicate :
       {LOOM_INDEX_CMP_PREDICATE_EQ, LOOM_INDEX_CMP_PREDICATE_NE}) {
    loom_op_t* compare = nullptr;
    IREE_ASSERT_OK(loom_index_cmp_build(&builder_, predicate, inputs_[0],
                                        refined, LOOM_LOCATION_UNKNOWN,
                                        &compare));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, compare));
    const auto facts =
        loom_value_fact_table_lookup(&table_, loom_index_cmp_result(compare));
    EXPECT_TRUE(loom_value_facts_is_exact(facts));
    EXPECT_EQ(facts.range_lo, predicate == LOOM_INDEX_CMP_PREDICATE_EQ);
  }
}

TEST_F(FactTableComputeTest, OpaqueDefinitionsRetainSelectedInputs) {
  const loom_value_id_t storage = DefineValue(loom_type_pool());
  loom_op_t* first = nullptr;
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, storage, type_,
                                               LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, storage, type_,
                                               LOOM_LOCATION_UNKNOWN, &second));
  const loom_value_id_t first_result = loom_test_read_resource_result(first);
  const loom_value_id_t second_result = loom_test_read_resource_result(second);

  loom_value_fact_table_t inputs;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&inputs, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &inputs, first_result, loom_value_facts_make(16, 64, 16)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&inputs, second_result,
                                              loom_value_facts_exact_i64(32)));
  IREE_ASSERT_OK(loom_value_fact_table_seed_values(
      &table_, {&inputs, &first_result, 1}, module_));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, second));
  auto facts = loom_value_fact_table_lookup(&table_, first_result);
  EXPECT_EQ(facts.range_lo, 16);
  EXPECT_EQ(facts.range_hi, 64);
  EXPECT_EQ(facts.known_divisor, 16);
  EXPECT_FALSE(loom_value_facts_is_exact(
      loom_value_fact_table_lookup(&table_, second_result)));

  // Cyclic inference may undefine local entries before computing them again.
  loom_value_fact_table_undefine(&table_, first_result);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  facts = loom_value_fact_table_lookup(&table_, first_result);
  EXPECT_EQ(facts.range_lo, 16);
  EXPECT_EQ(facts.range_hi, 64);
  EXPECT_EQ(facts.known_divisor, 16);

  loom_value_fact_table_clear_scope(&table_);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  EXPECT_NE(loom_value_fact_table_lookup(&table_, first_result).range_lo, 16);
  IREE_ASSERT_OK(loom_value_fact_table_define(&inputs, first_result,
                                              loom_value_facts_exact_i64(48)));
  IREE_ASSERT_OK(loom_value_fact_table_seed_values(
      &table_, {&inputs, &first_result, 1}, module_));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  facts = loom_value_fact_table_lookup(&table_, first_result);
  EXPECT_TRUE(loom_value_facts_is_exact(facts));
  EXPECT_EQ(facts.range_lo, 48);
}

TEST_F(FactTableComputeTest, BooleanBranchTruthIsRetainedByRegion) {
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_test_branch_build(
      &builder_, inputs_[0], /*result_types=*/nullptr, /*result_count=*/0,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN,
      &branch));
  loom_region_branch_t branch_interface =
      loom_region_branch_cast(module_, branch);
  for (uint8_t i = 0; i < branch->region_count; ++i) {
    loom_region_t* region =
        loom_region_branch_region(module_, branch_interface, i);
    loom_builder_ip_t saved =
        loom_builder_enter_region(&builder_, branch, region);
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, /*values=*/nullptr,
                                         /*values_count=*/0,
                                         LOOM_LOCATION_UNKNOWN, &yield));
    loom_builder_restore(&builder_, saved);
  }

  EXPECT_FALSE(table_.has_boolean_branch_regions);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, branch));
  EXPECT_TRUE(table_.has_boolean_branch_regions);
  EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                &table_, loom_test_branch_then_region(branch)),
            LOOM_REGION_BRANCH_TRUTH_TRUE);
  EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                &table_, loom_test_branch_else_region(branch)),
            LOOM_REGION_BRANCH_TRUTH_FALSE);

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_FALSE(table_.has_boolean_branch_regions);
  EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                &table_, loom_test_branch_then_region(branch)),
            LOOM_REGION_BRANCH_TRUTH_UNKNOWN);
}

TEST_F(FactTableComputeTest, SelectorBranchRegionsRemainTruthUnknown) {
  int64_t case_keys[] = {0, 1};
  loom_op_t* table_op = nullptr;
  IREE_ASSERT_OK(loom_test_region_table_build(
      &builder_, inputs_[0], case_keys, IREE_ARRAYSIZE(case_keys),
      LOOM_LOCATION_UNKNOWN, &table_op));
  loom_region_branch_t branch = loom_region_branch_cast(module_, table_op);
  for (uint8_t i = 0; i < table_op->region_count; ++i) {
    loom_region_t* region = loom_region_branch_region(module_, branch, i);
    loom_builder_ip_t saved =
        loom_builder_enter_region(&builder_, table_op, region);
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, /*values=*/nullptr,
                                         /*values_count=*/0,
                                         LOOM_LOCATION_UNKNOWN, &yield));
    loom_builder_restore(&builder_, saved);
  }

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, table_op));
  EXPECT_FALSE(table_.has_boolean_branch_regions);
  for (uint8_t i = 0; i < table_op->region_count; ++i) {
    EXPECT_EQ(loom_value_fact_table_lookup_region_branch_truth(
                  &table_, loom_region_branch_region(module_, branch, i)),
              LOOM_REGION_BRANCH_TRUTH_UNKNOWN);
  }
}

TEST_F(FactTableComputeTest, ExactDynamicRelationsAreRetainedLazily) {
  EXPECT_EQ(table_.exact_relations.ops, nullptr);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_exact_i64(5)));
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_LT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {inputs_[0], inputs_[1]},
  };
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1, &predicate,
                                         1, &type_, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(PendingExactRelations(table_), std::vector<loom_op_t*>({assume}));

  // A stable incremental refresh performs no candidate work and does not
  // duplicate the pending observation.
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(PendingExactRelations(table_), std::vector<loom_op_t*>({assume}));

  loom_value_fact_table_clear_pending_exact_relations(&table_);
  EXPECT_TRUE(PendingExactRelations(table_).empty());
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.args[1] = 10;
  IREE_ASSERT_OK(loom_index_assume_set_predicates(
      module_, assume, loom_attr_predicate_list(&predicate, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             assume, &changed));
  EXPECT_TRUE(PendingExactRelations(table_).empty());

  predicate.kind = LOOM_PREDICATE_POWER_OF_TWO;
  predicate.arg_count = 1;
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.args[0] = inputs_[1];
  loom_op_t* unary_assume = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1, &predicate,
                                         1, &type_, 1, LOOM_LOCATION_UNKNOWN,
                                         &unary_assume));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
      &table_, module_, unary_assume, &changed));
  EXPECT_EQ(PendingExactRelations(table_),
            std::vector<loom_op_t*>({unary_assume}));

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.exact_relations.ops, nullptr);
}

TEST_F(FactTableComputeTest,
       ExactDynamicRelationsAreRetainedWithoutChangeReporting) {
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_exact_i64(5)));
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_LT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {inputs_[0], inputs_[1]},
  };
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1, &predicate,
                                         1, &type_, 1, LOOM_LOCATION_UNKNOWN,
                                         &assume));

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));

  EXPECT_EQ(PendingExactRelations(table_), std::vector<loom_op_t*>({assume}));

  // Consuming a relation removes it from the pending journal. Revisiting the
  // same equation, directly or through a region refresh, must not publish it
  // again unless its facts change.
  loom_value_fact_table_clear_pending_exact_relations(&table_);
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, assume));
  EXPECT_TRUE(PendingExactRelations(table_).empty());

  loom_value_fact_table_clear_pending_exact_relations(&table_);
  IREE_ASSERT_OK(loom_value_fact_table_compute_region_tree(
      &table_, module_, loom_module_block(module_)->parent_region,
      /*parent_op=*/nullptr));
  EXPECT_TRUE(PendingExactRelations(table_).empty());
}

TEST_F(FactTableComputeTest, ExactRelationRetentionGrowsWithCandidates) {
  constexpr iree_host_size_t kAssumeCount = 64;
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[0],
                                              loom_value_facts_exact_i64(5)));
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_LT,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      .reserved = {},
      .args = {inputs_[0], inputs_[1]},
  };
  std::vector<loom_op_t*> assumes;
  assumes.reserve(kAssumeCount);
  for (iree_host_size_t i = 0; i < kAssumeCount; ++i) {
    loom_op_t* assume = nullptr;
    IREE_ASSERT_OK(loom_index_assume_build(&builder_, &inputs_[0], 1,
                                           &predicate, 1, &type_, 1,
                                           LOOM_LOCATION_UNKNOWN, &assume));
    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(
        &table_, module_, assume, &changed));
    EXPECT_TRUE(changed);
    assumes.push_back(assume);
  }
  EXPECT_EQ(PendingExactRelations(table_), assumes);
}

TEST_F(FactTableComputeTest, UnknownFloatResultsRetainTheirTypeDomain) {
  const loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_value_id_t input = DefineValue(f32_type);
  loom_op_t* negation = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&builder_, input, f32_type,
                                     LOOM_LOCATION_UNKNOWN, &negation));

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, negation));

  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(&table_, loom_test_neg_result(negation));
  EXPECT_TRUE(loom_value_facts_is_float(facts));
  EXPECT_FALSE(loom_value_facts_is_exact(facts));
  EXPECT_EQ(facts.range_lo, INT64_MIN);
  EXPECT_EQ(facts.range_hi, INT64_MAX);
}

TEST_F(FactTableComputeTest, IdentityCloneUndefineAndScopeReuse) {
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(BuildIdentity(inputs_[0], &alias));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, alias));
  const loom_value_id_t result = loom_op_const_results(alias)[0];
  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &clone, {&table_, &result, 1}, module_));
  EXPECT_EQ(loom_value_fact_table_query_identity(&clone, result), inputs_[0]);
  EXPECT_TRUE(loom_value_fact_table_values_equal(&clone, result, inputs_[0]));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&clone, inputs_[0]));
  EXPECT_FALSE(loom_value_fact_table_has_entry(&clone, inputs_[1]));
  loom_value_fact_table_undefine(&table_, result);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, result), result);
  EXPECT_FALSE(loom_value_fact_table_values_equal(&table_, result, inputs_[0]));
  bool changed = false;
  IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                             alias, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, result), inputs_[0]);

  const auto* storage = table_.identities.entries;
  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(table_.identities.entries, storage);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, result), result);
  EXPECT_FALSE(loom_value_fact_table_values_equal(&table_, result, inputs_[0]));
  EXPECT_TRUE(loom_value_fact_table_values_equal(&clone, result, inputs_[0]));
  EXPECT_EQ(loom_value_fact_table_query_identity(&clone, result), inputs_[0]);
  IREE_ASSERT_OK(loom_op_set_operand(module_, alias, 0, inputs_[1]));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, alias));
  EXPECT_EQ(table_.identities.entries, storage);
  EXPECT_EQ(loom_value_fact_table_query_identity(&table_, result), inputs_[1]);
  EXPECT_TRUE(loom_value_fact_table_values_equal(&table_, result, inputs_[1]));
  EXPECT_FALSE(loom_value_fact_table_values_equal(&table_, result, inputs_[0]));
}

TEST_F(FactTableComputeTest, NumericEqualityDoesNotCreateSSAIdentity) {
  loom_op_t* first = nullptr;
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(7), type_,
                                           LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(7), type_,
                                           LOOM_LOCATION_UNKNOWN, &second));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, first));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, second));
  const loom_value_id_t first_result = loom_op_const_results(first)[0];
  const loom_value_id_t second_result = loom_op_const_results(second)[0];
  EXPECT_NE(loom_value_fact_table_query_identity(&table_, first_result),
            loom_value_fact_table_query_identity(&table_, second_result));
  EXPECT_TRUE(
      loom_value_fact_table_values_equal(&table_, first_result, second_result));
  EXPECT_EQ(table_.identities.entries, nullptr);
}

TEST_F(FactTableComputeTest, DependentResultsPublishOnlyFinalChanges) {
  const loom_type_t tensor_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  const loom_value_id_t input = DefineValue(tensor_type);
  for (uint16_t extent_index = 0; extent_index < 2; ++extent_index) {
    SCOPED_TRACE(extent_index);
    const uint16_t tensor_index = 1 - extent_index;
    loom_type_t result_types[] = {type_, type_, type_};
    result_types[tensor_index] = tensor_type;
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_deflate_build(
        &builder_, input, result_types, IREE_ARRAYSIZE(result_types),
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        LOOM_LOCATION_UNKNOWN, &op));
    const loom_value_id_t* results = loom_op_const_results(op);
    bool changed = false;
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(loom_value_facts_is_unknown(
        loom_value_fact_table_lookup(&table_, results[extent_index])));

    // A new type use constrains a previously published unknown tuple.
    IREE_ASSERT_OK(loom_module_set_value_type(
        module_, results[tensor_index],
        loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                            loom_dim_pack_dynamic(results[extent_index]), 0)));

    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(loom_value_facts_is_non_negative(
        loom_value_fact_table_lookup(&table_, results[extent_index])));
    EXPECT_TRUE(loom_value_fact_table_has_entry(&table_, results[2]));
    EXPECT_TRUE(loom_value_facts_is_unknown(
        loom_value_fact_table_lookup(&table_, results[2])));

    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_FALSE(changed);
    EXPECT_TRUE(loom_value_facts_is_non_negative(
        loom_value_fact_table_lookup(&table_, results[extent_index])));

    // Type edits update the canonical use index before producer inference.
    // Removing the shaped use withdraws its domain without a separate cache.
    IREE_ASSERT_OK(loom_module_set_value_type(module_, results[tensor_index],
                                              tensor_type));
    IREE_ASSERT_OK(loom_value_fact_table_compute_op_and_report(&table_, module_,
                                                               op, &changed));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(loom_value_facts_is_unknown(
        loom_value_fact_table_lookup(&table_, results[extent_index])));
  }
}

class FactEntryInputTest : public FactTableComputeTest,
                           public ::testing::WithParamInterface<bool> {};

TEST_F(FactTableComputeTest, SingleLaneTargetMakesScalarArgumentsUniform) {
  std::vector<loom_type_t> arg_types;
  for (loom_scalar_type_t scalar_type = LOOM_SCALAR_TYPE_NONE + 1;
       scalar_type < LOOM_SCALAR_TYPE_COUNT_; ++scalar_type) {
    arg_types.push_back(loom_type_scalar(scalar_type));
  }
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("uniform"), &name));
  uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  loom_op_t* definition = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder_, 0, 0, 0, {0, symbol}, arg_types.data(), arg_types.size(),
      nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &definition));
  const auto function = loom_func_like_cast(module_, definition);
  loom_region_t* body = loom_func_like_body(function);
  loom_builder_enter_region(&builder_, definition, body);
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  const loom_value_id_t index_argument = loom_region_entry_arg_id(body, 0);
  loom_value_fact_table_t inputs;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&inputs, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &inputs, index_argument, loom_value_facts_make(-8, 64, 4)));

  for (uint32_t subgroup_size : {0u, 1u, 32u}) {
    SCOPED_TRACE(subgroup_size);
    loom_value_fact_table_clear_scope(&table_);
    IREE_ASSERT_OK(loom_value_fact_table_seed_values(
        &table_, {&inputs, &index_argument, 1}, module_));
    loom_target_facts_t target = {};
    target.storage.snapshot.index_bitwidth = 32;
    target.storage.snapshot.subgroup_size = subgroup_size;
    table_.context.target_facts = &target;
    IREE_ASSERT_OK(loom_value_fact_table_compute(&table_, module_, function));
    const loom_value_facts_t index_facts =
        loom_value_fact_table_lookup(&table_, index_argument);
    EXPECT_EQ(index_facts.range_lo, -8);
    EXPECT_EQ(index_facts.range_hi, 64);
    EXPECT_EQ(index_facts.known_divisor, 4);
    const loom_block_t* entry_block = loom_region_const_entry_block(body);
    for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
      SCOPED_TRACE(i);
      EXPECT_EQ(
          loom_value_facts_is_subgroup_uniform(loom_value_fact_table_lookup(
              &table_, loom_region_entry_arg_id(body, i))),
          subgroup_size == 1);
    }
  }
}

TEST_P(FactEntryInputTest, RefreshWithdrawsExtentConstraints) {
  const loom_type_t scalar_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t static_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(1), 0);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("extent"), &name));
  uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  loom_op_t* definition = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder_, 0, 0, 0, {0, symbol}, &scalar_type, 1, nullptr, 0, nullptr, 0,
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &definition));
  const auto function = loom_func_like_cast(module_, definition);
  loom_region_t* body = loom_func_like_body(function);
  body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_builder_enter_region(&builder_, definition, body);
  if (GetParam()) {
    const loom_value_id_t input = loom_region_entry_arg_id(body, 0);
    loom_op_t* nested = nullptr;
    IREE_ASSERT_OK(loom_test_block_args_build(&builder_, &input, 1,
                                              LOOM_LOCATION_UNKNOWN, &nested));
    loom_op_t* terminator = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &terminator));
    body = loom_test_block_args_body(nested);
    loom_builder_enter_region(&builder_, nested, body);
  }
  const loom_value_id_t extent = loom_region_entry_arg_id(body, 0);
  const loom_type_t dynamic_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(extent), 0);
  loom_op_t* zero = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(0), scalar_type, LOOM_LOCATION_UNKNOWN, &zero));
  loom_op_t* shaped = nullptr;
  IREE_ASSERT_OK(
      loom_test_attrs_build(&builder_, 0, loom_test_constant_result(zero), {},
                            static_type, LOOM_LOCATION_UNKNOWN, &shaped));
  const loom_value_id_t vector = loom_test_attrs_result(shaped);
  loom_op_t* comparison = nullptr;
  IREE_ASSERT_OK(loom_scalar_cmpi_build(
      &builder_, LOOM_SCALAR_CMPI_PREDICATE_SGE, extent,
      loom_test_constant_result(zero), LOOM_LOCATION_UNKNOWN, &comparison));
  const loom_value_id_t result = loom_scalar_cmpi_result(comparison);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &result, 1, LOOM_LOCATION_UNKNOWN, &use));
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  loom_value_fact_table_t inputs;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&inputs, &arena_, 0));
  const auto input_facts = loom_value_facts_make(-8, 64, 4);
  IREE_ASSERT_OK(loom_value_fact_table_define(&inputs, extent, input_facts));

  for (bool seeded : {false, true}) {
    SCOPED_TRACE(seeded);
    loom_value_fact_table_clear_scope(&table_);
    if (seeded) {
      IREE_ASSERT_OK(loom_value_fact_table_seed_values(
          &table_, {&inputs, &extent, 1}, module_));
    }
    // Exercise both introduction and withdrawal twice without rebuilding the
    // scope. Derived bounds must never become the next solve's input bounds.
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
      IREE_ASSERT_OK(loom_module_set_value_type(module_, vector, dynamic_type));
      IREE_ASSERT_OK(loom_value_fact_table_compute(&table_, module_, function));
      int64_t constant = 0;
      EXPECT_TRUE(loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(&table_, result), &constant));
      EXPECT_EQ(constant, 1);

      IREE_ASSERT_OK(loom_module_set_value_type(module_, vector, static_type));
      IREE_ASSERT_OK(loom_value_fact_table_compute(&table_, module_, function));
      const auto facts = loom_value_fact_table_lookup(&table_, extent);
      EXPECT_EQ(facts.range_lo, seeded ? -8 : INT32_MIN);
      EXPECT_EQ(facts.range_hi, seeded ? 64 : INT32_MAX);
      EXPECT_EQ(facts.known_divisor, seeded ? 4 : 1);
      EXPECT_FALSE(loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(&table_, result), &constant));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(IndependentEntries, FactEntryInputTest,
                         ::testing::Bool());

TEST_F(FactTableComputeTest, ConditionLoopRetainsAndReplacesBodyEntryFacts) {
  loom_op_t* bound_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, inputs_[1], inputs_[1], type_,
                                      LOOM_LOCATION_UNKNOWN, &bound_op));
  const loom_value_id_t bound = loom_index_add_result(bound_op);
  // Keep the dynamic upper bound in the signed carrier range so the true edge
  // of the unsigned comparison also proves that its lower operand is
  // nonnegative.
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, bound, loom_value_facts_make(0, 100, 1)));

  loom_op_t* loop = nullptr;
  const loom_type_t result_types[] = {
      type_,
      type_,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1),
  };
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder_, &inputs_[0], 1, /*iter_args_types=*/nullptr, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &loop));
  loom_region_t* before = loom_scf_while_before(loop);
  loom_region_t* body = loom_scf_while_after(loop);
  const loom_value_id_t before_argument = loom_region_entry_arg_id(before, 0);
  const loom_value_id_t body_arguments[] = {
      loom_region_entry_arg_id(body, 0),
      loom_region_entry_arg_id(body, 1),
  };
  const loom_value_id_t body_condition = loom_region_entry_arg_id(body, 2);

  const loom_value_id_t opaque_condition =
      DefineValue(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, loop, before);
  loom_op_t* compare = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_ULT,
                                      before_argument, bound,
                                      LOOM_LOCATION_UNKNOWN, &compare));
  loom_op_t* disjunction = nullptr;
  IREE_ASSERT_OK(loom_scalar_ori_build(
      &builder_, loom_index_cmp_result(compare), opaque_condition,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), LOOM_LOCATION_UNKNOWN,
      &disjunction));
  const loom_value_id_t forwarded[] = {
      before_argument,
      before_argument,
      loom_index_cmp_result(compare),
  };
  loom_op_t* condition = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(
      &builder_, loom_index_cmp_result(compare), forwarded,
      IREE_ARRAYSIZE(forwarded), LOOM_LOCATION_UNKNOWN, &condition));
  loom_builder_restore(&builder_, saved);

  saved = loom_builder_enter_region(&builder_, loop, body);
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, body_arguments, 1,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  const loom_condition_edge_projection_t* projection =
      loom_value_fact_table_lookup_region_condition_projection(&table_, body);
  ASSERT_NE(projection, nullptr);
  EXPECT_EQ(table_.condition_integer_projection_count, 1u);

  auto expect_relation = [&](loom_symbolic_integer_relation_t relation,
                             loom_value_id_t left,
                             loom_condition_integer_operand_t right,
                             bool expected_proven, bool expected_result) {
    const loom_condition_integer_relation_t queried = {
        .relation = relation,
        .left =
            {
                .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                .value_id = left,
                .constant = 0,
            },
        .right = right,
    };
    bool result = false;
    EXPECT_EQ(loom_condition_edge_projection_proves_integer_relation(
                  projection, &table_, &queried, &result),
              expected_proven);
    if (expected_proven) {
      EXPECT_EQ(result, expected_result);
    }
  };
  const loom_condition_integer_operand_t upper_bound = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = bound,
      .constant = 0,
  };
  const loom_condition_integer_operand_t zero = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
      .value_id = LOOM_VALUE_ID_INVALID,
      .constant = 0,
  };
  for (loom_value_id_t body_argument : body_arguments) {
    expect_relation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, body_argument,
                    upper_bound, /*expected_proven=*/true,
                    /*expected_result=*/true);
    expect_relation(LOOM_SYMBOLIC_INTEGER_RELATION_GE, body_argument, zero,
                    /*expected_proven=*/true, /*expected_result=*/true);
  }
  bool projected_condition = false;
  EXPECT_TRUE(loom_condition_edge_projection_query_boolean(
      projection, body_condition, &projected_condition));
  EXPECT_TRUE(projected_condition);

  // Replacing one duplicate payload must replace its projected relation. The
  // retained object is updated in place so consumers never observe stale facts
  // and repeated rewrites reuse its high-water storage.
  const loom_value_id_t* target_sources = projection->target_sources;
  const loom_condition_edge_mapping_t* mappings = projection->mappings;
  const loom_condition_integer_relation_t* relations =
      projection->source_derivation.integer_facts.integer_relations;
  const loom_condition_boolean_fact_t* boolean_facts =
      projection->source_derivation.boolean_facts;
  IREE_ASSERT_OK(loom_op_set_operand(module_, condition, 2, bound));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  EXPECT_EQ(
      loom_value_fact_table_lookup_region_condition_projection(&table_, body),
      projection);
  EXPECT_EQ(projection->target_sources, target_sources);
  EXPECT_EQ(projection->mappings, mappings);
  EXPECT_EQ(projection->source_derivation.integer_facts.integer_relations,
            relations);
  EXPECT_EQ(projection->source_derivation.boolean_facts, boolean_facts);
  EXPECT_EQ(table_.condition_integer_projection_count, 1u);
  expect_relation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, body_arguments[0],
                  upper_bound, /*expected_proven=*/true,
                  /*expected_result=*/true);
  expect_relation(LOOM_SYMBOLIC_INTEGER_RELATION_LT, body_arguments[1],
                  upper_bound, /*expected_proven=*/false,
                  /*expected_result=*/false);

  // A true disjunction does not prove the comparison, so replacing the exact
  // guard withdraws the table summary without scanning retained region
  // entries. Restoring it republishes the same projection object and slot.
  IREE_ASSERT_OK(loom_op_set_operand(module_, condition, 0,
                                     loom_scalar_ori_result(disjunction)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  EXPECT_TRUE(loom_condition_edge_projection_is_empty(projection));
  EXPECT_EQ(table_.condition_integer_projection_count, 0u);
  IREE_ASSERT_OK(loom_op_set_operand(module_, condition, 0,
                                     loom_index_cmp_result(compare)));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  EXPECT_FALSE(loom_condition_edge_projection_is_empty(projection));
  EXPECT_EQ(table_.condition_integer_projection_count, 1u);

  loom_value_fact_table_clear_scope(&table_);
  EXPECT_EQ(
      loom_value_fact_table_lookup_region_condition_projection(&table_, body),
      nullptr);
  EXPECT_EQ(table_.condition_integer_projection_count, 0u);
  EXPECT_EQ(table_.scratch.condition, nullptr);
}

TEST_F(FactTableComputeTest, ConditionLoopFactorsDuplicatePayloadRelations) {
  const loom_type_t result_types[] = {type_, type_, type_, type_};
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder_, inputs_, IREE_ARRAYSIZE(inputs_),
      /*iter_args_types=*/nullptr, result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN,
      &loop));
  loom_region_t* before = loom_scf_while_before(loop);
  loom_region_t* body = loom_scf_while_after(loop);
  const loom_value_id_t before_arguments[] = {
      loom_region_entry_arg_id(before, 0),
      loom_region_entry_arg_id(before, 1),
  };
  const loom_value_id_t body_arguments[] = {
      loom_region_entry_arg_id(body, 0),
      loom_region_entry_arg_id(body, 1),
      loom_region_entry_arg_id(body, 2),
      loom_region_entry_arg_id(body, 3),
  };

  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, loop, before);
  loom_op_t* compare = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_SLT,
                                      before_arguments[0], before_arguments[1],
                                      LOOM_LOCATION_UNKNOWN, &compare));
  const loom_value_id_t forwarded[] = {
      before_arguments[0],
      before_arguments[0],
      before_arguments[1],
      before_arguments[1],
  };
  loom_op_t* condition = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(
      &builder_, loom_index_cmp_result(compare), forwarded,
      IREE_ARRAYSIZE(forwarded), LOOM_LOCATION_UNKNOWN, &condition));
  loom_builder_restore(&builder_, saved);

  saved = loom_builder_enter_region(&builder_, loop, body);
  const loom_value_id_t yielded[] = {body_arguments[0], body_arguments[2]};
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, yielded,
                                      IREE_ARRAYSIZE(yielded),
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);

  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, loop));
  const loom_condition_edge_projection_t* projection =
      loom_value_fact_table_lookup_region_condition_projection(&table_, body);
  ASSERT_NE(projection, nullptr);
  EXPECT_EQ(projection->mapping_count, IREE_ARRAYSIZE(forwarded));
  // The source relation remains singular instead of expanding into every
  // pair of duplicate body arguments.
  EXPECT_EQ(projection->source_derivation.integer_facts.integer_relation_count,
            1u);
  for (iree_host_size_t left = 0; left < 2; ++left) {
    for (iree_host_size_t right = 2; right < 4; ++right) {
      const loom_condition_integer_relation_t query = {
          .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
          .left =
              {
                  .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                  .value_id = body_arguments[left],
              },
          .right =
              {
                  .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                  .value_id = body_arguments[right],
              },
      };
      bool result = false;
      EXPECT_TRUE(loom_condition_edge_projection_proves_integer_relation(
          projection, &table_, &query, &result));
      EXPECT_TRUE(result);
    }
  }
}

class FactTableCfgRecomputeTest
    : public FactTableComputeTest,
      public ::testing::WithParamInterface<iree_host_size_t> {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<FactTableCfgRecomputeTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        test->allocation_count_++ == test->failure_index_) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected snapshot allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    FactTableComputeTest::SetUp();
    iree_arena_block_pool_initialize(GetParam(), {this, Allocate},
                                     &scratch_pool_);
    iree_arena_initialize(&scratch_pool_, &scratch_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&scratch_pool_);
    FactTableComputeTest::TearDown();
  }

  // A captured initializer feeds a straight-line chain in a CFG cycle. Its
  // length crosses storage boundaries without changing the dataflow shape.
  void BuildCycle(iree_host_size_t count) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("cycle"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t callee = {0, symbol_id};
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, nullptr, 0,
                                        nullptr, 0, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &function_op));
    loom_func_like_t function = loom_func_like_cast(module_, function_op);
    loom_region_t* body = loom_func_like_body(function);
    body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    loom_builder_set_block(&builder_, loom_region_entry_block(body));
    builder_.ip.parent_op = function_op;
    IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(5), type_,
                                             LOOM_LOCATION_UNKNOWN, &initial_));
    loom_block_t* loop = nullptr;
    IREE_ASSERT_OK(loom_region_append_block(module_, body, &loop));
    loom_op_t* branch = nullptr;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, loop, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &branch));
    loom_builder_set_block(&builder_, loop);
    IREE_ASSERT_OK(loom_index_constant_build(&builder_, loom_attr_i64(1), type_,
                                             LOOM_LOCATION_UNKNOWN, &step_));
    loom_value_id_t previous = loom_index_constant_result(initial_);
    for (iree_host_size_t i = 0; i < count; ++i) {
      loom_op_t* add = nullptr;
      IREE_ASSERT_OK(loom_index_add_build(&builder_, previous,
                                          loom_index_constant_result(step_),
                                          type_, LOOM_LOCATION_UNKNOWN, &add));
      previous = loom_index_add_result(add);
      results_.push_back(previous);
    }
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, loop, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &branch));
    IREE_ASSERT_OK(loom_value_fact_table_compute(&table_, module_, function));
    structure_ = loom_value_fact_table_lookup_cfg_region(&table_, body);
    ASSERT_NE(structure_, nullptr);
    component_ =
        &structure_->control_flow.components
             .values[structure_->graph.blocks[loop->region_index].component];
    ASSERT_TRUE(component_->is_cycle);
  }

  void ChangeInitializer() {
    IREE_ASSERT_OK(loom_op_set_attr(module_, initial_, 0, loom_attr_i64(9)));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, initial_));
  }

  iree_status_t Recompute() {
    return loom_value_fact_table_recompute_cfg_component(
        &table_, module_, structure_, component_, &scratch_arena_,
        [](void* self, loom_value_id_t value) {
          static_cast<FactTableCfgRecomputeTest*>(self)->changed_.push_back(
              value);
          return iree_ok_status();
        },
        this);
  }

  void ExpectExact(loom_value_id_t value, int64_t expected) {
    ASSERT_TRUE(loom_value_fact_table_has_entry(&table_, value));
    const auto facts = loom_value_fact_table_lookup(&table_, value);
    EXPECT_EQ(facts.range_lo, expected);
    EXPECT_EQ(facts.range_hi, expected);
  }

  void CheckRecompute(iree_host_size_t result_count) {
    ASSERT_NO_FATAL_FAILURE(BuildCycle(result_count));
    for (iree_host_size_t i = 0; i < results_.size(); ++i) {
      ExpectExact(results_[i], 6 + i);
    }
    ASSERT_NO_FATAL_FAILURE(ChangeInitializer());
    IREE_ASSERT_OK(Recompute());
    EXPECT_EQ(changed_, results_);
    EXPECT_EQ(scratch_arena_.allocation_head, nullptr);
    ASSERT_GT(allocation_count_, 0u);
    const iree_host_size_t allocation_count = allocation_count_;
    const iree_host_size_t used_bytes = scratch_arena_.used_allocation_size;
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_initialize(&scratch_pool_, &scratch_arena_);
    changed_.clear();
    IREE_ASSERT_OK(Recompute());
    EXPECT_TRUE(changed_.empty());
    EXPECT_EQ(scratch_arena_.allocation_head, nullptr);
    EXPECT_EQ(allocation_count_, allocation_count);
    EXPECT_EQ(scratch_arena_.used_allocation_size, used_bytes);
    ExpectExact(loom_index_constant_result(initial_), 9);
    ExpectExact(loom_index_constant_result(step_), 1);
    for (iree_host_size_t i = 0; i < results_.size(); ++i) {
      ExpectExact(results_[i], 10 + i);
    }
  }

  void CheckAllocationFailure(iree_host_size_t failure_index) {
    // Both pool sizes require several allocations for this component.
    ASSERT_NO_FATAL_FAILURE(BuildCycle(4096));
    ASSERT_NO_FATAL_FAILURE(ChangeInitializer());
    failure_index_ = failure_index;
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Recompute());
    EXPECT_EQ(allocation_count_, failure_index + 1);
    EXPECT_TRUE(changed_.empty());
    EXPECT_EQ(scratch_arena_.allocation_head, nullptr);
    ExpectExact(loom_index_constant_result(initial_), 9);
    bool reached_unsaved = false;
    iree_host_size_t undefined_count = 0;
    for (iree_host_size_t i = 0; i < results_.size(); ++i) {
      if (loom_value_fact_table_has_entry(&table_, results_[i])) {
        reached_unsaved = true;
        ExpectExact(results_[i], 6 + i);
      } else {
        EXPECT_FALSE(reached_unsaved);
        ++undefined_count;
      }
    }
    EXPECT_LT(undefined_count, results_.size());
    if (failure_index == 0) {
      EXPECT_EQ(undefined_count, 0u);
      ExpectExact(loom_index_constant_result(step_), 1);
    } else {
      EXPECT_GT(undefined_count, 0u);
    }
  }

  // Pool backing only the component's temporary snapshot stream.
  iree_arena_block_pool_t scratch_pool_ = {};
  // Reinitialized between updates, matching the rewriter's scratch lifetime.
  iree_arena_allocator_t scratch_arena_ = {};
  // Number of attempted scratch backing allocations.
  iree_host_size_t allocation_count_ = 0;
  // Backing allocation to fail, or SIZE_MAX for unrestricted allocation.
  iree_host_size_t failure_index_ = SIZE_MAX;
  // Definition outside the component, edited to invalidate its dependents.
  loom_op_t* initial_ = nullptr;
  // Unchanged definition within the component.
  loom_op_t* step_ = nullptr;
  // Component results in definition order.
  std::vector<loom_value_id_t> results_;
  // Actual changes reported by the recomputation API.
  std::vector<loom_value_id_t> changed_;
  // Retained region analysis produced by the initial whole-function solve.
  const loom_value_fact_cfg_region_t* structure_ = nullptr;
  // The loop's component in the retained region analysis.
  const loom_scc_t* component_ = nullptr;
};

TEST_P(FactTableCfgRecomputeTest,
       ChangedResultsStreamThroughReusablePoolBlocks) {
  CheckRecompute(2050);
}

TEST_P(FactTableCfgRecomputeTest, ExactFullSnapshotTailReportsEveryChange) {
  // The unchanged step and these results fill sixteen 128-row chunks.
  CheckRecompute(2047);
}

TEST_P(FactTableCfgRecomputeTest, FirstSnapshotFailurePreservesAllFacts) {
  CheckAllocationFailure(0);
}

TEST_P(FactTableCfgRecomputeTest, LaterSnapshotFailurePreservesUnsavedFacts) {
  CheckAllocationFailure(1);
}

TEST_P(FactTableCfgRecomputeTest, StructuralEditsPublishWithUnchangedNumbers) {
  ASSERT_NO_FATAL_FAILURE(BuildCycle(2));
  loom_builder_set_before(&builder_,
                          structure_->graph.blocks[0].block->last_op);
  const auto i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_op_t* conditions[2] = {};
  for (unsigned i = 0; i < IREE_ARRAYSIZE(conditions); ++i) {
    IREE_ASSERT_OK(loom_scalar_constant_build(&builder_, loom_attr_i64(i), i1,
                                              LOOM_LOCATION_UNKNOWN,
                                              &conditions[i]));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, conditions[i]));
  }
  loom_builder_set_before(
      &builder_, structure_->graph.blocks[component_->nodes[0]].block->last_op);
  const auto initial = loom_index_constant_result(initial_);
  loom_op_t* select = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(
      &builder_, loom_scalar_constant_result(conditions[0]), initial, initial,
      type_, LOOM_LOCATION_UNKNOWN, &select));
  const auto selected = loom_scf_select_result(select);
  loom_op_t* add = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, selected,
                                      loom_index_constant_result(step_), type_,
                                      LOOM_LOCATION_UNKNOWN, &add));
  const auto sum = loom_index_add_result(add);
  IREE_ASSERT_OK(Recompute());
  ExpectSelector(sum, loom_scalar_constant_result(conditions[0]));

  changed_.clear();
  const auto replacement = loom_scalar_constant_result(conditions[1]);
  IREE_ASSERT_OK(loom_op_set_operand(module_, select, 0, replacement));
  IREE_ASSERT_OK(Recompute());
  EXPECT_EQ(changed_, std::vector<loom_value_id_t>({selected, sum}));
  ExpectSelector(selected, replacement);
  ExpectSelector(sum, replacement);
  ExpectExact(selected, 5);
  ExpectExact(sum, 6);
  changed_.clear();
  IREE_ASSERT_OK(Recompute());
  EXPECT_TRUE(changed_.empty());
}

INSTANTIATE_TEST_SUITE_P(PoolSizes, FactTableCfgRecomputeTest,
                         ::testing::Values(32 * 1024, 128 * 1024));

}  // namespace
}  // namespace loom
