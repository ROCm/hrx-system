// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_loop_replay.h"

#include <tuple>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/condition_edge_projection.h"
#include "loom/analysis/condition_facts.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/fact_snapshot.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactLoopReplayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("loop_replay"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&table_, &arena_, 4));
  }
  void CreateInputs(loom_type_t type, unsigned count, loom_value_id_t* values) {
    CreateInputs(std::vector<loom_type_t>(count, type), values);
  }
  void CreateInputs(const std::vector<loom_type_t>& types,
                    loom_value_id_t* values) {
    loom_string_id_t name;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("replay"), &name));
    uint16_t symbol;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    loom_op_t* function = nullptr;
    IREE_ASSERT_OK(loom_func_def_build(
        &builder_, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), {0, symbol},
        types.data(), types.size(), nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &function));
    auto* body = loom_func_like_body(loom_func_like_cast(module_, function));
    (void)loom_builder_enter_region(&builder_, function, body);
    for (unsigned i = 0; i < types.size(); ++i) {
      values[i] = loom_region_entry_arg_id(body, i);
    }
  }
  void FinishFunction() {
    loom_op_t* result = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&builder_, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &result));
  }
  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  // Backing pool for independent module, fact, and solve lifetimes.
  iree_arena_block_pool_t pool_;
  // Fact publications that outlive temporary nested solves.
  iree_arena_allocator_t arena_;
  // Production dialect registry for inference and structured control flow.
  loom_context_t context_;
  // IR whose function arguments and definitions carry every queried value.
  loom_module_t* module_ = nullptr;
  // Builder positioned in the current function or nested body.
  loom_builder_t builder_;
  // Shared fact state with root-owned temporary replay bindings.
  loom_value_fact_table_t table_;
};

TEST_F(FactLoopReplayTest, EqualPredicateSnapshotsHaveDifferentConsequences) {
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t values[3];
  CreateInputs(type, 3, values);
  const auto input = values[0], bound = values[1], one = values[2];
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, input, loom_value_facts_make(-100, 100, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, one,
                                              loom_value_facts_exact_i64(1)));
  loom_op_t* shifted = nullptr;
  IREE_ASSERT_OK(loom_scalar_addi_build(&builder_, 0, input, one, type,
                                        LOOM_LOCATION_UNKNOWN, &shifted));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, shifted));
  loom_op_t* comparison = nullptr;
  IREE_ASSERT_OK(loom_scalar_cmpi_build(&builder_,
                                        LOOM_SCALAR_CMPI_PREDICATE_SLT,
                                        loom_scalar_addi_result(shifted), bound,
                                        LOOM_LOCATION_UNKNOWN, &comparison));
  const auto predicate = loom_scalar_cmpi_result(comparison);
  FinishFunction();
  loom_condition_query_t query;
  loom_condition_query_initialize(module_, nullptr, &arena_, &query);
  loom_condition_derivation_t derivations[2];
  for (auto& derivation : derivations) {
    loom_condition_derivation_initialize(&arena_, &derivation);
  }
  loom_value_facts_t prior_facts;
  loom_value_fact_auxiliary_snapshot_t prior_auxiliary;
  for (int64_t limit : {5, 6}) {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, bound, loom_value_facts_exact_i64(limit)));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute_op(&table_, module_, comparison));
    auto facts = loom_value_fact_table_lookup(&table_, predicate);
    loom_value_fact_auxiliary_snapshot_t auxiliary;
    loom_value_fact_table_snapshot_auxiliary(&table_, predicate, &auxiliary);
    if (limit == 6) {
      EXPECT_TRUE(loom_value_facts_equal(prior_facts, facts));
      EXPECT_TRUE(loom_value_fact_auxiliary_snapshot_equal(&prior_auxiliary,
                                                           &auxiliary));
    }
    prior_facts = facts;
    prior_auxiliary = auxiliary;
    auto& derivation = derivations[limit - 5];
    IREE_ASSERT_OK(loom_condition_facts_query_complete(
        &query, &table_, predicate, true, &derivation));
    auto projected = loom_value_fact_table_lookup(&table_, input);
    EXPECT_TRUE(loom_condition_fact_set_apply_to_value_facts(
        &derivation.integer_facts, &table_, input, &projected));
    EXPECT_EQ(projected.range_hi, limit - 2);
    if (limit == 6) {
      auto stale_projection = loom_value_fact_table_lookup(&table_, input);
      EXPECT_TRUE(loom_condition_fact_set_apply_to_value_facts(
          &derivations[0].integer_facts, &table_, input, &stale_projection));
      EXPECT_EQ(stale_projection.range_hi, 3);
      EXPECT_NE(stale_projection.range_hi, projected.range_hi);
    }
  }
}

class FactLoopProjectionTest : public FactLoopReplayTest,
                               public ::testing::WithParamInterface<bool> {};

TEST_P(FactLoopProjectionTest, NestedSolveRefreshesPredicatePreimages) {
  const auto type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t values[4];
  CreateInputs(type, 4, values);
  const auto input = values[0], lower = values[1], upper = values[2],
             step = values[3];
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &table_, input, loom_value_facts_make(-100, 100, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, lower,
                                              loom_value_facts_exact_i64(5)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, upper,
                                              loom_value_facts_exact_i64(7)));
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, step,
                                              loom_value_facts_exact_i64(1)));
  const loom_value_id_t initial_values[] = {lower, input};
  loom_op_t* outer = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, 0, lower, upper, step, initial_values, 2, nullptr, nullptr, 0,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0, LOOM_LOCATION_UNKNOWN,
      &outer));
  auto* body = loom_scf_for_body(outer);
  const auto outer_saved = loom_builder_enter_region(&builder_, outer, body);
  const auto induction = loom_region_entry_arg_id(body, 1);
  const auto initial = loom_region_entry_arg_id(body, 2);
  loom_op_t* shifted = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, input, step, type,
                                      LOOM_LOCATION_UNKNOWN, &shifted));
  loom_op_t* predicate = nullptr;
  IREE_ASSERT_OK(loom_index_cmp_build(&builder_, LOOM_INDEX_CMP_PREDICATE_SLT,
                                      loom_index_add_result(shifted), induction,
                                      LOOM_LOCATION_UNKNOWN, &predicate));
  loom_op_t* inner = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(&builder_, &initial, 1, nullptr, &type, 1,
                                      nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                      &inner));
  auto* before = loom_scf_while_before(inner);
  auto saved = loom_builder_enter_region(&builder_, inner, before);
  const auto forwarded = loom_region_entry_arg_id(before, 0);
  loom_op_t* condition = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(
      &builder_, loom_index_cmp_result(predicate), &forwarded, 1,
      LOOM_LOCATION_UNKNOWN, &condition));
  loom_builder_restore(&builder_, saved);
  auto* after = loom_scf_while_after(inner);
  saved = loom_builder_enter_region(&builder_, inner, after);
  const auto unchanged = loom_region_entry_arg_id(after, 0);
  loom_op_t* nested = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, 0, lower, upper, step, nullptr, 0, nullptr, nullptr, 0,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0, LOOM_LOCATION_UNKNOWN,
      &nested));
  const auto nested_saved =
      loom_builder_enter_region(&builder_, nested, loom_scf_for_body(nested));
  loom_op_t* refined = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, input, step, type,
                                      LOOM_LOCATION_UNKNOWN, &refined));
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, nested_saved);
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, &unchanged, 1,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, saved);
  loom_op_t* increment = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, induction, step, type,
                                      LOOM_LOCATION_UNKNOWN, &increment));
  const loom_value_id_t next_values[] = {loom_index_add_result(increment),
                                         loom_op_const_results(inner)[0]};
  IREE_ASSERT_OK(loom_scf_yield_build(&builder_, next_values, 2,
                                      LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&builder_, outer_saved);
  FinishFunction();

  struct Observation {
    // Fact publications observed after the nested solve.
    loom_value_fact_table_t* table;
    // Inner loop whose conditioned body must refresh.
    const loom_op_t* inner;
    // Body carrying the retained true-edge projection.
    const loom_region_t* after;
    // Value behind the captured predicate's affine preimage.
    loom_value_id_t input;
    // Outer carried bound that changes while the predicate stays unknown.
    loom_value_id_t induction;
    // Ancestor equation at which the scope computes a retained refinement.
    const loom_op_t* shifted;
    // Numeric refinement cached before entering the innermost loop.
    loom_value_facts_t cached_bound;
    // Whether the solve consumes cached refinements.
    bool refine;
    // Equation whose operand consumes the ancestor's retained refinement.
    const loom_op_t* refined;
    // Number of completed inner summaries observed.
    unsigned count;
    // Whether the first exact bound was observed.
    bool saw_five;
    // Whether the next exact bound was observed.
    bool saw_six;
  } observation = {&table_, inner,      after,   input, induction, shifted,
                   {},      GetParam(), refined, 0,     false,     false};
  table_.context.refine_operands.user_data = &observation;
  if (GetParam()) {
    table_.context.refine_operands.fn =
        [](void* opaque, const loom_value_fact_table_t* table,
           const loom_op_t* op, loom_value_facts_t* operands) {
          auto* state = static_cast<Observation*>(opaque);
          if (op == state->shifted) {
            state->cached_bound =
                loom_value_fact_table_lookup(table, state->induction);
          } else if (op == state->refined) {
            operands[0] = state->cached_bound;
          }
        };
  }
  table_.summary_computed.user_data = &observation;
  table_.summary_computed.fn = [](void* opaque, const loom_op_t* op) {
    auto* state = static_cast<Observation*>(opaque);
    if (op != state->inner) {
      return;
    }
    const auto bound =
        loom_value_fact_table_lookup(state->table, state->induction);
    auto facts = loom_value_fact_table_lookup(state->table, state->input);
    auto* projection = loom_value_fact_table_lookup_region_condition_projection(
        state->table, state->after);
    ASSERT_NE(projection, nullptr);
    EXPECT_TRUE(loom_condition_edge_projection_apply_to_value_facts(
        projection, state->table, state->input, &facts));
    EXPECT_EQ(facts.range_hi, bound.range_hi - 2);
    if (state->refine) {
      const auto refined_facts = loom_value_fact_table_lookup(
          state->table, loom_index_add_result(state->refined));
      EXPECT_EQ(refined_facts.range_lo, bound.range_lo + 1);
      EXPECT_EQ(refined_facts.range_hi, bound.range_hi + 1);
    }
    ++state->count;
    state->saw_five |= loom_value_facts_is_exact(bound) && bound.range_lo == 5;
    state->saw_six |= loom_value_facts_is_exact(bound) && bound.range_lo == 6;
  };
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, outer));
  EXPECT_GE(observation.count, 3u);
  EXPECT_TRUE(observation.saw_five);
  EXPECT_TRUE(observation.saw_six);
}

INSTANTIATE_TEST_SUITE_P(RefinementScopes, FactLoopProjectionTest,
                         ::testing::Bool());

class FactLoopNestingTest
    : public FactLoopReplayTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    FactLoopReplayTest::SetUp();
    CreateInputs(type_, IREE_ARRAYSIZE(inputs_), inputs_);
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &table_, inputs_[0], loom_value_facts_make(1, 9, 1)));
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[1],
                                                loom_value_facts_exact_i64(0)));
    IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[2],
                                                loom_value_facts_exact_i64(1)));
  }

  loom_op_t* BuildNested(unsigned depth, loom_value_id_t initial) {
    const bool condition_loop = std::get<0>(GetParam());
    loom_op_t* loop = nullptr;
    if (condition_loop) {
      const loom_value_id_t values[] = {inputs_[1], initial};
      const loom_type_t types[] = {type_, type_};
      IREE_CHECK_OK(loom_scf_while_build(&builder_, values, 2, nullptr, types,
                                         2, nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                         &loop));
      auto* before = loom_scf_while_before(loop);
      const auto saved = loom_builder_enter_region(&builder_, loop, before);
      const loom_value_id_t forwarded[] = {loom_region_entry_arg_id(before, 0),
                                           loom_region_entry_arg_id(before, 1)};
      loom_op_t* comparison = nullptr;
      IREE_CHECK_OK(loom_index_cmp_build(
          &builder_, LOOM_INDEX_CMP_PREDICATE_SLT, forwarded[0], inputs_[0],
          LOOM_LOCATION_UNKNOWN, &comparison));
      loom_op_t* condition = nullptr;
      IREE_CHECK_OK(loom_scf_condition_build(
          &builder_, loom_index_cmp_result(comparison), forwarded, 2,
          LOOM_LOCATION_UNKNOWN, &condition));
      loom_builder_restore(&builder_, saved);
    } else {
      IREE_CHECK_OK(loom_scf_for_build(
          &builder_, 0, inputs_[1], inputs_[0], inputs_[2], &initial, 1,
          nullptr, nullptr, 0, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0,
          0, LOOM_LOCATION_UNKNOWN, &loop));
    }
    auto* body = loom_loop_like_body(loom_loop_like_cast(module_, loop));
    const auto saved = loom_builder_enter_region(&builder_, loop, body);
    const auto induction = loom_region_entry_arg_id(body, 0);
    const auto carried = loom_region_entry_arg_id(body, 1);
    loom_value_id_t increment = induction;
    if (depth > 1) {
      auto* nested = BuildNested(
          depth - 1, std::get<1>(GetParam()) ? carried : inputs_[1]);
      increment = loom_op_const_results(nested)[condition_loop ? 1 : 0];
    }
    loom_op_t* sum = nullptr;
    IREE_CHECK_OK(loom_index_add_build(&builder_, carried, increment, type_,
                                       LOOM_LOCATION_UNKNOWN, &sum));
    if (depth == 1) {
      leaf_ = loop;
    }
    loom_op_t* yield = nullptr;
    if (condition_loop) {
      loom_op_t* step = nullptr;
      IREE_CHECK_OK(loom_index_add_build(&builder_, induction, inputs_[2],
                                         type_, LOOM_LOCATION_UNKNOWN, &step));
      const loom_value_id_t values[] = {loom_index_add_result(step),
                                        loom_index_add_result(sum)};
      IREE_CHECK_OK(loom_scf_yield_build(&builder_, values, 2,
                                         LOOM_LOCATION_UNKNOWN, &yield));
    } else {
      const auto result = loom_index_add_result(sum);
      IREE_CHECK_OK(loom_scf_yield_build(&builder_, &result, 1,
                                         LOOM_LOCATION_UNKNOWN, &yield));
    }
    loom_builder_restore(&builder_, saved);
    return loop;
  }

  void CountLeafSummaries() {
    table_.summary_computed.user_data = this;
    table_.summary_computed.fn = [](void* user_data, const loom_op_t* op) {
      auto* self = static_cast<FactLoopNestingTest*>(user_data);
      if (op == self->leaf_) {
        ++self->leaf_summaries_;
      }
    };
  }

  // Scalar loop arithmetic shared by both structured control families.
  const loom_type_t type_ = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  // Function arguments supplying the runtime bound, zero, and unit step.
  loom_value_id_t inputs_[3];
  // Innermost loop whose visits count repeated subtree traversal.
  loom_op_t* leaf_ = nullptr;
  // Number of leaf summary notifications, including directly reused calls.
  unsigned leaf_summaries_ = 0;
};

TEST_P(FactLoopNestingTest, StableSubtreesDoNotMultiplySolverVisits) {
  auto* outer = BuildNested(8, inputs_[1]);
  FinishFunction();
  CountLeafSummaries();
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&table_, module_, outer));
  // Operand-equation reuse still visits every nested loop. Only whole-solve
  // reuse can avoid these summary entries as enclosing iterations repeat.
  EXPECT_GT(leaf_summaries_, 0u);
  EXPECT_LT(leaf_summaries_, 128u);
  EXPECT_EQ(table_.replay.loop, nullptr);
  EXPECT_EQ(table_.replay.reads, nullptr);
}

INSTANTIATE_TEST_SUITE_P(LoopKindsAndCaptures, FactLoopNestingTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

// Drives the same root/nested lifecycle as the solver while selecting exactly
// which publication a query consumes. No operation or inference is mocked.
class FactLoopDependencyTest : public FactLoopReplayTest {
 protected:
  virtual unsigned InputPrefixCount() const { return 0; }

  void SetUp() override {
    FactLoopReplayTest::SetUp();
    const auto index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    const auto scalar = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const auto vector = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(2), 0);
    std::vector<loom_type_t> types(InputPrefixCount(), scalar);
    types.insert(types.end(),
                 {index, index, index, scalar, scalar, vector, vector});
    std::vector<loom_value_id_t> arguments(types.size());
    CreateInputs(types, arguments.data());
    for (unsigned i = 0; i < IREE_ARRAYSIZE(inputs_); ++i) {
      inputs_[i] = arguments[InputPrefixCount() + i];
    }
    outer_ = BuildLoop(true);
    auto saved =
        loom_builder_enter_region(&builder_, outer_, loom_scf_for_body(outer_));
    for (unsigned i = 3; i < IREE_ARRAYSIZE(inputs_); ++i) {
      inputs_[i] = loom_region_entry_arg_id(loom_scf_for_body(outer_), i - 2);
    }
    inner_ = BuildLoop(false);
    auto inner_saved =
        loom_builder_enter_region(&builder_, inner_, loom_scf_for_body(inner_));
    leaf_ = BuildLoop(false);
    auto leaf_saved =
        loom_builder_enter_region(&builder_, leaf_, loom_scf_for_body(leaf_));
    Yield();
    loom_builder_restore(&builder_, leaf_saved);
    Yield();
    loom_builder_restore(&builder_, inner_saved);
    loom_op_t* outer_yield = nullptr;
    IREE_ASSERT_OK(loom_scf_yield_build(&builder_, inputs_ + 3, 4,
                                        LOOM_LOCATION_UNKNOWN, &outer_yield));
    loom_builder_restore(&builder_, saved);
    FinishFunction();
    iree_arena_initialize(&pool_, &solve_arena_);
    bool reused = true;
    IREE_ASSERT_OK(loom_value_fact_loop_replay_begin(
        &table_, module_, outer_, &solve_arena_, &outer_replay_, &reused));
    EXPECT_FALSE(reused);
    EXPECT_EQ(solve_arena_.used_allocation_size, 0u);
    table_.replaying = true;
  }

  void TearDown() override {
    IREE_EXPECT_OK(
        loom_value_fact_loop_replay_end(&table_, &outer_replay_, true));
    EXPECT_EQ(loom_value_fact_table_lookup_loop_replay(
                  &table_, loom_scf_for_body(inner_)),
              nullptr);
    EXPECT_EQ(table_.replay.loop, nullptr);
    EXPECT_EQ(table_.replay.reads, nullptr);
    iree_arena_deinitialize(&solve_arena_);
    FactLoopReplayTest::TearDown();
  }

  loom_op_t* BuildLoop(bool carried) {
    loom_op_t* loop = nullptr;
    loom_type_t types[4];
    for (unsigned i = 0; i < IREE_ARRAYSIZE(types); ++i) {
      types[i] = loom_module_value_type(module_, inputs_[i + 3]);
    }
    IREE_CHECK_OK(loom_scf_for_build(
        &builder_, 0, inputs_[0], inputs_[1], inputs_[2],
        carried ? inputs_ + 3 : nullptr, carried ? 4 : 0,
        carried ? types : nullptr, nullptr, 0, LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID, 0, 0, LOOM_LOCATION_UNKNOWN, &loop));
    return loop;
  }

  void Yield() {
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_scf_yield_build(&builder_, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &yield));
  }

  template <typename Query>
  void Visit(bool expected_reuse, Query query, loom_op_t* loop = nullptr) {
    loom_value_fact_loop_replay_t replay;
    bool reused = false;
    IREE_ASSERT_OK(loom_value_fact_loop_replay_begin(
        &table_, module_, loop ? loop : inner_, &solve_arena_, &replay,
        &reused));
    EXPECT_EQ(reused, expected_reuse);
    if (!reused) {
      query();
    }
    IREE_ASSERT_OK(loom_value_fact_loop_replay_end(&table_, &replay, true));
  }

  // Function bounds/step followed by the outer loop's carried scalar/vector
  // pairs.
  loom_value_id_t inputs_[7];
  // Outermost immutable solve whose return releases all reuse state.
  loom_op_t* outer_ = nullptr;
  // Nested solve with no carried results, still owning body facts.
  loom_op_t* inner_ = nullptr;
  // Child whose retained reads must also reach its enclosing solver.
  loom_op_t* leaf_ = nullptr;
  // Outer-solve working storage, separate from published fact payloads.
  iree_arena_allocator_t solve_arena_;
  // Active root lifetime, matching fact_loop.c's stack ownership.
  loom_value_fact_loop_replay_t outer_replay_;
};

class FactLoopLargePrefixTest : public FactLoopDependencyTest {
 protected:
  unsigned InputPrefixCount() const override { return 8192; }
};

TEST_F(FactLoopLargePrefixTest, ScratchExcludesUnrelatedDefinitions) {
  auto query = [&] { (void)loom_value_fact_table_lookup(&table_, inputs_[3]); };
  Visit(false, query);
  Visit(true, query);
  EXPECT_GT(module_->values.count, 8192u);
  EXPECT_LT(solve_arena_.used_allocation_size, 4096u);
}

TEST_F(FactLoopDependencyTest, ReusedChildForwardsItsDependencies) {
  auto leaf_query = [&] {
    (void)loom_value_fact_table_lookup(&table_, inputs_[3]);
  };
  auto inner_query = [&](bool child_reused) {
    Visit(child_reused, leaf_query, leaf_);
    (void)loom_value_fact_table_lookup(&table_, inputs_[4]);
  };
  Visit(false, [&] { inner_query(false); });
  Visit(true, [&] { FAIL() << "unchanged parent must reuse its solve"; });
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[4],
                                              loom_value_facts_exact_i64(2)));
  Visit(false, [&] { inner_query(true); });
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[3],
                                              loom_value_facts_exact_i64(1)));
  Visit(false, [&] { inner_query(false); });
}

TEST_F(FactLoopDependencyTest, AbsentAndPublishedUnknownAreDistinctInputs) {
  auto query = [&] {
    (void)loom_value_fact_table_has_entry(&table_, inputs_[3]);
  };
  Visit(false, query);
  Visit(true, query);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[3],
                                              loom_value_facts_unknown()));
  Visit(false, query);
  Visit(true, query);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table_, inputs_[3],
                                              loom_value_facts_exact_i64(7)));
  Visit(false, query);
}

TEST_F(FactLoopDependencyTest, WithdrawalExpiresBodyWithoutChangingInputs) {
  auto query = [&] { (void)loom_value_fact_table_lookup(&table_, inputs_[3]); };
  Visit(false, query);
  Visit(true, query);
  const auto argument = loom_region_entry_arg_id(loom_scf_for_body(inner_), 0);
  loom_value_fact_table_undefine(&table_, argument);
  Visit(false, query);
  Visit(true, query);
}

TEST_F(FactLoopDependencyTest, IdentityChangesWithoutNumericChange) {
  auto query = [&] {
    (void)loom_value_fact_table_query_identity(&table_, inputs_[3]);
  };
  Visit(false, query);
  IREE_ASSERT_OK(loom_value_fact_table_define_identity(&table_, inputs_[3],
                                                       inputs_[4], nullptr));
  Visit(false, query);
  Visit(true, query);
}

TEST_F(FactLoopDependencyTest, PartitionChangesWithoutNumericChange) {
  auto query = [&] {
    (void)loom_value_fact_table_query_vector_partition(&table_, inputs_[5]);
  };
  Visit(false, query);
  const loom_value_fact_vector_partition_component_t components[] = {
      {inputs_[3], 1}, {inputs_[4], 1}};
  IREE_ASSERT_OK(loom_value_fact_table_define_vector_partition(
      &table_, inputs_[5], {components, nullptr, 2, 2, 0, 0, 0}, nullptr));
  Visit(false, query);
  Visit(true, query);
}

class FactLoopOriginDependencyTest
    : public FactLoopDependencyTest,
      public ::testing::WithParamInterface<uint8_t> {
 protected:
  void Query() {
    switch (GetParam()) {
      case LOOM_VALUE_FACT_ORIGIN_UNIFORM_ELEMENT:
        (void)loom_value_fact_table_query_uniform_element_origin(
            &table_, module_, inputs_[5], nullptr);
        break;
      case LOOM_VALUE_FACT_ORIGIN_STATIC_LANE:
        (void)loom_value_fact_table_query_static_lane_origin(
            &table_, module_, inputs_[5], nullptr);
        break;
      case LOOM_VALUE_FACT_ORIGIN_EXACT_LANE:
        (void)loom_value_fact_table_query_exact_lane_origin(
            &table_, module_, inputs_[5], nullptr);
        break;
      case LOOM_VALUE_FACT_ORIGIN_EXACT_SCALAR:
        (void)loom_value_fact_table_query_exact_scalar_origin(
            &table_, module_, inputs_[3], nullptr);
        break;
      case LOOM_VALUE_FACT_ORIGIN_UNIFORM_SCALE:
        (void)loom_value_fact_table_query_uniform_scale_origin(
            &table_, module_, inputs_[5], nullptr);
        break;
      case LOOM_VALUE_FACT_ORIGIN_CONTEXTUAL_QUERY:
        (void)loom_value_fact_table_query_contextual_query_origin(
            &table_, module_, inputs_[3], nullptr);
        break;
      case LOOM_VALUE_FACT_ORIGIN_LAYOUT_STRIDES:
        (void)loom_value_fact_table_query_layout_strides(&table_, inputs_[5]);
        break;
    }
  }

  iree_status_t Publish() {
    switch (GetParam()) {
      case LOOM_VALUE_FACT_ORIGIN_UNIFORM_ELEMENT:
        return loom_value_fact_table_define_uniform_element_origin(
            &table_, inputs_[5], inputs_[3], inputs_[3]);
      case LOOM_VALUE_FACT_ORIGIN_STATIC_LANE:
        return loom_value_fact_table_define_static_lane_origin(
            &table_, inputs_[5], {inputs_[6], 0, 1});
      case LOOM_VALUE_FACT_ORIGIN_EXACT_LANE:
        return loom_value_fact_table_define_exact_lane_origin(
            &table_, inputs_[5], {inputs_[6], 0, 1});
      case LOOM_VALUE_FACT_ORIGIN_EXACT_SCALAR:
        return loom_value_fact_table_define_exact_scalar_origin(
            &table_, inputs_[3], {inputs_[4]});
      case LOOM_VALUE_FACT_ORIGIN_UNIFORM_SCALE:
        return loom_value_fact_table_define_uniform_scale_origin(
            &table_, inputs_[5], {inputs_[6], inputs_[3]});
      case LOOM_VALUE_FACT_ORIGIN_CONTEXTUAL_QUERY:
        return loom_value_fact_table_define_contextual_query_origin(
            &table_, inputs_[3],
            {
                .family_kind = LOOM_PARAMETERIZED_ATTR_KIND(3, 7),
                .key = loom_attr_i64(2),
            });
      case LOOM_VALUE_FACT_ORIGIN_LAYOUT_STRIDES: {
        const loom_value_id_t strides[] = {inputs_[2]};
        return loom_value_fact_table_define_layout_strides(&table_, inputs_[5],
                                                           {strides, 1});
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    }
  }
};

TEST_P(FactLoopOriginDependencyTest, AuxiliaryPublicationInvalidatesSolve) {
  auto query = [&] { Query(); };
  Visit(false, query);
  Visit(true, query);
  IREE_ASSERT_OK(Publish());
  Visit(false, query);
  Visit(true, query);
}

INSTANTIATE_TEST_SUITE_P(
    Families, FactLoopOriginDependencyTest,
    ::testing::Values(LOOM_VALUE_FACT_ORIGIN_UNIFORM_ELEMENT,
                      LOOM_VALUE_FACT_ORIGIN_STATIC_LANE,
                      LOOM_VALUE_FACT_ORIGIN_EXACT_LANE,
                      LOOM_VALUE_FACT_ORIGIN_EXACT_SCALAR,
                      LOOM_VALUE_FACT_ORIGIN_UNIFORM_SCALE,
                      LOOM_VALUE_FACT_ORIGIN_CONTEXTUAL_QUERY,
                      LOOM_VALUE_FACT_ORIGIN_LAYOUT_STRIDES));

}  // namespace
}  // namespace loom
