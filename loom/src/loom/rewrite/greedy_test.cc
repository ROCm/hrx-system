// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/greedy.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/control_uniformity.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"
#include "loom/util/fact_cfg.h"

namespace loom {
namespace {

static const loom_target_fact_type_t kTestTargetFactType = {
    .name = IREE_SVL("test"),
    .storage_size = sizeof(loom_target_facts_t),
};

static void InitializeTestTargetFacts(loom_target_facts_t* out_facts) {
  *out_facts = {
      .fact_type = &kTestTargetFactType,
      .selector = 0,
      .explicit_fields = 0,
      .storage =
          {
              .snapshot = {.name = IREE_SVL("test")},
              .export_plan = {.name = IREE_SVL("test")},
              .config = {.name = IREE_SVL("test")},
              .bundle = {.name = IREE_SVL("test")},
          },
  };
  loom_target_bundle_storage_rebind(&out_facts->storage);
}

class GreedyRewriteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    vtables = loom_cfg_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_CFG, vtables, (uint16_t)vtable_count));
    vtables = loom_buffer_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_BUFFER, vtables, (uint16_t)vtable_count));
    vtables = loom_scf_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, vtables, (uint16_t)vtable_count));
    vtables = loom_index_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_INDEX, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    function_ = loom_func_like_cast(module_, func_op);
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder_);
    loom_builder_enter_region(&builder_, function_.op,
                              loom_func_like_body(function_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Shared backing for module and analysis arenas.
  iree_arena_block_pool_t block_pool_;
  // Registered operation semantics used by the fixture.
  loom_context_t context_;
  // Source module owned by the fixture.
  loom_module_t* module_ = nullptr;
  // Function whose body is transformed.
  loom_func_like_t function_;
  // Builder positioned in the function body.
  loom_builder_t builder_;
};

static iree_status_t pattern_one_to_two(const loom_rewrite_pattern_t*, void*,
                                        loom_op_t* op,
                                        loom_rewriter_t* rewriter,
                                        bool* out_changed) {
  *out_changed = false;
  if (!loom_test_constant_isa(op)) {
    return iree_ok_status();
  }
  int64_t value = loom_attr_as_i64(loom_test_constant_value(op));
  if (value != 1) {
    return iree_ok_status();
  }
  loom_value_id_t old_result = loom_test_constant_result(op);
  loom_type_t type = loom_module_value_type(rewriter->module, old_result);
  loom_op_t* replacement = NULL;
  IREE_RETURN_IF_ERROR(loom_test_constant_build(
      &rewriter->builder, loom_attr_i64(2), type, op->location, &replacement));
  loom_value_id_t new_result = loom_test_constant_result(replacement);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &new_result, 1));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t pattern_two_no_match(const loom_rewrite_pattern_t*, void*,
                                          loom_op_t*, loom_rewriter_t*,
                                          bool* out_changed) {
  *out_changed = false;
  return iree_ok_status();
}

static iree_status_t pattern_two_to_ten(const loom_rewrite_pattern_t*, void*,
                                        loom_op_t* op,
                                        loom_rewriter_t* rewriter,
                                        bool* out_changed) {
  *out_changed = false;
  if (!loom_test_constant_isa(op)) {
    return iree_ok_status();
  }
  int64_t value = loom_attr_as_i64(loom_test_constant_value(op));
  if (value != 2) {
    return iree_ok_status();
  }
  loom_value_id_t old_result = loom_test_constant_result(op);
  loom_type_t type = loom_module_value_type(rewriter->module, old_result);
  loom_op_t* replacement = NULL;
  IREE_RETURN_IF_ERROR(loom_test_constant_build(
      &rewriter->builder, loom_attr_i64(10), type, op->location, &replacement));
  loom_value_id_t new_result = loom_test_constant_result(replacement);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &new_result, 1));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t pattern_two_error(const loom_rewrite_pattern_t*, void*,
                                       loom_op_t* op, loom_rewriter_t*,
                                       bool* out_changed) {
  *out_changed = false;
  if (!loom_test_constant_isa(op)) {
    return iree_ok_status();
  }
  int64_t value = loom_attr_as_i64(loom_test_constant_value(op));
  if (value != 2) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INTERNAL, "pattern error on value 2");
}

static void initialize_pattern_registry(
    const loom_rewrite_pattern_t* patterns, uint16_t pattern_count,
    loom_rewrite_pattern_registry_storage_t* out_storage) {
  const loom_rewrite_pattern_provider_t provider = {
      .name = IREE_SVL("test"),
      .patterns = patterns,
      .pattern_count = pattern_count,
  };
  const loom_rewrite_pattern_provider_t* provider_values[] = {&provider};
  IREE_ASSERT_OK(loom_rewrite_pattern_registry_storage_initialize(
      loom_rewrite_pattern_provider_list_make(provider_values,
                                              IREE_ARRAYSIZE(provider_values)),
      iree_allocator_system(), out_storage));
}

enum class PreparationCase { kApply, kNoMatch, kAllocationFailure };

class PreparedRewriteTest
    : public GreedyRewriteTest,
      public ::testing::WithParamInterface<PreparationCase> {};

TEST_P(PreparedRewriteTest, CollectsBeforeMutatingAndPublishesBeforeObserving) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* source = nullptr;
  loom_op_t* increment = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(7), type,
                                          LOOM_LOCATION_UNKNOWN, &source));
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(GetParam() == PreparationCase::kNoMatch ? 1 : 0),
      type, LOOM_LOCATION_UNKNOWN, &increment));
  // Storage order differs from dominance order. Reverse-preorder application
  // replaces the producer first, so the consumer must read its updated operand.
  loom_region_t* body = loom_func_like_body(function_);
  body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_block_t* consumer = nullptr;
  loom_block_t* producer = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &consumer));
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &producer));
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, producer, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder_, producer);
  loom_op_t* first = nullptr;
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_,
                                      loom_test_constant_result(source),
                                      loom_test_constant_result(increment),
                                      type, LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_cfg_br_build(&builder_, consumer, nullptr, 0,
                                   LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(&builder_, consumer);
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, loom_test_addi_result(first),
                                      loom_test_constant_result(increment),
                                      type, LOOM_LOCATION_UNKNOWN, &second));
  const loom_value_id_t original_result = loom_test_addi_result(second);
  loom_op_t* use = nullptr;
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  struct State {
    // Injection controls the dependency's preparation failure, not the driver.
    PreparationCase test_case;
    // Number of fully collected actions.
    uint32_t prepared = 0;
    // Number of actions already applied.
    uint32_t applied = 0;
    // Number of mutation notifications from the shared driver.
    uint32_t notifications = 0;
    // Ordinary observer visits after publication.
    uint32_t observations = 0;
    struct Action {
      // Shared action header; no captured operand IDs.
      loom_rewrite_action_t action;
      // Test instrumentation shared across matching and application.
      State* state;
    };
    static iree_status_t Prepare(const loom_rewrite_pattern_t* pattern,
                                 const loom_module_t*,
                                 const loom_value_fact_table_t* facts,
                                 const loom_op_t* op,
                                 iree_arena_allocator_t* arena,
                                 loom_rewrite_action_t** out_action) {
      auto* state =
          const_cast<State*>(static_cast<const State*>(pattern->user_data));
      *out_action = nullptr;
      EXPECT_EQ(state->applied, 0u);
      EXPECT_EQ(
          arena->used_allocation_size,
          state->prepared * iree_host_align(sizeof(Action), iree_max_align_t));
      int64_t rhs = 0;
      if (!loom_value_facts_as_exact_i64(
              loom_value_fact_table_lookup(facts,
                                           loom_op_const_operands(op)[1]),
              &rhs) ||
          rhs != 0) {
        return iree_ok_status();
      }
      if (state->test_case == PreparationCase::kAllocationFailure &&
          state->prepared == 1) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected plan allocation failure");
      }
      Action* action = nullptr;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          arena, sizeof(Action), reinterpret_cast<void**>(&action)));
      action->action.apply = [](const loom_rewrite_action_t* action,
                                loom_rewriter_t* rewriter) {
        auto* state = reinterpret_cast<const Action*>(action)->state;
        EXPECT_EQ(state->prepared, 2u);
        EXPECT_EQ(state->observations, 0u);
#ifndef NDEBUG
        if (state->applied) {
          EXPECT_TRUE(rewriter->fact_table->observation.pending);
        }
#endif
        const loom_value_id_t replacement =
            loom_op_const_operands(action->root)[0];
        IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
            rewriter, action->root, &replacement, 1));
        ++state->applied;
        return iree_ok_status();
      };
      action->state = state;
      *out_action = &action->action;
      ++state->prepared;
      return iree_ok_status();
    }
  } state = {GetParam()};
  const loom_rewrite_pattern_t patterns[] = {{
      .root_kind = LOOM_OP_TEST_ADDI,
      .match_and_rewrite = pattern_two_no_match,
      .user_data = &state,
      .prepare = State::Prepare,
  }};
  loom_rewrite_pattern_registry_storage_t registry = {};
  initialize_pattern_registry(patterns, IREE_ARRAYSIZE(patterns), &registry);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module_, &arena, facts, &driver);
  const loom_greedy_rewrite_options_t options = {
      .prepared_patterns = &registry.registry,
  };
  const loom_greedy_rewrite_callbacks_t callbacks = {
      .user_data = &state,
      .rewrite_op =
          [](void* user_data, loom_greedy_rewrite_driver_t* driver,
             loom_op_t* op, loom_greedy_rewrite_result_t*, bool* changed) {
            auto* state = static_cast<State*>(user_data);
            *changed = false;
            if (!loom_test_use_isa(op)) {
              return iree_ok_status();
            }
            EXPECT_EQ(state->applied, state->prepared);
            int64_t value = 0;
            EXPECT_TRUE(loom_value_facts_as_exact_i64(
                loom_rewriter_value_facts(&driver->rewriter,
                                          loom_op_const_operands(op)[0]),
                &value));
            EXPECT_EQ(value,
                      state->test_case == PreparationCase::kNoMatch ? 9 : 7);
            ++state->observations;
            return iree_ok_status();
          },
      .changed =
          [](void* user_data, loom_greedy_rewrite_driver_t*) {
            ++static_cast<State*>(user_data)->notifications;
          },
  };
  loom_greedy_rewrite_result_t result;
  iree_status_t status = loom_greedy_rewrite_run_region(
      &driver, function_, loom_func_like_body(function_), function_.op,
      &options, &callbacks, &result);
  if (GetParam() == PreparationCase::kAllocationFailure) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(state.applied, 0u);
    EXPECT_EQ(state.observations, 0u);
    EXPECT_EQ(loom_op_const_operands(use)[0], original_result);
  } else {
    IREE_EXPECT_OK(status);
    EXPECT_EQ(result.changed, GetParam() == PreparationCase::kApply);
    EXPECT_EQ(result.ops_modified, state.applied);
    EXPECT_EQ(state.observations, 1u);
    if (GetParam() == PreparationCase::kApply) {
      EXPECT_EQ(loom_op_const_operands(use)[0],
                loom_test_constant_result(source));
    }
  }
  EXPECT_EQ(state.notifications, state.applied);
  loom_greedy_rewrite_driver_deinitialize(&driver);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
  loom_rewrite_pattern_registry_storage_deinitialize(&registry);
}

INSTANTIATE_TEST_SUITE_P(
    Preparation, PreparedRewriteTest,
    ::testing::Values(PreparationCase::kApply, PreparationCase::kNoMatch,
                      PreparationCase::kAllocationFailure));

TEST_F(GreedyRewriteTest, ChainedPatternsReachFixedPoint) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t original = loom_test_constant_result(const_op);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  const loom_rewrite_pattern_t patterns[] = {
      {LOOM_OP_TEST_CONSTANT, pattern_one_to_two, nullptr},
      {LOOM_OP_TEST_CONSTANT, pattern_two_no_match, nullptr},
      {LOOM_OP_TEST_CONSTANT, pattern_two_to_ten, nullptr},
  };
  loom_rewrite_pattern_registry_storage_t pattern_storage = {};
  initialize_pattern_registry(patterns, IREE_ARRAYSIZE(patterns),
                              &pattern_storage);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  IREE_ASSERT_OK(loom_greedy_rewrite(
      &arena, module_, function_,
      loom_rewrite_pattern_registry_storage_registry(&pattern_storage),
      /*pattern_context=*/nullptr, /*config=*/nullptr));
  iree_arena_deinitialize(&arena);
  loom_rewrite_pattern_registry_storage_deinitialize(&pattern_storage);

  loom_value_id_t final_result = loom_op_operands(use)[0];
  loom_value_t* value = loom_module_value(module_, final_result);
  loom_op_t* final_const = loom_value_def_op(value);
  ASSERT_NE(final_const, nullptr);
  EXPECT_EQ(loom_attr_as_i64(loom_test_constant_value(final_const)), 10);
}

TEST_F(GreedyRewriteTest, PatternErrorPropagates) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t original = loom_test_constant_result(const_op);

  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  const loom_rewrite_pattern_t patterns[] = {
      {LOOM_OP_TEST_CONSTANT, pattern_one_to_two, nullptr},
      {LOOM_OP_TEST_CONSTANT, pattern_two_error, nullptr},
  };
  loom_rewrite_pattern_registry_storage_t pattern_storage = {};
  initialize_pattern_registry(patterns, IREE_ARRAYSIZE(patterns),
                              &pattern_storage);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      loom_greedy_rewrite(
          &arena, module_, function_,
          loom_rewrite_pattern_registry_storage_registry(&pattern_storage),
          /*pattern_context=*/nullptr, /*config=*/nullptr));
  iree_arena_deinitialize(&arena);
  loom_rewrite_pattern_registry_storage_deinitialize(&pattern_storage);
}

TEST_F(GreedyRewriteTest, UnmatchedPatternsLeaveIrUntouched) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));

  const loom_rewrite_pattern_t patterns[] = {
      {LOOM_OP_TEST_CONSTANT, pattern_one_to_two, nullptr},
      {LOOM_OP_TEST_CONSTANT, pattern_two_to_ten, nullptr},
  };
  loom_rewrite_pattern_registry_storage_t pattern_storage = {};
  initialize_pattern_registry(patterns, IREE_ARRAYSIZE(patterns),
                              &pattern_storage);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  IREE_ASSERT_OK(loom_greedy_rewrite(
      &arena, module_, function_,
      loom_rewrite_pattern_registry_storage_registry(&pattern_storage),
      /*pattern_context=*/nullptr, /*config=*/nullptr));
  iree_arena_deinitialize(&arena);
  loom_rewrite_pattern_registry_storage_deinitialize(&pattern_storage);

  EXPECT_EQ(loom_attr_as_i64(loom_test_constant_value(const_op)), 42);
}

TEST_F(GreedyRewriteTest, ExplicitTargetFactsSetAnalysisScope) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  ASSERT_NE(const_op, nullptr);

  loom_target_facts_t target_facts;
  InitializeTestTargetFacts(&target_facts);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t fact_owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &fact_owner);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &fact_owner, module_,
      loom_pass_value_fact_scope_function_for_target(function_, &target_facts),
      &fact_table));
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module_, &arena, fact_table, &driver);

  const loom_greedy_rewrite_options_t options = {
      .max_iterations = {},
  };
  IREE_ASSERT_OK(loom_greedy_rewrite_run_region(
      &driver, function_, loom_func_like_body(function_), function_.op,
      &options, /*callbacks=*/NULL, /*out_result=*/NULL));

  const loom_value_fact_table_t* maintained_facts =
      loom_greedy_rewrite_driver_fact_table(&driver);
  ASSERT_NE(maintained_facts, nullptr);
  EXPECT_EQ(maintained_facts->context.target_facts, &target_facts);

  loom_greedy_rewrite_driver_deinitialize(&driver);
  loom_pass_value_fact_owner_deinitialize(&fact_owner);
  iree_arena_deinitialize(&arena);
}

// The driver owns publication at callback entry and successful return. Exercise
// a transitive dependency whose consumer is first in the LIFO rewrite order.
enum class ObservationBoundary { kReturn, kBeforeWorklist, kRewriteOperation };

class GreedyObservationTest
    : public GreedyRewriteTest,
      public ::testing::WithParamInterface<ObservationBoundary> {};

TEST_P(GreedyObservationTest, ObservesPublishedFacts) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* source = nullptr;
  loom_op_t* one = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), type,
                                          LOOM_LOCATION_UNKNOWN, &source));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), type,
                                          LOOM_LOCATION_UNKNOWN, &one));
  loom_op_t* first = nullptr;
  loom_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(
      &builder_, loom_test_constant_result(source),
      loom_test_constant_result(one), type, LOOM_LOCATION_UNKNOWN, &first));
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, loom_test_addi_result(first),
                                      loom_test_constant_result(one), type,
                                      LOOM_LOCATION_UNKNOWN, &second));
  const loom_value_id_t result_value = loom_test_addi_result(second);
  loom_op_t* use = nullptr;
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &result_value, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module_, &arena, facts, &driver);
  struct State {
    // Producer changed by preparation before any consumer is visited.
    loom_op_t* source;
    // Final consumer replaced by the observing callback, when enabled.
    loom_op_t* consumer;
    static iree_status_t Materialize(void* user_data,
                                     loom_greedy_rewrite_driver_t* driver,
                                     loom_op_t* op,
                                     loom_greedy_rewrite_result_t* result,
                                     bool* out_changed) {
      auto* state = static_cast<State*>(user_data);
      *out_changed = false;
      if (op != state->consumer || !state->consumer) {
        return iree_ok_status();
      }
      auto* rewriter = &driver->rewriter;
      const loom_value_id_t value = loom_test_addi_result(op);
      const loom_value_facts_t facts =
          loom_rewriter_value_facts(rewriter, value);
      loom_builder_set_before(&rewriter->builder, op);
      loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_rewriter_build_constant(
          rewriter, facts, loom_module_value_type(driver->module, value),
          op->location, &replacement));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
          rewriter, op, &replacement, 1));
      loom_greedy_rewrite_result_record_change(
          result, rewriter, LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
      state->consumer = nullptr;
      *out_changed = true;
      return iree_ok_status();
    }
  } state = {source, second};
  const loom_greedy_rewrite_options_t options = {
      .constant_materializer = {
          .supports =
              [](loom_value_facts_t facts, loom_type_t type) {
                int64_t value = 0;
                return loom_type_is_scalar(type) &&
                       loom_value_facts_as_exact_i64(facts, &value);
              },
          .materialize =
              [](loom_builder_t* builder, loom_value_facts_t facts,
                 loom_type_t type, loom_location_id_t location,
                 loom_value_id_t* out_value) {
                int64_t value = 0;
                EXPECT_TRUE(loom_value_facts_as_exact_i64(facts, &value));
                loom_op_t* constant = nullptr;
                IREE_RETURN_IF_ERROR(loom_test_constant_build(
                    builder, loom_attr_i64(value), type, location, &constant));
                *out_value = loom_test_constant_result(constant);
                return iree_ok_status();
              },
      },
  };
  loom_greedy_rewrite_callbacks_t callbacks = {
      .user_data = &state,
      .prepare_region =
          [](void* user_data, loom_greedy_rewrite_driver_t* driver,
             loom_func_like_t, loom_region_t*, loom_op_t*) {
            return loom_test_constant_rewrite_value(
                &driver->rewriter, static_cast<State*>(user_data)->source,
                loom_attr_i64(9));
          },
  };
  if (GetParam() == ObservationBoundary::kBeforeWorklist) {
    callbacks.before_worklist =
        [](void* user_data, loom_greedy_rewrite_driver_t* driver,
           loom_region_t*, loom_greedy_rewrite_result_t* result,
           bool* out_changed) {
          auto* state = static_cast<State*>(user_data);
          return State::Materialize(user_data, driver, state->consumer, result,
                                    out_changed);
        };
  } else if (GetParam() == ObservationBoundary::kRewriteOperation) {
    callbacks.rewrite_op = State::Materialize;
  }
  loom_greedy_rewrite_result_t result;
  IREE_ASSERT_OK(loom_greedy_rewrite_run_region(
      &driver, function_, loom_func_like_body(function_), function_.op,
      &options, &callbacks, &result));
  if (GetParam() == ObservationBoundary::kReturn) {
    int64_t actual = 0;
    ASSERT_TRUE(loom_value_facts_as_exact_i64(
        loom_value_fact_table_lookup(facts, result_value), &actual));
    EXPECT_EQ(actual, 11);
  } else {
    const loom_op_t* actual = loom_value_def_op(
        loom_module_value(module_, loom_op_const_operands(use)[0]));
    ASSERT_TRUE(loom_test_constant_isa(actual));
    EXPECT_EQ(loom_attr_as_i64(loom_test_constant_value(actual)), 11);
  }
  EXPECT_TRUE(result.facts_changed);
  loom_greedy_rewrite_driver_deinitialize(&driver);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

INSTANTIATE_TEST_SUITE_P(
    Publication, GreedyObservationTest,
    ::testing::Values(ObservationBoundary::kReturn,
                      ObservationBoundary::kBeforeWorklist,
                      ObservationBoundary::kRewriteOperation));

TEST_F(GreedyRewriteTest, FactIndependentRewritesPublishAsOneBatch) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* lhs = nullptr;
  loom_op_t* rhs = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), type,
                                          LOOM_LOCATION_UNKNOWN, &lhs));
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(3), type,
                                          LOOM_LOCATION_UNKNOWN, &rhs));
  loom_op_t* sum = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, loom_test_constant_result(lhs),
                                      loom_test_constant_result(rhs), type,
                                      LOOM_LOCATION_UNKNOWN, &sum));
  const loom_value_id_t sum_result = loom_test_addi_result(sum);
  loom_op_t* use = nullptr;
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &sum_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_pass_value_fact_owner_t owner;
  loom_pass_value_fact_owner_initialize(&block_pool_, &owner);
  loom_value_fact_table_t* facts = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &owner, module_, loom_pass_value_fact_scope_function(function_), &facts));
  loom_greedy_rewrite_driver_t driver;
  loom_greedy_rewrite_driver_initialize(module_, &arena, facts, &driver);
  struct State {
    loom_op_t* lhs;
    loom_op_t* rhs;
    loom_op_t* sum;
    uint32_t mutation_count = 0;
    uint32_t before_worklist_count = 0;
    uint32_t observed_count = 0;
  } state = {lhs, rhs, sum};
  loom_greedy_rewrite_callbacks_t callbacks = {
      .user_data = &state,
      .before_worklist =
          [](void* user_data, loom_greedy_rewrite_driver_t* driver,
             loom_region_t*, loom_greedy_rewrite_result_t*, bool* out_changed) {
            auto* state = static_cast<State*>(user_data);
            *out_changed = false;
            EXPECT_EQ(state->mutation_count, 2u);
            int64_t value = 0;
            EXPECT_TRUE(loom_value_facts_as_exact_i64(
                loom_rewriter_value_facts(&driver->rewriter,
                                          loom_test_addi_result(state->sum)),
                &value));
            EXPECT_EQ(value, 18);
            ++state->before_worklist_count;
            return iree_ok_status();
          },
      .rewrite_op_without_facts =
          [](void* user_data, loom_greedy_rewrite_driver_t* driver,
             loom_op_t* op, loom_greedy_rewrite_result_t* result,
             bool* out_changed) {
            auto* state = static_cast<State*>(user_data);
            *out_changed = false;
            loom_attribute_t replacement = {};
            if (op == state->lhs) {
              replacement = loom_attr_i64(7);
              state->lhs = nullptr;
            } else if (op == state->rhs) {
              replacement = loom_attr_i64(11);
              state->rhs = nullptr;
            } else {
              return iree_ok_status();
            }
            EXPECT_EQ(state->observed_count, 0u);
#ifndef NDEBUG
            if (state->mutation_count != 0) {
              EXPECT_TRUE(driver->fact_table->observation.pending);
            }
#endif
            IREE_RETURN_IF_ERROR(loom_test_constant_rewrite_value(
                &driver->rewriter, op, replacement));
            ++state->mutation_count;
            loom_greedy_rewrite_result_record_change(
                result, &driver->rewriter,
                LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
            *out_changed = true;
            return iree_ok_status();
          },
      .rewrite_op =
          [](void* user_data, loom_greedy_rewrite_driver_t* driver,
             loom_op_t* op, loom_greedy_rewrite_result_t*, bool* out_changed) {
            auto* state = static_cast<State*>(user_data);
            *out_changed = false;
            if (op != state->sum) {
              return iree_ok_status();
            }
            EXPECT_EQ(state->mutation_count, 2u);
            int64_t value = 0;
            EXPECT_TRUE(loom_value_facts_as_exact_i64(
                loom_rewriter_value_facts(&driver->rewriter,
                                          loom_test_addi_result(op)),
                &value));
            EXPECT_EQ(value, 18);
            ++state->observed_count;
            return iree_ok_status();
          },
  };
  loom_greedy_rewrite_result_t result;
  IREE_ASSERT_OK(loom_greedy_rewrite_run_region(
      &driver, function_, loom_func_like_body(function_), function_.op,
      /*options=*/nullptr, &callbacks, &result));
  EXPECT_EQ(state.mutation_count, 2u);
  EXPECT_GT(state.before_worklist_count, 0u);
  EXPECT_GT(state.observed_count, 0u);
  EXPECT_TRUE(result.changed);
  EXPECT_TRUE(result.facts_changed);

  loom_greedy_rewrite_driver_deinitialize(&driver);
  loom_pass_value_fact_owner_deinitialize(&owner);
  iree_arena_deinitialize(&arena);
}

TEST_F(GreedyRewriteTest, MutationTrackingDoesNotRequireAWorklist) {
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* original = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &original));
  loom_value_id_t original_value = loom_test_constant_result(original);
  loom_op_t* user = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &original_value, 1,
                                     LOOM_LOCATION_UNKNOWN, &user));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module_, &arena);
  loom_builder_set_before(&rewriter.builder, original);
  loom_op_t* replacement = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter.builder, loom_attr_i64(7),
                                          i32, LOOM_LOCATION_UNKNOWN,
                                          &replacement));
  loom_value_id_t replacement_value = loom_test_constant_result(replacement);
  IREE_ASSERT_OK(loom_rewriter_replace_all_uses_and_erase(
      &rewriter, original, &replacement_value, 1));
  EXPECT_EQ(loom_op_operands(user)[0], replacement_value);
  EXPECT_EQ(loom_module_value(module_, original_value)->use_count, 0u);
  EXPECT_EQ(loom_module_value(module_, replacement_value)->use_count, 1u);
  EXPECT_TRUE(original->flags & LOOM_OP_FLAG_DEAD);
  EXPECT_EQ(rewriter.created_op_count, 1u);
  EXPECT_EQ(rewriter.erased_op_count, 1u);
  EXPECT_EQ(rewriter.worklist_chunk, nullptr);
  EXPECT_FALSE(replacement->flags & LOOM_OP_FLAG_ON_WORKLIST);
  EXPECT_FALSE(user->flags & LOOM_OP_FLAG_ON_WORKLIST);

  // Seeding starts scheduling from the current live IR, including operations
  // created before tracking was enabled. Enabling again preserves that queue.
  IREE_ASSERT_OK(loom_rewriter_seed_function(&rewriter, function_));
  IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter));
  EXPECT_EQ(loom_rewriter_pop(&rewriter), user);
  EXPECT_EQ(loom_rewriter_pop(&rewriter), replacement);
  EXPECT_EQ(loom_rewriter_pop(&rewriter), nullptr);

  // Subsequent builder notifications use the same queue. Deinitialization must
  // release membership even if its consumer stops before draining the queue.
  loom_builder_set_before(&rewriter.builder, user);
  loom_op_t* subsequent = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&rewriter.builder, loom_attr_i64(9),
                                          i32, LOOM_LOCATION_UNKNOWN,
                                          &subsequent));
  EXPECT_TRUE(subsequent->flags & LOOM_OP_FLAG_ON_WORKLIST);
  loom_rewriter_deinitialize(&rewriter);
  EXPECT_FALSE(subsequent->flags & LOOM_OP_FLAG_ON_WORKLIST);
  iree_arena_deinitialize(&arena);
}

TEST_F(GreedyRewriteTest, NamePolicyCanDisableOptionalNames) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index_type, &source));
  loom_value_id_t target = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(module_, index_type, &target));

  loom_string_id_t source_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("head"), &source_name));
  IREE_ASSERT_OK(loom_module_set_value_name(module_, source, source_name));

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module_, &arena);
  rewriter.name_policy = 0;

  IREE_ASSERT_OK(loom_rewriter_copy_value_name(&rewriter, source, target));
  EXPECT_EQ(loom_module_value(module_, target)->name_id,
            LOOM_STRING_ID_INVALID);
  IREE_ASSERT_OK(loom_rewriter_try_set_derived_value_name(
      &rewriter, source, target, IREE_SV("bounded")));
  EXPECT_EQ(loom_module_value(module_, target)->name_id,
            LOOM_STRING_ID_INVALID);

  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
