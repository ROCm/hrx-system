// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/scenario/values.h"

#include <array>

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/tooling/testbench/expectation.h"

namespace loom {
namespace {

TEST(ScenarioEntropyTest, StableRandomAccessAndNamedForks) {
  const loom_testbench_entropy_t root =
      loom_testbench_entropy_root(0x4c6f6f6dull);
  EXPECT_EQ(root.low, 0xaa5409db0d124e7dull);
  EXPECT_EQ(root.high, 0x5a04156236e3cc7bull);

  const loom_testbench_entropy_t input =
      loom_testbench_entropy_fork(root, IREE_SV("input"));
  EXPECT_EQ(input.low, 0x9af282807833e29aull);
  EXPECT_EQ(input.high, 0x9f9a7ff237537a11ull);
  EXPECT_EQ(loom_testbench_entropy_read(input, 0), 0x1a9d7a7cb2d83838ull);
  EXPECT_EQ(loom_testbench_entropy_read(input, 1), 0x9d679215b4c854e2ull);
  EXPECT_EQ(loom_testbench_entropy_read(input, 17), 0x2a1655b1581e659aull);

  // Reads and forks do not mutate their parent, so traversal order is inert.
  EXPECT_EQ(loom_testbench_entropy_read(input, 0), 0x1a9d7a7cb2d83838ull);
  const loom_testbench_entropy_t repeated =
      loom_testbench_entropy_fork(root, IREE_SV("input"));
  EXPECT_EQ(repeated.low, input.low);
  EXPECT_EQ(repeated.high, input.high);
  const loom_testbench_entropy_t other =
      loom_testbench_entropy_fork(root, IREE_SV("output"));
  EXPECT_TRUE(other.low != input.low || other.high != input.high);
}

class ScenarioValuesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(IREE_SV("scenario"), host_allocator_,
                                       host_allocator_, &device_allocator_));
  }

  void TearDown() override {
    iree_hal_allocator_release(device_allocator_);
    iree_arena_deinitialize(&plan_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_module_t* ParseModule() {
    static const char source[] = R"(
test.func @identity(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.scenario public @paired configure[2](%configuration: index, %configuration_entropy: check.entropy) {
  %configuration_stream = check.entropy.fork %configuration_entropy name("configuration") : check.entropy
  %configuration_word = check.entropy.read %configuration_stream[%configuration] : check.entropy -> i64
  check.trial[3](%trial: index, %entropy: check.entropy) {
    %input_stream = check.entropy.fork %entropy name("input") : check.entropy
    %seed = check.entropy.read %input_stream[0] : check.entropy -> i64
    %input = check.literal value(7) : i32
    %storage = check.generate.random.uniform seed(%seed) range(-16 to 16) : tensor<4xi32>
    %tail = check.tensor.view %storage offset(8) : tensor<4xi32> -> tensor<2xi32>
    check.compare<@identity>(%input) : (i32) -> [actual(%actual: i32), expected(%expected: i32)] {
      check.expect.equal actual(%actual) expected(%expected) : i32
      check.expect.bitwise actual(%storage) expected(%storage) : tensor<4xi32>
      check.expect.bitwise actual(%tail) expected(%tail) : tensor<2xi32>
    }
  }
  check.trial[2](%trial: index, %entropy: check.entropy) {
    %input = check.literal value(11) : i32
    check.invoke<@identity>(%input) : (i32) -> (i32)
  }
  check.return
}
)";
    loom_text_parse_options_t options = {.max_errors = 20};
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("scenario_values_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_testbench_module_plan_t PlanModule(loom_module_t* module) {
    loom_testbench_module_plan_t plan = {};
    IREE_EXPECT_OK(
        loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));
    return plan;
  }

  loom_testbench_value_materializer_options_t MaterializerOptions() {
    loom_testbench_value_materializer_options_t options = {};
    loom_testbench_value_materializer_options_initialize(&options);
    options.device_allocator = device_allocator_;
    return options;
  }

  static const loom_testbench_value_t* Lookup(
      const loom_testbench_value_table_t* table, loom_value_id_t value_id) {
    const loom_testbench_value_t* value = nullptr;
    IREE_EXPECT_OK(
        loom_testbench_value_table_lookup_borrow(table, value_id, &value));
    return value;
  }

  static void AssignI32(loom_testbench_value_table_t* table,
                        loom_value_id_t value_id, int32_t payload) {
    loom_testbench_value_t value = {};
    value.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
    value.scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
    value.scalar.storage.i32 = payload;
    IREE_ASSERT_OK(
        loom_testbench_value_table_assign_move(table, value_id, &value));
  }

  iree_allocator_t host_allocator_ = iree_allocator_system();
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

TEST_F(ScenarioValuesTest, MaterializesIndependentPairedAliasGraphs) {
  loom_module_t* module = ParseModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.scenario_count, 1u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[0];
  const loom_testbench_trial_plan_t& trial = scenario.trials[0];

  loom_testbench_scenario_configuration_values_t configuration = {};
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_initialize(
      module, &scenario, host_allocator_, &configuration));
  loom_testbench_value_materializer_options_t options = MaterializerOptions();
  const loom_testbench_entropy_t entropy_root =
      loom_testbench_entropy_root(0x123456789abcdef0ull);
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_materialize(
      &options, entropy_root, /*configuration_ordinal=*/1, &configuration));
  EXPECT_TRUE(iree_any_bit_set(
      configuration.flags, LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED));
  int64_t configuration_ordinal = -1;
  IREE_ASSERT_OK(loom_testbench_value_as_i64(
      Lookup(&configuration.values, scenario.configuration_ordinal_value_id),
      &configuration_ordinal));
  EXPECT_EQ(configuration_ordinal, 1);
  EXPECT_TRUE(loom_testbench_value_is_entropy(
      Lookup(&configuration.values, scenario.configuration_entropy_value_id)));

  loom_testbench_scenario_trial_values_t values = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_values_initialize(
      module, &scenario, /*trial_index=*/0,
      LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_AND_ORACLE,
      host_allocator_, &values));
  IREE_ASSERT_OK(loom_testbench_scenario_trial_values_materialize(
      &options, /*generator_executor=*/nullptr, &configuration,
      /*trial_ordinal=*/2, &values));
  EXPECT_TRUE(iree_all_bits_set(
      values.flags, LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED |
                        LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_HAS_ORACLE));
  EXPECT_EQ(values.identity.entropy_root.low, entropy_root.low);
  EXPECT_EQ(values.identity.entropy_root.high, entropy_root.high);
  EXPECT_EQ(values.identity.configuration_ordinal, 1u);
  EXPECT_EQ(values.identity.trial_index, 0u);
  EXPECT_EQ(values.identity.trial_ordinal, 2u);

  int64_t target_seed = 0;
  int64_t oracle_seed = 0;
  IREE_ASSERT_OK(loom_testbench_value_as_i64(
      Lookup(&values.target, trial.recipe_steps[1].value_source.value_id),
      &target_seed));
  IREE_ASSERT_OK(loom_testbench_value_as_i64(
      Lookup(&values.oracle, trial.recipe_steps[1].value_source.value_id),
      &oracle_seed));
  EXPECT_EQ(target_seed, oracle_seed);

  const loom_testbench_value_t* target_storage =
      Lookup(&values.target, trial.recipe_steps[3].value_source.value_id);
  const loom_testbench_value_t* target_tail =
      Lookup(&values.target, trial.recipe_steps[4].value_source.value_id);
  const loom_testbench_value_t* oracle_storage =
      Lookup(&values.oracle, trial.recipe_steps[3].value_source.value_id);
  const loom_testbench_value_t* oracle_tail =
      Lookup(&values.oracle, trial.recipe_steps[4].value_source.value_id);
  ASSERT_TRUE(loom_testbench_value_is_buffer(target_storage));
  ASSERT_TRUE(loom_testbench_value_is_buffer(target_tail));
  ASSERT_TRUE(loom_testbench_value_is_buffer(oracle_storage));
  ASSERT_TRUE(loom_testbench_value_is_buffer(oracle_tail));

  EXPECT_TRUE(target_storage->buffer_reference.is_traceable);
  EXPECT_TRUE(target_tail->buffer_reference.is_traceable);
  EXPECT_TRUE(oracle_storage->buffer_reference.is_traceable);
  EXPECT_TRUE(oracle_tail->buffer_reference.is_traceable);
  EXPECT_EQ(target_storage->buffer_reference.allocation_value_id,
            trial.recipe_steps[3].value_source.value_id);
  EXPECT_EQ(oracle_storage->buffer_reference.allocation_value_id,
            trial.recipe_steps[3].value_source.value_id);
  EXPECT_EQ(target_tail->buffer_reference.allocation_value_id,
            trial.recipe_steps[3].value_source.value_id);
  EXPECT_EQ(oracle_tail->buffer_reference.allocation_value_id,
            trial.recipe_steps[3].value_source.value_id);
  EXPECT_EQ(target_storage->buffer_reference.byte_offset, 0u);
  EXPECT_EQ(oracle_storage->buffer_reference.byte_offset, 0u);
  EXPECT_EQ(target_tail->buffer_reference.byte_offset, 8u);
  EXPECT_EQ(oracle_tail->buffer_reference.byte_offset, 8u);
  EXPECT_EQ(target_tail->buffer_reference.byte_length, 8u);
  EXPECT_EQ(oracle_tail->buffer_reference.byte_length, 8u);

  EXPECT_EQ(iree_hal_buffer_test_overlap(
                target_storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER,
                target_tail->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER),
            IREE_HAL_BUFFER_OVERLAP_PARTIAL);
  EXPECT_EQ(iree_hal_buffer_test_overlap(
                oracle_storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER,
                oracle_tail->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER),
            IREE_HAL_BUFFER_OVERLAP_PARTIAL);
  EXPECT_EQ(iree_hal_buffer_test_overlap(
                target_storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER,
                oracle_storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER),
            IREE_HAL_BUFFER_OVERLAP_DISJOINT);

  std::array<int32_t, 4> target_contents = {};
  std::array<int32_t, 4> oracle_contents = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_storage->buffer.buffer, 0,
                                          target_contents.data(),
                                          sizeof(target_contents)));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(oracle_storage->buffer.buffer, 0,
                                          oracle_contents.data(),
                                          sizeof(oracle_contents)));
  EXPECT_EQ(target_contents, oracle_contents);

  const loom_testbench_scenario_action_plan_t& action = trial.action;
  AssignI32(&values.target, action.target.result_value_ids[0], 7);
  AssignI32(&values.oracle, action.oracle.result_value_ids[0], 7);
  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      action.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_scenario_action_expectations(
      &action, &values.target, &values.oracle, nullptr, &report));
  EXPECT_EQ(report.passed_count, 3u);
  EXPECT_EQ(report.failure_count, 0u);

  const int32_t changed = target_contents[0] + 1;
  IREE_ASSERT_OK(iree_hal_buffer_map_write(target_storage->buffer.buffer, 0,
                                           &changed, sizeof(changed)));
  IREE_ASSERT_OK(loom_testbench_evaluate_scenario_action_expectations(
      &action, &values.target, &values.oracle, nullptr, &report));
  ASSERT_EQ(report.failure_count, 1u);
  EXPECT_EQ(report.failures[0].expectation, &action.expectations[1]);

  IREE_ASSERT_OK(loom_testbench_scenario_trial_values_materialize(
      &options, /*generator_executor=*/nullptr, &configuration,
      /*trial_ordinal=*/0, &values));
  int64_t other_seed = 0;
  IREE_ASSERT_OK(loom_testbench_value_as_i64(
      Lookup(&values.target, trial.recipe_steps[1].value_source.value_id),
      &other_seed));
  EXPECT_NE(other_seed, target_seed);
  IREE_ASSERT_OK(loom_testbench_scenario_trial_values_materialize(
      &options, /*generator_executor=*/nullptr, &configuration,
      /*trial_ordinal=*/2, &values));
  int64_t repeated_seed = 0;
  IREE_ASSERT_OK(loom_testbench_value_as_i64(
      Lookup(&values.target, trial.recipe_steps[1].value_source.value_id),
      &repeated_seed));
  EXPECT_EQ(repeated_seed, target_seed);

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_scenario_trial_values_deinitialize(&values);
  loom_testbench_scenario_configuration_values_deinitialize(&configuration);
  loom_module_free(module);
}

TEST_F(ScenarioValuesTest, InvokeMaterializesOnlyTargetValues) {
  loom_module_t* module = ParseModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[0];

  loom_testbench_value_materializer_options_t options = MaterializerOptions();
  loom_testbench_scenario_configuration_values_t configuration = {};
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_initialize(
      module, &scenario, host_allocator_, &configuration));
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_materialize(
      &options, loom_testbench_entropy_root(7), /*configuration_ordinal=*/0,
      &configuration));

  loom_testbench_scenario_trial_values_t values = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_values_initialize(
      module, &scenario, /*trial_index=*/1,
      LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_ONLY, host_allocator_,
      &values));
  EXPECT_FALSE(iree_any_bit_set(values.flags,
                                LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_HAS_ORACLE));
  EXPECT_EQ(values.oracle.slot_count, 0u);
  IREE_ASSERT_OK(loom_testbench_scenario_trial_values_materialize(
      &options, /*generator_executor=*/nullptr, &configuration,
      /*trial_ordinal=*/1, &values));
  EXPECT_TRUE(loom_testbench_value_table_contains(
      &values.target,
      values.trial_plan->recipe_steps[0].value_source.value_id));
  EXPECT_EQ(values.oracle.slot_count, 0u);

  loom_testbench_scenario_trial_values_deinitialize(&values);
  loom_testbench_scenario_configuration_values_deinitialize(&configuration);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
