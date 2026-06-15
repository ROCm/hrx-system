// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/executor.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

typedef struct DeltaProviderState {
  // Value added to each i32 input.
  int32_t delta;
  // Number of times the provider has been invoked.
  iree_host_size_t invocation_count;
  // Last i32 input observed by the provider.
  int32_t last_input;
} DeltaProviderState;

class ExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);
    iree_arena_initialize(&block_pool_, &execution_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&execution_arena_);
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

  loom_module_t* ParseModule(const char* source) {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("executor_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_testbench_module_plan_t PlanModule(loom_module_t* module) {
    loom_testbench_module_plan_t plan = {};
    IREE_EXPECT_OK(
        loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));
    return plan;
  }

  static iree_status_t InvokeDelta(
      void* user_data, const loom_testbench_invocation_plan_t* invocation,
      iree_host_size_t input_count, const iree_vm_variant_t* inputs,
      iree_host_size_t result_count, iree_vm_variant_t* out_results) {
    (void)invocation;
    DeltaProviderState* state = static_cast<DeltaProviderState*>(user_data);
    if (input_count != 1 || result_count != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "delta provider expects one input and result");
    }
    if (!iree_vm_variant_is_value(inputs[0])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "delta provider input is not a VM value");
    }
    iree_vm_value_t input_value = iree_vm_variant_value(inputs[0]);
    if (input_value.type != IREE_VM_VALUE_TYPE_I32) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "delta provider input is not i32");
    }
    ++state->invocation_count;
    state->last_input = input_value.i32;
    out_results[0] = iree_vm_make_variant_value(
        iree_vm_value_make_i32(input_value.i32 + state->delta));
    return iree_ok_status();
  }

  loom_testbench_case_execution_options_t DeltaExecutionOptions(
      DeltaProviderState* actual_state, DeltaProviderState* oracle_state,
      loom_testbench_oracle_provider_t* oracle_providers) {
    loom_testbench_case_execution_options_t options = {};
    loom_testbench_case_execution_options_initialize(&options);
    options.invocation.invoke_actual.fn = ExecutorTest::InvokeDelta;
    options.invocation.invoke_actual.user_data = actual_state;
    oracle_providers[0].name = IREE_SV("reference.scalar");
    oracle_providers[0].invoke.fn = ExecutorTest::InvokeDelta;
    oracle_providers[0].invoke.user_data = oracle_state;
    options.invocation.oracle_providers =
        loom_make_testbench_oracle_provider_list(oracle_providers, 1);
    return options;
  }

  static std::string WriteResultJson(
      const loom_testbench_case_sample_result_t& result) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    IREE_EXPECT_OK(
        loom_testbench_case_sample_result_write_json(&result, &stream));
    iree_string_view_t view = iree_string_builder_view(&builder);
    std::string json(view.data, view.size);
    iree_string_builder_deinitialize(&builder);
    return json;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  iree_arena_allocator_t execution_arena_;
  loom_context_t context_;
};

TEST_F(ExecutorTest, ReusesPreparedCaseAcrossSamples) {
  loom_module_t* module = ParseModule(R"(
test.func @callee(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.case @sampled {
  %input = check.param.choice values([5, 7]) : i32
  %actual = test.invoke @callee(%input) : (i32) -> (i32)
  %expected = check.oracle.call<reference.scalar> callee(@callee) inputs(%input) : (i32) -> (i32)
  check.expect.equal actual(%actual) expected(%expected) : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.cases[0].sample_count, 2u);

  DeltaProviderState actual_state = {};
  actual_state.delta = 3;
  DeltaProviderState oracle_state = {};
  oracle_state.delta = 3;
  loom_testbench_oracle_provider_t oracle_providers[1] = {};
  loom_testbench_case_execution_options_t options =
      DeltaExecutionOptions(&actual_state, &oracle_state, oracle_providers);

  loom_testbench_prepared_case_t prepared_case = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_execution(
      &options, &plan, 0, &execution_arena_, &prepared_case));
  loom_testbench_case_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_case_executor_initialize(&prepared_case,
                                                         &options, &executor));

  loom_testbench_case_sample_result_t result = {};
  IREE_ASSERT_OK(loom_testbench_run_case_sample(&executor, 0, &result));
  EXPECT_TRUE(result.passed);
  EXPECT_EQ(actual_state.last_input, 5);
  EXPECT_EQ(oracle_state.last_input, 5);
  EXPECT_EQ(result.expectation_report->passed_count, 1u);
  EXPECT_EQ(result.expectation_report->failure_count, 0u);

  IREE_ASSERT_OK(loom_testbench_run_case_sample(&executor, 1, &result));
  EXPECT_TRUE(result.passed);
  EXPECT_EQ(actual_state.last_input, 7);
  EXPECT_EQ(oracle_state.last_input, 7);
  EXPECT_EQ(actual_state.invocation_count, 2u);
  EXPECT_EQ(oracle_state.invocation_count, 2u);

  std::string json = WriteResultJson(result);
  EXPECT_THAT(json, ::testing::HasSubstr("\"case\":\"sampled\""));
  EXPECT_THAT(json, ::testing::HasSubstr("\"sample_ordinal\":1"));
  EXPECT_THAT(json, ::testing::HasSubstr("\"passed\":true"));
  EXPECT_THAT(json, ::testing::HasSubstr("\"expectation_count\":1"));
  EXPECT_THAT(json, ::testing::HasSubstr("\"failure_count\":0"));

  loom_testbench_case_executor_deinitialize(&executor);
  loom_module_free(module);
}

TEST_F(ExecutorTest, ReportsExpectationFailuresWithoutFailingExecution) {
  loom_module_t* module = ParseModule(R"(
test.func @callee(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.case @mismatch {
  %input = check.literal value(5) : i32
  %actual = test.invoke @callee(%input) : (i32) -> (i32)
  %expected = check.oracle.call<reference.scalar> callee(@callee) inputs(%input) : (i32) -> (i32)
  check.expect.equal actual(%actual) expected(%expected) : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 0u);

  DeltaProviderState actual_state = {};
  actual_state.delta = 0;
  DeltaProviderState oracle_state = {};
  oracle_state.delta = 1;
  loom_testbench_oracle_provider_t oracle_providers[1] = {};
  loom_testbench_case_execution_options_t options =
      DeltaExecutionOptions(&actual_state, &oracle_state, oracle_providers);

  loom_testbench_prepared_case_t prepared_case = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_execution(
      &options, &plan, 0, &execution_arena_, &prepared_case));
  loom_testbench_case_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_case_executor_initialize(&prepared_case,
                                                         &options, &executor));

  loom_testbench_case_sample_result_t result = {};
  IREE_ASSERT_OK(loom_testbench_run_case_sample(&executor, 0, &result));
  EXPECT_FALSE(result.passed);
  ASSERT_EQ(result.expectation_report->failure_count, 1u);

  std::string json = WriteResultJson(result);
  EXPECT_THAT(json, ::testing::HasSubstr("\"case\":\"mismatch\""));
  EXPECT_THAT(json, ::testing::HasSubstr("\"passed\":false"));
  EXPECT_THAT(json, ::testing::HasSubstr("\"failure_count\":1"));
  EXPECT_THAT(json, ::testing::HasSubstr("actual i32 value 5"));
  EXPECT_THAT(json, ::testing::HasSubstr("expected 6"));

  loom_testbench_case_executor_deinitialize(&executor);
  loom_module_free(module);
}

TEST_F(ExecutorTest, RejectsCasePlanningIssuesBeforePreparation) {
  loom_module_t* module = ParseModule(R"(
check.case @bad {
  %input = check.literal value(5) : i32
  test.use %input : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 1u);

  loom_testbench_case_execution_options_t options = {};
  loom_testbench_case_execution_options_initialize(&options);
  loom_testbench_prepared_case_t prepared_case = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_testbench_prepare_case_execution(&options, &plan, 0,
                                            &execution_arena_, &prepared_case));

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
