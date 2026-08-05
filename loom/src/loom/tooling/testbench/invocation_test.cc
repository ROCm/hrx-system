// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/invocation.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

typedef struct DeltaProviderState {
  // Value added to the i32 input.
  int32_t delta;
  // Number of issue queries received by the provider.
  iree_host_size_t issue_query_count;
} DeltaProviderState;

typedef struct LaunchProviderState {
  // Number of kernel launches observed by the provider.
  iree_host_size_t invocation_count;
  // Concrete workload value observed by the provider.
  int64_t workload;
  // Duplicated ABI value observed by the provider.
  int64_t abi_workload;
  // Scalar payload observed by the provider.
  int64_t payload;
} LaunchProviderState;

class InvocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
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
                                   IREE_SV("invocation_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_testbench_module_plan_t PlanModule(loom_module_t* module) {
    loom_testbench_module_plan_t plan = {};
    IREE_EXPECT_OK(loom_testbench_plan_module(module, nullptr, &arena_, &plan));
    return plan;
  }

  static iree_status_t InvokeDelta(
      void* user_data, const loom_testbench_invocation_plan_t* invocation,
      iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
      iree_host_size_t input_count, const loom_testbench_value_t* inputs,
      iree_host_size_t result_count, loom_testbench_value_t* out_results) {
    (void)invocation;
    (void)workloads;
    DeltaProviderState* state = static_cast<DeltaProviderState*>(user_data);
    if (workload_count != 0 || input_count != 1 || result_count != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "delta provider expects one input and result");
    }
    if (!loom_testbench_value_is_scalar(&inputs[0])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "delta provider input is not a scalar value");
    }
    const iree_tooling_value_t* input_value = &inputs[0].scalar;
    if (input_value->kind != IREE_TOOLING_VALUE_KIND_I32) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "delta provider input is not i32");
    }
    out_results[0] = {};
    out_results[0].kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
    out_results[0].scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
    out_results[0].scalar.storage.i32 = input_value->storage.i32 + state->delta;
    return iree_ok_status();
  }

  static iree_status_t QueryNoIssue(
      void* user_data, const loom_testbench_invocation_plan_t* invocation,
      loom_testbench_sample_issue_t* out_issue) {
    (void)invocation;
    DeltaProviderState* state = static_cast<DeltaProviderState*>(user_data);
    ++state->issue_query_count;
    *out_issue = {};
    return iree_ok_status();
  }

  static iree_status_t InvokeLaunch(
      void* user_data, const loom_testbench_invocation_plan_t* invocation,
      iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
      iree_host_size_t input_count, const loom_testbench_value_t* inputs,
      iree_host_size_t result_count, loom_testbench_value_t* out_results) {
    (void)invocation;
    (void)out_results;
    if (workload_count != 1 || input_count != 2 || result_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "launch provider expects one workload, two inputs, and no results");
    }
    LaunchProviderState* state = static_cast<LaunchProviderState*>(user_data);
    IREE_RETURN_IF_ERROR(
        loom_testbench_value_as_i64(&workloads[0], &state->workload));
    IREE_RETURN_IF_ERROR(
        loom_testbench_value_as_i64(&inputs[0], &state->payload));
    IREE_RETURN_IF_ERROR(
        loom_testbench_value_as_i64(&inputs[1], &state->abi_workload));
    ++state->invocation_count;
    return iree_ok_status();
  }

  static int32_t LookupI32(const loom_testbench_value_table_t* table,
                           loom_value_id_t value_id) {
    loom_testbench_value_t value = {};
    IREE_EXPECT_OK(
        loom_testbench_value_table_lookup_retain(table, value_id, &value));
    EXPECT_TRUE(loom_testbench_value_is_scalar(&value));
    EXPECT_EQ(value.scalar.kind, IREE_TOOLING_VALUE_KIND_I32);
    int32_t result = value.scalar.storage.i32;
    loom_testbench_value_deinitialize(&value);
    return result;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
};

TEST_F(InvocationTest, RunsActualAndOracleInvocations) {
  loom_module_t* module = ParseModule(R"(
test.func @callee(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.case @invoke {
  %input = check.literal value(5) : i32
  %actual = test.invoke @callee(%input) : (i32) -> (i32)
  %expected = check.oracle.call<reference.scalar> {delta = 20} callee(@callee) inputs(%input) : (i32) -> (i32)
  check.expect.equal actual(%actual) expected(%expected) : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];
  ASSERT_EQ(case_plan.invocation_count, 2u);
  EXPECT_EQ(case_plan.invocations[0].kind,
            LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL);
  EXPECT_EQ(case_plan.invocations[0].input_count, 1u);
  EXPECT_EQ(case_plan.invocations[0].result_count, 1u);
  EXPECT_EQ(case_plan.invocations[1].kind, LOOM_TESTBENCH_INVOCATION_ORACLE);
  EXPECT_TRUE(iree_string_view_equal(case_plan.invocations[1].provider,
                                     IREE_SV("reference.scalar")));
  EXPECT_EQ(case_plan.invocations[1].attrs.count, 1u);

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, iree_allocator_system(), &table));
  loom_testbench_value_materializer_options_t materializer_options = {};
  loom_testbench_value_materializer_options_initialize(&materializer_options);
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(
      &materializer_options, &case_plan, /*sample_ordinal=*/0, &table));

  DeltaProviderState actual_state = {};
  actual_state.delta = 10;
  DeltaProviderState oracle_state = {};
  oracle_state.delta = 20;
  loom_testbench_oracle_provider_t oracle_providers[1] = {};
  oracle_providers[0].name = IREE_SV("reference.scalar");
  oracle_providers[0].provider.invoke = InvocationTest::InvokeDelta;
  oracle_providers[0].provider.user_data = &oracle_state;
  loom_testbench_invocation_options_t invocation_options = {};
  loom_testbench_invocation_options_initialize(&invocation_options);
  invocation_options.function_call.invoke = InvocationTest::InvokeDelta;
  invocation_options.function_call.query_issue = InvocationTest::QueryNoIssue;
  invocation_options.function_call.user_data = &actual_state;
  invocation_options.oracle_providers =
      loom_make_testbench_oracle_provider_list(
          oracle_providers, IREE_ARRAYSIZE(oracle_providers));

  loom_testbench_invocation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_invocations(
      &invocation_options, &case_plan, &arena_, &schedule));
  ASSERT_EQ(schedule.invocation_count, 2u);
  EXPECT_EQ(schedule.max_workload_count, 0u);
  EXPECT_EQ(schedule.max_input_count, 1u);
  EXPECT_EQ(schedule.max_result_count, 1u);

  loom_testbench_invocation_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_invocation_executor_initialize(
      &schedule, iree_allocator_system(), &executor));
  IREE_ASSERT_OK(loom_testbench_run_case_invocations(
      &executor, /*sample_ordinal=*/0, &table));
  loom_testbench_invocation_executor_deinitialize(&executor);

  EXPECT_EQ(actual_state.issue_query_count, 1u);
  EXPECT_EQ(oracle_state.issue_query_count, 0u);

  EXPECT_EQ(LookupI32(&table, case_plan.invocations[0].result_value_ids[0]),
            15);
  EXPECT_EQ(LookupI32(&table, case_plan.invocations[1].result_value_ids[0]),
            25);

  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(InvocationTest, KeepsKernelWorkloadsIndependentFromAbiInputs) {
  loom_module_t* module = ParseModule(R"(
kernel.decl @callee(%workload: index) launch(%payload: i32, %abi_workload: index)

check.case @invoke {
  %workload = check.param.choice values([2, 5]) : index
  %payload = check.literal value(7) : i32
  kernel.launch @callee[%workload](%payload, %workload) : [index](i32, index)
  check.return
}
)");
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];
  ASSERT_EQ(case_plan.sample_count, 2u);
  ASSERT_EQ(case_plan.invocation_count, 1u);
  const loom_testbench_invocation_plan_t& invocation = case_plan.invocations[0];
  EXPECT_EQ(invocation.kind, LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH);
  ASSERT_EQ(invocation.workload_count, 1u);
  ASSERT_EQ(invocation.input_count, 2u);
  EXPECT_EQ(invocation.result_count, 0u);
  EXPECT_EQ(invocation.workload_value_ids[0], invocation.input_value_ids[1]);

  LaunchProviderState state = {};
  loom_testbench_invocation_options_t options = {};
  loom_testbench_invocation_options_initialize(&options);
  options.kernel_launch.invoke = InvocationTest::InvokeLaunch;
  options.kernel_launch.user_data = &state;

  loom_testbench_invocation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_invocations(&options, &case_plan,
                                                         &arena_, &schedule));
  EXPECT_EQ(schedule.max_workload_count, 1u);
  EXPECT_EQ(schedule.max_input_count, 2u);
  EXPECT_EQ(schedule.max_result_count, 0u);

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, iree_allocator_system(), &table));
  loom_testbench_value_materializer_options_t materializer_options = {};
  loom_testbench_value_materializer_options_initialize(&materializer_options);
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(
      &materializer_options, &case_plan, /*sample_ordinal=*/1, &table));
  loom_testbench_invocation_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_invocation_executor_initialize(
      &schedule, iree_allocator_system(), &executor));
  IREE_ASSERT_OK(loom_testbench_run_case_invocations(
      &executor, /*sample_ordinal=*/1, &table));

  EXPECT_EQ(state.invocation_count, 1u);
  EXPECT_EQ(state.workload, 5);
  EXPECT_EQ(state.abi_workload, 5);
  EXPECT_EQ(state.payload, 7);

  loom_testbench_invocation_executor_deinitialize(&executor);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(InvocationTest, ReportsMissingOracleProviderDuringPreparation) {
  loom_module_t* module = ParseModule(R"(
test.func @callee(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.case @invoke {
  %input = check.literal value(5) : i32
  %expected = check.oracle.call<reference.scalar> callee(@callee) inputs(%input) : (i32) -> (i32)
  check.return
}
)");
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.case_count, 1u);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.cases[0].invocation_count, 1u);

  loom_testbench_invocation_options_t invocation_options = {};
  loom_testbench_invocation_options_initialize(&invocation_options);
  loom_testbench_invocation_schedule_t schedule = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_testbench_prepare_case_invocations(
          &invocation_options, &plan.cases[0], &arena_, &schedule));

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
