// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/invocation.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

typedef struct DeltaProviderState {
  // Value added to the i32 input.
  int32_t delta;
} DeltaProviderState;

class InvocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
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
    out_results[0] = iree_vm_make_variant_value(
        iree_vm_value_make_i32(input_value.i32 + state->delta));
    return iree_ok_status();
  }

  static int32_t LookupI32(const loom_testbench_value_table_t* table,
                           loom_value_id_t value_id) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_EXPECT_OK(
        loom_testbench_value_table_lookup_retain(table, value_id, &variant));
    EXPECT_TRUE(iree_vm_variant_is_value(variant));
    iree_vm_value_t value = iree_vm_variant_value(variant);
    EXPECT_EQ(value.type, IREE_VM_VALUE_TYPE_I32);
    iree_vm_variant_reset(&variant);
    return value.i32;
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
  EXPECT_EQ(case_plan.invocations[0].kind, LOOM_TESTBENCH_INVOCATION_ACTUAL);
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
  oracle_providers[0].invoke.fn = InvocationTest::InvokeDelta;
  oracle_providers[0].invoke.user_data = &oracle_state;
  loom_testbench_invocation_options_t invocation_options = {};
  loom_testbench_invocation_options_initialize(&invocation_options);
  invocation_options.invoke_actual.fn = InvocationTest::InvokeDelta;
  invocation_options.invoke_actual.user_data = &actual_state;
  invocation_options.oracle_providers =
      loom_make_testbench_oracle_provider_list(
          oracle_providers, IREE_ARRAYSIZE(oracle_providers));

  loom_testbench_invocation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_invocations(
      &invocation_options, &case_plan, &arena_, &schedule));
  ASSERT_EQ(schedule.invocation_count, 2u);
  EXPECT_EQ(schedule.max_input_count, 1u);
  EXPECT_EQ(schedule.max_result_count, 1u);

  loom_testbench_invocation_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_invocation_executor_initialize(
      &schedule, iree_allocator_system(), &executor));
  IREE_ASSERT_OK(loom_testbench_run_case_invocations(&executor, &table));
  loom_testbench_invocation_executor_deinitialize(&executor);

  EXPECT_EQ(LookupI32(&table, case_plan.invocations[0].result_value_ids[0]),
            15);
  EXPECT_EQ(LookupI32(&table, case_plan.invocations[1].result_value_ids[0]),
            25);

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
