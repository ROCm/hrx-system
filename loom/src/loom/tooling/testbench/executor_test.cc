// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/executor.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
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
  // True when the provider returns an error after observing its input.
  bool fail_invocation;
  // Number of times the provider has been invoked.
  iree_host_size_t invocation_count;
  // Last i32 input observed by the provider.
  int32_t last_input;
  // Device events published after a successful invocation.
  const iree_hal_device_event_t* device_events;
  // Number of entries in |device_events|.
  iree_host_size_t device_event_count;
  // Sink receiving entries from |device_events|.
  iree_hal_device_event_sink_t device_event_sink;
} DeltaProviderState;

typedef struct DeviceEventExecutionResult {
  // True when the sample passed every result policy.
  bool passed;
  // Number of captured device events.
  iree_host_size_t captured_event_count;
  // Number of device events lost by the capture sink.
  iree_host_size_t dropped_event_count;
  // Number of error events not matched by an explicit expectation.
  iree_host_size_t unhandled_event_count;
  // Number of authored expectations that passed.
  iree_host_size_t passed_expectation_count;
  // Number of authored expectations that failed.
  iree_host_size_t failed_expectation_count;
  // Expected flags copied in capture order.
  std::vector<uint8_t> expected_events;
  // Serialized sample report.
  std::string json;
} DeviceEventExecutionResult;

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
    loom_text_parse_options_t options = {.max_errors = 20};
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
    ++state->invocation_count;
    state->last_input = input_value->storage.i32;
    if (state->fail_invocation) {
      return iree_make_status(IREE_STATUS_ABORTED,
                              "delta provider invocation failed");
    }
    for (iree_host_size_t i = 0; i < state->device_event_count; ++i) {
      iree_hal_device_event_sink_publish(state->device_event_sink,
                                         &state->device_events[i]);
    }
    out_results[0] = {};
    out_results[0].kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
    out_results[0].scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
    out_results[0].scalar.storage.i32 = input_value->storage.i32 + state->delta;
    return iree_ok_status();
  }

  loom_testbench_case_execution_options_t DeltaExecutionOptions(
      DeltaProviderState* actual_state, DeltaProviderState* oracle_state,
      loom_testbench_oracle_provider_t* oracle_providers) {
    loom_testbench_case_execution_options_t options = {};
    loom_testbench_case_execution_options_initialize(&options);
    options.invocation.function_call.invoke = ExecutorTest::InvokeDelta;
    options.invocation.function_call.user_data = actual_state;
    oracle_providers[0].name = IREE_SV("reference.scalar");
    oracle_providers[0].provider.invoke = ExecutorTest::InvokeDelta;
    oracle_providers[0].provider.user_data = oracle_state;
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

  DeviceEventExecutionResult RunDeviceEventCase(
      const char* expectation, const iree_hal_device_event_t* device_events,
      iree_host_size_t device_event_count,
      iree_host_size_t capture_capacity = 4) {
    std::string source = R"(
test.func @callee(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.case @device_events {
  %input = check.literal value(5) : i32
  %actual = test.invoke @callee(%input) : (i32) -> (i32)
)";
    source.append(expectation);
    source.append(R"(  check.return
}
)");

    loom_module_t* module = ParseModule(source.c_str());
    loom_testbench_module_plan_t plan = PlanModule(module);
    EXPECT_EQ(plan.issue_count, 0u);

    loom_testbench_device_event_capture_t capture = {};
    IREE_EXPECT_OK(loom_testbench_device_event_capture_initialize(
        capture_capacity, iree_allocator_system(), &capture));
    DeltaProviderState actual_state = {
        .device_events = device_events,
        .device_event_count = device_event_count,
        .device_event_sink =
            loom_testbench_device_event_capture_sink(&capture)};
    loom_testbench_case_execution_options_t options = {};
    loom_testbench_case_execution_options_initialize(&options);
    options.invocation.function_call.invoke = ExecutorTest::InvokeDelta;
    options.invocation.function_call.user_data = &actual_state;
    options.device_event_capture = &capture;

    loom_testbench_prepared_case_t prepared_case = {};
    IREE_EXPECT_OK(loom_testbench_prepare_case_execution(
        &options, &plan, 0, &execution_arena_, &prepared_case));
    loom_testbench_case_executor_t executor = {};
    IREE_EXPECT_OK(loom_testbench_case_executor_initialize(
        &prepared_case, &options, &executor));
    loom_testbench_case_sample_result_t sample_result = {};
    IREE_EXPECT_OK(
        loom_testbench_run_case_sample(&executor, 0, &sample_result));

    DeviceEventExecutionResult result =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment follows
             // nontrivial member initialization.
    result.passed = sample_result.passed;
    result.captured_event_count = sample_result.device_events->count;
    result.dropped_event_count = sample_result.device_events->dropped_count;
    result.unhandled_event_count = sample_result.unhandled_device_event_count;
    result.passed_expectation_count =
        sample_result.expectation_report->passed_count;
    result.failed_expectation_count =
        sample_result.expectation_report->failure_count;
    result.expected_events.assign(
        sample_result.expected_device_events,
        sample_result.expected_device_events + result.captured_event_count);
    result.json = WriteResultJson(sample_result);

    loom_testbench_case_executor_deinitialize(&executor);
    loom_testbench_device_event_capture_deinitialize(&capture);
    loom_module_free(module);
    return result;
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

  DeltaProviderState actual_state = {.delta = 3};
  DeltaProviderState oracle_state = {.delta = 3};
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

  DeltaProviderState actual_state = {.delta = 0};
  DeltaProviderState oracle_state = {.delta = 1};
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

TEST_F(ExecutorTest, PassesCleanExecutionWithDeviceEventCapture) {
  DeviceEventExecutionResult result = RunDeviceEventCase("", nullptr, 0);

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.captured_event_count, 0u);
  EXPECT_EQ(result.unhandled_event_count, 0u);
}

TEST_F(ExecutorTest, FailsOnUnhandledDeviceError) {
  iree_hal_device_event_site_t site = iree_hal_device_event_site_default();
  site.site_id = 7;
  site.source_file = IREE_SV("kernel.loom");
  site.start_line = 42;
  site.start_column = 3;
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.driver_id = IREE_SV("amdgpu");
  event.site = &site;
  DeviceEventExecutionResult result = RunDeviceEventCase("", &event, 1);

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.captured_event_count, 1u);
  EXPECT_EQ(result.unhandled_event_count, 1u);
  EXPECT_THAT(result.json, ::testing::HasSubstr("\"type\":\"asan_report\""));
  EXPECT_THAT(result.json, ::testing::HasSubstr("\"unhandled_error_count\":1"));
  EXPECT_THAT(result.json,
              ::testing::HasSubstr("\"source_file\":\"kernel.loom\""));
  EXPECT_THAT(result.json, ::testing::HasSubstr("\"start_line\":42"));
}

TEST_F(ExecutorTest, PassesDeviceWarning) {
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_WARNING;
  DeviceEventExecutionResult result = RunDeviceEventCase("", &event, 1);

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.captured_event_count, 1u);
  EXPECT_EQ(result.unhandled_event_count, 0u);
}

TEST_F(ExecutorTest, AccountsForExpectedDeviceError) {
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  DeviceEventExecutionResult result = RunDeviceEventCase(
      "  check.expect.event<device> {type = \"asan_report\"}\n", &event, 1);

  EXPECT_TRUE(result.passed);
  EXPECT_EQ(result.passed_expectation_count, 1u);
  EXPECT_EQ(result.unhandled_event_count, 0u);
  ASSERT_EQ(result.expected_events.size(), 1u);
  EXPECT_EQ(result.expected_events[0], 1u);
  EXPECT_THAT(result.json,
              ::testing::Not(::testing::HasSubstr("\"device_events\":")));
}

TEST_F(ExecutorTest, FailsOnDeviceErrorOutsideExpectedPattern) {
  iree_hal_device_event_t events[2] = {
      iree_hal_device_event_default(),
      iree_hal_device_event_default(),
  };
  events[0].type = IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT;
  events[0].severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  events[1].type = IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT;
  events[1].severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  DeviceEventExecutionResult result = RunDeviceEventCase(
      "  check.expect.event<device> {type = \"asan_report\"}\n", events,
      IREE_ARRAYSIZE(events));

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.failed_expectation_count, 0u);
  EXPECT_EQ(result.unhandled_event_count, 1u);
  ASSERT_EQ(result.expected_events.size(), 2u);
  EXPECT_EQ(result.expected_events[0], 1u);
  EXPECT_EQ(result.expected_events[1], 0u);
}

TEST_F(ExecutorTest, FailsWhenDeviceEventCaptureOverflows) {
  iree_hal_device_event_t events[2] = {
      iree_hal_device_event_default(),
      iree_hal_device_event_default(),
  };
  events[0].severity = IREE_HAL_DEVICE_EVENT_SEVERITY_INFO;
  events[1].severity = IREE_HAL_DEVICE_EVENT_SEVERITY_INFO;
  DeviceEventExecutionResult result =
      RunDeviceEventCase("", events, IREE_ARRAYSIZE(events), 1);

  EXPECT_FALSE(result.passed);
  EXPECT_EQ(result.captured_event_count, 1u);
  EXPECT_EQ(result.dropped_event_count, 1u);
}

TEST_F(ExecutorTest, AnnotatesProviderFailureWithCaseAndSample) {
  loom_module_t* module = ParseModule(R"(
test.func @callee(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.case @provider_error {
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

  DeltaProviderState actual_state = {.fail_invocation = true};
  DeltaProviderState oracle_state = {};
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
  iree::Status status(loom_testbench_run_case_sample(&executor, 1, &result));
  EXPECT_THAT(status,
              iree::testing::status::StatusIs(iree::StatusCode::kAborted));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("delta provider invocation failed"));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("check.case '@provider_error' sample 1"));
  EXPECT_EQ(actual_state.invocation_count, 1u);
  EXPECT_EQ(actual_state.last_input, 7);
  EXPECT_EQ(oracle_state.invocation_count, 0u);

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
