// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/expectation.h"

#include <string.h>

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

class ExpectationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);
    iree_arena_initialize(&block_pool_, &schedule_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(IREE_SV("testbench"), host_allocator_,
                                       host_allocator_, &device_allocator_));
  }

  void TearDown() override {
    iree_hal_allocator_release(device_allocator_);
    iree_arena_deinitialize(&schedule_arena_);
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
                                   IREE_SV("expectation_test.loom"), &context_,
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

  loom_testbench_value_materializer_options_t MaterializerOptions() {
    loom_testbench_value_materializer_options_t options = {};
    loom_testbench_value_materializer_options_initialize(&options);
    options.device_allocator = device_allocator_;
    return options;
  }

  std::string FailureDetail(
      const loom_testbench_expectation_report_t& report,
      const loom_testbench_expectation_failure_t& failure) {
    iree_string_view_t detail =
        loom_testbench_expectation_failure_detail(&report, &failure);
    return std::string(detail.data, detail.size);
  }

  iree_allocator_t host_allocator_ = iree_allocator_system();
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  iree_arena_allocator_t schedule_arena_;
  loom_context_t context_;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

TEST_F(ExpectationTest, EvaluatesScalarEqualityFailuresWithDetails) {
  loom_module_t* module = ParseModule(R"(
check.case @scalar_mismatch {
  %actual = check.literal value(43) : i32
  %expected = check.literal value(42) : i32
  check.expect.equal actual(%actual) expected(%expected) : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.case_count, 1u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];
  ASSERT_EQ(case_plan.expectation_count, 1u);
  EXPECT_EQ(case_plan.expectations[0].kind, LOOM_TESTBENCH_EXPECTATION_EQUAL);

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t materializer_options =
      MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(&materializer_options,
                                                        &case_plan, 0, &table));

  loom_testbench_expectation_options_t expectation_options = {};
  loom_testbench_expectation_options_initialize(&expectation_options);
  loom_testbench_expectation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_expectations(
      &expectation_options, &case_plan, &schedule_arena_, &schedule));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      schedule.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&schedule, &table,
                                                           nullptr, &report));

  EXPECT_EQ(report.expectation_count, 1u);
  EXPECT_EQ(report.passed_count, 0u);
  ASSERT_EQ(report.failure_count, 1u);
  EXPECT_EQ(report.failures[0].actual_value_id,
            case_plan.expectations[0].actual_value_id);
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("43"));
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("42"));
  iree_string_builder_t json_builder;
  iree_string_builder_initialize(host_allocator_, &json_builder);
  loom_output_stream_t json_stream;
  loom_output_stream_for_builder(&json_builder, &json_stream);
  IREE_ASSERT_OK(
      loom_testbench_expectation_report_write_json(&report, &json_stream));
  std::string json(iree_string_builder_view(&json_builder).data,
                   iree_string_builder_view(&json_builder).size);
  EXPECT_THAT(json, ::testing::HasSubstr("\"expectation_count\":1"));
  EXPECT_THAT(json, ::testing::HasSubstr("\"kind\":\"equal\""));
  EXPECT_THAT(json, ::testing::HasSubstr("\"detail\":"));
  EXPECT_THAT(json, ::testing::HasSubstr("43"));
  EXPECT_THAT(json, ::testing::HasSubstr("42"));
  iree_string_builder_deinitialize(&json_builder);

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(ExpectationTest, EvaluatesBufferShapeAndCloseExpectations) {
  loom_module_t* module = ParseModule(R"(
check.case @buffer_expectations {
  %m = check.param.choice values([4]) : index
  %actual_iota = check.generate.iota offset(0) step(1) : tensor<[%m]xi32>
  %actual = check.generate.fill value(1.0) : tensor<4xf32>
  %expected = check.generate.fill value(1.001) : tensor<4xf32>
  %actual_f16 = check.generate.fill value(0.5) : tensor<2xf16>
  %expected_f16 = check.generate.fill value(0.5005) : tensor<2xf16>
  %actual_bf16 = check.generate.fill value(0.25) : tensor<2xbf16>
  %expected_bf16 = check.generate.fill value(0.2505) : tensor<2xbf16>
  check.expect.shape value(%actual_iota) shape([%m]) : tensor<[%m]xi32>
  check.expect.close actual(%actual) expected(%expected) atol(0.01) rtol(0.0) nan(different) : tensor<4xf32>
  check.expect.close actual(%actual_f16) expected(%expected_f16) atol(0.01) rtol(0.0) nan(different) : tensor<2xf16>
  check.expect.close actual(%actual_bf16) expected(%expected_bf16) atol(0.01) rtol(0.0) nan(different) : tensor<2xbf16>
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.case_count, 1u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];
  ASSERT_EQ(case_plan.expectation_count, 4u);
  EXPECT_EQ(case_plan.expectations[0].kind, LOOM_TESTBENCH_EXPECTATION_SHAPE);
  EXPECT_EQ(case_plan.expectations[1].kind, LOOM_TESTBENCH_EXPECTATION_CLOSE);
  EXPECT_EQ(case_plan.expectations[2].kind, LOOM_TESTBENCH_EXPECTATION_CLOSE);
  EXPECT_EQ(case_plan.expectations[3].kind, LOOM_TESTBENCH_EXPECTATION_CLOSE);

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t materializer_options =
      MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(&materializer_options,
                                                        &case_plan, 0, &table));

  loom_testbench_expectation_options_t expectation_options = {};
  loom_testbench_expectation_options_initialize(&expectation_options);
  loom_testbench_expectation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_expectations(
      &expectation_options, &case_plan, &schedule_arena_, &schedule));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      schedule.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&schedule, &table,
                                                           nullptr, &report));

  EXPECT_EQ(report.expectation_count, 4u);
  EXPECT_EQ(report.passed_count, 4u);
  EXPECT_EQ(report.failure_count, 0u);

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(ExpectationTest, ReportsBufferCloseMismatchDetails) {
  loom_module_t* module = ParseModule(R"(
check.case @buffer_mismatch {
  %actual = check.generate.fill value(1.0) : tensor<4xf32>
  %expected = check.generate.fill value(1.2) : tensor<4xf32>
  check.expect.close actual(%actual) expected(%expected) atol(0.01) rtol(0.0) nan(different) : tensor<4xf32>
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t materializer_options =
      MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(&materializer_options,
                                                        &case_plan, 0, &table));

  loom_testbench_expectation_options_t expectation_options = {};
  loom_testbench_expectation_options_initialize(&expectation_options);
  loom_testbench_expectation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_expectations(
      &expectation_options, &case_plan, &schedule_arena_, &schedule));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      schedule.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&schedule, &table,
                                                           nullptr, &report));

  ASSERT_EQ(report.failure_count, 1u);
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("element at index 0"));
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("not close"));

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

static iree_status_t AlwaysFailsExpectation(
    void* user_data, const loom_testbench_expectation_plan_t* expectation,
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  (void)user_data;
  (void)actual;
  (void)expected;
  *out_matched = false;
  EXPECT_EQ(expectation->custom.attrs.count, 1u);
  return iree_string_builder_append_string(detail_builder,
                                           IREE_SV("custom validator failed"));
}

TEST_F(ExpectationTest, DispatchesCustomExpectationProvidersOncePrepared) {
  loom_module_t* module = ParseModule(R"(
check.case @custom {
  %actual = check.literal value(1) : i32
  %expected = check.literal value(1) : i32
  check.expect<always.fails> actual(%actual) expected(%expected) {k = 5} : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t materializer_options =
      MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(&materializer_options,
                                                        &case_plan, 0, &table));

  loom_testbench_expectation_provider_t providers[] = {
      {
          /*.name=*/IREE_SV("always.fails"),
          /*.evaluate=*/
          {
              /*.fn=*/AlwaysFailsExpectation,
          },
      },
  };
  loom_testbench_expectation_options_t expectation_options = {};
  loom_testbench_expectation_options_initialize(&expectation_options);
  expectation_options.providers = loom_make_testbench_expectation_provider_list(
      providers, IREE_ARRAYSIZE(providers));

  loom_testbench_expectation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_expectations(
      &expectation_options, &case_plan, &schedule_arena_, &schedule));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      schedule.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&schedule, &table,
                                                           nullptr, &report));

  ASSERT_EQ(report.failure_count, 1u);
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("custom validator failed"));

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(ExpectationTest, EvaluatesDeviceEventExpectations) {
  loom_module_t* module = ParseModule(R"(
check.case @device_event {
  check.expect.event<device> {type = "tsan_report", severity = "error", count = 1, driver = "amdgpu", tsan = {check = "data_race", memory = "workgroup", current_access = "write", prior_access = "read", access_length = 4, current_atomic = false, prior_atomic = false}}
  check.expect.event<device> {type = "asan_report", count = 0}
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.case_count, 1u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];
  ASSERT_EQ(case_plan.expectation_count, 2u);
  EXPECT_EQ(case_plan.expectations[0].kind, LOOM_TESTBENCH_EXPECTATION_EVENT);
  EXPECT_EQ(case_plan.expectations[1].kind, LOOM_TESTBENCH_EXPECTATION_EVENT);

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));

  loom_testbench_device_event_capture_t capture = {};
  IREE_ASSERT_OK(loom_testbench_device_event_capture_initialize(
      4, host_allocator_, &capture));
  iree_hal_device_tsan_report_t tsan_report = {};
  tsan_report.record_length = sizeof(tsan_report);
  tsan_report.abi_version = IREE_HAL_DEVICE_TSAN_REPORT_ABI_VERSION_0;
  tsan_report.check_kind = IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE;
  tsan_report.memory_space = IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP;
  tsan_report.current_access_kind = IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE;
  tsan_report.prior_access_kind = IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ;
  tsan_report.access_length = 4;
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.driver_id = IREE_SV("amdgpu");
  event.payload = iree_make_const_byte_span(&tsan_report, sizeof(tsan_report));
  iree_hal_device_event_sink_publish(
      loom_testbench_device_event_capture_sink(&capture), &event);
  loom_testbench_device_event_list_t event_list = {};
  loom_testbench_device_event_capture_events(&capture, &event_list);
  ASSERT_EQ(event_list.count, 1u);
  EXPECT_EQ((uintptr_t)0, (uintptr_t)event_list.records[0].event.payload.data %
                              iree_alignof(iree_hal_device_tsan_report_t));
  loom_testbench_case_sample_observations_t observations =
      loom_testbench_case_sample_observations_empty();
  observations.device_events = &event_list;

  loom_testbench_expectation_options_t expectation_options = {};
  loom_testbench_expectation_options_initialize(&expectation_options);
  loom_testbench_expectation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_expectations(
      &expectation_options, &case_plan, &schedule_arena_, &schedule));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      schedule.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(
      &schedule, &table, &observations, &report));

  EXPECT_EQ(report.expectation_count, 2u);
  EXPECT_EQ(report.passed_count, 2u);
  EXPECT_EQ(report.failure_count, 0u);

  uint8_t unaligned_payload_storage[sizeof(tsan_report) + 1] = {0};
  memcpy(unaligned_payload_storage + 1, &tsan_report, sizeof(tsan_report));
  loom_testbench_device_event_record_t unaligned_record = {};
  unaligned_record.event = event;
  unaligned_record.event.payload = iree_make_const_byte_span(
      unaligned_payload_storage + 1, sizeof(tsan_report));
  loom_testbench_device_event_list_t unaligned_event_list = {
      /*.records=*/&unaligned_record,
      /*.count=*/1,
  };
  observations.device_events = &unaligned_event_list;
  loom_testbench_expectation_report_reset(&report);
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(
      &schedule, &table, &observations, &report));
  EXPECT_EQ(report.expectation_count, 2u);
  EXPECT_EQ(report.passed_count, 2u);
  EXPECT_EQ(report.failure_count, 0u);

  iree_string_builder_t json_builder;
  iree_string_builder_initialize(host_allocator_, &json_builder);
  loom_output_stream_t json_stream;
  loom_output_stream_for_builder(&json_builder, &json_stream);
  IREE_ASSERT_OK(
      loom_testbench_expectation_report_write_json(&report, &json_stream));
  std::string json(iree_string_builder_view(&json_builder).data,
                   iree_string_builder_view(&json_builder).size);
  EXPECT_THAT(json, ::testing::HasSubstr("\"expectation_count\":2"));
  EXPECT_THAT(json, ::testing::HasSubstr("\"failure_count\":0"));
  iree_string_builder_deinitialize(&json_builder);

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_device_event_capture_deinitialize(&capture);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(ExpectationTest, RejectsUnsupportedDeviceEventExpectationKeys) {
  loom_module_t* module = ParseModule(R"(
check.case @device_event {
  check.expect.event<device> {typo = "tsan_report"}
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.case_count, 1u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_device_event_list_t event_list = {};
  loom_testbench_case_sample_observations_t observations =
      loom_testbench_case_sample_observations_empty();
  observations.device_events = &event_list;

  loom_testbench_expectation_options_t expectation_options = {};
  loom_testbench_expectation_options_initialize(&expectation_options);
  loom_testbench_expectation_schedule_t schedule = {};
  IREE_ASSERT_OK(loom_testbench_prepare_case_expectations(
      &expectation_options, &case_plan, &schedule_arena_, &schedule));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      schedule.expectation_count, host_allocator_, &report));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_testbench_evaluate_case_expectations(
                            &schedule, &table, &observations, &report));

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(ExpectationTest, FailsPreparationForMissingCustomProvider) {
  loom_module_t* module = ParseModule(R"(
check.case @custom {
  %actual = check.literal value(1) : i32
  %expected = check.literal value(1) : i32
  check.expect<missing.provider> actual(%actual) expected(%expected) : i32
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];

  loom_testbench_expectation_options_t expectation_options = {};
  loom_testbench_expectation_options_initialize(&expectation_options);
  loom_testbench_expectation_schedule_t schedule = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_testbench_prepare_case_expectations(&expectation_options, &case_plan,
                                               &schedule_arena_, &schedule));

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
