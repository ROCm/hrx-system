// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/expectation.h"

#include <string.h>

#include <limits>
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

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(IREE_SV("testbench"), host_allocator_,
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

  loom_module_t* ParseModule(const char* source) {
    loom_text_parse_options_t options = {.max_errors = 20};
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

  void ExpectScalarClose(loom_scalar_type_t scalar_type,
                         iree_tooling_value_t actual,
                         iree_tooling_value_t expected,
                         loom_testbench_close_expectation_plan_t close,
                         bool matched) {
    // Construct the typed slots consumed by expectation evaluation.
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("scalar_close"),
                                        &block_pool_, nullptr, host_allocator_,
                                        &module));
    loom_testbench_expectation_plan_t expectation = {};
    expectation.kind = LOOM_TESTBENCH_EXPECTATION_CLOSE;
    expectation.type = loom_type_scalar(scalar_type);
    expectation.close = close;
    IREE_ASSERT_OK(loom_module_define_value(module, expectation.type,
                                            &expectation.actual_value_id));
    IREE_ASSERT_OK(loom_module_define_value(module, expectation.type,
                                            &expectation.expected_value_id));
    loom_testbench_case_plan_t case_plan = {.expectations = &expectation,
                                            .expectation_count = 1};
    loom_testbench_value_table_t table = {};
    IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
        module, &case_plan, host_allocator_, &table));
    loom_testbench_value_t actual_value = {};
    actual_value.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
    actual_value.scalar = actual;
    loom_testbench_value_t expected_value = {};
    expected_value.kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
    expected_value.scalar = expected;
    IREE_ASSERT_OK(loom_testbench_value_table_assign_move(
        &table, expectation.actual_value_id, &actual_value));
    IREE_ASSERT_OK(loom_testbench_value_table_assign_move(
        &table, expectation.expected_value_id, &expected_value));
    loom_testbench_expectation_report_t report = {};
    IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
        1, host_allocator_, &report));
    IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&case_plan, &table,
                                                             nullptr, &report));
    EXPECT_EQ(report.expectation_count, 1u);
    EXPECT_EQ(report.passed_count, matched ? 1u : 0u);
    EXPECT_EQ(report.failure_count, matched ? 0u : 1u);
    loom_testbench_expectation_report_deinitialize(&report);
    loom_testbench_value_table_deinitialize(&table);
    loom_module_free(module);
  }

  iree_allocator_t host_allocator_ = iree_allocator_system();
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

TEST_F(ExpectationTest, ComparesNarrowScalarsUsingSourceTypes) {
  struct FloatFormat {
    // Source interpretation of the raw scalar carrier.
    loom_scalar_type_t type;
    // Encoding of one.
    uint32_t one;
    // Encoding of the next value above one.
    uint32_t next;
    // Quiet NaN encoding.
    uint32_t nan;
  };
  const FloatFormat formats[] = {
      {LOOM_SCALAR_TYPE_F8E4M3, 0x38, 0x39, 0x7F},
      {LOOM_SCALAR_TYPE_F8E5M2, 0x3C, 0x3D, 0x7E},
      {LOOM_SCALAR_TYPE_F16, 0x3C00, 0x3C01, 0x7E00},
      {LOOM_SCALAR_TYPE_BF16, 0x3F80, 0x3F81, 0x7FC0},
  };
  for (const auto& format : formats) {
    SCOPED_TRACE(loom_scalar_type_name(format.type));
    iree_tooling_value_t actual = {.kind = IREE_TOOLING_VALUE_KIND_RAW_U32};
    actual.storage.u32 = format.one;
    iree_tooling_value_t expected = actual;
    loom_testbench_close_expectation_plan_t close = {};
    close.nan_policy = LOOM_CHECK_EXPECT_CLOSE_NAN_DIFFERENT;
    ExpectScalarClose(format.type, actual, expected, close, true);
    actual.storage.u32 = format.next;
    ExpectScalarClose(format.type, actual, expected, close, false);
    close.absolute_tolerance = 0.25;
    ExpectScalarClose(format.type, actual, expected, close, true);
    close.absolute_tolerance = 0.0;
    close.relative_tolerance = 0.25;
    ExpectScalarClose(format.type, actual, expected, close, true);
    actual.storage.u32 = format.nan;
    expected.storage.u32 = format.nan;
    ExpectScalarClose(format.type, actual, expected, close, false);
    close.nan_policy = LOOM_CHECK_EXPECT_CLOSE_NAN_SAME;
    ExpectScalarClose(format.type, actual, expected, close, true);
    expected.storage.u32 = format.one;
    ExpectScalarClose(format.type, actual, expected, close, false);
  }
}

TEST_F(ExpectationTest, ComparesInfinitiesIndependentlyOfTolerance) {
  const double infinity = std::numeric_limits<double>::infinity();
  struct Comparison {
    // Observed scalar value.
    double actual;
    // Reference scalar value.
    double expected;
    // Whether the values are numerically close.
    bool matched;
  };
  const Comparison comparisons[] = {
      {infinity, infinity, true},
      {-infinity, -infinity, true},
      {infinity, -infinity, false},
      {-infinity, infinity, false},
      {1.0, infinity, false},
      {infinity, 1.0, false},
      {-1.0, -infinity, false},
      {-infinity, -1.0, false},
      {0.0, -0.0, true},
      {-0.0, 0.0, true},
  };
  for (const auto& comparison : comparisons) {
    SCOPED_TRACE(::testing::Message()
                 << comparison.actual << " versus " << comparison.expected);
    for (double tolerance : {0.0, 2.0}) {
      iree_tooling_value_t actual = {.kind = IREE_TOOLING_VALUE_KIND_F64};
      actual.storage.f64 = comparison.actual;
      iree_tooling_value_t expected = {.kind = IREE_TOOLING_VALUE_KIND_F64};
      expected.storage.f64 = comparison.expected;
      loom_testbench_close_expectation_plan_t close = {
          .absolute_tolerance = tolerance,
          .relative_tolerance = tolerance,
          .nan_policy = LOOM_CHECK_EXPECT_CLOSE_NAN_DIFFERENT};
      ExpectScalarClose(LOOM_SCALAR_TYPE_F64, actual, expected, close,
                        comparison.matched);
    }
  }
}

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
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t materializer_options =
      MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(&materializer_options,
                                                        &case_plan, 0, &table));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      case_plan.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&case_plan, &table,
                                                           nullptr, &report));

  EXPECT_EQ(report.expectation_count, 1u);
  EXPECT_EQ(report.passed_count, 0u);
  ASSERT_EQ(report.failure_count, 1u);
  EXPECT_EQ(report.failures[0].diagnostic_ref,
            LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_EXPECT, 1));
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
  EXPECT_THAT(json, ::testing::HasSubstr("\"diagnostic\":\"EXPECT/001\""));
  EXPECT_THAT(json,
              ::testing::HasSubstr(
                  "\"source_location\":{\"filename\":\"expectation_test.loom\","
                  "\"start_line\":5,\"start_column\":3"));
  EXPECT_THAT(json, ::testing::HasSubstr("\"detail\":"));
  EXPECT_THAT(json, ::testing::HasSubstr("43"));
  EXPECT_THAT(json, ::testing::HasSubstr("42"));
  iree_string_builder_deinitialize(&json_builder);

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

TEST_F(ExpectationTest, ComparesBufferReferencesByLogicalIdentity) {
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("buffer_reference"),
                                      &block_pool_, nullptr, host_allocator_,
                                      &module));
  loom_testbench_expectation_plan_t expectation = {};
  expectation.kind = LOOM_TESTBENCH_EXPECTATION_EQUAL;
  expectation.type = loom_type_buffer();
  IREE_ASSERT_OK(loom_module_define_value(module, expectation.type,
                                          &expectation.actual_value_id));
  IREE_ASSERT_OK(loom_module_define_value(module, expectation.type,
                                          &expectation.expected_value_id));
  loom_testbench_case_plan_t case_plan = {.expectations = &expectation,
                                          .expectation_count = 1};
  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      1, host_allocator_, &report));

  auto compare = [&](loom_value_id_t actual_allocation,
                     iree_device_size_t actual_offset,
                     loom_value_id_t expected_allocation,
                     iree_device_size_t expected_offset) {
    loom_testbench_value_table_reset(&table);
    loom_testbench_value_t actual = {};
    actual.kind = LOOM_TESTBENCH_VALUE_KIND_BUFFER;
    actual.buffer = {};
    loom_testbench_value_set_buffer_reference(actual_allocation, actual_offset,
                                              /*byte_length=*/64, &actual);
    loom_testbench_value_t expected = {};
    expected.kind = LOOM_TESTBENCH_VALUE_KIND_BUFFER;
    expected.buffer = {};
    loom_testbench_value_set_buffer_reference(
        expected_allocation, expected_offset, /*byte_length=*/64, &expected);
    IREE_ASSERT_OK(loom_testbench_value_table_assign_move(
        &table, expectation.actual_value_id, &actual));
    IREE_ASSERT_OK(loom_testbench_value_table_assign_move(
        &table, expectation.expected_value_id, &expected));
    IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&case_plan, &table,
                                                             nullptr, &report));
  };

  compare(/*actual_allocation=*/7, /*actual_offset=*/16,
          /*expected_allocation=*/7, /*expected_offset=*/16);
  EXPECT_EQ(report.passed_count, 1u);
  EXPECT_EQ(report.failure_count, 0u);

  compare(/*actual_allocation=*/8, /*actual_offset=*/16,
          /*expected_allocation=*/7, /*expected_offset=*/16);
  ASSERT_EQ(report.failure_count, 1u);
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("allocation=8"));
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("allocation=7"));

  compare(/*actual_allocation=*/7, /*actual_offset=*/20,
          /*expected_allocation=*/7, /*expected_offset=*/16);
  ASSERT_EQ(report.failure_count, 1u);
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("offset=20"));
  EXPECT_THAT(FailureDetail(report, report.failures[0]),
              ::testing::HasSubstr("offset=16"));

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
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t materializer_options =
      MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(&materializer_options,
                                                        &case_plan, 0, &table));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      case_plan.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&case_plan, &table,
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
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_value_materializer_options_t materializer_options =
      MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_materialize_case_sample(&materializer_options,
                                                        &case_plan, 0, &table));

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      case_plan.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(&case_plan, &table,
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

TEST_F(ExpectationTest, EvaluatesDeviceEventExpectations) {
  loom_module_t* module = ParseModule(R"(
check.case @device_event {
  check.expect.event<device> {type = "tsan_report", severity = "error", count = 1, driver = "amdgpu", tsan = {check = "data_race", memory = "workgroup", current_access = "write", prior_access = "read", access_length = 4, memory_address = 12, current_atomic = false, prior_atomic = false}}
  check.expect.event<device> {type = "ubsan_report", count = 1, ubsan = {check = "assertion", site_id = 17, operand0 = 3, operand1 = 4}}
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
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
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
  tsan_report.memory_address = 12;
  iree_hal_device_event_t tsan_event = iree_hal_device_event_default();
  tsan_event.type = IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT;
  tsan_event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  tsan_event.source.driver_id = IREE_SV("amdgpu");
  tsan_event.payload =
      iree_make_const_byte_span(&tsan_report, sizeof(tsan_report));
  iree_hal_device_event_sink_publish(
      loom_testbench_device_event_capture_sink(&capture), &tsan_event);
  iree_hal_device_ubsan_report_t ubsan_report = {};
  ubsan_report.record_length = sizeof(ubsan_report);
  ubsan_report.abi_version = IREE_HAL_DEVICE_UBSAN_REPORT_ABI_VERSION_0;
  ubsan_report.check_kind = IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ASSERTION;
  ubsan_report.site_id = 17;
  ubsan_report.operand0 = 3;
  ubsan_report.operand1 = 4;
  iree_hal_device_event_t ubsan_event = iree_hal_device_event_default();
  ubsan_event.type = IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT;
  ubsan_event.payload =
      iree_make_const_byte_span(&ubsan_report, sizeof(ubsan_report));
  iree_hal_device_event_sink_publish(
      loom_testbench_device_event_capture_sink(&capture), &ubsan_event);
  loom_testbench_device_event_list_t event_list = {};
  loom_testbench_device_event_capture_events(&capture, &event_list);
  ASSERT_EQ(event_list.count, 2u);
  EXPECT_EQ((uintptr_t)0, (uintptr_t)event_list.records[0].event.payload.data %
                              iree_alignof(iree_hal_device_tsan_report_t));
  EXPECT_EQ((uintptr_t)0, (uintptr_t)event_list.records[1].event.payload.data %
                              iree_alignof(iree_hal_device_ubsan_report_t));
  loom_testbench_sample_observations_t observations =
      loom_testbench_sample_observations_empty();
  uint8_t expected_device_events[4] = {0};
  observations.device_events = &event_list;
  observations.expected_device_events = expected_device_events;
  observations.expected_device_event_capacity =
      IREE_ARRAYSIZE(expected_device_events);

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      case_plan.expectation_count, host_allocator_, &report));
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(
      &case_plan, &table, &observations, &report));

  EXPECT_EQ(report.expectation_count, 2u);
  EXPECT_EQ(report.passed_count, 2u);
  EXPECT_EQ(report.failure_count, 0u);
  EXPECT_EQ(expected_device_events[0], 1u);
  EXPECT_EQ(expected_device_events[1], 1u);

  uint8_t unaligned_tsan_payload[sizeof(tsan_report) + 1] = {0};
  memcpy(unaligned_tsan_payload + 1, &tsan_report, sizeof(tsan_report));
  uint8_t unaligned_ubsan_payload[sizeof(ubsan_report) + 1] = {0};
  memcpy(unaligned_ubsan_payload + 1, &ubsan_report, sizeof(ubsan_report));
  loom_testbench_device_event_record_t unaligned_records[2] = {};
  unaligned_records[0].event = tsan_event;
  unaligned_records[0].event.payload = iree_make_const_byte_span(
      unaligned_tsan_payload + 1, sizeof(tsan_report));
  unaligned_records[1].event = ubsan_event;
  unaligned_records[1].event.payload = iree_make_const_byte_span(
      unaligned_ubsan_payload + 1, sizeof(ubsan_report));
  loom_testbench_device_event_list_t unaligned_event_list = {
      /*.records=*/unaligned_records,
      /*.count=*/IREE_ARRAYSIZE(unaligned_records),
  };
  observations.device_events = &unaligned_event_list;
  memset(expected_device_events, 0, sizeof(expected_device_events));
  loom_testbench_expectation_report_reset(&report);
  IREE_ASSERT_OK(loom_testbench_evaluate_case_expectations(
      &case_plan, &table, &observations, &report));
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

TEST_F(ExpectationTest, RejectsZeroDeviceEventCount) {
  loom_module_t* module = ParseModule(R"(
check.case @device_event {
  check.expect.event<device> {type = "asan_report", count = 0}
  check.return
}
)");
  ASSERT_NE(module, nullptr);

  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.case_count, 1u);
  const loom_testbench_case_plan_t& case_plan = plan.cases[0];

  loom_testbench_value_table_t table = {};
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_device_event_list_t event_list = {};
  loom_testbench_sample_observations_t observations =
      loom_testbench_sample_observations_empty();
  observations.device_events = &event_list;

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      case_plan.expectation_count, host_allocator_, &report));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_testbench_evaluate_case_expectations(
                            &case_plan, &table, &observations, &report));

  loom_testbench_expectation_report_deinitialize(&report);
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
  IREE_ASSERT_OK(loom_testbench_value_table_initialize_case(
      module, &case_plan, host_allocator_, &table));
  loom_testbench_device_event_list_t event_list = {};
  loom_testbench_sample_observations_t observations =
      loom_testbench_sample_observations_empty();
  observations.device_events = &event_list;

  loom_testbench_expectation_report_t report = {};
  IREE_ASSERT_OK(loom_testbench_expectation_report_initialize(
      case_plan.expectation_count, host_allocator_, &report));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_testbench_evaluate_case_expectations(
                            &case_plan, &table, &observations, &report));

  loom_testbench_expectation_report_deinitialize(&report);
  loom_testbench_value_table_deinitialize(&table);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
