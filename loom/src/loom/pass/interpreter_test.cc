// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/interpreter.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/pass/report.h"
#include "loom/pass/test/harness.h"

namespace loom {
namespace {

struct DiagnosticCapture {
  // Number of diagnostic emissions captured.
  int emission_count = 0;
  // Operation associated with the diagnostic emission.
  const loom_op_t* op = nullptr;
  // Structured diagnostic definition that was emitted.
  const loom_error_def_t* error = nullptr;
  // Number of captured diagnostic parameters.
  iree_host_size_t param_count = 0;
  // Bounded diagnostic parameter copy for assertions.
  loom_diagnostic_param_t params[4] = {};
};

class PassInterpreterTest : public PassTestHarness {
 protected:
  static iree_status_t CaptureDiagnostic(
      void* user_data, const loom_diagnostic_emission_t* emission) {
    DiagnosticCapture* capture = static_cast<DiagnosticCapture*>(user_data);
    ++capture->emission_count;
    capture->op = emission->op;
    capture->error = emission->error;
    capture->param_count = emission->param_count;
    EXPECT_LE(emission->param_count, IREE_ARRAYSIZE(capture->params));
    for (iree_host_size_t i = 0;
         i < emission->param_count && i < IREE_ARRAYSIZE(capture->params);
         ++i) {
      capture->params[i] = emission->params[i];
    }
    return iree_ok_status();
  }
};

TEST_F(PassInterpreterTest, RunsModuleAndFunctionPasses) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  test.module-noop\n"
                    "  for func {\n"
                    "    test.noop\n"
                    "  }\n"
                    "}\n"
                    "test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunModulePipeline(module, 0, &trace));

  EXPECT_EQ(trace.module_noop_invocation_count, 1);
  EXPECT_EQ(trace.noop_invocation_count, 2);
}

TEST_F(PassInterpreterTest, RunsFunctionRootProgram) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.noop\n"
                    "}\n"
                    "test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunFunctionPipeline(module, 0, 1, &trace));

  EXPECT_EQ(trace.noop_invocation_count, 1);
}

TEST_F(PassInterpreterTest, AppendsExecutionReportRecords) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  test.module-noop\n"
                    "  for func {\n"
                    "    test.mark-changed\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  PassProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  PassReportStorage report_storage;
  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options =
      InterpreterOptions(&trace, {}, &report_storage.report);
  loom_pass_run_result_t result = {};
  IREE_ASSERT_OK(loom_pass_interpreter_run_module(&storage.program, module,
                                                  &options, &result));

  ASSERT_EQ(report_storage.report.invocation_count, 2u);
  const loom_pass_report_invocation_t& module_record =
      report_storage.report.invocations[0];
  EXPECT_TRUE(iree_string_view_equal(module_record.pass_key,
                                     IREE_SV("test.module-noop")));
  EXPECT_TRUE(iree_string_view_equal(module_record.pipeline_symbol,
                                     IREE_SV("pipeline")));
  EXPECT_TRUE(
      iree_string_view_equal(module_record.symbol_name, IREE_SV("<none>")));
  EXPECT_EQ(module_record.anchor_kind, LOOM_PASS_MODULE);
  EXPECT_FALSE(module_record.changed);
  EXPECT_GE(module_record.duration_nanoseconds, 0);
  ASSERT_EQ(module_record.statistic_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(module_record.statistics[0].name,
                                     IREE_SV("invocations")));
  EXPECT_EQ(module_record.statistics[0].value, 1);

  const loom_pass_report_invocation_t& function_record =
      report_storage.report.invocations[1];
  EXPECT_TRUE(iree_string_view_equal(function_record.pass_key,
                                     IREE_SV("test.mark-changed")));
  EXPECT_TRUE(
      iree_string_view_equal(function_record.symbol_name, IREE_SV("main")));
  EXPECT_EQ(function_record.anchor_kind, LOOM_PASS_FUNCTION);
  EXPECT_TRUE(function_record.changed);
  EXPECT_GE(function_record.duration_nanoseconds, 0);
  ASSERT_EQ(function_record.statistic_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(function_record.statistics[0].name,
                                     IREE_SV("invocations")));
  EXPECT_EQ(function_record.statistics[0].value, 1);
  EXPECT_TRUE(iree_string_view_equal(function_record.statistics[1].name,
                                     IREE_SV("synthetic-events")));
  EXPECT_EQ(function_record.statistics[1].value, 1);
  ASSERT_EQ(function_record.detail_count, 1u);
  ASSERT_NE(function_record.details, nullptr);
  EXPECT_TRUE(iree_string_view_equal(function_record.details->category,
                                     IREE_SV("test.mark-changed")));
  ASSERT_EQ(function_record.details->field_count, 3u);
  EXPECT_TRUE(iree_string_view_equal(function_record.details->fields[0].name,
                                     IREE_SV("event")));
  EXPECT_TRUE(
      iree_string_view_equal(function_record.details->fields[0].string_value,
                             IREE_SV("synthetic-change")));
  EXPECT_TRUE(iree_string_view_equal(function_record.details->fields[1].name,
                                     IREE_SV("symbol")));
  EXPECT_TRUE(iree_string_view_equal(
      function_record.details->fields[1].string_value, IREE_SV("main")));
  EXPECT_TRUE(iree_string_view_equal(function_record.details->fields[2].name,
                                     IREE_SV("synthetic_event_count")));
  EXPECT_EQ(function_record.details->fields[2].uint64_value, 1u);
}

TEST_F(PassInterpreterTest, AppliesNamePredicateToCurrentFunction) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    where name(value = \"matmul\") {\n"
                    "      test.noop\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @matmul() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @not_matmul() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunModulePipeline(module, 0, &trace));

  EXPECT_EQ(trace.noop_invocation_count, 1);
}

TEST_F(PassInterpreterTest, AppliesAttrPredicateToCurrentFunction) {
  loom_module_t* module = Parse(
      IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
              "  for func {\n"
              "    where attr(name = \"visibility\", value = \"public\") {\n"
              "      test.noop\n"
              "    }\n"
              "  }\n"
              "}\n"
              "test.func public @exported() {\n"
              "  test.yield\n"
              "}\n"
              "test.func @private() {\n"
              "  test.yield\n"
              "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunModulePipeline(module, 0, &trace));

  ASSERT_EQ(trace.event_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(trace.events[0].symbol_name, IREE_SV("exported")));
}

TEST_F(PassInterpreterTest, AppliesTraitPredicateToCurrentFunction) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    where trait(name = \"symbol-define\") {\n"
                    "      test.noop\n"
                    "    }\n"
                    "    where trait(name = \"elementwise\") {\n"
                    "      test.mark-changed\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunModulePipeline(module, 0, &trace));

  EXPECT_EQ(trace.noop_invocation_count, 2);
  EXPECT_EQ(trace.mark_changed_invocation_count, 0);
}

TEST_F(PassInterpreterTest, AppliesProviderPredicateToCurrentFunction) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    where target(value = \"gfx-test\") {\n"
                    "      test.noop\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @selected() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @skipped() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  PassTestPredicateCapture predicate_capture = {
      /*.verify_count=*/{},
      /*.evaluate_count=*/{},
      /*.selected_symbol=*/IREE_SV("selected"),
  };
  loom_pass_predicate_provider_t predicate_provider =
      PassTestTargetPredicateProvider(&predicate_capture);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunModulePipeline(module, 0, &trace, predicate_provider));

  EXPECT_EQ(predicate_capture.verify_count, 1);
  EXPECT_EQ(predicate_capture.evaluate_count, 2);
  ASSERT_EQ(trace.event_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(trace.events[0].symbol_name, IREE_SV("selected")));
}

TEST_F(PassInterpreterTest, ExecutesFixedRepeat) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    repeat fixed(count = 3) {\n"
                    "      test.noop\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunModulePipeline(module, 0, &trace));

  EXPECT_EQ(trace.noop_invocation_count, 3);
}

TEST_F(PassInterpreterTest, StopsRepeatUntilConvergedWhenBodyIsStable) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    repeat until_converged(max_iterations = 4) {\n"
                    "      test.noop\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunModulePipeline(module, 0, &trace));

  EXPECT_EQ(trace.noop_invocation_count, 1);
}

TEST_F(PassInterpreterTest, ReportsRepeatUntilConvergedExhaustion) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    repeat until_converged(max_iterations = 2) {\n"
                    "      test.mark-changed\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        RunModulePipeline(module, 0, &trace));

  EXPECT_EQ(trace.mark_changed_invocation_count, 2);
}

TEST_F(PassInterpreterTest, ExposesDecodedOptionsToPassInstance) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    test.options(count = 5, mode = beta, string = "
                    "\"decoded\")\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  IREE_ASSERT_OK(RunModulePipeline(module, 0, &trace));

  EXPECT_EQ(trace.options_invocation_count, 1);
  EXPECT_EQ(trace.options_decoded_create_count, 1);
  EXPECT_EQ(trace.decoded_options_count_value, 5u);
  EXPECT_EQ(trace.decoded_options_mode_index, 1u);
  EXPECT_TRUE(iree_string_view_equal(trace.decoded_options_string_value,
                                     IREE_SV("decoded")));
}

TEST_F(PassInterpreterTest, PropagatesDescriptorCallbackFailure) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  test.fail\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  PassProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  DiagnosticCapture diagnostic_capture;
  loom_pass_interpreter_options_t options =
      InterpreterOptions(&trace, (iree_diagnostic_emitter_t){
                                     /*.fn=*/CaptureDiagnostic,
                                     /*.user_data=*/&diagnostic_capture,
                                 });
  loom_pass_run_result_t result = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        loom_pass_interpreter_run_module(
                            &storage.program, module, &options, &result));
  EXPECT_EQ(trace.fail_invocation_count, 1);
  EXPECT_EQ(diagnostic_capture.emission_count, 1);
  ASSERT_NE(diagnostic_capture.error, nullptr);
  EXPECT_EQ(diagnostic_capture.error->domain, LOOM_ERROR_DOMAIN_STRUCTURE);
  EXPECT_EQ(diagnostic_capture.error->code, 28);
  EXPECT_EQ(diagnostic_capture.op, storage.program.instructions[0].source.op);
  ASSERT_EQ(diagnostic_capture.param_count, 4);
  EXPECT_EQ(diagnostic_capture.params[0].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(diagnostic_capture.params[0].string,
                                     IREE_SV("test.fail")));
  EXPECT_EQ(diagnostic_capture.params[1].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(diagnostic_capture.params[1].string,
                                     IREE_SV("module")));
  EXPECT_EQ(diagnostic_capture.params[2].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(diagnostic_capture.params[2].string,
                                     IREE_SV("<none>")));
  EXPECT_EQ(diagnostic_capture.params[3].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(diagnostic_capture.params[3].string,
                                     IREE_SV("pipeline")));
}

TEST_F(PassInterpreterTest, ExecutesPassFailAndHalt) {
  loom_module_t* fail_module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  fail \"stop\"\n"
                    "}\n"));
  ASSERT_NE(fail_module, nullptr);
  PassProgramStorage fail_storage;
  IREE_ASSERT_OK(
      Compile(fail_module, Pipeline(fail_module, 0), &fail_storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  loom_pass_run_result_t fail_result = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_pass_interpreter_run_module(&fail_storage.program, fail_module,
                                       &options, &fail_result));

  loom_module_t* halt_module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  halt \"inspect\"\n"
                    "}\n"));
  ASSERT_NE(halt_module, nullptr);
  PassProgramStorage halt_storage;
  IREE_ASSERT_OK(
      Compile(halt_module, Pipeline(halt_module, 0), &halt_storage.program));

  loom_pass_run_result_t halt_result = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_pass_interpreter_run_module(&halt_storage.program, halt_module,
                                       &options, &halt_result));
}

}  // namespace
}  // namespace loom
