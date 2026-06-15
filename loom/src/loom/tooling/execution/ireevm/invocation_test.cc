// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/ireevm/invocation.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/ireevm/descriptors/low_registry.h"
#include "loom/target/arch/ireevm/ops/registry.h"
#include "loom/tooling/execution/ireevm/candidate.h"
#include "loom/tooling/execution/session.h"

namespace loom {
namespace {

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  return loom_ireevm_ops_register_dialect(context);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_ireevm_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

constexpr char kPreparedVmSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "@double(%value: reg<ireevm.i32>) -> (reg<ireevm.i32>) {\n"
    "  %sum = low.op<ireevm.add.i32>(%value, %value) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.return %sum : reg<ireevm.i32>\n"
    "}\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"branchy\") "
    "@branchy(%lhs: reg<ireevm.i32>, %rhs: reg<ireevm.i32>) -> "
    "(reg<ireevm.i32>) {\n"
    "  %c0 = low.const<ireevm.const.i32> {i32_value = 0} : reg<ireevm.i32>\n"
    "  %is_zero = low.op<ireevm.cmp.eq.i32>(%lhs, %c0) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.cond_br %is_zero, ^then, ^else : reg<ireevm.i32>\n"
    "^then:\n"
    "  %sum = low.func.call @double(%rhs) : "
    "(reg<ireevm.i32>) -> (reg<ireevm.i32>)\n"
    "  low.return %sum : reg<ireevm.i32>\n"
    "^else:\n"
    "  %diff = low.op<ireevm.sub.i32>(%lhs, %rhs) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.return %diff : reg<ireevm.i32>\n"
    "}\n";

constexpr char kPreparedVmWideSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"wide_numeric\") "
    "@wide_numeric(%lhs: reg<ireevm.i64 x2>, %rhs: reg<ireevm.i64 x2>, "
    "%x: reg<ireevm.f32>, %y: reg<ireevm.f32>, "
    "%z: reg<ireevm.f64 x2>) -> "
    "(reg<ireevm.i64 x2>, reg<ireevm.f32>, reg<ireevm.f64 x2>, "
    "reg<ireevm.i32>) {\n"
    "  %sum = low.op<ireevm.add.i64>(%lhs, %rhs) : "
    "(reg<ireevm.i64 x2>, reg<ireevm.i64 x2>) -> reg<ireevm.i64 x2>\n"
    "  %product = low.op<ireevm.mul.f32>(%x, %y) : "
    "(reg<ireevm.f32>, reg<ireevm.f32>) -> reg<ireevm.f32>\n"
    "  %one_point_five = low.const<ireevm.const.f64> "
    "{f64_bits = 4609434218613702656} : reg<ireevm.f64 x2>\n"
    "  %wide_sum = low.op<ireevm.add.f64>(%z, %one_point_five) : "
    "(reg<ireevm.f64 x2>, reg<ireevm.f64 x2>) -> reg<ireevm.f64 x2>\n"
    "  %less = low.op<ireevm.cmp.lt.o.f64>(%z, %wide_sum) : "
    "(reg<ireevm.f64 x2>, reg<ireevm.f64 x2>) -> reg<ireevm.i32>\n"
    "  low.return %sum, %product, %wide_sum, %less : "
    "reg<ireevm.i64 x2>, reg<ireevm.f32>, reg<ireevm.f64 x2>, "
    "reg<ireevm.i32>\n"
    "}\n";

class VmInvocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_run_session_options_t options = {};
    loom_run_session_options_initialize(&options);
    options.register_context = (loom_run_register_context_callback_t){
        /*.fn=*/RegisterContext,
    };
    options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            /*.fn=*/InitializeLowDescriptorRegistry,
        };
    IREE_ASSERT_OK(loom_run_session_initialize(&options, &session_));
    IREE_ASSERT_OK(
        loom_run_vm_runtime_initialize(iree_allocator_system(), &runtime_));
  }

  void TearDown() override {
    loom_run_vm_runtime_deinitialize(&runtime_);
    loom_run_session_deinitialize(&session_);
  }

  iree_status_t EmitArchive(loom_run_module_t* out_module,
                            loom_ireevm_run_candidate_t* out_candidate) {
    return EmitArchiveFromSource(IREE_SV(kPreparedVmSource), out_module,
                                 out_candidate);
  }

  iree_status_t EmitArchiveFromSource(
      iree_string_view_t source, loom_run_module_t* out_module,
      loom_ireevm_run_candidate_t* out_candidate) {
    loom_run_module_parse_options_t parse_options = {};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = IREE_SV("ireevm_invocation_test.loom");
    parse_options.source = source;
    IREE_RETURN_IF_ERROR(
        loom_run_module_parse(&session_, &parse_options, out_module));

    loom_run_candidate_compile_options_t archive_emit_options = {};
    loom_run_candidate_compile_options_initialize(&archive_emit_options);
    archive_emit_options.source_resolver =
        loom_run_module_source_resolver(out_module);
    iree_status_t status =
        loom_ireevm_run_candidate_emit(out_module, &archive_emit_options,
                                       iree_allocator_system(), out_candidate);
    if (!iree_status_is_ok(status)) {
      loom_run_module_deinitialize(out_module);
    }
    return status;
  }

  iree_status_t RunBranchy(const loom_ireevm_module_archive_t* archive,
                           iree_string_view_t lhs, iree_string_view_t rhs,
                           iree_string_view_t expected,
                           loom_run_vm_invocation_result_t* result) {
    iree_string_view_t inputs[] = {lhs, rhs};
    iree_string_view_t expected_outputs[] = {expected};

    loom_run_vm_invocation_request_t request = {};
    loom_run_vm_invocation_request_initialize(&request);
    request.runtime = &runtime_;
    request.archive = archive;
    request.options.function_name = IREE_SV("branchy");
    request.options.inputs = (loom_run_vm_value_specs_t){
        /*.values=*/inputs,
        /*.count=*/IREE_ARRAYSIZE(inputs),
    };
    request.options.expected_outputs = (loom_run_vm_value_specs_t){
        /*.values=*/expected_outputs,
        /*.count=*/IREE_ARRAYSIZE(expected_outputs),
    };
    return loom_run_vm_invocation_run(&request, iree_allocator_system(),
                                      result);
  }

  iree_status_t RunWideNumeric(const loom_ireevm_module_archive_t* archive,
                               loom_run_vm_invocation_result_t* result) {
    iree_string_view_t inputs[] = {
        IREE_SV("10"), IREE_SV("32"),   IREE_SV("2.5"),
        IREE_SV("4"),  IREE_SV("5.25"),
    };
    iree_string_view_t expected_outputs[] = {
        IREE_SV("42"),
        IREE_SV("10"),
        IREE_SV("6.75"),
        IREE_SV("1"),
    };

    loom_run_vm_invocation_request_t request = {};
    loom_run_vm_invocation_request_initialize(&request);
    request.runtime = &runtime_;
    request.archive = archive;
    request.options.function_name = IREE_SV("wide_numeric");
    request.options.inputs = (loom_run_vm_value_specs_t){
        /*.values=*/inputs,
        /*.count=*/IREE_ARRAYSIZE(inputs),
    };
    request.options.expected_outputs = (loom_run_vm_value_specs_t){
        /*.values=*/expected_outputs,
        /*.count=*/IREE_ARRAYSIZE(expected_outputs),
    };
    return loom_run_vm_invocation_run(&request, iree_allocator_system(),
                                      result);
  }

  void PrepareBranchyCandidate(
      const loom_ireevm_module_archive_t* archive,
      loom_run_vm_prepared_candidate_t* out_candidate) {
    loom_run_vm_prepared_candidate_options_t options = {};
    loom_run_vm_prepared_candidate_options_initialize(&options);
    options.function_name = IREE_SV("branchy");
    IREE_ASSERT_OK(loom_run_vm_prepared_candidate_prepare(
        &runtime_, archive, &options, iree_allocator_system(), out_candidate));
  }

  void PrepareBranchyPlan(loom_run_vm_prepared_candidate_t* candidate,
                          iree_string_view_t lhs, iree_string_view_t rhs,
                          iree_string_view_t expected,
                          loom_run_vm_invocation_plan_t* out_plan) {
    iree_string_view_t inputs[] = {lhs, rhs};
    iree_string_view_t expected_outputs[] = {expected};

    loom_run_vm_invocation_options_t options = {};
    loom_run_vm_invocation_options_initialize(&options);
    options.inputs = (loom_run_vm_value_specs_t){
        /*.values=*/inputs,
        /*.count=*/IREE_ARRAYSIZE(inputs),
    };
    options.expected_outputs = (loom_run_vm_value_specs_t){
        /*.values=*/expected_outputs,
        /*.count=*/IREE_ARRAYSIZE(expected_outputs),
    };
    IREE_ASSERT_OK(loom_run_vm_invocation_plan_prepare_from_prepared(
        candidate, &options, iree_allocator_system(), out_plan));
  }

  loom_run_session_t session_ = {};
  loom_run_vm_runtime_t runtime_ = {};
};

TEST_F(VmInvocationTest, RunMatchesExpectedOutputs) {
  loom_run_module_t module = {};
  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(EmitArchive(&module, &candidate));

  loom_run_vm_invocation_result_t result = {};
  loom_run_vm_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_ASSERT_OK(RunBranchy(&candidate.archive, IREE_SV("0"), IREE_SV("21"),
                            IREE_SV("42"), &result));
  EXPECT_EQ(result.exit_code, 0);
  std::string output(result.output.buffer, result.output.size);
  EXPECT_NE(output.find("EXEC @branchy"), std::string::npos);
  EXPECT_NE(output.find("[SUCCESS] all function outputs matched"),
            std::string::npos);

  loom_run_vm_invocation_result_deinitialize(&result);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&module);
}

TEST_F(VmInvocationTest, RunWideScalarRegisters) {
  loom_run_module_t module = {};
  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(EmitArchiveFromSource(IREE_SV(kPreparedVmWideSource), &module,
                                       &candidate));

  loom_run_vm_invocation_result_t result = {};
  loom_run_vm_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_ASSERT_OK(RunWideNumeric(&candidate.archive, &result));
  EXPECT_EQ(result.exit_code, 0);
  std::string output(result.output.buffer, result.output.size);
  EXPECT_NE(output.find("EXEC @wide_numeric"), std::string::npos);
  EXPECT_NE(output.find("[SUCCESS] all function outputs matched"),
            std::string::npos);

  loom_run_vm_invocation_result_deinitialize(&result);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&module);
}

TEST_F(VmInvocationTest, RunReportsExpectedOutputMismatch) {
  loom_run_module_t module = {};
  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(EmitArchive(&module, &candidate));

  loom_run_vm_invocation_result_t result = {};
  loom_run_vm_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_ASSERT_OK(RunBranchy(&candidate.archive, IREE_SV("50"), IREE_SV("8"),
                            IREE_SV("41"), &result));
  EXPECT_EQ(result.exit_code, 1);
  std::string output(result.output.buffer, result.output.size);
  EXPECT_NE(output.find("EXEC @branchy"), std::string::npos);
  EXPECT_NE(output.find("[FAILED]"), std::string::npos);

  loom_run_vm_invocation_result_deinitialize(&result);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&module);
}

TEST_F(VmInvocationTest, RunFormatsOutputsWhenNoExpectationIsProvided) {
  loom_run_module_t module = {};
  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(EmitArchive(&module, &candidate));

  iree_string_view_t inputs[] = {IREE_SV("50"), IREE_SV("8")};
  loom_run_vm_invocation_request_t request = {};
  loom_run_vm_invocation_request_initialize(&request);
  request.runtime = &runtime_;
  request.archive = &candidate.archive;
  request.options.function_name = IREE_SV("branchy");
  request.options.inputs = (loom_run_vm_value_specs_t){
      /*.values=*/inputs,
      /*.count=*/IREE_ARRAYSIZE(inputs),
  };

  loom_run_vm_invocation_result_t result = {};
  loom_run_vm_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_ASSERT_OK(
      loom_run_vm_invocation_run(&request, iree_allocator_system(), &result));
  EXPECT_EQ(result.exit_code, 0);
  std::string output(result.output.buffer, result.output.size);
  EXPECT_NE(output.find("EXEC @branchy"), std::string::npos);
  EXPECT_NE(output.find("result[0]"), std::string::npos);

  loom_run_vm_invocation_result_deinitialize(&result);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&module);
}

TEST_F(VmInvocationTest, PreparedCandidateRunsPlanTwice) {
  loom_run_module_t module = {};
  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(EmitArchive(&module, &candidate));

  loom_run_vm_prepared_candidate_t prepared = {};
  PrepareBranchyCandidate(&candidate.archive, &prepared);

  loom_run_vm_invocation_plan_t plan = {};
  PrepareBranchyPlan(&prepared, IREE_SV("0"), IREE_SV("21"), IREE_SV("42"),
                     &plan);

  loom_run_vm_invocation_result_t result = {};
  loom_run_vm_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_ASSERT_OK(loom_run_vm_invocation_run_prepared(
      &prepared, &plan, iree_allocator_system(), &result));
  EXPECT_EQ(result.exit_code, 0);

  IREE_ASSERT_OK(loom_run_vm_invocation_run_prepared(
      &prepared, &plan, iree_allocator_system(), &result));
  EXPECT_EQ(result.exit_code, 0);
  std::string output(result.output.buffer, result.output.size);
  EXPECT_NE(output.find("EXEC @branchy"), std::string::npos);
  EXPECT_NE(output.find("[SUCCESS] all function outputs matched"),
            std::string::npos);

  loom_run_vm_invocation_result_deinitialize(&result);
  loom_run_vm_invocation_plan_deinitialize(&plan);
  loom_run_vm_prepared_candidate_deinitialize(&prepared);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&module);
}

TEST_F(VmInvocationTest, InvokePlanCanSkipResultCollection) {
  loom_run_module_t module = {};
  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(EmitArchive(&module, &candidate));

  loom_run_vm_prepared_candidate_t prepared = {};
  PrepareBranchyCandidate(&candidate.archive, &prepared);

  loom_run_vm_invocation_plan_t plan = {};
  PrepareBranchyPlan(&prepared, IREE_SV("50"), IREE_SV("8"), IREE_SV("42"),
                     &plan);

  loom_run_vm_iteration_t iteration = {};
  IREE_ASSERT_OK(loom_run_vm_invocation_invoke_plan(
      &prepared, &plan, iree_allocator_system(), &iteration));
  EXPECT_NE(iteration.inputs, nullptr);
  EXPECT_NE(iteration.outputs, nullptr);

  loom_run_vm_iteration_deinitialize(&iteration);
  loom_run_vm_invocation_plan_deinitialize(&plan);
  loom_run_vm_prepared_candidate_deinitialize(&prepared);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&module);
}

TEST_F(VmInvocationTest, PlanPrepareRejectsFunctionMismatch) {
  loom_run_module_t module = {};
  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(EmitArchive(&module, &candidate));

  loom_run_vm_prepared_candidate_t prepared = {};
  PrepareBranchyCandidate(&candidate.archive, &prepared);

  loom_run_vm_invocation_options_t options = {};
  loom_run_vm_invocation_options_initialize(&options);
  options.function_name = IREE_SV("not_branchy");

  loom_run_vm_invocation_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_vm_invocation_plan_prepare_from_prepared(
          &prepared, &options, iree_allocator_system(), &plan));

  loom_run_vm_invocation_plan_deinitialize(&plan);
  loom_run_vm_prepared_candidate_deinitialize(&prepared);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&module);
}

TEST_F(VmInvocationTest, PlanPrepareFromSpecsMaterializesExpectedOutputs) {
  iree_string_view_t inputs[] = {IREE_SV("0"), IREE_SV("21")};
  iree_string_view_t expected_outputs[] = {IREE_SV("42")};

  loom_run_vm_invocation_options_t options = {};
  loom_run_vm_invocation_options_initialize(&options);
  options.inputs = (loom_run_vm_value_specs_t){
      /*.values=*/inputs,
      /*.count=*/IREE_ARRAYSIZE(inputs),
  };
  options.expected_outputs = (loom_run_vm_value_specs_t){
      /*.values=*/expected_outputs,
      /*.count=*/IREE_ARRAYSIZE(expected_outputs),
  };

  const loom_run_vm_invocation_plan_prepare_request_t prepare_request = {
      /*.options=*/&options,
      /*.arguments_cconv=*/IREE_SV("ii"),
      /*.results_cconv=*/IREE_SV("i"),
  };
  loom_run_vm_invocation_plan_t plan = {};

  IREE_ASSERT_OK(loom_run_vm_invocation_plan_prepare_from_specs(
      &prepare_request, iree_allocator_system(), &plan));
  EXPECT_EQ(plan.output_mode, LOOM_RUN_VM_OUTPUT_PLAN_MODE_COMPARE_EXPECTED);
  EXPECT_EQ(iree_vm_list_size(plan.inputs), IREE_ARRAYSIZE(inputs));
  EXPECT_EQ(iree_vm_list_size(plan.expected_outputs),
            IREE_ARRAYSIZE(expected_outputs));

  loom_run_vm_invocation_plan_deinitialize(&plan);
}

TEST_F(VmInvocationTest, PlanPrepareUsesOutputSpecsAsMaterializationPolicy) {
  iree_string_view_t inputs[] = {IREE_SV("50"), IREE_SV("8")};
  iree_string_view_t outputs[] = {IREE_SV("-")};

  loom_run_vm_invocation_options_t options = {};
  loom_run_vm_invocation_options_initialize(&options);
  options.inputs = (loom_run_vm_value_specs_t){
      /*.values=*/inputs,
      /*.count=*/IREE_ARRAYSIZE(inputs),
  };
  options.outputs = (loom_run_vm_value_specs_t){
      /*.values=*/outputs,
      /*.count=*/IREE_ARRAYSIZE(outputs),
  };

  const loom_run_vm_invocation_plan_prepare_request_t prepare_request = {
      /*.options=*/&options,
      /*.arguments_cconv=*/IREE_SV("ii"),
      /*.results_cconv=*/IREE_SV("i"),
  };
  loom_run_vm_invocation_plan_t plan = {};

  IREE_ASSERT_OK(loom_run_vm_invocation_plan_prepare_from_specs(
      &prepare_request, iree_allocator_system(), &plan));
  EXPECT_EQ(plan.output_mode, LOOM_RUN_VM_OUTPUT_PLAN_MODE_WRITE_SPECS);
  EXPECT_EQ(iree_vm_list_size(plan.inputs), IREE_ARRAYSIZE(inputs));
  EXPECT_EQ(plan.output_specs.count, IREE_ARRAYSIZE(outputs));
  EXPECT_EQ(plan.expected_outputs, nullptr);

  loom_run_vm_invocation_plan_deinitialize(&plan);
}

}  // namespace
}  // namespace loom
