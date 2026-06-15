// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/invocation.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace loom {
namespace {

iree_vm_instance_t* hal_invocation_test_vm_instance = nullptr;

class HalInvocationTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                           iree_allocator_system(),
                                           &hal_invocation_test_vm_instance));
  }

  static void TearDownTestSuite() {
    iree_vm_instance_release(hal_invocation_test_vm_instance);
    hal_invocation_test_vm_instance = nullptr;
  }
};

TEST_F(HalInvocationTest, RequestInitializeDefaultsToSingleWorkgroup) {
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);

  EXPECT_TRUE(iree_string_view_is_empty(request.options.function_name));
  EXPECT_EQ(request.options.workgroup_count[0], 1u);
  EXPECT_EQ(request.options.workgroup_count[1], 1u);
  EXPECT_EQ(request.options.workgroup_count[2], 1u);
  EXPECT_EQ(request.options.constant_count, 0u);
}

TEST_F(HalInvocationTest, DispatchBatchOptionsUseFastReusableDefaults) {
  loom_run_hal_dispatch_batch_options_t options = {};
  loom_run_hal_dispatch_batch_options_initialize(&options);

  EXPECT_EQ(options.dispatch_count, 1u);
  EXPECT_TRUE(iree_all_bits_set(options.command_buffer_mode,
                                IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED));
  EXPECT_TRUE(iree_all_bits_set(options.command_buffer_mode,
                                IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED));
  EXPECT_FALSE(iree_all_bits_set(options.command_buffer_mode,
                                 IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT));
  EXPECT_TRUE(
      iree_all_bits_set(options.execute_flags,
                        IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME));
}

TEST_F(HalInvocationTest, ResultOwnsOutputBuilder) {
  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_ASSERT_OK(iree_string_builder_append_cstring(&result.output, "hello"));
  EXPECT_TRUE(iree_string_view_equal(iree_string_builder_view(&result.output),
                                     IREE_SV("hello")));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST_F(HalInvocationTest, RunRejectsTooManyBindingsBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_artifact_t executable = {};
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);
  request.runtime = &runtime;
  request.artifact = &executable;
  request.bindings.count = LOOM_RUN_HAL_MAX_BINDING_COUNT + 1;

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_invocation_run(&request, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST_F(HalInvocationTest, RunRejectsMissingBindingStorageBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_artifact_t executable = {};
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);
  request.runtime = &runtime;
  request.artifact = &executable;
  request.bindings.count = 1;

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_run(&request, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST_F(HalInvocationTest,
       RunRejectsExpectedBindingCountMismatchBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_artifact_t executable = {};
  iree_string_view_t bindings[] = {IREE_SV("&4xi32")};
  const char binding_conventions[] = {'r'};
  iree_string_view_t expected_bindings[] = {IREE_SV("4xi32=0"),
                                            IREE_SV("i32=1")};
  const char expected_binding_conventions[] = {'r', 'r'};

  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);
  request.runtime = &runtime;
  request.artifact = &executable;
  request.bindings = (loom_run_hal_binding_specs_t){
      /*.values=*/bindings,
      /*.conventions=*/binding_conventions,
      /*.count=*/IREE_ARRAYSIZE(bindings),
  };
  request.expected_bindings = (loom_run_hal_binding_specs_t){
      /*.values=*/expected_bindings,
      /*.conventions=*/expected_binding_conventions,
      /*.count=*/IREE_ARRAYSIZE(expected_bindings),
  };

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_run(&request, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST_F(HalInvocationTest, RunRequiresInitializedRuntime) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_artifact_t executable = {};
  loom_run_hal_invocation_request_t request = {};
  loom_run_hal_invocation_request_initialize(&request);
  request.runtime = &runtime;
  request.artifact = &executable;

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_run(&request, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
}

TEST_F(HalInvocationTest,
       RunPlanRejectsExpectedBindingCountMismatchBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_artifact_t executable = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 2, iree_allocator_system(),
                                     &plan.bindings));
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 2, iree_allocator_system(),
                                     &plan.expected_bindings));
  iree_vm_value_t value = iree_vm_value_make_i32(0);
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.bindings, &value));
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.expected_bindings, &value));
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.expected_bindings, &value));

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_run_plan(&runtime, &executable, &plan,
                                       iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest, PreparedCandidatePrepareRequiresInitializedRuntime) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_artifact_t executable = {};
  loom_run_hal_prepared_candidate_t candidate = {};

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_hal_prepared_candidate_prepare(
                            &runtime, &executable, &candidate));

  loom_run_hal_prepared_candidate_deinitialize(&candidate);
}

TEST_F(HalInvocationTest, DispatchPlanRejectsMissingPreparedExecutable) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_prepared_candidate_t candidate = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &plan.bindings));

  loom_run_hal_iteration_t iteration = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_dispatch_plan(
          &runtime, &candidate, &plan, iree_allocator_system(), &iteration));

  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest, PreparePlanFromListsRetainsBindings) {
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);

  iree_vm_list_t* bindings = nullptr;
  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(
      iree_vm_list_create(value_type, 1, iree_allocator_system(), &bindings));
  iree_vm_value_t value = iree_vm_value_make_i32(0);
  IREE_ASSERT_OK(iree_vm_list_push_value(bindings, &value));

  loom_run_hal_invocation_plan_t plan = {};
  IREE_ASSERT_OK(loom_run_hal_invocation_plan_prepare_from_lists(
      &options, bindings, /*expected_bindings=*/nullptr,
      /*max_output_element_count=*/0, &plan));
  iree_vm_list_release(bindings);

  EXPECT_EQ(iree_vm_list_size(plan.bindings), 1u);

  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest, PreparePlanFromListsRejectsTooManyBindings) {
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);

  iree_vm_list_t* bindings = nullptr;
  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(
      iree_vm_list_create(value_type, 0, iree_allocator_system(), &bindings));
  iree_vm_value_t value = iree_vm_value_make_i32(0);
  for (iree_host_size_t i = 0; i <= LOOM_RUN_HAL_MAX_BINDING_COUNT; ++i) {
    IREE_ASSERT_OK(iree_vm_list_push_value(bindings, &value));
  }

  loom_run_hal_invocation_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_run_hal_invocation_plan_prepare_from_lists(
                            &options, bindings, /*expected_bindings=*/nullptr,
                            /*max_output_element_count=*/0, &plan));

  loom_run_hal_invocation_plan_deinitialize(&plan);
  iree_vm_list_release(bindings);
}

TEST_F(HalInvocationTest, DispatchPlanRejectsTooManyConstantsBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_prepared_candidate_t candidate = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);
  plan.options.constant_count = LOOM_RUN_HAL_MAX_CONSTANT_COUNT + 1;

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &plan.bindings));

  loom_run_hal_iteration_t iteration = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_invocation_dispatch_plan(
          &runtime, &candidate, &plan, iree_allocator_system(), &iteration));

  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest,
       DispatchBatchRejectsZeroDispatchCountBeforeDeviceUse) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_prepared_candidate_t candidate = {
      /*.target_bundle=*/{},
      /*.executable=*/reinterpret_cast<iree_hal_executable_t*>(1),
  };
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &plan.bindings));

  loom_run_hal_dispatch_batch_options_t options = {};
  loom_run_hal_dispatch_batch_options_initialize(&options);
  options.dispatch_count = 0;

  loom_run_hal_dispatch_batch_t batch = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_dispatch_batch_prepare(&runtime, &candidate, &plan, &options,
                                          iree_allocator_system(), &batch));

  candidate.executable = nullptr;
  loom_run_hal_dispatch_batch_deinitialize(&batch);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest,
       DispatchPlanRejectsTargetLimitViolationBeforeDeviceUse) {
  static const loom_target_snapshot_t snapshot = {
      /*.name=*/IREE_SVL("test-snapshot"),
      /*.codegen_format=*/{},
      /*.artifact_format=*/{},
      /*.default_pointer_bitwidth=*/{},
      /*.index_bitwidth=*/{},
      /*.offset_bitwidth=*/{},
      /*.max_workgroup_size=*/{},
      /*.max_flat_workgroup_size=*/{},
      /*.max_workgroup_storage_bytes=*/{},
      /*.subgroup_size=*/{},
      /*.max_grid_size=*/{},
      /*.max_flat_grid_size=*/{},
      /*.max_workgroup_count=*/{/*.x=*/4, /*.y=*/4, /*.z=*/4},
  };
  static const loom_target_export_plan_t export_plan = {
      /*.name=*/IREE_SVL("test-export"),
      /*.export_symbol=*/{},
      /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
      /*.linkage=*/{},
      /*.hal_kernel=*/
      {
          /*.required_workgroup_size=*/{/*.x=*/0, /*.y=*/0, /*.z=*/0},
      },
  };
  static const loom_target_bundle_t target_bundle = {
      /*.name=*/IREE_SVL("test-bundle"),
      /*.snapshot=*/&snapshot,
      /*.export_plan=*/&export_plan,
  };

  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_prepared_candidate_t candidate = {
      /*.target_bundle=*/&target_bundle,
      /*.executable=*/reinterpret_cast<iree_hal_executable_t*>(1),
  };
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);
  plan.options.workgroup_count[0] = 5;

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &plan.bindings));

  loom_run_hal_iteration_t iteration = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_run_hal_invocation_dispatch_plan(
          &runtime, &candidate, &plan, iree_allocator_system(), &iteration));

  candidate.executable = nullptr;
  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest, CollectResultsRejectsMissingIterationBindings) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &plan.bindings));

  loom_run_hal_iteration_t iteration = {};
  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_collect_results(
          &runtime, &plan, &iteration, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

TEST_F(HalInvocationTest, CollectResultsRejectsIterationPlanBindingMismatch) {
  loom_run_hal_runtime_t runtime = {};
  loom_run_hal_invocation_plan_t plan = {};
  loom_run_hal_invocation_plan_initialize(&plan);
  loom_run_hal_iteration_t iteration = {};
  loom_run_hal_iteration_initialize(&iteration);

  const iree_vm_type_def_t value_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 1, iree_allocator_system(),
                                     &plan.bindings));
  IREE_ASSERT_OK(iree_vm_list_create(value_type, 0, iree_allocator_system(),
                                     &iteration.bindings));
  iree_vm_value_t value = iree_vm_value_make_i32(0);
  IREE_ASSERT_OK(iree_vm_list_push_value(plan.bindings, &value));

  loom_run_hal_invocation_result_t result = {};
  loom_run_hal_invocation_result_initialize(iree_allocator_system(), &result);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_invocation_collect_results(
          &runtime, &plan, &iteration, iree_allocator_system(), &result));

  loom_run_hal_invocation_result_deinitialize(&result);
  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
}

}  // namespace
}  // namespace loom
