// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cmd/hal.h"
#include "loomc/target/cmd/program.h"
#include "loomc/target/cmd/program_plan.h"
#include "loomc/target/iree_hal.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CmdHalProgramPtr =
    HandlePtr<loomc_cmd_hal_program_t, loomc_cmd_hal_program_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ExecutablePtr =
    HandlePtr<iree_hal_executable_t, iree_hal_executable_release>;
using FrontierTrackerPtr = HandlePtr<iree_async_frontier_tracker_t,
                                     iree_async_frontier_tracker_release>;
using HalBufferPtr = HandlePtr<iree_hal_buffer_t, iree_hal_buffer_release>;
using HalDevicePtr = HandlePtr<iree_hal_device_t, iree_hal_device_release>;
using HalDeviceGroupPtr =
    HandlePtr<iree_hal_device_group_t, iree_hal_device_group_release>;
using LaunchConfigProgramPtr = HandlePtr<loomc_launch_config_program_t,
                                         loomc_launch_config_program_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PackagePtr =
    HandlePtr<loomc_cmd_program_package_t, loomc_cmd_program_package_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ProgramPlanPtr =
    HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ProactorPoolPtr =
    HandlePtr<iree_async_proactor_pool_t, iree_async_proactor_pool_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

constexpr char kSourceText[] = R"(
kernel.def @add_bias(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%bias: view<1xi32>, %source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %zero = index.constant 0 : index
  %source_aligned = buffer.assume.alignment %source {minimum_alignment = 4} : buffer
  %target_aligned = buffer.assume.alignment %target {minimum_alignment = 4} : buffer
  %source_view = buffer.view %source_aligned[%base] : buffer -> view<1xi32>
  %target_view = buffer.view %target_aligned[%base] : buffer -> view<1xi32>
  %is_first = index.cmp eq, %workgroup, %zero : index
  scf.if %is_first {
    %bias_value = view.load %bias[%zero] : view<1xi32> -> i32
    %value = view.load %source_view[%zero] : view<1xi32> -> i32
    %result = scalar.addi %value, %bias_value : i32
    view.store %result, %target_view[%zero] : i32, view<1xi32>
  }
  kernel.return
}

command.program.def public @prefill(%element_count: index) launch(%parameters: buffer, %source: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  %bias = command.parameter %parameters, "bias" : view<1xi32>
  kernel.launch @add_bias[%element_count](%bias, %source, %target) : [index](view<1xi32>, buffer, buffer)
  command.return
}

command.program.def public @decode() launch(%parameters: buffer, %source: buffer, %target: buffer) {
  %one = index.constant 1 : index
  %bias = command.parameter %parameters, "bias" : view<1xi32>
  kernel.launch @add_bias[%one](%bias, %source, %target) : [index](view<1xi32>, buffer, buffer)
  command.return
}
)";

void PrintResultDiagnostics(const loomc_result_t* result) {
  if (result == nullptr) return;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic == nullptr) continue;
    std::fprintf(stderr, "%.*s: %.*s\n",
                 static_cast<int>(diagnostic->code.size), diagnostic->code.data,
                 static_cast<int>(diagnostic->message.size),
                 diagnostic->message.data);
  }
}

bool ResultSucceeded(const loomc_result_t* result) {
  if (result == nullptr || !loomc_result_succeeded(result)) {
    PrintResultDiagnostics(result);
    return false;
  }
  return true;
}

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != nullptr && artifact->kind == kind &&
        loomc_string_view_equal(artifact->format,
                                loomc_make_cstring_view(format))) {
      return artifact;
    }
  }
  return nullptr;
}

iree_status_t CreateAmdgpuDevice(ProactorPoolPtr* out_proactor_pool,
                                 FrontierTrackerPtr* out_frontier_tracker,
                                 HalDeviceGroupPtr* out_device_group,
                                 HalDevicePtr* out_device) {
  IREE_RETURN_IF_ERROR(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));
  out_proactor_pool->reset(proactor_pool);

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_create_device(
      iree_hal_driver_registry_default(), IREE_SV("amdgpu"), &create_params,
      iree_allocator_system(), &device));
  out_device->reset(device);

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));
  out_frontier_tracker->reset(frontier_tracker);
  iree_hal_device_group_t* device_group = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &device_group));
  out_device_group->reset(device_group);
  return iree_ok_status();
}

iree_status_t LoadExecutable(iree_hal_device_t* device,
                             const loomc_artifact_t* artifact,
                             iree_hal_executable_t** out_executable) {
  const iree_hal_executable_target_selection_t selection = {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  const iree_hal_executable_target_selection_result_t selected =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(device), &selection);
  if (selected.outcome !=
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU HAL target selection failed");
  }
  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.executable_data = iree_make_const_byte_span(
      artifact->contents.data, artifact->contents.data_length);
  return iree_hal_device_load_executable(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                         selected.target, &params,
                                         out_executable);
}

iree_status_t AllocateBuffer(iree_hal_device_t* device,
                             iree_hal_memory_type_t memory_type,
                             iree_hal_memory_access_t access,
                             iree_hal_buffer_usage_t usage,
                             iree_device_size_t byte_length,
                             iree_hal_buffer_t** out_buffer) {
  const iree_hal_buffer_params_t params = {
      /*.usage=*/usage,
      /*.access=*/access,
      /*.type=*/memory_type,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

iree_status_t SubmitAndWait(iree_hal_device_t* device,
                            iree_hal_command_buffer_t* command_buffer,
                            iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signals = {
      /*.count=*/1,
      /*.semaphores=*/&semaphore,
      /*.payload_values=*/&signal_value,
  };
  iree_status_t status = iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signals, command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

TEST(CommandProgramExecutionTest,
     MaterializesAfterCompilerProductsAreDestroyed) {
  ProactorPoolPtr proactor_pool;
  FrontierTrackerPtr frontier_tracker;
  HalDeviceGroupPtr device_group;
  HalDevicePtr device;
  iree_status_t device_status = CreateAmdgpuDevice(
      &proactor_pool, &frontier_tracker, &device_group, &device);
  if (!iree_status_is_ok(device_status) && device.get() == nullptr) {
    const iree_status_code_t code = iree_status_code(device_status);
    if (code == IREE_STATUS_NOT_FOUND || code == IREE_STATUS_UNAVAILABLE ||
        code == IREE_STATUS_FAILED_PRECONDITION) {
      iree_status_free(device_status);
      GTEST_SKIP() << "no live AMDGPU HAL device is available";
    }
  }
  IREE_ASSERT_OK(device_status);

  loomc_target_environment_t* target_environment_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &target_environment_handle));
  TargetEnvironmentPtr target_environment(target_environment_handle);
  const loomc_context_target_options_t target_context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_context_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment.get(),
  };
  const loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_context_options,
  };
  loomc_context_t* context_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_context_create(
      &context_options, loomc_allocator_system(), &context_handle));
  ContextPtr context(context_handle);

  loomc_workspace_t* workspace_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(
      /*options=*/nullptr, loomc_allocator_system(), &workspace_handle));
  WorkspacePtr workspace(workspace_handle);
  const loomc_source_options_t source_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(source_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/
      loomc_make_cstring_view("command_program_execution_test.loom"),
      /*.contents=*/
      loomc_make_byte_span(kSourceText, std::strlen(kSourceText)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_source_create(&source_options, loomc_allocator_system(),
                                      &source_handle));
  SourcePtr source(source_handle);
  loomc_module_t* module_handle = nullptr;
  loomc_result_t* result_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context.get(), workspace.get(), source.get(), /*options=*/nullptr,
      loomc_allocator_system(), &module_handle, &result_handle));
  ModulePtr module(module_handle);
  ResultPtr result(result_handle);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  const loomc_iree_hal_profile_provider_t* profile_providers[] = {
      loomc_amdgpu_iree_hal_profile_provider(),
  };
  const loomc_iree_hal_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("live-amdgpu-command"),
      /*.device=*/device.get(),
      /*.physical_device_affinity=*/0,
      /*.providers=*/profile_providers,
      /*.provider_count=*/IREE_ARRAYSIZE(profile_providers),
  };
  loomc_target_profile_t* target_profile_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &profile_options, loomc_allocator_system(),
      &target_profile_handle, &result_handle));
  TargetProfilePtr target_profile(target_profile_handle);
  result.reset(result_handle);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  loomc_compiler_t* compiler_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(context.get(), /*options=*/nullptr,
                                        loomc_allocator_system(),
                                        &compiler_handle));
  CompilerPtr compiler(compiler_handle);
  loomc_pass_program_t* empty_pass_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
      context.get(), /*options=*/nullptr, loomc_allocator_system(),
      &empty_pass_handle));
  PassProgramPtr empty_pass(empty_pass_handle);

  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("add_bias"),
      /*.target_profile=*/target_profile.get(),
  };
  const loomc_target_specialization_options_t specialization_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(specialization_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
  };
  const loomc_compile_options_t specialize_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(specialize_options),
      /*.next=*/&specialization_options,
  };
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), empty_pass.get(), module.get(),
      &specialize_options, loomc_allocator_system(), &result_handle));
  result.reset(result_handle);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("prefill"),
      loomc_make_cstring_view("decode"),
  };
  loomc_program_plan_t* plan_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_prepare(
      workspace.get(), module.get(), roots, IREE_ARRAYSIZE(roots),
      /*options=*/nullptr, loomc_allocator_system(), &plan_handle,
      &result_handle));
  ProgramPlanPtr plan(plan_handle);
  result.reset(result_handle);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  const loomc_program_plan_root_t prefill_root =
      loomc_program_plan_root_at(plan.get(), 0);
  loomc_cmd_program_plan_root_info_t prefill_root_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_plan_root_info(plan.get(), prefill_root,
                                                   &prefill_root_info));
  ASSERT_TRUE(loomc_program_plan_unit_is_valid(prefill_root_info.package_unit));
  ASSERT_TRUE(
      loomc_program_plan_unit_is_valid(prefill_root_info.launch_config_unit));
  ASSERT_EQ(prefill_root_info.executable_requirement_count, 1u);
  const loomc_program_plan_root_t decode_root =
      loomc_program_plan_root_at(plan.get(), 1);
  loomc_cmd_program_plan_root_info_t decode_root_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_plan_root_info(plan.get(), decode_root,
                                                   &decode_root_info));
  EXPECT_EQ(decode_root_info.package_unit.value,
            prefill_root_info.package_unit.value);
  EXPECT_FALSE(
      loomc_program_plan_unit_is_valid(decode_root_info.launch_config_unit));
  ASSERT_EQ(decode_root_info.executable_requirement_count, 1u);
  EXPECT_EQ(decode_root_info.executable_requirements[0].unit.value,
            prefill_root_info.executable_requirements[0].unit.value);

  const loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("command-program-execution"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* target_pass_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
      context.get(), &pipeline_options, loomc_allocator_system(),
      &target_pass_handle, &result_handle));
  PassProgramPtr target_pass(target_pass_handle);
  result.reset(result_handle);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  loomc_result_t* package_result_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan.get(), compiler.get(), workspace.get(),
      prefill_root_info.package_unit, empty_pass.get(), /*options=*/nullptr,
      loomc_allocator_system(), &package_result_handle));
  ResultPtr package_result(package_result_handle);
  ASSERT_TRUE(ResultSucceeded(package_result.get()));
  const loomc_artifact_t* package_artifact =
      FindArtifact(package_result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE);
  ASSERT_NE(package_artifact, nullptr);
  loomc_cmd_program_package_t* package_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_package_load(
      package_artifact, /*release=*/nullptr, /*release_user_data=*/nullptr,
      loomc_allocator_system(), &package_handle));
  PackagePtr package(package_handle);

  loomc_result_t* launch_result_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan.get(), compiler.get(), workspace.get(),
      prefill_root_info.launch_config_unit, empty_pass.get(),
      /*options=*/nullptr, loomc_allocator_system(), &launch_result_handle));
  ResultPtr launch_result(launch_result_handle);
  ASSERT_TRUE(ResultSucceeded(launch_result.get()));
  const loomc_artifact_t* launch_artifact = FindArtifact(
      launch_result.get(), LOOMC_ARTIFACT_KIND_COMMAND_LAUNCH_CONFIG,
      LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
  ASSERT_NE(launch_artifact, nullptr);
  loomc_launch_config_program_t* launch_program_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_program_load(
      launch_artifact, /*release=*/nullptr, /*release_user_data=*/nullptr,
      loomc_allocator_system(), &launch_program_handle));
  LaunchConfigProgramPtr launch_program(launch_program_handle);

  loomc_result_t* executable_result_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan.get(), compiler.get(), workspace.get(),
      prefill_root_info.executable_requirements[0].unit, target_pass.get(),
      /*options=*/nullptr, loomc_allocator_system(),
      &executable_result_handle));
  ResultPtr executable_result(executable_result_handle);
  ASSERT_TRUE(ResultSucceeded(executable_result.get()));
  const loomc_artifact_t* executable_artifact =
      FindArtifact(executable_result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  ASSERT_NE(executable_artifact, nullptr);
  iree_hal_executable_t* executable_handle = nullptr;
  IREE_ASSERT_OK(
      LoadExecutable(device.get(), executable_artifact, &executable_handle));
  ExecutablePtr executable(executable_handle);

  loomc_cmd_program_export_t prefill_export =
      loomc_cmd_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_cmd_program_package_lookup_export(
      package.get(), loomc_make_cstring_view("prefill"), &prefill_export));
  loomc_cmd_program_info_t prefill_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_package_export_info(
      package.get(), prefill_export, &prefill_info));
  ASSERT_EQ(prefill_info.fixed_buffer_count, 1u);
  ASSERT_EQ(prefill_info.executable_count, 1u);
  ASSERT_EQ(prefill_info.rebindable_binding_count, 3u);
  ASSERT_EQ(prefill_info.parameter_root_count, 1u);
  ASSERT_EQ(prefill_info.parameter_count, 1u);
  ASSERT_NE(prefill_info.config.binding_index,
            LOOMC_CMD_PROGRAM_BINDING_INVALID);
  ASSERT_LE(prefill_info.config.required_byte_length,
            LOOMC_CMD_HAL_CONFIG_ALIGNMENT);
  loomc_cmd_program_export_t decode_export = loomc_cmd_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_cmd_program_package_lookup_export(
      package.get(), loomc_make_cstring_view("decode"), &decode_export));
  loomc_cmd_program_info_t decode_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_package_export_info(
      package.get(), decode_export, &decode_info));
  ASSERT_EQ(decode_info.fixed_buffer_count, 1u);
  ASSERT_EQ(decode_info.rebindable_binding_count, 2u);
  ASSERT_EQ(decode_info.config.binding_index,
            LOOMC_CMD_PROGRAM_BINDING_INVALID);

  loomc_cmd_program_parameter_root_info_t parameter_root_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_package_parameter_root_info(
      package.get(), prefill_export, 0, &parameter_root_info));
  ASSERT_EQ(parameter_root_info.fixed_buffer_index, 0u);
  ASSERT_GE(parameter_root_info.required_byte_length, sizeof(uint32_t));
  loomc_cmd_program_parameter_info_t parameter_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_package_parameter_info(
      package.get(), prefill_export, 0, &parameter_info));
  EXPECT_TRUE(loomc_string_view_equal(parameter_info.key,
                                      loomc_make_cstring_view("bias")));
  ASSERT_EQ(parameter_info.fixed_buffer_index, 0u);
  ASSERT_EQ(parameter_info.byte_length, sizeof(uint32_t));
  ASSERT_LE(parameter_info.byte_length,
            parameter_root_info.required_byte_length);
  ASSERT_LE(
      parameter_info.byte_offset,
      parameter_root_info.required_byte_length - parameter_info.byte_length);

  // The loaded package, launch program, and executable are now the only
  // surviving products. No later operation may reach compiler-owned state.
  package_result.reset();
  launch_result.reset();
  executable_result.reset();
  plan.reset();
  module.reset();
  source.reset();
  target_pass.reset();
  empty_pass.reset();
  compiler.reset();
  target_profile.reset();
  workspace.reset();
  context.reset();
  target_environment.reset();

  iree_hal_buffer_t* parameter_buffer_handle = nullptr;
  IREE_ASSERT_OK(AllocateBuffer(
      device.get(), IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      parameter_root_info.required_byte_length, &parameter_buffer_handle));
  HalBufferPtr parameter_buffer(parameter_buffer_handle);
  const uint32_t bias_value = 7;
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), &bias_value, parameter_buffer.get(),
      parameter_info.byte_offset, sizeof(bias_value),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  const iree_hal_buffer_ref_t fixed_buffers[] = {
      iree_hal_make_buffer_ref(parameter_buffer.get(), 0,
                               IREE_HAL_WHOLE_BUFFER),
  };
  iree_hal_executable_t* executables[] = {executable.get()};
  loomc_cmd_hal_program_options_t one_shot_materialization_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_HAL_PROGRAM_OPTIONS,
      /*.structure_size=*/sizeof(one_shot_materialization_options),
      /*.next=*/nullptr,
      /*.command_buffer_mode=*/IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.fixed_buffers=*/fixed_buffers,
      /*.fixed_buffer_count=*/IREE_ARRAYSIZE(fixed_buffers),
      /*.executables=*/executables,
      /*.executable_count=*/IREE_ARRAYSIZE(executables),
  };
  loomc_cmd_hal_program_t* one_shot_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_cmd_hal_program_create(
          package.get(), prefill_export, device.get(),
          &one_shot_materialization_options, loomc_allocator_system(),
          &one_shot_program));
  EXPECT_EQ(one_shot_program, nullptr);

  const loomc_cmd_hal_program_options_t materialization_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_HAL_PROGRAM_OPTIONS,
      /*.structure_size=*/sizeof(materialization_options),
      /*.next=*/nullptr,
      /*.command_buffer_mode=*/IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.fixed_buffers=*/fixed_buffers,
      /*.fixed_buffer_count=*/IREE_ARRAYSIZE(fixed_buffers),
      /*.executables=*/executables,
      /*.executable_count=*/IREE_ARRAYSIZE(executables),
  };
  loomc_cmd_hal_program_t* prefill_program_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_hal_program_create(
      package.get(), prefill_export, device.get(), &materialization_options,
      loomc_allocator_system(), &prefill_program_handle));
  CmdHalProgramPtr prefill_program(prefill_program_handle);
  ASSERT_EQ(loomc_cmd_hal_program_binding_count(prefill_program.get()), 3u);
  ASSERT_NE(loomc_cmd_hal_program_command_buffer(prefill_program.get()),
            nullptr);
  loomc_cmd_hal_program_t* decode_program_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_hal_program_create(
      package.get(), decode_export, device.get(), &materialization_options,
      loomc_allocator_system(), &decode_program_handle));
  CmdHalProgramPtr decode_program(decode_program_handle);
  ASSERT_EQ(loomc_cmd_hal_program_binding_count(decode_program.get()), 2u);

  // Default HAL recording retained every fixed resource. Prove that issue no
  // longer depends on the package bytes or the application's executable ref.
  package.reset();
  executable.reset();
  parameter_buffer.reset();

  loomc_launch_config_function_t launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
      launch_program.get(), loomc_make_cstring_view("prefill"),
      &launch_function));
  alignas(LOOMC_CMD_HAL_CONFIG_ALIGNMENT)
      std::array<uint8_t, LOOMC_CMD_HAL_CONFIG_ALIGNMENT>
          config_data = {};
  loomc_cmd_launch_config_t launch_config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(launch_config),
      /*.next=*/nullptr,
      /*.data=*/
      loomc_make_mutable_byte_span(config_data.data(),
                                   prefill_info.config.required_byte_length),
  };
  constexpr uint64_t kWorkload = 73;
  LOOMC_ASSERT_OK(loomc_launch_config_program_invoke_cmd(
      launch_program.get(), launch_function, &kWorkload,
      /*workload_argument_count=*/1, &launch_config));
  launch_program.reset();

  constexpr iree_device_size_t kDataByteLength = sizeof(uint32_t);
  iree_hal_buffer_t* source_buffer_handle = nullptr;
  IREE_ASSERT_OK(AllocateBuffer(
      device.get(), IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      kDataByteLength, &source_buffer_handle));
  HalBufferPtr source_buffer(source_buffer_handle);
  iree_hal_buffer_t* target_buffer_handle = nullptr;
  IREE_ASSERT_OK(AllocateBuffer(
      device.get(), IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      kDataByteLength, &target_buffer_handle));
  HalBufferPtr target_buffer(target_buffer_handle);
  iree_hal_buffer_t* config_buffer_handle = nullptr;
  IREE_ASSERT_OK(
      AllocateBuffer(device.get(), IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                     IREE_HAL_MEMORY_ACCESS_ALL,
                     IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                         IREE_HAL_BUFFER_USAGE_TRANSFER,
                     LOOMC_CMD_HAL_CONFIG_ALIGNMENT, &config_buffer_handle));
  HalBufferPtr config_buffer(config_buffer_handle);

  const std::array<uint32_t, 1> source_values = {1000u};
  const std::array<uint32_t, 1> zero_values = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), source_values.data(), source_buffer.get(), 0,
      kDataByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), zero_values.data(), target_buffer.get(), 0, kDataByteLength,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), config_data.data(), config_buffer.get(), 0,
      prefill_info.config.required_byte_length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  std::array<iree_hal_buffer_binding_t, 3> bindings = {{
      {source_buffer.get(), 0, IREE_HAL_WHOLE_BUFFER},
      {target_buffer.get(), 0, IREE_HAL_WHOLE_BUFFER},
      {config_buffer.get(), 0, prefill_info.config.required_byte_length},
  }};
  ASSERT_EQ(prefill_info.config.binding_index, 2u);
  IREE_ASSERT_OK(SubmitAndWait(
      device.get(), loomc_cmd_hal_program_command_buffer(prefill_program.get()),
      {/*.count=*/bindings.size(), /*.bindings=*/bindings.data()}));

  std::array<uint32_t, 1> actual_values = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device.get(), target_buffer.get(), 0, actual_values.data(),
      kDataByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  EXPECT_EQ(actual_values[0], 1007u);

  const std::array<uint32_t, 1> replay_source_values = {2000u};
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), replay_source_values.data(), source_buffer.get(), 0,
      kDataByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), zero_values.data(), target_buffer.get(), 0, kDataByteLength,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  IREE_ASSERT_OK(SubmitAndWait(
      device.get(), loomc_cmd_hal_program_command_buffer(prefill_program.get()),
      {/*.count=*/bindings.size(), /*.bindings=*/bindings.data()}));
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device.get(), target_buffer.get(), 0, actual_values.data(),
      kDataByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  EXPECT_EQ(actual_values[0], 2007u);

  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), zero_values.data(), target_buffer.get(), 0, kDataByteLength,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  IREE_ASSERT_OK(SubmitAndWait(
      device.get(), loomc_cmd_hal_program_command_buffer(decode_program.get()),
      {/*.count=*/2, /*.bindings=*/bindings.data()}));
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device.get(), target_buffer.get(), 0, actual_values.data(),
      kDataByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  EXPECT_EQ(actual_values[0], 2007u);
}

}  // namespace
