// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test/target/iree_hal_execution.h"

#include <array>
#include <cstdint>
#include <cstdio>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/init.h"
#include "iree/testing/gtest.h"
#include "loomc/target/kernel.h"
#include "loomc/target/vm.h"
#include "test/util.h"

namespace loomc::testing::target {
namespace {

using loomc::testing::HandlePtr;

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
using LaunchConfigProgramPtr =
    HandlePtr<loomc_vm_launch_config_program_t,
              loomc_vm_launch_config_program_release>;
using LinkIndexBuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
using ProactorPoolPtr =
    HandlePtr<iree_async_proactor_pool_t, iree_async_proactor_pool_release>;
using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

void PrintIreeStatus(const char* label, iree_status_t status) {
  iree_status_code_t code = iree_status_code(status);
  iree_string_view_t message = iree_status_message(status);
  if (iree_string_view_is_empty(message)) {
    std::fprintf(stderr, "%s: %s\n", label, iree_status_code_string(code));
    return;
  }
  std::fprintf(stderr, "%s: %s: %.*s\n", label, iree_status_code_string(code),
               static_cast<int>(message.size), message.data);
}

void PrintResultDiagnostics(const loomc_result_t* result) {
  if (result == nullptr) {
    std::fprintf(stderr, "loom result is null\n");
    return;
  }
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic == nullptr) {
      continue;
    }
    std::fprintf(stderr, "%.*s: %.*s\n",
                 static_cast<int>(diagnostic->code.size), diagnostic->code.data,
                 static_cast<int>(diagnostic->message.size),
                 diagnostic->message.data);
  }
}

bool ResultSucceeded(const loomc_result_t* result, const char* label) {
  if (result == nullptr) {
    std::fprintf(stderr, "%s failed without a result\n", label);
    return false;
  }
  if (!loomc_result_succeeded(result)) {
    PrintResultDiagnostics(result);
    return false;
  }
  return true;
}

bool IsLiveSkipStatus(iree_status_code_t code) {
  switch (code) {
    case IREE_STATUS_NOT_FOUND:
    case IREE_STATUS_FAILED_PRECONDITION:
    case IREE_STATUS_UNAVAILABLE:
    case IREE_STATUS_INCOMPATIBLE:
      return true;
    default:
      return false;
  }
}

iree_status_t CreateLiveHalDevice(iree_string_view_t device_uri,
                                  ProactorPoolPtr* out_proactor_pool,
                                  FrontierTrackerPtr* out_frontier_tracker,
                                  HalDeviceGroupPtr* out_device_group,
                                  HalDevicePtr* out_device) {
  iree_allocator_t host_allocator = iree_allocator_system();
  IREE_RETURN_IF_ERROR(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  iree_status_t status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), host_allocator,
      &proactor_pool);
  if (iree_status_is_ok(status)) {
    out_proactor_pool->reset(proactor_pool);
  }

  iree_hal_device_t* device = nullptr;
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = out_proactor_pool->get();
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_create_device(iree_hal_driver_registry_default(), device_uri,
                               &create_params, host_allocator, &device);
  }
  if (iree_status_is_ok(status)) {
    out_device->reset(device);
  }

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_options_t options =
        iree_async_frontier_tracker_options_default();
    status = iree_async_frontier_tracker_create(options, host_allocator,
                                                &frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    out_frontier_tracker->reset(frontier_tracker);
  }

  iree_hal_device_group_t* device_group = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        out_device->get(), out_frontier_tracker->get(), host_allocator,
        &device_group);
  }
  if (iree_status_is_ok(status)) {
    out_device_group->reset(device_group);
  }
  return status;
}

loomc_status_t CreateTargetContext(const IreeHalKernelExecutionTarget& target,
                                   TargetEnvironmentPtr* out_target_environment,
                                   ContextPtr* out_context) {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = target.create_target_environment(
      loomc_allocator_system(), &target_environment);
  if (loomc_status_is_ok(status)) {
    out_target_environment->reset(target_environment);
  }

  loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/out_target_environment->get(),
  };
  loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  if (loomc_status_is_ok(status)) {
    status = loomc_context_create(&context_options, loomc_allocator_system(),
                                  &context);
  }
  if (loomc_status_is_ok(status)) {
    out_context->reset(context);
  }
  return status;
}

loomc_status_t CreateSource(const IreeHalKernelExecutionTarget& target,
                            SourcePtr* out_source) {
  loomc_byte_span_t source_contents =
      loomc_make_byte_span(target.source_text.data, target.source_text.size);
  loomc_source_options_t source_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(source_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/target.source_identifier,
      /*.contents=*/source_contents,
      /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&source_options, loomc_allocator_system(), &source);
  if (loomc_status_is_ok(status)) {
    out_source->reset(source);
  }
  return status;
}

loomc_status_t CreateHalTargetProfile(
    const IreeHalKernelExecutionTarget& target,
    loomc_target_environment_t* target_environment, iree_hal_device_t* device,
    TargetProfilePtr* out_profile, ResultPtr* out_result) {
  loomc_iree_hal_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/target.target_profile_identifier,
      /*.device=*/device,
      /*.physical_device_affinity=*/
      target.executable_target_selection.physical_device_affinity,
      /*.providers=*/target.profile_providers,
      /*.provider_count=*/target.profile_provider_count,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_iree_hal(
      target_environment, &profile_options, loomc_allocator_system(), &profile,
      &result);
  if (loomc_status_is_ok(status)) {
    out_profile->reset(profile);
    out_result->reset(result);
  }
  return status;
}

loomc_status_t CreateTargetPipeline(const IreeHalKernelExecutionTarget& target,
                                    loomc_context_t* context,
                                    PassProgramPtr* out_pass_program,
                                    ResultPtr* out_result) {
  loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/nullptr,
      /*.identifier=*/target.target_pipeline_identifier,
      /*.kind=*/target.target_pipeline_kind,
      /*.control_flow_lowering=*/target.control_flow_lowering,
      /*.source_to_low_max_errors=*/target.source_to_low_max_errors,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_target_pipeline(
      context, &pipeline_options, loomc_allocator_system(), &pass_program,
      &result);
  if (loomc_status_is_ok(status)) {
    out_pass_program->reset(pass_program);
    out_result->reset(result);
  }
  return status;
}

loomc_status_t ConvertTextSourceToBytecode(loomc_context_t* context,
                                           loomc_workspace_t* workspace,
                                           const loomc_source_t* text_source,
                                           SourcePtr* out_bytecode_source,
                                           ResultPtr* out_result) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_text_from_source(
      context, workspace, text_source, /*options=*/nullptr,
      loomc_allocator_system(), &module, &result);
  ModulePtr module_ptr(module);
  if (loomc_status_is_ok(status)) {
    out_result->reset(result);
  }
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    return status;
  }

  const loomc_module_serialize_options_t serialize_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(serialize_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
      /*.identifier=*/loomc_make_cstring_view("iree_hal_execution.loombc"),
  };
  loomc_source_t* bytecode_source = nullptr;
  status = loomc_module_serialize_bytecode_to_source(
      module_ptr.get(), &serialize_options, loomc_allocator_system(),
      &bytecode_source);
  if (loomc_status_is_ok(status)) {
    out_bytecode_source->reset(bytecode_source);
  }
  return status;
}

loomc_status_t CreateKernelRequest(loomc_context_t* context,
                                   loomc_source_t* source,
                                   loomc_string_view_t kernel_export_name,
                                   RequestPtr* out_request,
                                   ResultPtr* out_result) {
  loomc_link_index_builder_t* index_builder = nullptr;
  loomc_status_t status = loomc_link_index_builder_create(
      context, /*options=*/nullptr, loomc_allocator_system(), &index_builder);
  LinkIndexBuilderPtr index_builder_ptr(index_builder);
  const loomc_link_index_source_options_t source_options = {
      /*.provider_name=*/loomc_make_cstring_view("iree-hal-execution"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_add_source(
        index_builder_ptr.get(), source, &source_options, /*out_slot=*/nullptr);
  }

  loomc_link_index_t* link_index = nullptr;
  loomc_result_t* result = nullptr;
  if (loomc_status_is_ok(status)) {
    status = loomc_link_index_builder_finish(index_builder_ptr.get(),
                                             &link_index, &result);
  }
  LinkIndexPtr link_index_ptr(link_index);
  if (loomc_status_is_ok(status)) {
    out_result->reset(result);
  }
  if (!loomc_status_is_ok(status) || !loomc_result_succeeded(result)) {
    return status;
  }

  loomc_link_index_module_t module = {};
  if (loomc_link_index_module_count(link_index_ptr.get()) != 1 ||
      !loomc_link_index_module_at(link_index_ptr.get(), 0, &module)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "test source must contain exactly one module");
  }
  loomc_link_index_symbol_t symbol = {};
  if (!loomc_link_index_lookup_private(link_index_ptr.get(), &module,
                                       kernel_export_name, &symbol)) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "kernel export was not found in the test module");
  }
  if (symbol.provider_module_ordinal > UINT32_MAX ||
      symbol.module_symbol_ordinal > UINT32_MAX) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "kernel root exceeds the request ordinal domain");
  }

  const loomc_request_root_t root = {
      /*.module_ordinal=*/
      static_cast<uint32_t>(symbol.provider_module_ordinal),
      /*.symbol_ordinal=*/static_cast<uint32_t>(symbol.module_symbol_ordinal),
      /*.goal=*/LOOMC_KERNEL_ROOT_GOAL_HOST_LAUNCHABLE,
      /*.reserved=*/0,
  };
  loomc_request_t* request = nullptr;
  status = loomc_request_create(loomc_kernel_product_descriptor(), source,
                                &root, /*root_count=*/1,
                                /*bindings=*/nullptr, /*binding_count=*/0,
                                loomc_allocator_system(), &request);
  if (loomc_status_is_ok(status)) {
    out_request->reset(request);
  }
  return status;
}

loomc_status_t BuildKernelProduct(
    const IreeHalKernelExecutionTarget& target, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, loomc_pass_program_t* pass_program,
    loomc_target_profile_t* target_profile, const loomc_request_t* request,
    ProductPtr* out_product, ResultPtr* out_result) {
  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/target.kernel_export_name,
      /*.target_profile=*/target_profile,
  };
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
  };
  loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
      /*.module_name=*/target.module_name,
  };
  loomc_product_t* product = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_kernel_product_build_request(
      compiler, workspace, pass_program, request, &compile_options,
      target.emit_options, loomc_allocator_system(), &product, &result);
  if (loomc_status_is_ok(status)) {
    out_product->reset(product);
    out_result->reset(result);
  }
  return status;
}

iree_status_t AllocateStorageBuffer(iree_hal_device_t* device,
                                    iree_device_size_t buffer_size,
                                    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, buffer_size, out_buffer);
}

iree_status_t PrepareExecutableFromArtifact(
    const IreeHalKernelExecutionTarget& target, iree_hal_device_t* device,
    const loomc_artifact_t* artifact, iree_hal_executable_t** out_executable) {
  *out_executable = nullptr;
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(device), &target.executable_target_selection);
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "HAL device does not advertise the test target");
  } else if (target_result.outcome ==
             IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL device advertises ambiguous test targets");
  }

  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  loomc_byte_span_t executable_data = loomc_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_byte_sequence_clone(
      artifact->contents, loomc_allocator_system(), &executable_data)));
  load_params.executable_data = iree_make_const_byte_span(
      executable_data.data, executable_data.data_length);
  iree_status_t status = iree_hal_device_load_executable(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, target_result.target, &load_params,
      out_executable);
  loomc_allocator_free(loomc_allocator_system(), (void*)executable_data.data);
  return status;
}

iree_status_t DispatchAndWait(iree_hal_device_t* device,
                              iree_hal_executable_t* executable,
                              iree_hal_executable_function_t function,
                              iree_hal_buffer_t* input_buffer,
                              iree_hal_buffer_t* output_buffer,
                              loomc_dimension3_t workgroup_count) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  iree_hal_semaphore_t* semaphore = nullptr;
  const uint64_t constants[] = {4};
  uint64_t signal_value = 1;

  iree_status_t status = iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_t binding_refs[2] = {
        iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                                 iree_hal_buffer_byte_length(input_buffer)),
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 iree_hal_buffer_byte_length(output_buffer)),
    };
    iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/2,
        /*.values=*/binding_refs,
    };
    iree_hal_dispatch_config_t dispatch_config =
        iree_hal_make_static_dispatch_config(
            workgroup_count.x, workgroup_count.y, workgroup_count.z);
    status = iree_hal_command_buffer_dispatch(
        command_buffer, executable, function, dispatch_config,
        iree_make_const_byte_span(constants, sizeof(constants)), bindings,
        IREE_HAL_DISPATCH_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t signal_semaphores = {
        /*.count=*/1,
        /*.semaphores=*/&semaphore,
        /*.payload_values=*/&signal_value,
    };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_semaphores, command_buffer,
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  return status;
}

}  // namespace

void RunIreeHalKernelExecutionTest(const IreeHalKernelExecutionTarget& target) {
  ASSERT_NE(target.label, nullptr);
  ASSERT_FALSE(iree_string_view_is_empty(target.device_uri));
  ASSERT_FALSE(loomc_string_view_is_empty(target.target_profile_identifier));
  ASSERT_FALSE(loomc_string_view_is_empty(target.source_identifier));
  ASSERT_FALSE(loomc_string_view_is_empty(target.source_text));
  ASSERT_FALSE(loomc_string_view_is_empty(target.module_name));
  ASSERT_FALSE(loomc_string_view_is_empty(target.kernel_export_name));
  ASSERT_FALSE(loomc_string_view_is_empty(target.target_pipeline_identifier));
  ASSERT_NE(target.emit_options, nullptr);
  ASSERT_FALSE(
      iree_string_view_is_empty(target.executable_target_selection.family));
  ASSERT_NE(target.profile_providers, nullptr);
  ASSERT_GT(target.profile_provider_count, 0u);
  ASSERT_NE(target.create_target_environment, nullptr);
  ASSERT_NE(target.validate_target_profile, nullptr);

  ProactorPoolPtr proactor_pool;
  FrontierTrackerPtr frontier_tracker;
  HalDeviceGroupPtr device_group;
  HalDevicePtr device;
  iree_status_t iree_status =
      CreateLiveHalDevice(target.device_uri, &proactor_pool, &frontier_tracker,
                          &device_group, &device);
  if (!iree_status_is_ok(iree_status) && device.get() == nullptr &&
      IsLiveSkipStatus(iree_status_code(iree_status))) {
    PrintIreeStatus("IREE HAL device creation failed", iree_status);
    iree_status_free(iree_status);
    GTEST_SKIP() << "no live IREE HAL device is available for " << target.label;
  }
  IREE_ASSERT_OK(iree_status);

  TargetEnvironmentPtr target_environment;
  ContextPtr context;
  LOOMC_ASSERT_OK(CreateTargetContext(target, &target_environment, &context));

  WorkspacePtr workspace;
  loomc_workspace_t* workspace_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &workspace_handle));
  workspace.reset(workspace_handle);

  SourcePtr text_source;
  LOOMC_ASSERT_OK(CreateSource(target, &text_source));

  TargetProfilePtr target_profile;
  ResultPtr result;
  LOOMC_ASSERT_OK(CreateHalTargetProfile(target, target_environment.get(),
                                         device.get(), &target_profile,
                                         &result));
  if (!ResultSucceeded(result.get(), "IREE HAL target profile creation")) {
    GTEST_SKIP() << target.label << " did not produce a usable target profile";
  }
  ASSERT_NE(target_profile.get(), nullptr);
  result.reset();

  const char* profile_skip_reason = nullptr;
  LOOMC_ASSERT_OK(target.validate_target_profile(target_profile.get(),
                                                 &profile_skip_reason));
  if (profile_skip_reason != nullptr) {
    std::fprintf(stderr, "%s\n", profile_skip_reason);
    GTEST_SKIP() << profile_skip_reason;
  }

  SourcePtr bytecode_source;
  LOOMC_ASSERT_OK(ConvertTextSourceToBytecode(context.get(), workspace.get(),
                                              text_source.get(),
                                              &bytecode_source, &result));
  ASSERT_TRUE(ResultSucceeded(result.get(), "source deserialization"));
  ASSERT_NE(bytecode_source.get(), nullptr);
  result.reset();

  RequestPtr request;
  LOOMC_ASSERT_OK(CreateKernelRequest(context.get(), bytecode_source.get(),
                                      target.kernel_export_name, &request,
                                      &result));
  ASSERT_TRUE(ResultSucceeded(result.get(), "source indexing"));
  ASSERT_NE(request.get(), nullptr);
  result.reset();

  CompilerPtr compiler;
  loomc_compiler_t* compiler_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &compiler_handle));
  compiler.reset(compiler_handle);

  PassProgramPtr pass_program;
  LOOMC_ASSERT_OK(
      CreateTargetPipeline(target, context.get(), &pass_program, &result));
  ASSERT_TRUE(ResultSucceeded(result.get(), "target pipeline creation"));
  result.reset();

  ProductPtr product;
  LOOMC_ASSERT_OK(BuildKernelProduct(target, compiler.get(), workspace.get(),
                                     pass_program.get(), target_profile.get(),
                                     request.get(), &product, &result));
  ASSERT_TRUE(ResultSucceeded(result.get(), "kernel product build"));
  ASSERT_NE(product.get(), nullptr);
  ASSERT_EQ(loomc_product_export_count(product.get()), 1u);

  loomc_kernel_product_root_t root = {};
  ASSERT_TRUE(loomc_kernel_product_root_at(product.get(), 0, &root));
  ASSERT_NE(root.launch_config_artifact_ordinal,
            LOOMC_KERNEL_ARTIFACT_ORDINAL_INVALID);
  ASSERT_NE(root.launch_config_function_ordinal,
            LOOMC_KERNEL_FUNCTION_ORDINAL_INVALID);
  const loomc_artifact_t* launch_config_artifact = loomc_product_artifact_at(
      product.get(), root.launch_config_artifact_ordinal);
  ASSERT_NE(launch_config_artifact, nullptr);
  ASSERT_EQ(launch_config_artifact->kind, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG);
  ASSERT_TRUE(loomc_string_view_equal(
      launch_config_artifact->format,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM_BYTECODE)));
  loomc_vm_launch_config_program_t* launch_program = nullptr;
  LOOMC_ASSERT_OK(loomc_vm_launch_config_program_load(
      launch_config_artifact, loomc_allocator_system(), &launch_program));
  LaunchConfigProgramPtr launch_program_ptr(launch_program);

  loomc_vm_launch_config_function_t launch_function =
      loomc_vm_launch_config_function_invalid();
  ASSERT_TRUE(loomc_vm_launch_config_program_function_at(
      launch_program_ptr.get(), root.launch_config_function_ordinal,
      &launch_function));
  loomc_launch_config_t launch_config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(launch_config),
  };
  LOOMC_ASSERT_OK(loomc_vm_launch_config_program_invoke(
      launch_program_ptr.get(), launch_function,
      /*workload_argument_bits=*/nullptr,
      /*workload_argument_count=*/0, &launch_config));
  EXPECT_EQ(launch_config.workgroup_size.x, 1u);
  EXPECT_EQ(launch_config.workgroup_size.y, 1u);
  EXPECT_EQ(launch_config.workgroup_size.z, 1u);
  loomc_dimension3_t workgroup_count = launch_config.workgroup_count;
  const loomc_artifact_t* artifact = loomc_product_artifact_at(
      product.get(), root.executable_artifact_ordinal);
  ASSERT_NE(artifact, nullptr);
  ASSERT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  ASSERT_TRUE(loomc_string_view_equal(artifact->format,
                                      target.emit_options->artifact_format));
  ASSERT_GE(loomc_byte_sequence_length(artifact->contents), sizeof(uint32_t));

  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(PrepareExecutableFromArtifact(target, device.get(), artifact,
                                               &executable));
  ExecutablePtr executable_ptr(executable);

  std::array<int32_t, 2> input = {7, 10};
  std::array<int32_t, 2> output = {0, 0};
  iree_hal_buffer_t* input_buffer = nullptr;
  IREE_ASSERT_OK(
      AllocateStorageBuffer(device.get(), sizeof(input), &input_buffer));
  HalBufferPtr input_buffer_ptr(input_buffer);
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      AllocateStorageBuffer(device.get(), sizeof(output), &output_buffer));
  HalBufferPtr output_buffer_ptr(output_buffer);

  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), input.data(), input_buffer_ptr.get(), /*target_offset=*/0,
      sizeof(input), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), output.data(), output_buffer_ptr.get(), /*target_offset=*/0,
      sizeof(output), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  IREE_ASSERT_OK(DispatchAndWait(
      device.get(), executable_ptr.get(),
      iree_hal_executable_function_from_index(root.executable_function_ordinal),
      input_buffer_ptr.get(), output_buffer_ptr.get(), workgroup_count));

  output = {0, 0};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device.get(), output_buffer_ptr.get(), /*source_offset=*/0, output.data(),
      sizeof(output), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  EXPECT_EQ(output[0], 0);
  EXPECT_EQ(output[1], 20);
}

}  // namespace loomc::testing::target
