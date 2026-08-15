// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/tooling/command_program.h"

#include <string.h>

#include "iree/io/parameter_provider.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cmd/hal.h"
#include "loomc/target/cmd/program_plan.h"

typedef struct qwen_tooling_parameter_request_t {
  // Concrete GGUF tensor key borrowed from the loaded command package.
  iree_string_view_t key;
  // Source and packed target range for the tensor payload.
  iree_io_parameter_span_t span;
} qwen_tooling_parameter_request_t;

struct qwen_tooling_command_program_t {
  // Allocator used to release this object.
  iree_allocator_t host_allocator;
  // Package retained so strings borrowed by info remain valid.
  loomc_cmd_program_package_t* package;
  // Materialized reusable command buffer and retained fixed resources.
  loomc_cmd_hal_program_t* hal_program;
  // Immutable root ABI borrowed from package.
  loomc_cmd_program_info_t info;
  // Packed fixed-parameter root size in bytes.
  iree_device_size_t parameter_byte_length;
};

static iree_status_t qwen_tooling_status_from_loomc(loomc_status_t status) {
  return iree_status_from_loomc(status);
}

static iree_status_t qwen_tooling_require_result(iree_string_view_t phase,
                                                 const loomc_result_t* result) {
  if (result && loomc_result_succeeded(result)) return iree_ok_status();
  iree_string_view_t message = IREE_SV("Loom operation failed");
  if (result && loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    if (diagnostic) message = iree_string_view_from_loomc(diagnostic->message);
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%.*s: %.*s",
                          (int)phase.size, phase.data, (int)message.size,
                          message.data);
}

static const loomc_artifact_t* qwen_tooling_find_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact && artifact->kind == kind &&
        loomc_string_view_equal(artifact->format,
                                loomc_make_cstring_view(format))) {
      return artifact;
    }
  }
  return NULL;
}

static iree_status_t qwen_tooling_parameter_enumerate(
    void* user_data, iree_host_size_t index, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const qwen_tooling_parameter_request_t* requests =
      (const qwen_tooling_parameter_request_t*)user_data;
  *out_key = requests[index].key;
  *out_span = requests[index].span;
  return iree_ok_status();
}

static iree_status_t qwen_tooling_allocate_buffer(
    iree_hal_device_t* device, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t usage, iree_device_size_t minimum_alignment,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  const iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = minimum_alignment,
  };
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t qwen_tooling_load_executable(
    iree_hal_device_t* device, const loomc_artifact_t* artifact,
    iree_hal_executable_t** out_executable) {
  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .target_key = iree_string_view_empty(),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      .physical_device_affinity = 0,
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

static iree_status_t qwen_tooling_query_kernels(
    loomc_module_t* module, iree_allocator_t host_allocator,
    loomc_module_function_t** out_functions,
    iree_host_size_t* out_function_count) {
  *out_functions = NULL;
  *out_function_count = 0;
  const loomc_module_function_query_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      .structure_size = sizeof(options),
      .kind = LOOMC_MODULE_FUNCTION_KIND_KERNEL,
  };

  loomc_result_t* result = NULL;
  iree_status_t status =
      qwen_tooling_status_from_loomc(loomc_module_query_functions(
          module, &options, loomc_allocator_from_iree(host_allocator), 0, NULL,
          out_function_count, &result));
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status) && *out_function_count == 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Qwen command source contains no kernels");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, *out_function_count,
                                         sizeof(**out_functions),
                                         (void**)out_functions);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_module_query_functions(
        module, &options, loomc_allocator_from_iree(host_allocator),
        *out_function_count, *out_functions, out_function_count, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, *out_functions);
    *out_functions = NULL;
    *out_function_count = 0;
  }
  return status;
}

static iree_status_t qwen_tooling_compile_unit(
    const loomc_program_plan_t* plan, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, loomc_program_plan_unit_t unit,
    const loomc_pass_program_t* pass_program, iree_string_view_t phase,
    iree_allocator_t host_allocator, loomc_result_t** out_result) {
  *out_result = NULL;
  iree_status_t status =
      qwen_tooling_status_from_loomc(loomc_program_plan_compile_unit(
          plan, compiler, workspace, unit, pass_program, /*options=*/NULL,
          loomc_allocator_from_iree(host_allocator), out_result));
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_require_result(phase, *out_result);
  }
  if (!iree_status_is_ok(status)) {
    loomc_result_release(*out_result);
    *out_result = NULL;
  }
  return status;
}

static iree_status_t qwen_tooling_begin_parameter_gather(
    loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export,
    const loomc_cmd_program_info_t* program_info,
    iree_io_parameter_provider_t* parameter_provider, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_parameter_buffer,
    qwen_tooling_parameter_request_t** out_requests,
    iree_hal_semaphore_t** out_ready_semaphore) {
  *out_parameter_buffer = NULL;
  *out_requests = NULL;
  *out_ready_semaphore = NULL;
  if (program_info->fixed_buffer_count != 1 ||
      program_info->parameter_root_count != 1 ||
      program_info->parameter_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen command programs require one nonempty parameter root");
  }

  loomc_cmd_program_parameter_root_info_t root_info = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      .structure_size = sizeof(root_info),
  };
  IREE_RETURN_IF_ERROR(qwen_tooling_status_from_loomc(
      loomc_cmd_program_package_parameter_root_info(package, program_export, 0,
                                                    &root_info)));
  if (root_info.fixed_buffer_index != 0 ||
      root_info.required_byte_length == 0 || root_info.minimum_alignment == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "the Qwen parameter root has an invalid ABI");
  }

  iree_status_t status = qwen_tooling_allocate_buffer(
      device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      root_info.minimum_alignment, root_info.required_byte_length,
      out_parameter_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, program_info->parameter_count, sizeof(**out_requests),
        (void**)out_requests);
  }
  for (iree_host_size_t i = 0;
       i < program_info->parameter_count && iree_status_is_ok(status); ++i) {
    loomc_cmd_program_parameter_info_t parameter_info = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
        .structure_size = sizeof(parameter_info),
    };
    status =
        qwen_tooling_status_from_loomc(loomc_cmd_program_package_parameter_info(
            package, program_export, i, &parameter_info));
    if (!iree_status_is_ok(status)) break;
    if (parameter_info.fixed_buffer_index != root_info.fixed_buffer_index ||
        parameter_info.byte_length == 0 ||
        parameter_info.byte_offset > root_info.required_byte_length ||
        parameter_info.byte_length >
            root_info.required_byte_length - parameter_info.byte_offset ||
        parameter_info.minimum_alignment == 0 ||
        parameter_info.byte_offset % parameter_info.minimum_alignment != 0) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled parameter '%.*s' has an invalid packed range",
          (int)parameter_info.key.size, parameter_info.key.data);
      break;
    }
    (*out_requests)[i] = (qwen_tooling_parameter_request_t){
        .key = iree_string_view_from_loomc(parameter_info.key),
        .span =
            {
                .parameter_offset = 0,
                .buffer_offset = parameter_info.byte_offset,
                .length = parameter_info.byte_length,
            },
    };
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_ready_semaphore);
  }
  uint64_t ready_value = 1;
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_t* ready_semaphore = *out_ready_semaphore;
    const iree_hal_semaphore_list_t signals = {
        .count = 1,
        .semaphores = &ready_semaphore,
        .payload_values = &ready_value,
    };
    status = iree_io_parameter_provider_gather(
        parameter_provider, device, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), signals, iree_string_view_empty(),
        *out_parameter_buffer, program_info->parameter_count,
        (iree_io_parameter_enumerator_t){
            .fn = qwen_tooling_parameter_enumerate,
            .user_data = *out_requests,
        });
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_release(*out_ready_semaphore);
    iree_allocator_free(host_allocator, *out_requests);
    iree_hal_buffer_release(*out_parameter_buffer);
    *out_ready_semaphore = NULL;
    *out_requests = NULL;
    *out_parameter_buffer = NULL;
  }
  return status;
}

iree_status_t qwen_tooling_command_program_create(
    qwen_tooling_runtime_context_t* runtime_context,
    const qwen_tooling_command_program_options_t* options,
    iree_allocator_t host_allocator,
    qwen_tooling_command_program_t** out_program) {
  if (!runtime_context || !options || !out_program ||
      iree_string_view_is_empty(options->source_identifier) ||
      iree_string_view_is_empty(options->root_name) ||
      iree_const_byte_span_is_empty(options->source_contents)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program options must be complete");
  }
  *out_program = NULL;
  const loomc_allocator_t loom_allocator =
      loomc_allocator_from_iree(host_allocator);
  iree_hal_device_t* device =
      qwen_tooling_runtime_context_device(runtime_context);

  loomc_target_environment_t* target_environment = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* workspace = NULL;
  loomc_source_t* source = NULL;
  loomc_module_t* module = NULL;
  loomc_target_profile_t* target_profile = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_pass_program_t* empty_pass_program = NULL;
  loomc_pass_program_t* executable_pass_program = NULL;
  loomc_module_function_t* kernel_functions = NULL;
  loomc_target_specialization_t* specializations = NULL;
  iree_host_size_t kernel_function_count = 0;
  loomc_program_plan_t* plan = NULL;
  loomc_cmd_program_package_t* package = NULL;
  loomc_cmd_program_export_t program_export =
      loomc_cmd_program_export_invalid();
  loomc_result_t* result = NULL;
  iree_hal_buffer_t* parameter_buffer = NULL;
  qwen_tooling_parameter_request_t* parameter_requests = NULL;
  iree_hal_semaphore_t* parameter_ready_semaphore = NULL;
  bool parameter_gather_submitted = false;
  iree_hal_executable_t** executables = NULL;
  iree_host_size_t executable_count = 0;
  qwen_tooling_command_program_t* program = NULL;

  iree_status_t status =
      qwen_tooling_status_from_loomc(loomc_target_environment_create_amdgpu(
          loom_allocator, &target_environment));
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .target_environment = target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_context_create(&context_options, loom_allocator, &context));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_workspace_create(
        /*options=*/NULL, loom_allocator, &workspace));
  }
  if (iree_status_is_ok(status)) {
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_string_view_from_iree(options->source_identifier),
        .contents = loomc_make_byte_span(options->source_contents.data,
                                         options->source_contents.data_length),
        .storage = LOOMC_SOURCE_STORAGE_BORROWED,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_source_create(&source_options, loom_allocator, &source));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_status_from_loomc(loomc_module_deserialize_from_source(
            context, workspace, source,
            /*options=*/NULL, loom_allocator, &module, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_require_result(IREE_SV("command source parse"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    const loomc_amdgpu_iree_hal_profile_options_t profile_options = {
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_PROFILE_OPTIONS,
        .structure_size = sizeof(profile_options),
        .identifier = loomc_make_cstring_view("qwen38-live-amdgpu"),
        .device = device,
        .physical_device_affinity = 0,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_target_profile_create_amdgpu_iree_hal(
            target_environment, &profile_options, loom_allocator,
            &target_profile, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_require_result(IREE_SV("AMDGPU target profile"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_query_kernels(
        module, host_allocator, &kernel_functions, &kernel_function_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, kernel_function_count,
                                         sizeof(*specializations),
                                         (void**)&specializations);
  }
  for (iree_host_size_t i = 0;
       i < kernel_function_count && iree_status_is_ok(status); ++i) {
    specializations[i] = (loomc_target_specialization_t){
        .function_symbol = kernel_functions[i].symbol_name,
        .target_profile = target_profile,
    };
  }

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_compiler_create(
        context, /*options=*/NULL, loom_allocator, &compiler));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_pass_program_create_empty(
        context, /*options=*/NULL, loom_allocator, &empty_pass_program));
  }
  if (iree_status_is_ok(status)) {
    const loomc_target_pipeline_options_t pipeline_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(pipeline_options),
        .identifier = loomc_make_cstring_view("qwen38-prepared-low"),
        .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        .source_to_low_max_errors = 20,
    };
    status = qwen_tooling_status_from_loomc(
        loomc_pass_program_create_from_target_pipeline(
            context, &pipeline_options, loom_allocator,
            &executable_pass_program, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_require_result(IREE_SV("AMDGPU pipeline"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    const loomc_target_specialization_options_t specialization_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(specialization_options),
        .specializations = specializations,
        .specialization_count = kernel_function_count,
    };
    const loomc_compile_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        .structure_size = sizeof(compile_options),
        .next = &specialization_options,
        .module_name = loomc_string_view_from_iree(options->root_name),
    };
    status = qwen_tooling_status_from_loomc(
        loomc_compile_module(compiler, workspace, empty_pass_program, module,
                             &compile_options, loom_allocator, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_require_result(IREE_SV("target specialization"), result);
  }
  loomc_result_release(result);
  result = NULL;

  const loomc_string_view_t root_name =
      loomc_string_view_from_iree(options->root_name);
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_status_from_loomc(loomc_program_plan_prepare(
        workspace, module, &root_name, 1, /*options=*/NULL, loom_allocator,
        &plan, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_require_result(IREE_SV("command program plan"), result);
  }
  loomc_result_release(result);
  result = NULL;

  loomc_program_plan_root_t root = loomc_program_plan_root_invalid();
  loomc_cmd_program_plan_root_info_t root_info = {0};
  if (iree_status_is_ok(status)) {
    root = loomc_program_plan_root_at(plan, 0);
    status = qwen_tooling_status_from_loomc(
        loomc_cmd_program_plan_root_info(plan, root, &root_info));
  }
  if (iree_status_is_ok(status) &&
      loomc_program_plan_unit_is_valid(root_info.launch_config_unit)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen command programs do not yet accept dynamic launch config");
  }

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_compile_unit(
        plan, compiler, workspace, root_info.package_unit, empty_pass_program,
        IREE_SV("command package compile"), host_allocator, &result);
  }
  if (iree_status_is_ok(status)) {
    const loomc_artifact_t* artifact =
        qwen_tooling_find_artifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                                   LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE);
    if (!artifact) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "command package artifact is absent");
    } else {
      status = qwen_tooling_status_from_loomc(loomc_cmd_program_package_load(
          artifact, /*release=*/NULL, /*release_user_data=*/NULL,
          loom_allocator, &package));
    }
  }
  loomc_result_release(result);
  result = NULL;

  loomc_cmd_program_info_t program_info = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
      .structure_size = sizeof(program_info),
  };
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_status_from_loomc(loomc_cmd_program_package_lookup_export(
            package, root_name, &program_export));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_status_from_loomc(loomc_cmd_program_package_export_info(
            package, program_export, &program_info));
  }

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_begin_parameter_gather(
        package, program_export, &program_info,
        runtime_context->parameter_provider, device, host_allocator,
        &parameter_buffer, &parameter_requests, &parameter_ready_semaphore);
    parameter_gather_submitted = iree_status_is_ok(status);
  }

  executable_count = root_info.executable_requirement_count;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, executable_count,
                                    sizeof(*executables), (void**)&executables);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < executable_count; ++i) {
      executables[i] = NULL;
    }
  }
  for (iree_host_size_t i = 0;
       i < executable_count && iree_status_is_ok(status); ++i) {
    const loomc_program_plan_unit_t unit =
        root_info.executable_requirements[i].unit;
    if (!loomc_program_plan_unit_is_valid(unit)) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Qwen command programs do not yet accept external executables");
      break;
    }
    status = qwen_tooling_compile_unit(
        plan, compiler, workspace, unit, executable_pass_program,
        IREE_SV("AMDGPU executable compile"), host_allocator, &result);
    if (iree_status_is_ok(status)) {
      const loomc_artifact_t* artifact =
          qwen_tooling_find_artifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                                     LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
      if (!artifact) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "AMDGPU executable artifact is absent");
      } else {
        status =
            qwen_tooling_load_executable(device, artifact, &executables[i]);
      }
    }
    loomc_result_release(result);
    result = NULL;
  }

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*program),
                                   (void**)&program);
  }
  if (iree_status_is_ok(status)) {
    memset(program, 0, sizeof(*program));
    program->host_allocator = host_allocator;
    const iree_hal_buffer_ref_t fixed_buffer =
        iree_hal_make_buffer_ref(parameter_buffer, 0, IREE_HAL_WHOLE_BUFFER);
    const loomc_cmd_hal_program_options_t materialization_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_HAL_PROGRAM_OPTIONS,
        .structure_size = sizeof(materialization_options),
        .command_buffer_mode = runtime_context->command_buffer_mode,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
        .fixed_buffers = &fixed_buffer,
        .fixed_buffer_count = 1,
        .executables = executables,
        .executable_count = executable_count,
    };
    status = qwen_tooling_status_from_loomc(loomc_cmd_hal_program_create(
        package, program_export, device, &materialization_options,
        loom_allocator, &program->hal_program));
  }

  if (parameter_gather_submitted) {
    const iree_status_t gather_status = iree_hal_semaphore_wait(
        parameter_ready_semaphore, 1, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
    status = iree_status_join(status, gather_status);
  }
  if (iree_status_is_ok(status)) {
    loomc_cmd_program_parameter_root_info_t parameter_root_info = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
        .structure_size = sizeof(parameter_root_info),
    };
    status = qwen_tooling_status_from_loomc(
        loomc_cmd_program_package_parameter_root_info(package, program_export,
                                                      0, &parameter_root_info));
    if (iree_status_is_ok(status)) {
      program->package = package;
      package = NULL;
      program->info = program_info;
      program->parameter_byte_length = parameter_root_info.required_byte_length;
      *out_program = program;
      program = NULL;
    }
  }

  qwen_tooling_command_program_release(program);
  for (iree_host_size_t i = 0; i < executable_count; ++i) {
    iree_hal_executable_release(executables ? executables[i] : NULL);
  }
  iree_allocator_free(host_allocator, executables);
  iree_hal_semaphore_release(parameter_ready_semaphore);
  iree_allocator_free(host_allocator, parameter_requests);
  iree_hal_buffer_release(parameter_buffer);
  loomc_result_release(result);
  loomc_cmd_program_package_release(package);
  loomc_program_plan_release(plan);
  iree_allocator_free(host_allocator, specializations);
  iree_allocator_free(host_allocator, kernel_functions);
  loomc_pass_program_release(executable_pass_program);
  loomc_pass_program_release(empty_pass_program);
  loomc_compiler_release(compiler);
  loomc_target_profile_release(target_profile);
  loomc_module_release(module);
  loomc_source_release(source);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(target_environment);
  return status;
}

void qwen_tooling_command_program_release(
    qwen_tooling_command_program_t* program) {
  if (!program) return;
  const iree_allocator_t host_allocator = program->host_allocator;
  loomc_cmd_hal_program_release(program->hal_program);
  loomc_cmd_program_package_release(program->package);
  iree_allocator_free(host_allocator, program);
}

const loomc_cmd_program_info_t* qwen_tooling_command_program_info(
    const qwen_tooling_command_program_t* program) {
  return program ? &program->info : NULL;
}

iree_hal_command_buffer_t* qwen_tooling_command_program_command_buffer(
    const qwen_tooling_command_program_t* program) {
  return program ? loomc_cmd_hal_program_command_buffer(program->hal_program)
                 : NULL;
}

iree_device_size_t qwen_tooling_command_program_parameter_byte_length(
    const qwen_tooling_command_program_t* program) {
  return program ? program->parameter_byte_length : 0;
}
