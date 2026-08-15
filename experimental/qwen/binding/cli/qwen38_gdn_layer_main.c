// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/qwen/programs/qwen38_gdn_layer_decode_source.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_provider.h"
#include "iree/tooling/device_util.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cmd/hal.h"
#include "loomc/target/cmd/program.h"
#include "loomc/target/cmd/program_plan.h"

#define QWEN38_HIDDEN_ELEMENT_COUNT 5120
#define QWEN38_HIDDEN_BYTE_LENGTH 20480
#define QWEN38_GDN_STATE_ELEMENT_COUNT 817152
#define QWEN38_GDN_STATE_BYTE_LENGTH 3268608

IREE_FLAG(string, output, "", "Optional raw F32 residual output path.");
IREE_FLAG(string, state_output, "", "Optional raw F32 GDN state output path.");
IREE_FLAG(int32_t, transition_count, 1,
          "Number of consecutive layer transitions to execute.");

static const char* const qwen38_gdn_layer_usage =
    "Runs the exact first Qwen3.8 GDN layer through a reusable Loom command "
    "program.\n"
    "\n"
    "Required flags:\n"
    "  --device=<AMDGPU device URI>\n"
    "  --parameters=<Qwen3.8-27B UD-Q5_K_XL GGUF path>\n"
    "\n"
    "Optional output:\n"
    "  --output=<raw F32 residual path>\n"
    "  --state_output=<raw F32 recurrent state and convolution history path>\n"
    "  --transition_count=<positive replay count>\n";

typedef struct qwen38_parameter_request_t {
  // Concrete GGUF tensor key borrowed from the loaded command package.
  iree_string_view_t key;
  // Source and packed target range for the tensor payload.
  iree_io_parameter_span_t span;
} qwen38_parameter_request_t;

typedef struct qwen38_layer_program_t {
  // Materialized reusable command buffer and retained fixed resources.
  loomc_cmd_hal_program_t* hal_program;
  // Immutable root ABI copied from the loaded package.
  loomc_cmd_program_info_t info;
  // Packed fixed-parameter root size in bytes.
  iree_device_size_t parameter_byte_length;
} qwen38_layer_program_t;

static iree_status_t qwen38_status_from_loomc(loomc_status_t status) {
  return iree_status_from_loomc(status);
}

static iree_status_t qwen38_require_result(iree_string_view_t phase,
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

static const loomc_artifact_t* qwen38_find_artifact(
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

static iree_status_t qwen38_parameter_enumerate(
    void* user_data, iree_host_size_t index, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const qwen38_parameter_request_t* requests =
      (const qwen38_parameter_request_t*)user_data;
  *out_key = requests[index].key;
  *out_span = requests[index].span;
  return iree_ok_status();
}

static iree_status_t qwen38_allocate_buffer(
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

static iree_status_t qwen38_load_executable(
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

static iree_status_t qwen38_query_kernels(
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
  iree_status_t status = qwen38_status_from_loomc(loomc_module_query_functions(
      module, &options, loomc_allocator_from_iree(host_allocator), 0, NULL,
      out_function_count, &result));
  if (iree_status_is_ok(status)) {
    status = qwen38_require_result(IREE_SV("kernel query"), result);
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
    status = qwen38_status_from_loomc(loomc_module_query_functions(
        module, &options, loomc_allocator_from_iree(host_allocator),
        *out_function_count, *out_functions, out_function_count, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, *out_functions);
    *out_functions = NULL;
    *out_function_count = 0;
  }
  return status;
}

static iree_status_t qwen38_compile_unit(
    const loomc_program_plan_t* plan, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, loomc_program_plan_unit_t unit,
    const loomc_pass_program_t* pass_program, iree_string_view_t phase,
    iree_allocator_t host_allocator, loomc_result_t** out_result) {
  *out_result = NULL;
  iree_status_t status =
      qwen38_status_from_loomc(loomc_program_plan_compile_unit(
          plan, compiler, workspace, unit, pass_program, /*options=*/NULL,
          loomc_allocator_from_iree(host_allocator), out_result));
  if (iree_status_is_ok(status)) {
    status = qwen38_require_result(phase, *out_result);
  }
  if (!iree_status_is_ok(status)) {
    loomc_result_release(*out_result);
    *out_result = NULL;
  }
  return status;
}

static iree_status_t qwen38_begin_parameter_gather(
    loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export,
    const loomc_cmd_program_info_t* program_info,
    iree_io_parameter_provider_t* parameter_provider, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_parameter_buffer,
    qwen38_parameter_request_t** out_requests,
    iree_hal_semaphore_t** out_ready_semaphore) {
  *out_parameter_buffer = NULL;
  *out_requests = NULL;
  *out_ready_semaphore = NULL;
  if (program_info->fixed_buffer_count != 1 ||
      program_info->parameter_root_count != 1 ||
      program_info->parameter_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "the first Qwen3.8 layer must publish one nonempty parameter root");
  }

  loomc_cmd_program_parameter_root_info_t root_info = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      .structure_size = sizeof(root_info),
  };
  IREE_RETURN_IF_ERROR(
      qwen38_status_from_loomc(loomc_cmd_program_package_parameter_root_info(
          package, program_export, 0, &root_info)));
  if (root_info.fixed_buffer_index != 0 ||
      root_info.required_byte_length == 0 || root_info.minimum_alignment == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "the Qwen3.8 parameter root has an invalid ABI");
  }

  iree_status_t status = qwen38_allocate_buffer(
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
    status = qwen38_status_from_loomc(loomc_cmd_program_package_parameter_info(
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
    (*out_requests)[i] = (qwen38_parameter_request_t){
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
            .fn = qwen38_parameter_enumerate,
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

static iree_status_t qwen38_prepare_layer_program(
    qwen_tooling_runtime_context_t* runtime_context,
    iree_allocator_t host_allocator, qwen38_layer_program_t* out_program) {
  memset(out_program, 0, sizeof(*out_program));
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
  qwen38_parameter_request_t* parameter_requests = NULL;
  iree_hal_semaphore_t* parameter_ready_semaphore = NULL;
  bool parameter_gather_submitted = false;
  iree_hal_executable_t** executables = NULL;
  iree_host_size_t executable_count = 0;

  iree_status_t status =
      qwen38_status_from_loomc(loomc_target_environment_create_amdgpu(
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
    status = qwen38_status_from_loomc(
        loomc_context_create(&context_options, loom_allocator, &context));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_status_from_loomc(loomc_workspace_create(
        /*options=*/NULL, loom_allocator, &workspace));
  }
  if (iree_status_is_ok(status)) {
    if (qwen38_gdn_layer_decode_source_size() != 1) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Qwen3.8 GDN layer source must contain exactly one file");
    } else {
      const iree_file_toc_t* files = qwen38_gdn_layer_decode_source_create();
      const loomc_source_options_t source_options = {
          .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
          .structure_size = sizeof(source_options),
          .format = LOOMC_SOURCE_FORMAT_TEXT,
          .identifier = loomc_make_cstring_view(files[0].name),
          .contents = loomc_make_byte_span(files[0].data, files[0].size),
          .storage = LOOMC_SOURCE_STORAGE_BORROWED,
      };
      status = qwen38_status_from_loomc(
          loomc_source_create(&source_options, loom_allocator, &source));
    }
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_status_from_loomc(loomc_module_deserialize_from_source(
        context, workspace, source, /*options=*/NULL, loom_allocator, &module,
        &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_require_result(IREE_SV("command source parse"), result);
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
    status =
        qwen38_status_from_loomc(loomc_target_profile_create_amdgpu_iree_hal(
            target_environment, &profile_options, loom_allocator,
            &target_profile, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_require_result(IREE_SV("AMDGPU target profile"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status = qwen38_query_kernels(module, host_allocator, &kernel_functions,
                                  &kernel_function_count);
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
    status = qwen38_status_from_loomc(loomc_compiler_create(
        context, /*options=*/NULL, loom_allocator, &compiler));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_status_from_loomc(loomc_pass_program_create_empty(
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
    status =
        qwen38_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
            context, &pipeline_options, loom_allocator,
            &executable_pass_program, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_require_result(IREE_SV("AMDGPU pipeline"), result);
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
        .module_name = loomc_make_cstring_view("qwen38_gdn_layer0_decode"),
    };
    status = qwen38_status_from_loomc(
        loomc_compile_module(compiler, workspace, empty_pass_program, module,
                             &compile_options, loom_allocator, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_require_result(IREE_SV("target specialization"), result);
  }
  loomc_result_release(result);
  result = NULL;

  const loomc_string_view_t root_name =
      loomc_make_cstring_view("qwen38_gdn_layer0_decode");
  if (iree_status_is_ok(status)) {
    status = qwen38_status_from_loomc(loomc_program_plan_prepare(
        workspace, module, &root_name, 1, /*options=*/NULL, loom_allocator,
        &plan, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_require_result(IREE_SV("command program plan"), result);
  }
  loomc_result_release(result);
  result = NULL;

  loomc_program_plan_root_t root = loomc_program_plan_root_invalid();
  loomc_cmd_program_plan_root_info_t root_info = {0};
  if (iree_status_is_ok(status)) {
    root = loomc_program_plan_root_at(plan, 0);
    status = qwen38_status_from_loomc(
        loomc_cmd_program_plan_root_info(plan, root, &root_info));
  }
  if (iree_status_is_ok(status) &&
      loomc_program_plan_unit_is_valid(root_info.launch_config_unit)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "the exact layer unexpectedly requires dynamic launch config");
  }

  if (iree_status_is_ok(status)) {
    status = qwen38_compile_unit(
        plan, compiler, workspace, root_info.package_unit, empty_pass_program,
        IREE_SV("command package compile"), host_allocator, &result);
  }
  if (iree_status_is_ok(status)) {
    const loomc_artifact_t* artifact =
        qwen38_find_artifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                             LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE);
    if (!artifact) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "command package artifact is absent");
    } else {
      status = qwen38_status_from_loomc(loomc_cmd_program_package_load(
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
    status = qwen38_status_from_loomc(loomc_cmd_program_package_lookup_export(
        package, root_name, &program_export));
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_status_from_loomc(loomc_cmd_program_package_export_info(
        package, program_export, &program_info));
  }
  if (iree_status_is_ok(status) &&
      (program_info.rebindable_binding_count != 3 ||
       program_info.transient.binding_index ==
           LOOMC_CMD_PROGRAM_BINDING_INVALID ||
       program_info.transient.required_byte_length == 0 ||
       program_info.config.binding_index !=
           LOOMC_CMD_PROGRAM_BINDING_INVALID)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "the exact layer command program published an incompatible ABI");
  }

  if (iree_status_is_ok(status)) {
    status = qwen38_begin_parameter_gather(
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
          "the exact layer requires an external executable import");
      break;
    }
    status = qwen38_compile_unit(
        plan, compiler, workspace, unit, executable_pass_program,
        IREE_SV("AMDGPU executable compile"), host_allocator, &result);
    if (iree_status_is_ok(status)) {
      const loomc_artifact_t* artifact =
          qwen38_find_artifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                               LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
      if (!artifact) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "AMDGPU executable artifact is absent");
      } else {
        status = qwen38_load_executable(device, artifact, &executables[i]);
      }
    }
    loomc_result_release(result);
    result = NULL;
  }

  if (iree_status_is_ok(status)) {
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
    status = qwen38_status_from_loomc(loomc_cmd_hal_program_create(
        package, program_export, device, &materialization_options,
        loom_allocator, &out_program->hal_program));
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
    status =
        qwen38_status_from_loomc(loomc_cmd_program_package_parameter_root_info(
            package, program_export, 0, &parameter_root_info));
    if (iree_status_is_ok(status)) {
      out_program->info = program_info;
      out_program->parameter_byte_length =
          parameter_root_info.required_byte_length;
    }
  }

  if (!iree_status_is_ok(status)) {
    loomc_cmd_hal_program_release(out_program->hal_program);
    memset(out_program, 0, sizeof(*out_program));
  }
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

static void qwen38_layer_program_deinitialize(qwen38_layer_program_t* program) {
  loomc_cmd_hal_program_release(program->hal_program);
  memset(program, 0, sizeof(*program));
}

static iree_status_t qwen38_fill_and_wait(iree_hal_device_t* device,
                                          iree_hal_buffer_t* buffer) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  const uint32_t zero = 0;
  iree_status_t status = iree_hal_device_queue_fill(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signals, buffer, 0, iree_hal_buffer_byte_length(buffer), &zero,
      sizeof(zero), IREE_HAL_FILL_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

static iree_status_t qwen38_submit_and_wait(
    iree_hal_device_t* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
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

static iree_status_t qwen38_gdn_layer_run(void) {
  const iree_allocator_t host_allocator = iree_allocator_system();
  qwen_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  qwen38_layer_program_t program;
  memset(&program, 0, sizeof(program));
  iree_hal_buffer_t* residual_buffer = NULL;
  iree_hal_buffer_t* state_buffer = NULL;
  iree_hal_buffer_t* transient_buffer = NULL;
  float* residual_values = NULL;
  float* state_values = NULL;
  iree_hal_profiling_from_flags_t* profiling = NULL;

  iree_status_t status = iree_ok_status();
  if (FLAG_transition_count < 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--transition_count must be positive");
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "qwen38: indexed GGUF; compiling and gathering layer 0...\n");
    status = qwen38_prepare_layer_program(&runtime_context, host_allocator,
                                          &program);
  }

  iree_hal_device_t* device =
      qwen_tooling_runtime_context_device(&runtime_context);
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_HIDDEN_BYTE_LENGTH, &residual_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_GDN_STATE_BYTE_LENGTH, &state_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        program.info.transient.minimum_alignment,
        program.info.transient.required_byte_length, &transient_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, QWEN38_HIDDEN_ELEMENT_COUNT, sizeof(*residual_values),
        (void**)&residual_values);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < QWEN38_HIDDEN_ELEMENT_COUNT; ++i) {
      const int32_t centered = (int32_t)(i % 257) - 128;
      residual_values[i] = (float)centered / 128.0f;
    }
    status = iree_hal_device_transfer_h2d(
        device, residual_values, residual_buffer, 0, QWEN38_HIDDEN_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, state_buffer);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "qwen38: materialized %" PRIhsz " commands; executing layer 0...\n",
            program.info.command_count);
    iree_hal_buffer_binding_t bindings[3] = {
        [0] =
            {
                .buffer = residual_buffer,
                .offset = 0,
                .length = IREE_HAL_WHOLE_BUFFER,
            },
        [1] =
            {
                .buffer = state_buffer,
                .offset = 0,
                .length = IREE_HAL_WHOLE_BUFFER,
            },
        [2] =
            {
                .buffer = transient_buffer,
                .offset = 0,
                .length = IREE_HAL_WHOLE_BUFFER,
            },
    };
    if (program.info.transient.binding_index != 2) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "the exact layer transient is not binding-table slot 2");
    } else {
      status = iree_hal_begin_profiling_from_flags(device, host_allocator,
                                                   &profiling);
      for (int32_t i = 0;
           i < FLAG_transition_count && iree_status_is_ok(status); ++i) {
        status = qwen38_submit_and_wait(
            device, loomc_cmd_hal_program_command_buffer(program.hal_program),
            (iree_hal_buffer_binding_table_t){
                .count = IREE_ARRAYSIZE(bindings),
                .bindings = bindings,
            });
      }
      status = iree_status_join(status,
                                iree_hal_end_profiling_from_flags(profiling));
      profiling = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, residual_buffer, 0, residual_values, QWEN38_HIDDEN_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status) && FLAG_state_output[0] != '\0') {
    status = iree_allocator_malloc_array(
        host_allocator, QWEN38_GDN_STATE_ELEMENT_COUNT, sizeof(*state_values),
        (void**)&state_values);
  }
  if (iree_status_is_ok(status) && state_values) {
    status = iree_hal_device_transfer_d2h(
        device, state_buffer, 0, state_values, QWEN38_GDN_STATE_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  double sum = 0.0;
  double sum_squares = 0.0;
  float minimum = FLT_MAX;
  float maximum = -FLT_MAX;
  iree_host_size_t finite_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < QWEN38_HIDDEN_ELEMENT_COUNT; ++i) {
      const float value = residual_values[i];
      if (isfinite(value)) ++finite_count;
      if (value < minimum) minimum = value;
      if (value > maximum) maximum = value;
      sum += value;
      sum_squares += (double)value * (double)value;
    }
    if (finite_count != QWEN38_HIDDEN_ELEMENT_COUNT) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "the Qwen3.8 layer produced %" PRIhsz
                                "/%d finite values",
                                finite_count, QWEN38_HIDDEN_ELEMENT_COUNT);
    }
  }
  if (iree_status_is_ok(status) && FLAG_output[0] != '\0') {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_output),
        iree_make_const_byte_span(residual_values, QWEN38_HIDDEN_BYTE_LENGTH),
        host_allocator);
  }
  if (iree_status_is_ok(status) && state_values) {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_state_output),
        iree_make_const_byte_span(state_values, QWEN38_GDN_STATE_BYTE_LENGTH),
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout,
            "Qwen3.8 GDN layer 0 executed %d time(s): %" PRIhsz
            " commands, %" PRIhsz " parameters, %" PRIu64
            " parameter bytes, %" PRIu64
            " transient bytes, residual[min=%g max=%g sum=%.9g "
            "sum_squares=%.9g]\n",
            FLAG_transition_count, program.info.command_count,
            program.info.parameter_count,
            (uint64_t)program.parameter_byte_length,
            (uint64_t)program.info.transient.required_byte_length, minimum,
            maximum, sum, sum_squares);
  }

  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  iree_allocator_free(host_allocator, state_values);
  iree_allocator_free(host_allocator, residual_values);
  iree_hal_buffer_release(transient_buffer);
  iree_hal_buffer_release(state_buffer);
  iree_hal_buffer_release(residual_buffer);
  qwen38_layer_program_deinitialize(&program);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen38-gdn-layer-cli", qwen38_gdn_layer_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) status = qwen38_gdn_layer_run();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
