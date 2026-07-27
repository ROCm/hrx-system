// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/ireevm/invocation.h"

#include <string.h>

#include "iree/hal/api.h"
#include "iree/io/vec_stream.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/vm/bytecode/module.h"

enum {
  LOOM_RUN_VM_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT = 1024,
  LOOM_RUN_VM_OUTPUT_STREAM_BLOCK_SIZE = 4096,
};

iree_status_t loom_run_vm_runtime_initialize(
    iree_allocator_t allocator, loom_run_vm_runtime_t* out_runtime) {
  *out_runtime = (loom_run_vm_runtime_t){0};
  iree_status_t status =
      iree_tooling_create_instance(allocator, &out_runtime->instance);
  if (!iree_status_is_ok(status)) {
    loom_run_vm_runtime_deinitialize(out_runtime);
  }
  return status;
}

void loom_run_vm_runtime_deinitialize(loom_run_vm_runtime_t* runtime) {
  if (runtime == NULL) {
    return;
  }
  iree_vm_instance_release(runtime->instance);
  *runtime = (loom_run_vm_runtime_t){0};
}

void loom_run_vm_invocation_options_initialize(
    loom_run_vm_invocation_options_t* out_options) {
  *out_options = (loom_run_vm_invocation_options_t){
      .max_output_element_count = LOOM_RUN_VM_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT,
  };
}

void loom_run_vm_invocation_plan_initialize(
    loom_run_vm_invocation_plan_t* out_plan) {
  *out_plan = (loom_run_vm_invocation_plan_t){
      .output_mode = LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL,
      .max_output_element_count = LOOM_RUN_VM_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT,
  };
}

void loom_run_vm_invocation_plan_deinitialize(
    loom_run_vm_invocation_plan_t* plan) {
  if (plan == NULL) {
    return;
  }
  iree_vm_list_release(plan->expected_outputs);
  iree_hal_allocator_release(plan->expected_output_allocator);
  iree_vm_list_release(plan->inputs);
  *plan = (loom_run_vm_invocation_plan_t){0};
}

void loom_run_vm_prepared_candidate_options_initialize(
    loom_run_vm_prepared_candidate_options_t* out_options) {
  *out_options = (loom_run_vm_prepared_candidate_options_t){0};
}

void loom_run_vm_prepared_candidate_initialize(
    loom_run_vm_prepared_candidate_t* out_candidate) {
  *out_candidate = (loom_run_vm_prepared_candidate_t){0};
}

void loom_run_vm_prepared_candidate_deinitialize(
    loom_run_vm_prepared_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  iree_hal_allocator_release(candidate->device_allocator);
  iree_hal_device_release(candidate->device);
  iree_vm_context_release(candidate->context);
  iree_vm_module_release(candidate->module);
  loom_ireevm_module_archive_deinitialize(&candidate->archive,
                                          candidate->host_allocator);
  *candidate = (loom_run_vm_prepared_candidate_t){0};
}

void loom_run_vm_iteration_initialize(loom_run_vm_iteration_t* out_iteration) {
  *out_iteration = (loom_run_vm_iteration_t){0};
}

void loom_run_vm_iteration_deinitialize(loom_run_vm_iteration_t* iteration) {
  if (iteration == NULL) {
    return;
  }
  iree_vm_list_release(iteration->outputs);
  iree_vm_list_release(iteration->inputs);
  *iteration = (loom_run_vm_iteration_t){0};
}

void loom_run_vm_invocation_request_initialize(
    loom_run_vm_invocation_request_t* out_request) {
  *out_request = (loom_run_vm_invocation_request_t){0};
  loom_run_vm_invocation_options_initialize(&out_request->options);
}

void loom_run_vm_invocation_result_initialize(
    iree_allocator_t allocator, loom_run_vm_invocation_result_t* out_result) {
  *out_result = (loom_run_vm_invocation_result_t){0};
  iree_string_builder_initialize(allocator, &out_result->output);
}

void loom_run_vm_invocation_result_deinitialize(
    loom_run_vm_invocation_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&result->output);
  *result = (loom_run_vm_invocation_result_t){0};
}

static iree_status_t loom_run_vm_value_specs_validate(
    const loom_run_vm_value_specs_t* specs, iree_string_view_t list_name) {
  if (specs->count > 0 && specs->values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s specs require values", (int)list_name.size,
                            list_name.data);
  }
  return iree_ok_status();
}

static iree_string_view_list_t loom_run_vm_value_specs_list(
    const loom_run_vm_value_specs_t* specs) {
  return (iree_string_view_list_t){
      .count = specs->count,
      .values = specs->values,
  };
}

iree_status_t loom_run_vm_invocation_plan_prepare_from_specs(
    const loom_run_vm_invocation_plan_prepare_request_t* request,
    iree_allocator_t allocator, loom_run_vm_invocation_plan_t* out_plan) {
  loom_run_vm_invocation_plan_initialize(out_plan);
  IREE_RETURN_IF_ERROR(loom_run_vm_value_specs_validate(
      &request->options->inputs, IREE_SV("VM input")));
  IREE_RETURN_IF_ERROR(loom_run_vm_value_specs_validate(
      &request->options->outputs, IREE_SV("VM output")));
  IREE_RETURN_IF_ERROR(loom_run_vm_value_specs_validate(
      &request->options->expected_outputs, IREE_SV("expected VM output")));

  iree_status_t status = iree_tooling_parse_variants(
      request->arguments_cconv,
      loom_run_vm_value_specs_list(&request->options->inputs), request->device,
      request->device_allocator, allocator, &out_plan->inputs);
  if (iree_status_is_ok(status) &&
      request->options->expected_outputs.count != 0) {
    status =
        iree_hal_allocator_create_heap(IREE_SV("heap"), allocator, allocator,
                                       &out_plan->expected_output_allocator);
  }
  if (iree_status_is_ok(status) &&
      request->options->expected_outputs.count != 0) {
    status = iree_tooling_parse_variants(
        request->results_cconv,
        loom_run_vm_value_specs_list(&request->options->expected_outputs),
        request->device, out_plan->expected_output_allocator, allocator,
        &out_plan->expected_outputs);
  }
  if (iree_status_is_ok(status)) {
    out_plan->max_output_element_count =
        request->options->max_output_element_count == 0
            ? LOOM_RUN_VM_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT
            : request->options->max_output_element_count;
    if (request->options->expected_outputs.count != 0) {
      out_plan->output_mode = LOOM_RUN_VM_OUTPUT_PLAN_MODE_COMPARE_EXPECTED;
    } else if (request->options->outputs.count != 0) {
      out_plan->output_mode = LOOM_RUN_VM_OUTPUT_PLAN_MODE_WRITE_SPECS;
      out_plan->output_specs = request->options->outputs;
    } else {
      out_plan->output_mode = LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL;
    }
  } else {
    loom_run_vm_invocation_plan_deinitialize(out_plan);
  }
  return status;
}

static iree_status_t loom_run_vm_load_archive_module(
    const loom_run_vm_runtime_t* runtime,
    const loom_ireevm_module_archive_t* archive, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  *out_module = NULL;
  if (runtime->instance == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM runtime is not initialized");
  }
  if (archive->data == NULL || archive->data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM archive is empty");
  }
  return iree_vm_bytecode_module_create(
      runtime->instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_make_const_byte_span(archive->data, archive->data_length),
      iree_allocator_null(), allocator, out_module);
}

static iree_status_t loom_run_vm_dependency_modules_validate(
    iree_vm_module_t* const* dependency_modules,
    iree_host_size_t dependency_module_count) {
  if (dependency_module_count > 0 && dependency_modules == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM dependency module list is NULL");
  }
  for (iree_host_size_t i = 0; i < dependency_module_count; ++i) {
    if (dependency_modules[i] == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM dependency module %" PRIhsz " is NULL", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_run_vm_create_context(
    const loom_run_vm_runtime_t* runtime, iree_vm_module_t* module,
    const loom_run_vm_prepared_candidate_options_t* options,
    iree_allocator_t allocator, iree_vm_context_t** out_context,
    iree_hal_device_t** out_device,
    iree_hal_allocator_t** out_device_allocator) {
  *out_context = NULL;
  *out_device = NULL;
  *out_device_allocator = NULL;

  IREE_RETURN_IF_ERROR(loom_run_vm_dependency_modules_validate(
      options->dependency_modules, options->dependency_module_count));
  if (options->dependency_module_count == 0) {
    iree_vm_module_t* modules[] = {module};
    return iree_tooling_create_context_from_flags(
        runtime->instance, IREE_ARRAYSIZE(modules), modules,
        options->default_device_uri, allocator, out_context, out_device,
        out_device_allocator, /*out_replay_recorder=*/NULL);
  }

  const iree_host_size_t module_count = options->dependency_module_count + 1;
  iree_vm_module_t** modules = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, module_count * sizeof(*modules), (void**)&modules));
  for (iree_host_size_t i = 0; i < options->dependency_module_count; ++i) {
    modules[i] = options->dependency_modules[i];
  }
  modules[options->dependency_module_count] = module;
  iree_status_t status = iree_tooling_create_context_from_flags(
      runtime->instance, module_count, modules, options->default_device_uri,
      allocator, out_context, out_device, out_device_allocator,
      /*out_replay_recorder=*/NULL);
  iree_allocator_free(allocator, modules);
  return status;
}

static iree_status_t loom_run_vm_lookup_function(
    iree_vm_module_t* module, iree_string_view_t function_name,
    iree_vm_function_t* out_function) {
  if (iree_string_view_is_empty(function_name)) {
    return iree_tooling_find_single_exported_function(module, out_function);
  }
  return iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, function_name, out_function);
}

static iree_status_t loom_run_vm_clone_archive(
    const loom_ireevm_module_archive_t* archive, iree_allocator_t allocator,
    loom_ireevm_module_archive_t* out_archive) {
  *out_archive = (loom_ireevm_module_archive_t){0};
  if (archive->data == NULL || archive->data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM archive is empty");
  }

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, archive->data_length,
                                             (void**)&out_archive->data));
  memcpy(out_archive->data, archive->data, archive->data_length);
  out_archive->data_length = archive->data_length;
  return iree_ok_status();
}

iree_status_t loom_run_vm_prepared_candidate_prepare(
    const loom_run_vm_runtime_t* runtime,
    const loom_ireevm_module_archive_t* archive,
    const loom_run_vm_prepared_candidate_options_t* options,
    iree_allocator_t allocator,
    loom_run_vm_prepared_candidate_t* out_candidate) {
  loom_run_vm_prepared_candidate_initialize(out_candidate);
  out_candidate->host_allocator = allocator;

  iree_status_t status =
      loom_run_vm_clone_archive(archive, allocator, &out_candidate->archive);
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_load_archive_module(runtime, &out_candidate->archive,
                                             allocator, &out_candidate->module);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_create_context(runtime, out_candidate->module, options,
                                        allocator, &out_candidate->context,
                                        &out_candidate->device,
                                        &out_candidate->device_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_lookup_function(out_candidate->module,
                                         options->function_name,
                                         &out_candidate->function);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_vm_prepared_candidate_deinitialize(out_candidate);
  }
  return status;
}

static iree_status_t loom_run_vm_prepared_candidate_validate(
    const loom_run_vm_prepared_candidate_t* candidate) {
  if (candidate->context == NULL ||
      iree_vm_function_is_null(candidate->function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM prepared candidate is not initialized");
  }
  if (candidate->device != NULL && candidate->device_allocator == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM device candidate requires a HAL allocator");
  }
  return iree_ok_status();
}

static iree_status_t loom_run_vm_invocation_plan_validate(
    const loom_run_vm_invocation_plan_t* plan) {
  if (plan->inputs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM invocation plan requires inputs");
  }
  switch (plan->output_mode) {
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL:
      return iree_ok_status();
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_WRITE_SPECS:
      return loom_run_vm_value_specs_validate(&plan->output_specs,
                                              IREE_SV("VM output"));
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_COMPARE_EXPECTED:
      if (plan->expected_outputs == NULL) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "VM expected output comparison requires values");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown VM output plan mode %d",
                              (int)plan->output_mode);
  }
}

iree_status_t loom_run_vm_invocation_plan_prepare_from_prepared(
    const loom_run_vm_prepared_candidate_t* candidate,
    const loom_run_vm_invocation_options_t* options, iree_allocator_t allocator,
    loom_run_vm_invocation_plan_t* out_plan) {
  IREE_RETURN_IF_ERROR(loom_run_vm_prepared_candidate_validate(candidate));
  if (!iree_string_view_is_empty(options->function_name)) {
    iree_string_view_t function_name =
        iree_vm_function_name(&candidate->function);
    if (!iree_string_view_equal(options->function_name, function_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM invocation function '%.*s' does not match prepared function "
          "'%.*s'",
          (int)options->function_name.size, options->function_name.data,
          (int)function_name.size, function_name.data);
    }
  }

  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&candidate->function);
  iree_string_view_t arguments_cconv = iree_string_view_empty();
  iree_string_view_t results_cconv = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
      &signature, &arguments_cconv, &results_cconv));

  const loom_run_vm_invocation_plan_prepare_request_t prepare_request = {
      .options = options,
      .arguments_cconv = arguments_cconv,
      .results_cconv = results_cconv,
      .device = candidate->device,
      .device_allocator = candidate->device_allocator,
  };
  return loom_run_vm_invocation_plan_prepare_from_specs(&prepare_request,
                                                        allocator, out_plan);
}

static iree_status_t loom_run_vm_annotate_status_with_function_decl(
    iree_status_t status, iree_vm_function_t function) {
  if (iree_status_is_ok(status) || iree_vm_function_is_null(function)) {
    return status;
  }
  iree_string_view_t declaration = iree_vm_function_lookup_attr_by_name(
      &function, IREE_SV("iree.abi.declaration"));
  if (iree_string_view_is_empty(declaration)) {
    return status;
  }
  return iree_status_annotate_f(status, "`%.*s`", (int)declaration.size,
                                declaration.data);
}

static iree_status_t loom_run_vm_append_stream_block(
    void* user_data, iree_const_byte_span_t block) {
  iree_string_builder_t* builder = (iree_string_builder_t*)user_data;
  return iree_string_builder_append_string(
      builder,
      iree_make_string_view((const char*)block.data, block.data_length));
}

static iree_status_t loom_run_vm_write_outputs_to_result(
    iree_vm_list_t* outputs, const loom_run_vm_value_specs_t* output_specs,
    iree_host_size_t max_output_element_count, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result) {
  iree_io_stream_t* output_stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE, LOOM_RUN_VM_OUTPUT_STREAM_BLOCK_SIZE,
      allocator, &output_stream);
  if (iree_status_is_ok(status)) {
    status = iree_tooling_write_variants(
        outputs, loom_run_vm_value_specs_list(output_specs),
        max_output_element_count, output_stream, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_enumerate_blocks(
        output_stream, loom_run_vm_append_stream_block, &result->output);
  }
  iree_io_stream_release(output_stream);
  return status;
}

static iree_status_t loom_run_vm_process_outputs(
    const loom_run_vm_invocation_plan_t* plan, iree_vm_list_t* outputs,
    iree_allocator_t allocator, loom_run_vm_invocation_result_t* result) {
  switch (plan->output_mode) {
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL:
      return iree_tooling_format_variants(IREE_SV("result"), outputs,
                                          plan->max_output_element_count,
                                          &result->output);
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_WRITE_SPECS:
      return loom_run_vm_write_outputs_to_result(outputs, &plan->output_specs,
                                                 plan->max_output_element_count,
                                                 allocator, result);
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_COMPARE_EXPECTED:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown VM output plan mode %d",
                              (int)plan->output_mode);
  }
  if (plan->expected_outputs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM expected output comparison requires values");
  }

  iree_status_t status = iree_ok_status();
  const bool did_match = iree_tooling_compare_variant_lists_and_append(
      plan->expected_outputs, outputs, allocator, &result->output);
  if (did_match) {
    status = iree_string_builder_append_cstring(
        &result->output,
        "[SUCCESS] all function outputs matched their expected values.\n");
  }
  result->exit_code = did_match ? 0 : 1;
  return status;
}

iree_status_t loom_run_vm_invocation_invoke_plan(
    const loom_run_vm_prepared_candidate_t* candidate,
    const loom_run_vm_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_vm_iteration_t* out_iteration) {
  loom_run_vm_iteration_initialize(out_iteration);
  IREE_RETURN_IF_ERROR(loom_run_vm_prepared_candidate_validate(candidate));
  IREE_RETURN_IF_ERROR(loom_run_vm_invocation_plan_validate(plan));

  iree_hal_fence_t* finish_fence = NULL;

  iree_status_t status =
      iree_vm_list_clone(plan->inputs, allocator, &out_iteration->inputs);
  if (iree_status_is_ok(status) && candidate->device == NULL) {
    iree_string_view_t model = iree_vm_function_lookup_attr_by_name(
        &candidate->function, IREE_SV("iree.abi.model"));
    if (iree_string_view_equal(model, IREE_SV("coarse-fences"))) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "async VM invocation requires a HAL device");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_append_async_fences(
        out_iteration->inputs, candidate->function, candidate->device,
        /*wait_fence=*/NULL, &finish_fence);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 16,
                                 allocator, &out_iteration->outputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(candidate->context, candidate->function,
                            IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
                            out_iteration->inputs, out_iteration->outputs,
                            allocator);
  }
  if (iree_status_is_ok(status) && finish_fence != NULL) {
    status = iree_hal_fence_wait(finish_fence, iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_vm_iteration_deinitialize(out_iteration);
  }

  iree_hal_fence_release(finish_fence);
  return status;
}

iree_status_t loom_run_vm_invocation_collect_results(
    const loom_run_vm_prepared_candidate_t* candidate,
    const loom_run_vm_invocation_plan_t* plan,
    const loom_run_vm_iteration_t* iteration, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  IREE_RETURN_IF_ERROR(loom_run_vm_prepared_candidate_validate(candidate));
  IREE_RETURN_IF_ERROR(loom_run_vm_invocation_plan_validate(plan));
  if (iteration->outputs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM iteration requires outputs");
  }

  iree_status_t status = iree_ok_status();
  iree_string_view_t function_name =
      iree_vm_function_name(&candidate->function);
  status = iree_string_builder_append_format(&result->output, "EXEC @%.*s\n",
                                             (int)function_name.size,
                                             function_name.data);
  if (iree_status_is_ok(status) && candidate->device != NULL) {
    iree_hal_buffer_params_t target_params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
        .min_alignment = 0,
    };
    status = iree_tooling_transfer_variants(
        iteration->outputs, candidate->device, candidate->device_allocator,
        target_params, /*wait_fence=*/NULL, /*signal_fence=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_process_outputs(plan, iteration->outputs, allocator,
                                         result);
  }
  return status;
}

iree_status_t loom_run_vm_invocation_run_prepared(
    const loom_run_vm_prepared_candidate_t* candidate,
    const loom_run_vm_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;

  loom_run_vm_iteration_t iteration = {0};
  iree_status_t status = loom_run_vm_invocation_invoke_plan(
      candidate, plan, allocator, &iteration);
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_invocation_collect_results(candidate, plan, &iteration,
                                                    allocator, result);
  }
  loom_run_vm_iteration_deinitialize(&iteration);
  return status;
}

iree_status_t loom_run_vm_invocation_run(
    const loom_run_vm_invocation_request_t* request, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;

  loom_run_vm_prepared_candidate_options_t candidate_options = {0};
  loom_run_vm_prepared_candidate_options_initialize(&candidate_options);
  candidate_options.function_name = request->options.function_name;
  candidate_options.default_device_uri = request->options.default_device_uri;

  loom_run_vm_prepared_candidate_t candidate = {0};
  loom_run_vm_invocation_plan_t plan = {0};

  iree_status_t status = loom_run_vm_prepared_candidate_prepare(
      request->runtime, request->archive, &candidate_options, allocator,
      &candidate);
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_invocation_plan_prepare_from_prepared(
        &candidate, &request->options, allocator, &plan);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_invocation_run_prepared(&candidate, &plan, allocator,
                                                 result);
  }
  status = loom_run_vm_annotate_status_with_function_decl(status,
                                                          candidate.function);

  loom_run_vm_invocation_plan_deinitialize(&plan);
  loom_run_vm_prepared_candidate_deinitialize(&candidate);
  return status;
}
