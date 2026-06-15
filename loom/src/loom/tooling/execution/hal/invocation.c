// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/invocation.h"

#include "iree/modules/hal/types.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "loom/target/launch.h"

enum {
  LOOM_RUN_HAL_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT = 1024,
};

void loom_run_hal_invocation_options_initialize(
    loom_run_hal_invocation_options_t* out_options) {
  *out_options = (loom_run_hal_invocation_options_t){
      .workgroup_count = {1, 1, 1},
  };
}

void loom_run_hal_invocation_request_initialize(
    loom_run_hal_invocation_request_t* out_request) {
  *out_request = (loom_run_hal_invocation_request_t){0};
  loom_run_hal_invocation_options_initialize(&out_request->options);
}

void loom_run_hal_invocation_plan_initialize(
    loom_run_hal_invocation_plan_t* out_plan) {
  *out_plan = (loom_run_hal_invocation_plan_t){0};
  loom_run_hal_invocation_options_initialize(&out_plan->options);
}

void loom_run_hal_invocation_plan_deinitialize(
    loom_run_hal_invocation_plan_t* plan) {
  if (plan == NULL) {
    return;
  }
  iree_vm_list_release(plan->expected_bindings);
  iree_hal_allocator_release(plan->expected_binding_allocator);
  iree_vm_list_release(plan->bindings);
  *plan = (loom_run_hal_invocation_plan_t){0};
}

void loom_run_hal_prepared_candidate_initialize(
    loom_run_hal_prepared_candidate_t* out_candidate) {
  *out_candidate = (loom_run_hal_prepared_candidate_t){0};
}

void loom_run_hal_prepared_candidate_deinitialize(
    loom_run_hal_prepared_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  iree_hal_executable_release(candidate->executable);
  *candidate = (loom_run_hal_prepared_candidate_t){0};
}

void loom_run_hal_iteration_initialize(
    loom_run_hal_iteration_t* out_iteration) {
  *out_iteration = (loom_run_hal_iteration_t){0};
}

void loom_run_hal_iteration_deinitialize(loom_run_hal_iteration_t* iteration) {
  if (iteration == NULL) {
    return;
  }
  iree_vm_list_release(iteration->bindings);
  *iteration = (loom_run_hal_iteration_t){0};
}

void loom_run_hal_dispatch_batch_options_initialize(
    loom_run_hal_dispatch_batch_options_t* out_options) {
  *out_options = (loom_run_hal_dispatch_batch_options_t){
      .dispatch_count = 1,
      .command_buffer_mode = IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED |
                             IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
      .execute_flags = IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME,
  };
}

void loom_run_hal_dispatch_batch_initialize(
    loom_run_hal_dispatch_batch_t* out_batch) {
  *out_batch = (loom_run_hal_dispatch_batch_t){0};
}

void loom_run_hal_dispatch_batch_deinitialize(
    loom_run_hal_dispatch_batch_t* batch) {
  if (batch == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < batch->binding_list_count; ++i) {
    iree_vm_list_release(batch->binding_lists[i]);
  }
  iree_allocator_free(batch->host_allocator, batch->binding_lists);
  iree_hal_command_buffer_release(batch->command_buffer);
  iree_hal_semaphore_release(batch->semaphore);
  *batch = (loom_run_hal_dispatch_batch_t){0};
}

void loom_run_hal_invocation_result_initialize(
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* out_result) {
  *out_result = (loom_run_hal_invocation_result_t){0};
  iree_string_builder_initialize(allocator, &out_result->output);
}

void loom_run_hal_invocation_result_deinitialize(
    loom_run_hal_invocation_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&result->output);
  *result = (loom_run_hal_invocation_result_t){0};
}

iree_status_t loom_run_hal_artifact_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_artifact_t* artifact,
    iree_hal_executable_t** out_hal_executable) {
  *out_hal_executable = NULL;
  if (runtime->device == NULL || runtime->executable_cache == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format = artifact->executable_format;
  executable_params.executable_data = artifact->executable_data;
  return iree_hal_executable_cache_prepare_executable(
      runtime->executable_cache, &executable_params, out_hal_executable);
}

iree_status_t loom_run_hal_prepared_candidate_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_artifact_t* artifact,
    loom_run_hal_prepared_candidate_t* out_candidate) {
  loom_run_hal_prepared_candidate_initialize(out_candidate);
  iree_status_t status = loom_run_hal_artifact_prepare(
      runtime, artifact, &out_candidate->executable);
  if (iree_status_is_ok(status)) {
    out_candidate->target_bundle = artifact->target_bundle;
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_prepared_candidate_deinitialize(out_candidate);
  }
  return status;
}

static iree_status_t loom_run_hal_binding_refs_from_list(
    iree_vm_list_t* binding_list, iree_hal_buffer_ref_t* binding_refs,
    iree_host_size_t binding_ref_capacity) {
  const iree_host_size_t binding_count = iree_vm_list_size(binding_list);
  if (binding_count > binding_ref_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL binding count %" PRIhsz
                            " exceeds capacity %" PRIhsz,
                            binding_count, binding_ref_capacity);
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    iree_vm_ref_t value = iree_vm_ref_null();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(binding_list, i, &value));
    iree_hal_buffer_t* buffer = NULL;
    if (iree_hal_buffer_isa(value)) {
      buffer = iree_hal_buffer_deref(value);
    } else if (iree_hal_buffer_view_isa(value)) {
      buffer = iree_hal_buffer_view_buffer(iree_hal_buffer_view_deref(value));
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL binding %" PRIhsz " is not a buffer or buffer view", i);
    }
    binding_refs[i] =
        iree_hal_make_buffer_ref(buffer, 0, IREE_HAL_WHOLE_BUFFER);
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_binding_list_total_byte_length(
    iree_vm_list_t* binding_list, uint64_t* out_byte_length) {
  *out_byte_length = 0;
  const iree_host_size_t binding_count = iree_vm_list_size(binding_list);
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    iree_vm_ref_t value = iree_vm_ref_null();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(binding_list, i, &value));
    iree_device_size_t byte_length = 0;
    if (iree_hal_buffer_isa(value)) {
      byte_length = iree_hal_buffer_byte_length(iree_hal_buffer_deref(value));
    } else if (iree_hal_buffer_view_isa(value)) {
      byte_length =
          iree_hal_buffer_view_byte_length(iree_hal_buffer_view_deref(value));
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL binding %" PRIhsz " is not a buffer or buffer view", i);
    }
    if (UINT64_MAX - *out_byte_length < byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL binding byte count overflowed uint64");
    }
    *out_byte_length += byte_length;
  }
  return iree_ok_status();
}

static iree_const_byte_span_t loom_run_hal_dispatch_constants(
    const loom_run_hal_invocation_options_t* options) {
  return iree_make_const_byte_span(
      (const uint8_t*)options->constants,
      options->constant_count * sizeof(options->constants[0]));
}

static iree_string_view_t loom_run_hal_normalize_function_name(
    iree_string_view_t function_name) {
  function_name = iree_string_view_trim(function_name);
  if (iree_string_view_starts_with_char(function_name, '@')) {
    function_name = iree_string_view_remove_prefix(function_name, 1);
  }
  return function_name;
}

static iree_status_t loom_run_hal_select_single_function_name(
    iree_hal_executable_t* executable, iree_string_view_t* out_function_name) {
  *out_function_name = iree_string_view_empty();
  const iree_host_size_t function_count =
      iree_hal_executable_function_count(executable);
  if (function_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL dispatch requires a function name when executable has %" PRIhsz
        " functions",
        function_count);
  }
  iree_hal_executable_function_info_t info = {0};
  IREE_RETURN_IF_ERROR(iree_hal_executable_function_info(
      executable, iree_hal_executable_function_from_index(0), &info));
  if (iree_string_view_is_empty(info.name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL dispatch requires a function name because the only executable "
        "function has no reflected name");
  }
  *out_function_name = info.name;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_lookup_dispatch_function(
    iree_hal_executable_t* executable,
    const loom_run_hal_invocation_options_t* options,
    iree_hal_executable_function_t* out_function) {
  iree_string_view_t function_name =
      loom_run_hal_normalize_function_name(options->function_name);
  if (iree_string_view_is_empty(function_name)) {
    IREE_RETURN_IF_ERROR(
        loom_run_hal_select_single_function_name(executable, &function_name));
  }
  return iree_hal_executable_lookup_function_by_name(executable, function_name,
                                                     out_function);
}

static iree_status_t loom_run_hal_record_dispatch_batch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_host_size_t binding_list_count, iree_vm_list_t* const* binding_lists,
    iree_host_size_t binding_list_offset,
    const loom_run_hal_invocation_options_t* options,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;

  iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
      options->workgroup_count[0], options->workgroup_count[1],
      options->workgroup_count[2]);
  iree_const_byte_span_t constants = loom_run_hal_dispatch_constants(options);
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(
      loom_run_hal_lookup_dispatch_function(executable, options, &function));

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device, batch_options->command_buffer_mode,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < batch_options->dispatch_count; ++i) {
    const iree_host_size_t binding_list_index =
        (binding_list_offset + i) % binding_list_count;
    iree_vm_list_t* binding_list = binding_lists[binding_list_index];
    iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
    status = loom_run_hal_binding_refs_from_list(binding_list, binding_refs,
                                                 IREE_ARRAYSIZE(binding_refs));
    if (!iree_status_is_ok(status)) {
      break;
    }
    iree_hal_buffer_ref_list_t bindings = {
        .count = iree_vm_list_size(binding_list),
        .values = binding_refs,
    };
    status = iree_hal_command_buffer_dispatch(
        command_buffer, executable, function, config, constants, bindings,
        IREE_HAL_DISPATCH_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

static iree_status_t loom_run_hal_record_dispatch_sequence_batch(
    iree_hal_device_t* device, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    iree_host_size_t plan_ring_count,
    const loom_run_hal_invocation_plan_t* const* plans,
    iree_vm_list_t* const* binding_lists, iree_host_size_t plan_ring_offset,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device, batch_options->command_buffer_mode,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  for (iree_host_size_t batch_index = 0;
       iree_status_is_ok(status) && batch_index < batch_options->dispatch_count;
       ++batch_index) {
    const iree_host_size_t ring_index =
        (plan_ring_offset + batch_index) % plan_ring_count;
    for (iree_host_size_t step_index = 0;
         iree_status_is_ok(status) && step_index < sequence_count;
         ++step_index) {
      const iree_host_size_t plan_index =
          ring_index * sequence_count + step_index;
      const loom_run_hal_invocation_plan_t* plan = plans[plan_index];
      iree_vm_list_t* binding_list = binding_lists[plan_index];
      iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
      status = loom_run_hal_binding_refs_from_list(
          binding_list, binding_refs, IREE_ARRAYSIZE(binding_refs));
      if (!iree_status_is_ok(status)) {
        break;
      }
      iree_hal_buffer_ref_list_t bindings = {
          .count = iree_vm_list_size(binding_list),
          .values = binding_refs,
      };
      const loom_run_hal_invocation_options_t* options = &plan->options;
      iree_hal_executable_function_t function =
          iree_hal_executable_function_invalid();
      status = loom_run_hal_lookup_dispatch_function(
          candidates[step_index]->executable, options, &function);
      if (!iree_status_is_ok(status)) {
        break;
      }
      iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
          options->workgroup_count[0], options->workgroup_count[1],
          options->workgroup_count[2]);
      status = iree_hal_command_buffer_dispatch(
          command_buffer, candidates[step_index]->executable, function, config,
          loom_run_hal_dispatch_constants(options), bindings,
          IREE_HAL_DISPATCH_FLAG_NONE);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

iree_status_t loom_run_hal_dispatch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options) {
  iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_refs_from_list(
      binding_list, binding_refs, IREE_ARRAYSIZE(binding_refs)));
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(
      loom_run_hal_lookup_dispatch_function(executable, options, &function));

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_hal_semaphore_t* semaphore = NULL;
  uint64_t signal_value = 1;

  iree_status_t status = iree_hal_command_buffer_create(
      device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_list_t bindings = {
        .count = iree_vm_list_size(binding_list),
        .values = binding_refs,
    };
    iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
        options->workgroup_count[0], options->workgroup_count[1],
        options->workgroup_count[2]);
    status = iree_hal_command_buffer_dispatch(
        command_buffer, executable, function, config,
        loom_run_hal_dispatch_constants(options), bindings,
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
    iree_hal_semaphore_list_t wait_semaphores = iree_hal_semaphore_list_empty();
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &semaphore,
        .payload_values = &signal_value,
    };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores, signal_semaphores,
        command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
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

iree_status_t loom_run_hal_invocation_execute(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_artifact_t* artifact, iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options) {
  loom_run_hal_prepared_candidate_t candidate = {0};
  iree_status_t status =
      loom_run_hal_prepared_candidate_prepare(runtime, artifact, &candidate);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch(runtime->device, candidate.executable,
                                   binding_list, options);
  }
  loom_run_hal_prepared_candidate_deinitialize(&candidate);
  return status;
}

iree_status_t loom_run_hal_transfer_bindings_to_host(
    const loom_run_hal_runtime_t* runtime, iree_vm_list_t* binding_list) {
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_hal_allocator_t* device_allocator =
      iree_hal_device_allocator(runtime->device);
  iree_hal_buffer_params_t host_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
  return iree_tooling_transfer_variants(
      binding_list, runtime->device, device_allocator, host_params,
      /*wait_fence=*/NULL, /*signal_fence=*/NULL);
}

static iree_status_t loom_run_hal_binding_specs_validate(
    const loom_run_hal_binding_specs_t* specs,
    iree_string_view_t binding_list_name) {
  if (specs->count > LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%.*s binding count %" PRIhsz " exceeds maximum %d",
                            (int)binding_list_name.size, binding_list_name.data,
                            specs->count, LOOM_RUN_HAL_MAX_BINDING_COUNT);
  }
  if (specs->count > 0 &&
      (specs->values == NULL || specs->conventions == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s binding specs require values and calling conventions",
        (int)binding_list_name.size, binding_list_name.data);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_run_hal_binding_specs_conventions(
    const loom_run_hal_binding_specs_t* specs) {
  return iree_make_string_view(specs->conventions, specs->count);
}

static iree_string_view_list_t loom_run_hal_binding_specs_values(
    const loom_run_hal_binding_specs_t* specs) {
  return (iree_string_view_list_t){
      .count = specs->count,
      .values = specs->values,
  };
}

static iree_status_t loom_run_hal_parse_binding_specs(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_binding_specs_t* specs,
    iree_hal_allocator_t* device_allocator, iree_allocator_t allocator,
    iree_vm_list_t** out_list) {
  IREE_RETURN_IF_ERROR(
      loom_run_hal_binding_specs_validate(specs, IREE_SV("HAL")));
  return iree_tooling_parse_variants(
      loom_run_hal_binding_specs_conventions(specs),
      loom_run_hal_binding_specs_values(specs), runtime->device,
      device_allocator, allocator, out_list);
}

static iree_status_t loom_run_hal_process_invocation_bindings(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan, iree_vm_list_t* binding_list,
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* result) {
  IREE_RETURN_IF_ERROR(
      loom_run_hal_transfer_bindings_to_host(runtime, binding_list));

  const iree_host_size_t max_output_element_count =
      plan->max_output_element_count == 0
          ? LOOM_RUN_HAL_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT
          : plan->max_output_element_count;
  if (plan->expected_bindings == NULL) {
    return iree_tooling_format_variants(IREE_SV("binding"), binding_list,
                                        max_output_element_count,
                                        &result->output);
  }

  iree_status_t status = iree_ok_status();
  const bool did_match = iree_tooling_compare_variant_lists_and_append(
      plan->expected_bindings, binding_list, allocator, &result->output);
  if (did_match) {
    status = iree_string_builder_append_cstring(
        &result->output,
        "[SUCCESS] all HAL bindings matched their expected values.\n");
  }
  result->exit_code = did_match ? 0 : 1;
  return status;
}

static iree_status_t loom_run_hal_invocation_plan_validate(
    const loom_run_hal_invocation_plan_t* plan) {
  if (plan->options.constant_count > LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HAL dispatch constant count %" PRIhsz " exceeds maximum %d",
        plan->options.constant_count, LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
  }
  if (plan->bindings == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation plan requires bindings");
  }
  if (iree_vm_list_size(plan->bindings) > LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL binding count %" PRIhsz " exceeds maximum %d",
                            iree_vm_list_size(plan->bindings),
                            LOOM_RUN_HAL_MAX_BINDING_COUNT);
  }
  if (plan->expected_bindings != NULL &&
      iree_vm_list_size(plan->expected_bindings) !=
          iree_vm_list_size(plan->bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected HAL binding count %" PRIhsz
                            " must match input binding count %" PRIhsz,
                            iree_vm_list_size(plan->expected_bindings),
                            iree_vm_list_size(plan->bindings));
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_prepared_candidate_validate_dispatch(
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan) {
  const loom_target_bundle_t* target_bundle = candidate->target_bundle;
  if (target_bundle == NULL) {
    return iree_ok_status();
  }
  if (target_bundle->snapshot == NULL || target_bundle->export_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL prepared candidate target bundle is missing "
                            "snapshot or export plan");
  }
  if (target_bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL prepared candidate target bundle must use HAL kernel ABI");
  }
  const loom_target_dispatch_workgroup_count_t workgroup_count = {
      .x = plan->options.workgroup_count[0],
      .y = plan->options.workgroup_count[1],
      .z = plan->options.workgroup_count[2],
  };
  return loom_target_validate_hal_dispatch_workgroup_count(
      target_bundle->snapshot, &target_bundle->export_plan->hal_kernel,
      &workgroup_count);
}

iree_status_t loom_run_hal_dispatch_batch_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_allocator_t allocator, loom_run_hal_dispatch_batch_t* out_batch) {
  iree_vm_list_t* binding_list = plan->bindings;
  return loom_run_hal_dispatch_batch_prepare_from_binding_ring(
      runtime, candidate, plan, /*binding_list_count=*/1, &binding_list,
      /*binding_list_offset=*/0, batch_options, allocator, out_batch);
}

static iree_status_t loom_run_hal_dispatch_binding_ring_validate(
    const loom_run_hal_invocation_plan_t* plan,
    iree_host_size_t binding_list_count, iree_vm_list_t* const* binding_lists) {
  if (binding_list_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch binding ring must contain at least "
                            "one binding list");
  }
  if (binding_lists == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch binding ring requires binding "
                            "lists");
  }
  const iree_host_size_t plan_binding_count = iree_vm_list_size(plan->bindings);
  for (iree_host_size_t i = 0; i < binding_list_count; ++i) {
    if (binding_lists[i] == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL dispatch binding ring entry %" PRIhsz " is null", i);
    }
    const iree_host_size_t binding_count = iree_vm_list_size(binding_lists[i]);
    if (binding_count != plan_binding_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL dispatch binding ring entry %" PRIhsz
                              " binding count %" PRIhsz
                              " must match plan binding count %" PRIhsz,
                              i, binding_count, plan_binding_count);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_dispatch_batch_prepare_from_binding_ring(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    iree_host_size_t binding_list_count, iree_vm_list_t* const* binding_lists,
    iree_host_size_t binding_list_offset,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_allocator_t allocator, loom_run_hal_dispatch_batch_t* out_batch) {
  loom_run_hal_dispatch_batch_initialize(out_batch);
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  if (batch_options->dispatch_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch batch must contain at least one "
                            "dispatch");
  }
  if (candidate->executable == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL prepared candidate requires an executable");
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_prepared_candidate_validate_dispatch(candidate, plan));
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_dispatch_binding_ring_validate(
      plan, binding_list_count, binding_lists));

  out_batch->host_allocator = allocator;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, binding_list_count, sizeof(*out_batch->binding_lists),
      (void**)&out_batch->binding_lists);
  if (iree_status_is_ok(status)) {
    out_batch->binding_list_count = binding_list_count;
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < out_batch->binding_list_count; ++i) {
    status = iree_vm_list_clone(binding_lists[i], allocator,
                                &out_batch->binding_lists[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_record_dispatch_batch(
        runtime->device, candidate->executable, out_batch->binding_list_count,
        out_batch->binding_lists, binding_list_offset, &plan->options,
        batch_options, &out_batch->command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        runtime->device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &out_batch->semaphore);
  }
  if (iree_status_is_ok(status)) {
    out_batch->next_signal_value = 1;
    out_batch->dispatch_count = batch_options->dispatch_count;
    out_batch->execute_flags = batch_options->execute_flags;
  } else {
    loom_run_hal_dispatch_batch_deinitialize(out_batch);
  }
  return status;
}

static iree_status_t loom_run_hal_dispatch_sequence_plan_ring_validate(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    iree_host_size_t plan_ring_count,
    const loom_run_hal_invocation_plan_t* const* plans) {
  if (sequence_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch sequence must contain at least one "
                            "dispatch step");
  }
  if (plan_ring_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch sequence plan ring must contain at "
                            "least one ring slot");
  }
  if (candidates == NULL || plans == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch sequence requires candidates and "
                            "plans");
  }
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }
  for (iree_host_size_t step_index = 0; step_index < sequence_count;
       ++step_index) {
    if (candidates[step_index] == NULL ||
        candidates[step_index]->executable == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL dispatch sequence step %" PRIhsz
                              " requires a prepared executable",
                              step_index);
    }
  }
  for (iree_host_size_t ring_index = 0; ring_index < plan_ring_count;
       ++ring_index) {
    for (iree_host_size_t step_index = 0; step_index < sequence_count;
         ++step_index) {
      const iree_host_size_t plan_index =
          ring_index * sequence_count + step_index;
      const loom_run_hal_invocation_plan_t* plan = plans[plan_index];
      if (plan == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HAL dispatch sequence ring slot %" PRIhsz
                                " step %" PRIhsz " has no invocation plan",
                                ring_index, step_index);
      }
      IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
      IREE_RETURN_IF_ERROR(loom_run_hal_prepared_candidate_validate_dispatch(
          candidates[step_index], plan));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_dispatch_sequence_batch_prepare_from_plan_ring(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    iree_host_size_t plan_ring_count,
    const loom_run_hal_invocation_plan_t* const* plans,
    iree_host_size_t plan_ring_offset,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_allocator_t allocator, loom_run_hal_dispatch_batch_t* out_batch) {
  loom_run_hal_dispatch_batch_initialize(out_batch);
  if (batch_options->dispatch_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch batch must contain at least one "
                            "dispatch sequence");
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_dispatch_sequence_plan_ring_validate(
      runtime, sequence_count, candidates, plan_ring_count, plans));

  iree_host_size_t plan_count = 0;
  if (!iree_host_size_checked_mul(plan_ring_count, sequence_count,
                                  &plan_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch sequence plan count overflowed host "
                            "size limits");
  }
  out_batch->host_allocator = allocator;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, plan_count, sizeof(*out_batch->binding_lists),
      (void**)&out_batch->binding_lists);
  if (iree_status_is_ok(status)) {
    out_batch->binding_list_count = plan_count;
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < out_batch->binding_list_count; ++i) {
    status = iree_vm_list_clone(plans[i]->bindings, allocator,
                                &out_batch->binding_lists[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_record_dispatch_sequence_batch(
        runtime->device, sequence_count, candidates, plan_ring_count, plans,
        out_batch->binding_lists, plan_ring_offset, batch_options,
        &out_batch->command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        runtime->device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &out_batch->semaphore);
  }
  if (iree_status_is_ok(status)) {
    out_batch->next_signal_value = 1;
    if (!iree_host_size_checked_mul(batch_options->dispatch_count,
                                    sequence_count,
                                    &out_batch->dispatch_count)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "HAL dispatch sequence batch count overflowed "
                                "host size limits");
    } else {
      out_batch->execute_flags = batch_options->execute_flags;
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_dispatch_batch_deinitialize(out_batch);
  }
  return status;
}

iree_status_t loom_run_hal_dispatch_batch_execute(
    const loom_run_hal_runtime_t* runtime,
    loom_run_hal_dispatch_batch_t* batch) {
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }
  if (batch->command_buffer == NULL || batch->semaphore == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch batch is not prepared");
  }

  uint64_t signal_value = batch->next_signal_value;
  iree_hal_semaphore_list_t wait_semaphores = iree_hal_semaphore_list_empty();
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &batch->semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t status = iree_hal_device_queue_execute(
      runtime->device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
      signal_semaphores, batch->command_buffer,
      iree_hal_buffer_binding_table_empty(), batch->execute_flags);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(batch->semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    ++batch->next_signal_value;
  }
  return status;
}

iree_status_t loom_run_hal_dispatch_batch_collect_results(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_dispatch_batch_t* batch, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  if (batch->binding_list_count == 0 || batch->binding_lists == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch batch has no binding lists");
  }
  loom_run_hal_iteration_t iteration = {
      .bindings = batch->binding_lists[0],
  };
  return loom_run_hal_invocation_collect_results(runtime, plan, &iteration,
                                                 allocator, result);
}

iree_status_t loom_run_hal_invocation_plan_prepare_from_specs(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_options_t* options,
    const loom_run_hal_binding_specs_t* bindings,
    const loom_run_hal_binding_specs_t* expected_bindings,
    iree_host_size_t max_output_element_count, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan) {
  loom_run_hal_invocation_plan_initialize(out_plan);
  IREE_RETURN_IF_ERROR(
      loom_run_hal_binding_specs_validate(bindings, IREE_SV("HAL")));
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_specs_validate(
      expected_bindings, IREE_SV("expected HAL")));
  if (options->constant_count > LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HAL dispatch constant count %" PRIhsz " exceeds maximum %d",
        options->constant_count, LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
  }
  if (expected_bindings->count != 0 &&
      expected_bindings->count != bindings->count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected HAL binding count %" PRIhsz
                            " must match input binding count %" PRIhsz,
                            expected_bindings->count, bindings->count);
  }
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_status_t status = loom_run_hal_parse_binding_specs(
      runtime, bindings, iree_hal_device_allocator(runtime->device), allocator,
      &out_plan->bindings);
  if (iree_status_is_ok(status) && expected_bindings->count != 0) {
    status =
        iree_hal_allocator_create_heap(IREE_SV("heap"), allocator, allocator,
                                       &out_plan->expected_binding_allocator);
  }
  if (iree_status_is_ok(status) && expected_bindings->count != 0) {
    status = loom_run_hal_parse_binding_specs(
        runtime, expected_bindings, out_plan->expected_binding_allocator,
        allocator, &out_plan->expected_bindings);
  }
  if (iree_status_is_ok(status)) {
    out_plan->options = *options;
    out_plan->max_output_element_count = max_output_element_count;
  } else {
    loom_run_hal_invocation_plan_deinitialize(out_plan);
  }
  return status;
}

iree_status_t loom_run_hal_invocation_plan_prepare_from_lists(
    const loom_run_hal_invocation_options_t* options, iree_vm_list_t* bindings,
    iree_vm_list_t* expected_bindings,
    iree_host_size_t max_output_element_count,
    loom_run_hal_invocation_plan_t* out_plan) {
  loom_run_hal_invocation_plan_initialize(out_plan);
  out_plan->options = *options;
  out_plan->bindings = bindings;
  out_plan->expected_bindings = expected_bindings;
  out_plan->max_output_element_count = max_output_element_count;
  iree_status_t status = loom_run_hal_invocation_plan_validate(out_plan);
  if (iree_status_is_ok(status)) {
    iree_vm_list_retain(out_plan->bindings);
    iree_vm_list_retain(out_plan->expected_bindings);
  } else {
    out_plan->bindings = NULL;
    out_plan->expected_bindings = NULL;
  }
  return status;
}

iree_status_t loom_run_hal_invocation_dispatch_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_iteration_t* out_iteration) {
  loom_run_hal_iteration_initialize(out_iteration);
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  if (candidate->executable == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL prepared candidate requires an executable");
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_prepared_candidate_validate_dispatch(candidate, plan));
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_status_t status =
      iree_vm_list_clone(plan->bindings, allocator, &out_iteration->bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch(runtime->device, candidate->executable,
                                   out_iteration->bindings, &plan->options);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_iteration_deinitialize(out_iteration);
  }
  return status;
}

iree_status_t loom_run_hal_invocation_collect_results(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_iteration_t* iteration, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  if (iteration->bindings == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL iteration requires bindings");
  }
  if (iree_vm_list_size(iteration->bindings) !=
      iree_vm_list_size(plan->bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL iteration binding count %" PRIhsz
                            " must match plan binding count %" PRIhsz,
                            iree_vm_list_size(iteration->bindings),
                            iree_vm_list_size(plan->bindings));
  }
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }
  return loom_run_hal_process_invocation_bindings(
      runtime, plan, iteration->bindings, allocator, result);
}

iree_status_t loom_run_hal_invocation_run_prepared(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;

  loom_run_hal_iteration_t iteration = {0};
  iree_status_t status = loom_run_hal_invocation_dispatch_plan(
      runtime, candidate, plan, allocator, &iteration);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_collect_results(runtime, plan, &iteration,
                                                     allocator, result);
  }
  loom_run_hal_iteration_deinitialize(&iteration);
  return status;
}

iree_status_t loom_run_hal_invocation_run_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_artifact_t* artifact,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  loom_run_hal_prepared_candidate_t candidate = {0};
  iree_status_t status =
      loom_run_hal_prepared_candidate_prepare(runtime, artifact, &candidate);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_run_prepared(runtime, &candidate, plan,
                                                  allocator, result);
  }
  loom_run_hal_prepared_candidate_deinitialize(&candidate);
  return status;
}

iree_status_t loom_run_hal_invocation_run(
    const loom_run_hal_invocation_request_t* request,
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  loom_run_hal_invocation_plan_t plan = {0};
  iree_status_t status = loom_run_hal_invocation_plan_prepare_from_specs(
      request->runtime, &request->options, &request->bindings,
      &request->expected_bindings, request->max_output_element_count, allocator,
      &plan);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_run_plan(
        request->runtime, request->artifact, &plan, allocator, result);
  }
  loom_run_hal_invocation_plan_deinitialize(&plan);
  return status;
}
