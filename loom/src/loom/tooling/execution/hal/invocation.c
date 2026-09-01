// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/invocation.h"

#include <string.h>

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
  loom_run_hal_binding_list_deinitialize(&plan->expected_bindings);
  iree_hal_allocator_release(plan->expected_binding_allocator);
  loom_run_hal_binding_list_deinitialize(&plan->bindings);
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
  loom_run_hal_binding_list_deinitialize(&iteration->bindings);
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

void loom_run_hal_queue_dispatch_initialize(
    loom_run_hal_queue_dispatch_t* out_dispatch) {
  *out_dispatch = (loom_run_hal_queue_dispatch_t){0};
}

void loom_run_hal_queue_dispatch_deinitialize(
    loom_run_hal_queue_dispatch_t* dispatch) {
  if (dispatch == NULL) {
    return;
  }
  iree_hal_semaphore_release(dispatch->semaphore);
  iree_hal_executable_release(dispatch->executable);
  *dispatch = (loom_run_hal_queue_dispatch_t){0};
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
    loom_run_hal_binding_list_deinitialize(&batch->binding_lists[i]);
  }
  iree_allocator_free(batch->host_allocator, batch->binding_lists);
  iree_hal_command_buffer_release(batch->command_buffer);
  iree_hal_semaphore_release(batch->semaphore);
  *batch = (loom_run_hal_dispatch_batch_t){0};
}

void loom_run_hal_dispatch_sequence_initialize(
    loom_run_hal_dispatch_sequence_t* out_sequence) {
  *out_sequence = (loom_run_hal_dispatch_sequence_t){0};
}

void loom_run_hal_dispatch_sequence_deinitialize(
    loom_run_hal_dispatch_sequence_t* sequence) {
  if (sequence == NULL) {
    return;
  }
  iree_hal_command_buffer_release(sequence->command_buffer);
  iree_hal_semaphore_release(sequence->semaphore);
  *sequence = (loom_run_hal_dispatch_sequence_t){0};
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

void loom_run_hal_binding_list_initialize(
    loom_run_hal_binding_list_t* out_list) {
  *out_list = (loom_run_hal_binding_list_t){0};
}

void loom_run_hal_binding_list_deinitialize(loom_run_hal_binding_list_t* list) {
  if (list == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_tooling_buffer_binding_deinitialize(&list->values[i]);
  }
  iree_allocator_free(list->host_allocator, list->values);
  *list = (loom_run_hal_binding_list_t){0};
}

iree_status_t loom_run_hal_binding_list_initialize_capacity(
    iree_host_size_t capacity, iree_allocator_t allocator,
    loom_run_hal_binding_list_t* out_list) {
  loom_run_hal_binding_list_initialize(out_list);
  out_list->host_allocator = allocator;
  if (capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, capacity,
                                                   sizeof(*out_list->values),
                                                   (void**)&out_list->values));
  memset(out_list->values, 0, capacity * sizeof(*out_list->values));
  out_list->capacity = capacity;
  return iree_ok_status();
}

iree_status_t loom_run_hal_binding_list_initialize_count(
    iree_host_size_t count, iree_allocator_t allocator,
    loom_run_hal_binding_list_t* out_list) {
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_initialize_capacity(
      count, allocator, out_list));
  out_list->count = count;
  return iree_ok_status();
}

static void loom_run_hal_binding_clone_retain(
    const iree_tooling_buffer_binding_t* source,
    iree_tooling_buffer_binding_t* target) {
  *target = *source;
  iree_hal_buffer_retain(target->buffer);
  iree_hal_buffer_view_retain(target->buffer_view);
}

iree_status_t loom_run_hal_binding_list_clone(
    const loom_run_hal_binding_list_t* source, iree_allocator_t allocator,
    loom_run_hal_binding_list_t* out_list) {
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_initialize_count(
      source->count, allocator, out_list));
  for (iree_host_size_t i = 0; i < source->count; ++i) {
    loom_run_hal_binding_clone_retain(&source->values[i], &out_list->values[i]);
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_artifact_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_device_artifact_t* artifact, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_hal_executable) {
  *out_hal_executable = NULL;
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }
  if (artifact->executable_target == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL artifact was not emitted for the active device target");
  }
  if (artifact->artifact == NULL ||
      artifact->artifact->executable_data == NULL ||
      iree_byte_sequence_length(artifact->artifact->executable_data) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL artifact has no executable contents");
  }

  iree_byte_span_t executable_data = iree_byte_span_empty();
  iree_status_t status = iree_byte_sequence_clone(
      artifact->artifact->executable_data, host_allocator, &executable_data);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_load_params_t load_params;
    iree_hal_executable_load_params_initialize(&load_params);
    load_params.executable_data = iree_const_cast_byte_span(executable_data);
    status = iree_hal_device_load_executable(
        runtime->device, IREE_HAL_QUEUE_AFFINITY_ANY,
        artifact->executable_target, &load_params, out_hal_executable);
  }
  iree_allocator_free(host_allocator, executable_data.data);
  return status;
}

iree_status_t loom_run_hal_prepared_candidate_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_device_artifact_t* artifact, iree_allocator_t host_allocator,
    loom_run_hal_prepared_candidate_t* out_candidate) {
  loom_run_hal_prepared_candidate_initialize(out_candidate);
  iree_status_t status = loom_run_hal_artifact_prepare(
      runtime, artifact, host_allocator, &out_candidate->executable);
  if (iree_status_is_ok(status)) {
    out_candidate->target_bundle = artifact->artifact->target_bundle;
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_prepared_candidate_deinitialize(out_candidate);
  }
  return status;
}

static iree_status_t loom_run_hal_binding_refs_from_list(
    const loom_run_hal_binding_list_t* binding_list,
    iree_hal_buffer_ref_t* binding_refs,
    iree_host_size_t binding_ref_capacity) {
  const iree_host_size_t binding_count = binding_list->count;
  if (binding_count > binding_ref_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL binding count %" PRIhsz
                            " exceeds capacity %" PRIhsz,
                            binding_count, binding_ref_capacity);
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    const iree_tooling_buffer_binding_t* binding = &binding_list->values[i];
    if (binding->buffer == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL binding %" PRIhsz " has no buffer", i);
    }
    binding_refs[i] = iree_hal_make_buffer_ref(
        binding->buffer, binding->byte_offset, binding->byte_length);
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_binding_list_total_byte_length(
    const loom_run_hal_binding_list_t* binding_list,
    uint64_t* out_byte_length) {
  *out_byte_length = 0;
  const iree_host_size_t binding_count = binding_list->count;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    const iree_tooling_buffer_binding_t* binding = &binding_list->values[i];
    if (binding->buffer == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL binding %" PRIhsz " has no buffer", i);
    }
    if (UINT64_MAX - *out_byte_length < binding->byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL binding byte count overflowed uint64");
    }
    *out_byte_length += binding->byte_length;
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

static iree_status_t loom_run_hal_record_dispatch_sequence_edge(
    iree_hal_command_buffer_t* command_buffer) {
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  return iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL);
}

static iree_status_t loom_run_hal_record_dispatch_batch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_host_size_t binding_list_count,
    const loom_run_hal_binding_list_t* binding_lists,
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
    const loom_run_hal_binding_list_t* binding_list =
        &binding_lists[binding_list_index];
    iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
    status = loom_run_hal_binding_refs_from_list(binding_list, binding_refs,
                                                 IREE_ARRAYSIZE(binding_refs));
    if (!iree_status_is_ok(status)) {
      break;
    }
    iree_hal_buffer_ref_list_t bindings = {
        .count = binding_list->count,
        .values = binding_refs,
    };
    status = iree_hal_command_buffer_dispatch(
        command_buffer, executable, function, config, constants, bindings,
        IREE_HAL_DISPATCH_FLAG_NONE);
    if (iree_status_is_ok(status) && i + 1 < batch_options->dispatch_count) {
      status = loom_run_hal_record_dispatch_sequence_edge(command_buffer);
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

static iree_status_t loom_run_hal_record_dispatch_sequence_batch(
    iree_hal_device_t* device, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    const iree_host_size_t* execution_epochs, iree_host_size_t plan_ring_count,
    const loom_run_hal_invocation_plan_t* const* plans,
    const loom_run_hal_binding_list_t* binding_lists,
    iree_host_size_t plan_ring_offset,
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
      const loom_run_hal_binding_list_t* binding_list =
          &binding_lists[plan_index];
      iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
      status = loom_run_hal_binding_refs_from_list(
          binding_list, binding_refs, IREE_ARRAYSIZE(binding_refs));
      if (!iree_status_is_ok(status)) {
        break;
      }
      iree_hal_buffer_ref_list_t bindings = {
          .count = binding_list->count,
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
      const bool is_terminal_dispatch =
          batch_index + 1 == batch_options->dispatch_count &&
          step_index + 1 == sequence_count;
      const bool is_sequence_boundary = step_index + 1 == sequence_count;
      const bool is_epoch_boundary =
          !is_sequence_boundary &&
          execution_epochs[step_index] != execution_epochs[step_index + 1];
      if (iree_status_is_ok(status) && !is_terminal_dispatch &&
          (is_sequence_boundary || is_epoch_boundary)) {
        status = loom_run_hal_record_dispatch_sequence_edge(command_buffer);
      }
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

static iree_status_t loom_run_hal_record_indirect_dispatch_sequence(
    iree_hal_device_t* device, iree_host_size_t step_count,
    const loom_run_hal_dispatch_sequence_step_t* steps,
    iree_host_size_t binding_count,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_count, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  uint32_t binding_slot = 0;
  for (iree_host_size_t step_index = 0;
       iree_status_is_ok(status) && step_index < step_count; ++step_index) {
    const loom_run_hal_dispatch_sequence_step_t* step = &steps[step_index];
    iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
    for (iree_host_size_t binding_index = 0;
         binding_index < step->binding_count; ++binding_index) {
      binding_refs[binding_index] = iree_hal_make_indirect_buffer_ref(
          binding_slot++, /*offset=*/0, step->binding_lengths[binding_index]);
    }
    const iree_hal_buffer_ref_list_t bindings = {
        .count = step->binding_count,
        .values = binding_refs,
    };
    const loom_run_hal_invocation_options_t* options = &step->options;
    iree_hal_executable_function_t function =
        iree_hal_executable_function_invalid();
    status = loom_run_hal_lookup_dispatch_function(step->candidate->executable,
                                                   options, &function);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const iree_hal_dispatch_config_t config =
        iree_hal_make_static_dispatch_config(options->workgroup_count[0],
                                             options->workgroup_count[1],
                                             options->workgroup_count[2]);
    status = iree_hal_command_buffer_dispatch(
        command_buffer, step->candidate->executable, function, config,
        loom_run_hal_dispatch_constants(options), bindings,
        IREE_HAL_DISPATCH_FLAG_NONE);
    if (iree_status_is_ok(status) && step_index + 1 < step_count &&
        step->execution_epoch != steps[step_index + 1].execution_epoch) {
      status = loom_run_hal_record_dispatch_sequence_edge(command_buffer);
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

static iree_status_t loom_run_hal_queue_dispatch_prepare_options(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    const loom_run_hal_invocation_options_t* options,
    iree_host_size_t binding_count,
    loom_run_hal_queue_dispatch_t* out_dispatch) {
  loom_run_hal_queue_dispatch_initialize(out_dispatch);
  if (device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL device is not initialized");
  }
  if (executable == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL queue dispatch requires an executable");
  }
  if (options->constant_count > LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HAL dispatch constant count %" PRIhsz " exceeds maximum %d",
        options->constant_count, LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
  }
  if (binding_count > LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL binding count %" PRIhsz " exceeds maximum %d",
                            binding_count, LOOM_RUN_HAL_MAX_BINDING_COUNT);
  }

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_RETURN_IF_ERROR(
      loom_run_hal_lookup_dispatch_function(executable, options, &function));

  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  out_dispatch->executable = executable;
  iree_hal_executable_retain(out_dispatch->executable);
  out_dispatch->function = function;
  out_dispatch->config = iree_hal_make_static_dispatch_config(
      options->workgroup_count[0], options->workgroup_count[1],
      options->workgroup_count[2]);
  memcpy(out_dispatch->constants, options->constants,
         options->constant_count * sizeof(options->constants[0]));
  out_dispatch->constant_count = options->constant_count;
  out_dispatch->binding_count = binding_count;
  out_dispatch->semaphore = semaphore;
  out_dispatch->next_signal_value = 1;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_queue_dispatch_execute_on_device(
    iree_hal_device_t* device, loom_run_hal_queue_dispatch_t* dispatch,
    const loom_run_hal_binding_list_t* binding_list) {
  if (device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL device is not initialized");
  }
  if (dispatch->executable == NULL || dispatch->semaphore == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL queue dispatch is not prepared");
  }
  if (binding_list->count != dispatch->binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL queue dispatch binding count %" PRIhsz
                            " must match prepared binding count %" PRIhsz,
                            binding_list->count, dispatch->binding_count);
  }

  iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_refs_from_list(
      binding_list, binding_refs, IREE_ARRAYSIZE(binding_refs)));
  const iree_hal_buffer_ref_list_t bindings = {
      .count = binding_list->count,
      .values = binding_refs,
  };
  const iree_const_byte_span_t constants = iree_make_const_byte_span(
      dispatch->constants,
      dispatch->constant_count * sizeof(dispatch->constants[0]));
  uint64_t signal_value = dispatch->next_signal_value;
  const iree_hal_semaphore_list_t wait_semaphores =
      iree_hal_semaphore_list_empty();
  const iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &dispatch->semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t status = iree_hal_device_queue_dispatch(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores, signal_semaphores,
      dispatch->executable, dispatch->function, dispatch->config, constants,
      bindings, IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(dispatch->semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(device, IREE_HAL_QUEUE_AFFINITY_ANY);
  }
  if (iree_status_is_ok(status)) {
    ++dispatch->next_signal_value;
  }
  return status;
}

iree_status_t loom_run_hal_dispatch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    const loom_run_hal_binding_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options) {
  loom_run_hal_queue_dispatch_t dispatch = {0};
  iree_status_t status = loom_run_hal_queue_dispatch_prepare_options(
      device, executable, options, binding_list->count, &dispatch);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_queue_dispatch_execute_on_device(device, &dispatch,
                                                           binding_list);
  }
  loom_run_hal_queue_dispatch_deinitialize(&dispatch);
  return status;
}

iree_status_t loom_run_hal_invocation_execute(
    const loom_run_hal_runtime_t* runtime,
    const loom_device_artifact_t* artifact,
    const loom_run_hal_binding_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options,
    iree_allocator_t host_allocator) {
  loom_run_hal_prepared_candidate_t candidate = {0};
  iree_status_t status = loom_run_hal_prepared_candidate_prepare(
      runtime, artifact, host_allocator, &candidate);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch(runtime->device, candidate.executable,
                                   binding_list, options);
  }
  loom_run_hal_prepared_candidate_deinitialize(&candidate);
  return status;
}

iree_status_t loom_run_hal_transfer_bindings_to_host(
    const loom_run_hal_runtime_t* runtime,
    loom_run_hal_binding_list_t* binding_list) {
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_hal_buffer_params_t host_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
  iree_hal_allocator_t* device_allocator =
      iree_hal_device_allocator(runtime->device);
  for (iree_host_size_t i = 0; i < binding_list->count; ++i) {
    iree_tooling_buffer_binding_t* binding = &binding_list->values[i];
    if (binding->buffer == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL binding %" PRIhsz " has no buffer", i);
    }

    iree_hal_buffer_t* host_buffer = NULL;
    iree_status_t status = iree_hal_allocator_allocate_buffer(
        device_allocator, host_params, binding->byte_length, &host_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_transfer_d2d(
          runtime->device, binding->buffer, binding->byte_offset, host_buffer,
          /*target_offset=*/0, binding->byte_length,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    }

    iree_hal_buffer_view_t* host_buffer_view = NULL;
    if (iree_status_is_ok(status) && binding->buffer_view != NULL) {
      status = iree_hal_buffer_view_create_like(
          host_buffer, binding->buffer_view,
          iree_hal_allocator_host_allocator(device_allocator),
          &host_buffer_view);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_release(binding->buffer);
      iree_hal_buffer_view_release(binding->buffer_view);
      binding->buffer = host_buffer;
      binding->buffer_view = host_buffer_view;
      binding->byte_offset = 0;
      binding->byte_length =
          host_buffer_view != NULL
              ? iree_hal_buffer_view_byte_length(host_buffer_view)
              : iree_hal_buffer_byte_length(host_buffer);
    } else {
      iree_hal_buffer_view_release(host_buffer_view);
      iree_hal_buffer_release(host_buffer);
      return status;
    }
  }
  return iree_ok_status();
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
  if (specs->count > 0 && specs->values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "%.*s binding specs require values",
        (int)binding_list_name.size, binding_list_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_parse_binding_specs(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_binding_specs_t* specs,
    iree_hal_allocator_t* device_allocator, iree_allocator_t allocator,
    loom_run_hal_binding_list_t* out_list) {
  IREE_RETURN_IF_ERROR(
      loom_run_hal_binding_specs_validate(specs, IREE_SV("HAL")));
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_initialize_count(
      specs->count, allocator, out_list));
  iree_tooling_value_io_context_t* context = NULL;
  iree_status_t status =
      iree_tooling_value_io_context_allocate(allocator, &context);
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < specs->count;
       ++i) {
    status = iree_tooling_buffer_binding_spec_parse(
        context, specs->values[i], runtime->device, device_allocator,
        &out_list->values[i]);
  }
  iree_tooling_value_io_context_free(context);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_binding_list_deinitialize(out_list);
  }
  return status;
}

static bool loom_run_hal_buffer_view_metadata_equal(
    iree_hal_buffer_view_t* expected_view,
    iree_hal_buffer_view_t* actual_view) {
  if (iree_hal_buffer_view_element_type(expected_view) !=
          iree_hal_buffer_view_element_type(actual_view) ||
      iree_hal_buffer_view_encoding_type(expected_view) !=
          iree_hal_buffer_view_encoding_type(actual_view) ||
      iree_hal_buffer_view_shape_rank(expected_view) !=
          iree_hal_buffer_view_shape_rank(actual_view)) {
    return false;
  }
  const iree_host_size_t rank = iree_hal_buffer_view_shape_rank(expected_view);
  const iree_hal_dim_t* expected_dims =
      iree_hal_buffer_view_shape_dims(expected_view);
  const iree_hal_dim_t* actual_dims =
      iree_hal_buffer_view_shape_dims(actual_view);
  for (iree_host_size_t i = 0; i < rank; ++i) {
    if (expected_dims[i] != actual_dims[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_run_hal_compare_binding_bytes(
    iree_host_size_t binding_ordinal,
    const iree_tooling_buffer_binding_t* expected,
    const iree_tooling_buffer_binding_t* actual, bool* inout_did_match,
    iree_string_builder_t* builder) {
  if (expected->byte_length != actual->byte_length) {
    *inout_did_match = false;
    return iree_string_builder_append_format(
        builder,
        "[FAILED] binding[%" PRIhsz "]: expected %" PRIu64
        " bytes but got %" PRIu64 " bytes.\n",
        binding_ordinal, (uint64_t)expected->byte_length,
        (uint64_t)actual->byte_length);
  }
  if (expected->buffer_view != NULL && actual->buffer_view != NULL &&
      !loom_run_hal_buffer_view_metadata_equal(expected->buffer_view,
                                               actual->buffer_view)) {
    *inout_did_match = false;
    return iree_string_builder_append_format(
        builder,
        "[FAILED] binding[%" PRIhsz "]: buffer view metadata does not match.\n",
        binding_ordinal);
  }
  if (expected->byte_length == 0) {
    return iree_ok_status();
  }

  iree_hal_buffer_mapping_t expected_mapping = {{0}};
  iree_hal_buffer_mapping_t actual_mapping = {{0}};
  bool expected_mapping_active = false;
  bool actual_mapping_active = false;
  iree_status_t status = iree_hal_buffer_map_range(
      expected->buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, expected->byte_offset, expected->byte_length,
      &expected_mapping);
  if (iree_status_is_ok(status)) {
    expected_mapping_active = true;
    status = iree_hal_buffer_map_range(
        actual->buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, actual->byte_offset, actual->byte_length,
        &actual_mapping);
    actual_mapping_active = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status) &&
      memcmp(expected_mapping.contents.data, actual_mapping.contents.data,
             expected->byte_length) != 0) {
    *inout_did_match = false;
    status = iree_string_builder_append_format(
        builder, "[FAILED] binding[%" PRIhsz "]: contents differ.\n",
        binding_ordinal);
  }
  if (actual_mapping_active) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&actual_mapping));
  }
  if (expected_mapping_active) {
    status = iree_status_join(status,
                              iree_hal_buffer_unmap_range(&expected_mapping));
  }
  return status;
}

static iree_status_t loom_run_hal_format_binding(
    iree_host_size_t binding_ordinal,
    const iree_tooling_buffer_binding_t* binding,
    iree_host_size_t max_output_element_count, iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "binding[%" PRIhsz "]: ", binding_ordinal));
  if (binding->buffer_view != NULL) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "hal.buffer_view\n"));
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_append_to_builder(
        binding->buffer_view, max_output_element_count, builder));
    return iree_string_builder_append_cstring(builder, "\n");
  }
  return iree_string_builder_append_format(builder,
                                           "hal.buffer\n%" PRIu64 " bytes\n",
                                           (uint64_t)binding->byte_length);
}

static iree_status_t loom_run_hal_process_invocation_bindings(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan,
    loom_run_hal_binding_list_t* binding_list, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  (void)allocator;
  IREE_RETURN_IF_ERROR(
      loom_run_hal_transfer_bindings_to_host(runtime, binding_list));

  const iree_host_size_t max_output_element_count =
      plan->max_output_element_count == 0
          ? LOOM_RUN_HAL_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT
          : plan->max_output_element_count;
  if (!plan->has_expected_bindings) {
    for (iree_host_size_t i = 0; i < binding_list->count; ++i) {
      IREE_RETURN_IF_ERROR(loom_run_hal_format_binding(
          i, &binding_list->values[i], max_output_element_count,
          &result->output));
    }
    return iree_ok_status();
  }

  bool did_match = true;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < plan->expected_bindings.count; ++i) {
    status = loom_run_hal_compare_binding_bytes(
        i, &plan->expected_bindings.values[i], &binding_list->values[i],
        &did_match, &result->output);
  }
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
  if (plan->bindings.count > LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL binding count %" PRIhsz " exceeds maximum %d",
                            plan->bindings.count,
                            LOOM_RUN_HAL_MAX_BINDING_COUNT);
  }
  if (plan->has_expected_bindings &&
      plan->expected_bindings.count != plan->bindings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected HAL binding count %" PRIhsz
                            " must match input binding count %" PRIhsz,
                            plan->expected_bindings.count,
                            plan->bindings.count);
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

iree_status_t loom_run_hal_queue_dispatch_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    loom_run_hal_queue_dispatch_t* out_dispatch) {
  loom_run_hal_queue_dispatch_initialize(out_dispatch);
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
  return loom_run_hal_queue_dispatch_prepare_options(
      runtime->device, candidate->executable, &plan->options,
      plan->bindings.count, out_dispatch);
}

iree_status_t loom_run_hal_queue_dispatch_execute(
    const loom_run_hal_runtime_t* runtime,
    loom_run_hal_queue_dispatch_t* dispatch,
    const loom_run_hal_binding_list_t* binding_list) {
  return loom_run_hal_queue_dispatch_execute_on_device(runtime->device,
                                                       dispatch, binding_list);
}

iree_status_t loom_run_hal_dispatch_batch_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_dispatch_batch_options_t* batch_options,
    iree_allocator_t allocator, loom_run_hal_dispatch_batch_t* out_batch) {
  return loom_run_hal_dispatch_batch_prepare_from_binding_ring(
      runtime, candidate, plan, /*binding_list_count=*/1, &plan->bindings,
      /*binding_list_offset=*/0, batch_options, allocator, out_batch);
}

static iree_status_t loom_run_hal_dispatch_binding_ring_validate(
    const loom_run_hal_invocation_plan_t* plan,
    iree_host_size_t binding_list_count,
    const loom_run_hal_binding_list_t* binding_lists) {
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
  const iree_host_size_t plan_binding_count = plan->bindings.count;
  for (iree_host_size_t i = 0; i < binding_list_count; ++i) {
    const iree_host_size_t binding_count = binding_lists[i].count;
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
    iree_host_size_t binding_list_count,
    const loom_run_hal_binding_list_t* binding_lists,
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
    memset(out_batch->binding_lists, 0,
           binding_list_count * sizeof(*out_batch->binding_lists));
    out_batch->binding_list_count = binding_list_count;
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < out_batch->binding_list_count; ++i) {
    status = loom_run_hal_binding_list_clone(&binding_lists[i], allocator,
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

static iree_status_t loom_run_hal_dispatch_execution_epochs_validate(
    iree_host_size_t sequence_count, const iree_host_size_t* execution_epochs) {
  if (execution_epochs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch sequence requires execution epochs");
  }
  for (iree_host_size_t step_index = 1; step_index < sequence_count;
       ++step_index) {
    if (execution_epochs[step_index] < execution_epochs[step_index - 1]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL dispatch sequence epoch decreases at step %" PRIhsz, step_index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_dispatch_sequence_plan_ring_validate(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    const iree_host_size_t* execution_epochs, iree_host_size_t plan_ring_count,
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
  IREE_RETURN_IF_ERROR(loom_run_hal_dispatch_execution_epochs_validate(
      sequence_count, execution_epochs));
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
    const iree_host_size_t* execution_epochs, iree_host_size_t plan_ring_count,
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
      runtime, sequence_count, candidates, execution_epochs, plan_ring_count,
      plans));

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
    memset(out_batch->binding_lists, 0,
           plan_count * sizeof(*out_batch->binding_lists));
    out_batch->binding_list_count = plan_count;
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < out_batch->binding_list_count; ++i) {
    status = loom_run_hal_binding_list_clone(&plans[i]->bindings, allocator,
                                             &out_batch->binding_lists[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_record_dispatch_sequence_batch(
        runtime->device, sequence_count, candidates, execution_epochs,
        plan_ring_count, plans, out_batch->binding_lists, plan_ring_offset,
        batch_options, &out_batch->command_buffer);
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

iree_status_t loom_run_hal_dispatch_sequence_prepare(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t step_count,
    const loom_run_hal_dispatch_sequence_step_t* steps,
    loom_run_hal_dispatch_sequence_t* out_sequence) {
  loom_run_hal_dispatch_sequence_initialize(out_sequence);
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }
  if (step_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch sequence must contain a step");
  }

  iree_host_size_t binding_count = 0;
  for (iree_host_size_t step_index = 0; step_index < step_count; ++step_index) {
    IREE_ASSERT(steps[step_index].candidate != NULL);
    IREE_ASSERT(steps[step_index].candidate->executable != NULL);
    IREE_ASSERT(steps[step_index].binding_count == 0 ||
                steps[step_index].binding_lengths != NULL);
    if (step_index != 0 && steps[step_index].execution_epoch <
                               steps[step_index - 1].execution_epoch) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL dispatch sequence epoch decreases at step %" PRIhsz, step_index);
    }
    if (steps[step_index].binding_count > LOOM_RUN_HAL_MAX_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL dispatch sequence step binding count exceeds capacity");
    }
    if (!iree_host_size_checked_add(
            binding_count, steps[step_index].binding_count, &binding_count) ||
        binding_count > UINT32_C(0x00FFFFFF)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL dispatch sequence indirect binding count exceeds slot limits");
    }
  }

  iree_status_t status = loom_run_hal_record_indirect_dispatch_sequence(
      runtime->device, step_count, steps, binding_count,
      &out_sequence->command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        runtime->device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &out_sequence->semaphore);
  }
  if (iree_status_is_ok(status)) {
    out_sequence->next_signal_value = 1;
    out_sequence->binding_count = binding_count;
  } else {
    loom_run_hal_dispatch_sequence_deinitialize(out_sequence);
  }
  return status;
}

iree_status_t loom_run_hal_dispatch_sequence_execute(
    const loom_run_hal_runtime_t* runtime,
    loom_run_hal_dispatch_sequence_t* sequence,
    iree_hal_buffer_binding_table_t binding_table) {
  IREE_ASSERT(runtime->device != NULL);
  IREE_ASSERT(sequence->command_buffer != NULL);
  IREE_ASSERT(sequence->semaphore != NULL);
  IREE_ASSERT(binding_table.count == sequence->binding_count);

  uint64_t signal_value = sequence->next_signal_value;
  const iree_hal_semaphore_list_t wait_semaphores =
      iree_hal_semaphore_list_empty();
  const iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &sequence->semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t status = iree_hal_device_queue_execute(
      runtime->device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores,
      signal_semaphores, sequence->command_buffer, binding_table,
      IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(sequence->semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(runtime->device,
                                         IREE_HAL_QUEUE_AFFINITY_ANY);
  }
  if (iree_status_is_ok(status)) {
    ++sequence->next_signal_value;
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
    status = iree_hal_device_queue_flush(runtime->device,
                                         IREE_HAL_QUEUE_AFFINITY_ANY);
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
  loom_run_hal_iteration_t iteration = {0};
  loom_run_hal_iteration_initialize(&iteration);
  iree_status_t status = loom_run_hal_binding_list_clone(
      &batch->binding_lists[0], allocator, &iteration.bindings);
  if (iree_status_is_ok(status)) {
    iteration.has_bindings = true;
    status = loom_run_hal_invocation_collect_results(runtime, plan, &iteration,
                                                     allocator, result);
  }
  loom_run_hal_iteration_deinitialize(&iteration);
  return status;
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
    out_plan->has_expected_bindings = expected_bindings->count != 0;
    out_plan->max_output_element_count = max_output_element_count;
  } else {
    loom_run_hal_invocation_plan_deinitialize(out_plan);
  }
  return status;
}

iree_status_t loom_run_hal_invocation_plan_prepare_from_lists(
    const loom_run_hal_invocation_options_t* options,
    const loom_run_hal_binding_list_t* bindings,
    const loom_run_hal_binding_list_t* expected_bindings,
    iree_host_size_t max_output_element_count, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan) {
  loom_run_hal_invocation_plan_initialize(out_plan);
  out_plan->options = *options;
  out_plan->has_expected_bindings = expected_bindings != NULL;
  out_plan->max_output_element_count = max_output_element_count;
  iree_status_t status =
      loom_run_hal_binding_list_clone(bindings, allocator, &out_plan->bindings);
  if (iree_status_is_ok(status) && expected_bindings != NULL) {
    status = loom_run_hal_binding_list_clone(expected_bindings, allocator,
                                             &out_plan->expected_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_plan_validate(out_plan);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_invocation_plan_deinitialize(out_plan);
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

  iree_status_t status = loom_run_hal_binding_list_clone(
      &plan->bindings, allocator, &out_iteration->bindings);
  if (iree_status_is_ok(status)) {
    out_iteration->has_bindings = true;
    status = loom_run_hal_dispatch(runtime->device, candidate->executable,
                                   &out_iteration->bindings, &plan->options);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_iteration_deinitialize(out_iteration);
  }
  return status;
}

iree_status_t loom_run_hal_invocation_collect_results(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan,
    loom_run_hal_iteration_t* iteration, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  if (!iteration->has_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL iteration requires bindings");
  }
  if (iteration->bindings.count != plan->bindings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL iteration binding count %" PRIhsz
                            " must match plan binding count %" PRIhsz,
                            iteration->bindings.count, plan->bindings.count);
  }
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }
  return loom_run_hal_process_invocation_bindings(
      runtime, plan, &iteration->bindings, allocator, result);
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
    const loom_device_artifact_t* artifact,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  loom_run_hal_prepared_candidate_t candidate = {0};
  iree_status_t status = loom_run_hal_prepared_candidate_prepare(
      runtime, artifact, allocator, &candidate);
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
