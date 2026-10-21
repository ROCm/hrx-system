// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/submission.h"

static iree_status_t iree_hal_amd_gpu_aql_submission_allocate(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t inline_storage,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    iree_host_size_t wait_count, iree_hal_semaphore_list_t signals,
    bool deferred, iree_hal_amd_gpu_aql_submission_t** out_submission) {
  *out_submission = NULL;
  iree_host_size_t reference_count = 0;
  const iree_host_size_t waits_offset =
      sizeof(iree_hal_amd_gpu_aql_submission_t) +
      (profile ? sizeof(*profile) : 0);
  iree_host_size_t payload_offset = waits_offset;
  iree_host_size_t size = 0;
  if (!iree_host_size_checked_add(
          request->bindings.count,
          iree_hal_dispatch_uses_indirect_parameters(request->flags),
          &reference_count) ||
      !iree_host_size_checked_mul_add(payload_offset, wait_count,
                                      sizeof(iree_hal_amd_gpu_aql_wait_t),
                                      &payload_offset) ||
      !iree_host_size_checked_mul_add(
          payload_offset, signals.count,
          sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t), &payload_offset) ||
      !iree_host_size_checked_align(payload_offset, 8, &payload_offset) ||
      !iree_host_size_checked_mul_add(payload_offset, reference_count,
                                      sizeof(iree_hal_buffer_ref_t), &size) ||
      (deferred &&
       (!iree_host_size_checked_mul_add(size, request->bindings.count,
                                        sizeof(uint64_t), &size) ||
        !iree_host_size_checked_add(size, request->constants.data_length,
                                    &size)))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AQL submission capture overflows");
  }
  iree_byte_span_t storage = inline_storage;
  if (size <= storage.data_length) {
    pool = NULL;
  } else {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_capture_pool_acquire(pool, size, &storage));
  }
  iree_hal_amd_gpu_aql_submission_t* submission = (void*)storage.data;
  *submission = (iree_hal_amd_gpu_aql_submission_t){
      .entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_SUBMISSION,
      .pool = pool,
      .allocation_size = storage.data_length,
      .capture_size = size,
      .wait_count = wait_count,
      .waits = (void*)(storage.data + waits_offset),
      .profiled = profile != NULL,
      .dispatch = *request,
  };
  if (profile) {
    *iree_hal_amd_gpu_aql_submission_profile(submission) = *profile;
  }
  uint8_t* cursor = (uint8_t*)(submission->waits + wait_count);
  submission->signals = (iree_hal_semaphore_list_t){
      .count = signals.count,
      .payload_values = (uint64_t*)cursor,
      .semaphores =
          (iree_hal_semaphore_t**)(cursor + signals.count * sizeof(uint64_t)),
  };
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    submission->signals.semaphores[i] = signals.semaphores[i];
    submission->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  iree_hal_buffer_ref_t* references = (void*)(storage.data + payload_offset);
  for (iree_host_size_t i = 0; i < request->bindings.count; ++i) {
    references[i] = request->bindings.values[i];
  }
  if (iree_hal_dispatch_uses_indirect_parameters(request->flags)) {
    references[request->bindings.count] = request->parameters;
  }
  submission->dispatch.bindings.values = references;
  if (!(request->flags & IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES)) {
    iree_hal_executable_retain(request->executable);
    for (iree_host_size_t i = 0; i < reference_count; ++i) {
      iree_hal_buffer_retain(references[i].buffer);
    }
  }
  submission->dispatch.constants = iree_const_byte_span_empty();
  if (deferred) {
    cursor = (uint8_t*)(references + reference_count) +
             request->bindings.count * sizeof(uint64_t);
    submission->dispatch.constants =
        iree_make_const_byte_span(cursor, request->constants.data_length);
    if (request->constants.data_length) {
      memcpy(cursor, request->constants.data, request->constants.data_length);
    }
  }
  *out_submission = submission;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_aql_submission_capture(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    uint32_t before, uint32_t after,
    iree_hal_amd_gpu_aql_submission_t** out_submission) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_allocate(
      pool, iree_byte_span_empty(), profile, request, waits.count, signals,
      true, out_submission));
  iree_hal_amd_gpu_aql_submission_t* submission = *out_submission;
  submission->before = before;
  submission->after = after;
  submission->native_wait_count = waits.count ? UINT32_MAX : 0;
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    submission->waits[i] = (iree_hal_amd_gpu_aql_wait_t){
        .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_aql_submission_retirement(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t inline_storage,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    const iree_hal_amd_gpu_aql_dispatch_request_t* request,
    iree_host_size_t wait_count, const iree_hal_amd_gpu_aql_wait_t* waits,
    uint32_t native_wait_count, iree_hal_semaphore_list_t signals,
    iree_hal_amd_gpu_aql_submission_t** out_submission) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_allocate(
      pool, inline_storage, profile, request, native_wait_count, signals, false,
      out_submission));
  iree_hal_amd_gpu_aql_submission_t* submission = *out_submission;
  submission->native_wait_count = native_wait_count;
  for (iree_host_size_t i = 0, target = 0; i < wait_count; ++i) {
    if (waits[i].signal) {
      submission->waits[target++] =
          (iree_hal_amd_gpu_aql_wait_t){.signal = waits[i].signal};
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_aql_submission_acquire(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    iree_hal_amd_gpu_aql_entry_kind_t kind, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t payload_length,
    iree_hal_amd_gpu_aql_submission_t** out_submission, void** out_payload) {
  *out_submission = NULL;
  iree_host_size_t size = 0, waits_offset = 0, signals_offset = 0;
  iree_host_size_t values_offset = 0, payload_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_aql_submission_t) +
          (profile ? sizeof(*profile) : 0),
      &size,
      IREE_STRUCT_FIELD(waits.count, iree_hal_amd_gpu_aql_wait_t,
                        &waits_offset),
      IREE_STRUCT_FIELD(signals.count, iree_hal_semaphore_t*, &signals_offset),
      IREE_STRUCT_FIELD(signals.count, uint64_t, &values_offset),
      IREE_STRUCT_FIELD(payload_length, uint8_t, &payload_offset)));
  iree_byte_span_t storage;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_capture_pool_acquire(pool, size, &storage));
  iree_hal_amd_gpu_aql_submission_t* submission = (void*)storage.data;
  *submission = (iree_hal_amd_gpu_aql_submission_t){
      .entry.kind = kind,
      .pool = pool,
      .allocation_size = storage.data_length,
      .capture_size = size,
      .wait_count = waits.count,
      .waits = (void*)(storage.data + waits_offset),
      .native_wait_count = waits.count ? UINT32_MAX : 0,
      .profiled = profile != NULL,
      .signals = {signals.count, (void*)(storage.data + signals_offset),
                  (void*)(storage.data + values_offset)},
  };
  if (profile) {
    *iree_hal_amd_gpu_aql_submission_profile(submission) = *profile;
  }
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    submission->waits[i] = (iree_hal_amd_gpu_aql_wait_t){
        .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    submission->signals.semaphores[i] = signals.semaphores[i];
    submission->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  *out_submission = submission;
  *out_payload = storage.data + payload_offset;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_aql_submission_execute(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_aql_submission_t** out_submission) {
  *out_submission = NULL;
  iree_host_size_t payload_length = 0;
  if (!iree_host_size_checked_mul(command_buffer->binding_count,
                                  sizeof(iree_hal_amd_gpu_command_binding_t),
                                  &payload_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AQL binding capture overflows");
  }
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_acquire(
      pool, profile, IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE, waits, signals,
      payload_length, &submission, &payload));
  iree_status_t status = iree_hal_amd_gpu_aql_command_invocation_initialize(
      command_buffer, bindings, payload, flags, &submission->execute);
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    iree_hal_amd_gpu_aql_submission_release(submission);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_aql_submission_control(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    const iree_hal_queue_family_t* family, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* target,
    iree_device_size_t offset, iree_hal_amd_gpu_control_t command,
    uint32_t before, uint32_t after,
    iree_hal_amd_gpu_aql_submission_t** out_submission) {
  *out_submission = NULL;
  iree_hal_amd_gpu_buffer_reference_t reference;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_buffer_reference_initialize(
      family, IREE_HAL_MEMORY_SITE_QUEUE, target, offset, &reference));
  const uint32_t required = iree_hal_amd_gpu_control_requirement(command);
  if (required) {
    const bool incompatible = command.target_error_mode ==
                              IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_control_validate_target(
        family, target, incompatible ? 0 : required,
        incompatible ? required : 0));
  }
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_acquire(
      pool, profile, IREE_HAL_AMD_GPU_AQL_ENTRY_CONTROL, waits, signals, 0,
      &submission, &payload));
  iree_hal_amd_gpu_aql_barrier_atomic(
      command.flags,
      (command.flags & IREE_HAL_ATOMIC_FLAG_RELEASE)
          ? IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE
          : 0,
      (command.flags & IREE_HAL_ATOMIC_FLAG_ACQUIRE)
          ? IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE
          : 0,
      &before, &after);
  submission->before = before;
  submission->after = after;
  submission->control.target = reference;
  submission->control.command = command;
  iree_hal_buffer_retain(target);
  *out_submission = submission;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_aql_submission_transfer(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    const iree_hal_queue_family_t* family, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations, uint32_t before,
    uint32_t after, iree_hal_amd_gpu_aql_submission_t** out_submission) {
  *out_submission = NULL;
  iree_hal_amd_gpu_transfer_layout_t layout;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_transfer_measure(count, operations, &layout));
  iree_hal_amd_gpu_aql_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_aql_submission_acquire(
      pool, profile, IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER, waits, signals,
      layout.byte_length, &submission, &payload));
  submission->before = before;
  submission->after = after;
  iree_atomic_ref_count_init(&submission->transfer.ref_count);
  iree_status_t status = iree_hal_amd_gpu_transfer_initialize(
      family, IREE_HAL_MEMORY_SITE_PROGRAM,
      IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_STAGING_LENGTH, count, operations,
      layout, payload, &submission->transfer.capture);
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    iree_hal_amd_gpu_aql_submission_release(submission);
  }
  return status;
}

void iree_hal_amd_gpu_aql_submission_reject(
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_code_t code) {
  int32_t expected = IREE_STATUS_OK;
  iree_atomic_compare_exchange_strong(&submission->failure_code, &expected,
                                      code, iree_memory_order_relaxed,
                                      iree_memory_order_relaxed);
}

void iree_hal_amd_gpu_aql_submission_fail(
    iree_hal_amd_gpu_aql_submission_t* submission, iree_status_t status) {
  int32_t expected = IREE_STATUS_OK;
  if (iree_atomic_compare_exchange_strong(
          &submission->failure_code, &expected, iree_status_code(status),
          iree_memory_order_relaxed, iree_memory_order_relaxed) &&
      submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER &&
      submission->transfer.callback.fn) {
    submission->transfer.callback.fn(submission->transfer.callback.user_data,
                                     iree_status_clone(status), NULL, false);
  }
}

void iree_hal_amd_gpu_aql_submission_set_frontier(
    iree_hal_amd_gpu_aql_submission_t* submission,
    const iree_async_single_frontier_t* frontier) {
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER &&
      submission->transfer.callback.fn && frontier) {
    submission->transfer.frontier = *frontier;
  }
}

bool iree_hal_amd_gpu_aql_submission_retirement_ready(
    const iree_hal_amd_gpu_aql_submission_t* submission,
    iree_status_code_t* out_status_code) {
  *out_status_code = IREE_STATUS_OK;
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    if (submission->waits[i].signal) {
      iree_status_code_t code;
      if (!iree_hal_amd_gpu_signal_query_retirement(submission->waits[i].signal,
                                                    &code)) {
        return false;
      }
      if (*out_status_code == IREE_STATUS_OK) {
        *out_status_code = code;
      }
    }
  }
  return true;
}

void iree_hal_amd_gpu_aql_submission_release_payload(
    iree_hal_amd_gpu_aql_submission_t* submission) {
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE) {
    iree_hal_amd_gpu_aql_command_invocation_deinitialize(&submission->execute);
    return;
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
    iree_hal_amd_gpu_transfer_deinitialize(&submission->transfer.capture);
    submission->transfer.capture.count = 0;
    return;
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_CONTROL) {
    iree_hal_buffer_release(submission->control.target.buffer);
    submission->control.target.buffer = NULL;
    return;
  }
  iree_hal_amd_gpu_aql_dispatch_request_t* request = &submission->dispatch;
  if (request->function) {
    if (!(request->flags & IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES)) {
      const iree_host_size_t count =
          request->bindings.count +
          iree_hal_dispatch_uses_indirect_parameters(request->flags);
      for (iree_host_size_t i = 0; i < count; ++i) {
        iree_hal_buffer_release(request->bindings.values[i].buffer);
      }
      iree_hal_executable_release(request->executable);
    }
    request->function = NULL;
  }
}

static void iree_hal_amd_gpu_aql_submission_destroy(
    iree_hal_amd_gpu_aql_submission_t* submission) {
  iree_hal_amd_gpu_aql_submission_release_payload(submission);
  const iree_status_code_t failure = (iree_status_code_t)iree_atomic_load(
      &submission->failure_code, iree_memory_order_relaxed);
  if (failure != IREE_STATUS_OK) {
    iree_hal_semaphore_list_fail(submission->signals,
                                 iree_status_from_code(failure));
  }
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    iree_hal_amd_gpu_signal_release(submission->waits[i].signal);
    iree_hal_semaphore_release(submission->waits[i].semaphore);
  }
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    iree_hal_semaphore_release(submission->signals.semaphores[i]);
  }
  if (submission->pool) {
    iree_hal_amd_gpu_capture_pool_release(
        submission->pool,
        iree_make_byte_span(submission, submission->allocation_size));
  }
}

static void iree_hal_amd_gpu_aql_submission_retire_transfer(
    iree_hal_amd_gpu_aql_submission_t* submission) {
  const iree_hal_amd_gpu_retirement_callback_t callback =
      submission->transfer.callback;
  const iree_async_single_frontier_t frontier = submission->transfer.frontier;
  const iree_status_code_t failure = (iree_status_code_t)iree_atomic_load(
      &submission->failure_code, iree_memory_order_relaxed);
  iree_hal_amd_gpu_aql_submission_destroy(submission);
  // The file owner may release the device and capture pool from its callback.
  // End every submission and pool access before returning those borrows.
  if (callback.fn) {
    callback.fn(callback.user_data, iree_status_from_code(failure),
                frontier.entry_count ? &frontier : NULL, true);
  }
}

void iree_hal_amd_gpu_aql_submission_release(
    iree_hal_amd_gpu_aql_submission_t* submission) {
  if (!submission) {
    return;
  }
  if (submission->entry.kind == IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER) {
    if (iree_atomic_ref_count_dec(&submission->transfer.ref_count) == 1) {
      iree_hal_amd_gpu_aql_submission_retire_transfer(submission);
    }
  } else if (submission->entry.kind != IREE_HAL_AMD_GPU_AQL_ENTRY_EXECUTE ||
             iree_atomic_ref_count_dec(&submission->execute.ref_count) == 1) {
    iree_hal_amd_gpu_aql_submission_destroy(submission);
  }
}
