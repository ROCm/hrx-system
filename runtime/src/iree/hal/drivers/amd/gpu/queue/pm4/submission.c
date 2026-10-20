// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"

// Inline file-only state. Ordinary submissions pay no additional storage.
typedef struct iree_hal_amd_gpu_pm4_submission_retirement_t {
  // Borrowed file owner, released only by the last capture reference.
  iree_hal_amd_gpu_retirement_callback_t callback;
  // First immediate failure reported to the owner, or OK before notification.
  iree_atomic_int32_t status_code;
  // Written by terminal native retirement and read after the reference join.
  iree_async_single_frontier_t frontier;
} iree_hal_amd_gpu_pm4_submission_retirement_t;

static iree_hal_amd_gpu_pm4_submission_retirement_t*
iree_hal_amd_gpu_pm4_submission_retirement(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  return (void*)((uint8_t*)submission + submission->capture_size -
                 sizeof(iree_hal_amd_gpu_pm4_submission_retirement_t));
}

// Captures common ownership and allocates the operation payload in one block.
static iree_status_t iree_hal_amd_gpu_pm4_submission_acquire_storage(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t inline_storage,
    bool profiled, iree_host_size_t wait_count,
    iree_hal_semaphore_list_t signals, iree_host_size_t payload_size,
    iree_hal_amd_gpu_pm4_barrier_t before, iree_hal_amd_gpu_pm4_barrier_t after,
    iree_hal_amd_gpu_pm4_submission_t** out_submission, void** out_payload) {
  const iree_host_size_t header_size =
      sizeof(iree_hal_amd_gpu_pm4_submission_t) +
      (profiled ? sizeof(iree_hal_amd_gpu_profile_operation_t) : 0);
  iree_host_size_t payload_offset = header_size;
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul_add(
          payload_offset, wait_count,
          sizeof(iree_hal_amd_gpu_pm4_submission_wait_t), &payload_offset) ||
      !iree_host_size_checked_mul_add(
          payload_offset, signals.count,
          sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t), &payload_offset) ||
      !iree_host_size_checked_align(payload_offset, 8, &payload_offset) ||
      !iree_host_size_checked_add(payload_offset, payload_size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "submission capture size overflows");
  }
  iree_byte_span_t storage = inline_storage;
  if (total_size <= storage.data_length) {
    pool = NULL;
  } else {
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_capture_pool_acquire(pool, total_size, &storage));
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = (void*)storage.data;
  *submission = (iree_hal_amd_gpu_pm4_submission_t){
      .native_wait_count = wait_count ? UINT32_MAX : 0,
      .pool = pool,
      .allocation_size = storage.data_length,
      .capture_size = total_size,
      .profiled = profiled,
      .wait_count = wait_count,
      .before = before,
      .after = after,
  };
  iree_atomic_ref_count_init(&submission->ref_count);
  uint8_t* cursor = (uint8_t*)submission + header_size;
  submission->waits = (iree_hal_amd_gpu_pm4_submission_wait_t*)cursor;
  cursor += wait_count * sizeof(*submission->waits);
  submission->signals.count = signals.count;
  // Values precede pointers so odd signal counts retain alignment on 32-bit
  // hosts. The payload begins at its independently aligned offset above.
  submission->signals.payload_values = (uint64_t*)cursor;
  cursor += signals.count * sizeof(uint64_t);
  submission->signals.semaphores = (iree_hal_semaphore_t**)cursor;
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    submission->signals.semaphores[i] = signals.semaphores[i];
    submission->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  *out_submission = submission;
  *out_payload = storage.data + payload_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_pm4_submission_acquire(
    iree_hal_amd_gpu_capture_pool_t* pool, bool profiled,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_host_size_t payload_size,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_pm4_submission_t** out_submission, void** out_payload) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire_storage(
      pool, iree_byte_span_empty(), profiled, waits.count, signals,
      payload_size,
      iree_hal_amd_gpu_pm4_barrier_prepare(profile,
                                           barriers ? barriers->before : NULL),
      iree_hal_amd_gpu_pm4_barrier_prepare(profile,
                                           barriers ? barriers->after : NULL),
      out_submission, out_payload));
  iree_hal_amd_gpu_pm4_submission_t* submission = *out_submission;
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    submission->waits[i] = (iree_hal_amd_gpu_pm4_submission_wait_t){
        .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_submission_create(
    iree_hal_amd_gpu_capture_pool_t* pool, bool profiled,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_executable_t* executable,
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_amd_gpu_pm4_dispatch_t dispatch, iree_hal_buffer_ref_t parameters,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_submission_t** out_submission) {
  *out_submission = NULL;
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  iree_host_size_t reference_count = 0;
  iree_host_size_t payload_size = 0;
  if (!iree_host_size_checked_add(bindings.count, indirect_parameters,
                                  &reference_count) ||
      !iree_host_size_checked_mul_add(constants.data_length, reference_count,
                                      sizeof(iree_hal_buffer_ref_t),
                                      &payload_size) ||
      !iree_host_size_checked_mul_add(payload_size, bindings.count,
                                      sizeof(uint64_t), &payload_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch capture size overflows");
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire(
      pool, profiled, waits, signals, payload_size, profile, barriers,
      &submission, &payload));
  submission->entry.kind = function ? IREE_HAL_AMD_GPU_PM4_ENTRY_DISPATCH
                                    : IREE_HAL_AMD_GPU_PM4_ENTRY_BARRIER;
  submission->dispatch.executable = executable;
  submission->dispatch.function = function;
  submission->dispatch.launch = dispatch;
  submission->dispatch.flags = flags;
  iree_hal_buffer_ref_t* references = payload;
  submission->dispatch.bindings =
      (iree_hal_buffer_ref_list_t){bindings.count, references};
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    references[i] = bindings.values[i];
  }
  if (indirect_parameters) {
    references[bindings.count] = parameters;
  }
  if (!(flags & IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES)) {
    iree_hal_executable_retain(executable);
    for (iree_host_size_t i = 0; i < reference_count; ++i) {
      iree_hal_buffer_retain(references[i].buffer);
    }
  }
  uint8_t* cursor = (uint8_t*)(references + reference_count) +
                    bindings.count * sizeof(uint64_t);
  submission->dispatch.constants =
      iree_make_const_byte_span(cursor, constants.data_length);
  if (constants.data_length) {
    memcpy(cursor, constants.data, constants.data_length);
  }
  *out_submission = submission;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_submission_create_retirement(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t inline_storage,
    const iree_hal_amd_gpu_pm4_dispatch_request_t* request,
    iree_host_size_t wait_count,
    const iree_hal_amd_gpu_pm4_submission_wait_t* waits,
    uint32_t native_wait_count, iree_hal_semaphore_list_t signals,
    iree_hal_amd_gpu_pm4_barrier_t before, iree_hal_amd_gpu_pm4_barrier_t after,
    const iree_hal_amd_gpu_profile_operation_t* profile,
    iree_hal_amd_gpu_pm4_submission_t** out_submission) {
  *out_submission = NULL;
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(request->flags);
  // The bounded inline caller has already admitted these counts. This payload
  // retains resources only: neither native address scratch nor constants live
  // beyond packet preparation.
  const iree_host_size_t reference_count =
      request->bindings.count + indirect_parameters;
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire_storage(
      pool, inline_storage, profile != NULL, native_wait_count, signals,
      reference_count * sizeof(iree_hal_buffer_ref_t), before, after,
      &submission, &payload));
  submission->entry.kind = request->function
                               ? IREE_HAL_AMD_GPU_PM4_ENTRY_DISPATCH
                               : IREE_HAL_AMD_GPU_PM4_ENTRY_BARRIER;
  submission->native_wait_count = native_wait_count;
  for (iree_host_size_t i = 0, target = 0; i < wait_count; ++i) {
    if (waits[i].signal) {
      submission->waits[target++] =
          (iree_hal_amd_gpu_pm4_submission_wait_t){.signal = waits[i].signal};
    }
  }
  submission->dispatch.executable = request->executable;
  submission->dispatch.function = request->function;
  submission->dispatch.launch = request->launch;
  submission->dispatch.flags = request->flags;
  iree_hal_buffer_ref_t* references = payload;
  submission->dispatch.bindings =
      (iree_hal_buffer_ref_list_t){request->bindings.count, references};
  for (iree_host_size_t i = 0; i < request->bindings.count; ++i) {
    references[i] = request->bindings.values[i];
  }
  if (indirect_parameters) {
    references[request->bindings.count] = request->parameters;
  }
  if (!(request->flags & IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES)) {
    iree_hal_executable_retain(request->executable);
    for (iree_host_size_t i = 0; i < reference_count; ++i) {
      iree_hal_buffer_retain(references[i].buffer);
    }
  }
  if (profile) {
    *iree_hal_amd_gpu_pm4_submission_profile(submission) = *profile;
  }
  *out_submission = submission;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_submission_create_transfer(
    iree_hal_amd_gpu_capture_pool_t* pool, bool profiled,
    const iree_hal_queue_family_t* family, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_pm4_submission_t** out_submission) {
  *out_submission = NULL;
  iree_hal_amd_gpu_pm4_transfer_layout_t layout;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_pm4_transfer_measure(count, operations, &layout));
  if (!layout.count) {
    return iree_hal_amd_gpu_pm4_submission_create(
        pool, profiled, waits, signals, NULL, NULL,
        (iree_hal_amd_gpu_pm4_dispatch_t){0}, (iree_hal_buffer_ref_t){0},
        iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(), profile,
        barriers, 0, out_submission);
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire(
      pool, profiled, waits, signals, layout.byte_length, profile, barriers,
      &submission, &payload));
  submission->entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER;
  iree_status_t status = iree_hal_amd_gpu_pm4_transfer_initialize(
      family, count, operations, layout, payload, &submission->transfer);
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    iree_hal_amd_gpu_pm4_submission_release(submission);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_pm4_submission_create_file(
    iree_hal_amd_gpu_capture_pool_t* pool,
    const iree_hal_queue_family_t* family,
    const iree_hal_transfer_operation_t* operation,
    const iree_hal_amd_gpu_pm4_barrier_t* before,
    const iree_hal_amd_gpu_pm4_barrier_t* after,
    iree_hal_amd_gpu_retirement_callback_t callback,
    iree_hal_amd_gpu_pm4_submission_t** out_submission) {
  *out_submission = NULL;
  // File copies borrow their captured host span instead of copying UPDATE
  // bytes. One UPLOAD/DOWNLOAD descriptor has a fixed capture size.
  const iree_hal_amd_gpu_pm4_transfer_layout_t layout = {
      .count = operation ? 1 : 0,
      .byte_length =
          operation ? sizeof(iree_hal_amd_gpu_pm4_transfer_operation_t) : 0,
  };
  const iree_host_size_t payload_size =
      iree_host_align(
          layout.byte_length,
          iree_alignof(iree_hal_amd_gpu_pm4_submission_retirement_t)) +
      sizeof(iree_hal_amd_gpu_pm4_submission_retirement_t);
  const iree_hal_amd_gpu_pm4_barrier_profile_t profile = {0};
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire(
      pool, false, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), payload_size, &profile, &barriers,
      &submission, &payload));
  if (before) {
    submission->before = *before;
  }
  if (after) {
    submission->after = *after;
  }
  iree_status_t status = iree_ok_status();
  if (layout.count) {
    submission->entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER;
    status = iree_hal_amd_gpu_pm4_transfer_initialize(
        family, 1, operation, layout, payload, &submission->transfer);
  }
  if (iree_status_is_ok(status)) {
    *iree_hal_amd_gpu_pm4_submission_retirement(submission) =
        (iree_hal_amd_gpu_pm4_submission_retirement_t){.callback = callback};
    submission->has_retirement_callback = true;
    *out_submission = submission;
  } else {
    iree_hal_amd_gpu_pm4_submission_release(submission);
  }
  return status;
}

void iree_hal_amd_gpu_pm4_submission_reject(
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_code_t code) {
  int32_t expected = IREE_STATUS_OK;
  iree_atomic_compare_exchange_strong(&submission->failure_code, &expected,
                                      code, iree_memory_order_relaxed,
                                      iree_memory_order_relaxed);
}

void iree_hal_amd_gpu_pm4_submission_fail(
    iree_hal_amd_gpu_pm4_submission_t* submission, iree_status_t status) {
  iree_hal_amd_gpu_pm4_submission_reject(submission, iree_status_code(status));
  if (!submission->has_retirement_callback) {
    return;
  }
  iree_hal_amd_gpu_pm4_submission_retirement_t* retirement =
      iree_hal_amd_gpu_pm4_submission_retirement(submission);
  const iree_status_code_t code = (iree_status_code_t)iree_atomic_load(
      &submission->failure_code, iree_memory_order_relaxed);
  int32_t expected = IREE_STATUS_OK;
  if (iree_atomic_compare_exchange_strong(&retirement->status_code, &expected,
                                          code, iree_memory_order_relaxed,
                                          iree_memory_order_relaxed)) {
    retirement->callback.fn(retirement->callback.user_data,
                            code == iree_status_code(status)
                                ? iree_status_clone(status)
                                : iree_status_from_code(code),
                            NULL, false);
  }
}

void iree_hal_amd_gpu_pm4_submission_set_frontier(
    iree_hal_amd_gpu_pm4_submission_t* submission,
    const iree_async_single_frontier_t* frontier) {
  if (submission->has_retirement_callback && frontier) {
    iree_hal_amd_gpu_pm4_submission_retirement(submission)->frontier =
        *frontier;
  }
}

iree_status_t iree_hal_amd_gpu_pm4_submission_create_execute(
    iree_hal_amd_gpu_capture_pool_t* pool, bool profiled,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_pm4_submission_t** out_submission) {
  *out_submission = NULL;
  iree_host_size_t payload_size = 0;
  if (!iree_host_size_checked_mul_add(
          0, command_buffer->binding_count,
          sizeof(iree_hal_amd_gpu_pm4_command_binding_t), &payload_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recorded invocation binding table overflows");
  }
  const iree_hal_barrier_list_t empty = {0};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire(
      pool, profiled, waits, signals, payload_size, profile, &barriers,
      &submission, &payload));
  submission->entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE;
  iree_status_t status = iree_hal_amd_gpu_pm4_command_invocation_initialize(
      command_buffer, bindings, payload, flags, &submission->execute);
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    iree_hal_amd_gpu_pm4_submission_release(submission);
  }
  return status;
}

void iree_hal_amd_gpu_pm4_submission_profile_transfer(
    iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal) {
  iree_hal_amd_gpu_profile_operation_t* operation =
      iree_hal_amd_gpu_pm4_submission_profile(submission);
  const iree_host_size_t count =
      submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER
          ? submission->transfer.count
          : 0;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile, queue_ordinal,
      count ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TRANSFER
            : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER,
      submission->wait_count, submission->signals.count, count, operation);
  if (count == 1) {
    switch (submission->transfer.operations[0].type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        operation->event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        operation->event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        operation->event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY;
        break;
    }
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    operation->event.payload_length +=
        iree_min(submission->transfer.operations[i].length,
                 UINT64_MAX - operation->event.payload_length);
  }
}

iree_byte_span_t iree_hal_amd_gpu_pm4_submission_tail_storage(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  const iree_host_size_t aligned_size =
      iree_host_align(submission->capture_size, iree_max_align_t);
  if (aligned_size >= submission->allocation_size) {
    return iree_byte_span_empty();
  }
  return iree_make_byte_span((uint8_t*)submission + aligned_size,
                             submission->allocation_size - aligned_size);
}

void iree_hal_amd_gpu_pm4_submission_retain(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission) {
    iree_atomic_ref_count_inc(&submission->ref_count);
  }
}

bool iree_hal_amd_gpu_pm4_submission_retirement_ready(
    const iree_hal_amd_gpu_pm4_submission_t* submission,
    iree_status_code_t* out_status_code) {
  *out_status_code = IREE_STATUS_OK;
  if (submission->native_wait_count) {
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      const iree_hal_amd_gpu_signal_t* signal = submission->waits[i].signal;
      if (signal) {
        iree_status_code_t status_code;
        if (!iree_hal_amd_gpu_signal_query_retirement(signal, &status_code)) {
          return false;
        }
        if (*out_status_code == IREE_STATUS_OK) {
          *out_status_code = status_code;
        }
      }
    }
  }
  return true;
}

void iree_hal_amd_gpu_pm4_submission_release_payload(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_DISPATCH) {
    if (!(submission->dispatch.flags &
          IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES)) {
      const iree_host_size_t reference_count =
          submission->dispatch.bindings.count +
          iree_hal_dispatch_uses_indirect_parameters(
              submission->dispatch.flags);
      for (iree_host_size_t i = 0; i < reference_count; ++i) {
        iree_hal_buffer_release(submission->dispatch.bindings.values[i].buffer);
      }
      iree_hal_executable_release(submission->dispatch.executable);
    }
  } else if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_TRANSFER) {
    iree_hal_amd_gpu_pm4_transfer_deinitialize(&submission->transfer);
  } else if (submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE) {
    iree_hal_amd_gpu_pm4_command_invocation_deinitialize(&submission->execute);
  }
  submission->entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_BARRIER;
}

static void iree_hal_amd_gpu_pm4_submission_destroy(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_hal_amd_gpu_capture_pool_t* pool = submission->pool;
  iree_hal_amd_gpu_pm4_submission_release_payload(submission);
  const iree_status_code_t failure_code = (iree_status_code_t)iree_atomic_load(
      &submission->failure_code, iree_memory_order_relaxed);
  if (failure_code != IREE_STATUS_OK) {
    iree_hal_semaphore_list_fail(submission->signals,
                                 iree_status_from_code(failure_code));
  }
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    iree_hal_amd_gpu_signal_release(submission->waits[i].signal);
    iree_hal_semaphore_release(submission->waits[i].semaphore);
  }
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    iree_hal_semaphore_release(submission->signals.semaphores[i]);
  }
  iree_hal_amd_gpu_signal_release(submission->signal);
  if (pool) {
    iree_hal_amd_gpu_capture_pool_release(
        pool, iree_make_byte_span(submission, submission->allocation_size));
  }
}

static void iree_hal_amd_gpu_pm4_submission_retire_file(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  const iree_hal_amd_gpu_pm4_submission_retirement_t* retirement =
      iree_hal_amd_gpu_pm4_submission_retirement(submission);
  const iree_hal_amd_gpu_retirement_callback_t callback = retirement->callback;
  const iree_status_code_t status_code = (iree_status_code_t)iree_atomic_load(
      &submission->failure_code, iree_memory_order_relaxed);
  const iree_async_single_frontier_t frontier = retirement->frontier;
  iree_hal_amd_gpu_pm4_submission_destroy(submission);
  // The file owner can release the device and its capture pool in this call.
  // All submission and pool accesses have ended before transferring ownership.
  callback.fn(callback.user_data, iree_status_from_code(status_code),
              frontier.entry_count ? &frontier : NULL, true);
}

void iree_hal_amd_gpu_pm4_submission_release(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission && iree_atomic_ref_count_dec(&submission->ref_count) == 1) {
    if (submission->has_retirement_callback) {
      iree_hal_amd_gpu_pm4_submission_retire_file(submission);
    } else {
      iree_hal_amd_gpu_pm4_submission_destroy(submission);
    }
  }
}
