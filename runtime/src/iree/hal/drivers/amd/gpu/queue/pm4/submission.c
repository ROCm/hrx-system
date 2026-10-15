// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"

// Captures common ownership and allocates the operation payload in one block.
static iree_status_t iree_hal_amd_gpu_pm4_submission_acquire(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t payload_size,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_pm4_submission_t** out_submission, void** out_payload) {
  iree_host_size_t payload_offset = sizeof(iree_hal_amd_gpu_pm4_submission_t);
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul_add(
          payload_offset, waits.count,
          sizeof(iree_hal_amd_gpu_pm4_submission_wait_t), &payload_offset) ||
      !iree_host_size_checked_mul_add(
          payload_offset, signals.count,
          sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t), &payload_offset) ||
      !iree_host_size_checked_align(payload_offset, 8, &payload_offset) ||
      !iree_host_size_checked_add(payload_offset, payload_size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "submission capture size overflows");
  }
  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_capture_pool_acquire(pool, total_size, &storage));
  iree_hal_amd_gpu_pm4_submission_t* submission = (void*)storage.data;
  *submission = (iree_hal_amd_gpu_pm4_submission_t){
      .native_wait_count = waits.count ? UINT32_MAX : 0,
      .pool = pool,
      .allocation_size = storage.data_length,
      .capture_size = total_size,
      .wait_count = waits.count,
      .before = iree_hal_amd_gpu_pm4_barrier_prepare(
          profile, barriers ? barriers->before : NULL),
      .after = iree_hal_amd_gpu_pm4_barrier_prepare(
          profile, barriers ? barriers->after : NULL),
  };
  iree_atomic_ref_count_init(&submission->ref_count);
  uint8_t* cursor = (uint8_t*)(submission + 1);
  submission->waits = (iree_hal_amd_gpu_pm4_submission_wait_t*)cursor;
  cursor += waits.count * sizeof(*submission->waits);
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    submission->waits[i] = (iree_hal_amd_gpu_pm4_submission_wait_t){
        .semaphore = waits.semaphores[i], .value = waits.payload_values[i]};
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
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

iree_status_t iree_hal_amd_gpu_pm4_submission_create(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_executable_t* executable,
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
      pool, waits, signals, payload_size, profile, barriers, &submission,
      &payload));
  submission->kind = function ? IREE_HAL_AMD_GPU_PM4_SUBMISSION_DISPATCH
                              : IREE_HAL_AMD_GPU_PM4_SUBMISSION_BARRIER;
  submission->dispatch.executable = executable;
  submission->dispatch.function = function;
  submission->dispatch.geometry = dispatch;
  submission->dispatch.flags = flags;
  iree_hal_executable_retain(executable);
  iree_hal_buffer_ref_t* references = payload;
  submission->dispatch.bindings =
      (iree_hal_buffer_ref_list_t){bindings.count, references};
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    references[i] = bindings.values[i];
    iree_hal_buffer_retain(references[i].buffer);
  }
  if (indirect_parameters) {
    references[bindings.count] = parameters;
    iree_hal_buffer_retain(parameters.buffer);
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

iree_status_t iree_hal_amd_gpu_pm4_submission_create_transfer(
    iree_hal_amd_gpu_capture_pool_t* pool,
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
        pool, waits, signals, NULL, NULL, (iree_hal_amd_gpu_pm4_dispatch_t){0},
        (iree_hal_buffer_ref_t){0}, iree_const_byte_span_empty(),
        iree_hal_buffer_ref_list_empty(), profile, barriers, 0, out_submission);
  }
  iree_hal_amd_gpu_pm4_submission_t* submission = NULL;
  void* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_submission_acquire(
      pool, waits, signals, layout.byte_length, profile, barriers, &submission,
      &payload));
  submission->kind = IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER;
  iree_status_t status = iree_hal_amd_gpu_pm4_transfer_initialize(
      family, count, operations, layout, payload, &submission->transfer);
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    iree_hal_amd_gpu_pm4_submission_release(submission);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_pm4_submission_create_execute(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
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
      pool, waits, signals, payload_size, profile, &barriers, &submission,
      &payload));
  submission->kind = IREE_HAL_AMD_GPU_PM4_SUBMISSION_EXECUTE;
  iree_status_t status = iree_hal_amd_gpu_pm4_command_invocation_initialize(
      command_buffer, bindings, payload, flags, &submission->execute);
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    iree_hal_amd_gpu_pm4_submission_release(submission);
  }
  return status;
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
    const iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission->native_wait_count) {
    for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
      const iree_hal_amd_gpu_pm4_signal_t* signal = submission->waits[i].signal;
      if (signal && !iree_hal_amd_gpu_pm4_signal_is_retired(signal)) {
        return false;
      }
    }
  }
  return true;
}

void iree_hal_amd_gpu_pm4_submission_retire(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_DISPATCH) {
    const iree_host_size_t reference_count =
        submission->dispatch.bindings.count +
        iree_hal_dispatch_uses_indirect_parameters(submission->dispatch.flags);
    for (iree_host_size_t i = 0; i < reference_count; ++i) {
      iree_hal_buffer_release(submission->dispatch.bindings.values[i].buffer);
    }
    iree_hal_executable_release(submission->dispatch.executable);
  } else if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_TRANSFER) {
    iree_hal_amd_gpu_pm4_transfer_deinitialize(&submission->transfer);
  } else if (submission->kind == IREE_HAL_AMD_GPU_PM4_SUBMISSION_EXECUTE) {
    iree_hal_amd_gpu_pm4_command_invocation_deinitialize(&submission->execute);
  }
  submission->kind = IREE_HAL_AMD_GPU_PM4_SUBMISSION_BARRIER;
  iree_hal_amd_gpu_pm4_signal_retire(submission->signal);
}

static void iree_hal_amd_gpu_pm4_submission_destroy(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  iree_hal_amd_gpu_capture_pool_t* pool = submission->pool;
  iree_hal_amd_gpu_pm4_submission_retire(submission);
  for (iree_host_size_t i = 0; i < submission->wait_count; ++i) {
    iree_hal_amd_gpu_pm4_signal_release(submission->waits[i].signal);
    iree_hal_semaphore_release(submission->waits[i].semaphore);
  }
  for (iree_host_size_t i = 0; i < submission->signals.count; ++i) {
    iree_hal_semaphore_release(submission->signals.semaphores[i]);
  }
  iree_hal_amd_gpu_pm4_signal_release(submission->signal);
  iree_hal_amd_gpu_capture_pool_release(
      pool, iree_make_byte_span(submission, submission->allocation_size));
}

void iree_hal_amd_gpu_pm4_submission_release(
    iree_hal_amd_gpu_pm4_submission_t* submission) {
  if (submission && iree_atomic_ref_count_dec(&submission->ref_count) == 1) {
    iree_hal_amd_gpu_pm4_submission_destroy(submission);
  }
}

iree_status_t iree_hal_amd_gpu_pm4_submission_resolve_address(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    iree_hal_buffer_ref_t reference, iree_hal_buffer_usage_t usage,
    iree_hal_memory_access_t access, uint64_t* out_address) {
  const iree_hal_memory_contract_t* contract =
      reference.buffer->memory.contract;
  const uint64_t scope_id = (uint64_t)family->memory.queue_scope_id +
                            (site - IREE_HAL_MEMORY_SITE_QUEUE);
  if (!contract || !family->memory.domain ||
      family->memory.domain != contract->domain ||
      scope_id >= contract->scope_count) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "buffer is outside the dispatch family domain");
  }
  const iree_hal_memory_scope_access_t* scope = &contract->scopes[scope_id];
  const iree_hal_buffer_interface_t interface =
      IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  const uint16_t index = scope->bindings[interface];
  if (!iree_any_bit_set(scope->interfaces, UINT32_C(1) << interface) ||
      index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "GPU address access is not prepared for this "
                            "dispatch family");
  }
  if (access) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
        iree_hal_buffer_allowed_access(reference.buffer), access));
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage_any(
      scope->usage & iree_hal_buffer_allowed_usage(reference.buffer), usage));
  const uint64_t base =
      iree_hal_buffer_native_binding(reference.buffer,
                                     (iree_hal_buffer_native_binding_slot_t){
                                         .index = index, .type = interface})
          .device_address;
  if (!base) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "dispatch buffer has no committed GPU address");
  }
  if (!iree_checked_add_u64(base, reference.offset, out_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch buffer address overflows");
  }
  return iree_ok_status();
}
