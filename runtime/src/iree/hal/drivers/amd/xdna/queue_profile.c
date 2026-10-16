// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_state.h"

iree_status_t iree_hal_amd_xdna_queue_profiling_begin(
    iree_hal_queue_t* base, uint32_t physical_device_ordinal,
    iree_hal_amd_xdna_command_arena_t* command_arena,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_amd_xdna_queue_t* queue = (void*)base;
  if (queue->profile) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "XDNA profiling session is already active");
  }
  return iree_hal_amd_xdna_profile_begin(
      options, ++queue->next_profile_session_id, physical_device_ordinal, 0,
      queue->context, command_arena, queue->host_allocator, &queue->profile);
}

iree_status_t iree_hal_amd_xdna_queue_profiling_flush(iree_hal_queue_t* base) {
  return iree_hal_amd_xdna_profile_flush(
      ((iree_hal_amd_xdna_queue_t*)base)->profile);
}

iree_status_t iree_hal_amd_xdna_queue_profiling_end(iree_hal_queue_t* base) {
  iree_hal_amd_xdna_queue_t* queue = (void*)base;
  iree_hal_amd_xdna_profile_t* profile = queue->profile;
  queue->profile = NULL;
  return iree_hal_amd_xdna_profile_end(profile);
}

static uint32_t iree_hal_amd_xdna_profile_count(iree_host_size_t count) {
  return (uint32_t)iree_min(count, UINT32_MAX);
}

// Summarizes one captured transfer in the existing public queue vocabulary.
static iree_device_size_t iree_hal_amd_xdna_profile_transfer(
    const iree_hal_transfer_operation_t* transfer,
    iree_hal_profile_queue_event_type_t* out_type) {
  switch (transfer->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      *out_type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL;
      return transfer->fill.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      *out_type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE;
      return transfer->update.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      *out_type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY;
      return transfer->copy.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      *out_type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE;
      return transfer->upload.length;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      *out_type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY;
      return transfer->download.length;
  }
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t iree_hal_amd_xdna_operation_profile_capture(
    iree_hal_amd_xdna_operation_t* operation) {
  if (!operation->queue->profile) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &operation->capture, sizeof(*operation->profile),
      (void**)&operation->profile));
  memset(operation->profile, 0, sizeof(*operation->profile));
  iree_hal_profile_queue_event_t* event = &operation->profile->event;
  iree_hal_amd_xdna_profile_initialize_event(operation->queue->profile, event);
  event->wait_count = iree_hal_amd_xdna_profile_count(operation->waits.count);
  event->signal_count =
      iree_hal_amd_xdna_profile_count(operation->signals.count);
  event->dependency_strategy =
      event->wait_count ? IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE
                        : IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE;
  event->operation_count = 1;
  switch (operation->kind) {
    case IREE_HAL_AMD_XDNA_OPERATION_BARRIER:
      event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_HOST_CALL:
      event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL;
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_ALLOCA:
      event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA;
      event->operation_count =
          iree_hal_amd_xdna_profile_count(operation->alloca.request_count);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA:
      event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA;
      event->operation_count =
          iree_hal_amd_xdna_profile_count(operation->dealloca.buffer_count);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_DISPATCH:
      event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_EXECUTE:
      event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
      event->command_buffer_id =
          iree_hal_command_buffer_profile_id(operation->execution->owner);
      event->operation_count = iree_hal_amd_xdna_profile_count(
          iree_hal_amd_xdna_command_buffer_record_count(
              operation->execution->owner));
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_TRANSFER:
      event->type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
      event->operation_count = 0;
      for (const iree_hal_amd_xdna_transfer_t* transfer =
               operation->transfer.head;
           transfer; transfer = transfer->next) {
        iree_hal_profile_queue_event_type_t type;
        const uint64_t length =
            iree_hal_amd_xdna_profile_transfer(&transfer->operation, &type);
        if (!length) {
          continue;
        }
        event->type = event->operation_count
                          ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_TRANSFER
                          : type;
        if (event->operation_count < UINT32_MAX) {
          ++event->operation_count;
        }
        event->payload_length +=
            iree_min(length, UINT64_MAX - event->payload_length);
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_FILE:
      event->type =
          operation->file.transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
              ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ
              : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE;
      event->payload_length = operation->file.transfer->length;
      break;
  }
  return iree_ok_status();
}

void iree_hal_amd_xdna_operation_profile_ready(
    iree_hal_amd_xdna_operation_t* operation) {
  if (!operation->profile ||
      iree_any_bit_set(operation->profile->flags,
                       IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_RECORDED)) {
    return;
  }
  operation->profile->event.ready_host_time_ns = iree_time_now();
  iree_hal_amd_xdna_operation_profile_complete(operation);
}

void iree_hal_amd_xdna_operation_profile_complete(
    iree_hal_amd_xdna_operation_t* operation) {
  if (!operation->profile ||
      iree_any_bit_set(operation->profile->flags,
                       IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_RECORDED)) {
    return;
  }
  iree_hal_amd_xdna_profile_append_queue(operation->queue->profile,
                                         &operation->profile->event);
  operation->profile->flags |=
      IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_RECORDED;
}

void iree_hal_amd_xdna_operation_profile_host_end(
    iree_hal_amd_xdna_operation_t* operation, iree_time_t start,
    iree_status_code_t status_code, uint32_t operation_count) {
  if (!start) {
    return;
  }
  iree_hal_profile_host_execution_event_flags_t flags = 0;
  if (iree_any_bit_set(operation->profile->flags,
                       IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_DEFERRED)) {
    flags |= IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_DEFERRED;
  }
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE) {
    flags |= IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_COMMAND_BUFFER;
  }
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_TRANSFER &&
      status_code == IREE_STATUS_OK) {
    operation_count = operation->profile->event.operation_count;
  }
  iree_hal_amd_xdna_profile_host_end(operation->queue->profile,
                                     &operation->profile->event, start, flags,
                                     status_code, operation_count);
}

void iree_hal_amd_xdna_operation_profile_prepare_native(
    iree_hal_amd_xdna_operation_t* operation, uint32_t count,
    iree_hal_amd_xdna_invocation_t* const* invocations,
    amdf_xdna_kernel_command_t* commands) {
  if (iree_any_bit_set(
          operation->profile->flags,
          IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_NATIVE_PREPARED)) {
    return;
  }
  iree_hal_amd_xdna_profile_counters_t* counters =
      iree_hal_amd_xdna_profile_counters(operation->queue->profile);
  if (counters) {
    const iree_hal_amd_xdna_record_t* record =
        operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE
            ? operation->execution->cursor
            : NULL;
    for (uint32_t i = 0; i < count; ++i) {
      iree_hal_amd_xdna_profile_counters_prepare(
          counters, &operation->profile->event,
          record ? record->ordinal : UINT32_MAX, invocations[i], &commands[i]);
      if (record) {
        record = record->next;
      }
    }
  }
  operation->profile->flags |=
      IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_NATIVE_PREPARED;
}

void iree_hal_amd_xdna_operation_profile_accept_native(
    iree_hal_amd_xdna_operation_t* operation, uint32_t count,
    iree_hal_amd_xdna_invocation_t* const* invocations,
    const amdf_xdna_kernel_command_t* commands) {
  iree_hal_amd_xdna_profile_counters_t* counters =
      iree_hal_amd_xdna_profile_counters(operation->queue->profile);
  if (counters) {
    const iree_hal_amd_xdna_record_t* record =
        operation->kind == IREE_HAL_AMD_XDNA_OPERATION_EXECUTE
            ? operation->execution->cursor
            : NULL;
    for (uint32_t i = 0; i < count; ++i) {
      iree_hal_amd_xdna_profile_counters_accept(
          counters, &operation->profile->event,
          record ? record->ordinal : UINT32_MAX, invocations[i], &commands[i]);
      if (record) {
        record = record->next;
      }
    }
  }
  operation->profile->flags &=
      ~IREE_HAL_AMD_XDNA_OPERATION_PROFILE_FLAG_NATIVE_PREPARED;
}
