// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_queue.h"

#include <string.h>

#include "iree/hal/pool.h"
#include "iree/hal/replay/recorder_allocator.h"
#include "iree/hal/replay/recorder_buffer.h"
#include "iree/hal/replay/recorder_command_buffer.h"
#include "iree/hal/replay/recorder_executable.h"
#include "iree/hal/replay/recorder_file.h"
#include "iree/hal/replay/recorder_record.h"

typedef struct iree_hal_replay_recorder_transfer_storage_t {
  // Operations rewritten to reference resources owned by the wrapped device.
  iree_hal_transfer_operation_t* base_operations;
  // Temporary buffer subspans retained until the wrapped call has captured
  // every operation.
  iree_hal_buffer_t** temporary_buffers;
  // Number of entries in |temporary_buffers|.
  iree_host_size_t temporary_buffer_count;
  // Serialized operation descriptors, or NULL for unsupported capture.
  iree_hal_replay_queue_transfer_operation_payload_t* operation_payloads;
  // Contiguous captured fill, update, and upload bytes.
  uint8_t* data;
  // Serialized wait semaphore timepoints.
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads;
  // Byte length of |wait_payloads|.
  iree_host_size_t wait_payloads_size;
  // Serialized signal semaphore timepoints.
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads;
  // Byte length of |signal_payloads|.
  iree_host_size_t signal_payloads_size;
} iree_hal_replay_recorder_transfer_storage_t;

typedef struct iree_hal_replay_recorder_alloca_storage_t {
  // Number of allocation requests represented by all parallel arrays.
  iree_host_size_t request_count;
  // Buffers returned by the wrapped queue before proxying.
  iree_hal_buffer_t** base_buffers;
  // Preallocated proxy objects consumed after the wrapped call succeeds.
  iree_hal_replay_recorder_buffer_t** proxy_buffers;
  // Initialized proxy buffers published to the caller on success.
  iree_hal_buffer_t** replay_buffers;
  // Serialized allocation requests and resulting buffer ids.
  iree_hal_replay_queue_alloca_request_payload_t* request_payloads;
  // Serialized metadata for created buffer object records.
  iree_hal_replay_buffer_object_payload_t* object_payloads;
  // Byte spans over each entry in |object_payloads|.
  iree_const_byte_span_t* object_iovecs;
  // Created object records appended with the allocation operation.
  iree_hal_replay_created_object_record_t* created_objects;
  // Serialized wait semaphore timepoints.
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads;
  // Byte length of |wait_payloads|.
  iree_host_size_t wait_payloads_size;
  // Serialized signal semaphore timepoints.
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads;
  // Byte length of |signal_payloads|.
  iree_host_size_t signal_payloads_size;
} iree_hal_replay_recorder_alloca_storage_t;

typedef struct iree_hal_replay_recorder_dealloca_storage_t {
  // Number of allocation roots represented by all parallel arrays.
  iree_host_size_t buffer_count;
  // Underlying allocation roots passed to the wrapped queue.
  iree_hal_buffer_t** base_buffers;
  // Temporary subspans retained while the wrapped call captures arguments.
  iree_hal_buffer_t** temporary_buffers;
  // Serialized allocation-root object ids.
  iree_hal_replay_object_id_t* buffer_ids;
  // Serialized wait semaphore timepoints.
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads;
  // Byte length of |wait_payloads|.
  iree_host_size_t wait_payloads_size;
  // Serialized signal semaphore timepoints.
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads;
  // Byte length of |signal_payloads|.
  iree_host_size_t signal_payloads_size;
} iree_hal_replay_recorder_dealloca_storage_t;

typedef struct iree_hal_replay_recorder_semaphore_storage_t {
  // Serialized wait semaphore timepoints.
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads;

  // Byte length of |wait_payloads|.
  iree_host_size_t wait_payloads_size;

  // Serialized signal semaphore timepoints.
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads;

  // Byte length of |signal_payloads|.
  iree_host_size_t signal_payloads_size;
} iree_hal_replay_recorder_semaphore_storage_t;

typedef struct iree_hal_replay_recorder_buffer_ref_list_storage_t {
  // Forwarded buffer references rewritten to underlying buffers.
  iree_hal_buffer_ref_list_t base_list;

  // Mutable storage backing |base_list|.
  iree_hal_buffer_ref_t* base_values;

  // Temporary subspans retained until the underlying call captures them.
  iree_hal_buffer_t** temporary_buffers;

  // Serialized references captured from wrapper buffers.
  iree_hal_replay_buffer_ref_payload_t* payloads;

  // Byte length of |payloads|.
  iree_host_size_t payloads_size;
} iree_hal_replay_recorder_buffer_ref_list_storage_t;

typedef struct iree_hal_replay_recorder_host_call_state_t {
  // Resource retained by the underlying queue until callback delivery.
  iree_hal_resource_t resource;

  // Host allocator used for this callback state.
  iree_allocator_t host_allocator;

  // Recording wrapper queue reported to the user callback.
  iree_hal_queue_t* queue;

  // User callback and state captured from the original queue call.
  iree_hal_host_call_t call;
} iree_hal_replay_recorder_host_call_state_t;

static iree_status_t iree_hal_replay_recorder_queue_allocate_array(
    iree_allocator_t host_allocator, iree_host_size_t element_count,
    iree_host_size_t element_size, void** out_ptr) {
  if (element_count == 0) {
    *out_ptr = NULL;
    return iree_ok_status();
  }
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(element_count, element_size,
                                                &allocation_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "replay queue storage size overflow");
  }
  void* ptr = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, allocation_size, &ptr);
  if (iree_status_is_ok(status)) *out_ptr = ptr;
  return status;
}

static void iree_hal_replay_recorder_semaphore_storage_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_semaphore_storage_t* storage) {
  iree_allocator_free(host_allocator, storage->signal_payloads);
  iree_allocator_free(host_allocator, storage->wait_payloads);
  memset(storage, 0, sizeof(*storage));
}

static iree_status_t iree_hal_replay_recorder_semaphore_storage_initialize(
    iree_hal_replay_recorder_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_replay_recorder_semaphore_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      queue->recorder, wait_semaphore_list, queue->host_allocator,
      &out_storage->wait_payloads, &out_storage->wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signal_semaphore_list, queue->host_allocator,
        &out_storage->signal_payloads, &out_storage->signal_payloads_size);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_recorder_semaphore_storage_deinitialize(
        queue->host_allocator, out_storage);
  }
  return status;
}

static void iree_hal_replay_recorder_buffer_ref_list_storage_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_buffer_ref_list_storage_t* storage) {
  if (storage->temporary_buffers) {
    for (iree_host_size_t i = 0; i < storage->base_list.count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(
          storage->temporary_buffers[i]);
    }
  }
  iree_allocator_free(host_allocator, storage->payloads);
  iree_allocator_free(host_allocator, storage->temporary_buffers);
  iree_allocator_free(host_allocator, storage->base_values);
  memset(storage, 0, sizeof(*storage));
}

static iree_status_t
iree_hal_replay_recorder_buffer_ref_list_storage_initialize(
    iree_hal_replay_recorder_queue_t* queue,
    const iree_hal_buffer_ref_list_t source_list, bool capture_payloads,
    iree_hal_replay_recorder_buffer_ref_list_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  out_storage->base_list = source_list;
  if (source_list.count == 0) return iree_ok_status();
  if (IREE_UNLIKELY(!source_list.values)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay queue buffer reference storage is NULL");
  }

  iree_status_t status = iree_hal_replay_recorder_queue_allocate_array(
      queue->host_allocator, source_list.count,
      sizeof(*out_storage->base_values), (void**)&out_storage->base_values);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, source_list.count,
        sizeof(*out_storage->temporary_buffers),
        (void**)&out_storage->temporary_buffers);
  }
  if (iree_status_is_ok(status) && capture_payloads) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, source_list.count,
        sizeof(*out_storage->payloads), (void**)&out_storage->payloads);
  }
  if (iree_status_is_ok(status)) {
    memset(out_storage->temporary_buffers, 0,
           source_list.count * sizeof(*out_storage->temporary_buffers));
    memcpy(out_storage->base_values, source_list.values,
           source_list.count * sizeof(*out_storage->base_values));
    out_storage->base_list.values = out_storage->base_values;
    if (capture_payloads) {
      out_storage->payloads_size =
          source_list.count * sizeof(*out_storage->payloads);
    }
  }
  for (iree_host_size_t i = 0;
       i < source_list.count && iree_status_is_ok(status); ++i) {
    if (capture_payloads) {
      iree_hal_replay_recorder_buffer_ref_make_payload(
          source_list.values[i], &out_storage->payloads[i]);
      out_storage->payloads[i].buffer_id =
          iree_hal_replay_recorder_find_buffer_id(queue->recorder,
                                                  source_list.values[i].buffer);
    }
    if (out_storage->base_values[i].buffer) {
      status = iree_hal_replay_recorder_buffer_unwrap_for_call(
          out_storage->base_values[i].buffer, queue->host_allocator,
          &out_storage->base_values[i].buffer,
          &out_storage->temporary_buffers[i]);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_ref_list_storage_deinitialize(
        queue->host_allocator, out_storage);
  }
  return status;
}

static void iree_hal_replay_recorder_transfer_storage_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_transfer_storage_t* storage) {
  if (storage->temporary_buffers) {
    for (iree_host_size_t i = 0; i < storage->temporary_buffer_count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(
          storage->temporary_buffers[i]);
    }
  }
  iree_allocator_free(host_allocator, storage->signal_payloads);
  iree_allocator_free(host_allocator, storage->wait_payloads);
  iree_allocator_free(host_allocator, storage->data);
  iree_allocator_free(host_allocator, storage->operation_payloads);
  iree_allocator_free(host_allocator, storage->temporary_buffers);
  iree_allocator_free(host_allocator, storage->base_operations);
  memset(storage, 0, sizeof(*storage));
}

static void iree_hal_replay_recorder_alloca_storage_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_alloca_storage_t* storage) {
  if (storage->replay_buffers) {
    for (iree_host_size_t i = 0; i < storage->request_count; ++i) {
      iree_hal_buffer_release(storage->replay_buffers[i]);
    }
  }
  if (storage->proxy_buffers) {
    for (iree_host_size_t i = 0; i < storage->request_count; ++i) {
      iree_hal_replay_recorder_buffer_free_proxy(host_allocator,
                                                 storage->proxy_buffers[i]);
    }
  }
  if (storage->base_buffers) {
    for (iree_host_size_t i = 0; i < storage->request_count; ++i) {
      iree_hal_buffer_release(storage->base_buffers[i]);
    }
  }
  iree_allocator_free(host_allocator, storage->signal_payloads);
  iree_allocator_free(host_allocator, storage->wait_payloads);
  iree_allocator_free(host_allocator, storage->created_objects);
  iree_allocator_free(host_allocator, storage->object_iovecs);
  iree_allocator_free(host_allocator, storage->object_payloads);
  iree_allocator_free(host_allocator, storage->request_payloads);
  iree_allocator_free(host_allocator, storage->replay_buffers);
  iree_allocator_free(host_allocator, storage->proxy_buffers);
  iree_allocator_free(host_allocator, storage->base_buffers);
  memset(storage, 0, sizeof(*storage));
}

static void iree_hal_replay_recorder_dealloca_storage_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_dealloca_storage_t* storage) {
  if (storage->temporary_buffers) {
    for (iree_host_size_t i = 0; i < storage->buffer_count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(
          storage->temporary_buffers[i]);
    }
  }
  iree_allocator_free(host_allocator, storage->signal_payloads);
  iree_allocator_free(host_allocator, storage->wait_payloads);
  iree_allocator_free(host_allocator, storage->buffer_ids);
  iree_allocator_free(host_allocator, storage->temporary_buffers);
  iree_allocator_free(host_allocator, storage->base_buffers);
  memset(storage, 0, sizeof(*storage));
}

static bool iree_hal_replay_recorder_queue_has_captured_semaphores(
    iree_hal_replay_recorder_t* recorder,
    const iree_hal_semaphore_list_t semaphore_list) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (iree_hal_replay_recorder_semaphore_id_or_none(
            recorder, semaphore_list.semaphores[i]) ==
        IREE_HAL_REPLAY_OBJECT_ID_NONE) {
      return false;
    }
  }
  return true;
}

static bool iree_hal_replay_recorder_queue_has_captured_buffer(
    iree_hal_replay_recorder_t* recorder, iree_hal_buffer_t* buffer) {
  return !buffer || iree_hal_replay_recorder_find_buffer_id(recorder, buffer) !=
                        IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

static bool iree_hal_replay_recorder_queue_can_record_target_operation(
    iree_hal_replay_recorder_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer) {
  return iree_hal_replay_recorder_queue_has_captured_buffer(queue->recorder,
                                                            target_buffer) &&
         iree_hal_replay_recorder_queue_has_captured_semaphores(
             queue->recorder, wait_semaphore_list) &&
         iree_hal_replay_recorder_queue_has_captured_semaphores(
             queue->recorder, signal_semaphore_list);
}

static iree_status_t iree_hal_replay_recorder_queue_analyze_transfer(
    iree_hal_replay_recorder_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations, bool* out_can_record,
    iree_host_size_t* out_data_length) {
  bool can_record = iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, wait_semaphore_list) &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, signal_semaphore_list);
  iree_host_size_t data_length = 0;
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const iree_hal_transfer_operation_t* operation = &operations[i];
    iree_host_size_t operation_data_length = 0;
    switch (operation->type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        if (operation->fill.length == 0) break;
        can_record &= iree_hal_replay_recorder_queue_has_captured_buffer(
            queue->recorder, operation->fill.target_buffer);
        operation_data_length = operation->fill.pattern_length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
        if (operation->update.length == 0) break;
        can_record &= iree_hal_replay_recorder_queue_has_captured_buffer(
            queue->recorder, operation->update.target_buffer);
        operation_data_length = (iree_host_size_t)operation->update.length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
        if (operation->copy.length == 0) break;
        can_record &= iree_hal_replay_recorder_queue_has_captured_buffer(
                          queue->recorder, operation->copy.source_buffer) &&
                      iree_hal_replay_recorder_queue_has_captured_buffer(
                          queue->recorder, operation->copy.target_buffer);
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        if (operation->upload.length == 0) break;
        can_record &= wait_semaphore_list.count == 0;
        can_record &= iree_hal_replay_recorder_queue_has_captured_buffer(
            queue->recorder, operation->upload.target_buffer);
        operation_data_length = (iree_host_size_t)operation->upload.length;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        if (operation->download.length == 0) break;
        can_record &= iree_hal_replay_recorder_queue_has_captured_buffer(
            queue->recorder, operation->download.source_buffer);
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown transfer operation type %u",
                                (uint32_t)operation->type);
    }
    if (can_record && IREE_UNLIKELY(!iree_host_size_checked_add(
                          data_length, operation_data_length, &data_length))) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "replay transfer data size overflow");
    }
  }
  *out_can_record = can_record;
  *out_data_length = can_record ? data_length : 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_queue_allocate_transfer_storage(
    iree_hal_replay_recorder_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_count, bool can_record,
    iree_host_size_t data_length,
    iree_hal_replay_recorder_transfer_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  iree_status_t status = iree_hal_replay_recorder_queue_allocate_array(
      queue->host_allocator, operation_count,
      sizeof(*out_storage->base_operations),
      (void**)&out_storage->base_operations);
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(!iree_host_size_checked_mul(
          operation_count, 2, &out_storage->temporary_buffer_count))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "replay temporary buffer count overflow");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, out_storage->temporary_buffer_count,
        sizeof(*out_storage->temporary_buffers),
        (void**)&out_storage->temporary_buffers);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, operation_count,
        sizeof(*out_storage->operation_payloads),
        (void**)&out_storage->operation_payloads);
  }
  if (iree_status_is_ok(status) && can_record && data_length != 0) {
    status = iree_allocator_malloc(queue->host_allocator, data_length,
                                   (void**)&out_storage->data);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, wait_semaphore_list, queue->host_allocator,
        &out_storage->wait_payloads, &out_storage->wait_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signal_semaphore_list, queue->host_allocator,
        &out_storage->signal_payloads, &out_storage->signal_payloads_size);
  }
  if (iree_status_is_ok(status)) {
    if (out_storage->base_operations) {
      memset(out_storage->base_operations, 0,
             operation_count * sizeof(*out_storage->base_operations));
    }
    if (out_storage->temporary_buffers) {
      memset(out_storage->temporary_buffers, 0,
             out_storage->temporary_buffer_count *
                 sizeof(*out_storage->temporary_buffers));
    }
    if (out_storage->operation_payloads) {
      memset(out_storage->operation_payloads, 0,
             operation_count * sizeof(*out_storage->operation_payloads));
    }
  } else {
    iree_hal_replay_recorder_transfer_storage_deinitialize(
        queue->host_allocator, out_storage);
  }
  return status;
}

static void iree_hal_replay_recorder_queue_make_buffer_ref_payload(
    iree_hal_replay_recorder_queue_t* queue, iree_hal_buffer_t* buffer,
    iree_device_size_t offset, iree_device_size_t length,
    iree_hal_replay_buffer_ref_payload_t* out_payload) {
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(buffer, offset, length), out_payload);
  out_payload->buffer_id =
      iree_hal_replay_recorder_find_buffer_id(queue->recorder, buffer);
}

static void iree_hal_replay_recorder_host_call_state_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_replay_recorder_host_call_state_t* state =
      (iree_hal_replay_recorder_host_call_state_t*)resource;
  iree_hal_resource_release(state->call.resource);
  iree_allocator_free(state->host_allocator, state);
}

static const iree_hal_resource_vtable_t
    iree_hal_replay_recorder_host_call_state_vtable = {
        .destroy = iree_hal_replay_recorder_host_call_state_destroy,
};

static iree_status_t iree_hal_replay_recorder_host_call_state_create(
    iree_hal_replay_recorder_queue_t* queue, iree_hal_host_call_t call,
    iree_hal_replay_recorder_host_call_state_t** out_state) {
  *out_state = NULL;
  iree_hal_replay_recorder_host_call_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  iree_hal_resource_initialize(&iree_hal_replay_recorder_host_call_state_vtable,
                               &state->resource);
  state->host_allocator = queue->host_allocator;
  state->queue = &queue->base;
  state->call = call;
  iree_hal_resource_retain(state->call.resource);
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_host_call_thunk(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  iree_hal_replay_recorder_host_call_state_t* state =
      (iree_hal_replay_recorder_host_call_state_t*)user_data;
  context->queue = state->queue;
  return state->call.fn(state->call.user_data, args, context);
}

static iree_status_t iree_hal_replay_recorder_queue_barrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_queue_barrier_flags_t flags) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const bool can_record =
      iree_hal_replay_recorder_queue_has_captured_semaphores(
          queue->recorder, wait_semaphore_list) &&
      iree_hal_replay_recorder_queue_has_captured_semaphores(
          queue->recorder, signal_semaphore_list);
  const iree_hal_replay_queue_barrier_payload_t payload = {
      .flags = flags,
      .wait_semaphore_count = wait_semaphore_list.count,
      .signal_semaphore_count = signal_semaphore_list.count,
  };

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_ok_status();
  if (can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, wait_semaphore_list, queue->host_allocator,
        &wait_payloads, &wait_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signal_semaphore_list, queue->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id,
        IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_BARRIER,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_barrier(queue->base_queue, wait_semaphore_list,
                                    signal_semaphore_list, flags);
  }
  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(wait_payloads, wait_payloads_size),
          iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_allocator_free(queue->host_allocator, signal_payloads);
  iree_allocator_free(queue->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_execute(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const iree_host_size_t used_binding_count = command_buffer->binding_count;
  const iree_hal_replay_object_id_t command_buffer_id =
      iree_hal_replay_recorder_command_buffer_id_or_none(command_buffer);
  bool can_record = command_buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, wait_semaphore_list) &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, signal_semaphore_list);
  for (iree_host_size_t i = 0; i < used_binding_count; ++i) {
    can_record &= iree_hal_replay_recorder_queue_has_captured_buffer(
        queue->recorder, binding_table.bindings[i].buffer);
  }
  const iree_hal_replay_queue_execute_payload_t payload = {
      .command_buffer_id = command_buffer_id,
      .flags = flags,
      .wait_semaphore_count = wait_semaphore_list.count,
      .signal_semaphore_count = signal_semaphore_list.count,
      .binding_count = used_binding_count,
  };

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_hal_replay_buffer_ref_payload_t* binding_payloads = NULL;
  iree_host_size_t binding_payloads_size = 0;
  iree_hal_buffer_binding_t* base_bindings = NULL;
  iree_status_t status = iree_ok_status();
  if (can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, wait_semaphore_list, queue->host_allocator,
        &wait_payloads, &wait_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signal_semaphore_list, queue->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  if (iree_status_is_ok(status) && used_binding_count != 0) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, used_binding_count, sizeof(*base_bindings),
        (void**)&base_bindings);
  }
  if (iree_status_is_ok(status) && can_record && used_binding_count != 0) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, used_binding_count, sizeof(*binding_payloads),
        (void**)&binding_payloads);
    if (iree_status_is_ok(status)) {
      binding_payloads_size = used_binding_count * sizeof(*binding_payloads);
    }
  }
  for (iree_host_size_t i = 0;
       i < used_binding_count && iree_status_is_ok(status); ++i) {
    const iree_hal_buffer_binding_t* binding = &binding_table.bindings[i];
    if (can_record) {
      iree_hal_replay_recorder_queue_make_buffer_ref_payload(
          queue, binding->buffer, binding->offset, binding->length,
          &binding_payloads[i]);
    }
    iree_hal_buffer_ref_t base_ref = iree_hal_make_buffer_ref(
        binding->buffer, binding->offset, binding->length);
    status = iree_hal_replay_recorder_buffer_ref_unwrap_for_call(&base_ref);
    if (iree_status_is_ok(status)) {
      base_bindings[i] = (iree_hal_buffer_binding_t){
          .buffer = base_ref.buffer,
          .offset = base_ref.offset,
          .length = base_ref.length,
      };
    }
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id, command_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_EXECUTE,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_EXECUTE
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_execute(
        queue->base_queue, wait_semaphore_list, signal_semaphore_list,
        iree_hal_replay_recorder_command_buffer_base_or_self(command_buffer),
        (iree_hal_buffer_binding_table_t){
            .count = used_binding_count,
            .bindings = base_bindings,
        },
        flags);
  }
  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(wait_payloads, wait_payloads_size),
          iree_make_const_byte_span(signal_payloads, signal_payloads_size),
          iree_make_const_byte_span(binding_payloads, binding_payloads_size),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_allocator_free(queue->host_allocator, base_bindings);
  iree_allocator_free(queue->host_allocator, binding_payloads);
  iree_allocator_free(queue->host_allocator, signal_payloads);
  iree_allocator_free(queue->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_host_call(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;

  iree_hal_replay_recorder_host_call_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_host_call_state_create(queue, call, &state));

  iree_hal_replay_pending_record_t pending_record = {0};
  iree_status_t status = iree_hal_replay_recorder_begin_operation(
      queue->recorder, queue->device_id, queue->queue_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_HOST_CALL,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record);
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
    const iree_hal_host_call_t forwarded_call =
        iree_hal_make_host_call_with_resource(
            iree_hal_replay_recorder_host_call_thunk, state, &state->resource);
    status = iree_hal_queue_host_call(queue->base_queue, wait_semaphore_list,
                                      signal_semaphore_list, forwarded_call,
                                      args, flags);
  }
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation(&pending_record, status);
  }
  iree_hal_resource_release(&state->resource);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_dispatch(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const iree_hal_replay_object_id_t executable_id =
      iree_hal_replay_recorder_executable_id_or_none(executable);
  bool can_record = executable_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, wait_semaphore_list) &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, signal_semaphore_list) &&
                    iree_hal_replay_recorder_queue_has_captured_buffer(
                        queue->recorder, config.workgroup_count_ref.buffer);
  for (iree_host_size_t i = 0; bindings.values && i < bindings.count; ++i) {
    can_record &= iree_hal_replay_recorder_queue_has_captured_buffer(
        queue->recorder, bindings.values[i].buffer);
  }

  iree_hal_replay_dispatch_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.executable_id = executable_id;
  payload.flags = flags;
  memcpy(payload.workgroup_size, config.workgroup_size,
         sizeof(payload.workgroup_size));
  memcpy(payload.workgroup_count, config.workgroup_count,
         sizeof(payload.workgroup_count));
  iree_hal_replay_recorder_queue_make_buffer_ref_payload(
      queue, config.workgroup_count_ref.buffer,
      config.workgroup_count_ref.offset, config.workgroup_count_ref.length,
      &payload.workgroup_count_ref);
  payload.dynamic_workgroup_local_memory =
      config.dynamic_workgroup_local_memory;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.constants_length = constants.data_length;
  payload.binding_count = bindings.count;

  iree_hal_replay_recorder_semaphore_storage_t semaphore_storage = {0};
  iree_hal_replay_recorder_buffer_ref_list_storage_t binding_storage = {0};
  iree_hal_dispatch_config_t base_config = config;
  iree_status_t status = iree_ok_status();
  if (executable_id != IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    status = iree_hal_replay_recorder_executable_recorded_ordinal(
        executable, function, &payload.function_ordinal);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_semaphore_storage_initialize(
        queue, wait_semaphore_list, signal_semaphore_list, &semaphore_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_ref_list_storage_initialize(
        queue, bindings, can_record, &binding_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_ref_unwrap_for_call(
        &base_config.workgroup_count_ref);
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id, executable_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_DISPATCH,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_dispatch(
        queue->base_queue, wait_semaphore_list, signal_semaphore_list,
        iree_hal_replay_recorder_executable_base_or_self(executable), function,
        base_config, constants, binding_storage.base_list, flags);
  }
  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(semaphore_storage.wait_payloads,
                                    semaphore_storage.wait_payloads_size),
          iree_make_const_byte_span(semaphore_storage.signal_payloads,
                                    semaphore_storage.signal_payloads_size),
          constants,
          iree_make_const_byte_span(binding_storage.payloads,
                                    binding_storage.payloads_size),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_hal_replay_recorder_buffer_ref_list_storage_deinitialize(
      queue->host_allocator, &binding_storage);
  iree_hal_replay_recorder_semaphore_storage_deinitialize(queue->host_allocator,
                                                          &semaphore_storage);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_atomic_wait(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const bool can_record =
      iree_hal_replay_recorder_queue_can_record_target_operation(
          queue, wait_semaphore_list, signal_semaphore_list, target_buffer);

  iree_hal_replay_queue_atomic_wait_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_queue_make_buffer_ref_payload(
      queue, target_buffer, target_offset,
      iree_hal_atomic_width_byte_count(params.width), &payload.target_ref);
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.params.value = params.value;
  payload.params.mask = params.mask;
  payload.params.flags = params.flags;
  payload.params.width = params.width;
  payload.params.condition = params.condition;
  payload.params.reserved0 = params.reserved;

  iree_hal_replay_recorder_semaphore_storage_t semaphore_storage = {0};
  iree_status_t status = iree_ok_status();
  if (can_record) {
    status = iree_hal_replay_recorder_semaphore_storage_initialize(
        queue, wait_semaphore_list, signal_semaphore_list, &semaphore_storage);
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_WAIT,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_WAIT
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }

  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, queue->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_atomic_wait(
        queue->base_queue, wait_semaphore_list, signal_semaphore_list,
        base_target_buffer, target_offset, params);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(semaphore_storage.wait_payloads,
                                    semaphore_storage.wait_payloads_size),
          iree_make_const_byte_span(semaphore_storage.signal_payloads,
                                    semaphore_storage.signal_payloads_size),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_hal_replay_recorder_semaphore_storage_deinitialize(queue->host_allocator,
                                                          &semaphore_storage);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_atomic_store(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const bool can_record =
      iree_hal_replay_recorder_queue_can_record_target_operation(
          queue, wait_semaphore_list, signal_semaphore_list, target_buffer);

  iree_hal_replay_queue_atomic_store_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_queue_make_buffer_ref_payload(
      queue, target_buffer, target_offset,
      iree_hal_atomic_width_byte_count(params.width), &payload.target_ref);
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.params.value = params.value;
  payload.params.flags = params.flags;
  payload.params.width = params.width;
  memcpy(payload.params.reserved0, params.reserved,
         sizeof(payload.params.reserved0));

  iree_hal_replay_recorder_semaphore_storage_t semaphore_storage = {0};
  iree_status_t status = iree_ok_status();
  if (can_record) {
    status = iree_hal_replay_recorder_semaphore_storage_initialize(
        queue, wait_semaphore_list, signal_semaphore_list, &semaphore_storage);
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_STORE,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_STORE
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }

  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, queue->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_atomic_store(
        queue->base_queue, wait_semaphore_list, signal_semaphore_list,
        base_target_buffer, target_offset, params);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(semaphore_storage.wait_payloads,
                                    semaphore_storage.wait_payloads_size),
          iree_make_const_byte_span(semaphore_storage.signal_payloads,
                                    semaphore_storage.signal_payloads_size),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_hal_replay_recorder_semaphore_storage_deinitialize(queue->host_allocator,
                                                          &semaphore_storage);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_atomic_rmw(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const bool can_record =
      iree_hal_replay_recorder_queue_can_record_target_operation(
          queue, wait_semaphore_list, signal_semaphore_list, target_buffer);

  iree_hal_replay_queue_atomic_rmw_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_queue_make_buffer_ref_payload(
      queue, target_buffer, target_offset,
      iree_hal_atomic_width_byte_count(params.width), &payload.target_ref);
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.params.operand = params.operand;
  payload.params.flags = params.flags;
  payload.params.width = params.width;
  payload.params.operation = params.operation;
  payload.params.reserved0 = params.reserved;

  iree_hal_replay_recorder_semaphore_storage_t semaphore_storage = {0};
  iree_status_t status = iree_ok_status();
  if (can_record) {
    status = iree_hal_replay_recorder_semaphore_storage_initialize(
        queue, wait_semaphore_list, signal_semaphore_list, &semaphore_storage);
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_RMW,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_RMW
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }

  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, queue->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_atomic_rmw(
        queue->base_queue, wait_semaphore_list, signal_semaphore_list,
        base_target_buffer, target_offset, params);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(semaphore_storage.wait_payloads,
                                    semaphore_storage.wait_payloads_size),
          iree_make_const_byte_span(semaphore_storage.signal_payloads,
                                    semaphore_storage.signal_payloads_size),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_hal_replay_recorder_semaphore_storage_deinitialize(queue->host_allocator,
                                                          &semaphore_storage);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_timestamp(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_timestamp_flags_t flags) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const bool can_record =
      iree_hal_replay_recorder_queue_can_record_target_operation(
          queue, wait_semaphore_list, signal_semaphore_list, target_buffer);

  iree_hal_replay_queue_timestamp_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_queue_make_buffer_ref_payload(
      queue, target_buffer, target_offset, sizeof(uint64_t),
      &payload.target_ref);
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_recorder_semaphore_storage_t semaphore_storage = {0};
  iree_status_t status = iree_ok_status();
  if (can_record) {
    status = iree_hal_replay_recorder_semaphore_storage_initialize(
        queue, wait_semaphore_list, signal_semaphore_list, &semaphore_storage);
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_TIMESTAMP,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TIMESTAMP
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }

  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, queue->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_timestamp(queue->base_queue, wait_semaphore_list,
                                      signal_semaphore_list, base_target_buffer,
                                      target_offset, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(semaphore_storage.wait_payloads,
                                    semaphore_storage.wait_payloads_size),
          iree_make_const_byte_span(semaphore_storage.signal_payloads,
                                    semaphore_storage.signal_payloads_size),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_hal_replay_recorder_semaphore_storage_deinitialize(queue->host_allocator,
                                                          &semaphore_storage);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_flush(
    iree_hal_queue_t* base_queue) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  iree_hal_replay_pending_record_t pending_record = {0};
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      queue->recorder, queue->device_id, queue->queue_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_FLUSH,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_queue_flush(queue->base_queue));
}

static iree_status_t iree_hal_replay_recorder_queue_prepare_transfer(
    iree_hal_replay_recorder_queue_t* queue, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations, bool can_record,
    iree_hal_replay_recorder_transfer_storage_t* storage) {
  iree_host_size_t data_offset = 0;
  for (iree_host_size_t i = 0; i < operation_count; ++i) {
    const iree_hal_transfer_operation_t* operation = &operations[i];
    iree_hal_transfer_operation_t* base_operation =
        &storage->base_operations[i];
    iree_hal_replay_queue_transfer_operation_payload_t* operation_payload =
        can_record ? &storage->operation_payloads[i] : NULL;
    iree_hal_buffer_t** temporary_source_buffer =
        &storage->temporary_buffers[i * 2];
    iree_hal_buffer_t** temporary_target_buffer =
        &storage->temporary_buffers[i * 2 + 1];
    base_operation->type = operation->type;
    switch (operation->type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL: {
        if (operation->fill.length == 0) break;
        *base_operation = *operation;
        IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_unwrap_for_call(
            operation->fill.target_buffer, queue->host_allocator,
            &base_operation->fill.target_buffer, temporary_target_buffer));
        if (can_record) {
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_FILL;
          operation_payload->flags = operation->fill.flags;
          iree_hal_replay_recorder_queue_make_buffer_ref_payload(
              queue, operation->fill.target_buffer,
              operation->fill.target_offset, operation->fill.length,
              &operation_payload->target_ref);
          operation_payload->data_offset = data_offset;
          operation_payload->data_length = operation->fill.pattern_length;
          memcpy(storage->data + data_offset, operation->fill.pattern,
                 operation->fill.pattern_length);
          data_offset += operation->fill.pattern_length;
        }
        break;
      }
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE: {
        if (operation->update.length == 0) break;
        *base_operation = *operation;
        IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_unwrap_for_call(
            operation->update.target_buffer, queue->host_allocator,
            &base_operation->update.target_buffer, temporary_target_buffer));
        if (can_record) {
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPDATE;
          operation_payload->flags = operation->update.flags;
          iree_hal_replay_recorder_queue_make_buffer_ref_payload(
              queue, operation->update.target_buffer,
              operation->update.target_offset, operation->update.length,
              &operation_payload->target_ref);
          operation_payload->data_offset = data_offset;
          operation_payload->data_length = operation->update.length;
          memcpy(storage->data + data_offset,
                 (const uint8_t*)operation->update.source_buffer +
                     operation->update.source_offset,
                 (iree_host_size_t)operation->update.length);
          data_offset += (iree_host_size_t)operation->update.length;
        }
        break;
      }
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY: {
        if (operation->copy.length == 0) break;
        *base_operation = *operation;
        IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_unwrap_for_call(
            operation->copy.source_buffer, queue->host_allocator,
            &base_operation->copy.source_buffer, temporary_source_buffer));
        IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_unwrap_for_call(
            operation->copy.target_buffer, queue->host_allocator,
            &base_operation->copy.target_buffer, temporary_target_buffer));
        if (can_record) {
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_COPY;
          operation_payload->flags = operation->copy.flags;
          iree_hal_replay_recorder_queue_make_buffer_ref_payload(
              queue, operation->copy.source_buffer,
              operation->copy.source_offset, operation->copy.length,
              &operation_payload->source_ref);
          iree_hal_replay_recorder_queue_make_buffer_ref_payload(
              queue, operation->copy.target_buffer,
              operation->copy.target_offset, operation->copy.length,
              &operation_payload->target_ref);
        }
        break;
      }
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD: {
        if (operation->upload.length == 0) break;
        *base_operation = *operation;
        IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_unwrap_for_call(
            operation->upload.target_buffer, queue->host_allocator,
            &base_operation->upload.target_buffer, temporary_target_buffer));
        if (can_record) {
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPLOAD;
          iree_hal_replay_recorder_queue_make_buffer_ref_payload(
              queue, operation->upload.target_buffer,
              operation->upload.target_offset, operation->upload.length,
              &operation_payload->target_ref);
          operation_payload->data_offset = data_offset;
          operation_payload->data_length = operation->upload.length;
          memcpy(storage->data + data_offset, operation->upload.source,
                 (iree_host_size_t)operation->upload.length);
          data_offset += (iree_host_size_t)operation->upload.length;
        }
        break;
      }
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD: {
        if (operation->download.length == 0) break;
        *base_operation = *operation;
        IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_unwrap_for_call(
            operation->download.source_buffer, queue->host_allocator,
            &base_operation->download.source_buffer, temporary_source_buffer));
        if (can_record) {
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_DOWNLOAD;
          iree_hal_replay_recorder_queue_make_buffer_ref_payload(
              queue, operation->download.source_buffer,
              operation->download.source_offset, operation->download.length,
              &operation_payload->source_ref);
        }
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown transfer operation type %u",
                                (uint32_t)operation->type);
    }
    if (can_record && operation_payload->type ==
                          IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_NONE) {
      switch (operation->type) {
        case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_FILL;
          break;
        case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPDATE;
          break;
        case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_COPY;
          break;
        case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPLOAD;
          break;
        case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
          operation_payload->type =
              IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_DOWNLOAD;
          break;
        default:
          break;
      }
    }
  }
  return iree_ok_status();
}

static void iree_hal_replay_recorder_queue_destroy(
    iree_hal_queue_t* base_queue) {
  // The proxy is embedded in its wrapper device allocation and has no
  // independently owned state.
  (void)base_queue;
}

static iree_status_t iree_hal_replay_recorder_queue_allocate_alloca_storage(
    iree_hal_replay_recorder_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests, bool can_record,
    iree_hal_replay_recorder_alloca_storage_t* out_storage) {
  iree_hal_replay_recorder_alloca_storage_t storage = {
      .request_count = request_count,
  };
  iree_status_t status = iree_hal_replay_recorder_queue_allocate_array(
      queue->host_allocator, request_count, sizeof(*storage.base_buffers),
      (void**)&storage.base_buffers);
  if (iree_status_is_ok(status)) {
    memset(storage.base_buffers, 0,
           request_count * sizeof(*storage.base_buffers));
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, request_count, sizeof(*storage.proxy_buffers),
        (void**)&storage.proxy_buffers);
  }
  if (iree_status_is_ok(status)) {
    memset(storage.proxy_buffers, 0,
           request_count * sizeof(*storage.proxy_buffers));
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, request_count, sizeof(*storage.replay_buffers),
        (void**)&storage.replay_buffers);
  }
  if (iree_status_is_ok(status)) {
    memset(storage.replay_buffers, 0,
           request_count * sizeof(*storage.replay_buffers));
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, request_count, sizeof(*storage.request_payloads),
        (void**)&storage.request_payloads);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, request_count, sizeof(*storage.object_payloads),
        (void**)&storage.object_payloads);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, request_count, sizeof(*storage.object_iovecs),
        (void**)&storage.object_iovecs);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, request_count, sizeof(*storage.created_objects),
        (void**)&storage.created_objects);
  }
  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_replay_recorder_buffer_allocate_proxy(
        queue->host_allocator, &storage.proxy_buffers[i]);
    if (iree_status_is_ok(status) && can_record) {
      iree_hal_replay_queue_alloca_request_payload_t* request_payload =
          &storage.request_payloads[i];
      status = iree_hal_replay_recorder_reserve_object_id(
          queue->recorder, &request_payload->buffer_id);
      if (iree_status_is_ok(status)) {
        iree_hal_buffer_params_t canonical_params = requests[i].params;
        iree_hal_buffer_params_canonicalize(&canonical_params);
        iree_hal_replay_recorder_allocator_make_allocate_buffer_payload(
            &canonical_params, requests[i].allocation_size,
            &request_payload->allocation);
      }
    }
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, wait_semaphore_list, queue->host_allocator,
        &storage.wait_payloads, &storage.wait_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signal_semaphore_list, queue->host_allocator,
        &storage.signal_payloads, &storage.signal_payloads_size);
  }
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_hal_replay_recorder_alloca_storage_deinitialize(queue->host_allocator,
                                                         &storage);
  }
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_alloca(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const bool can_record =
      iree_hal_replay_recorder_queue_has_captured_semaphores(
          queue->recorder, wait_semaphore_list) &&
      iree_hal_replay_recorder_queue_has_captured_semaphores(
          queue->recorder, signal_semaphore_list);

  iree_hal_replay_recorder_alloca_storage_t storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_queue_allocate_alloca_storage(
      queue, wait_semaphore_list, signal_semaphore_list, request_count,
      requests, can_record, &storage));

  iree_hal_replay_pending_record_t pending_record = {0};
  iree_status_t status = iree_hal_replay_recorder_begin_operation(
      queue->recorder, queue->device_id, queue->queue_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ALLOCA,
      can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ALLOCA
                 : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
      &pending_record);
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_alloca(queue->base_queue, wait_semaphore_list,
                                   signal_semaphore_list, pool, request_count,
                                   requests, storage.base_buffers);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      const iree_hal_replay_object_id_t buffer_id =
          can_record ? storage.request_payloads[i].buffer_id
                     : IREE_HAL_REPLAY_OBJECT_ID_NONE;
      storage.replay_buffers[i] =
          iree_hal_replay_recorder_buffer_initialize_proxy(
              queue->recorder, queue->device_id, buffer_id,
              queue->placement_device, storage.base_buffers[i],
              queue->host_allocator, storage.proxy_buffers[i]);
      storage.proxy_buffers[i] = NULL;
      if (can_record) {
        iree_hal_replay_recorder_buffer_make_object_payload(
            storage.base_buffers[i], &storage.object_payloads[i]);
        storage.object_iovecs[i] = iree_make_const_byte_span(
            &storage.object_payloads[i], sizeof(storage.object_payloads[i]));
        storage.created_objects[i] = (iree_hal_replay_created_object_record_t){
            .object_id = buffer_id,
            .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
            .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT,
            .iovec_count = 1,
            .iovecs = &storage.object_iovecs[i],
        };
      }
      iree_hal_buffer_release(storage.base_buffers[i]);
      storage.base_buffers[i] = NULL;
    }
  }

  if (pending_record.recorder) {
    if (can_record) {
      const iree_hal_replay_queue_alloca_payload_t payload = {
          .wait_semaphore_count = wait_semaphore_list.count,
          .signal_semaphore_count = signal_semaphore_list.count,
          .request_count = request_count,
      };
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(storage.wait_payloads,
                                    storage.wait_payloads_size),
          iree_make_const_byte_span(storage.signal_payloads,
                                    storage.signal_payloads_size),
          iree_make_const_byte_span(
              storage.request_payloads,
              request_count * sizeof(*storage.request_payloads)),
      };
      status = iree_hal_replay_recorder_end_creation_operation_list(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs,
          request_count, storage.created_objects);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_buffers[i] = storage.replay_buffers[i];
      storage.replay_buffers[i] = NULL;
    }
  }
  iree_hal_replay_recorder_alloca_storage_deinitialize(queue->host_allocator,
                                                       &storage);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_allocate_dealloca_storage(
    iree_hal_replay_recorder_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers,
    bool can_record, iree_hal_replay_recorder_dealloca_storage_t* out_storage) {
  iree_hal_replay_recorder_dealloca_storage_t storage = {
      .buffer_count = buffer_count,
  };
  iree_status_t status = iree_hal_replay_recorder_queue_allocate_array(
      queue->host_allocator, buffer_count, sizeof(*storage.base_buffers),
      (void**)&storage.base_buffers);
  if (iree_status_is_ok(status)) {
    memset(storage.base_buffers, 0,
           buffer_count * sizeof(*storage.base_buffers));
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, buffer_count, sizeof(*storage.temporary_buffers),
        (void**)&storage.temporary_buffers);
  }
  if (iree_status_is_ok(status)) {
    memset(storage.temporary_buffers, 0,
           buffer_count * sizeof(*storage.temporary_buffers));
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_queue_allocate_array(
        queue->host_allocator, buffer_count, sizeof(*storage.buffer_ids),
        (void**)&storage.buffer_ids);
  }
  for (iree_host_size_t i = 0; i < buffer_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        buffers[i], queue->host_allocator, &storage.base_buffers[i],
        &storage.temporary_buffers[i]);
    if (iree_status_is_ok(status) && can_record) {
      storage.buffer_ids[i] =
          iree_hal_replay_recorder_find_buffer_id(queue->recorder, buffers[i]);
    }
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, wait_semaphore_list, queue->host_allocator,
        &storage.wait_payloads, &storage.wait_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signal_semaphore_list, queue->host_allocator,
        &storage.signal_payloads, &storage.signal_payloads_size);
  }
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_hal_replay_recorder_dealloca_storage_deinitialize(
        queue->host_allocator, &storage);
  }
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_dealloca(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  bool can_record = iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, wait_semaphore_list) &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, signal_semaphore_list);
  for (iree_host_size_t i = 0; i < buffer_count && can_record; ++i) {
    can_record = iree_hal_replay_recorder_queue_has_captured_buffer(
        queue->recorder, buffers[i]);
  }

  iree_hal_replay_recorder_dealloca_storage_t storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_queue_allocate_dealloca_storage(
      queue, wait_semaphore_list, signal_semaphore_list, buffer_count, buffers,
      can_record, &storage));

  iree_hal_replay_pending_record_t pending_record = {0};
  iree_status_t status = iree_hal_replay_recorder_begin_operation(
      queue->recorder, queue->device_id, queue->queue_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_DEALLOCA,
      can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_DEALLOCA
                 : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
      &pending_record);
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_dealloca(queue->base_queue, wait_semaphore_list,
                                     signal_semaphore_list, buffer_count,
                                     storage.base_buffers);
  }
  if (pending_record.recorder) {
    if (can_record) {
      const iree_hal_replay_queue_dealloca_payload_t payload = {
          .wait_semaphore_count = wait_semaphore_list.count,
          .signal_semaphore_count = signal_semaphore_list.count,
          .buffer_count = buffer_count,
      };
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(storage.wait_payloads,
                                    storage.wait_payloads_size),
          iree_make_const_byte_span(storage.signal_payloads,
                                    storage.signal_payloads_size),
          iree_make_const_byte_span(storage.buffer_ids,
                                    buffer_count * sizeof(*storage.buffer_ids)),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }

  iree_hal_replay_recorder_dealloca_storage_deinitialize(queue->host_allocator,
                                                         &storage);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_transfer(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;

  bool can_record = false;
  iree_host_size_t data_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_queue_analyze_transfer(
      queue, wait_semaphore_list, signal_semaphore_list, operation_count,
      operations, &can_record, &data_length));

  iree_hal_replay_recorder_transfer_storage_t storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_queue_allocate_transfer_storage(
      queue, wait_semaphore_list, signal_semaphore_list, operation_count,
      can_record, data_length, &storage));
  iree_status_t status = iree_hal_replay_recorder_queue_prepare_transfer(
      queue, operation_count, operations, can_record, &storage);

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id,
        IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_TRANSFER,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TRANSFER
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_transfer(queue->base_queue, wait_semaphore_list,
                                     signal_semaphore_list, operation_count,
                                     storage.base_operations);
  }
  if (pending_record.recorder) {
    if (can_record) {
      const iree_hal_replay_queue_transfer_payload_t payload = {
          .wait_semaphore_count = wait_semaphore_list.count,
          .signal_semaphore_count = signal_semaphore_list.count,
          .operation_count = operation_count,
          .data_length = data_length,
      };
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(storage.wait_payloads,
                                    storage.wait_payloads_size),
          iree_make_const_byte_span(storage.signal_payloads,
                                    storage.signal_payloads_size),
          iree_make_const_byte_span(
              storage.operation_payloads,
              operation_count * sizeof(*storage.operation_payloads)),
          iree_make_const_byte_span(storage.data, data_length),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }

  iree_hal_replay_recorder_transfer_storage_deinitialize(queue->host_allocator,
                                                         &storage);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_read(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const iree_hal_replay_object_id_t source_file_id =
      iree_hal_replay_recorder_find_file_id(queue->recorder, source_file);
  const iree_hal_replay_object_id_t target_buffer_id =
      iree_hal_replay_recorder_find_buffer_id(queue->recorder, target_buffer);
  const bool uses_captured_ranges =
      iree_hal_replay_recorder_file_uses_captured_ranges(queue->recorder,
                                                         source_file);
  bool can_record = source_file_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
                    target_buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, wait_semaphore_list) &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, signal_semaphore_list);
  if (uses_captured_ranges && wait_semaphore_list.count != 0) {
    can_record = false;
  }

  iree_hal_replay_queue_read_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.source_file_id = source_file_id;
  payload.source_offset = source_offset;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  if (can_record) {
    iree_hal_replay_recorder_queue_make_buffer_ref_payload(
        queue, target_buffer, target_offset, length, &payload.target_ref);
  }

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_byte_span_t captured_data = iree_byte_span_empty();
  iree_status_t status = iree_ok_status();
  if (can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, wait_semaphore_list, queue->host_allocator,
        &wait_payloads, &wait_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signal_semaphore_list, queue->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record && uses_captured_ranges) {
    status = iree_hal_replay_recorder_file_capture_read_data(
        source_file, source_offset, length, queue->host_allocator,
        &captured_data);
    payload.captured_data_length = captured_data.data_length;
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id, target_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_READ,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }

  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, queue->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_read(
        queue->base_queue, wait_semaphore_list, signal_semaphore_list,
        iree_hal_replay_recorder_file_base_or_self(source_file), source_offset,
        base_target_buffer, target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);

  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(wait_payloads, wait_payloads_size),
          iree_make_const_byte_span(signal_payloads, signal_payloads_size),
          iree_make_const_byte_span(captured_data.data,
                                    captured_data.data_length),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_allocator_free(queue->host_allocator, captured_data.data);
  iree_allocator_free(queue->host_allocator, signal_payloads);
  iree_allocator_free(queue->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_recorder_queue_write(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_replay_recorder_queue_t* queue =
      (iree_hal_replay_recorder_queue_t*)base_queue;
  const iree_hal_replay_object_id_t source_buffer_id =
      iree_hal_replay_recorder_find_buffer_id(queue->recorder, source_buffer);
  const iree_hal_replay_object_id_t target_file_id =
      iree_hal_replay_recorder_find_file_id(queue->recorder, target_file);
  bool can_record = source_buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
                    target_file_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, wait_semaphore_list) &&
                    iree_hal_replay_recorder_queue_has_captured_semaphores(
                        queue->recorder, signal_semaphore_list);
  if (iree_hal_replay_recorder_file_uses_captured_ranges(queue->recorder,
                                                         target_file)) {
    can_record = false;
  }

  iree_hal_replay_queue_write_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.target_file_id = target_file_id;
  payload.target_offset = target_offset;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  if (can_record) {
    iree_hal_replay_recorder_queue_make_buffer_ref_payload(
        queue, source_buffer, source_offset, length, &payload.source_ref);
  }

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_ok_status();
  if (can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, wait_semaphore_list, queue->host_allocator,
        &wait_payloads, &wait_payloads_size);
  }
  if (iree_status_is_ok(status) && can_record) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        queue->recorder, signal_semaphore_list, queue->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        queue->recorder, queue->device_id, queue->queue_id, source_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
        IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_WRITE,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_WRITE
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }

  iree_hal_buffer_t* base_source_buffer = NULL;
  iree_hal_buffer_t* temporary_source_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        source_buffer, queue->host_allocator, &base_source_buffer,
        &temporary_source_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_write(
        queue->base_queue, wait_semaphore_list, signal_semaphore_list,
        base_source_buffer, source_offset,
        iree_hal_replay_recorder_file_base_or_self(target_file), target_offset,
        length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_source_buffer);

  if (pending_record.recorder) {
    if (can_record) {
      const iree_const_byte_span_t iovecs[] = {
          iree_make_const_byte_span(&payload, sizeof(payload)),
          iree_make_const_byte_span(wait_payloads, wait_payloads_size),
          iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      };
      status = iree_hal_replay_recorder_end_operation_with_payload(
          &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
    } else {
      status = iree_hal_replay_recorder_end_operation(&pending_record, status);
    }
  }
  iree_allocator_free(queue->host_allocator, signal_payloads);
  iree_allocator_free(queue->host_allocator, wait_payloads);
  return status;
}

static const iree_hal_queue_vtable_t iree_hal_replay_recorder_queue_vtable = {
    .destroy = iree_hal_replay_recorder_queue_destroy,
    .barrier = iree_hal_replay_recorder_queue_barrier,
    .execute = iree_hal_replay_recorder_queue_execute,
    .host_call = iree_hal_replay_recorder_queue_host_call,
    .dispatch = iree_hal_replay_recorder_queue_dispatch,
    .atomic_wait = iree_hal_replay_recorder_queue_atomic_wait,
    .atomic_store = iree_hal_replay_recorder_queue_atomic_store,
    .atomic_rmw = iree_hal_replay_recorder_queue_atomic_rmw,
    .timestamp = iree_hal_replay_recorder_queue_timestamp,
    .flush = iree_hal_replay_recorder_queue_flush,
    .alloca = iree_hal_replay_recorder_queue_alloca,
    .dealloca = iree_hal_replay_recorder_queue_dealloca,
    .transfer = iree_hal_replay_recorder_queue_transfer,
    .read = iree_hal_replay_recorder_queue_read,
    .write = iree_hal_replay_recorder_queue_write,
};

void iree_hal_replay_recorder_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t queue_id, iree_hal_queue_t* base_queue,
    iree_hal_device_t* placement_device, iree_allocator_t host_allocator,
    iree_hal_replay_recorder_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_queue);
  IREE_ASSERT_ARGUMENT(placement_device);
  IREE_ASSERT_ARGUMENT(out_queue);
  iree_hal_queue_initialize(
      queue_family, &iree_hal_replay_recorder_queue_vtable, &out_queue->base);
  out_queue->host_allocator = host_allocator;
  out_queue->recorder = recorder;
  out_queue->base_queue = base_queue;
  out_queue->placement_device = placement_device;
  out_queue->device_id = device_id;
  out_queue->queue_id = queue_id;
}
