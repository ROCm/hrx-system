// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_queue.h"

#include <string.h>

#include "iree/hal/replay/recorder_buffer.h"
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

static iree_status_t iree_hal_replay_recorder_queue_allocate_array(
    iree_allocator_t host_allocator, iree_host_size_t element_count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (element_count == 0) return iree_ok_status();
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(element_count, element_size,
                                                &allocation_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "replay transfer storage size overflow");
  }
  return iree_allocator_malloc(host_allocator, allocation_size, out_ptr);
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
  return iree_hal_replay_recorder_find_buffer_id(recorder, buffer) !=
         IREE_HAL_REPLAY_OBJECT_ID_NONE;
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

static const iree_hal_queue_vtable_t iree_hal_replay_recorder_queue_vtable = {
    .destroy = iree_hal_replay_recorder_queue_destroy,
    .transfer = iree_hal_replay_recorder_queue_transfer,
};

void iree_hal_replay_recorder_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t queue_id, iree_hal_queue_t* base_queue,
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_queue);
  IREE_ASSERT_ARGUMENT(out_queue);
  iree_hal_queue_initialize(
      queue_family, &iree_hal_replay_recorder_queue_vtable, &out_queue->base);
  out_queue->host_allocator = host_allocator;
  out_queue->recorder = recorder;
  out_queue->base_queue = base_queue;
  out_queue->device_id = device_id;
  out_queue->queue_id = queue_id;
}
