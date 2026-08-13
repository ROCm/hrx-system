// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute_operation.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/replay/execute_object.h"

static iree_status_t iree_hal_replay_executor_scope_event(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_scope_event_type_t event_type) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE,
      sizeof(iree_hal_replay_scope_payload_t)));
  iree_hal_replay_scope_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.flags != IREE_HAL_REPLAY_SCOPE_FLAG_NONE ||
                    payload.reserved0 != 0 || payload.reserved1 != 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay scope payload reserved fields must be "
                            "zero");
  }
  if (IREE_UNLIKELY(payload.name_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) + (iree_host_size_t)payload.name_length !=
                        record->payload.data_length ||
                    payload.name_length == 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay scope payload name length mismatch");
  }

  iree_hal_replay_scope_event_callback_t callback =
      executor->options->scope_event_callback;
  if (!callback.fn) return iree_ok_status();

  iree_hal_replay_scope_event_t event = {
      .sequence_ordinal = record->header.sequence_ordinal,
      .type = event_type,
      .name = iree_make_string_view(
          (const char*)record->payload.data + sizeof(payload),
          (iree_host_size_t)payload.name_length),
  };
  return callback.fn(callback.user_data, &event);
}

static iree_status_t iree_hal_replay_executor_queue_alloca(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA,
      sizeof(iree_hal_replay_device_queue_alloca_payload_t)));
  iree_hal_replay_device_queue_alloca_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_params_t params;
    status = iree_hal_replay_executor_make_buffer_params(&payload.allocation,
                                                         &params);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_alloca(
          device_entry->value.device, payload.queue_affinity, wait_storage.list,
          signal_storage.list, /*pool=*/NULL, params,
          payload.allocation.allocation_size, payload.flags, &buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_replay_object_entry_t entry = {.value.buffer = buffer};
    buffer = NULL;
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, entry);
  }
  iree_hal_buffer_release(buffer);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_dealloca(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA,
      sizeof(iree_hal_replay_device_queue_dealloca_payload_t)));
  iree_hal_replay_device_queue_dealloca_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_ref_t buffer_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.buffer_ref, &buffer_ref);
  }
  if (iree_status_is_ok(status) && !buffer_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue dealloca requires a direct buffer reference");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, buffer_ref.buffer, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_fill(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_FILL,
      sizeof(iree_hal_replay_device_queue_fill_payload_t)));
  iree_hal_replay_device_queue_fill_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t pattern;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, payload.pattern_length, &wait_storage,
      &signal_storage, &pattern);
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  if (iree_status_is_ok(status) && !target_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue fill requires a direct target buffer reference");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_fill(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, target_ref.buffer, target_ref.offset,
        target_ref.length, pattern.data, pattern.data_length, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_update(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE,
      sizeof(iree_hal_replay_device_queue_update_payload_t)));
  iree_hal_replay_device_queue_update_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t data;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, payload.data_length, &wait_storage,
      &signal_storage, &data);
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  if (iree_status_is_ok(status) && !target_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue update requires a direct target buffer reference");
  }
  if (iree_status_is_ok(status) && data.data_length != target_ref.length) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue update data length does not match target length");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, data.data, /*source_offset=*/0, target_ref.buffer,
        target_ref.offset, target_ref.length, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_copy(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_COPY,
      sizeof(iree_hal_replay_device_queue_copy_payload_t)));
  iree_hal_replay_device_queue_copy_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_ref_t source_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.source_ref, &source_ref);
  }
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  if (iree_status_is_ok(status) && (!source_ref.buffer || !target_ref.buffer)) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "replay queue copy requires direct source and "
                              "target buffer references");
  }
  if (iree_status_is_ok(status) && source_ref.length != target_ref.length) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue copy source and target lengths do not match");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, source_ref.buffer, source_ref.offset,
        target_ref.buffer, target_ref.offset, target_ref.length, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_read(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ,
      sizeof(iree_hal_replay_device_queue_read_payload_t)));
  iree_hal_replay_device_queue_read_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, payload.captured_data_length,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_replay_object_entry_t* file_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, payload.source_file_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_FILE,
                                             &file_entry);
  }
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  if (iree_status_is_ok(status) && !target_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue read requires a direct target buffer reference");
  }
  if (iree_status_is_ok(status) && payload.captured_data_length != 0 &&
      payload.captured_data_length != target_ref.length) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue read captured data length does not match target length");
  }
  if (iree_status_is_ok(status) && payload.captured_data_length != 0) {
    status = iree_hal_device_queue_update(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, trailing_payload.data, /*source_offset=*/0,
        target_ref.buffer, target_ref.offset, target_ref.length,
        IREE_HAL_UPDATE_FLAG_NONE);
  } else if (iree_status_is_ok(status)) {
    if (!file_entry->value.file) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "replay queue read requires an imported file or captured data");
    }
  }
  if (iree_status_is_ok(status) && payload.captured_data_length == 0) {
    status = iree_hal_device_queue_read(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, file_entry->value.file, payload.source_offset,
        target_ref.buffer, target_ref.offset, target_ref.length, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_write(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE,
      sizeof(iree_hal_replay_device_queue_write_payload_t)));
  iree_hal_replay_device_queue_write_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_ref_t source_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.source_ref, &source_ref);
  }
  iree_hal_replay_object_entry_t* file_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, payload.target_file_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_FILE,
                                             &file_entry);
  }
  if (iree_status_is_ok(status) && !source_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue write requires a direct source buffer reference");
  }
  if (iree_status_is_ok(status) && !file_entry->value.file) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "replay queue write requires an imported target file");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, source_ref.buffer, source_ref.offset,
        file_entry->value.file, payload.target_offset, source_ref.length,
        payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_command_buffer_execution_barrier(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER,
      sizeof(iree_hal_replay_command_buffer_execution_barrier_payload_t)));
  iree_hal_replay_command_buffer_execution_barrier_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t memory_payloads_size = 0;
  iree_host_size_t buffer_payloads_size = 0;
  iree_host_size_t total_payload_size = 0;
  if (IREE_UNLIKELY(
          payload.memory_barrier_count > IREE_HOST_SIZE_MAX ||
          payload.buffer_barrier_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.memory_barrier_count,
              sizeof(iree_hal_replay_memory_barrier_payload_t),
              &memory_payloads_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.buffer_barrier_count,
              sizeof(iree_hal_replay_buffer_barrier_payload_t),
              &buffer_payloads_size) ||
          !iree_host_size_checked_add(sizeof(payload), memory_payloads_size,
                                      &total_payload_size) ||
          !iree_host_size_checked_add(total_payload_size, buffer_payloads_size,
                                      &total_payload_size) ||
          total_payload_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay execution barrier payload length mismatch");
  }

  iree_hal_memory_barrier_t inline_memory_barriers
      [IREE_HAL_REPLAY_INLINE_MEMORY_BARRIER_LIST_CAPACITY];
  iree_hal_memory_barrier_t* memory_barriers = NULL;
  bool memory_barriers_allocated = false;
  if (payload.memory_barrier_count <=
      IREE_HAL_REPLAY_INLINE_MEMORY_BARRIER_LIST_CAPACITY) {
    memory_barriers = inline_memory_barriers;
  } else {
    iree_host_size_t memory_barriers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            (iree_host_size_t)payload.memory_barrier_count,
            sizeof(*memory_barriers), &memory_barriers_size))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay execution barrier memory barrier count overflow");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(executor->host_allocator,
                                               memory_barriers_size,
                                               (void**)&memory_barriers));
    memory_barriers_allocated = true;
  }
  const iree_hal_replay_memory_barrier_payload_t* memory_payloads =
      (const iree_hal_replay_memory_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload));
  for (iree_host_size_t i = 0; i < payload.memory_barrier_count; ++i) {
    memory_barriers[i].source_scope = memory_payloads[i].source_scope;
    memory_barriers[i].target_scope = memory_payloads[i].target_scope;
  }

  iree_hal_buffer_barrier_t inline_buffer_barriers
      [IREE_HAL_REPLAY_INLINE_BUFFER_BARRIER_LIST_CAPACITY];
  iree_hal_buffer_barrier_t* buffer_barriers = NULL;
  iree_status_t status = iree_ok_status();
  bool buffer_barriers_allocated = false;
  if (payload.buffer_barrier_count <=
      IREE_HAL_REPLAY_INLINE_BUFFER_BARRIER_LIST_CAPACITY) {
    buffer_barriers = inline_buffer_barriers;
  } else {
    iree_host_size_t buffer_barriers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            (iree_host_size_t)payload.buffer_barrier_count,
            sizeof(*buffer_barriers), &buffer_barriers_size))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay execution barrier buffer barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator, buffer_barriers_size,
                                (void**)&buffer_barriers);
    }
    buffer_barriers_allocated = iree_status_is_ok(status);
  }
  const iree_hal_replay_buffer_barrier_payload_t* buffer_payloads =
      (const iree_hal_replay_buffer_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload) +
                                                        memory_payloads_size);
  for (iree_host_size_t i = 0;
       i < payload.buffer_barrier_count && iree_status_is_ok(status); ++i) {
    buffer_barriers[i].source_scope = buffer_payloads[i].source_scope;
    buffer_barriers[i].target_scope = buffer_payloads[i].target_scope;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &buffer_payloads[i].buffer_ref,
        &buffer_barriers[i].buffer_ref);
  }

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, record->header.object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_execution_barrier(
        command_buffer_entry->value.command_buffer, payload.source_stage_mask,
        payload.target_stage_mask, payload.flags,
        (iree_host_size_t)payload.memory_barrier_count, memory_barriers,
        (iree_host_size_t)payload.buffer_barrier_count, buffer_barriers);
  }
  if (buffer_barriers_allocated) {
    iree_allocator_free(executor->host_allocator, buffer_barriers);
  }
  if (memory_barriers_allocated) {
    iree_allocator_free(executor->host_allocator, memory_barriers);
  }
  return status;
}

static iree_status_t iree_hal_replay_executor_command_buffer_event(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EVENT,
      sizeof(iree_hal_replay_command_buffer_event_payload_t)));
  iree_hal_replay_command_buffer_event_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_replay_object_entry_t* event_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, payload.event_id, IREE_HAL_REPLAY_OBJECT_TYPE_EVENT,
      &event_entry));

  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_SIGNAL_EVENT:
      return iree_hal_command_buffer_signal_event(
          command_buffer_entry->value.command_buffer, event_entry->value.event,
          payload.source_stage_mask);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_RESET_EVENT:
      return iree_hal_command_buffer_reset_event(
          command_buffer_entry->value.command_buffer, event_entry->value.event,
          payload.source_stage_mask);
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED, "replay operation %s is not an event op",
          iree_hal_replay_operation_code_string(record->header.operation_code));
  }
}

static iree_status_t iree_hal_replay_executor_command_buffer_wait_events(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_WAIT_EVENTS,
      sizeof(iree_hal_replay_command_buffer_wait_events_payload_t)));
  iree_hal_replay_command_buffer_wait_events_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_host_size_t event_payloads_size = 0;
  iree_host_size_t memory_payloads_size = 0;
  iree_host_size_t buffer_payloads_size = 0;
  iree_host_size_t total_payload_size = 0;
  if (IREE_UNLIKELY(
          payload.event_count > IREE_HOST_SIZE_MAX ||
          payload.memory_barrier_count > IREE_HOST_SIZE_MAX ||
          payload.buffer_barrier_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul((iree_host_size_t)payload.event_count,
                                      sizeof(iree_hal_replay_object_id_t),
                                      &event_payloads_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.memory_barrier_count,
              sizeof(iree_hal_replay_memory_barrier_payload_t),
              &memory_payloads_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.buffer_barrier_count,
              sizeof(iree_hal_replay_buffer_barrier_payload_t),
              &buffer_payloads_size) ||
          !iree_host_size_checked_add(sizeof(payload), event_payloads_size,
                                      &total_payload_size) ||
          !iree_host_size_checked_add(total_payload_size, memory_payloads_size,
                                      &total_payload_size) ||
          !iree_host_size_checked_add(total_payload_size, buffer_payloads_size,
                                      &total_payload_size) ||
          total_payload_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay wait events payload length mismatch");
  }

  const iree_hal_replay_object_id_t* event_payloads =
      (const iree_hal_replay_object_id_t*)(record->payload.data +
                                           sizeof(payload));
  const iree_hal_replay_memory_barrier_payload_t* memory_payloads =
      (const iree_hal_replay_memory_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload) +
                                                        event_payloads_size);
  const iree_hal_replay_buffer_barrier_payload_t* buffer_payloads =
      (const iree_hal_replay_buffer_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload) +
                                                        event_payloads_size +
                                                        memory_payloads_size);

  const iree_hal_event_t** events = NULL;
  iree_status_t status = iree_ok_status();
  iree_host_size_t events_size = 0;
  if (payload.event_count) {
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul((iree_host_size_t)payload.event_count,
                                        sizeof(*events), &events_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay wait event count overflow");
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(executor->host_allocator, events_size,
                                     (void**)&events);
    }
  }
  for (iree_host_size_t i = 0;
       i < payload.event_count && iree_status_is_ok(status); ++i) {
    iree_hal_replay_object_entry_t* event_entry = NULL;
    status = iree_hal_replay_executor_lookup(executor, event_payloads[i],
                                             IREE_HAL_REPLAY_OBJECT_TYPE_EVENT,
                                             &event_entry);
    if (iree_status_is_ok(status)) {
      events[i] = event_entry->value.event;
    }
  }

  iree_hal_memory_barrier_t* memory_barriers = NULL;
  iree_host_size_t memory_barriers_size = 0;
  if (iree_status_is_ok(status) && payload.memory_barrier_count) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            (iree_host_size_t)payload.memory_barrier_count,
            sizeof(*memory_barriers), &memory_barriers_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay wait memory barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator, memory_barriers_size,
                                (void**)&memory_barriers);
    }
  }
  for (iree_host_size_t i = 0;
       i < payload.memory_barrier_count && iree_status_is_ok(status); ++i) {
    memory_barriers[i].source_scope = memory_payloads[i].source_scope;
    memory_barriers[i].target_scope = memory_payloads[i].target_scope;
  }

  iree_hal_buffer_barrier_t* buffer_barriers = NULL;
  iree_host_size_t buffer_barriers_size = 0;
  if (iree_status_is_ok(status) && payload.buffer_barrier_count) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            (iree_host_size_t)payload.buffer_barrier_count,
            sizeof(*buffer_barriers), &buffer_barriers_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay wait buffer barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator, buffer_barriers_size,
                                (void**)&buffer_barriers);
    }
  }
  for (iree_host_size_t i = 0;
       i < payload.buffer_barrier_count && iree_status_is_ok(status); ++i) {
    buffer_barriers[i].source_scope = buffer_payloads[i].source_scope;
    buffer_barriers[i].target_scope = buffer_payloads[i].target_scope;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &buffer_payloads[i].buffer_ref,
        &buffer_barriers[i].buffer_ref);
  }

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, record->header.object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_wait_events(
        command_buffer_entry->value.command_buffer,
        (iree_host_size_t)payload.event_count, events,
        payload.source_stage_mask, payload.target_stage_mask,
        (iree_host_size_t)payload.memory_barrier_count, memory_barriers,
        (iree_host_size_t)payload.buffer_barrier_count, buffer_barriers);
  }

  iree_allocator_free(executor->host_allocator, buffer_barriers);
  iree_allocator_free(executor->host_allocator, memory_barriers);
  iree_allocator_free(executor->host_allocator, events);
  return status;
}

static iree_status_t iree_hal_replay_executor_command_buffer_dispatch(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH,
      sizeof(iree_hal_replay_dispatch_payload_t)));
  iree_hal_replay_dispatch_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (payload.wait_semaphore_count != 0 ||
      payload.signal_semaphore_count != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "command buffer dispatch semaphore lists are "
                            "reserved for immediate dispatch");
  }
  iree_host_size_t wait_offset = 0;
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_offset = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t binding_offset = 0;
  iree_host_size_t binding_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_dispatch_layout(
      record, &payload, &wait_offset, &wait_size, &signal_offset, &signal_size,
      &constants_offset, &binding_offset, &binding_size));
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(record->payload.data + constants_offset,
                                (iree_host_size_t)payload.constants_length);
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)(record->payload.data +
                                                    binding_offset);
  iree_hal_replay_buffer_ref_list_storage_t binding_storage = {0};
  IREE_RETURN_IF_ERROR(iree_hal_replay_buffer_ref_list_storage_initialize(
      executor, (iree_host_size_t)payload.binding_count, &binding_storage));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < payload.binding_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &binding_payloads[i], &binding_storage.values[i]);
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, record->header.object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  iree_hal_replay_object_entry_t* executable_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, payload.executable_id, IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE,
        &executable_entry);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_dispatch_config_t config;
    memset(&config, 0, sizeof(config));
    memcpy(config.workgroup_size, payload.workgroup_size,
           sizeof(config.workgroup_size));
    memcpy(config.workgroup_count, payload.workgroup_count,
           sizeof(config.workgroup_count));
    config.dynamic_workgroup_local_memory =
        payload.dynamic_workgroup_local_memory;
    iree_hal_executable_function_t function =
        iree_hal_executable_function_invalid();
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.workgroup_count_ref, &config.workgroup_count_ref);
    if (iree_status_is_ok(status)) {
      status = iree_hal_replay_executor_resolve_function(
          payload.executable_id, &executable_entry->value.executable,
          payload.function_ordinal, &function);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_command_buffer_dispatch(
          command_buffer_entry->value.command_buffer,
          executable_entry->value.executable.handle, function, config,
          constants, binding_storage.list, payload.flags);
    }
  }
  iree_hal_replay_buffer_ref_list_storage_deinitialize(
      &binding_storage, executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_dispatch(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH,
      sizeof(iree_hal_replay_dispatch_payload_t)));
  iree_hal_replay_dispatch_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t wait_offset = 0;
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_offset = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t binding_offset = 0;
  iree_host_size_t binding_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_dispatch_layout(
      record, &payload, &wait_offset, &wait_size, &signal_offset, &signal_size,
      &constants_offset, &binding_offset, &binding_size));
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(record->payload.data + constants_offset,
                                (iree_host_size_t)payload.constants_length);
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)(record->payload.data +
                                                    binding_offset);

  iree_hal_replay_semaphore_list_storage_t wait_storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_semaphore_list(
      executor,
      iree_make_const_byte_span(record->payload.data + wait_offset, wait_size),
      (iree_host_size_t)payload.wait_semaphore_count, &wait_storage));
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor,
      iree_make_const_byte_span(record->payload.data + signal_offset,
                                signal_size),
      (iree_host_size_t)payload.signal_semaphore_count, &signal_storage);

  iree_hal_replay_buffer_ref_list_storage_t binding_storage = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_buffer_ref_list_storage_initialize(
        executor, (iree_host_size_t)payload.binding_count, &binding_storage);
  }
  for (iree_host_size_t i = 0;
       i < payload.binding_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &binding_payloads[i], &binding_storage.values[i]);
  }
  iree_hal_replay_object_entry_t* device_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, record->header.object_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
                                             &device_entry);
  }
  iree_hal_replay_object_entry_t* executable_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, payload.executable_id, IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE,
        &executable_entry);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_dispatch_config_t config;
    memset(&config, 0, sizeof(config));
    memcpy(config.workgroup_size, payload.workgroup_size,
           sizeof(config.workgroup_size));
    memcpy(config.workgroup_count, payload.workgroup_count,
           sizeof(config.workgroup_count));
    config.dynamic_workgroup_local_memory =
        payload.dynamic_workgroup_local_memory;
    iree_hal_executable_function_t function =
        iree_hal_executable_function_invalid();
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.workgroup_count_ref, &config.workgroup_count_ref);
    if (iree_status_is_ok(status)) {
      status = iree_hal_replay_executor_resolve_function(
          payload.executable_id, &executable_entry->value.executable,
          payload.function_ordinal, &function);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_dispatch(
          device_entry->value.device, payload.queue_affinity, wait_storage.list,
          signal_storage.list, executable_entry->value.executable.handle,
          function, config, constants, binding_storage.list, payload.flags);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }

  iree_hal_replay_buffer_ref_list_storage_deinitialize(
      &binding_storage, executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_command_buffer_fill_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_FILL_BUFFER,
      sizeof(iree_hal_replay_command_buffer_fill_buffer_payload_t)));
  iree_hal_replay_command_buffer_fill_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.pattern_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) +
                            (iree_host_size_t)payload.pattern_length !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay command buffer fill payload length "
                            "mismatch");
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_ref(
      executor, &payload.target_ref, &target_ref));
  iree_const_byte_span_t pattern =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.pattern_length);
  return iree_hal_command_buffer_fill_buffer(
      command_buffer_entry->value.command_buffer, target_ref, pattern.data,
      pattern.data_length, payload.flags);
}

static iree_status_t iree_hal_replay_executor_command_buffer_update_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_UPDATE_BUFFER,
      sizeof(iree_hal_replay_command_buffer_update_buffer_payload_t)));
  iree_hal_replay_command_buffer_update_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.data_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) + (iree_host_size_t)payload.data_length !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay command buffer update payload length "
                            "mismatch");
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_ref(
      executor, &payload.target_ref, &target_ref));
  if (payload.data_length != target_ref.length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay command buffer update data length does not "
                            "match target length");
  }
  iree_const_byte_span_t data =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.data_length);
  return iree_hal_command_buffer_update_buffer(
      command_buffer_entry->value.command_buffer, data.data,
      /*source_offset=*/0, target_ref, payload.flags);
}

static iree_status_t iree_hal_replay_executor_command_buffer_copy_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_COPY_BUFFER,
      sizeof(iree_hal_replay_command_buffer_copy_buffer_payload_t)));
  iree_hal_replay_command_buffer_copy_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t source_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_ref(
      executor, &payload.source_ref, &source_ref));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_ref(
      executor, &payload.target_ref, &target_ref));
  return iree_hal_command_buffer_copy_buffer(
      command_buffer_entry->value.command_buffer, source_ref, target_ref,
      payload.flags);
}

static iree_status_t iree_hal_replay_executor_queue_execute(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE,
      sizeof(iree_hal_replay_device_queue_execute_payload_t)));
  iree_hal_replay_device_queue_execute_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t binding_size = 0;
  iree_host_size_t expected_size = 0;
  if (IREE_UNLIKELY(payload.wait_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.signal_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.binding_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.wait_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &wait_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.signal_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &signal_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.binding_count,
                        sizeof(iree_hal_replay_buffer_ref_payload_t),
                        &binding_size) ||
                    !iree_host_size_checked_add(sizeof(payload), wait_size,
                                                &expected_size) ||
                    !iree_host_size_checked_add(expected_size, signal_size,
                                                &expected_size) ||
                    !iree_host_size_checked_add(expected_size, binding_size,
                                                &expected_size) ||
                    expected_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue execute payload length mismatch");
  }
  const uint8_t* cursor = record->payload.data + sizeof(payload);
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, wait_size),
      (iree_host_size_t)payload.wait_semaphore_count, &wait_storage));
  cursor += wait_size;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, signal_size),
      (iree_host_size_t)payload.signal_semaphore_count, &signal_storage);
  cursor += signal_size;

  iree_hal_replay_buffer_binding_table_storage_t binding_storage = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_buffer_binding_table_storage_initialize(
        executor, (iree_host_size_t)payload.binding_count, &binding_storage);
  }
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)cursor;
  for (iree_host_size_t i = 0;
       i < payload.binding_count && iree_status_is_ok(status); ++i) {
    iree_hal_buffer_ref_t ref;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &binding_payloads[i], &ref);
    if (iree_status_is_ok(status)) {
      binding_storage.bindings[i] = (iree_hal_buffer_binding_t){
          .buffer = ref.buffer,
          .offset = ref.offset,
          .length = ref.length,
      };
    }
  }

  iree_hal_replay_object_entry_t* device_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, record->header.object_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
                                             &device_entry);
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status) &&
      payload.command_buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    status = iree_hal_replay_executor_lookup(
        executor, payload.command_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  if (iree_status_is_ok(status)) {
    if (command_buffer_entry) {
      status = iree_hal_device_queue_execute(
          device_entry->value.device, payload.queue_affinity, wait_storage.list,
          signal_storage.list, command_buffer_entry->value.command_buffer,
          binding_storage.table, payload.flags);
    } else if (payload.binding_count == 0) {
      status = iree_hal_device_queue_barrier(
          device_entry->value.device, payload.queue_affinity, wait_storage.list,
          signal_storage.list, payload.flags);
    } else {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "replay queue barrier payload unexpectedly has bindings");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(device_entry->value.device,
                                         payload.queue_affinity);
  }
  if (iree_status_is_ok(status) && signal_storage.list.count != 0) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_hal_replay_buffer_binding_table_storage_deinitialize(
      &binding_storage, executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

iree_status_t iree_hal_replay_executor_replay_operation(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  if (IREE_UNLIKELY(record->header.status_code != IREE_STATUS_OK)) {
    // Failed calls produced no replay object and are preserved only to explain
    // host fallback branches. Replay follows the later successful calls the
    // original host issued after observing the failure.
    return iree_ok_status();
  }
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN:
      return iree_hal_replay_executor_scope_event(
          executor, record, IREE_HAL_REPLAY_SCOPE_EVENT_TYPE_BEGIN);
    case IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_END:
      return iree_hal_replay_executor_scope_event(
          executor, record, IREE_HAL_REPLAY_SCOPE_EVENT_TYPE_END);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_TRIM:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_QUEUE_POOL_BACKEND:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_BEGIN:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_FLUSH:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_END:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_EXTERNAL_CAPTURE_BEGIN:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_EXTERNAL_CAPTURE_END:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_TRIM:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_QUERY_MEMORY_HEAPS:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_INVALIDATE_RANGE:
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_BEGIN_DEBUG_GROUP:
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_END_DEBUG_GROUP:
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ADVISE_BUFFER:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_COUNT:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_INFO:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_PARAMETERS:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_LOOKUP_FUNCTION_BY_NAME:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_LOAD_EXECUTABLE:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EVENT:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE:
      return iree_hal_replay_executor_replay_object_operation(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA:
      return iree_hal_replay_executor_queue_alloca(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA:
      return iree_hal_replay_executor_queue_dealloca(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FILL:
      return iree_hal_replay_executor_queue_fill(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE:
      return iree_hal_replay_executor_queue_update(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_COPY:
      return iree_hal_replay_executor_queue_copy(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ:
      return iree_hal_replay_executor_queue_read(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE:
      return iree_hal_replay_executor_queue_write(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_BEGIN: {
      iree_hal_replay_object_entry_t* entry = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
          executor, record->header.object_id,
          IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &entry));
      return iree_hal_command_buffer_begin(entry->value.command_buffer);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_END: {
      iree_hal_replay_object_entry_t* entry = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
          executor, record->header.object_id,
          IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &entry));
      return iree_hal_command_buffer_end(entry->value.command_buffer);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_EXECUTION_BARRIER:
      return iree_hal_replay_executor_command_buffer_execution_barrier(executor,
                                                                       record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_SIGNAL_EVENT:
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_RESET_EVENT:
      return iree_hal_replay_executor_command_buffer_event(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_WAIT_EVENTS:
      return iree_hal_replay_executor_command_buffer_wait_events(executor,
                                                                 record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_DISPATCH:
      return iree_hal_replay_executor_command_buffer_dispatch(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_FILL_BUFFER:
      return iree_hal_replay_executor_command_buffer_fill_buffer(executor,
                                                                 record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_UPDATE_BUFFER:
      return iree_hal_replay_executor_command_buffer_update_buffer(executor,
                                                                   record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_COPY_BUFFER:
      return iree_hal_replay_executor_command_buffer_copy_buffer(executor,
                                                                 record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH:
      return iree_hal_replay_executor_queue_dispatch(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE:
      return iree_hal_replay_executor_queue_execute(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FLUSH: {
      iree_hal_replay_object_entry_t* entry = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
          executor, record->header.object_id,
          IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE, &entry));
      return iree_hal_device_queue_flush(entry->value.device,
                                         IREE_HAL_QUEUE_AFFINITY_ANY);
    }
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED, "replay operation %s is not implemented",
          iree_hal_replay_operation_code_string(record->header.operation_code));
  }
}

iree_status_t iree_hal_replay_executor_replay_unsupported(
    const iree_hal_replay_file_record_t* record) {
  if (record->header.status_code != IREE_STATUS_OK) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "replay contains unsupported captured operation %s",
      iree_hal_replay_operation_code_string(record->header.operation_code));
}
