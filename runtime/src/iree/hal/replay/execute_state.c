// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute_state.h"

#include <inttypes.h>
#include <string.h>

static iree_status_t iree_hal_replay_executor_release_entry(
    iree_hal_replay_executor_t* executor,
    iree_hal_replay_object_entry_t* entry) {
  iree_status_t status = iree_ok_status();
  switch (entry->type) {
    case IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE:
      iree_hal_device_release(entry->value.device);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR:
      iree_hal_allocator_release(entry->value.allocator);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_POOL:
      iree_hal_pool_release(entry->value.pool);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER_MAPPING:
      status =
          iree_hal_buffer_unmap_range(&entry->value.buffer_mapping->mapping);
      iree_hal_buffer_release(entry->value.buffer_mapping->storage);
      iree_allocator_free(executor->host_allocator,
                          entry->value.buffer_mapping);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER:
      iree_hal_buffer_release(entry->value.buffer);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER:
      iree_hal_command_buffer_release(entry->value.command_buffer);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE:
      iree_hal_executable_release(entry->value.executable.handle);
      iree_allocator_free(executor->host_allocator,
                          entry->value.executable.function_map);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE:
      iree_hal_semaphore_release(entry->value.semaphore);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_FILE:
      iree_hal_file_release(entry->value.file);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE:
      iree_hal_queue_release(entry->value.queue);
      break;
    default:
      break;
  }
  memset(entry, 0, sizeof(*entry));
  return status;
}

iree_status_t iree_hal_replay_executor_deinitialize(
    iree_hal_replay_executor_t* executor) {
  iree_status_t status =
      iree_hal_replay_executor_drain_queue_completions(executor);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  iree_allocator_free(executor->host_allocator, executor->queue_completions);
  executor->queue_completions = NULL;
  executor->queue_completion_count = 0;
  executor->queue_completion_capacity = 0;

  // End host maps before detaching virtual buffers. Keep their actual storage
  // retained until all mappings are removed: a decommitted source may
  // have surviving aliases backed only by these host-map storage references.
  iree_status_t host_mapping_status = iree_ok_status();
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    if (executor->objects[i - 1].type ==
        IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER_MAPPING) {
      host_mapping_status = iree_status_join(
          host_mapping_status,
          iree_hal_buffer_unmap_range(
              &executor->objects[i - 1].value.buffer_mapping->mapping));
    }
  }
  IREE_RETURN_IF_ERROR(host_mapping_status);

  // Clear surviving virtual roots before releasing caller-owned backing.
  // Successful removal certifies reuse independently of final buffer release.
  iree_status_t detach_status = iree_ok_status();
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    const iree_hal_replay_object_entry_t* entry = &executor->objects[i - 1];
    if (entry->type != IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER) {
      continue;
    }
    iree_hal_buffer_t* buffer = entry->value.buffer;
    const iree_hal_virtual_buffer_view_t view =
        iree_hal_buffer_virtual_view(buffer);
    if (view.root && !view.byte_offset &&
        buffer->byte_length == view.root->buffer.byte_length) {
      const iree_hal_memory_scope_t scope = {
          .domain = buffer->memory.contract->domain,
          .id = IREE_HAL_MEMORY_SCOPE_ANY,
      };
      detach_status = iree_status_join(
          detach_status, iree_hal_virtual_buffer_detach(buffer, scope));
    }
  }
  IREE_RETURN_IF_ERROR(detach_status);

  // Release leaf objects before queue, pool, allocator, and device roots
  // regardless of replay object id assignment.
  iree_status_t object_status = iree_ok_status();
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    const iree_hal_replay_object_type_t type = executor->objects[i - 1].type;
    if (type == IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE ||
        type == IREE_HAL_REPLAY_OBJECT_TYPE_POOL ||
        type == IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR ||
        type == IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE) {
      continue;
    }
    object_status = iree_status_join(object_status,
                                     iree_hal_replay_executor_release_entry(
                                         executor, &executor->objects[i - 1]));
  }
  if (!iree_status_is_ok(object_status)) {
    return object_status;
  }
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    if (executor->objects[i - 1].type != IREE_HAL_REPLAY_OBJECT_TYPE_POOL) {
      continue;
    }
    object_status = iree_status_join(object_status,
                                     iree_hal_replay_executor_release_entry(
                                         executor, &executor->objects[i - 1]));
  }
  if (!iree_status_is_ok(object_status)) {
    return object_status;
  }
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    if (executor->objects[i - 1].type != IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE) {
      continue;
    }
    object_status = iree_status_join(object_status,
                                     iree_hal_replay_executor_release_entry(
                                         executor, &executor->objects[i - 1]));
  }
  if (!iree_status_is_ok(object_status)) {
    return object_status;
  }
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    if (executor->objects[i - 1].type !=
        IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR) {
      continue;
    }
    object_status = iree_status_join(object_status,
                                     iree_hal_replay_executor_release_entry(
                                         executor, &executor->objects[i - 1]));
  }
  if (!iree_status_is_ok(object_status)) {
    return object_status;
  }
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    if (executor->objects[i - 1].type != IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE) {
      continue;
    }
    object_status = iree_status_join(object_status,
                                     iree_hal_replay_executor_release_entry(
                                         executor, &executor->objects[i - 1]));
  }
  if (!iree_status_is_ok(object_status)) {
    return object_status;
  }
  iree_allocator_free(executor->host_allocator, executor->objects);
  executor->objects = NULL;
  executor->object_capacity = 0;
  iree_hal_device_group_release(executor->device_group);
  executor->device_group = NULL;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_lookup(
    iree_hal_replay_executor_t* executor, iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_type_t expected_type,
    iree_hal_replay_object_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  if (IREE_UNLIKELY(object_id == IREE_HAL_REPLAY_OBJECT_ID_NONE ||
                    object_id >= executor->object_capacity)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "replay object id %" PRIu64 " is not defined",
                            object_id);
  }
  iree_hal_replay_object_entry_t* entry = &executor->objects[object_id];
  if (IREE_UNLIKELY(entry->type != expected_type)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay object id %" PRIu64 " has type %s; expected %s", object_id,
        iree_hal_replay_object_type_string(entry->type),
        iree_hal_replay_object_type_string(expected_type));
  }
  *out_entry = entry;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_store(
    iree_hal_replay_executor_t* executor, iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_object_entry_t entry) {
  entry.type = object_type;
  if (IREE_UNLIKELY(object_id == IREE_HAL_REPLAY_OBJECT_ID_NONE ||
                    object_id >= executor->object_capacity)) {
    iree_status_t status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                            "replay object id %" PRIu64
                                            " exceeds object table capacity",
                                            object_id);
    return iree_status_join(
        status, iree_hal_replay_executor_release_entry(executor, &entry));
  }
  iree_hal_replay_object_entry_t* existing = &executor->objects[object_id];
  if (existing->type != IREE_HAL_REPLAY_OBJECT_TYPE_NONE) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "replay object id %" PRIu64 " is already assigned", object_id);
    return iree_status_join(
        status, iree_hal_replay_executor_release_entry(executor, &entry));
  }
  *existing = entry;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_allocate_function_map(
    iree_hal_replay_executor_t* executor, iree_host_size_t function_count,
    iree_hal_executable_function_t** out_function_map) {
  *out_function_map = NULL;
  if (function_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t function_map_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          function_count, sizeof(iree_hal_executable_function_t),
          &function_map_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay executable function map size overflow");
  }
  return iree_allocator_malloc(executor->host_allocator, function_map_size,
                               (void**)out_function_map);
}

iree_status_t iree_hal_replay_executor_resolve_function(
    iree_hal_replay_object_id_t executable_id,
    const iree_hal_replay_executable_entry_t* executable,
    uint32_t function_ordinal, iree_hal_executable_function_t* out_function) {
  if (IREE_UNLIKELY(function_ordinal >= executable->function_map_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "replay executable %" PRIu64
        " does not define captured function ordinal %" PRIu32,
        executable_id, function_ordinal);
  }
  *out_function = executable->function_map[function_ordinal];
  if (IREE_UNLIKELY(!iree_hal_executable_function_is_valid(*out_function))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "replay executable %" PRIu64
                            " captured function ordinal %" PRIu32
                            " did not resolve",
                            executable_id, function_ordinal);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_initialize(
    iree_hal_replay_executor_t* executor, iree_const_byte_span_t file_contents,
    uint16_t file_version_minor, iree_host_size_t object_capacity,
    iree_hal_device_group_t* device_group,
    const iree_hal_replay_execute_options_t* options,
    iree_allocator_t host_allocator) {
  memset(executor, 0, sizeof(*executor));
  executor->file_contents = file_contents;
  executor->file_version_minor = file_version_minor;
  executor->device_group = device_group;
  executor->host_allocator = host_allocator;
  executor->options = options;
  executor->object_capacity = object_capacity;
  if (executor->object_capacity == 0) {
    executor->device_group = device_group;
    iree_hal_device_group_retain(executor->device_group);
    return iree_ok_status();
  }
  iree_host_size_t object_table_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(executor->object_capacity,
                                                sizeof(*executor->objects),
                                                &object_table_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay object table size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, object_table_size,
                                             (void**)&executor->objects));
  memset(executor->objects, 0, object_table_size);
  executor->device_group = device_group;
  iree_hal_device_group_retain(executor->device_group);
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_require_payload(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_payload_type_t payload_type,
    iree_host_size_t minimum_length) {
  if (IREE_UNLIKELY(record->header.payload_type != payload_type)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED, "replay operation %s requires payload %s",
        iree_hal_replay_operation_code_string(record->header.operation_code),
        iree_hal_replay_payload_type_string(payload_type));
  }
  if (IREE_UNLIKELY(record->payload.data_length < minimum_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay payload %s is too short",
                            iree_hal_replay_payload_type_string(payload_type));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_write_buffer_data(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_memory_access_t memory_access,
    iree_const_byte_span_t data) {
  if (data.data_length == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(data.data_length > byte_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay buffer data overflows target range");
  }
  // Captured flushes can cover only part of a write mapping. Reproduce those
  // bytes without repeating the original map's discard operation.
  iree_hal_buffer_mapping_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, memory_access,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, byte_offset, byte_length, &mapping));
  iree_status_t status = iree_ok_status();
  iree_byte_span_t target_span;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_mapping_subspan(&mapping, IREE_HAL_MEMORY_ACCESS_WRITE,
                                        0, data.data_length, &target_span);
  }
  if (iree_status_is_ok(status)) {
    memcpy(target_span.data, data.data, data.data_length);
    status = iree_hal_buffer_mapping_flush_range(&mapping, 0, data.data_length);
  }
  iree_status_t unmap_status = iree_hal_buffer_unmap_range(&mapping);
  return iree_status_join(status, unmap_status);
}

iree_status_t iree_hal_replay_executor_make_buffer_ref(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_buffer_ref_payload_t* payload,
    iree_hal_buffer_ref_t* out_ref) {
  if (payload->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    *out_ref = iree_hal_make_indirect_buffer_ref(
        payload->buffer_slot, payload->offset, payload->length);
    return iree_ok_status();
  }
  iree_hal_replay_object_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, payload->buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      &entry));
  *out_ref = iree_hal_make_buffer_ref(entry->value.buffer, payload->offset,
                                      payload->length);
  return iree_ok_status();
}

void iree_hal_replay_semaphore_list_storage_deinitialize(
    iree_hal_replay_semaphore_list_storage_t* storage,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, storage->allocated.payload_values);
  iree_allocator_free(host_allocator, storage->allocated.semaphores);
  memset(storage, 0, sizeof(*storage));
}

iree_status_t iree_hal_replay_executor_make_semaphore_list(
    iree_hal_replay_executor_t* executor, iree_const_byte_span_t payloads,
    iree_host_size_t count, iree_host_size_t additional_capacity,
    iree_hal_replay_semaphore_list_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  iree_host_size_t expected_length = 0;
  iree_host_size_t capacity = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                        count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &expected_length) ||
                    payloads.data_length != expected_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay semaphore list payload length mismatch");
  }
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(count, additional_capacity, &capacity))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay semaphore list capacity overflow");
  }
  if (capacity == 0) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (capacity <= IREE_HAL_REPLAY_INLINE_SEMAPHORE_LIST_CAPACITY) {
    out_storage->semaphores = out_storage->inline_storage.semaphores;
    out_storage->payload_values = out_storage->inline_storage.payload_values;
  } else {
    iree_host_size_t semaphores_size = 0;
    iree_host_size_t payload_values_size = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(
                capacity, sizeof(*out_storage->semaphores), &semaphores_size) ||
            !iree_host_size_checked_mul(capacity,
                                        sizeof(*out_storage->payload_values),
                                        &payload_values_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "replay semaphore list allocation overflow");
    }
    status = iree_allocator_malloc(executor->host_allocator, semaphores_size,
                                   (void**)&out_storage->allocated.semaphores);
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator, payload_values_size,
                                (void**)&out_storage->allocated.payload_values);
    }
    if (iree_status_is_ok(status)) {
      out_storage->semaphores = out_storage->allocated.semaphores;
      out_storage->payload_values = out_storage->allocated.payload_values;
    } else {
      iree_hal_replay_semaphore_list_storage_deinitialize(
          out_storage, executor->host_allocator);
      return status;
    }
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_replay_semaphore_timepoint_payload_t timepoint;
    memcpy(&timepoint,
           payloads.data +
               i * sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
           sizeof(timepoint));
    iree_hal_replay_object_entry_t* entry = NULL;
    status = iree_hal_replay_executor_lookup(
        executor, timepoint.semaphore_id, IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE,
        &entry);
    if (!iree_status_is_ok(status)) {
      break;
    }
    out_storage->semaphores[i] = entry->value.semaphore;
    out_storage->payload_values[i] = timepoint.value;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_semaphore_list_storage_deinitialize(
        out_storage, executor->host_allocator);
    return status;
  }
  out_storage->list.count = count;
  out_storage->list.semaphores = out_storage->semaphores;
  out_storage->list.payload_values = out_storage->payload_values;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_make_queue_semaphore_lists(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record, iree_host_size_t header_length,
    uint64_t wait_semaphore_count, uint64_t signal_semaphore_count,
    uint64_t trailing_payload_length,
    iree_hal_replay_semaphore_list_storage_t* out_wait_storage,
    iree_hal_replay_semaphore_list_storage_t* out_signal_storage,
    iree_const_byte_span_t* out_trailing_payload) {
  memset(out_wait_storage, 0, sizeof(*out_wait_storage));
  memset(out_signal_storage, 0, sizeof(*out_signal_storage));
  *out_trailing_payload = iree_make_const_byte_span(NULL, 0);
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(
          wait_semaphore_count > IREE_HOST_SIZE_MAX ||
          signal_semaphore_count > IREE_HOST_SIZE_MAX ||
          trailing_payload_length > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)wait_semaphore_count,
              sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
              &wait_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)signal_semaphore_count,
              sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
              &signal_size) ||
          !iree_host_size_checked_add(header_length, wait_size, &total_size) ||
          !iree_host_size_checked_add(total_size, signal_size, &total_size) ||
          !iree_host_size_checked_add(total_size,
                                      (iree_host_size_t)trailing_payload_length,
                                      &total_size) ||
          total_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue payload length mismatch");
  }

  const uint8_t* cursor = record->payload.data + header_length;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, wait_size),
      (iree_host_size_t)wait_semaphore_count, /*additional_capacity=*/0,
      out_wait_storage));
  cursor += wait_size;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, signal_size),
      (iree_host_size_t)signal_semaphore_count, /*additional_capacity=*/1,
      out_signal_storage);
  cursor += signal_size;
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_semaphore_list_storage_deinitialize(
        out_wait_storage, executor->host_allocator);
    return status;
  }
  *out_trailing_payload = iree_make_const_byte_span(
      cursor, (iree_host_size_t)trailing_payload_length);
  return iree_ok_status();
}

void iree_hal_replay_buffer_ref_list_storage_deinitialize(
    iree_hal_replay_buffer_ref_list_storage_t* storage,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, storage->allocated.values);
  memset(storage, 0, sizeof(*storage));
}

iree_status_t iree_hal_replay_buffer_ref_list_storage_initialize(
    iree_hal_replay_executor_t* executor, iree_host_size_t count,
    iree_hal_replay_buffer_ref_list_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  if (count == 0) {
    return iree_ok_status();
  }
  if (count <= IREE_HAL_REPLAY_INLINE_BUFFER_REF_LIST_CAPACITY) {
    out_storage->values = out_storage->inline_storage.values;
  } else {
    iree_host_size_t values_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            count, sizeof(*out_storage->values), &values_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "replay buffer ref list count overflow");
    }
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(executor->host_allocator, values_size,
                              (void**)&out_storage->allocated.values));
    out_storage->values = out_storage->allocated.values;
  }
  out_storage->list.count = count;
  out_storage->list.values = out_storage->values;
  return iree_ok_status();
}

void iree_hal_replay_buffer_binding_table_storage_deinitialize(
    iree_hal_replay_buffer_binding_table_storage_t* storage,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, storage->allocated.bindings);
  memset(storage, 0, sizeof(*storage));
}

iree_status_t iree_hal_replay_buffer_binding_table_storage_initialize(
    iree_hal_replay_executor_t* executor, iree_host_size_t count,
    iree_hal_replay_buffer_binding_table_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  if (count == 0) {
    return iree_ok_status();
  }
  if (count <= IREE_HAL_REPLAY_INLINE_BUFFER_BINDING_TABLE_CAPACITY) {
    out_storage->bindings = out_storage->inline_storage.bindings;
  } else {
    iree_host_size_t bindings_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            count, sizeof(*out_storage->bindings), &bindings_size))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "replay buffer binding table count overflow");
    }
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(executor->host_allocator, bindings_size,
                              (void**)&out_storage->allocated.bindings));
    out_storage->bindings = out_storage->allocated.bindings;
  }
  out_storage->table.count = count;
  out_storage->table.bindings = out_storage->bindings;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_reserve_queue_completion_capacity(
    iree_hal_replay_executor_t* executor, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= executor->queue_completion_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = executor->queue_completion_capacity
                                      ? executor->queue_completion_capacity
                                      : 4;
  while (new_capacity < minimum_capacity) {
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(new_capacity, 2, &new_capacity))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "replay queue completion capacity overflow");
    }
  }
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          new_capacity, sizeof(*executor->queue_completions),
          &allocation_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue completion table size overflow");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc(executor->host_allocator, allocation_size,
                             (void**)&executor->queue_completions));
  executor->queue_completion_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_prepare_queue_completion(
    iree_hal_replay_executor_t* executor, iree_hal_device_t* device,
    iree_hal_queue_t* queue,
    iree_hal_replay_semaphore_list_storage_t* signal_storage,
    iree_hal_replay_queue_completion_t** out_completion) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(signal_storage);
  IREE_ASSERT_ARGUMENT(out_completion);
  *out_completion = NULL;
  iree_host_size_t required_capacity = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          executor->queue_completion_count, 1, &required_capacity))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue completion count overflow");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_executor_reserve_queue_completion_capacity(
          executor, required_capacity));
  iree_hal_replay_queue_completion_t* completion =
      &executor->queue_completions[executor->queue_completion_count];
  memset(completion, 0, sizeof(*completion));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device,
      iree_hal_make_queue_family_affinity(
          iree_hal_queue_family_ordinal(iree_hal_queue_family(queue))),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      &completion->semaphore));
  completion->queue = queue;
  completion->value = 1;
  completion->wait_immediately = signal_storage->list.count != 0;
  const iree_host_size_t signal_index = signal_storage->list.count++;
  signal_storage->semaphores[signal_index] = completion->semaphore;
  signal_storage->payload_values[signal_index] = completion->value;
  *out_completion = completion;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_finalize_queue_completion(
    iree_hal_replay_executor_t* executor,
    iree_hal_replay_queue_completion_t* completion, bool flush_queue,
    iree_status_t operation_status) {
  if (!completion) {
    return operation_status;
  }
  if (!iree_status_is_ok(operation_status)) {
    iree_allocator_free(executor->host_allocator,
                        completion->retained_host_allocation);
    iree_hal_semaphore_release(completion->semaphore);
    memset(completion, 0, sizeof(*completion));
    return operation_status;
  }
  IREE_ASSERT(completion ==
              &executor->queue_completions[executor->queue_completion_count]);
  ++executor->queue_completion_count;
  iree_status_t status =
      flush_queue ? iree_hal_queue_flush(completion->queue) : iree_ok_status();
  if (!iree_status_is_ok(status) || !completion->wait_immediately) {
    return status;
  }
  status = iree_hal_semaphore_wait(completion->semaphore, completion->value,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_allocator_free(executor->host_allocator,
                      completion->retained_host_allocation);
  iree_hal_semaphore_release(completion->semaphore);
  memset(completion, 0, sizeof(*completion));
  --executor->queue_completion_count;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_drain_queue_completions(
    iree_hal_replay_executor_t* executor) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < executor->queue_completion_count; ++i) {
    status = iree_status_join(
        status, iree_hal_queue_flush(executor->queue_completions[i].queue));
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  for (iree_host_size_t i = 0; i < executor->queue_completion_count; ++i) {
    iree_hal_replay_queue_completion_t* completion =
        &executor->queue_completions[i];
    status = iree_status_join(
        status, iree_hal_semaphore_wait(
                    completion->semaphore, completion->value,
                    iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  for (iree_host_size_t i = 0; i < executor->queue_completion_count; ++i) {
    iree_hal_replay_queue_completion_t* completion =
        &executor->queue_completions[i];
    iree_allocator_free(executor->host_allocator,
                        completion->retained_host_allocation);
    iree_hal_semaphore_release(completion->semaphore);
    memset(completion, 0, sizeof(*completion));
  }
  executor->queue_completion_count = 0;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_dispatch_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_dispatch_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_constants_offset,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size) {
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t constants_size = 0;
  iree_host_size_t binding_size = 0;
  if (IREE_UNLIKELY(payload->wait_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload->signal_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload->constants_length > IREE_HOST_SIZE_MAX ||
                    payload->binding_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->wait_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &wait_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->signal_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &signal_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->binding_count,
                        sizeof(iree_hal_replay_buffer_ref_payload_t),
                        &binding_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload count overflow");
  }
  constants_size = (iree_host_size_t)payload->constants_length;
  iree_host_size_t offset = sizeof(*payload);
  *out_wait_payloads_offset = offset;
  *out_wait_payloads_size = wait_size;
  if (!iree_host_size_checked_add(offset, wait_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_signal_payloads_offset = offset;
  *out_signal_payloads_size = signal_size;
  if (!iree_host_size_checked_add(offset, signal_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_constants_offset = offset;
  if (!iree_host_size_checked_add(offset, constants_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_binding_payloads_offset = offset;
  *out_binding_payloads_size = binding_size;
  if (!iree_host_size_checked_add(offset, binding_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay dispatch payload length mismatch");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Captured queue barriers
//===----------------------------------------------------------------------===//

static iree_hal_memory_transition_recipe_info_t
iree_hal_replay_executor_memory_transition_operation(
    const iree_hal_replay_memory_transition_operation_payload_t* payload) {
  return (iree_hal_memory_transition_recipe_info_t){
      .kind = payload->kind,
      .executor = payload->executor,
      .operation = payload->operation,
      .range_granularity = payload->range_granularity,
      .host =
          {
              .instruction = payload->host_instruction,
              .fence_before = payload->host_fence_before,
              .fence_after = payload->host_fence_after,
          },
  };
}

iree_status_t iree_hal_replay_executor_make_queue_barriers(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_queue_barrier_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  const iree_hal_replay_barrier_list_view_t* lists[2] = {
      &record->barriers.before, &record->barriers.after};
  const bool has_transition_recipes = iree_any_bit_set(
      record->header.record_flags,
      IREE_HAL_REPLAY_FILE_RECORD_FLAG_MEMORY_TRANSITION_RECIPES);
  iree_host_size_t allocation_size = 0;
  for (iree_host_size_t boundary = 0; boundary < 2; ++boundary) {
    if (lists[boundary]->count == UINT64_MAX) {
      continue;
    }
    const uint8_t* wire = lists[boundary]->payload.data;
    for (uint64_t i = 0; i < lists[boundary]->count; ++i) {
      iree_hal_replay_command_buffer_execution_barrier_payload_t header;
      memcpy(&header, wire, sizeof(header));
      wire += sizeof(header) +
              header.memory_barrier_count *
                  sizeof(iree_hal_replay_memory_barrier_payload_t) +
              header.buffer_barrier_count *
                  sizeof(iree_hal_replay_buffer_barrier_payload_t);
      iree_host_size_t transition_operation_count = 0;
      if (has_transition_recipes) {
        const uint8_t* recipe_wire = wire;
        wire += header.buffer_barrier_count *
                sizeof(iree_hal_replay_memory_transition_recipe_payload_t);
        for (iree_host_size_t j = 0; j < header.buffer_barrier_count; ++j) {
          iree_hal_replay_memory_transition_recipe_payload_t recipe;
          memcpy(&recipe, recipe_wire + j * sizeof(recipe), sizeof(recipe));
          transition_operation_count += recipe.operation_count;
        }
        wire += transition_operation_count *
                sizeof(iree_hal_replay_memory_transition_operation_payload_t);
      }
      iree_host_size_t memory_size = 0, buffer_size = 0, recipe_size = 0;
      iree_host_size_t transition_operation_size = 0;
      if (!iree_host_size_checked_mul(header.memory_barrier_count,
                                      sizeof(iree_hal_memory_barrier_t),
                                      &memory_size) ||
          !iree_host_size_checked_mul(header.buffer_barrier_count,
                                      sizeof(iree_hal_buffer_barrier_t),
                                      &buffer_size) ||
          !iree_host_size_checked_mul(
              has_transition_recipes ? header.buffer_barrier_count : 0,
              sizeof(iree_hal_memory_transition_recipe_t), &recipe_size) ||
          !iree_host_size_checked_mul(
              transition_operation_count,
              sizeof(iree_hal_memory_transition_recipe_info_t),
              &transition_operation_size) ||
          !iree_host_size_checked_add(
              allocation_size, sizeof(iree_hal_barrier_t), &allocation_size) ||
          !iree_host_size_checked_add(allocation_size, memory_size,
                                      &allocation_size) ||
          !iree_host_size_checked_add(allocation_size, buffer_size,
                                      &allocation_size) ||
          !iree_host_size_checked_add(allocation_size, recipe_size,
                                      &allocation_size) ||
          !iree_host_size_checked_add(
              allocation_size, transition_operation_size, &allocation_size)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "replay queue barrier allocation overflow");
      }
    }
  }
  if (allocation_size) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        executor->host_allocator, allocation_size, &out_storage->allocation));
  }
  uint8_t* native = out_storage->allocation;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t boundary = 0; boundary < 2 && iree_status_is_ok(status);
       ++boundary) {
    const iree_hal_replay_barrier_list_view_t* list = lists[boundary];
    if (list->count == UINT64_MAX) {
      continue;
    }
    iree_hal_barrier_t* barriers = (iree_hal_barrier_t*)native;
    if (list->count) {
      native += list->count * sizeof(*barriers);
    }
    out_storage->lists[boundary] =
        (iree_hal_barrier_list_t){.count = list->count, .values = barriers};
    const uint8_t* wire = list->payload.data;
    for (iree_host_size_t i = 0; i < list->count && iree_status_is_ok(status);
         ++i) {
      iree_hal_replay_command_buffer_execution_barrier_payload_t header;
      memcpy(&header, wire, sizeof(header));
      wire += sizeof(header);
      iree_hal_memory_barrier_t* memory = (iree_hal_memory_barrier_t*)native;
      native += header.memory_barrier_count * sizeof(*memory);
      for (iree_host_size_t j = 0; j < header.memory_barrier_count; ++j) {
        iree_hal_replay_memory_barrier_payload_t payload;
        memcpy(&payload, wire, sizeof(payload));
        wire += sizeof(payload);
        memory[j] = (iree_hal_memory_barrier_t){
            .source_scope = payload.source_scope,
            .target_scope = payload.target_scope,
        };
      }
      iree_hal_buffer_barrier_t* buffers = (iree_hal_buffer_barrier_t*)native;
      native += header.buffer_barrier_count * sizeof(*buffers);
      memset(buffers, 0, header.buffer_barrier_count * sizeof(*buffers));
      const uint8_t* buffer_wire = wire;
      for (iree_host_size_t j = 0;
           j < header.buffer_barrier_count && iree_status_is_ok(status); ++j) {
        iree_hal_replay_buffer_barrier_payload_t payload;
        memcpy(&payload, buffer_wire + j * sizeof(payload), sizeof(payload));
        buffers[j].source_scope = payload.source_scope;
        buffers[j].target_scope = payload.target_scope;
        status = iree_hal_replay_executor_make_buffer_ref(
            executor, &payload.buffer_ref, &buffers[j].buffer_ref);
      }
      wire += header.buffer_barrier_count *
              sizeof(iree_hal_replay_buffer_barrier_payload_t);
      uint32_t transition_effects = 0;
      if (has_transition_recipes) {
        iree_hal_memory_transition_recipe_t* recipes =
            (iree_hal_memory_transition_recipe_t*)native;
        native += header.buffer_barrier_count * sizeof(*recipes);
        memset(recipes, 0, header.buffer_barrier_count * sizeof(*recipes));
        const uint8_t* recipe_wire = wire;
        wire += header.buffer_barrier_count *
                sizeof(iree_hal_replay_memory_transition_recipe_payload_t);
        iree_host_size_t transition_operation_count = 0;
        for (iree_host_size_t j = 0; j < header.buffer_barrier_count; ++j) {
          iree_hal_replay_memory_transition_recipe_payload_t recipe;
          memcpy(&recipe, recipe_wire + j * sizeof(recipe), sizeof(recipe));
          transition_operation_count += recipe.operation_count;
        }
        iree_hal_memory_transition_recipe_info_t* operations =
            (iree_hal_memory_transition_recipe_info_t*)native;
        native += transition_operation_count * sizeof(*operations);
        iree_host_size_t operation_ordinal = 0;
        for (iree_host_size_t j = 0; j < header.buffer_barrier_count; ++j) {
          iree_hal_replay_memory_transition_recipe_payload_t recipe;
          memcpy(&recipe, recipe_wire + j * sizeof(recipe), sizeof(recipe));
          transition_effects |= recipe.effects;
          if (!recipe.operation_count) {
            continue;
          }
          recipes[j] = (iree_hal_memory_transition_recipe_t){
              .effects = {recipe.effects},
              .operation_count = recipe.operation_count,
              .operations = &operations[operation_ordinal],
          };
          buffers[j].recipe = &recipes[j];
          for (uint32_t k = 0; k < recipe.operation_count; ++k) {
            iree_hal_replay_memory_transition_operation_payload_t operation;
            memcpy(&operation, wire + operation_ordinal * sizeof(operation),
                   sizeof(operation));
            operations[operation_ordinal++] =
                iree_hal_replay_executor_memory_transition_operation(
                    &operation);
          }
        }
        wire += transition_operation_count *
                sizeof(iree_hal_replay_memory_transition_operation_payload_t);
      }
      barriers[i] = (iree_hal_barrier_t){
          .source_stage_mask = header.source_stage_mask,
          .target_stage_mask = header.target_stage_mask,
          .flags = header.flags,
          .effects = {transition_effects},
          .memory_barrier_count = header.memory_barrier_count,
          .memory_barriers = memory,
          .buffer_barrier_count = header.buffer_barrier_count,
          .buffer_barriers = buffers,
      };
    }
  }
  if (iree_status_is_ok(status)) {
    out_storage->barriers.before =
        lists[0]->count == UINT64_MAX ? NULL : &out_storage->lists[0];
    out_storage->barriers.after =
        lists[1]->count == UINT64_MAX ? NULL : &out_storage->lists[1];
  } else {
    iree_allocator_free(executor->host_allocator, out_storage->allocation);
    memset(out_storage, 0, sizeof(*out_storage));
  }
  return status;
}
