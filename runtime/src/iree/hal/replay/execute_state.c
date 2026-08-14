// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute_state.h"

#include <inttypes.h>
#include <string.h>

static void iree_hal_replay_executor_release_entry(
    iree_hal_replay_executor_t* executor,
    iree_hal_replay_object_entry_t* entry) {
  switch (entry->type) {
    case IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE:
      iree_hal_device_release(entry->value.device);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR:
      iree_hal_allocator_release(entry->value.allocator);
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
    default:
      break;
  }
  memset(entry, 0, sizeof(*entry));
}

void iree_hal_replay_executor_deinitialize(
    iree_hal_replay_executor_t* executor) {
  if (!executor->objects) return;
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    iree_hal_replay_executor_release_entry(executor, &executor->objects[i - 1]);
  }
  iree_allocator_free(executor->host_allocator, executor->objects);
  executor->objects = NULL;
  executor->object_capacity = 0;
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
    iree_hal_replay_executor_release_entry(executor, &entry);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay object id %" PRIu64
                            " exceeds object table capacity",
                            object_id);
  }
  iree_hal_replay_object_entry_t* existing = &executor->objects[object_id];
  if (existing->type != IREE_HAL_REPLAY_OBJECT_TYPE_NONE) {
    iree_hal_replay_executor_release_entry(executor, &entry);
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "replay object id %" PRIu64 " is already assigned",
                            object_id);
  }
  *existing = entry;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_allocate_function_map(
    iree_hal_replay_executor_t* executor, iree_host_size_t function_count,
    iree_hal_executable_function_t** out_function_map) {
  *out_function_map = NULL;
  if (function_count == 0) return iree_ok_status();
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
    iree_host_size_t object_capacity, iree_hal_device_group_t* device_group,
    const iree_hal_replay_execute_options_t* options,
    iree_allocator_t host_allocator) {
  memset(executor, 0, sizeof(*executor));
  executor->file_contents = file_contents;
  executor->device_group = device_group;
  executor->host_allocator = host_allocator;
  executor->options = options;
  executor->object_capacity = object_capacity;
  if (executor->object_capacity == 0) return iree_ok_status();
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

iree_status_t iree_hal_replay_executor_make_buffer_params(
    const iree_hal_replay_allocator_allocate_buffer_payload_t* payload,
    iree_hal_buffer_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->usage = payload->usage;
  out_params->type = payload->type;
  out_params->access = payload->access;
  out_params->queue_affinity = payload->queue_affinity;
  out_params->min_alignment = payload->min_alignment;
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
  iree_hal_buffer_mapping_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, memory_access, byte_offset,
      byte_length, &mapping));
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
    iree_host_size_t count,
    iree_hal_replay_semaphore_list_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  if (count == 0) return iree_ok_status();
  iree_host_size_t expected_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                        count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &expected_length) ||
                    payloads.data_length != expected_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay semaphore list payload length mismatch");
  }
  iree_status_t status = iree_ok_status();
  if (count <= IREE_HAL_REPLAY_INLINE_SEMAPHORE_LIST_CAPACITY) {
    out_storage->semaphores = out_storage->inline_storage.semaphores;
    out_storage->payload_values = out_storage->inline_storage.payload_values;
  } else {
    status = iree_allocator_malloc(executor->host_allocator,
                                   count * sizeof(*out_storage->semaphores),
                                   (void**)&out_storage->allocated.semaphores);
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator,
                                count * sizeof(*out_storage->payload_values),
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
  const iree_hal_replay_semaphore_timepoint_payload_t* timepoints =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)payloads.data;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_replay_object_entry_t* entry = NULL;
    status = iree_hal_replay_executor_lookup(
        executor, timepoints[i].semaphore_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE, &entry);
    if (!iree_status_is_ok(status)) break;
    out_storage->semaphores[i] = entry->value.semaphore;
    out_storage->payload_values[i] = timepoints[i].value;
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
      (iree_host_size_t)wait_semaphore_count, out_wait_storage));
  cursor += wait_size;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, signal_size),
      (iree_host_size_t)signal_semaphore_count, out_signal_storage);
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
  if (count == 0) return iree_ok_status();
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
  if (count == 0) return iree_ok_status();
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

iree_status_t iree_hal_replay_executor_flush_and_wait(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t signal_list) {
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_flush(device, queue_affinity));
  if (signal_list.count == 0) return iree_ok_status();
  return iree_hal_semaphore_list_wait(signal_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
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
