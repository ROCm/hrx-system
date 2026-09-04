// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute_operation.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/replay/execute_object.h"

static iree_status_t iree_hal_replay_executor_allocate_array(
    iree_hal_replay_executor_t* executor, iree_host_size_t element_count,
    iree_host_size_t element_size, void** out_ptr) {
  if (element_count == 0) return iree_ok_status();
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(element_count, element_size,
                                                &allocation_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue storage size overflow");
  }
  return iree_allocator_malloc(executor->host_allocator, allocation_size,
                               out_ptr);
}

static iree_status_t iree_hal_replay_executor_require_queue_allocation_pool(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_object_entry_t* queue_entry, iree_hal_pool_t** out_pool) {
  if (queue_entry->queue_allocation_pool) {
    *out_pool = queue_entry->queue_allocation_pool;
    return iree_ok_status();
  }

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.device_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  const iree_hal_queue_family_ordinal_t family_ordinal =
      iree_hal_queue_family_ordinal(
          iree_hal_queue_family(queue_entry->value.queue));
  if (IREE_UNLIKELY(family_ordinal >= IREE_HAL_MAX_QUEUE_FAMILIES)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue family ordinal %u is not maskable",
                            family_ordinal);
  }
  iree_hal_queue_pool_backend_t pool_backend;
  IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
      device_entry->value.device,
      iree_hal_make_queue_family_affinity(family_ordinal), &pool_backend));
  const iree_hal_passthrough_pool_options_t options = {
      .asan = pool_backend.asan,
      .trace_name = iree_make_cstring_view("iree_hal_replay_queue"),
  };
  iree_hal_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_passthrough_pool_create(
      options, pool_backend.slab_provider, pool_backend.notification,
      executor->host_allocator, &pool));
  queue_entry->queue_allocation_pool = pool;
  *out_pool = pool;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_find_provisioned_queue(
    iree_hal_replay_executor_t* executor, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_replay_object_entry_t** out_queue_entry) {
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queue_spec->family_count; ++i) {
    const iree_hal_queue_family_affinity_t family_affinity =
        iree_hal_make_queue_family_affinity((iree_hal_queue_family_ordinal_t)i);
    if (!iree_hal_queue_affinity_is_any(queue_affinity) &&
        !iree_any_bit_set(queue_affinity, family_affinity)) {
      continue;
    }
    iree_hal_queue_t* queue = iree_hal_device_queue(
        device, (iree_hal_queue_family_ordinal_t)i, /*queue_ordinal=*/0);
    if (!queue) continue;
    for (iree_host_size_t j = 0; j < executor->object_capacity; ++j) {
      iree_hal_replay_object_entry_t* entry = &executor->objects[j];
      if (entry->type == IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE &&
          entry->value.queue == queue) {
        *out_queue_entry = entry;
        return iree_ok_status();
      }
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "replay device has no captured provisioned queue "
                          "matching affinity 0x%016" PRIx64,
                          queue_affinity);
}

static iree_status_t iree_hal_replay_executor_require_fixed_payload(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_payload_type_t payload_type,
    iree_host_size_t payload_length) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, payload_type, payload_length));
  if (IREE_UNLIKELY(record->payload.data_length != payload_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay fixed payload length mismatch");
  }
  return iree_ok_status();
}

static iree_hal_atomic_wait_params_t
iree_hal_replay_executor_atomic_wait_params(
    iree_hal_replay_atomic_wait_params_payload_t payload) {
  return (iree_hal_atomic_wait_params_t){
      .value = payload.value,
      .mask = payload.mask,
      .flags = payload.flags,
      .width = payload.width,
      .condition = payload.condition,
      .reserved = payload.reserved0,
  };
}

static iree_hal_atomic_store_params_t
iree_hal_replay_executor_atomic_store_params(
    iree_hal_replay_atomic_store_params_payload_t payload) {
  iree_hal_atomic_store_params_t params = {
      .value = payload.value,
      .flags = payload.flags,
      .width = payload.width,
  };
  memcpy(params.reserved, payload.reserved0, sizeof(params.reserved));
  return params;
}

static iree_hal_atomic_rmw_params_t iree_hal_replay_executor_atomic_rmw_params(
    iree_hal_replay_atomic_rmw_params_payload_t payload) {
  return (iree_hal_atomic_rmw_params_t){
      .operand = payload.operand,
      .flags = payload.flags,
      .width = payload.width,
      .operation = payload.operation,
      .reserved = payload.reserved0,
  };
}

static iree_status_t iree_hal_replay_executor_make_atomic_buffer_ref(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_buffer_ref_payload_t* payload,
    iree_hal_atomic_width_t width, iree_hal_buffer_ref_t* out_ref) {
  if (IREE_UNLIKELY(payload->reserved0 != 0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay atomic buffer reference reserved fields must be zero");
  }
  if (IREE_UNLIKELY(payload->buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE &&
                    payload->buffer_slot != 0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay direct atomic buffer reference has an indirect slot");
  }
  if (IREE_UNLIKELY(payload->length !=
                    iree_hal_atomic_width_byte_count(width))) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay atomic target length does not match the selected width");
  }
  return iree_hal_replay_executor_make_buffer_ref(executor, payload, out_ref);
}

static iree_status_t iree_hal_replay_executor_make_direct_atomic_buffer_ref(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_buffer_ref_payload_t* payload,
    iree_hal_atomic_width_t width, iree_hal_buffer_ref_t* out_ref) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_atomic_buffer_ref(
      executor, payload, width, out_ref));
  if (IREE_UNLIKELY(!out_ref->buffer)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue atomic operation requires a direct target reference");
  }
  return iree_ok_status();
}

static bool iree_hal_replay_buffer_ref_payload_is_empty(
    const iree_hal_replay_buffer_ref_payload_t* payload) {
  return payload->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE &&
         payload->offset == 0 && payload->length == 0 &&
         payload->buffer_slot == 0 && payload->reserved0 == 0;
}

static iree_status_t iree_hal_replay_executor_make_direct_transfer_buffer_ref(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_buffer_ref_payload_t* payload,
    iree_hal_buffer_ref_t* out_ref) {
  if (IREE_UNLIKELY(payload->reserved0 != 0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay transfer buffer reference reserved fields must be zero");
  }
  if (IREE_UNLIKELY(payload->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE ||
                    payload->buffer_slot != 0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue transfer requires a direct buffer reference");
  }
  return iree_hal_replay_executor_make_buffer_ref(executor, payload, out_ref);
}

static iree_status_t iree_hal_replay_executor_transfer_data_span(
    iree_const_byte_span_t data,
    const iree_hal_replay_queue_transfer_operation_payload_t* payload,
    iree_const_byte_span_t* out_span) {
  *out_span = iree_make_const_byte_span(NULL, 0);
  if (IREE_UNLIKELY(payload->data_offset > data.data_length ||
                    payload->data_length >
                        data.data_length - payload->data_offset)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue transfer data range is out of bounds");
  }
  *out_span = iree_make_const_byte_span(
      data.data + (iree_host_size_t)payload->data_offset,
      (iree_host_size_t)payload->data_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_validate_empty_transfer_operation(
    const iree_hal_replay_queue_transfer_operation_payload_t* payload) {
  if (IREE_UNLIKELY(
          payload->flags != 0 ||
          !iree_hal_replay_buffer_ref_payload_is_empty(&payload->source_ref) ||
          !iree_hal_replay_buffer_ref_payload_is_empty(&payload->target_ref) ||
          payload->data_offset != 0 || payload->data_length != 0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay zero-length transfer operation has live payload fields");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_validate_transfer_operation(
    const iree_hal_replay_queue_transfer_operation_payload_t* payload,
    iree_const_byte_span_t data, iree_host_size_t* inout_download_data_length) {
  if (IREE_UNLIKELY(payload->reserved0 != 0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue transfer operation reserved fields must be zero");
  }

  switch (payload->type) {
    case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_FILL: {
      if (payload->target_ref.length == 0) {
        return iree_hal_replay_executor_validate_empty_transfer_operation(
            payload);
      }
      if (IREE_UNLIKELY(!iree_hal_replay_buffer_ref_payload_is_empty(
                            &payload->source_ref) ||
                        (payload->data_length != 1 &&
                         payload->data_length != 2 &&
                         payload->data_length != 4))) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay fill operation payload is invalid");
      }
      iree_const_byte_span_t fill_data;
      return iree_hal_replay_executor_transfer_data_span(data, payload,
                                                         &fill_data);
    }
    case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPDATE: {
      if (payload->target_ref.length == 0) {
        return iree_hal_replay_executor_validate_empty_transfer_operation(
            payload);
      }
      if (IREE_UNLIKELY(!iree_hal_replay_buffer_ref_payload_is_empty(
                            &payload->source_ref) ||
                        payload->data_length != payload->target_ref.length)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay update operation payload is invalid");
      }
      iree_const_byte_span_t update_data;
      return iree_hal_replay_executor_transfer_data_span(data, payload,
                                                         &update_data);
    }
    case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_COPY:
      if (payload->source_ref.length == 0 && payload->target_ref.length == 0) {
        return iree_hal_replay_executor_validate_empty_transfer_operation(
            payload);
      }
      if (IREE_UNLIKELY(
              payload->source_ref.length != payload->target_ref.length ||
              payload->data_offset != 0 || payload->data_length != 0)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay copy operation payload is invalid");
      }
      return iree_ok_status();
    case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPLOAD: {
      if (payload->target_ref.length == 0) {
        return iree_hal_replay_executor_validate_empty_transfer_operation(
            payload);
      }
      if (IREE_UNLIKELY(payload->flags != 0 ||
                        !iree_hal_replay_buffer_ref_payload_is_empty(
                            &payload->source_ref) ||
                        payload->data_length != payload->target_ref.length)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay upload operation payload is invalid");
      }
      iree_const_byte_span_t upload_data;
      return iree_hal_replay_executor_transfer_data_span(data, payload,
                                                         &upload_data);
    }
    case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      if (payload->source_ref.length == 0) {
        return iree_hal_replay_executor_validate_empty_transfer_operation(
            payload);
      }
      if (IREE_UNLIKELY(payload->flags != 0 ||
                        !iree_hal_replay_buffer_ref_payload_is_empty(
                            &payload->target_ref) ||
                        payload->data_offset != 0 ||
                        payload->data_length != 0 ||
                        payload->source_ref.length > IREE_HOST_SIZE_MAX ||
                        !iree_host_size_checked_add(
                            *inout_download_data_length,
                            (iree_host_size_t)payload->source_ref.length,
                            inout_download_data_length))) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay download operation payload is invalid");
      }
      return iree_ok_status();
    case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_NONE:
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "replay queue transfer operation type is none");
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "replay queue transfer operation type %u is unsupported",
          payload->type);
  }
}

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

static iree_status_t iree_hal_replay_executor_queue_transfer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TRANSFER,
      sizeof(iree_hal_replay_queue_transfer_payload_t)));
  iree_hal_replay_queue_transfer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.wait_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.signal_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.operation_count > IREE_HOST_SIZE_MAX ||
                    payload.data_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue transfer count exceeds host size");
  }

  const iree_host_size_t operation_count =
      (iree_host_size_t)payload.operation_count;
  iree_host_size_t operation_payloads_size = 0;
  iree_host_size_t trailing_payload_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(
              operation_count,
              sizeof(iree_hal_replay_queue_transfer_operation_payload_t),
              &operation_payloads_size) ||
          !iree_host_size_checked_add(operation_payloads_size,
                                      (iree_host_size_t)payload.data_length,
                                      &trailing_payload_size))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue transfer payload size overflow");
  }

  iree_hal_replay_object_entry_t* queue_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      &queue_entry));

  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, trailing_payload_size, &wait_storage,
      &signal_storage, &trailing_payload));
  const uint8_t* operation_payloads = trailing_payload.data;
  const iree_const_byte_span_t data =
      iree_make_const_byte_span(trailing_payload.data + operation_payloads_size,
                                (iree_host_size_t)payload.data_length);

  iree_status_t status = iree_ok_status();
  iree_host_size_t download_data_length = 0;
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_replay_queue_transfer_operation_payload_t operation_payload;
    memcpy(&operation_payload,
           operation_payloads + i * sizeof(operation_payload),
           sizeof(operation_payload));
    status = iree_hal_replay_executor_validate_transfer_operation(
        &operation_payload, data, &download_data_length);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "transfer operation %" PRIhsz, i);
    }
  }

  iree_hal_transfer_operation_t* operations = NULL;
  iree_host_size_t operations_size = 0;
  if (iree_status_is_ok(status) && operation_count != 0 &&
      IREE_UNLIKELY(!iree_host_size_checked_mul(
          operation_count, sizeof(*operations), &operations_size))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "replay transfer operation storage overflow");
  }
  if (iree_status_is_ok(status) && operations_size != 0) {
    status = iree_allocator_malloc(executor->host_allocator, operations_size,
                                   (void**)&operations);
    if (iree_status_is_ok(status)) memset(operations, 0, operations_size);
  }
  uint8_t* download_data = NULL;
  if (iree_status_is_ok(status) && download_data_length != 0) {
    status = iree_allocator_malloc(
        executor->host_allocator, download_data_length, (void**)&download_data);
  }

  iree_host_size_t download_data_offset = 0;
  for (iree_host_size_t i = 0; i < operation_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_replay_queue_transfer_operation_payload_t operation_payload;
    memcpy(&operation_payload,
           operation_payloads + i * sizeof(operation_payload),
           sizeof(operation_payload));
    iree_hal_transfer_operation_t* operation = &operations[i];
    iree_hal_buffer_ref_t source_ref = {0};
    iree_hal_buffer_ref_t target_ref = {0};
    iree_const_byte_span_t operation_data = iree_make_const_byte_span(NULL, 0);
    switch (operation_payload.type) {
      case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_FILL:
        operation->type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
        if (operation_payload.target_ref.length == 0) break;
        status = iree_hal_replay_executor_make_direct_transfer_buffer_ref(
            executor, &operation_payload.target_ref, &target_ref);
        if (iree_status_is_ok(status)) {
          status = iree_hal_replay_executor_transfer_data_span(
              data, &operation_payload, &operation_data);
        }
        if (iree_status_is_ok(status)) {
          operation->fill.target_buffer = target_ref.buffer;
          operation->fill.target_offset = target_ref.offset;
          operation->fill.length = target_ref.length;
          operation->fill.pattern = operation_data.data;
          operation->fill.pattern_length = operation_data.data_length;
          operation->fill.flags = operation_payload.flags;
        }
        break;
      case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPDATE:
        operation->type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
        if (operation_payload.target_ref.length == 0) break;
        status = iree_hal_replay_executor_make_direct_transfer_buffer_ref(
            executor, &operation_payload.target_ref, &target_ref);
        if (iree_status_is_ok(status)) {
          status = iree_hal_replay_executor_transfer_data_span(
              data, &operation_payload, &operation_data);
        }
        if (iree_status_is_ok(status)) {
          operation->update.source_buffer = operation_data.data;
          operation->update.target_buffer = target_ref.buffer;
          operation->update.target_offset = target_ref.offset;
          operation->update.length = target_ref.length;
          operation->update.flags = operation_payload.flags;
        }
        break;
      case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_COPY:
        operation->type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
        if (operation_payload.source_ref.length == 0) break;
        status = iree_hal_replay_executor_make_direct_transfer_buffer_ref(
            executor, &operation_payload.source_ref, &source_ref);
        if (iree_status_is_ok(status)) {
          status = iree_hal_replay_executor_make_direct_transfer_buffer_ref(
              executor, &operation_payload.target_ref, &target_ref);
        }
        if (iree_status_is_ok(status)) {
          operation->copy.source_buffer = source_ref.buffer;
          operation->copy.source_offset = source_ref.offset;
          operation->copy.target_buffer = target_ref.buffer;
          operation->copy.target_offset = target_ref.offset;
          operation->copy.length = source_ref.length;
          operation->copy.flags = operation_payload.flags;
        }
        break;
      case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPLOAD:
        operation->type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
        if (operation_payload.target_ref.length == 0) break;
        status = iree_hal_replay_executor_make_direct_transfer_buffer_ref(
            executor, &operation_payload.target_ref, &target_ref);
        if (iree_status_is_ok(status)) {
          status = iree_hal_replay_executor_transfer_data_span(
              data, &operation_payload, &operation_data);
        }
        if (iree_status_is_ok(status)) {
          operation->upload.source = operation_data.data;
          operation->upload.target_buffer = target_ref.buffer;
          operation->upload.target_offset = target_ref.offset;
          operation->upload.length = target_ref.length;
        }
        break;
      case IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        operation->type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
        if (operation_payload.source_ref.length == 0) break;
        status = iree_hal_replay_executor_make_direct_transfer_buffer_ref(
            executor, &operation_payload.source_ref, &source_ref);
        if (iree_status_is_ok(status)) {
          operation->download.source_buffer = source_ref.buffer;
          operation->download.source_offset = source_ref.offset;
          operation->download.target = download_data + download_data_offset;
          operation->download.length = source_ref.length;
          download_data_offset += (iree_host_size_t)source_ref.length;
        }
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "validated replay queue transfer operation type changed");
        break;
    }
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "transfer operation %" PRIhsz, i);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_transfer(queue_entry->value.queue,
                                     wait_storage.list, signal_storage.list,
                                     operation_count, operations);
  }
  if (iree_status_is_ok(status) && signal_storage.list.count != 0) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_allocator_free(executor->host_allocator, download_data);
  iree_allocator_free(executor->host_allocator, operations);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_read_exact(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ,
      sizeof(iree_hal_replay_queue_read_payload_t)));
  iree_hal_replay_queue_read_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* queue_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      &queue_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t captured_data;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, payload.captured_data_length,
      &wait_storage, &signal_storage, &captured_data);
  iree_hal_buffer_ref_t target_ref = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_direct_transfer_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  iree_hal_replay_object_entry_t* file_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, payload.source_file_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_FILE,
                                             &file_entry);
  }
  if (iree_status_is_ok(status) && payload.captured_data_length != 0 &&
      payload.captured_data_length != target_ref.length) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue read captured data length does not match target length");
  }

  if (iree_status_is_ok(status) && payload.captured_data_length != 0) {
    status = iree_hal_queue_update(
        queue_entry->value.queue, wait_storage.list, signal_storage.list,
        captured_data.data, /*source_offset=*/0, target_ref.buffer,
        target_ref.offset, target_ref.length, IREE_HAL_UPDATE_FLAG_NONE);
  } else if (iree_status_is_ok(status) && !file_entry->value.file) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay queue read requires an imported file or captured data");
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_queue_read(
        queue_entry->value.queue, wait_storage.list, signal_storage.list,
        file_entry->value.file, payload.source_offset, target_ref.buffer,
        target_ref.offset, target_ref.length, payload.flags);
  }
  if (iree_status_is_ok(status) && signal_storage.list.count != 0) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_write_exact(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_WRITE,
      sizeof(iree_hal_replay_queue_write_payload_t)));
  iree_hal_replay_queue_write_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* queue_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      &queue_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_ref_t source_ref = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_direct_transfer_buffer_ref(
        executor, &payload.source_ref, &source_ref);
  }
  iree_hal_replay_object_entry_t* file_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, payload.target_file_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_FILE,
                                             &file_entry);
  }
  if (iree_status_is_ok(status) && !file_entry->value.file) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "replay queue write requires an imported target file");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_write(
        queue_entry->value.queue, wait_storage.list, signal_storage.list,
        source_ref.buffer, source_ref.offset, file_entry->value.file,
        payload.target_offset, source_ref.length, payload.flags);
  }
  if (iree_status_is_ok(status) && signal_storage.list.count != 0) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_alloca(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ALLOCA,
      sizeof(iree_hal_replay_queue_alloca_payload_t)));
  iree_hal_replay_queue_alloca_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.reserved0 != 0 || payload.request_count == 0 ||
                    payload.request_count > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue alloca header is invalid");
  }
  const iree_host_size_t request_count =
      (iree_host_size_t)payload.request_count;
  iree_host_size_t request_payloads_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          request_count, sizeof(iree_hal_replay_queue_alloca_request_payload_t),
          &request_payloads_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue alloca request size overflow");
  }

  iree_hal_replay_object_entry_t* queue_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      &queue_entry));
  iree_hal_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_queue_allocation_pool(
      executor, record, queue_entry, &pool));

  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, request_payloads_size, &wait_storage,
      &signal_storage, &trailing_payload);
  iree_hal_replay_queue_alloca_request_payload_t* request_payloads = NULL;
  iree_hal_pool_reservation_request_t* requests = NULL;
  iree_hal_buffer_t** buffers = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_allocate_array(executor, request_count,
                                                     sizeof(*request_payloads),
                                                     (void**)&request_payloads);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_allocate_array(
        executor, request_count, sizeof(*requests), (void**)&requests);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_allocate_array(
        executor, request_count, sizeof(*buffers), (void**)&buffers);
  }
  if (iree_status_is_ok(status)) {
    memset(buffers, 0, request_count * sizeof(*buffers));
    memcpy(request_payloads, trailing_payload.data, request_payloads_size);
  }
  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_replay_object_id_t buffer_id = request_payloads[i].buffer_id;
    if (IREE_UNLIKELY(buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE ||
                      buffer_id >= executor->object_capacity ||
                      executor->objects[buffer_id].type !=
                          IREE_HAL_REPLAY_OBJECT_TYPE_NONE)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue alloca buffer id %" PRIu64
                                " is invalid or assigned",
                                buffer_id);
      break;
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (IREE_UNLIKELY(request_payloads[j].buffer_id == buffer_id)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay queue alloca repeats buffer id %" PRIu64, buffer_id);
        break;
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_replay_executor_make_buffer_params(
          &request_payloads[i].allocation, &requests[i].params);
      requests[i].allocation_size =
          request_payloads[i].allocation.allocation_size;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_alloca(queue_entry->value.queue, wait_storage.list,
                                   signal_storage.list, pool, request_count,
                                   requests, buffers);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      const iree_hal_replay_object_id_t buffer_id =
          request_payloads[i].buffer_id;
      executor->objects[buffer_id] = (iree_hal_replay_object_entry_t){
          .type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
          .value.buffer = buffers[i],
      };
      buffers[i] = NULL;
    }
  }
  if (buffers) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
  }
  iree_allocator_free(executor->host_allocator, buffers);
  iree_allocator_free(executor->host_allocator, requests);
  iree_allocator_free(executor->host_allocator, request_payloads);
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
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_DEALLOCA,
      sizeof(iree_hal_replay_queue_dealloca_payload_t)));
  iree_hal_replay_queue_dealloca_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.reserved0 != 0 || payload.buffer_count == 0 ||
                    payload.buffer_count > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue dealloca header is invalid");
  }
  const iree_host_size_t buffer_count = (iree_host_size_t)payload.buffer_count;
  iree_host_size_t buffer_ids_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          buffer_count, sizeof(iree_hal_replay_object_id_t),
          &buffer_ids_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue dealloca buffer id size overflow");
  }

  iree_hal_replay_object_entry_t* queue_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      &queue_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, buffer_ids_size, &wait_storage,
      &signal_storage, &trailing_payload);
  iree_hal_replay_object_id_t* buffer_ids = NULL;
  iree_hal_buffer_t** buffers = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_allocate_array(
        executor, buffer_count, sizeof(*buffer_ids), (void**)&buffer_ids);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_allocate_array(
        executor, buffer_count, sizeof(*buffers), (void**)&buffers);
  }
  if (iree_status_is_ok(status)) {
    memcpy(buffer_ids, trailing_payload.data, buffer_ids_size);
  }
  for (iree_host_size_t i = 0; i < buffer_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_replay_object_entry_t* buffer_entry = NULL;
    status = iree_hal_replay_executor_lookup(executor, buffer_ids[i],
                                             IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
                                             &buffer_entry);
    if (iree_status_is_ok(status)) buffers[i] = buffer_entry->value.buffer;
    for (iree_host_size_t j = 0; j < i && iree_status_is_ok(status); ++j) {
      if (IREE_UNLIKELY(buffer_ids[j] == buffer_ids[i])) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay queue dealloca repeats buffer id %" PRIu64, buffer_ids[i]);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_queue_dealloca(queue_entry->value.queue, wait_storage.list,
                                signal_storage.list, buffer_count, buffers);
  }
  if (iree_status_is_ok(status) && signal_storage.list.count != 0) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_allocator_free(executor->host_allocator, buffers);
  iree_allocator_free(executor->host_allocator, buffer_ids);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_device_queue_alloca(
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
  if (IREE_UNLIKELY(payload.flags != 0)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "legacy replay queue alloca flags 0x%016" PRIx64
                            " cannot be represented by exact queue allocation",
                            payload.flags);
  }
  iree_hal_replay_object_entry_t* queue_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_find_provisioned_queue(
      executor, device_entry->value.device, payload.queue_affinity,
      &queue_entry));
  iree_hal_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_queue_allocation_pool(
      executor, record, queue_entry, &pool));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_pool_reservation_request_t request = {
        .allocation_size = payload.allocation.allocation_size,
    };
    status = iree_hal_replay_executor_make_buffer_params(&payload.allocation,
                                                         &request.params);
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_alloca(queue_entry->value.queue,
                                     wait_storage.list, signal_storage.list,
                                     pool, 1, &request, &buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
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

static iree_status_t iree_hal_replay_executor_device_queue_dealloca(
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
  if (IREE_UNLIKELY((payload.flags & ~1ull) != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "legacy replay queue dealloca flags 0x%016" PRIx64
        " cannot be represented by exact queue deallocation",
        payload.flags);
  }
  iree_hal_replay_object_entry_t* queue_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_find_provisioned_queue(
      executor, device_entry->value.device, payload.queue_affinity,
      &queue_entry));
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
    iree_hal_buffer_t* buffers[] = {buffer_ref.buffer};
    status =
        iree_hal_queue_dealloca(queue_entry->value.queue, wait_storage.list,
                                signal_storage.list, 1, buffers);
  }
  if (iree_status_is_ok(status) && signal_storage.list.count != 0) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
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

static iree_status_t iree_hal_replay_executor_queue_atomic_wait(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT,
      sizeof(iree_hal_replay_device_queue_atomic_wait_payload_t)));
  iree_hal_replay_device_queue_atomic_wait_payload_t payload;
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
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_direct_atomic_buffer_ref(
        executor, &payload.target_ref, payload.params.width, &target_ref);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_atomic_wait(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, target_ref.buffer, target_ref.offset,
        iree_hal_replay_executor_atomic_wait_params(payload.params));
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

static iree_status_t iree_hal_replay_executor_queue_atomic_store(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
      sizeof(iree_hal_replay_device_queue_atomic_store_payload_t)));
  iree_hal_replay_device_queue_atomic_store_payload_t payload;
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
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_direct_atomic_buffer_ref(
        executor, &payload.target_ref, payload.params.width, &target_ref);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_atomic_store(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, target_ref.buffer, target_ref.offset,
        iree_hal_replay_executor_atomic_store_params(payload.params));
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

static iree_status_t iree_hal_replay_executor_queue_atomic_rmw(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_RMW,
      sizeof(iree_hal_replay_device_queue_atomic_rmw_payload_t)));
  iree_hal_replay_device_queue_atomic_rmw_payload_t payload;
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
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_direct_atomic_buffer_ref(
        executor, &payload.target_ref, payload.params.width, &target_ref);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_atomic_rmw(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, target_ref.buffer, target_ref.offset,
        iree_hal_replay_executor_atomic_rmw_params(payload.params));
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

static iree_status_t iree_hal_replay_executor_command_buffer_atomic_wait(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_WAIT,
      sizeof(iree_hal_replay_command_buffer_atomic_wait_payload_t)));
  iree_hal_replay_command_buffer_atomic_wait_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_atomic_buffer_ref(
      executor, &payload.target_ref, payload.params.width, &target_ref));
  return iree_hal_command_buffer_atomic_wait(
      command_buffer_entry->value.command_buffer, payload.source_stage_mask,
      payload.target_stage_mask, target_ref,
      iree_hal_replay_executor_atomic_wait_params(payload.params));
}

static iree_status_t iree_hal_replay_executor_command_buffer_atomic_store(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE,
      sizeof(iree_hal_replay_command_buffer_atomic_store_payload_t)));
  iree_hal_replay_command_buffer_atomic_store_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_atomic_buffer_ref(
      executor, &payload.target_ref, payload.params.width, &target_ref));
  return iree_hal_command_buffer_atomic_store(
      command_buffer_entry->value.command_buffer, payload.source_stage_mask,
      payload.target_stage_mask, target_ref,
      iree_hal_replay_executor_atomic_store_params(payload.params));
}

static iree_status_t iree_hal_replay_executor_command_buffer_atomic_rmw(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_RMW,
      sizeof(iree_hal_replay_command_buffer_atomic_rmw_payload_t)));
  iree_hal_replay_command_buffer_atomic_rmw_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_atomic_buffer_ref(
      executor, &payload.target_ref, payload.params.width, &target_ref));
  return iree_hal_command_buffer_atomic_rmw(
      command_buffer_entry->value.command_buffer, payload.source_stage_mask,
      payload.target_stage_mask, target_ref,
      iree_hal_replay_executor_atomic_rmw_params(payload.params));
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
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE:
      return iree_hal_replay_executor_replay_object_operation(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_TRANSFER:
      return iree_hal_replay_executor_queue_transfer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_READ:
      return iree_hal_replay_executor_queue_read_exact(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_WRITE:
      return iree_hal_replay_executor_queue_write_exact(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ALLOCA:
      return iree_hal_replay_executor_queue_alloca(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_DEALLOCA:
      return iree_hal_replay_executor_queue_dealloca(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA:
      return iree_hal_replay_executor_device_queue_alloca(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA:
      return iree_hal_replay_executor_device_queue_dealloca(executor, record);
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
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_WAIT:
      return iree_hal_replay_executor_queue_atomic_wait(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE:
      return iree_hal_replay_executor_queue_atomic_store(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_RMW:
      return iree_hal_replay_executor_queue_atomic_rmw(executor, record);
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
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_WAIT:
      return iree_hal_replay_executor_command_buffer_atomic_wait(executor,
                                                                 record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE:
      return iree_hal_replay_executor_command_buffer_atomic_store(executor,
                                                                  record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_RMW:
      return iree_hal_replay_executor_command_buffer_atomic_rmw(executor,
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
