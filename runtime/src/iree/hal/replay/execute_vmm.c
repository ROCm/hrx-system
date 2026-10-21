// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute_vmm.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/replay/execute_pool.h"

static iree_status_t iree_hal_replay_executor_require_fixed_vmm_payload(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_payload_type_t payload_type,
    iree_host_size_t payload_length) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, payload_type, payload_length));
  if (IREE_UNLIKELY(record->payload.data_length != payload_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay VMM payload length mismatch");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_lookup_owned_reservation(
    iree_hal_replay_executor_t* executor, iree_hal_allocator_t* allocator,
    iree_hal_replay_object_id_t virtual_buffer_id,
    iree_hal_replay_object_entry_t** out_entry) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, virtual_buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      out_entry));
  if (IREE_UNLIKELY(!(*out_entry)->owning_allocator ||
                    (*out_entry)->owning_allocator != allocator)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay virtual memory reservation belongs to another allocator");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_lookup_owned_physical_memory(
    iree_hal_replay_executor_t* executor, iree_hal_allocator_t* allocator,
    iree_hal_replay_object_id_t physical_memory_id,
    iree_hal_replay_object_entry_t** out_entry) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, physical_memory_id, IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY,
      out_entry));
  if (IREE_UNLIKELY((*out_entry)->owning_allocator != allocator)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay physical memory belongs to another allocator");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_virtual_memory_release(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_vmm_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
      sizeof(iree_hal_replay_allocator_virtual_memory_release_payload_t)));
  iree_hal_replay_allocator_virtual_memory_release_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.virtual_buffer_id !=
                    record->header.related_object_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay virtual memory release object ids do not match");
  }

  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));
  iree_hal_replay_object_entry_t* buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup_owned_reservation(
      executor, allocator_entry->value.allocator, payload.virtual_buffer_id,
      &buffer_entry));
  if (IREE_UNLIKELY(iree_hal_replay_executor_has_virtual_memory_mapping(
          executor, payload.virtual_buffer_id))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay virtual memory reservation is released while still mapped");
  }
  IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_release(
      allocator_entry->value.allocator, buffer_entry->value.buffer));
  return iree_hal_replay_executor_forget(executor, payload.virtual_buffer_id,
                                         IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER);
}

static iree_status_t iree_hal_replay_executor_physical_memory_free(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_vmm_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
      sizeof(iree_hal_replay_allocator_physical_memory_free_payload_t)));
  iree_hal_replay_allocator_physical_memory_free_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.physical_memory_id !=
                    record->header.related_object_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay physical memory free object ids do not match");
  }

  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));
  iree_hal_replay_object_entry_t* physical_memory_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup_owned_physical_memory(
      executor, allocator_entry->value.allocator, payload.physical_memory_id,
      &physical_memory_entry));
  if (IREE_UNLIKELY(iree_hal_replay_executor_has_physical_memory_mapping(
          executor, payload.physical_memory_id))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay physical memory is freed while still mapped");
  }
  IREE_RETURN_IF_ERROR(iree_hal_allocator_physical_memory_free(
      allocator_entry->value.allocator,
      physical_memory_entry->value.physical_memory.handle));
  return iree_hal_replay_executor_forget(
      executor, payload.physical_memory_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY);
}

static iree_status_t iree_hal_replay_executor_virtual_memory_map(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_vmm_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
      sizeof(iree_hal_replay_allocator_virtual_memory_map_payload_t)));
  iree_hal_replay_allocator_virtual_memory_map_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.virtual_buffer_id !=
                    record->header.related_object_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay virtual memory map object ids do not match");
  }

  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));
  return iree_hal_replay_executor_map_virtual_memory(
      executor, allocator_entry->value.allocator, payload.virtual_buffer_id,
      payload.physical_memory_id, payload.virtual_offset,
      payload.physical_offset, payload.size);
}

static iree_status_t iree_hal_replay_executor_virtual_memory_unmap(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_vmm_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
      sizeof(iree_hal_replay_allocator_virtual_memory_unmap_payload_t)));
  iree_hal_replay_allocator_virtual_memory_unmap_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.virtual_buffer_id !=
                    record->header.related_object_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay virtual memory unmap object ids do not match");
  }

  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));
  return iree_hal_replay_executor_unmap_virtual_memory(
      executor, allocator_entry->value.allocator, payload.virtual_buffer_id,
      payload.virtual_offset, payload.size);
}

static iree_status_t iree_hal_replay_executor_virtual_memory_protect(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_vmm_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
      sizeof(iree_hal_replay_allocator_virtual_memory_protect_payload_t)));
  iree_hal_replay_allocator_virtual_memory_protect_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.virtual_buffer_id !=
                        record->header.related_object_id ||
                    payload.reserved0 != 0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay virtual memory protect metadata is invalid");
  }

  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));
  iree_hal_replay_object_entry_t* buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup_owned_reservation(
      executor, allocator_entry->value.allocator, payload.virtual_buffer_id,
      &buffer_entry));
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_executor_drain_queue_completions(executor));
  return iree_hal_allocator_virtual_memory_protect(
      allocator_entry->value.allocator, buffer_entry->value.buffer,
      payload.virtual_offset, payload.size, payload.queue_family_affinity,
      payload.access_scope, payload.protection);
}

static iree_status_t iree_hal_replay_executor_virtual_memory_advise(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_vmm_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
      sizeof(iree_hal_replay_allocator_virtual_memory_advise_payload_t)));
  iree_hal_replay_allocator_virtual_memory_advise_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.virtual_buffer_id !=
                    record->header.related_object_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay virtual memory advise object ids do not match");
  }

  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));
  iree_hal_replay_object_entry_t* buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup_owned_reservation(
      executor, allocator_entry->value.allocator, payload.virtual_buffer_id,
      &buffer_entry));
  return iree_hal_allocator_virtual_memory_advise(
      allocator_entry->value.allocator, buffer_entry->value.buffer,
      payload.virtual_offset, payload.size, payload.queue_family_affinity,
      payload.advice);
}

static iree_status_t iree_hal_replay_executor_virtual_buffer_create(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_CREATE,
      sizeof(iree_hal_replay_virtual_buffer_create_payload_t)));
  iree_hal_replay_virtual_buffer_create_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t family_payload_length = 0;
  iree_host_size_t payload_length = 0;
  if (IREE_UNLIKELY(payload.reserved0 || payload.host_access > UINT16_MAX ||
                    payload.family_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.family_count,
                        sizeof(iree_hal_replay_pool_family_access_payload_t),
                        &family_payload_length) ||
                    !iree_host_size_checked_add(sizeof(payload),
                                                family_payload_length,
                                                &payload_length) ||
                    payload_length != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay reservation payload is invalid");
  }
  iree_hal_pool_family_access_t* families = NULL;
  if (payload.family_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        executor->host_allocator, (iree_host_size_t)payload.family_count,
        sizeof(*families), (void**)&families));
  }
  iree_status_t status = iree_hal_replay_executor_pool_decode_families(
      executor, (iree_host_size_t)payload.family_count,
      record->payload.data + sizeof(payload), families);
  const iree_hal_pool_scope_t scope = {
      .family_count = (iree_host_size_t)payload.family_count,
      .families = families,
      .host =
          {
              .access = (iree_hal_memory_access_t)payload.host_access,
              .modes = (iree_hal_mapping_mode_t)payload.host_mapping_modes,
              .cacheability =
                  (iree_hal_host_cacheability_t)payload.host_cacheability,
              .atomics =
                  {
                      .device_scope_32 = payload.host_atomic_device_scope_32,
                      .device_scope_64 = payload.host_atomic_device_scope_64,
                      .system_scope_32 = payload.host_atomic_system_scope_32,
                      .system_scope_64 = payload.host_atomic_system_scope_64,
                  },
          },
  };
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = payload.byte_length,
      .min_alignment = payload.min_alignment,
      .mapping_granularity = payload.mapping_granularity,
  };
  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_virtual_buffer_create(executor->device_group, scope, &options,
                                       executor->host_allocator, &buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
        (iree_hal_replay_object_entry_t){.value.buffer = buffer});
  }
  iree_allocator_free(executor->host_allocator, families);
  return status;
}

static iree_status_t iree_hal_replay_executor_virtual_buffer_detach(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_vmm_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_DETACH,
      sizeof(iree_hal_replay_virtual_buffer_detach_payload_t)));
  iree_hal_replay_virtual_buffer_detach_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (payload.reserved0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay virtual buffer detach metadata is invalid");
  }
  iree_hal_replay_object_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      &entry));
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_executor_drain_queue_completions(executor));
  const iree_hal_memory_scope_t scope = {
      .domain = iree_hal_device_group_memory_domain(executor->device_group),
      .id = payload.scope_id,
  };
  return iree_hal_virtual_buffer_detach(entry->value.buffer, scope);
}

static iree_status_t iree_hal_replay_executor_queue_map(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_MAP,
      sizeof(iree_hal_replay_queue_map_payload_t)));
  iree_hal_replay_queue_map_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t mapping_payload_length = 0;
  if (IREE_UNLIKELY(payload.mapping_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.mapping_count,
                        sizeof(iree_hal_replay_memory_mapping_payload_t),
                        &mapping_payload_length))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay mapping batch header is invalid");
  }
  iree_hal_replay_object_entry_t* queue_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      &queue_entry));
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.device_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t waits = {0};
  iree_hal_replay_semaphore_list_storage_t signals = {0};
  iree_const_byte_span_t mapping_data;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, mapping_payload_length, &waits, &signals,
      &mapping_data));
  iree_hal_memory_mapping_t* mappings = NULL;
  iree_status_t status = iree_ok_status();
  if (payload.mapping_count) {
    status = iree_allocator_malloc_array(
        executor->host_allocator, (iree_host_size_t)payload.mapping_count,
        sizeof(*mappings), (void**)&mappings);
  }
  for (iree_host_size_t i = 0;
       i < payload.mapping_count && iree_status_is_ok(status); ++i) {
    iree_hal_replay_memory_mapping_payload_t mapping;
    memcpy(&mapping, mapping_data.data + i * sizeof(mapping), sizeof(mapping));
    if (IREE_UNLIKELY(mapping.reserved0 || mapping.access > UINT16_MAX ||
                      mapping.source_ref.reserved0 ||
                      mapping.source_ref.buffer_slot ||
                      mapping.source_ref.length != mapping.byte_length)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay mapping descriptor is invalid");
    }
    iree_hal_replay_object_entry_t* target_entry = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_replay_executor_lookup(
          executor, mapping.target_buffer_id,
          IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, &target_entry);
    }
    iree_hal_replay_object_entry_t* source_entry = NULL;
    if (iree_status_is_ok(status) && mapping.source_ref.buffer_id) {
      status = iree_hal_replay_executor_lookup(
          executor, mapping.source_ref.buffer_id,
          IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, &source_entry);
    }
    if (iree_status_is_ok(status)) {
      mappings[i] = (iree_hal_memory_mapping_t){
          .target = target_entry->value.buffer,
          .scope =
              {
                  .domain = iree_hal_device_group_memory_domain(
                      executor->device_group),
                  .id = mapping.scope_id,
              },
          .target_range = iree_hal_make_buffer_range(mapping.target_offset,
                                                     mapping.byte_length),
          .source = source_entry ? source_entry->value.buffer : NULL,
          .source_offset = mapping.source_ref.offset,
          .access = (iree_hal_memory_access_t)mapping.access,
          .operation = (iree_hal_memory_mapping_operation_t)mapping.operation,
      };
    }
  }
  iree_hal_replay_queue_completion_t* completion = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_prepare_queue_completion(
        executor, device_entry->value.device, queue_entry->value.queue,
        &signals, &completion);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_queue_map(queue_entry->value.queue, waits.list, signals.list,
                           (iree_host_size_t)payload.mapping_count, mappings,
                           (iree_hal_queue_map_flags_t)payload.flags);
  }
  status = iree_hal_replay_executor_finalize_queue_completion(
      executor, completion, /*flush_queue=*/true, status);
  iree_allocator_free(executor->host_allocator, mappings);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signals,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&waits,
                                                      executor->host_allocator);
  return status;
}

iree_status_t iree_hal_replay_executor_replay_vmm_operation(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_CREATE:
      return iree_hal_replay_executor_virtual_buffer_create(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_DETACH:
      return iree_hal_replay_executor_virtual_buffer_detach(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_MAP:
      return iree_hal_replay_executor_queue_map(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE:
      return iree_hal_replay_executor_virtual_memory_release(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE:
      return iree_hal_replay_executor_physical_memory_free(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP:
      return iree_hal_replay_executor_virtual_memory_map(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP:
      return iree_hal_replay_executor_virtual_memory_unmap(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT:
      return iree_hal_replay_executor_virtual_memory_protect(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE:
      return iree_hal_replay_executor_virtual_memory_advise(executor, record);
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "replay VMM operation %s is not implemented",
          iree_hal_replay_operation_code_string(record->header.operation_code));
  }
}
