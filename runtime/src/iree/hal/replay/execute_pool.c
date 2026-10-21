// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute_pool.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/slab_pool.h"

typedef struct iree_hal_replay_import_allocation_t {
  // Allocator used to release this allocation after the imported buffer dies.
  iree_allocator_t host_allocator;
  // Captured external storage bytes follow this header.
  uint8_t data[];
} iree_hal_replay_import_allocation_t;

static iree_status_t iree_hal_replay_executor_require_fixed_pool_payload(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_payload_type_t payload_type,
    iree_host_size_t payload_length) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, payload_type, payload_length));
  if (IREE_UNLIKELY(record->payload.data_length != payload_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay pool payload length mismatch");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_allocate_pool_array(
    iree_hal_replay_executor_t* executor, iree_host_size_t element_count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (element_count == 0) {
    return iree_ok_status();
  }
  return iree_allocator_malloc_array(executor->host_allocator, element_count,
                                     element_size, out_ptr);
}

static iree_status_t iree_hal_replay_decode_external_storage_class(
    iree_hal_replay_external_storage_class_payload_t payload,
    iree_hal_external_storage_class_t* out_storage_class) {
  if (IREE_UNLIKELY(
          payload.reserved0 != 0 ||
          payload.type < IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER ||
          payload.type > IREE_HAL_EXTERNAL_STORAGE_TYPE_D3D12_RESOURCE)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay external storage class is invalid");
  }
  *out_storage_class = (iree_hal_external_storage_class_t){
      .type = (iree_hal_external_storage_type_t)payload.type,
      .provenance = {payload.provenance[0], payload.provenance[1]},
  };
  return iree_ok_status();
}

iree_status_t iree_hal_replay_executor_pool_decode_families(
    iree_hal_replay_executor_t* executor, iree_host_size_t family_count,
    const uint8_t* family_payload_data,
    iree_hal_pool_family_access_t* families) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < family_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_replay_pool_family_access_payload_t family_payload;
    memcpy(&family_payload, family_payload_data + i * sizeof(family_payload),
           sizeof(family_payload));
    if (IREE_UNLIKELY(
            family_payload.reserved0 != 0 ||
            family_payload.device_ordinal >=
                iree_hal_device_group_device_count(executor->device_group))) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay pool family row is invalid");
    }
    const iree_hal_queue_family_t* family = NULL;
    if (iree_status_is_ok(status)) {
      iree_hal_device_t* device = iree_hal_device_group_device_at(
          executor->device_group, family_payload.device_ordinal);
      family =
          iree_hal_device_queue_family(device, family_payload.family_ordinal);
      if (IREE_UNLIKELY(!family)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay device %u does not expose queue family %u",
            family_payload.device_ordinal, family_payload.family_ordinal);
      }
    }
    if (iree_status_is_ok(status)) {
      families[i] = (iree_hal_pool_family_access_t){
          .family = family,
          .usage = (iree_hal_buffer_usage_t)family_payload.usage,
          .interfaces = (iree_hal_buffer_interfaces_t)family_payload.interfaces,
          .requirements =
              (iree_hal_pool_access_requirements_t)family_payload.requirements,
          .atomics = {.queue =
                          {
                              .device_scope_32 =
                                  family_payload.queue_atomic_device_scope_32,
                              .device_scope_64 =
                                  family_payload.queue_atomic_device_scope_64,
                              .system_scope_32 =
                                  family_payload.queue_atomic_system_scope_32,
                              .system_scope_64 =
                                  family_payload.queue_atomic_system_scope_64,
                          },
                      .program =
                          {
                              .device_scope_32 =
                                  family_payload.program_atomic_device_scope_32,
                              .device_scope_64 =
                                  family_payload.program_atomic_device_scope_64,
                              .system_scope_32 =
                                  family_payload.program_atomic_system_scope_32,
                              .system_scope_64 =
                                  family_payload.program_atomic_system_scope_64,
                          }},
      };
    }
  }
  return status;
}

static iree_status_t iree_hal_replay_executor_pool_create_slab(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_CREATE_SLAB,
      sizeof(iree_hal_replay_pool_create_slab_payload_t)));
  iree_hal_replay_pool_create_slab_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.reserved0 != 0 ||
                    payload.host_access > UINT16_MAX ||
                    payload.preferred_host_access > UINT16_MAX ||
                    payload.family_count > IREE_HOST_SIZE_MAX ||
                    payload.owned_export_count > IREE_HOST_SIZE_MAX ||
                    payload.import_count > IREE_HOST_SIZE_MAX ||
                    payload.import_export_count > IREE_HOST_SIZE_MAX ||
                    payload.trace_name_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay slab pool header is invalid");
  }

  const iree_host_size_t family_count = (iree_host_size_t)payload.family_count;
  const iree_host_size_t owned_export_count =
      (iree_host_size_t)payload.owned_export_count;
  const iree_host_size_t import_count = (iree_host_size_t)payload.import_count;
  const iree_host_size_t import_export_count =
      (iree_host_size_t)payload.import_export_count;
  const iree_host_size_t trace_name_length =
      (iree_host_size_t)payload.trace_name_length;
  iree_host_size_t family_payload_length = 0;
  iree_host_size_t owned_export_payload_length = 0;
  iree_host_size_t import_payload_length = 0;
  iree_host_size_t import_export_payload_length = 0;
  iree_host_size_t expected_length = sizeof(payload);
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(
              family_count,
              sizeof(iree_hal_replay_pool_family_access_payload_t),
              &family_payload_length) ||
          !iree_host_size_checked_mul(
              owned_export_count,
              sizeof(iree_hal_replay_external_storage_class_payload_t),
              &owned_export_payload_length) ||
          !iree_host_size_checked_mul(
              import_count, sizeof(iree_hal_replay_pool_import_class_payload_t),
              &import_payload_length) ||
          !iree_host_size_checked_mul(
              import_export_count,
              sizeof(iree_hal_replay_external_storage_class_payload_t),
              &import_export_payload_length) ||
          !iree_host_size_checked_add(expected_length, family_payload_length,
                                      &expected_length) ||
          !iree_host_size_checked_add(
              expected_length, owned_export_payload_length, &expected_length) ||
          !iree_host_size_checked_add(expected_length, import_payload_length,
                                      &expected_length) ||
          !iree_host_size_checked_add(expected_length,
                                      import_export_payload_length,
                                      &expected_length) ||
          !iree_host_size_checked_add(expected_length, trace_name_length,
                                      &expected_length) ||
          expected_length != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay slab pool payload length mismatch");
  }

  const uint8_t* family_payload_data = record->payload.data + sizeof(payload);
  const uint8_t* owned_export_payload_data =
      family_payload_data + family_payload_length;
  const uint8_t* import_payload_data =
      owned_export_payload_data + owned_export_payload_length;
  const uint8_t* import_export_payload_data =
      import_payload_data + import_payload_length;
  const uint8_t* trace_name_data =
      import_export_payload_data + import_export_payload_length;

  iree_hal_pool_family_access_t* families = NULL;
  iree_hal_external_storage_class_t* owned_exports = NULL;
  iree_hal_pool_import_class_t* imports = NULL;
  iree_hal_external_storage_class_t* import_exports = NULL;
  iree_status_t status = iree_hal_replay_executor_allocate_pool_array(
      executor, family_count, sizeof(*families), (void**)&families);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_allocate_pool_array(
        executor, owned_export_count, sizeof(*owned_exports),
        (void**)&owned_exports);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_allocate_pool_array(
        executor, import_count, sizeof(*imports), (void**)&imports);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_allocate_pool_array(
        executor, import_export_count, sizeof(*import_exports),
        (void**)&import_exports);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_pool_decode_families(
        executor, family_count, family_payload_data, families);
  }
  for (iree_host_size_t i = 0;
       i < owned_export_count && iree_status_is_ok(status); ++i) {
    iree_hal_replay_external_storage_class_payload_t class_payload;
    memcpy(&class_payload,
           owned_export_payload_data + i * sizeof(class_payload),
           sizeof(class_payload));
    status = iree_hal_replay_decode_external_storage_class(class_payload,
                                                           &owned_exports[i]);
  }
  for (iree_host_size_t i = 0;
       i < import_export_count && iree_status_is_ok(status); ++i) {
    iree_hal_replay_external_storage_class_payload_t class_payload;
    memcpy(&class_payload,
           import_export_payload_data + i * sizeof(class_payload),
           sizeof(class_payload));
    status = iree_hal_replay_decode_external_storage_class(class_payload,
                                                           &import_exports[i]);
  }
  for (iree_host_size_t i = 0; i < import_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_replay_pool_import_class_payload_t import_payload;
    memcpy(&import_payload, import_payload_data + i * sizeof(import_payload),
           sizeof(import_payload));
    if (IREE_UNLIKELY(import_payload.export_offset > import_export_count ||
                      import_payload.export_count >
                          import_export_count - import_payload.export_offset)) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "replay pool import export-class range is out of bounds");
      break;
    }
    status = iree_hal_replay_decode_external_storage_class(
        import_payload.storage_class, &imports[i].storage_class);
    if (iree_status_is_ok(status)) {
      iree_hal_external_storage_class_t* import_export_classes = NULL;
      if (import_payload.export_count) {
        import_export_classes = import_exports + import_payload.export_offset;
      }
      imports[i].exports = (iree_hal_pool_export_classes_t){
          .class_count = (iree_host_size_t)import_payload.export_count,
          .classes = import_export_classes,
      };
    }
  }

  iree_hal_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_scope_t scope = {
        .family_count = family_count,
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
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.placement = (iree_hal_pool_placement_t){
        .mode = (iree_hal_pool_placement_mode_t)payload.placement_mode,
        .node = payload.placement_node,
    };
    options.owned.mode = (iree_hal_slab_pool_owned_mode_t)payload.owned_mode;
    options.owned.exports = (iree_hal_pool_export_classes_t){
        .class_count = owned_export_count,
        .classes = owned_exports,
    };
    options.imports.class_count = import_count;
    options.imports.classes = imports;
    options.preferences.host = (iree_hal_pool_host_access_t){
        .access = (iree_hal_memory_access_t)payload.preferred_host_access,
        .modes = (iree_hal_mapping_mode_t)payload.preferred_host_mapping_modes,
        .cacheability =
            (iree_hal_host_cacheability_t)payload.preferred_host_cacheability,
        .atomics =
            {
                .device_scope_32 =
                    payload.preferred_host_atomic_device_scope_32,
                .device_scope_64 =
                    payload.preferred_host_atomic_device_scope_64,
                .system_scope_32 =
                    payload.preferred_host_atomic_system_scope_32,
                .system_scope_64 =
                    payload.preferred_host_atomic_system_scope_64,
            },
    };
    options.trace_name =
        iree_make_string_view((const char*)trace_name_data, trace_name_length);
    status = iree_hal_slab_pool_create(executor->device_group, scope, &options,
                                       executor->host_allocator, &pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_replay_object_entry_t entry = {.value.pool = pool};
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_POOL, entry);
    pool = NULL;
  }
  iree_hal_pool_release(pool);
  iree_allocator_free(executor->host_allocator, import_exports);
  iree_allocator_free(executor->host_allocator, imports);
  iree_allocator_free(executor->host_allocator, owned_exports);
  iree_allocator_free(executor->host_allocator, families);
  return status;
}

static iree_status_t iree_hal_replay_executor_pool_allocate_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_pool_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_ALLOCATE_BUFFER,
      sizeof(iree_hal_replay_pool_allocate_buffer_payload_t)));
  iree_hal_replay_pool_allocate_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.allocation_size == 0 ||
                    payload.replay_allocation_size < payload.allocation_size)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay pool allocation geometry is invalid");
  }
  iree_hal_replay_object_entry_t* pool_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_POOL,
      &pool_entry));
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_pool_allocate_buffer(
      pool_entry->value.pool,
      (iree_hal_pool_reservation_request_t){
          .allocation_size = payload.replay_allocation_size,
          .min_alignment = payload.min_alignment,
      },
      iree_infinite_timeout(), &buffer));
  iree_hal_replay_object_entry_t entry = {.value.buffer = buffer};
  iree_status_t status =
      iree_hal_replay_executor_store(executor, record->header.related_object_id,
                                     IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, entry);
  if (iree_status_is_ok(status)) {
    buffer = NULL;
  }
  iree_hal_buffer_release(buffer);
  return status;
}

static iree_status_t iree_hal_replay_executor_pool_trim(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_fixed_pool_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_TRIM,
      sizeof(iree_hal_replay_pool_trim_payload_t)));
  iree_hal_replay_pool_trim_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(
          payload.reserved0 != 0 ||
          iree_any_bit_set(payload.flags, ~(IREE_HAL_POOL_TRIM_FLAG_ALL |
                                            IREE_HAL_POOL_TRIM_FLAG_EXCESS)))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay pool trim payload is invalid");
  }
  iree_hal_replay_object_entry_t* pool_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_POOL,
      &pool_entry));
  iree_hal_pool_trim(pool_entry->value.pool,
                     (iree_hal_pool_trim_flags_t)payload.flags,
                     payload.min_bytes_to_keep);
  return iree_ok_status();
}

static void iree_hal_replay_import_allocation_release(
    void* user_data, iree_hal_external_storage_class_t storage_class,
    iree_hal_external_storage_payload_t payload) {
  (void)storage_class;
  (void)payload;
  iree_hal_replay_import_allocation_t* allocation = user_data;
  iree_allocator_free_aligned(allocation->host_allocator, allocation);
}

static iree_status_t iree_hal_replay_executor_pool_import_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_IMPORT_BUFFER,
      sizeof(iree_hal_replay_pool_import_buffer_payload_t)));
  iree_hal_replay_pool_import_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t expected_length = 0;
  const bool has_valid_payload_length =
      payload.data_length <= IREE_HOST_SIZE_MAX &&
      iree_host_size_checked_add(sizeof(payload),
                                 (iree_host_size_t)payload.data_length,
                                 &expected_length) &&
      expected_length == record->payload.data_length;
  if (IREE_UNLIKELY(
          payload.reserved0 != 0 || payload.byte_length > IREE_HOST_SIZE_MAX ||
          payload.data_length != payload.byte_length ||
          payload.host_cacheability > IREE_HAL_HOST_CACHEABILITY_UNCACHED ||
          !has_valid_payload_length ||
          payload.logical_offset > payload.byte_length ||
          payload.logical_length >
              payload.byte_length - payload.logical_offset ||
          payload.maintenance_offset > payload.byte_length ||
          payload.maintenance_length >
              payload.byte_length - payload.maintenance_offset)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay pool import payload is invalid");
  }
  iree_hal_external_storage_class_t storage_class;
  IREE_RETURN_IF_ERROR(iree_hal_replay_decode_external_storage_class(
      payload.storage_class, &storage_class));
  if (IREE_UNLIKELY(storage_class.type !=
                    IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay pool import payload does not contain host storage");
  }
  iree_hal_replay_object_entry_t* pool_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_POOL,
      &pool_entry));
  iree_hal_pool_import_requirements_t import_requirements;
  if (IREE_UNLIKELY(!iree_hal_pool_query_import_requirements(
          pool_entry->value.pool, storage_class, &import_requirements))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay pool does not admit the captured external storage class");
  }
  if (IREE_UNLIKELY(import_requirements.base_alignment > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay pool import alignment is too large");
  }

  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          sizeof(iree_hal_replay_import_allocation_t),
          (iree_host_size_t)payload.data_length, &allocation_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay pool import allocation overflows");
  }
  iree_hal_replay_import_allocation_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      executor->host_allocator, allocation_size,
      (iree_host_size_t)import_requirements.base_alignment,
      offsetof(iree_hal_replay_import_allocation_t, data),
      (void**)&allocation));
  allocation->host_allocator = executor->host_allocator;
  memcpy(allocation->data, record->payload.data + sizeof(payload),
         (iree_host_size_t)payload.data_length);

  iree_hal_external_storage_t storage = {
      .storage_class = storage_class,
      .payload.host_pointer = allocation->data,
      .byte_length = payload.byte_length,
      .host_cacheability =
          (iree_hal_host_cacheability_t)payload.host_cacheability,
      .maintenance_range =
          {
              .byte_offset = payload.maintenance_offset,
              .byte_length = payload.maintenance_length,
          },
      .physical_backing_id =
          {
              .words = {payload.physical_backing_id[0],
                        payload.physical_backing_id[1]},
          },
      .release = iree_hal_replay_import_allocation_release,
      .release_user_data = allocation,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_hal_pool_import_buffer(pool_entry->value.pool, &storage,
                                  (iree_hal_buffer_range_t){
                                      .byte_offset = payload.logical_offset,
                                      .byte_length = payload.logical_length,
                                  },
                                  &buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_replay_object_entry_t entry = {.value.buffer = buffer};
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, entry);
    buffer = NULL;
  }
  iree_hal_buffer_release(buffer);
  iree_hal_external_storage_release(&storage);
  return status;
}

iree_status_t iree_hal_replay_executor_replay_pool_operation(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_POOL_CREATE_SLAB:
      return iree_hal_replay_executor_pool_create_slab(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_POOL_ALLOCATE_BUFFER:
      return iree_hal_replay_executor_pool_allocate_buffer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_POOL_TRIM:
      return iree_hal_replay_executor_pool_trim(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_POOL_IMPORT_BUFFER:
      return iree_hal_replay_executor_pool_import_buffer(executor, record);
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "replay pool operation %s is not implemented",
          iree_hal_replay_operation_code_string(record->header.operation_code));
  }
}
