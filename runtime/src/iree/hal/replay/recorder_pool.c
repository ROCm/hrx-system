// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_pool.h"

#include <string.h>

#include "iree/hal/replay/recorder_buffer.h"
#include "iree/hal/replay/recorder_device.h"
#include "iree/hal/replay/recorder_record.h"

typedef struct iree_hal_replay_recorder_pool_t {
  // Common HAL pool state.
  iree_hal_pool_t base;
  // Host allocator used for wrapper metadata.
  iree_allocator_t host_allocator;
  // Shared recorder receiving captured pool operations.
  iree_hal_replay_recorder_t* recorder;
  // Public replacement group used to map placement devices.
  iree_hal_device_group_t* device_group;
  // Underlying pool receiving forwarded operations.
  iree_hal_pool_t* base_pool;
  // Session-local object ID assigned to this pool.
  iree_hal_replay_object_id_t pool_id;
} iree_hal_replay_recorder_pool_t;

static const iree_hal_pool_vtable_t iree_hal_replay_recorder_pool_vtable;

static iree_status_t iree_hal_replay_recorder_pool_allocate_array(
    iree_allocator_t host_allocator, iree_host_size_t element_count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (element_count == 0) {
    return iree_ok_status();
  }
  return iree_allocator_malloc_array(host_allocator, element_count,
                                     element_size, out_ptr);
}

static bool iree_hal_replay_recorder_pool_isa(iree_hal_pool_t* pool) {
  return iree_hal_resource_is(pool, &iree_hal_replay_recorder_pool_vtable);
}

static iree_hal_replay_recorder_pool_t* iree_hal_replay_recorder_pool_cast(
    iree_hal_pool_t* pool) {
  IREE_HAL_ASSERT_TYPE(pool, &iree_hal_replay_recorder_pool_vtable);
  return (iree_hal_replay_recorder_pool_t*)pool;
}

iree_hal_replay_object_id_t iree_hal_replay_recorder_find_pool_id(
    iree_hal_replay_recorder_t* recorder, iree_hal_pool_t* pool) {
  return pool ? iree_hal_replay_recorder_find_pool_contract_id(
                    recorder, pool->memory_contract)
              : IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

iree_hal_pool_t* iree_hal_replay_recorder_pool_base_or_self(
    iree_hal_pool_t* pool) {
  return iree_hal_replay_recorder_pool_isa(pool)
             ? iree_hal_replay_recorder_pool_cast(pool)->base_pool
             : pool;
}

static iree_status_t iree_hal_replay_recorder_pool_begin_operation(
    iree_hal_replay_recorder_pool_t* pool,
    iree_hal_replay_object_id_t related_object_id,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      pool->recorder, IREE_HAL_REPLAY_OBJECT_ID_NONE, pool->pool_id,
      related_object_id, IREE_HAL_REPLAY_OBJECT_TYPE_POOL, operation_code,
      payload_type, out_pending_record);
}

static iree_hal_device_t* iree_hal_replay_recorder_pool_placement_device(
    iree_hal_replay_recorder_pool_t* pool, iree_hal_buffer_t* base_buffer,
    iree_hal_replay_object_id_t* out_device_id) {
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(base_buffer);
  if (!placement.device) {
    *out_device_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
    return NULL;
  }
  return iree_hal_replay_device_group_find_replacement(
      pool->device_group, placement.device, out_device_id);
}

static void iree_hal_replay_recorder_pool_destroy(iree_hal_pool_t* base_pool) {
  iree_hal_replay_recorder_pool_t* pool =
      iree_hal_replay_recorder_pool_cast(base_pool);
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_hal_pool_release(pool->base_pool);
  iree_hal_device_group_release(pool->device_group);
  iree_hal_replay_recorder_release(pool->recorder);
  iree_hal_pool_deinitialize(base_pool);
  iree_allocator_free(host_allocator, pool);
}

static iree_status_t iree_hal_replay_recorder_pool_intercept_allocation(
    void* user_data, iree_hal_pool_t* source_pool,
    iree_hal_pool_reservation_request_t request, iree_timeout_t timeout,
    iree_hal_pool_allocate_buffer_fn_t next, iree_hal_buffer_t** out_buffer) {
  iree_hal_replay_recorder_pool_t* pool = user_data;
  iree_hal_replay_recorder_buffer_t* proxy_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_allocate_proxy(
      pool->host_allocator, &proxy_storage));

  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  iree_status_t status =
      iree_hal_replay_recorder_reserve_object_id(pool->recorder, &buffer_id);
  iree_hal_replay_pool_allocate_buffer_payload_t operation_payload = {
      .allocation_size = request.allocation_size,
      .min_alignment = request.min_alignment,
  };
  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_pool_begin_operation(
        pool, buffer_id, IREE_HAL_REPLAY_OPERATION_CODE_POOL_ALLOCATE_BUFFER,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_ALLOCATE_BUFFER, &pending_record);
  }

  iree_hal_buffer_t* native_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = source_pool == &pool->base
                 ? iree_hal_pool_allocate_buffer(pool->base_pool, request,
                                                 timeout, &native_buffer)
                 : next(source_pool, request, timeout, &native_buffer);
  }
  iree_hal_replay_object_id_t device_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  iree_hal_device_t* placement_device = NULL;
  if (iree_status_is_ok(status)) {
    operation_payload.replay_allocation_size =
        iree_hal_buffer_allocation_size(native_buffer);
    placement_device = iree_hal_replay_recorder_pool_placement_device(
        pool, native_buffer, &device_id);
    const iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(native_buffer);
    if (IREE_UNLIKELY(placement.device && !placement_device)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "pool returned a buffer placed on a device outside the wrapped "
          "group");
    }
  }

  iree_hal_replay_buffer_object_payload_t object_payload;
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_make_object_payload(native_buffer,
                                                        &object_payload);
  } else {
    memset(&object_payload, 0, sizeof(object_payload));
  }
  if (pending_record.recorder) {
    const iree_const_byte_span_t operation_iovec = iree_make_const_byte_span(
        &operation_payload, sizeof(operation_payload));
    const iree_const_byte_span_t object_iovec =
        iree_make_const_byte_span(&object_payload, sizeof(object_payload));
    status = iree_hal_replay_recorder_end_creation_operation(
        &pending_record, status, 1, &operation_iovec,
        IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, buffer_id,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT, 1, &object_iovec);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = iree_hal_replay_recorder_buffer_initialize_proxy(
        pool->recorder, device_id, buffer_id, IREE_HAL_REPLAY_OBJECT_ID_NONE,
        placement_device, native_buffer, pool->host_allocator, proxy_storage);
    proxy_storage = NULL;
  }
  iree_hal_replay_recorder_buffer_free_proxy(pool->host_allocator,
                                             proxy_storage);
  iree_hal_buffer_release(native_buffer);
  return status;
}

static iree_status_t iree_hal_replay_recorder_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_callback_t completion,
    iree_hal_pool_reserve_flags_t flags) {
  iree_hal_replay_recorder_pool_t* pool =
      iree_hal_replay_recorder_pool_cast(base_pool);
  return iree_hal_pool_acquire_reservations(
      pool->base_pool, request_count, requests, requester_frontier,
      out_reservations, out_infos, completion, flags);
}

static void iree_hal_replay_recorder_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_replay_recorder_pool_t* pool =
      iree_hal_replay_recorder_pool_cast(base_pool);
  iree_hal_pool_release_reservations(pool->base_pool, reservation_count,
                                     reservations, death_frontier);
}

static iree_status_t iree_hal_replay_recorder_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  iree_hal_replay_recorder_pool_t* pool =
      iree_hal_replay_recorder_pool_cast(base_pool);
  iree_hal_buffer_t** native_buffers = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      pool->host_allocator, reservation_count, sizeof(*native_buffers),
      (void**)&native_buffers));
  memset(native_buffers, 0, reservation_count * sizeof(*native_buffers));
  iree_hal_replay_recorder_buffer_t** proxy_storage = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      pool->host_allocator, reservation_count, sizeof(*proxy_storage),
      (void**)&proxy_storage);
  if (iree_status_is_ok(status)) {
    memset(proxy_storage, 0, reservation_count * sizeof(*proxy_storage));
  }
  for (iree_host_size_t i = 0;
       i < reservation_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_replay_recorder_buffer_allocate_proxy(
        pool->host_allocator, &proxy_storage[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_materialize_reservations(
        pool->base_pool, reservation_count, requests, reservations, flags,
        native_buffers);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < reservation_count && iree_status_is_ok(status); ++i) {
      iree_hal_replay_object_id_t device_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
      iree_hal_device_t* placement_device =
          iree_hal_replay_recorder_pool_placement_device(
              pool, native_buffers[i], &device_id);
      const iree_hal_buffer_placement_t placement =
          iree_hal_buffer_allocation_placement(native_buffers[i]);
      if (IREE_UNLIKELY(placement.device && !placement_device)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "pool returned a buffer placed on a device outside the wrapped "
            "group");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      iree_hal_replay_object_id_t device_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
      iree_hal_device_t* placement_device =
          iree_hal_replay_recorder_pool_placement_device(
              pool, native_buffers[i], &device_id);
      out_buffers[i] = iree_hal_replay_recorder_buffer_initialize_proxy(
          pool->recorder, device_id, IREE_HAL_REPLAY_OBJECT_ID_NONE,
          IREE_HAL_REPLAY_OBJECT_ID_NONE, placement_device, native_buffers[i],
          pool->host_allocator, proxy_storage[i]);
      proxy_storage[i] = NULL;
    }
  }
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    if (proxy_storage) {
      iree_hal_replay_recorder_buffer_free_proxy(pool->host_allocator,
                                                 proxy_storage[i]);
    }
    iree_hal_buffer_release(native_buffers[i]);
  }
  iree_allocator_free(pool->host_allocator, proxy_storage);
  iree_allocator_free(pool->host_allocator, native_buffers);
  return status;
}

static void iree_hal_replay_recorder_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_replay_recorder_pool_t* pool =
      (const iree_hal_replay_recorder_pool_t*)base_pool;
  iree_hal_pool_query_capabilities(pool->base_pool, out_capabilities);
}

static iree_status_t iree_hal_replay_recorder_pool_validate_asan(
    const iree_hal_pool_t* base_pool,
    const iree_hal_asan_pool_options_t* options) {
  const iree_hal_replay_recorder_pool_t* pool =
      (const iree_hal_replay_recorder_pool_t*)base_pool;
  return iree_hal_pool_validate_asan_options(pool->base_pool, options);
}

static void iree_hal_replay_recorder_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  const iree_hal_replay_recorder_pool_t* pool =
      (const iree_hal_replay_recorder_pool_t*)base_pool;
  iree_hal_pool_query_stats(pool->base_pool, out_stats);
}

static void iree_hal_replay_recorder_pool_trim(
    iree_hal_pool_t* base_pool, iree_hal_pool_trim_flags_t flags,
    iree_device_size_t min_bytes_to_keep) {
  iree_hal_replay_recorder_pool_t* pool =
      iree_hal_replay_recorder_pool_cast(base_pool);
  const iree_hal_replay_pool_trim_payload_t payload = {
      .flags = flags,
      .min_bytes_to_keep = min_bytes_to_keep,
  };
  iree_hal_replay_pending_record_t pending_record = {0};
  iree_hal_replay_recorder_try_begin_operation(
      pool->recorder, IREE_HAL_REPLAY_OBJECT_ID_NONE, pool->pool_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_POOL,
      IREE_HAL_REPLAY_OPERATION_CODE_POOL_TRIM,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_TRIM, &pending_record);
  iree_hal_pool_trim(pool->base_pool, flags, min_bytes_to_keep);
  if (pending_record.recorder) {
    const iree_const_byte_span_t iovec =
        iree_make_const_byte_span(&payload, sizeof(payload));
    iree_hal_replay_recorder_end_void_operation_with_payload(&pending_record, 1,
                                                             &iovec);
  }
}

static void iree_hal_replay_recorder_pool_advise_asan_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_asan_range_advice_flags_t flags) {
  iree_hal_replay_recorder_pool_t* pool =
      iree_hal_replay_recorder_pool_cast(base_pool);
  iree_hal_pool_advise_asan_reservations(pool->base_pool, reservation_count,
                                         reservations, flags);
}

static void iree_hal_replay_make_storage_class_payload(
    iree_hal_external_storage_class_t storage_class,
    iree_hal_replay_external_storage_class_payload_t* out_payload) {
  *out_payload = (iree_hal_replay_external_storage_class_payload_t){
      .type = storage_class.type,
      .provenance = {storage_class.provenance[0], storage_class.provenance[1]},
  };
}

static iree_status_t iree_hal_replay_recorder_pool_import_buffer(
    iree_hal_pool_t* base_pool, iree_hal_external_storage_t* inout_storage,
    iree_hal_buffer_range_t logical_range, iree_hal_buffer_t** out_buffer) {
  iree_hal_replay_recorder_pool_t* pool =
      iree_hal_replay_recorder_pool_cast(base_pool);
  iree_hal_replay_recorder_buffer_t* proxy_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_allocate_proxy(
      pool->host_allocator, &proxy_storage));
  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  iree_status_t status =
      iree_hal_replay_recorder_reserve_object_id(pool->recorder, &buffer_id);

  iree_hal_replay_pool_import_buffer_payload_t operation_payload;
  memset(&operation_payload, 0, sizeof(operation_payload));
  iree_hal_replay_make_storage_class_payload(inout_storage->storage_class,
                                             &operation_payload.storage_class);
  operation_payload.byte_length = inout_storage->byte_length;
  operation_payload.logical_offset = logical_range.byte_offset;
  operation_payload.logical_length = logical_range.byte_length;
  operation_payload.maintenance_offset =
      inout_storage->maintenance_range.byte_offset;
  operation_payload.maintenance_length =
      inout_storage->maintenance_range.byte_length;
  operation_payload.physical_backing_id[0] =
      inout_storage->physical_backing_id.words[0];
  operation_payload.physical_backing_id[1] =
      inout_storage->physical_backing_id.words[1];
  operation_payload.host_cacheability = inout_storage->host_cacheability;
  const bool can_record = inout_storage->storage_class.type ==
                              IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER &&
                          inout_storage->byte_length <= IREE_HOST_SIZE_MAX &&
                          inout_storage->payload.host_pointer;
  operation_payload.data_length = can_record ? inout_storage->byte_length : 0;

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_pool_begin_operation(
        pool, buffer_id, IREE_HAL_REPLAY_OPERATION_CODE_POOL_IMPORT_BUFFER,
        can_record ? IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_IMPORT_BUFFER
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
        &pending_record);
  }
  if (iree_status_is_ok(status) && !can_record) {
    iree_hal_replay_recorder_mark_unsupported(&pending_record);
  }

  iree_const_byte_span_t operation_iovecs[2];
  iree_host_size_t operation_iovec_count = 0;
  if (can_record) {
    operation_iovecs[operation_iovec_count++] = iree_make_const_byte_span(
        &operation_payload, sizeof(operation_payload));
    operation_iovecs[operation_iovec_count++] =
        iree_make_const_byte_span(inout_storage->payload.host_pointer,
                                  (iree_host_size_t)inout_storage->byte_length);
  }

  iree_hal_buffer_t* native_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_import_buffer(pool->base_pool, inout_storage,
                                         logical_range, &native_buffer);
  }
  iree_hal_replay_object_id_t device_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  iree_hal_device_t* placement_device = NULL;
  if (iree_status_is_ok(status)) {
    placement_device = iree_hal_replay_recorder_pool_placement_device(
        pool, native_buffer, &device_id);
    const iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(native_buffer);
    if (IREE_UNLIKELY(placement.device && !placement_device)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "pool returned a buffer placed on a device outside the wrapped "
          "group");
    }
  }
  iree_hal_replay_buffer_object_payload_t object_payload;
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_make_object_payload(native_buffer,
                                                        &object_payload);
  } else {
    memset(&object_payload, 0, sizeof(object_payload));
  }
  if (pending_record.recorder) {
    const iree_const_byte_span_t object_iovec =
        iree_make_const_byte_span(&object_payload, sizeof(object_payload));
    status = iree_hal_replay_recorder_end_creation_operation(
        &pending_record, status, operation_iovec_count, operation_iovecs,
        IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, buffer_id,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT, 1, &object_iovec);
  }
  if (iree_status_is_ok(status)) {
    *out_buffer = iree_hal_replay_recorder_buffer_initialize_proxy(
        pool->recorder, device_id, buffer_id, IREE_HAL_REPLAY_OBJECT_ID_NONE,
        placement_device, native_buffer, pool->host_allocator, proxy_storage);
    proxy_storage = NULL;
  }
  iree_hal_replay_recorder_buffer_free_proxy(pool->host_allocator,
                                             proxy_storage);
  iree_hal_buffer_release(native_buffer);
  return status;
}

static bool iree_hal_replay_recorder_pool_query_import_requirements(
    const iree_hal_pool_t* base_pool,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  const iree_hal_replay_recorder_pool_t* pool =
      (const iree_hal_replay_recorder_pool_t*)base_pool;
  return iree_hal_pool_query_import_requirements(pool->base_pool, storage_class,
                                                 out_requirements);
}

static const iree_hal_pool_vtable_t iree_hal_replay_recorder_pool_vtable = {
    .destroy = iree_hal_replay_recorder_pool_destroy,
    .acquire_reservations = iree_hal_replay_recorder_pool_acquire_reservations,
    .release_reservations = iree_hal_replay_recorder_pool_release_reservations,
    .materialize_reservations =
        iree_hal_replay_recorder_pool_materialize_reservations,
    .query_capabilities = iree_hal_replay_recorder_pool_query_capabilities,
    .validate_asan = iree_hal_replay_recorder_pool_validate_asan,
    .query_stats = iree_hal_replay_recorder_pool_query_stats,
    .trim = iree_hal_replay_recorder_pool_trim,
    .advise_asan_reservations =
        iree_hal_replay_recorder_pool_advise_asan_reservations,
    .import_buffer = iree_hal_replay_recorder_pool_import_buffer,
    .query_import_requirements =
        iree_hal_replay_recorder_pool_query_import_requirements,
};

void iree_hal_replay_recorder_pool_capture_families(
    const iree_hal_pool_scope_t* scope,
    iree_hal_replay_pool_family_access_payload_t* family_payloads) {
  for (iree_host_size_t i = 0; i < scope->family_count; ++i) {
    const iree_hal_pool_family_access_t* access = &scope->families[i];
    const iree_hal_device_topology_info_t* topology_info =
        iree_hal_device_topology_info(
            iree_hal_queue_family_device(access->family));
    family_payloads[i] = (iree_hal_replay_pool_family_access_payload_t){
        .device_ordinal = topology_info->topology_index,
        .family_ordinal = iree_hal_queue_family_ordinal(access->family),
        .usage = access->usage,
        .interfaces = access->interfaces,
        .requirements = access->requirements,
        .atomic_device_scope_32 = access->atomic_operations.device_scope_32,
        .atomic_device_scope_64 = access->atomic_operations.device_scope_64,
        .atomic_system_scope_32 = access->atomic_operations.system_scope_32,
        .atomic_system_scope_64 = access->atomic_operations.system_scope_64,
    };
  }
}

iree_status_t iree_hal_replay_recorder_create_slab_pool(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_slab_pool_create_callback_t next,
    iree_hal_device_group_t* request_group, const iree_hal_pool_scope_t* scope,
    const iree_hal_slab_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(request_group);
  IREE_ASSERT_ARGUMENT(scope);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;

  iree_hal_replay_recorder_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));

  iree_hal_replay_pool_family_access_payload_t* family_payloads = NULL;
  iree_status_t status = iree_hal_replay_recorder_pool_allocate_array(
      host_allocator, scope->family_count, sizeof(*family_payloads),
      (void**)&family_payloads);
  uint64_t import_export_count = 0;
  for (iree_host_size_t i = 0;
       i < options->imports.class_count && iree_status_is_ok(status); ++i) {
    if (IREE_UNLIKELY(!iree_checked_add_u64(
            import_export_count,
            options->imports.classes[i].exports.class_count,
            &import_export_count))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "pool import export count overflow");
    }
  }
  iree_hal_replay_external_storage_class_payload_t* owned_exports = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_pool_allocate_array(
        host_allocator, options->owned.exports.class_count,
        sizeof(*owned_exports), (void**)&owned_exports);
  }
  iree_hal_replay_pool_import_class_payload_t* import_payloads = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_pool_allocate_array(
        host_allocator, options->imports.class_count, sizeof(*import_payloads),
        (void**)&import_payloads);
  }
  iree_hal_replay_external_storage_class_payload_t* import_exports = NULL;
  if (iree_status_is_ok(status) && import_export_count > IREE_HOST_SIZE_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "pool import export payload is too large");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_pool_allocate_array(
        host_allocator, (iree_host_size_t)import_export_count,
        sizeof(*import_exports), (void**)&import_exports);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_pool_capture_families(scope, family_payloads);
  }
  for (iree_host_size_t i = 0;
       i < options->owned.exports.class_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_replay_make_storage_class_payload(
        options->owned.exports.classes[i], &owned_exports[i]);
  }
  iree_host_size_t import_export_offset = 0;
  for (iree_host_size_t i = 0;
       i < options->imports.class_count && iree_status_is_ok(status); ++i) {
    const iree_hal_pool_import_class_t* import_class =
        &options->imports.classes[i];
    iree_hal_replay_make_storage_class_payload(
        import_class->storage_class, &import_payloads[i].storage_class);
    import_payloads[i].export_offset = import_export_offset;
    import_payloads[i].export_count = import_class->exports.class_count;
    for (iree_host_size_t j = 0; j < import_class->exports.class_count; ++j) {
      iree_hal_replay_make_storage_class_payload(
          import_class->exports.classes[j],
          &import_exports[import_export_offset++]);
    }
  }

  iree_hal_replay_object_id_t pool_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_reserve_object_id(recorder, &pool_id);
  }
  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        recorder, IREE_HAL_REPLAY_OBJECT_ID_NONE,
        IREE_HAL_REPLAY_OBJECT_ID_NONE, pool_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
        IREE_HAL_REPLAY_OPERATION_CODE_POOL_CREATE_SLAB,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_POOL_CREATE_SLAB, &pending_record);
  }

  iree_hal_pool_t* native_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = next.fn(next.user_data, request_group, scope, options,
                     host_allocator, &native_pool);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_initialize(
        &iree_hal_replay_recorder_pool_vtable, native_pool->memory_contract,
        native_pool->notification, native_pool->wait_sources,
        native_pool->frontier_tracker, host_allocator, &pool->base);
  }
  if (iree_status_is_ok(status)) {
    pool->base.epoch_query = native_pool->epoch_query;
    pool->base.asan_enabled = native_pool->asan_enabled;
    pool->base.allocation_interceptor =
        (iree_hal_pool_allocation_interceptor_t){
            .fn = iree_hal_replay_recorder_pool_intercept_allocation,
            .user_data = pool,
        };
    pool->host_allocator = host_allocator;
    pool->recorder = recorder;
    iree_hal_replay_recorder_retain(pool->recorder);
    pool->device_group = request_group;
    iree_hal_device_group_retain(pool->device_group);
    pool->base_pool = native_pool;
    iree_hal_pool_retain(pool->base_pool);
    pool->pool_id = pool_id;
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_register_pool_contract(
        &pending_record, pool->base.memory_contract, pool_id);
  }

  const iree_hal_replay_pool_create_slab_payload_t payload = {
      .family_count = scope->family_count,
      .owned_export_count = options->owned.exports.class_count,
      .import_count = options->imports.class_count,
      .import_export_count = import_export_count,
      .trace_name_length = options->trace_name.size,
      .placement_mode = options->placement.mode,
      .placement_node = options->placement.node,
      .owned_mode = options->owned.mode,
      .host_access = scope->host.access,
      .host_mapping_modes = scope->host.modes,
      .host_cacheability = scope->host.cacheability,
      .preferred_host_access = options->preferences.host.access,
      .preferred_host_mapping_modes = options->preferences.host.modes,
      .preferred_host_cacheability = options->preferences.host.cacheability,
  };
  if (pending_record.recorder) {
    const iree_const_byte_span_t operation_iovecs[] = {
        iree_make_const_byte_span(&payload, sizeof(payload)),
        iree_make_const_byte_span(
            family_payloads, scope->family_count * sizeof(*family_payloads)),
        iree_make_const_byte_span(
            owned_exports,
            options->owned.exports.class_count * sizeof(*owned_exports)),
        iree_make_const_byte_span(
            import_payloads,
            options->imports.class_count * sizeof(*import_payloads)),
        iree_make_const_byte_span(
            import_exports,
            (iree_host_size_t)import_export_count * sizeof(*import_exports)),
        iree_make_const_byte_span(options->trace_name.data,
                                  options->trace_name.size),
    };
    status = iree_hal_replay_recorder_end_creation_operation(
        &pending_record, status, IREE_ARRAYSIZE(operation_iovecs),
        operation_iovecs, IREE_HAL_REPLAY_OBJECT_TYPE_POOL, pool_id,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, 0, NULL);
  }

  if (iree_status_is_ok(status)) {
    *out_pool = &pool->base;
    pool = NULL;
  }
  if (pool) {
    if (pool->base.resource.vtable) {
      iree_hal_pool_release(&pool->base);
    } else {
      iree_allocator_free(host_allocator, pool);
    }
  }
  iree_hal_pool_release(native_pool);
  iree_allocator_free(host_allocator, import_exports);
  iree_allocator_free(host_allocator, import_payloads);
  iree_allocator_free(host_allocator, owned_exports);
  iree_allocator_free(host_allocator, family_payloads);
  return status;
}
