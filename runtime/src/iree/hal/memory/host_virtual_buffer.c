// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/memory.h"
#include "iree/base/internal/shm.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/host_memory_backend.h"

typedef struct iree_hal_host_virtual_buffer_t {
  // Ordinary buffer resource owning the native address interval.
  iree_hal_virtual_buffer_t base;
  // Borrowed diagnostic sink owned by the creating device.
  iree_hal_device_event_sink_t event_sink;
  // Owned process address interval, initially inaccessible.
  iree_shm_reservation_t native;
  // Sticky native replacement failure, serialized by the caller's queue edges.
  bool failed;
  // Prepared CPU execution address shared by every admitted site.
  iree_hal_buffer_native_binding_t binding;
  // Virtual geometry and borrowed progress services for child pools.
  iree_hal_buffer_backing_facts_t backing;
  // Captured process page-table identity per admitted scope.
  uint32_t projections[];
} iree_hal_host_virtual_buffer_t;

static void iree_hal_host_virtual_buffer_destroy(iree_hal_buffer_t* base) {
  iree_hal_host_virtual_buffer_t* buffer =
      (iree_hal_host_virtual_buffer_t*)base;
  iree_status_t status = iree_shm_reservation_deinitialize(&buffer->native);
  if (!iree_status_is_ok(status)) {
    const iree_hal_device_driver_failure_event_t payload = {
        .record_length = sizeof(payload),
        .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
        .status_code = iree_status_code(status),
        .message = IREE_SV("shared virtual address release failed"),
    };
    iree_hal_device_event_t event = iree_hal_device_event_default();
    event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
    event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
    event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
    iree_hal_device_event_sink_publish(buffer->event_sink, &event);
    // The diagnostic sink consumes the teardown error. Unreleased native
    // ranges remain unavailable for reuse; final release certifies no removal.
    iree_status_free(status);
  }
  iree_hal_memory_contract_release(base->memory.contract);
  iree_allocator_free(buffer->base.host_allocator, buffer);
}

static iree_status_t iree_hal_host_virtual_buffer_detach(
    iree_hal_buffer_t* base, uint32_t scope_id) {
  iree_hal_host_virtual_buffer_t* buffer =
      (iree_hal_host_virtual_buffer_t*)base;
  iree_status_t status = iree_shm_reservation_detach(&buffer->native);
  buffer->failed |= !iree_status_is_ok(status);
  return status;
}

static iree_status_t iree_hal_host_virtual_buffer_update(
    iree_hal_buffer_t* base, uint32_t projection, iree_host_size_t update_count,
    const iree_hal_memory_mapping_update_t* updates) {
  iree_hal_host_virtual_buffer_t* reservation =
      (iree_hal_host_virtual_buffer_t*)base;
  if (reservation->failed) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "mapping updates are unavailable after native failure");
  }
  iree_shm_mapping_update_t inline_updates[16];
  iree_shm_mapping_update_t* native_updates = inline_updates;
  if (update_count > IREE_ARRAYSIZE(inline_updates)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        reservation->base.host_allocator,
        update_count * sizeof(*native_updates), (void**)&native_updates));
  }
  for (iree_host_size_t i = 0; i < update_count; ++i) {
    iree_shm_access_t access = 0;
    if (iree_any_bit_set(updates[i].access, IREE_HAL_MEMORY_ACCESS_READ)) {
      access |= IREE_SHM_ACCESS_READ;
    }
    if (iree_any_bit_set(updates[i].access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
      // CPU write mappings can also be readable. HAL permissions describe
      // valid application accesses, not a hardware isolation boundary.
      access |= IREE_SHM_ACCESS_READ | IREE_SHM_ACCESS_WRITE;
    }
    native_updates[i] = (iree_shm_mapping_update_t){
        .target_offset = (iree_host_size_t)updates[i].target_range.byte_offset,
        .byte_length = (iree_host_size_t)updates[i].target_range.byte_length,
        .source = access ? updates[i].source.mapping_source.descriptor : NULL,
        .source_offset =
            access ? (iree_host_size_t)updates[i].source.mapping_source.offset
                   : 0,
        .access = access,
    };
  }
  iree_status_t status = iree_shm_reservation_update(
      &reservation->native, update_count, native_updates);
  reservation->failed = !iree_status_is_ok(status);
  if (native_updates != inline_updates) {
    iree_allocator_free(reservation->base.host_allocator, native_updates);
  }
  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_host_virtual_buffer_vtable = {
    .destroy = iree_hal_host_virtual_buffer_destroy,
    .export_range = iree_hal_virtual_buffer_export_range,
    .map_range = iree_hal_virtual_buffer_map_range,
    .unmap_range = iree_hal_virtual_buffer_unmap_range,
    .invalidate_range = iree_hal_virtual_buffer_invalidate_range,
    .flush_range = iree_hal_virtual_buffer_flush_range,
    .query_virtual = iree_hal_virtual_buffer_query_root,
    .update_mappings = iree_hal_host_virtual_buffer_update,
    .detach_mappings = iree_hal_host_virtual_buffer_detach,
};

iree_status_t iree_hal_host_virtual_buffer_create(
    const iree_hal_host_memory_backend_t* backend,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_hal_memory_contract_t* contract,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  if (options->byte_length > IREE_HOST_SIZE_MAX ||
      options->min_alignment > IREE_HOST_SIZE_MAX ||
      options->mapping_granularity > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reservation exceeds host address space");
  }
  iree_hal_host_virtual_buffer_t* reservation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      sizeof(*reservation) + contract->scope_count * sizeof(uint32_t),
      (void**)&reservation));
  const iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
  iree_status_t status = iree_shm_reservation_initialize(
      (iree_host_size_t)options->byte_length,
      iree_max((iree_host_size_t)options->min_alignment, page_size),
      iree_max((iree_host_size_t)options->mapping_granularity, page_size),
      &reservation->native);
  if (iree_status_is_ok(status)) {
    reservation->event_sink = backend->event_sink;
    reservation->binding.host_pointer = reservation->native.base;
    reservation->backing = (iree_hal_buffer_backing_facts_t){
        .notification = backend->notification,
        .tracker = frontier_tracker,
        .maintenance_range = {.byte_length = options->byte_length},
        .allocation_alignment =
            iree_max(options->min_alignment, reservation->native.granularity),
        .maintenance_alignment = 1,
    };
    iree_hal_virtual_buffer_initialize(
        contract, &reservation->binding, &reservation->backing,
        IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE, options->byte_length,
        reservation->native.granularity, IREE_HAL_MEMORY_MAPPING_OPERATIONS_ALL,
        IREE_HAL_MEMORY_BACKEND_HOST, &iree_hal_host_virtual_buffer_vtable,
        host_allocator, &reservation->base);
    reservation->base.projection_count = 1;
    reservation->base.projections = reservation->projections;
    for (uint32_t i = 0; i < contract->scope_count; ++i) {
      reservation->projections[i] =
          contract->scopes[i].interfaces ? 0 : IREE_HAL_MEMORY_SCOPE_INVALID;
    }
    *out_buffer = &reservation->base.buffer;
  } else {
    iree_allocator_free(host_allocator, reservation);
  }
  return status;
}
