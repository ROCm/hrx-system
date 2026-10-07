// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/transient_buffer.h"

#include "iree/base/threading/mutex.h"

typedef enum iree_hal_task_transient_buffer_deallocation_state_e {
  IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE = 0,
  IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_PENDING = 1,
  IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_COMPLETE = 2,
} iree_hal_task_transient_buffer_deallocation_state_t;

// Vtable dispatch for forwarding to the committed buffer's implementation.
// Equivalent to IREE_HAL_VTABLE_DISPATCH from detail.h but accessible from
// this driver-private implementation (detail.h is module-private to HAL core).
static inline const iree_hal_buffer_vtable_t*
iree_hal_task_transient_buffer_committed_vtable(iree_hal_buffer_t* buffer) {
  return (const iree_hal_buffer_vtable_t*)((const iree_hal_resource_t*)buffer)
      ->vtable;
}

struct iree_hal_task_transient_buffer_t {
  // Base HAL buffer resource exposed to callers.
  iree_hal_buffer_t base;

  // Host allocator used for wrapper storage and teardown.
  iree_allocator_t host_allocator;

  // Stable nonzero id used to join profile rows for this wrapper lifetime.
  uint64_t profile_id;

  // Guards all staged backing, committed backing, and reservation state.
  iree_slim_mutex_t mutex;

  // Materialized backing buffer staged for a future commit. Retained by the
  // wrapper while non-NULL.
  iree_hal_buffer_t* staged_backing;

  // Materialized backing buffer visible to accessors after commit.
  iree_hal_buffer_t* committed_backing;

  // Borrowed pool selected for this logical allocation epoch.
  iree_hal_pool_t* source_pool;

  // Optional queue-allocation reservation owned by this wrapper while armed.
  iree_hal_pool_reservation_t reservation;

  // True while the wrapper owns |reservation|.
  bool reservation_armed;

  // State controlling exclusive queue deallocation capture.
  iree_hal_task_transient_buffer_deallocation_state_t deallocation_state;

  // Source-sized native table shared by subspans made before commitment. The
  // caller orders native reads after commit and before decommit. Trailing
  // native facts borrow the retained backing's allocation epoch.
  iree_hal_buffer_native_binding_t bindings[];
};

static const iree_hal_buffer_vtable_t iree_hal_task_transient_buffer_vtable;

static iree_hal_task_transient_buffer_t* iree_hal_task_transient_buffer_cast(
    iree_hal_buffer_t* buffer) {
  return (iree_hal_task_transient_buffer_t*)buffer;
}

static const iree_hal_buffer_binding_layout_t*
iree_hal_task_transient_buffer_binding_layout(const iree_hal_pool_t* pool) {
  return pool->memory_contract ? &pool->memory_contract->binding_layout
                               : iree_hal_heap_buffer_binding_layout();
}

static iree_status_t iree_hal_task_transient_buffer_retain_host_backing(
    iree_hal_task_transient_buffer_t* buffer,
    iree_hal_buffer_t** out_backing_buffer) {
  iree_slim_mutex_lock(&buffer->mutex);
  iree_hal_buffer_t* backing_buffer = buffer->committed_backing;
  if (IREE_UNLIKELY(!backing_buffer)) {
    iree_slim_mutex_unlock(&buffer->mutex);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has not been committed; ensure the alloca signal "
        "semaphores are satisfied before accessing it");
  }
  iree_hal_buffer_retain(backing_buffer);
  iree_slim_mutex_unlock(&buffer->mutex);
  *out_backing_buffer = backing_buffer;
  return iree_ok_status();
}

iree_status_t iree_hal_task_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_hal_pool_t* source_pool, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(source_pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  if (IREE_UNLIKELY(byte_length > allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "transient buffer byte length (%" PRIu64
                            ") exceeds allocation size (%" PRIu64 ")",
                            (uint64_t)byte_length, (uint64_t)allocation_size);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_buffer_binding_layout_t* binding_layout =
      iree_hal_task_transient_buffer_binding_layout(source_pool);
  iree_hal_task_transient_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                sizeof(*buffer) + binding_layout->byte_length,
                                (void**)&buffer));

  iree_hal_buffer_initialize(
      placement, /*allocated_buffer=*/&buffer->base, allocation_size,
      /*byte_offset=*/0, byte_length, params.type, params.access, params.usage,
      &iree_hal_task_transient_buffer_vtable, &buffer->base);
  buffer->base.memory.bindings = buffer->bindings;
  buffer->base.memory.contract = source_pool->memory_contract;
  buffer->base.host_binding_index = binding_layout->host_binding_index;
  buffer->host_allocator = host_allocator;
  buffer->profile_id = iree_hal_buffer_allocation_next_id();
  iree_slim_mutex_initialize(&buffer->mutex);
  buffer->staged_backing = NULL;
  buffer->committed_backing = NULL;
  buffer->source_pool = source_pool;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
  buffer->reservation_armed = false;
  buffer->deallocation_state =
      IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE;

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_task_transient_buffer_isa(const iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is(&buffer->resource,
                              &iree_hal_task_transient_buffer_vtable);
}

static iree_hal_buffer_allocation_profile_t
iree_hal_task_transient_buffer_allocation_profile(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  return (iree_hal_buffer_allocation_profile_t){
      .id = buffer->profile_id,
  };
}

void iree_hal_task_transient_buffer_attach_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->source_pool == pool);
  IREE_ASSERT_FALSE(buffer->reservation_armed);
  buffer->reservation = *reservation;
  buffer->reservation_armed = true;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_task_transient_buffer_stage_backing(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* backing) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(backing);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->staged_backing == NULL);
  IREE_ASSERT_TRUE(buffer->committed_backing == NULL);
  iree_hal_buffer_retain(backing);
  buffer->staged_backing = backing;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_task_transient_buffer_commit(iree_hal_buffer_t* base_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->staged_backing != NULL);
  IREE_ASSERT_TRUE(buffer->committed_backing == NULL);
  buffer->committed_backing = buffer->staged_backing;
  if (buffer->source_pool->memory_contract) {
    iree_hal_buffer_memory_copy_bindings(
        &buffer->committed_backing->memory,
        &buffer->source_pool->memory_contract->binding_layout,
        buffer->bindings);
  } else {
    // Unscoped materialization qualifies only host access, whose slot may
    // differ from the Task wrapper's single host representation.
    buffer->bindings[0] = iree_hal_buffer_native_binding(
        buffer->committed_backing,
        (iree_hal_buffer_native_binding_slot_t){
            .index = buffer->committed_backing->host_binding_index,
            .type = IREE_HAL_BUFFER_INTERFACE_HOST,
        });
  }
  iree_slim_mutex_unlock(&buffer->mutex);
}

static void iree_hal_task_transient_buffer_decommit(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  iree_hal_buffer_t* staged_backing = buffer->staged_backing;
  buffer->staged_backing = NULL;
  buffer->committed_backing = NULL;
  memset(buffer->bindings, 0,
         iree_hal_task_transient_buffer_binding_layout(buffer->source_pool)
             ->byte_length);
  iree_slim_mutex_unlock(&buffer->mutex);
  iree_hal_buffer_release(staged_backing);
}

bool iree_hal_task_transient_buffer_query_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  const bool has_reservation = buffer->reservation_armed;
  if (has_reservation) {
    if (out_pool) {
      *out_pool = buffer->source_pool;
    }
    if (out_reservation) {
      *out_reservation = buffer->reservation;
    }
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return has_reservation;
}

static iree_status_t iree_hal_task_transient_buffer_begin_dealloca(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t** out_pool) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  iree_status_t status = iree_ok_status();
  if (buffer->deallocation_state !=
      IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has already been queued for deallocation");
  } else {
    buffer->deallocation_state =
        IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_PENDING;
    *out_pool = buffer->source_pool;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return status;
}

static void iree_hal_task_transient_buffer_abort_dealloca(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  if (buffer->deallocation_state ==
      IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_PENDING) {
    buffer->deallocation_state =
        IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_IDLE;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
}

static void iree_hal_task_transient_buffer_take_dealloca_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->deallocation_state ==
                   IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_PENDING);
  IREE_ASSERT_TRUE(buffer->reservation_armed);
  *out_pool = buffer->source_pool;
  *out_reservation = buffer->reservation;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
  buffer->reservation_armed = false;
  buffer->deallocation_state =
      IREE_HAL_TASK_TRANSIENT_BUFFER_DEALLOCATION_STATE_COMPLETE;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_task_transient_buffer_release_reservation(
    iree_hal_buffer_t* base_buffer,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_pool_t* pool = NULL;
  iree_hal_pool_reservation_t reservation;
  iree_slim_mutex_lock(&buffer->mutex);
  const bool had_reservation = buffer->reservation_armed;
  if (had_reservation) {
    pool = buffer->source_pool;
    reservation = buffer->reservation;
    memset(&buffer->reservation, 0, sizeof(buffer->reservation));
    buffer->reservation_armed = false;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  if (had_reservation) {
    if (!death_frontier || death_frontier->entry_count == 0) {
      iree_hal_pool_advise_asan_reservations(
          pool, 1, &reservation, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
    }
    iree_hal_pool_release_reservations(pool, 1, &reservation, death_frontier);
  }
}

static void iree_hal_task_transient_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_transient_buffer_decommit(base_buffer);
  iree_hal_task_transient_buffer_release_reservation(base_buffer,
                                                     /*death_frontier=*/NULL);

  iree_slim_mutex_deinitialize(&buffer->mutex);
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_task_transient_buffer_export_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* out_external_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_task_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_task_transient_buffer_committed_vtable(committed)->export_range(
          committed, iree_hal_buffer_byte_offset(committed) + local_byte_offset,
          local_byte_length, requested_type, requested_flags,
          out_external_buffer);
  iree_hal_buffer_release(committed);
  return status;
}

static iree_status_t iree_hal_task_transient_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_hal_buffer_map_flags_t flags,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_task_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_task_transient_buffer_committed_vtable(committed)->map_range(
          committed, mapping_mode, memory_access, flags,
          iree_hal_buffer_byte_offset(committed) + local_byte_offset,
          local_byte_length, mapping);
  if (iree_status_is_ok(status)) {
    if (mapping->impl.is_persistent) {
      iree_hal_buffer_release(committed);
    } else {
      iree_hal_buffer_t* mapped_buffer = mapping->buffer;
      // Scoped maps own their mapped storage until unmap. Transfer that mapping
      // ownership from the transient wrapper to the committed backing buffer so
      // a queue-ordered decommit cannot invalidate the unmap path.
      mapping->buffer = committed;
      mapping->impl.byte_offset =
          iree_hal_buffer_byte_offset(committed) + local_byte_offset;
      iree_hal_buffer_release(mapped_buffer);
    }
  } else {
    iree_hal_buffer_release(committed);
  }
  return status;
}

static iree_status_t iree_hal_task_transient_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_task_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_task_transient_buffer_committed_vtable(committed)->unmap_range(
          committed, iree_hal_buffer_byte_offset(committed) + local_byte_offset,
          local_byte_length, mapping);
  iree_hal_buffer_release(committed);
  return status;
}

static iree_status_t iree_hal_task_transient_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_task_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_task_transient_buffer_committed_vtable(committed)
          ->invalidate_range(
              committed,
              iree_hal_buffer_byte_offset(committed) + local_byte_offset,
              local_byte_length);
  iree_hal_buffer_release(committed);
  return status;
}

static iree_status_t iree_hal_task_transient_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_task_transient_buffer_retain_host_backing(buffer, &committed));
  iree_status_t status =
      iree_hal_task_transient_buffer_committed_vtable(committed)->flush_range(
          committed, iree_hal_buffer_byte_offset(committed) + local_byte_offset,
          local_byte_length);
  iree_hal_buffer_release(committed);
  return status;
}

static iree_hal_buffer_memory_view_t
iree_hal_task_transient_buffer_query_memory(
    const iree_hal_buffer_t* base_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      (iree_hal_task_transient_buffer_t*)base_buffer;
  iree_hal_buffer_memory_view_t view = buffer->base.memory;
  iree_slim_mutex_lock(&buffer->mutex);
  if (buffer->committed_backing) {
    view = iree_hal_buffer_memory_view(buffer->committed_backing);
    view.bindings = buffer->base.memory.bindings;
    view.binding_offset = 0;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return view;
}

static const iree_hal_buffer_allocation_vtable_t
    iree_hal_task_transient_buffer_allocation_vtable = {
        .profile = iree_hal_task_transient_buffer_allocation_profile,
        .begin_dealloca = iree_hal_task_transient_buffer_begin_dealloca,
        .abort_dealloca = iree_hal_task_transient_buffer_abort_dealloca,
        .take_dealloca_reservation =
            iree_hal_task_transient_buffer_take_dealloca_reservation,
        .decommit = iree_hal_task_transient_buffer_decommit,
};

static const iree_hal_buffer_vtable_t iree_hal_task_transient_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_task_transient_buffer_destroy,
    .export_range = iree_hal_task_transient_buffer_export_range,
    .map_range = iree_hal_task_transient_buffer_map_range,
    .unmap_range = iree_hal_task_transient_buffer_unmap_range,
    .invalidate_range = iree_hal_task_transient_buffer_invalidate_range,
    .flush_range = iree_hal_task_transient_buffer_flush_range,
    .query_memory = iree_hal_task_transient_buffer_query_memory,
    .allocation = &iree_hal_task_transient_buffer_allocation_vtable,
};
