// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_staging_pool.h"

#include "iree/async/semaphore.h"
#include "iree/io/file_handle.h"

// Host allocation wrapped as a HAL memory file for local queue staging.
typedef struct iree_hal_remote_server_bulk_host_allocation_t {
  // Host allocator used to free this allocation.
  iree_allocator_t host_allocator;

  // Byte-addressable file contents.
  uint8_t data[];
} iree_hal_remote_server_bulk_host_allocation_t;

typedef uint8_t iree_hal_remote_server_bulk_staging_slot_flags_t;
enum iree_hal_remote_server_bulk_staging_slot_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_BULK_STAGING_SLOT_FLAG_IN_USE = 1u << 0,
};

struct iree_hal_remote_server_bulk_staging_slot_t {
  // Owning staging pool retained while callbacks may reference this slot.
  iree_hal_remote_server_bulk_staging_pool_t* pool;

  // Slot index within |pool->slots|.
  iree_host_size_t slot_index;

  // Next free slot index, or IREE_HOST_SIZE_MAX when acquired.
  iree_host_size_t next_free_slot;

  // Local device this slot's HAL file and semaphore are bound to.
  iree_hal_device_t* local_device;

  // Imported HAL memory file wrapping |file_handle| for |local_device|.
  iree_hal_file_t* file;

  // Local semaphore sequencing queue operations into |file|.
  iree_hal_semaphore_t* semaphore;

  // Host allocation file handle for |contents|.
  iree_io_file_handle_t* file_handle;

  // Host bytes exposed through |file_handle|.
  iree_byte_span_t contents;

  // Owner-managed storage bytes assigned to this slot.
  iree_byte_span_t user_storage;

  // Last payload value signaled on |semaphore|.
  uint64_t last_signal_value;

  // Payload value for the active callback timepoint.
  uint64_t callback_signal_value;

  // Timepoint registered on |semaphore| for an active staged operation.
  iree_async_semaphore_timepoint_t callback_timepoint;

  // Owner callback invoked when |callback_timepoint| fires.
  iree_hal_remote_server_bulk_staging_slot_callback_fn_t callback;

  // Owner callback user data.
  void* callback_user_data;

  // State bits from iree_hal_remote_server_bulk_staging_slot_flag_bits_e.
  iree_hal_remote_server_bulk_staging_slot_flags_t flags;
};

struct iree_hal_remote_server_bulk_staging_pool_t {
  // Reference count for pool lifetime management.
  iree_atomic_ref_count_t ref_count;

  // Host allocator used for pool and slot storage.
  iree_allocator_t host_allocator;

  // Number of entries in |slots|.
  iree_host_size_t slot_count;

  // Number of entries currently marked in use.
  iree_host_size_t acquired_count;

  // Head free slot index, or IREE_HOST_SIZE_MAX when exhausted.
  iree_host_size_t free_slot_head;

  // FAM: staging slots.
  iree_hal_remote_server_bulk_staging_slot_t slots[];
};

static void iree_hal_remote_server_bulk_host_allocation_release(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  (void)user_data;
  IREE_ASSERT(handle_primitive.type ==
              IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION);
  iree_byte_span_t host_allocation = handle_primitive.value.host_allocation;
  if (!host_allocation.data) return;
  iree_hal_remote_server_bulk_host_allocation_t* allocation =
      (iree_hal_remote_server_bulk_host_allocation_t*)(host_allocation.data -
                                                       sizeof(*allocation));
  iree_allocator_t host_allocator = allocation->host_allocator;
  iree_allocator_free(host_allocator, allocation);
}

static void iree_hal_remote_server_bulk_host_contents_free(
    iree_byte_span_t host_contents) {
  if (!host_contents.data) return;
  iree_hal_remote_server_bulk_host_allocation_t* allocation =
      (iree_hal_remote_server_bulk_host_allocation_t*)(host_contents.data -
                                                       sizeof(*allocation));
  iree_allocator_t host_allocator = allocation->host_allocator;
  iree_allocator_free(host_allocator, allocation);
}

static iree_status_t iree_hal_remote_server_bulk_host_contents_allocate(
    iree_host_size_t allocation_size, iree_allocator_t host_allocator,
    iree_byte_span_t* out_host_contents) {
  *out_host_contents = iree_byte_span_empty();

  iree_host_size_t total_size = 0;
  iree_host_size_t data_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_server_bulk_host_allocation_t), &total_size,
      IREE_STRUCT_FIELD(allocation_size, uint8_t, &data_offset));

  iree_hal_remote_server_bulk_host_allocation_t* allocation = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&allocation);
  }

  if (iree_status_is_ok(status)) {
    allocation->host_allocator = host_allocator;
    iree_byte_span_t host_contents = iree_make_byte_span(
        (uint8_t*)allocation + data_offset, allocation_size);
    *out_host_contents = host_contents;
  }
  return status;
}

static iree_status_t iree_hal_remote_server_bulk_host_allocation_create(
    iree_host_size_t allocation_size, iree_allocator_t host_allocator,
    iree_byte_span_t* out_host_contents,
    iree_io_file_handle_t** out_file_handle) {
  *out_file_handle = NULL;
  iree_status_t status = iree_hal_remote_server_bulk_host_contents_allocate(
      allocation_size, host_allocator, out_host_contents);

  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_hal_remote_server_bulk_host_allocation_release,
      .user_data = NULL,
  };
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        *out_host_contents, release_callback, host_allocator, out_file_handle);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_host_contents_free(*out_host_contents);
    *out_host_contents = iree_byte_span_empty();
  }
  return status;
}

static iree_status_t iree_hal_remote_server_bulk_staging_pool_resolve_options(
    const iree_hal_remote_server_bulk_staging_pool_options_t* options,
    iree_hal_remote_server_bulk_staging_pool_options_t* out_options,
    iree_host_size_t* out_user_storage_stride) {
  iree_hal_remote_server_bulk_staging_pool_options_t resolved_options =
      iree_hal_remote_server_bulk_staging_pool_options_default();
  if (options) resolved_options = *options;

  if (resolved_options.slot_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk staging pool requires at least one slot");
  }
  if (resolved_options.slot_count == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk staging pool slot count too large");
  }
  if (resolved_options.slot_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk staging pool requires non-empty slots");
  }

  if (resolved_options.user_storage_alignment == 0) {
    resolved_options.user_storage_alignment = iree_max_align_t;
  }
  if (!iree_host_size_is_valid_alignment(
          resolved_options.user_storage_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bulk staging user storage alignment must be a power of two: %" PRIhsz,
        resolved_options.user_storage_alignment);
  }

  iree_host_size_t user_storage_stride = 0;
  if (resolved_options.user_storage_size > 0 &&
      !iree_host_size_checked_align(resolved_options.user_storage_size,
                                    resolved_options.user_storage_alignment,
                                    &user_storage_stride)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk staging user storage stride overflow");
  }

  *out_options = resolved_options;
  *out_user_storage_stride = user_storage_stride;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_bulk_staging_pool_calculate_layout(
    const iree_hal_remote_server_bulk_staging_pool_options_t* options,
    iree_host_size_t user_storage_stride, iree_host_size_t* out_total_size,
    iree_host_size_t* out_user_storage_offset) {
  iree_host_size_t user_storage_alignment =
      user_storage_stride > 0 ? options->user_storage_alignment : 1;
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_server_bulk_staging_pool_t), out_total_size,
      IREE_STRUCT_FIELD_FAM(options->slot_count,
                            iree_hal_remote_server_bulk_staging_slot_t),
      IREE_STRUCT_ARRAY_FIELD_ALIGNED(options->slot_count, user_storage_stride,
                                      uint8_t, user_storage_alignment,
                                      out_user_storage_offset));
}

static void iree_hal_remote_server_bulk_staging_slot_unbind(
    iree_hal_remote_server_bulk_staging_slot_t* slot) {
  iree_hal_semaphore_release(slot->semaphore);
  slot->semaphore = NULL;
  iree_hal_file_release(slot->file);
  slot->file = NULL;
  iree_hal_device_release(slot->local_device);
  slot->local_device = NULL;
  slot->last_signal_value = 0;
}

static void iree_hal_remote_server_bulk_staging_slot_deinitialize(
    iree_hal_remote_server_bulk_staging_slot_t* slot) {
  iree_hal_remote_server_bulk_staging_slot_unbind(slot);
  iree_io_file_handle_release(slot->file_handle);
  memset(slot, 0, sizeof(*slot));
}

static void iree_hal_remote_server_bulk_staging_pool_destroy(
    iree_hal_remote_server_bulk_staging_pool_t* pool) {
  if (!pool) return;
  iree_allocator_t host_allocator = pool->host_allocator;
  for (iree_host_size_t i = 0; i < pool->slot_count; ++i) {
    iree_hal_remote_server_bulk_staging_slot_deinitialize(&pool->slots[i]);
  }
  iree_allocator_free(host_allocator, pool);
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_bulk_staging_pool_create(
    const iree_hal_remote_server_bulk_staging_pool_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_bulk_staging_pool_t** out_pool) {
  *out_pool = NULL;

  iree_hal_remote_server_bulk_staging_pool_options_t resolved_options;
  iree_host_size_t user_storage_stride = 0;
  iree_status_t status =
      iree_hal_remote_server_bulk_staging_pool_resolve_options(
          options, &resolved_options, &user_storage_stride);

  iree_host_size_t total_size = 0;
  iree_host_size_t user_storage_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_bulk_staging_pool_calculate_layout(
        &resolved_options, user_storage_stride, &total_size,
        &user_storage_offset);
  }

  iree_hal_remote_server_bulk_staging_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void**)&pool);
  }
  if (iree_status_is_ok(status)) {
    memset(pool, 0, total_size);
    iree_atomic_ref_count_init(&pool->ref_count);
    pool->host_allocator = host_allocator;
    pool->slot_count = resolved_options.slot_count;
    pool->free_slot_head = 0;
  }

  uint8_t* user_storage_base = pool && user_storage_stride > 0
                                   ? (uint8_t*)pool + user_storage_offset
                                   : NULL;
  for (iree_host_size_t i = 0;
       i < resolved_options.slot_count && iree_status_is_ok(status); ++i) {
    iree_hal_remote_server_bulk_staging_slot_t* slot = &pool->slots[i];
    slot->pool = pool;
    slot->slot_index = i;
    slot->next_free_slot =
        i + 1 < resolved_options.slot_count ? i + 1 : IREE_HOST_SIZE_MAX;
    if (user_storage_stride > 0) {
      slot->user_storage =
          iree_make_byte_span(user_storage_base + i * user_storage_stride,
                              resolved_options.user_storage_size);
    }
    status = iree_hal_remote_server_bulk_host_allocation_create(
        resolved_options.slot_length, host_allocator, &slot->contents,
        &slot->file_handle);
  }

  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    iree_hal_remote_server_bulk_staging_pool_destroy(pool);
  }
  return status;
}

IREE_API_EXPORT void iree_hal_remote_server_bulk_staging_pool_retain(
    iree_hal_remote_server_bulk_staging_pool_t* pool) {
  if (!pool) return;
  iree_atomic_ref_count_inc(&pool->ref_count);
}

IREE_API_EXPORT void iree_hal_remote_server_bulk_staging_pool_release(
    iree_hal_remote_server_bulk_staging_pool_t* pool) {
  if (pool && iree_atomic_ref_count_dec(&pool->ref_count) == 1) {
    iree_hal_remote_server_bulk_staging_pool_destroy(pool);
  }
}

IREE_API_EXPORT iree_host_size_t
iree_hal_remote_server_bulk_staging_pool_capacity(
    const iree_hal_remote_server_bulk_staging_pool_t* pool) {
  return pool ? pool->slot_count : 0;
}

IREE_API_EXPORT iree_host_size_t iree_hal_remote_server_bulk_staging_pool_count(
    const iree_hal_remote_server_bulk_staging_pool_t* pool) {
  return pool ? pool->acquired_count : 0;
}

static iree_status_t iree_hal_remote_server_bulk_staging_slot_bind(
    iree_hal_remote_server_bulk_staging_slot_t* slot,
    iree_hal_device_t* local_device) {
  if (slot->local_device == local_device && slot->file && slot->semaphore) {
    return iree_ok_status();
  }

  iree_hal_remote_server_bulk_staging_slot_unbind(slot);
  iree_hal_device_retain(local_device);
  slot->local_device = local_device;

  iree_status_t status = iree_hal_file_import(
      local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      slot->file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &slot->file);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        local_device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &slot->semaphore);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_server_bulk_staging_slot_unbind(slot);
  }
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_remote_server_bulk_staging_pool_try_acquire(
    iree_hal_remote_server_bulk_staging_pool_t* pool,
    iree_hal_device_t* local_device,
    iree_hal_remote_server_bulk_staging_slot_t** out_slot) {
  *out_slot = NULL;
  if (pool->free_slot_head == IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }

  iree_hal_remote_server_bulk_staging_slot_t* slot =
      &pool->slots[pool->free_slot_head];
  pool->free_slot_head = slot->next_free_slot;
  slot->next_free_slot = IREE_HOST_SIZE_MAX;

  iree_status_t status =
      iree_hal_remote_server_bulk_staging_slot_bind(slot, local_device);
  if (iree_status_is_ok(status)) {
    slot->flags |= IREE_HAL_REMOTE_SERVER_BULK_STAGING_SLOT_FLAG_IN_USE;
    ++pool->acquired_count;
    *out_slot = slot;
  } else {
    slot->next_free_slot = pool->free_slot_head;
    pool->free_slot_head = slot->slot_index;
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_server_bulk_staging_pool_acquire(
    iree_hal_remote_server_bulk_staging_pool_t* pool,
    iree_hal_device_t* local_device,
    iree_hal_remote_server_bulk_staging_slot_t** out_slot) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_bulk_staging_pool_try_acquire(
      pool, local_device, out_slot));
  if (*out_slot) return iree_ok_status();
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "no remote bulk staging slots available");
}

IREE_API_EXPORT void iree_hal_remote_server_bulk_staging_slot_release(
    iree_hal_remote_server_bulk_staging_slot_t* slot,
    uint64_t last_signal_value) {
  if (!slot) return;
  slot->last_signal_value = last_signal_value;
  slot->callback_signal_value = 0;
  memset(&slot->callback_timepoint, 0, sizeof(slot->callback_timepoint));
  slot->callback = NULL;
  slot->callback_user_data = NULL;
  if (iree_all_bits_set(slot->flags,
                        IREE_HAL_REMOTE_SERVER_BULK_STAGING_SLOT_FLAG_IN_USE)) {
    --slot->pool->acquired_count;
    slot->next_free_slot = slot->pool->free_slot_head;
    slot->pool->free_slot_head = slot->slot_index;
  }
  slot->flags &= ~IREE_HAL_REMOTE_SERVER_BULK_STAGING_SLOT_FLAG_IN_USE;
}

IREE_API_EXPORT iree_hal_remote_server_bulk_staging_pool_t*
iree_hal_remote_server_bulk_staging_slot_pool(
    const iree_hal_remote_server_bulk_staging_slot_t* slot) {
  return slot ? slot->pool : NULL;
}

IREE_API_EXPORT iree_hal_device_t*
iree_hal_remote_server_bulk_staging_slot_device(
    const iree_hal_remote_server_bulk_staging_slot_t* slot) {
  return slot ? slot->local_device : NULL;
}

IREE_API_EXPORT iree_hal_file_t* iree_hal_remote_server_bulk_staging_slot_file(
    const iree_hal_remote_server_bulk_staging_slot_t* slot) {
  return slot ? slot->file : NULL;
}

IREE_API_EXPORT iree_hal_semaphore_t*
iree_hal_remote_server_bulk_staging_slot_semaphore(
    const iree_hal_remote_server_bulk_staging_slot_t* slot) {
  return slot ? slot->semaphore : NULL;
}

IREE_API_EXPORT iree_byte_span_t
iree_hal_remote_server_bulk_staging_slot_contents(
    const iree_hal_remote_server_bulk_staging_slot_t* slot) {
  return slot ? slot->contents : iree_byte_span_empty();
}

IREE_API_EXPORT iree_byte_span_t
iree_hal_remote_server_bulk_staging_slot_user_storage(
    const iree_hal_remote_server_bulk_staging_slot_t* slot) {
  return slot ? slot->user_storage : iree_byte_span_empty();
}

IREE_API_EXPORT uint64_t
iree_hal_remote_server_bulk_staging_slot_last_signal_value(
    const iree_hal_remote_server_bulk_staging_slot_t* slot) {
  return slot ? slot->last_signal_value : 0;
}

IREE_API_EXPORT uint64_t
iree_hal_remote_server_bulk_staging_slot_next_signal_value(
    const iree_hal_remote_server_bulk_staging_slot_t* slot) {
  return iree_hal_remote_server_bulk_staging_slot_last_signal_value(slot) + 1;
}

static void iree_hal_remote_server_bulk_staging_slot_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_remote_server_bulk_staging_slot_t* slot =
      (iree_hal_remote_server_bulk_staging_slot_t*)user_data;
  iree_hal_remote_server_bulk_staging_pool_t* pool = slot->pool;
  iree_hal_remote_server_bulk_staging_slot_callback_fn_t callback =
      slot->callback;
  void* callback_user_data = slot->callback_user_data;
  const uint64_t signal_value = slot->callback_signal_value;
  callback(callback_user_data, slot, signal_value, status);
  iree_hal_remote_server_bulk_staging_pool_release(pool);
}

IREE_API_EXPORT iree_status_t
iree_hal_remote_server_bulk_staging_slot_acquire_timepoint(
    iree_hal_remote_server_bulk_staging_slot_t* slot, uint64_t signal_value,
    iree_hal_remote_server_bulk_staging_slot_callback_fn_t callback,
    void* user_data) {
  slot->callback_signal_value = signal_value;
  slot->callback = callback;
  slot->callback_user_data = user_data;
  slot->callback_timepoint.callback =
      iree_hal_remote_server_bulk_staging_slot_timepoint_callback;
  slot->callback_timepoint.user_data = slot;
  iree_hal_remote_server_bulk_staging_pool_retain(slot->pool);
  iree_status_t status = iree_async_semaphore_acquire_timepoint(
      (iree_async_semaphore_t*)slot->semaphore, signal_value,
      &slot->callback_timepoint);
  if (!iree_status_is_ok(status)) {
    slot->callback_signal_value = 0;
    memset(&slot->callback_timepoint, 0, sizeof(slot->callback_timepoint));
    slot->callback = NULL;
    slot->callback_user_data = NULL;
    iree_hal_remote_server_bulk_staging_pool_release(slot->pool);
  }
  return status;
}
