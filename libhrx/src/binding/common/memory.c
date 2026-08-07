// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/memory.h"

#include <stdio.h>

#include "common/direct_transfer.h"
#include "common/graph.h"
#include "common/internal.h"
#include "common/stream.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/call_once.h"
#include "iree/hal/buffer_transfer.h"

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_streaming_command_buffer_barrier(
    iree_hal_command_buffer_t* command_buffer) {
  static const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  return iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, NULL);
}

static iree_status_t iree_hal_streaming_command_buffer_host_to_transfer_barrier(
    iree_hal_command_buffer_t* command_buffer) {
  static const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_HOST_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
  };
  return iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_HOST,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      1, &memory_barrier, 0, NULL);
}

typedef struct iree_hal_streaming_host_memcpy_callback_data_t {
  void* dst;
  const void* src;
  iree_device_size_t count;
} iree_hal_streaming_host_memcpy_callback_data_t;

static void iree_hal_streaming_host_memcpy_callback(void* user_data) {
  iree_hal_streaming_host_memcpy_callback_data_t* callback_data =
      (iree_hal_streaming_host_memcpy_callback_data_t*)user_data;
  memcpy(callback_data->dst, callback_data->src, callback_data->count);
}

static void iree_hal_streaming_buffer_free(iree_hal_streaming_buffer_t* buffer);

static void iree_hal_streaming_buffer_activate_pool_allocation(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer || !buffer->allocation_pool || buffer->is_pool_allocation_live) {
    return;
  }
  hrx_mem_pool_record_logical_allocation(buffer->allocation_pool,
                                         (size_t)buffer->size);
  buffer->is_pool_allocation_live = true;
}

static void iree_hal_streaming_buffer_release_pool_allocation(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer || !buffer->allocation_pool || !buffer->is_pool_allocation_live) {
    return;
  }
  hrx_mem_pool_record_logical_free(buffer->allocation_pool,
                                   (size_t)buffer->size);
  buffer->is_pool_allocation_live = false;
}

static void iree_hal_streaming_memory_account_device_free(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer || !buffer->context || !buffer->context->device_entry ||
      !iree_any_bit_set((iree_hal_memory_type_t)buffer->memory_type,
                        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    return;
  }
  iree_hal_streaming_device_t* device = buffer->context->device_entry;
  const uint64_t size = buffer->size;
  const uint64_t total_memory = device->total_memory;
  uint64_t observed =
      iree_atomic_load(&device->free_memory, iree_memory_order_relaxed);
  uint64_t desired = 0;
  do {
    desired = size >= total_memory || observed >= total_memory - size
                  ? total_memory
                  : observed + size;
  } while (!iree_atomic_compare_exchange_weak(
      &device->free_memory, &observed, desired, iree_memory_order_relaxed,
      iree_memory_order_relaxed));
}

static void iree_hal_streaming_memory_account_device_allocation(
    iree_hal_streaming_device_t* device, iree_device_size_t size) {
  if (!device) return;
  uint64_t observed =
      iree_atomic_load(&device->free_memory, iree_memory_order_relaxed);
  uint64_t desired = 0;
  do {
    desired = observed > size ? observed - size : 0;
  } while (!iree_atomic_compare_exchange_weak(
      &device->free_memory, &observed, desired, iree_memory_order_relaxed,
      iree_memory_order_relaxed));
}

static void iree_hal_streaming_buffer_set_context(
    iree_hal_streaming_buffer_t* buffer, iree_hal_streaming_context_t* context,
    iree_hal_streaming_buffer_context_ownership_t ownership) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(context);
  buffer->context = context;
  buffer->context_ownership = ownership;
  if (ownership == IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED) {
    iree_hal_streaming_context_retain(context);
  }
}

static void iree_hal_streaming_buffer_release_context(
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  iree_hal_streaming_context_t* context = buffer->context;
  iree_hal_streaming_buffer_context_ownership_t ownership =
      buffer->context_ownership;
  buffer->context = NULL;
  buffer->context_ownership = IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED;
  if (ownership == IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED) {
    IREE_ASSERT_ARGUMENT(context);
    iree_hal_streaming_context_release(context);
  }
}

static iree_status_t iree_hal_streaming_buffer_begin_allocation_lifetime(
    iree_hal_streaming_buffer_t* buffer) {
  static iree_atomic_uint64_t next_allocation_id = IREE_ATOMIC_VAR_INIT(1);
  const uint64_t allocation_id =
      iree_atomic_fetch_add(&next_allocation_id, 1, iree_memory_order_relaxed);
  if (IREE_UNLIKELY(allocation_id > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "process exhausted 32-bit allocation identifiers");
  }

  buffer->allocation_id = (uint32_t)allocation_id;
  iree_atomic_store(&buffer->synchronous_memory_operations, 0,
                    iree_memory_order_relaxed);
  return iree_ok_status();
}

// Wraps an HRX buffer in a stream buffer and caches exported pointer metadata.
static iree_status_t iree_hal_streaming_buffer_wrap_hrx_buffer(
    iree_hal_streaming_context_t* context, hrx_buffer_t hrx_buf,
    int memory_type, void* imported_host_ptr, hrx_mem_pool_t allocation_pool,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_wrapper) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(hrx_buf);
  IREE_ASSERT_ARGUMENT(out_wrapper);
  *out_wrapper = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(context->host_allocator, sizeof(*wrapper),
                                (void**)&wrapper));
  memset(wrapper, 0, sizeof(*wrapper));

  // Initialize wrapper.
  wrapper->hrx_buf = hrx_buf;
  hrx_buffer_retain(wrapper->hrx_buf);
  wrapper->buffer = hrx_buf->hal_buffer;
  iree_hal_streaming_buffer_set_context(wrapper, context, context_ownership);
  wrapper->allocation_pool = allocation_pool;
  if (wrapper->allocation_pool) {
    hrx_mem_pool_retain(wrapper->allocation_pool);
  }
  wrapper->is_pool_allocation_live = false;
  wrapper->memory_type = wrapper->buffer
                             ? (int)iree_hal_buffer_memory_type(wrapper->buffer)
                             : memory_type;
  wrapper->host_register_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  wrapper->imported_host_allocation = imported_host_ptr != NULL;
  wrapper->is_managed = false;
  wrapper->synchronous_memory_operations = IREE_ATOMIC_VAR_INIT(0);
  wrapper->owns_host_ptr = false;
  wrapper->has_host_mapping = false;
  memset(&wrapper->host_mapping, 0, sizeof(wrapper->host_mapping));
  wrapper->managed_page_count = 0;
  wrapper->managed_read_mostly_pages = NULL;
  wrapper->managed_preferred_locations = NULL;
  wrapper->managed_accessed_by_device_masks = NULL;
  wrapper->managed_last_prefetch_locations = NULL;
  wrapper->managed_coherency_modes = NULL;
  iree_slim_mutex_initialize(&wrapper->context_import_mutex);
  wrapper->context_imports = NULL;
  wrapper->ipc_handle = NULL;
  wrapper->size = hrx_buf->size;
  wrapper->logical_size = wrapper->size;
  iree_hal_streaming_buffer_activate_pool_allocation(wrapper);

  // Initialize unified memory attributes.
  wrapper->read_mostly_hint = false;
  wrapper->preferred_location = -2;  // Unspecified initially.
  wrapper->accessed_by_device_mask = 0;
  wrapper->last_prefetch_location = -2;  // Never prefetched.
  wrapper->coherency_mode = 0;           // Fine-grain by default.

  iree_hal_external_buffer_t external_ptr;
  iree_status_t status = iree_ok_status();
  bool have_device_ptr = false;
  bool have_host_ptr = false;

  // Try to export as device allocation (works for device-local memory
  // and mapped host memory).
  if (wrapper->buffer) {
    iree_status_t device_status = iree_hal_allocator_export_buffer(
        context->device_allocator, wrapper->buffer,
        IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
        IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_ptr);
    if (iree_status_is_ok(device_status)) {
      wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)
                                external_ptr.handle.device_allocation.ptr;
      have_device_ptr = true;
    } else {
      iree_status_ignore(device_status);
    }
  }

  // For host-local memory, also export as host allocation.
  // This is needed for hipHostMalloc which returns host pointers.
  if (wrapper->buffer &&
      (wrapper->memory_type & IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    iree_status_t host_status = iree_hal_allocator_export_buffer(
        context->device_allocator, wrapper->buffer,
        IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_ptr);
    if (iree_status_is_ok(host_status)) {
      wrapper->host_ptr = (void*)external_ptr.handle.host_allocation.ptr;
      have_host_ptr = true;
      // For host-local memory, use host_ptr as device_ptr if we don't have one.
      if (!have_device_ptr) {
        wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)wrapper->host_ptr;
        have_device_ptr = true;
      }
    } else {
      iree_status_ignore(host_status);
    }
  }
  if (imported_host_ptr) {
    wrapper->host_ptr = imported_host_ptr;
    have_host_ptr = true;
  }
  if (wrapper->buffer && !have_host_ptr &&
      (wrapper->memory_type & IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    iree_status_t map_status = iree_hal_buffer_map_range(
        wrapper->buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        IREE_HAL_MEMORY_ACCESS_ALL, 0, wrapper->size, &wrapper->host_mapping);
    if (iree_status_is_ok(map_status)) {
      wrapper->host_ptr = wrapper->host_mapping.contents.data;
      wrapper->has_host_mapping = true;
      have_host_ptr = true;
    } else {
      iree_status_ignore(map_status);
    }
  }

  // We need at least a device pointer for the buffer table.
  // For remote HAL buffers the allocator may not support export_buffer;
  // generate a synthetic device pointer so the buffer table can still map
  // this wrapper.
  if (!have_device_ptr && imported_host_ptr) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "registered host allocation did not export a device-visible pointer");
  } else if (!have_device_ptr) {
    static iree_atomic_uint64_t g_next_synthetic =
        IREE_ATOMIC_VAR_INIT(0xDEAD000000000000ULL);
    iree_device_size_t aligned_size = 0;
    if (IREE_UNLIKELY(!iree_device_size_checked_mul_add(wrapper->size, 1, 255,
                                                        &aligned_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "buffer size overflows synthetic alignment");
    } else {
      aligned_size &= ~(iree_device_size_t)255;
      uint64_t synthetic = iree_atomic_fetch_add(
          &g_next_synthetic, aligned_size, iree_memory_order_relaxed);
      wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)synthetic;
      have_device_ptr = true;
    }
  }
  (void)have_host_ptr;

  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_buffer_begin_allocation_lifetime(wrapper);
  }

  if (iree_status_is_ok(status)) {
    // Register buffer in context's mapping table.
    status = HRX_CALL(hrx_buffer_table_insert(
        &context->buffer_table, wrapper->device_ptr, wrapper->host_ptr,
        wrapper->size, wrapper->hrx_buf, wrapper));
  }

  if (iree_status_is_ok(status)) {
    *out_wrapper = wrapper;
  } else {
    iree_hal_streaming_buffer_free(wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Wraps a HAL buffer in a stream buffer and caches information.
static iree_status_t iree_hal_streaming_buffer_wrap(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    int memory_type, void* imported_host_ptr, hrx_mem_pool_t allocation_pool,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_wrapper) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_wrapper);
  *out_wrapper = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  hrx_buffer_t hrx_buf = NULL;
  hrx_device_t hrx_dev =
      context->device_entry ? context->device_entry->hrx_device : NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hrx_buffer_create_from_hal(
              buffer, hrx_dev, (hrx_memory_type_t)memory_type,
              (size_t)iree_hal_buffer_byte_length(buffer), NULL, &hrx_buf));

  iree_status_t status = iree_hal_streaming_buffer_wrap_hrx_buffer(
      context, hrx_buf, memory_type, imported_host_ptr, allocation_pool,
      context_ownership, out_wrapper);
  hrx_buffer_release(hrx_buf);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Frees a buffer wrapper and releases the underlying HRX buffer.
static void iree_hal_streaming_buffer_free(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_allocator_t host_allocator = buffer->context->host_allocator;
  void* owned_host_ptr = NULL;
  if (buffer->owns_host_ptr) {
    owned_host_ptr = buffer->host_ptr;
    buffer->owns_host_ptr = false;
  }
  iree_allocator_free(host_allocator, buffer->managed_read_mostly_pages);
  buffer->managed_read_mostly_pages = NULL;
  iree_allocator_free(host_allocator, buffer->managed_preferred_locations);
  buffer->managed_preferred_locations = NULL;
  iree_allocator_free(host_allocator, buffer->managed_accessed_by_device_masks);
  buffer->managed_accessed_by_device_masks = NULL;
  iree_allocator_free(host_allocator, buffer->managed_last_prefetch_locations);
  buffer->managed_last_prefetch_locations = NULL;
  iree_allocator_free(host_allocator, buffer->managed_coherency_modes);
  buffer->managed_coherency_modes = NULL;
  buffer->managed_page_count = 0;
  iree_slim_mutex_lock(&buffer->context_import_mutex);
  iree_hal_streaming_context_import_t* context_imports =
      buffer->context_imports;
  buffer->context_imports = NULL;
  iree_slim_mutex_unlock(&buffer->context_import_mutex);
  while (context_imports) {
    iree_hal_streaming_context_import_t* import = context_imports;
    context_imports = import->next;
    iree_hal_buffer_release(import->buffer);
    iree_hal_streaming_context_release(import->context);
    iree_allocator_free(host_allocator, import);
  }
  iree_slim_mutex_deinitialize(&buffer->context_import_mutex);
  if (buffer->has_host_mapping) {
    iree_status_ignore(iree_hal_buffer_unmap_range(&buffer->host_mapping));
    memset(&buffer->host_mapping, 0, sizeof(buffer->host_mapping));
    buffer->has_host_mapping = false;
  }
  iree_hal_streaming_buffer_release_pool_allocation(buffer);
  if (buffer->hrx_buf) {
    hrx_buffer_release(buffer->hrx_buf);
    buffer->hrx_buf = NULL;
    buffer->buffer = NULL;
  }
  hrx_mem_pool_release(buffer->allocation_pool);
  buffer->allocation_pool = NULL;
  iree_allocator_free_aligned(host_allocator, owned_host_ptr);
  if (owned_host_ptr == buffer->host_ptr) {
    buffer->host_ptr = NULL;
  }
  iree_hal_streaming_buffer_release_context(buffer);
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_streaming_memory_wrap_buffer(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  return iree_hal_streaming_buffer_wrap(
      context, buffer, (int)iree_hal_buffer_memory_type(buffer),
      /*imported_host_ptr=*/NULL, /*allocation_pool=*/NULL, context_ownership,
      out_buffer);
}

iree_status_t iree_hal_streaming_memory_wrap_virtual_reservation(
    iree_hal_streaming_context_t* context, hrx_buffer_t virtual_buffer,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  return iree_hal_streaming_buffer_wrap_hrx_buffer(
      context, virtual_buffer, HRX_MEMORY_TYPE_DEVICE_LOCAL,
      /*imported_host_ptr=*/NULL, /*allocation_pool=*/NULL,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED, out_buffer);
}

void iree_hal_streaming_memory_release_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) return;
  hrx_buffer_table_remove(&buffer->context->buffer_table, buffer->device_ptr);
  iree_hal_streaming_buffer_free(buffer);
}

void iree_hal_streaming_memory_prepare_virtual_reservation_release(
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(buffer->hrx_buf);
  hrx_buffer_t virtual_buffer = buffer->hrx_buf;
  buffer->hrx_buf = NULL;
  buffer->buffer = NULL;
  hrx_buffer_release(virtual_buffer);
}

void iree_hal_streaming_memory_restore_virtual_reservation(
    iree_hal_streaming_buffer_t* buffer, hrx_buffer_t virtual_buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(!buffer->hrx_buf);
  buffer->hrx_buf = virtual_buffer;
  hrx_buffer_retain(virtual_buffer);
  buffer->buffer = virtual_buffer->hal_buffer;
}

static void iree_hal_streaming_temporary_host_buffer_free(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) return;
  hrx_buffer_table_remove(&context->buffer_table, buffer->device_ptr);
  iree_hal_streaming_buffer_free(buffer);
}

void iree_hal_streaming_memory_release_pageable_staging(
    iree_hal_streaming_context_t* context) {
  if (!context) return;
  if (context->pageable_h2d_staging_buffer) {
    iree_hal_streaming_buffer_t* buffer = context->pageable_h2d_staging_buffer;
    context->pageable_h2d_staging_buffer = NULL;
    context->pageable_h2d_staging_size = 0;
    hrx_buffer_table_remove(&context->buffer_table, buffer->device_ptr);
    iree_hal_streaming_buffer_free(buffer);
  }
}

static iree_status_t iree_hal_streaming_buffer_ref_validate_range(
    const iree_hal_streaming_buffer_ref_t* ref, iree_device_size_t size) {
  if (!ref->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer reference is empty");
  }
  if (ref->offset > ref->buffer->size ||
      size > ref->buffer->size - ref->offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer reference range exceeds allocation");
  }
  return iree_ok_status();
}

iree_hal_streaming_deviceptr_t iree_hal_streaming_buffer_device_pointer(
    iree_hal_streaming_buffer_t* buffer) {
  return buffer ? buffer->device_ptr : 0;
}

iree_status_t iree_hal_streaming_memory_lookup(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));
  size_t offset = 0;
  // Hot path: explicit kernel parameter resolution and memory operations use
  // this for pointer-table lookups. Most lookups miss, so handle the miss
  // directly without allocating an error message.
  hrx_status_t hs =
      hrx_buffer_table_find(&context->buffer_table, device_ptr, NULL, &offset,
                            (void**)&out_ref->buffer);
  if (!hrx_status_is_ok(hs)) {
    iree_status_code_t code = (iree_status_code_t)hrx_status_code(hs);
    hrx_status_ignore(hs);
    return iree_status_from_code(code);
  }
  out_ref->offset = (iree_device_size_t)offset;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_lookup_range(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));
  size_t offset = 0;
  IREE_RETURN_IF_ERROR(HRX_CALL(hrx_buffer_table_find_range(
      &context->buffer_table, device_ptr, (size_t)size, NULL, &offset,
      (void**)&out_ref->buffer)));
  out_ref->offset = (iree_device_size_t)offset;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_lookup_range_across_contexts(
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_context_t** out_context,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_ASSERT_ARGUMENT(out_ref);
  *out_context = NULL;
  memset(out_ref, 0, sizeof(*out_ref));

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  bool found = false;
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    iree_hal_streaming_buffer_ref_t candidate_ref;
    iree_status_t status = iree_hal_streaming_memory_lookup_range(
        context, device_ptr, size, &candidate_ref);
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_context_retain(context);
      *out_context = context;
      *out_ref = candidate_ref;
      found = true;
      break;
    }
    iree_status_ignore(status);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  return found ? iree_ok_status()
               : iree_status_from_code(IREE_STATUS_NOT_FOUND);
}

iree_status_t iree_hal_streaming_memory_allocate_device(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  if (iree_any_bit_set(flags, IREE_HAL_STREAMING_MEMORY_FLAG_UNCACHED)) {
    memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED;
  }
  iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 64,
  };

  // Allocate HAL buffer.
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(context->device_allocator, params,
                                             size, &buffer));

  // Wrap in stream buffer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap(
      context, buffer, (int)memory_type, /*imported_host_ptr=*/NULL,
      /*allocation_pool=*/NULL, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
      &wrapper);

  // Release our reference (wrapper holds its own).
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
    iree_hal_streaming_memory_account_device_allocation(context->device_entry,
                                                        size);
  } else {
    iree_hal_streaming_buffer_free(wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_allocate_device_from_pool(
    iree_hal_streaming_context_t* context, hrx_mem_pool_t pool,
    iree_device_size_t size, iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_any_bit_set(flags, IREE_HAL_STREAMING_MEMORY_FLAG_UNCACHED)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "memory-pool allocations cannot satisfy uncached device memory");
  }
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 64,
  };
  hrx_buffer_params_t hrx_params = {
      .type = (hrx_memory_type_t)params.type,
      .access = (hrx_memory_access_t)params.access,
      .usage = (hrx_buffer_usage_t)params.usage,
      .queue_affinity = (hrx_queue_affinity_t)params.queue_affinity,
  };

  hrx_buffer_t hrx_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, HRX_CALL(hrx_mem_pool_allocate_buffer(pool, hrx_params, size,
                                                &hrx_buffer)));

  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap_hrx_buffer(
      context, hrx_buffer, (int)params.type, /*imported_host_ptr=*/NULL, pool,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED, &wrapper);
  hrx_buffer_release(hrx_buffer);

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
    iree_hal_streaming_memory_account_device_allocation(context->device_entry,
                                                        size);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

struct iree_hal_streaming_deferred_device_free_t {
  // Next pending async free in the owning context's reuse registry.
  iree_hal_streaming_deferred_device_free_t* next;
  // Context owning the registry and allocation bookkeeping.
  iree_hal_streaming_context_t* owner_context;
  // Stable identifier of the stream that orders this logical free.
  unsigned long long source_stream_id;
  // Timeline value reached after this free's host completion callback.
  uint64_t completion_value;
  // Detached allocation wrapper to release on the stream timeline.
  iree_hal_streaming_buffer_t* buffer;
  // True while this operation is visible in |owner_context|'s registry.
  bool is_registered;
  // True when the host call ran after all stream prerequisites completed.
  bool is_ready;
  // True after the queue has released its terminal resource reference.
  bool is_terminal;
  // True when a later stream-ordered allocation adopted |buffer|.
  bool is_reused;
  // Reuse policy captured when the free is enqueued.
  bool allow_opportunistic;
};

typedef struct iree_hal_streaming_pending_free_terminal_t {
  // Resource retained by the accepted host call until callback completion or
  // cancellation. It transitively owns the operation's context retain.
  iree_hal_resource_t resource;
  // Context-owned operation notified when the queued call becomes terminal.
  iree_hal_streaming_deferred_device_free_t* free_op;
} iree_hal_streaming_pending_free_terminal_t;

static void iree_hal_streaming_pending_free_terminal_destroy(
    iree_hal_resource_t* resource);

static const iree_hal_resource_vtable_t
    iree_hal_streaming_pending_free_terminal_vtable = {
        .destroy = iree_hal_streaming_pending_free_terminal_destroy,
};

static void iree_hal_streaming_pending_free_destroy(
    iree_hal_streaming_deferred_device_free_t* free_op) {
  iree_allocator_free(iree_allocator_system(), free_op);
}

static iree_status_t iree_hal_streaming_buffer_free_and_trim_pool(
    iree_hal_streaming_buffer_t* buffer, bool trim_to_release_threshold) {
  // Pool-backed buffers retain their pool. Preserve it across buffer
  // destruction so that any now-unused backing storage can be released.
  hrx_mem_pool_t allocation_pool = buffer->allocation_pool;
  if (allocation_pool) hrx_mem_pool_retain(allocation_pool);
  iree_hal_streaming_buffer_free(buffer);

  iree_status_t status = iree_ok_status();
  if (allocation_pool && trim_to_release_threshold) {
    status = HRX_CALL(hrx_mem_pool_release_unused(allocation_pool));
  }
  hrx_mem_pool_release(allocation_pool);
  return status;
}

static iree_status_t iree_hal_streaming_pending_free_release_buffer(
    iree_hal_streaming_deferred_device_free_t* free_op,
    bool trim_to_release_threshold) {
  iree_status_t status = iree_hal_streaming_buffer_free_and_trim_pool(
      free_op->buffer, trim_to_release_threshold);
  iree_hal_streaming_pending_free_destroy(free_op);
  return status;
}

static void iree_hal_streaming_pending_free_remove_locked(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deferred_device_free_t* free_op) {
  iree_hal_streaming_deferred_device_free_t** current =
      &context->pending_free_head;
  while (*current && *current != free_op) {
    current = &(*current)->next;
  }
  IREE_ASSERT(*current == free_op);
  *current = free_op->next;
  free_op->next = NULL;
  free_op->is_registered = false;
}

static void iree_hal_streaming_pending_free_terminal_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_streaming_pending_free_terminal_t* terminal =
      (iree_hal_streaming_pending_free_terminal_t*)resource;
  iree_hal_streaming_deferred_device_free_t* free_op = terminal->free_op;
  iree_hal_streaming_context_t* context = free_op->owner_context;
  iree_hal_streaming_buffer_t* buffer = NULL;
  bool destroy_free_op = false;

  iree_slim_mutex_lock(&context->pending_free_mutex);
  free_op->is_terminal = true;
  if (free_op->is_reused) {
    destroy_free_op = true;
  } else if (!free_op->is_ready || free_op->allow_opportunistic) {
    if (free_op->is_registered) {
      iree_hal_streaming_pending_free_remove_locked(context, free_op);
    }
    buffer = free_op->buffer;
    free_op->buffer = NULL;
    destroy_free_op = true;
  }
  iree_slim_mutex_unlock(&context->pending_free_mutex);

  if (buffer) {
    iree_status_t status = iree_hal_streaming_buffer_free_and_trim_pool(
        buffer, /*trim_to_release_threshold=*/true);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
    }
  }
  if (destroy_free_op) {
    iree_hal_streaming_pending_free_destroy(free_op);
  }
  iree_allocator_free(iree_allocator_system(), terminal);
  iree_hal_streaming_context_release(context);
}

static void iree_hal_streaming_pending_free_register(
    iree_hal_streaming_deferred_device_free_t* free_op) {
  iree_hal_streaming_context_t* context = free_op->owner_context;
  iree_slim_mutex_lock(&context->pending_free_mutex);
  free_op->next = context->pending_free_head;
  context->pending_free_head = free_op;
  free_op->is_registered = true;
  iree_slim_mutex_unlock(&context->pending_free_mutex);
}

static iree_status_t iree_hal_streaming_memory_try_reuse_pending_free(
    iree_hal_streaming_context_t* context, hrx_mem_pool_t pool,
    iree_device_size_t size, iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_t** out_buffer) {
  *out_buffer = NULL;

  bool destroy_free_op = false;
  iree_slim_mutex_lock(&context->pending_free_mutex);
  iree_hal_streaming_deferred_device_free_t** current =
      &context->pending_free_head;
  while (*current) {
    iree_hal_streaming_deferred_device_free_t* free_op = *current;
    iree_hal_streaming_buffer_t* buffer = free_op->buffer;
    bool can_reuse = buffer->allocation_pool == pool && buffer->size == size;
    if (can_reuse && free_op->source_stream_id != stream->stream_id) {
      uint64_t follow_event_dependencies = 0;
      uint64_t allow_opportunistic = 0;
      const bool has_reuse_policy =
          hrx_status_is_ok(hrx_mem_pool_get_attribute(
              pool, HRX_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
              &follow_event_dependencies)) &&
          hrx_status_is_ok(hrx_mem_pool_get_attribute(
              pool, HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
              &allow_opportunistic));
      can_reuse =
          has_reuse_policy &&
          ((follow_event_dependencies != 0 && free_op->completion_value != 0 &&
            iree_hal_streaming_stream_has_memory_reuse_dependency(
                stream, free_op->source_stream_id,
                free_op->completion_value)) ||
           (allow_opportunistic != 0 && free_op->is_ready));
    }
    if (can_reuse) {
      iree_status_t status =
          iree_hal_streaming_buffer_begin_allocation_lifetime(buffer);
      if (!iree_status_is_ok(status)) {
        iree_slim_mutex_unlock(&context->pending_free_mutex);
        return status;
      }
      hrx_status_t insert_status = hrx_buffer_table_insert(
          &context->buffer_table, buffer->device_ptr, buffer->host_ptr,
          buffer->size, buffer->hrx_buf, buffer);
      if (!hrx_status_is_ok(insert_status)) {
        iree_slim_mutex_unlock(&context->pending_free_mutex);
        return HRX_CALL(insert_status);
      }
      *current = free_op->next;
      free_op->next = NULL;
      free_op->is_registered = false;
      free_op->is_reused = true;
      free_op->buffer = NULL;
      destroy_free_op = free_op->is_terminal;
      iree_hal_streaming_buffer_activate_pool_allocation(buffer);
      iree_hal_streaming_memory_account_device_allocation(context->device_entry,
                                                          buffer->size);
      *out_buffer = buffer;
      iree_slim_mutex_unlock(&context->pending_free_mutex);
      if (destroy_free_op) {
        iree_hal_streaming_pending_free_destroy(free_op);
      }
      return iree_ok_status();
    }
    current = &free_op->next;
  }
  iree_slim_mutex_unlock(&context->pending_free_mutex);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_device_from_pool_async(
    iree_hal_streaming_context_t* context, hrx_mem_pool_t pool,
    iree_device_size_t size, iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_try_reuse_pending_free(
      context, pool, size, stream, out_buffer));
  if (*out_buffer) return iree_ok_status();
  return iree_hal_streaming_memory_allocate_device_from_pool(
      context, pool, size, flags, out_buffer);
}

iree_status_t iree_hal_streaming_memory_allocate_device_pitched(
    iree_hal_streaming_context_t* context, iree_device_size_t width_bytes,
    iree_device_size_t height, iree_device_size_t element_size_bytes,
    iree_device_size_t* out_pitch, iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_pitch);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_pitch = 0;
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Match HIP's observed pitched allocation granularity.
  const iree_device_size_t alignment = 512;
  iree_device_size_t pitch = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_mul_add(width_bytes, 1,
                                                      alignment - 1, &pitch))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pitched allocation width overflows");
  }
  pitch = pitch / alignment * alignment;

  // Element sizes of 4, 8, or 16 bytes preserve coalesced access assumptions.
  // The allocation contract does not require enforcing that here.

  // Calculate total size.
  iree_device_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul(pitch, height, &total_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pitched allocation size overflows");
  }

  // Allocate the buffer with the calculated total size.
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device(
      context, total_size, 0, &buffer);

  if (iree_status_is_ok(status)) {
    *out_pitch = pitch;
    *out_buffer = buffer;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_streaming_buffer_is_device_freeable_base(
    const iree_hal_streaming_buffer_t* buffer,
    iree_hal_streaming_deviceptr_t ptr, size_t offset) {
  if (!buffer || offset != 0) return false;
  if (buffer->device_ptr == ptr &&
      (buffer->memory_type & IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    return true;
  }
  if (!buffer->is_managed) return false;
  if (buffer->device_ptr == ptr) return true;
  return buffer->host_ptr &&
         (iree_hal_streaming_deviceptr_t)(uintptr_t)buffer->host_ptr == ptr;
}

static iree_status_t iree_hal_streaming_memory_find_device_allocation_context(
    iree_hal_streaming_context_t* preferred_context,
    iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_context_t** out_context,
    iree_hal_streaming_buffer_t** out_wrapper) {
  IREE_ASSERT_ARGUMENT(preferred_context);
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_ASSERT_ARGUMENT(out_wrapper);
  *out_context = NULL;
  *out_wrapper = NULL;

  iree_hal_streaming_buffer_t* wrapper = NULL;
  size_t offset = 0;
  iree_status_t status = HRX_CALL(hrx_buffer_table_find(
      &preferred_context->buffer_table, ptr, NULL, &offset, (void**)&wrapper));
  if (iree_status_is_ok(status)) {
    if (iree_hal_streaming_buffer_is_device_freeable_base(wrapper, ptr,
                                                          offset)) {
      iree_hal_streaming_context_retain(preferred_context);
      *out_context = preferred_context;
      *out_wrapper = wrapper;
      return iree_ok_status();
    }
    if (wrapper && wrapper->device_ptr == ptr && offset == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pointer is not a device allocation");
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device pointer is not an allocation base");
  }
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return status;
  }
  iree_status_free(status);

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  bool found_invalid_pointer = false;
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    if (context == preferred_context) continue;
    wrapper = NULL;
    offset = 0;
    hrx_status_t find_status = hrx_buffer_table_find(
        &context->buffer_table, ptr, NULL, &offset, (void**)&wrapper);
    if (hrx_status_is_ok(find_status)) {
      if (iree_hal_streaming_buffer_is_device_freeable_base(wrapper, ptr,
                                                            offset)) {
        iree_hal_streaming_context_retain(context);
        *out_context = context;
        *out_wrapper = wrapper;
        hrx_status_ignore(find_status);
        break;
      }
      if (wrapper && wrapper->device_ptr == ptr && offset == 0) {
        found_invalid_pointer = true;
        hrx_status_ignore(find_status);
        break;
      }
      found_invalid_pointer = true;
      hrx_status_ignore(find_status);
      break;
    }
    hrx_status_ignore(find_status);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  if (*out_context) return iree_ok_status();
  if (found_invalid_pointer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pointer is not a device allocation base");
  }
  return iree_status_from_code(IREE_STATUS_NOT_FOUND);
}

static iree_status_t iree_hal_streaming_memory_find_host_allocation_context(
    iree_hal_streaming_context_t* preferred_context, void* ptr,
    iree_hal_streaming_context_t** out_context,
    iree_hal_streaming_buffer_t** out_wrapper, size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(preferred_context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_ASSERT_ARGUMENT(out_wrapper);
  IREE_ASSERT_ARGUMENT(out_offset);
  *out_context = NULL;
  *out_wrapper = NULL;
  *out_offset = 0;

  iree_hal_streaming_buffer_t* wrapper = NULL;
  size_t offset = 0;
  iree_status_t status = HRX_CALL(hrx_buffer_table_find(
      &preferred_context->buffer_table, (uint64_t)(uintptr_t)ptr, NULL, &offset,
      (void**)&wrapper));
  if (iree_status_is_ok(status)) {
    if (wrapper && wrapper->host_ptr) {
      iree_hal_streaming_context_retain(preferred_context);
      *out_context = preferred_context;
      *out_wrapper = wrapper;
      *out_offset = offset;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pointer is not host-visible memory");
  }
  if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    return status;
  }
  iree_status_free(status);

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    if (context == preferred_context) continue;
    wrapper = NULL;
    offset = 0;
    hrx_status_t find_status =
        hrx_buffer_table_find(&context->buffer_table, (uint64_t)(uintptr_t)ptr,
                              NULL, &offset, (void**)&wrapper);
    if (hrx_status_is_ok(find_status)) {
      hrx_status_ignore(find_status);
      if (wrapper && wrapper->host_ptr) {
        iree_hal_streaming_context_retain(context);
        *out_context = context;
        *out_wrapper = wrapper;
        *out_offset = offset;
      }
      break;
    }
    hrx_status_ignore(find_status);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  return *out_context ? iree_ok_status()
                      : iree_status_from_code(IREE_STATUS_NOT_FOUND);
}

iree_status_t iree_hal_streaming_memory_free_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status =
      iree_hal_streaming_memory_find_device_allocation_context(
          context, ptr, &owner_context, &wrapper);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // A HIP pointer may be hidden in native kernargs or device memory, so freeing
  // cannot rely on launch-time binding discovery. Flush and wait every active
  // context before releasing the allocation.
  status = iree_hal_streaming_context_synchronize_all();
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  status = HRX_CALL(hrx_buffer_table_remove(&owner_context->buffer_table,
                                            wrapper->device_ptr));
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Update free memory tracking.
  iree_hal_streaming_memory_account_device_free(wrapper);

  // Synchronous frees return an entirely idle pool's backing storage to the
  // allocator. Ordinary device allocations have no pool to trim.
  status = iree_hal_streaming_buffer_free_and_trim_pool(
      wrapper, /*trim_to_release_threshold=*/true);
  iree_hal_streaming_context_release(owner_context);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_deferred_device_free(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* call_context) {
  (void)args;
  (void)call_context;
  iree_hal_streaming_deferred_device_free_t* free_op =
      (iree_hal_streaming_deferred_device_free_t*)user_data;
  iree_hal_streaming_context_t* context = free_op->owner_context;
  iree_slim_mutex_lock(&context->pending_free_mutex);
  free_op->is_ready = true;
  iree_slim_mutex_unlock(&context->pending_free_mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_pending_free_release_list(
    iree_hal_streaming_deferred_device_free_t* completed_frees,
    bool trim_to_release_threshold) {
  iree_status_t status = iree_ok_status();
  while (completed_frees) {
    iree_hal_streaming_deferred_device_free_t* free_op = completed_frees;
    completed_frees = free_op->next;
    status =
        iree_status_join(status, iree_hal_streaming_pending_free_release_buffer(
                                     free_op, trim_to_release_threshold));
  }
  return status;
}

iree_status_t iree_hal_streaming_memory_release_completed_async_frees(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  iree_hal_streaming_deferred_device_free_t* completed_frees = NULL;

  iree_slim_mutex_lock(&context->pending_free_mutex);
  iree_hal_streaming_deferred_device_free_t** current =
      &context->pending_free_head;
  while (*current) {
    iree_hal_streaming_deferred_device_free_t* free_op = *current;
    if (free_op->source_stream_id == stream->stream_id && free_op->is_ready &&
        free_op->is_terminal) {
      *current = free_op->next;
      free_op->next = completed_frees;
      free_op->is_registered = false;
      completed_frees = free_op;
    } else {
      current = &free_op->next;
    }
  }
  iree_slim_mutex_unlock(&context->pending_free_mutex);

  return iree_hal_streaming_pending_free_release_list(
      completed_frees, /*trim_to_release_threshold=*/true);
}

iree_status_t iree_hal_streaming_memory_release_terminal_async_frees(
    iree_hal_streaming_context_t* context) {
  iree_hal_streaming_deferred_device_free_t* completed_frees = NULL;

  iree_slim_mutex_lock(&context->pending_free_mutex);
  while (context->pending_free_head) {
    iree_hal_streaming_deferred_device_free_t* free_op =
        context->pending_free_head;
    IREE_ASSERT(free_op->is_terminal,
                "a nonterminal asynchronous free retained its context");
    context->pending_free_head = free_op->next;
    free_op->next = completed_frees;
    free_op->is_registered = false;
    completed_frees = free_op;
  }
  iree_slim_mutex_unlock(&context->pending_free_mutex);

  return iree_hal_streaming_pending_free_release_list(
      completed_frees, /*trim_to_release_threshold=*/true);
}

iree_status_t iree_hal_streaming_memory_release_completed_async_frees_from_pool(
    hrx_mem_pool_t pool) {
  if (!pool) return iree_ok_status();

  iree_hal_streaming_deferred_device_free_t* completed_frees = NULL;
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  // Pool trimming is an explicit cold-path operation. Scan contexts here
  // instead of imposing a process-wide registry on every allocation.
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    iree_slim_mutex_lock(&context->pending_free_mutex);
    iree_hal_streaming_deferred_device_free_t** current =
        &context->pending_free_head;
    while (*current) {
      iree_hal_streaming_deferred_device_free_t* free_op = *current;
      if (free_op->is_ready && free_op->is_terminal && free_op->buffer &&
          free_op->buffer->allocation_pool == pool) {
        *current = free_op->next;
        free_op->next = completed_frees;
        free_op->is_registered = false;
        completed_frees = free_op;
      } else {
        current = &free_op->next;
      }
    }
    iree_slim_mutex_unlock(&context->pending_free_mutex);
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  return iree_hal_streaming_pending_free_release_list(
      completed_frees, /*trim_to_release_threshold=*/false);
}

iree_status_t iree_hal_streaming_memory_free_device_async(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_stream_t* stream) {
  if (!ptr) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status =
      iree_hal_streaming_memory_find_device_allocation_context(
          context, ptr, &owner_context, &wrapper);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  if (owner_context != stream->context) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "stream-ordered free cannot be enqueued on a foreign device context");
  }

  // Non-pool allocations have no reusable backing and are retired as soon as
  // the stream reaches the free. Pool policy is sampled when the operation is
  // enqueued so terminal cleanup never needs to query a possibly reused
  // wrapper.
  uint64_t allow_opportunistic = wrapper->allocation_pool ? 0 : 1;
  if (wrapper->allocation_pool) {
    status = HRX_CALL(hrx_mem_pool_get_attribute(
        wrapper->allocation_pool, HRX_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
        &allow_opportunistic));
    if (!iree_status_is_ok(status)) {
      iree_hal_streaming_context_release(owner_context);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  iree_hal_streaming_deferred_device_free_t* free_op = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(*free_op),
                                 (void**)&free_op);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  free_op->owner_context = owner_context;
  free_op->source_stream_id = stream->stream_id;
  // No event can safely reuse this allocation until the host call has been
  // queued and its stream timeline value is known.
  free_op->completion_value = UINT64_MAX;
  free_op->buffer = wrapper;
  free_op->next = NULL;
  free_op->is_registered = false;
  free_op->is_ready = false;
  free_op->is_terminal = false;
  free_op->is_reused = false;
  free_op->allow_opportunistic = allow_opportunistic != 0;

  iree_hal_streaming_pending_free_terminal_t* terminal = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), sizeof(*terminal),
                                 (void**)&terminal);
  if (!iree_status_is_ok(status)) {
    free_op->buffer = NULL;
    iree_hal_streaming_pending_free_destroy(free_op);
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_hal_resource_initialize(&iree_hal_streaming_pending_free_terminal_vtable,
                               &terminal->resource);
  terminal->free_op = free_op;

  status =
      HRX_CALL(hrx_buffer_table_reserve_insert(&owner_context->buffer_table));
  if (!iree_status_is_ok(status)) {
    free_op->buffer = NULL;
    iree_hal_resource_release(&terminal->resource);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  const hrx_status_t remove_status = hrx_buffer_table_remove(
      &owner_context->buffer_table, wrapper->device_ptr);
  if (!hrx_status_is_ok(remove_status)) {
    hrx_buffer_table_cancel_reserved_insert(&owner_context->buffer_table);
    free_op->buffer = NULL;
    iree_hal_resource_release(&terminal->resource);
    IREE_TRACE_ZONE_END(z0);
    return HRX_CALL(remove_status);
  }

  // The allocation becomes unavailable to future streams immediately, but a
  // later allocation on this stream may adopt its backing before the queued
  // free reaches the host callback.
  iree_hal_streaming_buffer_release_pool_allocation(wrapper);
  iree_hal_streaming_memory_account_device_free(wrapper);
  uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call = iree_hal_make_host_call_with_resource(
      iree_hal_streaming_deferred_device_free, free_op, &terminal->resource);
  status = iree_hal_streaming_queue_host_call(stream, call, args,
                                              IREE_HAL_HOST_CALL_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    hrx_buffer_table_cancel_reserved_insert(&owner_context->buffer_table);
    iree_slim_mutex_lock(&stream->mutex);
    free_op->completion_value = stream->pending_value;
    iree_slim_mutex_unlock(&stream->mutex);
    // Publish only after the callback has a stable completion value. The
    // caller's resource reference prevents terminal cleanup from racing this
    // publication even when the callback executes inline.
    iree_hal_streaming_pending_free_register(free_op);
    iree_hal_resource_release(&terminal->resource);
  }
  if (!iree_status_is_ok(status)) {
    hrx_status_t insert_status = hrx_buffer_table_insert_reserved(
        &owner_context->buffer_table, wrapper->device_ptr, wrapper->host_ptr,
        wrapper->size, wrapper->hrx_buf, wrapper);
    if (!hrx_status_is_ok(insert_status)) {
      status = iree_status_join(status, HRX_CALL(insert_status));
      status = iree_status_join(
          status, iree_hal_streaming_buffer_free_and_trim_pool(
                      wrapper, /*trim_to_release_threshold=*/true));
    } else {
      iree_hal_streaming_buffer_activate_pool_allocation(wrapper);
      iree_hal_streaming_memory_account_device_allocation(
          owner_context->device_entry, wrapper->size);
    }
    free_op->buffer = NULL;
    iree_hal_resource_release(&terminal->resource);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_memory_allocate_host_with_context_mode(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_buffer_usage_t usage, iree_device_size_t min_alignment,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  const iree_host_size_t allocation_size = iree_max(size, (iree_host_size_t)8);
  const iree_host_size_t host_alignment =
      iree_max((iree_host_size_t)min_alignment, (iree_host_size_t)4096);
  iree_hal_memory_type_t memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = host_alignment,
  };

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      context->device_allocator, params, allocation_size, &buffer);

  iree_hal_streaming_buffer_t* wrapper = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_buffer_wrap(
        context, buffer, (int)memory_type, /*imported_host_ptr=*/NULL,
        /*allocation_pool=*/NULL, context_ownership, &wrapper);
  }
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status) && wrapper->host_ptr == NULL) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "host allocation did not export a host-visible pointer");
  }
  if (iree_status_is_ok(status)) {
    memset(wrapper->host_ptr, 0, allocation_size);
  }

  if (iree_status_is_ok(status)) {
    wrapper->imported_host_allocation = false;
    wrapper->host_register_flags = flags;
    *out_buffer = wrapper;
  } else {
    if (wrapper) {
      hrx_buffer_table_remove(&context->buffer_table, wrapper->device_ptr);
      iree_hal_streaming_buffer_free(wrapper);
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t
iree_hal_streaming_memory_allocate_owned_host_import_with_context_mode(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_buffer_usage_t usage, iree_device_size_t min_alignment,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  const iree_host_size_t allocation_size = iree_max(size, (iree_host_size_t)8);
  const iree_host_size_t host_alignment =
      iree_max((iree_host_size_t)min_alignment, (iree_host_size_t)4096);
  void* host_ptr = NULL;
  iree_status_t status = iree_allocator_malloc_aligned(
      context->host_allocator, allocation_size, host_alignment,
      /*offset=*/0, &host_ptr);

  // Managed and host allocations are process-wide HIP pointers. On a
  // multi-GPU process they must be imported through the all-visible logical
  // device so the pin operation grants every GPU agent access to the same VA.
  // A single-GPU process has no separate shared-memory device and can use the
  // context allocator directly.
  iree_hal_allocator_t* import_allocator =
      hrx_gpu_shared_memory_hal_allocator();
  if (!import_allocator) import_allocator = context->device_allocator;

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_params_t params = {
        .usage = usage,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
        .min_alignment = host_alignment,
    };
    iree_hal_external_buffer_t external_buffer = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
        .size = allocation_size,
        .handle.host_allocation.ptr = host_ptr,
    };
    status = iree_hal_allocator_import_buffer(
        import_allocator, params, &external_buffer,
        iree_hal_buffer_release_callback_null(), &buffer);
  }

  iree_hal_streaming_buffer_t* wrapper = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_buffer_wrap(
        context, buffer,
        (int)(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
        host_ptr, /*allocation_pool=*/NULL, context_ownership, &wrapper);
  }
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status)) {
    wrapper->owns_host_ptr = true;
    wrapper->imported_host_allocation = false;
    wrapper->host_register_flags = flags;
    *out_buffer = wrapper;
    host_ptr = NULL;
  } else {
    if (wrapper) {
      hrx_buffer_table_remove(&context->buffer_table, wrapper->device_ptr);
      iree_hal_streaming_buffer_free(wrapper);
    }
  }
  iree_allocator_free_aligned(context->host_allocator, host_ptr);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_allocate_host(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  return iree_hal_streaming_memory_allocate_owned_host_import_with_context_mode(
      context, size, flags, IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*min_alignment=*/64, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
      out_buffer);
}

iree_status_t iree_hal_streaming_memory_allocate_host_staging(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_buffer_t** out_buffer) {
  return iree_hal_streaming_memory_allocate_host_with_context_mode(
      context, size, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
      IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*min_alignment=*/64, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
      out_buffer);
}

static iree_status_t iree_hal_streaming_managed_metadata_allocate(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);

  const iree_host_size_t page_size = 4096;
  iree_host_size_t page_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul_add(
          buffer->size, 1, page_size - 1, &page_count))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "managed allocation size overflows page count");
  }
  page_count /= page_size;

  iree_host_size_t read_mostly_size = 0;
  iree_host_size_t location_size = 0;
  iree_host_size_t mask_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(page_count, sizeof(bool),
                                                &read_mostly_size) ||
                    !iree_host_size_checked_mul(page_count, sizeof(int32_t),
                                                &location_size) ||
                    !iree_host_size_checked_mul(page_count, sizeof(uint64_t),
                                                &mask_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "managed metadata size overflow");
  }

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(context->host_allocator, read_mostly_size,
                                   (void**)&buffer->managed_read_mostly_pages);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(context->host_allocator, location_size,
                              (void**)&buffer->managed_preferred_locations);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        context->host_allocator, mask_size,
        (void**)&buffer->managed_accessed_by_device_masks);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(context->host_allocator, location_size,
                              (void**)&buffer->managed_last_prefetch_locations);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(context->host_allocator, location_size,
                                   (void**)&buffer->managed_coherency_modes);
  }
  if (!iree_status_is_ok(status)) return status;

  memset(buffer->managed_read_mostly_pages, 0, read_mostly_size);
  memset(buffer->managed_accessed_by_device_masks, 0, mask_size);
  for (iree_host_size_t i = 0; i < page_count; ++i) {
    buffer->managed_preferred_locations[i] = -2;
    buffer->managed_last_prefetch_locations[i] = -2;
    buffer->managed_coherency_modes[i] = 0;
  }
  buffer->managed_page_count = page_count;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_managed(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    unsigned int allocation_flags, iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t allocation_size = iree_max(size, (iree_host_size_t)8);
  iree_hal_streaming_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_memory_allocate_owned_host_import_with_context_mode(
          context, allocation_size,
          (iree_hal_streaming_host_register_flags_t)allocation_flags,
          IREE_HAL_BUFFER_USAGE_DEFAULT,
          /*min_alignment=*/4096, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
          &buffer));
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    buffer->is_managed = true;
    buffer->host_register_flags =
        (iree_hal_streaming_host_register_flags_t)allocation_flags;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_managed_metadata_allocate(context, buffer);
  }
  if (!iree_status_is_ok(status)) {
    hrx_buffer_table_remove(&context->buffer_table, buffer->device_ptr);
    iree_hal_streaming_buffer_free(buffer);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  *out_buffer = buffer;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_context_ensure_pageable_h2d_staging(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_buffer_t** out_staging) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_staging);
  *out_staging = NULL;

  if (context->pageable_h2d_staging_buffer &&
      context->pageable_h2d_staging_size >= size) {
    *out_staging = context->pageable_h2d_staging_buffer;
    return iree_ok_status();
  }

  if (context->pageable_h2d_staging_buffer) {
    iree_hal_streaming_memory_release_pageable_staging(context);
  }

  iree_hal_streaming_buffer_t* staging = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_memory_allocate_host_with_context_mode(
          context, size, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
          IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*min_alignment=*/64, IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED,
          &staging));
  context->pageable_h2d_staging_buffer = staging;
  context->pageable_h2d_staging_size = size;
  *out_staging = staging;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_free_host(
    iree_hal_streaming_context_t* context, void* ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  size_t offset = 0;
  iree_status_t status = iree_hal_streaming_memory_find_host_allocation_context(
      context, ptr, &owner_context, &wrapper, &offset);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  if (!wrapper || wrapper->host_ptr != ptr || offset != 0) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host pointer is not an allocation base");
  }
  if (wrapper->imported_host_allocation) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "registered host memory must be unregistered");
  }
  if (wrapper->is_managed) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "managed memory must be freed with hipFree");
  }

  // Host allocations can be passed directly to kernels, and those pointers may
  // be hidden in native kernargs or other buffers. Flush and wait every active
  // context before returning the host allocation to the system.
  status = iree_hal_streaming_context_synchronize_all();
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Remove from mapping table.
  hrx_buffer_table_remove(&owner_context->buffer_table, wrapper->device_ptr);

  // Free wrapper and release its context ownership edge.
  iree_hal_streaming_buffer_free(wrapper);
  iree_hal_streaming_context_release(owner_context);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_register_host(
    iree_hal_streaming_context_t* context, void* ptr, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
      .size = (iree_device_size_t)size,
      .handle.host_allocation.ptr = ptr,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_hal_allocator_t* import_allocator =
      hrx_gpu_shared_memory_hal_allocator();
  if (!import_allocator) import_allocator = context->device_allocator;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_import_buffer(
              import_allocator, params, &external_buffer,
              iree_hal_buffer_release_callback_null(), &buffer));

  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap(
      context, buffer, (int)params.type, ptr, /*allocation_pool=*/NULL,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED, &wrapper);
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_managed_metadata_allocate(context, wrapper);
  }
  if (iree_status_is_ok(status)) {
    wrapper->host_register_flags = flags;
    *out_buffer = wrapper;
  } else {
    if (wrapper) {
      hrx_buffer_table_remove(&context->buffer_table, wrapper->device_ptr);
      iree_hal_streaming_buffer_free(wrapper);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_unregister_host(
    iree_hal_streaming_context_t* context, void* ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  size_t offset = 0;
  iree_status_t status = iree_hal_streaming_memory_find_host_allocation_context(
      context, ptr, &owner_context, &wrapper, &offset);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  if (!wrapper || !wrapper->imported_host_allocation ||
      wrapper->host_ptr != ptr || offset != 0) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host pointer is not a registered allocation base");
  }

  // Registered host pointers are process-visible HIP pointers. Unregistration
  // is blocking, so wait all active contexts instead of assuming launch-time
  // pointer discovery found every consumer.
  status = iree_hal_streaming_context_synchronize_all();
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_release(owner_context);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Remove from buffer table.
  hrx_buffer_table_remove(&owner_context->buffer_table, wrapper->device_ptr);

  // Free wrapper (this will release the HAL buffer and context references).
  iree_hal_streaming_buffer_free(wrapper);
  iree_hal_streaming_context_release(owner_context);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_address_range(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_deviceptr_t* out_base, iree_device_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_base);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_base = 0;
  *out_size = 0;

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = HRX_CALL(hrx_buffer_table_find(
      &context->buffer_table, ptr, NULL, NULL, (void**)&wrapper));
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (ptr >= wrapper->device_ptr && ptr - wrapper->device_ptr < wrapper->size) {
    *out_base = wrapper->device_ptr;
  } else if (wrapper->host_ptr) {
    *out_base = (iree_hal_streaming_deviceptr_t)wrapper->host_ptr;
  } else {
    *out_base = wrapper->device_ptr;
  }
  *out_size = wrapper->logical_size;

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_host_flags(
    iree_hal_streaming_context_t* context, void* ptr,
    iree_hal_streaming_host_register_flags_t* out_flags) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_flags);
  *out_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;

  iree_hal_streaming_context_t* owner_context = NULL;
  iree_hal_streaming_buffer_t* wrapper = NULL;
  size_t offset = 0;
  iree_status_t status = iree_hal_streaming_memory_find_host_allocation_context(
      context, ptr, &owner_context, &wrapper, &offset);
  if (iree_status_is_ok(status)) {
    *out_flags = wrapper->host_register_flags;
    iree_hal_streaming_context_release(owner_context);
  }

  return status;
}

static iree_status_t iree_hal_streaming_memory_lookup_memset_range(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, iree_hal_streaming_context_t** out_owner_context,
    iree_hal_streaming_buffer_ref_t* out_dst_ref) {
  *out_owner_context = NULL;
  iree_status_t lookup_status =
      iree_hal_streaming_memory_lookup_range(context, dst, length, out_dst_ref);
  if (!iree_status_is_ok(lookup_status) &&
      iree_status_code(lookup_status) == IREE_STATUS_NOT_FOUND) {
    iree_status_ignore(lookup_status);
    lookup_status = iree_hal_streaming_memory_lookup_range_across_contexts(
        dst, length, out_owner_context, out_dst_ref);
    if (iree_status_is_ok(lookup_status)) {
      if (!out_dst_ref->buffer->is_managed) {
        iree_hal_streaming_context_release(*out_owner_context);
        *out_owner_context = NULL;
        lookup_status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
      }
    }
  }
  return lookup_status;
}

iree_status_t iree_hal_streaming_memory_memset(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Add memset node to the graph instead of recording to command buffer.
    // Convert pattern to uint32_t (assuming pattern_length is 1, 2, or 4).
    uint32_t pattern_value = 0;
    if (pattern_length == 1 || pattern_length == 2 || pattern_length == 4) {
      memcpy(&pattern_value, pattern, pattern_length);
    } else {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "unsupported pattern length %zu", pattern_length));
    }
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_fill_ptr_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, dst, pattern_value,
                pattern_length, length / pattern_length, &node));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_set_last_node(stream, node));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up the entire destination range. Managed allocations are process-wide
  // HIP pointers, so a device switch after allocation must still resolve them.
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_hal_streaming_context_t* owner_context = NULL;
  iree_status_t lookup_status = iree_hal_streaming_memory_lookup_memset_range(
      context, dst, length, &owner_context, &dst_ref);
  if (!iree_status_is_ok(lookup_status)) {
    iree_hal_streaming_context_release(owner_context);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, lookup_status, "resolving `dst` buffer ref %p", (void*)dst);
  }

  // Host allocations must be queue-visible to preserve stream ordering.
  if (!dst_ref.buffer->buffer) {
    iree_hal_streaming_context_release(owner_context);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "memset destination is not represented by a HAL buffer"));
  }

  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);

  iree_hal_buffer_ref_t target_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, length);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_fill_buffer(
        stream->command_buffer, target_ref, pattern, pattern_length,
        IREE_HAL_FILL_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_barrier(stream->command_buffer);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  iree_hal_streaming_context_release(owner_context);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_complete_synchronous_memset(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(stream);

  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_hal_streaming_context_t* owner_context = NULL;
  iree_status_t status = iree_hal_streaming_memory_lookup_memset_range(
      context, dst, length, &owner_context, &dst_ref);
  if (iree_status_is_ok(status)) {
    const bool requires_synchronous_completion =
        dst_ref.buffer->is_managed || dst_ref.offset != 0 ||
        (dst_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_HOST_LOCAL) != 0;
    if (requires_synchronous_completion) {
      status =
          stream == context->default_stream
              ? iree_hal_streaming_context_synchronize_legacy_default(context)
              : iree_hal_streaming_stream_synchronize(stream);
    }
  }
  iree_hal_streaming_context_release(owner_context);
  return status;
}

iree_status_t iree_hal_streaming_memory_memcpy(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Add memcpy node to the graph instead of recording to command buffer.
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_copy_ptr_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, dst, src, size, &node));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_set_last_node(stream, node));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up buffers from device pointers.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
      "resolving `src` buffer ref %p", (void*)src);

  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);

  iree_hal_buffer_ref_t src_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
  iree_hal_buffer_ref_t dst_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_copy_buffer(stream->command_buffer,
                                                 src_buffer_ref, dst_buffer_ref,
                                                 IREE_HAL_COPY_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_barrier(stream->command_buffer);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_peer(
    iree_hal_streaming_context_t* dst_context,
    iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_context_t* src_context,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst_context);
  IREE_ASSERT_ARGUMENT(src_context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  bool can_access = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_device_can_access_peer(src_context->device_ordinal,
                                                    dst_context->device_ordinal,
                                                    &can_access));
  if (!can_access) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                             "P2P access not supported between devices %" PRIhsz
                             " and %" PRIhsz,
                             src_context->device_ordinal,
                             dst_context->device_ordinal));
  }

  // Look up buffers from device pointers.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(dst_context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(src_context, src, &src_ref),
      "resolving `src` buffer ref %p", (void*)src);

  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);

  iree_hal_buffer_ref_t src_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
  iree_hal_buffer_ref_t dst_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_copy_buffer(stream->command_buffer,
                                                 src_buffer_ref, dst_buffer_ref,
                                                 IREE_HAL_COPY_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_barrier(stream->command_buffer);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_enqueue_buffer_copy(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_ref_t src_ref,
    iree_hal_streaming_buffer_ref_t dst_ref, iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(stream);
  if (!src_ref.buffer || !src_ref.buffer->buffer || !dst_ref.buffer ||
      !dst_ref.buffer->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy operands are not queue-compatible buffers");
  }

  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);
  iree_hal_buffer_ref_t src_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
  iree_hal_buffer_ref_t dst_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
  if (iree_status_is_ok(status) &&
      iree_any_bit_set((iree_hal_memory_type_t)src_ref.buffer->memory_type,
                       IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    status = iree_hal_streaming_command_buffer_host_to_transfer_barrier(
        stream->command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_copy_buffer(stream->command_buffer,
                                                 src_buffer_ref, dst_buffer_ref,
                                                 IREE_HAL_COPY_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_barrier(stream->command_buffer);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return status;
}

typedef struct iree_hal_streaming_host_d2h_staging_t {
  // Resource retained while the staging completion is pending in a HAL queue.
  iree_hal_resource_t resource;
  // Allocator used to release this staging operation.
  iree_allocator_t host_allocator;
  // User host destination pointer populated after the stream D2H completes.
  void* dst;
  // Byte distance between consecutive destination rows.
  iree_device_size_t dst_pitch;
  // Host staging wrapper containing the completed D2H bytes.
  iree_hal_streaming_buffer_t* staging;
  // Number of bytes copied from each source row.
  iree_device_size_t width;
  // Number of rows copied from staging into the destination.
  iree_host_size_t height;
} iree_hal_streaming_host_d2h_staging_t;

static void iree_hal_streaming_host_d2h_staging_destroy(
    iree_hal_resource_t* base_resource) {
  iree_hal_streaming_host_d2h_staging_t* copy =
      (iree_hal_streaming_host_d2h_staging_t*)base_resource;
  iree_hal_streaming_temporary_host_buffer_free(copy->staging->context,
                                                copy->staging);
  iree_allocator_free(copy->host_allocator, copy);
}

static const iree_hal_resource_vtable_t
    iree_hal_streaming_host_d2h_staging_vtable = {
        .destroy = iree_hal_streaming_host_d2h_staging_destroy,
};

static void iree_hal_streaming_host_d2h_staging_scatter_rows(
    void* dst, iree_device_size_t dst_pitch, const void* src,
    iree_device_size_t width, iree_host_size_t height) {
  uint8_t* dst_ptr = (uint8_t*)dst;
  const uint8_t* src_ptr = (const uint8_t*)src;
  for (iree_host_size_t row = 0; row < height; ++row) {
    memcpy(dst_ptr + row * dst_pitch, src_ptr + row * width, width);
  }
}

static iree_status_t iree_hal_streaming_host_d2h_staging_call(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* call_context) {
  (void)args;
  (void)call_context;
  iree_hal_streaming_host_d2h_staging_t* copy =
      (iree_hal_streaming_host_d2h_staging_t*)user_data;
  iree_hal_streaming_host_d2h_staging_scatter_rows(copy->dst, copy->dst_pitch,
                                                   copy->staging->host_ptr,
                                                   copy->width, copy->height);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_enqueue_host_d2h_staging_copy(
    iree_hal_streaming_stream_t* stream, void* dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_buffer_t* staging,
    iree_device_size_t width, iree_host_size_t height) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(staging);

  iree_hal_streaming_host_d2h_staging_t* copy = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(*copy), (void**)&copy);
  if (!iree_status_is_ok(status)) {
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    if (iree_status_is_ok(sync_status)) {
      iree_hal_streaming_host_d2h_staging_scatter_rows(
          dst, dst_pitch, staging->host_ptr, width, height);
    }
    iree_hal_streaming_temporary_host_buffer_free(staging->context, staging);
    if (iree_status_is_ok(sync_status)) {
      iree_status_free(status);
      return iree_ok_status();
    }
    return iree_status_join(status, sync_status);
  }
  iree_hal_resource_initialize(&iree_hal_streaming_host_d2h_staging_vtable,
                               &copy->resource);
  copy->host_allocator = iree_allocator_system();
  copy->dst = dst;
  copy->dst_pitch = dst_pitch;
  copy->staging = staging;
  copy->width = width;
  copy->height = height;

  uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call = iree_hal_make_host_call_with_resource(
      iree_hal_streaming_host_d2h_staging_call, copy, &copy->resource);
  status = iree_hal_streaming_queue_host_call(stream, call, args,
                                              IREE_HAL_HOST_CALL_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    if (iree_status_is_ok(sync_status)) {
      iree_hal_streaming_host_d2h_staging_scatter_rows(
          dst, dst_pitch, staging->host_ptr, width, height);
    }
    iree_hal_resource_release(&copy->resource);
    if (iree_status_is_ok(sync_status)) {
      iree_status_free(status);
      return iree_ok_status();
    }
    return iree_status_join(status, sync_status);
  }
  iree_hal_resource_release(&copy->resource);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_enqueue_host_update(
    iree_hal_streaming_stream_t* stream, const void* src,
    iree_hal_streaming_buffer_ref_t dst_ref, iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(src);
  if (!dst_ref.buffer || !dst_ref.buffer->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host update destination is not a device buffer");
  }
  iree_device_size_t range_end = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_add(dst_ref.offset, size, &range_end) ||
          range_end > dst_ref.buffer->size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "host update exceeds destination buffer range");
  }

  const uint8_t* src_ptr = (const uint8_t*)src;
  iree_device_size_t remaining = size;
  iree_device_size_t chunk_offset = 0;
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);
  while (remaining > 0 && iree_status_is_ok(status)) {
    const iree_device_size_t this_chunk =
        remaining < IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE
            ? remaining
            : IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE;
    const iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
        dst_ref.buffer->buffer, dst_ref.offset + chunk_offset, this_chunk);
    status = iree_hal_command_buffer_update_buffer(
        stream->command_buffer, src_ptr + chunk_offset, 0, target_ref,
        IREE_HAL_UPDATE_FLAG_NONE);
    chunk_offset += this_chunk;
    remaining -= this_chunk;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_barrier(stream->command_buffer);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return status;
}

//===----------------------------------------------------------------------===//
// Memory copy helper functions
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_memcpy_host_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    const void* src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (stream &&
      stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    iree_hal_streaming_buffer_t* staging = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_allocate_host_staging(
                stream->capture_graph, size, &staging));

    iree_hal_streaming_host_memcpy_callback_data_t* callback_data = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_arena_allocate(&stream->capture_graph->arena,
                            sizeof(*callback_data), (void**)&callback_data));
    callback_data->dst = staging->host_ptr;
    callback_data->src = src;
    callback_data->count = size;

    iree_hal_streaming_graph_node_t* callback_node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_host_call_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count,
                iree_hal_streaming_host_memcpy_callback, callback_data,
                &callback_node));
    callback_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
    callback_node->attrs.host.user_data_size = sizeof(*callback_data);

    iree_hal_streaming_graph_node_t* copy_node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_copy_ptr_node_with_extra_dependency(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, callback_node, dst,
                staging->device_ptr, size, &copy_node));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_set_last_node(stream, copy_node));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up destination buffer from device pointer.
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t dst_status =
      iree_hal_streaming_memory_lookup(context, dst, &dst_ref);

  if (!iree_status_is_ok(dst_status)) {
    IREE_TRACE_ZONE_END(z0);
    return dst_status;
  }

  if (dst_ref.buffer->host_ptr &&
      iree_any_bit_set((iree_hal_memory_type_t)dst_ref.buffer->memory_type,
                       IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    iree_status_t status =
        iree_hal_streaming_buffer_ref_validate_range(&dst_ref, size);
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_context_synchronize_all();
    }
    if (iree_status_is_ok(status)) {
      memcpy((uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset, src, size);
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  const iree_device_size_t staging_threshold = 256 * 1024;
  if (stream) {
    iree_hal_streaming_buffer_ref_t src_ref;
    iree_status_t src_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)src, &src_ref);
    if (iree_status_is_ok(src_status)) {
      if (!(src_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "host-to-device source is not host memory");
      }
      iree_status_t queue_status = iree_hal_streaming_enqueue_buffer_copy(
          stream, src_ref, dst_ref, size);
      IREE_TRACE_ZONE_END(z0);
      return queue_status;
    }
    iree_status_ignore(src_status);
    if (size <= staging_threshold) {
      iree_status_t queue_status =
          iree_hal_streaming_enqueue_host_update(stream, src, dst_ref, size);
      IREE_TRACE_ZONE_END(z0);
      return queue_status;
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_stream_synchronize(stream));
  }

  if (size >= staging_threshold) {
    iree_hal_streaming_stream_t* copy_stream =
        stream ? stream : context->default_stream;
    iree_hal_streaming_buffer_t* staging = NULL;
    iree_slim_mutex_lock(&context->mutex);
    iree_status_t status =
        iree_hal_streaming_context_ensure_pageable_h2d_staging(context, size,
                                                               &staging);
    if (iree_status_is_ok(status)) {
      memcpy(staging->host_ptr, src, size);
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_streaming_buffer_ref_t staging_ref = {
          .buffer = staging,
          .offset = 0,
      };
      status = iree_hal_streaming_enqueue_buffer_copy(copy_stream, staging_ref,
                                                      dst_ref, size);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_stream_synchronize(copy_stream);
    }
    iree_slim_mutex_unlock(&context->mutex);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  } else {
    // Small host-to-device transfers are faster through the direct blocking
    // path than paying temporary host-visible allocation overhead.
    iree_status_t direct_status = iree_hal_streaming_direct_transfer_h2d(
        context, src, dst_ref.buffer->buffer, dst_ref.offset, size);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, direct_status);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_device_to_host(
    iree_hal_streaming_context_t* context, void* dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (stream &&
      stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    iree_hal_streaming_buffer_t* staging = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_allocate_host_staging(
                stream->capture_graph, size, &staging));

    iree_hal_streaming_graph_node_t* copy_node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_copy_ptr_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, staging->device_ptr, src,
                size, &copy_node));

    iree_hal_streaming_host_memcpy_callback_data_t* callback_data = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_arena_allocate(&stream->capture_graph->arena,
                            sizeof(*callback_data), (void**)&callback_data));
    callback_data->dst = dst;
    callback_data->src = staging->host_ptr;
    callback_data->count = size;

    iree_hal_streaming_graph_node_t* copy_deps[] = {copy_node};
    iree_hal_streaming_graph_node_t* callback_node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_host_call_node(
                stream->capture_graph, copy_deps, IREE_ARRAYSIZE(copy_deps),
                iree_hal_streaming_host_memcpy_callback, callback_data,
                &callback_node));
    callback_node->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN;
    callback_node->attrs.host.user_data_size = sizeof(*callback_data);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_set_last_node(stream, callback_node));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up source buffer from device pointer.
  iree_hal_streaming_buffer_ref_t src_ref;
  iree_status_t src_status =
      iree_hal_streaming_memory_lookup(context, src, &src_ref);

  if (!iree_status_is_ok(src_status)) {
    IREE_TRACE_ZONE_END(z0);
    return src_status;
  }

  if (src_ref.buffer->host_ptr &&
      iree_any_bit_set((iree_hal_memory_type_t)src_ref.buffer->memory_type,
                       IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    iree_status_t status =
        iree_hal_streaming_buffer_ref_validate_range(&src_ref, size);
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_context_synchronize_all();
    }
    if (iree_status_is_ok(status)) {
      memcpy(dst, (const uint8_t*)src_ref.buffer->host_ptr + src_ref.offset,
             size);
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (stream) {
    iree_hal_streaming_buffer_ref_t dst_ref;
    iree_status_t dst_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)dst, &dst_ref);
    if (iree_status_is_ok(dst_status)) {
      if (!(dst_ref.buffer->memory_type & IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "device-to-host destination is not host memory");
      }
    } else {
      iree_status_ignore(dst_status);
    }

    // Host allocations do not necessarily provide the system-scope visibility
    // required for the CPU to consume a command-buffer D2H result directly.
    // Copy into runtime-owned host-visible staging and publish the bytes to the
    // destination in a stream-ordered host call. The callback performs CPU work
    // only; starting another device transfer there can deadlock queue progress.
    iree_hal_streaming_buffer_t* staging = NULL;
    iree_status_t status =
        iree_hal_streaming_memory_allocate_host_with_context_mode(
            context, size, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
            IREE_HAL_BUFFER_USAGE_TRANSFER,
            /*min_alignment=*/64, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
            &staging);
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_buffer_ref_t staging_ref = {
          .buffer = staging,
          .offset = 0,
      };
      status = iree_hal_streaming_enqueue_buffer_copy(stream, src_ref,
                                                      staging_ref, size);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_buffer_t* callback_staging = staging;
      staging = NULL;
      status = iree_hal_streaming_enqueue_host_d2h_staging_copy(
          stream, dst, /*dst_pitch=*/size, callback_staging,
          /*width=*/size, /*height=*/1);
    }
    if (staging) {
      iree_hal_streaming_temporary_host_buffer_free(staging->context, staging);
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_status_t direct_status = iree_hal_streaming_direct_transfer_d2h(
      context, src_ref.buffer->buffer, src_ref.offset, dst, size);
  if (iree_status_is_ok(direct_status)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  iree_status_ignore(direct_status);

  iree_hal_streaming_stream_t* copy_stream =
      stream ? stream : context->default_stream;
  iree_hal_streaming_buffer_t* staging = NULL;
  iree_status_t status =
      iree_hal_streaming_memory_allocate_host_with_context_mode(
          context, size, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
          IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*min_alignment=*/64, IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED,
          &staging);
  if (iree_status_is_ok(status) && !copy_stream->command_buffer) {
    status = iree_hal_streaming_stream_begin(copy_stream);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&copy_stream->mutex);
    status = iree_hal_streaming_stream_begin_locked(copy_stream);
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_buffer_ref_t staging_ref = {
          .buffer = staging,
          .offset = 0,
      };
      iree_hal_buffer_ref_t src_buffer_ref =
          iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
      iree_hal_buffer_ref_t staging_buffer_ref =
          iree_hal_streaming_convert_range_buffer_ref(staging_ref, size);
      status = iree_hal_command_buffer_copy_buffer(
          copy_stream->command_buffer, src_buffer_ref, staging_buffer_ref,
          IREE_HAL_COPY_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_command_buffer_barrier(
          copy_stream->command_buffer);
    }
    iree_slim_mutex_unlock(&copy_stream->mutex);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_synchronize(copy_stream);
  }
  if (iree_status_is_ok(status)) {
    memcpy(dst, staging->host_ptr, size);
  }
  iree_hal_streaming_temporary_host_buffer_free(context, staging);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_host_to_device_2d(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t dst_pitch, const void* src, iree_device_size_t src_pitch,
    iree_device_size_t width, iree_host_size_t height,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (width > dst_pitch || width > src_pitch) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "copy width exceeds a row pitch"));
  }
  if (IREE_UNLIKELY((iree_host_size_t)(iree_device_size_t)height != height)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "H2D row count exceeds device address space"));
  }

  const iree_device_size_t row_count = (iree_device_size_t)height;
  iree_device_size_t source_span = 0;
  iree_device_size_t destination_span = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul(row_count - 1, src_pitch,
                                        &source_span) ||
          !iree_device_size_checked_add(source_span, width, &source_span) ||
          !iree_device_size_checked_mul(row_count - 1, dst_pitch,
                                        &destination_span) ||
          !iree_device_size_checked_add(destination_span, width,
                                        &destination_span))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "pitched H2D copy span overflows"));
  }

  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup_range(context, dst, destination_span,
                                                 &dst_ref));
  iree_hal_streaming_buffer_ref_t src_ref;
  iree_status_t src_status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)src, &src_ref);
  if (!iree_status_is_ok(src_status)) {
    iree_status_ignore(src_status);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_UNAVAILABLE,
                             "H2D source requires an owned host staging path"));
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_buffer_ref_validate_range(&src_ref, source_span));
  if (IREE_UNLIKELY(
          !src_ref.buffer || !src_ref.buffer->buffer || !dst_ref.buffer ||
          !dst_ref.buffer->buffer ||
          !iree_any_bit_set((iree_hal_memory_type_t)src_ref.buffer->memory_type,
                            IREE_HAL_MEMORY_TYPE_HOST_LOCAL))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "H2D source is not queue-visible host memory"));
  }

  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);
  if (iree_status_is_ok(status)) {
    // CPU writes to pinned memory must be published before the first transfer
    // packet reads them. One batch barrier avoids a system fence on every row.
    status = iree_hal_streaming_command_buffer_host_to_transfer_barrier(
        stream->command_buffer);
  }
  for (iree_device_size_t row = 0; row < row_count && iree_status_is_ok(status);
       ++row) {
    const iree_hal_streaming_buffer_ref_t row_src_ref = {
        .buffer = src_ref.buffer,
        .offset = src_ref.offset + row * src_pitch,
    };
    const iree_hal_streaming_buffer_ref_t row_dst_ref = {
        .buffer = dst_ref.buffer,
        .offset = dst_ref.offset + row * dst_pitch,
    };
    const iree_hal_buffer_ref_t src_buffer_ref =
        iree_hal_streaming_convert_range_buffer_ref(row_src_ref, width);
    const iree_hal_buffer_ref_t dst_buffer_ref =
        iree_hal_streaming_convert_range_buffer_ref(row_dst_ref, width);
    status = iree_hal_command_buffer_copy_buffer(stream->command_buffer,
                                                 src_buffer_ref, dst_buffer_ref,
                                                 IREE_HAL_COPY_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_command_buffer_barrier(stream->command_buffer);
  }
  iree_slim_mutex_unlock(&stream->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memcpy_device_to_host_2d(
    iree_hal_streaming_context_t* context, void* dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t src_pitch, iree_device_size_t width,
    iree_host_size_t height, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (width == 0 || height == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (width > dst_pitch || width > src_pitch) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "copy width exceeds a row pitch"));
  }
  if (IREE_UNLIKELY((iree_host_size_t)(iree_device_size_t)height != height)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "D2H row count exceeds device address space"));
  }
  iree_device_size_t destination_span = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul((iree_device_size_t)(height - 1),
                                        dst_pitch, &destination_span) ||
          !iree_device_size_checked_add(destination_span, width,
                                        &destination_span))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "D2H destination row address overflows"));
  }

  iree_host_size_t packed_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul((iree_host_size_t)width, height,
                                                &packed_size) ||
                    (iree_device_size_t)packed_size != packed_size)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "packed D2H staging size overflows"));
  }

  iree_host_size_t source_refs_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          height, sizeof(iree_hal_streaming_buffer_ref_t),
          &source_refs_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "D2H row reference array size overflows"));
  }

  iree_hal_streaming_buffer_ref_t* source_refs = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, source_refs_size, (void**)&source_refs);
  for (iree_host_size_t row = 0; row < height && iree_status_is_ok(status);
       ++row) {
    iree_device_size_t source_offset = 0;
    iree_hal_streaming_deviceptr_t row_src = 0;
    if (IREE_UNLIKELY(
            !iree_device_size_checked_mul((iree_device_size_t)row, src_pitch,
                                          &source_offset) ||
            !iree_device_size_checked_add(src, source_offset, &row_src))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "D2H source row address overflows");
      break;
    }
    status = iree_hal_streaming_memory_lookup_range(context, row_src, width,
                                                    &source_refs[row]);
    if (iree_status_is_ok(status) &&
        (!source_refs[row].buffer || !source_refs[row].buffer->buffer ||
         iree_any_bit_set(
             (iree_hal_memory_type_t)source_refs[row].buffer->memory_type,
             IREE_HAL_MEMORY_TYPE_HOST_LOCAL))) {
      iree_status_ignore(status);
      status =
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "D2H source requires the non-batched transfer path");
    }
  }

  iree_hal_streaming_buffer_t* staging = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_memory_allocate_host_with_context_mode(
        context, packed_size, IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT,
        IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*min_alignment=*/64, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
        &staging);
  }

  if (iree_status_is_ok(status)) {
    // The rows have independent packed destinations. Record all copies in one
    // command buffer and add a single completion barrier before the host call.
    iree_slim_mutex_lock(&stream->mutex);
    status = iree_hal_streaming_stream_begin_locked(stream);
    for (iree_host_size_t row = 0; row < height && iree_status_is_ok(status);
         ++row) {
      const iree_device_size_t packed_offset = row * width;
      const iree_hal_buffer_ref_t src_buffer_ref =
          iree_hal_streaming_convert_range_buffer_ref(source_refs[row], width);
      const iree_hal_streaming_buffer_ref_t staging_ref = {
          .buffer = staging,
          .offset = packed_offset,
      };
      const iree_hal_buffer_ref_t dst_buffer_ref =
          iree_hal_streaming_convert_range_buffer_ref(staging_ref, width);
      status = iree_hal_command_buffer_copy_buffer(
          stream->command_buffer, src_buffer_ref, dst_buffer_ref,
          IREE_HAL_COPY_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_streaming_command_buffer_barrier(stream->command_buffer);
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_streaming_buffer_t* callback_staging = staging;
    staging = NULL;
    status = iree_hal_streaming_enqueue_host_d2h_staging_copy(
        stream, dst, dst_pitch, callback_staging, width, height);
  }

  if (staging) {
    iree_status_t sync_status = iree_hal_streaming_stream_synchronize(stream);
    iree_hal_streaming_temporary_host_buffer_free(context, staging);
    status = iree_status_join(status, sync_status);
  }
  iree_allocator_free(context->host_allocator, source_refs);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memcpy_device_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!stream) {
    // Look up buffers from device pointers.
    iree_hal_streaming_buffer_ref_t dst_ref;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
        "resolving `dst` buffer ref %p", (void*)dst);
    iree_hal_streaming_buffer_ref_t src_ref;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
        "resolving `src` buffer ref %p", (void*)src);

    // Transfer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_device_transfer_d2d(
            context->device, src_ref.buffer->buffer, src_ref.offset,
            dst_ref.buffer->buffer, dst_ref.offset, size,
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  } else {
    // Device-to-device copy is the same as memcpy with offset 0.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_memcpy(context, dst, src, size, stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
