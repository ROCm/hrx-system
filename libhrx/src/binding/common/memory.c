// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdatomic.h>

#include "common/graph.h"
#include "common/internal.h"
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
  wrapper->memory_type = memory_type;
  wrapper->host_register_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  wrapper->ipc_handle = NULL;
  wrapper->size = hrx_buf->size;

  // Initialize unified memory attributes.
  wrapper->read_mostly_hint = false;
  wrapper->preferred_location = -2;      // Unspecified initially.
  wrapper->last_prefetch_location = -2;  // Never prefetched.

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
  if (wrapper->buffer && (memory_type & IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
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

  // We need at least a device pointer for the buffer table.
  // For remote HAL buffers the allocator may not support export_buffer;
  // generate a synthetic device pointer so the buffer table can still map
  // this wrapper.
  if (!have_device_ptr && imported_host_ptr) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "registered host allocation did not export a device-visible pointer");
  } else if (!have_device_ptr) {
    static atomic_uintptr_t g_next_synthetic = 0xDEAD000000000000ULL;
    iree_device_size_t aligned_size = 0;
    if (IREE_UNLIKELY(!iree_device_size_checked_mul_add(wrapper->size, 1, 255,
                                                        &aligned_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "buffer size overflows synthetic alignment");
    } else {
      aligned_size &= ~(iree_device_size_t)255;
      uintptr_t synthetic = atomic_fetch_add(&g_next_synthetic, aligned_size);
      wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)synthetic;
      have_device_ptr = true;
    }
  }
  (void)have_host_ptr;

  if (iree_status_is_ok(status)) {
    // Register buffer in context's mapping table.
    status = HRX_CALL(hrx_buffer_table_insert(
        &context->buffer_table, wrapper->device_ptr, wrapper->host_ptr,
        wrapper->size, wrapper->hrx_buf, wrapper));
  }

  if (iree_status_is_ok(status)) {
    *out_wrapper = wrapper;
  } else {
    // Clean up on failure.
    if (wrapper->hrx_buf) {
      hrx_buffer_release(wrapper->hrx_buf);
      wrapper->hrx_buf = NULL;
      wrapper->buffer = NULL;
    }
    iree_hal_streaming_buffer_release_context(wrapper);
    hrx_mem_pool_release(wrapper->allocation_pool);
    iree_allocator_free(context->host_allocator, wrapper);
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

// Frees a buffer wrapper and releases the underlying pyre buffer.
static void iree_hal_streaming_buffer_free(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_allocator_t host_allocator = buffer->context->host_allocator;
  if (buffer->hrx_buf) {
    hrx_buffer_release(buffer->hrx_buf);
    buffer->hrx_buf = NULL;
    buffer->buffer = NULL;
  }
  hrx_mem_pool_release(buffer->allocation_pool);
  buffer->allocation_pool = NULL;
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

void iree_hal_streaming_memory_release_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) return;
  hrx_buffer_table_remove(&buffer->context->buffer_table, buffer->device_ptr);
  iree_hal_streaming_buffer_free(buffer);
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
  // Hot path: this is called in a tight loop during kernel launch to probe
  // every 8-byte word in the kernarg buffer for potential device pointers,
  // so the vast majority of calls are expected to miss. Handle the miss
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
    // Update free memory tracking.
    if (context->device_entry) {
      // Use atomic or lock if needed for thread safety, but for now
      // we match the pattern used in free.
      if (context->device_entry->free_memory >= size) {
        context->device_entry->free_memory -= size;
      } else {
        context->device_entry->free_memory = 0;
      }
    }
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

  (void)flags;
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
    if (context->device_entry) {
      if (context->device_entry->free_memory >= size) {
        context->device_entry->free_memory -= size;
      } else {
        context->device_entry->free_memory = 0;
      }
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
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

  // Calculate pitch with 128-byte alignment for optimal memory access.
  // This is typical for both CUDA and HIP.
  const iree_device_size_t alignment = 128;
  iree_device_size_t pitch = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_mul_add(width_bytes, 1,
                                                      alignment - 1, &pitch))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pitched allocation width overflows");
  }
  pitch = pitch / alignment * alignment;

  // For CUDA, element_size_bytes should be 4, 8, or 16 for coalesced access.
  // We don't enforce this but could warn if needed.

  // Calculate total size.
  iree_device_size_t total_size = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_mul(pitch, height, &total_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
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

iree_status_t iree_hal_streaming_memory_free_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for all submitted work to complete before freeing.
  // We use wait_all_submitted instead of synchronize to avoid flushing
  // other threads' in-progress recordings.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_wait_all_submitted(context));

  // Look up buffer from device pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, HRX_CALL(hrx_buffer_table_find(&context->buffer_table, ptr, NULL,
                                         NULL, (void**)&wrapper)));

  // Remove from mapping table.
  hrx_buffer_table_remove(&context->buffer_table, wrapper->device_ptr);

  // Update free memory tracking.
  if (wrapper->context && wrapper->context->device_entry) {
    wrapper->context->device_entry->free_memory += wrapper->size;
  }

  // Free wrapper.
  iree_hal_streaming_buffer_free(wrapper);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_memory_allocate_host_with_context_mode(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_context_ownership_t context_ownership,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  const iree_host_size_t allocation_size = iree_max(size, (iree_host_size_t)8);
  iree_hal_memory_type_t memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 64,
  };

  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(context->device_allocator, params,
                                             allocation_size, &buffer));

  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap(
      context, buffer, (int)memory_type, /*imported_host_ptr=*/NULL,
      /*allocation_pool=*/NULL, context_ownership, &wrapper);
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status) && wrapper->host_ptr == NULL) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "host allocation did not export a host-visible pointer");
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

iree_status_t iree_hal_streaming_memory_allocate_host(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  return iree_hal_streaming_memory_allocate_host_with_context_mode(
      context, size, flags, IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED,
      out_buffer);
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
          IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED, &staging));
  context->pageable_h2d_staging_buffer = staging;
  context->pageable_h2d_staging_size = size;
  *out_staging = staging;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_free_host(
    iree_hal_streaming_context_t* context, void* ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for all submitted work to complete before freeing.
  // We use wait_all_submitted instead of synchronize to avoid flushing
  // other threads' in-progress recordings.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_wait_all_submitted(context));

  // For host memory, we need to find the buffer by host pointer.
  // Since we store host pointers as device pointers for host allocations,
  // we can look it up directly.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, HRX_CALL(hrx_buffer_table_find(&context->buffer_table,
                                         (uint64_t)(uintptr_t)ptr, NULL, NULL,
                                         (void**)&wrapper)));

  // Remove from mapping table.
  hrx_buffer_table_remove(&context->buffer_table, wrapper->device_ptr);

  // For host allocations (no HAL buffer), free the host pointer directly.
  if (wrapper->buffer == NULL && wrapper->host_ptr != NULL) {
    iree_allocator_free(context->host_allocator, wrapper->host_ptr);
    wrapper->host_ptr = NULL;
  }

  // Free wrapper and release its context ownership edge.
  iree_hal_streaming_buffer_free(wrapper);

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
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_import_buffer(
              context->device_allocator, params, &external_buffer,
              iree_hal_buffer_release_callback_null(), &buffer));

  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap(
      context, buffer, (int)params.type, ptr, /*allocation_pool=*/NULL,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_RETAINED, &wrapper);
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status)) {
    wrapper->host_register_flags = flags;
    *out_buffer = wrapper;
  } else {
    iree_hal_streaming_buffer_free(wrapper);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_unregister_host(
    iree_hal_streaming_context_t* context, void* ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Synchronize context to ensure all operations using this memory complete.
  // This matches CUDA/HIP behavior where unregistration is blocking.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_synchronize(context));

  // Look up buffer from host pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, HRX_CALL(hrx_buffer_table_find(&context->buffer_table,
                                         (uint64_t)(uintptr_t)ptr, NULL, NULL,
                                         (void**)&wrapper)));

  // Remove from buffer table.
  hrx_buffer_table_remove(&context->buffer_table, (uint64_t)(uintptr_t)ptr);

  // Free wrapper (this will release the HAL buffer and context references).
  iree_hal_streaming_buffer_free(wrapper);

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
  *out_size = wrapper->size;

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_host_flags(
    iree_hal_streaming_context_t* context, void* ptr,
    iree_hal_streaming_host_register_flags_t* out_flags) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_flags);
  *out_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;

  // Look up buffer from host pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = HRX_CALL(
      hrx_buffer_table_find(&context->buffer_table, (uint64_t)(uintptr_t)ptr,
                            NULL, NULL, (void**)&wrapper));
  if (iree_status_is_ok(status)) {
    *out_flags = wrapper->host_register_flags;
  }

  return status;
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
        z0, iree_hal_streaming_graph_add_memset_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, dst, pattern_value,
                pattern_length, length / pattern_length, &node));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_set_last_node(stream, node));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up buffer from device pointer.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);

  // Check if we have a HAL buffer. For system-allocated memory
  // (hipHostMalloc), there's no HAL buffer - use direct memset.
  if (!dst_ref.buffer->buffer) {
    // Direct memset for host memory.
    if (dst_ref.buffer->host_ptr) {
      uint8_t* dest = (uint8_t*)dst_ref.buffer->host_ptr + dst_ref.offset;
      if (pattern_length == 1) {
        memset(dest, *(const uint8_t*)pattern, length);
      } else {
        // For larger patterns, repeat the pattern.
        for (iree_device_size_t i = 0; i < length; i += pattern_length) {
          iree_device_size_t copy_size = iree_min(pattern_length, length - i);
          memcpy(dest + i, pattern, copy_size);
        }
      }
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "no buffer available for memset"));
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
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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
        z0, iree_hal_streaming_graph_add_memcpy_node(
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

    iree_hal_streaming_graph_node_t* copy_deps[] = {callback_node};
    iree_hal_streaming_graph_node_t* copy_node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_memcpy_node(
                stream->capture_graph, copy_deps, IREE_ARRAYSIZE(copy_deps),
                dst, staging->device_ptr, size, &copy_node));
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

  // Host allocations registered through hipHostMalloc are not guaranteed to be
  // backed by AMDGPU command-buffer-compatible allocations. Use the blocking
  // transfer path for host sources until registered host buffers carry the
  // required lifetime and backing-storage contract.
  if (stream) {
    iree_status_t flush_status = iree_hal_streaming_stream_flush(stream);
    if (!iree_status_is_ok(flush_status)) {
      IREE_TRACE_ZONE_END(z0);
      return flush_status;
    }
  }

  const iree_device_size_t staging_threshold = 256 * 1024;
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
      iree_slim_mutex_lock(&copy_stream->mutex);
      status = iree_hal_streaming_stream_begin_locked(copy_stream);
      if (iree_status_is_ok(status)) {
        iree_hal_streaming_buffer_ref_t staging_ref = {
            .buffer = staging,
            .offset = 0,
        };
        iree_hal_buffer_ref_t src_buffer_ref =
            iree_hal_streaming_convert_range_buffer_ref(staging_ref, size);
        iree_hal_buffer_ref_t dst_buffer_ref =
            iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
        status = iree_hal_command_buffer_copy_buffer(
            copy_stream->command_buffer, src_buffer_ref, dst_buffer_ref,
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
    iree_slim_mutex_unlock(&context->mutex);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  } else {
    // Small host-to-device transfers are faster through the direct blocking
    // path than paying temporary host-visible allocation overhead.
    const iree_device_size_t h2d_chunk_size = 63 * 1024;
    const uint8_t* src_ptr = (const uint8_t*)src;
    iree_device_size_t remaining = size;
    iree_device_size_t chunk_offset = 0;
    while (remaining > 0) {
      iree_device_size_t this_chunk =
          remaining < h2d_chunk_size ? remaining : h2d_chunk_size;
      iree_status_t chunk_status = iree_hal_device_transfer_h2d(
          context->device, src_ptr + chunk_offset, dst_ref.buffer->buffer,
          dst_ref.offset + chunk_offset, this_chunk,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
      IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, chunk_status);
      chunk_offset += this_chunk;
      remaining -= this_chunk;
    }
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
        z0, iree_hal_streaming_graph_add_memcpy_node(
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

  // Host allocations registered through hipHostMalloc are not guaranteed to be
  // backed by AMDGPU command-buffer-compatible allocations. Use the blocking
  // transfer path for host destinations until registered host buffers carry the
  // required lifetime and backing-storage contract.
  if (stream) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_stream_synchronize(stream));
  }

  const iree_device_size_t d2h_chunk_size = 4 * 1024 * 1024;
  uint8_t* dst_ptr = (uint8_t*)dst;
  iree_device_size_t remaining = size;
  iree_device_size_t chunk_offset = 0;
  iree_status_t direct_status = iree_ok_status();
  while (remaining > 0 && iree_status_is_ok(direct_status)) {
    iree_device_size_t this_chunk =
        remaining < d2h_chunk_size ? remaining : d2h_chunk_size;
    direct_status = iree_hal_device_transfer_d2h(
        context->device, src_ref.buffer->buffer, src_ref.offset + chunk_offset,
        dst_ptr + chunk_offset, this_chunk,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    chunk_offset += this_chunk;
    remaining -= this_chunk;
  }
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
          IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED, &staging);
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
