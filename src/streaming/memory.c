// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "streaming/internal.h"
#include "streaming/util/buffer_table.h"

#include <stdatomic.h>

#include "iree/hal/buffer_transfer.h"

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

// Wraps a HAL buffer in a stream buffer and caches information.
static iree_status_t
iree_hal_streaming_buffer_wrap(iree_hal_streaming_context_t *context,
                               iree_hal_buffer_t *buffer, int memory_type,
                               iree_hal_streaming_buffer_t **out_wrapper) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_wrapper);
  *out_wrapper = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_buffer_t *wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(context->host_allocator, sizeof(*wrapper),
                                (void **)&wrapper));

  // Initialize wrapper.
  wrapper->buffer = buffer;
  iree_hal_buffer_retain(buffer);
  wrapper->context = context;
  iree_hal_streaming_context_retain(context);
  wrapper->memory_type = memory_type;
  wrapper->host_register_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  wrapper->ipc_handle = NULL;
  wrapper->size = iree_hal_buffer_byte_length(buffer);

  // Initialize unified memory attributes.
  wrapper->read_mostly_hint = false;
  wrapper->preferred_location = -2;     // Unspecified initially.
  wrapper->last_prefetch_location = -2; // Never prefetched.

  iree_hal_external_buffer_t external_ptr;
  iree_status_t status = iree_ok_status();
  bool have_device_ptr = false;
  bool have_host_ptr = false;

  // Try to export as device allocation (works for device-local memory
  // and mapped host memory).
  iree_status_t device_status = iree_hal_allocator_export_buffer(
      context->device_allocator, buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_ptr);
  if (iree_status_is_ok(device_status)) {
    wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)
                              external_ptr.handle.device_allocation.ptr;
    have_device_ptr = true;
  } else {
    iree_status_ignore(device_status);
  }

  // For host-local memory, also export as host allocation.
  // This is needed for hipHostMalloc which returns host pointers.
  if (memory_type & IREE_HAL_MEMORY_TYPE_HOST_LOCAL) {
    iree_status_t host_status = iree_hal_allocator_export_buffer(
        context->device_allocator, buffer,
        IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_ptr);
    if (iree_status_is_ok(host_status)) {
      wrapper->host_ptr = (void *)external_ptr.handle.host_allocation.ptr;
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

  // We need at least a device pointer for the buffer table.
  // For remote HAL buffers the allocator may not support export_buffer;
  // generate a synthetic device pointer so the buffer table can still map
  // this wrapper.
  if (!have_device_ptr) {
    static atomic_uintptr_t g_next_synthetic =
        ATOMIC_VAR_INIT(0xDEAD000000000000ULL);
    iree_device_size_t buf_size = iree_hal_buffer_byte_length(buffer);
    iree_device_size_t aligned_size =
        (buf_size + 255) & ~(iree_device_size_t)255;
    uintptr_t synthetic = atomic_fetch_add(&g_next_synthetic, aligned_size);
    wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)synthetic;
    have_device_ptr = true;
  }

  if (iree_status_is_ok(status)) {
    // Register buffer in context's mapping table.
    status =
        iree_hal_streaming_buffer_table_insert(context->buffer_table, wrapper);
  }

  if (iree_status_is_ok(status)) {
    *out_wrapper = wrapper;
  } else {
    // Clean up on failure.
    if (wrapper->buffer) {
      iree_hal_buffer_release(wrapper->buffer);
    }
    iree_hal_streaming_context_release(wrapper->context);
    iree_allocator_free(context->host_allocator, wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Frees a buffer wrapper and releases the underlying buffer.
static void
iree_hal_streaming_buffer_free(iree_hal_streaming_buffer_t *buffer) {
  if (!buffer)
    return;
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_allocator_t host_allocator = buffer->context->host_allocator;
  // Release HAL buffer if present (may be NULL for registered/host memory).
  if (buffer->buffer) {
    iree_hal_buffer_release(buffer->buffer);
  }
  iree_hal_streaming_context_release(buffer->context);
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

iree_hal_streaming_deviceptr_t
iree_hal_streaming_buffer_device_pointer(iree_hal_streaming_buffer_t *buffer) {
  return buffer ? buffer->device_ptr : 0;
}

iree_status_t
iree_hal_streaming_memory_lookup(iree_hal_streaming_context_t *context,
                                 iree_hal_streaming_deviceptr_t device_ptr,
                                 iree_hal_streaming_buffer_ref_t *out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_buffer_table_lookup(
      context->buffer_table, device_ptr, &out_ref->buffer));
  // TODO(benvanik): make the buffer table return a ref so we can hide this?
  if (IREE_LIKELY(out_ref->buffer->device_ptr <= device_ptr &&
                  device_ptr <
                      out_ref->buffer->device_ptr + out_ref->buffer->size)) {
    out_ref->offset = (iree_device_size_t)device_ptr -
                      (iree_device_size_t)out_ref->buffer->device_ptr;
  } else {
    out_ref->offset = (iree_device_size_t)device_ptr -
                      (iree_device_size_t)out_ref->buffer->host_ptr;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_lookup_range(
    iree_hal_streaming_context_t *context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_ref_t *out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_buffer_table_lookup_range(
      context->buffer_table, device_ptr, size, &out_ref->buffer));
  // TODO(benvanik): make the buffer table return a ref so we can hide this?
  if (IREE_LIKELY(out_ref->buffer->device_ptr <= device_ptr &&
                  device_ptr <
                      out_ref->buffer->device_ptr + out_ref->buffer->size)) {
    out_ref->offset = (iree_device_size_t)device_ptr -
                      (iree_device_size_t)out_ref->buffer->device_ptr;
  } else {
    out_ref->offset = (iree_device_size_t)device_ptr -
                      (iree_device_size_t)out_ref->buffer->host_ptr;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_device(
    iree_hal_streaming_context_t *context, iree_device_size_t size,
    iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t **out_buffer) {
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
  iree_hal_buffer_t *buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(context->device_allocator, params,
                                             size, &buffer));

  // Wrap in stream buffer.
  iree_hal_streaming_buffer_t *wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap(
      context, buffer, (int)memory_type, &wrapper);

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

iree_status_t iree_hal_streaming_memory_allocate_device_pitched(
    iree_hal_streaming_context_t *context, iree_device_size_t width_bytes,
    iree_device_size_t height, iree_device_size_t element_size_bytes,
    iree_device_size_t *out_pitch, iree_hal_streaming_buffer_t **out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_pitch);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_pitch = 0;
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate pitch with 128-byte alignment for optimal memory access.
  // This is typical for both CUDA and HIP.
  const iree_device_size_t alignment = 128;
  iree_device_size_t pitch =
      (width_bytes + alignment - 1) / alignment * alignment;

  // For CUDA, element_size_bytes should be 4, 8, or 16 for coalesced access.
  // We don't enforce this but could warn if needed.

  // Calculate total size.
  iree_device_size_t total_size = pitch * height;

  // Allocate the buffer with the calculated total size.
  iree_hal_streaming_buffer_t *buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device(
      context, total_size, 0, &buffer);

  if (iree_status_is_ok(status)) {
    *out_pitch = pitch;
    *out_buffer = buffer;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t
iree_hal_streaming_memory_free_device(iree_hal_streaming_context_t *context,
                                      iree_hal_streaming_deviceptr_t ptr) {
  if (!ptr)
    return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for all submitted work to complete before freeing.
  // We use wait_all_submitted instead of synchronize to avoid flushing
  // other threads' in-progress recordings.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_wait_all_submitted(context));

  // Look up buffer from device pointer.
  iree_hal_streaming_buffer_t *wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_buffer_table_lookup(context->buffer_table, ptr,
                                                 &wrapper));

  // Remove from mapping table.
  if (wrapper->context && wrapper->context->buffer_table) {
    iree_hal_streaming_buffer_table_remove(wrapper->context->buffer_table,
                                           wrapper->device_ptr);
  }

  // Update free memory tracking.
  if (wrapper->context && wrapper->context->device_entry) {
    wrapper->context->device_entry->free_memory += wrapper->size;
  }

  // Free wrapper.
  iree_hal_streaming_buffer_free(wrapper);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_host(
    iree_hal_streaming_context_t *context, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t **out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  // For host memory allocation, we use a simpler approach:
  // Allocate wrapper directly with system memory.
  // This provides page-locked-like behavior for the streaming layer.
  iree_hal_streaming_buffer_t *wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(context->host_allocator, sizeof(*wrapper),
                                (void **)&wrapper));

  // Allocate the host memory with alignment.
  void *host_ptr = NULL;
  iree_status_t status =
      iree_allocator_malloc(context->host_allocator, size, &host_ptr);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(context->host_allocator, wrapper);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Initialize wrapper for host memory.
  memset(wrapper, 0, sizeof(*wrapper));
  wrapper->buffer = NULL; // No HAL buffer for host allocations.
  wrapper->context = context;
  iree_hal_streaming_context_retain(context);
  wrapper->memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  wrapper->host_register_flags = flags;
  wrapper->size = size;
  wrapper->host_ptr = host_ptr;
  wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)host_ptr;
  wrapper->read_mostly_hint = false;
  wrapper->preferred_location = -2;
  wrapper->last_prefetch_location = -2;

  // Register in buffer table using host pointer as key.
  status =
      iree_hal_streaming_buffer_table_insert(context->buffer_table, wrapper);

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
  } else {
    iree_allocator_free(context->host_allocator, host_ptr);
    iree_hal_streaming_context_release(context);
    iree_allocator_free(context->host_allocator, wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t
iree_hal_streaming_memory_free_host(iree_hal_streaming_context_t *context,
                                    void *ptr) {
  if (!ptr)
    return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for all submitted work to complete before freeing.
  // We use wait_all_submitted instead of synchronize to avoid flushing
  // other threads' in-progress recordings.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_wait_all_submitted(context));

  // For host memory, we need to find the buffer by host pointer.
  // Since we store host pointers as device pointers for host allocations,
  // we can look it up directly.
  iree_hal_streaming_buffer_t *wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_streaming_buffer_table_lookup(
                                            context->buffer_table,
                                            (iree_hal_streaming_deviceptr_t)ptr,
                                            &wrapper));

  // Remove from mapping table.
  if (wrapper->context && wrapper->context->buffer_table) {
    iree_hal_streaming_buffer_table_remove(wrapper->context->buffer_table,
                                           wrapper->device_ptr);
  }

  // For host allocations (no HAL buffer), free the host pointer directly.
  if (wrapper->buffer == NULL && wrapper->host_ptr != NULL) {
    iree_allocator_free(context->host_allocator, wrapper->host_ptr);
    wrapper->host_ptr = NULL;
  }

  // Free wrapper (this handles HAL buffer release if present).
  iree_hal_streaming_context_t *ctx = wrapper->context;
  if (wrapper->buffer) {
    iree_hal_buffer_release(wrapper->buffer);
  }
  iree_hal_streaming_context_release(ctx);
  iree_allocator_free(context->host_allocator, wrapper);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_register_host(
    iree_hal_streaming_context_t *context, void *ptr, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t **out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // For host registration, we simply track the existing host memory.
  // We don't actually need to import it through HAL since we're just
  // registering user-provided memory for use with the streaming layer.
  iree_hal_streaming_buffer_t *wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(context->host_allocator, sizeof(*wrapper),
                                (void **)&wrapper));

  // Initialize wrapper for registered host memory.
  memset(wrapper, 0, sizeof(*wrapper));
  wrapper->buffer = NULL; // No HAL buffer - we don't own this memory.
  wrapper->context = context;
  iree_hal_streaming_context_retain(context);
  wrapper->memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  wrapper->host_register_flags = flags;
  wrapper->size = size;
  wrapper->host_ptr = ptr;
  wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)ptr;
  wrapper->read_mostly_hint = false;
  wrapper->preferred_location = -2;
  wrapper->last_prefetch_location = -2;

  // Register in buffer table using host pointer as key.
  iree_status_t status =
      iree_hal_streaming_buffer_table_insert(context->buffer_table, wrapper);

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
  } else {
    iree_hal_streaming_context_release(context);
    iree_allocator_free(context->host_allocator, wrapper);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t
iree_hal_streaming_memory_unregister_host(iree_hal_streaming_context_t *context,
                                          void *ptr) {
  if (!ptr)
    return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Synchronize context to ensure all operations using this memory complete.
  // This matches CUDA/HIP behavior where unregistration is blocking.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_synchronize(context));

  // Look up buffer from host pointer.
  iree_hal_streaming_buffer_t *wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_streaming_buffer_table_lookup(
                                            context->buffer_table,
                                            (iree_hal_streaming_deviceptr_t)ptr,
                                            &wrapper));

  // Remove from buffer table.
  if (wrapper->context && wrapper->context->buffer_table) {
    iree_hal_streaming_buffer_table_remove(wrapper->context->buffer_table,
                                           (iree_hal_streaming_deviceptr_t)ptr);
  }

  // Free wrapper (this will release the HAL buffer and context references).
  iree_hal_streaming_buffer_free(wrapper);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_address_range(
    iree_hal_streaming_context_t *context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_deviceptr_t *out_base, iree_device_size_t *out_size) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_base);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_base = 0;
  *out_size = 0;

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_t *wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_table_lookup(
      context->buffer_table, ptr, &wrapper);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // For registered host memory, the base is the registered pointer.
  // For device memory, the base is the allocated device pointer.
  if (wrapper->host_ptr) {
    *out_base = (iree_hal_streaming_deviceptr_t)wrapper->host_ptr;
  } else {
    *out_base = wrapper->device_ptr;
  }
  *out_size = wrapper->size;

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_host_flags(
    iree_hal_streaming_context_t *context, void *ptr,
    iree_hal_streaming_host_register_flags_t *out_flags) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_flags);
  *out_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;

  // Look up buffer from host pointer.
  iree_hal_streaming_buffer_t *wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_table_lookup(
      context->buffer_table, (iree_hal_streaming_deviceptr_t)ptr, &wrapper);
  if (iree_status_is_ok(status)) {
    *out_flags = wrapper->host_register_flags;
  }

  return status;
}

iree_status_t iree_hal_streaming_memory_memset(
    iree_hal_streaming_context_t *context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, const void *pattern,
    iree_host_size_t pattern_length, iree_hal_streaming_stream_t *stream) {
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
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_memset_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, dst, pattern_value,
                pattern_length, length / pattern_length, NULL));
    // Clear dependencies after adding the node.
    stream->capture_dependency_count = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up buffer from device pointer.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void *)dst);

  // Check if we have a HAL buffer. For system-allocated memory
  // (hipHostMalloc), there's no HAL buffer - use direct memset.
  if (!dst_ref.buffer->buffer) {
    // Direct memset for host memory.
    if (dst_ref.buffer->host_ptr) {
      uint8_t *dest = (uint8_t *)dst_ref.buffer->host_ptr + dst_ref.offset;
      if (pattern_length == 1) {
        memset(dest, *(const uint8_t *)pattern, length);
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

  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_begin(stream));
  }

  // Record fill command.
  iree_hal_buffer_ref_t target_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_command_buffer_fill_buffer(
                                            stream->command_buffer, target_ref,
                                            pattern, pattern_length,
                                            IREE_HAL_FILL_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_memcpy(
    iree_hal_streaming_context_t *context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Add memcpy node to the graph instead of recording to command buffer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_memcpy_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, dst, src, size, NULL));
    // Clear dependencies after adding the node.
    stream->capture_dependency_count = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Look up buffers from device pointers.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void *)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
      "resolving `src` buffer ref %p", (void *)src);

  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_begin(stream));
  }

  // Record copy command.
  iree_hal_buffer_ref_t src_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
  iree_hal_buffer_ref_t dst_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_copy_buffer(stream->command_buffer,
                                              src_buffer_ref, dst_buffer_ref,
                                              IREE_HAL_COPY_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t
iree_hal_streaming_memcpy_peer(iree_hal_streaming_context_t *dst_context,
                               iree_hal_streaming_deviceptr_t dst,
                               iree_hal_streaming_context_t *src_context,
                               iree_hal_streaming_deviceptr_t src,
                               iree_device_size_t size,
                               iree_hal_streaming_stream_t *stream) {
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
      "resolving `dst` buffer ref %p", (void *)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(src_context, src, &src_ref),
      "resolving `src` buffer ref %p", (void *)src);

  // Ensure command buffer is available.
  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_begin(stream));
  }

  // Record copy command.
  iree_hal_buffer_ref_t src_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
  iree_hal_buffer_ref_t dst_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_copy_buffer(stream->command_buffer,
                                              src_buffer_ref, dst_buffer_ref,
                                              IREE_HAL_COPY_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Memory copy helper functions
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_memcpy_host_to_device(
    iree_hal_streaming_context_t *context, iree_hal_streaming_deviceptr_t dst,
    const void *src, iree_device_size_t size,
    iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Look up destination buffer from device pointer.
  iree_hal_streaming_buffer_ref_t dst_ref;
  iree_status_t dst_status =
      iree_hal_streaming_memory_lookup(context, dst, &dst_ref);

  // If destination is not in our buffer table, try raw device transfer.
  // This handles global symbols from loaded modules.
  if (!iree_status_is_ok(dst_status)) {
    iree_status_ignore(dst_status);
    // Attempt raw H2D transfer (synchronous).
    iree_status_t raw_status = iree_hal_device_transfer_h2d_raw(
        context->device, src, (uint64_t)dst, size, iree_infinite_timeout());
    IREE_TRACE_ZONE_END(z0);
    return raw_status;
  }

  // Check if src is pinned host memory backed by a HAL buffer.
  // For true async transfers, we need a HAL buffer to record commands.
  // Memory allocated via hipHostMalloc using system malloc does not have
  // a HAL buffer, so we use synchronous transfer for those cases.
  bool src_is_pinned = false;
  if (stream) {
    iree_hal_streaming_buffer_ref_t src_ref;
    iree_status_t src_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)src, &src_ref);
    // Only consider it pinned if there's a backing HAL buffer that we can
    // use for async copy commands. System-allocated memory uses sync path.
    src_is_pinned = iree_status_is_ok(src_status) && src_ref.buffer &&
                    src_ref.buffer->buffer != NULL;
    iree_status_ignore(src_status);
  }

  // For host-to-device, we can use the HAL update command if stream is NULL,
  // or copy command if stream is provided AND source is pinned memory.
  // Pageable memory requires synchronous transfer.
  if (!stream || !src_is_pinned) {
    // Flush pending dispatches before the H2D transfer.
    if (stream && stream->command_buffer) {
      iree_status_t flush_status = iree_hal_streaming_stream_flush(stream);
      if (!iree_status_is_ok(flush_status)) {
        IREE_TRACE_ZONE_END(z0);
        return flush_status;
      }
    }

    // Chunk large transfers into pieces that fit within the inline
    // BUFFER_UPDATE path (<=64KB per chunk). The remote HAL wire format
    // uses uint16_t for command length (max 65535). The BUFFER_UPDATE
    // header is 40 bytes, so max payload is ~65495. Use 63KB to be safe.
    const iree_device_size_t h2d_chunk_size = 63 * 1024;
    const uint8_t *src_ptr = (const uint8_t *)src;
    iree_device_size_t remaining = size;
    iree_device_size_t chunk_offset = 0;

    // Use synchronous iree_hal_device_transfer_h2d for both hipMemcpy
    // (stream==NULL) and hipMemcpyAsync with pageable host memory.
    // Pageable memory can't do true async DMA; the HIP spec falls back
    // to synchronous copy. Using the direct transfer path avoids flooding
    // the remote HAL's async buffer pool (32 slots) with thousands of
    // BUFFER_UPDATE submissions for large tensors.
    // Ordering is guaranteed because we already flushed pending dispatches
    // above, and iree_hal_device_transfer_h2d blocks until each chunk
    // completes.
    {
      while (remaining > 0) {
        iree_device_size_t this_chunk =
            remaining < h2d_chunk_size ? remaining : h2d_chunk_size;
        iree_status_t chunk_status = iree_hal_device_transfer_h2d(
            context->device, src_ptr + chunk_offset, dst_ref.buffer->buffer,
            dst_ref.offset + chunk_offset, this_chunk,
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
        if (!iree_status_is_ok(chunk_status)) {
          fprintf(
              stderr,
              "[H2D] chunk %d failed: buf_offset=%" PRIu64
              " chunk_offset=%" PRIu64 " size=%" PRIu64 " total=%" PRIu64
              " buf_len=%" PRIu64 "\n",
              (int)(chunk_offset / h2d_chunk_size), (uint64_t)dst_ref.offset,
              (uint64_t)chunk_offset, (uint64_t)this_chunk, (uint64_t)size,
              (uint64_t)iree_hal_buffer_byte_length(dst_ref.buffer->buffer));
          iree_status_fprint(stderr, chunk_status);
          fprintf(stderr, "\n");
          IREE_TRACE_ZONE_END(z0);
          return chunk_status;
        }
        chunk_offset += this_chunk;
        remaining -= this_chunk;
      }
    }
  } else {
    // Async: create a host buffer view and copy via command buffer.
    // Only used when source is pinned memory.
    if (!stream->command_buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_stream_begin(stream));
    }

    // Import host memory as external buffer.
    iree_hal_external_buffer_t external_buffer = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
        .size = size,
        .handle =
            {
                .host_allocation =
                    {
                        .ptr = (void *)src,
                    },
            },
    };

    iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .access = IREE_HAL_MEMORY_ACCESS_READ,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        .queue_affinity = stream->queue_affinity,
        .min_alignment = 1,
    };

    iree_hal_buffer_t *src_buffer = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_allocator_import_buffer(
                context->device_allocator, params, &external_buffer,
                iree_hal_buffer_release_callback_null(), &src_buffer));

    // Record copy command.
    iree_hal_buffer_ref_t src_buffer_ref =
        iree_hal_make_buffer_ref(src_buffer, 0, size);
    iree_hal_buffer_ref_t dst_buffer_ref =
        iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
    iree_status_t status = iree_hal_command_buffer_copy_buffer(
        stream->command_buffer, src_buffer_ref, dst_buffer_ref,
        IREE_HAL_COPY_FLAG_NONE);

    iree_hal_buffer_release(src_buffer);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_device_to_host(
    iree_hal_streaming_context_t *context, void *dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Look up source buffer from device pointer.
  iree_hal_streaming_buffer_ref_t src_ref;
  iree_status_t src_status =
      iree_hal_streaming_memory_lookup(context, src, &src_ref);

  // If source is not in our buffer table, try raw device transfer.
  // This handles global symbols from loaded modules.
  if (!iree_status_is_ok(src_status)) {
    iree_status_ignore(src_status);
    // Attempt raw D2H transfer (synchronous).
    iree_status_t raw_status = iree_hal_device_transfer_d2h_raw(
        context->device, (uint64_t)src, dst, size, iree_infinite_timeout());
    IREE_TRACE_ZONE_END(z0);
    return raw_status;
  }

  // Check if dst is pinned host memory backed by a HAL buffer.
  // For true async transfers, we need a HAL buffer to record commands.
  // Memory allocated via hipHostMalloc using system malloc does not have
  // a HAL buffer, so we use synchronous transfer for those cases.
  bool dst_is_pinned = false;
  if (stream) {
    iree_hal_streaming_buffer_ref_t dst_ref;
    iree_status_t dst_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)dst, &dst_ref);
    // Only consider it pinned if there's a backing HAL buffer.
    dst_is_pinned = iree_status_is_ok(dst_status) && dst_ref.buffer &&
                    dst_ref.buffer->buffer != NULL;
    iree_status_ignore(dst_status);
  }

  // For device-to-host, we can use the HAL transfer command if stream is
  // NULL, or copy command if stream is provided AND destination is pinned.
  // Pageable memory requires synchronous transfer.
  if (!stream || !dst_is_pinned) {
    // Synchronous transfer - used for NULL stream or pageable host memory.
    // If a stream is provided, we must flush it first to ensure any pending
    // operations (like memset) that write to the source buffer are completed
    // before we read from it.
    if (stream) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_stream_synchronize(stream));
    }
    // Chunk large transfers to avoid staging buffer map responses that
    // exceed the remote HAL receive buffer size. Each chunk allocates a
    // temporary staging buffer, copies device→staging, then maps the
    // staging buffer to pull data from the server. Keeping chunks small
    // ensures the map response fits within the receive buffer.
    const iree_device_size_t d2h_chunk_size = 4 * 1024 * 1024; // 4MB
    uint8_t *dst_ptr = (uint8_t *)dst;
    iree_device_size_t remaining = size;
    iree_device_size_t chunk_offset = 0;
    while (remaining > 0) {
      iree_device_size_t this_chunk =
          remaining < d2h_chunk_size ? remaining : d2h_chunk_size;
      iree_status_t chunk_status = iree_hal_device_transfer_d2h(
          context->device, src_ref.buffer->buffer,
          src_ref.offset + chunk_offset, dst_ptr + chunk_offset, this_chunk,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
      if (!iree_status_is_ok(chunk_status)) {
        IREE_TRACE_ZONE_END(z0);
        return chunk_status;
      }
      chunk_offset += this_chunk;
      remaining -= this_chunk;
    }
  } else {
    // Async: create a host buffer view and copy via command buffer.
    // Only used when destination is pinned memory.
    if (!stream->command_buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_stream_begin(stream));
    }

    // Import host memory as external buffer.
    iree_hal_external_buffer_t external_buffer = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
        .size = size,
        .handle =
            {
                .host_allocation =
                    {
                        .ptr = dst,
                    },
            },
    };

    iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .access = IREE_HAL_MEMORY_ACCESS_WRITE,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        .queue_affinity = stream->queue_affinity,
        .min_alignment = 1,
    };

    iree_hal_buffer_t *dst_buffer = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_allocator_import_buffer(
                context->device_allocator, params, &external_buffer,
                iree_hal_buffer_release_callback_null(), &dst_buffer));

    // Record copy command.
    iree_hal_buffer_ref_t src_buffer_ref =
        iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
    iree_hal_buffer_ref_t dst_buffer_ref =
        iree_hal_make_buffer_ref(dst_buffer, 0, size);
    iree_status_t status = iree_hal_command_buffer_copy_buffer(
        stream->command_buffer, src_buffer_ref, dst_buffer_ref,
        IREE_HAL_COPY_FLAG_NONE);

    iree_hal_buffer_release(dst_buffer);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_device_to_device(
    iree_hal_streaming_context_t *context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t *stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!stream) {
    // Look up buffers from device pointers.
    iree_hal_streaming_buffer_ref_t dst_ref;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
        "resolving `dst` buffer ref %p", (void *)dst);
    iree_hal_streaming_buffer_ref_t src_ref;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
        "resolving `src` buffer ref %p", (void *)src);

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
