// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "common/internal.h"
#include "iree/hal/buffer_transfer.h"

//===----------------------------------------------------------------------===//
// ROCm device runtime state
//===----------------------------------------------------------------------===//

// ROCm device libraries expect hidden_heap_v1 to point at a heap header with
// fixed recordable-kind slots and an initial slab window. Streaming owns that
// HIP runtime policy; HAL drivers only expose the metadata slot and patch the
// supplied bytes into native launch arguments.
#define IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE (128 * 1024)
#define IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT 16
#define IREE_HAL_STREAMING_ROCM_MALLOC_NUM_SDATA 256
#define IREE_HAL_STREAMING_ROCM_MALLOC_RECORDABLE_OFFSET 4096
#define IREE_HAL_STREAMING_ROCM_MALLOC_RECORDABLE_STRIDE 128
#define IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLABS_OFFSET 108544
#define IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLAB_SIZE (2 * 1024 * 1024)
static_assert(IREE_HAL_STREAMING_ROCM_MALLOC_RECORDABLE_OFFSET +
                      IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT *
                          IREE_HAL_STREAMING_ROCM_MALLOC_RECORDABLE_STRIDE <=
                  IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE,
              "device malloc recordable table must fit in heap header");
static_assert(IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLABS_OFFSET +
                      3 * sizeof(uint64_t) <=
                  IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE,
              "device malloc initial slab table must fit in heap header");

static void iree_hal_streaming_rocm_store_u32(uint8_t* storage,
                                              iree_host_size_t offset,
                                              uint32_t value) {
  memcpy(storage + offset, &value, sizeof(value));
}

static void iree_hal_streaming_rocm_store_u64(uint8_t* storage,
                                              iree_host_size_t offset,
                                              uint64_t value) {
  memcpy(storage + offset, &value, sizeof(value));
}

static iree_status_t iree_hal_streaming_rocm_populate_malloc_heap_header(
    void* heap_storage, uint64_t initial_slabs_device_ptr,
    iree_device_size_t initial_slab_bytes) {
  if (IREE_UNLIKELY(initial_slab_bytes >
                    UINT64_MAX - initial_slabs_device_ptr)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ROCm device malloc initial slab range overflow");
  }

  uint8_t* heap = (uint8_t*)heap_storage;
  memset(heap, 0, IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE);

  for (uint32_t i = 0; i < IREE_HAL_STREAMING_ROCM_MALLOC_KIND_COUNT; ++i) {
    const iree_host_size_t offset =
        IREE_HAL_STREAMING_ROCM_MALLOC_RECORDABLE_OFFSET +
        (iree_host_size_t)i * IREE_HAL_STREAMING_ROCM_MALLOC_RECORDABLE_STRIDE;
    iree_hal_streaming_rocm_store_u32(heap, offset,
                                      IREE_HAL_STREAMING_ROCM_MALLOC_NUM_SDATA);
  }

  const uint64_t initial_slabs_start = initial_slabs_device_ptr;
  const uint64_t initial_slabs_end =
      initial_slabs_start + (uint64_t)initial_slab_bytes;
  iree_hal_streaming_rocm_store_u64(
      heap, IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLABS_OFFSET,
      initial_slabs_start);
  iree_hal_streaming_rocm_store_u64(
      heap,
      IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLABS_OFFSET + sizeof(uint64_t),
      initial_slabs_end);
  iree_hal_streaming_rocm_store_u64(
      heap,
      IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLABS_OFFSET +
          2 * sizeof(uint64_t),
      initial_slabs_start);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_rocm_allocate_device_buffer(
    iree_hal_streaming_context_t* context, iree_hal_buffer_usage_t usage,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer,
    uint64_t* out_device_ptr) {
  *out_buffer = NULL;
  *out_device_ptr = 0;

  iree_hal_buffer_params_t params = {
      .usage = usage | IREE_HAL_BUFFER_USAGE_SHARING_EXPORT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = context->queue_affinity,
  };

  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      context->device_allocator, params, allocation_size, &buffer));

  iree_hal_external_buffer_t external_buffer;
  iree_status_t status = iree_hal_allocator_export_buffer(
      context->device_allocator, buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer);
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
    *out_device_ptr = external_buffer.handle.device_allocation.ptr;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t
iree_hal_streaming_context_initialize_rocm_device_runtime_locked(
    iree_hal_streaming_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_device_size_t requested_heap_size =
      (iree_device_size_t)context->limits.malloc_heap_size;
  const iree_device_size_t initial_slab_count = iree_device_size_ceil_div(
      iree_max(
          requested_heap_size,
          (iree_device_size_t)IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLAB_SIZE),
      IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLAB_SIZE);
  iree_device_size_t initial_slab_bytes = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_mul(
          initial_slab_count,
          (iree_device_size_t)IREE_HAL_STREAMING_ROCM_MALLOC_INITIAL_SLAB_SIZE,
          &initial_slab_bytes))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "ROCm device malloc heap size overflow");
  }

  iree_hal_buffer_t* heap_buffer = NULL;
  iree_hal_buffer_t* initial_slabs_buffer = NULL;
  uint64_t heap_device_ptr = 0;
  uint64_t initial_slabs_device_ptr = 0;
  uint8_t* heap_header = NULL;

  iree_status_t status = iree_hal_streaming_rocm_allocate_device_buffer(
      context,
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE, &heap_buffer, &heap_device_ptr);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_rocm_allocate_device_buffer(
        context, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE, initial_slab_bytes,
        &initial_slabs_buffer, &initial_slabs_device_ptr);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(context->host_allocator,
                                   IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE,
                                   (void**)&heap_header);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_rocm_populate_malloc_heap_header(
        heap_header, initial_slabs_device_ptr, initial_slab_bytes);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&context->direct_transfer_mutex);
    status = iree_hal_device_transfer_h2d(
        context->device, heap_header, heap_buffer, /*target_offset=*/0,
        IREE_HAL_STREAMING_ROCM_MALLOC_HEAP_SIZE,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    iree_slim_mutex_unlock(&context->direct_transfer_mutex);
  }

  if (iree_status_is_ok(status)) {
    context->rocm_device_runtime.malloc_heap_buffer = heap_buffer;
    context->rocm_device_runtime.malloc_initial_slabs_buffer =
        initial_slabs_buffer;
    iree_atomic_store(&context->rocm_device_runtime.malloc_heap_device_ptr,
                      heap_device_ptr, iree_memory_order_release);
    heap_buffer = NULL;
    initial_slabs_buffer = NULL;
  }

  iree_allocator_free(context->host_allocator, heap_header);
  iree_hal_buffer_release(initial_slabs_buffer);
  iree_hal_buffer_release(heap_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Thread-local current context.
static IREE_THREAD_LOCAL iree_hal_streaming_context_t*
    iree_hal_streaming_current_context = NULL;
static IREE_THREAD_LOCAL int iree_hal_streaming_thread_token_storage;

typedef struct iree_hal_streaming_context_stack_t {
  iree_hal_streaming_context_t** contexts;
  iree_host_size_t depth;
  iree_host_size_t capacity;
} iree_hal_streaming_context_stack_t;

// Thread-local context stack for push/pop.
static IREE_THREAD_LOCAL iree_hal_streaming_context_stack_t
    iree_hal_streaming_context_stack = {
        .contexts = NULL,
        .depth = 0,
        .capacity = 0,
};

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_context_destroy(
    iree_hal_streaming_context_t* context);
static iree_status_t iree_hal_streaming_context_synchronize_streams(
    iree_hal_streaming_context_t* context, bool include_non_blocking_streams,
    bool flush_before_wait);

iree_status_t iree_hal_streaming_context_create(
    iree_hal_streaming_device_t* device_entry,
    iree_hal_streaming_context_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device_entry);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (!registry) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "global symbol registry failed to initialize");
  }

  iree_hal_streaming_context_t* context = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*context),
                                (void**)&context));
  iree_atomic_ref_count_init(&context->ref_count);
  context->device = device_entry->hal_device;
  context->device_ordinal = device_entry->ordinal;
  context->device_entry = device_entry;
  context->queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  context->device_allocator =
      iree_hal_device_allocator(device_entry->hal_device);
  context->flags = flags;
  context->default_stream = NULL;
  context->next_stream_id = 1;
  context->next_capture_id = 1;
  context->peer_contexts = NULL;
  context->peer_count = 0;
  context->peer_capacity = 0;
  memset(&context->symbol_map, 0, sizeof(context->symbol_map));
  memset(&context->buffer_table, 0, sizeof(context->buffer_table));
  context->pageable_h2d_staging_buffer = NULL;
  context->pageable_h2d_staging_size = 0;
  memset(&context->rocm_device_runtime, 0,
         sizeof(context->rocm_device_runtime));
  context->rocm_device_runtime.malloc_heap_device_ptr = IREE_ATOMIC_VAR_INIT(0);
  context->rocm_device_runtime.hostcall_buffer_device_ptr =
      IREE_ATOMIC_VAR_INIT(0);
  iree_atomic_store(&context->capture_stream_count, 0,
                    iree_memory_order_relaxed);
  context->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&context->mutex);
  iree_slim_mutex_initialize(&context->direct_transfer_mutex);

  // Initialize global list pointers.
  context->context_list_entry.next = NULL;
  context->context_list_entry.prev = NULL;

  // Initialize stream tracking BEFORE creating default stream.
  iree_slim_mutex_initialize(&context->stream_list_mutex);
  context->stream_count = 0;
  context->stream_capacity =
      8;  // Pre-allocate for default stream + user streams.
  context->streams = NULL;

  // Initialize default limits. Buffered printf uses a per-dispatch FIFO sized
  // for the ROCm device-library workitem debug record window.
  context->limits.stack_size = 1024;                        // 1KB default
  context->limits.printf_fifo_size = 4 * 1024 * 1024;       // 4MB
  context->limits.malloc_heap_size = 8 * 1024 * 1024;       // 8MB
  context->limits.dev_runtime_sync_depth = 128;             // 128 levels
  context->limits.dev_runtime_pending_launch_count = 2048;  // 2048 launches
  context->limits.max_l2_fetch_granularity = 128;           // 128 bytes
  context->limits.persisting_l2_cache_size = 0;             // 0 = default

  // Retain the HAL device.
  iree_hal_device_retain(context->device);
  iree_hal_allocator_retain(context->device_allocator);

  // Initialize buffer mapping table.
  hrx_buffer_table_initialize(&context->buffer_table);

  // Initialize symbol map with global registry as the backing store.
  iree_status_t status = iree_hal_streaming_context_symbol_map_initialize(
      context, /*initial_capacity=*/16, registry, host_allocator,
      &context->symbol_map);

  // Allocate stream tracking array.
  if (iree_status_is_ok(status)) {
    iree_host_size_t stream_array_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(context->stream_capacity,
                                                  sizeof(context->streams[0]),
                                                  &stream_array_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream list capacity overflow");
    } else {
      status = iree_allocator_malloc(host_allocator, stream_array_size,
                                     (void**)&context->streams);
    }
  }

  // Create default stream.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_create(context, /*flags=*/0,
                                              /*priority=*/0, host_allocator,
                                              &context->default_stream);
  }

  if (iree_status_is_ok(status)) {
    // Register with global list.
    iree_hal_streaming_register_context(context);
    *out_context = context;
  } else {
    iree_hal_streaming_context_destroy(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_context_destroy(
    iree_hal_streaming_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unregister from global list.
  iree_hal_streaming_unregister_context(context);

  // Clean up peer contexts array.
  if (context->peer_contexts) {
    for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
      iree_hal_streaming_context_release(context->peer_contexts[i]);
    }
    iree_allocator_free(context->host_allocator, context->peer_contexts);
  }

  // Synchronize all streams before detaching them from the context; pending
  // command buffers require the context/device to flush correctly.
  iree_status_ignore(iree_hal_streaming_context_synchronize(context));

  iree_hal_streaming_memory_release_pageable_staging(context);
  iree_hal_streaming_context_deinitialize_rocm_hostcall_service(context);
  iree_hal_buffer_release(
      context->rocm_device_runtime.malloc_initial_slabs_buffer);
  iree_hal_buffer_release(context->rocm_device_runtime.malloc_heap_buffer);

  // Deinitialize symbol map and unload any statically-registered modules that
  // were on-demand loaded for this context.
  iree_hal_streaming_context_symbol_map_deinitialize(&context->symbol_map);

  // Deinitialize buffer mapping table.
  hrx_buffer_table_deinitialize(&context->buffer_table);

  // Release default stream.
  // This releases the context's reference but not the list's reference.
  iree_hal_streaming_stream_t* default_stream = context->default_stream;
  context->default_stream = NULL;

  // Unregister all remaining streams.
  // This releases the list's references, which may trigger stream destruction.
  while (context->stream_count > 0) {
    iree_hal_streaming_stream_t* stream = context->streams[0];
    // Detach surviving user-owned streams from the context being destroyed.
    stream->context = NULL;
    // Remove from list (swap with last).
    context->streams[0] = context->streams[context->stream_count - 1];
    --context->stream_count;
    // Release the list's reference.
    iree_hal_streaming_stream_release(stream);
  }

  // Now release the context's reference to default stream.
  iree_hal_streaming_stream_release(default_stream);

  // Free stream tracking resources.
  if (context->streams) {
    iree_allocator_free(context->host_allocator, context->streams);
  }
  iree_slim_mutex_deinitialize(&context->stream_list_mutex);

  iree_status_ignore(context->loop_status);
  iree_hal_allocator_release(context->device_allocator);
  iree_hal_device_release(context->device);

  // Deinitialize synchronization.
  iree_slim_mutex_deinitialize(&context->direct_transfer_mutex);
  iree_slim_mutex_deinitialize(&context->mutex);

  // Free context memory.
  const iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_context_retain(iree_hal_streaming_context_t* context) {
  if (context) {
    iree_atomic_ref_count_inc(&context->ref_count);
  }
}

void iree_hal_streaming_context_release(iree_hal_streaming_context_t* context) {
  if (context && iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    iree_hal_streaming_context_destroy(context);
  }
}

iree_hal_streaming_context_flags_t iree_hal_streaming_context_flags(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  return context->flags;
}

iree_hal_streaming_context_t* iree_hal_streaming_context_current(void) {
  iree_hal_streaming_context_t* context = iree_hal_streaming_current_context;
  return context;
}

uintptr_t iree_hal_streaming_current_thread_token(void) {
  return (uintptr_t)&iree_hal_streaming_thread_token_storage;
}

void iree_hal_streaming_context_set_current(
    iree_hal_streaming_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Retain new context and release old one.
  iree_hal_streaming_context_t* old_context =
      iree_hal_streaming_current_context;
  if (context) {
    iree_hal_streaming_context_retain(context);
  }
  iree_hal_streaming_current_context = context;
  iree_hal_streaming_context_release(old_context);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_streaming_context_push(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Grow stack if needed.
  if (iree_hal_streaming_context_stack.depth >=
      iree_hal_streaming_context_stack.capacity) {
    iree_host_size_t new_capacity =
        iree_hal_streaming_context_stack.capacity
            ? iree_hal_streaming_context_stack.capacity * 2
            : 8;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_realloc(
                context->host_allocator,
                new_capacity * sizeof(iree_hal_streaming_context_t*),
                (void**)&iree_hal_streaming_context_stack.contexts));
    iree_hal_streaming_context_stack.capacity = new_capacity;
  }

  // Push current context onto stack.
  if (iree_hal_streaming_current_context) {
    iree_hal_streaming_context_stack
        .contexts[iree_hal_streaming_context_stack.depth++] =
        iree_hal_streaming_current_context;
  }

  // Set new current context.
  iree_hal_streaming_context_retain(context);
  iree_hal_streaming_current_context = context;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_pop(
    iree_hal_streaming_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_context) *out_context = NULL;

  // Release current context.
  if (iree_hal_streaming_current_context) {
    if (out_context) {
      *out_context = iree_hal_streaming_current_context;
    } else {
      iree_hal_streaming_context_release(iree_hal_streaming_current_context);
    }
  }

  // Pop from stack.
  if (iree_hal_streaming_context_stack.depth > 0) {
    iree_hal_streaming_current_context =
        iree_hal_streaming_context_stack
            .contexts[--iree_hal_streaming_context_stack.depth];
  } else {
    iree_hal_streaming_current_context = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t* out_value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;

  // Return the limit value from context.
  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      *out_value = context->limits.stack_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      *out_value = context->limits.printf_fifo_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      *out_value = context->limits.malloc_heap_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      *out_value = context->limits.dev_runtime_sync_depth;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      *out_value = context->limits.dev_runtime_pending_launch_count;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      *out_value = context->limits.max_l2_fetch_granularity;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      *out_value = context->limits.persisting_l2_cache_size;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid limit type %d", limit);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_set_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate the limit value first without holding the lock.
  iree_status_t status = iree_ok_status();
  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      // Stack size must be at least 512 bytes.
      if (value < 512) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack size must be at least 512 bytes");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      // Printf FIFO must be at least 4KB.
      if (value < 4096) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "printf FIFO size must be at least 4KB");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      // Heap size must be at least 4KB.
      if (value < 4096) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malloc heap size must be at least 4KB");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      // Must be at least 1.
      if (value < 1) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "runtime sync depth must be at least 1");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      // Must be at least 1.
      if (value < 1) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "runtime pending launch count must be at least 1");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      // Must be 0, 32, 64, or 128 bytes.
      if (value != 0 && value != 32 && value != 64 && value != 128) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "L2 fetch granularity must be 0, 32, 64, or 128");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      // No specific validation for cache size.
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid limit type %d", limit);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // Now take the lock and set the value.
  iree_slim_mutex_lock(&context->mutex);

  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      context->limits.stack_size = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      context->limits.printf_fifo_size = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      if (context->rocm_device_runtime.malloc_heap_buffer &&
          context->limits.malloc_heap_size != value) {
        status =
            iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "device malloc heap has already been initialized");
      } else {
        context->limits.malloc_heap_size = value;
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      context->limits.dev_runtime_sync_depth = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      context->limits.dev_runtime_pending_launch_count = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      context->limits.max_l2_fetch_granularity = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      context->limits.persisting_l2_cache_size = value;
      break;
    default:
      // Already validated above, should not reach here.
      break;
  }

  iree_slim_mutex_unlock(&context->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_rocm_device_malloc_heap(
    iree_hal_streaming_context_t* context, uint64_t* out_heap_device_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_heap_device_ptr);
  *out_heap_device_ptr = 0;
  const uint64_t cached_heap_device_ptr =
      iree_hal_streaming_context_cached_rocm_malloc_heap(context);
  if (cached_heap_device_ptr != 0) {
    *out_heap_device_ptr = cached_heap_device_ptr;
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->mutex);
  iree_status_t status = iree_ok_status();
  if (!context->rocm_device_runtime.malloc_heap_buffer) {
    status = iree_hal_streaming_context_initialize_rocm_device_runtime_locked(
        context);
  }
  if (iree_status_is_ok(status)) {
    *out_heap_device_ptr =
        iree_hal_streaming_context_cached_rocm_malloc_heap(context);
  }
  iree_slim_mutex_unlock(&context->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_enable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(peer_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->mutex);

  // Check if already enabled.
  for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
    if (context->peer_contexts[i] == peer_context) {
      iree_slim_mutex_unlock(&context->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();  // Already enabled.
    }
  }

  // Grow peer array if needed.
  if (context->peer_count >= context->peer_capacity) {
    const iree_host_size_t new_capacity =
        context->peer_capacity ? context->peer_capacity * 2 : 4;
    iree_status_t status = iree_allocator_realloc(
        context->host_allocator,
        new_capacity * sizeof(iree_hal_streaming_context_t*),
        (void**)&context->peer_contexts);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&context->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    context->peer_capacity = new_capacity;
  }

  // Add peer context.
  iree_hal_streaming_context_retain(peer_context);
  context->peer_contexts[context->peer_count++] = peer_context;

  // Update P2P topology if we have the registry.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (device_registry && device_registry->p2p_topology) {
    const iree_host_size_t src_ordinal = context->device_ordinal;
    const iree_host_size_t dst_ordinal = peer_context->device_ordinal;
    const iree_host_size_t device_count = device_registry->device_count;
    if (src_ordinal < device_count && dst_ordinal < device_count) {
      // Find the link in topology.
      const iree_host_size_t link_index =
          src_ordinal * device_count + dst_ordinal;
      iree_hal_streaming_p2p_link_t* link =
          &device_registry->p2p_topology[link_index];
      // Enable P2P access.
      link->access_supported = true;
      // TODO: Query actual P2P capabilities.
    }
  }

  iree_slim_mutex_unlock(&context->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_disable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(peer_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->mutex);

  // Find and remove peer.
  for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
    if (context->peer_contexts[i] == peer_context) {
      const iree_host_size_t dst_ordinal = peer_context->device_ordinal;

      // Release peer context.
      iree_hal_streaming_context_release(peer_context);

      // Shift remaining peers.
      for (iree_host_size_t j = i + 1; j < context->peer_count; ++j) {
        context->peer_contexts[j - 1] = context->peer_contexts[j];
      }
      context->peer_count--;

      // Update P2P topology.
      iree_hal_streaming_device_registry_t* device_registry =
          iree_hal_streaming_device_registry();
      if (device_registry && device_registry->p2p_topology) {
        const iree_host_size_t src_ordinal = context->device_ordinal;
        const iree_host_size_t device_count = device_registry->device_count;
        if (src_ordinal < device_count && dst_ordinal < device_count) {
          // Find the link in topology.
          const iree_host_size_t link_index =
              src_ordinal * device_count + dst_ordinal;
          iree_hal_streaming_p2p_link_t* link =
              &device_registry->p2p_topology[link_index];
          // Disable P2P access.
          link->access_supported = false;
        }
      }

      iree_slim_mutex_unlock(&context->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  iree_slim_mutex_unlock(&context->mutex);

  // Peer not found.
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_NOT_FOUND, "peer context not found");
}

iree_status_t iree_hal_streaming_context_register_stream(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&context->stream_list_mutex);

  // Grow array if needed (double capacity).
  if (context->stream_count >= context->stream_capacity) {
    iree_host_size_t new_capacity = 0;
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(context->stream_capacity, 2,
                                                  &new_capacity) ||
                      !iree_host_size_checked_mul(new_capacity,
                                                  sizeof(context->streams[0]),
                                                  &allocation_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream list capacity overflow");
    } else {
      status = iree_allocator_realloc(context->host_allocator, allocation_size,
                                      (void**)&context->streams);
    }
    if (iree_status_is_ok(status)) {
      context->stream_capacity = new_capacity;
    }
  }

  if (iree_status_is_ok(status)) {
    if (context->next_stream_id == 0 || context->next_stream_id > UINT32_MAX) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream identifier space exhausted");
    } else {
      const unsigned long long device_id =
          ((unsigned long long)context->device_ordinal + 1ull) << 32;
      stream->stream_id = device_id | context->next_stream_id++;
    }
  }

  if (iree_status_is_ok(status)) {
    // Retain the stream - the context's stream list owns a reference.
    iree_hal_streaming_stream_retain(stream);
    context->streams[context->stream_count++] = stream;
  }

  iree_slim_mutex_unlock(&context->stream_list_mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_allocate_capture_id(
    iree_hal_streaming_context_t* context, unsigned long long* out_capture_id) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_capture_id);
  *out_capture_id = 0;

  iree_slim_mutex_lock(&context->stream_list_mutex);
  if (context->next_capture_id == 0) {
    iree_slim_mutex_unlock(&context->stream_list_mutex);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream capture identifier space exhausted");
  }
  *out_capture_id = context->next_capture_id++;
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  return iree_ok_status();
}

void iree_hal_streaming_context_unregister_stream(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  if (!context || !stream) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  bool found = false;
  iree_slim_mutex_lock(&context->stream_list_mutex);

  for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
    if (context->streams[i] == stream) {
      // Swap with last and remove.
      context->streams[i] = context->streams[context->stream_count - 1];
      --context->stream_count;
      found = true;
      break;
    }
  }

  iree_slim_mutex_unlock(&context->stream_list_mutex);

  // Release the list's reference after unlinking. The caller holds another
  // reference while requesting unregister, so the stream cannot be destroyed
  // out from under this operation.
  if (found) {
    iree_hal_streaming_stream_release(stream);
  }

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_streaming_context_has_peer_contexts(
    iree_hal_streaming_context_t* context) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) return false;

  bool has_peer = false;
  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* candidate =
           device_registry->context_list.head;
       candidate; candidate = candidate->context_list_entry.next) {
    if (candidate != context) {
      has_peer = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);
  return has_peer;
}

// Takes a retained snapshot of the current stream list so callers can wait or
// synchronize without holding the list mutex across potentially blocking work.
static iree_status_t iree_hal_streaming_context_snapshot_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t*** out_streams, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_streams);
  IREE_ASSERT_ARGUMENT(out_count);
  *out_streams = NULL;
  *out_count = 0;

  iree_slim_mutex_lock(&context->stream_list_mutex);
  const iree_host_size_t count = context->stream_count;
  iree_hal_streaming_stream_t** streams = NULL;
  iree_status_t status = iree_ok_status();
  if (count > 0) {
    iree_host_size_t streams_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(count, sizeof(streams[0]),
                                                  &streams_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream snapshot size overflow");
    } else {
      status = iree_allocator_malloc(context->host_allocator, streams_size,
                                     (void**)&streams);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < count; ++i) {
        streams[i] = context->streams[i];
        if (streams[i]) {
          iree_hal_streaming_stream_retain(streams[i]);
        }
      }
    }
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  if (iree_status_is_ok(status)) {
    *out_streams = streams;
    *out_count = count;
  }
  return status;
}

static void iree_hal_streaming_context_release_stream_snapshot(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t** streams, iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_stream_release(streams[i]);
  }
  if (streams) {
    iree_allocator_free(context->host_allocator, streams);
  }
}

iree_status_t iree_hal_streaming_context_wait_idle(
    iree_hal_streaming_context_t* context, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(context);
  (void)timeout;
  return iree_hal_streaming_context_synchronize_streams(
      context, /*include_non_blocking_streams=*/true,
      /*flush_before_wait=*/true);
}

iree_status_t iree_hal_streaming_context_flush(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t count = 0;
  iree_status_t status =
      iree_hal_streaming_context_snapshot_streams(context, &streams, &count);

  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    if (streams[i]) {
      status = iree_hal_streaming_stream_flush(streams[i]);
    }
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams, count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  if (context->default_stream) {
    status = iree_hal_streaming_stream_flush(context->default_stream);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_flush_all(void) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_hal_streaming_context_t** contexts = NULL;
  iree_host_size_t context_capacity = 0;
  iree_host_size_t context_count = 0;
  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    ++context_capacity;
  }
  if (context_capacity > 0) {
    iree_host_size_t contexts_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            context_capacity, sizeof(contexts[0]), &contexts_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "context snapshot size overflow");
    } else {
      status = iree_allocator_malloc(device_registry->host_allocator,
                                     contexts_size, (void**)&contexts);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t index = 0;
    for (iree_hal_streaming_context_t* context =
             device_registry->context_list.head;
         context; context = context->context_list_entry.next) {
      contexts[index++] = context;
      iree_hal_streaming_context_retain(context);
    }
    context_count = index;
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  for (iree_host_size_t i = 0; i < context_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_streaming_context_flush(contexts[i]);
  }

  for (iree_host_size_t i = 0; i < context_count; ++i) {
    iree_hal_streaming_context_release(contexts[i]);
  }
  if (contexts) {
    iree_allocator_free(device_registry->host_allocator, contexts);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_context_synchronize_streams(
    iree_hal_streaming_context_t* context, bool include_non_blocking_streams,
    bool flush_before_wait) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (flush_before_wait) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_context_flush(context));
  }

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Synchronize streams from the retained snapshot. Legacy default stream
  // ordering excludes non-blocking streams, while device/context-wide
  // synchronization includes them.
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!iree_status_is_ok(status)) break;
    iree_hal_streaming_stream_t* stream = streams_copy[i];
    if (!stream) continue;
    if (!include_non_blocking_streams &&
        (stream->flags & IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING)) {
      continue;
    }
    status = iree_hal_streaming_stream_synchronize_flushed(stream);
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Also synchronize the default stream, which may not be in the streams list.
  // The legacy default stream always participates in its own ordering.
  if (context->default_stream) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_stream_synchronize_flushed(context->default_stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_synchronize(
    iree_hal_streaming_context_t* context) {
  return iree_hal_streaming_context_synchronize_streams(
      context, /*include_non_blocking_streams=*/true,
      /*flush_before_wait=*/true);
}

iree_status_t iree_hal_streaming_context_synchronize_legacy_default(
    iree_hal_streaming_context_t* context) {
  return iree_hal_streaming_context_synchronize_streams(
      context, /*include_non_blocking_streams=*/false,
      /*flush_before_wait=*/true);
}

iree_status_t iree_hal_streaming_context_synchronize_all(void) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL stream layer not initialized");
  }

  iree_hal_streaming_context_t** contexts = NULL;
  iree_host_size_t context_capacity = 0;
  iree_host_size_t context_count = 0;
  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&device_registry->context_list.mutex);
  for (iree_hal_streaming_context_t* context =
           device_registry->context_list.head;
       context; context = context->context_list_entry.next) {
    ++context_capacity;
  }
  if (context_capacity > 0) {
    iree_host_size_t contexts_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            context_capacity, sizeof(contexts[0]), &contexts_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "context snapshot size overflow");
    } else {
      status = iree_allocator_malloc(device_registry->host_allocator,
                                     contexts_size, (void**)&contexts);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t index = 0;
    for (iree_hal_streaming_context_t* context =
             device_registry->context_list.head;
         context; context = context->context_list_entry.next) {
      contexts[index++] = context;
      iree_hal_streaming_context_retain(context);
    }
    context_count = index;
  }
  iree_slim_mutex_unlock(&device_registry->context_list.mutex);

  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < context_count;
       ++i) {
    status = iree_hal_streaming_context_flush(contexts[i]);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < context_count;
       ++i) {
    status = iree_hal_streaming_context_synchronize_streams(
        contexts[i], /*include_non_blocking_streams=*/true,
        /*flush_before_wait=*/false);
  }

  for (iree_host_size_t i = 0; i < context_count; ++i) {
    iree_hal_streaming_context_release(contexts[i]);
  }
  if (contexts) {
    iree_allocator_free(device_registry->host_allocator, contexts);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_synchronize_blocking_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* except_stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_hal_streaming_stream_t* stream = streams_copy[i];
    if (!stream || stream == except_stream ||
        stream == context->default_stream ||
        iree_any_bit_set(stream->flags,
                         IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING) ||
        stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      continue;
    }
    status = iree_hal_streaming_stream_synchronize(stream);
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_query(
    iree_hal_streaming_context_t* context, int* status) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(status);
  *status = 0;

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t query_status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(query_status);
       ++i) {
    int stream_status = 0;
    query_status =
        iree_hal_streaming_stream_query(streams_copy[i], &stream_status);
    if (iree_status_is_ok(query_status) && stream_status != 0) {
      *status = 1;
      break;
    }
  }
  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);
  return query_status;
}

iree_status_t iree_hal_streaming_context_wait_all_submitted(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Wait for submitted work on all streams (doesn't flush).
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (streams_copy[i]) {
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_stream_wait_submitted(streams_copy[i]);
      }
    }
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // For the default stream, synchronize fully since caller needs it complete.
  if (context->default_stream) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_stream_synchronize(context->default_stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
