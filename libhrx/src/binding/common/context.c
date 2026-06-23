// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "common/internal.h"

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Thread-local current context.
static iree_thread_local iree_hal_streaming_context_t*
    iree_hal_streaming_current_context = NULL;
static iree_thread_local int iree_hal_streaming_thread_token_storage;

typedef struct iree_hal_streaming_context_stack_t {
  iree_hal_streaming_context_t** contexts;
  iree_host_size_t depth;
  iree_host_size_t capacity;
} iree_hal_streaming_context_stack_t;

// Thread-local context stack for push/pop.
static iree_thread_local iree_hal_streaming_context_stack_t
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

iree_status_t iree_hal_streaming_context_create(
    iree_hal_streaming_device_t* device_entry,
    iree_hal_streaming_context_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device_entry);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

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
  context->executable_cache = NULL;
  context->flags = flags;
  context->default_stream = NULL;
  context->next_stream_id = 1;
  context->peer_contexts = NULL;
  context->peer_count = 0;
  context->peer_capacity = 0;
  memset(&context->symbol_map, 0, sizeof(context->symbol_map));
  memset(&context->buffer_table, 0, sizeof(context->buffer_table));
  context->pageable_h2d_staging_buffer = NULL;
  context->pageable_h2d_staging_size = 0;
  iree_atomic_store(&context->capture_stream_count, 0,
                    iree_memory_order_relaxed);
  context->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&context->mutex);

  // Initialize global list pointers.
  context->context_list_entry.next = NULL;
  context->context_list_entry.prev = NULL;

  // Initialize stream tracking BEFORE creating default stream.
  iree_slim_mutex_initialize(&context->stream_list_mutex);
  context->stream_count = 0;
  context->stream_capacity =
      8;  // Pre-allocate for default stream + user streams.
  context->streams = NULL;

  // Initialize default limits.
  // These are typical defaults matching CUDA/HIP behavior.
  context->limits.stack_size = 1024;                        // 1KB default
  context->limits.printf_fifo_size = 1024 * 1024;           // 1MB
  context->limits.malloc_heap_size = 8 * 1024 * 1024;       // 8MB
  context->limits.dev_runtime_sync_depth = 128;             // 128 levels
  context->limits.dev_runtime_pending_launch_count = 2048;  // 2048 launches
  context->limits.max_l2_fetch_granularity = 128;           // 128 bytes
  context->limits.persisting_l2_cache_size = 0;             // 0 = default

  // Retain the HAL device.
  iree_hal_device_retain(context->device);
  iree_hal_allocator_retain(context->device_allocator);

  // Create executable cache for this context.
  iree_status_t status = iree_hal_executable_cache_create(
      context->device, IREE_SV("stream_hal_cache"), &context->executable_cache);

  // Initialize buffer mapping table.
  if (iree_status_is_ok(status)) {
    hrx_buffer_table_initialize(&context->buffer_table);
  }

  // Initialize symbol map with global registry as the backing store.
  iree_hal_streaming_global_symbol_registry_t* registry =
      iree_hal_streaming_global_symbol_registry();
  if (iree_status_is_ok(status) && !registry) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "global symbol registry failed to initialize");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_symbol_map_initialize(
        context, /*initial_capacity=*/16, registry, host_allocator,
        &context->symbol_map);
  }

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
  iree_hal_executable_cache_release(context->executable_cache);
  iree_hal_allocator_release(context->device_allocator);
  iree_hal_device_release(context->device);

  // Deinitialize synchronization.
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
      context->limits.malloc_heap_size = value;
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
  return iree_ok_status();
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
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** temp_streams = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &temp_streams, &count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Synchronize all streams. Bail on the first failure.
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < count; ++i) {
    status = iree_hal_streaming_stream_synchronize(temp_streams[i]);
  }

  iree_hal_streaming_context_release_stream_snapshot(context, temp_streams,
                                                     count);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_streaming_context_synchronize_streams(
    iree_hal_streaming_context_t* context, bool include_non_blocking_streams) {
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
    status = iree_hal_streaming_stream_synchronize(stream);
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
        z0, iree_hal_streaming_stream_synchronize(context->default_stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_synchronize(
    iree_hal_streaming_context_t* context) {
  return iree_hal_streaming_context_synchronize_streams(
      context, /*include_non_blocking_streams=*/true);
}

iree_status_t iree_hal_streaming_context_synchronize_legacy_default(
    iree_hal_streaming_context_t* context) {
  return iree_hal_streaming_context_synchronize_streams(
      context, /*include_non_blocking_streams=*/false);
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
