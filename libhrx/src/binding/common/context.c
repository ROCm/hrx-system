// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/context.h"

#include <string.h>

#include "common/device.h"
#include "common/graph.h"
#include "common/init.h"
#include "common/memory.h"
#include "common/stream.h"
#include "common/stream_queue.h"
#include "libhrx/src/libhrx/mem_pool_backing.h"

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Thread-local current context.
static IREE_THREAD_LOCAL iree_hal_streaming_context_t*
    iree_hal_streaming_current_context = NULL;
static IREE_THREAD_LOCAL int iree_hal_streaming_thread_token_storage;

// Stream IDs identify timeline dependencies that can outlive the context that
// created them. Keeping the namespace process-wide prevents a dependency from
// being mistaken for one recorded by a different context.
static iree_atomic_uint64_t iree_hal_streaming_next_stream_id =
    IREE_ATOMIC_VAR_INIT(1);

typedef struct iree_hal_streaming_context_stack_t {
  // Storage allocator captured when the first stack array is acquired.
  iree_allocator_t allocator;
  // Owned references to contexts below the current one.
  iree_hal_streaming_context_t** contexts;
  // Number of stacked references.
  iree_host_size_t depth;
  // Allocated stack entry capacity.
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
  context->queue = NULL;
  context->device_allocator =
      iree_hal_device_allocator(device_entry->hal_device);
  context->timestamp_domain = iree_hal_streaming_query_timestamp_domain(
      iree_hal_device_spec(device_entry->hal_device));
  iree_hal_streaming_context_set_flags(context, flags);
  context->default_stream = NULL;
  context->next_capture_id = 1;
  context->peer_contexts = NULL;
  context->peer_count = 0;
  context->peer_capacity = 0;
  memset(&context->symbol_map, 0, sizeof(context->symbol_map));
  memset(&context->buffer_table, 0, sizeof(context->buffer_table));
  context->pending_free_head = NULL;
  iree_atomic_store(&context->capture_stream_count, 0,
                    iree_memory_order_relaxed);
  context->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&context->mutex);
  iree_slim_mutex_initialize(&context->pending_free_mutex);
  iree_hal_streaming_capture_admission_initialize(&context->capture_admission);
  iree_hal_streaming_value_wait_lanes_initialize(context);

  // Initialize global list pointers.
  context->context_list_entry.next = NULL;
  context->context_list_entry.prev = NULL;

  // Initialize stream tracking BEFORE creating default stream.
  iree_slim_mutex_initialize(&context->stream_list_mutex);
  context->stream_count = 0;
  context->stream_capacity =
      8;  // Pre-allocate for default stream + user streams.
  context->streams = NULL;
  context->stream_wait_frontier = NULL;
  context->event_record_timeline = (iree_hal_streaming_operation_timeline_t){0};
  context->event_record_queue = NULL;
  iree_slim_mutex_initialize(&context->event_record_mutex);
  context->transfer_queue = NULL;
  iree_slim_mutex_initialize(&context->transfer_queue_mutex);

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

  // Initialize buffer mapping table.
  hrx_buffer_table_initialize(&context->buffer_table);

  // Initialize the device timestamp slot pool from a coherent host source.
  iree_hal_pool_t* timestamp_memory_pool = NULL;
  iree_status_t status = hrx_mem_pool_backing_retain_allocation_source(
      device_entry->hrx_device, HRX_MEMORY_TYPE_HOST_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, &timestamp_memory_pool);
  iree_hal_streaming_event_timestamp_pool_initialize(
      timestamp_memory_pool, host_allocator, &context->timestamp_pool);
  iree_hal_pool_release(timestamp_memory_pool);

  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_device_acquire_primary_queue(device_entry,
                                                             &context->queue);
  }

  // Create the context-wide event record timeline. Resource-partitioned
  // binding contexts share this timeline so primary-context synchronization
  // includes their context-level records without relying on queue FIFO order.
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &context->event_record_timeline.semaphore);
  }

  // Initialize symbol map with global registry as the backing store.
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
    status = iree_hal_streaming_stream_create(
        context, context->queue, /*flags=*/0, /*priority=*/0, host_allocator,
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
  iree_status_t status = iree_hal_streaming_context_synchronize(context);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
  status = iree_hal_streaming_memory_release_terminal_async_frees(context);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  // A failed public event timeline is not proof that its native queue has
  // retired. Stop that queue before releasing the storage its records use.
  iree_hal_queue_release(context->event_record_queue);
  context->event_record_queue = NULL;
  iree_hal_queue_release(context->transfer_queue);
  context->transfer_queue = NULL;
  iree_slim_mutex_deinitialize(&context->transfer_queue_mutex);

  // Deinitialize symbol map and unload any statically-registered modules that
  // were on-demand loaded for this context.
  iree_hal_streaming_context_symbol_map_deinitialize(&context->symbol_map);

  // Deinitialize buffer mapping table.
  hrx_buffer_table_deinitialize(&context->buffer_table);
  IREE_ASSERT(context->pending_free_head == NULL,
              "pending asynchronous frees must be drained before context "
              "destruction");
  iree_slim_mutex_deinitialize(&context->pending_free_mutex);

  // Release default stream.
  // This releases the context's reference but not the list's reference.
  iree_hal_streaming_stream_t* default_stream = context->default_stream;
  context->default_stream = NULL;

  // Emptying the list under its own mutex is what takes the streams out of
  // reach of every other reader: they all walk the list under this lock and now
  // find nothing. It also disarms unregistration, whose scan of the emptied
  // list matches no stream and so writes nothing, which leaves registration as
  // the only writer that could still reach the array - and that cannot run
  // either. A context reaches destruction in one of two states, and neither
  // admits a registration: unpublished, which is how
  // iree_hal_streaming_context_create disposes of a context it failed to finish
  // building and where no thread but that one can name it; or published with
  // its last reference gone, and registering runs under a reference. The array
  // is therefore stable for the walk below.
  iree_slim_mutex_lock(&context->stream_list_mutex);
  const iree_host_size_t detached_stream_count = context->stream_count;
  for (iree_host_size_t i = 0; i < detached_stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = context->streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->context == context) {
      stream->registration_state =
          IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_DETACHING;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  context->stream_count = 0;
  iree_hal_fence_t* stream_wait_frontier = context->stream_wait_frontier;
  context->stream_wait_frontier = NULL;
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  // Each entry was marked DETACHING while the stream-list mutex was held, so
  // adoption and new capture publication reject it before the empty registry
  // becomes visible. A live capture graph retains this context, making last-ref
  // teardown mutually exclusive with a graph-backed capture transaction. Clear
  // the unowned context and queue fields under the stream mutex; list
  // references remain live until the separate release pass below.
  for (iree_host_size_t i = 0; i < detached_stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = context->streams[i];
    iree_hal_queue_t* queue = NULL;
    iree_hal_queue_t* cooperative_queue = NULL;
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->context == context) {
      queue = stream->queue;
      cooperative_queue = stream->cooperative_queue;
      stream->queue = NULL;
      stream->cooperative_queue = NULL;
      stream->registration_state =
          IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_UNREGISTERED;
      stream->context = NULL;
    }
    iree_slim_mutex_unlock(&stream->mutex);
    iree_hal_queue_release(cooperative_queue);
    iree_hal_queue_release(queue);
  }
  for (iree_host_size_t i = 0; i < detached_stream_count; ++i) {
    iree_hal_streaming_stream_release(context->streams[i]);
  }
  iree_hal_fence_release(stream_wait_frontier);

  // Now release the context's reference to default stream.
  iree_hal_streaming_stream_release(default_stream);

  // Free stream tracking resources.
  if (context->streams) {
    iree_allocator_free(context->host_allocator, context->streams);
  }
  iree_slim_mutex_deinitialize(&context->stream_list_mutex);

  iree_hal_semaphore_release(context->event_record_timeline.semaphore);
  iree_slim_mutex_deinitialize(&context->event_record_mutex);

  // A record draws its slot from the pool of the recording event's own context
  // and every event retains that context, so reaching here means every event
  // that could hold a slot from this pool is gone and every slot is back.
  iree_hal_streaming_event_timestamp_pool_deinitialize(
      &context->timestamp_pool);
  iree_hal_streaming_value_wait_lanes_deinitialize(context);
  iree_hal_streaming_capture_admission_deinitialize(
      &context->capture_admission);

  iree_status_ignore(context->loop_status);
  iree_hal_allocator_release(context->device_allocator);
  iree_hal_queue_release(context->queue);
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

bool iree_hal_streaming_context_try_retain(
    iree_hal_streaming_context_t* context) {
  if (!context) {
    return false;
  }
  int32_t reference_count = iree_atomic_ref_count_load(&context->ref_count);
  while (reference_count > 0) {
    if (iree_atomic_compare_exchange_weak(
            &context->ref_count, &reference_count, reference_count + 1,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return true;
    }
  }
  return false;
}

void iree_hal_streaming_context_release(iree_hal_streaming_context_t* context) {
  if (context && iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    iree_hal_streaming_context_destroy(context);
  }
}

iree_hal_streaming_context_flags_t iree_hal_streaming_context_flags(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  const uint32_t bits =
      iree_atomic_load(&context->flags, iree_memory_order_relaxed);
  return (iree_hal_streaming_context_flags_t){
      .scheduling_mode = (iree_hal_streaming_scheduling_mode_t)(bits & 0xFFu),
      .map_host_memory = (bits >> 8) & 1u,
      .resize_local_mem_to_max = (bits >> 9) & 1u,
  };
}

void iree_hal_streaming_context_set_flags(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_flags_t flags) {
  const uint32_t bits = (uint32_t)flags.scheduling_mode |
                        ((uint32_t)flags.map_host_memory << 8) |
                        ((uint32_t)flags.resize_local_mem_to_max << 9);
  iree_atomic_store(&context->flags, bits, iree_memory_order_relaxed);
}

iree_async_wait_flags_t iree_hal_streaming_context_wait_flags(
    iree_hal_streaming_context_t* context) {
  switch (iree_hal_streaming_context_flags(context).scheduling_mode) {
    case IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN:
      return IREE_ASYNC_WAIT_FLAG_ACTIVE;
    case IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC:
      return IREE_ASYNC_WAIT_FLAG_NONE;
    case IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD:
    case IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO:
    default:
      return IREE_ASYNC_WAIT_FLAG_YIELD;
  }
}

iree_hal_streaming_context_t* iree_hal_streaming_context_current(void) {
  iree_hal_streaming_context_t* context = iree_hal_streaming_current_context;
  return context;
}

uintptr_t iree_hal_streaming_current_thread_token(void) {
  return (uintptr_t)&iree_hal_streaming_thread_token_storage;
}

iree_status_t iree_hal_streaming_context_set_current(
    iree_hal_streaming_context_t* context) {
  if (!context) {
    iree_hal_streaming_context_clear_current();
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_streaming_tls_set(
      iree_hal_streaming_device_registry()->context_key,
      &iree_hal_streaming_context_stack));
  IREE_TRACE_ZONE_BEGIN(z0);

  // Retain new context and release old one.
  iree_hal_streaming_context_t* old_context =
      iree_hal_streaming_current_context;
  iree_hal_streaming_context_retain(context);
  iree_hal_streaming_current_context = context;
  iree_hal_streaming_context_release(old_context);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_streaming_context_clear_current(void) {
  iree_hal_streaming_context_t* context = iree_hal_streaming_current_context;
  iree_hal_streaming_current_context = NULL;
  iree_hal_streaming_context_release(context);
}

void iree_hal_streaming_context_cleanup_thread(void* state) {
  iree_hal_streaming_context_stack_t* stack = state;
  if (!stack) {
    return;
  }
  iree_hal_streaming_context_clear_current();
  while (stack->depth) {
    iree_hal_streaming_context_release(stack->contexts[--stack->depth]);
  }
  iree_allocator_free(stack->allocator, stack->contexts);
  *stack = (iree_hal_streaming_context_stack_t){0};
}

iree_status_t iree_hal_streaming_context_push(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_tls_set(
              iree_hal_streaming_device_registry()->context_key,
              &iree_hal_streaming_context_stack));

  // Grow stack if needed.
  if (iree_hal_streaming_context_stack.depth >=
      iree_hal_streaming_context_stack.capacity) {
    iree_host_size_t new_capacity =
        iree_hal_streaming_context_stack.capacity
            ? iree_hal_streaming_context_stack.capacity * 2
            : 8;
    if (!iree_hal_streaming_context_stack.contexts) {
      iree_hal_streaming_context_stack.allocator = context->host_allocator;
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_realloc(
                iree_hal_streaming_context_stack.allocator,
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
  if (out_context) {
    *out_context = NULL;
  }

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
      // Release peer context.
      iree_hal_streaming_context_release(peer_context);

      // Shift remaining peers.
      for (iree_host_size_t j = i + 1; j < context->peer_count; ++j) {
        context->peer_contexts[j - 1] = context->peer_contexts[j];
      }
      context->peer_count--;

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
  iree_hal_fence_t* wait_frontier = NULL;

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
    uint64_t stream_id = iree_atomic_load(&iree_hal_streaming_next_stream_id,
                                          iree_memory_order_relaxed);
    while (stream_id != 0 && stream_id != UINT64_MAX &&
           !iree_atomic_compare_exchange_weak(
               &iree_hal_streaming_next_stream_id, &stream_id, stream_id + 1,
               iree_memory_order_relaxed, iree_memory_order_relaxed)) {
    }
    if (stream_id == 0 || stream_id == UINT64_MAX) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream identifier space exhausted");
    } else {
      stream->stream_id = stream_id;
    }
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->context != context ||
        stream->registration_state !=
            IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_UNREGISTERED) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "stream is already registered");
    } else {
      stream->registration_state =
          IREE_HAL_STREAMING_STREAM_REGISTRATION_STATE_REGISTERED;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  if (iree_status_is_ok(status)) {
    // Retain the stream - the context's stream list owns a reference.
    iree_hal_streaming_stream_retain(stream);
    context->streams[context->stream_count++] = stream;
    wait_frontier = context->stream_wait_frontier;
    iree_hal_fence_retain(wait_frontier);
  }

  iree_slim_mutex_unlock(&context->stream_list_mutex);

  if (iree_status_is_ok(status) && wait_frontier) {
    status = iree_hal_streaming_stream_wait_semaphores(
        stream, iree_hal_fence_semaphore_list(wait_frontier));
  }
  iree_hal_fence_release(wait_frontier);
  if (!iree_status_is_ok(status) && stream->stream_id != 0) {
    iree_hal_streaming_context_unregister_stream(context, stream);
  }

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
  if (!context || !stream) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Capture detachment and registry removal are one lifecycle transaction;
  // releasing the list reference remains outside all capture locks.
  if (iree_hal_streaming_capture_unregister_stream(context, stream)) {
    // Completion observers can retain a detached stream after its public handle
    // is gone. Sever its borrowed context before removing the registry's final
    // ownership so later observer retirement cannot revisit a destroyed
    // context.
    iree_slim_mutex_lock(&stream->mutex);
    iree_hal_queue_t* queue = stream->queue;
    iree_hal_queue_t* cooperative_queue = stream->cooperative_queue;
    stream->queue = NULL;
    stream->cooperative_queue = NULL;
    stream->context = NULL;
    iree_slim_mutex_unlock(&stream->mutex);
    iree_hal_queue_release(cooperative_queue);
    iree_hal_queue_release(queue);
    iree_hal_streaming_stream_release(stream);
  }

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_streaming_context_has_peer_contexts(
    iree_hal_streaming_context_t* context) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    return false;
  }

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

// Takes a retained snapshot while the caller holds |stream_list_mutex|.
static iree_status_t iree_hal_streaming_context_snapshot_streams_locked(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t*** out_streams, iree_host_size_t* out_count) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_streams);
  IREE_ASSERT_ARGUMENT(out_count);

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
  if (iree_status_is_ok(status)) {
    *out_streams = streams;
    *out_count = count;
  }
  return status;
}

iree_status_t iree_hal_streaming_context_snapshot_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t*** out_streams,
    iree_host_size_t* out_stream_count) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_streams);
  IREE_ASSERT_ARGUMENT(out_stream_count);

  iree_slim_mutex_lock(&context->stream_list_mutex);
  iree_status_t status = iree_hal_streaming_context_snapshot_streams_locked(
      context, out_streams, out_stream_count);
  iree_slim_mutex_unlock(&context->stream_list_mutex);
  return status;
}

void iree_hal_streaming_context_release_stream_snapshot(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t** streams, iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_stream_release(streams[i]);
  }
  if (streams) {
    iree_allocator_free(context->host_allocator, streams);
  }
}

iree_status_t iree_hal_streaming_context_record_event(
    iree_hal_streaming_context_t* context, iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(event);
  if (IREE_UNLIKELY(event->context != context)) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "event belongs to a different context");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams, &stream_count);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_event_record_after_streams(
        event, streams, stream_count, /*additional_timeline=*/NULL);
  }
  iree_hal_streaming_context_release_stream_snapshot(context, streams,
                                                     stream_count);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_wait_frontier_extend(
    iree_hal_fence_t* previous_frontier, iree_hal_semaphore_t* semaphore,
    uint64_t value, iree_allocator_t host_allocator,
    iree_hal_fence_t** out_frontier) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_frontier);
  const iree_hal_semaphore_list_t previous_points =
      iree_hal_fence_semaphore_list(previous_frontier);
  iree_host_size_t capacity = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(previous_points.count, 1, &capacity))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream wait frontier capacity overflow");
  }

  iree_hal_fence_t* frontier = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_create(capacity, host_allocator, &frontier));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < previous_points.count && iree_status_is_ok(status); ++i) {
    uint64_t current_value = 0;
    status =
        iree_hal_semaphore_query(previous_points.semaphores[i], &current_value);
    if (iree_status_is_ok(status) &&
        current_value < previous_points.payload_values[i]) {
      status = iree_hal_fence_insert(frontier, previous_points.semaphores[i],
                                     previous_points.payload_values[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_fence_insert(frontier, semaphore, value);
  }

  if (iree_status_is_ok(status)) {
    *out_frontier = frontier;
  } else {
    iree_hal_fence_release(frontier);
  }
  return status;
}

iree_status_t iree_hal_streaming_context_wait_event(
    iree_hal_streaming_context_t* context, iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(event);
  if (IREE_UNLIKELY(iree_hal_streaming_event_has_capture_graph(event))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "captured event has no submitted point");
  }

  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);
  if (!recorded_point.semaphore) {
    iree_hal_streaming_event_release_recorded_point(&recorded_point);
    return iree_ok_status();
  }

  uint64_t current_value = 0;
  iree_status_t status =
      iree_hal_semaphore_query(recorded_point.semaphore, &current_value);
  if (!iree_status_is_ok(status) || current_value >= recorded_point.value) {
    iree_hal_streaming_event_release_recorded_point(&recorded_point);
    return status;
  }

  iree_hal_streaming_stream_t** streams = NULL;
  iree_host_size_t stream_count = 0;
  iree_hal_fence_t* new_frontier = NULL;
  iree_hal_fence_t* old_frontier = NULL;
  iree_slim_mutex_lock(&context->stream_list_mutex);
  status = iree_hal_streaming_wait_frontier_extend(
      context->stream_wait_frontier, recorded_point.semaphore,
      recorded_point.value, context->host_allocator, &new_frontier);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_snapshot_streams_locked(
        context, &streams, &stream_count);
  }
  if (iree_status_is_ok(status)) {
    old_frontier = context->stream_wait_frontier;
    context->stream_wait_frontier = new_frontier;
    new_frontier = NULL;
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t wait = {
        .count = 1,
        .semaphores = &recorded_point.semaphore,
        .payload_values = &recorded_point.value,
    };
    for (iree_host_size_t i = 0; i < stream_count; ++i) {
      status = iree_status_join(
          status, iree_hal_streaming_stream_wait_semaphores(streams[i], wait));
    }
  }

  iree_hal_fence_release(new_frontier);
  iree_hal_fence_release(old_frontier);
  iree_hal_streaming_context_release_stream_snapshot(context, streams,
                                                     stream_count);
  iree_hal_streaming_event_release_recorded_point(&recorded_point);
  return status;
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

// Drains the finite set of streams registered when this fallback begins
// without allocating snapshot storage. Stream IDs are process-wide monotonic,
// so streams registered later are excluded by |max_stream_id|. A stream that
// unregisters before this scan retains it is safe to omit: ordinary stream
// destruction synchronizes before unlinking, and construction rollback ends in
// that same destruction path after unregistering. Context destruction does not
// unlink streams concurrently with its own synchronization.
static iree_status_t
iree_hal_streaming_context_synchronize_streams_without_snapshot(
    iree_hal_streaming_context_t* context, bool include_non_blocking_streams,
    bool flush_before_wait, iree_status_t status) {
  unsigned long long max_stream_id = 0;
  iree_slim_mutex_lock(&context->stream_list_mutex);
  for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
    if (context->streams[i]) {
      max_stream_id = iree_max(max_stream_id, context->streams[i]->stream_id);
    }
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  unsigned long long stream_id = 0;
  for (;;) {
    iree_hal_streaming_stream_t* stream = NULL;
    unsigned long long next_stream_id = UINT64_MAX;
    iree_slim_mutex_lock(&context->stream_list_mutex);
    for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
      iree_hal_streaming_stream_t* candidate = context->streams[i];
      if (candidate && candidate->stream_id > stream_id &&
          candidate->stream_id <= max_stream_id &&
          candidate->stream_id < next_stream_id) {
        stream = candidate;
        next_stream_id = candidate->stream_id;
      }
    }
    iree_hal_streaming_stream_retain(stream);
    iree_slim_mutex_unlock(&context->stream_list_mutex);
    if (!stream) {
      break;
    }
    stream_id = next_stream_id;
    if (flush_before_wait) {
      status =
          iree_status_join(status, iree_hal_streaming_stream_flush(stream));
    }
    if (include_non_blocking_streams ||
        !(stream->flags & IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING)) {
      status = iree_status_join(
          status, iree_hal_streaming_stream_synchronize_flushed(stream));
    }
    iree_hal_streaming_stream_release(stream);
  }

  if (context->default_stream) {
    if (flush_before_wait) {
      status = iree_status_join(
          status, iree_hal_streaming_stream_flush(context->default_stream));
    }
    status = iree_status_join(
        status,
        iree_hal_streaming_stream_synchronize_flushed(context->default_stream));
  }
  if (include_non_blocking_streams) {
    status = iree_status_join(
        status, iree_hal_streaming_context_synchronize_event_records(context));
  }
  return status;
}

static iree_status_t iree_hal_streaming_context_synchronize_streams(
    iree_hal_streaming_context_t* context, bool include_non_blocking_streams,
    bool flush_before_wait) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Retain one stable stream set for both submission and draining. This keeps
  // accepted work alive across a post-accept flush failure and gives the
  // synchronization operation a finite target set; streams registered later
  // are outside this call.
  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t snapshot_status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  if (!iree_status_is_ok(snapshot_status)) {
    // Preserve the allocation failure while still draining every stream that
    // can own an accepted tail before cleanup releases borrowed resources.
    iree_status_t status =
        iree_hal_streaming_context_synchronize_streams_without_snapshot(
            context, include_non_blocking_streams, flush_before_wait,
            snapshot_status);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_status_t status = iree_ok_status();
  if (flush_before_wait) {
    // Attempt every flush. A stream may publish its accepted tail before its
    // queue reports a flush failure, and a failure on one stream must not
    // prevent other retained streams from being submitted or drained.
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (streams_copy[i]) {
        status = iree_status_join(
            status, iree_hal_streaming_stream_flush(streams_copy[i]));
      }
    }
    if (context->default_stream) {
      status = iree_status_join(
          status, iree_hal_streaming_stream_flush(context->default_stream));
    }
  }

  // Drain every selected stream even when a preceding flush or wait failed.
  // Legacy default-stream ordering intentionally excludes non-blocking streams;
  // device/context-wide synchronization includes them.
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_stream_t* stream = streams_copy[i];
    if (!stream) {
      continue;
    }
    if (!include_non_blocking_streams &&
        (stream->flags & IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING)) {
      continue;
    }
    status = iree_status_join(
        status, iree_hal_streaming_stream_synchronize_flushed(stream));
  }

  // The default stream may not be in the registered stream list and always
  // participates in its own ordering.
  if (context->default_stream) {
    status = iree_status_join(
        status,
        iree_hal_streaming_stream_synchronize_flushed(context->default_stream));
  }
  if (include_non_blocking_streams) {
    status = iree_status_join(
        status, iree_hal_streaming_context_synchronize_event_records(context));
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_synchronize_event_records(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);

  iree_hal_semaphore_t* semaphore = NULL;
  uint64_t value = 0;
  iree_slim_mutex_lock(&context->event_record_mutex);
  if (context->event_record_timeline.pending_value > 0) {
    semaphore = context->event_record_timeline.semaphore;
    value = context->event_record_timeline.pending_value;
    iree_hal_semaphore_retain(semaphore);
  }
  iree_slim_mutex_unlock(&context->event_record_mutex);

  iree_status_t status = iree_ok_status();
  if (semaphore) {
    status =
        iree_hal_semaphore_wait(semaphore, value, iree_infinite_timeout(),
                                iree_hal_streaming_context_wait_flags(context));
  }
  iree_hal_semaphore_release(semaphore);
  return status;
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

  // Preserve the first failure, but every retained context must still publish
  // and drain its accepted tails before a caller can use this as a quiescence
  // boundary for unretained resources.
  for (iree_host_size_t i = 0; i < context_count; ++i) {
    status = iree_status_join(
        status, iree_hal_streaming_context_synchronize_streams(
                    contexts[i], /*include_non_blocking_streams=*/true,
                    /*flush_before_wait=*/true));
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

iree_status_t iree_hal_streaming_context_wait_blocking_streams(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_host_size_t source_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_stream_t* source_stream = streams_copy[i];
    if (!source_stream || source_stream == stream ||
        source_stream == context->default_stream ||
        iree_any_bit_set(source_stream->flags,
                         IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING)) {
      continue;
    }
    iree_hal_streaming_capture_status_t capture_status =
        IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
    status =
        iree_hal_streaming_capture_status(source_stream, &capture_status, NULL);
    if (!iree_status_is_ok(status) ||
        capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      if (!iree_status_is_ok(status)) {
        break;
      }
      continue;
    }
    // Partition selected sources into the front while preserving every
    // retained entry for snapshot release below.
    iree_hal_streaming_stream_t* displaced_stream = streams_copy[source_count];
    streams_copy[source_count++] = source_stream;
    streams_copy[i] = displaced_stream;
  }
  if (source_count > 0) {
    status = iree_hal_streaming_stream_wait_streams(stream, streams_copy,
                                                    source_count);
  }

  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_query(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_completion_state_t* out_completion_state) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_completion_state);

  iree_hal_streaming_completion_state_t completion_state =
      IREE_HAL_STREAMING_COMPLETION_STATE_COMPLETE;

  iree_hal_streaming_stream_t** streams_copy = NULL;
  iree_host_size_t count = 0;
  iree_status_t query_status = iree_hal_streaming_context_snapshot_streams(
      context, &streams_copy, &count);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(query_status);
       ++i) {
    // The legacy default stream does not order with non-blocking streams.
    if (iree_any_bit_set(streams_copy[i]->flags,
                         IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING)) {
      continue;
    }
    iree_hal_streaming_completion_state_t stream_completion_state =
        IREE_HAL_STREAMING_COMPLETION_STATE_PENDING;
    query_status = iree_hal_streaming_stream_query(streams_copy[i],
                                                   &stream_completion_state);
    if (iree_status_is_ok(query_status) &&
        stream_completion_state ==
            IREE_HAL_STREAMING_COMPLETION_STATE_PENDING) {
      completion_state = IREE_HAL_STREAMING_COMPLETION_STATE_PENDING;
      break;
    }
  }
  iree_hal_streaming_context_release_stream_snapshot(context, streams_copy,
                                                     count);
  if (iree_status_is_ok(query_status)) {
    *out_completion_state = completion_state;
  }
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
