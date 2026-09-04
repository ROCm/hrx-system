// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "common/internal.h"
#include "common/stream.h"
#include "iree/base/internal/math.h"

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

iree_hal_streaming_timestamp_domain_t iree_hal_streaming_query_timestamp_domain(
    const iree_hal_device_spec_t* spec) {
  const iree_hal_streaming_timestamp_domain_t none = {0};
  if (!spec) return none;
  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(spec);
  if (!iree_all_bits_set(timing->flags,
                         IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS)) {
    return none;
  }
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(spec);
  if (queues->family_count != 1) return none;
  const iree_hal_queue_family_spec_t* family = &queues->families[0];
  // Exactly one physical device: the header guarantees a single comparable
  // domain only for one family covering one physical device.
  if (iree_math_count_ones_u64(family->physical_device_affinity) != 1) {
    return none;
  }
  // The flag comes from the device-scope summary and the numbers from the
  // family, which is sound only while the two describe the same domain. On a
  // single-family device they must, so facets that disagree contradict the
  // device's own facts and name no domain to convert with.
  if (timing->timestamp_frequency_hz != family->timestamp_frequency_hz ||
      timing->timestamp_valid_bits != family->timestamp_valid_bits) {
    return none;
  }
  if (family->timestamp_frequency_hz == 0 ||
      family->timestamp_valid_bits == 0 || family->timestamp_valid_bits > 64) {
    return none;
  }
  const iree_hal_streaming_timestamp_domain_t domain = {
      .frequency_hz = family->timestamp_frequency_hz,
      .valid_bits = family->timestamp_valid_bits,
  };
  return domain;
}

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
  context->timestamp_domain = iree_hal_streaming_query_timestamp_domain(
      iree_hal_device_spec(device_entry->hal_device));
  context->flags = flags;
  context->default_stream = NULL;
  context->next_capture_id = 1;
  context->peer_contexts = NULL;
  context->peer_count = 0;
  context->peer_capacity = 0;
  memset(&context->symbol_map, 0, sizeof(context->symbol_map));
  memset(&context->buffer_table, 0, sizeof(context->buffer_table));
  context->pending_free_head = NULL;
  context->pageable_h2d_staging_buffer = NULL;
  context->pageable_h2d_staging_size = 0;
  iree_atomic_store(&context->capture_stream_count, 0,
                    iree_memory_order_relaxed);
  context->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&context->mutex);
  iree_slim_mutex_initialize(&context->pending_free_mutex);

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

  // Initialize buffer mapping table.
  hrx_buffer_table_initialize(&context->buffer_table);

  // Initialize the device timestamp slot pool.
  iree_hal_streaming_event_timestamp_pool_initialize(
      context->device_allocator, host_allocator, &context->timestamp_pool);

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

  iree_hal_streaming_memory_release_pageable_staging(context);

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
  context->stream_count = 0;
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  // Detach under each stream's own mutex, which is the lock
  // iree_hal_streaming_stream_retain_context reads the context under, so no
  // reader can be part way through that read when the field is cleared. A
  // reader that gets there first is refused anyway: it retains through
  // iree_hal_streaming_context_try_retain, which fails once the last reference
  // is gone, and an unpublished context has no such reader to refuse. The list
  // mutex is not held: end_capture holds a stream mutex while walking the list,
  // and taking these in the other order deadlocks against it. The list still
  // holds its reference to every stream, so none can be destroyed while the
  // loop runs; those references are released afterwards, outside both locks,
  // because the last one destroys the stream.
  for (iree_host_size_t i = 0; i < detached_stream_count; ++i) {
    iree_hal_streaming_stream_t* stream = context->streams[i];
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->context == context) {
      stream->queue = NULL;
      stream->context = NULL;
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }
  for (iree_host_size_t i = 0; i < detached_stream_count; ++i) {
    iree_hal_streaming_stream_release(context->streams[i]);
  }

  // Now release the context's reference to default stream.
  iree_hal_streaming_stream_release(default_stream);

  // Free stream tracking resources.
  if (context->streams) {
    iree_allocator_free(context->host_allocator, context->streams);
  }
  iree_slim_mutex_deinitialize(&context->stream_list_mutex);

  // A record draws its slot from the pool of the recording event's own context
  // and every event retains that context, so reaching here means every event
  // that could hold a slot from this pool is gone and every slot is back.
  iree_hal_streaming_event_timestamp_pool_deinitialize(
      &context->timestamp_pool);

  iree_status_ignore(context->loop_status);
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

bool iree_hal_streaming_context_try_retain(
    iree_hal_streaming_context_t* context) {
  if (!context) return false;
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
                         IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING) ||
        source_stream->capture_status !=
            IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
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
    // The legacy default stream does not order with non-blocking streams.
    if (iree_any_bit_set(streams_copy[i]->flags,
                         IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING)) {
      continue;
    }
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
