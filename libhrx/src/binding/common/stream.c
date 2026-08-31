// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/internal.h"
#include "common/kernel_arguments.h"

// Env-gated timing for launch-path investigation. This intentionally uses plain
// counters because the current perf probes run single-threaded and we want the
// lowest possible instrumentation overhead.
typedef struct hrx_launch_timing_counters_t {
  uint64_t launch_count;
  uint64_t launch_total_ns;
  uint64_t launch_begin_ns;
  uint64_t launch_params_ns;
  uint64_t launch_dispatch_ns;
  uint64_t launch_barrier_ns;
  uint64_t flush_count;
  uint64_t flush_total_ns;
  uint64_t flush_end_ns;
  uint64_t flush_execute_ns;
  uint64_t flush_release_ns;
  uint64_t sync_count;
  uint64_t sync_total_ns;
  uint64_t sync_flush_ns;
  uint64_t sync_query_ns;
  uint64_t sync_wait_ns;
} hrx_launch_timing_counters_t;

static hrx_launch_timing_counters_t g_hrx_launch_timing;
static int g_hrx_launch_timing_initialized = 0;
static int g_hrx_launch_timing_enabled = 0;
static int g_hrx_disable_dispatch_barrier_initialized = 0;
static int g_hrx_disable_dispatch_barrier_enabled = 0;
static int g_hrx_flush_each_launch_initialized = 0;
static int g_hrx_flush_each_launch_enabled = 0;
static int g_hrx_flush_interval_initialized = 0;
static int g_hrx_flush_interval = 0;
static int g_hrx_direct_queue_dispatch_initialized = 0;
static int g_hrx_direct_queue_dispatch_enabled = 0;

static uint64_t hrx_launch_timing_now_ns(void) {
  return (uint64_t)iree_time_now();
}

static double hrx_launch_timing_avg_us(uint64_t total_ns, uint64_t count) {
  return count ? (double)total_ns / (double)count / 1000.0 : 0.0;
}

static uint64_t hrx_launch_timing_subtract_or_zero(uint64_t total, uint64_t a,
                                                   uint64_t b, uint64_t c,
                                                   uint64_t d) {
  const uint64_t subtotal = a + b + c + d;
  return total > subtotal ? total - subtotal : 0;
}

static void hrx_launch_timing_dump(void) {
  const uint64_t launch_count = g_hrx_launch_timing.launch_count;
  const uint64_t flush_count = g_hrx_launch_timing.flush_count;
  const uint64_t sync_count = g_hrx_launch_timing.sync_count;
  fprintf(stderr,
          "[HRX_TIMING] launch count=%" PRIu64
          " total_us=%.3f begin_us=%.3f params_us=%.3f dispatch_us=%.3f"
          " barrier_us=%.3f unaccounted_us=%.3f\n",
          launch_count,
          hrx_launch_timing_avg_us(g_hrx_launch_timing.launch_total_ns,
                                   launch_count),
          hrx_launch_timing_avg_us(g_hrx_launch_timing.launch_begin_ns,
                                   launch_count),
          hrx_launch_timing_avg_us(g_hrx_launch_timing.launch_params_ns,
                                   launch_count),
          hrx_launch_timing_avg_us(g_hrx_launch_timing.launch_dispatch_ns,
                                   launch_count),
          hrx_launch_timing_avg_us(g_hrx_launch_timing.launch_barrier_ns,
                                   launch_count),
          hrx_launch_timing_avg_us(hrx_launch_timing_subtract_or_zero(
                                       g_hrx_launch_timing.launch_total_ns,
                                       g_hrx_launch_timing.launch_begin_ns,
                                       g_hrx_launch_timing.launch_params_ns,
                                       g_hrx_launch_timing.launch_dispatch_ns,
                                       g_hrx_launch_timing.launch_barrier_ns),
                                   launch_count));
  fprintf(
      stderr,
      "[HRX_TIMING] flush count=%" PRIu64
      " total_us=%.3f end_us=%.3f execute_us=%.3f release_us=%.3f"
      " unaccounted_us=%.3f\n",
      flush_count,
      hrx_launch_timing_avg_us(g_hrx_launch_timing.flush_total_ns, flush_count),
      hrx_launch_timing_avg_us(g_hrx_launch_timing.flush_end_ns, flush_count),
      hrx_launch_timing_avg_us(g_hrx_launch_timing.flush_execute_ns,
                               flush_count),
      hrx_launch_timing_avg_us(g_hrx_launch_timing.flush_release_ns,
                               flush_count),
      hrx_launch_timing_avg_us(hrx_launch_timing_subtract_or_zero(
                                   g_hrx_launch_timing.flush_total_ns,
                                   g_hrx_launch_timing.flush_end_ns,
                                   g_hrx_launch_timing.flush_execute_ns,
                                   g_hrx_launch_timing.flush_release_ns, 0),
                               flush_count));
  fprintf(
      stderr,
      "[HRX_TIMING] sync count=%" PRIu64
      " total_us=%.3f flush_us=%.3f query_us=%.3f wait_us=%.3f"
      " unaccounted_us=%.3f\n",
      sync_count,
      hrx_launch_timing_avg_us(g_hrx_launch_timing.sync_total_ns, sync_count),
      hrx_launch_timing_avg_us(g_hrx_launch_timing.sync_flush_ns, sync_count),
      hrx_launch_timing_avg_us(g_hrx_launch_timing.sync_query_ns, sync_count),
      hrx_launch_timing_avg_us(g_hrx_launch_timing.sync_wait_ns, sync_count),
      hrx_launch_timing_avg_us(hrx_launch_timing_subtract_or_zero(
                                   g_hrx_launch_timing.sync_total_ns,
                                   g_hrx_launch_timing.sync_flush_ns,
                                   g_hrx_launch_timing.sync_query_ns,
                                   g_hrx_launch_timing.sync_wait_ns, 0),
                               sync_count));
}

static int hrx_launch_timing_enabled(void) {
  if (!g_hrx_launch_timing_initialized) {
    g_hrx_launch_timing_initialized = 1;
    const char* enabled = getenv("HRX_LAUNCH_TIMING");
    g_hrx_launch_timing_enabled = enabled && enabled[0] && enabled[0] != '0';
    if (g_hrx_launch_timing_enabled) {
      atexit(hrx_launch_timing_dump);
    }
  }
  return g_hrx_launch_timing_enabled;
}

static int hrx_disable_dispatch_barrier_enabled(void) {
  if (!g_hrx_disable_dispatch_barrier_initialized) {
    g_hrx_disable_dispatch_barrier_initialized = 1;
    const char* enabled = getenv("HRX_DISABLE_DISPATCH_BARRIER");
    g_hrx_disable_dispatch_barrier_enabled =
        enabled && enabled[0] && enabled[0] != '0';
  }
  return g_hrx_disable_dispatch_barrier_enabled;
}

static int hrx_flush_each_launch_enabled(void) {
  if (!g_hrx_flush_each_launch_initialized) {
    g_hrx_flush_each_launch_initialized = 1;
    const char* enabled = getenv("HRX_FLUSH_EACH_LAUNCH");
    g_hrx_flush_each_launch_enabled =
        enabled && enabled[0] && enabled[0] != '0';
  }
  return g_hrx_flush_each_launch_enabled;
}

static int hrx_flush_interval(void) {
  if (!g_hrx_flush_interval_initialized) {
    g_hrx_flush_interval_initialized = 1;
    const char* value = getenv("HRX_FLUSH_INTERVAL");
    g_hrx_flush_interval = value ? atoi(value) : 0;
    if (g_hrx_flush_interval < 0) g_hrx_flush_interval = 0;
  }
  return g_hrx_flush_interval;
}

static int hrx_direct_queue_dispatch_enabled(void) {
  if (!g_hrx_direct_queue_dispatch_initialized) {
    g_hrx_direct_queue_dispatch_initialized = 1;
    const char* enabled = getenv("HRX_DIRECT_QUEUE_DISPATCH");
    g_hrx_direct_queue_dispatch_enabled =
        enabled && enabled[0] && enabled[0] != '0';
  }
  return g_hrx_direct_queue_dispatch_enabled;
}

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_stream_destroy(
    iree_hal_streaming_stream_t* stream);

static iree_status_t iree_hal_streaming_stream_reserve_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id,
    bool* out_was_added) {
  *out_was_added = false;
  iree_slim_mutex_lock(&stream->mutex);
  for (iree_host_size_t i = 0; i < stream->memory_reuse_dependency_count; ++i) {
    if (stream->memory_reuse_dependencies[i].source_stream_id ==
        source_stream_id) {
      iree_slim_mutex_unlock(&stream->mutex);
      return iree_ok_status();
    }
  }

  iree_status_t status = iree_ok_status();
  if (stream->memory_reuse_dependency_count ==
      stream->memory_reuse_dependency_capacity) {
    // A stream may wait on many producer streams. Grow geometrically so
    // recording those dependencies remains amortized constant time.
    iree_host_size_t new_capacity =
        stream->memory_reuse_dependency_capacity
            ? stream->memory_reuse_dependency_capacity
            : 4;
    iree_host_size_t allocation_size = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(new_capacity, 2, &new_capacity) ||
            !iree_host_size_checked_mul(
                new_capacity, sizeof(*stream->memory_reuse_dependencies),
                &allocation_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream dependency count overflow");
    } else {
      status =
          iree_allocator_realloc(stream->host_allocator, allocation_size,
                                 (void**)&stream->memory_reuse_dependencies);
      if (iree_status_is_ok(status)) {
        stream->memory_reuse_dependency_capacity = new_capacity;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    stream->memory_reuse_dependencies[stream->memory_reuse_dependency_count++] =
        (iree_hal_streaming_memory_reuse_dependency_t){
            .source_stream_id = source_stream_id,
            .source_timeline_value = 0,
        };
    *out_was_added = true;
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return status;
}

static void iree_hal_streaming_stream_record_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id,
    uint64_t source_timeline_value) {
  iree_slim_mutex_lock(&stream->mutex);
  for (iree_host_size_t i = 0; i < stream->memory_reuse_dependency_count; ++i) {
    iree_hal_streaming_memory_reuse_dependency_t* dependency =
        &stream->memory_reuse_dependencies[i];
    if (dependency->source_stream_id == source_stream_id) {
      dependency->source_timeline_value =
          iree_max(dependency->source_timeline_value, source_timeline_value);
      break;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
}

static void
iree_hal_streaming_stream_remove_uncommitted_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id) {
  iree_slim_mutex_lock(&stream->mutex);
  for (iree_host_size_t i = 0; i < stream->memory_reuse_dependency_count; ++i) {
    iree_hal_streaming_memory_reuse_dependency_t* dependency =
        &stream->memory_reuse_dependencies[i];
    if (dependency->source_stream_id == source_stream_id &&
        dependency->source_timeline_value == 0) {
      --stream->memory_reuse_dependency_count;
      stream->memory_reuse_dependencies[i] =
          stream->memory_reuse_dependencies
              [stream->memory_reuse_dependency_count];
      break;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
}

bool iree_hal_streaming_stream_has_memory_reuse_dependency(
    iree_hal_streaming_stream_t* stream, unsigned long long source_stream_id,
    uint64_t source_timeline_value) {
  bool has_dependency = false;
  iree_slim_mutex_lock(&stream->mutex);
  for (iree_host_size_t i = 0; i < stream->memory_reuse_dependency_count; ++i) {
    const iree_hal_streaming_memory_reuse_dependency_t* dependency =
        &stream->memory_reuse_dependencies[i];
    if (dependency->source_stream_id == source_stream_id &&
        dependency->source_timeline_value >= source_timeline_value) {
      has_dependency = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return has_dependency;
}

iree_status_t iree_hal_streaming_stream_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_flags_t flags, int priority,
    iree_allocator_t host_allocator, iree_hal_streaming_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void**)&stream));
  iree_atomic_ref_count_init(&stream->ref_count);
  stream->context = context;
  stream->flags = flags;
  stream->priority = priority;
  stream->stream_id = 0;
  stream->command_buffer = NULL;
  stream->pending_launch_count = 0;
  stream->timeline_semaphore = NULL;
  stream->pending_value = 0;
  stream->completed_value = 0;
  stream->queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  stream->memory_reuse_dependencies = NULL;
  stream->memory_reuse_dependency_count = 0;
  stream->memory_reuse_dependency_capacity = 0;

  stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  stream->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
  stream->capture_graph = NULL;
  stream->capture_graph_owned = false;
  stream->capture_origin = false;
  stream->capture_joined_to_origin = false;
  stream->capture_id = 0;
  stream->capture_owner_thread_id = 0;
  stream->capture_dependencies = NULL;
  stream->capture_dependency_count = 0;
  stream->capture_dependency_capacity = 0;

  stream->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&stream->mutex);

  // Create timeline semaphore for synchronization.
  iree_status_t status = iree_hal_semaphore_create(
      context->device, IREE_HAL_QUEUE_AFFINITY_ANY, 0ULL,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &stream->timeline_semaphore);

  // Register stream with context.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_register_stream(context, stream);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_hal_streaming_stream_destroy(stream);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_stream_destroy(
    iree_hal_streaming_stream_t* stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_streaming_context_t* context = stream->context;
  iree_slim_mutex_unlock(&stream->mutex);
  if (context) {
    iree_status_ignore(iree_hal_streaming_stream_synchronize(stream));
    if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_NONE);
    }
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->context == context) stream->context = NULL;
    iree_slim_mutex_unlock(&stream->mutex);
    iree_hal_streaming_context_unregister_stream(context, stream);
  }

  iree_allocator_free(stream->host_allocator,
                      stream->memory_reuse_dependencies);

  // Release command buffer.
  iree_hal_command_buffer_release(stream->command_buffer);

  if (stream->capture_graph_owned) {
    iree_hal_streaming_graph_release(stream->capture_graph);
  }
  iree_allocator_free(stream->host_allocator, stream->capture_dependencies);

  // Release timeline semaphore.
  iree_hal_semaphore_release(stream->timeline_semaphore);

  // Deinitialize synchronization.
  iree_slim_mutex_deinitialize(&stream->mutex);

  // Free stream memory.
  const iree_allocator_t host_allocator = stream->host_allocator;
  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_stream_retain(iree_hal_streaming_stream_t* stream) {
  if (stream) {
    iree_atomic_ref_count_inc(&stream->ref_count);
  }
}

void iree_hal_streaming_stream_release(iree_hal_streaming_stream_t* stream) {
  if (stream && iree_atomic_ref_count_dec(&stream->ref_count) == 1) {
    iree_hal_streaming_stream_destroy(stream);
  }
}

bool iree_hal_streaming_stream_retain_context(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_streaming_context_t* context = stream->context;
  if (iree_hal_streaming_context_try_retain(context)) {
    *out_context = context;
  }
  iree_slim_mutex_unlock(&stream->mutex);
  return *out_context != NULL;
}

iree_status_t iree_hal_streaming_stream_begin_locked(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);

  // Create command buffer if not already created. HIP pointer lifetime is
  // enforced by free/unregister ordering, not by launch-time pointer discovery:
  // synchronous destruction waits for active streams, and asynchronous
  // destruction queues the release after prior stream work.
  iree_status_t status = iree_ok_status();
  if (!stream->command_buffer) {
    status = iree_hal_command_buffer_create(
        stream->context->device,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
        IREE_HAL_COMMAND_CATEGORY_ANY, stream->queue_affinity,
        /*binding_capacity=*/0, &stream->command_buffer);
    if (!iree_status_is_ok(status)) return status;
    status = iree_hal_command_buffer_begin(stream->command_buffer);
  }

  return status;
}

iree_status_t iree_hal_streaming_stream_begin(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status = iree_hal_streaming_stream_begin_locked(stream);
  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_flush(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  const int timing_enabled = hrx_launch_timing_enabled();
  const uint64_t timing_start_ns =
      timing_enabled ? hrx_launch_timing_now_ns() : 0;
  uint64_t timing_end_ns = 0;
  uint64_t timing_execute_ns = 0;
  uint64_t timing_release_ns = 0;
  iree_slim_mutex_lock(&stream->mutex);

  iree_status_t status = iree_ok_status();
  if (stream->command_buffer) {
    // End recording and submit command buffer.
    uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    status = iree_hal_command_buffer_end(stream->command_buffer);
    if (timing_enabled) {
      timing_end_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }

    // Wait for the previous submission. This chains each flush after the one
    // before it, so that operations split across multiple command buffers
    // (e.g. by an intervening hipMemcpy) still execute in order.
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    status = iree_hal_streaming_stream_reserve_next_value_locked(
        stream, &wait_value, &signal_value);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }

    // Submit to device queue with timeline semaphore.
    // Wait for the previous submission to complete before executing.
    iree_hal_queue_affinity_t queue_affinity = stream->queue_affinity;
    iree_hal_semaphore_list_t wait_semaphores = {
        .count = wait_value > 0
                     ? 1
                     : 0,  // Only wait if there was a previous submission.
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &wait_value,
    };
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &signal_value,
    };

    timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    status = iree_hal_device_queue_execute(
        stream->context->device, queue_affinity, wait_semaphores,
        signal_semaphores, stream->command_buffer,
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      // The accepted submission owns the value it signals, so the timeline
      // advances here and stays advanced even when the flush below fails.
      stream->pending_value = signal_value;
      status =
          iree_hal_device_queue_flush(stream->context->device, queue_affinity);
    }
    if (timing_enabled) {
      timing_execute_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }

    // Release command buffer (we're done with it).
    timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    iree_hal_command_buffer_release(stream->command_buffer);
    stream->command_buffer = NULL;
    stream->pending_launch_count = 0;
    if (timing_enabled) {
      timing_release_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
  }

  iree_slim_mutex_unlock(&stream->mutex);
  if (timing_enabled) {
    ++g_hrx_launch_timing.flush_count;
    g_hrx_launch_timing.flush_total_ns +=
        hrx_launch_timing_now_ns() - timing_start_ns;
    g_hrx_launch_timing.flush_end_ns += timing_end_ns;
    g_hrx_launch_timing.flush_execute_ns += timing_execute_ns;
    g_hrx_launch_timing.flush_release_ns += timing_release_ns;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_query(
    iree_hal_streaming_stream_t* stream, int* status) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(status);

  IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_flush(stream));

  uint64_t current_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_query(stream->timeline_semaphore, &current_value));

  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t pending_value = stream->pending_value;
  if (current_value >= pending_value) {
    *status = 0;  // Complete
    stream->completed_value = iree_max(stream->completed_value, current_value);
  } else {
    *status = 1;  // Not complete
  }
  iree_slim_mutex_unlock(&stream->mutex);

  return iree_ok_status();
}

typedef struct iree_hal_streaming_wait_dependency_t {
  iree_hal_streaming_stream_t* source_stream;
  uint64_t source_timeline_value;
  bool added_memory_reuse_dependency;
} iree_hal_streaming_wait_dependency_t;

enum { IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT = 8 };

iree_status_t iree_hal_streaming_stream_wait_streams(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_stream_t* const* source_streams,
    iree_host_size_t source_stream_count) {
  IREE_ASSERT_ARGUMENT(stream);
  if (source_stream_count == 0) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(source_streams);

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&stream->mutex);
  const bool destination_is_capturing =
      stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  iree_slim_mutex_unlock(&stream->mutex);
  if (destination_is_capturing) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "destination stream is capturing");
  }

  iree_status_t status = iree_ok_status();
  iree_hal_streaming_wait_dependency_t
      inline_dependencies[IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT];
  iree_hal_semaphore_t*
      inline_wait_semaphores[IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT +
                             1];
  uint64_t
      inline_wait_values[IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT + 1];
  iree_hal_streaming_wait_dependency_t* dependencies = inline_dependencies;
  iree_hal_semaphore_t** wait_semaphores_storage = inline_wait_semaphores;
  uint64_t* wait_values_storage = inline_wait_values;
  bool uses_heap_storage =
      source_stream_count > IREE_HAL_STREAMING_INLINE_WAIT_DEPENDENCY_COUNT;
  if (uses_heap_storage) {
    iree_host_size_t wait_capacity = 0;
    iree_host_size_t dependencies_size = 0;
    iree_host_size_t semaphores_size = 0;
    iree_host_size_t values_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(source_stream_count, 1,
                                                  &wait_capacity) ||
                      !iree_host_size_checked_mul(
                          source_stream_count,
                          sizeof(iree_hal_streaming_wait_dependency_t),
                          &dependencies_size) ||
                      !iree_host_size_checked_mul(wait_capacity,
                                                  sizeof(iree_hal_semaphore_t*),
                                                  &semaphores_size) ||
                      !iree_host_size_checked_mul(
                          wait_capacity, sizeof(uint64_t), &values_size))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "stream dependency count overflow");
    }
    dependencies = NULL;
    wait_semaphores_storage = NULL;
    wait_values_storage = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(stream->host_allocator, dependencies_size,
                                     (void**)&dependencies);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(stream->host_allocator, semaphores_size,
                                     (void**)&wait_semaphores_storage);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(stream->host_allocator, values_size,
                                     (void**)&wait_values_storage);
    }
  }

  iree_host_size_t dependency_count = 0;
  for (iree_host_size_t i = 0;
       i < source_stream_count && iree_status_is_ok(status); ++i) {
    iree_hal_streaming_stream_t* source_stream = source_streams[i];
    if (!source_stream || source_stream == stream) continue;
    if (source_stream->context != stream->context) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stream dependency crosses contexts");
      break;
    }
    iree_slim_mutex_lock(&source_stream->mutex);
    const bool source_is_capturing =
        source_stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
    iree_slim_mutex_unlock(&source_stream->mutex);
    if (source_is_capturing) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "source stream is capturing");
      break;
    }

    // Flush first so the captured value names all source work preceding this
    // operation. A source that has never submitted contributes no wait.
    status = iree_hal_streaming_stream_flush(source_stream);
    if (!iree_status_is_ok(status)) break;
    iree_slim_mutex_lock(&source_stream->mutex);
    const uint64_t source_timeline_value = source_stream->pending_value;
    iree_slim_mutex_unlock(&source_stream->mutex);

    status = iree_status_join(
        status,
        iree_hal_streaming_memory_release_completed_async_frees(source_stream));
    if (!iree_status_is_ok(status) || source_timeline_value == 0) continue;

    iree_hal_streaming_wait_dependency_t* dependency =
        &dependencies[dependency_count];
    dependency->source_stream = source_stream;
    dependency->source_timeline_value = source_timeline_value;
    dependency->added_memory_reuse_dependency = false;
    status = iree_hal_streaming_stream_reserve_memory_reuse_dependency(
        stream, source_stream->stream_id,
        &dependency->added_memory_reuse_dependency);
    if (iree_status_is_ok(status)) {
      ++dependency_count;
    }
  }

  // Submit prior destination work before appending a queue dependency. The
  // barrier advances its timeline only after every captured source point.
  bool submitted = false;
  if (iree_status_is_ok(status) && dependency_count > 0) {
    status = iree_hal_streaming_stream_flush(stream);
  }
  if (iree_status_is_ok(status) && dependency_count > 0) {
    iree_slim_mutex_lock(&stream->mutex);
    uint64_t destination_timeline_value = 0;
    uint64_t signal_value = 0;
    status = iree_hal_streaming_stream_reserve_next_value_locked(
        stream, &destination_timeline_value, &signal_value);
    if (iree_status_is_ok(status)) {
      iree_host_size_t wait_count = 0;
      if (destination_timeline_value > 0) {
        wait_semaphores_storage[wait_count] = stream->timeline_semaphore;
        wait_values_storage[wait_count] = destination_timeline_value;
        ++wait_count;
      }
      for (iree_host_size_t i = 0; i < dependency_count; ++i) {
        wait_semaphores_storage[wait_count] =
            dependencies[i].source_stream->timeline_semaphore;
        wait_values_storage[wait_count] = dependencies[i].source_timeline_value;
        ++wait_count;
      }
      const iree_hal_semaphore_list_t wait_semaphores = {
          .count = wait_count,
          .semaphores = wait_semaphores_storage,
          .payload_values = wait_values_storage,
      };
      const iree_hal_semaphore_list_t signal_semaphores = {
          .count = 1,
          .semaphores = &stream->timeline_semaphore,
          .payload_values = &signal_value,
      };
      status = iree_hal_device_queue_barrier(
          stream->context->device, stream->queue_affinity, wait_semaphores,
          signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE);
      if (iree_status_is_ok(status)) {
        // The accepted barrier owns the value it signals, so the timeline
        // advances here and stays advanced even when the flush below fails.
        submitted = true;
        stream->pending_value = signal_value;
        status = iree_hal_device_queue_flush(stream->context->device,
                                             stream->queue_affinity);
      }
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }

  for (iree_host_size_t i = 0; i < dependency_count; ++i) {
    iree_hal_streaming_wait_dependency_t* dependency = &dependencies[i];
    if (submitted) {
      iree_hal_streaming_stream_record_memory_reuse_dependency(
          stream, dependency->source_stream->stream_id,
          dependency->source_timeline_value);
    } else if (dependency->added_memory_reuse_dependency) {
      iree_hal_streaming_stream_remove_uncommitted_memory_reuse_dependency(
          stream, dependency->source_stream->stream_id);
    }
  }

  if (uses_heap_storage) {
    iree_allocator_free(stream->host_allocator, wait_values_storage);
    iree_allocator_free(stream->host_allocator, wait_semaphores_storage);
    iree_allocator_free(stream->host_allocator, dependencies);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_wait_stream(
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_stream_t* source_stream) {
  IREE_ASSERT_ARGUMENT(source_stream);
  return iree_hal_streaming_stream_wait_streams(stream, &source_stream, 1);
}

static iree_status_t iree_hal_streaming_stream_synchronize_impl(
    iree_hal_streaming_stream_t* stream, bool flush_context) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  const int timing_enabled = hrx_launch_timing_enabled();
  const uint64_t timing_start_ns =
      timing_enabled ? hrx_launch_timing_now_ns() : 0;
  uint64_t timing_flush_ns = 0;
  uint64_t timing_query_ns = 0;
  uint64_t timing_wait_ns = 0;

  uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
  if (flush_context) {
    // HIP launches are logically submitted work even when HRX batches command
    // buffer recording. Before waiting on a stream, submit all pending work in
    // the same context so stream-ordered dependencies and device-side waits can
    // make forward progress without repeatedly flushing unrelated contexts.
    iree_status_t flush_status =
        iree_hal_streaming_context_flush(stream->context);
    if (timing_enabled) {
      timing_flush_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, flush_status);
  }

  // Snapshot the synchronization target under the same mutex used to reserve
  // timeline values. Operations submitted after this point are not part of
  // this synchronization operation.
  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t target_value = stream->pending_value;
  uint64_t completed_value = stream->completed_value;
  iree_slim_mutex_unlock(&stream->mutex);

  timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
  uint64_t current_value = 0;
  iree_status_t query_status =
      iree_hal_semaphore_query(stream->timeline_semaphore, &current_value);
  if (iree_status_is_ok(query_status)) {
    if (current_value > completed_value) {
      completed_value = current_value;
    }
  }
  if (timing_enabled) {
    timing_query_ns += hrx_launch_timing_now_ns() - timing_step_ns;
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, query_status);

  // Wait for the timeline semaphore to reach the captured target.
  if (target_value > completed_value) {
    timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    iree_status_t wait_status = iree_hal_semaphore_wait(
        stream->timeline_semaphore, target_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
    if (timing_enabled) {
      timing_wait_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
    if (!iree_status_is_ok(wait_status)) {
      IREE_TRACE_ZONE_END(z0);
      return wait_status;
    }
    completed_value = target_value;
  }

  iree_slim_mutex_lock(&stream->mutex);
  stream->completed_value = iree_max(stream->completed_value, completed_value);
  iree_slim_mutex_unlock(&stream->mutex);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_release_completed_async_frees(stream));

  if (timing_enabled) {
    ++g_hrx_launch_timing.sync_count;
    g_hrx_launch_timing.sync_total_ns +=
        hrx_launch_timing_now_ns() - timing_start_ns;
    g_hrx_launch_timing.sync_flush_ns += timing_flush_ns;
    g_hrx_launch_timing.sync_query_ns += timing_query_ns;
    g_hrx_launch_timing.sync_wait_ns += timing_wait_ns;
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_stream_synchronize(
    iree_hal_streaming_stream_t* stream) {
  return iree_hal_streaming_stream_synchronize_impl(stream,
                                                    /*flush_context=*/true);
}

iree_status_t iree_hal_streaming_stream_synchronize_flushed(
    iree_hal_streaming_stream_t* stream) {
  return iree_hal_streaming_stream_synchronize_impl(stream,
                                                    /*flush_context=*/false);
}

iree_status_t iree_hal_streaming_stream_wait_submitted(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for already-submitted work to complete WITHOUT flushing.
  // Snapshot the tail of accepted work under the stream lock. New submissions
  // after this point are outside this wait operation.
  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t pending_value = stream->pending_value;
  iree_slim_mutex_unlock(&stream->mutex);
  if (pending_value > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_wait(stream->timeline_semaphore, pending_value,
                                    iree_infinite_timeout(),
                                    IREE_ASYNC_WAIT_FLAG_NONE));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Joins |stream| to the capture |event|'s last capture-time record belongs to,
// adopting |capture_graph| as |stream|'s own capture when the stream is not
// already capturing and then adding the event's dependency frontier to the
// stream's.
//
// |capture_graph| is borrowed for the call; a stream that adopts it takes its
// own reference.
static iree_status_t iree_hal_streaming_stream_wait_captured_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event,
    iree_hal_streaming_graph_t* capture_graph) {
  bool adopt_capture_graph = false;
  iree_slim_mutex_lock(&stream->mutex);
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
    adopt_capture_graph = true;
  } else if (stream->capture_graph != capture_graph) {
    iree_slim_mutex_unlock(&stream->mutex);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "event wait crosses different active capture graphs");
  }
  iree_slim_mutex_unlock(&stream->mutex);

  if (adopt_capture_graph) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_stream_flush(stream));

    unsigned long long capture_id = 0;
    if (!event->recording_stream) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_context_allocate_capture_id(
          stream->context, &capture_id));
    }

    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      stream->capture_graph = capture_graph;
      stream->capture_graph_owned = true;
      stream->capture_origin = false;
      stream->capture_joined_to_origin = false;
      if (event->recording_stream) {
        stream->capture_mode = event->recording_stream->capture_mode;
        stream->capture_id = event->recording_stream->capture_id;
        stream->capture_owner_thread_id =
            event->recording_stream->capture_owner_thread_id;
      } else {
        stream->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
        stream->capture_id = capture_id;
        stream->capture_owner_thread_id = 0;
      }
      iree_hal_streaming_graph_retain(stream->capture_graph);
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE);
    } else if (stream->capture_graph != capture_graph) {
      iree_slim_mutex_unlock(&stream->mutex);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "event wait crosses different active capture graphs");
    }
    iree_slim_mutex_unlock(&stream->mutex);
  }

  return iree_hal_streaming_update_capture_dependencies(
      stream, event->capture_dependencies, event->capture_dependency_count,
      IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD);
}

iree_status_t iree_hal_streaming_stream_wait_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event,
    bool capture_external_wait) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  // An external wait remains an explicit node so each graph launch resolves
  // the event point supplied by the application at execution time.
  if (capture_external_wait &&
      stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_event_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count,
                IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT, event, &node));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_set_last_node(stream, node));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // A capture-time record leaves the event naming the graph it was captured
  // into. A wait on such an event joins that capture and submits no timeline
  // wait. The association is read once and held for the whole branch, so the
  // graph the branch works from cannot be freed underneath it.
  iree_hal_streaming_graph_t* capture_graph =
      iree_hal_streaming_event_acquire_capture_graph(event);
  if (capture_graph) {
    const iree_status_t capture_status =
        iree_hal_streaming_stream_wait_captured_event(stream, event,
                                                      capture_graph);
    // Released with no lock held: the last reference to a graph frees the
    // allocations it owns, which synchronizes every context and relocks this
    // stream.
    iree_hal_streaming_graph_release(capture_graph);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, capture_status);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Read the point once. The barrier below waits on exactly the point whose
  // stream ordering is filed as a reuse dependency, so a record landing on this
  // event concurrently cannot make the filed dependency describe a point other
  // than the one waited on.
  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);

  // Waiting on the point orders every later submission on this stream behind
  // it, and therefore behind the stream timeline point it follows. That is the
  // ordering a deferred free on that stream needs before its allocation may be
  // reused here. A point that follows no stream timeline point files nothing:
  // there is no ordering to claim. A point on this stream's own timeline files
  // nothing either, because same-stream ordering already covers it.
  const unsigned long long source_stream_id =
      recorded_point.ordered_after_stream_id;
  const uint64_t source_timeline_value =
      recorded_point.ordered_after_stream_value;
  const bool files_memory_reuse_dependency =
      source_stream_id != 0 && source_stream_id != stream->stream_id &&
      source_timeline_value != 0;
  bool added_memory_reuse_dependency = false;
  // True once the queue has accepted the barrier that establishes the ordering.
  bool submitted = false;
  iree_status_t status = iree_ok_status();
  if (files_memory_reuse_dependency) {
    status = iree_hal_streaming_stream_reserve_memory_reuse_dependency(
        stream, source_stream_id, &added_memory_reuse_dependency);
  }

  // Flush the stream to ensure all prior operations are submitted.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_flush(stream);
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&stream->mutex);

    // Reserve the next stream timeline value and submit a barrier that
    // completes only after the event is signaled. The value is submitted, not
    // completed; query/synchronize advance completed_value after observing the
    // semaphore.
    uint64_t wait_value = 0;
    uint64_t signal_value = 0;
    status = iree_hal_streaming_stream_reserve_next_value_locked(
        stream, &wait_value, &signal_value);
    if (iree_status_is_ok(status)) {
      // The barrier waits on the point the event was recorded at and on
      // everything already on this stream, so the value it signals stays behind
      // the value below it. Either wait is dropped when there is nothing behind
      // it: a stream that has never submitted, or an event with no submitted
      // record.
      iree_hal_semaphore_t* wait_semaphore_storage[2];
      uint64_t wait_value_storage[2];
      iree_host_size_t wait_count = 0;
      if (wait_value > 0) {
        wait_semaphore_storage[wait_count] = stream->timeline_semaphore;
        wait_value_storage[wait_count] = wait_value;
        ++wait_count;
      }
      if (recorded_point.semaphore) {
        wait_semaphore_storage[wait_count] = recorded_point.semaphore;
        wait_value_storage[wait_count] = recorded_point.value;
        ++wait_count;
      }
      iree_hal_semaphore_list_t wait_semaphores = {
          .count = wait_count,
          .semaphores = wait_semaphore_storage,
          .payload_values = wait_value_storage,
      };
      iree_hal_semaphore_list_t signal_semaphores = {
          .count = 1,
          .semaphores = &stream->timeline_semaphore,
          .payload_values = &signal_value,
      };

      status = iree_hal_device_queue_barrier(
          stream->context->device, stream->queue_affinity, wait_semaphores,
          signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE);
      if (iree_status_is_ok(status)) {
        // The accepted barrier owns the value it signals, so the timeline
        // advances here and stays advanced even when the flush below fails.
        submitted = true;
        stream->pending_value = signal_value;
        status = iree_hal_device_queue_flush(stream->context->device,
                                             stream->queue_affinity);
      }
    }

    iree_slim_mutex_unlock(&stream->mutex);
  }

  iree_hal_streaming_event_release_recorded_point(&recorded_point);

  // The reservation holds the dependency slot from before the submission so a
  // concurrent reservation for the same source stream cannot displace it. It
  // carries no value until the submission that establishes the ordering is
  // accepted, and is withdrawn when that submission is not.
  if (files_memory_reuse_dependency) {
    if (submitted) {
      iree_hal_streaming_stream_record_memory_reuse_dependency(
          stream, source_stream_id, source_timeline_value);
    } else if (added_memory_reuse_dependency) {
      iree_hal_streaming_stream_remove_uncommitted_memory_reuse_dependency(
          stream, source_stream_id);
    }
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

typedef struct iree_hal_streaming_launch_arguments_t {
  // Native constants passed to HAL dispatch.
  void* constants;
  // Number of bytes in |constants|.
  iree_host_size_t constants_size;
  // HAL buffer bindings passed to dispatch.
  iree_hal_buffer_ref_list_t bindings;
  // Whether |constants| already contains a target-native argument image.
  bool use_raw_arguments;
} iree_hal_streaming_launch_arguments_t;

static iree_status_t iree_hal_streaming_prepare_launch_arguments(
    const iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_context_t* context, bool is_native_kernel,
    bool is_empty_native_kernel, bool is_pre_packed, bool is_args_array,
    uint8_t* argument_storage, iree_host_size_t constants_storage_size,
    iree_host_size_t constants_capacity, iree_host_size_t binding_capacity,
    iree_hal_streaming_launch_arguments_t* out_arguments) {
  *out_arguments = (iree_hal_streaming_launch_arguments_t){
      .constants = constants_capacity ? argument_storage : NULL,
      .constants_size = symbol->parameters.constant_bytes,
      .bindings =
          {
              .count = binding_capacity,
              .values = binding_capacity
                            ? (iree_hal_buffer_ref_t*)(argument_storage +
                                                       constants_storage_size)
                            : NULL,
          },
      .use_raw_arguments = false,
  };

  if (is_pre_packed) {
    // Pre-packed buffers are already in native kernarg layout. They may
    // contain device pointers either as formal pointer arguments or inside
    // opaque data, so preserve the bytes exactly and rely on HIP lifetime
    // ordering instead of trying to translate visible pointer slots.
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_validate_prepacked_kernel_arguments(symbol, params));
    out_arguments->constants = params->buffer;
    out_arguments->constants_size = params->buffer_size;
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  if (is_args_array) {
    // Pointer-array launches are converted to native kernarg bytes. This keeps
    // formal pointer arguments and pointers nested inside copied structs on the
    // same device-pointer contract.
    out_arguments->constants_size = symbol->parameters.direct_arg_bytes
                                        ? symbol->parameters.direct_arg_bytes
                                        : symbol->parameters.constant_bytes;
    if (out_arguments->constants_size == 0) {
      out_arguments->constants_size = symbol->parameters.buffer_size;
    }
    IREE_RETURN_IF_ERROR(iree_hal_streaming_pack_raw_argument_list(
        &symbol->parameters, (void**)params->buffer, out_arguments->constants,
        &out_arguments->constants_size));
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  if (is_native_kernel && params->buffer) {
    // Native kernel with pre-packed buffer: pass raw arguments directly. This
    // path is only valid when the caller did not declare the buffer as a HIP
    // args array, because a void** parameter list is not a native kernarg pack.
    out_arguments->constants = params->buffer;
    if (params->buffer_size > 0) {
      out_arguments->constants_size = params->buffer_size;
    }
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  if (params->buffer) {
    // Unflagged launches use the shared metadata path. HIP entry points mark
    // native kernarg bytes with PRE_PACKED or provide ARGS_ARRAY so device
    // pointer values are preserved without depending on reflected pointer
    // slots.
    iree_status_t status = iree_hal_streaming_unpack_parameters(
        context, &symbol->parameters, params->buffer, out_arguments->constants,
        &out_arguments->bindings);
    if (iree_status_is_ok(status)) return status;
    if (iree_status_code(status) != IREE_STATUS_NOT_FOUND) return status;

    // External device pointers cannot be expressed as HAL bindings. Preserve
    // the caller's raw argument image instead of partially translating it.
    iree_status_ignore(status);
    out_arguments->constants = params->buffer;
    out_arguments->constants_size = params->buffer_size;
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  if (is_empty_native_kernel) {
    out_arguments->constants = NULL;
    out_arguments->constants_size = 0;
    out_arguments->bindings.count = 0;
    out_arguments->use_raw_arguments = true;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "kernel launch missing parameter storage");
}

iree_status_t iree_hal_streaming_launch_kernel(
    iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  const int timing_enabled = hrx_launch_timing_enabled();
  const uint64_t timing_start_ns =
      timing_enabled ? hrx_launch_timing_now_ns() : 0;
  uint64_t timing_begin_ns = 0;
  uint64_t timing_params_ns = 0;
  uint64_t timing_dispatch_ns = 0;
  uint64_t timing_barrier_ns = 0;
  const bool direct_queue_dispatch_requested =
      hrx_direct_queue_dispatch_enabled();

  // Verify the symbol is a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a function (type=%d)", symbol->type);
  }

  // Check if cooperative launch is requested.
  if (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE) {
    // Cooperative launch requires a backend dispatch mode that reserves the
    // full grid concurrently. The HAL dispatch path does not expose that
    // contract, so fail loudly.
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "cooperative kernel launch not yet implemented in HAL layer");
  }

  // Verify parameter storage early for metadata-described launches. The
  // concrete branch below decides whether the storage is a native byte image,
  // a pointer array to pack, or a shared unflagged metadata buffer.
  if (!params->buffer && symbol->parameters.buffer_size > 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "direct kernel launch missing expected parameters");
  }

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Add kernel node to the graph instead of recording to command buffer.
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_kernel_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, symbol, params, &node));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_set_last_node(stream, node));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Ensure prior command-buffer work is submitted before direct dispatches.
  // Direct dispatches use the stream timeline wait/signal chain below; command
  // buffer dispatches continue recording into the current stream command
  // buffer.
  if (direct_queue_dispatch_requested && stream->command_buffer) {
    uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    iree_status_t flush_status = iree_hal_streaming_stream_flush(stream);
    if (timing_enabled) {
      timing_begin_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, flush_status);
  }

  // Check if this is a "native" kernel without IREE parameter metadata.
  // Native kernels have no bindings and no copy operations.
  const bool is_native_kernel = symbol->parameters.binding_count == 0 &&
                                symbol->parameters.copy_count == 0;
  const bool is_empty_native_kernel =
      is_native_kernel &&
      iree_hal_streaming_parameter_info_is_empty(&symbol->parameters);

  // Check if this is a pre-packed buffer (HIP_LAUNCH_PARAM_BUFFER format).
  // Pre-packed buffers are already in the kernel's native ABI format and must
  // be passed directly without unpacking or pointer rewriting.
  const bool is_pre_packed =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED) != 0;
  const bool is_args_array =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) != 0;
  if (is_args_array && is_native_kernel && !is_empty_native_kernel) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-empty args-array kernel launch requires parameter metadata");
  }
  if (!is_pre_packed && !is_args_array && !params->buffer &&
      !is_empty_native_kernel) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel launch missing parameter storage");
  }

  // Use one aligned temporary allocation for constants and binding refs. Most
  // kernels fit in a bounded inline buffer; larger reflected layouts fall back
  // to the stream host allocator and leave through one cleanup path.
  iree_host_size_t constants_capacity = 0;
  iree_host_size_t binding_capacity = 0;
  if (is_args_array && !is_empty_native_kernel) {
    constants_capacity = iree_max(symbol->parameters.direct_arg_bytes,
                                  iree_max(symbol->parameters.constant_bytes,
                                           symbol->parameters.buffer_size));
  } else if (!is_pre_packed && !is_native_kernel) {
    constants_capacity = symbol->parameters.constant_bytes;
    binding_capacity = symbol->parameters.binding_count;
  }
  iree_host_size_t constants_storage_size = 0;
  iree_host_size_t bindings_storage_size = 0;
  iree_host_size_t temporary_storage_size = 0;
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
                        constants_capacity, iree_alignof(iree_hal_buffer_ref_t),
                        &constants_storage_size) ||
                    !iree_host_size_checked_mul(binding_capacity,
                                                sizeof(iree_hal_buffer_ref_t),
                                                &bindings_storage_size) ||
                    !iree_host_size_checked_add(constants_storage_size,
                                                bindings_storage_size,
                                                &temporary_storage_size))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "kernel argument storage size overflow");
  }

  enum { IREE_HAL_STREAMING_INLINE_ARGUMENT_STORAGE_SIZE = 256 };
  iree_alignas(iree_max_align_t) uint8_t
      inline_argument_storage[IREE_HAL_STREAMING_INLINE_ARGUMENT_STORAGE_SIZE];
  uint8_t* allocated_argument_storage = NULL;
  uint8_t* argument_storage = inline_argument_storage;
  if (iree_status_is_ok(status) &&
      temporary_storage_size > sizeof(inline_argument_storage)) {
    status = iree_allocator_malloc_uninitialized(
        stream->host_allocator, temporary_storage_size,
        (void**)&allocated_argument_storage);
    if (iree_status_is_ok(status)) {
      argument_storage = allocated_argument_storage;
    }
  }

  uint64_t timing_params_start_ns =
      timing_enabled ? hrx_launch_timing_now_ns() : 0;
  iree_hal_streaming_launch_arguments_t arguments = {0};
  if (iree_status_is_ok(status)) {
    if (temporary_storage_size > 0 && !is_args_array) {
      memset(argument_storage, 0, temporary_storage_size);
    }
    status = iree_hal_streaming_prepare_launch_arguments(
        symbol, params, stream->context, is_native_kernel,
        is_empty_native_kernel, is_pre_packed, is_args_array, argument_storage,
        constants_storage_size, constants_capacity, binding_capacity,
        &arguments);
  }
  if (iree_status_is_ok(status) && timing_enabled) {
    timing_params_ns += hrx_launch_timing_now_ns() - timing_params_start_ns;
  }

  bool dispatch_directly = direct_queue_dispatch_requested;
  if (iree_status_is_ok(status) && !dispatch_directly) {
    for (iree_host_size_t i = 0; i < arguments.bindings.count; ++i) {
      const iree_hal_buffer_ref_t* binding = &arguments.bindings.values[i];
      if (!binding->buffer && binding->reserved == 0 &&
          binding->buffer_slot == 0 && binding->offset == 0 &&
          binding->length == 0) {
        dispatch_directly = true;
        break;
      }
    }
    if (dispatch_directly && stream->command_buffer) {
      uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
      status = iree_hal_streaming_stream_flush(stream);
      if (timing_enabled) {
        timing_begin_ns += hrx_launch_timing_now_ns() - timing_step_ns;
      }
    }
  }

  bool dispatch_attempted = false;
  if (iree_status_is_ok(status)) {
    dispatch_attempted = true;

    // Create IREE dispatch config.
    const iree_hal_dispatch_config_t config = {
        .workgroup_size =
            {
                params->block_dim[0],
                params->block_dim[1],
                params->block_dim[2],
            },
        .workgroup_count =
            {
                params->grid_dim[0],
                params->grid_dim[1],
                params->grid_dim[2],
            },
        .dynamic_workgroup_local_memory = params->shared_memory_bytes,
    };

    // HIP launches use native kernarg bytes. The AMDGPU queue code still
    // populates the dispatch implicit arguments for CUSTOM_DIRECT_ARGUMENTS;
    // this flag only says that the explicit argument payload is already native.
    iree_hal_dispatch_flags_t flags =
        (arguments.use_raw_arguments || is_pre_packed)
            ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS
            : IREE_HAL_DISPATCH_FLAG_NONE;

    uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    bool should_flush = false;
    iree_slim_mutex_lock(&stream->mutex);
    if (dispatch_directly) {
      uint64_t wait_value = 0;
      uint64_t signal_value = 0;
      status = iree_hal_streaming_stream_reserve_next_value_locked(
          stream, &wait_value, &signal_value);
      const iree_hal_semaphore_list_t wait_semaphores = {
          .count = wait_value > 0 ? 1 : 0,
          .semaphores = &stream->timeline_semaphore,
          .payload_values = &wait_value,
      };
      const iree_hal_semaphore_list_t signal_semaphores = {
          .count = 1,
          .semaphores = &stream->timeline_semaphore,
          .payload_values = &signal_value,
      };
      if (iree_status_is_ok(status)) {
        status = iree_hal_device_queue_dispatch(
            stream->context->device, stream->queue_affinity, wait_semaphores,
            signal_semaphores, symbol->executable,
            iree_hal_executable_function_from_index(symbol->export_ordinal),
            config,
            iree_make_const_byte_span(arguments.constants,
                                      arguments.constants_size),
            arguments.bindings, flags);
      }
      if (iree_status_is_ok(status)) {
        // The accepted dispatch owns the value it signals, so the timeline
        // advances here and stays advanced even when the flush below fails.
        stream->pending_value = signal_value;
        status = iree_hal_device_queue_flush(stream->context->device,
                                             stream->queue_affinity);
      }
    } else {
      uint64_t timing_begin_step_ns =
          timing_enabled ? hrx_launch_timing_now_ns() : 0;
      status = iree_hal_streaming_stream_begin_locked(stream);
      if (timing_enabled) {
        timing_begin_ns += hrx_launch_timing_now_ns() - timing_begin_step_ns;
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_command_buffer_dispatch(
            stream->command_buffer, symbol->executable,
            iree_hal_executable_function_from_index(symbol->export_ordinal),
            config,
            iree_make_const_byte_span(arguments.constants,
                                      arguments.constants_size),
            arguments.bindings, flags);
      }

      // Insert an execution + memory barrier after each dispatch to enforce
      // serial ordering within the command buffer, emulating HIP stream
      // semantics. This allows batching multiple dispatches per CB submission
      // while maintaining correctness. Inter-CB ordering is handled by timeline
      // semaphore chaining in iree_hal_streaming_stream_flush.
      //
      // The memory barrier with non-host (DISPATCH/TRANSFER) access scopes is
      // important: under the AMDGPU HAL backend it resolves to an AGENT-scoped
      // AQL release+acquire fence between this dispatch and the next, which
      // flushes the GPU L1/L2 caches so the next dispatch sees this dispatch's
      // writes. A bare execution barrier with no memory barriers does not
      // publish dispatch memory side effects under backends that preserve empty
      // barrier scopes, so later dispatches can observe stale device cache
      // contents.
      if (iree_status_is_ok(status) &&
          !hrx_disable_dispatch_barrier_enabled()) {
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
        uint64_t timing_barrier_step_ns =
            timing_enabled ? hrx_launch_timing_now_ns() : 0;
        status = iree_hal_command_buffer_execution_barrier(
            stream->command_buffer,
            IREE_HAL_EXECUTION_STAGE_DISPATCH |
                IREE_HAL_EXECUTION_STAGE_TRANSFER,
            IREE_HAL_EXECUTION_STAGE_DISPATCH |
                IREE_HAL_EXECUTION_STAGE_TRANSFER,
            IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, NULL);
        if (timing_enabled) {
          timing_barrier_ns +=
              hrx_launch_timing_now_ns() - timing_barrier_step_ns;
        }
      }

      if (iree_status_is_ok(status)) {
        ++stream->pending_launch_count;
        const int flush_interval = hrx_flush_interval();
        should_flush = hrx_flush_each_launch_enabled() ||
                       (flush_interval > 0 && stream->pending_launch_count >=
                                                  (uint32_t)flush_interval);
      }
    }
    iree_slim_mutex_unlock(&stream->mutex);
    if (timing_enabled) {
      timing_dispatch_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
    if (!dispatch_directly && iree_status_is_ok(status) && should_flush) {
      status = iree_hal_streaming_stream_flush(stream);
    }
  }
  if (timing_enabled && dispatch_attempted) {
    ++g_hrx_launch_timing.launch_count;
    g_hrx_launch_timing.launch_total_ns +=
        hrx_launch_timing_now_ns() - timing_start_ns;
    g_hrx_launch_timing.launch_begin_ns += timing_begin_ns;
    g_hrx_launch_timing.launch_params_ns += timing_params_ns;
    g_hrx_launch_timing.launch_dispatch_ns += timing_dispatch_ns;
    g_hrx_launch_timing.launch_barrier_ns += timing_barrier_ns;
  }
  iree_allocator_free(stream->host_allocator, allocated_argument_storage);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Host callback wrapper structure to adapt HIP callbacks to HAL callbacks.
typedef struct iree_hal_streaming_host_callback_t {
  void (*fn)(void* user_data);
  void* user_data;
} iree_hal_streaming_host_callback_t;

// HAL host call function that invokes the HIP-style callback.
static iree_status_t iree_hal_streaming_host_callback_thunk(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  iree_hal_streaming_host_callback_t* callback =
      (iree_hal_streaming_host_callback_t*)user_data;
  callback->fn(callback->user_data);
  iree_allocator_free(iree_allocator_system(), callback);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_queue_host_call(
    iree_hal_streaming_stream_t* stream, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(call.fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);
  uint64_t wait_value = 0;
  uint64_t signal_value = 0;
  iree_status_t status = iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &wait_value, &signal_value);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = wait_value > 0 ? 1 : 0,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &signal_value,
  };

  status = iree_hal_device_queue_host_call(
      stream->context->device, stream->queue_affinity, wait_semaphores,
      signal_semaphores, call, args, flags);
  if (iree_status_is_ok(status)) {
    stream->pending_value = signal_value;
  }
  iree_slim_mutex_unlock(&stream->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_launch_host_function(
    iree_hal_streaming_stream_t* stream, void (*fn)(void*), void* user_data) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Add host call node to the graph instead of executing immediately.
    iree_hal_streaming_graph_node_t* node = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_host_call_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, fn, user_data, &node));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_capture_set_last_node(stream, node));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Allocate a wrapper structure to hold the callback and user data.
  iree_hal_streaming_host_callback_t* callback = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(iree_allocator_system(), sizeof(*callback),
                                (void**)&callback));
  callback->fn = fn;
  callback->user_data = user_data;

  uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call =
      iree_hal_make_host_call(iree_hal_streaming_host_callback_thunk, callback);

  iree_status_t status = iree_hal_streaming_queue_host_call(
      stream, call, args, IREE_HAL_HOST_CALL_FLAG_NONE);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), callback);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
