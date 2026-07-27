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
  stream->synchronization_policy =
      IREE_HAL_STREAMING_SYNCHRONIZATION_POLICY_AUTO;
  stream->priority = priority;
  stream->cu_mask_count = 0;
  stream->cu_mask = NULL;
  stream->stream_id = 0;
  stream->command_buffer = NULL;
  stream->pending_launch_count = 0;
  stream->timeline_semaphore = NULL;
  stream->pending_value = 0;
  stream->submitted_value = 0;
  stream->completed_value = 0;
  stream->queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  stream->recorded_events = NULL;
  stream->event_count = 0;
  stream->event_capacity = 0;
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

  iree_hal_streaming_context_t* context = stream->context;
  if (context) {
    iree_status_ignore(iree_hal_streaming_stream_synchronize(stream));
    if (stream->capture_status != IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      iree_hal_streaming_stream_set_capture_status(
          stream, IREE_HAL_STREAMING_CAPTURE_STATUS_NONE);
    }
    stream->context = NULL;
    iree_hal_streaming_context_unregister_stream(context, stream);
  }

  // Clean up recorded events.
  if (stream->recorded_events) {
    for (iree_host_size_t i = 0; i < stream->event_count; ++i) {
      iree_hal_streaming_event_release(stream->recorded_events[i]);
    }
    iree_allocator_free(stream->host_allocator, stream->recorded_events);
  }
  iree_allocator_free(stream->host_allocator,
                      stream->memory_reuse_dependencies);

  // Release command buffer.
  iree_hal_command_buffer_release(stream->command_buffer);

  iree_allocator_free(stream->host_allocator, stream->cu_mask);

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

    // Wait for the previous submission (pending_value before increment).
    // This chains each flush after the one before it, so that operations
    // split across multiple command buffers (e.g. by an intervening
    // hipMemcpy) still execute in order.
    uint64_t wait_value = stream->pending_value;
    stream->pending_value++;

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
        .payload_values = &stream->pending_value,
    };

    timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    status = iree_hal_device_queue_execute(
        stream->context->device, queue_affinity, wait_semaphores,
        signal_semaphores, stream->command_buffer,
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_device_queue_flush(stream->context->device, queue_affinity);
    }
    if (timing_enabled) {
      timing_execute_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }

    if (!iree_status_is_ok(status)) {
      // Error will propagate via iree_status_t return.
    }

    // Track the submitted value for wait_submitted.
    if (iree_status_is_ok(status)) {
      stream->submitted_value = stream->pending_value;
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
  iree_status_t query_status =
      iree_hal_semaphore_query(stream->timeline_semaphore, &current_value);
  if (iree_status_is_unavailable(query_status)) {
    iree_status_ignore(query_status);
    *status = 1;  // Not complete
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(query_status);

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
  if (iree_status_is_unavailable(query_status)) {
    iree_status_ignore(query_status);
    query_status = iree_ok_status();
  } else if (iree_status_is_ok(query_status)) {
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
  // Snapshot the last submitted value under the stream lock. New submissions
  // after this point are outside this wait operation.
  iree_slim_mutex_lock(&stream->mutex);
  const uint64_t submitted_value = stream->submitted_value;
  iree_slim_mutex_unlock(&stream->mutex);
  if (submitted_value > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_wait(stream->timeline_semaphore, submitted_value,
                                    iree_infinite_timeout(),
                                    IREE_ASYNC_WAIT_FLAG_NONE));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_stream_wait_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (event->capture_graph) {
    bool adopt_capture_graph = false;
    iree_slim_mutex_lock(&stream->mutex);
    if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
      adopt_capture_graph = true;
    } else if (stream->capture_graph != event->capture_graph) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "event wait crosses different active capture graphs");
    }
    iree_slim_mutex_unlock(&stream->mutex);

    if (adopt_capture_graph) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_stream_flush(stream));

      unsigned long long capture_id = 0;
      if (!event->recording_stream) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_streaming_context_allocate_capture_id(stream->context,
                                                               &capture_id));
      }

      iree_slim_mutex_lock(&stream->mutex);
      if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
        stream->capture_graph = event->capture_graph;
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
      } else if (stream->capture_graph != event->capture_graph) {
        iree_slim_mutex_unlock(&stream->mutex);
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "event wait crosses different active capture graphs");
      }
      iree_slim_mutex_unlock(&stream->mutex);
    }

    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_update_capture_dependencies(
                stream, event->capture_dependencies,
                event->capture_dependency_count,
                IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  const unsigned long long source_stream_id =
      event->recording_stream ? event->recording_stream->stream_id : 0;
  const uint64_t source_timeline_value = event->signal_value;
  bool added_memory_reuse_dependency = false;
  if (source_stream_id != 0 && source_stream_id != stream->stream_id &&
      source_timeline_value != 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_stream_reserve_memory_reuse_dependency(
                stream, source_stream_id, &added_memory_reuse_dependency));
  }

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);

  // Reserve the next stream timeline value and submit a barrier that completes
  // only after the event is signaled. The value is submitted, not completed;
  // query/synchronize advance completed_value after observing the semaphore.
  uint64_t wait_value = stream->pending_value;
  uint64_t signal_value = wait_value + 1;

  // The barrier waits on the point the event was recorded at and on everything
  // already on this stream. Both waits are needed: dropping the stream wait
  // would let this barrier signal the next stream value before the value below
  // it, which both breaks the ordering every later operation on the stream
  // relies on and signals the stream timeline out of order. The stream wait is
  // dropped when the stream has never submitted, and the event wait when the
  // event has never had a record submitted, as neither has a point to wait for.
  // The recorded point is copied out under the event's own mutex; reading the
  // event's fields in place would read them under this stream's mutex, which is
  // not the mutex a record of this event takes.
  iree_hal_semaphore_t* event_semaphore = NULL;
  uint64_t event_signal_value = 0;
  iree_hal_streaming_event_acquire_recorded_point(event, &event_semaphore,
                                                  &event_signal_value);
  iree_hal_semaphore_t* wait_semaphore_storage[2];
  uint64_t wait_value_storage[2];
  iree_host_size_t wait_count = 0;
  if (wait_value > 0) {
    wait_semaphore_storage[wait_count] = stream->timeline_semaphore;
    wait_value_storage[wait_count] = wait_value;
    ++wait_count;
  }
  if (event_semaphore) {
    wait_semaphore_storage[wait_count] = event_semaphore;
    wait_value_storage[wait_count] = event_signal_value;
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

  iree_status_t status = iree_hal_device_queue_barrier(
      stream->context->device, stream->queue_affinity, wait_semaphores,
      signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(stream->context->device,
                                         stream->queue_affinity);
  }
  if (iree_status_is_ok(status)) {
    stream->pending_value = signal_value;
    stream->submitted_value = signal_value;
  }

  iree_slim_mutex_unlock(&stream->mutex);
  iree_hal_semaphore_release(event_semaphore);
  if (!iree_status_is_ok(status) && added_memory_reuse_dependency) {
    iree_hal_streaming_stream_remove_uncommitted_memory_reuse_dependency(
        stream, source_stream_id);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  if (source_stream_id != 0 && source_stream_id != stream->stream_id &&
      source_timeline_value != 0) {
    iree_hal_streaming_stream_record_memory_reuse_dependency(
        stream, source_stream_id, source_timeline_value);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

static bool iree_hal_streaming_buffer_can_import_for_context(
    const iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) return false;
  if (buffer->is_managed) return true;
  return iree_all_bits_set(
      (iree_hal_memory_type_t)buffer->memory_type,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
}

static iree_status_t iree_hal_streaming_device_buffer_for_context(
    iree_hal_streaming_context_t* context, iree_hal_streaming_buffer_t* buffer,
    iree_hal_buffer_t** out_buffer,
    iree_hal_streaming_deviceptr_t* out_device_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (out_device_ptr) *out_device_ptr = 0;

  if (buffer->context == context) {
    *out_buffer = buffer->buffer;
    if (out_device_ptr) *out_device_ptr = buffer->device_ptr;
    return iree_ok_status();
  }
  if (!iree_hal_streaming_buffer_can_import_for_context(buffer)) {
    return iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }
  if (buffer->is_managed &&
      (!buffer->host_ptr ||
       (iree_hal_streaming_deviceptr_t)(uintptr_t)buffer->host_ptr !=
           buffer->device_ptr)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "cross-device managed memory requires one stable host/device address");
  }
  if (!buffer->buffer || buffer->device_ptr == 0 || buffer->size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation is missing device import metadata");
  }
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&buffer->context_import_mutex);
  for (iree_hal_streaming_context_import_t* import = buffer->context_imports;
       import; import = import->next) {
    if (import->context == context) {
      *out_buffer = import->buffer;
      if (out_device_ptr) *out_device_ptr = buffer->device_ptr;
      iree_slim_mutex_unlock(&buffer->context_import_mutex);
      return iree_ok_status();
    }
  }

  iree_hal_buffer_t* imported_buffer = NULL;
  const bool import_host_allocation = iree_all_bits_set(
      (iree_hal_memory_type_t)buffer->memory_type,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
  iree_hal_buffer_params_t params = {
      .usage = iree_hal_buffer_allowed_usage(buffer->buffer),
      .access = iree_hal_buffer_allowed_access(buffer->buffer),
      .type = (iree_hal_memory_type_t)buffer->memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
  iree_hal_external_buffer_t external_buffer = {
      .type = import_host_allocation
                  ? IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION
                  : IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
      .size = buffer->size,
  };
  if (import_host_allocation) {
    external_buffer.handle.host_allocation.ptr = buffer->host_ptr;
  } else {
    external_buffer.handle.device_allocation.ptr = buffer->device_ptr;
  }
  status = iree_hal_allocator_import_buffer(
      context->device_allocator, params, &external_buffer,
      iree_hal_buffer_release_callback_null(), &imported_buffer);

  iree_hal_streaming_context_import_t* import = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(buffer->context->host_allocator,
                                   sizeof(*import), (void**)&import);
  }
  if (iree_status_is_ok(status)) {
    import->next = buffer->context_imports;
    import->context = context;
    iree_hal_streaming_context_retain(context);
    import->buffer = imported_buffer;
    buffer->context_imports = import;
    imported_buffer = NULL;
    *out_buffer = import->buffer;
    if (out_device_ptr) *out_device_ptr = buffer->device_ptr;
  }
  iree_slim_mutex_unlock(&buffer->context_import_mutex);
  iree_hal_buffer_release(imported_buffer);
  return status;
}

static iree_status_t iree_hal_streaming_lookup_kernel_buffer_ref(
    iree_hal_streaming_context_t* context, void* device_ptr,
    iree_hal_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(device_ptr);
  IREE_ASSERT_ARGUMENT(out_ref);
  *out_ref = (iree_hal_buffer_ref_t){0};

  // This resolves explicit pointer arguments described by kernel metadata into
  // HAL buffer bindings. It is not a lifetime analysis: HIP device pointers can
  // be hidden in opaque kernarg bytes or device memory, so free/unregister
  // paths must conservatively order destruction without relying on lookup
  // coverage.
  iree_hal_streaming_buffer_ref_t stream_ref;
  iree_hal_streaming_context_t* owner_context = NULL;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)(uintptr_t)device_ptr,
      &stream_ref);
  if (iree_status_is_ok(status) ||
      iree_status_code(status) != IREE_STATUS_NOT_FOUND) {
    if (!iree_status_is_ok(status)) return status;
  } else {
    iree_status_ignore(status);
    if (!iree_hal_streaming_context_has_peer_contexts(context)) {
      return iree_status_from_code(IREE_STATUS_NOT_FOUND);
    }
    status = iree_hal_streaming_memory_lookup_range_across_contexts(
        (iree_hal_streaming_deviceptr_t)(uintptr_t)device_ptr, 1,
        &owner_context, &stream_ref);
    if (!iree_status_is_ok(status)) return status;

    if (!iree_hal_streaming_buffer_can_import_for_context(stream_ref.buffer)) {
      iree_hal_streaming_context_release(owner_context);
      return iree_status_from_code(IREE_STATUS_NOT_FOUND);
    }
  }

  iree_hal_buffer_t* device_buffer = NULL;
  status = iree_hal_streaming_device_buffer_for_context(
      context, stream_ref.buffer, &device_buffer, NULL);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_release(owner_context);
    return status;
  }
  const iree_device_size_t length =
      stream_ref.offset < stream_ref.buffer->size
          ? stream_ref.buffer->size - stream_ref.offset
          : 0;
  *out_ref = iree_hal_make_buffer_ref(device_buffer, stream_ref.offset, length);
  iree_hal_streaming_context_release(owner_context);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_unpack_parameters(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_parameter_info_t* parameters,
    const void* parameter_buffer_ptr, void* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(parameters);
  if (iree_hal_streaming_parameter_info_is_empty(parameters)) {
    return iree_ok_status();
  }
  const bool requires_parameter_storage = parameters->buffer_size > 0 ||
                                          parameters->binding_count > 0 ||
                                          parameters->copy_count > 0;
  if (!requires_parameter_storage) {
    return iree_ok_status();
  }
  if (!parameter_buffer_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel parameter buffer is required");
  }
  if (parameters->copy_count > 0 && !out_constants) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel constant storage is required");
  }
  IREE_ASSERT_ARGUMENT(out_bindings);
  if (!out_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel binding list is required");
  }
  if (parameters->binding_count > 0 && !out_bindings->values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel binding storage is required");
  }

  const uint8_t* parameter_buffer = (const uint8_t*)parameter_buffer_ptr;

  // Copy constant data spans into the HAL constants table. Native ABI packing
  // uses a separate destination offset so dense constants do not imply ABI
  // layout.
  uint8_t* constants = (uint8_t*)out_constants;
  const iree_hal_streaming_parameter_op_t* op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    if (copy_op.size > 0) {
      memcpy(constants + copy_op.constant_destination_offset,
             parameter_buffer + copy_op.source_offset, copy_op.size);
    }
  }

  // Resolve bindings, if any.
  // A NULL HIP kernel pointer is a valid literal kernarg for optional buffers.
  // Represent it as a zeroed direct binding; the AMDGPU direct queue path
  // materializes that as a zero pointer in the final kernarg block.
  iree_hal_buffer_ref_t* bindings =
      (iree_hal_buffer_ref_t*)out_bindings->values;
  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    void* device_ptr = *(void**)(parameter_buffer + resolve_op.source_offset);
    // Kernel metadata identifies pointer slots but not the dynamic object
    // extent. Resolve with an unknown length; the HAL buffer reference owns
    // the allocation for dispatch.

    if (!device_ptr) {
      bindings[resolve_op.destination_ordinal] = (iree_hal_buffer_ref_t){0};
      continue;
    }

    iree_status_t lookup_status = iree_hal_streaming_lookup_kernel_buffer_ref(
        context, device_ptr, &bindings[resolve_op.destination_ordinal]);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_unpack_parameter_list(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_parameter_info_t* parameters,
    void** parameter_list, void* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(parameters);
  if (iree_hal_streaming_parameter_info_is_empty(parameters)) {
    return iree_ok_status();
  }
  const bool requires_parameter_storage = parameters->buffer_size > 0 ||
                                          parameters->binding_count > 0 ||
                                          parameters->copy_count > 0;
  if (!requires_parameter_storage) {
    return iree_ok_status();
  }
  if (!parameter_list) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel parameter list is required");
  }
  if (parameters->copy_count > 0 && !out_constants) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel constant storage is required");
  }
  IREE_ASSERT_ARGUMENT(out_bindings);
  if (!out_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel binding list is required");
  }
  if (parameters->binding_count > 0 && !out_bindings->values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel binding storage is required");
  }

  // When parameters are provided as an array of pointers, each element in the
  // array points to the actual parameter value. Metadata-described pointer
  // slots can be resolved to HAL bindings; arbitrary bytes stay raw.

  // Copy constant data spans into the HAL constants table. For pointer-array
  // launches, the source ordinal selects the element that points at the value.
  uint8_t* constants = (uint8_t*)out_constants;
  const iree_hal_streaming_parameter_op_t* op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    // In pointer array mode, source_ordinal is an index into the parameter_list
    // array. Each parameter_list[index] is a pointer to the actual value.
    // We need to dereference it to get the value.
    void* param_ptr = parameter_list[copy_op.source_ordinal];
    if (!param_ptr && copy_op.size > 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel argument %" PRIu32 " is NULL",
                              (uint32_t)copy_op.source_ordinal);
    }
    if (copy_op.size > 0) {
      memcpy(constants + copy_op.constant_destination_offset, param_ptr,
             copy_op.size);
    }
  }

  // Resolve bindings, if any.
  // For bindings, each parameter in the list is a pointer to a device pointer.
  // A NULL HIP kernel pointer is a valid literal kernarg for optional buffers.
  // Represent it as a zeroed direct binding; the AMDGPU direct queue path
  // materializes that as a zero pointer in the final kernarg block.
  iree_hal_buffer_ref_t* bindings =
      (iree_hal_buffer_ref_t*)out_bindings->values;
  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    // In pointer array mode, source_ordinal is an index into the
    // parameter_list.
    void* param_ptr = parameter_list[resolve_op.source_ordinal];
    if (!param_ptr) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel argument %" PRIu32 " is NULL",
                              (uint32_t)resolve_op.source_ordinal);
    }
    // The parameter points to a device pointer (void*)
    void* device_ptr = *(void**)param_ptr;
    // Kernel metadata identifies pointer slots but not the dynamic object
    // extent. Resolve with an unknown length; the HAL buffer reference owns
    // the allocation for dispatch.

    if (!device_ptr) {
      bindings[resolve_op.destination_ordinal] = (iree_hal_buffer_ref_t){0};
      continue;
    }

    iree_status_t lookup_status = iree_hal_streaming_lookup_kernel_buffer_ref(
        context, device_ptr, &bindings[resolve_op.destination_ordinal]);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_pack_raw_argument_list(
    const iree_hal_streaming_parameter_info_t* parameters,
    void** parameter_list, void* out_constants,
    iree_host_size_t* out_constants_size) {
  IREE_ASSERT_ARGUMENT(parameters);
  IREE_ASSERT_ARGUMENT(out_constants_size);

  if (iree_hal_streaming_parameter_info_is_empty(parameters)) {
    *out_constants_size = 0;
    return iree_ok_status();
  }

  *out_constants_size = parameters->direct_arg_bytes
                            ? parameters->direct_arg_bytes
                            : parameters->constant_bytes;
  if (*out_constants_size == 0) {
    *out_constants_size = parameters->buffer_size;
  }
  if (*out_constants_size == 0) return iree_ok_status();
  if (!out_constants || (!parameter_list && (parameters->buffer_size > 0 ||
                                             parameters->binding_count > 0 ||
                                             parameters->copy_count > 0))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "raw kernel arguments require parameter storage");
  }

  uint8_t* constants = (uint8_t*)out_constants;
  const iree_hal_streaming_parameter_op_t* op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    void* param_ptr = parameter_list[copy_op.source_ordinal];
    if (!param_ptr) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel argument %" PRIu32 " is NULL",
                              (uint32_t)copy_op.source_ordinal);
    }
    if ((iree_host_size_t)copy_op.native_abi_destination_offset >
            *out_constants_size ||
        (iree_host_size_t)copy_op.size >
            *out_constants_size - copy_op.native_abi_destination_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "kernel argument copy exceeds kernarg size");
    }
    memcpy(constants + copy_op.native_abi_destination_offset, param_ptr,
           copy_op.size);
  }

  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    void* param_ptr = parameter_list[resolve_op.source_ordinal];
    if (!param_ptr) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kernel argument %" PRIu32 " is NULL",
                              (uint32_t)resolve_op.source_ordinal);
    }
    void* device_ptr = *(void**)param_ptr;
    if ((iree_host_size_t)resolve_op.native_abi_destination_offset >
            *out_constants_size ||
        sizeof(device_ptr) >
            *out_constants_size - resolve_op.native_abi_destination_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "kernel pointer argument exceeds kernarg size");
    }
    memcpy(constants + resolve_op.native_abi_destination_offset, &device_ptr,
           sizeof(void*));
  }

  return iree_ok_status();
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

  // Stack allocate arrays based on cached sizes.
  // Zero-initialize constants so ABI padding is deterministic.
  void* constants = symbol->parameters.constant_bytes
                        ? iree_alloca(symbol->parameters.constant_bytes)
                        : NULL;
  if (constants) memset(constants, 0, symbol->parameters.constant_bytes);
  iree_hal_buffer_ref_list_t binding_list = {
      .count = symbol->parameters.binding_count,
      .values = symbol->parameters.binding_count
                    ? iree_alloca(symbol->parameters.binding_count *
                                  sizeof(iree_hal_buffer_ref_t))
                    : NULL,
  };

  // Check if this is a "native" kernel without IREE parameter metadata.
  // Native kernels have no bindings and no copy operations.
  bool is_native_kernel = (symbol->parameters.binding_count == 0 &&
                           symbol->parameters.copy_count == 0);
  const bool is_empty_native_kernel =
      is_native_kernel &&
      iree_hal_streaming_parameter_info_is_empty(&symbol->parameters);

  size_t constants_size = symbol->parameters.constant_bytes;
  // Track if we need to use raw argument passing (e.g., for external pointers).
  bool use_raw_arguments = false;

  // Check if this is a pre-packed buffer (HIP_LAUNCH_PARAM_BUFFER format).
  // Pre-packed buffers are already in the kernel's native ABI format and must
  // be passed directly without unpacking or pointer rewriting.
  bool is_pre_packed =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED) != 0;

  uint64_t timing_params_start_ns =
      timing_enabled ? hrx_launch_timing_now_ns() : 0;
  if (is_pre_packed) {
    // Pre-packed buffers are already in native kernarg layout. They may
    // contain device pointers either as formal pointer arguments or inside
    // opaque data, so preserve the bytes exactly and rely on HIP lifetime
    // ordering instead of trying to translate visible pointer slots.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_streaming_validate_prepacked_kernel_arguments(symbol, params));
    constants = params->buffer;
    constants_size = params->buffer_size;
    binding_list.count = 0;  // No IREE bindings, using raw pointers.
    use_raw_arguments = true;
  } else if (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) {
    if (is_native_kernel && params->buffer && !is_empty_native_kernel) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "non-empty args-array kernel launch requires parameter metadata");
    }
    // Pointer-array launches are converted to native kernarg bytes. This keeps
    // formal pointer arguments and pointers nested inside copied structs on the
    // same device-pointer contract.
    constants_size = symbol->parameters.direct_arg_bytes
                         ? symbol->parameters.direct_arg_bytes
                         : symbol->parameters.constant_bytes;
    if (constants_size == 0) {
      constants_size = symbol->parameters.buffer_size;
    }
    constants = constants_size ? iree_alloca(constants_size) : NULL;
    if (constants) memset(constants, 0, constants_size);
    iree_status_t pack_status = iree_hal_streaming_pack_raw_argument_list(
        &symbol->parameters, (void**)params->buffer, constants,
        &constants_size);
    if (!iree_status_is_ok(pack_status)) {
      IREE_TRACE_ZONE_END(z0);
      return pack_status;
    }
    binding_list.count = 0;
    use_raw_arguments = true;
  } else if (is_native_kernel && params->buffer) {
    // Native kernel with pre-packed buffer: pass raw arguments directly. This
    // path is only valid when the caller did not declare the buffer as a HIP
    // args array, because a void** parameter list is not a native kernarg pack.
    constants = params->buffer;
    if (params->buffer_size > 0) {
      constants_size = params->buffer_size;
    }
    binding_list.count = 0;
    use_raw_arguments = true;
  } else if (params->buffer) {
    // Unflagged launches use the shared metadata path. HIP entry points mark
    // native kernarg bytes with PRE_PACKED or provide ARGS_ARRAY so device
    // pointer values are preserved without depending on reflected pointer
    // slots.
    iree_status_t unpack_status = iree_hal_streaming_unpack_parameters(
        stream->context, &symbol->parameters, params->buffer, constants,
        &binding_list);
    if (!iree_status_is_ok(unpack_status)) {
      // If unpack fails due to NULL or external device pointers, fall back
      // to raw argument passing. This handles native kernels with optional
      // parameters or external allocations.
      if (iree_status_code(unpack_status) == IREE_STATUS_NOT_FOUND) {
        iree_status_ignore(unpack_status);
        constants = params->buffer;
        constants_size = params->buffer_size;
        binding_list.count = 0;
        use_raw_arguments = true;
      } else {
        IREE_TRACE_ZONE_END(z0);
        return unpack_status;
      }
    }
  } else if (is_empty_native_kernel) {
    constants = NULL;
    constants_size = 0;
    binding_list.count = 0;
    use_raw_arguments = true;
  } else {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel launch missing parameter storage");
  }
  if (timing_enabled) {
    timing_params_ns += hrx_launch_timing_now_ns() - timing_params_start_ns;
  }

  bool dispatch_directly = direct_queue_dispatch_requested;
  if (!dispatch_directly) {
    for (iree_host_size_t i = 0; i < binding_list.count; ++i) {
      const iree_hal_buffer_ref_t* binding = &binding_list.values[i];
      if (!binding->buffer && binding->reserved == 0 &&
          binding->buffer_slot == 0 && binding->offset == 0 &&
          binding->length == 0) {
        dispatch_directly = true;
        break;
      }
    }
    if (dispatch_directly && stream->command_buffer) {
      uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
      iree_status_t flush_status = iree_hal_streaming_stream_flush(stream);
      if (timing_enabled) {
        timing_begin_ns += hrx_launch_timing_now_ns() - timing_step_ns;
      }
      IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, flush_status);
    }
  }

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
  // populates the dispatch implicit arguments for CUSTOM_DIRECT_ARGUMENTS; this
  // flag only says that the explicit argument payload is already native.
  iree_hal_dispatch_flags_t flags =
      (use_raw_arguments || is_pre_packed)
          ? IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS
          : IREE_HAL_DISPATCH_FLAG_NONE;

  uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
  iree_status_t status = iree_ok_status();
  bool should_flush = false;
  iree_slim_mutex_lock(&stream->mutex);
  if (dispatch_directly) {
    uint64_t wait_value = stream->pending_value;
    uint64_t signal_value = wait_value + 1;
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
    status = iree_hal_device_queue_dispatch(
        stream->context->device, stream->queue_affinity, wait_semaphores,
        signal_semaphores, symbol->executable,
        iree_hal_executable_function_from_index(symbol->export_ordinal), config,
        iree_make_const_byte_span(constants, constants_size), binding_list,
        flags);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_flush(stream->context->device,
                                           stream->queue_affinity);
    }
    if (iree_status_is_ok(status)) {
      stream->pending_value = signal_value;
      stream->submitted_value = signal_value;
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
          config, iree_make_const_byte_span(constants, constants_size),
          binding_list, flags);
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
    // writes. A bare execution barrier with no memory barriers does not publish
    // dispatch memory side effects under backends that preserve empty barrier
    // scopes, so later dispatches can observe stale device cache contents.
    if (iree_status_is_ok(status) && !hrx_disable_dispatch_barrier_enabled()) {
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
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
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
                     (flush_interval > 0 &&
                      stream->pending_launch_count >= (uint32_t)flush_interval);
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
  if (timing_enabled) {
    timing_dispatch_ns += hrx_launch_timing_now_ns() - timing_step_ns;
  }
  if (!dispatch_directly && iree_status_is_ok(status) && should_flush) {
    status = iree_hal_streaming_stream_flush(stream);
  }
  if (timing_enabled) {
    ++g_hrx_launch_timing.launch_count;
    g_hrx_launch_timing.launch_total_ns +=
        hrx_launch_timing_now_ns() - timing_start_ns;
    g_hrx_launch_timing.launch_begin_ns += timing_begin_ns;
    g_hrx_launch_timing.launch_params_ns += timing_params_ns;
    g_hrx_launch_timing.launch_dispatch_ns += timing_dispatch_ns;
    g_hrx_launch_timing.launch_barrier_ns += timing_barrier_ns;
  }
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
  uint64_t wait_value = stream->pending_value;
  if (IREE_UNLIKELY(wait_value == UINT64_MAX)) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "stream timeline value overflow");
  }
  uint64_t signal_value = wait_value + 1;
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

  iree_status_t status = iree_hal_device_queue_host_call(
      stream->context->device, stream->queue_affinity, wait_semaphores,
      signal_semaphores, call, args, flags);
  if (iree_status_is_ok(status)) {
    stream->pending_value = signal_value;
    stream->submitted_value = signal_value;
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
