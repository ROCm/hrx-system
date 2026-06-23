// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/internal.h"

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
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
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
  stream->submitted_value = 0;
  stream->completed_value = 0;
  stream->queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  stream->recorded_events = NULL;
  stream->event_count = 0;
  stream->event_capacity = 0;

  stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  stream->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
  stream->capture_graph = NULL;
  stream->capture_graph_owned = false;
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

iree_status_t iree_hal_streaming_stream_begin_locked(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);

  // Create command buffer if not already created.
  // Note that we set UNRETAINED as we ensure the resources we have to track are
  // retained at the graph exec level and CUDA/HIP don't make any statements
  // about resource lifetime.
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

  uint64_t current_value = 0;
  iree_status_t query_status =
      iree_hal_semaphore_query(stream->timeline_semaphore, &current_value);
  if (iree_status_is_unavailable(query_status)) {
    iree_status_ignore(query_status);
    *status = 1;  // Not complete
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(query_status);

  if (current_value >= stream->pending_value) {
    *status = 0;  // Complete
    stream->completed_value = current_value;
  } else {
    *status = 1;  // Not complete
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_stream_synchronize(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  const int timing_enabled = hrx_launch_timing_enabled();
  const uint64_t timing_start_ns =
      timing_enabled ? hrx_launch_timing_now_ns() : 0;
  uint64_t timing_flush_ns = 0;
  uint64_t timing_query_ns = 0;
  uint64_t timing_wait_ns = 0;

  int status = 0;
  uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
  iree_status_t flush_status = iree_hal_streaming_stream_flush(stream);
  if (timing_enabled) {
    timing_flush_ns += hrx_launch_timing_now_ns() - timing_step_ns;
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, flush_status);

  timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
  iree_status_t query_status = iree_hal_streaming_stream_query(stream, &status);
  if (timing_enabled) {
    timing_query_ns += hrx_launch_timing_now_ns() - timing_step_ns;
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, query_status);
  (void)status;

  // Wait for timeline semaphore to reach pending value.
  if (stream->pending_value > stream->completed_value) {
    // fprintf(stderr, "[STREAM] sync: waiting for semaphore pending=%"PRIu64"
    // completed=%"PRIu64"\n",
    //         stream->pending_value, stream->completed_value);
    timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    iree_status_t wait_status = iree_hal_semaphore_wait(
        stream->timeline_semaphore, stream->pending_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    if (timing_enabled) {
      timing_wait_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
    if (!iree_status_is_ok(wait_status)) {
      IREE_TRACE_ZONE_END(z0);
      return wait_status;
    }
    // fprintf(stderr, "[STREAM] sync: wait OK\n");
    stream->completed_value = stream->pending_value;
  }

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

iree_status_t iree_hal_streaming_stream_wait_submitted(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Wait for already-submitted work to complete WITHOUT flushing.
  // This is safe to call from other threads as it doesn't modify stream state.
  // We wait for submitted_value (the last value that was actually submitted).
  if (stream->submitted_value > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_wait(
                stream->timeline_semaphore, stream->submitted_value,
                iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
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

      iree_slim_mutex_lock(&stream->mutex);
      if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_NONE) {
        stream->capture_graph = event->capture_graph;
        stream->capture_graph_owned = true;
        if (event->recording_stream) {
          stream->capture_mode = event->recording_stream->capture_mode;
          stream->capture_id = event->recording_stream->capture_id;
          stream->capture_owner_thread_id =
              event->recording_stream->capture_owner_thread_id;
        } else {
          stream->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
          stream->capture_id = stream->capture_id + 1;
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

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  // Get the current stream pending value to signal after waiting for the event.
  uint64_t signal_value = stream->pending_value + 1;
  stream->pending_value = signal_value;

  // Create a queue barrier that waits for the event and signals the stream.
  // This ensures the stream continues only after the event is signaled.
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = 1,
      .semaphores = &event->semaphore,
      .payload_values = &event->signal_value,
  };
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &signal_value,
  };

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_barrier(
              stream->context->device, stream->queue_affinity, wait_semaphores,
              signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE));

  // Update completed value to track this barrier.
  stream->completed_value = signal_value;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_unpack_parameters(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_parameter_info_t* parameters,
    const void* parameter_buffer_ptr, void* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(parameters);
  if (parameters->buffer_size == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(parameter_buffer_ptr);
  IREE_ASSERT_ARGUMENT(out_bindings);

  const uint8_t* parameter_buffer = (const uint8_t*)parameter_buffer_ptr;

  // Copy constant data spans.
  // Each copy represents one or more constants laid out contiguously and
  // copied in order. Constants are placed at their ABI offsets (dst_offset)
  // matching how they'll appear in the kernel argument buffer.
  uint8_t* constants = (uint8_t*)out_constants;
  const iree_hal_streaming_parameter_op_t* op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    memcpy(constants + copy_op.dst_offset,
           parameter_buffer + copy_op.src_offset, copy_op.size);
  }

  // Resolve bindings, if any.
  // A NULL HIP kernel pointer is a valid literal kernarg for optional buffers.
  // Represent it as a zeroed direct binding; the AMDGPU direct queue path
  // materializes that as a zero pointer in the final kernarg block.
  iree_hal_buffer_ref_t* bindings =
      (iree_hal_buffer_ref_t*)out_bindings->values;
  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    void* device_ptr = *(void**)(parameter_buffer + resolve_op.src_offset);
    // TODO(benvanik): possibly calculate proper range here? We could easily
    // (at only the cost of a cache miss) get the total buffer size and then
    // subtract the offset to get the remaining size.

    if (!device_ptr) {
      bindings[resolve_op.dst_ordinal] = (iree_hal_buffer_ref_t){0};
      continue;
    }

    iree_hal_streaming_buffer_ref_t stream_ref;
    iree_status_t lookup_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)device_ptr, &stream_ref);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
    bindings[resolve_op.dst_ordinal] =
        iree_hal_streaming_convert_buffer_ref(stream_ref);
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
  if (parameters->buffer_size == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(parameter_list);
  IREE_ASSERT_ARGUMENT(out_bindings);

  // When parameters are provided as an array of pointers, each element in the
  // array points to the actual parameter value. We need to dereference each
  // pointer and handle buffer translation.

  // Copy constant data spans.
  // For each copy operation, we read from the parameter list at the source
  // ordinal and copy to the ABI offset in the constants buffer.
  uint8_t* constants = (uint8_t*)out_constants;
  const iree_hal_streaming_parameter_op_t* op = &parameters->ops[0];
  for (uint32_t i = 0; i < parameters->copy_count; ++i, ++op) {
    const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
    // In pointer array mode, src_ordinal is an index into the parameter_list
    // array. Each parameter_list[index] is a pointer to the actual value.
    // We need to dereference it to get the value.
    void* param_ptr = parameter_list[copy_op.src_ordinal];
    memcpy(constants + copy_op.dst_offset, param_ptr, copy_op.size);
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
    // In pointer array mode, src_offset is an index into the parameter_list.
    void* param_ptr = parameter_list[resolve_op.src_ordinal];
    // The parameter points to a device pointer (void*)
    void* device_ptr = *(void**)param_ptr;
    // TODO(benvanik): possibly calculate proper range here? We could easily
    // (at only the cost of a cache miss) get the total buffer size and then
    // subtract the offset to get the remaining size.

    if (!device_ptr) {
      bindings[resolve_op.dst_ordinal] = (iree_hal_buffer_ref_t){0};
      continue;
    }

    iree_hal_streaming_buffer_ref_t stream_ref;
    iree_status_t lookup_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)device_ptr, &stream_ref);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
    bindings[resolve_op.dst_ordinal] =
        iree_hal_streaming_convert_buffer_ref(stream_ref);
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
    // TODO: Add HAL dispatch flag for cooperative kernel support and pass
    // through to the backend. For now, return unimplemented.
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "cooperative kernel launch not yet implemented in HAL layer");
  }

  // Verify parameter buffer.
  // TODO(benvanik): pass size when we have it so we can check it.
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
  // Zero-initialize constants to prevent uninitialized padding bytes from
  // being misinterpreted as device pointers by the overlay scan.
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

  size_t constants_size = symbol->parameters.constant_bytes;
  // Track if we need to use raw argument passing (e.g., for external pointers).
  bool use_raw_arguments = false;

  // Check if this is a pre-packed buffer (HIP_LAUNCH_PARAM_BUFFER format).
  // Pre-packed buffers are already in the kernel's native ABI format and should
  // be passed directly without any unpacking or translation.
  bool is_pre_packed =
      (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_PRE_PACKED) != 0;

  uint64_t timing_params_start_ns =
      timing_enabled ? hrx_launch_timing_now_ns() : 0;
  if (is_pre_packed && params->buffer) {
    // Pre-packed buffer: pass the raw buffer directly to the kernel.
    // This is used for kernels launched via HIP_LAUNCH_PARAM_BUFFER_POINTER
    // (e.g., hipBLASLt GEMM kernels) where the buffer is already packed.
    if (params->buffer_size != 0) {
      if (params->buffer_size < symbol->parameters.constant_bytes) {
        constants_size = symbol->parameters.constant_bytes;
        constants = iree_alloca(constants_size);
        memset(constants, 0, constants_size);
        memcpy(constants, params->buffer, params->buffer_size);
      } else {
        constants = params->buffer;
        constants_size = params->buffer_size;
      }
    } else {
      constants = params->buffer;
    }
    binding_list.count = 0;  // No IREE bindings, using raw pointers.
    use_raw_arguments = true;
  } else if (is_native_kernel && params->buffer) {
    // Native kernel with pre-packed buffer: pass raw arguments directly.
    // For native kernels, params->buffer contains the pre-packed kernel
    // arguments that should be passed as-is to the GPU.
    constants = params->buffer;
    if (params->buffer_size > 0) {
      constants_size = params->buffer_size;
    }
    use_raw_arguments = true;
  } else if (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) {
    // Unpack parameters from array of pointers (void**).
    iree_status_t unpack_status = iree_hal_streaming_unpack_parameter_list(
        stream->context, &symbol->parameters, (void**)params->buffer, constants,
        &binding_list);
    if (!iree_status_is_ok(unpack_status)) {
      // If unpack fails due to NULL or external device pointers, fall back
      // to raw argument passing. This handles native kernels with optional
      // parameters or external allocations.
      if (iree_status_code(unpack_status) == IREE_STATUS_NOT_FOUND) {
        iree_status_ignore(unpack_status);

        // Convert void** to packed buffer format.
        // Pack all parameters (copies + resolves) into a single constants
        // buffer. For native kernels, we use the parameter metadata to know the
        // layout.
        const iree_hal_streaming_parameter_info_t* params_info =
            &symbol->parameters;
        void** args_array = (void**)params->buffer;

        // Build a native direct-argument buffer from the args array. This is
        // only for fallback cases where pointers cannot be represented as HAL
        // bindings.
        constants_size = params_info->direct_arg_bytes
                             ? params_info->direct_arg_bytes
                             : params_info->constant_bytes;
        constants = iree_alloca(constants_size);
        memset(constants, 0, constants_size);

        // Process all copy operations (constants/scalars).
        const iree_hal_streaming_parameter_op_t* op = &params_info->ops[0];
        for (uint32_t i = 0; i < params_info->copy_count; ++i, ++op) {
          const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
          // Dereference the arg pointer to get the value.
          void* param_ptr = args_array[copy_op.src_ordinal];
          memcpy((uint8_t*)constants + copy_op.direct_dst_offset, param_ptr,
                 copy_op.size);
        }

        // Process all resolve operations (bindings/pointers) - copy the raw
        // pointer. For native kernels, we pass the device pointer value
        // directly.
        for (uint32_t i = 0; i < params_info->binding_count; ++i, ++op) {
          const iree_hal_streaming_parameter_resolve_op_t resolve_op =
              op->resolve;
          void* param_ptr = args_array[resolve_op.src_ordinal];
          // The parameter points to a device pointer (void*)
          void* device_ptr = *(void**)param_ptr;
          // Copy the raw device pointer value into the constants buffer at the
          // correct kernel ABI offset. dst_offset is the kernel's argument
          // offset (from the code object metadata), NOT src_offset (which is a
          // running total used for source buffer layout in non-args-array
          // mode).
          memcpy((uint8_t*)constants + resolve_op.dst_offset, &device_ptr,
                 sizeof(void*));
        }

        binding_list.count = 0;  // No IREE bindings, using raw pointers.
        use_raw_arguments = true;
      } else {
        IREE_TRACE_ZONE_END(z0);
        return unpack_status;
      }
    }
  } else {
    // Unpack parameters from packed buffer.
    iree_status_t unpack_status = iree_hal_streaming_unpack_parameters(
        stream->context, &symbol->parameters, params->buffer, constants,
        &binding_list);
    if (!iree_status_is_ok(unpack_status)) {
      // If unpack fails due to NULL or external device pointers, fall back
      // to raw argument passing. This handles native kernels with optional
      // parameters or external allocations (e.g., hipBLASLt workspace).
      if (iree_status_code(unpack_status) == IREE_STATUS_NOT_FOUND) {
        iree_status_ignore(unpack_status);
        // Use the raw packed buffer directly.
        constants = params->buffer;
        constants_size = params->buffer_size;
        binding_list.count = 0;  // No IREE bindings, using raw pointers.
        use_raw_arguments = true;
      } else {
        IREE_TRACE_ZONE_END(z0);
        return unpack_status;
      }
    }
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

  // Use the metadata-described constants+bindings path whenever possible. The
  // AMDGPU backend uses HSACO metadata on that path to synthesize HIP/OpenCL
  // hidden kernargs such as group size and block count. CUSTOM_DIRECT_ARGUMENTS
  // is reserved for callers that supply or require native kernarg bytes.
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
    // writes. A bare execution barrier with no memory_barriers (count=0)
    // resolves to NONE/NONE scopes after upstream IREE commit 48af1651a1
    // ("Preserve command-buffer barrier scopes") and lets later dispatches
    // launch with stale cache state, producing garbage output (e.g. NaN
    // logits in GPT-2 forward).
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

// Host callback wrapper structure to adapt CUDA/HIP callbacks to HAL callbacks.
typedef struct iree_hal_streaming_host_callback_t {
  void (*fn)(void* user_data);
  void* user_data;
} iree_hal_streaming_host_callback_t;

// HAL host call function that invokes the CUDA/HIP style callback.
static iree_status_t iree_hal_streaming_host_callback_thunk(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  iree_hal_streaming_host_callback_t* callback =
      (iree_hal_streaming_host_callback_t*)user_data;
  callback->fn(callback->user_data);
  iree_allocator_free(iree_allocator_system(), callback);
  return iree_ok_status();
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

  // Flush any pending operations in the stream's command buffer.
  if (stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_flush(stream));
  }

  // Allocate a wrapper structure to hold the callback and user data.
  iree_hal_streaming_host_callback_t* callback = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(iree_allocator_system(), sizeof(*callback),
                                (void**)&callback));
  callback->fn = fn;
  callback->user_data = user_data;

  // Set up semaphores for the host call.
  // Wait for the current stream position.
  uint64_t wait_value = stream->pending_value;
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = wait_value > 0 ? 1 : 0,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };

  // Signal the next value after the host call completes.
  uint64_t signal_value = stream->pending_value + 1;
  stream->pending_value = signal_value;
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &signal_value,
  };

  // Create the host call with our wrapper function.
  iree_hal_host_call_t call =
      iree_hal_make_host_call(iree_hal_streaming_host_callback_thunk, callback);

  // Empty args array (not used by CUDA/HIP callbacks).
  uint64_t args[4] = {0, 0, 0, 0};

  // Enqueue the host call on the device queue.
  // Use blocking mode so that stream synchronization waits for the host
  // function to complete before returning.
  iree_status_t status = iree_hal_device_queue_host_call(
      stream->context->device, stream->queue_affinity, wait_semaphores,
      signal_semaphores, call, args, /*flags=*/0);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), callback);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
