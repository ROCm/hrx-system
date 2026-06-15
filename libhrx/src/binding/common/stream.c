// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
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
  stream->capture_id = 0;
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

  // Capture and clear context pointer to prevent re-entry during unregister.
  iree_hal_streaming_context_t* context = stream->context;
  stream->context = NULL;

  // Synchronize stream before cleanup to ensure all operations complete.
  // This is important to avoid leaking resources from pending operations.
  iree_status_ignore(iree_hal_streaming_stream_synchronize(stream));

  // Unregister from context before cleanup.
  // Note: We already cleared stream->context, so if unregister tries to
  // release and that triggers another destroy, it will be a no-op.
  if (context) {
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

iree_status_t iree_hal_streaming_stream_begin(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&stream->mutex);

  iree_status_t status = iree_ok_status();

  // Create command buffer if not already created.
  // Note that we set UNRETAINED as we ensure the resources we have to track are
  // retained at the graph exec level and CUDA/HIP don't make any statements
  // about resource lifetime.
  if (!stream->command_buffer) {
    status = iree_hal_command_buffer_create(
        stream->context->device,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
        IREE_HAL_COMMAND_CATEGORY_ANY, stream->queue_affinity,
        /*binding_capacity=*/0, &stream->command_buffer);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stream->mutex);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Begin recording.
  status = iree_hal_command_buffer_begin(stream->command_buffer);

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
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Event wait during graph capture is not yet implemented.
    // TODO(#graph-capture): Add wait node to graph.
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "event wait during graph capture not yet implemented");
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
  // For native kernels with NULL or external device pointers, we can't use
  // IREE's binding mechanism. In that case, the caller should use
  // CUSTOM_DIRECT_ARGUMENTS to pass the raw parameter buffer directly.
  iree_hal_buffer_ref_t* bindings =
      (iree_hal_buffer_ref_t*)out_bindings->values;
  for (uint32_t i = 0; i < parameters->binding_count; ++i, ++op) {
    const iree_hal_streaming_parameter_resolve_op_t resolve_op = op->resolve;
    void* device_ptr = *(void**)(parameter_buffer + resolve_op.src_offset);
    // TODO(benvanik): possibly calculate proper range here? We could easily
    // (at only the cost of a cache miss) get the total buffer size and then
    // subtract the offset to get the remaining size.

    // Handle NULL device pointers - some kernels pass NULL for optional
    // buffers. Return NOT_FOUND to signal that this kernel needs raw argument
    // passing.
    if (!device_ptr) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "binding %u has NULL device pointer", i);
    }

    iree_hal_streaming_buffer_ref_t stream_ref;
    iree_status_t lookup_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)device_ptr, &stream_ref);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
    bindings[resolve_op.dst_ordinal] = iree_hal_make_buffer_ref(
        stream_ref.buffer->buffer, stream_ref.offset, IREE_HAL_WHOLE_BUFFER);
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
  // For native kernels with NULL or external device pointers, we can't use
  // IREE's binding mechanism. In that case, the caller should use
  // CUSTOM_DIRECT_ARGUMENTS to pass the raw parameter buffer directly.
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

    // Handle NULL device pointers - some kernels pass NULL for optional
    // buffers. Return NOT_FOUND to signal that this kernel needs raw argument
    // passing.
    if (!device_ptr) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "binding %u has NULL device pointer", i);
    }

    iree_hal_streaming_buffer_ref_t stream_ref;
    iree_status_t lookup_status = iree_hal_streaming_memory_lookup(
        context, (iree_hal_streaming_deviceptr_t)device_ptr, &stream_ref);
    // If lookup fails, the kernel uses external device pointers.
    // Return NOT_FOUND to signal that this kernel needs raw argument passing.
    if (!iree_status_is_ok(lookup_status)) {
      return lookup_status;
    }
    bindings[resolve_op.dst_ordinal] = iree_hal_make_buffer_ref(
        stream_ref.buffer->buffer, stream_ref.offset, IREE_HAL_WHOLE_BUFFER);
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
  const bool use_direct_queue_dispatch = hrx_direct_queue_dispatch_enabled();

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
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_kernel_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, symbol, params, NULL));
    // Clear dependencies after adding the node.
    stream->capture_dependency_count = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Ensure prior command-buffer work is submitted before direct dispatches.
  // Direct dispatches use the stream timeline wait/signal chain below; command
  // buffer dispatches continue recording into the current stream command
  // buffer.
  if (use_direct_queue_dispatch && stream->command_buffer) {
    uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    iree_status_t flush_status = iree_hal_streaming_stream_flush(stream);
    if (timing_enabled) {
      timing_begin_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, flush_status);
  }

  // Ensure command buffer is recording for the existing batched path.
  if (!use_direct_queue_dispatch && !stream->command_buffer) {
    uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    iree_status_t begin_status = iree_hal_streaming_stream_begin(stream);
    if (timing_enabled) {
      timing_begin_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, begin_status);
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

        // Build packed buffer from args array using the ops.
        // Stack-allocate the buffer (using constant_bytes as size).
        constants = iree_alloca(params_info->constant_bytes);
        memset(constants, 0, params_info->constant_bytes);
        constants_size = params_info->constant_bytes;

        // Process all copy operations (constants/scalars).
        const iree_hal_streaming_parameter_op_t* op = &params_info->ops[0];
        for (uint32_t i = 0; i < params_info->copy_count; ++i, ++op) {
          const iree_hal_streaming_parameter_copy_op_t copy_op = op->copy;
          // Dereference the arg pointer to get the value.
          void* param_ptr = args_array[copy_op.src_ordinal];
          memcpy((uint8_t*)constants + copy_op.dst_offset, param_ptr,
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

  // Ensure constants_size is 4-byte aligned as required by HAL.
  constants_size = (constants_size + 3) & ~(size_t)3;

  // --- Resolve pointer-valued constants to overlay bindings ---
  // The constants buffer may contain device pointers passed as raw values
  // (e.g., embedded in struct-typed copy parameters like CatArrInputTensor).
  // For the remote HAL, these are synthetic addresses (0xDEAD...) that the
  // remote device cannot use. We scan the constants for known buffer table
  // entries and convert them to HAL buffer bindings. The dispatch handler
  // overlays the resolved device pointers on top of the constants in the
  // kernarg buffer.
  //
  // When the kernel also has regular bindings (binding_count > 0), we must
  // convert those to overlay format too: write their device pointer values
  // into the constants at their ABI offsets so the scan can find them.
  if (constants && constants_size >= sizeof(void*)) {
    // If we have regular bindings, convert them to overlay format by writing
    // the device pointer values into the constants buffer at their ABI
    // offsets. This allows the overlay scan below to find them alongside
    // any pointers embedded in copy data.
    if (binding_list.count > 0 && !use_raw_arguments && !is_pre_packed) {
      bool is_args_array =
          (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_ARGS_ARRAY) != 0;
      const iree_hal_streaming_parameter_op_t* op =
          &symbol->parameters.ops[symbol->parameters.copy_count];
      for (uint32_t i = 0; i < symbol->parameters.binding_count; ++i, ++op) {
        const iree_hal_streaming_parameter_resolve_op_t resolve_op =
            op->resolve;
        void* device_ptr;
        if (is_args_array) {
          void** args_array = (void**)params->buffer;
          device_ptr = *(void**)args_array[resolve_op.src_ordinal];
        } else {
          const uint8_t* parameter_buffer = (const uint8_t*)params->buffer;
          device_ptr = *(void**)(parameter_buffer + resolve_op.src_offset);
        }
        if (device_ptr &&
            resolve_op.dst_offset + sizeof(void*) <= constants_size) {
          memcpy((uint8_t*)constants + resolve_op.dst_offset, &device_ptr,
                 sizeof(void*));
        }
      }
      binding_list.count = 0;
    }

    // NOTE: earlier drafts of this code tried to extract device pointers
    // from the constants buffer and hand them to the HAL as separate
    // bindings with a |buffer_slot = byte_offset| overlay trick. The
    // AMDGPU aql_command_buffer / host_queue_dispatch backends do not
    // implement that overlay: for CUSTOM_DIRECT_ARGUMENTS they just
    // memcpy |constants| into the kernarg block and ignore |bindings|
    // entirely. Zeroing the pointer in the constants buffer therefore
    // caused the GPU to dereference NULL and page-fault.
    //
    // Pyre's hipMalloc returns the real GPU virtual address of the
    // allocation (sub-allocated from a contiguous pool), so whatever
    // pointers PyTorch writes into the HIP launch parameter buffer are
    // already valid device addresses. We just let the constants flow
    // through unmodified; the HAL copies them into kernargs verbatim and
    // the kernel dereferences them directly.
    //
    // We still walk the buffer once below purely to run the lookup as
    // validation (so that invalid device pointers are surfaced in debug
    // logs), but we do not mutate the constants or build overlay
    // bindings.
    (void)iree_hal_streaming_memory_lookup;
  }

  // After the overlay scan, all bindings are in overlay format (buffer_slot
  // encodes the constant byte offset). Always use CUSTOM_DIRECT_ARGUMENTS
  // so the server overlays resolved device pointers into the constants.
  //
  // For pre-packed buffers (HIP_LAUNCH_PARAM_BUFFER format used by
  // hipBLAS/hipBLASLt) we rely on the same CUSTOM_DIRECT_ARGUMENTS path: the
  // AMDGPU HAL simply memcpys |constants| into the kernarg block and ignores
  // any binding list, which is exactly what pre-packed callers want.
  iree_hal_dispatch_flags_t flags =
      IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;

  uint64_t timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
  iree_status_t status = iree_ok_status();
  if (use_direct_queue_dispatch) {
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
    status = iree_hal_command_buffer_dispatch(
        stream->command_buffer, symbol->executable,
        iree_hal_executable_function_from_index(symbol->export_ordinal), config,
        iree_make_const_byte_span(constants, constants_size), binding_list,
        flags);
  }
  if (timing_enabled) {
    timing_dispatch_ns += hrx_launch_timing_now_ns() - timing_step_ns;
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
  if (!use_direct_queue_dispatch && iree_status_is_ok(status) &&
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
    timing_step_ns = timing_enabled ? hrx_launch_timing_now_ns() : 0;
    status = iree_hal_command_buffer_execution_barrier(
        stream->command_buffer,
        IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, NULL);
    if (timing_enabled) {
      timing_barrier_ns += hrx_launch_timing_now_ns() - timing_step_ns;
    }
  }

  if (!use_direct_queue_dispatch && iree_status_is_ok(status)) {
    ++stream->pending_launch_count;
  }

  const int flush_interval = hrx_flush_interval();
  if (!use_direct_queue_dispatch && iree_status_is_ok(status) &&
      (hrx_flush_each_launch_enabled() ||
       (flush_interval > 0 &&
        stream->pending_launch_count >= (uint32_t)flush_interval))) {
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
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_host_call_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, fn, user_data, NULL));
    // Clear dependencies after adding the node.
    stream->capture_dependency_count = 0;
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
