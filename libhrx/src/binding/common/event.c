// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/internal.h"

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_event_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_event_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event));

  // Initialize event.
  iree_atomic_ref_count_init(&event->ref_count);
  event->flags = flags;
  iree_slim_mutex_initialize(&event->mutex);
  event->recorded_point = (iree_hal_streaming_recorded_point_t){0};
  event->recording_stream = NULL;
  event->context = context;
  iree_hal_streaming_context_retain(context);
  event->ipc_handle = NULL;
  event->capture_graph = NULL;
  event->capture_dependencies = NULL;
  event->capture_dependency_count = 0;
  event->capture_dependency_capacity = 0;
  event->host_allocator = host_allocator;

  *out_event = event;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_streaming_event_destroy(
    iree_hal_streaming_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release the recorded point.
  iree_hal_streaming_event_release_recorded_point(&event->recorded_point);

  // Release recording stream reference.
  iree_hal_streaming_stream_release(event->recording_stream);

  // Release context.
  iree_hal_streaming_context_release(event->context);

  iree_hal_streaming_graph_release(event->capture_graph);
  iree_allocator_free(event->host_allocator, event->capture_dependencies);

  // Free event memory.
  iree_slim_mutex_deinitialize(&event->mutex);
  iree_allocator_t host_allocator = event->host_allocator;
  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_event_retain(iree_hal_streaming_event_t* event) {
  if (event) {
    iree_atomic_ref_count_inc(&event->ref_count);
  }
}

void iree_hal_streaming_event_release(iree_hal_streaming_event_t* event) {
  if (event && iree_atomic_ref_count_dec(&event->ref_count) == 1) {
    iree_hal_streaming_event_destroy(event);
  }
}

void iree_hal_streaming_event_acquire_recorded_point(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_recorded_point_t* out_point) {
  iree_slim_mutex_lock(&event->mutex);
  *out_point = event->recorded_point;
  iree_hal_semaphore_retain(out_point->semaphore);
  iree_hal_streaming_event_timestamp_slot_retain(out_point->timestamp_slot);
  iree_slim_mutex_unlock(&event->mutex);
}

void iree_hal_streaming_event_release_recorded_point(
    iree_hal_streaming_recorded_point_t* point) {
  // Released before the semaphore: the slot's retirement condition is the
  // point's own semaphore payload, so the pool reads it while this reference
  // still holds the semaphore up.
  iree_hal_streaming_event_timestamp_slot_release(
      point->timestamp_slot, point->semaphore, point->value);
  iree_hal_semaphore_release(point->semaphore);
  *point = (iree_hal_streaming_recorded_point_t){0};
}

void iree_hal_streaming_event_commit_recorded_point(
    iree_hal_streaming_event_t* event,
    iree_hal_streaming_recorded_point_t point) {
  iree_slim_mutex_lock(&event->mutex);
  iree_hal_streaming_recorded_point_t previous = event->recorded_point;
  event->recorded_point = point;
  iree_slim_mutex_unlock(&event->mutex);
  // Dropped outside the lock: returning a tick slot to its pool takes the pool
  // mutex, and that mutex and this one are both leaves that no path holds at
  // the same time.
  iree_hal_streaming_event_release_recorded_point(&previous);
}

iree_hal_streaming_stream_t* iree_hal_streaming_event_exchange_recording_stream(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream) {
  iree_slim_mutex_lock(&event->mutex);
  iree_hal_streaming_stream_t* previous_stream = event->recording_stream;
  if (previous_stream != stream) {
    iree_hal_streaming_stream_retain(stream);
    event->recording_stream = stream;
  } else {
    previous_stream = NULL;
  }
  iree_slim_mutex_unlock(&event->mutex);
  return previous_stream;
}

iree_hal_streaming_graph_t* iree_hal_streaming_event_acquire_capture_graph(
    iree_hal_streaming_event_t* event) {
  iree_slim_mutex_lock(&event->mutex);
  iree_hal_streaming_graph_t* capture_graph = event->capture_graph;
  iree_hal_streaming_graph_retain(capture_graph);
  iree_slim_mutex_unlock(&event->mutex);
  return capture_graph;
}

iree_hal_streaming_graph_t* iree_hal_streaming_event_exchange_capture_graph(
    iree_hal_streaming_event_t* event, iree_hal_streaming_graph_t* graph) {
  iree_slim_mutex_lock(&event->mutex);
  iree_hal_streaming_graph_t* previous_graph = event->capture_graph;
  if (previous_graph != graph) {
    iree_hal_streaming_graph_retain(graph);
    event->capture_graph = graph;
  } else {
    previous_graph = NULL;
  }
  iree_slim_mutex_unlock(&event->mutex);
  return previous_graph;
}

// Returns whether |point| has been reached. A point naming no timeline has no
// submitted record and reads as reached. Borrows |point|: the reference taken
// when the point was acquired stays with the acquirer, which releases it.
static iree_status_t iree_hal_streaming_recorded_point_query(
    const iree_hal_streaming_recorded_point_t* point, bool* out_reached) {
  *out_reached = true;
  if (!point->semaphore) return iree_ok_status();
  uint64_t current_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_query(point->semaphore, &current_value));
  *out_reached = current_value >= point->value;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_query(iree_hal_streaming_event_t* event,
                                             int* status) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(status);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);
  bool reached = false;
  iree_status_t query_status =
      iree_hal_streaming_recorded_point_query(&recorded_point, &reached);
  iree_hal_streaming_event_release_recorded_point(&recorded_point);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, query_status);

  // 0=complete, 1=not complete.
  *status = reached ? 0 : 1;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_enqueue_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream,
    iree_hal_semaphore_list_t wait_semaphores,
    iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_streaming_recorded_point_t* point) {
  iree_hal_streaming_context_t* context = stream->context;
  // The slot a timed record captures into is suballocated from |context|'s
  // pool and outlives the record on the point the event holds. Nothing the
  // point names keeps that pool alive - a slot reference is a count on the
  // slot alone - so only the reference the event holds on its own context
  // does, and a slot drawn from any other context's pool can be left naming
  // storage that context freed. Ticks are counted on the recording device's
  // clock besides, and iree_hal_streaming_event_elapsed_time converts them
  // with the event's own context domain.
  if (event->context != context) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "an event can only be recorded on a stream of the context that "
        "created it");
  }

  const bool captures_tick =
      context->timestamp_domain.frequency_hz != 0 &&
      !(event->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING);

  iree_hal_streaming_event_timestamp_slot_t* slot = NULL;
  if (captures_tick) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_event_timestamp_pool_acquire(
        &context->timestamp_pool, &slot));
  }

  const iree_status_t status =
      slot ? iree_hal_device_queue_timestamp(
                 context->device, stream->queue_affinity, wait_semaphores,
                 signal_semaphores,
                 iree_hal_streaming_event_timestamp_slot_buffer(slot),
                 iree_hal_streaming_event_timestamp_slot_offset(slot),
                 IREE_HAL_TIMESTAMP_FLAG_NONE)
           : iree_hal_device_queue_barrier(
                 context->device, stream->queue_affinity, wait_semaphores,
                 signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    // A rejected enqueue leaves no write outstanding against the slot, which
    // only this path can say; every other release names the point's own
    // retirement condition.
    iree_hal_streaming_event_timestamp_slot_release(slot, NULL, 0);
    return status;
  }

  // The point owns what it names from here.
  iree_hal_semaphore_retain(point->semaphore);
  point->timestamp_slot = slot;
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_record(
    iree_hal_streaming_event_t* event, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    if (event->capture_dependency_capacity < stream->capture_dependency_count) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_allocator_realloc(event->host_allocator,
                                     stream->capture_dependency_count *
                                         sizeof(*event->capture_dependencies),
                                     (void**)&event->capture_dependencies));
      event->capture_dependency_capacity = stream->capture_dependency_count;
    }
    if (stream->capture_dependency_count > 0) {
      memcpy(event->capture_dependencies, stream->capture_dependencies,
             stream->capture_dependency_count *
                 sizeof(*event->capture_dependencies));
    }
    event->capture_dependency_count = stream->capture_dependency_count;
    // No lock is held here, so the displaced graph can be released inline.
    iree_hal_streaming_graph_release(
        iree_hal_streaming_event_exchange_capture_graph(event,
                                                        stream->capture_graph));
    // A captured record produces no submission, so it leaves the recorded point
    // alone; only the stream is adopted, for the later wait that picks the
    // capture up from the stream that captured it.
    iree_hal_streaming_stream_release(
        iree_hal_streaming_event_exchange_recording_stream(event, stream));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  iree_slim_mutex_lock(&stream->mutex);

  // The record marks a point on the recording stream's timeline: the barrier
  // takes the stream's next value and signals it once everything already on the
  // stream has completed.
  uint64_t stream_wait_value = 0;
  uint64_t stream_signal_value = 0;
  iree_status_t status = iree_hal_streaming_stream_reserve_next_value_locked(
      stream, &stream_wait_value, &stream_signal_value);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&stream->mutex);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // A stream that has never submitted has nothing behind it to wait on.
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = stream_wait_value > 0 ? 1 : 0,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &stream_wait_value,
  };
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &stream_signal_value,
  };

  // The record signals the stream's own timeline, so reaching the point is
  // exactly the stream reaching that value.
  iree_hal_streaming_recorded_point_t recorded_point = {
      .semaphore = stream->timeline_semaphore,
      .value = stream_signal_value,
      .ordered_after_stream_id = stream->stream_id,
      .ordered_after_stream_value = stream_signal_value,
  };
  iree_hal_streaming_stream_t* previous_stream = NULL;
  status = iree_hal_streaming_event_enqueue_record(
      event, stream, wait_semaphores, signal_semaphores, &recorded_point);
  if (iree_status_is_ok(status)) {
    // The accepted enqueue owns the value it signals, so the timeline advances
    // here and stays advanced even when the flush below fails.
    stream->pending_value = stream_signal_value;
    // A rejected submission signals nothing, so the event keeps its old point.
    iree_hal_streaming_event_commit_recorded_point(event, recorded_point);
    previous_stream =
        iree_hal_streaming_event_exchange_recording_stream(event, stream);
    status = iree_hal_device_queue_flush(stream->context->device,
                                         stream->queue_affinity);
  }
  iree_slim_mutex_unlock(&stream->mutex);
  // Stream teardown re-enters the streaming layer, so the displaced reference
  // is dropped outside this stream's mutex.
  iree_hal_streaming_stream_release(previous_stream);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_synchronize(
    iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_recorded_point_t recorded_point;
  iree_hal_streaming_event_acquire_recorded_point(event, &recorded_point);
  // A point naming no timeline has no submitted record behind it and nothing to
  // wait for.
  iree_status_t status = iree_ok_status();
  if (recorded_point.semaphore) {
    status = iree_hal_semaphore_wait(
        recorded_point.semaphore, recorded_point.value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_streaming_event_release_recorded_point(&recorded_point);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

float iree_hal_streaming_timestamp_domain_elapsed_ms(
    iree_hal_streaming_timestamp_domain_t domain, uint64_t start_tick,
    uint64_t stop_tick) {
  const uint64_t mask = domain.valid_bits == 64
                            ? UINT64_MAX
                            : ((1ull << domain.valid_bits) - 1ull);
  const uint64_t delta = (stop_tick - start_tick) & mask;
  // The counter's top bit, derived from the mask so it is defined at every
  // width the domain can carry, including none: there the mask takes the pair
  // to zero, which sits below this.
  const uint64_t sign_bit = (mask >> 1) + 1ull;
  // Formed as a double because the scaling below is floating point and the
  // result is a float. The offset itself fits int64_t at every width the
  // domain can carry: for delta >= sign_bit, mask - delta is at most
  // INT64_MAX, so -(mask - delta) - 1 bottoms out at exactly INT64_MIN.
  const double signed_delta =
      delta >= sign_bit ? -(double)(mask - delta) - 1.0 : (double)delta;
  return (float)(signed_delta * 1000.0 / (double)domain.frequency_hz);
}

// Reads the ticks the reached records |start_point| and |stop_point| captured
// and converts the interval between them with |domain|, writing |*out_ms| only
// when both reads succeed. Both points must name a tick slot, which the caller
// establishes before querying either record.
static iree_status_t iree_hal_streaming_recorded_point_pair_elapsed_ms(
    iree_hal_streaming_timestamp_domain_t domain,
    const iree_hal_streaming_recorded_point_t* start_point,
    const iree_hal_streaming_recorded_point_t* stop_point, float* out_ms) {
  uint64_t start_tick = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_read(iree_hal_streaming_event_timestamp_slot_buffer(
                                   start_point->timestamp_slot),
                               iree_hal_streaming_event_timestamp_slot_offset(
                                   start_point->timestamp_slot),
                               &start_tick, sizeof(start_tick)));
  uint64_t stop_tick = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_read(iree_hal_streaming_event_timestamp_slot_buffer(
                                   stop_point->timestamp_slot),
                               iree_hal_streaming_event_timestamp_slot_offset(
                                   stop_point->timestamp_slot),
                               &stop_tick, sizeof(stop_tick)));
  *out_ms = iree_hal_streaming_timestamp_domain_elapsed_ms(domain, start_tick,
                                                           stop_tick);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_event_elapsed_time(
    float* ms, iree_hal_streaming_event_t* start,
    iree_hal_streaming_event_t* stop,
    iree_hal_streaming_event_timing_t* out_timing) {
  IREE_ASSERT_ARGUMENT(ms);
  IREE_ASSERT_ARGUMENT(start);
  IREE_ASSERT_ARGUMENT(stop);
  IREE_ASSERT_ARGUMENT(out_timing);
  *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_UNTIMED;

  // An event with timing disabled captures no tick, so there is no interval
  // between two such records to measure.
  if ((start->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING) ||
      (stop->flags & IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING)) {
    return iree_ok_status();
  }

  // Each record is taken once and the interval is measured between exactly the
  // two records queried below, so a record landing on either event concurrently
  // cannot pair one record's tick with another record's point.
  iree_hal_streaming_recorded_point_t start_point;
  iree_hal_streaming_recorded_point_t stop_point;
  iree_hal_streaming_event_acquire_recorded_point(start, &start_point);
  iree_hal_streaming_event_acquire_recorded_point(stop, &stop_point);

  iree_status_t status = iree_ok_status();
  if (!start_point.semaphore || !stop_point.semaphore) {
    // An event with no submitted record names no point to measure from, which
    // is what the outcome already reads as.
  } else if (!start_point.timestamp_slot || !stop_point.timestamp_slot) {
    // Both records were submitted with timing enabled, and a record is made
    // only on a stream of its own event's context, so the one thing that can
    // leave a record without a tick is that context advertising no domain to
    // capture in. Answered before the timelines are queried: reaching a record
    // that captured nothing produces no measurement either.
    *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_UNSUPPORTED;
  } else {
    // Both ticks are defined once both records have been reached. The stop
    // record is only queried once the start record is known to have been
    // reached, so an outstanding start reports the interval as incomplete
    // without touching the stop timeline.
    bool reached = false;
    status = iree_hal_streaming_recorded_point_query(&start_point, &reached);
    if (iree_status_is_ok(status) && reached) {
      status = iree_hal_streaming_recorded_point_query(&stop_point, &reached);
    }
    if (iree_status_is_ok(status)) {
      if (reached) {
        // Ticks are comparable only inside one device's domain, and this
        // converts with the start event's context domain. A record is made
        // only on a stream of its own event's context, and both bindings
        // refuse a pair whose events come from different contexts before it
        // reaches here, so both ticks were captured in that domain.
        status = iree_hal_streaming_recorded_point_pair_elapsed_ms(
            start->context->timestamp_domain, &start_point, &stop_point, ms);
        if (iree_status_is_ok(status)) {
          *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_MEASURED;
        }
      } else {
        *out_timing = IREE_HAL_STREAMING_EVENT_TIMING_INCOMPLETE;
      }
    }
  }

  iree_hal_streaming_event_release_recorded_point(&start_point);
  iree_hal_streaming_event_release_recorded_point(&stop_point);
  return status;
}
