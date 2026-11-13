// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/profile.h"

#include <inttypes.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/utils/profile_event_ring.h"

// Each enabled stream retains this many records between explicit flushes.
#define IREE_HAL_AMD_XDNA_PROFILE_CAPACITY 4096

struct iree_hal_amd_xdna_profile_t {
  // Allocator owning this object and its placed ring storage.
  iree_allocator_t host_allocator;
  // Retained sink, invoked only at begin/flush/end.
  iree_hal_profile_sink_t* sink;
  // Optional finite native counter and executable metadata capture.
  iree_hal_amd_xdna_profile_counters_t* counters;
  // Guards ring positions and contents during append and snapshot commit.
  iree_slim_mutex_t mutex;
  // Session-local producer identity shared by metadata and event chunks.
  iree_hal_profile_chunk_metadata_t metadata;
  // Public transaction identities, independent of native FIFO epochs.
  iree_atomic_int64_t next_submission_id;
  // Host submission records; disabled when its capacity is zero.
  iree_hal_profile_event_ring_t queue_events;
  // Completed host execution spans; disabled when its capacity is zero.
  iree_hal_profile_event_ring_t host_events;
};

static iree_status_t iree_hal_amd_xdna_profile_write(
    iree_hal_amd_xdna_profile_t* profile, iree_string_view_t content_type,
    iree_const_byte_span_t bytes) {
  iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
  metadata.content_type = content_type;
  return iree_hal_profile_sink_write(profile->sink, &metadata, 1, &bytes);
}

static void iree_hal_amd_xdna_profile_free(
    iree_hal_amd_xdna_profile_t* profile) {
  iree_hal_profile_sink_release(profile->sink);
  iree_slim_mutex_deinitialize(&profile->mutex);
  iree_allocator_free(profile->host_allocator, profile);
}

iree_status_t iree_hal_amd_xdna_profile_begin(
    const iree_hal_device_profiling_options_t* options,
    iree_hal_profile_session_id_t session_id, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, iree_hal_amd_xdna_context_t* context,
    iree_hal_amd_xdna_command_arena_t* command_arena,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_profile_t** out_profile) {
  *out_profile = NULL;
  const iree_hal_device_profiling_data_families_t lightweight =
      IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS;
  const iree_hal_device_profiling_data_families_t supported =
      lightweight | IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  if (iree_any_bit_set(
          options->flags,
          ~IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported XDNA profiling flags 0x%x",
                            options->flags);
  }
  iree_hal_device_profiling_data_families_t families = options->data_families;
  if (!families &&
      iree_hal_device_profiling_options_requests_lightweight_statistics(
          options)) {
    families = lightweight;
  }
  if (!families) {
    return iree_ok_status();
  }
  if (families & ~supported) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported XDNA profiling data families 0x%" PRIx64,
        families & ~supported);
  }
  if (!options->sink) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA profiling requires a sink");
  }
  if (options->capture_filter.reserved0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profiling filter reserved fields must be zero");
  }
  const iree_hal_profile_capture_filter_flags_t supported_filters =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN |
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID |
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX |
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL |
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL;
  if (options->capture_filter.flags & ~supported_filters) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported XDNA capture filter flags");
  }
  if (options->counter_set_count &&
      (!options->counter_sets ||
       !(families & IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "counter selections require XDNA counter capture");
  }
  const iree_host_size_t queue_capacity =
      families & IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS
          ? IREE_HAL_AMD_XDNA_PROFILE_CAPACITY
          : 0;
  const iree_host_size_t host_capacity =
      families & IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS
          ? IREE_HAL_AMD_XDNA_PROFILE_CAPACITY
          : 0;
  iree_host_size_t size = 0, queue_offset = 0, host_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_profile_t), &size,
      IREE_STRUCT_FIELD(queue_capacity, iree_hal_profile_queue_event_t,
                        &queue_offset),
      IREE_STRUCT_FIELD(host_capacity, iree_hal_profile_host_execution_event_t,
                        &host_offset)));
  iree_hal_amd_xdna_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&profile));
  memset(profile, 0, sizeof(*profile));
  profile->host_allocator = host_allocator;
  profile->sink = options->sink;
  iree_hal_profile_sink_retain(profile->sink);
  iree_slim_mutex_initialize(&profile->mutex);
  iree_atomic_store(&profile->next_submission_id, 1, iree_memory_order_relaxed);
  profile->metadata = iree_hal_profile_chunk_metadata_default();
  profile->metadata.name = IREE_SV("xdna");
  profile->metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  profile->metadata.session_id = session_id;
  profile->metadata.physical_device_ordinal = physical_device_ordinal;
  profile->metadata.queue_ordinal = queue_ordinal;
  profile->metadata.stream_id =
      ((uint64_t)physical_device_ordinal << 32) | queue_ordinal;
  iree_hal_profile_event_ring_initialize(
      (uint8_t*)profile + queue_offset, sizeof(iree_hal_profile_queue_event_t),
      queue_capacity, &profile->queue_events);
  iree_hal_profile_event_ring_initialize(
      (uint8_t*)profile + host_offset,
      sizeof(iree_hal_profile_host_execution_event_t), host_capacity,
      &profile->host_events);
  iree_status_t status =
      iree_hal_profile_sink_begin_session(profile->sink, &profile->metadata);
  const bool begun = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    iree_hal_profile_device_record_t device =
        iree_hal_profile_device_record_default();
    device.physical_device_ordinal = physical_device_ordinal;
    device.queue_count = 1;
    status = iree_hal_amd_xdna_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES,
        iree_make_const_byte_span(&device, sizeof(device)));
  }
  if (iree_status_is_ok(status)) {
    iree_hal_profile_queue_record_t queue =
        iree_hal_profile_queue_record_default();
    queue.physical_device_ordinal = physical_device_ordinal;
    queue.queue_ordinal = queue_ordinal;
    queue.stream_id = profile->metadata.stream_id;
    status = iree_hal_amd_xdna_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES,
        iree_make_const_byte_span(&queue, sizeof(queue)));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_profile_counters_begin(
        options, &profile->metadata, context, command_arena, host_allocator,
        &profile->counters);
  }
  if (iree_status_is_ok(status)) {
    *out_profile = profile;
  } else {
    if (begun) {
      const iree_status_code_t code = iree_status_code(status);
      status = iree_status_join(
          status, iree_hal_profile_sink_end_session(profile->sink,
                                                    &profile->metadata, code));
    }
    iree_hal_amd_xdna_profile_free(profile);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_profile_flush_ring(
    iree_hal_amd_xdna_profile_t* profile, iree_string_view_t content_type,
    iree_hal_profile_event_ring_t* ring) {
  iree_hal_profile_event_ring_snapshot_t snapshot;
  iree_slim_mutex_lock(&profile->mutex);
  iree_status_t status = iree_hal_profile_event_ring_snapshot(ring, &snapshot);
  iree_slim_mutex_unlock(&profile->mutex);
  IREE_RETURN_IF_ERROR(status);
  if (!snapshot.record_count && !snapshot.dropped_record_count) {
    return iree_ok_status();
  }
  iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
  metadata.content_type = content_type;
  metadata.dropped_record_count = snapshot.dropped_record_count;
  if (snapshot.dropped_record_count) {
    metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  }
  IREE_RETURN_IF_ERROR(iree_hal_profile_sink_write(
      profile->sink, &metadata, snapshot.record_span_count,
      snapshot.record_span_count ? snapshot.record_spans : NULL));
  iree_slim_mutex_lock(&profile->mutex);
  iree_hal_profile_event_ring_commit_snapshot(ring, &snapshot);
  iree_slim_mutex_unlock(&profile->mutex);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_profile_flush(
    iree_hal_amd_xdna_profile_t* profile) {
  if (!profile) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_profile_flush_ring(
      profile, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
      &profile->queue_events));
  return iree_hal_amd_xdna_profile_flush_ring(
      profile, IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
      &profile->host_events);
}

iree_status_t iree_hal_amd_xdna_profile_end(
    iree_hal_amd_xdna_profile_t* profile) {
  if (!profile) {
    return iree_ok_status();
  }
  iree_status_t status = iree_hal_amd_xdna_profile_flush(profile);
  status = iree_status_join(
      status, iree_hal_amd_xdna_profile_counters_end(profile->counters));
  profile->counters = NULL;
  const iree_status_code_t code = iree_status_code(status);
  status =
      iree_status_join(status, iree_hal_profile_sink_end_session(
                                   profile->sink, &profile->metadata, code));
  iree_hal_amd_xdna_profile_free(profile);
  return status;
}

void iree_hal_amd_xdna_profile_initialize_event(
    iree_hal_amd_xdna_profile_t* profile,
    iree_hal_profile_queue_event_t* event) {
  *event = iree_hal_profile_queue_event_default();
  event->submission_id = iree_atomic_fetch_add(&profile->next_submission_id, 1,
                                               iree_memory_order_relaxed);
  event->host_time_ns = iree_time_now();
  event->physical_device_ordinal = profile->metadata.physical_device_ordinal;
  event->queue_ordinal = profile->metadata.queue_ordinal;
  event->stream_id = profile->metadata.stream_id;
}

void iree_hal_amd_xdna_profile_append_queue(
    iree_hal_amd_xdna_profile_t* profile,
    const iree_hal_profile_queue_event_t* event) {
  if (!profile->queue_events.capacity) {
    return;
  }
  uint64_t position = 0, id = 0;
  iree_slim_mutex_lock(&profile->mutex);
  if (iree_hal_profile_event_ring_try_append(&profile->queue_events, &position,
                                             &id)) {
    iree_hal_profile_queue_event_t* target =
        iree_hal_profile_event_ring_record_at(&profile->queue_events, position);
    *target = *event;
    target->event_id = id;
  }
  iree_slim_mutex_unlock(&profile->mutex);
}

iree_time_t iree_hal_amd_xdna_profile_host_begin(
    iree_hal_amd_xdna_profile_t* profile) {
  return profile && profile->host_events.capacity ? iree_time_now() : 0;
}

void iree_hal_amd_xdna_profile_host_end(
    iree_hal_amd_xdna_profile_t* profile,
    const iree_hal_profile_queue_event_t* queue_event, iree_time_t start,
    iree_hal_profile_host_execution_event_flags_t flags,
    iree_status_code_t status_code, uint32_t operation_count) {
  if (!start) {
    return;
  }
  iree_hal_profile_host_execution_event_t event =
      iree_hal_profile_host_execution_event_default();
  event.end_host_time_ns = iree_time_now();
  event.start_host_time_ns = start;
  event.type = queue_event->type;
  event.flags = flags;
  event.status_code = status_code;
  event.submission_id = queue_event->submission_id;
  event.command_buffer_id = queue_event->command_buffer_id;
  event.stream_id = queue_event->stream_id;
  event.physical_device_ordinal = queue_event->physical_device_ordinal;
  event.queue_ordinal = queue_event->queue_ordinal;
  event.operation_count = operation_count;
  event.payload_length = operation_count ? queue_event->payload_length : 0;
  uint64_t position = 0, id = 0;
  iree_slim_mutex_lock(&profile->mutex);
  if (iree_hal_profile_event_ring_try_append(&profile->host_events, &position,
                                             &id)) {
    event.event_id = id;
    *(iree_hal_profile_host_execution_event_t*)
        iree_hal_profile_event_ring_record_at(&profile->host_events, position) =
        event;
  }
  iree_slim_mutex_unlock(&profile->mutex);
}

iree_hal_amd_xdna_profile_counters_t* iree_hal_amd_xdna_profile_counters(
    iree_hal_amd_xdna_profile_t* profile) {
  return profile->counters;
}
