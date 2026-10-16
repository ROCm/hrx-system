// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/profile.h"

#include <inttypes.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/utils/profile_event_ring.h"

// Each queue retains this many public operations between explicit flushes.
#define IREE_HAL_AMD_GPU_PROFILE_CAPACITY 4096

typedef struct iree_hal_amd_gpu_profile_stream_t {
  // Protects ring publication and flush commits, never a sink callback.
  iree_slim_mutex_t mutex;
  // Queue-local acceptance identity, independent of native chunk epochs.
  iree_atomic_int64_t next_submission_id;
  // Bounded records stored in the session's trailing allocation.
  iree_hal_profile_event_ring_t events;
} iree_hal_amd_gpu_profile_stream_t;

struct iree_hal_amd_gpu_profile_t {
  // Allocator owning this session and all inline stream storage.
  iree_allocator_t host_allocator;
  // Retained output sink; only explicit lifecycle calls may invoke it.
  iree_hal_profile_sink_t* sink;
  // Immutable session metadata shared by every chunk.
  iree_hal_profile_chunk_metadata_t metadata;
  // Number of provisioned queues and initialized stream mutexes.
  uint32_t queue_count;
  // Inline queue streams, indexed directly by physical queue ordinal.
  iree_hal_amd_gpu_profile_stream_t* streams;
};

static void iree_hal_amd_gpu_profile_free(iree_hal_amd_gpu_profile_t* profile) {
  for (uint32_t i = 0; i < profile->queue_count; ++i) {
    iree_slim_mutex_deinitialize(&profile->streams[i].mutex);
  }
  iree_hal_profile_sink_release(profile->sink);
  iree_allocator_free(profile->host_allocator, profile);
}

static iree_status_t iree_hal_amd_gpu_profile_write(
    iree_hal_amd_gpu_profile_t* profile, iree_string_view_t content_type,
    uint32_t queue_ordinal, iree_const_byte_span_t bytes) {
  iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
  metadata.content_type = content_type;
  metadata.queue_ordinal = queue_ordinal;
  metadata.stream_id = queue_ordinal == UINT32_MAX ? 0 : queue_ordinal + 1ull;
  return iree_hal_profile_sink_write(profile->sink, &metadata, 1, &bytes);
}

iree_status_t iree_hal_amd_gpu_profile_begin(
    const iree_hal_device_profiling_options_t* options, uint64_t session_id,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_profile_t** out_profile) {
  *out_profile = NULL;
  iree_hal_device_profiling_data_families_t families = options->data_families;
  if (!families &&
      iree_hal_device_profiling_options_requests_lightweight_statistics(
          options)) {
    families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
  }
  if (!families) {
    return iree_ok_status();
  }
  if (families & ~IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported GPU profiling data families 0x%" PRIx64,
        families & ~IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS);
  }
  if (!iree_hal_profile_capture_filter_is_default(&options->capture_filter) ||
      options->counter_set_count) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "GPU queue profiling does not support filters or counters");
  }
  iree_host_size_t record_count = 0;
  if (!iree_host_size_checked_mul(
          queue_count, IREE_HAL_AMD_GPU_PROFILE_CAPACITY, &record_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU profiling stream capacity overflows");
  }
  iree_host_size_t size = 0, streams_offset = 0, records_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_profile_t), &size,
      IREE_STRUCT_FIELD(queue_count, iree_hal_amd_gpu_profile_stream_t,
                        &streams_offset),
      IREE_STRUCT_FIELD(record_count, iree_hal_profile_queue_event_t,
                        &records_offset)));
  iree_hal_amd_gpu_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&profile));
  profile->host_allocator = host_allocator;
  profile->sink = options->sink;
  iree_hal_profile_sink_retain(profile->sink);
  profile->queue_count = queue_count;
  profile->streams = (void*)((uint8_t*)profile + streams_offset);
  profile->metadata = iree_hal_profile_chunk_metadata_default();
  profile->metadata.name = IREE_SV("amd-gpu");
  profile->metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  profile->metadata.session_id = session_id;
  profile->metadata.physical_device_ordinal = 0;
  iree_hal_profile_queue_event_t* records =
      (void*)((uint8_t*)profile + records_offset);
  for (uint32_t i = 0; i < queue_count; ++i) {
    iree_hal_amd_gpu_profile_stream_t* stream = &profile->streams[i];
    iree_slim_mutex_initialize(&stream->mutex);
    iree_atomic_store(&stream->next_submission_id, 1,
                      iree_memory_order_relaxed);
    iree_hal_profile_event_ring_initialize(
        records + (iree_host_size_t)i * IREE_HAL_AMD_GPU_PROFILE_CAPACITY,
        sizeof(*records), IREE_HAL_AMD_GPU_PROFILE_CAPACITY, &stream->events);
  }
  iree_status_t status =
      iree_hal_profile_sink_begin_session(profile->sink, &profile->metadata);
  const bool begun = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    iree_hal_profile_device_record_t record =
        iree_hal_profile_device_record_default();
    record.physical_device_ordinal = 0;
    record.queue_count = queue_count;
    status = iree_hal_amd_gpu_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES, UINT32_MAX,
        iree_make_const_byte_span(&record, sizeof(record)));
  }
  for (uint32_t i = 0; i < queue_count && iree_status_is_ok(status); ++i) {
    iree_hal_profile_queue_record_t record =
        iree_hal_profile_queue_record_default();
    record.physical_device_ordinal = 0;
    record.queue_ordinal = i;
    record.stream_id = i + 1ull;
    status = iree_hal_amd_gpu_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES, i,
        iree_make_const_byte_span(&record, sizeof(record)));
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
    iree_hal_amd_gpu_profile_free(profile);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_profile_flush(
    iree_hal_amd_gpu_profile_t* profile) {
  if (!profile) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < profile->queue_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amd_gpu_profile_stream_t* stream = &profile->streams[i];
    iree_hal_profile_event_ring_snapshot_t snapshot;
    iree_slim_mutex_lock(&stream->mutex);
    status = iree_hal_profile_event_ring_snapshot(&stream->events, &snapshot);
    iree_slim_mutex_unlock(&stream->mutex);
    if (!iree_status_is_ok(status) ||
        (!snapshot.record_count && !snapshot.dropped_record_count)) {
      continue;
    }
    iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
    metadata.queue_ordinal = i;
    metadata.stream_id = i + 1ull;
    metadata.dropped_record_count = snapshot.dropped_record_count;
    if (snapshot.dropped_record_count) {
      metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
    }
    status = iree_hal_profile_sink_write(
        profile->sink, &metadata, snapshot.record_span_count,
        snapshot.record_span_count ? snapshot.record_spans : NULL);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&stream->mutex);
      iree_hal_profile_event_ring_commit_snapshot(&stream->events, &snapshot);
      iree_slim_mutex_unlock(&stream->mutex);
    }
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_profile_end(
    iree_hal_amd_gpu_profile_t* profile) {
  if (!profile) {
    return iree_ok_status();
  }
  iree_status_t status = iree_hal_amd_gpu_profile_flush(profile);
  const iree_status_code_t code = iree_status_code(status);
  status =
      iree_status_join(status, iree_hal_profile_sink_end_session(
                                   profile->sink, &profile->metadata, code));
  iree_hal_amd_gpu_profile_free(profile);
  return status;
}

void iree_hal_amd_gpu_profile_operation_initialize(
    iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal,
    iree_hal_profile_queue_event_type_t type, iree_host_size_t wait_count,
    iree_host_size_t signal_count, iree_host_size_t operation_count,
    iree_hal_amd_gpu_profile_operation_t* operation) {
  *operation = (iree_hal_amd_gpu_profile_operation_t){
      .event = iree_hal_profile_queue_event_default()};
  iree_hal_profile_queue_event_t* event = &operation->event;
  event->type = type;
  event->host_time_ns = iree_time_now();
  event->submission_id =
      iree_atomic_fetch_add(&profile->streams[queue_ordinal].next_submission_id,
                            1, iree_memory_order_relaxed);
  event->physical_device_ordinal = 0;
  event->queue_ordinal = queue_ordinal;
  event->stream_id = queue_ordinal + 1ull;
  event->wait_count = (uint32_t)iree_min(wait_count, UINT32_MAX);
  event->signal_count = (uint32_t)iree_min(signal_count, UINT32_MAX);
  event->operation_count = (uint32_t)iree_min(operation_count, UINT32_MAX);
  event->dependency_strategy =
      wait_count ? IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE
                 : IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE;
}

void iree_hal_amd_gpu_profile_operation_record(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation, bool ready,
    uint32_t native_wait_count) {
  if (!operation || operation->recorded) {
    return;
  }
  operation->recorded = true;
  iree_hal_profile_queue_event_t* event = &operation->event;
  event->ready_host_time_ns = ready ? iree_time_now() : 0;
  event->barrier_count = native_wait_count;
  if (native_wait_count &&
      !iree_any_bit_set(event->flags,
                        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED)) {
    event->dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER;
  }
  iree_hal_amd_gpu_profile_stream_t* stream =
      &profile->streams[event->queue_ordinal];
  uint64_t position = 0, event_id = 0;
  iree_slim_mutex_lock(&stream->mutex);
  if (iree_hal_profile_event_ring_try_append(&stream->events, &position,
                                             &event_id)) {
    event->event_id = event_id;
    *(iree_hal_profile_queue_event_t*)iree_hal_profile_event_ring_record_at(
        &stream->events, position) = *event;
  }
  iree_slim_mutex_unlock(&stream->mutex);
}
