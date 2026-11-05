// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/heartbeat.h"

#include <string.h>

#include "loom/runtime/parameter/telemetry.h"
#include "loom/runtime/storage/telemetry.h"
#include "loom/service/telemetry/record.h"
#include "loom/service/text/telemetry.h"

iree_status_t loom_serve_text_heartbeat_sample(
    loom_serve_text_heartbeat_t* heartbeat,
    loom_runtime_telemetry_sample_sink_t sink) {
  const loom_service_text_statistics_t state =
      loom_serve_text_heartbeat_read(heartbeat);
  return loom_service_text_sample_statistics(&state, sink);
}

loom_service_text_statistics_t loom_serve_text_heartbeat_read(
    loom_serve_text_heartbeat_t* heartbeat) {
  iree_slim_mutex_lock(&heartbeat->mutex);
  const loom_service_text_statistics_t state = heartbeat->snapshot;
  iree_slim_mutex_unlock(&heartbeat->mutex);
  return state;
}

static void loom_serve_text_heartbeat_report(
    loom_serve_text_heartbeat_t* heartbeat) {
  const loom_service_text_statistics_t state =
      loom_serve_text_heartbeat_read(heartbeat);
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(), .event = &LOOM_SERVICE_TELEMETRY_SNAPSHOT};
  loom_service_telemetry_snapshot_record_t telemetry_payload = {
      .counters = *heartbeat->counters,
      .queued = state.queued_requests,
      .active = state.active_rows};
  memcpy(telemetry_payload.phase, state.phase,
         iree_min(strlen(state.phase), sizeof(telemetry_payload.phase) - 1));
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(heartbeat->stream, &record);
  const loom_service_text_activity_record_t activity = {
      .phase_start = state.phase_start,
      .last_completion = state.last_completion,
      .oldest_queued_arrival =
          state.queued_requests ? state.oldest_queued_arrival : 0,
      .oldest_active_arrival =
          state.active_rows ? state.oldest_active_arrival : 0,
      .prefill_rows = state.prefill_rows,
      .decode_rows = state.decode_rows,
      .backpressured_rows = state.backpressured_rows,
      .epoch_spans = state.epoch_spans,
      .epoch_tokens = state.epoch_tokens,
      .session_capacity = state.session_capacity,
      .max_sessions = state.max_sessions};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &LOOM_SERVICE_TEXT_ACTIVITY;
  memcpy(record.data.words, &activity, sizeof(activity));
  loom_runtime_telemetry_stream_append(heartbeat->stream, &record);
  const loom_service_text_work_record_t work = {
      .issued_epochs = state.issued_epochs,
      .completed_epochs = state.completed_epochs,
      .traversals = state.traversals,
      .prefill_tokens = state.prefill_tokens,
      .decode_tokens = state.decode_tokens,
      .model_ns = state.model_duration,
      .proposed_tokens = state.draft.proposed_tokens,
      .accepted_inputs = state.draft.accepted_inputs};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &LOOM_SERVICE_TEXT_WORK;
  memcpy(record.data.words, &work, sizeof(work));
  loom_runtime_telemetry_stream_append(heartbeat->stream, &record);
  const loom_service_text_memory_record_t memory = {
      .capacity = state.pool.capacity,
      .reserved = state.pool.reserved,
      .resident = state.pool.resident,
      .checkpoint_host_bytes = state.checkpoint_host_bytes,
      .cache_target_bytes = state.cache_target_bytes,
      .cache_additional_bytes = state.cache_additional_bytes,
      .session_cache = {state.session_cache.capacity,
                        state.session_cache.endpoints},
      .prefix_cache = {state.prefix_cache.capacity,
                       state.prefix_cache.endpoints}};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &LOOM_SERVICE_TEXT_MEMORY;
  memcpy(record.data.words, &memory, sizeof(memory));
  loom_runtime_telemetry_stream_append(heartbeat->stream, &record);
  const loom_service_text_backing_record_t backing = {
      .state = state.state_memory, .parameters = state.parameter_memory};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &LOOM_SERVICE_TEXT_BACKING;
  memcpy(record.data.words, &backing, sizeof(backing));
  loom_runtime_telemetry_stream_append(heartbeat->stream, &record);
  loom_runtime_parameter_record_activity(heartbeat->stream, record.time_ns,
                                         &state.parameter_activity);
  loom_runtime_storage_record_shared_memory(heartbeat->stream, record.time_ns,
                                            &state.pool_memory);
}

void loom_serve_text_heartbeat_initialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  *heartbeat = (loom_serve_text_heartbeat_t){0};
  iree_slim_mutex_initialize(&heartbeat->mutex);
  loom_runtime_timer_initialize(&heartbeat->timer);
  heartbeat->snapshot.phase = "starting";
  heartbeat->snapshot.phase_start = iree_time_now();
  heartbeat->snapshot.last_completion = heartbeat->snapshot.phase_start;
}

void loom_serve_text_heartbeat_configure(
    loom_serve_text_heartbeat_t* heartbeat, iree_async_proactor_t* proactor,
    loom_runtime_telemetry_stream_t* stream,
    const loom_service_telemetry_counters_t* counters,
    iree_duration_t interval) {
  heartbeat->stream = stream;
  heartbeat->counters = counters;
  loom_runtime_timer_configure(&heartbeat->timer, proactor, interval);
}

bool loom_serve_text_heartbeat_is_pending(
    loom_serve_text_heartbeat_t* heartbeat) {
  return loom_runtime_timer_is_pending(&heartbeat->timer);
}

iree_status_t loom_serve_text_heartbeat_advance(
    loom_serve_text_heartbeat_t* heartbeat, bool stopping, bool* out_progress) {
  bool tick = false;
  iree_status_t status = loom_runtime_timer_advance(&heartbeat->timer, stopping,
                                                    true, out_progress, &tick);
  if (tick) {
    loom_serve_text_heartbeat_report(heartbeat);
  }
  return status;
}

void loom_serve_text_heartbeat_deinitialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  loom_runtime_timer_deinitialize(&heartbeat->timer);
  if (heartbeat->stream) {
    loom_serve_text_heartbeat_report(heartbeat);
  }
  iree_slim_mutex_deinitialize(&heartbeat->mutex);
}
