// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/heartbeat.h"

#include <stdio.h>
#include <string.h>

loom_serve_text_heartbeat_snapshot_t loom_serve_text_heartbeat_read(
    loom_serve_text_heartbeat_t* heartbeat) {
  iree_slim_mutex_lock(&heartbeat->mutex);
  const loom_serve_text_heartbeat_snapshot_t state = heartbeat->snapshot;
  iree_slim_mutex_unlock(&heartbeat->mutex);
  return state;
}

static void text_heartbeat_report(loom_serve_text_heartbeat_t* heartbeat) {
  const loom_serve_text_heartbeat_snapshot_t state =
      loom_serve_text_heartbeat_read(heartbeat);
  const iree_time_t now = iree_time_now();
  const double seconds = (now - heartbeat->previous.time) / 1e9;
  fprintf(
      stderr,
      "{\"model\":%.*s,\"event\":\"heartbeat\",\"phase\":\"%s\",\"phase_ms\":"
      "%.3f,"
      "\"since_completion_ms\":%.3f,\"active_rows\":%zu,"
      "\"oldest_queued_age_seconds\":%.9f,"
      "\"oldest_active_age_seconds\":%.9f,"
      "\"queued_requests\":%zu,\"pool\":{\"capacity_tokens\":%zu,"
      "\"reserved_tokens\":%zu,\"resident_tokens\":%zu},"
      "\"checkpoint_host_bytes\":%zu,"
      "\"elastic_state\":{\"reserved_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64 ",\"peak_bytes\":%" PRIu64
      ",\"released_bytes\":%" PRIu64
      "},"
      "\"elastic_parameters\":{\"reserved_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64 ",\"peak_bytes\":%" PRIu64
      ",\"released_bytes\":%" PRIu64
      "},"
      "\"device_memory\":{\"retained_committed_bytes\":%" PRIu64
      ",\"workspace_committed_bytes\":%" PRIu64
      ",\"workspace_live_bytes\":%" PRIu64
      ",\"workspace_slab_count\":%u,\"workspace_reuse_count\":%" PRIu64
      "},"
      "\"prefill_rows\":%zu,\"decode_rows\":%zu,"
      "\"backpressured_rows\":%zu,\"issued_epochs\":%" PRIu64
      ",\"completed_epochs\":%" PRIu64 ",\"traversals\":%" PRIu64
      ",\"prefill_tokens\":%" PRIu64 ",\"decode_tokens\":%" PRIu64
      ",\"output_tokens_including_eos\":%" PRIu64
      ",\"model_ms\":%.3f,\"epoch_spans\":%zu,\"epoch_tokens\":%zu,"
      "\"mtp\":{\"proposed_tokens\":%" PRIu64
      ",\"accepted_draft_inputs\":%" PRIu64
      "},"
      "\"interval_prefill_tokens_per_second\":%.3f,"
      "\"interval_output_tokens_per_second\":%.3f}\n",
      (int)heartbeat->name_json.size, heartbeat->name_json.data, state.phase,
      (now - state.phase_start) / 1e6, (now - state.last_completion) / 1e6,
      state.active_rows,
      state.queued_requests ? (now - state.oldest_queued_arrival) / 1e9 : 0.0,
      state.active_rows ? (now - state.oldest_active_arrival) / 1e9 : 0.0,
      state.queued_requests, state.pool.capacity, state.pool.reserved,
      state.pool.resident, state.checkpoint_host_bytes,
      state.state_memory.reserved_bytes, state.state_memory.committed_bytes,
      state.state_memory.peak_bytes, state.state_memory.released_bytes,
      state.weight_memory.reserved_bytes, state.weight_memory.committed_bytes,
      state.weight_memory.peak_bytes, state.weight_memory.released_bytes,
      state.device_memory.retained.committed_bytes,
      (uint64_t)state.device_memory.workspace.bytes_committed,
      (uint64_t)state.device_memory.workspace.bytes_reserved,
      state.device_memory.workspace.slab_count,
      state.device_memory.workspace.reuse_count, state.prefill_rows,
      state.decode_rows, state.backpressured_rows, state.issued_epochs,
      state.completed_epochs, state.traversals, state.prefill_tokens,
      state.decode_tokens, state.output_tokens, state.model_duration / 1e6,
      state.epoch_spans, state.epoch_tokens, state.mtp.proposed_tokens,
      state.mtp.accepted_inputs,
      seconds > 0
          ? (state.prefill_tokens - heartbeat->previous.prefill) / seconds
          : 0,
      seconds > 0 ? (state.output_tokens - heartbeat->previous.output) / seconds
                  : 0);
  heartbeat->previous.time = now;
  heartbeat->previous.prefill = state.prefill_tokens;
  heartbeat->previous.output = state.output_tokens;
}

void loom_serve_text_heartbeat_initialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  *heartbeat = (loom_serve_text_heartbeat_t){0};
  iree_slim_mutex_initialize(&heartbeat->mutex);
  loom_serve_http_timer_initialize(&heartbeat->timer);
  heartbeat->snapshot.phase = "starting";
  heartbeat->snapshot.phase_start = iree_time_now();
  heartbeat->snapshot.last_completion = heartbeat->snapshot.phase_start;
  heartbeat->previous.time = heartbeat->snapshot.phase_start;
}

void loom_serve_text_heartbeat_configure(loom_serve_text_heartbeat_t* heartbeat,
                                         iree_async_proactor_t* proactor,
                                         iree_string_view_t name_json,
                                         iree_duration_t interval) {
  heartbeat->name_json = name_json;
  loom_serve_http_timer_configure(&heartbeat->timer, proactor, interval);
}

bool loom_serve_text_heartbeat_is_pending(
    loom_serve_text_heartbeat_t* heartbeat) {
  return loom_serve_http_timer_is_pending(&heartbeat->timer);
}

iree_status_t loom_serve_text_heartbeat_advance(
    loom_serve_text_heartbeat_t* heartbeat, bool stopping, bool* out_progress) {
  bool tick = false;
  iree_status_t status = loom_serve_http_timer_advance(
      &heartbeat->timer, stopping, out_progress, &tick);
  if (tick) {
    text_heartbeat_report(heartbeat);
  }
  return status;
}

void loom_serve_text_heartbeat_deinitialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  loom_serve_http_timer_deinitialize(&heartbeat->timer);
  if (heartbeat->timer.interval) {
    text_heartbeat_report(heartbeat);
  }
  iree_slim_mutex_deinitialize(&heartbeat->mutex);
}
