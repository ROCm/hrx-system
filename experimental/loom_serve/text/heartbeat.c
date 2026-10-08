// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/heartbeat.h"

#include <stdio.h>

static int text_heartbeat_main(void* argument) {
  loom_serve_text_heartbeat_t* heartbeat = argument;
  iree_time_t previous_time = iree_time_now();
  uint64_t previous_prefill = 0;
  uint64_t previous_output = 0;
  for (;;) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(&heartbeat->notification);
    iree_slim_mutex_lock(&heartbeat->mutex);
    const bool stopping = heartbeat->stopping;
    const loom_serve_text_heartbeat_snapshot_t state = heartbeat->snapshot;
    iree_slim_mutex_unlock(&heartbeat->mutex);
    const iree_time_t now = iree_time_now();
    const double seconds = (now - previous_time) / 1e9;
    fprintf(
        stderr,
        "{\"model\":%.*s,\"event\":\"heartbeat\",\"phase\":\"%s\",\"phase_ms\":"
        "%.3f,"
        "\"since_completion_ms\":%.3f,\"active_rows\":%zu,"
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
        state.active_rows, state.queued_requests, state.pool.capacity,
        state.pool.reserved, state.pool.resident, state.checkpoint_host_bytes,
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
        seconds > 0 ? (state.prefill_tokens - previous_prefill) / seconds : 0,
        seconds > 0 ? (state.output_tokens - previous_output) / seconds : 0);
    previous_time = now;
    previous_prefill = state.prefill_tokens;
    previous_output = state.output_tokens;
    if (stopping) {
      iree_notification_cancel_wait(&heartbeat->notification);
      break;
    }
    iree_notification_commit_wait(&heartbeat->notification, token,
                                  IREE_DURATION_ZERO,
                                  now + heartbeat->interval);
  }
  return 0;
}

void loom_serve_text_heartbeat_initialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  *heartbeat = (loom_serve_text_heartbeat_t){0};
  iree_slim_mutex_initialize(&heartbeat->mutex);
  iree_notification_initialize(&heartbeat->notification);
  heartbeat->snapshot.phase = "starting";
  heartbeat->snapshot.phase_start = iree_time_now();
  heartbeat->snapshot.last_completion = heartbeat->snapshot.phase_start;
}

iree_status_t loom_serve_text_heartbeat_start(
    loom_serve_text_heartbeat_t* heartbeat, iree_string_view_t name_json,
    iree_duration_t interval, iree_allocator_t host_allocator) {
  heartbeat->name_json = name_json;
  heartbeat->interval = interval;
  if (!interval) {
    return iree_ok_status();
  }
  const iree_thread_create_params_t parameters = {
      .name = IREE_SV("text-heartbeat"),
  };
  return iree_thread_create(text_heartbeat_main, heartbeat, parameters,
                            host_allocator, &heartbeat->thread);
}

void loom_serve_text_heartbeat_deinitialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  iree_slim_mutex_lock(&heartbeat->mutex);
  heartbeat->stopping = true;
  iree_slim_mutex_unlock(&heartbeat->mutex);
  iree_notification_post(&heartbeat->notification, IREE_ALL_WAITERS);
  if (heartbeat->thread) {
    // Releasing the sole thread reference joins its final snapshot access.
    iree_thread_release(heartbeat->thread);
  }
  iree_notification_deinitialize(&heartbeat->notification);
  iree_slim_mutex_deinitialize(&heartbeat->mutex);
}
