// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/heartbeat.h"

#include <stdio.h>
#include <string.h>

enum {
  TEXT_HEARTBEAT_TIMER = 1u << 0,
  TEXT_HEARTBEAT_STOP = 1u << 1,
  TEXT_HEARTBEAT_CANCEL = 1u << 2,
};

static void text_heartbeat_report(loom_serve_text_heartbeat_t* heartbeat) {
  iree_slim_mutex_lock(&heartbeat->mutex);
  const loom_serve_text_heartbeat_snapshot_t state = heartbeat->snapshot;
  iree_slim_mutex_unlock(&heartbeat->mutex);
  const iree_time_t now = iree_time_now();
  const double seconds = (now - heartbeat->previous.time) / 1e9;
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
      seconds > 0
          ? (state.prefill_tokens - heartbeat->previous.prefill) / seconds
          : 0,
      seconds > 0 ? (state.output_tokens - heartbeat->previous.output) / seconds
                  : 0);
  heartbeat->previous.time = now;
  heartbeat->previous.prefill = state.prefill_tokens;
  heartbeat->previous.output = state.output_tokens;
}

static void text_heartbeat_cancel_retired(void* user_data) {
  loom_serve_text_heartbeat_t* heartbeat = user_data;
  iree_slim_mutex_lock(&heartbeat->mutex);
  heartbeat->events.pending &= ~TEXT_HEARTBEAT_CANCEL;
  heartbeat->events.ready = true;
  heartbeat->observer.wake(heartbeat->observer.user_data);
  iree_slim_mutex_unlock(&heartbeat->mutex);
}

static void text_heartbeat_timer_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_text_heartbeat_t* heartbeat = user_data;
  iree_slim_mutex_lock(&heartbeat->mutex);
  heartbeat->events.status = iree_status_join(heartbeat->events.status, status);
  heartbeat->events.pending &= ~TEXT_HEARTBEAT_TIMER;
  heartbeat->events.ready = true;
  const bool cancel_pending =
      iree_any_bit_set(heartbeat->events.pending, TEXT_HEARTBEAT_CANCEL);
  if (!cancel_pending) {
    heartbeat->observer.wake(heartbeat->observer.user_data);
  }
  iree_slim_mutex_unlock(&heartbeat->mutex);
  if (cancel_pending) {
    // The receipt may publish the last borrowed access. No access follows it.
    iree_async_proactor_cancel_request_target_retired(
        heartbeat->proactor, &heartbeat->timer.cancellation);
  }
}

static void text_heartbeat_stop_received(void* user_data,
                                         iree_async_operation_t* operation,
                                         iree_status_t status,
                                         iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_text_heartbeat_t* heartbeat = user_data;
  iree_slim_mutex_lock(&heartbeat->mutex);
  const bool cancel =
      iree_status_is_ok(status) &&
      iree_any_bit_set(heartbeat->events.pending, TEXT_HEARTBEAT_TIMER);
  heartbeat->events.status = iree_status_join(heartbeat->events.status, status);
  if (cancel) {
    heartbeat->events.pending |= TEXT_HEARTBEAT_CANCEL;
  }
  iree_slim_mutex_unlock(&heartbeat->mutex);
  if (cancel) {
    // This poll-owner callback owns a private live timer and an idle receipt.
    IREE_CHECK_OK(iree_async_proactor_request_cancel(
        heartbeat->proactor, &heartbeat->timer.operation.base,
        &heartbeat->timer.cancellation));
  }
  iree_slim_mutex_lock(&heartbeat->mutex);
  heartbeat->events.pending &= ~TEXT_HEARTBEAT_STOP;
  heartbeat->events.ready = true;
  heartbeat->observer.wake(heartbeat->observer.user_data);
  iree_slim_mutex_unlock(&heartbeat->mutex);
}

void loom_serve_text_heartbeat_initialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  *heartbeat = (loom_serve_text_heartbeat_t){0};
  iree_slim_mutex_initialize(&heartbeat->mutex);
  iree_async_cancel_request_initialize(
      (iree_async_cancel_callback_t){text_heartbeat_cancel_retired, heartbeat},
      &heartbeat->timer.cancellation);
  heartbeat->snapshot.phase = "starting";
  heartbeat->snapshot.phase_start = iree_time_now();
  heartbeat->snapshot.last_completion = heartbeat->snapshot.phase_start;
  heartbeat->previous.time = heartbeat->snapshot.phase_start;
}

void loom_serve_text_heartbeat_configure(loom_serve_text_heartbeat_t* heartbeat,
                                         iree_async_proactor_t* proactor,
                                         iree_string_view_t name_json,
                                         iree_duration_t interval) {
  heartbeat->proactor = proactor;
  heartbeat->name_json = name_json;
  heartbeat->interval = interval;
}

bool loom_serve_text_heartbeat_is_pending(
    loom_serve_text_heartbeat_t* heartbeat) {
  iree_slim_mutex_lock(&heartbeat->mutex);
  const bool pending = heartbeat->events.pending || heartbeat->events.ready;
  iree_slim_mutex_unlock(&heartbeat->mutex);
  return pending;
}

iree_status_t loom_serve_text_heartbeat_advance(
    loom_serve_text_heartbeat_t* heartbeat, bool stopping, bool* out_progress) {
  iree_slim_mutex_lock(&heartbeat->mutex);
  iree_status_t status = heartbeat->events.status;
  heartbeat->events.status = iree_ok_status();
  *out_progress = heartbeat->events.ready;
  heartbeat->events.ready = false;
  const uint32_t pending = heartbeat->events.pending;
  iree_slim_mutex_unlock(&heartbeat->mutex);
  stopping |= !iree_status_is_ok(status);
  if (stopping && !heartbeat->stopping) {
    heartbeat->stopping = true;
    *out_progress = true;
    if (iree_any_bit_set(pending, TEXT_HEARTBEAT_TIMER)) {
      iree_async_nop_operation_t* stop = &heartbeat->timer.stop;
      memset(stop, 0, sizeof(*stop));
      iree_async_operation_initialize(&stop->base,
                                      IREE_ASYNC_OPERATION_TYPE_NOP,
                                      IREE_ASYNC_OPERATION_FLAG_NONE,
                                      text_heartbeat_stop_received, heartbeat);
      iree_slim_mutex_lock(&heartbeat->mutex);
      heartbeat->events.pending |= TEXT_HEARTBEAT_STOP;
      iree_slim_mutex_unlock(&heartbeat->mutex);
      iree_status_t stop_status =
          iree_async_proactor_submit_one(heartbeat->proactor, &stop->base);
      if (!iree_status_is_ok(stop_status)) {
        iree_slim_mutex_lock(&heartbeat->mutex);
        heartbeat->events.pending &= ~TEXT_HEARTBEAT_STOP;
        iree_slim_mutex_unlock(&heartbeat->mutex);
      }
      status = iree_status_join(status, stop_status);
    }
  }
  if (!heartbeat->stopping && !pending && heartbeat->interval) {
    text_heartbeat_report(heartbeat);
    *out_progress = true;
    iree_async_timer_operation_t* timer = &heartbeat->timer.operation;
    memset(timer, 0, sizeof(*timer));
    iree_async_operation_initialize(
        &timer->base, IREE_ASYNC_OPERATION_TYPE_TIMER,
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        text_heartbeat_timer_completed, heartbeat);
    timer->deadline_ns = iree_time_now() + heartbeat->interval;
    iree_slim_mutex_lock(&heartbeat->mutex);
    heartbeat->events.pending |= TEXT_HEARTBEAT_TIMER;
    iree_slim_mutex_unlock(&heartbeat->mutex);
    status = iree_async_proactor_submit_one(heartbeat->proactor, &timer->base);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&heartbeat->mutex);
      heartbeat->events.pending &= ~TEXT_HEARTBEAT_TIMER;
      iree_slim_mutex_unlock(&heartbeat->mutex);
      heartbeat->stopping = true;
    }
  }
  return status;
}

void loom_serve_text_heartbeat_deinitialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  IREE_ASSERT(!loom_serve_text_heartbeat_is_pending(heartbeat));
  if (heartbeat->interval) {
    text_heartbeat_report(heartbeat);
  }
  iree_slim_mutex_deinitialize(&heartbeat->mutex);
}
