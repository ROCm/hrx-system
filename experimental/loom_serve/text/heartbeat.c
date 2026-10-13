// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/heartbeat.h"

#include <inttypes.h>
#include <string.h>

#include "loom/serve/runtime/parameter/telemetry.h"

// Related records share a timestamp but remain independently readable after
// overwrite. Shared device memory is labeled separately from model ownership.
typedef struct text_activity_record_t {
  // Host monotonic start of the current phase.
  uint64_t phase_start;
  // Host monotonic time of the last completed epoch.
  uint64_t last_completion;
  // Oldest queued arrival, or zero when the queue is empty.
  uint64_t oldest_queued_arrival;
  // Oldest active arrival, or zero when no row is active.
  uint64_t oldest_active_arrival;
  // Rows consuming prompt input.
  uint64_t prefill_rows;
  // Rows generating tokens.
  uint64_t decode_rows;
  // Rows waiting for peer output credit.
  uint64_t backpressured_rows;
  // Span count of the last issued epoch.
  uint64_t epoch_spans;
  // Token count of the last issued epoch.
  uint64_t epoch_tokens;
} text_activity_record_t;

typedef struct text_work_record_t {
  // Issued epochs including device continuations.
  uint64_t issued_epochs;
  // Successfully completed epochs.
  uint64_t completed_epochs;
  // Target weight traversals including isolated spans.
  uint64_t traversals;
  // Consumed prompt tokens excluding retained history.
  uint64_t prefill_tokens;
  // Consumed pending and accepted draft tokens.
  uint64_t decode_tokens;
  // Cumulative host submission-to-readiness nanoseconds.
  uint64_t model_ns;
  // Draft candidates from completed verification.
  uint64_t proposed_tokens;
  // Accepted draft inputs excluding pending anchors.
  uint64_t accepted_inputs;
} text_work_record_t;

typedef struct text_memory_record_t {
  // Addressable logical token positions.
  uint64_t capacity;
  // Completion credit disjoint from resident positions.
  uint64_t reserved;
  // Unique resident token positions.
  uint64_t resident;
  // Explicitly suspended endpoint payload.
  uint64_t checkpoint_host_bytes;
  // Detached idle endpoints, independent of active execution rows.
  struct {
    // Configured retained endpoint bound.
    uint64_t capacity;
    // Owned endpoints, including one possible pending admission exchange.
    uint64_t endpoints;
  } session_cache;
  // Exact token-prefix frontiers, independent of completed session histories.
  struct {
    // Configured frontier bound.
    uint64_t capacity;
    // Owned frontiers, including a pending capture.
    uint64_t endpoints;
  } prefix_cache;
  // Model mutable-state backing excluding weights/workspace.
  loom_serve_memory_statistics_t state;
  // Independently reclaimable model parameter backing.
  loom_serve_memory_statistics_t weights;
} text_memory_record_t;

typedef struct text_shared_memory_record_t {
  // Entire shared device owner's retained backing, not this model's charge.
  uint64_t retained_committed_bytes;
  // Physical workspace backing, including reusable idle ranges.
  uint64_t workspace_committed_bytes;
  // Workspace backing charged to live reservations.
  uint64_t workspace_live_bytes;
  // Shared workspace slab count.
  uint64_t workspace_slab_count;
  // Cumulative successful workspace reuse across all device users.
  uint64_t workspace_reuse_count;
} text_shared_memory_record_t;

static iree_status_t text_shared_memory_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  text_shared_memory_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"shared\":true,\"retained_committed_bytes\":%" PRIu64
      ",\"workspace_committed_bytes\":%" PRIu64
      ",\"workspace_live_bytes\":%" PRIu64 ",\"workspace_slab_count\":%" PRIu64
      ",\"workspace_reuse_count\":%" PRIu64,
      value.retained_committed_bytes, value.workspace_committed_bytes,
      value.workspace_live_bytes, value.workspace_slab_count,
      value.workspace_reuse_count);
}

static const loom_serve_telemetry_event_t text_shared_memory_event = {
    "shared_device_memory", text_shared_memory_format};

static iree_status_t text_activity_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  text_activity_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"phase_start_ns\":%" PRIu64 ",\"last_completion_ns\":%" PRIu64
      ",\"oldest_queued_arrival_ns\":%" PRIu64
      ",\"oldest_active_arrival_ns\":%" PRIu64 ",\"prefill_rows\":%" PRIu64
      ",\"decode_rows\":%" PRIu64 ",\"backpressured_rows\":%" PRIu64
      ",\"epoch_spans\":%" PRIu64 ",\"epoch_tokens\":%" PRIu64,
      value.phase_start, value.last_completion, value.oldest_queued_arrival,
      value.oldest_active_arrival, value.prefill_rows, value.decode_rows,
      value.backpressured_rows, value.epoch_spans, value.epoch_tokens);
}

static iree_status_t text_work_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  text_work_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"issued_epochs\":%" PRIu64 ",\"completed_epochs\":%" PRIu64
      ",\"traversals\":%" PRIu64 ",\"prefill_tokens\":%" PRIu64
      ",\"decode_tokens\":%" PRIu64 ",\"model_ns\":%" PRIu64
      ",\"draft\":{\"proposed_tokens\":%" PRIu64
      ",\"accepted_draft_inputs\":%" PRIu64 "}",
      value.issued_epochs, value.completed_epochs, value.traversals,
      value.prefill_tokens, value.decode_tokens, value.model_ns,
      value.proposed_tokens, value.accepted_inputs);
}

static iree_status_t text_memory_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  text_memory_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"pool\":{\"capacity_tokens\":%" PRIu64 ",\"reserved_tokens\":%" PRIu64
      ",\"resident_tokens\":%" PRIu64 "},\"checkpoint_host_bytes\":%" PRIu64
      ",\"session_cache\":{\"capacity\":%" PRIu64 ",\"endpoints\":%" PRIu64
      "}"
      ",\"prefix_cache\":{\"capacity\":%" PRIu64 ",\"endpoints\":%" PRIu64
      "}"
      ",\"elastic_state\":{\"reserved_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64 ",\"peak_bytes\":%" PRIu64
      ",\"released_bytes\":%" PRIu64
      "}"
      ",\"elastic_parameters\":{\"reserved_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64 ",\"peak_bytes\":%" PRIu64
      ",\"released_bytes\":%" PRIu64 "}",
      value.capacity, value.reserved, value.resident,
      value.checkpoint_host_bytes, value.session_cache.capacity,
      value.session_cache.endpoints, value.prefix_cache.capacity,
      value.prefix_cache.endpoints, value.state.reserved_bytes,
      value.state.committed_bytes, value.state.peak_bytes,
      value.state.released_bytes, value.weights.reserved_bytes,
      value.weights.committed_bytes, value.weights.peak_bytes,
      value.weights.released_bytes);
}

static const loom_serve_telemetry_event_t text_activity_event = {
    "text_activity", text_activity_format};
static const loom_serve_telemetry_event_t text_work_event = {"text_work",
                                                             text_work_format};
static const loom_serve_telemetry_event_t text_memory_event = {
    "text_memory", text_memory_format};

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
  loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .event = &LOOM_SERVE_TELEMETRY_SNAPSHOT,
      .data.snapshot = {.counters = *heartbeat->counters,
                        .queued = state.queued_requests,
                        .active = state.active_rows}};
  memcpy(record.data.snapshot.phase, state.phase,
         iree_min(strlen(state.phase), sizeof(record.data.snapshot.phase) - 1));
  loom_serve_telemetry_stream_append(heartbeat->stream, &record);
  const text_activity_record_t activity = {
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
      .epoch_tokens = state.epoch_tokens};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &text_activity_event;
  memcpy(record.data.words, &activity, sizeof(activity));
  loom_serve_telemetry_stream_append(heartbeat->stream, &record);
  const text_work_record_t work = {
      .issued_epochs = state.issued_epochs,
      .completed_epochs = state.completed_epochs,
      .traversals = state.traversals,
      .prefill_tokens = state.prefill_tokens,
      .decode_tokens = state.decode_tokens,
      .model_ns = state.model_duration,
      .proposed_tokens = state.draft.proposed_tokens,
      .accepted_inputs = state.draft.accepted_inputs};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &text_work_event;
  memcpy(record.data.words, &work, sizeof(work));
  loom_serve_telemetry_stream_append(heartbeat->stream, &record);
  const text_memory_record_t memory = {
      .capacity = state.pool.capacity,
      .reserved = state.pool.reserved,
      .resident = state.pool.resident,
      .checkpoint_host_bytes = state.checkpoint_host_bytes,
      .session_cache = {state.session_cache.capacity,
                        state.session_cache.endpoints},
      .prefix_cache = {state.prefix_cache.capacity,
                       state.prefix_cache.endpoints},
      .state = state.state_memory,
      .weights = state.weight_memory};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &text_memory_event;
  memcpy(record.data.words, &memory, sizeof(memory));
  loom_serve_telemetry_stream_append(heartbeat->stream, &record);
  loom_serve_telemetry_record_weights(heartbeat->stream, record.time_ns,
                                      &state.weight_activity);
  const text_shared_memory_record_t shared_memory = {
      .retained_committed_bytes = state.device_memory.retained.committed_bytes,
      .workspace_committed_bytes =
          state.device_memory.workspace.bytes_committed,
      .workspace_live_bytes = state.device_memory.workspace.bytes_reserved,
      .workspace_slab_count = state.device_memory.workspace.slab_count,
      .workspace_reuse_count = state.device_memory.workspace.reuse_count};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &text_shared_memory_event;
  memcpy(record.data.words, &shared_memory, sizeof(shared_memory));
  loom_serve_telemetry_stream_append(heartbeat->stream, &record);
}

void loom_serve_text_heartbeat_initialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  *heartbeat = (loom_serve_text_heartbeat_t){0};
  iree_slim_mutex_initialize(&heartbeat->mutex);
  loom_serve_timer_initialize(&heartbeat->timer);
  heartbeat->snapshot.phase = "starting";
  heartbeat->snapshot.phase_start = iree_time_now();
  heartbeat->snapshot.last_completion = heartbeat->snapshot.phase_start;
}

void loom_serve_text_heartbeat_configure(
    loom_serve_text_heartbeat_t* heartbeat, iree_async_proactor_t* proactor,
    loom_serve_telemetry_stream_t* stream,
    const loom_serve_telemetry_counters_t* counters, iree_duration_t interval) {
  heartbeat->stream = stream;
  heartbeat->counters = counters;
  loom_serve_timer_configure(&heartbeat->timer, proactor, interval);
}

bool loom_serve_text_heartbeat_is_pending(
    loom_serve_text_heartbeat_t* heartbeat) {
  return loom_serve_timer_is_pending(&heartbeat->timer);
}

iree_status_t loom_serve_text_heartbeat_advance(
    loom_serve_text_heartbeat_t* heartbeat, bool stopping, bool* out_progress) {
  bool tick = false;
  iree_status_t status = loom_serve_timer_advance(&heartbeat->timer, stopping,
                                                  out_progress, &tick);
  if (tick) {
    text_heartbeat_report(heartbeat);
  }
  return status;
}

void loom_serve_text_heartbeat_deinitialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  loom_serve_timer_deinitialize(&heartbeat->timer);
  if (heartbeat->stream) {
    text_heartbeat_report(heartbeat);
  }
  iree_slim_mutex_deinitialize(&heartbeat->mutex);
}
