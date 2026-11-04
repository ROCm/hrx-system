// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/heartbeat.h"

#include <string.h>

#include "loom/runtime/telemetry/field.h"
#include "loom/serve/telemetry/weights.h"

iree_status_t loom_serve_text_heartbeat_sample(
    loom_serve_text_heartbeat_t* heartbeat,
    loom_runtime_telemetry_sample_sink_t sink) {
  const loom_serve_text_heartbeat_snapshot_t state =
      loom_serve_text_heartbeat_read(heartbeat);
  static const loom_runtime_telemetry_metric_t fields[] = {
      {"prefill_rows", "requests", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"decode_rows", "requests", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"backpressured_rows", "requests", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"session_capacity", "sessions", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"pool_capacity", "tokens", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"pool_resident", "tokens", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"pool_reserved", "tokens", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"session_cache_endpoints", "sessions", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"prefix_cache_endpoints", "prefixes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"checkpoint_host_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"elastic_state_committed_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"elastic_state_reserved_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"completed_epochs", "epochs", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"traversals", "traversals", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"prefill_tokens", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"decode_tokens", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"model_duration", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"draft_proposed_tokens", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"draft_accepted_inputs", "tokens", LOOM_RUNTIME_TELEMETRY_COUNTER},
  };
  const uint64_t values[] = {
      state.prefill_rows,
      state.decode_rows,
      state.backpressured_rows,
      state.session_capacity,
      state.pool.capacity,
      state.pool.resident,
      state.pool.reserved,
      state.session_cache.endpoints,
      state.prefix_cache.endpoints,
      state.checkpoint_host_bytes,
      state.state_memory.committed_bytes,
      state.state_memory.reserved_bytes,
      state.completed_epochs,
      state.traversals,
      state.prefill_tokens,
      state.decode_tokens,
      (uint64_t)state.model_duration,
      state.draft.proposed_tokens,
      state.draft.accepted_inputs,
  };
  static_assert(IREE_ARRAYSIZE(fields) == IREE_ARRAYSIZE(values),
                "one value per text field");
  const loom_runtime_telemetry_topic_t topic = {"text", 1,
                                                IREE_ARRAYSIZE(fields), fields};
  return sink.emit(sink.self, &topic, values);
}

// Related records share a timestamp but remain independently readable after
// overwrite. Shared device memory is labeled separately from model ownership.
typedef struct loom_serve_text_activity_record_t {
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
  // Allocated resident owners, independent of the latest epoch's span count.
  uint64_t session_capacity;
  // Configured resident ceiling; zero leaves concurrency memory-limited.
  uint64_t max_sessions;
} loom_serve_text_activity_record_t;

typedef struct loom_serve_text_work_record_t {
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
} loom_serve_text_work_record_t;

typedef struct loom_serve_text_memory_record_t {
  // Addressable logical token positions.
  uint64_t capacity;
  // Completion credit disjoint from resident positions.
  uint64_t reserved;
  // Unique resident token positions.
  uint64_t resident;
  // Explicitly suspended endpoint payload.
  uint64_t checkpoint_host_bytes;
  // Cold automatic sizing target, independent of active working-set admission.
  uint64_t cache_target_bytes;
  // Selected initial extra declared state and native checkpoint maps.
  uint64_t cache_additional_bytes;
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
} loom_serve_text_memory_record_t;

typedef struct loom_serve_text_backing_record_t {
  // Model mutable-state backing excluding weights/workspace.
  loom_serve_memory_statistics_t state;
  // Independently reclaimable model parameter backing.
  loom_serve_memory_usage_t weights;
} loom_serve_text_backing_record_t;

typedef struct loom_serve_text_shared_memory_record_t {
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
} loom_serve_text_shared_memory_record_t;

static iree_status_t loom_serve_text_shared_memory_fields(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_serve_text_shared_memory_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_bool("shared", true),
      loom_runtime_telemetry_field_uint64("retained_committed_bytes",
                                          value.retained_committed_bytes),
      loom_runtime_telemetry_field_uint64("workspace_committed_bytes",
                                          value.workspace_committed_bytes),
      loom_runtime_telemetry_field_uint64("workspace_live_bytes",
                                          value.workspace_live_bytes),
      loom_runtime_telemetry_field_uint64("workspace_slab_count",
                                          value.workspace_slab_count),
      loom_runtime_telemetry_field_uint64("workspace_reuse_count",
                                          value.workspace_reuse_count),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static const loom_serve_telemetry_event_t text_shared_memory_event = {
    "shared_pool_memory", loom_serve_text_shared_memory_fields};

static iree_status_t loom_serve_text_activity_fields(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_serve_text_activity_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("phase_start_ns", value.phase_start),
      loom_runtime_telemetry_field_uint64("last_completion_ns",
                                          value.last_completion),
      loom_runtime_telemetry_field_uint64("oldest_queued_arrival_ns",
                                          value.oldest_queued_arrival),
      loom_runtime_telemetry_field_uint64("oldest_active_arrival_ns",
                                          value.oldest_active_arrival),
      loom_runtime_telemetry_field_uint64("prefill_rows", value.prefill_rows),
      loom_runtime_telemetry_field_uint64("decode_rows", value.decode_rows),
      loom_runtime_telemetry_field_uint64("backpressured_rows",
                                          value.backpressured_rows),
      loom_runtime_telemetry_field_uint64("epoch_spans", value.epoch_spans),
      loom_runtime_telemetry_field_uint64("epoch_tokens", value.epoch_tokens),
      loom_runtime_telemetry_field_uint64("session_capacity",
                                          value.session_capacity),
      loom_runtime_telemetry_field_uint64("max_sessions", value.max_sessions),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_serve_text_work_fields(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_serve_text_work_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t draft[] = {
      loom_runtime_telemetry_field_uint64("proposed_tokens",
                                          value.proposed_tokens),
      loom_runtime_telemetry_field_uint64("accepted_draft_inputs",
                                          value.accepted_inputs),
  };
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("issued_epochs", value.issued_epochs),
      loom_runtime_telemetry_field_uint64("completed_epochs",
                                          value.completed_epochs),
      loom_runtime_telemetry_field_uint64("traversals", value.traversals),
      loom_runtime_telemetry_field_uint64("prefill_tokens",
                                          value.prefill_tokens),
      loom_runtime_telemetry_field_uint64("decode_tokens", value.decode_tokens),
      loom_runtime_telemetry_field_uint64("model_ns", value.model_ns),
      loom_runtime_telemetry_field_object("draft", IREE_ARRAYSIZE(draft),
                                          draft),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_serve_text_memory_fields(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_serve_text_memory_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t pool[] = {
      loom_runtime_telemetry_field_uint64("capacity_tokens", value.capacity),
      loom_runtime_telemetry_field_uint64("reserved_tokens", value.reserved),
      loom_runtime_telemetry_field_uint64("resident_tokens", value.resident),
  };
  const loom_runtime_telemetry_field_t session_cache[] = {
      loom_runtime_telemetry_field_uint64("capacity",
                                          value.session_cache.capacity),
      loom_runtime_telemetry_field_uint64("endpoints",
                                          value.session_cache.endpoints),
  };
  const loom_runtime_telemetry_field_t prefix_cache[] = {
      loom_runtime_telemetry_field_uint64("capacity",
                                          value.prefix_cache.capacity),
      loom_runtime_telemetry_field_uint64("endpoints",
                                          value.prefix_cache.endpoints),
  };
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_object("pool", IREE_ARRAYSIZE(pool), pool),
      loom_runtime_telemetry_field_uint64("checkpoint_host_bytes",
                                          value.checkpoint_host_bytes),
      loom_runtime_telemetry_field_object(
          "session_cache", IREE_ARRAYSIZE(session_cache), session_cache),
      loom_runtime_telemetry_field_object(
          "prefix_cache", IREE_ARRAYSIZE(prefix_cache), prefix_cache),
      loom_runtime_telemetry_field_uint64("cache_target_bytes",
                                          value.cache_target_bytes),
      loom_runtime_telemetry_field_uint64("cache_additional_bytes",
                                          value.cache_additional_bytes),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_serve_text_backing_fields(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_serve_text_backing_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t elastic_state[] = {
      loom_runtime_telemetry_field_uint64("reserved_bytes",
                                          value.state.reserved_bytes),
      loom_runtime_telemetry_field_uint64("committed_bytes",
                                          value.state.committed_bytes),
      loom_runtime_telemetry_field_uint64("pending_bytes",
                                          value.state.pending_bytes),
      loom_runtime_telemetry_field_uint64("releasing_bytes",
                                          value.state.releasing_bytes),
      loom_runtime_telemetry_field_uint64("peak_bytes", value.state.peak_bytes),
      loom_runtime_telemetry_field_uint64("released_bytes",
                                          value.state.released_bytes),
  };
  const loom_runtime_telemetry_field_t elastic_parameters[] = {
      loom_runtime_telemetry_field_uint64("reserved_bytes",
                                          value.weights.reserved_bytes),
      loom_runtime_telemetry_field_uint64("committed_bytes",
                                          value.weights.committed_bytes),
      loom_runtime_telemetry_field_uint64("pending_bytes",
                                          value.weights.pending_bytes),
      loom_runtime_telemetry_field_uint64("releasing_bytes",
                                          value.weights.releasing_bytes),
      loom_runtime_telemetry_field_uint64("released_bytes",
                                          value.weights.released_bytes),
  };
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_object(
          "elastic_state", IREE_ARRAYSIZE(elastic_state), elastic_state),
      loom_runtime_telemetry_field_object("elastic_parameters",
                                          IREE_ARRAYSIZE(elastic_parameters),
                                          elastic_parameters),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static const loom_serve_telemetry_event_t text_activity_event = {
    "text_activity", loom_serve_text_activity_fields};
static const loom_serve_telemetry_event_t text_work_event = {
    "text_work", loom_serve_text_work_fields};
static const loom_serve_telemetry_event_t text_memory_event = {
    "text_memory", loom_serve_text_memory_fields};
static const loom_serve_telemetry_event_t text_backing_event = {
    "text_backing", loom_serve_text_backing_fields};

loom_serve_text_heartbeat_snapshot_t loom_serve_text_heartbeat_read(
    loom_serve_text_heartbeat_t* heartbeat) {
  iree_slim_mutex_lock(&heartbeat->mutex);
  const loom_serve_text_heartbeat_snapshot_t state = heartbeat->snapshot;
  iree_slim_mutex_unlock(&heartbeat->mutex);
  return state;
}

static void loom_serve_text_heartbeat_report(
    loom_serve_text_heartbeat_t* heartbeat) {
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
  const loom_serve_text_activity_record_t activity = {
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
  record.event = &text_activity_event;
  memcpy(record.data.words, &activity, sizeof(activity));
  loom_serve_telemetry_stream_append(heartbeat->stream, &record);
  const loom_serve_text_work_record_t work = {
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
  const loom_serve_text_memory_record_t memory = {
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
  record.event = &text_memory_event;
  memcpy(record.data.words, &memory, sizeof(memory));
  loom_serve_telemetry_stream_append(heartbeat->stream, &record);
  const loom_serve_text_backing_record_t backing = {
      .state = state.state_memory, .weights = state.weight_memory};
  memset(&record.data, 0, sizeof(record.data));
  record.event = &text_backing_event;
  memcpy(record.data.words, &backing, sizeof(backing));
  loom_serve_telemetry_stream_append(heartbeat->stream, &record);
  loom_serve_telemetry_record_weights(heartbeat->stream, record.time_ns,
                                      &state.weight_activity);
  const loom_serve_text_shared_memory_record_t shared_memory = {
      .retained_committed_bytes = state.pool_memory.retained.committed_bytes,
      .workspace_committed_bytes = state.pool_memory.workspace.bytes_committed,
      .workspace_live_bytes = state.pool_memory.workspace.bytes_reserved,
      .workspace_slab_count = state.pool_memory.workspace.slab_count,
      .workspace_reuse_count = state.pool_memory.workspace.reuse_count};
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
                                                  true, out_progress, &tick);
  if (tick) {
    loom_serve_text_heartbeat_report(heartbeat);
  }
  return status;
}

void loom_serve_text_heartbeat_deinitialize(
    loom_serve_text_heartbeat_t* heartbeat) {
  loom_serve_timer_deinitialize(&heartbeat->timer);
  if (heartbeat->stream) {
    loom_serve_text_heartbeat_report(heartbeat);
  }
  iree_slim_mutex_deinitialize(&heartbeat->mutex);
}
