// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/text/telemetry.h"

#include <string.h>

#include "loom/runtime/telemetry/field.h"
#include "loom/service/telemetry/record.h"
#include "loom/service/text/schedule.h"

void loom_service_text_record(loom_runtime_telemetry_stream_t* stream,
                              const loom_runtime_telemetry_event_t* event,
                              uint64_t request, const void* payload,
                              iree_host_size_t payload_size) {
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(), .request = request, .event = event};
  memcpy(record.data.words, payload, payload_size);
  loom_runtime_telemetry_stream_append(stream, &record);
}

static iree_status_t loom_service_text_admission_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_admission_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("session", value.session),
      loom_runtime_telemetry_field_uint64("row", value.row),
      loom_runtime_telemetry_field_uint64("retained_tokens",
                                          value.retained_tokens),
      loom_runtime_telemetry_field_uint64("appended_tokens",
                                          value.appended_tokens),
      loom_runtime_telemetry_field_uint64("reservation_tokens",
                                          value.reservation_tokens),
      loom_runtime_telemetry_field_uint64("restored_bytes",
                                          value.restored_bytes),
      loom_runtime_telemetry_field_uint64("max_tokens", value.max_tokens),
      loom_runtime_telemetry_field_uint64("prepare_ns", value.prepare_ns),
      loom_runtime_telemetry_field_uint64("queue_rank", value.queue_rank),
      loom_runtime_telemetry_field_uint64("overtakes", value.overtakes),
      loom_runtime_telemetry_field_uint64("overtake_budget",
                                          value.overtake_budget),
      loom_runtime_telemetry_field_uint64("queue_ns", value.queue_ns),
      loom_runtime_telemetry_field_uint64("pool_reserved_tokens",
                                          value.pool_reserved_tokens),
      loom_runtime_telemetry_field_uint64("pool_resident_tokens",
                                          value.pool_resident_tokens),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_text_request_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_request_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  loom_runtime_telemetry_field_t
      counters[LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT];
  loom_service_telemetry_counters_read_fields(&value.counters, counters);
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("session", value.session),
      loom_runtime_telemetry_field_uint64("retained_tokens",
                                          value.retained_tokens),
      loom_runtime_telemetry_field_uint64("position", value.position),
      loom_runtime_telemetry_field_uint64("queue_ns", value.queue_ns),
      loom_runtime_telemetry_field_uint64("first_token_ns",
                                          value.first_token_ns),
      loom_runtime_telemetry_field_uint64("duration_ns", value.duration_ns),
      loom_runtime_telemetry_field_string(
          "finish_reason", iree_make_cstring_view(value.finish_reason)),
      loom_runtime_telemetry_field_object("counters", IREE_ARRAYSIZE(counters),
                                          counters),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_text_session_begin_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_session_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("session", value.session),
      loom_runtime_telemetry_field_uint64("row", value.row),
      loom_runtime_telemetry_field_string("name",
                                          iree_make_cstring_view(value.name)),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_text_session_end_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_session_end_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  loom_runtime_telemetry_field_t
      counters[LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT];
  loom_service_telemetry_counters_read_fields(&value.counters, counters);
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("session", value.session),
      loom_runtime_telemetry_field_uint64("position", value.position),
      loom_runtime_telemetry_field_uint64("resident_tokens",
                                          value.resident_tokens),
      loom_runtime_telemetry_field_object("counters", IREE_ARRAYSIZE(counters),
                                          counters),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_text_epoch_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_epoch_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const char* schedule =
      value.schedule_mode == LOOM_SERVICE_TEXT_SCHEDULE_PACKED    ? "packed"
      : value.schedule_mode == LOOM_SERVICE_TEXT_SCHEDULE_MATCHED ? "matched"
                                                                  : "isolated";
  const loom_runtime_telemetry_field_t draft[] = {
      loom_runtime_telemetry_field_uint64("verifications", value.verifications),
      loom_runtime_telemetry_field_uint64(
          "proposed_tokens", value.verifications * value.draft_depth),
      loom_runtime_telemetry_field_uint64("accepted_draft_inputs",
                                          value.accepted_draft_inputs),
  };
  const loom_runtime_telemetry_field_t fields[] = {
      value.token_slots ? loom_runtime_telemetry_field_uint64("token_slots",
                                                              value.token_slots)
                        : loom_runtime_telemetry_field_null("token_slots"),
      loom_runtime_telemetry_field_uint64("epoch", value.epoch),
      loom_runtime_telemetry_field_string("scheduler",
                                          iree_make_cstring_view(schedule)),
      loom_runtime_telemetry_field_string(
          "packing",
          iree_make_cstring_view(value.packing_mode ==
                                         LOOM_SERVICE_TEXT_PACKING_SEPARATE
                                     ? "separate"
                                     : "mixed")),
      loom_runtime_telemetry_field_uint64("spans", value.spans),
      loom_runtime_telemetry_field_uint64("shape", value.shape),
      loom_runtime_telemetry_field_uint64("token_capacity",
                                          value.token_capacity),
      loom_runtime_telemetry_field_uint64("span_capacity", value.span_capacity),
      loom_runtime_telemetry_field_uint64("prefill_tokens",
                                          value.prefill_tokens),
      loom_runtime_telemetry_field_uint64("decode_tokens", value.decode_tokens),
      loom_runtime_telemetry_field_uint64("selected_tokens_including_eos",
                                          value.output_tokens),
      loom_runtime_telemetry_field_uint64("traversals", value.traversals),
      loom_runtime_telemetry_field_uint64("model_ns", value.duration_ns),
      loom_runtime_telemetry_field_uint64("device_epochs", value.device_epochs),
      loom_runtime_telemetry_field_uint64("continued_prefill_tokens",
                                          value.continued_prefill_tokens),
      loom_runtime_telemetry_field_object("draft", IREE_ARRAYSIZE(draft),
                                          draft),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_text_span_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_span_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("epoch", value.epoch),
      loom_runtime_telemetry_field_uint64("session", value.session),
      loom_runtime_telemetry_field_uint64("row", value.row),
      loom_runtime_telemetry_field_string(
          "kind", iree_make_cstring_view(value.prefill        ? "prefill"
                                         : value.output_limit ? "verify"
                                                              : "decode")),
      loom_runtime_telemetry_field_uint64("position", value.position),
      loom_runtime_telemetry_field_uint64("tokens", value.tokens),
      loom_runtime_telemetry_field_bool("select", value.select != 0),
      loom_runtime_telemetry_field_uint64("consumed_tokens",
                                          value.consumed_tokens),
      loom_runtime_telemetry_field_uint64("output_tokens", value.output_tokens),
      loom_runtime_telemetry_field_uint64("output_limit", value.output_limit),
      loom_runtime_telemetry_field_uint64("known_tokens", value.known_tokens),
      loom_runtime_telemetry_field_uint64("verification_epochs",
                                          value.verification_epochs),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_text_checkpoint_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_checkpoint_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_string("checkpoint",
                                          iree_make_cstring_view(value.name)),
      loom_runtime_telemetry_field_uint64("position", value.position),
      loom_runtime_telemetry_field_uint64("host_snapshot_bytes",
                                          value.host_bytes),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_text_prefix_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_prefix_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("readiness_ns", value.readiness_ns),
      loom_runtime_telemetry_field_uint64("capacity", value.capacity),
      loom_runtime_telemetry_field_uint64("endpoints", value.endpoints),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_PREFIX_CAPTURE = {
    "prefix_capture", loom_service_text_prefix_fields};

static iree_status_t loom_service_text_trim_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_trim_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("duration_ns", value.duration_ns),
      loom_runtime_telemetry_field_uint64("moved_blocks", value.moved_blocks),
      loom_runtime_telemetry_field_uint64("copied_bytes", value.copied_bytes),
      loom_runtime_telemetry_field_uint64("released_bytes",
                                          value.released_bytes),
      loom_runtime_telemetry_field_uint64("committed_bytes",
                                          value.committed_bytes),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_PREPARED = {
    "prepared", loom_service_text_admission_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_ADMIT = {
    "admit", loom_service_text_admission_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_REQUEST_STATS = {
    "request_stats", loom_service_text_request_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_SESSION_BEGIN = {
    "session_begin", loom_service_text_session_begin_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_SESSION_END = {
    "session_end", loom_service_text_session_end_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_EPOCH = {
    .name = "epoch", .read_fields = loom_service_text_epoch_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_SPAN = {
    "span", loom_service_text_span_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_CHECKPOINT_PIN = {
    "checkpoint_pin", loom_service_text_checkpoint_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_CHECKPOINT_SUSPEND = {
    "checkpoint_suspend", loom_service_text_checkpoint_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_CHECKPOINT_RELEASE = {
    "checkpoint_release", loom_service_text_checkpoint_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_TRIM = {
    "state_trim", loom_service_text_trim_fields};

static iree_status_t loom_service_text_activity_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_activity_record_t value;
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

static iree_status_t loom_service_text_work_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_work_record_t value;
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

static iree_status_t loom_service_text_memory_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_memory_record_t value;
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

static iree_status_t loom_service_text_backing_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_text_backing_record_t value;
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
                                          value.parameters.reserved_bytes),
      loom_runtime_telemetry_field_uint64("committed_bytes",
                                          value.parameters.committed_bytes),
      loom_runtime_telemetry_field_uint64("pending_bytes",
                                          value.parameters.pending_bytes),
      loom_runtime_telemetry_field_uint64("releasing_bytes",
                                          value.parameters.releasing_bytes),
      loom_runtime_telemetry_field_uint64("released_bytes",
                                          value.parameters.released_bytes),
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

const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_ACTIVITY = {
    "text_activity", loom_service_text_activity_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_WORK = {
    "text_work", loom_service_text_work_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_MEMORY = {
    "text_memory", loom_service_text_memory_fields};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TEXT_BACKING = {
    "text_backing", loom_service_text_backing_fields};
