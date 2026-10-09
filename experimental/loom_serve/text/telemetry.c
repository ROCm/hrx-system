// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/telemetry.h"

#include <inttypes.h>
#include <string.h>

#include "experimental/loom_serve/text/schedule.h"

iree_status_t loom_serve_text_telemetry_initialize(
    loom_serve_text_telemetry_t* telemetry, iree_allocator_t allocator) {
  *telemetry = (loom_serve_text_telemetry_t){0};
  IREE_RETURN_IF_ERROR(
      loom_serve_telemetry_stream_create(256, &telemetry->stream, allocator));
  const loom_serve_telemetry_record_t begin = {
      .time_ns = iree_time_now(), .event = &LOOM_SERVE_TELEMETRY_STREAM_BEGIN};
  loom_serve_telemetry_stream_append(telemetry->stream, &begin);
  return iree_ok_status();
}

loom_serve_text_receipt_t loom_serve_text_telemetry_begin(
    loom_serve_text_telemetry_t* telemetry) {
  const loom_serve_text_receipt_t receipt = {++telemetry->next_serial,
                                             iree_time_now()};
  ++telemetry->counters.requests;
  const loom_serve_telemetry_record_t begin = {
      .time_ns = receipt.arrival_time,
      .request = receipt.serial,
      .event = &LOOM_SERVE_TELEMETRY_REQUEST_BEGIN};
  loom_serve_telemetry_stream_append(telemetry->stream, &begin);
  return receipt;
}

void loom_serve_text_record(loom_serve_telemetry_stream_t* stream,
                            const loom_serve_telemetry_event_t* event,
                            uint64_t request, const void* payload,
                            iree_host_size_t payload_size) {
  loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(), .request = request, .event = event};
  memcpy(record.data.words, payload, payload_size);
  loom_serve_telemetry_stream_append(stream, &record);
}

static iree_status_t text_admission_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_text_admission_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"session\":%" PRIu64 ",\"row\":%" PRIu64
      ",\"retained_tokens\":%" PRIu64 ",\"appended_tokens\":%" PRIu64
      ",\"reservation_tokens\":%" PRIu64 ",\"restored_bytes\":%" PRIu64
      ",\"max_tokens\":%" PRIu64 ",\"prepare_ns\":%" PRIu64
      ",\"queue_ns\":%" PRIu64 ",\"pool_reserved_tokens\":%" PRIu64
      ",\"pool_resident_tokens\":%" PRIu64,
      value.session, value.row, value.retained_tokens, value.appended_tokens,
      value.reservation_tokens, value.restored_bytes, value.max_tokens,
      value.prepare_ns, value.queue_ns, value.pool_reserved_tokens,
      value.pool_resident_tokens);
}

static iree_status_t text_request_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_text_request_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"session\":%" PRIu64 ",\"retained_tokens\":%" PRIu64
      ",\"position\":%" PRIu64 ",\"queue_ns\":%" PRIu64
      ",\"first_token_ns\":%" PRIu64 ",\"duration_ns\":%" PRIu64
      ",\"finish_reason\":\"%s\",\"counters\":",
      value.session, value.retained_tokens, value.position, value.queue_ns,
      value.first_token_ns, value.duration_ns, value.finish_reason));
  return loom_serve_telemetry_counters_format(&value.counters, output);
}

static iree_status_t text_session_begin_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_text_session_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  // Session keys have already passed the ASCII identifier boundary.
  return iree_string_builder_append_format(
      output, ",\"session\":%" PRIu64 ",\"row\":%" PRIu64 ",\"name\":\"%s\"",
      value.session, value.row, value.name);
}

static iree_status_t text_session_end_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_text_session_end_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"session\":%" PRIu64 ",\"position\":%" PRIu64
      ",\"resident_tokens\":%" PRIu64 ",\"counters\":",
      value.session, value.position, value.resident_tokens));
  return loom_serve_telemetry_counters_format(&value.counters, output);
}

static iree_status_t text_epoch_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_text_epoch_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const char* schedule =
      value.schedule_mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED    ? "packed"
      : value.schedule_mode == LOOM_SERVE_TEXT_SCHEDULE_MATCHED ? "matched"
                                                                : "isolated";
  return iree_string_builder_append_format(
      output,
      ",\"epoch\":%" PRIu64
      ",\"scheduler\":\"%s\",\"packing\":\"%s\""
      ",\"spans\":%" PRIu64 ",\"shape\":%" PRIu64 ",\"token_capacity\":%" PRIu64
      ",\"span_capacity\":%" PRIu64 ",\"prefill_tokens\":%" PRIu64
      ",\"decode_tokens\":%" PRIu64
      ",\"selected_tokens_including_eos\":%" PRIu64 ",\"traversals\":%" PRIu64
      ",\"model_ns\":%" PRIu64 ",\"device_epochs\":%" PRIu64
      ",\"continued_prefill_tokens\":%" PRIu64
      ",\"mtp\":{\"verifications\":%" PRIu64 ",\"proposed_tokens\":%" PRIu64
      ",\"accepted_draft_inputs\":%" PRIu64 "}",
      value.epoch, schedule,
      value.packing_mode == LOOM_SERVE_TEXT_PACKING_SEPARATE ? "separate"
                                                             : "mixed",
      value.spans, value.shape, value.token_capacity, value.span_capacity,
      value.prefill_tokens, value.decode_tokens, value.output_tokens,
      value.traversals, value.duration_ns, value.device_epochs,
      value.continued_prefill_tokens, value.verifications,
      value.verifications * 3, value.accepted_draft_inputs);
}

static iree_status_t text_span_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_text_span_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"epoch\":%" PRIu64 ",\"session\":%" PRIu64 ",\"row\":%" PRIu64
      ",\"kind\":\"%s\",\"position\":%" PRIu64 ",\"tokens\":%" PRIu64
      ",\"select\":%s,\"consumed_tokens\":%" PRIu64
      ",\"output_tokens\":%" PRIu64 ",\"output_limit\":%" PRIu64
      ",\"known_tokens\":%" PRIu64 ",\"verification_epochs\":%" PRIu64,
      value.epoch, value.session, value.row,
      value.prefill        ? "prefill"
      : value.output_limit ? "verify"
                           : "decode",
      value.position, value.tokens, value.select ? "true" : "false",
      value.consumed_tokens, value.output_tokens, value.output_limit,
      value.known_tokens, value.verification_epochs);
}

static iree_status_t text_checkpoint_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_text_checkpoint_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"checkpoint\":\"%s\",\"position\":%" PRIu64
      ",\"host_snapshot_bytes\":%" PRIu64,
      value.name, value.position, value.host_bytes);
}

static iree_status_t text_trim_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_text_trim_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"duration_ns\":%" PRIu64 ",\"moved_blocks\":%" PRIu64
      ",\"copied_bytes\":%" PRIu64 ",\"released_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64,
      value.duration_ns, value.moved_blocks, value.copied_bytes,
      value.released_bytes, value.committed_bytes);
}

const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_PREPARED = {
    "prepared", text_admission_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_ADMIT = {
    "admit", text_admission_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_REQUEST_STATS = {
    "request_stats", text_request_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_SESSION_BEGIN = {
    "session_begin", text_session_begin_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_SESSION_END = {
    "session_end", text_session_end_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_EPOCH = {"epoch",
                                                            text_epoch_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_SPAN = {"span",
                                                           text_span_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_CHECKPOINT_PIN = {
    "checkpoint_pin", text_checkpoint_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_CHECKPOINT_SUSPEND = {
    "checkpoint_suspend", text_checkpoint_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_CHECKPOINT_RELEASE = {
    "checkpoint_release", text_checkpoint_format};
const loom_serve_telemetry_event_t LOOM_SERVE_TEXT_TRIM = {"state_trim",
                                                           text_trim_format};

void loom_serve_text_telemetry_diagnose(loom_serve_text_telemetry_t* telemetry,
                                        uint64_t request,
                                        iree_status_t status) {
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .request = request,
                                          .event = &LOOM_SERVE_TELEMETRY_ERROR};
  loom_serve_telemetry_diagnostic_capture(status, &record.data.error);
  loom_serve_telemetry_stream_append(telemetry->stream, &record);
  iree_status_free(status);
}

void loom_serve_text_telemetry_fail(loom_serve_text_telemetry_t* telemetry,
                                    iree_status_t status) {
  telemetry->failure_code = iree_status_code(status);
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .event = &LOOM_SERVE_TELEMETRY_ERROR};
  loom_serve_telemetry_diagnostic_capture(status, &record.data.error);
  loom_serve_telemetry_stream_append(telemetry->stream, &record);
}

static void text_count_outcome(loom_serve_telemetry_counters_t* counters,
                               loom_serve_telemetry_outcome_t outcome) {
  switch (outcome) {
    case LOOM_SERVE_TELEMETRY_COMPLETED:
      ++counters->completed;
      break;
    case LOOM_SERVE_TELEMETRY_REJECTED:
      ++counters->rejected;
      break;
    case LOOM_SERVE_TELEMETRY_CANCELLED:
      ++counters->cancelled;
      break;
    case LOOM_SERVE_TELEMETRY_FAILED:
      ++counters->failed;
      break;
  }
}

void loom_serve_text_telemetry_end(loom_serve_text_telemetry_t* telemetry,
                                   loom_serve_text_receipt_t receipt,
                                   loom_serve_telemetry_outcome_t outcome,
                                   uint64_t input_tokens,
                                   iree_status_t borrowed_status) {
  text_count_outcome(&telemetry->counters, outcome);
  loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .request = receipt.serial,
      .event = &LOOM_SERVE_TELEMETRY_REQUEST_END};
  record.data.end.outcome = outcome;
  record.data.end.duration_ns = record.time_ns - receipt.arrival_time;
  record.data.end.input_tokens = input_tokens;
  loom_serve_telemetry_diagnostic_capture(borrowed_status,
                                          &record.data.end.diagnostic);
  loom_serve_telemetry_stream_append(telemetry->stream, &record);
}

void loom_serve_text_telemetry_request_end(
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_session_t* session,
    loom_serve_telemetry_outcome_t outcome) {
  text_count_outcome(&session->request.counters, outcome);
  text_count_outcome(&session->telemetry.counters, outcome);
  const loom_serve_text_receipt_t receipt = {
      session->serial,
      session->request.start_time - session->request.queue_duration};
  loom_serve_text_request_record_t value = {
      .session = session->telemetry.id,
      .counters = session->request.counters,
      .retained_tokens = session->request.retained_count,
      .position = loom_serve_text_row_position(session->row),
      .queue_ns = session->request.queue_duration,
      .first_token_ns =
          session->request.first_token_time
              ? session->request.first_token_time - session->request.start_time
              : 0,
      .duration_ns = iree_time_now() - receipt.arrival_time};
  if (session->request.finish_reason) {
    memcpy(value.finish_reason, session->request.finish_reason,
           strlen(session->request.finish_reason));
  }
  loom_serve_text_record(telemetry->stream, &LOOM_SERVE_TEXT_REQUEST_STATS,
                         session->serial, &value, sizeof(value));
  loom_serve_text_telemetry_end(telemetry, receipt, outcome,
                                value.counters.input_tokens, iree_ok_status());
}

void loom_serve_text_telemetry_session_end(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_text_session_t* session) {
  if (!session->telemetry.id) {
    return;
  }
  const loom_serve_text_session_end_record_t value = {
      .session = session->telemetry.id,
      .counters = session->telemetry.counters,
      .position = session->telemetry.position,
      .resident_tokens = session->telemetry.resident_tokens};
  loom_serve_text_record(telemetry->stream, &LOOM_SERVE_TEXT_SESSION_END, 0,
                         &value, sizeof(value));
  memset(&session->telemetry, 0, sizeof(session->telemetry));
}
