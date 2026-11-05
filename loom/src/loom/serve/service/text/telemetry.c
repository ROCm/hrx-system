// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/telemetry.h"

#include <string.h>

#include "loom/service/telemetry/record.h"
#include "loom/service/text/schedule.h"

iree_status_t loom_serve_text_telemetry_initialize(
    loom_runtime_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_text_telemetry_t* out_telemetry) {
  *out_telemetry = (loom_serve_text_telemetry_t){.stream = stream};
  if (stream) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_runtime_telemetry_stream_allocate(
      256, host_allocator, &out_telemetry->stream));
  out_telemetry->owns_stream = true;
  const loom_runtime_telemetry_record_t begin = {
      .time_ns = iree_time_now(),
      .event = &LOOM_SERVICE_TELEMETRY_STREAM_BEGIN};
  loom_runtime_telemetry_stream_append(out_telemetry->stream, &begin);
  return iree_ok_status();
}

void loom_serve_text_telemetry_deinitialize(
    loom_serve_text_telemetry_t* telemetry) {
  iree_status_free(telemetry->failure);
  if (telemetry->owns_stream) {
    loom_runtime_telemetry_stream_free(telemetry->stream);
  }
  *telemetry = (loom_serve_text_telemetry_t){0};
}

loom_serve_text_receipt_t loom_serve_text_telemetry_begin(
    loom_serve_text_telemetry_t* telemetry) {
  const loom_serve_text_receipt_t receipt = {++telemetry->next_serial,
                                             iree_time_now()};
  ++telemetry->counters.requests;
  const loom_runtime_telemetry_record_t begin = {
      .time_ns = receipt.arrival_time,
      .request = receipt.serial,
      .event = &LOOM_SERVICE_TELEMETRY_REQUEST_BEGIN};
  loom_runtime_telemetry_stream_append(telemetry->stream, &begin);
  return receipt;
}

void loom_serve_text_telemetry_diagnose(loom_serve_text_telemetry_t* telemetry,
                                        uint64_t request,
                                        iree_status_t status) {
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .request = request,
      .event = &LOOM_SERVICE_TELEMETRY_ERROR};
  loom_service_telemetry_diagnostic_t telemetry_payload = {0};
  loom_service_telemetry_diagnostic_capture(status, &telemetry_payload);
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(telemetry->stream, &record);
  iree_status_free(status);
}

void loom_serve_text_telemetry_fail(loom_serve_text_telemetry_t* telemetry,
                                    const iree_status_t status) {
  if (iree_status_is_ok(telemetry->failure)) {
    telemetry->failure = iree_status_clone(status);
  }
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(), .event = &LOOM_SERVICE_TELEMETRY_ERROR};
  loom_service_telemetry_diagnostic_t telemetry_payload = {0};
  loom_service_telemetry_diagnostic_capture(status, &telemetry_payload);
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(telemetry->stream, &record);
}

static void loom_serve_text_telemetry_end_diagnostic(
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_receipt_t receipt,
    loom_service_telemetry_outcome_t outcome, uint64_t input_tokens,
    const loom_service_telemetry_diagnostic_t* diagnostic) {
  loom_service_telemetry_count_outcome(&telemetry->counters, outcome);
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .request = receipt.serial,
      .event = &LOOM_SERVICE_TELEMETRY_REQUEST_END};
  loom_service_telemetry_end_record_t telemetry_payload = {0};
  telemetry_payload.outcome = outcome;
  telemetry_payload.duration_ns = record.time_ns - receipt.arrival_time;
  loom_service_telemetry_latency_observe(&telemetry->latency.outcomes[outcome],
                                         telemetry_payload.duration_ns);
  telemetry_payload.input_tokens = input_tokens;
  telemetry_payload.diagnostic = *diagnostic;
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(telemetry->stream, &record);
}

void loom_serve_text_telemetry_end(loom_serve_text_telemetry_t* telemetry,
                                   loom_serve_text_receipt_t receipt,
                                   loom_service_telemetry_outcome_t outcome,
                                   uint64_t input_tokens,
                                   const iree_status_t borrowed_status) {
  loom_service_telemetry_diagnostic_t diagnostic;
  loom_service_telemetry_diagnostic_capture(borrowed_status, &diagnostic);
  loom_serve_text_telemetry_end_diagnostic(telemetry, receipt, outcome,
                                           input_tokens, &diagnostic);
}

void loom_serve_text_telemetry_request_end(
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_session_t* session,
    loom_service_telemetry_outcome_t outcome) {
  loom_service_telemetry_count_outcome(&session->request.counters, outcome);
  loom_service_telemetry_count_outcome(&session->lifetime.counters, outcome);
  const loom_serve_text_receipt_t receipt = {
      session->lifetime.serial,
      session->request.start_time - session->request.queue_duration};
  loom_service_text_request_record_t value = {
      .session = session->lifetime.id,
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
  loom_service_text_record(telemetry->stream, &LOOM_SERVICE_TEXT_REQUEST_STATS,
                           session->lifetime.serial, &value, sizeof(value));
  if (!iree_status_is_ok(telemetry->failure)) {
    loom_serve_text_telemetry_end(telemetry, receipt, outcome,
                                  value.counters.input_tokens,
                                  telemetry->failure);
  } else {
    loom_serve_text_telemetry_end_diagnostic(telemetry, receipt, outcome,
                                             value.counters.input_tokens,
                                             &session->request.diagnostic);
  }
}

void loom_serve_text_telemetry_session_end(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_text_session_lifetime_t* lifetime) {
  if (!lifetime->id) {
    return;
  }
  const loom_service_text_session_end_record_t value = {
      .session = lifetime->id,
      .counters = lifetime->counters,
      .position = lifetime->position,
      .resident_tokens = lifetime->resident_tokens};
  loom_service_text_record(telemetry->stream, &LOOM_SERVICE_TEXT_SESSION_END, 0,
                           &value, sizeof(value));
  memset(lifetime, 0, sizeof(*lifetime));
}

void loom_serve_text_telemetry_epoch(
    loom_serve_text_telemetry_t* telemetry,
    loom_serve_text_heartbeat_t* heartbeat,
    const loom_serve_text_epoch_options_t* options,
    const loom_serve_text_epoch_t* cohort,
    loom_serve_text_session_t* const* sessions) {
  const iree_host_size_t count = cohort->count;
  const iree_host_size_t shape_index = cohort->shape_index;
  const loom_serve_text_span_t* spans = cohort->spans;
  const loom_serve_text_result_t* results = cohort->results;
  const iree_host_size_t* positions = cohort->positions;
  const uint32_t* output_limits = cohort->output_limits;
  const iree_time_t start = cohort->start.time;
  const uint64_t epoch = cohort->start.serial;
  const iree_host_size_t device_epochs = cohort->device_epochs;
  const iree_host_size_t continued_prefill_count =
      cohort->continued_prefill_count;
  const bool packed =
      options->schedule_mode == LOOM_SERVICE_TEXT_SCHEDULE_PACKED;
  iree_host_size_t prefill_count = 0, decode_count = 0, output_count = 0;
  iree_host_size_t verification_count = 0, accepted_drafts = 0;
  loom_service_text_statistics_t* state = &heartbeat->snapshot;
  const iree_time_t completed = iree_time_now();
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_serve_text_session_t* session = sessions[spans[i].row_index];
    const bool prefill = cohort->prefill[i];
    const bool select =
        iree_any_bit_set(spans[i].flags, LOOM_SERVICE_TEXT_SPAN_FLAG_SELECT);
    prefill_count += prefill ? results[i].known_count : 0;
    decode_count +=
        results[i].consumed_count - (prefill ? results[i].known_count : 0);
    output_count += results[i].output_count;
    verification_count += results[i].verification_count;
    accepted_drafts += results[i].consumed_count - results[i].known_count -
                       results[i].verification_count;
    session->request.counters.input_tokens += results[i].consumed_count;
    session->request.counters.output_tokens += results[i].output_count;
    ++session->request.counters.batches;
    session->lifetime.counters.input_tokens += results[i].consumed_count;
    session->lifetime.counters.output_tokens += results[i].output_count;
    ++session->lifetime.counters.batches;
    session->lifetime.position = loom_serve_text_row_position(session->row);
    session->lifetime.resident_tokens =
        loom_serve_text_row_pool_usage(session->row);
    const loom_service_text_span_record_t value = {
        .epoch = epoch,
        .session = session->lifetime.id,
        .row = spans[i].row_index,
        .position = positions[i],
        .tokens = spans[i].token_count,
        .consumed_tokens = results[i].consumed_count,
        .output_tokens = results[i].output_count,
        .output_limit = output_limits[i],
        .known_tokens = results[i].known_count,
        .verification_epochs = results[i].verification_count,
        .prefill = prefill,
        .select = select};
    loom_service_text_record(telemetry->stream, &LOOM_SERVICE_TEXT_SPAN,
                             session->lifetime.serial, &value, sizeof(value));
  }
  iree_slim_mutex_lock(&heartbeat->mutex);
  state->completed_epochs += device_epochs;
  state->traversals += packed ? device_epochs : count;
  state->prefill_tokens += prefill_count;
  state->decode_tokens += decode_count;
  state->model_duration += completed - start;
  state->draft.proposed_tokens += verification_count * options->draft_depth;
  state->draft.accepted_inputs += accepted_drafts;
  state->last_completion = completed;
  iree_slim_mutex_unlock(&heartbeat->mutex);
  telemetry->counters.input_tokens += prefill_count + decode_count;
  telemetry->counters.output_tokens += output_count;
  ++telemetry->counters.batches;
  const loom_service_text_epoch_record_t value = {
      .epoch = epoch,
      .spans = count,
      .shape = shape_index,
      .token_capacity = options->shapes[shape_index].token_capacity,
      .token_slots = cohort->token_slots,
      .span_capacity = options->shapes[shape_index].span_capacity,
      .prefill_tokens = prefill_count,
      .decode_tokens = decode_count,
      .output_tokens = output_count,
      .traversals = packed ? device_epochs : count,
      .duration_ns = completed - start,
      .device_epochs = device_epochs,
      .continued_prefill_tokens = continued_prefill_count,
      .verifications = verification_count,
      .accepted_draft_inputs = accepted_drafts,
      .schedule_mode = options->schedule_mode,
      .packing_mode = options->packing_mode,
      .draft_depth = (uint8_t)options->draft_depth};
  loom_service_text_record(telemetry->stream, &LOOM_SERVICE_TEXT_EPOCH, 0,
                           &value, sizeof(value));
}
