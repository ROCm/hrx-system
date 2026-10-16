// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/recognition/http.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/serve/media/audio_decoder.h"
#include "loom/serve/runtime/parameter/telemetry.h"
#include "loom/serve/service/recognition/transcription.h"
#include "loom/util/json.h"

typedef struct loom_serve_recognition_transcription_row_t {
  // Claimed connection; NULL marks a reusable adapter slot.
  loom_serve_http_connection_t* connection;
  // Borrowed immutable multipart views, retained by the connection claim.
  loom_serve_transcription_t request;
  // Monotonic request identity, also defining FIFO admission order.
  uint64_t id;
  // Owner receipt time, including host queue residence.
  iree_time_t arrival;
  // Caller handle into the typed service; NULL before admission or after close.
  loom_serve_recognition_stream_t* stream;
  // Input state selected by the checked request's PCM declaration.
  union {
    // Independent codec/resampler state borrowing this upload's file bytes.
    loom_serve_audio_decoder_t* decoder;
    // PCM bytes converted by retired or currently accepted CPU work.
    iree_host_size_t pcm_offset;
  } input;
  // Owned bounded F32LE staging, written only by a captured CPU task.
  float* samples;
  // Whole decoded frames still awaiting the typed stream's input credit.
  iree_host_size_t frame_count;
  // Frames already copied from the current staging block.
  iree_host_size_t frame_offset;
  // Input decoding reached EOF, independently of staged input consumption.
  bool decoded;
  // All decoded samples and explicit EOF have reached the typed stream.
  bool finished;
} loom_serve_recognition_transcription_row_t;

struct loom_serve_recognition_http_t {
  // Allocator owning this adapter, request slots, and decoded staging.
  iree_allocator_t host_allocator;
  // Immutable bounds and deployment identity.
  loom_serve_recognition_http_options_t options;
  // Borrowed source metadata and tokenizer; execution belongs to recognition.
  loom_serve_recognition_generator_t generator;
  // Owned typed service, independent of HTTP upload/decode lifetimes.
  loom_serve_recognition_service_t* recognition;
  // Shared workers for finite decode operations.
  iree_task_executor_t* executor;
  // Stable adapter slots, including the bounded host admission queue.
  loom_serve_recognition_transcription_row_t* rows;
  // Number of allocated adapter slots.
  iree_host_size_t capacity;
  // Rotating input selection; one stalled input cannot monopolize the worker.
  iree_host_size_t cursor;
  // No new decode or source admission is accepted after stopping.
  bool stopping;
  // Reused owner-side response storage; transport copies before the next use.
  iree_string_builder_t response;
  // Borrowed registration-wide history, or NULL when observation is omitted.
  loom_serve_telemetry_stream_t* telemetry;
  // Cumulative outcomes independent of bounded history retention.
  loom_serve_telemetry_counters_t counters;
  // Receipt-to-completion observations independent of event history.
  loom_serve_telemetry_latency_t latency;
  // One accepted conversion operation, separate from the typed model CPU owner.
  struct {
    // Intrusive storage reused only after the task's final release callback.
    iree_task_process_t process;
    // Borrowed stable upload, held through release publication.
    loom_serve_recognition_transcription_row_t* row;
  } cpu;
  // Final task publication joins observer replacement and adapter destruction.
  struct {
    // Serializes completion publication with the serving owner.
    iree_slim_mutex_t mutex;
    // Nonblocking owner wake installed before the first task.
    loom_serve_wake_callback_t observer;
    // The accepted conversion task no longer borrows its row or process
    // storage.
    bool ready;
    // Owned conversion failure, published before final release.
    iree_status_t status;
  } events;
};

static void loom_serve_transcription_end(
    loom_serve_recognition_http_t* service,
    const loom_serve_recognition_transcription_row_t* row,
    loom_serve_telemetry_outcome_t outcome,
    const iree_status_t borrowed_status) {
  const iree_time_t now = iree_time_now();
  loom_serve_telemetry_count_outcome(&service->counters, outcome);
  loom_serve_telemetry_latency_observe(&service->latency.outcomes[outcome],
                                       now - row->arrival);
  if (service->telemetry) {
    loom_serve_telemetry_record_t record = {
        .time_ns = now,
        .request = row->id,
        .event = &LOOM_SERVE_TELEMETRY_REQUEST_END};
    record.data.end.outcome = outcome;
    record.data.end.duration_ns = now - row->arrival;
    loom_serve_telemetry_diagnostic_capture(borrowed_status,
                                            &record.data.end.diagnostic);
    loom_serve_telemetry_stream_append(service->telemetry, &record);
    if (service->generator.weight_activity) {
      const loom_serve_weight_activity_t activity =
          service->generator.weight_activity(service->generator.self);
      loom_serve_telemetry_record_weights(service->telemetry, now, &activity);
    }
  }
}

// The CPU owner has retired before this relinquishes borrowed upload bytes.
static void loom_serve_transcription_release(
    loom_serve_recognition_http_t* service,
    loom_serve_recognition_transcription_row_t* row, bool finish,
    loom_serve_telemetry_outcome_t outcome,
    const iree_status_t borrowed_status) {
  loom_serve_transcription_end(service, row, outcome, borrowed_status);
  if (row->stream) {
    loom_serve_recognition_stream_close(row->stream);
  }
  if (!row->request.pcm) {
    loom_serve_audio_decoder_free(row->input.decoder);
  }
  iree_allocator_free(service->host_allocator, row->samples);
  if (finish) {
    loom_serve_http_connection_finish(row->connection);
  } else {
    loom_serve_http_connection_abort(row->connection);
  }
  memset(row, 0, sizeof(*row));
}

// Sends a complete finite response. No partial carrier message or borrowed
// transcript survives this call, including failed/reset peers.
static void loom_serve_transcription_send(
    loom_serve_recognition_http_t* service,
    loom_serve_recognition_transcription_row_t* row,
    loom_serve_telemetry_outcome_t outcome,
    const iree_status_t borrowed_status) {
  if (loom_serve_http_connection_failed(row->connection) ||
      !loom_serve_http_connection_can_send(row->connection)) {
    loom_serve_transcription_release(
        service, row, false, LOOM_SERVE_TELEMETRY_CANCELLED, borrowed_status);
    return;
  }
  iree_status_t status = loom_serve_http_connection_send(
      row->connection, iree_string_builder_view(&service->response));
  if (iree_status_is_ok(status)) {
    loom_serve_transcription_release(service, row, true, outcome,
                                     borrowed_status);
  } else {
    loom_serve_transcription_release(service, row, false,
                                     LOOM_SERVE_TELEMETRY_FAILED, status);
    iree_status_free(status);
  }
}

static iree_status_t loom_serve_transcription_reject(
    loom_serve_recognition_http_t* service,
    loom_serve_recognition_transcription_row_t* row, const char* code,
    const iree_status_t borrowed_status, iree_string_view_t message) {
  iree_string_builder_reset(&service->response);
  iree_status_t status = iree_string_builder_append_format(
      &service->response,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n"
      "{\"error\":{\"type\":\"transcription_request_error\",\"message\":",
      code);
  if (iree_status_is_ok(status)) {
    loom_output_stream_t output;
    loom_output_stream_for_builder(&service->response, &output);
    status = loom_json_write_escaped_string(&output, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&service->response, "}}");
  }
  if (iree_status_is_ok(status)) {
    loom_serve_transcription_send(service, row, LOOM_SERVE_TELEMETRY_REJECTED,
                                  borrowed_status);
  } else {
    loom_serve_transcription_release(service, row, false,
                                     LOOM_SERVE_TELEMETRY_FAILED, status);
  }
  return status;
}

static iree_status_t loom_serve_transcription_fail(
    loom_serve_recognition_http_t* service,
    loom_serve_recognition_transcription_row_t* row,
    const iree_status_t borrowed_status) {
  iree_string_view_t message = iree_status_message(borrowed_status);
  if (!message.size) {
    message = iree_make_cstring_view(
        iree_status_code_string(iree_status_code(borrowed_status)));
  }
  const char* code = "500 Internal Server Error";
  switch (iree_status_code(borrowed_status)) {
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
    case IREE_STATUS_UNIMPLEMENTED:
      code = "400 Bad Request";
      break;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
    case IREE_STATUS_UNAVAILABLE:
      code = "503 Service Unavailable";
      break;
    default:
      break;
  }
  return loom_serve_transcription_reject(service, row, code, borrowed_status,
                                         message);
}

static iree_status_t loom_serve_transcription_cpu_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_recognition_http_t* service = process->user_data;
  loom_serve_recognition_transcription_row_t* row = service->cpu.row;
  result->completed = true;
  result->did_work = true;
  if (row->request.pcm) {
    const iree_host_size_t frame_bytes =
        service->generator.format.channels * sizeof(int16_t);
    const iree_host_size_t remaining =
        row->request.audio.data_length - row->input.pcm_offset;
    row->frame_count =
        iree_min(service->options.decode_frames, remaining / frame_bytes);
    const uint8_t* samples = row->request.audio.data + row->input.pcm_offset;
    for (iree_host_size_t i = 0;
         i < row->frame_count * service->generator.format.channels; ++i) {
      const int16_t sample =
          (int16_t)iree_unaligned_load_le_u16(samples + i * sizeof(int16_t));
      iree_unaligned_store_le_f32(row->samples + i, (float)sample / 32768.0f);
    }
    row->input.pcm_offset += row->frame_count * frame_bytes;
    row->decoded = row->input.pcm_offset == row->request.audio.data_length;
    return iree_ok_status();
  }
  if (!row->input.decoder) {
    IREE_RETURN_IF_ERROR(loom_serve_audio_decoder_allocate(
        row->request.audio, service->generator.format.sample_rate,
        service->generator.format.channels, service->host_allocator,
        &row->input.decoder));
  }
  IREE_RETURN_IF_ERROR(loom_serve_audio_decoder_read(
      row->input.decoder, service->options.decode_frames, row->samples,
      &row->frame_count, &row->decoded));
  // The codec's native F32 representation becomes the source ABI's F32LE at
  // this producer boundary. Little-endian targets eliminate these stores.
  for (iree_host_size_t i = 0;
       i < row->frame_count * service->generator.format.channels; ++i) {
    iree_unaligned_store_le_f32(row->samples + i, row->samples[i]);
  }
  return iree_ok_status();
}

static void loom_serve_transcription_cpu_complete(iree_task_process_t* process,
                                                  iree_status_t status) {
  loom_serve_recognition_http_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.status = status;
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_transcription_cpu_release(iree_task_process_t* process) {
  loom_serve_recognition_http_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.ready = true;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_transcription_decode(
    loom_serve_recognition_http_t* service) {
  for (iree_host_size_t i = 0; i < service->capacity; ++i) {
    const iree_host_size_t index = (service->cursor + i) % service->capacity;
    loom_serve_recognition_transcription_row_t* row = &service->rows[index];
    if (!row->stream || row->decoded || row->frame_count ||
        loom_serve_recognition_stream_is_complete(row->stream)) {
      continue;
    }
    service->cursor = (index + 1) % service->capacity;
    service->cpu.row = row;
    iree_task_process_initialize(loom_serve_transcription_cpu_run, 0, 1,
                                 &service->cpu.process);
    service->cpu.process.user_data = service;
    service->cpu.process.completion_fn = loom_serve_transcription_cpu_complete;
    service->cpu.process.release_fn = loom_serve_transcription_cpu_release;
    iree_task_executor_schedule_process(service->executor,
                                        &service->cpu.process);
    break;
  }
}

static iree_status_t loom_serve_transcription_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_service_flags_t flags) {
  loom_serve_recognition_http_t* service = self;
  loom_serve_recognition_transcription_row_t pending = {
      .connection = connection,
      .id = ++service->counters.requests,
      .arrival = iree_time_now()};
  if (service->telemetry) {
    const loom_serve_telemetry_record_t record = {
        .time_ns = pending.arrival,
        .request = pending.id,
        .event = &LOOM_SERVE_TELEMETRY_REQUEST_BEGIN};
    loom_serve_telemetry_stream_append(service->telemetry, &record);
  }
  if (service->stopping || (flags & LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    return loom_serve_transcription_reject(
        service, &pending, "503 Service Unavailable", iree_ok_status(),
        IREE_SV("transcription is stopping"));
  }
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(request->target,
                              IREE_SV("/v1/audio/transcriptions"))) {
    return loom_serve_transcription_reject(
        service, &pending, "404 Not Found", iree_ok_status(),
        IREE_SV("unknown transcription endpoint"));
  }
  if (!request->multipart) {
    return loom_serve_transcription_reject(
        service, &pending, "415 Unsupported Media Type", iree_ok_status(),
        IREE_SV("use multipart/form-data"));
  }
  iree_status_t status = loom_serve_transcription_parse(
      loom_serve_http_multipart_parts(request->multipart),
      service->options.name, service->generator.format, &pending.request);
  if (iree_status_is_ok(status) && pending.request.pcm &&
      pending.request.audio.data_length /
              (service->generator.format.channels * sizeof(int16_t)) >
          service->generator.frame_limit) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "PCM upload exceeds the model input frame limit");
  }
  if (!iree_status_is_ok(status)) {
    iree_status_t response =
        loom_serve_transcription_fail(service, &pending, status);
    iree_status_free(status);
    return response;
  }
  for (iree_host_size_t i = 0; i < service->capacity; ++i) {
    if (!service->rows[i].connection) {
      service->rows[i] = pending;
      return iree_ok_status();
    }
  }
  return loom_serve_transcription_reject(
      service, &pending, "503 Service Unavailable", iree_ok_status(),
      IREE_SV("transcription admission queue is full"));
}

static iree_status_t loom_serve_transcription_deliver(
    loom_serve_recognition_http_t* service,
    loom_serve_recognition_transcription_row_t* row) {
  const iree_status_t failure =
      loom_serve_recognition_stream_status(row->stream);
  if (!iree_status_is_ok(failure)) {
    iree_status_t status = loom_serve_transcription_fail(service, row, failure);
    iree_status_free(failure);
    return status;
  }
  // Typed sources may terminate an input session themselves. A finite upload
  // cannot report success while its input tail has never reached the model.
  if (!row->finished) {
    iree_status_t status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "recognition completed before the uploaded audio reached end-of-input");
    iree_status_t response =
        loom_serve_transcription_fail(service, row, status);
    iree_status_free(status);
    return response;
  }
  iree_string_builder_reset(&service->response);
  iree_status_t status = iree_string_builder_append_format(
      &service->response,
      "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nConnection: close\r\n\r\n",
      row->request.text_response ? "text/plain; charset=utf-8"
                                 : "application/json");
  if (iree_status_is_ok(status)) {
    status = loom_serve_transcription_format(
        loom_serve_recognition_stream_document(row->stream),
        row->request.text_response, &service->response);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_transcription_send(service, row, LOOM_SERVE_TELEMETRY_COMPLETED,
                                  iree_ok_status());
  } else {
    loom_serve_transcription_release(service, row, false,
                                     LOOM_SERVE_TELEMETRY_FAILED, status);
  }
  return status;
}

static iree_status_t loom_serve_transcription_advance(
    void* self, loom_serve_service_flags_t flags, bool* out_progress) {
  loom_serve_recognition_http_t* service = self;
  *out_progress = false;
  service->stopping |= (flags & LOOM_SERVE_SERVICE_FLAG_STOPPING) != 0;
  iree_slim_mutex_lock(&service->events.mutex);
  const bool ready = service->events.ready;
  iree_status_t cpu_status = iree_ok_status();
  if (ready) {
    service->events.ready = false;
    cpu_status = service->events.status;
    service->events.status = iree_ok_status();
  }
  iree_slim_mutex_unlock(&service->events.mutex);
  iree_status_t status = iree_ok_status();
  if (ready) {
    loom_serve_recognition_transcription_row_t* row = service->cpu.row;
    service->cpu.row = NULL;
    *out_progress = true;
    if (!iree_status_is_ok(cpu_status)) {
      if (service->stopping ||
          loom_serve_http_connection_failed(row->connection)) {
        loom_serve_transcription_release(
            service, row, false, LOOM_SERVE_TELEMETRY_CANCELLED, cpu_status);
      } else {
        status = loom_serve_transcription_fail(service, row, cpu_status);
      }
      iree_status_free(cpu_status);
    }
  }
  for (iree_host_size_t i = 0;
       i < service->capacity && iree_status_is_ok(status); ++i) {
    loom_serve_recognition_transcription_row_t* row = &service->rows[i];
    if (!row->connection) {
      continue;
    }
    if (service->stopping ||
        loom_serve_http_connection_failed(row->connection)) {
      if (row->stream) {
        loom_serve_recognition_stream_close(row->stream);
        row->stream = NULL;
        *out_progress = true;
      }
      if (row != service->cpu.row) {
        loom_serve_transcription_release(service, row, false,
                                         LOOM_SERVE_TELEMETRY_CANCELLED,
                                         iree_ok_status());
        *out_progress = true;
      }
      continue;
    }
    if (!row->stream || row == service->cpu.row) {
      continue;
    }
    if (loom_serve_recognition_stream_peek(row->stream)) {
      loom_serve_recognition_stream_consume(row->stream);
      *out_progress = true;
    }
    if (loom_serve_recognition_stream_is_complete(row->stream)) {
      status = loom_serve_transcription_deliver(service, row);
      *out_progress = true;
      continue;
    }
    if (row->frame_count) {
      const iree_host_size_t channels = service->generator.format.channels;
      iree_host_size_t accepted = 0;
      iree_status_t input_status = loom_serve_recognition_stream_write(
          row->stream,
          iree_make_const_byte_span(
              row->samples + row->frame_offset * channels,
              row->frame_count * channels * sizeof(float)),
          &accepted);
      if (!iree_status_is_ok(input_status)) {
        status = loom_serve_transcription_fail(service, row, input_status);
        iree_status_free(input_status);
        *out_progress = true;
        continue;
      }
      row->frame_count -= accepted;
      row->frame_offset = row->frame_count ? row->frame_offset + accepted : 0;
      *out_progress |= accepted != 0;
    }
    if (row->decoded && !row->frame_count && !row->finished) {
      loom_serve_recognition_stream_finish(row->stream);
      row->finished = true;
      *out_progress = true;
    }
  }
  // FIFO host admission is independent of resident numerical batch order.
  bool admission_full = service->stopping;
  while (!admission_full && iree_status_is_ok(status)) {
    loom_serve_recognition_transcription_row_t* queued = NULL;
    for (iree_host_size_t i = 0; i < service->capacity; ++i) {
      loom_serve_recognition_transcription_row_t* row = &service->rows[i];
      if (row->connection && !row->stream && row != service->cpu.row &&
          (!queued || row->id < queued->id)) {
        queued = row;
      }
    }
    if (!queued) {
      break;
    }
    iree_status_t admission = loom_serve_recognition_service_try_open(
        service->recognition, &queued->request.request, &queued->stream);
    if (iree_status_is_ok(admission) && queued->stream) {
      admission = iree_allocator_malloc_array(
          service->host_allocator,
          service->options.decode_frames * service->generator.format.channels,
          sizeof(float), (void**)&queued->samples);
    }
    if (!iree_status_is_ok(admission)) {
      status = loom_serve_transcription_fail(service, queued, admission);
      iree_status_free(admission);
      *out_progress = true;
    } else {
      admission_full = queued->stream == NULL;
      *out_progress |= !admission_full;
    }
  }
  if (iree_status_is_ok(status)) {
    bool progress = false;
    status = loom_serve_recognition_service_advance(
        service->recognition, service->stopping,
        (flags & LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY) != 0, &progress);
    *out_progress |= progress;
  }
  if (iree_status_is_ok(status) && !service->stopping && !service->cpu.row) {
    loom_serve_transcription_decode(service);
    *out_progress |= service->cpu.row != NULL;
  }
  return status;
}

static void loom_serve_transcription_set_observer(
    void* self, loom_serve_wake_callback_t observer) {
  loom_serve_recognition_http_t* service = self;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.observer = observer;
  iree_slim_mutex_unlock(&service->events.mutex);
  loom_serve_recognition_service_set_observer(service->recognition, observer);
}

static bool loom_serve_transcription_is_active(void* self) {
  loom_serve_recognition_http_t* service = self;
  return service->cpu.row ||
         loom_serve_recognition_service_is_active(service->recognition);
}

static bool loom_serve_transcription_is_pending(void* self) {
  loom_serve_recognition_http_t* service = self;
  for (iree_host_size_t i = 0; i < service->capacity; ++i) {
    if (service->rows[i].connection) {
      return true;
    }
  }
  return loom_serve_transcription_is_active(service);
}

static loom_serve_service_snapshot_t loom_serve_transcription_snapshot(
    void* self) {
  loom_serve_recognition_http_t* service = self;
  loom_serve_service_snapshot_t snapshot = {.phase = "idle"};
  const iree_time_t now = iree_time_now();
  for (iree_host_size_t i = 0; i < service->capacity; ++i) {
    const loom_serve_recognition_transcription_row_t* row = &service->rows[i];
    if (!row->connection) {
      continue;
    }
    if (row->stream || row == service->cpu.row) {
      ++snapshot.active_requests;
      snapshot.oldest_active_age =
          iree_max(snapshot.oldest_active_age, now - row->arrival);
    } else {
      ++snapshot.queued_requests;
      snapshot.oldest_queued_age =
          iree_max(snapshot.oldest_queued_age, now - row->arrival);
    }
  }
  snapshot.phase = snapshot.active_requests   ? "recognizing"
                   : snapshot.queued_requests ? "queued"
                                              : "idle";
  return snapshot;
}

loom_serve_service_t loom_serve_recognition_http_interface(
    loom_serve_recognition_http_t* service) {
  static const iree_string_view_t input[] = {IREE_SVL("audio")};
  static const iree_string_view_t output[] = {IREE_SVL("text")};
  return (loom_serve_service_t){
      .name = service->options.name,
      .self = service,
      .set_observer = loom_serve_transcription_set_observer,
      .is_pending = loom_serve_transcription_is_pending,
      .is_active = loom_serve_transcription_is_active,
      .accept = loom_serve_transcription_accept,
      .advance = loom_serve_transcription_advance,
      .tokenizer = service->generator.tokenizer,
      .kind = "recognition",
      .endpoint = "/v1/audio/transcriptions",
      .capabilities = {.input_modalities = {IREE_ARRAYSIZE(input), input},
                       .output_modalities = {IREE_ARRAYSIZE(output), output}},
      .snapshot = loom_serve_transcription_snapshot,
      .capacity_domain = service->generator.capacity_domain,
      .telemetry = {.stream = service->telemetry,
                    .counters = &service->counters,
                    .latency = &service->latency}};
}

void loom_serve_recognition_http_free(loom_serve_recognition_http_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!service->cpu.row);
  loom_serve_recognition_service_free(service->recognition);
  for (iree_host_size_t i = 0; service->rows && i < service->capacity; ++i) {
    IREE_ASSERT(!service->rows[i].connection);
  }
  iree_string_builder_deinitialize(&service->response);
  iree_status_free(service->events.status);
  iree_slim_mutex_deinitialize(&service->events.mutex);
  iree_allocator_free(service->host_allocator, service->rows);
  iree_allocator_free_aligned(service->host_allocator, service);
}

iree_status_t loom_serve_recognition_http_allocate(
    loom_serve_host_t* host, loom_serve_recognition_generator_t generator,
    const loom_serve_recognition_http_options_t* options,
    loom_serve_telemetry_stream_t* telemetry, iree_allocator_t host_allocator,
    loom_serve_recognition_http_t** out_service) {
  *out_service = NULL;
  if (!options->name.size || !options->pending_capacity ||
      !options->decode_frames || !generator.format.channels ||
      options->decode_frames >
          SIZE_MAX / sizeof(float) / generator.format.channels ||
      generator.request_capacity >
          SIZE_MAX / sizeof(loom_serve_recognition_transcription_row_t) ||
      options->pending_capacity >
          SIZE_MAX / sizeof(loom_serve_recognition_transcription_row_t) -
              generator.request_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transcription requires a name and bounded request/decode capacity");
  }
  loom_serve_recognition_http_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*service),
      iree_alignof(loom_serve_recognition_http_t), 0, (void**)&service));
  service->host_allocator = host_allocator;
  service->options = *options;
  service->generator = generator;
  service->executor = loom_serve_host_executor(host);
  service->telemetry = telemetry;
  service->capacity = generator.request_capacity + options->pending_capacity;
  iree_slim_mutex_initialize(&service->events.mutex);
  iree_string_builder_initialize(host_allocator, &service->response);
  const loom_serve_recognition_service_options_t recognition_options = {
      .request_limit = options->request_limit,
      .document_limit = options->document_limit};
  iree_status_t status = loom_serve_recognition_service_allocate(
      host, generator, &recognition_options, host_allocator,
      &service->recognition);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, service->capacity,
                                         sizeof(*service->rows),
                                         (void**)&service->rows);
  }
  if (iree_status_is_ok(status)) {
    *out_service = service;
  } else {
    loom_serve_recognition_http_free(service);
  }
  return status;
}
