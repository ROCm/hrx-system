// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/recognition/service.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/runtime/parameter/telemetry.h"
#include "loom/serve/media/audio_decoder.h"
#include "loom/serve/media/audio_resampler.h"
#include "loom/serve/media/audio_source.h"
#include "loom/serve/service/endpoint.h"
#include "loom/serve/service/recognition/transcription.h"
#include "loom/service/kind.h"
#include "loom/service/telemetry/record.h"
#include "loom/util/json.h"

typedef struct loom_serve_recognition_job_t {
  // Common owner retained through native stream completion.
  loom_serve_recognition_service_t* owner;
  // Claimed HTTP connection, or NULL for a native conversation.
  loom_serve_http_connection_t* connection;
  // Borrowed immutable multipart views, retained by the connection claim.
  loom_serve_transcription_t request;
  // Monotonic request identity and FIFO order; zero denotes a reusable slot.
  uint64_t id;
  // Owner receipt time, including host queue residence.
  iree_time_t arrival;
  // HTTP caller handle, or a native handle borrowed through core completion.
  // NULL before admission or after the HTTP caller has closed it.
  loom_serve_recognition_stream_t* stream;
  // Input state selected by the checked request's PCM declaration.
  union {
    // Independent codec/resampler state borrowing this upload's file bytes.
    loom_serve_audio_decoder_t* decoder;
    // PCM conversion retained independently of the model's encoder history.
    struct {
      // Upload frames converted by retired or currently accepted CPU work.
      iree_host_size_t offset;
      // Immutable rate conversion, initialized only for a rate mismatch.
      loom_serve_audio_resampler_filter_t filter;
      // Independent history borrowing the stable filter in this request slot.
      loom_serve_audio_resampler_t resampler;
    } pcm;
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
} loom_serve_recognition_job_t;

struct loom_serve_recognition_service_t {
  // Allocator owning this adapter, request slots, and decoded staging.
  iree_allocator_t host_allocator;
  // Immutable bounds and deployment identity.
  loom_serve_recognition_service_options_t options;
  // Borrowed source metadata and tokenizer; execution belongs to recognition.
  loom_serve_recognition_generator_t generator;
  // Owned stream execution, independent of HTTP upload/decode lifetimes.
  loom_serve_recognition_stream_service_t* streams;
  // Shared workers for finite decode operations.
  iree_task_executor_t* executor;
  // Stable HTTP/native request ledger, including the bounded admission queue.
  loom_serve_recognition_job_t* rows;
  // Number of allocated request ledger slots.
  iree_host_size_t capacity;
  // Rotating input selection; one stalled input cannot monopolize the worker.
  iree_host_size_t cursor;
  // No new decode or source admission is accepted after stopping.
  bool stopping;
  // Composition-owned observations over independent source and decode work.
  struct {
    // Retained storage and its one summary publisher.
    loom_runtime_service_control_binding_t binding;
    // Child work facts copied synchronously after serialized source changes.
    loom_runtime_service_observation_t streams;
    // One independently legal host conversion chunk.
    loom_runtime_service_offer_t offer;
    // Candidate upload, borrowed until the next owner mutation.
    loom_serve_recognition_job_t* queued;
    // Distinguishes reuse of the same request-ledger slot.
    uint64_t request_id;
    // Accepted CPU conversion, retained through its final release callback.
    loom_runtime_service_operation_t operation;
    // Monotonic decode identity, independent of source operation identities.
    uint64_t next_serial;
    // Last published decode lifetime, avoiding unchanged atomic stores.
    uint64_t published_serial;
    // Combined request/work facts, published only after owner transfers finish.
    loom_runtime_service_observation_t observation;
  } scheduling;
  // Reused owner-side response storage; transport copies before the next use.
  iree_string_builder_t response;
  // Borrowed registration-wide history, or NULL when observation is omitted.
  loom_runtime_telemetry_stream_t* telemetry;
  // Cumulative outcomes independent of bounded history retention.
  loom_service_telemetry_counters_t counters;
  // Receipt-to-completion observations independent of event history.
  loom_service_telemetry_latency_t latency;
  // One accepted conversion operation, separate from the typed model CPU owner.
  struct {
    // Intrusive storage reused only after the task's final release callback.
    iree_task_process_t process;
    // Borrowed stable upload, held through release publication.
    loom_serve_recognition_job_t* row;
  } cpu;
  // Final task publication joins observer replacement and adapter destruction.
  struct {
    // Serializes completion publication with the serving owner.
    iree_slim_mutex_t mutex;
    // Nonblocking owner wake installed before the first task.
    loom_runtime_wake_callback_t observer;
    // The accepted conversion task no longer borrows its row or process
    // storage.
    bool ready;
    // Owned conversion failure, published before final release.
    iree_status_t status;
  } events;
};

static void loom_serve_recognition_publish(
    loom_serve_recognition_service_t* service);

static void loom_serve_recognition_streams_changed(
    void* self, const loom_runtime_service_observation_t* observation) {
  loom_serve_recognition_service_t* service = self;
  service->scheduling.streams = *observation;
}

static void loom_serve_recognition_request_end(
    loom_serve_recognition_service_t* service,
    const loom_serve_recognition_job_t* row,
    loom_service_telemetry_outcome_t outcome,
    const iree_status_t borrowed_status) {
  const iree_time_t now = iree_time_now();
  loom_service_telemetry_count_outcome(&service->counters, outcome);
  loom_service_telemetry_latency_observe(&service->latency.outcomes[outcome],
                                         now - row->arrival);
  if (service->telemetry) {
    loom_runtime_telemetry_record_t record = {
        .time_ns = now,
        .request = row->id,
        .event = &LOOM_SERVICE_TELEMETRY_REQUEST_END};
    loom_service_telemetry_end_record_t telemetry_payload = {0};
    telemetry_payload.outcome = outcome;
    telemetry_payload.duration_ns = now - row->arrival;
    loom_service_telemetry_diagnostic_capture(borrowed_status,
                                              &telemetry_payload.diagnostic);
    memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
    loom_runtime_telemetry_stream_append(service->telemetry, &record);
    if (service->generator.parameter_activity) {
      const loom_runtime_parameter_activity_t activity =
          service->generator.parameter_activity(service->generator.self);
      loom_runtime_parameter_record_activity(service->telemetry, now,
                                             &activity);
    }
  }
}

static loom_serve_recognition_job_t loom_serve_recognition_request_begin(
    loom_serve_recognition_service_t* service) {
  const loom_serve_recognition_job_t job = {.owner = service,
                                            .id = ++service->counters.requests,
                                            .arrival = iree_time_now()};
  if (service->telemetry) {
    const loom_runtime_telemetry_record_t record = {
        .time_ns = job.arrival,
        .request = job.id,
        .event = &LOOM_SERVICE_TELEMETRY_REQUEST_BEGIN};
    loom_runtime_telemetry_stream_append(service->telemetry, &record);
  }
  return job;
}

static void loom_serve_recognition_native_end(
    void* self, loom_service_telemetry_outcome_t outcome,
    const iree_status_t status) {
  loom_serve_recognition_job_t* job = self;
  loom_serve_recognition_request_end(job->owner, job, outcome, status);
  memset(job, 0, sizeof(*job));
}

static iree_status_t loom_serve_recognition_try_open(
    void* self, const loom_serve_recognition_request_t* request,
    loom_serve_recognition_stream_t** out_stream) {
  loom_serve_recognition_service_t* service = self;
  *out_stream = NULL;
  if (service->stopping) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "recognition service is stopping");
  }
  loom_serve_recognition_job_t* available = NULL;
  for (iree_host_size_t i = 0; i < service->capacity; ++i) {
    loom_serve_recognition_job_t* job = &service->rows[i];
    if (job->connection && !job->stream) {
      return iree_ok_status();
    }
    if (!job->id && !available) {
      available = job;
    }
  }
  if (!available) {
    return iree_ok_status();
  }
  const loom_serve_recognition_completion_t completion = {
      .self = available, .end = loom_serve_recognition_native_end};
  IREE_RETURN_IF_ERROR(loom_serve_recognition_stream_service_try_open(
      service->streams, request, completion, out_stream));
  if (*out_stream) {
    *available = loom_serve_recognition_request_begin(service);
    available->stream = *out_stream;
    loom_serve_recognition_publish(service);
  }
  return iree_ok_status();
}

// The CPU owner has retired before this relinquishes borrowed upload bytes.
static void loom_serve_transcription_release(
    loom_serve_recognition_service_t* service,
    loom_serve_recognition_job_t* row, bool finish,
    loom_service_telemetry_outcome_t outcome,
    const iree_status_t borrowed_status) {
  loom_serve_recognition_request_end(service, row, outcome, borrowed_status);
  if (row->stream) {
    loom_serve_recognition_stream_close(row->stream);
  }
  if (row->request.audio.sample_rate) {
    loom_serve_audio_resampler_deinitialize(&row->input.pcm.resampler);
    loom_serve_audio_resampler_filter_deinitialize(&row->input.pcm.filter);
  } else {
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
    loom_serve_recognition_service_t* service,
    loom_serve_recognition_job_t* row, loom_service_telemetry_outcome_t outcome,
    const iree_status_t borrowed_status) {
  if (loom_serve_http_connection_failed(row->connection) ||
      !loom_serve_http_connection_can_send(row->connection)) {
    loom_serve_transcription_release(
        service, row, false, LOOM_SERVICE_TELEMETRY_CANCELLED, borrowed_status);
    return;
  }
  iree_status_t status = loom_serve_http_connection_send(
      row->connection, iree_string_builder_view(&service->response));
  if (iree_status_is_ok(status)) {
    loom_serve_transcription_release(service, row, true, outcome,
                                     borrowed_status);
  } else {
    loom_serve_transcription_release(service, row, false,
                                     LOOM_SERVICE_TELEMETRY_FAILED, status);
    iree_status_free(status);
  }
}

static iree_status_t loom_serve_transcription_reject(
    loom_serve_recognition_service_t* service,
    loom_serve_recognition_job_t* row, const char* code,
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
    loom_serve_transcription_send(service, row, LOOM_SERVICE_TELEMETRY_REJECTED,
                                  borrowed_status);
  } else {
    loom_serve_transcription_release(service, row, false,
                                     LOOM_SERVICE_TELEMETRY_FAILED, status);
  }
  return status;
}

static iree_status_t loom_serve_transcription_fail(
    loom_serve_recognition_service_t* service,
    loom_serve_recognition_job_t* row, const iree_status_t borrowed_status) {
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

static iree_status_t loom_serve_transcription_read_pcm(
    loom_serve_recognition_service_t* service,
    loom_serve_recognition_job_t* row) {
  const loom_serve_recognition_format_t format = service->generator.format;
  if (row->request.audio.sample_rate == format.sample_rate) {
    const uint64_t remaining =
        row->request.audio.frame_count - row->input.pcm.offset;
    row->frame_count = iree_min(service->options.decode_frames, remaining);
    IREE_RETURN_IF_ERROR(
        loom_serve_audio_source_read(&row->request.audio, row->input.pcm.offset,
                                     row->frame_count, row->samples));
    row->input.pcm.offset += row->frame_count;
    row->decoded = row->input.pcm.offset == row->request.audio.frame_count;
    return iree_ok_status();
  }
  loom_serve_audio_resampler_t* resampler = &row->input.pcm.resampler;
  if (!resampler->samples) {
    IREE_RETURN_IF_ERROR(loom_serve_audio_resampler_filter_initialize(
        row->request.audio.sample_rate, format.sample_rate,
        service->options.conversion_limit, service->host_allocator,
        &row->input.pcm.filter));
    IREE_RETURN_IF_ERROR(loom_serve_audio_resampler_initialize(
        &row->input.pcm.filter, format.channels, service->options.decode_frames,
        service->options.conversion_limit - row->input.pcm.filter.storage_size,
        service->host_allocator, resampler));
  }
  iree_host_size_t received = 0;
  row->frame_count = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         row->frame_count < service->options.decode_frames) {
    row->frame_count += loom_serve_audio_resampler_read(
        resampler, service->options.decode_frames - row->frame_count,
        row->samples + row->frame_count * format.channels);
    row->decoded = loom_serve_audio_resampler_is_complete(resampler);
    if (row->decoded || row->frame_count == service->options.decode_frames ||
        received == service->options.decode_frames) {
      break;
    }
    iree_host_size_t capacity = 0;
    float* input = loom_serve_audio_resampler_reserve(resampler, &capacity);
    const uint64_t remaining =
        row->request.audio.frame_count - row->input.pcm.offset;
    const iree_host_size_t count =
        iree_min(remaining,
                 iree_min(capacity, service->options.decode_frames - received));
    status = loom_serve_audio_source_read(&row->request.audio,
                                          row->input.pcm.offset, count, input);
    if (iree_status_is_ok(status)) {
      loom_serve_audio_resampler_commit(resampler, count);
      row->input.pcm.offset += count;
      received += count;
      if (row->input.pcm.offset == row->request.audio.frame_count) {
        loom_serve_audio_resampler_finish(resampler);
      }
    }
  }
  return status;
}

static iree_status_t loom_serve_transcription_cpu_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_recognition_service_t* service = process->user_data;
  loom_serve_recognition_job_t* row = service->cpu.row;
  result->completed = true;
  result->did_work = true;
  if (row->request.audio.sample_rate) {
    IREE_RETURN_IF_ERROR(loom_serve_transcription_read_pcm(service, row));
  } else {
    if (!row->input.decoder) {
      IREE_RETURN_IF_ERROR(loom_serve_audio_decoder_allocate(
          row->request.audio.data, service->generator.format.sample_rate,
          service->generator.format.channels, service->host_allocator,
          &row->input.decoder));
    }
    IREE_RETURN_IF_ERROR(loom_serve_audio_decoder_read(
        row->input.decoder, service->options.decode_frames, row->samples,
        &row->frame_count, &row->decoded));
  }
  // Decoded native F32 representation becomes the source ABI's F32LE at
  // this producer boundary. Little-endian targets eliminate these stores.
  for (iree_host_size_t i = 0;
       i < row->frame_count * service->generator.format.channels; ++i) {
    iree_unaligned_store_le_f32(row->samples + i, row->samples[i]);
  }
  return iree_ok_status();
}

static void loom_serve_transcription_cpu_complete(iree_task_process_t* process,
                                                  iree_status_t status) {
  loom_serve_recognition_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.status = status;
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_transcription_cpu_release(iree_task_process_t* process) {
  loom_serve_recognition_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.ready = true;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

// Only the owner reads this predicate, excluding the row borrowed by decoding.
static bool loom_serve_transcription_can_decode(
    const loom_serve_recognition_job_t* row) {
  return row->connection && row->stream && !row->decoded && !row->frame_count &&
         !loom_serve_recognition_stream_is_complete(row->stream);
}

static void loom_serve_transcription_decode(
    loom_serve_recognition_service_t* service,
    loom_serve_recognition_job_t* row) {
  service->cursor = (row - service->rows + 1) % service->capacity;
  service->cpu.row = row;
  iree_task_process_initialize(loom_serve_transcription_cpu_run, 0, 1,
                               &service->cpu.process);
  service->cpu.process.user_data = service;
  service->cpu.process.completion_fn = loom_serve_transcription_cpu_complete;
  service->cpu.process.release_fn = loom_serve_transcription_cpu_release;
  iree_task_executor_schedule_process(service->executor, &service->cpu.process);
}

static void loom_serve_recognition_publish(
    loom_serve_recognition_service_t* service) {
  loom_serve_recognition_job_t* queued = NULL;
  uint64_t owned_count = 0, waiting_count = 0;
  for (iree_host_size_t i = 0; service->rows && i < service->capacity; ++i) {
    const iree_host_size_t index = (service->cursor + i) % service->capacity;
    loom_serve_recognition_job_t* row = &service->rows[index];
    if (!row->id) {
      continue;
    }
    ++owned_count;
    // A request may need decoding and source work simultaneously. Count its
    // ledger receipt once, without reading a live CPU writer's frame fields.
    const bool live = !service->stopping &&
                      (!row->connection ||
                       !loom_serve_http_connection_failed(row->connection));
    const bool decode = live && row != service->cpu.row &&
                        loom_serve_transcription_can_decode(row);
    waiting_count +=
        live &&
        (!row->stream ||
         loom_serve_recognition_stream_is_waiting(row->stream) || decode);
    if (decode && !service->cpu.row && !queued) {
      queued = row;
    }
  }
  const uint64_t request_id = queued ? queued->id : 0;
  loom_runtime_service_offer_t* offer = &service->scheduling.offer;
  loom_runtime_service_control_t* control = service->scheduling.binding.control;
  if (request_id != service->scheduling.request_id) {
    IREE_ASSERT(offer->key.revision != UINT64_MAX);
    ++offer->key.revision;
    offer->work = queued ? LOOM_RUNTIME_SERVICE_WORK_PREPARATION : 0;
    offer->choice_count = queued ? 1 : 0;
    service->scheduling.request_id = request_id;
    const loom_runtime_service_choice_publication_t choice = {
        .choice = {
            .instance_ordinal = service->scheduling.binding.instance_ordinal,
            .request_count = queued ? 1 : 0}};
    loom_runtime_service_publish_offer(control, offer, &choice);
  }
  service->scheduling.queued = queued;
  loom_runtime_service_operation_t* operation = &service->scheduling.operation;
  if (operation->key.serial != service->scheduling.published_serial) {
    operation->waits =
        operation->key.serial ? LOOM_RUNTIME_SERVICE_WAIT_CPU : 0;
    loom_runtime_service_publish_operation(control, operation, NULL);
    service->scheduling.published_serial = operation->key.serial;
  }
  loom_runtime_service_observation_t observation = service->scheduling.streams;
  observation.owned_request_count = owned_count;
  observation.waiting_request_count = waiting_count;
  observation.offer_count += offer->choice_count;
  observation.pending_operation_count += operation->key.serial != 0;
  const loom_runtime_service_observation_t* published =
      &service->scheduling.observation;
  if (observation.owned_request_count != published->owned_request_count ||
      observation.waiting_request_count != published->waiting_request_count ||
      observation.offer_count != published->offer_count ||
      observation.pending_operation_count !=
          published->pending_operation_count ||
      observation.offered_devices != published->offered_devices ||
      observation.accepted_devices != published->accepted_devices) {
    service->scheduling.binding.publish_observation(
        service->scheduling.binding.publisher, &observation);
    service->scheduling.observation = observation;
  }
}

static iree_status_t loom_serve_transcription_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_runtime_service_flags_t flags) {
  loom_serve_recognition_service_t* service = self;
  loom_serve_recognition_job_t pending =
      loom_serve_recognition_request_begin(service);
  pending.connection = connection;
  if (service->stopping || (flags & LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
    service->stopping = true;
    loom_serve_recognition_stream_service_stop(service->streams);
    loom_serve_recognition_publish(service);
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
  if (iree_status_is_ok(status) &&
      pending.request.normalized_frame_count > service->generator.frame_limit) {
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
    if (!service->rows[i].id) {
      service->rows[i] = pending;
      loom_serve_recognition_publish(service);
      return iree_ok_status();
    }
  }
  return loom_serve_transcription_reject(
      service, &pending, "503 Service Unavailable", iree_ok_status(),
      IREE_SV("transcription admission queue is full"));
}

static iree_status_t loom_serve_transcription_deliver(
    loom_serve_recognition_service_t* service,
    loom_serve_recognition_job_t* row) {
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
    loom_serve_transcription_send(
        service, row, LOOM_SERVICE_TELEMETRY_COMPLETED, iree_ok_status());
  } else {
    loom_serve_transcription_release(service, row, false,
                                     LOOM_SERVICE_TELEMETRY_FAILED, status);
  }
  return status;
}

iree_status_t loom_serve_recognition_service_progress(
    loom_serve_recognition_service_t* service,
    loom_runtime_service_flags_t flags, bool* out_progress) {
  *out_progress = false;
  service->stopping |= (flags & LOOM_RUNTIME_SERVICE_FLAG_STOPPING) != 0;
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
    loom_serve_recognition_job_t* row = service->cpu.row;
    service->cpu.row = NULL;
    service->scheduling.operation.key.serial = 0;
    *out_progress = true;
    if (!iree_status_is_ok(cpu_status)) {
      if (service->stopping ||
          loom_serve_http_connection_failed(row->connection)) {
        loom_serve_transcription_release(
            service, row, false, LOOM_SERVICE_TELEMETRY_CANCELLED, cpu_status);
      } else {
        status = loom_serve_transcription_fail(service, row, cpu_status);
      }
      iree_status_free(cpu_status);
    }
  }
  for (iree_host_size_t i = 0;
       i < service->capacity && iree_status_is_ok(status); ++i) {
    loom_serve_recognition_job_t* row = &service->rows[i];
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
                                         LOOM_SERVICE_TELEMETRY_CANCELLED,
                                         iree_ok_status());
        *out_progress = true;
      }
      continue;
    }
    if (!row->stream || row == service->cpu.row) {
      continue;
    }
    // A source failure stops input before asynchronous release completes.
    // Preserve that result instead of replacing it with a late-write error.
    iree_status_t stream_status =
        loom_serve_recognition_stream_status(row->stream);
    if (!iree_status_is_ok(stream_status)) {
      status = loom_serve_transcription_fail(service, row, stream_status);
      iree_status_free(stream_status);
      *out_progress = true;
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
    loom_serve_recognition_job_t* queued = NULL;
    for (iree_host_size_t i = 0; i < service->capacity; ++i) {
      loom_serve_recognition_job_t* row = &service->rows[i];
      if (row->connection && !row->stream && row != service->cpu.row &&
          (!queued || row->id < queued->id)) {
        queued = row;
      }
    }
    if (!queued) {
      break;
    }
    iree_status_t admission = loom_serve_recognition_stream_service_try_open(
        service->streams, &queued->request.request,
        (loom_serve_recognition_completion_t){0}, &queued->stream);
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
    status = loom_serve_recognition_stream_service_progress(
        service->streams,
        flags | (service->stopping ? LOOM_RUNTIME_SERVICE_FLAG_STOPPING : 0),
        &progress);
    *out_progress |= progress;
  }
  if (!iree_status_is_ok(status)) {
    service->stopping = true;
    loom_serve_recognition_stream_service_stop(service->streams);
  }
  loom_serve_recognition_publish(service);
  return status;
}

static iree_status_t loom_serve_recognition_try_schedule(
    void* self, loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags,
    loom_runtime_service_schedule_outcome_t* out_outcome) {
  loom_serve_recognition_service_t* service = self;
  *out_outcome = (loom_runtime_service_schedule_outcome_t){0};
  if (service->stopping ||
      iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (offer.ordinal == service->scheduling.binding.offer_ordinal) {
    status = loom_serve_recognition_stream_service_try_schedule(
        service->streams, offer, choice_ordinal, flags,
        &out_outcome->operation);
  } else if (offer.ordinal == service->scheduling.offer.key.ordinal &&
             offer.revision == service->scheduling.offer.key.revision &&
             choice_ordinal < service->scheduling.offer.choice_count) {
    loom_serve_recognition_publish(service);
    if (offer.revision == service->scheduling.offer.key.revision) {
      IREE_ASSERT(service->scheduling.next_serial != UINT64_MAX);
      service->scheduling.operation = (loom_runtime_service_operation_t){
          .key = {.ordinal = service->scheduling.binding.operation_ordinal + 2,
                  .serial = ++service->scheduling.next_serial},
          .origin = offer,
          .work = LOOM_RUNTIME_SERVICE_WORK_PREPARATION};
      out_outcome->operation = service->scheduling.operation.key;
      loom_serve_transcription_decode(service, service->scheduling.queued);
    }
  }
  out_outcome->accepted = out_outcome->operation.serial != 0;
  out_outcome->status_code = iree_status_code(status);
  loom_serve_recognition_publish(service);
  return status;
}

static iree_status_t loom_serve_recognition_advance(
    void* self, loom_runtime_service_flags_t flags, bool* out_progress) {
  loom_serve_recognition_service_t* service = self;
  IREE_RETURN_IF_ERROR(
      loom_serve_recognition_service_progress(service, flags, out_progress));
  iree_status_t status = iree_ok_status();
  for (uint64_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    loom_runtime_service_offer_t offer;
    if (loom_runtime_service_read_offer(
            service->scheduling.binding.control,
            service->scheduling.binding.offer_ordinal + i,
            &offer) == LOOM_RUNTIME_SERVICE_READ_COMPLETE &&
        offer.choice_count) {
      loom_runtime_service_schedule_outcome_t outcome;
      status = loom_serve_recognition_try_schedule(self, offer.key, 0, flags,
                                                   &outcome);
      *out_progress |= outcome.accepted;
    }
  }
  return status;
}

static void loom_serve_recognition_set_observer(
    void* self, loom_runtime_wake_callback_t observer) {
  loom_serve_recognition_service_t* service = self;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.observer = observer;
  iree_slim_mutex_unlock(&service->events.mutex);
  loom_serve_recognition_stream_service_set_observer(service->streams,
                                                     observer);
}

static bool loom_serve_recognition_is_active(void* self) {
  loom_serve_recognition_service_t* service = self;
  return service->cpu.row ||
         loom_serve_recognition_stream_service_is_active(service->streams);
}

static bool loom_serve_recognition_has_requests(void* self) {
  loom_serve_recognition_service_t* service = self;
  for (iree_host_size_t i = 0; i < service->capacity; ++i) {
    if (service->rows[i].id) {
      return true;
    }
  }
  return false;
}

static bool loom_serve_recognition_is_pending(void* self) {
  return loom_serve_recognition_has_requests(self) ||
         loom_serve_recognition_is_active(self);
}

static loom_service_snapshot_t loom_serve_recognition_snapshot(void* self) {
  loom_serve_recognition_service_t* service = self;
  loom_service_snapshot_t snapshot = {.phase = "idle"};
  const iree_time_t now = iree_time_now();
  for (iree_host_size_t i = 0; i < service->capacity; ++i) {
    const loom_serve_recognition_job_t* row = &service->rows[i];
    if (!row->id) {
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

loom_serve_service_t loom_serve_recognition_service_interface(
    loom_serve_recognition_service_t* service) {
  static const loom_runtime_service_vtable_t vtable = {
      .set_observer = loom_serve_recognition_set_observer,
      .has_requests = loom_serve_recognition_has_requests,
      .is_pending = loom_serve_recognition_is_pending,
      .is_active = loom_serve_recognition_is_active,
      .advance = loom_serve_recognition_advance,
      .try_schedule = loom_serve_recognition_try_schedule,
  };
  static const iree_string_view_t input[] = {IREE_SVL("audio")};
  static const iree_string_view_t output[] = {IREE_SVL("text"),
                                              IREE_SVL("activity")};
  static const iree_string_view_list_t outputs[] = {
      [LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT] = {1, output},
      [LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY] = {1, output + 1},
      [LOOM_SERVE_RECOGNITION_OUTPUT_ALL] = {2, output},
  };
  return (loom_serve_service_t){
      .name = service->options.name,
      .runtime = {.self = service,
                  .vtable = &vtable,
                  .control = service->scheduling.binding.control,
                  .capacity_domain = service->generator.capacity_domain},
      .accept = service->options.kind == LOOM_SERVICE_KIND_RECOGNITION
                    ? loom_serve_transcription_accept
                    : NULL,
      .try_open_recognition = loom_serve_recognition_try_open,
      .tokenizer = service->generator.tokenizer,
      .kind = service->options.kind,
      .endpoint = loom_serve_service_endpoint(service->options.kind),
      .capabilities = {.input_modalities = {IREE_ARRAYSIZE(input), input},
                       .output_modalities =
                           outputs[service->generator.outputs]},
      .snapshot = loom_serve_recognition_snapshot,
      .telemetry = {.stream = service->telemetry,
                    .counters = &service->counters,
                    .latency = &service->latency}};
}

void loom_serve_recognition_service_free(
    loom_serve_recognition_service_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!service->cpu.row);
  loom_serve_recognition_stream_service_free(service->streams);
  for (iree_host_size_t i = 0; service->rows && i < service->capacity; ++i) {
    IREE_ASSERT(!service->rows[i].id);
  }
  service->stopping = true;
  loom_serve_recognition_publish(service);
  iree_string_builder_deinitialize(&service->response);
  iree_status_free(service->events.status);
  iree_slim_mutex_deinitialize(&service->events.mutex);
  iree_allocator_free(service->host_allocator, service->rows);
  iree_allocator_free_aligned(service->host_allocator, service);
}

iree_status_t loom_serve_recognition_service_allocate(
    loom_runtime_host_t* host, loom_serve_recognition_generator_t generator,
    const loom_serve_recognition_service_options_t* options,
    loom_runtime_telemetry_stream_t* telemetry,
    loom_runtime_service_control_binding_t scheduling,
    iree_allocator_t host_allocator,
    loom_serve_recognition_service_t** out_service) {
  *out_service = NULL;
  if (options->kind != LOOM_SERVICE_KIND_RECOGNITION &&
      options->kind != LOOM_SERVICE_KIND_ACTIVITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recognition owner requires recognition or "
                            "activity deployment kind");
  }
  const loom_serve_recognition_output_flags_t required =
      options->kind == LOOM_SERVICE_KIND_RECOGNITION
          ? LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT
          : LOOM_SERVE_RECOGNITION_OUTPUT_ACTIVITY;
  if (!(generator.outputs & required)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source outputs do not support '%s' deployment",
                            loom_service_kind_name(options->kind));
  }
  if (!options->name.size || !options->pending_capacity ||
      !options->decode_frames || !options->conversion_limit ||
      !generator.format.channels ||
      options->decode_frames >
          SIZE_MAX / sizeof(float) / generator.format.channels ||
      generator.request_capacity >
          SIZE_MAX / sizeof(loom_serve_recognition_job_t) ||
      options->pending_capacity >
          SIZE_MAX / sizeof(loom_serve_recognition_job_t) -
              generator.request_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "recognition requires a name and bounded request/decode capacity and "
        "conversion storage");
  }
  if (!scheduling.control || !scheduling.publish_observation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recognition requires retained scheduling storage");
  }
  const loom_runtime_service_control_layout_t layout =
      loom_runtime_service_control_layout(scheduling.control);
  if (scheduling.offer_ordinal > layout.offer_capacity ||
      layout.offer_capacity - scheduling.offer_ordinal < 2 ||
      scheduling.operation_ordinal > layout.operation_capacity ||
      layout.operation_capacity - scheduling.operation_ordinal < 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recognition owner needs two offers and three "
                            "operation slots");
  }
  loom_serve_recognition_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*service),
      iree_alignof(loom_serve_recognition_service_t), 0, (void**)&service));
  service->host_allocator = host_allocator;
  service->options = *options;
  service->generator = generator;
  service->scheduling.binding = scheduling;
  service->scheduling.offer.key.ordinal = scheduling.offer_ordinal + 1;
  service->scheduling.operation.key.ordinal = scheduling.operation_ordinal + 2;
  service->executor = loom_runtime_host_executor(host);
  service->telemetry = telemetry;
  service->capacity = generator.request_capacity + options->pending_capacity;
  iree_slim_mutex_initialize(&service->events.mutex);
  iree_string_builder_initialize(host_allocator, &service->response);
  const loom_serve_recognition_stream_service_options_t recognition_options = {
      .name = options->name,
      .request_limit = options->request_limit,
      .document_limit = options->document_limit};
  loom_runtime_service_control_binding_t stream_binding = scheduling;
  stream_binding.publish_observation = loom_serve_recognition_streams_changed;
  stream_binding.publisher = service;
  iree_status_t status = loom_serve_recognition_stream_service_allocate(
      host, generator, &recognition_options, stream_binding, host_allocator,
      &service->streams);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, service->capacity,
                                         sizeof(*service->rows),
                                         (void**)&service->rows);
  }
  if (iree_status_is_ok(status)) {
    *out_service = service;
  } else {
    loom_serve_recognition_service_free(service);
  }
  return status;
}
