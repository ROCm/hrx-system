// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/image/service.h"

#include <inttypes.h>
#include <string.h>

#include "iree/async/operations/semaphore.h"
#include "iree/base/internal/base64.h"
#include "iree/base/threading/mutex.h"
#include "iree/vm/buffer.h"
#include "loom/serve/runtime/host/clock.h"
#include "loom/serve/runtime/host/timer.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/runtime/parameter/telemetry.h"
#include "loom/serve/service/execution_reconfiguration.h"
#include "loom/serve/service/image/output.h"
#include "loom/serve/service/image/telemetry.h"
#include "loom/serve/service/kind.h"
#include "loom/serve/telemetry/tracing.h"
#include "loom/util/json.h"

typedef struct loom_serve_image_job_t {
  // Monotonic process-local request identity.
  uint64_t id;
  // Owner-only response destination; NULL after peer reset.
  loom_serve_http_connection_t* connection;
  // Immutable monotonic receipt time, preserved through queue and invocation.
  iree_time_t arrival_time;
  // Owned decoded input, stable until the CPU preparation task retires.
  loom_serve_image_request_t request;
  // Request work returned by CPU tasks through their existing release handoff.
  loom_serve_image_work_t work;
  // Owned source bytes and compatibility facts, published by CPU preparation.
  loom_serve_prepared_request_t prepared;
  // Owned input rejection, published by CPU preparation.
  iree_status_t status;
  // Stable byte offset in the complete cohort RGB result.
  iree_host_size_t output_offset;
  // Protocol-validated F32 CHW byte length for this request.
  iree_host_size_t output_length;
  // Successfully completed model command parts, excluding cancellation cleanup.
  uint64_t completed_parts;
} loom_serve_image_job_t;

typedef enum loom_serve_image_phase_e {
  IMAGE_PHASE_IDLE = 0,
  IMAGE_PHASE_PREPARING,
  IMAGE_PHASE_BATCHING,
  IMAGE_PHASE_SPECIALIZING,
  IMAGE_PHASE_WAITING,
  IMAGE_PHASE_READY,
  IMAGE_PHASE_GENERATING,
  IMAGE_PHASE_ENCODING,
} loom_serve_image_phase_t;

static const char* const image_phases[] = {
    "idle",    "preparing", "batching",   "specializing",
    "waiting", "ready",     "generating", "encoding"};

enum { IMAGE_EVENT_CPU = 1u << 0, IMAGE_EVENT_DEVICE = 1u << 1 };

struct loom_serve_image_service_t {
  // Allocator for queued inputs and encoded response storage.
  iree_allocator_t host_allocator;
  // Borrowed, serialized native model invocation.
  loom_serve_image_generator_t generator;
  // Candidate ownership and pending selection, retired before the generator.
  loom_serve_execution_reconfiguration_t* reconfiguration;
  // Copied model geometry and policy; the route name is borrowed.
  loom_serve_image_service_options_t options;
  // Borrowed application executor, shared with JIT and all CPU/model work.
  iree_task_executor_t* executor;
  // Borrowed host proactor for model multi-waits and heartbeat timers.
  iree_async_proactor_t* proactor;
  // Current invocation phase, mutated only by scheduler turns.
  loom_serve_image_phase_t phase;
  // Stops admission while accepted tasks and model work finish naturally.
  bool stopping;
  // Owned failure accumulated on a turn, transferred to the router on return.
  iree_status_t status;
  // Total host request credit shared by queued and active requests.
  iree_host_size_t capacity;
  // Owned jobs across the pending queue and active cohort.
  iree_host_size_t count;
  // Completion publication shared by CPU workers, proactor and scheduler.
  struct {
    // Joins publication callbacks' final access before owner destruction.
    iree_slim_mutex_t mutex;
    // Borrowed router wake, joined by mutex when detached after retirement.
    loom_serve_wake_callback_t observer;
    // CPU/device completions awaiting consumption on the scheduler.
    uint32_t ready;
    // Joined asynchronous outcomes awaiting consumption by the scheduler.
    iree_status_t status;
    // CPU result consumed only with its release event, before classifying input
    // rejection separately from terminal device/timer/encoding failures.
    iree_status_t cpu_status;
  } events;
  // Exclusive finite CPU job, reused only after the process release callback.
  struct {
    // Source preparation or image encoding, never device waiting.
    iree_task_process_t process;
    // Immutable operation selector while the CPU process is live.
    loom_serve_image_phase_t phase;
    // Actual CPU preparation/capture work, counted once per released task.
    iree_duration_t elapsed_ns;
    // Borrowed successful model output, retained through encoding.
    iree_const_byte_span_t rgb;
    // Owned complete HTTP result, independent of health/metadata responses.
    iree_string_builder_t response;
  } cpu;
  // Intrusive proactor wait storage, never reused before its final callback.
  struct {
    // Waits for both model work and feedback without occupying a task worker.
    iree_async_semaphore_wait_operation_t operation;
    // Borrowed HAL handles and values from the accepted model invocation.
    loom_serve_execution_frontier_t frontier;
    // Toll-free async semaphore handles used by the public proactor API.
    iree_async_semaphore_t* semaphores[2];
    // The accepted part releases invocation resources at its final frontier.
    bool complete;
    // The accepted part performs cancellation cleanup instead of model work.
    bool cancelled;
    // Accepted model part through readiness; cancellation cleanup is untraced.
    IREE_TRACE(iree_section_id_t section;)
  } device;
  // Periodic readiness and owned cancellation, independent of device waits.
  loom_serve_timer_t timer;
  // Bounded FIFO of owned host requests, excluding the preparation/cohort
  // snapshot.
  loom_serve_image_job_t** pending;
  // Number of occupied FIFO entries.
  iree_host_size_t pending_count;
  // Stable preparation snapshot or selected cohort, never compacted while
  // borrowed.
  struct {
    // Owned jobs in source packing order.
    loom_serve_image_job_t** jobs;
    // Snapshot/cohort length, including cancelled slots until retirement.
    iree_host_size_t count;
    // Next terminal encoding slot; earlier jobs have already retired.
    iree_host_size_t next;
    // Borrowed prepared values, contiguous for source batch capture.
    loom_serve_prepared_request_t* requests;
    // Protocol-validated byte lengths in the same source packing order.
    iree_host_size_t* output_lengths;
    // Copied source selection, published by CPU batch capture.
    loom_serve_image_prepared_t prepared;
    // Borrowed complete model output, retained through the final CPU encoder.
    iree_const_byte_span_t rgb;
  } cohort;
  // Start of the current admission wait or submitted device execution.
  iree_time_t phase_start;
  // Single-owner observation, independent of CPU borrowers and log readers.
  struct {
    // Registration-borrowed or service-owned history through retirement.
    loom_serve_telemetry_stream_t* stream;
    // Only standalone services destroy their history with the producer.
    bool owns_stream;
    // Protocol outcomes and completed device work, never reconstructed from
    // logs.
    loom_serve_telemetry_counters_t counters;
    // Request latency distributions retained independently of log readers.
    loom_serve_telemetry_latency_t latency;
    // Image-specific units, deliberately separate from token counters.
    loom_serve_image_totals_t totals;
    // This model's terminal failure, distinct from a sibling's shutdown.
    iree_status_code_t failure_code;
  } telemetry;
  // Reusable application-side HTTP/JSON response storage.
  iree_string_builder_t response;
};

static uint64_t loom_serve_image_active(
    const loom_serve_image_service_t* service) {
  return service->cohort.next < service->cohort.count
             ? service->cohort.jobs[service->cohort.next]->id
             : 0;
}

static bool loom_serve_image_has_peer(
    const loom_serve_image_service_t* service) {
  for (iree_host_size_t i = service->cohort.next; i < service->cohort.count;
       ++i) {
    if (service->cohort.jobs[i]->connection) {
      return true;
    }
  }
  return false;
}

static void loom_serve_image_event(loom_serve_image_service_t* service,
                                   const loom_serve_telemetry_event_t* event,
                                   uint64_t request) {
  const loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(), .request = request, .event = event};
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

static void loom_serve_image_record_weights(loom_serve_image_service_t* service,
                                            iree_time_t time_ns) {
  if (service->generator.weight_activity) {
    const loom_serve_weight_activity_t activity =
        service->generator.weight_activity(service->generator.self);
    loom_serve_telemetry_record_weights(service->telemetry.stream, time_ns,
                                        &activity);
  }
}

// Copies a borrowed status before it propagates or is consumed at a boundary.
static void loom_serve_image_diagnose(loom_serve_image_service_t* service,
                                      uint64_t request,
                                      const iree_status_t failure) {
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .request = request,
                                          .event = &LOOM_SERVE_TELEMETRY_ERROR};
  loom_serve_telemetry_diagnostic_capture(failure, &record.data.error);
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

static void loom_serve_image_request_end(loom_serve_image_service_t* service,
                                         const loom_serve_image_job_t* job,
                                         loom_serve_telemetry_outcome_t outcome,
                                         const iree_status_t borrowed_status) {
  loom_serve_telemetry_count_outcome(&service->telemetry.counters, outcome);
  loom_serve_image_record_work(service->telemetry.stream, job->id, &job->work);
  loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .request = job->id,
      .event = &LOOM_SERVE_TELEMETRY_REQUEST_END};
  record.data.end.outcome = outcome;
  record.data.end.duration_ns = record.time_ns - job->arrival_time;
  loom_serve_telemetry_latency_observe(
      &service->telemetry.latency.outcomes[outcome],
      record.data.end.duration_ns);
  loom_serve_telemetry_diagnostic_capture(borrowed_status,
                                          &record.data.end.diagnostic);
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
  loom_serve_image_record_totals(service->telemetry.stream,
                                 &service->telemetry.totals);
  loom_serve_image_record_weights(service, record.time_ns);
}

static iree_status_t loom_serve_image_quote(iree_string_builder_t* output,
                                            iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static iree_status_t loom_serve_image_response_begin(
    iree_string_builder_t* output, const char* status) {
  iree_string_builder_reset(output);
  return iree_string_builder_append_format(
      output,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      status);
}

// Each connection sends exactly one complete response. The transport copies
// admitted bytes, so a slow or reset peer never holds a generator buffer.
static loom_serve_telemetry_outcome_t loom_serve_image_response_send(
    loom_serve_image_service_t* service,
    loom_serve_http_connection_t* connection, uint64_t request,
    iree_string_view_t response) {
  // A fresh response has no occupied send slots. Zero budget therefore means
  // the carrier is closing; that can precede its failure callback publication.
  if (loom_serve_http_connection_failed(connection) ||
      !loom_serve_http_connection_can_send(connection)) {
    loom_serve_http_connection_abort(connection);
    return LOOM_SERVE_TELEMETRY_CANCELLED;
  }
  iree_status_t status = loom_serve_http_connection_send(connection, response);
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
    return LOOM_SERVE_TELEMETRY_COMPLETED;
  } else {
    const loom_serve_telemetry_outcome_t outcome =
        iree_status_is_cancelled(status) || iree_status_is_unavailable(status)
            ? LOOM_SERVE_TELEMETRY_CANCELLED
            : LOOM_SERVE_TELEMETRY_FAILED;
    loom_serve_image_diagnose(service, request, status);
    iree_status_free(status);
    loom_serve_http_connection_abort(connection);
    return outcome;
  }
}

static iree_status_t loom_serve_image_rejection_response(
    loom_serve_image_service_t* service,
    loom_serve_http_connection_t* connection, uint64_t request,
    const char* code, iree_string_view_t message) {
  iree_status_t status =
      loom_serve_image_response_begin(&service->response, code);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        &service->response,
        "{\"error\":{\"type\":\"image_request_error\",\"message\":");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_image_quote(&service->response, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&service->response, "}}");
  }
  if (iree_status_is_ok(status)) {
    loom_serve_image_response_send(
        service, connection, request,
        iree_string_builder_view(&service->response));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

// Rejections before admission own only a local receipt, not the active job.
static iree_status_t loom_serve_image_reject(
    loom_serve_image_service_t* service,
    loom_serve_http_connection_t* connection, const loom_serve_image_job_t* job,
    const char* code, iree_string_view_t message) {
  const iree_status_t status = loom_serve_image_rejection_response(
      service, connection, job->id, code, message);
  loom_serve_image_request_end(service, job, LOOM_SERVE_TELEMETRY_REJECTED,
                               status);
  return status;
}

static iree_status_t loom_serve_image_result_json(iree_string_builder_t* output,
                                                  iree_const_byte_span_t png) {
  IREE_RETURN_IF_ERROR(loom_serve_image_response_begin(output, "200 OK"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"created\":%" PRId64 ",\"data\":[{\"b64_json\":\"",
      loom_serve_clock_now_unix_seconds()));
  const iree_host_size_t encoded_size =
      iree_base64_encoded_size(png.data_length);
  char* encoded = NULL;
  iree_host_size_t capacity = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      output, encoded_size, &encoded, &capacity));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_base64_encode(
      png, iree_make_mutable_string_view(encoded, capacity), &length));
  iree_string_builder_commit_append(output, length);
  return iree_string_builder_append_cstring(output, "\"}]}");
}

static iree_status_t loom_serve_image_cpu_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_image_service_t* service = process->user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->options.name.data,
                              service->options.name.size);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, image_phases[service->cpu.phase]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0,
      loom_serve_telemetry_stream_identity(service->telemetry.stream).ordinal);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, loom_serve_image_active(service));
  result->completed = true;
  result->did_work = true;
  if (service->cpu.phase == IMAGE_PHASE_PREPARING) {
    const iree_time_t begin = iree_time_now();
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < service->cohort.count && iree_status_is_ok(status); ++i) {
      loom_serve_image_job_t* job = service->cohort.jobs[i];
      if (job->prepared.bytes) {
        continue;
      }
      const iree_time_t request_begin = iree_time_now();
      status = service->generator.prepare(service->generator.self,
                                          &job->request, &job->prepared);
      job->work.prepare_ns = iree_time_now() - request_begin;
      if (iree_status_is_invalid_argument(status)) {
        job->status = status;
        status = iree_ok_status();
      }
    }
    service->cpu.elapsed_ns = iree_time_now() - begin;
    IREE_RETURN_AND_END_ZONE(z0, status);
  }
  if (service->cpu.phase == IMAGE_PHASE_BATCHING) {
    const iree_time_t begin = iree_time_now();
    const iree_status_t status = service->generator.prepare_batch(
        service->generator.self, service->cohort.count,
        service->cohort.requests, service->cohort.output_lengths,
        &service->cohort.prepared);
    service->cpu.elapsed_ns = iree_time_now() - begin;
    IREE_RETURN_AND_END_ZONE(z0, status);
  }
  loom_serve_image_job_t* job = service->cohort.jobs[service->cohort.next];
  const iree_time_t begin = iree_time_now();
  iree_byte_span_t png = iree_byte_span_empty();
  IREE_TRACE_ZONE_BEGIN_NAMED(z_png, "loom_serve_image_encode_png");
  iree_status_t status = loom_serve_image_encode_rgb_f32_png(
      job->request.width, job->request.height, service->cpu.rgb,
      service->host_allocator, &png);
  IREE_TRACE_ZONE_END(z_png);
  const iree_time_t encoded = iree_time_now();
  if (iree_status_is_ok(status)) {
    job->work.png_bytes = png.data_length;
    IREE_TRACE_ZONE_BEGIN_NAMED(z_json, "loom_serve_image_encode_json");
    status = loom_serve_image_result_json(
        &service->cpu.response,
        iree_make_const_byte_span(png.data, png.data_length));
    IREE_TRACE_ZONE_END(z_json);
  }
  iree_allocator_free(service->host_allocator, png.data);
  job->work.png_ns = encoded - begin;
  job->work.json_ns = iree_time_now() - encoded;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_serve_image_cpu_completed(iree_task_process_t* process,
                                           iree_status_t status) {
  loom_serve_image_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.cpu_status = status;
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_image_cpu_released(iree_task_process_t* process) {
  loom_serve_image_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.ready |= IMAGE_EVENT_CPU;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_image_schedule_cpu(loom_serve_image_service_t* service,
                                          loom_serve_image_phase_t phase) {
  service->phase = phase;
  service->cpu.phase = phase;
  iree_task_process_initialize(loom_serve_image_cpu_run, 0, 1,
                               &service->cpu.process);
  service->cpu.process.user_data = service;
  service->cpu.process.completion_fn = loom_serve_image_cpu_completed;
  service->cpu.process.release_fn = loom_serve_image_cpu_released;
  iree_task_executor_schedule_process(service->executor, &service->cpu.process);
}

static void loom_serve_image_device_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  loom_serve_image_service_t* service = user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.status = iree_status_join(service->events.status, status);
  service->events.ready |= IMAGE_EVENT_DEVICE;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static iree_status_t loom_serve_image_service_submit(
    loom_serve_image_service_t* service) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->options.name.data,
                              service->options.name.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0,
      loom_serve_telemetry_stream_identity(service->telemetry.stream).ordinal);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, loom_serve_image_active(service));
  const iree_time_t begin = iree_time_now();
  service->device.cancelled =
      service->stopping || !loom_serve_image_has_peer(service);
  if (service->device.cancelled) {
    loom_serve_execution_permit_cancel(service->generator.permit);
  }
  const loom_serve_image_submit_flags_t flags =
      service->device.cancelled ? LOOM_SERVE_IMAGE_SUBMIT_FLAG_CANCEL : 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, service->generator.submit(service->generator.self, flags,
                                    &service->device.frontier,
                                    &service->device.complete));
  service->phase_start = iree_time_now();
  const iree_duration_t submit_ns = service->phase_start - begin;
  for (iree_host_size_t i = 0; i < service->cohort.count; ++i) {
    service->cohort.jobs[i]->work.submit_ns += submit_ns;
    loom_serve_image_event(service, &LOOM_SERVE_IMAGE_SUBMITTED,
                           service->cohort.jobs[i]->id);
  }
  service->telemetry.totals.work.submit_ns += submit_ns;
  iree_async_semaphore_wait_operation_t* wait = &service->device.operation;
  memset(wait, 0, sizeof(*wait));
  iree_async_operation_initialize(&wait->base,
                                  IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  loom_serve_image_device_completed, service);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    service->device.semaphores[i] =
        (iree_async_semaphore_t*)service->device.frontier.semaphores[i];
  }
  wait->semaphores = service->device.semaphores;
  wait->values = service->device.frontier.values;
  wait->count = 2;
  wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_submit_one(service->proactor, &wait->base));
  service->phase = IMAGE_PHASE_GENERATING;
  IREE_TRACE({
    if (!service->device.cancelled) {
      service->device.section = loom_serve_trace_section_enter(
          LOOM_SERVE_TRACE_SECTION_EXECUTE, service->options.name);
    }
  });
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void loom_serve_image_job_finish(
    loom_serve_image_service_t* service, loom_serve_image_job_t* job,
    loom_serve_telemetry_outcome_t outcome) {
  if (job->connection) {
    if (outcome == LOOM_SERVE_TELEMETRY_COMPLETED) {
      outcome = loom_serve_image_response_send(
          service, job->connection, job->id,
          iree_string_builder_view(&service->cpu.response));
      if (outcome == LOOM_SERVE_TELEMETRY_COMPLETED) {
        loom_serve_telemetry_latency_publish_output(&service->telemetry.latency,
                                                    service->telemetry.stream,
                                                    job->id, job->arrival_time);
      }
    } else {
      loom_serve_http_connection_abort(job->connection);
    }
  } else if (outcome == LOOM_SERVE_TELEMETRY_COMPLETED) {
    outcome = service->telemetry.failure_code ? LOOM_SERVE_TELEMETRY_FAILED
                                              : LOOM_SERVE_TELEMETRY_CANCELLED;
  }
  loom_serve_image_request_end(
      service, job, outcome,
      iree_status_is_ok(job->status) ? service->status : job->status);
  iree_status_free(job->status);
  iree_vm_buffer_release(job->prepared.bytes);
  loom_serve_image_request_deinitialize(&job->request);
  iree_allocator_free(service->host_allocator, job);
  --service->count;
}

static void loom_serve_image_finish(loom_serve_image_service_t* service,
                                    loom_serve_telemetry_outcome_t outcome) {
  for (iree_host_size_t i = service->cohort.next; i < service->cohort.count;
       ++i) {
    loom_serve_image_job_finish(service, service->cohort.jobs[i], outcome);
  }
  service->cohort.count = 0;
  service->cohort.next = 0;
  service->phase = IMAGE_PHASE_IDLE;
}

static void loom_serve_image_reject_cohort(loom_serve_image_service_t* service,
                                           const char* code,
                                           iree_string_view_t message) {
  for (iree_host_size_t i = 0; i < service->cohort.count; ++i) {
    loom_serve_image_job_t* job = service->cohort.jobs[i];
    if (job->connection) {
      service->status = iree_status_join(
          service->status,
          loom_serve_image_rejection_response(service, job->connection, job->id,
                                              code, message));
      job->connection = NULL;
      loom_serve_image_job_finish(service, job, LOOM_SERVE_TELEMETRY_REJECTED);
    } else {
      loom_serve_image_job_finish(service, job, LOOM_SERVE_TELEMETRY_CANCELLED);
    }
  }
  service->cohort.count = 0;
  service->cohort.next = 0;
  service->phase = IMAGE_PHASE_IDLE;
}

// Requeue the prepared snapshot ahead of arrivals that came while CPU work was
// live. Request-local rejection cannot invalidate another request's source
// bytes.
static void loom_serve_image_preparation_complete(
    loom_serve_image_service_t* service) {
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->cohort.count; ++i) {
    loom_serve_image_job_t* job = service->cohort.jobs[i];
    if (!job->connection || service->stopping) {
      loom_serve_image_job_finish(service, job,
                                  service->telemetry.failure_code
                                      ? LOOM_SERVE_TELEMETRY_FAILED
                                      : LOOM_SERVE_TELEMETRY_CANCELLED);
    } else if (!iree_status_is_ok(job->status)) {
      iree_string_view_t message = iree_status_message(job->status);
      if (!message.size) {
        message = IREE_SV("source rejected image input");
      }
      loom_serve_image_diagnose(service, job->id, job->status);
      service->status = iree_status_join(
          service->status,
          loom_serve_image_rejection_response(service, job->connection, job->id,
                                              "400 Bad Request", message));
      job->connection = NULL;
      loom_serve_image_job_finish(service, job, LOOM_SERVE_TELEMETRY_REJECTED);
    } else {
      service->cohort.jobs[retained++] = job;
    }
  }
  memmove(service->pending + retained, service->pending,
          service->pending_count * sizeof(*service->pending));
  memcpy(service->pending, service->cohort.jobs,
         retained * sizeof(*service->pending));
  service->pending_count += retained;
  service->cohort.count = 0;
  service->phase = IMAGE_PHASE_IDLE;
}

static void loom_serve_image_select(loom_serve_image_service_t* service) {
  uint64_t remaining = service->generator.batch_limit;
  const uint64_t tag = service->pending[0]->prepared.tag;
  const iree_time_t now = iree_time_now();
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    loom_serve_image_job_t* job = service->pending[i];
    if (job->prepared.bytes && job->prepared.tag == tag &&
        job->prepared.units <= remaining) {
      remaining -= job->prepared.units;
      const iree_host_size_t index = service->cohort.count++;
      service->cohort.jobs[index] = job;
      service->cohort.requests[index] = job->prepared;
      service->cohort.output_lengths[index] = job->output_length;
      job->work.queue_ns = now - job->arrival_time - job->work.prepare_ns;
      service->telemetry.totals.work.queue_ns += job->work.queue_ns;
      loom_serve_image_event(service, &LOOM_SERVE_IMAGE_STARTED, job->id);
    } else {
      service->pending[retained++] = job;
    }
  }
  service->pending_count = retained;
  loom_serve_image_schedule_cpu(service, IMAGE_PHASE_BATCHING);
}

// Device work has retired, but its complete RGB result remains borrowed until
// the last surviving encoder releases it. Only one PNG/base64 buffer is live.
static void loom_serve_image_encode_next(loom_serve_image_service_t* service) {
  while (service->cohort.next < service->cohort.count &&
         !service->cohort.jobs[service->cohort.next]->connection) {
    loom_serve_image_job_finish(service,
                                service->cohort.jobs[service->cohort.next++],
                                LOOM_SERVE_TELEMETRY_COMPLETED);
  }
  if (service->cohort.next == service->cohort.count) {
    loom_serve_image_finish(service, LOOM_SERVE_TELEMETRY_COMPLETED);
  } else {
    const loom_serve_image_job_t* job =
        service->cohort.jobs[service->cohort.next];
    service->cpu.rgb = iree_make_const_byte_span(
        service->cohort.rgb.data + job->output_offset, job->output_length);
    loom_serve_image_schedule_cpu(service, IMAGE_PHASE_ENCODING);
  }
}

static void loom_serve_image_prepared(loom_serve_image_service_t* service,
                                      loom_serve_service_flags_t flags,
                                      bool* progress) {
  if (service->stopping || !loom_serve_image_has_peer(service)) {
    service->generator.discard_batch(service->generator.self);
    loom_serve_image_finish(service, service->telemetry.failure_code
                                         ? LOOM_SERVE_TELEMETRY_FAILED
                                         : LOOM_SERVE_TELEMETRY_CANCELLED);
    *progress = true;
    return;
  }
  loom_serve_residency_admission_t admission;
  service->status =
      service->generator.try_admit(service->generator.self, &admission);
  if (iree_status_is_ok(service->status) &&
      admission == LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED) {
    return;
  }
  if (iree_status_is_ok(service->status) &&
      admission == LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY) {
    if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY)) {
      return;
    }
    // No accepted producer can release capacity; waiting on queued peers
    // would deadlock mutually inadmissible requests.
    service->generator.discard_batch(service->generator.self);
    loom_serve_image_reject_cohort(
        service, "503 Service Unavailable",
        IREE_SV("model parameters exceed available capacity"));
    *progress = true;
    return;
  } else if (iree_status_is_ok(service->status)) {
    const iree_duration_t admission_ns = iree_time_now() - service->phase_start;
    for (iree_host_size_t i = 0; i < service->cohort.count; ++i) {
      service->cohort.jobs[i]->work.admission_ns = admission_ns;
    }
    service->telemetry.totals.work.admission_ns += admission_ns;
    loom_serve_execution_permit_request(service->generator.permit,
                                        LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
    service->phase = IMAGE_PHASE_READY;
  }
  *progress = true;
  if (!iree_status_is_ok(service->status)) {
    loom_serve_image_finish(service, LOOM_SERVE_TELEMETRY_FAILED);
  }
}

static void loom_serve_image_collect(loom_serve_image_service_t* service,
                                     bool* progress) {
  iree_slim_mutex_lock(&service->events.mutex);
  uint32_t ready = service->events.ready;
  service->events.ready = 0;
  service->status = iree_status_join(service->status, service->events.status);
  service->events.status = iree_ok_status();
  iree_status_t cpu_status = iree_ok_status();
  if (iree_any_bit_set(ready, IMAGE_EVENT_CPU)) {
    cpu_status = service->events.cpu_status;
    service->events.cpu_status = iree_ok_status();
  }
  iree_slim_mutex_unlock(&service->events.mutex);
  if (!ready) {
    return;
  }
  *progress = true;
  IREE_TRACE({
    if (iree_any_bit_set(ready, IMAGE_EVENT_DEVICE)) {
      IREE_TRACE_SECTION_LEAVE(service->device.section);
      service->device.section = 0;
    }
  });
  if (iree_any_bit_set(ready, IMAGE_EVENT_CPU)) {
    if (service->phase == IMAGE_PHASE_PREPARING ||
        service->phase == IMAGE_PHASE_BATCHING) {
      service->telemetry.totals.work.prepare_ns += service->cpu.elapsed_ns;
      if (service->phase == IMAGE_PHASE_BATCHING) {
        for (iree_host_size_t i = 0; i < service->cohort.count; ++i) {
          service->cohort.jobs[i]->work.prepare_ns += service->cpu.elapsed_ns;
        }
      }
    } else {
      const loom_serve_image_work_t* work =
          &service->cohort.jobs[service->cohort.next]->work;
      service->telemetry.totals.work.png_ns += work->png_ns;
      service->telemetry.totals.work.json_ns += work->json_ns;
      service->telemetry.totals.work.png_bytes += work->png_bytes;
    }
  }
  if (service->phase == IMAGE_PHASE_BATCHING &&
      iree_status_is_invalid_argument(cpu_status) &&
      iree_status_is_ok(service->status)) {
    iree_string_view_t message = iree_status_message(cpu_status);
    if (!message.size) {
      message = IREE_SV("source rejected image cohort");
    }
    loom_serve_image_diagnose(service, loom_serve_image_active(service),
                              cpu_status);
    loom_serve_image_reject_cohort(service, "400 Bad Request", message);
    iree_status_free(cpu_status);
    return;
  }
  service->status = iree_status_join(service->status, cpu_status);
  if (!iree_status_is_ok(service->status)) {
    loom_serve_image_finish(service, LOOM_SERVE_TELEMETRY_FAILED);
    return;
  }
  if (iree_any_bit_set(ready, IMAGE_EVENT_DEVICE)) {
    const iree_duration_t execution_ns = iree_time_now() - service->phase_start;
    for (iree_host_size_t i = 0; i < service->cohort.count; ++i) {
      service->cohort.jobs[i]->work.execution_ns += execution_ns;
      if (!service->device.cancelled) {
        ++service->cohort.jobs[i]->completed_parts;
      }
    }
    service->telemetry.totals.work.execution_ns += execution_ns;
    if (!service->device.cancelled) {
      loom_serve_execution_permit_complete(service->generator.permit);
      ++service->telemetry.counters.batches;
    }
    if (!service->device.complete) {
      loom_serve_execution_permit_request(service->generator.permit,
                                          LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
      service->phase = IMAGE_PHASE_READY;
      return;
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z_complete, "loom_serve_image_complete");
    IREE_TRACE_ZONE_APPEND_TEXT(z_complete, service->options.name.data,
                                service->options.name.size);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(
        z_complete,
        loom_serve_telemetry_stream_identity(service->telemetry.stream)
            .ordinal);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_complete,
                                     loom_serve_image_active(service));
    service->cohort.rgb = service->generator.complete(service->generator.self);
    IREE_TRACE_ZONE_END(z_complete);
    if (service->device.cancelled) {
      loom_serve_image_finish(service, LOOM_SERVE_TELEMETRY_CANCELLED);
      return;
    }
    service->telemetry.totals.generated_images += service->cohort.count;
    for (iree_host_size_t i = 0; i < service->cohort.count; ++i) {
      const loom_serve_image_job_t* job = service->cohort.jobs[i];
      service->telemetry.totals.generated_pixels +=
          (uint64_t)job->request.width * job->request.height;
      loom_serve_image_event(service, &LOOM_SERVE_IMAGE_GENERATED, job->id);
    }
    loom_serve_image_encode_next(service);
  } else if (service->phase == IMAGE_PHASE_PREPARING) {
    loom_serve_image_preparation_complete(service);
  } else if (service->phase == IMAGE_PHASE_BATCHING) {
    // The generator checked the total output extent before accepting capture.
    iree_host_size_t offset = 0;
    for (iree_host_size_t i = 0; i < service->cohort.count; ++i) {
      loom_serve_image_job_t* job = service->cohort.jobs[i];
      job->output_offset = offset;
      offset += job->output_length;
      // Capture owns the packed payload now; request-local source storage is
      // no longer borrowed by CPU or device work.
      iree_vm_buffer_release(job->prepared.bytes);
      job->prepared.bytes = NULL;
      loom_serve_image_record_prepared(service->telemetry.stream, job->id,
                                       &service->cohort.prepared);
    }
    service->phase = IMAGE_PHASE_SPECIALIZING;
    service->phase_start = iree_time_now();
  } else {
    loom_serve_image_job_finish(service,
                                service->cohort.jobs[service->cohort.next++],
                                LOOM_SERVE_TELEMETRY_COMPLETED);
    loom_serve_image_encode_next(service);
  }
}

static void loom_serve_image_prune(loom_serve_image_service_t* service,
                                   bool* progress) {
  for (iree_host_size_t i = service->cohort.next; i < service->cohort.count;
       ++i) {
    loom_serve_image_job_t* job = service->cohort.jobs[i];
    if (job->connection && loom_serve_http_connection_failed(job->connection)) {
      loom_serve_http_connection_abort(job->connection);
      job->connection = NULL;
      *progress = true;
    }
  }
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    loom_serve_image_job_t* job = service->pending[i];
    if (loom_serve_http_connection_failed(job->connection)) {
      loom_serve_image_job_finish(service, job, LOOM_SERVE_TELEMETRY_CANCELLED);
      *progress = true;
    } else {
      service->pending[retained++] = job;
    }
  }
  service->pending_count = retained;
}

static iree_status_t loom_serve_image_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_service_flags_t flags) {
  loom_serve_image_service_t* service = self;
  loom_serve_image_job_t receipt = {
      .id = ++service->telemetry.counters.requests,
      .connection = connection,
      .arrival_time = iree_time_now()};
  const loom_serve_telemetry_record_t begin = {
      .time_ns = receipt.arrival_time,
      .request = receipt.id,
      .event = &LOOM_SERVE_TELEMETRY_REQUEST_BEGIN};
  loom_serve_telemetry_stream_append(service->telemetry.stream, &begin);
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(request->target,
                              IREE_SV("/v1/images/generations"))) {
    return loom_serve_image_reject(service, connection, &receipt,
                                   "404 Not Found",
                                   IREE_SV("unknown image endpoint"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("content-type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return loom_serve_image_reject(
        service, connection, &receipt, "415 Unsupported Media Type",
        IREE_SV("Content-Type must be application/json"));
  }
  iree_status_t status = loom_serve_image_request_initialize(
      request->body, service->options.name, service->options.width,
      service->options.height, service->host_allocator, &receipt.request);
  if (!iree_status_is_ok(status)) {
    iree_string_view_t message = iree_status_message(status);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(status)));
    }
    iree_status_t response_status = loom_serve_image_reject(
        service, connection, &receipt,
        iree_status_is_resource_exhausted(status) ? "503 Service Unavailable"
                                                  : "400 Bad Request",
        message);
    loom_serve_image_diagnose(service, receipt.id, status);
    iree_status_free(status);
    return response_status;
  }
  if (service->count == service->capacity) {
    loom_serve_image_request_deinitialize(&receipt.request);
    return loom_serve_image_reject(service, connection, &receipt,
                                   "503 Service Unavailable",
                                   IREE_SV("image admission queue is full"));
  }
  loom_serve_image_job_t* job = NULL;
  status = iree_allocator_malloc(service->host_allocator, sizeof(*job),
                                 (void**)&job);
  if (iree_status_is_ok(status)) {
    *job = receipt;
    job->output_length =
        (iree_host_size_t)job->request.width * job->request.height * 12;
    service->pending[service->pending_count++] = job;
    ++service->count;
  } else {
    loom_serve_image_request_end(service, &receipt, LOOM_SERVE_TELEMETRY_FAILED,
                                 status);
    loom_serve_image_request_deinitialize(&receipt.request);
  }
  return status;
}

static loom_serve_service_snapshot_t loom_serve_image_snapshot(void* self) {
  const loom_serve_image_service_t* service = self;
  const iree_time_t now = iree_time_now();
  return (loom_serve_service_snapshot_t){
      .phase = image_phases[service->phase],
      .queued_requests = service->pending_count,
      .active_requests = service->cohort.count - service->cohort.next,
      .oldest_queued_age =
          service->pending_count ? now - service->pending[0]->arrival_time : 0,
      .oldest_active_age =
          service->cohort.count
              ? now - service->cohort.jobs[service->cohort.next]->arrival_time
              : 0};
}

static bool loom_serve_image_request_at(
    void* self, iree_host_size_t index,
    loom_serve_http_request_snapshot_t* out_snapshot) {
  const loom_serve_image_service_t* service = self;
  const loom_serve_image_job_t* job = NULL;
  const char* phase = "queued";
  if (index < service->pending_count) {
    job = service->pending[index];
  } else if (index - service->pending_count <
             service->cohort.count - service->cohort.next) {
    job = service->cohort
              .jobs[service->cohort.next + index - service->pending_count];
    phase = image_phases[service->phase];
  } else {
    return false;
  }
  *out_snapshot = (loom_serve_http_request_snapshot_t){
      .id = job->id,
      .phase = phase,
      .arrival_time = job->arrival_time,
      .counters = {.requests = 1, .batches = job->completed_parts}};
  return true;
}

// Heartbeat work is a scheduler turn, woken by an asynchronous timer. These
// phases describe ownership, not individual completed device dispatches.
static void loom_serve_image_heartbeat(loom_serve_image_service_t* service) {
  const loom_serve_service_snapshot_t state =
      loom_serve_image_snapshot(service);
  loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .request = loom_serve_image_active(service),
      .event = &LOOM_SERVE_TELEMETRY_SNAPSHOT,
      .data.snapshot = {.counters = service->telemetry.counters,
                        .queued = state.queued_requests,
                        .active = state.active_requests}};
  memcpy(record.data.snapshot.phase, state.phase, strlen(state.phase));
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
  loom_serve_image_record_totals(service->telemetry.stream,
                                 &service->telemetry.totals);
  loom_serve_image_record_weights(service, record.time_ns);
}

static void loom_serve_image_stop(loom_serve_image_service_t* service) {
  service->stopping = true;
  if (service->telemetry.stream) {
    loom_serve_image_event(service, &LOOM_SERVE_IMAGE_SHUTDOWN,
                           loom_serve_image_active(service));
  }
  const loom_serve_telemetry_outcome_t outcome =
      service->telemetry.failure_code ? LOOM_SERVE_TELEMETRY_FAILED
                                      : LOOM_SERVE_TELEMETRY_CANCELLED;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    loom_serve_image_job_finish(service, service->pending[i], outcome);
  }
  service->pending_count = 0;
  for (iree_host_size_t i = service->cohort.next; i < service->cohort.count;
       ++i) {
    loom_serve_image_job_t* job = service->cohort.jobs[i];
    if (job->connection) {
      loom_serve_http_connection_abort(job->connection);
      job->connection = NULL;
    }
  }
}

static void loom_serve_image_set_observer(void* self,
                                          loom_serve_wake_callback_t observer) {
  loom_serve_image_service_t* service = self;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.observer = observer;
  iree_slim_mutex_unlock(&service->events.mutex);
  loom_serve_timer_set_observer(&service->timer, observer);
  loom_serve_execution_reconfiguration_set_observer(service->reconfiguration,
                                                    observer);
  service->generator.set_observer(service->generator.self, observer);
}

static bool loom_serve_image_is_active(void* self) {
  const loom_serve_image_service_t* service = self;
  return service->phase == IMAGE_PHASE_PREPARING ||
         service->phase == IMAGE_PHASE_BATCHING ||
         service->phase == IMAGE_PHASE_SPECIALIZING ||
         service->phase == IMAGE_PHASE_READY ||
         service->phase == IMAGE_PHASE_GENERATING ||
         service->phase == IMAGE_PHASE_ENCODING;
}

static bool loom_serve_image_is_pending(void* self) {
  loom_serve_image_service_t* service = self;
  return loom_serve_image_is_active(service) ||
         (service->generator.is_pending &&
          service->generator.is_pending(service->generator.self)) ||
         loom_serve_execution_reconfiguration_pending(
             service->reconfiguration) ||
         loom_serve_timer_is_pending(&service->timer);
}

static iree_status_t loom_serve_image_advance(void* self,
                                              loom_serve_service_flags_t flags,
                                              bool* out_progress) {
  loom_serve_image_service_t* service = self;
  *out_progress = false;
  if (!service->stopping &&
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    loom_serve_image_stop(service);
    *out_progress = true;
  }
  loom_serve_image_prune(service, out_progress);
  loom_serve_image_collect(service, out_progress);
  if (service->generator.query_status) {
    service->status = iree_status_join(
        service->status,
        service->generator.query_status(service->generator.self));
  }
  *out_progress |= loom_serve_execution_reconfiguration_advance(
      service->reconfiguration,
      iree_status_is_ok(service->status) &&
          (service->phase == IMAGE_PHASE_IDLE ||
           service->phase == IMAGE_PHASE_WAITING),
      service->stopping || !iree_status_is_ok(service->status));
  if (!service->stopping && !iree_status_is_ok(service->status)) {
    service->telemetry.failure_code = iree_status_code(service->status);
    loom_serve_image_stop(service);
    *out_progress = true;
  }
  if (service->phase == IMAGE_PHASE_SPECIALIZING) {
    const bool cancel =
        service->stopping || !loom_serve_image_has_peer(service);
    bool complete = false;
    iree_status_t status = service->generator.advance_batch(
        service->generator.self, cancel, &complete);
    if (!iree_status_is_ok(status)) {
      service->generator.discard_batch(service->generator.self);
      loom_serve_image_diagnose(service, loom_serve_image_active(service),
                                status);
      loom_serve_image_reject_cohort(service,
                                     iree_status_is_resource_exhausted(status)
                                         ? "503 Service Unavailable"
                                         : "422 Unprocessable Entity",
                                     iree_status_message(status));
      iree_status_free(status);
      *out_progress = true;
    } else if (complete) {
      if (cancel) {
        service->generator.discard_batch(service->generator.self);
        loom_serve_image_finish(service, LOOM_SERVE_TELEMETRY_CANCELLED);
      } else {
        service->phase = IMAGE_PHASE_WAITING;
        service->phase_start = iree_time_now();
      }
      *out_progress = true;
    }
  }
  if (service->phase == IMAGE_PHASE_WAITING) {
    loom_serve_image_prepared(service, flags, out_progress);
  }
  if (service->phase == IMAGE_PHASE_READY &&
      (service->stopping || !loom_serve_image_has_peer(service) ||
       loom_serve_execution_permit_try_acquire(service->generator.permit))) {
    service->status = loom_serve_image_service_submit(service);
    *out_progress = true;
    if (!iree_status_is_ok(service->status)) {
      loom_serve_image_finish(service, LOOM_SERVE_TELEMETRY_FAILED);
    }
  }
  if (!service->stopping && iree_status_is_ok(service->status) &&
      service->phase == IMAGE_PHASE_IDLE && service->pending_count) {
    if (service->pending[0]->prepared.bytes) {
      loom_serve_image_select(service);
    } else {
      memcpy(service->cohort.jobs, service->pending,
             service->pending_count * sizeof(*service->cohort.jobs));
      service->cohort.count = service->pending_count;
      service->pending_count = 0;
      loom_serve_image_schedule_cpu(service, IMAGE_PHASE_PREPARING);
    }
    *out_progress = true;
  }
  bool timer_progress = false;
  bool tick = false;
  service->status = iree_status_join(
      service->status,
      loom_serve_timer_advance(&service->timer,
                               service->stopping && !service->cohort.count,
                               true, &timer_progress, &tick));
  *out_progress |= timer_progress;
  if (tick) {
    loom_serve_image_heartbeat(service);
  }
  if (!service->stopping && !iree_status_is_ok(service->status)) {
    service->telemetry.failure_code = iree_status_code(service->status);
    loom_serve_image_stop(service);
    *out_progress = true;
  }
  iree_status_t status = service->status;
  service->status = iree_ok_status();
  if (!iree_status_is_ok(status)) {
    service->telemetry.failure_code = iree_status_code(status);
    loom_serve_image_diagnose(service, loom_serve_image_active(service),
                              status);
  }
  return status;
}

loom_serve_service_t loom_serve_image_service_interface(
    loom_serve_image_service_t* service) {
  static const iree_string_view_t input_modalities[] = {IREE_SVL("text")};
  static const iree_string_view_t output_modalities[] = {IREE_SVL("image")};
  return (loom_serve_service_t){
      .name = service->options.name,
      .self = service,
      .set_observer = loom_serve_image_set_observer,
      .is_pending = loom_serve_image_is_pending,
      .is_active = loom_serve_image_is_active,
      .accept = loom_serve_image_accept,
      .advance = loom_serve_image_advance,
      .tokenizer = service->generator.tokenizer,
      .kind = LOOM_SERVE_MODEL_KIND_IMAGE,
      .endpoint = loom_serve_model_kind_endpoint(LOOM_SERVE_MODEL_KIND_IMAGE),
      .capabilities = {.input_modalities = {IREE_ARRAYSIZE(input_modalities),
                                            input_modalities},
                       .output_modalities = {IREE_ARRAYSIZE(output_modalities),
                                             output_modalities}},
      .snapshot = loom_serve_image_snapshot,
      .inspection = {.request_at = loom_serve_image_request_at},
      .reconfiguration = loom_serve_execution_reconfiguration_interface(
          service->reconfiguration),
      .telemetry = {.stream = service->telemetry.stream,
                    .counters = &service->telemetry.counters,
                    .latency = &service->telemetry.latency},
      .capacity_domain = service->generator.capacity_domain};
}

void loom_serve_image_service_free(loom_serve_image_service_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!loom_serve_image_is_pending(service));
  const iree_allocator_t host_allocator = service->host_allocator;
  loom_serve_execution_permit_cancel(service->generator.permit);
  if (!service->stopping) {
    loom_serve_image_stop(service);
  }
  if (service->phase == IMAGE_PHASE_WAITING) {
    service->generator.discard_batch(service->generator.self);
    loom_serve_image_finish(service, service->telemetry.failure_code
                                         ? LOOM_SERVE_TELEMETRY_FAILED
                                         : LOOM_SERVE_TELEMETRY_CANCELLED);
  }
  loom_serve_execution_reconfiguration_free(service->reconfiguration);
  service->generator.set_observer(service->generator.self,
                                  (loom_serve_wake_callback_t){0});
  loom_serve_timer_deinitialize(&service->timer);
  iree_string_builder_deinitialize(&service->cpu.response);
  iree_string_builder_deinitialize(&service->response);
  iree_allocator_free(host_allocator, service->pending);
  iree_allocator_free(host_allocator, service->cohort.jobs);
  iree_allocator_free(host_allocator, service->cohort.requests);
  iree_allocator_free(host_allocator, service->cohort.output_lengths);
  iree_slim_mutex_deinitialize(&service->events.mutex);
  if (service->telemetry.stream) {
    loom_serve_image_heartbeat(service);
  }
  if (service->telemetry.owns_stream) {
    loom_serve_telemetry_stream_free(service->telemetry.stream);
  }
  iree_allocator_free_aligned(host_allocator, service);
}

iree_status_t loom_serve_image_service_allocate(
    loom_serve_host_t* host, loom_serve_image_generator_t generator,
    const loom_serve_image_service_options_t* options,
    loom_serve_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_image_service_t** out_service) {
  *out_service = NULL;
  if (!generator.try_admit || !generator.batch_limit || !generator.prepare ||
      !generator.prepare_batch || !generator.set_observer ||
      !generator.advance_batch || !generator.discard_batch ||
      !generator.submit || !generator.complete || !options->width ||
      !options->height || !options->name.size || !options->pending_capacity ||
      options->heartbeat_interval < 0 ||
      options->pending_capacity >=
          IREE_HOST_SIZE_MAX / sizeof(loom_serve_prepared_request_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image service requires a generator, geometry, "
                            "name and queue capacity");
  }
  loom_serve_image_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*service),
      iree_alignof(loom_serve_image_service_t), 0, (void**)&service));
  service->host_allocator = host_allocator;
  service->generator = generator;
  service->options = *options;
  service->capacity = options->pending_capacity + 1;
  service->telemetry.stream = stream;
  service->executor = loom_serve_host_executor(host);
  iree_slim_mutex_initialize(&service->events.mutex);
  loom_serve_timer_initialize(&service->timer);
  iree_string_builder_initialize(host_allocator, &service->response);
  iree_string_builder_initialize(host_allocator, &service->cpu.response);
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, service->capacity, sizeof(*service->pending),
      (void**)&service->pending);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, service->capacity,
                                         sizeof(*service->cohort.jobs),
                                         (void**)&service->cohort.jobs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, service->capacity,
                                         sizeof(*service->cohort.requests),
                                         (void**)&service->cohort.requests);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, service->capacity,
                                    sizeof(*service->cohort.output_lengths),
                                    (void**)&service->cohort.output_lengths);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_reconfiguration_allocate(
        generator.execution_update, options->variant_capacity, host_allocator,
        &service->reconfiguration);
  }
  if (iree_status_is_ok(status) && !stream) {
    status = loom_serve_telemetry_stream_allocate(128, host_allocator,
                                                  &service->telemetry.stream);
    service->telemetry.owns_stream = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    if (stream) {
      loom_serve_image_heartbeat(service);
    } else {
      loom_serve_image_event(service, &LOOM_SERVE_TELEMETRY_STREAM_BEGIN, 0);
    }
    status = iree_async_proactor_pool_get(loom_serve_host_proactor_pool(host),
                                          0, &service->proactor);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_timer_configure(&service->timer, service->proactor,
                               options->heartbeat_interval);
    *out_service = service;
  } else {
    loom_serve_image_service_free(service);
  }
  return status;
}
