// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/service.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "experimental/loom_serve/http/timer.h"
#include "experimental/loom_serve/image/output.h"
#include "iree/async/operations/semaphore.h"
#include "iree/base/internal/base64.h"
#include "iree/base/threading/mutex.h"
#include "loom/util/json.h"

typedef struct image_job_t {
  // Monotonic process-local request identity; zero denotes no active job.
  uint64_t id;
  // Owned decoded input, stable until the CPU preparation task retires.
  loom_serve_image_request_t request;
} image_job_t;

typedef struct image_pending_t {
  // Borrowed claimed connection, released by response or cancellation.
  loom_serve_http_connection_t* connection;
  // Owned pending request, moved to the invocation on admission.
  image_job_t job;
} image_pending_t;

typedef enum image_phase_e {
  IMAGE_PHASE_IDLE = 0,
  IMAGE_PHASE_PREPARING,
  IMAGE_PHASE_WAITING,
  IMAGE_PHASE_GENERATING,
  IMAGE_PHASE_ENCODING,
} image_phase_t;

enum { IMAGE_EVENT_CPU = 1u << 0, IMAGE_EVENT_DEVICE = 1u << 1 };

struct loom_serve_image_service_t {
  // Allocator for queued inputs and encoded response storage.
  iree_allocator_t allocator;
  // Borrowed, serialized native model invocation.
  loom_serve_image_generator_t generator;
  // Copied model geometry and policy; the route name is borrowed.
  loom_serve_image_service_options_t options;
  // Borrowed application executor, shared with JIT and all CPU/model work.
  iree_task_executor_t* executor;
  // Borrowed host proactor for model multi-waits and heartbeat timers.
  iree_async_proactor_t* proactor;
  // Current invocation phase, mutated only by scheduler turns.
  image_phase_t phase;
  // Stops admission while accepted tasks and model work finish naturally.
  bool stopping;
  // Owned failure accumulated on a turn, transferred to the router on return.
  iree_status_t status;
  // Input retained while preparation, device work or encoding is outstanding.
  image_job_t job;
  // Completion publication shared by CPU workers, proactor and scheduler.
  struct {
    // Joins publication callbacks' final access before owner destruction.
    iree_slim_mutex_t mutex;
    // Borrowed router wake, joined by mutex when detached after retirement.
    loom_serve_http_observer_t observer;
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
    image_phase_t phase;
    // Borrowed successful model output, retained through encoding.
    iree_const_byte_span_t rgb;
    // Owned complete HTTP result, independent of health/metadata responses.
    iree_string_builder_t response;
    // Encoded PNG size for metrics, excluding JSON/base64 overhead.
    iree_host_size_t png_bytes;
  } cpu;
  // Intrusive proactor wait storage, never reused before its final callback.
  struct {
    // Waits for both model work and feedback without occupying a task worker.
    iree_async_semaphore_wait_operation_t operation;
    // Borrowed HAL handles and values from the accepted model invocation.
    loom_serve_execution_frontier_t frontier;
    // Toll-free async semaphore handles used by the public proactor API.
    iree_async_semaphore_t* semaphores[2];
  } device;
  // Periodic readiness and owned cancellation, independent of device waits.
  loom_serve_http_timer_t timer;
  // Bounded FIFO of owned host requests, excluding the active image.
  image_pending_t* pending;
  // Number of occupied FIFO entries.
  iree_host_size_t pending_count;
  // Active request's peer; NULL after a reset while the image still runs.
  loom_serve_http_connection_t* connection;
  // Active identity remains nonzero until completion, including after reset.
  uint64_t active;
  // Admission time of the active image, including generation and encoding.
  iree_time_t active_start;
  // Last assigned request identity.
  uint64_t sequence;
  // Successfully generated images, including results discarded after reset.
  uint64_t completed;
  // Reusable application-side HTTP/JSON response storage.
  iree_string_builder_t response;
};

static void image_diagnose(const char* operation, iree_status_t failure) {
  fprintf(stderr, "%s: ", operation);
  iree_status_fprint(stderr, failure);
  iree_status_free(failure);
}

static iree_status_t image_quote(iree_string_builder_t* output,
                                 iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static iree_status_t image_response_begin(iree_string_builder_t* output,
                                          const char* status) {
  iree_string_builder_reset(output);
  return iree_string_builder_append_format(
      output,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      status);
}

// Each connection sends exactly one complete response. The transport copies
// admitted bytes, so a slow or reset peer never holds a generator buffer.
static void image_response_send(loom_serve_http_connection_t* connection,
                                iree_string_view_t response) {
  // A fresh response has no occupied send slots. Zero budget therefore means
  // the carrier is closing; that can precede its failure callback publication.
  if (loom_serve_http_connection_failed(connection) ||
      !loom_serve_http_connection_can_send(connection)) {
    fprintf(stderr, "{\"event\":\"image_response_abandoned\"}\n");
    loom_serve_http_connection_abort(connection);
    return;
  }
  iree_status_t status = loom_serve_http_connection_send(connection, response);
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
  } else {
    image_diagnose("Image response peer failed", status);
    loom_serve_http_connection_abort(connection);
  }
}

static iree_status_t image_reject(loom_serve_image_service_t* service,
                                  loom_serve_http_connection_t* connection,
                                  const char* code,
                                  iree_string_view_t message) {
  iree_status_t status = image_response_begin(&service->response, code);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        &service->response,
        "{\"error\":{\"type\":\"image_request_error\",\"message\":");
  }
  if (iree_status_is_ok(status)) {
    status = image_quote(&service->response, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&service->response, "}}");
  }
  if (iree_status_is_ok(status)) {
    image_response_send(connection,
                        iree_string_builder_view(&service->response));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t image_result_json(iree_string_builder_t* output,
                                       iree_const_byte_span_t png) {
  IREE_RETURN_IF_ERROR(image_response_begin(output, "200 OK"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"created\":%" PRId64 ",\"data\":[{\"b64_json\":\"",
      iree_time_now() / INT64_C(1000000000)));
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

static iree_status_t image_cpu_run(iree_task_process_t* process,
                                   const iree_task_worker_context_t* context,
                                   iree_task_process_drain_result_t* result) {
  (void)context;
  loom_serve_image_service_t* service = process->user_data;
  result->completed = true;
  result->did_work = true;
  if (service->cpu.phase == IMAGE_PHASE_PREPARING) {
    return service->generator.prepare(service->generator.self,
                                      &service->job.request);
  }
  const iree_time_t begin = iree_time_now();
  iree_byte_span_t png = iree_byte_span_empty();
  iree_status_t status = loom_serve_image_encode_rgb_f32_png(
      service->options.width, service->options.height, service->cpu.rgb, &png,
      service->allocator);
  const iree_time_t encoded = iree_time_now();
  if (iree_status_is_ok(status)) {
    service->cpu.png_bytes = png.data_length;
    status =
        image_result_json(&service->cpu.response,
                          iree_make_const_byte_span(png.data, png.data_length));
  }
  iree_allocator_free(service->allocator, png.data);
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "{\"event\":\"image_encoding\",\"request\":%" PRIu64
            ",\"png_ns\":%" PRId64 ",\"json_ns\":%" PRId64 "}\n",
            service->job.id, encoded - begin, iree_time_now() - encoded);
  }
  return status;
}

static void image_cpu_completed(iree_task_process_t* process,
                                iree_status_t status) {
  loom_serve_image_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.cpu_status = status;
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void image_cpu_released(iree_task_process_t* process) {
  loom_serve_image_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.ready |= IMAGE_EVENT_CPU;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void image_schedule_cpu(loom_serve_image_service_t* service,
                               image_phase_t phase) {
  service->phase = phase;
  service->cpu.phase = phase;
  iree_task_process_initialize(image_cpu_run, 0, 1, &service->cpu.process);
  service->cpu.process.user_data = service;
  service->cpu.process.completion_fn = image_cpu_completed;
  service->cpu.process.release_fn = image_cpu_released;
  iree_task_executor_schedule_process(service->executor, &service->cpu.process);
}

static void image_start(loom_serve_image_service_t* service,
                        image_pending_t pending) {
  service->active = pending.job.id;
  service->active_start = iree_time_now();
  service->connection = pending.connection;
  service->job = pending.job;
  fprintf(stderr, "{\"event\":\"image_started\",\"request\":%" PRIu64 "}\n",
          service->active);
  image_schedule_cpu(service, IMAGE_PHASE_PREPARING);
}

static void image_device_completed(void* user_data,
                                   iree_async_operation_t* operation,
                                   iree_status_t status,
                                   iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_image_service_t* service = user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.status = iree_status_join(service->events.status, status);
  service->events.ready |= IMAGE_EVENT_DEVICE;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static iree_status_t image_submit(loom_serve_image_service_t* service) {
  IREE_RETURN_IF_ERROR(service->generator.submit(service->generator.self,
                                                 &service->device.frontier));
  iree_async_semaphore_wait_operation_t* wait = &service->device.operation;
  memset(wait, 0, sizeof(*wait));
  iree_async_operation_initialize(
      &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, image_device_completed, service);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    service->device.semaphores[i] =
        (iree_async_semaphore_t*)service->device.frontier.semaphores[i];
  }
  wait->semaphores = service->device.semaphores;
  wait->values = service->device.frontier.values;
  wait->count = 2;
  wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_submit_one(service->proactor, &wait->base));
  service->phase = IMAGE_PHASE_GENERATING;
  return iree_ok_status();
}

static void image_finish(loom_serve_image_service_t* service) {
  if (service->connection) {
    if (iree_status_is_ok(service->status)) {
      const iree_time_t begin = iree_time_now();
      image_response_send(service->connection,
                          iree_string_builder_view(&service->cpu.response));
      fprintf(stderr,
              "{\"event\":\"image_response_transport\",\"request\":%" PRIu64
              ",\"send_call_ns\":%" PRId64 "}\n",
              service->active, iree_time_now() - begin);
    } else {
      loom_serve_http_connection_abort(service->connection);
    }
    service->connection = NULL;
  }
  loom_serve_image_request_deinitialize(&service->job.request);
  memset(&service->job, 0, sizeof(service->job));
  service->active = 0;
  service->phase = IMAGE_PHASE_IDLE;
}

static void image_prepared(loom_serve_image_service_t* service,
                           loom_serve_http_service_flags_t flags,
                           bool* progress) {
  if (service->stopping || !service->connection) {
    service->generator.discard_prepared_request(service->generator.self);
    image_finish(service);
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
    if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY)) {
      return;
    }
    // No accepted producer can release capacity; waiting on queued peers
    // would deadlock mutually inadmissible requests.
    service->generator.discard_prepared_request(service->generator.self);
    service->status =
        image_reject(service, service->connection, "503 Service Unavailable",
                     IREE_SV("model parameters exceed available capacity"));
    service->connection = NULL;
    image_finish(service);
    *progress = true;
    return;
  } else if (iree_status_is_ok(service->status)) {
    service->status = image_submit(service);
  }
  *progress = true;
  if (!iree_status_is_ok(service->status)) {
    image_finish(service);
  }
}

static void image_collect(loom_serve_image_service_t* service, bool* progress) {
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
  if (service->phase == IMAGE_PHASE_PREPARING &&
      iree_status_is_invalid_argument(cpu_status) &&
      iree_status_is_ok(service->status)) {
    if (service->connection) {
      iree_string_view_t message = iree_status_message(cpu_status);
      if (!message.size) {
        message = IREE_SV("source rejected image input");
      }
      service->status = image_reject(service, service->connection,
                                     "400 Bad Request", message);
      service->connection = NULL;
    }
    image_diagnose("Image source input rejected", cpu_status);
    image_finish(service);
    return;
  }
  service->status = iree_status_join(service->status, cpu_status);
  if (!iree_status_is_ok(service->status)) {
    image_finish(service);
    return;
  }
  if (iree_any_bit_set(ready, IMAGE_EVENT_DEVICE)) {
    service->cpu.rgb = service->generator.complete(service->generator.self);
    image_schedule_cpu(service, IMAGE_PHASE_ENCODING);
  } else if (service->phase == IMAGE_PHASE_PREPARING) {
    service->phase = IMAGE_PHASE_WAITING;
  } else {
    ++service->completed;
    fprintf(stderr,
            "{\"event\":\"image_generated\",\"request\":%" PRIu64
            ",\"png_bytes\":%zu,\"completed\":%" PRIu64 "}\n",
            service->active, service->cpu.png_bytes, service->completed);
    image_finish(service);
  }
}

static void image_cancel(loom_serve_http_connection_t* connection, uint64_t id,
                         const char* phase) {
  fprintf(stderr,
          "{\"event\":\"image_cancelled\",\"request\":%" PRIu64
          ",\"phase\":\"%s\"}\n",
          id, phase);
  loom_serve_http_connection_abort(connection);
}

static void image_prune(loom_serve_image_service_t* service, bool* progress) {
  if (service->connection &&
      loom_serve_http_connection_failed(service->connection)) {
    image_cancel(service->connection, service->active, "active");
    service->connection = NULL;
    *progress = true;
  }
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    image_pending_t pending = service->pending[i];
    if (loom_serve_http_connection_failed(pending.connection)) {
      image_cancel(pending.connection, pending.job.id, "queued");
      loom_serve_image_request_deinitialize(&pending.job.request);
      *progress = true;
    } else {
      service->pending[retained++] = pending;
    }
  }
  service->pending_count = retained;
}

static iree_status_t image_accept(void* self,
                                  loom_serve_http_connection_t* connection,
                                  const loom_serve_http_request_t* request,
                                  loom_serve_http_service_flags_t flags) {
  (void)flags;
  loom_serve_image_service_t* service = self;
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(request->target,
                              IREE_SV("/v1/images/generations"))) {
    return image_reject(service, connection, "404 Not Found",
                        IREE_SV("unknown image endpoint"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("content-type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return image_reject(service, connection, "415 Unsupported Media Type",
                        IREE_SV("Content-Type must be application/json"));
  }
  image_pending_t pending = {.connection = connection};
  iree_status_t status = loom_serve_image_request_initialize(
      request->body, service->options.name, service->options.width,
      service->options.height, &pending.job.request, service->allocator);
  if (!iree_status_is_ok(status)) {
    iree_string_view_t message = iree_status_message(status);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(status)));
    }
    iree_status_t response_status = image_reject(
        service, connection,
        iree_status_is_resource_exhausted(status) ? "503 Service Unavailable"
                                                  : "400 Bad Request",
        message);
    image_diagnose("Image request rejected", status);
    return response_status;
  }
  if (!service->options.adapter_enabled &&
      pending.job.request.strength != 1.0f) {
    loom_serve_image_request_deinitialize(&pending.job.request);
    return image_reject(service, connection, "400 Bad Request",
                        IREE_SV("strength requires a loaded adapter"));
  }
  if (service->active &&
      service->pending_count == service->options.pending_capacity) {
    loom_serve_image_request_deinitialize(&pending.job.request);
    return image_reject(service, connection, "503 Service Unavailable",
                        IREE_SV("image admission queue is full"));
  }
  pending.job.id = ++service->sequence;
  fprintf(stderr,
          "{\"event\":\"image_accepted\",\"request\":%" PRIu64
          ",\"seed\":%" PRIu64 "}\n",
          pending.job.id, pending.job.request.seed);
  if (!service->active) {
    image_start(service, pending);
  } else {
    service->pending[service->pending_count++] = pending;
  }
  return iree_ok_status();
}

// Heartbeat work is a scheduler turn, woken by an asynchronous timer. These
// phases describe ownership, not individual completed device dispatches.
static void image_heartbeat(loom_serve_image_service_t* service) {
  const iree_time_t now = iree_time_now();
  static const char* const phases[] = {"idle", "preparing", "waiting",
                                       "generating", "encoding"};
  fprintf(stderr,
          "{\"event\":\"image_heartbeat\",\"time_ns\":%" PRId64
          ",\"active_request\":%" PRIu64
          ",\"active_ms\":%.3f,"
          "\"phase\":\"%s\",\"queued\":%zu,\"completed\":%" PRIu64
          ",\"stopping\":%s}\n",
          now, service->active,
          service->active ? (now - service->active_start) / 1e6 : 0.0,
          phases[service->phase], service->pending_count, service->completed,
          service->stopping ? "true" : "false");
}

static iree_status_t image_ready(loom_serve_image_service_t* service,
                                 loom_serve_http_server_t* server) {
  char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_address_format(loom_serve_http_server_address(server),
                                sizeof(storage), storage, &address));
  fprintf(stderr,
          "{\"event\":\"image_ready\",\"address\":\"%.*s\",\"width\":%u,"
          "\"height\":%u,\"pending_capacity\":%zu}\n",
          (int)address.size, address.data, service->options.width,
          service->options.height, service->options.pending_capacity);
  return iree_ok_status();
}

static void image_stop(loom_serve_image_service_t* service) {
  service->stopping = true;
  fprintf(stderr, "{\"event\":\"image_shutdown\",\"active\":%" PRIu64 "}\n",
          service->active);
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    image_pending_t* pending = &service->pending[i];
    image_cancel(pending->connection, pending->job.id, "shutdown");
    loom_serve_image_request_deinitialize(&pending->job.request);
  }
  service->pending_count = 0;
  if (service->connection) {
    image_cancel(service->connection, service->active, "shutdown");
    service->connection = NULL;
  }
}

static void image_set_observer(void* self,
                               loom_serve_http_observer_t observer) {
  loom_serve_image_service_t* service = self;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.observer = observer;
  iree_slim_mutex_unlock(&service->events.mutex);
  loom_serve_http_timer_set_observer(&service->timer, observer);
}

static bool image_is_active(void* self) {
  const loom_serve_image_service_t* service = self;
  return service->phase == IMAGE_PHASE_PREPARING ||
         service->phase == IMAGE_PHASE_GENERATING ||
         service->phase == IMAGE_PHASE_ENCODING;
}

static bool image_is_pending(void* self) {
  loom_serve_image_service_t* service = self;
  return image_is_active(service) ||
         loom_serve_http_timer_is_pending(&service->timer);
}

static iree_status_t image_advance(void* self,
                                   loom_serve_http_service_flags_t flags,
                                   bool* out_progress) {
  loom_serve_image_service_t* service = self;
  *out_progress = false;
  if (!service->stopping &&
      iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
    image_stop(service);
    *out_progress = true;
  }
  image_prune(service, out_progress);
  image_collect(service, out_progress);
  if (!service->stopping && !iree_status_is_ok(service->status)) {
    image_stop(service);
    *out_progress = true;
  }
  if (service->phase == IMAGE_PHASE_WAITING) {
    image_prepared(service, flags, out_progress);
  }
  if (!service->stopping && iree_status_is_ok(service->status) &&
      !service->active && service->pending_count) {
    const image_pending_t pending = service->pending[0];
    --service->pending_count;
    memmove(service->pending, service->pending + 1,
            service->pending_count * sizeof(*service->pending));
    image_start(service, pending);
    *out_progress = true;
  }
  bool timer_progress = false;
  bool tick = false;
  service->status = iree_status_join(
      service->status,
      loom_serve_http_timer_advance(&service->timer,
                                    service->stopping && !service->active,
                                    &timer_progress, &tick));
  *out_progress |= timer_progress;
  if (tick) {
    image_heartbeat(service);
  }
  if (!service->stopping && !iree_status_is_ok(service->status)) {
    image_stop(service);
    *out_progress = true;
  }
  iree_status_t status = service->status;
  service->status = iree_ok_status();
  return status;
}

loom_serve_http_service_t loom_serve_image_service_interface(
    loom_serve_image_service_t* service) {
  return (loom_serve_http_service_t){.name = service->options.name,
                                     .self = service,
                                     .set_observer = image_set_observer,
                                     .is_pending = image_is_pending,
                                     .is_active = image_is_active,
                                     .accept = image_accept,
                                     .advance = image_advance,
                                     .tokenizer = service->generator.tokenizer};
}

void loom_serve_image_service_destroy(loom_serve_image_service_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!image_is_pending(service));
  const iree_allocator_t allocator = service->allocator;
  if (!service->stopping) {
    image_stop(service);
  }
  if (service->phase == IMAGE_PHASE_WAITING) {
    service->generator.discard_prepared_request(service->generator.self);
    image_finish(service);
  }
  loom_serve_http_timer_deinitialize(&service->timer);
  iree_string_builder_deinitialize(&service->cpu.response);
  iree_string_builder_deinitialize(&service->response);
  iree_allocator_free(allocator, service->pending);
  iree_slim_mutex_deinitialize(&service->events.mutex);
  fprintf(stderr, "{\"event\":\"image_stopped\",\"completed\":%" PRIu64 "}\n",
          service->completed);
  iree_allocator_free(allocator, service);
}

iree_status_t loom_serve_image_service_create(
    loom_serve_host_t* host, loom_serve_image_generator_t generator,
    const loom_serve_image_service_options_t* options,
    loom_serve_image_service_t** out_service, iree_allocator_t host_allocator) {
  *out_service = NULL;
  if (!generator.try_admit || !generator.prepare ||
      !generator.discard_prepared_request || !generator.submit ||
      !generator.complete || !options->width || !options->height ||
      !options->name.size || !options->pending_capacity ||
      options->heartbeat_interval < 0 ||
      options->pending_capacity >
          IREE_HOST_SIZE_MAX / sizeof(image_pending_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image service requires a generator, geometry, "
                            "name and queue capacity");
  }
  loom_serve_image_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*service),
                                             (void**)&service));
  service->allocator = host_allocator;
  service->generator = generator;
  service->options = *options;
  service->executor = loom_serve_host_executor(host);
  iree_slim_mutex_initialize(&service->events.mutex);
  loom_serve_http_timer_initialize(&service->timer);
  iree_string_builder_initialize(host_allocator, &service->response);
  iree_string_builder_initialize(host_allocator, &service->cpu.response);
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, options->pending_capacity, sizeof(*service->pending),
      (void**)&service->pending);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_get(loom_serve_host_proactor_pool(host),
                                          0, &service->proactor);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_timer_configure(&service->timer, service->proactor,
                                    options->heartbeat_interval);
    *out_service = service;
  } else {
    loom_serve_image_service_destroy(service);
  }
  return status;
}

iree_status_t loom_serve_image_service_run(
    loom_serve_host_t* host, loom_serve_image_generator_t generator,
    loom_serve_http_server_t* server,
    const loom_serve_image_service_options_t* options,
    iree_allocator_t host_allocator) {
  loom_serve_image_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_image_service_create(
      host, generator, options, &service, host_allocator));
  iree_status_t status = image_ready(service, server);
  if (iree_status_is_ok(status)) {
    const loom_serve_http_service_t interface =
        loom_serve_image_service_interface(service);
    status = loom_serve_http_router_run(
        loom_serve_host_executor(host), server, 1, &interface,
        options->pending_capacity, host_allocator);
  }
  loom_serve_image_service_destroy(service);
  return status;
}
