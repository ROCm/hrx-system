// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image_service.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "experimental/loom_serve/image_output.h"
#include "iree/base/internal/base64.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "loom/util/json.h"

typedef struct image_job_t {
  // Monotonic process-local request identity; zero denotes no active job.
  uint64_t id;
  // Owned decoded input, stable until the worker retires complete image work.
  loom_serve_image_request_t request;
} image_job_t;

typedef struct image_pending_t {
  // Borrowed claimed connection, released by response or cancellation.
  loom_serve_http_connection_t* connection;
  // Owned pending request, moved to the worker on admission.
  image_job_t job;
} image_pending_t;

typedef enum image_worker_state_e {
  IMAGE_WORKER_IDLE = 0,
  IMAGE_WORKER_READY,
  IMAGE_WORKER_RUNNING,
  IMAGE_WORKER_COMPLETE,
} image_worker_state_t;

typedef struct image_worker_t {
  // Publication lock for state, stopping and the request/result handoff.
  iree_slim_mutex_t mutex;
  // Wakes the worker for a newly published job or shutdown.
  iree_notification_t notification;
  // Thread reference; release joins before any borrowed state is freed.
  iree_thread_t* thread;
  // Single request slot's ownership phase, protected by mutex.
  image_worker_state_t state;
  // Stops after retiring the published job, if any; protected by mutex.
  bool stopping;
  // Input owned by the worker from READY through COMPLETE publication.
  image_job_t job;
  // Owned terminal generation/encoding result, published with COMPLETE.
  iree_status_t status;
  // Owned encoded result, moved back to the application at COMPLETE.
  iree_byte_span_t png;
} image_worker_t;

typedef struct image_service_t {
  // Allocator for queued inputs and encoded response storage.
  iree_allocator_t allocator;
  // Borrowed, serialized native model invocation.
  loom_serve_image_generator_t generator;
  // Borrowed network owner, outliving the worker.
  loom_serve_http_server_t* server;
  // Fixed model geometry and admission policy, borrowed for service_run.
  const loom_serve_image_service_options_t* options;
  // Full-image execution and cross-thread ownership handoff.
  image_worker_t worker;
  // Bounded FIFO of owned host requests, excluding the active image.
  image_pending_t* pending;
  // Number of occupied FIFO entries.
  iree_host_size_t pending_count;
  // Active request's peer; NULL after a reset while the image still runs.
  loom_serve_http_connection_t* connection;
  // Active identity remains nonzero until completion, including after reset.
  uint64_t active;
  // Last assigned request identity.
  uint64_t sequence;
  // Successfully generated images, including results discarded after reset.
  uint64_t completed;
  // Reusable application-side HTTP/JSON response storage.
  iree_string_builder_t response;
} image_service_t;

// Full-image compute runs outside the application and network ownership paths.
// COMPLETE is published only after RGB has been converted into owned bytes.
static int image_worker_main(void* argument) {
  image_service_t* service = argument;
  image_worker_t* worker = &service->worker;
  for (;;) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(&worker->notification);
    iree_slim_mutex_lock(&worker->mutex);
    const bool ready = worker->state == IMAGE_WORKER_READY;
    const bool stopping = worker->stopping;
    if (ready) {
      worker->state = IMAGE_WORKER_RUNNING;
    }
    iree_slim_mutex_unlock(&worker->mutex);
    if (ready) {
      iree_notification_cancel_wait(&worker->notification);
      iree_const_byte_span_t rgb = iree_const_byte_span_empty();
      iree_byte_span_t png = iree_byte_span_empty();
      iree_status_t status = service->generator.generate(
          service->generator.self, &worker->job.request, &rgb);
      const iree_time_t encode_begin = iree_time_now();
      if (iree_status_is_ok(status)) {
        status = loom_serve_image_encode_rgb_f32_png(
            service->options->width, service->options->height, rgb, &png,
            service->allocator);
      }
      if (iree_status_is_ok(status)) {
        fprintf(stderr,
                "{\"event\":\"image_encoding\",\"request\":%" PRIu64
                ",\"png_ns\":%" PRId64 "}\n",
                worker->job.id, iree_time_now() - encode_begin);
      }
      iree_slim_mutex_lock(&worker->mutex);
      worker->png = png;
      worker->status = status;
      worker->state = IMAGE_WORKER_COMPLETE;
      iree_slim_mutex_unlock(&worker->mutex);
      iree_notification_post(
          loom_serve_http_server_notification(service->server), 1);
    } else if (stopping) {
      iree_notification_cancel_wait(&worker->notification);
      break;
    } else {
      iree_notification_commit_wait(&worker->notification, token,
                                    IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  return 0;
}

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

static iree_status_t image_response_begin(image_service_t* service,
                                          const char* status) {
  iree_string_builder_reset(&service->response);
  return iree_string_builder_append_format(
      &service->response,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      status);
}

// Each connection sends exactly one complete response. The transport copies
// admitted bytes, so a slow or reset peer never holds a generator buffer.
static void image_response_send(image_service_t* service,
                                loom_serve_http_connection_t* connection) {
  // A fresh response has no occupied send slots. Zero budget therefore means
  // the carrier is closing; that can precede its failure callback publication.
  if (loom_serve_http_connection_failed(connection) ||
      !loom_serve_http_connection_can_send(connection)) {
    fprintf(stderr, "{\"event\":\"image_response_abandoned\"}\n");
    loom_serve_http_connection_abort(connection);
    return;
  }
  iree_status_t status = loom_serve_http_connection_send(
      connection, iree_string_builder_view(&service->response));
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
  } else {
    image_diagnose("Image response peer failed", status);
    loom_serve_http_connection_abort(connection);
  }
}

static iree_status_t image_reject(image_service_t* service,
                                  loom_serve_http_connection_t* connection,
                                  const char* code,
                                  iree_string_view_t message) {
  iree_status_t status = image_response_begin(service, code);
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
    image_response_send(service, connection);
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static void image_start(image_service_t* service, image_pending_t pending) {
  service->active = pending.job.id;
  service->connection = pending.connection;
  iree_slim_mutex_lock(&service->worker.mutex);
  service->worker.job = pending.job;
  service->worker.state = IMAGE_WORKER_READY;
  iree_slim_mutex_unlock(&service->worker.mutex);
  fprintf(stderr, "{\"event\":\"image_started\",\"request\":%" PRIu64 "}\n",
          service->active);
  iree_notification_post(&service->worker.notification, 1);
}

static iree_status_t image_result_json(image_service_t* service,
                                       iree_const_byte_span_t png) {
  IREE_RETURN_IF_ERROR(image_response_begin(service, "200 OK"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &service->response, "{\"created\":%" PRId64 ",\"data\":[{\"b64_json\":\"",
      iree_time_now() / INT64_C(1000000000)));
  const iree_host_size_t encoded_size =
      iree_base64_encoded_size(png.data_length);
  char* encoded = NULL;
  iree_host_size_t capacity = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      &service->response, encoded_size, &encoded, &capacity));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_base64_encode(
      png, iree_make_mutable_string_view(encoded, capacity), &length));
  iree_string_builder_commit_append(&service->response, length);
  return iree_string_builder_append_cstring(&service->response, "\"}]}");
}

static iree_status_t image_collect(image_service_t* service, bool* progress) {
  image_worker_t* worker = &service->worker;
  iree_slim_mutex_lock(&worker->mutex);
  if (worker->state != IMAGE_WORKER_COMPLETE) {
    iree_slim_mutex_unlock(&worker->mutex);
    return iree_ok_status();
  }
  const image_job_t job = worker->job;
  const iree_byte_span_t png = worker->png;
  iree_status_t status = worker->status;
  memset(&worker->job, 0, sizeof(worker->job));
  worker->png = iree_byte_span_empty();
  worker->status = iree_ok_status();
  worker->state = IMAGE_WORKER_IDLE;
  iree_slim_mutex_unlock(&worker->mutex);
  *progress = true;
  if (iree_status_is_ok(status)) {
    ++service->completed;
    fprintf(stderr,
            "{\"event\":\"image_generated\",\"request\":%" PRIu64
            ",\"png_bytes\":%zu,\"completed\":%" PRIu64 "}\n",
            job.id, png.data_length, service->completed);
    if (service->connection) {
      const iree_time_t encode_begin = iree_time_now();
      status = image_result_json(
          service, iree_make_const_byte_span(png.data, png.data_length));
      if (iree_status_is_ok(status)) {
        fprintf(stderr,
                "{\"event\":\"image_response_encoding\",\"request\":%" PRIu64
                ",\"json_ns\":%" PRId64 ",\"response_bytes\":%zu}\n",
                job.id, iree_time_now() - encode_begin,
                iree_string_builder_view(&service->response).size);
      }
    }
  }
  if (service->connection) {
    if (iree_status_is_ok(status)) {
      const iree_time_t send_begin = iree_time_now();
      image_response_send(service, service->connection);
      // The carrier copies/queues bytes here; client drain is asynchronous.
      fprintf(stderr,
              "{\"event\":\"image_response_transport\",\"request\":%" PRIu64
              ",\"send_call_ns\":%" PRId64 "}\n",
              job.id, iree_time_now() - send_begin);
    } else {
      loom_serve_http_connection_abort(service->connection);
    }
    service->connection = NULL;
  }
  service->active = 0;
  loom_serve_image_request_t request = job.request;
  loom_serve_image_request_deinitialize(&request);
  iree_allocator_free(service->allocator, png.data);
  return status;
}

static void image_cancel(loom_serve_http_connection_t* connection, uint64_t id,
                         const char* phase) {
  fprintf(stderr,
          "{\"event\":\"image_cancelled\",\"request\":%" PRIu64
          ",\"phase\":\"%s\"}\n",
          id, phase);
  loom_serve_http_connection_abort(connection);
}

static void image_prune(image_service_t* service, bool* progress) {
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

static iree_status_t image_health(image_service_t* service,
                                  loom_serve_http_connection_t* connection) {
  iree_status_t status = image_response_begin(service, "200 OK");
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &service->response,
        "{\"ready\":true,\"active\":%u,\"queued\":%zu,\"pending_capacity\":%zu,"
        "\"completed\":%" PRIu64 "}",
        service->active != 0, service->pending_count,
        service->options->pending_capacity, service->completed);
  }
  if (iree_status_is_ok(status)) {
    image_response_send(service, connection);
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t image_models(image_service_t* service,
                                  loom_serve_http_connection_t* connection) {
  iree_status_t status = image_response_begin(service, "200 OK");
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        &service->response, "{\"object\":\"list\",\"data\":[{\"id\":");
  }
  if (iree_status_is_ok(status)) {
    status = image_quote(&service->response, service->options->model);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &service->response,
        ",\"object\":\"model\",\"modality\":\"text-to-image\",\"size\":\"%ux%"
        "u\","
        "\"adapter_strength\":%s,\"reference_images\":false}]}",
        service->options->width, service->options->height,
        service->options->adapter_enabled ? "true" : "false");
  }
  if (iree_status_is_ok(status)) {
    image_response_send(service, connection);
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t image_admit(image_service_t* service,
                                 loom_serve_http_connection_t* connection,
                                 const loom_serve_http_request_t* request) {
  if (iree_string_view_equal(request->method, IREE_SV("GET"))) {
    if (iree_string_view_equal(request->target, IREE_SV("/healthz"))) {
      return image_health(service, connection);
    }
    if (iree_string_view_equal(request->target, IREE_SV("/v1/models"))) {
      return image_models(service, connection);
    }
  }
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
      request->body, service->options->model, service->options->width,
      service->options->height, &pending.job.request, service->allocator);
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
  if (!service->options->adapter_enabled &&
      pending.job.request.strength != 1.0f) {
    loom_serve_image_request_deinitialize(&pending.job.request);
    return image_reject(service, connection, "400 Bad Request",
                        IREE_SV("strength requires a loaded adapter"));
  }
  if (service->active &&
      service->pending_count == service->options->pending_capacity) {
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

static iree_status_t image_loop(image_service_t* service) {
  char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t address;
  IREE_RETURN_IF_ERROR(
      iree_async_address_format(loom_serve_http_server_address(service->server),
                                sizeof(storage), storage, &address));
  fprintf(stderr,
          "{\"event\":\"image_ready\",\"address\":\"%.*s\",\"width\":%u,"
          "\"height\":%u,\"pending_capacity\":%zu}\n",
          (int)address.size, address.data, service->options->width,
          service->options->height, service->options->pending_capacity);
  iree_notification_t* notification =
      loom_serve_http_server_notification(service->server);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(notification);
    bool progress = false;
    image_prune(service, &progress);
    status = image_collect(service, &progress);
    if (!iree_status_is_ok(status) ||
        loom_serve_http_server_is_stopping(service->server)) {
      iree_notification_cancel_wait(notification);
      break;
    }
    if (!service->active && service->pending_count) {
      const image_pending_t pending = service->pending[0];
      --service->pending_count;
      memmove(service->pending, service->pending + 1,
              service->pending_count * sizeof(*service->pending));
      image_start(service, pending);
      progress = true;
    }
    // Bound each admission cohort so continuous client traffic cannot postpone
    // completed-image publication or peer cancellation indefinitely.
    for (iree_host_size_t i = 0; i < 32 && iree_status_is_ok(status); ++i) {
      const loom_serve_http_request_t* request = NULL;
      loom_serve_http_connection_t* connection =
          loom_serve_http_server_take_request(service->server, &request);
      if (!connection) {
        break;
      }
      progress = true;
      status = image_admit(service, connection, request);
    }
    if (progress || !iree_status_is_ok(status)) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  return status;
}

iree_status_t loom_serve_image_service_run(
    loom_serve_image_generator_t generator, loom_serve_http_server_t* server,
    const loom_serve_image_service_options_t* options,
    iree_allocator_t host_allocator) {
  if (!generator.generate || !options->width || !options->height ||
      !options->model.size || !options->pending_capacity ||
      options->pending_capacity >
          IREE_HOST_SIZE_MAX / sizeof(image_pending_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image service requires a generator, geometry, "
                            "model and queue capacity");
  }
  image_service_t service = {.allocator = host_allocator,
                             .generator = generator,
                             .server = server,
                             .options = options};
  iree_slim_mutex_initialize(&service.worker.mutex);
  iree_notification_initialize(&service.worker.notification);
  iree_string_builder_initialize(host_allocator, &service.response);
  iree_status_t status = iree_allocator_malloc(
      host_allocator, options->pending_capacity * sizeof(*service.pending),
      (void**)&service.pending);
  if (iree_status_is_ok(status)) {
    const iree_thread_create_params_t params = {.name = IREE_SV("image-model")};
    status = iree_thread_create(image_worker_main, &service, params,
                                host_allocator, &service.worker.thread);
  }
  if (iree_status_is_ok(status)) {
    status = image_loop(&service);
  }
  fprintf(stderr, "{\"event\":\"image_shutdown\",\"active\":%" PRIu64 "}\n",
          service.active);
  for (iree_host_size_t i = 0; i < service.pending_count; ++i) {
    image_pending_t* pending = &service.pending[i];
    image_cancel(pending->connection, pending->job.id, "shutdown");
    loom_serve_image_request_deinitialize(&pending->job.request);
  }
  if (service.connection) {
    image_cancel(service.connection, service.active, "shutdown");
    service.connection = NULL;
  }
  iree_slim_mutex_lock(&service.worker.mutex);
  service.worker.stopping = true;
  iree_slim_mutex_unlock(&service.worker.mutex);
  iree_notification_post(&service.worker.notification, 1);
  iree_thread_release(service.worker.thread);
  bool progress = false;
  status = iree_status_join(status, image_collect(&service, &progress));
  iree_string_builder_deinitialize(&service.response);
  iree_allocator_free(host_allocator, service.pending);
  iree_notification_deinitialize(&service.worker.notification);
  iree_slim_mutex_deinitialize(&service.worker.mutex);
  fprintf(stderr, "{\"event\":\"image_stopped\",\"completed\":%" PRIu64 "}\n",
          service.completed);
  return status;
}
