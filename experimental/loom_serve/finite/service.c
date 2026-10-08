// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/finite/service.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "experimental/loom_serve/http/timer.h"
#include "iree/async/operations/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "loom/util/json.h"

typedef struct finite_job_t {
  // Owned raw JSON, independent of a reset transport while CPU preparation
  // runs.
  iree_string_view_t body;
  // Claimed response destination; NULL after cancellation.
  loom_serve_http_connection_t* connection;
  // Protocol input packet retained through the full cohort lifecycle.
  loom_serve_finite_request_t* request;
  // Source preparation retained across admission waits and incompatible
  // cohorts.
  loom_serve_finite_prepared_t prepared;
  // Request-scoped parsing/source rejection, consumed after CPU release.
  iree_status_t status;
  // Borrowed row views populated only during result encoding.
  iree_const_byte_span_t* rows;
  // Owned complete response, copied into transport before recycling the cohort.
  iree_string_builder_t response;
} finite_job_t;

typedef enum finite_phase_e {
  FINITE_IDLE = 0,
  FINITE_PREPARING,
  FINITE_BATCHING,
  FINITE_WAITING,
  FINITE_RUNNING,
  FINITE_ENCODING,
} finite_phase_t;

struct loom_serve_finite_service_t {
  // Allocator owning the queue, jobs and encoded results.
  iree_allocator_t allocator;
  // Shared source residency, borrowing the device owner through model lifetime.
  loom_serve_finite_model_t* model;
  // Immutable native modality contract, borrowed through service lifetime.
  const loom_serve_finite_protocol_t* protocol;
  // Borrowed catalog route identifier.
  iree_string_view_t name;
  // Bound on total accepted requests across all ownership phases.
  iree_host_size_t capacity;
  // Accepted jobs still owned by this service.
  iree_host_size_t count;
  // Borrowed common CPU executor.
  iree_task_executor_t* executor;
  // Borrowed host proactor for device dependency readiness.
  iree_async_proactor_t* proactor;
  // Scheduler-owned exclusive model invocation phase.
  finite_phase_t phase;
  // Stops new work while accepted CPU/device obligations retire.
  bool stopping;
  // Pending jobs in FIFO order; arrivals do not alter a live CPU snapshot.
  finite_job_t** pending;
  // Number of pending jobs.
  iree_host_size_t pending_count;
  // Stable CPU preparation snapshot or selected device cohort.
  finite_job_t** cohort;
  // Number of live entries in cohort.
  iree_host_size_t cohort_count;
  // Contiguous source request arguments captured for batch preparation.
  loom_serve_finite_prepared_t* prepared;
  // Protocol result row counts captured alongside the cohort.
  iree_host_size_t* rows;
  // Completion publication; callbacks only publish and wake.
  struct {
    // Joins final callback accesses with owner collection/destruction.
    iree_slim_mutex_t mutex;
    // Borrowed common router wake interface.
    loom_serve_http_observer_t observer;
    // CPU release or device readiness awaiting collection.
    bool ready;
    // Owned terminal result, consumed once after publication.
    iree_status_t status;
  } event;
  // Exclusive finite CPU process, recycled only after release.
  iree_task_process_t process;
  // Stable asynchronous ALL wait and its borrowed model frontier.
  struct {
    // Intrusive wait owned until its callback finishes.
    iree_async_semaphore_wait_operation_t operation;
    // Accepted work and feedback values.
    loom_serve_execution_frontier_t frontier;
    // Toll-free async handles of the same HAL semaphores.
    iree_async_semaphore_t* semaphores[2];
  } device;
  // Independent heartbeat readiness, drained before service destruction.
  loom_serve_http_timer_t timer;
  // Scheduler-side error response storage, disjoint from CPU job outputs.
  iree_string_builder_t response;
  // Completed device cohorts, independent of HTTP request count.
  uint64_t batches;
};

static iree_status_t finite_begin(iree_string_builder_t* output,
                                  const char* status) {
  iree_string_builder_reset(output);
  return iree_string_builder_append_format(
      output,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n",
      status);
}

static void finite_send(loom_serve_http_connection_t* connection,
                        iree_string_view_t response) {
  if (loom_serve_http_connection_failed(connection) ||
      !loom_serve_http_connection_can_send(connection)) {
    loom_serve_http_connection_abort(connection);
    return;
  }
  iree_status_t status = loom_serve_http_connection_send(connection, response);
  if (iree_status_is_ok(status)) {
    loom_serve_http_connection_finish(connection);
  } else {
    fprintf(stderr, "Finite response peer failed: ");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    loom_serve_http_connection_abort(connection);
  }
}

static iree_status_t finite_reject(loom_serve_finite_service_t* service,
                                   loom_serve_http_connection_t* connection,
                                   const char* code,
                                   iree_string_view_t message) {
  iree_status_t status = finite_begin(&service->response, code);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        &service->response,
        "{\"error\":{\"type\":\"finite_request_error\",\"message\":");
  }
  if (iree_status_is_ok(status)) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&service->response, &stream);
    status = loom_json_write_escaped_string(&stream, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&service->response, "}}");
  }
  if (iree_status_is_ok(status)) {
    finite_send(connection, iree_string_builder_view(&service->response));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static void finite_job_destroy(loom_serve_finite_service_t* service,
                               finite_job_t* job) {
  if (job->connection) {
    loom_serve_http_connection_abort(job->connection);
  }
  if (!iree_status_is_ok(job->status)) {
    fprintf(stderr, "Finite input rejected: ");
    iree_status_fprint(stderr, job->status);
    iree_status_free(job->status);
  }
  iree_vm_buffer_release(job->prepared.bytes);
  service->protocol->deinitialize(job->request);
  iree_string_builder_deinitialize(&job->response);
  iree_allocator_free(service->allocator, (void*)job->body.data);
  iree_allocator_free(service->allocator, job->rows);
  iree_allocator_free(service->allocator, job);
  --service->count;
}

static void finite_finish(loom_serve_finite_service_t* service) {
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    finite_job_destroy(service, service->cohort[i]);
  }
  service->cohort_count = 0;
  service->phase = FINITE_IDLE;
}

static iree_status_t finite_cpu_run(iree_task_process_t* process,
                                    const iree_task_worker_context_t* context,
                                    iree_task_process_drain_result_t* result) {
  (void)context;
  loom_serve_finite_service_t* service = process->user_data;
  result->completed = true;
  result->did_work = true;
  if (service->phase == FINITE_BATCHING) {
    return loom_serve_finite_model_prepare_batch(
        service->model, service->cohort_count, service->prepared,
        service->rows);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->cohort_count && iree_status_is_ok(status); ++i) {
    finite_job_t* job = service->cohort[i];
    if (service->phase == FINITE_PREPARING) {
      if (job->prepared.bytes) {
        continue;
      }
      job->status = service->protocol->initialize(job->body, job->request,
                                                  service->allocator);
      // Parser range failures are malformed user numbers, not device errors.
      if (iree_status_is_invalid_argument(job->status) ||
          iree_status_is_out_of_range(job->status)) {
        continue;
      }
      status = job->status;
      job->status = iree_ok_status();
      if (iree_status_is_ok(status)) {
        job->status = loom_serve_finite_model_prepare_request(
            service->model, job->request->input, &job->prepared);
        if (iree_status_is_invalid_argument(job->status)) {
          continue;
        }
        status = job->status;
        job->status = iree_ok_status();
      }
      if (iree_status_is_ok(status)) {
        status =
            iree_allocator_malloc_array(service->allocator, job->request->count,
                                        sizeof(*job->rows), (void**)&job->rows);
      }
      iree_allocator_free(service->allocator, (void*)job->body.data);
      job->body = iree_string_view_empty();
    } else {
      for (iree_host_size_t row = 0; row < job->request->count; ++row) {
        job->rows[row] = loom_serve_finite_model_row(service->model, i, row);
      }
      status = finite_begin(&job->response, "200 OK");
      if (iree_status_is_ok(status)) {
        status = service->protocol->response(
            job->request, service->name, job->rows,
            loom_serve_finite_model_input_tokens(service->model, i),
            &job->response);
      }
    }
  }
  return status;
}

static void finite_cpu_completed(iree_task_process_t* process,
                                 iree_status_t status) {
  loom_serve_finite_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.status = status;
  iree_slim_mutex_unlock(&service->event.mutex);
}

static void finite_cpu_released(iree_task_process_t* process) {
  loom_serve_finite_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.ready = true;
  service->event.observer.wake(service->event.observer.user_data);
  iree_slim_mutex_unlock(&service->event.mutex);
}

static void finite_schedule_cpu(loom_serve_finite_service_t* service,
                                finite_phase_t phase) {
  service->phase = phase;
  iree_task_process_initialize(finite_cpu_run, 0, 1, &service->process);
  service->process.user_data = service;
  service->process.completion_fn = finite_cpu_completed;
  service->process.release_fn = finite_cpu_released;
  iree_task_executor_schedule_process(service->executor, &service->process);
}

static void finite_device_completed(void* user_data,
                                    iree_async_operation_t* operation,
                                    iree_status_t status,
                                    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  loom_serve_finite_service_t* service = user_data;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.status = status;
  service->event.ready = true;
  service->event.observer.wake(service->event.observer.user_data);
  iree_slim_mutex_unlock(&service->event.mutex);
}

static iree_status_t finite_submit(loom_serve_finite_service_t* service) {
  IREE_RETURN_IF_ERROR(loom_serve_finite_model_submit(
      service->model, &service->device.frontier));
  iree_async_semaphore_wait_operation_t* wait = &service->device.operation;
  memset(wait, 0, sizeof(*wait));
  iree_async_operation_initialize(
      &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, finite_device_completed, service);
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
  service->phase = FINITE_RUNNING;
  fprintf(stderr, "{\"event\":\"%s_submitted\",\"requests\":%zu}\n",
          service->protocol->name, service->cohort_count);
  return iree_ok_status();
}

// The preparation snapshot precedes later arrivals. Filtering it in place and
// prepending survivors preserves age without sorting or per-session VM state.
static iree_status_t finite_prepared(loom_serve_finite_service_t* service) {
  iree_status_t status = iree_ok_status();
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    finite_job_t* job = service->cohort[i];
    if (!iree_status_is_ok(job->status) || !job->connection ||
        service->stopping) {
      if (job->connection && !service->stopping) {
        iree_string_view_t message = iree_status_message(job->status);
        if (!message.size) {
          message = IREE_SV("source rejected finite input");
        }
        status = iree_status_join(
            status, finite_reject(service, job->connection,
                                  service->protocol->invalid_status, message));
        job->connection = NULL;
      }
      finite_job_destroy(service, job);
    } else {
      service->cohort[retained++] = job;
    }
  }
  memmove(service->pending + retained, service->pending,
          service->pending_count * sizeof(*service->pending));
  memcpy(service->pending, service->cohort,
         retained * sizeof(*service->pending));
  service->pending_count += retained;
  service->cohort_count = 0;
  service->phase = FINITE_IDLE;
  return status;
}

static void finite_select(loom_serve_finite_service_t* service) {
  uint64_t remaining = loom_serve_finite_model_batch_limit(service->model);
  const uint64_t tag = service->pending[0]->prepared.tag;
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    finite_job_t* job = service->pending[i];
    if (job->prepared.bytes && job->prepared.tag == tag &&
        job->prepared.units <= remaining) {
      remaining -= job->prepared.units;
      const iree_host_size_t index = service->cohort_count++;
      service->cohort[index] = job;
      service->prepared[index] = job->prepared;
      service->rows[index] = job->request->count;
    } else {
      service->pending[retained++] = job;
    }
  }
  service->pending_count = retained;
  finite_schedule_cpu(service, FINITE_BATCHING);
}

static bool finite_is_active(void* self) {
  const loom_serve_finite_service_t* service = self;
  return service->phase != FINITE_IDLE && service->phase != FINITE_WAITING;
}

static bool finite_is_pending(void* self) {
  loom_serve_finite_service_t* service = self;
  return finite_is_active(service) ||
         loom_serve_http_timer_is_pending(&service->timer);
}

static iree_status_t finite_heartbeat(loom_serve_finite_service_t* service) {
  static const char* const phases[] = {"idle",    "preparing", "batching",
                                       "waiting", "running",   "encoding"};
  iree_string_builder_t* output = &service->response;
  iree_string_builder_reset(output);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"event\":\"%s_heartbeat\",\"time_ns\":%" PRId64 ",\"model\":",
      service->protocol->name, iree_time_now()));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, service->name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"phase\":\"%s\",\"queued\":%zu,\"cohort_requests\":%zu,"
      "\"batches\":%" PRIu64 "}\n",
      phases[service->phase], service->pending_count, service->cohort_count,
      service->batches));
  const iree_string_view_t line = iree_string_builder_view(output);
  fprintf(stderr, "%.*s", (int)line.size, line.data);
  return iree_ok_status();
}

static void finite_cancel_peers(loom_serve_finite_service_t* service) {
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    finite_job_t* job = service->pending[i];
    if (service->stopping ||
        loom_serve_http_connection_failed(job->connection)) {
      finite_job_destroy(service, job);
    } else {
      service->pending[retained++] = job;
    }
  }
  service->pending_count = retained;
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    finite_job_t* job = service->cohort[i];
    if (job->connection &&
        (service->stopping ||
         loom_serve_http_connection_failed(job->connection))) {
      loom_serve_http_connection_abort(job->connection);
      job->connection = NULL;
    }
  }
}

static iree_status_t finite_collect(loom_serve_finite_service_t* service,
                                    bool* progress) {
  iree_slim_mutex_lock(&service->event.mutex);
  const bool ready = service->event.ready;
  iree_status_t status = iree_ok_status();
  if (ready) {
    service->event.ready = false;
    status = service->event.status;
    service->event.status = iree_ok_status();
  }
  iree_slim_mutex_unlock(&service->event.mutex);
  if (!ready) {
    return iree_ok_status();
  }
  *progress = true;
  if (!iree_status_is_ok(status)) {
    // Model-owned payload/pin survives partial enqueue or failed readiness.
    finite_finish(service);
    return status;
  }
  if (service->phase == FINITE_PREPARING) {
    status = finite_prepared(service);
    if (iree_status_is_ok(status) && !service->stopping &&
        service->pending_count && service->pending[0]->prepared.bytes) {
      finite_select(service);
    }
  } else if (service->phase == FINITE_BATCHING) {
    service->phase = FINITE_WAITING;
  } else if (service->phase == FINITE_RUNNING) {
    loom_serve_finite_model_complete(service->model);
    finite_schedule_cpu(service, FINITE_ENCODING);
  } else {
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      finite_job_t* job = service->cohort[i];
      if (job->connection) {
        finite_send(job->connection, iree_string_builder_view(&job->response));
        job->connection = NULL;
      }
    }
    ++service->batches;
    fprintf(stderr,
            "{\"event\":\"%s_completed\",\"batches\":%" PRIu64
            ",\"requests\":%zu}\n",
            service->protocol->name, service->batches, service->cohort_count);
    loom_serve_finite_model_release_batch(service->model);
    finite_finish(service);
  }
  return status;
}

static iree_status_t finite_advance(void* self,
                                    loom_serve_http_service_flags_t flags,
                                    bool* out_progress) {
  loom_serve_finite_service_t* service = self;
  *out_progress = false;
  if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
    service->stopping = true;
  }
  finite_cancel_peers(service);
  iree_status_t status = finite_collect(service, out_progress);
  if (iree_status_is_ok(status) && service->phase == FINITE_WAITING) {
    bool has_peer = false;
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      has_peer |= service->cohort[i]->connection != NULL;
    }
    if (service->stopping || !has_peer) {
      loom_serve_finite_model_release_batch(service->model);
      finite_finish(service);
      *out_progress = true;
    } else {
      bool admitted = false;
      status = loom_serve_residency_try_acquire(
          loom_serve_finite_model_residency(service->model), &admitted);
      if (iree_status_is_ok(status) && admitted) {
        status = finite_submit(service);
        *out_progress = true;
      } else if (iree_status_is_ok(status) &&
                 !iree_any_bit_set(
                     flags, LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY)) {
        for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
          finite_job_t* job = service->cohort[i];
          if (job->connection) {
            status = iree_status_join(
                status,
                finite_reject(
                    service, job->connection, "503 Service Unavailable",
                    IREE_SV("model parameters exceed available capacity")));
            job->connection = NULL;
          }
        }
        loom_serve_finite_model_release_batch(service->model);
        finite_finish(service);
        *out_progress = true;
      }
      if (!iree_status_is_ok(status)) {
        finite_finish(service);
      }
    }
  }
  if (iree_status_is_ok(status) && !service->stopping &&
      service->phase == FINITE_IDLE && service->pending_count) {
    memcpy(service->cohort, service->pending,
           service->pending_count * sizeof(*service->cohort));
    service->cohort_count = service->pending_count;
    service->pending_count = 0;
    finite_schedule_cpu(service, FINITE_PREPARING);
    *out_progress = true;
  }
  bool timer_progress = false;
  bool tick = false;
  status = iree_status_join(
      status,
      loom_serve_http_timer_advance(
          &service->timer, service->stopping || !iree_status_is_ok(status),
          &timer_progress, &tick));
  *out_progress |= timer_progress;
  if (tick) {
    status = iree_status_join(status, finite_heartbeat(service));
  }
  return status;
}

static iree_status_t finite_accept(void* self,
                                   loom_serve_http_connection_t* connection,
                                   const loom_serve_http_request_t* request,
                                   loom_serve_http_service_flags_t flags) {
  (void)flags;
  loom_serve_finite_service_t* service = self;
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(
          request->target,
          iree_make_cstring_view(service->protocol->endpoint))) {
    return finite_reject(service, connection, "404 Not Found",
                         IREE_SV("unknown finite-result endpoint"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("content-type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return finite_reject(service, connection, "415 Unsupported Media Type",
                         IREE_SV("Content-Type must be application/json"));
  }
  if (service->count == service->capacity) {
    return finite_reject(service, connection, "503 Service Unavailable",
                         IREE_SV("finite admission queue is full"));
  }
  finite_job_t* job = NULL;
  iree_status_t status = iree_allocator_malloc(
      service->allocator,
      iree_sizeof_struct(finite_job_t) + service->protocol->request_size,
      (void**)&job);
  if (!iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    return status;
  }
  ++service->count;
  job->request =
      (loom_serve_finite_request_t*)((uint8_t*)job +
                                     iree_sizeof_struct(finite_job_t));
  job->connection = connection;
  iree_string_builder_initialize(service->allocator, &job->response);
  char* body = NULL;
  status = iree_allocator_clone(
      service->allocator,
      iree_make_const_byte_span(request->body.data, request->body.size),
      (void**)&body);
  if (iree_status_is_ok(status)) {
    job->body = iree_make_string_view(body, request->body.size);
    service->pending[service->pending_count++] = job;
  } else {
    finite_job_destroy(service, job);
  }
  return status;
}

static void finite_set_observer(void* self,
                                loom_serve_http_observer_t observer) {
  loom_serve_finite_service_t* service = self;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.observer = observer;
  iree_slim_mutex_unlock(&service->event.mutex);
  loom_serve_http_timer_set_observer(&service->timer, observer);
}

loom_serve_http_service_t loom_serve_finite_service_interface(
    loom_serve_finite_service_t* service) {
  return (loom_serve_http_service_t){
      .name = service->name,
      .self = service,
      .set_observer = finite_set_observer,
      .is_pending = finite_is_pending,
      .is_active = finite_is_active,
      .accept = finite_accept,
      .advance = finite_advance,
      .tokenizer = loom_serve_finite_model_tokenizer(service->model)};
}

void loom_serve_finite_service_destroy(loom_serve_finite_service_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!finite_is_pending(service));
  service->stopping = true;
  finite_cancel_peers(service);
  if (service->phase == FINITE_WAITING) {
    loom_serve_finite_model_release_batch(service->model);
  }
  finite_finish(service);
  loom_serve_http_timer_deinitialize(&service->timer);
  iree_string_builder_deinitialize(&service->response);
  iree_slim_mutex_deinitialize(&service->event.mutex);
  iree_allocator_free(service->allocator, service->pending);
  iree_allocator_free(service->allocator, service->cohort);
  iree_allocator_free(service->allocator, service->prepared);
  iree_allocator_free(service->allocator, service->rows);
  iree_allocator_free(service->allocator, service);
}

iree_status_t loom_serve_finite_service_create(
    loom_serve_host_t* host, loom_serve_finite_model_t* model,
    const loom_serve_finite_protocol_t* protocol, iree_string_view_t name,
    iree_host_size_t capacity, iree_duration_t heartbeat_interval,
    loom_serve_finite_service_t** out_service, iree_allocator_t allocator) {
  *out_service = NULL;
  if (!name.size || !capacity || heartbeat_interval < 0 ||
      capacity > SIZE_MAX / sizeof(loom_serve_finite_prepared_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "finite service requires a name and queue capacity");
  }
  loom_serve_finite_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*service), (void**)&service));
  service->allocator = allocator;
  service->model = model;
  service->protocol = protocol;
  service->name = name;
  service->capacity = capacity;
  service->executor = loom_serve_host_executor(host);
  iree_slim_mutex_initialize(&service->event.mutex);
  loom_serve_http_timer_initialize(&service->timer);
  iree_string_builder_initialize(allocator, &service->response);
  iree_status_t status = iree_allocator_malloc_array(allocator, capacity,
                                                     sizeof(*service->pending),
                                                     (void**)&service->pending);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, capacity,
                                         sizeof(*service->cohort),
                                         (void**)&service->cohort);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, capacity,
                                         sizeof(*service->prepared),
                                         (void**)&service->prepared);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        allocator, capacity, sizeof(*service->rows), (void**)&service->rows);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_get(loom_serve_host_proactor_pool(host),
                                          0, &service->proactor);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_timer_configure(&service->timer, service->proactor,
                                    heartbeat_interval);
    *out_service = service;
  } else {
    loom_serve_finite_service_destroy(service);
  }
  return status;
}
