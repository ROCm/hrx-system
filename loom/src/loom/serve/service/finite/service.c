// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/service.h"

#include <string.h>

#include "iree/async/operations/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "loom/serve/runtime/host/timer.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/service/execution_reconfiguration.h"
#include "loom/util/json.h"

typedef struct finite_job_t {
  // Immutable monotonic receipt time, preserved through every cohort phase.
  iree_time_t arrival_time;
  // Monotonic request lifetime ID; never reused with a queue slot.
  uint64_t id;
  // Completed model input work, including work for a cancelled peer.
  uint64_t input_tokens;
  // Final owner outcome; cancellation is the default until work is resolved.
  loom_serve_telemetry_outcome_t outcome;
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
  // Original device-cohort ordinal, preserved when cancelled peers are removed.
  iree_host_size_t result_index;
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

static const char finite_phases[][16] = {"idle",    "preparing", "batching",
                                         "waiting", "running",   "encoding"};

struct loom_serve_finite_service_t {
  // Allocator owning the queue, jobs and encoded results.
  iree_allocator_t host_allocator;
  // Shared source residency, borrowing the device owner through model lifetime.
  loom_serve_finite_model_t* model;
  // Candidate pool and selection, retired before the borrowed model.
  loom_serve_execution_reconfiguration_t* reconfiguration;
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
  // Detached exact result retained through CPU encoding process release.
  loom_serve_finite_result_t result;
  // Completion publication; callbacks only publish and wake.
  struct {
    // Joins final callback accesses with owner collection/destruction.
    iree_slim_mutex_t mutex;
    // Borrowed common router wake interface.
    loom_serve_wake_callback_t observer;
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
  loom_serve_timer_t timer;
  // Scheduler-side error response storage, disjoint from CPU job outputs.
  iree_string_builder_t response;
  // Producer-owned observation; neither counters nor publication wait on sinks.
  struct {
    // Bounded typed event storage with independent reader cursors.
    loom_serve_telemetry_stream_t* stream;
    // The service allocated this stream instead of borrowing its
    // registration's.
    bool owns_stream;
    // Cumulative work retained even when every event has been overwritten.
    loom_serve_telemetry_counters_t counters;
  } telemetry;
};

static uint64_t loom_serve_finite_request_begin(
    loom_serve_finite_service_t* service, iree_time_t time_ns) {
  const uint64_t id = ++service->telemetry.counters.requests;
  const loom_serve_telemetry_record_t record = {
      .time_ns = time_ns,
      .request = id,
      .event = &LOOM_SERVE_TELEMETRY_REQUEST_BEGIN};
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
  return id;
}

static void loom_serve_finite_request_end(
    loom_serve_finite_service_t* service, uint64_t id, iree_time_t arrival_time,
    uint64_t input_tokens, loom_serve_telemetry_outcome_t outcome,
    const iree_status_t status) {
  loom_serve_telemetry_count_outcome(&service->telemetry.counters, outcome);
  const iree_time_t now = iree_time_now();
  loom_serve_telemetry_record_t record = {
      .time_ns = now,
      .request = id,
      .event = &LOOM_SERVE_TELEMETRY_REQUEST_END,
      .data.end = {.outcome = outcome,
                   .duration_ns = now - arrival_time,
                   .input_tokens = input_tokens}};
  loom_serve_telemetry_diagnostic_capture(status, &record.data.end.diagnostic);
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

static void loom_serve_finite_error(loom_serve_finite_service_t* service,
                                    uint64_t id, const iree_status_t status) {
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .request = id,
                                          .event = &LOOM_SERVE_TELEMETRY_ERROR};
  loom_serve_telemetry_diagnostic_capture(status, &record.data.error);
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

static void loom_serve_finite_batch_event(
    loom_serve_finite_service_t* service,
    const loom_serve_telemetry_event_t* event, uint64_t id) {
  const loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .event = event,
      .data.batch = {.id = id, .requests = service->cohort_count}};
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

static iree_status_t loom_serve_finite_begin(iree_string_builder_t* output,
                                             const char* status) {
  iree_string_builder_reset(output);
  return iree_string_builder_append_format(
      output,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n",
      status);
}

static loom_serve_telemetry_outcome_t loom_serve_finite_send(
    loom_serve_http_connection_t* connection, iree_string_view_t response,
    iree_status_t* out_status) {
  *out_status = iree_ok_status();
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
    *out_status = status;
    loom_serve_http_connection_abort(connection);
    return LOOM_SERVE_TELEMETRY_FAILED;
  }
}

static iree_status_t loom_serve_finite_reject(
    loom_serve_finite_service_t* service, uint64_t id,
    loom_serve_http_connection_t* connection, const char* code,
    iree_string_view_t message) {
  iree_status_t status = loom_serve_finite_begin(&service->response, code);
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
    iree_status_t peer_status = iree_ok_status();
    loom_serve_finite_send(
        connection, iree_string_builder_view(&service->response), &peer_status);
    if (!iree_status_is_ok(peer_status)) {
      loom_serve_finite_error(service, id, peer_status);
      iree_status_free(peer_status);
    }
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

// Rejections before job allocation still have complete request lifetimes.
static iree_status_t loom_serve_finite_decline(
    loom_serve_finite_service_t* service, uint64_t id, iree_time_t arrival_time,
    loom_serve_http_connection_t* connection, const char* code,
    iree_string_view_t message) {
  iree_status_t status =
      loom_serve_finite_reject(service, id, connection, code, message);
  loom_serve_finite_request_end(service, id, arrival_time, 0,
                                iree_status_is_ok(status)
                                    ? LOOM_SERVE_TELEMETRY_REJECTED
                                    : LOOM_SERVE_TELEMETRY_FAILED,
                                status);
  return status;
}

static void loom_serve_finite_job_destroy(loom_serve_finite_service_t* service,
                                          finite_job_t* job,
                                          const iree_status_t terminal_status) {
  if (job->connection) {
    loom_serve_http_connection_abort(job->connection);
  }
  loom_serve_finite_request_end(
      service, job->id, job->arrival_time, job->input_tokens,
      iree_status_is_ok(terminal_status) ? job->outcome
                                         : LOOM_SERVE_TELEMETRY_FAILED,
      iree_status_is_ok(terminal_status) ? job->status : terminal_status);
  iree_status_free(job->status);
  iree_vm_buffer_release(job->prepared.bytes);
  service->protocol->deinitialize(job->request);
  iree_string_builder_deinitialize(&job->response);
  iree_allocator_free(service->host_allocator, (void*)job->body.data);
  iree_allocator_free(service->host_allocator, job->rows);
  iree_allocator_free(service->host_allocator, job);
  --service->count;
}

static void loom_serve_finite_finish(loom_serve_finite_service_t* service,
                                     const iree_status_t terminal_status) {
  loom_serve_residency_cancel_admission(
      loom_serve_finite_model_residency(service->model));
  loom_serve_finite_result_deinitialize(&service->result);
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    loom_serve_finite_job_destroy(service, service->cohort[i], terminal_status);
  }
  service->cohort_count = 0;
  service->phase = FINITE_IDLE;
}

static iree_status_t loom_serve_finite_cpu_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
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
      job->status = service->protocol->initialize(
          job->body, service->host_allocator, job->request);
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
        status = iree_allocator_malloc_array(
            service->host_allocator, job->request->count, sizeof(*job->rows),
            (void**)&job->rows);
      }
      iree_allocator_free(service->host_allocator, (void*)job->body.data);
      job->body = iree_string_view_empty();
    } else {
      for (iree_host_size_t row = 0; row < job->request->count; ++row) {
        job->rows[row] = loom_serve_finite_result_row(&service->result,
                                                      job->result_index, row);
      }
      status = loom_serve_finite_begin(&job->response, "200 OK");
      if (iree_status_is_ok(status)) {
        status =
            service->protocol->response(job->request, service->name, job->rows,
                                        job->input_tokens, &job->response);
      }
    }
  }
  return status;
}

static void loom_serve_finite_cpu_completed(iree_task_process_t* process,
                                            iree_status_t status) {
  loom_serve_finite_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.status = status;
  iree_slim_mutex_unlock(&service->event.mutex);
}

static void loom_serve_finite_cpu_released(iree_task_process_t* process) {
  loom_serve_finite_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.ready = true;
  service->event.observer.wake(service->event.observer.user_data);
  iree_slim_mutex_unlock(&service->event.mutex);
}

static void loom_serve_finite_schedule_cpu(loom_serve_finite_service_t* service,
                                           finite_phase_t phase) {
  service->phase = phase;
  iree_task_process_initialize(loom_serve_finite_cpu_run, 0, 1,
                               &service->process);
  service->process.user_data = service;
  service->process.completion_fn = loom_serve_finite_cpu_completed;
  service->process.release_fn = loom_serve_finite_cpu_released;
  iree_task_executor_schedule_process(service->executor, &service->process);
}

static void loom_serve_finite_device_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  loom_serve_finite_service_t* service = user_data;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.status = status;
  service->event.ready = true;
  service->event.observer.wake(service->event.observer.user_data);
  iree_slim_mutex_unlock(&service->event.mutex);
}

static iree_status_t loom_serve_finite_submit(
    loom_serve_finite_service_t* service) {
  IREE_RETURN_IF_ERROR(loom_serve_finite_model_submit(
      service->model, &service->device.frontier));
  iree_async_semaphore_wait_operation_t* wait = &service->device.operation;
  memset(wait, 0, sizeof(*wait));
  iree_async_operation_initialize(&wait->base,
                                  IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  loom_serve_finite_device_completed, service);
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
  loom_serve_finite_batch_event(service, &LOOM_SERVE_TELEMETRY_BATCH_SUBMIT,
                                service->telemetry.counters.batches + 1);
  return iree_ok_status();
}

// The preparation snapshot precedes later arrivals. Filtering it in place and
// prepending survivors preserves age without sorting or per-session VM state.
static iree_status_t loom_serve_finite_prepared(
    loom_serve_finite_service_t* service) {
  iree_status_t status = iree_ok_status();
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    finite_job_t* job = service->cohort[i];
    if (!iree_status_is_ok(job->status) || !job->connection ||
        service->stopping) {
      if (!iree_status_is_ok(job->status)) {
        job->outcome = LOOM_SERVE_TELEMETRY_REJECTED;
      }
      if (job->connection && !service->stopping) {
        iree_string_view_t message = iree_status_message(job->status);
        if (!message.size) {
          message = IREE_SV("source rejected finite input");
        }
        status = iree_status_join(
            status, loom_serve_finite_reject(service, job->id, job->connection,
                                             service->protocol->invalid_status,
                                             message));
        job->connection = NULL;
      }
      loom_serve_finite_job_destroy(service, job, iree_ok_status());
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

static void loom_serve_finite_select(loom_serve_finite_service_t* service) {
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
  loom_serve_finite_schedule_cpu(service, FINITE_BATCHING);
}

static bool loom_serve_finite_is_active(void* self) {
  const loom_serve_finite_service_t* service = self;
  return service->phase != FINITE_IDLE && service->phase != FINITE_WAITING;
}

static bool loom_serve_finite_is_pending(void* self) {
  loom_serve_finite_service_t* service = self;
  return loom_serve_finite_is_active(service) ||
         loom_serve_execution_reconfiguration_pending(
             service->reconfiguration) ||
         loom_serve_timer_is_pending(&service->timer);
}

static loom_serve_http_service_snapshot_t loom_serve_finite_snapshot(
    void* self) {
  const loom_serve_finite_service_t* service = self;
  const iree_time_t now = iree_time_now();
  return (loom_serve_http_service_snapshot_t){
      .phase = finite_phases[service->phase],
      .queued_requests = service->pending_count,
      .active_requests = service->cohort_count,
      .oldest_queued_age =
          service->pending_count ? now - service->pending[0]->arrival_time : 0,
      .oldest_active_age =
          service->cohort_count ? now - service->cohort[0]->arrival_time : 0};
}

static void loom_serve_finite_heartbeat(loom_serve_finite_service_t* service) {
  loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .event = &LOOM_SERVE_TELEMETRY_SNAPSHOT,
      .data.snapshot = {.counters = service->telemetry.counters,
                        .queued = service->pending_count,
                        .active = service->cohort_count}};
  memcpy(record.data.snapshot.phase, finite_phases[service->phase],
         sizeof(record.data.snapshot.phase));
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

static bool loom_serve_finite_request_at(
    void* self, iree_host_size_t index,
    loom_serve_http_request_snapshot_t* out_snapshot) {
  const loom_serve_finite_service_t* service = self;
  const finite_job_t* job = NULL;
  const char* phase = "queued";
  if (index < service->pending_count) {
    job = service->pending[index];
  } else if (index - service->pending_count < service->cohort_count) {
    job = service->cohort[index - service->pending_count];
    phase = finite_phases[service->phase];
  } else {
    return false;
  }
  *out_snapshot = (loom_serve_http_request_snapshot_t){
      .id = job->id,
      .phase = phase,
      .arrival_time = job->arrival_time,
      .counters = {.requests = 1,
                   .input_tokens = job->input_tokens,
                   .batches = index >= service->pending_count &&
                              service->phase == FINITE_ENCODING}};
  return true;
}

static void loom_serve_finite_cancel_peers(
    loom_serve_finite_service_t* service) {
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    finite_job_t* job = service->pending[i];
    if (service->stopping ||
        loom_serve_http_connection_failed(job->connection)) {
      loom_serve_finite_job_destroy(service, job, iree_ok_status());
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

static void loom_serve_finite_complete_execution(
    loom_serve_finite_service_t* service) {
  loom_serve_finite_model_complete(service->model);
  loom_serve_finite_model_take_result(service->model, &service->result);
  loom_serve_finite_batch_event(service, &LOOM_SERVE_TELEMETRY_BATCH_END,
                                ++service->telemetry.counters.batches);
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    finite_job_t* job = service->cohort[i];
    job->input_tokens =
        loom_serve_finite_result_input_tokens(&service->result, i);
    service->telemetry.counters.input_tokens += job->input_tokens;
    if (!job->connection) {
      loom_serve_finite_job_destroy(service, job, iree_ok_status());
      continue;
    }
    // The surviving encoding cohort can have holes in the original result
    // plan. Preserve that ordinal without exposing mutable connection state
    // to the CPU borrower or moving its per-row work onto the shared owner.
    job->result_index = i;
    service->cohort[retained++] = job;
  }
  service->cohort_count = retained;
  if (retained) {
    loom_serve_finite_schedule_cpu(service, FINITE_ENCODING);
  } else {
    loom_serve_finite_finish(service, iree_ok_status());
  }
}

static iree_status_t loom_serve_finite_collect(
    loom_serve_finite_service_t* service, bool* progress) {
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
    loom_serve_finite_error(service, 0, status);
    loom_serve_finite_finish(service, status);
    return status;
  }
  if (service->phase == FINITE_PREPARING) {
    status = loom_serve_finite_prepared(service);
    if (iree_status_is_ok(status) && !service->stopping &&
        service->pending_count && service->pending[0]->prepared.bytes) {
      loom_serve_finite_select(service);
    }
  } else if (service->phase == FINITE_BATCHING) {
    service->phase = FINITE_WAITING;
  } else if (service->phase == FINITE_RUNNING) {
    loom_serve_finite_complete_execution(service);
  } else {
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      finite_job_t* job = service->cohort[i];
      if (job->connection) {
        job->outcome = loom_serve_finite_send(
            job->connection, iree_string_builder_view(&job->response),
            &job->status);
        job->connection = NULL;
      }
    }
    loom_serve_finite_finish(service, iree_ok_status());
  }
  return status;
}

static iree_status_t loom_serve_finite_advance(
    void* self, loom_serve_http_service_flags_t flags, bool* out_progress) {
  loom_serve_finite_service_t* service = self;
  *out_progress = false;
  if (iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
    service->stopping = true;
  }
  loom_serve_finite_cancel_peers(service);
  iree_status_t status = loom_serve_finite_collect(service, out_progress);
  *out_progress |= loom_serve_execution_reconfiguration_advance(
      service->reconfiguration,
      iree_status_is_ok(status) &&
          (service->phase == FINITE_IDLE || service->phase == FINITE_WAITING),
      service->stopping || !iree_status_is_ok(status));
  if (iree_status_is_ok(status) && service->phase == FINITE_WAITING) {
    bool has_peer = false;
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      has_peer |= service->cohort[i]->connection != NULL;
    }
    if (service->stopping || !has_peer) {
      loom_serve_finite_model_release_batch(service->model);
      loom_serve_finite_finish(service, iree_ok_status());
      *out_progress = true;
    } else {
      loom_serve_residency_admission_t admission;
      status = loom_serve_residency_try_admit(
          loom_serve_finite_model_residency(service->model), &admission);
      if (iree_status_is_ok(status) &&
          admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
        status = loom_serve_finite_submit(service);
        *out_progress = true;
      } else if (iree_status_is_ok(status) &&
                 admission == LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY &&
                 !iree_any_bit_set(
                     flags, LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY)) {
        for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
          finite_job_t* job = service->cohort[i];
          job->outcome = LOOM_SERVE_TELEMETRY_REJECTED;
          if (job->connection) {
            status = iree_status_join(
                status,
                loom_serve_finite_reject(
                    service, job->id, job->connection,
                    "503 Service Unavailable",
                    IREE_SV("model parameters exceed available capacity")));
            job->connection = NULL;
          }
        }
        loom_serve_finite_model_release_batch(service->model);
        loom_serve_finite_finish(service, status);
        *out_progress = true;
      }
      if (!iree_status_is_ok(status)) {
        loom_serve_finite_error(service, 0, status);
        loom_serve_finite_finish(service, status);
      }
    }
  }
  if (iree_status_is_ok(status) && !service->stopping &&
      service->phase == FINITE_IDLE && service->pending_count) {
    memcpy(service->cohort, service->pending,
           service->pending_count * sizeof(*service->cohort));
    service->cohort_count = service->pending_count;
    service->pending_count = 0;
    loom_serve_finite_schedule_cpu(service, FINITE_PREPARING);
    *out_progress = true;
  }
  bool timer_progress = false;
  bool tick = false;
  status = iree_status_join(
      status,
      loom_serve_timer_advance(&service->timer,
                               service->stopping || !iree_status_is_ok(status),
                               &timer_progress, &tick));
  *out_progress |= timer_progress;
  if (tick) {
    loom_serve_finite_heartbeat(service);
  }
  return status;
}

static iree_status_t loom_serve_finite_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_http_service_flags_t flags) {
  loom_serve_finite_service_t* service = self;
  const iree_time_t arrival_time = iree_time_now();
  const uint64_t id = loom_serve_finite_request_begin(service, arrival_time);
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(
          request->target,
          iree_make_cstring_view(service->protocol->endpoint))) {
    return loom_serve_finite_decline(service, id, arrival_time, connection,
                                     "404 Not Found",
                                     IREE_SV("unknown finite-result endpoint"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("content-type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return loom_serve_finite_decline(
        service, id, arrival_time, connection, "415 Unsupported Media Type",
        IREE_SV("Content-Type must be application/json"));
  }
  if (service->count == service->capacity) {
    return loom_serve_finite_decline(service, id, arrival_time, connection,
                                     "503 Service Unavailable",
                                     IREE_SV("finite admission queue is full"));
  }
  finite_job_t* job = NULL;
  iree_status_t status = iree_allocator_malloc(
      service->host_allocator,
      iree_sizeof_struct(finite_job_t) + service->protocol->request_size,
      (void**)&job);
  if (!iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    loom_serve_finite_request_end(service, id, arrival_time, 0,
                                  LOOM_SERVE_TELEMETRY_FAILED, status);
    return status;
  }
  ++service->count;
  job->id = id;
  job->arrival_time = arrival_time;
  job->outcome = LOOM_SERVE_TELEMETRY_CANCELLED;
  job->request =
      (loom_serve_finite_request_t*)((uint8_t*)job +
                                     iree_sizeof_struct(finite_job_t));
  job->connection = connection;
  iree_string_builder_initialize(service->host_allocator, &job->response);
  char* body = NULL;
  status = iree_allocator_clone(
      service->host_allocator,
      iree_make_const_byte_span(request->body.data, request->body.size),
      (void**)&body);
  if (iree_status_is_ok(status)) {
    job->body = iree_make_string_view(body, request->body.size);
    service->pending[service->pending_count++] = job;
  } else {
    loom_serve_finite_job_destroy(service, job, status);
  }
  return status;
}

static void loom_serve_finite_set_observer(
    void* self, loom_serve_wake_callback_t observer) {
  loom_serve_finite_service_t* service = self;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.observer = observer;
  iree_slim_mutex_unlock(&service->event.mutex);
  loom_serve_timer_set_observer(&service->timer, observer);
  loom_serve_execution_reconfiguration_set_observer(service->reconfiguration,
                                                    observer);
}

loom_serve_http_service_t loom_serve_finite_service_interface(
    loom_serve_finite_service_t* service) {
  static const iree_string_view_t input_modalities[] = {IREE_SVL("text")};
  return (loom_serve_http_service_t){
      .name = service->name,
      .self = service,
      .set_observer = loom_serve_finite_set_observer,
      .is_pending = loom_serve_finite_is_pending,
      .is_active = loom_serve_finite_is_active,
      .accept = loom_serve_finite_accept,
      .advance = loom_serve_finite_advance,
      .tokenizer = loom_serve_finite_model_tokenizer(service->model),
      .kind = service->protocol->name,
      .endpoint = service->protocol->endpoint,
      .capabilities =
          {.input_modalities = {IREE_ARRAYSIZE(input_modalities),
                                input_modalities},
           .output_modalities = {1, &service->protocol->output_modality}},
      .snapshot = loom_serve_finite_snapshot,
      .inspection = {.request_at = loom_serve_finite_request_at},
      .reconfiguration = loom_serve_execution_reconfiguration_interface(
          service->reconfiguration),
      .capacity_domain = loom_serve_residency_cache(
          loom_serve_finite_model_residency(service->model)),
      .telemetry = {.stream = service->telemetry.stream,
                    .counters = &service->telemetry.counters}};
}

void loom_serve_finite_service_free(loom_serve_finite_service_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!loom_serve_finite_is_pending(service));
  service->stopping = true;
  loom_serve_finite_cancel_peers(service);
  if (service->phase == FINITE_WAITING) {
    loom_serve_finite_model_release_batch(service->model);
  }
  loom_serve_finite_finish(service, iree_ok_status());
  loom_serve_execution_reconfiguration_free(service->reconfiguration);
  loom_serve_timer_deinitialize(&service->timer);
  iree_string_builder_deinitialize(&service->response);
  iree_slim_mutex_deinitialize(&service->event.mutex);
  iree_allocator_free(service->host_allocator, service->pending);
  iree_allocator_free(service->host_allocator, service->cohort);
  iree_allocator_free(service->host_allocator, service->prepared);
  iree_allocator_free(service->host_allocator, service->rows);
  if (service->telemetry.owns_stream) {
    loom_serve_telemetry_stream_free(service->telemetry.stream);
  }
  iree_allocator_free_aligned(service->host_allocator, service);
}

iree_status_t loom_serve_finite_service_allocate(
    loom_serve_host_t* host, loom_serve_finite_model_t* model,
    const loom_serve_finite_protocol_t* protocol, iree_string_view_t name,
    iree_host_size_t capacity, iree_host_size_t variant_capacity,
    iree_duration_t heartbeat_interval, loom_serve_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_serve_finite_service_t** out_service) {
  *out_service = NULL;
  if (!name.size || !capacity || heartbeat_interval < 0 ||
      capacity > SIZE_MAX / sizeof(loom_serve_finite_prepared_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "finite service requires a name and queue capacity");
  }
  loom_serve_finite_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*service),
      iree_alignof(loom_serve_finite_service_t), 0, (void**)&service));
  service->host_allocator = host_allocator;
  service->model = model;
  service->protocol = protocol;
  service->name = name;
  service->capacity = capacity;
  service->telemetry.stream = stream;
  service->telemetry.owns_stream = !stream;
  service->executor = loom_serve_host_executor(host);
  iree_slim_mutex_initialize(&service->event.mutex);
  loom_serve_timer_initialize(&service->timer);
  iree_string_builder_initialize(host_allocator, &service->response);
  iree_status_t status = iree_allocator_malloc_array(host_allocator, capacity,
                                                     sizeof(*service->pending),
                                                     (void**)&service->pending);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, capacity,
                                         sizeof(*service->cohort),
                                         (void**)&service->cohort);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, capacity,
                                         sizeof(*service->prepared),
                                         (void**)&service->prepared);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, capacity,
                                         sizeof(*service->rows),
                                         (void**)&service->rows);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_get(loom_serve_host_proactor_pool(host),
                                          0, &service->proactor);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_reconfiguration_allocate(
        loom_serve_finite_model_execution_update(model), variant_capacity,
        host_allocator, &service->reconfiguration);
  }
  if (iree_status_is_ok(status) && !stream) {
    status = loom_serve_telemetry_stream_allocate(128, host_allocator,
                                                  &service->telemetry.stream);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_timer_configure(&service->timer, service->proactor,
                               heartbeat_interval);
    const loom_serve_telemetry_record_t record = {
        .time_ns = iree_time_now(),
        .event = stream ? &LOOM_SERVE_TELEMETRY_SNAPSHOT
                        : &LOOM_SERVE_TELEMETRY_STREAM_BEGIN,
        .data.snapshot.phase = "idle"};
    loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
    *out_service = service;
  } else {
    loom_serve_finite_service_free(service);
  }
  return status;
}
