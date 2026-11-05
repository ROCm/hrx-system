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
#include "loom/serve/service/finite/http.h"
#include "loom/serve/service/finite/result.h"
#include "loom/serve/service/kind.h"
#include "loom/service/telemetry/record.h"
#include "loom/service/telemetry/tracing.h"

typedef struct loom_serve_finite_job_t {
  // Immutable monotonic receipt time, preserved through every cohort phase.
  iree_time_t arrival_time;
  // Monotonic request lifetime ID; never reused with a queue slot.
  uint64_t id;
  // Completed model input work, including work for a cancelled peer.
  uint64_t input_tokens;
  // Final owner outcome; cancellation is the default until work is resolved.
  loom_service_telemetry_outcome_t outcome;
  // Immutable input borrowed from the consumer through preparation release.
  iree_string_view_t body;
  // Owned receipt; interest and source/CPU input lifetime are independent.
  loom_serve_finite_consumer_t consumer;
  // Protocol input packet retained through the full cohort lifecycle.
  loom_serve_finite_request_t* request;
  // Source preparation retained across admission waits and incompatible
  // cohorts.
  loom_serve_prepared_request_t prepared;
  // Request-scoped parsing/source rejection, consumed after CPU release.
  iree_status_t status;
  // Original device-cohort ordinal, preserved when cancelled peers are removed.
  iree_host_size_t result_index;
  // Borrowed row views populated only during result encoding.
  iree_const_byte_span_t* rows;
  // Owned complete response, copied into transport before recycling the cohort.
  iree_string_builder_t response;
} loom_serve_finite_job_t;

typedef enum loom_serve_finite_phase_e {
  FINITE_IDLE = 0,
  FINITE_PREPARING,
  FINITE_BATCHING,
  FINITE_SPECIALIZING,
  FINITE_WAITING,
  FINITE_RUNNING,
  FINITE_ENCODING,
} loom_serve_finite_phase_t;

static const char finite_phases[][16] = {
    "idle",    "preparing", "batching", "specializing",
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
  loom_serve_finite_phase_t phase;
  // Stops new work while accepted CPU/device obligations retire.
  bool stopping;
  // First terminal failure retained across outstanding CPU/source borrowers.
  iree_status_t failure;
  // Pending jobs in FIFO order; arrivals do not alter a live CPU snapshot.
  loom_serve_finite_job_t** pending;
  // Number of pending jobs.
  iree_host_size_t pending_count;
  // Stable CPU preparation snapshot or selected device cohort.
  loom_serve_finite_job_t** cohort;
  // Number of live entries in cohort.
  iree_host_size_t cohort_count;
  // Contiguous source request arguments captured for batch preparation.
  loom_serve_prepared_request_t* prepared;
  // Protocol result row counts captured alongside the cohort.
  iree_host_size_t* rows;
  // Checked device-cohort metadata retained until successful readiness or
  // failure collection. Input work is counted only at that boundary.
  loom_serve_finite_result_t* result;
  // Explicit host materialization retained through CPU encoding release.
  loom_serve_finite_snapshot_t* snapshot;
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
    // Accepted batch through readiness collection, including result download.
    IREE_TRACE(iree_section_id_t section;)
  } device;
  // Independent heartbeat readiness, drained before service destruction.
  loom_serve_timer_t timer;
  // Scheduler-side error response storage, disjoint from CPU job outputs.
  iree_string_builder_t response;
  // Producer-owned observation; neither counters nor publication wait on sinks.
  struct {
    // Bounded typed event storage with independent reader cursors.
    loom_runtime_telemetry_stream_t* stream;
    // The service allocated this stream instead of borrowing its
    // registration's.
    bool owns_stream;
    // Cumulative work retained even when every event has been overwritten.
    loom_service_telemetry_counters_t counters;
    // Request latency distributions retained independently of log readers.
    loom_service_telemetry_latency_t latency;
  } telemetry;
};

static uint64_t loom_serve_finite_request_begin(
    loom_serve_finite_service_t* service, iree_time_t time_ns) {
  const uint64_t id = ++service->telemetry.counters.requests;
  const loom_runtime_telemetry_record_t record = {
      .time_ns = time_ns,
      .request = id,
      .event = &LOOM_SERVICE_TELEMETRY_REQUEST_BEGIN};
  loom_runtime_telemetry_stream_append(service->telemetry.stream, &record);
  return id;
}

static void loom_serve_finite_request_end(
    loom_serve_finite_service_t* service, uint64_t id, iree_time_t arrival_time,
    uint64_t input_tokens, loom_service_telemetry_outcome_t outcome,
    const iree_status_t status) {
  loom_service_telemetry_count_outcome(&service->telemetry.counters, outcome);
  const iree_time_t now = iree_time_now();
  loom_runtime_telemetry_record_t record = {
      .time_ns = now,
      .request = id,
      .event = &LOOM_SERVICE_TELEMETRY_REQUEST_END};
  loom_service_telemetry_end_record_t telemetry_payload = {
      .outcome = outcome,
      .duration_ns = now - arrival_time,
      .input_tokens = input_tokens};
  loom_service_telemetry_latency_observe(
      &service->telemetry.latency.outcomes[outcome],
      telemetry_payload.duration_ns);
  loom_service_telemetry_diagnostic_capture(status,
                                            &telemetry_payload.diagnostic);
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(service->telemetry.stream, &record);
}

static void loom_serve_finite_error(loom_serve_finite_service_t* service,
                                    uint64_t id, const iree_status_t status) {
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .request = id,
      .event = &LOOM_SERVICE_TELEMETRY_ERROR};
  loom_service_telemetry_diagnostic_t telemetry_payload = {0};
  loom_service_telemetry_diagnostic_capture(status, &telemetry_payload);
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(service->telemetry.stream, &record);
}

static void loom_serve_finite_batch_event(
    loom_serve_finite_service_t* service,
    const loom_runtime_telemetry_event_t* event, uint64_t id) {
  loom_runtime_telemetry_record_t record = {.time_ns = iree_time_now(),
                                            .event = event};
  loom_service_telemetry_batch_record_t telemetry_payload = {
      .id = id, .requests = service->cohort_count};
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(service->telemetry.stream, &record);
}

// Rejections before job allocation still have complete request lifetimes.
static void loom_serve_finite_decline(loom_serve_finite_service_t* service,
                                      uint64_t id, iree_time_t arrival_time,
                                      loom_serve_http_connection_t* connection,
                                      const char* code,
                                      iree_string_view_t message) {
  bool delivered = false;
  iree_status_t status =
      loom_serve_finite_http_reject(connection, code, "finite_request_error",
                                    message, &service->response, &delivered);
  if (!iree_status_is_ok(status)) {
    loom_serve_finite_error(service, id, status);
  }
  loom_serve_finite_request_end(service, id, arrival_time, 0,
                                !iree_status_is_ok(status)
                                    ? LOOM_SERVICE_TELEMETRY_FAILED
                                : delivered ? LOOM_SERVICE_TELEMETRY_REJECTED
                                            : LOOM_SERVICE_TELEMETRY_CANCELLED,
                                status);
  iree_status_free(status);
}

static void loom_serve_finite_job_destroy(loom_serve_finite_service_t* service,
                                          loom_serve_finite_job_t* job,
                                          const iree_status_t terminal_status) {
  if (!iree_status_is_ok(terminal_status) ||
      !iree_status_is_ok(service->failure)) {
    job->outcome = LOOM_SERVICE_TELEMETRY_FAILED;
  }
  iree_status_t status = iree_status_clone(
      !iree_status_is_ok(terminal_status)    ? terminal_status
      : !iree_status_is_ok(service->failure) ? service->failure
      : job->outcome == LOOM_SERVICE_TELEMETRY_CANCELLED
          ? iree_status_from_code(IREE_STATUS_CANCELLED)
          : job->status);
  iree_vm_buffer_release(job->prepared.bytes);
  service->protocol->deinitialize(job->request);
  bool delivered = false;
  iree_status_t delivery_status =
      job->consumer.vtable->end(job->consumer.self, service->protocol, status,
                                job->outcome == LOOM_SERVICE_TELEMETRY_REJECTED,
                                &job->response, &delivered);
  if (!iree_status_is_ok(delivery_status)) {
    loom_serve_finite_error(service, job->id, delivery_status);
    job->outcome = LOOM_SERVICE_TELEMETRY_FAILED;
  } else if (!delivered) {
    job->outcome = LOOM_SERVICE_TELEMETRY_CANCELLED;
  } else if (job->outcome == LOOM_SERVICE_TELEMETRY_COMPLETED) {
    loom_service_telemetry_latency_publish_output(&service->telemetry.latency,
                                                  service->telemetry.stream,
                                                  job->id, job->arrival_time);
  }
  loom_serve_finite_request_end(
      service, job->id, job->arrival_time, job->input_tokens, job->outcome,
      iree_status_is_ok(delivery_status) ? status : delivery_status);
  iree_status_free(delivery_status);
  iree_status_free(status);
  iree_status_free(job->status);
  iree_string_builder_deinitialize(&job->response);
  iree_allocator_free(service->host_allocator, job->rows);
  iree_allocator_free(service->host_allocator, job);
  --service->count;
}

static void loom_serve_finite_finish(loom_serve_finite_service_t* service,
                                     const iree_status_t terminal_status) {
  loom_serve_finite_result_release(service->result);
  service->result = NULL;
  loom_serve_finite_snapshot_release(service->snapshot);
  service->snapshot = NULL;
  loom_serve_residency_cancel_admission(
      loom_serve_finite_model_residency(service->model));
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    loom_serve_finite_job_destroy(service, service->cohort[i], terminal_status);
  }
  service->cohort_count = 0;
  service->phase = FINITE_IDLE;
}

// A prepared cohort can exceed the configured capacity without invalidating
// the model or its queues. No device work has been submitted for these jobs.
static void loom_serve_finite_reject_cohort(
    loom_serve_finite_service_t* service, const iree_status_t failure) {
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    loom_serve_finite_job_t* job = service->cohort[i];
    job->outcome = LOOM_SERVICE_TELEMETRY_REJECTED;
    job->status = iree_status_clone(failure);
  }
  loom_serve_finite_model_release_batch(service->model);
  loom_serve_finite_finish(service, iree_ok_status());
}

static iree_status_t loom_serve_finite_cpu_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_finite_service_t* service = process->user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->name.data, service->name.size);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, finite_phases[service->phase]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, loom_runtime_telemetry_stream_identity(service->telemetry.stream)
              .ordinal);
  result->completed = true;
  result->did_work = true;
  if (service->phase == FINITE_BATCHING) {
    const iree_status_t status = loom_serve_finite_model_prepare_batch(
        service->model, service->cohort_count, service->prepared,
        service->rows);
    IREE_RETURN_AND_END_ZONE(z0, status);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->cohort_count && iree_status_is_ok(status); ++i) {
    loom_serve_finite_job_t* job = service->cohort[i];
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, job->id);
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
      if (iree_status_is_ok(status) && job->consumer.vtable->encode) {
        status = iree_allocator_malloc_array(
            service->host_allocator, job->request->count, sizeof(*job->rows),
            (void**)&job->rows);
      }
      job->body = iree_string_view_empty();
    } else {
      for (iree_host_size_t row = 0; row < job->request->count; ++row) {
        job->rows[row] = loom_serve_finite_snapshot_row(service->snapshot,
                                                        job->result_index, row);
      }
      status = job->consumer.vtable->encode(
          job->consumer.self, service->protocol, job->request, service->name,
          job->rows, job->input_tokens, &job->response);
    }
  }
  IREE_TRACE_ZONE_END(z0);
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
                                           loom_serve_finite_phase_t phase) {
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
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->name.data, service->name.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, loom_runtime_telemetry_stream_identity(service->telemetry.stream)
              .ordinal);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, service->telemetry.counters.batches + 1);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loom_serve_finite_model_submit(service->model, &service->result,
                                         &service->device.frontier));
  loom_serve_finite_result_t* result = service->result;
  bool needs_host = false;
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    loom_serve_finite_job_t* job = service->cohort[i];
    needs_host |= job->consumer.vtable->encode &&
                  job->consumer.vtable->is_open(job->consumer.self);
  }
  iree_status_t status = iree_ok_status();
  if (needs_host) {
    status = loom_serve_finite_result_download(result, &service->snapshot,
                                               &service->device.frontier);
  }
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
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_submit_one(service->proactor, &wait->base);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      loom_serve_finite_job_t* job = service->cohort[i];
      if (job->consumer.vtable->publish &&
          job->consumer.vtable->is_open(job->consumer.self)) {
        job->consumer.vtable->publish(job->consumer.self, result, i,
                                      job->request->count);
      }
    }
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  service->phase = FINITE_RUNNING;
  IREE_TRACE(service->device.section = loom_service_trace_section_enter(
                 LOOM_SERVICE_TRACE_SECTION_EXECUTE, service->name));
  loom_serve_finite_batch_event(service, &LOOM_SERVICE_TELEMETRY_BATCH_SUBMIT,
                                service->telemetry.counters.batches + 1);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// The preparation snapshot precedes later arrivals. Filtering it in place and
// prepending survivors preserves age without sorting or per-session VM state.
static void loom_serve_finite_prepared(loom_serve_finite_service_t* service) {
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    loom_serve_finite_job_t* job = service->cohort[i];
    if (!iree_status_is_ok(job->status) ||
        !job->consumer.vtable->is_open(job->consumer.self) ||
        service->stopping) {
      if (!iree_status_is_ok(job->status)) {
        job->outcome = LOOM_SERVICE_TELEMETRY_REJECTED;
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
}

static void loom_serve_finite_select(loom_serve_finite_service_t* service) {
  uint64_t remaining = loom_serve_finite_model_batch_limit(service->model);
  const uint64_t tag = service->pending[0]->prepared.tag;
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    loom_serve_finite_job_t* job = service->pending[i];
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
         loom_serve_finite_model_is_pending(service->model) ||
         loom_serve_execution_reconfiguration_pending(
             service->reconfiguration) ||
         loom_serve_timer_is_pending(&service->timer);
}

static loom_serve_service_snapshot_t loom_serve_finite_snapshot(void* self) {
  const loom_serve_finite_service_t* service = self;
  const iree_time_t now = iree_time_now();
  return (loom_serve_service_snapshot_t){
      .phase = finite_phases[service->phase],
      .queued_requests = service->pending_count,
      .active_requests = service->cohort_count,
      .oldest_queued_age =
          service->pending_count ? now - service->pending[0]->arrival_time : 0,
      .oldest_active_age =
          service->cohort_count ? now - service->cohort[0]->arrival_time : 0};
}

static void loom_serve_finite_heartbeat(loom_serve_finite_service_t* service) {
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(), .event = &LOOM_SERVICE_TELEMETRY_SNAPSHOT};
  loom_service_telemetry_snapshot_record_t telemetry_payload = {
      .counters = service->telemetry.counters,
      .queued = service->pending_count,
      .active = service->cohort_count};
  memcpy(telemetry_payload.phase, finite_phases[service->phase],
         sizeof(telemetry_payload.phase));
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(service->telemetry.stream, &record);
}

static bool loom_serve_finite_request_at(
    void* self, iree_host_size_t index,
    loom_serve_http_request_snapshot_t* out_snapshot) {
  const loom_serve_finite_service_t* service = self;
  const loom_serve_finite_job_t* job = NULL;
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
    loom_serve_finite_job_t* job = service->pending[i];
    if (service->stopping ||
        !job->consumer.vtable->is_open(job->consumer.self)) {
      loom_serve_finite_job_destroy(service, job, iree_ok_status());
    } else {
      service->pending[retained++] = job;
    }
  }
  service->pending_count = retained;
  if (service->stopping) {
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      const loom_serve_finite_consumer_t consumer =
          service->cohort[i]->consumer;
      if (consumer.vtable->cancel && consumer.vtable->is_open(consumer.self)) {
        consumer.vtable->cancel(consumer.self);
      }
    }
  }
}

static void loom_serve_finite_complete_execution(
    loom_serve_finite_service_t* service) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->name.data, service->name.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, loom_runtime_telemetry_stream_identity(service->telemetry.stream)
              .ordinal);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, service->telemetry.counters.batches + 1);
  loom_serve_finite_model_complete(service->model);
  loom_serve_finite_batch_event(service, &LOOM_SERVICE_TELEMETRY_BATCH_END,
                                ++service->telemetry.counters.batches);
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
    loom_serve_finite_job_t* job = service->cohort[i];
    job->input_tokens =
        loom_serve_finite_result_input_tokens(service->result, i);
    service->telemetry.counters.input_tokens += job->input_tokens;
    const bool wanted =
        !service->stopping && job->consumer.vtable->is_open(job->consumer.self);
    if (!wanted || !job->consumer.vtable->encode) {
      if (wanted) {
        job->outcome = LOOM_SERVICE_TELEMETRY_COMPLETED;
      }
      loom_serve_finite_job_destroy(service, job, iree_ok_status());
      continue;
    }
    // The surviving encoding cohort can have holes in the original result
    // plan. Preserve that ordinal without exposing mutable consumer interest
    // to the CPU borrower or moving its per-row work onto the shared owner.
    job->result_index = i;
    service->cohort[retained++] = job;
  }
  loom_serve_finite_result_release(service->result);
  service->result = NULL;
  service->cohort_count = retained;
  if (retained) {
    loom_serve_finite_schedule_cpu(service, FINITE_ENCODING);
  } else {
    loom_serve_finite_finish(service, iree_ok_status());
  }
  IREE_TRACE_ZONE_END(z0);
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
  IREE_TRACE_SECTION_LEAVE(service->device.section);
  IREE_TRACE(service->device.section = 0);
  if (!iree_status_is_ok(status)) {
    // Model-owned payload/pin survives partial enqueue or failed readiness.
    loom_serve_finite_error(service, 0, status);
    loom_serve_finite_finish(service, status);
    return status;
  }
  if (service->phase == FINITE_PREPARING) {
    loom_serve_finite_prepared(service);
    if (!service->stopping && service->pending_count &&
        service->pending[0]->prepared.bytes) {
      loom_serve_finite_select(service);
    }
  } else if (service->phase == FINITE_BATCHING) {
    service->phase = FINITE_SPECIALIZING;
  } else if (service->phase == FINITE_RUNNING) {
    loom_serve_finite_complete_execution(service);
  } else {
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      loom_serve_finite_job_t* job = service->cohort[i];
      if (!service->stopping &&
          job->consumer.vtable->is_open(job->consumer.self)) {
        job->outcome = LOOM_SERVICE_TELEMETRY_COMPLETED;
      }
    }
    loom_serve_finite_finish(service, iree_ok_status());
  }
  return status;
}

static iree_status_t loom_serve_finite_advance(void* self,
                                               loom_serve_service_flags_t flags,
                                               bool* out_progress) {
  loom_serve_finite_service_t* service = self;
  *out_progress = false;
  if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    service->stopping = true;
  }
  loom_serve_finite_cancel_peers(service);
  iree_status_t status = loom_serve_finite_collect(service, out_progress);
  status = iree_status_join(
      status, loom_serve_finite_model_query_status(service->model));
  *out_progress |= loom_serve_execution_reconfiguration_advance(
      service->reconfiguration,
      iree_status_is_ok(status) &&
          (service->phase == FINITE_IDLE || service->phase == FINITE_WAITING),
      service->stopping || !iree_status_is_ok(status));
  if (iree_status_is_ok(status) && service->phase == FINITE_SPECIALIZING) {
    bool has_peer = false;
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      const loom_serve_finite_consumer_t consumer =
          service->cohort[i]->consumer;
      has_peer |= consumer.vtable->is_open(consumer.self);
    }
    const bool cancel = service->stopping || !has_peer;
    bool complete = false;
    status = loom_serve_finite_model_advance_batch(service->model, cancel,
                                                   &complete);
    if (iree_status_is_resource_exhausted(status)) {
      loom_serve_finite_error(service, 0, status);
      loom_serve_finite_reject_cohort(service, status);
      iree_status_free(status);
      status = iree_ok_status();
      *out_progress = true;
    } else if (!iree_status_is_ok(status)) {
      loom_serve_finite_model_release_batch(service->model);
      loom_serve_finite_error(service, 0, status);
      loom_serve_finite_finish(service, status);
    } else if (complete) {
      if (cancel) {
        loom_serve_finite_model_release_batch(service->model);
        loom_serve_finite_finish(service, iree_ok_status());
      } else {
        service->phase = FINITE_WAITING;
      }
      *out_progress = true;
    }
  }
  if (iree_status_is_ok(status) && service->phase == FINITE_WAITING) {
    bool has_peer = false;
    for (iree_host_size_t i = 0; i < service->cohort_count; ++i) {
      const loom_serve_finite_consumer_t consumer =
          service->cohort[i]->consumer;
      has_peer |= consumer.vtable->is_open(consumer.self);
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
                 !iree_any_bit_set(flags,
                                   LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY)) {
        iree_status_t rejection =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "model parameters exceed available capacity");
        loom_serve_finite_reject_cohort(service, rejection);
        iree_status_free(rejection);
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
  const bool stop_timer = service->stopping || !iree_status_is_ok(status);
  status = iree_status_join(
      status, loom_serve_timer_advance(&service->timer, stop_timer, true,
                                       &timer_progress, &tick));
  *out_progress |= timer_progress;
  if (tick) {
    loom_serve_finite_heartbeat(service);
  }
  if (!iree_status_is_ok(status)) {
    service->stopping = true;
    if (iree_status_is_ok(service->failure)) {
      service->failure = iree_status_clone(status);
    }
    loom_serve_finite_cancel_peers(service);
  }
  return status;
}

static iree_status_t loom_serve_finite_job_allocate(
    loom_serve_finite_service_t* service, loom_serve_finite_job_t** out_job) {
  *out_job = NULL;
  loom_serve_finite_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(service->host_allocator,
                            iree_sizeof_struct(loom_serve_finite_job_t) +
                                service->protocol->request_size,
                            (void**)&job));
  job->outcome = LOOM_SERVICE_TELEMETRY_CANCELLED;
  job->request = (loom_serve_finite_request_t*)((uint8_t*)job +
                                                iree_sizeof_struct(
                                                    loom_serve_finite_job_t));
  iree_string_builder_initialize(service->host_allocator, &job->response);
  *out_job = job;
  return iree_ok_status();
}

iree_status_t loom_serve_finite_service_try_open(
    loom_serve_finite_service_t* service, iree_string_builder_t* body,
    loom_serve_finite_call_t** out_call) {
  *out_call = NULL;
  if (service->stopping) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "finite service admission has stopped");
  }
  if (service->count == service->capacity) {
    return iree_ok_status();
  }
  loom_serve_finite_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_finite_job_allocate(service, &job));
  iree_status_t status = loom_serve_finite_call_allocate(
      body, service->host_allocator, out_call, &job->consumer);
  if (iree_status_is_ok(status)) {
    job->arrival_time = iree_time_now();
    job->id = loom_serve_finite_request_begin(service, job->arrival_time);
    job->body = loom_serve_finite_call_body(*out_call);
    service->pending[service->pending_count++] = job;
    ++service->count;
  } else {
    iree_string_builder_deinitialize(&job->response);
    iree_allocator_free(service->host_allocator, job);
  }
  return status;
}

static iree_status_t loom_serve_finite_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_service_flags_t flags) {
  loom_serve_finite_service_t* service = self;
  const iree_time_t arrival_time = iree_time_now();
  const uint64_t id = loom_serve_finite_request_begin(service, arrival_time);
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(
          request->target,
          iree_make_cstring_view(
              loom_serve_model_kind_endpoint(service->protocol->kind)))) {
    loom_serve_finite_decline(service, id, arrival_time, connection,
                              "404 Not Found",
                              IREE_SV("unknown finite-result endpoint"));
    return iree_ok_status();
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("content-type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    loom_serve_finite_decline(service, id, arrival_time, connection,
                              "415 Unsupported Media Type",
                              IREE_SV("Content-Type must be application/json"));
    return iree_ok_status();
  }
  if (service->count == service->capacity) {
    loom_serve_finite_decline(service, id, arrival_time, connection,
                              "503 Service Unavailable",
                              IREE_SV("finite admission queue is full"));
    return iree_ok_status();
  }
  loom_serve_finite_job_t* job = NULL;
  iree_status_t status = loom_serve_finite_job_allocate(service, &job);
  if (!iree_status_is_ok(status)) {
    loom_serve_http_connection_abort(connection);
    loom_serve_finite_request_end(service, id, arrival_time, 0,
                                  LOOM_SERVICE_TELEMETRY_FAILED, status);
    return status;
  }
  ++service->count;
  job->id = id;
  job->arrival_time = arrival_time;
  job->consumer = loom_serve_finite_http_consumer(connection);
  job->body = request->body;
  service->pending[service->pending_count++] = job;
  return iree_ok_status();
}

static void loom_serve_finite_set_observer(
    void* self, loom_serve_wake_callback_t observer) {
  loom_serve_finite_service_t* service = self;
  iree_slim_mutex_lock(&service->event.mutex);
  service->event.observer = observer;
  iree_slim_mutex_unlock(&service->event.mutex);
  loom_serve_timer_set_observer(&service->timer, observer);
  loom_serve_finite_model_set_observer(service->model, observer);
  loom_serve_execution_reconfiguration_set_observer(service->reconfiguration,
                                                    observer);
}

loom_serve_service_t loom_serve_finite_service_interface(
    loom_serve_finite_service_t* service) {
  static const iree_string_view_t input_modalities[] = {IREE_SVL("text")};
  return (loom_serve_service_t){
      .name = service->name,
      .self = service,
      .set_observer = loom_serve_finite_set_observer,
      .is_pending = loom_serve_finite_is_pending,
      .is_active = loom_serve_finite_is_active,
      .accept = loom_serve_finite_accept,
      .advance = loom_serve_finite_advance,
      .tokenizer = loom_serve_finite_model_tokenizer(service->model),
      .kind = service->protocol->kind,
      .endpoint = loom_serve_model_kind_endpoint(service->protocol->kind),
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
                    .counters = &service->telemetry.counters,
                    .latency = &service->telemetry.latency}};
}

void loom_serve_finite_service_free(loom_serve_finite_service_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!loom_serve_finite_is_pending(service));
  loom_serve_finite_model_set_observer(service->model,
                                       (loom_serve_wake_callback_t){0});
  service->stopping = true;
  loom_serve_finite_cancel_peers(service);
  if (service->phase == FINITE_WAITING) {
    loom_serve_finite_model_release_batch(service->model);
  }
  loom_serve_finite_finish(service, iree_ok_status());
  iree_status_free(service->failure);
  loom_serve_execution_reconfiguration_free(service->reconfiguration);
  loom_serve_timer_deinitialize(&service->timer);
  iree_string_builder_deinitialize(&service->response);
  iree_slim_mutex_deinitialize(&service->event.mutex);
  iree_allocator_free(service->host_allocator, service->pending);
  iree_allocator_free(service->host_allocator, service->cohort);
  iree_allocator_free(service->host_allocator, service->prepared);
  iree_allocator_free(service->host_allocator, service->rows);
  if (service->telemetry.owns_stream) {
    loom_runtime_telemetry_stream_free(service->telemetry.stream);
  }
  iree_allocator_free_aligned(service->host_allocator, service);
}

iree_status_t loom_serve_finite_service_allocate(
    loom_serve_host_t* host, loom_serve_finite_model_t* model,
    const loom_serve_finite_protocol_t* protocol, iree_string_view_t name,
    iree_host_size_t capacity, iree_host_size_t variant_capacity,
    iree_duration_t heartbeat_interval, loom_runtime_telemetry_stream_t* stream,
    iree_allocator_t host_allocator,
    loom_serve_finite_service_t** out_service) {
  *out_service = NULL;
  if (!name.size || !capacity || heartbeat_interval < 0 ||
      capacity > SIZE_MAX / sizeof(loom_serve_prepared_request_t)) {
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
    status = loom_runtime_telemetry_stream_allocate(128, host_allocator,
                                                    &service->telemetry.stream);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_timer_configure(&service->timer, service->proactor,
                               heartbeat_interval);
    loom_runtime_telemetry_record_t record = {
        .time_ns = iree_time_now(),
        .event = stream ? &LOOM_SERVICE_TELEMETRY_SNAPSHOT
                        : &LOOM_SERVICE_TELEMETRY_STREAM_BEGIN};
    loom_service_telemetry_snapshot_record_t telemetry_payload = {.phase =
                                                                      "idle"};
    memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
    loom_runtime_telemetry_stream_append(service->telemetry.stream, &record);
    *out_service = service;
  } else {
    loom_serve_finite_service_free(service);
  }
  return status;
}
