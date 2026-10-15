// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/service.h"

#include <string.h>

#include "iree/async/operations/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "loom/serve/runtime/host/timer.h"
#include "loom/serve/runtime/parameter/telemetry.h"
#include "loom/serve/service/speech/output.h"
#include "loom/util/json.h"

typedef struct speech_job_t {
  // Monotonic request identity within this registration.
  uint64_t id;
  // Receipt time retained across queueing, execution and retirement.
  iree_time_t arrival_time;
  // Owned decoded source input retained through CPU preparation.
  loom_serve_speech_request_t request;
  // Completed model units, independently of waveform block count.
  uint64_t completed_parts;
  // HTTP headers have been copied into the carrier for this request.
  bool header_sent;
} speech_job_t;

typedef struct speech_pending_t {
  // Claimed connection, relinquished by response completion or abort.
  loom_serve_http_connection_t* connection;
  // Owned queued request, moved into a resident row without copying strings.
  speech_job_t job;
} speech_pending_t;

typedef enum speech_row_phase_e {
  SPEECH_ROW_EMPTY = 0,
  SPEECH_ROW_QUEUED,
  SPEECH_ROW_WAITING,
  SPEECH_ROW_READY,
  SPEECH_ROW_EXECUTING,
  SPEECH_ROW_DELIVERING,
  SPEECH_ROW_COMPLETE,
} speech_row_phase_t;

static const char* const speech_row_phases[] = {
    "idle",       "queued",     "waiting", "ready",
    "generating", "delivering", "complete"};

typedef struct speech_row_t {
  // Stable resident row, independent of selected batch ordinal.
  iree_host_size_t index;
  // Request lifecycle; shared CPU/device work separately owns borrower cuts.
  speech_row_phase_t phase;
  // Immutable receipt and owned decoded request retained through preparation.
  speech_job_t job;
  // Claim remains owner-only; NULL suppresses publication after cancellation.
  loom_serve_http_connection_t* connection;
  // Explicit source handle, transferred to model teardown on terminal failure.
  void* invocation;
  // One copied block, independent of source feedback and other readers' credit.
  iree_string_builder_t response;
  // CPU-published EOF for the current encoded block.
  bool last_block;
} speech_row_t;

typedef enum speech_phase_e {
  SPEECH_PHASE_IDLE = 0,
  SPEECH_PHASE_PREPARING,
  SPEECH_PHASE_RESERVING,
  SPEECH_PHASE_PREPARING_BATCH,
  SPEECH_PHASE_GENERATING,
  SPEECH_PHASE_ENCODING,
} speech_phase_t;

static const char* const speech_phases[] = {"idle",       "preparing",
                                            "reserving",  "preparing_batch",
                                            "generating", "encoding"};

enum { SPEECH_EVENT_CPU = 1u << 0, SPEECH_EVENT_DEVICE = 1u << 1 };

struct loom_serve_speech_service_t {
  // Allocator owning queued requests and per-row bounded output storage.
  iree_allocator_t host_allocator;
  // Borrowed source model, outliving every accepted operation.
  loom_serve_speech_generator_t generator;
  // Copied policy; deployment name remains borrowed.
  loom_serve_speech_service_options_t options;
  // Shared host executor for finite source CPU and waveform encoding tasks.
  iree_task_executor_t* executor;
  // Shared host proactor for device waits and observation timers.
  iree_async_proactor_t* proactor;
  // One model CPU/device operation. IDLE has no service borrower; terminal
  // device failure leaves actual retirement with the model owner.
  speech_phase_t phase;
  // Stops new work while accepted CPU/device borrowers retire.
  bool stopping;
  // Owned failure accumulated on this turn and transferred to the router.
  iree_status_t status;
  // Stable source-capacity slots; free rows retain only reusable output
  // storage.
  speech_row_t* rows;
  // Occupied host slots, including queued and unadmitted requests.
  iree_host_size_t occupied_count;
  // At most one prepared admission waiter, retaining FIFO position on pressure.
  speech_row_t* admission;
  // Next ready row considered first when selecting compatible numerical work.
  iree_host_size_t cursor;
  // Admission gets at most one turn between batches when work is ready.
  bool admit_next;
  // Completion publication, joined before the service can be destroyed.
  struct {
    // Serializes final callback accesses with collection and observer removal.
    iree_slim_mutex_t mutex;
    // Nonblocking owner wake installed before asynchronous submission.
    loom_serve_wake_callback_t observer;
    // Completed CPU/device operations awaiting owner consumption.
    uint32_t ready;
    // Joined device outcomes awaiting consumption.
    iree_status_t status;
    // CPU outcome kept separate to classify source input rejection.
    iree_status_t cpu_status;
  } events;
  // One finite task, never blocked on device work, IO, or send credit.
  struct {
    // Intrusive task storage, reused only after its release callback.
    iree_task_process_t process;
    // Immutable operation selector while the task is live.
    speech_phase_t phase;
  } cpu;
  // Candidate snapshot becomes the original selected numerical cohort.
  struct {
    // CPU task compacts candidates to selected members before publishing.
    speech_row_t** rows;
    // Selected source arguments, retained through the batch's completion.
    void** invocations;
    // Candidate/selected count; only the CPU task mutates while it is live.
    iree_host_size_t count;
  } batch;
  // One accepted ALL wait, including admission uploads and batch retirement.
  struct {
    // Wait storage retained through its final callback.
    iree_async_semaphore_wait_operation_t operation;
    // Last accepted frontier; an idle cut has successfully consumed it.
    loom_serve_execution_frontier_t frontier;
    // Toll-free semaphore handles for the proactor API.
    iree_async_semaphore_t* semaphores[2];
  } device;
  // Periodic observation and its independently retired cancellation receipt.
  loom_serve_timer_t timer;
  // Bounded host-only FIFO, excluding requests assigned to resident slots.
  speech_pending_t* pending;
  // Occupied queue entries in arrival order.
  iree_host_size_t pending_count;
  // Owner-only counters and bounded immutable observation history.
  struct {
    // Registration-borrowed or standalone-owned record stream.
    loom_serve_telemetry_stream_t* stream;
    // Standalone construction owns stream destruction.
    bool owns_stream;
    // HTTP outcomes and completed numerical cohorts.
    loom_serve_telemetry_counters_t counters;
    // Persistent terminal failure distinct from orderly server shutdown.
    iree_status_code_t failure_code;
  } telemetry;
  // Reusable rejection response, disjoint from CPU-borrowed row outputs.
  iree_string_builder_t response;
};

static void loom_serve_speech_event(loom_serve_speech_service_t* service,
                                    const loom_serve_telemetry_event_t* event,
                                    uint64_t request) {
  const loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(), .request = request, .event = event};
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

static void loom_serve_speech_record_weights(
    loom_serve_speech_service_t* service, iree_time_t time_ns) {
  if (service->generator.weight_activity) {
    const loom_serve_weight_activity_t activity =
        service->generator.weight_activity(service->generator.self);
    loom_serve_telemetry_record_weights(service->telemetry.stream, time_ns,
                                        &activity);
  }
}

// Copies a borrowed status before it propagates or is consumed at a boundary.
static void loom_serve_speech_diagnose(loom_serve_speech_service_t* service,
                                       uint64_t request,
                                       const iree_status_t failure) {
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .request = request,
                                          .event = &LOOM_SERVE_TELEMETRY_ERROR};
  loom_serve_telemetry_diagnostic_capture(failure, &record.data.error);
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

static void loom_serve_speech_request_end(
    loom_serve_speech_service_t* service, const speech_job_t* job,
    loom_serve_telemetry_outcome_t outcome,
    const iree_status_t borrowed_status) {
  loom_serve_telemetry_count_outcome(&service->telemetry.counters, outcome);
  loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .request = job->id,
      .event = &LOOM_SERVE_TELEMETRY_REQUEST_END};
  record.data.end.outcome = outcome;
  record.data.end.duration_ns = record.time_ns - job->arrival_time;
  loom_serve_telemetry_diagnostic_capture(borrowed_status,
                                          &record.data.end.diagnostic);
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
  loom_serve_speech_record_weights(service, record.time_ns);
}

static iree_status_t loom_serve_speech_quote(iree_string_builder_t* output,
                                             iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static iree_status_t loom_serve_speech_response_begin(
    iree_string_builder_t* output, const char* status) {
  iree_string_builder_reset(output);
  return iree_string_builder_append_format(
      output,
      "HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: "
      "close\r\n\r\n",
      status);
}

// Each rejected connection sends exactly one complete JSON response. The
// transport copies admitted bytes, so a slow or reset peer never holds a
// generator buffer.
static loom_serve_telemetry_outcome_t loom_serve_speech_response_send(
    loom_serve_speech_service_t* service,
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
    loom_serve_speech_diagnose(service, request, status);
    iree_status_free(status);
    loom_serve_http_connection_abort(connection);
    return outcome;
  }
}

static iree_status_t loom_serve_speech_rejection_response(
    loom_serve_speech_service_t* service,
    loom_serve_http_connection_t* connection, uint64_t request,
    const char* code, iree_string_view_t message) {
  iree_status_t status =
      loom_serve_speech_response_begin(&service->response, code);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        &service->response,
        "{\"error\":{\"type\":\"speech_request_error\",\"message\":");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_speech_quote(&service->response, message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&service->response, "}}");
  }
  if (iree_status_is_ok(status)) {
    loom_serve_speech_response_send(
        service, connection, request,
        iree_string_builder_view(&service->response));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

// Rejections before admission own only a local receipt, not the active job.
static iree_status_t loom_serve_speech_reject(
    loom_serve_speech_service_t* service,
    loom_serve_http_connection_t* connection, const speech_job_t* job,
    const char* code, iree_string_view_t message) {
  const iree_status_t status = loom_serve_speech_rejection_response(
      service, connection, job->id, code, message);
  loom_serve_speech_request_end(service, job, LOOM_SERVE_TELEMETRY_REJECTED,
                                status);
  return status;
}

static iree_status_t loom_serve_speech_cpu_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_speech_service_t* service = process->user_data;
  result->completed = true;
  result->did_work = true;
  if (service->cpu.phase == SPEECH_PHASE_PREPARING) {
    speech_row_t* row = service->admission;
    return service->generator.prepare(service->generator.self, row->index,
                                      &row->job.request, &row->invocation);
  }
  if (service->cpu.phase == SPEECH_PHASE_PREPARING_BATCH) {
    uint64_t tag = 0;
    uint64_t remaining = service->generator.batch_units;
    iree_host_size_t selected = 0;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < service->batch.count && iree_status_is_ok(status); ++i) {
      speech_row_t* row = service->batch.rows[i];
      loom_serve_speech_work_t work;
      status = service->generator.prepare_work(service->generator.self,
                                               row->invocation, &work);
      if (iree_status_is_ok(status)) {
        if (!selected) {
          tag = work.tag;
        }
        if (work.tag == tag && work.units <= remaining) {
          remaining -= work.units;
          service->batch.rows[selected] = row;
          service->batch.invocations[selected++] = row->invocation;
        }
      }
    }
    service->batch.count = selected;
    if (iree_status_is_ok(status)) {
      status = service->generator.prepare_batch(
          service->generator.self, selected, service->batch.invocations);
    }
    return status;
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->batch.count && iree_status_is_ok(status); ++i) {
    speech_row_t* row = service->batch.rows[i];
    iree_string_builder_reset(&row->response);
    loom_serve_speech_block_t block;
    status = service->generator.collect(service->generator.self, i, &block);
    if (iree_status_is_ok(status) && !row->job.header_sent &&
        (block.frame_count || block.complete)) {
      status = iree_string_builder_append_cstring(
          &row->response,
          "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm;rate=24000;channels=1\r\n"
          "Cache-Control: no-store\r\nConnection: close\r\n\r\n");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_speech_encode_pcm16(block.samples, &row->response);
      row->last_block = block.complete;
    }
  }
  return status;
}

static void loom_serve_speech_cpu_completed(iree_task_process_t* process,
                                            iree_status_t status) {
  loom_serve_speech_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.cpu_status = status;
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_speech_cpu_released(iree_task_process_t* process) {
  loom_serve_speech_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.ready |= SPEECH_EVENT_CPU;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_speech_schedule_cpu(loom_serve_speech_service_t* service,
                                           speech_phase_t phase) {
  service->phase = phase;
  service->cpu.phase = phase;
  iree_task_process_initialize(loom_serve_speech_cpu_run, 0, 1,
                               &service->cpu.process);
  service->cpu.process.user_data = service;
  service->cpu.process.completion_fn = loom_serve_speech_cpu_completed;
  service->cpu.process.release_fn = loom_serve_speech_cpu_released;
  iree_task_executor_schedule_process(service->executor, &service->cpu.process);
}

static void loom_serve_speech_start(loom_serve_speech_service_t* service,
                                    speech_row_t* row) {
  row->phase = SPEECH_ROW_WAITING;
  service->admission = row;
  service->admit_next = false;
  loom_serve_speech_schedule_cpu(service, SPEECH_PHASE_PREPARING);
}

static void loom_serve_speech_device_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  loom_serve_speech_service_t* service = user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.status = iree_status_join(service->events.status, status);
  service->events.ready |= SPEECH_EVENT_DEVICE;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static iree_status_t loom_serve_speech_wait(
    loom_serve_speech_service_t* service, speech_phase_t phase) {
  iree_async_semaphore_wait_operation_t* wait = &service->device.operation;
  memset(wait, 0, sizeof(*wait));
  iree_async_operation_initialize(&wait->base,
                                  IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  loom_serve_speech_device_completed, service);
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
  service->phase = phase;
  return iree_ok_status();
}

static void loom_serve_speech_batch_event(
    loom_serve_speech_service_t* service,
    const loom_serve_telemetry_event_t* event, uint64_t id) {
  const loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .event = event,
      .data.batch = {.id = id, .requests = service->batch.count}};
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
}

// The caller has either released source ownership at a retired cut, or left
// every accepted handle with terminal model teardown.
static void loom_serve_speech_finish(loom_serve_speech_service_t* service,
                                     speech_row_t* row,
                                     loom_serve_telemetry_outcome_t outcome) {
  if (row->connection) {
    if (outcome == LOOM_SERVE_TELEMETRY_COMPLETED) {
      loom_serve_http_connection_finish(row->connection);
    } else {
      loom_serve_http_connection_abort(row->connection);
    }
    row->connection = NULL;
  } else if (outcome == LOOM_SERVE_TELEMETRY_COMPLETED) {
    outcome = service->telemetry.failure_code ? LOOM_SERVE_TELEMETRY_FAILED
                                              : LOOM_SERVE_TELEMETRY_CANCELLED;
  }
  loom_serve_speech_request_end(service, &row->job, outcome, service->status);
  loom_serve_speech_request_deinitialize(&row->job.request);
  memset(&row->job, 0, sizeof(row->job));
  row->invocation = NULL;
  row->phase = SPEECH_ROW_EMPTY;
  row->last_block = false;
  iree_string_builder_reset(&row->response);
  if (service->admission == row) {
    service->admission = NULL;
  }
  --service->occupied_count;
}

// Whole-arena roots can be borrowed by another row's accepted batch. All
// source releases therefore happen at a retired model cut, even for peers
// whose final encoded output was delivered while that batch was running.
static void loom_serve_speech_release_finished(
    loom_serve_speech_service_t* service, bool* progress) {
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    speech_row_t* row = &service->rows[i];
    if (row->phase == SPEECH_ROW_EMPTY ||
        (row->connection && !service->stopping &&
         row->phase != SPEECH_ROW_COMPLETE)) {
      continue;
    }
    const bool failed =
        service->telemetry.failure_code || !iree_status_is_ok(service->status);
    if (!failed) {
      if (row->phase == SPEECH_ROW_WAITING) {
        service->generator.discard(service->generator.self, row->invocation);
      } else if (row->phase != SPEECH_ROW_QUEUED) {
        service->status = service->generator.release(service->generator.self,
                                                     row->invocation);
      }
    }
    loom_serve_speech_finish(service, row,
                             failed || !iree_status_is_ok(service->status)
                                 ? LOOM_SERVE_TELEMETRY_FAILED
                                 : LOOM_SERVE_TELEMETRY_COMPLETED);
    *progress = true;
  }
}

static bool loom_serve_speech_has_resident(
    const loom_serve_speech_service_t* service) {
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    if (service->rows[i].phase >= SPEECH_ROW_READY) {
      return true;
    }
  }
  return false;
}

static void loom_serve_speech_prepared(loom_serve_speech_service_t* service,
                                       loom_serve_http_service_flags_t flags,
                                       bool* progress) {
  speech_row_t* row = service->admission;
  loom_serve_residency_admission_t admission;
  const loom_serve_execution_frontier_t previous = service->device.frontier;
  loom_serve_execution_frontier_t frontier;
  service->status = service->generator.try_admit(
      service->generator.self, row->invocation, &admission, &frontier);
  if (iree_status_is_ok(service->status) &&
      admission == LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED) {
    return;
  }
  if (iree_status_is_ok(service->status) &&
      admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    // An unchanged, successfully consumed frontier needs no new wait. Warm
    // state-capacity denial must not continually wake its own admission retry.
    const bool complete = previous.semaphores[0] == frontier.semaphores[0] &&
                          previous.semaphores[1] == frontier.semaphores[1] &&
                          previous.values[0] == frontier.values[0] &&
                          previous.values[1] == frontier.values[1];
    if (!complete) {
      service->device.frontier = frontier;
      service->status = loom_serve_speech_wait(service, SPEECH_PHASE_RESERVING);
    } else if (service->generator.complete_admit(service->generator.self,
                                                 row->invocation)) {
      row->phase = SPEECH_ROW_READY;
      service->admission = NULL;
    } else {
      admission = LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY;
    }
  }
  if (iree_status_is_ok(service->status) &&
      admission == LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY) {
    if (loom_serve_speech_has_resident(service) ||
        iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY)) {
      return;
    }
    service->generator.discard(service->generator.self, row->invocation);
    service->status = loom_serve_speech_rejection_response(
        service, row->connection, row->job.id, "503 Service Unavailable",
        IREE_SV(
            "model parameters and request state exceed available capacity"));
    row->connection = NULL;
    loom_serve_speech_finish(service, row, LOOM_SERVE_TELEMETRY_REJECTED);
  }
  *progress = true;
}

static void loom_serve_speech_collect(loom_serve_speech_service_t* service,
                                      bool* progress) {
  iree_slim_mutex_lock(&service->events.mutex);
  const uint32_t ready = service->events.ready;
  service->events.ready = 0;
  service->status = iree_status_join(service->status, service->events.status);
  service->events.status = iree_ok_status();
  iree_status_t cpu_status = iree_ok_status();
  if (iree_any_bit_set(ready, SPEECH_EVENT_CPU)) {
    cpu_status = service->events.cpu_status;
    service->events.cpu_status = iree_ok_status();
  }
  iree_slim_mutex_unlock(&service->events.mutex);
  if (!ready) {
    return;
  }
  *progress = true;
  const speech_phase_t phase = service->phase;
  service->phase = SPEECH_PHASE_IDLE;
  if (phase == SPEECH_PHASE_PREPARING &&
      iree_status_is_invalid_argument(cpu_status) &&
      iree_status_is_ok(service->status)) {
    speech_row_t* row = service->admission;
    const bool connected = row->connection != NULL;
    if (connected) {
      iree_string_view_t message = iree_status_message(cpu_status);
      if (!message.size) {
        message = IREE_SV("source rejected speech input");
      }
      service->status = loom_serve_speech_rejection_response(
          service, row->connection, row->job.id, "400 Bad Request", message);
      row->connection = NULL;
    }
    loom_serve_speech_diagnose(service, row->job.id, cpu_status);
    iree_status_free(cpu_status);
    loom_serve_speech_finish(service, row,
                             connected ? LOOM_SERVE_TELEMETRY_REJECTED
                                       : LOOM_SERVE_TELEMETRY_CANCELLED);
    return;
  }
  service->status = iree_status_join(service->status, cpu_status);
  if (!iree_status_is_ok(service->status)) {
    return;
  }
  if (phase == SPEECH_PHASE_RESERVING) {
    speech_row_t* row = service->admission;
    if (service->generator.complete_admit(service->generator.self,
                                          row->invocation)) {
      row->phase = SPEECH_ROW_READY;
      service->admission = NULL;
    }
  } else if (phase == SPEECH_PHASE_GENERATING) {
    ++service->telemetry.counters.batches;
    for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
      ++service->batch.rows[i]->job.completed_parts;
    }
    loom_serve_speech_batch_event(service, &LOOM_SERVE_TELEMETRY_BATCH_END,
                                  service->telemetry.counters.batches);
    loom_serve_speech_schedule_cpu(service, SPEECH_PHASE_ENCODING);
  } else if (phase == SPEECH_PHASE_PREPARING_BATCH) {
    bool cancelled = service->stopping;
    for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
      cancelled |= service->batch.rows[i]->connection == NULL;
    }
    if (cancelled) {
      service->generator.release_batch(service->generator.self);
      service->batch.count = 0;
    } else {
      for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
        service->batch.rows[i]->phase = SPEECH_ROW_EXECUTING;
      }
      service->cursor = (service->batch.rows[0]->index + 1) %
                        service->generator.request_capacity;
      service->status = service->generator.submit(service->generator.self,
                                                  &service->device.frontier);
      if (iree_status_is_ok(service->status)) {
        loom_serve_speech_batch_event(service,
                                      &LOOM_SERVE_TELEMETRY_BATCH_SUBMIT,
                                      service->telemetry.counters.batches + 1);
        service->status =
            loom_serve_speech_wait(service, SPEECH_PHASE_GENERATING);
      }
    }
  } else if (phase == SPEECH_PHASE_ENCODING) {
    service->generator.release_batch(service->generator.self);
    for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
      service->batch.rows[i]->phase = SPEECH_ROW_DELIVERING;
    }
    service->batch.count = 0;
    service->admit_next = true;
  }
}

// Only rows with copied output are touched; a CPU task never borrows these
// builders. Accepted sends copy the bytes before a row becomes eligible again.
static void loom_serve_speech_deliver(loom_serve_speech_service_t* service,
                                      bool* progress) {
  for (iree_host_size_t i = 0; i < service->generator.request_capacity &&
                               iree_status_is_ok(service->status);
       ++i) {
    speech_row_t* row = &service->rows[i];
    if (row->phase != SPEECH_ROW_DELIVERING || !row->connection ||
        service->stopping) {
      continue;
    }
    const iree_string_view_t bytes = iree_string_builder_view(&row->response);
    if (bytes.size) {
      if (!loom_serve_http_connection_can_send(row->connection)) {
        continue;
      }
      iree_status_t status =
          loom_serve_http_connection_send(row->connection, bytes);
      if (iree_status_is_cancelled(status) ||
          iree_status_is_unavailable(status)) {
        loom_serve_speech_diagnose(service, row->job.id, status);
        iree_status_free(status);
        loom_serve_http_connection_abort(row->connection);
        row->connection = NULL;
      } else if (!iree_status_is_ok(status)) {
        service->status = status;
      } else {
        row->job.header_sent = true;
      }
    }
    row->phase = row->last_block ? SPEECH_ROW_COMPLETE : SPEECH_ROW_READY;
    iree_string_builder_reset(&row->response);
    *progress = true;
  }
}

static void loom_serve_speech_prune(loom_serve_speech_service_t* service,
                                    bool* progress) {
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    speech_row_t* row = &service->rows[i];
    if (row->connection && loom_serve_http_connection_failed(row->connection)) {
      loom_serve_http_connection_abort(row->connection);
      row->connection = NULL;
      *progress = true;
    }
  }
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    speech_pending_t pending = service->pending[i];
    if (loom_serve_http_connection_failed(pending.connection)) {
      loom_serve_speech_request_end(service, &pending.job,
                                    LOOM_SERVE_TELEMETRY_CANCELLED,
                                    iree_ok_status());
      loom_serve_http_connection_abort(pending.connection);
      loom_serve_speech_request_deinitialize(&pending.job.request);
      *progress = true;
    } else {
      service->pending[retained++] = pending;
    }
  }
  service->pending_count = retained;
}

static speech_row_t* loom_serve_speech_free_row(
    loom_serve_speech_service_t* service) {
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    if (service->rows[i].phase == SPEECH_ROW_EMPTY) {
      return &service->rows[i];
    }
  }
  return NULL;
}

static void loom_serve_speech_assign(loom_serve_speech_service_t* service,
                                     speech_row_t* row,
                                     speech_pending_t pending) {
  row->connection = pending.connection;
  row->job = pending.job;
  row->phase = SPEECH_ROW_QUEUED;
  ++service->occupied_count;
}

// Moving a host-only queued receipt borrows no model state. Existing CPU tasks
// hold stable rows, never pointers into this compact FIFO.
static void loom_serve_speech_fill_rows(loom_serve_speech_service_t* service) {
  speech_row_t* row = NULL;
  while (service->pending_count &&
         (row = loom_serve_speech_free_row(service)) != NULL) {
    loom_serve_speech_assign(service, row, service->pending[0]);
    --service->pending_count;
    memmove(service->pending, service->pending + 1,
            service->pending_count * sizeof(*service->pending));
  }
}

static iree_status_t loom_serve_speech_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_http_service_flags_t flags) {
  loom_serve_speech_service_t* service = self;
  speech_pending_t pending = {
      .connection = connection,
      .job = {.id = ++service->telemetry.counters.requests,
              .arrival_time = iree_time_now()}};
  loom_serve_speech_event(service, &LOOM_SERVE_TELEMETRY_REQUEST_BEGIN,
                          pending.job.id);
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      !iree_string_view_equal(request->target, IREE_SV("/v1/audio/speech"))) {
    return loom_serve_speech_reject(service, connection, &pending.job,
                                    "404 Not Found",
                                    IREE_SV("unknown speech endpoint"));
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("content-type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return loom_serve_speech_reject(
        service, connection, &pending.job, "415 Unsupported Media Type",
        IREE_SV("Content-Type must be application/json"));
  }
  iree_status_t status = loom_serve_speech_request_initialize(
      request->body, service->options.name, service->host_allocator,
      &pending.job.request);
  if (!iree_status_is_ok(status)) {
    iree_string_view_t message = iree_status_message(status);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(status)));
    }
    iree_status_t response_status = loom_serve_speech_reject(
        service, connection, &pending.job,
        iree_status_is_resource_exhausted(status) ? "503 Service Unavailable"
                                                  : "400 Bad Request",
        message);
    loom_serve_speech_diagnose(service, pending.job.id, status);
    iree_status_free(status);
    return response_status;
  }
  loom_serve_speech_fill_rows(service);
  speech_row_t* row = loom_serve_speech_free_row(service);
  if (row) {
    loom_serve_speech_assign(service, row, pending);
  } else if (service->pending_count < service->options.pending_capacity) {
    service->pending[service->pending_count++] = pending;
  } else {
    loom_serve_speech_request_deinitialize(&pending.job.request);
    return loom_serve_speech_reject(service, connection, &pending.job,
                                    "503 Service Unavailable",
                                    IREE_SV("speech admission queue is full"));
  }
  return iree_ok_status();
}

static const char* loom_serve_speech_row_phase(const speech_row_t* row) {
  if (row->phase == SPEECH_ROW_READY && row->connection &&
      !loom_serve_http_connection_can_send(row->connection)) {
    return "delivering";
  }
  return speech_row_phases[row->phase];
}

static loom_serve_http_service_snapshot_t loom_serve_speech_snapshot(
    void* self) {
  const loom_serve_speech_service_t* service = self;
  const iree_time_t now = iree_time_now();
  loom_serve_http_service_snapshot_t snapshot = {
      .phase = speech_phases[service->phase],
      .queued_requests = service->pending_count,
      .oldest_queued_age = service->pending_count
                               ? now - service->pending[0].job.arrival_time
                               : 0};
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    const speech_row_t* row = &service->rows[i];
    if (row->phase == SPEECH_ROW_EMPTY) {
      continue;
    }
    const iree_duration_t age = now - row->job.arrival_time;
    if (row->phase == SPEECH_ROW_QUEUED) {
      ++snapshot.queued_requests;
      snapshot.oldest_queued_age = iree_max(snapshot.oldest_queued_age, age);
    } else {
      ++snapshot.active_requests;
      snapshot.oldest_active_age = iree_max(snapshot.oldest_active_age, age);
    }
    if (service->phase == SPEECH_PHASE_IDLE &&
        (row->phase == SPEECH_ROW_READY || !strcmp(snapshot.phase, "idle"))) {
      snapshot.phase = loom_serve_speech_row_phase(row);
    }
  }
  return snapshot;
}

static bool loom_serve_speech_request_at(
    void* self, iree_host_size_t index,
    loom_serve_http_request_snapshot_t* out_snapshot) {
  const loom_serve_speech_service_t* service = self;
  const speech_job_t* job = NULL;
  const char* phase = "queued";
  if (index < service->pending_count) {
    job = &service->pending[index].job;
  } else {
    index -= service->pending_count;
    for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
      const speech_row_t* row = &service->rows[i];
      if (row->phase == SPEECH_ROW_EMPTY) {
        continue;
      }
      if (index) {
        --index;
        continue;
      }
      job = &row->job;
      phase =
          service->admission == row && service->phase == SPEECH_PHASE_PREPARING
              ? "preparing"
              : loom_serve_speech_row_phase(row);
      break;
    }
  }
  if (!job) {
    return false;
  }
  *out_snapshot = (loom_serve_http_request_snapshot_t){
      .id = job->id,
      .phase = phase,
      .arrival_time = job->arrival_time,
      .counters = {.requests = 1, .batches = job->completed_parts}};
  return true;
}

static void loom_serve_speech_heartbeat(loom_serve_speech_service_t* service) {
  const loom_serve_http_service_snapshot_t state =
      loom_serve_speech_snapshot(service);
  loom_serve_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .event = &LOOM_SERVE_TELEMETRY_SNAPSHOT,
      .data.snapshot = {.counters = service->telemetry.counters,
                        .queued = state.queued_requests,
                        .active = state.active_requests}};
  memcpy(record.data.snapshot.phase, state.phase, strlen(state.phase));
  loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
  loom_serve_speech_record_weights(service, record.time_ns);
}

static void loom_serve_speech_stop(loom_serve_speech_service_t* service) {
  service->stopping = true;
  const loom_serve_telemetry_outcome_t outcome =
      service->telemetry.failure_code ? LOOM_SERVE_TELEMETRY_FAILED
                                      : LOOM_SERVE_TELEMETRY_CANCELLED;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    speech_pending_t* pending = &service->pending[i];
    loom_serve_speech_request_end(service, &pending->job, outcome,
                                  service->status);
    loom_serve_http_connection_abort(pending->connection);
    loom_serve_speech_request_deinitialize(&pending->job.request);
  }
  service->pending_count = 0;
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    speech_row_t* row = &service->rows[i];
    if (row->connection) {
      loom_serve_http_connection_abort(row->connection);
      row->connection = NULL;
    }
  }
}

static void loom_serve_speech_set_observer(
    void* self, loom_serve_wake_callback_t observer) {
  loom_serve_speech_service_t* service = self;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.observer = observer;
  iree_slim_mutex_unlock(&service->events.mutex);
  loom_serve_timer_set_observer(&service->timer, observer);
}

static bool loom_serve_speech_is_active(void* self) {
  const loom_serve_speech_service_t* service = self;
  return service->phase != SPEECH_PHASE_IDLE ||
         loom_serve_speech_has_resident(service);
}

static bool loom_serve_speech_is_pending(void* self) {
  loom_serve_speech_service_t* service = self;
  return loom_serve_speech_is_active(service) ||
         loom_serve_timer_is_pending(&service->timer);
}

// Offers are queried from a fresh snapshot on the CPU. A blocked output row
// never enters this snapshot; neighbors continue using their own send credit.
static void loom_serve_speech_schedule(loom_serve_speech_service_t* service,
                                       loom_serve_http_service_flags_t flags,
                                       bool* progress) {
  if (service->admission) {
    loom_serve_speech_prepared(service, flags, progress);
    if (service->phase != SPEECH_PHASE_IDLE ||
        !iree_status_is_ok(service->status)) {
      return;
    }
  }
  speech_row_t* queued = NULL;
  service->batch.count = 0;
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    const iree_host_size_t index =
        (service->cursor + i) % service->generator.request_capacity;
    speech_row_t* row = &service->rows[index];
    if (row->phase == SPEECH_ROW_QUEUED &&
        (!queued || row->job.id < queued->job.id)) {
      queued = row;
    }
    if (row->phase == SPEECH_ROW_READY && row->connection &&
        loom_serve_http_connection_can_send(row->connection)) {
      service->batch.rows[service->batch.count++] = row;
    }
  }
  if (!service->admission && queued &&
      (service->admit_next || !service->batch.count)) {
    service->batch.count = 0;
    loom_serve_speech_start(service, queued);
    *progress = true;
  } else if (service->batch.count) {
    loom_serve_speech_schedule_cpu(service, SPEECH_PHASE_PREPARING_BATCH);
    *progress = true;
  }
}

static iree_status_t loom_serve_speech_advance(
    void* self, loom_serve_http_service_flags_t flags, bool* out_progress) {
  loom_serve_speech_service_t* service = self;
  *out_progress = false;
  if (!service->stopping &&
      iree_any_bit_set(flags, LOOM_SERVE_HTTP_SERVICE_FLAG_STOPPING)) {
    loom_serve_speech_stop(service);
    *out_progress = true;
  }
  loom_serve_speech_prune(service, out_progress);
  loom_serve_speech_collect(service, out_progress);
  if (!service->stopping && !iree_status_is_ok(service->status)) {
    service->telemetry.failure_code = iree_status_code(service->status);
    loom_serve_speech_stop(service);
    *out_progress = true;
  }
  bool delivered = false;
  if (iree_status_is_ok(service->status)) {
    loom_serve_speech_deliver(service, &delivered);
    *out_progress |= delivered;
  }
  if (service->phase == SPEECH_PHASE_IDLE) {
    loom_serve_speech_release_finished(service, out_progress);
    if (!service->stopping && iree_status_is_ok(service->status)) {
      loom_serve_speech_fill_rows(service);
      // Delivery publication ends this bounded owner turn. The router observes
      // progress and rotates before the next admission/numerical selection.
      if (!delivered) {
        loom_serve_speech_schedule(service, flags, out_progress);
      }
    }
  }
  bool timer_progress = false;
  bool tick = false;
  service->status = iree_status_join(
      service->status,
      loom_serve_timer_advance(&service->timer,
                               service->stopping && !service->occupied_count,
                               &timer_progress, &tick));
  *out_progress |= timer_progress;
  if (tick) {
    loom_serve_speech_heartbeat(service);
  }
  if (!service->stopping && !iree_status_is_ok(service->status)) {
    service->telemetry.failure_code = iree_status_code(service->status);
    loom_serve_speech_stop(service);
    *out_progress = true;
  }
  if (service->stopping && service->phase == SPEECH_PHASE_IDLE) {
    loom_serve_speech_release_finished(service, out_progress);
  }
  iree_status_t status = service->status;
  service->status = iree_ok_status();
  if (!iree_status_is_ok(status)) {
    service->telemetry.failure_code = iree_status_code(status);
    loom_serve_speech_diagnose(service, 0, status);
  }
  return status;
}

loom_serve_http_service_t loom_serve_speech_service_interface(
    loom_serve_speech_service_t* service) {
  static const iree_string_view_t input_modalities[] = {IREE_SVL("text")};
  static const iree_string_view_t output_modalities[] = {IREE_SVL("audio")};
  return (loom_serve_http_service_t){
      .name = service->options.name,
      .self = service,
      .set_observer = loom_serve_speech_set_observer,
      .is_pending = loom_serve_speech_is_pending,
      .is_active = loom_serve_speech_is_active,
      .accept = loom_serve_speech_accept,
      .advance = loom_serve_speech_advance,
      .tokenizer = service->generator.tokenizer,
      .kind = "speech",
      .endpoint = "/v1/audio/speech",
      .capabilities = {.input_modalities = {IREE_ARRAYSIZE(input_modalities),
                                            input_modalities},
                       .output_modalities = {IREE_ARRAYSIZE(output_modalities),
                                             output_modalities}},
      .snapshot = loom_serve_speech_snapshot,
      .inspection = {.request_at = loom_serve_speech_request_at},
      .telemetry = {service->telemetry.stream, &service->telemetry.counters},
      .capacity_domain = service->generator.capacity_domain};
}

void loom_serve_speech_service_free(loom_serve_speech_service_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!loom_serve_speech_is_pending(service));
  const iree_allocator_t host_allocator = service->host_allocator;
  if (!service->stopping) {
    loom_serve_speech_stop(service);
  }
  bool progress = false;
  loom_serve_speech_release_finished(service, &progress);
  loom_serve_timer_deinitialize(&service->timer);
  if (service->telemetry.stream) {
    loom_serve_speech_heartbeat(service);
  }
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    iree_string_builder_deinitialize(&service->rows[i].response);
  }
  iree_string_builder_deinitialize(&service->response);
  iree_allocator_free(host_allocator, service->rows);
  iree_allocator_free(host_allocator, service->batch.rows);
  iree_allocator_free(host_allocator, service->batch.invocations);
  iree_allocator_free(host_allocator, service->pending);
  iree_slim_mutex_deinitialize(&service->events.mutex);
  if (service->telemetry.owns_stream) {
    loom_serve_telemetry_stream_free(service->telemetry.stream);
  }
  iree_allocator_free_aligned(host_allocator, service);
}

iree_status_t loom_serve_speech_service_allocate(
    loom_serve_host_t* host, loom_serve_speech_generator_t generator,
    const loom_serve_speech_service_options_t* options,
    loom_serve_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_speech_service_t** out_service) {
  *out_service = NULL;
  if (!generator.prepare || !generator.discard || !generator.try_admit ||
      !generator.complete_admit || !generator.prepare_work ||
      !generator.prepare_batch || !generator.submit || !generator.collect ||
      !generator.release_batch || !generator.release ||
      !generator.block_capacity || !generator.request_capacity ||
      !generator.batch_units ||
      generator.request_capacity > SIZE_MAX / sizeof(speech_row_t) ||
      generator.request_capacity > SIZE_MAX / sizeof(void*) ||
      generator.block_capacity > (SIZE_MAX - 128) / sizeof(int16_t) ||
      !options->name.size || !options->pending_capacity ||
      options->pending_capacity > SIZE_MAX / sizeof(speech_pending_t) ||
      options->heartbeat_interval < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech service requires a complete generator, "
                            "bounded blocks, name and queue capacity");
  }
  if (generator.format.sample_rate != 24000 || generator.format.channels != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PCM speech responses require 24000 Hz mono output");
  }
  loom_serve_speech_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*service),
      iree_alignof(loom_serve_speech_service_t), 0, (void**)&service));
  service->host_allocator = host_allocator;
  service->generator = generator;
  service->options = *options;
  service->telemetry.stream = stream;
  service->executor = loom_serve_host_executor(host);
  service->admit_next = true;
  iree_slim_mutex_initialize(&service->events.mutex);
  loom_serve_timer_initialize(&service->timer);
  iree_string_builder_initialize(host_allocator, &service->response);
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, generator.request_capacity, sizeof(*service->rows),
      (void**)&service->rows);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < generator.request_capacity; ++i) {
      service->rows[i].index = i;
      iree_string_builder_initialize(host_allocator,
                                     &service->rows[i].response);
    }
    status = iree_allocator_malloc_array(
        host_allocator, generator.request_capacity,
        sizeof(*service->batch.rows), (void**)&service->batch.rows);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, generator.request_capacity,
                                    sizeof(*service->batch.invocations),
                                    (void**)&service->batch.invocations);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, options->pending_capacity, sizeof(*service->pending),
        (void**)&service->pending);
  }
  for (iree_host_size_t i = 0;
       i < generator.request_capacity && iree_status_is_ok(status); ++i) {
    status = iree_string_builder_reserve(
        &service->rows[i].response,
        generator.block_capacity * sizeof(int16_t) + 128);
  }
  if (iree_status_is_ok(status) && !stream) {
    status = loom_serve_telemetry_stream_allocate(128, host_allocator,
                                                  &service->telemetry.stream);
    service->telemetry.owns_stream = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    if (stream) {
      loom_serve_speech_heartbeat(service);
    } else {
      loom_serve_speech_event(service, &LOOM_SERVE_TELEMETRY_STREAM_BEGIN, 0);
    }
    status = iree_async_proactor_pool_get(loom_serve_host_proactor_pool(host),
                                          0, &service->proactor);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_timer_configure(&service->timer, service->proactor,
                               options->heartbeat_interval);
    *out_service = service;
  } else {
    loom_serve_speech_service_free(service);
  }
  return status;
}
