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
#include "loom/serve/service/kind.h"
#include "loom/serve/service/speech/http.h"
#include "loom/serve/service/speech/stream.h"
#include "loom/serve/service/speech/voice_http.h"
#include "loom/serve/telemetry/latency.h"
#include "loom/serve/telemetry/tracing.h"

typedef enum loom_serve_speech_voice_phase_e {
  SPEECH_VOICE_EMPTY = 0,
  SPEECH_VOICE_PREPARING,
  SPEECH_VOICE_READY,
  SPEECH_VOICE_REMOVED,
} loom_serve_speech_voice_phase_t;

typedef struct loom_serve_speech_voice_slot_t {
  // Owned identifier; a removed incarnation no longer participates in lookup.
  iree_string_view_t id;
  // Publication and retirement state, mutated only by the service owner.
  loom_serve_speech_voice_phase_t phase;
  // Accepted uploads/synthesis requests retaining this exact slot incarnation.
  iree_host_size_t references;
  // Immutable model-owned facts, published by CPU preparation after task join.
  void* prepared;
} loom_serve_speech_voice_slot_t;

typedef struct loom_serve_speech_job_t {
  // Monotonic request identity within this registration.
  uint64_t id;
  // Receipt time retained across queueing, execution and retirement.
  iree_time_t arrival_time;
  // An upload creates the captured slot; synthesis reads an already-ready slot.
  bool uploading;
  // Captured slot incarnation, retained through queued and accepted work.
  loom_serve_speech_voice_slot_t* voice;
  union {
    // Owned synthesis input retained through CPU preparation.
    loom_serve_speech_request_t request;
    // Owned original recording retained through upload preparation.
    loom_serve_speech_voice_request_t upload;
  };

  // Completed model units, independently of waveform block count.
  uint64_t completed_parts;
  // At least one nonempty waveform block has reached the consumer.
  bool output_published;
} loom_serve_speech_job_t;

typedef struct loom_serve_speech_pending_t {
  // Output owner retained until rejection, cancellation, or source retirement.
  loom_serve_speech_consumer_t consumer;
  // Owned queued request, moved into a resident row without copying strings.
  loom_serve_speech_job_t job;
} loom_serve_speech_pending_t;

typedef enum loom_serve_speech_row_phase_e {
  SPEECH_ROW_EMPTY = 0,
  SPEECH_ROW_QUEUED,
  SPEECH_ROW_WAITING,
  SPEECH_ROW_READY,
  SPEECH_ROW_EXECUTING,
  SPEECH_ROW_DELIVERING,
  SPEECH_ROW_COMPLETE,
} loom_serve_speech_row_phase_t;

static const char* const speech_row_phases[] = {
    "idle",       "queued",     "waiting", "ready",
    "generating", "delivering", "complete"};

typedef struct loom_serve_speech_row_t {
  // Stable resident row, independent of selected batch ordinal.
  iree_host_size_t index;
  // Request lifecycle; shared CPU/device work separately owns borrower cuts.
  loom_serve_speech_row_phase_t phase;
  // Immutable receipt and owned decoded request retained through preparation.
  loom_serve_speech_job_t job;
  // Independent output owner, retained through all accepted CPU borrowers.
  loom_serve_speech_consumer_t consumer;
  // Owner has withdrawn publication; accepted work still retains this row.
  bool cancelled;
  // Explicit source handle, transferred to model teardown on terminal failure.
  void* invocation;
  // CPU-published EOF for the current encoded block.
  bool last_block;
  // The encoded block contains waveform frames, not only source progress.
  bool has_audio;
  // Request-local source/output error retained through asynchronous release.
  // A CPU writer is joined before the owner reads this status.
  iree_status_t status;
  // Error classification established where the request fails.
  loom_serve_telemetry_outcome_t outcome;
} loom_serve_speech_row_t;

typedef enum loom_serve_speech_phase_e {
  SPEECH_PHASE_IDLE = 0,
  SPEECH_PHASE_PREPARING,
  SPEECH_PHASE_RESERVING,
  SPEECH_PHASE_PREPARING_BATCH,
  SPEECH_PHASE_GENERATING,
  SPEECH_PHASE_ENCODING,
  SPEECH_PHASE_RELEASING,
} loom_serve_speech_phase_t;

static const char* const speech_phases[] = {
    "idle",       "preparing", "reserving", "preparing_batch",
    "generating", "encoding",  "releasing"};

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
  loom_serve_speech_phase_t phase;
  // Stops new work while accepted CPU/device borrowers retire.
  bool stopping;
  // Owned failure accumulated on this turn and transferred to the router.
  iree_status_t status;
  // Terminal failure retained for request results on subsequent draining turns.
  iree_status_t failure;
  // Bounded voice identities; removed slots survive while requests retain them.
  loom_serve_speech_voice_slot_t* voices;
  // Stable source-capacity slots, independent of output consumer ownership.
  loom_serve_speech_row_t* rows;
  // Occupied host slots, including queued and unadmitted requests.
  iree_host_size_t occupied_count;
  // At most one prepared admission waiter, retaining FIFO position on pressure.
  loom_serve_speech_row_t* admission;
  // Request retained through asynchronous row-backing release readiness.
  loom_serve_speech_row_t* retiring;
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
    loom_serve_speech_phase_t phase;
  } cpu;
  // Candidate snapshot becomes the original selected numerical cohort.
  struct {
    // CPU task compacts candidates to selected members before publishing.
    loom_serve_speech_row_t** rows;
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
    // Accepted numerical batch through readiness; zero for admission/release.
    IREE_TRACE(iree_section_id_t section;)
  } device;
  // Periodic observation and its independently retired cancellation receipt.
  loom_serve_timer_t timer;
  // Bounded host-only FIFO, excluding requests assigned to resident slots.
  loom_serve_speech_pending_t* pending;
  // Occupied queue entries in arrival order.
  iree_host_size_t pending_count;
  // Owner-only counters and bounded immutable observation history.
  struct {
    // Registration-borrowed or standalone-owned record stream.
    loom_serve_telemetry_stream_t* stream;
    // Standalone construction owns stream destruction.
    bool owns_stream;
    // Request outcomes and completed numerical cohorts.
    loom_serve_telemetry_counters_t counters;
    // Request latency distributions retained independently of log readers.
    loom_serve_telemetry_latency_t latency;
  } telemetry;
};

static loom_serve_speech_voice_slot_t* loom_serve_speech_voice_find(
    loom_serve_speech_service_t* service, iree_string_view_t id) {
  for (iree_host_size_t i = 0; i < service->generator.voice_capacity; ++i) {
    loom_serve_speech_voice_slot_t* slot = &service->voices[i];
    if ((slot->phase == SPEECH_VOICE_PREPARING ||
         slot->phase == SPEECH_VOICE_READY) &&
        iree_string_view_equal(slot->id, id)) {
      return slot;
    }
  }
  return NULL;
}

static void loom_serve_speech_voice_collect(
    loom_serve_speech_service_t* service,
    loom_serve_speech_voice_slot_t* slot) {
  if (slot->phase == SPEECH_VOICE_REMOVED && !slot->references) {
    if (slot->prepared) {
      service->generator.release_voice(service->generator.self, slot->prepared);
    }
    iree_allocator_free(service->host_allocator, (void*)slot->id.data);
    *slot = (loom_serve_speech_voice_slot_t){0};
  }
}

static iree_status_t loom_serve_speech_voice_capture(
    loom_serve_speech_service_t* service, iree_string_view_t id,
    loom_serve_speech_voice_slot_t** out_voice) {
  *out_voice = NULL;
  if (!service->generator.voice_capacity) {
    return iree_ok_status();
  }
  loom_serve_speech_voice_slot_t* voice =
      loom_serve_speech_voice_find(service, id);
  if (!voice || voice->phase != SPEECH_VOICE_READY) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "voice '%.*s' is not registered; upload a recording first",
        (int)id.size, id.data);
  }
  ++voice->references;
  *out_voice = voice;
  return iree_ok_status();
}

static iree_status_t loom_serve_speech_voice_reserve(
    loom_serve_speech_service_t* service, iree_string_view_t id,
    loom_serve_speech_voice_slot_t** out_voice) {
  *out_voice = NULL;
  if (!service->generator.voice_capacity) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "this model does not support voice uploads");
  }
  if (loom_serve_speech_voice_find(service, id)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "voice '%.*s' is already registered or preparing",
                            (int)id.size, id.data);
  }
  for (iree_host_size_t i = 0; i < service->generator.voice_capacity; ++i) {
    loom_serve_speech_voice_slot_t* slot = &service->voices[i];
    if (slot->phase != SPEECH_VOICE_EMPTY) {
      continue;
    }
    char* name = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_clone(
        service->host_allocator, iree_make_const_byte_span(id.data, id.size),
        (void**)&name));
    *slot = (loom_serve_speech_voice_slot_t){
        .id = iree_make_string_view(name, id.size),
        .phase = SPEECH_VOICE_PREPARING,
        .references = 1};
    *out_voice = slot;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      "all voice slots are occupied; remove an unused voice");
}

static void loom_serve_speech_job_deinitialize(
    loom_serve_speech_service_t* service, loom_serve_speech_job_t* job) {
  if (job->voice) {
    if (job->uploading && job->voice->phase == SPEECH_VOICE_PREPARING) {
      job->voice->phase = SPEECH_VOICE_REMOVED;
    }
    --job->voice->references;
    loom_serve_speech_voice_collect(service, job->voice);
  }
  if (job->uploading) {
    loom_serve_speech_voice_request_deinitialize(&job->upload);
  } else {
    loom_serve_speech_request_deinitialize(&job->request);
  }
  job->voice = NULL;
}

iree_status_t loom_serve_speech_service_remove_voice(
    loom_serve_speech_service_t* service, iree_string_view_t id) {
  loom_serve_speech_voice_slot_t* voice =
      loom_serve_speech_voice_find(service, id);
  if (!voice || voice->phase != SPEECH_VOICE_READY) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "voice is not registered");
  }
  voice->phase = SPEECH_VOICE_REMOVED;
  loom_serve_speech_voice_collect(service, voice);
  return iree_ok_status();
}

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
    loom_serve_speech_service_t* service, const loom_serve_speech_job_t* job,
    loom_serve_telemetry_outcome_t outcome,
    const iree_status_t borrowed_status) {
  loom_serve_telemetry_count_outcome(&service->telemetry.counters, outcome);
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
  loom_serve_speech_record_weights(service, record.time_ns);
}

// Rejections before admission own only a local receipt, not the active job.
static iree_status_t loom_serve_speech_reject(
    loom_serve_speech_service_t* service,
    loom_serve_http_connection_t* connection,
    const loom_serve_speech_job_t* job, const char* code,
    iree_string_view_t message) {
  const iree_status_t status = loom_serve_speech_http_reject(
      connection, code, message, service->host_allocator);
  loom_serve_speech_request_end(service, job, LOOM_SERVE_TELEMETRY_REJECTED,
                                status);
  return status;
}

static iree_status_t loom_serve_speech_cpu_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_speech_service_t* service = process->user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->options.name.data,
                              service->options.name.size);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, speech_phases[service->cpu.phase]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0,
      loom_serve_telemetry_stream_identity(service->telemetry.stream).ordinal);
  result->completed = true;
  result->did_work = true;
  if (service->cpu.phase == SPEECH_PHASE_PREPARING) {
    loom_serve_speech_row_t* row = service->admission;
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, row->job.id);
    const iree_status_t status =
        row->job.uploading
            ? service->generator.prepare_voice(
                  service->generator.self, row->index,
                  row->job.voice - service->voices, &row->job.upload.audio,
                  &row->invocation, &row->job.voice->prepared)
            : service->generator.prepare(
                  service->generator.self, row->index, &row->job.request,
                  row->job.voice ? row->job.voice->prepared : NULL,
                  &row->invocation);
    IREE_RETURN_AND_END_ZONE(z0, status);
  }
  if (service->cpu.phase == SPEECH_PHASE_PREPARING_BATCH) {
    uint64_t tag = 0;
    uint64_t remaining = service->generator.batch_units;
    iree_host_size_t selected = 0;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < service->batch.count && iree_status_is_ok(status); ++i) {
      loom_serve_speech_row_t* row = service->batch.rows[i];
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
    IREE_RETURN_AND_END_ZONE(z0, status);
  }
  for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
    loom_serve_speech_row_t* row = service->batch.rows[i];
    loom_serve_speech_block_t block;
    row->status =
        service->generator.collect(service->generator.self, i, &block);
    if (iree_status_is_ok(row->status)) {
      row->status = row->consumer.encode(row->consumer.self, &block);
      row->last_block = block.complete;
      row->has_audio = block.frame_count != 0;
    }
    row->outcome = LOOM_SERVE_TELEMETRY_FAILED;
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
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
                                           loom_serve_speech_phase_t phase) {
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
                                    loom_serve_speech_row_t* row) {
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
    loom_serve_speech_service_t* service, loom_serve_speech_phase_t phase) {
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
  IREE_TRACE({
    if (phase == SPEECH_PHASE_GENERATING) {
      service->device.section = loom_serve_trace_section_enter(
          LOOM_SERVE_TRACE_SECTION_EXECUTE, service->options.name);
    }
  });
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
                                     loom_serve_speech_row_t* row,
                                     loom_serve_telemetry_outcome_t outcome,
                                     const iree_status_t request_status) {
  const bool row_failed = outcome == LOOM_SERVE_TELEMETRY_COMPLETED &&
                          !iree_status_is_ok(row->status);
  if (row_failed) {
    outcome = row->outcome;
  }
  if (!row->consumer.is_open(row->consumer.self) &&
      (outcome == LOOM_SERVE_TELEMETRY_COMPLETED ||
       outcome == LOOM_SERVE_TELEMETRY_REJECTED)) {
    outcome = !iree_status_is_ok(service->failure)
                  ? LOOM_SERVE_TELEMETRY_FAILED
                  : LOOM_SERVE_TELEMETRY_CANCELLED;
  }
  if (row->job.uploading && outcome == LOOM_SERVE_TELEMETRY_COMPLETED) {
    row->job.voice->phase = SPEECH_VOICE_READY;
  }
  loom_serve_speech_request_end(service, &row->job, outcome,
                                row_failed ? row->status : request_status);
  service->status = iree_status_join(
      service->status,
      row->consumer.end(row->consumer.self, outcome,
                        row_failed ? row->status : request_status));
  row->consumer = (loom_serve_speech_consumer_t){0};
  row->cancelled = false;
  iree_status_free(row->status);
  row->status = iree_ok_status();
  row->outcome = LOOM_SERVE_TELEMETRY_COMPLETED;
  loom_serve_speech_job_deinitialize(service, &row->job);
  memset(&row->job, 0, sizeof(row->job));
  row->invocation = NULL;
  row->phase = SPEECH_ROW_EMPTY;
  row->last_block = false;
  row->has_audio = false;
  if (service->admission == row) {
    service->admission = NULL;
  }
  --service->occupied_count;
}

// Returns false while the model still owns asynchronous row release. The
// existing wait retains the request independently of its connection lifetime.
static bool loom_serve_speech_release(loom_serve_speech_service_t* service,
                                      loom_serve_speech_row_t* row) {
  bool complete = false;
  service->status =
      service->generator.release(service->generator.self, row->invocation,
                                 &complete, &service->device.frontier);
  if (iree_status_is_ok(service->status) && !complete) {
    service->retiring = row;
    service->status = loom_serve_speech_wait(service, SPEECH_PHASE_RELEASING);
    if (iree_status_is_ok(service->status)) {
      return false;
    }
    service->retiring = NULL;
  }
  return true;
}

// Whole-arena roots can be borrowed by another row's accepted batch. All
// source releases therefore happen at a retired model cut, even for peers
// whose final encoded output was delivered while that batch was running.
static void loom_serve_speech_release_finished(
    loom_serve_speech_service_t* service, bool* progress) {
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity &&
       service->phase == SPEECH_PHASE_IDLE;
       ++i) {
    loom_serve_speech_row_t* row = &service->rows[i];
    if (row->phase == SPEECH_ROW_EMPTY ||
        (row->consumer.is_open(row->consumer.self) && !service->stopping &&
         row->phase != SPEECH_ROW_COMPLETE)) {
      continue;
    }
    const bool failed = !iree_status_is_ok(service->failure) ||
                        !iree_status_is_ok(service->status);
    if (!failed) {
      if (row->phase == SPEECH_ROW_WAITING) {
        service->generator.discard(service->generator.self, row->invocation);
      } else if (row->phase != SPEECH_ROW_QUEUED) {
        if (!loom_serve_speech_release(service, row)) {
          *progress = true;
          continue;
        }
      }
    }
    loom_serve_speech_finish(service, row,
                             failed || !iree_status_is_ok(service->status)
                                 ? LOOM_SERVE_TELEMETRY_FAILED
                                 : LOOM_SERVE_TELEMETRY_COMPLETED,
                             iree_status_is_ok(service->status)
                                 ? service->failure
                                 : service->status);
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
                                       loom_serve_service_flags_t flags,
                                       bool* progress) {
  loom_serve_speech_row_t* row = service->admission;
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
    // An unchanged, successfully consumed frontier needs no new wait.
    const bool complete = previous.semaphores[0] == frontier.semaphores[0] &&
                          previous.semaphores[1] == frontier.semaphores[1] &&
                          previous.values[0] == frontier.values[0] &&
                          previous.values[1] == frontier.values[1];
    if (!complete) {
      service->device.frontier = frontier;
      service->status = loom_serve_speech_wait(service, SPEECH_PHASE_RESERVING);
    } else {
      service->generator.complete_admit(service->generator.self,
                                        row->invocation);
      row->phase = SPEECH_ROW_READY;
      service->admission = NULL;
    }
  }
  if (iree_status_is_ok(service->status) &&
      admission == LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY) {
    if (loom_serve_speech_has_resident(service) ||
        iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY)) {
      return;
    }
    service->generator.discard(service->generator.self, row->invocation);
    iree_status_t status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "model parameters and request state exceed available capacity");
    loom_serve_speech_finish(service, row, LOOM_SERVE_TELEMETRY_REJECTED,
                             status);
    iree_status_free(status);
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
  const loom_serve_speech_phase_t phase = service->phase;
  service->phase = SPEECH_PHASE_IDLE;
  IREE_TRACE_SECTION_LEAVE(service->device.section);
  IREE_TRACE(service->device.section = 0);
  if (phase == SPEECH_PHASE_PREPARING &&
      (iree_status_is_invalid_argument(cpu_status) ||
       (service->admission->job.uploading &&
        iree_status_is_unimplemented(cpu_status))) &&
      iree_status_is_ok(service->status)) {
    loom_serve_speech_row_t* row = service->admission;
    const bool connected = row->consumer.is_open(row->consumer.self);
    loom_serve_speech_diagnose(service, row->job.id, cpu_status);
    loom_serve_speech_finish(service, row,
                             connected ? LOOM_SERVE_TELEMETRY_REJECTED
                                       : LOOM_SERVE_TELEMETRY_CANCELLED,
                             cpu_status);
    iree_status_free(cpu_status);
    return;
  }
  service->status = iree_status_join(service->status, cpu_status);
  if (!iree_status_is_ok(service->status)) {
    service->retiring = NULL;
    return;
  }
  if (phase == SPEECH_PHASE_PREPARING && service->admission->job.uploading) {
    // The model now owns decoded/preprocessed input independently. Retire the
    // original wire recording as soon as its CPU borrower joins.
    loom_serve_speech_voice_request_deinitialize(
        &service->admission->job.upload);
  }
  if (phase == SPEECH_PHASE_RELEASING) {
    loom_serve_speech_row_t* row = service->retiring;
    service->retiring = NULL;
    if (loom_serve_speech_release(service, row)) {
      loom_serve_speech_finish(service, row,
                               iree_status_is_ok(service->status)
                                   ? LOOM_SERVE_TELEMETRY_COMPLETED
                                   : LOOM_SERVE_TELEMETRY_FAILED,
                               service->status);
    }
  } else if (phase == SPEECH_PHASE_RESERVING) {
    loom_serve_speech_row_t* row = service->admission;
    service->generator.complete_admit(service->generator.self, row->invocation);
    row->phase = SPEECH_ROW_READY;
    service->admission = NULL;
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
      const loom_serve_speech_consumer_t consumer =
          service->batch.rows[i]->consumer;
      cancelled |= !consumer.is_open(consumer.self);
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
      loom_serve_speech_row_t* row = service->batch.rows[i];
      if (!iree_status_is_ok(row->status)) {
        loom_serve_speech_diagnose(service, row->job.id, row->status);
      }
      row->phase = iree_status_is_ok(row->status) ? SPEECH_ROW_DELIVERING
                                                  : SPEECH_ROW_COMPLETE;
    }
    service->batch.count = 0;
    service->admit_next = true;
  }
}

// Publication only touches consumers whose finite encoding borrower has joined.
// A consumer owns its copied output independently of source feedback lifetime.
static void loom_serve_speech_deliver(loom_serve_speech_service_t* service,
                                      bool* progress) {
  for (iree_host_size_t i = 0; i < service->generator.request_capacity &&
                               iree_status_is_ok(service->status);
       ++i) {
    loom_serve_speech_row_t* row = &service->rows[i];
    if (row->phase != SPEECH_ROW_DELIVERING || row->cancelled ||
        service->stopping) {
      continue;
    }
    bool published = false;
    iree_status_t status =
        row->consumer.publish(row->consumer.self, &published);
    if (iree_status_is_cancelled(status) ||
        iree_status_is_unavailable(status)) {
      loom_serve_speech_diagnose(service, row->job.id, status);
      iree_status_free(status);
      row->consumer.cancel(row->consumer.self);
      row->cancelled = true;
      *progress = true;
    } else if (!iree_status_is_ok(status)) {
      service->status = status;
    } else if (published) {
      if (row->has_audio && !row->job.output_published) {
        loom_serve_telemetry_latency_publish_output(
            &service->telemetry.latency, service->telemetry.stream, row->job.id,
            row->job.arrival_time);
        row->job.output_published = true;
      }
      row->phase = row->last_block ? SPEECH_ROW_COMPLETE : SPEECH_ROW_READY;
      *progress = true;
    }
  }
}

static void loom_serve_speech_prune(loom_serve_speech_service_t* service,
                                    bool* progress) {
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    loom_serve_speech_row_t* row = &service->rows[i];
    if (row->phase != SPEECH_ROW_EMPTY && !row->cancelled &&
        !row->consumer.is_open(row->consumer.self)) {
      row->consumer.cancel(row->consumer.self);
      row->cancelled = true;
      *progress = true;
    }
  }
  iree_host_size_t retained = 0;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    loom_serve_speech_pending_t pending = service->pending[i];
    if (!pending.consumer.is_open(pending.consumer.self)) {
      loom_serve_speech_request_end(service, &pending.job,
                                    LOOM_SERVE_TELEMETRY_CANCELLED,
                                    iree_ok_status());
      service->status = iree_status_join(
          service->status, pending.consumer.end(pending.consumer.self,
                                                LOOM_SERVE_TELEMETRY_CANCELLED,
                                                iree_ok_status()));
      loom_serve_speech_job_deinitialize(service, &pending.job);
      *progress = true;
    } else {
      service->pending[retained++] = pending;
    }
  }
  service->pending_count = retained;
}

static loom_serve_speech_row_t* loom_serve_speech_free_row(
    loom_serve_speech_service_t* service) {
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    if (service->rows[i].phase == SPEECH_ROW_EMPTY) {
      return &service->rows[i];
    }
  }
  return NULL;
}

static void loom_serve_speech_assign(loom_serve_speech_service_t* service,
                                     loom_serve_speech_row_t* row,
                                     loom_serve_speech_pending_t pending) {
  row->consumer = pending.consumer;
  row->job = pending.job;
  row->phase = SPEECH_ROW_QUEUED;
  ++service->occupied_count;
}

// Moving a host-only queued receipt borrows no model state. Existing CPU tasks
// hold stable rows, never pointers into this compact FIFO.
static void loom_serve_speech_fill_rows(loom_serve_speech_service_t* service) {
  loom_serve_speech_row_t* row = NULL;
  while (service->pending_count &&
         (row = loom_serve_speech_free_row(service)) != NULL) {
    loom_serve_speech_assign(service, row, service->pending[0]);
    --service->pending_count;
    memmove(service->pending, service->pending + 1,
            service->pending_count * sizeof(*service->pending));
  }
}

iree_status_t loom_serve_speech_service_try_open(
    loom_serve_speech_service_t* service, loom_serve_speech_request_t* request,
    loom_serve_speech_stream_t** out_stream) {
  *out_stream = NULL;
  if (service->stopping) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "speech service is stopping");
  }
  loom_serve_speech_fill_rows(service);
  loom_serve_speech_row_t* row = loom_serve_speech_free_row(service);
  if (!row && service->pending_count == service->options.pending_capacity) {
    return iree_ok_status();
  }
  loom_serve_speech_pending_t pending = {0};
  IREE_RETURN_IF_ERROR(loom_serve_speech_voice_capture(
      service, iree_string_builder_view(&request->voice), &pending.job.voice));
  const iree_status_t status = loom_serve_speech_stream_allocate(
      service->generator.format, service->generator.block_capacity,
      service->host_allocator, out_stream, &pending.consumer);
  if (!iree_status_is_ok(status)) {
    loom_serve_speech_job_deinitialize(service, &pending.job);
    return status;
  }
  pending.job =
      (loom_serve_speech_job_t){.id = ++service->telemetry.counters.requests,
                                .arrival_time = iree_time_now(),
                                .voice = pending.job.voice,
                                .request = *request};
  *request = (loom_serve_speech_request_t){0};
  loom_serve_speech_event(service, &LOOM_SERVE_TELEMETRY_REQUEST_BEGIN,
                          pending.job.id);
  if (row) {
    loom_serve_speech_assign(service, row, pending);
  } else {
    service->pending[service->pending_count++] = pending;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_speech_service_try_upload_voice(
    loom_serve_speech_service_t* service,
    loom_serve_speech_voice_request_t* request,
    loom_serve_speech_voice_upload_t** out_upload) {
  *out_upload = NULL;
  if (service->stopping) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "speech service is stopping");
  }
  loom_serve_speech_fill_rows(service);
  loom_serve_speech_row_t* row = loom_serve_speech_free_row(service);
  if (!row && service->pending_count == service->options.pending_capacity) {
    return iree_ok_status();
  }
  loom_serve_speech_pending_t pending = {.job = {.uploading = true}};
  IREE_RETURN_IF_ERROR(loom_serve_speech_voice_reserve(
      service, iree_string_builder_view(&request->id), &pending.job.voice));
  const iree_status_t status = loom_serve_speech_voice_upload_allocate(
      service->host_allocator, out_upload, &pending.consumer);
  if (!iree_status_is_ok(status)) {
    loom_serve_speech_job_deinitialize(service, &pending.job);
    return status;
  }
  pending.job.id = ++service->telemetry.counters.requests;
  pending.job.arrival_time = iree_time_now();
  pending.job.upload = *request;
  *request = (loom_serve_speech_voice_request_t){0};
  loom_serve_speech_event(service, &LOOM_SERVE_TELEMETRY_REQUEST_BEGIN,
                          pending.job.id);
  if (row) {
    loom_serve_speech_assign(service, row, pending);
  } else {
    service->pending[service->pending_count++] = pending;
  }
  return iree_ok_status();
}

static const char* loom_serve_speech_rejection_code(
    const iree_status_t status) {
  switch (iree_status_code(status)) {
    case IREE_STATUS_NOT_FOUND:
      return "404 Not Found";
    case IREE_STATUS_ALREADY_EXISTS:
      return "409 Conflict";
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return "503 Service Unavailable";
    case IREE_STATUS_UNIMPLEMENTED:
      return "501 Not Implemented";
    default:
      return "400 Bad Request";
  }
}

static iree_status_t loom_serve_speech_accept_voice(
    loom_serve_speech_service_t* service,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_speech_pending_t* pending, bool removing) {
  iree_status_t status = iree_ok_status();
  if (removing) {
    iree_string_builder_t id;
    iree_string_builder_initialize(service->host_allocator, &id);
    status = loom_serve_speech_voice_http_remove_parse(
        request, service->options.name, &id);
    if (iree_status_is_ok(status)) {
      status = loom_serve_speech_service_remove_voice(
          service, iree_string_builder_view(&id));
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_speech_http_voice_consumer_allocate(
          connection, iree_string_builder_view(&id), service->host_allocator,
          &pending->consumer);
    }
    iree_string_builder_deinitialize(&id);
    if (iree_status_is_ok(status)) {
      loom_serve_speech_request_end(service, &pending->job,
                                    LOOM_SERVE_TELEMETRY_COMPLETED, status);
      return pending->consumer.end(pending->consumer.self,
                                   LOOM_SERVE_TELEMETRY_COMPLETED, status);
    }
  } else {
    pending->job.uploading = true;
    status = loom_serve_speech_voice_http_parse(request, service->options.name,
                                                service->host_allocator,
                                                &pending->job.upload);
    loom_serve_speech_fill_rows(service);
    loom_serve_speech_row_t* row = loom_serve_speech_free_row(service);
    if (iree_status_is_ok(status) && !row &&
        service->pending_count == service->options.pending_capacity) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "speech admission queue is full");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_speech_voice_reserve(
          service, iree_string_builder_view(&pending->job.upload.id),
          &pending->job.voice);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_speech_http_voice_consumer_allocate(
          connection, iree_string_builder_view(&pending->job.upload.id),
          service->host_allocator, &pending->consumer);
    }
    if (iree_status_is_ok(status)) {
      if (row) {
        loom_serve_speech_assign(service, row, *pending);
      } else {
        service->pending[service->pending_count++] = *pending;
      }
      return iree_ok_status();
    }
    loom_serve_speech_job_deinitialize(service, &pending->job);
  }
  const char* response_code = loom_serve_speech_rejection_code(status);
  const iree_status_t response =
      loom_serve_speech_reject(service, connection, &pending->job,
                               response_code, iree_status_message(status));
  loom_serve_speech_diagnose(service, pending->job.id, status);
  iree_status_free(status);
  return response;
}

static iree_status_t loom_serve_speech_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_service_flags_t flags) {
  loom_serve_speech_service_t* service = self;
  loom_serve_speech_pending_t pending = {
      .job = {.id = ++service->telemetry.counters.requests,
              .arrival_time = iree_time_now()}};
  loom_serve_speech_event(service, &LOOM_SERVE_TELEMETRY_REQUEST_BEGIN,
                          pending.job.id);
  const bool post = iree_string_view_equal(request->method, IREE_SV("POST"));
  const bool remove_voice = iree_string_view_equal(
      request->target, IREE_SV("/v1/audio/voices/remove"));
  if (post &&
      (remove_voice ||
       iree_string_view_equal(request->target, IREE_SV("/v1/audio/voices")))) {
    return loom_serve_speech_accept_voice(service, connection, request,
                                          &pending, remove_voice);
  }
  if (!post ||
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
  if (iree_status_is_ok(status)) {
    status = loom_serve_speech_voice_capture(
        service, iree_string_builder_view(&pending.job.request.voice),
        &pending.job.voice);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_speech_job_deinitialize(service, &pending.job);
    iree_string_view_t message = iree_status_message(status);
    if (!message.size) {
      message = iree_make_cstring_view(
          iree_status_code_string(iree_status_code(status)));
    }
    iree_status_t response_status = loom_serve_speech_reject(
        service, connection, &pending.job,
        loom_serve_speech_rejection_code(status), message);
    loom_serve_speech_diagnose(service, pending.job.id, status);
    iree_status_free(status);
    return response_status;
  }
  loom_serve_speech_fill_rows(service);
  loom_serve_speech_row_t* row = loom_serve_speech_free_row(service);
  if (!row && service->pending_count == service->options.pending_capacity) {
    loom_serve_speech_job_deinitialize(service, &pending.job);
    return loom_serve_speech_reject(service, connection, &pending.job,
                                    "503 Service Unavailable",
                                    IREE_SV("speech admission queue is full"));
  }
  status = loom_serve_speech_http_consumer_allocate(
      connection, service->generator.format, service->generator.block_capacity,
      service->host_allocator, &pending.consumer);
  if (!iree_status_is_ok(status)) {
    loom_serve_speech_job_deinitialize(service, &pending.job);
    iree_status_t response_status = loom_serve_speech_reject(
        service, connection, &pending.job,
        iree_status_is_invalid_argument(status) ? "400 Bad Request"
                                                : "503 Service Unavailable",
        iree_status_message(status));
    loom_serve_speech_diagnose(service, pending.job.id, status);
    iree_status_free(status);
    return response_status;
  }
  if (row) {
    loom_serve_speech_assign(service, row, pending);
  } else {
    service->pending[service->pending_count++] = pending;
  }
  return iree_ok_status();
}

static const char* loom_serve_speech_row_phase(
    const loom_serve_speech_row_t* row) {
  if (row->phase == SPEECH_ROW_READY && !row->cancelled &&
      !row->consumer.can_write(row->consumer.self)) {
    return "delivering";
  }
  return speech_row_phases[row->phase];
}

static loom_serve_service_snapshot_t loom_serve_speech_snapshot(void* self) {
  const loom_serve_speech_service_t* service = self;
  const iree_time_t now = iree_time_now();
  loom_serve_service_snapshot_t snapshot = {
      .phase = speech_phases[service->phase],
      .queued_requests = service->pending_count,
      .oldest_queued_age = service->pending_count
                               ? now - service->pending[0].job.arrival_time
                               : 0};
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    const loom_serve_speech_row_t* row = &service->rows[i];
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
  const loom_serve_speech_job_t* job = NULL;
  const char* phase = "queued";
  if (index < service->pending_count) {
    job = &service->pending[index].job;
  } else {
    index -= service->pending_count;
    for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
      const loom_serve_speech_row_t* row = &service->rows[i];
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
  const loom_serve_service_snapshot_t state =
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
  // A borrower can fail after orderly shutdown has already started. Preserve
  // that first terminal failure through later drain turns and result
  // detachment.
  if (iree_status_is_ok(service->failure) &&
      !iree_status_is_ok(service->status)) {
    service->failure = iree_status_clone(service->status);
  }
  service->stopping = true;
  const loom_serve_telemetry_outcome_t outcome =
      !iree_status_is_ok(service->failure) ? LOOM_SERVE_TELEMETRY_FAILED
                                           : LOOM_SERVE_TELEMETRY_CANCELLED;
  for (iree_host_size_t i = 0; i < service->pending_count; ++i) {
    loom_serve_speech_pending_t* pending = &service->pending[i];
    loom_serve_speech_request_end(service, &pending->job, outcome,
                                  service->status);
    service->status = iree_status_join(
        service->status, pending->consumer.end(pending->consumer.self, outcome,
                                               service->status));
    loom_serve_speech_job_deinitialize(service, &pending->job);
  }
  service->pending_count = 0;
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    loom_serve_speech_row_t* row = &service->rows[i];
    if (row->phase != SPEECH_ROW_EMPTY && !row->cancelled) {
      row->consumer.cancel(row->consumer.self);
      row->cancelled = true;
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
                                       loom_serve_service_flags_t flags,
                                       bool* progress) {
  if (service->admission) {
    loom_serve_speech_prepared(service, flags, progress);
    if (service->phase != SPEECH_PHASE_IDLE ||
        !iree_status_is_ok(service->status)) {
      return;
    }
  }
  loom_serve_speech_row_t* queued = NULL;
  service->batch.count = 0;
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    const iree_host_size_t index =
        (service->cursor + i) % service->generator.request_capacity;
    loom_serve_speech_row_t* row = &service->rows[index];
    if (row->phase == SPEECH_ROW_QUEUED &&
        (!queued || row->job.id < queued->job.id)) {
      queued = row;
    }
    if (row->phase == SPEECH_ROW_READY && !row->cancelled &&
        row->consumer.can_write(row->consumer.self)) {
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

static iree_status_t loom_serve_speech_advance(void* self,
                                               loom_serve_service_flags_t flags,
                                               bool* out_progress) {
  loom_serve_speech_service_t* service = self;
  *out_progress = false;
  if (!service->stopping &&
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    loom_serve_speech_stop(service);
    *out_progress = true;
  }
  loom_serve_speech_prune(service, out_progress);
  loom_serve_speech_collect(service, out_progress);
  if (!iree_status_is_ok(service->status)) {
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
    if (!service->stopping && iree_status_is_ok(service->status) &&
        service->phase == SPEECH_PHASE_IDLE) {
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
                               true, &timer_progress, &tick));
  *out_progress |= timer_progress;
  if (tick) {
    loom_serve_speech_heartbeat(service);
  }
  if (!iree_status_is_ok(service->status)) {
    loom_serve_speech_stop(service);
    *out_progress = true;
  }
  if (service->stopping && service->phase == SPEECH_PHASE_IDLE) {
    loom_serve_speech_release_finished(service, out_progress);
  }
  if (!iree_status_is_ok(service->status)) {
    loom_serve_speech_stop(service);
    loom_serve_speech_diagnose(service, 0, service->status);
  }
  iree_status_t status = service->status;
  service->status = iree_ok_status();
  return status;
}

static iree_status_t loom_serve_speech_open(
    void* self, loom_serve_speech_request_t* request,
    loom_serve_speech_stream_t** out_stream) {
  return loom_serve_speech_service_try_open(self, request, out_stream);
}

static iree_status_t loom_serve_speech_upload_voice(
    void* self, loom_serve_speech_voice_request_t* request,
    loom_serve_speech_voice_upload_t** out_upload) {
  return loom_serve_speech_service_try_upload_voice(self, request, out_upload);
}

static iree_status_t loom_serve_speech_remove_voice(void* self,
                                                    iree_string_view_t id) {
  return loom_serve_speech_service_remove_voice(self, id);
}

loom_serve_service_t loom_serve_speech_service_interface(
    loom_serve_speech_service_t* service) {
  static const iree_string_view_t input_modalities[] = {IREE_SVL("text"),
                                                        IREE_SVL("audio")};
  static const iree_string_view_t output_modalities[] = {IREE_SVL("audio")};
  return (loom_serve_service_t){
      .name = service->options.name,
      .self = service,
      .set_observer = loom_serve_speech_set_observer,
      .is_pending = loom_serve_speech_is_pending,
      .is_active = loom_serve_speech_is_active,
      .accept = loom_serve_speech_accept,
      .try_open_speech = loom_serve_speech_open,
      .try_upload_voice = service->generator.voice_capacity
                              ? loom_serve_speech_upload_voice
                              : NULL,
      .remove_voice = service->generator.voice_capacity
                          ? loom_serve_speech_remove_voice
                          : NULL,
      .advance = loom_serve_speech_advance,
      .tokenizer = service->generator.tokenizer,
      .kind = LOOM_SERVE_MODEL_KIND_SPEECH,
      .endpoint = loom_serve_model_kind_endpoint(LOOM_SERVE_MODEL_KIND_SPEECH),
      .capabilities = {.input_modalities = {service->generator.voice_capacity
                                                ? 2
                                                : 1,
                                            input_modalities},
                       .output_modalities = {IREE_ARRAYSIZE(output_modalities),
                                             output_modalities}},
      .snapshot = loom_serve_speech_snapshot,
      .inspection = {.request_at = loom_serve_speech_request_at},
      .telemetry = {.stream = service->telemetry.stream,
                    .counters = &service->telemetry.counters,
                    .latency = &service->telemetry.latency},
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
  iree_status_free(service->failure);
  for (iree_host_size_t i = 0;
       service->voices && i < service->generator.voice_capacity; ++i) {
    service->voices[i].phase = SPEECH_VOICE_REMOVED;
    loom_serve_speech_voice_collect(service, &service->voices[i]);
  }
  iree_allocator_free(host_allocator, service->voices);
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
      !generator.batch_units || !generator.format.sample_rate ||
      (generator.voice_capacity &&
       (!generator.prepare_voice || !generator.release_voice)) ||
      !generator.format.channels ||
      generator.request_capacity > SIZE_MAX / sizeof(loom_serve_speech_row_t) ||
      generator.request_capacity > SIZE_MAX / sizeof(void*) ||
      !options->name.size || !options->pending_capacity ||
      options->pending_capacity >
          SIZE_MAX / sizeof(loom_serve_speech_pending_t) ||
      options->heartbeat_interval < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech service requires a complete generator, "
                            "bounded blocks, name and queue capacity");
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
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, generator.request_capacity, sizeof(*service->rows),
      (void**)&service->rows);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < generator.request_capacity; ++i) {
      service->rows[i].index = i;
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
  if (iree_status_is_ok(status) && generator.voice_capacity) {
    status = iree_allocator_malloc_array(
        host_allocator, generator.voice_capacity, sizeof(*service->voices),
        (void**)&service->voices);
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
