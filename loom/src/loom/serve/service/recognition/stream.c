// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/recognition/stream.h"

#include <math.h>
#include <string.h>

#include "iree/async/operations/semaphore.h"
#include "iree/base/internal/unicode.h"
#include "iree/base/threading/mutex.h"
#include "loom/service/telemetry/record.h"
#include "loom/service/telemetry/tracing.h"

typedef enum loom_serve_recognition_row_phase_e {
  RECOGNITION_ROW_QUEUED = 0,
  RECOGNITION_ROW_WAITING,
  RECOGNITION_ROW_READY,
  RECOGNITION_ROW_OFFERING,
  RECOGNITION_ROW_AWAIT_INPUT,
  RECOGNITION_ROW_EXECUTING,
  RECOGNITION_ROW_PUBLISHING,
  RECOGNITION_ROW_COMPLETE,
} loom_serve_recognition_row_phase_t;

struct loom_serve_recognition_stream_t {
  // Allocator owning this handle, trailing request strings, and results.
  iree_allocator_t host_allocator;
  // Immutable normalized input geometry, independent of model lifetime.
  loom_serve_recognition_format_t format;
  // Immutable source input-window capacity, independent of execution storage.
  iree_host_size_t input_capacity;
  // Maximum cumulative input frames accepted from the caller.
  uint64_t frame_limit;
  // Stable resident index, independent of a packed execution ordinal.
  iree_host_size_t index;
  // Owner-published lifecycle; CPU workers never mutate it.
  loom_serve_recognition_row_phase_t phase;
  // Caller still owns this handle and can receive source events.
  bool caller_open;
  // Execution ownership, consumed after accepted host borrowers retire.
  bool service_open;
  // Stop suppresses input/output without consuming the caller's ownership.
  bool cancelled;
  // Monotonic arrival order used for bounded host admission.
  uint64_t arrival;
  // Request options borrowing strings in this handle's trailing allocation.
  loom_serve_recognition_request_t request;
  // Source request, released only at a retired model-wide cut.
  void* invocation;
  // Bounded producer ring; CPU offers borrow only captured disjoint spans.
  loom_serve_recognition_input_t input;
  // Captured input held constant through CPU work selection and collection.
  loom_serve_recognition_input_block_t snapshot;
  // Latest per-segment revisions; CPU collection owns mutation until publish.
  loom_serve_recognition_document_t document;
  // Reusable caller-owned output, mutated only by accepted CPU collection.
  loom_serve_recognition_events_t* output;
  // Owner-published events still awaiting consumption, independent of workers.
  iree_host_size_t pending_count;
  // Owned terminal failure, independent of service/model lifetime.
  iree_status_t status;
  // Producer-established terminal classification, independent of status codes.
  loom_service_telemetry_outcome_t outcome;
  // Optional recipient retained through execution ownership, never by results.
  loom_serve_recognition_completion_t completion;
  // CPU-published offer, read only after its task's final release callback.
  loom_serve_recognition_work_t work;
  // CPU-published collection, separated from owner-visible output.
  struct {
    // Source-validated input prefix acknowledged by the retired batch.
    iree_host_size_t consumed_frames;
    // Source has no further numerical work or events for this request.
    bool complete;
    // Owned collection/document error, transferred after CPU retirement.
    iree_status_t status;
  } result;
};

typedef enum loom_serve_recognition_phase_e {
  RECOGNITION_PHASE_IDLE = 0,
  RECOGNITION_PHASE_PREPARING,
  RECOGNITION_PHASE_RESERVING,
  RECOGNITION_PHASE_OFFERING,
  RECOGNITION_PHASE_EXECUTING,
  RECOGNITION_PHASE_COLLECTING,
  RECOGNITION_PHASE_RELEASING,
} loom_serve_recognition_phase_t;

enum { RECOGNITION_EVENT_CPU = 1u << 0, RECOGNITION_EVENT_DEVICE = 1u << 1 };

enum {
  RECOGNITION_OPERATION_ADMISSION = 0,
  RECOGNITION_OPERATION_BATCH,
  RECOGNITION_OPERATION_COUNT,
};

struct loom_serve_recognition_stream_service_t {
  // Allocator owning service, fixed row arrays, and request storage.
  iree_allocator_t host_allocator;
  // Borrowed source model, owning all PCM that escapes CPU offers.
  loom_serve_recognition_generator_t generator;
  // Copied host retention bounds and borrowed deployment identity.
  loom_serve_recognition_stream_service_options_t options;
  // Shared finite-task executor; no service task blocks on device work.
  iree_task_executor_t* executor;
  // Shared proactor for model admission and numerical retirement readiness.
  iree_async_proactor_t* proactor;
  // One active CPU/device borrow, independent of individual stream phases.
  loom_serve_recognition_phase_t phase;
  // Stops new source work while accepted callbacks drain.
  bool stopping;
  // Owned terminal service failure, copied into affected stream results.
  iree_status_t failure;
  // Fixed resident slots, cleared when execution ownership ends independently
  // of caller ownership. NULL denotes an available source row.
  loom_serve_recognition_stream_t** rows;
  // Next request arrival identity; independent of resident row reuse.
  uint64_t next_arrival;
  // At most one CPU-prepared request waiting for source admission.
  loom_serve_recognition_stream_t* admission;
  // Request retained through asynchronous row-backing release readiness.
  loom_serve_recognition_stream_t* retiring;
  // First row in the next rotating compatibility selection.
  iree_host_size_t cursor;
  // Allows one new admission between completed numerical batches.
  bool admit_next;
  // Owner-published facts; readers never inspect live input or worker storage.
  struct {
    // Parent-retained storage and its single summary publisher.
    loom_runtime_service_control_binding_t binding;
    // Source's current admission-versus-batch opportunity.
    loom_runtime_service_offer_t offer;
    // Offered queued request, borrowed until the next owner mutation.
    loom_serve_recognition_stream_t* queued;
    // Distinguishes reused rows with otherwise identical request geometry.
    uint64_t arrival;
    // Candidate participation before source selection on its CPU borrower.
    uint64_t request_count;
    // Independent admission and exclusive source/release obligations.
    loom_runtime_service_operation_t operations[RECOGNITION_OPERATION_COUNT];
    // Last published lifetimes, independent of mutable readiness reasons.
    uint64_t published_serials[RECOGNITION_OPERATION_COUNT];
    // Monotonic accepted identity, independent of legal offer revisions.
    uint64_t next_serial;
    // Last published summary, avoiding stores on unchanged owner turns.
    loom_runtime_service_observation_t observation;
  } scheduling;
  // Callback publication and observer detachment share one final-access lock.
  struct {
    // Serializes worker/proactor publication with owner collection.
    iree_slim_mutex_t mutex;
    // Nonblocking wake installed before any asynchronous operation.
    loom_runtime_wake_callback_t observer;
    // Completed CPU/device operations awaiting owner consumption.
    uint32_t ready;
    // Owned readiness failure, independent of source input rejection.
    iree_status_t device_status;
    // Owned CPU task outcome, published before final task release.
    iree_status_t cpu_status;
  } events;
  // One reusable finite source task.
  struct {
    // Intrusive task storage, reused only after its final release callback.
    iree_task_process_t process;
    // Immutable selector while the CPU task is live.
    loom_serve_recognition_phase_t phase;
  } cpu;
  // Offers and selected members are separate to retain every candidate's cut.
  struct {
    // Candidate snapshot, immutable until offer task publication.
    loom_serve_recognition_stream_t** candidates;
    // Number of candidate snapshots borrowed by the offer task.
    iree_host_size_t candidate_count;
    // Original selected ordinals, retained even after caller cancellation.
    loom_serve_recognition_stream_t** rows;
    // Source handles parallel to selected rows.
    void** invocations;
    // Selected numerical member count, written by the offer task.
    iree_host_size_t count;
  } batch;
  // One accepted ALL wait, retaining both readiness and retirement frontiers.
  struct {
    // Proactor-owned wait storage through completion callback return.
    iree_async_semaphore_wait_operation_t operation;
    // Last accepted frontier, consumed successfully before reuse on idle cuts.
    loom_runtime_execution_frontier_t frontier;
    // Toll-free semaphore interfaces borrowed from the frontier.
    iree_async_semaphore_t* semaphores[2];
    // Accepted numerical batch through readiness; zero for admission/release.
    IREE_TRACE(iree_section_id_t section;)
  } device;
};

static void loom_serve_recognition_stream_publish(
    loom_serve_recognition_stream_service_t* service, bool allow_start);

static loom_runtime_service_operation_key_t
loom_serve_recognition_begin_operation(
    loom_serve_recognition_stream_service_t* service, uint64_t ordinal,
    loom_runtime_service_offer_key_t origin,
    loom_runtime_service_work_flags_t work) {
  IREE_ASSERT(service->scheduling.next_serial != UINT64_MAX);
  loom_runtime_service_operation_t* operation =
      &service->scheduling.operations[ordinal];
  *operation = (loom_runtime_service_operation_t){
      .key = {.ordinal =
                  service->scheduling.binding.operation_ordinal + ordinal,
              .serial = ++service->scheduling.next_serial},
      .origin = origin,
      .work = work};
  return operation->key;
}

static iree_status_t loom_serve_recognition_cpu_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  loom_serve_recognition_stream_service_t* service = process->user_data;
  result->completed = true;
  result->did_work = true;
  if (service->cpu.phase == RECOGNITION_PHASE_PREPARING) {
    loom_serve_recognition_stream_t* row = service->admission;
    return service->generator.prepare(service->generator.self, row->index,
                                      &row->request, &row->invocation);
  }
  iree_status_t status = iree_ok_status();
  if (service->cpu.phase == RECOGNITION_PHASE_OFFERING) {
    uint64_t tag = 0;
    uint64_t remaining = service->generator.batch_units;
    service->batch.count = 0;
    for (iree_host_size_t i = 0;
         i < service->batch.candidate_count && iree_status_is_ok(status); ++i) {
      loom_serve_recognition_stream_t* row = service->batch.candidates[i];
      status = service->generator.prepare_work(
          service->generator.self, row->invocation, &row->snapshot, &row->work);
      if (iree_status_is_ok(status) && row->work.units) {
        if (!service->batch.count) {
          tag = row->work.tag;
        }
        if (row->work.tag == tag && row->work.units <= remaining) {
          remaining -= row->work.units;
          service->batch.rows[service->batch.count] = row;
          service->batch.invocations[service->batch.count++] = row->invocation;
        }
      }
    }
    if (iree_status_is_ok(status) && service->batch.count) {
      status = service->generator.prepare_batch(service->generator.self,
                                                service->batch.count,
                                                service->batch.invocations);
    }
  } else {
    for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
      loom_serve_recognition_stream_t* row = service->batch.rows[i];
      row->result.consumed_frames = 0;
      row->result.complete = false;
      loom_serve_recognition_result_t collected;
      row->result.status = service->generator.collect(
          service->generator.self, i, row->output, &collected);
      if (iree_status_is_ok(row->result.status)) {
        row->result.consumed_frames = collected.consumed_frames;
        row->result.complete = collected.complete;
        for (iree_host_size_t j = 0;
             j < loom_serve_recognition_events_count(row->output) &&
             iree_status_is_ok(row->result.status);
             ++j) {
          const loom_serve_recognition_event_t* event =
              loom_serve_recognition_events_at(row->output, j);
          if (event->kind == LOOM_SERVE_RECOGNITION_EVENT_TRANSCRIPT) {
            const loom_serve_recognition_transcript_t* stored = NULL;
            row->result.status = loom_serve_recognition_document_apply(
                &row->document, &event->transcript, &stored);
          }
        }
        if (iree_status_is_ok(row->result.status) && collected.complete) {
          row->result.status =
              loom_serve_recognition_document_complete(&row->document);
        }
      }
    }
  }
  return status;
}

static void loom_serve_recognition_cpu_completed(iree_task_process_t* process,
                                                 iree_status_t status) {
  loom_serve_recognition_stream_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.cpu_status = status;
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_recognition_cpu_released(iree_task_process_t* process) {
  loom_serve_recognition_stream_service_t* service = process->user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.ready |= RECOGNITION_EVENT_CPU;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static void loom_serve_recognition_schedule_cpu(
    loom_serve_recognition_stream_service_t* service,
    loom_serve_recognition_phase_t phase) {
  service->phase = phase;
  service->cpu.phase = phase;
  iree_task_process_initialize(loom_serve_recognition_cpu_run, 0, 1,
                               &service->cpu.process);
  service->cpu.process.user_data = service;
  service->cpu.process.completion_fn = loom_serve_recognition_cpu_completed;
  service->cpu.process.release_fn = loom_serve_recognition_cpu_released;
  iree_task_executor_schedule_process(service->executor, &service->cpu.process);
}

static void loom_serve_recognition_device_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  loom_serve_recognition_stream_service_t* service = user_data;
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.device_status = status;
  service->events.ready |= RECOGNITION_EVENT_DEVICE;
  service->events.observer.wake(service->events.observer.user_data);
  iree_slim_mutex_unlock(&service->events.mutex);
}

static iree_status_t loom_serve_recognition_wait(
    loom_serve_recognition_stream_service_t* service,
    loom_serve_recognition_phase_t phase) {
  iree_async_semaphore_wait_operation_t* wait = &service->device.operation;
  memset(wait, 0, sizeof(*wait));
  iree_async_operation_initialize(
      &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, loom_serve_recognition_device_completed,
      service);
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
    if (phase == RECOGNITION_PHASE_EXECUTING) {
      service->device.section = loom_service_trace_section_enter(
          LOOM_SERVICE_TRACE_SECTION_EXECUTE, service->options.name);
    }
  });
  return iree_ok_status();
}

static void loom_serve_recognition_stream_release(
    loom_serve_recognition_stream_t* stream) {
  if (!stream->caller_open && !stream->service_open) {
    loom_serve_recognition_document_deinitialize(&stream->document);
    loom_serve_recognition_events_free(stream->output);
    iree_status_free(stream->status);
    iree_status_free(stream->result.status);
    iree_allocator_free(stream->host_allocator, stream);
  }
}

// The collected result is stable only in these owner-published phases.
static bool loom_serve_recognition_row_is_complete(
    const loom_serve_recognition_stream_t* row) {
  return row->phase == RECOGNITION_ROW_COMPLETE ||
         (row->phase == RECOGNITION_ROW_PUBLISHING && row->result.complete);
}

static void loom_serve_recognition_stream_end(
    loom_serve_recognition_stream_t* stream) {
  stream->service_open = false;
  loom_serve_recognition_input_deinitialize(&stream->input);
  if (stream->cancelled && iree_status_is_ok(stream->status)) {
    stream->status = iree_status_from_code(IREE_STATUS_CANCELLED);
    stream->outcome = LOOM_SERVICE_TELEMETRY_CANCELLED;
  }
  if (!iree_status_is_ok(stream->status)) {
    stream->pending_count = 0;
  }
  if (stream->completion.end) {
    stream->completion.end(stream->completion.self, stream->outcome,
                           stream->status);
  }
  stream->completion = (loom_serve_recognition_completion_t){0};
  loom_serve_recognition_stream_release(stream);
}

// Retains the request through asynchronous backing release. Failed readiness
// leaves accepted ownership with model teardown, independently of caller close.
static iree_status_t loom_serve_recognition_release(
    loom_serve_recognition_stream_service_t* service,
    loom_serve_recognition_stream_t* row) {
  loom_runtime_service_operation_t* operation =
      &service->scheduling.operations[RECOGNITION_OPERATION_BATCH];
  if (!operation->key.serial) {
    loom_serve_recognition_begin_operation(
        service, RECOGNITION_OPERATION_BATCH,
        (loom_runtime_service_offer_key_t){0},
        LOOM_RUNTIME_SERVICE_WORK_RELEASE);
  }
  bool complete = false;
  iree_status_t status =
      service->generator.release(service->generator.self, row->invocation,
                                 &complete, &service->device.frontier);
  if (iree_status_is_ok(status) && !complete) {
    service->retiring = row;
    status = loom_serve_recognition_wait(service, RECOGNITION_PHASE_RELEASING);
    if (!iree_status_is_ok(status)) {
      service->retiring = NULL;
    }
  }
  if (!iree_status_is_ok(status) || complete) {
    row->invocation = NULL;
  }
  if (iree_status_is_ok(status) && complete) {
    operation->key.serial = 0;
  }
  return status;
}

// Whole model roots may be borrowed by a neighboring row's accepted batch.
// Even a closed or completed peer releases its source state only at this cut.
static iree_status_t loom_serve_recognition_release_finished(
    loom_serve_recognition_stream_service_t* service, bool* progress) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->generator.request_capacity && iree_status_is_ok(status) &&
       service->phase == RECOGNITION_PHASE_IDLE;
       ++i) {
    loom_serve_recognition_stream_t* row = service->rows[i];
    if (!row || (row->caller_open && !service->stopping &&
                 !loom_serve_recognition_row_is_complete(row))) {
      continue;
    }
    if (row->invocation) {
      if (iree_status_is_ok(service->failure)) {
        if (row->phase == RECOGNITION_ROW_WAITING) {
          service->generator.discard(service->generator.self, row->invocation);
          service->scheduling.operations[RECOGNITION_OPERATION_ADMISSION]
              .key.serial = 0;
        } else {
          status = loom_serve_recognition_release(service, row);
          if (service->phase != RECOGNITION_PHASE_IDLE) {
            *progress = true;
            continue;
          }
        }
      }
      // Failed release leaves the accepted handle with model teardown.
      row->invocation = NULL;
      *progress = true;
    }
    if (service->admission == row) {
      service->admission = NULL;
    }
    if (!iree_status_is_ok(status) && iree_status_is_ok(row->status)) {
      row->outcome = LOOM_SERVICE_TELEMETRY_FAILED;
    }
    row->status = iree_status_join(row->status, iree_status_clone(status));
    service->rows[i] = NULL;
    loom_serve_recognition_stream_end(row);
    *progress = true;
  }
  return status;
}

bool loom_serve_recognition_stream_service_is_active(
    const loom_serve_recognition_stream_service_t* service) {
  if (service->phase != RECOGNITION_PHASE_IDLE) {
    return true;
  }
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    const loom_serve_recognition_stream_t* row = service->rows[i];
    if (row && row->invocation && row->phase >= RECOGNITION_ROW_READY) {
      return true;
    }
  }
  return false;
}

bool loom_serve_recognition_stream_service_is_pending(
    const loom_serve_recognition_stream_service_t* service) {
  return service->phase != RECOGNITION_PHASE_IDLE;
}

static iree_status_t loom_serve_recognition_admit(
    loom_serve_recognition_stream_service_t* service, bool shared_activity,
    bool* progress) {
  loom_serve_recognition_stream_t* row = service->admission;
  loom_runtime_residency_admission_t admission;
  loom_runtime_execution_frontier_t frontier;
  IREE_RETURN_IF_ERROR(service->generator.try_admit(
      service->generator.self, row->invocation, &admission, &frontier));
  if (admission == LOOM_RUNTIME_RESIDENCY_ADMISSION_DEFERRED) {
    return iree_ok_status();
  }
  if (admission == LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED) {
    const loom_runtime_execution_frontier_t previous = service->device.frontier;
    const bool complete = previous.semaphores[0] == frontier.semaphores[0] &&
                          previous.semaphores[1] == frontier.semaphores[1] &&
                          previous.values[0] == frontier.values[0] &&
                          previous.values[1] == frontier.values[1];
    if (!complete) {
      service->device.frontier = frontier;
      *progress = true;
      return loom_serve_recognition_wait(service, RECOGNITION_PHASE_RESERVING);
    }
    service->generator.complete_admit(service->generator.self, row->invocation);
    row->phase = RECOGNITION_ROW_READY;
    service->admission = NULL;
    service->scheduling.operations[RECOGNITION_OPERATION_ADMISSION].key.serial =
        0;
    *progress = true;
    return iree_ok_status();
  }
  if (admission == LOOM_RUNTIME_RESIDENCY_ADMISSION_CAPACITY &&
      !shared_activity &&
      !loom_serve_recognition_stream_service_is_active(service)) {
    service->generator.discard(service->generator.self, row->invocation);
    row->invocation = NULL;
    row->status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "model parameters and recognition state exceed available capacity");
    row->outcome = LOOM_SERVICE_TELEMETRY_REJECTED;
    row->phase = RECOGNITION_ROW_COMPLETE;
    service->admission = NULL;
    service->scheduling.operations[RECOGNITION_OPERATION_ADMISSION].key.serial =
        0;
    *progress = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_recognition_collect(
    loom_serve_recognition_stream_service_t* service, bool* progress) {
  iree_slim_mutex_lock(&service->events.mutex);
  const uint32_t ready = service->events.ready;
  service->events.ready = 0;
  iree_status_t status = service->events.device_status;
  service->events.device_status = iree_ok_status();
  iree_status_t cpu_status = iree_ok_status();
  if (iree_any_bit_set(ready, RECOGNITION_EVENT_CPU)) {
    cpu_status = service->events.cpu_status;
    service->events.cpu_status = iree_ok_status();
  }
  iree_slim_mutex_unlock(&service->events.mutex);
  if (!ready) {
    return status;
  }
  *progress = true;
  const loom_serve_recognition_phase_t phase = service->phase;
  service->phase = RECOGNITION_PHASE_IDLE;
  IREE_TRACE_SECTION_LEAVE(service->device.section);
  IREE_TRACE(service->device.section = 0);
  if (phase == RECOGNITION_PHASE_PREPARING &&
      iree_status_is_invalid_argument(cpu_status) &&
      iree_status_is_ok(status)) {
    loom_serve_recognition_stream_t* row = service->admission;
    row->status = cpu_status;
    row->outcome = LOOM_SERVICE_TELEMETRY_REJECTED;
    row->phase = RECOGNITION_ROW_COMPLETE;
    service->admission = NULL;
    service->scheduling.operations[RECOGNITION_OPERATION_ADMISSION].key.serial =
        0;
    return iree_ok_status();
  }
  status = iree_status_join(status, cpu_status);
  if (!iree_status_is_ok(status)) {
    service->retiring = NULL;
    return status;
  }
  if (phase == RECOGNITION_PHASE_RELEASING) {
    loom_serve_recognition_stream_t* row = service->retiring;
    service->retiring = NULL;
    return loom_serve_recognition_release(service, row);
  } else if (phase == RECOGNITION_PHASE_RESERVING) {
    loom_serve_recognition_stream_t* row = service->admission;
    service->generator.complete_admit(service->generator.self, row->invocation);
    row->phase = RECOGNITION_ROW_READY;
    service->admission = NULL;
    service->scheduling.operations[RECOGNITION_OPERATION_ADMISSION].key.serial =
        0;
  } else if (phase == RECOGNITION_PHASE_OFFERING) {
    for (iree_host_size_t i = 0; i < service->batch.candidate_count; ++i) {
      loom_serve_recognition_stream_t* row = service->batch.candidates[i];
      row->phase =
          row->work.units ? RECOGNITION_ROW_READY : RECOGNITION_ROW_AWAIT_INPUT;
    }
    service->batch.candidate_count = 0;
    bool cancelled = service->stopping;
    for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
      cancelled |= !service->batch.rows[i]->caller_open;
    }
    if (service->batch.count && cancelled) {
      service->generator.release_batch(service->generator.self);
      service->batch.count = 0;
    } else if (service->batch.count) {
      for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
        service->batch.rows[i]->phase = RECOGNITION_ROW_EXECUTING;
      }
      service->cursor = (service->batch.rows[0]->index + 1) %
                        service->generator.request_capacity;
      IREE_RETURN_IF_ERROR(service->generator.submit(
          service->generator.self, &service->device.frontier));
      return loom_serve_recognition_wait(service, RECOGNITION_PHASE_EXECUTING);
    }
    service->scheduling.operations[RECOGNITION_OPERATION_BATCH].key.serial = 0;
  } else if (phase == RECOGNITION_PHASE_EXECUTING) {
    // Collection is part of accepted work, including during cancellation and
    // shutdown. It releases source feedback and acknowledges each original row.
    loom_serve_recognition_schedule_cpu(service, RECOGNITION_PHASE_COLLECTING);
  } else if (phase == RECOGNITION_PHASE_COLLECTING) {
    for (iree_host_size_t i = 0; i < service->batch.count; ++i) {
      loom_serve_recognition_stream_t* row = service->batch.rows[i];
      loom_serve_recognition_input_consume(&row->input,
                                           row->result.consumed_frames);
      if (!iree_status_is_ok(row->result.status) &&
          iree_status_is_ok(row->status)) {
        row->outcome = LOOM_SERVICE_TELEMETRY_FAILED;
      }
      row->status = iree_status_join(row->status, row->result.status);
      row->result.status = iree_ok_status();
      row->pending_count =
          iree_status_is_ok(row->status)
              ? loom_serve_recognition_events_count(row->output)
              : 0;
      row->phase = row->pending_count ? RECOGNITION_ROW_PUBLISHING
                   : row->result.complete || !iree_status_is_ok(row->status)
                       ? RECOGNITION_ROW_COMPLETE
                       : RECOGNITION_ROW_READY;
    }
    service->generator.release_batch(service->generator.self);
    service->batch.count = 0;
    service->scheduling.operations[RECOGNITION_OPERATION_BATCH].key.serial = 0;
    service->admit_next = true;
  }
  return iree_ok_status();
}

// Refreshes candidate identity and input extent only at a retired source cut.
// Captured spans remain immutable while source work borrows them; policy
// queries read only the published control and never enter this scan.
static void loom_serve_recognition_stream_publish(
    loom_serve_recognition_stream_service_t* service, bool allow_start) {
  const bool can_offer = allow_start && !service->stopping &&
                         service->phase == RECOGNITION_PHASE_IDLE;
  loom_serve_recognition_stream_t* queued = NULL;
  uint64_t owned_count = 0, waiting_count = 0;
  iree_host_size_t candidate_count = 0;
  bool candidates_changed = false;
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    const iree_host_size_t index =
        (service->cursor + i) % service->generator.request_capacity;
    loom_serve_recognition_stream_t* row = service->rows[index];
    if (!row) {
      continue;
    }
    ++owned_count;
    waiting_count += loom_serve_recognition_stream_is_waiting(row);
    if (!can_offer || !row->caller_open || row->cancelled) {
      continue;
    }
    if (row->phase == RECOGNITION_ROW_QUEUED &&
        (!queued || row->arrival < queued->arrival)) {
      queued = row;
    }
    if (row->phase == RECOGNITION_ROW_AWAIT_INPUT &&
        (row->input.received !=
             row->snapshot.position + row->snapshot.frame_count ||
         row->input.complete != row->snapshot.complete)) {
      row->phase = RECOGNITION_ROW_READY;
    }
    if (row->phase == RECOGNITION_ROW_READY) {
      const loom_serve_recognition_input_block_t input =
          loom_serve_recognition_input_peek(&row->input);
      candidates_changed |= candidate_count >= service->batch.candidate_count ||
                            service->batch.candidates[candidate_count] != row ||
                            row->snapshot.position != input.position ||
                            row->snapshot.frame_count != input.frame_count ||
                            row->snapshot.complete != input.complete;
      row->snapshot = input;
      service->batch.candidates[candidate_count++] = row;
    }
  }
  loom_runtime_service_work_flags_t work = 0;
  uint64_t request_count = 0, arrival = 0;
  if (can_offer) {
    candidates_changed |= candidate_count != service->batch.candidate_count;
    service->batch.candidate_count = candidate_count;
    if (!service->admission && queued &&
        (service->admit_next || !candidate_count)) {
      work = LOOM_RUNTIME_SERVICE_WORK_PREPARATION |
             LOOM_RUNTIME_SERVICE_WORK_ACQUISITION;
      request_count = 1;
      arrival = queued->arrival;
      service->batch.candidate_count = 0;
    } else if (candidate_count) {
      work = LOOM_RUNTIME_SERVICE_WORK_EXECUTION;
      request_count = candidate_count;
    }
  }
  const uint64_t devices = UINT64_C(1) << service->generator.device_ordinal;
  const loom_runtime_service_use_t use = {
      .resource_ordinal = service->generator.device_ordinal};
  loom_runtime_service_control_t* control = service->scheduling.binding.control;
  loom_runtime_service_offer_t* offer = &service->scheduling.offer;
  if (offer->work != work || service->scheduling.arrival != arrival ||
      service->scheduling.request_count != request_count ||
      (work == LOOM_RUNTIME_SERVICE_WORK_EXECUTION && candidates_changed)) {
    IREE_ASSERT(offer->key.revision != UINT64_MAX);
    ++offer->key.revision;
    offer->work = work;
    offer->choice_count = work ? 1 : 0;
    service->scheduling.arrival = arrival;
    service->scheduling.request_count = request_count;
    const loom_runtime_service_choice_publication_t choice = {
        .choice = {.instance_ordinal =
                       service->scheduling.binding.instance_ordinal,
                   .request_count = request_count,
                   .devices = devices,
                   .use_count = 1},
        .uses = &use};
    loom_runtime_service_publish_offer(control, offer, &choice);
  }
  service->scheduling.queued = arrival ? queued : NULL;
  loom_runtime_service_wait_flags_t waits[RECOGNITION_OPERATION_COUNT] = {0};
  if (service->phase == RECOGNITION_PHASE_PREPARING) {
    waits[RECOGNITION_OPERATION_ADMISSION] = LOOM_RUNTIME_SERVICE_WAIT_CPU;
  } else if (service->phase == RECOGNITION_PHASE_RESERVING) {
    waits[RECOGNITION_OPERATION_ADMISSION] = LOOM_RUNTIME_SERVICE_WAIT_DEVICE;
  } else {
    waits[RECOGNITION_OPERATION_ADMISSION] = LOOM_RUNTIME_SERVICE_WAIT_CAPACITY;
  }
  if (service->phase == RECOGNITION_PHASE_OFFERING ||
      service->phase == RECOGNITION_PHASE_COLLECTING) {
    waits[RECOGNITION_OPERATION_BATCH] = LOOM_RUNTIME_SERVICE_WAIT_CPU;
  } else if (service->phase == RECOGNITION_PHASE_EXECUTING ||
             service->phase == RECOGNITION_PHASE_RELEASING) {
    waits[RECOGNITION_OPERATION_BATCH] = LOOM_RUNTIME_SERVICE_WAIT_DEVICE;
  }
  uint64_t pending_count = 0;
  for (iree_host_size_t i = 0; i < RECOGNITION_OPERATION_COUNT; ++i) {
    loom_runtime_service_operation_t* operation =
        &service->scheduling.operations[i];
    pending_count += operation->key.serial != 0;
    const uint64_t wait =
        operation->key.serial && iree_status_is_ok(service->failure) ? waits[i]
                                                                     : 0;
    if (operation->key.serial != service->scheduling.published_serials[i] ||
        operation->waits != wait) {
      operation->waits = wait;
      operation->devices = operation->key.serial ? devices : 0;
      operation->use_count = operation->key.serial ? 1 : 0;
      loom_runtime_service_publish_operation(control, operation, &use);
      service->scheduling.published_serials[i] = operation->key.serial;
    }
  }
  const loom_runtime_service_observation_t observation = {
      .owned_request_count = owned_count,
      .waiting_request_count = waiting_count,
      .offer_count = offer->choice_count,
      .pending_operation_count = pending_count,
      .offered_devices = offer->choice_count ? devices : 0,
      .accepted_devices = pending_count ? devices : 0};
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

void loom_serve_recognition_stream_service_stop(
    loom_serve_recognition_stream_service_t* service) {
  if (service->stopping) {
    return;
  }
  service->stopping = true;
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    loom_serve_recognition_stream_t* row = service->rows[i];
    if (row && !loom_serve_recognition_row_is_complete(row)) {
      row->cancelled = true;
    }
  }
  loom_serve_recognition_stream_publish(service, false);
}

iree_status_t loom_serve_recognition_stream_service_progress(
    loom_serve_recognition_stream_service_t* service,
    loom_runtime_service_flags_t flags, bool* out_progress) {
  *out_progress = false;
  if (iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING)) {
    loom_serve_recognition_stream_service_stop(service);
  }
  iree_status_t status = loom_serve_recognition_collect(service, out_progress);
  if (iree_status_is_ok(status) && service->phase == RECOGNITION_PHASE_IDLE) {
    status = loom_serve_recognition_release_finished(service, out_progress);
    // Publication ends this bounded turn before the next source CPU operation.
    if (iree_status_is_ok(status) && !service->stopping && !*out_progress &&
        service->phase == RECOGNITION_PHASE_IDLE && service->admission) {
      status = loom_serve_recognition_admit(
          service,
          iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_SHARED_ACTIVITY),
          out_progress);
    }
  }
  if (!iree_status_is_ok(status)) {
    service->stopping = true;
    service->failure =
        iree_status_join(service->failure, iree_status_clone(status));
    for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
      loom_serve_recognition_stream_t* row = service->rows[i];
      if (row) {
        row->cancelled = true;
        if (iree_status_is_ok(row->status)) {
          row->outcome = LOOM_SERVICE_TELEMETRY_FAILED;
        }
        row->status =
            iree_status_join(row->status, iree_status_clone(service->failure));
      }
    }
    *out_progress = true;
  }
  loom_serve_recognition_stream_publish(service, !*out_progress);
  return status;
}

iree_status_t loom_serve_recognition_stream_service_try_schedule(
    loom_serve_recognition_stream_service_t* service,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_flags_t flags,
    loom_runtime_service_operation_key_t* out_operation) {
  *out_operation = (loom_runtime_service_operation_key_t){0};
  if (service->stopping ||
      iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING) ||
      offer.ordinal != service->scheduling.offer.key.ordinal ||
      offer.revision != service->scheduling.offer.key.revision ||
      choice_ordinal >= service->scheduling.offer.choice_count) {
    return iree_ok_status();
  }
  // Input writes and event consumption can change the legal extent without
  // touching this service. Refresh on the owner before accepting the key.
  loom_serve_recognition_stream_publish(service, true);
  if (offer.revision != service->scheduling.offer.key.revision) {
    return iree_ok_status();
  }
  loom_serve_recognition_stream_t* queued = service->scheduling.queued;
  *out_operation = loom_serve_recognition_begin_operation(
      service,
      queued ? RECOGNITION_OPERATION_ADMISSION : RECOGNITION_OPERATION_BATCH,
      offer, service->scheduling.offer.work);
  if (queued) {
    queued->phase = RECOGNITION_ROW_WAITING;
    service->admission = queued;
    service->admit_next = false;
    loom_serve_recognition_schedule_cpu(service, RECOGNITION_PHASE_PREPARING);
  } else {
    for (iree_host_size_t i = 0; i < service->batch.candidate_count; ++i) {
      service->batch.candidates[i]->phase = RECOGNITION_ROW_OFFERING;
    }
    loom_serve_recognition_schedule_cpu(service, RECOGNITION_PHASE_OFFERING);
  }
  loom_serve_recognition_stream_publish(service, false);
  return iree_ok_status();
}

void loom_serve_recognition_stream_service_set_observer(
    loom_serve_recognition_stream_service_t* service,
    loom_runtime_wake_callback_t observer) {
  iree_slim_mutex_lock(&service->events.mutex);
  service->events.observer = observer;
  iree_slim_mutex_unlock(&service->events.mutex);
}

iree_status_t loom_serve_recognition_stream_service_try_open(
    loom_serve_recognition_stream_service_t* service,
    const loom_serve_recognition_request_t* request,
    loom_serve_recognition_completion_t completion,
    loom_serve_recognition_stream_t** out_stream) {
  *out_stream = NULL;
  if (service->stopping) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "recognition service is stopping");
  }
  if (!request->outputs || (request->outputs & ~service->generator.outputs)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recognition output selection is unsupported");
  }
  if (request->language.size > service->options.request_limit ||
      request->prompt.size >
          service->options.request_limit - request->language.size ||
      !isfinite(request->temperature) || request->temperature < 0.0f ||
      !iree_unicode_utf8_validate(request->language) ||
      !iree_unicode_utf8_validate(request->prompt)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recognition options require bounded UTF-8 "
                            "strings and finite nonnegative temperature");
  }
  iree_host_size_t index = service->generator.request_capacity;
  for (iree_host_size_t i = 0; i < service->generator.request_capacity; ++i) {
    if (!service->rows[i]) {
      index = i;
      break;
    }
  }
  if (index == service->generator.request_capacity) {
    return iree_ok_status();
  }
  loom_serve_recognition_stream_t* row = NULL;
  const iree_host_size_t request_size =
      request->language.size + request->prompt.size;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      service->host_allocator, sizeof(*row) + request_size, (void**)&row));
  row->host_allocator = service->host_allocator;
  row->format = service->generator.format;
  row->input_capacity = service->generator.input_capacity;
  row->frame_limit = service->generator.frame_limit;
  row->index = index;
  iree_status_t status = loom_serve_recognition_input_initialize(
      service->generator.input_capacity, service->generator.format.channels,
      service->host_allocator, &row->input);
  if (iree_status_is_ok(status)) {
    status = loom_serve_recognition_events_allocate(
        service->generator.event_capacity, service->generator.event_bytes,
        service->host_allocator, &row->output);
  }
  if (iree_status_is_ok(status)) {
    row->request = *request;
    char* storage = (char*)(row + 1);
    if (request->language.size) {
      memcpy(storage, request->language.data, request->language.size);
      row->request.language =
          iree_make_string_view(storage, request->language.size);
      storage += request->language.size;
    }
    if (request->prompt.size) {
      memcpy(storage, request->prompt.data, request->prompt.size);
      row->request.prompt =
          iree_make_string_view(storage, request->prompt.size);
    }
    loom_serve_recognition_document_initialize(service->options.document_limit,
                                               service->host_allocator,
                                               &row->document);
    row->arrival = ++service->next_arrival;
    row->caller_open = true;
    row->service_open = true;
    row->completion = completion;
    row->phase = RECOGNITION_ROW_QUEUED;
    service->rows[index] = row;
    *out_stream = row;
    loom_serve_recognition_stream_publish(service, true);
  } else {
    loom_serve_recognition_input_deinitialize(&row->input);
    iree_allocator_free(service->host_allocator, row);
  }
  return status;
}

loom_serve_recognition_format_t loom_serve_recognition_stream_format(
    const loom_serve_recognition_stream_t* stream) {
  return stream->format;
}

iree_host_size_t loom_serve_recognition_stream_input_capacity(
    const loom_serve_recognition_stream_t* stream) {
  return stream->input_capacity;
}

iree_status_t loom_serve_recognition_stream_write(
    loom_serve_recognition_stream_t* stream, iree_const_byte_span_t samples,
    iree_host_size_t* out_frame_count) {
  *out_frame_count = 0;
  if (!stream->service_open || stream->cancelled || stream->input.complete ||
      loom_serve_recognition_row_is_complete(stream)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "recognition input is already complete");
  }
  if (samples.data_length % stream->input.frame_bytes ||
      samples.data_length / stream->input.frame_bytes >
          stream->frame_limit - stream->input.received) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PCM must contain whole frames within the "
                            "recognition input limit");
  }
  for (iree_host_size_t i = 0; i < samples.data_length; i += sizeof(float)) {
    if (!isfinite(iree_unaligned_load_le_f32(samples.data + i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "recognition PCM contains a nonfinite sample");
    }
  }
  *out_frame_count =
      loom_serve_recognition_input_append(&stream->input, samples);
  return iree_ok_status();
}

iree_host_size_t loom_serve_recognition_stream_credit(
    const loom_serve_recognition_stream_t* stream) {
  if (!stream->service_open || stream->cancelled || stream->input.complete ||
      loom_serve_recognition_row_is_complete(stream)) {
    return 0;
  }
  return (iree_host_size_t)iree_min(
      (uint64_t)loom_serve_recognition_input_credit(&stream->input),
      stream->frame_limit - stream->input.received);
}

void loom_serve_recognition_stream_finish(
    loom_serve_recognition_stream_t* stream) {
  stream->input.complete = true;
}

const loom_serve_recognition_event_t* loom_serve_recognition_stream_peek(
    const loom_serve_recognition_stream_t* stream) {
  return stream->phase == RECOGNITION_ROW_PUBLISHING && !stream->cancelled &&
                 iree_status_is_ok(stream->status)
             ? loom_serve_recognition_events_at(
                   stream->output,
                   loom_serve_recognition_events_count(stream->output) -
                       stream->pending_count)
             : NULL;
}

void loom_serve_recognition_stream_consume(
    loom_serve_recognition_stream_t* stream) {
  if (--stream->pending_count == 0) {
    stream->phase = stream->result.complete ? RECOGNITION_ROW_COMPLETE
                                            : RECOGNITION_ROW_READY;
  }
}

bool loom_serve_recognition_stream_is_complete(
    const loom_serve_recognition_stream_t* stream) {
  return !stream->service_open && !stream->pending_count;
}

bool loom_serve_recognition_stream_is_waiting(
    const loom_serve_recognition_stream_t* stream) {
  if (!stream->service_open || !stream->caller_open || stream->cancelled) {
    return false;
  }
  return stream->phase == RECOGNITION_ROW_QUEUED ||
         stream->phase == RECOGNITION_ROW_READY ||
         stream->phase == RECOGNITION_ROW_AWAIT_INPUT ||
         (stream->phase == RECOGNITION_ROW_PUBLISHING &&
          !stream->result.complete);
}

iree_status_t loom_serve_recognition_stream_status(
    const loom_serve_recognition_stream_t* stream) {
  return iree_status_clone(stream->status);
}

const loom_serve_recognition_document_t* loom_serve_recognition_stream_document(
    const loom_serve_recognition_stream_t* stream) {
  return iree_any_bit_set(stream->request.outputs,
                          LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT) &&
                 loom_serve_recognition_stream_is_complete(stream) &&
                 iree_status_is_ok(stream->status)
             ? &stream->document
             : NULL;
}

void loom_serve_recognition_stream_close(
    loom_serve_recognition_stream_t* stream) {
  stream->caller_open = false;
  stream->cancelled = true;
  stream->pending_count = 0;
  loom_serve_recognition_stream_release(stream);
}

void loom_serve_recognition_stream_service_free(
    loom_serve_recognition_stream_service_t* service) {
  if (!service) {
    return;
  }
  IREE_ASSERT(!loom_serve_recognition_stream_service_is_pending(service));
  for (iree_host_size_t i = 0;
       service->rows && i < service->generator.request_capacity; ++i) {
    IREE_ASSERT(!service->rows[i]);
  }
  const iree_allocator_t host_allocator = service->host_allocator;
  iree_status_free(service->failure);
  iree_slim_mutex_deinitialize(&service->events.mutex);
  iree_allocator_free(host_allocator, service->batch.candidates);
  iree_allocator_free(host_allocator, service->batch.rows);
  iree_allocator_free(host_allocator, service->batch.invocations);
  iree_allocator_free(host_allocator, service->rows);
  iree_allocator_free_aligned(host_allocator, service);
}

iree_status_t loom_serve_recognition_stream_service_allocate(
    loom_runtime_host_t* host, loom_serve_recognition_generator_t generator,
    const loom_serve_recognition_stream_service_options_t* options,
    loom_runtime_service_control_binding_t scheduling,
    iree_allocator_t host_allocator,
    loom_serve_recognition_stream_service_t** out_service) {
  *out_service = NULL;
  if (!generator.prepare || !generator.discard || !generator.try_admit ||
      !generator.complete_admit || !generator.prepare_work ||
      !generator.prepare_batch || !generator.submit || !generator.collect ||
      !generator.release_batch || !generator.release ||
      generator.device_ordinal >= 64 || !generator.batch_units ||
      !generator.request_capacity || !generator.input_capacity ||
      !generator.frame_limit || !generator.event_capacity ||
      !generator.event_bytes || !generator.outputs ||
      (generator.outputs & ~LOOM_SERVE_RECOGNITION_OUTPUT_ALL) ||
      !generator.format.sample_rate || !generator.format.channels ||
      !options->request_limit || !options->document_limit ||
      options->request_limit >
          SIZE_MAX - sizeof(loom_serve_recognition_stream_t) ||
      generator.request_capacity >
          SIZE_MAX / sizeof(loom_serve_recognition_stream_t*) ||
      generator.input_capacity >
          SIZE_MAX / sizeof(float) / generator.format.channels) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recognition service requires a complete "
                            "generator and bounded input/output geometry");
  }
  if (!scheduling.control || !scheduling.publish_observation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recognition requires retained scheduling storage");
  }
  const loom_runtime_service_control_layout_t layout =
      loom_runtime_service_control_layout(scheduling.control);
  if (scheduling.offer_ordinal >= layout.offer_capacity ||
      scheduling.operation_ordinal > layout.operation_capacity ||
      layout.operation_capacity - scheduling.operation_ordinal <
          RECOGNITION_OPERATION_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recognition scheduling needs one offer and two "
                            "operation slots");
  }
  loom_serve_recognition_stream_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*service),
      iree_alignof(loom_serve_recognition_stream_service_t), 0,
      (void**)&service));
  service->host_allocator = host_allocator;
  service->generator = generator;
  service->options = *options;
  service->scheduling.binding = scheduling;
  service->scheduling.offer.key.ordinal = scheduling.offer_ordinal;
  for (iree_host_size_t i = 0; i < RECOGNITION_OPERATION_COUNT; ++i) {
    service->scheduling.operations[i].key.ordinal =
        scheduling.operation_ordinal + i;
  }
  service->executor = loom_runtime_host_executor(host);
  service->admit_next = true;
  iree_slim_mutex_initialize(&service->events.mutex);
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, generator.request_capacity, sizeof(*service->rows),
      (void**)&service->rows);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, generator.request_capacity,
        sizeof(*service->batch.candidates), (void**)&service->batch.candidates);
  }
  if (iree_status_is_ok(status)) {
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
    status = iree_async_proactor_pool_get(loom_runtime_host_proactor_pool(host),
                                          0, &service->proactor);
  }
  if (iree_status_is_ok(status)) {
    *out_service = service;
  } else {
    loom_serve_recognition_stream_service_free(service);
  }
  return status;
}
