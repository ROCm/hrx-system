// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service.h"

#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/mutex.h"
#include "loom/serve/media/image_codec.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/service/execution_reconfiguration.h"
#include "loom/serve/service/kind.h"
#include "loom/serve/service/text/admission.h"
#include "loom/serve/service/text/chat.h"
#include "loom/serve/service/text/checkpoint_http.h"
#include "loom/serve/service/text/checkpoint_registry.h"
#include "loom/serve/service/text/checkpoint_request.h"
#include "loom/serve/service/text/consumer.h"
#include "loom/serve/service/text/epoch.h"
#include "loom/serve/service/text/heartbeat.h"
#include "loom/serve/service/text/http.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/service/text/input_count.h"
#include "loom/serve/service/text/media_projector.h"
#include "loom/serve/service/text/packing.h"
#include "loom/serve/service/text/pending_queue.h"
#include "loom/serve/service/text/pending_request.h"
#include "loom/serve/service/text/prefix_cache.h"
#include "loom/serve/service/text/readiness.h"
#include "loom/serve/service/text/reconfiguration.h"
#include "loom/serve/service/text/schedule.h"
#include "loom/serve/service/text/session.h"
#include "loom/serve/service/text/session_cache.h"
#include "loom/serve/service/text/telemetry.h"

// Device work retained until readiness is consumed by the model owner.
typedef enum loom_serve_text_work_e {
  TEXT_WORK_NONE,
  TEXT_WORK_EPOCH,
  TEXT_WORK_CHECKPOINT,
  TEXT_WORK_TRIM,
  TEXT_WORK_RESERVE,
  TEXT_WORK_RETAIN,
  TEXT_WORK_PREFIX,
} loom_serve_text_work_t;

// Exclusive host/VM borrow; input preparation may overlap an accepted epoch.
typedef enum loom_serve_text_cpu_work_e {
  TEXT_CPU_WORK_NONE,
  TEXT_CPU_WORK_PREPARE,
  TEXT_CPU_WORK_OUTPUT,
  TEXT_CPU_WORK_INPUT_COUNT,
} loom_serve_text_cpu_work_t;

struct loom_serve_text_service_t {
  // Allocator for the service's host-only storage.
  iree_allocator_t host_allocator;
  // Borrowed model residency and its single host execution owner.
  loom_serve_text_model_t* model;
  // Borrowed optional image computation with independent preparation/residency.
  loom_serve_finite_component_t* image;
  // Lazy independent projection owner; text-only requests allocate no window.
  loom_serve_text_media_projector_t* projector;
  // Last receipt in the current finite lookahead window; later arrivals do not
  // postpone a gathered cohort. Zero means no lookahead is pending.
  uint64_t media_preparation_end;
  // Native choice pool and text policy, joined before model retirement.
  loom_serve_execution_reconfiguration_t* reconfiguration;
  // Borrowed deployment name, independent of the source policy identity.
  iree_string_view_t name;
  // Owned fallback shape for isolated execution.
  loom_serve_packing_shape_t isolated_shape;
  // Number of allocated device/session rows.
  iree_host_size_t row_count;
  // Output bound when the client omits one.
  iree_host_size_t default_max_tokens;
  // Immutable model epoch geometry and execution choices.
  loom_serve_text_epoch_options_t scheduling;
  // Rotating admission priority, independent of row or request identity.
  iree_host_size_t cursor;
  // Irreversible admission seal after stop or a terminal service failure.
  bool stopping;
  // Proactor-timed reporting of copied state, independent of model readiness.
  loom_serve_text_heartbeat_t heartbeat;
  // Independent CPU/device publication; only the owner mutates model state.
  loom_serve_text_readiness_t readiness;
  // Owner's device operation, retained until readiness is consumed.
  struct {
    // Owner-only operation discriminator; NONE means no callback is accepted.
    loom_serve_text_work_t work;
    // Exact work and feedback values captured by submission.
    loom_serve_execution_frontier_t frontier;
    // Host observation before registering this operation's readiness wait.
    iree_time_t start_time;
  } completion;
  // Exclusive host/VM borrow, cleared only after task process release. Input
  // preparation selects among idle rows and may overlap device epoch work on
  // active rows; the owner leaves all idle completions unchanged until release.
  loom_serve_text_cpu_work_t cpu_work;
  // Retained cohort inputs and feedback through response CPU release.
  loom_serve_text_epoch_t epoch;
  // One delivery receipt retaining validated names through device readiness.
  loom_serve_text_checkpoint_operation_t control;
  // Stateless CPU input queries, independent of generation rows and capacity.
  loom_serve_text_input_count_t input_counts;
  // Single-owner observation; CPU jobs never publish into this stream.
  loom_serve_text_telemetry_t telemetry;
  // Stable queued claims and selected-attempt bookkeeping, not device rows.
  loom_serve_text_pending_queue_t pending;
  // Request policy over the logical pool, not a second block allocator.
  struct {
    // Immutable logical geometry; availability is queried at observation.
    loom_serve_text_pool_usage_t geometry;
    // One maintenance cut, retained through model readiness.
    struct {
      // Retired rows released storage since the last completed cut.
      bool pending;
      // Host observation time before submitting maintenance.
      iree_time_t start_time;
      // Native relocation plan consumed only after successful readiness.
      loom_serve_text_trim_result_t result;
    } trim;
  } pool;
  // Named native endpoints and their immutable host continuation state.
  loom_serve_text_checkpoint_registry_t checkpoints;
  // Evictable idle histories, independent of execution rows and explicit pins.
  loom_serve_text_session_cache_t cache;
  // Shared consumed-token frontiers, independent of completed chat histories.
  loom_serve_text_prefix_cache_t prefixes;
  // Shared CPU encoding scratch, copied into consumer-owned staged output.
  iree_string_builder_t scratch;
  // Shared validated tool records used after complete source parsing.
  loom_serve_text_tool_calls_t tool_calls;
  // Completion parser/formatter backing reused by all serialized output jobs.
  loom_serve_text_chat_scratch_t completion_scratch;
  // Fixed retained rows sharing one model, command set and execution timeline.
  loom_serve_text_session_t sessions[LOOM_SERVE_TEXT_ROW_CAPACITY];
};

static bool loom_serve_text_has_work(const loom_serve_text_service_t* service) {
  return service->completion.work != TEXT_WORK_NONE ||
         service->cpu_work != TEXT_CPU_WORK_NONE;
}

static iree_status_t loom_serve_text_await_work(
    loom_serve_text_service_t* service, loom_serve_text_work_t work) {
  service->completion.start_time = iree_time_now();
  IREE_RETURN_IF_ERROR(loom_serve_text_readiness_await(
      &service->readiness, &service->completion.frontier));
  service->completion.work = work;
  return iree_ok_status();
}

static void loom_serve_text_schedule_cpu(loom_serve_text_service_t* service,
                                         loom_serve_text_cpu_work_t work,
                                         loom_serve_text_cpu_work_fn_t run) {
  service->cpu_work = work;
  loom_serve_text_readiness_schedule(&service->readiness, run, service);
}

static iree_status_t loom_serve_text_count_input(void* user_data) {
  loom_serve_text_service_t* service = user_data;
  return loom_serve_text_input_count_run(
      &service->input_counts, loom_serve_text_model_chat_policy(service->model),
      service->name, service->default_max_tokens,
      service->scheduling.context_capacity);
}

// Ready inference has first use of the owner turn; a query can borrow the
// source VM after submission or when no generation can currently progress.
static bool loom_serve_text_begin_input_count(
    loom_serve_text_service_t* service) {
  if (!service->input_counts.count || service->cpu_work != TEXT_CPU_WORK_NONE ||
      (service->completion.work != TEXT_WORK_NONE &&
       service->completion.work != TEXT_WORK_EPOCH)) {
    return false;
  }
  loom_serve_text_schedule_cpu(service, TEXT_CPU_WORK_INPUT_COUNT,
                               loom_serve_text_count_input);
  return true;
}

static void loom_serve_text_observe(loom_serve_text_service_t* service,
                                    const char* phase) {
  const loom_serve_text_pool_usage_t pool =
      loom_serve_text_model_pool_usage(service->model);
  iree_host_size_t active = 0, prefill = 0, decode = 0, backpressured = 0;
  iree_time_t oldest_active_arrival = IREE_TIME_INFINITE_FUTURE;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    const loom_serve_text_session_t* session = &service->sessions[i];
    if (session->request.consumer.self) {
      ++active;
      const iree_time_t arrival =
          session->request.start_time - session->request.queue_duration;
      oldest_active_arrival = iree_min(oldest_active_arrival, arrival);
      prefill += session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL;
      decode += session->request.phase == LOOM_SERVE_TEXT_REQUEST_DECODE;
      backpressured += session->request.consumer.vtable->is_blocked(
          session->request.consumer.self);
    }
  }
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  loom_serve_text_heartbeat_snapshot_t* state = &service->heartbeat.snapshot;
  if (strcmp(state->phase, phase)) {
    state->phase = phase;
    state->phase_start = iree_time_now();
  }
  state->active_rows = active;
  state->oldest_active_arrival = oldest_active_arrival;
  state->pool.capacity = pool.capacity;
  state->pool.reserved = pool.pending;
  state->pool.resident = pool.capacity - pool.available - pool.pending;
  state->session_cache.capacity = service->cache.capacity;
  state->session_cache.endpoints = service->cache.count;
  state->prefix_cache.capacity = service->prefixes.capacity;
  state->prefix_cache.endpoints = service->prefixes.count;
  state->checkpoint_host_bytes =
      loom_serve_text_model_checkpoint_suspended_bytes(service->model);
  state->state_memory = loom_serve_text_model_memory_statistics(service->model);
  state->weight_memory =
      loom_serve_text_model_weight_statistics(service->model);
  state->weight_activity =
      loom_serve_text_model_weight_activity(service->model);
  state->device_memory = loom_serve_device_memory_statistics(
      loom_serve_text_model_device(service->model));
  state->prefill_rows = prefill;
  state->decode_rows = decode;
  state->backpressured_rows = backpressured;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
}

static void loom_serve_text_request_release(
    loom_serve_text_service_t* service, loom_serve_text_session_t* session) {
  service->pool.trim.pending = true;
  loom_serve_text_session_release_request(session);
}

static void loom_serve_text_request_cancel(loom_serve_text_service_t* service,
                                           loom_serve_text_session_t* session) {
  loom_serve_text_telemetry_request_end(
      &service->telemetry, session,
      service->telemetry.failure_code || session->request.diagnostic.code
          ? LOOM_SERVE_TELEMETRY_FAILED
          : LOOM_SERVE_TELEMETRY_CANCELLED);
  const loom_serve_text_consumer_t consumer = session->request.consumer;
  const iree_status_code_t code = service->telemetry.failure_code
                                      ? service->telemetry.failure_code
                                      : session->request.diagnostic.code;
  loom_serve_text_chat_completion_deinitialize(&session->completion);
  loom_serve_text_request_release(service, session);
  consumer.vtable->end(consumer.self, false,
                       code ? code : IREE_STATUS_CANCELLED);
}

static void loom_serve_text_request_finish(loom_serve_text_service_t* service,
                                           loom_serve_text_session_t* session) {
  loom_serve_text_telemetry_request_end(&service->telemetry, session,
                                        session->request.diagnostic.code
                                            ? LOOM_SERVE_TELEMETRY_FAILED
                                            : LOOM_SERVE_TELEMETRY_COMPLETED);
  const loom_serve_text_consumer_t consumer = session->request.consumer;
  const iree_status_code_t code = session->request.diagnostic.code;
  loom_serve_text_request_release(service, session);
  consumer.vtable->end(consumer.self, true, code);
}

// The admission owner resolves information the checkpoint handler cannot know:
// live rows, detached history and generation requests borrowing the same name.
static iree_status_t loom_serve_text_submit_checkpoint(
    loom_serve_text_service_t* service) {
  const iree_string_view_t session_name = service->control.request.session;
  loom_serve_text_checkpoint_source_t source = {0};
  if (service->control.request.action ==
      LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN) {
    source.session = loom_serve_text_session_select(
        service->row_count, service->sessions, session_name, NULL);
    if (source.session &&
        !iree_string_view_equal(
            session_name,
            iree_make_cstring_view(source.session->lifetime.name))) {
      source.session = NULL;
    }
    const loom_serve_text_session_cache_entry_t* cached =
        loom_serve_text_session_cache_find(&service->cache, session_name);
    source.cached = cached ? &cached->endpoint : NULL;
    source.busy = source.session && source.session->request.consumer.self;
    for (iree_host_size_t i = 0; i < service->pending.count; ++i) {
      source.busy |= iree_string_view_equal(session_name,
                                            service->pending.values[i]->name);
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_request_submit(
      &service->control, &service->checkpoints, source,
      service->control.request.action ==
              LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND &&
          loom_serve_text_model_memory_statistics(service->model)
                  .reserved_bytes != 0,
      &service->telemetry, &service->scratch, &service->pool.trim.pending,
      &service->completion.frontier));
  if (!service->control.checkpoint) {
    return iree_ok_status();
  }
  if (service->control.request.action ==
      LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND) {
    loom_serve_text_observe(service, "checkpoint_suspend");
  }
  return loom_serve_text_await_work(service, TEXT_WORK_CHECKPOINT);
}

// Queue ownership changes independently of model phases and CPU/device loans.
// Publication reads only owner-managed entries, never borrowed row/VM state.
static void loom_serve_text_pending_observe(
    loom_serve_text_service_t* service) {
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  service->heartbeat.snapshot.queued_requests =
      service->pending.count + service->input_counts.count;
  service->heartbeat.snapshot.oldest_queued_arrival =
      service->pending.count ? service->pending.values[0]->receipt.arrival_time
                             : 0;
  if (service->input_counts.count) {
    const iree_time_t arrival =
        service->input_counts.requests[0].receipt.arrival_time;
    service->heartbeat.snapshot.oldest_queued_arrival =
        service->pending.count
            ? iree_min(service->heartbeat.snapshot.oldest_queued_arrival,
                       arrival)
            : arrival;
  }
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
}

static void loom_serve_text_pending_remove(loom_serve_text_service_t* service,
                                           iree_host_size_t index) {
  loom_serve_text_pending_queue_remove(&service->pending, index);
  loom_serve_text_pending_observe(service);
}

static void loom_serve_text_pending_cancel(loom_serve_text_service_t* service,
                                           iree_host_size_t index) {
  loom_serve_text_pending_queue_cancel(&service->pending, index,
                                       service->projector);
  loom_serve_text_pending_observe(service);
}

static void loom_serve_text_pending_cancel_failed(
    loom_serve_text_service_t* service, bool* out_progress) {
  const bool candidate_borrowed =
      service->cpu_work == TEXT_CPU_WORK_PREPARE ||
      service->completion.work == TEXT_WORK_RESERVE ||
      service->completion.work == TEXT_WORK_RETAIN;
  if (loom_serve_text_pending_queue_cancel_failed(
          &service->pending,
          candidate_borrowed ? service->pending.prepared.request : NULL,
          service->projector)) {
    *out_progress = true;
    loom_serve_text_pending_observe(service);
  }
}

// Reclaim completed ownership before trying the next physical reservation.
// Compaction itself admits its temporary destinations and may leave a sparse
// map when moving would exceed the shared physical budget.
static iree_status_t loom_serve_text_submit_trim(
    loom_serve_text_service_t* service) {
  if (!service->pool.trim.pending) {
    return iree_ok_status();
  }
  loom_serve_text_observe(service, "trim");
  service->pool.trim.start_time = iree_time_now();
  IREE_RETURN_IF_ERROR(loom_serve_text_model_submit_trim(
      service->model, &service->pool.trim.result,
      &service->completion.frontier));
  return loom_serve_text_await_work(service, TEXT_WORK_TRIM);
}

static iree_status_t loom_serve_text_complete_trim(
    loom_serve_text_service_t* service) {
  loom_serve_text_trim_result_t* result = &service->pool.trim.result;
  bool complete = false;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_complete_trim(
      service->model, result, &complete, &service->completion.frontier));
  if (!complete) {
    return loom_serve_text_await_work(service, TEXT_WORK_TRIM);
  }
  service->pool.trim.pending = false;
  const loom_serve_memory_statistics_t memory =
      loom_serve_text_model_memory_statistics(service->model);
  const loom_serve_text_trim_record_t value = {
      .duration_ns = iree_time_now() - service->pool.trim.start_time,
      .moved_blocks = result->moved_blocks,
      .copied_bytes = result->copied_bytes,
      .released_bytes = result->released_bytes,
      .committed_bytes = memory.committed_bytes};
  loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_TRIM, 0,
                         &value, sizeof(value));
  return iree_ok_status();
}

static void loom_serve_text_reset_session(loom_serve_text_service_t* service,
                                          loom_serve_text_session_t* session) {
  loom_serve_text_telemetry_session_end(&service->telemetry,
                                        &session->lifetime);
  loom_serve_text_row_reset_retired(session->row);
  service->pool.trim.pending = true;
  loom_serve_text_chat_completion_deinitialize(&session->completion);
}

static bool loom_serve_text_cache_reclaim_pending(void* self) {
  const loom_serve_text_service_t* service = self;
  return service->pool.trim.pending;
}

static bool loom_serve_text_cache_reclaim_request(void* self) {
  loom_serve_text_service_t* service = self;
  if (loom_serve_text_has_work(service) || service->control.consumer.self) {
    return false;
  }
  if (!loom_serve_text_prefix_cache_evict(&service->prefixes,
                                          service->pending.prepared.prefix) &&
      !loom_serve_text_session_cache_evict(
          &service->cache, service->row_count, service->sessions,
          service->pending.prepared.session, service->pending.prepared.cached,
          &service->telemetry)) {
    return false;
  }
  service->pool.trim.pending = true;
  loom_serve_text_readiness_notify(&service->readiness);
  return true;
}

typedef enum loom_serve_text_admission_result_e {
  // Model parameters cannot fit; changing requests cannot relieve this limit.
  TEXT_ADMISSION_CAPACITY = 0,
  // Request-specific logical/physical state cannot fit; another request may.
  TEXT_ADMISSION_STATE_CAPACITY,
  // An older weight-pressure waiter must receive its next owner turn.
  TEXT_ADMISSION_QUEUED,
  // Parameter activation and state reservation have both completed.
  TEXT_ADMISSION_READY,
} loom_serve_text_admission_result_t;

// Native ownership knows the source geometry and exact physical slab union;
// the service only decides which idle histories may yield. Selected history,
// named endpoints and active reservations remain protected on every denial.
static iree_status_t loom_serve_text_reclaim_idle(
    loom_serve_text_service_t* service, loom_serve_text_session_t* selected,
    const loom_serve_text_checkpoint_t* checkpoint, iree_host_size_t retained,
    iree_host_size_t reservation,
    loom_serve_text_admission_result_t* out_result) {
  *out_result = TEXT_ADMISSION_CAPACITY;
  const loom_serve_text_reserve_mode_t mode =
      checkpoint ? LOOM_SERVE_TEXT_RESERVE_CHECKPOINT
      : retained ? LOOM_SERVE_TEXT_RESERVE_CONTINUE
                 : LOOM_SERVE_TEXT_RESERVE_FRESH;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    status = loom_serve_text_submit_trim(service);
    if (!iree_status_is_ok(status) ||
        service->completion.work != TEXT_WORK_NONE) {
      break;
    }
    loom_serve_residency_admission_t admission;
    loom_serve_execution_frontier_t frontier;
    status = loom_serve_text_row_submit_reserve(
        selected->row, mode, checkpoint, reservation, &admission, &frontier);
    if (iree_status_is_ok(status) &&
        admission != LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
      // Parameter admission owns shared pressure and idle-cache reclamation.
      // Its denial is not a row-state shortage: local eviction cannot override
      // a decision to wait for pinned parameters or another admission owner.
      *out_result = admission == LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED
                        ? TEXT_ADMISSION_QUEUED
                        : TEXT_ADMISSION_CAPACITY;
      break;
    }
    if (iree_status_is_ok(status) &&
        admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
      // The previous frontier was consumed successfully before this call. An
      // unchanged pair needs no new wait; warm denial must not wake itself.
      const loom_serve_execution_frontier_t previous =
          service->completion.frontier;
      const bool complete = previous.semaphores[0] == frontier.semaphores[0] &&
                            previous.semaphores[1] == frontier.semaphores[1] &&
                            previous.values[0] == frontier.values[0] &&
                            previous.values[1] == frontier.values[1];
      service->completion.frontier = frontier;
      if (complete) {
        if (loom_serve_text_row_complete_reserve(selected->row)) {
          *out_result = TEXT_ADMISSION_READY;
        } else {
          *out_result = TEXT_ADMISSION_STATE_CAPACITY;
        }
      } else {
        status = loom_serve_text_await_work(service, TEXT_WORK_RESERVE);
        break;
      }
    }
    if (!iree_status_is_ok(status) || *out_result == TEXT_ADMISSION_READY) {
      break;
    }
    if (loom_serve_residency_cache_is_reclaiming(loom_serve_residency_cache(
            loom_serve_text_model_residency(service->model)))) {
      *out_result = TEXT_ADMISSION_QUEUED;
      break;
    }
    if (!loom_serve_text_prefix_cache_evict(&service->prefixes,
                                            service->pending.prepared.prefix) &&
        !loom_serve_text_session_cache_evict(
            &service->cache, service->row_count, service->sessions, selected,
            service->pending.prepared.cached, &service->telemetry)) {
      break;
    }
    service->pool.trim.pending = true;
  }
  return status;
}

static iree_status_t loom_serve_text_prepare_admission(void* user_data) {
  loom_serve_text_service_t* service = user_data;
  return loom_serve_text_pending_queue_prepare(
      &service->pending, service->name, service->default_max_tokens,
      service->row_count, service->sessions, &service->pool.geometry);
}

// The owner lends idle rows and the selected request's host storage. A running
// epoch owns active rows and independent transfer buffers; its completion
// remains unconsumed until this job releases the shared model invocation.
static bool loom_serve_text_begin_preparation_at(
    loom_serve_text_service_t* service, iree_host_size_t index) {
  if (!service->pending.count || service->pending.prepared.session) {
    return false;
  }
  loom_serve_text_pending_request_t* pending = service->pending.values[index];
  if (loom_serve_text_media_projector_borrows(service->projector,
                                              pending->media)) {
    return false;
  }
  loom_serve_text_session_t* session = loom_serve_text_session_select(
      service->row_count, service->sessions, pending->name,
      pending->chat.policy && !pending->checkpoint ? &pending->chat : NULL);
  if (!session) {
    return false;
  }
  service->pending.prepared.request = pending;
  service->pending.prepared.index = index;
  service->pending.prepared.session = session;
  service->pending.prepared.cached =
      loom_serve_text_session_cache_find(&service->cache, pending->name);
  if (loom_serve_text_input_is_current(
          &pending->input, &pending->chat,
          loom_serve_text_pending_queue_prefix(&service->pending, session))) {
    return true;
  }
  if (service->completion.work == TEXT_WORK_NONE) {
    loom_serve_text_observe(service, "prepare");
  }
  loom_serve_text_schedule_cpu(service, TEXT_CPU_WORK_PREPARE,
                               loom_serve_text_prepare_admission);
  return true;
}

static bool loom_serve_text_begin_preparation(
    loom_serve_text_service_t* service) {
  return loom_serve_text_begin_preparation_at(
      service, service->pending.admission.cursor);
}

// Resolves prefix elision before starting the independent image component.
// Image preparation owns no language parameter or completion-capacity pin.
static iree_status_t loom_serve_text_prepare_media(
    loom_serve_text_service_t* service, bool* out_ready) {
  *out_ready = true;
  loom_serve_text_pending_request_t* pending =
      service->pending.prepared.request;
  if (!pending->input.retained_count && !pending->checkpoint) {
    service->pending.prepared.prefix = loom_serve_text_prefix_cache_find(
        &service->prefixes, &pending->chat, pending->input.count,
        pending->input.tokens);
  }
  const iree_host_size_t consumed =
      service->pending.prepared.prefix ? service->pending.prepared.prefix->count
                                       : 0;
  if (!pending->media) {
    bool needed = false;
    for (iree_host_size_t i = 0; i < pending->input.media_span_count; ++i) {
      const loom_serve_text_media_span_t* span = &pending->input.media_spans[i];
      needed |=
          (iree_host_size_t)span->input_offset + span->input_count > consumed;
    }
    if (!needed) {
      return iree_ok_status();
    }
    if (!service->image) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "model has no image component");
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_media_request_allocate(
        &pending->chat, service->host_allocator, &pending->media));
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_text_media_request_take_status(pending->media));
  *out_ready = loom_serve_text_media_request_bind(pending->media,
                                                  &pending->input, consumed);
  if (*out_ready) {
    return iree_ok_status();
  }
  if (!service->media_preparation_end &&
      loom_serve_text_media_projector_is_collecting(service->projector)) {
    service->media_preparation_end =
        service->pending.values[service->pending.count - 1]->receipt.serial;
  }
  if (!service->projector) {
    loom_serve_finite_model_t* model = NULL;
    iree_status_t status =
        loom_serve_finite_component_prepare(service->image, &model);
    if (!iree_status_is_ok(status)) {
      // Registration/source preparation is service availability, not a defect
      // in this HTTP payload. Preserve the component's original diagnostic.
      return iree_status_join(
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "image component is unavailable"),
          status);
    }
    if (!model) {
      return iree_ok_status();
    }
    iree_host_size_t capacity = 0;
    if (!iree_host_size_checked_mul(service->pending.capacity,
                                    pending->chat.policy->images.maximum_count,
                                    &capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "image queue extent overflow");
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_media_projector_allocate(
        model, capacity, service->readiness.cpu.executor,
        service->readiness.device.proactor, service->readiness.observer,
        service->host_allocator, &service->projector));
  }
  loom_serve_text_media_projector_offer(service->projector, pending->media);
  return loom_serve_text_media_request_take_status(pending->media);
}

static iree_status_t loom_serve_text_reject_prepared(
    loom_serve_text_service_t* service, iree_status_t status) {
  loom_serve_text_pending_request_t* pending =
      service->pending.prepared.request;
  const iree_host_size_t index = service->pending.prepared.index;
  loom_serve_text_media_projector_cancel(service->projector, pending->media);
  status = loom_serve_text_pending_request_reject(
      pending, &service->telemetry, status, service->host_allocator);
  loom_serve_text_pending_remove(service, index);
  return status;
}

static iree_status_t loom_serve_text_complete_preparation(
    loom_serve_text_service_t* service, loom_serve_service_flags_t flags,
    iree_status_t status) {
  loom_serve_text_pending_request_t* pending =
      service->pending.prepared.request;
  const iree_host_size_t index = service->pending.prepared.index;
  const bool cancelled =
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING) ||
      !pending->consumer.vtable->is_open(pending->consumer.self);
  if (!iree_status_is_ok(status)) {
    switch (iree_status_code(status)) {
      case IREE_STATUS_INVALID_ARGUMENT:
      case IREE_STATUS_OUT_OF_RANGE:
      case IREE_STATUS_UNIMPLEMENTED:
      case IREE_STATUS_RESOURCE_EXHAUSTED:
        break;
      default:
        return status;
    }
    if (cancelled) {
      loom_serve_text_telemetry_diagnose(&service->telemetry,
                                         pending->receipt.serial, status);
      loom_serve_text_pending_cancel(service, index);
      return iree_ok_status();
    }
    return loom_serve_text_reject_prepared(service, status);
  }
  if (cancelled) {
    loom_serve_text_pending_cancel(service, index);
  } else {
    service->pending.prepared.session =
        service->pending.prepared.resolved_session;
    if (index != service->pending.admission.cursor) {
      bool media_ready = false;
      status = loom_serve_text_prepare_media(service, &media_ready);
      if (!iree_status_is_ok(status)) {
        return loom_serve_text_reject_prepared(service, status);
      }
    }
    const loom_serve_text_admission_record_t value = {
        .row = service->pending.prepared.session - service->sessions,
        .retained_tokens = pending->input.retained_count,
        .appended_tokens = pending->input.count,
        .reservation_tokens = pending->reservation,
        .max_tokens = pending->chat.max_tokens,
        .prepare_ns = pending->prepare_duration};
    loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_PREPARED,
                           pending->receipt.serial, &value, sizeof(value));
    if (index != service->pending.admission.cursor) {
      memset(&service->pending.prepared, 0, sizeof(service->pending.prepared));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_admit_request(
    loom_serve_text_service_t* service, loom_serve_service_flags_t flags) {
  loom_serve_text_pending_request_t* pending =
      service->pending.prepared.request;
  loom_serve_text_session_t* session = service->pending.prepared.session;
  loom_serve_text_prefix_entry_t* prefix = service->pending.prepared.prefix;
  const iree_host_size_t input_origin = prefix ? prefix->count : 0;
  const iree_host_size_t input_count = pending->input.count - input_origin;
  const iree_host_size_t retained_count =
      pending->input.retained_count + input_origin;
  const iree_host_size_t reservation = pending->reservation;
  const iree_host_size_t restored_bytes =
      service->pending.prepared.restored_bytes;
  if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING) ||
      !pending->consumer.vtable->is_open(pending->consumer.self)) {
    loom_serve_text_reset_session(service, session);
    loom_serve_text_pending_cancel(service, service->pending.admission.cursor);
    return iree_ok_status();
  }
  const loom_serve_text_pending_request_t request = *pending;
  // Admission may discover a new retained prefix while projection was queued.
  // Detach obsolete jobs before their borrowed pending chat changes owners.
  loom_serve_text_media_projector_cancel(service->projector, pending->media);
  loom_serve_text_media_request_begin(pending->media);
  const iree_host_size_t queue_rank = service->pending.admission.cursor;
  loom_serve_text_admission_acquired(&service->pending.admission);
  const iree_host_size_t overtakes = service->pending.admission.overtakes;
  memcpy(session->tokens, pending->input.tokens,
         pending->input.count * sizeof(*session->tokens));
  if (prefix) {
    loom_serve_text_prefix_cache_touch(&service->prefixes, prefix);
  }
  const iree_duration_t prepare_duration = pending->prepare_duration;
  loom_serve_text_session_cache_entry_t* cached =
      service->pending.prepared.cached;
  if (cached) {
    loom_serve_text_telemetry_session_end(&service->telemetry,
                                          &session->lifetime);
    session->lifetime = cached->lifetime;
    memset(&cached->lifetime, 0, sizeof(cached->lifetime));
    loom_serve_text_session_cache_release(&service->cache, cached,
                                          &service->telemetry);
  }
  loom_serve_text_input_deinitialize(&pending->input, service->host_allocator);
  loom_serve_text_pending_remove(service, service->pending.admission.cursor);
  if (!request.name.size ||
      !iree_string_view_equal(request.name,
                              iree_make_cstring_view(session->lifetime.name))) {
    loom_serve_text_telemetry_session_end(&service->telemetry,
                                          &session->lifetime);
  }
  if (request.name.size) {
    memcpy(session->lifetime.name, request.name.data, request.name.size);
  }
  session->lifetime.name[request.name.size] = 0;
  session->lifetime.serial = request.receipt.serial;
  if (!session->lifetime.id) {
    session->lifetime.id = session->lifetime.serial;
    loom_serve_text_session_record_t value = {
        .session = session->lifetime.id, .row = session - service->sessions};
    memcpy(value.name, session->lifetime.name, sizeof(value.name));
    loom_serve_text_record(service->telemetry.stream,
                           &LOOM_SERVE_TEXT_SESSION_BEGIN,
                           session->lifetime.serial, &value, sizeof(value));
  }
  ++session->lifetime.counters.requests;
  session->request.counters.requests = 1;
  session->request.consumer = request.consumer;
  session->request.chat = request.chat;
  session->request.media = request.media;
  session->request.phase = LOOM_SERVE_TEXT_REQUEST_PREFILL;
  session->request.input_count = input_count;
  session->request.input_origin = input_origin;
  session->request.retained_count = retained_count;
  session->request.start_time = iree_time_now();
  session->request.queue_duration =
      session->request.start_time - request.receipt.arrival_time;
  session->request.reserved_tokens = reservation;
  iree_status_t status =
      loom_serve_text_row_bind_request(session->row, request.chat.options);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_session_begin(
        session, loom_serve_text_model_tokenizer(service->model),
        service->name);
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_text_pool_usage_t pool =
        loom_serve_text_model_pool_usage(service->model);
    const loom_serve_text_admission_record_t value = {
        .session = session->lifetime.id,
        .row = session - service->sessions,
        .retained_tokens = retained_count,
        .appended_tokens = input_count,
        .reservation_tokens = reservation,
        .restored_bytes = restored_bytes,
        .max_tokens = request.chat.max_tokens,
        .queue_rank = queue_rank,
        .overtakes = overtakes,
        .overtake_budget = service->pending.admission.budget,
        .prepare_ns = prepare_duration,
        .queue_ns = session->request.queue_duration,
        .pool_reserved_tokens = pool.pending,
        .pool_resident_tokens = pool.capacity - pool.available - pool.pending};
    loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_ADMIT,
                           session->lifetime.serial, &value, sizeof(value));
  }
  return status;
}

static iree_status_t loom_serve_text_complete_reservation(
    loom_serve_text_service_t* service, loom_serve_service_flags_t flags) {
  if (loom_serve_text_row_complete_reserve(
          service->pending.prepared.session->row)) {
    return loom_serve_text_admit_request(service, flags);
  }
  if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING) ||
      !service->pending.prepared.request->consumer.vtable->is_open(
          service->pending.prepared.request->consumer.self)) {
    loom_serve_text_pending_cancel(service, service->pending.admission.cursor);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_retain_selected(
    loom_serve_text_service_t* service) {
  loom_serve_text_session_t* session = service->pending.prepared.session;
  if (!service->cache.capacity || !session->lifetime.name[0] ||
      !session->completion.reusable ||
      iree_string_view_equal(service->pending.prepared.request->name,
                             iree_make_cstring_view(session->lifetime.name))) {
    return iree_ok_status();
  }
  if (service->cache.count == service->cache.capacity &&
      !service->pending.prepared.cached) {
    loom_serve_text_session_cache_release(
        &service->cache,
        loom_serve_text_session_cache_oldest(&service->cache, NULL),
        &service->telemetry);
    service->pool.trim.pending = true;
  }
  loom_serve_text_observe(service, "retain");
  IREE_RETURN_IF_ERROR(loom_serve_text_session_cache_submit(
      &service->cache, session, &service->pending.prepared.detaching,
      &service->completion.frontier));
  return loom_serve_text_await_work(service, TEXT_WORK_RETAIN);
}

static void loom_serve_text_complete_retention(
    loom_serve_text_service_t* service) {
  loom_serve_text_session_t* session = service->pending.prepared.session;
  loom_serve_text_session_cache_publish(service->pending.prepared.detaching,
                                        session);
  service->pending.prepared.detaching = NULL;
  loom_serve_text_row_reset_retired(session->row);
  service->pool.trim.pending = true;
}

static iree_status_t loom_serve_text_admit_pending(
    loom_serve_text_service_t* service, loom_serve_service_flags_t flags,
    bool allow_preparation) {
  // Cancellation can leave a completed full-cache exchange without its
  // incoming admission. Settle that excess here, never in terminal teardown's
  // host-only pending removal while device ownership may remain unresolved.
  if (!service->pending.prepared.session &&
      service->cache.count > service->cache.capacity) {
    loom_serve_text_session_cache_release(
        &service->cache,
        loom_serve_text_session_cache_oldest(&service->cache, NULL),
        &service->telemetry);
    service->pool.trim.pending = true;
  }
  iree_status_t status = iree_ok_status();
  while (service->pending.count && iree_status_is_ok(status)) {
    loom_serve_text_pending_request_t* pending =
        service->pending.values[service->pending.admission.cursor];
    if (!service->pending.prepared.session) {
      if (allow_preparation) {
        loom_serve_text_begin_preparation(service);
      }
      if (!service->pending.prepared.session ||
          loom_serve_text_has_work(service)) {
        break;
      }
    }
    loom_serve_text_session_t* session = service->pending.prepared.session;
    const iree_host_size_t retained_count = pending->input.retained_count;
    const iree_host_size_t reservation = pending->reservation;
    bool media_ready = true;
    status = loom_serve_text_prepare_media(service, &media_ready);
    if (!iree_status_is_ok(status)) {
      status = loom_serve_text_reject_prepared(service, status);
      continue;
    }
    if (!media_ready) {
      memset(&service->pending.prepared, 0, sizeof(service->pending.prepared));
      if (loom_serve_text_admission_blocked(&service->pending.admission,
                                            service->pending.count)) {
        continue;
      }
      break;
    }
    status = loom_serve_text_retain_selected(service);
    if (!iree_status_is_ok(status) ||
        service->completion.work != TEXT_WORK_NONE) {
      break;
    }
    const loom_serve_text_checkpoint_t* checkpoint =
        pending->checkpoint ? pending->checkpoint->state : NULL;
    if (!checkpoint && retained_count && service->pending.prepared.cached) {
      checkpoint = service->pending.prepared.cached->endpoint.state;
    }
    if (!checkpoint && !retained_count) {
      if (service->pending.prepared.prefix) {
        checkpoint = service->pending.prepared.prefix->state;
      }
    }
    service->pending.prepared.restored_bytes =
        checkpoint ? loom_serve_text_checkpoint_suspended_bytes(checkpoint) : 0;
    loom_serve_text_admission_result_t admission;
    status = loom_serve_text_reclaim_idle(
        service, session, checkpoint, retained_count, reservation, &admission);
    if (!iree_status_is_ok(status) ||
        service->completion.work != TEXT_WORK_NONE) {
      break;
    }
    if (admission == TEXT_ADMISSION_QUEUED) {
      break;
    }
    if (admission == TEXT_ADMISSION_CAPACITY ||
        admission == TEXT_ADMISSION_STATE_CAPACITY) {
      if (service->pending.prepared.prefix) {
        // Opportunistic sharing must not make an otherwise fitting request
        // impossible. The original full replay tokens remain unchanged.
        loom_serve_text_prefix_cache_release(&service->prefixes,
                                             service->pending.prepared.prefix);
        service->pending.prepared.prefix = NULL;
        service->pool.trim.pending = true;
        continue;
      }
      bool active =
          iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY);
      active |= loom_serve_text_media_projector_is_pending(service->projector);
      for (iree_host_size_t i = 0; i < service->row_count; ++i) {
        active |= service->sessions[i].request.consumer.self != NULL;
      }
      if (active) {
        if (admission == TEXT_ADMISSION_CAPACITY) {
          break;
        }
        const iree_host_size_t previous = service->pending.admission.cursor;
        const bool probe = loom_serve_text_admission_blocked(
            &service->pending.admission, service->pending.count);
        if (previous != service->pending.admission.cursor) {
          loom_serve_residency_cancel_admission(
              loom_serve_text_model_residency(service->model));
          memset(&service->pending.prepared, 0,
                 sizeof(service->pending.prepared));
        }
        if (probe) {
          continue;
        }
        break;
      }
      status = loom_serve_text_reject_prepared(
          service,
          iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "retained state leaves insufficient completion capacity; "
              "release a checkpoint, lower max_tokens, or increase "
              "pool_capacity or memory_bytes"));
      continue;
    }
    status = loom_serve_text_admit_request(service, flags);
  }
  return status;
}

static void loom_serve_text_pending_push(
    loom_serve_text_service_t* service, loom_serve_text_consumer_t consumer,
    iree_string_view_t body, iree_string_view_t name,
    loom_serve_text_checkpoint_entry_t* checkpoint,
    loom_serve_text_receipt_t receipt) {
  loom_serve_text_pending_queue_push(&service->pending, consumer, body, name,
                                     checkpoint, receipt);
  loom_serve_text_pending_observe(service);
  // Both ingress paths can prepare the oldest queued input while an accepted
  // epoch owns different rows. Scheduling this finite loan cannot fail or
  // change the accepted request's ownership.
  if (service->completion.work == TEXT_WORK_EPOCH &&
      service->cpu_work == TEXT_CPU_WORK_NONE) {
    loom_serve_text_begin_preparation(service);
  }
}

iree_status_t loom_serve_text_service_try_open(
    loom_serve_text_service_t* service, loom_serve_text_request_t* request,
    loom_serve_text_stream_t** out_stream) {
  *out_stream = NULL;
  if (service->stopping) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "text service is stopping");
  }
  loom_serve_text_checkpoint_entry_t* checkpoint = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_pending_queue_select_request(
      &service->pending, request->session, request->checkpoint,
      service->row_count, service->sessions, &service->checkpoints,
      &checkpoint));
  if (service->pending.count == service->pending.capacity) {
    return iree_ok_status();
  }
  loom_serve_text_consumer_t consumer;
  IREE_RETURN_IF_ERROR(loom_serve_text_stream_allocate(
      service->name, request, service->host_allocator, out_stream, &consumer));
  const loom_serve_text_request_t* input =
      loom_serve_text_stream_request(*out_stream);
  const loom_serve_text_receipt_t receipt =
      loom_serve_text_telemetry_begin(&service->telemetry);
  loom_serve_text_pending_push(service, consumer,
                               iree_string_builder_view(&input->body),
                               input->session, checkpoint, receipt);
  return iree_ok_status();
}

iree_status_t loom_serve_text_service_try_open_checkpoint(
    loom_serve_text_service_t* service,
    const loom_serve_text_checkpoint_request_t* request,
    loom_serve_text_checkpoint_control_t** out_control) {
  *out_control = NULL;
  if (service->stopping) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "text service is stopping");
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_request_validate(request));
  if (service->control.consumer.self) {
    return iree_ok_status();
  }
  loom_serve_text_checkpoint_consumer_t consumer;
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_control_allocate(
      request, service->host_allocator, out_control, &consumer));
  service->control = (loom_serve_text_checkpoint_operation_t){
      .consumer = consumer,
      .request = *loom_serve_text_checkpoint_control_request(*out_control),
      .receipt = loom_serve_text_telemetry_begin(&service->telemetry)};
  return iree_ok_status();
}

static iree_status_t loom_serve_text_enqueue(
    loom_serve_text_service_t* service,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_text_receipt_t receipt) {
  iree_string_view_t checkpoint_name = request->target;
  if (iree_string_view_consume_prefix(&checkpoint_name,
                                      IREE_SV("/v1/checkpoints/"))) {
    if (service->control.consumer.self) {
      return loom_serve_text_http_reject(
          &service->telemetry, connection, receipt, 503, "Service Unavailable",
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "model control queue is full"),
          service->host_allocator);
    }
    loom_serve_text_checkpoint_request_t input;
    iree_status_t status =
        loom_serve_text_checkpoint_http_parse(request, &input);
    if (!iree_status_is_ok(status)) {
      return loom_serve_text_http_reject(&service->telemetry, connection,
                                         receipt, 400, "Bad Request", status,
                                         service->host_allocator);
    }
    service->control = (loom_serve_text_checkpoint_operation_t){
        .consumer = loom_serve_text_checkpoint_http_consumer(connection),
        .request = input,
        .receipt = receipt};
    return loom_serve_text_has_work(service)
               ? iree_ok_status()
               : loom_serve_text_submit_checkpoint(service);
  }
  const bool input_count = iree_string_view_equal(
      request->target, IREE_SV("/v1/chat/completions/input_tokens"));
  if (!iree_string_view_equal(request->method, IREE_SV("POST")) ||
      (!input_count && !iree_string_view_equal(
                           request->target, IREE_SV("/v1/chat/completions")))) {
    return loom_serve_text_http_reject(
        &service->telemetry, connection, receipt, 404, "Not Found",
        iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "use /v1/chat/completions, "
            "/v1/chat/completions/input_tokens, or /v1/checkpoints/NAME"),
        service->host_allocator);
  }
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("Content-Type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8")))) {
    return loom_serve_text_http_reject(
        &service->telemetry, connection, receipt, 415, "Unsupported Media Type",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Content-Type must be application/json"),
        service->host_allocator);
  }
  if (input_count) {
    iree_status_t status = loom_serve_text_input_count_enqueue(
        &service->input_counts, connection, request, receipt,
        &service->telemetry);
    loom_serve_text_pending_observe(service);
    return status;
  }
  iree_string_view_t name;
  loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Session"),
                                        &name);
  if (loom_serve_http_request_lookup_header(
          request, IREE_SV("X-Loom-Checkpoint"), &checkpoint_name) &&
      !checkpoint_name.size) {
    return loom_serve_text_http_reject(
        &service->telemetry, connection, receipt, 400, "Bad Request",
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "X-Loom-Checkpoint requires a nonempty identifier"),
        service->host_allocator);
  }
  loom_serve_text_checkpoint_entry_t* checkpoint = NULL;
  iree_status_t status = loom_serve_text_pending_queue_select_request(
      &service->pending, name, checkpoint_name, service->row_count,
      service->sessions, &service->checkpoints, &checkpoint);
  if (!iree_status_is_ok(status)) {
    const bool missing = iree_status_is_not_found(status);
    const bool conflict = iree_status_is_already_exists(status);
    return loom_serve_text_http_reject(&service->telemetry, connection, receipt,
                                       missing    ? 404
                                       : conflict ? 409
                                                  : 400,
                                       missing    ? "Not Found"
                                       : conflict ? "Conflict"
                                                  : "Bad Request",
                                       status, service->host_allocator);
  }
  if (service->pending.count == service->pending.capacity) {
    return loom_serve_text_http_reject(
        &service->telemetry, connection, receipt, 503, "Service Unavailable",
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "pending request queue is full"),
        service->host_allocator);
  }
  loom_serve_text_consumer_t consumer;
  status = loom_serve_text_http_consumer_allocate(
      connection, &service->telemetry, receipt, service->host_allocator,
      &consumer);
  if (!iree_status_is_ok(status)) {
    return loom_serve_text_http_reject(&service->telemetry, connection, receipt,
                                       503, "Service Unavailable", status,
                                       service->host_allocator);
  }
  loom_serve_text_pending_push(service, consumer, request->body, name,
                               checkpoint, receipt);
  return iree_ok_status();
}

// Delivery owns a published slot and a staging slot. An empty stage permits
// another epoch even while the preceding output is unread. A full stage behind
// a stalled consumer pauses only that row; cancellation waits for retirement.
static iree_status_t loom_serve_text_prepare_ready(
    loom_serve_text_service_t* service, loom_serve_text_session_t* session,
    bool* out_progress, loom_serve_ready_span_t* out_ready) {
  *out_ready = (loom_serve_ready_span_t){0};
  const loom_serve_text_consumer_t consumer = session->request.consumer;
  if (!consumer.self) {
    return iree_ok_status();
  }
  if (!consumer.vtable->is_open(consumer.self)) {
    loom_serve_text_request_cancel(service, session);
    *out_progress = true;
    loom_serve_text_row_reset_retired(session->row);
    return iree_ok_status();
  }
  bool ready = false;
  bool progress = false;
  iree_status_t status =
      consumer.vtable->publish(consumer.self, &ready, &progress);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_telemetry_diagnose(&service->telemetry,
                                       session->lifetime.serial, status);
    loom_serve_text_request_cancel(service, session);
    *out_progress = true;
    loom_serve_text_row_reset_retired(session->row);
    return iree_ok_status();
  }
  *out_progress |= progress;
  if (!ready) {
    return iree_ok_status();
  }
  if (session->request.publication.pending &&
      !session->request.publication.recorded) {
    loom_serve_telemetry_latency_publish_output(
        &service->telemetry.latency, service->telemetry.stream,
        session->lifetime.serial,
        session->request.start_time - session->request.queue_duration);
    session->request.publication.recorded = true;
  }
  session->request.publication.pending = false;
  if (session->request.phase == LOOM_SERVE_TEXT_REQUEST_FINISHING) {
    loom_serve_text_request_finish(service, session);
    *out_progress = true;
    return iree_ok_status();
  }
  if (session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL) {
    *out_ready = (loom_serve_ready_span_t){
        session->request.input_count - session->request.input_offset, 1};
  } else {
    // A final single output cannot amortize drafting. Near the context limit,
    // ordinary decode avoids writing speculative KV beyond allocated capacity.
    const iree_host_size_t width = service->scheduling.draft_depth + 1;
    const bool verify =
        service->scheduling.draft_depth &&
        session->request.chat.max_tokens - session->request.output_count > 1 &&
        service->scheduling.context_capacity -
                loom_serve_text_row_position(session->row) >=
            width;
    *out_ready = verify ? (loom_serve_ready_span_t){width, width}
                        : (loom_serve_ready_span_t){1, 1};
  }
  return iree_ok_status();
}

// The rotating first ready row chooses the phase; other rows of that phase
// can still batch together. Explicit request phase matters: a one-token prompt
// tail is not a decode. Credit and cancellation were settled before this
// filter.
static void loom_serve_text_separate_ready(
    const loom_serve_text_service_t* service, loom_serve_ready_span_t* ready) {
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    const iree_host_size_t first = (service->cursor + i) % service->row_count;
    if (!ready[first].token_count) {
      continue;
    }
    const loom_serve_text_request_phase_t phase =
        service->sessions[first].request.phase;
    for (iree_host_size_t row = 0; row < service->row_count; ++row) {
      if (service->sessions[row].request.phase != phase) {
        ready[row] = (loom_serve_ready_span_t){0};
      }
    }
    return;
  }
}

static iree_status_t loom_serve_text_submit_epoch(
    loom_serve_text_service_t* service, iree_host_size_t shape_index,
    iree_host_size_t count, const loom_serve_packed_span_t* scheduled) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->name.data, service->name.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0,
      loom_serve_telemetry_stream_identity(service->telemetry.stream).ordinal);
  loom_serve_text_epoch_row_t rows[LOOM_SERVE_TEXT_ROW_CAPACITY];
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t row = scheduled[i].row_index;
    const loom_serve_text_session_t* session = &service->sessions[row];
    const bool prefill =
        session->request.phase == LOOM_SERVE_TEXT_REQUEST_PREFILL;
    rows[row] = (loom_serve_text_epoch_row_t){
        .input_tokens = prefill
                            ? session->tokens + session->request.input_origin +
                                  session->request.input_offset
                            : NULL,
        .input_count = prefill ? session->request.input_count -
                                     session->request.input_offset
                               : 0,
        .input_offset =
            session->request.input_origin + session->request.input_offset,
        .media = prefill ? loom_serve_text_media_request_input(
                               session->request.media)
                         : NULL,
        .output_count =
            session->request.chat.max_tokens - session->request.output_count,
        .position = loom_serve_text_row_position(session->row),
        .token = prefill ? 0 : loom_serve_text_row_token(session->row),
    };
  }
  loom_serve_text_epoch_continuation_t continuation;
  loom_serve_text_epoch_prepare(&service->scheduling, rows, shape_index, count,
                                scheduled, &service->epoch, &continuation);
  loom_serve_text_observe(service, "execute");
  const iree_time_t start = iree_time_now();
  iree_slim_mutex_lock(&service->heartbeat.mutex);
  loom_serve_text_heartbeat_snapshot_t* state = &service->heartbeat.snapshot;
  service->epoch.start.serial = state->issued_epochs + 1;
  service->epoch.start.time = start;
  state->issued_epochs += service->epoch.device_epochs;
  state->phase_start = start;
  state->epoch_spans = count;
  state->epoch_tokens = service->epoch.input_count;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, service->epoch.start.serial);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loom_serve_text_epoch_submit(
              service->model, service->scheduling.schedule_mode,
              &service->epoch, &continuation, &service->completion.frontier));
  const iree_status_t status =
      loom_serve_text_await_work(service, TEXT_WORK_EPOCH);
  IREE_RETURN_AND_END_ZONE(z0, status);
}

static void loom_serve_text_publish_epoch(loom_serve_text_service_t* service) {
  loom_serve_text_telemetry_epoch(&service->telemetry, &service->heartbeat,
                                  &service->scheduling, &service->epoch,
                                  service->sessions);
  loom_serve_text_observe(service, "output");
  for (iree_host_size_t i = 0; i < service->epoch.count; ++i) {
    loom_serve_text_session_t* session =
        &service->sessions[service->epoch.spans[i].row_index];
    const bool prefill = service->epoch.prefill[i];
    if (prefill) {
      session->request.input_offset += service->epoch.results[i].known_count;
      loom_serve_text_media_request_consume(
          session->request.media,
          session->request.input_origin + session->request.input_offset);
      if (session->request.input_offset == session->request.input_count) {
        session->request.phase = LOOM_SERVE_TEXT_REQUEST_DECODE;
      }
    }
  }
}

static iree_status_t loom_serve_text_process_output(void* user_data) {
  loom_serve_text_service_t* service = user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->name.data, service->name.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0,
      loom_serve_telemetry_stream_identity(service->telemetry.stream).ordinal);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, service->epoch.start.serial);
  iree_arena_allocator_t arena;
  iree_arena_initialize(loom_serve_host_block_pool(loom_serve_device_host(
                            loom_serve_text_model_device(service->model))),
                        &arena);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->epoch.count && iree_status_is_ok(status); ++i) {
    const loom_serve_text_result_t* output = &service->epoch.results[i];
    if (output->output_count) {
      loom_serve_text_session_t* session =
          &service->sessions[service->epoch.spans[i].row_index];
      status = loom_serve_text_session_append_tokens(
          session, service->name, output->output_count, output->tokens,
          &service->scratch, &service->tool_calls, &service->completion_scratch,
          iree_arena_allocator(&arena));
    }
  }
  iree_arena_deinitialize(&arena);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_serve_text_complete_output(
    loom_serve_text_service_t* service) {
  for (iree_host_size_t i = 0; i < service->epoch.count; ++i) {
    loom_serve_text_session_t* session =
        &service->sessions[service->epoch.spans[i].row_index];
    if (session->request.diagnostic.code) {
      loom_serve_telemetry_record_t record = {
          .time_ns = iree_time_now(),
          .request = session->lifetime.serial,
          .event = &LOOM_SERVE_TELEMETRY_ERROR,
          .data.error = session->request.diagnostic};
      loom_serve_telemetry_stream_append(service->telemetry.stream, &record);
    }
    if (session->request.phase == LOOM_SERVE_TEXT_REQUEST_FINISHING) {
      loom_serve_text_row_release_reservation(session->row);
      service->pool.trim.pending = true;
    }
  }
  service->epoch.count = 0;
}

// Output processing borrows the VM only after any captured prefix has retired.
static void loom_serve_text_begin_output(loom_serve_text_service_t* service) {
  bool has_output = false;
  for (iree_host_size_t i = 0; i < service->epoch.count; ++i) {
    has_output |= service->epoch.results[i].output_count != 0;
  }
  if (has_output) {
    loom_serve_text_schedule_cpu(service, TEXT_CPU_WORK_OUTPUT,
                                 loom_serve_text_process_output);
  } else {
    service->epoch.count = 0;
  }
}

// Capture at most one new frontier per retired epoch. Decode rows perform no
// token lookup; capture adds no model traversal, prefill split or cohort hold.
static iree_status_t loom_serve_text_capture_prefix(
    loom_serve_text_service_t* service) {
  if (!service->prefixes.capacity) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->epoch.count && iree_status_is_ok(status); ++i) {
    loom_serve_text_session_t* session =
        &service->sessions[service->epoch.spans[i].row_index];
    const iree_host_size_t count =
        session->request.input_origin + session->request.input_offset;
    if (!service->epoch.prefill[i] ||
        session->request.phase != LOOM_SERVE_TEXT_REQUEST_PREFILL ||
        session->request.retained_count != session->request.input_origin ||
        count < service->pool.geometry.block_size ||
        !session->request.consumer.vtable->is_open(
            session->request.consumer.self)) {
      continue;
    }
    // A later checkpoint uses spare capacity near the prompt's end without
    // forcing a new epoch boundary or replacing an already useful ancestor.
    const iree_host_size_t window =
        iree_min(service->scheduling.chunk_size,
                 loom_serve_text_model_prefill_capacity(service->model)) *
        service->scheduling.continuation_epochs;
    const iree_host_size_t input_count =
        session->request.input_origin + session->request.input_count;
    const iree_host_size_t extension_start =
        input_count - iree_min(input_count, window);
    bool evicted = false;
    loom_serve_execution_frontier_t frontier;
    status = loom_serve_text_prefix_cache_submit(
        &service->prefixes, &session->request.chat, session->row, count,
        session->tokens, extension_start, service->pending.prepared.prefix,
        &evicted, &frontier);
    service->pool.trim.pending |= evicted;
    if (iree_status_is_ok(status) && frontier.semaphores[0]) {
      service->completion.frontier = frontier;
      loom_serve_text_observe(service, "prefix_capture");
      status = loom_serve_text_await_work(service, TEXT_WORK_PREFIX);
      break;
    }
  }
  return status;
}

static iree_status_t loom_serve_text_complete_epoch(
    loom_serve_text_service_t* service, loom_serve_service_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, service->name.data, service->name.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0,
      loom_serve_telemetry_stream_identity(service->telemetry.stream).ordinal);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, service->epoch.start.serial);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      loom_serve_text_epoch_complete(
          service->model, service->scheduling.schedule_mode, &service->epoch));
  if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    if (service->scheduling.schedule_mode != LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
      // Only these isolated spans were submitted; the remainder never ran.
      service->epoch.count = service->epoch.cursor;
    }
    loom_serve_text_publish_epoch(service);
    service->epoch.count = 0;
    IREE_RETURN_AND_END_ZONE(z0, iree_ok_status());
  }
  if (service->scheduling.schedule_mode != LOOM_SERVE_TEXT_SCHEDULE_PACKED &&
      service->epoch.cursor < service->epoch.count) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, loom_serve_text_epoch_submit(
                service->model, service->scheduling.schedule_mode,
                &service->epoch, NULL, &service->completion.frontier));
    const iree_status_t status =
        loom_serve_text_await_work(service, TEXT_WORK_EPOCH);
    IREE_RETURN_AND_END_ZONE(z0, status);
  }
  loom_serve_text_publish_epoch(service);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    loom_serve_text_capture_prefix(service));
  if (service->completion.work == TEXT_WORK_NONE) {
    loom_serve_text_begin_output(service);
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t loom_serve_text_service_initialize(
    loom_serve_text_service_t* service, loom_serve_telemetry_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_serve_text_telemetry_initialize(
      stream, service->host_allocator, &service->telemetry));
  IREE_RETURN_IF_ERROR(loom_serve_text_input_count_initialize(
      service->pending.capacity, service->host_allocator,
      &service->input_counts));
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_registry_initialize(
      loom_serve_text_model_checkpoint_capacity(service->model) -
          (service->cache.capacity ? service->cache.capacity + 1 : 0) -
          service->prefixes.capacity,
      service->host_allocator, &service->checkpoints));
  IREE_RETURN_IF_ERROR(loom_serve_text_session_cache_initialize(
      service->cache.capacity, service->host_allocator, &service->cache));
  IREE_RETURN_IF_ERROR(loom_serve_text_prefix_cache_initialize(
      service->prefixes.capacity, service->host_allocator, &service->prefixes));
  IREE_RETURN_IF_ERROR(loom_serve_text_pending_queue_initialize(
      service->model, &service->telemetry, service->pending.capacity,
      service->pending.admission.budget, service->host_allocator,
      &service->pending));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < service->row_count && iree_status_is_ok(status); ++i) {
    status = loom_serve_text_session_initialize(
        loom_serve_text_model_row(service->model, i),
        loom_serve_text_model_tokenizer(service->model),
        service->scheduling.context_capacity, service->host_allocator,
        &service->sessions[i]);
  }
  return status;
}

// Resolves protocol and session lifetimes before the registry snapshots final
// counters. Accepted CPU/device work has relinquished its service payloads;
// failed device readiness leaves model storage pinned for actual retirement.
static void loom_serve_text_stop(loom_serve_text_service_t* service,
                                 iree_status_code_t terminal_code) {
  service->stopping = true;
  loom_serve_residency_cancel_admission(
      loom_serve_text_model_residency(service->model));
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    loom_serve_text_session_t* session = &service->sessions[i];
    if (session->request.consumer.self) {
      // Failed device readiness keeps pins and reserved views until model
      // destruction actually retires accepted work. Normal shutdown is at a
      // completed model boundary and can release unused completion capacity.
      if (terminal_code == IREE_STATUS_OK) {
        loom_serve_text_row_release_reservation(session->row);
      }
      loom_serve_text_request_cancel(service, session);
    }
    loom_serve_text_telemetry_session_end(&service->telemetry,
                                          &session->lifetime);
  }
  while (service->pending.count) {
    loom_serve_text_pending_cancel(service, 0);
  }
  loom_serve_text_input_count_cancel(&service->input_counts, false, true,
                                     &service->telemetry);
  loom_serve_text_checkpoint_request_abort(&service->control,
                                           &service->telemetry);
  loom_serve_text_checkpoint_registry_reset(&service->checkpoints,
                                            terminal_code);
  loom_serve_text_session_cache_reset(&service->cache, &service->telemetry,
                                      terminal_code);
  loom_serve_text_prefix_cache_reset(&service->prefixes, terminal_code);
  service->pool.trim.pending = false;
  loom_serve_text_observe(
      service, terminal_code == IREE_STATUS_OK ? "stopped" : "failed");
}

void loom_serve_text_service_release_projector(
    loom_serve_text_service_t* service) {
  loom_serve_text_media_projector_free(service->projector);
  service->projector = NULL;
  service->media_preparation_end = 0;
}

void loom_serve_text_service_free(loom_serve_text_service_t* service,
                                  iree_status_code_t terminal_code) {
  if (!service) {
    return;
  }
  loom_serve_residency_set_reclaimer(
      loom_serve_text_model_residency(service->model),
      (loom_serve_residency_reclaimer_t){0});
  if (service->telemetry.failure_code) {
    terminal_code = service->telemetry.failure_code;
  }
  loom_serve_text_stop(service, terminal_code);
  loom_serve_text_media_projector_free(service->projector);
  const iree_allocator_t host_allocator = service->host_allocator;
  loom_serve_execution_reconfiguration_free(service->reconfiguration);
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    loom_serve_text_session_deinitialize(&service->sessions[i]);
  }
  loom_serve_text_readiness_deinitialize(&service->readiness);
  loom_serve_text_tool_calls_deinitialize(&service->tool_calls);
  loom_serve_text_chat_scratch_deinitialize(&service->completion_scratch);
  iree_string_builder_deinitialize(&service->scratch);
  loom_serve_text_pending_queue_deinitialize(&service->pending);
  loom_serve_text_input_count_deinitialize(&service->input_counts);
  loom_serve_text_checkpoint_registry_deinitialize(&service->checkpoints);
  loom_serve_text_session_cache_deinitialize(&service->cache);
  loom_serve_text_prefix_cache_deinitialize(&service->prefixes);
  loom_serve_text_heartbeat_deinitialize(&service->heartbeat);
  if (service->telemetry.owns_stream) {
    loom_serve_telemetry_stream_free(service->telemetry.stream);
  }
  iree_allocator_free_aligned(host_allocator, service);
}

iree_status_t loom_serve_text_service_allocate(
    loom_serve_text_model_t* model,
    const loom_serve_text_service_options_t* options,
    loom_serve_telemetry_stream_t* stream, iree_allocator_t host_allocator,
    loom_serve_text_service_t** out_service) {
  *out_service = NULL;
  const iree_host_size_t draft_depth = loom_serve_text_model_draft_depth(model);
  if ((draft_depth &&
       options->schedule_mode != LOOM_SERVE_TEXT_SCHEDULE_PACKED) ||
      options->continuation_epochs < 1 || options->continuation_epochs > 2 ||
      (options->continuation_epochs == 2 && !draft_depth)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "drafting requires packed scheduling; two device "
                            "epochs require a nonzero draft depth");
  }
  if (options->session_cache_capacity &&
      options->session_cache_capacity >=
          loom_serve_text_model_checkpoint_capacity(model)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "session cache requires capacity plus one native exchange endpoint");
  }
  const iree_host_size_t session_endpoints =
      options->session_cache_capacity ? options->session_cache_capacity + 1 : 0;
  if (options->prefix_cache_capacity >
      loom_serve_text_model_checkpoint_capacity(model) - session_endpoints) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prefix cache exceeds remaining native endpoints");
  }
  loom_serve_host_t* host =
      loom_serve_device_host(loom_serve_text_model_device(model));
  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(host), 0, &proactor));
  loom_serve_text_service_t* service = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, sizeof(*service), iree_alignof(loom_serve_text_service_t),
      0, (void**)&service));
  *service = (loom_serve_text_service_t){
      .host_allocator = host_allocator,
      .model = model,
      .image = options->image,
      .name = options->name,
      .isolated_shape = {loom_serve_text_model_prefill_capacity(model),
                         options->row_count},
      .row_count = options->row_count,
      .default_max_tokens = options->default_max_tokens,
      .pending = {.capacity = options->pending_capacity,
                  .admission = {.budget = options->admission_overtakes}},
      .cache = {.capacity = options->session_cache_capacity},
      .prefixes = {.capacity = options->prefix_cache_capacity},
      .pool = {.geometry = loom_serve_text_model_pool_usage(model)},
      .scheduling = {
          .chunk_size = options->chunk_size,
          .context_capacity = loom_serve_text_model_context_capacity(model),
          .schedule_mode = options->schedule_mode,
          .packing_mode = options->packing_mode,
          .continuation_epochs = options->continuation_epochs,
          .draft_depth = draft_depth,
          .shapes = loom_serve_text_model_shapes(model),
          .shape_count = loom_serve_text_model_shape_count(model)}};
  if (!service->scheduling.shape_count) {
    service->scheduling.shapes = &service->isolated_shape;
    service->scheduling.shape_count = 1;
  }
  loom_serve_text_heartbeat_initialize(&service->heartbeat);
  loom_serve_text_readiness_initialize(loom_serve_host_executor(host), proactor,
                                       &service->readiness);
  iree_string_builder_initialize(host_allocator, &service->scratch);
  loom_serve_text_tool_calls_initialize(host_allocator, &service->tool_calls);
  loom_serve_text_chat_scratch_initialize(host_allocator,
                                          &service->completion_scratch);
  iree_status_t status = loom_serve_text_service_initialize(service, stream);
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_reconfiguration_allocate(
        loom_serve_text_model_execution_update(model),
        options->variant_capacity, host_allocator, &service->reconfiguration);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_text_heartbeat_configure(
        &service->heartbeat, proactor, service->telemetry.stream,
        &service->telemetry.counters, options->heartbeat_interval);
    *out_service = service;
  } else {
    loom_serve_text_service_free(service, iree_status_code(status));
  }
  return status;
}

static void loom_serve_text_set_observer(void* self,
                                         loom_serve_wake_callback_t observer) {
  loom_serve_text_service_t* service = self;
  loom_serve_text_readiness_set_observer(&service->readiness, observer);
  loom_serve_text_media_projector_set_observer(service->projector, observer);
  loom_serve_execution_reconfiguration_set_observer(service->reconfiguration,
                                                    observer);
  loom_serve_residency_set_reclaimer(
      loom_serve_text_model_residency(service->model),
      observer.wake
          ? (loom_serve_residency_reclaimer_t){service,
                                               loom_serve_text_cache_reclaim_request,
                                               loom_serve_text_cache_reclaim_pending}
          : (loom_serve_residency_reclaimer_t){0});
  loom_serve_timer_set_observer(&service->heartbeat.timer, observer);
}

static bool loom_serve_text_is_pending(void* self) {
  loom_serve_text_service_t* service = self;
  return loom_serve_text_has_work(service) ||
         loom_serve_text_media_projector_is_pending(service->projector) ||
         loom_serve_finite_component_is_pending(service->image) ||
         loom_serve_execution_reconfiguration_pending(
             service->reconfiguration) ||
         loom_serve_text_heartbeat_is_pending(&service->heartbeat);
}

static bool loom_serve_text_is_active(void* self) {
  const loom_serve_text_service_t* service = self;
  if (loom_serve_text_has_work(service) || service->pool.trim.pending ||
      loom_serve_text_media_projector_is_pending(service->projector) ||
      service->control.consumer.self) {
    return true;
  }
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    if (service->sessions[i].request.consumer.self) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_serve_text_advance_work(
    void* self, loom_serve_service_flags_t flags, bool* out_progress) {
  loom_serve_text_service_t* service = self;
  *out_progress = false;
  loom_serve_text_pending_cancel_failed(service, out_progress);
  const bool cancelled_counts = loom_serve_text_input_count_cancel(
      &service->input_counts, service->cpu_work == TEXT_CPU_WORK_INPUT_COUNT,
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING),
      &service->telemetry);
  if (cancelled_counts) {
    *out_progress = true;
    loom_serve_text_pending_observe(service);
  }
  bool yield_preparation = false;
  if (service->cpu_work != TEXT_CPU_WORK_NONE) {
    iree_status_t status = iree_ok_status();
    if (!loom_serve_text_readiness_take_cpu(&service->readiness, &status)) {
      return iree_ok_status();
    }
    const loom_serve_text_cpu_work_t work = service->cpu_work;
    service->cpu_work = TEXT_CPU_WORK_NONE;
    *out_progress = true;
    if (work == TEXT_CPU_WORK_PREPARE) {
      status = loom_serve_text_complete_preparation(service, flags, status);
      // Rejected/cancelled input leaves no prepared row. Offer already-ready
      // rows an epoch before another CPU borrower can delay their submission.
      yield_preparation = !service->pending.prepared.session;
    } else if (work == TEXT_CPU_WORK_INPUT_COUNT) {
      status = loom_serve_text_input_count_complete(
          &service->input_counts,
          iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING), status,
          &service->telemetry, &service->scratch);
      loom_serve_text_pending_observe(service);
      yield_preparation = true;
    }
    IREE_RETURN_IF_ERROR(status);
    if (work == TEXT_CPU_WORK_OUTPUT) {
      loom_serve_text_complete_output(service);
    }
  }
  if (service->completion.work != TEXT_WORK_NONE) {
    iree_status_t status = iree_ok_status();
    if (!loom_serve_text_readiness_take_device(&service->readiness, &status)) {
      if (service->completion.work == TEXT_WORK_EPOCH &&
          !iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
        *out_progress |= loom_serve_text_begin_preparation(service);
        *out_progress |= loom_serve_text_begin_input_count(service);
      }
      return iree_ok_status();
    }
    const loom_serve_text_work_t work = service->completion.work;
    service->completion.work = TEXT_WORK_NONE;
    *out_progress = true;
    IREE_RETURN_IF_ERROR(status);
    switch (work) {
      case TEXT_WORK_EPOCH:
        status = loom_serve_text_complete_epoch(service, flags);
        break;
      case TEXT_WORK_CHECKPOINT:
        status = loom_serve_text_checkpoint_request_complete(
            &service->control,
            iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING),
            &service->telemetry, &service->scratch,
            &service->pool.trim.pending);
        break;
      case TEXT_WORK_TRIM:
        status = loom_serve_text_complete_trim(service);
        break;
      case TEXT_WORK_RESERVE:
        status = loom_serve_text_complete_reservation(service, flags);
        break;
      case TEXT_WORK_RETAIN:
        loom_serve_text_complete_retention(service);
        break;
      case TEXT_WORK_PREFIX: {
        const loom_serve_text_prefix_record_t record = {
            .readiness_ns = iree_time_now() - service->completion.start_time,
            .capacity = service->prefixes.capacity,
            .endpoints = service->prefixes.count};
        loom_serve_text_record(service->telemetry.stream,
                               &LOOM_SERVE_TEXT_PREFIX_CAPTURE, 0, &record,
                               sizeof(record));
        if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
          service->epoch.count = 0;
        } else {
          loom_serve_text_begin_output(service);
        }
        break;
      }
      case TEXT_WORK_NONE:
        break;
    }
    IREE_RETURN_IF_ERROR(status);
    if (loom_serve_text_has_work(service)) {
      return iree_ok_status();
    }
  }
  if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    if (loom_serve_text_media_projector_is_pending(service->projector)) {
      return iree_ok_status();
    }
    *out_progress |= service->pending.count || service->control.consumer.self;
    for (iree_host_size_t i = 0; i < service->row_count; ++i) {
      *out_progress |= service->sessions[i].request.consumer.self != NULL;
    }
    loom_serve_text_stop(service, service->telemetry.failure_code);
    return iree_ok_status();
  }
  *out_progress |= loom_serve_execution_reconfiguration_advance(
      service->reconfiguration, /*quiescent=*/true, /*stopping=*/false);
  // Release/suspend controls can free capacity for a stalled generation queue.
  // They run only after the previous epoch relinquishes its retained views.
  if (service->control.consumer.self) {
    *out_progress = true;
    if (!service->control.consumer.is_open(service->control.consumer.self)) {
      loom_serve_text_checkpoint_request_abort(&service->control,
                                               &service->telemetry);
    } else {
      IREE_RETURN_IF_ERROR(loom_serve_text_submit_checkpoint(service));
    }
    if (loom_serve_text_has_work(service)) {
      return iree_ok_status();
    }
  }
  // Collection may have returned the selected candidate's last loan. Retire it
  // before another reservation without rescanning the unborrowed tails.
  if (service->pending.prepared.session &&
      !service->pending.prepared.request->consumer.vtable->is_open(
          service->pending.prepared.request->consumer.self)) {
    loom_serve_text_pending_cancel(service, service->pending.admission.cursor);
    *out_progress = true;
  }
  // Another model may have completed requests or maintenance since the last
  // turn. Retry admission even without a service-local capacity event.
  const iree_host_size_t pending_count = service->pending.count;
  loom_serve_text_observe(service, "admit");
  iree_status_t status =
      loom_serve_text_admit_pending(service, flags, !yield_preparation);
  *out_progress |= pending_count != service->pending.count;
  if (loom_serve_text_has_work(service)) {
    *out_progress = true;
    return status;
  }
  loom_serve_ready_span_t ready[LOOM_SERVE_TEXT_ROW_CAPACITY] = {0};
  for (iree_host_size_t i = 0;
       i < service->row_count && iree_status_is_ok(status); ++i) {
    status = loom_serve_text_prepare_ready(service, &service->sessions[i],
                                           out_progress, &ready[i]);
  }
  if (iree_status_is_ok(status)) {
    *out_progress |= service->pool.trim.pending;
    status = loom_serve_text_submit_trim(service);
  }
  if (iree_status_is_ok(status) && !loom_serve_text_has_work(service)) {
    if (service->scheduling.packing_mode == LOOM_SERVE_TEXT_PACKING_SEPARATE) {
      loom_serve_text_separate_ready(service, ready);
    }
    loom_serve_packed_span_t spans[LOOM_SERVE_TEXT_ROW_CAPACITY];
    loom_serve_packed_span_t scratch[LOOM_SERVE_TEXT_ROW_CAPACITY];
    iree_host_size_t shape_index = 0;
    const iree_host_size_t count = loom_serve_pack_shapes(
        service->row_count, ready, service->scheduling.shape_count,
        service->scheduling.shapes, service->scheduling.chunk_size,
        &service->cursor, spans, scratch, &shape_index);
    if (count) {
      *out_progress = true;
      status = loom_serve_text_submit_epoch(service, shape_index, count, spans);
    } else if (yield_preparation) {
      // With no ready device work, rejection never stalls the remaining FIFO.
      *out_progress |= loom_serve_text_begin_preparation(service);
    }
  }
  if (!loom_serve_text_has_work(service)) {
    loom_serve_text_observe(service,
                            iree_status_is_ok(status) ? "idle" : "failed");
  }
  if (iree_status_is_ok(status)) {
    *out_progress |= loom_serve_text_begin_input_count(service);
  }
  return status;
}

// Lookahead prepares only the already-observed queue window. It never changes
// the admission cursor or spends/replenishes overtaking credit.
static bool loom_serve_text_media_lookahead(
    loom_serve_text_service_t* service) {
  if (!service->media_preparation_end || service->pending.prepared.session ||
      service->cpu_work != TEXT_CPU_WORK_NONE ||
      (service->completion.work != TEXT_WORK_NONE &&
       service->completion.work != TEXT_WORK_EPOCH)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < service->pending.count; ++i) {
    const loom_serve_text_pending_request_t* request =
        service->pending.values[i];
    if (request->receipt.serial > service->media_preparation_end) {
      break;
    }
    if (service->projector && request->media) {
      loom_serve_text_media_projector_offer(service->projector, request->media);
    }
    if (!request->input.tokens &&
        loom_serve_text_begin_preparation_at(service, i)) {
      return true;
    }
  }
  if (!service->projector) {
    return false;
  }
  service->media_preparation_end = 0;
  return loom_serve_text_media_projector_start(service->projector);
}

static iree_status_t loom_serve_text_advance(void* self,
                                             loom_serve_service_flags_t flags,
                                             bool* out_progress) {
  loom_serve_text_service_t* service = self;
  service->stopping |=
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING);
  if (service->stopping) {
    flags |= LOOM_SERVE_SERVICE_FLAG_STOPPING;
  }
  const bool image_progress = loom_serve_finite_component_advance(
      service->image,
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING));
  const bool configuration_progress =
      loom_serve_execution_reconfiguration_advance(
          service->reconfiguration, /*quiescent=*/false,
          iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING));
  iree_status_t status = loom_serve_text_heartbeat_advance(
      &service->heartbeat,
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING), out_progress);
  if (!iree_status_is_ok(status)) {
    flags |= LOOM_SERVE_SERVICE_FLAG_STOPPING;
  }
  bool projection_progress = false;
  loom_serve_service_flags_t projection_flags = flags;
  if (loom_serve_text_is_active(service)) {
    projection_flags |= LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY;
  }
  iree_status_t projection_status = loom_serve_text_media_projector_advance(
      service->projector, projection_flags, &projection_progress);
  if (!iree_status_is_ok(projection_status)) {
    flags |= LOOM_SERVE_SERVICE_FLAG_STOPPING;
  }
  status = iree_status_join(status, projection_status);
  bool work_progress = false;
  iree_status_t work_status =
      loom_serve_text_advance_work(self, flags, &work_progress);
  if (iree_status_is_ok(work_status) &&
      !iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    work_progress |= loom_serve_text_media_lookahead(service);
  }
  // A new model failure needs another owner turn to stop a pending timer.
  *out_progress |= image_progress || projection_progress ||
                   configuration_progress || work_progress ||
                   !iree_status_is_ok(work_status);
  status = iree_status_join(status, work_status);
  if (!iree_status_is_ok(status)) {
    service->stopping = true;
    loom_serve_text_telemetry_fail(&service->telemetry, status);
  }
  return status;
}

static iree_status_t loom_serve_text_accept(
    void* self, loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request,
    loom_serve_service_flags_t flags) {
  loom_serve_text_service_t* service = self;
  const loom_serve_text_receipt_t receipt =
      loom_serve_text_telemetry_begin(&service->telemetry);
  if (service->stopping ||
      iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    service->stopping = true;
    return loom_serve_text_http_reject(
        &service->telemetry, connection, receipt, 503, "Service Unavailable",
        iree_make_status(IREE_STATUS_UNAVAILABLE, "text service is stopping"),
        service->host_allocator);
  }
  iree_status_t status =
      loom_serve_text_enqueue(service, connection, request, receipt);
  if (iree_status_is_ok(status) && !loom_serve_text_has_work(service)) {
    status = loom_serve_text_admit_pending(service, flags,
                                           /*allow_preparation=*/true);
  } else if (iree_status_is_ok(status) &&
             service->completion.work == TEXT_WORK_EPOCH &&
             service->cpu_work == TEXT_CPU_WORK_NONE) {
    loom_serve_text_begin_preparation(service);
  }
  if (iree_status_is_ok(status) &&
      !iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING)) {
    loom_serve_text_begin_input_count(service);
  }
  if (!iree_status_is_ok(status)) {
    service->stopping = true;
    loom_serve_text_telemetry_fail(&service->telemetry, status);
  }
  return status;
}

static iree_status_t loom_serve_text_open(
    void* self, loom_serve_text_request_t* request,
    loom_serve_text_stream_t** out_stream) {
  return loom_serve_text_service_try_open(self, request, out_stream);
}

static iree_status_t loom_serve_text_open_checkpoint(
    void* self, const loom_serve_text_checkpoint_request_t* request,
    loom_serve_text_checkpoint_control_t** out_control) {
  return loom_serve_text_service_try_open_checkpoint(self, request,
                                                     out_control);
}

static loom_serve_service_snapshot_t loom_serve_text_snapshot(void* self) {
  loom_serve_text_service_t* service = self;
  const loom_serve_text_heartbeat_snapshot_t state =
      loom_serve_text_heartbeat_read(&service->heartbeat);
  const iree_time_t now = iree_time_now();
  const char* image_phase =
      loom_serve_text_media_projector_phase(service->projector);
  const bool active_control = service->control.consumer.self &&
                              service->completion.work == TEXT_WORK_CHECKPOINT;
  const bool active_count = service->cpu_work == TEXT_CPU_WORK_INPUT_COUNT;
  iree_time_t queued_arrival =
      service->pending.count ? service->pending.values[0]->receipt.arrival_time
                             : IREE_TIME_INFINITE_FUTURE;
  iree_time_t active_arrival = state.active_rows ? state.oldest_active_arrival
                                                 : IREE_TIME_INFINITE_FUTURE;
  if (service->control.consumer.self) {
    if (active_control) {
      active_arrival =
          iree_min(active_arrival, service->control.receipt.arrival_time);
    } else {
      queued_arrival =
          iree_min(queued_arrival, service->control.receipt.arrival_time);
    }
  }
  if (active_count) {
    active_arrival = iree_min(
        active_arrival, service->input_counts.requests[0].receipt.arrival_time);
  }
  if (service->input_counts.count > (iree_host_size_t)active_count) {
    queued_arrival = iree_min(
        queued_arrival,
        service->input_counts.requests[active_count].receipt.arrival_time);
  }
  return (loom_serve_service_snapshot_t){
      .phase = image_phase && !strcmp(state.phase, "idle") ? image_phase
                                                           : state.phase,
      .queued_requests = service->pending.count +
                         (service->control.consumer.self && !active_control) +
                         service->input_counts.count - active_count,
      .active_requests = state.active_rows + active_control + active_count,
      .oldest_queued_age = queued_arrival == IREE_TIME_INFINITE_FUTURE
                               ? 0
                               : now - queued_arrival,
      .oldest_active_age = active_arrival == IREE_TIME_INFINITE_FUTURE
                               ? 0
                               : now - active_arrival};
}

static bool loom_serve_text_request_at(
    void* self, iree_host_size_t index,
    loom_serve_http_request_snapshot_t* out_snapshot) {
  const loom_serve_text_service_t* service = self;
  if (index < service->pending.count) {
    const loom_serve_text_pending_request_t* pending =
        service->pending.values[index];
    *out_snapshot = (loom_serve_http_request_snapshot_t){
        .id = pending->receipt.serial,
        .phase = "queued",
        .arrival_time = pending->receipt.arrival_time,
        .counters = {.requests = 1}};
  } else if (index - service->pending.count < service->row_count) {
    *out_snapshot = loom_serve_text_session_request_snapshot(
        &service->sessions[index - service->pending.count]);
  } else if (index - service->pending.count == service->row_count) {
    *out_snapshot = (loom_serve_http_request_snapshot_t){
        .id = service->control.consumer.self ? service->control.receipt.serial
                                             : 0,
        .phase = "control",
        .arrival_time = service->control.receipt.arrival_time,
        .counters = {.requests = 1}};
  } else if (index - service->pending.count - service->row_count - 1 <
             service->input_counts.count) {
    const iree_host_size_t query_index =
        index - service->pending.count - service->row_count - 1;
    const loom_serve_text_input_count_request_t* query =
        &service->input_counts.requests[query_index];
    *out_snapshot = (loom_serve_http_request_snapshot_t){
        .id = query->receipt.serial,
        .phase =
            query_index == 0 && service->cpu_work == TEXT_CPU_WORK_INPUT_COUNT
                ? "count_input"
                : "queued",
        .arrival_time = query->receipt.arrival_time,
        .counters = {.requests = 1}};
  } else {
    return false;
  }
  return true;
}

static bool loom_serve_text_session_at(
    void* self, iree_host_size_t index,
    loom_serve_http_session_snapshot_t* out_snapshot) {
  const loom_serve_text_service_t* service = self;
  return loom_serve_text_session_cache_inspect(
      &service->cache, service->row_count, service->sessions, index,
      out_snapshot);
}

loom_serve_service_t loom_serve_text_service_interface(
    loom_serve_text_service_t* service) {
  static const iree_string_view_t modalities[] = {IREE_SVL("text")};
  static const iree_string_view_t image_modalities[] = {IREE_SVL("text"),
                                                        IREE_SVL("image")};
  const loom_serve_media_image_codec_t* const* codecs = NULL;
  const bool images =
      service->image &&
      loom_serve_text_model_chat_policy(service->model)->images.maximum_count &&
      loom_serve_media_image_codecs(&codecs);
  return (loom_serve_service_t){
      .name = service->name,
      .self = service,
      .set_observer = loom_serve_text_set_observer,
      .is_pending = loom_serve_text_is_pending,
      .is_active = loom_serve_text_is_active,
      .accept = loom_serve_text_accept,
      .try_open_text = loom_serve_text_open,
      .try_open_text_checkpoint = loom_serve_text_open_checkpoint,
      .advance = loom_serve_text_advance,
      .tokenizer = loom_serve_text_model_tokenizer(service->model),
      .kind = LOOM_SERVE_MODEL_KIND_TEXT,
      .endpoint = loom_serve_model_kind_endpoint(LOOM_SERVE_MODEL_KIND_TEXT),
      .capabilities =
          {.input_modalities = {images ? IREE_ARRAYSIZE(image_modalities)
                                       : IREE_ARRAYSIZE(modalities),
                                images ? image_modalities : modalities},
           .output_modalities = {IREE_ARRAYSIZE(modalities), modalities},
           .context_length = service->scheduling.context_capacity,
           .max_output_tokens =
               iree_min(service->scheduling.context_capacity,
                        LOOM_SERVE_TEXT_CHAT_MAX_OUTPUT_TOKENS),
           .input_token_count = true},
      .snapshot = loom_serve_text_snapshot,
      .inspection = {.request_at = loom_serve_text_request_at,
                     .session_at = loom_serve_text_session_at},
      .reconfiguration = loom_serve_execution_reconfiguration_interface(
          service->reconfiguration),
      .telemetry = {.stream = service->telemetry.stream,
                    .counters = &service->telemetry.counters,
                    .latency = &service->telemetry.latency},
      .capacity_domain = loom_serve_residency_cache(
          loom_serve_text_model_residency(service->model))};
}
