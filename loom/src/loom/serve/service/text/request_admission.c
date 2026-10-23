// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/request_admission.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "loom/serve/service/text/pending_continuation.h"
#include "loom/serve/service/text/reservation.h"

// Publication reads only owner-managed entries, never borrowed row/VM state.
void loom_serve_text_pending_observe(loom_serve_text_service_t* service) {
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

void loom_serve_text_pending_cancel(loom_serve_text_service_t* service,
                                    iree_host_size_t index) {
  loom_serve_text_pending_queue_cancel(&service->pending, index,
                                       service->projector);
  loom_serve_text_pending_observe(service);
}

void loom_serve_text_pending_cancel_failed(loom_serve_text_service_t* service,
                                           bool* out_progress) {
  const bool candidate_borrowed =
      service->cpu_work == TEXT_CPU_WORK_PREPARE ||
      service->completion.work == TEXT_WORK_RESERVE ||
      service->completion.work == TEXT_WORK_RETAIN;
  if (loom_serve_text_pending_queue_cancel_failed(
          &service->pending,
          candidate_borrowed ? service->pending.prepared.request : NULL,
          service->cpu_work == TEXT_CPU_WORK_PREPARE, service->projector)) {
    *out_progress = true;
    loom_serve_text_pending_observe(service);
  }
}

// Reclaim completed ownership at an epoch boundary or after admission denial.
// Compaction itself admits its temporary destinations and may leave a sparse
// map when moving would exceed the shared physical budget.
iree_status_t loom_serve_text_submit_trim(loom_serve_text_service_t* service) {
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

iree_status_t loom_serve_text_complete_trim(
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

bool loom_serve_text_cache_reclaim_pending(void* self) {
  const loom_serve_text_service_t* service = self;
  return service->pool.trim.pending;
}

bool loom_serve_text_cache_reclaim_request(void* self) {
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
  // Shared working-set pressure; preserve this model's admission turn.
  TEXT_ADMISSION_CAPACITY = 0,
  // Request-specific logical state cannot fit; another request may.
  TEXT_ADMISSION_STATE_CAPACITY,
  // Native retirement or an older admission owner can make progress.
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
    // Sparse backing is already valid for admission. Fitting requests retain
    // their completion credit while maintenance accumulates across the cohort.
    loom_serve_text_reserve_result_t admission;
    loom_serve_execution_frontier_t frontier;
    loom_serve_text_pending_continuation_t* continuation =
        &service->pending.prepared.request->continuation;
    status = continuation->session
                 ? loom_serve_text_reservation_submit(
                       &continuation->reservation, &admission, &frontier)
                 : loom_serve_text_row_submit_reserve(selected->row, mode,
                                                      checkpoint, reservation,
                                                      &admission, &frontier);
    if (iree_status_is_ok(status) &&
        (admission == LOOM_SERVE_TEXT_RESERVE_CAPACITY ||
         admission == LOOM_SERVE_TEXT_RESERVE_DEFERRED)) {
      // Shared admission owns physical pressure and idle-cache reclamation.
      // No parameter activation or partial state reservation was submitted.
      *out_result = admission == LOOM_SERVE_TEXT_RESERVE_DEFERRED
                        ? TEXT_ADMISSION_QUEUED
                        : TEXT_ADMISSION_CAPACITY;
      status = loom_serve_text_submit_trim(service);
      break;
    }
    *out_result = TEXT_ADMISSION_STATE_CAPACITY;
    if (iree_status_is_ok(status) &&
        admission == LOOM_SERVE_TEXT_RESERVE_ACQUIRED) {
      // The previous frontier was consumed successfully before this call. An
      // unchanged pair needs no new wait.
      const loom_serve_execution_frontier_t previous =
          service->completion.frontier;
      const bool complete = previous.semaphores[0] == frontier.semaphores[0] &&
                            previous.semaphores[1] == frontier.semaphores[1] &&
                            previous.values[0] == frontier.values[0] &&
                            previous.values[1] == frontier.values[1];
      service->completion.frontier = frontier;
      if (complete) {
        loom_serve_text_row_complete_reserve(selected->row);
        *out_result = TEXT_ADMISSION_READY;
      } else {
        status = loom_serve_text_await_work(service, TEXT_WORK_RESERVE);
        break;
      }
    }
    if (!iree_status_is_ok(status) || *out_result == TEXT_ADMISSION_READY) {
      break;
    }
    // Pending local maintenance contributes to shared reclamation readiness.
    // Submit it before parking, or this request could wait on its own trim.
    status = loom_serve_text_submit_trim(service);
    if (!iree_status_is_ok(status) ||
        service->completion.work != TEXT_WORK_NONE) {
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
  if (pending->continuation.session) {
    service->pending.prepared.request = pending;
    service->pending.prepared.index = index;
    service->pending.prepared.session = pending->continuation.session;
    service->pending.prepared.cached = pending->continuation.cached;
    service->pending.prepared.prefix = pending->continuation.prefix;
    service->pending.prepared.restored_bytes =
        pending->continuation.restored_bytes;
    return true;
  }
  loom_serve_text_session_t* session = loom_serve_text_session_select(
      service->row_count, service->sessions, pending->name,
      pending->chat.policy && !pending->checkpoint ? &pending->chat : NULL);
  if (!session || session->preparing) {
    return false;
  }
  service->pending.prepared.request = pending;
  service->pending.prepared.index = index;
  service->pending.prepared.session = session;
  service->pending.prepared.cached =
      loom_serve_text_session_cache_find(&service->cache, pending->name);
  const loom_serve_text_input_prefix_t prefix =
      loom_serve_text_pending_queue_prefix(&service->pending, session);
  const bool current =
      session->mode == LOOM_SERVE_TEXT_MODE_ENCODE
          ? pending->input.tokens &&
                ((!pending->sequence.append && !pending->checkpoint) ||
                 (prefix.completion &&
                  prefix.position == pending->sequence.position))
          : loom_serve_text_input_is_current(&pending->input, &pending->chat,
                                             prefix);
  if (current) {
    return true;
  }
  if (service->completion.work == TEXT_WORK_NONE) {
    loom_serve_text_observe(service, "prepare");
  }
  loom_serve_text_schedule_cpu(service, TEXT_CPU_WORK_PREPARE,
                               loom_serve_text_prepare_admission);
  return true;
}

bool loom_serve_text_begin_preparation(loom_serve_text_service_t* service) {
  return loom_serve_text_begin_preparation_at(
      service, service->pending.admission.cursor);
}

static iree_status_t loom_serve_text_retain_selected(
    loom_serve_text_service_t* service, bool* out_ready) {
  *out_ready = true;
  if (service->pending.prepared.request->continuation.session) {
    return iree_ok_status();
  }
  loom_serve_text_session_t* session = service->pending.prepared.session;
  if (!service->cache.capacity || !session->lifetime.name[0] ||
      !loom_serve_text_session_has_history(session) ||
      iree_string_view_equal(service->pending.prepared.request->name,
                             iree_make_cstring_view(session->lifetime.name))) {
    return iree_ok_status();
  }
  *out_ready = false;
  if (service->cache.count > service->cache.capacity) {
    return iree_ok_status();
  }
  if (service->cache.count == service->cache.capacity &&
      !service->pending.prepared.cached) {
    loom_serve_text_session_cache_entry_t* oldest =
        loom_serve_text_session_cache_oldest(&service->cache, NULL);
    if (!oldest) {
      return iree_ok_status();
    }
    loom_serve_text_session_cache_release(&service->cache, oldest,
                                          &service->telemetry);
    service->pool.trim.pending = true;
  }
  loom_serve_text_observe(service, "retain");
  IREE_RETURN_IF_ERROR(loom_serve_text_session_cache_submit(
      &service->cache, session, &service->pending.prepared.detaching,
      &service->completion.frontier));
  return loom_serve_text_await_work(service, TEXT_WORK_RETAIN);
}

void loom_serve_text_complete_retention(loom_serve_text_service_t* service) {
  loom_serve_text_session_t* session = service->pending.prepared.session;
  loom_serve_text_session_cache_publish(service->pending.prepared.detaching,
                                        session);
  service->pending.prepared.detaching = NULL;
  loom_serve_text_row_reset_retired(session->row);
  service->pool.trim.pending = true;
}

// Resolves prefix elision before starting the independent image component.
// Image preparation holds logical consumer capacity, not language parameters.
static iree_status_t loom_serve_text_prepare_media(
    loom_serve_text_service_t* service, bool* out_ready,
    bool* out_state_capacity) {
  *out_ready = true;
  *out_state_capacity = false;
  if (service->scheduling.model_mode == LOOM_SERVE_TEXT_MODE_ENCODE) {
    return iree_ok_status();
  }
  loom_serve_text_pending_request_t* pending =
      service->pending.prepared.request;
  if (!pending->continuation.session && !pending->replay_prefix &&
      !pending->input.retained_count && !pending->checkpoint) {
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
  if (!pending->continuation.session) {
    // Prepared media consumers form a leading contiguous cohort. Reserving
    // past an unprepared request could consume its completion capacity while
    // FIFO/overtake policy prevents the later consumer from releasing it.
    for (iree_host_size_t i = 0; i < service->pending.prepared.index; ++i) {
      if (!service->pending.values[i]->continuation.session) {
        return iree_ok_status();
      }
    }
    // Tokenization may finish alongside an epoch. Protect the native
    // continuation only after that epoch returns the shared model owner.
    if (service->completion.work != TEXT_WORK_NONE) {
      return iree_ok_status();
    }
    bool retained = false;
    IREE_RETURN_IF_ERROR(loom_serve_text_retain_selected(service, &retained));
    if (!retained) {
      *out_state_capacity = service->completion.work == TEXT_WORK_NONE;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_submit_trim(service));
    if (service->completion.work != TEXT_WORK_NONE) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_pending_continuation_prepare(
        service->pending.prepared.session, service->pending.prepared.cached,
        service->pending.prepared.prefix,
        pending->checkpoint ? pending->checkpoint->state : NULL,
        pending->input.retained_count, pending->reservation,
        &pending->continuation));
    if (!pending->continuation.session) {
      *out_state_capacity = true;
      return iree_ok_status();
    }
    service->pending.prepared.restored_bytes =
        pending->continuation.restored_bytes;
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
  loom_serve_text_media_projector_offer(
      service->projector, pending->media,
      loom_serve_text_reservation_memory_footprint(
          &pending->continuation.reservation));
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

iree_status_t loom_serve_text_complete_preparation(
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
      case IREE_STATUS_NOT_FOUND:
      case IREE_STATUS_FAILED_PRECONDITION:
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
    const loom_serve_text_admission_record_t value = {
        .row = service->pending.prepared.session - service->sessions,
        .retained_tokens = pending->input.retained_count,
        .appended_tokens = pending->input.count,
        .reservation_tokens = pending->reservation,
        .max_tokens = pending->chat.max_tokens,
        .prepare_ns = pending->prepare_duration};
    loom_serve_text_record(service->telemetry.stream, &LOOM_SERVE_TEXT_PREPARED,
                           pending->receipt.serial, &value, sizeof(value));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_admit_request(
    loom_serve_text_service_t* service, loom_serve_service_flags_t flags) {
  loom_serve_text_pending_request_t* pending =
      service->pending.prepared.request;
  if (iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_STOPPING) ||
      !pending->consumer.vtable->is_open(pending->consumer.self)) {
    loom_serve_text_reset_session(service, service->pending.prepared.session);
    loom_serve_text_pending_cancel(service, service->pending.prepared.index);
    return iree_ok_status();
  }
  // Detach obsolete media loans before moving request ownership to the row.
  loom_serve_text_media_projector_cancel(service->projector, pending->media);
  if (service->pending.prepared.prefix) {
    loom_serve_text_prefix_cache_touch(&service->prefixes,
                                       service->pending.prepared.prefix);
  }
  const iree_status_t status = loom_serve_text_pending_queue_admit(
      &service->pending, &service->cache, service->sessions, service->name);
  loom_serve_text_pending_observe(service);
  return status;
}

iree_status_t loom_serve_text_complete_reservation(
    loom_serve_text_service_t* service, loom_serve_service_flags_t flags) {
  loom_serve_text_row_complete_reserve(service->pending.prepared.session->row);
  return loom_serve_text_admit_request(service, flags);
}

iree_status_t loom_serve_text_admit_pending(loom_serve_text_service_t* service,
                                            loom_serve_service_flags_t flags,
                                            bool allow_preparation) {
  // Cancellation can leave a completed full-cache exchange without its
  // incoming admission. Settle that excess here, never in terminal teardown's
  // host-only pending removal while device ownership may remain unresolved.
  if (!service->pending.prepared.session &&
      service->cache.count > service->cache.capacity) {
    // A media continuation can own the incoming endpoint between selected
    // attempts. Its eventual admission consumes that entry and settles the
    // exchange without evicting the outgoing history.
    bool incoming = false;
    for (iree_host_size_t i = 0; i < service->pending.count; ++i) {
      incoming |= service->pending.values[i]->continuation.cached != NULL;
    }
    loom_serve_text_session_cache_entry_t* oldest =
        incoming ? NULL
                 : loom_serve_text_session_cache_oldest(&service->cache, NULL);
    if (oldest) {
      loom_serve_text_session_cache_release(&service->cache, oldest,
                                            &service->telemetry);
      service->pool.trim.pending = true;
    }
  }
  iree_status_t status = iree_ok_status();
  while (service->pending.count && iree_status_is_ok(status)) {
    if (!service->pending.prepared.session) {
      if (allow_preparation) {
        loom_serve_text_begin_preparation(service);
      }
      if (!service->pending.prepared.session ||
          loom_serve_text_has_work(service)) {
        break;
      }
    }
    loom_serve_text_pending_request_t* pending =
        service->pending.prepared.request;
    loom_serve_text_session_t* session = service->pending.prepared.session;
    const iree_host_size_t retained_count = pending->input.retained_count;
    const iree_host_size_t reservation = pending->reservation;
    bool media_ready = true;
    bool state_capacity = false;
    status =
        loom_serve_text_prepare_media(service, &media_ready, &state_capacity);
    if (!iree_status_is_ok(status)) {
      status = loom_serve_text_reject_prepared(service, status);
      continue;
    }
    if (service->completion.work != TEXT_WORK_NONE) {
      break;
    }
    if (service->pending.prepared.index != service->pending.admission.cursor) {
      // Lookahead may reserve a future consumer and gather its images, but it
      // cannot admit text work or spend admission-policy overtaking credit.
      if (state_capacity && service->projector) {
        service->media_preparation_end = 0;
        loom_serve_text_media_projector_start(service->projector);
      }
      memset(&service->pending.prepared, 0, sizeof(service->pending.prepared));
      continue;
    }
    if (!media_ready && !state_capacity) {
      memset(&service->pending.prepared, 0, sizeof(service->pending.prepared));
      if (loom_serve_text_admission_blocked(&service->pending.admission,
                                            service->pending.count)) {
        continue;
      }
      break;
    }
    bool retained = false;
    if (!state_capacity) {
      status = loom_serve_text_retain_selected(service, &retained);
      state_capacity = !retained;
    }
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
    if (!pending->continuation.session) {
      service->pending.prepared.restored_bytes =
          checkpoint ? loom_serve_text_checkpoint_suspended_bytes(checkpoint)
                     : 0;
    }
    loom_serve_text_admission_result_t admission =
        TEXT_ADMISSION_STATE_CAPACITY;
    if (!state_capacity) {
      status =
          loom_serve_text_reclaim_idle(service, session, checkpoint,
                                       retained_count, reservation, &admission);
    }
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
        loom_serve_text_pending_continuation_deinitialize(
            &pending->continuation);
        if (!service->pending.prepared.prefix->pending) {
          loom_serve_text_prefix_cache_release(
              &service->prefixes, service->pending.prepared.prefix);
        }
        service->pending.prepared.prefix = NULL;
        pending->replay_prefix = true;
        service->pool.trim.pending = true;
        continue;
      }
      if (state_capacity &&
          (loom_serve_text_prefix_cache_evict(&service->prefixes, NULL) ||
           loom_serve_text_session_cache_evict(
               &service->cache, service->row_count, service->sessions, session,
               service->pending.prepared.cached, &service->telemetry))) {
        service->pool.trim.pending = true;
        continue;
      }
      if (state_capacity && service->projector) {
        service->media_preparation_end = 0;
        loom_serve_text_media_projector_start(service->projector);
      }
      bool active =
          iree_any_bit_set(flags, LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY);
      active |= loom_serve_text_media_projector_is_pending(service->projector);
      for (iree_host_size_t i = 0; i < service->pending.count; ++i) {
        active |= service->pending.values[i] != pending &&
                  service->pending.values[i]->continuation.session != NULL;
      }
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

void loom_serve_text_pending_push(
    loom_serve_text_service_t* service, loom_serve_text_consumer_t consumer,
    iree_string_view_t body, iree_string_view_t name,
    loom_serve_text_checkpoint_entry_t* checkpoint,
    const loom_serve_text_sequence_t* sequence,
    loom_serve_text_receipt_t receipt) {
  loom_serve_text_pending_queue_push(&service->pending, consumer, body, name,
                                     checkpoint, sequence, receipt);
  loom_serve_text_pending_observe(service);
  // Both ingress paths can prepare the oldest queued input while an accepted
  // epoch owns different rows. Scheduling this finite loan cannot fail or
  // change the accepted request's ownership.
  if (service->completion.work == TEXT_WORK_EPOCH &&
      service->cpu_work == TEXT_CPU_WORK_NONE) {
    loom_serve_text_begin_preparation(service);
  }
}

// Lookahead prepares only the already-observed queue window. It never changes
// the admission cursor or spends/replenishes overtaking credit.
bool loom_serve_text_media_lookahead(loom_serve_text_service_t* service) {
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
    if (request->continuation.session) {
      if (service->projector) {
        loom_serve_text_media_projector_offer(
            service->projector, request->media,
            loom_serve_text_reservation_memory_footprint(
                &request->continuation.reservation));
      }
      continue;
    }
    if ((!request->input.tokens || request->media) &&
        loom_serve_text_begin_preparation_at(service, i)) {
      return true;
    }
    // An earlier text-only, unavailable-row, or capacity-blocked request is
    // a cohort boundary, not a gap for later consumers to reserve across.
    break;
  }
  if (!service->projector) {
    return false;
  }
  service->media_preparation_end = 0;
  return loom_serve_text_media_projector_start(service->projector);
}
