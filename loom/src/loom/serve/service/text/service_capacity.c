// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_capacity.h"

#include <string.h>

#include "loom/serve/service/text/service_owner.h"

static iree_status_t loom_serve_text_service_capacity_run(void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_serve_text_service_t* service = user_data;
  loom_serve_text_service_capacity_t* growth = &service->growth;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loom_serve_text_model_prepare_resize(
              service->model, growth->requested, &growth->resize));
  if (!growth->resize) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_array(
              service->host_allocator, growth->requested,
              sizeof(*growth->sessions), (void**)&growth->sessions));
  memcpy(growth->sessions, service->sessions,
         service->row_count * sizeof(*growth->sessions));
  growth->count = service->row_count;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_array(service->host_allocator,
                                      growth->requested, sizeof(*growth->ready),
                                      (void**)&growth->ready));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = growth->count;
       i < growth->requested && iree_status_is_ok(status); ++i) {
    status = iree_arena_allocate(&service->session_records,
                                 sizeof(**growth->sessions),
                                 (void**)&growth->sessions[i]);
    if (iree_status_is_ok(status)) {
      ++growth->count;
      status = loom_serve_text_session_initialize(
          service->scheduling.model_mode,
          loom_serve_text_model_tokenizer(service->model),
          service->scheduling.context_capacity, service->host_allocator,
          growth->sessions[i]);
    }
  }
  IREE_RETURN_AND_END_ZONE(z0, status);
}

bool loom_serve_text_service_capacity_begin(
    loom_serve_text_service_t* service) {
  const iree_host_size_t maximum =
      service->max_sessions ? iree_min(service->max_sessions, INT32_MAX)
                            : INT32_MAX;
  if (loom_serve_text_has_work(service) || service->pending.prepared.session ||
      service->row_count == maximum) {
    return false;
  }
  for (iree_host_size_t i = 0; i < service->pending.count; ++i) {
    if (service->pending.values[i]->continuation.session) {
      // A media continuation has captured address-dependent restore plans.
      // Its ordinary admission/cancellation releases that native borrower.
      return false;
    }
  }
  service->growth.requested = iree_min(2 * service->row_count, maximum);
  loom_serve_text_observe(service, "grow");
  loom_serve_text_schedule_cpu(service, TEXT_CPU_WORK_GROW,
                               loom_serve_text_service_capacity_run);
  return true;
}

iree_status_t loom_serve_text_service_capacity_await(
    loom_serve_text_service_t* service) {
  if (!service->growth.resize) {
    loom_serve_text_service_capacity_discard(service);
    return iree_ok_status();
  }
  const iree_status_t status = loom_serve_text_model_submit_resize(
      service->model, service->growth.resize, &service->completion.frontier);
  loom_serve_text_resize_free(service->growth.resize);
  service->growth.resize = NULL;
  IREE_RETURN_IF_ERROR(status);
  for (iree_host_size_t i = service->row_count; i < service->growth.count;
       ++i) {
    service->growth.sessions[i]->row =
        loom_serve_text_model_row(service->model, i);
  }
  return loom_serve_text_await_work(service, TEXT_WORK_GROW);
}

void loom_serve_text_service_capacity_publish(
    loom_serve_text_service_t* service) {
  loom_serve_text_model_complete_resize(service->model);
  iree_allocator_free(service->host_allocator, service->sessions);
  iree_allocator_free(service->host_allocator, service->ready);
  service->sessions = service->growth.sessions;
  service->ready = service->growth.ready;
  service->row_count = service->growth.count;
  service->growth = (loom_serve_text_service_capacity_t){0};
}

void loom_serve_text_service_capacity_discard(
    loom_serve_text_service_t* service) {
  loom_serve_text_resize_free(service->growth.resize);
  for (iree_host_size_t i = service->row_count; i < service->growth.count;
       ++i) {
    loom_serve_text_session_deinitialize(service->growth.sessions[i]);
  }
  iree_allocator_free(service->host_allocator, service->growth.sessions);
  iree_allocator_free(service->host_allocator, service->growth.ready);
  service->growth = (loom_serve_text_service_capacity_t){0};
}
