// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_owner.h"

#include <string.h>

#include "iree/base/threading/mutex.h"

bool loom_serve_text_has_work(const loom_serve_text_service_t* service) {
  return service->completion.work != TEXT_WORK_NONE ||
         service->cpu_work != TEXT_CPU_WORK_NONE;
}

iree_status_t loom_serve_text_await_work(loom_serve_text_service_t* service,
                                         loom_serve_text_work_t work) {
  service->completion.start_time = iree_time_now();
  IREE_RETURN_IF_ERROR(loom_serve_text_readiness_await(
      &service->readiness, &service->completion.frontier));
  service->completion.work = work;
  IREE_TRACE({
    if (work == TEXT_WORK_EPOCH) {
      service->completion.section = loom_serve_trace_section_enter(
          LOOM_SERVE_TRACE_SECTION_EXECUTE, service->name);
    }
  });
  return iree_ok_status();
}

void loom_serve_text_schedule_cpu(loom_serve_text_service_t* service,
                                  loom_serve_text_cpu_work_t work,
                                  loom_serve_text_cpu_work_fn_t run) {
  service->cpu_work = work;
  loom_serve_text_readiness_schedule(&service->readiness, run, service);
}

void loom_serve_text_observe(loom_serve_text_service_t* service,
                             const char* phase) {
  const loom_serve_text_pool_usage_t pool =
      loom_serve_text_model_pool_usage(service->model);
  iree_host_size_t active = 0, prefill = 0, decode = 0, backpressured = 0;
  iree_time_t oldest_active_arrival = IREE_TIME_INFINITE_FUTURE;
  for (iree_host_size_t i = 0; i < service->row_count; ++i) {
    const loom_serve_text_session_t* session = service->sessions[i];
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
  state->session_capacity = service->row_count;
  state->max_sessions = service->max_sessions;
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
