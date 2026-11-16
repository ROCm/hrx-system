// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_owner.h"

#include <string.h>

bool loom_serve_text_has_work(const loom_serve_text_service_t* service) {
  return service->completion.work != TEXT_WORK_NONE ||
         service->cpu_work != TEXT_CPU_WORK_NONE;
}

iree_status_t loom_serve_text_await_work(loom_serve_text_service_t* service,
                                         loom_serve_text_work_t work) {
  service->completion.start_time = iree_time_now();
  IREE_RETURN_IF_ERROR(loom_runtime_execution_await(
      &service->readiness, &service->completion.frontier,
      &service->device_wait));
  service->completion.work = work;
  IREE_TRACE({
    if (work == TEXT_WORK_EPOCH) {
      service->completion.section = loom_service_trace_section_enter(
          LOOM_SERVICE_TRACE_SECTION_EXECUTE, service->name);
    }
  });
  return iree_ok_status();
}

void loom_serve_text_schedule_cpu(loom_serve_text_service_t* service,
                                  loom_serve_text_cpu_work_t work,
                                  loom_runtime_cpu_work_fn_t run) {
  service->cpu_work = work;
  loom_runtime_readiness_schedule(&service->readiness, run, service,
                                  &service->cpu_task);
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
  loom_service_text_statistics_t* state = &service->heartbeat.snapshot;
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
  const loom_serve_text_cache_selection_t cache =
      loom_serve_text_model_cache_selection(service->model);
  state->cache_target_bytes = cache.target_bytes;
  state->cache_additional_bytes = cache.additional_bytes;
  state->state_memory = loom_serve_text_model_memory_statistics(service->model);
  state->parameter_memory =
      loom_serve_text_model_weight_statistics(service->model);
  state->parameter_activity =
      loom_serve_text_model_parameter_activity(service->model);
  const loom_runtime_pool_statistics_t pool_memory =
      loom_runtime_pool_statistics(loom_serve_text_model_pool(service->model));
  state->pool_memory = (loom_runtime_storage_shared_memory_record_t){
      .retained_committed_bytes = pool_memory.retained.committed_bytes,
      .workspace_committed_bytes = pool_memory.workspace.bytes_committed,
      .workspace_live_bytes = pool_memory.workspace.bytes_reserved,
      .workspace_slab_count = pool_memory.workspace.slab_count,
      .workspace_reuse_count = pool_memory.workspace.reuse_count};
  state->prefill_rows = prefill;
  state->decode_rows = decode;
  state->backpressured_rows = backpressured;
  iree_slim_mutex_unlock(&service->heartbeat.mutex);
}

loom_runtime_service_operation_key_t loom_serve_text_operation_begin(
    loom_serve_text_service_t* service, loom_serve_text_operation_slot_t slot,
    loom_runtime_service_work_flags_t work,
    loom_runtime_service_offer_key_t origin) {
  IREE_ASSERT(service->scheduling.next_serial != UINT64_MAX);
  loom_runtime_service_operation_t* operation =
      &service->scheduling.operations[slot];
  *operation = (loom_runtime_service_operation_t){
      .key = {service->scheduling.binding.operation_ordinal + slot,
              ++service->scheduling.next_serial},
      .origin = origin,
      .work = work,
      .devices = slot == TEXT_OPERATION_INPUT_COUNT
                     ? 0
                     : UINT64_C(1) << service->scheduling.device_ordinal,
      .use_count = slot == TEXT_OPERATION_INPUT_COUNT ? 0 : 1};
  return operation->key;
}

void loom_serve_text_operation_end(loom_serve_text_service_t* service,
                                   loom_serve_text_operation_slot_t slot) {
  if (iree_status_is_ok(service->telemetry.failure)) {
    service->scheduling.operations[slot] = (loom_runtime_service_operation_t){
        .key.ordinal = service->scheduling.binding.operation_ordinal + slot};
  }
}

void loom_serve_text_withdraw_choice(loom_serve_text_service_t* service) {
  if (service->scheduling.choice == TEXT_CHOICE_NONE) {
    return;
  }
  IREE_ASSERT(service->scheduling.offer.key.revision != UINT64_MAX);
  ++service->scheduling.offer.key.revision;
  service->scheduling.offer.work = 0;
  service->scheduling.offer.choice_count = 0;
  service->scheduling.choice = TEXT_CHOICE_NONE;
}

void loom_serve_text_state_changed(loom_serve_text_service_t* service) {
  IREE_ASSERT(service->scheduling.state_revision != UINT64_MAX);
  ++service->scheduling.state_revision;
  loom_serve_text_withdraw_choice(service);
}

void loom_serve_text_publish_control(loom_serve_text_service_t* service) {
  const loom_runtime_service_control_binding_t binding =
      service->scheduling.binding;
  const loom_runtime_service_use_t use = {
      .resource_ordinal = service->scheduling.device_ordinal};
  if (service->scheduling.offer.key.revision !=
      service->scheduling.published_revision) {
    const loom_runtime_service_choice_publication_t choice = {
        .choice = {.instance_ordinal = binding.instance_ordinal,
                   .request_count = service->scheduling.choice_request_count,
                   .devices = service->scheduling.choice_devices,
                   .use_count = service->scheduling.choice_devices ? 1 : 0},
        .uses = &use};
    loom_runtime_service_publish_offer(binding.control,
                                       &service->scheduling.offer, &choice);
    service->scheduling.published_revision =
        service->scheduling.offer.key.revision;
  }
  uint64_t pending_operations = 0, accepted_devices = 0;
  for (iree_host_size_t i = 0; i < TEXT_OPERATION_COUNT; ++i) {
    loom_runtime_service_operation_t* operation =
        &service->scheduling.operations[i];
    if (operation->key.serial) {
      ++pending_operations;
      accepted_devices |= operation->devices;
      if (!iree_status_is_ok(service->telemetry.failure)) {
        operation->waits = 0;
      } else if (i == TEXT_OPERATION_INPUT_COUNT) {
        operation->waits = LOOM_RUNTIME_SERVICE_WAIT_CPU;
      } else if (i == TEXT_OPERATION_ADMISSION) {
        if (service->cpu_work == TEXT_CPU_WORK_PREPARE ||
            service->cpu_work == TEXT_CPU_WORK_GROW) {
          operation->waits = LOOM_RUNTIME_SERVICE_WAIT_CPU;
        } else if (service->completion.work == TEXT_WORK_RESERVE ||
                   service->completion.work == TEXT_WORK_RETAIN ||
                   service->completion.work == TEXT_WORK_GROW) {
          operation->waits = LOOM_RUNTIME_SERVICE_WAIT_DEVICE;
        } else {
          operation->waits = LOOM_RUNTIME_SERVICE_WAIT_CAPACITY;
        }
      } else {
        operation->waits = service->cpu_work == TEXT_CPU_WORK_OUTPUT
                               ? LOOM_RUNTIME_SERVICE_WAIT_CPU
                               : LOOM_RUNTIME_SERVICE_WAIT_DEVICE;
      }
    }
    if (memcmp(operation, &service->scheduling.published[i],
               sizeof(*operation))) {
      loom_runtime_service_publish_operation(binding.control, operation, &use);
      service->scheduling.published[i] = *operation;
    }
  }
  const loom_service_telemetry_counters_t* counters =
      &service->telemetry.counters;
  const loom_runtime_service_observation_t projection =
      service->scheduling.projection;
  const loom_runtime_service_observation_t management =
      service->scheduling.management;
  const loom_runtime_service_observation_t observation = {
      .owned_request_count = counters->requests - counters->completed -
                             counters->rejected - counters->cancelled -
                             counters->failed,
      .waiting_request_count =
          service->pending.count - (service->pending.prepared.session != NULL) +
          service->input_counts.count -
          (service->cpu_work == TEXT_CPU_WORK_INPUT_COUNT) +
          (service->control.consumer.self && !service->scheduling.checkpoint) +
          service->scheduling.generation_count -
          service->scheduling.epoch_request_count,
      .offer_count =
          service->scheduling.offer.choice_count + projection.offer_count,
      .pending_operation_count = pending_operations +
                                 management.pending_operation_count +
                                 projection.pending_operation_count,
      .offered_devices = (service->scheduling.offer.choice_count
                              ? service->scheduling.choice_devices
                              : 0) |
                         projection.offered_devices,
      .accepted_devices = accepted_devices | management.accepted_devices |
                          projection.accepted_devices};
  if (memcmp(&observation, &service->scheduling.observation,
             sizeof(observation))) {
    binding.publish_observation(binding.publisher, &observation);
    service->scheduling.observation = observation;
  }
}
