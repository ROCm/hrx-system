// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/finite/component.h"

#include "iree/hal/api.h"

struct loom_service_finite_component_t {
  // Allocator owning the handle and its preparation operation.
  iree_allocator_t host_allocator;
  // Borrowed execution device, surviving final model retirement.
  loom_runtime_device_t* device;
  // Borrowed storage owner, preserved across lazy preparation and reload.
  loom_runtime_pool_t* pool;
  // Borrowed definition and package, independent of any requesting connection.
  loom_service_finite_model_options_t options;
  // Immutable owner wake target, live until all callbacks have retired.
  loom_runtime_wake_callback_t observer;
  // Exclusive retained observation slots and the composition's summary owner.
  loom_runtime_service_control_binding_t scheduling;
  // Immutable placement, independent of prepared model storage.
  uint64_t device_ordinal;
  // Shared preparation demand; direct grants use the same legal identity.
  loom_runtime_service_offer_t offer;
  // Accepted preparation or release, retained through actual cleanup.
  loom_runtime_service_operation_t operation;
  // Monotonic accepted identities, including fixed-backing trim/reload cycles.
  uint64_t next_serial;
  // Owner-only lifecycle; only READY lends a model to an invocation.
  loom_service_finite_component_phase_t phase;
  // Monotonic start of the current PREPARING interval; phase owns its validity.
  iree_time_t preparation_start;
  // Completed PREPARING intervals, including attempts before a fixed trim.
  uint64_t preparation_duration_ns;
  // Irreversible deployment shutdown, independent of request cancellation.
  bool stopping;
  // Persistent eviction policy, including before lazy preparation completes.
  bool pinned;
  // Owned asynchronous load or terminal load whose cleanup failed.
  loom_service_finite_load_t* load;
  // Owned prepared model, retained through actual queue ownership retirement.
  loom_service_finite_model_t* model;
  // Owned terminal preparation error, cloned only for affected consumers.
  iree_status_t status;
  // Cumulative evidence independent of load/model destruction.
  loom_runtime_compilation_report_t report;
};

static void loom_service_finite_component_publish(
    loom_service_finite_component_t* component) {
  const loom_runtime_service_observation_t observation = {
      .offer_count = component->offer.choice_count,
      .pending_operation_count = component->operation.key.serial != 0,
      .offered_devices = component->offer.choice_count
                             ? UINT64_C(1) << component->device_ordinal
                             : 0,
      .accepted_devices = component->operation.devices};
  component->scheduling.publish_observation(component->scheduling.publisher,
                                            &observation);
}

static void loom_service_finite_component_publish_offer(
    loom_service_finite_component_t* component) {
  const loom_runtime_service_use_t use = {
      .resource_ordinal = component->device_ordinal,
  };
  const loom_runtime_service_choice_publication_t choice = {
      .choice = {.instance_ordinal = component->scheduling.instance_ordinal,
                 .devices = UINT64_C(1) << component->device_ordinal,
                 .use_count = 1},
      .uses = &use};
  loom_runtime_service_publish_offer(component->scheduling.control,
                                     &component->offer, &choice);
}

static bool loom_service_finite_component_withdraw(
    loom_service_finite_component_t* component) {
  if (!component->offer.choice_count) {
    return false;
  }
  IREE_ASSERT(component->offer.key.revision != UINT64_MAX);
  ++component->offer.key.revision;
  component->offer.choice_count = 0;
  loom_service_finite_component_publish_offer(component);
  return true;
}

static void loom_service_finite_component_begin_work(
    loom_service_finite_component_t* component,
    loom_runtime_service_work_flags_t work) {
  if (component->operation.key.serial) {
    // Mandatory cleanup of a failed attempt retains that accepted identity.
    return;
  }
  loom_runtime_service_offer_key_t origin = {
      .ordinal = component->scheduling.offer_ordinal,
  };
  if (component->offer.choice_count && component->offer.work == work) {
    origin = component->offer.key;
  }
  IREE_ASSERT(component->next_serial != UINT64_MAX);
  component->operation = (loom_runtime_service_operation_t){
      .key = {component->scheduling.operation_ordinal,
              ++component->next_serial},
      .origin = origin,
      .work = work,
      .devices = UINT64_C(1) << component->device_ordinal,
      .use_count = 1};
  loom_service_finite_component_withdraw(component);
  const loom_runtime_service_use_t use = {
      .resource_ordinal = component->device_ordinal,
  };
  loom_runtime_service_publish_operation(component->scheduling.control,
                                         &component->operation, &use);
  loom_service_finite_component_publish(component);
}

static void loom_service_finite_component_end_work(
    loom_service_finite_component_t* component) {
  if (!component->operation.key.serial) {
    return;
  }
  component->operation = (loom_runtime_service_operation_t){
      .key.ordinal = component->scheduling.operation_ordinal};
  loom_runtime_service_publish_operation(component->scheduling.control,
                                         &component->operation, NULL);
  loom_service_finite_component_publish(component);
}

static void loom_service_finite_component_set_phase(
    loom_service_finite_component_t* component,
    loom_service_finite_component_phase_t phase) {
  if (component->phase != phase &&
      (component->phase == LOOM_SERVICE_FINITE_COMPONENT_PREPARING ||
       phase == LOOM_SERVICE_FINITE_COMPONENT_PREPARING)) {
    const iree_time_t now = iree_time_now();
    if (component->phase == LOOM_SERVICE_FINITE_COMPONENT_PREPARING) {
      component->preparation_duration_ns += now - component->preparation_start;
    } else {
      component->preparation_start = now;
    }
  }
  component->phase = phase;
}

iree_status_t loom_service_finite_component_allocate(
    loom_runtime_device_t* device, loom_runtime_pool_t* pool,
    const loom_service_finite_model_options_t* options,
    loom_runtime_wake_callback_t observer,
    loom_runtime_service_control_binding_t scheduling,
    iree_allocator_t host_allocator,
    loom_service_finite_component_t** out_component) {
  *out_component = NULL;
  if (!scheduling.control || !scheduling.publish_observation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "finite component requires retained scheduling "
                            "storage and its summary publisher");
  }
  const loom_runtime_service_control_layout_t layout =
      loom_runtime_service_control_layout(scheduling.control);
  if (scheduling.offer_ordinal >= layout.offer_capacity ||
      scheduling.operation_ordinal >= layout.operation_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "finite component control lacks its owner slots");
  }
  const uint64_t device_ordinal =
      iree_hal_device_topology_info(loom_runtime_device_handle(device))
          ->topology_index;
  if (device_ordinal >= 64) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "component device exceeds scheduling device mask");
  }
  loom_service_finite_component_t* component = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*component),
                                             (void**)&component));
  component->host_allocator = host_allocator;
  component->device = device;
  component->pool = pool;
  component->options = *options;
  component->observer = observer;
  component->scheduling = scheduling;
  component->device_ordinal = device_ordinal;
  component->offer.key.ordinal = scheduling.offer_ordinal;
  component->operation.key.ordinal = scheduling.operation_ordinal;
  *out_component = component;
  return iree_ok_status();
}

loom_runtime_service_control_binding_t
loom_service_finite_component_scheduling_binding(
    const loom_service_finite_component_t* component) {
  return component->scheduling;
}

void loom_service_finite_component_request_preparation(
    loom_service_finite_component_t* component) {
  if (component->stopping ||
      component->phase != LOOM_SERVICE_FINITE_COMPONENT_DECLARED ||
      component->offer.choice_count) {
    return;
  }
  IREE_ASSERT(component->offer.key.revision != UINT64_MAX);
  ++component->offer.key.revision;
  component->offer.work = LOOM_RUNTIME_SERVICE_WORK_PREPARATION;
  component->offer.choice_count = 1;
  loom_service_finite_component_publish_offer(component);
  loom_service_finite_component_publish(component);
}

loom_service_finite_model_t* loom_service_finite_component_prepared_model(
    const loom_service_finite_component_t* component) {
  return component->phase == LOOM_SERVICE_FINITE_COMPONENT_READY
             ? component->model
             : NULL;
}

loom_runtime_service_schedule_outcome_t
loom_service_finite_component_try_schedule(
    loom_service_finite_component_t* component,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal) {
  loom_runtime_service_schedule_outcome_t outcome = {0};
  if (component->stopping || !component->offer.choice_count ||
      offer.ordinal != component->offer.key.ordinal ||
      offer.revision != component->offer.key.revision || choice_ordinal != 0) {
    return outcome;
  }
  loom_service_finite_component_begin_work(
      component, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
  outcome.accepted = true;
  outcome.operation = component->operation.key;
  loom_service_finite_model_t* model = NULL;
  iree_status_t status =
      loom_service_finite_component_prepare(component, &model);
  outcome.status_code = iree_status_code(status);
  iree_status_free(status);
  return outcome;
}

loom_service_finite_component_phase_t loom_service_finite_component_phase(
    const loom_service_finite_component_t* component) {
  return component->phase;
}

uint64_t loom_service_finite_component_preparation_duration_ns(
    const loom_service_finite_component_t* component) {
  return component->preparation_duration_ns +
         (component->phase == LOOM_SERVICE_FINITE_COMPONENT_PREPARING
              ? iree_time_now() - component->preparation_start
              : 0);
}

bool loom_service_finite_component_is_pending(
    const loom_service_finite_component_t* component) {
  return component &&
         (component->phase == LOOM_SERVICE_FINITE_COMPONENT_PREPARING ||
          loom_service_finite_model_is_pending(component->model));
}

const loom_runtime_compilation_report_t* loom_service_finite_component_report(
    const loom_service_finite_component_t* component) {
  return &component->report;
}

iree_status_t loom_service_finite_component_status(
    const loom_service_finite_component_t* component) {
  return component->status;
}

bool loom_service_finite_component_parameters(
    const loom_service_finite_component_t* component,
    loom_runtime_residency_snapshot_t* out_parameters) {
  *out_parameters = (loom_runtime_residency_snapshot_t){0};
  if (!component->model) {
    return false;
  }
  *out_parameters = loom_service_finite_model_parameters(component->model);
  return true;
}

void loom_service_finite_component_set_pinned(
    loom_service_finite_component_t* component, bool pinned) {
  component->pinned = pinned;
  if (component->model) {
    loom_service_finite_model_set_pinned(component->model, pinned);
  }
}

bool loom_service_finite_component_advance(
    loom_service_finite_component_t* component, bool stopping) {
  if (!component) {
    return false;
  }
  component->stopping |= stopping;
  const bool withdrawn =
      component->stopping && loom_service_finite_component_withdraw(component);
  if (withdrawn) {
    loom_service_finite_component_publish(component);
  }
  if (component->phase != LOOM_SERVICE_FINITE_COMPONENT_PREPARING) {
    const bool pending = loom_service_finite_model_is_pending(component->model);
    iree_status_t status =
        loom_service_finite_model_query_status(component->model);
    if (!iree_status_is_ok(status)) {
      component->status = iree_status_join(component->status, status);
      loom_service_finite_component_set_phase(
          component, LOOM_SERVICE_FINITE_COMPONENT_FAILED);
    }
    return withdrawn ||
           (pending && !loom_service_finite_model_is_pending(component->model));
  }
  bool complete = false;
  component->status = loom_service_finite_load_advance(
      component->load,
      component->stopping ? LOOM_SERVICE_FINITE_LOAD_FLAG_CANCEL : 0,
      &complete);
  if (!complete) {
    return false;
  }
  loom_runtime_compilation_merge(
      &component->report, loom_service_finite_load_report(component->load));
  if (iree_status_is_ok(component->status)) {
    component->model = loom_service_finite_load_take_model(component->load);
    loom_service_finite_model_set_pinned(component->model, component->pinned);
  }
  iree_status_t cleanup = loom_service_finite_load_free(component->load);
  if (iree_status_is_ok(cleanup)) {
    component->load = NULL;
  }
  component->status = iree_status_join(component->status, cleanup);
  loom_service_finite_component_set_phase(
      component, iree_status_is_ok(component->status)
                     ? LOOM_SERVICE_FINITE_COMPONENT_READY
                     : LOOM_SERVICE_FINITE_COMPONENT_FAILED);
  if (iree_status_is_ok(component->status) ||
      (!component->load && !component->model)) {
    loom_service_finite_component_end_work(component);
  }
  return true;
}

iree_status_t loom_service_finite_component_prepare(
    loom_service_finite_component_t* component,
    loom_service_finite_model_t** out_model) {
  *out_model = NULL;
  // A borrowed model must not escape a concurrent quiescent release or removal.
  IREE_ASSERT(!component->stopping &&
              component->phase != LOOM_SERVICE_FINITE_COMPONENT_RELEASING);
  if (component->phase == LOOM_SERVICE_FINITE_COMPONENT_DECLARED) {
    loom_service_finite_component_begin_work(
        component, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
    loom_service_finite_component_set_phase(
        component, LOOM_SERVICE_FINITE_COMPONENT_PREPARING);
    component->status = loom_service_finite_load_allocate(
        component->device, component->pool, &component->options,
        component->observer.wake, component->observer.user_data,
        component->host_allocator, &component->load);
    if (!iree_status_is_ok(component->status)) {
      loom_service_finite_component_set_phase(
          component, LOOM_SERVICE_FINITE_COMPONENT_FAILED);
      loom_service_finite_component_end_work(component);
    }
  }
  loom_service_finite_component_advance(component, false);
  if (component->phase == LOOM_SERVICE_FINITE_COMPONENT_READY) {
    *out_model = component->model;
  }
  return iree_status_clone(component->status);
}

static iree_status_t loom_service_finite_component_release_model(
    loom_service_finite_component_t* component, bool* out_complete) {
  *out_complete = false;
  if (component->model &&
      component->phase != LOOM_SERVICE_FINITE_COMPONENT_RELEASING) {
    loom_service_finite_model_retire(component->model, component->observer.wake,
                                     component->observer.user_data);
    loom_service_finite_component_set_phase(
        component, LOOM_SERVICE_FINITE_COMPONENT_RELEASING);
  }
  if (!loom_service_finite_model_is_retired(component->model)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_service_finite_model_free_retired(component->model));
  component->model = NULL;
  *out_complete = true;
  return iree_ok_status();
}

iree_status_t loom_service_finite_component_trim(
    loom_service_finite_component_t* component, bool* out_complete) {
  *out_complete = true;
  IREE_ASSERT(!component->stopping &&
              !loom_service_finite_component_is_pending(component));
  if (loom_service_finite_component_withdraw(component)) {
    loom_service_finite_component_publish(component);
  }
  if (component->load) {
    IREE_RETURN_IF_ERROR(loom_service_finite_load_free(component->load));
    component->load = NULL;
  }
  if (!component->model) {
    loom_service_finite_component_end_work(component);
    return iree_ok_status();
  }
  if (component->phase != LOOM_SERVICE_FINITE_COMPONENT_RELEASING) {
    loom_runtime_residency_t* residency =
        loom_service_finite_model_residency(component->model);
    if (loom_runtime_residency_is_pinned(residency)) {
      return iree_ok_status();
    }
    if (loom_runtime_residency_snapshot(residency).elastic) {
      loom_service_finite_component_begin_work(
          component, LOOM_RUNTIME_SERVICE_WORK_RELEASE);
      iree_status_t status =
          loom_service_finite_model_deactivate(component->model, out_complete);
      if (iree_status_is_ok(status) && *out_complete) {
        loom_service_finite_component_end_work(component);
      }
      return status;
    }
  }
  loom_service_finite_component_begin_work(component,
                                           LOOM_RUNTIME_SERVICE_WORK_RELEASE);
  IREE_RETURN_IF_ERROR(
      loom_service_finite_component_release_model(component, out_complete));
  if (*out_complete) {
    loom_service_finite_component_set_phase(
        component, LOOM_SERVICE_FINITE_COMPONENT_DECLARED);
    loom_service_finite_component_end_work(component);
  }
  return iree_ok_status();
}

iree_status_t loom_service_finite_component_retire(
    loom_service_finite_component_t* component, bool* out_complete) {
  *out_complete = true;
  if (!component) {
    return iree_ok_status();
  }
  loom_service_finite_component_advance(component, true);
  if (loom_service_finite_component_is_pending(component)) {
    *out_complete = false;
    return iree_ok_status();
  }
  if (component->load) {
    IREE_RETURN_IF_ERROR(loom_service_finite_load_free(component->load));
    component->load = NULL;
  }
  if (component->model) {
    loom_service_finite_component_begin_work(component,
                                             LOOM_RUNTIME_SERVICE_WORK_RELEASE);
  }
  IREE_RETURN_IF_ERROR(
      loom_service_finite_component_release_model(component, out_complete));
  if (*out_complete) {
    loom_service_finite_component_set_phase(
        component, LOOM_SERVICE_FINITE_COMPONENT_RETIRED);
    loom_service_finite_component_end_work(component);
  }
  return iree_ok_status();
}

void loom_service_finite_component_free(
    loom_service_finite_component_t* component) {
  if (!component) {
    return;
  }
  IREE_ASSERT(component->phase == LOOM_SERVICE_FINITE_COMPONENT_DECLARED ||
              component->phase == LOOM_SERVICE_FINITE_COMPONENT_RETIRED);
  if (loom_service_finite_component_withdraw(component)) {
    loom_service_finite_component_publish(component);
  }
  iree_status_free(component->status);
  iree_allocator_free(component->host_allocator, component);
}
