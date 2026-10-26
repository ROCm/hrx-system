// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/component.h"

struct loom_serve_finite_component_t {
  // Allocator owning the handle and its preparation operation.
  iree_allocator_t host_allocator;
  // Borrowed shared physical owner, surviving final model retirement.
  loom_serve_device_t* device;
  // Borrowed definition and package, independent of any requesting connection.
  loom_serve_finite_model_options_t options;
  // Immutable owner wake target, live until all callbacks have retired.
  loom_serve_wake_callback_t observer;
  // Owner-only lifecycle; only READY lends a model to an invocation.
  loom_serve_finite_component_phase_t phase;
  // Monotonic start of the current PREPARING interval; phase owns its validity.
  iree_time_t preparation_start;
  // Completed PREPARING intervals, including attempts before a fixed trim.
  uint64_t preparation_duration_ns;
  // Irreversible deployment shutdown, independent of request cancellation.
  bool stopping;
  // Persistent eviction policy, including before lazy preparation completes.
  bool pinned;
  // Owned asynchronous load or terminal load whose cleanup failed.
  loom_serve_finite_load_t* load;
  // Owned prepared model, retained through actual queue ownership retirement.
  loom_serve_finite_model_t* model;
  // Owned terminal preparation error, cloned only for affected consumers.
  iree_status_t status;
  // Cumulative evidence independent of load/model destruction.
  loom_serve_compilation_report_t report;
};

static void loom_serve_finite_component_set_phase(
    loom_serve_finite_component_t* component,
    loom_serve_finite_component_phase_t phase) {
  if (component->phase != phase &&
      (component->phase == LOOM_SERVE_FINITE_COMPONENT_PREPARING ||
       phase == LOOM_SERVE_FINITE_COMPONENT_PREPARING)) {
    const iree_time_t now = iree_time_now();
    if (component->phase == LOOM_SERVE_FINITE_COMPONENT_PREPARING) {
      component->preparation_duration_ns += now - component->preparation_start;
    } else {
      component->preparation_start = now;
    }
  }
  component->phase = phase;
}

iree_status_t loom_serve_finite_component_allocate(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_finite_component_t** out_component) {
  *out_component = NULL;
  loom_serve_finite_component_t* component = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*component),
                                             (void**)&component));
  component->host_allocator = host_allocator;
  component->device = device;
  component->options = *options;
  component->observer = observer;
  *out_component = component;
  return iree_ok_status();
}

loom_serve_finite_component_phase_t loom_serve_finite_component_phase(
    const loom_serve_finite_component_t* component) {
  return component->phase;
}

uint64_t loom_serve_finite_component_preparation_duration_ns(
    const loom_serve_finite_component_t* component) {
  return component->preparation_duration_ns +
         (component->phase == LOOM_SERVE_FINITE_COMPONENT_PREPARING
              ? iree_time_now() - component->preparation_start
              : 0);
}

bool loom_serve_finite_component_is_pending(
    const loom_serve_finite_component_t* component) {
  return component &&
         (component->phase == LOOM_SERVE_FINITE_COMPONENT_PREPARING ||
          loom_serve_finite_model_is_pending(component->model));
}

const loom_serve_compilation_report_t* loom_serve_finite_component_report(
    const loom_serve_finite_component_t* component) {
  return &component->report;
}

iree_status_t loom_serve_finite_component_status(
    const loom_serve_finite_component_t* component) {
  return component->status;
}

bool loom_serve_finite_component_parameters(
    const loom_serve_finite_component_t* component,
    loom_serve_residency_snapshot_t* out_parameters) {
  *out_parameters = (loom_serve_residency_snapshot_t){0};
  if (!component->model) {
    return false;
  }
  *out_parameters = loom_serve_finite_model_parameters(component->model);
  return true;
}

void loom_serve_finite_component_set_pinned(
    loom_serve_finite_component_t* component, bool pinned) {
  component->pinned = pinned;
  if (component->model) {
    loom_serve_finite_model_set_pinned(component->model, pinned);
  }
}

bool loom_serve_finite_component_advance(
    loom_serve_finite_component_t* component, bool stopping) {
  if (!component) {
    return false;
  }
  component->stopping |= stopping;
  if (component->phase != LOOM_SERVE_FINITE_COMPONENT_PREPARING) {
    const bool pending = loom_serve_finite_model_is_pending(component->model);
    iree_status_t status =
        loom_serve_finite_model_query_status(component->model);
    if (!iree_status_is_ok(status)) {
      component->status = iree_status_join(component->status, status);
      loom_serve_finite_component_set_phase(component,
                                            LOOM_SERVE_FINITE_COMPONENT_FAILED);
    }
    return pending && !loom_serve_finite_model_is_pending(component->model);
  }
  bool complete = false;
  component->status = loom_serve_finite_load_advance(
      component->load,
      component->stopping ? LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL : 0, &complete);
  if (!complete) {
    return false;
  }
  loom_serve_compilation_merge(&component->report,
                               loom_serve_finite_load_report(component->load));
  if (iree_status_is_ok(component->status)) {
    component->model = loom_serve_finite_load_take_model(component->load);
    loom_serve_finite_model_set_pinned(component->model, component->pinned);
  }
  iree_status_t cleanup = loom_serve_finite_load_free(component->load);
  if (iree_status_is_ok(cleanup)) {
    component->load = NULL;
  }
  component->status = iree_status_join(component->status, cleanup);
  loom_serve_finite_component_set_phase(
      component, iree_status_is_ok(component->status)
                     ? LOOM_SERVE_FINITE_COMPONENT_READY
                     : LOOM_SERVE_FINITE_COMPONENT_FAILED);
  return true;
}

iree_status_t loom_serve_finite_component_prepare(
    loom_serve_finite_component_t* component,
    loom_serve_finite_model_t** out_model) {
  *out_model = NULL;
  // A borrowed model must not escape a concurrent quiescent release or removal.
  IREE_ASSERT(!component->stopping &&
              component->phase != LOOM_SERVE_FINITE_COMPONENT_RELEASING);
  if (component->phase == LOOM_SERVE_FINITE_COMPONENT_DECLARED) {
    loom_serve_finite_component_set_phase(
        component, LOOM_SERVE_FINITE_COMPONENT_PREPARING);
    component->status = loom_serve_finite_load_allocate(
        component->device, &component->options, component->observer.wake,
        component->observer.user_data, component->host_allocator,
        &component->load);
    if (!iree_status_is_ok(component->status)) {
      loom_serve_finite_component_set_phase(component,
                                            LOOM_SERVE_FINITE_COMPONENT_FAILED);
    }
  }
  loom_serve_finite_component_advance(component, false);
  if (component->phase == LOOM_SERVE_FINITE_COMPONENT_READY) {
    *out_model = component->model;
  }
  return iree_status_clone(component->status);
}

static iree_status_t loom_serve_finite_component_release_model(
    loom_serve_finite_component_t* component, bool* out_complete) {
  *out_complete = false;
  if (component->model &&
      component->phase != LOOM_SERVE_FINITE_COMPONENT_RELEASING) {
    loom_serve_finite_model_retire(component->model, component->observer.wake,
                                   component->observer.user_data);
    loom_serve_finite_component_set_phase(
        component, LOOM_SERVE_FINITE_COMPONENT_RELEASING);
  }
  if (!loom_serve_finite_model_is_retired(component->model)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_finite_model_free_retired(component->model));
  component->model = NULL;
  *out_complete = true;
  return iree_ok_status();
}

iree_status_t loom_serve_finite_component_trim(
    loom_serve_finite_component_t* component, bool* out_complete) {
  *out_complete = true;
  IREE_ASSERT(!component->stopping &&
              !loom_serve_finite_component_is_pending(component));
  if (component->load) {
    IREE_RETURN_IF_ERROR(loom_serve_finite_load_free(component->load));
    component->load = NULL;
  }
  if (!component->model) {
    return iree_ok_status();
  }
  if (component->phase != LOOM_SERVE_FINITE_COMPONENT_RELEASING) {
    loom_serve_residency_t* residency =
        loom_serve_finite_model_residency(component->model);
    if (loom_serve_residency_is_pinned(residency)) {
      return iree_ok_status();
    }
    if (loom_serve_residency_snapshot(residency).elastic) {
      return loom_serve_finite_model_deactivate(component->model, out_complete);
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_finite_component_release_model(component, out_complete));
  if (*out_complete) {
    loom_serve_finite_component_set_phase(component,
                                          LOOM_SERVE_FINITE_COMPONENT_DECLARED);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_finite_component_retire(
    loom_serve_finite_component_t* component, bool* out_complete) {
  *out_complete = true;
  if (!component) {
    return iree_ok_status();
  }
  loom_serve_finite_component_advance(component, true);
  if (loom_serve_finite_component_is_pending(component)) {
    *out_complete = false;
    return iree_ok_status();
  }
  if (component->load) {
    IREE_RETURN_IF_ERROR(loom_serve_finite_load_free(component->load));
    component->load = NULL;
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_finite_component_release_model(component, out_complete));
  if (*out_complete) {
    loom_serve_finite_component_set_phase(component,
                                          LOOM_SERVE_FINITE_COMPONENT_RETIRED);
  }
  return iree_ok_status();
}

void loom_serve_finite_component_free(
    loom_serve_finite_component_t* component) {
  if (!component) {
    return;
  }
  IREE_ASSERT(component->phase == LOOM_SERVE_FINITE_COMPONENT_DECLARED ||
              component->phase == LOOM_SERVE_FINITE_COMPONENT_RETIRED);
  iree_status_free(component->status);
  iree_allocator_free(component->host_allocator, component);
}
