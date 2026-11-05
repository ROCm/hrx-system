// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/driver.h"

void loom_serve_driver_initialize(loom_serve_registry_t* registry,
                                  loom_serve_driver_t* out_driver) {
  *out_driver = (loom_serve_driver_t){
      .registry = registry,
      .capacity = loom_serve_registry_capacity(registry),
  };
}

void loom_serve_driver_set_observer(loom_serve_driver_t* driver,
                                    loom_runtime_wake_callback_t observer) {
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(driver->registry, i);
    if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
      model->service.runtime.vtable->set_observer(model->service.runtime.self,
                                                  observer);
    }
  }
}

loom_runtime_service_flags_t loom_serve_driver_shared_activity(
    const loom_serve_driver_t* driver, iree_host_size_t selected) {
  const void* capacity_domain =
      loom_serve_registry_at(driver->registry, selected)
          ->service.runtime.capacity_domain;
  if (!capacity_domain) {
    return 0;
  }
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(driver->registry, i);
    if (i == selected ||
        model->service.runtime.capacity_domain != capacity_domain) {
      continue;
    }
    // Retirement still owns backing after service callbacks are gone. Its
    // ownership wake may release capacity; a quiescent failed release cannot.
    if (model->phase == LOOM_SERVE_REGISTRATION_RETIRING ||
        (model->phase == LOOM_SERVE_REGISTRATION_READY &&
         (model->service.runtime.vtable->is_active(
              model->service.runtime.self) ||
          model->parameter_operation ==
              LOOM_SERVE_PARAMETER_OPERATION_ACTIVATING))) {
      return LOOM_RUNTIME_SERVICE_FLAG_SHARED_ACTIVITY;
    }
  }
  return 0;
}

void loom_serve_driver_advance(loom_serve_driver_t* driver,
                               loom_runtime_wake_callback_t wake,
                               loom_serve_driver_observer_t observer,
                               bool* out_progress) {
  *out_progress = false;
  driver->stopping |= !iree_status_is_ok(driver->status);
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const iree_host_size_t index = (driver->cursor + i) % driver->capacity;
    loom_serve_registered_model_t* model =
        loom_serve_registry_at(driver->registry, index);
    if (!model->service.name.size) {
      continue;
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z_model, "loom_serve_model_advance");
    IREE_TRACE_ZONE_APPEND_TEXT(z_model, model->service.name.data,
                                model->service.name.size);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_model, model->incarnation);
    if (driver->stopping && model->incarnation &&
        model->removal != LOOM_SERVE_REMOVAL_ABORT) {
      driver->status = iree_status_join(
          driver->status,
          loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
    }
    if (model->phase == LOOM_SERVE_REGISTRATION_READY) {
      const loom_serve_service_t* service = &model->service;
      bool progress = false;
      const loom_runtime_service_flags_t flags =
          loom_serve_driver_shared_activity(driver, index) |
          loom_serve_registry_service_flags(model) |
          (driver->stopping ? LOOM_RUNTIME_SERVICE_FLAG_STOPPING : 0);
      iree_status_t status = service->runtime.vtable->advance(
          service->runtime.self, flags, &progress);
      if (!iree_status_is_ok(status)) {
        // The first failure requires another turn to stop admission. Repeated
        // terminal status is not progress while accepted borrowers retire.
        progress |= iree_status_is_ok(model->status);
        status = loom_serve_registry_fail(driver->registry, index, status);
      }
      driver->status = iree_status_join(driver->status, status);
      driver->stopping |= !iree_status_is_ok(driver->status);
      *out_progress |= progress;
    }
    if (observer.observe) {
      bool progress = false;
      driver->status = iree_status_join(
          driver->status,
          observer.observe(
              observer.self, index,
              driver->stopping || !iree_status_is_ok(model->status),
              &progress));
      driver->stopping |= !iree_status_is_ok(driver->status);
      *out_progress |= progress;
    }
    bool progress = false;
    driver->status = iree_status_join(
        driver->status,
        loom_serve_registry_advance(driver->registry, index, wake, &progress));
    driver->stopping |= !iree_status_is_ok(driver->status);
    *out_progress |= progress;
    IREE_TRACE_ZONE_END(z_model);
  }
  if (++driver->cursor >= driver->capacity) {
    driver->cursor = 0;
  }
}

bool loom_serve_driver_collect(loom_serve_driver_t* driver,
                               bool* out_progress) {
  bool pending = false;
  driver->stopping |= !iree_status_is_ok(driver->status);
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    loom_serve_registered_model_t* model =
        loom_serve_registry_at(driver->registry, i);
    if (driver->stopping && model->incarnation &&
        model->removal != LOOM_SERVE_REMOVAL_ABORT) {
      driver->status = iree_status_join(
          driver->status,
          loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
      *out_progress = true;
    }
    pending |= loom_serve_registry_pending(model) ||
               (driver->stopping && model->incarnation &&
                model->phase != LOOM_SERVE_REGISTRATION_RELEASE_FAILED);
  }
  return pending;
}
