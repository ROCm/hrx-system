// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/driver.h"

void loom_runtime_driver_initialize(iree_host_size_t capacity,
                                    loom_runtime_service_t* services,
                                    loom_runtime_driver_t* out_driver) {
  *out_driver = (loom_runtime_driver_t){
      .services = services,
      .capacity = capacity,
  };
}

void loom_runtime_driver_set_observer(loom_runtime_driver_t* driver,
                                      loom_runtime_wake_callback_t observer) {
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const loom_runtime_service_t* service = &driver->services[i];
    if (service->vtable) {
      service->vtable->set_observer(service->self, observer);
    }
  }
}

loom_runtime_service_flags_t loom_runtime_driver_shared_activity(
    const loom_runtime_driver_t* driver, iree_host_size_t selected) {
  const void* capacity_domain = driver->services[selected].capacity_domain;
  if (!capacity_domain) {
    return 0;
  }
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const loom_runtime_service_t* service = &driver->services[i];
    if (i != selected && service->vtable &&
        service->capacity_domain == capacity_domain &&
        service->vtable->is_active(service->self)) {
      return LOOM_RUNTIME_SERVICE_FLAG_SHARED_ACTIVITY;
    }
  }
  return 0;
}

void loom_runtime_driver_advance(loom_runtime_driver_t* driver,
                                 bool* out_progress) {
  *out_progress = false;
  driver->stopping |= !iree_status_is_ok(driver->status);
  driver->stopping_applied = driver->stopping;
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const iree_host_size_t index = (driver->cursor + i) % driver->capacity;
    const loom_runtime_service_t* service = &driver->services[index];
    if (!service->vtable) {
      continue;
    }
    const loom_runtime_service_flags_t flags =
        loom_runtime_driver_shared_activity(driver, index) |
        (driver->stopping ? LOOM_RUNTIME_SERVICE_FLAG_STOPPING : 0);
    bool progress = false;
    iree_status_t status =
        service->vtable->advance(service->self, flags, &progress);
    if (!iree_status_is_ok(status)) {
      // A first failure schedules a stopping turn. Repeated terminal failure
      // is not progress while accepted borrowers await their retirement wake.
      progress |= iree_status_is_ok(driver->status);
      driver->status = iree_status_join(driver->status, status);
      driver->stopping = true;
    }
    *out_progress |= progress;
  }
  if (++driver->cursor >= driver->capacity) {
    driver->cursor = 0;
  }
}

bool loom_runtime_driver_collect(loom_runtime_driver_t* driver,
                                 bool* out_progress) {
  driver->stopping |= !iree_status_is_ok(driver->status);
  bool pending =
      driver->capacity && driver->stopping && !driver->stopping_applied;
  *out_progress |= pending;
  for (iree_host_size_t i = 0; i < driver->capacity; ++i) {
    const loom_runtime_service_t* service = &driver->services[i];
    pending |= service->vtable && service->vtable->is_pending(service->self);
  }
  return pending;
}
