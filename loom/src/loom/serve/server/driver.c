// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/driver.h"

void loom_serve_driver_initialize(loom_serve_registry_t* registry,
                                  loom_serve_driver_t* out_driver) {
  *out_driver = (loom_serve_driver_t){.registry = registry};
  loom_runtime_driver_initialize(loom_serve_registry_capacity(registry),
                                 loom_serve_registry_services(registry),
                                 &out_driver->runtime);
}

void loom_serve_driver_set_observer(loom_serve_driver_t* driver,
                                    loom_runtime_wake_callback_t observer) {
  loom_serve_registry_set_advance_observer(driver->registry, observer,
                                           (loom_serve_registry_observer_t){0});
  loom_runtime_driver_set_observer(&driver->runtime, observer);
}

loom_runtime_service_flags_t loom_serve_driver_shared_activity(
    const loom_serve_driver_t* driver, iree_host_size_t selected) {
  return loom_runtime_driver_shared_activity(&driver->runtime, selected);
}

void loom_serve_driver_advance(loom_serve_driver_t* driver,
                               loom_runtime_wake_callback_t wake,
                               loom_serve_registry_observer_t observer,
                               bool* out_progress) {
  loom_serve_registry_set_advance_observer(driver->registry, wake, observer);
  loom_runtime_driver_advance(&driver->runtime, out_progress);
}

bool loom_serve_driver_collect(loom_serve_driver_t* driver,
                               bool* out_progress) {
  driver->runtime.stopping |= !iree_status_is_ok(driver->runtime.status);
  if (driver->runtime.stopping) {
    // Seal managed admission before the owner unlocks, including when adapter
    // delivery failed after the last service turn. Accepted borrowers still
    // retire through subsequent runtime turns.
    for (iree_host_size_t i = 0; i < driver->runtime.capacity; ++i) {
      loom_serve_registered_model_t* model =
          loom_serve_registry_at(driver->registry, i);
      if (model->incarnation && model->removal != LOOM_SERVE_REMOVAL_ABORT) {
        driver->runtime.status = iree_status_join(
            driver->runtime.status,
            loom_serve_registry_remove(model, LOOM_SERVE_REMOVAL_ABORT));
        *out_progress = true;
      }
    }
  }
  return loom_runtime_driver_collect(&driver->runtime, out_progress);
}
