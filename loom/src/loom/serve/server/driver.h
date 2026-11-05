// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_DRIVER_H_
#define LOOM_SERVE_SERVER_DRIVER_H_

#include "loom/serve/server/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Common service/registration progression on one serialized application owner.
// Admission, resource progress, and transport delivery surround this turn; they
// do not implement a second capacity or model-retirement policy. No worker,
// model, or registry storage is owned here.
typedef struct loom_serve_driver_t {
  // Borrowed stable registration table, alive through final collection.
  loom_serve_registry_t* registry;
  // Immutable slot count, including vacant registration credit.
  iree_host_size_t capacity;
  // Rotating service priority, advanced once per complete traversal.
  iree_host_size_t cursor;
  // Monotonic abort intent; accepted borrowers still require collection.
  bool stopping;
  // Owned terminal application failure, consumed by the outer owner after
  // retirement. Managed model-local failures remain in their registry slots.
  iree_status_t status;
} loom_serve_driver_t;

// Optional observer of a slot after its service advances and before its model
// can retire. It copies terminal observations, never extends model lifetime,
// waits, or retains borrowed slot data. Failure is an application failure.
typedef struct loom_serve_driver_observer_t {
  // Borrowed adapter state, valid through this synchronous traversal.
  void* self;
  // Absent for native-only use. Reports actual observation progress.
  iree_status_t (*observe)(void* self, iree_host_size_t index, bool stopping,
                           bool* out_progress);
} loom_serve_driver_observer_t;

// Initializes inert owner state. The caller later consumes driver.status.
void loom_serve_driver_initialize(loom_serve_registry_t* registry,
                                  loom_serve_driver_t* out_driver);

// Installs or detaches the owner wake on already-published services. Managed
// preparation receives the same wake through advance. Detachment follows final
// collection, not merely an error or readiness signal.
void loom_serve_driver_set_observer(loom_serve_driver_t* driver,
                                    loom_runtime_wake_callback_t observer);

// Another member of the same nonnull capacity domain owns activity that can
// release capacity. Queued requests alone never justify waiting on a peer.
loom_runtime_service_flags_t loom_serve_driver_shared_activity(
    const loom_serve_driver_t* driver, iree_host_size_t selected);

// Advances each occupied slot once in rotating order, including after failure.
// Shared failures latch stopping/status; accepted work still drains. The
// observer is the sole boundary between service progress and model retirement.
void loom_serve_driver_advance(loom_serve_driver_t* driver,
                               loom_runtime_wake_callback_t wake,
                               loom_serve_driver_observer_t observer,
                               bool* out_progress);

// Collects pending ownership after the turn and any adapter delivery. A late
// application abort seals every managed slot and reports progress so the owner
// runs the required retirement turn without relying on another external event.
// False while stopping permits detachment; failed release still leaves the
// registry owned.
bool loom_serve_driver_collect(loom_serve_driver_t* driver, bool* out_progress);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_DRIVER_H_
