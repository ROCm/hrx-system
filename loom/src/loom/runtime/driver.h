// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_DRIVER_H_
#define LOOM_RUNTIME_DRIVER_H_

#include "loom/runtime/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared service progression on one serialized application owner. Admission,
// resource progress, and delivery surround each turn. The driver owns neither
// service storage nor an executor, and performs no allocation or waiting.
typedef struct loom_runtime_driver_t {
  // Borrowed stable slots through final collection. A NULL vtable is vacant.
  // The owner may publish or clear slots between calls or retire the current
  // slot during its advance; it must preserve accepted callback lifetimes.
  loom_runtime_service_t* services;
  // Immutable slot count, including vacant publication credit.
  iree_host_size_t capacity;
  // Rotating priority, advanced once per complete traversal.
  iree_host_size_t cursor;
  // Monotonic abort intent; accepted borrowers still require collection.
  bool stopping;
  // Every slot has received a traversal that began with abort intent.
  bool stopping_applied;
  // Owned terminal application failure, consumed after final collection.
  // Compositions retain model-local failures instead of returning them here.
  iree_status_t status;
} loom_runtime_driver_t;

// Initializes inert state over caller-owned slots. The caller later consumes
// driver.status. Services and their capacity owners outlive final collection.
void loom_runtime_driver_initialize(iree_host_size_t capacity,
                                    loom_runtime_service_t* services,
                                    loom_runtime_driver_t* out_driver);

// Installs or detaches the owner wake on published services. A later publisher
// installs the same wake before submitting asynchronous work. Detachment
// follows final collection, not merely a terminal status or readiness signal.
void loom_runtime_driver_set_observer(loom_runtime_driver_t* driver,
                                      loom_runtime_wake_callback_t observer);

// Another occupied slot in the same nonnull capacity domain owns activity that
// can release capacity. Queued requests alone never justify waiting on a peer.
loom_runtime_service_flags_t loom_runtime_driver_shared_activity(
    const loom_runtime_driver_t* driver, iree_host_size_t selected);

// Advances every occupied slot once in rotating order, including after failure.
// A returned failure latches stopping/status; accepted work still drains.
void loom_runtime_driver_advance(loom_runtime_driver_t* driver,
                                 bool* out_progress);

// Collects pending ownership after the turn and any adapter delivery. A late
// abort requires a complete stopping turn before detachment, even when earlier
// services own queued requests but no asynchronous callbacks. Once that turn
// completes, pending services owe the next wake; collection never polls them.
bool loom_runtime_driver_collect(loom_runtime_driver_t* driver,
                                 bool* out_progress);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_DRIVER_H_
