// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_SERVICE_H_
#define LOOM_RUNTIME_SERVICE_H_

#include "iree/base/api.h"
#include "loom/runtime/wake.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_runtime_service_flag_bits_e {
  // Admission has stopped. Consume only already accepted asynchronous work;
  // new model submissions and queued request admission are forbidden.
  LOOM_RUNTIME_SERVICE_FLAG_STOPPING = 1u << 0,
};
typedef uint32_t loom_runtime_service_flags_t;

// Shared progression independent of modality, transport, and model lifetime.
// All methods run on one serialized owner; callbacks only notify that owner.
// Request admission, inspection, and preparation/retirement are separate
// contracts owned by the service and its composition.
typedef struct loom_runtime_service_vtable_t {
  // Installs the owner's nonblocking wake target before the first turn. The
  // owner clears it after all accepted asynchronous callbacks have retired.
  void (*set_observer)(void* self, loom_runtime_wake_callback_t observer);
  // Whether accepted requests still own queue, execution, or delivery storage.
  // Includes service-owned utility/control requests and blocked consumers;
  // excludes idle retained history, maintenance, and observation callbacks.
  // False permits graceful work to stop, but is_pending still governs storage
  // retirement. Reads published owner facts, never task-owned mutable payloads.
  bool (*has_requests)(void* self);
  // Whether an accepted asynchronous callback still borrows service storage.
  // True requires an eventual wake even after transport admission stops.
  bool (*is_pending)(void* self);
  // Whether execution or maintenance still owns model resources. Queued
  // requests alone do not prevent idle resource management.
  bool (*is_active)(void* self);
  // Advances at most one bounded model unit and reports actual progress. A
  // turn can retry admission after another service released shared capacity.
  // False means awaiting readiness, never polling. STOPPING consumes accepted
  // readiness without launching another model unit. Terminal failure belongs
  // to this service; shared resource owners report wider failure independently.
  // Even after failure, is_pending governs release of service storage.
  iree_status_t (*advance)(void* self, loom_runtime_service_flags_t flags,
                           bool* out_progress);
} loom_runtime_service_vtable_t;

// Borrowed service capability. Its composition retains both the implementation
// and resource owners through every accepted asynchronous obligation.
// A composition may include preparation and retirement in its own progression;
// only it can invoke the underlying model service while that service is live.
// Metadata without a progress implementation has no vtable and is not invoked.
typedef struct loom_runtime_service_t {
  // Borrowed service state, live through observer detachment and all requests.
  void* self;
  // Immutable implementation methods, shared by instances of the same service.
  const loom_runtime_service_vtable_t* vtable;
} loom_runtime_service_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_SERVICE_H_
