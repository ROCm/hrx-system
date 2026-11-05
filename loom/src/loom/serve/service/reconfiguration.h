// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_RECONFIGURATION_H_
#define LOOM_SERVE_SERVICE_RECONFIGURATION_H_

#include "loom/runtime/program/sources.h"
#include "loom/runtime/program/variant_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// Delivers an owned terminal selection status on the serialized service owner.
// selected identifies the actual preferred choice after the attempt, including
// failure/cancellation. The callback never waits or reenters service advance.
typedef void (*loom_serve_variant_selection_fn_t)(
    void* user_data, iree_status_t status, loom_runtime_variant_id_t selected);

// Optional cold management capability, borrowed through service retirement.
// The pool owns compiled choices, not an active setting. These operations
// express this service's native-update and preferred-choice policies; request
// consumers may separately acquire entries without changing that preference.
// Calls use the same serialized owner as service accept/advance.
typedef struct loom_serve_reconfiguration_t {
  // Borrowed modality policy, alive through every accepted completion.
  void* self;
  // Owned by the service; completed choices and diagnostics outlive requests.
  // NULL means this service has not enabled reconfiguration.
  loom_runtime_variant_pool_t* pool;
  // Snapshots native provider locators and starts independent compilation
  // against a ready base's retained resource contract. prepare/control must be
  // empty: this capability keeps source declarations, VM and state unchanged.
  // Success returns an observable entry, not a selected executable; compiler
  // failure remains on that entry and does not fail the running service.
  // Source and bytecode providers use the same contract. No file payload is
  // read on the owner, and no caller storage is borrowed after return.
  iree_status_t (*stage_native)(void* self, loom_runtime_variant_id_t base,
                                const loom_runtime_sources_t* sources,
                                loom_runtime_variant_id_t* out_id);
  // Queues a ready choice at the next modality-defined safe boundary. An
  // expected mismatch or an already queued policy change rejects without a
  // callback. Success owes completion once, including shutdown cancellation.
  // The service retains the choice through completion; no consumer wait occurs.
  iree_status_t (*select)(void* self, loom_runtime_variant_id_t id,
                          loom_runtime_variant_id_t expected,
                          loom_serve_variant_selection_fn_t completion,
                          void* user_data);
  // Actual preferred choice; a queued policy change is not yet publication.
  loom_runtime_variant_id_t (*selected)(void* self);
} loom_serve_reconfiguration_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_RECONFIGURATION_H_
