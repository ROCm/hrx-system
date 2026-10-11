// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_ACTIVATION_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_ACTIVATION_H_

#include "experimental/loom_serve/runtime/residency.h"
#include "iree/async/util/proactor_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// One explicit parameter warmup, independent of an inference request. Borrows
// the residency through terminal advance. The shared residency owner serializes
// every call; only completion publication runs on the proactor thread.
typedef struct loom_serve_activation_t loom_serve_activation_t;

typedef enum loom_serve_activation_phase_e {
  // No pin or device work; retry after other owner activity releases capacity.
  LOOM_SERVE_ACTIVATION_WAITING = 0,
  // Accepted activation and its asynchronous readiness callback still own work.
  LOOM_SERVE_ACTIVATION_LOADING,
  // Callback storage is joined; success released the warmup pin. Failure keeps
  // parameter protection until actual model retirement, not failed readiness.
  LOOM_SERVE_ACTIVATION_COMPLETE,
} loom_serve_activation_phase_t;

typedef enum loom_serve_activation_flag_bits_e {
  LOOM_SERVE_ACTIVATION_FLAG_NONE = 0,
  // Withdraw pending admission; already accepted work still joins readiness.
  // The caller keeps cancellation asserted on subsequent owner turns.
  LOOM_SERVE_ACTIVATION_FLAG_CANCEL = 1u << 0,
} loom_serve_activation_flag_bits_t;
typedef uint32_t loom_serve_activation_flags_t;

// Allocates unstarted warmup storage and retains a shared proactor. Does not
// load or pin parameters. Wake only schedules the owner, never consumes model
// state inline. Failure returns no operation.
iree_status_t loom_serve_activation_create(
    loom_serve_residency_t* residency, iree_async_proactor_pool_t* pool,
    void (*wake)(void* user_data), void* user_data,
    loom_serve_activation_t** out_activation, iree_allocator_t allocator);

// Attempts FIFO admission or consumes readiness without waiting for device
// completion. Initial admission can synchronously allocate/map native backing.
// WAITING needs another owner activity/capacity change, not a polling timer.
// LOADING owes a wake. COMPLETE transfers terminal status and permits
// destruction. The next pressure event may evict successfully warmed
// parameters; this is not pinning.
iree_status_t loom_serve_activation_advance(
    loom_serve_activation_t* activation, loom_serve_activation_flags_t flags,
    loom_serve_activation_phase_t* out_phase);

// Called only after terminal advance, with all completion access joined.
void loom_serve_activation_destroy(loom_serve_activation_t* activation);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_ACTIVATION_H_
