// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_STATE_MEMORY_H_
#define LOOM_SERVE_SERVICE_TEXT_STATE_MEMORY_H_

#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/service/text/storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_state_t loom_serve_text_state_t;

// Next owner action after the current maintenance frontier succeeds.
typedef enum loom_serve_text_trim_phase_e {
  LOOM_SERVICE_TEXT_TRIM_COMPLETE,
  LOOM_SERVICE_TEXT_TRIM_RELOCATE,
  LOOM_SERVICE_TEXT_TRIM_RELEASE,
} loom_serve_text_trim_phase_t;

// Physical commitment, separate from logical row/block ownership.
typedef struct loom_serve_text_memory_t {
  // Borrowed shared allocation domain for state and model parameters.
  loom_serve_memory_pool_t* pool;
  // Mutable virtual state only, excluding weights and execution workspace.
  loom_runtime_storage_statistics_t statistics;
  // Native backing changes exclude state mutation through their readiness.
  struct {
    // Owned native borrower, joined separately from failed queue readiness.
    loom_serve_memory_operation_t* operation;
    // Owned timeline imported before dependent transfers or maintenance.
    iree_hal_semaphore_t* semaphore;
    // Last admitted native readiness value.
    uint64_t value;
  } backing;
  // Independent native mapping borrowers during cold reservation growth.
  // Each owns its own readiness value; completion of one cannot publish
  // another reservation's translations.
  struct {
    // Owned native operations, joined before replacing their directories.
    loom_serve_memory_operation_t*
        operations[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
    // Owned readiness semaphores imported into the model timeline.
    iree_hal_semaphore_t* semaphores[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  } growth;
  // Compaction spans native releases and device copies on one model owner.
  struct {
    // Action admitted after successful readiness of the preceding phase.
    loom_serve_text_trim_phase_t phase;
    // State-local release counter before this maintenance cut.
    uint64_t released_before;
  } trim;
  // Optional reservations in source allocation order.
  loom_serve_virtual_buffer_t*
      buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  // Always-live initialized prefixes outside row-private storage.
  iree_device_size_t private_lengths[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
} loom_serve_text_memory_t;

// Allocates validated source roots and retains every accepted native mapping.
// Device, pool, and execution ownership are established by state construction.
iree_status_t loom_serve_text_memory_allocate(loom_serve_text_state_t* state);

// Submits one source clear/upload batch after all adapter views are
// constructed. Source payloads remain owned until the initialization frontier
// retires.
iree_status_t loom_serve_text_memory_initialize(loom_serve_text_state_t* state);

// Native borrowers retire independently of failed queue readiness.
void loom_serve_text_memory_set_observer(loom_serve_text_memory_t* memory,
                                         void (*wake)(void*), void* user_data);
bool loom_serve_text_memory_is_retired(const loom_serve_text_memory_t* memory);
void loom_serve_text_memory_await_retirement(loom_serve_text_memory_t* memory);

// Joins successful backing before reusing its operation slot. The serialized
// owner has already observed the enclosing model frontier.
void loom_serve_text_memory_complete_backing(loom_serve_text_memory_t* memory);

// Releases every independent growth operation after its frontier retires.
void loom_serve_text_memory_complete_growth(loom_serve_text_memory_t* memory);

// Releases native ownership after all roots and operations retire. A platform
// release failure preserves remaining reservations for the caller's retry.
iree_status_t loom_serve_text_memory_deinitialize(
    loom_serve_text_state_t* state);

// Submits the logical owner's already selected keep sets, importing native
// readiness into the maintenance execution before its next dependent work.
iree_status_t loom_serve_text_memory_trim(loom_serve_text_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_STATE_MEMORY_H_
