// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_SESSION_CACHE_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_SESSION_CACHE_H_

#include "experimental/loom_serve/text/checkpoint_registry.h"
#include "experimental/loom_serve/text/session.h"
#include "experimental/loom_serve/text/telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Idle continuation detached from execution scratch. The serialized service
// owner mutates records only after preparation and accepted device work retire.
typedef struct loom_serve_text_session_cache_entry_t {
  // Owned immutable native endpoint and canonical transcript.
  loom_serve_text_checkpoint_entry_t endpoint;
  // Logical session moved here after pin readiness; zero before publication.
  loom_serve_text_session_lifetime_t lifetime;
} loom_serve_text_session_cache_entry_t;

typedef struct loom_serve_text_session_cache_t {
  // Allocator owning the fixed record array.
  iree_allocator_t allocator;
  // Stable records, with one extra slot for a full-cache admission exchange.
  loom_serve_text_session_cache_entry_t* entries;
  // Configured retained endpoint bound, excluding the transient exchange slot.
  iree_host_size_t capacity;
  // Occupied records, including a pending pin before publication.
  iree_host_size_t count;
} loom_serve_text_session_cache_t;

// Allocates host records only. Zero disables automatic retention. The caller
// reserves capacity+1 native endpoints when enabled and bounds capacity before
// calling. Partial initialization may be reset and deinitialized.
iree_status_t loom_serve_text_session_cache_initialize(
    iree_host_size_t capacity, loom_serve_text_session_cache_t* cache,
    iree_allocator_t allocator);

// Ends logical lifetimes after all borrowers retire. Failed readiness leaves
// accepted native handles to outer model destruction, as with explicit pins.
void loom_serve_text_session_cache_reset(loom_serve_text_session_cache_t* cache,
                                         loom_serve_text_telemetry_t* telemetry,
                                         iree_status_code_t terminal_code);
void loom_serve_text_session_cache_deinitialize(
    loom_serve_text_session_cache_t* cache);

// Returns a published named session, or NULL. The owner protects a selected
// entry from eviction until its preparation/admission borrower relinquishes it.
loom_serve_text_session_cache_entry_t* loom_serve_text_session_cache_find(
    loom_serve_text_session_cache_t* cache, iree_string_view_t name);

// Returns the least recently admitted published entry other than protected, or
// NULL. Lookup is a cold admission operation, never per-token work.
loom_serve_text_session_cache_entry_t* loom_serve_text_session_cache_oldest(
    loom_serve_text_session_cache_t* cache,
    const loom_serve_text_session_cache_entry_t* protected_entry);

// Captures a row into an available record. The caller enforces the configured
// bound, allowing one extra record only while exchanging a protected incoming
// endpoint. Native capacity is reserved independently of explicit pins. Failure
// leaves accepted native ownership with the model; publication requires ALL
// readiness, and neither the row nor the record may be reused before it.
iree_status_t loom_serve_text_session_cache_submit(
    loom_serve_text_session_cache_t* cache, loom_serve_text_session_t* session,
    loom_serve_text_session_cache_entry_t** out_entry,
    loom_serve_execution_frontier_t* out_frontier);

// Moves logical ownership after successful pin readiness. The row's canonical
// transcript reference is released; the endpoint retained its own reference.
// Native row reset remains with the caller, after this ownership transfer.
void loom_serve_text_session_cache_publish(
    loom_serve_text_session_cache_entry_t* entry,
    loom_serve_text_session_t* session);

// Ends an unborrowed cached lifetime and releases its endpoint at a completed
// model boundary. To resume instead, first move lifetime into the destination
// row and zero the entry's lifetime; release then emits no session-end event.
void loom_serve_text_session_cache_release(
    loom_serve_text_session_cache_t* cache,
    loom_serve_text_session_cache_entry_t* entry,
    loom_serve_text_telemetry_t* telemetry);

// Releases the least-recent idle logical owner across resident rows and the
// detached cache. Explicit checkpoint references survive. The caller excludes
// CPU/device borrowers and supplies its prepared continuation's protected row
// and endpoint, if any. True requires a subsequent model trim; logical release
// alone does not make physical backing available. No device work is submitted.
bool loom_serve_text_session_cache_evict(
    loom_serve_text_session_cache_t* cache, iree_host_size_t row_count,
    loom_serve_text_session_t* rows,
    const loom_serve_text_session_t* protected_row,
    const loom_serve_text_session_cache_entry_t* protected_entry,
    loom_serve_text_telemetry_t* telemetry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_SESSION_CACHE_H_
