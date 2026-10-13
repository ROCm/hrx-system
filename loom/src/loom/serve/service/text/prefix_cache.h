// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_PREFIX_CACHE_H_
#define LOOM_SERVE_SERVICE_TEXT_PREFIX_CACHE_H_

#include "loom/serve/service/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// An exact consumed-token frontier, independent of canonical chat histories.
// The service owner protects a selected record through reservation readiness.
typedef struct loom_serve_text_prefix_entry_t {
  // Owned immutable endpoint; NULL denotes an available record.
  loom_serve_text_checkpoint_t* state;
  // Owned token IDs that produced state, with no selected/unconsumed token.
  int32_t* tokens;
  // Consumed positions and number of IDs in tokens.
  iree_host_size_t count;
  // Monotonic last successful admission or capture, for cold LRU eviction.
  uint64_t serial;
} loom_serve_text_prefix_entry_t;

typedef struct loom_serve_text_prefix_cache_t {
  // Allocator owning records and their compact token copies.
  iree_allocator_t allocator;
  // Fixed records mutated only by the serialized service owner.
  loom_serve_text_prefix_entry_t* entries;
  // Configured endpoint bound; zero disables lookup and capture.
  iree_host_size_t capacity;
  // Occupied records, including captures awaiting device readiness.
  iree_host_size_t count;
  // Monotonic recency identity, independent of session/request identity.
  uint64_t serial;
} loom_serve_text_prefix_cache_t;

// Allocates host records only. Native checkpoint capacity is reserved by the
// model configuration. Partial initialization supports reset/deinitialize.
iree_status_t loom_serve_text_prefix_cache_initialize(
    iree_host_size_t capacity, loom_serve_text_prefix_cache_t* cache,
    iree_allocator_t allocator);

// Releases normal completed owners. Failed readiness leaves native handles to
// model destruction; host copies can retire after CPU borrowers relinquish.
void loom_serve_text_prefix_cache_reset(loom_serve_text_prefix_cache_t* cache,
                                        iree_status_code_t terminal_code);
void loom_serve_text_prefix_cache_deinitialize(
    loom_serve_text_prefix_cache_t* cache);

// Finds the longest exact strict prefix. At least one known input remains so
// selection runs normally; this never synthesizes a selected pending token.
// Lookup is cold admission work and does not change recency on failed retries.
loom_serve_text_prefix_entry_t* loom_serve_text_prefix_cache_find(
    loom_serve_text_prefix_cache_t* cache, iree_host_size_t count,
    const int32_t* tokens);

// Records one successful admission using an endpoint, not a lookup attempt.
void loom_serve_text_prefix_cache_touch(loom_serve_text_prefix_cache_t* cache,
                                        loom_serve_text_prefix_entry_t* entry);

// Captures a natural intermediate prefill frontier unless an ancestor already
// represents it. The row has completed accepted work; its active completion
// reservation survives capture. A full table evicts its oldest unprotected
// entry. Capacity denial is ordinary: a zero frontier means no submitted pin.
// Any returned frontier must retire before lookup, eviction or further model
// mutation. Failure leaves accepted native ownership with the outer model.
iree_status_t loom_serve_text_prefix_cache_submit(
    loom_serve_text_prefix_cache_t* cache, loom_serve_text_row_t* row,
    iree_host_size_t count, const int32_t* tokens,
    const loom_serve_text_prefix_entry_t* protected_entry, bool* out_evicted,
    loom_serve_execution_frontier_t* out_frontier);

// Releases one unborrowed endpoint at a successfully retired model boundary.
// The caller schedules physical trim after logical release.
void loom_serve_text_prefix_cache_release(
    loom_serve_text_prefix_cache_t* cache,
    loom_serve_text_prefix_entry_t* entry);
bool loom_serve_text_prefix_cache_evict(
    loom_serve_text_prefix_cache_t* cache,
    const loom_serve_text_prefix_entry_t* protected_entry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_PREFIX_CACHE_H_
