// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_VARIANT_POOL_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_VARIANT_POOL_H_

#include "loom/runtime/wake.h"
#include "loom/serve/runtime/program/compilation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Process-lifetime identity, never reused across pools or model registrations.
// Zero is absent. Identities are not persistent names across server restarts.
typedef uint64_t loom_serve_variant_id_t;

typedef enum loom_serve_variant_phase_e {
  LOOM_SERVE_VARIANT_COMPILING = 0,
  LOOM_SERVE_VARIANT_READY,
  LOOM_SERVE_VARIANT_FAILED,
} loom_serve_variant_phase_t;

typedef struct loom_serve_variant_entry_t loom_serve_variant_entry_t;
typedef struct loom_serve_variant_pool_t loom_serve_variant_pool_t;

// Idle reuse policy owned by the consumer. One active construction/execution
// may exceed these limits and retire instead of entering the reusable cache.
typedef struct loom_serve_variant_cache_options_t {
  // Maximum idle executable choices; zero disables idle retention.
  iree_host_size_t capacity;
  // Maximum retained payload/entry/key/artifact bytes. Fixed source catalogs,
  // index capacity, compiler scratch, and driver-private storage are separate.
  iree_host_size_t max_bytes;
} loom_serve_variant_cache_options_t;

// Runs on the serialized owner after every producer and consumer has retired.
// The payload owns its compiled objects and resource references. NULL is valid
// for a base choice whose executable ownership remains with the model.
typedef void (*loom_serve_variant_dispose_fn_t)(void* value);

// Terminal compiler ownership transfer after its final model/input access.
// Takes status and payload even on failure. The report borrows payload storage
// during the callback. Publication may wake the owner, never mutate its model.
typedef void (*loom_serve_variant_completion_fn_t)(
    void* user_data, iree_status_t status, void* value,
    const loom_serve_compilation_report_t* report);

// Owner-thread observation. Message/report views borrow the entry until its
// removal; no compiler-owned data escapes. Compiling entries expose neither
// partially published results nor diagnostics still being written by a job.
typedef struct loom_serve_variant_snapshot_t {
  // Nonrecycled identity of this choice.
  loom_serve_variant_id_t id;
  // Ready choice against whose resources construction was requested; zero for
  // the original. This is provenance, not an executable merge or policy edge.
  loom_serve_variant_id_t base;
  // Compilation outcome, independent of any service selection policy.
  loom_serve_variant_phase_t phase;
  // Explicit consumer borrows, excluding the pool's own cached ownership.
  iree_host_size_t users;
  // Terminal construction error, or OK for pending/successful construction.
  iree_status_code_t code;
  // Borrowed terminal error message, empty before completion or on success.
  iree_string_view_t message;
  // Borrowed copied evidence, NULL until construction has completed.
  const loom_serve_compilation_report_t* compilation;
} loom_serve_variant_snapshot_t;

// Creates a bounded cache with one ready original choice in out_original.
// Takes original_value only on success. The returned entry is borrowed;
// retain it before storing a consumer reference. Capacity includes all ready,
// pending and failed entries. There is no selected/default variant: services
// retain choices for their epochs, requests or preferred policies. The pool
// never invokes a VM, changes parameter residency, or determines semantic
// compatibility.
iree_status_t loom_serve_variant_pool_allocate(
    iree_host_size_t capacity, void* original_value,
    loom_serve_variant_dispose_fn_t dispose, iree_allocator_t host_allocator,
    loom_serve_variant_entry_t** out_original,
    loom_serve_variant_pool_t** out_pool);

// All producer completions have been collected and all consumer borrows have
// been released. Disposes every payload before the model's resources retire.
// NULL is accepted.
void loom_serve_variant_pool_free(loom_serve_variant_pool_t* pool);

// Installs the owner's scheduling wake. Only completion publication uses this
// from another thread. Detachment follows collection of all accepted jobs.
void loom_serve_variant_pool_set_observer(
    loom_serve_variant_pool_t* pool, loom_runtime_wake_callback_t observer);

iree_host_size_t loom_serve_variant_pool_count(
    const loom_serve_variant_pool_t* pool);
// Requested bytes for pool and entry metadata, excluding consumer payloads.
iree_host_size_t loom_serve_variant_pool_storage_size(
    const loom_serve_variant_pool_t* pool);
bool loom_serve_variant_pool_pending(const loom_serve_variant_pool_t* pool);

// Owner-only linear traversal in cache order. Entries are borrowed until
// removal. Save next before discarding the current entry; consumer payloads
// remain unavailable until the entry's snapshot reports READY.
loom_serve_variant_entry_t* loom_serve_variant_pool_first(
    const loom_serve_variant_pool_t* pool);
loom_serve_variant_entry_t* loom_serve_variant_entry_next(
    const loom_serve_variant_entry_t* entry);
loom_serve_variant_snapshot_t loom_serve_variant_entry_snapshot(
    const loom_serve_variant_entry_t* entry);

// Moves a ready entry to the end of traversal order for owner-selected LRU.
// Management pools that never touch entries retain creation order.
void loom_serve_variant_entry_touch(loom_serve_variant_entry_t* entry);

// Clones the collected terminal construction status for a request consumer.
// READY yields OK; the caller consumes the returned status.
iree_status_t loom_serve_variant_entry_status(
    const loom_serve_variant_entry_t* entry);

// Reserves one construction against a ready same-pool base. The modality owns
// input capture, compilation and validation of its resource contract. Success
// requires exactly one complete call, including failure before job submission.
// The entry is stable until its completion has been collected. The producer
// borrows the enclosing model independently of the entry's base provenance.
iree_status_t loom_serve_variant_pool_begin(
    loom_serve_variant_pool_t* pool, loom_serve_variant_id_t base,
    loom_serve_variant_entry_t** out_entry);

// Transfers terminal status and payload after the producer's last model use.
// Copies report when non-NULL, then wakes the owner without selecting code.
// This is the only entry operation callable from a compiler callback. It must
// be the producer's final access to the entry and its enclosing pool.
void loom_serve_variant_entry_complete(
    loom_serve_variant_entry_t* entry, iree_status_t status, void* value,
    const loom_serve_compilation_report_t* report);

// Collects terminal producers on the serialized owner, returning actual
// progress. Failed payloads are disposed here; copied diagnostics stay cached.
// No allocation, wait, model submission or service selection occurs.
bool loom_serve_variant_pool_collect(loom_serve_variant_pool_t* pool);

// Acquires a ready choice by externally supplied identity. Selection policy
// and the semantic boundary at which it may run belong to the caller. After
// acquisition, id/value queries and release are infallible owner operations.
iree_status_t loom_serve_variant_pool_acquire(
    loom_serve_variant_pool_t* pool, loom_serve_variant_id_t id,
    loom_serve_variant_entry_t** out_entry);

// Retains/releases an explicit consumer borrow. NULL is accepted so partial
// construction and unsuccessful acquisition use the same cleanup path.
void loom_serve_variant_entry_release(loom_serve_variant_entry_t* entry);
void loom_serve_variant_entry_retain(loom_serve_variant_entry_t* entry);
loom_serve_variant_id_t loom_serve_variant_entry_id(
    const loom_serve_variant_entry_t* entry);
void* loom_serve_variant_entry_value(const loom_serve_variant_entry_t* entry);

// Drops one unborrowed terminal cached choice, including the original. Accepted
// producers and live consumers cannot be removed. There is no implicit pinning,
// LRU/default policy or ID recycling after removal.
iree_status_t loom_serve_variant_pool_remove(loom_serve_variant_pool_t* pool,
                                             loom_serve_variant_id_t id);

// Infallible owner eviction after observing a terminal entry with no consumer
// borrows. Does not search by identity, allocate, or wait. External removal
// requests use pool_remove to validate identity and lifetime.
void loom_serve_variant_entry_discard(loom_serve_variant_entry_t* entry);

// Visits stable owner-thread observations in cache order. False means the
// requested identity/index is absent, not a construction error.
bool loom_serve_variant_pool_snapshot(
    const loom_serve_variant_pool_t* pool, loom_serve_variant_id_t id,
    loom_serve_variant_snapshot_t* out_snapshot);
bool loom_serve_variant_pool_at(const loom_serve_variant_pool_t* pool,
                                iree_host_size_t index,
                                loom_serve_variant_snapshot_t* out_snapshot);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_VARIANT_POOL_H_
