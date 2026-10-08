// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-side materialization of target storage leases.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_LEASE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_LEASE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/storage_lease_index.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_placement_table_t loom_low_placement_table_t;

// Decision-lifetime structural content identity shared by every interval
// assignment attempt for one function. Empty lease tables and placement graphs
// without structural aliases leave this table inert.
typedef struct loom_low_allocation_storage_identity_t {
  // Canonical content origins indexed by allocation unit.
  const uint32_t* origins;
} loom_low_allocation_storage_identity_t;

// Resolves final structural aliases once, after destructive-reuse refinement.
// The result belongs to |arena| and can be shared by first-fit and repair
// assignment attempts.
iree_status_t loom_low_allocation_storage_identity_initialize(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena,
    loom_low_allocation_storage_identity_t* out_identity);

typedef enum loom_low_allocation_storage_release_policy_e {
  // Active storage leases are hard conflicts.
  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN = 0,
  // Pressure-friendly storage leases may be released before the candidate.
  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FOR_PRESSURE = 1,
  // Active storage leases may be released before the candidate assignment.
  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED = 2,
} loom_low_allocation_storage_release_policy_t;

// Mutable allocation-side lease state derived from a storage-lease table.
typedef struct loom_low_allocation_storage_lease_state_t {
  // Borrowed storage-lease table being materialized.
  const loom_low_storage_lease_table_t* lease_table;
  // Borrowed value domain used to map value IDs to allocation-local ordinals.
  const loom_local_value_domain_t* value_domain;
  // Borrowed allocation-owned physical lifetimes for candidate units.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Borrowed canonical content origins indexed by allocation unit.
  // NULL when this function has no structural aliases to compare with leases.
  const uint32_t* identity_origins;
  // Borrowed stable assignment array after the first leased assignment is
  // published. Every materialized lease indexes this array.
  const loom_low_allocation_assignment_t* assignments;
  // Mutable assignment-backed storage-lease records being built.
  loom_low_allocation_storage_lease_t* instances;
  // Mutable allocator-requested storage release actions being built.
  loom_low_storage_release_action_t* release_actions;
  // Storage-lease record heads indexed by allocation-local value ordinal.
  uint32_t* record_heads_by_value_ordinal;
  // Next storage-lease record index for the same allocation-local value.
  uint32_t* next_record_indices;
  // True when the storage-lease record has a materialized instance.
  uint8_t* instance_written;
  // Temporal index for materialized register-like storage-lease units.
  loom_low_allocation_storage_lease_unit_index_t* unit_index;
  // Min-heap of materialized lease ordinals keyed by mutable end point. The
  // heap is absent for small lease tables where linear probing is cheaper.
  uint32_t* availability_expiration_heap;
  // Heap positions by lease ordinal, or UINT32_MAX when not present.
  uint32_t* availability_expiration_positions;
  // Monotone candidate start represented by ordered availability summaries.
  uint32_t availability_start_point;
  // Number of initialized entries in |availability_expiration_heap|.
  uint32_t availability_expiration_count;
  // Number of initialized assignment-backed storage-lease records.
  iree_host_size_t instance_count;
  // Number of initialized storage release actions.
  iree_host_size_t release_action_count;
  // Number of storage-lease records marked releasable for pressure.
  iree_host_size_t pressure_release_record_count;
} loom_low_allocation_storage_lease_state_t;

// Initializes |out_state| and builds the value-to-lease-record index for
// |lease_table|. Empty lease tables leave |out_state| inert.
iree_status_t loom_low_allocation_storage_lease_state_initialize(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_storage_identity_t* storage_identity,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena,
    loom_low_allocation_storage_lease_state_t* out_state);

// Returns true when |candidate| conflicts with materialized storage leases
// under |policy|. Per-unit demand excludes unmaterialized storage; complete
// candidate storage segments additionally exclude lifetime holes.
bool loom_low_allocation_storage_lease_state_conflicts(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    loom_low_allocation_storage_release_policy_t policy);

// Returns true when ordered lease availability can prove definite conflicts
// for |candidate| under |policy|. The allowed policy remains candidate-specific
// and is never summarized. An explicit segment equal to the candidate's full
// interval is continuous. Self-root scalar candidates can use the summary;
// forwarded identities, lifetime holes, refined units, tuples, and explicit
// registers retain the exact conflict path.
bool loom_low_allocation_storage_lease_state_can_order_candidate(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate,
    loom_low_allocation_storage_release_policy_t policy);

// Finds the first location in the inclusive range that is not a definite lease
// conflict under |policy|. Calls advance the retained expiration frontier to
// the candidate's monotone start point. Other conflict sources and complete
// release legality still require the normal allocation predicate.
bool loom_low_allocation_storage_lease_state_find_next_available_location(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate,
    loom_low_allocation_storage_release_policy_t policy, uint32_t minimum_base,
    uint32_t maximum_base, uint32_t* out_base);
bool loom_low_allocation_storage_lease_state_find_previous_available_location(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* candidate,
    loom_low_allocation_storage_release_policy_t policy, uint32_t minimum_base,
    uint32_t maximum_base, uint32_t* out_base);

// Returns true when |value_id| has storage-lease records that must be
// materialized from a register-like assignment.
bool loom_low_allocation_storage_lease_state_value_has_records(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id);

// Records release actions for every materialized lease conflicting with
// |candidate|, moving a recorded release earlier when an aggregate reservation
// precedes allocation of its writers. Each lease retains its earliest release.
// All conflicts must be legally releasable before |candidate|.
iree_status_t loom_low_allocation_storage_lease_state_record_release_actions(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Publishes every storage lease indexed under |value_ordinal| into preallocated
// instances and the temporal unit index using |assignment_index|'s concrete
// register-like storage. Initialization established each record's value, and
// the lease producer established its unit subrange and issue-time start. The
// |assignments| is the stable preallocated interval-assignment array and the
// indexed assignment covers that subrange. |liveness| describes the lease
// schedule. Each value's assignment is published exactly once, including
// inherited ties.
void loom_low_allocation_storage_lease_state_record_assignment(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* assignments,
    uint32_t assignment_index, loom_value_ordinal_t value_ordinal);

// Verifies that every input storage-lease record was materialized exactly once.
iree_status_t loom_low_allocation_storage_lease_state_finalize(
    const loom_low_allocation_storage_lease_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_LEASE_H_
