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
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_storage_release_policy_e {
  // Active storage leases are hard conflicts.
  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN = 0,
  // Active storage leases may be released before the candidate assignment.
  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED = 1,
} loom_low_allocation_storage_release_policy_t;

typedef struct loom_low_allocation_storage_lease_unit_entry_t
    loom_low_allocation_storage_lease_unit_entry_t;

// Hash index over units owned by materialized storage leases.
typedef struct loom_low_allocation_storage_lease_unit_index_t {
  // Bucket heads into |entries|. Missing buckets contain UINT32_MAX.
  uint32_t* bucket_heads;
  // Power-of-two number of entries in |bucket_heads|.
  uint32_t bucket_count;
  // Unit entries stored for materialized storage leases.
  loom_low_allocation_storage_lease_unit_entry_t* entries;
  // Maximum number of entries that can be appended to |entries|.
  iree_host_size_t entry_capacity;
  // Number of initialized entries in |entries|.
  iree_host_size_t entry_count;
} loom_low_allocation_storage_lease_unit_index_t;

// Mutable allocation-side lease state derived from a storage-lease table.
typedef struct loom_low_allocation_storage_lease_state_t {
  // Borrowed storage-lease table being materialized.
  const loom_low_storage_lease_table_t* lease_table;
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
  // Hash index for materialized register-like storage-lease units.
  loom_low_allocation_storage_lease_unit_index_t units;
  // Number of initialized assignment-backed storage-lease records.
  iree_host_size_t instance_count;
  // Number of initialized storage release actions.
  iree_host_size_t release_action_count;
} loom_low_allocation_storage_lease_state_t;

// Initializes |out_state| and builds the value-to-lease-record index for
// |lease_table|. Empty lease tables leave |out_state| inert.
iree_status_t loom_low_allocation_storage_lease_state_initialize(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_storage_lease_state_t* out_state);

// Returns true when |candidate| conflicts with materialized storage leases
// under |policy|.
bool loom_low_allocation_storage_lease_state_conflicts(
    const loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    loom_low_allocation_storage_release_policy_t policy);

// Appends release actions for every materialized lease conflicting with
// |candidate|. All conflicts must be legally releasable before |candidate|.
iree_status_t loom_low_allocation_storage_lease_state_record_release_actions(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count);

// Materializes every storage lease attached to |value_ordinal| using
// |assignment_index|'s concrete physical storage.
iree_status_t loom_low_allocation_storage_lease_state_record_assignment(
    loom_low_allocation_storage_lease_state_t* state,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t assignment_index, loom_value_ordinal_t value_ordinal);

// Verifies that every input storage-lease record was materialized exactly once.
iree_status_t loom_low_allocation_storage_lease_state_finalize(
    const loom_low_allocation_storage_lease_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_LEASE_H_
