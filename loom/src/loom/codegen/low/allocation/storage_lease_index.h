// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Indexed temporal and selected physical storage leases.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_LEASE_INDEX_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_LEASE_INDEX_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_storage_lease_index_node_t
    loom_low_allocation_storage_lease_index_node_t;

// Physical-unit and temporal radix indexes over materialized register leases.
// Construction and endpoint updates belong to allocation. After allocation,
// the index and its borrowed instances are immutable and support independent
// caller-owned selections. Each radix path has at most 64 branches.
struct loom_low_allocation_storage_lease_unit_index_t {
  // Borrowed assignment-backed instances, indexed by storage-lease ordinal.
  const loom_low_allocation_storage_lease_t* instances;
  // Physical-register and target-ID unit-directory roots, or UINT32_MAX.
  uint32_t unit_roots[2];
  // Radix nodes shared by unit directories and their temporal trees.
  loom_low_allocation_storage_lease_index_node_t* nodes;
  // Number of initialized radix nodes.
  uint32_t node_count;
  // Fixed capacity covering directories, temporal branches and lease leaves.
  uint32_t node_capacity;
  // First temporal leaf for each lease, or UINT32_MAX before materialization.
  uint32_t* first_nodes_by_lease;
};

// Membership over an immutable lease index. The owning analysis updates this
// summary at the same boundary that establishes or removes lease membership.
typedef struct loom_low_allocation_storage_lease_selection_t {
  // Borrowed immutable index whose temporal nodes define the summary domain.
  const loom_low_allocation_storage_lease_unit_index_t* index;
  // Selected leaf counts for temporal subtrees; directory entries stay zero.
  uint32_t* subtree_counts;
} loom_low_allocation_storage_lease_selection_t;

// Cursor over physical units and matching temporal or selected leases. Units
// are visited in ascending order; lease order within a unit is unspecified.
// A lease spanning several queried units may be returned more than once.
typedef struct loom_low_allocation_storage_lease_unit_query_t {
  // Borrowed index being queried.
  const loom_low_allocation_storage_lease_unit_index_t* index;
  // Optional membership unioned with temporal matches, even outside time.
  const loom_low_allocation_storage_lease_selection_t* selection;
  // Target-storage identity key shared by aliasing register classes.
  uint32_t storage_key;
  // Physical-register or target-ID unit-directory ordinal.
  uint32_t unit_root_ordinal;
  // First physical unit in the queried range.
  uint32_t location_base;
  // Number of physical units in the queried range.
  uint32_t location_count;
  // Next physical unit offset not yet visited.
  uint32_t next_unit_offset;
  // Physical unit whose temporal tree is currently being visited.
  uint32_t active_location;
  // Inclusive lower bound on a matching lease's end point.
  uint64_t minimum_end_point;
  // Exclusive upper bound on a matching lease's start point. The widened
  // domain represents one past UINT32_MAX without wrapping.
  uint64_t start_point_limit;
  // Number of initialized entries in the pending-node stack.
  uint32_t stack_count;
  // Pending temporal subtrees; strictly decreasing radix bits bound depth.
  uint32_t stack[65];
} loom_low_allocation_storage_lease_unit_query_t;

// Initializes fixed storage for up to |lease_unit_capacity| materialized units
// of |lease_count| borrowed instances. |distinct_unit_capacity| bounds distinct
// (storage key, location kind, location) tuples across those units. All
// inserted leases are register-like and obey these producer-owned capacity
// bounds.
iree_status_t loom_low_allocation_storage_lease_unit_index_initialize(
    loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_allocation_storage_lease_t* instances,
    iree_host_size_t lease_count, iree_host_size_t lease_unit_capacity,
    iree_host_size_t distinct_unit_capacity, iree_arena_allocator_t* arena);

// Publishes one initialized instance, once, within the planned unit capacity.
void loom_low_allocation_storage_lease_unit_index_insert(
    loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t storage_lease_index);

// Retains a changed endpoint for a materialized lease. Cursors remain valid
// while endpoints shorten. Construction is complete before any selection is
// initialized.
void loom_low_allocation_storage_lease_unit_index_update(
    loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t storage_lease_index);

// Returns true when storage exists for physical-unit queries.
bool loom_low_allocation_storage_lease_unit_index_is_enabled(
    const loom_low_allocation_storage_lease_unit_index_t* index);

// Initializes an empty membership summary for a completed index.
iree_status_t loom_low_allocation_storage_lease_selection_initialize(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    iree_arena_allocator_t* arena,
    loom_low_allocation_storage_lease_selection_t* out_selection);

// Records one membership transition. The previous membership must differ from
// |active|; the producer owns deduplication when merging or draining bitsets.
// Membership can shrink during a query; additions occur between queries.
void loom_low_allocation_storage_lease_selection_set_active(
    loom_low_allocation_storage_lease_selection_t* selection,
    uint32_t storage_lease_index, bool active);

// Initializes an exact physical-range query. Temporal matches have
// start < start_point_limit and end >= minimum_end_point. The bounds are
// independent: nonempty [start, end) overlap uses (start + 1, end), while an
// inclusive point query uses (point, point + 1). Widened endpoints preserve
// zero and maximum-u32 boundaries. Selection adds matches outside those bounds.
// An allocator-requested release shortens a lease's endpoint; the remaining
// interval still participates in queries for earlier conflicting writes.
void loom_low_allocation_storage_lease_unit_query_initialize(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, uint64_t minimum_end_point,
    uint64_t start_point_limit,
    const loom_low_allocation_storage_lease_selection_t* selection,
    loom_low_allocation_storage_lease_unit_query_t* out_query);

// Advances without allocation and returns the next matching lease ordinal.
bool loom_low_allocation_storage_lease_unit_query_next(
    loom_low_allocation_storage_lease_unit_query_t* query,
    uint32_t* out_storage_lease_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_LEASE_INDEX_H_
