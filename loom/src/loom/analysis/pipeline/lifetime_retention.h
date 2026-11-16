// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_LIFETIME_RETENTION_H_
#define LOOM_ANALYSIS_PIPELINE_LIFETIME_RETENTION_H_

#include "loom/analysis/pipeline/lifetime_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// An interchangeable immutable resource currently copied into several groups.
// Lengths use the caller's uniform allocation unit, including required padding.
typedef struct loom_pipeline_lifetime_retention_item_t {
  // Length of one physical copy in allocation units.
  uint64_t length;
  // Distinct request ordinals containing a copy, in any order.
  const iree_host_size_t* groups;
  // Number of copies; zero marks an unused input ordinal.
  iree_host_size_t group_count;
} loom_pipeline_lifetime_retention_item_t;

typedef struct loom_pipeline_lifetime_retention_policy_t {
  // Enclosing construction's storage, live throughout every item's use.
  iree_host_size_t resident_group;
  // Required remaining units in each nonresident group after any move.
  uint64_t minimum_group_length;
  // Maximum simultaneous footprint in allocation units.
  uint64_t capacity;
} loom_pipeline_lifetime_retention_policy_t;

// Selects immutable items to keep in the enclosing lifetime instead of copying
// them into each child. Candidates are ordered by nonresident volume removed,
// with input ordinal breaking ties. Each item receives one capacity proposal;
// an accepted proposal replaces all of its copies with one resident copy.
// This is a bounded greedy policy, not optimal packing or a visit-rate model.
//
// Requests retain the prepared projection's identities/order. This call writes
// their lengths and uses unit alignment; all demand comes from the item copies.
// The producer supplies representable total storage and unique memberships.
// Caller-owned out_retained has item_count slots. False preserves the original
// memberships; true selects resident_group for every use of that item.
// The projection is reusable for final placement with exact lengths/alignment.
// Selection does not admit the final footprint; the physical owner checks that
// bound even when no candidate fits. Status reports allocation or extent
// errors.
iree_status_t loom_pipeline_lifetime_retain(
    loom_pipeline_lifetime_projection_t* projection,
    loom_pipeline_lifetime_request_t* requests, iree_host_size_t request_count,
    const loom_pipeline_lifetime_retention_item_t* items,
    iree_host_size_t item_count,
    const loom_pipeline_lifetime_retention_policy_t* policy,
    iree_arena_allocator_t* arena, bool* out_retained);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_LIFETIME_RETENTION_H_
