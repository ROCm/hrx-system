// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical candidate preferences retained across scalar storage affinities.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_PHYSICAL_DOMAINS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_PHYSICAL_DOMAINS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable preferences for one allocation attempt. Full scalar affinities
// carry their common physical domain and storage horizon. Narrower overlapping
// domains discourage a broader value from occupying their candidates. These
// preferences never establish interference or permission to alias storage.
typedef struct loom_low_allocation_physical_domains_t {
  // Word offsets indexed by liveness interval index. UINT32_MAX denotes an
  // inert row; NULL denotes a function without physical scalar candidates.
  const uint32_t* offsets;
  // One penalty bit per semantic candidate ordinal in the interval's class.
  // Rows are word-aligned and have ceil(allocatable_count / 64) words.
  const uint64_t* words;
} loom_low_allocation_physical_domains_t;

// Returns the retained candidate-penalty row for an interval, or NULL when
// no preference row is needed. Bits address semantic candidate
// ordinals in the interval's register class, not physical-register IDs.
const uint64_t* loom_low_allocation_physical_domains_for_interval(
    const loom_low_allocation_physical_domains_t* domains,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_interval_t* interval);

// Builds preferences from final producer-owned placement and unit liveness.
// Inputs are borrowed; all retained storage belongs to |arena|. Construction
// scratch is released before returning. Changed input facts require rebuilding
// the plan. Linear classes and aggregate views have no preference rows.
iree_status_t loom_low_allocation_physical_domains_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    loom_low_allocation_physical_domains_t* out_domains);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_PHYSICAL_DOMAINS_H_
