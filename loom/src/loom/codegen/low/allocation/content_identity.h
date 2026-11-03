// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structural content identity and destructive-write provenance.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_CONTENT_IDENTITY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_CONTENT_IDENTITY_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_placement_table_t loom_low_placement_table_t;
typedef struct loom_low_allocation_unit_liveness_t
    loom_low_allocation_unit_liveness_t;

// Decision-lifetime content facts shared by interval assignment attempts.
// Content identity is independent of physical placement: optional copies and
// projections preserve bits, while destructive ties and control-flow edges
// do not. A written content component cannot justify new storage sharing.
typedef struct loom_low_allocation_content_identity_t {
  // Canonical content origins indexed by allocation-unit lifetime record.
  // NULL when the placement graph has no structural aliases.
  const uint32_t* origins;
  // Origins whose contents may be overwritten through a required storage tie.
  // Empty when the placement graph has no destructive writes.
  iree_bitmap_t written_origins;
} loom_low_allocation_content_identity_t;

// Resolves final structural aliases and destructive writes once, after
// destructive-reuse refinement. The result belongs to |arena|; callers request
// it only when storage leases or aggregate placement consume content facts.
iree_status_t loom_low_allocation_content_identity_initialize(
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena,
    loom_low_allocation_content_identity_t* out_identity);

// Returns true when two allocation units retain the same read-only contents.
// The caller establishes exact physical correspondence and execution coverage.
static inline bool loom_low_allocation_content_identity_units_share(
    const loom_low_allocation_content_identity_t* identity, uint32_t lhs_unit,
    uint32_t rhs_unit) {
  const uint32_t origin = identity->origins[lhs_unit];
  return origin == identity->origins[rhs_unit] &&
         (identity->written_origins.words == NULL ||
          !iree_bitmap_test(identity->written_origins, origin));
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_CONTENT_IDENTITY_H_
