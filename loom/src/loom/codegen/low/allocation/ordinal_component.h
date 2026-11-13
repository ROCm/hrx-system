// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transitive hard same-register-ordinal placement components.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_ORDINAL_COMPONENT_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_ORDINAL_COMPONENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_ordinal_components_t {
  // Component index by local value ordinal, or UINT32_MAX when unconstrained.
  uint32_t* indices_by_value_ordinal;
  // Packed member ranges indexed by component.
  loom_low_placement_ordinal_component_range_t* ranges;
  // Local value ordinals packed by |ranges|.
  loom_value_ordinal_t* members;
  // Number of entries in |ranges|.
  uint32_t count;
} loom_low_allocation_ordinal_components_t;

// Builds the transitive components established by hard register-ordinal
// relations. Tied-storage aliases join their origin's component so every
// published alias can move with an atomic recoloring.
iree_status_t loom_low_allocation_ordinal_components_build(
    loom_value_ordinal_t value_count,
    const loom_value_ordinal_t* tied_storage_origins_by_value_ordinal,
    const loom_low_placement_relation_t* relations,
    iree_host_size_t relation_count, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_ordinal_components_t* out_components);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ORDINAL_COMPONENT_H_
