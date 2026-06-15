// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assignment lookup by liveness-local value identity.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_ASSIGNMENT_MAP_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_ASSIGNMENT_MAP_H_

#include "iree/base/api.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Completed assignment lookup table over the current local value domain.
typedef struct loom_low_allocation_assignment_map_t {
  // Module containing the allocated low function.
  const loom_module_t* module;
  // Liveness analysis whose value IDs define the local ordinal domain.
  const loom_liveness_analysis_t* liveness;
  // Completed assignment records.
  const loom_low_allocation_assignment_t* assignments;
  // Number of records in |assignments|.
  iree_host_size_t assignment_count;
  // Assignment indices by liveness local value ordinal.
  const uint32_t* assignment_indices_by_value_ordinal;
} loom_low_allocation_assignment_map_t;

// Maps a value ID to the local liveness ordinal when the value belongs to the
// allocated function body.
bool loom_low_allocation_assignment_map_value_ordinal_for_value(
    const loom_low_allocation_assignment_map_t* map, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal);

// Returns the current assignment for |value_ordinal|, or NULL when the value
// has not been assigned.
const loom_low_allocation_assignment_t*
loom_low_allocation_assignment_map_assignment_for_value_ordinal(
    const loom_low_allocation_assignment_map_t* map,
    loom_value_ordinal_t value_ordinal, uint32_t* out_assignment_index);

// Requires a current assignment for |value_id| and reports an allocation
// construction failure when none exists. |out_assignment_index| and
// |out_assignment| may be NULL when the caller only needs one side of the
// lookup.
iree_status_t loom_low_allocation_assignment_map_require_assignment_for_value(
    const loom_low_allocation_assignment_map_t* map, loom_value_id_t value_id,
    uint32_t* out_assignment_index,
    const loom_low_allocation_assignment_t** out_assignment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_ASSIGNMENT_MAP_H_
