// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/assignment_map.h"

#include "loom/ir/module.h"

bool loom_low_allocation_assignment_map_value_ordinal_for_value(
    const loom_low_allocation_assignment_map_t* map, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(map->module, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= map->liveness->value_count) {
    return false;
  }
  *out_value_ordinal = value_ordinal;
  return true;
}

const loom_low_allocation_assignment_t*
loom_low_allocation_assignment_map_assignment_for_value_ordinal(
    const loom_low_allocation_assignment_map_t* map,
    loom_value_ordinal_t value_ordinal, uint32_t* out_assignment_index) {
  if (out_assignment_index) {
    *out_assignment_index = UINT32_MAX;
  }
  if (value_ordinal >= map->liveness->value_count) {
    return NULL;
  }
  const uint32_t assignment_index =
      map->assignment_indices_by_value_ordinal[value_ordinal];
  if (assignment_index == UINT32_MAX) {
    return NULL;
  }
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return &map->assignments[assignment_index];
}

iree_status_t loom_low_allocation_assignment_map_require_assignment_for_value(
    const loom_low_allocation_assignment_map_t* map, loom_value_id_t value_id,
    uint32_t* out_assignment_index,
    const loom_low_allocation_assignment_t** out_assignment) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (loom_low_allocation_assignment_map_value_ordinal_for_value(
          map, value_id, &value_ordinal)) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_assignment_map_assignment_for_value_ordinal(
            map, value_ordinal, out_assignment_index);
    if (assignment) {
      if (out_assignment) {
        *out_assignment = assignment;
      }
      return iree_ok_status();
    }
  }
  if (out_assignment_index) {
    *out_assignment_index = UINT32_MAX;
  }
  if (out_assignment) {
    *out_assignment = NULL;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "low allocation references value %u without an "
                          "assignment",
                          (unsigned)value_id);
}
