// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-owned physical move rows.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_MOVE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_MOVE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/assignment.h"

#ifdef __cplusplus
extern "C" {
#endif

// One target-visible allocation unit.
typedef struct loom_low_move_location_t {
  // Target-visible storage kind.
  loom_low_allocation_location_kind_t location_kind;
  // Storage class for the unit.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Physical register, target ID, or spill slot ordinal.
  uint32_t location;
} loom_low_move_location_t;

// One physical move from an old source unit to a destination unit.
typedef struct loom_low_move_t {
  // Unit overwritten by the move.
  loom_low_move_location_t destination;
  // Unit read by the move.
  loom_low_move_location_t source;
} loom_low_move_t;

// Contiguous range in an allocation-owned move row table.
typedef struct loom_low_move_range_t {
  // First move row in the table.
  iree_host_size_t start;
  // Number of move rows in the range.
  iree_host_size_t count;
} loom_low_move_range_t;

// Final move rows and cycle-scratch metadata for one parallel move group.
typedef struct loom_low_move_group_t {
  // Sequential physical moves emitted for the group.
  loom_low_move_range_t moves;
  // First entry in the allocation scratch-move index table.
  iree_host_size_t scratch_move_index_start;
  // Number of distinct cycle-scratch locations used by the group.
  iree_host_size_t scratch_move_index_count;
} loom_low_move_group_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MOVE_H_
