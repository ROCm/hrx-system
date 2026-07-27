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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MOVE_H_
