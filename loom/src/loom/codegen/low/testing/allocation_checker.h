// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Independent test-only validation for accepted target-low allocation plans.

#ifndef LOOM_CODEGEN_LOW_TESTING_ALLOCATION_CHECKER_H_
#define LOOM_CODEGEN_LOW_TESTING_ALLOCATION_CHECKER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_check_violation_kind_e {
  // No allocation-plan violation was found.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_NONE = 0,
  // Schedule, allocation, or emission-frame ownership disagrees.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_FRAME_IDENTITY = 1,
  // Scheduled node order is incomplete, duplicated, or inconsistent.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_SCHEDULE_STRUCTURE = 2,
  // Allocation and liveness local value domains disagree.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_VALUE_DOMAIN = 3,
  // Dense value-to-assignment indexing is invalid or non-bijective.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_INDEX = 4,
  // An assignment disagrees with its source liveness interval.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_ASSIGNMENT_SHAPE = 5,
  // A fixed value does not occupy its resolved physical location.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_FIXED_LOCATION = 6,
  // An assignment overlaps a whole-function reserved location range.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_RESERVED_LOCATION = 7,
  // A hard placement relation is not satisfied by physical assignments.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_PLACEMENT_RELATION = 8,
  // Two simultaneously live values occupy unapproved aliasing storage.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_CONFLICT = 9,
  // An early-clobber result overlaps an untied input.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_EARLY_CLOBBER = 10,
  // A storage lease or its release action disagrees with its assignment.
  LOOM_LOW_ALLOCATION_CHECK_VIOLATION_STORAGE_LEASE = 11,
} loom_low_allocation_check_violation_kind_t;

// Compact witness for one allocation-plan violation.
typedef struct loom_low_allocation_check_violation_t {
  // Violation category.
  loom_low_allocation_check_violation_kind_t kind;
  // Primary table index associated with the violation, or UINT32_MAX.
  uint32_t primary_index;
  // Secondary table index associated with the violation, or UINT32_MAX.
  uint32_t secondary_index;
  // Primary SSA value associated with the violation, or VALUE_ID_INVALID.
  loom_value_id_t value_id;
  // Related SSA value associated with the violation, or VALUE_ID_INVALID.
  loom_value_id_t related_value_id;
  // Program point associated with the violation, or UINT32_MAX.
  uint32_t program_point;
} loom_low_allocation_check_violation_t;

// Aggregate independent checker result.
typedef struct loom_low_allocation_check_result_t {
  // Total violations found. Saturates at UINT32_MAX.
  uint32_t violation_count;
  // First violation found in deterministic checker order.
  loom_low_allocation_check_violation_t first_violation;
} loom_low_allocation_check_result_t;

// Returns the stable spelling for |kind|.
iree_string_view_t loom_low_allocation_check_violation_kind_name(
    loom_low_allocation_check_violation_kind_t kind);

// Independently validates the accepted schedule and allocation sidecars in
// |frame|. Plan violations are returned through |out_result|; status is
// reserved for checker workspace allocation failures.
//
// This is test-only infrastructure. Production frame construction and target
// emission never invoke the checker or retain checker-specific state.
iree_status_t loom_low_allocation_check_frame(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_low_allocation_check_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TESTING_ALLOCATION_CHECKER_H_
