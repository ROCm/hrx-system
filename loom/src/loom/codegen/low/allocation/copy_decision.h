// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Post-allocation low.copy coalescing/materialization decisions.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_COPY_DECISION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_COPY_DECISION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable allocation facts needed to classify low.copy operations.
typedef struct loom_low_allocation_copy_decision_context_t {
  // Module containing the allocated low function.
  const loom_module_t* module;
  // Body region of the low function.
  loom_region_t* body;
  // Descriptor set selected by the low function target.
  const loom_low_descriptor_set_t* descriptor_set;
  // Completed assignment lookup map.
  loom_low_allocation_assignment_map_t assignment_map;
} loom_low_allocation_copy_decision_context_t;

// low.copy decision table rows and aggregate counts.
typedef struct loom_low_allocation_copy_decision_plan_t {
  // Copy/coalescing decisions in source order.
  loom_low_allocation_copy_decision_t* decisions;
  // Number of records in |decisions|.
  iree_host_size_t decision_count;
  // Number of low.copy operations that share one assigned location.
  iree_host_size_t coalesced_count;
  // Number of low.copy operations that require a materialized move/copy.
  iree_host_size_t materialized_count;
} loom_low_allocation_copy_decision_plan_t;

// Builds source-order low.copy decisions from completed assignment facts.
iree_status_t loom_low_allocation_copy_decision_plan_build(
    const loom_low_allocation_copy_decision_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_copy_decision_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_COPY_DECISION_H_
