// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Producer reconstruction facts shared by allocation and spill repair.

#ifndef LOOM_CODEGEN_LOW_REMATERIALIZATION_FACTS_H_
#define LOOM_CODEGEN_LOW_REMATERIALIZATION_FACTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_low_rematerialization_value_flag_bits_e {
  // All structural input edges have been classified.
  LOOM_LOW_REMATERIALIZATION_VALUE_CLASSIFIED = 1u << 0,
  // The producer permits independent recreation of this result.
  LOOM_LOW_REMATERIALIZATION_VALUE_ELIGIBLE = 1u << 1,
  // The producer is input-free or composes closed values of the same class
  // without narrowing a wider input.
  LOOM_LOW_REMATERIALIZATION_VALUE_CLOSED = 1u << 2,
  // At least one operand use consumes this value's storage.
  LOOM_LOW_REMATERIALIZATION_VALUE_CONSUMED = 1u << 3,
};

// One immutable function snapshot. The caller owns the module ordinal map
// populated with exactly |value_ids| for the duration of classification.
typedef struct loom_low_rematerialization_scope_t {
  // Module containing the original producers and their uses.
  loom_module_t* module;
  // Target descriptor contract governing producer recreation.
  const loom_low_resolved_target_t* target;
  // Borrowed function-local IDs indexed by the acquired module ordinals.
  const loom_value_id_t* value_ids;
  // Number of entries in the borrowed local value domain.
  iree_host_size_t value_count;
  // Retained definition points, or NULL before allocation has run.
  const loom_liveness_analysis_t* liveness;
  // Function body delimiting definitions when |liveness| is NULL.
  const loom_region_t* region;
} loom_low_rematerialization_scope_t;

// Classifies producer eligibility and consuming uses once. When requested,
// follows structural composition edges once to classify closed recipes too.
// Only the byte-per-value result survives; traversal scratch is rewound.
iree_status_t loom_low_rematerialization_facts_build(
    const loom_low_rematerialization_scope_t* scope, bool reconstruct_constants,
    iree_arena_allocator_t* arena, uint8_t** out_value_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REMATERIALIZATION_FACTS_H_
