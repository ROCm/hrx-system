// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-resource pressure projected from structured source IR.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_PRESSURE_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_PRESSURE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t loom_low_source_pressure_option_flags_t;
enum loom_low_source_pressure_option_flag_bits_e {
  // The supplied contributors bound all non-source target storage pressure.
  LOOM_LOW_SOURCE_PRESSURE_OPTION_FLAG_RESERVES_COMPLETE = 1u << 0,
};

// Options controlling one source pressure projection.
typedef struct loom_low_source_pressure_options_t {
  // Nested function region whose execution defines the pressure scope. Null
  // analyzes the whole function body.
  const loom_region_t* region;
  // Additional named reserve contributors supplied by the caller.
  const loom_low_lower_pressure_reserve_t* reserves;
  // Number of entries in |reserves|.
  iree_host_size_t reserve_count;
  // Completeness claims for the supplied projection inputs.
  loom_low_source_pressure_option_flags_t flags;
} loom_low_source_pressure_options_t;

static inline loom_low_source_pressure_options_t
loom_low_source_pressure_options_empty(void) {
  return (loom_low_source_pressure_options_t){0};
}

typedef uint16_t loom_low_source_pressure_flags_t;
enum loom_low_source_pressure_flag_bits_e {
  // The selected lowering policy supplied a target residency model.
  LOOM_LOW_SOURCE_PRESSURE_FLAG_MODEL_AVAILABLE = 1u << 0,
  // Source values and named reserves form a complete pressure projection.
  LOOM_LOW_SOURCE_PRESSURE_FLAG_PROJECTION_COMPLETE = 1u << 1,
};

// Whole-function target pressure envelope over structured source liveness.
//
// Arrays and copied reserve records are owned by the analysis arena. Direct
// resource IDs match descriptor register-class IDs by lowering-policy contract.
typedef struct loom_low_source_pressure_t {
  // Model availability and projection completeness bits.
  loom_low_source_pressure_flags_t flags;
  // Target-owned model used to evaluate this pressure envelope, or NULL.
  const loom_target_residency_model_t* residency_model;
  // Worst residency tier reached at any feasible structured program point.
  uint32_t minimum_tier;
  // Earliest program point attaining |minimum_tier|.
  uint32_t minimum_tier_point;
  // Maximum units observed independently for each direct resource.
  const uint64_t* peak_direct_resource_units;
  // Simultaneously live resource vector at |minimum_tier_point|.
  const uint64_t* minimum_tier_direct_resource_units;
  // Sum of all named reserves applied at every program point.
  const uint64_t* reserved_direct_resource_units;
  // Number of entries in each direct-resource vector.
  iree_host_size_t direct_resource_count;
  // Arena-owned copy of named reserve contributors.
  const loom_low_lower_pressure_reserve_t* reserves;
  // Number of entries in |reserves|.
  iree_host_size_t reserve_count;
  // Source values mapped to target register resources.
  iree_host_size_t mapped_value_count;
  // Live source values explicitly mapped to no target register.
  iree_host_size_t non_register_value_count;
  // Mapped sparse liveness segments swept by the analysis.
  iree_host_size_t live_segment_count;
  // Leaf-operation result issue segments included in |live_segment_count|.
  iree_host_size_t transient_segment_count;
} loom_low_source_pressure_t;

static inline bool loom_low_source_pressure_model_available(
    const loom_low_source_pressure_t* pressure) {
  return pressure != NULL &&
         iree_any_bit_set(pressure->flags,
                          LOOM_LOW_SOURCE_PRESSURE_FLAG_MODEL_AVAILABLE);
}

static inline bool loom_low_source_pressure_projection_complete(
    const loom_low_source_pressure_t* pressure) {
  return pressure != NULL &&
         iree_any_bit_set(pressure->flags,
                          LOOM_LOW_SOURCE_PRESSURE_FLAG_PROJECTION_COMPLETE);
}

// Projects structured source liveness into the selected target's residency
// resources.
//
// |environment| must identify one source function and descriptor set. The
// analysis reuses an active region-tree value domain and view-region table when
// supplied, otherwise it builds scoped instances. Mutating the source module
// invalidates the returned envelope.
iree_status_t loom_low_source_pressure_analyze(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_policy_t* policy,
    const loom_low_source_pressure_options_t* options,
    iree_arena_allocator_t* arena, loom_low_source_pressure_t* out_pressure);

// Projects several nested function regions from one shared liveness and target
// mapping analysis.
//
// Each null entry in |regions| selects the whole function body. The scoped
// region in |options| is ignored because the explicit region array owns scope.
// Output records may share immutable reserve contributor storage while their
// peak and limiting resource vectors remain scope-specific.
iree_status_t loom_low_source_pressure_analyze_regions(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_policy_t* policy,
    const loom_low_source_pressure_options_t* options,
    const loom_region_t* const* regions, iree_host_size_t region_count,
    iree_arena_allocator_t* arena, loom_low_source_pressure_t* out_pressures);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_PRESSURE_H_
