// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-provided register-pressure policy shared by low codegen.

#ifndef LOOM_CODEGEN_LOW_PRESSURE_H_
#define LOOM_CODEGEN_LOW_PRESSURE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// One target-provided register-pressure cliff.
//
// A live or allocated footprint at or above |cliff_units| occupies
// |tier_after| instead of |tier_before| target throughput tiers. The owning
// table defines whether |pressure_source_id| names a descriptor register class
// or a derived resource.
typedef struct loom_low_pressure_cliff_t {
  // Table-local register-class or derived-resource pressure source ID.
  uint16_t pressure_source_id;
  // Live or allocated units at which this cliff is crossed.
  uint32_t cliff_units;
  // Occupancy or throughput tier before crossing the cliff.
  uint32_t tier_before;
  // Occupancy or throughput tier after crossing the cliff.
  uint32_t tier_after;
} loom_low_pressure_cliff_t;

// Contiguous pressure-cliff rows for one descriptor register class.
typedef struct loom_low_pressure_cliff_range_t {
  // First pressure-cliff row for the register class.
  uint32_t start;
  // Number of pressure-cliff rows for the register class.
  uint32_t count;
} loom_low_pressure_cliff_range_t;

// Immutable target pressure-cliff table.
//
// |values| are sorted by pressure_source_id and then cliff_units.
// |ranges| is dense by descriptor register-class ID and permits direct lookup
// from scheduler and allocator hot paths.
typedef struct loom_low_pressure_cliff_table_t {
  // Pressure-cliff rows sorted by register class and unit threshold.
  const loom_low_pressure_cliff_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
  // Dense pressure-cliff ranges indexed by descriptor register-class ID.
  const loom_low_pressure_cliff_range_t* ranges;
} loom_low_pressure_cliff_table_t;

// One register-class contribution to a derived pressure resource.
typedef struct loom_low_pressure_resource_member_t {
  // Dense resource ID owning this member.
  uint16_t resource_id;
  // Descriptor-set-local contributing register-class ID.
  uint16_t descriptor_reg_class_id;
  // Member units are rounded up to this granularity before aggregation.
  uint32_t contribution_granularity;
} loom_low_pressure_resource_member_t;

// One target-defined pressure resource derived from register-class extents.
typedef struct loom_low_pressure_resource_t {
  // Stable target resource name used by diagnostics and reports.
  iree_string_view_t name;
  // Total resource units shared by resident target execution units.
  uint32_t pool_units;
  // Aggregate units are rounded up to this allocation granularity.
  uint32_t allocation_granularity;
  // First member row in loom_low_pressure_resource_table_t::members.
  uint16_t member_start;
  // Number of contiguous member rows owned by this resource.
  uint16_t member_count;
  // First cliff row in loom_low_pressure_resource_table_t::cliffs.
  uint16_t cliff_start;
  // Number of contiguous cliff rows owned by this resource.
  uint16_t cliff_count;
} loom_low_pressure_resource_t;

// Contiguous reverse-index rows for one descriptor register class.
typedef struct loom_low_pressure_resource_member_range_t {
  // First member-index row for the register class.
  uint16_t start;
  // Number of member-index rows for the register class.
  uint16_t count;
} loom_low_pressure_resource_member_range_t;

// Immutable target pressure resources and their register-class memberships.
//
// Resource members are grouped by resource for cold reporting and occupancy
// evaluation. |member_indices_by_reg_class| and
// |member_ranges_by_reg_class| form a dense generated reverse index used by
// scheduler hot paths.
typedef struct loom_low_pressure_resource_table_t {
  // Derived pressure-resource rows in stable target order.
  const loom_low_pressure_resource_t* resources;
  // Member rows grouped by resource.
  const loom_low_pressure_resource_member_t* members;
  // Resource cliff rows grouped by resource and increasing unit threshold.
  const loom_low_pressure_cliff_t* cliffs;
  // Member-row indices grouped by descriptor register class.
  const uint16_t* member_indices_by_reg_class;
  // Dense reverse-index ranges by descriptor register-class ID.
  const loom_low_pressure_resource_member_range_t* member_ranges_by_reg_class;
  // Number of entries in |resources|.
  uint16_t resource_count;
} loom_low_pressure_resource_table_t;

// Immutable target pressure policy shared by scheduling and final resource
// accounting.
typedef struct loom_low_pressure_model_t {
  // Direct pressure cliffs indexed by descriptor register-class ID.
  loom_low_pressure_cliff_table_t register_class_cliffs;
  // Derived pressure resources and reverse membership indexes.
  loom_low_pressure_resource_table_t resources;
} loom_low_pressure_model_t;

static inline loom_low_pressure_cliff_table_t
loom_low_pressure_cliff_table_empty(void) {
  return (loom_low_pressure_cliff_table_t){0};
}

static inline bool loom_low_pressure_cliff_table_is_empty(
    loom_low_pressure_cliff_table_t table) {
  return table.count == 0;
}

static inline loom_low_pressure_cliff_range_t
loom_low_pressure_cliff_table_range(
    const loom_low_pressure_cliff_table_t* table,
    uint16_t descriptor_reg_class_id) {
  return table->ranges[descriptor_reg_class_id];
}

static inline bool loom_low_pressure_resource_table_is_empty(
    const loom_low_pressure_resource_table_t* table) {
  return table->resource_count == 0;
}

// Returns true when |model| contributes no direct or derived pressure policy.
static inline bool loom_low_pressure_model_is_empty(
    const loom_low_pressure_model_t* model) {
  return model == NULL ||
         (loom_low_pressure_cliff_table_is_empty(
              model->register_class_cliffs) &&
          loom_low_pressure_resource_table_is_empty(&model->resources));
}

// Rounds one derived-resource member contribution without wrapping.
static inline uint64_t loom_low_pressure_round_resource_units(
    uint64_t units, uint32_t granularity) {
  IREE_ASSERT_NE(granularity, 0);
  const uint64_t round_mask = (uint64_t)granularity - 1u;
  if ((granularity & (granularity - 1u)) == 0) {
    return units > UINT64_MAX - round_mask ? UINT64_MAX
                                           : (units + round_mask) & ~round_mask;
  }
  const uint64_t remainder = units % granularity;
  if (remainder == 0) return units;
  const uint64_t delta = granularity - remainder;
  return units > UINT64_MAX - delta ? UINT64_MAX : units + delta;
}

static inline loom_low_pressure_resource_member_range_t
loom_low_pressure_resource_table_member_range(
    const loom_low_pressure_resource_table_t* table,
    uint16_t descriptor_reg_class_id) {
  return table->member_ranges_by_reg_class[descriptor_reg_class_id];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PRESSURE_H_
