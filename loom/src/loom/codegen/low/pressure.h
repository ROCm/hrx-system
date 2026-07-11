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
// Cliffs are keyed by descriptor-set-local register-class ID. A live or
// allocated footprint at or above |cliff_units| occupies |tier_after| instead
// of |tier_before| target throughput tiers.
typedef struct loom_low_pressure_cliff_t {
  // Descriptor-set-local register class affected by this cliff.
  uint16_t descriptor_reg_class_id;
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
// |values| are sorted by descriptor_reg_class_id and then cliff_units.
// |ranges| is dense by descriptor register-class ID and permits direct lookup
// from scheduler and allocator hot paths.
typedef struct loom_low_pressure_cliff_table_t {
  // Pressure-cliff rows sorted by register class and unit threshold.
  const loom_low_pressure_cliff_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
  // Dense pressure-cliff ranges indexed by descriptor register-class ID.
  const loom_low_pressure_cliff_range_t* ranges;
  // Number of entries in |ranges|.
  iree_host_size_t range_count;
} loom_low_pressure_cliff_table_t;

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
  IREE_ASSERT_LT(descriptor_reg_class_id, table->range_count);
  return table->ranges[descriptor_reg_class_id];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PRESSURE_H_
