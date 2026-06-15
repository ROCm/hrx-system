// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU occupancy and register-pressure diagnostics over target-low allocation
// tables.
//
// The generic low allocator reports physical register assignment and predicted
// spills without knowing target occupancy rules. This layer owns the AMDGPU
// interpretation of those facts: per-register-file high-water use, spill
// pressure in scratch, wave occupancy estimates, and next-cliff feedback for
// search/tuning loops.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_OCCUPANCY_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_OCCUPANCY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for no limiting register class.
#define LOOM_AMDGPU_OCCUPANCY_CLASS_NONE UINT32_MAX

// Sentinel for no limiting occupancy resource.
#define LOOM_AMDGPU_OCCUPANCY_RESOURCE_NONE UINT32_MAX

typedef enum loom_amdgpu_occupancy_limiting_resource_kind_e {
  // Occupancy is capped only by the target's maximum wave count.
  LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_MAX_WAVES = 0,
  // Occupancy is limited by a physical register class.
  LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_REGISTER_CLASS = 1,
  // Occupancy is limited by a derived target pressure resource.
  LOOM_AMDGPU_OCCUPANCY_LIMITING_RESOURCE_PRESSURE_RESOURCE = 2,
} loom_amdgpu_occupancy_limiting_resource_kind_t;

typedef enum loom_amdgpu_occupancy_diagnostic_bits_e {
  // Emits BACKEND/010 remarks summarizing estimated occupancy.
  LOOM_AMDGPU_OCCUPANCY_DIAGNOSTIC_SUMMARY = 1u << 0,
} loom_amdgpu_occupancy_diagnostic_bits_t;

// Bitset of loom_amdgpu_occupancy_diagnostic_bits_t values.
typedef uint32_t loom_amdgpu_occupancy_diagnostic_flags_t;

// Options controlling AMDGPU occupancy table construction.
typedef struct loom_amdgpu_occupancy_options_t {
  // Structured diagnostic emitter for occupancy feedback.
  iree_diagnostic_emitter_t emitter;
  // Optional structured occupancy feedback to emit.
  loom_amdgpu_occupancy_diagnostic_flags_t diagnostic_flags;
} loom_amdgpu_occupancy_options_t;

// Occupancy facts for one AMDGPU register class.
typedef struct loom_amdgpu_occupancy_register_class_t {
  // Stable register-class name such as "amdgpu.vgpr".
  iree_string_view_t register_class;
  // Descriptor-set-local register class ID for |register_class|.
  uint16_t descriptor_reg_class_id;
  // Highest allocated physical register unit plus one.
  uint32_t allocated_units;
  // Allocation units rounded according to the occupancy model.
  uint32_t rounded_units;
  // Target register-file capacity available to waves for this class.
  uint32_t pool_units;
  // Allocation granularity applied before estimating occupancy.
  uint32_t allocation_granularity;
  // Maximum resident waves allowed by this register class.
  uint32_t wave_limit;
  // Smallest allocated unit count that would reduce |wave_limit|, or 0 when
  // this class is already at zero occupancy or has no lower modeled cliff.
  uint32_t next_cliff_units;
  // Additional units available before |next_cliff_units|, or UINT32_MAX when no
  // lower modeled cliff exists.
  uint32_t units_until_next_cliff;
  // Number of spilled assignments for this register class.
  uint32_t spill_count;
  // Total scratch spill-slot bytes for this register class.
  uint32_t spill_bytes;
  // Predicted spill stores for this register class.
  uint32_t spill_store_count;
  // Predicted spill reloads for this register class.
  uint32_t spill_reload_count;
} loom_amdgpu_occupancy_register_class_t;

// Occupancy facts for one AMDGPU target pressure resource.
typedef struct loom_amdgpu_occupancy_pressure_resource_t {
  // Stable resource name such as "amdgpu.vgpr_agpr".
  iree_string_view_t resource;
  // Highest allocated physical unit count after member contribution rounding.
  uint32_t allocated_units;
  // Allocation units rounded according to the occupancy model.
  uint32_t rounded_units;
  // Target resource capacity available to waves.
  uint32_t pool_units;
  // Allocation granularity applied before estimating occupancy.
  uint32_t allocation_granularity;
  // Maximum resident waves allowed by this resource.
  uint32_t wave_limit;
  // Smallest allocated unit count that would reduce |wave_limit|, or 0 when
  // this resource is already at zero occupancy or has no lower modeled cliff.
  uint32_t next_cliff_units;
  // Additional units available before |next_cliff_units|, or UINT32_MAX when no
  // lower modeled cliff exists.
  uint32_t units_until_next_cliff;
} loom_amdgpu_occupancy_pressure_resource_t;

// AMDGPU occupancy table for one allocated target-low function body.
typedef struct loom_amdgpu_occupancy_table_t {
  // Allocation table this occupancy estimate was built from.
  const loom_low_allocation_table_t* allocation;
  // AMDGPU processor selected by the low target snapshot.
  iree_string_view_t processor;
  // AMDGPU wave size used by this model.
  uint32_t wave_size;
  // Maximum resident waves per SIMD in this model.
  uint32_t max_waves_per_simd;
  // Fixed flat workgroup size selected by the target export plan, or 0 when the
  // export ABI does not provide one.
  uint32_t flat_workgroup_size;
  // Number of waves required by one workgroup, or 0 when the workgroup size is
  // unavailable.
  uint32_t waves_per_workgroup;
  // Estimated resident waves per SIMD after register limits.
  uint32_t resident_waves_per_simd;
  // Estimated resident wave occupancy as a percentage of |max_waves_per_simd|.
  uint32_t occupancy_percent;
  // Kind of resource that limited occupancy.
  loom_amdgpu_occupancy_limiting_resource_kind_t limiting_resource_kind;
  // Index into |register_classes| or |pressure_resources| according to
  // |limiting_resource_kind|, or LOOM_AMDGPU_OCCUPANCY_RESOURCE_NONE when
  // occupancy is capped by max waves.
  uint32_t limiting_resource_index;
  // Per-register-class summaries in target model order.
  const loom_amdgpu_occupancy_register_class_t* register_classes;
  // Number of records in |register_classes|.
  iree_host_size_t register_class_count;
  // Derived target pressure resources in target model order.
  const loom_amdgpu_occupancy_pressure_resource_t* pressure_resources;
  // Number of records in |pressure_resources|.
  iree_host_size_t pressure_resource_count;
  // Number of spilled assignments across all AMDGPU register classes.
  uint32_t spill_count;
  // Total scratch spill-slot bytes across all AMDGPU register classes.
  uint32_t scratch_spill_bytes;
  // Predicted spill stores across all AMDGPU register classes.
  uint32_t spill_store_count;
  // Predicted spill reloads across all AMDGPU register classes.
  uint32_t spill_reload_count;
} loom_amdgpu_occupancy_table_t;

// Builds an AMDGPU occupancy estimate from |allocation|. The caller must keep
// |allocation| immutable and |arena| alive for as long as |out_table| is
// used.
iree_status_t loom_amdgpu_occupancy_build(
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_occupancy_options_t* options,
    iree_arena_allocator_t* arena, loom_amdgpu_occupancy_table_t* out_table);

// Builds target-provided schedule pressure cliffs for |descriptor_set|. The
// returned list is sorted by descriptor register-class ID and cliff unit count
// and is suitable for loom_low_schedule_options_t::pressure_cliffs.
iree_status_t loom_amdgpu_occupancy_build_schedule_pressure_cliffs(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_pressure_cliff_list_t* out_pressure_cliffs);

// Appends a compact JSON representation of |table| to |builder|.
iree_status_t loom_amdgpu_occupancy_format_json(
    const loom_amdgpu_occupancy_table_t* table, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_OCCUPANCY_H_
