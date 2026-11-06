// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whole-program AMDGPU resources retained across native function compilation.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_RESOURCES_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_RESOURCES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/emit/native/amdgpu/function_plan.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// Complete resource facts for one entry and its retained helper closure.
typedef struct loom_amdgpu_program_resources_t {
  // Shared fixed-segment sizes after placing every native function.
  loom_amdgpu_storage_layout_segment_sizes_t storage_segment_sizes;
  // Maximum allocatable physical extent indexed by descriptor register class.
  const uint32_t* direct_resource_units;
  // Number of entries in |direct_resource_units|.
  iree_host_size_t direct_resource_unit_count;
  // Minimum final SGPR count across entry, helpers, and call control.
  uint32_t sgpr_count;
  // Minimum final architectural VGPR count across entry and helpers.
  uint32_t vgpr_count;
  // Minimum final AGPR count across entry and helpers.
  uint32_t agpr_count;
} loom_amdgpu_program_resources_t;

// Mutable resource accumulator used while compiling one native program.
typedef struct loom_amdgpu_program_resource_builder_t {
  // Maximum segment extents already occupied by compiled functions.
  loom_amdgpu_storage_layout_segment_sizes_t storage_segment_sizes;
  // Arena-owned maximum physical extents indexed by descriptor register class.
  uint32_t* direct_resource_units;
  // Number of entries in |direct_resource_units|.
  iree_host_size_t direct_resource_unit_count;
  // Minimum final SGPR count across entry, helpers, and call control.
  uint32_t sgpr_count;
  // Minimum final architectural VGPR count across entry and helpers.
  uint32_t vgpr_count;
  // Minimum final AGPR count across entry and helpers.
  uint32_t agpr_count;
} loom_amdgpu_program_resource_builder_t;

// Initializes an empty resource accumulator for one descriptor set.
iree_status_t loom_amdgpu_program_resource_builder_initialize(
    iree_host_size_t direct_resource_unit_count, iree_arena_allocator_t* arena,
    loom_amdgpu_program_resource_builder_t* out_builder);

// Includes one compiled function's register demand and final segment extents.
// Shared leaf helpers have mutually exclusive private frames beginning at zero.
// Compile the entry with their maximum extent as its storage prefix before
// including its final resources here. No storage addresses are reassigned.
void loom_amdgpu_program_resource_builder_require_function(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_function_plan_t* function_plan,
    loom_amdgpu_program_resource_builder_t* builder);

// Adds register floors used outside ordinary allocated function storage, such
// as fixed argument carriers and direct-call control registers.
void loom_amdgpu_program_resource_builder_require_registers(
    uint32_t sgpr_count, uint32_t vgpr_count, uint32_t agpr_count,
    loom_amdgpu_program_resource_builder_t* builder);

// Publishes an immutable view borrowing storage from |builder|.
void loom_amdgpu_program_resource_builder_finish(
    const loom_amdgpu_program_resource_builder_t* builder,
    loom_amdgpu_program_resources_t* out_resources);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_RESOURCES_H_
