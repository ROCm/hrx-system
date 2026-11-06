// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/program_resources.h"

#include <string.h>

static uint32_t loom_amdgpu_program_resources_max_u32(uint32_t lhs,
                                                      uint32_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

iree_status_t loom_amdgpu_program_resource_builder_initialize(
    iree_host_size_t direct_resource_unit_count, iree_arena_allocator_t* arena,
    loom_amdgpu_program_resource_builder_t* out_builder) {
  *out_builder = (loom_amdgpu_program_resource_builder_t){
      .direct_resource_unit_count = direct_resource_unit_count,
  };
  if (direct_resource_unit_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, direct_resource_unit_count,
                                  sizeof(*out_builder->direct_resource_units),
                                  (void**)&out_builder->direct_resource_units));
    memset(out_builder->direct_resource_units, 0,
           direct_resource_unit_count *
               sizeof(*out_builder->direct_resource_units));
  }
  return iree_ok_status();
}

void loom_amdgpu_program_resource_builder_require_function(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_function_plan_t* function_plan,
    loom_amdgpu_program_resource_builder_t* builder) {
  IREE_ASSERT_EQ(frame->allocation.physical_extents.count,
                 builder->direct_resource_unit_count);
  for (iree_host_size_t i = 0; i < builder->direct_resource_unit_count; ++i) {
    builder->direct_resource_units[i] = loom_amdgpu_program_resources_max_u32(
        builder->direct_resource_units[i],
        frame->allocation.physical_extents.ends_by_reg_class[i]);
  }
  builder->sgpr_count = loom_amdgpu_program_resources_max_u32(
      builder->sgpr_count, function_plan->preflight.next_free_sgpr);
  builder->vgpr_count = loom_amdgpu_program_resources_max_u32(
      builder->vgpr_count, function_plan->preflight.next_free_vgpr);
  builder->vgpr_count = loom_amdgpu_program_resources_max_u32(
      builder->vgpr_count,
      function_plan->packet_plan.wait_states.minimum_vgpr_count);
  builder->agpr_count = loom_amdgpu_program_resources_max_u32(
      builder->agpr_count, function_plan->preflight.next_free_agpr);
  builder->storage_segment_sizes.group_segment_fixed_size = iree_max(
      builder->storage_segment_sizes.group_segment_fixed_size,
      function_plan->storage_layout.segment_sizes.group_segment_fixed_size);
  builder->storage_segment_sizes.private_segment_fixed_size = iree_max(
      builder->storage_segment_sizes.private_segment_fixed_size,
      function_plan->storage_layout.segment_sizes.private_segment_fixed_size);
}

void loom_amdgpu_program_resource_builder_require_registers(
    uint32_t sgpr_count, uint32_t vgpr_count, uint32_t agpr_count,
    loom_amdgpu_program_resource_builder_t* builder) {
  builder->sgpr_count =
      loom_amdgpu_program_resources_max_u32(builder->sgpr_count, sgpr_count);
  builder->vgpr_count =
      loom_amdgpu_program_resources_max_u32(builder->vgpr_count, vgpr_count);
  builder->agpr_count =
      loom_amdgpu_program_resources_max_u32(builder->agpr_count, agpr_count);
}

void loom_amdgpu_program_resource_builder_finish(
    const loom_amdgpu_program_resource_builder_t* builder,
    loom_amdgpu_program_resources_t* out_resources) {
  *out_resources = (loom_amdgpu_program_resources_t){
      .storage_segment_sizes = builder->storage_segment_sizes,
      .direct_resource_units = builder->direct_resource_units,
      .direct_resource_unit_count = builder->direct_resource_unit_count,
      .sgpr_count = builder->sgpr_count,
      .vgpr_count = builder->vgpr_count,
      .agpr_count = builder->agpr_count,
  };
}
