// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/packet_plan.h"

#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/planning/matrix_coexecution.h"

iree_status_t loom_amdgpu_packet_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_packet_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_packet_plan_t){0};
  *out_plan = (loom_amdgpu_packet_plan_t){
      .schedule = schedule,
      .allocation = allocation,
  };
  iree_arena_allocator_t transient_arena;
  iree_arena_initialize(arena->block_pool, &transient_arena);
  const loom_amdgpu_processor_properties_t* processor_properties =
      loom_amdgpu_target_processor_properties_from_resolved_target(
          &schedule->target);
  const loom_amdgpu_matrix_coexecution_profile_t matrix_coexecution_profile =
      processor_properties != NULL
          ? processor_properties->features.matrix_coexecution
          : LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE;
  loom_amdgpu_matrix_coexecution_t* matrix_coexecution = NULL;
  iree_status_t status = loom_amdgpu_matrix_coexecution_allocate(
      schedule, allocation, matrix_coexecution_profile, &transient_arena,
      &matrix_coexecution);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_address_state_plan_build(schedule, allocation, arena,
                                                  &out_plan->address_state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_plan_build(
        schedule, allocation, arena, &transient_arena, &out_plan->wait_plan);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_packet_plan_build(&out_plan->wait_plan, arena,
                                                &out_plan->wait_packets);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_vopd_plan_build(
        schedule, allocation, processor_properties, &out_plan->address_state,
        &out_plan->wait_packets, matrix_coexecution, arena, &transient_arena,
        &out_plan->vopd_plan);
  }
  if (iree_status_is_ok(status) && matrix_coexecution != NULL) {
    status = loom_amdgpu_matrix_coexecution_finalize_static(matrix_coexecution);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_state_plan_build(
        schedule, allocation, processor_properties, &out_plan->vopd_plan,
        matrix_coexecution, arena, &transient_arena, &out_plan->wait_states);
  }
  iree_arena_deinitialize(&transient_arena);
  return status;
}
