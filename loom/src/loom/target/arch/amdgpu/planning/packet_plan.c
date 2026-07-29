// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/packet_plan.h"

iree_status_t loom_amdgpu_packet_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_packet_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_packet_plan_t){0};
  *out_plan = (loom_amdgpu_packet_plan_t){
      .schedule = schedule,
      .allocation = allocation,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_state_plan_build(
      schedule, allocation, arena, &out_plan->address_state));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_build(schedule, allocation, arena,
                                                   &out_plan->wait_plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_plan_build(
      &out_plan->wait_plan, arena, &out_plan->wait_packets));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_build(
      schedule, allocation, &out_plan->address_state, &out_plan->wait_packets,
      arena, &out_plan->vopd_plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_plan_build(
      schedule, allocation, &out_plan->vopd_plan, arena,
      &out_plan->wait_states));
  return iree_ok_status();
}
