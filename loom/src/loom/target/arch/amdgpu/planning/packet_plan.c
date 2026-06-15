// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/packet_plan.h"

#include "loom/codegen/low/packet.h"

iree_status_t loom_amdgpu_packet_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_packet_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_packet_plan_t){0};
  if (schedule == NULL || allocation == NULL || arena == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "schedule, allocation, and arena are required for "
                            "AMDGPU packet planning");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));

  *out_plan = (loom_amdgpu_packet_plan_t){
      .schedule = schedule,
      .allocation = allocation,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_build(schedule, allocation, arena,
                                                   &out_plan->wait_plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_plan_build(
      &out_plan->wait_plan, arena, &out_plan->wait_packets));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_plan_build(
      schedule, allocation, arena, &out_plan->wait_states));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_build(
      schedule, allocation, &out_plan->wait_packets, &out_plan->wait_states,
      arena, &out_plan->vopd_plan));
  return iree_ok_status();
}

iree_status_t loom_amdgpu_packet_plan_verify(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_packet_plan_t* plan) {
  if (plan == NULL) {
    return iree_ok_status();
  }
  if (plan->schedule != schedule || plan->allocation != allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU packet plan must be derived from the "
                            "emitted schedule and allocation");
  }
  if (plan->wait_plan.schedule != schedule ||
      plan->wait_plan.allocation != allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU packet plan wait-counter table must use "
                            "the emitted schedule and allocation");
  }
  if (plan->wait_plan.progress.schedule != schedule ||
      plan->wait_plan.progress.allocation != allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU packet plan wait-counter progress table "
                            "must use the emitted schedule and allocation");
  }
  if (plan->wait_plan.hazard_plan.schedule != schedule ||
      plan->wait_plan.hazard_plan.allocation != allocation ||
      plan->wait_plan.hazard_plan.progress != &plan->wait_plan.progress) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU packet plan wait-counter hazard plan must "
                            "use the emitted schedule, allocation, and "
                            "progress table");
  }
  if (plan->wait_packets.wait_plan != &plan->wait_plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU packet plan wait-packet table must be "
                            "derived from its wait-counter table");
  }
  if (plan->wait_states.schedule != schedule ||
      plan->wait_states.allocation != allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU packet plan fixed wait-state table must "
                            "use the emitted schedule and allocation");
  }
  if (plan->wait_states.progress.schedule != schedule ||
      plan->wait_states.progress.allocation != allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU packet plan fixed wait-state progress "
                            "table must use the emitted schedule and "
                            "allocation");
  }
  if (plan->wait_states.hazard_plan.schedule != schedule ||
      plan->wait_states.hazard_plan.allocation != allocation ||
      plan->wait_states.hazard_plan.progress != &plan->wait_states.progress) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU packet plan fixed wait-state hazard plan "
                            "must use the emitted schedule, allocation, and "
                            "progress table");
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vopd_plan_verify(schedule, allocation, &plan->vopd_plan));
  return loom_amdgpu_vopd_plan_verify_wait_insertions(
      &plan->vopd_plan, &plan->wait_packets, &plan->wait_states);
}

uint64_t loom_amdgpu_packet_plan_instruction_count(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_packet_plan_t* plan) {
  if (schedule == NULL) {
    return 0;
  }
  if (plan == NULL) {
    return schedule->scheduled_node_count;
  }
  const uint64_t wait_packet_count = plan->wait_packets.packet_count;
  const uint64_t wait_state_instruction_count =
      loom_amdgpu_wait_state_plan_instruction_count(&plan->wait_states);
  const uint64_t vopd_pair_count = plan->vopd_plan.pair_count;
  return schedule->scheduled_node_count + wait_packet_count +
         wait_state_instruction_count - vopd_pair_count;
}
