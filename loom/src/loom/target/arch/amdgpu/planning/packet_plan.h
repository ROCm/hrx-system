// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU post-schedule packet plan.
//
// The low schedule is target-independent and keeps one scheduled node per low
// packet. AMDGPU still needs target-owned edits before native emission:
// inserting wait packets, inserting fixed wait-state noops, and replacing
// native-adjacent VALU packets with VOPD packets. This plan is the single
// boundary for those stream edits so native emitters do not grow one option per
// hardware feature.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_PACKET_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_PACKET_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/planning/wait_plan.h"
#include "loom/target/arch/amdgpu/planning/wait_states.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_packet_plan_t {
  // Schedule table this plan was built from.
  const loom_low_schedule_table_t* schedule;
  // Allocation table this plan was built from.
  const loom_low_allocation_table_t* allocation;
  // Logical wait-counter actions in scheduled order.
  loom_amdgpu_wait_plan_t wait_plan;
  // Concrete wait packets inserted into the native packet stream.
  loom_amdgpu_wait_packet_plan_t wait_packets;
  // Concrete fixed wait states inserted into the native packet stream.
  loom_amdgpu_wait_state_plan_t wait_states;
  // Concrete VOPD pairings applied to native-adjacent scheduled packets.
  loom_amdgpu_vopd_plan_t vopd_plan;
} loom_amdgpu_packet_plan_t;

// Builds the target-owned AMDGPU packet plan for a scheduled and allocated low
// function. The caller must keep |schedule|, |allocation|, and |arena|
// immutable/alive for as long as |out_plan| is used.
iree_status_t loom_amdgpu_packet_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_packet_plan_t* out_plan);

// Verifies that |plan| describes |schedule| and |allocation|.
iree_status_t loom_amdgpu_packet_plan_verify(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_packet_plan_t* plan);

// Returns the final native instruction count implied by |schedule| and |plan|.
// A NULL plan means no target-owned packet insertions or pair replacements.
uint64_t loom_amdgpu_packet_plan_instruction_count(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_packet_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_PACKET_PLAN_H_
