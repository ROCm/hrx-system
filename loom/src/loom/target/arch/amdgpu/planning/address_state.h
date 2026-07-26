// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU post-allocation address-state packet planning.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_ADDRESS_STATE_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_ADDRESS_STATE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet.h"

#ifdef __cplusplus
extern "C" {
#endif

// Address-state bits required while encoding one scheduled packet.
typedef struct loom_amdgpu_address_state_requirement_t {
  // Two-bit S_SET_VGPR_MSB slot mask in low-immediate layout.
  uint8_t mask;
  // Two-bit S_SET_VGPR_MSB slot values in low-immediate layout.
  uint8_t value;
} loom_amdgpu_address_state_requirement_t;

// One S_SET_VGPR_MSB packet inserted before a scheduled node.
typedef struct loom_amdgpu_address_state_transition_t {
  // Region block ordinal containing the insertion point.
  uint32_t block_index;
  // Schedule node receiving the transition immediately before it.
  uint32_t node_index;
  // Scheduled ordinal of |node_index| within |block_index|.
  uint32_t scheduled_ordinal;
  // Packed previous/new S_SET_VGPR_MSB mode immediate.
  uint16_t mode_immediate;
  // Reserved for stable layout; must be zero.
  uint16_t reserved;
} loom_amdgpu_address_state_transition_t;

// Concrete address-state packet insertions for one scheduled allocation.
typedef struct loom_amdgpu_address_state_plan_t {
  // Schedule table this plan was built from.
  const loom_low_schedule_table_t* schedule;
  // Allocation table this plan was built from.
  const loom_low_allocation_table_t* allocation;
  // Planned transitions in scheduled insertion order.
  const loom_amdgpu_address_state_transition_t* transitions;
  // Number of entries in |transitions|.
  iree_host_size_t transition_count;
} loom_amdgpu_address_state_plan_t;

// Returns the address-state bits required by a verified |packet| from an
// addressability-accepted emission frame. Structural packets return an empty
// requirement; their target-generated move sequences manage and restore
// address state inside native emission.
loom_amdgpu_address_state_requirement_t
loom_amdgpu_address_state_requirement_for_packet(
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet);

// Builds concrete S_SET_VGPR_MSB insertions for an addressability-accepted
// emission frame in scheduled order. Generated transitions are a
// post-allocation packet overlay: they do not mutate low IR, reschedule the
// function, or invalidate the allocation that selected them.
iree_status_t loom_amdgpu_address_state_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_address_state_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_ADDRESS_STATE_H_
