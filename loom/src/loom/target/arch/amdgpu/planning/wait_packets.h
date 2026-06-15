// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU wait packet materialization over logical wait-counter plans.
//
// wait_plan.h records target-counter actions in scheduled order. This layer
// maps those logical actions to concrete wait packet descriptors for one
// descriptor set. That split keeps schedule analysis independent of target
// packet spelling while still giving all emit paths the same coalesced
// insertion plan for combined s_waitcnt targets and split load/store/ALU wait
// targets.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKETS_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKETS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/target/arch/amdgpu/planning/wait_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_wait_packet_immediate_t {
  // Descriptor-local immediate index populated by this row.
  uint16_t descriptor_immediate_index;
  // Borrowed immediate field name from the selected descriptor set.
  iree_string_view_t name;
  // Concrete immediate value to materialize.
  uint16_t value;
} loom_amdgpu_wait_packet_immediate_t;

#define LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY 4

// One concrete AMDGPU wait packet selected for an immediate counter drain.
typedef struct loom_amdgpu_wait_packet_selection_t {
  // Borrowed descriptor row selected from the descriptor set.
  const loom_low_descriptor_t* descriptor;
  // Logical counter mask concretely drained by this packet.
  uint32_t counter_mask;
  // Immediate rows to materialize on the selected descriptor.
  loom_amdgpu_wait_packet_immediate_t
      immediates[LOOM_AMDGPU_WAIT_PACKET_SELECTION_IMMEDIATE_CAPACITY];
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
} loom_amdgpu_wait_packet_selection_t;

// One planned concrete AMDGPU wait packet to insert before a scheduled node.
typedef struct loom_amdgpu_wait_packet_t {
  // Borrowed descriptor row selected from |wait_plan|'s descriptor set.
  const loom_low_descriptor_t* descriptor;
  // Region block containing the insertion point.
  uint32_t block_index;
  // Schedule node before which the packet is inserted.
  uint32_t node_index;
  // Scheduled ordinal before which the packet is inserted.
  uint32_t scheduled_ordinal;
  // Logical counter mask concretely drained by this packet.
  uint32_t counter_mask;
  // First logical wait action in the coalesced insertion group.
  iree_host_size_t source_action_start;
  // Number of logical wait actions in the coalesced insertion group.
  iree_host_size_t source_action_count;
  // First immediate row owned by this packet.
  iree_host_size_t immediate_start;
  // Number of immediate rows owned by this packet.
  iree_host_size_t immediate_count;
} loom_amdgpu_wait_packet_t;

// Concrete wait-packet insertion table for one scheduled AMDGPU function.
typedef struct loom_amdgpu_wait_packet_plan_t {
  // Logical wait plan this concrete packet plan was built from.
  const loom_amdgpu_wait_plan_t* wait_plan;
  // Planned concrete wait packets in scheduled insertion order.
  const loom_amdgpu_wait_packet_t* packets;
  // Number of concrete packet rows.
  iree_host_size_t packet_count;
  // Immediate rows referenced by packet records.
  const loom_amdgpu_wait_packet_immediate_t* immediates;
  // Number of immediate rows.
  iree_host_size_t immediate_count;
} loom_amdgpu_wait_packet_plan_t;

// Selects one concrete wait packet that drains |counter_mask| to
// |target_count| on |descriptor_set|.
iree_status_t loom_amdgpu_wait_packet_select_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t counter_mask,
    uint16_t target_count, loom_amdgpu_wait_packet_selection_t* out_selection);

// Tries to select one concrete wait packet that drains |counter_mask| to
// |target_count| on |descriptor_set|. Missing target coverage is reported as
// |out_selected| false; malformed descriptor tables remain status failures.
iree_status_t loom_amdgpu_wait_packet_try_select_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t counter_mask,
    uint16_t target_count, loom_amdgpu_wait_packet_selection_t* out_selection,
    bool* out_selected);

// Builds concrete AMDGPU wait packet insertions from |wait_plan|. The caller
// must keep |wait_plan->schedule| immutable and |arena| alive for as long as
// |out_plan| is used.
iree_status_t loom_amdgpu_wait_packet_plan_build(
    const loom_amdgpu_wait_plan_t* wait_plan, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_packet_plan_t* out_plan);

// Appends a compact JSON representation of |plan| to |builder|.
iree_status_t loom_amdgpu_wait_packet_plan_format_json(
    const loom_amdgpu_wait_packet_plan_t* plan, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PACKETS_H_
