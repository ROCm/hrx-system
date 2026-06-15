// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU wait-counter planning over scheduled target-low functions.
//
// The shared low scheduler records target-neutral descriptor facts in scheduled
// order. This layer owns the AMDGPU interpretation of those facts: memory
// packets create outstanding wait-counter work, explicit wait packets drain
// counters, and missing waits are reported as planned insertions before the
// packet that needs the wait. The plan is a table only; IR materialization is
// a later target-owned pass.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet_hazard_plan.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// AMDGPU target counter ids used by the current descriptor overlays.
enum loom_amdgpu_wait_counter_e {
  // Descriptor did not name a concrete AMDGPU wait counter.
  LOOM_AMDGPU_WAIT_COUNTER_NONE = 0,
  // VMEM/global load-result dependency counter.
  LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD = 1,
  // VMEM/global store completion counter.
  LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE = 2,
  // LDS/DS dependency and completion counter.
  LOOM_AMDGPU_WAIT_COUNTER_LDS = 3,
  // Scalar-memory dependency counter.
  LOOM_AMDGPU_WAIT_COUNTER_SMEM = 4,
  // ALU dependency counter used by depctr-style wait packets.
  LOOM_AMDGPU_WAIT_COUNTER_ALU = 5,
};

// Bit masks for AMDGPU wait counters. These are descriptor-overlay ids, not
// native instruction bit encodings.
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD ((uint32_t)1u << 0)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE ((uint32_t)1u << 1)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS ((uint32_t)1u << 2)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM ((uint32_t)1u << 3)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU ((uint32_t)1u << 4)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM   \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD | \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_READ   \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD | \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS | LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE | LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_WORKGROUP \
  LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_READ | LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY | LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU)

typedef enum loom_amdgpu_wait_plan_action_kind_e {
  // Unknown or uninitialized action kind.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_UNKNOWN = 0,
  // Wait already exists in the scheduled low stream.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT = 1,
  // Wait must be inserted before final emission.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED = 2,
} loom_amdgpu_wait_plan_action_kind_t;

typedef enum loom_amdgpu_wait_plan_reason_e {
  // Unknown or uninitialized wait reason.
  LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN = 0,
  // Explicit wait packet in the low stream drains this counter.
  LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET = 1,
  // A consumer uses a value produced by an outstanding memory load.
  LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE = 2,
  // A barrier observes memory that may still have outstanding packets.
  LOOM_AMDGPU_WAIT_PLAN_REASON_BARRIER = 3,
  // A packet reuses physical VGPRs still consumed by an outstanding store.
  LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE = 4,
  // A packet overwrites physical registers that still receive an outstanding
  // memory-read result.
  LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE = 5,
  // An RDNA VALU consumes a nearby TRANS result before va_vdst is drained.
  LOOM_AMDGPU_WAIT_PLAN_REASON_TRANS_RESULT_USE = 6,
  // A GFX12 ALU packet reads SGPR state with outstanding ALU dependencies.
  LOOM_AMDGPU_WAIT_PLAN_REASON_VALU_SGPR_READ = 7,
} loom_amdgpu_wait_plan_reason_t;

typedef enum loom_amdgpu_wait_plan_residual_action_e {
  // Unknown or uninitialized residual action id.
  LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_UNKNOWN = 0,
  // Insert an AMDGPU wait packet before the affected packet.
  LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET = 1,
} loom_amdgpu_wait_plan_residual_action_t;

// One AMDGPU wait-counter action in scheduled packet order.
typedef struct loom_amdgpu_wait_plan_action_t {
  // Whether the action is present in the IR or must be inserted.
  loom_amdgpu_wait_plan_action_kind_t kind;
  // Why this wait action exists.
  loom_amdgpu_wait_plan_reason_t reason;
  // AMDGPU wait counter affected by the action.
  uint16_t counter_id;
  // Wait target value. The first planning slice always drains to zero.
  uint16_t target_count;
  // Region block containing the insertion point or explicit wait.
  uint32_t block_index;
  // Node before which a planned wait is inserted, or explicit wait node.
  uint32_t node_index;
  // Scheduled ordinal before which a planned wait is inserted, or the explicit
  // wait node's scheduled ordinal.
  uint32_t scheduled_ordinal;
  // Producer node that forced the wait, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t producer_node;
  // Consumer node that needs the wait, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t consumer_node;
  // Outstanding packet count for this counter before the wait action.
  uint32_t outstanding_before;
} loom_amdgpu_wait_plan_action_t;

// AMDGPU wait-counter table for one scheduled low function.
typedef struct loom_amdgpu_wait_plan_t {
  // Schedule table this plan was built from.
  const loom_low_schedule_table_t* schedule;
  // Optional allocation table used for physical-assignment hazards.
  const loom_low_allocation_table_t* allocation;
  // Canonical counter-progress sidecar for explicit waits and producers.
  loom_low_packet_progress_table_t progress;
  // Canonical residual hazard sidecar for planned wait actions.
  loom_low_packet_hazard_plan_t hazard_plan;
  // AMDGPU implementation action rows in scheduled packet order.
  const loom_amdgpu_wait_plan_action_t* actions;
  // Number of action records.
  iree_host_size_t action_count;
} loom_amdgpu_wait_plan_t;

// Returns the stable diagnostic spelling for an AMDGPU wait counter id.
iree_string_view_t loom_amdgpu_wait_counter_name(uint16_t counter_id);

// Returns the stable progress-class spelling for an AMDGPU wait counter id.
iree_string_view_t loom_amdgpu_wait_counter_progress_class_name(
    uint16_t counter_id);

// Returns the bit mask for one AMDGPU wait counter id.
iree_status_t loom_amdgpu_wait_counter_mask(uint16_t counter_id,
                                            uint32_t* out_mask);

// Returns the stable diagnostic spelling for an AMDGPU wait-plan reason.
iree_string_view_t loom_amdgpu_wait_plan_reason_name(
    loom_amdgpu_wait_plan_reason_t reason);

// Returns the stable diagnostic spelling for an AMDGPU residual action id.
iree_string_view_t loom_amdgpu_wait_plan_residual_action_name(
    uint16_t action_id);

// Builds an AMDGPU wait-counter plan from a scheduled low function. When
// |allocation| is provided, the plan also materializes target storage-release
// actions requested by allocation, such as outstanding memory reads whose
// destination registers have not yet been written and VMEM stores whose VGPR
// sources have not yet been consumed by the memory pipe. The caller must keep
// |schedule|, |allocation|, and |arena| immutable/alive for as long as
// |out_plan| is used.
iree_status_t loom_amdgpu_wait_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_wait_plan_t* out_plan);

// Appends the common packet hazard JSON representation of |plan| to |builder|.
iree_status_t loom_amdgpu_wait_plan_format_json(
    const loom_amdgpu_wait_plan_t* plan, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PLAN_H_
