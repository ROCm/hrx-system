// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final AIE2P VLIW bundle planning over a scheduled and allocated Low frame.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_PLAN_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel used when a physical slot was synthesized by the target planner.
#define LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE UINT32_MAX

// Architectural core program-memory capacity in bytes.
#define LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE 16384u

enum loom_aie2p_planned_slot_flag_bits_e {
  // Slot is a planner-synthesized architectural NOP.
  LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP = 1u << 0,
  // Slot materializes a structural Low control operation.
  LOOM_AIE2P_PLANNED_SLOT_FLAG_STRUCTURAL_CONTROL = 1u << 1,
};
typedef uint16_t loom_aie2p_planned_slot_flags_t;

// One physical instruction assigned to a bundle slot.
typedef struct loom_aie2p_planned_slot_t {
  // Encoded instruction value and its physical slot.
  loom_aie2p_encoded_slot_t encoded_slot;
  // Scheduled Low packet represented by this slot, or PACKET_NONE.
  uint32_t scheduled_packet_index;
  // Planner provenance flags.
  loom_aie2p_planned_slot_flags_t flags;
} loom_aie2p_planned_slot_t;

// One variable-width physical VLIW bundle.
typedef struct loom_aie2p_planned_bundle_t {
  // Physical issue cycle in the contiguous core program.
  uint32_t issue_cycle;
  // First slot record in the owning plan.
  uint32_t slot_start;
  // Physical bundle format selected from the owned target table.
  loom_aie2p_bundle_format_id_t format;
  // Number of slot records in this bundle.
  uint8_t slot_count;
} loom_aie2p_planned_bundle_t;

// Immutable AIE2P physical packet plan for one Low function.
typedef struct loom_aie2p_bundle_plan_t {
  // Emission frame from which this plan was built.
  const loom_low_emission_frame_t* frame;
  // Physical bundles in increasing issue-cycle order.
  const loom_aie2p_planned_bundle_t* bundles;
  // Number of physical bundles.
  iree_host_size_t bundle_count;
  // Encoded physical slots grouped by |bundles|.
  const loom_aie2p_planned_slot_t* slots;
  // Number of encoded physical slots.
  iree_host_size_t slot_count;
  // Exact byte length after variable-width bundle packing.
  iree_host_size_t encoded_byte_length;
} loom_aie2p_bundle_plan_t;

// Plans physical bundles for one successful, spill-free AIE2P Low frame.
//
// The current core program representation accepts one basic block. Structural
// low.return is materialized as the architectural RET instruction and hoisted
// over useful delay-slot work when the physical bundle table permits it. Empty
// issue cycles are materialized as NOP bundles so schedule cycles remain exact.
// The returned plan borrows |frame| and owns its tables in |arena|.
iree_status_t loom_aie2p_bundle_plan_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_bundle_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_PLAN_H_
