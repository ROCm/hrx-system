// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU VOPD packetization over scheduled target-low functions.
//
// VOPD is a target-owned post-allocation packetization decision: two VALU
// packets that remain adjacent in the emitted native instruction stream may
// become one native dual-issue packet when their descriptors, physical
// registers, and insertion points satisfy the architectural constraints. The
// plan records that final emission decision without changing the
// target-independent low schedule.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet.h"
#include "loom/target/arch/amdgpu/planning/address_state.h"
#include "loom/target/arch/amdgpu/planning/vopd_component.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for scheduled packets that do not belong to a VOPD pair.
#define LOOM_AMDGPU_VOPD_PAIR_NONE UINT32_MAX

typedef enum loom_amdgpu_vopd_packet_role_e {
  // Packet is not part of a VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE = 0,
  // Packet is the X component and emission point for a VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST = 1,
  // Packet is the Y component consumed by the previous VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND = 2,
} loom_amdgpu_vopd_packet_role_t;

typedef enum loom_amdgpu_vopd_rejection_reason_e {
  // Unknown or uninitialized VOPD rejection reason.
  LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN = 0,
  // Component opcodes cannot form one dual packet.
  LOOM_AMDGPU_VOPD_REJECTION_REASON_COMPONENT_OPCODE_MISMATCH = 1,
  // First component result is consumed by the second component.
  LOOM_AMDGPU_VOPD_REJECTION_REASON_FIRST_RESULT_USED_BY_SECOND = 2,
  // Component literal payloads cannot share one VOPD literal word.
  LOOM_AMDGPU_VOPD_REJECTION_REASON_LITERAL_MISMATCH = 3,
  // Physical register parity, bank, or cross-component constraints failed.
  LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS = 4,
  // Native wait insertion before the second component prevents fusion.
  LOOM_AMDGPU_VOPD_REJECTION_REASON_SECOND_PACKET_HAS_INSERTION = 5,
  // VOPD compression would invalidate the active TRANS-result wait proof.
  LOOM_AMDGPU_VOPD_REJECTION_REASON_TRANS_RESULT_WINDOW = 6,
} loom_amdgpu_vopd_rejection_reason_t;

typedef enum loom_amdgpu_vopd_register_constraint_flag_bits_e {
  // Component destination VGPRs have the same parity.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_DESTINATION_PARITY = 1u << 0,
  // Component SRC0 VGPRs occupy the same register bank.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC0_BANK = 1u << 1,
  // Component VSRC1 VGPRs occupy the same register bank.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_VSRC1_BANK = 1u << 2,
  // X destination aliases the Y SRC0 VGPR.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_SRC0 = 1u << 3,
  // X destination aliases the Y VSRC1 VGPR.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_VSRC1 = 1u << 4,
  // Y destination aliases the X SRC0 VGPR.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_SRC0 = 1u << 5,
  // Y destination aliases the X VSRC1 VGPR.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_VSRC1 = 1u << 6,
} loom_amdgpu_vopd_register_constraint_flag_bits_t;
typedef uint8_t loom_amdgpu_vopd_register_constraint_flags_t;

typedef enum loom_amdgpu_vopd_pair_flag_bits_e {
  // VOPD pair has no additional payload flags.
  LOOM_AMDGPU_VOPD_PAIR_FLAG_NONE = 0u,
  // VOPD pair uses the shared 32-bit literal payload word.
  LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL = 1u << 0,
  // X component encodes its commutable sources in the opposite orientation.
  LOOM_AMDGPU_VOPD_PAIR_FLAG_X_SOURCES_SWAPPED = 1u << 1,
  // Y component encodes its commutable sources in the opposite orientation.
  LOOM_AMDGPU_VOPD_PAIR_FLAG_Y_SOURCES_SWAPPED = 1u << 2,
} loom_amdgpu_vopd_pair_flag_bits_t;
typedef uint32_t loom_amdgpu_vopd_pair_flags_t;

// One scheduled packet's membership in a planned VOPD pair.
typedef struct loom_amdgpu_vopd_packet_t {
  // Role this scheduled packet plays in a VOPD pair.
  loom_amdgpu_vopd_packet_role_t role;
  // VOPD pair index, or LOOM_AMDGPU_VOPD_PAIR_NONE.
  uint32_t pair_index;
} loom_amdgpu_vopd_packet_t;

// Component facts captured for a rejected VOPD component.
typedef struct loom_amdgpu_vopd_rejection_component_t {
  // VOPD operation id encoded in this component slot.
  uint16_t op;
  // Destination VGPR encoded in this component slot.
  uint16_t vdst;
  // First explicit source VGPR encoded in this component slot.
  uint16_t src0;
  // Second explicit source VGPR encoded in this component slot.
  uint16_t vsrc1;
  // Component-local payload and encoding flags.
  loom_amdgpu_vopd_pair_flags_t flags;
  // Component literal payload when LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL is set.
  uint32_t literal_u32;
} loom_amdgpu_vopd_rejection_component_t;

// Final native-emission facts for one component of a planned VOPD packet.
typedef struct loom_amdgpu_vopd_component_t {
  // VOPD operation id encoded in this component slot.
  uint16_t op;
  // Operand/register form selected by the source descriptor rule.
  loom_amdgpu_vopd_component_form_t form;
  // Destination VGPR encoded in this component slot.
  uint16_t vdst;
  // First explicit source VGPR, or zero when the component has no VGPR SRC0.
  uint16_t src0;
  // Second explicit source VGPR, or zero when the component has no VGPR VSRC1.
  uint16_t vsrc1;
  // Unified architectural selector encoded in the component SRC0 field.
  uint16_t src0_selector;
  // Component immediate payload, or zero when the form has no immediate.
  uint32_t immediate_u32;
} loom_amdgpu_vopd_component_t;

// One native VOPD packet replacing two schedule-visible component packets.
typedef struct loom_amdgpu_vopd_pair_t {
  // Why this VOPD pair was formed.
  loom_amdgpu_vopd_pair_reason_t reason;
  // Region block containing both component packets.
  uint32_t block_index;
  // Scheduled packet index for the X component.
  uint32_t first_packet_index;
  // Scheduled packet index for the Y component.
  uint32_t second_packet_index;
  // Schedule node index for the X component.
  uint32_t first_node_index;
  // Schedule node index for the Y component.
  uint32_t second_node_index;
  // Final X-slot component emission facts.
  loom_amdgpu_vopd_component_t x;
  // Final Y-slot component emission facts.
  loom_amdgpu_vopd_component_t y;
  // Pair-local payload and encoding flags.
  loom_amdgpu_vopd_pair_flags_t flags;
  // Shared literal payload when LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL is set.
  uint32_t literal_u32;
} loom_amdgpu_vopd_pair_t;

// One adjacent packet pair that looked like a VOPD opportunity but was
// rejected.
typedef struct loom_amdgpu_vopd_rejection_t {
  // Why this adjacent packet pair could not form a VOPD pair.
  loom_amdgpu_vopd_rejection_reason_t reason;
  // Failed register constraints when reason is REGISTER_CONSTRAINTS.
  loom_amdgpu_vopd_register_constraint_flags_t register_constraint_flags;
  // Region block containing both component packets.
  uint32_t block_index;
  // Scheduled packet index for the first visible component.
  uint32_t first_packet_index;
  // Scheduled packet index for the second visible component.
  uint32_t second_packet_index;
  // Schedule node index for the first visible component.
  uint32_t first_node_index;
  // Schedule node index for the second visible component.
  uint32_t second_node_index;
  // Decoded first-component facts.
  loom_amdgpu_vopd_rejection_component_t first;
  // Decoded second-component facts.
  loom_amdgpu_vopd_rejection_component_t second;
} loom_amdgpu_vopd_rejection_t;

// AMDGPU VOPD packetization table for one scheduled and allocated low function.
typedef struct loom_amdgpu_vopd_plan_t {
  // Schedule table this plan was built from.
  const loom_low_schedule_table_t* schedule;
  // Allocation table this plan was built from.
  const loom_low_allocation_table_t* allocation;
  // VOPD pairs in scheduled order.
  const loom_amdgpu_vopd_pair_t* pairs;
  // Number of VOPD pair records.
  iree_host_size_t pair_count;
  // Rejected adjacent VOPD candidates in scheduled order.
  const loom_amdgpu_vopd_rejection_t* rejections;
  // Number of VOPD rejection records.
  iree_host_size_t rejection_count;
  // Per-scheduled-packet VOPD membership records.
  const loom_amdgpu_vopd_packet_t* packets;
  // Number of packet membership records.
  iree_host_size_t packet_count;
} loom_amdgpu_vopd_plan_t;

struct loom_amdgpu_matrix_coexecution_t;
struct loom_amdgpu_processor_properties_t;

// Returns the stable spelling for a VOPD packet role.
iree_string_view_t loom_amdgpu_vopd_packet_role_name(
    loom_amdgpu_vopd_packet_role_t role);

// Returns the stable spelling for a VOPD pair reason.
iree_string_view_t loom_amdgpu_vopd_pair_reason_name(
    loom_amdgpu_vopd_pair_reason_t reason);

// Returns the stable spelling for a VOPD rejection reason.
iree_string_view_t loom_amdgpu_vopd_rejection_reason_name(
    loom_amdgpu_vopd_rejection_reason_t reason);

// Builds conservative AMDGPU VOPD pairings from a scheduled and allocated low
// function. Optional address-state and wait-packet plans suppress pairs that
// would consume an insertion point before the second component. Fixed
// wait-state planning runs over the resulting native packet stream. The caller
// must keep |schedule| and |allocation| immutable and |arena| alive for as long
// as |out_plan| is used.
iree_status_t loom_amdgpu_vopd_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const struct loom_amdgpu_processor_properties_t* processor_properties,
    const loom_amdgpu_address_state_plan_t* address_state,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    struct loom_amdgpu_matrix_coexecution_t* matrix_coexecution,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* transient_arena,
    loom_amdgpu_vopd_plan_t* out_plan);

// Builds AMDGPU scheduling affinities for descriptors that can later form VOPD
// pairs on |target|. These are scheduling hints only;
// loom_amdgpu_vopd_plan_build remains the final post-allocation legality
// check.
iree_status_t loom_amdgpu_vopd_build_schedule_pair_affinities(
    const loom_low_resolved_target_t* target, iree_arena_allocator_t* arena,
    loom_low_schedule_pair_affinity_list_t* out_affinities);

// Returns the VOPD membership record for |packet_index|, or NULL.
const loom_amdgpu_vopd_packet_t* loom_amdgpu_vopd_plan_packet_at(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t packet_index);

// Appends a compact JSON representation of |plan| to |builder|.
iree_status_t loom_amdgpu_vopd_plan_format_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_PLAN_H_
