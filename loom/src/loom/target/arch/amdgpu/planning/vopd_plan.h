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
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/planning/wait_states.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for scheduled packets that do not belong to a VOPD pair.
#define LOOM_AMDGPU_VOPD_PAIR_NONE UINT32_MAX

// Component opcode for v_fmac_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_FMAC_F32 UINT16_C(0)
// Component opcode for v_fmaak_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_FMAAK_F32 UINT16_C(1)
// Component opcode for v_fmamk_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_FMAMK_F32 UINT16_C(2)
// Component opcode for v_mul_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_MUL_F32 UINT16_C(3)
// Component opcode for v_add_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_ADD_F32 UINT16_C(4)
// Component opcode for v_sub_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_SUB_F32 UINT16_C(5)
// Component opcode for v_subrev_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_SUBREV_F32 UINT16_C(6)
// Component opcode for v_mov_b32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_MOV_B32 UINT16_C(8)
// Component opcode for v_max_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_MAX_F32 UINT16_C(10)
// Component opcode for v_min_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_MIN_F32 UINT16_C(11)
// Component opcode for v_dot2acc_f32_f16 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_DOT2_F32_F16 UINT16_C(12)
// Component opcode for v_dot2acc_f32_bf16 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_DOT2_F32_BF16 UINT16_C(13)
// Component opcode for v_add_nc_u32 in a VOPD Y slot.
#define LOOM_AMDGPU_VOPD_OP_ADD_U32 UINT16_C(16)
// Component opcode for v_lshlrev_b32 in a VOPD Y slot.
#define LOOM_AMDGPU_VOPD_OP_LSHLREV_B32 UINT16_C(17)
// Component opcode for v_and_b32 in a VOPD Y slot.
#define LOOM_AMDGPU_VOPD_OP_AND_B32 UINT16_C(18)
// Component opcode for v_sub_nc_u32 in a VOPD Y slot.
#define LOOM_AMDGPU_VOPD_OP_SUB_U32 UINT16_C(20)
// Component opcode for v_lshrrev_b32 in a VOPD Y slot.
#define LOOM_AMDGPU_VOPD_OP_LSHRREV_B32 UINT16_C(21)
// Component opcode for v_ashrrev_i32 in a VOPD Y slot.
#define LOOM_AMDGPU_VOPD_OP_ASHRREV_I32 UINT16_C(22)
// Component opcode for v_max_i32 in a VOPD Y slot.
#define LOOM_AMDGPU_VOPD_OP_MAX_I32 UINT16_C(23)
// Component opcode for v_min_i32 in a VOPD Y slot.
#define LOOM_AMDGPU_VOPD_OP_MIN_I32 UINT16_C(24)

typedef enum loom_amdgpu_vopd_packet_role_e {
  // Packet is not part of a VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE = 0,
  // Packet is the X component and emission point for a VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST = 1,
  // Packet is the Y component consumed by the previous VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND = 2,
} loom_amdgpu_vopd_packet_role_t;

typedef enum loom_amdgpu_vopd_pair_reason_e {
  // Unknown or uninitialized VOPD pair reason.
  LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN = 0,
  // Two independent v_fmac_f32 packets were fused into v_dual_fmac_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAC_F32 = 1,
  // Two independent v_fmaak_f32 packets were fused into v_dual_fmaak_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAAK_F32 = 2,
  // Two independent v_fmamk_f32 packets were fused into v_dual_fmamk_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAMK_F32 = 3,
  // Two independent v_mul_f32 packets were fused into v_dual_mul_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_MUL_F32 = 4,
  // Two independent v_add_f32 packets were fused into v_dual_add_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_ADD_F32 = 5,
  // Two independent v_sub_f32 packets were fused into v_dual_sub_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_SUB_F32 = 6,
  // Two independent inline-source v_mov_b32 packets were fused into
  // v_dual_mov_b32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_MOV_B32 = 7,
  // Two independent v_max_f32 packets were fused into v_dual_max_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_MAX_F32 = 8,
  // Two independent v_min_f32 packets were fused into v_dual_min_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_MIN_F32 = 9,
  // Two independent v_subrev_f32 packets were fused into v_dual_subrev_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_SUBREV_F32 = 10,
  // Two different component opcodes were fused into one legal VOPD packet.
  LOOM_AMDGPU_VOPD_PAIR_REASON_MIXED_COMPONENTS = 11,
  // Two independent v_dot2_f32_f16 packets were fused into
  // v_dual_dot2acc_f32_f16.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_DOT2_F32_F16 = 12,
  // Two independent v_dot2_f32_bf16 packets were fused into
  // v_dual_dot2acc_f32_bf16.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_DOT2_F32_BF16 = 13,
} loom_amdgpu_vopd_pair_reason_t;

typedef enum loom_amdgpu_vopd_component_form_e {
  // Component form whose result is tied to one accumulator operand.
  LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE = 0,
  // Two-source FMA component with a shared K literal in the last asm operand.
  LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAAK_LITERAL = 1,
  // Two-source FMA component with a shared K literal in the middle asm operand.
  LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAMK_LITERAL = 2,
  // Ordinary two-VGPR-source VALU component form.
  LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR = 3,
  // Inline-source move component form.
  LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV = 4,
} loom_amdgpu_vopd_component_form_t;

typedef enum loom_amdgpu_vopd_component_source_bits_e {
  // Component has no register source operands.
  LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_NONE = 0u,
  // Component source 0 is a VGPR and participates in VOPD constraints.
  LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0 = 1u << 0,
  // Component source 1 is a VGPR and participates in VOPD constraints.
  LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1 = 1u << 1,
  // Component has both VOPD source operands modeled as VGPRs.
  LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY =
      LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0 |
      LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1,
} loom_amdgpu_vopd_component_source_bits_t;
typedef uint8_t loom_amdgpu_vopd_component_source_mask_t;

typedef enum loom_amdgpu_vopd_component_lane_bits_e {
  // Component may not occupy either VOPD lane.
  LOOM_AMDGPU_VOPD_COMPONENT_LANE_NONE = 0u,
  // Component may occupy the X lane.
  LOOM_AMDGPU_VOPD_COMPONENT_LANE_X = 1u << 0,
  // Component may occupy the Y lane.
  LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y = 1u << 1,
  // Component may occupy either VOPD lane.
  LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY =
      LOOM_AMDGPU_VOPD_COMPONENT_LANE_X | LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
} loom_amdgpu_vopd_component_lane_bits_t;
typedef uint8_t loom_amdgpu_vopd_component_lane_mask_t;

typedef enum loom_amdgpu_vopd_component_pair_bits_e {
  // Component cannot form a VOPD pair.
  LOOM_AMDGPU_VOPD_COMPONENT_PAIR_NONE = 0u,
  // Component may pair with the same VOPD opcode.
  LOOM_AMDGPU_VOPD_COMPONENT_PAIR_SAME_OPCODE = 1u << 0,
  // Component may pair with a different VOPD opcode.
  LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE = 1u << 1,
  // Component may pair with any lane-compatible VOPD opcode.
  LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY =
      LOOM_AMDGPU_VOPD_COMPONENT_PAIR_SAME_OPCODE |
      LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
} loom_amdgpu_vopd_component_pair_bits_t;
typedef uint8_t loom_amdgpu_vopd_component_pair_mask_t;

// Operand layout for VOPD component forms whose sources are row-defined.
typedef struct loom_amdgpu_vopd_component_operand_layout_t {
  // Operand index of the accumulator tied to the result register.
  uint8_t accumulator_index;
  // Operand index of the source encoded in the VOPD SRC0 field.
  uint8_t src0_index;
  // Operand index of the source encoded in the VOPD VSRC1 field.
  uint8_t vsrc1_index;
} loom_amdgpu_vopd_component_operand_layout_t;

// Descriptor-independent facts for one native VOPD component opcode.
typedef struct loom_amdgpu_vopd_component_info_t {
  // VOPD operation id encoded in this component slot.
  uint16_t op;
  // Same-op pair reason used when two adjacent components match this opcode.
  loom_amdgpu_vopd_pair_reason_t same_op_reason;
  // Stable JSON/report spelling for |op|.
  iree_string_view_t op_name;
  // Stable JSON/report spelling for |same_op_reason|.
  iree_string_view_t same_op_reason_name;
  // Native assembly mnemonic for this component inside a VOPD packet.
  iree_string_view_t assembly_mnemonic;
  // RDNA4 native assembly mnemonic override, or empty to use assembly_mnemonic.
  iree_string_view_t rdna4_assembly_mnemonic;
  // Operand/register form shared by planning, assembly, and encoding.
  loom_amdgpu_vopd_component_form_t form;
  // Operand indexes interpreted by forms with row-defined source layout.
  loom_amdgpu_vopd_component_operand_layout_t operands;
  // VOPD lanes this component opcode may occupy.
  loom_amdgpu_vopd_component_lane_mask_t lane_mask;
  // Pairing modes this component opcode may participate in.
  loom_amdgpu_vopd_component_pair_mask_t pairing_mask;
  // Source operand slots that contain real VGPRs.
  loom_amdgpu_vopd_component_source_mask_t source_register_mask;
} loom_amdgpu_vopd_component_info_t;

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
} loom_amdgpu_vopd_rejection_reason_t;

typedef enum loom_amdgpu_vopd_pair_flag_bits_e {
  // VOPD pair has no additional payload flags.
  LOOM_AMDGPU_VOPD_PAIR_FLAG_NONE = 0u,
  // VOPD pair uses the shared 32-bit literal payload word.
  LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL = 1u << 0,
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
  // VOPD operation id encoded in the X slot.
  uint16_t op_x;
  // VOPD operation id encoded in the Y slot.
  uint16_t op_y;
  // Destination VGPR encoded in the X slot.
  uint16_t x_vdst;
  // First explicit source VGPR encoded in the X slot.
  uint16_t x_src0;
  // Second explicit source VGPR encoded in the X slot.
  uint16_t x_vsrc1;
  // Destination VGPR encoded in the Y slot.
  uint16_t y_vdst;
  // First explicit source VGPR encoded in the Y slot.
  uint16_t y_src0;
  // Second explicit source VGPR encoded in the Y slot.
  uint16_t y_vsrc1;
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

// Returns the stable spelling for a VOPD packet role.
iree_string_view_t loom_amdgpu_vopd_packet_role_name(
    loom_amdgpu_vopd_packet_role_t role);

// Returns the stable spelling for a VOPD pair reason.
iree_string_view_t loom_amdgpu_vopd_pair_reason_name(
    loom_amdgpu_vopd_pair_reason_t reason);

// Returns the stable spelling for a VOPD rejection reason.
iree_string_view_t loom_amdgpu_vopd_rejection_reason_name(
    loom_amdgpu_vopd_rejection_reason_t reason);

// Returns descriptor-independent facts for the native VOPD component opcode.
const loom_amdgpu_vopd_component_info_t* loom_amdgpu_vopd_component_info_for_op(
    uint16_t op);

// Builds conservative AMDGPU VOPD pairings from a scheduled and allocated low
// function. Optional wait packet/state plans suppress pairs that would consume
// an insertion point before the second component. The caller must keep
// |schedule|, |allocation|, and |arena| immutable/alive for as long as
// |out_plan| is used.
iree_status_t loom_amdgpu_vopd_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_state_plan_t* wait_states,
    iree_arena_allocator_t* arena, loom_amdgpu_vopd_plan_t* out_plan);

// Builds AMDGPU scheduling affinities for descriptors that can later form VOPD
// pairs. These are scheduling hints only; loom_amdgpu_vopd_plan_build remains
// the final post-allocation legality check.
iree_status_t loom_amdgpu_vopd_build_schedule_pair_affinities(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_pair_affinity_list_t* out_affinities);

// Verifies that |plan| describes |schedule| and |allocation|.
iree_status_t loom_amdgpu_vopd_plan_verify(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_vopd_plan_t* plan);

// Verifies that wait insertions do not target the second component of any VOPD
// pair. Emission cannot preserve such an insertion without breaking the dual
// packet.
iree_status_t loom_amdgpu_vopd_plan_verify_wait_insertions(
    const loom_amdgpu_vopd_plan_t* plan,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_state_plan_t* wait_states);

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
