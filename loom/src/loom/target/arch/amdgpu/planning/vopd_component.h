// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU VOPD component vocabulary and physical pair legality.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_COMPONENT_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_COMPONENT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

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
// Component opcode for v_cndmask_b32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_CNDMASK_B32 UINT16_C(9)
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
  // Two independent v_mov_b32 packets were fused into v_dual_mov_b32.
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
  // Two independent v_cndmask_b32 packets were fused into
  // v_dual_cndmask_b32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_CNDMASK_B32 = 14,
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
  // Register-source move component form.
  LOOM_AMDGPU_VOPD_COMPONENT_FORM_REGISTER_MOV = 5,
  // Two-source conditional select with an unencoded VCC predicate operand.
  LOOM_AMDGPU_VOPD_COMPONENT_FORM_CNDMASK_VCC = 6,
} loom_amdgpu_vopd_component_form_t;

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
  // Numeric-minimum/maximum mnemonic, or empty to use assembly_mnemonic.
  iree_string_view_t numeric_minmax_mnemonic;
  // VOPD lanes this component opcode may occupy.
  loom_amdgpu_vopd_component_lane_mask_t lane_mask;
  // Pairing modes this component opcode may participate in.
  loom_amdgpu_vopd_component_pair_mask_t pairing_mask;
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
  // VOPD compression would invalidate the active TRANS-result wait proof.
  LOOM_AMDGPU_VOPD_REJECTION_REASON_TRANS_RESULT_WINDOW = 6,
} loom_amdgpu_vopd_rejection_reason_t;

typedef enum loom_amdgpu_vopd_register_constraint_flag_bits_e {
  // Component destination VGPRs have the same parity.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_DESTINATION_PARITY = 1u << 0,
  // Component SRC0 VGPRs occupy the same register bank.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC0_BANK = 1u << 1,
  // Component SRC1 cache inputs occupy the same register bank.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_VSRC1_BANK = 1u << 2,
  // X destination aliases the Y SRC0 VGPR.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_SRC0 = 1u << 3,
  // X destination aliases the Y VSRC1 VGPR.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_X_DESTINATION_Y_VSRC1 = 1u << 4,
  // Y destination aliases the X SRC0 VGPR.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_SRC0 = 1u << 5,
  // Y destination aliases the X VSRC1 VGPR.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_Y_DESTINATION_X_VSRC1 = 1u << 6,
  // Component SRC2 cache inputs have the same register parity.
  LOOM_AMDGPU_VOPD_REGISTER_CONSTRAINT_FLAG_SRC2_BANK = 1u << 7,
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

struct loom_amdgpu_vopd_component_rule_t;

typedef struct loom_amdgpu_vopd_candidate_component_t {
  // Descriptor-local rule that decoded this component.
  const struct loom_amdgpu_vopd_component_rule_t* rule;
  // Destination VGPR.
  uint16_t vdst;
  // First explicit source VGPR before unified-source encoding bias.
  uint16_t src0;
  // Second explicit source VGPR.
  uint16_t vsrc1;
  // Unified source selector for forms without a VGPR SRC0.
  uint16_t src0_selector;
  // Component-local payload flags.
  loom_amdgpu_vopd_pair_flags_t flags;
  // Component literal payload when LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL is set.
  uint32_t literal_u32;
} loom_amdgpu_vopd_candidate_component_t;

typedef struct loom_amdgpu_vopd_candidate_pair_t {
  // Why this VOPD pair can be formed.
  loom_amdgpu_vopd_pair_reason_t reason;
  // X-slot component.
  loom_amdgpu_vopd_candidate_component_t x;
  // Y-slot component.
  loom_amdgpu_vopd_candidate_component_t y;
  // Pair-local payload and encoding flags.
  loom_amdgpu_vopd_pair_flags_t flags;
  // Shared literal payload when LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL is set.
  uint32_t literal_u32;
} loom_amdgpu_vopd_candidate_pair_t;

// Returns descriptor-independent facts for the native VOPD component opcode.
const loom_amdgpu_vopd_component_info_t* loom_amdgpu_vopd_component_info_for_op(
    uint16_t op);

// Resolves the legal X/Y opcode combination, independent of physical operands.
bool loom_amdgpu_vopd_component_infos_pair_reason(
    const loom_amdgpu_vopd_component_info_t* first_info,
    const loom_amdgpu_vopd_component_info_t* second_info,
    loom_amdgpu_vopd_pair_reason_t* out_reason);

// Resolves a shared literal and legal source orientation for decoded
// components. Opcode compatibility and independence are established by the
// caller. Success updates the pair and returns UNKNOWN. Register rejection
// retains the original orientation and reports its failed constraints; literal
// rejection reports none.
loom_amdgpu_vopd_rejection_reason_t loom_amdgpu_vopd_candidate_pair_resolve(
    loom_amdgpu_vopd_candidate_pair_t* candidate,
    loom_amdgpu_vopd_register_constraint_flags_t* out_constraint_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_COMPONENT_H_
