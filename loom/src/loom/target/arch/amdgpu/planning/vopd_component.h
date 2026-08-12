// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-independent AMDGPU VOPD component vocabulary.

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

// Returns descriptor-independent facts for the native VOPD component opcode.
const loom_amdgpu_vopd_component_info_t* loom_amdgpu_vopd_component_info_for_op(
    uint16_t op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_VOPD_COMPONENT_H_
