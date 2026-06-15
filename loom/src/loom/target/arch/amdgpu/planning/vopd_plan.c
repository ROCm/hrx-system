// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/vopd_plan.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

typedef struct loom_amdgpu_vopd_candidate_component_t {
  // Descriptor-independent metadata row for this component.
  const loom_amdgpu_vopd_component_info_t* info;
  // VOPD operation id encoded in this component slot.
  uint16_t op;
  // Destination VGPR.
  uint16_t vdst;
  // First explicit source VGPR before unified-source encoding bias.
  uint16_t src0;
  // Second explicit source VGPR.
  uint16_t vsrc1;
  // Source operand slots that contain real VGPRs.
  uint8_t source_register_mask;
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

typedef struct loom_amdgpu_vopd_pair_analysis_t {
  // Decoded first component facts.
  loom_amdgpu_vopd_candidate_component_t first_component;
  // True if the first packet decoded as a supported VOPD component form.
  bool first_eligible;
  // Decoded second component facts.
  loom_amdgpu_vopd_candidate_component_t second_component;
  // True if the second packet decoded as a supported VOPD component form.
  bool second_eligible;
  // Candidate pair facts populated when both components can be paired.
  loom_amdgpu_vopd_candidate_pair_t candidate;
  // True if the adjacent packets satisfy VOPD pair legality.
  bool matched;
  // Stable rejection reason when |matched| is false.
  loom_amdgpu_vopd_rejection_reason_t rejection_reason;
} loom_amdgpu_vopd_pair_analysis_t;

typedef struct loom_amdgpu_vopd_component_rule_t {
  // Descriptor reference for the low packet component.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Descriptor set ordinals where this component row is legal.
  uint32_t descriptor_set_mask;
  // Descriptor-independent VOPD component facts.
  loom_amdgpu_vopd_component_info_t info;
} loom_amdgpu_vopd_component_rule_t;

typedef struct loom_amdgpu_vopd_plan_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying physical register assignments.
  const loom_low_allocation_table_t* allocation;
  // Optional planned wait packets that block second-component fusion.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Optional planned wait states that block second-component fusion.
  const loom_amdgpu_wait_state_plan_t* wait_states;
  // Arena owning all output and scratch arrays.
  iree_arena_allocator_t* arena;
  // Scheduled packets with wait insertions before them.
  bool* insertion_blocked_packets;
  // Output VOPD pair records.
  loom_amdgpu_vopd_pair_t* pairs;
  // Number of populated VOPD pair records.
  iree_host_size_t pair_count;
  // Allocated VOPD pair capacity.
  iree_host_size_t pair_capacity;
  // Output rejected adjacent VOPD candidates.
  loom_amdgpu_vopd_rejection_t* rejections;
  // Number of populated VOPD rejection records.
  iree_host_size_t rejection_count;
  // Allocated VOPD rejection capacity.
  iree_host_size_t rejection_capacity;
  // Output per-packet membership records.
  loom_amdgpu_vopd_packet_t* packets;
} loom_amdgpu_vopd_plan_builder_t;

#define LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_BIT(ordinal) (UINT32_C(1) << (ordinal))
#define LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA3 \
  LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_BIT(LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3)
#define LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4 \
  LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_BIT(LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4)
#define LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4_GFX125X \
  LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_BIT(                     \
      LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X)
#define LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD \
  (LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA3 |        \
   LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4 |        \
   LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4_GFX125X)
#define LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_GFX11_GFX12 \
  (LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA3 |          \
   LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4)
#define LOOM_AMDGPU_VOPD_OPERAND_LAYOUT_TIED(accumulator, src0, vsrc1) \
  {                                                                    \
      .accumulator_index = (accumulator),                              \
      .src0_index = (src0),                                            \
      .vsrc1_index = (vsrc1),                                          \
  }

static const loom_amdgpu_vopd_component_rule_t kVopdComponentRules[] = {
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAC_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_FMAC_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAC_F32,
                .op_name = IREE_SVL("fmac_f32"),
                .same_op_reason_name = IREE_SVL("dual_fmac_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_fmac_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE,
                .operands = LOOM_AMDGPU_VOPD_OPERAND_LAYOUT_TIED(0, 1, 2),
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAAK_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_FMAAK_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAAK_F32,
                .op_name = IREE_SVL("fmaak_f32"),
                .same_op_reason_name = IREE_SVL("dual_fmaak_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_fmaak_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAAK_LITERAL,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAMK_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_FMAMK_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAMK_F32,
                .op_name = IREE_SVL("fmamk_f32"),
                .same_op_reason_name = IREE_SVL("dual_fmamk_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_fmamk_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAMK_LITERAL,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_MUL_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_MUL_F32,
                .op_name = IREE_SVL("mul_f32"),
                .same_op_reason_name = IREE_SVL("dual_mul_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_mul_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_ADD_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_ADD_F32,
                .op_name = IREE_SVL("add_f32"),
                .same_op_reason_name = IREE_SVL("dual_add_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_add_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_SUB_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_SUB_F32,
                .op_name = IREE_SVL("sub_f32"),
                .same_op_reason_name = IREE_SVL("dual_sub_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_sub_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SUBREV_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_SUBREV_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_SUBREV_F32,
                .op_name = IREE_SVL("subrev_f32"),
                .same_op_reason_name = IREE_SVL("dual_subrev_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_subrev_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_MOV_B32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_MOV_B32,
                .op_name = IREE_SVL("mov_b32"),
                .same_op_reason_name = IREE_SVL("dual_mov_b32"),
                .assembly_mnemonic = IREE_SVL("v_dual_mov_b32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask = LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_NONE,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_MAX_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_MAX_F32,
                .op_name = IREE_SVL("max_f32"),
                .same_op_reason_name = IREE_SVL("dual_max_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_max_f32"),
                .rdna4_assembly_mnemonic = IREE_SVL("v_dual_max_num_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_F32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_MIN_F32,
                .same_op_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_MIN_F32,
                .op_name = IREE_SVL("min_f32"),
                .same_op_reason_name = IREE_SVL("dual_min_f32"),
                .assembly_mnemonic = IREE_SVL("v_dual_min_f32"),
                .rdna4_assembly_mnemonic = IREE_SVL("v_dual_min_num_f32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_DOT2_F32_F16,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_DOT2_F32_F16,
                .same_op_reason =
                    LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_DOT2_F32_F16,
                .op_name = IREE_SVL("dot2_f32_f16"),
                .same_op_reason_name = IREE_SVL("dual_dot2_f32_f16"),
                .assembly_mnemonic = IREE_SVL("v_dual_dot2acc_f32_f16"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE,
                .operands = LOOM_AMDGPU_VOPD_OPERAND_LAYOUT_TIED(2, 0, 1),
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_DOT2_F32_BF16,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_DOT2_F32_BF16,
                .same_op_reason =
                    LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_DOT2_F32_BF16,
                .op_name = IREE_SVL("dot2_f32_bf16"),
                .same_op_reason_name = IREE_SVL("dual_dot2_f32_bf16"),
                .assembly_mnemonic = IREE_SVL("v_dual_dot2acc_f32_bf16"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE,
                .operands = LOOM_AMDGPU_VOPD_OPERAND_LAYOUT_TIED(2, 0, 1),
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_ADD_U32,
                .op_name = IREE_SVL("add_u32"),
                .assembly_mnemonic = IREE_SVL("v_dual_add_nc_u32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA_VOPD,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_LSHLREV_B32,
                .op_name = IREE_SVL("lshlrev_b32"),
                .assembly_mnemonic = IREE_SVL("v_dual_lshlrev_b32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32,
        .descriptor_set_mask = LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_GFX11_GFX12,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_AND_B32,
                .op_name = IREE_SVL("and_b32"),
                .assembly_mnemonic = IREE_SVL("v_dual_and_b32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MAX_I32,
        .descriptor_set_mask =
            LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4_GFX125X,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_MAX_I32,
                .op_name = IREE_SVL("max_i32"),
                .assembly_mnemonic = IREE_SVL("v_dual_max_i32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_I32,
        .descriptor_set_mask =
            LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4_GFX125X,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_MIN_I32,
                .op_name = IREE_SVL("min_i32"),
                .assembly_mnemonic = IREE_SVL("v_dual_min_i32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_U32,
        .descriptor_set_mask =
            LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4_GFX125X,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_SUB_U32,
                .op_name = IREE_SVL("sub_u32"),
                .assembly_mnemonic = IREE_SVL("v_dual_sub_nc_u32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32,
        .descriptor_set_mask =
            LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4_GFX125X,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_LSHRREV_B32,
                .op_name = IREE_SVL("lshrrev_b32"),
                .assembly_mnemonic = IREE_SVL("v_dual_lshrrev_b32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
    {
        .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32,
        .descriptor_set_mask =
            LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_MASK_RDNA4_GFX125X,
        .info =
            {
                .op = LOOM_AMDGPU_VOPD_OP_ASHRREV_I32,
                .op_name = IREE_SVL("ashrrev_i32"),
                .assembly_mnemonic = IREE_SVL("v_dual_ashrrev_i32"),
                .form = LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR,
                .lane_mask = LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y,
                .pairing_mask = LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE,
                .source_register_mask =
                    LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY,
            },
    },
};

const loom_amdgpu_vopd_component_info_t* loom_amdgpu_vopd_component_info_for_op(
    uint16_t op) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVopdComponentRules); ++i) {
    if (kVopdComponentRules[i].info.op == op) {
      return &kVopdComponentRules[i].info;
    }
  }
  return NULL;
}

static const loom_amdgpu_vopd_component_info_t*
loom_amdgpu_vopd_component_info_for_reason(
    loom_amdgpu_vopd_pair_reason_t reason) {
  if (reason == LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVopdComponentRules); ++i) {
    if (kVopdComponentRules[i].info.same_op_reason == reason) {
      return &kVopdComponentRules[i].info;
    }
  }
  return NULL;
}

iree_string_view_t loom_amdgpu_vopd_packet_role_name(
    loom_amdgpu_vopd_packet_role_t role) {
  switch (role) {
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST:
      return IREE_SV("first");
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND:
      return IREE_SV("second");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_amdgpu_vopd_pair_reason_name(
    loom_amdgpu_vopd_pair_reason_t reason) {
  if (reason == LOOM_AMDGPU_VOPD_PAIR_REASON_MIXED_COMPONENTS) {
    return IREE_SV("mixed_components");
  }
  const loom_amdgpu_vopd_component_info_t* info =
      loom_amdgpu_vopd_component_info_for_reason(reason);
  return info != NULL ? info->same_op_reason_name : IREE_SV("unknown");
}

iree_string_view_t loom_amdgpu_vopd_rejection_reason_name(
    loom_amdgpu_vopd_rejection_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_COMPONENT_OPCODE_MISMATCH:
      return IREE_SV("component_opcode_mismatch");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_FIRST_RESULT_USED_BY_SECOND:
      return IREE_SV("first_result_used_by_second");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_LITERAL_MISMATCH:
      return IREE_SV("literal_mismatch");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS:
      return IREE_SV("register_constraints");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_SECOND_PACKET_HAS_INSERTION:
      return IREE_SV("second_packet_has_insertion");
    case LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_amdgpu_vopd_op_name(uint16_t op) {
  const loom_amdgpu_vopd_component_info_t* info =
      loom_amdgpu_vopd_component_info_for_op(op);
  return info != NULL ? info->op_name : IREE_SV("unknown");
}

const loom_amdgpu_vopd_packet_t* loom_amdgpu_vopd_plan_packet_at(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t packet_index) {
  if (plan == NULL || packet_index >= plan->packet_count) {
    return NULL;
  }
  const loom_amdgpu_vopd_packet_t* packet = &plan->packets[packet_index];
  return packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE ? NULL : packet;
}

static bool loom_amdgpu_vopd_target_supports_base_vopd(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL ||
      descriptor_set->target_stable_id != LOOM_AMDGPU_TARGET_STABLE_ID) {
    return false;
  }
  switch (descriptor_set->descriptor_set_ordinal) {
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X:
      return true;
    default:
      return false;
  }
}

static void loom_amdgpu_vopd_append_schedule_pair_affinity(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t first_descriptor_ref,
    loom_amdgpu_descriptor_ref_t second_descriptor_ref, uint16_t priority,
    loom_low_schedule_pair_affinity_t* affinities,
    iree_host_size_t* affinity_count) {
  const loom_low_descriptor_t* first_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                            first_descriptor_ref);
  const loom_low_descriptor_t* second_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                            second_descriptor_ref);
  if (first_descriptor == NULL || second_descriptor == NULL) {
    return;
  }
  affinities[(*affinity_count)++] = (loom_low_schedule_pair_affinity_t){
      .first_descriptor = first_descriptor,
      .second_descriptor = second_descriptor,
      .priority = priority,
  };
}

static bool loom_amdgpu_vopd_component_can_use_lane(
    const loom_amdgpu_vopd_component_info_t* info,
    loom_amdgpu_vopd_component_lane_mask_t lane) {
  return info != NULL && iree_all_bits_set(info->lane_mask, lane);
}

static bool loom_amdgpu_vopd_component_can_pair(
    const loom_amdgpu_vopd_component_info_t* info,
    loom_amdgpu_vopd_component_pair_mask_t pairing_mode) {
  return info != NULL && iree_all_bits_set(info->pairing_mask, pairing_mode);
}

static bool loom_amdgpu_vopd_component_infos_pair_reason(
    const loom_amdgpu_vopd_component_info_t* first_info,
    const loom_amdgpu_vopd_component_info_t* second_info,
    loom_amdgpu_vopd_pair_reason_t* out_reason) {
  *out_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN;
  if (!loom_amdgpu_vopd_component_can_use_lane(
          first_info, LOOM_AMDGPU_VOPD_COMPONENT_LANE_X) ||
      !loom_amdgpu_vopd_component_can_use_lane(
          second_info, LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y)) {
    return false;
  }
  if (first_info->op == second_info->op) {
    if (!loom_amdgpu_vopd_component_can_pair(
            first_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_SAME_OPCODE)) {
      return false;
    }
    *out_reason = first_info->same_op_reason;
    return *out_reason != LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN;
  }
  if (!loom_amdgpu_vopd_component_can_pair(
          first_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE) ||
      !loom_amdgpu_vopd_component_can_pair(
          second_info, LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE)) {
    return false;
  }
  *out_reason = LOOM_AMDGPU_VOPD_PAIR_REASON_MIXED_COMPONENTS;
  return true;
}

static bool loom_amdgpu_vopd_component_rule_applies_to_descriptor_set(
    const loom_amdgpu_vopd_component_rule_t* rule,
    const loom_low_descriptor_set_t* descriptor_set) {
  if (rule == NULL || descriptor_set == NULL ||
      descriptor_set->descriptor_set_ordinal >=
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT) {
    return false;
  }
  return iree_any_bit_set(rule->descriptor_set_mask,
                          LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_BIT(
                              descriptor_set->descriptor_set_ordinal));
}

iree_status_t loom_amdgpu_vopd_build_schedule_pair_affinities(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_pair_affinity_list_t* out_affinities) {
  *out_affinities = loom_low_schedule_pair_affinity_list_empty();
  if (descriptor_set == NULL || arena == NULL ||
      !loom_amdgpu_vopd_target_supports_base_vopd(descriptor_set)) {
    return iree_ok_status();
  }

  const iree_host_size_t max_affinity_count =
      IREE_ARRAYSIZE(kVopdComponentRules) * IREE_ARRAYSIZE(kVopdComponentRules);
  loom_low_schedule_pair_affinity_t* affinities = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, max_affinity_count, sizeof(*affinities), (void**)&affinities));
  iree_host_size_t affinity_count = 0;
  for (iree_host_size_t first_index = 0;
       first_index < IREE_ARRAYSIZE(kVopdComponentRules); ++first_index) {
    const loom_amdgpu_vopd_component_rule_t* first_rule =
        &kVopdComponentRules[first_index];
    if (!loom_amdgpu_vopd_component_rule_applies_to_descriptor_set(
            first_rule, descriptor_set)) {
      continue;
    }
    for (iree_host_size_t second_index = 0;
         second_index < IREE_ARRAYSIZE(kVopdComponentRules); ++second_index) {
      const loom_amdgpu_vopd_component_rule_t* second_rule =
          &kVopdComponentRules[second_index];
      if (!loom_amdgpu_vopd_component_rule_applies_to_descriptor_set(
              second_rule, descriptor_set)) {
        continue;
      }
      loom_amdgpu_vopd_pair_reason_t reason =
          LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN;
      if (!loom_amdgpu_vopd_component_infos_pair_reason(
              &first_rule->info, &second_rule->info, &reason)) {
        continue;
      }
      const uint16_t priority =
          first_rule->descriptor_ref == second_rule->descriptor_ref ? 2 : 1;
      loom_amdgpu_vopd_append_schedule_pair_affinity(
          descriptor_set, first_rule->descriptor_ref,
          second_rule->descriptor_ref, priority, affinities, &affinity_count);
    }
  }
  *out_affinities = (loom_low_schedule_pair_affinity_list_t){
      .values = affinities,
      .count = affinity_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_verify_wait_packet_plan(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  if (wait_packets == NULL) {
    return iree_ok_status();
  }
  if (wait_packets->wait_plan == NULL ||
      wait_packets->wait_plan->schedule != schedule) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD wait packets must be derived from "
                            "the planned schedule");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_verify_wait_state_plan(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_state_plan_t* wait_states) {
  if (wait_states != NULL && (wait_states->schedule != schedule ||
                              wait_states->allocation != allocation)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD wait states must be derived from the "
                            "planned schedule and allocation");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_packet_index_for_insertion(
    const loom_low_schedule_table_t* schedule, uint32_t block_index,
    uint32_t node_index, uint32_t scheduled_ordinal,
    uint32_t* out_packet_index) {
  *out_packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  if (block_index >= schedule->block_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VOPD insertion block index %" PRIu32
                            " is out of range",
                            block_index);
  }
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  if (scheduled_ordinal >= block->scheduled_node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VOPD insertion scheduled ordinal %" PRIu32
                            " is out of range",
                            scheduled_ordinal);
  }
  const uint32_t packet_index = block->scheduled_node_start + scheduled_ordinal;
  uint32_t packet_node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  IREE_RETURN_IF_ERROR(loom_low_packet_node_index_at(schedule, packet_index,
                                                     &packet_node_index));
  if (packet_node_index != node_index) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD insertion node %" PRIu32
                            " does not match scheduled packet node %" PRIu32,
                            node_index, packet_node_index);
  }
  *out_packet_index = packet_index;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_mark_wait_packet_insertions(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  if (builder->wait_packets == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < builder->wait_packets->packet_count; ++i) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &builder->wait_packets->packets[i];
    uint32_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_packet_index_for_insertion(
        builder->schedule, wait_packet->block_index, wait_packet->node_index,
        wait_packet->scheduled_ordinal, &packet_index));
    builder->insertion_blocked_packets[packet_index] = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_mark_wait_state_insertions(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  if (builder->wait_states == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < builder->wait_states->state_count; ++i) {
    const loom_amdgpu_wait_state_t* wait_state =
        &builder->wait_states->states[i];
    uint32_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_packet_index_for_insertion(
        builder->schedule, wait_state->block_index, wait_state->node_index,
        wait_state->scheduled_ordinal, &packet_index));
    builder->insertion_blocked_packets[packet_index] = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_allocate(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  const iree_host_size_t packet_count = builder->schedule->scheduled_node_count;
  builder->pair_capacity = packet_count / 2;
  builder->rejection_capacity = packet_count == 0 ? 0 : packet_count - 1;
  if (packet_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, packet_count,
                                                   sizeof(*builder->packets),
                                                   (void**)&builder->packets));
    for (iree_host_size_t i = 0; i < packet_count; ++i) {
      builder->packets[i] = (loom_amdgpu_vopd_packet_t){
          .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE,
          .pair_index = LOOM_AMDGPU_VOPD_PAIR_NONE,
      };
    }
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->arena, packet_count,
                                  sizeof(*builder->insertion_blocked_packets),
                                  (void**)&builder->insertion_blocked_packets));
    memset(builder->insertion_blocked_packets, 0,
           packet_count * sizeof(*builder->insertion_blocked_packets));
  }
  if (builder->pair_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->pair_capacity, sizeof(*builder->pairs),
        (void**)&builder->pairs));
  }
  if (builder->rejection_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->rejection_capacity,
        sizeof(*builder->rejections), (void**)&builder->rejections));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_mark_wait_packet_insertions(builder));
  return loom_amdgpu_vopd_mark_wait_state_insertions(builder);
}

static const loom_low_allocation_assignment_t* loom_amdgpu_vopd_map_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
}

static bool loom_amdgpu_vopd_assignment_single_physical_vgpr(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t* out_register) {
  *out_register = 0;
  if (assignment == NULL ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      assignment->location_count != 1 || assignment->location_base > 255) {
    return false;
  }
  *out_register = (uint16_t)assignment->location_base;
  return true;
}

static bool loom_amdgpu_vopd_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs != NULL && rhs != NULL &&
         lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static bool loom_amdgpu_vopd_component_result_is_used_by(
    const loom_low_packet_view_t* producer,
    const loom_low_packet_view_t* consumer) {
  const loom_op_t* producer_op = producer->node->op;
  const loom_op_t* consumer_op = consumer->node->op;
  const loom_value_id_t* results = loom_op_const_results(producer_op);
  const loom_value_id_t* operands = loom_op_const_operands(consumer_op);
  for (iree_host_size_t result_index = 0;
       result_index < producer_op->result_count; ++result_index) {
    for (iree_host_size_t operand_index = 0;
         operand_index < consumer_op->operand_count; ++operand_index) {
      if (results[result_index] == operands[operand_index]) {
        return true;
      }
    }
  }
  return false;
}

static loom_named_attr_slice_t loom_amdgpu_vopd_packet_attrs(
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  if (loom_low_op_isa(op)) {
    return loom_low_op_attrs(op);
  }
  if (loom_low_const_isa(op)) {
    return loom_low_const_attrs(op);
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static const loom_named_attr_t* loom_amdgpu_vopd_find_packet_attr_by_name(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet, iree_string_view_t name) {
  const loom_module_t* module = builder->schedule->module;
  loom_named_attr_slice_t attrs = loom_amdgpu_vopd_packet_attrs(packet);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id < module->strings.count &&
        iree_string_view_equal(module->strings.entries[attr->name_id], name)) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_vopd_read_immediate_u32(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    uint32_t* out_value) {
  *out_value = 0;
  if (descriptor_immediate_index >= packet->descriptor->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VOPD descriptor immediate index %" PRIu16
                            " is out of range",
                            descriptor_immediate_index);
  }
  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + descriptor_immediate_index;
  if (immediate_row >= descriptor_set->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VOPD descriptor immediate row %" PRIu32
                            " is out of range",
                            immediate_row);
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_row];
  iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  const loom_named_attr_t* attr = loom_amdgpu_vopd_find_packet_attr_by_name(
      builder, packet, immediate_name);
  if (attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD immediate '%.*s' is required",
                            (int)immediate_name.size, immediate_name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD immediate '%.*s' must be an i64",
                            (int)immediate_name.size, immediate_name.data);
  }
  const int64_t value = attr->value.i64;
  if (value < 0 || value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU VOPD immediate '%.*s' value %" PRId64 " is not a u32",
        (int)immediate_name.size, immediate_name.data, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static bool loom_amdgpu_vopd_cross_component_destinations_are_independent(
    const loom_amdgpu_vopd_candidate_pair_t* candidate) {
  if (iree_any_bit_set(candidate->y.source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      candidate->x.vdst == candidate->y.src0) {
    return false;
  }
  if (iree_any_bit_set(candidate->y.source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1) &&
      candidate->x.vdst == candidate->y.vsrc1) {
    return false;
  }
  if (iree_any_bit_set(candidate->x.source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      candidate->y.vdst == candidate->x.src0) {
    return false;
  }
  if (iree_any_bit_set(candidate->x.source_register_mask,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1) &&
      candidate->y.vdst == candidate->x.vsrc1) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vopd_bank_compatible(uint16_t x_register,
                                             uint16_t y_register,
                                             uint16_t bank_mask) {
  return (x_register & bank_mask) != (y_register & bank_mask);
}

static bool loom_amdgpu_vopd_registers_satisfy_base_constraints(
    const loom_amdgpu_vopd_candidate_pair_t* candidate) {
  if (((candidate->x.vdst ^ candidate->y.vdst) & 1u) == 0) {
    return false;
  }
  const loom_amdgpu_vopd_component_source_mask_t paired_source_registers =
      candidate->x.source_register_mask & candidate->y.source_register_mask;
  if (iree_any_bit_set(paired_source_registers,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0) &&
      !loom_amdgpu_vopd_bank_compatible(candidate->x.src0, candidate->y.src0,
                                        3)) {
    return false;
  }
  if (iree_any_bit_set(paired_source_registers,
                       LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1) &&
      !loom_amdgpu_vopd_bank_compatible(candidate->x.vsrc1, candidate->y.vsrc1,
                                        3)) {
    return false;
  }
  return loom_amdgpu_vopd_cross_component_destinations_are_independent(
      candidate);
}

static bool loom_amdgpu_vopd_immediate_is_inline_u32(
    const loom_amdgpu_vopd_plan_builder_t* builder, uint32_t value) {
  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  const loom_amdgpu_encoding_table_t* encoding_table =
      loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
          descriptor_set->descriptor_set_ordinal);
  uint16_t source_selector = 0;
  return loom_amdgpu_encoding_inline_u32_source(encoding_table, value,
                                                &source_selector);
}

static iree_status_t loom_amdgpu_vopd_read_tied_accumulate_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_vopd_component_info_t* info,
    loom_amdgpu_vopd_candidate_component_t* out_component, bool* out_eligible) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};
  *out_eligible = false;

  const loom_op_t* op = packet->node->op;
  if (op->result_count != 1 || packet->descriptor->immediate_count != 0 ||
      info->operands.accumulator_index >= op->operand_count ||
      info->operands.src0_index >= op->operand_count ||
      info->operands.vsrc1_index >= op->operand_count) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  const loom_low_allocation_assignment_t* accumulator_assignment =
      loom_amdgpu_vopd_map_assignment(
          builder->allocation, operands[info->operands.accumulator_index]);
  const loom_low_allocation_assignment_t* src0_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation,
                                      operands[info->operands.src0_index]);
  const loom_low_allocation_assignment_t* vsrc1_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation,
                                      operands[info->operands.vsrc1_index]);
  if (!loom_amdgpu_vopd_assignments_match(result_assignment,
                                          accumulator_assignment)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(src0_assignment,
                                                        &out_component->src0)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(
          vsrc1_assignment, &out_component->vsrc1)) {
    return iree_ok_status();
  }
  *out_eligible = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_read_literal_fma_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component, bool* out_eligible) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){
      .flags = LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL,
  };
  *out_eligible = false;

  const loom_op_t* op = packet->node->op;
  if (op->result_count != 1 || op->operand_count != 2 ||
      packet->descriptor->immediate_count != 1) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  const loom_low_allocation_assignment_t* src0_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[0]);
  const loom_low_allocation_assignment_t* vsrc1_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[1]);
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(src0_assignment,
                                                        &out_component->src0)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(
          vsrc1_assignment, &out_component->vsrc1)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_read_immediate_u32(
      builder, packet, 0, &out_component->literal_u32));
  *out_eligible = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_read_binary_vgpr_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component, bool* out_eligible) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};
  *out_eligible = false;

  const loom_op_t* op = packet->node->op;
  if (op->result_count != 1 || op->operand_count != 2 ||
      packet->descriptor->immediate_count != 0) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  const loom_low_allocation_assignment_t* src0_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[0]);
  const loom_low_allocation_assignment_t* vsrc1_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[1]);
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(src0_assignment,
                                                        &out_component->src0)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(
          vsrc1_assignment, &out_component->vsrc1)) {
    return iree_ok_status();
  }
  *out_eligible = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_read_mov_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component, bool* out_eligible) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};
  *out_eligible = false;

  const loom_op_t* op = packet->node->op;
  if (op->result_count != 1 || op->operand_count != 0 ||
      packet->descriptor->immediate_count != 1) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        &out_component->vdst)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_read_immediate_u32(
      builder, packet, 0, &out_component->literal_u32));
  if (!loom_amdgpu_vopd_immediate_is_inline_u32(builder,
                                                out_component->literal_u32)) {
    return iree_ok_status();
  }
  *out_eligible = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_read_component(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_vopd_candidate_component_t* out_component, bool* out_eligible) {
  *out_component = (loom_amdgpu_vopd_candidate_component_t){0};
  *out_eligible = false;

  const loom_low_descriptor_set_t* descriptor_set =
      builder->schedule->target.descriptor_set;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kVopdComponentRules); ++i) {
    const loom_amdgpu_vopd_component_rule_t* rule = &kVopdComponentRules[i];
    if (!loom_amdgpu_vopd_component_rule_applies_to_descriptor_set(
            rule, descriptor_set)) {
      continue;
    }
    const loom_low_descriptor_t* descriptor =
        loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                              rule->descriptor_ref);
    if (descriptor == NULL || packet->descriptor != descriptor) {
      continue;
    }
    iree_status_t status = iree_ok_status();
    switch (rule->info.form) {
      case LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE:
        status = loom_amdgpu_vopd_read_tied_accumulate_component(
            builder, packet, &rule->info, out_component, out_eligible);
        break;
      case LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAAK_LITERAL:
      case LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAMK_LITERAL:
        status = loom_amdgpu_vopd_read_literal_fma_component(
            builder, packet, out_component, out_eligible);
        break;
      case LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR:
        status = loom_amdgpu_vopd_read_binary_vgpr_component(
            builder, packet, out_component, out_eligible);
        break;
      case LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV:
        status = loom_amdgpu_vopd_read_mov_component(
            builder, packet, out_component, out_eligible);
        break;
      default:
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "AMDGPU VOPD component rule has unknown form");
    }
    if (iree_status_is_ok(status) && *out_eligible) {
      out_component->info = &rule->info;
      out_component->op = rule->info.op;
      out_component->source_register_mask = rule->info.source_register_mask;
    }
    return status;
  }
  return iree_ok_status();
}

static bool loom_amdgpu_vopd_pair_reason_for_components(
    const loom_amdgpu_vopd_candidate_component_t* first,
    const loom_amdgpu_vopd_candidate_component_t* second,
    loom_amdgpu_vopd_pair_reason_t* out_reason) {
  return loom_amdgpu_vopd_component_infos_pair_reason(first->info, second->info,
                                                      out_reason);
}

static bool loom_amdgpu_vopd_resolve_pair_literal(
    const loom_amdgpu_vopd_candidate_component_t* first,
    const loom_amdgpu_vopd_candidate_component_t* second,
    loom_amdgpu_vopd_candidate_pair_t* candidate) {
  const bool first_has_literal =
      iree_any_bit_set(first->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL);
  const bool second_has_literal =
      iree_any_bit_set(second->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL);
  if (first_has_literal && second_has_literal &&
      first->literal_u32 != second->literal_u32) {
    return false;
  }
  candidate->flags = first->flags | second->flags;
  if (first_has_literal) {
    candidate->literal_u32 = first->literal_u32;
  } else if (second_has_literal) {
    candidate->literal_u32 = second->literal_u32;
  }
  return true;
}

static iree_status_t loom_amdgpu_vopd_analyze_pair(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second,
    loom_amdgpu_vopd_pair_analysis_t* out_analysis) {
  *out_analysis = (loom_amdgpu_vopd_pair_analysis_t){
      .rejection_reason = LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN,
  };

  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_read_component(
      builder, first, &out_analysis->first_component,
      &out_analysis->first_eligible));
  if (!out_analysis->first_eligible) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_read_component(
      builder, second, &out_analysis->second_component,
      &out_analysis->second_eligible));
  if (!out_analysis->second_eligible) {
    return iree_ok_status();
  }

  loom_amdgpu_vopd_candidate_pair_t candidate = {
      .x = out_analysis->first_component,
      .y = out_analysis->second_component,
  };
  out_analysis->rejection_reason =
      LOOM_AMDGPU_VOPD_REJECTION_REASON_COMPONENT_OPCODE_MISMATCH;
  if (!loom_amdgpu_vopd_pair_reason_for_components(
          &out_analysis->first_component, &out_analysis->second_component,
          &candidate.reason)) {
    return iree_ok_status();
  }
  out_analysis->rejection_reason =
      LOOM_AMDGPU_VOPD_REJECTION_REASON_FIRST_RESULT_USED_BY_SECOND;
  if (loom_amdgpu_vopd_component_result_is_used_by(first, second)) {
    return iree_ok_status();
  }
  out_analysis->rejection_reason =
      LOOM_AMDGPU_VOPD_REJECTION_REASON_LITERAL_MISMATCH;
  if (!loom_amdgpu_vopd_resolve_pair_literal(&out_analysis->first_component,
                                             &out_analysis->second_component,
                                             &candidate)) {
    return iree_ok_status();
  }
  out_analysis->rejection_reason =
      LOOM_AMDGPU_VOPD_REJECTION_REASON_REGISTER_CONSTRAINTS;
  if (!loom_amdgpu_vopd_registers_satisfy_base_constraints(&candidate)) {
    return iree_ok_status();
  }
  out_analysis->candidate = candidate;
  out_analysis->matched = true;
  out_analysis->rejection_reason = LOOM_AMDGPU_VOPD_REJECTION_REASON_UNKNOWN;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_append_pair(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second,
    const loom_amdgpu_vopd_candidate_pair_t* candidate) {
  if (builder->pair_count >= builder->pair_capacity ||
      builder->pair_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU VOPD plan exceeded precomputed pair capacity");
  }
  const uint32_t pair_index = (uint32_t)builder->pair_count;
  builder->pairs[builder->pair_count++] = (loom_amdgpu_vopd_pair_t){
      .reason = candidate->reason,
      .block_index = first->node->block_index,
      .first_packet_index = (uint32_t)first->packet_index,
      .second_packet_index = (uint32_t)second->packet_index,
      .first_node_index = first->node_index,
      .second_node_index = second->node_index,
      .op_x = candidate->x.op,
      .op_y = candidate->y.op,
      .x_vdst = candidate->x.vdst,
      .x_src0 = candidate->x.src0,
      .x_vsrc1 = candidate->x.vsrc1,
      .y_vdst = candidate->y.vdst,
      .y_src0 = candidate->y.src0,
      .y_vsrc1 = candidate->y.vsrc1,
      .flags = candidate->flags,
      .literal_u32 = candidate->literal_u32,
  };
  builder->packets[first->packet_index] = (loom_amdgpu_vopd_packet_t){
      .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST,
      .pair_index = pair_index,
  };
  builder->packets[second->packet_index] = (loom_amdgpu_vopd_packet_t){
      .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND,
      .pair_index = pair_index,
  };
  return iree_ok_status();
}

static loom_amdgpu_vopd_rejection_component_t
loom_amdgpu_vopd_rejection_component_from_candidate(
    const loom_amdgpu_vopd_candidate_component_t* component) {
  return (loom_amdgpu_vopd_rejection_component_t){
      .op = component->op,
      .vdst = component->vdst,
      .src0 = component->src0,
      .vsrc1 = component->vsrc1,
      .flags = component->flags,
      .literal_u32 = component->literal_u32,
  };
}

static iree_status_t loom_amdgpu_vopd_append_rejection(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second,
    const loom_amdgpu_vopd_pair_analysis_t* analysis,
    loom_amdgpu_vopd_rejection_reason_t reason) {
  if (!analysis->first_eligible || !analysis->second_eligible) {
    return iree_ok_status();
  }
  if (builder->rejection_count >= builder->rejection_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU VOPD plan exceeded precomputed rejection capacity");
  }
  builder->rejections[builder->rejection_count++] =
      (loom_amdgpu_vopd_rejection_t){
          .reason = reason,
          .block_index = first->node->block_index,
          .first_packet_index = (uint32_t)first->packet_index,
          .second_packet_index = (uint32_t)second->packet_index,
          .first_node_index = first->node_index,
          .second_node_index = second->node_index,
          .first = loom_amdgpu_vopd_rejection_component_from_candidate(
              &analysis->first_component),
          .second = loom_amdgpu_vopd_rejection_component_from_candidate(
              &analysis->second_component),
      };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_packet_is_transparent(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    iree_host_size_t packet_index, bool* out_transparent) {
  *out_transparent = false;
  if (builder->insertion_blocked_packets[packet_index]) {
    return iree_ok_status();
  }

  loom_low_packet_view_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
      builder->schedule, builder->allocation, packet_index, &packet));
  if (packet.descriptor != NULL) {
    return iree_ok_status();
  }

  iree_host_size_t move_count = 0;
  const loom_op_t* op = packet.node->op;
  switch (op->kind) {
    case LOOM_OP_LOW_SLICE: {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_slice_units(
          builder->allocation, op, &move_count));
      break;
    }
    case LOOM_OP_LOW_CONCAT: {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_concat_units(
          builder->allocation, op, &move_count));
      break;
    }
    default:
      return iree_ok_status();
  }
  *out_transparent = move_count == 0;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_find_visible_packet(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_schedule_block_t* block,
    iree_host_size_t search_packet_index, iree_host_size_t* out_packet_index,
    bool* out_found) {
  *out_packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  *out_found = false;

  const iree_host_size_t block_end =
      block->scheduled_node_start + block->scheduled_node_count;
  for (iree_host_size_t packet_index = search_packet_index;
       packet_index < block_end; ++packet_index) {
    bool transparent = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_packet_is_transparent(
        builder, packet_index, &transparent));
    if (transparent) {
      continue;
    }
    *out_packet_index = packet_index;
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_block(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_schedule_block_t* block) {
  const iree_host_size_t block_end =
      block->scheduled_node_start + block->scheduled_node_count;
  for (iree_host_size_t search_packet_index = block->scheduled_node_start;
       search_packet_index < block_end;) {
    iree_host_size_t first_packet_index = LOOM_LOW_PACKET_INDEX_NONE;
    bool found_first = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_find_visible_packet(
        builder, block, search_packet_index, &first_packet_index,
        &found_first));
    if (!found_first) {
      break;
    }

    iree_host_size_t second_packet_index = LOOM_LOW_PACKET_INDEX_NONE;
    bool found_second = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_find_visible_packet(
        builder, block, first_packet_index + 1, &second_packet_index,
        &found_second));
    if (!found_second) {
      break;
    }
    loom_low_packet_view_t first = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
        builder->schedule, builder->allocation, first_packet_index, &first));
    loom_low_packet_view_t second = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
        builder->schedule, builder->allocation, second_packet_index, &second));
    loom_amdgpu_vopd_pair_analysis_t analysis = {0};
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_analyze_pair(builder, &first, &second, &analysis));
    if (!analysis.matched) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_append_rejection(
          builder, &first, &second, &analysis, analysis.rejection_reason));
      search_packet_index = first_packet_index + 1;
      continue;
    }
    if (builder->insertion_blocked_packets[second_packet_index]) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_append_rejection(
          builder, &first, &second, &analysis,
          LOOM_AMDGPU_VOPD_REJECTION_REASON_SECOND_PACKET_HAS_INSERTION));
      search_packet_index = first_packet_index + 1;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_append_pair(builder, &first, &second,
                                                      &analysis.candidate));
    search_packet_index = second_packet_index + 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_build_pairs(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  loom_low_allocation_value_scratch_t scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(builder->allocation, &scratch));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < builder->schedule->block_count && iree_status_is_ok(status); ++i) {
    status =
        loom_amdgpu_vopd_plan_block(builder, &builder->schedule->blocks[i]);
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

iree_status_t loom_amdgpu_vopd_plan_verify(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_vopd_plan_t* plan) {
  if (plan == NULL) {
    return iree_ok_status();
  }
  if (plan->schedule != schedule || plan->allocation != allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD plan must be derived from the emitted "
                            "schedule and allocation");
  }
  if (plan->packet_count != 0 &&
      plan->packet_count != schedule->scheduled_node_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD plan has %" PRIhsz
                            " packet records for %" PRIhsz " scheduled packets",
                            plan->packet_count, schedule->scheduled_node_count);
  }
  if (plan->pair_count != 0 && (plan->pairs == NULL || plan->packets == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD plan pair records are incomplete");
  }
  if (plan->rejection_count != 0 && plan->rejections == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU VOPD plan rejection records are incomplete");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_verify_insertion_packet(
    const loom_amdgpu_vopd_plan_t* plan, uint32_t block_index,
    uint32_t node_index, uint32_t scheduled_ordinal,
    iree_string_view_t insertion_kind) {
  if (plan == NULL || plan->packet_count == 0) {
    return iree_ok_status();
  }
  uint32_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_packet_index_for_insertion(
      plan->schedule, block_index, node_index, scheduled_ordinal,
      &packet_index));
  const loom_amdgpu_vopd_packet_t* packet =
      loom_amdgpu_vopd_plan_packet_at(plan, packet_index);
  if (packet != NULL && packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU VOPD plan cannot preserve %.*s insertion before fused second "
        "packet %" PRIu32,
        (int)insertion_kind.size, insertion_kind.data, packet_index);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_vopd_plan_verify_wait_insertions(
    const loom_amdgpu_vopd_plan_t* plan,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_state_plan_t* wait_states) {
  if (plan == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vopd_verify_wait_packet_plan(plan->schedule, wait_packets));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_verify_wait_state_plan(
      plan->schedule, plan->allocation, wait_states));
  if (wait_packets != NULL) {
    for (iree_host_size_t i = 0; i < wait_packets->packet_count; ++i) {
      const loom_amdgpu_wait_packet_t* wait_packet = &wait_packets->packets[i];
      IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_verify_insertion_packet(
          plan, wait_packet->block_index, wait_packet->node_index,
          wait_packet->scheduled_ordinal, IREE_SV("wait-packet")));
    }
  }
  if (wait_states != NULL) {
    for (iree_host_size_t i = 0; i < wait_states->state_count; ++i) {
      const loom_amdgpu_wait_state_t* wait_state = &wait_states->states[i];
      IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_verify_insertion_packet(
          plan, wait_state->block_index, wait_state->node_index,
          wait_state->scheduled_ordinal, IREE_SV("wait-state")));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_write_packet_descriptor_json(
    const loom_amdgpu_vopd_plan_t* plan, uint32_t packet_index,
    loom_output_stream_t* stream) {
  if (packet_index >= plan->packet_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU VOPD packet index %" PRIu32 " is out of range", packet_index);
  }
  loom_low_packet_view_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_packet_view_at(plan->schedule, plan->allocation,
                                               packet_index, &packet));
  if (packet.descriptor == NULL) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(
          plan->schedule->target.descriptor_set, packet.descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU VOPD packet descriptor row does not belong to the selected "
        "descriptor set");
  }
  iree_string_view_t descriptor_key =
      loom_low_descriptor_set_string(plan->schedule->target.descriptor_set,
                                     packet.descriptor->key_string_offset);
  return loom_json_write_escaped_string(stream, descriptor_key);
}

static iree_status_t loom_amdgpu_vopd_plan_write_component_json(
    const loom_amdgpu_vopd_plan_t* plan, uint32_t packet_index,
    uint32_t node_index, uint16_t op, uint16_t vdst, uint16_t src0,
    uint16_t vsrc1, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"packet\":%" PRIu32 ",\"node\":%" PRIu32 ",\"descriptor\":",
      packet_index, node_index));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_packet_descriptor_json(
      plan, packet_index, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"op_id\":%" PRIu16 ",\"op\":", op));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, loom_amdgpu_vopd_op_name(op)));
  return loom_output_stream_write_format(
      stream,
      ",\"vdst\":%" PRIu16 ",\"src0\":%" PRIu16 ",\"vsrc1\":%" PRIu16 "}", vdst,
      src0, vsrc1);
}

static iree_status_t loom_amdgpu_vopd_plan_write_pair_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t pair_index,
    loom_output_stream_t* stream) {
  const loom_amdgpu_vopd_pair_t* pair = &plan->pairs[pair_index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"reason\":", pair_index));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_amdgpu_vopd_pair_reason_name(pair->reason)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"block\":%" PRIu32 ",\"flags\":%" PRIu32 ",\"literal_u32\":",
      pair->block_index, pair->flags));
  if (iree_any_bit_set(pair->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, "%" PRIu32, pair->literal_u32));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"x\":"));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_component_json(
      plan, pair->first_packet_index, pair->first_node_index, pair->op_x,
      pair->x_vdst, pair->x_src0, pair->x_vsrc1, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"y\":"));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_component_json(
      plan, pair->second_packet_index, pair->second_node_index, pair->op_y,
      pair->y_vdst, pair->y_src0, pair->y_vsrc1, stream));
  return loom_output_stream_write_char(stream, '}');
}

static iree_status_t loom_amdgpu_vopd_plan_write_rejection_component_json(
    const loom_amdgpu_vopd_plan_t* plan, uint32_t packet_index,
    uint32_t node_index,
    const loom_amdgpu_vopd_rejection_component_t* component,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"packet\":%" PRIu32 ",\"node\":%" PRIu32 ",\"descriptor\":",
      packet_index, node_index));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_packet_descriptor_json(
      plan, packet_index, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"op_id\":%" PRIu16 ",\"op\":", component->op));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_amdgpu_vopd_op_name(component->op)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"vdst\":%" PRIu16 ",\"src0\":%" PRIu16 ",\"vsrc1\":%" PRIu16
      ",\"flags\":%" PRIu32 ",\"literal_u32\":",
      component->vdst, component->src0, component->vsrc1, component->flags));
  if (iree_any_bit_set(component->flags, LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "%" PRIu32, component->literal_u32));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  }
  return loom_output_stream_write_char(stream, '}');
}

static iree_status_t loom_amdgpu_vopd_plan_write_rejection_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t rejection_index,
    loom_output_stream_t* stream) {
  const loom_amdgpu_vopd_rejection_t* rejection =
      &plan->rejections[rejection_index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"reason\":", rejection_index));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_amdgpu_vopd_rejection_reason_name(rejection->reason)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"block\":%" PRIu32 ",\"first\":", rejection->block_index));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_rejection_component_json(
      plan, rejection->first_packet_index, rejection->first_node_index,
      &rejection->first, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"second\":"));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_write_rejection_component_json(
      plan, rejection->second_packet_index, rejection->second_node_index,
      &rejection->second, stream));
  return loom_output_stream_write_char(stream, '}');
}

static iree_status_t loom_amdgpu_vopd_plan_write_packet_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t packet_index,
    loom_output_stream_t* stream) {
  const loom_amdgpu_vopd_packet_t* packet = &plan->packets[packet_index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"role\":", packet_index));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_amdgpu_vopd_packet_role_name(packet->role)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"pair\":"));
  if (packet->pair_index == LOOM_AMDGPU_VOPD_PAIR_NONE) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "null"));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, "%" PRIu32,
                                                         packet->pair_index));
  }
  return loom_output_stream_write_char(stream, '}');
}

iree_status_t loom_amdgpu_vopd_plan_format_json(
    const loom_amdgpu_vopd_plan_t* plan, iree_string_builder_t* builder) {
  if (plan == NULL || builder == NULL || plan->schedule == NULL ||
      plan->allocation == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD plan and builder are required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.amdgpu.vopd_plan.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_low_diagnostic_function_name(plan->schedule->module,
                                                 plan->schedule->function_op)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, plan->schedule->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, plan->schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ",\"pair_count\":%zu,\"rejection_count\":%zu,\"packet_count\":%zu"
      ",\"pairs\":[",
      plan->pair_count, plan->rejection_count, plan->packet_count));
  for (iree_host_size_t i = 0; i < plan->pair_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_plan_write_pair_json(plan, i, &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "],\"rejections\":["));
  for (iree_host_size_t i = 0; i < plan->rejection_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_plan_write_rejection_json(plan, i, &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "],\"packets\":["));
  for (iree_host_size_t i = 0; i < plan->packet_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_plan_write_packet_json(plan, i, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]}"));
  return iree_ok_status();
}

iree_status_t loom_amdgpu_vopd_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_state_plan_t* wait_states,
    iree_arena_allocator_t* arena, loom_amdgpu_vopd_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vopd_plan_t){0};
  if (schedule == NULL || allocation == NULL || arena == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "schedule, allocation, and arena are required for "
                            "AMDGPU VOPD planning");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vopd_verify_wait_packet_plan(schedule, wait_packets));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_verify_wait_state_plan(
      schedule, allocation, wait_states));
  if (!loom_amdgpu_vopd_target_supports_base_vopd(
          schedule->target.descriptor_set)) {
    *out_plan = (loom_amdgpu_vopd_plan_t){
        .schedule = schedule,
        .allocation = allocation,
    };
    return iree_ok_status();
  }

  loom_amdgpu_vopd_plan_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .wait_packets = wait_packets,
      .wait_states = wait_states,
      .arena = arena,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_allocate(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_build_pairs(&builder));
  *out_plan = (loom_amdgpu_vopd_plan_t){
      .schedule = schedule,
      .allocation = allocation,
      .pairs = builder.pairs,
      .pair_count = builder.pair_count,
      .rejections = builder.rejections,
      .rejection_count = builder.rejection_count,
      .packets = builder.packets,
      .packet_count = schedule->scheduled_node_count,
  };
  return iree_ok_status();
}
