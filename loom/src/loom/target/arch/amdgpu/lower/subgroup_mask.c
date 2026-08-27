// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static bool loom_amdgpu_subgroup_mask_bit_count(const loom_module_t* module,
                                                loom_value_id_t value_id,
                                                uint32_t* out_bit_count) {
  *out_bit_count = 0;
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (loom_amdgpu_type_is_i32(type)) {
    *out_bit_count = 32;
    return true;
  }
  if (loom_amdgpu_type_is_i64(type)) {
    *out_bit_count = 64;
    return true;
  }
  return false;
}

static bool loom_amdgpu_subgroup_mask_covers_wavefront(
    uint32_t mask_bit_count, uint32_t wavefront_size) {
  return mask_bit_count >= wavefront_size;
}

static bool loom_amdgpu_subgroup_mask_requires_wave32_zero_extend(
    uint32_t mask_bit_count, uint32_t wavefront_size) {
  return mask_bit_count == 64 && wavefront_size == 32;
}

typedef enum loom_amdgpu_subgroup_vote_kind_e {
  LOOM_AMDGPU_SUBGROUP_VOTE_ANY = 0,
  LOOM_AMDGPU_SUBGROUP_VOTE_ALL = 1,
  LOOM_AMDGPU_SUBGROUP_VOTE_COUNT_,
} loom_amdgpu_subgroup_vote_kind_t;

typedef uint8_t loom_amdgpu_subgroup_vote_rule_flags_t;

enum loom_amdgpu_subgroup_vote_rule_flag_bits_e {
  LOOM_AMDGPU_SUBGROUP_VOTE_RULE_WAVE32_REQUIRES_PEER_MASK = 1u << 0,
};

typedef struct loom_amdgpu_subgroup_vote_rule_t {
  // Width-specific descriptor requirements.
  loom_amdgpu_subgroup_vote_rule_flags_t flags;
  // Descriptor row selected for the predicate comparison.
  loom_amdgpu_descriptor_ref_t compare_descriptor_ref;
  // Descriptor row selected for wave32 predicate comparisons.
  loom_amdgpu_descriptor_ref_t wave32_compare_descriptor_ref;
  // Descriptor row selected for the mask compared against the predicate.
  loom_amdgpu_descriptor_ref_t peer_mask_descriptor_ref;
  // Constraint key used when subgroup width selection fails.
  iree_string_view_t wavefront_constraint_key;
  // Constraint key used when the predicate is not a native mask.
  iree_string_view_t native_predicate_constraint_key;
  // Constraint key used when the compare descriptor is unavailable.
  iree_string_view_t compare_descriptor_constraint_key;
  // Constraint key used when the wave32 compare descriptor is unavailable.
  iree_string_view_t wave32_compare_descriptor_constraint_key;
  // Constraint key used when the peer-mask descriptor is unavailable.
  iree_string_view_t peer_mask_descriptor_constraint_key;
} loom_amdgpu_subgroup_vote_rule_t;

static const loom_amdgpu_subgroup_vote_rule_t
    kLoomAmdgpuSubgroupVoteRules[LOOM_AMDGPU_SUBGROUP_VOTE_COUNT_] = {
        [LOOM_AMDGPU_SUBGROUP_VOTE_ANY] =
            {
                .compare_descriptor_ref =
                    LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
                .wave32_compare_descriptor_ref =
                    LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32_SRC1_INLINE,
                .peer_mask_descriptor_ref =
                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
                .wavefront_constraint_key =
                    IREE_SVL("subgroup_vote_any.wavefront_size"),
                .native_predicate_constraint_key =
                    IREE_SVL("subgroup_vote_any.native_predicate"),
                .compare_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_cmp_lg_u64"),
                .wave32_compare_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_cmp_lg_i32_src1_inline"),
                .peer_mask_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_mov_b32"),
            },
        [LOOM_AMDGPU_SUBGROUP_VOTE_ALL] =
            {
                .flags =
                    LOOM_AMDGPU_SUBGROUP_VOTE_RULE_WAVE32_REQUIRES_PEER_MASK,
                .compare_descriptor_ref =
                    LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_U64,
                .wave32_compare_descriptor_ref =
                    LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32,
                .peer_mask_descriptor_ref =
                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
                .wavefront_constraint_key =
                    IREE_SVL("subgroup_vote_all.wavefront_size"),
                .native_predicate_constraint_key =
                    IREE_SVL("subgroup_vote_all.native_predicate"),
                .compare_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_cmp_eq_u64"),
                .wave32_compare_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_cmp_eq_i32"),
                .peer_mask_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_mov_b64_exec_read"),
            },
};

static iree_status_t loom_amdgpu_select_subgroup_vote_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_subgroup_vote_rule_t* rule, loom_value_id_t predicate,
    loom_low_lower_resolved_descriptor_t* out_compare_descriptor,
    loom_low_lower_resolved_descriptor_t* out_peer_mask_descriptor,
    uint32_t* out_wavefront_size, bool* out_selected) {
  *out_compare_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_peer_mask_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_wavefront_size = 0;
  *out_selected = false;

  if (!loom_amdgpu_select_subgroup_wavefront_size(context,
                                                  out_wavefront_size)) {
    return iree_ok_status();
  }

  bool predicate_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_is_native_i1_mask(
      context, predicate, &predicate_is_native_mask));
  if (!predicate_is_native_mask) {
    return iree_ok_status();
  }

  const bool is_wave32 = *out_wavefront_size == 32;
  const bool peer_mask_required =
      !is_wave32 ||
      iree_all_bits_set(
          rule->flags,
          LOOM_AMDGPU_SUBGROUP_VOTE_RULE_WAVE32_REQUIRES_PEER_MASK);
  loom_amdgpu_descriptor_resolution_t resolutions[2] = {
      {.descriptor_ref = is_wave32 ? rule->wave32_compare_descriptor_ref
                                   : rule->compare_descriptor_ref,
       .out_descriptor = out_compare_descriptor},
  };
  iree_host_size_t resolution_count = 1;
  if (peer_mask_required) {
    resolutions[resolution_count++] = (loom_amdgpu_descriptor_resolution_t){
        .descriptor_ref = rule->peer_mask_descriptor_ref,
        .out_descriptor = out_peer_mask_descriptor,
    };
  }
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, resolution_count, &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_select_subgroup_mask_result(
    loom_low_lower_context_t* context, loom_value_id_t mask,
    uint32_t* out_mask_bit_count, uint32_t* out_wavefront_size) {
  *out_mask_bit_count = 0;
  *out_wavefront_size = 0;

  if (!loom_amdgpu_select_subgroup_wavefront_size(context,
                                                  out_wavefront_size)) {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_subgroup_mask_bit_count(module, mask, out_mask_bit_count) ||
      !loom_amdgpu_subgroup_mask_covers_wavefront(*out_mask_bit_count,
                                                  *out_wavefront_size)) {
    return false;
  }
  if (loom_amdgpu_subgroup_mask_requires_wave32_zero_extend(
          *out_mask_bit_count, *out_wavefront_size) &&
      !loom_amdgpu_descriptor_set_has_ref(
          loom_low_lower_context_descriptor_set(context),
          LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32)) {
    return false;
  }

  return true;
}

iree_status_t loom_amdgpu_select_kernel_subgroup_active_mask_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_active_mask_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_active_mask_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_active_mask_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t mask = loom_kernel_subgroup_active_mask_mask(source_op);
  uint32_t mask_bit_count = 0;
  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_subgroup_mask_result(context, mask, &mask_bit_count,
                                               &wavefront_size)) {
    return iree_ok_status();
  }

  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      &out_plan->exec_read_descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  out_plan->mask = mask;
  out_plan->mask_bit_count = mask_bit_count;
  out_plan->wavefront_size = wavefront_size;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_ballot_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_ballot_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_ballot_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_vote_ballot_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t predicate =
      loom_kernel_subgroup_vote_ballot_predicate(source_op);
  bool predicate_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_is_native_i1_mask(
      context, predicate, &predicate_is_native_mask));
  if (!predicate_is_native_mask) {
    return iree_ok_status();
  }

  const loom_value_id_t mask = loom_kernel_subgroup_vote_ballot_mask(source_op);
  uint32_t mask_bit_count = 0;
  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_subgroup_mask_result(context, mask, &mask_bit_count,
                                               &wavefront_size)) {
    return iree_ok_status();
  }

  out_plan->predicate = predicate;
  out_plan->mask = mask;
  out_plan->mask_bit_count = mask_bit_count;
  out_plan->wavefront_size = wavefront_size;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_vote_any_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_vote_any_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_vote_any_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_vote_any_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t predicate =
      loom_kernel_subgroup_vote_any_predicate(source_op);
  uint32_t wavefront_size = 0;
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_vote_plan(
      context, &kLoomAmdgpuSubgroupVoteRules[LOOM_AMDGPU_SUBGROUP_VOTE_ANY],
      predicate, &out_plan->compare_descriptor, &out_plan->zero_descriptor,
      &wavefront_size, &selected));
  if (!selected) {
    return iree_ok_status();
  }

  out_plan->predicate = predicate;
  out_plan->result = loom_kernel_subgroup_vote_any_result(source_op);
  out_plan->wavefront_size = wavefront_size;
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_kernel_subgroup_vote_all_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_vote_all_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_subgroup_vote_all_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_subgroup_vote_all_isa(source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t predicate =
      loom_kernel_subgroup_vote_all_predicate(source_op);
  uint32_t wavefront_size = 0;
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_vote_plan(
      context, &kLoomAmdgpuSubgroupVoteRules[LOOM_AMDGPU_SUBGROUP_VOTE_ALL],
      predicate, &out_plan->compare_descriptor, &out_plan->exec_read_descriptor,
      &wavefront_size, &selected));
  if (!selected) {
    return iree_ok_status();
  }

  out_plan->predicate = predicate;
  out_plan->result = loom_kernel_subgroup_vote_all_result(source_op);
  out_plan->wavefront_size = wavefront_size;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_subgroup_exec_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, /*unit_count=*/2, &mask_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(NULL, 0), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bind_subgroup_lane_mask_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_mask, uint32_t mask_bit_count,
    uint32_t wavefront_size, loom_value_id_t low_mask) {
  if (mask_bit_count == 64) {
    if (loom_amdgpu_subgroup_mask_requires_wave32_zero_extend(mask_bit_count,
                                                              wavefront_size)) {
      loom_type_t sgpr_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

      loom_value_id_t low_half = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_mask, /*offset=*/0, sgpr_type, &low_half));
      loom_value_id_t zero_high = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
          sgpr_type, &zero_high));

      loom_type_t source_mask_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_range_type(
          context, /*unit_count=*/2, &source_mask_type));
      loom_value_id_t zero_extended_halves[] = {low_half, zero_high};
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
          context, source_op, zero_extended_halves,
          IREE_ARRAYSIZE(zero_extended_halves), source_mask_type, &low_result));
      return loom_low_lower_bind_value(context, source_mask, low_result);
    }
    return loom_low_lower_bind_value(context, source_mask, low_mask);
  }
  IREE_ASSERT_EQ(mask_bit_count, 32);

  loom_type_t low_mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &low_mask_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_mask, /*offset=*/0, low_mask_type, &low_result));
  return loom_low_lower_bind_value(context, source_mask, low_result);
}

static iree_status_t loom_amdgpu_emit_subgroup_zero_lane_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("imm32"), &imm32_attr_name_id));

  loom_value_id_t zero_halves[2] = {0};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(zero_halves); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
        context, source_op, descriptor, imm32_attr_name_id, 0, sgpr_type,
        &zero_halves[i]));
  }

  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, /*unit_count=*/2, &mask_type));
  return loom_amdgpu_build_low_register_range(context, source_op, zero_halves,
                                              IREE_ARRAYSIZE(zero_halves),
                                              mask_type, out_mask);
}

static iree_status_t loom_amdgpu_emit_subgroup_resolved_mask_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &result_type));
  const loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_active_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_active_mask_plan_t* plan) {
  loom_value_id_t low_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_exec_mask(
      context, source_op, &plan->exec_read_descriptor, &low_mask));
  return loom_amdgpu_bind_subgroup_lane_mask_result(
      context, source_op, plan->mask, plan->mask_bit_count,
      plan->wavefront_size, low_mask);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_ballot(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_ballot_plan_t* plan) {
  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->predicate, &low_predicate));
  return loom_amdgpu_bind_subgroup_lane_mask_result(
      context, source_op, plan->mask, plan->mask_bit_count,
      plan->wavefront_size, low_predicate);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_any(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_any_plan_t* plan) {
  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->predicate, &low_predicate));
  if (plan->wavefront_size == 32) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_nonzero_scc(
        context, source_op, low_predicate, plan->wavefront_size, &low_result));
    return loom_low_lower_bind_value(context, plan->result, low_result);
  }

  loom_value_id_t zero_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_zero_lane_mask(
      context, source_op, &plan->zero_descriptor, &zero_mask));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_resolved_mask_compare(
      context, source_op, &plan->compare_descriptor, low_predicate, zero_mask,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_all(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_all_plan_t* plan) {
  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->predicate, &low_predicate));
  loom_value_id_t exec_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_exec_mask(
      context, source_op, &plan->exec_read_descriptor, &exec_mask));
  if (plan->wavefront_size == 32) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_lane_mask_equal_scc(
        context, source_op, low_predicate, exec_mask, plan->wavefront_size,
        &low_result));
    return loom_low_lower_bind_value(context, plan->result, low_result);
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_resolved_mask_compare(
      context, source_op, &plan->compare_descriptor, low_predicate, exec_mask,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}
static iree_status_t loom_amdgpu_low_legality_verify_subgroup_native_predicate(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t predicate, iree_string_view_t constraint_key) {
  bool predicate_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_is_native_i1_mask(
      context, predicate, &predicate_is_native_mask));
  if (!predicate_is_native_mask) {
    return loom_amdgpu_low_legality_reject(context, op, constraint_key);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_mask_result(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t mask, uint32_t wavefront_size,
    uint32_t* out_mask_bit_count) {
  *out_mask_bit_count = 0;
  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_subgroup_mask_bit_count(module, mask, out_mask_bit_count)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_mask.result_type"));
  }
  if (!loom_amdgpu_subgroup_mask_covers_wavefront(*out_mask_bit_count,
                                                  wavefront_size)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_mask.wavefront_coverage"));
  }
  if (!loom_amdgpu_source_value_is_uniform_subgroup_lane_mask(
          module, loom_target_low_legality_fact_table(context), mask)) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("subgroup_mask.uniform_lane_mask"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_vote(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_amdgpu_subgroup_vote_rule_t* rule, loom_value_id_t predicate) {
  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, rule->wavefront_constraint_key, &unused_wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_native_predicate(
          context, op, predicate, rule->native_predicate_constraint_key));
  const bool is_wave32 = unused_wavefront_size == 32;
  if (is_wave32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, rule->wave32_compare_descriptor_ref,
        rule->wave32_compare_descriptor_constraint_key));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, rule->compare_descriptor_ref,
        rule->compare_descriptor_constraint_key));
  }
  if (is_wave32 &&
      !iree_all_bits_set(
          rule->flags,
          LOOM_AMDGPU_SUBGROUP_VOTE_RULE_WAVE32_REQUIRES_PEER_MASK)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, rule->peer_mask_descriptor_ref,
      rule->peer_mask_descriptor_constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_active_mask(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_active_mask.wavefront_size"),
      &wavefront_size));
  uint32_t unused_mask_bit_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_mask_result(
      context, op, loom_kernel_subgroup_active_mask_mask(op), wavefront_size,
      &unused_mask_bit_count));
  if (loom_amdgpu_subgroup_mask_requires_wave32_zero_extend(
          unused_mask_bit_count, wavefront_size)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        IREE_SV("descriptor.s_mov_b32")));
  }
  return loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      IREE_SV("descriptor.s_mov_b64_exec_read"));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_ballot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_ballot.wavefront_size"), &wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_native_predicate(
          context, op, loom_kernel_subgroup_vote_ballot_predicate(op),
          IREE_SV("subgroup_ballot.native_predicate")));
  uint32_t unused_mask_bit_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_mask_result(
      context, op, loom_kernel_subgroup_vote_ballot_mask(op), wavefront_size,
      &unused_mask_bit_count));
  if (loom_amdgpu_subgroup_mask_requires_wave32_zero_extend(
          unused_mask_bit_count, wavefront_size)) {
    return loom_amdgpu_low_legality_verify_descriptor_requirement(
        context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        IREE_SV("descriptor.s_mov_b32"));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_vote_any(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  return loom_amdgpu_low_legality_verify_subgroup_vote(
      context, op, &kLoomAmdgpuSubgroupVoteRules[LOOM_AMDGPU_SUBGROUP_VOTE_ANY],
      loom_kernel_subgroup_vote_any_predicate(op));
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_vote_all(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  return loom_amdgpu_low_legality_verify_subgroup_vote(
      context, op, &kLoomAmdgpuSubgroupVoteRules[LOOM_AMDGPU_SUBGROUP_VOTE_ALL],
      loom_kernel_subgroup_vote_all_predicate(op));
}
