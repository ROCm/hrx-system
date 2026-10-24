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
#include "loom/target/arch/amdgpu/lower/materializers.h"
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

static bool loom_amdgpu_subgroup_vote_predicate_is_uniform(
    const loom_value_fact_table_t* fact_table, loom_value_id_t predicate) {
  return loom_value_facts_is_subgroup_uniform(
      loom_value_fact_table_lookup(fact_table, predicate));
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
                .compare_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_cmp_eq_u64"),
                .wave32_compare_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_cmp_eq_i32"),
                .peer_mask_descriptor_constraint_key =
                    IREE_SVL("descriptor.s_mov_b64_exec_read"),
            },
};

static iree_status_t loom_amdgpu_select_subgroup_predicate_mask_descriptors(
    loom_low_lower_context_t* context,
    loom_amdgpu_subgroup_predicate_mask_descriptors_t* out_descriptors,
    bool* out_present) {
  const loom_amdgpu_descriptor_resolution_t resolutions[] = {
      {.descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
       .out_descriptor = &out_descriptors->exec_read},
      {.descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
       .out_descriptor = &out_descriptors->intersect},
  };
  return loom_amdgpu_resolve_descriptor_refs_if_present(
      context, resolutions, IREE_ARRAYSIZE(resolutions), out_present);
}

// Selects the canonical native-mask carrier or a use-local materialization.
// Lane-varying predicates already occupy an EXEC-width mask. Uniform and exact
// predicates use SCC/SGPR carriers and must be expanded at the collective so a
// stronger fact analysis cannot make otherwise valid source illegal.
static iree_status_t loom_amdgpu_select_subgroup_predicate_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t predicate, const void** out_plan, bool* out_selected) {
  *out_plan = NULL;
  *out_selected = false;

  bool predicate_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_is_native_i1_mask(
      context, predicate, &predicate_is_native_mask));
  if (!predicate_is_native_mask) {
    bool can_materialize = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_value_can_materialize_as_native_i1_mask(
        context, source_op, predicate, &can_materialize));
    if (!can_materialize) {
      return iree_ok_status();
    }

    const loom_amdgpu_descriptor_ref_t descriptor_refs[] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32_SRC1_INLINE,
        LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32,
    };
    if (!loom_amdgpu_descriptor_set_has_all_refs(
            loom_low_lower_context_descriptor_set(context), descriptor_refs,
            IREE_ARRAYSIZE(descriptor_refs))) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_prepare_native_i1_mask(context, predicate, out_plan));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_subgroup_vote_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_rule_t* rule, loom_value_id_t predicate,
    const void** out_predicate_plan,
    loom_low_lower_resolved_descriptor_t* out_compare_descriptor,
    loom_low_lower_resolved_descriptor_t* out_peer_mask_descriptor,
    uint32_t* out_wavefront_size, bool* out_selected) {
  *out_compare_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_peer_mask_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_predicate_plan = NULL;
  *out_wavefront_size = 0;
  *out_selected = false;

  if (!loom_amdgpu_select_subgroup_wavefront_size(context,
                                                  out_wavefront_size)) {
    return iree_ok_status();
  }

  bool predicate_selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_predicate_plan(
      context, source_op, predicate, out_predicate_plan, &predicate_selected));
  if (!predicate_selected) {
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
      context, mask, &out_plan->result_in_vgpr));
  *out_selected = true;
  return loom_low_lower_plan_value_type(
      context, mask,
      loom_low_register_type(
          loom_low_lower_context_descriptor_set(context)->stable_id,
          out_plan->result_in_vgpr ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
                                   : LOOM_AMDGPU_REG_CLASS_ID_SGPR,
          mask_bit_count / 32));
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
  bool predicate_selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_predicate_plan(
      context, source_op, predicate, &out_plan->predicate_plan,
      &predicate_selected));
  if (!predicate_selected) {
    return iree_ok_status();
  }

  const loom_value_id_t mask = loom_kernel_subgroup_vote_ballot_mask(source_op);
  uint32_t mask_bit_count = 0;
  uint32_t wavefront_size = 0;
  if (!loom_amdgpu_select_subgroup_mask_result(context, mask, &mask_bit_count,
                                               &wavefront_size)) {
    return iree_ok_status();
  }

  bool active_descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_predicate_mask_descriptors(
      context, &out_plan->active, &active_descriptors_present));
  if (!active_descriptors_present) {
    return iree_ok_status();
  }

  out_plan->predicate = predicate;
  out_plan->mask = mask;
  out_plan->mask_bit_count = mask_bit_count;
  out_plan->wavefront_size = wavefront_size;
  IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
      context, mask, &out_plan->result_in_vgpr));
  *out_selected = true;
  return loom_low_lower_plan_value_type(
      context, mask,
      loom_low_register_type(
          loom_low_lower_context_descriptor_set(context)->stable_id,
          out_plan->result_in_vgpr ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
                                   : LOOM_AMDGPU_REG_CLASS_ID_SGPR,
          mask_bit_count / 32));
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
  out_plan->predicate = predicate;
  out_plan->result = loom_kernel_subgroup_vote_any_result(source_op);
  if (loom_amdgpu_subgroup_vote_predicate_is_uniform(
          loom_low_lower_context_fact_table(context), predicate)) {
    out_plan->result_aliases_predicate = true;
    loom_low_lower_inherit_value_type(context, predicate, out_plan->result);
    *out_selected = true;
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_vote_plan(
      context, source_op,
      &kLoomAmdgpuSubgroupVoteRules[LOOM_AMDGPU_SUBGROUP_VOTE_ANY], predicate,
      &out_plan->predicate_plan, &out_plan->compare_descriptor,
      &out_plan->zero_descriptor, &wavefront_size, &selected));
  if (!selected) {
    return iree_ok_status();
  }

  bool active_descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_predicate_mask_descriptors(
      context, &out_plan->active, &active_descriptors_present));
  if (!active_descriptors_present) {
    return iree_ok_status();
  }

  out_plan->wavefront_size = wavefront_size;
  *out_selected = true;
  return loom_low_lower_plan_value_type(
      context, out_plan->result,
      loom_low_register_type(
          loom_low_lower_context_descriptor_set(context)->stable_id,
          LOOM_AMDGPU_REG_CLASS_ID_SCC, 1));
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
  out_plan->predicate = predicate;
  out_plan->result = loom_kernel_subgroup_vote_all_result(source_op);
  if (loom_amdgpu_subgroup_vote_predicate_is_uniform(
          loom_low_lower_context_fact_table(context), predicate)) {
    out_plan->result_aliases_predicate = true;
    loom_low_lower_inherit_value_type(context, predicate, out_plan->result);
    *out_selected = true;
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  bool selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_subgroup_vote_plan(
      context, source_op,
      &kLoomAmdgpuSubgroupVoteRules[LOOM_AMDGPU_SUBGROUP_VOTE_ALL], predicate,
      &out_plan->predicate_plan, &out_plan->compare_descriptor,
      &out_plan->active.exec_read, &wavefront_size, &selected));
  if (!selected) {
    return iree_ok_status();
  }

  bool intersect_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      &out_plan->active.intersect, &intersect_descriptor_present));
  if (!intersect_descriptor_present) {
    return iree_ok_status();
  }

  out_plan->wavefront_size = wavefront_size;
  *out_selected = true;
  return loom_low_lower_plan_value_type(
      context, out_plan->result,
      loom_low_register_type(
          loom_low_lower_context_descriptor_set(context)->stable_id,
          LOOM_AMDGPU_REG_CLASS_ID_SCC, 1));
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

// Native predicate masks may have been produced before divergence. Only bits
// active at the collective's use participate in its ballot or vote.
static iree_status_t loom_amdgpu_emit_subgroup_active_predicate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_predicate_mask_descriptors_t* descriptors,
    loom_value_id_t predicate, loom_value_id_t* out_predicate,
    loom_value_id_t* out_exec_mask) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_exec_mask(
      context, source_op, &descriptors->exec_read, out_exec_mask));
  const loom_type_t mask_type = loom_module_value_type(
      loom_low_lower_context_module(context), *out_exec_mask);
  const loom_value_id_t operands[] = {predicate, *out_exec_mask};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptors->intersect, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_predicate = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bind_subgroup_lane_mask_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_mask, uint32_t mask_bit_count,
    uint32_t wavefront_size, bool result_in_vgpr, loom_value_id_t low_mask) {
  if (mask_bit_count == 64 && wavefront_size == 64 && !result_in_vgpr) {
    return loom_low_lower_bind_value(context, source_mask, low_mask);
  }
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  const uint32_t register_count = mask_bit_count / 32;
  loom_value_id_t registers[2];
  for (uint32_t i = 0; i < register_count; ++i) {
    if (i == 1 && wavefront_size == 32) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
          sgpr_type, &registers[i]));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_mask, i, sgpr_type, &registers[i]));
    }
    if (result_in_vgpr) {
      // EXEC masks the copy so lanes that leave a loop retain their own
      // observation while later iterations update the remaining lanes.
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
          context, source_op, registers[i], &registers[i]));
    }
  }
  return loom_amdgpu_bind_low_register_range(context, source_op, source_mask,
                                             registers, register_count);
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
      plan->wavefront_size, plan->result_in_vgpr, low_mask);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_ballot(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_ballot_plan_t* plan) {
  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_native_i1_mask(
      context, source_op, plan->predicate, plan->predicate_plan,
      &low_predicate));
  loom_value_id_t exec_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_active_predicate(
      context, source_op, &plan->active, low_predicate, &low_predicate,
      &exec_mask));
  return loom_amdgpu_bind_subgroup_lane_mask_result(
      context, source_op, plan->mask, plan->mask_bit_count,
      plan->wavefront_size, plan->result_in_vgpr, low_predicate);
}

iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_any(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_any_plan_t* plan) {
  if (plan->result_aliases_predicate) {
    return loom_low_lower_bind_value_alias(context, plan->predicate,
                                           plan->result);
  }

  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_native_i1_mask(
      context, source_op, plan->predicate, plan->predicate_plan,
      &low_predicate));
  loom_value_id_t exec_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_active_predicate(
      context, source_op, &plan->active, low_predicate, &low_predicate,
      &exec_mask));
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
  if (plan->result_aliases_predicate) {
    return loom_low_lower_bind_value_alias(context, plan->predicate,
                                           plan->result);
  }

  loom_value_id_t low_predicate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_native_i1_mask(
      context, source_op, plan->predicate, plan->predicate_plan,
      &low_predicate));
  loom_value_id_t exec_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_active_predicate(
      context, source_op, &plan->active, low_predicate, &low_predicate,
      &exec_mask));
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
static iree_status_t
loom_amdgpu_low_legality_verify_subgroup_predicate_materialization(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t predicate) {
  bool predicate_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_is_native_i1_mask(
      context, predicate, &predicate_is_native_mask));
  if (predicate_is_native_mask) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      IREE_SV("descriptor.s_mov_b32")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32_SRC1_INLINE,
      IREE_SV("descriptor.s_cmp_lg_i32_src1_inline")));
  return loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32,
      IREE_SV("descriptor.s_cselect_b32"));
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
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_predicate_mask(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      IREE_SV("descriptor.s_mov_b64_exec_read")));
  return loom_amdgpu_low_legality_verify_descriptor_requirement(
      context, op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      IREE_SV("descriptor.s_and_b64"));
}

static iree_status_t loom_amdgpu_low_legality_verify_subgroup_vote(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_amdgpu_subgroup_vote_rule_t* rule, loom_value_id_t predicate) {
  if (loom_amdgpu_subgroup_vote_predicate_is_uniform(
          loom_target_low_legality_fact_table(context), predicate)) {
    return iree_ok_status();
  }

  uint32_t unused_wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, rule->wavefront_constraint_key, &unused_wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_predicate_materialization(
          context, op, predicate));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_predicate_mask(context, op));
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
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
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
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  *out_handled = true;

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_subgroup_wavefront(
      context, op, IREE_SV("subgroup_ballot.wavefront_size"), &wavefront_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_predicate_materialization(
          context, op, loom_kernel_subgroup_vote_ballot_predicate(op)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_legality_verify_subgroup_predicate_mask(context, op));
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
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
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
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  *out_handled = true;

  return loom_amdgpu_low_legality_verify_subgroup_vote(
      context, op, &kLoomAmdgpuSubgroupVoteRules[LOOM_AMDGPU_SUBGROUP_VOTE_ALL],
      loom_kernel_subgroup_vote_all_predicate(op));
}
