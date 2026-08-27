// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/bit_count.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/util/fact_table.h"

typedef struct loom_amdgpu_descriptor_requirement_span_t {
  // Descriptor requirements required by this span.
  const loom_amdgpu_descriptor_requirement_t* requirements;
  // Number of entries in requirements.
  iree_host_size_t requirement_count;
} loom_amdgpu_descriptor_requirement_span_t;

#define LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(requirements_) \
  {                                                            \
      .requirements = requirements_,                           \
      .requirement_count = IREE_ARRAYSIZE(requirements_),      \
  }

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarCttzSgprB32Requirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_ctz_i32_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_CTZ_I32_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_min_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MIN_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarCttzSgprB64Requirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_ctz_i32_b64"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_CTZ_I32_B64,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_min_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MIN_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarCttzVgprB32Requirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_ctz_i32_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CTZ_I32_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_min_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarCttzVgprB64Requirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_ctz_i32_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CTZ_I32_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_min_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_add_u32.src0_inline"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_SRC0_INLINE,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
};

static const loom_amdgpu_descriptor_requirement_span_t
    kAmdgpuScalarCttzRequirementRows[LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64 +
                                     1] = {
        [LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B32] =
            LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                kAmdgpuScalarCttzSgprB32Requirements),
        [LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B64] =
            LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                kAmdgpuScalarCttzSgprB64Requirements),
        [LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B32] =
            LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                kAmdgpuScalarCttzVgprB32Requirements),
        [LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64] =
            LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                kAmdgpuScalarCttzVgprB64Requirements),
};

static const loom_amdgpu_scalar_cttz_kind_t
    kAmdgpuScalarCttzKinds[LOOM_AMDGPU_REG_CLASS_ID_VGPR +
                           1][LOOM_AMDGPU_REG_CLASS_ID_VGPR + 1][2] = {
        [LOOM_AMDGPU_REG_CLASS_ID_SGPR][LOOM_AMDGPU_REG_CLASS_ID_SGPR] =
            {
                LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B32,
                LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B64,
            },
        [LOOM_AMDGPU_REG_CLASS_ID_VGPR][LOOM_AMDGPU_REG_CLASS_ID_SGPR] =
            {
                LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B32,
                LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64,
            },
        [LOOM_AMDGPU_REG_CLASS_ID_VGPR][LOOM_AMDGPU_REG_CLASS_ID_VGPR] =
            {
                LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B32,
                LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64,
            },
};

static const loom_amdgpu_descriptor_ref_t kAmdgpuScalarCttzNativeDescriptorRefs
    [LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64 + 1] = {
        [LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B32] =
            LOOM_AMDGPU_DESCRIPTOR_REF_S_CTZ_I32_B32,
        [LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B64] =
            LOOM_AMDGPU_DESCRIPTOR_REF_S_CTZ_I32_B64,
        [LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B32] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CTZ_I32_B32,
        [LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CTZ_I32_B32,
};

#undef LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN

static bool loom_amdgpu_scalar_cttz_requirements_present(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_scalar_cttz_kind_t kind,
    iree_string_view_t* out_constraint_key) {
  const loom_amdgpu_descriptor_requirement_span_t span =
      kAmdgpuScalarCttzRequirementRows[kind];
  return loom_amdgpu_descriptor_requirements_present(
      descriptor_set, span.requirements, span.requirement_count,
      out_constraint_key);
}

static uint8_t loom_amdgpu_scalar_cttz_semantic_bit_width(
    const loom_module_t* module, loom_value_id_t value) {
  const loom_type_t type = loom_module_value_type(module, value);
  const int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  return bit_width == 32 || bit_width == 64 ? (uint8_t)bit_width : 0;
}

iree_status_t loom_amdgpu_select_scalar_cttz_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_cttz_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_scalar_cttz_plan_t){0};
  *out_selected = false;

  const loom_value_id_t source = loom_scalar_cttzi_input(source_op);
  const loom_value_id_t result = loom_scalar_cttzi_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint8_t semantic_bit_width =
      loom_amdgpu_scalar_cttz_semantic_bit_width(module, source);
  if (semantic_bit_width == 0) {
    return iree_ok_status();
  }

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type) ||
      !loom_low_type_is_register(result_low_type)) {
    return iree_ok_status();
  }

  const uint32_t source_unit_count =
      loom_low_register_type_unit_count(source_low_type);
  if (source_unit_count != 1 && source_unit_count != 2) {
    return iree_ok_status();
  }
  const uint16_t source_register_class =
      loom_low_register_type_class_id(source_low_type);
  const uint16_t result_register_class =
      loom_low_register_type_class_id(result_low_type);
  if (source_register_class > LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      result_register_class > LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_ok_status();
  }
  const loom_amdgpu_scalar_cttz_kind_t kind =
      kAmdgpuScalarCttzKinds[result_register_class][source_register_class]
                            [source_unit_count - 1];
  if (kind == LOOM_AMDGPU_SCALAR_CTTZ_KIND_NONE) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_scalar_cttz_requirements_present(
          loom_low_lower_context_descriptor_set(context), kind,
          &constraint_key)) {
    return iree_ok_status();
  }

  loom_amdgpu_scalar_cttz_flags_t flags = 0;
  const loom_value_facts_t source_facts = loom_value_fact_table_lookup(
      loom_low_lower_context_fact_table(context), source);
  if (loom_value_facts_is_non_zero(source_facts)) {
    flags |= LOOM_AMDGPU_SCALAR_CTTZ_FLAG_SOURCE_NONZERO;
  }
  *out_plan = (loom_amdgpu_scalar_cttz_plan_t){
      .source = source,
      .result = result,
      .kind = kind,
      .semantic_bit_width = semantic_bit_width,
      .flags = flags,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_cttz(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }

  const loom_value_id_t source = loom_scalar_cttzi_input(op);
  const loom_module_t* module = loom_target_low_legality_module(context);
  const uint8_t semantic_bit_width =
      loom_amdgpu_scalar_cttz_semantic_bit_width(module, source);
  if (semantic_bit_width == 0) {
    return iree_ok_status();
  }
  *out_handled = true;

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, loom_scalar_cttzi_result(op), &result_prefers_vgpr));
  const loom_amdgpu_scalar_cttz_kind_t kinds[] = {
      result_prefers_vgpr ? LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B32
                          : LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B32,
      result_prefers_vgpr ? LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64
                          : LOOM_AMDGPU_SCALAR_CTTZ_KIND_SGPR_B64,
  };
  const iree_host_size_t kind_count = semantic_bit_width == 64 ? 2 : 1;
  iree_string_view_t constraint_key = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < kind_count; ++i) {
    if (!loom_amdgpu_scalar_cttz_requirements_present(
            loom_target_low_legality_descriptor_set(context), kinds[i],
            &constraint_key)) {
      return loom_amdgpu_low_legality_reject(context, op, constraint_key);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_scalar_cttz_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t low_source,
    loom_type_t lane_type, loom_value_id_t* out_low_count) {
  *out_low_count = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {low_source};
  loom_op_t* count_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &lane_type, 1, &count_op));
  *out_low_count = loom_value_slice_get(loom_low_op_results(count_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_repair_scalar_cttz_zero(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t semantic_bit_width, loom_amdgpu_scalar_cttz_kind_t kind,
    loom_type_t lane_type, loom_value_id_t low_native_count,
    loom_value_id_t* out_low_count) {
  const bool use_vgpr = kind == LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B32 ||
                        kind == LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64;
  loom_value_id_t low_width = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op,
      use_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
               : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      semantic_bit_width, lane_type, &low_width));
  if (use_vgpr) {
    return loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32,
        low_native_count, low_width, lane_type, out_low_count);
  }
  return loom_amdgpu_emit_sgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MIN_U32,
      low_native_count, low_width, lane_type, out_low_count);
}

static iree_status_t loom_amdgpu_emit_scalar_cttz_vgpr_b64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_cttz_plan_t* plan, loom_value_id_t low_source,
    loom_type_t vgpr_type, loom_value_id_t* out_low_count) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source, &low_source));

  loom_value_id_t low_source_half = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                  low_source, /*offset=*/0,
                                                  vgpr_type, &low_source_half));
  loom_value_id_t high_source_half = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_low_slice(context, source_op, low_source, /*offset=*/1,
                                 vgpr_type, &high_source_half));

  loom_value_id_t low_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scalar_cttz_native(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CTZ_I32_B32,
      low_source_half, vgpr_type, &low_count));
  loom_value_id_t high_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scalar_cttz_native(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CTZ_I32_B32,
      high_source_half, vgpr_type, &high_count));

  // Native CTZ returns -1 for zero. Repairing the high half to 32 before
  // adding its bit offset makes unsigned min select the low count, high count,
  // or Loom's required 64 for an all-zero source without a lane-mask branch.
  if (!iree_any_bit_set(plan->flags,
                        LOOM_AMDGPU_SCALAR_CTTZ_FLAG_SOURCE_NONZERO)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_repair_scalar_cttz_zero(
        context, source_op, /*semantic_bit_width=*/32, plan->kind, vgpr_type,
        high_count, &high_count));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, high_count,
      /*immediate=*/32, vgpr_type, &high_count));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32, low_count,
      high_count, vgpr_type, out_low_count);
}

iree_status_t loom_amdgpu_lower_scalar_cttz(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_cttz_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  const uint32_t result_unit_count =
      loom_low_register_type_unit_count(result_type);
  const bool use_vgpr = plan->kind == LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B32 ||
                        plan->kind == LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64;

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  loom_type_t lane_type = loom_type_none();
  if (use_vgpr) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &lane_type));
  }

  loom_value_id_t low_count = LOOM_VALUE_ID_INVALID;
  if (plan->kind == LOOM_AMDGPU_SCALAR_CTTZ_KIND_VGPR_B64) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scalar_cttz_vgpr_b64(
        context, source_op, plan, low_source, lane_type, &low_count));
  } else {
    const loom_amdgpu_descriptor_ref_t descriptor_ref =
        kAmdgpuScalarCttzNativeDescriptorRefs[plan->kind];
    if (use_vgpr) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, low_source, &low_source));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scalar_cttz_native(
        context, source_op, descriptor_ref, low_source, lane_type, &low_count));
    if (!iree_any_bit_set(plan->flags,
                          LOOM_AMDGPU_SCALAR_CTTZ_FLAG_SOURCE_NONZERO)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_repair_scalar_cttz_zero(
          context, source_op, plan->semantic_bit_width, plan->kind, lane_type,
          low_count, &low_count));
    }
  }

  if (result_unit_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_count);
  }
  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op,
      use_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
               : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      0, lane_type, &low_zero));
  const loom_value_id_t low_halves[] = {
      low_count,
      low_zero,
  };
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, low_halves, IREE_ARRAYSIZE(low_halves), result_type,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}
