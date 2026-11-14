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

typedef uint8_t loom_amdgpu_scalar_bit_scan_operation_t;

enum loom_amdgpu_scalar_bit_scan_operation_e {
  LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_NONE = 0,
  LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING = 1,
  LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_TRAILING = 2,
};

static const loom_amdgpu_scalar_bit_scan_kind_t
    kAmdgpuScalarBitScanKinds[LOOM_AMDGPU_REG_CLASS_ID_VGPR +
                              1][LOOM_AMDGPU_REG_CLASS_ID_VGPR + 1][2] = {
        [LOOM_AMDGPU_REG_CLASS_ID_SGPR][LOOM_AMDGPU_REG_CLASS_ID_SGPR] =
            {
                LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B32,
                LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B64,
            },
        [LOOM_AMDGPU_REG_CLASS_ID_VGPR][LOOM_AMDGPU_REG_CLASS_ID_SGPR] =
            {
                LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B32,
                LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64,
            },
        [LOOM_AMDGPU_REG_CLASS_ID_VGPR][LOOM_AMDGPU_REG_CLASS_ID_VGPR] =
            {
                LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B32,
                LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64,
            },
};

static const loom_amdgpu_descriptor_ref_t
    kAmdgpuScalarBitScanNativeDescriptorRefs
        [LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_TRAILING + 1]
        [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64 + 1] = {
            [LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING] =
                {
                    [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B32] =
                        LOOM_AMDGPU_DESCRIPTOR_REF_S_CLZ_I32_U32,
                    [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B64] =
                        LOOM_AMDGPU_DESCRIPTOR_REF_S_CLZ_I32_U32,
                    [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B32] =
                        LOOM_AMDGPU_DESCRIPTOR_REF_V_CLZ_I32_U32,
                    [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64] =
                        LOOM_AMDGPU_DESCRIPTOR_REF_V_CLZ_I32_U32,
                },
            [LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_TRAILING] =
                {
                    [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B32] =
                        LOOM_AMDGPU_DESCRIPTOR_REF_S_CTZ_I32_B32,
                    [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B64] =
                        LOOM_AMDGPU_DESCRIPTOR_REF_S_CTZ_I32_B64,
                    [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B32] =
                        LOOM_AMDGPU_DESCRIPTOR_REF_V_CTZ_I32_B32,
                    [LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64] =
                        LOOM_AMDGPU_DESCRIPTOR_REF_V_CTZ_I32_B32,
                },
};

static bool loom_amdgpu_scalar_bit_scan_op_values(
    const loom_op_t* op, loom_amdgpu_scalar_bit_scan_operation_t* out_operation,
    loom_value_id_t* out_source, loom_value_id_t* out_result) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_CTLZI:
      *out_operation = LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING;
      *out_source = loom_scalar_ctlzi_input(op);
      *out_result = loom_scalar_ctlzi_result(op);
      return true;
    case LOOM_OP_SCALAR_CTTZI:
      *out_operation = LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_TRAILING;
      *out_source = loom_scalar_cttzi_input(op);
      *out_result = loom_scalar_cttzi_result(op);
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_scalar_bit_scan_kind_uses_vgpr(
    loom_amdgpu_scalar_bit_scan_kind_t kind) {
  return kind == LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B32 ||
         kind == LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64;
}

static bool loom_amdgpu_scalar_bit_scan_kind_uses_b64(
    loom_amdgpu_scalar_bit_scan_kind_t kind) {
  return kind == LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B64 ||
         kind == LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64;
}

static iree_string_view_t loom_amdgpu_scalar_bit_scan_native_constraint_key(
    loom_amdgpu_scalar_bit_scan_operation_t operation,
    loom_amdgpu_scalar_bit_scan_kind_t kind) {
  const bool use_vgpr = loom_amdgpu_scalar_bit_scan_kind_uses_vgpr(kind);
  if (operation == LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING) {
    return use_vgpr ? IREE_SV("descriptor.v_clz_i32_u32")
                    : IREE_SV("descriptor.s_clz_i32_u32");
  }
  if (use_vgpr) {
    return IREE_SV("descriptor.v_ctz_i32_b32");
  }
  return loom_amdgpu_scalar_bit_scan_kind_uses_b64(kind)
             ? IREE_SV("descriptor.s_ctz_i32_b64")
             : IREE_SV("descriptor.s_ctz_i32_b32");
}

static bool loom_amdgpu_scalar_bit_scan_requirements_present(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_scalar_bit_scan_operation_t operation,
    loom_amdgpu_scalar_bit_scan_kind_t kind, uint8_t semantic_bit_width,
    iree_string_view_t* out_constraint_key) {
  const bool use_vgpr = loom_amdgpu_scalar_bit_scan_kind_uses_vgpr(kind);
  if (!loom_amdgpu_descriptor_requirement_present(
          descriptor_set,
          loom_amdgpu_scalar_bit_scan_native_constraint_key(operation, kind),
          kAmdgpuScalarBitScanNativeDescriptorRefs[operation][kind],
          out_constraint_key)) {
    return false;
  }

  if (!loom_amdgpu_descriptor_requirement_present(
          descriptor_set,
          use_vgpr ? IREE_SV("descriptor.v_min_u32")
                   : IREE_SV("descriptor.s_min_u32"),
          use_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32
                   : LOOM_AMDGPU_DESCRIPTOR_REF_S_MIN_U32,
          out_constraint_key)) {
    return false;
  }

  const bool needs_word_offset =
      (operation == LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING &&
       semantic_bit_width == 64) ||
      (operation == LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_TRAILING &&
       kind == LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64);
  if (needs_word_offset &&
      !loom_amdgpu_descriptor_requirement_present(
          descriptor_set,
          use_vgpr ? IREE_SV("descriptor.v_add_u32.src0_inline")
                   : IREE_SV("descriptor.s_add_u32.rhs_inline"),
          use_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_SRC0_INLINE
                   : LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32_RHS_INLINE,
          out_constraint_key)) {
    return false;
  }

  if (!loom_amdgpu_descriptor_requirement_present(
          descriptor_set,
          use_vgpr ? IREE_SV("descriptor.v_mov_b32")
                   : IREE_SV("descriptor.s_mov_b32"),
          use_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                   : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          out_constraint_key)) {
    return false;
  }
  return !use_vgpr ||
         loom_amdgpu_descriptor_requirement_present(
             descriptor_set, IREE_SV("descriptor.v_mov_b32_copy"),
             LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY, out_constraint_key);
}

static uint8_t loom_amdgpu_scalar_bit_scan_semantic_bit_width(
    const loom_module_t* module, loom_value_id_t value) {
  const loom_type_t type = loom_module_value_type(module, value);
  const int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  return bit_width == 32 || bit_width == 64 ? (uint8_t)bit_width : 0;
}

iree_status_t loom_amdgpu_select_scalar_bit_scan_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_bit_scan_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_scalar_bit_scan_plan_t){0};
  *out_selected = false;

  loom_amdgpu_scalar_bit_scan_operation_t operation =
      LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_NONE;
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_bit_scan_op_values(source_op, &operation, &source,
                                             &result)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint8_t semantic_bit_width =
      loom_amdgpu_scalar_bit_scan_semantic_bit_width(module, source);
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
  const loom_amdgpu_scalar_bit_scan_kind_t kind =
      kAmdgpuScalarBitScanKinds[result_register_class][source_register_class]
                               [source_unit_count - 1];
  if (kind == LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_NONE) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_scalar_bit_scan_requirements_present(
          loom_low_lower_context_descriptor_set(context), operation, kind,
          semantic_bit_width, &constraint_key)) {
    return iree_ok_status();
  }

  loom_amdgpu_scalar_bit_scan_flags_t flags =
      operation == LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING
          ? LOOM_AMDGPU_SCALAR_BIT_SCAN_FLAG_LEADING
          : 0;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_value_facts_t source_facts =
      loom_value_fact_table_lookup(fact_table, source);
  const loom_value_facts_t result_facts =
      loom_value_fact_table_lookup(fact_table, result);
  // A guarded definition can exclude the zero-input result even when the
  // incoming value retains its unrestricted facts for other uses.
  if (loom_value_facts_is_non_zero(source_facts) ||
      result_facts.range_hi < semantic_bit_width) {
    flags |= LOOM_AMDGPU_SCALAR_BIT_SCAN_FLAG_SOURCE_NONZERO;
  }
  *out_plan = (loom_amdgpu_scalar_bit_scan_plan_t){
      .source = source,
      .result = result,
      .kind = kind,
      .result_unit_count =
          (uint8_t)loom_low_register_type_unit_count(result_low_type),
      .semantic_bit_width = semantic_bit_width,
      .flags = flags,
  };
  *out_selected = true;
  return loom_low_lower_plan_value_type(context, result, result_low_type);
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_bit_scan(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;

  loom_amdgpu_scalar_bit_scan_operation_t operation =
      LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_NONE;
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_bit_scan_op_values(op, &operation, &source,
                                             &result)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_target_low_legality_module(context);
  const uint8_t semantic_bit_width =
      loom_amdgpu_scalar_bit_scan_semantic_bit_width(module, source);
  if (semantic_bit_width == 0) {
    return iree_ok_status();
  }
  *out_handled = true;

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  const loom_amdgpu_scalar_bit_scan_kind_t kinds[] = {
      result_prefers_vgpr ? LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B32
                          : LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B32,
      result_prefers_vgpr ? LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_VGPR_B64
                          : LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B64,
  };
  const iree_host_size_t kind_count = semantic_bit_width == 64 ? 2 : 1;
  iree_string_view_t constraint_key = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < kind_count; ++i) {
    if (!loom_amdgpu_scalar_bit_scan_requirements_present(
            loom_target_low_legality_descriptor_set(context), operation,
            kinds[i], semantic_bit_width, &constraint_key)) {
      return loom_amdgpu_low_legality_reject(context, op, constraint_key);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_scalar_bit_scan_native(
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

static iree_status_t loom_amdgpu_repair_scalar_bit_scan_zero(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t zero_result, loom_amdgpu_scalar_bit_scan_kind_t kind,
    loom_type_t lane_type, loom_value_id_t low_native_count,
    loom_value_id_t* out_low_count) {
  const bool use_vgpr = loom_amdgpu_scalar_bit_scan_kind_uses_vgpr(kind);
  loom_value_id_t low_zero_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op,
      use_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
               : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      zero_result, lane_type, &low_zero_result));
  return loom_amdgpu_emit_binary(
      context, source_op,
      use_vgpr ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32
               : LOOM_AMDGPU_DESCRIPTOR_REF_S_MIN_U32,
      low_native_count, low_zero_result, lane_type, out_low_count);
}

static iree_status_t loom_amdgpu_add_scalar_bit_scan_word_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_bit_scan_kind_t kind, loom_type_t lane_type,
    loom_value_id_t low_count, loom_value_id_t* out_low_count) {
  if (loom_amdgpu_scalar_bit_scan_kind_uses_vgpr(kind)) {
    return loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, low_count,
        /*immediate=*/32, lane_type, out_low_count);
  }
  return loom_amdgpu_emit_sgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, low_count,
      /*immediate=*/32, lane_type, out_low_count);
}

static iree_status_t loom_amdgpu_emit_scalar_bit_scan_b64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_bit_scan_plan_t* plan,
    loom_amdgpu_scalar_bit_scan_operation_t operation,
    loom_value_id_t low_source, loom_type_t lane_type,
    loom_value_id_t* out_low_count) {
  if (loom_amdgpu_scalar_bit_scan_kind_uses_vgpr(plan->kind)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_source, &low_source));
  }

  const uint32_t first_offset =
      operation == LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING ? 1 : 0;
  const uint32_t second_offset = first_offset ^ 1u;
  loom_value_id_t first_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source, first_offset, lane_type, &first_source));
  loom_value_id_t second_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                  low_source, second_offset,
                                                  lane_type, &second_source));

  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      kAmdgpuScalarBitScanNativeDescriptorRefs[operation][plan->kind];
  loom_value_id_t first_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scalar_bit_scan_native(
      context, source_op, descriptor_ref, first_source, lane_type,
      &first_count));
  loom_value_id_t second_count = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scalar_bit_scan_native(
      context, source_op, descriptor_ref, second_source, lane_type,
      &second_count));

  // A native zero-word scan returns UINT32_MAX. Repair the second candidate
  // before adding its word offset so unsigned min also returns 64 for an
  // all-zero source. When the full source is known nonzero, the wrapped value
  // cannot win over the scan of its nonzero word.
  if (!iree_any_bit_set(plan->flags,
                        LOOM_AMDGPU_SCALAR_BIT_SCAN_FLAG_SOURCE_NONZERO)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_repair_scalar_bit_scan_zero(
        context, source_op, /*zero_result=*/32, plan->kind, lane_type,
        second_count, &second_count));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_add_scalar_bit_scan_word_offset(
      context, source_op, plan->kind, lane_type, second_count, &second_count));
  return loom_amdgpu_emit_binary(
      context, source_op,
      loom_amdgpu_scalar_bit_scan_kind_uses_vgpr(plan->kind)
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32
          : LOOM_AMDGPU_DESCRIPTOR_REF_S_MIN_U32,
      first_count, second_count, lane_type, out_low_count);
}

iree_status_t loom_amdgpu_lower_scalar_bit_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_bit_scan_plan_t* plan) {
  const loom_amdgpu_scalar_bit_scan_operation_t operation =
      iree_any_bit_set(plan->flags, LOOM_AMDGPU_SCALAR_BIT_SCAN_FLAG_LEADING)
          ? LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING
          : LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_TRAILING;
  const bool use_vgpr = loom_amdgpu_scalar_bit_scan_kind_uses_vgpr(plan->kind);

  loom_value_id_t low_source =
      loom_low_lower_lookup_value(context, plan->source);
  loom_type_t lane_type = loom_type_none();
  if (use_vgpr) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &lane_type));
  }

  const bool use_composed_b64 =
      loom_amdgpu_scalar_bit_scan_kind_uses_b64(plan->kind) &&
      !(operation == LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_TRAILING &&
        plan->kind == LOOM_AMDGPU_SCALAR_BIT_SCAN_KIND_SGPR_B64);
  loom_value_id_t low_count = LOOM_VALUE_ID_INVALID;
  if (use_composed_b64) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scalar_bit_scan_b64(
        context, source_op, plan, operation, low_source, lane_type,
        &low_count));
  } else {
    if (use_vgpr) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
          context, source_op, low_source, &low_source));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_scalar_bit_scan_native(
        context, source_op,
        kAmdgpuScalarBitScanNativeDescriptorRefs[operation][plan->kind],
        low_source, lane_type, &low_count));
    if (!iree_any_bit_set(plan->flags,
                          LOOM_AMDGPU_SCALAR_BIT_SCAN_FLAG_SOURCE_NONZERO)) {
      const uint32_t zero_result =
          operation == LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING
              ? 32
              : plan->semantic_bit_width;
      IREE_RETURN_IF_ERROR(loom_amdgpu_repair_scalar_bit_scan_zero(
          context, source_op, zero_result, plan->kind, lane_type, low_count,
          &low_count));
    }
    if (operation == LOOM_AMDGPU_SCALAR_BIT_SCAN_OPERATION_LEADING &&
        plan->semantic_bit_width == 64) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_add_scalar_bit_scan_word_offset(
          context, source_op, plan->kind, lane_type, low_count, &low_count));
    }
  }

  if (plan->result_unit_count == 1) {
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
  return loom_amdgpu_bind_low_register_range(
      context, source_op, plan->result, low_halves, IREE_ARRAYSIZE(low_halves));
}
