// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_repack_packed_b16.h"

#include <stdint.h>

#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum {
  // V_PERM_B32 indices 4..7 select SRC0 bytes and indices 0..3 select SRC1
  // bytes. The low selector packs SRC0.low16 before SRC1.low16; the reversed
  // selector packs SRC1.low16 before SRC0.low16.
  LOOM_AMDGPU_FRAGMENT_REPACK_PACK_LOW16_SELECTOR = 0x01000504u,
  LOOM_AMDGPU_FRAGMENT_REPACK_PACK_LOW16_REVERSED_SELECTOR = 0x05040100u,
  // Wave32 lanes 16..31 contain the odd result rows paired with lanes 0..15.
  LOOM_AMDGPU_FRAGMENT_REPACK_ODD_ROW_LANE_MASK = 0xFFFF0000u,
};

static bool loom_amdgpu_fragment_repack_axis_layout_matches(
    const loom_matrix_fragment_axis_layout_t* axis, uint16_t outer_count,
    uint16_t thread_count, uint16_t thread_stride, uint16_t element_count) {
  return axis->outer_count == outer_count &&
         axis->thread_count == thread_count &&
         axis->thread_stride == thread_stride &&
         axis->element_count == element_count;
}

bool loom_amdgpu_select_result_to_rhs_packed_b16_fragment_repack_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* source_role_layout,
    const loom_matrix_fragment_role_layout_t* result_role_layout,
    loom_amdgpu_fragment_repack_plan_t* plan) {
  if (plan->source_role != LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
      plan->result_role != LOOM_CONTRACT_OPERAND_ROLE_RHS ||
      !loom_type_equal(plan->source_type, plan->result_type) ||
      layout->wave_size != 32 || layout->tile_shape.block_count != 1 ||
      layout->tile_shape.result_row_count !=
          layout->tile_shape.reduction_count ||
      layout->tile_shape.result_column_count != 16 ||
      source_role_layout->element_bit_count != 16 ||
      result_role_layout->element_bit_count != 16 ||
      source_role_layout->coordinate_element_offset != 0 ||
      source_role_layout->coordinate_element_stride != 2 ||
      result_role_layout->coordinate_element_offset != 0 ||
      result_role_layout->coordinate_element_stride != 1 ||
      loom_amdgpu_matrix_fragment_payload_elements_per_register(
          source_role_layout) != 2 ||
      loom_amdgpu_matrix_fragment_payload_elements_per_register(
          result_role_layout) != 2 ||
      source_role_layout->coordinate_flags !=
          (LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |
           LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN) ||
      result_role_layout->coordinate_flags !=
          (LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN |
           LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION) ||
      source_role_layout->register_count == 0 ||
      source_role_layout->register_count !=
          result_role_layout->register_count ||
      source_role_layout->payload_element_count !=
          source_role_layout->register_count * 2u ||
      result_role_layout->payload_element_count !=
          result_role_layout->register_count * 2u ||
      layout->tile_shape.result_row_count !=
          source_role_layout->register_count * 2u ||
      layout->tile_shape.reduction_count !=
          result_role_layout->register_count * 2u) {
    return false;
  }

  const loom_matrix_fragment_axis_layout_t* source_row_layout =
      &source_role_layout->axes[LOOM_MATRIX_FRAGMENT_AXIS_ROW];
  const loom_matrix_fragment_axis_layout_t* source_column_layout =
      &source_role_layout->axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN];
  const loom_matrix_fragment_axis_layout_t* result_column_layout =
      &result_role_layout->axes[LOOM_MATRIX_FRAGMENT_AXIS_COLUMN];
  const loom_matrix_fragment_axis_layout_t* result_reduction_layout =
      &result_role_layout->axes[LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION];
  if (!loom_amdgpu_fragment_repack_axis_layout_matches(
          source_row_layout, source_role_layout->register_count, 2,
          layout->tile_shape.result_column_count, 1) ||
      !loom_amdgpu_fragment_repack_axis_layout_matches(
          source_column_layout, 1, layout->tile_shape.result_column_count, 1,
          1) ||
      !loom_amdgpu_fragment_repack_axis_layout_matches(
          result_column_layout, 1, layout->tile_shape.result_column_count, 1,
          1) ||
      !loom_amdgpu_fragment_repack_axis_layout_matches(
          result_reduction_layout, 1, 1, 1,
          layout->tile_shape.reduction_count)) {
    return false;
  }

  static const loom_amdgpu_descriptor_ref_t kRequiredDescriptorRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_VCC_IMM,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC0_LIT_VCC,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32,
  };
  if (!loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, kRequiredDescriptorRefs,
          IREE_ARRAYSIZE(kRequiredDescriptorRefs))) {
    return false;
  }

  loom_amdgpu_direct_xor_lane_recipe_t exchange = {0};
  if (!loom_amdgpu_select_direct_xor_lane_recipe(
          descriptor_set, layout->wave_size,
          layout->tile_shape.result_column_count, &exchange)) {
    return false;
  }

  plan->strategy =
      LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_RHS_PACKED_B16_XOR_PERMUTE;
  plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE;
  plan->layout_kind = (loom_amdgpu_matrix_fragment_layout_kind_t)layout->kind;
  plan->source_register_count = source_role_layout->register_count;
  plan->result_register_count = result_role_layout->register_count;
  plan->strategy_payload.result_to_rhs_exchange = exchange;
  return true;
}

static iree_status_t loom_amdgpu_fragment_repack_source_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan, loom_value_id_t low_source,
    loom_type_t vgpr_type, loom_value_id_t* out_source_registers) {
  if (plan->source_register_count == 1) {
    out_source_registers[0] = low_source;
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                    low_source, i, vgpr_type,
                                                    &out_source_registers[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_repack_vcc_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, uint32_t mask,
    loom_type_t vcc_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1];
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), mask, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  loom_op_t* low_const = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, descriptor, loom_make_named_attr_slice(attrs, attr_count),
      vcc_type, source_op->location, &low_const));
  *out_mask = loom_low_const_result(low_const);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_result_to_rhs_packed_b16_fragment_repack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t vcc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vcc_type(context, &vcc_type));

  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_repack_source_registers(
      context, source_op, plan, low_source, vgpr_type, source_registers));

  loom_low_lower_resolved_descriptor_t exchange_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, plan->strategy_payload.result_to_rhs_exchange.descriptor_ref,
      &exchange_descriptor));
  loom_low_lower_resolved_descriptor_t vcc_constant_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_VCC_IMM,
      &vcc_constant_descriptor));
  loom_low_lower_resolved_descriptor_t selector_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC0_LIT_VCC,
      &selector_descriptor));
  loom_low_lower_resolved_descriptor_t permute_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32, &permute_descriptor));

  loom_value_id_t low_odd_row_lanes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_repack_vcc_constant(
      context, source_op, &vcc_constant_descriptor,
      LOOM_AMDGPU_FRAGMENT_REPACK_ODD_ROW_LANE_MASK, vcc_type,
      &low_odd_row_lanes));
  loom_value_id_t low_reversed_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_FRAGMENT_REPACK_PACK_LOW16_REVERSED_SELECTOR, vgpr_type,
      &low_reversed_selector));

  loom_named_attr_t selector_attrs[1];
  iree_host_size_t selector_attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("imm32"),
      LOOM_AMDGPU_FRAGMENT_REPACK_PACK_LOW16_SELECTOR, selector_attrs,
      IREE_ARRAYSIZE(selector_attrs), &selector_attr_count));
  const loom_value_id_t selector_operands[] = {
      low_reversed_selector,
      low_odd_row_lanes,
  };
  loom_op_t* low_selector_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &selector_descriptor, selector_operands,
      IREE_ARRAYSIZE(selector_operands),
      loom_make_named_attr_slice(selector_attrs, selector_attr_count),
      &vgpr_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_selector_op));
  const loom_value_id_t low_selector =
      loom_value_slice_get(loom_low_op_results(low_selector_op), 0);

  // Issue every independent lane exchange before consuming any result so the
  // scheduler can overlap their latency behind one wait frontier.
  loom_value_id_t paired_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_direct_crosslane_register(
        context, source_op, &exchange_descriptor,
        plan->strategy_payload.result_to_rhs_exchange.crosslane_kind,
        source_registers[i],
        plan->strategy_payload.result_to_rhs_exchange.immediate, vgpr_type,
        &paired_registers[i]));
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    const loom_value_id_t permute_operands[] = {
        source_registers[i],
        paired_registers[i],
        low_selector,
    };
    loom_op_t* low_permute_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &permute_descriptor, permute_operands,
        IREE_ARRAYSIZE(permute_operands), loom_named_attr_slice_empty(),
        &vgpr_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &low_permute_op));
    result_registers[i] =
        loom_value_slice_get(loom_low_op_results(low_permute_op), 0);
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}
