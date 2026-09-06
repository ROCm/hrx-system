// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_access.h"

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_bank_service.h"
#include "loom/target/arch/amdgpu/lower/memory_subgroup_access.h"
#include "loom/target/arch/amdgpu/lower/types.h"

iree_status_t loom_amdgpu_fragment_memory_packet_type(
    loom_low_lower_context_t* context, uint16_t packet_register_count,
    loom_type_t vgpr_type, loom_type_t* out_type) {
  if (packet_register_count == 1) {
    *out_type = vgpr_type;
    return iree_ok_status();
  }
  return loom_amdgpu_make_vgpr_range_type(context, packet_register_count,
                                          out_type);
}

static loom_low_memory_space_t loom_amdgpu_fragment_memory_low_space(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return LOOM_LOW_MEMORY_SPACE_GLOBAL;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return LOOM_LOW_MEMORY_SPACE_GENERIC;
  }
}

static bool loom_amdgpu_fragment_memory_uses_buffer_descriptor(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  return plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR;
}

static iree_string_view_t loom_amdgpu_fragment_memory_report_address_form(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (loom_amdgpu_fragment_memory_uses_buffer_descriptor(plan)) {
    return IREE_SV("buffer_vaddr");
  }
  if (plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
      plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) {
    return loom_amdgpu_memory_address_form_name(
        LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR);
  }
  return loom_amdgpu_memory_address_form_name(
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT);
}

static bool loom_amdgpu_fragment_memory_packet_static_offset(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, int64_t* out_static_byte_offset) {
  return loom_amdgpu_fragment_memory_static_offset_i64(
      plan, packet->register_index, element_index, out_static_byte_offset);
}

static void loom_amdgpu_fragment_memory_add_runtime_packet_source_interval(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, loom_low_lower_memory_report_row_t* row) {
  const loom_low_byte_interval_precision_flags_t required_precision =
      LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
      LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE;
  if (!iree_all_bits_set(row->source_interval.precision_flags,
                         required_precision)) {
    return;
  }

  loom_value_facts_t runtime_packet_offset = loom_value_facts_exact_i64(0);
  bool has_runtime_packet_offset = false;
  for (uint8_t view_axis = 0; view_axis < plan->view_rank; ++view_axis) {
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
        &plan->runtime_axes[view_axis];
    const uint64_t packet_coordinate =
        (uint64_t)runtime_axis->register_coordinates[packet->register_index] +
        (uint64_t)element_index *
            runtime_axis->packed_element_coordinate_stride;
    if (packet_coordinate == 0) {
      continue;
    }
    IREE_ASSERT_LE(packet_coordinate, INT64_MAX);
    const loom_value_facts_t coordinate_facts =
        loom_value_facts_exact_i64((int64_t)packet_coordinate);
    loom_value_facts_t axis_offset = loom_value_facts_unknown();
    loom_value_facts_muli(&runtime_axis->byte_stride.byte_facts,
                          &coordinate_facts, &axis_offset);
    loom_value_facts_addi(&runtime_packet_offset, &axis_offset,
                          &runtime_packet_offset);
    has_runtime_packet_offset = true;
  }
  if (!has_runtime_packet_offset) {
    return;
  }

  loom_value_facts_addi(&row->source_interval.begin_facts,
                        &runtime_packet_offset,
                        &row->source_interval.begin_facts);
  loom_value_facts_addi(&row->source_interval.end_facts, &runtime_packet_offset,
                        &row->source_interval.end_facts);
  row->source_interval.begin_expr_id = LOOM_LOW_MEMORY_EXPR_ID_NONE;
  row->source_interval.end_expr_id = LOOM_LOW_MEMORY_EXPR_ID_NONE;
  row->source_interval.precision_flags &=
      ~(LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_EXPR |
        LOOM_LOW_BYTE_INTERVAL_PRECISION_END_EXPR);
}

iree_status_t loom_amdgpu_fragment_memory_packet_resource(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_binding,
    loom_value_id_t* out_low_packet_resource,
    loom_value_id_t* out_low_soffset) {
  *out_low_packet_resource = low_binding;
  *out_low_soffset = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_fragment_memory_uses_buffer_descriptor(plan)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, low_binding, &plan->source, out_low_packet_resource));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
                                    sgpr_type, out_low_soffset);
}

static iree_status_t loom_amdgpu_record_fragment_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_op_t* low_op, const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint16_t element_index,
    uint32_t vector_lane_count) {
  loom_low_memory_access_summary_t summary = {
      .memory_space =
          loom_amdgpu_fragment_memory_low_space(plan->source.memory_space),
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
  };
  if (summary.memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC) {
    summary.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE;
  }
  if (plan->source.alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    summary.alias_root_id = plan->source.alias_scope_id;
    summary.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  }
  // Workgroup allocations have compiler-owned identities that remain comparable
  // after source lowering. Preserve those summaries so final packet scheduling
  // can distinguish disjoint LDS allocations from real async-memory hazards.
  const bool preserve_memory_access =
      plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
      plan->source.alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
  const loom_low_lower_memory_access_record_flags_t record_flags =
      preserve_memory_access ? LOOM_LOW_LOWER_MEMORY_ACCESS_RECORD_PRESERVE : 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_record_memory_access_summary(
      context, low_op, &summary, record_flags));
  if (!loom_low_lower_context_wants_report_rows(context)) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  const loom_low_descriptor_memory_effect_summary_t issued =
      loom_low_descriptor_memory_effect_summary(descriptor_set, descriptor);
  iree_string_view_t packet_key = iree_string_view_empty();
  if (descriptor != NULL) {
    packet_key = loom_low_descriptor_set_string(descriptor_set,
                                                descriptor->key_string_offset);
  }
  int64_t static_offset_bytes = plan->source.static_byte_offset;
  (void)loom_amdgpu_fragment_memory_packet_static_offset(
      plan, packet, element_index, &static_offset_bytes);
  loom_low_source_memory_access_plan_t packet_source = plan->source;
  packet_source.static_byte_offset = static_offset_bytes;
  packet_source.element_byte_count = plan->element_byte_count;
  packet_source.vector_lane_count = vector_lane_count;
  packet_source.vector_lane_byte_stride = plan->element_byte_count;
  loom_amdgpu_fragment_memory_packet_report_t packet_report = {0};
  loom_amdgpu_fragment_memory_query_packet_report(plan, packet, &packet_report);
  loom_low_lower_memory_subgroup_access_report_t subgroup_access = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_report_subgroup_access(
      context, source_op, layout, plan, packet, element_index, &issued,
      &subgroup_access));
  const bool runtime_packet_offset_is_subgroup_uniform =
      loom_amdgpu_fragment_memory_runtime_packet_offset_is_subgroup_uniform(
          plan, packet->register_index, element_index);
  loom_low_lower_memory_report_row_t row = {
      .function_name = loom_low_lower_context_function_name(context),
      .source_op_name =
          loom_op_name(loom_low_lower_context_module(context), source_op),
      .source_op_kind = source_op->kind,
      .source_root_name = loom_module_value_name(
          loom_low_lower_context_module(context), plan->source.root_value_id),
      .source_root_argument_index =
          loom_low_lower_source_memory_root_argument_index(context,
                                                           &plan->source),
      .memory_space = loom_amdgpu_memory_space_name(plan->source.memory_space),
      .operation_kind = loom_amdgpu_memory_operation_name(plan->operation_kind),
      .packet_key = packet_key,
      .strategy_key = packet_report.strategy_key,
      .address_form = loom_amdgpu_fragment_memory_report_address_form(plan),
      .dynamic_term_kind = IREE_SV("vaddr"),
      .fallback_reason = packet_report.fallback_reason,
      .static_offset_bytes = static_offset_bytes,
      .element_byte_count = plan->element_byte_count,
      .vector_lane_count = vector_lane_count,
      .issued_read_byte_count = issued.read_byte_count,
      .issued_write_byte_count = issued.write_byte_count,
      .issued_read_unknown_width_count = issued.read_unknown_width_count,
      .issued_write_unknown_width_count = issued.write_unknown_width_count,
      .dynamic_stride_bytes = runtime_packet_offset_is_subgroup_uniform
                                  ? plan->address_layout.linear_lane_byte_stride
                                  : 0,
      .vector_lane_stride_bytes = plan->element_byte_count,
      .subgroup_access = subgroup_access,
  };
  loom_amdgpu_memory_report_row_populate_storage_schema(context, &plan->source,
                                                        &row);
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_report_bank_service(
      context, source_op, layout, plan, packet, element_index,
      &row.bank_service));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_memory_report_row_populate_source_interval(
          context, &packet_source, &row));
  loom_amdgpu_fragment_memory_add_runtime_packet_source_interval(
      plan, packet, element_index, &row);
  return loom_low_lower_record_memory_report_row(context, source_op, &row);
}

static iree_status_t loom_amdgpu_make_fragment_memory_attrs(
    loom_low_lower_context_t* context, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, int64_t immediate_offset,
    iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  return loom_amdgpu_append_i64_attr(context, IREE_SV("offset"),
                                     immediate_offset, attrs, attr_capacity,
                                     out_attr_count);
}

static iree_status_t loom_amdgpu_emit_fragment_load_packet_with_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_id_t low_tied_source, uint16_t element_index,
    uint32_t vector_lane_count, loom_type_t result_type,
    const loom_amdgpu_fragment_memory_address_t* address,
    loom_value_id_t low_resource, loom_value_id_t low_soffset,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_fragment_memory_attrs(
      context, attrs, IREE_ARRAYSIZE(attrs), address->immediate_offset,
      &attr_count));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
  if (loom_low_descriptor_implicit_resource_operand(
          loom_low_lower_context_descriptor_set(context),
          descriptor.descriptor) != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_m0_u32(context, source_op, &descriptor, 0, &low_m0));
  }

  loom_value_id_t operands[5] = {0};
  iree_host_size_t operand_count = 0;
  if (low_tied_source != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = low_tied_source;
  }
  if (loom_amdgpu_fragment_memory_uses_buffer_descriptor(plan)) {
    operands[operand_count++] = low_resource;
    operands[operand_count++] = address->low_vaddr;
    operands[operand_count++] = low_soffset;
  } else {
    operands[operand_count++] = address->low_vaddr;
    if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      operands[operand_count++] = low_resource;
    }
  }
  if (low_m0 != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = low_m0;
  }
  loom_op_t* low_op = NULL;
  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  const loom_tied_result_t* tied_results =
      low_tied_source != LOOM_VALUE_ID_INVALID ? &tied_result : NULL;
  const iree_host_size_t tied_result_count =
      low_tied_source != LOOM_VALUE_ID_INVALID ? 1 : 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
      tied_results, tied_result_count, source_op->location, &low_op));
  IREE_RETURN_IF_ERROR(loom_amdgpu_record_fragment_memory_packet(
      context, source_op, low_op, layout, plan, packet, descriptor_ref,
      element_index, vector_lane_count));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fragment_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, uint32_t vector_lane_count, loom_type_t result_type,
    const loom_amdgpu_fragment_memory_address_t* address,
    loom_value_id_t low_resource, loom_value_id_t low_soffset,
    loom_value_id_t* out_low_packet) {
  return loom_amdgpu_emit_fragment_load_packet_with_descriptor(
      context, source_op, layout, plan, packet, packet->descriptor_ref,
      LOOM_VALUE_ID_INVALID, element_index, vector_lane_count, result_type,
      address, low_resource, low_soffset, out_low_packet);
}

iree_status_t loom_amdgpu_emit_fragment_load_high_half_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, uint32_t vector_lane_count, loom_type_t result_type,
    const loom_amdgpu_fragment_memory_address_t* address,
    loom_value_id_t low_partial_packet, loom_value_id_t low_resource,
    loom_value_id_t low_soffset, loom_value_id_t* out_low_packet) {
  return loom_amdgpu_emit_fragment_load_packet_with_descriptor(
      context, source_op, layout, plan, packet,
      plan->packed_b16_high_descriptor_ref, low_partial_packet, element_index,
      vector_lane_count, result_type, address, low_resource, low_soffset,
      out_low_packet);
}

iree_status_t loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_packet, loom_type_t vgpr_type,
    loom_value_id_t* out_full_packet) {
  *out_full_packet = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_low_value_defines_vgpr_low16(context, low_packet)) {
    return loom_amdgpu_emit_vgpr_unary(
        context, source_op,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_0_WIDTH_16_LOW16,
        low_packet, vgpr_type, out_full_packet);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_packet, out_full_packet));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      *out_full_packet, UINT32_C(0xFFFF), vgpr_type, out_full_packet);
}

iree_status_t loom_amdgpu_emit_fragment_store_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, uint32_t vector_lane_count,
    const loom_amdgpu_fragment_memory_address_t* address,
    loom_value_id_t low_payload_register, loom_value_id_t low_resource,
    loom_value_id_t low_soffset) {
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_fragment_memory_attrs(
      context, attrs, IREE_ARRAYSIZE(attrs), address->immediate_offset,
      &attr_count));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, packet->descriptor_ref, &descriptor));
  loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
  if (loom_low_descriptor_implicit_resource_operand(
          loom_low_lower_context_descriptor_set(context),
          descriptor.descriptor) != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_m0_u32(context, source_op, &descriptor, 0, &low_m0));
  }

  loom_value_id_t operands[5] = {0};
  iree_host_size_t operand_count = 0;
  if (loom_amdgpu_fragment_memory_uses_buffer_descriptor(plan)) {
    operands[operand_count++] = low_payload_register;
    operands[operand_count++] = low_resource;
    operands[operand_count++] = address->low_vaddr;
    operands[operand_count++] = low_soffset;
  } else {
    operands[operand_count++] = address->low_vaddr;
    operands[operand_count++] = low_payload_register;
    if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      operands[operand_count++] = low_resource;
    }
  }
  if (low_m0 != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = low_m0;
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  return loom_amdgpu_record_fragment_memory_packet(
      context, source_op, low_op, layout, plan, packet, packet->descriptor_ref,
      element_index, vector_lane_count);
}
