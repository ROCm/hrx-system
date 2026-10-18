// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_report.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/operand.h"
#include "loom/ops/encoding/storage.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/address.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/packet.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_bank_service.h"
#include "loom/target/arch/amdgpu/lower/memory_packet.h"
#include "loom/target/arch/amdgpu/lower/memory_subgroup_access.h"
#include "loom/util/numeric_format.h"

static uint32_t loom_amdgpu_memory_report_positive_u32(int64_t value) {
  return value > 0 && value <= UINT32_MAX ? (uint32_t)value : 0;
}

static uint32_t loom_amdgpu_memory_report_dynamic_stride_bytes(
    const loom_low_source_memory_access_plan_t* source) {
  return source->dynamic_term_count == 1
             ? loom_amdgpu_memory_report_positive_u32(
                   source->dynamic_terms[0].byte_stride)
             : 0;
}

static iree_string_view_t loom_amdgpu_memory_report_storage_name(
    loom_encoding_operand_parameter_t parameter, uint64_t value,
    uint64_t omitted_value) {
  if (value == omitted_value) {
    return iree_string_view_empty();
  }
  return loom_encoding_operand_fact_name(parameter, value);
}

static void loom_amdgpu_memory_report_row_set_storage_schema(
    const loom_value_fact_storage_schema_t* schema,
    loom_low_lower_memory_report_row_t* row) {
  if (schema == NULL || loom_value_fact_encoded_operand_schema_is_unknown(
                            schema->encoded_operand)) {
    return;
  }
  const loom_value_fact_encoded_operand_schema_t encoded =
      schema->encoded_operand;
  row->storage_element_format = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_ELEMENT_FORMAT, encoded.element_format,
      LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  row->storage_scale_format = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_SCALE_FORMAT, encoded.scale_format,
      LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  row->storage_secondary_scale_format = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_SECONDARY_SCALE_FORMAT,
      encoded.secondary_scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  row->storage_payload_packing = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_PACKING, encoded.payload_packing,
      LOOM_VALUE_FACT_PAYLOAD_PACKING_UNKNOWN);
  row->storage_scale_topology = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_SCALE_TOPOLOGY, encoded.scale_topology,
      LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE);
  row->storage_affine_policy = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_AFFINE, encoded.affine_policy,
      LOOM_VALUE_FACT_AFFINE_POLICY_NONE);
  row->storage_rounding_policy = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_ROUNDING, encoded.rounding_policy,
      LOOM_VALUE_FACT_ROUNDING_POLICY_NONE);
  row->storage_codebook_policy = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_CODEBOOK, encoded.codebook_policy,
      LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE);
  row->storage_sparsity_policy = loom_amdgpu_memory_report_storage_name(
      LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY, encoded.sparsity_policy,
      LOOM_VALUE_FACT_SPARSITY_POLICY_NONE);
}

static void loom_amdgpu_memory_report_row_populate_storage_schema(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source,
    loom_low_lower_memory_report_row_t* row) {
  if (source->view_value_id == LOOM_VALUE_ID_INVALID) {
    return;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t view_type =
      loom_module_value_type(module, source->view_value_id);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_fact_context_t* fact_context =
      fact_table != NULL ? &fact_table->context : NULL;
  loom_value_fact_storage_schema_t storage_schema = {0};
  if (loom_encoding_query_type_storage_schema(fact_context, module, view_type,
                                              &storage_schema)) {
    loom_amdgpu_memory_report_row_set_storage_schema(&storage_schema, row);
  }
  if (iree_string_view_is_empty(row->storage_element_format)) {
    const loom_value_fact_numeric_format_flags_t element_format =
        loom_numeric_format_from_scalar_type(loom_type_element_type(view_type));
    row->storage_element_format = loom_amdgpu_memory_report_storage_name(
        LOOM_ENCODING_OPERAND_PARAMETER_ELEMENT_FORMAT, element_format,
        LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  }
}

static iree_status_t loom_amdgpu_record_memory_packet_report(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    iree_string_view_t strategy_key, uint64_t execution_count_plus_one) {
  loom_low_source_memory_access_plan_t packet_source;
  loom_amdgpu_memory_access_source_plan(&packet->access, &packet_source);
  const loom_low_source_memory_access_plan_t* source = &packet_source;

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_memory_effect_summary_t issued =
      loom_low_descriptor_memory_effect_summary(descriptor_set,
                                                packet->access.descriptor);
  const iree_string_view_t packet_key = loom_low_descriptor_set_string(
      descriptor_set, packet->access.descriptor->key_string_ref);
  const loom_low_source_memory_operation_kind_t operation_kind =
      source->operation_kind;
  iree_string_view_t fallback_reason = iree_string_view_empty();
  if (source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    fallback_reason = loom_amdgpu_memory_ds_addtid_reason_key(
        descriptor_set, loom_low_lower_context_module(context),
        loom_low_lower_context_source_function(context),
        loom_low_lower_context_bundle(context), &packet->access,
        operation_kind);
  }
  loom_low_lower_memory_report_row_t row = {
      .function_name = loom_low_lower_context_function_name(context),
      .source_op_name =
          loom_op_name(loom_low_lower_context_module(context), source_op),
      .source_op_kind = source_op->kind,
      .execution_count_plus_one = execution_count_plus_one,
      .source_root_name = loom_module_value_name(
          loom_low_lower_context_module(context), source->root_value_id),
      .source_root_argument_index =
          loom_low_lower_source_memory_root_argument_index(context, source),
      .memory_space = loom_amdgpu_memory_space_name(source->memory_space),
      .operation_kind = loom_amdgpu_memory_operation_name(operation_kind),
      .packet_key = packet_key,
      .strategy_key = strategy_key,
      .address_form =
          loom_amdgpu_memory_address_form_name(packet->access.address_form),
      .dynamic_term_kind =
          loom_amdgpu_memory_access_dynamic_term_kind_name(&packet->access),
      .fallback_reason = fallback_reason,
      .static_offset_bytes = source->static_byte_offset,
      .element_byte_count = source->element_byte_count,
      .vector_lane_count = source->vector_lane_count,
      .issued_read_byte_count = issued.read_byte_count,
      .issued_write_byte_count = issued.write_byte_count,
      .issued_read_unknown_width_count = issued.read_unknown_width_count,
      .issued_write_unknown_width_count = issued.write_unknown_width_count,
      .dynamic_stride_bytes =
          loom_amdgpu_memory_report_dynamic_stride_bytes(source),
      .vector_lane_stride_bytes = loom_amdgpu_memory_report_positive_u32(
          source->vector_lane_byte_stride),
  };
  loom_amdgpu_memory_report_row_populate_storage_schema(context, source, &row);
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_report_bank_service(
      context, source_op, packet->access.descriptor, source,
      &row.bank_service));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_memory_report_row_populate_source_interval(context, source,
                                                                &row));
  return loom_low_lower_record_memory_report_row(context, &row);
}

static iree_string_view_t loom_amdgpu_fragment_memory_report_address_form(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR) {
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

static void loom_amdgpu_fragment_memory_add_runtime_packet_source_interval(
    loom_value_facts_t runtime_packet_offset,
    loom_low_lower_memory_report_row_t* row) {
  const loom_low_byte_interval_precision_flags_t required_precision =
      LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE |
      LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE;
  if (!iree_all_bits_set(row->source_interval.precision_flags,
                         required_precision) ||
      (runtime_packet_offset.range_lo == 0 &&
       runtime_packet_offset.range_hi == 0)) {
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

static iree_status_t loom_amdgpu_record_fragment_memory_access_report(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_amdgpu_fragment_memory_issued_access_t* access,
    uint64_t execution_count_plus_one) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                            access->descriptor_ref);
  const loom_low_descriptor_memory_effect_summary_t issued =
      loom_low_descriptor_memory_effect_summary(descriptor_set, descriptor);
  const iree_string_view_t packet_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_ref);
  int64_t static_offset_bytes = plan->source.static_byte_offset;
  const bool has_static_offset = loom_amdgpu_fragment_memory_static_offset_i64(
      plan, access->register_index, access->element_index,
      &static_offset_bytes);
  loom_low_source_memory_access_plan_t packet_source = plan->source;
  packet_source.static_byte_offset = static_offset_bytes;
  packet_source.element_byte_count = plan->element_byte_count;
  packet_source.vector_lane_count = access->element_count;
  packet_source.vector_lane_byte_stride = plan->element_byte_count;
  const loom_amdgpu_fragment_memory_packet_offset_t runtime_packet_offset =
      loom_amdgpu_fragment_memory_runtime_packet_offset(
          plan, access->register_index, access->element_index);
  loom_amdgpu_fragment_memory_packet_report_t packet_report = {0};
  loom_amdgpu_fragment_memory_query_packet_report(plan, packet, &packet_report);
  loom_low_lower_memory_subgroup_access_report_t subgroup_access = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_report_subgroup_access(
      context, source_op, layout, plan, &runtime_packet_offset, &issued,
      access->flags, &subgroup_access));
  loom_low_lower_memory_report_row_t row = {
      .function_name = loom_low_lower_context_function_name(context),
      .source_op_name =
          loom_op_name(loom_low_lower_context_module(context), source_op),
      .source_op_kind = source_op->kind,
      .execution_count_plus_one = execution_count_plus_one,
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
      .vector_lane_count = access->element_count,
      .issued_read_byte_count = issued.read_byte_count,
      .issued_write_byte_count = issued.write_byte_count,
      .issued_read_unknown_width_count = issued.read_unknown_width_count,
      .issued_write_unknown_width_count = issued.write_unknown_width_count,
      .dynamic_stride_bytes = runtime_packet_offset.is_subgroup_uniform
                                  ? plan->address_layout.linear_lane_byte_stride
                                  : 0,
      .vector_lane_stride_bytes = plan->element_byte_count,
      .subgroup_access = subgroup_access,
  };
  loom_amdgpu_memory_report_row_populate_storage_schema(context, &plan->source,
                                                        &row);
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_report_bank_service(
      context, source_op, descriptor, layout, plan, access->register_index,
      access->element_index, &runtime_packet_offset, &row.bank_service));
  if (has_static_offset) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_memory_report_row_populate_source_interval(
            context, &packet_source, &row));
  }
  loom_amdgpu_fragment_memory_add_runtime_packet_source_interval(
      runtime_packet_offset.byte_facts, &row);
  return loom_low_lower_record_memory_report_row(context, &row);
}

iree_status_t loom_amdgpu_report_memory_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan,
    uint64_t execution_count_plus_one) {
  const iree_string_view_t strategy_key =
      iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_MEMORY_ACCESS_PLAN_FLAG_DESCRIPTOR_ZERO_FILL)
          ? IREE_SV("descriptor_zero_fill")
          : iree_string_view_empty();
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < plan->packet_count && iree_status_is_ok(status);
       ++i) {
    status = loom_amdgpu_record_memory_packet_report(
        context, source_op, &plan->packets[i], strategy_key,
        execution_count_plus_one);
  }
  return status;
}

iree_status_t loom_amdgpu_report_fragment_memory_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    uint64_t execution_count_plus_one) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < plan->packet_count && iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet = &plan->packets[i];
    loom_amdgpu_fragment_memory_issued_access_t
        accesses[LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_ISSUED_ACCESSES_PER_PACKET];
    const uint16_t access_count =
        loom_amdgpu_fragment_memory_query_issued_accesses(plan, packet,
                                                          accesses);
    for (uint16_t j = 0; j < access_count && iree_status_is_ok(status); ++j) {
      status = loom_amdgpu_record_fragment_memory_access_report(
          context, source_op, layout, plan, packet, &accesses[j],
          execution_count_plus_one);
    }
  }
  return status;
}
