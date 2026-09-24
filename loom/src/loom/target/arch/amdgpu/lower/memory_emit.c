// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/encoding/operand.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_bank_service.h"
#include "loom/target/arch/amdgpu/lower/memory_coherence.h"
#include "loom/target/arch/amdgpu/lower/memory_ordering.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
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

void loom_amdgpu_memory_report_row_populate_storage_schema(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source,
    loom_low_lower_memory_report_row_t* row) {
  if (!loom_low_lower_context_wants_report_rows(context) ||
      source->view_value_id == LOOM_VALUE_ID_INVALID) {
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
    const loom_amdgpu_memory_packet_plan_t* packet) {
  const loom_low_source_memory_access_plan_t* source = &packet->access.source;
  if (!loom_low_lower_context_wants_report_rows(context)) {
    return iree_ok_status();
  }

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
      .source_root_name = loom_module_value_name(
          loom_low_lower_context_module(context), source->root_value_id),
      .source_root_argument_index =
          loom_low_lower_source_memory_root_argument_index(context, source),
      .memory_space = loom_amdgpu_memory_space_name(source->memory_space),
      .operation_kind = loom_amdgpu_memory_operation_name(operation_kind),
      .packet_key = packet_key,
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
  return loom_low_lower_record_memory_report_row(context, source_op, &row);
}

static iree_status_t loom_amdgpu_emit_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_op_t** out_op) {
  IREE_ASSERT(packet->access.descriptor != NULL);
  *out_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      loom_low_lower_context_builder(context),
      loom_low_lower_context_descriptor_set(context), packet->access.descriptor,
      packet->access.source.access_flags, operands, operand_count, attrs,
      result_types, result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, out_op));
  return loom_amdgpu_record_memory_packet_report(context, source_op, packet);
}

static iree_status_t loom_amdgpu_memory_payload_low_type(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_type_t* out_type) {
  if (access->payload_register_class ==
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_SGPR) {
    if (access->payload_register_count == 1) {
      return loom_amdgpu_make_sgpr_type(context, out_type);
    }
    return loom_amdgpu_make_sgpr_range_type(
        context, access->payload_register_count, out_type);
  }
  if (access->payload_register_count == 1) {
    return loom_amdgpu_make_vgpr_type(context, out_type);
  }
  return loom_amdgpu_make_vgpr_range_type(
      context, access->payload_register_count, out_type);
}

static iree_status_t loom_amdgpu_try_emit_exact_vgpr_store_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t source_register_offset,
    uint32_t payload_register_count, loom_value_id_t* out_low_value,
    bool* out_emitted) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL || payload_register_count == 0 ||
      payload_register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_ok_status();
  }

  uint32_t bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < payload_register_count; ++i) {
    if (source_register_offset > UINT32_MAX - i ||
        !loom_amdgpu_source_lane_as_u32_bits(fact_table, module, source_value,
                                             source_register_offset + i,
                                             &bit_patterns[i])) {
      return iree_ok_status();
    }
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < payload_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        bit_patterns[i], vgpr_type, &low_lanes[i]));
  }
  if (payload_register_count == 1) {
    *out_low_value = low_lanes[0];
    *out_emitted = true;
    return iree_ok_status();
  }

  loom_type_t vgpr_range_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, payload_register_count, &vgpr_range_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context), low_lanes,
                            payload_register_count, vgpr_range_type,
                            source_op->location, &concat_op));
  *out_low_value = loom_low_concat_result(concat_op);
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_ensure_memory_store_payload_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t source_value,
    uint32_t source_register_offset, loom_value_id_t low_value,
    loom_value_id_t* out_low_value) {
  *out_low_value = low_value;
  if (access->payload_register_class !=
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_VGPR) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    if (access->payload_register_count == 1 && access->packet_byte_count == 2) {
      return loom_amdgpu_materialize_full_low_vgpr_b32(
          context, source_op, low_value, out_low_value);
    }
    return iree_ok_status();
  }
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr && loom_low_register_type_unit_count(low_type) ==
                     access->payload_register_count) {
    bool emitted_exact = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_exact_vgpr_store_payload(
        context, source_op, source_value, source_register_offset,
        access->payload_register_count, out_low_value, &emitted_exact));
    if (emitted_exact) {
      return iree_ok_status();
    }
    return loom_amdgpu_materialize_low_vgpr_b32_registers(
        context, source_op, low_value, out_low_value);
  }
  return iree_ok_status();
}

typedef struct loom_amdgpu_hal_buffer_descriptor_extent_t {
  // Static descriptor range word used when dynamic_extent is absent.
  int64_t static_extent;
  // Optional SGPR descriptor range word for dynamically sized views.
  loom_value_id_t dynamic_extent;
} loom_amdgpu_hal_buffer_descriptor_extent_t;

static bool loom_amdgpu_source_access_view_range_facts(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_access,
    int64_t* out_static_base, loom_value_facts_t* out_range_facts) {
  *out_static_base = 0;
  *out_range_facts = loom_value_facts_unknown();
  if (source_access == NULL) {
    return false;
  }

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (source_access->view_value_id >= module->values.count) {
    return false;
  }

  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table,
                                       source_access->view_value_id),
          &view_reference)) {
    return false;
  }

  if (!loom_value_facts_fit_unsigned_bit_count(view_reference.base_byte_offset,
                                               32)) {
    return false;
  }
  const bool has_dynamic_view_base =
      source_access->dynamic_view_base_term_count != 0;
  int64_t static_base = 0;
  if (!has_dynamic_view_base &&
      !loom_value_facts_as_exact_i64(view_reference.base_byte_offset,
                                     &static_base)) {
    return false;
  }
  loom_value_facts_t range_facts = view_reference.base_byte_offset;
  loom_value_facts_addi(&range_facts, &view_reference.footprint_byte_length,
                        &range_facts);
  if (!loom_value_facts_fit_unsigned_bit_count(range_facts, 32)) {
    return false;
  }

  if (has_dynamic_view_base) {
    *out_static_base =
        source_access->dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID
            ? 0
            : source_access->static_view_base_byte_offset;
  } else {
    *out_static_base = static_base;
  }
  *out_range_facts = range_facts;
  return true;
}

static bool loom_amdgpu_source_access_view_has_dense_layout(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_access) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL || source_access == NULL) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (source_access->view_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t view_type =
      loom_module_value_type(module, source_access->view_value_id);
  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  return loom_encoding_query_type_address_layout(
             &fact_table->context, module, view_type, stride_storage,
             IREE_ARRAYSIZE(stride_storage), &layout) &&
         layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE;
}

static iree_status_t
loom_amdgpu_source_access_dynamic_view_base_term_can_emit_u32(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_dynamic_term_t* term, bool* out_can_emit) {
  *out_can_emit = false;
  if (term->byte_stride < 0 || term->byte_stride > UINT32_MAX ||
      !loom_value_facts_fit_unsigned_bit_count(term->byte_facts, 32)) {
    return iree_ok_status();
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, term->index, &low_index));
  const bool is_sgpr_b32 = loom_amdgpu_low_value_is_register_class_count(
      context, low_index, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  if (!is_sgpr_b32) {
    return iree_ok_status();
  }

  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t low_stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->stride_values[i], &low_stride));
    const bool stride_is_sgpr_b32 =
        loom_amdgpu_low_value_is_register_class_count(
            context, low_stride, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
    if (!stride_is_sgpr_b32) {
      return iree_ok_status();
    }
  }

  *out_can_emit = true;
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_source_access_dynamic_view_base_value_can_emit_u32(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_access,
    bool* out_can_emit) {
  *out_can_emit = false;
  if (source_access->dynamic_view_base_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_value_id_t low_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, source_access->dynamic_view_base_value_id, &low_base));
  *out_can_emit = loom_amdgpu_low_value_is_register_class_count(
      context, low_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_source_access_dynamic_view_base_can_emit_u32(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* source_access,
    bool* out_can_emit) {
  *out_can_emit = false;
  if (source_access->dynamic_view_base_term_count == 0) {
    *out_can_emit = true;
    return iree_ok_status();
  }
  IREE_ASSERT_LE(source_access->dynamic_view_base_term_count,
                 source_access->dynamic_term_count);
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_access_dynamic_view_base_value_can_emit_u32(
          context, source_access, out_can_emit));
  if (*out_can_emit) {
    return iree_ok_status();
  }

  for (uint8_t i = 0; i < source_access->dynamic_view_base_term_count; ++i) {
    bool can_emit = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_source_access_dynamic_view_base_term_can_emit_u32(
            context, &source_access->dynamic_terms[i], &can_emit));
    if (!can_emit) {
      return iree_ok_status();
    }
  }
  *out_can_emit = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_source_access_dynamic_view_base_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source_access,
    loom_type_t sgpr_type, loom_value_id_t* out_low_base, bool* out_emitted) {
  *out_low_base = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;
  if (source_access->dynamic_view_base_term_count == 0) {
    return iree_ok_status();
  }
  bool can_emit = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_access_dynamic_view_base_can_emit_u32(
      context, source_access, &can_emit));
  if (!can_emit) {
    return iree_ok_status();
  }

  bool value_can_emit = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_access_dynamic_view_base_value_can_emit_u32(
          context, source_access, &value_can_emit));
  if (value_can_emit) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, source_access->dynamic_view_base_value_id, out_low_base));
    *out_emitted = true;
    return iree_ok_status();
  }

  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < source_access->dynamic_view_base_term_count; ++i) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_term(
        context, source_op, &source_access->dynamic_terms[i], &low_term));
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      low_accumulator = low_term;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        low_accumulator, low_term, sgpr_type, &low_accumulator));
  }

  *out_low_base = low_accumulator;
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_source_access_dense_dynamic_range(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source_access,
    int64_t static_view_base, loom_value_id_t* out_dynamic_extent,
    bool* out_emitted) {
  *out_dynamic_extent = LOOM_VALUE_ID_INVALID;
  *out_emitted = false;
  if (!loom_amdgpu_source_access_view_has_dense_layout(context,
                                                       source_access) ||
      static_view_base < 0 || static_view_base > UINT32_MAX) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t view_type =
      loom_module_value_type(module, source_access->view_value_id);
  if (!loom_type_is_view(view_type)) {
    return iree_ok_status();
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (element_bit_count <= 0 || (element_bit_count % 8) != 0) {
    return iree_ok_status();
  }

  int64_t static_scale = element_bit_count / 8;
  loom_value_id_t low_extent = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  bool can_emit_dynamic_base = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_access_dynamic_view_base_can_emit_u32(
      context, source_access, &can_emit_dynamic_base));
  if (!can_emit_dynamic_base) {
    return iree_ok_status();
  }

  const uint8_t rank = loom_type_rank(view_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (!loom_type_dim_is_dynamic_at(view_type, axis)) {
      const int64_t dim_size = loom_type_dim_static_size_at(view_type, axis);
      if (dim_size < 0 ||
          !iree_checked_mul_i64(static_scale, dim_size, &static_scale) ||
          static_scale > UINT32_MAX) {
        return iree_ok_status();
      }
      continue;
    }

    loom_value_id_t low_dim = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_type_dim_value_id_at(view_type, axis), &low_dim));
    const bool is_sgpr_b32 = loom_amdgpu_low_value_is_register_class_count(
        context, low_dim, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
    if (!is_sgpr_b32) {
      return iree_ok_status();
    }
    if (low_extent == LOOM_VALUE_ID_INVALID) {
      low_extent = low_dim;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32, low_extent,
        low_dim, sgpr_type, &low_extent));
  }

  if (low_extent == LOOM_VALUE_ID_INVALID &&
      source_access->dynamic_view_base_term_count == 0) {
    return iree_ok_status();
  }
  if (static_scale < 0 || static_scale > UINT32_MAX) {
    return iree_ok_status();
  }
  if (low_extent == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        (uint32_t)static_scale, sgpr_type, &low_extent));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_scale_u32(
        context, source_op, low_extent, (uint32_t)static_scale, sgpr_type,
        &low_extent));
  }

  loom_value_id_t low_dynamic_base = LOOM_VALUE_ID_INVALID;
  bool emitted_dynamic_base = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_source_access_dynamic_view_base_u32(
      context, source_op, source_access, sgpr_type, &low_dynamic_base,
      &emitted_dynamic_base));
  if (emitted_dynamic_base) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        low_dynamic_base, low_extent, sgpr_type, &low_extent));
  }
  if (static_view_base != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32, low_extent,
        (uint32_t)static_view_base, sgpr_type, &low_extent));
  }

  *out_dynamic_extent = low_extent;
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_buffer_descriptor_extent_from_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source_access,
    loom_amdgpu_hal_buffer_descriptor_extent_t* inout_extent) {
  int64_t static_base = 0;
  loom_value_facts_t range_facts = loom_value_facts_unknown();
  if (!loom_amdgpu_source_access_view_range_facts(context, source_access,
                                                  &static_base, &range_facts)) {
    return iree_ok_status();
  }

  int64_t exact_range = 0;
  if (loom_value_facts_as_exact_i64(range_facts, &exact_range)) {
    IREE_ASSERT_GE(exact_range, 0);
    IREE_ASSERT_LE((uint64_t)exact_range, (uint64_t)UINT32_MAX);
    inout_extent->static_extent = exact_range;
    return iree_ok_status();
  }

  bool emitted = false;
  loom_value_id_t dynamic_extent = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_source_access_dense_dynamic_range(
      context, source_op, source_access, static_base, &dynamic_extent,
      &emitted));
  if (emitted) {
    inout_extent->dynamic_extent = dynamic_extent;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_hal_buffer_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_binding,
    const loom_low_source_memory_access_plan_t* source_access,
    loom_value_id_t* out_low_descriptor) {
  *out_low_descriptor = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_hal_buffer_descriptor_extent_t extent = {
      .static_extent = UINT32_MAX,
      .dynamic_extent = LOOM_VALUE_ID_INVALID,
  };
  bool resource_has_extent = false;
  int64_t cache_swizzle_stride = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* binding = loom_module_value(module, low_binding);
  const loom_op_t* binding_op = loom_value_def_op(binding);
  if (binding_op != NULL && loom_low_resource_isa(binding_op)) {
    if (loom_low_resource_extent_value_is_present(binding_op)) {
      extent.dynamic_extent = loom_low_resource_extent_value(binding_op);
      resource_has_extent = true;
    } else {
      if (loom_low_resource_has_extent(binding_op)) {
        const int64_t resource_extent = loom_low_resource_extent(binding_op);
        if (resource_extent <= UINT32_MAX) {
          extent.static_extent = resource_extent;
        }
        resource_has_extent = true;
      }
    }

    if (loom_low_resource_has_cache_swizzle_stride(binding_op)) {
      cache_swizzle_stride = loom_low_resource_cache_swizzle_stride(binding_op);
    }
  }
  if (!resource_has_extent) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_buffer_descriptor_extent_from_source(
        context, source_op, source_access, &extent));
  }

  loom_type_t descriptor_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 4, &descriptor_type));
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
      context, IREE_SV("cache_swizzle_stride"), cache_swizzle_stride, attrs,
      IREE_ARRAYSIZE(attrs), &attr_count));
  loom_value_id_t operands[2] = {low_binding, extent.dynamic_extent};
  uint16_t operand_count = 1;
  uint16_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR;
  if (extent.dynamic_extent != LOOM_VALUE_ID_INVALID) {
    descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_HAL_BUFFER_DESCRIPTOR_EXTENT;
    operand_count = 2;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("extent"), extent.static_extent, attrs,
        IREE_ARRAYSIZE(attrs), &attr_count));
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &descriptor_type, 1,
      &low_op));
  *out_low_descriptor = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static bool loom_amdgpu_memory_packet_operand_matches_field(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, iree_string_view_t field_name) {
  if (!loom_low_operand_role_is_packet_operand(operand->role)) {
    return false;
  }
  const iree_string_view_t operand_field_name = loom_low_descriptor_set_string(
      descriptor_set, operand->field_name_string_ref);
  return iree_string_view_equal(operand_field_name, field_name);
}

static uint32_t loom_amdgpu_memory_packet_addr_operand_unit_count(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_packet_plan_t* packet) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor = packet->access.descriptor;
  for (uint16_t descriptor_operand_index = descriptor->result_count;
       descriptor_operand_index < descriptor->operand_count;
       ++descriptor_operand_index) {
    const uint32_t operand_row =
        descriptor->operand_start + descriptor_operand_index;
    IREE_ASSERT_LT(operand_row, descriptor_set->operand_count);
    const loom_low_operand_t* operand = &descriptor_set->operands[operand_row];
    if (loom_amdgpu_memory_packet_operand_matches_field(descriptor_set, operand,
                                                        IREE_SV("addr"))) {
      return operand->unit_count;
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "AMDGPU flat memory packet selected without an addr operand");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_fit_memory_flat_vaddr_to_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet, loom_value_id_t low_vaddr,
    loom_value_id_t* out_low_vaddr) {
  *out_low_vaddr = low_vaddr;
  const uint32_t required_unit_count =
      loom_amdgpu_memory_packet_addr_operand_unit_count(context, packet);

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_vaddr_type = loom_module_value_type(module, low_vaddr);
  IREE_ASSERT(loom_low_type_is_register(low_vaddr_type));
  const uint32_t actual_unit_count =
      loom_low_register_type_unit_count(low_vaddr_type);
  if (actual_unit_count == required_unit_count) {
    return iree_ok_status();
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  if (actual_unit_count == 2 && required_unit_count == 1) {
    return loom_amdgpu_emit_low_slice(context, source_op, low_vaddr,
                                      /*offset=*/0, vgpr_type, out_low_vaddr);
  }
  if (actual_unit_count == 1 && required_unit_count == 2) {
    return loom_amdgpu_emit_vgpr64_from_u32(context, source_op, low_vaddr,
                                            out_low_vaddr);
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU flat memory packet selected unsupported vaddr width");
  IREE_BUILTIN_UNREACHABLE();
}

typedef struct loom_amdgpu_memory_cache_attr_field_t {
  // Presence bit required for this descriptor attribute.
  loom_amdgpu_memory_cache_policy_attr_flags_t flag;
  // Descriptor attribute name.
  iree_string_view_t name;
  // Byte offset to the encoded attribute value field.
  iree_host_size_t value_offset;
} loom_amdgpu_memory_cache_attr_field_t;

static const loom_amdgpu_memory_cache_attr_field_t
    kLoomAmdgpuMemoryCacheAttrFields[] = {
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE,
            .name = {.data = "scope", .size = 5},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, scope),
        },
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH,
            .name = {.data = "th", .size = 2},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, th),
        },
        {
            .flag = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT,
            .name = {.data = "nt", .size = 2},
            .value_offset =
                offsetof(loom_amdgpu_memory_cache_policy_attrs_t, nt),
        },
};

static int64_t loom_amdgpu_memory_cache_attr_field_value(
    const loom_amdgpu_memory_cache_policy_attrs_t* cache_attrs,
    const loom_amdgpu_memory_cache_attr_field_t* field) {
  const uint8_t* attrs_bytes = (const uint8_t*)cache_attrs;
  const void* value_bytes = attrs_bytes + field->value_offset;
  return *(const int64_t*)value_bytes;
}

static iree_status_t loom_amdgpu_append_memory_cache_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  if (access->source.operation_kind ==
      LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_STORE) {
    return loom_amdgpu_system_memory_append_release_store_attrs_scoped(
        loom_low_lower_context_builder(context),
        loom_low_lower_context_descriptor_set(context),
        loom_amdgpu_memory_coherence_scope(access->source.atomic.scope), attrs,
        attr_capacity, inout_attr_count);
  }
  uint8_t read_scope = access->source.read_visibility_scope;
  if (access->source.operation_kind ==
      LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_LOAD) {
    read_scope = iree_max(read_scope, access->source.atomic.scope);
  }
  if (read_scope != LOOM_ATOMIC_SCOPE_THREAD) {
    // The retained visibility obligation and atomic observation scope both
    // constrain these reads. Advisory cache preferences cannot weaken them.
    return loom_amdgpu_system_memory_append_load_attrs_scoped(
        loom_low_lower_context_builder(context),
        loom_low_lower_context_descriptor_set(context),
        loom_amdgpu_memory_coherence_scope(read_scope), attrs, attr_capacity,
        inout_attr_count);
  }
  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  if (!loom_amdgpu_memory_cache_policy_is_present(policy)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_memory_cache_policy_attrs_t cache_attrs = {0};
  const loom_amdgpu_memory_cache_policy_resolution_t resolution =
      loom_amdgpu_memory_cache_policy_resolve(descriptor_set, access,
                                              &cache_attrs);
  IREE_ASSERT_NE(resolution,
                 LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_REJECTED);
  if (resolution != LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_ENCODED) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuMemoryCacheAttrFields); ++i) {
    const loom_amdgpu_memory_cache_attr_field_t* field =
        &kLoomAmdgpuMemoryCacheAttrFields[i];
    if (!iree_any_bit_set(cache_attrs.flags, field->flag)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, field->name,
        loom_amdgpu_memory_cache_attr_field_value(&cache_attrs, field), attrs,
        attr_capacity, inout_attr_count));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_make_memory_cache_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  return loom_amdgpu_append_memory_cache_attrs(context, access, attrs,
                                               attr_capacity, out_attr_count);
}

iree_status_t loom_amdgpu_make_memory_attrs(
    loom_low_lower_context_t* context,
    const loom_amdgpu_memory_access_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset0"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset1"), access->secondary_immediate_offset, attrs,
        attr_capacity, out_attr_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_i64_attr(
        context, IREE_SV("offset"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
  }
  return loom_amdgpu_append_memory_cache_attrs(context, access, attrs,
                                               attr_capacity, out_attr_count);
}

static loom_value_id_t loom_amdgpu_memory_load_result(
    const loom_op_t* source_op) {
  IREE_ASSERT_EQ(source_op->result_count, 1u);
  return loom_op_const_results(source_op)[0];
}

static iree_status_t loom_amdgpu_memory_load_result_is_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_is_vgpr) {
  *out_is_vgpr = false;
  const loom_value_id_t source_result =
      loom_amdgpu_memory_load_result(source_op);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  *out_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_materialize_memory_load_packet_for_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool result_is_vgpr, loom_value_id_t low_packet,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = low_packet;
  if (!result_is_vgpr) {
    return iree_ok_status();
  }
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_packet, out_low_packet);
}

static bool loom_amdgpu_memory_load_packet_needs_signed_i16_repair(
    const loom_amdgpu_memory_access_t* access) {
  return access->source.memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         access->payload_format ==
             LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_SIGNED_16BIT_INTEGER;
}

static iree_status_t loom_amdgpu_emit_signed_i16_memory_load_repair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_packet, loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), 0, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("width"), 16, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));

  const loom_value_id_t operands[] = {low_packet};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_I32_OFFSET_WIDTH_INLINE, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &lane_type, 1, &low_op));
  *out_low_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_repair_memory_load_packet_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t low_packet,
    loom_value_id_t* out_low_result) {
  if (!loom_amdgpu_memory_load_packet_needs_signed_i16_repair(access)) {
    *out_low_result = low_packet;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_signed_i16_memory_load_repair(
      context, source_op, low_packet, out_low_result);
}

static loom_value_id_t loom_amdgpu_memory_store_value(
    const loom_module_t* module, const loom_op_t* source_op) {
  const loom_memory_access_t access =
      loom_memory_access_cast(module, source_op);
  IREE_ASSERT(loom_memory_access_isa(access));
  const loom_value_id_t value = loom_memory_access_value(access);
  IREE_ASSERT_NE(value, LOOM_VALUE_ID_INVALID);
  return value;
}

static iree_status_t loom_amdgpu_bind_memory_load_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_result) {
  const loom_value_id_t source_result =
      loom_amdgpu_memory_load_result(source_op);
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_load_result_is_vgpr(
      context, source_op, &result_is_vgpr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_memory_load_packet_for_result(
      context, source_op, result_is_vgpr, low_result, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static bool loom_amdgpu_memory_access_needs_hal_resource(
    const loom_amdgpu_memory_access_t* access) {
  return access->source.memory_space !=
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
}

static iree_status_t loom_amdgpu_lower_memory_packet_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_memory_access_t* access = &packet->access;
  loom_amdgpu_memory_dynamic_term_sequence_t sequence = {0};
  loom_amdgpu_memory_access_resolve_dynamic_terms(context, access, &sequence);
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_memory_access_needs_hal_resource(access)) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context,
        loom_low_source_memory_access_base_view_value_id(&access->source),
        &low_resource));
  }

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
          context, source_op, access, &sequence, low_resource, &low_vaddr));
      IREE_RETURN_IF_ERROR(loom_amdgpu_fit_memory_flat_vaddr_to_packet(
          context, source_op, packet, low_vaddr, &low_vaddr));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_memory_vaddr(context, source_op, access, &sequence,
                                        LOOM_VALUE_ID_INVALID, &low_vaddr));
    }
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_memory_payload_low_type(context, access, &result_type));

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      const loom_low_lower_resolved_descriptor_t packet_descriptor = {
          .descriptor = access->descriptor,
      };
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_m0_u32(
          context, source_op, &packet_descriptor, 0, &low_m0));
      loom_value_id_t operands[] = {low_m0};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
          &low_op));
      const loom_value_id_t raw_result =
          loom_value_slice_get(loom_low_op_results(low_op), 0);
      return loom_amdgpu_repair_memory_load_packet_result(
          context, source_op, access, raw_result, out_low_result);
    }
    loom_value_id_t operands[] = {low_vaddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, access, &sequence, low_resource, &low_saddr));
    const loom_value_id_t operands[] = {low_vaddr, low_saddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    loom_value_id_t low_sbase = low_resource;
    loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
    if (access->scalar_offset_placement ==
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
          context, source_op, access, &sequence, low_resource, &low_sbase));
      loom_type_t sgpr_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
          sgpr_type, &low_soffset));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_terms(
          context, source_op, &sequence, access->scalar_byte_offset,
          &low_soffset));
    }
    loom_value_id_t operands[] = {
        low_sbase,
        low_soffset,
    };
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
        context, source_op, low_resource, &access->source, &low_descriptor));
    loom_value_id_t operands[] = {low_descriptor};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT ||
      access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR) {
    const loom_value_id_t operands[] = {low_vaddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
        &low_op));
    const loom_value_id_t raw_result =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    return loom_amdgpu_repair_memory_load_packet_result(
        context, source_op, access, raw_result, out_low_result);
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_terms(
      context, source_op, &sequence, access->scalar_byte_offset, &low_soffset));
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, low_resource, &access->source, &low_descriptor));
  loom_value_id_t operands[] = {
      low_descriptor,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_packet(
      context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, &low_op));
  const loom_value_id_t raw_result =
      loom_value_slice_get(loom_low_op_results(low_op), 0);
  return loom_amdgpu_repair_memory_load_packet_result(
      context, source_op, access, raw_result, out_low_result);
}

static iree_status_t loom_amdgpu_lower_memory_packet_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_packet_plan_t* packet,
    loom_value_id_t source_value, loom_value_id_t low_value) {
  const loom_amdgpu_memory_access_t* access = &packet->access;
  loom_amdgpu_memory_dynamic_term_sequence_t sequence = {0};
  loom_amdgpu_memory_access_resolve_dynamic_terms(context, access, &sequence);
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_memory_store_payload_vgpr(
      context, source_op, access, source_value,
      packet->payload_byte_offset / 4u, low_value, &low_value));
  const uint32_t register_byte_offset = packet->payload_byte_offset % 4u;
  if (register_byte_offset != 0) {
    loom_type_t lane_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        register_byte_offset * 8u, low_value, lane_type, &low_value));
  }
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_memory_access_needs_hal_resource(access)) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context,
        loom_low_source_memory_access_base_view_value_id(&access->source),
        &low_resource));
  }

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_flat_vaddr(
          context, source_op, access, &sequence, low_resource, &low_vaddr));
      IREE_RETURN_IF_ERROR(loom_amdgpu_fit_memory_flat_vaddr_to_packet(
          context, source_op, packet, low_vaddr, &low_vaddr));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_memory_vaddr(context, source_op, access, &sequence,
                                        LOOM_VALUE_ID_INVALID, &low_vaddr));
    }
  }

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      const loom_low_lower_resolved_descriptor_t packet_descriptor = {
          .descriptor = access->descriptor,
      };
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_m0_u32(
          context, source_op, &packet_descriptor, 0, &low_m0));
      loom_value_id_t operands[] = {
          low_value,
          low_m0,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
          /*result_count=*/0, &low_op);
    }
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
      loom_type_t lane_type = loom_type_none();
      const uint32_t lane_register_count = access->payload_register_count / 2u;
      if (lane_register_count == 1u) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
            context, lane_register_count, &lane_type));
      }
      loom_value_id_t low_value0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, 0, lane_type, &low_value0));
      loom_value_id_t low_value1 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, lane_register_count, lane_type,
          &low_value1));
      loom_value_id_t operands[] = {
          low_vaddr,
          low_value0,
          low_value1,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_memory_packet(
          context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
          /*result_count=*/0, &low_op);
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
        context, source_op, low_resource, &access->source, &low_descriptor));
    loom_value_id_t operands[] = {
        low_value,
        low_descriptor,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, access, &sequence, low_resource, &low_saddr));
    const loom_value_id_t operands[] = {low_vaddr, low_value, low_saddr};
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT ||
      access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR) {
    const loom_value_id_t operands[] = {low_vaddr, low_value};
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_memory_packet(
        context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
        /*result_count=*/0, &low_op);
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset_terms(
      context, source_op, &sequence, access->scalar_byte_offset, &low_soffset));
  loom_value_id_t low_descriptor = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, low_resource, &access->source, &low_descriptor));
  loom_value_id_t operands[] = {
      low_value,
      low_descriptor,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_memory_packet(
      context, source_op, packet, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, &low_op);
}

iree_status_t loom_amdgpu_lower_memory_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan) {
  IREE_ASSERT_GT(plan->packet_count, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_ordering_prefix(
      context, source_op, &plan->packets[0].access.source));
  if (plan->packet_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_load(
        context, source_op, &plan->packets[0], &low_result));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_ordering_suffix(
        context, source_op, &plan->packets[0].access.source));
    return loom_amdgpu_bind_memory_load_result(context, source_op, low_result);
  }

  loom_value_id_t low_results[LOOM_AMDGPU_MAX_MEMORY_PACKET_COUNT];
  uint32_t low_result_count = 0;
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_load_result_is_vgpr(
      context, source_op, &result_is_vgpr));
  for (uint32_t i = 0; i < plan->packet_count; ++i) {
    const loom_amdgpu_memory_packet_plan_t* packet = &plan->packets[i];
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_load(
        context, source_op, packet, &low_result));
    const uint32_t register_byte_offset = packet->payload_byte_offset % 4u;
    if (register_byte_offset == 0) {
      low_results[low_result_count++] = low_result;
    } else {
      // The final byte completes a zero-extended halfword in the same VGPR.
      // Assemble before bank materialization so uniform results cross once.
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          register_byte_offset * 8u, low_result, lane_type, &low_result));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
          low_results[low_result_count - 1u], low_result, lane_type,
          &low_results[low_result_count - 1u]));
    }
    // Materialize complete words immediately, retaining the existing ordering
    // for full-register packets. A following partial packet finishes this word.
    if (i + 1u == plan->packet_count ||
        plan->packets[i + 1u].payload_byte_offset % 4u == 0) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_materialize_memory_load_packet_for_result(
              context, source_op, result_is_vgpr,
              low_results[low_result_count - 1u],
              &low_results[low_result_count - 1u]));
    }
  }
  if (low_result_count == 1) {
    return loom_amdgpu_bind_memory_load_result(context, source_op,
                                               low_results[0]);
  }

  const loom_value_id_t source_result =
      loom_amdgpu_memory_load_result(source_op);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_results, low_result_count,
      result_type, source_op->location, &concat_op));
  return loom_amdgpu_bind_memory_load_result(context, source_op,
                                             loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_memory_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan) {
  IREE_ASSERT_GT(plan->packet_count, 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_ordering_prefix(
      context, source_op, &plan->packets[0].access.source));
  const loom_value_id_t source_value = loom_amdgpu_memory_store_value(
      loom_low_lower_context_module(context), source_op);
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  if (plan->packet_count == 1) {
    return loom_amdgpu_lower_memory_packet_store(
        context, source_op, &plan->packets[0], source_value, low_value);
  }

  for (uint32_t i = 0; i < plan->packet_count; ++i) {
    const loom_amdgpu_memory_packet_plan_t* packet = &plan->packets[i];
    loom_type_t packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_memory_payload_low_type(
        context, &packet->access, &packet_type));
    loom_value_id_t packet_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_value, packet->payload_byte_offset / 4u,
        packet_type, &packet_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_memory_packet_store(
        context, source_op, packet, source_value, packet_value));
  }
  return iree_ok_status();
}
