// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/lower_internal.h"
#include "loom/codegen/low/lower/lower_rule_descriptor.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

static iree_status_t
loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor) {
  *out_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  IREE_ASSERT_NE(descriptor_ref, LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE);
  const loom_low_lower_rule_match_context_t match_context = {
      .descriptor_set = loom_low_lower_context_descriptor_set(context),
      .descriptor_ref =
          {
              .fn = loom_low_lower_rule_match_descriptor_ref_from_lowering,
              .user_data = context,
          },
  };
  const loom_low_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
      &match_context, rule_set, descriptor_ref, &descriptor));
  IREE_ASSERT(descriptor != NULL,
              "generated source-memory materializer references a "
              "missing descriptor");
  out_descriptor->descriptor = descriptor;
  return iree_ok_status();
}

static iree_status_t
loom_low_lower_rule_source_memory_emit_resolved_integer_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    iree_string_view_t immediate_name, int64_t value, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_string_id_t immediate_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                immediate_name, &immediate_name_id));
  const loom_named_attr_t attr = {
      .name_id = immediate_name_id,
      .value = loom_attr_i64(value),
  };

  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, descriptor, loom_make_named_attr_slice(&attr, 1), result_type,
      location, &const_op));
  *out_value_id = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t
loom_low_lower_rule_source_memory_emit_dynamic_byte_offset_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_t* source_memory, int64_t value,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set,
          source_memory->byte_offset_const_i64_descriptor_ref, &descriptor));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
      context, descriptor.descriptor, 0, &result_type));
  return loom_low_lower_rule_source_memory_emit_resolved_integer_const(
      context, &descriptor, source_memory->byte_offset_const_i64_immediate,
      value, result_type, location, out_value_id);
}

static iree_status_t
loom_low_lower_rule_source_memory_emit_address_coordinate_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory, int64_t value,
    loom_value_id_t* out_value_id) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set,
          source_memory->address_const_coordinate_descriptor_ref, &descriptor));
  loom_scalar_type_t source_scalar_type = LOOM_SCALAR_TYPE_OFFSET;
  if (source_memory->address_coordinate_type ==
      LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_INDEX) {
    source_scalar_type = LOOM_SCALAR_TYPE_INDEX;
  } else {
    IREE_ASSERT_EQ(source_memory->address_coordinate_type,
                   LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
      context, source_op, loom_type_scalar(source_scalar_type), &result_type));
  return loom_low_lower_rule_source_memory_emit_resolved_integer_const(
      context, &descriptor, source_memory->address_const_coordinate_immediate,
      value, result_type, source_op->location, out_value_id);
}

static iree_status_t
loom_low_lower_rule_source_memory_materializer_copy_operands(
    loom_low_lower_context_t* context, loom_location_id_t location,
    uint16_t copy_operand_mask, uint16_t operand_count,
    loom_value_id_t* operands) {
  if (copy_operand_mask == 0) return iree_ok_status();
  IREE_ASSERT_LE(operand_count, 16);
  for (uint16_t i = 0; i < operand_count; ++i) {
    const uint16_t operand_bit = (uint16_t)((uint16_t)1u << i);
    if (!iree_any_bit_set(copy_operand_mask, operand_bit)) continue;
    const loom_type_t copy_type = loom_module_value_type(
        loom_low_lower_context_module(context), operands[i]);
    IREE_ASSERT(loom_low_type_is_register(copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_copy_build(loom_low_lower_context_builder(context),
                            operands[i], false, copy_type, location, &copy_op));
    operands[i] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_lower_rule_source_memory_materializer_tied_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    uint16_t operand_count, loom_value_id_t* operands,
    loom_location_id_t location, const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor_row = descriptor->descriptor;
  uint16_t copy_operand_mask = 0;
  uint16_t tied_result_count = 0;
  IREE_ASSERT((uint64_t)descriptor_row->constraint_start +
                  (uint64_t)descriptor_row->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor_row->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set
             ->constraints[descriptor_row->constraint_start + (uint32_t)i];
    switch (constraint->kind) {
      case LOOM_LOW_CONSTRAINT_KIND_TIED:
        ++tied_result_count;
        break;
      case LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE: {
        IREE_ASSERT_NE(constraint->rhs_operand_index, LOOM_LOW_ID_NONE);
        const loom_low_operand_t* packet_operand =
            &descriptor_set->operands[descriptor_row->operand_start +
                                      constraint->rhs_operand_index];
        IREE_ASSERT_TRUE(
            loom_low_operand_role_is_packet_operand(packet_operand->role));
        const uint16_t packet_operand_index =
            packet_operand->source_value_index;
        IREE_ASSERT_LT(packet_operand_index, operand_count);
        copy_operand_mask |= (uint16_t)((uint16_t)1u << packet_operand_index);
        break;
      }
      default:
        break;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_materializer_copy_operands(
          context, location, copy_operand_mask, operand_count, operands));
  if (tied_result_count == 0) return iree_ok_status();

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, tied_result_count, sizeof(*tied_results),
      (void**)&tied_results));
  uint16_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor_row->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set
             ->constraints[descriptor_row->constraint_start + (uint32_t)i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) continue;
    IREE_ASSERT_NE(constraint->rhs_operand_index, LOOM_LOW_ID_NONE);
    const loom_low_operand_t* result_operand =
        &descriptor_set->operands[descriptor_row->operand_start +
                                  constraint->lhs_operand_index];
    const loom_low_operand_t* packet_operand =
        &descriptor_set->operands[descriptor_row->operand_start +
                                  constraint->rhs_operand_index];
    IREE_ASSERT_EQ(result_operand->role, LOOM_LOW_OPERAND_ROLE_RESULT);
    IREE_ASSERT_TRUE(
        loom_low_operand_role_is_packet_operand(packet_operand->role));
    const uint16_t result_index = result_operand->source_value_index;
    const uint16_t packet_operand_index = packet_operand->source_value_index;
    IREE_ASSERT_LT(packet_operand_index, operand_count);
    tied_results[tied_result_index++] = (loom_tied_result_t){
        .result_index = result_index,
        .operand_index = packet_operand_index,
        .has_type_change = false,
    };
  }
  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_source_memory_emit_binary_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, descriptor_ref, &descriptor));

  const loom_value_id_t operands[2] = {lhs, rhs};
  const loom_type_t result_type =
      loom_module_value_type(loom_low_lower_context_module(context), lhs);
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  loom_value_id_t materializer_operands[2] = {operands[0], operands[1]};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_materializer_tied_results(
          context, &descriptor, IREE_ARRAYSIZE(materializer_operands),
          materializer_operands, location, &tied_results, &tied_result_count));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, materializer_operands,
      IREE_ARRAYSIZE(materializer_operands), loom_named_attr_slice_empty(),
      &result_type, 1, tied_results, tied_result_count, location, &low_op));
  const loom_value_slice_t results = loom_low_op_results(low_op);
  IREE_ASSERT_EQ(results.count, 1u);
  *out_value_id = results.values[0];
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_source_memory_emit_unary_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref, loom_value_id_t input,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, descriptor_ref, &descriptor));

  loom_value_id_t materializer_operand = input;
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_materializer_tied_results(
          context, &descriptor, 1, &materializer_operand, location,
          &tied_results, &tied_result_count));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
      context, descriptor.descriptor, 0, &result_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, &materializer_operand, 1,
      loom_named_attr_slice_empty(), &result_type, 1, tied_results,
      tied_result_count, location, &low_op));
  const loom_value_slice_t results = loom_low_op_results(low_op);
  IREE_ASSERT_EQ(results.count, 1u);
  *out_value_id = results.values[0];
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_materialize_source_memory_term(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;

  loom_value_id_t index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, term->index, &index));
  loom_value_id_t accumulator = index;
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t stride_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, term->stride_values[i], &stride_value));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, source_memory->byte_offset_mul_i64_descriptor_ref,
        accumulator, stride_value, source_op->location, &accumulator));
  }

  if (term->byte_stride == 1) {
    *out_value_id = accumulator;
    return iree_ok_status();
  }

  if (term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE &&
      source_memory->byte_offset_shl_i64_descriptor_ref !=
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_dynamic_byte_offset_const(
            context, rule_set, source_memory, term->byte_shift,
            source_op->location, &shift));
    return loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, source_memory->byte_offset_shl_i64_descriptor_ref,
        accumulator, shift, source_op->location, out_value_id);
  }

  loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_emit_dynamic_byte_offset_const(
          context, rule_set, source_memory, term->byte_stride,
          source_op->location, &stride));
  return loom_low_lower_rule_source_memory_emit_binary_op(
      context, rule_set, source_memory->byte_offset_mul_i64_descriptor_ref,
      accumulator, stride, source_op->location, out_value_id);
}

iree_status_t loom_low_lower_rule_materialize_source_memory_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(source_memory_access != NULL);
  IREE_ASSERT_GT(source_memory_access->dynamic_term_count, 0);

  if (loom_low_source_memory_access_dynamic_offset_has_materialized_view_base(
          source_memory_access) &&
      loom_low_lower_source_value_has_low_mapping(
          context, source_memory_access->dynamic_view_base_value_id)) {
    return loom_low_lower_lookup_value(
        context, source_memory_access->dynamic_view_base_value_id,
        out_value_id);
  }

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  uint8_t term_ordinal = 0;
  uint8_t realization_ordinal = 0;
  while (term_ordinal < source_memory_access->dynamic_term_count) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source_memory_access->dynamic_terms[term_ordinal];
    uint8_t consumed_term_count = 1;
    if (realization_ordinal < source_memory_access->dynamic_realization_count &&
        source_memory_access->dynamic_realizations[realization_ordinal]
                .first_term == term_ordinal) {
      const loom_low_source_memory_dynamic_realization_t* realization =
          &source_memory_access->dynamic_realizations[realization_ordinal++];
      if (loom_low_lower_source_value_has_low_mapping(
              context, realization->term.index)) {
        term = &realization->term;
        consumed_term_count = realization->term_count;
      }
    }
    loom_value_id_t term_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_source_memory_term(
        context, rule_set, source_op, source_memory, term, &term_value));
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      accumulator = term_value;
    } else {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
          context, rule_set, source_memory->byte_offset_add_i64_descriptor_ref,
          accumulator, term_value, source_op->location, &accumulator));
    }
    term_ordinal = (uint8_t)(term_ordinal + consumed_term_count);
  }

  IREE_ASSERT_NE(accumulator, LOOM_VALUE_ID_INVALID);
  *out_value_id = accumulator;
  return iree_ok_status();
}

static iree_status_t
loom_low_lower_rule_materialize_source_memory_address_coordinate_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    loom_value_id_t source_value_id, loom_value_id_t* out_value_id) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  loom_value_id_t low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value_id, &low_value_id));
  if (source_memory->address_coordinate_type ==
      LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_INDEX) {
    IREE_ASSERT(
        loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)));
    *out_value_id = low_value_id;
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(source_memory->address_coordinate_type,
                 LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET);
  if (loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET))) {
    *out_value_id = low_value_id;
    return iree_ok_status();
  }

  IREE_ASSERT(
      loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)));
  if (source_memory->address_index_to_coordinate_input_descriptor_ref !=
      LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_unary_op(
        context, rule_set,
        source_memory->address_index_to_coordinate_input_descriptor_ref,
        low_value_id, source_op->location, &low_value_id));
  }
  return loom_low_lower_rule_source_memory_emit_unary_op(
      context, rule_set,
      source_memory->address_index_to_coordinate_descriptor_ref, low_value_id,
      source_op->location, out_value_id);
}

static iree_status_t loom_low_lower_rule_materialize_source_memory_address_term(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t* out_value_id) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_materialize_source_memory_address_coordinate_value(
          context, rule_set, source_op, source_memory, term->index,
          &accumulator));
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t stride_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_materialize_source_memory_address_coordinate_value(
            context, rule_set, source_op, source_memory, term->stride_values[i],
            &stride_value));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, source_memory->address_mul_coordinate_descriptor_ref,
        accumulator, stride_value, source_op->location, &accumulator));
  }

  const int64_t unit_byte_count =
      source_memory->address_coordinate_unit_byte_count;
  IREE_ASSERT_GT(unit_byte_count, 0);
  IREE_ASSERT_EQ(term->byte_stride % unit_byte_count, 0);
  const int64_t coordinate_stride = term->byte_stride / unit_byte_count;
  if (coordinate_stride == 1) {
    *out_value_id = accumulator;
    return iree_ok_status();
  }

  if (coordinate_stride > 0 &&
      iree_math_is_power_of_two_i64(coordinate_stride) &&
      source_memory->address_shl_coordinate_descriptor_ref !=
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_address_coordinate_const(
            context, rule_set, source_op, source_memory,
            iree_math_count_trailing_zeros_u64((uint64_t)coordinate_stride),
            &shift));
    return loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, source_memory->address_shl_coordinate_descriptor_ref,
        accumulator, shift, source_op->location, out_value_id);
  }

  loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_emit_address_coordinate_const(
          context, rule_set, source_op, source_memory, coordinate_stride,
          &stride));
  return loom_low_lower_rule_source_memory_emit_binary_op(
      context, rule_set, source_memory->address_mul_coordinate_descriptor_ref,
      accumulator, stride, source_op->location, out_value_id);
}

static iree_status_t
loom_low_lower_rule_materialize_source_memory_complete_coordinate(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  uint8_t first_canonical_term = 0;
  int64_t static_byte_offset = source_memory_access->static_byte_offset;
  if (source_memory->address_coordinate_unit_byte_count == 1 &&
      source_memory->address_coordinate_type ==
          LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET &&
      source_memory_access->dynamic_view_base_term_count != 0 &&
      source_memory_access->dynamic_view_base_value_id !=
          LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, source_memory_access->dynamic_view_base_value_id,
        &accumulator));
    first_canonical_term = source_memory_access->dynamic_view_base_term_count;
    const bool subtracted_view_base = iree_checked_sub_i64(
        static_byte_offset,
        source_memory_access->dynamic_view_base_value_static_byte_offset,
        &static_byte_offset);
    IREE_ASSERT(subtracted_view_base);
    (void)subtracted_view_base;
  }

  for (uint8_t term_ordinal = first_canonical_term;
       term_ordinal < source_memory_access->dynamic_term_count;
       ++term_ordinal) {
    loom_value_id_t term_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_materialize_source_memory_address_term(
            context, rule_set, source_op, source_memory,
            &source_memory_access->dynamic_terms[term_ordinal], &term_value));
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      accumulator = term_value;
    } else {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
          context, rule_set,
          source_memory->address_add_coordinate_descriptor_ref, accumulator,
          term_value, source_op->location, &accumulator));
    }
  }

  const int64_t unit_byte_count =
      source_memory->address_coordinate_unit_byte_count;
  IREE_ASSERT_GT(unit_byte_count, 0);
  IREE_ASSERT_EQ(static_byte_offset % unit_byte_count, 0);
  const int64_t static_coordinate = static_byte_offset / unit_byte_count;
  if (static_coordinate != 0 || accumulator == LOOM_VALUE_ID_INVALID) {
    loom_value_id_t static_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_address_coordinate_const(
            context, rule_set, source_op, source_memory, static_coordinate,
            &static_value));
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      accumulator = static_value;
    } else {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
          context, rule_set,
          source_memory->address_add_coordinate_descriptor_ref, accumulator,
          static_value, source_op->location, &accumulator));
    }
  }

  *out_value_id = accumulator;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_materialize_source_memory_address(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id) {
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t source_base =
      source_memory->address_base_kind ==
              LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_BASE_VIEW
          ? loom_low_source_memory_access_base_view_value_id(
                source_memory_access)
          : source_memory_access->root_value_id;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_base, &base));
  loom_value_id_t coordinate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_materialize_source_memory_complete_coordinate(
          context, rule_set, source_op, source_memory, source_memory_access,
          &coordinate));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, source_memory->address_descriptor_ref,
          &descriptor));
  loom_value_id_t operands[2] = {base, coordinate};
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_materializer_tied_results(
          context, &descriptor, IREE_ARRAYSIZE(operands), operands,
          source_op->location, &tied_results, &tied_result_count));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
      context, descriptor.descriptor, 0, &result_type));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, tied_results,
      tied_result_count, source_op->location, &address_op));
  const loom_value_slice_t results = loom_low_op_results(address_op);
  IREE_ASSERT_EQ(results.count, 1u);
  *out_value_id = results.values[0];
  return iree_ok_status();
}
