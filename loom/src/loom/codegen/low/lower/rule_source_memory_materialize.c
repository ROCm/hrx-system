// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/rule_descriptor.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
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

static iree_status_t loom_low_lower_rule_source_memory_emit_byte_offset_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_byte_offset_materializer_t* materializer,
    loom_low_lower_descriptor_ref_t descriptor_ref, int64_t value,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, descriptor_ref, &descriptor));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
      context, descriptor.descriptor, 0, &result_type));
  return loom_low_lower_rule_source_memory_emit_resolved_integer_const(
      context, &descriptor,
      loom_low_lower_rule_set_string(
          rule_set, materializer->constant_immediate_string_ref),
      value, result_type, location, out_value_id);
}

static iree_status_t
loom_low_lower_rule_source_memory_emit_address_coordinate_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_address_materializer_t* materializer,
    int64_t value, loom_value_id_t* out_value_id) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, materializer->const_coordinate_descriptor_ref,
          &descriptor));
  loom_scalar_type_t source_scalar_type = LOOM_SCALAR_TYPE_OFFSET;
  if (materializer->coordinate_type ==
      LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_INDEX) {
    source_scalar_type = LOOM_SCALAR_TYPE_INDEX;
    // The complete coordinate is representable, but a factored constant may
    // not be. Project it into the same modular carrier as the dynamic terms.
    const uint32_t bitwidth =
        loom_low_lower_context_bundle(context)->snapshot->index_bitwidth;
    if (bitwidth < 64) {
      const uint64_t sign_bit = UINT64_C(1) << (bitwidth - 1);
      const uint64_t bits =
          iree_math_mask_low_bits_u64((uint64_t)value, bitwidth);
      value = (int64_t)(bits ^ sign_bit) - (int64_t)sign_bit;
    }
  } else {
    IREE_ASSERT_EQ(materializer->coordinate_type,
                   LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
      context, source_op, loom_type_scalar(source_scalar_type), &result_type));
  return loom_low_lower_rule_source_memory_emit_resolved_integer_const(
      context, &descriptor,
      loom_low_lower_rule_set_string(
          rule_set, materializer->const_coordinate_immediate_string_ref),
      value, result_type, source_op->location, out_value_id);
}

static iree_status_t
loom_low_lower_rule_source_memory_materializer_copy_operands(
    loom_low_lower_context_t* context, loom_location_id_t location,
    uint16_t copy_operand_mask, uint16_t operand_count,
    loom_value_id_t* operands) {
  if (copy_operand_mask == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(operand_count, 16);
  for (uint16_t i = 0; i < operand_count; ++i) {
    const uint16_t operand_bit = (uint16_t)((uint16_t)1u << i);
    if (!iree_any_bit_set(copy_operand_mask, operand_bit)) {
      continue;
    }
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
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, tied_result_count, sizeof(*tied_results),
      (void**)&tied_results));
  uint16_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor_row->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set
             ->constraints[descriptor_row->constraint_start + (uint32_t)i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) {
      continue;
    }
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

static iree_status_t loom_low_lower_rule_source_memory_emit_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref, loom_value_id_t* operands,
    uint16_t operand_count, loom_type_t result_type,
    loom_named_attr_slice_t attributes, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, descriptor_ref, &descriptor));

  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_materializer_tied_results(
          context, &descriptor, operand_count, operands, location,
          &tied_results, &tied_result_count));
  if (loom_type_equal(result_type, loom_type_none())) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
        context, descriptor.descriptor, 0, &result_type));
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count, attributes, &result_type,
      1, tied_results, tied_result_count, location, &low_op));
  const loom_value_slice_t results = loom_low_op_results(low_op);
  IREE_ASSERT_EQ(results.count, 1u);
  *out_value_id = results.values[0];
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_source_memory_emit_binary_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  loom_value_id_t operands[2] = {lhs, rhs};
  const loom_type_t result_type =
      loom_module_value_type(loom_low_lower_context_module(context), lhs);
  return loom_low_lower_rule_source_memory_emit_op(
      context, rule_set, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      result_type, loom_named_attr_slice_empty(), location, out_value_id);
}

static iree_status_t loom_low_lower_rule_source_memory_emit_multiply_add(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref, loom_value_id_t addend,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  loom_value_id_t operands[3] = {addend, lhs, rhs};
  const loom_type_t result_type =
      loom_module_value_type(loom_low_lower_context_module(context), addend);
  return loom_low_lower_rule_source_memory_emit_op(
      context, rule_set, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      result_type, loom_named_attr_slice_empty(), location, out_value_id);
}

static iree_status_t loom_low_lower_rule_source_memory_emit_static_bias(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_byte_offset_materializer_t* materializer,
    int64_t value, loom_location_id_t location, loom_value_id_t* out_value_id) {
  if (materializer->static_bias_descriptor_ref ==
      LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    return loom_low_lower_rule_source_memory_emit_byte_offset_const(
        context, rule_set, materializer, materializer->constant_descriptor_ref,
        value, location, out_value_id);
  }

  loom_string_id_t immediate_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context),
      loom_low_lower_rule_set_string(
          rule_set, materializer->constant_immediate_string_ref),
      &immediate_name_id));
  const loom_named_attr_t attr = {
      .name_id = immediate_name_id,
      .value = loom_attr_i64(value),
  };
  return loom_low_lower_rule_source_memory_emit_op(
      context, rule_set, materializer->static_bias_descriptor_ref, NULL, 0,
      loom_type_none(), loom_make_named_attr_slice(&attr, 1), location,
      out_value_id);
}

static iree_status_t loom_low_lower_rule_source_memory_convert_integer(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_integer_conversion_t* conversion,
    loom_low_lower_descriptor_ref_t constant_descriptor_ref,
    loom_string_ref_t constant_immediate_string_ref, loom_value_id_t input,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  loom_named_attr_t attribute = {0};
  loom_named_attr_slice_t attributes = loom_named_attr_slice_empty();
  if (conversion->immediate_string_ref != LOOM_STRING_REF_NONE) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        loom_low_lower_context_module(context),
        loom_low_lower_rule_set_string(rule_set,
                                       conversion->immediate_string_ref),
        &attribute.name_id));
    attribute.value = loom_attr_i64(conversion->immediate_value);
    attributes = loom_make_named_attr_slice(&attribute, 1);
  }
  loom_value_id_t operands[3] = {input, LOOM_VALUE_ID_INVALID,
                                 LOOM_VALUE_ID_INVALID};
  if (conversion->input_count == 3) {
    loom_low_lower_resolved_descriptor_t constant = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
            context, rule_set, constant_descriptor_ref, &constant));
    loom_type_t carrier_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
        context, constant.descriptor, 0, &carrier_type));
    const iree_string_view_t immediate_name =
        loom_low_lower_rule_set_string(rule_set, constant_immediate_string_ref);
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_resolved_integer_const(
            context, &constant, immediate_name, 1, carrier_type, location,
            &operands[1]));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_resolved_integer_const(
            context, &constant, immediate_name, 0, carrier_type, location,
            &operands[2]));
  }
  return loom_low_lower_rule_source_memory_emit_op(
      context, rule_set, conversion->descriptor_ref, operands,
      conversion->input_count, loom_type_none(), attributes, location,
      out_value_id);
}

// Canonical terms retain their numeric source domain after index casts have
// been factored out. Convert that domain before byte arithmetic; equal register
// unit counts say nothing about scalar width or signed extension. Projection
// within one register class handles wide physical tuples without instructions.
static iree_status_t loom_low_lower_rule_source_memory_lookup_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_byte_offset_materializer_t* materializer,
    loom_value_id_t source_value_id, loom_type_t offset_type,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value_id, out_value_id));
  const loom_scalar_type_t scalar_type =
      loom_type_element_type(loom_module_value_type(
          loom_low_lower_context_module(context), source_value_id));
  if (loom_scalar_type_is_integer(scalar_type)) {
    const loom_low_lower_source_memory_integer_conversion_t* conversion =
        &materializer->integer_conversions[scalar_type - LOOM_SCALAR_TYPE_I1];
    if (conversion->descriptor_ref != LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
      return loom_low_lower_rule_source_memory_convert_integer(
          context, rule_set, conversion, materializer->constant_descriptor_ref,
          materializer->constant_immediate_string_ref, *out_value_id, location,
          out_value_id);
    }
  }
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), *out_value_id);
  IREE_ASSERT_EQ(loom_low_register_type_class_id(source_type),
                 loom_low_register_type_class_id(offset_type),
                 "canonical address term requires a carrier conversion");
  if (loom_low_register_type_unit_count(source_type) <=
      loom_low_register_type_unit_count(offset_type)) {
    return iree_ok_status();
  }
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_slice_build(loom_low_lower_context_builder(context),
                           *out_value_id, 0, offset_type, location, &slice_op));
  *out_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_materialize_source_memory_dynamic_term(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    uint8_t term_ordinal, loom_value_id_t* out_value_id) {
  IREE_ASSERT_NE(source_memory->byte_offset_materializer_ordinal,
                 LOOM_LOW_LOWER_SOURCE_MEMORY_MATERIALIZER_NONE);
  IREE_ASSERT_LT(term_ordinal, source_memory_access->dynamic_term_count);
  const loom_low_lower_source_memory_byte_offset_materializer_t* materializer =
      loom_low_lower_rule_set_source_memory_byte_offset_materializer(
          rule_set, source_memory);
  const loom_value_id_t source_value_id =
      source_memory_access->dynamic_terms[term_ordinal].index;
  const loom_scalar_type_t scalar_type =
      loom_type_element_type(loom_module_value_type(
          loom_low_lower_context_module(context), source_value_id));
  if (!loom_scalar_type_is_integer(scalar_type)) {
    return loom_low_lower_lookup_value(context, source_value_id, out_value_id);
  }
  const loom_low_lower_source_memory_integer_conversion_t* conversion =
      &materializer->integer_conversions[scalar_type - LOOM_SCALAR_TYPE_I1];
  if (conversion->descriptor_ref != LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_value_id, out_value_id));
    return loom_low_lower_rule_source_memory_convert_integer(
        context, rule_set, conversion, materializer->constant_descriptor_ref,
        materializer->constant_immediate_string_ref, *out_value_id,
        source_op->location, out_value_id);
  }
  loom_low_lower_resolved_descriptor_t constant_descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, materializer->constant_descriptor_ref,
          &constant_descriptor));
  loom_type_t carrier_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
      context, constant_descriptor.descriptor, 0, &carrier_type));
  return loom_low_lower_rule_source_memory_lookup_byte_offset(
      context, rule_set, materializer, source_value_id, carrier_type,
      source_op->location, out_value_id);
}

static iree_status_t loom_low_lower_rule_materialize_source_memory_term(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_byte_offset_materializer_t* materializer,
    const loom_low_source_memory_dynamic_term_t* term, loom_type_t offset_type,
    loom_value_id_t addend, loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;

  loom_value_id_t index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_lookup_byte_offset(
      context, rule_set, materializer, term->index, offset_type,
      source_op->location, &index));
  loom_value_id_t accumulator = index;
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t stride_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_lookup_byte_offset(
        context, rule_set, materializer, term->stride_values[i], offset_type,
        source_op->location, &stride_value));
    const bool can_accumulate = addend != LOOM_VALUE_ID_INVALID &&
                                i + 1 == term->stride_value_count &&
                                term->byte_stride == 1;
    if (can_accumulate) {
      return loom_low_lower_rule_source_memory_emit_multiply_add(
          context, rule_set, materializer->multiply_add_descriptor_ref, addend,
          accumulator, stride_value, source_op->location, out_value_id);
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, materializer->multiply_descriptor_ref, accumulator,
        stride_value, source_op->location, &accumulator));
  }

  if (term->byte_stride == 1) {
    IREE_ASSERT_EQ(addend, LOOM_VALUE_ID_INVALID);
    *out_value_id = accumulator;
    return iree_ok_status();
  }

  if (addend != LOOM_VALUE_ID_INVALID) {
    IREE_ASSERT_NE(materializer->multiply_add_descriptor_ref,
                   LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE);
    loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_byte_offset_const(
            context, rule_set, materializer,
            materializer->constant_descriptor_ref, term->byte_stride,
            source_op->location, &stride));
    return loom_low_lower_rule_source_memory_emit_multiply_add(
        context, rule_set, materializer->multiply_add_descriptor_ref, addend,
        accumulator, stride, source_op->location, out_value_id);
  }

  if (term->byte_shift != LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE &&
      materializer->shift_left_descriptor_ref !=
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_byte_offset_const(
            context, rule_set, materializer,
            materializer->constant_descriptor_ref, term->byte_shift,
            source_op->location, &shift));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, materializer->shift_left_descriptor_ref, accumulator,
        shift, source_op->location, &accumulator));
  } else {
    loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_byte_offset_const(
            context, rule_set, materializer,
            materializer->constant_descriptor_ref, term->byte_stride,
            source_op->location, &stride));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, materializer->multiply_descriptor_ref, accumulator,
        stride, source_op->location, &accumulator));
  }
  IREE_ASSERT_EQ(addend, LOOM_VALUE_ID_INVALID);
  *out_value_id = accumulator;
  return iree_ok_status();
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
  const loom_low_lower_source_memory_byte_offset_materializer_t* materializer =
      loom_low_lower_rule_set_source_memory_byte_offset_materializer(
          rule_set, source_memory);
  loom_low_lower_resolved_descriptor_t constant_descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, materializer->constant_descriptor_ref,
          &constant_descriptor));
  loom_type_t offset_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
      context, constant_descriptor.descriptor, 0, &offset_type));

  if (loom_low_source_memory_access_dynamic_offset_has_materialized_view_base(
          source_memory_access) &&
      loom_low_lower_source_value_has_low_mapping(
          context, source_memory_access->dynamic_view_base_value_id)) {
    return loom_low_lower_rule_source_memory_lookup_byte_offset(
        context, rule_set, materializer,
        source_memory_access->dynamic_view_base_value_id, offset_type,
        source_op->location, out_value_id);
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
        context, rule_set, source_op, materializer, term, offset_type,
        LOOM_VALUE_ID_INVALID, &term_value));
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      accumulator = term_value;
    } else {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
          context, rule_set, materializer->add_descriptor_ref, accumulator,
          term_value, source_op->location, &accumulator));
    }
    term_ordinal = (uint8_t)(term_ordinal + consumed_term_count);
  }

  IREE_ASSERT_NE(accumulator, LOOM_VALUE_ID_INVALID);
  *out_value_id = accumulator;
  return iree_ok_status();
}

iree_status_t
loom_low_lower_rule_materialize_source_memory_complete_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(source_memory_access != NULL);
  IREE_ASSERT_GT(source_memory_access->dynamic_term_count, 0);
  const loom_low_lower_source_memory_byte_offset_materializer_t* materializer =
      loom_low_lower_rule_set_source_memory_byte_offset_materializer(
          rule_set, source_memory);
  const int64_t static_byte_offset = source_memory_access->static_byte_offset;
  const loom_low_source_memory_dynamic_term_t* term =
      &source_memory_access->dynamic_terms[0];
  const bool term_requires_multiply =
      term->stride_value_count != 0 || term->byte_stride != 1;
  const bool can_multiply_add =
      static_byte_offset != 0 &&
      source_memory_access->dynamic_term_count == 1 &&
      materializer->multiply_add_descriptor_ref !=
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE &&
      term_requires_multiply &&
      !(loom_low_source_memory_access_dynamic_offset_has_materialized_view_base(
            source_memory_access) &&
        loom_low_lower_source_value_has_low_mapping(
            context, source_memory_access->dynamic_view_base_value_id));
  loom_value_id_t static_value = LOOM_VALUE_ID_INVALID;
  if (static_byte_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_static_bias(
        context, rule_set, materializer, static_byte_offset,
        source_op->location, &static_value));
  }
  if (can_multiply_add) {
    const loom_type_t offset_type = loom_module_value_type(
        loom_low_lower_context_module(context), static_value);
    return loom_low_lower_rule_materialize_source_memory_term(
        context, rule_set, source_op, materializer, term, offset_type,
        static_value, out_value_id);
  }

  loom_value_id_t dynamic_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_materialize_source_memory_byte_offset(
          context, rule_set, source_op, source_memory, source_memory_access,
          &dynamic_byte_offset));
  if (static_byte_offset == 0) {
    *out_value_id = dynamic_byte_offset;
    return iree_ok_status();
  }

  return loom_low_lower_rule_source_memory_emit_binary_op(
      context, rule_set, materializer->add_descriptor_ref, dynamic_byte_offset,
      static_value, source_op->location, out_value_id);
}

static iree_status_t
loom_low_lower_rule_materialize_source_memory_address_coordinate_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_address_materializer_t* materializer,
    loom_value_id_t source_value_id, loom_value_id_t* out_value_id) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, source_value_id);
  loom_value_id_t low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value_id, &low_value_id));
  const loom_scalar_type_t scalar_type = loom_type_element_type(source_type);
  if (loom_scalar_type_is_integer(scalar_type)) {
    return loom_low_lower_rule_source_memory_convert_integer(
        context, rule_set,
        &materializer->integer_conversions[scalar_type - LOOM_SCALAR_TYPE_I1],
        materializer->const_coordinate_descriptor_ref,
        materializer->const_coordinate_immediate_string_ref, low_value_id,
        source_op->location, out_value_id);
  }
  if (materializer->coordinate_type ==
      LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_INDEX) {
    IREE_ASSERT(
        loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)));
    *out_value_id = low_value_id;
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(materializer->coordinate_type,
                 LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET);
  if (loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET))) {
    *out_value_id = low_value_id;
    return iree_ok_status();
  }

  IREE_ASSERT(
      loom_type_equal(source_type, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)));
  return loom_low_lower_rule_source_memory_emit_op(
      context, rule_set, materializer->index_to_coordinate_descriptor_ref,
      &low_value_id, 1, loom_type_none(), loom_named_attr_slice_empty(),
      source_op->location, out_value_id);
}

static iree_status_t loom_low_lower_rule_materialize_source_memory_address_term(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_address_materializer_t* materializer,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t* out_value_id) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_materialize_source_memory_address_coordinate_value(
          context, rule_set, source_op, materializer, term->index,
          &accumulator));
  for (uint8_t i = 0; i < term->stride_value_count; ++i) {
    loom_value_id_t stride_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_materialize_source_memory_address_coordinate_value(
            context, rule_set, source_op, materializer, term->stride_values[i],
            &stride_value));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, materializer->mul_coordinate_descriptor_ref,
        accumulator, stride_value, source_op->location, &accumulator));
  }

  const int64_t unit_byte_count = materializer->coordinate_unit_byte_count;
  IREE_ASSERT_GT(unit_byte_count, 0);
  IREE_ASSERT_EQ(term->byte_stride % unit_byte_count, 0);
  const int64_t coordinate_stride = term->byte_stride / unit_byte_count;
  if (coordinate_stride == 1) {
    *out_value_id = accumulator;
    return iree_ok_status();
  }

  if (coordinate_stride > 0 &&
      iree_math_is_power_of_two_i64(coordinate_stride) &&
      materializer->shl_coordinate_descriptor_ref !=
          LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    loom_value_id_t shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_address_coordinate_const(
            context, rule_set, source_op, materializer,
            iree_math_count_trailing_zeros_u64((uint64_t)coordinate_stride),
            &shift));
    return loom_low_lower_rule_source_memory_emit_binary_op(
        context, rule_set, materializer->shl_coordinate_descriptor_ref,
        accumulator, shift, source_op->location, out_value_id);
  }

  loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_emit_address_coordinate_const(
          context, rule_set, source_op, materializer, coordinate_stride,
          &stride));
  return loom_low_lower_rule_source_memory_emit_binary_op(
      context, rule_set, materializer->mul_coordinate_descriptor_ref,
      accumulator, stride, source_op->location, out_value_id);
}

static iree_status_t
loom_low_lower_rule_materialize_source_memory_complete_coordinate(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_address_materializer_t* materializer,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  uint8_t first_canonical_term = 0;
  int64_t static_byte_offset = source_memory_access->static_byte_offset;
  if (materializer->coordinate_unit_byte_count == 1 &&
      materializer->coordinate_type ==
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
            context, rule_set, source_op, materializer,
            &source_memory_access->dynamic_terms[term_ordinal], &term_value));
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      accumulator = term_value;
    } else {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
          context, rule_set, materializer->add_coordinate_descriptor_ref,
          accumulator, term_value, source_op->location, &accumulator));
    }
  }

  const int64_t unit_byte_count = materializer->coordinate_unit_byte_count;
  IREE_ASSERT_GT(unit_byte_count, 0);
  IREE_ASSERT_EQ(static_byte_offset % unit_byte_count, 0);
  const int64_t static_coordinate = static_byte_offset / unit_byte_count;
  if (static_coordinate != 0 || accumulator == LOOM_VALUE_ID_INVALID) {
    loom_value_id_t static_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_source_memory_emit_address_coordinate_const(
            context, rule_set, source_op, materializer, static_coordinate,
            &static_value));
    if (accumulator == LOOM_VALUE_ID_INVALID) {
      accumulator = static_value;
    } else {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_memory_emit_binary_op(
          context, rule_set, materializer->add_coordinate_descriptor_ref,
          accumulator, static_value, source_op->location, &accumulator));
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
  const loom_low_lower_source_memory_address_materializer_t* materializer =
      loom_low_lower_rule_set_source_memory_address_materializer(rule_set,
                                                                 source_memory);
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t source_base =
      materializer->base_kind == LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_BASE_VIEW
          ? loom_low_source_memory_access_base_view_value_id(
                source_memory_access)
          : source_memory_access->root_value_id;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_base, &base));
  loom_value_id_t coordinate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_materialize_source_memory_complete_coordinate(
          context, rule_set, source_op, materializer, source_memory_access,
          &coordinate));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_source_memory_resolve_materializer_descriptor(
          context, rule_set, materializer->address_descriptor_ref,
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
