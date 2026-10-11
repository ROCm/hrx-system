// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_plan.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/module_state.h"
#include "loom/codegen/low/lower/rule_descriptor.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/lower/rule_value.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"

typedef struct loom_low_lower_rule_source_t {
  // Selected source operation whose attributes and values are projected.
  const loom_op_t* source_op;
  // Selected source graph, including the root, or NULL for root-only rules.
  const loom_op_t* const* source_nodes;
  // Number of operations in the selected source graph.
  uint8_t source_node_count;
} loom_low_lower_rule_source_t;

static loom_value_id_t loom_low_lower_rule_plan_source_value(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_source_t* state, uint16_t value_ref_index) {
  loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
  const bool resolved = loom_low_lower_rule_resolve_source_value_from_nodes(
      context->module, loom_low_lower_context_fact_table(context),
      (loom_target_contract_vector_lane_projection_t){0}, rule_set,
      state->source_op, state->source_nodes, state->source_node_count,
      value_ref_index, &source_value_id);
  IREE_ASSERT(resolved);
  return source_value_id;
}

static uint64_t loom_low_lower_rule_attr_copy_float_bits(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_source_t* state,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const loom_value_id_t source_value_id = loom_low_lower_rule_plan_source_value(
      context, rule_set, state, attr_copy->value_ref_index);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  loom_value_facts_t facts = loom_value_facts_unknown();
  const loom_module_t* module = loom_low_lower_context_module(context);
  const bool has_float_facts = loom_low_lower_rule_float_immediate_facts(
      module, fact_table, source_value_id, &facts);
  IREE_ASSERT(has_float_facts);
  const loom_scalar_type_t scalar_type =
      loom_type_element_type(loom_module_value_type(module, source_value_id));
  double value = 0.0;
  const bool has_value =
      loom_value_facts_as_exact_float(scalar_type, facts, &value);
  IREE_ASSERT(has_value);
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return iree_math_f32_to_f8e4m3fn((float)value);
    case LOOM_SCALAR_TYPE_F8E5M2:
      return iree_math_f32_to_f8e5m2((float)value);
    case LOOM_SCALAR_TYPE_F16:
      return iree_math_f32_to_f16((float)value);
    case LOOM_SCALAR_TYPE_BF16:
      return iree_math_f32_to_bf16((float)value);
    case LOOM_SCALAR_TYPE_F32: {
      const float f32_value = (float)value;
      uint32_t bits = 0;
      memcpy(&bits, &f32_value, sizeof(bits));
      return bits;
    }
    case LOOM_SCALAR_TYPE_F64: {
      uint64_t bits = 0;
      memcpy(&bits, &value, sizeof(bits));
      return bits;
    }
    default:
      IREE_BUILTIN_UNREACHABLE();
  }
}

static int64_t loom_low_lower_rule_attr_copy_float_power_of_two_exponent(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_source_t* state,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const loom_value_id_t source_value_id = loom_low_lower_rule_plan_source_value(
      context, rule_set, state, attr_copy->value_ref_index);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  loom_value_facts_t facts = loom_value_facts_unknown();
  const loom_module_t* module = loom_low_lower_context_module(context);
  const bool has_float_facts = loom_low_lower_rule_float_immediate_facts(
      module, fact_table, source_value_id, &facts);
  IREE_ASSERT(has_float_facts);
  const loom_scalar_type_t scalar_type =
      loom_type_element_type(loom_module_value_type(module, source_value_id));
  int32_t exponent = 0;
  const bool is_power_of_two = loom_value_facts_as_exact_power_of_two_float(
      scalar_type, facts, &exponent);
  IREE_ASSERT(is_power_of_two);
  return exponent;
}

static int64_t loom_low_lower_rule_attr_copy_exact_i64(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_source_t* state,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const loom_value_id_t source_value_id = loom_low_lower_rule_plan_source_value(
      context, rule_set, state, attr_copy->value_ref_index);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  loom_value_facts_t facts = loom_value_facts_unknown();
  const bool has_integer_facts = loom_low_lower_rule_integer_immediate_facts(
      loom_low_lower_context_module(context), fact_table, source_value_id,
      &facts);
  IREE_ASSERT(has_integer_facts);
  int64_t value = 0;
  const bool has_value = loom_value_facts_as_exact_i64(facts, &value);
  IREE_ASSERT(has_value);
  return value;
}

static int64_t loom_low_lower_rule_attr_copy_static_dim_projected(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_source_t* state,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const loom_value_id_t source_value_id = loom_low_lower_rule_plan_source_value(
      context, rule_set, state, attr_copy->value_ref_index);
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_value_id);
  IREE_ASSERT(loom_type_is_shaped(source_type));
  const uint16_t dimension = attr_copy->source_element_index;
  IREE_ASSERT_LT(dimension, loom_type_rank(source_type));
  IREE_ASSERT(!loom_type_dim_is_dynamic_at(source_type, dimension));
  const int64_t static_dimension =
      loom_type_dim_static_size_at(source_type, dimension);
  IREE_ASSERT_GE(static_dimension, 0);
  IREE_ASSERT_GT(attr_copy->source_element_count, 0);
  IREE_ASSERT_LE((uint64_t)static_dimension,
                 (uint64_t)INT64_MAX / attr_copy->source_element_count);
  const int64_t scaled_dimension =
      static_dimension * attr_copy->source_element_count;
  if (attr_copy->kind ==
      LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_LITERAL_MINUS_STATIC_DIM_SCALED) {
    IREE_ASSERT_GE(attr_copy->literal_i64, 0);
    return attr_copy->literal_i64 - scaled_dimension;
  }
  if (attr_copy->literal_i64 > 0) {
    IREE_ASSERT_LE(scaled_dimension, INT64_MAX - attr_copy->literal_i64);
  }
  const int64_t projected_dimension = scaled_dimension + attr_copy->literal_i64;
  if (attr_copy->kind ==
      LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_STATIC_DIM_LOW_BITS_MASK) {
    IREE_ASSERT_GE(projected_dimension, 0);
    IREE_ASSERT_LE(projected_dimension, 32);
    return (int64_t)(int32_t)iree_math_mask_low_bits_u32(
        UINT32_MAX, (int32_t)projected_dimension);
  }
  IREE_ASSERT_EQ(attr_copy->kind,
                 LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_STATIC_DIM_SCALED);
  return projected_dimension;
}

static void loom_low_lower_rule_set_projected_bits_attr(
    const loom_low_lower_attr_copy_t* attr_copy, uint64_t bit_pattern,
    loom_named_attr_t* attr) {
  if (attr_copy->target_bit_offset != 0) {
    IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
    IREE_ASSERT_LE(bit_pattern,
                   (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
    bit_pattern <<= attr_copy->target_bit_offset;
  }
  attr->value = loom_attr_i64((int64_t)bit_pattern);
}

static int64_t loom_low_lower_rule_i64_source_attr(
    const loom_op_t* source_op, const loom_attribute_t* source_attrs,
    uint16_t source_attr_index) {
  IREE_ASSERT_LT(source_attr_index, source_op->attribute_count);
  loom_attribute_t source_attr = source_attrs[source_attr_index];
  IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64);
  return source_attr.i64;
}

static uint32_t loom_low_lower_rule_u32_low_bit_mask(uint32_t width) {
  IREE_ASSERT_GT(width, 0u);
  IREE_ASSERT_LE(width, 32u);
  return width == 32u ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
}

static uint32_t loom_low_lower_rule_attr_copy_u32_bit_mask(
    const loom_op_t* source_op, const loom_attribute_t* source_attrs,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const int64_t width_i64 = loom_low_lower_rule_i64_source_attr(
      source_op, source_attrs, attr_copy->source_attr_index);
  IREE_ASSERT_GE(width_i64, 1);
  IREE_ASSERT_LE(width_i64, 32);
  uint32_t mask = loom_low_lower_rule_u32_low_bit_mask((uint32_t)width_i64);
  switch (attr_copy->kind) {
    case LOOM_LOW_LOWER_ATTR_COPY_I64_LOW_BIT_MASK:
      return mask;
    case LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_MASK:
    case LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_CLEAR_MASK: {
      const int64_t offset_i64 = loom_low_lower_rule_i64_source_attr(
          source_op, source_attrs, attr_copy->other_source_attr_index);
      IREE_ASSERT_GE(offset_i64, 0);
      IREE_ASSERT_LE(offset_i64, 31);
      IREE_ASSERT_LE(offset_i64 + width_i64, 32);
      mask <<= (uint32_t)offset_i64;
      return attr_copy->kind ==
                     LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_CLEAR_MASK
                 ? ~mask
                 : mask;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated bit-mask attr copy kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static int64_t loom_low_lower_rule_attr_copy_literal_minus_i64_attrs(
    const loom_op_t* source_op, const loom_attribute_t* source_attrs,
    const loom_low_lower_attr_copy_t* attr_copy) {
  int64_t projected_value = attr_copy->literal_i64;
  const int64_t source_value = loom_low_lower_rule_i64_source_attr(
      source_op, source_attrs, attr_copy->source_attr_index);
  IREE_ASSERT_GE(source_value, 0);
  IREE_ASSERT_LE(source_value, projected_value);
  projected_value -= source_value;
  if (attr_copy->kind == LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTRS) {
    const int64_t other_source_value = loom_low_lower_rule_i64_source_attr(
        source_op, source_attrs, attr_copy->other_source_attr_index);
    IREE_ASSERT_GE(other_source_value, 0);
    IREE_ASSERT_LE(other_source_value, projected_value);
    projected_value -= other_source_value;
  } else {
    IREE_ASSERT_EQ(attr_copy->kind,
                   LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTR);
  }
  return projected_value;
}

static int64_t loom_low_lower_rule_attr_copy_i64_attr_minus_literal(
    const loom_op_t* source_op, const loom_attribute_t* source_attrs,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const int64_t source_value = loom_low_lower_rule_i64_source_attr(
      source_op, source_attrs, attr_copy->source_attr_index);
  int64_t projected_value = 0;
  const bool in_range = iree_checked_sub_i64(
      source_value, attr_copy->literal_i64, &projected_value);
  IREE_ASSERT(in_range);
  return projected_value;
}

static loom_memory_access_flags_t loom_low_lower_resolve_emit_access_flags(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, const loom_low_lower_emit_t* emit,
    loom_memory_access_flags_t access_flags) {
  // Numerical requirements are consumed by descriptor selection. Only physical
  // access semantics remain on the selected Low memory instruction.
  access_flags &= LOOM_MEMORY_ACCESS_FLAG_VOLATILE;
  if (access_flags == 0 || emit->source_memory_ordinal == 0) {
    return 0;
  }
  // Source-memory rows also describe address materialization. Only actual
  // memory packets inherit the source operation's observable access.
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (loom_low_effect_is_memory_access(effect)) {
      return access_flags;
    }
  }
  return 0;
}

static iree_status_t loom_low_lower_rule_project_read_only_data(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_attribute_t* source_attrs,
    const loom_low_lower_attr_copy_t* attr_copy,
    loom_low_lower_read_only_data_id_t* out_id) {
  IREE_ASSERT_LT(attr_copy->source_attr_index, source_op->attribute_count);
  const loom_attribute_t source_attr =
      source_attrs[attr_copy->source_attr_index];
  IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);

  iree_host_size_t byte_length = 0;
  uint8_t* bytes = NULL;
  switch (attr_copy->kind) {
    case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_ELEMENTS: {
      IREE_ASSERT(attr_copy->source_element_bit_width == 8 ||
                  attr_copy->source_element_bit_width == 16 ||
                  attr_copy->source_element_bit_width == 32 ||
                  attr_copy->source_element_bit_width == 64);
      const iree_host_size_t element_byte_count =
          attr_copy->source_element_bit_width / 8;
      const bool has_byte_length = iree_host_size_checked_mul(
          source_attr.count, element_byte_count, &byte_length);
      IREE_ASSERT(has_byte_length);
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_planning_array(
          context, source_attr.count, element_byte_count, (void**)&bytes));
      for (iree_host_size_t i = 0; i < source_attr.count; ++i) {
        const uint64_t value = (uint64_t)source_attr.i64_array[i];
        for (iree_host_size_t byte = 0; byte < element_byte_count; ++byte) {
          bytes[i * element_byte_count + byte] = (uint8_t)(value >> (byte * 8));
        }
      }
      break;
    }
    case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_BYTE_SEGMENT: {
      const uint32_t bytes_per_lane = attr_copy->source_element_count;
      const uint32_t source_byte_count = attr_copy->source_element_bit_width;
      const uint64_t source_byte_offset = (uint64_t)attr_copy->literal_i64;
      IREE_ASSERT_GT(bytes_per_lane, 0);
      IREE_ASSERT_GT(source_byte_count, 0);
      IREE_ASSERT_GE(attr_copy->literal_i64, 0);
      const bool has_byte_length = iree_host_size_checked_mul(
          source_attr.count, bytes_per_lane, &byte_length);
      IREE_ASSERT(has_byte_length);
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_planning_array(
          context, byte_length, 1, (void**)&bytes));
      for (iree_host_size_t output_byte = 0; output_byte < byte_length;
           ++output_byte) {
        const iree_host_size_t output_lane = output_byte / bytes_per_lane;
        const uint32_t lane_byte = (uint32_t)(output_byte % bytes_per_lane);
        const uint64_t source_byte =
            (uint64_t)source_attr.i64_array[output_lane] * bytes_per_lane +
            lane_byte;
        bytes[output_byte] =
            source_byte >= source_byte_offset &&
                    source_byte < source_byte_offset + source_byte_count
                ? (uint8_t)(source_byte - source_byte_offset)
                : 0x80;
      }
      break;
    }
    case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_BYTE_WORDS: {
      IREE_ASSERT_EQ(source_attr.count % 2, 0u);
      IREE_ASSERT_LE(attr_copy->source_element_index, 1u);
      byte_length = source_attr.count;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_planning_array(
          context, byte_length, 1, (void**)&bytes));
      const uint32_t byte_parity = attr_copy->source_element_index;
      for (iree_host_size_t output_word = 0;
           output_word < source_attr.count / 2; ++output_word) {
        const uint16_t source_lane =
            (uint16_t)source_attr.i64_array[output_word * 2 + byte_parity];
        const uint16_t source_word = source_lane / 2;
        const uint16_t shift_count = (source_lane & 1) * 8;
        const uint16_t encoded_control =
            source_word | (uint16_t)(shift_count << 5);
        bytes[output_word * 2] = (uint8_t)encoded_control;
        bytes[output_word * 2 + 1] = (uint8_t)(encoded_control >> 8);
      }
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown read-only data projection kind");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_GT(byte_length, 0);
  const uint64_t natural_alignment =
      iree_min(UINT64_C(64),
               UINT64_C(1) << iree_math_count_trailing_zeros_u64(byte_length));
  return loom_low_lower_module_state_intern_read_only_data(
      loom_low_lower_context_module_state(context),
      iree_make_const_byte_span(bytes, byte_length), natural_alignment,
      source_op->location, out_id);
}

static iree_status_t loom_low_lower_rule_build_attrs(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_source_t* state,
    const loom_low_lower_emit_t* emit,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_named_attr_t* attrs,
    loom_low_lower_read_only_attributes_t* read_only_attributes) {
  const loom_attribute_t* source_attrs = loom_op_const_attrs(source_op);
  const int64_t static_byte_offset =
      emit->source_memory_ordinal != 0
          ? loom_low_lower_rule_source_memory_static_byte_offset(
                &rule_set->source_memories[emit->source_memory_ordinal - 1],
                source_memory_access)
          : 0;
  for (uint16_t i = 0; i < emit->attr_copy_count; ++i) {
    uint16_t attr_copy_index =
        (uint16_t)(emit->payload.descriptor.attr_copy_start + i);
    const loom_low_lower_attr_copy_t* attr_copy =
        &rule_set->attr_copies[attr_copy_index];
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        loom_low_lower_context_module(context),
        loom_low_lower_rule_set_string(rule_set,
                                       attr_copy->target_name_string_ref),
        &attrs[i].name_id));
    switch (attr_copy->kind) {
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_ELEMENTS:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_BYTE_SEGMENT:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_BYTE_WORDS: {
        loom_low_lower_read_only_data_id_t* ids =
            (loom_low_lower_read_only_data_id_t*)(read_only_attributes + 1);
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_project_read_only_data(
            context, source_op, source_attrs, attr_copy, &ids[i]));
        read_only_attributes->attribute_mask |= UINT32_C(1) << i;
        attrs[i].value = loom_attr_symbol(loom_symbol_ref_null());
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LOG2:
        attrs[i].value = loom_attr_i64(iree_math_floor_log2_u64(
            (uint64_t)source_attrs[attr_copy->source_attr_index].i64));
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_DIRECT:
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        attrs[i].value = source_attrs[attr_copy->source_attr_index];
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_ENUM_ORDINAL: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_ENUM);
        attrs[i].value = loom_attr_i64(loom_attr_as_enum(source_attr));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_ENUM_REMAP: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        const loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_ENUM);
        const uint64_t source_ordinal = loom_attr_as_enum(source_attr);
        IREE_ASSERT_LT(source_ordinal, attr_copy->source_element_count);
        const uint32_t bit_width = attr_copy->source_element_bit_width;
        IREE_ASSERT_GT(bit_width, 0);
        IREE_ASSERT_LE(bit_width, 32);
        const uint32_t bit_offset = (uint32_t)source_ordinal * bit_width;
        IREE_ASSERT_LE(bit_offset + bit_width, 95);
        const uint64_t lower_bits = (uint64_t)attr_copy->literal_i64;
        const uint32_t upper_bits =
            (uint32_t)attr_copy->other_source_attr_index |
            ((uint32_t)attr_copy->source_element_index << 16);
        uint64_t projected_value = 0;
        if (bit_offset < 63) {
          projected_value = lower_bits >> bit_offset;
          if (bit_offset + bit_width > 63) {
            projected_value |= (uint64_t)upper_bits << (63 - bit_offset);
          }
        } else {
          projected_value = upper_bits >> (bit_offset - 63);
        }
        const uint64_t value_mask = (UINT64_C(1) << bit_width) - 1u;
        attrs[i].value = loom_attr_i64((int64_t)(projected_value & value_mask));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);
        IREE_ASSERT_LT(attr_copy->source_element_index, source_attr.count);
        const int64_t source_value =
            source_attr.i64_array[attr_copy->source_element_index];
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(source_value);
          break;
        }
        IREE_ASSERT_GE(source_value, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)source_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)source_value << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT_PLUS_LITERAL: {
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        int64_t projected_value = 0;
        const bool has_projected_value = iree_checked_add_i64(
            source_attr.i64_array[attr_copy->source_element_index],
            attr_copy->literal_i64, &projected_value);
        IREE_ASSERT(has_projected_value);
        attrs[i].value = loom_attr_i64(projected_value);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT_QUOTIENT:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT_REMAINDER: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        const loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);
        IREE_ASSERT_LT(attr_copy->source_element_index, source_attr.count);
        IREE_ASSERT_GT(attr_copy->literal_i64, 0);
        const int64_t source_value =
            source_attr.i64_array[attr_copy->source_element_index];
        int64_t projected_value =
            attr_copy->kind ==
                    LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT_QUOTIENT
                ? source_value / attr_copy->literal_i64
                : source_value % attr_copy->literal_i64;
        if (attr_copy->target_bit_offset != 0) {
          IREE_ASSERT_GE(projected_value, 0);
          IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
          IREE_ASSERT_LE((uint64_t)projected_value,
                         (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
          projected_value = (int64_t)((uint64_t)projected_value
                                      << attr_copy->target_bit_offset);
        }
        attrs[i].value = loom_attr_i64(projected_value);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_PACK_ELEMENTS: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);
        IREE_ASSERT_GT(attr_copy->source_element_count, 0);
        IREE_ASSERT_GT(attr_copy->source_element_bit_width, 0);
        const uint32_t packed_bit_count =
            (uint32_t)attr_copy->source_element_count *
            attr_copy->source_element_bit_width;
        IREE_ASSERT_LE(packed_bit_count + attr_copy->target_bit_offset, 64);
        IREE_ASSERT_LE((uint32_t)attr_copy->source_element_index +
                           attr_copy->source_element_count,
                       source_attr.count);
        const uint64_t element_mask =
            (UINT64_C(1) << attr_copy->source_element_bit_width) - 1u;
        uint64_t packed_value = 0;
        for (uint16_t j = 0; j < attr_copy->source_element_count; ++j) {
          const int64_t source_value =
              source_attr.i64_array[attr_copy->source_element_index + j];
          IREE_ASSERT_GE(source_value, 0);
          IREE_ASSERT_LE((uint64_t)source_value, element_mask);
          packed_value |= (uint64_t)source_value
                          << (j * attr_copy->source_element_bit_width);
        }
        packed_value <<= attr_copy->target_bit_offset;
        attrs[i].value = loom_attr_i64((int64_t)packed_value);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_ATTRS_PACK_CONSECUTIVE: {
        uint64_t packed_value = 0;
        for (uint16_t j = 0; j < attr_copy->source_element_count; ++j) {
          // Integer and enum attributes share the raw integer payload.
          packed_value |= source_attrs[attr_copy->source_attr_index + j].raw
                          << (j * attr_copy->source_element_bit_width);
        }
        packed_value <<= attr_copy->target_bit_offset;
        attrs[i].value = loom_attr_i64((int64_t)packed_value);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_LANE_BYTE: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);
        IREE_ASSERT_LT(attr_copy->source_element_index, source_attr.count);
        IREE_ASSERT_GT(attr_copy->source_element_count, 0);
        const int64_t source_lane =
            source_attr.i64_array[attr_copy->source_element_index];
        IREE_ASSERT_GE(source_lane, 0);
        const int64_t byte_lane =
            source_lane * attr_copy->source_element_count +
            attr_copy->literal_i64;
        attrs[i].value = loom_attr_i64(byte_lane);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_SHUFFLE_MASK_CHUNK: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        const loom_attribute_t source_attr =
            source_attrs[attr_copy->source_attr_index];
        IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64_ARRAY);
        const uint32_t output_byte_offset = attr_copy->source_element_index;
        const uint32_t bytes_per_lane = attr_copy->source_element_count;
        const uint32_t source_byte_count = attr_copy->source_element_bit_width;
        const int64_t source_byte_offset = attr_copy->literal_i64;
        IREE_ASSERT_GT(bytes_per_lane, 0);
        IREE_ASSERT_GT(source_byte_count, 0);
        IREE_ASSERT_LE(source_byte_count, 128);
        IREE_ASSERT_GE(source_byte_offset, 0);
        uint64_t packed_mask = 0;
        for (uint32_t byte_ordinal = 0; byte_ordinal < 8; ++byte_ordinal) {
          const uint32_t output_byte = output_byte_offset + byte_ordinal;
          const uint32_t output_lane = output_byte / bytes_per_lane;
          IREE_ASSERT_LT(output_lane, source_attr.count);
          const int64_t source_lane = source_attr.i64_array[output_lane];
          IREE_ASSERT_GE(source_lane, 0);
          const int64_t source_byte =
              source_lane * bytes_per_lane + output_byte % bytes_per_lane;
          const bool in_segment =
              source_byte >= source_byte_offset &&
              source_byte < source_byte_offset + source_byte_count;
          const uint8_t selector =
              in_segment ? (uint8_t)(source_byte - source_byte_offset)
                         : UINT8_C(0x80);
          packed_mask |= (uint64_t)selector << (byte_ordinal * 8);
        }
        attrs[i].value = loom_attr_i64((int64_t)packed_mask);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LOW_BIT_MASK:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_MASK:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_CLEAR_MASK:
        IREE_ASSERT_EQ(attr_copy->target_bit_offset, 0);
        attrs[i].value =
            loom_attr_i64(loom_low_lower_rule_attr_copy_u32_bit_mask(
                source_op, source_attrs, attr_copy));
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTR:
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTRS:
        IREE_ASSERT_EQ(attr_copy->target_bit_offset, 0);
        attrs[i].value =
            loom_attr_i64(loom_low_lower_rule_attr_copy_literal_minus_i64_attrs(
                source_op, source_attrs, attr_copy));
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ATTR_MINUS_LITERAL:
        IREE_ASSERT_EQ(attr_copy->target_bit_offset, 0);
        attrs[i].value =
            loom_attr_i64(loom_low_lower_rule_attr_copy_i64_attr_minus_literal(
                source_op, source_attrs, attr_copy));
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL:
        attrs[i].value = loom_attr_i64(attr_copy->literal_i64);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_STATIC_DIM_SCALED:
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_LITERAL_MINUS_STATIC_DIM_SCALED:
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_STATIC_DIM_LOW_BITS_MASK:
        attrs[i].value =
            loom_attr_i64(loom_low_lower_rule_attr_copy_static_dim_projected(
                context, rule_set, state, attr_copy));
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64: {
        const int64_t source_value = loom_low_lower_rule_attr_copy_exact_i64(
            context, rule_set, state, attr_copy);
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(source_value);
          break;
        }
        IREE_ASSERT_GE(source_value, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)source_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)source_value << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_I32_WORD: {
        const uint64_t bit_pattern =
            (uint64_t)loom_low_lower_rule_attr_copy_exact_i64(context, rule_set,
                                                              state, attr_copy);
        const uint32_t word =
            (uint32_t)(bit_pattern >> (attr_copy->source_element_index * 32));
        int32_t signed_word = 0;
        memcpy(&signed_word, &word, sizeof(signed_word));
        attrs[i].value = loom_attr_i64(signed_word);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_NEGATE: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_plan_source_value(context, rule_set, state,
                                                  attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        IREE_ASSERT_GT(source_value, INT64_MIN);
        const int64_t projected_value = -source_value;
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(projected_value);
          break;
        }
        IREE_ASSERT_GE(projected_value, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)projected_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value =
            loom_attr_i64((int64_t)((uint64_t)projected_value
                                    << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_LOG2: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_plan_source_value(context, rule_set, state,
                                                  attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        IREE_ASSERT(iree_math_is_power_of_two_i64(source_value));
        const int64_t log2_value = iree_math_floor_log2_u64(source_value);
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(log2_value);
          break;
        }
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)log2_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)log2_value << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_MINUS_ONE: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_plan_source_value(context, rule_set, state,
                                                  attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        IREE_ASSERT_GT(source_value, INT64_MIN);
        const int64_t projected_value = source_value - 1;
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(projected_value);
          break;
        }
        IREE_ASSERT_GE(projected_value, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)projected_value,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value =
            loom_attr_i64((int64_t)((uint64_t)projected_value
                                    << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER:
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER_AS_I32:
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_SHIFT: {
        if (attr_copy->kind ==
                LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER &&
            attr_copy->literal_i64 == 32) {
          const uint32_t divisor =
              (uint32_t)loom_low_lower_rule_attr_copy_exact_i64(
                  context, rule_set, state, attr_copy);
          attrs[i].value = loom_attr_i64(
              (int64_t)loom_low_lower_u32_divisor_reciprocal(divisor));
          break;
        }
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_plan_source_value(context, rule_set, state,
                                                  attr_copy->value_ref_index);
        loom_low_lower_unsigned_divisor_magic_info_t info = {0};
        const loom_value_id_t numerator = loom_low_lower_rule_plan_source_value(
            context, rule_set, state, attr_copy->other_value_ref_index);
        const bool has_magic_info =
            loom_low_lower_rule_value_facts_u32_divisor_magic_info(
                loom_low_lower_context_module(context),
                loom_low_lower_context_fact_table(context), numerator,
                source_value_id, &info);
        IREE_ASSERT(has_magic_info);
        if (attr_copy->kind ==
            LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER_AS_I32) {
          attrs[i].value = loom_attr_i64((int32_t)info.multiplier);
          break;
        }
        uint64_t projected_value =
            attr_copy->kind ==
                    LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER
                ? info.multiplier
                : info.post_shift + attr_copy->literal_i64;
        if (attr_copy->target_bit_offset != 0) {
          IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
          IREE_ASSERT_LE(projected_value,
                         (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
          projected_value <<= attr_copy->target_bit_offset;
        }
        attrs[i].value = loom_attr_i64((int64_t)projected_value);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_I32_AS_U32_BITS: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_plan_source_value(context, rule_set, state,
                                                  attr_copy->value_ref_index);
        const loom_value_fact_table_t* fact_table =
            loom_low_lower_context_fact_table(context);
        loom_value_facts_t facts = loom_value_facts_unknown();
        const bool has_integer_facts =
            loom_low_lower_rule_integer_immediate_facts(
                loom_low_lower_context_module(context), fact_table,
                source_value_id, &facts);
        IREE_ASSERT(has_integer_facts);
        int64_t source_value = 0;
        const bool has_exact_value =
            loom_value_facts_as_exact_i64(facts, &source_value);
        IREE_ASSERT(has_exact_value);
        IREE_ASSERT_GE(source_value, INT32_MIN);
        IREE_ASSERT_LE(source_value, INT32_MAX);
        const uint32_t bit_pattern = (uint32_t)(int32_t)source_value;
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(bit_pattern);
          break;
        }
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)bit_pattern,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)bit_pattern << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_BITS: {
        const uint64_t bit_pattern = loom_low_lower_rule_attr_copy_float_bits(
            context, rule_set, state, attr_copy);
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F32_I32: {
        const uint32_t bit_pattern =
            (uint32_t)loom_low_lower_rule_attr_copy_float_bits(
                context, rule_set, state, attr_copy);
        int32_t signed_bit_pattern = 0;
        memcpy(&signed_bit_pattern, &bit_pattern, sizeof(signed_bit_pattern));
        attrs[i].value = loom_attr_i64(signed_bit_pattern);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F64_I64: {
        const uint64_t bit_pattern = loom_low_lower_rule_attr_copy_float_bits(
            context, rule_set, state, attr_copy);
        int64_t signed_bit_pattern = 0;
        memcpy(&signed_bit_pattern, &bit_pattern, sizeof(signed_bit_pattern));
        attrs[i].value = loom_attr_i64(signed_bit_pattern);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F64_I32_WORD: {
        const uint64_t bit_pattern = loom_low_lower_rule_attr_copy_float_bits(
            context, rule_set, state, attr_copy);
        const uint32_t word =
            (uint32_t)(bit_pattern >> (attr_copy->source_element_index * 32));
        int32_t signed_word = 0;
        memcpy(&signed_word, &word, sizeof(signed_word));
        attrs[i].value = loom_attr_i64(signed_word);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_POWER_OF_TWO_EXPONENT:
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_POWER_OF_TWO_NEGATED_EXPONENT: {
        int64_t exponent =
            loom_low_lower_rule_attr_copy_float_power_of_two_exponent(
                context, rule_set, state, attr_copy);
        if (attr_copy->kind ==
            LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_POWER_OF_TWO_NEGATED_EXPONENT) {
          exponent = -exponent;
        }
        if (attr_copy->target_bit_offset == 0) {
          attrs[i].value = loom_attr_i64(exponent);
          break;
        }
        IREE_ASSERT_GE(exponent, 0);
        IREE_ASSERT_LT(attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint64_t)exponent,
                       (uint64_t)INT64_MAX >> attr_copy->target_bit_offset);
        attrs[i].value = loom_attr_i64(
            (int64_t)((uint64_t)exponent << attr_copy->target_bit_offset));
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET:
        attrs[i].value = loom_attr_i64(static_byte_offset);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_PLUS_LITERAL: {
        attrs[i].value =
            loom_attr_i64(static_byte_offset + attr_copy->literal_i64);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT:
        IREE_ASSERT_GT(attr_copy->literal_i64, 0);
        attrs[i].value =
            loom_attr_i64(static_byte_offset / attr_copy->literal_i64);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER:
        IREE_ASSERT_GT(attr_copy->literal_i64, 0);
        attrs[i].value =
            loom_attr_i64(static_byte_offset % attr_copy->literal_i64);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_DYNAMIC_BYTE_STRIDE:
        attrs[i].value = loom_attr_i64(
            source_memory_access->dynamic_terms[attr_copy->dynamic_term_index]
                .byte_stride);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_OP_INSTANCE_FLAGS:
        attrs[i].value = loom_attr_i64(source_op->instance_flags);
        break;
      default:
        IREE_ASSERT_UNREACHABLE("unknown generated attr copy kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  return iree_ok_status();
}

static bool loom_low_lower_rule_value_ref_needs_facts(
    const loom_low_lower_value_ref_t* value_ref) {
  return value_ref->kind ==
             LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND ||
         value_ref->kind ==
             LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND ||
         value_ref->kind ==
             LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND;
}

typedef struct loom_low_lower_rule_operand_masks_t {
  // Operand ordinals whose source identities require fact projection.
  uint8_t source_values;
  // Operand ordinals with a source-dependent materializer recipe.
  uint8_t materializers;
} loom_low_lower_rule_operand_masks_t;

static loom_low_lower_rule_operand_masks_t loom_low_lower_rule_operand_masks(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  loom_low_lower_rule_operand_masks_t masks = {0};
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    const loom_low_lower_value_ref_t* ref =
        &rule_set->value_refs[emit->operand_ref_start + i];
    if (loom_low_lower_rule_value_ref_needs_facts(ref)) {
      masks.source_values |= (uint8_t)(1u << i);
    }
    if (ref->materializer_index &&
        loom_low_lower_rule_value_materializer(rule_set, ref)->prepare) {
      masks.materializers |= (uint8_t)(1u << i);
    }
  }
  return masks;
}

// Changes a carrier width only when doing so cannot silently invent or discard
// a semantic register value type. An unchanged width preserves the complete
// type; a changed carrier-only type remains carrier-only. Typed width changes
// require a target-owned relation and are rejected by the caller.
static bool loom_low_lower_rule_try_register_type_with_unit_count(
    loom_type_t type, uint32_t unit_count, loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_ASSERT(loom_low_type_is_register(type));
  IREE_ASSERT_GT(unit_count, 0);
  if (unit_count == loom_low_register_type_unit_count(type)) {
    *out_type = type;
    return true;
  }
  if (loom_type_register_has_value_type(type)) {
    return false;
  }
  *out_type = loom_low_register_carrier_type_with_unit_count(type, unit_count);
  return true;
}

static uint8_t loom_low_lower_rule_emit_result_type_mask(
    const loom_low_lower_emit_t* emit) {
  return (uint8_t)((1u << emit->result_ref_count) - 1u);
}

static loom_scalar_type_t loom_low_lower_rule_type_pattern_element(
    const loom_low_lower_type_pattern_t* pattern) {
  IREE_ASSERT(iree_any_bit_set(pattern->flags,
                               LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT));
  const uint64_t element_type_mask = pattern->element_type_mask;
  IREE_ASSERT_NE(element_type_mask, 0u);
  IREE_ASSERT_EQ(element_type_mask & (element_type_mask - 1), 0u);
  uint32_t element_type = 0;
  uint64_t shifted_mask = element_type_mask;
  while ((shifted_mask & 1u) == 0u) {
    ++element_type;
    shifted_mask >>= 1;
  }
  return (loom_scalar_type_t)element_type;
}

static loom_type_t loom_low_lower_rule_type_pattern_exact_type(
    const loom_low_lower_type_pattern_t* pattern) {
  IREE_ASSERT(iree_all_bits_set(pattern->flags,
                                LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                                    LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT));
  const loom_scalar_type_t element_type =
      loom_low_lower_rule_type_pattern_element(pattern);
  if (pattern->type_kind == LOOM_TYPE_SCALAR) {
    return loom_type_scalar(element_type);
  }
  IREE_ASSERT(iree_all_bits_set(
      pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                          LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0));
  IREE_ASSERT_GE(pattern->shape.exact.dim0, 0);
  if (pattern->rank == 2) {
    IREE_ASSERT(iree_all_bits_set(
        pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1));
    IREE_ASSERT_GE(pattern->shape.exact.dim1, 0);
    return loom_type_shaped_2d(pattern->type_kind, element_type,
                               loom_dim_pack_static(pattern->shape.exact.dim0),
                               loom_dim_pack_static(pattern->shape.exact.dim1),
                               /*encoding_id=*/0);
  }
  IREE_ASSERT_EQ(pattern->rank, 1);
  return loom_type_shaped_1d(pattern->type_kind, element_type,
                             loom_dim_pack_static(pattern->shape.exact.dim0),
                             0);
}

static loom_value_id_t loom_low_lower_rule_plan_operand_source(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_source_t* source,
    const loom_low_lower_resolved_emit_t* resolved, uint16_t ordinal) {
  const uint16_t ref_index = resolved->emit->operand_ref_start + ordinal;
  return resolved->source_value_mask & (1u << ordinal)
             ? loom_low_lower_resolved_emit_source_value(resolved, ordinal)
             : loom_low_lower_rule_source_value_from_nodes(
                   context->module, rule_set, source->source_op,
                   source->source_nodes, source->source_node_count, ref_index);
}

static loom_type_t loom_low_lower_rule_plan_operand_type(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_source_t* source,
    const loom_low_lower_resolved_emit_t* resolved, uint16_t ordinal,
    const loom_type_id_t* temporary_types) {
  const uint16_t ref_index = resolved->emit->operand_ref_start + ordinal;
  const loom_low_lower_value_ref_t* ref = &rule_set->value_refs[ref_index];
  if (ref->kind == LOOM_LOW_LOWER_VALUE_REF_TEMPORARY) {
    return loom_type_table_get(&context->module->types,
                               temporary_types[ref->index]);
  }
  const loom_value_id_t value = loom_low_lower_rule_plan_operand_source(
      context, rule_set, source, resolved, ordinal);
  if (ref->materializer_index != 0) {
    return loom_low_lower_rule_value_materializer(rule_set, ref)
        ->result_type(context, value);
  }
  return loom_low_lower_value_binding_type(context, value);
}

static iree_status_t loom_low_lower_rule_plan_carriers(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_source_t* source,
    const loom_low_lower_resolved_emit_t* resolved,
    loom_type_id_t* temporary_types, loom_type_id_t* copy_type_ids,
    loom_type_id_t* result_type_ids, loom_low_lower_lane_plan_t* lanes) {
  const loom_low_lower_emit_t* emit = resolved->emit;
  const bool explicit_slice =
      emit->kind == LOOM_LOW_LOWER_EMIT_REGISTER_SLICE &&
      emit->payload.structural.unit_count != 0;
  const bool expands_lanes =
      emit->kind == LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE;
  const bool accumulates_lanes =
      emit->kind == LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES;
  const bool sequences_lanes =
      emit->kind == LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE;
  const bool first_lane =
      emit->kind == LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_FIRST_LANE;
  loom_type_id_t* lane_operand_types = NULL;
  loom_type_id_t* lane_result_types = NULL;
  if (lanes) {
    *lanes = (loom_low_lower_lane_plan_t){.lane_count = first_lane ? 1 : 0};
    lane_operand_types = (loom_type_id_t*)(lanes + 1);
    lane_result_types = lane_operand_types + emit->operand_ref_count;
  }
  loom_type_t operand_types[7];
  const bool needs_all_operands = explicit_slice || lanes != NULL;
  if (needs_all_operands || emit->copy_operand_mask) {
    uint16_t copy_index = 0;
    for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
      const bool copies_operand = (emit->copy_operand_mask & (1u << i)) != 0;
      if (!needs_all_operands && !copies_operand) {
        continue;
      }
      operand_types[i] = loom_low_lower_rule_plan_operand_type(
          context, rule_set, source, resolved, i, temporary_types);
      const uint32_t source_units =
          loom_low_register_type_unit_count(operand_types[i]);
      const uint32_t lane_units =
          expands_lanes || sequences_lanes
              ? loom_low_lower_rule_descriptor_packet_operand(
                    context->descriptor_set, resolved->descriptor.descriptor, i)
                    ->unit_count
              : 1;
      // Sequence operands are sliced before copying. Other forms copy the
      // aggregate, then derive the slice from that copy's selected class.
      loom_type_t lane_type = loom_type_none();
      if (sequences_lanes) {
        if (!loom_low_lower_rule_try_register_type_with_unit_count(
                operand_types[i], lane_units, &lane_type)) {
          return loom_low_lower_emit_register_width_relation_unsupported(
              context, source->source_op, operand_types[i], lane_units);
        }
        operand_types[i] = lane_type;
      }
      if (copies_operand) {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_copy_operand_type(
            context, resolved->descriptor.descriptor, i, operand_types[i],
            &operand_types[i]));
        IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
            context->module, operand_types[i], &copy_type_ids[copy_index++]));
      }
      if (!lanes) {
        continue;
      }
      if (!sequences_lanes &&
          !loom_low_lower_rule_try_register_type_with_unit_count(
              operand_types[i], lane_units, &lane_type)) {
        return loom_low_lower_emit_register_width_relation_unsupported(
            context, source->source_op, operand_types[i], lane_units);
      }
      if (source_units != lane_units) {
        lanes->slice_operand_mask |= 1u << i;
        IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
            context->module, lane_type, &lane_operand_types[i]));
      }
      const uint32_t operand_lane_count = source_units / lane_units;
      IREE_ASSERT_EQ(source_units % lane_units, 0);
      if (accumulates_lanes && i != emit->accumulator_operand_index) {
        IREE_ASSERT(lanes->lane_count == 0 ||
                    lanes->lane_count == operand_lane_count);
        lanes->lane_count = operand_lane_count;
      } else if (expands_lanes || sequences_lanes) {
        IREE_ASSERT(lanes->lane_count <= 1 || operand_lane_count == 1 ||
                    lanes->lane_count == operand_lane_count);
        lanes->lane_count = iree_max(lanes->lane_count, operand_lane_count);
      }
    }
  }
  if (accumulates_lanes) {
    const bool seed_first_lane = iree_any_bit_set(
        emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE);
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(
                       operand_types[emit->accumulator_operand_index]),
                   seed_first_lane ? lanes->lane_count : 1);
  }
  loom_type_t result_types[3];
  uint16_t remaining_mask = resolved->result_type_mask;
  while (remaining_mask != 0) {
    const uint16_t ordinal =
        (uint16_t)iree_math_count_trailing_zeros_u32(remaining_mask);
    remaining_mask &= (uint16_t)(remaining_mask - 1u);
    loom_type_t result_type = loom_type_none();
    if (explicit_slice) {
      if (!loom_low_lower_rule_try_register_type_with_unit_count(
              operand_types[0], emit->payload.structural.unit_count,
              &result_type)) {
        return loom_low_lower_emit_register_width_relation_unsupported(
            context, source->source_op, operand_types[0],
            emit->payload.structural.unit_count);
      }
    } else if (iree_any_bit_set(emit->flags,
                                LOOM_LOW_LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN)) {
      const loom_type_t exact_type =
          loom_low_lower_rule_type_pattern_exact_type(
              &rule_set->type_patterns[emit->result_type.type_pattern_start +
                                       ordinal]);
      IREE_RETURN_IF_ERROR(loom_low_lower_map_type(context, source->source_op,
                                                   exact_type, &result_type));
    } else if (iree_any_bit_set(
                   emit->flags,
                   LOOM_LOW_LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE)) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
          context, resolved->descriptor.descriptor, ordinal, &result_type));
    } else {
      const uint16_t ref_index = emit->result_type.value_ref_start + ordinal;
      const loom_low_lower_value_ref_t* ref = &rule_set->value_refs[ref_index];
      if (ref->kind == LOOM_LOW_LOWER_VALUE_REF_TEMPORARY) {
        result_type = loom_type_table_get(&context->module->types,
                                          temporary_types[ref->index]);
      } else {
        const loom_value_id_t source_value =
            loom_low_lower_rule_plan_source_value(context, rule_set, source,
                                                  ref_index);
        IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
            context,
            loom_low_lower_rule_source_op(rule_set, source->source_op,
                                          source->source_nodes,
                                          source->source_node_count, ref_index),
            source_value, &result_type));
      }
    }
    if (context->result->error_count != 0) {
      return iree_ok_status();
    }
    IREE_ASSERT(loom_low_type_is_register(result_type));
    result_types[ordinal] = result_type;
    if (expands_lanes) {
      const uint32_t packet_units =
          loom_low_lower_rule_descriptor_result_operand(
              context->descriptor_set, resolved->descriptor.descriptor, ordinal)
              ->unit_count;
      const uint32_t result_units =
          loom_low_register_type_unit_count(result_type);
      IREE_ASSERT_EQ(result_units % packet_units, 0);
      lanes->lane_count =
          iree_max(lanes->lane_count, result_units / packet_units);
    }
  }
  // Packet-width inputs may be broadcast across a wider result. Resolve every
  // aggregate carrier before binding any result so all bindings share the final
  // repetition count, including descriptors with several results.
  remaining_mask = resolved->result_type_mask;
  uint16_t result_index = 0;
  while (remaining_mask != 0) {
    const uint16_t ordinal =
        (uint16_t)iree_math_count_trailing_zeros_u32(remaining_mask);
    remaining_mask &= (uint16_t)(remaining_mask - 1u);
    const loom_type_t result_type = result_types[ordinal];
    const uint16_t bind_ref_index =
        loom_low_lower_rule_emit_result_bind_ref_index(emit, ordinal);
    const loom_low_lower_value_ref_t* bind_ref =
        &rule_set->value_refs[bind_ref_index];
    loom_type_t bound_type = result_type;
    if (lanes) {
      const uint32_t packet_units =
          expands_lanes || sequences_lanes
              ? loom_low_lower_rule_descriptor_result_operand(
                    context->descriptor_set, resolved->descriptor.descriptor,
                    ordinal)
                    ->unit_count
              : 1;
      loom_type_t packet_type = loom_type_none();
      if (!loom_low_lower_rule_try_register_type_with_unit_count(
              result_type, packet_units, &packet_type)) {
        return loom_low_lower_emit_register_width_relation_unsupported(
            context, source->source_op, result_type, packet_units);
      }
      IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
          context->module, packet_type, &lane_result_types[ordinal]));
      if (expands_lanes) {
        const uint32_t bound_units = packet_units * lanes->lane_count;
        if (!loom_low_lower_rule_try_register_type_with_unit_count(
                result_type, bound_units, &bound_type)) {
          return loom_low_lower_emit_register_width_relation_unsupported(
              context, source->source_op, result_type, bound_units);
        }
      } else if (sequences_lanes) {
        const uint32_t result_units =
            loom_low_register_type_unit_count(result_type);
        const uint32_t result_lane_count = result_units / packet_units;
        IREE_ASSERT_EQ(result_units % packet_units, 0);
        IREE_ASSERT(lanes->lane_count <= 1 || result_lane_count == 1 ||
                    lanes->lane_count == result_lane_count);
        if (bind_ref->kind == LOOM_LOW_LOWER_VALUE_REF_TEMPORARY) {
          bound_type = packet_type;
          lanes->lane_count = iree_max(lanes->lane_count, result_lane_count);
        } else {
          IREE_ASSERT(lanes->lane_count <= 1 ||
                      lanes->lane_count == result_lane_count);
          lanes->lane_count = result_lane_count;
        }
      } else if (accumulates_lanes && lanes->lane_count == 1 &&
                 iree_any_bit_set(
                     emit->flags,
                     LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE |
                         LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE)) {
        bound_type = operand_types[emit->accumulator_operand_index];
      }
    }
    loom_type_id_t type_id;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_type_id(context->module, bound_type, &type_id));
    result_type_ids[result_index++] = type_id;
    if (bind_ref->kind == LOOM_LOW_LOWER_VALUE_REF_TEMPORARY) {
      temporary_types[bind_ref->index] = type_id;
    } else {
      const loom_value_id_t result =
          loom_low_lower_rule_source_value_from_nodes(
              context->module, rule_set, source->source_op,
              source->source_nodes, source->source_node_count, bind_ref_index);
      loom_low_lower_plan_value_type_id(context, result, type_id);
    }
  }
  return iree_ok_status();
}

static const loom_low_lower_source_memory_address_materializer_t*
loom_low_lower_rule_emit_address_materializer(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  if (!emit->source_memory_ordinal) {
    return NULL;
  }
  const loom_low_lower_source_memory_t* source_memory =
      &rule_set->source_memories[emit->source_memory_ordinal - 1];
  return source_memory->address_materializer_ordinal
             ? loom_low_lower_rule_set_source_memory_address_materializer(
                   rule_set, source_memory)
             : NULL;
}

static uint32_t loom_low_lower_rule_emit_data_size(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit, uint8_t result_type_mask,
    loom_low_lower_rule_operand_masks_t operand_masks) {
  return (uint32_t)iree_host_align(
      loom_low_lower_rule_materializer_data_size(operand_masks.materializers) +
          emit->attr_copy_count * sizeof(loom_named_attr_t) +
          loom_low_lower_rule_read_only_attributes_size(emit) +
          iree_math_count_ones_u32(emit->copy_operand_mask) *
              sizeof(loom_type_id_t) +
          iree_math_count_ones_u32(result_type_mask) * sizeof(loom_type_id_t) +
          loom_low_lower_rule_lane_plan_size(emit) +
          iree_math_count_ones_u32(operand_masks.source_values) *
              sizeof(loom_value_id_t) +
          (loom_low_lower_rule_emit_address_materializer(rule_set, emit)
               ? sizeof(loom_type_id_t)
               : 0),
      iree_alignof(loom_named_attr_t));
}

static iree_status_t loom_low_lower_rule_plan_alias_types(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  for (uint16_t i = 0; i < rule->metadata.value.alias_ref_count; ++i) {
    const uint16_t source_ref =
        (uint16_t)(rule->action.alias_ref_start + i * 2);
    const uint16_t result_ref = (uint16_t)(source_ref + 1);
    if (iree_any_bit_set(rule->flags,
                         LOOM_LOW_LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS)) {
      const loom_value_slice_t sources =
          loom_low_lower_rule_value_ref_field_span(context->module, rule_set,
                                                   source_op, source_ref);
      const loom_value_slice_t results =
          loom_low_lower_rule_value_ref_field_span(context->module, rule_set,
                                                   source_op, result_ref);
      for (iree_host_size_t ordinal = 0; ordinal < sources.count; ++ordinal) {
        IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
            context, results.values[ordinal],
            loom_low_lower_value_binding_type(context,
                                              sources.values[ordinal])));
      }
    } else {
      const loom_value_id_t source = loom_low_lower_rule_source_value(
          context->module, rule_set, source_op, source_ref);
      const loom_value_id_t result = loom_low_lower_rule_source_value(
          context->module, rule_set, source_op, result_ref);
      IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
          context, result, loom_low_lower_value_binding_type(context, source)));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_plan_finalize(
    loom_low_lower_context_t* context,
    loom_low_lower_selected_plan_t* selected_plan) {
  const loom_low_lower_rule_set_t* rule_set = selected_plan->rule_set;
  const loom_low_lower_rule_t* rule = selected_plan->rule;
  const loom_op_t* source_op = selected_plan->source_op;
  const loom_low_source_memory_access_plan_t* source_memory_access =
      selected_plan->source_memory_access;
  const bool elided = iree_any_bit_set(selected_plan->flags,
                                       LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED);
  selected_plan->resolved_emits = NULL;
  if (rule->emit_count == 0) {
    return elided ? iree_ok_status()
                  : loom_low_lower_rule_plan_alias_types(context, rule_set,
                                                         source_op, rule);
  }
  loom_type_id_t* temporary_types = NULL;
  if (!elided) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_planning_array(
        context, rule->temporary_count, sizeof(*temporary_types),
        (void**)&temporary_types));
  }
  // Table counts bound a rule to 65535 emits with at most 31 attributes and
  // three result carriers, seven copied carriers, and seven source references
  // each, plus seven optional materializer recipe pointers. Read-only
  // attributes add at most 32 u32 words and lane plans add at most 12 u32
  // words per row. The allocation and row-relative offsets fit in
  // u32. Payload alignment permits the next row to carry named attributes.
  const uint32_t rows_size = (uint32_t)iree_host_align(
      rule->emit_count * sizeof(loom_low_lower_resolved_emit_t),
      iree_alignof(loom_named_attr_t));
  uint32_t allocation_size = rows_size;
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const loom_low_lower_emit_t* emit = loom_low_lower_rule_set_emit_at(
        rule_set, (uint16_t)(rule->action.emit_start + i));
    if (!elided) {
      allocation_size += loom_low_lower_rule_emit_data_size(
          rule_set, emit, loom_low_lower_rule_emit_result_type_mask(emit),
          loom_low_lower_rule_operand_masks(rule_set, emit));
    }
  }
  loom_low_lower_resolved_emit_t* resolved_emits = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, allocation_size, (void**)&resolved_emits));
  uint8_t* data = (uint8_t*)resolved_emits + rows_size;
  const loom_low_lower_rule_source_t source = {
      .source_op = source_op,
      .source_nodes = selected_plan->data.source_nodes,
      .source_node_count = selected_plan->source_node_count,
  };
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, /*view_regions=*/NULL, /*source_memory_state=*/NULL,
      &match_context);
  match_context.policy_rule_set_ordinal =
      (uint16_t)(selected_plan->rule_set_index + 1u);
  uint32_t sequence_lane_count = 0;
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const loom_low_lower_emit_t* emit = loom_low_lower_rule_set_emit_at(
        rule_set, (uint16_t)(rule->action.emit_start + i));
    loom_low_lower_resolved_emit_t* resolved = &resolved_emits[i];
    *resolved = (loom_low_lower_resolved_emit_t){.emit = emit};
    if (emit->descriptor_ref != LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
      const loom_low_descriptor_t* descriptor = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
          &match_context, rule_set, emit->descriptor_ref, &descriptor));
      IREE_ASSERT(descriptor != NULL,
                  "generated target-low rule references a missing descriptor");
      resolved->descriptor.descriptor = descriptor;
      resolved->access_flags = loom_low_lower_resolve_emit_access_flags(
          context->descriptor_set, descriptor, emit,
          source_memory_access ? source_memory_access->access_flags : 0);
    }
    if (elided) {
      continue;
    }
    resolved->result_type_mask =
        loom_low_lower_rule_emit_result_type_mask(emit);
    const loom_low_lower_rule_operand_masks_t operand_masks =
        loom_low_lower_rule_operand_masks(rule_set, emit);
    resolved->source_value_mask = operand_masks.source_values;
    resolved->materializer_mask = operand_masks.materializers;
    const uint32_t data_size = loom_low_lower_rule_emit_data_size(
        rule_set, emit, resolved->result_type_mask, operand_masks);
    if (data_size != 0) {
      resolved->data_offset = (uint32_t)(data - (uint8_t*)resolved);
    }
    const void** materializations = (const void**)data;
    uint8_t* attributes = data + loom_low_lower_rule_materializer_data_size(
                                     resolved->materializer_mask);
    uint8_t* attribute_payload_end =
        attributes + emit->attr_copy_count * sizeof(loom_named_attr_t);
    loom_low_lower_read_only_attributes_t* read_only_attributes = NULL;
    if (emit->has_read_only_data_attributes) {
      read_only_attributes =
          (loom_low_lower_read_only_attributes_t*)attribute_payload_end;
      read_only_attributes->attribute_mask = 0;
    }
    if (emit->attr_copy_count != 0) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
          context, rule_set, source_op, &source, emit,
          emit->source_memory_ordinal ? source_memory_access : NULL,
          (loom_named_attr_t*)attributes, read_only_attributes));
    }
    loom_type_id_t* copy_types =
        (loom_type_id_t*)(attribute_payload_end +
                          loom_low_lower_rule_read_only_attributes_size(emit));
    loom_type_id_t* result_types =
        copy_types + iree_math_count_ones_u32(emit->copy_operand_mask);
    uint8_t* lane_payload =
        (uint8_t*)(result_types +
                   iree_math_count_ones_u32(resolved->result_type_mask));
    const uint32_t lane_plan_size = loom_low_lower_rule_lane_plan_size(emit);
    loom_low_lower_lane_plan_t* lanes =
        lane_plan_size ? (loom_low_lower_lane_plan_t*)lane_payload : NULL;
    loom_value_id_t* source_values =
        (loom_value_id_t*)(lane_payload + lane_plan_size);
    uint16_t source_value_index = 0;
    uint16_t remaining_mask = resolved->source_value_mask;
    while (remaining_mask != 0) {
      const uint16_t ordinal =
          (uint16_t)iree_math_count_trailing_zeros_u32(remaining_mask);
      remaining_mask &= (uint16_t)(remaining_mask - 1u);
      const uint16_t ref_index = emit->operand_ref_start + ordinal;
      source_values[source_value_index++] =
          loom_low_lower_rule_plan_source_value(context, rule_set, &source,
                                                ref_index);
    }
    uint16_t materializer_index = 0;
    remaining_mask = resolved->materializer_mask;
    while (remaining_mask != 0) {
      const uint16_t ordinal =
          (uint16_t)iree_math_count_trailing_zeros_u32(remaining_mask);
      remaining_mask &= (uint16_t)(remaining_mask - 1u);
      const loom_low_lower_value_materializer_t* materializer =
          loom_low_lower_rule_value_materializer(
              rule_set,
              &rule_set->value_refs[emit->operand_ref_start + ordinal]);
      const loom_value_id_t value = loom_low_lower_rule_plan_operand_source(
          context, rule_set, &source, resolved, ordinal);
      IREE_RETURN_IF_ERROR(materializer->prepare(
          context, value, &materializations[materializer_index++]));
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_plan_carriers(
        context, rule_set, &source, resolved, temporary_types, copy_types,
        result_types, lanes));
    if (context->result->error_count != 0) {
      return iree_ok_status();
    }
    if (emit->kind == LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE) {
      IREE_ASSERT(sequence_lane_count <= 1 || lanes->lane_count == 1 ||
                  sequence_lane_count == lanes->lane_count);
      sequence_lane_count = iree_max(sequence_lane_count, lanes->lane_count);
      if (i + 1 == rule->emit_count) {
        IREE_ASSERT_EQ(sequence_lane_count, lanes->lane_count);
      }
    }
    const loom_low_lower_source_memory_address_materializer_t* materializer =
        loom_low_lower_rule_emit_address_materializer(rule_set, emit);
    if (materializer) {
      const loom_scalar_type_t coordinate_type =
          materializer->coordinate_type ==
                  LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_INDEX
              ? LOOM_SCALAR_TYPE_INDEX
              : LOOM_SCALAR_TYPE_OFFSET;
      loom_type_t low_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
          context, source_op, loom_type_scalar(coordinate_type), &low_type));
      if (context->result->error_count != 0) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
          context->module, low_type,
          (loom_type_id_t*)(source_values + source_value_index)));
    }
    data += data_size;
  }
  selected_plan->resolved_emits = resolved_emits;
  return iree_ok_status();
}
