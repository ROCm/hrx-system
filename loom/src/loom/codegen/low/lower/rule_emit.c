// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_emit.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/rule_descriptor.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/lower/rule_value.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/registers.h"

typedef struct loom_low_lower_rule_emit_state_t {
  // Rule-local low SSA values captured by earlier emit rows.
  loom_value_id_t* temporaries;
  // Number of entries in temporaries.
  uint16_t temporary_count;
} loom_low_lower_rule_emit_state_t;

static iree_status_t loom_low_lower_rule_emit_state_initialize(
    loom_low_lower_context_t* context, const loom_low_lower_rule_t* rule,
    loom_low_lower_rule_emit_state_t* out_state) {
  *out_state = (loom_low_lower_rule_emit_state_t){
      .temporaries = NULL,
      .temporary_count = rule->temporary_count,
  };
  if (rule->temporary_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, rule->temporary_count, sizeof(*out_state->temporaries),
      (void**)&out_state->temporaries));
  for (uint16_t i = 0; i < rule->temporary_count; ++i) {
    out_state->temporaries[i] = LOOM_VALUE_ID_INVALID;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_low_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    uint16_t value_ref_index, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
      loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
          context->module, rule_set, source_op, value_ref_index);
      if (value_ref->materializer_index != 0) {
        const loom_low_lower_value_materializer_t* materializer =
            loom_low_lower_rule_value_materializer(rule_set, value_ref);
        return materializer->materialize(context, source_op, source_value_id,
                                         out_low_value_id);
      }
      return loom_low_lower_lookup_value(context, source_value_id,
                                         out_low_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_LT(value_ref->index, state->temporary_count);
      IREE_ASSERT(state->temporaries != NULL);
      IREE_ASSERT(state->temporaries[value_ref->index] !=
                  LOOM_VALUE_ID_INVALID);
      *out_low_value_id = state->temporaries[value_ref->index];
      return iree_ok_status();
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM: {
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT(source_memory_access != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      IREE_ASSERT_LT(value_ref->index,
                     source_memory_access->dynamic_term_count);
      const loom_value_id_t source_value_id =
          source_memory_access->dynamic_terms[value_ref->index].index;
      return loom_low_lower_lookup_value(context, source_value_id,
                                         out_low_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      return loom_low_lower_rule_materialize_source_memory_byte_offset(
          context, rule_set, source_op, source_memory, source_memory_access,
          out_low_value_id);
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS: {
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT(source_memory_access != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      return loom_low_lower_rule_materialize_source_memory_address(
          context, rule_set, source_op, source_memory, source_memory_access,
          out_low_value_id);
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
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
  IREE_ASSERT_GE(pattern->static_dim0, 0);
  if (pattern->rank == 2) {
    IREE_ASSERT(iree_all_bits_set(
        pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1));
    IREE_ASSERT_GE(pattern->static_dim1, 0);
    return loom_type_shaped_2d(pattern->type_kind, element_type,
                               loom_dim_pack_static(pattern->static_dim0),
                               loom_dim_pack_static(pattern->static_dim1),
                               /*encoding_id=*/0);
  }
  IREE_ASSERT_EQ(pattern->rank, 1);
  return loom_type_shaped_1d(pattern->type_kind, element_type,
                             loom_dim_pack_static(pattern->static_dim0), 0);
}

static double loom_low_lower_rule_attr_copy_exact_float(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_attr_copy_t* attr_copy) {
  const loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
      context->module, rule_set, source_op, attr_copy->value_ref_index);
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
  return value;
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
  IREE_ASSERT(iree_checked_sub_i64(source_value, attr_copy->literal_i64,
                                   &projected_value));
  return projected_value;
}

iree_status_t loom_low_lower_rule_set_resolve_emit_program(
    loom_low_lower_context_t* context, uint16_t rule_set_index,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t** out_resolved_emits) {
  *out_resolved_emits = NULL;
  if (rule->emit_count == 0) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_emit_t* resolved_emits = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, rule->emit_count, sizeof(*resolved_emits),
      (void**)&resolved_emits));
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, /*view_regions=*/NULL, /*source_memory_state=*/NULL,
      &match_context);
  match_context.policy_rule_set_ordinal = (uint16_t)(rule_set_index + 1u);
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    resolved_emits[i].emit = emit;
    resolved_emits[i].descriptor = (loom_low_lower_resolved_descriptor_t){0};
    if (emit->descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
      continue;
    }
    const loom_low_descriptor_t* descriptor = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
        &match_context, rule_set, emit->descriptor_ref, &descriptor));
    IREE_ASSERT(descriptor != NULL,
                "generated target-low rule references a missing descriptor");
    resolved_emits[i].descriptor.descriptor = descriptor;
  }
  *out_resolved_emits = resolved_emits;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_build_attrs(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  if (emit->attr_copy_count == 0) {
    return iree_ok_status();
  }
  const loom_attribute_t* source_attrs = loom_op_const_attrs(source_op);
  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->attr_copy_count, sizeof(*attrs), (void**)&attrs));
  for (uint16_t i = 0; i < emit->attr_copy_count; ++i) {
    uint16_t attr_copy_index = (uint16_t)(emit->attr_copy_start + i);
    const loom_low_lower_attr_copy_t* attr_copy =
        &rule_set->attr_copies[attr_copy_index];
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        loom_low_lower_context_module(context),
        loom_low_lower_rule_set_string(rule_set,
                                       attr_copy->target_name_string_offset),
        &attrs[i].name_id));
    switch (attr_copy->kind) {
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
        IREE_ASSERT_LE(packed_bit_count + attr_copy->target_bit_offset, 63);
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
      case LOOM_LOW_LOWER_ATTR_COPY_I64_ATTRS_PACK_CONSECUTIVE: {
        IREE_ASSERT_LT(attr_copy->source_attr_index,
                       source_op->attribute_count);
        IREE_ASSERT_GT(attr_copy->source_element_count, 0);
        IREE_ASSERT_GT(attr_copy->source_element_bit_width, 0);
        const uint32_t packed_bit_count =
            (uint32_t)attr_copy->source_element_count *
            attr_copy->source_element_bit_width;
        IREE_ASSERT_LE(packed_bit_count + attr_copy->target_bit_offset, 63);
        IREE_ASSERT_LE((uint32_t)attr_copy->source_attr_index +
                           attr_copy->source_element_count,
                       source_op->attribute_count);
        const uint64_t element_mask =
            (UINT64_C(1) << attr_copy->source_element_bit_width) - 1u;
        uint64_t packed_value = 0;
        for (uint16_t j = 0; j < attr_copy->source_element_count; ++j) {
          loom_attribute_t source_attr =
              source_attrs[attr_copy->source_attr_index + j];
          IREE_ASSERT_EQ(source_attr.kind, LOOM_ATTR_I64);
          const int64_t source_value = source_attr.i64;
          IREE_ASSERT_GE(source_value, 0);
          IREE_ASSERT_LE((uint64_t)source_value, element_mask);
          packed_value |= (uint64_t)source_value
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
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(context->module, rule_set,
                                             source_op,
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
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_NEGATE: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(context->module, rule_set,
                                             source_op,
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
            loom_low_lower_rule_source_value(context->module, rule_set,
                                             source_op,
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
            loom_low_lower_rule_source_value(context->module, rule_set,
                                             source_op,
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
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_SHIFT: {
        const loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value(context->module, rule_set,
                                             source_op,
                                             attr_copy->value_ref_index);
        loom_low_lower_u32_divisor_magic_info_t info = {0};
        const bool has_magic_info =
            loom_low_lower_rule_value_facts_u32_divisor_magic_info(
                loom_low_lower_context_module(context),
                loom_low_lower_context_fact_table(context), source_value_id,
                &info);
        IREE_ASSERT(has_magic_info);
        uint64_t projected_value =
            attr_copy->kind ==
                    LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER
                ? info.multiplier
                : info.post_shift;
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
            loom_low_lower_rule_source_value(context->module, rule_set,
                                             source_op,
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
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F16_BITS: {
        const float f32_value =
            (float)loom_low_lower_rule_attr_copy_exact_float(
                context, rule_set, source_op, attr_copy);
        const uint16_t bit_pattern = iree_math_f32_to_f16(f32_value);
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_BF16_BITS: {
        const float f32_value =
            (float)loom_low_lower_rule_attr_copy_exact_float(
                context, rule_set, source_op, attr_copy);
        const uint16_t bit_pattern = iree_math_f32_to_bf16(f32_value);
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F32_BITS: {
        const float f32_value =
            (float)loom_low_lower_rule_attr_copy_exact_float(
                context, rule_set, source_op, attr_copy);
        uint32_t bit_pattern = 0;
        memcpy(&bit_pattern, &f32_value, sizeof(bit_pattern));
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F64_BITS: {
        const double f64_value = loom_low_lower_rule_attr_copy_exact_float(
            context, rule_set, source_op, attr_copy);
        uint64_t bit_pattern = 0;
        memcpy(&bit_pattern, &f64_value, sizeof(bit_pattern));
        loom_low_lower_rule_set_projected_bits_attr(attr_copy, bit_pattern,
                                                    &attrs[i]);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET:
        attrs[i].value =
            loom_attr_i64(source_memory_access->static_byte_offset);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_PLUS_LITERAL: {
        attrs[i].value = loom_attr_i64(
            source_memory_access->static_byte_offset + attr_copy->literal_i64);
        break;
      }
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT:
        IREE_ASSERT_GT(attr_copy->literal_i64, 0);
        attrs[i].value = loom_attr_i64(
            source_memory_access->static_byte_offset / attr_copy->literal_i64);
        break;
      case LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER:
        IREE_ASSERT_GT(attr_copy->literal_i64, 0);
        attrs[i].value = loom_attr_i64(
            source_memory_access->static_byte_offset % attr_copy->literal_i64);
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
  *out_attrs = loom_make_named_attr_slice(attrs, emit->attr_copy_count);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_map_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source_value_id, out_type));
  IREE_ASSERT(loom_low_type_is_register(*out_type));
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_build_low_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_emit_t* emit,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t** out_operands) {
  *out_operands = NULL;
  if (emit->operand_ref_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->operand_ref_count, sizeof(*low_operands),
      (void**)&low_operands));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_low_value(
        context, rule_set, source_op, state, source_memory,
        source_memory_access, (uint16_t)(emit->operand_ref_start + i),
        &low_operands[i]));
  }
  *out_operands = low_operands;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_copy_low_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    loom_value_id_t* low_operands) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  if (emit->copy_operand_mask == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(emit->operand_ref_count, 16);
  const uint16_t valid_operand_mask =
      emit->operand_ref_count == 16
          ? UINT16_MAX
          : (uint16_t)(((uint16_t)1u << emit->operand_ref_count) - 1u);
  IREE_ASSERT_FALSE(
      iree_any_bit_set(emit->copy_operand_mask, (uint16_t)~valid_operand_mask));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    const uint16_t operand_bit = (uint16_t)((uint16_t)1u << i);
    if (!iree_any_bit_set(emit->copy_operand_mask, operand_bit)) {
      continue;
    }
    const loom_type_t source_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    loom_type_t copy_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_copy_operand_type(
        context, resolved_emit->descriptor.descriptor, i, source_type,
        &copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_copy_build(
        loom_low_lower_context_builder(context), low_operands[i], false,
        copy_type, source_op->location, &copy_op));
    low_operands[i] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static void loom_low_lower_rule_apply_operand_flags(
    const loom_low_lower_emit_t* emit, loom_value_id_t* low_operands) {
  if (iree_any_bit_set(emit->flags,
                       LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1)) {
    IREE_ASSERT_GE(emit->operand_ref_count, 2);
    const loom_value_id_t temporary = low_operands[0];
    low_operands[0] = low_operands[1];
    low_operands[1] = temporary;
  }
}

static iree_status_t loom_low_lower_rule_build_result_types(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  if (emit->result_ref_count == 0) {
    return iree_ok_status();
  }
  const bool use_type_patterns = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN);
  const bool use_descriptor_types = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE);
  IREE_ASSERT_FALSE(use_type_patterns && use_descriptor_types);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->result_ref_count, sizeof(*result_types),
      (void**)&result_types));
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    if (use_type_patterns) {
      const uint16_t type_pattern_index =
          (uint16_t)(emit->result_type_pattern_start + i);
      const loom_type_t exact_type =
          loom_low_lower_rule_type_pattern_exact_type(
              &rule_set->type_patterns[type_pattern_index]);
      IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
          context, source_op, exact_type, &result_types[i]));
      IREE_ASSERT(loom_low_type_is_register(result_types[i]));
    } else if (use_descriptor_types) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_result_type(
          context, resolved_emit->descriptor.descriptor, i, &result_types[i]));
    } else {
      loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
          context->module, rule_set, source_op,
          (uint16_t)(emit->result_ref_start + i));
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_map_result_type(
          context, source_op, source_value_id, &result_types[i]));
    }
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static uint16_t loom_low_lower_rule_emit_result_bind_ref_index(
    const loom_low_lower_emit_t* emit, uint16_t result_ordinal) {
  IREE_ASSERT_LT(result_ordinal, emit->result_ref_count);
  const uint16_t result_bind_ref_start =
      iree_any_bit_set(emit->flags,
                       LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS)
          ? emit->result_bind_ref_start
          : emit->result_ref_start;
  return (uint16_t)(result_bind_ref_start + result_ordinal);
}

static iree_status_t loom_low_lower_rule_bind_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state, const loom_low_lower_emit_t* emit,
    const loom_value_id_t* low_results) {
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    const uint16_t value_ref_index =
        loom_low_lower_rule_emit_result_bind_ref_index(emit, i);
    const loom_low_lower_value_ref_t* value_ref =
        &rule_set->value_refs[value_ref_index];
    switch (value_ref->kind) {
      case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
        loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
            context->module, rule_set, source_op, value_ref_index);
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(context, source_value_id,
                                                       low_results[i]));
        break;
      }
      case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
        IREE_ASSERT_LT(value_ref->index, state->temporary_count);
        IREE_ASSERT(state->temporaries != NULL);
        IREE_ASSERT(state->temporaries[value_ref->index] ==
                        LOOM_VALUE_ID_INVALID ||
                    state->temporaries[value_ref->index] == low_results[i]);
        state->temporaries[value_ref->index] = low_results[i];
        break;
      case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      default:
        IREE_ASSERT_UNREACHABLE("result binding must target result refs");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_elide_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  for (uint16_t i = 0; i < rule->elide_ref_count; ++i) {
    const uint16_t value_ref_index = (uint16_t)(rule->elide_ref_start + i);
    const loom_low_lower_value_ref_t* value_ref =
        &rule_set->value_refs[value_ref_index];
    IREE_ASSERT_EQ(value_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
    loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
        context->module, rule_set, source_op, value_ref_index);
    IREE_RETURN_IF_ERROR(loom_low_lower_elide_value(context, source_value_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_bind_aliases(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  if (iree_all_bits_set(rule->flags,
                        LOOM_LOW_LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS)) {
    IREE_ASSERT_EQ(rule->alias_ref_count, 1);
    const uint16_t source_ref_index = rule->alias_ref_start;
    const uint16_t result_ref_index = (uint16_t)(source_ref_index + 1);
    const loom_low_lower_value_ref_t* source_ref =
        &rule_set->value_refs[source_ref_index];
    const loom_low_lower_value_ref_t* result_ref =
        &rule_set->value_refs[result_ref_index];
    IREE_ASSERT_EQ(source_ref->kind, LOOM_LOW_LOWER_VALUE_REF_OPERAND);
    IREE_ASSERT_EQ(result_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
    const loom_value_slice_t source_span =
        loom_low_lower_rule_value_ref_field_span(context->module, rule_set,
                                                 source_op, source_ref_index);
    const loom_value_slice_t result_span =
        loom_low_lower_rule_value_ref_field_span(context->module, rule_set,
                                                 source_op, result_ref_index);
    IREE_ASSERT_EQ(source_span.count, result_span.count);
    for (iree_host_size_t i = 0; i < source_span.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value_alias(
          context, source_span.values[i], result_span.values[i]));
    }
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < rule->alias_ref_count; ++i) {
    const uint16_t source_ref_index = (uint16_t)(rule->alias_ref_start + i * 2);
    const uint16_t result_ref_index = (uint16_t)(source_ref_index + 1);
    const loom_low_lower_value_ref_t* source_ref =
        &rule_set->value_refs[source_ref_index];
    const loom_low_lower_value_ref_t* result_ref =
        &rule_set->value_refs[result_ref_index];
    IREE_ASSERT_EQ(source_ref->kind, LOOM_LOW_LOWER_VALUE_REF_OPERAND);
    IREE_ASSERT_EQ(result_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
    loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
        context->module, rule_set, source_op, source_ref_index);
    loom_value_id_t result_value_id = loom_low_lower_rule_source_value(
        context->module, rule_set, source_op, result_ref_index);
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value_alias(
        context, source_value_id, result_value_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_emit_descriptor_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, NULL, &attrs));

  loom_op_t* low_const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &resolved_emit->descriptor, attrs, result_types[0],
      source_op->location, &low_const_op));
  const loom_value_id_t low_result = loom_low_const_result(low_const_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    const loom_low_source_memory_access_plan_t* source_memory_access) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  const loom_low_lower_source_memory_t* source_memory = NULL;
  const loom_low_source_memory_access_plan_t* emit_source_memory_access = NULL;
  if (emit->source_memory_ordinal != 0) {
    const uint16_t source_memory_index =
        (uint16_t)(emit->source_memory_ordinal - 1);
    source_memory = &rule_set->source_memories[source_memory_index];
    IREE_ASSERT(source_memory_access != NULL);
    emit_source_memory_access = source_memory_access;
  }

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, source_memory,
      emit_source_memory_access, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, low_operands));
  loom_low_lower_rule_apply_operand_flags(emit, low_operands);

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, emit_source_memory_access, &attrs));

  const loom_tied_result_t* tied_results = NULL;
  if (emit->tied_result_count != 0) {
    tied_results = &rule_set->tied_results[emit->tied_result_start];
  }

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_emit->descriptor, low_operands,
      emit->operand_ref_count, attrs, result_types, emit->result_ref_count,
      tied_results, emit->tied_result_count, source_op->location, &low_op));
  loom_value_slice_t low_results = loom_low_op_results(low_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results.values);
}

static iree_status_t loom_low_lower_rule_emit_register_slice(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(emit->descriptor_ref, LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE);
  IREE_ASSERT_EQ(emit->operand_ref_count, 1);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, NULL, NULL, &low_operands));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));

  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_slice_build(loom_low_lower_context_builder(context),
                           low_operands[0], emit->structural_offset,
                           result_types[0], source_op->location, &slice_op));
  const loom_value_id_t low_result = loom_low_slice_result(slice_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_register_concat(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(emit->descriptor_ref, LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE);
  IREE_ASSERT_GT(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, NULL, NULL, &low_operands));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            low_operands, emit->operand_ref_count,
                            result_types[0], source_op->location, &concat_op));
  const loom_value_id_t low_result = loom_low_concat_result(concat_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static const loom_tied_result_t* loom_low_lower_rule_emit_tied_results(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  return emit->tied_result_count == 0
             ? NULL
             : &rule_set->tied_results[emit->tied_result_start];
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

static iree_status_t loom_low_lower_rule_slice_register_units(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value_id, uint32_t unit_offset, uint32_t unit_count,
    loom_value_id_t* out_slice_value_id) {
  *out_slice_value_id = LOOM_VALUE_ID_INVALID;
  const loom_type_t low_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_value_id);
  IREE_ASSERT(loom_low_type_is_register(low_type));
  const uint32_t total_unit_count = loom_low_register_type_unit_count(low_type);
  IREE_ASSERT_GT(unit_count, 0);
  IREE_ASSERT_LE(unit_offset, total_unit_count);
  IREE_ASSERT_LE(unit_count, total_unit_count - unit_offset);
  if (unit_offset == 0 && unit_count == total_unit_count) {
    *out_slice_value_id = low_value_id;
    return iree_ok_status();
  }
  loom_type_t slice_type = loom_type_none();
  if (!loom_low_lower_rule_try_register_type_with_unit_count(
          low_type, unit_count, &slice_type)) {
    return loom_low_lower_emit_register_width_relation_unsupported(
        context, source_op, low_type, unit_count);
  }
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_value_id, unit_offset,
      slice_type, source_op->location, &slice_op));
  *out_slice_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_slice_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value_id, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_lane_value_id) {
  *out_lane_value_id = LOOM_VALUE_ID_INVALID;
  const loom_type_t low_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_value_id);
  IREE_ASSERT(loom_low_type_is_register(low_type));
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(low_type),
                 loom_low_register_type_descriptor_set_stable_id(lane_type));
  IREE_ASSERT_EQ(loom_low_register_type_class_id(low_type),
                 loom_low_register_type_class_id(lane_type));
  IREE_ASSERT_LT(lane_index, loom_low_register_type_unit_count(low_type));
  if (loom_low_register_type_unit_count(low_type) == 1) {
    *out_lane_value_id = low_value_id;
    return iree_ok_status();
  }
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_value_id, lane_index,
      lane_type, source_op->location, &slice_op));
  *out_lane_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static bool loom_low_lower_rule_register_lane_type(const loom_module_t* module,
                                                   loom_value_id_t low_value_id,
                                                   loom_type_t* out_lane_type) {
  const loom_type_t low_type = loom_module_value_type(module, low_value_id);
  IREE_ASSERT(loom_low_type_is_register(low_type));
  IREE_ASSERT_GT(loom_low_register_type_unit_count(low_type), 0);
  return loom_low_lower_rule_try_register_type_with_unit_count(low_type, 1,
                                                               out_lane_type);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_first_lane(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_GT(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, NULL, NULL, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, low_operands));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));
  IREE_ASSERT(loom_low_type_is_register(result_types[0]));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_types[0]), 1);

  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    const loom_type_t operand_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    IREE_ASSERT(loom_low_type_is_register(operand_type));
    loom_type_t operand_lane_type = loom_type_none();
    if (!loom_low_lower_rule_try_register_type_with_unit_count(
            operand_type, 1, &operand_lane_type)) {
      return loom_low_lower_emit_register_width_relation_unsupported(
          context, source_op, operand_type, 1);
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_slice_lane(context, source_op, low_operands[i], 0,
                                       operand_lane_type, &lane_operands[i]));
  }
  loom_low_lower_rule_apply_operand_flags(emit, lane_operands);

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, NULL, &attrs));

  loom_op_t* low_op = NULL;
  const loom_tied_result_t* tied_results =
      loom_low_lower_rule_emit_tied_results(rule_set, emit);
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &resolved_emit->descriptor, lane_operands,
      emit->operand_ref_count, attrs, result_types, emit->result_ref_count,
      tied_results, emit->tied_result_count, source_op->location, &low_op));
  loom_value_slice_t low_results = loom_low_op_results(low_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results.values);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_per_lane(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_GT(emit->operand_ref_count, 0);
  IREE_ASSERT_GT(emit->result_ref_count, 0);
  IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, NULL, NULL, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, low_operands));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, resolved_emit, &result_types));

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      resolved_emit->descriptor.descriptor;
  uint32_t lane_count = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* packet_operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (!loom_low_operand_role_is_packet_operand(packet_operand->role)) {
      continue;
    }
    const uint16_t packet_operand_index = packet_operand->source_value_index;
    IREE_ASSERT_LT(packet_operand_index, emit->operand_ref_count);
    IREE_ASSERT_GT(packet_operand->unit_count, 0);
    const loom_type_t operand_type =
        loom_module_value_type(loom_low_lower_context_module(context),
                               low_operands[packet_operand_index]);
    IREE_ASSERT(loom_low_type_is_register(operand_type));
    const uint32_t operand_unit_count =
        loom_low_register_type_unit_count(operand_type);
    IREE_ASSERT_GT(operand_unit_count, 0);
    IREE_ASSERT_EQ(operand_unit_count % packet_operand->unit_count, 0);
    const uint32_t operand_lane_count =
        operand_unit_count / packet_operand->unit_count;
    if (lane_count == 0) {
      lane_count = operand_lane_count;
    } else {
      IREE_ASSERT_EQ(operand_lane_count, lane_count);
    }
  }
  IREE_ASSERT_GT(lane_count, 0);

  loom_type_t* lane_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->result_ref_count, sizeof(*lane_result_types),
      (void**)&lane_result_types));
  loom_type_t* aggregate_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->result_ref_count, sizeof(*aggregate_result_types),
      (void**)&aggregate_result_types));
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    const loom_low_operand_t* result_operand =
        loom_low_lower_rule_descriptor_result_operand(descriptor_set,
                                                      descriptor, i);
    IREE_ASSERT_GT(result_operand->unit_count, 0);
    IREE_ASSERT(loom_low_type_is_register(result_types[i]));
    const uint32_t type_unit_count =
        loom_low_register_type_unit_count(result_types[i]);
    IREE_ASSERT(type_unit_count == result_operand->unit_count ||
                type_unit_count == result_operand->unit_count * lane_count);
    if (!loom_low_lower_rule_try_register_type_with_unit_count(
            result_types[i], result_operand->unit_count,
            &lane_result_types[i])) {
      return loom_low_lower_emit_register_width_relation_unsupported(
          context, source_op, result_types[i], result_operand->unit_count);
    }
    const uint32_t aggregate_unit_count =
        result_operand->unit_count * lane_count;
    if (!loom_low_lower_rule_try_register_type_with_unit_count(
            result_types[i], aggregate_unit_count,
            &aggregate_result_types[i])) {
      return loom_low_lower_emit_register_width_relation_unsupported(
          context, source_op, result_types[i], aggregate_unit_count);
    }
  }

  const loom_tied_result_t* tied_results =
      loom_low_lower_rule_emit_tied_results(rule_set, emit);
  if (lane_count == 1) {
    loom_low_lower_rule_apply_operand_flags(emit, low_operands);
    loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
        context, rule_set, source_op, emit, NULL, &attrs));
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &resolved_emit->descriptor, low_operands,
        emit->operand_ref_count, attrs, result_types, emit->result_ref_count,
        tied_results, emit->tied_result_count, source_op->location, &low_op));
    return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                            emit,
                                            loom_low_op_results(low_op).values);
  }

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, NULL, &attrs));
  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  loom_value_id_t* lane_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->result_ref_count * lane_count, sizeof(*lane_results),
      (void**)&lane_results));
  for (uint32_t lane_index = 0; lane_index < lane_count; ++lane_index) {
    for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
         ++i) {
      const loom_low_operand_t* packet_operand =
          &descriptor_set->operands[descriptor->operand_start + i];
      if (!loom_low_operand_role_is_packet_operand(packet_operand->role)) {
        continue;
      }
      const uint16_t operand_index = packet_operand->source_value_index;
      IREE_ASSERT_LT(operand_index, emit->operand_ref_count);
      const uint32_t operand_unit_count = packet_operand->unit_count;
      const uint32_t register_offset = lane_index * operand_unit_count;
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_register_units(
          context, source_op, low_operands[operand_index], register_offset,
          operand_unit_count, &lane_operands[operand_index]));
    }
    loom_low_lower_rule_apply_operand_flags(emit, lane_operands);
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &resolved_emit->descriptor, lane_operands,
        emit->operand_ref_count, attrs, lane_result_types,
        emit->result_ref_count, tied_results, emit->tied_result_count,
        source_op->location, &lane_op));
    const loom_value_slice_t low_results = loom_low_op_results(lane_op);
    for (uint16_t result_index = 0; result_index < emit->result_ref_count;
         ++result_index) {
      lane_results[result_index * lane_count + lane_index] =
          loom_value_slice_get(low_results, result_index);
    }
  }

  loom_value_id_t* low_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->result_ref_count, sizeof(*low_results),
      (void**)&low_results));
  for (uint16_t result_index = 0; result_index < emit->result_ref_count;
       ++result_index) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        loom_low_lower_context_builder(context),
        &lane_results[result_index * lane_count], lane_count,
        aggregate_result_types[result_index], source_op->location, &concat_op));
    low_results[result_index] = loom_low_concat_result(concat_op);
  }
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results);
}

static iree_status_t loom_low_lower_rule_build_lane_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit, uint32_t lane_count,
    uint32_t lane_index, loom_value_id_t* lane_operands) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  for (uint16_t operand_index = 0; operand_index < emit->operand_ref_count;
       ++operand_index) {
    loom_value_id_t low_operand = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_low_value(
        context, rule_set, source_op, state, NULL, NULL,
        (uint16_t)(emit->operand_ref_start + operand_index), &low_operand));
    const loom_type_t operand_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operand);
    IREE_ASSERT(loom_low_type_is_register(operand_type));
    const uint32_t operand_unit_count =
        loom_low_register_type_unit_count(operand_type);
    if (operand_unit_count == 1) {
      lane_operands[operand_index] = low_operand;
      continue;
    }
    IREE_ASSERT_EQ(operand_unit_count, lane_count);
    loom_type_t operand_lane_type = loom_type_none();
    if (!loom_low_lower_rule_try_register_type_with_unit_count(
            operand_type, 1, &operand_lane_type)) {
      return loom_low_lower_emit_register_width_relation_unsupported(
          context, source_op, operand_type, 1);
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_lane(
        context, source_op, low_operand, lane_index, operand_lane_type,
        &lane_operands[operand_index]));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, lane_operands));
  loom_low_lower_rule_apply_operand_flags(emit, lane_operands);
  return iree_ok_status();
}

static void loom_low_lower_rule_bind_per_lane_sequence_result(
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_emit_state_t* state, const loom_low_lower_emit_t* emit,
    bool is_final_emit, loom_value_id_t low_result, uint32_t lane_index,
    loom_value_id_t* lane_results) {
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  const uint16_t value_ref_index =
      loom_low_lower_rule_emit_result_bind_ref_index(emit, 0);
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_FALSE(is_final_emit);
      IREE_ASSERT_LT(value_ref->index, state->temporary_count);
      IREE_ASSERT(state->temporaries != NULL);
      state->temporaries[value_ref->index] = low_result;
      break;
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      IREE_ASSERT(is_final_emit);
      lane_results[lane_index] = low_result;
      break;
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
    default:
      IREE_ASSERT_UNREACHABLE(
          "per-lane sequence results must bind temporaries or results");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_per_lane_sequence(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state, const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t* resolved_emits) {
  IREE_ASSERT_GT(rule->emit_count, 1);
  const uint16_t final_emit_ordinal = (uint16_t)(rule->emit_count - 1);
  const loom_low_lower_emit_t* final_emit =
      resolved_emits[final_emit_ordinal].emit;
  IREE_ASSERT_EQ(final_emit->kind,
                 LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE);
  IREE_ASSERT_EQ(final_emit->result_ref_count, 1);

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, rule->emit_count, sizeof(*result_types), (void**)&result_types));
  loom_type_t* lane_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, rule->emit_count, sizeof(*lane_result_types),
      (void**)&lane_result_types));
  loom_named_attr_slice_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, rule->emit_count, sizeof(*attrs), (void**)&attrs));
  uint16_t max_operand_ref_count = 0;
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const loom_low_lower_resolved_emit_t* resolved_emit =
        &resolved_emits[emit_ordinal];
    const loom_low_lower_emit_t* emit = resolved_emit->emit;
    IREE_ASSERT_EQ(emit->kind,
                   LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE);
    IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);
    IREE_ASSERT_EQ(emit->result_ref_count, 1);
    const uint16_t bind_ref_index =
        loom_low_lower_rule_emit_result_bind_ref_index(emit, 0);
    const loom_low_lower_value_ref_t* bind_ref =
        &rule_set->value_refs[bind_ref_index];
    if (emit_ordinal == final_emit_ordinal) {
      IREE_ASSERT_EQ(bind_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
    } else {
      IREE_ASSERT_EQ(bind_ref->kind, LOOM_LOW_LOWER_VALUE_REF_TEMPORARY);
    }
    loom_type_t* emit_result_types = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
        context, rule_set, source_op, resolved_emit, &emit_result_types));
    result_types[emit_ordinal] = emit_result_types[0];
    IREE_ASSERT(loom_low_type_is_register(result_types[emit_ordinal]));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
        context, rule_set, source_op, emit, NULL, &attrs[emit_ordinal]));
    max_operand_ref_count =
        iree_max(max_operand_ref_count, emit->operand_ref_count);
  }

  const loom_type_t final_result_type = result_types[final_emit_ordinal];
  const uint32_t lane_count =
      loom_low_register_type_unit_count(final_result_type);
  IREE_ASSERT_GT(lane_count, 0);
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const loom_type_t result_type = result_types[emit_ordinal];
    const uint32_t result_unit_count =
        loom_low_register_type_unit_count(result_type);
    IREE_ASSERT(result_unit_count == 1 || result_unit_count == lane_count);
    if (!loom_low_lower_rule_try_register_type_with_unit_count(
            result_type, 1, &lane_result_types[emit_ordinal])) {
      return loom_low_lower_emit_register_width_relation_unsupported(
          context, source_op, result_type, 1);
    }
  }

  loom_value_id_t* lane_operands = NULL;
  if (max_operand_ref_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, max_operand_ref_count, sizeof(*lane_operands),
        (void**)&lane_operands));
  }
  loom_value_id_t* lane_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, lane_count, sizeof(*lane_results), (void**)&lane_results));

  for (uint32_t lane_index = 0; lane_index < lane_count; ++lane_index) {
    for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
         ++emit_ordinal) {
      const loom_low_lower_resolved_emit_t* resolved_emit =
          &resolved_emits[emit_ordinal];
      const loom_low_lower_emit_t* emit = resolved_emit->emit;
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_lane_operands(
          context, rule_set, source_op, state, resolved_emit, lane_count,
          lane_index, lane_operands));
      loom_op_t* lane_op = NULL;
      const loom_tied_result_t* tied_results =
          loom_low_lower_rule_emit_tied_results(rule_set, emit);
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &resolved_emit->descriptor, lane_operands,
          emit->operand_ref_count, attrs[emit_ordinal],
          &lane_result_types[emit_ordinal], 1, tied_results,
          emit->tied_result_count, source_op->location, &lane_op));
      loom_low_lower_rule_bind_per_lane_sequence_result(
          rule_set, state, emit, emit_ordinal == final_emit_ordinal,
          loom_value_slice_get(loom_low_op_results(lane_op), 0), lane_index,
          lane_results);
    }
  }

  if (lane_count == 1) {
    return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                            final_emit, lane_results);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      final_result_type, source_op->location, &concat_op));
  const loom_value_id_t low_result = loom_low_concat_result(concat_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          final_emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_accumulate_lanes(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_GT(emit->operand_ref_count, 1);
  IREE_ASSERT_LT(emit->accumulator_operand_index, emit->operand_ref_count);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  IREE_ASSERT_EQ(emit->attr_copy_count, 0);
  IREE_ASSERT_EQ(emit->tied_result_count, 0);
  IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);
  const loom_low_lower_emit_flags_t supported_flags =
      LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE |
      LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED |
      LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE;
  IREE_ASSERT_EQ(emit->flags & ~supported_flags, 0);
  const bool seed_first_lane = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE);
  const bool balanced_tree = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED);
  const bool skip_first_lane = iree_any_bit_set(
      emit->flags, LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE);
  IREE_ASSERT_FALSE(seed_first_lane && skip_first_lane);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, NULL, NULL, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, low_operands));

  loom_value_id_t source_result = loom_low_lower_rule_source_value(
      context->module, rule_set, source_op, emit->result_ref_start);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_map_result_type(
      context, source_op, source_result, &result_type));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_type), 1);

  uint32_t lane_count = 0;
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    if (i == emit->accumulator_operand_index) {
      continue;
    }
    const loom_type_t operand_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    IREE_ASSERT(loom_low_type_is_register(operand_type));
    if (lane_count == 0) {
      lane_count = loom_low_register_type_unit_count(operand_type);
      IREE_ASSERT_GT(lane_count, 0);
    } else {
      IREE_ASSERT_EQ(loom_low_register_type_unit_count(operand_type),
                     lane_count);
    }
  }
  IREE_ASSERT_GT(lane_count, 0);

  const loom_type_t result_type_scalar = result_type;
  loom_value_id_t accumulator = low_operands[emit->accumulator_operand_index];
  const loom_type_t accumulator_type = loom_module_value_type(
      loom_low_lower_context_module(context), accumulator);
  IREE_ASSERT(loom_low_type_is_register(accumulator_type));
  uint32_t first_lane_index = 0;
  if (seed_first_lane) {
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(accumulator_type),
                   lane_count);
    loom_type_t accumulator_lane_type = loom_type_none();
    if (!loom_low_lower_rule_try_register_type_with_unit_count(
            accumulator_type, 1, &accumulator_lane_type)) {
      return loom_low_lower_emit_register_width_relation_unsupported(
          context, source_op, accumulator_type, 1);
    }
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_slice_lane(context, source_op, accumulator, 0,
                                       accumulator_lane_type, &accumulator));
    first_lane_index = 1;
  } else {
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(accumulator_type), 1);
    first_lane_index = skip_first_lane ? 1 : 0;
  }

  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  if (balanced_tree) {
    IREE_ASSERT_EQ(emit->operand_ref_count, 2);
    const uint16_t term_operand_index =
        emit->accumulator_operand_index == 0 ? 1 : 0;
    const uint32_t term_count = (seed_first_lane || skip_first_lane)
                                    ? lane_count
                                    : (uint32_t)(lane_count + 1);
    loom_value_id_t* lane_terms = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, term_count, sizeof(*lane_terms), (void**)&lane_terms));
    uint32_t term_index = 0;
    lane_terms[term_index++] = accumulator;
    loom_type_t term_operand_lane_type = loom_type_none();
    if (!loom_low_lower_rule_register_lane_type(
            loom_low_lower_context_module(context),
            low_operands[term_operand_index], &term_operand_lane_type)) {
      return loom_low_lower_emit_register_width_relation_unsupported(
          context, source_op,
          loom_module_value_type(loom_low_lower_context_module(context),
                                 low_operands[term_operand_index]),
          1);
    }
    for (uint32_t lane_index = first_lane_index; lane_index < lane_count;
         ++lane_index) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_lane(
          context, source_op, low_operands[term_operand_index], lane_index,
          term_operand_lane_type, &lane_terms[term_index++]));
    }
    IREE_ASSERT_EQ(term_index, term_count);

    for (uint32_t step = 1; step < term_count; step <<= 1) {
      for (uint32_t lane_index = 0; lane_index + step < term_count;
           lane_index += step << 1) {
        lane_operands[emit->accumulator_operand_index] = lane_terms[lane_index];
        lane_operands[term_operand_index] = lane_terms[lane_index + step];
        loom_op_t* lane_op = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
            context, &resolved_emit->descriptor, lane_operands,
            emit->operand_ref_count, loom_make_named_attr_slice(NULL, 0),
            &result_type_scalar, 1, NULL, 0, source_op->location, &lane_op));
        lane_terms[lane_index] =
            loom_value_slice_get(loom_low_op_results(lane_op), 0);
      }
    }
    accumulator = lane_terms[0];
    return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                            emit, &accumulator);
  }

  for (uint32_t lane_index = first_lane_index; lane_index < lane_count;
       ++lane_index) {
    for (uint16_t operand_index = 0; operand_index < emit->operand_ref_count;
         ++operand_index) {
      if (operand_index == emit->accumulator_operand_index) {
        lane_operands[operand_index] = accumulator;
        continue;
      }
      loom_type_t operand_lane_type = loom_type_none();
      if (!loom_low_lower_rule_register_lane_type(
              loom_low_lower_context_module(context),
              low_operands[operand_index], &operand_lane_type)) {
        return loom_low_lower_emit_register_width_relation_unsupported(
            context, source_op,
            loom_module_value_type(loom_low_lower_context_module(context),
                                   low_operands[operand_index]),
            1);
      }
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_lane(
          context, source_op, low_operands[operand_index], lane_index,
          operand_lane_type, &lane_operands[operand_index]));
    }
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &resolved_emit->descriptor, lane_operands,
        emit->operand_ref_count, loom_make_named_attr_slice(NULL, 0),
        &result_type, 1, NULL, 0, source_op->location, &lane_op));
    accumulator = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &accumulator);
}

iree_status_t loom_low_lower_rule_set_emit_rule(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule,
    const loom_low_lower_resolved_emit_t* resolved_emits,
    const loom_low_source_memory_access_plan_t* source_memory_access) {
  IREE_ASSERT(rule->emit_count == 0 || resolved_emits != NULL);

  loom_low_lower_rule_emit_state_t state = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_emit_state_initialize(context, rule, &state));
  if (rule->emit_count != 0 &&
      resolved_emits[0].emit->kind ==
          LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_emit_descriptor_op_per_lane_sequence(
            context, rule_set, source_op, &state, rule, resolved_emits));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_bind_aliases(context, rule_set, source_op, rule));
    return loom_low_lower_rule_elide_results(context, rule_set, source_op,
                                             rule);
  }
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    const loom_low_lower_resolved_emit_t* resolved_emit = &resolved_emits[i];
    IREE_ASSERT(resolved_emit->emit == emit);
    switch (emit->kind) {
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op(
            context, rule_set, source_op, &state, resolved_emit,
            source_memory_access));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_const(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_FIRST_LANE: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op_first_lane(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op_per_lane(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE:
        IREE_ASSERT_UNREACHABLE(
            "per-lane sequence emits must be the whole emit program");
        IREE_BUILTIN_UNREACHABLE();
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES: {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_rule_emit_descriptor_op_accumulate_lanes(
                context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_REGISTER_SLICE: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_register_slice(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_REGISTER_CONCAT: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_register_concat(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE("unknown generated lower emit kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_bind_aliases(context, rule_set, source_op, rule));
  return loom_low_lower_rule_elide_results(context, rule_set, source_op, rule);
}
