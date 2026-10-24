// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_emit.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/control_plan.h"
#include "loom/codegen/low/lower/rule_descriptor.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/lower/rule_source_memory_plan.h"
#include "loom/codegen/low/lower/rule_value.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/registers.h"

typedef struct loom_low_lower_rule_emit_state_t {
  // Root source operation selected by the rule.
  const loom_op_t* source_op;
  // Selected source graph with the root at index zero. NULL for root-only
  // rules.
  const loom_op_t* const* source_nodes;
  // Rule-local low SSA values captured by earlier emit rows.
  loom_value_id_t* temporaries;
  // Number of entries in temporaries.
  uint16_t temporary_count;
  // Number of source operations owned by the selected rule.
  uint8_t source_node_count;
} loom_low_lower_rule_emit_state_t;

static iree_status_t loom_low_lower_rule_emit_state_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* const* source_nodes, uint8_t source_node_count,
    const loom_low_lower_rule_t* rule,
    loom_low_lower_rule_emit_state_t* out_state) {
  *out_state = (loom_low_lower_rule_emit_state_t){
      .source_op = source_op,
      .source_nodes = source_nodes,
      .temporaries = NULL,
      .temporary_count = rule->temporary_count,
      .source_node_count = source_node_count,
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

static const loom_op_t* loom_low_lower_rule_emit_source_op(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_emit_state_t* state, uint16_t value_ref_index) {
  return loom_low_lower_rule_source_op(
      rule_set, state->source_op, state->source_nodes, state->source_node_count,
      value_ref_index);
}

static iree_status_t loom_low_lower_rule_materialize_attributes(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    loom_named_attr_slice_t* out_attributes) {
  const loom_named_attr_slice_t attributes =
      loom_low_lower_resolved_emit_attributes(resolved_emit);
  if (!resolved_emit->emit->has_read_only_data_attributes) {
    *out_attributes = attributes;
    return iree_ok_status();
  }
  loom_named_attr_t* materialized_attributes = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, attributes.count, sizeof(*materialized_attributes),
      (void**)&materialized_attributes));
  memcpy(materialized_attributes, attributes.entries,
         attributes.count * sizeof(*materialized_attributes));
  const loom_low_lower_read_only_attributes_t* read_only_attributes =
      loom_low_lower_resolved_emit_read_only_attributes(resolved_emit);
  const loom_low_lower_read_only_data_id_t* ids =
      (const loom_low_lower_read_only_data_id_t*)(read_only_attributes + 1);
  uint32_t remaining_mask = read_only_attributes->attribute_mask;
  while (remaining_mask) {
    const uint32_t ordinal = iree_math_count_trailing_zeros_u32(remaining_mask);
    remaining_mask &= remaining_mask - 1u;
    loom_symbol_ref_t symbol = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_low_lower_module_state_reference_read_only_data(
        context->module_state, context->module, ids[ordinal], &symbol));
    materialized_attributes[ordinal].value = loom_attr_symbol(symbol);
  }
  *out_attributes =
      loom_make_named_attr_slice(materialized_attributes, attributes.count);
  return iree_ok_status();
}

static loom_value_id_t loom_low_lower_rule_emit_source_value(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    uint16_t reference_ordinal, uint16_t value_ref_index) {
  if (resolved_emit->source_value_mask & (1u << reference_ordinal)) {
    return loom_low_lower_resolved_emit_source_value(resolved_emit,
                                                     reference_ordinal);
  }
  return loom_low_lower_rule_source_value_from_nodes(
      context->module, rule_set, state->source_op, state->source_nodes,
      state->source_node_count, value_ref_index);
}

static iree_status_t loom_low_lower_rule_low_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    uint16_t operand_ordinal, loom_value_id_t* out_low_value_id) {
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  const uint16_t value_ref_index =
      resolved_emit->emit->operand_ref_start + operand_ordinal;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
    case LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND: {
      loom_value_id_t source_value_id = loom_low_lower_rule_emit_source_value(
          context, rule_set, state, resolved_emit, operand_ordinal,
          value_ref_index);
      if (value_ref->materializer_index != 0) {
        const loom_low_lower_value_materializer_t* materializer =
            loom_low_lower_rule_value_materializer(rule_set, value_ref);
        const loom_op_t* materialized_source_op =
            loom_low_lower_rule_emit_source_op(rule_set, state,
                                               value_ref_index);
        if (resolved_emit->materializer_mask & (1u << operand_ordinal)) {
          return materializer->emit.planned(
              context, materialized_source_op, source_value_id,
              loom_low_lower_resolved_emit_materializer(resolved_emit,
                                                        operand_ordinal),
              out_low_value_id);
        }
        return materializer->emit.direct(context, materialized_source_op,
                                         source_value_id, out_low_value_id);
      }
      *out_low_value_id = loom_low_lower_lookup_value(context, source_value_id);
      return iree_ok_status();
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
      if (source_memory->byte_offset_materializer_ordinal !=
          LOOM_LOW_LOWER_SOURCE_MEMORY_MATERIALIZER_NONE) {
        return loom_low_lower_rule_materialize_source_memory_dynamic_term(
            context, rule_set, source_op, source_memory, source_memory_access,
            value_ref->index, out_low_value_id);
      }
      const loom_value_id_t source_value_id =
          source_memory_access->dynamic_terms[value_ref->index].index;
      *out_low_value_id = loom_low_lower_lookup_value(context, source_value_id);
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET:
      return loom_low_lower_rule_source_memory_emit_byte_offset(
          context, rule_set, source_op, source_memory, source_memory_access,
          resolved_emit, operand_ordinal, out_low_value_id);
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS: {
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT(source_memory_access != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      return loom_low_lower_rule_materialize_source_memory_address(
          context, rule_set, source_op, source_memory, source_memory_access,
          loom_type_table_get(
              &context->module->types,
              loom_low_lower_resolved_emit_address_coordinate_type_id(
                  resolved_emit)),
          out_low_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ELEMENT_STRIDE:
      return loom_low_lower_rule_materialize_source_memory_element_stride(
          context, rule_set, source_op, source_memory, source_memory_access,
          (uint8_t)value_ref->index,
          loom_type_table_get(
              &context->module->types,
              loom_low_lower_resolved_emit_address_coordinate_type_id(
                  resolved_emit)),
          out_low_value_id);
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ROOT:
      IREE_ASSERT(source_memory != NULL);
      IREE_ASSERT(source_memory_access != NULL);
      IREE_ASSERT_EQ(value_ref->materializer_index, 0);
      *out_low_value_id = loom_low_lower_lookup_value(
          context, source_memory_access->root_value_id);
      return iree_ok_status();
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_rule_build_low_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t** out_operands) {
  *out_operands = NULL;
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
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
        source_memory_access, resolved_emit, i, &low_operands[i]));
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
  const loom_type_id_t* copy_types =
      loom_low_lower_resolved_emit_copy_types(resolved_emit);
  uint16_t remaining_mask = emit->copy_operand_mask;
  uint16_t copy_index = 0;
  while (remaining_mask) {
    const uint16_t i =
        (uint16_t)iree_math_count_trailing_zeros_u32(remaining_mask);
    remaining_mask &= (uint16_t)(remaining_mask - 1u);
    const loom_type_t copy_type =
        loom_type_table_get(&context->module->types, copy_types[copy_index++]);
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

static iree_status_t loom_low_lower_rule_materialize_descriptor_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit, loom_value_id_t* low_operands) {
  if (emit->operand_materialization ==
      LOOM_LOW_LOWER_OPERAND_MATERIALIZATION_DIRECT) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(emit->operand_materialization,
                 LOOM_LOW_LOWER_OPERAND_MATERIALIZATION_TARGET);
  const loom_low_lower_materialize_descriptor_operands_fn_t materialize =
      context->policy->materialize_descriptor_operands;
  IREE_ASSERT(materialize != NULL);
  return materialize(context, source_op, low_operands, emit->operand_ref_count);
}

static iree_status_t loom_low_lower_rule_build_result_types(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  if (emit->result_ref_count == 0) {
    return iree_ok_status();
  }
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->result_ref_count, sizeof(*result_types),
      (void**)&result_types));
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    result_types[i] = loom_type_table_get(
        &context->module->types,
        loom_low_lower_resolved_emit_result_type_id(resolved_emit, i));
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_bind_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state, const loom_low_lower_emit_t* emit,
    const loom_value_id_t* low_results) {
  IREE_ASSERT_EQ(source_op, state->source_op);
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    const uint16_t value_ref_index =
        loom_low_lower_rule_emit_result_bind_ref_index(emit, i);
    const loom_low_lower_value_ref_t* value_ref =
        &rule_set->value_refs[value_ref_index];
    switch (value_ref->kind) {
      case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
        loom_value_id_t source_value_id =
            loom_low_lower_rule_source_value_from_nodes(
                context->module, rule_set, state->source_op,
                state->source_nodes, state->source_node_count, value_ref_index);
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(context, source_value_id,
                                                       low_results[i]));
        break;
      }
      case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
        IREE_ASSERT_LT(value_ref->index, state->temporary_count);
        IREE_ASSERT(state->temporaries != NULL);
        // Generated temporary slots are liveness-packed and may retain a dead
        // value when the next nonoverlapping value is bound to the same slot.
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

static void loom_low_lower_rule_elide_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  for (uint16_t i = 0; i < rule->metadata.value.elide_ref_count; ++i) {
    const uint16_t value_ref_index =
        (uint16_t)(rule->action.elide_ref_start + i);
    const loom_low_lower_value_ref_t* value_ref =
        &rule_set->value_refs[value_ref_index];
    IREE_ASSERT_EQ(value_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
    loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
        context->module, rule_set, source_op, value_ref_index);
    loom_low_lower_elide_value(context, source_value_id);
  }
}

static iree_status_t loom_low_lower_rule_bind_aliases(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  if (iree_all_bits_set(rule->flags,
                        LOOM_LOW_LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS)) {
    IREE_ASSERT_EQ(rule->metadata.value.alias_ref_count, 1);
    const uint16_t source_ref_index = rule->action.alias_ref_start;
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
  for (uint16_t i = 0; i < rule->metadata.value.alias_ref_count; ++i) {
    const uint16_t source_ref_index =
        (uint16_t)(rule->action.alias_ref_start + i * 2);
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
    const loom_low_lower_resolved_emit_t* resolved_emit,
    const loom_low_source_memory_access_plan_t* source_memory_access) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, resolved_emit, &result_types));

  loom_named_attr_slice_t attrs;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_attributes(
      context, resolved_emit, &attrs));

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
    const loom_low_lower_selected_source_memory_t* selected_source_memory) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  const loom_low_lower_source_memory_t* source_memory = NULL;
  const loom_low_source_memory_access_plan_t* emit_source_memory_access = NULL;
  if (emit->source_memory_ordinal != 0) {
    const uint16_t source_memory_index =
        (uint16_t)(emit->source_memory_ordinal - 1);
    source_memory = &rule_set->source_memories[source_memory_index];
    IREE_ASSERT(selected_source_memory != NULL);
    emit_source_memory_access = &selected_source_memory->access;
  }

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, resolved_emit, source_memory,
      emit_source_memory_access, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, low_operands));
  loom_low_lower_rule_apply_operand_flags(emit, low_operands);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_descriptor_operands(
      context, source_op, emit, low_operands));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, resolved_emit, &result_types));

  loom_named_attr_slice_t attrs;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_attributes(
      context, resolved_emit, &attrs));

  const loom_tied_result_t* tied_results = NULL;
  if (emit->tied_result_count != 0) {
    tied_results =
        &rule_set->tied_results[emit->payload.descriptor.tied_result_start];
  }

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &context->builder, context->descriptor_set,
      resolved_emit->descriptor.descriptor, resolved_emit->access_flags,
      low_operands, emit->operand_ref_count, attrs, result_types,
      emit->result_ref_count, tied_results, emit->tied_result_count,
      source_op->location, &low_op));
  if (emit_source_memory_access != NULL) {
    const loom_low_lower_memory_packet_geometry_t geometry =
        loom_low_lower_memory_packet_geometry_from_source(
            emit_source_memory_access);
    IREE_RETURN_IF_ERROR(loom_low_lower_record_memory_packet(
        context, low_op, resolved_emit->descriptor.descriptor,
        selected_source_memory->origin, emit_source_memory_access, &geometry,
        loom_value_facts_exact_i64(0)));
  }
  loom_value_slice_t low_results = loom_low_op_results(low_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results.values);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_cond_br(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(source_op->successor_count, 2);
  IREE_ASSERT_EQ(emit->result_ref_count, 0);
  IREE_ASSERT_EQ(emit->source_memory_ordinal, 0);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, resolved_emit,
      /*source_memory=*/NULL, /*source_memory_access=*/NULL, &low_operands));

  loom_named_attr_slice_t attrs;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_attributes(
      context, resolved_emit, &attrs));

  loom_op_t* low_cond_br_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_cond_br(
      &context->builder, context->descriptor_set,
      resolved_emit->descriptor.descriptor, low_operands,
      emit->operand_ref_count, attrs,
      loom_low_lower_lookup_successor_dest(context, source_op, 0),
      loom_low_lower_lookup_successor_dest(context, source_op, 1),
      source_op->location, &low_cond_br_op));
  return loom_low_lower_control_attach_cond_branch_facts(context, source_op,
                                                         low_cond_br_op);
}

static iree_status_t loom_low_lower_rule_emit_register_slice(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(emit->operand_ref_count, 1);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, resolved_emit, NULL, NULL,
      &low_operands));
  const loom_type_t result_type = loom_type_table_get(
      &context->module->types,
      loom_low_lower_resolved_emit_result_type_id(resolved_emit, 0));

  loom_type_t slice_result_type = result_type;
  if (emit->kind == LOOM_LOW_LOWER_EMIT_REGISTER_SLICE_CLASS) {
    const loom_type_t source_type =
        loom_module_value_type(context->module, low_operands[0]);
    if (emit->payload.structural.unit_count ==
        loom_low_register_type_unit_count(source_type)) {
      slice_result_type = source_type;
    } else {
      IREE_ASSERT(!loom_type_register_has_value_type(source_type),
                  "planned width-changing register slice must be untyped");
      slice_result_type = loom_low_register_carrier_type_with_unit_count(
          source_type, emit->payload.structural.unit_count);
    }
  }

  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_slice_build(loom_low_lower_context_builder(context),
                           low_operands[0], emit->payload.structural.offset,
                           slice_result_type, source_op->location, &slice_op));
  loom_value_id_t low_result = loom_low_slice_result(slice_op);
  if (emit->kind == LOOM_LOW_LOWER_EMIT_REGISTER_SLICE_CLASS) {
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_copy_build(
        loom_low_lower_context_builder(context), low_result,
        /*detached=*/false, result_type, source_op->location, &copy_op));
    low_result = loom_low_copy_result(copy_op);
  }
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_register_concat(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_GT(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, resolved_emit, NULL, NULL,
      &low_operands));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, resolved_emit, &result_types));

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            low_operands, emit->operand_ref_count,
                            result_types[0], source_op->location, &concat_op));
  const loom_value_id_t low_result = loom_low_concat_result(concat_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_register_copy(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    const loom_low_lower_selected_source_memory_t* selected_source_memory) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(emit->operand_ref_count, 1);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);

  const loom_low_lower_source_memory_t* source_memory = NULL;
  const loom_low_source_memory_access_plan_t* source_memory_access = NULL;
  if (emit->source_memory_ordinal != 0) {
    source_memory = &rule_set->source_memories[emit->source_memory_ordinal - 1];
    IREE_ASSERT(selected_source_memory != NULL);
    source_memory_access = &selected_source_memory->access;
  }

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, resolved_emit, source_memory,
      source_memory_access, &low_operands));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, resolved_emit, &result_types));

  loom_op_t* copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(
      loom_low_lower_context_builder(context), low_operands[0], false,
      result_types[0], source_op->location, &copy_op));
  const loom_value_id_t low_result = loom_low_copy_result(copy_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_register_move(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  IREE_ASSERT_EQ(emit->operand_ref_count, 1);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, resolved_emit, NULL, NULL,
      &low_operands));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, resolved_emit, &result_types));

  loom_op_t* move_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_build(
      loom_low_lower_context_builder(context), low_operands[0], false,
      result_types[0], source_op->location, &move_op));
  const loom_value_id_t low_result = loom_low_move_result(move_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static const loom_tied_result_t* loom_low_lower_rule_emit_tied_results(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  return emit->tied_result_count == 0
             ? NULL
             : &rule_set
                    ->tied_results[emit->payload.descriptor.tied_result_start];
}

static iree_status_t loom_low_lower_rule_slice_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_emit_t* resolved_emit, uint16_t operand_index,
    loom_value_id_t low_value_id, uint32_t lane_index,
    loom_value_id_t* out_slice_value_id) {
  const loom_low_lower_lane_plan_t* lanes =
      loom_low_lower_resolved_emit_lane_plan(resolved_emit);
  if (!(lanes->slice_operand_mask & (1u << operand_index))) {
    *out_slice_value_id = low_value_id;
    return iree_ok_status();
  }
  const loom_type_id_t* operand_types = (const loom_type_id_t*)(lanes + 1);
  const loom_type_t slice_type = loom_type_table_get(
      &context->module->types, operand_types[operand_index]);
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_value_id,
      lane_index * loom_low_register_type_unit_count(slice_type), slice_type,
      source_op->location, &slice_op));
  *out_slice_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
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
      context, rule_set, source_op, state, resolved_emit, NULL, NULL,
      &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, low_operands));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, resolved_emit, &result_types));
  IREE_ASSERT(loom_low_type_is_register(result_types[0]));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_types[0]), 1);

  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_operand(
        context, source_op, resolved_emit, i, low_operands[i], 0,
        &lane_operands[i]));
  }
  loom_low_lower_rule_apply_operand_flags(emit, lane_operands);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_descriptor_operands(
      context, source_op, emit, lane_operands));

  loom_named_attr_slice_t attrs;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_attributes(
      context, resolved_emit, &attrs));

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
      context, rule_set, source_op, state, resolved_emit, NULL, NULL,
      &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, low_operands));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, resolved_emit, &result_types));

  const uint32_t lane_count =
      loom_low_lower_resolved_emit_lane_plan(resolved_emit)->lane_count;
  loom_type_t lane_result_types[3];
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    lane_result_types[i] = loom_type_table_get(
        &context->module->types,
        loom_low_lower_resolved_emit_lane_result_type_id(resolved_emit, i));
  }

  const loom_tied_result_t* tied_results =
      loom_low_lower_rule_emit_tied_results(rule_set, emit);
  if (lane_count == 1) {
    loom_low_lower_rule_apply_operand_flags(emit, low_operands);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_descriptor_operands(
        context, source_op, emit, low_operands));
    loom_named_attr_slice_t attrs;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_attributes(
        context, resolved_emit, &attrs));
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &resolved_emit->descriptor, low_operands,
        emit->operand_ref_count, attrs, result_types, emit->result_ref_count,
        tied_results, emit->tied_result_count, source_op->location, &low_op));
    return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                            emit,
                                            loom_low_op_results(low_op).values);
  }

  loom_named_attr_slice_t attrs;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_attributes(
      context, resolved_emit, &attrs));
  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  loom_value_id_t* lane_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, emit->result_ref_count * lane_count, sizeof(*lane_results),
      (void**)&lane_results));
  for (uint32_t lane_index = 0; lane_index < lane_count; ++lane_index) {
    for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_operand(
          context, source_op, resolved_emit, i, low_operands[i], lane_index,
          &lane_operands[i]));
    }
    loom_low_lower_rule_apply_operand_flags(emit, lane_operands);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_descriptor_operands(
        context, source_op, emit, lane_operands));
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
        result_types[result_index], source_op->location, &concat_op));
    low_results[result_index] = loom_low_concat_result(concat_op);
  }
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results);
}

static iree_status_t loom_low_lower_rule_build_lane_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_resolved_emit_t* resolved_emit, uint32_t lane_index,
    loom_value_id_t* lane_operands) {
  const loom_low_lower_emit_t* emit = resolved_emit->emit;
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    loom_value_id_t low_operand = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_low_value(context, rule_set, source_op, state, NULL,
                                      NULL, resolved_emit, i, &low_operand));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_operand(
        context, source_op, resolved_emit, i, low_operand, lane_index,
        &lane_operands[i]));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, lane_operands));
  loom_low_lower_rule_apply_operand_flags(emit, lane_operands);
  return loom_low_lower_rule_materialize_descriptor_operands(
      context, source_op, emit, lane_operands);
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
    const loom_low_lower_resolved_emit_t* resolved_emits,
    uint16_t sequence_start_ordinal) {
  IREE_ASSERT_LT(sequence_start_ordinal, rule->emit_count);
  const uint16_t sequence_emit_count =
      (uint16_t)(rule->emit_count - sequence_start_ordinal);
  IREE_ASSERT_GT(sequence_emit_count, 0);
  const uint16_t final_emit_ordinal = (uint16_t)(rule->emit_count - 1);
  const loom_low_lower_emit_t* final_emit =
      resolved_emits[final_emit_ordinal].emit;
  IREE_ASSERT_EQ(final_emit->kind,
                 LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE);
  IREE_ASSERT_EQ(final_emit->result_ref_count, 1);

  loom_type_t* lane_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, sequence_emit_count, sizeof(*lane_result_types),
      (void**)&lane_result_types));
  loom_named_attr_slice_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, sequence_emit_count, sizeof(*attrs), (void**)&attrs));
  uint16_t max_operand_ref_count = 0;
  for (uint16_t emit_ordinal = sequence_start_ordinal;
       emit_ordinal < rule->emit_count; ++emit_ordinal) {
    const uint16_t sequence_ordinal =
        (uint16_t)(emit_ordinal - sequence_start_ordinal);
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
    lane_result_types[sequence_ordinal] = loom_type_table_get(
        &context->module->types,
        loom_low_lower_resolved_emit_lane_result_type_id(resolved_emit, 0));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_attributes(
        context, resolved_emit, &attrs[sequence_ordinal]));
    max_operand_ref_count =
        iree_max(max_operand_ref_count, emit->operand_ref_count);
  }

  const loom_low_lower_resolved_emit_t* final_resolved =
      &resolved_emits[final_emit_ordinal];
  const loom_type_t final_result_type = loom_type_table_get(
      &context->module->types,
      loom_low_lower_resolved_emit_result_type_id(final_resolved, 0));
  const uint32_t lane_count =
      loom_low_lower_resolved_emit_lane_plan(final_resolved)->lane_count;

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
    for (uint16_t emit_ordinal = sequence_start_ordinal;
         emit_ordinal < rule->emit_count; ++emit_ordinal) {
      const uint16_t sequence_ordinal =
          (uint16_t)(emit_ordinal - sequence_start_ordinal);
      const loom_low_lower_resolved_emit_t* resolved_emit =
          &resolved_emits[emit_ordinal];
      const loom_low_lower_emit_t* emit = resolved_emit->emit;
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_lane_operands(
          context, rule_set, source_op, state, resolved_emit, lane_index,
          lane_operands));
      loom_op_t* lane_op = NULL;
      const loom_tied_result_t* tied_results =
          loom_low_lower_rule_emit_tied_results(rule_set, emit);
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &resolved_emit->descriptor, lane_operands,
          emit->operand_ref_count, attrs[sequence_ordinal],
          &lane_result_types[sequence_ordinal], 1, tied_results,
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
      context, rule_set, source_op, state, resolved_emit, NULL, NULL,
      &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, resolved_emit, low_operands));

  const loom_type_t result_type = loom_type_table_get(
      &context->module->types,
      loom_low_lower_resolved_emit_lane_result_type_id(resolved_emit, 0));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_type), 1);

  const uint32_t lane_count =
      loom_low_lower_resolved_emit_lane_plan(resolved_emit)->lane_count;
  const uint16_t accumulator_index = emit->accumulator_operand_index;
  loom_value_id_t accumulator = low_operands[accumulator_index];
  if (seed_first_lane) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_operand(
        context, source_op, resolved_emit, accumulator_index, accumulator, 0,
        &accumulator));
  }
  const uint32_t first_lane_index = seed_first_lane || skip_first_lane ? 1 : 0;

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
    for (uint32_t lane_index = first_lane_index; lane_index < lane_count;
         ++lane_index) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_operand(
          context, source_op, resolved_emit, term_operand_index,
          low_operands[term_operand_index], lane_index,
          &lane_terms[term_index++]));
    }
    IREE_ASSERT_EQ(term_index, term_count);

    for (uint32_t step = 1; step < term_count; step <<= 1) {
      for (uint32_t lane_index = 0; lane_index + step < term_count;
           lane_index += step << 1) {
        lane_operands[emit->accumulator_operand_index] = lane_terms[lane_index];
        lane_operands[term_operand_index] = lane_terms[lane_index + step];
        IREE_RETURN_IF_ERROR(
            loom_low_lower_rule_materialize_descriptor_operands(
                context, source_op, emit, lane_operands));
        loom_op_t* lane_op = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
            context, &resolved_emit->descriptor, lane_operands,
            emit->operand_ref_count, loom_make_named_attr_slice(NULL, 0),
            &result_type, 1, NULL, 0, source_op->location, &lane_op));
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
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_operand(
          context, source_op, resolved_emit, operand_index,
          low_operands[operand_index], lane_index,
          &lane_operands[operand_index]));
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_materialize_descriptor_operands(
        context, source_op, emit, lane_operands));
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
    const loom_low_lower_selected_source_memory_t* source_memory,
    const loom_op_t* const* source_nodes, uint8_t source_node_count) {
  IREE_ASSERT(rule->emit_count == 0 || resolved_emits != NULL);
  IREE_ASSERT_EQ(source_node_count,
                 (uint8_t)(loom_low_lower_rule_source_node_count(rule) + 1));

  loom_low_lower_rule_emit_state_t state = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_state_initialize(
      context, source_op, source_nodes, source_node_count, rule, &state));
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    const uint16_t emit_ref_index = (uint16_t)(rule->action.emit_start + i);
    const loom_low_lower_emit_t* emit =
        loom_low_lower_rule_set_emit_at(rule_set, emit_ref_index);
    const loom_low_lower_resolved_emit_t* resolved_emit = &resolved_emits[i];
    IREE_ASSERT(resolved_emit->emit == emit);
    if (emit->kind == LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_emit_descriptor_op_per_lane_sequence(
              context, rule_set, source_op, &state, rule, resolved_emits, i));
      break;
    }
    switch (emit->kind) {
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op(
            context, rule_set, source_op, &state, resolved_emit,
            source_memory));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_const(
            context, rule_set, source_op, &state, resolved_emit,
            source_memory != NULL ? &source_memory->access : NULL));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_COND_BR: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_cond_br(
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
            "per-lane sequence emits are handled before dispatch");
        IREE_BUILTIN_UNREACHABLE();
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES: {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_rule_emit_descriptor_op_accumulate_lanes(
                context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_REGISTER_SLICE:
      case LOOM_LOW_LOWER_EMIT_REGISTER_SLICE_CLASS: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_register_slice(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_REGISTER_CONCAT: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_register_concat(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_REGISTER_COPY: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_register_copy(
            context, rule_set, source_op, &state, resolved_emit,
            source_memory));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_REGISTER_MOVE: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_register_move(
            context, rule_set, source_op, &state, resolved_emit));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE("unknown generated lower emit kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  if (rule->emit_count != 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_bind_aliases(context, rule_set, source_op, rule));
  loom_low_lower_rule_elide_results(context, rule_set, source_op, rule);
  return iree_ok_status();
}
