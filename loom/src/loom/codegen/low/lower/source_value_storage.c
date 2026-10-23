// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_value_storage.h"

#include "loom/analysis/contract_vector.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/lower/rule_value.h"
#include "loom/codegen/low/lower/source_plan.h"
#include "loom/codegen/low/lower/structural_plan.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"

bool loom_low_lower_source_value_storage_op_requires_emission(
    const loom_op_t* source_op, loom_trait_flags_t traits) {
  if (source_op->result_count == 0 || source_op->region_count != 0 ||
      source_op->tied_result_count != 0) {
    return true;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_HINT |
                                   LOOM_TRAIT_CONVERGENT |
                                   LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return true;
  }
  return loom_traits_may_read(traits) || loom_traits_may_write(traits);
}

bool loom_low_lower_source_value_storage_op_is_identity(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_trait_flags_t traits) {
  if ((!loom_traits_are_fact_identity(traits) &&
       !loom_traits_are_value_alias(traits)) ||
      loom_low_lower_source_value_storage_op_requires_emission(source_op,
                                                               traits)) {
    return false;
  }
  if (loom_traits_are_fact_identity(traits)) {
    return true;
  }
  const loom_target_contract_index_t* index = context->policy->contract.index;
  return index == NULL ||
         loom_target_contract_index_lookup_kind(index, source_op->kind)
                 .case_count == 0;
}

static void loom_low_lower_mark_value_storage_required(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_low_lower_value_flags_t required_flags =
      LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED |
      (context->lowering->source_plan.is_analyzing_storage_demands
           ? 0
           : LOOM_LOW_LOWER_VALUE_STORAGE_BASELINE_REQUIRED);
  // Storage demand flows backward through structural identities. The required
  // bit is both the result and the visited set, so even shared identity chains
  // are traversed at most once across the whole function plan.
  while (true) {
    const loom_value_ordinal_t source_ordinal =
        loom_low_lowering_frame_value_ordinal(context->lowering,
                                              source_value_id);
    loom_low_lower_value_flags_t* storage_flags =
        &context->lowering->source_plan.value_flags[source_ordinal];
    if (iree_all_bits_set(*storage_flags, required_flags)) {
      return;
    }
    *storage_flags |= required_flags;

    const loom_value_t* source_value =
        loom_module_value(context->module, source_value_id);
    if (loom_value_is_block_arg(source_value)) {
      return;
    }
    const loom_op_t* defining_op = loom_value_def_op(source_value);
    if (defining_op == NULL) {
      return;
    }
    const loom_trait_flags_t traits = defining_op->traits;
    const uint16_t result_ordinal = loom_value_def_index(source_value);
    if ((loom_traits_are_fact_identity(traits) ||
         loom_traits_are_value_alias(traits)) &&
        !loom_low_lower_source_value_storage_op_is_identity(
            context, defining_op, traits)) {
      return;
    }
    if (loom_traits_are_fact_identity(traits)) {
      IREE_ASSERT_EQ(defining_op->operand_count, defining_op->result_count);
      IREE_ASSERT_LT(result_ordinal, defining_op->operand_count);
      source_value_id = loom_op_const_operands(defining_op)[result_ordinal];
      continue;
    }
    if (loom_traits_are_value_alias(traits)) {
      IREE_ASSERT_EQ(result_ordinal, 0);
      IREE_ASSERT(defining_op->operand_count >= 1);
      source_value_id = loom_op_const_operands(defining_op)[0];
      continue;
    }
    if (loom_buffer_assume_same_root_isa(defining_op)) {
      IREE_ASSERT_EQ(result_ordinal, 0);
      source_value_id = loom_buffer_assume_same_root_buffer(defining_op);
      continue;
    }
    return;
  }
}

void loom_low_lower_require_source_value_storage(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  loom_low_lower_mark_value_storage_required(context, source_value_id);
}

void loom_low_lower_require_source_operands_storage(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    loom_low_lower_mark_value_storage_required(context, operands[i]);
  }
}

static bool loom_low_lower_value_storage_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  return iree_any_bit_set(
      context->lowering->source_plan.value_flags[source_ordinal],
      LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED);
}

bool loom_low_lower_source_value_storage_result_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  return loom_low_lower_value_storage_required(context, source_value_id) ||
         loom_module_value_has_type_uses(context->module, source_value_id);
}

static bool loom_low_lower_rule_value_ref_source_value(
    const loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan,
    uint16_t value_ref_index, loom_value_id_t* out_source_value_id) {
  const loom_low_lower_rule_set_t* rule_set = selected_plan->rule_set;
  return loom_low_lower_rule_resolve_source_value_from_nodes(
      context->module, context->fact_table,
      (loom_target_contract_vector_lane_projection_t){0}, rule_set,
      selected_plan->source_op, selected_plan->data.source_nodes,
      selected_plan->source_node_count, value_ref_index, out_source_value_id);
}

static bool loom_low_lower_value_has_fact_storage_demand(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  return iree_any_bit_set(
      context->lowering->source_plan.value_flags[source_ordinal],
      LOOM_LOW_LOWER_VALUE_STORAGE_FACT_REFERENCE);
}

static bool loom_low_lower_rule_value_ref_binds_source_value(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, const loom_op_t* const* source_nodes,
    uint8_t source_node_count, uint16_t value_ref_index,
    loom_value_id_t source_value_id) {
  if (rule_set->value_refs[value_ref_index].kind !=
      LOOM_LOW_LOWER_VALUE_REF_RESULT) {
    return false;
  }
  const loom_value_slice_t result_span =
      loom_low_lower_rule_value_ref_field_span_from_nodes(
          module, rule_set, source_op, source_nodes, source_node_count,
          value_ref_index);
  for (iree_host_size_t i = 0; i < result_span.count; ++i) {
    if (result_span.values[i] == source_value_id) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_rule_binds_source_value(
    const loom_module_t* module, const loom_low_lower_rule_set_t* rule_set,
    const loom_op_t* source_op, const loom_op_t* const* source_nodes,
    uint8_t source_node_count, const loom_low_lower_rule_t* rule,
    loom_value_id_t source_value_id) {
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const uint16_t emit_ref_index =
        (uint16_t)(rule->action.emit_start + emit_ordinal);
    const loom_low_lower_emit_t* emit =
        loom_low_lower_rule_set_emit_at(rule_set, emit_ref_index);
    const uint16_t result_ref_start =
        iree_any_bit_set(emit->flags,
                         LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS)
            ? emit->result_bind_ref_start
            : emit->result_type.value_ref_start;
    for (uint16_t result_ordinal = 0; result_ordinal < emit->result_ref_count;
         ++result_ordinal) {
      const uint16_t value_ref_index =
          (uint16_t)(result_ref_start + result_ordinal);
      if (loom_low_lower_rule_value_ref_binds_source_value(
              module, rule_set, source_op, source_nodes, source_node_count,
              value_ref_index, source_value_id)) {
        return true;
      }
    }
  }
  if (rule->emit_count == 0) {
    for (uint16_t alias_ordinal = 0;
         alias_ordinal < rule->metadata.value.alias_ref_count;
         ++alias_ordinal) {
      const uint16_t result_ref_index =
          (uint16_t)(rule->action.alias_ref_start + alias_ordinal * 2 + 1);
      if (loom_low_lower_rule_value_ref_binds_source_value(
              module, rule_set, source_op, source_nodes, source_node_count,
              result_ref_index, source_value_id)) {
        return true;
      }
    }
  }
  return false;
}

bool loom_low_lower_source_value_storage_rule_preserves_fact_demands(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_op_t* const* source_nodes, uint8_t source_node_count,
    const loom_low_lower_rule_t* rule) {
  if (context->lowering->source_plan.fact_storage_demand_count == 0) {
    return true;
  }
  for (uint8_t source_node_index = 0; source_node_index < source_node_count;
       ++source_node_index) {
    const loom_op_t* source_node = source_op;
    if (source_node_index != 0) {
      IREE_ASSERT(source_nodes != NULL);
      source_node = source_nodes[source_node_index];
    }
    const loom_value_id_t* results = loom_op_const_results(source_node);
    for (uint16_t result_ordinal = 0;
         result_ordinal < source_node->result_count; ++result_ordinal) {
      const loom_value_id_t result = results[result_ordinal];
      if (loom_low_lower_value_has_fact_storage_demand(context, result) &&
          !loom_low_lower_rule_binds_source_value(
              context->module, rule_set, source_op, source_nodes,
              source_node_count, rule, result)) {
        return false;
      }
    }
  }
  return true;
}

static void loom_low_lower_mark_rule_value_ref_storage_demand(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan,
    uint16_t value_ref_index) {
  loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_low_lower_rule_value_ref_source_value(
          context, selected_plan, value_ref_index, &source_value_id)) {
    return;
  }
  const loom_low_lower_value_ref_t* value_ref =
      &selected_plan->rule_set->value_refs[value_ref_index];
  if (value_ref->kind == LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND ||
      value_ref->kind ==
          LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND ||
      value_ref->kind ==
          LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND) {
    const loom_value_ordinal_t source_ordinal =
        loom_low_lowering_frame_value_ordinal(context->lowering,
                                              source_value_id);
    loom_low_lower_value_flags_t* storage_flags =
        &context->lowering->source_plan.value_flags[source_ordinal];
    if (!iree_any_bit_set(*storage_flags,
                          LOOM_LOW_LOWER_VALUE_STORAGE_FACT_REFERENCE)) {
      *storage_flags |= LOOM_LOW_LOWER_VALUE_STORAGE_FACT_REFERENCE;
      ++context->lowering->source_plan.fact_storage_demand_count;
    }
  }
  loom_low_lower_mark_value_storage_required(context, source_value_id);
}

static bool loom_low_lower_rule_value_ref_uses_source_memory_plan(
    const loom_low_lower_rule_set_t* rule_set, uint16_t value_ref_index) {
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ELEMENT_STRIDE:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ROOT:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_emit_uses_source_memory_plan(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  for (uint16_t operand_ordinal = 0; operand_ordinal < emit->operand_ref_count;
       ++operand_ordinal) {
    const uint16_t value_ref_index =
        (uint16_t)(emit->operand_ref_start + operand_ordinal);
    if (loom_low_lower_rule_value_ref_uses_source_memory_plan(
            rule_set, value_ref_index)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_emit_materializes_source_memory_address(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_emit_t* emit) {
  for (uint16_t operand_ordinal = 0; operand_ordinal < emit->operand_ref_count;
       ++operand_ordinal) {
    const uint16_t value_ref_index =
        (uint16_t)(emit->operand_ref_start + operand_ordinal);
    if (rule_set->value_refs[value_ref_index].kind ==
        LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS) {
      return true;
    }
  }
  return false;
}

static void loom_low_lower_mark_source_memory_realization_seen(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  loom_low_lower_value_flags_t* storage_flags =
      &context->lowering->source_plan.value_flags[source_ordinal];
  if (iree_any_bit_set(*storage_flags,
                       LOOM_LOW_LOWER_VALUE_STORAGE_MEMORY_REALIZATION_SEEN)) {
    loom_low_lower_mark_value_storage_required(context, source_value_id);
  } else {
    *storage_flags |= LOOM_LOW_LOWER_VALUE_STORAGE_MEMORY_REALIZATION_SEEN;
  }
}

static void loom_low_lower_mark_reused_source_memory_realizations(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* access) {
  for (uint8_t i = 0; i < access->dynamic_realization_count; ++i) {
    loom_low_lower_mark_source_memory_realization_seen(
        context, access->dynamic_realizations[i].term.index);
  }
}

static void loom_low_lower_mark_source_memory_access_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_lower_source_memory_address_materializer_t*
        address_materializer,
    const loom_low_source_memory_access_plan_t* access) {
  uint8_t first_canonical_term = 0;
  if (address_materializer != NULL) {
    const loom_value_id_t base_value_id =
        address_materializer->base_kind ==
                LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_BASE_VIEW
            ? loom_low_source_memory_access_base_view_value_id(access)
            : access->root_value_id;
    IREE_ASSERT_NE(base_value_id, LOOM_VALUE_ID_INVALID);
    loom_low_lower_mark_value_storage_required(context, base_value_id);
    if (address_materializer->coordinate_unit_byte_count == 1 &&
        address_materializer->coordinate_type ==
            LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET &&
        access->dynamic_view_base_term_count != 0 &&
        access->dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID) {
      loom_low_lower_mark_value_storage_required(
          context, access->dynamic_view_base_value_id);
      first_canonical_term = access->dynamic_view_base_term_count;
    }
  } else if (loom_low_lower_rule_source_memory_reuses_view_base(source_memory,
                                                                access)) {
    loom_low_lower_mark_value_storage_required(
        context, access->dynamic_view_base_value_id);
    first_canonical_term = access->dynamic_view_base_term_count;
  } else if (
      loom_low_source_memory_access_dynamic_offset_has_materialized_view_base(
          access)) {
    // Canonical address terms normally replace the source view-base
    // expression. Preserve the exact view base when that expression also
    // carries an integer-to-index conversion required by target lowering.
    for (uint8_t term_ordinal = 0;
         term_ordinal < access->dynamic_view_base_term_count; ++term_ordinal) {
      const loom_type_t term_type = loom_module_value_type(
          context->module, access->dynamic_terms[term_ordinal].index);
      if (!loom_type_equal(term_type,
                           loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)) &&
          !loom_type_equal(term_type,
                           loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET))) {
        loom_low_lower_mark_value_storage_required(
            context, access->dynamic_view_base_value_id);
        break;
      }
    }
  }
  const loom_low_source_memory_dynamic_component_t* component =
      loom_low_source_memory_access_retained_component(
          access, first_canonical_term,
          address_materializer != NULL
              ? address_materializer->coordinate_unit_byte_count
              : 1);
  if (component != NULL) {
    loom_low_lower_mark_value_storage_required(context, component->term->index);
    for (uint8_t i = 0; i < component->term->stride_value_count; ++i) {
      loom_low_lower_mark_value_storage_required(
          context, component->term->stride_values[i]);
    }
  }
  for (uint8_t term_ordinal = first_canonical_term;
       term_ordinal < access->dynamic_term_count; ++term_ordinal) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->dynamic_terms[term_ordinal];
    loom_low_lower_mark_value_storage_required(context, term->index);
    for (uint8_t stride_ordinal = 0; stride_ordinal < term->stride_value_count;
         ++stride_ordinal) {
      loom_low_lower_mark_value_storage_required(
          context, term->stride_values[stride_ordinal]);
    }
  }
}

void loom_low_lower_require_source_memory_complete_byte_offset_storage(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* access) {
  loom_low_lower_mark_value_storage_required(context, access->root_value_id);
  const loom_low_lower_source_memory_t source_memory = {
      .flags = LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_REUSE_VIEW_BASE,
  };
  loom_low_lower_mark_source_memory_access_storage_demands(
      context, &source_memory, /*address_materializer=*/NULL, access);
}

static void loom_low_lower_mark_rule_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_low_lower_rule_set_t* rule_set = selected_plan->rule_set;
  const loom_low_lower_rule_t* rule = selected_plan->rule;
  const loom_low_source_memory_access_plan_t* source_memory_access =
      selected_plan->source_memory != NULL
          ? &selected_plan->source_memory->access
          : NULL;
  IREE_ASSERT(rule_set != NULL);
  IREE_ASSERT(rule != NULL);
  bool uses_dynamic_byte_offset = false;
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const uint16_t emit_ref_index =
        (uint16_t)(rule->action.emit_start + emit_ordinal);
    const loom_low_lower_emit_t* emit =
        loom_low_lower_rule_set_emit_at(rule_set, emit_ref_index);
    for (uint16_t operand_ordinal = 0;
         operand_ordinal < emit->operand_ref_count; ++operand_ordinal) {
      const uint16_t value_ref_index =
          (uint16_t)(emit->operand_ref_start + operand_ordinal);
      const loom_low_lower_value_ref_t* value_ref =
          &rule_set->value_refs[value_ref_index];
      uses_dynamic_byte_offset |=
          value_ref->kind ==
          LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET;
      if (value_ref->kind == LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ROOT) {
        IREE_ASSERT(source_memory_access != NULL);
        loom_low_lower_mark_value_storage_required(
            context, source_memory_access->root_value_id);
        continue;
      }
      if (value_ref->kind ==
          LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ELEMENT_STRIDE) {
        const loom_low_source_memory_axis_byte_stride_t* stride =
            &source_memory_access->axis_byte_strides[value_ref->index];
        for (uint8_t i = 0; i < stride->dynamic_factor_count; ++i) {
          loom_low_lower_mark_value_storage_required(
              context, stride->dynamic_factors[i]);
        }
        continue;
      }
      loom_low_lower_mark_rule_value_ref_storage_demand(context, selected_plan,
                                                        value_ref_index);
    }
    if (emit->source_memory_ordinal == 0 ||
        !loom_low_lower_emit_uses_source_memory_plan(rule_set, emit)) {
      continue;
    }
    IREE_ASSERT(source_memory_access != NULL);
    const loom_low_lower_source_memory_t* source_memory =
        &rule_set->source_memories[emit->source_memory_ordinal - 1];
    const loom_low_lower_source_memory_address_materializer_t*
        address_materializer =
            loom_low_lower_emit_materializes_source_memory_address(rule_set,
                                                                   emit)
                ? loom_low_lower_rule_set_source_memory_address_materializer(
                      rule_set, source_memory)
                : NULL;
    loom_low_lower_mark_source_memory_access_storage_demands(
        context, source_memory, address_materializer, source_memory_access);
  }
  if (uses_dynamic_byte_offset) {
    IREE_ASSERT(source_memory_access != NULL);
    loom_low_lower_mark_reused_source_memory_realizations(context,
                                                          source_memory_access);
  }
  if (rule->emit_count != 0) {
    return;
  }

  if (iree_all_bits_set(rule->flags,
                        LOOM_LOW_LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS)) {
    IREE_ASSERT_EQ(rule->metadata.value.alias_ref_count, 1);
    const loom_value_slice_t source_span =
        loom_low_lower_rule_value_ref_field_span_from_nodes(
            context->module, rule_set, selected_plan->source_op,
            selected_plan->data.source_nodes, selected_plan->source_node_count,
            rule->action.alias_ref_start);
    for (iree_host_size_t i = 0; i < source_span.count; ++i) {
      loom_low_lower_mark_value_storage_required(context,
                                                 source_span.values[i]);
    }
  } else {
    for (uint16_t alias_ordinal = 0;
         alias_ordinal < rule->metadata.value.alias_ref_count;
         ++alias_ordinal) {
      const uint16_t value_ref_index =
          (uint16_t)(rule->action.alias_ref_start + alias_ordinal * 2);
      loom_low_lower_mark_rule_value_ref_storage_demand(context, selected_plan,
                                                        value_ref_index);
    }
  }
  if (rule->metadata.value.alias_ref_count == 0) {
    return;
  }

  // Canonical source-memory plans own the storage demands for addresses through
  // buffer.view aliases. Requiring the original byte-offset expression here
  // would make an equivalent source realization mandatory target IR.
  const loom_op_t* source_op = selected_plan->source_op;
  if (loom_buffer_view_isa(source_op)) {
    return;
  }

  // Other variadic aliases can erase projection ops whose non-exact operands
  // remain referenced by facts consumed by later plans.
  const loom_value_fact_table_t* fact_table = context->fact_table;
  const loom_op_vtable_t* vtable = loom_op_vtable(context->module, source_op);
  if (vtable == NULL || !iree_any_bit_set(vtable->vtable_flags,
                                          LOOM_OP_VTABLE_VARIADIC_OPERANDS)) {
    return;
  }
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = vtable->fixed_operand_count; i < source_op->operand_count;
       ++i) {
    int64_t exact_value = 0;
    if (fact_table != NULL &&
        loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(fact_table, operands[i]),
            &exact_value)) {
      continue;
    }
    loom_low_lower_mark_value_storage_required(context, operands[i]);
  }
}

static void loom_low_lower_mark_descriptor_matrix_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_low_lower_descriptor_matrix_plan_t* plan =
      (const loom_low_lower_descriptor_matrix_plan_t*)
          selected_plan->data.target_plan.target_data;
  IREE_ASSERT(plan != NULL);
  loom_low_lower_require_source_operands_storage(context,
                                                 selected_plan->source_op);
  const loom_contract_operand_t* operands[] = {
      &plan->contract_request.lhs,
      &plan->contract_request.rhs,
      &plan->contract_request.accumulator,
      &plan->contract_request.result,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
    const loom_contract_encoded_operand_t* encoded = &operands[i]->encoded;
    for (uint8_t key = 0; key < LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_COUNT_;
         ++key) {
      if (!iree_any_bit_set(encoded->required_auxiliary_operands,
                            loom_contract_auxiliary_operand_key_flag(key))) {
        continue;
      }
      const loom_contract_value_ref_t ref = encoded->auxiliary_value_refs[key];
      IREE_ASSERT(loom_contract_value_ref_is_present(ref));
      loom_low_lower_mark_value_storage_required(
          context, loom_contract_value_ref_value_id(ref));
    }
  }
}

static void loom_low_lower_mark_callback_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  IREE_ASSERT_EQ(selected_plan->kind, LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK);
  if (context->policy->mark_plan_storage_demands.fn != NULL) {
    context->policy->mark_plan_storage_demands.fn(
        context->policy->mark_plan_storage_demands.user_data, context,
        selected_plan->source_op, selected_plan->data.target_plan);
    return;
  }
  loom_low_lower_require_source_operands_storage(context,
                                                 selected_plan->source_op);
}

void loom_low_lower_source_value_storage_mark_structural_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_trait_flags_t traits) {
  if (loom_traits_are_fact_identity(traits) ||
      loom_traits_are_value_alias(traits) ||
      loom_buffer_assume_same_root_isa(source_op)) {
    const loom_value_id_t* results = loom_op_const_results(source_op);
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      if (loom_module_value_has_type_uses(context->module, results[i])) {
        loom_low_lower_mark_value_storage_required(context, results[i]);
      }
    }
    return;
  }
  if (loom_cfg_cond_br_isa(source_op) &&
      loom_low_lower_structural_branch_exact_bool(context, source_op, NULL)) {
    return;
  }
  loom_low_lower_require_source_operands_storage(context, source_op);
}

static bool loom_low_lower_source_op_requires_emission(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  return loom_low_lower_source_value_storage_op_requires_emission(source_op,
                                                                  traits);
}

static bool loom_low_lower_source_op_result_storage_required(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t* results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_ASSERT_NE(results[i], LOOM_VALUE_ID_INVALID);
    if (loom_low_lower_source_value_storage_result_required(context,
                                                            results[i])) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_selected_plan_storage_required(
    const loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const uint8_t source_node_count = selected_plan->source_node_count != 0
                                        ? selected_plan->source_node_count
                                        : 1;
  for (uint8_t i = 0; i < source_node_count; ++i) {
    const loom_op_t* source_op = selected_plan->source_op;
    if (i != 0) {
      IREE_ASSERT(selected_plan->data.source_nodes != NULL);
      source_op = selected_plan->data.source_nodes[i];
    }
    if (loom_low_lower_source_op_requires_emission(context, source_op) ||
        loom_low_lower_source_op_result_storage_required(context, source_op)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_selected_plan_result_storage_required(
    const loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const uint8_t source_node_count = selected_plan->source_node_count != 0
                                        ? selected_plan->source_node_count
                                        : 1;
  for (uint8_t i = 0; i < source_node_count; ++i) {
    const loom_op_t* source_op = selected_plan->source_op;
    if (i != 0) {
      IREE_ASSERT(selected_plan->data.source_nodes != NULL);
      source_op = selected_plan->data.source_nodes[i];
    }
    if (loom_low_lower_source_op_result_storage_required(context, source_op)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_can_elide_selected_plan(
    const loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  return selected_plan->kind != LOOM_LOW_LOWER_SELECTED_PLAN_INVOKE &&
         selected_plan->kind != LOOM_LOW_LOWER_SELECTED_PLAN_PROJECTED_CALL &&
         !loom_low_lower_selected_plan_storage_required(context, selected_plan);
}

static void loom_low_lower_mark_selected_plan_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  switch (selected_plan->kind) {
    case LOOM_LOW_LOWER_SELECTED_PLAN_RULE:
      loom_low_lower_mark_rule_storage_demands(context, selected_plan);
      break;
    case LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX:
      loom_low_lower_mark_descriptor_matrix_storage_demands(context,
                                                            selected_plan);
      break;
    case LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK:
      loom_low_lower_mark_callback_plan_storage_demands(context, selected_plan);
      break;
    case LOOM_LOW_LOWER_SELECTED_PLAN_FUNCTION_STORAGE:
      // The selected plan retains the extent; no source operand is emitted.
      break;
    case LOOM_LOW_LOWER_SELECTED_PLAN_INVOKE:
    case LOOM_LOW_LOWER_SELECTED_PLAN_PROJECTED_CALL:
      loom_low_lower_require_source_operands_storage(context,
                                                     selected_plan->source_op);
      break;
    case LOOM_LOW_LOWER_SELECTED_PLAN_STRUCTURAL:
      loom_low_lower_source_value_storage_mark_structural_demands(
          context, selected_plan->source_op,
          loom_op_effective_traits(context->module, selected_plan->source_op));
      break;
  }
}

void loom_low_lower_source_value_storage_analyze(
    loom_low_lower_context_t* context) {
  loom_low_lower_source_plan_t* source_plan = &context->lowering->source_plan;
  source_plan->is_analyzing_storage_demands = true;
  for (iree_host_size_t i = source_plan->selected_plan_count; i > 0; --i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &source_plan->selected_plans[i - 1];
    if (!iree_any_bit_set(selected_plan->flags,
                          LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED) &&
        loom_low_lower_selected_plan_storage_required(context, selected_plan)) {
      loom_low_lower_mark_selected_plan_storage_demands(context, selected_plan);
    }
  }
  for (iree_host_size_t i = 0; i < source_plan->selected_plan_count; ++i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &source_plan->selected_plans[i];
    if (iree_any_bit_set(selected_plan->flags,
                         LOOM_LOW_LOWER_SELECTED_PLAN_CLAIMED)) {
      continue;
    }
    if (loom_low_lower_can_elide_selected_plan(context, selected_plan)) {
      selected_plan->flags |= LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED;
    }
  }
  source_plan->is_analyzing_storage_demands = false;
}

bool loom_low_lower_source_value_storage_plans_preserve_fact_demands(
    const loom_low_lower_context_t* context) {
  const loom_low_lower_source_plan_t* source_plan =
      &context->lowering->source_plan;
  for (iree_host_size_t i = 0; i < source_plan->selected_plan_count; ++i) {
    const loom_low_lower_selected_plan_t* selected_plan =
        &source_plan->selected_plans[i];
    if (selected_plan->kind != LOOM_LOW_LOWER_SELECTED_PLAN_RULE ||
        selected_plan->rule == NULL) {
      continue;
    }
    if (!loom_low_lower_source_value_storage_rule_preserves_fact_demands(
            context, selected_plan->rule_set, selected_plan->source_op,
            selected_plan->data.source_nodes, selected_plan->source_node_count,
            selected_plan->rule)) {
      return false;
    }
  }
  return true;
}

static void loom_low_lower_source_value_storage_reset_demands(
    loom_low_lower_context_t* context) {
  loom_low_lower_source_plan_t* source_plan = &context->lowering->source_plan;
  const loom_value_ordinal_t value_count =
      context->lowering->value_domain.value_count;
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    source_plan->value_flags[i] &=
        LOOM_LOW_LOWER_VALUE_STORAGE_FACT_REFERENCE |
        LOOM_LOW_LOWER_VALUE_STORAGE_BASELINE_REQUIRED |
        LOOM_LOW_LOWER_VALUE_INHERITED_TYPE |
        LOOM_LOW_LOWER_VALUE_MEMORY_RECURRENCE_PRODUCER;
    if (iree_any_bit_set(source_plan->value_flags[i],
                         LOOM_LOW_LOWER_VALUE_STORAGE_BASELINE_REQUIRED)) {
      source_plan->value_flags[i] |= LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED;
    }
  }
}

void loom_low_lower_source_value_storage_prepare_refinement(
    loom_low_lower_context_t* context) {
  loom_low_lower_source_value_storage_reset_demands(context);
  loom_low_lower_source_plan_t* source_plan = &context->lowering->source_plan;
  source_plan->memory.cursor = source_plan->memory.first;
  source_plan->memory.current = NULL;
  source_plan->selected_plan_count = 0;
  source_plan->selected_plan_emit_index = 0;
}

void loom_low_lower_source_value_storage_refine_selected_plans(
    loom_low_lower_context_t* context) {
  loom_low_lower_source_value_storage_reset_demands(context);
  loom_low_lower_source_plan_t* source_plan = &context->lowering->source_plan;
  // Newly selected realizations may retain source dependencies after the first
  // storage pass. Reactivate their producers before the reverse demand walk so
  // each producer can propagate demand to its own operands. Claimed results are
  // supplied by their realization and remain elided.
  for (iree_host_size_t i = 0; i < source_plan->selected_plan_count; ++i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &source_plan->selected_plans[i];
    if (!iree_all_bits_set(selected_plan->flags,
                           LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED) ||
        iree_any_bit_set(selected_plan->flags,
                         LOOM_LOW_LOWER_SELECTED_PLAN_CLAIMED)) {
      continue;
    }
    if (loom_low_lower_selected_plan_result_storage_required(context,
                                                             selected_plan)) {
      selected_plan->flags &= ~LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED;
    }
  }
  loom_low_lower_source_value_storage_analyze(context);
}
