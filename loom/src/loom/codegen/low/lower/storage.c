// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/storage.h"

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
#include "loom/codegen/low/lower/plan.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scf/ops.h"

static void loom_low_lower_mark_value_storage_required(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  context->lowering.value_storage_flags[source_ordinal] |=
      LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED;
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
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  return iree_any_bit_set(context->lowering.value_storage_flags[source_ordinal],
                          LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED);
}

bool loom_low_lower_result_storage_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  return loom_low_lower_value_storage_required(context, source_value_id) ||
         loom_module_value_has_type_uses(context->module, source_value_id);
}

static void loom_low_lower_mark_structural_value_storage_required(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  loom_low_lower_mark_value_storage_required(context, source_value_id);
  if (context->policy->materialize_structural_operand.mark_storage_demands !=
      NULL) {
    context->policy->materialize_structural_operand.mark_storage_demands(
        context->policy->materialize_structural_operand.user_data, context,
        source_value_id);
  }
}

static void loom_low_lower_mark_structural_value_slice_storage_required(
    loom_low_lower_context_t* context, loom_value_slice_t values) {
  for (uint16_t i = 0; i < values.count; ++i) {
    loom_low_lower_mark_structural_value_storage_required(context,
                                                          values.values[i]);
  }
}

bool loom_low_lower_cfg_cond_br_exact_bool(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_condition) {
  if (out_condition != NULL) *out_condition = false;
  if (!loom_cfg_cond_br_isa(source_op) ||
      context->lowering.fact_table == NULL) {
    return false;
  }
  const loom_value_facts_t facts = loom_value_fact_table_lookup(
      context->lowering.fact_table, loom_cfg_cond_br_condition(source_op));
  bool condition = false;
  if (!loom_value_facts_as_exact_bool(facts, &condition)) {
    return false;
  }
  if (out_condition != NULL) *out_condition = condition;
  return true;
}

static bool loom_low_lower_rule_value_ref_source_value(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t* out_source_value_id) {
  *out_source_value_id = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND: {
      const loom_op_vtable_t* vtable =
          loom_op_vtable(context->module, source_op);
      const loom_value_slice_t span =
          loom_op_operand_field_span(vtable, source_op, value_ref->index);
      IREE_ASSERT_LT(value_ref->element_index, span.count);
      *out_source_value_id = span.values[value_ref->element_index];
      return true;
    }
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      IREE_ASSERT_LT(value_ref->index, source_op->result_count);
      *out_source_value_id = loom_op_const_results(source_op)[value_ref->index];
      return true;
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
      return false;
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
      return false;
    default:
      IREE_ASSERT_UNREACHABLE("unknown source-low value ref kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_low_lower_rule_value_ref_uses_source_memory_plan(
    const loom_low_lower_rule_set_t* rule_set, uint16_t value_ref_index) {
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
    case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
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

enum loom_low_lower_source_memory_storage_demand_flag_bits_e {
  LOOM_LOW_LOWER_SOURCE_MEMORY_STORAGE_DEMAND_FLAG_COMPLETE_ADDRESS = 1u << 0,
};
typedef uint32_t loom_low_lower_source_memory_storage_demand_flags_t;

static void loom_low_lower_mark_source_memory_access_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access,
    loom_low_lower_source_memory_storage_demand_flags_t flags) {
  uint8_t first_canonical_term = 0;
  if (iree_any_bit_set(
          flags,
          LOOM_LOW_LOWER_SOURCE_MEMORY_STORAGE_DEMAND_FLAG_COMPLETE_ADDRESS)) {
    const loom_value_id_t base_value_id =
        source_memory->address_base_kind ==
                LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_BASE_VIEW
            ? loom_low_source_memory_access_base_view_value_id(access)
            : access->root_value_id;
    IREE_ASSERT_NE(base_value_id, LOOM_VALUE_ID_INVALID);
    loom_low_lower_mark_value_storage_required(context, base_value_id);
    if (source_memory->address_coordinate_unit_byte_count == 1 &&
        source_memory->address_coordinate_type ==
            LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET &&
        access->dynamic_view_base_term_count != 0 &&
        access->dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID) {
      loom_low_lower_mark_value_storage_required(
          context, access->dynamic_view_base_value_id);
      first_canonical_term = access->dynamic_view_base_term_count;
    }
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

static void loom_low_lower_mark_rule_storage_demands(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_low_lower_rule_set_t* rule_set = selected_plan->rule_set;
  const loom_low_lower_rule_t* rule = selected_plan->rule;
  IREE_ASSERT(rule_set != NULL);
  IREE_ASSERT(rule != NULL);
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const uint16_t emit_index = (uint16_t)(rule->emit_start + emit_ordinal);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    for (uint16_t operand_ordinal = 0;
         operand_ordinal < emit->operand_ref_count; ++operand_ordinal) {
      const uint16_t value_ref_index =
          (uint16_t)(emit->operand_ref_start + operand_ordinal);
      loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
      if (loom_low_lower_rule_value_ref_source_value(
              context, rule_set, selected_plan->source_op, value_ref_index,
              &source_value_id)) {
        loom_low_lower_mark_value_storage_required(context, source_value_id);
      }
    }
    if (emit->source_memory_ordinal == 0 ||
        !loom_low_lower_emit_uses_source_memory_plan(rule_set, emit)) {
      continue;
    }
    IREE_ASSERT(selected_plan->source_memory_access != NULL);
    const loom_low_lower_source_memory_storage_demand_flags_t flags =
        loom_low_lower_emit_materializes_source_memory_address(rule_set, emit)
            ? LOOM_LOW_LOWER_SOURCE_MEMORY_STORAGE_DEMAND_FLAG_COMPLETE_ADDRESS
            : 0;
    const loom_low_lower_source_memory_t* source_memory =
        &rule_set->source_memories[emit->source_memory_ordinal - 1];
    loom_low_lower_mark_source_memory_access_storage_demands(
        context, source_memory, selected_plan->source_memory_access, flags);
  }
  if (iree_all_bits_set(rule->flags,
                        LOOM_LOW_LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS)) {
    IREE_ASSERT_EQ(rule->alias_ref_count, 1);
    const loom_value_slice_t source_span =
        loom_low_lower_rule_value_ref_field_span(context->module, rule_set,
                                                 selected_plan->source_op,
                                                 rule->alias_ref_start);
    for (iree_host_size_t i = 0; i < source_span.count; ++i) {
      loom_low_lower_mark_value_storage_required(context,
                                                 source_span.values[i]);
    }
  } else {
    for (uint16_t alias_ordinal = 0; alias_ordinal < rule->alias_ref_count;
         ++alias_ordinal) {
      const uint16_t value_ref_index =
          (uint16_t)(rule->alias_ref_start + alias_ordinal * 2);
      loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
      if (loom_low_lower_rule_value_ref_source_value(
              context, rule_set, selected_plan->source_op, value_ref_index,
              &source_value_id)) {
        loom_low_lower_mark_value_storage_required(context, source_value_id);
      }
    }
  }
  if (rule->alias_ref_count == 0) return;

  // Canonical source-memory plans own the storage demands for addresses through
  // buffer.view aliases. Requiring the original byte-offset expression here
  // would make an equivalent source realization mandatory target IR.
  const loom_op_t* source_op = selected_plan->source_op;
  if (loom_buffer_view_isa(source_op)) {
    return;
  }

  // Other variadic aliases can erase projection ops whose non-exact operands
  // remain referenced by facts consumed by later plans.
  const loom_value_fact_table_t* fact_table = context->lowering.fact_table;
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
          selected_plan->plan.target_data;
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
        selected_plan->source_op, selected_plan->plan);
    return;
  }
  loom_low_lower_require_source_operands_storage(context,
                                                 selected_plan->source_op);
}

static void loom_low_lower_mark_structural_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (loom_traits_are_fact_identity(traits)) {
    const loom_value_id_t* operands = loom_op_const_operands(source_op);
    for (uint16_t i = 0; i < source_op->operand_count; ++i) {
      loom_low_lower_mark_value_storage_required(context, operands[i]);
    }
    return;
  }
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(source_op->operand_count >= 1);
    loom_low_lower_mark_value_storage_required(
        context, loom_op_const_operands(source_op)[0]);
    return;
  }
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
      loom_low_lower_mark_value_storage_required(
          context, loom_buffer_assume_same_root_buffer(source_op));
      return;
    case LOOM_OP_FUNC_COMPARE_NULL:
      loom_low_lower_mark_structural_value_storage_required(
          context, loom_func_compare_null_function(source_op));
      return;
    case LOOM_OP_FUNC_CALL:
      loom_low_lower_mark_structural_value_slice_storage_required(
          context, loom_func_call_operands(source_op));
      return;
    case LOOM_OP_FUNC_CALL_INDIRECT: {
      const loom_value_id_t* operands = loom_op_const_operands(source_op);
      for (uint16_t i = 0; i < source_op->operand_count; ++i) {
        loom_low_lower_mark_structural_value_storage_required(context,
                                                              operands[i]);
      }
      return;
    }
    case LOOM_OP_FUNC_RETURN:
      loom_low_lower_mark_structural_value_slice_storage_required(
          context, loom_func_return_operands(source_op));
      return;
    case LOOM_OP_CFG_BR:
      loom_low_lower_mark_structural_value_slice_storage_required(
          context, loom_cfg_br_args(source_op));
      return;
    case LOOM_OP_CFG_COND_BR: {
      if (loom_low_lower_cfg_cond_br_exact_bool(context, source_op, NULL)) {
        return;
      }
      loom_low_lower_mark_value_storage_required(
          context, loom_cfg_cond_br_condition(source_op));
      return;
    }
    case LOOM_OP_CFG_SWITCH:
      loom_low_lower_mark_value_storage_required(
          context, loom_cfg_switch_selector(source_op));
      return;
    case LOOM_OP_SCF_FOR:
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_for_lower_bound(source_op));
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_for_upper_bound(source_op));
      loom_low_lower_mark_value_storage_required(context,
                                                 loom_scf_for_step(source_op));
      loom_low_lower_mark_structural_value_slice_storage_required(
          context, loom_scf_for_iter_args(source_op));
      if (loom_scf_for_unroll_factor_is_present(source_op)) {
        loom_low_lower_mark_value_storage_required(
            context, loom_scf_for_unroll_factor(source_op));
      }
      return;
    case LOOM_OP_SCF_IF:
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_if_condition(source_op));
      return;
    case LOOM_OP_SCF_WHILE:
      loom_low_lower_mark_structural_value_slice_storage_required(
          context, loom_scf_while_iter_args(source_op));
      return;
    case LOOM_OP_SCF_CONDITION:
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_condition_condition(source_op));
      loom_low_lower_mark_structural_value_slice_storage_required(
          context, loom_scf_condition_forwarded(source_op));
      return;
    case LOOM_OP_SCF_YIELD:
      loom_low_lower_mark_structural_value_slice_storage_required(
          context, loom_scf_yield_values(source_op));
      return;
    case LOOM_OP_KERNEL_RETURN:
    default:
      return;
  }
}

bool loom_low_lower_source_op_requires_emission(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (source_op->result_count == 0 || source_op->region_count != 0 ||
      source_op->tied_result_count != 0) {
    return true;
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (iree_any_bit_set(traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_HINT |
                                   LOOM_TRAIT_UNIQUE_IDENTITY |
                                   LOOM_TRAIT_CONVERGENT)) {
    return true;
  }
  if (loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
      loom_op_regions_have_write_effects(source_op) ||
      loom_op_regions_have_convergent_effects(source_op) ||
      loom_op_regions_have_hints(context->module, source_op)) {
    return true;
  }
  return false;
}

static bool loom_low_lower_source_op_result_storage_required(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t* results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_ASSERT_NE(results[i], LOOM_VALUE_ID_INVALID);
    if (loom_low_lower_result_storage_required(context, results[i])) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_selected_plan_storage_required(
    const loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_op_t* source_op = selected_plan->source_op;
  return loom_low_lower_source_op_requires_emission(context, source_op) ||
         loom_low_lower_source_op_result_storage_required(context, source_op);
}

static bool loom_low_lower_can_elide_source_storage(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return !loom_low_lower_source_op_requires_emission(context, source_op) &&
         !loom_low_lower_source_op_result_storage_required(context, source_op);
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
  }
}

static void loom_low_lower_mark_region_structural_storage_demands(
    loom_low_lower_context_t* context, loom_region_t* source_region) {
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_lower_op_is_structural(context->module, op)) {
        loom_low_lower_mark_structural_storage_demands(context, op);
      }
      if (loom_low_lower_supported_structured_source_op(context, op)) {
        loom_region_t* const* regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          if (regions[i] != NULL) {
            loom_low_lower_mark_region_structural_storage_demands(context,
                                                                  regions[i]);
          }
        }
      }
    }
  }
}

void loom_low_lower_analyze_storage_demands(loom_low_lower_context_t* context,
                                            loom_region_t* source_body) {
  loom_low_lower_mark_region_structural_storage_demands(context, source_body);
  for (iree_host_size_t i = context->lowering.selected_plan_count; i > 0; --i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &context->lowering.selected_plans[i - 1];
    if (!iree_any_bit_set(selected_plan->flags,
                          LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED) &&
        loom_low_lower_selected_plan_storage_required(context, selected_plan)) {
      loom_low_lower_mark_selected_plan_storage_demands(context, selected_plan);
    }
  }
  for (iree_host_size_t i = 0; i < context->lowering.selected_plan_count; ++i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &context->lowering.selected_plans[i];
    if (loom_low_lower_can_elide_source_storage(context,
                                                selected_plan->source_op)) {
      selected_plan->flags |= LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED;
    }
  }
}
