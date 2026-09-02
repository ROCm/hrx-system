// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_plan.h"

#include <string.h>

#include "loom/analysis/contract_vector.h"
#include "loom/codegen/low/lower/contract_query.h"
#include "loom/codegen/low/lower/lower_context.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
#include "loom/codegen/low/lower/source_query.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"

iree_status_t loom_low_lower_source_plan_check_mapped_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_low_type) {
  uint32_t previous_error_count = context->result->error_count;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_value_id, out_low_type));
  if (loom_type_kind(*out_low_type) == LOOM_TYPE_NONE) {
    if (context->result->error_count == previous_error_count) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("source")),
          loom_param_u64(source_value_id),
      };
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_target_context_error(
          context, source_op, LOOM_ERR_TARGET_027, params,
          IREE_ARRAYSIZE(params)));
    }
  }
  return iree_ok_status();
}

bool loom_low_lower_source_plan_uses_structured_control_flow(
    const loom_low_lower_context_t* context) {
  return context->options->control_flow_lowering ==
         LOOM_LOW_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
}

static bool loom_low_lower_supported_structured_source_op(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_low_lower_source_plan_uses_structured_control_flow(context)) {
    return false;
  }
  switch (source_op->kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_is_structural(const loom_module_t* module,
                                            const loom_op_t* op) {
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (loom_traits_are_fact_identity(traits) ||
      loom_traits_are_value_alias(traits)) {
    return true;
  }
  switch (op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_FUNC_CALL:
    case LOOM_OP_FUNC_RETURN:
    case LOOM_OP_KERNEL_RETURN:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_SCHEDULE_FENCE:
    case LOOM_OP_SCF_YIELD:
      return true;
    default:
      return false;
  }
}

bool loom_low_lower_source_plan_op_is_metadata(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_ENCODING_ASSUME_SPEC:
    case LOOM_OP_ENCODING_DEFINE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED:
    case LOOM_OP_ENCODING_LAYOUT_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_STRIDED:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_uses_policy(const loom_module_t* module,
                                          const loom_op_t* op) {
  return !loom_low_lower_op_is_structural(module, op) &&
         !loom_low_lower_source_plan_op_is_metadata(op->kind);
}

static bool loom_low_lower_op_is_discardable_hint(const loom_module_t* module,
                                                  const loom_op_t* op) {
  if (op->result_count != 0 || op->region_count != 0 ||
      op->tied_result_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  return iree_any_bit_set(traits, LOOM_TRAIT_HINT);
}

static void loom_low_lower_mark_value_storage_required(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  const loom_value_ordinal_t source_ordinal =
      loom_low_lowering_frame_value_ordinal(&context->lowering,
                                            source_value_id);
  context->lowering.source_plan.value_storage_flags[source_ordinal] |=
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
  return iree_any_bit_set(
      context->lowering.source_plan.value_storage_flags[source_ordinal],
      LOOM_LOW_LOWER_VALUE_STORAGE_REQUIRED);
}

bool loom_low_lower_source_plan_result_storage_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  return loom_low_lower_value_storage_required(context, source_value_id) ||
         loom_module_value_has_type_uses(context->module, source_value_id);
}

static void loom_low_lower_mark_value_slice_storage_required(
    loom_low_lower_context_t* context, loom_value_slice_t values) {
  for (uint16_t i = 0; i < values.count; ++i) {
    loom_low_lower_mark_value_storage_required(context, values.values[i]);
  }
}

bool loom_low_lower_source_plan_cfg_cond_br_exact_bool(
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
    case LOOM_OP_FUNC_CALL:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_func_call_operands(source_op));
      return;
    case LOOM_OP_FUNC_RETURN:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_func_return_operands(source_op));
      return;
    case LOOM_OP_CFG_BR:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_cfg_br_args(source_op));
      return;
    case LOOM_OP_CFG_COND_BR: {
      if (loom_low_lower_source_plan_cfg_cond_br_exact_bool(context, source_op,
                                                            NULL)) {
        return;
      }
      loom_low_lower_mark_value_storage_required(
          context, loom_cfg_cond_br_condition(source_op));
      return;
    }
    case LOOM_OP_SCF_FOR:
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_for_lower_bound(source_op));
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_for_upper_bound(source_op));
      loom_low_lower_mark_value_storage_required(context,
                                                 loom_scf_for_step(source_op));
      loom_low_lower_mark_value_slice_storage_required(
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
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_scf_while_iter_args(source_op));
      return;
    case LOOM_OP_SCF_CONDITION:
      loom_low_lower_mark_value_storage_required(
          context, loom_scf_condition_condition(source_op));
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_scf_condition_forwarded(source_op));
      return;
    case LOOM_OP_SCF_YIELD:
      loom_low_lower_mark_value_slice_storage_required(
          context, loom_scf_yield_values(source_op));
      return;
    case LOOM_OP_KERNEL_RETURN:
    default:
      return;
  }
}

static bool loom_low_lower_source_op_requires_emission(
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
    if (loom_low_lower_source_plan_result_storage_required(context,
                                                           results[i])) {
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

static void loom_low_lower_analyze_storage_demands(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_low_lower_mark_region_structural_storage_demands(context, source_body);
  for (iree_host_size_t i = context->lowering.source_plan.selected_plan_count;
       i > 0; --i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &context->lowering.source_plan.selected_plans[i - 1];
    if (!iree_any_bit_set(selected_plan->flags,
                          LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED) &&
        loom_low_lower_selected_plan_storage_required(context, selected_plan)) {
      loom_low_lower_mark_selected_plan_storage_demands(context, selected_plan);
    }
  }
  for (iree_host_size_t i = 0;
       i < context->lowering.source_plan.selected_plan_count; ++i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &context->lowering.source_plan.selected_plans[i];
    if (loom_low_lower_can_elide_source_storage(context,
                                                selected_plan->source_op)) {
      selected_plan->flags |= LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED;
    }
  }
}

static void loom_low_lower_count_region_plan_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    iree_host_size_t* inout_plan_capacity) {
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_lower_supported_structured_source_op(context, op)) {
        loom_region_t* const* regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          if (regions[i] != NULL) {
            loom_low_lower_count_region_plan_ops(context, regions[i],
                                                 inout_plan_capacity);
          }
        }
        continue;
      }
      if (loom_low_lower_op_uses_policy(context->module, op)) {
        ++(*inout_plan_capacity);
      }
    }
  }
}

static iree_status_t loom_low_lower_prepare_plan(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  iree_host_size_t plan_capacity = 0;
  loom_low_lower_count_region_plan_ops(context, source_body, &plan_capacity);
  context->lowering.source_plan.selected_plan_capacity = plan_capacity;
  context->lowering.source_plan.selected_plan_count = 0;
  context->lowering.source_plan.selected_plan_emit_index = 0;
  if (plan_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, plan_capacity,
      sizeof(*context->lowering.source_plan.selected_plans),
      (void**)&context->lowering.source_plan.selected_plans));
  if (context->options->table_arena != NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->options->table_arena, plan_capacity,
        sizeof(*context->lowering.memory_access_records),
        (void**)&context->lowering.memory_access_records));
    context->lowering.memory_access_record_capacity = plan_capacity;
  }
  return iree_ok_status();
}

static void loom_low_lower_record_selected_plan(
    loom_low_lower_context_t* context,
    loom_low_lower_selected_plan_t selected_plan) {
  IREE_ASSERT_LT(context->lowering.source_plan.selected_plan_count,
                 context->lowering.source_plan.selected_plan_capacity);
  context->lowering.source_plan
      .selected_plans[context->lowering.source_plan.selected_plan_count++] =
      selected_plan;
}

static void loom_low_lower_record_elided_hint_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
                   .flags = LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = UINT16_MAX,
                   .rule_set = NULL,
                   .rule = NULL,
                   .resolved_emits = NULL,
                   .plan = loom_low_lower_plan_empty(),
               });
}

static iree_status_t loom_low_lower_try_select_op_callback(
    loom_low_lower_context_t* context,
    loom_low_lower_select_op_callback_t callback, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  if (callback.fn == NULL) {
    return iree_ok_status();
  }

  loom_low_lower_plan_t plan = loom_low_lower_plan_empty();
  IREE_RETURN_IF_ERROR(
      callback.fn(callback.user_data, context, source_op, &plan));
  if (loom_low_lower_plan_is_empty(plan)) {
    return iree_ok_status();
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = UINT16_MAX,
                   .rule_set = NULL,
                   .rule = NULL,
                   .plan = plan,
               });
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_record_selected_rule_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t rule_set_index, const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_selection_t rule_selection,
    const loom_low_lower_rule_source_memory_state_t* source_memory_state) {
  const loom_low_lower_resolved_emit_t* resolved_emits = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_resolve_emit_program(
      context, rule_set_index, rule_set, rule_selection.rule, &resolved_emits));
  const loom_low_source_memory_access_plan_t* source_memory_access = NULL;
  if (rule_selection.uses_source_memory_access) {
    IREE_ASSERT(source_memory_state != NULL);
    IREE_ASSERT_EQ(source_memory_state->source_op, source_op);
    IREE_ASSERT(source_memory_state->plan_available);
    loom_low_source_memory_access_plan_t* retained_source_memory_access = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_source_memory_access),
        (void**)&retained_source_memory_access));
    *retained_source_memory_access = *source_memory_state->access_plan;
    source_memory_access = retained_source_memory_access;
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_RULE,
                   .rule_set_index = rule_set_index,
                   .rule_index = rule_selection.rule_index,
                   .rule_set = rule_set,
                   .rule = rule_selection.rule,
                   .resolved_emits = resolved_emits,
                   .source_memory_access = source_memory_access,
                   .plan = loom_low_lower_plan_empty(),
               });
  return iree_ok_status();
}

static bool loom_low_lower_rule_selection_is_better_failure(
    const loom_low_lower_rule_set_t* failed_rule_set,
    loom_low_lower_rule_selection_t failed_rule_selection,
    loom_low_lower_rule_selection_t rule_selection) {
  return rule_selection.has_source_op_span &&
         (failed_rule_set == NULL ||
          (rule_selection.source_memory_compatible &&
           !failed_rule_selection.source_memory_compatible) ||
          (rule_selection.source_memory_compatible ==
               failed_rule_selection.source_memory_compatible &&
           rule_selection.matched_guard_count >
               failed_rule_selection.matched_guard_count));
}

static iree_status_t loom_low_lower_emit_contract_query_rejection(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_query_result_t* result) {
  if (result->rejection != NULL) {
    return loom_low_lower_emit_error_ref(
        context, source_op, result->rejection->error_ref,
        result->rejection->params, result->rejection->param_count);
  }
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_status_t loom_low_lower_record_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_descriptor_matrix_rule_t* matrix_rule,
    const loom_contract_request_t* contract_request,
    const loom_target_contract_query_result_t* query_result) {
  loom_low_lower_descriptor_matrix_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  plan_data->source = matrix_rule->source;
  if (query_result->selected_descriptor == NULL) {
    IREE_ASSERT_UNREACHABLE("descriptor-matrix legal query has no descriptor");
    IREE_BUILTIN_UNREACHABLE();
  }
  plan_data->descriptor.descriptor = query_result->selected_descriptor;
  plan_data->contract_request = *contract_request;
  plan_data->attrs = loom_named_attr_slice_empty();
  plan_data->native_contraction_facts =
      query_result->selected_native_contraction_facts;
  if (plan_data->descriptor.descriptor->immediate_count != 0) {
    if (context->policy->descriptor_matrix.attrs == NULL) {
      IREE_ASSERT_UNREACHABLE("descriptor-matrix policy has no attrs callback");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(context->policy->descriptor_matrix.attrs(
        context->policy->descriptor_matrix.user_data, context, matrix_rule,
        &plan_data->contract_request, plan_data->descriptor.descriptor,
        &plan_data->attrs));
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = query_result->rule_index,
                   .rule_set = NULL,
                   .rule = NULL,
                   .resolved_emits = NULL,
                   .plan = loom_low_lower_plan_make(source_op->kind, plan_data),
               });
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_op_from_contract_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_rule_set_t** inout_failed_rule_set,
    loom_low_lower_rule_selection_t* inout_failed_rule_selection,
    loom_low_lower_rule_source_memory_state_t* source_memory_state,
    bool* out_selected) {
  *out_selected = false;
  const loom_target_contract_index_t* index = &context->contract_index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, /*view_regions=*/NULL, source_memory_state, &match_context);
  bool view_regions_resolved = false;

  for (uint16_t i = 0; i < op_entry.case_count; ++i) {
    const uint16_t case_index = (uint16_t)(op_entry.case_start + i);
    const loom_target_contract_case_t* contract_case =
        &index->cases[case_index];
    const loom_target_contract_binding_t* binding =
        &index->bindings[contract_case->binding_index];
    if (contract_case->system ==
        LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX) {
      const loom_target_contract_descriptor_matrix_rule_t* matrix_rule =
          &binding->fragment->descriptor_matrices[contract_case->row_index];
      loom_target_contract_query_result_t query_result =
          loom_target_contract_query_result_empty();
      loom_contract_request_t contract_request = {0};
      loom_target_contract_query_environment_t environment = {0};
      IREE_RETURN_IF_ERROR(loom_low_lower_source_query_environment_initialize(
          context, context->descriptor_set, &environment));
      IREE_RETURN_IF_ERROR(loom_low_lower_query_descriptor_matrix_contract(
          &environment, &context->policy->descriptor_matrix, matrix_rule,
          source_op, &contract_request, &query_result));
      if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
        query_result.rule_index = contract_case->row_index;
        IREE_RETURN_IF_ERROR(loom_low_lower_record_descriptor_matrix_plan(
            context, source_op, matrix_rule, &contract_request, &query_result));
        *out_selected = true;
        return iree_ok_status();
      }
      if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED ||
          query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
        IREE_RETURN_IF_ERROR(loom_low_lower_emit_contract_query_rejection(
            context, source_op, &query_result));
        *out_selected = true;
        return iree_ok_status();
      }
      continue;
    }
    uint16_t rule_index = UINT16_MAX;
    if (!loom_low_lower_contract_case_lower_rule_index(index, contract_case,
                                                       &rule_index)) {
      continue;
    }
    const loom_low_lower_rule_set_t* rule_set =
        context->policy->rule_sets.values[binding->rule_set_index];
    match_context.policy_rule_set_ordinal =
        (uint16_t)(binding->rule_set_index + 1u);
    if (rule_set->source_memory_count != 0 && !view_regions_resolved) {
      IREE_RETURN_IF_ERROR(loom_low_lower_context_view_regions(
          context, &match_context.view_regions));
      view_regions_resolved = true;
    }
    IREE_ASSERT(rule_set->source_memory_count == 0 ||
                source_memory_state->access_plan != NULL);
    loom_low_lower_rule_selection_t rule_selection = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_lower_rule_set_select_rule_range_with_match_context(
            &match_context, rule_set, source_op, rule_index, 1,
            &rule_selection));
    if (rule_selection.rule != NULL) {
      IREE_RETURN_IF_ERROR(loom_low_lower_record_selected_rule_plan(
          context, source_op, binding->rule_set_index, rule_set, rule_selection,
          source_memory_state));
      *out_selected = true;
      return iree_ok_status();
    }
    if (loom_low_lower_rule_selection_is_better_failure(
            *inout_failed_rule_set, *inout_failed_rule_selection,
            rule_selection)) {
      *inout_failed_rule_set = rule_set;
      *inout_failed_rule_selection = rule_selection;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  if (source_op->region_count != 0) {
    if (loom_low_lower_supported_structured_source_op(context, source_op)) {
      return iree_ok_status();
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(source_op->region_count),
    };
    return loom_low_lower_emit_target_context_error(context, source_op,
                                                    LOOM_ERR_TARGET_030, params,
                                                    IREE_ARRAYSIZE(params));
  }
  if (loom_low_lower_op_is_structural(context->module, source_op)) {
    return iree_ok_status();
  }
  if (loom_low_lower_source_plan_op_is_metadata(source_op->kind)) {
    return iree_ok_status();
  }

  bool selected_callback = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_try_select_op_callback(
      context, context->policy->preselect_op, source_op, &selected_callback));
  if (selected_callback) {
    return iree_ok_status();
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_selection_t failed_rule_selection = {0};
  loom_low_source_memory_access_plan_t source_memory_access;
  loom_low_lower_rule_source_memory_state_t source_memory_state;
  loom_low_lower_rule_source_memory_state_initialize(
      source_op, &source_memory_access, &source_memory_state);
  bool selected_rule = false;
  if (context->contract_index.case_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_op_from_contract_index(
        context, source_op, &failed_rule_set, &failed_rule_selection,
        &source_memory_state, &selected_rule));
    if (selected_rule) {
      return iree_ok_status();
    }
    if (failed_rule_set != NULL) {
      return loom_low_lower_rule_set_emit_selection_failure(
          context, failed_rule_set, source_op, failed_rule_selection,
          &source_memory_state);
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_lower_try_select_op_callback(
      context, context->policy->select_op, source_op, &selected_callback));
  if (selected_callback) {
    return iree_ok_status();
  }

  if (failed_rule_set != NULL) {
    return loom_low_lower_rule_set_emit_selection_failure(
        context, failed_rule_set, source_op, failed_rule_selection,
        &source_memory_state);
  }
  if (loom_low_lower_op_is_discardable_hint(context->module, source_op)) {
    loom_low_lower_record_elided_hint_plan(context, source_op);
    return iree_ok_status();
  }
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static void loom_low_lower_planning_scope_begin(
    loom_low_lower_context_t* context) {
  context->planning_arena_active = true;
}

static void loom_low_lower_planning_scope_end(
    loom_low_lower_context_t* context) {
  context->planning_arena_active = false;
  iree_arena_reset(&context->planning_arena);
}

static iree_status_t loom_low_lower_plan_region(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    const loom_op_t* block_arg_context_op, bool skip_entry_block_args) {
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_region, block_index);
    if (!(skip_entry_block_args && block_index == 0)) {
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        loom_type_t low_type = loom_type_none();
        IREE_RETURN_IF_ERROR(loom_low_lower_source_plan_check_mapped_value(
            context, block_arg_context_op, block->arg_ids[i], &low_type));
        if (loom_low_lower_context_should_stop(context)) {
          return iree_ok_status();
        }
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_low_lower_planning_scope_begin(context);
      iree_status_t status = loom_low_lower_plan_op(context, op);
      loom_low_lower_planning_scope_end(context);
      IREE_RETURN_IF_ERROR(status);
      if (loom_low_lower_context_should_stop(context)) {
        return iree_ok_status();
      }
      if (!loom_low_lower_supported_structured_source_op(context, op)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_lower_plan_region(context, regions[i], op,
                                       /*skip_entry_block_args=*/false));
        if (loom_low_lower_context_should_stop(context)) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_source_plan_build(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_low_lower_source_plan_t* source_plan = &context->lowering.source_plan;
  *source_plan = (loom_low_lower_source_plan_t){0};
  const loom_value_ordinal_t value_count =
      context->lowering.value_domain.value_count;
  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
        context, value_count, sizeof(*source_plan->value_storage_flags),
        (void**)&source_plan->value_storage_flags));
    memset(source_plan->value_storage_flags, 0,
           value_count * sizeof(*source_plan->value_storage_flags));
  }

  iree_arena_initialize(context->module->arena.block_pool,
                        &context->planning_arena);
  iree_status_t status = loom_low_lower_prepare_plan(context, source_body);
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_plan_region(context, source_body,
                                        context->source_function.op,
                                        /*skip_entry_block_args=*/true);
  }
  if (iree_status_is_ok(status)) {
    loom_low_lower_analyze_storage_demands(context, source_body);
  }
  iree_arena_deinitialize(&context->planning_arena);
  return status;
}
