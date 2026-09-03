// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_call.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/lower/call_predicates.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/target/facts.h"
#include "loom/target/function_contract.h"

static iree_string_view_t loom_low_source_call_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<invalid>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_low_source_call_emit_contract_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t reason) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(reason),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_072, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_source_call_emit_count_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t value_kind,
    uint32_t actual_count, uint32_t expected_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(value_kind),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_073, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_source_call_emit_type_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t value_kind,
    loom_type_t actual_type, loom_type_t expected_type) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(value_kind),
      loom_param_type(actual_type),
      loom_param_type(expected_type),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_074, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_source_call_emit_representation_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t callee_contract,
    iree_string_view_t caller_target_contract) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(callee_contract),
      loom_param_string(caller_target_contract),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_075, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_source_call_emit_target_error(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, iree_string_view_t callee_target,
    iree_string_view_t caller_target) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(callee_target),
      loom_param_string(caller_target),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_076, params, IREE_ARRAYSIZE(params));
}

static iree_string_view_t loom_low_source_call_function_descriptor_set(
    const loom_module_t* module, const loom_op_t* function_op) {
  const loom_string_id_t descriptor_set =
      loom_low_func_def_descriptor_set(function_op);
  if (descriptor_set >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[descriptor_set];
}

static iree_status_t loom_low_source_call_resolve_callee_target(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_symbol_ref_t callee_ref, const loom_op_t* callee_op,
    iree_string_view_t callee_name,
    const loom_target_facts_t** out_callee_target_facts,
    const loom_low_descriptor_set_t** out_callee_descriptor_set) {
  *out_callee_target_facts = NULL;
  *out_callee_descriptor_set = NULL;

  const iree_string_view_t callee_contract =
      loom_low_source_call_function_descriptor_set(context->module, callee_op);
  *out_callee_descriptor_set = loom_low_descriptor_registry_lookup(
      context->options->descriptor_registry, callee_contract);
  if (*out_callee_descriptor_set == NULL) {
    return loom_low_source_call_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("the helper representation contract is not linked"));
  }

  if (!loom_symbol_ref_is_valid(loom_low_func_def_target(callee_op))) {
    return iree_ok_status();
  }

  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(
      &symbol_facts, loom_low_lower_context_emission_arena(context));
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, context->module, callee_ref, &base_facts));
  const loom_func_symbol_facts_t* function_facts =
      loom_func_symbol_facts_cast(base_facts);
  if (function_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified low helper '@%.*s' has no function symbol facts",
        (int)callee_name.size, callee_name.data);
  }

  bool target_valid = true;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve_facts(
      context->module, &symbol_facts, function_facts, context->options->emitter,
      loom_low_lower_context_emission_arena(context), &target_valid,
      out_callee_target_facts));
  if (!target_valid) {
    *out_callee_descriptor_set = NULL;
    if (!loom_low_lower_context_should_stop(context)) {
      ++context->result->error_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_call_map_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_type_t* out_low_type) {
  const uint32_t previous_error_count = context->result->error_count;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, out_low_type));
  if (loom_type_kind(*out_low_type) != LOOM_TYPE_NONE) {
    return iree_ok_status();
  }
  if (context->result->error_count != previous_error_count) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("result")),
      loom_param_u64(source_result),
  };
  return loom_low_lower_emit_target_context_error(
      context, source_op, LOOM_ERR_TARGET_027, params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_low_lower_source_invoke(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_symbol_ref_t callee_ref = loom_low_invoke_callee(source_op);
  const iree_string_view_t callee_name =
      loom_low_source_call_symbol_name(module, callee_ref);
  if (!loom_symbol_ref_is_valid(callee_ref) || callee_ref.module_id != 0 ||
      callee_ref.symbol_id >= module->symbols.count) {
    return loom_low_source_call_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("the callee is not a module-local symbol"));
  }

  const loom_symbol_t* callee_symbol =
      &module->symbols.entries[callee_ref.symbol_id];
  const loom_op_t* callee_op = callee_symbol->defining_op;
  if (callee_op == NULL || !loom_low_func_def_isa(callee_op)) {
    return loom_low_source_call_emit_contract_error(
        context, source_op, callee_name,
        IREE_SV("the callee is not a low.func.def with a body"));
  }

  const loom_target_facts_t* callee_target_facts = NULL;
  const loom_low_descriptor_set_t* callee_descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_low_source_call_resolve_callee_target(
      context, source_op, callee_ref, callee_op, callee_name,
      &callee_target_facts, &callee_descriptor_set));
  if (callee_descriptor_set == NULL) {
    return iree_ok_status();
  }

  const loom_target_facts_t* caller_target_facts =
      loom_low_lower_context_target_facts(context);
  if (callee_target_facts != NULL &&
      !loom_target_facts_satisfy_specialization_requirement(
          caller_target_facts, callee_target_facts)) {
    return loom_low_source_call_emit_target_error(
        context, source_op, callee_name,
        loom_target_facts_identity_name(callee_target_facts),
        loom_target_facts_identity_name(caller_target_facts));
  }

  const loom_low_descriptor_set_t* caller_descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (callee_descriptor_set != caller_descriptor_set) {
    const iree_string_view_t callee_contract =
        loom_low_source_call_function_descriptor_set(module, callee_op);
    const iree_string_view_t caller_target_contract =
        loom_low_lower_context_bundle(context)->config->contract_set_key;
    return loom_low_source_call_emit_representation_error(
        context, source_op, callee_name, callee_contract,
        caller_target_contract);
  }

  const loom_func_like_t callee = loom_func_like_const_cast(module, callee_op);
  uint16_t callee_argument_count = 0;
  const loom_value_id_t* callee_arguments =
      loom_func_like_arg_ids(callee, &callee_argument_count);
  const loom_value_slice_t source_operands =
      loom_low_invoke_operands(source_op);
  if (source_operands.count != callee_argument_count) {
    return loom_low_source_call_emit_count_error(
        context, source_op, callee_name, IREE_SV("operand"),
        source_operands.count, callee_argument_count);
  }

  const loom_value_slice_t callee_results =
      loom_low_func_def_results(callee_op);
  const loom_value_slice_t source_results = loom_low_invoke_results(source_op);
  if (source_results.count != callee_results.count) {
    return loom_low_source_call_emit_count_error(
        context, source_op, callee_name, IREE_SV("result"),
        source_results.count, callee_results.count);
  }

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, callee_argument_count, sizeof(*low_operands),
      (void**)&low_operands));
  for (uint16_t i = 0; i < callee_argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, source_operands.values[i], &low_operands[i]));
    const loom_type_t actual_type =
        loom_module_value_type(module, low_operands[i]);
    const loom_type_t expected_type =
        loom_module_value_type(module, callee_arguments[i]);
    if (!loom_type_equal(actual_type, expected_type)) {
      return loom_low_source_call_emit_type_error(
          context, source_op, callee_name, IREE_SV("operand"), actual_type,
          expected_type);
    }
  }

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, source_results.count, sizeof(*result_types),
      (void**)&result_types));
  for (uint16_t i = 0; i < source_results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_source_call_map_result_type(
        context, source_op, source_results.values[i], &result_types[i]));
    if (loom_type_kind(result_types[i]) == LOOM_TYPE_NONE) {
      return iree_ok_status();
    }
    const loom_type_t expected_type =
        loom_module_value_type(module, callee_results.values[i]);
    if (!loom_type_equal(result_types[i], expected_type)) {
      return loom_low_source_call_emit_type_error(
          context, source_op, callee_name, IREE_SV("result"), result_types[i],
          expected_type);
    }
  }

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      module, module, loom_low_lower_context_emission_arena(context), NULL,
      &remap));
  for (uint16_t i = 0; i < callee_argument_count; ++i) {
    const loom_type_t expected_type =
        loom_module_value_type(module, callee_arguments[i]);
    IREE_RETURN_IF_ERROR(loom_low_lower_materialize_structural_operand(
        context, source_op, i, source_operands.values[i], expected_type,
        &low_operands[i]));
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, callee_arguments[i], low_operands[i]));
  }

  IREE_RETURN_IF_ERROR(loom_low_materialize_call_argument_contract(
      context, source_op, callee_name, callee, callee_arguments,
      callee_argument_count, source_operands, &remap));
  for (uint16_t i = 0; i < callee_argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        &remap, callee_arguments[i], &low_operands[i]));
  }

  loom_low_func_call_build_flags_t build_flags = 0;
  const uint8_t purity = loom_low_invoke_purity(source_op);
  const uint8_t inline_policy = loom_low_invoke_inline_policy(source_op);
  if (purity != 0) {
    build_flags |= LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_PURITY;
  }
  if (inline_policy != 0) {
    build_flags |= LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY;
  }
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_call_build(
      loom_low_lower_context_builder(context), build_flags, purity,
      inline_policy, callee_ref, low_operands, callee_argument_count,
      result_types, source_results.count, loom_op_tied_results(source_op),
      source_op->tied_result_count, source_op->location, &call_op));

  const loom_value_id_t* low_results = loom_op_const_results(call_op);
  for (uint16_t i = 0; i < source_results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
        context, source_results.values[i], low_results[i]));
  }
  return iree_ok_status();
}
