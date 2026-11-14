// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/call_boundary.h"

#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/callable_effects.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/error_catalog.h"

bool loom_spirv_source_callable_requires_inline(const loom_module_t* module,
                                                loom_func_like_t function) {
  if (!loom_func_like_body(function) ||
      loom_callable_effects_may_create_resources(function)) {
    return true;
  }
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  const loom_value_id_t* results = loom_op_const_results(function.op);
  const iree_host_size_t value_count =
      (iree_host_size_t)argument_count + function.op->result_count;
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    const loom_value_id_t value =
        i < argument_count ? arguments[i] : results[i - argument_count];
    const loom_type_t type = loom_module_value_type(module, value);
    if (!loom_type_is_scalar(type) && !loom_type_is_vector(type)) {
      return true;
    }
  }
  return false;
}

// All interpretation comes from the existing value-fact owner. In particular,
// an unavailable exact matrix shape still denotes an opaque matrix payload.
static loom_value_id_t loom_spirv_call_boundary_fragment(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_op_t* op, loom_func_like_t function, bool allow_pending_inline,
    loom_symbol_ref_t* out_callee) {
  if (!loom_func_call_isa(op) && !loom_func_return_isa(op)) {
    return LOOM_VALUE_ID_INVALID;
  }
  loom_value_id_t fragment_value = LOOM_VALUE_ID_INVALID;
  const iree_host_size_t value_count =
      (iree_host_size_t)op->operand_count + op->result_count;
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    const loom_value_id_t value =
        i < op->operand_count
            ? loom_op_const_operands(op)[i]
            : loom_op_const_results(op)[i - op->operand_count];
    if (!loom_type_is_vector(loom_module_value_type(module, value))) {
      continue;
    }
    loom_vector_fragment_fact_t fragment;
    if (loom_vector_fragment_fact_query_value_facts(
            &facts->context, loom_value_fact_table_lookup(facts, value),
            &fragment) &&
        loom_vector_fragment_fact_has_matrix_shape(fragment)) {
      fragment_value = value;
      break;
    }
  }
  if (fragment_value == LOOM_VALUE_ID_INVALID) {
    return LOOM_VALUE_ID_INVALID;
  }
  if (loom_func_call_isa(op)) {
    const loom_call_like_t call = loom_call_like_const_cast(module, op);
    *out_callee = loom_call_like_callee(call);
    if (allow_pending_inline) {
      function = loom_func_like_const_cast(
          module, module->symbols.entries[out_callee->symbol_id].defining_op);
      if (loom_call_like_inline_policy(call) == LOOM_INLINE_POLICY_INLINE) {
        return LOOM_VALUE_ID_INVALID;
      }
    }
  } else {
    *out_callee = loom_func_like_callee(function);
  }
  // Source completion expands required boundaries with callable exits after
  // target control-flow legalization. Their bodies are still source here.
  const loom_inline_policy_t inline_policy =
      loom_func_like_inline_policy(function);
  if (allow_pending_inline &&
      (inline_policy == LOOM_INLINE_POLICY_INLINE ||
       (inline_policy != LOOM_INLINE_POLICY_NOINLINE &&
        loom_spirv_source_callable_requires_inline(module, function)))) {
    return LOOM_VALUE_ID_INVALID;
  }
  return fragment_value;
}

static void loom_spirv_call_boundary_params(const loom_module_t* module,
                                            const loom_op_t* op,
                                            loom_symbol_ref_t callee,
                                            loom_value_id_t value,
                                            loom_diagnostic_param_t params[3]) {
  params[0] = loom_param_string(loom_op_name(module, op));
  params[1] = loom_param_string(loom_string_table_get(
      &module->strings, module->symbols.entries[callee.symbol_id].name_id));
  params[2] = loom_param_type(loom_module_value_type(module, value));
}

bool loom_spirv_match_call_boundary(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    return false;
  }
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  return loom_spirv_call_boundary_fragment(
             context->module, context->fact_table, op, context->function,
             /*allow_pending_inline=*/true, &callee) != LOOM_VALUE_ID_INVALID;
}

iree_status_t loom_spirv_legalize_call_boundary(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){0};
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  const loom_value_id_t value = loom_spirv_call_boundary_fragment(
      context->module, context->fact_table, op, context->function,
      /*allow_pending_inline=*/true, &callee);
  loom_diagnostic_param_t params[3];
  loom_spirv_call_boundary_params(context->module, op, callee, value, params);
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_SPIRV_033,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(
      iree_diagnostic_emit(context->pass->diagnostic_emitter, &emission));
  return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
}

static iree_status_t loom_spirv_verify_call_boundary(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  const loom_module_t* module = loom_target_low_legality_module(context);
  loom_symbol_ref_t callee = loom_symbol_ref_null();
  const loom_value_id_t value = loom_spirv_call_boundary_fragment(
      module, loom_target_low_legality_fact_table(context), op,
      loom_target_low_legality_function(context),
      /*allow_pending_inline=*/false, &callee);
  if (value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  *out_handled = true;
  loom_diagnostic_param_t params[3];
  loom_spirv_call_boundary_params(module, op, callee, value, params);
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_SPIRV_033_REF, params, IREE_ARRAYSIZE(params));
}

const loom_target_low_legality_provider_t
    loom_spirv_call_boundary_legality_provider = {
        .name = IREE_SVL("spirv-call-boundary"),
        .scope = {.target_stable_id = SPIRV_LOGICAL_CORE_TARGET_ID},
        .builtin_dialect_bits = UINT64_C(1) << LOOM_DIALECT_FUNC,
        .try_verify_op = loom_spirv_verify_call_boundary,
};
