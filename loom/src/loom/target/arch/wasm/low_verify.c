// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/low_verify.h"

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/registers.h"

typedef struct loom_wasm_low_verify_state_t {
  // Module being verified.
  const loom_module_t* module;
  // Resolved low target for this function.
  const loom_low_resolved_target_t* target;
  // Borrowed function symbol name for diagnostics.
  iree_string_view_t function_name;
} loom_wasm_low_verify_state_t;

static bool loom_wasm_low_type_is_i32(loom_type_t type) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             WASM_CORE_SIMD128_DESCRIPTOR_SET_ID &&
         loom_low_register_type_class_id(type) ==
             WASM_CORE_SIMD128_REG_CLASS_ID_I32 &&
         loom_low_register_type_unit_count(type) == 1;
}

#define LOOM_WASM_LOW_VERIFY_TARGET_CONTEXT_PARAM_COUNT 5

static void loom_wasm_low_make_target_context_params(
    loom_wasm_low_verify_state_t* state, const loom_op_t* op,
    loom_diagnostic_param_t* params) {
  params[0] = loom_param_string(loom_low_diagnostic_target_key(state->target));
  params[1] = loom_param_string(loom_low_diagnostic_export_name(state->target));
  params[2] = loom_param_string(loom_low_diagnostic_config_key(state->target));
  params[3] = loom_param_string(state->function_name);
  params[4] = loom_param_string(loom_op_name(state->module, op));
}

static iree_status_t loom_wasm_low_emit_target_context_error(
    loom_low_verify_context_t* context, loom_wasm_low_verify_state_t* state,
    const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* extra_params,
    iree_host_size_t extra_param_count) {
  IREE_ASSERT_LE(extra_param_count, 4);
  loom_diagnostic_param_t
      params[LOOM_WASM_LOW_VERIFY_TARGET_CONTEXT_PARAM_COUNT + 4];
  loom_wasm_low_make_target_context_params(state, op, params);
  for (iree_host_size_t i = 0; i < extra_param_count; ++i) {
    params[LOOM_WASM_LOW_VERIFY_TARGET_CONTEXT_PARAM_COUNT + i] =
        extra_params[i];
  }
  return loom_low_verify_context_emit(
      context, op, error, params,
      LOOM_WASM_LOW_VERIFY_TARGET_CONTEXT_PARAM_COUNT + extra_param_count);
}

static iree_status_t loom_wasm_low_emit_branch_condition_type_mismatch(
    loom_low_verify_context_t* context, loom_wasm_low_verify_state_t* state,
    const loom_op_t* op, loom_type_t actual_type) {
  const loom_diagnostic_param_t extra_params[] = {
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("reg<wasm.i32>")),
  };
  return loom_wasm_low_emit_target_context_error(
      context, state, op, LOOM_ERR_TARGET_035, extra_params,
      IREE_ARRAYSIZE(extra_params));
}

static iree_status_t loom_wasm_low_emit_type_constraint_mismatch(
    loom_low_verify_context_t* context, loom_wasm_low_verify_state_t* state,
    const loom_op_t* op, loom_type_t actual_type) {
  const loom_diagnostic_param_t extra_params[] = {
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("reg<wasm.i32>")),
  };
  return loom_wasm_low_emit_target_context_error(
      context, state, op, LOOM_ERR_TARGET_031, extra_params,
      IREE_ARRAYSIZE(extra_params));
}

static iree_status_t loom_wasm_low_verify_scf_if(
    loom_low_verify_context_t* context, loom_wasm_low_verify_state_t* state,
    const loom_op_t* op) {
  const loom_type_t condition_type =
      loom_module_value_type(state->module, loom_low_scf_if_condition(op));
  if (loom_wasm_low_type_is_i32(condition_type)) {
    return iree_ok_status();
  }
  return loom_wasm_low_emit_branch_condition_type_mismatch(context, state, op,
                                                           condition_type);
}

static iree_status_t loom_wasm_low_verify_scf_for(
    loom_low_verify_context_t* context, loom_wasm_low_verify_state_t* state,
    const loom_op_t* op) {
  const loom_type_t lower_bound_type =
      loom_module_value_type(state->module, loom_low_scf_for_lower_bound(op));
  if (!loom_wasm_low_type_is_i32(lower_bound_type)) {
    return loom_wasm_low_emit_type_constraint_mismatch(context, state, op,
                                                       lower_bound_type);
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_low_begin_function(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void** out_provider_state) {
  (void)provider;
  *out_provider_state = NULL;
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  if (target->descriptor_set == NULL ||
      target->descriptor_set->stable_id !=
          WASM_CORE_SIMD128_DESCRIPTOR_SET_ID) {
    return iree_ok_status();
  }

  loom_wasm_low_verify_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      loom_low_verify_context_arena(context), sizeof(*state), (void**)&state));
  *state = (loom_wasm_low_verify_state_t){
      .module = loom_low_verify_context_module(context),
      .target = target,
      .function_name = loom_low_diagnostic_function_name(
          loom_low_verify_context_module(context),
          loom_low_verify_context_function_op(context)),
  };
  *out_provider_state = state;
  return iree_ok_status();
}

static iree_status_t loom_wasm_low_verify_op(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  (void)provider;
  loom_wasm_low_verify_state_t* state =
      (loom_wasm_low_verify_state_t*)provider_state;
  if (state == NULL || loom_low_verify_context_should_stop(context)) {
    return iree_ok_status();
  }
  if (loom_low_scf_if_isa(packet->op)) {
    return loom_wasm_low_verify_scf_if(context, state, packet->op);
  }
  if (loom_low_scf_for_isa(packet->op)) {
    return loom_wasm_low_verify_scf_for(context, state, packet->op);
  }
  return iree_ok_status();
}

const loom_low_verify_provider_t loom_wasm_low_verify_provider = {
    .name = IREE_SVL("wasm"),
    .begin_function = loom_wasm_low_begin_function,
    .verify_op = loom_wasm_low_verify_op,
};
