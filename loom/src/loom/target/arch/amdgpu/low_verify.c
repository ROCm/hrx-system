// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/low_verify.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/low_aliases.h"
#include "loom/target/arch/amdgpu/ops/ops.h"

typedef struct loom_amdgpu_low_verify_state_t {
  // Target resolved for the current function.
  const loom_low_resolved_target_t* target;
  // Borrowed function name used in diagnostics.
  iree_string_view_t function_name;
} loom_amdgpu_low_verify_state_t;

static iree_status_t loom_amdgpu_low_verify_begin_function(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void** out_provider_state) {
  (void)provider;
  *out_provider_state = NULL;
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  if (!loom_amdgpu_target_isa(target->target_op)) {
    return iree_ok_status();
  }

  loom_amdgpu_low_verify_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      loom_low_verify_context_arena(context), sizeof(*state), (void**)&state));
  *state = (loom_amdgpu_low_verify_state_t){
      .target = target,
      .function_name = loom_low_diagnostic_function_name(
          loom_low_verify_context_module(context),
          loom_low_verify_context_function_op(context)),
  };
  *out_provider_state = state;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_emit_blocked_alias(
    loom_low_verify_context_t* context,
    const loom_amdgpu_low_verify_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_amdgpu_low_blocked_alias_t* alias) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_with_field_ref(
          loom_param_string(packet->key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    packet->key_attr_index)),
      loom_param_string(state->target->descriptor_set_key),
      loom_param_string(alias->alias_mnemonic),
      loom_param_string(alias->alias_semantics),
      loom_param_string(alias->replacement_descriptor_name),
      loom_param_string(alias->replacement_mnemonic),
      loom_param_string(alias->decision),
      loom_param_string(alias->reason),
  };
  return loom_low_verify_context_emit(context, packet->op, LOOM_ERR_AMDGPU_031,
                                      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_low_verify_op(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  (void)provider;
  loom_amdgpu_low_verify_state_t* state =
      (loom_amdgpu_low_verify_state_t*)provider_state;
  if (state == NULL || packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE ||
      packet->descriptor != NULL ||
      loom_low_verify_context_should_stop(context)) {
    return iree_ok_status();
  }
  const loom_amdgpu_low_blocked_alias_t* alias =
      loom_amdgpu_low_blocked_alias_lookup(packet->key);
  if (alias == NULL) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_emit_blocked_alias(context, state, packet, alias);
}

const loom_low_verify_provider_t loom_amdgpu_low_verify_provider = {
    .name = IREE_SVL("amdgpu"),
    .begin_function = loom_amdgpu_low_verify_begin_function,
    .verify_op = loom_amdgpu_low_verify_op,
};
