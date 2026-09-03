// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/low_verify.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"
#include "loom/target/projection.h"

typedef struct loom_aie2p_low_verify_state_t {
  // Module containing the current array program.
  const loom_module_t* module;
  // Target resolved for the current array program.
  const loom_low_resolved_target_t* target;
  // Borrowed array-program function name used in diagnostics.
  iree_string_view_t function_name;
} loom_aie2p_low_verify_state_t;

static iree_status_t loom_aie2p_low_verify_begin_function(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void** out_provider_state) {
  (void)provider;
  *out_provider_state = NULL;
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  if (target->descriptor_set == NULL ||
      target->descriptor_set->stable_id != AIE2P_ARRAY_DESCRIPTOR_SET_ID) {
    return iree_ok_status();
  }

  loom_aie2p_low_verify_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      loom_low_verify_context_arena(context), sizeof(*state), (void**)&state));
  *state = (loom_aie2p_low_verify_state_t){
      .module = loom_low_verify_context_module(context),
      .target = target,
      .function_name = loom_low_diagnostic_function_name(
          loom_low_verify_context_module(context),
          loom_low_verify_context_function_op(context)),
  };
  *out_provider_state = state;
  return iree_ok_status();
}

static const loom_named_attr_t* loom_aie2p_low_find_packet_attr(
    const loom_aie2p_low_verify_state_t* state, const loom_op_t* op,
    iree_string_view_t name, uint16_t* out_attrs_attr_index) {
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  if (!loom_low_packet_try_op_attrs(op, &attrs, out_attrs_attr_index)) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id < state->module->strings.count &&
        iree_string_view_equal(state->module->strings.entries[attr->name_id],
                               name)) {
      return attr;
    }
  }
  return NULL;
}

static iree_string_view_t loom_aie2p_low_function_contract_name(
    const loom_module_t* module, loom_func_like_t function) {
  if (!loom_func_like_isa(function)) {
    return IREE_SV("<not-a-function>");
  }
  const loom_string_id_t contract_id = loom_func_like_repr_contract(function);
  if (contract_id < module->strings.count) {
    return module->strings.entries[contract_id];
  }

  // A resident worker may name a source function in the same mixed-level
  // module. Before source-to-low conversion the function has no representation
  // contract, but its target record already promises the contract that lowering
  // must produce. Resolve that promise so pre-lowering verification and the
  // concrete post-lowering check enforce the same worker-entry requirement.
  const loom_symbol_ref_t target_ref = loom_func_like_target(function);
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unbound>");
  }
  const loom_symbol_t* target_symbol =
      &module->symbols.entries[target_ref.symbol_id];
  const loom_target_like_t target =
      loom_target_like_cast(module, target_symbol->defining_op);
  const loom_target_like_descriptor_t* descriptor =
      loom_target_like_descriptor(target);
  if (descriptor == NULL) {
    return IREE_SV("<unresolved-target>");
  }
  const uint32_t selector =
      (uint32_t)loom_attr_as_enum(loom_target_like_selector(target));
  const loom_target_bundle_t* bundle =
      loom_target_bundle_table_lookup(descriptor->bundle_table, selector);
  return bundle ? bundle->config->contract_set_key
                : IREE_SV("<unresolved-target>");
}

static iree_status_t loom_aie2p_low_verify_worker(
    loom_low_verify_context_t* context,
    const loom_aie2p_low_verify_state_t* state,
    const loom_low_descriptor_packet_t* packet) {
  if (packet->descriptor_ordinal != AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER) {
    return iree_ok_status();
  }

  uint16_t attrs_attr_index = UINT16_MAX;
  const loom_named_attr_t* entry_attr = loom_aie2p_low_find_packet_attr(
      state, packet->op, IREE_SV("entry"), &attrs_attr_index);
  if (entry_attr == NULL || entry_attr->value.kind != LOOM_ATTR_SYMBOL) {
    return iree_ok_status();
  }

  const loom_symbol_ref_t entry_ref = loom_attr_as_symbol(entry_attr->value);
  if (!loom_symbol_ref_is_valid(entry_ref) || entry_ref.module_id != 0 ||
      entry_ref.symbol_id >= state->module->symbols.count) {
    return iree_ok_status();
  }
  const loom_symbol_t* entry_symbol =
      &state->module->symbols.entries[entry_ref.symbol_id];
  loom_func_like_t entry_function =
      loom_func_like_cast(state->module, entry_symbol->defining_op);
  const iree_string_view_t actual_contract =
      loom_aie2p_low_function_contract_name(state->module, entry_function);
  const iree_string_view_t expected_contract = IREE_SV("amd.xdna.aie2p.core");
  if (iree_string_view_equal(actual_contract, expected_contract)) {
    return iree_ok_status();
  }

  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_with_field_ref(
          loom_param_string(loom_low_descriptor_packet_diagnostic_key(
              state->target->descriptor_set, packet)),
          loom_diagnostic_field_ref(
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
              loom_low_descriptor_packet_attribute_index(packet))),
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("entry")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
      loom_param_string(
          loom_low_diagnostic_symbol_name(state->module, entry_ref)),
      loom_param_string(actual_contract),
      loom_param_string(expected_contract),
  };
  return loom_low_verify_context_emit(context, packet->op, LOOM_ERR_TARGET_079,
                                      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_aie2p_low_verify_op(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_descriptor_packet_t* packet) {
  (void)provider;
  const loom_aie2p_low_verify_state_t* state =
      (const loom_aie2p_low_verify_state_t*)provider_state;
  if (state == NULL || packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE ||
      loom_low_verify_context_should_stop(context)) {
    return iree_ok_status();
  }
  return loom_aie2p_low_verify_worker(context, state, packet);
}

const loom_low_verify_provider_t loom_aie2p_low_verify_provider = {
    .name = IREE_SVL("amd-xdna-aie2p"),
    .begin_function = loom_aie2p_low_verify_begin_function,
    .verify_op = loom_aie2p_low_verify_op,
};
