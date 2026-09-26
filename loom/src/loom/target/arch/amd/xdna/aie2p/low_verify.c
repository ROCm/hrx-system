// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/low_verify.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/array/abi_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/core_structure.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"
#include "loom/target/projection.h"
#include "loom/target/registers.h"

typedef enum loom_aie2p_low_verify_program_kind_e {
  LOOM_AIE2P_LOW_VERIFY_PROGRAM_ARRAY = 0,
  LOOM_AIE2P_LOW_VERIFY_PROGRAM_CORE = 1,
} loom_aie2p_low_verify_program_kind_t;

typedef struct loom_aie2p_low_verify_state_t {
  // Module containing the current AIE2P program.
  const loom_module_t* module;
  // Target resolved for the current AIE2P program.
  const loom_low_resolved_target_t* target;
  // Borrowed function name used in diagnostics.
  iree_string_view_t function_name;
  // AIE2P representation contract selected by the target.
  loom_aie2p_low_verify_program_kind_t program_kind;
  // Whether the array ABI layout passed its one public schema boundary.
  bool abi_layout_valid;
} loom_aie2p_low_verify_state_t;

static loom_diagnostic_param_t loom_aie2p_low_abi_layout_field_param(
    iree_string_view_t value) {
  return loom_param_with_field_ref(
      loom_param_string(value), loom_low_func_def_abi_layout_diagnostic_ref());
}

static iree_status_t loom_aie2p_low_verify_array_abi_layout(
    loom_low_verify_context_t* context, loom_aie2p_low_verify_state_t* state,
    const loom_op_t* function_op) {
  loom_aie2p_array_abi_layout_issue_t issue = {0};
  if (loom_aie2p_array_abi_layout_validate(
          state->module, loom_low_func_def_abi_layout(function_op), &issue)) {
    return iree_ok_status();
  }

  state->abi_layout_valid = false;
  switch (issue.kind) {
    case LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_UNEXPECTED_FIELD: {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(state->function_name),
          loom_aie2p_low_abi_layout_field_param(issue.field_name),
      };
      return loom_low_verify_context_emit(context, function_op,
                                          LOOM_ERR_XDNA_030, params,
                                          IREE_ARRAYSIZE(params));
    }
    case LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_BINDING_COUNT_KIND: {
      const loom_diagnostic_param_t params[] = {
          loom_aie2p_low_abi_layout_field_param(
              IREE_SV("abi_layout.binding_count")),
          loom_param_u32(issue.attribute.kind),
          loom_param_u32(LOOM_ATTR_I64),
      };
      return loom_low_verify_context_emit(context, function_op,
                                          LOOM_ERR_TYPE_005, params,
                                          IREE_ARRAYSIZE(params));
    }
    case LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_BINDING_COUNT_RANGE: {
      const loom_diagnostic_param_t params[] = {
          loom_aie2p_low_abi_layout_field_param(
              IREE_SV("abi_layout.binding_count")),
          loom_param_i64(issue.attribute.i64),
          loom_param_string(IREE_SV("an integer in [0, 65535]")),
      };
      return loom_low_verify_context_emit(context, function_op,
                                          LOOM_ERR_STRUCTURE_014, params,
                                          IREE_ARRAYSIZE(params));
    }
    case LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("array ABI layout issue kind");
  return iree_ok_status();
}

static iree_status_t loom_aie2p_low_verify_empty_signature(
    loom_low_verify_context_t* context, const loom_module_t* module,
    const loom_op_t* function_op, iree_string_view_t emitter_key) {
  const loom_func_like_t function =
      loom_func_like_const_cast(module, function_op);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(function, &argument_count);
  const uint16_t result_count = function_op->result_count;
  if (argument_count == 0 && result_count == 0) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, function_op)),
      loom_param_string(emitter_key),
      loom_param_string(argument_count != 0 ? IREE_SV("register argument")
                                            : IREE_SV("register result")),
      loom_param_u32(argument_count != 0 ? argument_count : result_count),
      loom_param_u32(0),
  };
  return loom_low_verify_context_emit(context, function_op, LOOM_ERR_TARGET_054,
                                      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_aie2p_low_verify_begin_function(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void** out_provider_state) {
  (void)provider;
  *out_provider_state = NULL;
  const loom_low_resolved_target_t* target =
      loom_low_verify_context_target(context);
  if (target->descriptor_set == NULL) {
    return iree_ok_status();
  }
  loom_aie2p_low_verify_program_kind_t program_kind;
  if (target->descriptor_set->stable_id == AIE2P_ARRAY_DESCRIPTOR_SET_ID) {
    program_kind = LOOM_AIE2P_LOW_VERIFY_PROGRAM_ARRAY;
  } else if (target->descriptor_set->stable_id ==
             AIE2P_CORE_DESCRIPTOR_SET_ID) {
    program_kind = LOOM_AIE2P_LOW_VERIFY_PROGRAM_CORE;
  } else {
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
      .program_kind = program_kind,
      .abi_layout_valid = true,
  };
  *out_provider_state = state;
  if (program_kind == LOOM_AIE2P_LOW_VERIFY_PROGRAM_CORE) {
    return iree_ok_status();
  }
  const loom_op_t* function_op = loom_low_verify_context_function_op(context);
  IREE_RETURN_IF_ERROR(
      loom_aie2p_low_verify_array_abi_layout(context, state, function_op));
  if (!state->abi_layout_valid) {
    return iree_ok_status();
  }
  return loom_aie2p_low_verify_empty_signature(
      context, state->module, function_op, IREE_SV("aie2p-array-plan"));
}

static iree_status_t loom_aie2p_low_verify_core_resource(
    loom_low_verify_context_t* context,
    const loom_aie2p_low_verify_state_t* state, const loom_op_t* op) {
  if (!loom_low_resource_isa(op) ||
      loom_low_resource_import_kind(op) !=
          LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER) {
    return iree_ok_status();
  }
  const loom_type_t result_type =
      loom_module_value_type(state->module, loom_low_resource_result(op));
  // Shared Low verification diagnoses non-register and foreign-descriptor
  // resource results. This provider owns the narrower AIE2P native-pointer
  // ABI and must not duplicate those diagnostics.
  if (!loom_low_type_is_register(result_type) ||
      loom_low_register_type_descriptor_set_stable_id(result_type) !=
          AIE2P_CORE_DESCRIPTOR_SET_ID) {
    return iree_ok_status();
  }
  if (loom_low_register_type_class_id(result_type) ==
          AIE2P_CORE_REG_CLASS_ID_AIE2P_EP &&
      loom_low_register_type_unit_count(result_type) == 1) {
    return iree_ok_status();
  }

  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("result")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      loom_param_type(result_type),
      loom_param_string(IREE_SV("register class in [aie2p.ep] with 1 unit(s)")),
  };
  return loom_low_verify_context_emit(context, op, LOOM_ERR_TYPE_004, params,
                                      IREE_ARRAYSIZE(params));
}

static bool loom_aie2p_low_verify_is_core_body_op(
    const loom_low_verify_context_t* context, const loom_op_t* op) {
  const loom_region_t* function_body =
      loom_low_verify_context_function_body(context);
  return op->parent_block->parent_region == function_body;
}

static iree_status_t loom_aie2p_low_verify_emit_call_policy_error(
    loom_low_verify_context_t* context,
    const loom_aie2p_low_verify_state_t* state, const loom_op_t* op) {
  const loom_symbol_ref_t callee = loom_low_func_call_callee(op);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(state->target)),
      loom_param_string(loom_low_diagnostic_export_name(state->target)),
      loom_param_string(loom_low_diagnostic_config_key(state->target)),
      loom_param_string(state->function_name),
      loom_param_string(loom_low_diagnostic_operation_name(state->module, op)),
      loom_param_with_field_ref(
          loom_param_string(
              loom_low_diagnostic_symbol_name(state->module, callee)),
          loom_low_func_call_callee_diagnostic_ref()),
      loom_param_string(IREE_SV(
          "the selected target requires every Low call to be inlined before "
          "emission")),
  };
  return loom_low_verify_context_emit(context, op, LOOM_ERR_TARGET_072, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_aie2p_low_verify_emit_unsupported_core_op(
    loom_low_verify_context_t* context,
    const loom_aie2p_low_verify_state_t* state, const loom_op_t* op) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(state->target)),
      loom_param_string(loom_low_diagnostic_export_name(state->target)),
      loom_param_string(loom_low_diagnostic_config_key(state->target)),
      loom_param_string(state->function_name),
      loom_param_string(loom_low_diagnostic_operation_name(state->module, op)),
  };
  return loom_low_verify_context_emit(context, op, LOOM_ERR_TARGET_001, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_aie2p_low_verify_core_op(
    loom_low_verify_context_t* context,
    const loom_aie2p_low_verify_state_t* state,
    const loom_low_descriptor_packet_t* packet) {
  // AIE2P core emission consumes flat CFG body operations. Any nested op is
  // owned by an unsupported outer structural operation diagnosed at this
  // boundary, so descending into it would only produce redundant errors.
  if (!loom_aie2p_low_verify_is_core_body_op(context, packet->op)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_low_verify_core_resource(context, state, packet->op));
  if (packet->kind != LOOM_LOW_DESCRIPTOR_PACKET_NONE ||
      loom_aie2p_core_structure_classify(packet->op) !=
          LOOM_AIE2P_CORE_STRUCTURE_UNSUPPORTED) {
    return iree_ok_status();
  }
  if (loom_low_func_call_isa(packet->op)) {
    return loom_aie2p_low_verify_emit_call_policy_error(context, state,
                                                        packet->op);
  }
  return loom_aie2p_low_verify_emit_unsupported_core_op(context, state,
                                                        packet->op);
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
        iree_string_view_equal(
            loom_string_table_get(&state->module->strings, attr->name_id),
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
    return loom_string_table_get(&module->strings, contract_id);
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
  if (packet->descriptor_ordinal != AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER &&
      packet->descriptor_ordinal !=
          AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER_FOLD) {
    return iree_ok_status();
  }

  uint16_t attrs_attr_index = UINT16_MAX;
  const loom_named_attr_t* entry_attr = loom_aie2p_low_find_packet_attr(
      state, packet->op, IREE_SV("entry"), &attrs_attr_index);
  // Symbolic ordinals permit numeric values in the shared descriptor contract,
  // but a resident worker names a core function. Other malformed attribute
  // kinds are diagnosed by shared Low verification.
  if (entry_attr != NULL && entry_attr->value.kind == LOOM_ATTR_I64) {
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
        loom_param_u32(entry_attr->value.kind),
        loom_param_string(IREE_SV("symbol reference")),
    };
    return loom_low_verify_context_emit(context, packet->op,
                                        LOOM_ERR_TARGET_049, params,
                                        IREE_ARRAYSIZE(params));
  }
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
  if (state == NULL || !state->abi_layout_valid ||
      loom_low_verify_context_should_stop(context)) {
    return iree_ok_status();
  }
  if (state->program_kind == LOOM_AIE2P_LOW_VERIFY_PROGRAM_CORE) {
    return loom_aie2p_low_verify_core_op(context, state, packet);
  }
  if (packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    if (loom_low_return_isa(packet->op)) {
      return iree_ok_status();
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_op_name(state->module, packet->op)),
    };
    return loom_low_verify_context_emit(context, packet->op, LOOM_ERR_XDNA_014,
                                        params, IREE_ARRAYSIZE(params));
  }
  switch (packet->descriptor_ordinal) {
    case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_SENDER:
    case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_RECEIVER:
    case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_VIEW_SENDER:
    case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_VIEW_RECEIVER:
    case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION_SENDER:
    case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION_RECEIVER:
    case AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CHANNEL: {
      // Shared verification reports malformed result counts and non-register
      // types. Providers also run after those diagnostics to collect errors.
      if (packet->op->result_count != 1) {
        break;
      }
      const loom_type_t result_type =
          loom_module_value_type(state->module, loom_op_results(packet->op)[0]);
      if (!loom_type_is_register(result_type)) {
        break;
      }
      const loom_type_t* value_type =
          loom_type_register_value_type(result_type);
      if (value_type != NULL && loom_type_is_tile(*value_type)) {
        break;
      }
      const loom_diagnostic_param_t params[] = {
          loom_param_string(loom_low_descriptor_packet_diagnostic_key(
              state->target->descriptor_set, packet)),
          loom_param_with_field_ref(
              loom_param_type(result_type),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      };
      return loom_low_verify_context_emit(context, packet->op,
                                          LOOM_ERR_XDNA_015, params,
                                          IREE_ARRAYSIZE(params));
    }
    default:
      break;
  }
  return loom_aie2p_low_verify_worker(context, state, packet);
}

const loom_low_verify_provider_t loom_aie2p_low_verify_provider = {
    .name = IREE_SVL("amd-xdna-aie2p"),
    .begin_function = loom_aie2p_low_verify_begin_function,
    .verify_op = loom_aie2p_low_verify_op,
};
