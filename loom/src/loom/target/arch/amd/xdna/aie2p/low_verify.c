// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/low_verify.h"

#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/array/abi_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/core_structure.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/configuration_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"
#include "loom/target/projection.h"

enum {
  LOOM_AIE2P_CONFIGURATION_PHASE_ENTRY = 0,
  LOOM_AIE2P_CONFIGURATION_PHASE_INITIALIZE = 1,
  LOOM_AIE2P_CONFIGURATION_PHASE_INVOKE = 2,
  LOOM_AIE2P_CONFIGURATION_REFERENCE_PROGRAM = 3,
};

// Facts collected by the shared verifier walk, indexed by function symbol.
// Deferred reference checks consume these facts after all definitions have
// been visited; source definition order does not affect admission.
typedef struct loom_aie2p_low_function_contract_t {
  // Function definition observed by the shared Low verifier.
  const loom_op_t* function;
  // First unbound resource import, when present.
  const loom_op_t* unbound_resource;
  // First command unavailable in each configuration phase, when present.
  const loom_op_t* phase_conflicts[3];
  // Number of entry commands in this function.
  uint32_t entry_count;
} loom_aie2p_low_function_contract_t;

typedef struct loom_aie2p_low_function_reference_t {
  // Referenced function's module-local symbol identity.
  loom_symbol_id_t symbol_id;
  // Configuration phase or complete core program required by this use.
  uint32_t kind;
} loom_aie2p_low_function_reference_t;

typedef struct loom_aie2p_low_verify_module_state_t {
  // Shared verifier arena owning the retained facts and references.
  iree_arena_allocator_t* arena;
  // Function contracts indexed by the module's symbol domain.
  loom_aie2p_low_function_contract_t* functions;
  // Function references retained in shared walk order.
  loom_aie2p_low_function_reference_t* references;
  // Number of retained function references.
  iree_host_size_t reference_count;
  // Allocated reference capacity.
  iree_host_size_t reference_capacity;
} loom_aie2p_low_verify_module_state_t;

static iree_status_t loom_aie2p_low_verify_begin_module(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_module_context_t* context, void** out_provider_state) {
  (void)provider;
  iree_arena_allocator_t* arena = loom_low_verify_module_context_arena(context);
  loom_aie2p_low_verify_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*state), (void**)&state));
  *state = (loom_aie2p_low_verify_module_state_t){.arena = arena};
  *out_provider_state = state;
  return iree_ok_status();
}

typedef struct loom_aie2p_low_verify_state_t {
  // Facts retained for reference checks after the shared module walk.
  loom_aie2p_low_verify_module_state_t* module_state;
  // Contract for this function in the module symbol domain.
  loom_aie2p_low_function_contract_t* contract;
  // Module containing the current function.
  const loom_module_t* module;
  // Target resolved for the current array program.
  const loom_low_resolved_target_t* target;
  // Borrowed array-program function name used in diagnostics.
  iree_string_view_t function_name;
  // A function-level diagnostic has already rejected configuration execution.
  bool skip_body_checks;
  // Whether the array ABI layout passed its public schema boundary.
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
  if (target->descriptor_set == NULL ||
      (target->descriptor_set->stable_id != AIE2P_ARRAY_DESCRIPTOR_SET_ID &&
       target->descriptor_set->stable_id !=
           AIE2P_CONFIGURATION_DESCRIPTOR_SET_ID &&
       target->descriptor_set->stable_id != AIE2P_CORE_DESCRIPTOR_SET_ID)) {
    return iree_ok_status();
  }

  loom_aie2p_low_verify_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      loom_low_verify_context_arena(context), sizeof(*state), (void**)&state));
  const loom_module_t* module = loom_low_verify_context_module(context);
  const loom_op_t* function = loom_low_verify_context_function_op(context);
  loom_aie2p_low_verify_module_state_t* module_state =
      loom_low_verify_context_provider_module_state(context);
  if (!module_state->functions) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        module_state->arena, module->symbols.count,
        sizeof(*module_state->functions), (void**)&module_state->functions));
    memset(module_state->functions, 0,
           module->symbols.count * sizeof(*module_state->functions));
  }
  const loom_symbol_ref_t symbol =
      loom_func_like_callee(loom_func_like_const_cast(module, function));
  *state = (loom_aie2p_low_verify_state_t){
      .abi_layout_valid = true,
      .module_state = module_state,
      .contract = &module_state->functions[symbol.symbol_id],
      .module = loom_low_verify_context_module(context),
      .target = target,
      .function_name = loom_low_diagnostic_function_name(
          loom_low_verify_context_module(context),
          loom_low_verify_context_function_op(context)),
  };
  state->contract->function = function;
  *out_provider_state = state;
  if (target->descriptor_set->stable_id == AIE2P_ARRAY_DESCRIPTOR_SET_ID) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_low_verify_array_abi_layout(context, state, function));
    if (!state->abi_layout_valid) {
      return iree_ok_status();
    }
    return loom_aie2p_low_verify_empty_signature(
        context, state->module, function, IREE_SV("aie2p-array-plan"));
  }
  if (target->descriptor_set->stable_id ==
      AIE2P_CONFIGURATION_DESCRIPTOR_SET_ID) {
    const loom_region_t* body = loom_low_verify_context_function_body(context);
    const loom_block_t* entry = loom_region_const_entry_block(body);
    if (entry->arg_count || function->result_count) {
      state->skip_body_checks = true;
      const loom_diagnostic_param_t params[] = {
          loom_param_string(state->function_name),
          loom_param_u32(entry->arg_count),
          loom_param_u32(function->result_count),
      };
      IREE_RETURN_IF_ERROR(
          loom_low_verify_context_emit(context, function, LOOM_ERR_XDNA_050,
                                       params, IREE_ARRAYSIZE(params)));
    }
    if (body->block_count != 1) {
      state->skip_body_checks = true;
      const loom_diagnostic_param_t params[] = {
          loom_param_string(state->function_name),
          loom_param_string(
              loom_low_diagnostic_operation_name(module, function)),
      };
      IREE_RETURN_IF_ERROR(
          loom_low_verify_context_emit(context, function, LOOM_ERR_XDNA_035,
                                       params, IREE_ARRAYSIZE(params)));
    }
  }
  return iree_ok_status();
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
  // Low helpers are valid authored and intermediate IR. The common pipeline
  // applies the target's required-inline policy before native frame admission.
  if (loom_low_func_call_isa(packet->op)) {
    return iree_ok_status();
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

static iree_status_t loom_aie2p_low_verify_function_reference(
    loom_low_verify_context_t* context,
    const loom_aie2p_low_verify_state_t* state,
    const loom_low_descriptor_packet_t* packet, iree_string_view_t field_name,
    iree_string_view_t expected_contract) {
  uint16_t attrs_attr_index = UINT16_MAX;
  const loom_named_attr_t* entry_attr = loom_aie2p_low_find_packet_attr(
      state, packet->op, field_name, &attrs_attr_index);
  // Symbolic ordinals permit numeric values in the shared descriptor contract,
  // but a program dependency names a function. Other malformed attribute
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
            loom_param_string(field_name),
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
  if (iree_string_view_equal(actual_contract, expected_contract)) {
    if (state->target->descriptor_set->stable_id !=
        AIE2P_CONFIGURATION_DESCRIPTOR_SET_ID) {
      return iree_ok_status();
    }
    const bool is_program =
        iree_string_view_equal(field_name, IREE_SV("program"));
    const bool is_source_worker =
        is_program && loom_func_def_isa(entry_function.op);
    if (!loom_low_func_def_isa(entry_function.op) && !is_source_worker) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(field_name),
          loom_param_string(
              loom_low_diagnostic_symbol_name(state->module, entry_ref)),
          loom_param_string(is_program ? IREE_SV("source or Low function")
                                       : IREE_SV("Low function")),
      };
      return loom_low_verify_context_emit(context, packet->op,
                                          LOOM_ERR_XDNA_036, params,
                                          IREE_ARRAYSIZE(params));
    }
    uint16_t argument_count = 0;
    loom_func_like_arg_ids(entry_function, &argument_count);
    if (argument_count || entry_function.op->result_count) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(field_name),
          loom_param_string(
              loom_low_diagnostic_symbol_name(state->module, entry_ref)),
          loom_param_u32(argument_count),
          loom_param_u32(entry_function.op->result_count),
      };
      return loom_low_verify_context_emit(context, packet->op,
                                          LOOM_ERR_XDNA_037, params,
                                          IREE_ARRAYSIZE(params));
    }
    // Source workers carry their target contract through ordinary source
    // lowering. Their resulting Low bodies participate in the same complete
    // image checks as authored Low workers; there is no Low body to inspect
    // during this verification of the mixed-level source module.
    if (is_source_worker) {
      return iree_ok_status();
    }
    loom_aie2p_low_verify_module_state_t* module_state = state->module_state;
    if (module_state->reference_count == module_state->reference_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          module_state->arena, module_state->reference_count,
          module_state->reference_count + 1, sizeof(*module_state->references),
          &module_state->reference_capacity,
          (void**)&module_state->references));
    }
    const uint32_t kind =
        iree_string_view_equal(field_name, IREE_SV("initialize"))
            ? LOOM_AIE2P_CONFIGURATION_PHASE_INITIALIZE
        : iree_string_view_equal(field_name, IREE_SV("invoke"))
            ? LOOM_AIE2P_CONFIGURATION_PHASE_INVOKE
            : LOOM_AIE2P_CONFIGURATION_REFERENCE_PROGRAM;
    module_state->references[module_state->reference_count++] =
        (loom_aie2p_low_function_reference_t){entry_ref.symbol_id, kind};
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
          loom_param_string(field_name),
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

// Phase permissions describe the physical command semantics. The shared walk
// retains the first conflict for each possible use of a configuration helper.
static void loom_aie2p_low_record_configuration_command(
    loom_aie2p_low_function_contract_t* contract,
    const loom_low_descriptor_packet_t* packet) {
  uint32_t phases = 0;
  switch (packet->descriptor_ordinal) {
    case AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_CONSTANT:
      phases = 7;
      break;
    case AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_ENTRY:
      phases = 1u << LOOM_AIE2P_CONFIGURATION_PHASE_ENTRY;
      ++contract->entry_count;
      break;
    case AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_PROGRAM_LOAD:
    case AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_DATA_RESERVE:
      phases = 1u << LOOM_AIE2P_CONFIGURATION_PHASE_INITIALIZE;
      break;
    case AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_BINDING:
    case AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_RANGE:
    case AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_WRITE_ADDRESS:
    case AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_SHIM_DESCRIPTOR:
      phases = 1u << LOOM_AIE2P_CONFIGURATION_PHASE_INVOKE;
      break;
    default:
      phases = (1u << LOOM_AIE2P_CONFIGURATION_PHASE_INITIALIZE) |
               (1u << LOOM_AIE2P_CONFIGURATION_PHASE_INVOKE);
      break;
  }
  for (uint32_t i = 0; i < 3; ++i) {
    if (!(phases & (1u << i)) && !contract->phase_conflicts[i]) {
      contract->phase_conflicts[i] = packet->op;
    }
  }
}

static iree_status_t loom_aie2p_low_verify_op(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state,
    const loom_low_descriptor_packet_t* packet) {
  (void)provider;
  loom_aie2p_low_verify_state_t* state = provider_state;
  if (state == NULL || state->skip_body_checks || !state->abi_layout_valid ||
      loom_low_verify_context_should_stop(context)) {
    return iree_ok_status();
  }
  if (state->target->descriptor_set->stable_id ==
      AIE2P_CORE_DESCRIPTOR_SET_ID) {
    if (!state->contract->unbound_resource &&
        loom_low_resource_isa(packet->op)) {
      state->contract->unbound_resource = packet->op;
    }
    return loom_aie2p_low_verify_core_op(context, state, packet);
  }
  if (state->target->descriptor_set->stable_id ==
      AIE2P_CONFIGURATION_DESCRIPTOR_SET_ID) {
    if (packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
      if (!loom_low_return_isa(packet->op) &&
          !loom_low_assume_isa(packet->op)) {
        const loom_diagnostic_param_t params[] = {
            loom_param_string(state->function_name),
            loom_param_string(
                loom_low_diagnostic_operation_name(state->module, packet->op)),
        };
        return loom_low_verify_context_emit(context, packet->op,
                                            LOOM_ERR_XDNA_035, params,
                                            IREE_ARRAYSIZE(params));
      }
      return iree_ok_status();
    }
    loom_aie2p_low_record_configuration_command(state->contract, packet);
  } else if (state->target->descriptor_set->stable_id ==
             AIE2P_ARRAY_DESCRIPTOR_SET_ID) {
    if (packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
      if (loom_low_return_isa(packet->op)) {
        return iree_ok_status();
      }
      const loom_diagnostic_param_t params[] = {
          loom_param_string(
              loom_low_diagnostic_operation_name(state->module, packet->op)),
      };
      return loom_low_verify_context_emit(context, packet->op,
                                          LOOM_ERR_XDNA_014, params,
                                          IREE_ARRAYSIZE(params));
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
        // types. The AIE array contract additionally requires a tile payload.
        if (packet->op->result_count != 1) {
          break;
        }
        const loom_type_t result_type = loom_module_value_type(
            state->module, loom_op_results(packet->op)[0]);
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
  } else if (packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return iree_ok_status();
  }
  if (state->target->descriptor_set->stable_id ==
      AIE2P_ARRAY_DESCRIPTOR_SET_ID) {
    if (packet->descriptor_ordinal == AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER ||
        packet->descriptor_ordinal ==
            AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER_FOLD) {
      return loom_aie2p_low_verify_function_reference(
          context, state, packet, IREE_SV("entry"),
          IREE_SV("amd.xdna.aie2p.core"));
    }
  } else if (packet->descriptor_ordinal ==
             AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_ENTRY) {
    IREE_RETURN_IF_ERROR(loom_aie2p_low_verify_function_reference(
        context, state, packet, IREE_SV("initialize"),
        IREE_SV("amd.xdna.aie2p.configuration")));
    return loom_aie2p_low_verify_function_reference(
        context, state, packet, IREE_SV("invoke"),
        IREE_SV("amd.xdna.aie2p.configuration"));
  } else if (packet->descriptor_ordinal ==
             AIE2P_CONFIGURATION_DESCRIPTOR_REF_CONFIGURATION_PROGRAM_LOAD) {
    return loom_aie2p_low_verify_function_reference(
        context, state, packet, IREE_SV("program"),
        IREE_SV("amd.xdna.aie2p.core"));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_aie2p_low_configuration_command_name(
    const loom_op_t* op) {
  const loom_low_descriptor_set_t* descriptors =
      loom_aie2p_configuration_descriptor_set();
  loom_low_descriptor_packet_t packet;
  loom_low_descriptor_packet_initialize(descriptors, op, &packet);
  return loom_low_descriptor_packet_diagnostic_key(descriptors, &packet);
}

static iree_status_t loom_aie2p_low_verify_end_function(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_context_t* context, void* provider_state) {
  (void)provider;
  const loom_aie2p_low_verify_state_t* state = provider_state;
  if (!state || state->skip_body_checks ||
      state->target->descriptor_set->stable_id !=
          AIE2P_CONFIGURATION_DESCRIPTOR_SET_ID) {
    return iree_ok_status();
  }
  const loom_op_t* function = state->contract->function;
  const bool is_entry =
      loom_func_like_abi(loom_func_like_const_cast(state->module, function)) ==
      LOOM_TARGET_ABI_ARRAY_PROGRAM;
  const uint32_t expected_count = is_entry ? 1 : 0;
  if (state->contract->entry_count != expected_count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(state->function_name),
        loom_param_u32(expected_count),
        loom_param_u32(state->contract->entry_count),
    };
    IREE_RETURN_IF_ERROR(loom_low_verify_context_emit(
        context, function, LOOM_ERR_XDNA_039, params, IREE_ARRAYSIZE(params)));
  }
  if (is_entry && state->contract->phase_conflicts[0]) {
    const loom_op_t* conflict = state->contract->phase_conflicts[0];
    const loom_diagnostic_param_t params[] = {
        loom_param_string(state->function_name),
        loom_param_string(IREE_SV("entry")),
        loom_param_string(loom_aie2p_low_configuration_command_name(conflict)),
    };
    IREE_RETURN_IF_ERROR(loom_low_verify_context_emit(
        context, conflict, LOOM_ERR_XDNA_038, params, IREE_ARRAYSIZE(params)));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_low_verify_end_module(
    const loom_low_verify_provider_t* provider,
    loom_low_verify_module_context_t* context, void* provider_state) {
  (void)provider;
  const loom_aie2p_low_verify_module_state_t* state = provider_state;
  const loom_module_t* module = loom_low_verify_module_context_module(context);
  for (iree_host_size_t i = 0;
       i < state->reference_count &&
       !loom_low_verify_module_context_should_stop(context);
       ++i) {
    const loom_aie2p_low_function_reference_t* reference =
        &state->references[i];
    const loom_aie2p_low_function_contract_t* contract =
        &state->functions[reference->symbol_id];
    const iree_string_view_t name =
        loom_low_diagnostic_function_name(module, contract->function);
    if (reference->kind == LOOM_AIE2P_CONFIGURATION_REFERENCE_PROGRAM) {
      if (contract->unbound_resource) {
        const loom_diagnostic_param_t params[] = {
            loom_param_string(name),
            loom_param_string(loom_low_diagnostic_operation_name(
                module, contract->unbound_resource)),
        };
        IREE_RETURN_IF_ERROR(loom_low_verify_module_context_emit(
            context, contract->unbound_resource, LOOM_ERR_XDNA_040, params,
            IREE_ARRAYSIZE(params)));
      }
    } else if (contract->phase_conflicts[reference->kind]) {
      const loom_op_t* conflict = contract->phase_conflicts[reference->kind];
      const loom_diagnostic_param_t params[] = {
          loom_param_string(name),
          loom_param_string(reference->kind ==
                                    LOOM_AIE2P_CONFIGURATION_PHASE_INITIALIZE
                                ? IREE_SV("initialization")
                                : IREE_SV("invocation")),
          loom_param_string(
              loom_aie2p_low_configuration_command_name(conflict)),
      };
      IREE_RETURN_IF_ERROR(loom_low_verify_module_context_emit(
          context, conflict, LOOM_ERR_XDNA_038, params,
          IREE_ARRAYSIZE(params)));
    }
  }
  return iree_ok_status();
}

const loom_low_verify_provider_t loom_aie2p_low_verify_provider = {
    .name = IREE_SVL("amd-xdna-aie2p"),
    .begin_module = loom_aie2p_low_verify_begin_module,
    .begin_function = loom_aie2p_low_verify_begin_function,
    .verify_op = loom_aie2p_low_verify_op,
    .end_function = loom_aie2p_low_verify_end_function,
    .end_module = loom_aie2p_low_verify_end_module,
};
