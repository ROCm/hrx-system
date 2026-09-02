// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/target_binding.h"

#include <inttypes.h>
#include <stdint.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/target/function_contract.h"
#include "loom/target/registers.h"

bool loom_low_register_type_resolver_try_resolve(
    const loom_low_register_type_resolver_t* resolver, loom_type_t type,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class) {
  *out_descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (out_descriptor_register_class) {
    *out_descriptor_register_class = NULL;
  }
  if (!loom_low_type_is_register(type)) {
    return false;
  }
  if (loom_low_register_type_descriptor_set_stable_id(type) !=
      resolver->descriptor_set->stable_id) {
    return false;
  }
  const uint16_t descriptor_register_class_id =
      loom_low_register_type_class_id(type);
  if (descriptor_register_class_id == LOOM_LOW_REG_CLASS_NONE ||
      descriptor_register_class_id >=
          resolver->descriptor_set->reg_class_count) {
    return false;
  }
  *out_descriptor_register_class_id = descriptor_register_class_id;
  if (out_descriptor_register_class) {
    *out_descriptor_register_class =
        &resolver->descriptor_set->reg_classes[descriptor_register_class_id];
  }
  return true;
}

bool loom_low_register_type_resolver_has_class_flags(
    const loom_low_register_type_resolver_t* resolver, loom_type_t type,
    loom_low_reg_class_flags_t flags) {
  uint16_t descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* descriptor_register_class = NULL;
  return loom_low_register_type_resolver_try_resolve(
             resolver, type, &descriptor_register_class_id,
             &descriptor_register_class) &&
         iree_all_bits_set(descriptor_register_class->flags, flags);
}

static iree_status_t loom_low_emit(iree_diagnostic_emitter_t emitter,
                                   const loom_op_t* op,
                                   const loom_error_def_t* error,
                                   const loom_diagnostic_param_t* params,
                                   iree_host_size_t param_count,
                                   const loom_diagnostic_related_op_t* related,
                                   iree_host_size_t related_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related,
      .related_op_count = related_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static const loom_symbol_t* loom_low_lookup_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return NULL;
  }
  return symbol;
}

static iree_string_view_t loom_low_symbol_name(const loom_module_t* module,
                                               loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_function_name(const loom_module_t* module,
                                                 const loom_op_t* low_func_op) {
  if (loom_low_func_def_isa(low_func_op)) {
    return loom_low_symbol_name(module, loom_low_func_def_callee(low_func_op));
  }
  if (loom_low_kernel_def_isa(low_func_op)) {
    return loom_low_symbol_name(module,
                                loom_low_kernel_def_callee(low_func_op));
  }
  if (loom_low_func_decl_isa(low_func_op)) {
    return loom_low_symbol_name(module, loom_low_func_decl_callee(low_func_op));
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_string_or_empty(const loom_module_t* module,
                                                   loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_low_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) {
    return IREE_SV("unresolved");
  }
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

static iree_status_t loom_low_emit_symbol_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* op, loom_symbol_ref_t ref, const loom_symbol_t* symbol,
    uint16_t attr_index, iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, ref)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_string(loom_low_symbol_definition_name(symbol)),
      loom_param_string(expected_kind),
  };
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("defined here"),
      .op = symbol ? symbol->defining_op : NULL,
  }};
  return loom_low_emit(
      emitter, op, LOOM_ERR_SYMBOL_003, params, IREE_ARRAYSIZE(params), related,
      symbol && symbol->defining_op ? IREE_ARRAYSIZE(related) : 0);
}

static iree_status_t loom_low_emit_unresolved_symbol(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* op, loom_symbol_ref_t ref, uint16_t attr_index) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, ref)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_SYMBOL_002, params,
                       IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_emit_missing_descriptor_set(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* low_func_op, uint16_t contract_attr_index,
    iree_string_view_t descriptor_set_key) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_function_name(module, low_func_op)),
      loom_param_with_field_ref(
          loom_param_string(descriptor_set_key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    contract_attr_index)),
  };
  return loom_low_emit(emitter, low_func_op, LOOM_ERR_TARGET_044, params,
                       IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_emit_unsupported_representation_contract(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* low_func_op, uint16_t repr_contract_attr_index,
    iree_string_view_t representation_contract, iree_string_view_t target_name,
    iree_string_view_t target_contract) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_function_name(module, low_func_op)),
      loom_param_with_field_ref(
          loom_param_string(representation_contract),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    repr_contract_attr_index)),
      loom_param_string(target_name),
      loom_param_string(target_contract),
  };
  return loom_low_emit(emitter, low_func_op, LOOM_ERR_TARGET_065, params,
                       IREE_ARRAYSIZE(params), NULL, 0);
}

static bool loom_low_get_function_target_ref(const loom_op_t* low_func_op,
                                             loom_symbol_ref_t* out_target_ref,
                                             uint16_t* out_target_attr_index) {
  if (loom_low_func_def_isa(low_func_op)) {
    *out_target_ref = loom_low_func_def_target(low_func_op);
    *out_target_attr_index = loom_low_func_def_target_ATTR_INDEX;
    return true;
  }
  if (loom_low_kernel_def_isa(low_func_op)) {
    *out_target_ref = loom_low_kernel_def_target(low_func_op);
    *out_target_attr_index = loom_low_kernel_def_target_ATTR_INDEX;
    return true;
  }
  if (loom_low_func_decl_isa(low_func_op)) {
    *out_target_ref = loom_low_func_decl_target(low_func_op);
    *out_target_attr_index = loom_low_func_decl_target_ATTR_INDEX;
    return true;
  }
  return false;
}

static iree_status_t loom_low_resolve_function_representation(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* registry,
    iree_diagnostic_emitter_t emitter,
    iree_string_view_t* out_descriptor_set_key,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set_key = iree_string_view_empty();
  *out_descriptor_set = NULL;
  if (registry == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low function representation resolution requires a descriptor "
        "registry");
  }

  const loom_func_like_t low_func =
      loom_func_like_const_cast(module, low_func_op);
  *out_descriptor_set_key =
      loom_low_string_or_empty(module, loom_func_like_repr_contract(low_func));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup(registry, *out_descriptor_set_key);
  if (!descriptor_set) {
    return loom_low_emit_missing_descriptor_set(
        emitter, module, low_func_op, low_func.vtable->repr_contract_attr_index,
        *out_descriptor_set_key);
  }
  *out_descriptor_set = descriptor_set;
  return iree_ok_status();
}

static iree_status_t loom_low_resolve_function_target_facts(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_diagnostic_emitter_t emitter, loom_symbol_fact_table_t* fact_table,
    loom_low_resolved_target_t* out_target) {
  const loom_func_like_t low_func =
      loom_func_like_const_cast(module, low_func_op);
  const loom_symbol_ref_t func_ref = loom_func_like_callee(low_func);

  const loom_symbol_facts_base_t* base_facts = NULL;
  iree_status_t status = loom_symbol_fact_table_lookup_ref(
      fact_table, module, func_ref, &base_facts);
  const loom_func_symbol_facts_t* func_facts =
      iree_status_is_ok(status) ? loom_func_symbol_facts_cast(base_facts)
                                : NULL;
  if (iree_status_is_ok(status) && func_facts == NULL) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low function symbol must resolve to func symbol facts");
  }
  bool contract_valid = function_target_facts != NULL;
  if (iree_status_is_ok(status) && function_target_facts != NULL) {
    out_target->target_facts = function_target_facts;
  } else if (iree_status_is_ok(status)) {
    status = loom_target_function_contract_resolve_facts(
        module, fact_table, func_facts, emitter, fact_table->arena,
        &contract_valid, &out_target->target_facts);
  }
  loom_target_workgroup_size_t workgroup_size = {0};
  if (iree_status_is_ok(status) && contract_valid &&
      loom_low_kernel_def_static_workgroup_size(low_func_op, &workgroup_size)) {
    status = loom_target_function_contract_refine_hal_workgroup_size(
        func_facts, out_target->target_name, &workgroup_size,
        out_target->target_facts, emitter, fact_table->arena, &contract_valid,
        &out_target->target_facts);
  }
  if (iree_status_is_ok(status) && contract_valid) {
    out_target->feature_bits = loom_low_resolved_target_bundle(out_target)
                                   ->config->contract_feature_bits;
  }

  if (!iree_status_is_ok(status)) {
    return status;
  }
  if (!contract_valid) {
    return iree_ok_status();
  }

  const iree_string_view_t target_contract_key =
      loom_low_resolved_target_bundle(out_target)->config->contract_set_key;
  if (!loom_low_descriptor_set_supports_target_contract(descriptor_set,
                                                        target_contract_key)) {
    return loom_low_emit_unsupported_representation_contract(
        emitter, module, low_func_op, low_func.vtable->repr_contract_attr_index,
        out_target->descriptor_set_key, out_target->target_name,
        target_contract_key);
  }
  out_target->descriptor_set = descriptor_set;
  return iree_ok_status();
}

iree_status_t loom_low_resolve_function_target(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    const loom_op_t* low_func_op,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_registry_t* registry,
    iree_diagnostic_emitter_t emitter, loom_low_resolved_target_t* out_target) {
  *out_target = (loom_low_resolved_target_t){0};
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  uint16_t target_attr_index = 0;
  if (!loom_low_get_function_target_ref(low_func_op, &target_ref,
                                        &target_attr_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def, low.kernel.def, or "
                            "low.func.decl");
  }

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_representation(
      module, low_func_op, registry, emitter, &out_target->descriptor_set_key,
      &descriptor_set));
  if (descriptor_set == NULL) {
    return iree_ok_status();
  }

  if (function_target_facts == NULL && !loom_symbol_ref_is_valid(target_ref)) {
    out_target->descriptor_set = descriptor_set;
    return iree_ok_status();
  }

  if (function_target_facts != NULL) {
    out_target->target_name =
        loom_target_facts_identity_name(function_target_facts);
    return loom_low_resolve_function_target_facts(
        module, low_func_op, function_target_facts, descriptor_set, emitter,
        fact_table, out_target);
  }

  const loom_symbol_t* target_symbol =
      loom_low_lookup_defined_symbol(module, target_ref);
  if (!target_symbol) {
    return loom_low_emit_unresolved_symbol(emitter, module, low_func_op,
                                           target_ref, target_attr_index);
  }

  out_target->target_name = loom_low_symbol_name(module, target_ref);

  if (loom_symbol_implements(target_symbol, LOOM_SYMBOL_INTERFACE_TARGET)) {
    return loom_low_resolve_function_target_facts(
        module, low_func_op, /*function_target_facts=*/NULL, descriptor_set,
        emitter, fact_table, out_target);
  }
  return loom_low_emit_symbol_kind_mismatch(
      emitter, module, low_func_op, target_ref, target_symbol,
      target_attr_index, IREE_SV("target record"));
}
