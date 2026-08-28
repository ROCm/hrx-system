// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/facts.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"

static iree_string_view_t loom_vm_target_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) return IREE_SV("<unknown>");
  return module->strings.entries[string_id];
}

static iree_status_t loom_vm_target_reject_function_abi_attr(
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_func_symbol_facts_t* func_facts, const loom_error_def_t* error,
    iree_string_view_t field_name, bool* out_valid) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(func_facts->name),
      loom_param_string(field_name),
  };
  const loom_diagnostic_emission_t emission = {
      .op = func_facts->func_op,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  *out_valid = false;
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static iree_status_t loom_vm_target_validate_function_abi_attrs(
    const loom_module_t* module, const loom_func_symbol_facts_t* func_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid) {
  *out_valid = true;
  const loom_named_attr_slice_t attrs = func_facts->abi_attrs;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    const iree_string_view_t name =
        loom_vm_target_string_from_id(module, attr->name_id);
    if (!iree_string_view_equal(name, IREE_SV("suspending")) ||
        !func_facts->imports ||
        func_facts->abi_kind != LOOM_TARGET_ABI_VM_FUNCTION) {
      return loom_vm_target_reject_function_abi_attr(
          diagnostic_emitter, func_facts, LOOM_ERR_TARGET_021, name, out_valid);
    }
    if (attr->value.kind != LOOM_ATTR_BOOL) {
      return loom_vm_target_reject_function_abi_attr(
          diagnostic_emitter, func_facts, LOOM_ERR_TARGET_028, name, out_valid);
    }
  }
  return iree_ok_status();
}

const loom_target_fact_type_t loom_vm_target_fact_type = {
    .name = IREE_SVL("vm"),
    .storage_size = sizeof(loom_target_facts_t),
    .satisfies_identity_requirement =
        loom_target_facts_selector_satisfies_identity_requirement,
    .satisfies_specialization_requirement =
        loom_target_facts_structural_satisfy_specialization_requirement,
    .validate_function_abi_attrs = loom_vm_target_validate_function_abi_attrs,
};
