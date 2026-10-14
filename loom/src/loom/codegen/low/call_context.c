// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/call_context.h"

#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static iree_string_view_t loom_low_call_context_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t reference) {
  return loom_symbol_ref_is_valid(reference)
             ? loom_string_table_get(
                   &module->strings,
                   module->symbols.entries[reference.symbol_id].name_id)
             : IREE_SV("<unnamed>");
}

// Authored target witnesses can differ inside one specialized call closure.
// The retained context identity owns compatibility once specialization runs.
iree_status_t loom_low_verify_call_context(
    const loom_module_t* module, const loom_op_t* caller,
    const loom_target_function_version_t* caller_version,
    const loom_target_function_version_snapshot_t* versions,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  const loom_symbol_ref_t callee = loom_low_func_call_callee(op);
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= module->symbols.count) {
    return iree_ok_status();
  }
  const loom_op_t* callee_op =
      module->symbols.entries[callee.symbol_id].defining_op;
  const loom_func_like_t callee_function =
      loom_func_like_const_cast(module, callee_op);
  if (!loom_func_like_isa(callee_function)) {
    return iree_ok_status();
  }
  const loom_target_function_version_t* callee_version =
      loom_target_function_version_snapshot_at(versions, callee.symbol_id);
  const loom_symbol_ref_t caller_target = loom_low_function_target(caller);
  const loom_symbol_ref_t callee_target =
      loom_func_like_target(callee_function);
  const bool same_context =
      caller_version != NULL || callee_version != NULL
          ? caller_version != NULL && callee_version != NULL &&
                caller_version->target_context_ordinal ==
                    callee_version->target_context_ordinal
          : caller_target.module_id == callee_target.module_id &&
                caller_target.symbol_id == callee_target.symbol_id;
  if (same_context) {
    return iree_ok_status();
  }
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("callee defined here"),
      .op = callee_op,
  }};
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_with_field_ref(loom_param_string(IREE_SV("callee")),
                                loom_low_func_call_callee_diagnostic_ref()),
      loom_param_string(
          callee_version != NULL
              ? loom_target_facts_identity_name(
                    callee_version->resolved_target.facts)
              : loom_low_call_context_symbol_name(module, callee_target)),
      loom_param_string(
          caller_version != NULL
              ? loom_target_facts_identity_name(
                    caller_version->resolved_target.facts)
              : loom_low_call_context_symbol_name(module, caller_target)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TARGET_040,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related,
      .related_op_count = IREE_ARRAYSIZE(related),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
