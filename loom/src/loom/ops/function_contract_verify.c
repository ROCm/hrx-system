// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/function_contract_verify.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/projection.h"

static iree_string_view_t loom_function_contract_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_func_provider_emit_non_identity_target_field(
    const loom_module_t* module, const loom_op_t* provider_op,
    const loom_symbol_t* provider_symbol, const loom_op_t* target_op,
    const loom_symbol_t* target_symbol, uint8_t target_attr_index,
    iree_diagnostic_emitter_t emitter) {
  const loom_op_vtable_t* target_vtable = loom_op_vtable(module, target_op);
  const iree_string_view_t field_name = loom_attr_descriptor_name(
      &target_vtable->attr_descriptors[target_attr_index]);
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_function_contract_symbol_name(module, provider_symbol)),
      loom_param_string(
          loom_function_contract_symbol_name(module, target_symbol)),
      loom_param_string(field_name),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("target field authored here"),
      .op = target_op,
      .field_ref = loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                             target_attr_index),
  }};
  loom_diagnostic_emission_t emission = {
      .op = provider_op,
      .error = LOOM_ERR_TARGET_067,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_function_contract_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  // Targetless functions are valid generic program representations. A compile
  // invocation may bind an exact target later, so source verification cannot
  // require an authored target attribute.
  (void)module;
  (void)op;
  (void)emitter;
  return iree_ok_status();
}

iree_status_t loom_function_provider_contract_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));

  loom_func_like_t provider = loom_func_like_cast(module, (loom_op_t*)op);
  const loom_symbol_ref_t target_ref = loom_func_like_target(provider);
  if (!loom_symbol_ref_is_valid(target_ref)) return iree_ok_status();

  const loom_symbol_t* target_symbol =
      &module->symbols.entries[target_ref.symbol_id];
  const loom_symbol_ref_t provider_ref = loom_func_like_callee(provider);
  const loom_symbol_t* provider_symbol =
      &module->symbols.entries[provider_ref.symbol_id];
  loom_target_like_t target =
      loom_target_like_cast(module, target_symbol->defining_op);
  if (!loom_target_like_isa(target)) {
    // Partial target declarations have identity but no authored projection.
    return iree_ok_status();
  }

  const loom_target_like_descriptor_t* descriptor =
      loom_target_like_descriptor(target);
  for (uint8_t i = 0; i < descriptor->projection_count; ++i) {
    const uint8_t attr_index = descriptor->projections[i].attr_index;
    if (loom_attr_is_absent(loom_op_const_attrs(target.op)[attr_index])) {
      continue;
    }
    return loom_func_provider_emit_non_identity_target_field(
        module, op, provider_symbol, target.op, target_symbol, attr_index,
        emitter);
  }
  return iree_ok_status();
}
