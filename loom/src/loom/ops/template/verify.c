// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/function_contract_verify.h"
#include "loom/ops/template/ops.h"

iree_status_t loom_template_decl_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  return loom_function_contract_verify(module, op, emitter);
}

iree_status_t loom_template_def_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  return loom_function_provider_contract_verify(module, op, emitter);
}

iree_status_t loom_template_ukernel_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  return loom_function_provider_contract_verify(module, op, emitter);
}

iree_status_t loom_template_apply_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  return loom_function_call_contract_verify(
      module, op, loom_template_apply_family(op),
      loom_template_apply_operands(op), loom_template_apply_results(op),
      emitter);
}

iree_status_t loom_template_call_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  return loom_function_call_contract_verify(
      module, op, loom_template_call_callee(op),
      loom_template_call_operands(op), loom_template_call_results(op), emitter);
}
