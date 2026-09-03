// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/execution.h"

static iree_vm_ref_t* iree_vm_call_ref_argument_slot(
    const iree_vm_call_packet_t* call, uint16_t ordinal) {
  return ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? &call->ref_arguments.direct[ordinal]
             : &call->ref_arguments
                    .overflow[ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT];
}

static iree_vm_ref_t* iree_vm_call_ref_result_slot(
    const iree_vm_call_packet_t* call, uint16_t ordinal) {
  return ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? &call->ref_results.direct[ordinal]
             : &call->ref_results
                    .overflow[ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT];
}

IREE_API_EXPORT void iree_vm_call_ref_argument_load_borrow(
    const iree_vm_call_packet_t* call, uint16_t ordinal,
    iree_vm_ref_t* inout_ref) {
  const iree_vm_ref_t source = *iree_vm_call_ref_argument_slot(call, ordinal);
  iree_vm_ref_reset(inout_ref);
  *inout_ref = source.object ? iree_vm_ref_from_ptr_borrowed(
                                   source.object, iree_vm_ref_type(source))
                             : iree_vm_ref_null();
}

IREE_API_EXPORT void iree_vm_call_ref_argument_load_move(
    const iree_vm_call_packet_t* call, uint16_t ordinal,
    iree_vm_ref_t* inout_ref) {
  iree_vm_ref_t* source = iree_vm_call_ref_argument_slot(call, ordinal);
  iree_vm_ref_reset(inout_ref);
  *inout_ref = *source;
  *source = iree_vm_ref_null();
}

IREE_API_EXPORT void iree_vm_call_ref_result_store_move(
    const iree_vm_call_packet_t* call, uint16_t ordinal,
    iree_vm_ref_t* inout_ref) {
  iree_vm_ref_t* result = iree_vm_call_ref_result_slot(call, ordinal);
  iree_vm_ref_reset(result);
  *result = iree_vm_ref_move(inout_ref);
}

IREE_API_EXPORT iree_status_t iree_vm_module_function_resume_unreachable(
    iree_vm_module_t* module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  (void)module;
  (void)params;
  (void)out_outcome;
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "non-yielding module function was resumed");
}
