// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter.h"

#include "iree/vm/bytecode/interpreter_call.h"
#include "iree/vm/bytecode/interpreter_dispatch.h"
#include "iree/vm/bytecode/interpreter_frame.h"
#include "iree/vm/bytecode/module_reader.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/invocation_storage.h"

static bool iree_vm_bytecode_function_is_scalar_leaf(
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature) {
  // Verification requires the register bank to cover every result, so the
  // final comparison is equality for every published image.
  return function->local_byte_length_u16 == 0 &&
         function->ref_register_count_u16 == 0 &&
         function->local_ref_count_u32 == 0 &&
         function->function_register_count_u16 == 0 &&
         function->local_function_count_u32 == 0 &&
         function->value_register_count_u16 <=
             IREE_VM_CALL_DIRECT_REGISTER_COUNT &&
         function->value_register_count_u16 <=
             signature->result_value_count_u16;
}

static iree_status_t iree_vm_bytecode_function_start_transient(
    iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_t* invocation = params->execution.invocation;
  uint8_t* frame_checkpoint = invocation->stack_cursor;
  iree_vm_bytecode_execution_state_t state = {
      .call = &params->call,
      .program_counter = module->layout.functions.bytecode_data +
                         function->bytecode_offset_u32 +
                         sizeof(iree_vm_isa_control_block_record_t),
      .frame = NULL,
  };
  if (iree_vm_bytecode_function_is_scalar_leaf(function, signature)) {
    // Result banks are invocation-owned staging until the root succeeds. A
    // scalar-only leaf whose complete register bank fits the direct result
    // prefix can execute in place without reserving or copying a frame.
    state.values = params->call.value_results.direct;
    const uint16_t direct_value_count =
        iree_min(IREE_VM_CALL_DIRECT_REGISTER_COUNT,
                 signature->argument_value_count_u16);
    iree_vm_bytecode_frame_copy_direct_values(
        state.values, params->call.value_arguments.direct, direct_value_count);
  } else {
    const uint32_t ref_count =
        function->ref_register_count_u16 + function->local_ref_count_u32;
    const uint32_t function_count = function->function_register_count_u16 +
                                    function->local_function_count_u32;
    const iree_host_size_t frame_storage_size =
        function->value_register_count_u16 * sizeof(uint64_t) +
        (iree_host_size_t)ref_count * sizeof(iree_vm_ref_t) +
        (iree_host_size_t)function_count * sizeof(iree_vm_function_ref_t) +
        function->local_byte_length_u16;
    uint8_t* frame_storage = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_invocation_stack_reserve(
        invocation, frame_storage_size,
        iree_max(iree_max(iree_alignof(uint64_t), iree_alignof(iree_vm_ref_t)),
                 iree_alignof(iree_vm_function_ref_t)),
        &frame_checkpoint, &frame_storage));
    state.values = (uint64_t*)frame_storage;
    state.refs =
        (iree_vm_ref_t*)(frame_storage +
                         function->value_register_count_u16 * sizeof(uint64_t));
    state.local_refs = state.refs + function->ref_register_count_u16;
    state.functions = (iree_vm_function_ref_t*)(state.local_refs +
                                                function->local_ref_count_u32);
    state.local_functions =
        state.functions + function->function_register_count_u16;
    state.local_bytes =
        (uint8_t*)(state.local_functions + function->local_function_count_u32);
    iree_vm_bytecode_frame_initialize(function, signature, &state);
  }
  iree_status_t status = iree_vm_bytecode_dispatch(
      module, function, &params->execution, &state, out_outcome);
  const uint32_t ref_count =
      function->ref_register_count_u16 + function->local_ref_count_u32;
  iree_vm_bytecode_frame_reset_refs(state.refs, ref_count);
  iree_vm_invocation_stack_rewind(invocation, frame_checkpoint);
  return status;
}

static iree_status_t iree_vm_bytecode_function_start_durable(
    iree_vm_bytecode_module_t* module,
    const iree_vm_bytecode_v0_function_row_t* function,
    const iree_vm_bytecode_v0_signature_row_t* signature,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_bytecode_frame_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_query_frame_layout(function, &layout));
  iree_vm_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_invocation_push_frame(
      params, layout.frame, iree_vm_bytecode_frame_cleanup, &frame));

  uint8_t* frame_storage = (uint8_t*)iree_vm_frame_storage(frame);
  iree_vm_bytecode_execution_state_t* state =
      (iree_vm_bytecode_execution_state_t*)frame_storage;
  iree_vm_call_packet_t* call =
      (iree_vm_call_packet_t*)(frame_storage + layout.call_offset);
  *call = params->call;
  *state = (iree_vm_bytecode_execution_state_t){
      .call = call,
      .program_counter = module->layout.functions.bytecode_data +
                         function->bytecode_offset_u32 +
                         sizeof(iree_vm_isa_control_block_record_t),
      .frame = frame,
      .values = (uint64_t*)(frame_storage + layout.values_offset),
      .refs = (iree_vm_ref_t*)(frame_storage + layout.refs_offset),
      .local_refs = (iree_vm_ref_t*)(frame_storage + layout.local_refs_offset),
      .functions =
          (iree_vm_function_ref_t*)(frame_storage + layout.functions_offset),
      .local_functions =
          (iree_vm_function_ref_t*)(frame_storage +
                                    layout.local_functions_offset),
      .local_bytes = frame_storage + layout.local_bytes_offset,
  };
  iree_vm_bytecode_frame_initialize(function, signature, state);

  iree_status_t status = iree_vm_bytecode_dispatch(
      module, function, &params->execution, state, out_outcome);
  if (iree_status_is_ok(status) &&
      *out_outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    iree_vm_invocation_pop_frame(params->execution.invocation, frame);
  }
  return status;
}

iree_status_t iree_vm_bytecode_function_start(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  const iree_vm_bytecode_v0_function_row_t* function =
      &module->layout.functions.rows[params->function_ordinal];
  const iree_vm_bytecode_v0_signature_row_t* signature =
      iree_vm_bytecode_function_signature(&module->layout, function);
  return iree_any_bit_set(function->flags_u16,
                          IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD |
                              IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL)
             ? iree_vm_bytecode_function_start_durable(
                   module, function, signature, params, out_outcome)
             : iree_vm_bytecode_function_start_transient(
                   module, function, signature, params, out_outcome);
}

iree_status_t iree_vm_bytecode_function_resume(
    iree_vm_module_t* base_module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  const iree_vm_cancel_reason_t cancel_reason =
      iree_vm_invocation_cancel_reason(params->execution.invocation);
  if (cancel_reason != IREE_VM_CANCEL_REASON_NONE) {
    return iree_vm_invocation_cancel_status(cancel_reason);
  }

  iree_vm_bytecode_module_t* module = iree_vm_bytecode_module_cast(base_module);
  const uint16_t function_ordinal =
      iree_vm_frame_function_ordinal(params->frame);
  const iree_vm_bytecode_v0_function_row_t* function =
      &module->layout.functions.rows[function_ordinal];
  iree_vm_bytecode_execution_state_t* state =
      (iree_vm_bytecode_execution_state_t*)iree_vm_frame_storage(params->frame);
  iree_vm_bytecode_call_cleanup_completed(state);
  iree_status_t status = iree_vm_bytecode_dispatch(
      module, function, &params->execution, state, out_outcome);
  if (iree_status_is_ok(status) &&
      *out_outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED) {
    iree_vm_invocation_pop_frame(params->execution.invocation, params->frame);
  }
  return status;
}
