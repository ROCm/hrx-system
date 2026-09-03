// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/execution.h"

#include "iree/vm/invocation_storage.h"

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

//===----------------------------------------------------------------------===//
// Invocation State
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_vm_invocation_wake_callback_t
iree_vm_invocation_wake_callback(iree_vm_invocation_t* invocation) {
  return invocation->wake_callback;
}

IREE_API_EXPORT iree_vm_cancel_reason_t
iree_vm_invocation_cancel_reason(const iree_vm_invocation_t* invocation) {
  const int32_t reason =
      iree_atomic_load(&invocation->cancel_reason, iree_memory_order_acquire);
  return reason == IREE_VM_INVOCATION_CANCEL_REASON_IDLE
             ? IREE_VM_CANCEL_REASON_NONE
             : (iree_vm_cancel_reason_t)reason;
}

//===----------------------------------------------------------------------===//
// Composite Frames
//===----------------------------------------------------------------------===//

static bool iree_vm_execution_align_address(uintptr_t address,
                                            iree_host_size_t alignment,
                                            uintptr_t* out_address) {
  iree_host_size_t aligned_address = 0;
  if (!iree_host_size_checked_align((iree_host_size_t)address, alignment,
                                    &aligned_address)) {
    return false;
  }
  *out_address = (uintptr_t)aligned_address;
  return true;
}

static void iree_vm_invocation_pop_top_frame(iree_vm_invocation_t* invocation) {
  iree_vm_frame_t* frame = invocation->top_frame;
  if (frame->cleanup) frame->cleanup(frame);
  invocation->top_frame = frame->parent;
  invocation->stack_cursor = frame->allocation_begin;
}

void iree_vm_invocation_unwind_to(iree_vm_invocation_t* invocation,
                                  uint8_t* stack_cursor) {
  while (invocation->top_frame && invocation->stack_cursor > stack_cursor) {
    iree_vm_invocation_pop_top_frame(invocation);
  }
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_push_frame(
    const iree_vm_module_function_start_params_t* start_params,
    iree_vm_frame_layout_t layout, iree_vm_frame_cleanup_fn_t cleanup,
    iree_vm_frame_t** out_frame) {
  if (layout.storage_alignment == 0 ||
      !iree_host_size_is_power_of_two(layout.storage_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "frame alignment must be a power of two");
  }
  iree_vm_invocation_t* invocation = start_params->execution.invocation;
  const iree_vm_callback_context_t* callback_context =
      invocation->callback_context;

  uintptr_t header_address = 0;
  if (!iree_vm_execution_align_address((uintptr_t)invocation->stack_cursor,
                                       iree_alignof(iree_vm_frame_t),
                                       &header_address) ||
      header_address > (uintptr_t)invocation->storage_end ||
      (uintptr_t)invocation->storage_end - header_address <
          sizeof(iree_vm_frame_t)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "invocation frame header exceeds capacity");
  }
  uintptr_t storage_address = 0;
  if (!iree_vm_execution_align_address(header_address + sizeof(iree_vm_frame_t),
                                       layout.storage_alignment,
                                       &storage_address) ||
      storage_address > (uintptr_t)invocation->storage_end ||
      layout.storage_size >
          (iree_host_size_t)((uintptr_t)invocation->storage_end -
                             storage_address)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "invocation frame payload exceeds capacity");
  }

  iree_vm_frame_t* frame = (iree_vm_frame_t*)header_address;
  frame->allocation_begin = invocation->stack_cursor;
  frame->storage = (void*)storage_address;
  frame->parent = invocation->top_frame;
  frame->linked_module = start_params->execution.linked_module;
  frame->cleanup = cleanup;
  frame->function_ordinal = start_params->function_ordinal;
  frame->may_yield = callback_context->may_yield;
  invocation->stack_cursor = (uint8_t*)storage_address + layout.storage_size;
  invocation->top_frame = frame;
  *out_frame = frame;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_invocation_pop_frame(
    iree_vm_invocation_t* invocation, iree_vm_frame_t* frame) {
  IREE_ASSERT(invocation->state == IREE_VM_INVOCATION_STATE_RUNNING);
  IREE_ASSERT(invocation->top_frame == frame);
  iree_vm_invocation_pop_top_frame(invocation);
}

IREE_API_EXPORT void* iree_vm_frame_storage(iree_vm_frame_t* frame) {
  return frame->storage;
}

IREE_API_EXPORT uint16_t
iree_vm_frame_function_ordinal(const iree_vm_frame_t* frame) {
  return frame->function_ordinal;
}

//===----------------------------------------------------------------------===//
// Nested Calls And Function Values
//===----------------------------------------------------------------------===//

static iree_status_t iree_vm_execution_check_child_yield(
    const iree_vm_callback_context_t* callback_context, bool may_yield) {
  if (may_yield && !callback_context->may_yield) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "non-yielding module function requested a yielding child");
  }
  return iree_ok_status();
}

static void iree_vm_execution_publish_call_request(
    iree_vm_invocation_t* invocation,
    const iree_vm_linked_module_t* linked_module, uint16_t function_ordinal,
    bool may_yield, const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome) {
  const iree_vm_callback_context_t* callback_context =
      invocation->callback_context;
  IREE_ASSERT(invocation->state == IREE_VM_INVOCATION_STATE_RUNNING);
  IREE_ASSERT(callback_context != NULL);
  IREE_ASSERT(callback_context->call_request->linked_module == NULL);
  *callback_context->call_request = (iree_vm_call_request_t){
      .linked_module = linked_module,
      .packet = *call,
      .function_ordinal = function_ordinal,
      .may_yield = may_yield,
  };
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
}

IREE_API_EXPORT iree_status_t
iree_vm_invocation_call_local(const iree_vm_module_execution_t* execution,
                              iree_vm_module_local_function_t local_function,
                              const iree_vm_call_packet_t* call,
                              iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_t* invocation = execution->invocation;
  const iree_vm_callback_context_t* callback_context =
      invocation->callback_context;
  const bool may_yield = iree_any_bit_set(
      local_function.flags, IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD);
  IREE_RETURN_IF_ERROR(
      iree_vm_execution_check_child_yield(callback_context, may_yield));
  iree_vm_execution_publish_call_request(invocation, execution->linked_module,
                                         local_function.function_ordinal,
                                         may_yield, call, out_outcome);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_call_import(
    const iree_vm_module_execution_t* execution, uint16_t import_ordinal,
    const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_t* invocation = execution->invocation;
  const uint64_t target_bits =
      execution->linked_module->import_target_bits[import_ordinal];
  if (target_bits == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "optional import is unresolved");
  }
  const bool may_yield = iree_vm_program_target_may_yield(target_bits);
  IREE_RETURN_IF_ERROR(iree_vm_execution_check_child_yield(
      invocation->callback_context, may_yield));
  const iree_vm_program_t* program = invocation->process->program;
  const iree_vm_linked_module_t* target_module =
      &program
           ->linked_modules[iree_vm_program_target_module_ordinal(target_bits)];
  iree_vm_execution_publish_call_request(
      invocation, target_module,
      iree_vm_program_target_function_ordinal(target_bits), may_yield, call,
      out_outcome);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_invocation_call_function_ref(
    const iree_vm_module_execution_t* execution,
    iree_vm_function_ref_t function_ref,
    uint16_t expected_callable_type_ordinal, const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_vm_invocation_t* invocation = execution->invocation;
  const iree_vm_program_t* program = invocation->process->program;
  if (iree_vm_function_ref_is_null(function_ref)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot call a null function ref");
  }
  if (!iree_vm_program_function_ref_matches(program, function_ref,
                                            execution->linked_module,
                                            expected_callable_type_ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function ref does not match the call contract");
  }
  const bool may_yield =
      iree_vm_program_target_may_yield(function_ref.target_bits);
  IREE_RETURN_IF_ERROR(iree_vm_execution_check_child_yield(
      invocation->callback_context, may_yield));
  const iree_vm_linked_module_t* target_module =
      &program->linked_modules[iree_vm_program_target_module_ordinal(
          function_ref.target_bits)];
  iree_vm_execution_publish_call_request(
      invocation, target_module,
      iree_vm_program_target_function_ordinal(function_ref.target_bits),
      may_yield, call, out_outcome);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_local_function(
    const iree_vm_module_execution_t* execution,
    iree_vm_module_local_function_t local_function,
    iree_vm_function_ref_t* out_function_ref) {
  const iree_vm_invocation_t* invocation = execution->invocation;
  const iree_vm_program_t* program = invocation->process->program;
  uint32_t callable_mapping =
      program->callable_mappings[execution->linked_module->callable_base +
                                 local_function.callable_type_ordinal];
  const bool may_yield = iree_any_bit_set(
      local_function.flags, IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD);
  if (may_yield && !iree_vm_program_callable_may_yield(callable_mapping)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "local function behavior exceeds its callable contract");
  }
  callable_mapping &= ~IREE_VM_PROGRAM_CALLABLE_MAY_YIELD;
  if (may_yield) callable_mapping |= IREE_VM_PROGRAM_CALLABLE_MAY_YIELD;
  const uint16_t module_ordinal =
      (uint16_t)(execution->linked_module - program->linked_modules);
  const iree_vm_function_ref_t function_ref = {
      (uint64_t)(uintptr_t)program,
      iree_vm_program_pack_target_bits(
          module_ordinal, local_function.function_ordinal, callable_mapping),
  };
  *out_function_ref = function_ref;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_import(
    const iree_vm_module_execution_t* execution, uint16_t import_ordinal,
    iree_vm_function_ref_t* out_function_ref) {
  const iree_vm_program_t* program = execution->invocation->process->program;
  const uint64_t target_bits =
      execution->linked_module->import_target_bits[import_ordinal];
  const iree_vm_function_ref_t function_ref = {
      (uint64_t)(uintptr_t)program,
      target_bits,
  };
  *out_function_ref = target_bits ? function_ref : iree_vm_function_ref_null();
  return iree_ok_status();
}
