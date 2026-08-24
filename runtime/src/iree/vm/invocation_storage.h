// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_INVOCATION_STORAGE_H_
#define IREE_VM_INVOCATION_STORAGE_H_

#include "iree/vm/invocation.h"
#include "iree/vm/process_storage.h"

// Private invocation representation shared with process construction and
// built-in module providers. No declaration in this file is part of the public
// ABI.

typedef uint8_t iree_vm_invocation_state_t;
enum iree_vm_invocation_state_e {
  IREE_VM_INVOCATION_STATE_IDLE = 0u,
  IREE_VM_INVOCATION_STATE_RUNNING = 1u,
  IREE_VM_INVOCATION_STATE_SUSPENDED = 2u,
};

typedef uint8_t iree_vm_invocation_operation_t;
enum iree_vm_invocation_operation_e {
  IREE_VM_INVOCATION_OPERATION_NONE = 0u,
  IREE_VM_INVOCATION_OPERATION_CALL = 1u,
  IREE_VM_INVOCATION_OPERATION_PROCESS_CREATE = 2u,
};

typedef struct iree_vm_root_call_t {
  // Direct canonical callable plan retained by the borrowed process program.
  const iree_vm_program_callable_t* callable;
  // Canonical physical root call banks.
  iree_vm_call_packet_t packet;
  // First root staging byte rewound at terminal cleanup.
  uint8_t* allocation_begin;
} iree_vm_root_call_t;

struct iree_vm_frame_t {
  // Stack cursor restored when this frame is removed.
  uint8_t* allocation_begin;
  // Aligned module-owned payload.
  void* storage;
  // Exact parent frame at the pre-callee boundary.
  iree_vm_frame_t* parent;
  // Sole linked-module and resume-routing identity.
  const iree_vm_linked_module_t* linked_module;
  // No-fail cleanup for the module-owned payload.
  iree_vm_frame_cleanup_fn_t cleanup;
  // Module-local function ordinal.
  uint16_t function_ordinal;
};

struct iree_vm_invocation_t {
  // First byte in the complete caller-provided storage span.
  uint8_t* storage_begin;
  // One-past-last byte in the complete storage span.
  uint8_t* storage_end;
  // First byte available after the private invocation header.
  uint8_t* stack_base;
  // Current composite frame-stack allocation cursor.
  uint8_t* stack_cursor;
  // Current top module-owned frame.
  iree_vm_frame_t* top_frame;
  // Process borrowed by calls or owned while construction is unpublished.
  iree_vm_process_t* process;
  // Linked module whose callback is currently executing.
  const iree_vm_linked_module_t* executing_linked_module;
  // Complete root call transaction.
  iree_vm_root_call_t root_call;
  // Original allocation base for allocated invocations, otherwise null.
  void* allocation_base;
  // Allocator owning |allocation_base| when nonnull.
  iree_allocator_t host_allocator;
  // Wake callback stable for the complete active operation.
  iree_vm_invocation_wake_callback_t wake_callback;
  // Root target word.
  uint64_t root_target_bits;
  // IDLE sentinel, active NONE, or the first cancellation reason.
  iree_atomic_int32_t cancel_reason;
  // Module-local function whose callback is currently executing.
  uint16_t executing_function_ordinal;
  // Current exclusive-driver state.
  iree_vm_invocation_state_t state;
  // Current root operation kind.
  iree_vm_invocation_operation_t operation;
  // True when root staging contains a nonnull external borrowed ref.
  bool has_external_borrowed_arguments;
};

// Aligns one invocation-storage address without truncation.
static inline bool iree_vm_invocation_align_address(uintptr_t address,
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

// Reserves transient callback-local storage above the durable frame stack.
// The caller must rewind the returned checkpoint before leaving the callback
// and must not suspend while the reservation is live.
static inline iree_status_t iree_vm_invocation_stack_reserve(
    iree_vm_invocation_t* invocation, iree_host_size_t storage_size,
    iree_host_size_t storage_alignment, uint8_t** out_checkpoint,
    uint8_t** out_storage) {
  uintptr_t storage_address = 0;
  if (!iree_vm_invocation_align_address((uintptr_t)invocation->stack_cursor,
                                        storage_alignment, &storage_address) ||
      storage_address > (uintptr_t)invocation->storage_end ||
      storage_size > (uintptr_t)invocation->storage_end - storage_address) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "invocation transient storage exceeds capacity");
  }
  *out_checkpoint = invocation->stack_cursor;
  *out_storage = (uint8_t*)storage_address;
  invocation->stack_cursor = (uint8_t*)storage_address + storage_size;
  return iree_ok_status();
}

// Rewinds one live transient reservation to its exact allocation checkpoint.
static inline void iree_vm_invocation_stack_rewind(
    iree_vm_invocation_t* invocation, uint8_t* checkpoint) {
  invocation->stack_cursor = checkpoint;
}

// Private atomic sentinel distinguishing idle from active uncancelled state.
#define IREE_VM_INVOCATION_CANCEL_REASON_IDLE INT32_MAX

// Returns true when |invocation| can accept a new operation.
bool iree_vm_invocation_is_idle(const iree_vm_invocation_t* invocation);

// Stages and consumes one spatially validated root call before provider entry.
iree_status_t iree_vm_invocation_prepare_root(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_program_callable_t* callable,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_invocation_wake_callback_t wake_callback);

// Validates one drive boundary without mutating invocation or caller storage.
iree_status_t iree_vm_invocation_validate_boundary(
    const iree_vm_invocation_t* invocation, iree_vm_variant_span_t arguments,
    iree_vm_variant_span_t results, iree_byte_span_t outcome_storage);

// Drives the validated root target from an already prepared operation.
iree_status_t iree_vm_invocation_drive_start(
    iree_vm_invocation_t* invocation, iree_vm_execution_outcome_t* out_outcome);

// Drives suspended frames until the root completes or suspends again.
iree_status_t iree_vm_invocation_drive_resume(
    iree_vm_invocation_t* invocation, iree_vm_execution_outcome_t* out_outcome);

// Enters one already resolved target without repeating the public module-call
// boundary checks. Built-in providers may use this only after proving the
// target identity, callable contract, and current invocation state.
iree_status_t iree_vm_invocation_dispatch_start(
    iree_vm_invocation_t* invocation,
    const iree_vm_linked_module_t* linked_module, uint16_t function_ordinal,
    bool may_yield, const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome);

// Validates every root result without touching caller storage.
iree_status_t iree_vm_invocation_validate_root_results(
    const iree_vm_invocation_t* invocation);

// Atomically closes an uncancelled operation. Failure reports the winning
// cancellation reason without changing invocation-owned storage.
bool iree_vm_invocation_try_claim_completion(
    iree_vm_invocation_t* invocation,
    iree_vm_cancel_reason_t* out_cancel_reason);

// Releases root ownership and marks an atomically closed operation idle after
// every frame has been removed. Other private fields are dead while idle and
// are overwritten before the next provider entry.
static inline void iree_vm_invocation_finish(iree_vm_invocation_t* invocation) {
  const uint16_t argument_ref_count =
      invocation->root_call.callable->argument_counts.ref_count;
  for (uint16_t i = 0; i < argument_ref_count; ++i) {
    iree_vm_ref_t* ref =
        i < 16 ? &invocation->root_call.packet.ref_arguments.direct[i]
               : &invocation->root_call.packet.ref_arguments.overflow[i - 16];
    iree_vm_ref_reset(ref);
  }
  const uint16_t result_ref_count =
      invocation->root_call.callable->result_counts.ref_count;
  for (uint16_t i = 0; i < result_ref_count; ++i) {
    iree_vm_ref_t* ref =
        i < 16 ? &invocation->root_call.packet.ref_results.direct[i]
               : &invocation->root_call.packet.ref_results.overflow[i - 16];
    iree_vm_ref_reset(ref);
  }
  invocation->state = IREE_VM_INVOCATION_STATE_IDLE;
  invocation->operation = IREE_VM_INVOCATION_OPERATION_NONE;
}

// Atomically closes, unwinds, and resets one failed operation.
void iree_vm_invocation_abort(iree_vm_invocation_t* invocation);

// Converts one winning cancellation reason into its terminal status.
iree_status_t iree_vm_invocation_cancel_status(
    iree_vm_cancel_reason_t cancel_reason);

#endif  // IREE_VM_INVOCATION_STORAGE_H_
