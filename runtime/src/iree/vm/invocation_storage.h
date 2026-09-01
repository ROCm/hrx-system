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
  // Program-linked callable ABI borrowed for the active root operation.
  const iree_vm_program_callable_abi_t* callable_abi;
  // Complete packed root target bits.
  uint64_t target_bits;
} iree_vm_root_call_t;

// Validated facts carried from root preflight into its no-fail commit.
typedef struct iree_vm_root_preflight_t {
  // Exact root-bank bytes reserved before the first provider frame.
  iree_host_size_t root_storage_size;
  // True when staged arguments will contain a nonnull external borrowed ref.
  bool has_external_borrowed_arguments;
} iree_vm_root_preflight_t;

typedef struct iree_vm_callback_context_t iree_vm_callback_context_t;

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
  // True when the framed function is permitted to yield.
  bool may_yield;
};

struct iree_vm_invocation_t {
  // First byte in the complete caller-provided storage span.
  uint8_t* storage_begin;
  // One-past-last byte in the complete storage span.
  uint8_t* storage_end;
  // Current composite frame-stack allocation cursor.
  uint8_t* stack_cursor;
  // Current top module-owned frame.
  iree_vm_frame_t* top_frame;
  // Process borrowed by calls or owned while construction is unpublished.
  iree_vm_process_t* process;
  // Native-stack callback context, or null outside a module callback.
  const iree_vm_callback_context_t* callback_context;
  // Active root call description.
  iree_vm_root_call_t root_call;
  // Wake callback stable for the complete active operation.
  iree_vm_invocation_wake_callback_t wake_callback;
  // IDLE sentinel, active NONE, or the first cancellation reason.
  iree_atomic_int32_t cancel_reason;
  // Current exclusive-driver state.
  iree_vm_invocation_state_t state;
  // Current root operation kind.
  iree_vm_invocation_operation_t operation;
  // True when root staging contains a nonnull external borrowed ref.
  bool has_external_borrowed_arguments;
  // True when storage was returned by |iree_vm_invocation_allocate|.
  bool is_allocated;
};

static_assert(sizeof(void*) != 8 || sizeof(iree_vm_root_call_t) == 16,
              "64-bit root call descriptions must remain 16 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_frame_t) == 48,
              "64-bit generic frames must remain 48 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_invocation_t) == 88,
              "64-bit invocation headers must remain 88 bytes");

// Returns the max-aligned base of invocation-owned root call banks.
static inline uint8_t* iree_vm_invocation_stack_base(
    iree_vm_invocation_t* invocation) {
  return (uint8_t*)invocation + iree_sizeof_struct(*invocation);
}

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

// Validates one root call before any process allocation or provider entry.
// Semantic failure consumes |arguments|; success leaves them unchanged.
iree_status_t iree_vm_invocation_preflight_root(
    iree_vm_invocation_t* invocation, const iree_vm_program_t* program,
    uint64_t target_bits, const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_root_preflight_t* out_preflight);

// Stages and consumes a successfully preflighted root call and returns its
// derived physical packet. The packet may be discarded across suspension and
// reconstructed from the active root description.
iree_vm_call_packet_t iree_vm_invocation_commit_root(
    iree_vm_invocation_t* invocation, iree_vm_invocation_operation_t operation,
    iree_vm_process_t* process, uint64_t target_bits,
    const iree_vm_program_callable_abi_t* callable_abi,
    iree_vm_variant_span_t arguments,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_vm_root_preflight_t preflight);

// Validates one drive boundary without mutating invocation or caller storage.
iree_status_t iree_vm_invocation_validate_boundary(
    const iree_vm_invocation_t* invocation, iree_vm_variant_span_t arguments,
    iree_vm_variant_span_t results, iree_byte_span_t outcome_storage);

// Drives the validated root target from an already prepared operation.
iree_status_t iree_vm_invocation_drive_start(
    iree_vm_invocation_t* invocation, const iree_vm_call_packet_t* root_packet,
    iree_vm_execution_outcome_t* out_outcome);

// Drives suspended frames until the root completes or suspends again.
iree_status_t iree_vm_invocation_drive_resume(
    iree_vm_invocation_t* invocation, iree_vm_execution_outcome_t* out_outcome);

// Requests one already resolved child call without entering another provider.
// Built-in providers may use this only after proving the target identity and
// callable contract. On success |out_outcome| is SUSPENDED so the current
// callback returns control to the iterative invocation driver.
iree_status_t iree_vm_invocation_request_call(
    iree_vm_invocation_t* invocation,
    const iree_vm_linked_module_t* linked_module, uint16_t function_ordinal,
    bool may_yield, const iree_vm_call_packet_t* call,
    iree_vm_execution_outcome_t* out_outcome);

// Atomically closes an uncancelled operation. Failure reports the winning
// cancellation reason without changing invocation-owned storage.
bool iree_vm_invocation_try_claim_completion(
    iree_vm_invocation_t* invocation,
    iree_vm_cancel_reason_t* out_cancel_reason);

// Releases root ownership and marks an atomically closed operation idle after
// every frame has been removed. Other private fields are dead while idle and
// are overwritten before the next provider entry.
void iree_vm_invocation_finish(iree_vm_invocation_t* invocation);

// Atomically closes, unwinds, and resets one failed operation.
void iree_vm_invocation_abort(iree_vm_invocation_t* invocation);

// Converts one winning cancellation reason into its terminal status.
iree_status_t iree_vm_invocation_cancel_status(
    iree_vm_cancel_reason_t cancel_reason);

#endif  // IREE_VM_INVOCATION_STORAGE_H_
