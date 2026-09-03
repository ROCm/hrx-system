// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_INVOCATION_H_
#define IREE_VM_INVOCATION_H_

#include "iree/vm/execution.h"
#include "iree/vm/variant.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Invocation storage has one externally serialized driver. Initialize,
// deinitialize, start, and resume never overlap. Only cancellation may run
// concurrently with an active operation under its retirement contract.

// Initializes one reusable invocation in the complete caller-owned |storage|.
// Leading alignment padding and the private header consume fixed capacity.
// |out_invocation| is set to null before validation.
IREE_API_EXPORT iree_status_t iree_vm_invocation_initialize(
    iree_byte_span_t storage, iree_vm_invocation_t** out_invocation);

// Deinitializes an idle placement invocation without freeing caller storage.
// A null invocation is ignored.
IREE_API_EXPORT void iree_vm_invocation_deinitialize(
    iree_vm_invocation_t* invocation);

// Allocates one reusable invocation with exact immutable |storage_size|. The
// allocator is retained by value for the matching free operation.
IREE_API_EXPORT iree_status_t iree_vm_invocation_allocate(
    iree_host_size_t storage_size, iree_allocator_t host_allocator,
    iree_vm_invocation_t** out_invocation);

// Frees an idle invocation returned by |iree_vm_invocation_allocate|. A null
// invocation is ignored.
IREE_API_EXPORT void iree_vm_invocation_free(iree_vm_invocation_t* invocation);

// Starts one asynchronously driven function call. Well-formed disjoint
// arguments are consumed on every semantic return after storage preflight.
// Result storage contains no live owners on entry and is touched only on
// terminal OK. A non-OK return leaves |out_outcome| untouched and returns a
// begun operation to idle.
IREE_API_EXPORT iree_status_t iree_vm_invocation_start(
    iree_vm_invocation_t* invocation, iree_vm_function_t function,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_vm_execution_outcome_t* out_outcome);

// Resumes or spuriously polls one suspended function call. The caller may
// provide different result storage on every drive; it contains no live owners
// and is touched only on terminal OK. Invalid input fails before mutation or
// provider entry and leaves the operation suspended.
IREE_API_EXPORT iree_status_t iree_vm_invocation_resume(
    iree_vm_invocation_t* invocation, iree_vm_variant_span_t results,
    iree_vm_execution_outcome_t* out_outcome);

// Atomically records the first valid cancellation reason for an active
// operation and invokes its level-triggered wake callback when nonnull. The
// requesting thread never enters a provider. The host joins cancel callers and
// retires stale wakes before invocation reuse.
IREE_API_EXPORT bool iree_vm_invocation_request_cancel(
    iree_vm_invocation_t* invocation, iree_vm_cancel_reason_t reason);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_INVOCATION_H_
