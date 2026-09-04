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

// Reusable fixed-capacity execution storage with one externally serialized
// driver. Initialize, deinitialize, start, resume, and synchronous driving
// never overlap. Only |iree_vm_invocation_request_cancel| may run concurrently
// with an active operation, subject to its retirement contract below.

// Initializes one reusable invocation in the complete caller-owned |storage|.
// The private header is aligned forward within the span and leading padding is
// charged against fixed capacity. Header-only storage is valid; the first
// operation requiring more space returns RESOURCE_EXHAUSTED. Storage unable to
// hold the aligned header is not touched. |out_invocation| is required and set
// to null before validation.
IREE_API_EXPORT iree_status_t iree_vm_invocation_initialize(
    iree_byte_span_t storage, iree_vm_invocation_t** out_invocation);

// Deinitializes an idle placement invocation without freeing caller storage.
// A null invocation is ignored.
IREE_API_EXPORT void iree_vm_invocation_deinitialize(
    iree_vm_invocation_t* invocation);

// Allocates one owner prefix followed by an exact immutable
// |storage_size|-byte invocation span. The allocator is retained by value for
// the matching free operation. |out_invocation| is required and set to null
// before allocation.
IREE_API_EXPORT iree_status_t iree_vm_invocation_allocate(
    iree_host_size_t storage_size, iree_allocator_t host_allocator,
    iree_vm_invocation_t** out_invocation);

// Frees an idle invocation returned by |iree_vm_invocation_allocate|. A null
// invocation is ignored.
IREE_API_EXPORT void iree_vm_invocation_free(iree_vm_invocation_t* invocation);

// Starts one asynchronously driven function call. Argument and result counts
// must exactly match |function|. Their spans must be well-formed and disjoint
// from each other, |out_outcome|, and the invocation storage. Failure of that
// structural boundary check consumes nothing and touches no caller storage.
// After it succeeds, every return consumes all argument carriers. A function
// declared as possibly yielding rejects borrowed ref arguments before module
// entry; an actual unexpected yield fails rather than suspending with the
// borrow.
//
// Result storage must contain no live owners on entry. OK+SUSPENDED leaves it
// untouched and publishes SUSPENDED. OK+COMPLETED initializes every result,
// transfers its owners to the caller, and publishes COMPLETED. Non-OK leaves
// results and |out_outcome| untouched. Rejection before a new operation begins
// preserves the prior invocation state; failure after one begins unwinds its
// owners and returns the invocation to idle. Unwind does not roll back process
// state or define whether the process remains meaningful after program failure.
IREE_API_EXPORT iree_status_t iree_vm_invocation_start(
    iree_vm_invocation_t* invocation, iree_vm_function_t function,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_vm_execution_outcome_t* out_outcome);

// Resumes or spuriously polls one suspended function call. |results| must have
// the exact count and may use different backing storage on every drive. It
// contains no live owners and is touched only by OK+COMPLETED. OK+SUSPENDED
// updates only |out_outcome|. Terminal failure leaves both outputs untouched
// and unwinds the operation to idle. Invalid input fails before state mutation
// or module entry and leaves the operation suspended.
IREE_API_EXPORT iree_status_t iree_vm_invocation_resume(
    iree_vm_invocation_t* invocation, iree_vm_variant_span_t results,
    iree_vm_execution_outcome_t* out_outcome);

// Atomically records the first valid cancellation reason for an active
// operation and invokes its level-triggered wake callback when nonnull.
// Invalid, repeated, and idle requests return false. The requesting thread
// never enters a module. The host joins every thread authorized to cancel and
// retires stale queued wakes before reclaiming callback data or reusing the
// invocation.
IREE_API_EXPORT bool iree_vm_invocation_request_cancel(
    iree_vm_invocation_t* invocation, iree_vm_cancel_reason_t reason);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_INVOCATION_H_
