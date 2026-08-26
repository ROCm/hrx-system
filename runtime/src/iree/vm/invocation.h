// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_INVOCATION_H_
#define IREE_VM_INVOCATION_H_

#include "iree/vm/module.h"
#include "iree/vm/variant.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Invocation storage has one externally serialized driver. Initialize,
// deinitialize, start, resume, and synchronous invoke never overlap. Only
// |iree_vm_invocation_request_cancel| may run concurrently with an active
// drive, subject to its retirement contract below.

// Initializes one reusable invocation in the complete caller-owned |storage|.
//
// The implementation aligns its private header forward within the span and
// charges leading padding against the fixed capacity. Storage containing only
// the aligned header is valid; the first operation requiring more space fails
// with RESOURCE_EXHAUSTED. A span unable to contain the header fails without
// touching its bytes. |out_invocation| is required and is set to null first.
IREE_API_EXPORT iree_status_t iree_vm_invocation_initialize(
    iree_byte_span_t storage, iree_vm_invocation_t** out_invocation);

// Deinitializes an idle placement invocation without freeing caller storage.
// A null invocation is ignored.
IREE_API_EXPORT void iree_vm_invocation_deinitialize(
    iree_vm_invocation_t* invocation);

// Coallocates one invocation with exact fixed |storage_size| bytes.
// |out_invocation| is required and is set to null first.
IREE_API_EXPORT iree_status_t iree_vm_invocation_allocate(
    iree_host_size_t storage_size, iree_allocator_t host_allocator,
    iree_vm_invocation_t** out_invocation);

// Frees an idle invocation returned by |iree_vm_invocation_allocate|. A null
// invocation is ignored.
IREE_API_EXPORT void iree_vm_invocation_free(iree_vm_invocation_t* invocation);

// Synchronously invokes |function| through the asynchronous execution core.
//
// Both spans require exact counts and are disjoint from each other, the
// invocation storage, and internal outcome storage. After that boundary
// preflight, every argument is consumed even when execution fails. Result
// storage must contain no live variant owners on entry. Non-OK leaves every
// result byte untouched; terminal OK initializes every result and transfers
// its owners to the caller. The wrapper may deadlock when the host cannot make
// provider progress while this thread blocks.
IREE_API_EXPORT iree_status_t iree_vm_invoke(iree_vm_invocation_t* invocation,
                                             iree_vm_function_t function,
                                             iree_vm_variant_span_t arguments,
                                             iree_vm_variant_span_t results);

// Starts one asynchronously driven invocation operation.
//
// Both spans require exact counts and are disjoint from each other,
// |out_outcome|, and invocation storage. After that boundary preflight,
// arguments are consumed on every return. Result storage must contain no live
// variant owners on entry. OK+SUSPENDED leaves results untouched.
// OK+COMPLETED initializes every result. Non-OK leaves results and
// |out_outcome| untouched. Rejection before a new operation begins preserves
// the current driver state; failure after one begins unwinds its actual owners
// and returns the invocation to idle.
IREE_API_EXPORT iree_status_t iree_vm_invocation_start(
    iree_vm_invocation_t* invocation, iree_vm_function_t function,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_vm_execution_outcome_t* out_outcome);

// Resumes or spuriously polls one suspended invocation. |results| may use
// different exact backing storage on every drive. Invalid input fails before
// mutation or provider entry and leaves the invocation suspended. Result
// storage must contain no live variant owners on entry. Suspension and non-OK
// leave result bytes untouched; terminal OK initializes all results.
IREE_API_EXPORT iree_status_t iree_vm_invocation_resume(
    iree_vm_invocation_t* invocation, iree_vm_variant_span_t results,
    iree_vm_execution_outcome_t* out_outcome);

// Atomically records the first valid cancellation reason for an active
// operation and invokes its level-triggered wake callback when nonnull.
//
// The requesting thread never enters a module. Invalid, repeated, and idle
// requests return false. The host joins every authorized cancellation call and
// retires stale queued wakes before reclaiming callback context or reusing the
// invocation.
IREE_API_EXPORT bool iree_vm_invocation_request_cancel(
    iree_vm_invocation_t* invocation, iree_vm_cancel_reason_t reason);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_INVOCATION_H_
