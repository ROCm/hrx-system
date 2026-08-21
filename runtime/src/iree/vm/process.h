// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_PROCESS_H_
#define IREE_VM_PROCESS_H_

#include "iree/vm/invocation.h"
#include "iree/vm/program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A process contains one mutable program-state instance and has no internal
// execution lock. The host externally serializes invocations using a process.
// Independent processes from the same immutable program may execute freely on
// different threads.

// Retains |process| for the caller. A null process is ignored.
IREE_API_EXPORT void iree_vm_process_retain(iree_vm_process_t* process);

// Releases |process| from the caller. Final release reverse-detaches every
// module state slice and releases the immutable program. Null is ignored.
IREE_API_EXPORT void iree_vm_process_release(iree_vm_process_t* process);

// Binds one reflected export to a process of the containing program. Failure
// leaves |out_function| untouched.
IREE_API_EXPORT iree_status_t iree_vm_function_from_export(
    iree_vm_process_t* process, iree_vm_export_t export_value,
    iree_vm_function_t* out_function);

// Binds one program-bound function value to a process of the same program. A
// null or foreign-program value fails and leaves |out_function| untouched.
IREE_API_EXPORT iree_status_t iree_vm_function_from_function_ref(
    iree_vm_process_t* process, iree_vm_function_ref_t function_ref,
    iree_vm_function_t* out_function);

// Finds and binds one exact exported function by module and export name.
// Failure leaves |out_function| untouched.
IREE_API_EXPORT iree_status_t iree_vm_process_lookup_function(
    iree_vm_process_t* process, iree_string_view_t module_name,
    iree_string_view_t export_name, iree_vm_function_t* out_function);

// Composite asynchronous process-construction outcome.
typedef struct iree_vm_process_create_outcome_t {
  // COMPLETED or SUSPENDED on an OK driving return.
  iree_vm_execution_outcome_t execution_outcome;
  // Owned process on COMPLETED; null on SUSPENDED.
  iree_vm_process_t* process;
} iree_vm_process_create_outcome_t;

// Synchronously constructs and initializes one process through the same
// asynchronous core. |arguments| must exactly match executable.initialize and
// is consumed on every return. An absent initializer is exact () -> ().
// |out_process| is required and is set to null first.
IREE_API_EXPORT iree_status_t iree_vm_process_create(
    iree_vm_program_t* program, iree_vm_invocation_t* invocation,
    iree_vm_variant_span_t arguments, iree_allocator_t host_allocator,
    iree_vm_process_t** out_process);

// Starts asynchronous process construction. Arguments and |out_outcome| must
// be disjoint from each other and invocation storage. After that boundary
// preflight, arguments are consumed on every return. OK+SUSPENDED publishes a
// null process and leaves the unpublished process owned by |invocation|.
// OK+COMPLETED transfers one process owner. Non-OK leaves |out_outcome|
// untouched and returns |invocation| to idle.
IREE_API_EXPORT iree_status_t iree_vm_process_create_start(
    iree_vm_program_t* program, iree_vm_invocation_t* invocation,
    iree_vm_variant_span_t arguments,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_allocator_t host_allocator,
    iree_vm_process_create_outcome_t* out_outcome);

// Resumes asynchronous process construction. Spurious polling is valid. A
// rejected boundary leaves the suspended invocation and |out_outcome|
// untouched and calls no module.
IREE_API_EXPORT iree_status_t
iree_vm_process_create_resume(iree_vm_invocation_t* invocation,
                              iree_vm_process_create_outcome_t* out_outcome);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_PROCESS_H_
