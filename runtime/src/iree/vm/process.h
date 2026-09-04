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

// A process contains one mutable program-state instance and no execution lock.
// The host serializes uses of one process; independent processes may execute
// freely on different threads. Process-bound functions borrow their process,
// which must remain live through synchronous completion or the complete
// suspended asynchronous operation.

// Retains |process| for the caller. A null process is ignored.
IREE_API_EXPORT void iree_vm_process_retain(iree_vm_process_t* process);

// Releases |process| and reverse-detaches every module on final release. A null
// process is ignored.
IREE_API_EXPORT void iree_vm_process_release(iree_vm_process_t* process);

// Binds one reflected export to a process of the containing program. The
// result borrows |process| and failure leaves |out_function| untouched.
IREE_API_EXPORT iree_status_t iree_vm_function_from_export(
    iree_vm_process_t* process, iree_vm_export_t export_value,
    iree_vm_function_t* out_function);

// Binds one nonnull program-bound function to a process of the same program.
// Failure leaves |out_function| untouched.
IREE_API_EXPORT iree_status_t iree_vm_function_from_function_ref(
    iree_vm_process_t* process, iree_vm_function_ref_t function_ref,
    iree_vm_function_t* out_function);

// Finds and binds one exact exported function by module and export name.
// Failure leaves |out_function| untouched.
IREE_API_EXPORT iree_status_t iree_vm_process_lookup_function(
    iree_vm_process_t* process, iree_string_view_t module_name,
    iree_string_view_t export_name, iree_vm_function_t* out_function);

// Composite successful asynchronous process-construction outcome.
typedef struct iree_vm_process_create_outcome_t {
  // COMPLETED or SUSPENDED on an OK driving return.
  iree_vm_execution_outcome_t execution_outcome;
  // Owned process on COMPLETED and null on SUSPENDED.
  iree_vm_process_t* process;
} iree_vm_process_create_outcome_t;

// Starts process construction and the executable's optional initializer.
// Argument count and types must exactly match the initializer, or be empty when
// none exists. Structural boundary failure consumes nothing. After that check,
// every return consumes all arguments. OK+SUSPENDED publishes a null process
// and leaves the unpublished process owned only by |invocation|. OK+COMPLETED
// publishes the sole process owner. Terminal non-OK leaves |out_outcome|
// untouched, releases the unpublished process, and returns the invocation idle.
IREE_API_EXPORT iree_status_t iree_vm_process_create_start(
    iree_vm_program_t* program, iree_vm_invocation_t* invocation,
    iree_vm_variant_span_t arguments,
    iree_vm_invocation_wake_callback_t wake_callback,
    iree_allocator_t host_allocator,
    iree_vm_process_create_outcome_t* out_outcome);

// Resumes suspended process construction. Boundary rejection leaves the output,
// unpublished process, and invocation unchanged and enters no module. Terminal
// failure leaves the output untouched, unwinds state, and returns the
// invocation idle. Successful outcomes follow |iree_vm_process_create_start|.
IREE_API_EXPORT iree_status_t
iree_vm_process_create_resume(iree_vm_invocation_t* invocation,
                              iree_vm_process_create_outcome_t* out_outcome);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_PROCESS_H_
