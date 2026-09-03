// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SYNC_H_
#define IREE_VM_SYNC_H_

#include "iree/vm/process.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Synchronously invokes |function| through the asynchronous execution core.
// Both spans require exact counts and are disjoint. Well-formed arguments are
// consumed on every return. Non-OK leaves every result byte untouched, while
// OK initializes every result and transfers its owners to the caller. The
// caller has exclusive access to |invocation| for the complete call and must
// not request cancellation concurrently. This method may deadlock when the
// host cannot make provider progress while the calling thread blocks.
IREE_API_EXPORT iree_status_t iree_vm_invoke(iree_vm_invocation_t* invocation,
                                             iree_vm_function_t function,
                                             iree_vm_variant_span_t arguments,
                                             iree_vm_variant_span_t results);

// Synchronously constructs and initializes one process through the
// asynchronous execution core. |arguments| must exactly match the optional
// executable initializer and is consumed on every well-formed return.
// |out_process| is required and is set to null before any other work. The
// caller has exclusive access to |invocation| for the complete call and must
// not request cancellation concurrently. This method may deadlock when the
// host cannot make provider progress while the calling thread blocks.
IREE_API_EXPORT iree_status_t iree_vm_process_create(
    iree_vm_program_t* program, iree_vm_invocation_t* invocation,
    iree_vm_variant_span_t arguments, iree_allocator_t host_allocator,
    iree_vm_process_t** out_process);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_SYNC_H_
