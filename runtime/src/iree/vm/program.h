// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_PROGRAM_H_
#define IREE_VM_PROGRAM_H_

#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Borrowed module input storage. Library input order has no semantic effect.
typedef struct iree_vm_module_span_t {
  // Contiguous module pointers borrowed for the call.
  iree_vm_module_t* const* data;
  // Number of module pointers in |data|.
  iree_host_size_t count;
} iree_vm_module_span_t;

// Complete composition input for one immutable program.
typedef struct iree_vm_program_modules_t {
  // Module defining the host contract and optional automatic initializer.
  iree_vm_module_t* executable;
  // Exact library modules available to satisfy imports.
  iree_vm_module_span_t libraries;
} iree_vm_program_modules_t;

// Returns a borrowed module span over |data|.
static inline iree_vm_module_span_t iree_vm_module_span_from_ptr(
    iree_vm_module_t* const* data, iree_host_size_t count) {
  const iree_vm_module_span_t span = {data, count};
  return span;
}

// Returns the canonical empty module span.
static inline iree_vm_module_span_t iree_vm_module_span_empty(void) {
  const iree_vm_module_span_t span = {NULL, 0};
  return span;
}

#define iree_vm_module_span_from_array(array) \
  iree_vm_module_span_from_ptr((array), IREE_ARRAYSIZE(array))

// Exactly links and failure-atomically publishes one immutable program.
//
// Every input module already contains canonical resolved ref types. Program
// creation retains and name-sorts the modules, interns structural callable
// contracts, assigns opaque process-storage offsets, resolves all imports, and
// selects only the executable module's exact `initialize` export. Libraries
// have no implicit initialization. The operation performs no environment
// lookup or guest execution and allocates exactly one program slab.
//
// |out_program| is required and is set to null before any other work. Failure
// releases every temporary module owner and leaves no partially linked state.
IREE_API_EXPORT iree_status_t iree_vm_program_create(
    iree_vm_program_modules_t modules, iree_allocator_t host_allocator,
    iree_vm_program_t** out_program);

// Retains |program| for the caller. A null program is ignored.
IREE_API_EXPORT void iree_vm_program_retain(iree_vm_program_t* program);

// Releases |program| from the caller. A null program is ignored.
IREE_API_EXPORT void iree_vm_program_release(iree_vm_program_t* program);

// Resolves one module-bound export into a program-bound function value.
//
// The export's module must participate in |program|. The returned function
// reference borrows |program| and remains valid only while the program is live.
// Failure leaves |out_function_ref| untouched.
IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_export(
    const iree_vm_program_t* program, iree_vm_export_t export_value,
    iree_vm_function_ref_t* out_function_ref);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_PROGRAM_H_
