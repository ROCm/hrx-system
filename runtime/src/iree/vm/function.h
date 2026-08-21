// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_FUNCTION_H_
#define IREE_VM_FUNCTION_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_process_t iree_vm_process_t;
typedef struct iree_vm_program_t iree_vm_program_t;

// Non-owning reference to a function linked into a program. Non-null values are
// produced by the VM or trusted module implementations. The referenced program
// must remain live for every use of the value.
typedef struct iree_vm_function_ref_t {
  // Borrowed iree_vm_program_t* represented as integer bits.
  uint64_t program_bits;
  // Opaque VM-produced target word with bits 0 and 1 clear.
  uint64_t target_bits;
} iree_vm_function_ref_t;
static_assert(sizeof(iree_vm_function_ref_t) == 16,
              "VM function refs must remain 16 bytes");
static_assert(iree_alignof(iree_vm_function_ref_t) == iree_alignof(uint64_t),
              "VM function refs must retain native 64-bit alignment");
static_assert(offsetof(iree_vm_function_ref_t, program_bits) == 0,
              "VM function ref owner lane must be first");
static_assert(offsetof(iree_vm_function_ref_t, target_bits) == 8,
              "VM function ref target lane must be second");

// Non-owning host invocation target bound to a process. Non-null values are
// produced by the VM. The referenced process must remain live for every use,
// including asynchronous invocation.
typedef struct iree_vm_function_t {
  // Borrowed iree_vm_process_t* represented as integer bits.
  uint64_t process_bits;
  // Opaque target word copied from iree_vm_function_ref_t.
  uint64_t target_bits;
} iree_vm_function_t;
static_assert(sizeof(iree_vm_function_t) == 16,
              "VM functions must remain 16 bytes");
static_assert(iree_alignof(iree_vm_function_t) == iree_alignof(uint64_t),
              "VM functions must retain native 64-bit alignment");
static_assert(offsetof(iree_vm_function_t, process_bits) == 0,
              "VM function owner lane must be first");
static_assert(offsetof(iree_vm_function_t, target_bits) == 8,
              "VM function target lane must be second");
static_assert(sizeof(uintptr_t) <= sizeof(uint64_t),
              "VM functions require pointers no wider than 64 bits");

// Returns the canonical all-zero program-bound function value.
static inline iree_vm_function_ref_t iree_vm_function_ref_null(void) {
  iree_vm_function_ref_t function_ref = {0, 0};
  return function_ref;
}

// Returns the canonical all-zero process-bound function value.
static inline iree_vm_function_t iree_vm_function_null(void) {
  iree_vm_function_t function = {0, 0};
  return function;
}

// Returns true when |function_ref| is canonical null.
static inline bool iree_vm_function_ref_is_null(
    iree_vm_function_ref_t function_ref) {
  return function_ref.program_bits == 0 && function_ref.target_bits == 0;
}

// Returns true when |function| is canonical null.
static inline bool iree_vm_function_is_null(iree_vm_function_t function) {
  return function.process_bits == 0 && function.target_bits == 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_FUNCTION_H_
