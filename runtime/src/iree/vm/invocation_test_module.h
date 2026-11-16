// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_INVOCATION_TEST_MODULE_H_
#define IREE_VM_INVOCATION_TEST_MODULE_H_

#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Mutable observations from one exclusively driven test module.
typedef struct iree_vm_invocation_test_counters_t {
  // Number of function-start callbacks.
  int start_count;
  // Number of function-resume callbacks.
  int resume_count;
  // Number of frame cleanup callbacks.
  int cleanup_count;
  // Number of final module releases.
  int destroy_count;
  // Descendant entries with no ref packet but a live external root borrow.
  int unforwarded_borrow_count;
} iree_vm_invocation_test_counters_t;

// Dispatch routes exercised below a root invocation's borrow scope.
typedef enum iree_vm_invocation_test_call_route_e {
  IREE_VM_INVOCATION_TEST_CALL_LOCAL = 0,
  IREE_VM_INVOCATION_TEST_CALL_IMPORT = 1,
  IREE_VM_INVOCATION_TEST_CALL_INDIRECT_LOCAL = 2,
  IREE_VM_INVOCATION_TEST_CALL_INDIRECT_IMPORT = 3,
} iree_vm_invocation_test_call_route_t;

// Leaf behaviors used to distinguish admission from actual suspension.
typedef enum iree_vm_invocation_test_child_behavior_e {
  IREE_VM_INVOCATION_TEST_CHILD_COMPLETE = 0,
  IREE_VM_INVOCATION_TEST_CHILD_DECLARED_YIELD = 1,
  IREE_VM_INVOCATION_TEST_CHILD_UNDECLARED_YIELD = 2,
} iree_vm_invocation_test_child_behavior_t;

// Caller-owned ref object used to exercise invocation ownership transfers.
typedef struct iree_vm_invocation_test_object_t {
  // Required offset-zero VM ref prefix.
  iree_vm_ref_object_t ref_object;
  // Counter incremented when the final owner is released.
  int* destroy_count;
} iree_vm_invocation_test_object_t;

// Fixed stack-backed module used only by invocation tests and fuzzing.
typedef struct iree_vm_invocation_test_module_t {
  // Generic module base published at offset zero.
  iree_vm_module_t base;
  // Immutable descriptor published through |base|.
  iree_vm_module_descriptor_t descriptor;
  // Test-owned callback observations.
  iree_vm_invocation_test_counters_t* counters;
} iree_vm_invocation_test_module_t;

// Initializes one caller-owned object with one owner.
void iree_vm_invocation_test_object_initialize(
    int* destroy_count, iree_vm_invocation_test_object_t* out_object);

// Returns the canonical primary object type used by the test module.
iree_vm_ref_type_t iree_vm_invocation_test_object_type(void);

// Initializes one fixed module with one owner.
iree_status_t iree_vm_invocation_test_module_initialize(
    iree_vm_invocation_test_counters_t* counters,
    iree_vm_invocation_test_module_t* out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_INVOCATION_TEST_MODULE_H_
