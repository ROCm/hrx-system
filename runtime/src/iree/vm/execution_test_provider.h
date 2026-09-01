// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_EXECUTION_TEST_PROVIDER_H_
#define IREE_VM_EXECUTION_TEST_PROVIDER_H_

#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint32_t iree_vm_execution_test_module_kind_t;
enum iree_vm_execution_test_module_kind_e {
  IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION = 0u,
  IREE_VM_EXECUTION_TEST_MODULE_KIND_MATH = 1u,
};

enum iree_vm_execution_test_flag_bits_e {
  IREE_VM_EXECUTION_TEST_FLAG_NONE = 0u,
  IREE_VM_EXECUTION_TEST_FLAG_FAIL_ATTACH = 1u << 0,
  IREE_VM_EXECUTION_TEST_FLAG_FAIL_SEAL = 1u << 1,
  IREE_VM_EXECUTION_TEST_FLAG_FAIL_FUNCTION = 1u << 2,
  IREE_VM_EXECUTION_TEST_FLAG_RETURN_WRONG_REF_TYPE = 1u << 3,
};
typedef uint32_t iree_vm_execution_test_flags_t;

// Mutable test observations. Tests provide exclusive access.
typedef struct iree_vm_execution_test_counters_t {
  // Number of process-state attach callbacks.
  int attach_count;
  // Number of failing attach callbacks that self-cleaned.
  int attach_self_cleanup_count;
  // Number of process-state seal callbacks.
  int seal_count;
  // Number of process-state detach callbacks.
  int detach_count;
  // Number of function start callbacks.
  int function_start_count;
  // Number of function resume callbacks.
  int function_resume_count;
  // Number of frame cleanups.
  int frame_cleanup_count;
  // Number of final module releases.
  int module_destruction_count;
} iree_vm_execution_test_counters_t;

// Native provider behavior selected at module construction.
typedef struct iree_vm_execution_test_options_t {
  // Scripted module failures.
  iree_vm_execution_test_flags_t flags;
  // Number of suspensions before application initialization completes.
  uint32_t initializer_suspension_count;
} iree_vm_execution_test_options_t;

// Ref-counted object used to verify root ownership transactions.
typedef struct iree_vm_execution_test_object_t {
  // Required offset-zero VM-visible ownership prefix.
  iree_vm_ref_object_t ref_object;
  // Incremented by the final-release callback.
  int* destruction_count;
} iree_vm_execution_test_object_t;

// Initializes one caller-owned test object with one owner.
void iree_vm_execution_test_object_initialize(
    int* destruction_count, iree_vm_execution_test_object_t* out_object);

// Returns the canonical execution-test object type.
iree_vm_ref_type_t iree_vm_execution_test_object_type(void);

// Creates one immutable native test module.
iree_status_t iree_vm_execution_test_module_create(
    iree_vm_execution_test_module_kind_t kind,
    iree_vm_execution_test_options_t options,
    iree_vm_execution_test_counters_t* counters,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_EXECUTION_TEST_PROVIDER_H_
