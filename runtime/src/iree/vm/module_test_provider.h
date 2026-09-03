// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_MODULE_TEST_PROVIDER_H_
#define IREE_VM_MODULE_TEST_PROVIDER_H_

#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stack-backed native provider used by generic module/reflection tests.
typedef struct iree_vm_module_test_provider_t {
  // Generic module base.
  iree_vm_module_t base;
  // Immutable descriptor published through |base|.
  iree_vm_module_descriptor_t descriptor;
  // Resolved vm.buffer type storage.
  iree_vm_ref_type_t buffer_type;
  // Test-owned counter incremented on final release.
  int* destroy_count;
} iree_vm_module_test_provider_t;

// Initializes one fixed native provider with one owner.
iree_status_t iree_vm_module_test_provider_initialize(
    iree_vm_ref_type_t buffer_type, int* destroy_count,
    iree_vm_module_test_provider_t* out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_MODULE_TEST_PROVIDER_H_
