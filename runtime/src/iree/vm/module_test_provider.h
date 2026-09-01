// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_MODULE_TEST_PROVIDER_H_
#define IREE_VM_MODULE_TEST_PROVIDER_H_

#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_test_module_counters_t {
  // Number of successfully attached process-state slices.
  int attach_count;
  // Number of successfully sealed process-state slices.
  int seal_count;
  // Number of detached process-state slices.
  int detach_count;
  // Number of final module destruction callbacks.
  int destroy_count;
} iree_vm_test_module_counters_t;

// Creates one immutable native C module used by generic module API tests.
iree_status_t iree_vm_test_module_create(
    iree_vm_environment_t* environment,
    iree_vm_test_module_counters_t* counters, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_MODULE_TEST_PROVIDER_H_
