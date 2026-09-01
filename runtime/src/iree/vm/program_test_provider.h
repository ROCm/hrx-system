// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_PROGRAM_TEST_PROVIDER_H_
#define IREE_VM_PROGRAM_TEST_PROVIDER_H_

#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Static native C module definition used by immutable program-linking tests.
// Every pointer remains borrowed and must outlive the created module.
typedef struct iree_vm_program_test_module_definition_t {
  // Complete fixed generic module descriptor.
  iree_vm_module_descriptor_t descriptor;
  // Import groups matching |descriptor.counts.import_group_count|.
  const iree_vm_module_import_group_t* import_groups;
  // Imports matching |descriptor.counts.import_count|.
  const iree_vm_module_import_declaration_t* imports;
  // Exports matching |descriptor.counts.export_count|.
  const iree_vm_module_export_declaration_t* exports;
  // Callable types matching |descriptor.counts.callable_type_count|.
  const iree_vm_module_callable_type_declaration_t* callable_types;
} iree_vm_program_test_module_definition_t;

// Returns the first canonical test-provider ref type.
iree_vm_ref_type_t iree_vm_program_test_auxiliary_ref_type(void);

// Returns the second canonical test-provider ref type.
iree_vm_ref_type_t iree_vm_program_test_shared_ref_type(void);

// Creates one immutable native C module borrowing |definition|.
iree_status_t iree_vm_program_test_module_create(
    const iree_vm_program_test_module_definition_t* definition,
    int* destruction_count, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_PROGRAM_TEST_PROVIDER_H_
