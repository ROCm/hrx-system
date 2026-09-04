// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_ENVIRONMENT_H_
#define IREE_VM_ENVIRONMENT_H_

#include "iree/base/api.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_environment_t iree_vm_environment_t;

// Thread-safe construction registry for host-managed ref-type providers.
// Registration and lookup borrow provider tables; the environment never owns
// or copies their descriptors, strings, callbacks, or containing code. Modules
// resolve descriptor pointers during construction and do not retain the
// environment. The host therefore keeps each provider scope live while any
// table lookup, module, ref, or variant can name one of its descriptors.

// Allocates a thread-safe construction environment containing the complete
// core "vm" provider table. Provider tables are borrowed from host-managed
// definition scopes. |out_environment| is set null before construction.
IREE_API_EXPORT iree_status_t iree_vm_environment_allocate(
    iree_allocator_t host_allocator, iree_vm_environment_t** out_environment);

// Frees |environment| without affecting modules or provider tables. A null
// environment is ignored. Registration and lookup operations must already be
// joined by the caller.
IREE_API_EXPORT void iree_vm_environment_free(
    iree_vm_environment_t* environment);

// Registers one complete provider table failure-atomically. The namespace must
// not already be registered, including by the same table. Registration borrows
// every provider-owned pointer and performs no retain or allocation.
IREE_API_EXPORT iree_status_t iree_vm_environment_register_ref_type_table(
    iree_vm_environment_t* environment, const iree_vm_ref_type_table_t* table);

// Looks up the table owning |namespace_name|. Absence returns null as ordinary
// query control flow. The returned table remains borrowed from its host-managed
// provider scope.
IREE_API_EXPORT const iree_vm_ref_type_table_t*
iree_vm_environment_lookup_ref_type_table(iree_vm_environment_t* environment,
                                          iree_string_view_t namespace_name);

// Common provider registration callback shape. A successful callback registers
// the provider table in |environment| and returns the same borrowed table in
// |out_table|. Failure leaves |out_table| untouched. The VM defines no dynamic
// library symbol name for this callback, allowing multiple providers per
// dynamic library and the same providers under static linkage.
typedef iree_status_t(IREE_API_PTR* iree_vm_ref_type_table_register_fn_t)(
    iree_vm_environment_t* environment,
    const iree_vm_ref_type_table_t** out_table);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_ENVIRONMENT_H_
