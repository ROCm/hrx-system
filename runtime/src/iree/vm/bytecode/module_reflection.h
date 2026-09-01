// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_MODULE_REFLECTION_H_
#define IREE_VM_BYTECODE_MODULE_REFLECTION_H_

#include "iree/vm/bytecode/module_storage.h"

// Shared generic reflection callbacks over one verified mapped module image.
void iree_vm_bytecode_module_query_import_group(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group);
void iree_vm_bytecode_module_query_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import);
void iree_vm_bytecode_module_query_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export);
void iree_vm_bytecode_module_query_callable_type(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type);
void iree_vm_bytecode_module_query_presentation(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation);
void iree_vm_bytecode_module_metadata_by_ordinal(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry);

#endif  // IREE_VM_BYTECODE_MODULE_REFLECTION_H_
