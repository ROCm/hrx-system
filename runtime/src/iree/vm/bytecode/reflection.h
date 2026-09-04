// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_REFLECTION_H_
#define IREE_VM_BYTECODE_REFLECTION_H_

#include "iree/vm/bytecode/image.h"
#include "iree/vm/reflection.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Populates one generic import group from verified bytecode tables.
void iree_vm_bytecode_reflection_query_import_group(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group);

// Populates one generic import declaration from verified bytecode tables.
void iree_vm_bytecode_reflection_query_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import);

// Populates one generic export declaration from verified bytecode tables.
void iree_vm_bytecode_reflection_query_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export);

// Populates one generic callable declaration from verified bytecode tables.
void iree_vm_bytecode_reflection_query_callable_type(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type);

// Populates optional authored presentation from verified bytecode tables.
void iree_vm_bytecode_reflection_query_presentation(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation);

// Populates one stable typed metadata entry from verified bytecode tables.
void iree_vm_bytecode_reflection_metadata_by_ordinal(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_REFLECTION_H_
