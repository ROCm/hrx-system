// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical Core VM module string and type table planning.

#ifndef LOOM_TARGET_EMIT_VM_MODULE_TYPES_H_
#define LOOM_TARGET_EMIT_VM_MODULE_TYPES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/vm/bytecode/wire/module_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vm_module_layout_t loom_vm_module_layout_t;

// Canonical wire-ready string and type tables for one VM module.
typedef struct loom_vm_module_type_tables_t {
  // Arena-owned unique strings in strict byte order.
  iree_string_view_t* strings;
  // Number of entries in |strings|.
  uint32_t string_count;
  // Arena-owned ref-type namespace groups in strict byte order.
  iree_vm_bytecode_v0_ref_type_group_row_t* ref_type_groups;
  // Number of entries in |ref_type_groups|.
  uint32_t ref_type_group_count;
  // Arena-owned ref-type entries grouped by namespace and sorted by name.
  iree_vm_bytecode_v0_ref_type_entry_row_t* ref_type_entries;
  // Number of entries in |ref_type_entries|.
  uint32_t ref_type_entry_count;
  // Arena-owned source-ordered signature rows in callable-type order.
  iree_vm_bytecode_v0_signature_row_t* signatures;
  // Number of entries in |signatures|.
  uint32_t signature_count;
  // Arena-owned argument-then-result descriptors for all signatures.
  iree_vm_bytecode_v0_signature_descriptor_row_t* signature_descriptors;
  // Number of entries in |signature_descriptors|.
  uint32_t signature_descriptor_count;
  // Arena-owned callable rows in canonical structural order.
  iree_vm_bytecode_v0_callable_type_row_t* callable_types;
  // Number of entries in |callable_types|.
  uint32_t callable_type_count;
} loom_vm_module_type_tables_t;

// Builds all canonical string, ref-type, signature, and callable-type tables.
//
// Function and import logical signatures and names must already be populated
// in |layout|. The build assigns their final string and callable-type ordinals.
iree_status_t loom_vm_module_type_tables_build(iree_arena_allocator_t* arena,
                                               loom_vm_module_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_TYPES_H_
