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
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vm_module_layout_t loom_vm_module_layout_t;

// Canonical callable ordinals for one structural function signature.
typedef struct loom_vm_module_callable_type_ordinals_t {
  // Ordinal of the synchronous callable type, or UINT16_MAX when unused.
  uint16_t synchronous;
  // Ordinal of the yieldable callable type, or UINT16_MAX when unused.
  uint16_t yieldable;
} loom_vm_module_callable_type_ordinals_t;

// Canonical wire-ready type tables and compiler lookups for one VM module.
typedef struct loom_vm_module_type_tables_t {
  // Arena-owned unique strings in strict byte order.
  iree_string_view_t* strings;
  // Number of entries in |strings|.
  uint32_t string_count;
  // Arena-owned ref-type namespace groups in strict byte order.
  iree_vm_bytecode_v0_ref_type_group_row_t* ref_type_groups;
  // Number of entries in |ref_type_groups|.
  uint32_t ref_type_group_count;
  // Arena-owned namespace names parallel to |ref_type_groups|.
  iree_string_view_t* ref_type_group_names;
  // Arena-owned ref-type entries grouped by namespace and sorted by name.
  iree_vm_bytecode_v0_ref_type_entry_row_t* ref_type_entries;
  // Number of entries in |ref_type_entries|.
  uint32_t ref_type_entry_count;
  // Arena-owned local names parallel to |ref_type_entries|.
  iree_string_view_t* ref_type_entry_names;
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
  // Arena-owned compiler lookup indexed by Loom function-signature type ID.
  loom_vm_module_callable_type_ordinals_t* callable_type_ordinals_by_signature;
  // Number of entries in |callable_type_ordinals_by_signature|.
  iree_host_size_t callable_type_ordinal_signature_count;
} loom_vm_module_type_tables_t;

// Builds canonical ref-type, signature, and callable-type structure.
//
// Function and import logical signatures must already be populated in
// |layout|. The build assigns their final callable-type ordinals but leaves
// string-backed row fields unresolved so import canonicalization and public
// presentation can contribute to the one module string table.
iree_status_t loom_vm_module_type_tables_build_structure(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout);

// Builds the one canonical string table and resolves every string-backed row.
//
// Canonical imports and optional presentation must be finalized before this
// call. No string-bearing module resource may be added afterward.
iree_status_t loom_vm_module_type_tables_build_strings(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout);

// Resolves |value| in the canonical module string table.
//
// Returns false when |value| was not included during string planning.
bool loom_vm_module_type_tables_try_resolve_string_ordinal(
    const loom_vm_module_type_tables_t* tables, iree_string_view_t value,
    uint16_t* out_ordinal);

// Resolves the canonical module-local callable ordinal for |function_ref_type|.
//
// Returns false when the type is not a planned func.ref type. This is a
// compiler-side lookup over the already canonicalized module plan and performs
// no interning, hashing, allocation, or structural walk.
bool loom_vm_module_type_tables_try_resolve_callable_ordinal(
    const loom_vm_module_type_tables_t* tables, loom_type_t function_ref_type,
    uint16_t* out_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_TYPES_H_
