// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical typed metadata planning for Core VM modules.

#ifndef LOOM_TARGET_EMIT_VM_MODULE_METADATA_H_
#define LOOM_TARGET_EMIT_VM_MODULE_METADATA_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vm_module_layout_t loom_vm_module_layout_t;

// One canonical typed metadata entry.
typedef struct loom_vm_module_metadata_entry_layout_t {
  // Borrowed nonempty metadata key.
  iree_string_view_t key;
  // String-table ordinal of |key|.
  uint16_t key_string_ordinal;
  // Stable VM metadata value type.
  uint16_t value_type;
  // Exact value bytes selected by |value_type|.
  union {
    // Canonical little-endian storage for fixed-width scalar values.
    uint8_t scalar[8];
    // Borrowed storage for UTF-8 strings and opaque byte values.
    iree_const_byte_span_t variable;
  } value;
} loom_vm_module_metadata_entry_layout_t;

// One nonempty import or export metadata scope.
typedef struct loom_vm_module_metadata_scope_layout_t {
  // Ordinal in the selected import or export declaration domain.
  uint16_t declaration_ordinal;
  // Nonzero number of entries in this scope.
  uint16_t entry_count;
  // Canonical running base in the shared entry array.
  uint32_t entry_base;
} loom_vm_module_metadata_scope_layout_t;

// Canonical metadata tables for one emitted module.
typedef struct loom_vm_module_metadata_layout_t {
  // Arena-owned entries ordered by module, import, then export scope.
  loom_vm_module_metadata_entry_layout_t* entries;
  // Number of module-scope entries at the start of |entries|.
  uint32_t module_entry_count;
  // Total number of entries in |entries|.
  uint32_t total_entry_count;
  // Arena-owned nonempty import scopes in declaration ordinal order.
  loom_vm_module_metadata_scope_layout_t* import_scopes;
  // Number of entries in |import_scopes|.
  uint32_t import_scope_count;
  // Arena-owned nonempty export scopes in declaration ordinal order.
  loom_vm_module_metadata_scope_layout_t* export_scopes;
  // Number of entries in |export_scopes|.
  uint32_t export_scope_count;
  // Arena-owned canonical value offsets with |total_entry_count + 1| entries.
  uint64_t* value_offsets;
} loom_vm_module_metadata_layout_t;

// Builds canonical typed metadata from finalized public declarations.
//
// Module entries are sorted by key. Import and export dictionaries are already
// canonical by construction and are flattened in declaration ordinal order.
// Unsupported Loom attribute kinds fail emission instead of being stringified.
iree_status_t loom_vm_module_metadata_layout_build(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout);

// Resolves every metadata key against the finalized string table.
void loom_vm_module_metadata_resolve_string_ordinals(
    loom_vm_module_layout_t* layout);

// Returns the immutable encoded value bytes for |entry|.
iree_const_byte_span_t loom_vm_module_metadata_entry_value(
    const loom_vm_module_metadata_entry_layout_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_METADATA_H_
