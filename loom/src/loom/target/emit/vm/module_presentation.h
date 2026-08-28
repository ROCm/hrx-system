// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Optional public declaration presentation for Core VM modules.

#ifndef LOOM_TARGET_EMIT_VM_MODULE_PRESENTATION_H_
#define LOOM_TARGET_EMIT_VM_MODULE_PRESENTATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vm_module_layout_t loom_vm_module_layout_t;

// Presentation for one machine argument or result field.
typedef struct loom_vm_module_presentation_field_layout_t {
  // Borrowed typed register anchored at this machine field, or NULL.
  // Aggregate continuation fields have no independent register type.
  const loom_type_t* register_type;
  // Borrowed authored field name, or empty when unavailable.
  iree_string_view_t name;
  // Arena-owned canonical authored logical type.
  iree_string_view_t authored_type;
  // String-table ordinal of |name|, or UINT16_MAX when unavailable.
  uint16_t name_string_ordinal;
  // String-table ordinal of |authored_type|.
  uint16_t authored_type_string_ordinal;
} loom_vm_module_presentation_field_layout_t;

// Presentation for one public import or export declaration.
typedef struct loom_vm_module_presentation_entry_layout_t {
  // Import or export declaration kind.
  uint16_t declaration_kind;
  // Ordinal in the selected public declaration domain.
  uint16_t declaration_ordinal;
  // Borrowed declaration documentation, or empty when unavailable.
  iree_string_view_t documentation;
  // Arena-owned canonical authored function type.
  iree_string_view_t authored_type;
  // String-table ordinal of |documentation|, or UINT16_MAX when unavailable.
  uint16_t documentation_string_ordinal;
  // String-table ordinal of |authored_type|.
  uint16_t authored_type_string_ordinal;
  // Canonical running base in the shared field array.
  uint32_t field_base;
} loom_vm_module_presentation_entry_layout_t;

// Canonical presentation table for one emitted module.
typedef struct loom_vm_module_presentation_layout_t {
  // Arena-owned imports followed by exports in ordinal order.
  loom_vm_module_presentation_entry_layout_t* entries;
  // Number of entries in |entries|.
  uint32_t entry_count;
  // Arena-owned fields in entry and machine-signature order.
  loom_vm_module_presentation_field_layout_t* fields;
  // Number of entries in |fields|.
  uint32_t field_count;
} loom_vm_module_presentation_layout_t;

// Builds canonical presentation text from finalized public declarations.
//
// Logical types are recovered from the semantic value types retained by VM
// typed registers. Declaration and field strings are allocated together from
// |arena|. Missing source names and documentation remain absent rather than
// being synthesized.
iree_status_t loom_vm_module_presentation_layout_build(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout);

// Resolves every present presentation string against the finalized string
// table. All presentation text must have participated in string planning.
void loom_vm_module_presentation_resolve_string_ordinals(
    loom_vm_module_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_PRESENTATION_H_
