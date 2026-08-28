// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deterministic Core VM module layout planning.

#ifndef LOOM_TARGET_EMIT_VM_MODULE_LAYOUT_H_
#define LOOM_TARGET_EMIT_VM_MODULE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/vm/module_resources.h"
#include "loom/target/emit/vm/module_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolved direct-call target for one module symbol.
typedef struct loom_vm_module_call_target_t {
  // Selector identifying a local, required-import, or optional-import target.
  uint8_t kind;
  // Kind-selected target ordinal, or UINT16_MAX when the symbol is not
  // callable.
  uint16_t ordinal;
} loom_vm_module_call_target_t;

// One module-local bytecode function in module symbol order.
typedef struct loom_vm_module_function_layout_t {
  // Prepared low.func.def operation serialized for this function.
  loom_op_t* function_op;
  // Borrowed public export name, or empty for a private function.
  iree_string_view_t export_name;
  // Preserved source-ordered logical callable signature.
  loom_type_t logical_signature;
  // String-table ordinal of |export_name|, or UINT16_MAX when private.
  uint16_t export_name_string_ordinal;
  // Canonical module-local function ordinal.
  uint16_t function_ordinal;
  // Canonical structural callable-type ordinal.
  uint16_t callable_type_ordinal;
  // Derived wire function flags such as conservative suspension permission.
  uint16_t flags;
  // Aggregate switch-target entries required by the function body.
  uint32_t switch_target_entry_count;
} loom_vm_module_function_layout_t;

// One runtime function import declaration in module symbol order.
typedef struct loom_vm_module_import_layout_t {
  // Prepared low.func.decl operation represented by this import.
  loom_op_t* declaration_op;
  // Module symbol ID naming |declaration_op|.
  loom_symbol_id_t symbol_id;
  // Borrowed target-module name.
  iree_string_view_t module_name;
  // Borrowed target export name.
  iree_string_view_t symbol_name;
  // Preserved source-ordered logical callable signature.
  loom_type_t logical_signature;
  // String-table ordinal of |module_name|.
  uint16_t module_name_string_ordinal;
  // String-table ordinal of |symbol_name|.
  uint16_t symbol_name_string_ordinal;
  // Canonical structural callable-type ordinal.
  uint16_t callable_type_ordinal;
  // Callable permission flags derived from the imported ABI contract.
  uint16_t callable_flags;
  // Flat import ordinal assigned after canonical ordering and deduplication.
  uint16_t import_ordinal;
  // Wire import flags shared by equivalent declarations.
  uint16_t flags;
} loom_vm_module_import_layout_t;

// One canonical target-module group in import-table order.
typedef struct loom_vm_module_import_group_layout_t {
  // String-table ordinal of the target-module name.
  uint16_t module_name_string_ordinal;
  // Number of canonical import entries in the group.
  uint32_t import_count;
} loom_vm_module_import_group_layout_t;

// Complete deterministic table plan for one emitted Core VM module.
struct loom_vm_module_layout_t {
  // Module supplying symbols, types, and prepared target-low function bodies.
  loom_module_t* module;
  // Arena-owned functions in deterministic module symbol order.
  loom_vm_module_function_layout_t* functions;
  // Number of entries in |functions|.
  iree_host_size_t function_count;
  // Arena-owned import declarations in deterministic module symbol order.
  loom_vm_module_import_layout_t* import_declarations;
  // Number of entries in |import_declarations|.
  iree_host_size_t import_declaration_count;
  // Arena-owned canonical imports in target-module and symbol order.
  loom_vm_module_import_layout_t** imports;
  // Number of unique entries in |imports|.
  iree_host_size_t import_count;
  // Arena-owned target-module groups in strict target-module order.
  loom_vm_module_import_group_layout_t* import_groups;
  // Number of entries in |import_groups|.
  iree_host_size_t import_group_count;
  // Arena-owned direct-call targets indexed by module symbol ID.
  loom_vm_module_call_target_t* call_targets_by_symbol;
  // Arena-owned exported functions in strict byte-sorted export-name order.
  loom_vm_module_function_layout_t** exports;
  // Number of entries in |exports|.
  iree_host_size_t export_count;
  // Canonical wire-ready string and type tables.
  loom_vm_module_type_tables_t type_tables;
  // Canonical physical global and read-only data layout.
  loom_vm_module_resource_layout_t resources;
  // Aggregate switch-target entries across all functions.
  uint32_t switch_target_entry_count;
};

// Collects supported low functions and assigns all module wire ordinals.
//
// The returned arrays borrow module strings and are owned by |arena|. The
// Prepared low.func.decl operations with runtime link names become canonical
// module imports. Plain declarations and kernel entry points are rejected.
iree_status_t loom_vm_module_layout_build(loom_module_t* module,
                                          iree_arena_allocator_t* arena,
                                          loom_vm_module_layout_t* out_layout);

// Resolves one module-local symbol reference to its direct-call target.
//
// Returns false for an invalid, external-module, or non-callable symbol.
bool loom_vm_module_layout_try_resolve_call_target(
    const loom_vm_module_layout_t* layout, loom_symbol_ref_t symbol_ref,
    loom_vm_module_call_target_t* out_target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_LAYOUT_H_
