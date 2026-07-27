// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM module ordinal planning.
//
// VM bytecode calls encode local and imported callees as dense ordinals. This
// plan assigns those ordinals once for the whole emitted module and exposes a
// symbol-ID-indexed callee table so per-call bytecode emission never searches.

#ifndef LOOM_TARGET_EMIT_IREEVM_MODULE_PLAN_H_
#define LOOM_TARGET_EMIT_IREEVM_MODULE_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_IREEVM_IMPORT_ORDINAL_BIT UINT32_C(0x80000000)

typedef enum loom_ireevm_callee_kind_e {
  // Symbol is not part of the emitted VM module callee table.
  LOOM_IREEVM_CALLEE_NONE = 0,
  // Symbol names a local VM function body.
  LOOM_IREEVM_CALLEE_LOCAL = 1,
  // Symbol names a runtime-resolved VM import.
  LOOM_IREEVM_CALLEE_IMPORT = 2,
} loom_ireevm_callee_kind_t;

typedef struct loom_ireevm_callee_slot_t {
  // Callee category for this module-local symbol ID.
  loom_ireevm_callee_kind_t kind;
  // Dense import or local function ordinal before VM import-bit encoding.
  uint32_t ordinal;
} loom_ireevm_callee_slot_t;

typedef struct loom_ireevm_module_plan_import_t {
  // Module-local symbol reference for this import declaration.
  loom_symbol_ref_t symbol_ref;
  // Low import declaration op.
  const loom_op_t* op;
  // Fully-qualified runtime import name stored in the VM import table.
  iree_string_view_t full_name;
  // VM calling convention string derived from the low signature.
  iree_string_view_t calling_convention;
} loom_ireevm_module_plan_import_t;

typedef struct loom_ireevm_module_plan_function_t {
  // Module-local symbol reference for this local function body.
  loom_symbol_ref_t symbol_ref;
  // Low function definition op to schedule, allocate, and emit.
  loom_op_t* op;
  // Local function symbol name used in diagnostics.
  iree_string_view_t symbol_name;
  // VM calling convention string derived from the low signature.
  iree_string_view_t calling_convention;
  // Resolved target bundle for this function.
  loom_target_bundle_storage_t bundle_storage;
} loom_ireevm_module_plan_function_t;

typedef struct loom_ireevm_module_plan_export_t {
  // Module-local symbol reference for the exported local function.
  loom_symbol_ref_t symbol_ref;
  // Exported local name in the VM module namespace.
  iree_string_view_t local_name;
  // Dense local function ordinal implementing this export.
  uint32_t internal_ordinal;
} loom_ireevm_module_plan_export_t;

typedef struct loom_ireevm_module_plan_t {
  // Module this plan was built from.
  const loom_module_t* module;
  // Symbol-ID-indexed callee table with |callee_slot_count| entries.
  loom_ireevm_callee_slot_t* callee_slots;
  // Number of entries in |callee_slots|.
  iree_host_size_t callee_slot_count;
  // VM imports in deterministic module symbol order.
  loom_ireevm_module_plan_import_t* imports;
  // Number of VM imports.
  uint32_t import_count;
  // Local VM functions in deterministic module symbol order.
  loom_ireevm_module_plan_function_t* functions;
  // Number of local VM functions.
  uint32_t function_count;
  // Exported VM functions in deterministic module symbol order.
  loom_ireevm_module_plan_export_t* exports;
  // Number of exported VM functions.
  uint32_t export_count;
} loom_ireevm_module_plan_t;

// Builds the VM module plan for all compatible low.func.def and VM
// low.func.decl symbols in |module|. All arrays and calling-convention strings
// are allocated from |arena|. Returns status for infrastructure failures.
// Invalid target contracts emit diagnostics, set |out_valid| to false, and
// return OK.
iree_status_t loom_ireevm_module_plan_build(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, loom_ireevm_module_plan_t* out_plan);

// Resolves |callee| to the encoded VM call ordinal used by bytecode CALL.
// Imports have LOOM_IREEVM_IMPORT_ORDINAL_BIT set; local functions do not.
iree_status_t loom_ireevm_module_plan_resolve_callee(
    const loom_ireevm_module_plan_t* plan, loom_symbol_ref_t callee,
    uint32_t* out_encoded_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_IREEVM_MODULE_PLAN_H_
