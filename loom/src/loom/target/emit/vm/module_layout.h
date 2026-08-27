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

#ifdef __cplusplus
extern "C" {
#endif

// One module-local bytecode function in module symbol order.
typedef struct loom_vm_module_function_layout_t {
  // Prepared low.func.def operation serialized for this function.
  loom_op_t* function_op;
  // Borrowed public export name, or empty for a private function.
  iree_string_view_t export_name;
  // Source-ordered VM signature kinds for function arguments.
  const uint16_t* argument_kinds;
  // Number of entries in |argument_kinds|.
  uint16_t argument_count;
  // Source-ordered VM signature kinds for function results.
  const uint16_t* result_kinds;
  // Number of entries in |result_kinds|.
  uint16_t result_count;
  // Canonical module-local function ordinal.
  uint16_t function_ordinal;
  // Canonical structural callable-type ordinal.
  uint16_t callable_type_ordinal;
  // Aggregate switch-target entries required by the function body.
  uint32_t switch_target_entry_count;
} loom_vm_module_function_layout_t;

// Complete deterministic table plan for one emitted Core VM module.
typedef struct loom_vm_module_layout_t {
  // Module supplying symbols, types, and prepared target-low function bodies.
  loom_module_t* module;
  // Arena-owned functions in deterministic module symbol order.
  loom_vm_module_function_layout_t* functions;
  // Number of entries in |functions|.
  iree_host_size_t function_count;
  // Arena-owned representatives in strict structural callable-type order.
  loom_vm_module_function_layout_t** callable_types;
  // Number of unique entries in |callable_types|.
  iree_host_size_t callable_type_count;
  // Arena-owned exported functions in strict byte-sorted export-name order.
  loom_vm_module_function_layout_t** exports;
  // Number of entries in |exports|.
  iree_host_size_t export_count;
  // Aggregate source-ordered descriptors in unique signatures.
  uint32_t signature_descriptor_count;
  // Aggregate switch-target entries across all functions.
  uint32_t switch_target_entry_count;
} loom_vm_module_layout_t;

// Collects supported low functions and assigns all module wire ordinals.
//
// The returned arrays borrow module strings and are owned by |arena|. The
// current vertical slice accepts low.func.def declarations with direct scalar
// VM value signatures and rejects imports and kernel entry points.
iree_status_t loom_vm_module_layout_build(loom_module_t* module,
                                          iree_arena_allocator_t* arena,
                                          loom_vm_module_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_LAYOUT_H_
