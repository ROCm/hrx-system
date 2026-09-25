// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_VM_MODULE_H_
#define LOOM_TARGET_ARCH_VM_MODULE_H_

#include "loom/target/arch/vm/function.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Module-owned call bindings shared by table and instruction emission.
typedef struct loom_vm_module_callable_t {
  // Borrowed function definition or import declaration and signature values.
  loom_func_like_t function;
  // Concrete compiler products retained by the shared compilation pipeline.
  const loom_target_function_version_t* function_version;
  // Source-ordered entry argument IDs in the module.
  const loom_value_id_t* arguments;
  // Source-ordered signature result IDs in the module.
  loom_value_slice_t results;
  // Public name, or empty for an internal function.
  iree_string_view_t export_name;
  // Runtime import identity; empty for a local function.
  struct {
    // Module namespace owning the imported callable.
    iree_string_view_t module_name;
    // Export name within that module, independent of the local symbol name.
    iree_string_view_t symbol_name;
  } import;
  // Local-function or flat-import ordinal in the emitted image.
  uint16_t ordinal;
  // Source-ordered logical argument count.
  uint16_t argument_count;
  // Canonical callable ordinal assigned by signature sorting.
  uint16_t callable_ordinal;
  // Wire control.call target kind, shared by local and imported calls.
  uint8_t target_kind;
  // Exact logical fields and their physical argument/result bank counts.
  loom_vm_function_signature_t signature;
} loom_vm_module_callable_t;

// Module-local emission plan. Function ordinals come from the symbol walk;
// data ordinals are assigned on first emitted use, excluding other targets'
// payloads without a second traversal of function bodies.
typedef struct loom_vm_module_plan_t {
  // Arena-owned local and imported callable records in source symbol order.
  loom_vm_module_callable_t* values;
  // Direct symbol-indexed bindings for calls within the VM target contract.
  // Open declarations without an executable binding remain NULL until the
  // selected caller's schedule is validated for artifact emission.
  loom_vm_module_callable_t** bindings_by_symbol;
  // Number of records in |values|, bounded by the module symbol ID space.
  uint32_t count;
  // Number of local definitions, retaining their collected ordinal space.
  uint32_t definition_count;
  // Read-only payloads retained for the module's data section.
  struct {
    // Symbol-indexed data ordinals; UINT16_MAX marks an unreferenced payload.
    uint16_t* ordinals_by_symbol;
    // Module symbol IDs in declaration order for numeric Low operands.
    const loom_symbol_id_t* symbols;
    // Number of entries in |symbols|.
    uint32_t symbol_count;
    // Borrowed definitions in first-use order, with symbol_count capacity.
    const loom_op_t** values;
    // Number of definitions in |values|.
    uint32_t count;
    // Maximum block alignment, at least the image's eight-byte alignment.
    uint32_t alignment;
  } rodata;
} loom_vm_module_plan_t;

// Emits VM functions in a prepared mixed-target module as one immutable .vm
// artifact. Signature and export tables are sorted for runtime consumption;
// the common compiler has already resolved the functions participating in the
// module. Modules without definitions are valid; callable and function sections
// are omitted when empty. Referenced read-only payloads retain their source
// alignment and map to module-owned immutable buffers. Bytes are appended once
// to a segmented stream and fixed table rows are backpatched. No instruction
// sizing pass or contiguous image is required. Success transfers the byte
// sequence to |out_artifact| and sets |out_emitted| true. A structured
// diagnostic returns OK with |out_emitted| false and publishes nothing.
// Infrastructure failure also publishes nothing. All emission-local scratch is
// reclaimed before returning, preserving the caller's preceding allocations.
iree_status_t loom_vm_module_emit(const loom_target_emit_request_t* request,
                                  bool* out_emitted,
                                  loom_target_emit_artifact_t* out_artifact);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_MODULE_H_
