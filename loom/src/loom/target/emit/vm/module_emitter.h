// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM bytecode module emission.

#ifndef LOOM_TARGET_EMIT_VM_MODULE_EMITTER_H_
#define LOOM_TARGET_EMIT_VM_MODULE_EMITTER_H_

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/vm/module_selection.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling one Core VM module emission.
typedef struct loom_vm_module_emitter_options_t {
  // Descriptor registry used to schedule and allocate target-low functions.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional per-class register budgets used by allocation.
  const loom_low_allocation_budget_t* allocation_budgets;
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Structured diagnostic emitter for user-facing compilation failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Optional concrete compiler function versions represented in the module.
  const loom_function_version_list_t* function_versions;
  // Optional caller-owned buffer receiving public export ordinals for
  // functions with stable compiler versions.
  loom_target_emit_export_projection_buffer_t* export_projection;
  // Optional artifact-local callable and module-state selection.
  const loom_vm_module_emission_selection_t* selection;
} loom_vm_module_emitter_options_t;

// Emits one immutable segmented VM bytecode image from prepared target-low IR.
//
// The image is serialized once into 32KB stream blocks. Forward-declared
// function rows and the section directory are patched in place after their
// payloads are known; no contiguous staging allocation or sizing
// serialization is performed. User IR diagnostics return OK with
// |*out_contents| set to NULL.
iree_status_t loom_vm_emit_module(
    loom_module_t* module, const loom_vm_module_emitter_options_t* options,
    iree_arena_allocator_t* scratch_arena, iree_allocator_t host_allocator,
    iree_byte_sequence_t** out_contents);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_MODULE_EMITTER_H_
