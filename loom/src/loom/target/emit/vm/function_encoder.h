// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM function packet encoding.

#ifndef LOOM_TARGET_EMIT_VM_FUNCTION_ENCODER_H_
#define LOOM_TARGET_EMIT_VM_FUNCTION_ENCODER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/error/emitter.h"
#include "loom/format/bytecode/writer/encoder.h"
#include "loom/target/emit/vm/module_layout.h"
#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Inputs shared by each function encoded into one module.
typedef struct loom_vm_function_encoder_options_t {
  // Descriptor registry used to resolve the function's exact low target.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional per-class register budgets for scheduling and allocation.
  const loom_low_allocation_budget_t* allocation_budgets;
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Structured diagnostic emitter for scheduling and allocation failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
} loom_vm_function_encoder_options_t;

// Result of one function encoding attempt.
typedef struct loom_vm_function_encoding_t {
  // Fully populated Functions section row when |is_complete| is true.
  iree_vm_bytecode_v0_function_row_t row;
  // False when diagnostics prevented function packet emission.
  bool is_complete;
} loom_vm_function_encoding_t;

// Schedules, allocates, and appends one function's instruction records.
//
// The function body is serialized exactly once. All temporary compiler state
// is owned by |scratch_arena| and may be discarded immediately after return.
// The caller supplies bytecode_offset_u32 after encoding all preceding bodies.
iree_status_t loom_vm_function_encode(
    loom_module_t* module, const loom_vm_module_function_layout_t* function,
    const loom_vm_function_encoder_options_t* options,
    iree_arena_allocator_t* scratch_arena, loom_bytecode_page_writer_t* writer,
    loom_vm_function_encoding_t* out_encoding);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_FUNCTION_ENCODER_H_
