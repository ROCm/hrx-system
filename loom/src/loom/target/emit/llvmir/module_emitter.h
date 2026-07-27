// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVMIR low-function module emission.
//
// This stage consumes verified, already-selected target-low IR. It does not
// perform source legality, source-to-low lowering, target specialization,
// scheduling, allocation, or projection. The output is Loom's structured
// LLVMIR module model so callers can serialize text or bitcode through the
// normal LLVMIR writers.

#ifndef LOOM_TARGET_EMIT_LLVMIR_MODULE_EMITTER_H_
#define LOOM_TARGET_EMIT_LLVMIR_MODULE_EMITTER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/llvmir/module.h"
#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_llvmir_emit_low_module_options_t {
  // Selected target-low function ops to emit. NULL emits every target-low
  // function definition in module body order.
  loom_op_t* const* entry_ops;
  // Number of entries in |entry_ops|. Zero keeps the default all-entry
  // behavior.
  iree_host_size_t entry_count;
  // Optional registry of linked target profiles for non-object ABI projection.
  const loom_llvmir_target_profile_registry_t* target_profile_registry;
} loom_llvmir_emit_low_module_options_t;

void loom_llvmir_emit_low_module_options_initialize(
    loom_llvmir_emit_low_module_options_t* out_options);

// Emits target-low function bodies in |module| as one structured LLVMIR module.
//
// Infrastructure and API contract failures return a non-OK status. User IR
// emission failures are reported through |diagnostic_emitter| and return OK
// with |*out_module| set to NULL, preserving diagnostics as the compiler's
// semantic feedback channel.
iree_status_t loom_llvmir_emit_low_module(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    const loom_llvmir_emit_low_module_options_t* options,
    loom_llvmir_module_t** out_module, iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_LLVMIR_MODULE_EMITTER_H_
