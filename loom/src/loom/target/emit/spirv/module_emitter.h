// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V low-function binary emission.
//
// This stage consumes already-selected target-low IR. It does not perform
// source legality, source-to-low lowering, target specialization, scheduling,
// or allocation. The mutable module argument is used only for the
// function-local value-domain scratch map.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_emit_low_module_options_t {
  // Selected target-low function ops to emit. NULL emits every target-low
  // function definition in the module.
  loom_op_t* const* entry_ops;
  // Number of entries in |entry_ops|. Zero keeps the default all-entry
  // behavior.
  iree_host_size_t entry_count;
} loom_spirv_emit_low_module_options_t;

void loom_spirv_emit_low_module_options_initialize(
    loom_spirv_emit_low_module_options_t* out_options);

// Emits target-low function bodies in |module| as one SPIR-V module.
//
// SPIR-V modules may contain multiple entry points. By default every
// target-low function definition is emitted. Callers may provide |options| to
// select one or more entries when an artifact container describes a narrower
// dispatch set than the whole source module.
//
// Every emitted function must resolve to the same SPIR-V module contract. Raw
// BDA HAL kernel entries must also share one dispatch ABI layout because the
// current raw SPIR-V executable format exposes BDA metadata at module scope.
// The output module owns allocator-backed word storage and must be
// deinitialized by the caller.
iree_status_t loom_spirv_emit_low_module(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    const loom_spirv_emit_low_module_options_t* options,
    loom_spirv_module_binary_t* out_module, iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_
