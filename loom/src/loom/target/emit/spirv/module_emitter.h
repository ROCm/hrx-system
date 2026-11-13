// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V binary emission from a prepared target program.
//
// This stage writes already-selected and target-bound Low functions. It does
// not perform entry selection, target resolution, descriptor-set binding,
// verification, scheduling, or allocation. Format encoding limits are diagnosed
// at their source-owned producers. The mutable module in
// the program plan is used only for function-local value-domain scratch.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits one SPIR-V binary module from |program|.
//
// The output module owns allocator-backed word storage and must be
// deinitialized by the caller. The prepared program is trusted compiler-owned
// state and is not revalidated during emission.
iree_status_t loom_spirv_program_emit_binary(
    const loom_spirv_program_plan_t* program,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena, iree_allocator_t allocator,
    bool* out_accepted, loom_spirv_module_binary_t* out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_EMITTER_H_
