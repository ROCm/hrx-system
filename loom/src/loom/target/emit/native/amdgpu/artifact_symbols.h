// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final AMDGPU artifact symbol-namespace admission.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_ARTIFACT_SYMBOLS_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_ARTIFACT_SYMBOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/target/emit/native/amdgpu/data_symbol_directory.h"
#include "loom/target/emit/native/amdgpu/program_artifact.h"
#include "loom/target/entry_selection.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates the final HSACO symbol namespace represented by |input|.
//
// Module verification owns source symbol uniqueness. This admission step adds
// target export aliases and synthesized kernel-descriptor names to that
// namespace. It indexes the module's existing interned strings and retains no
// target-private string table. Entries, helpers, and data rows retain source
// identity in the same order as the final symbol rows. Rejections emit at that
// source and clear |out_accepted|; statuses report infrastructure failures.
iree_status_t loom_amdgpu_artifact_symbols_validate(
    const loom_module_t* module, const loom_amdgpu_hsaco_input_t* input,
    loom_target_entry_list_t entries,
    const loom_amdgpu_program_artifact_t* artifact,
    const loom_amdgpu_data_symbol_directory_t* data_symbols,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_accepted);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_ARTIFACT_SYMBOLS_H_
