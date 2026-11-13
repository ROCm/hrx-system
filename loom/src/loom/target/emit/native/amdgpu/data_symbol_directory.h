// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Artifact-wide AMDGPU data-symbol indexing and HSACO rows.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_DATA_SYMBOL_DIRECTORY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_DATA_SYMBOL_DIRECTORY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/target/emit/native/amdgpu/data_symbol_fixup.h"
#include "loom/target/emit/native/amdgpu/hsaco_build.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_DATA_SYMBOL_ORDINAL_INVALID UINT32_MAX

// Immutable artifact-wide data-symbol directory.
//
// Runtime support globals precede authored read-only data to preserve the HSA
// runtime ABI. |ordinals_by_module_symbol| binds module-local identities to
// those final rows so native encoding never recovers identity from names.
typedef struct loom_amdgpu_data_symbol_directory_t {
  // Immutable module owning |ordinals_by_module_symbol| identities.
  const loom_module_t* module;
  // Final HSACO data-symbol rows in layout order.
  const loom_amdgpu_hsaco_data_symbol_t* symbols;
  // Module-local source symbol for each final row, or INVALID for a runtime
  // global without a declaration. Retained for artifact admission diagnostics.
  const loom_symbol_id_t* source_symbols;
  // Number of entries in |symbols| and |source_symbols|.
  iree_host_size_t symbol_count;
  // Final data-symbol ordinal by module-local symbol ID, or
  // LOOM_AMDGPU_DATA_SYMBOL_ORDINAL_INVALID for non-data symbols.
  const uint32_t* ordinals_by_module_symbol;
  // Number of entries in |ordinals_by_module_symbol|.
  iree_host_size_t module_symbol_count;
  // Placement contract applied to |symbols|.
  loom_amdgpu_hsaco_data_layout_t layout;
} loom_amdgpu_data_symbol_directory_t;

// Builds the final data-symbol directory for one AMDGPU artifact.
iree_status_t loom_amdgpu_data_symbol_directory_build(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_amdgpu_data_symbol_directory_t* out_directory);

// Binds module-local data-symbol fixups to final HSACO symbol ordinals.
iree_status_t loom_amdgpu_data_symbol_directory_bind_fixups(
    const loom_amdgpu_data_symbol_directory_t* directory,
    const loom_amdgpu_data_symbol_fixup_t* fixups, iree_host_size_t fixup_count,
    iree_arena_allocator_t* arena,
    const loom_amdgpu_hsaco_text_fixup_t** out_fixups);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_DATA_SYMBOL_DIRECTORY_H_
