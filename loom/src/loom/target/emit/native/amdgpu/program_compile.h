// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU retained-leaf native program compilation.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_COMPILE_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_COMPILE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_references.h"
#include "loom/target/emit/native/amdgpu/function_compile.h"
#include "loom/target/emit/native/amdgpu/program_call.h"
#include "loom/target/emit/native/amdgpu/program_resources.h"
#include "loom/target/function_version.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable inputs for one kernel entry and its direct helper closure.
typedef struct loom_amdgpu_program_compile_options_t {
  // Canonical module symbol-use snapshot.
  const loom_symbol_reference_table_t* references;
  // Concrete target versions indexed by the same module snapshot.
  const loom_target_function_version_snapshot_t* function_versions;
  // Complete function-compilation policy for the kernel entry. Fixed values
  // describe loader-supplied invocation coordinates: the no-call path retains
  // them, while a retained-call closure transports them into allocatable
  // storage before the body CFG.
  loom_amdgpu_function_compile_options_t entry;
} loom_amdgpu_program_compile_options_t;

// Native compilation product for one kernel and its private leaf helpers.
typedef struct loom_amdgpu_program_compilation_t {
  // Compiled loader-visible kernel entry.
  loom_amdgpu_function_compilation_t entry;
  // Module-private leaf helper compilations in ascending symbol order.
  loom_amdgpu_program_callable_t* helpers;
  // Number of entries in |helpers|.
  iree_host_size_t helper_count;
  // Symbol-indexed helper contracts retained for final program layout.
  loom_amdgpu_program_call_table_t call_table;
  // Whole-program storage and register demand when helpers are retained.
  loom_amdgpu_program_resources_t resources;
  // True when |resources| and |helpers| describe a retained-call program.
  bool has_retained_calls;
} loom_amdgpu_program_compilation_t;

// Compiles one kernel and its direct module-private leaf helper closure.
//
// The canonical symbol-reference table supplies the complete closure without
// another IR walk. Entries without direct calls use the ordinary function
// compiler unchanged. User-dependent rejection emits diagnostics and leaves
// |out_compiled| false; infrastructure failures return a status.
iree_status_t loom_amdgpu_program_compile(
    loom_module_t* module, loom_op_t* entry_function_op,
    const loom_amdgpu_program_compile_options_t* options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_amdgpu_program_compilation_t* out_compilation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_COMPILE_H_
