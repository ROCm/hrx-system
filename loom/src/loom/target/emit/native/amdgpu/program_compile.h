// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native root compilation against an artifact-wide helper closure.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_COMPILE_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_COMPILE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/emit/native/amdgpu/function_compile.h"
#include "loom/target/emit/native/amdgpu/program_artifact.h"
#include "loom/target/emit/native/amdgpu/program_resources.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable inputs for one kernel entry and its retained artifact closure.
typedef struct loom_amdgpu_program_compile_options_t {
  // Artifact-planned root closure paired with |entry|.
  const loom_amdgpu_program_artifact_root_t* artifact_root;
  // Complete function-compilation policy for the kernel entry. Fixed values
  // describe loader-supplied invocation coordinates: the no-call path retains
  // them, while a retained-call closure transports them into allocatable
  // storage before the body CFG.
  loom_amdgpu_function_compile_options_t entry;
} loom_amdgpu_program_compile_options_t;

// Native compilation product for one kernel root.
typedef struct loom_amdgpu_program_compilation_t {
  // Compiled loader-visible kernel entry.
  loom_amdgpu_function_compilation_t entry;
  // Artifact-planned closure consumed by this entry.
  const loom_amdgpu_program_artifact_root_t* artifact_root;
  // Whole-program storage and register demand when helpers are retained.
  loom_amdgpu_program_resources_t resources;
  // True when |resources| and |artifact_root| describe a retained-call root.
  bool has_retained_calls;
} loom_amdgpu_program_compilation_t;

// Compiles one kernel against its artifact-planned direct helper closure.
//
// Entries without direct calls use the ordinary function compiler unchanged.
// User-dependent rejection emits diagnostics and leaves |out_compiled| false;
// infrastructure failures return a status.
iree_status_t loom_amdgpu_program_compile(
    loom_module_t* module, loom_op_t* entry_function_op,
    const loom_amdgpu_program_compile_options_t* options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_amdgpu_program_compilation_t* out_compilation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_COMPILE_H_
