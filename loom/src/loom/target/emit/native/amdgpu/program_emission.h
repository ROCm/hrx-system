// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU retained-program emission from independently compiled functions.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_EMISSION_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_EMISSION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/target/emit/native/amdgpu/kernel_emission.h"
#include "loom/target/emit/native/amdgpu/kernel_plan.h"
#include "loom/target/emit/native/amdgpu/program_artifact.h"
#include "loom/target/emit/native/amdgpu/program_compile.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encoded artifact-wide module-local helper functions.
typedef struct loom_amdgpu_program_artifact_emission_t {
  // Compiled artifact paired with every encoded helper stream.
  const loom_amdgpu_program_artifact_t* artifact;
  // Final HSACO function rows in artifact helper-ordinal order.
  const loom_amdgpu_hsaco_function_t* functions;
  // Encoded streams retained for exact assembly and entry-level reporting.
  const loom_amdgpu_encoded_instruction_stream_t* helper_streams;
  // Number of entries in |functions| and |helper_streams|.
  iree_host_size_t function_count;
} loom_amdgpu_program_artifact_emission_t;

// Encodes every artifact helper exactly once for final HSACO layout.
iree_status_t loom_amdgpu_program_artifact_emission_build(
    const loom_amdgpu_program_artifact_t* artifact,
    loom_amdgpu_program_artifact_emission_t* out_emission,
    iree_arena_allocator_t* table_arena);

// Emits one loader-visible entry with artifact-relative direct call fixups.
//
// |include_artifact_functions_in_listing| prints the shared helper definitions
// after this entry. Exactly one retained-call entry in a combined target
// listing should set it. Body instruction counts continue to describe the
// complete entry closure even though helper machine text is stored once.
iree_status_t loom_amdgpu_program_emission_build(
    const loom_amdgpu_program_compilation_t* compilation,
    const loom_amdgpu_program_artifact_emission_t* artifact_emission,
    const loom_amdgpu_kernel_plan_t* entry_plan,
    loom_amdgpu_kernel_emission_flags_t flags,
    bool include_artifact_functions_in_listing,
    iree_string_builder_t* target_listing,
    loom_amdgpu_kernel_emission_result_t* out_result,
    iree_arena_allocator_t* table_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_EMISSION_H_
