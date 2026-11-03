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
#include "loom/target/emit/native/amdgpu/program_compile.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits one loader-visible entry followed by its module-private leaf helpers.
//
// Function-local data fixups are rebased into the combined text payload and
// internal direct calls are resolved before the kernel row reaches HSACO
// layout. Aggregate text and body instruction counts cover the entry and all
// helpers; function-detail sidecars remain entry-owned because the kernel
// report describes the loader-visible frame. The compilation and entry plan
// must describe the same accepted retained-call program.
iree_status_t loom_amdgpu_program_emission_build(
    const loom_amdgpu_program_compilation_t* compilation,
    const loom_amdgpu_kernel_plan_t* entry_plan,
    loom_amdgpu_kernel_emission_flags_t flags,
    iree_string_builder_t* target_listing,
    loom_amdgpu_kernel_emission_result_t* out_result,
    iree_arena_allocator_t* table_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_EMISSION_H_
