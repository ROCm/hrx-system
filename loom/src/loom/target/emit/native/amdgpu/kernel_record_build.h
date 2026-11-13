// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native record construction from an allocated target-low function.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_RECORD_BUILD_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_RECORD_BUILD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/emit/native/amdgpu/kernel_record.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_native_preflight_t loom_amdgpu_native_preflight_t;

typedef struct loom_amdgpu_kernel_record_options_t {
  // Verified ABI facts captured before allocation.
  const loom_amdgpu_hal_kernel_abi_verify_result_t* abi_verify;
  // Preflight result retained by the admitted native function plan.
  const loom_amdgpu_native_preflight_t* preflight;
  // Transitive scalar-register floor required by the complete native program.
  uint32_t minimum_sgpr_count;
  // Hardware allocation floor retained by final physical packet planning.
  uint32_t minimum_vgpr_count;
  // Transitive accumulator-register floor required by the native program.
  uint32_t minimum_agpr_count;
} loom_amdgpu_kernel_record_options_t;

// Builds the shared native record for one scheduled and allocated target-low
// HAL kernel. |options->abi_verify| must contain the facts retained by the
// successful ABI verification that preceded allocation. The returned record
// points into input IR and |scratch_arena|. The compiler validates target
// storage limits before passing |storage_layout| and validates remaining
// resource feasibility before passing the record to emission. User-dependent
// artifact rejection emits diagnostics and leaves |out_accepted| false.
iree_status_t loom_amdgpu_kernel_record_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_storage_layout_t* storage_layout,
    const loom_amdgpu_kernel_record_options_t* options,
    iree_arena_allocator_t* scratch_arena, iree_diagnostic_emitter_t emitter,
    bool* out_accepted, loom_amdgpu_kernel_record_t* out_record);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_RECORD_BUILD_H_
