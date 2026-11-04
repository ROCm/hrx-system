// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native kernel emission from a completed target-low frame.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_EMISSION_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_EMISSION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/hsaco_build.h"
#include "loom/target/emit/native/amdgpu/kernel_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_amdgpu_kernel_emission_flags_t;
enum loom_amdgpu_kernel_emission_flag_bits_e {
  // No optional emission facts are retained.
  LOOM_AMDGPU_KERNEL_EMISSION_FLAG_NONE = 0u,
  // Retain native insertion rows for post-emission reporting.
  LOOM_AMDGPU_KERNEL_EMISSION_FLAG_CAPTURE_NATIVE_INSERTIONS = 1u << 0,
};

// Complete output of one successful native kernel emission.
typedef struct loom_amdgpu_kernel_emission_result_t {
  // Kernel row consumed by final HSACO construction.
  loom_amdgpu_hsaco_kernel_t kernel;
  // Encoded function body before the kernel entry envelope is prepended.
  loom_amdgpu_encoded_instruction_stream_t body;
  // Number of instructions prepended by the native entry envelope.
  uint64_t entry_instruction_count;
  // Number of VOPD packets physically emitted in |body|.
  uint64_t coissued_instruction_count;
} loom_amdgpu_kernel_emission_result_t;

// Emits one final HSACO kernel row and optional assembly listing.
//
// |plan| is the accepted compiler-prepared native contract for |frame|.
// Emission consumes it without resolving ABI facts, making semantic acceptance
// decisions, or planning native packets. Assembly and machine-code emission
// consume the same packet plan and exact branch layout.
iree_status_t loom_amdgpu_kernel_emission_build(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_kernel_plan_t* plan,
    loom_amdgpu_kernel_emission_flags_t flags,
    iree_string_builder_t* target_listing,
    loom_amdgpu_kernel_emission_result_t* out_result,
    iree_arena_allocator_t* table_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_EMISSION_H_
