// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native assembly-fragment emission from target-low packet tables.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_assembly_fragment_options_t {
  // Optional target-owned packet plan applied during assembly emission.
  const struct loom_amdgpu_packet_plan_t* packet_plan;
} loom_amdgpu_assembly_fragment_options_t;

// Emits an AMDGPU assembly fragment for one scheduled and allocated AMDGPU
// target-low function. The fragment assumes exact physical-register inputs and
// outputs; it does not emit kernel metadata, PAL metadata, or an ELF code
// object envelope. Values must be physically allocated and unspilled.
iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

// Emits an AMDGPU assembly fragment with target-owned insertion plans.
iree_status_t loom_amdgpu_emit_assembly_fragment_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_assembly_fragment_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_
