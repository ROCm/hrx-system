// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA kernel assembly envelopes over target-low native fragments.
//
// The fragment emitter owns instruction syntax. This layer owns the first
// loadable-kernel boundary: symbol selection, .amdgcn_target, AMDHSA kernel
// descriptor directives, and the strict subset that is safe before full HAL
// ABI lowering exists.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ASSEMBLY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ASSEMBLY_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_kernel_assembly_options_t {
  // Optional ABI layout captured before target resource materialization.
  const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout;
  // Optional target-owned packet plan applied during assembly emission.
  const struct loom_amdgpu_packet_plan_t* packet_plan;
} loom_amdgpu_kernel_assembly_options_t;

// Emits complete AMDGPU assembly for one ABI-lowered target-low HAL kernel.
// The output is assembler input containing a text function body and an AMDHSA
// kernel descriptor. It deliberately remains text: assembling, disassembling,
// loading, and launching are tool/runtime adapter responsibilities.
iree_status_t loom_amdgpu_emit_kernel_assembly(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

// Emits complete AMDGPU assembly with optional target-owned emission tables.
// |scratch_arena| receives transient ABI layout and metadata adapter storage.
iree_status_t loom_amdgpu_emit_kernel_assembly_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_assembly_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ASSEMBLY_H_
