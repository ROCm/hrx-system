// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU per-kernel native emission facts.
//
// This layer derives the target, ABI, metadata, descriptor, and register facts
// shared by text assembly and direct HSACO emission from target-low tables.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_RECORD_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_RECORD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/descriptor.h"
#include "loom/target/emit/native/amdgpu/metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_native_preflight_t loom_amdgpu_native_preflight_t;

typedef struct loom_amdgpu_kernel_record_t {
  // Exported kernel entry symbol.
  iree_string_view_t symbol;
  // Loader-visible kernel descriptor symbol.
  iree_string_view_t descriptor_symbol;
  // Full AMDHSA target id such as `amdgcn-amd-amdhsa--gfx1100`.
  iree_string_view_t target_id;
  // Processor facts selected by the AMDGPU target record.
  const loom_amdgpu_processor_info_t* processor;
  // HAL kernel ABI layout derived from function-local low.resource imports.
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout;
  // Metadata row shared by assembly notes and direct HSACO notes.
  loom_amdgpu_metadata_kernel_t metadata;
  // Descriptor flags not represented in the metadata row.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags;
  // Encoded workitem-id mode for AMDHSA text assembly directives.
  uint32_t system_vgpr_workitem_id;
  // User SGPR count required by ABI live-ins.
  uint32_t user_sgpr_count;
} loom_amdgpu_kernel_record_t;

typedef struct loom_amdgpu_kernel_record_options_t {
  // Optional ABI layout captured before target resource materialization.
  const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout;
  // Optional preflight result captured before kernel record construction.
  const loom_amdgpu_native_preflight_t* preflight;
} loom_amdgpu_kernel_record_options_t;

// Builds the shared emission record for one scheduled and allocated target-low
// HAL kernel. The returned record points into input IR and |scratch_arena|.
iree_status_t loom_amdgpu_kernel_record_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_record_options_t* options,
    loom_amdgpu_kernel_record_t* out_record,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_RECORD_H_
