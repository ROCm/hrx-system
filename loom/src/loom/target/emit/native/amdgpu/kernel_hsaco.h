// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA code-object emission from target-low native fragments.
//
// This layer consumes a scheduled and physically allocated target-low HAL
// kernel and writes a loadable HSACO ELF containing metadata, a kernel
// descriptor, and encoded native text.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_HSACO_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_HSACO_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_native_preflight_t loom_amdgpu_native_preflight_t;

typedef struct loom_amdgpu_kernel_hsaco_summary_t {
  // Number of native instructions emitted into the kernel text stream.
  uint64_t instruction_count;
  // Number of bytes in the semantic kernel text stream.
  uint64_t text_byte_count;
  // Number of bytes in the stored kernel text stream including local padding.
  uint64_t text_storage_byte_count;
  // Number of private/scratch memory bytes declared by kernel metadata.
  uint64_t private_segment_fixed_size;
  // Number of local/shared memory bytes declared by kernel metadata.
  uint64_t group_segment_fixed_size;
} loom_amdgpu_kernel_hsaco_summary_t;

typedef struct loom_amdgpu_kernel_hsaco_options_t {
  // Optional ABI layout captured before target resource materialization.
  const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout;
  // Optional preflight result captured before HSACO contribution construction.
  const loom_amdgpu_native_preflight_t* preflight;
  // Optional target-owned packet plan applied during native encoding.
  const struct loom_amdgpu_packet_plan_t* packet_plan;
  // Optional target-owned emission summary populated after successful emission.
  loom_amdgpu_kernel_hsaco_summary_t* summary;
} loom_amdgpu_kernel_hsaco_options_t;

typedef struct loom_amdgpu_kernel_hsaco_contribution_t {
  // Full AMDHSA target id such as `amdgcn-amd-amdhsa--gfx1100`.
  iree_string_view_t target;
  // Processor such as `gfx1100`, used for ELF flags and descriptor packing.
  iree_string_view_t processor;
  // Kernel entry metadata, descriptor flags, and encoded native text.
  loom_amdgpu_hsaco_kernel_t kernel;
  // Emission summary for this kernel contribution.
  loom_amdgpu_kernel_hsaco_summary_t summary;
} loom_amdgpu_kernel_hsaco_contribution_t;

// Builds one arena-owned AMDGPU kernel contribution from a scheduled low func.
//
// The returned contribution points into the input IR and |scratch_arena|. It
// can be written by loom_amdgpu_write_kernel_hsaco_contributions() after all
// worker-local contributions for the code object are complete.
iree_status_t loom_amdgpu_build_kernel_hsaco_contribution(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    iree_arena_allocator_t* scratch_arena);

// Writes one code object containing all |contributions|.
//
// Contributions must all target the same AMDHSA target id and processor. The
// writer uses |scratch_arena| only for final layout tables and can run after
// kernel contributions were produced independently, provided their backing
// storage remains live for the duration of this call.
iree_status_t loom_amdgpu_write_kernel_hsaco_contributions(
    const loom_amdgpu_kernel_hsaco_contribution_t* contributions,
    iree_host_size_t contribution_count, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

// Emits complete AMDGPU HSACO for one ABI-lowered target-low HAL kernel.
//
// The output stream receives a self-contained ELF code object with metadata,
// one kernel descriptor, and one encoded text entry. |options| may provide
// a packet plan and an optional emission summary. Values must be
// physically allocated and unspilled.
iree_status_t loom_amdgpu_emit_kernel_hsaco(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_HSACO_H_
