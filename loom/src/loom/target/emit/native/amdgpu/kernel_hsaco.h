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
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/residency.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_native_preflight_t loom_amdgpu_native_preflight_t;

typedef struct loom_amdgpu_kernel_hsaco_target_resources_t {
  // Stable scalar register class represented by |scalar_register_count|.
  iree_string_view_t scalar_register_class;
  // Final scalar register units declared by target metadata.
  uint32_t scalar_register_count;
  // Stable vector register class represented by |vector_register_count|.
  iree_string_view_t vector_register_class;
  // Final vector register units declared by target metadata.
  uint32_t vector_register_count;
  // Target wavefront width in lanes.
  uint32_t wave_size;
  // Maximum resident waves per SIMD in the target occupancy model.
  uint32_t max_waves_per_simd;
  // Estimated resident waves per SIMD after final target resources.
  uint32_t resident_waves_per_simd;
  // Estimated final occupancy as a percentage of |max_waves_per_simd|.
  uint32_t occupancy_percent;
  // Stable resource name limiting final occupancy, or "max_waves".
  iree_string_view_t limiting_resource;
  // Exact target residency transition summary, or zero when unavailable.
  loom_target_residency_summary_t residency_summary;
} loom_amdgpu_kernel_hsaco_target_resources_t;

typedef struct loom_amdgpu_kernel_hsaco_summary_t {
  // Number of native instructions emitted into the kernel text stream.
  uint64_t instruction_count;
  // Number of native instructions in the final scheduled kernel body.
  uint64_t body_instruction_count;
  // Number of target-owned native instructions in the kernel entry envelope.
  uint64_t entry_instruction_count;
  // Number of final native instructions containing coissued components.
  uint64_t coissued_instruction_count;
  // Number of semantic components carried by coissued instructions.
  uint64_t coissued_component_count;
  // Number of bytes in the semantic kernel text stream.
  uint64_t text_byte_count;
  // Number of bytes in the stored kernel text stream including local padding.
  uint64_t text_storage_byte_count;
  // Number of private/scratch memory bytes declared by kernel metadata.
  uint64_t private_segment_fixed_size;
  // Number of local/shared memory bytes declared by kernel metadata.
  uint64_t group_segment_fixed_size;
  // Final target resource and occupancy facts from kernel metadata.
  loom_amdgpu_kernel_hsaco_target_resources_t target_resources;
} loom_amdgpu_kernel_hsaco_summary_t;

typedef struct loom_amdgpu_kernel_hsaco_options_t {
  // Optional ABI layout captured before target resource materialization.
  const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout;
  // Verified ABI facts captured before allocation.
  const loom_amdgpu_hal_kernel_abi_verify_result_t* abi_verify;
  // Optional preflight result captured before HSACO contribution construction.
  const loom_amdgpu_native_preflight_t* preflight;
  // Optional target-owned packet plan applied during native encoding.
  const struct loom_amdgpu_packet_plan_t* packet_plan;
  // Optional code-object data symbols emitted by loom_amdgpu_emit_kernel_hsaco.
  const loom_amdgpu_hsaco_data_symbol_t* data_symbols;
  // Number of entries in |data_symbols|.
  iree_host_size_t data_symbol_count;
  // Optional target-owned emission summary populated after successful emission.
  loom_amdgpu_kernel_hsaco_summary_t* summary;
  // Optional encoding products retained in the kernel contribution.
  loom_amdgpu_encode_instruction_stream_flags_t encoding_flags;
} loom_amdgpu_kernel_hsaco_options_t;

typedef struct loom_amdgpu_kernel_hsaco_write_options_t {
  // Optional code-object data symbols emitted alongside the kernels.
  const loom_amdgpu_hsaco_data_symbol_t* data_symbols;
  // Number of entries in |data_symbols|.
  iree_host_size_t data_symbol_count;
} loom_amdgpu_kernel_hsaco_write_options_t;

typedef struct loom_amdgpu_kernel_hsaco_contribution_t {
  // Canonical artifact target key retaining every exact target feature.
  iree_string_view_t artifact_target_key;
  // Full AMDHSA code-object target ID such as
  // `amdgcn-amd-amdhsa--gfx11-generic`.
  iree_string_view_t code_object_target_id;
  // Exact or generic processor used for ELF flags and descriptor packing.
  iree_string_view_t processor;
  // Kernel entry metadata, descriptor flags, and encoded native text.
  loom_amdgpu_hsaco_kernel_t kernel;
  // Exact branch-island layout applied to |kernel.text|.
  loom_amdgpu_branch_layout_t branch_layout;
  // Target-owned instructions inserted during native encoding when capture was
  // requested.
  const loom_amdgpu_native_insertion_t* native_insertions;
  // Number of entries in |native_insertions|.
  iree_host_size_t native_insertion_count;
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
// Contributions must all carry the same artifact identity, AMDHSA code-object
// target ID, and processor. The writer uses |scratch_arena| only for final
// layout tables and can run after kernel contributions were produced
// independently, provided their backing storage remains live for the duration
// of this call.
iree_status_t loom_amdgpu_write_kernel_hsaco_contributions(
    const loom_amdgpu_kernel_hsaco_contribution_t* contributions,
    iree_host_size_t contribution_count,
    const loom_amdgpu_kernel_hsaco_write_options_t* options,
    iree_io_stream_t* stream, iree_arena_allocator_t* scratch_arena);

// Emits complete AMDGPU HSACO for one ABI-lowered target-low HAL kernel.
//
// The output stream receives a self-contained ELF code object with metadata,
// one kernel descriptor, and one encoded text entry. |options| carries
// verified ABI facts and may provide a packet plan and emission summary.
// Values must be physically allocated and unspilled.
iree_status_t loom_amdgpu_emit_kernel_hsaco(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_HSACO_H_
