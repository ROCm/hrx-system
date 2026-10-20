// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_PROGRAM_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_PROGRAM_H_

#include "iree/hal/drivers/amd/gpu/code.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/blit.h"
#include "iree/hal/drivers/amd/gpu/queue/program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Concrete PM4 executor. Programs recorded against storage borrow these fixed
// launches; the encoder recovers this owner from its embedded storage.
typedef struct iree_hal_amd_gpu_pm4_program_pool_t {
  // Shared placement, causal instance ownership, and explicit trim registry.
  iree_hal_amd_gpu_program_pool_t storage;
  // Owned instruction bytes shared by every recording and invocation.
  iree_hal_amd_fabric_memory_t code;
  // Optional builtin initialization backing, retired before device publication.
  iree_hal_amd_gpu_code_upload_t* code_upload;
  // Prepared parallel relocation kernel.
  iree_hal_amd_gpu_pm4_launch_t fixup;
  // Prepared dynamic-extent copy/fill kernel.
  iree_hal_amd_gpu_pm4_launch_t blit;
  // KERNEL submission enters IB1 and needs an explicit exit continuation.
  bool chain;
} iree_hal_amd_gpu_pm4_program_pool_t;

// Cold construction; failure leaves a deinitializable partial owner.
iree_status_t iree_hal_amd_gpu_pm4_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, bool chain, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_program_pool_t* out_pool);
void iree_hal_amd_gpu_pm4_program_pool_deinitialize(
    iree_hal_amd_gpu_pm4_program_pool_t* pool);

// A sealed block's native entry. Block identities and extents are independent
// of invocation placement; successor packets and private-data pointers are
// ordinary fixups relative to the selected instance's base. An exit can return
// to a USER ring or chain to an invocation-private KERNEL continuation.
typedef struct iree_hal_amd_gpu_pm4_program_block_t {
  // DWORD-aligned entry relative to instance byte zero.
  uint32_t offset;
  // Native extent including padding before the final exit packet.
  uint32_t word_count;
} iree_hal_amd_gpu_pm4_program_block_t;

// Fixed invocation header followed by the dense uint64_t input vector. The
// queue owns this upload until native retirement; program emission initializes
// the builtin arguments after the recorder resolves the input vector.
typedef struct iree_hal_amd_gpu_pm4_program_arguments_t {
  // Independent full-word replacements for this instance.
  iree_hal_amd_gpu_fixup_arguments_t fixup;
  // Cold initialization or overlap clone, used only for a large native copy.
  iree_hal_amd_gpu_pm4_blit_arguments_t copy;
} iree_hal_amd_gpu_pm4_program_arguments_t;

// Native invocation extent, excluding caller-owned waits, optional padding,
// and retirement. The reservation remains exclusive through emission, so the
// same initialized/join facts govern both sizing and encoding.
uint32_t iree_hal_amd_gpu_pm4_program_word_count(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile);

// Publishes host input, initializes a cold placement, applies all fixups, and
// enters the resident block. Actual producer/consumer cache paths determine
// visibility. Padding precedes the final entry, including terminal CHAIN.
// No command, binding, or fixup entry is walked on the host here.
uint32_t* iree_hal_amd_gpu_pm4_program_emit(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance,
    iree_hal_amd_gpu_pm4_program_block_t entry,
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    uint64_t argument_address,
    iree_hal_amd_gpu_pm4_program_arguments_t* arguments, uint32_t padding,
    uint32_t* words);

// Emits a launch for a nonempty, admitted fixup table. Workgroups contain 64
// lanes, with one complete DWORD per lane. The caller owns visibility before
// the dispatch and the shader-write -> command-fetch boundary afterward.
uint32_t* iree_hal_amd_gpu_pm4_fixup_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    uint64_t arguments_address, uint32_t entry_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_PROGRAM_H_
