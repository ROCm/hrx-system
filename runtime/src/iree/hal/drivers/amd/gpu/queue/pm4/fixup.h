// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_FIXUP_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_FIXUP_H_

#include "iree/hal/drivers/amd/gpu/queue/pm4/launch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One independent DWORD replacement in a resident program instance. The
// recorder establishes disjoint four-byte targets, valid value indices,
// and a shift in [0, 63]. Each consumed mutable word has exactly one writer.
// The kernel stores uint32_t((values[index] + addend) >> shift); it never reads
// the destination. Byte-aligned stores permit packed kernargs as well as native
// packet words. Two entries represent a 64-bit pointer without depending on the
// pointer's old value or on inter-lane execution order.
typedef struct iree_hal_amd_gpu_pm4_fixup_t {
  // Byte offset from the instance base, including commands and private data.
  uint32_t target_offset;
  // Low 24 bits select an input value; bits 24..29 encode the right shift.
  uint32_t source;
  // Byte displacement or immediate integer added before shifting.
  uint64_t addend;
} iree_hal_amd_gpu_pm4_fixup_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_fixup_t) == 16,
              "four independent fixups occupy one cache line");

// Invocation-private kernargs. The queue uploads these and its dense value
// table once, then retains the upload through the native completion boundary.
typedef struct iree_hal_amd_gpu_pm4_fixup_arguments_t {
  // GPU address of immutable fixup entries shared by every instance.
  uint64_t entries;
  // GPU address of this invocation's uint64_t value table.
  uint64_t values;
  // GPU base of the selected command/data instance.
  uint64_t target;
  // Number of independent DWORD replacements.
  uint32_t entry_count;
  // Initialized ABI padding, ignored by the shader.
  uint32_t reserved;
} iree_hal_amd_gpu_pm4_fixup_arguments_t;
static_assert(sizeof(iree_hal_amd_gpu_pm4_fixup_arguments_t) == 32,
              "fixup kernargs match the compiled ABI");

// Returns the relocation-free kernel and descriptor for a supported physical
// target. Cold device construction places this code once and prepares its
// launch state. No compiler or code-object parser is required by the runtime.
iree_status_t iree_hal_amd_gpu_pm4_fixup_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel);

// Emits a launch for a nonempty, admitted fixup table. Workgroups contain 64
// lanes, with one complete DWORD per lane. The caller owns visibility before
// the dispatch and the shader-write -> command-fetch boundary afterward.
uint32_t* iree_hal_amd_gpu_pm4_fixup_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    uint64_t arguments_address, uint32_t entry_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_FIXUP_H_
