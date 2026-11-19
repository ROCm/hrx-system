// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_BLIT_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_BLIT_H_

#include "iree/hal/driver/amd/gpu/image/image.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Resident-transfer arguments. Rows may be unaligned; their padding is never
// accessed. Native address admission bounds displacements below the signed
// offset limit. A linear transfer has one row and slice with zero strides.
// Every invocation overwrites all consumed fields before dispatch.
typedef struct iree_hal_amd_gpu_blit_arguments_t {
  // Native source byte zero, unused for fills.
  uint64_t source;
  // Native destination byte zero.
  uint64_t target;
  // Useful bytes per row; zero is a no-op.
  uint64_t length;
  // Four repeating bytes, beginning at each target row's byte zero.
  uint32_t pattern;
  // Zero copies; one fills with pattern.
  uint32_t fill;
  // Byte displacement between source rows; unused for fills or one row.
  uint64_t source_row_stride;
  // Byte displacement between source slices; unused for fills or one slice.
  uint64_t source_slice_stride;
  // Byte displacement between target rows; unused for one row.
  uint64_t target_row_stride;
  // Byte displacement between target slices; unused for one slice.
  uint64_t target_slice_stride;
  // Rows in this native chunk; zero is a no-op.
  uint32_t row_count;
  // Slices in this native chunk; zero is a no-op.
  uint32_t slice_count;
  // Launched workgroups along Y, used as the row iteration step.
  uint32_t groups_y;
  // Launched workgroups along Z, used as the slice iteration step.
  uint32_t groups_z;
} iree_hal_amd_gpu_blit_arguments_t;
static_assert(sizeof(iree_hal_amd_gpu_blit_arguments_t) == 80,
              "blit arguments match the compiled ABI");

// Each 64-lane group covers 1024 bytes per iteration. Small ranges need only
// their active groups; larger ranges use the kernel's fixed 256-group stride.
static inline uint32_t iree_hal_amd_gpu_blit_workgroup_count(uint64_t length) {
  return length > 256 * 1024 ? 256 : (uint32_t)((length + 1023) / 1024);
}

// Prepares a rectangular prefix of the remaining useful bytes, starting at
// |offset| in dense slice/row/byte order. Public admission has established a
// nonempty region, nonoverflowing useful byte count, and valid address spans.
// |capacity| bounds useful bytes, independently of either endpoint's padding.
// Returns consumed useful bytes, or zero when capacity is zero. Source/target
// contain relative byte displacements for the caller to add to native origins.
// The grid has at most 256 groups in total. Native signed index limits split
// large dimensions without narrowing the public geometry or expanding rows.
uint64_t iree_hal_amd_gpu_blit_prepare_region(
    const iree_hal_transfer_region_t* region, uint64_t offset,
    uint64_t capacity, iree_hal_amd_gpu_blit_arguments_t* out_arguments);

// Selects relocation-free native instructions and their descriptor. The cold
// owner places this once; no compiler or HSACO parser is in the runtime path.
iree_status_t iree_hal_amd_gpu_blit_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_BLIT_H_
