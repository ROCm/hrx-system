// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_

#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/device/kernels.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Blit Kernels
//===----------------------------------------------------------------------===//

// Builtin transfer kernel table used when populating blit dispatch packets.
// Queue reservation, packet header commit, completion-signal assignment, and
// doorbell writes are handled by the caller's queue implementation.
typedef struct iree_hal_amdgpu_device_buffer_transfer_context_t {
  // Handles to opaque kernel objects used to dispatch builtin kernels.
  const iree_hal_amdgpu_device_kernels_t* kernels;

  // Device wavefront width used when choosing the builtin blit workgroup size.
  // This is kept explicit so future wave32/wave64-specialized kernels can
  // select variants without guessing from the loaded code object.
  uint16_t wavefront_size;
  // X-dimension workgroup size used for all builtin blit dispatches. Y/Z are
  // always 1; the kernels are 1D along the global linear index.
  uint16_t workgroup_size_x;

  // Maximum number of blit workgroups to launch for one transfer. Kernels use
  // grid-stride loops, so large transfers bound resident work and let each
  // lane process multiple elements instead of launching one lane per element.
  uint32_t max_workgroup_count;
} iree_hal_amdgpu_device_buffer_transfer_context_t;

// Initializes a builtin transfer context from device properties. The caller
// must ensure |compute_unit_count| is non-zero and |wavefront_size| is one of
// {32, 64}; see physical_device.c for the HSA-query-backed validation path.
void iree_hal_amdgpu_device_buffer_transfer_context_initialize(
    const iree_hal_amdgpu_device_kernels_t* kernels,
    uint32_t compute_unit_count, uint32_t wavefront_size,
    iree_hal_amdgpu_device_buffer_transfer_context_t* out_context);

// Kernel arguments for the `iree_hal_amdgpu_device_buffer_fill_*` family.
typedef struct iree_hal_amdgpu_device_buffer_fill_kernargs_t {
  // Device-visible target address.
  void* target_ptr;
  // Number of elements processed by the selected kernel.
  uint64_t element_length;
  // Pattern represented as expected by the selected kernel.
  uint64_t pattern;
  // Dispatch grid X dimension in work-items.
  uint32_t grid_size_x;
  // Dispatch grid Y dimension in work-items.
  uint32_t grid_size_y;
  // Dispatch workgroup X dimension in work-items.
  uint32_t workgroup_size_x;
} iree_hal_amdgpu_device_buffer_fill_kernargs_t;
// Byte length consumed by the code object. Excludes trailing C struct padding.
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE                \
  (IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t, \
                        workgroup_size_x) +                            \
   sizeof(uint32_t))
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_ALIGNMENT \
  IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t)
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t,
                         target_ptr) == 0 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t,
                             element_length) == 8 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t,
                             pattern) == 16 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t,
                             grid_size_x) == 24 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t,
                             grid_size_y) == 28 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_fill_kernargs_t,
                             workgroup_size_x) == 32 &&
        IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE == 36,
    "fill kernargs must match the shared device kernel ABI");

// Kernel arguments for the `iree_hal_amdgpu_device_buffer_copy_*` family.
typedef struct iree_hal_amdgpu_device_buffer_copy_kernargs_t {
  // Device-visible source address.
  const void* source_ptr;
  // Device-visible target address.
  void* target_ptr;
  // Number of elements processed by the selected kernel.
  uint64_t element_length;
  // Dispatch grid X dimension in work-items.
  uint32_t grid_size_x;
  // Dispatch grid Y dimension in work-items.
  uint32_t grid_size_y;
  // Dispatch workgroup X dimension in work-items.
  uint32_t workgroup_size_x;
} iree_hal_amdgpu_device_buffer_copy_kernargs_t;
// Byte length consumed by the code object. Excludes trailing C struct padding.
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE                \
  (IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t, \
                        workgroup_size_x) +                            \
   sizeof(uint32_t))
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_ALIGNMENT \
  IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t)
IREE_AMDGPU_STATIC_ASSERT(
    IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t,
                         source_ptr) == 0 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t,
                             target_ptr) == 8 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t,
                             element_length) == 16 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t,
                             grid_size_x) == 24 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t,
                             grid_size_y) == 28 &&
        IREE_AMDGPU_OFFSETOF(iree_hal_amdgpu_device_buffer_copy_kernargs_t,
                             workgroup_size_x) == 32 &&
        IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE == 36,
    "copy kernargs must match the shared device kernel ABI");
// Alignment used for host-staged update payloads consumed by copy kernels.
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT 16
// Byte offset to a host-staged update payload following copy kernargs.
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET       \
  ((IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE +                 \
    IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT - 1) & \
   ~(IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT - 1))

// Builtin kernel variant selected by a transfer plan.
typedef enum iree_hal_amdgpu_device_buffer_transfer_kernel_e {
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_FILL_X1 = 0,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_FILL_X2,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_FILL_X4,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_FILL_X8,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_FILL_BLOCK_X16,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_FILL_BLOCK_UNALIGNED_X16,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COPY_X1,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COPY_BLOCK_X4,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COPY_BLOCK_X8,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COPY_BLOCK_X16,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COPY_BLOCK_UNALIGNED_X16,
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TRANSFER_KERNEL_COUNT,
} iree_hal_amdgpu_device_buffer_transfer_kernel_t;

// Immutable launch plan for one builtin fill operation.
typedef struct iree_hal_amdgpu_device_buffer_fill_plan_t {
  // Builtin kernel selected for this operation.
  iree_hal_amdgpu_device_buffer_transfer_kernel_t kernel;
  // Dispatch grid dimensions in work-items.
  uint32_t grid_size[3];
  // Dispatch workgroup X dimension in work-items.
  uint32_t workgroup_size_x;
  // Number of elements passed to the selected kernel.
  uint64_t element_length;
  // Fill pattern represented as expected by the selected kernel.
  uint64_t pattern;
} iree_hal_amdgpu_device_buffer_fill_plan_t;

// Immutable launch plan for one builtin copy operation.
typedef struct iree_hal_amdgpu_device_buffer_copy_plan_t {
  // Builtin kernel selected for this operation.
  iree_hal_amdgpu_device_buffer_transfer_kernel_t kernel;
  // Dispatch grid dimensions in work-items.
  uint32_t grid_size[3];
  // Dispatch workgroup X dimension in work-items.
  uint32_t workgroup_size_x;
  // Number of elements passed to the selected kernel.
  uint64_t element_length;
} iree_hal_amdgpu_device_buffer_copy_plan_t;

// Returns borrowed HSA launch metadata for a validated transfer kernel.
const iree_hal_amdgpu_device_kernel_args_t*
iree_hal_amdgpu_device_buffer_transfer_kernel_args(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* context,
    iree_hal_amdgpu_device_buffer_transfer_kernel_t kernel);

// Plans a builtin fill using the minimum guaranteed |target_alignment|.
//
// The alignment must be a non-zero power of two. Planning depends only on
// immutable device properties and validated operation parameters, allowing the
// result to be reused when the eventual target address is not yet available.
// Returns false if the operation cannot be represented; |out_plan| is left
// unmodified on failure.
bool iree_hal_amdgpu_device_buffer_fill_plan(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* context,
    uint64_t target_alignment, uint64_t length, uint64_t pattern,
    uint8_t pattern_length,
    iree_hal_amdgpu_device_buffer_fill_plan_t* out_plan);

// Initializes fill kernargs for |plan| and |target_ptr|.
void iree_hal_amdgpu_device_buffer_fill_plan_initialize_kernargs(
    const iree_hal_amdgpu_device_buffer_fill_plan_t* plan, void* target_ptr,
    iree_hal_amdgpu_device_buffer_fill_kernargs_t* out_kernargs);

// Emplaces a planned builtin fill into already-reserved AQL storage.
void iree_hal_amdgpu_device_buffer_fill_plan_emplace(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    const iree_hal_amdgpu_device_buffer_fill_plan_t* IREE_AMDGPU_RESTRICT plan,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* target_ptr, void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Plans a builtin copy using the minimum guaranteed source and target
// alignments.
//
// Alignments must be non-zero powers of two. Returns false if the operation
// cannot be represented; |out_plan| is left unmodified on failure.
bool iree_hal_amdgpu_device_buffer_copy_plan(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* context,
    uint64_t source_alignment, uint64_t target_alignment, uint64_t length,
    iree_hal_amdgpu_device_buffer_copy_plan_t* out_plan);

// Initializes copy kernargs for |plan|, |source_ptr|, and |target_ptr|.
void iree_hal_amdgpu_device_buffer_copy_plan_initialize_kernargs(
    const iree_hal_amdgpu_device_buffer_copy_plan_t* plan,
    const void* source_ptr, void* target_ptr,
    iree_hal_amdgpu_device_buffer_copy_kernargs_t* out_kernargs);

// Returns the maximum alignment up to 16 bytes guaranteed by |pointer|.
uint64_t iree_hal_amdgpu_device_buffer_transfer_pointer_alignment(
    const void* pointer);

// Emplaces a planned builtin copy into already-reserved AQL storage.
void iree_hal_amdgpu_device_buffer_copy_plan_emplace(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    const iree_hal_amdgpu_device_buffer_copy_plan_t* IREE_AMDGPU_RESTRICT plan,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    const void* source_ptr, void* target_ptr,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates a builtin fill dispatch packet and its kernargs in already-reserved
// storage. The caller owns packet header commit, completion signal assignment,
// and queue doorbell signaling.
//
// Returns false if |pattern_length| is unsupported, the target pointer/length
// alignment is incompatible with that pattern width, or |length| cannot be
// represented by the dispatch packet grid dimensions. On failure,
// |dispatch_packet| and |kernarg_ptr| are left unmodified.
bool iree_hal_amdgpu_device_buffer_fill_emplace(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* target_ptr, uint64_t length, uint64_t pattern, uint8_t pattern_length,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates a builtin copy dispatch packet and its kernargs in already-reserved
// storage. The caller owns packet header commit, completion signal assignment,
// and queue doorbell signaling.
//
// Returns false if |length| cannot be represented by the dispatch packet grid
// dimensions. On failure, |dispatch_packet| and |kernarg_ptr| are left
// unmodified.
bool iree_hal_amdgpu_device_buffer_copy_emplace(
    const iree_hal_amdgpu_device_buffer_transfer_context_t* IREE_AMDGPU_RESTRICT
        context,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    const void* source_ptr, void* target_ptr, uint64_t length,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_BLIT_H_
