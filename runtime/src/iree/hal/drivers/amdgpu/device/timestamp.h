// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_TIMESTAMP_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_TIMESTAMP_H_

#include "iree/hal/drivers/amdgpu/abi/timestamp.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Kernel arguments for initializing raw dispatch profiling completion signals.
typedef struct iree_hal_amdgpu_dispatch_timestamp_signal_initialize_args_t {
  // Raw completion signal records paired with dispatch profiling slots.
  iree_amd_signal_t* signals;
  // Number of signal records in |signals|.
  uint32_t signal_count;
  // Reserved for future ABI expansion; must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_dispatch_timestamp_signal_initialize_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_dispatch_timestamp_signal_initialize_args_t) == 16,
    "dispatch timestamp signal initialization args size is part of the ABI");

// Populates a builtin dispatch packet and kernargs that initialize raw
// completion-signal records into the armed user-signal ABI shape expected by
// the CP.
//
// |dispatch_packet| and |kernarg_ptr| must point to reserved queue storage.
// The caller owns completion-signal assignment and header commit.
void iree_hal_amdgpu_device_timestamp_emplace_signal_initialization(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        signal_initialization_kernel_args,
    iree_amd_signal_t* signals, uint32_t signal_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Returns the byte offset of the harvest source table after the kernel args.
static inline size_t
iree_hal_amdgpu_device_timestamp_dispatch_harvest_source_offset(void) {
  return iree_amdgpu_align(
      sizeof(iree_hal_amdgpu_dispatch_timestamp_harvest_args_t),
      IREE_AMDGPU_ALIGNOF(iree_hal_amdgpu_dispatch_timestamp_harvest_source_t));
}

// Returns the kernarg byte length required for |source_count| harvest sources.
static inline size_t
iree_hal_amdgpu_device_timestamp_dispatch_harvest_kernarg_length(
    uint32_t source_count) {
  return iree_hal_amdgpu_device_timestamp_dispatch_harvest_source_offset() +
         (size_t)source_count *
             sizeof(iree_hal_amdgpu_dispatch_timestamp_harvest_source_t);
}

// Returns the harvest source table embedded in |kernarg_ptr|.
static inline iree_hal_amdgpu_dispatch_timestamp_harvest_source_t*
iree_hal_amdgpu_device_timestamp_dispatch_harvest_sources(
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  uint8_t* source_ptr =
      (uint8_t*)kernarg_ptr +
      iree_hal_amdgpu_device_timestamp_dispatch_harvest_source_offset();
  return (iree_hal_amdgpu_dispatch_timestamp_harvest_source_t*)source_ptr;
}

// Populates a builtin dispatch packet and kernargs that harvest timestamps from
// dispatch completion signals into fixed binary timestamp records.
//
// |dispatch_packet| and |kernarg_ptr| must point to reserved queue storage.
// The caller owns completion-signal assignment and header commit.
iree_hal_amdgpu_dispatch_timestamp_harvest_source_t*
iree_hal_amdgpu_device_timestamp_emplace_dispatch_harvest(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        harvest_kernel_args,
    uint32_t source_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Device builtin that initializes raw completion signals used for CP timestamp
// capture. The records are device-only scratch and are never host HSA objects.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_timestamp_initialize_completion_signals(
    iree_amd_signal_t* signals, uint32_t signal_count);

// Device builtin that copies per-dispatch CP timestamps into timestamp records.
// Each consumed completion signal is re-armed for later ring-slot reuse.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_timestamp_harvest_dispatch_records(
    const iree_hal_amdgpu_dispatch_timestamp_harvest_source_t*
        IREE_AMDGPU_RESTRICT sources,
    uint32_t source_count);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_TIMESTAMP_H_
