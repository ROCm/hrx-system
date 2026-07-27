// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_ASAN_H_
#define IREE_HAL_MEMORY_ASAN_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// ASAN pool policy
//===----------------------------------------------------------------------===//

// ASAN protection mode requested by a pool.
typedef enum iree_hal_asan_pool_mode_e {
  // No ASAN redzones or shadow updates are requested.
  IREE_HAL_ASAN_POOL_MODE_DISABLED = 0,
  // Allocate hidden redzones and advise their state through the provider.
  IREE_HAL_ASAN_POOL_MODE_SHADOW = 1,
} iree_hal_asan_pool_mode_t;

// Options controlling how HAL pools shape ASAN-protected allocations.
//
// These options are target-neutral. Pools use them to compute backing ranges
// and redzones; slab providers use them to decide whether they can publish the
// target-specific shadow state required by those ranges.
typedef struct iree_hal_asan_pool_options_t {
  // ASAN protection mode requested from the pool and slab provider.
  iree_hal_asan_pool_mode_t mode;

  // Number of application bytes described by each shadow byte.
  iree_device_size_t shadow_granule_size;

  // Minimum poisoned bytes to reserve before and after each user allocation.
  iree_device_size_t redzone_size;

  // Additional provider-required alignment for hidden backing ranges.
  iree_device_size_t backing_alignment;

  // Target backing bytes a pool may keep poisoned before returning them.
  iree_device_size_t quarantine_size;
} iree_hal_asan_pool_options_t;

// Returns true when |options| requests ASAN allocation protection.
IREE_API_EXPORT bool iree_hal_asan_pool_options_is_enabled(
    const iree_hal_asan_pool_options_t* options);

// Validates |options| independent of any target-specific provider capability.
IREE_API_EXPORT iree_status_t iree_hal_asan_pool_options_validate(
    const iree_hal_asan_pool_options_t* options);

//===----------------------------------------------------------------------===//
// ASAN allocation layout
//===----------------------------------------------------------------------===//

// Hidden backing layout for one ASAN-protected user allocation.
//
// The backing range is what a pool reserves from its offset allocator and slab.
// The user range is what the public HAL reservation/buffer exposes. Redzones
// occupy the remaining bytes and are expected to be poisoned by the provider.
typedef struct iree_hal_asan_allocation_layout_t {
  // Minimum alignment required for the backing range offset.
  iree_device_size_t backing_offset_alignment;

  // Minimum alignment required for the backing range byte length.
  iree_device_size_t backing_length_alignment;

  // Total byte length consumed from the pool/slab backing range.
  iree_device_size_t backing_length;

  // Byte offset from the backing range base to the user-visible range.
  iree_device_size_t user_offset;

  // User-visible allocation byte length.
  iree_device_size_t user_length;

  // Poisoned byte length before the user-visible range.
  iree_device_size_t left_redzone_length;

  // Poisoned byte length after the user-visible range.
  iree_device_size_t right_redzone_length;
} iree_hal_asan_allocation_layout_t;

// Flags describing a lifecycle transition for an ASAN backing range.
typedef uint32_t iree_hal_asan_range_advice_flags_t;
enum iree_hal_asan_range_advice_flag_bits_e {
  IREE_HAL_ASAN_RANGE_ADVICE_FLAG_NONE = 0u,
  // The range was allocated and its user bytes should be addressable.
  IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED = 1u << 0,
  // The range was released and all backing bytes should be poisoned.
  IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED = 1u << 1,
};

// Computes the hidden backing layout for an ASAN-protected allocation.
//
// |user_length| and |user_alignment| are the public allocation request. The
// returned layout guarantees that the backing offset can satisfy
// |user_alignment| while the backing length remains aligned only to target
// shadow/provider granularity. |options| must request an enabled ASAN mode.
IREE_API_EXPORT iree_status_t iree_hal_asan_calculate_allocation_layout(
    const iree_hal_asan_pool_options_t* options, iree_device_size_t user_length,
    iree_device_size_t user_alignment,
    iree_hal_asan_allocation_layout_t* out_layout);

// Extends |layout| to consume at least |backing_length| backing bytes.
//
// Suballocating pools may reserve a larger physical range than the minimum
// ASAN backing range because their allocator rounds to a block or slab
// granularity. The user range remains unchanged and the extra backing bytes
// become part of the right redzone.
IREE_API_EXPORT iree_status_t iree_hal_asan_extend_allocation_layout(
    iree_device_size_t backing_length,
    iree_hal_asan_allocation_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_ASAN_H_
