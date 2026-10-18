// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_
#define IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Indicates that a queue has no entry in a provisioned queue-scope table.
#define IREE_HAL_AMDGPU_PHYSICAL_QUEUE_ORDINAL_NONE \
  ((iree_hal_queue_ordinal_t)UINT32_MAX)

// Immutable identity for one logical AMDGPU host queue.
//
// Executable-load paths use this cold-path metadata to publish queue-specific
// globals without adding per-dispatch host bookkeeping.
typedef struct iree_hal_amdgpu_queue_scope_t {
  // Physical GPU device ordinal owning this queue.
  iree_host_size_t physical_device_ordinal;

  // Queue ordinal relative to |physical_device_ordinal|.
  iree_hal_queue_ordinal_t physical_queue_ordinal;

  // Queue-owned TSAN shadow facts used for executable-global publication.
  struct {
    // Device-visible base of queue-local shadow storage, or zero.
    uint64_t shadow_base;
    // Byte length of |shadow_base|.
    uint64_t shadow_size;
  } tsan;
} iree_hal_amdgpu_queue_scope_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_QUEUE_SCOPE_H_
