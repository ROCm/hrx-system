// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SPEC_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SPEC_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/device_spec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_AMDGPU_DEVICE_SPEC_SCHEMA_ID "iree.hal.drivers.amdgpu.device"
#define IREE_HAL_AMDGPU_DEVICE_SPEC_SCHEMA_VERSION 1u

// Stable AMDGPU device spec flags.
typedef uint32_t iree_hal_amdgpu_device_spec_flags_t;
typedef enum iree_hal_amdgpu_device_spec_flag_bits_e {
  // No AMDGPU device spec flags are present.
  IREE_HAL_AMDGPU_DEVICE_SPEC_FLAG_NONE = 0u,
  // The GPU link to the nearest host fine-grained memory pool supports native
  // atomic transactions.
  IREE_HAL_AMDGPU_DEVICE_SPEC_FLAG_HOST_NATIVE_ATOMICS = 1u << 0,
} iree_hal_amdgpu_device_spec_flag_bits_t;

// Pointer-free AMDGPU device facts preserved in a driver-local facet.
typedef struct iree_hal_amdgpu_device_spec_t {
  // Stable AMDGPU device spec flags.
  iree_hal_amdgpu_device_spec_flags_t flags;
} iree_hal_amdgpu_device_spec_t;

// Finds the AMDGPU device spec facet in |device_spec| or NULL.
IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_amdgpu_device_spec_find_facet(
    const iree_hal_device_spec_t* device_spec);

// Decodes an AMDGPU device spec |facet| into |out_spec|.
IREE_API_EXPORT iree_status_t iree_hal_amdgpu_device_spec_decode_facet(
    const iree_hal_device_spec_facet_t* facet,
    iree_hal_amdgpu_device_spec_t* out_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SPEC_H_
