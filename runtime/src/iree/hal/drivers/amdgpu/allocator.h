// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;
typedef struct iree_hal_amdgpu_topology_t iree_hal_amdgpu_topology_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_allocator_t
//===----------------------------------------------------------------------===//

// Creates a buffer allocator that allocates from HSA memory pools.
//
// Normal allocations are direct HSA memory-pool allocations. When the logical
// device has HAL ASAN enabled, device-local persistent allocations route
// through the device default pool set so redzones, shadow publication, release
// poisoning, and quarantine follow the same policy as queue allocations.
//
// |logical_device| is unretained and must outlive the allocator.
iree_status_t iree_hal_amdgpu_allocator_create(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator);

#endif  // IREE_HAL_DRIVERS_AMDGPU_ALLOCATOR_H_
