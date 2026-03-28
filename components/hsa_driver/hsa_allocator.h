// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_HSA_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "hsa_driver/status_util.h"

typedef struct iree_hal_hsa_device_topology_t iree_hal_hsa_device_topology_t;

// Creates an HSA memory allocator.
iree_status_t iree_hal_hsa_allocator_create(
    iree_hal_device_t* parent_device,
    iree_hal_hsa_device_topology_t topology,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

bool iree_hal_hsa_allocator_isa(iree_hal_allocator_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HSA_ALLOCATOR_H_

