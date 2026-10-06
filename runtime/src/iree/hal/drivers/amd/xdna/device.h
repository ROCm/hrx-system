// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_DEVICE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_DEVICE_H_

#include "iree/hal/drivers/amd/xdna/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a HAL device around a prepared native context. Success transfers
// context ownership to the device. Failure leaves context with the caller.
iree_status_t iree_hal_amd_xdna_device_create(
    iree_hal_amd_xdna_context_t* context, iree_string_view_t display_name,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_DEVICE_H_
