// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_NOTIFICATION_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_NOTIFICATION_H_

#include "iree/base/api.h"
#include "iree/hal/host_notification.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;

// Creates an AMDGPU-backed host notification.
//
// The notification retains the driver HSA library independently, allowing it
// to remain valid until its final release without exposing HSA to callers.
iree_status_t iree_hal_amdgpu_host_notification_create(
    iree_hal_amdgpu_logical_device_t* device,
    iree_hal_host_notification_t** out_notification);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_NOTIFICATION_H_
