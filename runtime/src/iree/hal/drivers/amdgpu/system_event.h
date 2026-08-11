// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SYSTEM_EVENT_H_
#define IREE_HAL_DRIVERS_AMDGPU_SYSTEM_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;

// Registers |logical_device| to receive process-wide HSA system events.
// Registration is a cold device-creation operation. The event handler retains
// no device references; callers must begin and finish the two-phase teardown
// around destruction of the device's HSA queues.
iree_status_t iree_hal_amdgpu_system_event_register_device(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_allocator_t host_allocator);

// Stops delivering events into |logical_device| while continuing to claim
// faults from its agents during hardware queue teardown.
void iree_hal_amdgpu_system_event_begin_device_teardown(
    iree_hal_amdgpu_logical_device_t* logical_device);

// Removes |logical_device| from process-wide HSA system event delivery.
// Must be called after all of the device's HSA queues have been destroyed.
void iree_hal_amdgpu_system_event_unregister_device(
    iree_hal_amdgpu_logical_device_t* logical_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SYSTEM_EVENT_H_
