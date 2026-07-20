// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_HOST_NOTIFICATION_H_
#define IREE_HAL_HOST_NOTIFICATION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_host_notification_t
//===----------------------------------------------------------------------===//

// An opaque device signal that a host thread can wait on.
//
// Device-runtime ABIs may serialize |device_token| into device-visible memory
// without interpreting it. The driver owns the token's representation and all
// native synchronization needed to wait on it.
typedef struct iree_hal_host_notification_t iree_hal_host_notification_t;

// Initial value observed by a newly-created notification.
#define IREE_HAL_HOST_NOTIFICATION_INITIAL_VALUE UINT64_C(1)

// Creates a host notification for device-runtime communication.
IREE_API_EXPORT iree_status_t iree_hal_host_notification_create(
    iree_hal_device_t* device, iree_hal_host_notification_t** out_notification);

// Generic vtable implementation for devices that do not support host
// notifications. Device vtables must assign this function instead of NULL.
IREE_API_EXPORT iree_status_t iree_hal_host_notification_create_unimplemented(
    iree_hal_device_t* device, iree_hal_host_notification_t** out_notification);

// Retains |notification| for the caller.
IREE_API_EXPORT void iree_hal_host_notification_retain(
    iree_hal_host_notification_t* notification);

// Releases |notification| from the caller.
IREE_API_EXPORT void iree_hal_host_notification_release(
    iree_hal_host_notification_t* notification);

// Returns the opaque device ABI token for |notification|.
//
// The token is only valid while the notification remains retained. Callers may
// copy it into a backend-specific device-runtime ABI but must not inspect or
// modify it.
IREE_API_EXPORT uint64_t iree_hal_host_notification_device_token(
    iree_hal_host_notification_t* notification);

// Waits until the notification's value differs from |observed_value|.
//
// This is an unbounded wait. Callers that need to stop a listener must arrange
// an independent stop condition and call iree_hal_host_notification_wake.
IREE_API_EXPORT uint64_t iree_hal_host_notification_wait(
    iree_hal_host_notification_t* notification, uint64_t observed_value);

// Wakes host threads blocked in iree_hal_host_notification_wait.
//
// This is intended for listener shutdown. It must not be used to provide
// device-runtime responses, which are carried by the caller's shared memory
// protocol.
IREE_API_EXPORT void iree_hal_host_notification_wake(
    iree_hal_host_notification_t* notification);

//===----------------------------------------------------------------------===//
// iree_hal_host_notification_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_host_notification_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_host_notification_t* notification);
  uint64_t(IREE_API_PTR* device_token)(
      iree_hal_host_notification_t* notification);
  uint64_t(IREE_API_PTR* wait)(iree_hal_host_notification_t* notification,
                               uint64_t observed_value);
  void(IREE_API_PTR* wake)(iree_hal_host_notification_t* notification);
} iree_hal_host_notification_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_host_notification_vtable_t);

IREE_API_EXPORT void iree_hal_host_notification_destroy(
    iree_hal_host_notification_t* notification);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_HOST_NOTIFICATION_H_
