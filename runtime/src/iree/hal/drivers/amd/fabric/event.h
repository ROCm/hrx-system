// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_EVENT_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_EVENT_H_

#include "amdf/native_event.h"
#include "iree/async/event.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Borrows an initialized async event's native signal destination. The event
// owner joins all native notification requests before releasing its handles.
// Platforms without a libamdf-compatible event return an empty descriptor;
// callers check it against the selected queue's advertised notification types.
amdf_native_event_t iree_hal_amd_fabric_event_native_handle(
    const iree_async_event_native_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_EVENT_H_
