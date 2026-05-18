// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_EVENT_H_
#define IREE_HAL_REMOTE_CLIENT_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;

// Creates an event proxy backed by a server-side HAL event resource.
iree_status_t iree_hal_remote_client_event_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_event_t** out_event);

// Returns true if |event| is a remote client event proxy.
bool iree_hal_remote_client_event_isa(const iree_hal_event_t* event);

// Returns the server-side event resource ID for |event|.
iree_hal_remote_resource_id_t iree_hal_remote_client_event_resource_id(
    const iree_hal_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_EVENT_H_
