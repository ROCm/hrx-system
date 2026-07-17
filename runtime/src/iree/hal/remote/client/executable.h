// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client-side proxy executable for remote HAL devices.
//
// Wraps a server-assigned resource_id and cached metadata (export count).
// The proxy does not hold executable binary data — the binary lives on the
// server. Dispatch operations reference the proxy's resource_id in the
// COMMAND frame.

#ifndef IREE_HAL_REMOTE_CLIENT_EXECUTABLE_H_
#define IREE_HAL_REMOTE_CLIENT_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;

// Loads an executable artifact on the server and returns a client proxy.
//
// |target| must be borrowed from the remote client's bootstrapped device spec.
// Its ordinal identifies the matching immutable target row in the server's
// local device spec.
iree_status_t iree_hal_remote_client_executable_load(
    iree_hal_remote_client_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Creates a remote executable proxy with the resolved resource ID and function
// count returned by EXECUTABLE_UPLOAD. Ownership of the server-side resource is
// transferred to this call and released if proxy creation fails.
iree_status_t iree_hal_remote_client_executable_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id, iree_host_size_t function_count,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the resource_id from a remote client executable proxy.
iree_hal_remote_resource_id_t iree_hal_remote_client_executable_resource_id(
    iree_hal_executable_t* executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_EXECUTABLE_H_
