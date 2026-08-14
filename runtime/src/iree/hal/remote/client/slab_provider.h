// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_SLAB_PROVIDER_H_
#define IREE_HAL_REMOTE_CLIENT_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;
typedef struct iree_hal_device_spec_t iree_hal_device_spec_t;

// Creates a slab provider that backs HAL pools with remote device buffers.
iree_status_t iree_hal_remote_client_slab_provider_create(
    iree_hal_remote_client_device_t* device, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider);

// Configures immutable slab properties from the connected remote device spec.
// Must be called exactly once before the provider is exposed to queue pools.
void iree_hal_remote_client_slab_provider_configure(
    iree_hal_slab_provider_t* provider,
    const iree_hal_device_spec_t* device_spec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_SLAB_PROVIDER_H_
