// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_NET_CHANNEL_H_
#define IREE_HAL_NET_CHANNEL_H_

#include "iree/hal/net/channel_provider.h"

// Copies validated cold configuration into one owned allocation.
iree_status_t iree_hal_net_channel_options_clone(
    const iree_hal_net_channel_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_net_channel_options_t** out_options);

// Poll-owner failure and progress entry points shared with peer callbacks.
void iree_hal_net_channel_fail(iree_hal_channel_t* channel,
                               iree_status_t status);
void iree_hal_net_channel_service(iree_hal_channel_t* channel);

// Creates a rank channel on the calling poll owner.
iree_status_t iree_hal_net_channel_create(
    iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_net_transport_factory_t* factory,
    iree_async_proactor_t* proactor,
    const iree_hal_net_channel_options_t* options,
    iree_allocator_t host_allocator, iree_hal_channel_t** out_channel);

// Returns whether the resource belongs to this provider implementation.
bool iree_hal_net_channel_isa(const iree_hal_channel_t* channel);

#endif  // IREE_HAL_NET_CHANNEL_H_
